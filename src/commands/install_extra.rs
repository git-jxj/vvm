use colored::Colorize;
use indicatif::{ProgressBar, ProgressStyle};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Duration;

use crate::config::Config;
use crate::cuda;
use crate::errors::VvmError;
use crate::metadata::VersionMetadata;

fn make_spinner() -> ProgressBar {
    if std::env::var("VVM_VERBOSE").is_ok() {
        return ProgressBar::hidden();
    }
    let sp = ProgressBar::new_spinner();
    sp.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.green} {msg}")
            .unwrap(),
    );
    sp.enable_steady_tick(Duration::from_millis(100));
    sp
}

/// Install an auxiliary library into the current venv using a bundled recipe.
///
/// These are libraries vllm users commonly need but PyPI can't build cleanly
/// (missing git submodules, vendored CUDA sources, etc). Instead of shelling
/// out to vllm's `tools/install_*.sh` we embed the recipe so vvm works even
/// without a local vllm checkout.
pub fn run(
    config: &Config,
    name: &str,
    git_ref: Option<&str>,
    repo_override: Option<&str>,
) -> Result<(), VvmError> {
    let link = config.current_link();
    if !link.is_symlink() {
        return Err(VvmError::NoCurrentVersion);
    }
    let venv_path = std::fs::read_link(&link)?.join("venv");
    if !venv_path.exists() {
        return Err(VvmError::NoCurrentVersion);
    }

    match name.replace('-', "_").to_lowercase().as_str() {
        "deepgemm" | "deep_gemm" => install_deepgemm(config, &venv_path, git_ref, repo_override),
        "flashinfer" => install_flashinfer(config, &venv_path, git_ref, repo_override),
        "deepep" | "deep_ep" => install_deepep(config, &venv_path, git_ref, repo_override),
        other => Err(VvmError::InvalidSpecifier(format!(
            "unknown extra '{}'. Available: deepgemm, flashinfer, deepep",
            other
        ))),
    }
}

/// Shared clone-and-checkout step for recipes that pull from a git repo with
/// submodules. Returns the source directory inside `tmp`.
fn clone_at_ref(
    tmp: &TempDir,
    repo: &str,
    git_ref: &str,
    subdir: &str,
    spinner: &ProgressBar,
) -> Result<PathBuf, VvmError> {
    let src = tmp.path().join(subdir);
    spinner.set_message(format!("Cloning {} ...", subdir));
    run_checked(
        Command::new("git")
            .args(["clone", "--recursive", "--shallow-submodules"])
            .arg(repo)
            .arg(&src),
        "git clone",
    )?;
    spinner.set_message(format!("Checking out {} ...", git_ref));
    run_checked(
        Command::new("git")
            .arg("-C")
            .arg(&src)
            .args(["checkout", git_ref]),
        "git checkout",
    )?;
    // Submodule pointers at the target ref may differ from the default branch.
    spinner.set_message("Updating submodules ...");
    run_checked(
        Command::new("git").arg("-C").arg(&src).args([
            "submodule",
            "update",
            "--init",
            "--recursive",
        ]),
        "git submodule update",
    )?;
    Ok(src)
}

/// Vendored fallback when we can't read the active vllm's pin (no current
/// version, offline, or older vllm without `tools/install_deepgemm.sh`).
const FALLBACK_DEEPGEMM_REF: &str = "477618cd51baffca09c4b0b87e97c03fe827ef03";

/// Read `tools/install_deepgemm.sh` from the currently-active vllm and pull
/// out the `DEEPGEMM_GIT_REF=...` value. Source-of-truth order:
///   1. Local source tree (`source_path` from metadata) — covers `--path` /
///      `--source` / `--compile` installs and works offline.
///   2. raw.githubusercontent.com at the install's commit (or release tag) —
///      covers wheel-based installs where `tools/` isn't shipped.
///
/// Returns None on any failure; caller falls back to the vendored default.
fn resolve_deepgemm_ref_from_active_vllm(config: &Config) -> Option<String> {
    let link = config.current_link();
    let target = std::fs::read_link(&link).ok()?;
    let meta = VersionMetadata::load(&target).ok()?;

    if let Some(sp) = &meta.source_path {
        let local = Path::new(sp).join("tools/install_deepgemm.sh");
        if local.is_file() {
            if let Ok(text) = std::fs::read_to_string(&local) {
                if let Some(sha) = parse_deepgemm_git_ref(&text) {
                    return Some(sha);
                }
            }
        }
    }

    let github_ref = meta.commit.clone().or_else(|| {
        // Releases store the tag in `name` (e.g. "v0.18.0"); that's a valid
        // ref on github.com. Other types without a `commit` (nightly with no
        // resolved SHA) fall through to None.
        if meta.install_type == "release" {
            Some(meta.name.clone())
        } else {
            None
        }
    })?;

    let url = format!(
        "https://raw.githubusercontent.com/vllm-project/vllm/{}/tools/install_deepgemm.sh",
        github_ref
    );
    let client = reqwest::blocking::Client::builder()
        .user_agent("vvm/0.1.0")
        .timeout(Duration::from_secs(10))
        .build()
        .ok()?;
    let resp = client.get(&url).send().ok()?;
    if !resp.status().is_success() {
        return None;
    }
    let text = resp.text().ok()?;
    parse_deepgemm_git_ref(&text)
}

/// Parse `DEEPGEMM_GIT_REF=<sha>` (with optional single/double quotes) from a
/// shell script. Returns the first match.
fn parse_deepgemm_git_ref(text: &str) -> Option<String> {
    for raw in text.lines() {
        let line = raw.trim();
        let Some(rest) = line.strip_prefix("DEEPGEMM_GIT_REF=") else {
            continue;
        };
        let token = rest.split_whitespace().next()?;
        let sha = token.trim_matches(|c: char| c == '"' || c == '\'');
        if !sha.is_empty() {
            return Some(sha.to_string());
        }
    }
    None
}

fn install_deepgemm(
    config: &Config,
    venv_path: &Path,
    git_ref: Option<&str>,
    repo_override: Option<&str>,
) -> Result<(), VvmError> {
    // DeepGEMM requires CUDA 12.8+ (FP8 TMA descriptors on Hopper).
    require_cuda_at_least(12, 8)?;

    let repo = repo_override.unwrap_or("https://github.com/deepseek-ai/DeepGEMM.git");
    let (git_ref_owned, ref_source) = if let Some(r) = git_ref {
        (r.to_string(), "user")
    } else {
        match resolve_deepgemm_ref_from_active_vllm(config) {
            Some(r) => (r, "active vllm"),
            None => {
                eprintln!(
                    "  {} couldn't read DeepGEMM pin from active vllm — falling back to vvm's bundled default",
                    "warning:".yellow().bold()
                );
                (FALLBACK_DEEPGEMM_REF.to_string(), "bundled fallback")
            }
        }
    };
    let git_ref: &str = &git_ref_owned;

    println!(
        "{} DeepGEMM ({} @ {}) [pin: {}]",
        "Installing".green().bold(),
        repo.cyan(),
        git_ref.yellow(),
        ref_source.blue()
    );

    let spinner = make_spinner();

    // --recursive + --shallow-submodules pulls cutlass etc. without full history.
    // This is exactly what the PyPI sdist is missing, hence the direct-pip-install
    // failure the user hit.
    let tmp = TempDir::new("vvm-deepgemm")?;
    let src = clone_at_ref(&tmp, repo, git_ref, "deepgemm", &spinner).inspect_err(|_| {
        spinner.finish_and_clear();
    })?;

    // Upstream install.sh recommends wiping these before a rebuild.
    for dirty in ["build", "dist"] {
        let _ = std::fs::remove_dir_all(src.join(dirty));
    }
    for entry in std::fs::read_dir(&src)?.flatten() {
        if entry.file_name().to_string_lossy().ends_with(".egg-info") {
            let _ = std::fs::remove_dir_all(entry.path());
        }
    }

    spinner.set_message("Building wheel (this can take a while)...");
    let python = venv_path.join("bin/python");
    let build_result = run_checked(
        Command::new(&python)
            .current_dir(&src)
            .args(["setup.py", "bdist_wheel"])
            .env("VIRTUAL_ENV", venv_path),
        "python setup.py bdist_wheel",
    );
    if build_result.is_err() {
        spinner.finish_and_clear();
    }
    build_result?;

    let wheel = find_built_wheel(&src.join("dist"))?;
    spinner.println(format!("  {} {}", "Wheel:".blue().bold(), wheel.display()));

    spinner.set_message("Installing wheel into venv ...");
    let bin_dir = venv_path.join("bin");
    let current_path = std::env::var("PATH").unwrap_or_default();
    let new_path = format!("{}:{}", bin_dir.display(), current_path);
    let install_result = run_checked(
        Command::new("uv")
            .args(["pip", "install"])
            .arg(&wheel)
            .env("PATH", &new_path)
            .env("VIRTUAL_ENV", venv_path),
        "uv pip install",
    );
    spinner.finish_and_clear();
    install_result?;

    println!("{} DeepGEMM installed", "✓".green().bold());
    Ok(())
}

fn install_flashinfer(
    config: &Config,
    venv_path: &Path,
    git_ref: Option<&str>,
    repo_override: Option<&str>,
) -> Result<(), VvmError> {
    // FlashInfer supports CUDA 12.6+; vllm's wheels target 12.8+ so that's
    // our floor too — pre-Hopper CUDA just doesn't exercise the code paths
    // anyone installs flashinfer for.
    require_cuda_at_least(12, 6)?;

    // Nightly: skip source build entirely, pull prebuilt nightly wheels
    // (flashinfer-python + cubin + jit-cache) from flashinfer.ai/whl/nightly.
    if matches!(git_ref, Some("nightly")) {
        return install_flashinfer_nightly(venv_path);
    }

    let repo = repo_override.unwrap_or("https://github.com/flashinfer-ai/flashinfer.git");
    let git_ref = git_ref.unwrap_or("main");

    println!(
        "{} FlashInfer ({} @ {})",
        "Installing".green().bold(),
        repo.cyan(),
        git_ref.yellow()
    );

    // CUDA-version-gated arch list matches vllm's tools/flashinfer-build.sh.
    // These are the arches flashinfer can actually build kernels for at each
    // CUDA version; going wider causes nvcc errors, going narrower leaves
    // GPUs unserved. TORCH_CUDA_ARCH_LIST for the torch-side build,
    // FLASHINFER_CUDA_ARCH_LIST for flashinfer's JIT cache targets.
    // Computed before cloning because it's part of the wheel-cache key.
    let arch_list = std::env::var("TORCH_CUDA_ARCH_LIST")
        .ok()
        .or_else(flashinfer_default_arch_list);
    let variant = cuda::CudaVariant::detect();

    let spinner = make_spinner();

    // Fast path: the ref resolves to a commit we've already built a wheel for
    // (same CUDA variant + arch list) — install it without cloning anything.
    // The wheel is py3-none-any (kernels ship as cubin/JIT data, no torch-ABI
    // binding), so one build serves every version's venv.
    let mut sha = resolve_remote_sha(repo, git_ref);
    if let Some(wheel) =
        cached_flashinfer_wheel(config, sha.as_deref(), &variant, arch_list.as_deref())
    {
        return install_flashinfer_from_wheel(venv_path, &wheel, &spinner, "cached");
    }

    let tmp = TempDir::new("vvm-flashinfer")?;
    let src = clone_at_ref(&tmp, repo, git_ref, "flashinfer", &spinner).inspect_err(|_| {
        spinner.finish_and_clear();
    })?;

    // Short SHAs (and refs ls-remote couldn't see) resolve after checkout —
    // re-check the cache before paying for a build.
    if sha.is_none() {
        sha = rev_parse_head(&src);
        if let Some(wheel) =
            cached_flashinfer_wheel(config, sha.as_deref(), &variant, arch_list.as_deref())
        {
            return install_flashinfer_from_wheel(venv_path, &wheel, &spinner, "cached");
        }
    }

    if let Some(ref archs) = arch_list {
        spinner.println(format!("  {} {}", "Arches:".blue().bold(), archs));
    }

    let bin_dir = venv_path.join("bin");
    let current_path = std::env::var("PATH").unwrap_or_default();
    let new_path = format!("{}:{}", bin_dir.display(), current_path);
    let python = venv_path.join("bin/python");

    // `python -m build` needs the `build` package; tiny, no-op when present.
    run_checked(
        Command::new("uv")
            .args(["pip", "install", "build"])
            .env("PATH", &new_path)
            .env("VIRTUAL_ENV", venv_path),
        "uv pip install build",
    )
    .inspect_err(|_| spinner.finish_and_clear())?;

    spinner.set_message("Building wheel (this can take 10-20 min)...");

    // Build to a wheel file (rather than `uv pip install <src>`) so the result
    // can be cached and reused across versions. -n reuses the venv's torch
    // instead of an isolated build env; -x skips the build-deps check to match
    // uv's --no-build-isolation behavior (which this flow used previously).
    let dist = tmp.path().join("dist");
    let mut cmd = Command::new(&python);
    cmd.current_dir(&src)
        .args(["-m", "build", "--wheel", "-n", "-x", "-o"])
        .arg(&dist)
        .env("PATH", &new_path)
        .env("VIRTUAL_ENV", venv_path);
    if let Some(ref archs) = arch_list {
        cmd.env("TORCH_CUDA_ARCH_LIST", archs)
            .env("FLASHINFER_CUDA_ARCH_LIST", archs);
    }
    let result = run_checked(&mut cmd, "python -m build flashinfer");
    if result.is_err() {
        spinner.finish_and_clear();
    }
    result?;

    let mut wheel = find_built_wheel(&dist)?;
    if let Some(sha) = &sha {
        if std::env::var("VVM_NO_WHEEL_CACHE").is_err() {
            let cache_dir =
                wheel_cache_dir(config, "flashinfer", sha, &variant, arch_list.as_deref());
            match store_cached_wheel(&cache_dir, &wheel) {
                Ok(cached) => {
                    spinner.println(format!(
                        "  {} {}",
                        "Cached wheel:".blue().bold(),
                        cached.display()
                    ));
                    wheel = cached;
                }
                Err(e) => spinner.println(format!(
                    "  {} could not cache wheel: {}",
                    "Warning:".yellow().bold(),
                    e
                )),
            }
        }
    }

    install_flashinfer_from_wheel(venv_path, &wheel, &spinner, "built")
}

/// Install a flashinfer wheel file into the venv, then pull the companion
/// cubin/jit-cache packages. Shared tail of the cache-hit and build paths.
fn install_flashinfer_from_wheel(
    venv_path: &Path,
    wheel: &Path,
    spinner: &ProgressBar,
    source: &str,
) -> Result<(), VvmError> {
    spinner.println(format!(
        "  {} {} ({})",
        "Wheel:".blue().bold(),
        wheel.display(),
        source
    ));
    spinner.set_message("Installing wheel into venv ...");

    let bin_dir = venv_path.join("bin");
    let current_path = std::env::var("PATH").unwrap_or_default();
    let new_path = format!("{}:{}", bin_dir.display(), current_path);

    let result = run_checked(
        Command::new("uv")
            .args(["pip", "install"])
            .arg(wheel)
            .env("PATH", &new_path)
            .env("VIRTUAL_ENV", venv_path),
        "uv pip install flashinfer wheel",
    );
    if result.is_err() {
        spinner.finish_and_clear();
    }
    result?;

    // Pull matching flashinfer-cubin + flashinfer-jit-cache so runtime JIT/cubin
    // loads don't go searching upstream on first use. Non-fatal — dev builds
    // and CPU variants skip with a note.
    let _ = install_flashinfer_companion_caches(venv_path, spinner);

    spinner.finish_and_clear();
    println!("{} FlashInfer installed", "✓".green().bold());
    Ok(())
}

/// Install prebuilt flashinfer nightly wheels (python + cubin + jit-cache).
///
/// Index layout at flashinfer.ai/whl/nightly:
/// - `/nightly/` hosts flashinfer-python and flashinfer-cubin (CUDA-agnostic).
/// - `/nightly/cu{128,129,130}/` hosts flashinfer-jit-cache (CUDA-specific).
///
/// uv needs BOTH indexes because PEP 503 flat listings don't walk subpaths.
fn install_flashinfer_nightly(venv_path: &Path) -> Result<(), VvmError> {
    let cu_tag = match cuda::CudaVariant::detect() {
        cuda::CudaVariant::Cpu => {
            return Err(VvmError::CommandFailed {
                cmd: "flashinfer nightly".to_string(),
                detail: "flashinfer nightly has no CPU wheels".to_string(),
            });
        }
        other => other.as_str().to_string(),
    };
    let root_index = "https://flashinfer.ai/whl/nightly/".to_string();
    let cu_index = format!("https://flashinfer.ai/whl/nightly/{}", cu_tag);

    println!(
        "{} FlashInfer nightly ({})",
        "Installing".green().bold(),
        cu_tag.yellow()
    );

    let spinner = make_spinner();
    spinner.set_message("Resolving + downloading flashinfer nightly wheels ...");

    let bin_dir = venv_path.join("bin");
    let current_path = std::env::var("PATH").unwrap_or_default();
    let new_path = format!("{}:{}", bin_dir.display(), current_path);

    // -U --pre: nightlies are pre-release-labeled; without --pre uv skips them
    // entirely. -U ensures we actually upgrade an older install.
    let result = run_checked(
        Command::new("uv")
            .args([
                "pip",
                "install",
                "-U",
                "--pre",
                "--extra-index-url",
                &root_index,
                "--extra-index-url",
                &cu_index,
                "flashinfer-python",
                "flashinfer-cubin",
                "flashinfer-jit-cache",
            ])
            .env("PATH", &new_path)
            .env("VIRTUAL_ENV", venv_path),
        "uv pip install flashinfer nightly",
    );
    spinner.finish_and_clear();
    result?;

    println!("{} FlashInfer nightly installed", "✓".green().bold());
    Ok(())
}

/// Installed version of a distribution in the venv via importlib.metadata.
/// Deliberately not `import <pkg>` — a stale companion package makes
/// flashinfer's own import raise, which must not blind this probe.
fn pkg_version(python: &Path, dist: &str) -> Option<String> {
    let code = format!(
        "import importlib.metadata as m; print(m.version('{}'))",
        dist
    );
    let output = Command::new(python).args(["-c", &code]).output().ok()?;
    if !output.status.success() {
        return None;
    }
    let v = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if v.is_empty() {
        None
    } else {
        Some(v)
    }
}

/// True for plain release versions like "0.6.7" — the only shape that has
/// matching wheels on PyPI and flashinfer.ai/whl. Source builds off main
/// yield dev versions like "0.7.0.dev0+abc123" which won't resolve.
fn is_plain_release_version(v: &str) -> bool {
    let parts: Vec<&str> = v.split('.').collect();
    !parts.is_empty()
        && parts
            .iter()
            .all(|p| !p.is_empty() && p.chars().all(|c| c.is_ascii_digit()))
}

/// Install flashinfer-cubin + flashinfer-jit-cache at the version that was
/// just built from source. Both pin to flashinfer-python's exact version —
/// mixing versions produces undefined-symbol errors at import time; flashinfer
/// even refuses to import outright on a cubin version mismatch. So whenever
/// matching companions can't be installed (dev/fork version, CPU build, or
/// resolver failure), stale mismatched ones from a previous install are
/// removed — flashinfer then downloads cubins on demand at runtime.
fn install_flashinfer_companion_caches(
    venv_path: &Path,
    spinner: &ProgressBar,
) -> Result<(), VvmError> {
    let python = venv_path.join("bin/python");
    let version = match pkg_version(&python, "flashinfer-python") {
        Some(v) => v,
        None => {
            spinner.println(format!(
                "  {} could not detect flashinfer version — skipping cubin/jit-cache",
                "Warning:".yellow().bold()
            ));
            return Ok(());
        }
    };

    let cu_tag = match cuda::CudaVariant::detect() {
        cuda::CudaVariant::Cpu => None,
        other => Some(other.as_str().to_string()),
    };

    if is_plain_release_version(&version) {
        let Some(cu_tag) = cu_tag else {
            spinner.println(format!(
                "  {} CPU build detected — skipping flashinfer-jit-cache (no CPU wheel)",
                "Skipping companion caches:".blue().bold()
            ));
            remove_stale_flashinfer_companions(&python, venv_path, &version, spinner);
            return Ok(());
        };
        let jit_index = format!("https://flashinfer.ai/whl/{}", cu_tag);

        spinner.set_message(format!(
            "Installing flashinfer-cubin + flashinfer-jit-cache {} ({}) ...",
            version, cu_tag
        ));

        let bin_dir = venv_path.join("bin");
        let current_path = std::env::var("PATH").unwrap_or_default();
        let new_path = format!("{}:{}", bin_dir.display(), current_path);

        // Single uv pip invocation: cubin from PyPI, jit-cache from flashinfer.ai.
        // --extra-index-url keeps PyPI as the primary resolver.
        let cubin_spec = format!("flashinfer-cubin=={}", version);
        let jit_spec = format!("flashinfer-jit-cache=={}", version);
        let result = run_checked(
            Command::new("uv")
                .args(["pip", "install", "--extra-index-url", &jit_index])
                .arg(&cubin_spec)
                .arg(&jit_spec)
                .env("PATH", &new_path)
                .env("VIRTUAL_ENV", venv_path),
            "uv pip install flashinfer cubin+jit-cache",
        );
        if result.is_ok() {
            spinner.println(format!(
                "  {} flashinfer-cubin + flashinfer-jit-cache {} ({})",
                "Cache packages installed:".blue().bold(),
                version,
                cu_tag
            ));
            return Ok(());
        }
        spinner.println(format!(
            "  {} no flashinfer-cubin / flashinfer-jit-cache wheels for {} — cubins will download on demand",
            "Skipping companion caches:".blue().bold(),
            version
        ));
    } else {
        spinner.println(format!(
            "  {} flashinfer is a dev build ({}); no matching flashinfer-cubin / flashinfer-jit-cache wheels",
            "Skipping companion caches:".blue().bold(),
            version
        ));
    }

    remove_stale_flashinfer_companions(&python, venv_path, &version, spinner);
    Ok(())
}

/// Uninstall flashinfer-cubin / flashinfer-jit-cache whose version doesn't
/// match the installed flashinfer-python. Best-effort: a failed uninstall
/// leaves the venv no worse than it already was.
fn remove_stale_flashinfer_companions(
    python: &Path,
    venv_path: &Path,
    version: &str,
    spinner: &ProgressBar,
) {
    let stale: Vec<&str> = ["flashinfer-cubin", "flashinfer-jit-cache"]
        .into_iter()
        .filter(|dist| pkg_version(python, dist).is_some_and(|v| v != version))
        .collect();
    if stale.is_empty() {
        return;
    }
    spinner.println(format!(
        "  {} {} (version mismatch breaks flashinfer import)",
        "Removing stale companions:".yellow().bold(),
        stale.join(", ")
    ));
    let bin_dir = venv_path.join("bin");
    let current_path = std::env::var("PATH").unwrap_or_default();
    let new_path = format!("{}:{}", bin_dir.display(), current_path);
    let _ = run_checked(
        Command::new("uv")
            .args(["pip", "uninstall"])
            .args(&stale)
            .env("PATH", &new_path)
            .env("VIRTUAL_ENV", venv_path),
        "uv pip uninstall stale flashinfer companions",
    );
}

fn install_deepep(
    config: &Config,
    venv_path: &Path,
    git_ref: Option<&str>,
    repo_override: Option<&str>,
) -> Result<(), VvmError> {
    // DeepEP builds NVSHMEM-backed kernels against Hopper+ (arch 9.0a).
    // CUDA 12.8+ matches both the vllm Dockerfile floor and NVSHMEM 3.3.x toolchain.
    require_cuda_at_least(12, 8)?;

    let repo = repo_override.unwrap_or("https://github.com/deepseek-ai/DeepEP.git");
    // Matches vllm docker/Dockerfile DEEPEP_COMMIT_HASH. Override with --ref main for latest.
    let git_ref = git_ref.unwrap_or("73b6ea4");
    // NVSHMEM 3.3.24 ships both CUDA 12 and 13 tarballs. Override via env for pre-releases.
    let nvshmem_ver = std::env::var("VVM_NVSHMEM_VER").unwrap_or_else(|_| "3.3.24".to_string());
    // Mirror the shell script's path-traversal guard — this string goes into a URL and a filename.
    if !nvshmem_ver
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '.' || c == '-')
    {
        return Err(VvmError::InvalidSpecifier(format!(
            "VVM_NVSHMEM_VER '{}' contains invalid characters (allowed: alphanumeric, '.', '-')",
            nvshmem_ver
        )));
    }

    let cuda_ver = cuda::detect_cuda_version().ok_or_else(|| VvmError::CommandFailed {
        cmd: "cuda detection".to_string(),
        detail: "could not detect CUDA version".to_string(),
    })?;
    let cuda_major: u32 = cuda_ver
        .split('.')
        .next()
        .and_then(|s| s.parse().ok())
        .unwrap_or(12);

    let nvshmem_subdir = match std::env::consts::ARCH {
        "x86_64" => "linux-x86_64",
        "aarch64" => "linux-sbsa",
        other => {
            return Err(VvmError::CommandFailed {
                cmd: "deepep arch check".to_string(),
                detail: format!("unsupported CPU arch '{}' (need x86_64 or aarch64)", other),
            })
        }
    };

    println!(
        "{} DeepEP ({} @ {})",
        "Installing".green().bold(),
        repo.cyan(),
        git_ref.yellow()
    );
    println!(
        "  {} NVSHMEM {} for {} (CUDA {})",
        "Using:".blue().bold(),
        nvshmem_ver.yellow(),
        nvshmem_subdir,
        cuda_major
    );

    let spinner = make_spinner();
    let finish = |sp: &ProgressBar| sp.finish_and_clear();

    // Resolve NVSHMEM: (1) user-supplied $NVSHMEM_DIR → use as-is; (2) vvm cache
    // hit → reuse; (3) download into cache. Tarball is ~350MB, so persisting
    // across installs matters more than for DeepEP source itself.
    let nvshmem_dir = resolve_nvshmem(config, &nvshmem_ver, nvshmem_subdir, cuda_major, &spinner)
        .inspect_err(|_| finish(&spinner))?;

    let tmp = TempDir::new("vvm-deepep")?;
    let src =
        clone_at_ref(&tmp, repo, git_ref, "DeepEP", &spinner).inspect_err(|_| finish(&spinner))?;

    // CUDA 13 moved CCCL headers under $CUDA_HOME/include/cccl; upstream DeepEP
    // setup.py only knows about $nvshmem_dir/include. Patch matches vllm's sed.
    if cuda_major >= 13 {
        let cuda_home =
            std::env::var("CUDA_HOME").unwrap_or_else(|_| "/usr/local/cuda".to_string());
        let setup_py = src.join("setup.py");
        let contents = std::fs::read_to_string(&setup_py).inspect_err(|_| finish(&spinner))?;
        let needle = "f'{nvshmem_dir}/include']";
        if !contents.contains(needle) {
            finish(&spinner);
            return Err(VvmError::CommandFailed {
                cmd: "deepep setup.py patch".to_string(),
                detail: format!(
                    "expected pattern not found in {} (DeepEP may have changed)",
                    setup_py.display()
                ),
            });
        }
        let patched = contents.replace(
            needle,
            &format!("f'{{nvshmem_dir}}/include', '{}/include/cccl']", cuda_home),
        );
        std::fs::write(&setup_py, patched).inspect_err(|_| finish(&spinner))?;
        spinner.println(format!(
            "  {} CUDA 13 cccl include patch applied",
            "Patched:".blue().bold()
        ));
    }

    let bin_dir = venv_path.join("bin");
    let current_path = std::env::var("PATH").unwrap_or_default();
    let new_path = format!("{}:{}", bin_dir.display(), current_path);

    // cmake + ninja are build-time requirements. torch is already present (vllm venv),
    // but the ep_kernels script lists it explicitly so we mirror that for robustness.
    spinner.set_message("Installing build dependencies (cmake, ninja) ...");
    run_checked(
        Command::new("uv")
            .args(["pip", "install", "cmake", "ninja"])
            .env("PATH", &new_path)
            .env("VIRTUAL_ENV", venv_path),
        "uv pip install cmake ninja",
    )
    .inspect_err(|_| finish(&spinner))?;

    // NVSHMEM_DIR + CMAKE_PREFIX_PATH are what DeepEP's setup.py reads to find
    // the freshly downloaded NVSHMEM. TORCH_CUDA_ARCH_LIST='9.0a 10.0a' matches
    // vllm Dockerfile — override via env for other arches.
    let nvshmem_cmake = nvshmem_dir.join("lib").join("cmake");
    let existing_prefix = std::env::var("CMAKE_PREFIX_PATH").unwrap_or_default();
    let cmake_prefix = if existing_prefix.is_empty() {
        nvshmem_cmake.display().to_string()
    } else {
        format!("{}:{}", nvshmem_cmake.display(), existing_prefix)
    };
    // Arch list: respect user env first, else auto-detect from nvidia-smi and
    // map to the 'a' suffix form DeepEP needs (wgmma etc. are arch-conditional).
    // Fallback "9.0a 10.0a" matches vllm Dockerfile when nvidia-smi is absent.
    let arch_list = std::env::var("TORCH_CUDA_ARCH_LIST")
        .ok()
        .or_else(deepep_detect_arch_list)
        .unwrap_or_else(|| "9.0a 10.0a".to_string());

    // DeepEP only has Hopper+Blackwell kernels. Warn if the detected arches
    // don't include either — the build may succeed but produce nothing useful.
    if !arch_list
        .split(|c: char| c.is_whitespace() || c == ';')
        .any(|a| {
            a.starts_with("9.")
                || a.starts_with("10.")
                || a.starts_with("11.")
                || a.starts_with("12.")
        })
    {
        spinner.println(format!(
            "  {} arch list '{}' has no Hopper/Blackwell entry — DeepEP kernels require sm_90+",
            "Warning:".yellow().bold(),
            arch_list
        ));
    }

    spinner.println(format!("  {} {}", "Arches:".blue().bold(), arch_list));
    spinner.set_message("Building + installing DeepEP (this can take 10-30 min)...");

    let mut cmd = Command::new("uv");
    cmd.args(["pip", "install", "--no-build-isolation"])
        .arg(&src)
        .env("PATH", &new_path)
        .env("VIRTUAL_ENV", venv_path)
        .env("NVSHMEM_DIR", &nvshmem_dir)
        .env("CMAKE_PREFIX_PATH", &cmake_prefix)
        .env("TORCH_CUDA_ARCH_LIST", &arch_list);
    if std::env::var("VVM_VERBOSE").is_ok() {
        cmd.arg("-vvv");
    }
    let result = run_checked(&mut cmd, "uv pip install deepep");
    spinner.finish_and_clear();
    result?;

    println!("{} DeepEP installed", "✓".green().bold());
    // Multi-node users still need IBGDA enabled in the NVIDIA driver. Single-node
    // runs are fine without this. Mirrors vllm's ep_kernels README step 2.
    println!(
        "{} multi-node EP needs IBGDA. On the host, run:",
        "note:".yellow().bold()
    );
    println!(
        "  {}",
        "sudo bash ~/vllm/tools/ep_kernels/configure_system_drivers.sh && sudo reboot".dimmed()
    );
    Ok(())
}

/// Map nvidia-smi compute capabilities to DeepEP's expected arch strings
/// (space-separated, '.a' suffix). Returns None when nvidia-smi can't be
/// queried — caller falls back to the vllm Dockerfile default.
fn deepep_detect_arch_list() -> Option<String> {
    // detect_gpu_arch_list returns e.g. "9.0;10.0" (CMake-list form).
    let raw = cuda::detect_gpu_arch_list()?;
    let parts: Vec<String> = raw
        .split(';')
        .filter(|s| !s.is_empty())
        .map(|a| {
            // '9.0' -> '9.0a'. Skip if already has letter suffix.
            if a.chars().last().is_some_and(|c| c.is_alphabetic()) {
                a.to_string()
            } else {
                format!("{}a", a)
            }
        })
        .collect();
    if parts.is_empty() {
        None
    } else {
        Some(parts.join(" "))
    }
}

/// Resolve an NVSHMEM install directory usable by DeepEP.
///
/// Priority:
/// 1. `$NVSHMEM_DIR` — honor user/env override unconditionally (validate the marker).
/// 2. `~/.vvm/cache/nvshmem/<ver>_cuda<major>_<arch>/nvshmem/` — reuse across installs.
/// 3. Download tarball into cache, extract, return the new path.
///
/// The marker we validate is the CMake config file, since that's what DeepEP's
/// `CMAKE_PREFIX_PATH` lookup consumes.
fn resolve_nvshmem(
    config: &Config,
    nvshmem_ver: &str,
    nvshmem_subdir: &str,
    cuda_major: u32,
    spinner: &ProgressBar,
) -> Result<PathBuf, VvmError> {
    // 1. User override
    if let Ok(dir) = std::env::var("NVSHMEM_DIR") {
        let path = PathBuf::from(dir);
        if has_nvshmem_cmake(&path) {
            spinner.println(format!(
                "  {} {} (from $NVSHMEM_DIR)",
                "Using cached NVSHMEM:".blue().bold(),
                path.display()
            ));
            return Ok(path);
        }
        spinner.println(format!(
            "  {} $NVSHMEM_DIR={} has no nvshmem-config.cmake, ignoring",
            "Warning:".yellow().bold(),
            path.display()
        ));
    }

    // 2. vvm cache
    let cache_key = format!("{}_cuda{}_{}", nvshmem_ver, cuda_major, nvshmem_subdir);
    let cache_root = config.cache_dir().join("nvshmem").join(&cache_key);
    let cached_nvshmem = cache_root.join("nvshmem");
    if has_nvshmem_cmake(&cached_nvshmem) {
        spinner.println(format!(
            "  {} {}",
            "Using cached NVSHMEM:".blue().bold(),
            cached_nvshmem.display()
        ));
        return Ok(cached_nvshmem);
    }

    // 3. Download into cache
    std::fs::create_dir_all(&cache_root)?;
    let nvshmem_file = format!(
        "libnvshmem-{}-{}_cuda{}-archive.tar.xz",
        nvshmem_subdir, nvshmem_ver, cuda_major
    );
    let nvshmem_url = format!(
        "https://developer.download.nvidia.com/compute/nvshmem/redist/libnvshmem/{}/{}",
        nvshmem_subdir, nvshmem_file
    );
    let tarball = cache_root.join(&nvshmem_file);

    spinner.set_message(format!(
        "Downloading NVSHMEM {} (cached for reuse) ...",
        nvshmem_ver
    ));
    run_checked(
        Command::new("curl")
            .args(["-fSL", "--retry", "3"])
            .arg(&nvshmem_url)
            .arg("-o")
            .arg(&tarball),
        "curl nvshmem",
    )?;

    spinner.set_message("Extracting NVSHMEM ...");
    run_checked(
        Command::new("tar")
            .arg("-xf")
            .arg(&tarball)
            .arg("-C")
            .arg(&cache_root),
        "tar extract nvshmem",
    )?;

    let extracted = cache_root.join(nvshmem_file.trim_end_matches(".tar.xz"));
    // Remove any stale half-extracted dir before rename.
    let _ = std::fs::remove_dir_all(&cached_nvshmem);
    std::fs::rename(&extracted, &cached_nvshmem).map_err(|e| VvmError::CommandFailed {
        cmd: "nvshmem rename".to_string(),
        detail: format!(
            "failed to rename {} -> {}: {}",
            extracted.display(),
            cached_nvshmem.display(),
            e
        ),
    })?;
    // Keep the tarball around too — re-extracting is cheap and users can inspect it.
    Ok(cached_nvshmem)
}

fn has_nvshmem_cmake(dir: &Path) -> bool {
    // NVSHMEM 3.x installs `lib/cmake/nvshmem/nvshmem-config.cmake` (lowercase) —
    // spot-check that, plus the include header as a belt-and-braces sanity check.
    dir.join("lib")
        .join("cmake")
        .join("nvshmem")
        .join("nvshmem-config.cmake")
        .exists()
        && dir.join("include").join("nvshmem.h").exists()
}

/// Map the system CUDA version onto the arch set flashinfer actually supports
/// for that version. Returns None when CUDA isn't detectable — the install
/// will then default to torch's own arch list, which is usually fine.
fn flashinfer_default_arch_list() -> Option<String> {
    let ver = cuda::detect_cuda_version()?;
    let mut parts = ver.split('.');
    let major: u32 = parts.next()?.parse().ok()?;
    let minor: u32 = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0);
    Some(match (major, minor) {
        (11, _) => "7.5 8.0 8.9".to_string(),
        (12, 0..=7) => "7.5 8.0 8.9 9.0a".to_string(),
        (12, _) => "7.5 8.0 8.9 9.0a 10.0a 10.3a 12.0".to_string(),
        _ => "7.5 8.0 8.9 9.0a 10.0f 12.0".to_string(),
    })
}

fn require_cuda_at_least(req_major: u32, req_minor: u32) -> Result<(), VvmError> {
    let ver = cuda::detect_cuda_version().ok_or_else(|| VvmError::CommandFailed {
        cmd: "cuda detection".to_string(),
        detail: "could not detect CUDA version (needed nvidia-smi / nvcc / version.json)"
            .to_string(),
    })?;
    let mut parts = ver.split('.');
    let major: u32 = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0);
    let minor: u32 = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0);
    if major < req_major || (major == req_major && minor < req_minor) {
        return Err(VvmError::CommandFailed {
            cmd: "cuda check".to_string(),
            detail: format!(
                "requires CUDA >= {}.{}, found {}",
                req_major, req_minor, ver
            ),
        });
    }
    Ok(())
}

/// Run a Command. When VVM_VERBOSE is set, output streams live. Otherwise
/// we capture stdout+stderr and only surface the tail on failure — spinner
/// mode wants a clean line, but a broken CUDA compile still needs diagnosable
/// output when it blows up.
fn run_checked(cmd: &mut Command, label: &str) -> Result<(), VvmError> {
    if std::env::var("VVM_VERBOSE").is_ok() {
        let status = cmd.status().map_err(|e| VvmError::CommandFailed {
            cmd: label.to_string(),
            detail: e.to_string(),
        })?;
        if !status.success() {
            return Err(VvmError::CommandFailed {
                cmd: label.to_string(),
                detail: format!("exit code: {:?}", status.code()),
            });
        }
        return Ok(());
    }

    let output = cmd
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .map_err(|e| VvmError::CommandFailed {
            cmd: label.to_string(),
            detail: e.to_string(),
        })?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        // Combine and tail to the last ~60 lines — enough to see the nvcc / linker error.
        let combined = if stderr.trim().is_empty() {
            stdout.to_string()
        } else if stdout.trim().is_empty() {
            stderr.to_string()
        } else {
            format!("{}\n{}", stdout, stderr)
        };
        let tail: Vec<&str> = combined.lines().rev().take(60).collect();
        let tail_str: String = tail.into_iter().rev().collect::<Vec<_>>().join("\n");
        return Err(VvmError::CommandFailed {
            cmd: label.to_string(),
            detail: format!(
                "exit code: {:?}\n--- last output ---\n{}\n--- end output ---\n(run with -v for full output)",
                output.status.code(),
                tail_str
            ),
        });
    }
    Ok(())
}

/// Resolve a git ref to a full commit SHA without cloning. Full SHAs pass
/// through; branch/tag names go through `git ls-remote`. Returns None for
/// short SHAs, unknown refs, or unreachable remotes — callers fall back to
/// resolving after clone.
fn resolve_remote_sha(repo: &str, git_ref: &str) -> Option<String> {
    if git_ref.len() == 40 && git_ref.chars().all(|c| c.is_ascii_hexdigit()) {
        return Some(git_ref.to_ascii_lowercase());
    }
    let output = Command::new("git")
        .args(["ls-remote", repo, git_ref])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&output.stdout);
    let mut head = None;
    let mut peeled_tag = None;
    let mut tag = None;
    let mut first = None;
    let mut matches = 0;
    for line in text.lines() {
        let Some((sha, name)) = line.split_once('\t') else {
            continue;
        };
        matches += 1;
        first.get_or_insert(sha);
        if name == format!("refs/heads/{}", git_ref) {
            head = Some(sha);
        } else if name == format!("refs/tags/{}^{{}}", git_ref) {
            peeled_tag = Some(sha);
        } else if name == format!("refs/tags/{}", git_ref) {
            tag = Some(sha);
        }
    }
    // Mirror `git checkout`: a branch wins over a same-named tag. Annotated
    // tags peel (`^{}`) to the commit they point at — prefer that over the
    // tag object's own SHA so the key matches a directly-passed commit.
    head.or(peeled_tag)
        .or(tag)
        .or(if matches == 1 { first } else { None })
        .map(str::to_string)
}

/// Full SHA of HEAD in a checked-out clone, or None if git fails.
fn rev_parse_head(repo_dir: &Path) -> Option<String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(repo_dir)
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let sha = String::from_utf8_lossy(&output.stdout).trim().to_string();
    (sha.len() == 40).then_some(sha)
}

/// Directory holding cached wheels for one (lib, commit, CUDA variant,
/// arch list) combination. The arch list is part of the key because it
/// determines which cubins get baked into the wheel.
fn wheel_cache_dir(
    config: &Config,
    lib: &str,
    sha: &str,
    variant: &cuda::CudaVariant,
    arch_list: Option<&str>,
) -> PathBuf {
    let arch_key = arch_list
        .map(|a| a.replace([' ', '/'], "_"))
        .unwrap_or_else(|| "default".to_string());
    config.cache_dir().join("wheels").join(lib).join(format!(
        "{}-{}-{}",
        sha,
        variant.as_str(),
        arch_key
    ))
}

/// Cached flashinfer wheel for this exact build configuration, if present.
/// VVM_NO_WHEEL_CACHE=1 disables reads (and writes, checked at store time).
fn cached_flashinfer_wheel(
    config: &Config,
    sha: Option<&str>,
    variant: &cuda::CudaVariant,
    arch_list: Option<&str>,
) -> Option<PathBuf> {
    if std::env::var("VVM_NO_WHEEL_CACHE").is_ok() {
        return None;
    }
    let dir = wheel_cache_dir(config, "flashinfer", sha?, variant, arch_list);
    let mut wheels: Vec<PathBuf> = std::fs::read_dir(dir)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "whl"))
        .collect();
    wheels.sort();
    wheels.pop()
}

/// Copy a freshly built wheel into the cache dir. Copies to a temp name and
/// renames so a concurrent reader never sees a half-written wheel.
fn store_cached_wheel(cache_dir: &Path, wheel: &Path) -> Result<PathBuf, std::io::Error> {
    std::fs::create_dir_all(cache_dir)?;
    let name = wheel.file_name().ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "wheel path has no filename",
        )
    })?;
    let dest = cache_dir.join(name);
    let tmp = cache_dir.join(format!(".tmp-{}", std::process::id()));
    std::fs::copy(wheel, &tmp)?;
    std::fs::rename(&tmp, &dest)?;
    Ok(dest)
}

fn find_built_wheel(dist: &Path) -> Result<PathBuf, VvmError> {
    let mut wheels: Vec<PathBuf> = std::fs::read_dir(dist)?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "whl"))
        .collect();
    if wheels.is_empty() {
        return Err(VvmError::CommandFailed {
            cmd: "wheel lookup".to_string(),
            detail: format!("no .whl produced in {}", dist.display()),
        });
    }
    wheels.sort();
    Ok(wheels.pop().unwrap())
}

/// Tiny scoped temp dir. Avoid adding a `tempfile` crate just for this;
/// build failures still need the dir cleaned, hence Drop.
struct TempDir {
    path: PathBuf,
}

impl TempDir {
    fn new(prefix: &str) -> Result<Self, VvmError> {
        let path = std::env::temp_dir().join(format!(
            "{}-{}-{}",
            prefix,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::create_dir_all(&path)?;
        Ok(Self { path })
    }
    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

#[cfg(test)]
mod tests {
    use super::{parse_deepgemm_git_ref, resolve_remote_sha, wheel_cache_dir};
    use crate::config::Config;
    use crate::cuda::CudaVariant;

    #[test]
    fn full_sha_resolves_without_network() {
        let sha = "3bc903383ce865445439d9cc76903e07f37d375c";
        assert_eq!(
            resolve_remote_sha("ssh://nonexistent.invalid/repo.git", sha).as_deref(),
            Some(sha)
        );
        // Short SHA must not pass through — it can't key the cache uniquely.
        assert_eq!(
            resolve_remote_sha("ssh://nonexistent.invalid/repo.git", "3bc9033"),
            None
        );
    }

    #[test]
    fn wheel_cache_key_includes_sha_variant_and_archs() {
        let config = Config {
            base_dir: std::path::PathBuf::from("/vvm"),
        };
        let dir = wheel_cache_dir(
            &config,
            "flashinfer",
            "abc123",
            &CudaVariant::Cu130,
            Some("7.5 8.0 10.0a"),
        );
        assert_eq!(
            dir,
            std::path::PathBuf::from("/vvm/cache/wheels/flashinfer/abc123-cu130-7.5_8.0_10.0a")
        );
        let no_arch = wheel_cache_dir(&config, "flashinfer", "abc123", &CudaVariant::Cu130, None);
        assert!(no_arch.ends_with("abc123-cu130-default"));
    }

    #[test]
    fn parses_double_quoted_pin() {
        let s = r#"#!/usr/bin/env bash
set -e
DEEPGEMM_GIT_REF="891d57b4db1071624b5c8fa0d1e51cb317fa709f"
git checkout "$DEEPGEMM_GIT_REF"
"#;
        assert_eq!(
            parse_deepgemm_git_ref(s).as_deref(),
            Some("891d57b4db1071624b5c8fa0d1e51cb317fa709f")
        );
    }

    #[test]
    fn parses_single_quoted_pin() {
        let s = "DEEPGEMM_GIT_REF='abc123'\n";
        assert_eq!(parse_deepgemm_git_ref(s).as_deref(), Some("abc123"));
    }

    #[test]
    fn parses_unquoted_pin_with_trailing_comment() {
        let s = "DEEPGEMM_GIT_REF=deadbeef # inline comment\n";
        assert_eq!(parse_deepgemm_git_ref(s).as_deref(), Some("deadbeef"));
    }

    #[test]
    fn returns_none_when_missing() {
        assert_eq!(parse_deepgemm_git_ref("echo hi\n"), None);
    }
}
