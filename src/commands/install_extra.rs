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

/// Vendored fallbacks for when we can't read the active vllm's pin (no current
/// version, offline, or a vllm predating the file the pin lives in).
const FALLBACK_DEEPGEMM_REF: &str = "477618cd51baffca09c4b0b87e97c03fe827ef03";
const FALLBACK_DEEPEP_REF: &str = "73b6ea4";
const FALLBACK_FLASHINFER_REF: &str = "main";

/// Where a pinned ref came from, for the `[pin: ...]` line at install time.
/// Worth surfacing: "active vllm" and "bundled fallback" can differ by months,
/// and that difference is exactly what breaks a build.
const PIN_FROM_USER: &str = "user";
const PIN_FROM_VLLM: &str = "active vllm";
const PIN_FROM_FALLBACK: &str = "bundled fallback";

/// Read a file out of the currently-active vllm. Source-of-truth order:
///   1. Local source tree (`source_path` from metadata) — covers `--path` /
///      `--source` / `--compile` installs and works offline.
///   2. raw.githubusercontent.com at the install's commit (or release tag) —
///      covers wheel-based installs, where only the Python package ships and
///      `tools/` + `docker/` are absent.
///
/// Returns None on any failure; callers fall back to a vendored default.
fn read_file_from_active_vllm(config: &Config, rel_path: &str) -> Option<String> {
    let link = config.current_link();
    let target = std::fs::read_link(&link).ok()?;
    let meta = VersionMetadata::load(&target).ok()?;

    if let Some(sp) = &meta.source_path {
        let local = Path::new(sp).join(rel_path);
        if local.is_file() {
            if let Ok(text) = std::fs::read_to_string(&local) {
                return Some(text);
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
        "https://raw.githubusercontent.com/vllm-project/vllm/{}/{}",
        github_ref, rel_path
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
    resp.text().ok()
}

/// Read `tools/install_deepgemm.sh` from the active vllm and pull out the
/// `DEEPGEMM_GIT_REF=...` value.
fn resolve_deepgemm_ref_from_active_vllm(config: &Config) -> Option<String> {
    let text = read_file_from_active_vllm(config, "tools/install_deepgemm.sh")?;
    parse_deepgemm_git_ref(&text)
}

/// Read a pinned version out of the active vllm's `docker/versions.json`.
///
/// That file is auto-generated from the Dockerfile's `ARG` defaults and is
/// vllm's own centralized version manifest — the same source its release
/// images build from. Keys look like `DEEPEP_COMMIT_HASH`, `FLASHINFER_VERSION`.
fn resolve_versions_json_pin(config: &Config, key: &str) -> Option<String> {
    let text = read_file_from_active_vllm(config, "docker/versions.json")?;
    parse_versions_json_pin(&text, key)
}

/// True when a string is plausibly a git object name rather than a version.
/// Short SHAs start with a digit as often as not (`73b6ea4`), so "starts with
/// a digit" alone can't tell the two apart — a dot is what a version has and
/// a SHA never does.
fn looks_like_sha(s: &str) -> bool {
    (7..=40).contains(&s.len()) && s.chars().all(|c| c.is_ascii_hexdigit())
}

/// Turn a release version into its conventional `v`-prefixed git tag.
///
/// vllm pins `FLASHINFER_VERSION=0.6.15.post1`; flashinfer tags that commit
/// `v0.6.15.post1`. Anything already ref-shaped — leading `v`, a SHA, a branch
/// name — is passed through untouched.
fn version_to_tag(version: &str) -> String {
    let v = version.trim();
    let is_version = v.starts_with(|c: char| c.is_ascii_digit()) && v.contains('.');
    if !is_version || looks_like_sha(v) {
        return v.to_string();
    }
    format!("v{}", v)
}

/// Resolve what the user meant by `--ref <release version>`.
///
/// People name a flashinfer release `0.6.7`; upstream tags it `v0.6.7`, and a
/// bare `0.6.7` dies at `git checkout` with an unhelpful pathspec error. Probe
/// the literal ref first — a fork may well tag without the prefix, and that
/// must keep working — and only then try the `v`-prefixed form.
///
/// Returns Some(rewritten) only when the literal ref is unresolvable and the
/// prefixed one exists. Anything ls-remote can't see (short SHAs especially)
/// is left alone for the post-checkout resolution to handle.
fn resolve_versionish_ref(repo: &str, git_ref: &str) -> Option<String> {
    let tagged = version_to_tag(git_ref);
    if tagged == git_ref {
        return None;
    }
    if resolve_remote_sha(repo, git_ref).is_some() {
        return None;
    }
    resolve_remote_sha(repo, &tagged).map(|_| tagged)
}

/// Reject a ref that resolves nowhere, before anything expensive runs.
///
/// A typo like `--ref v0.67` (for `v0.6.7`) is version-shaped enough to look
/// real: the PyPI probe misses, then a full clone runs only to die at
/// `git checkout` with a bare pathspec error. "No published wheel" and "no such
/// ref" need different answers — the first falls back to a source build, the
/// second can only be a mistake.
///
/// Short SHAs are exempt: `ls-remote` genuinely cannot see them, and the
/// existing post-checkout resolution already covers that case.
///
/// Returns the resolved SHA when ls-remote knew it, so callers don't re-probe.
fn ensure_ref_exists(repo: &str, git_ref: &str) -> Result<Option<String>, VvmError> {
    if looks_like_sha(git_ref) {
        // A full 40-char SHA is its own resolution, no network needed — and it
        // must stay resolved here or the wheel-cache fast path loses its key
        // and pays for a clone. Shorter abbreviations stay None either way.
        return Ok((git_ref.len() == 40).then(|| git_ref.to_ascii_lowercase()));
    }
    if let Some(sha) = resolve_remote_sha(repo, git_ref) {
        return Ok(Some(sha));
    }
    Err(VvmError::InvalidSpecifier(format!(
        "could not resolve '{}' as a branch, tag, or commit in {}{}",
        git_ref,
        repo,
        suggest_similar_refs(repo, git_ref)
    )))
}

/// Best-effort "did you mean" list for a ref that doesn't exist.
fn suggest_similar_refs(repo: &str, git_ref: &str) -> String {
    let Ok(output) = Command::new("git")
        .args(["ls-remote", "--tags", repo])
        .output()
    else {
        return String::new();
    };
    if !output.status.success() {
        return String::new();
    }
    let text = String::from_utf8_lossy(&output.stdout);
    let tags: Vec<&str> = text
        .lines()
        .filter_map(|l| l.split_once('\t'))
        .map(|(_, name)| name.trim_end_matches("^{}"))
        .filter_map(|name| name.strip_prefix("refs/tags/"))
        .collect();
    format_close_tags(&tags, git_ref)
}

/// Pure half of `suggest_similar_refs`, factored out so it's testable.
///
/// Walks back the query's own prefix until some tags match, so `v0.67` lands on
/// the `v0.6*` family, then shows the newest five. Returns an empty string when
/// nothing helpful turns up — a guess is a courtesy, never a reason to fail
/// differently.
fn format_close_tags(tags: &[&str], query: &str) -> String {
    for take in (2..=query.len()).rev() {
        // Byte positions; splitting mid-char would panic on non-ASCII input.
        if !query.is_char_boundary(take) {
            continue;
        }
        let prefix = &query[..take];
        let mut hits: Vec<&str> = tags
            .iter()
            .copied()
            .filter(|t| t.starts_with(prefix))
            .collect();
        if hits.is_empty() {
            continue;
        }
        // Numeric-aware, newest first: plain lexicographic order buries every
        // 0.x.1N release below 0.x.9, hiding exactly the tags worth suggesting.
        hits.sort_unstable_by(|a, b| natural_order(b, a));
        hits.dedup();
        let shown: Vec<&str> = hits.into_iter().take(5).collect();
        return format!(". Close tags: {}", shown.join(", "));
    }
    String::new()
}

/// Compare strings the way humans read versions: digit runs compare as
/// numbers, everything else as text, so `v0.6.15` > `v0.6.9`.
fn natural_order(a: &str, b: &str) -> std::cmp::Ordering {
    let (mut a, mut b) = (a, b);
    loop {
        if a.is_empty() || b.is_empty() {
            return a.len().cmp(&b.len());
        }
        let (ra, rest_a) = split_leading_run(a);
        let (rb, rest_b) = split_leading_run(b);
        let ord = match (ra.parse::<u128>(), rb.parse::<u128>()) {
            (Ok(x), Ok(y)) => x.cmp(&y),
            _ => ra.cmp(rb),
        };
        if ord != std::cmp::Ordering::Equal {
            return ord;
        }
        (a, b) = (rest_a, rest_b);
    }
}

/// Split off the leading run of digits, or of non-digits.
fn split_leading_run(s: &str) -> (&str, &str) {
    let digits = s.starts_with(|c: char| c.is_ascii_digit());
    let end = s
        .char_indices()
        .find(|(_, c)| c.is_ascii_digit() != digits)
        .map_or(s.len(), |(i, _)| i);
    s.split_at(end)
}

/// Take `--ref` as the user gave it, rewriting a bare release version to the
/// `v`-prefixed tag when that's the only form the repo actually carries.
/// Costs nothing for the common cases — a SHA or branch name never reaches the
/// network probe.
fn user_ref(repo: &str, git_ref: &str) -> String {
    match resolve_versionish_ref(repo, git_ref) {
        Some(tagged) => {
            eprintln!(
                "  {} '{}' isn't a ref in {} — using tag {}",
                "note:".blue().bold(),
                git_ref,
                repo,
                tagged.yellow()
            );
            tagged
        }
        None => git_ref.to_string(),
    }
}

/// Extract `.variable.<key>.default` from a bake-style versions.json.
fn parse_versions_json_pin(text: &str, key: &str) -> Option<String> {
    let json: serde_json::Value = serde_json::from_str(text).ok()?;
    let value = json.get("variable")?.get(key)?.get("default")?.as_str()?;
    let value = value.trim();
    if value.is_empty() {
        None
    } else {
        Some(value.to_string())
    }
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
        (user_ref(repo, r), PIN_FROM_USER)
    } else {
        match resolve_deepgemm_ref_from_active_vllm(config) {
            Some(r) => (r, PIN_FROM_VLLM),
            None => {
                eprintln!(
                    "  {} couldn't read DeepGEMM pin from active vllm — falling back to vvm's bundled default",
                    "warning:".yellow().bold()
                );
                (FALLBACK_DEEPGEMM_REF.to_string(), PIN_FROM_FALLBACK)
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

    // Catch a mistyped ref before the spinner starts or the clone begins.
    ensure_ref_exists(repo, git_ref)?;

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
    // Follow the active vllm's FLASHINFER_VERSION. vllm pins a PyPI version
    // ("0.6.15.post1"); upstream tags the matching commit "v0.6.15.post1", so
    // the version maps straight onto a git ref. Pinning also makes the wheel
    // cache actually hit — `main` moves, a tag doesn't.
    let (git_ref_owned, ref_source) = if let Some(r) = git_ref {
        (user_ref(repo, r), PIN_FROM_USER)
    } else {
        match resolve_versions_json_pin(config, "FLASHINFER_VERSION").map(|v| version_to_tag(&v)) {
            Some(r) => (r, PIN_FROM_VLLM),
            None => {
                eprintln!(
                    "  {} couldn't read FlashInfer pin from active vllm — falling back to {}",
                    "warning:".yellow().bold(),
                    FALLBACK_FLASHINFER_REF
                );
                (FALLBACK_FLASHINFER_REF.to_string(), PIN_FROM_FALLBACK)
            }
        }
    };
    let git_ref: &str = &git_ref_owned;

    println!(
        "{} FlashInfer ({} @ {}) [pin: {}]",
        "Installing".green().bold(),
        repo.cyan(),
        git_ref.yellow(),
        ref_source.blue()
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

    // Fastest path: a released version needs no build at all. flashinfer-python
    // publishes a py3-none-any wheel to PyPI — the CUDA-specific kernels live
    // in the cubin/jit-cache companions, which get installed right after either
    // way. Compiling is only genuinely required for a branch, a raw commit, or
    // a fork, so don't spend 10-20 minutes reproducing a published artifact.
    let mut pypi_miss = None;
    if repo_override.is_none() && std::env::var("VVM_FLASHINFER_BUILD").is_err() {
        if let Some(version) = release_version_from_ref(git_ref) {
            match install_flashinfer_release(venv_path, &version, &spinner) {
                Ok(()) => return Ok(()),
                Err(_) => pypi_miss = Some(version),
            }
        }
    }

    // Only now pay for git. Deliberately after the PyPI attempt: a published
    // release installs with zero ls-remote round-trips, and a version on PyPI
    // whose tag was never pushed still installs. A PyPI miss splits two ways —
    // a ref that exists (rc pulled, yanked release) falls back to the source
    // build below; a ref that resolves nowhere is a typo, and dies here
    // instead of after a full clone.
    let resolved_sha =
        ensure_ref_exists(repo, git_ref).inspect_err(|_| spinner.finish_and_clear())?;
    if let Some(version) = pypi_miss {
        spinner.println(format!(
            "  {} no published wheel for flashinfer-python {} — building from source",
            "Note:".yellow().bold(),
            version
        ));
    }

    // Next: the ref resolves to a commit we've already built a wheel for
    // (same CUDA variant + arch list) — install it without cloning anything.
    // The wheel is py3-none-any (kernels ship as cubin/JIT data, no torch-ABI
    // binding), so one build serves every version's venv.
    let mut sha = resolved_sha;
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

/// The PyPI version a ref names, if it names one at all.
///
/// `v0.6.15.post1` and `0.6.15.post1` both mean the released package; `main`,
/// a SHA, or any other branch name means "whatever that tree contains", which
/// only a source build can produce.
fn release_version_from_ref(git_ref: &str) -> Option<String> {
    let v = git_ref.strip_prefix('v').unwrap_or(git_ref);
    if looks_like_sha(v) || !is_published_release_version(v) {
        return None;
    }
    // A release is at least `<major>.<minor>`; a bare `1` is a branch name.
    if !v.contains('.') {
        return None;
    }
    Some(v.to_string())
}

/// Install a released flashinfer-python straight from PyPI, then its
/// companions. No clone, no compile — seconds instead of tens of minutes.
fn install_flashinfer_release(
    venv_path: &Path,
    version: &str,
    spinner: &ProgressBar,
) -> Result<(), VvmError> {
    let bin_dir = venv_path.join("bin");
    let current_path = std::env::var("PATH").unwrap_or_default();
    let new_path = format!("{}:{}", bin_dir.display(), current_path);

    spinner.set_message(format!(
        "Installing prebuilt flashinfer-python {} ...",
        version
    ));
    // --pre so rc tags resolve; pinned exactly, so it can't drift to another.
    run_checked(
        Command::new("uv")
            .args(["pip", "install", "--pre"])
            .arg(format!("flashinfer-python=={}", version))
            .env("PATH", &new_path)
            .env("VIRTUAL_ENV", venv_path),
        "uv pip install flashinfer-python",
    )?;

    spinner.println(format!(
        "  {} flashinfer-python {} (prebuilt, no compile)",
        "Wheel:".blue().bold(),
        version
    ));
    install_flashinfer_companion_caches(venv_path, spinner)?;
    verify_flashinfer(venv_path, spinner);
    spinner.finish_and_clear();
    println!("{} FlashInfer installed", "✓".green().bold());
    Ok(())
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

    verify_flashinfer(venv_path, spinner);
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
    if result.is_err() {
        spinner.finish_and_clear();
    }
    result?;

    verify_flashinfer(venv_path, &spinner);
    spinner.finish_and_clear();
    println!("{} FlashInfer nightly installed", "✓".green().bold());
    Ok(())
}

/// Post-install check: run `flashinfer show-config` inside the venv.
///
/// This exercises the entire load path — flashinfer's import, torch, the CUDA
/// runtime, and the cubin/jit-cache wiring — and reports which versions
/// actually landed, so a companion mismatch or broken import surfaces now
/// instead of mid-inference. Failures are warnings, not errors: the wheels are
/// already correctly installed, and show-config also reaches NVIDIA's cubin
/// repository, so a network hiccup must not fail the install.
///
/// Skipped without GPU access (login nodes), same policy as vllm's own
/// post-install verification.
fn verify_flashinfer(venv_path: &Path, spinner: &ProgressBar) {
    if !cuda::has_gpu_access() {
        spinner.println(format!(
            "  {} no GPU on this node — skipping `flashinfer show-config`",
            "Verify:".blue().bold()
        ));
        return;
    }

    let bin_dir = venv_path.join("bin");
    let current_path = std::env::var("PATH").unwrap_or_default();
    let new_path = format!("{}:{}", bin_dir.display(), current_path);

    let cli = bin_dir.join("flashinfer");
    let mut cmd = if cli.exists() {
        let mut c = Command::new(&cli);
        c.arg("show-config");
        c
    } else {
        // Pre-CLI flashinfer: an import + version print still proves it loads.
        let mut c = Command::new(bin_dir.join("python"));
        c.args([
            "-c",
            "import flashinfer; print('FlashInfer version:', flashinfer.__version__)",
        ]);
        c
    };
    cmd.env("PATH", &new_path).env("VIRTUAL_ENV", venv_path);

    spinner.set_message("Verifying installation (flashinfer show-config)...");

    if std::env::var("VVM_VERBOSE").is_ok() {
        // Stream the full dump (cubin fetches, module registration and all).
        match cmd.status() {
            Ok(s) if s.success() => {}
            _ => spinner.println(format!(
                "  {} `flashinfer show-config` failed — the install may not import cleanly",
                "Warning:".yellow().bold()
            )),
        }
        return;
    }

    match cmd.stdout(Stdio::piped()).stderr(Stdio::piped()).output() {
        Ok(out) if out.status.success() => {
            let stdout = String::from_utf8_lossy(&out.stdout);
            let lines = summarize_show_config(&stdout);
            if lines.is_empty() {
                spinner.println(format!(
                    "  {} flashinfer show-config OK",
                    "Verify:".blue().bold()
                ));
            }
            for l in lines {
                spinner.println(format!("  {} {}", "Verify:".blue().bold(), l));
            }
        }
        Ok(out) => {
            let stderr = String::from_utf8_lossy(&out.stderr);
            let stdout = String::from_utf8_lossy(&out.stdout);
            let combined = if stderr.trim().is_empty() {
                stdout
            } else {
                stderr
            };
            let tail: Vec<&str> = {
                let mut t: Vec<&str> = combined.lines().rev().take(8).collect();
                t.reverse();
                t
            };
            spinner.println(format!(
                "  {} `flashinfer show-config` failed — the install may not import cleanly:\n    {}",
                "Warning:".yellow().bold(),
                tail.join("\n    ")
            ));
        }
        Err(e) => {
            spinner.println(format!(
                "  {} could not run flashinfer verification: {}",
                "Warning:".yellow().bold(),
                e
            ));
        }
    }
}

/// Pick the identity lines out of `flashinfer show-config`'s dump.
///
/// The full output runs to hundreds of lines (checksum downloads, module
/// registration); at install time only the version/torch/CUDA identity matters
/// — that's where a companion mismatch or a wrong-variant torch shows up.
fn summarize_show_config(stdout: &str) -> Vec<String> {
    const KEYS: [&str; 5] = [
        "FlashInfer version:",
        "flashinfer-cubin",
        "flashinfer-jit-cache",
        "Torch version:",
        "CUDA runtime available:",
    ];
    stdout
        .lines()
        .map(str::trim)
        .filter(|l| KEYS.iter().any(|k| l.starts_with(k)))
        .map(str::to_string)
        .collect()
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

/// A version with its PEP 440 local label dropped (`0.6.17rc1+cu130` →
/// `0.6.17rc1`).
///
/// flashinfer-jit-cache encodes the CUDA variant there, so its installed
/// version never equals flashinfer-python's as a plain string even when the
/// pair is correctly matched. Comparing raw versions read a freshly installed
/// jit-cache as stale and uninstalled it.
fn public_version(v: &str) -> &str {
    v.split_once('+').map(|(base, _)| base).unwrap_or(v)
}

/// True for a version the companion indexes could plausibly carry a wheel for.
///
/// Excludes only what never gets published: a PEP 440 local label (`+g1a2b3c4`,
/// which is what a source build stamps on) and `.devN`. `.postN` and `rcN` are
/// ordinary published releases — flashinfer ships jit-cache wheels for both
/// (`0.6.8.post1`, `0.6.9rc1`), and an all-digits-only test misread them as dev
/// builds and silently skipped the companions.
fn is_published_release_version(v: &str) -> bool {
    !v.is_empty()
        && v.starts_with(|c: char| c.is_ascii_digit())
        && !v.contains('+')
        && !v.contains(".dev")
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

    if is_published_release_version(&version) {
        let Some(cu_tag) = cu_tag else {
            spinner.println(format!(
                "  {} CPU build detected — skipping flashinfer-jit-cache (no CPU wheel)",
                "Skipping companion caches:".blue().bold()
            ));
            remove_stale_flashinfer_companions(&python, venv_path, &version, spinner);
            return Ok(());
        };
        let bin_dir = venv_path.join("bin");
        let current_path = std::env::var("PATH").unwrap_or_default();
        let new_path = format!("{}:{}", bin_dir.display(), current_path);

        // The two companions are not equivalent and must not share one uv
        // invocation — a miss on the optional one would take the required one
        // down with it.
        //
        // flashinfer-cubin is a *vllm* runtime requirement (requirements/cuda.txt
        // pins it alongside flashinfer-python). It left PyPI at 0.6.14 and now
        // lives only on flashinfer.ai's root index — which is why vllm's
        // setup.py strips it from install_requires: a published wheel can't
        // carry a pin nothing on PyPI resolves. Both indexes are offered so
        // pre-0.6.14 versions still resolve from PyPI.
        spinner.set_message(format!("Installing flashinfer-cubin {} ...", version));
        let cubin_ok = run_checked(
            Command::new("uv")
                .args([
                    "pip",
                    "install",
                    "--extra-index-url",
                    "https://flashinfer.ai/whl/",
                ])
                .arg(format!("flashinfer-cubin=={}", version))
                .env("PATH", &new_path)
                .env("VIRTUAL_ENV", venv_path),
            "uv pip install flashinfer-cubin",
        )
        .is_ok();
        if cubin_ok {
            spinner.println(format!(
                "  {} flashinfer-cubin {}",
                "Installed:".blue().bold(),
                version
            ));
        } else {
            spinner.println(format!(
                "  {} no flashinfer-cubin wheel for {} — cubins will download on demand",
                "Warning:".yellow().bold(),
                version
            ));
        }

        // flashinfer-jit-cache is a prebuilt JIT cache, CUDA-and-arch specific
        // and ~1.4 GB. vllm installs it in its Dockerfile but does not list it
        // as a requirement — it buys startup latency, not correctness. Failing
        // to get it is a warning, never an error.
        if std::env::var("VVM_FLASHINFER_NO_JIT_CACHE").is_ok() {
            spinner.println(format!(
                "  {} flashinfer-jit-cache (VVM_FLASHINFER_NO_JIT_CACHE) — kernels JIT on first use",
                "Skipping:".blue().bold()
            ));
        } else {
            let jit_index = format!("https://flashinfer.ai/whl/{}", cu_tag);
            spinner.set_message(format!(
                "Installing flashinfer-jit-cache {} ({}, ~1.4 GB) ...",
                version, cu_tag
            ));
            let jit_ok = run_checked(
                Command::new("uv")
                    .args(["pip", "install", "--extra-index-url", &jit_index])
                    .arg(format!("flashinfer-jit-cache=={}", version))
                    .env("PATH", &new_path)
                    .env("VIRTUAL_ENV", venv_path),
                "uv pip install flashinfer-jit-cache",
            )
            .is_ok();
            spinner.println(if jit_ok {
                format!(
                    "  {} flashinfer-jit-cache {} ({})",
                    "Installed:".blue().bold(),
                    version,
                    cu_tag
                )
            } else {
                format!(
                    "  {} no flashinfer-jit-cache wheel for {} ({}) — kernels JIT on first use",
                    "Warning:".yellow().bold(),
                    version,
                    cu_tag
                )
            });
        }
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
        .filter(|dist| {
            pkg_version(python, dist).is_some_and(|v| public_version(&v) != public_version(version))
        })
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
    // Follow the active vllm's DEEPEP_COMMIT_HASH — the same pin its release
    // images build from. Override with --ref main for latest.
    let (git_ref_owned, ref_source) = if let Some(r) = git_ref {
        (user_ref(repo, r), PIN_FROM_USER)
    } else {
        match resolve_versions_json_pin(config, "DEEPEP_COMMIT_HASH") {
            Some(r) => (r, PIN_FROM_VLLM),
            None => {
                eprintln!(
                    "  {} couldn't read DeepEP pin from active vllm — falling back to vvm's bundled default",
                    "warning:".yellow().bold()
                );
                (FALLBACK_DEEPEP_REF.to_string(), PIN_FROM_FALLBACK)
            }
        }
    };
    let git_ref: &str = &git_ref_owned;
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
        "{} DeepEP ({} @ {}) [pin: {}]",
        "Installing".green().bold(),
        repo.cyan(),
        git_ref.yellow(),
        ref_source.blue()
    );
    println!(
        "  {} NVSHMEM {} for {} (CUDA {})",
        "Using:".blue().bold(),
        nvshmem_ver.yellow(),
        nvshmem_subdir,
        cuda_major
    );

    // Catch a mistyped ref before anything expensive — most of all the NVSHMEM
    // download below (~350MB on a cache miss).
    ensure_ref_exists(repo, git_ref)?;

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
    use super::{
        format_close_tags, is_published_release_version, looks_like_sha, natural_order,
        parse_deepgemm_git_ref, parse_versions_json_pin, public_version, release_version_from_ref,
        resolve_remote_sha, summarize_show_config, version_to_tag, wheel_cache_dir,
    };
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

    /// Shape of vllm's real docker/versions.json (auto-generated from the
    /// Dockerfile's ARG defaults), trimmed to the keys vvm reads.
    const VERSIONS_JSON: &str = r#"{
      "_comment": "Auto-generated from Dockerfile ARGs. Do not edit manually.",
      "variable": {
        "CUDA_VERSION": { "default": "13.0.3" },
        "DEEPEP_COMMIT_HASH": { "default": "d4f41e4e93" },
        "FLASHINFER_VERSION": { "default": "0.6.15.post1" },
        "NVSHMEM_VER": { "default": "" }
      }
    }"#;

    #[test]
    fn reads_pins_from_versions_json() {
        assert_eq!(
            parse_versions_json_pin(VERSIONS_JSON, "DEEPEP_COMMIT_HASH").as_deref(),
            Some("d4f41e4e93")
        );
        assert_eq!(
            parse_versions_json_pin(VERSIONS_JSON, "FLASHINFER_VERSION").as_deref(),
            Some("0.6.15.post1")
        );
    }

    #[test]
    fn versions_json_misses_are_none_not_panics() {
        // Key absent (older vllm predating the ARG), empty default (ARG with
        // no value), and a file that isn't versions.json at all.
        assert_eq!(parse_versions_json_pin(VERSIONS_JSON, "NOPE_VERSION"), None);
        assert_eq!(parse_versions_json_pin(VERSIONS_JSON, "NVSHMEM_VER"), None);
        assert_eq!(
            parse_versions_json_pin("not json", "DEEPEP_COMMIT_HASH"),
            None
        );
        assert_eq!(parse_versions_json_pin("{}", "DEEPEP_COMMIT_HASH"), None);
    }

    #[test]
    fn version_maps_to_upstream_tag() {
        assert_eq!(version_to_tag("0.6.15.post1"), "v0.6.15.post1");
        assert_eq!(version_to_tag("0.6.8"), "v0.6.8");
        assert_eq!(version_to_tag("0.6.15rc4"), "v0.6.15rc4");
        // Already a ref — don't double-prefix.
        assert_eq!(version_to_tag("v0.6.9"), "v0.6.9");
        assert_eq!(version_to_tag("main"), "main");
    }

    #[test]
    fn digit_leading_shas_are_never_treated_as_versions() {
        // The trap: a short SHA can start with a digit just like a version.
        // Prefixing one with `v` would produce a ref that resolves to nothing.
        assert_eq!(version_to_tag("73b6ea4"), "73b6ea4");
        assert_eq!(version_to_tag("d4f41e4e93"), "d4f41e4e93");
        assert_eq!(
            version_to_tag("477618cd51baffca09c4b0b87e97c03fe827ef03"),
            "477618cd51baffca09c4b0b87e97c03fe827ef03"
        );
    }

    #[test]
    fn post_and_rc_releases_are_published_versions() {
        // Regression: an all-digits test read these as dev builds and skipped
        // the cubin/jit-cache companions. flashinfer publishes wheels for both.
        assert!(is_published_release_version("0.6.8"));
        assert!(is_published_release_version("0.6.8.post1"));
        assert!(is_published_release_version("0.6.15.post1"));
        assert!(is_published_release_version("0.6.9rc1"));
        // Genuinely unpublished: a source build's local label, or a dev cut.
        assert!(!is_published_release_version("0.6.9+g1a2b3c4"));
        assert!(!is_published_release_version("0.3.0.dev20260101"));
        assert!(!is_published_release_version(""));
    }

    #[test]
    fn release_refs_map_to_pypi_versions() {
        assert_eq!(
            release_version_from_ref("v0.6.15.post1").as_deref(),
            Some("0.6.15.post1")
        );
        assert_eq!(release_version_from_ref("0.6.7").as_deref(), Some("0.6.7"));
        assert_eq!(
            release_version_from_ref("v0.6.9rc1").as_deref(),
            Some("0.6.9rc1")
        );
        // Not a release — these can only come from a source build.
        assert_eq!(release_version_from_ref("main"), None);
        assert_eq!(release_version_from_ref("73b6ea4"), None);
        assert_eq!(
            release_version_from_ref("477618cd51baffca09c4b0b87e97c03fe827ef03"),
            None
        );
        // No dot: a branch called "v2", not a version.
        assert_eq!(release_version_from_ref("v2"), None);
    }

    #[test]
    fn local_label_does_not_make_a_companion_stale() {
        // flashinfer-jit-cache stamps the CUDA variant into a PEP 440 local
        // label; flashinfer-python and flashinfer-cubin carry none. Comparing
        // raw strings read a correctly matched pair as a mismatch and
        // uninstalled the good wheel.
        assert_eq!(public_version("0.6.17rc1+cu130"), "0.6.17rc1");
        assert_eq!(public_version("0.6.17rc1"), "0.6.17rc1");
        assert_eq!(
            public_version("0.6.17rc1+cu130"),
            public_version("0.6.17rc1")
        );
        // A genuine mismatch still reads as one.
        assert_ne!(public_version("0.6.16+cu130"), public_version("0.6.17rc1"));
        // Differing CUDA variants of the same version are still a match — the
        // variant is vvm's business, not a version skew.
        assert_eq!(
            public_version("0.6.17rc1+cu129"),
            public_version("0.6.17rc1+cu130")
        );
    }

    #[test]
    fn close_tags_prefer_newest_and_survive_multibyte_queries() {
        let tags = [
            "v0.6.7",
            "v0.6.8",
            "v0.6.9",
            "v0.6.9rc1",
            "v0.6.15",
            "v0.6.15.post1",
            "v0.6.16",
            "v0.6.17rc1",
            "v1.0.0",
        ];
        // The typo the whole feature exists for: v0.67 → the v0.6* family,
        // newest first. Lexicographic order buried v0.6.15+ below v0.6.9.
        let s = format_close_tags(&tags, "v0.67");
        let listed: Vec<&str> = s.trim_start_matches(". Close tags: ").split(", ").collect();
        assert_eq!(
            listed,
            [
                "v0.6.17rc1",
                "v0.6.16",
                "v0.6.15.post1",
                "v0.6.15",
                "v0.6.9rc1"
            ]
        );
        // Byte-slicing the query must not panic mid-char (full-width digits).
        let _ = format_close_tags(&tags, "v0.６７");
        // Nothing plausible → stay silent.
        assert_eq!(format_close_tags(&tags, "zz"), "");
        assert_eq!(format_close_tags(&[], "v0.67"), "");
    }

    #[test]
    fn natural_order_reads_digit_runs_as_numbers() {
        use std::cmp::Ordering;
        assert_eq!(natural_order("v0.6.15", "v0.6.9"), Ordering::Greater);
        assert_eq!(natural_order("v0.6.9", "v0.6.9"), Ordering::Equal);
        assert_eq!(natural_order("v0.6.9", "v0.6.9rc1"), Ordering::Less);
        assert_eq!(natural_order("v0.10.0", "v0.9.9"), Ordering::Greater);
    }

    #[test]
    fn show_config_summary_keeps_identity_lines_only() {
        // Trimmed from a real `flashinfer show-config` run: the identity
        // sections plus the noise around them (downloads, module counts).
        let dump = "\
=== Version Info ===
FlashInfer version: 0.6.17rc1
flashinfer-cubin version: 0.6.17rc1
flashinfer-jit-cache: Not installed
=== Torch Version Info ===
Torch version: 2.11.0+cu130
CUDA runtime available: Yes

=== Environment Variables ===
FLASHINFER_CACHE_DIR: /home/u/.cache/flashinfer
2026-08-01 05:56:30,315 - INFO - cubin_loader.py:114 - flashinfer.jit: File downloaded successfully
Downloaded 23090/23090 cubins
Total modules: 792
";
        let got = summarize_show_config(dump);
        assert_eq!(
            got,
            [
                "FlashInfer version: 0.6.17rc1",
                "flashinfer-cubin version: 0.6.17rc1",
                "flashinfer-jit-cache: Not installed",
                "Torch version: 2.11.0+cu130",
                "CUDA runtime available: Yes",
            ]
        );
        // Unrecognized format → empty, caller prints a plain OK instead.
        assert!(summarize_show_config("something else entirely\n").is_empty());
    }

    #[test]
    fn sha_detection_bounds() {
        assert!(looks_like_sha("73b6ea4"));
        assert!(looks_like_sha("477618cd51baffca09c4b0b87e97c03fe827ef03"));
        // Too short to be a usable abbreviation, and a real word that happens
        // to be all hex-ish must not pass.
        assert!(!looks_like_sha("abc"));
        assert!(!looks_like_sha("main"));
        assert!(!looks_like_sha("0.6.15"));
        // Over 40 chars is not an object name.
        assert!(!looks_like_sha(&"a".repeat(41)));
    }
}
