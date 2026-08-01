# AGENTS.md

## Project Overview

vvm (vLLM Version Manager) is a Rust CLI tool that manages multiple vLLM installations in isolated Python virtual environments. Inspired by nvm/fnm for Node.js. Each version gets its own venv under `~/.vvm/versions/`.

## Build & Test

```bash
# Build release binary
cargo build --release

# Run all tests
cargo test

# Lint (CI runs this with -D warnings)
cargo clippy --all-targets --all-features -- -D warnings

# Format + manifest sort (CI enforces both)
cargo fmt --all
cargo sort -w

# Build + install to ~/.local/bin + configure shell integration
./install.sh
```

Install the pre-commit hook once per clone so unformatted code never reaches CI:

```bash
git config core.hooksPath .githooks
```

## Architecture

~7,600 lines of Rust across 24 files. Single binary, no runtime dependencies beyond `uv` and `python3`.

### Source Layout

- **`src/main.rs`** — CLI entry point. Clap derive-based command dispatch (14 subcommands + 1 hidden). `Cli` struct must be `pub` for clap_complete in init.rs. `Uninstall` takes `Vec<String>` for multi-version support. `exec_with_version` resolves `pip`/`pip3` to `uv pip` and falls back to `python -m {cmd}` for commands absent from `venv/bin`.
- **`src/config.rs`** — `Config` struct managing `$VVM_DIR` paths (versions/, aliases/, repos/, cache/, current symlink).
- **`src/errors.rs`** — `VvmError` enum via thiserror. All commands return `Result<(), VvmError>`.
- **`src/cuda.rs`** — CUDA variant detection and `CudaVariant` enum (cu128/cu129/cu130/cpu). Detection order: `$VVM_CUDA_VARIANT` → nvidia-smi → `/usr/local/cuda/version.json` (or `$CUDA_HOME`) → `nvcc --version` → **cpu** when none of them report CUDA. A detected-but-unmapped major (CUDA 11 and older, or a future 14+) maps to the newest variant, cu130. `has_gpu_access()` checks nvidia-smi for login-node detection. `detect_gpu_arch_list()` queries nvidia-smi for compute capabilities (used by `--compile` to set `TORCH_CUDA_ARCH_LIST`).
- **`src/venv.rs`** — Python venv creation via `uv venv` and command execution. `run_in_venv_with_env` for passing extra env vars (used by `--path` installs).
- **`src/fsutil.rs`** — NFS-tolerant directory removal. `std::fs::remove_dir_all` can fail with ENOTEMPTY on NFS even after every entry is unlinked (stale attribute caches, `.nfs*` silly-rename files from another client). Renames the tree to a hidden sibling first — one atomic rename frees the caller's path immediately — then removes with retries.
- **`src/metadata.rs`** — `VersionMetadata` (serde JSON) stored as `metadata.json` per installed version. Fields: type, name, version, commit, pr, branch, source_path, cuda, installed_at, arch_list (TORCH_CUDA_ARCH_LIST of compiled kernels; local-compiled installs only).
- **`src/github.rs`** — GitHub API client (reqwest blocking). PR→commit, branch→commit, releases list, recent commits (by branch or with `until` timestamp), commit SHA resolution. Token resolved from `$GITHUB_TOKEN` → `$GH_TOKEN` → `gh auth token`, cached per process.
- **`src/resolve/specifier.rs`** — `VersionSpecifier` parser: release, commit:, branch:, pr:, nightly. `dir_name()` sanitizes `/` → `-` in branch names. Rejects refs that would escape the versions directory when interpolated into a path.
- **`src/commands/install.rs`** — Core install logic. Install paths: `install_from_wheel` (release), `install_nightly_with_fallback` (nightly), `install_branch_with_fallback` (branch), `install_from_local` (--path), `install_from_local_compiled` (--path --compile), `install_from_source` (--source), `install_from_pr` (pr). `find_wheel_url` fetches metadata.json for direct wheel URL. `verify_installation` skipped on login nodes. `install_from_local_compiled` auto-detects GPU arch, clears stale CMake cache, and saves build log. Also holds the cross-node install lock and `--repo` clone/fetch caching.
- **`src/commands/install_extra.rs`** — Bundled recipes for auxiliary libraries that can't install via plain pip: `deepgemm` (pin read from the active vllm's `tools/install_deepgemm.sh`), `flashinfer` (source build + matching cubin/jit-cache companions, or prebuilt nightly wheels), `deepep` (NVSHMEM download + build). Owns the wheel cache under `~/.vvm/cache/wheels/`.
- **`src/commands/use_cmd.rs`** — Updates `current` symlink. Tries raw dir name first (for local-* names), then parses as specifier. Reads `.vvmrc` with upward directory walk. Hints user to `eval "$(vvm env)"` if shell wrapper not active.
- **`src/commands/env.rs`** — Outputs shell-evaluable `export` statements for PATH/VIRTUAL_ENV/VVM_DIR. Strips previous vvm bin from PATH before prepending new one.
- **`src/commands/init.rs`** — Generates shell wrapper function + tab completions (via clap_complete) for bash/zsh/fish. Dynamic completion for `use`/`uninstall`/`exec` via hidden `list-installed` subcommand.
- **`src/commands/alias.rs`** — Create/list/show/remove aliases (symlinks in aliases/ dir).
- **`src/commands/ls.rs`** — Lists installed versions with version string, branch name, source_path, and alias info. `*` marker uses symlink target path comparison (handles branch names with slashes). `--json` emits machine-readable output.
- **`src/commands/current.rs`** — Prints the active version name from the `current` symlink.
- **`src/commands/info.rs`** — Detailed metadata for one version (commit SHA, CUDA variant, source path, install time). `--json` for scripting.
- **`src/commands/pip.rs`** — Runs `uv pip <args>` inside the active version's venv.
- **`src/commands/ls_remote.rs`** — Fetches and displays GitHub releases.
- **`src/commands/uninstall.rs`** — Removes version directory and cleans up aliases. Called per-specifier (main.rs loops for multi-version support).
- **`tests/concurrent_install.rs`** — Integration tests for race-free `current` symlink and `default` alias updates (atomic symlink swap vs. the old remove-then-create pattern).
- **`install.sh`** — Build + install binary + add `eval "$(vvm init bash)"` to shell profile.

### Key Design Decisions

- **uv, not pip**: All installs use `uv pip install` with `--torch-backend={variant}`. Venvs created with `uv venv`.
- **Direct wheel URL for commit/branch/nightly**: Fetches `metadata.json` from wheels.vllm.ai to get exact wheel URL, bypassing uv version resolution. Necessary because wheels have PEP 440 local version labels (e.g. `+g65b2f405d.cu130`) that uv ignores when resolving by package name.
- **Index strategy for release CUDA variants**: Uses `--extra-index-url` (wheels.vllm.ai) + `--index-strategy unsafe-best-match`. This prevents uv from silently falling back to PyPI's default cu129 wheel.
- **Release fallback + nonexistent-release fail-fast**: When the release index lacks a wheel for the variant, falls back to the commit-based wheel at the tag's commit (SHA resolved up-front in `resolve_commit` via `github::tag_commit_sha`). If the tag doesn't exist (GitHub 422/404), the install fails fast — before venv creation — with the latest release and a `vvm install nightly` hint when the requested version is newer. Transient GitHub failures stay non-fatal.
- **Branch/nightly install fallback**: If HEAD commit has no wheel yet, walks back up to 30 recent commits on that branch (or main) to find one with a precompiled wheel on wheels.vllm.ai.
- **Merge-base wheel matching for `--path` installs**: A fork typically carries local commits on top of an older `main` base, so the borrowed precompiled `_C.so` must match that **base**, not HEAD. `install_from_local` computes `merge_base_with_main` (`git merge-base HEAD <ref>` over `upstream/main`/`origin/main`/`main`) and uses the base's own wheel, or — if that commit has no wheel yet — walks back recent main commits no newer than the **base's** committer date (not HEAD's, which may be months of fork commits ahead). If `csrc/`/`cmake/`/`CMakeLists.txt` changed between the wheel commit and HEAD (`compiled_drift_since`), it defaults to compiling: `-y` compiles, an interactive prompt defaults to Yes, and a non-interactive run without `-y` errors (rather than silently installing a wheel missing the fork's kernels).
- **Post-install variant verification**: After install, imports every top-level compiled extension actually present (`vllm/*.abi3.so` — names vary by version: `vllm._C` for ≤0.23, `vllm._C_stable_libtorch` for ≥0.24) to catch CUDA library mismatches and duplicate op registration from stale binaries. Rolls back on failure. Skipped on login nodes (`has_gpu_access()` returns false) to avoid false `libcudart.so` errors.
- **Wheel cache for `install-extra` source builds**: flashinfer built from source (`--ref`/`--repo`) is compiled via `python -m build --wheel -n -x` and the wheel cached under `~/.vvm/cache/wheels/flashinfer/{sha}-{cuda}-{archs}/`. The ref is resolved to a SHA via `git ls-remote` before cloning, so a cache hit installs in seconds with no clone. flashinfer wheels are py3-none-any (kernels are cubin/JIT data, no torch-ABI binding), so one build serves every version's venv. `VVM_NO_WHEEL_CACHE=1` bypasses.
- **Stale flashinfer companion cleanup**: flashinfer refuses to import when flashinfer-cubin's version mismatches flashinfer-python's. When matching cubin/jit-cache wheels can't be installed (dev/fork versions, CPU, resolver failure), mismatched leftovers are uninstalled; version probing uses `importlib.metadata`, not `import flashinfer`, since the mismatch breaks the import itself.
- **Cross-node install lock**: Version directories often live on a shared filesystem (NFS/Lustre) where several nodes install concurrently. The lock file records `hostname:pid`, so a *local* pid can be checked against `/proc` while a remote owner is judged only by an mtime heartbeat — a remote pid number means nothing on this node and must never be treated as dead.
- **Branch name sanitization**: `/` in branch names is replaced with `-` in `dir_name()` to avoid nested directory structures.
- **Shell integration via eval**: `vvm init bash` outputs shell wrapper + clap_complete completions. The wrapper evals `vvm env` after `use`/`install`. When not wrapped, `vvm use` prints a hint.
- **Symlink for current version**: `~/.vvm/current` → `~/.vvm/versions/{name}`. The `env` command reads this. Updates go through an atomic rename (symlink-and-swap) rather than remove-then-create, so a concurrent reader never observes a missing link. `vvm ls` compares symlink target path with `starts_with` to handle any nested paths.
- **`--compile` mode**: Editable source build (`uv pip install -e repo_path --no-build-isolation-package vllm --torch-backend={variant}`). Editable so `.py` edits take effect live (consistent with the precompiled `--path` mode); the compiled `_C.abi3.so` lands in the source tree, so rebuilding kernels means re-running `--compile`, and multiple `--compile` installs of the same repo path share that one binary (no side-by-side snapshots). Auto-sets `TORCH_CUDA_ARCH_LIST` from `nvidia-smi --query-gpu=compute_cap`. Clears stale CMake cache (`build/`) before compiling — uv build isolation caches ninja paths in `build/temp.*/CMakeCache.txt` that break across runs. Also deletes all `vllm/**/*.abi3.so` in the tree before compiling — binaries left by previous installs (wheel extractions from precompiled `--path` installs, builds of other commits) can outlive the extension targets the current commit produces (e.g. `_C` → `_C_stable_libtorch`) and double-register torch ops at import. Saves build log to `build.log` in version dir. Does NOT override `MAX_JOBS`/`NVCC_THREADS` — vllm's setup.py defaults (all CPUs, NVCC_THREADS=1) are optimal.
- **Kernel reuse for repeat `--compile` installs**: Before compiling, `try_reuse_compiled_kernels` looks for the most recent `local-compiled` install of the same repo (canonicalized source_path + same CUDA variant, `arch_list` recorded in metadata). If nothing feeding the compiled extensions changed since that build — `git diff <prev-commit> -- csrc cmake CMakeLists.txt` against the *worktree* is empty (uncommitted edits count), the pyproject torch pin is unchanged (`git show <commit>:pyproject.toml`), the recorded arch list covers the local GPU (`arch_list_covers`; "all" = unrestricted build), and `vllm/_C.abi3.so` is still in the tree — it installs editable with `VLLM_USE_PRECOMPILED=1` + `VLLM_PRECOMPILED_WHEEL_LOCATION` pointing at an empty stub zip: setup.py skips build_ext, extracting zero members is a no-op, and the kernels already in the shared source tree are used. Any check failing (or `VVM_FORCE_COMPILE=1`) falls back to a real compile; reused kernels failing post-install verification auto-recompile without prompting. Compiles with a kernel-dirty worktree don't record `arch_list` (commit wouldn't identify what was built), so they're never reused.
- **gh as a git credential helper**: For `https://github.com/...` clones with `gh` authenticated, vvm passes `-c credential.helper=!gh auth git-credential` (after resetting inherited helpers) so private repos work without prompting on `/dev/tty`. The token is never interpolated into a URL or an argv.

### Install Flow (commands/install.rs)

1. Parse specifier → resolve commit hash if needed (GitHub API for branch/PR; short hashes expanded to full 40-char SHA)
2. Detect CUDA variant (`--cuda` > `$VVM_CUDA_VARIANT` > nvidia-smi > version.json > nvcc > cpu). Auto-detecting cpu prints a `Note:` — a silently CPU-only vllm otherwise surfaces much later, as a model that won't load.
3. Create venv at `$VVM_DIR/versions/{name}/venv` via `uv venv`
4. Install vllm via one of: `install_from_wheel`, `install_nightly_with_fallback`, `install_branch_with_fallback`, `install_from_local`, `install_from_local_compiled`, `install_from_source`, `install_from_pr`
5. Verify installed variant (skip on login node); capture `vllm.__version__`
6. Write `metadata.json` (includes version string, branch name, source_path)
7. Set `default` alias if first install; update `current` symlink

### Version Specifier Resolution + Directory Names

- `0.18.0` → `Release("0.18.0")` → dir `v0.18.0`
- `commit:abc123` → `Commit("abc123")` → dir `commit-abc123de` (short hash expanded via API)
- `branch:main` → `Branch("main")` → GitHub API → dir `branch-main`
- `pr:1234` → `Pr(1234)` → GitHub API → dir `pr-1234`
- `nightly` → `Nightly` → dir `nightly`
- `--path ~/vllm` → derived from git state → dir `local-{branch}-{short_hash}`

`vvm use` and `vvm uninstall` try the raw string as a directory name first (for `local-*` etc.), then fall back to specifier parsing.

## Conventions

- All public functions in command modules: `pub fn run(config, ...) -> Result<(), VvmError>`
- Colored output: green for success, cyan for version names, yellow for CUDA variants/branch names, blue for info
- User-facing errors go through `VvmError` — displayed with `error: {message}` format in main.rs
- Failed installs clean up partial venv directories before returning error
- Commits follow [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `perf:`, `refactor:`, `test:`, `doc:`, `chore:`) — `cliff.toml` groups them into the release changelog
- All commits must carry a `Signed-off-by` line (`git commit -s`) per the [DCO](DCO)
