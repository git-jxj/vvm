use std::collections::BTreeSet;
use std::process::Command;

/// Supported CUDA variants for vllm wheels.
#[derive(Debug, Clone, PartialEq)]
pub enum CudaVariant {
    Cu128,
    Cu129,
    Cu130,
    Cpu,
}

impl CudaVariant {
    /// Short name used in wheel URLs and torch backend flags.
    pub fn as_str(&self) -> &str {
        match self {
            Self::Cu128 => "cu128",
            Self::Cu129 => "cu129",
            Self::Cu130 => "cu130",
            Self::Cpu => "cpu",
        }
    }

    pub fn from_str_loose(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "cu128" => Some(Self::Cu128),
            "cu129" => Some(Self::Cu129),
            "cu130" => Some(Self::Cu130),
            "cpu" => Some(Self::Cpu),
            _ => None,
        }
    }

    /// Detect the best CUDA variant for this system.
    ///
    /// Priority:
    /// 1. $VVM_CUDA_VARIANT env var
    /// 2. Detected CUDA version (nvidia-smi → version.json → nvcc)
    /// 3. Fall back to cpu
    ///
    /// Falling back to cpu — rather than guessing a CUDA variant — keeps the
    /// install honest: nothing on this machine advertises CUDA, so a GPU wheel
    /// would only fail later at import. Override with `--cuda` or
    /// `$VVM_CUDA_VARIANT` when building on a host that has no CUDA visible
    /// but will run on one that does.
    pub fn detect() -> Self {
        // 1. Env var override
        if let Ok(v) = std::env::var("VVM_CUDA_VARIANT") {
            if let Some(variant) = Self::from_str_loose(&v) {
                return variant;
            }
        }

        // 2. nvidia-smi / version.json / nvcc detection
        if let Some(cuda_ver) = detect_cuda_version() {
            return map_cuda_version(&cuda_ver);
        }

        // 3. No CUDA found anywhere
        Self::Cpu
    }
}

impl std::fmt::Display for CudaVariant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Returns true if this node has live GPU access (nvidia-smi works).
/// False on login nodes where the driver isn't accessible.
pub fn has_gpu_access() -> bool {
    Command::new("nvidia-smi")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Detect CUDA version string (e.g. "12.9") via multiple methods.
pub fn detect_cuda_version() -> Option<String> {
    // 1. nvidia-smi (works on compute nodes with GPU access)
    if let Some(v) = detect_via_nvidia_smi() {
        return Some(v);
    }

    // 2. /usr/local/cuda/version.json (works on login nodes with CUDA toolkit)
    if let Some(v) = detect_via_version_json() {
        return Some(v);
    }

    // 3. nvcc --version
    if let Some(v) = detect_via_nvcc() {
        return Some(v);
    }

    None
}

fn detect_via_nvidia_smi() -> Option<String> {
    let full_output = Command::new("nvidia-smi").output().ok()?;
    if !full_output.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&full_output.stdout);
    for line in text.lines() {
        if let Some(pos) = line.find("CUDA Version:") {
            let rest = &line[pos + 13..];
            let ver = rest.split_whitespace().next()?;
            return Some(ver.to_string());
        }
    }
    None
}

fn detect_via_version_json() -> Option<String> {
    // Try $CUDA_HOME/version.json then /usr/local/cuda/version.json
    let candidates = [
        std::env::var("CUDA_HOME")
            .ok()
            .map(|h| format!("{}/version.json", h)),
        Some("/usr/local/cuda/version.json".to_string()),
    ];
    for path in candidates.into_iter().flatten() {
        if let Ok(contents) = std::fs::read_to_string(&path) {
            if let Ok(json) = serde_json::from_str::<serde_json::Value>(&contents) {
                if let Some(ver) = json["cuda"]["version"].as_str() {
                    // "13.1.1" -> "13.1"
                    let short = ver.splitn(3, '.').take(2).collect::<Vec<_>>().join(".");
                    return Some(short);
                }
            }
        }
    }
    None
}

fn detect_via_nvcc() -> Option<String> {
    let output = Command::new("nvcc").arg("--version").output().ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&output.stdout);
    // "Cuda compilation tools, release 13.1, V13.1.115"
    for line in text.lines() {
        if let Some(pos) = line.find("release ") {
            let rest = &line[pos + 8..];
            let ver = rest.split(',').next()?.trim();
            return Some(ver.to_string());
        }
    }
    None
}

/// Detect GPU compute capabilities via nvidia-smi (e.g. "9.0", "10.0").
/// Returns a deduplicated, sorted list like ["9.0", "10.0"].
pub fn detect_gpu_arch_list() -> Option<String> {
    let output = Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&output.stdout);
    let mut archs: Vec<String> = text
        .lines()
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty())
        .collect();
    archs.sort();
    archs.dedup();
    if archs.is_empty() {
        None
    } else {
        Some(archs.join(";"))
    }
}

/// If `env_value` is a strict superset of `local`, narrow it to `local`.
///
/// Returns `None` when narrowing isn't applicable (env doesn't cover local,
/// equals local, or either side is empty/unparseable). When the env value
/// contained a `+PTX` arch anywhere, the narrowed result keeps `+PTX` on the
/// highest-numeric local arch so forward-compat behavior is preserved.
///
/// Motivating case: NGC PyTorch containers pre-export
/// `TORCH_CUDA_ARCH_LIST="8.0 8.6 9.0 10.0 11.0 12.0+PTX"`, which makes the
/// vllm `--compile` build target every arch (~6× slower). On a single-GPU
/// host of one of those archs we narrow to the local one.
pub fn narrow_arch_list_if_superset(env_value: &str, local: &str) -> Option<String> {
    let (env_set, has_ptx) = parse_arch_list(env_value);
    let (local_set, _) = parse_arch_list(local);
    if local_set.is_empty() || env_set.is_empty() {
        return None;
    }
    if !local_set.is_subset(&env_set) || local_set == env_set {
        return None;
    }
    Some(format_arch_list(&local_set, has_ptx))
}

/// True when kernels compiled for arch list `compiled` cover every arch in
/// `local` (so a binary built with `compiled` runs on the `local` GPUs).
/// Unparseable/empty lists return false — callers should then recompile.
pub fn arch_list_covers(compiled: &str, local: &str) -> bool {
    let (compiled_set, _) = parse_arch_list(compiled);
    let (local_set, _) = parse_arch_list(local);
    !compiled_set.is_empty() && !local_set.is_empty() && local_set.is_subset(&compiled_set)
}

fn parse_arch_list(s: &str) -> (BTreeSet<String>, bool) {
    let mut has_ptx = false;
    let archs: BTreeSet<String> = s
        .split(|c: char| c.is_whitespace() || c == ';' || c == ',')
        .filter(|t| !t.is_empty())
        .map(|t| {
            if let Some(stripped) = t.strip_suffix("+PTX") {
                has_ptx = true;
                stripped.to_string()
            } else {
                t.to_string()
            }
        })
        .collect();
    (archs, has_ptx)
}

fn format_arch_list(archs: &BTreeSet<String>, with_ptx: bool) -> String {
    let mut sorted: Vec<&String> = archs.iter().collect();
    sorted.sort_by_key(|a| {
        let mut parts = a.split('.');
        let major: u32 = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0);
        let minor: u32 = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0);
        (major, minor)
    });
    if sorted.is_empty() {
        return String::new();
    }
    let last_idx = sorted.len() - 1;
    sorted
        .iter()
        .enumerate()
        .map(|(i, a)| {
            if with_ptx && i == last_idx {
                format!("{}+PTX", a)
            } else {
                a.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Map a CUDA version string like "12.8" to the nearest supported variant.
///
/// A string we can't parse a major version out of counts as "no CUDA
/// identified" and maps to cpu — guessing a variant here would install a GPU
/// wheel on evidence we don't actually have.
fn map_cuda_version(ver: &str) -> CudaVariant {
    let parts: Vec<&str> = ver.split('.').collect();
    let Some(major) = parts.first().and_then(|s| s.parse::<u32>().ok()) else {
        return CudaVariant::Cpu;
    };
    let minor: u32 = parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(0);

    match (major, minor) {
        (12, m) if m >= 9 => CudaVariant::Cu129,
        (12, _) => CudaVariant::Cu128,
        // CUDA 13, future majors, and anything older than 12 all land on the
        // newest supported variant. For a future major that's the closest
        // thing we have; for CUDA 11 nothing would work anyway (vllm ships no
        // cu11x wheel), so failing against the newest is the honest outcome.
        _ => CudaVariant::Cu130,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn narrow_ngc_default_to_single_arch_keeps_ptx() {
        let env = "8.0 8.6 9.0 10.0 11.0 12.0+PTX";
        let got = narrow_arch_list_if_superset(env, "10.0").unwrap();
        assert_eq!(got, "10.0+PTX");
    }

    #[test]
    fn narrow_ngc_default_without_ptx() {
        let env = "8.0 8.6 9.0 10.0 11.0 12.0";
        let got = narrow_arch_list_if_superset(env, "10.0").unwrap();
        assert_eq!(got, "10.0");
    }

    #[test]
    fn narrow_multi_local_arch() {
        let env = "8.0 8.6 9.0 10.0 11.0 12.0+PTX";
        let got = narrow_arch_list_if_superset(env, "9.0;10.0").unwrap();
        assert_eq!(got, "9.0 10.0+PTX");
    }

    #[test]
    fn narrow_returns_none_when_equal() {
        assert!(narrow_arch_list_if_superset("10.0", "10.0").is_none());
        assert!(narrow_arch_list_if_superset("10.0+PTX", "10.0").is_none());
    }

    #[test]
    fn narrow_returns_none_when_local_not_in_env() {
        // user explicitly cross-compiles for an older arch — respect it.
        assert!(narrow_arch_list_if_superset("8.0", "10.0").is_none());
        // env covers some local archs but not all — bail out, can't safely narrow.
        assert!(narrow_arch_list_if_superset("8.0 9.0", "9.0;10.0").is_none());
    }

    #[test]
    fn narrow_returns_none_on_empty() {
        assert!(narrow_arch_list_if_superset("", "10.0").is_none());
        assert!(narrow_arch_list_if_superset("10.0", "").is_none());
    }

    #[test]
    fn arch_list_covers_subset_and_separators() {
        assert!(arch_list_covers("9.0", "9.0"));
        assert!(arch_list_covers("8.0 9.0 10.0", "9.0"));
        assert!(arch_list_covers("8.0;9.0", "8.0,9.0"));
        // +PTX suffix is ignored for coverage purposes.
        assert!(arch_list_covers("9.0+PTX", "9.0"));
        // compiled list missing a local arch → not covered.
        assert!(!arch_list_covers("8.0", "9.0"));
        assert!(!arch_list_covers("8.0 9.0", "9.0;10.0"));
        // unparseable / empty → not covered.
        assert!(!arch_list_covers("", "9.0"));
        assert!(!arch_list_covers("9.0", ""));
    }

    #[test]
    fn unidentifiable_cuda_version_maps_to_cpu() {
        // No parseable major version — we never actually identified CUDA.
        assert_eq!(map_cuda_version(""), CudaVariant::Cpu);
        assert_eq!(map_cuda_version("unknown"), CudaVariant::Cpu);
        assert_eq!(map_cuda_version("N/A"), CudaVariant::Cpu);
        assert_eq!(map_cuda_version("Not Supported"), CudaVariant::Cpu);
    }

    #[test]
    fn known_cuda_versions_still_map_to_gpu_variants() {
        assert_eq!(map_cuda_version("13.1"), CudaVariant::Cu130);
        assert_eq!(map_cuda_version("12.9"), CudaVariant::Cu129);
        assert_eq!(map_cuda_version("12.10"), CudaVariant::Cu129);
        assert_eq!(map_cuda_version("12.8"), CudaVariant::Cu128);
        // Bare major with no minor is still an identification.
        assert_eq!(map_cuda_version("13"), CudaVariant::Cu130);
    }

    #[test]
    fn unmapped_cuda_majors_fall_back_to_newest_variant() {
        // Future majors get the newest variant we know about, never an older
        // one than CUDA 13 receives.
        assert_eq!(map_cuda_version("14.0"), CudaVariant::Cu130);
        assert_eq!(map_cuda_version("99.9"), CudaVariant::Cu130);
        // Pre-12 majors have no matching vllm wheel at all; newest is as good
        // a target as any.
        assert_eq!(map_cuda_version("11.8"), CudaVariant::Cu130);
    }

    #[test]
    fn detect_falls_back_to_cpu_when_env_override_is_absent_and_bogus() {
        // A bogus $VVM_CUDA_VARIANT is ignored, not fatal; detection continues
        // and — on a host with no CUDA — ends at cpu rather than guessing.
        assert!(CudaVariant::from_str_loose("cu999").is_none());
        assert_eq!(CudaVariant::from_str_loose("cpu"), Some(CudaVariant::Cpu));
    }

    #[test]
    fn narrow_handles_semicolon_and_comma_separators() {
        let got = narrow_arch_list_if_superset("8.0;9.0;10.0", "10.0").unwrap();
        assert_eq!(got, "10.0");
        let got = narrow_arch_list_if_superset("8.0,9.0,10.0", "10.0").unwrap();
        assert_eq!(got, "10.0");
    }
}
