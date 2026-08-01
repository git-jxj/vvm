use serde::{Deserialize, Serialize};
use std::path::Path;

/// Metadata stored alongside each installed vllm version.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionMetadata {
    /// Install type: "release", "commit", "branch", "pr", "nightly"
    #[serde(rename = "type")]
    pub install_type: String,
    /// Display name (e.g. "v0.18.0", "commit-abc123", "pr-1234")
    pub name: String,
    /// Exact version or commit hash
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
    /// Full commit hash (for commit/branch/pr installs)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub commit: Option<String>,
    /// PR number (for pr installs)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pr: Option<u64>,
    /// Branch name (for branch installs)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub branch: Option<String>,
    /// Source path for local repo installs (--path)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_path: Option<String>,
    /// CUDA variant used
    pub cuda: String,
    /// ISO timestamp of installation
    pub installed_at: String,
    /// TORCH_CUDA_ARCH_LIST the kernels were built for ("all" = no restriction,
    /// i.e. vllm's full default arch set). Only set for local-compiled installs
    /// whose in-tree kernels are safe to reuse (clean csrc/cmake worktree).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub arch_list: Option<String>,
}

impl VersionMetadata {
    pub fn save(&self, dir: &Path) -> Result<(), std::io::Error> {
        let path = dir.join("metadata.json");
        let json = serde_json::to_string_pretty(self).map_err(std::io::Error::other)?;
        std::fs::write(path, json)
    }

    pub fn load(dir: &Path) -> Result<Self, std::io::Error> {
        let path = dir.join("metadata.json");
        let json = std::fs::read_to_string(path)?;
        serde_json::from_str(&json).map_err(std::io::Error::other)
    }
}

#[cfg(test)]
mod tests {
    use super::VersionMetadata;

    #[test]
    fn deserializes_metadata_written_before_arch_list_existed() {
        let json = r#"{
            "type": "local-compiled",
            "name": "local-dev-abc12345",
            "commit": "abc1234500000000000000000000000000000000",
            "source_path": "/home/u/mke",
            "cuda": "cu130",
            "installed_at": "2026-01-01T00:00:00Z"
        }"#;
        let meta: VersionMetadata = serde_json::from_str(json).unwrap();
        assert_eq!(meta.arch_list, None);
    }
}
