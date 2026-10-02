use std::path::PathBuf;

/// Global configuration for vvm.
#[derive(Debug, Clone)]
pub struct Config {
    /// Root directory for vvm data (~/.vvm or $VVM_DIR)
    pub base_dir: PathBuf,
}

impl Config {
    pub fn load() -> Self {
        let base_dir = std::env::var("VVM_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                dirs::home_dir()
                    .expect("cannot determine home directory")
                    .join(".vvm")
            });
        // Symlink targets and shell activation paths must remain valid after
        // changing directories, including when VVM_DIR is a relative path.
        let base_dir = if base_dir.is_relative() {
            std::env::current_dir()
                .expect("cannot determine current directory")
                .join(base_dir)
        } else {
            base_dir
        };
        Self { base_dir }
    }

    /// Directory containing all installed versions
    pub fn versions_dir(&self) -> PathBuf {
        self.base_dir.join("versions")
    }

    /// Directory containing alias symlinks
    pub fn aliases_dir(&self) -> PathBuf {
        self.base_dir.join("aliases")
    }

    /// Symlink pointing to the currently active version
    pub fn current_link(&self) -> PathBuf {
        self.base_dir.join("current")
    }

    /// Directory for cached git repo clones (used by --repo)
    pub fn repos_dir(&self) -> PathBuf {
        self.base_dir.join("repos")
    }

    /// Directory for cached downloads (NVSHMEM tarballs, etc).
    /// Keyed by artifact so reinstalls don't re-pull ~350MB each time.
    pub fn cache_dir(&self) -> PathBuf {
        self.base_dir.join("cache")
    }

    /// Ensure all required directories exist
    pub fn ensure_dirs(&self) -> std::io::Result<()> {
        std::fs::create_dir_all(self.versions_dir())?;
        std::fs::create_dir_all(self.aliases_dir())?;
        Ok(())
    }
}
