use thiserror::Error;

#[derive(Error, Debug)]
pub enum VvmError {
    #[error("Failed to create virtual environment at {0}")]
    VenvCreationFailed(String),

    #[error(
        "uv is not installed. Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    )]
    UvNotFound,

    #[error("Command failed: {cmd}: {detail}")]
    CommandFailed { cmd: String, detail: String },

    #[error("Version not found: {0}")]
    VersionNotFound(String),

    #[error("No current version set. Use `vvm use <version>` first.")]
    NoCurrentVersion,

    #[error("GitHub API error: {0}")]
    GitHubApi(String),

    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Invalid version specifier: {0}")]
    InvalidSpecifier(String),

    #[error("Unsupported CUDA variant: {0}")]
    UnsupportedVariant(String),
}
