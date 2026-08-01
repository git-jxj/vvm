use crate::config::Config;
use crate::errors::VvmError;

/// Run `uv pip <args>` in the current version's venv.
pub fn run(config: &Config, args: &[String]) -> Result<(), VvmError> {
    let link = config.current_link();
    if !link.is_symlink() {
        return Err(VvmError::NoCurrentVersion);
    }

    let target = std::fs::read_link(&link)?;
    let venv_path = target.join("venv");

    if !venv_path.exists() {
        return Err(VvmError::NoCurrentVersion);
    }

    let bin_dir = venv_path.join("bin");
    let current_path = std::env::var("PATH").unwrap_or_default();
    let new_path = format!("{}:{}", bin_dir.display(), current_path);

    let args_str: Vec<&str> = std::iter::once("pip")
        .chain(args.iter().map(|s| s.as_str()))
        .collect();

    let status = std::process::Command::new("uv")
        .args(&args_str)
        .env("PATH", &new_path)
        .env("VIRTUAL_ENV", &venv_path)
        .status()
        .map_err(|e| VvmError::CommandFailed {
            cmd: "uv pip".to_string(),
            detail: e.to_string(),
        })?;

    if !status.success() {
        return Err(VvmError::CommandFailed {
            cmd: format!("uv pip {}", args.join(" ")),
            detail: format!("exit code: {:?}", status.code()),
        });
    }

    Ok(())
}
