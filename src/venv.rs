use std::path::Path;
use std::process::Command;

use crate::errors::VvmError;

/// Create a Python virtual environment at the given path using `uv venv`.
pub fn create_venv(path: &Path, python: Option<&str>) -> Result<(), VvmError> {
    let path_str = path.to_string_lossy().into_owned();
    let mut args = vec!["venv".to_string(), path_str];
    let python_ver = python.unwrap_or("3.12");
    args.push("--python".to_string());
    args.push(python_ver.to_string());

    let verbose = std::env::var("VVM_VERBOSE").is_ok();
    let mut command = Command::new("uv");
    command.args(&args);
    if !verbose {
        command.stdout(std::process::Stdio::null());
        command.stderr(std::process::Stdio::null());
    }

    let status = command.status().map_err(|e| VvmError::CommandFailed {
        cmd: "uv venv".to_string(),
        detail: e.to_string(),
    })?;

    if !status.success() {
        return Err(VvmError::VenvCreationFailed(path.display().to_string()));
    }
    Ok(())
}

/// Run a command inside the given venv (prepends venv/bin to PATH).
pub fn run_in_venv(venv_path: &Path, cmd: &str, args: &[&str]) -> Result<(), VvmError> {
    run_in_venv_with_env(venv_path, cmd, args, &[])
}

/// Run a command inside the given venv with extra environment variables.
/// For `uv` commands, stdout is suppressed (hides verbose dependency lists).
/// Stderr is kept for progress bars and errors.
pub fn run_in_venv_with_env(
    venv_path: &Path,
    cmd: &str,
    args: &[&str],
    extra_env: &[(&str, &str)],
) -> Result<(), VvmError> {
    let bin_dir = venv_path.join("bin");
    let current_path = std::env::var("PATH").unwrap_or_default();
    let new_path = format!("{}:{}", bin_dir.display(), current_path);

    let mut command = Command::new(cmd);
    command
        .args(args)
        .env("PATH", &new_path)
        .env("VIRTUAL_ENV", venv_path);

    // For uv/pip: capture output unless --verbose
    let verbose = std::env::var("VVM_VERBOSE").is_ok();
    let quiet = !verbose && (cmd == "uv" || cmd == "pip");
    if quiet {
        command.stdout(std::process::Stdio::null());
        command.stderr(std::process::Stdio::piped());
    }

    for (key, value) in extra_env {
        command.env(key, value);
    }

    if quiet {
        let output = command.output().map_err(|e| VvmError::CommandFailed {
            cmd: cmd.to_string(),
            detail: e.to_string(),
        })?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(VvmError::CommandFailed {
                cmd: format!("{} {}", cmd, args.join(" ")),
                detail: format!("{}", stderr),
            });
        }
    } else {
        let status = command.status().map_err(|e| VvmError::CommandFailed {
            cmd: cmd.to_string(),
            detail: e.to_string(),
        })?;
        if !status.success() {
            return Err(VvmError::CommandFailed {
                cmd: format!("{} {}", cmd, args.join(" ")),
                detail: format!("exit code: {:?}", status.code()),
            });
        }
    }
    Ok(())
}

/// Check if `uv` is available on the system.
pub fn ensure_uv() -> Result<String, VvmError> {
    // Check system uv first
    if let Ok(output) = Command::new("uv").arg("--version").output() {
        if output.status.success() {
            return Ok("uv".to_string());
        }
    }
    Err(VvmError::UvNotFound)
}
