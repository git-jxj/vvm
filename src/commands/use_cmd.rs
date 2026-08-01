use colored::Colorize;
use std::path::PathBuf;

use crate::config::Config;
use crate::errors::VvmError;
use crate::resolve::VersionSpecifier;

pub fn run(config: &Config, specifier_str: Option<&str>) -> Result<(), VvmError> {
    let specifier_str = match specifier_str {
        Some(s) => s.to_string(),
        None => read_vvmrc()?,
    };

    // Try the raw string as a directory name first (for local-*, etc.)
    let (dir_name, version_dir) = {
        let raw_dir = resolve_version_dir(config, &specifier_str)?;
        if raw_dir.exists() {
            (specifier_str.clone(), raw_dir)
        } else {
            // Fall back to specifier parsing (adds v prefix for releases, etc.)
            let specifier = VersionSpecifier::parse(&specifier_str)?;
            let parsed_name = specifier.dir_name();
            let parsed_dir = resolve_version_dir(config, &parsed_name)?;
            if !parsed_dir.exists() {
                return Err(VvmError::VersionNotFound(format!(
                    "'{}' is not installed. Run: vvm install {}",
                    parsed_name, specifier_str
                )));
            }
            (parsed_name, parsed_dir)
        }
    };

    // Update the "current" symlink atomically via rename (same as install.rs).
    // Avoids the remove+create window where current doesn't exist, and handles
    // the case where the link was already removed by a concurrent process.
    let current_link = config.current_link();
    let tmp_link = config
        .base_dir
        .join(format!(".current.tmp.{}", std::process::id()));
    let _ = std::fs::remove_file(&tmp_link);
    std::os::unix::fs::symlink(&version_dir, &tmp_link).map_err(VvmError::Io)?;
    std::fs::rename(&tmp_link, &current_link).map_err(VvmError::Io)?;

    println!("{} Using {}", "✓".green().bold(), dir_name.cyan());

    // Check if the shell wrapper is active (it sets VVM_DIR after eval).
    // If not, the user called the binary directly — PATH won't be updated
    // automatically, so print an actionable hint.
    if std::env::var("VVM_DIR").is_err() {
        eprintln!(
            "\n{} Run this to activate in your current shell:\n  eval \"$(vvm env)\"\n\n\
             Or add to your shell profile for automatic activation:\n  \
             eval \"$(vvm init bash)\"",
            "Hint:".yellow().bold()
        );
    }

    Ok(())
}

fn resolve_version_dir(config: &Config, dir_name: &str) -> Result<PathBuf, VvmError> {
    // Try direct version first
    let direct = config.versions_dir().join(dir_name);
    if direct.exists() {
        return Ok(direct);
    }

    // Try alias
    let alias = config.aliases_dir().join(dir_name);
    if alias.exists() {
        let target = std::fs::read_link(&alias)?;
        if target.is_absolute() {
            return Ok(target);
        }
        // Resolve relative symlink from aliases dir
        return Ok(config.aliases_dir().join(target));
    }

    Ok(direct)
}

fn read_vvmrc() -> Result<String, VvmError> {
    let cwd = std::env::current_dir().map_err(VvmError::Io)?;
    let mut dir = cwd.as_path();

    loop {
        let rc = dir.join(".vvmrc");
        if rc.exists() {
            let content = std::fs::read_to_string(&rc).map_err(VvmError::Io)?;
            let trimmed = content.trim().to_string();
            if !trimmed.is_empty() {
                return Ok(trimmed);
            }
        }
        match dir.parent() {
            Some(parent) => dir = parent,
            None => break,
        }
    }

    Err(VvmError::InvalidSpecifier(
        "No version specified and no .vvmrc found".to_string(),
    ))
}
