use colored::Colorize;

use crate::config::Config;
use crate::errors::VvmError;

/// Create an alias, or list all aliases if no arguments given.
pub fn run(config: &Config, name: Option<&str>, target: Option<&str>) -> Result<(), VvmError> {
    config.ensure_dirs()?;

    match (name, target) {
        (Some(name), Some(target)) => create(config, name, target),
        (Some(name), None) => show(config, name),
        (None, _) => list(config),
    }
}

/// Remove an alias.
pub fn remove(config: &Config, name: &str) -> Result<(), VvmError> {
    let alias_path = config.aliases_dir().join(name);
    if !alias_path.exists() && !alias_path.is_symlink() {
        return Err(VvmError::VersionNotFound(format!("alias '{}'", name)));
    }
    std::fs::remove_file(&alias_path)?;
    println!("{} Removed alias '{}'", "✓".green().bold(), name);
    Ok(())
}

fn create(config: &Config, name: &str, target: &str) -> Result<(), VvmError> {
    // Resolve target: try as direct dir name first, then as specifier
    let version_dir = {
        let direct = config.versions_dir().join(target);
        if direct.exists() {
            direct
        } else {
            let spec = crate::resolve::VersionSpecifier::parse(target)?;
            let parsed = config.versions_dir().join(spec.dir_name());
            if !parsed.exists() {
                return Err(VvmError::VersionNotFound(format!(
                    "'{}' is not installed",
                    target
                )));
            }
            parsed
        }
    };

    let alias_path = config.aliases_dir().join(name);

    // Remove existing alias if present
    if alias_path.exists() || alias_path.is_symlink() {
        std::fs::remove_file(&alias_path)?;
    }

    std::os::unix::fs::symlink(&version_dir, &alias_path)?;

    println!(
        "{} {} -> {}",
        "✓".green().bold(),
        name.cyan(),
        version_dir
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .yellow()
    );
    Ok(())
}

fn show(config: &Config, name: &str) -> Result<(), VvmError> {
    let alias_path = config.aliases_dir().join(name);
    if !alias_path.exists() && !alias_path.is_symlink() {
        return Err(VvmError::VersionNotFound(format!("alias '{}'", name)));
    }
    let target = std::fs::read_link(&alias_path)?;
    let target_name = target
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();
    println!("{} -> {}", name.cyan(), target_name.yellow());
    Ok(())
}

fn list(config: &Config) -> Result<(), VvmError> {
    let aliases_dir = config.aliases_dir();
    if !aliases_dir.exists() {
        println!("No aliases defined.");
        return Ok(());
    }

    let mut entries: Vec<_> = std::fs::read_dir(&aliases_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_symlink())
        .collect();

    if entries.is_empty() {
        println!("No aliases defined.");
        return Ok(());
    }

    entries.sort_by_key(|e| e.file_name());

    for entry in entries {
        let name = entry.file_name().to_string_lossy().to_string();
        let target = std::fs::read_link(entry.path())
            .ok()
            .and_then(|t| t.file_name().map(|f| f.to_string_lossy().to_string()))
            .unwrap_or_else(|| "???".to_string());
        println!("{} -> {}", name.cyan(), target.yellow());
    }

    Ok(())
}
