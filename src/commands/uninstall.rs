use colored::Colorize;

use crate::config::Config;
use crate::errors::VvmError;
use crate::fsutil::remove_dir_all_robust;
use crate::resolve::VersionSpecifier;

pub fn run(config: &Config, specifier_str: &str) -> Result<(), VvmError> {
    // Try raw string as directory name first (handles local-*, etc.)
    let dir_name = {
        let raw_dir = config.versions_dir().join(specifier_str);
        if raw_dir.exists() {
            specifier_str.to_string()
        } else {
            let specifier = VersionSpecifier::parse(specifier_str)?;
            specifier.dir_name()
        }
    };
    let version_dir = config.versions_dir().join(&dir_name);

    if !version_dir.exists() {
        return Err(VvmError::VersionNotFound(dir_name));
    }

    // Check if it's the current version
    let current_link = config.current_link();
    if current_link.is_symlink() {
        if let Ok(target) = std::fs::read_link(&current_link) {
            if target == version_dir {
                eprintln!(
                    "{} '{}' is the current version. Removing current link.",
                    "Warning:".yellow().bold(),
                    dir_name
                );
                std::fs::remove_file(&current_link)?;
            }
        }
    }

    // Remove aliases pointing to this version
    cleanup_aliases(config, &dir_name)?;

    // Remove the version directory
    remove_dir_all_robust(&version_dir)?;

    println!("{} {} uninstalled.", "✓".green().bold(), dir_name.cyan());

    Ok(())
}

fn cleanup_aliases(config: &Config, version_name: &str) -> Result<(), VvmError> {
    let aliases_dir = config.aliases_dir();
    if !aliases_dir.exists() {
        return Ok(());
    }

    for entry in std::fs::read_dir(&aliases_dir)? {
        let entry = entry?;
        if entry.path().is_symlink() {
            if let Ok(target) = std::fs::read_link(entry.path()) {
                let target_name = target.file_name().map(|f| f.to_string_lossy().to_string());
                if target_name.as_deref() == Some(version_name) {
                    std::fs::remove_file(entry.path())?;
                    eprintln!("  Removed alias '{}'", entry.file_name().to_string_lossy());
                }
            }
        }
    }

    Ok(())
}
