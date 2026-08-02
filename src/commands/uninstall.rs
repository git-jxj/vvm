use colored::Colorize;

use crate::config::Config;
use crate::errors::VvmError;
use crate::fsutil::remove_dir_all_robust;
use crate::resolve::VersionSpecifier;

pub fn run(config: &Config, specifier_str: &str) -> Result<(), VvmError> {
    // Resolution order mirrors `vvm use`: raw directory name first (handles
    // local-*, etc.), then alias, then version specifier. Without the alias
    // step, `vvm uninstall dev` fell through to release parsing and died on
    // the mangled name ("Version not found: vdev").
    let dir_name = {
        let raw_dir = config.versions_dir().join(specifier_str);
        if raw_dir.exists() {
            specifier_str.to_string()
        } else if let Some(target) = resolve_alias(config, specifier_str) {
            println!(
                "  Uninstalling {} (alias '{}')",
                target.cyan(),
                specifier_str
            );
            target
        } else {
            let specifier = VersionSpecifier::parse(specifier_str)?;
            specifier.dir_name()
        }
    };
    let version_dir = config.versions_dir().join(&dir_name);

    if !version_dir.exists() {
        // Report what the user typed, not the parsed directory name — an
        // unknown alias like `dev` would otherwise surface as "vdev".
        return Err(VvmError::VersionNotFound(specifier_str.to_string()));
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

/// Resolve an alias to the version directory name it points at.
/// Reads the symlink itself (same as `cleanup_aliases`) rather than following
/// it, so it works for absolute and relative targets alike.
fn resolve_alias(config: &Config, name: &str) -> Option<String> {
    let alias_path = config.aliases_dir().join(name);
    if !alias_path.is_symlink() {
        return None;
    }
    let target = std::fs::read_link(&alias_path).ok()?;
    target.file_name().map(|f| f.to_string_lossy().to_string())
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
