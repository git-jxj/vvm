use colored::Colorize;

use crate::config::Config;
use crate::errors::VvmError;
use crate::metadata::VersionMetadata;
use crate::resolve::VersionSpecifier;

pub fn run(config: &Config, specifier_str: Option<&str>, json: bool) -> Result<(), VvmError> {
    let dir_name = match specifier_str {
        Some(s) => {
            let raw = config.versions_dir().join(s);
            if raw.exists() {
                s.to_string()
            } else {
                // Try alias
                let alias_path = config.aliases_dir().join(s);
                if alias_path.is_symlink() {
                    if let Ok(target) = std::fs::read_link(&alias_path) {
                        let resolved = if target.is_absolute() {
                            target
                        } else {
                            config.aliases_dir().join(&target)
                        };
                        if let Some(name) = resolved.file_name() {
                            name.to_string_lossy().to_string()
                        } else {
                            VersionSpecifier::parse(s)?.dir_name()
                        }
                    } else {
                        VersionSpecifier::parse(s)?.dir_name()
                    }
                } else {
                    VersionSpecifier::parse(s)?.dir_name()
                }
            }
        }
        None => {
            let link = config.current_link();
            if !link.is_symlink() {
                return Err(VvmError::InvalidSpecifier(
                    "No active version. Run: vvm use <version>".to_string(),
                ));
            }
            let target = std::fs::read_link(&link)?;
            target
                .file_name()
                .ok_or_else(|| {
                    VvmError::InvalidSpecifier("Could not read current link".to_string())
                })?
                .to_string_lossy()
                .to_string()
        }
    };

    let version_dir = config.versions_dir().join(&dir_name);
    if !version_dir.exists() {
        return Err(VvmError::VersionNotFound(dir_name));
    }

    let meta = VersionMetadata::load(&version_dir).map_err(|_| {
        VvmError::InvalidSpecifier(format!(
            "'{}' has no metadata (incomplete install?)",
            dir_name
        ))
    })?;

    let is_current = config
        .current_link()
        .is_symlink()
        .then(|| std::fs::read_link(config.current_link()).ok())
        .flatten()
        .map(|t| t == version_dir || t.starts_with(&version_dir))
        .unwrap_or(false);

    let aliases = find_aliases(config, &dir_name);
    let venv_path = version_dir.join("venv");

    if json {
        let mut obj = serde_json::json!({
            "name": dir_name,
            "current": is_current,
            "type": meta.install_type,
            "cuda": meta.cuda,
            "version": meta.version,
            "commit": meta.commit,
            "branch": meta.branch,
            "pr": meta.pr,
            "source_path": meta.source_path,
            "installed_at": meta.installed_at,
            "venv": venv_path.to_string_lossy(),
            "aliases": aliases,
        });
        // Abbreviate $HOME to ~ in source_path
        if let Some(path) = meta.source_path {
            let home = std::env::var("HOME").unwrap_or_default();
            let display = if path.starts_with(&home) {
                format!("~{}", &path[home.len()..])
            } else {
                path
            };
            obj["source_path"] = serde_json::Value::String(display);
        }
        println!(
            "{}",
            serde_json::to_string_pretty(&obj).expect("serializable")
        );
        return Ok(());
    }

    // ── human-readable ───────────────────────────────────────────────────────

    let title = if is_current {
        format!("{} {}", dir_name.cyan().bold(), "(current)".green())
    } else {
        dir_name.cyan().bold().to_string()
    };
    println!("{}", title);

    row("type", &meta.install_type);
    row("cuda", &meta.cuda);

    if let Some(ref v) = meta.version {
        if !v.is_empty() {
            row("version", v);
        }
    }
    if let Some(ref commit) = meta.commit {
        row("commit", commit);
    }
    if let Some(ref branch) = meta.branch {
        row("branch", branch);
    }
    if let Some(pr) = meta.pr {
        row("pr", &format!("#{}", pr));
    }
    if let Some(ref path) = meta.source_path {
        let home = std::env::var("HOME").unwrap_or_default();
        let display = if path.starts_with(&home) {
            format!("~{}", &path[home.len()..])
        } else {
            path.clone()
        };
        row("source", &display);
    }
    row("installed", &meta.installed_at);
    row("venv", &venv_path.to_string_lossy());
    if !aliases.is_empty() {
        row("aliases", &aliases.join(", "));
    }

    Ok(())
}

fn row(label: &str, value: &str) {
    println!("  {:12} {}", format!("{}:", label).dimmed(), value);
}

fn find_aliases(config: &Config, version_name: &str) -> Vec<String> {
    let aliases_dir = config.aliases_dir();
    if !aliases_dir.exists() {
        return vec![];
    }
    std::fs::read_dir(&aliases_dir)
        .ok()
        .map(|entries| {
            entries
                .filter_map(|e| e.ok())
                .filter(|e| e.path().is_symlink())
                .filter(|e| {
                    std::fs::read_link(e.path())
                        .ok()
                        .and_then(|t| t.file_name().map(|f| f.to_string_lossy().to_string()))
                        .as_deref()
                        == Some(version_name)
                })
                .map(|e| e.file_name().to_string_lossy().to_string())
                .collect()
        })
        .unwrap_or_default()
}
