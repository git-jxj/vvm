use colored::Colorize;

use crate::config::Config;
use crate::errors::VvmError;
use crate::metadata::VersionMetadata;

pub fn run(config: &Config, json: bool) -> Result<(), VvmError> {
    let versions_dir = config.versions_dir();
    if !versions_dir.exists() {
        if json {
            println!("[]");
        } else {
            println!("No versions installed. Run: vvm install <version>");
        }
        return Ok(());
    }

    let current_target = {
        let link = config.current_link();
        if link.is_symlink() {
            let target = std::fs::read_link(&link).ok();
            if let Some(ref t) = target {
                if !t.exists() && !json {
                    eprintln!(
                        "{}: current symlink points to missing {} — run `vvm use <version>`",
                        "warning".yellow().bold(),
                        t.display()
                    );
                }
            }
            target
        } else {
            None
        }
    };

    let mut entries: Vec<String> = std::fs::read_dir(&versions_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir())
        .map(|e| e.file_name().to_string_lossy().to_string())
        .collect();

    entries.sort();

    if entries.is_empty() {
        if json {
            println!("[]");
        } else {
            println!("No versions installed. Run: vvm install <version>");
        }
        return Ok(());
    }

    if json {
        let home = std::env::var("HOME").unwrap_or_default();
        let items: Vec<serde_json::Value> = entries
            .iter()
            .map(|name| {
                let version_path = versions_dir.join(name);
                let is_current = current_target
                    .as_ref()
                    .map(|t| t == &version_path || t.starts_with(&version_path))
                    .unwrap_or(false);
                let aliases = find_aliases_for(config, name);
                let venv = version_path.join("venv").to_string_lossy().to_string();

                let meta = VersionMetadata::load(&version_path).ok();
                let source_path = meta.as_ref().and_then(|m| {
                    m.source_path.as_ref().map(|p| {
                        if p.starts_with(&home) {
                            format!("~{}", &p[home.len()..])
                        } else {
                            p.clone()
                        }
                    })
                });

                serde_json::json!({
                    "name": name,
                    "current": is_current,
                    "type": meta.as_ref().map(|m| m.install_type.as_str()),
                    "cuda": meta.as_ref().map(|m| m.cuda.as_str()),
                    "version": meta.as_ref().and_then(|m| m.version.as_deref()),
                    "commit": meta.as_ref().and_then(|m| m.commit.as_deref()),
                    "branch": meta.as_ref().and_then(|m| m.branch.as_deref()),
                    "pr": meta.as_ref().and_then(|m| m.pr),
                    "source_path": source_path,
                    "installed_at": meta.as_ref().map(|m| m.installed_at.as_str()),
                    "venv": venv,
                    "aliases": aliases,
                })
            })
            .collect();

        println!(
            "{}",
            serde_json::to_string_pretty(&items).expect("serializable")
        );
        return Ok(());
    }

    for name in &entries {
        let version_path = versions_dir.join(name);
        let is_current = current_target
            .as_ref()
            .map(|t| t == &version_path || t.starts_with(&version_path))
            .unwrap_or(false);
        let marker = if is_current { " *" } else { "  " };

        let mut line = format!("{} {}", marker, name);

        let meta_path = versions_dir.join(name);
        if let Ok(meta) = VersionMetadata::load(&meta_path) {
            if let Some(ref v) = meta.version {
                if !v.is_empty() {
                    line.push_str(&format!("   {}", v.cyan()));
                }
            }
            if let Some(ref branch) = meta.branch {
                line.push_str(&format!("   branch:{}", branch.yellow()));
            }
            if let Some(ref path) = meta.source_path {
                let home = std::env::var("HOME").unwrap_or_default();
                let display = if path.starts_with(&home) {
                    format!("~{}", &path[home.len()..])
                } else {
                    path.clone()
                };
                line.push_str(&format!("   {}", display));
            }
        }

        let aliases = find_aliases_for(config, name);
        if !aliases.is_empty() {
            line.push_str(&format!(" [{}]", aliases.join(", ")));
        }

        if is_current {
            println!("{}", line.green());
        } else {
            println!("{}", line);
        }
    }

    Ok(())
}

fn find_aliases_for(config: &Config, version_name: &str) -> Vec<String> {
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
