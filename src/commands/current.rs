use colored::Colorize;

use crate::config::Config;
use crate::errors::VvmError;
use crate::metadata::VersionMetadata;

pub fn run(config: &Config) -> Result<(), VvmError> {
    let link = config.current_link();
    if !link.is_symlink() {
        return Err(VvmError::NoCurrentVersion);
    }

    let target = std::fs::read_link(&link)?;
    let name = target
        .file_name()
        .map(|f| f.to_string_lossy().to_string())
        .unwrap_or_else(|| "unknown".to_string());

    let mut info_parts = vec![name.clone()];

    if let Ok(meta) = VersionMetadata::load(&target) {
        info_parts.push(format!("cuda: {}", meta.cuda));
        if let Some(ref commit) = meta.commit {
            let short = if commit.len() > 8 {
                &commit[..8]
            } else {
                commit
            };
            info_parts.push(format!("commit: {}", short));
        }
    }

    println!("{}", info_parts.join(" | ").cyan());

    Ok(())
}
