use colored::Colorize;

use crate::errors::VvmError;
use crate::github;

pub fn run(pattern: Option<&str>, show_prerelease: bool) -> Result<(), VvmError> {
    println!("{}", "Fetching releases from GitHub...".dimmed());

    let releases = github::list_releases()?;

    let filtered: Vec<_> = releases
        .iter()
        .filter(|r| show_prerelease || !r.prerelease)
        .filter(|r| {
            if let Some(pat) = pattern {
                r.tag_name.contains(pat)
            } else {
                true
            }
        })
        .collect();

    if filtered.is_empty() {
        println!("No matching releases found.");
        return Ok(());
    }

    for release in &filtered {
        let tag = &release.tag_name;
        if release.prerelease {
            println!("  {} {}", tag, "(pre-release)".dimmed());
        } else {
            println!("  {}", tag.green());
        }
    }

    println!(
        "\n{} releases listed. Install with: vvm install <version>",
        filtered.len()
    );

    Ok(())
}
