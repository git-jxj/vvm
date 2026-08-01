use crate::config::Config;
use crate::errors::VvmError;

/// Print shell-evaluable environment variables for the current version.
/// Called by the shell hook after `vvm use` / `vvm install`,
/// or manually by the user: eval "$(vvm env)"
pub fn run(config: &Config) -> Result<(), VvmError> {
    let link = config.current_link();
    if !link.is_symlink() {
        return Ok(());
    }

    let target = std::fs::read_link(&link)?;
    let venv_path = target.join("venv");

    if !venv_path.exists() {
        return Ok(());
    }

    let bin_path = venv_path.join("bin");

    // Output shell commands to eval.
    // Strategy: strip any previous vvm venv bin from PATH, then prepend the new one.
    println!("export VVM_DIR=\"{}\";", config.base_dir.display());
    println!("export VIRTUAL_ENV=\"{}\";", venv_path.display());

    // Remove previous vvm bin from PATH using shell parameter expansion (no sed)
    println!(
        r#"case ":$PATH:" in *":${{__VVM_PREV_BIN:-}}:"*) PATH="${{PATH//$__VVM_PREV_BIN:/}}";; esac;"#
    );
    println!("export __VVM_PREV_BIN=\"{}\";", bin_path.display());
    println!("export PATH=\"{}:$PATH\";", bin_path.display());
    println!("hash -r 2>/dev/null;");

    Ok(())
}
