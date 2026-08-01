use std::io;
use std::path::Path;
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Remove a directory tree, tolerating NFS quirks.
///
/// On NFS, `std::fs::remove_dir_all` can fail with ENOTEMPTY even after every
/// entry has been unlinked: stale directory-attribute caches or `.nfs*`
/// silly-rename files from another client keep `rmdir` from succeeding. Work
/// around this by renaming the directory to a hidden sibling first (one
/// atomic rename — the caller's path is immediately free for reuse), then
/// best-effort `remove_dir_all` the renamed copy with backoff. If the
/// physical delete still fails, warn and leave the orphan behind; the
/// caller's path is usable regardless.
pub fn remove_dir_all_robust(path: &Path) -> io::Result<()> {
    if !path.try_exists()? {
        return Ok(());
    }

    let parent = path
        .parent()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "path has no parent"))?;
    let file_name = path
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "path has no valid name"))?;

    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let trash = parent.join(format!(
        ".{}.vvm-trash-{}-{}",
        file_name,
        std::process::id(),
        nanos
    ));

    std::fs::rename(path, &trash)?;

    for delay_ms in [0u64, 100, 500, 2000] {
        if delay_ms > 0 {
            thread::sleep(Duration::from_millis(delay_ms));
        }
        if std::fs::remove_dir_all(&trash).is_ok() {
            return Ok(());
        }
    }

    eprintln!(
        "warning: couldn't fully delete {} on this filesystem; leftover at {} — rm -rf it to reclaim space",
        path.display(),
        trash.display()
    );
    Ok(())
}
