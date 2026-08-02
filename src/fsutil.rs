use std::io;
use std::path::{Path, PathBuf};
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

/// An old version directory renamed aside while its replacement installs.
///
/// Updating a dynamic version (nightly/branch/pr) used to delete the old
/// install before the new one existed, so an interrupt or a failed download
/// left the user with nothing. Parking is one atomic rename, and a venv is
/// full of absolute paths — restoring it by renaming back yields exactly the
/// pre-update install, while a deleted one is gone for good. The parked copy
/// is removed only in `commit()`, after the new install fully succeeded; if
/// the guard drops uncommitted (any early error return), it restores.
///
/// Ctrl+C exits the process without running Drop, so the install's SIGINT
/// handler must perform the same restore itself (see `restore_paths`).
pub struct ParkedVersion {
    parked: PathBuf,
    original: PathBuf,
    committed: bool,
}

impl ParkedVersion {
    /// Rename `original` to a hidden sibling (`.{name}.vvm-prev-{pid}-{nanos}`).
    /// Any older parked copy of the same name is superseded wreckage from a
    /// previous interrupted update — removed here so orphans don't accumulate.
    pub fn park(original: &Path) -> io::Result<Self> {
        let parent = original
            .parent()
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "path has no parent"))?;
        let file_name = original
            .file_name()
            .and_then(|n| n.to_str())
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "path has no valid name"))?;

        let stale_prefix = format!(".{}.vvm-prev-", file_name);
        if let Ok(entries) = std::fs::read_dir(parent) {
            for entry in entries.filter_map(|e| e.ok()) {
                if entry
                    .file_name()
                    .to_str()
                    .is_some_and(|n| n.starts_with(&stale_prefix))
                {
                    let _ = remove_dir_all_robust(&entry.path());
                }
            }
        }

        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let parked = parent.join(format!("{}{}-{}", stale_prefix, std::process::id(), nanos));
        std::fs::rename(original, &parked)?;
        Ok(ParkedVersion {
            parked,
            original: original.to_path_buf(),
            committed: false,
        })
    }

    /// (parked, original) for a signal handler to replay the restore:
    /// remove the partial new install at `original`, then rename `parked` back.
    pub fn restore_paths(&self) -> (PathBuf, PathBuf) {
        (self.parked.clone(), self.original.clone())
    }

    /// The new install fully succeeded — delete the old copy.
    pub fn commit(mut self) {
        self.committed = true;
        let _ = remove_dir_all_robust(&self.parked);
    }
}

impl Drop for ParkedVersion {
    fn drop(&mut self) {
        if self.committed {
            return;
        }
        // Error paths remove the partial install before returning, but make
        // sure the original path is free either way.
        let _ = std::fs::remove_dir_all(&self.original);
        if std::fs::rename(&self.parked, &self.original).is_err()
            && self.parked.try_exists().unwrap_or(false)
        {
            eprintln!(
                "warning: couldn't restore previous version to {}; it is preserved at {}",
                self.original.display(),
                self.parked.display()
            );
        }
    }
}

/// Best-effort removal of orphaned `.vvm-trash-*` directories left behind when
/// a `remove_dir_all_robust` was interrupted between its rename and its delete.
/// Only entries untouched for over an hour are removed: an in-flight delete
/// (possibly on another node) keeps bumping the directory's mtime as entries
/// unlink, so a stale mtime means nobody is working on it.
pub fn sweep_stale_trash(parent: &Path) {
    let Ok(entries) = std::fs::read_dir(parent) else {
        return;
    };
    for entry in entries.filter_map(|e| e.ok()) {
        if !entry
            .file_name()
            .to_str()
            .is_some_and(|n| n.starts_with('.') && n.contains(".vvm-trash-"))
        {
            continue;
        }
        let stale = entry
            .metadata()
            .and_then(|m| m.modified())
            .ok()
            .and_then(|t| t.elapsed().ok())
            .is_some_and(|age| age > Duration::from_secs(3600));
        if stale {
            let _ = std::fs::remove_dir_all(entry.path());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "vvm-fsutil-{}-{}-{}",
            name,
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn make_version(parent: &Path, name: &str) -> PathBuf {
        let dir = parent.join(name);
        std::fs::create_dir(&dir).unwrap();
        std::fs::write(dir.join("metadata.json"), b"{}").unwrap();
        dir
    }

    #[test]
    fn uncommitted_park_restores_original_on_drop() {
        let root = scratch("restore");
        let version = make_version(&root, "nightly");
        {
            let _guard = ParkedVersion::park(&version).unwrap();
            assert!(!version.exists(), "original must be freed while parked");
        }
        assert!(version.exists(), "drop without commit must restore");
        assert!(version.join("metadata.json").exists(), "contents intact");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn committed_park_deletes_old_and_keeps_replacement() {
        let root = scratch("commit");
        let version = make_version(&root, "nightly");
        let guard = ParkedVersion::park(&version).unwrap();
        // Simulate the new install landing at the original path.
        std::fs::create_dir(&version).unwrap();
        std::fs::write(version.join("metadata.json"), b"{\"new\":1}").unwrap();
        guard.commit();
        assert_eq!(
            std::fs::read(version.join("metadata.json")).unwrap(),
            b"{\"new\":1}",
            "commit must not touch the replacement install"
        );
        let leftovers: Vec<_> = std::fs::read_dir(&root)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().starts_with('.'))
            .collect();
        assert!(
            leftovers.is_empty(),
            "parked copy must be gone: {:?}",
            leftovers
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    /// A SIGKILL (or pre-handler Ctrl+C) can strand a parked copy. The next
    /// update of the same version must clear it rather than accumulate.
    #[test]
    fn park_supersedes_orphans_of_same_name_only() {
        let root = scratch("supersede");
        let orphan = root.join(".nightly.vvm-prev-999-1");
        std::fs::create_dir(&orphan).unwrap();
        let other = root.join(".pr-123.vvm-prev-999-1");
        std::fs::create_dir(&other).unwrap();

        let version = make_version(&root, "nightly");
        let guard = ParkedVersion::park(&version).unwrap();
        assert!(!orphan.exists(), "same-name orphan must be removed");
        assert!(other.exists(), "other versions' backups must survive");
        guard.commit();
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn sweep_removes_only_stale_trash() {
        let root = scratch("sweep");
        let fresh = root.join(".nightly.vvm-trash-1-1");
        std::fs::create_dir(&fresh).unwrap();
        let stale = root.join(".pr-1.vvm-trash-2-2");
        std::fs::create_dir(&stale).unwrap();
        // A parked backup must never be swept, no matter how old.
        let parked = root.join(".nightly.vvm-prev-3-3");
        std::fs::create_dir(&parked).unwrap();
        let old_mtime = SystemTime::now() - Duration::from_secs(2 * 3600);
        for dir in [&stale, &parked] {
            std::fs::File::open(dir)
                .unwrap()
                .set_modified(old_mtime)
                .unwrap();
        }

        sweep_stale_trash(&root);
        assert!(fresh.exists(), "recent trash may be an in-flight delete");
        assert!(!stale.exists(), "stale trash must be reclaimed");
        assert!(parked.exists(), "parked backups are not trash");
        let _ = std::fs::remove_dir_all(&root);
    }
}
