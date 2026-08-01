/// Tests for race conditions in concurrent vvm installs.
///
/// `vvm install` in parallel:
///   - "IO error: No such file or directory (os error 2)"  ← current symlink race
///   - "File exists (os error 17)"                         ← default alias race
use std::os::unix::fs as unixfs;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

const N: usize = 18;

fn make_tmpdir(suffix: &str) -> std::path::PathBuf {
    let p = std::path::PathBuf::from(format!("/tmp/vvm_test_{}", suffix));
    let _ = std::fs::remove_dir_all(&p);
    std::fs::create_dir_all(&p).unwrap();
    p
}

// ── current symlink ──────────────────────────────────────────────────────────

/// OLD approach: remove_file then symlink.
/// Expected to fail frequently with ENOENT when processes race.
fn update_current_old(current: &std::path::Path, target: &std::path::Path) -> std::io::Result<()> {
    if current.exists() || current.is_symlink() {
        std::fs::remove_file(current)?; // ← can return ENOENT if another thread won the race
    }
    unixfs::symlink(target, current) // ← can return EEXIST
}

/// NEW approach: write to per-PID temp then rename (atomic on Linux).
fn update_current_new(
    base_dir: &std::path::Path,
    current: &std::path::Path,
    target: &std::path::Path,
    id: usize,
) -> std::io::Result<()> {
    let tmp = base_dir.join(format!(".current.tmp.{}", id));
    let _ = std::fs::remove_file(&tmp);
    unixfs::symlink(target, &tmp)?;
    std::fs::rename(&tmp, current)
}

#[test]
fn test_current_symlink_old_races() {
    let dir = make_tmpdir("symlink_old");
    let current = dir.join("current");
    let target = dir.join("v0");
    std::fs::create_dir_all(&target).unwrap();
    unixfs::symlink(&target, &current).unwrap();

    let errors = Arc::new(AtomicUsize::new(0));
    let mut handles = vec![];

    for i in 1..=N {
        let cur = current.clone();
        let tgt = dir.join(format!("v{}", i));
        std::fs::create_dir_all(&tgt).unwrap();
        let errs = errors.clone();
        handles.push(std::thread::spawn(move || {
            if update_current_old(&cur, &tgt).is_err() {
                errs.fetch_add(1, Ordering::Relaxed);
            }
        }));
    }
    for h in handles {
        h.join().unwrap();
    }

    let failures = errors.load(Ordering::Relaxed);
    println!("OLD: {} / {} threads failed", failures, N);
    // This should regularly produce failures — document the known-bad behavior
    // (not asserting > 0 because the race is probabilistic, but in practice it fires)
}

#[test]
fn test_current_symlink_new_no_races() {
    let dir = make_tmpdir("symlink_new");
    let current = dir.join("current");
    let target = dir.join("v0");
    std::fs::create_dir_all(&target).unwrap();
    unixfs::symlink(&target, &current).unwrap();

    let errors = Arc::new(AtomicUsize::new(0));
    let mut handles = vec![];

    for i in 1..=N {
        let cur = current.clone();
        let tgt = dir.join(format!("v{}", i));
        let base = dir.clone();
        std::fs::create_dir_all(&tgt).unwrap();
        let errs = errors.clone();
        handles.push(std::thread::spawn(move || {
            if update_current_new(&base, &cur, &tgt, i).is_err() {
                errs.fetch_add(1, Ordering::Relaxed);
            }
        }));
    }
    for h in handles {
        h.join().unwrap();
    }

    let failures = errors.load(Ordering::Relaxed);
    println!("NEW: {} / {} threads failed", failures, N);
    assert_eq!(failures, 0, "atomic rename should produce zero failures");
    assert!(
        current.is_symlink(),
        "current link must still exist after concurrent updates"
    );
}

// ── default alias ────────────────────────────────────────────────────────────

/// OLD approach: check exists, then symlink — propagates AlreadyExists.
/// The artificial sleep widens the TOCTOU window to reliably trigger the race.
fn set_default_old(link: &std::path::Path, target: &std::path::Path) -> std::io::Result<()> {
    if !link.exists() {
        unixfs::symlink(target, link)?; // ← EEXIST if another thread won the race
    }
    Ok(())
}

/// NEW approach: attempt symlink, ignore AlreadyExists.
fn set_default_new(link: &std::path::Path, target: &std::path::Path) -> std::io::Result<()> {
    if !link.exists() {
        match unixfs::symlink(target, link) {
            Ok(_) => {}
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

#[test]
fn test_default_alias_old_races() {
    let dir = make_tmpdir("alias_old");
    let link = dir.join("default");
    let target = dir.join("v1");
    std::fs::create_dir_all(&target).unwrap();

    let errors = Arc::new(AtomicUsize::new(0));
    let mut handles = vec![];

    for _ in 0..N {
        let l = link.clone();
        let t = target.clone();
        let errs = errors.clone();
        handles.push(std::thread::spawn(move || {
            if set_default_old(&l, &t).is_err() {
                errs.fetch_add(1, Ordering::Relaxed);
            }
        }));
    }
    for h in handles {
        h.join().unwrap();
    }

    let failures = errors.load(Ordering::Relaxed);
    println!("OLD alias: {} / {} threads failed", failures, N);
}

#[test]
fn test_default_alias_new_no_races() {
    let dir = make_tmpdir("alias_new");
    let link = dir.join("default");
    let target = dir.join("v1");
    std::fs::create_dir_all(&target).unwrap();

    let errors = Arc::new(AtomicUsize::new(0));
    let mut handles = vec![];

    for _ in 0..N {
        let l = link.clone();
        let t = target.clone();
        let errs = errors.clone();
        handles.push(std::thread::spawn(move || {
            if set_default_new(&l, &t).is_err() {
                errs.fetch_add(1, Ordering::Relaxed);
            }
        }));
    }
    for h in handles {
        h.join().unwrap();
    }

    let failures = errors.load(Ordering::Relaxed);
    println!("NEW alias: {} / {} threads failed", failures, N);
    assert_eq!(
        failures, 0,
        "AlreadyExists-tolerant symlink should produce zero failures"
    );
    assert!(
        link.is_symlink(),
        "default alias must exist after concurrent creation"
    );
}
