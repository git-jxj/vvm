use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::time::{SystemTime, UNIX_EPOCH};

struct TestDir(PathBuf);

impl TestDir {
    fn new() -> Self {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "vvm_relative_data_dir_{}_{}",
            std::process::id(),
            nonce
        ));
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn run(root: &Path, data_dir: &Path, args: &[&str]) -> Output {
    let output = Command::new(env!("CARGO_BIN_EXE_vvm"))
        .current_dir(root)
        .env("VVM_DIR", data_dir)
        .env("NO_COLOR", "1")
        .args(args)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "vvm {args:?} failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    output
}

fn check_version_switching(root: &Path, configured_dir: &Path, data_dir: &Path) {
    let version = data_dir.join("versions/v0.18.0");
    std::fs::create_dir_all(version.join("venv/bin")).unwrap();
    std::fs::write(
        version.join("metadata.json"),
        r#"{
            "type": "release",
            "name": "v0.18.0",
            "version": "0.18.0",
            "cuda": "cpu",
            "installed_at": "2026-01-01T00:00:00Z"
        }"#,
    )
    .unwrap();

    run(root, configured_dir, &["alias", "stable", "0.18.0"]);
    assert_eq!(
        data_dir.join("aliases/stable").canonicalize().unwrap(),
        version.canonicalize().unwrap()
    );

    run(root, configured_dir, &["use", "stable"]);
    assert_eq!(
        data_dir.join("current").canonicalize().unwrap(),
        version.canonicalize().unwrap()
    );

    let info = run(root, configured_dir, &["info", "--json"]);
    let info: serde_json::Value = serde_json::from_slice(&info.stdout).unwrap();
    assert_eq!(info["name"], "v0.18.0");
    assert_eq!(info["current"], true);

    let env = run(root, configured_dir, &["env"]);
    let env = String::from_utf8(env.stdout).unwrap();
    assert!(env.contains(&format!("export VVM_DIR=\"{}\";", data_dir.display())));
    assert!(env.contains(&format!(
        "export VIRTUAL_ENV=\"{}\";",
        version.join("venv").display()
    )));
    assert!(env.contains(&format!(
        "export PATH=\"{}:$PATH\";",
        version.join("venv/bin").display()
    )));
}

#[test]
fn relative_data_dir_supports_aliases_switching_and_activation() {
    for relative in ["data", "nested/data", "."] {
        let root = TestDir::new();
        check_version_switching(&root.0, Path::new(relative), &root.0.join(relative));
    }
}

#[test]
fn absolute_data_dir_supports_aliases_switching_and_activation() {
    let root = TestDir::new();
    let data_dir = root.0.join("data");
    check_version_switching(&root.0, &data_dir, &data_dir);
}

#[test]
fn relative_data_dir_can_be_created_on_first_command() {
    let root = TestDir::new();
    let relative = Path::new("not-created/nested");
    run(&root.0, relative, &["alias"]);
    assert!(root.0.join(relative).join("versions").is_dir());
    assert!(root.0.join(relative).join("aliases").is_dir());
}
