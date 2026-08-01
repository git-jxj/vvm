use crate::errors::VvmError;

/// Parsed version specifier from user input.
#[derive(Debug, Clone)]
pub enum VersionSpecifier {
    /// A release version like "0.18.0" or "v0.18.0"
    Release(String),
    /// A specific commit hash like "commit:abc123"
    Commit(String),
    /// A branch name like "branch:main"
    Branch(String),
    /// A PR number like "pr:1234"
    Pr(u64),
    /// The nightly build
    Nightly,
}

/// Reject refs that would be unsafe when interpolated into filesystem paths
/// or shell/git invocations: path-escape fragments, glob/ref metacharacters,
/// and whitespace. Keep this conservative — git's own ref rules permit more,
/// but we never need them and the vvm dir_name() downstream can't express them.
pub fn validate_ref(kind: &str, value: &str) -> Result<(), VvmError> {
    let bad_chars = [
        ' ', '\t', '\n', '\r', '~', '^', ':', '?', '*', '[', '\\', '"', '\'', '`', '$',
    ];
    if value.contains("..") {
        return Err(VvmError::InvalidSpecifier(format!(
            "{} must not contain '..': {}",
            kind, value
        )));
    }
    if value.starts_with('-') || value.starts_with('/') || value.ends_with('/') {
        return Err(VvmError::InvalidSpecifier(format!(
            "{} has an invalid leading/trailing character: {}",
            kind, value
        )));
    }
    if let Some(c) = value
        .chars()
        .find(|c| bad_chars.contains(c) || c.is_control())
    {
        return Err(VvmError::InvalidSpecifier(format!(
            "{} contains disallowed character {:?}: {}",
            kind, c, value
        )));
    }
    Ok(())
}

impl VersionSpecifier {
    /// Parse a user-provided version string.
    pub fn parse(input: &str) -> Result<Self, VvmError> {
        let input = input.trim();

        if input.eq_ignore_ascii_case("nightly") {
            return Ok(Self::Nightly);
        }

        if let Some(hash) = input.strip_prefix("commit:") {
            let hash = hash.trim();
            if hash.is_empty() {
                return Err(VvmError::InvalidSpecifier(
                    "commit: requires a hash".to_string(),
                ));
            }
            if !hash.chars().all(|c| c.is_ascii_hexdigit()) {
                return Err(VvmError::InvalidSpecifier(format!(
                    "commit hash must be hex: {}",
                    hash
                )));
            }
            return Ok(Self::Commit(hash.to_string()));
        }

        if let Some(branch) = input.strip_prefix("branch:") {
            let branch = branch.trim();
            if branch.is_empty() {
                return Err(VvmError::InvalidSpecifier(
                    "branch: requires a name".to_string(),
                ));
            }
            validate_ref("branch name", branch)?;
            return Ok(Self::Branch(branch.to_string()));
        }

        if let Some(pr) = input.strip_prefix("pr:") {
            let pr = pr.trim();
            let num: u64 = pr
                .parse()
                .map_err(|_| VvmError::InvalidSpecifier(format!("invalid PR number: {}", pr)))?;
            return Ok(Self::Pr(num));
        }

        // Catch common mistakes: users copying directory names from `vvm ls`
        // (which use hyphens) instead of specifiers (which use colons).
        if let Some(corrected) = suggest_corrected_specifier(input) {
            return Err(VvmError::InvalidSpecifier(format!(
                "'{}' looks like a directory name. Did you mean '{}'?",
                input, corrected
            )));
        }

        // Treat as release version, strip leading 'v' if present
        let version = input.strip_prefix('v').unwrap_or(input);
        if version.is_empty() {
            return Err(VvmError::InvalidSpecifier(
                "empty version string".to_string(),
            ));
        }
        validate_ref("release version", version)?;
        Ok(Self::Release(version.to_string()))
    }

    /// Directory name under $VVM_DIR/versions/
    pub fn dir_name(&self) -> String {
        match self {
            Self::Release(v) => format!("v{}", v),
            Self::Commit(h) => {
                let short = if h.len() > 8 { &h[..8] } else { h };
                format!("commit-{}", short)
            }
            Self::Branch(b) => format!("branch-{}", b.replace('/', "-")),
            Self::Pr(n) => format!("pr-{}", n),
            Self::Nightly => "nightly".to_string(),
        }
    }
}

/// If `input` looks like a version directory name (as printed by `vvm ls`),
/// return the equivalent `kind:value` specifier form. Returns None for
/// inputs that should parse as release versions or are otherwise unrelated.
///
/// Note on branch ambiguity: vvm encodes `release/v0.10` as the directory
/// `branch-release-v0.10`, so parsing a `branch-<name>` directory name back
/// into a branch is ambiguous (can't distinguish `-` from the slash
/// replacement). We suggest the literal hyphenated branch; the user should
/// re-issue with the real branch name if the suggestion is wrong.
pub fn suggest_corrected_specifier(input: &str) -> Option<String> {
    if let Some(rest) = input.strip_prefix("commit-") {
        if !rest.is_empty() && rest.chars().all(|c| c.is_ascii_hexdigit()) {
            return Some(format!("commit:{}", rest));
        }
    }
    if let Some(rest) = input.strip_prefix("pr-") {
        if !rest.is_empty() && rest.chars().all(|c| c.is_ascii_digit()) {
            return Some(format!("pr:{}", rest));
        }
    }
    if let Some(rest) = input.strip_prefix("branch-") {
        if !rest.is_empty() {
            return Some(format!("branch:{}", rest));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hyphen_commit_form_suggests_colon() {
        let err = VersionSpecifier::parse("commit-2a69949b").unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("commit:2a69949b"), "got: {}", msg);
    }

    #[test]
    fn hyphen_pr_form_suggests_colon() {
        let err = VersionSpecifier::parse("pr-1234").unwrap_err();
        assert!(err.to_string().contains("pr:1234"));
    }

    #[test]
    fn hyphen_branch_form_suggests_colon() {
        let err = VersionSpecifier::parse("branch-main").unwrap_err();
        assert!(err.to_string().contains("branch:main"));
    }

    #[test]
    fn plain_version_still_parses() {
        assert!(matches!(
            VersionSpecifier::parse("0.18.0").unwrap(),
            VersionSpecifier::Release(ref v) if v == "0.18.0"
        ));
        assert!(matches!(
            VersionSpecifier::parse("v0.18.0").unwrap(),
            VersionSpecifier::Release(ref v) if v == "0.18.0"
        ));
    }
}
