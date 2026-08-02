use reqwest::blocking::{Client, RequestBuilder, Response};
use reqwest::header::{ACCEPT, AUTHORIZATION};
use serde::de::DeserializeOwned;
use serde::Deserialize;
use std::sync::OnceLock;

use crate::errors::VvmError;

const REPO: &str = "vllm-project/vllm";
const API_BASE: &str = "https://api.github.com";

fn build_client() -> Result<Client, VvmError> {
    Ok(Client::builder()
        .user_agent(concat!("vvm/", env!("CARGO_PKG_VERSION")))
        .timeout(std::time::Duration::from_secs(30))
        .build()?)
}

/// Where a token came from. Kept alongside the token so a rejected one can
/// name the exact thing to fix — "$GITHUB_TOKEN is expired" is actionable,
/// "HTTP 401" is not.
#[derive(Clone, Copy)]
enum TokenSource {
    GithubTokenEnv,
    GhTokenEnv,
    GhCli,
}

impl TokenSource {
    fn name(self) -> &'static str {
        match self {
            TokenSource::GithubTokenEnv => "$GITHUB_TOKEN",
            TokenSource::GhTokenEnv => "$GH_TOKEN",
            TokenSource::GhCli => "`gh auth token`",
        }
    }

    fn fix_hint(self) -> &'static str {
        match self {
            TokenSource::GithubTokenEnv => "Unset GITHUB_TOKEN or replace it with a valid token",
            TokenSource::GhTokenEnv => "Unset GH_TOKEN or replace it with a valid token",
            TokenSource::GhCli => "Run `gh auth login` to re-authenticate",
        }
    }
}

/// Resolve a GitHub token from env vars, falling back to `gh auth token`.
/// Cached per-process so we don't fork `gh` on every API call.
fn resolve_token() -> Option<(String, TokenSource)> {
    static CACHED: OnceLock<Option<(String, TokenSource)>> = OnceLock::new();
    CACHED
        .get_or_init(|| {
            if let Ok(t) = std::env::var("GITHUB_TOKEN") {
                if !t.is_empty() {
                    return Some((t, TokenSource::GithubTokenEnv));
                }
            }
            if let Ok(t) = std::env::var("GH_TOKEN") {
                if !t.is_empty() {
                    return Some((t, TokenSource::GhTokenEnv));
                }
            }
            let output = std::process::Command::new("gh")
                .args(["auth", "token"])
                .output()
                .ok()?;
            if !output.status.success() {
                return None;
            }
            let token = String::from_utf8(output.stdout).ok()?.trim().to_string();
            if token.is_empty() {
                None
            } else {
                Some((token, TokenSource::GhCli))
            }
        })
        .clone()
}

fn auth_header() -> Option<String> {
    resolve_token().map(|(t, _)| format!("Bearer {}", t))
}

/// Attach standard GitHub API headers (Accept + optional auth) to a request.
fn with_headers(req: RequestBuilder) -> RequestBuilder {
    let mut req = req.header(ACCEPT, "application/vnd.github.v3+json");
    if let Some(token) = auth_header() {
        req = req.header(AUTHORIZATION, token);
    }
    req
}

/// Convert a non-success response into a typed error. 403/429 with
/// X-RateLimit-Remaining: 0 become a targeted "rate limit" message
/// that tells the user to set GITHUB_TOKEN, since that's the fix 99% of the time.
/// 401 is the mirror image: GitHub only answers 401 when credentials were sent
/// and rejected, so the request would have succeeded anonymously — the token,
/// not the ref, is what's broken.
fn response_error(resp: Response, context: &str) -> VvmError {
    let status = resp.status();
    let remaining = resp
        .headers()
        .get("x-ratelimit-remaining")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<u64>().ok());
    let reset = resp
        .headers()
        .get("x-ratelimit-reset")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<i64>().ok());

    let is_rate_limited = matches!(status.as_u16(), 403 | 429) && remaining == Some(0);

    if is_rate_limited {
        let wait_hint = reset
            .map(|ts| {
                let now = chrono::Utc::now().timestamp();
                let mins = ((ts - now).max(0) + 59) / 60;
                format!(" (resets in ~{} min)", mins)
            })
            .unwrap_or_default();
        let token_hint = if auth_header().is_none() {
            " Set GITHUB_TOKEN or run `gh auth login` to get a higher rate limit."
        } else {
            ""
        };
        return VvmError::GitHubApi(format!(
            "{}: rate limit exceeded{}.{}",
            context, wait_hint, token_hint
        ));
    }

    if status.as_u16() == 401 {
        if let Some((_, source)) = resolve_token() {
            return VvmError::GitHubApi(format!(
                "{}: the token from {} was rejected (invalid, expired, or revoked). {} \
                 — public repos need no token at all.",
                context,
                source.name(),
                source.fix_hint()
            ));
        }
    }

    VvmError::GitHubApi(format!("{}: HTTP {}", context, status))
}

/// Shared GET + JSON decode helper. All public functions funnel through this.
fn get_json<T: DeserializeOwned>(path: &str, context: &str) -> Result<T, VvmError> {
    let client = build_client()?;
    let url = format!("{}{}", API_BASE, path);
    let resp = with_headers(client.get(&url)).send()?;
    if !resp.status().is_success() {
        return Err(response_error(resp, context));
    }
    Ok(resp.json()?)
}

#[derive(Deserialize)]
struct PrResponse {
    head: PrHead,
    #[serde(default)]
    merged: bool,
    merge_commit_sha: Option<String>,
}

#[derive(Deserialize)]
struct PrHead {
    sha: String,
}

impl PrResponse {
    fn merged_commit(self) -> Option<String> {
        if self.merged {
            self.merge_commit_sha
        } else {
            None
        }
    }
}

#[derive(Deserialize)]
struct BranchResponse {
    commit: BranchCommit,
}

#[derive(Deserialize)]
struct BranchCommit {
    sha: String,
}

#[derive(Deserialize)]
pub struct Release {
    pub tag_name: String,
    #[serde(default)]
    pub prerelease: bool,
}

#[derive(Deserialize)]
struct CommitResponse {
    sha: String,
}

/// Resolve a (possibly short) commit hash to its full 40-char SHA.
pub fn resolve_commit_sha(sha: &str) -> Result<String, VvmError> {
    if sha.len() == 40 && sha.chars().all(|c| c.is_ascii_hexdigit()) {
        return Ok(sha.to_string());
    }
    let path = format!("/repos/{}/commits/{}", REPO, sha);
    let commit: CommitResponse = get_json(&path, &format!("Commit '{}'", sha))?;
    Ok(commit.sha)
}

/// Resolve a git tag to its commit SHA. Returns Ok(None) when the tag does
/// not exist (GitHub answers 422/404 for unknown refs on /commits/{ref}),
/// so callers can distinguish "no such release" from transient API failures.
pub fn tag_commit_sha(tag: &str) -> Result<Option<String>, VvmError> {
    let client = build_client()?;
    let url = format!("{}/repos/{}/commits/{}", API_BASE, REPO, tag);
    let resp = with_headers(client.get(&url)).send()?;
    if matches!(resp.status().as_u16(), 404 | 422) {
        return Ok(None);
    }
    if !resp.status().is_success() {
        return Err(response_error(resp, &format!("Tag '{}'", tag)));
    }
    let commit: CommitResponse = resp.json()?;
    Ok(Some(commit.sha))
}

/// Get the HEAD commit SHA for a PR.
pub fn pr_head_commit(pr_number: u64) -> Result<String, VvmError> {
    let path = format!("/repos/{}/pulls/{}", REPO, pr_number);
    let pr: PrResponse = get_json(&path, &format!("PR #{}", pr_number))?;
    Ok(pr.head.sha)
}

/// The commit a merged PR landed on `main` as, or `None` while it is still
/// open or was closed unmerged. GitHub fills `merge_commit_sha` for open PRs
/// too — with the throwaway test-merge commit behind `refs/pull/N/merge`, which
/// exists on no branch and has no wheel — so `merged` has to gate it.
pub fn pr_merge_commit(pr_number: u64) -> Result<Option<String>, VvmError> {
    let path = format!("/repos/{}/pulls/{}", REPO, pr_number);
    let pr: PrResponse = get_json(&path, &format!("PR #{}", pr_number))?;
    Ok(pr.merged_commit())
}

/// Get the HEAD commit SHA for a branch.
pub fn branch_head_commit(branch: &str) -> Result<String, VvmError> {
    let path = format!("/repos/{}/branches/{}", REPO, branch);
    let resp: BranchResponse = get_json(&path, &format!("Branch '{}'", branch))?;
    Ok(resp.commit.sha)
}

/// List releases from GitHub.
pub fn list_releases() -> Result<Vec<Release>, VvmError> {
    let mut all_releases = Vec::new();
    for page in 1..=10 {
        let path = format!("/repos/{}/releases?per_page=100&page={}", REPO, page);
        let releases: Vec<Release> = get_json(&path, "Releases")?;
        if releases.is_empty() {
            break;
        }
        all_releases.extend(releases);
    }
    Ok(all_releases)
}

/// Resolve the latest stable (non-prerelease) release tag, e.g. "v0.7.0".
/// Uses /releases/latest which GitHub already filters to non-prerelease, non-draft.
pub fn latest_stable_release() -> Result<String, VvmError> {
    let path = format!("/repos/{}/releases/latest", REPO);
    let release: Release = get_json(&path, "Latest stable release")?;
    Ok(release.tag_name)
}

/// Get recent commits on a branch up to `count`.
pub fn recent_branch_commits(branch: &str, count: usize) -> Result<Vec<String>, VvmError> {
    let path = format!("/repos/{}/commits?sha={}&per_page={}", REPO, branch, count);
    let commits: Vec<CommitResponse> = get_json(&path, &format!("Commits on '{}'", branch))?;
    Ok(commits.into_iter().map(|c| c.sha).collect())
}

#[derive(Deserialize)]
struct FileEntry {
    filename: String,
}

#[derive(Deserialize)]
struct CommitMessage {
    message: String,
}

#[derive(Deserialize)]
struct CompareCommit {
    sha: String,
    commit: CommitMessage,
}

#[derive(Deserialize)]
struct CompareResponse {
    #[serde(default)]
    files: Vec<FileEntry>,
    #[serde(default)]
    commits: Vec<CompareCommit>,
}

#[derive(Deserialize)]
struct CommitDetailResponse {
    #[serde(default)]
    files: Vec<FileEntry>,
}

/// List files that changed between two refs (base..head, three-dot semantics).
/// `/compare` truncates files at 300; for our coarse csrc/ detection that's fine.
pub fn compare_changed_files(base: &str, head: &str) -> Result<Vec<String>, VvmError> {
    if base == head {
        return Ok(Vec::new());
    }
    let path = format!("/repos/{}/compare/{}...{}", REPO, base, head);
    let resp: CompareResponse = get_json(&path, &format!("Compare {}..{}", base, head))?;
    Ok(resp.files.into_iter().map(|f| f.filename).collect())
}

/// For each commit in base..head, return (sha, subject) for those whose own
/// diff modified at least one file matching `predicate`. Used to attribute
/// range-diff drift to the upstream commits responsible for it, since the
/// requested ref may itself be clean while in-range commits are not.
///
/// One compare call + one detail call per commit in the range. Caller decides
/// the range size (we walk back at most ~30 wheel candidates, so N is bounded).
pub fn commits_in_range_touching<F>(
    base: &str,
    head: &str,
    predicate: F,
) -> Result<Vec<(String, String)>, VvmError>
where
    F: Fn(&str) -> bool,
{
    if base == head {
        return Ok(Vec::new());
    }
    let path = format!("/repos/{}/compare/{}...{}", REPO, base, head);
    let resp: CompareResponse = get_json(&path, &format!("Compare {}..{}", base, head))?;

    let mut result = Vec::new();
    for c in resp.commits {
        let detail_path = format!("/repos/{}/commits/{}", REPO, c.sha);
        let detail: CommitDetailResponse = match get_json(
            &detail_path,
            &format!("Commit {}", &c.sha[..7.min(c.sha.len())]),
        ) {
            Ok(d) => d,
            Err(_) => continue,
        };
        if detail.files.iter().any(|f| predicate(&f.filename)) {
            let subject = c.commit.message.lines().next().unwrap_or("").to_string();
            result.push((c.sha, subject));
        }
    }
    Ok(result)
}

/// List files changed in a PR (paginated).
pub fn pr_changed_files(pr_number: u64) -> Result<Vec<String>, VvmError> {
    let mut all = Vec::new();
    for page in 1..=10 {
        let path = format!(
            "/repos/{}/pulls/{}/files?per_page=100&page={}",
            REPO, pr_number, page
        );
        let files: Vec<FileEntry> = get_json(&path, &format!("PR #{} files", pr_number))?;
        if files.is_empty() {
            break;
        }
        all.extend(files.into_iter().map(|f| f.filename));
    }
    Ok(all)
}

/// Get recent commits on main up to `count`, optionally filtered to commits at
/// or before `until` (ISO 8601).
pub fn recent_main_commits_until(
    until: Option<&str>,
    count: usize,
) -> Result<Vec<String>, VvmError> {
    let path = match until {
        Some(ts) => format!(
            "/repos/{}/commits?sha=main&per_page={}&until={}",
            REPO, count, ts
        ),
        None => format!("/repos/{}/commits?sha=main&per_page={}", REPO, count),
    };
    let commits: Vec<CommitResponse> = get_json(&path, "Commits")?;
    Ok(commits.into_iter().map(|c| c.sha).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(json: &str) -> Option<String> {
        serde_json::from_str::<PrResponse>(json)
            .unwrap()
            .merged_commit()
    }

    #[test]
    fn merged_pr_yields_its_merge_commit() {
        let json = r#"{"head":{"sha":"5a3eba03"},"merged":true,
                       "merge_commit_sha":"aeeb36b1"}"#;
        assert_eq!(parse(json).as_deref(), Some("aeeb36b1"));
    }

    /// An open PR carries a merge_commit_sha too — the throwaway test-merge
    /// behind refs/pull/N/merge, which is on no branch and has no wheel.
    /// Taking it would pin the install to a commit wheels.vllm.ai never built.
    #[test]
    fn open_pr_yields_nothing_despite_merge_commit_sha() {
        let json = r#"{"head":{"sha":"5a3eba03"},"merged":false,
                       "merge_commit_sha":"deadbeef"}"#;
        assert_eq!(parse(json), None);
    }

    /// Older API payloads (and some proxies) omit `merged` entirely.
    #[test]
    fn missing_merged_field_defaults_to_not_merged() {
        let json = r#"{"head":{"sha":"5a3eba03"},"merge_commit_sha":"deadbeef"}"#;
        assert_eq!(parse(json), None);
    }
}
