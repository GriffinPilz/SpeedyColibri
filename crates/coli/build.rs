//! Stamp the build with a source revision, exposed as `COLI_BUILD_REV`.
//!
//! Deployments run `coli` from a container image, and an image built from stale
//! source is indistinguishable from a fresh one at runtime — a trap that cost real
//! debugging time (a node quietly ran old code because its build predated a fix).
//! `coli version` and the `serve`/`worker` startup banners print this, so image drift
//! is visible instead of mysterious, and two nodes can be compared at a glance.
//!
//! Resolution order:
//!   1. `COLI_GIT_SHA` from the environment — the Docker path. `.dockerignore`
//!      excludes `.git`, so the image build has no repo to query; the Dockerfile
//!      takes the sha as a build arg and passes it in.
//!   2. `git rev-parse` in the source tree, with a `-dirty` marker for uncommitted
//!      changes — the local `cargo build` path.
//!   3. `unknown` — no git, no arg. Honest rather than fabricated.

use std::process::Command;

fn main() {
    println!("cargo:rerun-if-env-changed=COLI_GIT_SHA");
    // Re-stamp when the checked-out commit changes (HEAD) or the index does (which
    // is what flips the -dirty marker in practice).
    for p in [".git/HEAD", ".git/index"] {
        let repo = std::path::Path::new("../..").join(p);
        if repo.exists() {
            println!("cargo:rerun-if-changed={}", repo.display());
        }
    }

    let rev = std::env::var("COLI_GIT_SHA")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .or_else(git_rev)
        .unwrap_or_else(|| "unknown".to_string());

    println!("cargo:rustc-env=COLI_BUILD_REV={rev}");
}

/// `<short-sha>` or `<short-sha>-dirty`, or `None` outside a git checkout.
fn git_rev() -> Option<String> {
    let out = Command::new("git").args(["rev-parse", "--short", "HEAD"]).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let sha = String::from_utf8(out.stdout).ok()?.trim().to_string();
    if sha.is_empty() {
        return None;
    }
    // `--quiet` exits non-zero when the working tree has changes.
    let clean = Command::new("git")
        .args(["diff", "--quiet", "HEAD"])
        .status()
        .map(|s| s.success())
        .unwrap_or(true);
    Some(if clean { sha } else { format!("{sha}-dirty") })
}
