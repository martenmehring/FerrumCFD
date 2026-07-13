# Continuous Integration

FerrumCFD uses the `CI` GitHub Actions workflow as its deterministic pull
request gate. It runs for pull requests targeting `main`, pushes to `main`, and
manual dispatches.

The `Rust quality gates` job uses Rust 1.94.0 and mirrors the repository's
release-validation commands:

```text
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --locked --offline -- -D warnings
cargo test --workspace --locked --offline
```

GitHub-hosted runners fetch dependencies before the offline release validator
runs the same locked build graph. OpenFOAM 13 comparisons and longer benchmark
packages remain explicit validation jobs because they require external tools
and curated reference environments.

## Security Boundaries

- The workflow has read-only repository contents permission.
- Checkout credentials are not persisted.
- Third-party actions must be pinned to a full commit SHA. The adjacent version
  comment records the reviewed release.
- The Rust toolchain is pinned. Toolchain upgrades must be deliberate and must
  remain compatible with the deterministic release-validation environment.
- Pull-request data is never interpolated into shell commands.
- A concurrency group cancels superseded runs for the same branch or pull
  request, and the job has a fixed timeout.

## Local Reproduction

Install Rust 1.94.0 with the `rustfmt` and `clippy` components, then run:

```powershell
rustup toolchain install 1.94.0 --profile minimal --component rustfmt --component clippy
cargo +1.94.0 fmt --all -- --check
cargo +1.94.0 fetch --locked
cargo +1.94.0 clippy --workspace --all-targets --locked --offline -- -D warnings
cargo +1.94.0 test --workspace --locked --offline
```

## Main-Branch Gate

After the repository plan supports rules for private repositories and this
workflow has completed successfully, configure `main` to require pull requests
and the exact successful check context reported for `Rust quality gates`.
Require resolved review conversations and block force pushes and branch
deletion. Do not guess or pre-create a status-check name; GitHub must first
observe the real workflow result.
