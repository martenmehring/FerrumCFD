# Versioning And Releases

FerrumCFD follows Semantic Versioning. The current pre-stable development
version is `0.1.0`.

## Pre-1.0 Versions

- A patch release such as `0.1.1` contains compatible fixes, security
  hardening, validation corrections, and documentation or build repairs.
- A minor release such as `0.2.0` marks a new development milestone and may
  include explicitly documented changes to pre-1.0 case, CLI, or library
  contracts.
- A pre-release such as `0.2.0-alpha.1` identifies an incomplete milestone
  candidate that is not yet a supported release.

## Version 1.0 And Later

Version `1.0.0` establishes the first supported public CLI, case, library,
interoperability, result, and restart contracts. After 1.0, a compatible fix
increments the patch number, a backward-compatible feature increments the
minor number, and an incompatible public-contract change increments the major
number.

## Release Gate

A release tag is created only after:

- all workspace packages report the intended version consistently;
- formatting, clippy, workspace tests, and documented numerical regressions
  pass from a clean checkout;
- the release candidate passes Codex Security change-set review;
- the changelog describes user-visible changes and known limitations;
- license and third-party provenance are reviewed; and
- the reviewed pull request is merged into `main`.

Workflow attempt numbers, roadmap task revisions, and benchmark iterations are
never product versions. The first `v0.1.0` tag remains deferred until the
current release gate is complete.
