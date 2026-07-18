# OpenAI Build Week 2026 Development Record

This document records the FerrumCFD work completed with Codex, the decisions
made by the project owner, and the versioned evidence used for the OpenAI Build
Week 2026 submission. Dates and times use Europe/Berlin (CEST).

## Ownership And Roles

FerrumCFD's product direction, physical scope, numerical requirements, and
performance objective are set by project owner Marten Mehring. In particular,
the owner identified runtime performance as the next engineering priority,
required fair fixed-work comparisons, and approved the optimization sequence.

Codex with GPT-5.6 inspected and profiled the implementation, proposed bounded
technical changes, implemented the accepted changes in Rust, and verified them
with numerical and performance regressions. Codex Security is used separately
to scan changes and the full repository for security-relevant defects and
hardening opportunities. It does not define FerrumCFD's product direction.

## Model Timeline

The main FerrumCFD development task initially used Codex with GPT-5.5. Its first
recorded GPT-5.6 work occurred on 2026-07-09 at 21:30:11. After a short model
comparison that evening, sustained GPT-5.6 use began at 23:15:43 and continued
for the recorded development work through 2026-07-17.

The Codex task ID retained for the submission feedback record is
`019f3f02-4ede-7f41-b56b-5fac5e14607a`.

## Collaboration And Optimization Log

| Date | Project-owner direction | Codex contribution | Versioned evidence |
| --- | --- | --- | --- |
| 2026-07-16 | Investigate why FerrumCFD was slower than the external reference and separate correctness from performance work | Distinguished build-mode distortion from solver runtime, profiled solver phases, and proposed a bounded sequence of generic hot-path changes | [CPU performance foundation](benchmarks/cpu-performance-foundation.md) |
| 2026-07-16 to 2026-07-17 | Optimize the generic solver without changing equations, case semantics, or convergence criteria; report both individual and cumulative results | Implemented reusable CSR topology, cached mesh geometry, reusable pressure matrices and workspaces, PCG/IC(0) improvements, GAMG integration and profiling, regression tests, and repeatable measurement tooling | [Solver roadmap](solver-roadmap.md#performance-foundation---scalar-cpu) |
| 2026-07-18 | Integrate the optimization work as one reviewable candidate | Consolidated the Rust implementation, tests, profiles, and documentation | Commit [`0213504`](https://github.com/martenmehring/FerrumCFD/commit/0213504662b2a18e7aaf1b39f617cee6a2752906) |
| 2026-07-18 | Scan the complete optimized codebase and remove avoidable non-Rust runtime requirements | Codex Security identified the residual-plot interpreter boundary; Codex replaced it with native Rust SVG output and added focused regressions | Commit [`af69432`](https://github.com/martenmehring/FerrumCFD/commit/af69432) |

The project owner originated the optimization goal and the measurement rules.
Codex did not originate that goal; it supplied analysis, implementation, tests,
and documentation after the direction was approved.

## Measurement Contract

Every accepted scalar-CPU optimization must:

- use a prebuilt release executable and exclude compilation time;
- compare an unchanged fixed SIMPLE work budget;
- use at least one warmup and five measured runs, reporting the median;
- preserve convergence state, iteration counts, residuals, continuity, and
  field summaries within the documented contract;
- apply to generic solver paths rather than case-specific shortcuts; and
- keep analytical and OpenFOAM 13 reference processing outside Ferrum case
  semantics.

The latest recorded fixed-work medians are `2.2228 s` for the 10-iteration
laminar pipe and `8.1320 s` for the 500-iteration plane channel. Relative to
the original historical Ferrum baselines, these are observed improvements of
`29.13x` and `139.55x`. The original baselines were single-run diagnostics,
whereas the current results are five-run medians, so these figures document
the total observed project improvement rather than a controlled per-change
A/B result. They are not a claim that FerrumCFD is generally faster than
OpenFOAM.

The complete per-change measurements, neutral results, host-load caveats, and
external accuracy comparison remain in the
[CPU performance foundation](benchmarks/cpu-performance-foundation.md).

## Security And Rust-Only Runtime Direction

The 2026-07-18 full-repository Codex Security scan covered the complete
optimization candidate. Its highest-priority actionable boundary was optional
non-SVG residual plotting through Python and Matplotlib. Commit `af69432`
removed that subprocess and its interpreter discovery entirely. Residual CSV
output remains available, and plots are now rendered as SVG by native Rust
code. Python and Matplotlib are not FerrumCFD requirements.

Lower-priority filesystem and input-resource hardening opportunities are kept
as bounded follow-up packages. They are not mixed into numerical changes, and
no unresolved item is represented as fixed before its regression and review
gate passes.

## Publication Record

The optimization candidate is developed on
`codex/ferrum-optimization-integration-20260718`. Its deterministic commits use
the `Project Automation <project-automation@users.noreply.github.com>` identity.
The final pull request, CI result, review outcome, merge commit, and submission
URL will be added here when those events exist; this record does not pre-claim
publication or acceptance.
