# incompressibleFluid Validation Automation

PowerShell orchestration for the Ferrum/OpenFOAM 13/analytical incompressible
flow validation bundles. All generated artifacts are written below `target/`.

The scripts remain together because they call one another through
`$PSScriptRoot`; the common repository root is resolved three levels above
this directory.

`run_cpu_performance_baseline.ps1` accepts `-RunProfile fixed|converged` and
`-PressureSolver pcg|gamg`. It builds the release executable once, copies
solver-specific overlays into disposable cases when required, verifies the
solver reported the requested pressure path, and writes JSON/Markdown below
`target/benchmarks`.

`run_matched_cpu_solver_benchmark.ps1` remains the product-host lane: Ferrum
runs natively on Windows and OpenFOAM may run through WSL. Its historical
schema and timing fields are retained, but that mixed operating-system lane is
not a solver-only speed comparison.

`run_matched_linux_cpu_solver_benchmark.ps1` is the canonical same-Linux lane.
It exports an exact Git commit, stages source, build, matched cases, logs, and
reports on WSL ext4, then runs Linux Ferrum and OpenFOAM Foundation 13 in one
WSL worker. Both engines use the same CPU affinity, serial thread environment,
alternating paired order, and GNU `time` process measurement. Their native
internal timers remain diagnostics only. The default `2` warm-ups and `9`
measured pairs are appropriate for evaluating changes below five percent; a
`0/1` invocation is only a smoke test.

The Linux lane never installs tools. Install and verify exact Rust `1.94.0` in
the selected WSL distribution before running it. A read-only preflight is:

```powershell
.\validation\scripts\incompressibleFluid\run_matched_linux_cpu_solver_benchmark.ps1 `
  -Distro Ubuntu-22.04 `
  -CpuSet 2 `
  -PreflightOnly
```

The portable reference and all host-specific builds are deliberately separate.
`native` sets only `-C target-cpu=native`; `native-codegen1`,
`native-thin-lto`, and `native-fat-lto` add exactly one Cargo release-profile
setting to that native base. The controller clears inherited Rust/Cargo
optimization variables, records the effective settings, and rejects a result
whose returned build metadata differs. PGO is measured only in the separate
Native-PGO lane below:

```powershell
# Portable, reproducible reference.
.\validation\scripts\incompressibleFluid\run_matched_linux_cpu_solver_benchmark.ps1 `
  -Distro Ubuntu-22.04 -CpuSet 2 -PressureSolver gamg -CaseName all `
  -WarmupRuns 2 -MeasuredRuns 9 -BuildVariant portable

# Host-specific target-cpu=native lane; do not merge its claim with portable.
.\validation\scripts\incompressibleFluid\run_matched_linux_cpu_solver_benchmark.ps1 `
  -Distro Ubuntu-22.04 -CpuSet 2 -PressureSolver gamg -CaseName all `
  -WarmupRuns 2 -MeasuredRuns 9 -BuildVariant native

# Isolated native build leaves: native-codegen1, native-thin-lto, native-fat-lto.
.\validation\scripts\incompressibleFluid\run_matched_linux_cpu_solver_benchmark.ps1 `
  -Distro Ubuntu-22.04 -CpuSet 2 -PressureSolver gamg -CaseName laminarPipe `
  -WarmupRuns 1 -MeasuredRuns 5 -BuildVariant native-thin-lto
```

The controller rejects DrvFS/`/mnt/c` benchmark roots, verifies the exact source
archive and matched polyMesh hashes, excludes compilation and staging from run
timing, and writes schema-v2 JSON/Markdown below
`target/benchmarks/matched_linux_cpu_solver`. On failure, diagnostic staging is
preserved; successful runs remove temporary staging unless
`-KeepWslWorkspace` is supplied.

`run_ferrum_linux_pgo_ab_benchmark.ps1` compares two builds of one exact
source commit on WSL ext4: `-C target-cpu=native` and that same native build
with profile-guided optimization. It does not change `Cargo.toml` or the
portable release profile. Exact Rust `1.94.0`, target
`x86_64-unknown-linux-gnu`, and the `llvm-profdata` shipped by that toolchain's
`llvm-tools-preview` component are mandatory; there is no system-tool
fallback. The instrumented binary is trained once, untimed, in fixed order on
the canonical `matched-fixed/gamg` Pipe (10 SIMPLE) and Channel (500 SIMPLE)
cases. Both training runs must add distinct raw profiles before the sorted
inventory is merged with `llvm-profdata merge -sparse`.

The full decision protocol is exactly `2` warm-ups plus `20` alternating,
balanced measured pairs for both cases. PGO must win at least `14/20` pairs,
have a lower median in both order cohorts, and exceed twice the paired-ratio
MAD separately for Pipe and Channel. Canonical solve reports and final `U`/`p`
internal plus boundary IEEE-754 hashes must be exact. A `0+2` run is smoke-only
and always records `decisionEligible=false`; one failed full-protocol case
preserves a reject summary, returns nonzero, and forbids a general/default
claim.

```powershell
# Read-only environment/toolchain check.
.\validation\scripts\incompressibleFluid\run_ferrum_linux_pgo_ab_benchmark.ps1 `
  -Distro Ubuntu-22.04 -CpuSet 2 -PreflightOnly

# Cheap integration smoke; never decision evidence.
.\validation\scripts\incompressibleFluid\run_ferrum_linux_pgo_ab_benchmark.ps1 `
  -SourceRef <exact-commit> -Distro Ubuntu-22.04 -CpuSet 2 `
  -WarmupRuns 0 -MeasuredRuns 2

# Only accepted decision protocol.
.\validation\scripts\incompressibleFluid\run_ferrum_linux_pgo_ab_benchmark.ps1 `
  -SourceRef <exact-commit> -Distro Ubuntu-22.04 -CpuSet 2 `
  -WarmupRuns 2 -MeasuredRuns 20
```

The result under `target/benchmarks/ferrum_linux_native_pgo_ab` binds the
commit/tree/archive and `Cargo.lock`, Rust/LLVM tool identities, raw and merged
profile hashes, all three binary hashes and profiling-section proofs, exact
run order, canonical reports, and field oracles. Exported binaries and merged
profile remain available for independent hash verification.

For `-PressureSolver gamg`, `run_cpu_performance_baseline.ps1` also enables the diagnostic
`--profileGamg` flag, requires a GAMG timing object in every solve report, and
records aggregate and per-level phase medians. Profiling remains external
validation behavior; it is not copied into `fvSolution` or a tutorial default.
The matched Linux timing lane instead requires GAMG profiling to remain
disabled.

`run_ferrum_linux_ref_ab_benchmark.ps1` is the fail-closed same-Linux lane for
one isolated Ferrum change. The candidate must be the direct single-parent
child of the baseline, the diff must contain exactly `-ExpectedChangedPath`,
and both commits must carry the identical `Cargo.lock` blob. The worker builds
the two SHA-bound Git archives in separate symmetric ext4 paths with identical
sanitized settings and `CARGO_INCREMENTAL=0`, then alternates baseline and
candidate on the same CPU. `-MeasuredRuns` must be even so candidate-first and
candidate-second cohorts are balanced. Timing runs do not profile or write
fields; a separate untimed run strictly parses final `U` and `p` and requires
their IEEE-754 hashes to match. Canonical solve reports must also be identical
after removing only path and seconds fields. A performance classification is
reported only when both order cohorts agree:

```powershell
.\validation\scripts\incompressibleFluid\run_ferrum_linux_ref_ab_benchmark.ps1 `
  -BaselineRef <exact-baseline> -CandidateRef <exact-direct-child> `
  -ExpectedChangedPath src/ferrumMesh/src/flow.rs `
  -Distro Ubuntu-22.04 -CpuSet 2 -PressureSolver gamg -CaseName all `
  -WarmupRuns 2 -MeasuredRuns 10 -BuildVariant portable
```
