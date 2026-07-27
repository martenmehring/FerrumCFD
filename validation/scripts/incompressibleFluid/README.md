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

The portable reference and host-specific `target-cpu=native` builds are
deliberately separate. LTO, reduced codegen units, and PGO remain later,
separately measured optimization lanes:

```powershell
# Portable, reproducible reference.
.\validation\scripts\incompressibleFluid\run_matched_linux_cpu_solver_benchmark.ps1 `
  -Distro Ubuntu-22.04 -CpuSet 2 -PressureSolver gamg -CaseName all `
  -WarmupRuns 2 -MeasuredRuns 9 -BuildVariant portable

# Host-specific target-cpu=native lane; do not merge its claim with portable.
.\validation\scripts\incompressibleFluid\run_matched_linux_cpu_solver_benchmark.ps1 `
  -Distro Ubuntu-22.04 -CpuSet 2 -PressureSolver gamg -CaseName all `
  -WarmupRuns 2 -MeasuredRuns 9 -BuildVariant native
```

The controller rejects DrvFS/`/mnt/c` benchmark roots, verifies the exact source
archive and matched polyMesh hashes, excludes compilation and staging from run
timing, and writes schema-v2 JSON/Markdown below
`target/benchmarks/matched_linux_cpu_solver`. On failure, diagnostic staging is
preserved; successful runs remove temporary staging unless
`-KeepWslWorkspace` is supplied.

For `-PressureSolver gamg`, the driver also enables the diagnostic
`--profileGamg` flag, requires a GAMG timing object in every solve report, and
records aggregate and per-level phase medians. Profiling remains external
validation behavior; it is not copied into `fvSolution` or a tutorial default.
