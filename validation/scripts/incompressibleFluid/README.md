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

For `-PressureSolver gamg`, the driver also enables the diagnostic
`--profileGamg` flag, requires a GAMG timing object in every solve report, and
records aggregate and per-level phase medians. Profiling remains external
validation behavior; it is not copied into `fvSolution` or a tutorial default.
