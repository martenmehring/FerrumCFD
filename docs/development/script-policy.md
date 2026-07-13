# Validation Script Policy

Stable Rust executables are the end-user interface. PowerShell files under
`validation/scripts/` exist only to reproduce Ferrum/OpenFOAM/analytical
comparisons, generate controlled validation cases, or investigate an open
numerical readiness issue. They are not part of the public solver API.

## Current Incompressible-Flow Inventory

| Category | Scripts | Disposition |
| --- | --- | --- |
| Canonical comparison | `compare_laminar_pipe`, `run_openfoam_laminar_pipe`, `run_poiseuille_benchmark`, `run_laminar_simple_matched_time_benchmark` | Keep until equivalent Rust validation commands exist |
| Reference convergence | `run_openfoam_laminar_pipe_step_sweep`, `run_laminar_simple_iteration_sweep`, `run_laminar_simple_mesh_study`, `run_laminar_simple_pressure_sweep` | Keep while their readiness questions remain open |
| Reproducible case preparation | `generate_laminar_pipe_case`, `prepare_plane_channel_case` | Keep until native case tooling provides parity |
| Transitional smoke/wrapper | `run_gmsh_pipe_import`, `run_laminar_simple_benchmark` | Remove after direct commands and documentation replace them |
| Historical Poiseuille studies | `run_laminar_pipe_convergence`, `run_gmsh_pipe_mesh_study` | Archive or remove after current-driver regressions supersede them |

No script is deleted merely to make the tree shorter. Deletion requires that
its reproducibility or regression coverage exists in a stable Rust command,
test, or retained reference artifact and that all documentation links have
been migrated.

## Maintenance Guidance

1. Make `ferrumRun -solver incompressibleFluid` the only documented solver
   entry point.
2. Keep script inventories in development documentation rather than tutorial
   user workflows.
3. Remove a thin wrapper only after direct commands preserve its useful
   behavior and no maintained document depends on it.
4. Consolidate validation helpers only when doing so reduces real maintenance;
   no master Rust or PowerShell runner is required.
5. Archive historical studies when their recorded benchmark evidence is
   sufficient and the scripts no longer answer an active engineering question.

Generated meshes, time directories, logs, plots, JSON, and Markdown reports
must stay below `target/`; only source cases, scripts, and stable reference
summaries are versioned.
