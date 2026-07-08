# Laminar Pipe Benchmark

This is a FerrumCFD benchmark case for laminar water flow through a straight
circular pipe.

Current purpose:

- exercise `polyMesh` reading on a real circular pipe mesh
- exercise `volScalarField` and `volVectorField` initial field parsing
- materialize both uniform and nonuniform CPU field buffers
- keep an analytical Hagen-Poiseuille pressure-loss target next to the case

The mesh is a generated structured circular pipe with axial, radial, and angular
resolution. Regenerate it with:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\generate_laminar_pipe_case.ps1
```

The default reference uses `D = 0.02 m`, `L = 1 m`, mean velocity
`U = 0.02 m/s`, and water near 20 C. FerrumCFD values are SI by default:
pressure is stored in Pa, length in m, temperature in K, and velocity in m/s.
Use explicit units only when a value is not SI.

The inlet velocity boundary is a fully developed parabolic profile. The
generator scales the discrete inlet values so the patch-integrated flow matches
`U_mean * inlet_area` for each mesh resolution.

OpenFOAM comparison:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_poiseuille_benchmark.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_laminar_simple_benchmark.ps1 -SkipOpenFoam -UseExistingOpenFoamJson
```

The OpenFOAM reference case is generated under `target/openfoam/laminar_pipe`.
It is only a comparison/benchmark artifact, not the normal FerrumCFD case
workflow. Benchmark JSON/Markdown files are written under `target/benchmarks/`.
OpenFOAM incompressible solvers use kinematic pressure in `m2/s2`; the script
converts Ferrum's SI pressure field to kinematic pressure for OpenFOAM and
converts the result back to Pa using `rho`.

Generated benchmark files:

- `target/benchmarks/laminar_pipe_openfoam.json`
- `target/benchmarks/laminar_pipe_compare.json`
- `target/benchmarks/laminar_pipe_compare.md`
- `target/benchmarks/laminar_pipe_compare.ferrum_poiseuille.log`
- `target/benchmarks/laminar_pipe_laminar_simple.json`
- `target/benchmarks/laminar_pipe_laminar_simple.md`
- `target/benchmarks/laminar_pipe_laminar_simple_compare.json`
- `target/benchmarks/laminar_pipe_laminar_simple_compare.md`

Mesh convergence:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_laminar_pipe_convergence.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_laminar_pipe_convergence.ps1 -OpenFoamSteps 1000
```

The convergence script writes generated cases, OpenFOAM cases, logs, JSON, and
Markdown reports under `target/benchmarks/laminar_pipe_convergence/`. It records
Ferrum Poiseuille pressure-loss error, Ferrum solve time, OpenFOAM pressure-loss
error, and OpenFOAM wall time for each mesh. Increase `-OpenFoamSteps` when a
fine OpenFOAM case still shows moving SIMPLE residuals.

The pressure-loss comparison averages the first and last axial cell slices, so
the result is not tied to a single cell pair in the circular mesh.

Useful checks:

```powershell
checkFerrumMesh -case examples\laminar_pipe
ferrumSolver -case examples\laminar_pipe --runnerDryRun --maxRunnerSteps 2 --planJson target\laminar_pipe_plan.json
ferrumSolver -case examples\laminar_pipe --solvePoiseuille --linearSolver cg
ferrumSolver -case examples\laminar_pipe --solveLaminarSimple --solveTolerance 1e-6 --maxIterations 100 --solveReportJson target\benchmarks\laminar_pipe_laminar_simple.json --solveReportMarkdown target\benchmarks\laminar_pipe_laminar_simple.md
ferrumSolver -case examples\laminar_pipe --solveLaminarSimple --maxSimpleIterations 20 --solveReportJson target\benchmarks\laminar_pipe_laminar_simple_coupled_pcg_dic.json --solveReportMarkdown target\benchmarks\laminar_pipe_laminar_simple_coupled_pcg_dic.md
```

The full CFD time loop is not implemented yet. This case already executes the
source-driven CPU Poiseuille benchmark and the first guarded laminar SIMPLE
path, and remains the contract for the later flow and heat-transfer solvers.
Multi-step SIMPLE reports are intentionally conservative: continuity, pressure
loss, and U/p field changes must all stabilize before Ferrum marks the run as
converged, and the coupled update limiter must no longer be clipping the step.
The default multi-step run also bounds each coupled U/p/phi update to 2% to
keep the intermediate velocity field finite and physically inspectable while
the pressure-velocity coupling is still under development.
