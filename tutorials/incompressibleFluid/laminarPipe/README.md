# Laminar Pipe Benchmark

This is a FerrumCFD benchmark case for laminar water flow through a straight
circular pipe.

Current purpose:

- exercise `polyMesh` reading on a real circular pipe mesh
- exercise `volScalarField` and `volVectorField` initial field parsing
- materialize both uniform and nonuniform CPU field buffers
- validate generic solved fields with an external Hagen-Poiseuille post-processor

The mesh is a generated structured circular pipe with axial, radial, and angular
resolution. Regenerate it with:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File validation\scripts\incompressibleFluid\generate_laminar_pipe_case.ps1
```

The default reference uses `D = 0.02 m`, `L = 1 m`, mean velocity
`U = 0.02 m/s`, and water near 20 C. FerrumCFD values are SI by default:
pressure is stored in Pa, length in m, temperature in K, and velocity in m/s.
Use explicit units only when a value is not SI.
The analytic inputs are not part of the simulation case. They are stored in
`tutorials/incompressibleFluid/laminarPipe/analytical/pipeBenchmark` and are consumed only by benchmark scripts.

The inlet velocity boundary is a fully developed parabolic profile. The
generator scales the discrete inlet values so the patch-integrated flow matches
`U_mean * inlet_area` for each mesh resolution.

OpenFOAM comparison:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File validation\scripts\incompressibleFluid\run_poiseuille_benchmark.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File validation\scripts\incompressibleFluid\run_laminar_simple_matched_time_benchmark.ps1 -MatchedTimeSeconds 100
powershell -NoProfile -ExecutionPolicy Bypass -File validation\scripts\incompressibleFluid\run_openfoam_laminar_pipe_step_sweep.ps1 -OpenFoamSteps 100,200,400,800,1200 -TargetRelativeError 0.01
powershell -NoProfile -ExecutionPolicy Bypass -File validation\scripts\incompressibleFluid\run_laminar_simple_benchmark.ps1 -SkipOpenFoam -UseExistingOpenFoamJson
powershell -NoProfile -ExecutionPolicy Bypass -File validation\scripts\incompressibleFluid\run_laminar_simple_iteration_sweep.ps1 -SimpleIterations 2,5,10,20,30
powershell -NoProfile -ExecutionPolicy Bypass -File validation\scripts\incompressibleFluid\run_laminar_simple_mesh_study.ps1 -OpenFoamSteps 400 -FerrumSimpleIterations 100
powershell -NoProfile -ExecutionPolicy Bypass -File validation\scripts\incompressibleFluid\run_laminar_simple_pressure_sweep.ps1 -VariantName medium,fine -SimpleIterations 50,100,200
```

The independent source cases live under `ferrum/case` and
`openfoam-v13/case`. The benchmark runner copies the OpenFOAM 13 source
configuration to `target/openfoam/laminar_pipe`, overlays the selected mesh and
initial velocity for mesh studies, then runs
`foamRun -solver incompressibleFluid`. OpenFOAM uses kinematic pressure in
`m2/s2`; the runner converts the result back to Pa using `rho`.
Benchmark JSON/Markdown files are written under `target/benchmarks/`.

Generated benchmark files:

- `target/benchmarks/laminar_pipe_openfoam.json`
- `target/benchmarks/laminar_pipe_compare.json`
- `target/benchmarks/laminar_pipe_compare.md`
- `target/benchmarks/laminar_pipe_compare.ferrum_poiseuille.log`
- `target/benchmarks/laminar_pipe_laminar_simple.json`
- `target/benchmarks/laminar_pipe_laminar_simple.md`
- `target/benchmarks/laminar_pipe_laminar_simple_compare.json`
- `target/benchmarks/laminar_pipe_laminar_simple_compare.md`
- `target/benchmarks/laminar_pipe_laminar_simple_compare.ferrum_laminar_simple.json`
- `target/benchmarks/laminar_pipe_laminar_simple_compare.ferrum_pipe_benchmark.json`
- `target/benchmarks/laminar_pipe_matched_100s.compare.json`
- `target/benchmarks/laminar_pipe_matched_100s.compare.md`
- `target/benchmarks/openfoam_laminar_pipe_step_sweep/openfoam_laminar_pipe_step_sweep.json`
- `target/benchmarks/openfoam_laminar_pipe_step_sweep/openfoam_laminar_pipe_step_sweep.md`
- `target/benchmarks/laminar_simple_iteration_sweep/laminar_simple_iteration_sweep.json`
- `target/benchmarks/laminar_simple_iteration_sweep/laminar_simple_iteration_sweep.md`
- `target/benchmarks/laminar_simple_mesh_study/laminar_simple_mesh_study.json`
- `target/benchmarks/laminar_simple_mesh_study/laminar_simple_mesh_study.md`
- `target/benchmarks/laminar_simple_pressure_sweep/laminar_simple_pressure_sweep.json`
- `target/benchmarks/laminar_simple_pressure_sweep/laminar_simple_pressure_sweep.md`

Historical source-driven Poiseuille mesh convergence (retained only for
reproducibility; current pressure-velocity refinement uses
`run_laminar_simple_mesh_study.ps1`):

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File validation\scripts\incompressibleFluid\run_laminar_pipe_convergence.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File validation\scripts\incompressibleFluid\run_laminar_pipe_convergence.ps1 -OpenFoamSteps 1000
```

The convergence script writes generated cases, OpenFOAM cases, logs, JSON, and
Markdown reports under `target/benchmarks/laminar_pipe_convergence/`. It records
Ferrum Poiseuille pressure-loss error, Ferrum solve time, OpenFOAM pressure-loss
error, and OpenFOAM wall time for each mesh. Increase `-OpenFoamSteps` when a
fine OpenFOAM case still shows moving SIMPLE residuals.

The direct pressure-loss comparison averages owner cells adjacent to the named
`inlet` and `outlet` patches in both Ferrum and OpenFOAM. It does not
assume a particular cell ordering and fails if those patches cannot be sampled.

Useful checks:

```powershell
checkFerrumMesh -case tutorials\incompressibleFluid\laminarPipe\ferrum\case
ferrumRun -solver incompressibleFluid -case tutorials\incompressibleFluid\laminarPipe\ferrum\case --runnerDryRun --maxRunnerSteps 2 --planJson target\laminar_pipe_plan.json
ferrum solve -case tutorials\incompressibleFluid\laminarPipe\ferrum\case --solvePoiseuille --pressureDrop 1.6032 --mu 0.001002 --length 1 --diameter 0.02 --linearSolver cg
ferrumRun -solver incompressibleFluid -case tutorials\incompressibleFluid\laminarPipe\ferrum\case --solveTolerance 1e-6 --maxIterations 100 --solveReportJson target\benchmarks\laminar_pipe_laminar_simple.json --solveReportMarkdown target\benchmarks\laminar_pipe_laminar_simple.md
ferrumRun -solver incompressibleFluid -case tutorials\incompressibleFluid\laminarPipe\ferrum\case --maxSimpleIterations 2 --writeFinalFields target\benchmarks\laminar_pipe_fields\1
ferrumPipeBenchmark -case tutorials\incompressibleFluid\laminarPipe\ferrum\case --fields target\benchmarks\laminar_pipe_fields\1 --pressureDrop 1.6032 --mu 0.001002 --length 1 --diameter 0.02 --axis x --inletPatch inlet --outletPatch outlet
ferrumRun -solver incompressibleFluid -case tutorials\incompressibleFluid\laminarPipe\ferrum\case --momentumLinearSolver bicgstab --pressureLinearSolver pcg --pressurePreconditioner DIC --maxSimpleIterations 20 --solveReportJson target\benchmarks\laminar_pipe_laminar_simple_bicgstab_pcg.json --solveReportMarkdown target\benchmarks\laminar_pipe_laminar_simple_bicgstab_pcg.md
ferrumRun -solver incompressibleFluid -case tutorials\incompressibleFluid\laminarPipe\ferrum\case --minSimpleIterations 30 --maxSimpleIterations 30 --solveReportJson target\benchmarks\laminar_pipe_laminar_simple_30iter.json
```

The full CFD time loop is not implemented yet. This case already executes the
source-driven CPU Poiseuille benchmark and the first laminar SIMPLE path, and
remains the contract for the later flow and heat-transfer solvers. Multi-step
SIMPLE reports use OpenFOAM-style scalar `SIMPLE.residualControl` as the normal
early-convergence path. Continuity stays a reported diagnostic. `U` uses the
maximum normalized initial component residual and `p` uses the first
pressure-solve initial residual; linear initial/final residuals and convergence
flags remain separately visible.
Hagen-Poiseuille agreement cannot stop the solver. The generic report contains
only solver, residual, operator, boundary, and field diagnostics. Stored-field
pressure loss and mean-flow reference error are added later by
`ferrumPipeBenchmark`, which cannot cap, roll back, or force a SIMPLE step. The
current multi-step run computes boundary-constrained `phiHbyA` from `HbyA`,
solves an absolute pressure equation, carries the corrected `phi` into the next
SIMPLE iteration, applies OpenFOAM-like `adjustPhi` on pressure-controlled open
boundaries, reports per-component momentum residuals, `A/H1` ranges, and HbyA
operator diagnostics, corrects velocity as `U = HbyA - rAtU grad(p)`, supports
`pRefCell`/`pRefValue`, `nNonOrthogonalCorrectors`, `SIMPLE.consistent`, and
flux-dependent `inletOutlet`/`pressureInletOutletVelocity` velocity patches,
reads the case `fvSchemes` subset including `div(phi,U) Gauss linearUpwind
grad(U)`, `Gauss linear corrected` laplacians, and `corrected` snGrad,
executes OpenFOAM `smoothSolver` with its configured `symGaussSeidel` momentum smoother,
keeps explicit `bicgstab` available for nonsymmetric experiments, reports
linear-solve convergence profiles, and leaves finite U/p/phi updates uncapped.
With `--writeFinalFields <dir>`, it also writes final OpenFOAM-like `U` and
`p` files to the selected time directory.
