# Laminar Pipe Poiseuille Benchmark

Benchmark records from 2026-07-08 through 2026-07-10 with WSL OpenFOAM.
All FerrumCFD-facing values are SI. OpenFOAM incompressible pressure is
converted from kinematic pressure to Pa with `rho = 998.2 kg/m3`.

The 2026-07-10 pipeline separates generic SIMPLE reports from
`ferrumPipeBenchmark` output and samples named inlet/outlet patch owner cells
for both Ferrum and OpenFOAM. Older tables below that use axial-slice
extrapolation are retained as historical records and must not be mixed with the
current direct-pressure comparison.

Commands:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File validation\scripts\incompressibleFluid\run_poiseuille_benchmark.ps1 -OpenFoamSteps 200
powershell -NoProfile -ExecutionPolicy Bypass -File validation\scripts\incompressibleFluid\run_laminar_simple_benchmark.ps1 -SkipOpenFoam -UseExistingOpenFoamJson
powershell -NoProfile -ExecutionPolicy Bypass -File validation\scripts\incompressibleFluid\run_laminar_pipe_convergence.ps1 -OpenFoamSteps 200
```

## Medium Case

Reference input:

- `L = 1 m`
- `D = 0.02 m`
- `mu = 0.001002 Pa s`
- analytical Hagen-Poiseuille pressure loss: `1.6032 Pa`

| Source | DeltaP [Pa] | Error to analytic | Wall/solve time [s] |
| --- | ---: | ---: | ---: |
| FerrumCFD Poiseuille | 1.631086 | 1.739% | 0.006566 solve |
| OpenFOAM simpleFoam | 1.6401231 | 2.303% | 11.0631 wall |

Ferrum currently solves the source-driven axial Stokes/Poiseuille benchmark on
CPU and reconstructs pressure loss from mean velocity. This is not yet the full
SIMPLE-like pressure-velocity solver.

## incompressibleFluid SIMPLE Path

The first `ferrumRun -solver incompressibleFluid` SIMPLE benchmark uses the
same medium pipe case, OpenFOAM-like field files, and SI inputs.

| Source | Mean velocity [m/s] | DeltaP from mean [Pa] | Stored p-field deltaP [Pa] | Error to analytic | Solve/wall time [s] |
| --- | ---: | ---: | ---: | ---: | ---: |
| FerrumCFD incompressibleFluid | 0.019989 | 1.602328 | 2.143763 | -0.054% | 22.441697 solve |
| OpenFOAM simpleFoam | n/a | 1.6401231 | n/a | 2.303% | 12.7306 wall |

This path is a real finite-volume pressure-velocity assembly bridge, but it is
still a development solver rather than a `simpleFoam` equivalent. The normal
path no longer caps finite `U`, `p`, or `phi` updates. The mean-flow pressure
loss and continuity are now close to the reference, while the stored pressure
field is still too high and is the next pressure-coupling target.

### Historical Matched 100-Step SIMPLE Comparison

The table below is the 2026-07-10 `simpleFoam` baseline. The command now uses
OpenFOAM 13 `foamRun -solver incompressibleFluid`; running it creates a new
module-based result and must not silently relabel the historical values below:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File validation\scripts\incompressibleFluid\run_laminar_simple_matched_time_benchmark.ps1 -MatchedTimeSeconds 100
```

For steady SIMPLE solvers, the comparison is an equal pseudo-time/iteration budget rather
than a transient physical-time integration: OpenFOAM uses `endTime=100` with
`deltaT=1`, and Ferrum uses
`minSimpleIterations=maxSimpleIterations=100`.

| Source | DeltaP [Pa] | Error to analytic | Mean U [m/s] | Wall/solve time [s] |
| --- | ---: | ---: | ---: | ---: |
| Analytic Hagen-Poiseuille | 1.6032 | 0.000% | 0.0200000 | n/a |
| FerrumCFD laminar SIMPLE, pressure-owner-cells | 1.6175321 | 0.894% | 0.0199655 | 144.9928 solve / 146.1225 command wall |
| FerrumCFD laminar SIMPLE, from mean U | 1.6004323 | -0.173% | 0.0199655 | 144.9928 solve / 146.1225 command wall |
| OpenFOAM `simpleFoam`, pressure-owner-cells | 1.6270463 | 1.487% | n/a | 4.2124 execution / 7.8476 driver wall |

Both direct-pressure rows use 192 inlet and 192 outlet owner cells. The Ferrum
solver report itself contains only generic fields and residuals; both Ferrum
rows in this table are produced by external post-processing. Ferrum completed
the fixed 100 iterations but reports `converged=false` with
`ConvergenceCriteriaNotConfigured`, because the case has no
`SIMPLE.residualControl`. OpenFOAM remains much faster on CPU.

### OpenFOAM Steps To 1% Error

Historical OpenFOAM-only sweep on 2026-07-09:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File validation\scripts\incompressibleFluid\run_openfoam_laminar_pipe_step_sweep.ps1 -OpenFoamSteps 100,200,400,800,1200 -TargetRelativeError 0.01
```

This answers the inverse question: how long OpenFOAM needs to reach the same
order of pressure-loss error as Ferrum's direct pressure-field comparison
against Hagen-Poiseuille.
These rows used the previous axial-slice/full-length extrapolation and need a
fresh run with the current named-patch owner-cell sampler before they are used
for a current under-1% claim.

| OpenFOAM steps | DeltaP [Pa] | Error to analytic | Execution [s] | Wall [s] |
| ---: | ---: | ---: | ---: | ---: |
| 100 | 1.6977875 | 5.900% | 3.68342 | 6.58617 |
| 200 | 1.6401231 | 2.303% | 11.6697 | 18.0568 |
| 400 | 1.6346116 | 1.959% | 14.7151 | 18.5456 |
| 800 | 1.6345927 | 1.958% | 26.8776 | 32.1153 |
| 1200 | 1.6345927 | 1.958% | 31.0251 | 39.8400 |

No run in this sweep reaches the 1% target. The error plateaus after roughly
400 steps, so on the current medium mesh and OpenFOAM setup this is a
mesh/discretization/sampling limit rather than a lack of additional runtime.
To compare OpenFOAM below 1%, the next step is to improve the OpenFOAM
reference setup or use a finer/better-resolved mesh, not only to extend
`endTime`.

### Laminar SIMPLE Mesh Study

Local Ferrum SIMPLE coarse/medium/fine study on 2026-07-09:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File validation\scripts\incompressibleFluid\run_laminar_simple_mesh_study.ps1 -OpenFoamSteps 400 -FerrumSimpleIterations 100
```

| Variant | Cells | Ferrum p-owner deltaP [Pa] | Ferrum p-owner error | Ferrum mean-U deltaP [Pa] | Ferrum mean-U error | OpenFOAM deltaP [Pa] | OpenFOAM error | Ferrum solve [s] | OpenFOAM execution [s] |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| coarse | 1152 | 1.4991789 | -6.488% | 1.5932245 | -0.622% | 1.6560364 | 3.296% | 23.8588 | 3.04366 |
| medium | 4608 | 1.6183604 | 0.946% | 1.6004125 | -0.174% | 1.6346116 | 1.959% | 141.987 | 17.5703 |
| fine | 12288 | 1.8388040 | 14.696% | 1.6019912 | -0.075% | 1.6445118 | 2.577% | 436.401 | 68.0909 |

Ferrum's mean-flow pressure loss improves monotonically with mesh refinement
and is already within `0.075%` on the fine mesh. The direct stored pressure
field does not converge with mesh refinement: it is good on medium but too low
on coarse and too high on fine. That makes pressure-field coupling and
residual-control robustness the next solver target before using this SIMPLE path
as a finished `simpleFoam` replacement. OpenFOAM also does not show monotonic
pressure-loss convergence in this setup with 400 steps, so the OpenFOAM
reference setup still needs refinement for stricter mesh studies.

### Pressure-Field Iteration Sweep

Local Ferrum-only pressure-field sweep on 2026-07-09:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File validation\scripts\incompressibleFluid\run_laminar_simple_pressure_sweep.ps1 -VariantName medium,fine -SimpleIterations 50,100,200
```

| Variant | Cells | SIMPLE iterations | p-owner deltaP [Pa] | p-owner error | pEqn owner deltaP [Pa] | pEqn owner error | mean-U deltaP [Pa] | mean-U error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| medium | 4608 | 50 | 1.7573149 | 9.613% | 1.7459790 | 8.906% | 1.6004239 | -0.173% |
| medium | 4608 | 100 | 1.6183604 | 0.946% | 1.6183584 | 0.946% | 1.6004125 | -0.174% |
| medium | 4608 | 200 | 1.6033722 | 0.011% | 1.6033722 | 0.011% | 1.6004343 | -0.173% |
| fine | 12288 | 50 | 1.9880827 | 24.007% | 1.9647419 | 22.551% | 1.6019464 | -0.078% |
| fine | 12288 | 100 | 1.8388040 | 14.696% | 1.8387680 | 14.694% | 1.6019912 | -0.075% |
| fine | 12288 | 200 | 1.7740013 | 10.654% | 1.7740013 | 10.654% | 1.6019256 | -0.079% |

This separates two issues. On the medium mesh the direct pressure field mainly
needed more SIMPLE iterations: by 200 iterations it is essentially at the
analytic Hagen-Poiseuille pressure loss. On the fine mesh, additional
iterations help but are not sufficient; the p-field remains more than `10%`
high while the mean-flow diagnostic stays within `0.1%`. That points to a
pressure-correction/coupling or discretization issue on finer meshes, not a
mass-flow problem.

Current laminar SIMPLE JSON/Markdown reports include a `pressureAssembly`
diagnostic block. For the next medium/fine sweep, compare `rAU/rAtU`, `HbyA`,
`phiHbyA` before and after `adjustPhi`, `pressureSource`,
`pressureEquationFlux`, `pressureFlux`, and `correctedPhi` to determine whether
the fine-mesh p-owner error enters through predictor fluxes, pressure-source
assembly, pressure-flux correction, or boundary contributions.

Follow-up isolation on 2026-07-09 with the pressure-assembly reports shows:

- medium is not fundamentally broken; by 200 SIMPLE iterations, p-owner and
  pEqn owner deltaP both reach `1.603372 Pa` (`0.011%` error);
- fine already has correct mass balance and mean-U pressure loss, but its pEqn
  owner value still sits at `1.964742 Pa` after 50 iterations (`22.551%`
  error), so the issue is pressure-velocity coupling/pressure operator behavior
  on the finer mesh rather than a global `adjustPhi` mass-balance failure;
- the absolute pressure linear residual reaches the configured `1e-10` scale,
  while the normalized residual remains higher on fine, so stronger
  pressure preconditioning and operator conditioning are part of the next
  solver-readiness step;
- the diffusion/laplacian coefficient has been corrected to use projected
  face-normal distance instead of raw cell-centre distance; this is geometrically
  closer to OpenFOAM-style corrected laplacian assembly, but still needs a
  settled benchmark rerun before claiming fine-mesh accuracy improvement.
- OpenFOAM `DIC`/`FDIC` now maps to a CPU IC(0) incomplete-Cholesky
  preconditioner for pressure PCG. A 2-SIMPLE-iteration smoke run on the medium
  pipe reports `pressurePreconditioner=incompleteCholesky`, pressure-matrix
  diagnostics, and `424` pressure linear iterations, down from the earlier
  diagonal-preconditioned order of `1133` for the same short run.

### Historical SIMPLE Solver Experiments

The commands below are immutable provenance for experiments recorded before
the public runner migration. They are no longer executable; equivalent new
runs use `ferrumRun -solver incompressibleFluid` without an algorithm-selection
flag.

```powershell
ferrumSolver -case tutorials\incompressibleFluid\laminarPipe\ferrum\case --solveLaminarSimple --momentumLinearSolver bicgstab --pressureLinearSolver pcg --pressurePreconditioner DIC --maxSimpleIterations 20
ferrumSolver -case tutorials\incompressibleFluid\laminarPipe\ferrum\case --solveLaminarSimple --momentumLinearSolver bicgstab --pressureLinearSolver pcg --minSimpleIterations 30 --maxSimpleIterations 30
ferrumSolver -case tutorials\incompressibleFluid\laminarPipe\ferrum\case --solveLaminarSimple --solveTolerance 1e-6 --maxIterations 100 --maxSimpleIterations 20
ferrumSolver -case tutorials\incompressibleFluid\laminarPipe\ferrum\case --solveLaminarSimple --maxSimpleIterations 20
ferrumSolver -case tutorials\incompressibleFluid\laminarPipe\ferrum\case --solveLaminarSimple --maxSimpleIterations 80
```

| Momentum solver | Pressure solver | Linear controls | Relaxation source | SIMPLE tries | DeltaP from mean [Pa] | Error to analytic | Final continuity L2 | Solve time [s] | Notes |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Jacobi | Jacobi | CLI 1e-6/100 | CLI 0.1/0.02 | 13 | 1.584929 | -1.140% | 3.551e-6 | 5.323929 | best pressure-loss error, but local axial velocity oscillates |
| CG | CG | CLI 1e-6/20000 | CLI 0.1/0.02 | 4 | 1.684419 | 5.066% | 4.547e-7 | 0.554151 | fast, pressure correction effectively stalls |
| CG | Jacobi | CLI 1e-6/100 | CLI 0.1/0.02 | 4 | 1.684419 | 5.066% | 4.547e-7 | 0.719120 | confirms the current CG-momentum path is not yet the accuracy bottleneck alone |
| Jacobi | Jacobi | CLI 1e-6/100 | fvSolution 0.7/0.3 | 2 | 1.531687 | -4.461% | 5.547e-7 | 1.208509 | broad CLI tolerance/iteration overrides still affect both equations |
| Jacobi | Jacobi | fvSolution 1e-10/default 10000 | fvSolution 0.7/0.3 | 3 | 1.416486 | -11.646% | 1.797e-5 | 78.180144 | per-equation tolerances are read from `solvers.U/p`; Jacobi pressure correction reaches the guard |
| Jacobi | PCG + diagonal | fvSolution 1e-10/default 10000 | fvSolution 0.7/0.3 + upwind convection + bounded 2% U/p/phi update | 9 | 1.605975 | 0.173% | 6.062e-11 | 56.472017 | upwind momentum convection keeps local U positive and moves U changes from about 9.6% to about 1.9%; convergence stays `no` because the update limiter is still active |
| Jacobi | PCG + diagonal | fvSolution 1e-10/default 10000 | fvSolution 0.7/0.3 + implicit upwind momentum + equation relaxation + pressure-field check | 80 | 1.607913 | 0.294% | 9.504e-9 | 92.977633 | U changes fall to about 1% and the momentum limiter becomes inactive, but convergence stays `no` because pressure-field deltaP is still 2.259878 Pa and pressure updates remain clipped |
| Jacobi | PCG + diagonal | fvSolution 1e-10/default 10000 | fvSolution 0.7/0.3 + absolute `p` solve from `phiHbyA` + full corrected `phi` + bounded U/p update | 80 | 1.604076 | 0.055% | 8.515e-11 | 98.653404 | OpenFOAM-like absolute pressure step greatly improves continuity and mean pressure loss; convergence stays `no` because pressure-field deltaP is still 2.092740 Pa |
| Jacobi | PCG + diagonal | fvSolution 1e-10/default 10000 | fvSolution 0.7/0.3 + absolute `p` solve from `phiHbyA` + uncapped finite U/p/phi updates | 15 | 1.602316 | -0.055% | 8.473e-11 | 15.097480 | normal path no longer clips or rolls back finite SIMPLE updates; stored pressure-field deltaP remains high at 2.143114 Pa |
| BiCGStab + diagonal | PCG + diagonal | fvSolution 1e-10/default 10000 | fvSolution 0.7/0.3 + pRef/non-orthogonal pressure loop support | 15 | 1.602328 | -0.054% | 9.808e-11 | 22.441697 | Historical explicit BiCGStab experiment; the current case selects Ferrum `symGaussSeidel` through OpenFOAM `smoothSolver`; p-field deltaP remained high at 2.143763 Pa |
| BiCGStab + diagonal | PCG + diagonal | fvSolution 1e-10/default 10000 | forced 30 SIMPLE iterations | 30 | 1.603415 | 0.013% | 6.881e-11 | 60.421159 | p-field deltaP improves to 1.910522 Pa, so the stored pressure field is still converging more slowly than the mean-flow metric |

The former continuity-growth and coupled field-update guards are no longer part
of the normal solver path. Ferrum now lets finite SIMPLE updates proceed and
uses convergence checks rather than hidden field clipping. Non-finite fields
still terminate the solve as numerical failure. The absolute `p`/`phiHbyA` run
keeps the local axial velocity positive, drives continuity to `8.473e-11`, and
gets the mean pressure loss to `-0.055%` error. It still remains a
solver-development result rather than a `simpleFoam` equivalent because the
stored pressure field is too high. The next numerical target is tighter
pressure-field coupling, consistent-SIMPLE terms, and residual-control-driven
stopping, then true incomplete preconditioning for pressure and momentum.

## Mesh Study

OpenFOAM used 200 SIMPLE steps per variant. The fine OpenFOAM case is likely
not fully converged at this step count, so increase `-OpenFoamSteps` for a
stricter OpenFOAM convergence reference.

| Variant | Cells | Ferrum deltaP [Pa] | Ferrum error | Ferrum solve [s] | OpenFOAM deltaP [Pa] | OpenFOAM error | OpenFOAM wall [s] |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| coarse | 1152 | 1.670275 | 4.184% | 0.00205 | 1.6560363 | 3.296% | 4.70993 |
| medium | 4608 | 1.631086 | 1.739% | 0.006873 | 1.6401231 | 2.303% | 11.4135 |
| fine | 12288 | 1.621025 | 1.112% | 0.02458 | 1.7353206 | 8.241% | 32.6051 |

Ferrum error decreases from coarse to fine for this benchmark. OpenFOAM error
does not decrease monotonically with only 200 SIMPLE steps, so that column
should be treated as a smoke benchmark until the reference is rerun with a
higher step count.
