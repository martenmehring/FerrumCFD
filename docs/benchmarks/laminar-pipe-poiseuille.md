# Laminar Pipe Poiseuille Benchmark

Local benchmark run on 2026-07-08 with WSL OpenFOAM.
All FerrumCFD-facing values are SI. OpenFOAM incompressible pressure is
converted from kinematic pressure to Pa with `rho = 998.2 kg/m3`.

Commands:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_poiseuille_benchmark.ps1 -OpenFoamSteps 200
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_laminar_simple_benchmark.ps1 -SkipOpenFoam -UseExistingOpenFoamJson
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_laminar_pipe_convergence.ps1 -OpenFoamSteps 200
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

## Laminar SIMPLE Path

The first `--solveLaminarSimple` benchmark uses the same medium pipe case,
OpenFOAM-like field files, and SI inputs.

| Source | Mean velocity [m/s] | DeltaP from mean [Pa] | Stored p-field deltaP [Pa] | Error to analytic | Solve/wall time [s] |
| --- | ---: | ---: | ---: | ---: | ---: |
| FerrumCFD laminarSimple | 0.019989 | 1.602328 | 2.143763 | -0.054% | 22.441697 solve |
| OpenFOAM simpleFoam | n/a | 1.6401231 | n/a | 2.303% | 12.7306 wall |

This path is a real finite-volume pressure-velocity assembly bridge, but it is
still a development solver rather than a `simpleFoam` equivalent. The normal
path no longer caps finite `U`, `p`, or `phi` updates. The mean-flow pressure
loss and continuity are now close to the reference, while the stored pressure
field is still too high and is the next pressure-coupling target.

### SIMPLE Solver Experiments

Local experiment on the same medium pipe case:

```powershell
ferrumSolver -case examples\laminar_pipe --solveLaminarSimple --momentumLinearSolver bicgstab --pressureLinearSolver pcg --pressurePreconditioner DIC --maxSimpleIterations 20
ferrumSolver -case examples\laminar_pipe --solveLaminarSimple --momentumLinearSolver bicgstab --pressureLinearSolver pcg --minSimpleIterations 30 --maxSimpleIterations 30
ferrumSolver -case examples\laminar_pipe --solveLaminarSimple --solveTolerance 1e-6 --maxIterations 100 --maxSimpleIterations 20
ferrumSolver -case examples\laminar_pipe --solveLaminarSimple --maxSimpleIterations 20
ferrumSolver -case examples\laminar_pipe --solveLaminarSimple --maxSimpleIterations 80
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
| BiCGStab + diagonal | PCG + diagonal | fvSolution 1e-10/default 10000 | fvSolution 0.7/0.3 + pRef/non-orthogonal pressure loop support | 15 | 1.602328 | -0.054% | 9.808e-11 | 22.441697 | OpenFOAM `smoothSolver` on U now maps to Ferrum `bicgstab`; p-field deltaP remains high at 2.143763 Pa |
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
