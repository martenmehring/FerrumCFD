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

## Guarded Laminar SIMPLE Path

The first `--solveLaminarSimple` benchmark uses the same medium pipe case,
OpenFOAM-like field files, and SI inputs. Current default settings are one
damped Jacobi CPU SIMPLE step with `--solveTolerance 1e-6` and
`--maxIterations 100`.

| Source | Mean velocity [m/s] | DeltaP from mean [Pa] | Error to analytic | Solve/wall time [s] |
| --- | ---: | ---: | ---: | ---: |
| FerrumCFD laminarSimple | 0.0191079 | 1.531687 | -4.461% | 0.468459 solve |
| OpenFOAM simpleFoam | n/a | 1.6401231 | 2.303% | 12.7306 wall |

This path is a real finite-volume pressure-velocity assembly bridge, but it is
still guarded. Multi-step SIMPLE correction and CG/PCG momentum solves are the
next numerical-stability targets before treating it as a `simpleFoam`
equivalent.

### SIMPLE Solver Experiments

Local experiment on the same medium pipe case:

```powershell
ferrumSolver -case examples\laminar_pipe --solveLaminarSimple --linearSolver jacobi --solveTolerance 1e-6 --maxIterations 100 --maxSimpleIterations 20 --velocityRelaxation 0.1 --pressureRelaxation 0.02
ferrumSolver -case examples\laminar_pipe --solveLaminarSimple --linearSolver cg --solveTolerance 1e-6 --maxIterations 20000 --maxSimpleIterations 20 --velocityRelaxation 0.1 --pressureRelaxation 0.02
ferrumSolver -case examples\laminar_pipe --solveLaminarSimple --linearSolver jacobi --momentumLinearSolver cg --pressureLinearSolver jacobi --solveTolerance 1e-6 --maxIterations 100 --maxSimpleIterations 20 --velocityRelaxation 0.1 --pressureRelaxation 0.02
ferrumSolver -case examples\laminar_pipe --solveLaminarSimple --solveTolerance 1e-6 --maxIterations 100 --maxSimpleIterations 20
ferrumSolver -case examples\laminar_pipe --solveLaminarSimple --maxSimpleIterations 20
```

| Momentum solver | Pressure solver | Linear controls | Relaxation source | SIMPLE tries | DeltaP from mean [Pa] | Error to analytic | Final continuity L2 | Solve time [s] | Notes |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Jacobi | Jacobi | CLI 1e-6/100 | CLI 0.1/0.02 | 13 | 1.584929 | -1.140% | 3.551e-6 | 5.323929 | best pressure-loss error, but local axial velocity oscillates |
| CG | CG | CLI 1e-6/20000 | CLI 0.1/0.02 | 4 | 1.684419 | 5.066% | 4.547e-7 | 0.554151 | fast, pressure correction effectively stalls |
| CG | Jacobi | CLI 1e-6/100 | CLI 0.1/0.02 | 4 | 1.684419 | 5.066% | 4.547e-7 | 0.719120 | confirms the current CG-momentum path is not yet the accuracy bottleneck alone |
| Jacobi | Jacobi | CLI 1e-6/100 | fvSolution 0.7/0.3 | 2 | 1.531687 | -4.461% | 5.547e-7 | 1.208509 | broad CLI tolerance/iteration overrides still affect both equations |
| Jacobi | Jacobi | fvSolution 1e-10/default 10000 | fvSolution 0.7/0.3 | 3 | 1.416486 | -11.646% | 1.797e-5 | 78.180144 | per-equation tolerances are read from `solvers.U/p`; Jacobi pressure correction reaches the guard |
| Jacobi | PCG + diagonal | fvSolution 1e-10/default 10000 | fvSolution 0.7/0.3 | 3 | 1.596293 | -0.431% | 9.295e-11 | 23.478386 | corrected `phi` is carried between SIMPLE iterations; convergence stays `no` because U/p field changes are still too large and the reference-error guard rolls back iteration 3 |

The continuity-growth guard prevents the old runaway behavior where long
multi-step trials produced infinite or astronomically large values. The
multi-step guard now also refuses convergence when the Hagen-Poiseuille
pressure-drop reference or the relative U/p field changes are not stable. That
keeps the PCG pressure-correction run honest: its mean pressure loss is close
to analytic after two accepted steps, but the local velocity field is still
oscillatory, so the run remains a guarded solver-development result rather than
a `simpleFoam` equivalent. The next numerical target is a better
pressure-correction operator, bounded momentum update, and true
incomplete-Cholesky-style pressure preconditioning.

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
