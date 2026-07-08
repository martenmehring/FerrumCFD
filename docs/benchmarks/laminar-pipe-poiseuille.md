# Laminar Pipe Poiseuille Benchmark

Local benchmark run on 2026-07-08 with WSL OpenFOAM.
All FerrumCFD-facing values are SI. OpenFOAM incompressible pressure is
converted from kinematic pressure to Pa with `rho = 998.2 kg/m3`.

Commands:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_poiseuille_benchmark.ps1 -OpenFoamSteps 200
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
