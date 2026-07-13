# Laminar 2D Plane-Channel Benchmark

Date: 2026-07-10

This benchmark compares Ferrum, OpenFOAM, and the analytic plane-Poiseuille
solution. Solver execution, OpenFOAM execution, and analytic post-processing
are separate artifacts.

## Shared Setup

- Gmsh source: `tutorials/incompressibleFluid/planeChannel/shared/geometry/plane_channel.geo`
- shared imported mesh: 2000 hex cells, 100 axial x 20 wall-normal x 1 depth
- `front` and `back`: OpenFOAM `empty`
- length: `1 m`
- full plate gap: `0.02 m`
- mesh depth: `0.001 m`
- water density: `998.2 kg/m3`
- dynamic viscosity: `0.001002 Pa s`
- imposed boundary pressure difference: `0.6012 Pa`
- analytic mean velocity: `0.02000000 m/s`
- schemes and relaxation: the same case `fvSchemes` and `fvSolution`

The analytic relation is:

```text
meanU = deltaP*H^2/(12*mu*L)
```

## Results

| Source | SIMPLE iterations | Outer converged | Final linear solves | Mean U [m/s] | Mean-U error | DeltaP from mean U [Pa] | Mean-U DeltaP error | Owner-cell DeltaP [Pa] | Owner-cell error | Runtime |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Analytic | n/a | n/a | n/a | 0.02000000 | 0.0000% | 0.6012000 | 0.0000% | n/a | n/a | n/a |
| Ferrum release | 545 | yes | U=yes, p=yes | 0.02003293 | +0.1646% | 0.6021898 | +0.1646% | 0.5924641 | -1.4531% | 23.4258 s |
| Historical OpenFOAM `simpleFoam` baseline | 1000 | no | Ux/Uy/p=yes | 0.02009984 | +0.4992% | 0.6042012 | +0.4992% | 0.5951878 | -1.0000% | 7.7109 s execution / 14.9744 s WSL wall |

A full matched OpenFOAM 13 `foamRun -solver incompressibleFluid` plane-channel
result has not yet replaced that historical row.

Ferrum stopped before its maximum budget of 600 with final outer initial
residuals `U=9.974217e-6` and `p=4.210363e-8` for tolerances `1e-5`.
OpenFOAM reached its maximum of 1000 iterations: its final `Ux` and `p`
initial residuals were below `1e-5`, but vector `U` was not outer-converged
because `Uy=4.690712e-3`. The OpenFOAM row is therefore a fixed-budget result,
not a claim of outer convergence.

Owner-cell pressure sampling averages cells adjacent to the named inlet and
outlet patches. With 100 uniform axial cells, these centres span about 99% of
the boundary-to-boundary length, so owner-cell DeltaP is expected to differ
from the imposed boundary DeltaP. Mean-velocity error is the cleaner analytic
metric for this first plane-channel study.

## Current Finding

Both solvers reproduce the analytic mean velocity below 1% on the same Gmsh
mesh. Ferrum is closer in this run (`0.1646%` versus `0.4992%`) but slower than
OpenFOAM. This table is a recorded result, not a required combined user
workflow. Future maintainers may append separately executed results when they
need additional evidence such as a wall-normal velocity-profile norm.
