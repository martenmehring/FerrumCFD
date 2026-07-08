# Gmsh Pipe Mesh Study

Local benchmark run on the Gmsh pipe mesh study with WSL OpenFOAM.
All FerrumCFD-facing values are SI. OpenFOAM incompressible pressure is
converted from kinematic pressure to Pa with `rho = 998.2 kg/m3`.

Run command:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_gmsh_pipe_mesh_study.ps1 -OpenFoamSteps 1000
```

Reference:

- pipe length: `1.0 m`
- pipe diameter: `0.02 m`
- mean velocity: `0.02 m/s`
- Reynolds number: `398.483`
- analytical Hagen-Poiseuille pressure loss: `1.6032 Pa`

## Result

`boundaryPatchOwnerAverage` samples the owner cells adjacent to inlet and outlet.
That is an owner-cell-center to owner-cell-center pressure difference, not the
full boundary-to-boundary pipe length. For the uniformly extruded pipe variants,
the sampled value is extrapolated by `(axialCells - 1) / axialCells` to estimate
the full-length pressure loss.

| Variant | Axial cells | Cells | OpenFOAM wall [s] | Raw sampled deltaP [Pa] | Full-length deltaP [Pa] | Error to analytic |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| coarse | 16 | 3008 | 12.847 | 1.541368 | 1.644126 | +2.553% |
| medium | 32 | 13760 | 75.533 | 1.572728 | 1.623461 | +1.264% |
| fine | 48 | 38976 | 557.752 | 1.583261 | 1.616948 | +0.858% |

## Interpretation

`medium` is the practical development mesh: it is much faster than `fine` and
already near the analytical solution. `fine` is the current reference candidate
for the first FerrumCFD solver comparison because it is below 1% error after the
sampling-length correction.

A later `veryFine` run is useful before claiming final mesh independence, but it
is not necessary before starting the first FerrumCFD laminar pipe solver.
