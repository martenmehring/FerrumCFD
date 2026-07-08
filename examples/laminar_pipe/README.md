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

OpenFOAM comparison:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_openfoam_laminar_pipe.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\compare_laminar_pipe.ps1
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

The pressure-loss comparison averages the first and last axial cell slices, so
the result is not tied to a single cell pair in the circular mesh.

Useful checks:

```powershell
checkFerrumMesh -case examples\laminar_pipe
ferrumSolver -case examples\laminar_pipe --runnerDryRun --maxRunnerSteps 2 --planJson target\laminar_pipe_plan.json
```

No solver kernels are executed yet. This case is a preflight and benchmark
contract for the future flow and heat-transfer solvers.
