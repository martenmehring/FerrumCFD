# Laminar Pipe Benchmark

This is a small OpenFOAM-like FerrumCFD benchmark case for laminar water flow
through a straight pipe.

Current purpose:

- exercise `polyMesh` reading on a simple pipe-like duct
- exercise `volScalarField` and `volVectorField` initial field parsing
- materialize both uniform and nonuniform CPU field buffers
- keep an analytical Hagen-Poiseuille pressure-loss target next to the case

The mesh is intentionally tiny: 4 cells along the flow direction and one coarse
square surrogate across the section. It is not a production pipe mesh. The
analytical reference in `constant/pipeBenchmark` uses a circular pipe with
`D = 0.02 m`, `L = 1 m`, mean velocity `U = 0.02 m/s`, and water near 20 C.
FerrumCFD values are SI by default: pressure is stored in Pa, length in m,
temperature in K, and velocity in m/s.

OpenFOAM comparison:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_openfoam_laminar_pipe.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\compare_laminar_pipe.ps1
```

The OpenFOAM reference case is generated under `target/openfoam/laminar_pipe`
and benchmark JSON/Markdown files are written under `target/benchmarks/`.
OpenFOAM incompressible solvers use kinematic pressure in `m2/s2`; the script
converts that value back to SI pressure in Pa using `rho`.

Generated benchmark files:

- `target/benchmarks/laminar_pipe_openfoam.json`
- `target/benchmarks/laminar_pipe_compare.json`
- `target/benchmarks/laminar_pipe_compare.md`

The first OpenFOAM comparison uses the same very coarse square surrogate mesh,
so its pressure drop is expected to differ from the circular Hagen-Poiseuille
reference. That difference records mesh/model error separately from future
FerrumCFD solver error.

Useful checks:

```powershell
checkFerrumMesh -case examples\laminar_pipe
ferrumSolver -case examples\laminar_pipe --runnerDryRun --maxRunnerSteps 2 --planJson target\laminar_pipe_plan.json
```

No solver kernels are executed yet. This case is a preflight and benchmark
contract for the future flow and heat-transfer solvers.
