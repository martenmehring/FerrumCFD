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

Useful checks:

```powershell
checkFerrumMesh -case examples\laminar_pipe
ferrumSolver -case examples\laminar_pipe --runnerDryRun --maxRunnerSteps 2 --planJson target\laminar_pipe_plan.json
```

No solver kernels are executed yet. This case is a preflight and benchmark
contract for the future flow and heat-transfer solvers.
