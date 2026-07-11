# Laminar SIMPLE residualControl validation

Date: 2026-07-10

This is a generic solver-convergence validation on the medium laminar pipe
mesh. It is separate from the Hagen-Poiseuille and OpenFOAM accuracy benchmark.

## Configuration

The run used an external copy of `tutorials/incompressibleFluid/laminarPipe/ferrum/case` with:

```text
SIMPLE
{
    nNonOrthogonalCorrectors 0;
    consistent false;

    residualControl
    {
        U 1e-3;
        p 1e-2;
    }
}
```

The release solver received `--maxSimpleIterations 250`. This was only a
maximum budget; `residualControl` was allowed to stop the run earlier.

## Result

| Quantity | Result |
| --- | ---: |
| Maximum SIMPLE iterations | 250 |
| Executed SIMPLE iterations | 207 |
| Stop reason | Converged |
| U initial residual | 9.983499e-4 |
| U tolerance | 1.000000e-3 |
| p initial residual | 2.585656e-5 |
| p tolerance | 1.000000e-2 |
| Final U linear residual | 1.483160e-11 |
| Final p linear residual | 4.110662e-11 |
| Final U linear solve converged | yes |
| Final p linear solve converged | yes |
| Final continuity L2 | 7.171136e-18 |
| Release solve wall clock | 33.542977 s |

Ferrum stopped at iteration 207 because all configured outer field criteria
were satisfied. Continuity and linear-solver convergence were reported but did
not silently add extra outer stopping criteria.

## Residual definition

For each scalar equation Ferrum now uses the OpenFOAM `SolverPerformance`
normalisation shape:

```text
initialResidual = sum(abs(source - A*psi)) / normFactor
```

`normFactor` uses the row-sum reference based on the average initial field, as
in OpenFOAM Foundation 13. `U` is the maximum initial residual of its component
solves. For pressure, outer SIMPLE control uses the first pressure solve in the
iteration. Initial residual, final residual, linear iterations, linear
convergence, and outer convergence are stored separately in JSON, Markdown,
CSV, and console output.
