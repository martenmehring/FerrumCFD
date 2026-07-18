# Incompressible-flow validation profiles

These files are validation-only overlays for versioned Ferrum tutorial cases.
They keep benchmark stopping criteria and iteration budgets outside the generic
`incompressibleFluid` solver and outside the source tutorial defaults.

Each profile mirrors the OpenFOAM case layout. A validation runner copies the
tutorial case into `target/`, then overlays the profile's `system` directory.
The resulting working case is disposable and must not be used as a solver
fallback or a hidden case-specific default.

Profiles:

- `laminarPipe/converged`: PCG, at most 250 SIMPLE iterations, with
  `residualControl` of `U=1e-3` and `p=1e-2`.
- `planeChannel/converged`: PCG, at most 600 SIMPLE iterations, with
  `residualControl` of `U=1e-5` and `p=1e-5`.
- `laminarPipe/gamg-fixed` and `planeChannel/gamg-fixed`: replace only
  `fvSolution` for fixed-work GAMG comparisons.
- `laminarPipe/gamg-converged` and `planeChannel/gamg-converged`: combine the
  same outer convergence criteria with pressure `GAMG`, `faceAreaPair`, and
  `symGaussSeidel`.

The CPU performance runner selects these overlays explicitly with
`-PressureSolver pcg` or `-PressureSolver gamg`. PCG remains the default.
