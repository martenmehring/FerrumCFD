# Incompressible-flow validation profiles

These files are validation-only overlays for versioned Ferrum tutorial cases.
They keep benchmark stopping criteria and iteration budgets outside the generic
`incompressibleFluid` solver and outside the source tutorial defaults.

Each profile contains only Ferrum-owned configuration data. A maintainer may
copy a tutorial case into an untracked directory and overlay the profile's
`system` directory. The resulting working case is disposable and must not be
used as a solver fallback or a hidden case-specific default.

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
- `cylinder/c3`: the bounded physical-acceptance profile for the generated
  `Coarse` and `Fine` Cylinder O-grids. It allows at most 5,000 SIMPLE
  iterations, requires `U=1e-5` and `p=1e-5` residual control, and retains one
  corrected non-orthogonal pressure pass. Its Coarse/Coarse/Fine release gate
  passed on 2026-07-29.

These profiles document accepted validation controls; they are not executable
launchers and add no user dependency. PCG remains the default.
