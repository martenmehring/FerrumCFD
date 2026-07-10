# Changelog

All notable FerrumCFD development milestones are recorded here. The project is
still pre-release; current solver capabilities and remaining production work
are distinguished explicitly.

## Unreleased

### First Executable Flow Solver - 2026-07-10

FerrumCFD now has its first independent finite-volume flow solver. The Rust
`ferrumSolver --solveLaminarSimple` path solves steady, laminar,
incompressible pressure-velocity systems on the generic runtime mesh. Analytic
relations and OpenFOAM comparison logic remain external benchmark tools and do
not influence solver convergence or field updates.

Added:

- OpenFOAM-like case input for `U`, `p`, `transportProperties`, `fvSchemes`,
  `fvSolution`, `controlDict`, and `constant/polyMesh`;
- finite-volume momentum prediction, pressure correction, flux correction,
  velocity correction, equation relaxation, `adjustPhi`, pressure reference,
  and non-orthogonal correction support;
- CPU CSR linear algebra with Gauss-Seidel variants, CG, PCG, BiCGStab,
  diagonal preconditioning, and incomplete-Cholesky IC(0) preconditioning;
- OpenFOAM Foundation-style scalar `SIMPLE.residualControl`, with outer SIMPLE
  convergence kept separate from each linear subsolver's convergence;
- initial and final residuals, iteration counts, continuity diagnostics,
  timing, convergence state, and solver metadata in console, JSON, Markdown,
  CSV, and residual plots;
- OpenFOAM-compatible flow boundary behavior for the currently supported
  `fixedValue`, `zeroGradient`, `noSlip`, `inletOutlet`,
  `pressureInletOutletVelocity`, `fixedFluxPressure`, `empty`, `wedge`, and
  `symmetryPlane` paths;
- external pipe and plane-channel benchmark binaries, with analytic acceptance
  data kept outside generic solver cases;
- reproducible Gmsh import and Ferrum/OpenFOAM benchmark scripts.

Validated:

- the 3D laminar pipe residual-control case stopped at SIMPLE iteration 207 of
  a maximum 250 after satisfying both configured outer field criteria; both
  final linear subsolvers converged;
- the shared 2D plane-channel Gmsh mesh produced a Ferrum mean velocity error
  of 0.1646% relative to plane-Poiseuille flow; OpenFOAM produced 0.4992% in
  the recorded fixed-budget comparison;
- 160 Rust tests, strict Clippy, Rust formatting, PowerShell syntax checks, and
  an end-to-end Gmsh import pass.

Still planned before calling the solver a production `simpleFoam` replacement:

- broader arbitrary-geometry, skewed-mesh, boundary-condition, and
  coarse/medium/fine validation;
- OpenFOAM-compatible `relTol`, `minIter`, and configurable smoother sweeps;
- stronger pressure-momentum coupling and performance optimization;
- parallel CPU execution and executable GPU linear/nonlinear solver backends.

See `docs/solver-roadmap.md` and the reports under `docs/benchmarks` for the
measured results and current definition of done.
