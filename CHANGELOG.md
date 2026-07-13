# Changelog

All notable FerrumCFD development milestones are recorded here. The project is
still pre-release; current solver capabilities and remaining production work
are distinguished explicitly.

## Unreleased

### Focused Tutorial Scope - 2026-07-13

- Simplified the user-facing tutorial contract to an independently runnable
  Ferrum compatibility case, a native OpenFOAM 13 case, an analytical reference
  when useful, concise English documentation, and optional recorded results.
- Kept existing benchmark scripts and contract tests as maintainer tools while
  removing mandatory master-runner, same-mesh, parameter-hash, and follow-up
  hardening requirements from the physics roadmap.
- Marked `laminarPipe` and `planeChannel` as established cases and selected the
  official OpenFOAM 13 steady laminar `cylinder` tutorial as the next Driver 1
  compatibility target, beginning with its required limited schemes.

### OpenFOAM-13-Inspired Layout And Driver Roadmap - 2026-07-11

- Moved reusable implementation code to `src/` and the current combined
  executable package to `applications/legacy/` without changing Cargo package
  names.
- Added the canonical `ferrumRun -solver incompressibleFluid` public command,
  including `controlDict` solver selection only for cases explicitly marked
  `application ferrumRun`, effective-dispatch provenance in plan JSON, and an
  execution guard requiring steady-state laminar SIMPLE without PISO/PIMPLE or
  RAS/LES configuration.
- Versioned solver-plan JSON as schema 2 and added the effective module plus
  `cli`/`controlDict` selection provenance without overwriting raw case input.
- Updated schema-version-2 solver reports to identify
  `solver=incompressibleFluid`, `algorithm=SIMPLE`, and `regime=laminar`.
- Removed the legacy `ferrumSolver` executable and its algorithm-selection
  flag. Application execution now enters the same internal SIMPLE kernel only
  through `ferrumRun -solver incompressibleFluid`; developer-only scalar and
  Poiseuille utilities remain under `ferrum solve`.
- Added tracked responsibility boundaries for `ferrumRun`, `ferrumMultiRun`,
  `incompressibleFluid`, mesh/case/post-processing utilities, native I/O,
  OpenFOAM interoperability, finite-volume code, core runtime, and models.
- Replaced the tracked `examples/`/`benchmarks/` split with Driver 1 tutorial
  bundles under `tutorials/incompressibleFluid/`.
- Added separate Ferrum, OpenFOAM Foundation 13, shared-geometry, analytical,
  and comparison inputs for `laminarPipe` and `planeChannel`.
- Added independently runnable OpenFOAM 13 source cases using
  `foamRun -solver incompressibleFluid`, and pinned the automated pipe runner
  to `WM_PROJECT_VERSION=13`.
- Defined the six-case steady laminar validation matrix and the fixed
  seven-driver sequence. Porous-media, Ergun, and packed-bed work is explicitly
  deferred until all seven drivers pass the common readiness gate.
- Defined `ferrumMultiRun` as a coupled multi-region runner, not a parameter
  sweep, with a shared CPU/GPU lifecycle, per-region/stage placement, explicit
  interface barriers, mixed-backend operation, and a multi-GPU path.
- Moved PowerShell orchestration from the root `scripts/` directory to
  `validation/scripts/incompressibleFluid/` and documented retention/removal
  criteria so reproducible comparisons are not deleted prematurely.
- Documented `FerrumFile v1` as a target contract while keeping
  the current OpenFOAM-like Ferrum input as an explicit compatibility bridge.

### Command Naming And Licensing - 2026-07-10

- Added the repository-level MIT license.
- Renamed the Gmsh importer from `gmshToFerrumFoam` to `gmshToFerrum`.
- Removed the unbranded Ferrum aliases for the upstream-style `gmshToFoam`,
  `checkMesh`, and `splitMeshRegions` names. Use `gmshToFerrum`,
  `checkFerrumMesh`, and `splitFerrumMeshRegions` instead.
- Standardized FerrumCFD's public command names on lowerCamelCase. Native
  OpenFOAM utility names remain only in migration diagnostics or in
  documentation and scripts that describe or execute the external OpenFOAM
  benchmark toolchain.
- Audited project-maintained documentation, comments, diagnostics, warnings,
  and help text for English-language consistency.

### First Executable Flow Solver - 2026-07-10

FerrumCFD now has its first independent finite-volume flow solver. In the
historical 2026-07-10 milestone it was invoked as
`ferrumSolver --solveLaminarSimple`; the current public command is
`ferrumRun -solver incompressibleFluid`. The implementation solves steady,
laminar, incompressible pressure-velocity systems on the generic runtime mesh. Analytic
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
