# FerrumCFD Solver Roadmap

This roadmap tracks the path from the current executable laminar SIMPLE
prototype toward a production `simpleFoam`-class incompressible laminar solver
and later CPU/GPU solver backends.

## Current Status

`ferrumSolver --solveLaminarSimple` is an executable finite-volume
pressure-velocity solver prototype. It reads OpenFOAM-like `U`, `p`,
`transportProperties`, `fvSchemes`, and `fvSolution`, uses the runtime
`constant/polyMesh` geometry, runs an uncapped SIMPLE correction path, and
writes JSON/Markdown reports including pressure-assembly diagnostics for
`rAU/rAtU`, `HbyA`, `phiHbyA`, pressure source, pressure flux, and corrected
`phi`. Reports also contain a `linearSolves` profile so medium/fine runs expose
whether the bottleneck is the non-symmetric momentum predictor or the pressure
PCG/IC(0) correction. OpenFOAM Foundation-style outer `residualControl` and
linear-solver convergence are reported separately. `converged=true` means the
configured outer field criteria were checked and satisfied; each linear solve
still exposes its own initial/final residual, iterations, and convergence flag.

The 2026-07-10 residual-control validation used a maximum budget of `250` with
`U 1e-3` and `p 1e-2`. The release solver stopped early at SIMPLE iteration
`207`: `U=9.983499e-4`, `p=2.585656e-5`, both final linear solves converged,
and wall-clock solve time was `33.54 s`. This validation is separate from the
analytic pipe benchmark and is recorded in
`docs/benchmarks/laminar-simple-residual-control.md`.

For the medium laminar pipe benchmark (`4608` cells, SI units), using the
matched steady SIMPLE budget OpenFOAM `endTime=100`/`deltaT=1` and Ferrum
`100` SIMPLE iterations:

| Source | DeltaP [Pa] | Error to analytic | Mean U [m/s] | Runtime |
| --- | ---: | ---: | ---: | ---: |
| Analytic Hagen-Poiseuille | 1.603200 | 0.000% | 0.0200000 | n/a |
| Ferrum SIMPLE, pressure owner cells | 1.617532 | 0.894% | 0.0199655 | 144.99 s solve |
| Ferrum SIMPLE, from mean U | 1.600432 | -0.173% | 0.0199655 | 144.99 s solve |
| OpenFOAM `simpleFoam`, pressure owner cells | 1.627046 | 1.487% | n/a | 4.21 s execution / 7.85 s driver wall |

This 2026-07-10 rerun uses the same named-patch owner-cell averaging for Ferrum
and OpenFOAM, with no axial-cell ordering assumption or full-length
extrapolation. `Ferrum SIMPLE, from mean U` is an external benchmark diagnostic:
it back-calculates pressure loss from the simulated mean velocity with the
Hagen-Poiseuille formula. The generic solver report contains neither value.
Ferrum completed 100 iterations but reports `converged=false` because this case
does not yet configure `SIMPLE.residualControl`.

The solver is therefore promising for the pipe case, but it is not yet a
production `simpleFoam` replacement.

## Definition Of Done For The First Solver

The first laminar incompressible solver should be considered ready when it:

- solves arbitrary supported OpenFOAM-like cases without pipe geometry or
  analytic stopping criteria in the normal solver path;
- reports convergence from OpenFOAM-style equation residual controls, not from
  Hagen-Poiseuille error, while keeping continuity visible as a diagnostic;
- keeps generic continuity/residual/field diagnostics stable and lets external
  benchmarks validate direct pressure-field and mean-flow deltaP on
  coarse/medium/fine meshes;
- supports the boundary conditions needed for common inlet/outlet/wall,
  2D, and axisymmetric laminar cases;
- writes final `U`, `p`, residual history, timing, and solver metadata in
  machine-readable and human-readable form;
- has a CPU baseline that is correct and a clear backend contract for later
  GPU acceleration.

## Milestone 1: Numerical Completeness

Goal: make the SIMPLE algorithm converge for pressure and velocity by solver
criteria, not only by benchmark agreement.

- Keep the current uncapped finite update path.
- Tighten pressure-field coupling so stored `p` converges as reliably as
  mean-flow pressure loss.
- Validate `pRefCell`/`pRefValue`, `constrainPressure`, `adjustPhi`, and
  `phi = phiHbyA - pEqn.flux()` on open and closed pressure systems.
- Validate `nNonOrthogonalCorrectors` on skewed/non-orthogonal meshes.
- Keep the implemented OpenFOAM-normalized initial/final residual reporting for
  vector momentum, component momentum, pressure correction, continuity, and
  field changes under regression test.
- Implement OpenFOAM `relTol`, `minIter`, and configurable `smoothSolver`
  `nSweeps`; until then, reject non-default values instead of silently changing
  their meaning.
- Use the new `linearSolves` profile to compare the OpenFOAM-like
  symmetric Gauss-Seidel momentum smoother against explicit BiCGStab experiments, then
  add an ILU/DILU preconditioner before moving the same contract to GPU.
- Add regression gates for the medium pipe case and at least one deliberately
  skewed mesh.

Near-term implementation targets:

- use the pressure-assembly report to compare medium-vs-fine correction terms,
  especially `rAU/rAtU`, `HbyA`, `phiHbyA`, pressure source, pressure flux, and
  boundary contributions;
- record generic continuity/residual-control status and external final `p`
  drop/mean-flow drop as linked but separate reports;
- add tests for pressure reference and `constrainPressure` on closed-pressure
  and open-pressure cases.

## Milestone 2: Boundary Conditions

Goal: read and execute the OpenFOAM-style boundary types expected by a laminar
pipe, 2D, axisymmetric, heat-transfer, and membrane-reactor workflow.

Already started:

- `U`: `fixedValue`, `zeroGradient`, `noSlip`, `inletOutlet`,
  `pressureInletOutletVelocity`;
- `p`: `fixedValue`, `zeroGradient`, `fixedFluxPressure`, `inletOutlet`;
- mesh constraints: `empty`, `wedge`, `symmetryPlane` as solver constraints.

Next boundary-condition targets:

- confirm reverse-flow behavior for `inletOutlet` and
  `pressureInletOutletVelocity` with changing pressure direction;
- add explicit tests for `empty` as true 2D and `wedge` as axisymmetric solver
  constraints;
- add `symmetryPlane`/`slip` handling where appropriate for vector fields;
- document which patch types are executable, parsed-only, or unsupported.

## Milestone 3: Discretization And Operators

Goal: expand the supported OpenFOAM-like finite-volume schemes without hiding
solver instability behind artificial clipping.

Current executable subset:

- `grad(p)`: `Gauss linear`;
- `grad(U)`: `Gauss linear`;
- `div(phi,U)`: `Gauss upwind`, `Gauss linearUpwind grad(U)`;
- `laplacian`: `Gauss linear corrected`, `orthogonal`, `uncorrected`;
- `snGrad`: `corrected`, `orthogonal`, `uncorrected`;
- interpolation: `linear`.

Next scheme targets:

- validate corrected `snGrad` and non-orthogonal flux correction on generated
  skewed meshes;
- add bounded/limited convection schemes as explicit schemes, not hidden
  clamps;
- keep operator assembly independent from CPU/GPU solver backend code;
- add operator-level tests for face orientation, owner/neighbour signs, and
  boundary-face flux signs.

## Milestone 4: Benchmark Matrix

Goal: make correctness measurable with analytic and OpenFOAM references.

Current benchmark:

- medium circular pipe, laminar water, SI units;
- Ferrum SIMPLE vs OpenFOAM `simpleFoam` vs Hagen-Poiseuille;
- generic solver and pipe-reference diagnostics are separate artifacts;
- matched steady pseudo-time comparison is available, for example OpenFOAM
  `endTime=100`/`deltaT=1` against Ferrum `100` SIMPLE iterations.
- The current matched 100-step run gives `0.894%` Ferrum owner-cell error,
  `-0.173%` Ferrum mean-flow error, and `1.487%` OpenFOAM owner-cell
  error with identical sampling.
- Earlier OpenFOAM step-sweep and coarse/medium/fine pressure tables used an
  axial-cell metadata/extrapolation path on the OpenFOAM side. Rerun them before
  using those historical direct-pressure values as acceptance data.
- Ferrum pressure-field iteration sweep shows medium p-owner deltaP improves
  from `9.613%` at 50 SIMPLE iterations to `0.011%` at 200, while fine improves
  from `24.007%` to only `10.654%`; fine therefore needs pressure-coupling or
  discretization work, not only more iterations.
- Pressure-assembly isolation shows fine keeps global mass balance and reaches
  absolute pressure linear residuals around the configured tolerance. The
  pressure/laplacian coefficient now uses
  projected face-normal distance; benchmark validation and stronger pressure
  preconditioning are the next readiness blockers.

Next benchmark targets:

- run coarse/medium/fine with Ferrum SIMPLE and OpenFOAM on the same meshes;
- run matched 100-step comparisons as the default pipe benchmark table before
  increasing OpenFOAM steps for settled reference studies;
- improve or replace the medium OpenFOAM reference setup when an under-1%
  OpenFOAM comparison is required, because more `endTime` alone plateaus;
- require monotonic or explained convergence for Ferrum against
  Hagen-Poiseuille;
- rerun OpenFOAM with enough steps for the fine mesh reference to settle;
- add a skewed pipe mesh and an axisymmetric smoke case;
- add a separate 2D plane-Poiseuille benchmark between parallel plates using
  `empty` front/back patches, one shared Gmsh mesh, Ferrum SIMPLE, OpenFOAM,
  and the analytic parabolic profile/pressure-flow relation;
- store benchmark summaries as JSON/Markdown under `target/benchmarks` and
  keep stable reference snapshots under `docs/benchmarks` when useful.

## Milestone 5: Performance And Backend Policy

Goal: make the CPU solver competitive enough to serve as a baseline, then move
the expensive linear algebra and nonlinear stages onto selectable backends.

Current status:

- Ferrum SIMPLE medium pipe: `144.99 s` solve time in the 2026-07-10
  100-iteration rerun;
- OpenFOAM `simpleFoam`: `4.21 s` solver execution and `7.85 s` driver
  wall time in the matched 100-step rerun;
- CPU pressure PCG now has an IC(0) incomplete-Cholesky path for OpenFOAM
  `DIC`/`FDIC`, replacing the earlier diagonal-only mapping for pressure;
- CG/PCG breakdown tests are scale-relative rather than absolute
  `f64::EPSILON` cutoffs, so valid small SI-scaled pressure systems are not
  terminated prematurely;
- backend policy already supports CPU/GPU/auto declarations and resource
  metadata, but executable GPU equation kernels are still future work.

Next performance targets:

- profile pressure solve cost and matrix assembly cost separately;
- validate IC(0) on medium/fine/skewed pressure matrices and add a true
  nonsymmetric ILU/DILU path for momentum/BiCGStab;
- reduce repeated allocations in SIMPLE history and operator assembly;
- add thread/resource controls for CPU solves where the backend supports them;
- keep GPU optional and selectable per stage (`flow`, linear solves,
  nonlinear/interface/ODE stages), with CPU as a valid choice when GPU is busy
  or unnecessary.

## Milestone 6: Generalization Beyond The Pipe

Goal: turn the pipe solver into a reusable laminar CFD foundation.

Next physics/application targets:

- double-pipe or annular heat-transfer case with constant wall temperature;
- scalar energy equation and wall heat-flux/Nusselt diagnostics;
- multiregion mesh handling for membrane-reactor-style cases;
- interface flux models that allow flow reversal from pressure differences
  instead of encoding physical direction in mesh metadata;
- later: species transport, membrane source terms, nonlinear material laws,
  and optional GPU acceleration for linear, nonlinear, and ODE subproblems.

The membrane-reactor case remains a target application, not a hard-coded solver
assumption. Any pressure reversal, sweep-side backflow, or water transport must
come from equations and boundary/interface models, not from `flipMap` or fixed
mesh orientation choices.

## Immediate Next Steps

1. Build the separate 2D plane-Poiseuille case from a Gmsh `.geo`, import the
   same mesh into Ferrum and OpenFOAM, and keep analytic comparison in an
   external benchmark report.
2. Rerun the medium/fine pressure sweep with the new `pressureAssembly`
   diagnostics and isolate whether the fine-mesh p-owner error starts in
   `phiHbyA`, pressure source assembly, pressure flux, or boundary
   contributions.
3. Validate the projected-distance pressure/laplacian coefficient on the pipe
   mesh and a deliberately skewed mesh, then decide whether corrected
   non-orthogonal fluxes need additional pressure loops by default.
4. Add regression checks for pressure-field deltaP and mean-flow deltaP on the
   medium pipe case.
5. Profile the pressure correction solve and identify the first CPU
   preconditioner improvement.
6. Add one `wedge` axisymmetric smoke case after the 2D benchmark.
7. Improve the OpenFOAM mesh/reference setup when an under-1% OpenFOAM pressure
   loss reference is required.
