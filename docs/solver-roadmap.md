# FerrumCFD Solver Roadmap

This roadmap first completes and broadens the steady laminar incompressible
foundation, then implements exactly six additional application drivers in a
fixed order. Every driver is validated with independent Ferrum and OpenFOAM 13
cases and an analytical, manufactured, or documented benchmark reference.
Porous-media, Ergun, and packed-bed development starts only after all seven
application drivers have passed their readiness gates.

## Current Status

The canonical public entry point is now
`ferrumRun -solver incompressibleFluid`. It dispatches the executable
finite-volume pressure-velocity prototype only for unambiguous steady-state
laminar cases with exactly one SIMPLE section and no PISO/PIMPLE section.
Explicit `momentumTransport`/`turbulenceProperties` input must select
`simulationType laminar`; RAS/LES is not dispatched to the laminar kernel.
The older `ferrumSolver --solveLaminarSimple` spelling remains only as a
temporary compatibility and benchmark interface. The implementation reads
OpenFOAM-like `U`, `p`,
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

The following medium-pipe table (`4608` cells, SI units) is the historical
matched `simpleFoam` baseline recorded before the comparison runner migrated
to OpenFOAM 13 `foamRun -solver incompressibleFluid`. It preserves provenance
and must not be relabeled as a `foamRun` result. The current module-based
reference still needs a full matched-budget regeneration.

| Source | DeltaP [Pa] | Error to analytic | Mean U [m/s] | Runtime |
| --- | ---: | ---: | ---: | ---: |
| Analytic Hagen-Poiseuille | 1.603200 | 0.000% | 0.0200000 | n/a |
| Ferrum SIMPLE, pressure owner cells | 1.617532 | 0.894% | 0.0199655 | 144.99 s solve |
| Ferrum SIMPLE, from mean U | 1.600432 | -0.173% | 0.0199655 | 144.99 s solve |
| OpenFOAM `simpleFoam`, pressure owner cells | 1.627046 | 1.487% | n/a | 4.21 s execution / 7.85 s driver wall |

This historical 2026-07-10 rerun uses the same named-patch owner-cell averaging for Ferrum
and OpenFOAM, with no axial-cell ordering assumption or full-length
extrapolation. `Ferrum SIMPLE, from mean U` is an external benchmark diagnostic:
it back-calculates pressure loss from the simulated mean velocity with the
Hagen-Poiseuille formula. The generic solver report contains neither value.
Ferrum completed 100 iterations but reports `converged=false` because this case
does not yet configure `SIMPLE.residualControl`.

The solver is therefore promising for the pipe case, but it is not yet a
production `simpleFoam` replacement.

## Definition Of Done For Driver 1

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
- historical Ferrum SIMPLE vs OpenFOAM `simpleFoam` vs Hagen-Poiseuille, with
  the current `foamRun -solver incompressibleFluid` regeneration pending;
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
- historical OpenFOAM `simpleFoam`: `4.21 s` solver execution and `7.85 s`
  driver wall time in the matched 100-step rerun; do not use this as current
  `foamRun` performance data;
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

## Milestone 6: Driver 1 Laminar Validation Matrix

Before Driver 2 starts, steady incompressible SIMPLE/SIMPLEC must pass this
tutorial matrix:

| Order | Case | Primary coverage | Reference |
| ---: | --- | --- | --- |
| 1 | `laminarPipe` | 3D internal flow and pressure loss | Hagen-Poiseuille analytical solution |
| 2 | `planeChannel` | true 2D `empty` handling | Plane-Poiseuille analytical solution |
| 3 | `couettePoiseuille` | moving wall and combined pressure/shear forcing | Analytical velocity profile |
| 4 | `lidDrivenCavity` | recirculation and closed-pressure reference | Published benchmark |
| 5 | `backwardFacingStep` | separation, reattachment, and outlet robustness | Published benchmark |
| 6 | `axisymmetricPipe` | `wedge` handling | Hagen-Poiseuille analytical solution |

Every case contains independently runnable `ferrum/` and `openfoam-v13/`
directories. Analytic cases also contain `analytical/`; benchmark-only cases
contain `benchmark/`. Coarse/medium/fine, skewed, and non-orthogonal variants
belong to these bundles instead of becoming unrelated cases.

## Runner And Multi-Region Milestone

The solver lifecycle must be shared by both public dispatchers:

- `ferrumRun`: one region and one runtime-selected module;
- `ferrumMultiRun`: multiple coupled regions and one module per region.

`ferrumMultiRun` follows OpenFOAM 13 `foamMultiRun` semantics. It is not an
independent-case batch runner and has no `-solver` option. Region-to-module
selection comes entirely from the case. It advances regions through one
capability/dependency graph, applies a global time-step limited by global
constraints and all active transient regions, and declares convergence only
after all participating region criteria pass. Mixed steady/transient regions
are an explicit supported scheduling mode.

Implementation order:

1. extract a module registry and common solver lifecycle while completing
   `ferrumRun`;
2. define a backend-neutral execution context that distinguishes sockets,
   cores, worker threads, process ranks, domain partitions, GPU devices,
   memory, and queues while preventing oversubscription;
3. implement a deterministic CPU scheduler with a capability/dependency graph,
   rank/partition mapping, halo/ghost exchange, interface barriers, and failure
   propagation;
4. add per-region and per-stage CPU/GPU placement, a data-residency/transfer
   graph, backend capability checks including `f64`, and mixed-backend parity
   tests;
5. add multi-GPU placement, deterministic cross-device reductions, and
   conservative region/partition interface exchange;
6. require CPU/GPU, mixed-backend, and multi-GPU parity plus
   mass/energy/species conservation at every coupled interface with stated
   tolerances.

The lifecycle and backend contracts are established during Drivers 1 and 2 so
GPU support does not require a later architectural rewrite. A working coupled
CPU runner plus the CPU/GPU execution contract is required before Driver 6;
mixed CPU/GPU and multi-GPU Driver 6 acceptance follows as kernels become
available. Independent parameter studies will use a separate future batch or
sweep command.

## Application Driver Portfolio

Drivers are implemented and accepted in this fixed order:

| Driver | Application driver | Required first validation cases |
| ---: | --- | --- |
| 1 | Steady incompressible SIMPLE/SIMPLEC | Complete laminar matrix above |
| 2 | Transient incompressible PISO/PIMPLE | `taylorGreenVortex`, `startUpPlaneChannel`, `womersleyPipe` |
| 3 | Low-Mach thermal/buoyant | `heatedPlaneChannel`, `rayleighBenardConduction`, `differentiallyHeatedCavity` |
| 4 | Low-Mach reacting flow | `manufacturedAdvectionDiffusionReaction`, `laminarPremixedFlame` |
| 5 | Compressible flow | `linearAcousticWave`, `sodShockTube`, `isentropicNozzle` |
| 6 | Multi-region conjugate/reacting | `compositeSlab`, `conjugateHeatedChannel`, `surfaceReactionChannel` |
| 7 | Immiscible two-phase VOF | `interfaceAdvection`, `staticDroplet`, `capillaryRise`, `damBreak` |

Drivers 1 and 2 are separate readiness gates but share the public
`incompressibleFluid` module. Steady/transient mode, SIMPLE/SIMPLEC/PISO/PIMPLE,
and laminar/turbulence selection come from the case rather than executable
names.

Within each driver, cases are implemented in the listed order. Packed-bed
geometry, Ergun resistance, porous momentum sources, and pseudo-homogeneous
reactor models are explicitly outside this seven-driver phase.

Reference selection follows the strongest available independent contract:

- exact or analytical solutions for Taylor-Green decay, start-up channel flow,
  Womersley flow, acoustic waves, Sod shock tubes, isentropic nozzles,
  composite slabs, interface advection, Laplace pressure, and capillary
  equilibrium;
- analytical or semi-analytical heat-transfer references for heated channels
  and the subcritical Rayleigh-Benard conduction state;
- manufactured solutions for coupled transport/reaction and multiregion
  coupling where a useful closed form is unavailable;
- documented external benchmarks for cavities, separated flows, flames,
  conjugate channels, surface reactions, and dam breaks.

## Driver Readiness Gate

A driver is complete only when:

- its Ferrum cases run from a clean checkout with native `FerrumFile` input;
- each OpenFOAM 13 sibling case runs independently without Ferrum conversion;
- an analytical or manufactured reference exists wherever mathematically
  valid;
- otherwise a benchmark reference records source, units, sampling, and
  tolerance;
- the comparison runner produces machine-readable JSON and human-readable
  Markdown Ferrum/OpenFOAM/reference reports;
- conservation, residual, field, and runtime diagnostics are regression-tested;
- mesh or time-step refinement is demonstrated, or non-monotonic behavior is
  explained;
- case-specific acceptance logic remains outside the generic driver.

## Deferred Phase: Porous Media And Packed Beds

Porous-media, Darcy-Forchheimer/Ergun, packed-bed, pellet, membrane-reactor,
and pseudo-homogeneous reactor development starts only after Driver 7 passes
the readiness gate. Architecture may keep generic source and interface
extension points, but this phase must not displace the seven-driver validation
sequence.

## Immediate Next Steps

1. Merge the repository-layout, tutorial-bundle, and canonical `ferrumRun`
   naming migration.
2. Extract the `incompressibleFluid` module registry and common solver
   lifecycle from the transitional combined crates with parity tests.
3. Specify and implement `FerrumFile v1`; isolate OpenFOAM support behind the
   `openfoamIO` interoperability layer.
4. Complete Driver 1 SIMPLE/SIMPLEC and the remaining laminar validation
   matrix.
5. Establish the shared CPU/GPU execution context and deterministic CPU
   `ferrumMultiRun` scheduler while Driver 2 PISO/PIMPLE is developed.
6. Implement Drivers 2 through 7 in the fixed order above, applying the common
   readiness gate and completing coupled `ferrumMultiRun` before Driver 6.
7. Begin porous-media and packed-bed work only after Driver 7 is complete.
