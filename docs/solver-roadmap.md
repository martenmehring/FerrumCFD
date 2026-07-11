# FerrumCFD Solver Roadmap

This roadmap first completes and broadens the steady laminar incompressible
foundation, then implements exactly six additional application drivers in a
fixed order. Every driver is validated with independent Ferrum and OpenFOAM 13
cases and an analytical, manufactured, or documented benchmark reference.
Porous-media, Ergun, and packed-bed development starts only after all seven
application drivers have passed their readiness gates.

## Project Identity, Version, And Provenance Policy

FerrumCFD is a distinct Rust finite-volume CFD platform. Its native
architecture, public commands, case format, numerical kernels, backend model,
and optimization work are defined and maintained within the FerrumCFD project.
Numerical behavior is
verified against analytical solutions, published benchmarks, and independently
executed external reference solvers. Optional OpenFOAM Foundation 13
interoperability and comparison assets remain isolated from native Ferrum
architecture and do not imply affiliation or endorsement.

OPENFOAM® is a registered trademark of OpenCFD Ltd. FerrumCFD is a separate
project and is not affiliated with or endorsed by OpenCFD Ltd or The OpenFOAM
Foundation. This acknowledgement accompanies releases and interoperability
documentation without using the mark as a Ferrum product or command name.

The current pre-stable product version is `0.1.0`. Version policy is:

- `0.1.1`-style patch releases contain compatible fixes, security hardening,
  validation corrections, and documentation/build repairs;
- `0.2.0`-style minor releases add a development milestone and may include
  explicitly documented pre-1.0 case, CLI, or library contract changes;
- pre-releases such as `0.2.0-alpha.1` identify incomplete milestone candidates;
- `1.0.0` establishes the first supported public CLI, case, library,
  interoperability, and result/restart contracts;
- after 1.0, incompatible public contracts require a major version.

Leaf `F-REL-0.1.0` centralizes `version = "0.1.0"` in
`[workspace.package]`, makes every package inherit it, adds consistent
`--version` output to public executables, creates `docs/versioning.md`, and
defines reproducible release/tag gates. A `v0.1.0` tag is created only after
those gates pass; workflow attempt numbers are never product versions.

Public documentation may describe automated security scanning, including
Codex Security, as a source of candidate findings. Scanner output still
requires manual validation and targeted tests. Public project documentation
does not expose internal implementation-provider or model details.

## Target Repository Layout

The repository converges on the following Ferrum-owned responsibility layout.
External solver layouts may be studied for interoperability and validation, but
they do not define Ferrum ownership. Third-party implementation code must not be
incorporated into MIT-licensed native crates without compatible authorization
and recorded provenance.

```text
FerrumCFD/
|-- applications/
|   |-- solvers/
|   |   |-- ferrumRun/              # single-region, CPU and GPU capable
|   |   `-- ferrumMultiRun/         # coupled multi-region, same backends
|   |-- modules/
|   |   |-- incompressibleFluid/      # inspected source confirms name; formal F-REF-1 pending
|   |   |-- thermalFluid/             # provisional; audit pending
|   |   |-- speciesTransport/         # provisional; audit pending
|   |   |-- porousMedia/              # provisional and deferred
|   |   |-- chemistry/                # provisional; may become a model library
|   |   `-- ...
|   `-- utilities/
|       |-- mesh/
|       |-- case/
|       `-- postProcessing/
|-- src/
|   |-- ferrumCore/
|   |-- ferrumMesh/
|   |-- ferrumFiniteVolume/
|   |-- ferrumIO/
|   |-- openfoamIO/                  # interoperability only
|   `-- ferrumModels/
|-- tutorials/
|   |-- incompressibleFluid/
|   |   |-- laminarPipe/
|   |   |   |-- shared/
|   |   |   |   |-- geometry/
|   |   |   |   `-- physicalParameters.toml
|   |   |   |-- ferrum/
|   |   |   |   `-- case/
|   |   |   |-- openfoam-v13/
|   |   |   |   `-- case/
|   |   |   |-- analytical/
|   |   |   |-- comparison.toml
|   |   |   `-- README.md
|   |   `-- planeChannel/            # same complete case bundle
|   `-- porousMedia/
|       `-- ergunPressureDrop/        # same bundle, deferred until Driver 7
|-- validation/
|-- test/
|-- docs/
|-- Cargo.toml
`-- target/                           # generated and ignored
```

The tree is a migration target, not permission to create empty architecture
for its own sake. Each directory becomes executable or gains a narrow ownership
contract before the next layer depends on it. `applications/solvers` owns only
dispatch and lifecycle control; physics lives in `applications/modules`, while
reusable implementation belongs under `src`.

Only `incompressibleFluid` is currently a confirmed permanent module name.
`thermalFluid`, `speciesTransport`, `porousMedia`, and `chemistry` preserve the
requested target-tree intent but remain provisional until their OpenFOAM 13 and
mathematical ownership audits decide whether each is an application module, a
reusable `ferrumModels` capability, or part of another module. Renaming a
provisional boundary requires a recorded architecture decision, not guesswork.

Every selected tutorial, including `planeChannel` and later porous-media cases,
uses the complete `laminarPipe` bundle shape. `shared/physicalParameters.toml`
is the comparison source of truth but neither implementation depends on it at
runtime: the Ferrum and OpenFOAM cases remain independently runnable. The
`analytical/` directory is always present. If no useful closed form exists, its
README explains why and identifies the manufactured or documented benchmark
used instead; a sibling `benchmark/` directory may then hold reference data.
Keeping `analytical/` present even when it contains a documented
not-applicable decision is a deliberate bundle-consistency policy; it must never
be presented as evidence that a closed-form solution exists.

The existing `laminarPipe` and `planeChannel` bundles predate this complete
contract and currently keep some physical inputs in comparison/case files.
Their retrospective migration must add `shared/physicalParameters.toml`, derive
or validate both program-specific cases reproducibly, remove duplicated
authoritative parameters from `comparison.toml`, fail on drift, and record the
effective parameter hash/provenance in every comparison report.
`comparison.toml` also records the effective time/coupling classification and
its evidence (`ddtSchemes`, `consistent`, outer/inner corrector counts); a
directory or dictionary-section name alone is not accepted as the algorithm
classification.

## Mandatory OpenFOAM 13 Reference Audit

OpenFOAM Foundation v13 MUST be inspected before every new Ferrum module,
tutorial, utility, model, boundary condition, dictionary, or `src` component is
placed or implemented. The already implemented layout and the existing
`laminarPipe`/`planeChannel` bundles are the sole retrospective `F-REF-1`
exception. Guessing from executable names or older OpenFOAM releases is not an
acceptable substitute.

For every selected physics area, first create or update an English reference
map under `docs/reference/openfoam-v13/` that records:

1. the exact OpenFOAM 13 commit/release, build ID, local source paths, and
   hashes of the decisive source files inspected;
2. the relevant `applications/solvers`, `applications/modules`, `src`, and
   `tutorials` entries and the runtime-selection path through `foamRun` or
   `foamMultiRun`;
3. the selected OpenFOAM 13 tutorial inventory and its mapping to Ferrum module
   and case names;
4. required fields, dictionaries, models, boundary conditions, finite-volume
   schemes, utilities, function objects, and validation references;
5. effective algorithm classification derived from `ddtSchemes`,
   `consistent`, `nOuterCorrectors`, inner correctors, and the complete control
   dictionary rather than only from a section label: SIMPLEC may be a
   consistent SIMPLE case, while PISO may be a transient PIMPLE-configured case
   with one outer corrector;
6. the proposed Ferrum ownership location and dependency direction for every
   reusable capability;
7. the OpenFOAM decomposition, rank communication, reconstruction, and
   multi-region interface path where parallel execution applies;
8. at least one unchanged official OpenFOAM 13 tutorial execution from an
   isolated disposable copy on a separate supported Linux/WSL/CI environment,
   never in place under `/opt/openfoam13/tutorials`, with logs and reference
   results stored under `target/reference-audits/`;
9. a decision table separating externally observable behavior to validate,
   numerical methods to implement from cited primary references, formats
   confined to interoperability, and deferred features;
10. independent mathematical primary references and acceptance observables;
11. license and provenance notes for source, case, mesh, and benchmark material;
12. a capability/provenance matrix with columns
    `Capability | Mathematical reference | Ferrum owner | Ferrum status | External validation | OpenFOAM reference scope | License/provenance decision`.

The currently verified local baseline is OpenFOAM Foundation 13 build
`13-441953dfbb42` under `/opt/openfoam13`. Its `foamRun` path selects one module
for one region. Its `foamMultiRun` path selects one module per region and
advances the coupled regions through shared phase and time loops. The
`multiRegion/CHT/heatedDuct` reference demonstrates the parallel lifecycle with
`decomposePar -allRegions`, `runParallel foamMultiRun`, and
`reconstructPar -allRegions`; Ferrum must audit that behavior before defining
its coupled decomposition contract.

The pinned tree contains 54 top-level `incompressibleFluid` entries and 53
detected runnable case roots identified by `system/controlDict`. The first
inventory leaf, `F-REF-D1-MODULE-INVENTORY`, records top-level grouping,
generation, and overlay entries separately from runnable case roots. Of the 53
case roots, 48 declare `solver incompressibleFluid`; their 53
`momentumTransport` dictionaries classify 15 as laminar, 35 as RAS, and 3 as
LES. Coupling, time scheme, fields, and capability gaps are classified per
runnable case root, not per top-level directory. Auditing every entry does not
claim that every case is already supported. Each runnable row receives one
status: `native`, `partial`, `planned`, `interoperability-only`, or
`out-of-scope`, plus the exact blocking capability and its future driver.

The first audit pass for each driver covers at least these OpenFOAM 13 areas:

| Driver | Mandatory OpenFOAM 13 reference scope |
| ---: | --- |
| 1 | `applications/modules/incompressibleFluid`; planar Poiseuille/Couette, cavity, and steady separated-flow tutorials |
| 2 | transient `incompressibleFluid` PISO/PIMPLE lifecycle and tutorials |
| 3 | `fluid`/`isothermalFluid`; buoyant cavity, Benard-cell, and heated-room tutorials |
| 4 | `multicomponentFluid`; species, chemistry, flame, and reacting-channel tutorials |
| 5 | `fluid`, `isothermalFluid`, and `shockFluid`; official `fluid/shockTube`, `shockFluid/shockTube`, and `fluid/helmholtzResonance` paths |
| 6 | `foamMultiRun`, `regionSolvers`, multi-region control, `fluid`, `solid`, and CHT tutorials |
| 7 | `incompressibleVoF`, `twoPhaseVoFSolver`, `VoFSolver`, and official capillary-rise and dam-break tutorials |

These paths define what must be inspected, not what may be copied or which
Ferrum module name is automatically correct. The audit confirms or revises the
provisional module boundaries through a recorded architecture decision.
`isentropicNozzle` and `staticDroplet` remain useful Ferrum analytical
acceptance cases, but the current local OpenFOAM 13 tree does not contain
official tutorial directories with those names; reports must not label them as
official OpenFOAM tutorials.

The audit is architectural and behavioral reference work, not source copying.
Ferrum remains MIT-licensed: GPL-licensed OpenFOAM implementation code is not
copied into Ferrum crates. External OpenFOAM names and formats are confined to
`openfoamIO`, independently runnable `openfoam-v13` cases, provenance records,
and comparison tooling.

Every distributed `openfoam-v13` bundle receives an explicit classification:
either independently authored/generated, or derived from OpenFOAM material. A
derived bundle is excluded from the MIT license scope and carries the required
upstream license plus a root `THIRD_PARTY_NOTICES` entry. A provenance note by
itself is not a license grant.

The OpenFOAM sibling cases are validation/interoperability assets.
`ferrumRun` neither embeds nor requires an OpenFOAM executable. Users may copy a
sibling case to a supported Linux machine and execute it with their own
OpenFOAM Foundation 13 installation. Developer validation tooling may
explicitly invoke a separately installed and pinned OpenFOAM Foundation 13
environment against disposable case copies below `target/`. FerrumCFD does not
redistribute the OpenFOAM executable.

Before Driver 2 is accepted, the reference map must contain a frozen inventory
of all OpenFOAM 13 SIMPLE/SIMPLEC/PISO/PIMPLE tutorials selected for Ferrum's
incompressible scope. Each selected row has a Ferrum case, an independent
OpenFOAM v13 case, an analytical/manufactured/benchmark reference, acceptance
tolerances, and a status. This inventory defines what “all SIMPLE/PIMPLE cases”
means and prevents silent scope drift. Every `native` case and every `partial`
case whose missing capability belongs to the active driver receives its own
bounded implementation-and-run leaf; unsupported rows remain explicit instead
of being silently skipped or forced through the wrong solver. The accepted
inventory is frozen by version and content hash before `F-D1-GATE`; changing it
requires a reviewed roadmap amendment rather than a silent dynamic expansion.

The native source split follows a reviewed, acyclic dependency graph:

- `ferrumCore`: fundamental types, dimensions, registries, errors, execution
  context, and backend-neutral contracts;
- `ferrumMesh`: topology, geometry, decomposition, partitions, and interfaces;
- `ferrumFiniteVolume`: fields, operators, matrices, discretization, and
  equation assembly;
- `ferrumIO`: native `FerrumFile` parsing, writing, and case I/O;
- `openfoamIO`: OpenFOAM import/export adapters only;
- `ferrumModels`: reusable physical and constitutive models.

Applications may depend on these libraries, but the libraries never depend on
an application executable. `ferrumIO` owns the native format; `openfoamIO` is
an optional adapter and must not define native Ferrum semantics.

## Current Status

The canonical public entry point is now
`ferrumRun -solver incompressibleFluid`. It dispatches the executable
finite-volume pressure-velocity prototype only for unambiguous steady-state
laminar cases with exactly one SIMPLE section and no PISO/PIMPLE section.
Explicit `momentumTransport`/`turbulenceProperties` input must select
`simulationType laminar`; RAS/LES is not dispatched to the laminar kernel.
No public algorithm-specific executable or `--solveLaminarSimple` selector is
retained. Currently only steady laminar SIMPLE executes through
`ferrumRun -solver incompressibleFluid`. SIMPLEC and future PISO/PIMPLE remain
case-selected modes behind that same public command; they are not implemented
yet. The implementation reads
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

Goal: preserve one numerical contract while scaling both public runners from a
correct CPU baseline to shared-memory CPUs, distributed partitions, one GPU,
and multiple GPUs.

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
- keep fields, operators, equation assembly, convergence criteria, reports, and
  case semantics independent of execution backend now;
- defer parallel optimization until the complete selected Driver 1 and Driver 2
  SIMPLE/SIMPLEC/PISO/PIMPLE case inventory passes on the scalar CPU baseline;
- then execute the following acceptance phases in order:
  1. one process and one CPU worker as the correctness reference;
  2. one process with multiple CPU worker threads and explicit affinity/NUMA
     policy;
  3. partitioned multi-process CPU execution on one host, followed by
     multi-node execution when required;
  4. one GPU with explicit capability and `f64` checks;
  5. multiple GPUs on one host with deterministic placement, peer-transfer
     policy, and conservative halo exchange;
  6. multi-node CPU/GPU execution with an explicit communication transport;
- require every phase to run the identical case inputs and numerical schemes,
  with stated tolerance parity, conservation checks, deterministic regression
  mode, scaling efficiency, and memory-transfer measurements;
- keep GPU optional and selectable per stage (`flow`, linear solves,
  nonlinear/interface/ODE stages), with CPU as a valid choice when GPU is busy,
  unsupported, or inefficient.

Rust threads can use all cores of one shared-memory host without MPI. Rust does
not remove the distributed-memory problem: multiple processes, multiple nodes,
and some multi-GPU layouts still require MPI or an equivalent transport. Before
Phase 3, record an architecture decision comparing a Rust MPI binding with
UCX/libfabric or a project-owned transport. One-process multi-GPU execution may
use vendor peer/collective APIs, but cross-node execution still requires a
network communication layer. No transport choice may leak into finite-volume
operators or case semantics.

The backend-neutral design work completed before parallel implementation must
provide:

- a `SolverModule` lifecycle shared by both runners;
- an `ExecutionContext` describing backend, resources, precision, affinity,
  queues, and communicator without exposing vendor APIs to physics modules;
- a serial `Communicator` plus partitionable mesh and field storage with owned
  and ghost/halo entities;
- bulk operator APIs without per-cell or per-face dynamic dispatch;
- explicit host/device data residency and transfer ownership;
- separate deterministic and performance-oriented reduction policies;
- run provenance covering workers, ranks, partitions, devices, precision,
  transport, and backend versions.

Threaded CPU kernels use a bounded worker pool; an asynchronous I/O runtime is
not treated as numerical CPU parallelism. Distributed acceptance separates
intra-region halo exchange, global reductions, failure propagation, and restart
state. GPU acceptance keeps fields, geometry, matrices, and solver state device
resident across iterations and reports every unavoidable host transfer.

## Milestone 6: Driver 1 Laminar Validation Matrix

Before Driver 2 starts, steady incompressible SIMPLE/SIMPLEC must pass this
tutorial matrix:

| Order | Case | Case origin | Primary coverage | Reference |
| ---: | --- | --- | --- | --- |
| 1 | `laminarPipe` | project-maintained matched bundle; origin audit pending `F-REF-1` | 3D internal flow and pressure loss | Hagen-Poiseuille analytical solution |
| 2 | `planeChannel` | project-maintained matched bundle; origin audit pending `F-REF-1` | true 2D `empty` handling | Plane-Poiseuille analytical solution |
| 3 | `couettePoiseuille` | planned project-authored matched bundle | moving wall and combined pressure/shear forcing | Analytical velocity profile |
| 4 | `lidDrivenCavity` | planned project-authored matched bundle | recirculation and closed-pressure reference | Published benchmark |
| 5 | `backwardFacingStep` | planned project-authored matched bundle | separation, reattachment, and outlet robustness | Published benchmark |
| 6 | `axisymmetricPipe` | planned project-authored matched bundle | `wedge` handling | Hagen-Poiseuille analytical solution |

Every case contains independently runnable `ferrum/` and `openfoam-v13/`
directories plus `shared/geometry`, `shared/physicalParameters.toml`,
`analytical/`, `comparison.toml`, and an English case README. If no closed form
is useful, `analytical/README.md` records that decision and the case adds a
documented `benchmark/` or manufactured reference. Coarse/medium/fine, skewed,
and non-orthogonal variants belong to these bundles instead of becoming
unrelated cases. The same bundle contract applies to every selected physics
module; it is not special to incompressible flow.

Official `planarCouette`, `planarPoiseuille`, `cavity`, `pitzDailySteady`,
`cylinder`, and `venturiTube` configurations are classified from their actual
v13 dictionaries before reuse as references. A similarly named Ferrum case is
not labelled an unchanged official tutorial when its time scheme, turbulence
model, geometry, or control algorithm differs.

## Runner And Multi-Region Milestone

The solver lifecycle must be shared by both public dispatchers:

- `ferrumRun`: one region, one runtime-selected module, and the full CPU,
  threaded, distributed, GPU, and multi-GPU backend ladder;
- `ferrumMultiRun`: multiple coupled regions and one module per region, reusing
  the same execution context, partitions, kernels, and backends.

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
2. keep `ferrumRun` on the scalar CPU correctness backend until the selected
   SIMPLE/SIMPLEC/PISO/PIMPLE inventory passes, while making operators and
   storage backend-neutral;
3. define a backend-neutral execution context that distinguishes sockets,
   cores, worker threads, process ranks, domain partitions, GPU devices,
   memory, and queues while preventing oversubscription;
4. implement and accept the `ferrumRun` threaded, distributed CPU, single-GPU,
   and multi-GPU phases from Milestone 5;
5. implement a deterministic `ferrumMultiRun` CPU scheduler with a
   capability/dependency graph,
   rank/partition mapping, halo/ghost exchange, interface barriers, and failure
   propagation;
6. add per-region and per-stage CPU/GPU placement, a data-residency/transfer
   graph, backend capability checks including `f64`, and mixed-backend parity
   tests;
7. add multi-GPU placement, deterministic cross-device reductions, and
   conservative region/partition interface exchange;
8. require CPU/GPU, mixed-backend, and multi-GPU parity plus
   mass/energy/species conservation at every coupled interface with stated
   tolerances.

The lifecycle and backend contracts are established during Drivers 1 and 2 so
later acceleration does not require an architectural rewrite, but parallel
kernel implementation starts only after the Driver 1/2 correctness gate.
`ferrumMultiRun` does not create a second backend stack: it schedules the same
module kernels over a coupled dependency graph. A working coupled CPU runner
plus the accepted single-region backend contract is required before Driver 6;
mixed CPU/GPU and multi-GPU Driver 6 acceptance follows as kernels become
available. Independent parameter studies use a separate future batch or sweep
command.

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

## Automated Engineering Delivery

Editing this roadmap is a planning operation. When the project owner asks to
"work through the roadmaps" or gives an equivalent execution instruction, only
a bounded leaf task enters the accepted isolated engineering workflow; an epic
or complete driver portfolio is decomposed before implementation begins.

The authoritative worktree, branch, security, persistence, independent-review,
and Draft-PR policy lives in the external delivery system and is referenced
here as dependency `F-AUTO-1`. It must pin a clean `ferrumcfd/main` SHA, isolate
the implementation worktree, enforce the bounded path set, run declared
numerical and security gates, keep implementation and review independent,
publish only a Draft PR, and return reproducible evidence for approval.

The delivery workflow passed its complete live acceptance on July 11, 2026;
`F-AUTO-1` is therefore satisfied. Implementation, deterministic validation,
independent review, cleanup, and explicit Draft-PR publication remain separate
from read-only analysis. Any change to that boundary requires renewed
acceptance before FerrumCFD uses it.

Only leaf tasks may enter that workflow. The immediate-next-step IDs below are
epics unless explicitly marked as a leaf. They decompose as follows:

- reference work: `F-REF-D<driver>-MODULE` and one
  `F-REF-D<driver>-CASE-<case>` per official or analytical case;
- existing bundle migration: `F-LAYOUT-PARAMS-LAMINARPIPE-CONTRACT` and
  `F-LAYOUT-PARAMS-PLANECHANNEL-CONTRACT`, each with its own drift test;
- Driver 1/2 implementation: one ID per boundary condition, operator,
  SIMPLEC/PISO/PIMPLE behavior, or tutorial case, followed by a separate driver
  gate task;
- backend work: `F-BE-THREADS`, `F-BE-PARTITION`, `F-BE-MULTIPROCESS`,
  `F-BE-MULTINODE`, `F-BE-GPU1`, `F-BE-GPUN`, and
  `F-BE-MULTINODE-GPU` in that order;
- Drivers 3-7: separate audit, module/lifecycle, model, individual case, and
  final readiness-gate tasks for each driver.

A leaf task has one bounded objective, explicit allowed paths, acceptance
observables, and a finite test command set. Completing a leaf never marks its
parent epic or driver complete automatically.

## Deferred Phase: Porous Media And Packed Beds

Porous-media, Darcy-Forchheimer/Ergun, packed-bed, pellet, membrane-reactor,
and pseudo-homogeneous reactor development starts only after Driver 7 passes
the readiness gate. Architecture may keep generic source and interface
extension points, but this phase must not displace the seven-driver validation
sequence.

## Immediate Next Steps

1. **F-REL-0.1.0 (leaf):** Centralize and expose version `0.1.0`, document
   SemVer/release gates, and defer tag `v0.1.0` until those gates pass.
2. **F-POSITIONING-1 (leaf):** Apply the independent-project wording and
   provider-neutral delivery language across public documentation; retain
   truthful provenance, licensing, security-scanner, and non-affiliation notes.
   The bounded audit covers `README.md`, `docs/architecture.md`,
   `applications/README.md`, `src/README.md`, `CHANGELOG.md`, and tutorial
   documentation.
3. **F-LAYOUT-PARAMS-LAMINARPIPE-CONTRACT (maintainer publication gate, not a
   coding leaf):** Review Draft PR #4; merge only after explicit maintainer
   approval and required checks, then synchronize `main`. If the publisher
   identity policy requires a replacement, close #4 only after the replacement
   Draft PR has reproduced the sealed tree and evidence.
4. **F-LAYOUT-PARAMS-PLANECHANNEL-CONTRACT (leaf):** Complete the canonical
   shared parameters and full field/property/mesh/geometry drift contract.
5. **F-REF-D1-MODULE-INVENTORY (leaf):** Inventory the 54 top-level entries and
   53 detected runnable case roots, then create the capability/provenance
   matrix. Select all currently supported cases and record the exact blocker
   for every deferred case.
6. **F-ARCH-1:** Extract the `incompressibleFluid` module registry and common
   solver lifecycle from transitional combined crates with parity tests.
7. **F-IO-1:** Specify `FerrumFile v1` and isolate OpenFOAM support behind the
   `openfoamIO` interoperability layer before new permanent case bundles.
8. **F-D1-CASES (leaf series):** Implement and independently run each remaining
   selected Driver 1 bundle: Ferrum, external OpenFOAM 13 case, and analytical
   or documented benchmark reference.
9. **F-D1-GATE:** Pass the frozen, versioned, hash-bound Driver 1 inventory on
   the scalar CPU reference backend.
10. **F-D2-PISO-PIMPLE (leaf series):** Add transient lifecycle capability and
   execute the selected incompressible PISO/PIMPLE inventory.
11. **F-BACKEND-1:** After Driver 1/2 scalar correctness, accept `ferrumRun` on
    threaded CPU, distributed CPU, one GPU, and multiple GPUs without changing
    case numerics.
12. **F-D3D7-1:** Implement Drivers 3 through 7 in order, completing coupled
    `ferrumMultiRun` before Driver 6.
13. **F-POROUS-1:** Begin porous-media and packed-bed work only after Driver 7.
