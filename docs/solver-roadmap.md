# FerrumCFD Solver Roadmap

This roadmap first completes and broadens the steady laminar incompressible
foundation, then implements exactly six additional application drivers in a
fixed order. Every driver is validated with independent Ferrum and OpenFOAM 13
cases and an analytical, manufactured, or documented benchmark reference.
Porous-media, Ergun, and packed-bed development starts only after all seven
application drivers have passed their readiness gates.

## Target Repository Layout

The repository converges on the following OpenFOAM-inspired responsibility
layout. The names describe Ferrum ownership; they do not permit OpenFOAM
implementation code to leak into native Ferrum components.

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
|   |   |   |-- shared/               # optional neutral inputs
|   |   |   |   `-- geometry/
|   |   |   |-- ferrum/
|   |   |   |   `-- case/
|   |   |   |-- openfoam-v13/
|   |   |   |   `-- case/
|   |   |   |-- analytical/           # optional when useful
|   |   |   |-- comparison.toml       # optional reference mapping
|   |   |   `-- README.md
|   |   `-- planeChannel/            # same independent program references
|   `-- porousMedia/
|       `-- ergunPressureDrop/        # deferred until Driver 7
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

Every selected tutorial has a small user-facing contract:

- one independently runnable Ferrum case;
- one independently runnable OpenFOAM Foundation 13 case;
- an analytical reference when a useful closed form exists, otherwise a
  documented benchmark reference;
- an English README that explains the physics, assumptions, and program-specific run
  commands;
- an optional stable result summary when maintainers have recorded a run.

`shared/geometry`, shared parameter metadata, `comparison.toml`, case
generation helpers, machine-readable reports, and refinement studies are
optional maintainer tools. They are added only when they materially help a
specific case, are never runtime dependencies, and are not prerequisites for a
user to run either case.

`laminarPipe` and `planeChannel` are established functional bundles. Their
current tests and metadata remain useful, but no retrospective parity,
parameter-hash, lexical-hardening, or source-drift project blocks the next
physics case.

## Focused OpenFOAM 13 Reference Check

OpenFOAM Foundation v13 MUST be inspected before a new Ferrum physics module,
solver behavior, tutorial case, or permanent ownership boundary is implemented.
Routine documentation, case-data maintenance, and unrelated utilities do not
need a new audit. Guessing from executable names or older OpenFOAM releases is
not an acceptable substitute.

Before implementing a new physics capability or case, inspect the relevant
OpenFOAM 13 module and tutorial rather than guessing from an older executable
name. Record only what is needed for the bounded work:

1. the OpenFOAM 13 module or application and the native case used as reference;
2. the fields, models, boundary conditions, schemes, and coupling algorithm
   required by that case;
3. the corresponding Ferrum ownership and any capability that is still
   missing;
4. an independent mathematical or published reference where useful;
5. license and provenance information for copied case or mesh material.

Source hashes, exhaustive inventories, unchanged reference runs, decomposition
audits, and detailed decision tables are produced only when the task actually
depends on them. They do not block a normal tutorial-case addition.

The currently verified local baseline is OpenFOAM Foundation 13 package/tag
`20260407`, build `13-441953dfbb42`, under `/opt/openfoam13`. Its `foamRun` path selects one module
for one region. Its `foamMultiRun` path selects one module per region and
advances the coupled regions through shared phase and time loops. The
`multiRegion/CHT/heatedDuct` reference demonstrates the parallel lifecycle with
`decomposePar -allRegions`, `runParallel foamMultiRun`, and
`reconstructPar -allRegions`; Ferrum must audit that behavior before defining
its coupled decomposition contract.

The first focused reference pass for each driver starts with these OpenFOAM 13
areas and expands only when a selected case needs more:

| Driver | Mandatory OpenFOAM 13 reference scope |
| ---: | --- |
| 1 | `applications/modules/incompressibleFluid`; the official steady laminar `cylinder` case and the selected pipe/channel references |
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

Before Driver 2 is accepted, the roadmap lists the Driver 1 cases that were
actually selected and implemented, with their Ferrum case, independent
OpenFOAM v13 sibling, available reference, and status. The list may grow as
useful official cases are evaluated; an exhaustive tutorial inventory does not
block the next bounded case.

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

The current 2026-07-17 external convergence profiles reproduce the pipe stop at
iteration `207` (`U=9.986584e-4`, `p=2.618382e-5`) in `15.95 s` and converge
the plane channel at iteration `545` (`U=9.974216e-6`, `p=4.210369e-8`) in
`8.16 s`. Both outer solves report `converged=true`, and every recorded
momentum-component and pressure linear solve converged. A run without
`residualControl` is reported as `not-evaluated`; configured criteria exhausted
at the iteration budget are reported as `not-reached`. The current performance
and external accuracy evidence is recorded in
`docs/benchmarks/cpu-performance-foundation.md`.

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

The existing pipe and plane-channel benchmark records remain historical
evidence. Maintainers may refresh them or add focused mesh studies when a
numerical question requires it, but same-mesh orchestration, parameter sweeps,
and a master comparison runner are not tutorial or user requirements. New
stable result tables are stored under `docs/benchmarks` only after a run has
actually been performed.

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
- solver report schema version 2 now includes additive phase timings for setup,
  momentum matrix assembly, momentum linear solves, pressure matrix assembly,
  pressure linear solves, finalization, and remaining solver work;
- `run_cpu_performance_baseline.ps1` builds `ferrumRun` once in release mode,
  excludes compilation and warmup runs from medians, and executes the existing
  `laminarPipe` and `planeChannel` cases as independent regressions.
- The first 2026-07-16 release diagnostic identified redundant per-iteration
  convection diagnostics. Deferring that diagnostic operator to finalization
  reduced the pipe from `64.75 s` to `16.64 s` and the plane channel from
  `1134.82 s` to `354.95 s`, with identical SIMPLE/linear iteration counts,
  continuity, residuals, and field norms. This is one measured run per case,
  retained as diagnostic provenance rather than a stable median claim.
- mesh-dependent momentum CSR sparsity is now built once and shared by all
  component matrices. Splitting the old assembly timer showed that matrix fill
  itself was small and that repeated scalar-gradient geometry dominated it;
- mesh-dependent scalar-gradient interpolation weights, boundary distances, and
  inverse cell volumes are now cached once;
- pressure matrices now share mesh-dependent CSR topology and reuse coefficient
  and right-hand-side storage. Pressure reference elimination updates that
  storage in place;
- pressure PCG reuses work vectors, while IC(0) builds its symbolic dependency
  structure once and only refactors numerical values for each equation;
- one scalar-solve workspace now retains the outer zero-initial, matrix-product,
  and residual vectors across momentum and pressure equations. The pressure
  corrector also reuses its stored solution as the next iterate without a
  separate full-field clone;
- a deterministic pressure-matrix integration gate covers `13,824`-row medium,
  `38,912`-row fine, and `12,288`-row/`56.31 deg` skewed conservative systems.
  Reused and fresh PCG/IC(0) solves agree exactly, with true relative residuals
  below `1.7e-9`;
- opt-in PCG kernel profiling now reaches the normal SIMPLE console, JSON, and
  Markdown reports and the external performance driver. Release measurements
  on the accepted matrices put IC(0) applications at `52.6%` to `55.1%` of the
  PCG kernel; current SIMPLE cases put them at about `45.8%` to `47.6%` through
  convergence. IC(0) numerical refactorization remains below `1%` throughout.
  All fixed-work and converged numerical reports remain exactly equal to their
  pre-profile counterparts after removing only timing and case-path fields;
- IC(0) backward dependencies now use one contiguous offset/entry layout
  instead of one nested vector per row, preserving exact traversal and
  arithmetic order. On the `38,912`-row gate this removes about `622.6 kB` of
  row metadata plus thousands of small allocations. Three alternating
  same-process release diagnostics measured `1.0749x`, `1.1159x`, and
  `1.1630x` application speedups with bit-identical output. Sequential
  full-case batches remained host-load-sensitive, so no new end-to-end speedup
  is claimed;
- the current fixed-work release gate (one warmup plus five measured runs) has
  medians of `2.2228 s` for 10 pipe iterations and `8.1320 s` for 500 channel
  iterations. This is `29.13x` and `139.55x` faster than the original recorded
  basis. Host load was more variable than during the preceding checkpoint, but
  all numerical reports remained byte-identical after timing fields were
  removed;
- opt-in GAMG cycle profiling now reports hierarchy, residual, transfer,
  smoothing, scaling, correction, and coarsest-solve time for every level. It
  identified repeated diagonal lookup in symmetric Gauss-Seidel as the bounded
  hot path. Reusing the hierarchy's diagonal slots preserves CSR operation
  order and reduced five-run median solver time from `9.6535 s` to `9.2245 s`
  for the pipe and from `9.0066 s` to `8.1576 s` for the channel, with identical
  SIMPLE counts, V-cycle counts, residuals, continuity, and field summaries;
- pressure solves still dominate both convergence-profile runs: the pipe
  converges in `15.95 s` at iteration `207`, and the channel in `8.16 s` at
  iteration `545`. Both preserve the previous iteration counts and final
  numerical observables exactly.

### Performance Foundation - Scalar CPU

This work starts only after the selected numerical cases are correct enough to
act as regressions. It changes storage and execution mechanics, not equations,
boundary conditions, convergence criteria, or case semantics.

Acceptance criteria for every scalar-CPU optimization:

- use the release executable directly; record build time separately;
- run at least one warmup and five measured runs, reporting the median;
- preserve convergence state, SIMPLE and linear-iteration observables, final
  continuity, residuals, and field summaries within stated tolerances on both
  `laminarPipe` and `planeChannel`;
- reject pipe-only, channel-only, analytical, or benchmark-specific branches in
  the generic `incompressibleFluid` solver;
- change one bounded hot path at a time and retain before/after JSON evidence;
- accept a performance claim only when it improves both cases or when a
  documented mesh/solver characteristic explains a neutral result;
- keep current OpenFOAM 13 comparisons external to Ferrum case semantics and
  run them with matched hardware, process/thread counts, schemes, stopping
  criteria, and clearly separated solver/process wall times.

The first optimization sequence is tracked as follows:

1. completed: establish release baselines and phase profiles;
2. completed: remove redundant diagnostic operator evaluations from the SIMPLE
   hot path;
3. completed: precompute and share mesh-dependent momentum CSR sparsity;
4. completed: cache mesh-dependent scalar-gradient geometry;
5. completed: reuse pressure CSR topology, matrix values, and right-hand sides;
6. completed: reuse pressure PCG work vectors and IC(0) symbolic/factor storage;
7. completed: remove the remaining outer scalar-solve residual/matvec allocations
   and avoidable full-field clones from the SIMPLE hot path;
8. completed: validate reusable PCG/IC(0) on medium, fine, and deliberately skewed
   pressure matrices;
9. completed: split pressure-kernel time into IC(0) refactorization,
   matrix-vector, preconditioner-application, and vector-update phases;
10. completed: flatten repeated IC(0) backward-application dependencies while
    preserving floating-point operation order;
11. completed: establish the matrix-level OpenFOAM-compatible GAMG pressure
    foundation with reusable hierarchy storage, algebraic-pair agglomeration,
    V-cycles, dictionary-control mapping, and explicit unsupported-control
    errors while retaining PCG/IC(0) as a selectable solver;
12. completed: connect GAMG to the symmetric SIMPLE pressure path with
    mesh-geometric `faceAreaPair`, per-equation runtime controls, reusable
    pressure hierarchy ownership, explicit momentum rejection, and the
    two-case numerical/performance gate. Both cases converge, but the first
    paired release diagnostic is slower than PCG/IC(0), so no GAMG speedup is
    claimed;
13. completed: profile GAMG hierarchy refresh and every V-cycle phase, identify
    smoothing as the dominant phase, and reuse cached diagonal slots in the
    Gauss-Seidel smoother while preserving floating-point operation order;
14. validate DILU/ILU for nonsymmetric momentum systems;
15. add further in-place history, residual, and reporting operations only where
    measurement shows material cost;
16. rerun the two-case gate after every step, then proceed to threaded CPU work.

n8n may build, execute, collect artifacts, compare tolerances, and reject a
regression. It must not combine several numerical or storage optimizations into
one unattended change. Each optimization remains a bounded reviewed task with
its own before/after evidence.

Next performance targets:

- retain the completed per-level GAMG profile and cached-diagonal smoother. A
  further GAMG performance leaf must first isolate the remaining row-kernel
  cost; it must not tune case tolerances or cycle counts as a shortcut;
- retain the completed medium/fine/skewed IC(0) gate and add a true
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

| Order | Case | Primary coverage | Reference |
| ---: | --- | --- | --- |
| 1 | `laminarPipe` | 3D internal flow and pressure loss | Hagen-Poiseuille analytical solution |
| 2 | `planeChannel` | true 2D `empty` handling | Plane-Poiseuille analytical solution |
| 3 | `cylinder` | official OpenFOAM 13 steady laminar external flow at `Re = 1`; limited-scheme compatibility is the next prerequisite | Official-case observables selected during the focused case task |
| 4 | `lidDrivenCavity` | recirculation and closed-pressure reference | Published benchmark |
| 5 | `backwardFacingStep` | separation, reattachment, and outlet robustness | Published benchmark |
| 6 | `axisymmetricPipe` | `wedge` handling | Hagen-Poiseuille analytical solution |

Every case contains independently runnable `ferrum/case` and
`openfoam-v13/case` directories, an English README, and an analytical reference
when one is useful. Otherwise the README points to a documented benchmark.
Shared inputs, comparison metadata, recorded results, and mesh variants are
optional and case-specific. No combined runner is required.

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

- its selected Ferrum cases run from a clean checkout using the supported
  compatibility format until `FerrumFile v1` is complete;
- each OpenFOAM 13 sibling case runs independently without Ferrum conversion;
- an analytical reference is supplied where useful, otherwise a documented
  benchmark identifies the source and observables;
- implemented solver behavior has focused automated unit or integration
  regression coverage with stated acceptance tolerances;
- maintainers record at least one successful result for each selected case;
- case-specific reference logic remains outside the generic driver.

Master comparison runners and exhaustive refinement studies remain optional
engineering tools selected for a concrete numerical risk. They are not part of
the user-facing case contract.

## Roadmap Execution Through The Coding-Agent Workflow

Editing this roadmap is a planning operation. When the user asks to "work
through the roadmaps" or gives an equivalent execution instruction, only a
bounded leaf task is delegated through the separate AI Dev Orchestrator/n8n
repository; an epic, driver portfolio, or open-ended "continue" request is not
a valid coding task.

The authoritative worktree, branch, model, security, persistence, and Draft-PR
policy lives in the orchestrator repository and is referenced here as external
dependency `F-AUTO-1`. FerrumCFD requires that accepted workflow to pin a clean
`ferrumcfd/main` SHA, isolate the implementation worktree, use Codex for the
bounded implementation, run an independent secondary review plus the declared
numerical and Codex Security gates, publish only a Draft PR, and return evidence
to chat.

The separate roadmap-coding workflow passed its complete live acceptance on
July 11, 2026; `F-AUTO-1` is therefore satisfied. Its implementation,
validation, independent review, cleanup and explicit Draft-PR publication
boundary remain separate from the read-only analysis workflow. Any future
change to that boundary must pass the orchestrator repository's acceptance
procedure again before FerrumCFD uses it.

Only leaf tasks may enter that workflow. The immediate-next-step IDs below are
epics unless explicitly marked as a leaf. They decompose as follows:

- reference work: `F-REF-D<driver>-MODULE` and one
  `F-REF-D<driver>-CASE-<case>` per official or analytical case;
- existing `laminarPipe` and `planeChannel` bundles remain accepted as
  implemented; no additional hardening leaf is required for parity;
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

Completed 2026-07-17: **F-PERF-GAMG-SIMPLE-INTEGRATION** connects
OpenFOAM-compatible `faceAreaPair` and reusable GAMG hierarchy storage to the
symmetric pressure equation. Pipe and channel reach the same outer iteration
counts as PCG with all linear solves converged. The paired one-run performance
gate is slower, so GAMG remains opt-in and no speedup is claimed.

Completed 2026-07-17: **F-PERF-GAMG-CYCLE-PROFILE** adds opt-in aggregate and
per-level timing without changing case semantics. Smoothing was the dominant
phase. Reusing the existing unique diagonal-slot layout reduced controlled
five-run median solver time by `4.44%` for the pipe and `9.43%` for the channel
with identical numerical observables. PCG/IC(0) remains the default; GAMG
remains explicitly selectable.

1. **F-D1-CYLINDER-LIMITED-SCHEMES:** Implement and regression-test the two
   finite-volume scheme capabilities required by the selected official
   OpenFOAM Foundation 13 `incompressibleFluid/cylinder` reference:
   `cellLimited Gauss linear 1` for `grad(U)` and
   `bounded Gauss linearUpwind limited` for `div(phi,U)`. Keep this a numerical
   implementation task; add no generator, comparison runner, or hardening
   framework.
2. **F-D1-CASE-CYLINDER:** After those schemes are supported, add the Ferrum
   compatibility case and provide the independent OpenFOAM 13 sibling, select
   sourced observables with tolerances, and explain why no suitable closed-form
   reference is used for this finite-domain case.
3. **F-AUTO-1 (accepted external dependency):** Keep the accepted isolated n8n
   coding workflow in the AI Dev Orchestrator repository and preserve the
   existing analysis workflow as a separate read-only path.
4. **F-REF-1:** Keep a focused OpenFOAM 13 module/case reference and
   license/provenance note for each newly selected physics area. Expand it only
   when a bounded implementation task needs more detail.
5. **F-ARCH-1:** Extract the `incompressibleFluid` module registry and common solver
   lifecycle from the transitional combined crates with parity tests.
6. **F-IO-1:** Specify and implement `FerrumFile v1`; isolate OpenFOAM support behind the
   `openfoamIO` interoperability layer.
7. **F-D1D2-1:** Complete Driver 1 SIMPLE/SIMPLEC and Driver 2 PISO/PIMPLE on the scalar CPU
   reference backend for the frozen selected-case inventory.
8. **F-BACKEND-1:** Accept `ferrumRun` successively on threaded CPU, distributed CPU, one GPU,
   and multiple GPUs without changing case numerics.
9. **F-D3D7-1:** Implement Drivers 3 through 7 in the fixed order above, applying the common
   readiness gate and completing coupled `ferrumMultiRun` before Driver 6.
10. **F-POROUS-1:** Begin porous-media and packed-bed work only after Driver 7 is complete.
