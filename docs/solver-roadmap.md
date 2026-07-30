# FerrumCFD Solver Roadmap

This roadmap first completes and broadens the steady laminar incompressible
foundation, then implements exactly six additional application drivers in a
fixed order. Every driver is validated with independently authored Ferrum cases
and an analytical, manufactured, or documented external benchmark reference.
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
|   `-- ferrumModels/
|-- tutorials/
|   |-- incompressibleFluid/
|   |   |-- laminarPipe/
|   |   |   |-- shared/               # optional neutral inputs
|   |   |   |   `-- geometry/
|   |   |   |-- ferrum/
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
requested target-tree intent but remain provisional until their external
behavioral-reference and mathematical ownership audits decide whether each is an application module, a
reusable `ferrumModels` capability, or part of another module. Renaming a
provisional boundary requires a recorded architecture decision, not guesswork.

Every selected tutorial has a small user-facing contract:

- one independently runnable Ferrum case;
- an analytical reference when a useful closed form exists, otherwise a
  documented benchmark reference;
- an English README that explains the physics, assumptions, and Ferrum run
  commands;
- an optional stable result summary when maintainers have recorded a run.

`shared/geometry`, shared parameter metadata, `comparison.toml`, case
generation helpers, machine-readable reports, and refinement studies are
optional maintainer tools. They are added only when they materially help a
specific case, are never runtime dependencies, and are not prerequisites for a
user to run the Ferrum case.

`laminarPipe` and `planeChannel` are established functional bundles. Their
current tests and metadata remain useful, but no retrospective parity,
parameter-hash, lexical-hardening, or source-drift project blocks the next
physics case.

## Focused External Reference Check

The relevant external implementation and independent mathematical literature
MUST be inspected before a new Ferrum physics module, solver behavior, tutorial
case, or permanent ownership boundary is implemented.
Routine documentation, case-data maintenance, and unrelated utilities do not
need a new audit. Guessing from executable names or older OpenFOAM releases is
not an acceptable substitute.

Before implementing a new physics capability or case, inspect the relevant
OpenFOAM 13 module and tutorial rather than guessing from an older executable
name. Record only what is needed for the bounded work:

1. the external module or application and case used as a behavioral reference;
2. the fields, models, boundary conditions, schemes, and coupling algorithm
   required by that case;
3. the corresponding Ferrum ownership and any capability that is still
   missing;
4. an independent mathematical or published reference where useful;
5. license and provenance information for the external references. External
   case or mesh material is not added to the FerrumCFD repository.

Source hashes, exhaustive inventories, unchanged external reference runs,
decomposition audits, and detailed decision tables are produced only in the
maintainer environment when the task actually depends on them. They are not
distributed as part of a normal tutorial-case addition.

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
Ferrum is licensed under `GPL-3.0-or-later`; it bundles no OpenFOAM source,
binary, tutorial case, or distribution artifact. External version, protocol,
and result provenance remains documentation-only. Compatible dictionary and
mesh parsing is independently authored Rust inside Ferrum's I/O boundary.

Before Driver 2 is accepted, the roadmap lists the Driver 1 cases that were
actually selected and implemented, with their Ferrum case, available
analytical or documented external reference, and status. External cases stay
outside this repository.

The native source split follows a reviewed, acyclic dependency graph:

- `ferrumCore`: fundamental types, dimensions, registries, errors, execution
  context, and backend-neutral contracts;
- `ferrumMesh`: topology, geometry, decomposition, partitions, and interfaces;
- `ferrumFiniteVolume`: fields, operators, matrices, discretization, and
  equation assembly;
- `ferrumIO`: native `FerrumFile` parsing, writing, case I/O, and isolated
  independently authored compatibility adapters;
- `ferrumModels`: reusable physical and constitutive models.

Applications may depend on these libraries, but the libraries never depend on
an application executable. `ferrumIO` owns native Ferrum semantics; external
format compatibility remains an isolated adapter concern.

## Current Status

The canonical public entry point is now
`ferrumRun -solver incompressibleFluid`. It dispatches the executable
finite-volume pressure-velocity prototype only for unambiguous steady-state
laminar cases with exactly one SIMPLE section and no PISO/PIMPLE section.
Explicit `momentumTransport`/`turbulenceProperties` input must select
`simulationType laminar`; RAS/LES is not dispatched to the laminar kernel.
No public algorithm-specific executable or `--solveLaminarSimple` selector is
retained. Steady laminar SIMPLE and the explicit `SIMPLE.consistent` SIMPLEC
mode execute through `ferrumRun -solver incompressibleFluid`; future
PISO/PIMPLE modes remain case-selected behind that same public command and are
not implemented yet. SIMPLEC applies `adjustPhi` before the consistent
`rAtU`-based flux and HbyA correction, matching the OpenFOAM Foundation 13
operation order while deliberately retaining Ferrum's uncapped denominator
semantics. The implementation reads
OpenFOAM-like `U`, `p`,
`transportProperties`, `fvSchemes`, and `fvSolution`, uses the runtime
`constant/polyMesh` geometry, runs an uncapped SIMPLE correction path, and
writes JSON/Markdown reports including pressure-assembly diagnostics for
`rAU/rAtU`, `HbyA`, `phiHbyA`, pressure source, pressure flux, and corrected
`phi`. Reports also contain a `linearSolves` profile so medium/fine runs expose
whether the bottleneck is the non-symmetric momentum predictor or the pressure
PCG/IC(0) correction. Resolved `solvers.U.relTol` and `solvers.p.relTol` use a
strict normalized-L1 target for every linear solve. Standalone GAMG already
checks that target directly; the non-GAMG PCG path still uses a conservative
normalized-L1-to-L2 bridge followed by final normalized-L1 acceptance. A direct
normalized-L1 PCG loop remains a planned performance leaf and must preserve the
same strict public stopping contract. OpenFOAM Foundation-style outer
`residualControl` and linear-solver convergence are reported separately.
`converged=true` means the configured outer field criteria were checked and
satisfied; every momentum-component and pressure-correction solve exposes its
initial/final residuals, effective target, stop reason, iterations, and
convergence flag.

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

The merged finite-SIMPLE semantic invariant is unconditional: finite iterates
and corrections are never clipped, capped, replaced, or rolled back merely
because their magnitude is large. NaN/Inf, arithmetic overflow, linear-solver
breakdown or singularity, allocation or resource exhaustion, and
user-configured explicit iteration bounds continue to fail closed. Those
conditions are errors or explicit bounds, not finite-value magnitude caps.

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
- Keep the implemented OpenFOAM `relTol` contract under regression test.
  Implement non-GAMG `minIter` and configurable `smoothSolver` `nSweeps` in
  separate leaves; until then, reject non-default values instead of silently
  changing their meaning.
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

- `grad(p)`: `Gauss linear`, `cellLimited Gauss linear k`;
- `grad(U)`: `Gauss linear`, `cellLimited Gauss linear k`;
- `div(phi,U)`: `Gauss upwind`, `Gauss linearUpwind grad(U)`;
- `laplacian`: `Gauss linear corrected`, `orthogonal`, `uncorrected`;
- `snGrad`: `corrected`, `orthogonal`, `uncorrected`;
- interpolation: `linear`.

Next scheme targets:

- retain the accepted corrected non-orthogonal pressure-flux parity proof and
  extend it to a generated skewness/corrector matrix rather than treating the
  corrected path as wholly unimplemented;
- add bounded/limited convection schemes as explicit schemes, not hidden
  clamps;
- keep operator assembly independent from CPU/GPU solver backend code;
- add operator-level tests for face orientation, owner/neighbour signs, and
  boundary-face flux signs.

### Spatial Accuracy Track

Keep spatial accuracy independent from the pressure-kernel timing sequence:

1. add weighted least-squares scalar and vector gradients with exact constant
   and linear reproduction in an intrinsic active-dimensional basis,
   deterministic stencils, and fail-closed deficiency handling relative to
   that active dimension so valid `empty` and `wedge` meshes remain supported;
2. propagate the selected gradient through momentum assembly, pressure
   non-orthogonal correction, and explicit `0/1/2` corrector gates on open,
   closed, orthogonal, non-orthogonal, and deliberately skewed meshes;
3. replace the current lowest-order wall-force baseline with reconstructed
   wall-face pressure and full deviatoric viscous traction from reconstructed
   `grad(U)`, while preserving pressure-gauge, orientation, and area contracts;
4. run at least three geometrically similar meshes and report observed order,
   Richardson extrapolation, and GCI for `Cd`, fields, wall pressure, and wall
   shear, with iterative error demonstrably below spatial error;
5. only then repeat same-mesh OpenFOAM parity. Cross-engine parity and formal
   grid convergence remain separate claims.

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
- CPU pressure PCG has a full CSR IC(0) incomplete-Cholesky path. The current
  `DIC` and `FDIC` dictionary names are compatibility aliases to that IC(0)
  implementation, not exact OpenFOAM DIC/FDIC algorithm parity. A true
  face-LDU DIC/FDIC leaf is planned below;
- CG/PCG breakdown tests are scale-relative rather than absolute
  `f64::EPSILON` cutoffs, so valid small SI-scaled pressure systems are not
  terminated prematurely;
- backend policy already supports CPU/GPU/auto declarations and resource
  metadata, but executable GPU equation kernels are still future work.
- solver report schema version 2 now includes additive phase timings for setup,
  momentum matrix assembly, momentum linear solves, pressure matrix assembly,
  pressure linear solves, finalization, and remaining solver work;
- recorded CPU baselines exclude compilation and warmup runs from medians and
  execute the `laminarPipe` and `planeChannel` cases as independent
  regressions; their external harness is not part of the product repository.
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
- pressure-PCG kernel profiling originally existed as a low-level opt-in API,
  but the normal SIMPLE route still invoked the instrumented workspace through
  C4. C5 separates the paths explicitly: unchanged public SIMPLE entry points
  are unprofiled, while additive profiled entry points and `--profilePcg`
  collect kernel timing; incompatible solver selections fail without fallback.
  Profiled and unprofiled numerical results are bit-identical. Historical release measurements
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
  numerical observables exactly;
- the recorded canonical Linux-parity measurement built exact source commit
  `5ee13a3cde87620460f3b36d8e496a561d3a7601` with Rust `1.94.0` on WSL ext4
  and runs Ferrum and OpenFOAM Foundation 13 on pinned CPU `2` with the same
  external GNU-time metric. The portable `1+5` fixed-work run measured Pipe
  medians of `7.82 s` versus `7.40 s` and Channel medians of `7.21 s` versus
  `11.61 s`. The separate `target-cpu=native` `1+5` lane measured `7.36 s`
  versus `6.58 s` for Pipe and `6.27 s` versus `12.04 s` for Channel. A
  stronger Native Pipe `2+9` run measured a median paired Ferrum/OpenFOAM
  ratio of `1.2792` with MAD `0.1326`, confirming that Pipe remains slower;
  the Native Channel `1+5` median paired ratio was `0.5289` with MAD `0.0128`,
  so Channel is clearly faster in this fixed-work lane. Build time is excluded,
  engine-native timers remain diagnostic only, and no general all-case speedup
  is claimed;
- C4 extends the canonical Linux-parity lane to the official 5,388-cell
  Foundation 13 Cylinder mesh on exact commit `3d84b33f2406`. With one warmup
  and six measured AB/BA pairs, the exact 1,000-step fixed-work medians were
  `85.170 s` for Ferrum and `31.335 s` for OpenFOAM; the median paired
  Ferrum/OpenFOAM ratio was `2.752571` with MAD `0.495953`. The separate
  `U,p=1e-5` TTA replay used 986 Ferrum and 995 OpenFOAM outer steps and
  measured `129.680 s` versus `38.735 s`, with paired ratio `3.335669` and MAD
  `0.608799`. `Cd` differed by `0.153984%`; full-field relative L2 differences
  were `0.525295%` for `U` and `0.926074%` for `p`. Cylinder is therefore a
  remaining performance hotspot and prevents a general all-case speedup
  claim. See [Cylinder same-Linux parity](benchmarks/cylinder-linux-parity.md);
- C5 makes pressure-PCG kernel profiling genuinely opt-in on exact commit
  `1e3bbc42ad06`. Two Linux correctness oracles proved byte-identical final
  `U`/`p` and canonical reports between plain and `--profilePcg`, while the
  plain report retained exact-zero kernel timing and counters. In the pinned
  Cylinder Fixed-1,000 `1+6` diagnostic, plain median elapsed time was
  `64.280 s` (MAD `0.770 s`) and profiled was `65.055 s` (MAD `2.285 s`). The
  paired profiled/plain median was `0.993060` with MAD `0.029328`, and each
  variant won three pairs. The result establishes no measurable speed change;
  C5 is accepted for the explicit instrumentation boundary and exact parity,
  without a performance claim. A claimed sub-5% effect still requires the
  stronger `2+9` protocol;
- isolated native build screening did not establish a general compiler-profile
  gain. Native `codegen-units=1` and Fat-LTO did not improve the Pipe screen.
  Thin-LTO improved the stronger Pipe `2+9` median paired ratio from `1.2792`
  to `1.1933`, but paired MAD remained essentially unchanged (`0.1310`) and
  the ratio of separate medians remained `1.3109`. On Channel, Thin-LTO's
  `1+5` median paired ratio was `0.5493`, slightly worse than Native's `0.5289`.
  All profiles remain reproducible lanes, but none supports a general speedup
  claim; PGO remains a later isolated leaf.

### Performance Foundation - Scalar CPU

This work starts only after the selected numerical cases are correct enough to
act as regressions. It changes storage and execution mechanics, not equations,
boundary conditions, convergence criteria, or case semantics.

Acceptance criteria for every scalar-CPU optimization:

- use the release executable directly; record build time separately;
- run at least one warmup and five measured runs, reporting the median;
- require both fixed-work evidence and time-to-accuracy evidence. Measure
  convergence and external accuracy separately, and exclude build time from
  both release-run medians;
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

Performance evidence uses external maintainer environments rather than tracked
launchers or reference cases:

- same-Linux parity is the canonical external solver-comparison protocol.
  Ferrum and the reference solver run inside the same pinned native Linux
  environment with the same CPU affinity, serial thread environment,
  alternating execution order, controls, and stopping criteria. Record the
  exact Rust compiler, external solver build, kernel, CPU, and power state;
- Ferrum operating-system parity may run the same commit, build profile, and
  Ferrum-owned cases on Windows and Linux to quantify the
  operating-system/toolchain contribution. It is diagnostic only.

Compilation is excluded from all run medians. The Linux lane records one common
process measurement for both engines (`elapsed`, `user`, and `system`) and
retains each engine's native internal timer as a separate diagnostic; ratios
must not treat Ferrum wall-in-solver time and OpenFOAM CPU `ExecutionTime` as
the same metric. Portable and `target-cpu=native` builds remain separate lanes.
For a claimed gain below `5%`, use at least two warmups and nine paired measured
runs and report MAD or IQR in addition to the median.

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
14. completed: establish strict absolute and relative normalized-L1 stopping
    as the first post-flow correctness gate, retaining L2 telemetry wherever it
    is exposed, and extend the static user `relTol` contract to every current
    scalar linear solver;
15. freeze `laminarPipe` and `planeChannel` as the A/B corpus after an accepted
    post-flow clean baseline;
16. completed and accepted: retain allocation-free cached-diagonal CSR
    symmetric Gauss-Seidel as the production pressure smoother, preserving the
    frozen Pipe/Channel numerical, convergence, failure-behavior, and
    floating-point-order contract;
17. evaluated and rejected as an experimental leaf: the OpenFOAM-style
    symmetric LDU-addressed pressure Gauss-Seidel from PR `#76` changed the
    Pipe fixed-work pressure workload from `6995` to `7059` V-cycles. Its timing
    run was host-load-contaminated and therefore not accepted. PR `#77`
    restored the exact accepted pre-B2 tree from `905d698`; LDU SymGS is not a
    production path and no speedup is claimed;
18. evaluated, not accepted: two isolated SIMPLE/momentum persistence leaves
    preserved numerical semantics but failed their paired Native Linux timing
    gates. Future persistence work must isolate a measured sub-hot-path rather
    than bundle another broad workspace;
19. completed: establish the canonical Linux-parity benchmark lane before
    making further Ferrum-versus-OpenFOAM solver-performance claims;
20. continue isolated CSR residual, scaling, and row-traversal leaves because
    smoothing plus scaling dominate the measured GAMG pressure time. Cached
    diagonal values made the measured scaling phase faster but failed the
    frozen two-case total pressure-time gate and remain unmerged. Exact FCG
    reduction-traversal reuse passed every source and numerical gate and showed
    a promising Stage A, but the stricter two-case `2+10` decision gate rejected
    it; that leaf also remains local and unmerged. The separate
    reciprocal-diagonal SymGS research leaf used an explicit accuracy and
    failure-semantics contract because it changes floating-point arithmetic. It
    passed its source and field-accuracy gates but changed
    the frozen pressure-work fingerprint in both cases, so it stopped before
    Stage A and remains local and unmerged. Return to isolated hierarchy and
    exact-work reductions before attempting further arithmetic substitutions.
    A later one-file C8 PCG residual-product leaf on exact base `7b08e7d`
    fused `r dot z` and `z dot z` while reusing the known residual norm. Its
    exact-semantics proofs and isolated mechanism gate passed; Native Linux won
    `15/15` mechanism pairs with paired ratio `0.335978`. The frozen PCG/DIC
    `2+10` end-to-end gate nevertheless rejected the candidate. Pipe
    external/pressure paired ratios were `1.039508/1.051784`, Channel
    `1.119530/1.121522`, and Cylinder `0.968107/0.984870`. Cylinder won `8/10`
    pressure pairs, but its `1.513%` paired pressure gain was below
    `2 x MAD = 3.406%`. All final fields, canonical reports, work counters, and
    solver outcomes remained exact. Candidate `812ff714` stays local and
    unmerged; no rerun, publication, Fable approval, or speedup is claimed.
    This closes residual-side PCG micro-fusion as a generic default. Do not
    stack further changes onto the rejected candidate; select the next
    materially larger measured hotspot from a fresh clean base;
21. completed: add a profiled-only GAMG hierarchy diagnostic gate with exact
    level/transfer shapes, aggregate histograms, grid/operator-complexity terms,
    NNZ-weighted work proxies, and coarsest-iteration counts. Cache the static
    descriptor after the first successful profiled solve; the unprofiled path
    and all solver controls remain unchanged;
22. continue testing hierarchy leaves one at a time. The isolated
    `mergeLevels 2` experiment failed the frozen two-case work gate and remains
    unmerged; production keeps requiring `mergeLevels 1`. Cached direct coarse
    factorization also passed its semantic gates but failed the frozen two-case
    total-time gate and remains unmerged; production keeps
    `directSolveCoarsest false`. Inspect reusable work in that active iterative
    coarse path next. Keep coarsest-level target, deterministic
    aggregation/coarsening, sweep placement, and weighted prolongation
    isolated. Larger gains require fewer V-cycles or less weighted work in both
    frozen cases;
23. completed: gate the static user-bounded `relTol` implementation with Linux
    time-to-accuracy evidence and evaluate SIMPLEC as a separate same-binary
    leaf. SIMPLEC passed the frozen Channel gate but failed the frozen Pipe
    accuracy and linear-work gates, so it remains an explicit case control and
    is not a generic default. Any autonomous runtime tolerance controller is a
    separate future user-bounded experiment;
24. accept portable and native scalar profiles, then proceed through SIMD,
    shared-memory threading, and GPU as separate reviewed leaves.

n8n may build, execute, collect artifacts, compare tolerances, and reject a
regression. It must not combine several numerical or storage optimizations into
one unattended change. Each optimization remains a bounded reviewed task with
its own before/after evidence.

Next performance targets:

- retain normalized L1 as the completed post-flow correctness gate. Stopping
  requires strict absolute and relative normalized-L1 criteria; preserve L2
  telemetry where already exposed. Do not tune tolerances or iteration budgets
  to manufacture parity;
- freeze `laminarPipe` and `planeChannel` as the mandatory A/B corpus using
  identical inputs, schemes, controls, hardware lane, process/thread settings,
  and an accepted post-flow clean baseline;
- keep the accepted cached-diagonal CSR symmetric Gauss-Seidel path as the
  production baseline and preserve its exact frozen Pipe/Channel contract;
- treat symmetric LDU-addressed pressure Gauss-Seidel as a rejected experiment.
  Reopen it only as a new isolated leaf with a predeclared numerical contract
  and uncontaminated paired A/B evidence against accepted CSR;
- retain the accepted profiled-only hierarchy quality telemetry: rows and
  nonzeros per level, exact grid/operator-complexity terms, aggregate-size
  distribution, unmatched cells, per-level contraction, coarse-solve work,
  NNZ-weighted work proxies, and total V-cycles. Current Pipe/Channel grid
  complexity is already about `1.99`, so prioritize contraction quality and
  per-level work while retaining hierarchy depth and the coarsest-level target
  as isolated variables;
- run every hierarchy change as an isolated generic A/B leaf. Do not select
  settings per case, loosen tolerances, add artificial caps, or hide a fallback.
  Preserve outer convergence and physical accuracy, and report intentional
  changes in linear work instead of requiring identical V-cycle counts;
- SIMPLE/momentum persistence and the three isolated C7 scan-reuse leaves have
  failed their end-to-end timing gates despite exact semantics. Do not merge
  C6/C7 or repeat these broad workspace, final-pressure, net-flux, or
  cell-adjacency candidates. Revisit this area only after a fresh profile
  identifies a different dominant mechanism with a predeclared cross-case
  gate;
- the exact-order C8 PCG residual dot/norm fusion passed its isolated mechanism
  and complete semantic gates but failed the frozen Native Linux Pipe/Channel/
  Cylinder end-to-end gate. Keep it unmerged and do not repeat reduction-only
  PCG micro-fusions without a fresh profile showing a materially larger share;
  prioritize a larger dominant kernel such as repeated IC(0) application;
- accept the implemented static OpenFOAM-style `relTol` only with unchanged
  final outer acceptance, explicit effective-target telemetry, linear-work
  accounting, and time-to-accuracy evidence. Persistence is not a prerequisite
  for this static case control;
- keep a genuinely autonomous runtime tolerance controller separate and
  user-bounded. SIMPLEC is implemented and evaluated as its own explicit leaf:
  Channel passed the frozen accuracy, work, and timing gates, while Pipe failed
  accuracy and pressure-work gates before timing classification. Do not enable
  it automatically or select it through a hidden case heuristic; any expanded
  acceptance must be user-visible and evidence-backed on the selected case;
- define a portable release profile for reproducible comparison and correctness
  and a native release profile that is explicitly hardware-specific. Never mix
  timing claims between these lanes;
- require every accepted performance leaf to provide fixed-work and
  time-to-accuracy evidence: at least one warmup and five measured median
  release runs, build time excluded, with convergence and external accuracy
  measured separately;
- only after all scalar gates pass, accept acceleration in the order SIMD,
  shared-memory threading, then GPU. Each is a separate reviewed leaf under the
  unchanged frozen A/B contract and the universal fixed-work plus
  time-to-accuracy gate;
- keep fields, operators, equation assembly, convergence criteria, reports, and
  case semantics independent of execution backend;
- after those acceleration leaves and the selected Driver 1/2 inventory gate,
  undertake explicitly named partitioned multi-process CPU, multi-node CPU,
  multi-GPU, and multi-node CPU/GPU backend integration;
- require every backend leaf to use identical case inputs and numerical
  schemes, with stated tolerance parity, conservation checks, deterministic
  regression mode, scaling efficiency, and memory-transfer measurements;
- keep GPU optional and selectable per stage (`flow`, linear solves,
  nonlinear/interface/ODE stages), with CPU as a valid choice when GPU is busy,
  unsupported, or inefficient.

Rust threads can use all cores of one shared-memory host without MPI. Rust does
not remove the distributed-memory problem: multiple processes, multiple nodes,
and some multi-GPU layouts still require MPI or an equivalent transport. Before
partitioned multi-process CPU work, record an architecture decision comparing a Rust MPI binding with
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
tutorial matrix. The status column is the audited repository state on
2026-07-30; a smoke run is not the same as full physical acceptance.

| Order | Case | Primary coverage | Current status | Reference |
| ---: | --- | --- | --- | --- |
| 1 | `laminarPipe` | 3D internal flow and pressure loss | Runnable case, analytical checks, convergence profiles, and performance evidence present | Hagen-Poiseuille analytical solution |
| 2 | `planeChannel` | true 2D `empty` handling | Runnable case, analytical checks, executable `empty` coverage, convergence profiles, and performance evidence present | Plane-Poiseuille analytical solution |
| 3 | `cylinder` | steady laminar external flow at `Re = 1` and corrected non-orthogonal pressure coupling | Independently authored 48-cell `LegacySmoke` case, accepted C2 deterministic Coarse/Fine O-grids, generic force/continuity reporting, complete C3 Coarse/Coarse/Fine physical gates, and accepted C4 same-Linux fixed-work/TTA/field-parity evidence | [Cylinder same-Linux parity](benchmarks/cylinder-linux-parity.md) against documented OpenFOAM Foundation 13 observables |
| 4 | `lidDrivenCavity` | recirculation and closed-pressure reference | Independently authored 4-cell case plus preflight and two-SIMPLE-iteration E2E smoke present; the nonzero `pRefValue` is verified in the written field and all eight fixed-velocity faces exercise the pressure constraint; refined centerline/vortex acceptance remains open | Ghia, Ghia, and Shin Re=100 published benchmark |
| 5 | `backwardFacingStep` | separation, reattachment, outlet robustness, and actual reverse-flow switching | Not implemented as an executable Ferrum tutorial | Published benchmark |
| 6 | `axisymmetricPipe` | executable `wedge` handling | Not implemented as an executable Ferrum tutorial | Hagen-Poiseuille analytical solution |

For acceptance, every case must contain an independently runnable `ferrum/case`
directory, an English README, and an analytical reference when one is useful.
Otherwise the README points to a documented external benchmark.
Shared inputs, comparison metadata, recorded results, and mesh variants are
optional and case-specific. No combined runner is required.

### Audited Driver 1 priority gate (2026-07-29)

The next quality jump is the correctness matrix and ownership split, not
another unbounded sequence of GAMG micro-optimizations:

1. retain the new Lid Cavity closed-pressure E2E smoke for
   `pRefCell`/`pRefValue` and `constrainPressure`, then complete the remaining
   deliberately skewed corrected non-orthogonal E2E mesh and physical cavity
   acceptance; open-pressure `adjustPhi` and pressure-flux coverage remains
   part of the combined gate;
2. retain the accepted Cylinder force/continuity gate, refine the
   `lidDrivenCavity` physical acceptance, then add `backwardFacingStep` and
   `axisymmetricPipe` in matrix order;
3. add executable direction-changing backflow coverage and executable
   `wedge` and `symmetryPlane` solver cases. Existing unit coverage and the
   Plane Channel/Cylinder `empty` runs remain necessary but are not the full
   boundary-condition matrix;
4. split the transitional 10,018-line `flow.rs` and 10,390-line
   `linear/gamg.rs` behind parity-tested crate and module APIs before adding
   PISO/PIMPLE, thermal physics, or GPU kernels;
5. resume acceleration only as separately reviewed leaves in the order SIMD,
   deterministic shared-memory momentum threading, then GPU. The first AVX2
   leaf was rejected; threading and executable GPU kernels are still planned,
   not accepted product capabilities.

A bounded micro-optimization may still be measured when profiling identifies a
generic cross-case hotspot, but it must not displace these gates and must retain
the existing two-case numerical and timing acceptance contract.

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
2. accept the scalar correctness and performance sequence, then the separate
   SIMD, shared-memory threading, and single-GPU leaves on the frozen A/B
   contract, while making operators and storage backend-neutral;
3. define a backend-neutral execution context that distinguishes sockets,
   cores, worker threads, process ranks, domain partitions, GPU devices,
   memory, and queues while preventing oversubscription;
4. after the selected SIMPLE/SIMPLEC/PISO/PIMPLE inventory passes, implement
   and accept the named `ferrumRun` partitioned multi-process CPU, multi-node
   CPU, multi-GPU, and multi-node CPU/GPU backend integration leaves in that
   order, reusing the already accepted SIMD, shared-memory, and single-GPU
   leaves;
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
later acceleration does not require an architectural rewrite. The partitioned,
distributed, multi-node, and multi-GPU integration kernels start only after the
selected Driver 1/2 inventory gate; this does not repeat the earlier accepted
SIMD, shared-memory, or single-GPU leaves.
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
- an analytical reference is supplied where useful, otherwise a documented
  benchmark identifies the source and observables;
- implemented solver behavior has focused automated unit or integration
  regression coverage with stated acceptance tolerances;
- maintainers record at least one successful result for each selected case;
- case-specific reference logic remains outside the generic driver.

External comparison runners and exhaustive refinement studies remain optional
maintainer tools selected for a concrete numerical risk. They are not tracked
product files or part of the user-facing case contract.

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
- later backend expansion: `F-BE-PARTITION`, `F-BE-MULTIPROCESS`,
  `F-BE-MULTINODE`, `F-BE-GPUN`, and `F-BE-MULTINODE-GPU` in that order,
  reusing the earlier accepted `F-BE-THREADS` and `F-BE-GPU1` leaves;
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

The immediate sequence is:

1. **F-CORRECT-NL1 (completed/accepted):** Strict absolute and relative
   normalized-L1 stopping is established with exposed L2 telemetry retained
   and without tolerance or iteration-budget tuning.
2. **F-PERF-CSR-SYMGS (completed/accepted):** Cached-diagonal,
   allocation-free CSR SymGS is the production baseline and passed the frozen
   two-case parity gate.
3. **F-PERF-LDU-PRESSURE-SYMGS (experiment rejected/reset):** PR `#76` was not
   accepted because workload parity changed and timing evidence was
   contaminated; PR `#77` restored the exact pre-B2 tree. LDU remains
   experimental only.
4. **F-PERF-SIMPLE-PERSISTENCE (first leaf rejected):** Persist valid
   SIMPLE/momentum topology, coefficients, preconditioner state, histories, and
   workspaces with explicit invalidation and unchanged equations, boundaries,
   and convergence semantics. The first unmerged P1 candidate preserved exact
   reports, work counters, and final `U`/`p` bits, but the same-session Native
   Linux `2+10` A/B was order-sensitive in both cases: Pipe had a paired median
   ratio of `0.9500` with opposing cohorts and Channel `0.9837` with opposing
   cohorts. It therefore provides no accepted speedup and remains unmerged.
   A second one-file C6 leaf on base `c4347229` retained momentum assembly
   matrices, right-hand sides, old fields, optional gradients, source,
   diagonal, relaxation, and H1 storage. Exact Pipe/Channel/Cylinder field and
   canonical-report parity passed. Native Linux `1+6` medians nevertheless
   regressed by `9.16%`, `8.27%`, and `0.80%`, respectively, with only `3/6`
   wins in every case and split order cohorts. Candidate `c3a30ea` remains
   local and unmerged; no performance or Fable claim is made. Any later
   persistence leaf must be smaller and independently profiled.
   Three subsequent one-file C7 leaves isolated intermediate diagnostic
   summarization of pressure data (`9f0fdd2d`), shared bounded net-cell flux
   (`2c332663`), and lazy
   solve-local cell-to-face adjacency (`038b78b8`). All three retained exact
   fixed-work fields, reports, iterations, failures, and input hashes across
   Pipe, Channel, and Cylinder. The frozen Native Linux `1+6` gates rejected
   every leaf: C7-A regressed Pipe; C7-B crossed the Pipe no-path regression
   gate; and the directionally favorable C7-B/C Cylinder paired gains of
   `4.011%` and `2.600%` did not exceed their respective `2 x MAD` thresholds
   of `4.443%` and `8.884%`. No confirmation run was permitted. All C7 source
   commits remain local and unmerged, and no speedup or Fable claim is made.
5. **F-PERF-LINUX-PARITY (completed evidence, harness retired):** The external
   same-Linux protocol produced accepted portable, Native `1+5`, and Native
   Pipe `2+9` baseline evidence with pinned Rust, CPU affinity, serial
   environment, and common process timing. PR `#80` merged the former harness
   as `7f71427`; the July 2026 repository cleanup removed that harness and all
   external cases while retaining version, protocol, and result provenance.
6. **F-PERF-GAMG-SCALING-ROW-KERNELS (first leaf rejected):** Continue with
   isolated CSR residual, scaling, and row-traversal leaves before structural
   hierarchy changes. A bit-exact paired-dot scaling candidate passed the full
   workspace and field-parity gates, but the stronger Native Linux `4+20` A/B
   exposed a dominant order effect in Pipe: candidate-first had median ratio
   `1.0011`, while candidate-second had `0.8128`. Channel's paired median ratio
   was `0.9714`, but MAD was `0.0993`. The candidate therefore remains
   unmerged and supports no speed claim; later row-kernel leaves must retain
   the same exact-semantics and order-cohort gates.
   Smoothing plus scaling currently account for about `86.7%` of Pipe and
   `67.8%` of Channel GAMG profile time, substantially more than hierarchy
   infrastructure.
   A second bit-exact leaf (`eb04d7a`) fused the existing CSR matrix-vector
   traversal with scaling-denominator accumulation while preserving the exact
   entry, row, and floating-point addition order. It passed the complete
   workspace, Rust 1.94 Clippy, adversarial re-association, lifecycle, failure-
   boundary, and field-parity gates. The stronger Native Linux `2+10` A/B
   nevertheless rejected it: Pipe's process medians were `2.445 vs 2.390 s`,
   but candidate-first and candidate-second cohorts disagreed, giving an
   inconclusive paired ratio of `0.9594` with `0.1061` MAD. Channel was slower
   in both cohorts, with process medians `6.605 vs 6.930 s` and paired ratio
   `1.0077`. All canonical reports and final `U`/`p` fields remained bit-exact.
   The leaf therefore stays local and unmerged, and no speedup is claimed.
7. **F-PERF-GAMG-HIERARCHY-DIAGNOSTIC (accepted telemetry-only package):** A
   clean-base package on `517fdfa` adds exact level and transfer shapes,
   aggregate-size histograms, unmatched-cell counts, grid/operator-complexity
   terms, NNZ-weighted work proxies, and iterative coarsest-solve counts. The
   static descriptor is cached only after the first successful profiled solve
   and then reused by `Arc`; unprofiled solves perform no diagnostic build.
   Rust 1.94 formatting, locked/offline all-target Clippy, 488 workspace tests
   plus one intentionally ignored release-only performance test, exact schema,
   fail-closed, accumulation-atomicity, cache-lifecycle, and ten-coefficient
   proofs passed. Fresh WSL2/ext4 CPU2 profiled FCG oracles retained exactly
   `745/7021` Pipe/Channel V-cycles, all prior counters, byte-identical final
   `U` and `p`, and reports identical to the prior profiled oracles after
   removing timing and the new telemetry. Pipe/Channel have 9/8 levels, exact
   grid-complexity terms `9197/4608` and `3983/2000`, and exact
   operator-complexity terms `61667/30336` and `19007/9760`. Their
   NNZ-weighted smoothing/sparse-work proxies are respectively
   `266304720/381274010` and `736011430/1071131186`; iterative coarsest work is
   `2137/6771` iterations. These profiled oracles prove observational parity,
   not a speedup. Focused Fable review remains explicitly deferred.
8. **F-PERF-GAMG-HIERARCHY-LEAVES (equal-weight pairing accepted;
   `mergeLevels 2` and cached direct factorization rejected):** Test one
   predeclared variable per A/B:
   coarsest-level target; cached direct versus iterative coarse solve;
   deterministic pairing/coarsening; pre/post/finest sweep placement; weighted
   prolongation; and, later, GAMG as a PCG/FCG preconditioner. Invalidate and
   rebuild any cached direct coarse factorization for every coarse-coefficient
   lifecycle. Treat `2-4%` as an experimental pressure-solver hypothesis only
   when V-cycles or weighted work fall on both cases; the approximately `3%`
   measured infrastructure time is a ceiling, not a savings target.
   The first local leaf, `f13af9b`, made a zero `nPreSweeps` base disable its
   level multiplier like OpenFOAM Foundation 13. Source, GAMG, workspace, and
   Clippy gates passed (with the test-only profile update in child commit
   `a153db2`), but the Native Linux `0+2` Go/No-Go failed decisively: Pipe hit
   the 1000-iteration pressure limit in all 10 SIMPLE steps (`10000` pressure
   iterations versus baseline `6995`), while Channel rose to `33950` versus
   `10654`. Candidate process times were also slower in every Channel run and
   in both Pipe runs. The leaf remains unmerged and did not proceed to `4+20`.
   The next isolated leaf, commit `9b6befb`, keeps connection weight as the
   unrestricted primary criterion and changes only exact-weight ties: it
   prefers the pair with the smaller external neighbour stencil, with
   direction-symmetric canonical endpoint ordering. The one-file candidate
   passed the complete Rust 1.94 workspace and Clippy gates plus an independent
   determinism audit. On the profiled Pipe, pressure V-cycles fell from `6995`
   to `1687`, weighted smoothing visits from `2520214560` to `603028272`, and
   level-3 nonzeros from `4352` to `4064`; Channel hierarchy, work, and final
   fields remained exact. The Native Linux `2+10` same-session A/B measured a
   Pipe internal-time median of `7.3913 -> 2.2088 s` with paired ratio
   `0.2895`; both order cohorts agreed. Channel retained identical work and
   fields, while timing remained order-sensitive and is classified neutral.
   Pipe pointwise relative L2 field differences were `3.17e-11` for `U` and
   `4.90e-10` for `p`; all pressure solves converged and boundary values were
   exact. A separate matched Native Linux `2+9` run against OpenFOAM Foundation
   13 measured process medians of `2.24 vs 6.65 s` for Pipe and
   `6.47 vs 10.97 s` for Channel, ratios `0.3368` and `0.5898`. These are
   fixed-work results on the two frozen cases, not a general all-case speed
   claim. PR `#79` merged the isolated one-file implementation as `4a0e2f3`
   after CI and Trusted Merge passed.
   A direct child experiment (`73e3c5b`) then replaced the local cell-order
   pass with deterministic global heavy-edge matching. Its 37 focused GAMG
   tests, complete workspace, and Rust 1.94 Clippy gates passed after updating
   the intentionally hierarchy-bound normalized-L1 profile oracle in child
   commit `c445b39`. The Native Linux `0+2` Go/No-Go nevertheless failed
   decisively against `9b6befb`: Pipe pressure iterations rose from `1687` to
   `7764` and process times from `1.98/3.05 s` to `8.15/8.82 s`; Channel rose
   from `10654` to `16808` iterations and `6.93/6.59 s` to `9.51/8.55 s`.
   Every pressure solve still converged and boundary fields remained exact,
   so this is an algorithmic efficiency rejection rather than a breakdown.
   The global-matching leaf remains local and unmerged.
   A subsequent control-only `nFinestSweeps 2 -> 1` probe also remains
   unmerged. Pipe's Native Linux `0+2` gate initially suggested a `0.9679`
   paired ratio, but the required `2+10` run reversed that result: process
   medians were `2.29 vs 2.30 s`, the paired median ratio was `1.1141` with
   `0.1669` MAD, and both order cohorts classified the candidate as slower.
   Reports and final fields stayed bit-exact. Channel's `0+2` probe was
   inconclusive because its two order cohorts disagreed, so the predeclared
   both-case gate stopped before a stronger Channel run.
   A later clean-base `mergeLevels 2` implementation on `44946ce` matched
   OpenFOAM Foundation 13 grouping semantics, retained a separate exact
   `mergeLevels 1` path, and passed its focused/GAMG, formatting, Clippy, and
   independent-review gates. The fresh Native Linux CPU2 `A-B-B-A` Go/No-Go
   nevertheless rejected it as a generic default. Pipe levels fell `9 -> 5`
   and NNZ-weighted sparse work fell `381274010 -> 345285600` (`-9.44%`), even
   though V-cycles rose `745 -> 1215`; its two-run process and internal
   pressure medians improved by about `13.33%` and `11.94%`. Channel levels
   fell `8 -> 5`, but V-cycles rose `7021 -> 18776`, sparse work rose
   `1071131186 -> 1654088736` (`+54.42%`), and process and internal pressure
   medians regressed by about `19.55%` and `37.95%`. All fixed-work,
   convergence, determinism, and field-parity gates passed, and final
   continuity remained below `3e-17`, but the predeclared both-case work gate
   failed, so the expensive `2+10` and converged runs were not started. The
   implementation remains unmerged, `mergeLevels 1` remains the default, and
   no speedup is claimed. Focused Fable review remains deferred.
   The following one-file clean-base experiment on `44946ce` cached the dense
   direct coarsest-level factorization within one coarse-coefficient lifecycle.
   It preserved lazy construction, invalidation before coefficient mutation,
   stable allocations across ten lifecycles, failure and retry behavior, and
   exact legacy arithmetic. Its 56 focused GAMG tests, full workspace,
   formatting, Rust 1.94 locked/offline Clippy, and two independent reviews
   passed. A fresh Native Linux CPU2 `A-B-B-A` Stage A compared the old and
   cached implementations with `directSolveCoarsest true` on both sides. All
   12 reports, final `U` and `p` fields, convergence counts, and numerical
   semantics were exact. Pipe total internal and pressure medians improved by
   `27.01%` and `29.51%`, while its profiled coarse phase improved by `80.87%`.
   Channel's coarse phase also improved by `68.50%`, but its total internal and
   pressure medians regressed by `8.27%` and `9.95%`. The predeclared both-case
   total-time gate therefore stopped before `2+10` and Stage B. The cache stays
   local and unmerged, production retains `directSolveCoarsest false`, and no
   general speedup is claimed. Evidence manifest SHA256 is
   `241d6d9e722382af268b9dcb4560ef9607187407f41b74187d6573bb0c249b3f`.
   Focused Fable review remains deferred.
9. **F-PERF-ADAPTIVE-LINEAR (semantics/work accepted; timing not accepted):**
   Commit `206f7ee` implements static, user-bounded OpenFOAM-style `relTol` for
   all current scalar linear solvers while preserving final outer acceptance.
   It reports every effective target, stop reason, and linear-work contribution.
   This is not an autonomous runtime controller. The canonical same-Linux
   `2+10` Pipe/Channel proof passed every accuracy and work gate, reducing total
   linear work by `22.37%` and `38.34%`. Pipe's process median was `13.34%`
   lower with `9/10` wins, but its gain did not exceed `2 x MAD`; Channel's
   median was `4.55%` lower with only `6/10` wins and opposing order cohorts.
   The feature is therefore publishable as a correctness, compatibility, and
   work-reduction package, but the performance leaf remains unaccepted and no
   stable end-to-end speedup is claimed.
10. **F-D1-SIMPLEC (implemented; case-dependent acceptance):** The correction
    now follows the OpenFOAM Foundation 13 order (`adjustPhi` before the
    consistent `rAtU` flux/HbyA correction) without importing OpenFOAM's
    artificial denominator cap. A same-binary Native Linux `2+10` A/B changed
    only the direct top-level `SIMPLE.consistent false -> true` token and kept
    p/U `relTol` at zero. Pipe was rejected before timing classification: its
    velocity relative L2/Linf were `4.976e-5 / 1.091e-4`, gauge-pressure L2 was
    `1.308e-4`, pressure-drop difference was `1.158e-4`, and total linear work
    increased `41.57%` (`37,968 -> 53,750`) because pressure work increased
    `57.35%`. Channel passed separately: SIMPLE iterations fell `545 -> 441`,
    total linear work fell `21.44%` (`23,442 -> 18,417`), and the process-time
    median fell `6.67 -> 5.89 s` (`11.69%`) with `8/10` paired wins and both
    order cohorts faster. SIMPLEC therefore remains an explicit opt-in with no
    general speed claim or global-default change. A future autonomous tolerance
    controller remains a separate user-bounded experiment.
11. **F-PERF-PORTABLE-NATIVE / NATIVE-PGO (harness accepted, performance leaf
    rejected):** Thin/Fat LTO, `codegen-units=1`, and the separate
    `target-cpu=native` leaves did not establish a both-case general win. The
    isolated Native-PGO lane builds one exact commit three ways (Native,
    instrumented, PGO), trains only on canonical fixed-work GAMG Pipe then
    Channel, and binds the toolchain LLVM/profile/binary evidence. Smoke is
    exactly `0+2/all` and never decision-eligible; only `2+20/all` may be
    accepted, and both cases must independently pass median, `14/20` wins,
    both-order-cohort, `> 2 x MAD`, canonical-report, and final-field IEEE-754
    gates. The exact `2f16dd9` decision run preserved canonical reports and
    final `U`/`p` bits but rejected PGO. Pipe medians were `3.12 -> 3.38 s`
    (ratio of medians `1.0833`, paired ratio `1.0443`, `7/20` wins); both order
    cohorts were slower. Channel medians were `6.67 -> 6.305 s`, but only
    `11/20` pairs won, order cohorts disagreed (`1.0356` versus `0.9260`), and
    the gain did not exceed `2 x MAD`. Native-PGO therefore remains a
    reproducible diagnostic lane, not a default or a general speed claim. Keep
    the portable release as the distribution profile. Focused Fable review is
    deferred until its quota is available; no Fable approval is claimed by
    this harness package.
12. **F-PERF-GAMG-INTERPOLATE-CORRECTION (capability accepted; performance leaf
    rejected):** PR `#93` implemented Foundation-style correction
    interpolation for the serial symmetric-CSR GAMG path while keeping it
    disabled by default. Rust 1.94 formatting, Clippy, 477 passed tests plus one
    intentionally ignored performance test, release preflights, PR CI, Trusted
    Merge, and exact post-merge CI passed on merge `be039401`; focused Fable
    review remains deferred. A same-binary
    Native Linux `0+2` go/no-go then changed only the direct
    `solvers.p.interpolateCorrection false -> true` control. Pipe pressure work
    increased `1687 -> 2448` iterations and its process median increased
    `2.645 -> 4.225 s` (ratio `1.5974`). Channel pressure work increased
    `10654 -> 11064` iterations and its process median increased
    `6.815 -> 8.125 s` (ratio `1.1922`). All linear solves completed and the
    report-level velocity/pressure L2 differences stayed below `4e-11`, but the
    work and timing regressions in both cases stopped the expensive `2+10`
    decision run. Keep the option explicit and default-off for compatibility;
    do not use it as a performance default or claim a speedup from it.
13. **F-PERF-GAMG-STATIC-LEAF-CLOSURE (generic defaults rejected):** Five
    Native Linux CPU2 `0+2` BCCB smokes on exact main `504f4a49` closed the
    selected small static leaves without a decision-eligible speed claim.
    The bit-exact residual/matrix-vector fusion candidate `a9da122` changed
    Pipe process/pressure medians by `-3.65%/+10.58%` and Channel by
    `+18.96%/+28.05%`; reports and final fields were bit-exact, but the Channel
    regression rejected the unmerged leaf. Switching the coarsest solve from
    iterative to direct changed Pipe process/pressure medians by
    `+7.45%/+12.54%` and Channel by `+1.26%/-2.72%`; Pipe pressure iterations
    changed only from `1687` to `1668`, so direct solve is not a generic
    default. Increasing the coarsest-cell target from `10` to `20` reduced
    Pipe process/pressure medians by `47.36%/56.92%`, pressure iterations from
    `1687` to `830`, and weighted smoothing visits from `43,727,040` to
    `21,274,560`; Channel instead regressed by `8.15%/12.95%`, with iterations
    rising from `10,654` to `12,282` and visits from `118,461,826` to
    `133,517,622`. Reducing `maxPostSweeps` from `4` to `3` improved Pipe
    process/pressure medians by `5.08%/2.15%`, with iterations `1687 -> 1660`
    and visits `43,727,040 -> 41,174,640`, but regressed Channel by
    `2.00%/11.12%`, with iterations `10,654 -> 11,489` and visits
    `118,461,826 -> 122,369,339`. Setting
    `postSweepsLevelMultiplier 1 -> 0` regressed Pipe process/pressure medians
    by `3.55%/1.37%`, with iterations `1687 -> 1966` and visits
    `43,727,040 -> 44,305,776`; it improved Channel medians by
    `15.15%/9.20%` despite iterations rising `10,654 -> 11,931`, while visits
    fell `118,461,826 -> 115,527,873`. The option-only leaves are therefore
    case-dependent controls, not global defaults. No general speedup is
    claimed from these smoke runs.
    A later one-file clean-base leaf on `ec791d8` reused the existing contiguous
    per-level diagonal-value cache in correction scaling instead of loading
    each diagonal through its CSR slot. It changed no arithmetic, division,
    traversal, allocation, counter, hierarchy, tolerance, or failure order.
    Its focused legacy oracle, 53 GAMG tests, 489 workspace tests plus one
    intentional performance ignore, Rust 1.94 locked/offline Clippy, and two
    independent reviews passed. A fresh Native Linux CPU2 Stage A then ran 48
    valid FCG Pipe/Channel jobs with one warmup plus five measured runs per
    variant and mode. Reports, final `U`/`p`, residuals, iterations, V-cycles,
    hierarchy, and weighted work remained bit-exact. The profiled scaling
    medians improved by `7.21%` for Pipe and `2.69%` for Channel, confirming the
    intended local effect. The primary unprofiled pressure gate nevertheless
    regressed: Pipe raw and paired medians increased by `16.04%` and `19.35%`,
    with both order cohorts slower; Channel's raw median increased by `20.89%`
    under high drift and its paired median increased by `0.52%`, again with
    both cohorts slower. The predeclared gate therefore stopped before `2+10`
    and Stage B. The leaf stays local and unmerged, and no speedup is claimed.
    Evidence manifest SHA256 is
    `d70997d936d10bfd1c960cbb04028e32432c7f1eedaaedfc63fe8bf6e58064d6`.
    A subsequent two-file clean-base research leaf on `5608321` replaced the
    private per-level GAMG diagonal-value cache with same-sized reciprocal
    storage. It used multiplication only for normal reciprocals and finite
    products separated from IEEE underflow and overflow boundaries; signed
    zero, subnormal, non-finite, and extreme updates retained the exact prior
    CSR-diagonal division. This selected arithmetic rather than clamping a
    result, added no second cache, and left public standalone Gauss-Seidel,
    hierarchy, sweeps, tolerances, FCG recurrence, scaling, and interpolation
    semantics unchanged. Eleven focused reciprocal proofs, 56/56 debug and
    56/56 release GAMG tests, all workspace and integration gates, Rust 1.94
    locked/offline Clippy, and two independent source audits passed. The final
    audit included `faceAreaPair`, `symGaussSeidel`, iterative coarse solve, L2
    and normalized L1, profile parity, and a scanned-division oracle.

    A fresh Native Linux CPU2 converged pre-gate then ran exact base/candidate
    binaries for Pipe and Channel. All four jobs exited successfully with empty
    stderr. Both cases retained their SIMPLE counts, exact written boundary
    blocks, convergence, continuity below `3e-17`, and field and pressure-drop
    differences inside the frozen contract. The complete pressure-work
    fingerprint nevertheless changed. Pipe pressure iterations and V-cycles
    increased from `11725` to `11735`, and logical reductions increased from
    `105318` to `105408`. Channel retained `7381` pressure V-cycles but increased
    iterative coarsest-level work from `7131` to `7133` iterations. The
    predeclared gate therefore rejected the leaf before any Stage A timing;
    Stage A and `2+10` each ran zero jobs. The code stays local and unmerged, and
    no speedup or publication claim is made. The 546-entry evidence manifest
    SHA256 is
    `31a155c59a2b70700cfe2c498439090eefb6cbb7e63b1608a20f992e5807c86a`.
    A subsequent same-binary Native Linux CPU2 hierarchy leaf changed only
    `nPostSweeps 2 -> 1` in external copies of the frozen FCG Pipe and Channel
    cases. The initial unprofiled/profiled `0+2` smoke passed every numerical
    and structural gate. Weighted sparse work fell by `6.88%` for Pipe and
    `5.26%` for Channel, while V-cycles rose by `4.03%` and `6.20%`, remaining
    below the predeclared work break-even limits. All linear solves completed,
    final continuity stayed below `3e-17`, and final velocity, gauge-pressure,
    boundaries, hierarchy, and all options except `nPostSweeps` remained within
    the frozen contract.

    The strict same-binary `2+10` decision run did not establish a robust
    two-case time win. Pipe's raw and paired pressure medians improved by
    `0.88%` and `3.22%`, but only `6/10` pairs won and the paired improvement
    `0.03224` was below `2 * MAD = 0.22043`. Channel's raw median improved by
    `4.23%`, while its paired median regressed by `0.45%`; only `5/10` pairs
    won and the order cohorts disagreed (`+0.96%` versus `-13.96%`). The leaf
    is therefore rejected as a generic default. No converged or
    time-to-accuracy runs followed, no source default changed, and no speedup
    is claimed. The 1,575-entry evidence manifest SHA256 is
    `ca7700052f30a37b24448f19a27a38065ce9f41949eda488ceeaa44f8dd638b0`.
    A further same-binary Native Linux CPU2 smoke changed only
    `scaleCorrection true -> false` in external copies of the same frozen FCG
    cases. All 16 fixed-work processes exited successfully with empty stderr,
    all linear solves completed, the candidate recorded zero scaling calls and
    scaling time on every GAMG level, and the final fields, gauge pressure,
    pressure-drop proxy, hierarchy, and boundaries stayed inside the frozen
    A/B contract. Pipe pressure V-cycles fell `745 -> 667` and weighted sparse
    work fell `381274010 -> 289916094` (`-23.96%`), but the candidate's final
    continuity L2 was `3.16860996327233e-17`, above the strict `3e-17` gate.
    Channel exposed the decisive generic regression: pressure V-cycles and
    linear iterations rose `7021 -> 18270` (`2.6022x`), iterative coarsest work
    rose `6771 -> 17468` (`2.5798x`), and weighted sparse work rose
    `1071131186 -> 2357471360` (`+120.09%`). The smoke therefore stopped before
    `2+10`; `scaleCorrection=true` remains the generic default, no source or
    case changed, and no speedup is claimed. The 389-entry failure-evidence
    manifest SHA256 is
    `699ddbbc4ce725e8b1a077034544544c0195245a68b551a36ffd157189b24cfa`.
    A subsequent two-file clean-base research leaf on `48f3bc5` packed each
    GAMG level's off-diagonal CSR row entries into reusable contiguous storage.
    Column indices used a lossless `u32` representation when possible and an
    automatic `usize` fallback otherwise. Values were refreshed in place after
    each finest or coarse coefficient update. Diagonal division, row and
    arithmetic order, hierarchy, sweeps, tolerances, failure semantics, public
    APIs, and work counters remained unchanged. Five focused packed-layout
    proofs, an integrated FCG/`faceAreaPair` cross-product proof, all 58 debug
    and 58 release GAMG tests, the full 494-test workspace gate plus one
    intentional performance ignore, the release pressure-matrix gate, Rust
    1.94 locked/offline Clippy, and two independent source audits passed.

    Native Linux CPU2 pre-gates then passed all 16 jobs with bit-exact reports,
    fields, stop reasons, hierarchy, options, and work fingerprints. Stage A
    completed all 48 planned jobs with exact semantics. The primary unprofiled
    pressure medians improved by `8.03%` for Pipe and `4.09%` for Channel, and
    paired medians improved by `10.52%` and `2.79%`. The strict robustness gate
    nevertheless rejected the leaf: Pipe won only `3/5` pairs and its
    B-leading cohort regressed to a `1.08252` candidate/base ratio, despite an
    A-leading ratio of `0.89485`. Profiled evidence also showed smoothing
    regressions of `17.48%` for Pipe and `9.33%` for Channel, with total
    profiled GAMG-path regressions of `23.26%` and `17.52%`. Stage B and all
    additional measurements therefore remained at zero. The source stays local
    and unmerged; no speedup or publication claim is made. The 1,593-entry
    evidence manifest SHA256 is
    `e345cafd36641646d776089af793adbafec7c8846b6410acbcd49adb0f859db5`;
    its manifest-seal SHA256 is
    `82579ff52d8ff35a632cf3837fbbb6fa37203b24652b5c0863c20152aecba120`.
    A later one-file clean-base leaf on `970a20d` retained the checked public
    `CsrMatrix::matvec_into` contract but moved its hot loop into a private
    validated kernel. Narrowly documented unchecked access removed redundant
    row-offset, entry, and gathered-vector bounds checks after constructor and
    shape validation. The exact row order, CSR-entry order, multiplication and
    addition expression, error order and text, pre-error output invariance,
    allocation behavior, and public API remained unchanged. Five focused
    debug and release proofs covered every public constructor, rectangular and
    zero-column CSR, unsorted duplicate columns, ten `values_mut` coefficient
    lifecycles including IEEE special values, legal disjoint alias sources, and
    exact bit parity with the former safe-index oracle. All 52 debug and 52
    release GAMG tests, 493 workspace tests plus one intentional performance
    ignore, the release pressure-matrix gate, Rust 1.94 locked/offline Clippy,
    formatting, and an independent unsafe-soundness review passed.

    Native Linux CPU2 pre-gates passed all 16 jobs with exact fields, reports,
    hierarchy, options, stop reasons, convergence, and work fingerprints.
    Stage A completed 48 further jobs with exact semantics, but did not pass the
    frozen robustness gate. Pipe's raw and paired pressure medians improved by
    `10.41%` and `4.59%`, yet only `3/5` pairs won and the B-leading cohort
    regressed to a `1.1355` candidate/base ratio. Channel improved by `3.44%`
    raw and `3.40%` paired with both cohorts below one, but also won only `3/5`
    pairs instead of the required four. Supporting profiled smoothing medians,
    which this leaf does not change, regressed by `20.04%` for Pipe and `4.06%`
    for Channel, confirming material run-order drift rather than a robust local
    kernel result. One malformed non-primary GNU-time wall sample caused by a
    WSL clock jump was excluded and documented without a rerun; every primary
    internal report remained valid. Stage B and all additional measurements
    stayed at zero. The source remains local and unmerged, and no speedup is
    claimed. The 1,602-entry evidence manifest SHA256 is
    `77cd6b32859edff70c258b3afbcef37cdfaa7923a7081969655c298567e0945b`;
    its manifest-seal SHA256 is
    `79bfdfada22d0ec6c75035a0a9efaa28b88820656ebc4705bff367744574ab41`.
    A further two-file clean-base research leaf on `41ed598` retained the
    public symmetric Gauss-Seidel implementation as the exact legacy oracle
    while adding an internal caller-owned path for momentum solves. Three old
    and three solution component vectors were retained across SIMPLE steps,
    and the existing residual and matrix-product workspace was shared
    sequentially across the three components. Arithmetic and traversal order,
    tolerance comparisons, errors, reports, aggregation, non-SymGS behavior,
    and public APIs remained unchanged. Five focused linear and three momentum
    proofs passed in both debug and release, including exact and `next_up`
    stopping boundaries, all failure paths, repaired retries, ten coefficient
    lifecycles, XYZ aggregation, and stable pointers and capacities. The final
    496-test workspace gate plus one intentional performance ignore, Rust 1.94
    locked/offline Clippy, formatting, and an independent source audit passed.

    Native Linux CPU2 pre-gates passed all 16 jobs, and Stage A completed all
    48 jobs with exact semantic parity. An independent seal audit found one
    non-decision metadata defect: `stage-a-results.json` retained the stale
    descriptive string `unprofiled pressureLinearSolveSeconds`, while the
    frozen protocol, gate calculations, summary, and decision correctly used
    `unprofiled solverTotalSeconds` as the primary metric. The immutable sealed
    package records that inconsistency; it does not change the result because
    all four required metric/case gates failed. The performance evidence was
    not robust. Pipe solver-total raw medians improved by `6.81%`, but paired
    medians regressed by `9.66%`, only `2/5` pairs won, and the A-leading and
    B-leading ratios were `0.9926` and `1.1354`. Pipe momentum-linear raw
    medians improved by `10.52%`, but paired medians regressed by `12.04%`,
    only `1/5` pairs won, and both cohort ratios regressed to `1.0299` and
    `1.2059`. Channel solver-total raw medians regressed by `11.07%`; paired
    medians improved by `1.65%`, but only `3/5` pairs won and the cohorts split
    to `1.1766` and `0.7138`. Channel momentum-linear raw medians regressed by
    `9.00%`; paired medians improved by `7.37%`, but only `3/5` pairs won and
    the cohorts split to `1.0948` and `0.6903`. The strict gate therefore
    rejected the leaf before Stage B, with zero additional measurements. The
    source remains local and unmerged, and no speedup is claimed. A focused
    Fable review was neither used nor claimed for the rejected source. The
    1,606-entry evidence manifest SHA256 is
    `c0c7d75985f98df4a14a173278124de0cf72944bcd22aece165047a4dd87c2ad`;
    its manifest-seal SHA256 is
    `5e87e9020d37cd2f1b7455235ba3b94de9f3c800a4d6e15dc4c92b3e28de03ea`.
    A following two-file clean-base research leaf on `40c6a80` replaced only
    the GAMG cached-diagonal smoother hot loop with a private validated
    index-elided kernel. Full CSR, diagonal, buffer, and coefficient-refresh
    invariants precede every call. Row, entry, multiplication, addition, and
    diagonal-division order; non-finite failure order and partial mutation;
    sweeps, tolerances, hierarchy, work counters, reports, and public APIs all
    remained exact. The general checked kernel became a unit-test-only oracle.
    No reciprocal, packed storage, reordering, reassociation, new allocation,
    hierarchy change, or artificial magnitude cap was introduced. Two direct
    debug and release proofs covered diagonal positions, multiple forward and
    reverse half-sweeps, and exact late-failure mutation. All 54 GAMG tests in
    debug and release, the 490-test workspace gate plus one intentional
    performance ignore, the release pressure-matrix gate, Rust 1.94
    locked/offline Clippy, formatting, and independent unsafe-invariant reviews
    passed.

    Native Linux CPU2 build and 16/16 semantic pre-gates passed. Stage A then
    completed exactly 48 jobs with exact fields, reports, hierarchy, options,
    stop reasons, convergence, and work fingerprints. Its single authoritative
    gate specification has SHA256
    `271f84d4327ef29bd40d7576d5fc5ca7238df39dc024f22b848a6d11eaff2cfd`.
    Channel passed the complete gate: its primary raw and paired candidate/base
    ratios were `0.82236` and `0.78960`, with both order cohorts below one and
    `4/5` wins; its profiled smoothing paired ratio was `0.69591` with `5/5`
    wins. Pipe also showed a strong mechanism signal: the smoothing raw and
    paired ratios were `0.79037` and `0.82912`, with `5/5` wins. Pipe's primary
    pressure ratios were `0.82415` raw and `0.92583` paired with `4/5` wins, but
    the B-leading cohort was `1.00155`, and its `0.07417` paired improvement was
    not greater than the `0.10900` paired-ratio MAD. Both predeclared criteria
    therefore failed. All solver/process guards passed, but the strict decision
    was `NO-GO`; Stage B was neither authorized nor created, and no extra runs
    occurred. The source remains local and unmerged, and no speedup is claimed.
    A focused Fable review was neither used nor claimed for the rejected source.
    The independently verified 1,914-entry evidence manifest SHA256 is
    `8aede4d77678e66191ff4507d620ab4a89bc6302f849ee2d3ab6c955401adb98`;
    its manifest-seal SHA256 is
    `ce5daefe6004b2a10a102545a278601accf0e458207a2f92f58f6c395f0fcdec`.
    A subsequent two-file clean-base research leaf on `951aa7f` retained the
    exact public PCG path while allowing GAMG's iterative coarsest solve to
    reuse one numerical IC(0) factorization per matrix-coefficient lifecycle.
    Every finest and coarse coefficient refresh invalidates the cache before
    mutation; failed refactorization stays invalid, and a repaired retry is
    bit-exact with a fresh workspace. Sixteen route combinations covered
    standalone and FCG, L2 and normalized-L1, profiled and unprofiled, and
    cached and fresh execution across ten non-proportional SPD lifecycles.
    Additional proofs covered wrong sparsity, zero V-cycles, an indefinite
    coarse failure, exact reports and factors, and stable PCG/IC(0)
    allocations. All 54 release GAMG tests, the full workspace gate plus one
    intentional performance ignore, the release pressure-matrix gate, Rust
    1.94 locked/offline Clippy, formatting, and independent source and proof
    reviews passed. No tolerance, hierarchy, sweep, arithmetic, traversal,
    reciprocal, reordering, public API, or artificial magnitude cap changed.

    The native Linux WSL2/ext4 CPU2 pre-gate passed all 16 jobs with exact
    reports, fields, work, hierarchy, options, and lifecycle counters. Stage A
    then completed all 132 prescribed jobs: three warmup pairs and 30 measured
    ABBA pairs per case. Pipe's primary internal-time raw candidate/base ratio
    was `1.00299`; the paired ratio was `0.96900`, but the A-leading and
    B-leading cohorts split to `1.08886` and `0.83418`, only `17/30` pairs won,
    and the one-sided 97.5% bootstrap upper log-ratio was `0.09023`. Its solver
    guard also failed the A-leading cohort. Channel's raw and paired primary
    ratios were `1.00295` and `1.00151`, the cohorts were `0.99732` and
    `1.00570`, only `15/30` pairs won, and the bootstrap upper log-ratio was
    `0.04888`; its solver guard failed as well. Both process-time guards passed,
    all 148 executed processes retained exact semantics, and no wall-clock
    anomaly occurred. Because both case gates failed, the decision is
    `NO-GO`; confirmation was neither authorized nor created. The source
    remains local, uncommitted, unpushed, and without a pull request. No Fable
    review or speedup is claimed. The sealed 3,856-entry final evidence manifest
    SHA256 is
    `6d0422f6f4bb49b0a8b0f0ebe61ade9718637ca215b6fa433895d1c80c8fe9a7`;
    its final manifest-seal SHA256 is
    `8a5c2a8adde7554fa295841ea9bba72e0c33a95ec67f69a9d844f49017eb37f0`.
14. **F-PERF-GAMG-FCG (accepted opt-in scalar path; default unchanged):** The
    existing GAMG V-cycle is now available as the explicit preconditioner for
    an `outerSolver FCG` pressure path. Seven focused proofs cover the real
    plain/profiled two-step recurrence, normalized-L1 strict boundaries,
    exact-zero minimum iterations and counters, both independent breakdown
    products, integrated non-finite failure, non-mutation, and ten coefficient
    lifecycles with stable scratch allocations. Standalone GAMG remains the
    default and allocates no per-cell FCG scratch. A same-binary Native Linux
    CPU2 `1+5` fixed-work comparison on base `504f4a49` reduced Pipe pressure
    time by `49.39%` and process time by `38.13%`, with pressure V-cycles
    `1687 -> 745`. Channel pressure time fell `23.14%` and process time fell
    `20.41%`, with V-cycles `10654 -> 7021`. All measured momentum and pressure
    solves converged; the worst relative final-field summary differences were
    `3.30e-9` for Pipe and `7.52e-9` for Channel, and final continuity L2 stayed
    below `3e-17`. Accept FCG as an explicit opt-in path only. These two
    fixed-work cases do not justify a general speed claim or a default change;
    time-to-accuracy evidence and deferred focused Fable review still follow.
    A later one-file clean-base leaf on `e0f0c90` fused the six logical FCG
    scalar reductions into one row-ordered traversal while retaining five
    independent accumulators and the logical `outerReductions += 6` telemetry.
    Product order, row order, square roots, comparisons, breakdown precedence,
    workspace state, reports, and final fields remained exact. Eight focused
    debug and release FCG proofs, all 53 GAMG tests, the full 489-test workspace
    gate plus one intentional performance ignore, Rust 1.94 locked/offline
    Clippy, and two independent source reviews passed. Native Linux CPU2 Stage A
    was promising, reducing the raw unprofiled pressure median by `13.19%` for
    Pipe and `14.31%` for Channel. The predeclared unprofiled `2+10` decision
    gate did not reproduce a robust two-case win. Pipe regressed by `9.41%` raw
    and `19.48%` paired, won only `4/10` pairs, and split by order cohort into
    `-14.43%` and `+44.58%`. Channel improved by `6.10%` raw and `8.09%` paired,
    won `7/10` pairs, and improved both cohorts, but its paired improvement of
    `0.08090` was below the required `2 * MAD = 0.23528` robustness margin. All
    canonical reports, final `U`/`p`, counters, and weighted-work values remained
    exact. The leaf is rejected, local, and unmerged; no speedup or publication
    claim is made. The `2+10` evidence manifest SHA256 is
    `69531f6a7952e52646f1a13b193bbdad55be169bb8d74d526ab094d82c0056b6`.
    Focused Fable review remains deferred with the other accepted-source audits.
15. **F-PERF-ADAPTIVE-POLICY (telemetry first):** After the FCG experiment,
    record per-solve normalized-L1 reduction, iterations, V-cycles, weighted
    work, and convergence-rate history before changing any tolerance or sweep
    schedule. Any later adaptive controller must be explicit, user-bounded,
    deterministic, and opt-in. It must not add finite-magnitude caps, hidden
    rollbacks, or case-name heuristics, and must preserve the final outer
    acceptance contract from the static `relTol` implementation.
16. **F-PERF-SIMD-RUST (first AVX2 leaf rejected; portable contract retained):**
    Evaluate portable scalar kernel layout and compiler vectorization as a
    Rust/Cargo-only product path.
    No `llvm-tools-preview`, external profiler, or system package may become a
    user build or runtime prerequisite; maintainer-only diagnostics remain
    outside the product contract. Apply numerical-parity, fixed-work, and
    time-to-accuracy gates independently from threading.
    A subsequent one-file clean-base AVX2 row-4 leaf on `5d47281` vectorized
    only the final correction-scaling row update, with runtime dispatch on
    `x86_64` and the exact scalar path as the portable fallback. It preserved
    CSR row and diagonal traversal, multiplication/subtraction/division/addition
    order, error and partial-mutation semantics, allocation behavior, hierarchy,
    sweeps, tolerances, counters, reports, and public APIs. The focused
    bit-parity, failure, malformed-structure, and ten-coefficient-lifecycle
    proofs, release validation, assembly checks, and Rust 1.94 locked/offline
    Clippy gates passed without adding a dependency or user prerequisite.

    Native Linux WSL2/ext4 pre-gates and Stage A executed exactly 280 processes:
    16 semantic pre-gates plus 264 paired `A-B-B-A` jobs across profiled and
    unprofiled Pipe/Channel lanes. All retained exact numerical semantics, but
    the frozen robustness gate rejected the leaf. Pipe's primary paired ratio
    was `0.930576`, but only `20/30` pairs won and the 98.75% bootstrap upper
    ratio was `1.028686`; its scaling-mechanism ratio was `0.947121`, with
    `19/30` wins and upper ratio `1.039300`. Channel's primary paired ratio was
    `0.962118`, with `20/30` wins and upper ratio `1.007354`; only Channel's
    scaling mechanism passed (`0.946624`, `22/30` wins, upper ratio `0.990073`).
    Profiled Channel solver and process guards also failed. Confirmation was
    not authorized, the source remains local, uncommitted, unpushed, and
    without a pull request, and no speedup, SIMD-default, or Fable-review claim
    is made. The independently verified 6,950-entry evidence manifest SHA256 is
    `ca9af93032cdaa06e110672b71b45783ca95c0a61fbc514578245aca04ed0c6b`;
    its final manifest-seal SHA256 is
    `78b0f4dbd2938fe524e27739682b1f30813ef5d35828d76215ffd3c943e09fbb`.
17. **F-PERF-THREAD-MOMENTUM (planned separate leaf):** Use the independent
    momentum components as the first bounded shared-memory threading candidate
    before parallelizing coupled pressure reductions. Preserve deterministic
    assembly, convergence reporting, and serial fallback. Keep the product
    build Rust/Cargo-only with no external system dependency, and require
    both-case scaling evidence before changing defaults.
18. **F-PERF-GPU (planned later leaf):** Start GPU work only after the scalar,
    SIMD, and shared-memory contracts above are measured. Keep it isolated from
    CPU acceptance and apply the same fixed-work plus time-to-accuracy evidence.
19. **F-D1-CYLINDER-LIMITED-SCHEMES / F-D1-CASE-CYLINDER (C1-C4 accepted):**
   Limited schemes, the
   independently authored Cylinder case, preflight, and a two-iteration smoke
   are present. The first `ferrumFiniteVolume` leaf provides generic
   stationary-no-slip pressure, viscous-force, and Cd/Cl integration for
   `zeroGradient` wall pressure with explicit pressure-gauge and extruded-2D
   reference-area handling. Moment/Cm remains gated on area-centroid geometry.

   C2 is a pure-Rust deterministic O-grid generator with three presets:
   `LegacySmoke` (`16 x 3`, 48 cells), `Coarse` (`128 x 42`, 5,376 cells), and
   `Fine` (`256 x 84`, 21,504 cells). Both production presets use a
   `-100D .. +100D` domain, depth `D`, and the continuous exponential grading
   parameter `R = 1000` in `g(t) = (R^t - 1) / (R - 1)`. This parameter is not
   a claim that the last discrete cell is exactly 1000 times wider than the
   first. Large generated meshes stay under ignored `target/` output; they are
   not repository fixtures. A neutral Gmsh 2.2 ASCII writer plus Ferrum
   importer readback must prove exact topology and patch parity without an
   OpenFOAM runtime, utility, or case dependency.

   C2 acceptance requires deterministic ordering and repeated hashes, exact
   counts and periodic seam closure, finite positive geometry with no
   problematic indices, and exact neutral round-trip parity for every preset.
   The production `Coarse` and `Fine` presets additionally require maximum
   internal non-orthogonality `<= 50 deg`, maximum normalized internal skewness
   `<= 0.55`, and maximum active 2-D edge aspect ratio `<= 4.0`. These numerical
   gates intentionally do not apply to the tiny 48-cell `LegacySmoke`
   compatibility regression; changing that topology would defeat its purpose.
   The geometry API reports raw values, and the tests do not clip or modify the
   mesh. The applicable deterministic generation, neutral readback, topology,
   and quality proofs pass for all three presets. The accepted raw maxima are
   `48.099448 deg / 0.490957 / 13.940787` for `LegacySmoke`,
   `43.608220 deg / 0.499852 / 3.626150` for `Coarse`, and
   `44.296535 deg / 0.499926 / 3.477364` for `Fine`, in
   non-orthogonality / normalized skewness / active-edge-aspect order.

   C3 connects the final SIMPLE fields to the existing generic wall-force API
   through optional `ferrumRun` reporting. It retains the raw continuity
   summary and adds the Foundation-compatible quantities
   `(sumAbs(netCellFlux)/totalVolume)*deltaT` and
   `(globalSum(netCellFlux)/totalVolume)*deltaT`, including cumulative global
   error. Its explicit release-only Rust gate regenerates and round-trips the
   `Coarse` and `Fine` meshes, re-applies the C2 quality policy to those exact
   solve inputs, repeats `Coarse`, and requires outer/linear convergence,
   finite forces, `|Cl| <= 1e-6`, normalized local/global continuity
   `<= 1e-6`, `Cd` within 15% of the documented value, and Coarse/Fine `Cd`
   drift `<= 5%`. No external runtime or case is part of C3.

   The accepted force path is intentionally a lowest-order baseline: wall
   pressure is taken from the owner cell and the viscous term uses a one-sided
   wall-normal velocity derivative. It is not yet the reconstructed wall-face
   pressure and full deviatoric traction required by the spatial-accuracy
   track.

   The release gate passed on 2026-07-29. Both identical `Coarse` runs stopped
   after 1,181 SIMPLE iterations with bit-identical final `U` and `p`; the
   first recorded normalized local/global/cumulative errors were
   `7.518411e-13 / -2.859259e-14 / -4.148956e-11`, with
   `Cd=11.50464804` and `Cl=1.074589e-8`. `Fine` stopped after 3,583 SIMPLE
   iterations with `Cd=11.53648` and `Cl=9.706568e-9`; the Coarse/Fine drag
   drift was `0.275907%`. Every force, convergence, mesh-quality, continuity,
   refinement, report, and determinism assertion passed. This two-level drift
   is a useful regression gate, not a formal observed-order, Richardson, or GCI
   result.

   This gate exposed a corrected-non-orthogonal flux inconsistency: final
   `phi` now subtracts the solved pressure flux from the exact face-flux base
   used to assemble the final pressure equation. A skewed-mesh regression
   proves matrix-residual/flux-divergence parity, while the zero-corrector path
   retains its existing behavior.

   C4 passed on 2026-07-30 using exact commit `3d84b33f2406`, one common
   generated official 5,388-cell `polyMesh`, OpenFOAM Foundation 13 build
   `13-441953dfbb42`, serial CPU2 WSL2/ext4 execution, and one warmup plus six
   measured alternating pairs. The exact 1,000-step fixed-work paired ratio
   was `2.752571` (MAD `0.495953`); the common `U,p=1e-5` TTA paired ratio was
   `3.335669` (MAD `0.608799`) at 986 Ferrum versus 995 OpenFOAM steps. Drag,
   lift, continuity, and cross-engine field gates passed, including relative
   L2 `0.525295%` for `U` and `0.926074%` for `p`. Residual stopping remains
   optional: TTA supplies `residualControl`, while Fixed-1,000 omits it and
   executes the full budget. The external evidence, post-run TTA field-gate
   provenance, and fixed-only packaging-recovery caveat are recorded in
   [Cylinder same-Linux parity](benchmarks/cylinder-linux-parity.md). No
   OpenFOAM case, source, executable, or comparison launcher is tracked.
   Validation order remains Pipe, Channel, then Cylinder.

   **Post-C4 Cylinder pressure and accuracy sequence (planned):**

   1. **F-CYL-PCG-PROFILE:** produce a fresh opt-in PCG phase profile on the
      exact accepted post-merge `main`, proving plain/profiled field, report,
      iteration, counter, and failure parity before choosing a kernel;
   2. **F-CYL-DIC-FDIC:** add true symmetric face-LDU `DIC` and `FDIC`
      preconditioners while preserving the existing full CSR IC(0) path behind
      `ic0`/`incompleteCholesky`. Match the OpenFOAM Foundation 13 mathematical
      contract: one owner/neighbour coefficient per internal face, the
      diagonal DIC recurrence, reciprocal preconditioned diagonal, and
      deterministic forward/reverse face sweeps. `FDIC` uses the same
      recurrence with cached face multipliers. Implement this independently in
      safe Rust with explicit finite, positive-pivot, allocation, ordering, and
      failure gates. This is not a reopening of the rejected LDU-addressed
      symmetric Gauss-Seidel experiment;
   3. **F-CYL-PCG-NL1:** make non-GAMG PCG evaluate the public normalized-L1
      residual directly on each convergence check and reuse the already
      written residual. Preserve strict absolute/relative boundaries,
      zero-iteration initial convergence, exact iteration-count/max-iteration
      lifecycle, L2 telemetry, breakdown behavior, final acceptance, and
      existing outer SIMPLE bounds. This is not the rejected C8
      reduction-only fusion;
   4. **F-CYL-PRESSURE-GEOMETRY-CACHE:** build immutable mesh-bound pressure
      geometry once, including face areas, projected owner/neighbour distances,
      and non-orthogonal area-vector terms. Refresh only when the mesh changes,
      keep coefficient fields solve-local, and prove exact assembly parity;
   5. **F-CYL-COMPACT-FACES / F-PERF-THREAD-MOMENTUM:** introduce compact
      owner/neighbour internal-face arrays as an isolated storage leaf, then
      benchmark deterministic parallel momentum components as the existing
      separate threading leaf. Keep a serial fallback and do not combine the
      storage and threading acceptance decisions;
   6. **F-CYL-SPATIAL-ACCURACY:** execute the separate Spatial Accuracy Track
      above: weighted least-squares gradients, the skewness/corrector matrix,
      reconstructed wall traction, and at least three-mesh observed-order plus
      GCI evidence before making a higher-accuracy claim.

   Performance leaves 1-4 and the compact-face storage leaf use the same
   generated official 5,388-cell Cylinder mesh and recorded input hashes,
   serial Linux CPU lane, separate Fixed-1,000 and time-to-accuracy results,
   and unchanged schemes, tolerances, iteration budgets, and physical gates.
   Effects below 5% require at least two warmups and nine alternating measured
   pairs. The threading leaf instead compares pinned single-thread and
   multi-thread lanes while retaining an exact serial oracle and reporting
   scaling efficiency. The accuracy leaf follows its separate multi-mesh
   protocol above and may change only the explicitly selected discretization or
   reconstruction scheme.

   Record `U`, `p`, `Cd`, `Cl`, continuity, iterations, preconditioner
   applications, and relevant work counters in every applicable lane. No
   case-specific heuristic, hidden fallback, tolerance relaxation, or
   artificial finite-magnitude cap may manufacture a pass.
20. **F-AUTO-1 (accepted external dependency):** Keep the accepted isolated n8n
   coding workflow in the AI Dev Orchestrator repository and preserve the
   existing analysis workflow as a separate read-only path.
21. **F-REF-1:** Keep focused, documentation-only external version, result, and
   protocol provenance for each newly selected physics area. Do not bundle
   external solver cases or sources.
22. **F-ARCH-1A / F-ARCH-1B / F-ARCH-1C (mandatory before Driver 2, thermal
   physics, or GPU kernels):** Replace the transitional combined ownership in
   three independently reviewable, parity-preserving leaves:
   - **F-ARCH-1A:** move finite-volume geometry, gradients, fluxes, boundary
     operators, and equation assembly from `ferrumMesh` into
     `ferrumFiniteVolume`; keep mesh and topology ownership in `ferrumMesh`.
     The independent boundary-force/post-processing leaf is the first active
     crate boundary; the solver operators still require separate parity-only
     migration packages;
   - **F-ARCH-1B:** move backend-neutral CSR, PCG/BiCGStab, smoothers, and GAMG
     kernels behind a narrow reusable linear-algebra API owned under
     `ferrumCore`, without changing arithmetic order, reports, or solver
     semantics;
   - **F-ARCH-1C:** move SIMPLE/SIMPLEC orchestration, the module registry, and
     the common solver lifecycle into `applications/modules/incompressibleFluid`
     and leave `ferrumRun` as dispatch only.
   Each leaf must keep the existing unit, workspace, fixed-work, convergence,
   and field-parity gates green. Do not combine this migration with new
   numerics or performance claims.
23. **F-IO-1:** Specify and implement `FerrumFile v1`; isolate independently
   authored external-format compatibility behind the `ferrumIO` adapter
   boundary.
24. **F-D1D2-1:** Complete Driver 1 SIMPLE/SIMPLEC and Driver 2 PISO/PIMPLE on the scalar CPU
   reference backend for the frozen selected-case inventory.
25. **F-BACKEND-1:** After the Driver 1/2 inventory gate, accept `ferrumRun`
   successively on partitioned multi-process CPU, multi-node CPU, multi-GPU,
   and multi-node CPU/GPU integration without changing case numerics, reusing
   the earlier accepted shared-memory and single-GPU leaves.
26. **F-D3D7-1:** Implement Drivers 3 through 7 in the fixed order above, applying the common
   readiness gate and completing coupled `ferrumMultiRun` before Driver 6.
27. **F-POROUS-1:** Begin porous-media and packed-bed work only after Driver 7 is complete.
