# FerrumCFD Architecture Notes

FerrumCFD should feel familiar to OpenFOAM users at the workflow level, while
using a new Rust implementation and a backend-aware solver architecture.

## Repository And Tutorial Layout

FerrumCFD follows the OpenFOAM 13 separation of compiled applications,
reusable implementation modules, utilities, and tutorials while retaining
Cargo as the Rust build system.

```text
applications/
  solvers/
    ferrumRun/
    ferrumMultiRun/
  modules/
    incompressibleFluid/
  utilities/
    mesh/
    case/
    postProcessing/
src/
  ferrumCore/
  ferrumMesh/
  ferrumFiniteVolume/
  ferrumIO/
  openfoamIO/                # interoperability only
  ferrumModels/
tutorials/
  incompressibleFluid/
    <case>/
      ferrum/case/
      openfoam-v13/case/
      analytical/            # optional when a useful solution exists
      shared/                # optional neutral inputs
      comparison.toml        # optional reference mapping
      README.md
validation/
  scripts/
test/
docs/
target/                      # generated and ignored
```

Every tutorial presents independent program-specific references, not one
shared runtime case. The current `ferrum/case` compatibility directory and the
native `openfoam-v13/case` directory must each run independently. `shared/` may
contain neutral geometry, units, and physical inputs, but never
program-specific dictionaries. Each program converts the shared geometry into
its own native mesh and owns its numerical configuration.

Small canonical validation meshes may be versioned in each program-specific
source case so a clean checkout is independently runnable. Regenerated mesh
variants, time directories, logs, and reports belong below `target/`. An
`analytical/` directory is included only when a useful closed-form,
semi-analytical, or manufactured solution exists. Otherwise the case README
identifies an appropriate published benchmark or observable. Empty or invented
analytical references are forbidden. Stable recorded results belong under
`docs/benchmarks/`.

The first layout migration places the reusable mesh/finite-volume foundation
under `src/ferrumMesh` and the still-combined implementation package under
`applications/legacy/ferrumCli`. The canonical `ferrumRun` executable crate
already lives at `applications/solvers/ferrumRun` and delegates to that legacy
library. Solver modules and utilities move behind their permanent boundaries
only with behavior-parity tests. This staged move preserves a buildable
workspace throughout the architecture transition.

## Public Runner And Module Naming

The public single-region command is:

```text
ferrumRun -solver incompressibleFluid -case <case>
```

This mirrors the OpenFOAM 13 runtime-module boundary. `incompressibleFluid` is
the equation-family module name. `laminar` is a physical-model regime and
SIMPLE, SIMPLEC, PISO, and PIMPLE are coupling algorithms selected by case
configuration. None of them should be baked into a permanent executable or
module name.

The current `ferrumRun` command dispatches the existing steady laminar SIMPLE
kernel only when `ddtSchemes.default=steadyState`, exactly one `SIMPLE` section
exists, and no competing `PISO` or `PIMPLE` section is present. It rejects
transient execution until those algorithms exist. Existing Ferrum compatibility
cases without a transport-regime dictionary are treated as laminar; when
`momentumTransport` or legacy `turbulenceProperties` is present, it must set
exactly `simulationType laminar`. RAS/LES is rejected instead of silently
running the wrong kernel. There is no public algorithm-specific executable or
`--solveLaminarSimple` selector; `ferrumRun` calls the selected module lifecycle
directly.

Drivers 1 and 2 are separate validation/readiness gates, but both are served
by the same `incompressibleFluid` module: Driver 1 validates steady
SIMPLE/SIMPLEC and Driver 2 validates transient PISO/PIMPLE.

## Single-Region And Multi-Region Runners

`ferrumRun` owns one case, one region, and one selected solver module.
`ferrumMultiRun` owns one coupled case with multiple named regions and one
module per region. The latter follows OpenFOAM 13 `foamMultiRun`; it is not a
batch launcher for unrelated cases or parameter sweeps.

Both runners use the same backend-neutral execution context and module kernels.
After the complete selected SIMPLE/SIMPLEC/PISO/PIMPLE case inventory passes on
the scalar CPU reference backend, `ferrumRun` is accepted successively on
multi-threaded CPU, partitioned/distributed CPU, one GPU, and multiple GPUs.
`ferrumMultiRun` then schedules those accepted kernels over its coupled region
dependency graph instead of introducing a second compute stack.

A future native control dictionary will express a mapping such as the
following illustrative form. The actual thermal-fluid and solid-energy module
names remain intentionally undecided until Drivers 3 and 6:

```text
regionSolvers
{
    fluidRegion    <thermal-fluid-module>;
    wallRegion     <solid-energy-module>;
}
```

`ferrumMultiRun` has no `-solver` option; every region-to-module selection
comes from the case's region registry. It advances regions through a
capability- and dependency-based phase graph covering mesh motion, model
correction, momentum, energy, species, chemistry, pressure correction,
VOF/interface advection, and post-solve work. Only independent region tasks may
run concurrently. Transfers and synchronization barriers are placed at every
actual data dependency, including coupled interface values needed before an
energy, species, reaction, pressure, or phase-fraction step.

The global time-step is the minimum of global/function constraints and the
limits reported by active transient regions; steady regions do not contribute
a transient stability limit. Mixed steady/transient region operation must be
represented explicitly. Convergence is reached only when all participating
region criteria pass.

Both runners must use the same backend-neutral lifecycle and execution
context. The resource contract distinguishes sockets, cores, worker threads,
process ranks, and domain partitions; maps regions and partitions to ranks and
GPU devices; defines halo/ghost and conservative interface exchange; and
tracks a data-residency/transfer graph. It also includes GPU memory and queue
selection, backend capability checks including required `f64` support,
oversubscription prevention, per-region and per-stage backend choice, mixed
CPU/GPU operation, multi-GPU placement, deterministic reductions,
cancellation/error propagation, and conservation checks with stated
tolerances. Independent cases and parameter studies will use a separate future
batch/sweep tool so multi-region semantics remain clear.

## Native Ferrum And OpenFOAM Boundary

The target Ferrum case format is native:

- every Ferrum dictionary uses the `FerrumFile` header;
- Ferrum configuration uses names such as `ferrumControl`, `ferrumSchemes`,
  `ferrumSolution`, and `ferrumModels`;
- physical field names such as `U`, `p`, `T`, and species names may remain
  standard scientific notation;
- Ferrum-specific parameters use explicit SI-oriented names;
- the user-facing command flow remains `initFerrumCase`, `gmshToFerrum`,
  `checkFerrumMesh`, `splitFerrumMeshRegions`, and `ferrumRun`.

The sibling `openfoam-v13/` case remains a genuine OpenFOAM 13 case. It uses
`FoamFile`, OpenFOAM dictionaries and parameter names, and its own mesh
conversion and run commands. OpenFOAM parsing and conversion belong in the
separate `openfoamIO` interoperability layer and must not define the native
Ferrum format.

This is the target contract. Until `FerrumFile v1` and its reader/writer are
implemented, the existing OpenFOAM-like Ferrum reader remains a documented
compatibility bridge. It currently reads `FoamFile`, `dimensions`,
`internalField`, and `boundaryField`. Existing executable cases must remain
usable until native-format parity tests pass.

The bridge accepts the exact unquoted nonuniform type names `List<scalar>`,
`scalarField`, `Field<scalar>`, `List<vector>`, `vectorField`, and
`Field<vector>`. It currently rejects OpenFOAM dictionary directives such as
`#include` and `#includeFunc` with a path- and line-aware error. Resolving those
directives safely requires a separately bounded include stack; silently
skipping them could change physics inputs and is not permitted.

Solver-state `materializable` means that a self-contained state plan can build
the CPU buffer directly, which currently applies to valid uniform fields.
Valid nonuniform fields are transfer-ready: the full runtime takes ownership
of the already validated source allocation exactly once, while Summary mode
retains descriptors without values.

`checkFerrumMesh` validates field boundary entries against mesh patches. This
is deliberately solver-neutral: it checks names and special patch
compatibility such as `empty` fields on `empty` mesh patches, but it does not
decide whether a pressure or velocity boundary condition is physically
appropriate for a driver.

## Reduced Dimensions And Axisymmetry

The mesh importer can now write OpenFOAM-compatible patch types such as
`empty` and `wedge`. This is only the mesh/import side.

Solver rule:

- `empty` must be interpreted by every relevant solver as a true reduced
  dimension patch for 1D/2D cases.
- `wedge` must be interpreted by every relevant solver as an axisymmetric
  wedge patch.
- A solver must not silently treat `empty` or `wedge` as a normal wall or
  generic patch.

Validation rule:

- `checkFerrumMesh` now counts `empty`, `wedge`, and `symmetryPlane` patches,
  checks boundary patch face ranges, and warns about odd wedge patch counts.
- `checkFerrumMesh` should eventually reject deeper invalid `empty` setups,
  such as non-empty patches in the suppressed direction or more than one cell
  through the reduced dimension.
- `checkFerrumMesh` should eventually reject deeper invalid `wedge` setups,
  such as wrong patch pairing, inconsistent angles, or geometry that cannot be
  treated as axisymmetric.
- Field files in `0/` must later use boundary conditions compatible with the
  mesh patch type.

This keeps the OpenFOAM habit: the mesh stays formally 3D, while special patch
types define reduced-dimensional or axisymmetric behavior.

## Backend Selection

FerrumCFD must not assume that all work should run on the GPU. GPU acceleration
should be selectable per solver and per major compute stage, because small
cases, setup work, mesh operations, or stiff chemistry may sometimes be more
efficient or easier to debug on the CPU.

Planned backend policy:

- CPU is always available.
- GPU is optional and selected explicitly or by an `auto` policy.
- Backends are chosen per physics module and per solver component where useful.
- The code should allow mixed execution, for example flow on GPU, chemistry on
  CPU, or linear algebra on GPU while setup and checks remain on CPU.
- Host/device transfers must be visible in the design, not hidden inside random
  helper calls.

An initial dictionary could look like this:

```text
ferrumBackends
{
    default cpu;

    mesh
    {
        import cpu;
        checks cpu;
    }

    flow
    {
        nonlinearSolve gpu;
        residual gpu;
        jacobian gpu;
        linearSolve gpu;
        pressureCorrection gpu;
    }

    interfaces
    {
        flux auto;
        coupling auto;
        sourceTerms auto;
    }

    chemistry
    {
        nonlinearSolve gpu;
        residual gpu;
        jacobian gpu;
        odeSolve gpu;
    }

    cpu
    {
        cpus auto;          // physical CPU packages/sockets, or a positive integer
        coresPerCpu auto;   // physical cores per CPU package, or a positive integer
        threads auto;
        threadPinning off;
        numa auto;
    }

    gpu
    {
        backend auto;     // auto, wgpu, cuda, hip
        devices (auto);   // auto, one device id, or multiple ids
        multiGpu auto;    // auto, on, off
        precision f64;
    }
}
```

This dictionary is parsed and validated as case metadata, but not yet consumed
by executable solvers.

Nonlinear solver interfaces and data ownership must stay backend-neutral from
the beginning. A Newton-style solver should not be CPU-bound by design:
residual evaluation, Jacobian assembly, linear correction solves, convergence
checks, and batched chemistry ODE solves must all be able to target CPU, GPU,
or an auto policy. Actual parallel CPU/GPU kernels are implemented only after
the selected SIMPLE/SIMPLEC/PISO/PIMPLE correctness matrix passes on the scalar
CPU reference backend. This preserves a deliberate validation order without
requiring a later architectural rewrite.

CPU remains a deliberate execution target, not a fallback of last resort. Users
must be able to keep a solve on CPU when the GPU is needed elsewhere, when a
small case would not amortize device transfers, or when a specific model has
better CPU behavior.

Multi-CPU systems must be represented explicitly enough for a future scheduler
to make reproducible decisions. `cpus` describes physical CPU packages/sockets,
`coresPerCpu` describes physical cores per package, and `threads` describes the
worker-thread budget FerrumCFD may use. For mixed CPU/GPU policies, the case
should provide both CPU and GPU resource blocks so the solver can report where
each major stage is intended to run.

Those names are the current compatibility schema. Before distributed
`ferrumMultiRun`, the native resource schema will use unambiguous `sockets`,
`cores`, `workers`, `ranks`, and `partitions` fields while retaining a
documented migration path from `cpus`/`coresPerCpu`.

Backend policy validation should catch obvious configuration mistakes without
blocking future physics modules. Known built-in sections such as `mesh`,
`interfaces`, `flow`, `chemistry`, `heat`, and `species` can warn about
misspelled stages or duplicate entries. Unknown sections remain allowed as
forward-compatible custom policy, but the preflight should report that current
built-in solvers do not consume them yet.

## Units Contract

FerrumCFD's user-facing model and solver data should be SI-first. Bare numeric
values represent SI units for their dimension: metres, kilograms, seconds,
kelvin, pascal, and derived SI units. Non-SI values should require explicit
unit syntax once unit suffix parsing exists. This keeps solver kernels and
benchmark comparisons dimensionally predictable and avoids hidden display-unit
conventions.

Compatibility layers may adapt external tools. OpenFOAM incompressible
benchmarks use kinematic pressure internally. The OpenFOAM 13
`incompressibleFluid` comparison therefore converts pressure back to SI before
writing FerrumCFD benchmark JSON.

## Solver Preflight Boundary

`ferrumRun --preflight` currently builds a solver-neutral case plan without
executing CFD kernels. The plan is the boundary between the
OpenFOAM-like case layout and the future backend-specific solver runtime.
The normal output is human-readable text; `--planJson <file>` writes the same
plan as machine-readable JSON for future solver launchers, GUIs, benchmarks,
and regression tests.

The preflight reads:

- `system/controlDict` for run timing and the selected application name
- `system/fvSchemes` for user-facing discretisation choices
- `system/fvSolution` for user-facing solver and algorithm settings
- `constant/polyMesh` for topology, patches, and special reduced-dimension
  patch types
- constant property dictionaries such as `transportProperties` and
  `thermophysicalProperties`
- region-local property dictionaries below `constant/<region>/`
- generated region meshes below `constant/<region>/polyMesh`
- initial fields below `0/`
- optional `constant/interfaces` for model-facing interface sign conventions;
  absence means an empty configuration, while a present file must contain
  exactly one unquoted, ordinary `interfaces { ... }` block
- `system/ferrumBackends` for CPU/GPU resource and stage policy

The plan classifies the case as `3d`, `2d-empty`, `axisymmetric-wedge`, or
`mixed-special-patches`. Later solver modules should consume this explicit
classification rather than rediscovering reduced-dimensional behavior from
raw patch strings in scattered equation code.

The plan also derives a run schedule from `controlDict` when the time controls
are fixed enough to do so. `startTime`, `endTime`, and positive `deltaT` allow
an estimated step count. `writeControl timeStep` with an integer
`writeInterval` allows an estimated write-event count. Other OpenFOAM-style
stop/write modes remain valid, but the current preflight keeps their schedule
open until a runtime exists.

Backend policy resolution belongs in the run plan. Built-in stages are
expanded into concrete `section.step=choice` entries, with a source marker
showing whether the choice came from an explicit `ferrumBackends` stage or the
default backend. This includes nonlinear solver stages, chemistry ODE solves,
and interface stages such as `interfaces.flux`,
`interfaces.coupling`, and `interfaces.sourceTerms`.

Solver state is the boundary between OpenFOAM-like field files and future
equation kernels. The preflight should convert initial fields below `0/` into
typed storage plans for `volScalarField`, `volVectorField`, and eventually
surface fields used by fluxes. Volume fields must match mesh cell counts when
nonuniform data is supplied; surface fields must match face counts. The state
layer should estimate component counts, f64 slot counts, and byte footprints so
CPU/GPU buffers can later be allocated reproducibly. It reports CPU/GPU storage
capability and marks correctly shaped uniform fields as CPU-buffer
materializable. Summary plans count-check the six supported exact nonuniform
scalar/vector type names but retain no values. Full solve plans validate those
payloads and transfer the source allocation once into runtime storage without
cloning it. Unsupported nonuniform types are rejected rather than represented
as summary-only fields. None of this implies that solver kernels have run.

`--runnerDryRun` is the first runner boundary. It expands the run plan into a
capped sequence of time-step starts, stage dispatch decisions, and planned
write events. It also resolves lightweight CPU/GPU runtime handles from
`ferrumBackends`, including CPU thread metadata and GPU backend/device
metadata. CPU linear algebra availability is reported separately from full CFD
kernel availability. GPU dispatch must be reported as unavailable until
executable GPU solver kernels exist. It must remain explicit that this mode
does not update fields, advance physics, assemble matrices, or solve equations.
Its job is to harden the scheduling contract before CPU/GPU solver kernels
exist.

`fvSchemes` and `fvSolution` parsing is broad and structural at preflight
level. The preflight can report entries such as `ddtSchemes.default=Euler` or
`SIMPLE.nNonOrthogonalCorrectors=0`, while executable solver code decides which
schemes and linear/nonlinear solver settings are valid for each equation
system. The current laminar SIMPLE bridge already consumes a focused
`fvSchemes` subset: `grad(p)`, `grad(U)`, `div(phi,U)` with `Gauss upwind` or
`Gauss linearUpwind grad(U)`, `Gauss linear` laplacians with
`corrected`/`orthogonal`/`uncorrected` snGrad behavior, `linear`
interpolation, and matching `snGradSchemes`.

Basic structural validation belongs in the preflight. Examples include missing
standard `fvSchemes` sections, missing `default` entries, missing
`fvSolution.solvers`, or initial fields that have no matching solver entry.
Equation-specific validation, such as whether a convection scheme is valid for
a particular transport equation, stays with the future solver modules.

`controlDict` validation is also structural. The preflight should catch
invalid run-control modes, missing or non-positive `deltaT`, invalid
`writeInterval`, and inconsistent `startTime`/`endTime` before a backend
runtime tries to enter a time loop.

Property dictionary parsing follows the same rule. The preflight can report
entries such as `transportProperties.nu=[0 2 -1 0 0 0 0] 1e-05` and warn about
malformed dimension vectors, but physics modules decide later whether a
particular model requires `nu`, `rho`, species diffusivity, thermal
conductivity, membrane permeance, or another coefficient.

## Mesh Geometry Direction

The first geometry pass derives face centres, oriented face area vectors,
approximate cell centres, cell volumes, and boundary area from
`constant/polyMesh`. These values are now summarized by `checkFerrumMesh`.

This is still a geometry foundation, not a full quality checker. Future checks
should add non-orthogonality, skewness, aspect ratio, wedge validity, `empty`
validity, and interface-normal consistency.

## Solver Architecture Direction

The solver stack should be written against backend-neutral data and execution
traits:

```text
ferrumRun / ferrumMultiRun dispatchers
Shared module registry and solver lifecycle
Application-driver readiness contracts
Equation and coupling modules
Physical models
Finite-volume operators
Fields and mesh topology
Linear/nonlinear solvers
Backend implementations: CPU, WGPU, CUDA, HIP
```

The public solver portfolio consists of seven application-driver readiness
contracts behind a shared module registry, not seven copied solver programs.
Single-region drivers run through `ferrumRun`; Driver 6 exercises coupled
modules through `ferrumMultiRun`. Drivers compose reusable equation, coupling,
and physical-model modules. A model required by several drivers has one
implementation with driver-specific configuration.

Physics code should express operations in terms of fields, operators, and
solver steps. Backend implementations should decide where and how those
operations run.

The first executable solver foundation is CPU linear algebra: CSR matrices,
matrix-vector products, residuals, Jacobi, conjugate gradient,
preconditioned-CG, BiCGStab, and a reusable GAMG hierarchy and V-cycle. This is
the minimal substrate used by the first scalar
Poisson/diffusion and laminar flow assemblies from runtime mesh geometry. It
should remain a small backend-neutral contract so later GPU implementations
can provide the same operations without changing the equation assembly layer.
The SIMPLE pressure path supports GAMG on symmetric pressure CSR systems with
explicit `algebraicPair` or mesh-geometric `faceAreaPair`. The latter derives
its initial weights from runtime face-area vectors and sums weights while
coarsening. One hierarchy is retained per pressure topology; matrix values are
refreshed for each pressure equation. Case-level `tolerance` and `relTol`
retain OpenFOAM's normalized LDU L1-residual meaning: the GAMG core receives a
conservative absolute L2 limit, and the reporting layer checks the strict L1
criteria before it marks the linear solve converged. GAMG remains invalid for the
nonsymmetric momentum equation, and unsupported controls fail without
substituting PCG or another agglomerator.

GAMG cycle profiling is an explicit diagnostic path selected with
`--profileGamg`; it is not a case-dictionary control and does not alter the
equation, convergence criteria, cycle controls, or solver selection. The normal
path performs no per-phase clock reads. The profiled path executes the same
operation order and reports hierarchy build/refresh, finest residual, V-cycle,
restriction, prolongation, smoothing, scaling, coarse residual, correction, and
coarsest-solve time, including per-level work counts. Profile parity is tested
against the unprofiled solve with bit-identical fields and residuals.

Each GAMG level caches the unique CSR diagonal slot for every row. The smoother
uses those slots to traverse the entries before and after the diagonal in the
same CSR order instead of searching for the diagonal during every sweep. GAMG
therefore requires exactly one diagonal entry per matrix row and rejects an
invalid layout explicitly; it does not substitute another smoother.

The first equation assembly layer is scalar diffusion/Poisson on CPU. It
converts runtime mesh geometry into a CSR system with internal face coupling,
`fixedValue` and `zeroGradient` boundary contributions, and volume source
terms. Constraint patches such as `empty`, `wedge`, and `symmetryPlane` remain
solver constraints rather than ordinary diffusive boundary faces. This assembly
layer must stay separate from the linear solver implementation: equation code
builds a system, while CPU/GPU backends decide how that system is solved.
The developer utility `ferrum solve --solveScalarDiffusion <field>` is the first
executable path through that stack: it reads one scalar field, assembles one CPU
system, solves it with CG or Jacobi, reports residual and wall-clock time, and
deliberately does not write fields or enter the full CFD time loop. It is not a
public application solver.

The developer utility `ferrum solve --solvePoiseuille` is the first flow
benchmark path. It uses the same scalar operator for the fully developed axial
Stokes balance driven by `deltaP/L`, applies wall no-slip as `Ux=0`, compares
the resulting volume average against Hagen-Poiseuille, and reports timing and
residuals. This is a controlled validation utility, not a public application
solver and not a SIMPLE, PISO, or full Navier-Stokes implementation.

The implementation behind `ferrumRun -solver incompressibleFluid` invokes the
internal laminar SIMPLE kernel directly. Its temporary physical location in the
combined compatibility library does not create a second public solver command.
It reads `U`, `p`, `transportProperties`, `fvSchemes`, and `fvSolution`,
constructs the first flow operators on the same runtime `polyMesh` geometry,
writes solver reports as JSON/Markdown, and can write final `U`/`p` fields into
an explicitly selected OpenFOAM-like time directory. The current implementation is an executable laminar SIMPLE
development path rather than a production `simpleFoam` replacement: it uses
OpenFOAM-style equation relaxation and pressure relaxation, continues SIMPLE
iterations without artificial field clipping, and reports continuity,
equation residuals, stored fields, and relative `U`/`p` field changes.
Momentum and pressure-correction linear solvers can be selected separately, so
the OpenFOAM-style `smoothSolver` entry on `U` dispatches to its configured CPU
`GaussSeidel` or `symGaussSeidel` smoothing path, while experiments can still run the explicit
non-symmetric `bicgstab` momentum path with a PCG pressure solve without
changing the case files. OpenFOAM-style `fvSolution`
entries are the default source for pressure and velocity under-relaxation and
for per-equation linear
tolerances: `relaxationFactors.equations.U`,
`relaxationFactors.fields.p`, `solvers.U.tolerance`, `solvers.p.tolerance`,
`solvers.p.solver PCG`, `solvers.p.preconditioner DIC`,
`SIMPLE.nNonOrthogonalCorrectors`, `SIMPLE.pRefCell`, `SIMPLE.pRefValue`, and
`SIMPLE.consistent`, and optional `maxIter` values. OpenFOAM-style
`SIMPLE.residualControl` entries for `U` and `p` are read as the normal
early-convergence criteria. As in OpenFOAM Foundation 13, each steady SIMPLE
criterion is one absolute scalar. Ferrum evaluates the initial residual from
the first equation solve in the SIMPLE iteration; `U` uses the maximum vector
component and `p` uses the first pressure solve before any further
non-orthogonal corrector solves. Linear-solver initial/final residuals,
iteration counts, and convergence flags remain a separate reporting layer.
Continuity is diagnostic and is not an extra hidden convergence gate.
Ferrum-specific SIMPLE entries can additionally set `minSimpleIterations`.
Only OpenFOAM-style residual controls can mark the generic solver converged.
Hagen-Poiseuille acceptance, OpenFOAM comparison, and matched-time decisions
belong to the external benchmark scripts and never alter SIMPLE convergence.
The SIMPLE options/report types contain no pipe diameter, pipe length, named
inlet/outlet reference, pressure-loss target, or analytic solution. External
validation tooling may read already written `U`/`p` fields and add those
case-specific diagnostics without changing solver behavior. Validation profiles
and reference inputs are stored under `validation/`; generated comparison
artifacts remain under `target/benchmarks`, never inside a simulation case's
`constant/` directory.
The production-readiness plan for this solver lives in
`docs/solver-roadmap.md`; it tracks the remaining numerical, boundary-condition,
scheme, benchmark, performance, and generalization work.
`PCG` dispatches to Ferrum's CPU preconditioned-CG path, OpenFOAM
`smoothSolver` on `U` requires and executes a supported `GaussSeidel` or
`symGaussSeidel` smoother, and
explicit `bicgstab` remains available for nonsymmetric momentum experiments.
OpenFOAM `DIC`/`FDIC` on pressure PCG maps to IC(0). `DILU` is rejected until a
true nonsymmetric ILU/DILU preconditioner exists; Ferrum never substitutes a
diagonal preconditioner silently. The selected pressure PCG path shares the
mesh-dependent CSR pattern, reuses matrix/RHS and PCG work storage, and retains
the IC(0) symbolic structure while refactoring its numerical values for each
pressure equation. The OpenFOAM-normalized scalar-solve reporting layer also
retains its zero-initial, matrix-product, and residual buffers across momentum
and pressure equations. A topology, workspace, or preconditioner mismatch is an
error, not a fallback. CLI flags remain explicit experiment overrides. Solver
execution also requires `system/fvSchemes` and
`system/fvSolution` instead of inventing a missing case configuration.
Absent `tolerance` and `maxIter` entries use the OpenFOAM 13
`lduMatrix::solver` defaults (`1e-6` and `1000`). Ferrum currently supports the
OpenFOAM defaults `relTol=0`, `minIter=0`, and `smoothSolver nSweeps=1`; a
different configured value is rejected explicitly until its stopping semantics
are implemented.
The pressure bridge now follows the OpenFOAM shape more closely: it applies
equation relaxation through an internal momentum-equation object, builds
cell-wise `rAU` from the original momentum diagonal, exposes per-component
momentum residuals plus `A/H1` ranges in reports, reconstructs `HbyA`, computes
`phiHbyA` from that HbyA field with velocity boundary constraints applied,
applies an OpenFOAM-like `adjustPhi` mass-balance correction only on
pressure-controlled open boundaries, treats velocity
`inletOutlet`/`pressureInletOutletVelocity` as flux-dependent open boundaries
for backflow, solves an absolute variable-coefficient pressure equation,
corrects `phi` with the pressure-equation flux, corrects velocity as
`U = HbyA - rAtU grad(p)`, and carries that corrected surface flux into the next
SIMPLE iteration. The normal solver path no longer
bounds or rolls back finite `U`, `p`, or `phi` updates; only non-finite fields
are treated as numerical failure. The pressure equation also supports
OpenFOAM-like pressure reference anchoring and runs
`nNonOrthogonalCorrectors + 1` pressure solves, updating `phi` from the final
pressure solve. `SIMPLE.consistent true` switches the pressure and velocity
correction to a Rust `rAtU` value derived from the current momentum matrix
(`1/rAU - H1`), and the non-orthogonal corrector loop now rebuilds the pressure
source with an explicit non-orthogonal pressure-flux correction between solves.
The momentum convection term is scheme-driven. `Gauss upwind` uses a fully
implicit upwind contribution. `Gauss linearUpwind grad(U)` keeps the same
non-symmetric upwind matrix and adds the gradient part as a deferred
right-hand-side correction, matching the OpenFOAM workflow more closely while
keeping the first executable path robust. The optional field writer preserves
dimensions and `boundaryField` entries from the initial `0/U` and `0/p` files
while replacing the internal cell values with the final SIMPLE solution.
Operator and report data remain backend-neutral so the same assembly path can later dispatch linear
and nonlinear solves to CPU, GPU, or mixed CPU/GPU resources.

Important design constraint:

```text
OpenFOAM-like user workflow outside.
Rust/GPU-first architecture inside.
```

That means FerrumCFD can remain comfortable for CFD users while still avoiding
OpenFOAM's CPU-centered internal data layout.

## Interface Orientation

Interface orientation is mesh metadata. It should not be redefined separately
inside every physics equation.

For a named interface such as `mantle_inner_membrane_complete`, FerrumCFD should
track:

- the patch or faceZone name
- the two adjacent regions
- the oriented face normal
- the source `flipMap` value where the interface comes from a faceZone
- the sign convention used by models that consume fluxes across the interface

Physics modules should then use that oriented interface normal. For example, a
membrane model can define positive species flux from `inner_zone` into
`membrane`, while the discretisation backend maps that sign convention onto the
actual face owner/neighbour orientation.

The same registry must work for non-membrane cases too. Examples include:

- pressure-jump interfaces
- porous jumps
- baffles
- conjugate heat transfer
- species transfer between regions
- generic coupled regions

The interface registry should therefore stay model-neutral. It should describe
the geometry and orientation. Physics modules decide which law to apply, such
as a pressure-difference law, temperature-difference law, concentration jump,
or membrane permeance law.

User-facing model orientation should be configured in `constant/interfaces`.
Users should not normally edit `flipMap` manually; it is source mesh metadata.
The interface dictionary expresses intent, for example `orientation
fluid_to_solid`, and FerrumCFD maps that intent onto owner/neighbour and
faceZone orientation data.

That orientation is only a sign convention. It must not clamp or force the
later physical flux direction. If a pressure jump reverses, a pressure-driven
interface model should produce a negative flux with respect to the configured
positive direction.

## Reference Points

- [OpenFOAM User Guide](https://www.openfoam.com/documentation/user-guide):
  case layout, running applications, mesh conversion, solving, and
  post-processing workflow.
- [OpenFOAM mesh boundary documentation](https://doc.cfd.direct/openfoam/user-guide-v13/boundaries):
  patch `type` entries and the `empty`/`wedge` semantics for reduced dimensions
  and axisymmetric cases.
- [OpenFOAM numerical schemes](https://www.openfoam.com/documentation/user-guide/6-solving/6.2-numerical-schemes):
  `system/fvSchemes` as the user-facing dictionary for discretisation choices.
- [OpenFOAM standard utilities](https://doc.cfd.direct/openfoam/user-guide-v13/standard-utilities):
  upstream workflow references. FerrumCFD deliberately exposes the distinct
  `gmshToFerrum`, `checkFerrumMesh`, and `splitFerrumMeshRegions` names instead
  of reusing the OpenFOAM utility names.
