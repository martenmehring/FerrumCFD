# FerrumCFD Architecture Notes

FerrumCFD should feel familiar to OpenFOAM users at the workflow level, while
using a new Rust implementation and a backend-aware solver architecture.

## OpenFOAM-Compatible Workflow Target

FerrumCFD should preserve the common OpenFOAM case workflow where practical:

```text
case/
  0/
    U
    p
    ...
  constant/
    polyMesh/
      points
      faces
      owner
      neighbour
      boundary
      faceZones
      cellZones
    transportProperties
    thermophysicalProperties
    ...
  system/
    controlDict
    fvSchemes
    fvSolution
    ferrumBackends
```

The user-facing command flow should remain familiar:

```powershell
initFerrumCase case
gmshToFerrumFoam mesh.msh -case case
checkFerrumMesh -case case
splitFerrumMeshRegions -case case -cellZones
ferrumSolver -case case --preflight
```

The goal is not to copy OpenFOAM internals. The goal is to keep the established
case layout, patch naming, command rhythm, and dictionary style where that
reduces user friction.

## Dictionary And Field Parsing

FerrumCFD now has a shared OpenFOAM-style token/cursor parser used by case
dictionaries such as:

- `constant/interfaces`
- `system/controlDict`
- `system/ferrumBackends`
- initial field files below `0/`

Initial field parsing is intentionally structural at this stage. It reads
`FoamFile`, `dimensions`, `internalField`, and `boundaryField`, then reports the
setup in `checkFerrumMesh`. Solver modules will later interpret these fields in
the context of equations, patch constraints, and dimensions.

`checkFerrumMesh` now validates field boundary entries against mesh patches.
This is deliberately solver-neutral: it checks names and special patch
compatibility such as `empty` fields on `empty` mesh patches, but it does not
yet decide whether a pressure or velocity boundary condition is physically
appropriate for a solver.

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

Nonlinear solver stages must stay backend-selectable from the beginning. A
Newton-style solver should not be CPU-bound by design: residual evaluation,
Jacobian assembly, linear correction solves, convergence checks, and batched
chemistry ODE solves must all be able to target CPU, GPU, or an auto policy.
This is one of the architectural differences FerrumCFD should preserve over a
CPU-first OpenFOAM-style implementation.

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
benchmarks, for example, can use kinematic pressure internally because that is
what `simpleFoam` expects, but comparison scripts must convert back to SI
pressure before writing FerrumCFD benchmark JSON.

## Solver Preflight Boundary

`ferrumSolver` currently builds a solver-neutral case plan instead of executing
CFD kernels. This is intentional. The plan is the boundary between the
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
- `constant/interfaces` for model-facing interface sign conventions
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
materializable. Nonuniform `List<scalar>` and `List<vector>` fields are loaded
as flattened CPU-buffer values when their counts and component shapes match.
Other nonuniform field types remain count-checked only until type-specific
loaders exist. None of this implies that solver kernels exist or have run.

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

`fvSchemes` and `fvSolution` parsing is currently structural. The preflight can
report entries such as `ddtSchemes.default=Euler` or
`SIMPLE.nNonOrthogonalCorrectors=0`, but executable solver code must later
decide which schemes and linear/nonlinear solver settings are valid for each
equation system.

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
Mesh topology
Fields
Operators
Physics modules
Linear/nonlinear solvers
Backend implementations: CPU, WGPU, CUDA, HIP
```

Physics code should express operations in terms of fields, operators, and
solver steps. Backend implementations should decide where and how those
operations run.

The first executable solver foundation is CPU linear algebra: CSR matrices,
matrix-vector products, residuals, Jacobi, and conjugate gradient. This is the
minimal substrate used by the first scalar Poisson/diffusion assembly from
runtime mesh geometry. It should remain a small backend-neutral contract so
later GPU implementations can provide the same operations without changing the
equation assembly layer.

The first equation assembly layer is scalar diffusion/Poisson on CPU. It
converts runtime mesh geometry into a CSR system with internal face coupling,
`fixedValue` and `zeroGradient` boundary contributions, and volume source
terms. Constraint patches such as `empty`, `wedge`, and `symmetryPlane` remain
solver constraints rather than ordinary diffusive boundary faces. This assembly
layer must stay separate from the linear solver implementation: equation code
builds a system, while CPU/GPU backends decide how that system is solved.
`ferrumSolver --solveScalarDiffusion <field>` is the first executable path
through that stack: it reads one scalar field, assembles one CPU system, solves
it with CG or Jacobi, reports residual and wall-clock time, and deliberately
does not write fields or enter the full CFD time loop.

`ferrumSolver --solvePoiseuille` is the first flow benchmark path. It uses the
same scalar operator for the fully developed axial Stokes balance driven by
`deltaP/L`, applies wall no-slip as `Ux=0`, compares the resulting volume
average against Hagen-Poiseuille, and reports timing and residuals. This is a
controlled bridge toward the later momentum-pressure solver; it is not yet a
SIMPLE, PISO, or full Navier-Stokes implementation.

`ferrumSolver --solveLaminarSimple` is the next bridge: it reads `U`, `p`,
`transportProperties`, `fvSchemes`, and `fvSolution`, constructs the first flow
operators on the same runtime `polyMesh` geometry, and writes solver reports as
JSON/Markdown. The current implementation is deliberately guarded: one damped
Jacobi CPU SIMPLE step is the stable default for the pipe benchmark, while
multi-step pressure correction remains an active solver-development target.
Momentum and pressure-correction linear solvers can be selected separately, so
experiments can run CG for momentum and Jacobi for pressure correction without
changing the case files. OpenFOAM-style `fvSolution` entries are the default
source for pressure and velocity under-relaxation and for per-equation linear
tolerances: `relaxationFactors.equations.U`, `relaxationFactors.fields.p`,
`solvers.U.tolerance`, `solvers.p.tolerance`, `solvers.p.solver PCG`,
`solvers.p.preconditioner DIC`, and optional `maxIter` values. `PCG` dispatches
to Ferrum's CPU preconditioned-CG path. OpenFOAM `DIC`/`FDIC` currently maps to
a diagonal PCG preconditioner, while true incomplete-Cholesky is tracked as a
later numerical upgrade. CLI flags remain explicit experiment overrides.
The pressure-correction bridge now follows the OpenFOAM shape more closely:
it builds cell-wise `rAU = V/A(U)`, assembles a variable-coefficient pressure
correction equation, and corrects `phi` with the pressure-equation flux instead
of recomputing the surface flux only from corrected cell velocity. The operator
and report boundaries are kept backend-neutral so the same assembly path can
later dispatch linear and nonlinear solves to CPU, GPU, or mixed CPU/GPU
resources.

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
  reference names and workflow anchors such as `gmshToFoam`, `checkMesh`, and
  `splitMeshRegions`.
