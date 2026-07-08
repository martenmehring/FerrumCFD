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
ferrumSolver -case case
```

The goal is not to copy OpenFOAM internals. The goal is to keep the established
case layout, patch naming, command rhythm, and dictionary style where that
reduces user friction.

## Dictionary And Field Parsing

FerrumCFD now has a shared OpenFOAM-style token/cursor parser used by case
dictionaries such as:

- `constant/interfaces`
- `system/ferrumBackends`
- initial field files below `0/`

Initial field parsing is intentionally structural at this stage. It reads
`FoamFile`, `dimensions`, `internalField`, and `boundaryField`, then reports the
setup in `checkFerrumMesh`. Solver modules will later interpret these fields in
the context of equations, patch constraints, and dimensions.

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

- `checkFerrumMesh` should eventually reject invalid `empty` setups, such as
  non-empty patches in the suppressed direction or more than one cell through
  the reduced dimension.
- `checkFerrumMesh` should eventually reject invalid `wedge` setups, such as
  missing wedge pairs, wrong patch pairing, inconsistent angles, or geometry
  that cannot be treated as axisymmetric.
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

    chemistry
    {
        nonlinearSolve gpu;
        residual gpu;
        jacobian gpu;
        odeSolve cpu;
    }

    cpu
    {
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
