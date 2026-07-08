# FerrumCFD User Guide

This guide describes the current FerrumCFD workflow. FerrumCFD is still early,
but the command style and case layout intentionally follow familiar OpenFOAM
patterns where that helps users keep their existing habits.

## Build

From the repository root:

```powershell
cargo build --bins
```

The debug binaries are written to:

```text
target/debug/ferrum.exe
target/debug/initFerrumCase.exe
target/debug/gmshToFerrumFoam.exe
target/debug/checkFerrumMesh.exe
target/debug/splitFerrumMeshRegions.exe
target/debug/ferrumSolver.exe
```

During development, commands can also be run through Cargo:

```powershell
cargo run -p ferrum-cli --bin gmshToFerrumFoam -- --help
```

## Initialize A Case

Create a basic FerrumCFD case structure with:

```powershell
initFerrumCase cases\my_case
```

Equivalent combined command:

```powershell
ferrum initCase cases\my_case
```

For a multi-region case, region folders can be created immediately:

```powershell
initFerrumCase cases\reactor --regions inner_zone,membrane,outer_zone
```

The initializer writes templates for:

```text
0/
constant/
constant/polyMesh/
constant/interfaces
constant/transportProperties
system/controlDict
system/fvSchemes
system/fvSolution
system/ferrumBackends
```

Existing template files are not overwritten unless `--force` is passed.

## Case Layout

FerrumCFD writes an OpenFOAM-like case structure:

```text
case/
  0/
    p
    U
    T
    <region>/
      p
      T
  constant/
    polyMesh/
      points
      faces
      owner
      neighbour
      boundary
      faceZones
      cellZones
    interfaces
    transportProperties
    ferrumMeshSummary.txt
  system/
    controlDict
    fvSchemes
    fvSolution
    ferrumBackends
```

Multi-region splitting writes region meshes below `constant/<region>/polyMesh`:

```text
case/
  constant/
    inner_zone/polyMesh/
    membrane/polyMesh/
    outer_zone/polyMesh/
```

## Initial Field Files

FerrumCFD can read OpenFOAM-like initial field files from `0/`. This is the
case-input side for later solvers; it does not solve equations yet.

Single-region examples:

```text
0/p
0/U
0/T
0/YH2O
```

Multi-region examples:

```text
0/fluid/p
0/fluid/U
0/membrane/T
0/solid/T
```

Supported field entries for the current parser:

- `FoamFile` metadata, especially `class` and `object`
- `dimensions [ ... ];`
- `internalField uniform ...;`
- `internalField nonuniform List<...> ...;` as a summary
- `boundaryField { patch { type ...; value ...; } }`

Example:

```text
FoamFile
{
    version 2.0;
    format ascii;
    class volScalarField;
    object p;
}

dimensions [0 2 -2 0 0 0 0];
internalField uniform 0;

boundaryField
{
    inlet
    {
        type fixedValue;
        value uniform 10;
    }
    outlet
    {
        type zeroGradient;
    }
}
```

`checkFerrumMesh` reports the parsed field setup:

```text
initial fields:
  p: class=volScalarField dimensions=[0 2 -2 0 0 0 0] internal=uniform 0 boundaryPatches=2
    patch inlet type=fixedValue value=uniform 10
    patch outlet type=zeroGradient
```

## Import A Gmsh Mesh

The first supported mesh path is Gmsh 2.2 ASCII with `quad4` physical surfaces
and `hex8` physical volumes:

```powershell
gmshToFerrumFoam path\to\mesh.msh -case cases\my_case
```

Equivalent Cargo command:

```powershell
cargo run -p ferrum-cli --bin gmshToFerrumFoam -- path\to\mesh.msh -case cases\my_case
```

The importer maps:

- Gmsh physical surfaces to boundary patches where they are external faces
- all Gmsh physical surfaces to `faceZones`
- Gmsh physical volumes to `cellZones`

Internal multi-region interfaces are therefore preserved as `faceZones` even
when they are not external boundary patches.

## Interface Registry

FerrumCFD derives a general interface registry from the imported mesh. It is not
specific to a membrane reactor. The registry uses:

- `cellZones` to determine which region each cell belongs to
- `faceZones` to identify named interface surfaces
- `owner` and `neighbour` to determine the two adjacent regions
- `flipMap` to retain the source faceZone orientation

For example, a generic multi-region mesh can produce output like:

```text
interfaces:
  interface_name: region_a <-> region_b faces=100
```

For the membrane reactor test case this detects:

```text
mantle_inner_membrane_complete: inner_zone <-> membrane
mantle_membrane_outer_complete: membrane <-> outer_zone
```

Future models can use this registry for pressure-jump, heat-transfer,
species-transfer, membrane, conjugate, or other coupled-interface laws.

## Check A Mesh

Run:

```powershell
checkFerrumMesh -case cases\my_case
```

The current checker reports:

- point, cell, and face counts
- internal and boundary face counts
- boundary patches and patch types
- face zones
- cell zones
- geometry summary: face areas, boundary area, and cell volumes
- special patch validation for `empty`, `wedge`, and `symmetryPlane`
- generated region meshes below `constant/<region>/polyMesh`
- topology warnings from import
- field boundary entries against mesh patches

This is not yet a full OpenFOAM-grade `checkMesh`, but it is the command that
will grow into that role.

Example geometry output:

```text
geometry: cells=523600 faces=1580785 totalVolume=4.921636e4 minCellVolume=1.413155e-2 maxCellVolume=8.414263e-1 nonPositiveCellVolumes=0
geometry faces: minArea=3.532886e-3 maxArea=2.714353e0 totalBoundaryArea=1.437881e4
patch validation: patches=7 empty=0 wedge=0 symmetryPlane=0 warnings=0
```

When initial fields exist, their `boundaryField` entries are checked against
the mesh patches. `checkFerrumMesh` warns about missing entries, extra entries,
duplicates, and special mesh patches whose field boundary type should match the
mesh patch type, for example `empty` on an `empty` patch or `wedge` on a
`wedge` patch.

## Split Multi-Region Meshes

When a mesh contains volume physical groups, the importer writes them as
`cellZones`. Region meshes can then be written with:

```powershell
splitFerrumMeshRegions -case cases\my_case -cellZones
```

The splitter reads the Ferrum-generated ASCII `constant/polyMesh` and writes one
mesh per cell zone:

```text
constant/<cellZoneName>/polyMesh/
```

For region interface patches:

- existing external boundary patch names and types are preserved
- internal interface names are taken from `faceZones` where available
- interface patch type is currently written as `patch`
- `sourceFlippedFaces` is reported when source `faceZone` entries use
  `flipMap true`

OpenFOAM-style `faceZones` contain `faceLabels` and a `flipMap`. FerrumCFD
reads both. `faceLabels` identify interface faces. `flipMap` records whether a
face orientation is flipped relative to the zone orientation. The current
region splitter still determines each region boundary orientation from
`owner` and `neighbour`, but the `flipMap` data is retained in memory for later
interface and flux models.

For membrane and conjugate-transfer models, the positive flux direction should
be defined by interface metadata, not hidden inside each differential equation.
The equations should consume an oriented interface normal and then apply their
physical law, for example heat flux or species flux through a membrane.

## 2D Meshes

FerrumCFD follows the OpenFOAM convention: a 2D case is represented as a thin
3D mesh, and the suppressed-direction patches use the `empty` patch type.

Example:

```powershell
gmshToFerrumFoam path\to\mesh2d.msh -case cases\plate2d -emptyPatch frontAndBack
```

This writes:

```text
frontAndBack
{
    type empty;
    nFaces ...
    startFace ...
}
```

Important solver rule: `empty` must later be interpreted by FerrumCFD solvers as
a true reduced-dimension constraint. It must not be treated as a normal patch.
`checkFerrumMesh` now counts `empty` patches and warns about invalid patch face
ranges, but full reduced-dimension geometry validation is still a future
quality check.

## Axisymmetric Meshes

Axisymmetric cases use wedge meshes, again following OpenFOAM's workflow. The
two angular patches must be separate patches of type `wedge`.

Example:

```powershell
gmshToFerrumFoam path\to\axisymmetric.msh -case cases\reactor_axi -wedgePatch wedgeMin -wedgePatch wedgeMax
```

Important solver rule: `wedge` must later be interpreted as an axisymmetric
constraint by the discretisation and field operations.
`checkFerrumMesh` now counts `wedge` patches and warns when the number of wedge
patches is odd, because axisymmetric wedge patches normally come in pairs.

## Generic Patch Types

OpenFOAM-compatible patch types can be assigned during import:

```powershell
gmshToFerrumFoam path\to\mesh.msh -case cases\my_case -patchType symmetry=symmetryPlane
```

Shortcuts:

```powershell
-emptyPatch <patch>       # writes type empty
-wedgePatch <patch>       # writes type wedge
-symmetryPatch <patch>    # writes type symmetryPlane
```

## Combined CLI

The `ferrum` binary exposes OpenFOAM-like subcommands:

```powershell
ferrum initCase cases\my_case
ferrum gmshToFoam path\to\mesh.msh -case cases\my_case
ferrum checkMesh -case cases\my_case
ferrum splitMeshRegions -case cases\my_case -cellZones
ferrum solve -case cases\my_case --preflight --planJson target\ferrumSolverPlan.json
```

The dedicated aliases remain available because they are closer to OpenFOAM
muscle memory:

```powershell
initFerrumCase
gmshToFerrumFoam
checkFerrumMesh
splitFerrumMeshRegions
ferrumSolver
```

## Solver Preflight

`ferrumSolver` is the solver front door. At this stage it does not execute CFD
kernels yet. It reads the case and prints the solver-neutral run plan that
later CPU/GPU solver code should consume.

```powershell
ferrumSolver -case cases\my_case --preflight
ferrumSolver -case cases\my_case --preflight --planJson target\ferrumSolverPlan.json
```

Equivalent combined command:

```powershell
ferrum solve -case cases\my_case --preflight --planJson target\ferrumSolverPlan.json
```

The preflight reads:

- `system/controlDict`
- `system/fvSchemes`
- `system/fvSolution`
- `system/ferrumBackends`
- `constant/polyMesh`
- constant property dictionaries such as `transportProperties`
- region-local property dictionaries below `constant/<region>/`
- generated region meshes below `constant/<region>/polyMesh`
- `constant/interfaces`
- initial fields below `0/`

The output reports the detected dimensionality:

- `3d` for normal 3D meshes
- `2d-empty` when `empty` patches are present
- `axisymmetric-wedge` when `wedge` patches are present
- `mixed-special-patches` when both `empty` and `wedge` appear

It also prints the parsed numerical setup from `fvSchemes` and `fvSolution`,
the backend plan, and a run schedule. The run schedule estimates time steps and
write events when `controlDict` provides fixed `startTime`, `endTime`, and
`deltaT` values. It also resolves built-in run stages to CPU/GPU/auto, including
choices such as `flow.residual=gpu`, `chemistry.odeSolve=cpu`, and
`interfaces.flux=auto`. This is metadata only for now, but it is the intended
boundary between OpenFOAM-like case input and the future Rust/GPU solver stack.

The preflight warns about basic numerical setup gaps, such as missing standard
`fvSchemes` sections, missing `default` scheme entries, or initial fields that
do not have a matching `fvSolution.solvers` entry.

It also checks basic `controlDict` consistency: recognized `startFrom`,
`stopAt`, and `writeControl` modes, positive finite `deltaT`, valid
`writeInterval`, and an `endTime` that is not earlier than `startTime` for
`stopAt endTime`.

It also reads material and transport property dictionaries below `constant/`
and `constant/<region>/`. At this stage FerrumCFD checks the structure and
dimension-vector shape, but solver modules will later decide which properties
are required for each physics model.

With `--planJson <file>`, the same solver-neutral plan is also written as JSON.
That file is intended for future run managers, GUI tools, benchmark scripts,
and CPU/GPU solver launch code. The text preflight remains the normal
human-readable output.

## Interface Model Setup

Users should normally not edit `flipMap` by hand. `flipMap` belongs to the
mesh/faceZone definition and is read from the mesh data. Model intent belongs in
`constant/interfaces`.

Example:

```text
interfaces
{
    reactor_wall
    {
        regions (fluid solid);
        faceZone wall_interface;
        orientation fluid_to_solid;
        model heatTransfer;
    }
}
```

The orientation says which direction is positive for model quantities such as
pressure jump, heat flux, species flux, or membrane permeation. FerrumCFD then
maps that model direction onto mesh `owner`/`neighbour` and `flipMap`
orientation metadata.

This does not force the physical flow direction. If pressure, temperature, or
concentration differences reverse during a solve, the model should return a
negative value relative to this sign convention. The case dictionary only
defines what "positive" means.

`checkFerrumMesh` reads `constant/interfaces` when the file exists and checks
configured entries against the imported faceZones and region pairs:

```text
interface config:
  reactor_wall: faceZone=wall_interface sign=fluid->solid model=heatTransfer meshFaces=240
```

In a membrane reactor this is the correct place to define the positive
reference direction for permeation. If the sweep pressure becomes high enough
to push water back, the membrane model should compute the opposite sign. No
mesh `flipMap` change is required.

## Backend Selection Direction

Backend selection is parsed and validated as case configuration, but it is not
executable solver behavior yet. The long-term goal is to let users choose CPU,
GPU, or mixed execution per solver component.

Example direction:

```text
ferrumBackends
{
    default cpu;

    cpu
    {
        cpus auto;
        coresPerCpu auto;
        threads auto;
        threadPinning off;
        numa auto;
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

    gpu
    {
        backend auto;
        devices (auto);
        multiGpu auto;
        precision f64;
    }
}
```

The important rule is practical resource use: small or non-time-critical cases
must be allowed to stay on CPU, while expensive residuals, linear solves, or
other suitable kernels can run on GPU.

Nonlinear solvers are treated as first-class GPU candidates. A Newton-style
solve can select backend execution for `residual`, `jacobian`,
`linearSolve`, and the enclosing `nonlinearSolve` loop. Chemistry ODEs can
also run on GPU as batched per-cell ODE solves. `odeSolve cpu` is still a
valid choice when the GPU is busy, unavailable, memory-limited, or when a
particular stiff chemistry setup performs better on CPU.

Interface stages are also first-class backend candidates. `interfaces.flux`
belongs to model flux evaluation, `interfaces.coupling` to region-to-region
coupling work, and `interfaces.sourceTerms` to equation source-term assembly.
For a membrane model, pressure or concentration differences should determine
the physical flux sign; the backend choice only decides where the computation
runs.

CPU resource policy:

- `cpus auto;` lets FerrumCFD discover the number of physical CPU packages or
  sockets.
- `cpus N;` declares that `N` physical CPUs may be used.
- `coresPerCpu auto;` lets FerrumCFD discover cores per CPU package.
- `coresPerCpu N;` declares `N` physical cores per CPU package.
- `threads auto;` lets FerrumCFD choose a sensible worker count.
- `threads N;` pins the solver policy to `N` CPU worker threads.
- `threadPinning auto|on|off;` is reserved for explicit CPU affinity control.
- `numa auto|on|off;` leaves room for multi-socket CPU machines without forcing
  a NUMA policy before the runtime exists.

For mixed CPU/GPU runs, both `cpu { ... }` and `gpu { ... }` should be present.
`checkFerrumMesh` warns if a policy selects or may select both CPU and GPU but
does not explicitly describe both resource pools.

GPU resource policy:

- `devices (auto);` lets FerrumCFD pick the GPU.
- `devices (0);` selects one GPU.
- `devices (0 1);` permits multi-GPU execution when a backend and solver
  support it.
- `multiGpu auto|on|off;` controls whether multi-GPU execution may be used.

`checkFerrumMesh` reads `system/ferrumBackends` when the file exists:

```text
backend config: default=cpu cpuCpus=auto cpuCoresPerCpu=auto cpuThreads=auto cpuPinning=off cpuNuma=auto gpuBackend=auto gpuDevices=auto multiGpu=auto precision=f64
  mesh: import=cpu, checks=cpu
  interfaces: flux=auto, coupling=auto, sourceTerms=auto
  flow: nonlinearSolve=auto, residual=auto, jacobian=auto, linearSolve=auto, pressureCorrection=auto
  chemistry: residual=auto, jacobian=auto, nonlinearSolve=auto, odeSolve=auto
backend resources: usesCpu=true usesGpu=true mixed=true
```

Allowed execution choices are `cpu`, `gpu`, and `auto`. The `gpu.backend`
setting currently accepts `auto`, `wgpu`, `cuda`, and `hip`; `gpu.precision`
accepts `auto`, `f32`, and `f64`. CPU `cpus`, `coresPerCpu`, and `threads`
accept `auto` or a positive integer.

The backend preflight also warns about duplicate stage entries, likely
misspelled built-in stage names, and resource contradictions such as selecting
multiple GPU devices while `multiGpu off` is configured. Custom backend
sections are allowed, but the current preflight reports that they are not yet
consumed by built-in solver code.

## Current Limitations

- Gmsh import currently supports Gmsh 2.2 ASCII, `quad4` surfaces, and `hex8`
  cells.
- Region splitting currently reads Ferrum-generated ASCII `polyMesh` files.
- `checkFerrumMesh` is currently a structural summary plus basic topology
  warning report, with field, interface, and backend configuration validation.
- `controlDict` validation and run scheduling are structural; adaptive time
  stepping and solver-specific time-loop behavior are not implemented yet.
- Geometry computation currently reports summary values; full OpenFOAM-grade
  geometry quality checks are not implemented yet.
- Initial field parsing currently summarizes fields and boundary entries; it
  validates boundary patch names and special patch boundary types, but it does
  not yet validate dimensions against solver equations.
- `fvSchemes` and `fvSolution` are parsed and checked structurally for the
  solver preflight; their entries are not yet consumed by executable
  discretisation or linear solver kernels.
- Constant property dictionaries are parsed structurally; solver-specific
  required material models and coefficients are not enforced yet.
- `ferrumSolver` is currently a preflight/run planner; CFD solver kernels are
  not implemented yet.
- CPU/GPU backend selection is validated as configuration and not yet
  executable solver behavior.
