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
target/debug/gmshToFerrumFoam.exe
target/debug/checkFerrumMesh.exe
target/debug/splitFerrumMeshRegions.exe
```

During development, commands can also be run through Cargo:

```powershell
cargo run -p ferrum-cli --bin gmshToFerrumFoam -- --help
```

## Case Layout

FerrumCFD writes an OpenFOAM-like case structure:

```text
case/
  constant/
    polyMesh/
      points
      faces
      owner
      neighbour
      boundary
      faceZones
      cellZones
    ferrumMeshSummary.txt
  system/
    controlDict
    fvSchemes
    fvSolution
```

Multi-region splitting writes region meshes below `constant/<region>/polyMesh`:

```text
case/
  constant/
    inner_zone/polyMesh/
    membrane/polyMesh/
    outer_zone/polyMesh/
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
- generated region meshes below `constant/<region>/polyMesh`
- topology warnings from import

This is not yet a full OpenFOAM-grade `checkMesh`, but it is the command that
will grow into that role.

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

## Axisymmetric Meshes

Axisymmetric cases use wedge meshes, again following OpenFOAM's workflow. The
two angular patches must be separate patches of type `wedge`.

Example:

```powershell
gmshToFerrumFoam path\to\axisymmetric.msh -case cases\reactor_axi -wedgePatch wedgeMin -wedgePatch wedgeMax
```

Important solver rule: `wedge` must later be interpreted as an axisymmetric
constraint by the discretisation and field operations.

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
ferrum gmshToFoam path\to\mesh.msh -case cases\my_case
ferrum checkMesh -case cases\my_case
ferrum splitMeshRegions -case cases\my_case -cellZones
```

The dedicated aliases remain available because they are closer to OpenFOAM
muscle memory:

```powershell
gmshToFerrumFoam
checkFerrumMesh
splitFerrumMeshRegions
```

## Backend Selection Direction

Backend selection is a design target, not a finished feature yet. The long-term
goal is to let users choose CPU, GPU, or mixed execution per solver component.

Example direction:

```text
ferrumBackends
{
    default cpu;

    flow
    {
        residual gpu;
        linearSolve gpu;
        pressureCorrection gpu;
    }

    chemistry
    {
        odeSolve cpu;
    }
}
```

The important rule is practical resource use: small or non-time-critical cases
must be allowed to stay on CPU, while expensive residuals, linear solves, or
other suitable kernels can run on GPU.

## Current Limitations

- Gmsh import currently supports Gmsh 2.2 ASCII, `quad4` surfaces, and `hex8`
  cells.
- Region splitting currently reads Ferrum-generated ASCII `polyMesh` files.
- `checkFerrumMesh` is currently a structural summary plus basic topology
  warning report.
- Solver support is not implemented yet.
- CPU/GPU backend selection is documented as a design target and not yet
  executable behavior.
