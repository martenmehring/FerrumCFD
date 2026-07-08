# FerrumCFD

FerrumCFD is an early Rust CFD platform prototype. The first milestone is
`ferrum-mesh`: import existing Gmsh meshes into an OpenFOAM-like case layout
without forcing users to change their usual workflow.

Start with the [User Guide](docs/user-guide.md). Longer-term design notes are
tracked in [docs/architecture.md](docs/architecture.md).

## First Commands

```powershell
cargo run -p ferrum-cli --bin ferrum -- initCase examples\membrane_reactor
cargo run -p ferrum-cli --bin ferrum -- gmshToFoam path\to\mesh.msh -case examples\membrane_reactor
cargo run -p ferrum-cli --bin ferrum -- checkMesh -case examples\membrane_reactor
```

Alias binaries are provided too:

```powershell
cargo run -p ferrum-cli --bin initFerrumCase -- examples\membrane_reactor
cargo run -p ferrum-cli --bin gmshToFerrumFoam -- path\to\mesh.msh -case examples\membrane_reactor
cargo run -p ferrum-cli --bin checkFerrumMesh -- -case examples\membrane_reactor
cargo run -p ferrum-cli --bin splitFerrumMeshRegions -- -case examples\membrane_reactor -cellZones
```

## 2D And Axisymmetric Meshes

FerrumCFD follows the OpenFOAM mesh workflow:

- 2D planar cases are imported as one-cell-thick 3D meshes. The front/back
  patches must use the OpenFOAM `empty` patch type.
- Axisymmetric cases are imported as wedge meshes. The two angular patches
  must use the OpenFOAM `wedge` patch type.

Examples:

```powershell
gmshToFerrumFoam path\to\mesh2d.msh -case cases\plate2d -emptyPatch frontAndBack
gmshToFerrumFoam path\to\axisymmetric.msh -case cases\reactor_axi -wedgePatch wedgeMin -wedgePatch wedgeMax
```

Generic OpenFOAM-compatible patch types can be written with:

```powershell
gmshToFerrumFoam path\to\mesh.msh -case cases\mesh -patchType symmetry=symmetryPlane
```

## Current Mesh Scope

The importer currently targets the membrane reactor test mesh shape:

- Gmsh 2.2 ASCII `.msh`
- `quad4` physical surfaces as boundary patches
- `hex8` physical volumes as cell zones
- OpenFOAM-like `constant/polyMesh` output: `points`, `faces`, `owner`,
  `neighbour`, `boundary`, `faceZones`, `cellZones`
- external Gmsh physical surfaces become boundary patches
- all Gmsh physical surfaces, including internal multi-region interfaces,
  are preserved as `faceZones`
- patch types can be written as OpenFOAM-compatible `patch`, `empty`, `wedge`,
  `symmetryPlane`, or custom patch types
- OpenFOAM-like initial fields below `0/` are parsed for `dimensions`,
  `internalField`, and `boundaryField` summaries
- field `boundaryField` entries are checked against mesh patch names and
  special patch types
- mesh geometry summaries compute face areas, boundary area, and cell volumes
- special patch validation counts `empty`, `wedge`, and `symmetryPlane`
  patches and reports basic patch-range warnings
- backend policy can select CPU/GPU/auto per solver stage, including nonlinear
  solver steps, with multi-CPU, core-count, thread, and GPU device metadata

`splitFerrumMeshRegions` can write one region mesh per imported cell zone under
`constant/<region>/polyMesh`.

## Local Test Mesh

The importer was first tested with a private membrane reactor mesh generated
with Gmsh. Mesh files and generated case output are intentionally ignored by
Git because they can be large and may contain private geometry.
