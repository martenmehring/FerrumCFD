# FerrumCFD

FerrumCFD is an early Rust CFD platform prototype. The first milestone is
`ferrum-mesh`: import existing Gmsh meshes into an OpenFOAM-like case layout
without forcing users to change their usual workflow.

## First Commands

```powershell
cargo run -p ferrum-cli --bin ferrum -- gmshToFoam path\to\mesh.msh -case examples\membrane_reactor
cargo run -p ferrum-cli --bin ferrum -- checkMesh -case examples\membrane_reactor
```

Alias binaries are provided too:

```powershell
cargo run -p ferrum-cli --bin gmshToFerrumFoam -- path\to\mesh.msh -case examples\membrane_reactor
cargo run -p ferrum-cli --bin checkFerrumMesh -- -case examples\membrane_reactor
cargo run -p ferrum-cli --bin splitFerrumMeshRegions -- -case examples\membrane_reactor -cellZones
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

`splitFerrumMeshRegions` currently lists detected cell zones. Full region
mesh splitting is the next milestone.

## Local Test Mesh

The importer was first tested with a private membrane reactor mesh generated
with Gmsh. Mesh files and generated case output are intentionally ignored by
Git because they can be large and may contain private geometry.
