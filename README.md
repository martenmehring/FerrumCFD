# FerrumCFD

FerrumCFD is an early Rust CFD platform prototype. The first milestone is
`ferrum-mesh`: import existing Gmsh meshes into an OpenFOAM-like case layout
without forcing users to change their usual workflow.

Start with the [User Guide](docs/user-guide.md). Longer-term design notes are
tracked in [docs/architecture.md](docs/architecture.md).
Current benchmark notes are kept under [docs/benchmarks](docs/benchmarks).

## First Commands

```powershell
cargo run -p ferrum-cli --bin ferrum -- initCase examples\membrane_reactor
cargo run -p ferrum-cli --bin ferrum -- gmshToFoam path\to\mesh.msh -case examples\membrane_reactor
cargo run -p ferrum-cli --bin ferrum -- checkMesh -case examples\membrane_reactor
cargo run -p ferrum-cli --bin ferrum -- solve -case examples\membrane_reactor --preflight --planJson target\ferrumSolverPlan.json
```

Alias binaries are provided too:

```powershell
cargo run -p ferrum-cli --bin initFerrumCase -- examples\membrane_reactor
cargo run -p ferrum-cli --bin gmshToFerrumFoam -- path\to\mesh.msh -case examples\membrane_reactor
cargo run -p ferrum-cli --bin checkFerrumMesh -- -case examples\membrane_reactor
cargo run -p ferrum-cli --bin splitFerrumMeshRegions -- -case examples\membrane_reactor -cellZones
cargo run -p ferrum-cli --bin ferrumSolver -- -case examples\membrane_reactor --preflight --planJson target\ferrumSolverPlan.json
cargo run -p ferrum-cli --bin ferrumSolver -- -case examples\membrane_reactor --runnerDryRun --maxRunnerSteps 2
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
- SI-first case values: unqualified lengths, pressures, temperatures, and
  velocities are interpreted as m, Pa, K, and m/s in FerrumCFD-facing data
- `tri3` and `quad4` physical surfaces as boundary patches
- `prism6` and `hex8` physical volumes as cell zones
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
- solver-state preflight recognizes `volScalarField`, `volVectorField`, and
  `surfaceScalarField`, checks internal field counts against mesh cells/faces,
  estimates components, f64 slots, byte counts, and reports CPU/GPU
  field-storage capability; valid `uniform`, `List<scalar>`, and
  `List<vector>` initial fields can be materialized as CPU f64 buffers without
  solving equations
- solver runtime preparation builds owner/neighbour connectivity, patch face
  ranges, cell centres, face centres, owner-oriented face-area vectors, cell
  volumes, and materialized CPU f64 field buffers for later CPU/GPU kernels
- mesh geometry summaries compute face areas, boundary area, and cell volumes
- special patch validation counts `empty`, `wedge`, and `symmetryPlane`
  patches and reports basic patch-range warnings
- `system/fvSchemes` and `system/fvSolution` are parsed and checked
  structurally for the solver preflight
- constant property dictionaries such as `transportProperties` and
  region-local property files are parsed structurally for the solver preflight
- `system/controlDict` is checked for basic run-control consistency such as
  positive `deltaT`, valid time controls, and write intervals
- backend policy can select CPU/GPU/auto per solver stage, including nonlinear
  solver steps and interface flux/coupling/source-term stages, with multi-CPU,
  core-count, thread, and GPU device metadata
- backend policy validation warns about duplicate stages, likely misspelled
  built-in stage names, and inconsistent CPU/GPU resource declarations
- `ferrumSolver` currently performs a solver preflight and prints a
  solver-neutral case plan, including the estimated time/write schedule and
  resolved backend choice per built-in run stage; `--planJson <file>` also
  writes the same plan as machine-readable JSON; executable solver kernels are
  not implemented yet
- `--runnerDryRun` previews the future solver runner for a capped number of
  steps and logs planned field state, CPU/GPU stage dispatch, runtime handles,
  and missing executable backend status without updating fields or solving
  equations
- `examples/laminar_pipe` provides a generated circular-pipe SI benchmark with
  a flow-normalized parabolic inlet, analytical Hagen-Poiseuille data, and
  OpenFOAM comparison/convergence scripts that record wall-clock runtime
- `examples/gmsh_pipe/pipe_prism2.geo` provides a parametric Gmsh pipe with two
  near-wall prism layers; `scripts/run_gmsh_pipe_mesh_study.ps1` creates
  coarse/medium/fine Gmsh meshes for OpenFOAM convergence and later FerrumCFD
  solver validation on the selected reference mesh

`splitFerrumMeshRegions` can write one region mesh per imported cell zone under
`constant/<region>/polyMesh`.

## Local Test Mesh

The importer was first tested with a private membrane reactor mesh generated
with Gmsh. Mesh files and generated case output are intentionally ignored by
Git because they can be large and may contain private geometry.
