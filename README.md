# FerrumCFD

FerrumCFD is an early Rust CFD platform prototype. The current compatibility
reader imports existing Gmsh meshes and OpenFOAM-like dictionaries while the
native `FerrumFile v1` format is being introduced.

The first executable flow-solver milestone is now available:
`ferrumSolver --solveLaminarSimple` is a Rust finite-volume SIMPLE solver for
steady, laminar, incompressible flow. It reads OpenFOAM-like case dictionaries,
reports OpenFOAM-style outer and linear residuals separately, and has been
validated on a 3D circular pipe and a true 2D plane channel. It is the CPU
baseline for the planned parallel CPU and GPU backends; the remaining work
toward a production `simpleFoam`-class solver is tracked explicitly.

Start with the [User Guide](docs/user-guide.md). Longer-term design notes are
tracked in [docs/architecture.md](docs/architecture.md).
Current benchmark notes are kept under [docs/benchmarks](docs/benchmarks).
Release-level changes are summarized in [CHANGELOG.md](CHANGELOG.md), and the
solver completion criteria remain in [docs/solver-roadmap.md](docs/solver-roadmap.md).

The repository follows an OpenFOAM-13-inspired separation: reusable Rust code
lives under `src/`, compiled applications under `applications/`, and curated
validation bundles under `tutorials/`. Each bundle keeps Ferrum and OpenFOAM 13
cases independent and adds an analytical or documented benchmark reference
where appropriate.

## First Commands

```powershell
cargo run -p ferrum-cli --bin ferrum -- checkFerrumMesh -case tutorials\steadyIncompressible\laminarPipe\ferrum\case
cargo run -p ferrum-cli --bin ferrum -- solve -case tutorials\steadyIncompressible\laminarPipe\ferrum\case --preflight --planJson target\ferrumSolverPlan.json
cargo run -p ferrum-cli --bin ferrum -- solve -case tutorials\steadyIncompressible\laminarPipe\ferrum\case --solveLaminarSimple --maxSimpleIterations 2
```

Alias binaries are provided too:

```powershell
cargo run -p ferrum-cli --bin checkFerrumMesh -- -case tutorials\steadyIncompressible\laminarPipe\ferrum\case
cargo run -p ferrum-cli --bin ferrumSolver -- -case tutorials\steadyIncompressible\laminarPipe\ferrum\case --preflight --planJson target\ferrumSolverPlan.json
cargo run -p ferrum-cli --bin ferrumSolver -- -case tutorials\steadyIncompressible\laminarPipe\ferrum\case --runnerDryRun --maxRunnerSteps 2
cargo run -p ferrum-cli --bin ferrumSolver -- -case tutorials\steadyIncompressible\laminarPipe\ferrum\case --solveScalarDiffusion T --diffusivity 1 --linearSolver cg
cargo run -p ferrum-cli --bin ferrumSolver -- -case tutorials\steadyIncompressible\laminarPipe\ferrum\case --solvePoiseuille --pressureDrop 1.6032 --mu 0.001002 --length 1 --diameter 0.02 --linearSolver cg
cargo run -p ferrum-cli --bin ferrumSolver -- -case tutorials\steadyIncompressible\laminarPipe\ferrum\case --solveLaminarSimple --solveTolerance 1e-6 --maxIterations 100 --solveReportJson target\benchmarks\laminar_pipe_laminar_simple.json --solveReportMarkdown target\benchmarks\laminar_pipe_laminar_simple.md
cargo run -p ferrum-cli --bin ferrumSolver -- -case tutorials\steadyIncompressible\laminarPipe\ferrum\case --solveLaminarSimple --maxSimpleIterations 2 --writeFinalFields target\benchmarks\laminar_pipe_fields\1
cargo run -p ferrum-cli --bin ferrumPipeBenchmark -- -case tutorials\steadyIncompressible\laminarPipe\ferrum\case --fields target\benchmarks\laminar_pipe_fields\1 --pressureDrop 1.6032 --mu 0.001002 --length 1 --diameter 0.02 --axis x --inletPatch inlet --outletPatch outlet
cargo run -p ferrum-cli --bin ferrumPlaneChannelBenchmark -- -case tutorials\steadyIncompressible\planeChannel\ferrum\case --fields target\benchmarks\plane_channel\ferrum_fields\1 --pressureDrop 0.6012 --mu 0.001002 --length 1 --gap 0.02 --depth 0.001
```

Run the first Ferrum/OpenFOAM/analytic pipe benchmarks with:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_poiseuille_benchmark.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_laminar_simple_matched_time_benchmark.ps1 -MatchedTimeSeconds 100
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_openfoam_laminar_pipe_step_sweep.ps1 -OpenFoamSteps 100,200,400,800,1200 -TargetRelativeError 0.01
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_laminar_simple_benchmark.ps1 -SkipOpenFoam -UseExistingOpenFoamJson
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_laminar_simple_iteration_sweep.ps1 -SimpleIterations 2,5,10,20,30
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_laminar_simple_mesh_study.ps1 -OpenFoamSteps 400 -FerrumSimpleIterations 100
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_laminar_simple_pressure_sweep.ps1 -VariantName medium,fine -SimpleIterations 50,100,200
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_laminar_pipe_convergence.ps1 -OpenFoamSteps 200
```

Prepare the separate 2D plane-channel test case from its shared Gmsh geometry
with:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\prepare_plane_channel_case.ps1 -GmshExe "C:\path\to\gmsh.exe" -Force
```

## 2D And Axisymmetric Meshes

FerrumCFD follows the OpenFOAM mesh workflow:

- 2D planar cases are imported as one-cell-thick 3D meshes. The front/back
  patches must use the OpenFOAM `empty` patch type.
- Axisymmetric cases are imported as wedge meshes. The two angular patches
  must use the OpenFOAM `wedge` patch type.

Examples:

```powershell
gmshToFerrum path\to\mesh2d.msh -case cases\plate2d -emptyPatch frontAndBack
gmshToFerrum path\to\axisymmetric.msh -case cases\reactor_axi -wedgePatch wedgeMin -wedgePatch wedgeMax
```

Generic OpenFOAM-compatible patch types can be written with:

```powershell
gmshToFerrum path\to\mesh.msh -case cases\mesh -patchType symmetry=symmetryPlane
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
- CPU linear algebra now has a small executable CSR foundation with matrix-vector
  products, residuals, Jacobi, conjugate-gradient, preconditioned-CG, and
  BiCGStab solves for the first Poisson/diffusion and flow assembly steps,
  including diagonal and incomplete-Cholesky IC(0) preconditioners for CPU PCG
- scalar diffusion/Poisson assembly can build and solve an opt-in CPU CSR system
  from runtime mesh geometry with internal-face diffusion, `fixedValue`,
  `zeroGradient`, and volume source terms; it reports iterations, residual,
  solution summary, and wall-clock time without writing field files
- `--solvePoiseuille` runs a source-driven axial Stokes/Poiseuille benchmark on
  the pipe mesh, requires explicit `deltaP`, `L`, and `D` benchmark inputs
  (`mu` may come from `transportProperties`), and reports mean velocity, flow rate,
  Hagen-Poiseuille reference values, relative error, residual, and wall-clock
  time without writing field files
- `--solveLaminarSimple` is the first laminar incompressible SIMPLE
  path. It reads `U`, `p`, `transportProperties`, `fvSchemes`, and
  `fvSolution`, builds finite-volume `phi`, `grad(p)`, `div(phi,U)`, and
  `laplacian(nu,U)` operators on the runtime `constant/polyMesh` geometry,
  writes JSON/Markdown reports, can write final OpenFOAM-like `U`/`p` fields
  with `--writeFinalFields <dir>`, supports separate momentum and pressure
  correction linear-solver choices, executes OpenFOAM `smoothSolver` with the
  configured CPU `GaussSeidel` or `symGaussSeidel` smoother while keeping
  explicit `bicgstab` available for nonsymmetric experiments, supports `pRefCell`/`pRefValue` and
  `nNonOrthogonalCorrectors`, reads `SIMPLE.consistent` and
  OpenFOAM Foundation-style scalar `SIMPLE.residualControl`, leaves
  Hagen-Poiseuille acceptance to external benchmark tooling, reports the
  maximum initial `U` component residual, per-component initial/final
  residuals, separate linear-solve convergence profiles, and
  `adjustPhi` mass-balance data, reads the supported `fvSchemes` subset for
  `grad(p)`, `grad(U)`, `div(phi,U)`, `laplacian`, `interpolation`, and
  `snGrad`, supports flux-dependent open velocity boundaries such as
  `inletOutlet`, reconstructs `HbyA`, computes boundary-constrained `phiHbyA`,
  records pressure-assembly diagnostics for `rAU/rAtU`, `HbyA`, pressure
  source, pressure matrix, pressure flux, and corrected `phi`, maps
  OpenFOAM `DIC`/`FDIC` on pressure PCG to a CPU IC(0) preconditioner,
  and runs an uncapped OpenFOAM-shaped pressure-velocity correction path without
  embedding pipe geometry or analytic acceptance criteria
- `ferrumPipeBenchmark` is a separate post-processor. It reads stored `U`/`p`
  fields, applies explicit pipe geometry/reference inputs, and writes
  Hagen-Poiseuille diagnostics without changing solver convergence or fields
- `ferrumPlaneChannelBenchmark` is the corresponding external plane-Poiseuille
  post-processor for a 2D parallel-plate case; it supports an explicit pressure
  scale when reading OpenFOAM kinematic pressure
- mesh geometry summaries compute face areas, boundary area, and cell volumes
- special patch validation counts `empty`, `wedge`, and `symmetryPlane`
  patches and reports basic patch-range warnings
- `system/fvSchemes` and `system/fvSolution` are parsed and checked for the
  solver preflight; the laminar SIMPLE path already executes a supported
  `fvSchemes` subset
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
  writes the same plan as machine-readable JSON; `--solveScalarDiffusion
  <field>`, `--solvePoiseuille`, and `--solveLaminarSimple` can execute current
  CPU equation paths, but full CFD time-loop execution is not implemented yet
- `--runnerDryRun` previews the future solver runner for a capped number of
  steps and logs planned field state, CPU/GPU stage dispatch, runtime handles,
  and missing executable backend status without updating fields or solving
  equations
- `tutorials/steadyIncompressible/laminarPipe/ferrum/case` provides a generated circular-pipe SI simulation case
  with a flow-normalized parabolic inlet. Analytic reference data lives outside
  the case in `tutorials/steadyIncompressible/laminarPipe/analytical/pipeBenchmark`; comparison scripts record
  Ferrum/OpenFOAM wall-clock runtime under `target/benchmarks`
- `scripts/run_laminar_simple_iteration_sweep.ps1` runs fixed
  `minSimpleIterations=maxSimpleIterations` Ferrum SIMPLE sweeps, stores generic
  reports and fields, then produces separate external pipe diagnostics
- `scripts/run_laminar_simple_matched_time_benchmark.ps1` runs the current
  laminar SIMPLE comparison with the same steady pseudo-time/iteration budget
  for OpenFOAM and Ferrum, for example OpenFOAM `endTime=100` and Ferrum
  `100` SIMPLE iterations
- `scripts/run_openfoam_laminar_pipe_step_sweep.ps1` measures how many
  OpenFOAM 13 `foamRun -solver incompressibleFluid` steady iterations are
  needed to reach a target
  Hagen-Poiseuille pressure-loss error
- `scripts/run_laminar_simple_mesh_study.ps1` runs coarse/medium/fine mesh
  studies for the current Ferrum laminar SIMPLE path and the OpenFOAM reference
- `scripts/run_laminar_simple_pressure_sweep.ps1` runs Ferrum-only
  pressure-field convergence sweeps over SIMPLE iteration budgets and mesh
  variants
- `docs/solver-roadmap.md` tracks the work needed to turn the current
  `--solveLaminarSimple` prototype into the first production laminar
  incompressible solver, including numerics, boundary conditions, schemes,
  benchmarks, performance, and later CPU/GPU execution
- `tutorials/steadyIncompressible/laminarPipe/shared/geometry/pipe_prism2.geo` provides a parametric Gmsh pipe with two
  near-wall prism layers; `scripts/run_gmsh_pipe_mesh_study.ps1` creates
  coarse/medium/fine Gmsh meshes for OpenFOAM convergence and FerrumCFD
  Poiseuille validation on the selected reference mesh

`splitFerrumMeshRegions` can write one region mesh per imported cell zone under
`constant/<region>/polyMesh`.

## Local Test Mesh

The importer was first tested with a private membrane reactor mesh generated
with Gmsh. Mesh files and generated case output are intentionally ignored by
Git because they can be large and may contain private geometry.

## License

FerrumCFD is licensed under the [MIT License](LICENSE).
