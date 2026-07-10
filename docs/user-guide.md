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
target/debug/gmshToFerrum.exe
target/debug/checkFerrumMesh.exe
target/debug/splitFerrumMeshRegions.exe
target/debug/ferrumSolver.exe
target/debug/ferrumPipeBenchmark.exe
target/debug/ferrumPlaneChannelBenchmark.exe
```

During development, commands can also be run through Cargo:

```powershell
cargo run -p ferrum-cli --bin gmshToFerrum -- --help
```

## Initialize A Case

Create a basic FerrumCFD case structure with:

```powershell
initFerrumCase cases\my_case
```

Equivalent combined command:

```powershell
ferrum initFerrumCase cases\my_case
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
- `internalField nonuniform List<scalar> ...;` with numeric values
- `internalField nonuniform List<vector> ...;` with flattened numeric values
- other `internalField nonuniform List<...> ...;` forms as a summary
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

The first supported mesh path is Gmsh 2.2 ASCII with `tri3`/`quad4` physical
surfaces and `prism6`/`hex8` physical volumes:

```powershell
gmshToFerrum path\to\mesh.msh -case cases\my_case
```

Equivalent Cargo command:

```powershell
cargo run -p ferrum-cli --bin gmshToFerrum -- path\to\mesh.msh -case cases\my_case
```

The importer maps:

- Gmsh physical surfaces to boundary patches where they are external faces
- all Gmsh physical surfaces to `faceZones`
- Gmsh physical volumes to `cellZones`

Internal multi-region interfaces are therefore preserved as `faceZones` even
when they are not external boundary patches.

The repository also contains a small SI pipe `.geo` with two near-wall prism
layers:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_gmsh_pipe_import.ps1
```

The script uses `tutorials/steadyIncompressible/laminarPipe/shared/geometry/pipe_prism2.geo`, writes the generated
`.msh` below `target/gmsh/`, imports it to `target/cases/gmsh_pipe`, and runs
`checkFerrumMesh`. It finds `gmsh.exe` from `PATH`; pass the trusted installation
explicitly with `-GmshExe <path-to-gmsh.exe>` when needed. This Gmsh pipe is
a benchmark fixture for comparing FerrumCFD and OpenFOAM on the same mesh. It
does not make OpenFOAM part of the normal FerrumCFD workflow.

For a Gmsh-based pipe mesh study, run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_gmsh_pipe_mesh_study.ps1
```

This generates `coarse`, `medium`, and `fine` variants from the same `.geo`,
imports each one into FerrumCFD, writes SI fields and benchmark metadata, runs
`checkFerrumMesh`, and optionally runs OpenFOAM for the same imported mesh. Use
`-SkipOpenFoam` for a quick Ferrum-only preparation pass, or increase
`-OpenFoamSteps` for a proper OpenFOAM convergence study.

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

This is not yet a full production-grade mesh validator, but `checkFerrumMesh`
is the command that will grow into that role.

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
gmshToFerrum path\to\mesh2d.msh -case cases\plate2d -emptyPatch frontAndBack
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
gmshToFerrum path\to\axisymmetric.msh -case cases\reactor_axi -wedgePatch wedgeMin -wedgePatch wedgeMax
```

Important solver rule: `wedge` must later be interpreted as an axisymmetric
constraint by the discretisation and field operations.
`checkFerrumMesh` now counts `wedge` patches and warns when the number of wedge
patches is odd, because axisymmetric wedge patches normally come in pairs.

## Generic Patch Types

OpenFOAM-compatible patch types can be assigned during import:

```powershell
gmshToFerrum path\to\mesh.msh -case cases\my_case -patchType symmetry=symmetryPlane
```

Shortcuts:

```powershell
-emptyPatch <patch>       # writes type empty
-wedgePatch <patch>       # writes type wedge
-symmetryPatch <patch>    # writes type symmetryPlane
```

## Combined CLI

The `ferrum` binary exposes lowerCamelCase commands. Utilities inspired by an
OpenFOAM workflow include `Ferrum` in their name so that FerrumCFD commands are
unambiguous:

```powershell
ferrum initFerrumCase cases\my_case
ferrum gmshToFerrum path\to\mesh.msh -case cases\my_case
ferrum checkFerrumMesh -case cases\my_case
ferrum splitFerrumMeshRegions -case cases\my_case -cellZones
ferrum solve -case cases\my_case --preflight --planJson target\ferrumSolverPlan.json
ferrum solve -case cases\my_case --runnerDryRun --maxRunnerSteps 2
```

The same naming convention is used by the dedicated binaries:

```powershell
initFerrumCase
gmshToFerrum
checkFerrumMesh
splitFerrumMeshRegions
ferrumSolver
ferrumPipeBenchmark
ferrumPlaneChannelBenchmark
```

## Units Policy

FerrumCFD-facing case data is SI-first. Unqualified numeric values are treated
as SI values by default:

- length: `m`
- pressure: `Pa`
- temperature: `K`
- velocity: `m/s`
- density: `kg/m3`
- dynamic viscosity: `Pa s`
- kinematic viscosity: `m2/s`

If a future parser accepts unit suffixes, non-SI values must be explicit, such
as `1 km` or `25 degC`. A bare `1` for a length-like quantity means `1 m`, not
`1 mm`, `1 cm`, or a solver-specific display unit.

OpenFOAM comparison cases are allowed to use OpenFOAM's native conventions when
needed. For example, incompressible OpenFOAM solvers commonly store `p` as
kinematic pressure in `m2/s2`. FerrumCFD benchmark scripts must convert those
results back to SI pressure in `Pa` before comparison.

## Laminar Pipe Benchmark

`tutorials/steadyIncompressible/laminarPipe/ferrum/case` is the first SI pipe simulation case used by the
benchmark suite. It contains the mesh, fields, material properties, and solver
dictionaries. The analytical Hagen-Poiseuille reference is separate at
`tutorials/steadyIncompressible/laminarPipe/analytical/pipeBenchmark`.

Regenerate the versioned medium-resolution case with:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\generate_laminar_pipe_case.ps1
```

The inlet velocity is a fully developed parabolic profile. The generator scales
the discrete patch values so the patch-integrated flow equals
`U_mean * inlet_area` for each mesh resolution.

Run the OpenFOAM comparison only as a benchmark artifact:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_poiseuille_benchmark.ps1
```

Run the mesh convergence study with:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_laminar_pipe_convergence.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_laminar_pipe_convergence.ps1 -OpenFoamSteps 1000
```

For the Gmsh-first workflow, use:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_gmsh_pipe_mesh_study.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_gmsh_pipe_mesh_study.ps1 -OpenFoamSteps 1000
```

If `gmsh.exe` is not on `PATH`, pass a trusted installation explicitly with
`-GmshExe <path-to-gmsh.exe>`. FerrumCFD does not auto-execute a same-named
binary discovered in Downloads.

The intended validation order is:

- generate several Gmsh meshes
- run OpenFOAM on the imported meshes and compare pressure loss to
  Hagen-Poiseuille
- select the converged reference mesh
- run the FerrumCFD Poiseuille benchmark on exactly that mesh
- use the selected mesh later for the full pressure-velocity and heat-transfer
  solvers

The generated OpenFOAM cases and reports stay below `target/benchmarks/`.
They are not part of the normal FerrumCFD workflow. Increase `-OpenFoamSteps`
when fine OpenFOAM cases still have moving SIMPLE residuals.
The current local Gmsh pipe mesh-study record is summarized in
`docs/benchmarks/gmsh-pipe-mesh-study.md`.

`scripts\run_poiseuille_benchmark.ps1` runs OpenFOAM Foundation 13
`foamRun -solver incompressibleFluid`, runs
`ferrumSolver --solvePoiseuille`, compares both with Hagen-Poiseuille, and
writes `target/benchmarks/laminar_pipe_compare.json` plus
`target/benchmarks/laminar_pipe_compare.md`. Use `-SkipOpenFoam
-UseExistingOpenFoamJson` when only the Ferrum side should be rerun against an
existing OpenFOAM result.

## Solver Preflight

`ferrumSolver` is the solver front door. Its `--preflight` and
`--runnerDryRun` modes do not execute CFD kernels; they read the case and print
the solver-neutral run plan used by current CPU and later GPU solver paths.

```powershell
ferrumSolver -case cases\my_case --preflight
ferrumSolver -case cases\my_case --preflight --planJson target\ferrumSolverPlan.json
ferrumSolver -case cases\my_case --runnerDryRun --maxRunnerSteps 2
ferrumSolver -case tutorials\steadyIncompressible\laminarPipe\ferrum\case --solveScalarDiffusion T --diffusivity 1 --linearSolver cg
ferrumSolver -case tutorials\steadyIncompressible\laminarPipe\ferrum\case --solvePoiseuille --linearSolver cg
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

It also builds a solver-state preview from initial fields below `0/`.
`volScalarField`, `volVectorField`, and `surfaceScalarField` are recognized as
field-storage candidates. Volume fields are checked against mesh cell counts;
surface fields are checked against mesh face counts. The report shows the
field region, class, internal value count, expected count, components, f64 slot
count, byte estimate, boundary patch counts, and whether the field storage is
CPU/GPU-capable. Uniform scalar/vector values are parsed into numeric
components when possible. Correctly shaped uniform fields are marked as
materializable CPU f64 buffers. Nonuniform `List<scalar>` and `List<vector>`
fields are count-checked and loaded into flattened f64 buffers when their value
count matches the mesh. Other nonuniform value types remain summary-only until
their type-specific loader exists. This still does not solve equations or
change field values.

The preflight also prepares solver runtime data. It builds compact
owner/neighbour connectivity, patch face ranges, cell centres, face centres,
owner-oriented face-area vectors, positive cell volumes, and materialized CPU
f64 buffers for fields that passed the solver-state checks. `--planJson` writes
a `runtimeData` summary with array sizes and buffer sizes, but it intentionally
does not dump the full geometry or field arrays into JSON. These runtime arrays
are the handoff point for the future CPU/GPU equation kernels.

FerrumCFD also contains the first executable CPU linear algebra foundation:
CSR matrices, matrix-vector products, residual calculation, Jacobi,
Gauss-Seidel, conjugate gradient, preconditioned-CG, and BiCGStab. The
preflight reports these as CPU linear-solver capabilities. They are the
solve-side substrate for the scalar diffusion and laminar flow assemblies
described below, but they are not yet driven by a complete CFD time-loop.

The first equation assembly foundation is now present as well. It can assemble
a scalar diffusion/Poisson CSR system on CPU from runtime mesh geometry with
internal-face diffusion coupling, `fixedValue` Dirichlet boundaries,
`zeroGradient` boundaries, and uniform volume source terms. Constraint patch
types such as `empty`, `wedge`, and `symmetryPlane` are not treated as normal
diffusive boundary faces. This is still an internal solver building block; it
is not yet automatically driven by `fvSchemes`, `fvSolution`, or a full
time-loop.

`--solveScalarDiffusion <field>` is the first opt-in executable equation path.
It reads the selected `volScalarField` from `0/`, converts supported
`boundaryField` entries into diffusion boundary conditions, assembles a CPU CSR
system, and solves it with `cg` or `jacobi`:

```powershell
ferrumSolver -case tutorials\steadyIncompressible\laminarPipe\ferrum\case --solveScalarDiffusion T --diffusivity 1 --linearSolver cg --solveTolerance 1e-8 --maxIterations 20000
```

Supported field boundary types for this path are currently `fixedValue uniform
<scalar>`, `zeroGradient`, and the constraint types `empty`, `wedge`, and
`symmetryPlane`. The command reports matrix nonzeros, boundary-face counts,
iteration count, convergence, residual norm, solution min/max/mean, and
wall-clock seconds. It does not write updated field files back to the case.

`--solvePoiseuille` is the first pressure-loss benchmark path. It solves the
fully developed axial Stokes balance as a source-driven scalar equation:

```text
-mu * laplacian(Ux) = deltaP / L
```

with `Ux=0` on wall patches and `zeroGradient` elsewhere. It is
benchmark-oriented. `deltaP`, `L`, and `D` must be supplied explicitly;
`mu` may be explicit or read from `constant/transportProperties`:

```powershell
ferrumSolver -case tutorials\steadyIncompressible\laminarPipe\ferrum\case --solvePoiseuille --pressureDrop 1.6032 --mu 0.001002 --length 1 --diameter 0.02 --wallPatch wall --linearSolver cg
```

The analytical reference is Hagen-Poiseuille:

```text
U_mean = deltaP * D^2 / (32 * mu * L)
deltaP = 32 * mu * L * U_mean / D^2
```

The command reports numerical mean velocity, analytical mean velocity,
relative error, flow rate, reconstructed pressure drop, solver iterations,
residual, and wall-clock seconds. It does not write velocity or pressure fields
back to the case.

`--solveLaminarSimple` is the first laminar incompressible pressure-velocity
path. It reads the OpenFOAM-like case dictionaries and fields that a
`simpleFoam` user expects:

- `0/U`
- `0/p`
- `constant/transportProperties`
- `system/fvSchemes`
- `system/fvSolution`
- `constant/polyMesh`

It builds the first finite-volume flow operators on the runtime mesh:
`phi = U_f . S_f`, `grad(p)`, `div(phi,U)`, and `laplacian(nu,U)`. The SIMPLE
path now reads the supported `system/fvSchemes` subset directly:

- `gradSchemes`: `Gauss linear` for `grad(p)` and `grad(U)`
- `divSchemes`: `div(phi,U) Gauss upwind` or
  `div(phi,U) Gauss linearUpwind grad(U)`
- `laplacianSchemes`: `Gauss linear corrected`, `orthogonal`, or `uncorrected`
- `interpolationSchemes`: `linear`
- `snGradSchemes`: `corrected`, `orthogonal`, or `uncorrected`

`Gauss upwind` remains the fully implicit conservative baseline.
`Gauss linearUpwind grad(U)` keeps that upwind matrix and adds the gradient
part as a deferred correction to the right-hand side. This is closer to the
OpenFOAM workflow without hiding artificial field clipping in the solver. For
pipe/axisymmetric benchmarks and general inlet/outlet workflows, the supported
boundary-condition contract is:

- `U`: inlet `fixedValue` including nonuniform/parabolic values, wall `noSlip`,
  outlet `zeroGradient`, plus OpenFOAM-style `inletOutlet` and
  `pressureInletOutletVelocity` for pressure-driven open boundaries. For
  `inletOutlet`, Ferrum uses `inletValue` on backflow and zero-gradient owner
  values on outflow.
- `p`: inlet `zeroGradient`, outlet `fixedValue`, and OpenFOAM-style
  `fixedFluxPressure` as a dynamic pressure-gradient boundary for flux
  consistency
- constraint patches: `empty`, `wedge`, and `symmetryPlane`

Current practical command:

```powershell
ferrumSolver -case tutorials\steadyIncompressible\laminarPipe\ferrum\case --solveLaminarSimple --solveTolerance 1e-6 --maxIterations 100 --solveReportJson target\benchmarks\laminar_pipe_laminar_simple.json --solveReportMarkdown target\benchmarks\laminar_pipe_laminar_simple.md
ferrumSolver -case tutorials\steadyIncompressible\laminarPipe\ferrum\case --solveLaminarSimple --maxSimpleIterations 2 --writeFinalFields target\benchmarks\laminar_pipe_fields\1
ferrumPipeBenchmark -case tutorials\steadyIncompressible\laminarPipe\ferrum\case --fields target\benchmarks\laminar_pipe_fields\1 --pressureDrop 1.6032 --mu 0.001002 --length 1 --diameter 0.02 --axis x --inletPatch inlet --outletPatch outlet --outJson target\benchmarks\laminar_pipe_fields\1.pipe.json
```

The first two commands are geometry-independent SIMPLE execution. The third is
optional external post-processing of stored fields. `--pressureDrop`, `--length`,
`--diameter`, `--axis`, and the sampling patch names are intentionally rejected
by `--solveLaminarSimple`; they belong only to `ferrumPipeBenchmark`.

The generic `--linearSolver` value is still accepted, but the laminar SIMPLE
path can also split the linear solver choice and linear controls by equation:

```powershell
ferrumSolver -case tutorials\steadyIncompressible\laminarPipe\ferrum\case --solveLaminarSimple --momentumLinearSolver bicgstab --pressureLinearSolver pcg --pressurePreconditioner DIC --maxSimpleIterations 20
ferrumSolver -case tutorials\steadyIncompressible\laminarPipe\ferrum\case --solveLaminarSimple --momentumSolveTolerance 1e-7 --pressureSolveTolerance 1e-9 --momentumMaxIterations 300 --pressureMaxIterations 400
ferrumSolver -case tutorials\steadyIncompressible\laminarPipe\ferrum\case --solveLaminarSimple --nNonOrthogonalCorrectors 1 --pRefCell 0 --pRefValue 0
ferrumSolver -case tutorials\steadyIncompressible\laminarPipe\ferrum\case --solveLaminarSimple --simpleConsistent true --maxSimpleIterations 20
```

By default, `--solveLaminarSimple` reads OpenFOAM-style relaxation factors from
`system/fvSolution`: `relaxationFactors.equations.U` for velocity and
`relaxationFactors.fields.p` for pressure. The CLI flags above are explicit
overrides for experiments. It also reads `solvers.U.tolerance`,
`solvers.p.tolerance`, `solvers.p.solver PCG`, `solvers.p.preconditioner DIC`,
`SIMPLE.nNonOrthogonalCorrectors`, `SIMPLE.pRefCell`, `SIMPLE.pRefValue`, and
`SIMPLE.consistent`, and optional `maxIter` values from `system/fvSolution`.
For pressure PCG, OpenFOAM `DIC`/`FDIC` maps to Ferrum's CPU IC(0)
incomplete-Cholesky preconditioner. `DILU` is rejected until a true
nonsymmetric ILU/DILU preconditioner exists; no diagonal fallback is applied.
OpenFOAM `smoothSolver` on `U` requires a `smoother` entry and executes the
matching CPU `GaussSeidel` or `symGaussSeidel` path. Explicit `bicgstab` remains available for nonsymmetric momentum
experiments. The generic `--solveTolerance` and `--maxIterations` flags remain
broad overrides for both equations. If present, OpenFOAM-style
`SIMPLE.residualControl` entries for `U` and `p` are the primary
early-convergence criteria. Ferrum follows the OpenFOAM Foundation 13 steady
SIMPLE form, where each field entry is one absolute scalar tolerance:

```text
SIMPLE
{
    nNonOrthogonalCorrectors 0;
    residualControl
    {
        U 1e-3;
        p 1e-2;
    }
}
```

Dictionary-valued criteria and criteria for fields not solved by the current
laminar path are rejected instead of being ignored. `U` uses the maximum
OpenFOAM-normalized initial residual over its three component solves. `p` uses
the initial residual from the first pressure solve in the SIMPLE iteration,
including when later non-orthogonal correctors perform additional pressure
solves. The linear-solver final residual and convergence flag remain separate
from this outer SIMPLE decision.

If `tolerance` or `maxIter` is absent, the SIMPLE path uses the OpenFOAM 13
`lduMatrix::solver` defaults `1e-6` and `1000`. Non-zero `relTol`, non-zero
`minIter`, and `smoothSolver nSweeps` values other than `1` are rejected for
now instead of being ignored or replaced silently.

Without `--maxSimpleIterations`, Ferrum uses the positive iteration count
derived from `controlDict` (`endTime - startTime` divided by `deltaT`). When
the resulting budget is greater than one, Ferrum defaults to at least
two SIMPLE iterations before convergence can be accepted. `endTime` or
`--maxSimpleIterations` is the maximum budget; all configured
`SIMPLE.residualControl` field tolerances permit an earlier stop. Continuity is
reported as a diagnostic and is not an undocumented extra stopping criterion.
Without `residualControl`, the solver runs to `--maxSimpleIterations` and reports
`converged=false` with `convergence-criteria-not-configured`. Hagen-Poiseuille
error, OpenFOAM comparison, and matched-time acceptance are evaluated by the
external benchmark scripts; they cannot stop, cap, roll back, or force a flow
direction in the generic solver. `minSimpleIterations` can still be set as a
case-level `SIMPLE` value.

Without `--writeFinalFields`, `--solveLaminarSimple` only reports and does not
write fields back to the case. With `--writeFinalFields <dir>`, Ferrum writes
final `U` and `p` files into the selected OpenFOAM-like time directory. The
internal fields come from the solved cell values, while the dimensions and
`boundaryField` entries are preserved from `0/U` and `0/p`.

The generic solver report records residuals, SIMPLE iterations, wall-clock time,
the active `fvSchemes` subset,
finite-volume operator summaries, boundary counts, general `U`/`p` field summaries,
continuity, per-iteration field changes, per-component momentum residuals,
momentum `A/H1` ranges, `adjustPhi` mass-balance changes, and final
pressure-assembly diagnostics under `pressureAssembly` in JSON and
`Pressure Assembly Diagnostics` in Markdown. These diagnostics include
`rAU/rAtU`, `HbyA`, `phiHbyA` before and after `adjustPhi`, pressure source,
pressure-equation flux, pressure matrix size/diagonal/off-diagonal summaries,
pressure flux, and corrected `phi`. JSON and Markdown reports also include a
`linearSolves` profile with converged/non-converged momentum predictors,
component momentum solves, pressure-correction solves, max/average linear
iterations per SIMPLE step, and final linear-solver convergence flags. The
iteration history, CSV, console, JSON, and Markdown outputs distinguish each
field's OpenFOAM-normalized initial residual from its final linear residual and
show the outer `residualControl` state independently.
`ferrumPipeBenchmark` writes a different JSON/Markdown report containing mean
axial velocity, Hagen-Poiseuille values, and named-patch pressure loss. Ferrum
sets `converged=true` when the configured outer `SIMPLE.residualControl`
criteria are checked and satisfied. Linear convergence remains explicitly
reported through the corresponding `SolverPerformance`-style fields. The pressure
bridge uses an internal momentum-equation object to apply equation relaxation,
retain cell-wise `A` and `H1` diagnostics for `rAU/rAtU`, reconstruct
`HbyA`, compute `phiHbyA` from HbyA with velocity boundary constraints applied,
run OpenFOAM-like `adjustPhi` on pressure-controlled open boundaries, including
`inletOutlet` faces only while they are outflowing, solve an absolute pressure
equation, correct `phi` from the pressure-equation flux, correct velocity as
`U = HbyA - rAtU grad(p)`, and carry that corrected surface flux into the next
SIMPLE iteration. The pressure equation now supports
OpenFOAM-like pressure reference anchoring for closed-pressure cases and executes
`nNonOrthogonalCorrectors + 1` pressure solves, with `phi` updated from the
final pressure solve. With `SIMPLE.consistent true`, Ferrum builds a
consistent `rAtU` correction from the current Rust momentum matrix and applies
the matching pressure-flux and velocity-correction terms. Non-orthogonal
correctors use an explicit face-flux correction from the pressure gradient and
the face area component not aligned with the cell-centre connection. The normal
solver path does not cap finite `U`, `p`, or `phi` updates and does not roll
back a finite SIMPLE step; non-finite values are treated as numerical failure.
True nonsymmetric ILU/DILU preconditioning is still solver-development work.

`ferrumPlaneChannelBenchmark` provides the same external separation for a 2D
parallel-plate case. It reads stored `U`/`p`, applies
`meanU = deltaP*H^2/(12*mu*L)`, and writes JSON/Markdown without changing the
simulation. Use `--pressureScale <rho>` only when post-processing OpenFOAM's
kinematic incompressible pressure; Ferrum fields remain SI Pa by default. The
reference `.geo`, case dictionaries, and SI inputs are under
`tutorials/steadyIncompressible/planeChannel/`.

For the standard pipe benchmark, the automated comparison command is:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_poiseuille_benchmark.ps1 -OpenFoamSteps 200
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_laminar_simple_matched_time_benchmark.ps1 -MatchedTimeSeconds 100
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_openfoam_laminar_pipe_step_sweep.ps1 -OpenFoamSteps 100,200,400,800,1200 -TargetRelativeError 0.01
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_laminar_simple_benchmark.ps1 -SkipOpenFoam -UseExistingOpenFoamJson
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_laminar_simple_iteration_sweep.ps1 -SimpleIterations 2,5,10,20,30
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_laminar_simple_mesh_study.ps1 -OpenFoamSteps 400 -FerrumSimpleIterations 100
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_laminar_simple_pressure_sweep.ps1 -VariantName medium,fine -SimpleIterations 50,100,200
```

The resulting Markdown tables record Ferrum pressure-loss error, OpenFOAM
pressure-loss error, Ferrum solve time, OpenFOAM wall time, and the shared SI
inputs such as `deltaP`, `rho`, `mu`, `L`, and `D`.
The solver roadmap in `docs/solver-roadmap.md` records the remaining work from
the current laminar SIMPLE prototype to the first production laminar
incompressible solver.
The iteration sweep is Ferrum-only: it fixes
`minSimpleIterations=maxSimpleIterations`, writes one generic solver report and
field directory per budget, and then runs the external pipe post-processor.
Solver convergence and benchmark agreement therefore remain separate artifacts.
The matched-time benchmark fixes OpenFOAM `endTime=MatchedTimeSeconds` with
`deltaT=1` and Ferrum `minSimpleIterations=maxSimpleIterations` to the same
number. For steady SIMPLE solvers this is an equal pseudo-time/iteration budget,
not a transient physical-time integration.
The OpenFOAM step sweep answers the inverse question: how many OpenFOAM 13
`foamRun -solver incompressibleFluid` steady iterations are needed to reach a target analytic
pressure-loss error. The older 100-1200 step table used axial-slice
extrapolation; rerun it with the current named-patch owner-cell sampler before
using it as an acceptance result.
The laminar SIMPLE mesh study generates coarse, medium, and fine pipe cases and
runs the current Ferrum SIMPLE path plus OpenFOAM on each. The earlier
coarse/medium/fine direct-pressure table must also be regenerated after the
sampler and report-separation change. The pressure-field sweep remains
Ferrum-only, fixes `minSimpleIterations=maxSimpleIterations`, and now writes a
generic report plus an external pipe report for every row. Pressure coupling
still requires validation on fine and deliberately skewed meshes.

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

With `--runnerDryRun`, FerrumCFD expands the current run plan into a capped
runner preview. The preview logs time-step starts, planned stage dispatch such
as `flow.residual` or `interfaces.flux`, backend choice, and planned write
events. It also prints runtime handles derived from `system/ferrumBackends`,
including CPU thread policy, CPU linear-solver availability, and GPU
backend/device metadata. GPU stages are reported as planned dispatch only until
executable GPU solver kernels exist. The same dry-run output also lists the
solver-state fields that would be available to the future runner, including
whether an initial field can already be materialized into a CPU buffer.
`--maxRunnerSteps <n>` limits the preview length. This does not update fields,
advance physics, or solve equations.

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

- Gmsh import currently supports Gmsh 2.2 ASCII, `tri3`/`quad4` surfaces, and
  `prism6`/`hex8` cells.
- Region splitting currently reads Ferrum-generated ASCII `polyMesh` files.
- `checkFerrumMesh` is currently a structural summary plus basic topology
  warning report, with field, interface, and backend configuration validation.
- `controlDict` validation and run scheduling are structural; adaptive time
  stepping and solver-specific time-loop behavior are not implemented yet.
- Geometry computation currently reports summary values; full OpenFOAM-grade
  geometry quality checks are not implemented yet.
- Initial field parsing currently summarizes fields, boundary entries, and
  solver-state storage shape; it validates boundary patch names, special patch
  boundary types, and internal value counts, but it does not yet validate
  dimensions against solver equations.
- `fvSchemes` and `fvSolution` are parsed and checked structurally for the
  broad solver preflight. The laminar SIMPLE path already consumes the
  documented `fvSchemes` subset and selected `fvSolution` entries, but many
  OpenFOAM schemes and solver controls are still future work.
- Constant property dictionaries are parsed structurally; solver-specific
  required material models and coefficients are not enforced yet.
- `ferrumSolver` is currently a preflight/run planner; `--runnerDryRun`
  previews scheduling only. `--solveScalarDiffusion <field>` and
  `--solvePoiseuille` can each execute one CPU equation solve, but full CFD
  time-loop execution is not implemented yet.
- CPU/GPU backend selection is validated as configuration and not yet
  executable solver behavior.
