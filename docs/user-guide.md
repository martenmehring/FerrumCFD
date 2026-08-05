# FerrumCFD User Guide

This guide describes the current FerrumCFD workflow. FerrumCFD is still early,
but the command style and case layout intentionally follow familiar OpenFOAM
patterns where that helps users keep their existing habits.

## Build

The only normal build prerequisite is a current Rust toolchain. From the
repository root:

```console
cargo build --bins
```

The debug binaries are written below `target/debug/`. On Windows, executable
names receive the normal `.exe` suffix:

```text
target/debug/ferrum
target/debug/initFerrumCase
target/debug/gmshToFerrum
target/debug/checkFerrumMesh
target/debug/splitFerrumMeshRegions
target/debug/ferrumRun
```

During development, commands can also be run through Cargo:

```console
cargo run -p ferrum-cli --bin gmshToFerrum -- --help
```

## Initialize A Case

Create a basic FerrumCFD case structure with:

```console
initFerrumCase cases/my_case
```

Equivalent combined command:

```console
ferrum initFerrumCase cases/my_case
```

For a multi-region case, region folders can be created immediately:

```console
initFerrumCase cases/reactor --regions inner_zone,membrane,outer_zone
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
- `dimensions [ ... ];` with exactly five legacy or seven current exponents;
  five-entry sets are normalized with zero electric-current and
  luminous-intensity terms
- `internalField uniform ...;`
- `internalField nonuniform List<scalar> ...;`, `scalarField ...;`, or
  `Field<scalar> ...;` with numeric values
- `internalField nonuniform List<vector> ...;`, `vectorField ...;`, or
  `Field<vector> ...;` with vector values
- `boundaryField { patch { type ...; value ...; } }`

Only those six exact, unquoted nonuniform type names are accepted. Dictionary
directives such as `#include` and `#includeFunc` are not resolved yet and are
rejected with their source path and line instead of being silently skipped.

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

```console
gmshToFerrum path/to/mesh.msh -case cases/my_case
```

Equivalent Cargo command:

```console
cargo run -p ferrum-cli --bin gmshToFerrum -- path/to/mesh.msh -case cases/my_case
```

The importer maps:

- Gmsh physical surfaces to boundary patches where they are external faces
- all Gmsh physical surfaces to `faceZones`
- Gmsh physical volumes to `cellZones`

Internal multi-region interfaces are therefore preserved as `faceZones` even
when they are not external boundary patches.

The repository also contains neutral Gmsh source examples under the tutorial
bundles. Users who already have a Gmsh mesh may import its supported 2.2 ASCII
form with `gmshToFerrum`; Gmsh is not required to build or solve the checked-in
Ferrum cases. No repository wrapper or combined solver workflow is required.

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

```console
checkFerrumMesh -case cases/my_case
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

```console
splitFerrumMeshRegions -case cases/my_case -cellZones
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

```console
gmshToFerrum path/to/mesh2d.msh -case cases/plate2d -emptyPatch frontAndBack
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

```console
gmshToFerrum path/to/axisymmetric.msh -case cases/reactor_axi -wedgePatch wedgeMin -wedgePatch wedgeMax
```

Important solver rule: `wedge` must later be interpreted as an axisymmetric
constraint by the discretisation and field operations.
`checkFerrumMesh` now counts `wedge` patches and warns when the number of wedge
patches is odd, because axisymmetric wedge patches normally come in pairs.

## Generic Patch Types

OpenFOAM-compatible patch types can be assigned during import:

```console
gmshToFerrum path/to/mesh.msh -case cases/my_case -patchType symmetry=symmetryPlane
```

Shortcuts:

```console
-emptyPatch <patch>       # writes type empty
-wedgePatch <patch>       # writes type wedge
-symmetryPatch <patch>    # writes type symmetryPlane
```

## Combined CLI

The `ferrum` binary exposes lowerCamelCase commands. Utilities inspired by an
OpenFOAM workflow include `Ferrum` in their name so that FerrumCFD commands are
unambiguous:

```console
ferrum initFerrumCase cases/my_case
ferrum gmshToFerrum path/to/mesh.msh -case cases/my_case
ferrum checkFerrumMesh -case cases/my_case
ferrum splitFerrumMeshRegions -case cases/my_case -cellZones
ferrum run -solver incompressibleFluid -case cases/my_case --preflight --planJson target/ferrumRunPlan.json
ferrum run -solver incompressibleFluid -case cases/my_case --runnerDryRun --maxRunnerSteps 2
```

The same naming convention is used by the dedicated binaries:

```console
initFerrumCase
gmshToFerrum
checkFerrumMesh
splitFerrumMeshRegions
ferrumRun
```

Application cases are run only through `ferrumRun`. The combined `ferrum
solve` subcommand retains developer-only scalar-diffusion and Poiseuille
equation benchmarks; it is not an application solver entry point.

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

Archived external comparisons use the external solver's native conventions.
For example, incompressible OpenFOAM solvers commonly store `p` as kinematic
pressure in `m2/s2`. The recorded reports convert those results back to SI
pressure in `Pa` before comparison.

## Laminar Pipe Tutorial

`tutorials/incompressibleFluid/laminarPipe/` contains an independently
authored Ferrum case and the Hagen-Poiseuille analytical reference. Run the
Ferrum case from the repository root:

```console
cargo run --locked -p ferrum-run --bin ferrumRun -- -solver incompressibleFluid -case tutorials/incompressibleFluid/laminarPipe/ferrum/case
```

Stable analytical and external OpenFOAM Foundation 13 comparison results,
including the tested version and protocol, are retained in
`docs/benchmarks/laminar-pipe-poiseuille.md`. External reference cases and
their execution tooling are not distributed with FerrumCFD.

## Solver Selection And Preflight

`ferrumRun` is the solver front door. Its `--preflight` and
`--runnerDryRun` modes do not execute CFD kernels; they read the case and print
the solver-neutral run plan used by current CPU and later GPU solver paths.

```console
ferrumRun -solver incompressibleFluid -case cases/my_case --preflight
ferrumRun -solver incompressibleFluid -case cases/my_case --preflight --planJson target/ferrumRunPlan.json
ferrumRun -solver incompressibleFluid -case cases/my_case --runnerDryRun --maxRunnerSteps 2
```

The solver may instead be selected in `system/controlDict`:

```text
application ferrumRun;
solver incompressibleFluid;
```

Control-dictionary fallback is accepted only with the explicit
`application ferrumRun;` marker, so an unrelated case is not silently adopted
as a Ferrum case. An explicit CLI `-solver` is the deliberate module-selection
override. Plan JSON keeps the raw control values and records the effective
dispatch separately as `dispatch.module` and
`dispatch.source=cli|controlDict`.

Equivalent combined command:

```console
ferrum run -solver incompressibleFluid -case cases/my_case --preflight --planJson target/ferrumRunPlan.json
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
state-materializable CPU f64 buffers. The normal preflight uses Summary mode:
all six supported nonuniform scalar/vector type names are count-checked and
reported, but their payloads are not retained. A real solve uses Full mode and
moves a correctly shaped nonuniform source buffer into runtime storage without
cloning it. Unsupported nonuniform types are rejected by the field parser.
This still does not solve equations or change field values.

The preflight also prepares solver runtime data. It builds compact
owner/neighbour connectivity, patch face ranges, cell centres, face centres,
owner-oriented face-area vectors, positive cell volumes, and descriptors for
fields that passed the solver-state checks. Summary mode leaves every field
payload absent; Full solve mode materializes uniform values and transfers
validated nonuniform buffers. `--planJson` writes a `runtimeData` summary with
array sizes and buffer sizes, but it intentionally does not dump the full
geometry or field arrays into JSON. These runtime arrays are the handoff point
for the CPU/GPU equation kernels.

FerrumCFD also contains the first executable CPU linear algebra foundation:
CSR matrices, matrix-vector products, residual calculation, Jacobi,
Gauss-Seidel, conjugate gradient, preconditioned-CG, BiCGStab, and GAMG. The
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

`--solveScalarDiffusion <field>` is a low-level developer equation path rather
than a normal application command. It reads the selected `volScalarField` from
`0/`, converts supported `boundaryField` entries into diffusion boundary
conditions, assembles a CPU CSR system, and solves it with `cg` or `jacobi`.

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

with `Ux=0` on wall patches and `zeroGradient` elsewhere. It is an internal
benchmark utility, not part of the normal user workflow. `deltaP`, `L`, and
`D` must be supplied explicitly; `mu` may be explicit or read from
`constant/transportProperties`.

The analytical reference is Hagen-Poiseuille:

```text
U_mean = deltaP * D^2 / (32 * mu * L)
deltaP = 32 * mu * L * U_mean / D^2
```

The command reports numerical mean velocity, analytical mean velocity,
relative error, flow rate, reconstructed pressure drop, solver iterations,
residual, and wall-clock seconds. It does not write velocity or pressure fields
back to the case.

`incompressibleFluid` is the first public pressure-velocity module. Its
current executable path is steady laminar SIMPLE and reads the OpenFOAM-like
case dictionaries and fields that a
`simpleFoam` user expects:

- `0/U`
- `0/p`
- `constant/transportProperties`
- `system/fvSchemes`
- `system/fvSolution`
- `constant/polyMesh`

Execution requires `ddtSchemes.default=steadyState`, exactly one `SIMPLE`
section, no `PISO`/`PIMPLE` section, and a laminar transport regime. A present
`momentumTransport` or legacy `turbulenceProperties` dictionary must contain
exactly `simulationType laminar`; RAS/LES cases are rejected.

It builds the first finite-volume flow operators on the runtime mesh:
`phi = U_f . S_f`, `grad(p)`, `div(phi,U)`, and `laplacian(nu,U)`. The SIMPLE
path now reads the supported `system/fvSchemes` subset directly:

- `gradSchemes`: `Gauss linear`, `leastSquares`, or
  `cellLimited Gauss linear k` for `grad(p)` and `grad(U)`
- `divSchemes`: `div(phi,U) Gauss upwind` or
  `div(phi,U) Gauss linearUpwind grad(U)`
- `laplacianSchemes`: `Gauss linear corrected`, `orthogonal`, or `uncorrected`
- `interpolationSchemes`: `linear`
- `snGradSchemes`: `corrected`, `orthogonal`, or `uncorrected`

`Gauss upwind` remains the fully implicit conservative baseline.
`Gauss linearUpwind grad(U)` keeps that upwind matrix and adds the gradient
part as a deferred correction to the right-hand side. This is closer to the
OpenFOAM workflow without hiding artificial field clipping in the solver. For
`leastSquares`, Ferrum builds a deterministic weighted stencil in the mesh's
intrinsic active dimension and applies the same selected reconstruction to
pressure and to every velocity component consumed by `linearUpwind`. Invalid
or rank-deficient geometry is rejected before the one-shot initial field
payloads are consumed. `empty`, paired `wedge`, and `symmetryPlane` constraints
are handled explicitly; no artificial field-magnitude cap is introduced.

For diagnostics, the library function
`reconstruct_laminar_initial_gradients` exposes the cell-centred initial
`grad(p)` and, when the selected convection path consumes it, the three
component rows of `grad(U)`. The call is non-consuming and does not assemble or
advance SIMPLE.

For pipe/axisymmetric benchmarks and general inlet/outlet workflows, the
supported boundary-condition contract is:

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

```console
ferrumRun -solver incompressibleFluid -case tutorials/incompressibleFluid/laminarPipe/ferrum/case --solveTolerance 1e-6 --maxIterations 100 --solveReportJson target/laminar-pipe.json --solveReportMarkdown target/laminar-pipe.md
ferrumRun -solver incompressibleFluid -case tutorials/incompressibleFluid/laminarPipe/ferrum/case --maxSimpleIterations 2 --writeFinalFields target/laminar-pipe-fields/1
```

Solver report schema version 3 records `solver=incompressibleFluid`,
`algorithm=SIMPLE`, and `regime=laminar`. Its additive `timing` object separates
solver total, driver measurement, setup, operator evaluation, momentum
assembly, momentum gradient reconstruction, momentum matrix fill, momentum
solve, pressure-coupling setup, pressure assembly/solve, field correction,
finalization, and remaining solver work. These timings measure the executable
solver only; they never include Cargo compilation.

Wall-force reporting is opt-in and does not add gradient-reconstruction work to
the normal solve path. Supply `--wallForcePatches`,
`--forceReferenceSpeed`, and `--forceReferenceArea` together:

```console
ferrumRun -solver incompressibleFluid -case tutorials/incompressibleFluid/cylinder/ferrum/case --wallForcePatches cylinder --forceReferenceSpeed 0.015 --forceReferenceArea 1e-6 --solveReportJson target/cylinder.json --solveReportMarkdown target/cylinder.md
```

Every selected patch must use `U` type `noSlip` and `p` type
`zeroGradient`. After the solve, Ferrum reconstructs the final velocity
gradient with the active `grad(U)` scheme and integrates wall-face owner
pressure plus the full deviatoric Newtonian viscous traction. Reported forces
use the force exerted by the fluid on the body; face area vectors point outward
from the fluid. The existing `wallForces` summary remains stable, and a
separate `wallForceMethod` line plus additive JSON/Markdown fields record the
traction method, method version, sign convention, face orientation,
zero-gradient owner-pressure treatment, and velocity-gradient scheme.

For spatial-convergence studies and force audits, add
`--wallFaceLoadsCsv <file>`. This fourth option is valid only together with the
complete wall-force triple and remains absent by default:

```console
ferrumRun -solver incompressibleFluid -case tutorials/incompressibleFluid/cylinder/ferrum/case --wallForcePatches cylinder --forceReferenceSpeed 0.015 --forceReferenceArea 1e-6 --wallFaceLoadsCsv target/cylinder-wall-face-loads.csv
```

The versioned CSV writes one deterministic row per selected face in requested
patch order and then increasing global face index. It records the face centre,
outward fluid-area vector, raw kinematic owner pressure, resolved dynamic
pressure, full pressure and viscous tractions, tangential wall shear, and
pressure/viscous/total force-on-body components. Floating-point values use
round-trip text. Method, pressure-reference, sign, units, density, viscosity,
and coefficient-reference provenance are repeated in every row so that the
CSV remains independently auditable. Text fields are RFC-4180 quoted with
spreadsheet-formula protection, and output paths use the same capability-scoped
no-follow replacement contract as the other solver reports.

Pressure-PCG kernel timing is disabled by default. With pressure `PCG`,
`--profilePcg` enables diagnostic timing for total PCG work, selected
preconditioner update/application, matrix-vector products, and vector
operations. The flag is rejected for non-PCG pressure solvers and cannot be
combined with `--profileGamg`. JSON records `options.profilePcg`; when disabled, the PCG
kernel timing and counter fields are zero, the console emits no PCG-kernel
line, and Markdown omits the `Pressure PCG Kernel Profile` section.

Both commands are geometry-independent SIMPLE execution. Analytic formulas,
external comparisons, and geometry-specific field integration are separate
from the normal user workflow. Parameters such as
pipe length, diameter, analytic pressure loss, and sampling patch names are
intentionally rejected by the generic `incompressibleFluid` execution path.

The generic `--linearSolver` value is still accepted, but the laminar SIMPLE
path can also split the linear solver choice and linear controls by equation:

```console
ferrumRun -solver incompressibleFluid -case tutorials/incompressibleFluid/laminarPipe/ferrum/case --momentumLinearSolver bicgstab --pressureLinearSolver pcg --pressurePreconditioner DIC --maxSimpleIterations 20
ferrumRun -solver incompressibleFluid -case tutorials/incompressibleFluid/laminarPipe/ferrum/case --momentumSolveTolerance 1e-7 --pressureSolveTolerance 1e-9 --momentumMaxIterations 300 --pressureMaxIterations 400
ferrumRun -solver incompressibleFluid -case tutorials/incompressibleFluid/laminarPipe/ferrum/case --nNonOrthogonalCorrectors 1 --pRefCell 0 --pRefValue 0
ferrumRun -solver incompressibleFluid -case tutorials/incompressibleFluid/laminarPipe/ferrum/case --simpleConsistent true --maxSimpleIterations 20
ferrumRun -solver incompressibleFluid -case tutorials/incompressibleFluid/laminarPipe/ferrum/case --pressureLinearSolver pcg --profilePcg --solveReportJson target/profile.json --solveReportMarkdown target/profile.md
```

By default, the current SIMPLE implementation reads OpenFOAM-style relaxation factors from
`system/fvSolution`: `relaxationFactors.equations.U`, falling back to
`relaxationFactors.equations.default`, for velocity and
`relaxationFactors.fields.p` for pressure. The CLI flags above are explicit
overrides for experiments. Matching OpenFOAM Foundation 13, momentum-equation
relaxation is disabled only when both `U` and `equations.default` are absent,
while an explicit effective value of `1` still performs the diagonal-dominance
step of equation relaxation. It also reads `solvers.U.tolerance`,
`solvers.p.tolerance`, `solvers.U.relTol`, `solvers.p.relTol`,
`solvers.p.solver PCG`, `solvers.p.preconditioner DIC`,
`SIMPLE.nNonOrthogonalCorrectors`, `SIMPLE.pRefCell`, `SIMPLE.pRefValue`, and
`SIMPLE.consistent`, and optional `maxIter` values from `system/fvSolution`.
`SIMPLE.consistent true` is an explicit SIMPLEC choice, not an automatic
performance default. The frozen Linux acceptance gate found a robust benefit
for the current Plane Channel configuration but rejected the current Pipe
configuration on accuracy and pressure-work limits; select it deliberately and
validate the chosen case rather than assuming a universal speedup.
For symmetric pressure PCG, `DIC` selects Ferrum's face-LDU diagonal
incomplete-Cholesky recurrence and deterministic forward/reverse face sweeps.
`FDIC` uses the same recurrence and result while caching the two
diagonal-scaled face multipliers used by those sweeps. The implementation is
independent safe Rust with explicit symmetry, finite-value, and positive-pivot
gates. `ic0` and `incompleteCholesky` continue to select Ferrum's separate full
CSR IC(0) factorization. `DILU` is rejected until a true nonsymmetric ILU/DILU
preconditioner exists; no diagonal fallback is applied.
OpenFOAM `smoothSolver` on `U` requires a `smoother` entry and executes the
matching CPU `GaussSeidel` or `symGaussSeidel` path. Explicit `bicgstab` remains available for nonsymmetric momentum
experiments. The generic `--solveTolerance` and `--maxIterations` flags remain
broad overrides for both equations. If present, OpenFOAM-style
`SIMPLE.residualControl` entries for `U` and `p` are the primary
early-convergence criteria. Ferrum follows the OpenFOAM Foundation 13 steady
SIMPLE form, where each field entry is one absolute scalar tolerance:

GAMG is selectable for the symmetric pressure equation directly in
`system/fvSolution`:

```text
solvers
{
    p
    {
        solver GAMG;
        smoother symGaussSeidel;
        tolerance 1e-10;
        relTol 0;
        cacheAgglomeration true;
        agglomerator faceAreaPair;
        nCellsInCoarsestLevel 10;
        mergeLevels 1;
        nPreSweeps 0;
        nPostSweeps 2;
        nFinestSweeps 2;
        interpolateCorrection false;
        scaleCorrection true;
        directSolveCoarsest false;
    }
}
```

Ferrum reads `minIter`, `maxIter`, the pre/post sweep level multipliers and
maximums, and all controls shown above. If `agglomerator` is omitted,
OpenFOAM's `faceAreaPair` default is used. `faceAreaPair` consumes the runtime
mesh geometry; `algebraicPair` uses pressure-matrix connection strengths.
`mergeLevels` currently must be `1`. `interpolateCorrection` defaults to
`false`; `true` is an explicit compatibility and experimentation control for
the serial symmetric-CSR GAMG path, not a performance recommendation or a
default. GAMG cannot be chosen for `solvers.U`. There is no PCG fallback. JSON
and Markdown solve reports record the effective controls under
`options.pressureGamg`.

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
`lduMatrix::solver` defaults `1e-6` and `1000`. `BiCGStab`, `CG`, `Jacobi`,
`GaussSeidel`, `symGaussSeidel`, `PCG`, and pressure `GAMG` support finite,
non-negative `relTol`. For every linear solve the authoritative normalized-L1
target is
`max(tolerance, relTol * initialNormalizedResidual)`, and convergence requires
the final normalized residual to be strictly smaller than that target. As in
OpenFOAM Foundation 13, every solver activates the relative criterion only for
`relTol > 1e-20`. The LDU normalisation factor adds `1e-20` to its accumulated
value. `relTol` is not capped at one. GAMG derives conservative internal L2
controls for both normalized-L1 limits and rechecks the strict normalized-L1
criterion before reporting convergence.

The configured `relTol` is a static, user-bounded case control. It remains
active in the accepting steady SIMPLE iteration; Ferrum does not invent a
synthetic `UFinal` or `pFinal` switch and does not force the last accepted
solve to `relTol 0`. `minIter` remains supported only by GAMG; a non-zero
`minIter` on another solver and a `smoothSolver nSweeps` value other than `1`
are rejected instead of being ignored or replaced silently.

Without `--maxSimpleIterations`, Ferrum uses the positive iteration count
derived from `controlDict` (`endTime - startTime` divided by `deltaT`). When
the resulting budget is greater than one, Ferrum defaults to at least
two SIMPLE iterations before convergence can be accepted. `endTime` or
`--maxSimpleIterations` is the maximum budget; all configured
`SIMPLE.residualControl` field tolerances permit an earlier stop. Continuity is
reported as a diagnostic and is not an undocumented extra stopping criterion.
Without `residualControl`, the solver runs to `--maxSimpleIterations`, reports
`converged=false`, and records outer convergence as `not-evaluated` with stop
reason `ConvergenceCriteriaNotConfigured`. Configured criteria that are not met
within the budget report `not-reached`. Hagen-Poiseuille
error and external comparison acceptance are independent validation decisions;
they cannot stop, cap, roll back, or force a flow direction in the generic
solver. `minSimpleIterations` can still be set as a case-level `SIMPLE` value.

Without `--writeFinalFields`, the current `incompressibleFluid` SIMPLE path only reports and does not
write fields back to the case. With `--writeFinalFields <dir>`, Ferrum writes
final `U` and `p` files into the selected OpenFOAM-like time directory. The
internal fields come from the solved cell values, while the dimensions and
`boundaryField` entries are preserved from `0/U` and `0/p`.

The generic solver report records residuals, SIMPLE iterations, wall-clock time,
the active `fvSchemes` subset, resolved absolute and relative linear
tolerances (`momentumLinearRelativeTolerance` and
`pressureLinearRelativeTolerance`),
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
iterations per SIMPLE step, and final linear-solver convergence flags. Each
`linearSolves` history is numerical solver telemetry and is independent of the
optional pressure-PCG kernel timing selected by `--profilePcg`. Each
iteration also contains exactly the `x`, `y`, and `z`
`momentumComponentLinearSolves` plus the one-based
`pressureCorrectionLinearSolves`. Every solve records its iteration count,
convergence flag, initial normalized residual, raw L2 residual, final normalized
residual, effective normalized tolerance, and stop reason (`NotRun`,
`ExactZero`, `AbsoluteTolerance`, `RelativeTolerance`, `MaxIterations`, or
`Breakdown`). These fields are additive: the existing aggregate fields remain
available. JSON additionally records the first non-converged momentum and
pressure solves under `linearNonConvergenceDiagnostics`. Their worst algebraic
row contains `incidentFaces`: `maxAbsFace*` identifies the largest finite
incident flux, while `firstNonFiniteFace*` identifies the lowest global
incident mesh-face index with a NaN or infinite flux; unavailable values are
`null`. The iteration history, CSV, console, JSON, and Markdown outputs
distinguish each field's OpenFOAM-normalized initial residual from its final
linear residual and show the outer `residualControl` state independently.
The top-level JSON `outerConvergence` object records `status`, `configured`,
`evaluated`, `converged`, and `reason`. Ferrum
sets `converged=true` when the configured outer `SIMPLE.residualControl`
criteria are checked and satisfied. Linear convergence remains explicitly
reported through the corresponding `SolverPerformance`-style fields. The pressure
bridge uses an internal momentum-equation object to apply equation relaxation,
retain cell-wise `A` and `H1` diagnostics for `rAU/rAtU`, reconstruct
`HbyA`, compute `phiHbyA` from HbyA with velocity boundary constraints applied,
run `adjustPhi` only when the original pressure field needs an explicit
reference, and leave systems with pressure `fixedValue` or pressure
`inletOutlet` unchanged in every flow direction. In a reference-needing
system, positive adjustable outflow is scaled by the OpenFOAM mass-correction
ratio derived from prescribed inflow and fixed outflow. A literal velocity
`inletOutlet` face is adjustable only while it is outflowing;
`pressureInletOutletVelocity` remains backflow-sensitive for momentum but is
fixed positive outflow for `adjustPhi`. The `empty`, `wedge`, and
`symmetryPlane` velocity constraints always contribute exact zero normal flux
and are never adjusted; a nonzero constraint flux is rejected before mutation.
The bridge then solves an absolute pressure equation, corrects `phi` from the
pressure-equation flux, corrects velocity as `U = HbyA - rAtU grad(p)`, and
carries that corrected surface flux into the next SIMPLE iteration. The
pressure equation now supports
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

The 2D parallel-plate validation uses the same separation. External processing
of stored `U`/`p` applies `meanU = deltaP*H^2/(12*mu*L)` without changing the
simulation. OpenFOAM's kinematic incompressible pressure must be multiplied by
`rho`; Ferrum fields remain SI Pa. The reference `.geo`, case dictionaries,
and SI inputs are under `tutorials/incompressibleFluid/planeChannel/`.

Recorded pipe and plane-channel results are available under `docs/benchmarks`.
They document external reference versions, protocols, and maintainer runs and
are not a required user workflow. Reference cases and benchmark orchestration
are not distributed as part of FerrumCFD.

The solver roadmap in `docs/solver-roadmap.md` records the remaining physics,
case, and backend work. Users run Ferrum cases directly and may choose their
own external references and meshes for independent studies.

It also checks basic `controlDict` consistency: recognized `startFrom`,
`stopAt`, and `writeControl` modes, positive finite `deltaT`, valid
`writeInterval`, and an `endTime` that is not earlier than `startTime` for
`stopAt endTime`.

It also reads material and transport property dictionaries below `constant/`
and `constant/<region>/`. At this stage FerrumCFD checks the structure and
dimension-vector shape, but solver modules will later decide which properties
are required for each physics model.

With `--planJson <file>`, the same solver-neutral plan is also written as JSON.
Only an explicit absolute path outside the current working directory authorizes
a trusted create-new plan destination. Relative and inside-directory plan paths,
and every solver output, remain strictly rooted at the current working directory.
If plan-only creation fails while creating a missing parent or the final leaf,
rollback removes only newly created empty directories, preserves the original
error, and may leave nonempty residue. Existing or concurrent content is never
deleted, and this rollback does not apply to strict case roots.
That file is intended for future run managers, GUI tools, measurement tooling,
and CPU/GPU solver launch code. The text preflight remains the normal
human-readable output.

With `--runnerDryRun`, FerrumCFD expands the current run plan into a capped
runner preview. The preview logs time-step starts, planned stage dispatch such
as `flow.residual` or `interfaces.flux`, backend choice, and planned write
events. It also prints runtime handles derived from `system/ferrumBackends`,
including CPU thread policy, CPU linear-solver availability, and GPU
backend/device metadata. GPU stages are reported as planned dispatch only until
executable GPU solver kernels exist. The same dry-run output also lists the
solver-state fields that would be available to the future runner. A uniform
field is state-materializable because its complete value is carried by the
plan. A validated nonuniform field is transfer-ready instead: its source
buffer is moved once into the full runtime without cloning it. Summary plans
retain the same field descriptors but deliberately omit payloads.
`--maxRunnerSteps <n>` limits the preview length. This does not update fields,
advance physics, or solve equations.

## Interface Model Setup

Users should normally not edit `flipMap` by hand. `flipMap` belongs to the
mesh/faceZone definition and is read from the mesh data. Model intent belongs in
`constant/interfaces`.

`constant/interfaces` is optional: when the file is absent, FerrumCFD treats
the interface configuration as empty. When the file exists, it must contain
exactly one unquoted, ordinary `interfaces { ... }` block; optional `FoamFile`
metadata does not replace that block, and duplicate or quoted `interfaces`
blocks are rejected.

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
- `ferrumRun` is the public preflight/run dispatcher; `--runnerDryRun`
  previews scheduling only. `--solveScalarDiffusion <field>` and
  `--solvePoiseuille` remain low-level `ferrum solve` developer utilities, and
  full general CFD time-loop execution is not implemented yet.
- CPU/GPU backend selection is validated as configuration and not yet
  executable solver behavior.
