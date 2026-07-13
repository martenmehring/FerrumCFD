# Plane-channel benchmark

This validation bundle keeps three independent implementations or references:

- `ferrum/case` for `ferrumRun -solver incompressibleFluid`;
- `openfoam-v13/case` for OpenFOAM Foundation 13
  `foamRun -solver incompressibleFluid`;
- `analytical/planeChannelBenchmark` for the plane-Poiseuille solution.

The geometry is a channel of length `L`, full plate gap `H`, and one thin cell
in `z`. The `front` and `back` physical surfaces are imported as OpenFOAM
`empty` patches, so the solver performs a true 2D calculation.

For plates at `y = +/- H/2`:

```text
u(y) = deltaP/(2*mu*L) * ((H/2)^2 - y^2)
meanU = deltaP*H^2/(12*mu*L)
deltaP = 12*mu*L*meanU/H^2
```

The values in `analytical/planeChannelBenchmark` give `deltaP = 0.6012 Pa`.
Canonical SI inputs are recorded in `shared/physicalParameters.toml`; that file
is comparison metadata only and is not a runtime dictionary. `comparison.toml`
links to it and records comparison targets without duplicating the physics.
Analytic errors and OpenFOAM comparisons belong in external
JSON/Markdown reports and must not enter the generic SIMPLE convergence
decision.

The Ferrum and OpenFOAM cases are native, independently runnable cases; neither
reads the shared TOML at runtime. Gmsh-source validation, analytical-dictionary
semantics, and strict lexical OpenFOAM hardening are explicitly deferred to
`F-LAYOUT-PARAMS-PLANECHANNEL-GMSH`,
`F-LAYOUT-PARAMS-PLANECHANNEL-ANALYTICAL`, and
`F-LAYOUT-PARAMS-PLANECHANNEL-LEXICAL`.

Generate the shared Gmsh mesh and prepare the Ferrum case with:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File validation\scripts\incompressibleFluid\prepare_plane_channel_case.ps1 -GmshExe "C:\path\to\gmsh.exe" -Force
```

The script deliberately requires the Gmsh executable path. It writes generated
files only below `target`, imports `front` and `back` as `empty`, copies the
current Ferrum compatibility dictionaries, and runs `checkFerrumMesh`.

Post-process Ferrum fields with:

```powershell
ferrumPlaneChannelBenchmark -case tutorials\incompressibleFluid\planeChannel\ferrum\case --fields target\benchmarks\plane_channel\ferrum_fields\1 --pressureDrop 0.6012 --mu 0.001002 --length 1 --gap 0.02 --depth 0.001 --outJson target\benchmarks\plane_channel\ferrum_analytic.json --outMarkdown target\benchmarks\plane_channel\ferrum_analytic.md
```

For OpenFOAM incompressible output, pass `--pressureScale 998.2` to convert
kinematic pressure to Pa before named-patch pressure sampling. The first shared
mesh result is recorded in `docs/benchmarks/laminar-plane-channel.md`.
