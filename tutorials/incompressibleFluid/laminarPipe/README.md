# Laminar circular-pipe flow

This tutorial keeps three independent references side by side:

- `ferrum/case` is the independently runnable Ferrum compatibility case;
- `openfoam-v13/case` is the native OpenFOAM Foundation 13 case;
- `analytical/` documents the Hagen-Poiseuille solution.

The supplied case represents steady laminar water flow through a straight
circular pipe. Its default reference uses `D = 0.02 m`, `L = 1 m`, mean
velocity `0.02 m/s`, and dynamic viscosity `0.001002 Pa s`. Ferrum stores
pressure in Pa; the incompressible OpenFOAM case stores kinematic pressure.
The canonical SI inputs are defined once in
`shared/physicalParameters.toml`; `comparison.toml` links that file instead of
duplicating the physical values.

For pipe radius `R`, length `L`, dynamic viscosity `mu`, and pressure loss
`deltaP`:

```text
u(r) = deltaP/(4*mu*L) * (R^2 - r^2)
meanU = deltaP*R^2/(8*mu*L)
deltaP = 32*mu*L*meanU/D^2
```

The supplied reference values give `deltaP = 1.6032 Pa`.

## Run the cases

Run these commands from the repository root.

Ferrum:

```powershell
cargo run --locked -p ferrum-run --bin ferrumRun -- -solver incompressibleFluid -case tutorials\incompressibleFluid\laminarPipe\ferrum\case
```

OpenFOAM Foundation 13 on a compatible Linux installation:

```bash
mkdir -p target
case_dir="$(mktemp -d target/openfoam-laminarPipe.XXXXXX)"
cp -R tutorials/incompressibleFluid/laminarPipe/openfoam-v13/case/. "$case_dir/"
foamRun -solver incompressibleFluid -case "$case_dir"
```

The two cases are independent and may be run, modified, and meshed separately.
No comparison script or shared mesh is required. The analytical reference is
external evidence and never controls SIMPLE convergence.

Recorded numerical results are available in
`docs/benchmarks/laminar-pipe-poiseuille.md`. Optional mesh generation,
OpenFOAM comparison, parameter sweeps, and historical reproduction helpers are
maintainer tools documented in `docs/development/script-policy.md`; they are not
part of the normal user workflow.
