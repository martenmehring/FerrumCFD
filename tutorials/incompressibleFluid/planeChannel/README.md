# Laminar plane-channel flow

This tutorial keeps three independent references side by side:

- `ferrum/case` is the independently runnable Ferrum compatibility case;
- `openfoam-v13/case` is the native OpenFOAM Foundation 13 case;
- `analytical/` documents the plane-Poiseuille solution.

The channel has length `L`, full plate gap `H`, and one thin cell in `z`.
`front` and `back` are `empty` patches, so both solver cases represent a true
two-dimensional calculation.

For stationary plates at `y = +/- H/2`:

```text
u(y) = deltaP/(2*mu*L) * ((H/2)^2 - y^2)
meanU = deltaP*H^2/(12*mu*L)
deltaP = 12*mu*L*meanU/H^2
```

The supplied reference values give `deltaP = 0.6012 Pa` and
`meanU = 0.02 m/s`.

## Run the cases

Run these commands from the repository root.

Ferrum:

```powershell
cargo run --locked -p ferrum-run --bin ferrumRun -- -solver incompressibleFluid -case tutorials\incompressibleFluid\planeChannel\ferrum\case
```

OpenFOAM Foundation 13 on a compatible Linux installation:

```bash
mkdir -p target
case_dir="$(mktemp -d target/openfoam-planeChannel.XXXXXX)"
cp -R tutorials/incompressibleFluid/planeChannel/openfoam-v13/case/. "$case_dir/"
foamRun -solver incompressibleFluid -case "$case_dir"
```

The cases are independent. Neither reads `shared/physicalParameters.toml` or
`comparison.toml` at runtime. Those files record neutral reference metadata and
may be useful to maintainers, but users do not need a combined runner or a
generated shared mesh to execute either case.

Recorded Ferrum, OpenFOAM, and analytical results are listed in
`docs/benchmarks/laminar-plane-channel.md`. Optional case-generation and
validation helpers remain developer tools under `validation/scripts/`.
