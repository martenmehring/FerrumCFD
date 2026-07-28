# Laminar plane-channel flow

This tutorial keeps two independently authored local references:

- `ferrum/case` is the runnable Ferrum case;
- `analytical/` documents the plane-Poiseuille solution.

The channel has length `L`, full plate gap `H`, and one thin cell in `z`.
`front` and `back` are `empty` patches, so the solver case represents a true
two-dimensional calculation.

For stationary plates at `y = +/- H/2`:

```text
u(y) = deltaP/(2*mu*L) * ((H/2)^2 - y^2)
meanU = deltaP*H^2/(12*mu*L)
deltaP = 12*mu*L*meanU/H^2
```

The supplied reference values give `deltaP = 0.6012 Pa` and
`meanU = 0.02 m/s`.

## Run the case

Run this command from the repository root.

```console
cargo run --locked -p ferrum-run --bin ferrumRun -- -solver incompressibleFluid -case tutorials/incompressibleFluid/planeChannel/ferrum/case
```

The Ferrum case reads neither `shared/physicalParameters.toml` nor
`comparison.toml` at runtime. Those files record neutral reference metadata;
users do not need a combined runner or generated shared mesh to execute the
case.

Recorded Ferrum, OpenFOAM Foundation 13, and analytical results, including the
external version and protocol, are listed in
`docs/benchmarks/laminar-plane-channel.md`. External reference cases and
benchmark orchestration are not distributed with FerrumCFD.
