# Laminar circular-pipe flow

This tutorial keeps two independently authored local references:

- `ferrum/case` is the runnable Ferrum case;
- `analytical/` documents the Hagen-Poiseuille solution.

The supplied case represents steady laminar water flow through a straight
circular pipe. Its default reference uses `D = 0.02 m`, `L = 1 m`, mean
velocity `0.02 m/s`, and dynamic viscosity `0.001002 Pa s`. Ferrum stores
pressure in Pa. The canonical SI inputs are defined once in
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

## Run the case

Run this command from the repository root.

```console
cargo run --locked -p ferrum-run --bin ferrumRun -- -solver incompressibleFluid -case tutorials/incompressibleFluid/laminarPipe/ferrum/case
```

No comparison script or external solver installation is required to run the
Ferrum case. The analytical reference is evidence and never controls SIMPLE
convergence.

Recorded analytical and OpenFOAM Foundation 13 comparison results, including
their exact external version and protocol, are available in
`docs/benchmarks/laminar-pipe-poiseuille.md`. The external reference case and
benchmark orchestration are not distributed with FerrumCFD.
