# Steady Re=1 cylinder flow

This tutorial is a two-dimensional, steady, laminar, incompressible flow past a
no-slip cylinder. The cylinder diameter is `0.001 m`, the uniform inlet speed
is `0.015 m/s`, and the kinematic viscosity is `1.5e-5 m2/s`, hence
`Re = U D / nu = 1` exactly. The single-cell-thick `frontAndBack` patch is
`empty`. The outlet fixes kinematic pressure to zero; the other pressure and
velocity conditions are zero-gradient where appropriate.

The independently authored `ferrum/case` contains a complete 48-cell
body-fitted mesh. It is deliberately small and deterministic for a bounded
regression, not for production-quality force prediction.

From the repository root, run Ferrum without modifying the source case:

```console
cargo run --locked -p ferrum-run --bin ferrumRun -- -solver incompressibleFluid -case tutorials/incompressibleFluid/cylinder/ferrum/case
```

## Numerical benchmark

The behavioral reference is the official
`tutorials/incompressibleFluid/cylinder` case installed by OpenFOAM Foundation
13 package `20260407`, build `13-441953dfbb42`. The Ferrum dictionaries and
mesh in this repository were newly authored or generated for FerrumCFD. No
OpenFOAM case or runtime tool is distributed here. See the repository's
`THIRD_PARTY_NOTICES.md` for external-reference provenance.

The official 5388-cell run records, after 5000 iterations, `Re=1`, final
`Cd=10.6558580`, `Cl=4.6316142e-11`, final local continuity `2.200139e-10`, and
final global continuity `-1.2322084e-11`. These are provenance targets, not
results produced by either new checked-in case. `comparison.toml` selects drag,
lift, and continuity and records a 15% relative drag tolerance plus `1e-6`
absolute tolerances for the near-zero quantities, justified by the coarse smoke
mesh.

There is no useful closed-form solution for this finite-domain viscous cylinder
problem, so the documented numerical benchmark is used instead. Current
limitations are the coarse mesh, simplified finite outer boundary, steady
laminar model, and absence of an automated force-comparison runner. Runtime is
intentionally not an acceptance value because it depends on hardware.
