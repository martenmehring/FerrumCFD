# Steady Re=1 cylinder flow

This tutorial is a two-dimensional, steady, laminar, incompressible flow past a
no-slip cylinder. The cylinder diameter is `0.001 m`, the uniform inlet speed
is `0.015 m/s`, and the kinematic viscosity is `1.5e-5 m2/s`, hence
`Re = U D / nu = 1` exactly. The single-cell-thick `frontAndBack` patch is
`empty`. The outlet fixes kinematic pressure to zero; the other pressure and
velocity conditions are zero-gradient where appropriate.

The independently authored `ferrum/case` and `openfoam-v13/case` directories
each contain a complete, physically separate copy of the same 48-cell
body-fitted mesh. They share no runtime files. The mesh is deliberately small
and deterministic for a bounded regression, not for production-quality force
prediction.

From the repository root, run Ferrum without modifying the source case:

```bash
cargo run --locked -p ferrum-run --bin ferrumRun -- -solver incompressibleFluid -case tutorials/incompressibleFluid/cylinder/ferrum/case
```

Run the native OpenFOAM Foundation 13 sibling in a disposable copy:

```bash
mkdir -p target
case_dir="$(mktemp -d target/openfoam-cylinder.XXXXXX)"
cp -R tutorials/incompressibleFluid/cylinder/openfoam-v13/case/. "$case_dir/"
foamRun -solver incompressibleFluid -case "$case_dir"
```

## Numerical benchmark

The behavioral reference is the official
`tutorials/incompressibleFluid/cylinder` case installed by OpenFOAM Foundation
13 package `20260407`, build `13-441953dfbb42`. The dictionaries and mesh in
this repository were newly authored or generated for FerrumCFD while using the
documented OpenFOAM file formats and the stated external reference parameters.
OpenFOAM names and generated format headers are retained only where the sibling
case requires them. See the repository's `THIRD_PARTY_NOTICES.md`.

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
