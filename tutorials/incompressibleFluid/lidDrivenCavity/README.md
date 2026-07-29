# Lid-driven cavity at Re=100

This independently authored Ferrum case is the first executable closed-pressure
Driver 1 proof. A unit square is driven by a `1 m/s` top lid with stationary
no-slip side and bottom walls. With `nu = 0.01 m2/s`, the Reynolds number based
on lid speed and cavity width is `100`.

All physical pressure boundaries start as `zeroGradient`; there is no fixed
pressure outlet. Ferrum's pressure constraint converts the fixed-velocity wall
faces to the required flux-matching pressure gradients during each correction.
`SIMPLE.pRefCell = 0` and the deliberately nonzero
`SIMPLE.pRefValue = 3` therefore make the closed-system anchor observable in
the written field. One corrected non-orthogonal pass exercises
the same pressure-coupling path used by larger meshes. The front and back
planes are `empty`.

The checked-in `2 x 2 x 1` hexahedral mesh is deliberately tiny. It is suitable
for deterministic parser, boundary-condition, pressure-reference, and solver
wiring tests only. It does **not** resolve the primary vortex and is not the
physical Driver 1 acceptance mesh. That later acceptance uses a refined,
grid-independent mesh and published centerline/vortex observables.

Run the smoke case from the repository root:

```console
cargo run --locked -p ferrum-run --bin ferrumRun -- -solver incompressibleFluid -case tutorials/incompressibleFluid/lidDrivenCavity/ferrum/case --maxSimpleIterations 2
```

The mesh source is Gmsh 2.2 ASCII written for FerrumCFD. It can be regenerated
using only the Rust workspace utility:

```console
cargo run --locked -p ferrum-cli --bin gmshToFerrum -- tutorials/incompressibleFluid/lidDrivenCavity/shared/geometry/lid_cavity_2x2x1.msh -case tutorials/incompressibleFluid/lidDrivenCavity/ferrum/case -emptyPatch frontAndBack -patchType walls=wall -patchType lid=wall
```

No external solver case or source is distributed. The future physical
acceptance compares independently generated Ferrum results with the published
Re=100 benchmark identified in `comparison.toml`.
