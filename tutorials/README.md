# Tutorials

Tutorials are grouped by public application module. Their user-facing purpose
is to provide independently authored Ferrum cases beside a mathematical or
documented external reference:

```text
<module>/<case>/
  ferrum/case/
  analytical/       # when a useful closed form exists
  README.md
```

`ferrum/case` is independently runnable with the Rust tools shipped by this
repository. Small canonical Ferrum meshes may be versioned with the tutorial;
generated time directories and logs belong below `target/`. External solver
cases and execution scripts are not bundled.

`shared/`, `comparison.toml`, and benchmark metadata are optional
case-specific aids. A Ferrum solver case must not depend on them at runtime.

When present, `comparison.toml` records `module`, `readiness_driver`,
`algorithm`, and `regime` separately. This keeps the shared
`incompressibleFluid` module distinct from the steady SIMPLE/SIMPLEC and
transient PISO/PIMPLE readiness drivers.

Stable recorded results live under `docs/benchmarks/`; generated run output
belongs below `target/`.

Current Driver 1 bundles:

- `incompressibleFluid/laminarPipe`: 3D circular-pipe flow with the
  Hagen-Poiseuille analytical reference;
- `incompressibleFluid/planeChannel`: true 2D plane-Poiseuille flow with
  `empty` front/back patches and an analytical reference;
- `incompressibleFluid/cylinder`: steady Re=1 flow around a cylinder, with an
  OpenFOAM Foundation 13 documented numerical benchmark.

Drivers 1 and 2 both use the `incompressibleFluid` module while validating
different coupling algorithms. The remaining Driver 1 cases and Drivers 2
through 7 are ordered in
`docs/solver-roadmap.md`. Porous-media and packed-bed work is deferred until
all seven drivers pass their readiness gates.
