# Tutorials

Tutorials are curated validation bundles grouped by public application module. A
bundle keeps program-specific cases separate while sharing only neutral
geometry and physical inputs:

```text
<module>/<case>/
  shared/
  ferrum/
  openfoam-v13/
  analytical/       # when mathematically valid
  benchmark/        # otherwise, when an external reference is needed
  comparison.toml
  README.md
```

`ferrum/` and `openfoam-v13/` must each be independently runnable. Small,
canonical validation meshes may be versioned inside a source case so a clean
checkout is reproducible. Regenerated mesh variants, solver time directories,
logs, and comparison reports belong below `target/`.

`comparison.toml` records `module`, `readiness_driver`, `algorithm`, and
`regime` separately. This keeps the shared `incompressibleFluid` module
distinct from the steady SIMPLE/SIMPLEC and transient PISO/PIMPLE readiness
drivers.

Current Driver 1 bundles:

- `incompressibleFluid/laminarPipe`: 3D circular-pipe flow with the
  Hagen-Poiseuille analytical reference;
- `incompressibleFluid/planeChannel`: true 2D plane-Poiseuille flow with
  `empty` front/back patches and an analytical reference.

Drivers 1 and 2 both use the `incompressibleFluid` module while validating
different coupling algorithms. The remaining Driver 1 cases and Drivers 2
through 7 are ordered in
`docs/solver-roadmap.md`. Porous-media and packed-bed work is deferred until
all seven drivers pass their readiness gates.
