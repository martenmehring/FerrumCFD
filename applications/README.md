# Applications

Compiled FerrumCFD applications live here, following the OpenFOAM 13
separation between solver entry points, runtime-selectable modules, and
utilities.

- `solvers/`: public dispatchers including the executable `ferrumRun` crate
  and the planned `ferrumMultiRun` crate;
- `modules/`: runtime-selectable equation/physics families such as
  `incompressibleFluid`;
- `utilities/`: mesh, case, conversion, and post-processing commands;
- `legacy/`: buildable transitional packages awaiting a responsibility-based
  split.

Cargo remains the build system. This directory organization does not reproduce
OpenFOAM's `wmake` or generated `platforms/` tree.
