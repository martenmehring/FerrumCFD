# Applications

Compiled FerrumCFD applications live here, following the OpenFOAM 13
separation between solver entry points, runtime-selectable modules, and
utilities.

- `solvers/`: public solver dispatchers such as the planned `ferrumRun`;
- `modules/`: the seven application drivers and their equation/coupling code;
- `utilities/`: mesh, case, conversion, and post-processing commands;
- `legacy/`: buildable transitional packages awaiting a responsibility-based
  split.

Cargo remains the build system. This directory organization does not reproduce
OpenFOAM's `wmake` or generated `platforms/` tree.
