# Solver Dispatchers

This directory contains the permanent public solver entry-point boundaries:

- `ferrumRun`: one case, one region, one runtime-selectable module;
- `ferrumMultiRun`: one coupled case with multiple regions and one module per
  region.

`ferrumRun` is compiled from its permanent dispatcher crate. Its behavior
still delegates to `../legacy/ferrumCli` during the staged split.
`ferrumSolver --solveLaminarSimple` remains a temporary compatibility and
benchmark interface.
