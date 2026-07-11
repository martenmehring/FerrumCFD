# Solver Dispatchers

This directory contains the permanent public solver entry-point boundaries:

- `ferrumRun`: one case, one region, one runtime-selectable module, and the
  shared serial/threaded/distributed CPU plus GPU backend ladder;
- `ferrumMultiRun`: one coupled case with multiple regions and one module per
  region, reusing the same compute backends.

`ferrumRun` is compiled from its permanent dispatcher crate. Its behavior
still delegates to `../legacy/ferrumCli` during the staged split.
Developer-only equation benchmarks remain under the combined `ferrum` utility
while application execution is exclusively routed through `ferrumRun`.
