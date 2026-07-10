# Solver Dispatchers

This directory is reserved for public solver entry points. The first target is
`ferrumRun`, which selects one of the seven application drivers without
duplicating physical-model or finite-volume implementations.

The existing `ferrumSolver` binary remains under `../legacy/ferrumCli` until
`FerrumFile v1`, the driver registry, and behavior-parity tests are complete.
