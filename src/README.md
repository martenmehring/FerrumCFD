# Reusable Rust Implementation

Reusable mesh, field, finite-volume, linear/nonlinear solver, I/O, and physical
model libraries live below `src/`. Cargo packages remain independently
testable even though the top-level responsibilities mirror OpenFOAM 13.

`ferrumMesh` remains the transitional combined solver foundation. The first
active split is the tested boundary-force API in `ferrumFiniteVolume`; further
operators move only when their module boundaries are backed by stable APIs and
parity tests.
