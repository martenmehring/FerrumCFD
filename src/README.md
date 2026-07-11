# Reusable Rust Implementation

Reusable mesh, field, finite-volume, linear/nonlinear solver, I/O, and physical
model libraries live below `src/`. Cargo packages remain independently
testable even though the top-level responsibilities mirror OpenFOAM 13.

`ferrumMesh` is the current combined foundation. It will be split only when
module boundaries are backed by stable APIs and parity tests.
