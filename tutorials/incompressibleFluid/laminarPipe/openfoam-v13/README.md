# OpenFOAM 13 Laminar-Pipe Case

This directory is an independent OpenFOAM Foundation 13 reference case. It
uses native `FoamFile` dictionaries, kinematic pressure, `physicalProperties`,
`momentumTransport`, and the `incompressibleFluid` solver module.

The checked-in mesh and inlet profile form one immutable validation input. Run
the case from a disposable copy below `target/`; do not write OpenFOAM time
directories into this source template. The repository benchmark runner copies
this case unchanged by default, verifies `WM_PROJECT_VERSION=13`, and launches
`foamRun -solver incompressibleFluid`.

Generated mesh studies may explicitly pass `-FerrumOverlayCaseRoot` to the
runner. That compatibility path overlays only the selected mesh, initial
fields, and viscosity onto the native OpenFOAM 13 configuration; it is never
used by the canonical default run.

Ferrum does not parse or execute this case. Shared SI values and comparison
targets live in `../comparison.toml`.

Provenance classification: the geometry, mesh, dictionaries, and fields in
this bundle were independently authored or generated for FerrumCFD comparison;
they were not copied from an OpenFOAM tutorial or implementation source.
