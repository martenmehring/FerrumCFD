# OpenFOAM 13 Plane-Channel Case

This directory contains the independent OpenFOAM Foundation 13 sibling of the
Ferrum plane-channel case. It uses kinematic pressure and the native
`incompressibleFluid` solver module. The `front` and `back` mesh patches are
`empty`, making this a true two-dimensional calculation.

Run it from a disposable copy below `target/`. The source mesh is generated
from `../shared/geometry/plane_channel.geo` and is versioned with the case so a
clean checkout has a reproducible OpenFOAM input without invoking Ferrum.

This is a native, independently runnable OpenFOAM case. Canonical SI values
live in `../shared/physicalParameters.toml`, which is comparison-only metadata
and is not read by OpenFOAM; comparison targets live in `../comparison.toml`.
No Ferrum conversion or combined runner is required.

Provenance classification: the geometry, mesh, dictionaries, and fields in
this bundle were newly authored or generated for FerrumCFD comparison while
using the documented OpenFOAM file formats and the stated reference parameters.
Required OpenFOAM names and generated headers remain in this external sibling
case. See the repository's `THIRD_PARTY_NOTICES.md`.
