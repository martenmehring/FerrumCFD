# OpenFOAM 13 Plane-Channel Case

This directory contains the independent OpenFOAM Foundation 13 sibling of the
Ferrum plane-channel case. It uses kinematic pressure and the native
`incompressibleFluid` solver module. The `front` and `back` mesh patches are
`empty`, making this a true two-dimensional calculation.

Run it from a disposable copy below `target/`. The source mesh is generated
from `../shared/geometry/plane_channel.geo` and is versioned with the case so a
clean checkout has a reproducible OpenFOAM input without invoking Ferrum.

This native case is independently runnable. Shared SI values live in
`../shared/physicalParameters.toml` and comparison targets live in
`../comparison.toml`; neither TOML file is an OpenFOAM runtime dictionary.
