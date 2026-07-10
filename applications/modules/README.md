# Application Modules

Runtime-selectable application modules live here. A module is an equation and
physics family, not a coupling algorithm or flow regime. The first module is
`incompressibleFluid`; it covers both the steady SIMPLE/SIMPLEC and transient
PISO/PIMPLE readiness drivers.

Later module names are finalized against their OpenFOAM 13 comparison modules
when each driver begins. Reusable thermal, species, chemistry, turbulence,
porous, and interface models belong under `src/ferrumModels`, rather than
being duplicated in dispatcher crates.

Porous-media and packed-bed implementation is deliberately deferred until all
seven drivers pass their readiness gates.
