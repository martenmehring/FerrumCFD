# incompressibleFluid

This is the permanent application-module boundary for incompressible flow. It
will serve both steady and transient cases and select SIMPLE/SIMPLEC or
PISO/PIMPLE from case configuration. Laminar flow is the first implemented
regime; turbulence models are later compositions, not new public solver names.

The current executable implementation remains inside `src/ferrumMesh` and
`applications/legacy/ferrumCli` until the module lifecycle and parity tests are
complete.
