# ferrumRun

`ferrumRun` is the public single-region solver dispatcher. Its executable
crate lives in this permanent directory. During the staged split it delegates
to the shared implementation still housed in
`applications/legacy/ferrumCli`.

Canonical usage:

```text
ferrumRun -solver incompressibleFluid -case <case>
```

The module name identifies the equation family. Coupling algorithms such as
SIMPLE, SIMPLEC, PISO, and PIMPLE and physical regimes such as laminar, RANS,
or LES are selected by the case. They are not separate executable names.

The current executable kernel accepts only an unambiguous steady configuration:
`ddtSchemes.default=steadyState`, exactly one `SIMPLE` section, and no `PISO`
or `PIMPLE` section. A present `momentumTransport` or legacy
`turbulenceProperties` dictionary must select `simulationType laminar`; RAS/LES
and other configurations fail explicitly until their kernels exist.

The module registry and common solver lifecycle move out of `ferrumCli` after
behavior-parity tests establish the new responsibility boundaries.
