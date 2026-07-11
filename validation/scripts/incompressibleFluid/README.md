# incompressibleFluid Validation Automation

PowerShell orchestration for the Ferrum/OpenFOAM 13/analytical incompressible
flow validation bundles. All generated artifacts are written below `target/`.

The scripts remain together because they call one another through
`$PSScriptRoot`; the common repository root is resolved three levels above
this directory.
