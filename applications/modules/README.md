# Application Modules

Application-driver modules will be implemented here in the fixed order from
`docs/solver-roadmap.md`:

1. steady incompressible SIMPLE/SIMPLEC;
2. transient incompressible PISO/PIMPLE;
3. low-Mach thermal/buoyant;
4. low-Mach reacting flow;
5. compressible flow;
6. multi-region conjugate/reacting;
7. immiscible two-phase VOF.

Drivers compose reusable code from `src/`. Porous-media and packed-bed modules
are deliberately deferred until all seven drivers pass their readiness gates.
