# Hagen-Poiseuille Reference

This reference applies to steady, incompressible, fully developed, laminar flow
of a Newtonian fluid through a straight circular pipe with no slip at the wall.

For pipe radius `R`, length `L`, dynamic viscosity `mu`, and pressure loss
`deltaP`:

```text
u(r) = deltaP / (4 * mu * L) * (R^2 - r^2)
meanU = deltaP * R^2 / (8 * mu * L)
deltaP = 32 * mu * L * meanU / D^2
Q = pi * R^4 * deltaP / (8 * mu * L)
```

The canonical SI inputs are recorded in `../shared/physicalParameters.toml`.
The current benchmark scripts independently consume the detailed analytical
dictionary `pipeBenchmark`; drift tests keep that runtime input aligned with
the shared comparison metadata. For `D=0.02 m`, `L=1 m`,
`mu=0.001002 Pa s`, and `meanU=0.02 m/s`, the expected pressure loss is
`1.6032 Pa`.

The analytical error is external validation data. It must never be used as a
generic solver convergence or stopping criterion.
