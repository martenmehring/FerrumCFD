# Plane-Poiseuille Reference

This reference applies to steady, incompressible, fully developed, laminar flow
of a Newtonian fluid between two stationary parallel plates separated by the
full gap `H`.

With `y=0` at the channel centreline:

```text
u(y) = deltaP / (2 * mu * L) * ((H / 2)^2 - y^2)
meanU = deltaP * H^2 / (12 * mu * L)
deltaP = 12 * mu * L * meanU / H^2
```

The canonical SI inputs are recorded in `../comparison.toml` and
`planeChannelBenchmark`. For `H=0.02 m`, `L=1 m`, `mu=0.001002 Pa s`, and
`meanU=0.02 m/s`, the expected pressure loss is `0.6012 Pa`.

The analytical error is external validation data and must not affect the
generic SIMPLE convergence decision.
