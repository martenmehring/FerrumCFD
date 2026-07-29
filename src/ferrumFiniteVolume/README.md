# ferrumFiniteVolume

Solver-independent finite-volume operators and post-processing. The first
accepted leaf owns boundary-force integration for stationary no-slip walls,
including pressure-gauge handling, viscous traction, and Cd/Cl. The first API
is explicitly limited to `zeroGradient` pressure on the selected wall.

Field interpolation, gradients, divergence, Laplacian discretization, and
equation assembly move here from the transitional combined foundation only in
separate parity-preserving architecture packages.
