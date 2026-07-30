# Steady Re=1 cylinder flow

This tutorial is a two-dimensional, steady, laminar, incompressible flow past a
no-slip cylinder. The cylinder diameter is `0.001 m`, the uniform inlet speed
is `0.015 m/s`, and the kinematic viscosity is `1.5e-5 m2/s`, hence
`Re = U D / nu = 1` exactly. The single-cell-thick `frontAndBack` patch is
`empty`. The outlet fixes kinematic pressure to zero; the other pressure and
velocity conditions are zero-gradient where appropriate.

The independently authored `ferrum/case` contains a complete 48-cell
body-fitted mesh. It is deliberately small and deterministic for a bounded
regression, not for production-quality force prediction.

## Deterministic mesh family (C2)

C2 defines a pure-Rust, deterministic O-grid generator. It is a Ferrum mesh
capability, not a wrapper around an external mesher. The cylinder diameter
`D` is the geometry scale. Every preset has only the `inlet`, `outlet`,
`cylinder`, and `frontAndBack` boundary patches plus the `fluid` volume zone;
there are no separate `top` or `bottom` patches. For `Coarse` and `Fine`, the
left, upper, and lower outer-boundary segments belong to `inlet`, while only
the right segment belongs to `outlet`. `LegacySmoke` preserves the existing
four-face `inlet` and twelve-face `outlet` outer-boundary split.

| Preset | Circumferential sectors | Radial layers | Cells | Intended use |
| --- | ---: | ---: | ---: | --- |
| `LegacySmoke` | 16 | 3 | 48 | Preserve the bounded checked-in smoke topology |
| `Coarse` | 128 | 42 | 5,376 | First production acceptance mesh |
| `Fine` | 256 | 84 | 21,504 | Refinement and C3 physical-acceptance mesh |

The `Coarse` and `Fine` presets use a `-100D .. +100D` outer domain, an
extrusion depth of `D`, and deterministic exponential radial grading with the
continuous endpoint spacing-density parameter `R = 1000` in
`g(t) = (R^t - 1) / (R - 1)`. This is not a claim that the final discrete cell
is exactly 1000 times wider than the first. Their generated mesh files belong
under `target/` and are not tracked. The small `LegacySmoke` case remains the
fast repository fixture.

The generator also writes neutral Gmsh 2.2 ASCII and reads that output back
through Ferrum's independently authored importer. Acceptance requires exact
point, cell, patch, and topology parity after that round trip. Generating or
running these meshes therefore requires no OpenFOAM runtime, utility, or case
data.

C2 records raw geometry measurements rather than hiding poor cells behind a
quality cap. The Rust acceptance tests must demonstrate, for every preset:

- deterministic ordering and repeated-generation hashes, exact cell counts,
  a closed periodic seam, and the expected patch inventory;
- zero non-finite geometry, zero non-positive face areas, zero non-positive
  cell volumes, and no reported problematic face or cell indices; and
- exact neutral-writer/readback topology and patch parity.

The production `Coarse` and `Fine` presets must additionally demonstrate:

- maximum internal non-orthogonality of at most `50 deg`;
- maximum normalized internal skewness of at most `0.55`; and
- maximum active two-dimensional edge aspect ratio of at most `4.0`.

Those production limits do not apply to the intentionally tiny 48-cell
`LegacySmoke` regression mesh. Its raw values remain observable and its safety,
topology, and determinism contracts still apply; changing its topology to meet
production-quality limits would defeat its compatibility purpose.

The accepted Rust readback path records these raw maxima:

| Preset | Non-orthogonality | Normalized skewness | Active edge aspect |
| --- | ---: | ---: | ---: |
| `LegacySmoke` | 48.099448 deg | 0.490957 | 13.940787 |
| `Coarse` | 43.608220 deg | 0.499852 | 3.626150 |
| `Fine` | 44.296535 deg | 0.499926 | 3.477364 |

All applicable C2 tests pass. The larger Legacy aspect value is retained as
raw evidence rather than hidden by a cap. C3 supplies the automated force,
continuity, convergence, refinement, and determinism acceptance. C4 now adds
the accepted same-Linux comparison with OpenFOAM Foundation 13, documented in
[Cylinder same-Linux parity](../../../docs/benchmarks/cylinder-linux-parity.md).

From the repository root, run Ferrum without modifying the source case:

```console
cargo run --locked -p ferrum-run --bin ferrumRun -- -solver incompressibleFluid -case tutorials/incompressibleFluid/cylinder/ferrum/case
```

## C3 physical-acceptance gate

The solver report can optionally integrate the final stationary no-slip wall
force without changing the equations or the SIMPLE stopping rules. For the
Cylinder, select patch `cylinder`, reference speed `0.015 m/s`, and projected
area `D x depth = 1e-6 m2`. Ferrum reports pressure, viscous, and total force
components plus `Cd` and `Cl` in the console and optional JSON/Markdown
reports. The current incompressible solver stores kinematic pressure; density
and dynamic viscosity therefore come from the resolved case properties.

C3 also reports continuity with the same volume normalization used by the
documented Foundation reference:

```text
local  = (sum(abs(netCellFlux)) / totalCellVolume) * deltaT
global = (sum(netCellFlux)      / totalCellVolume) * deltaT
```

The existing raw L2, maximum, absolute-sum, and global-sum diagnostics remain
available and unchanged. They are not substituted for the normalized
reference quantities.

The complete C3 release gate generates fresh `Coarse` and `Fine` meshes in a
temporary directory, repeats `Coarse`, applies the C2 quality limits to the
same meshes that are solved, requires converged outer and linear solves, and
checks continuity, `Cd`, `Cl`, refinement drift, and deterministic final
fields. It bundles no external case and invokes no external CFD utility:

```console
cargo +1.94.0 test --locked --release -p ferrum-run --test cylinder_c3_acceptance -- --ignored --nocapture
```

This explicit ignored test is a release gate rather than part of the fast
default test suite. It passed on 2026-07-29 with the following evidence:

| Mesh/run | SIMPLE iterations | Local continuity | Global continuity | Cd | Cl |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Coarse` A | 1,181 | 7.518411e-13 | -2.859259e-14 | 11.50464804 | 1.074589e-8 |
| `Coarse` B | 1,181 | bit-identical to A | bit-identical to A | bit-identical to A | bit-identical to A |
| `Fine` | 3,583 | <= 1e-6 gate passed | <= 1e-6 gate passed | 11.53648 | 9.706568e-9 |

The Coarse/Fine drag drift is `0.275907%`, below the `5%` gate. The two Coarse
runs also produced bit-identical final `U` and `p`. These are physical and
deterministic C3 results, not a speed comparison. The separate C4 same-Linux
comparison against Foundation 13 is recorded in
[Cylinder same-Linux parity](../../../docs/benchmarks/cylinder-linux-parity.md).

## C4 same-Linux comparison

C4 uses the same generated official 5,388-cell `polyMesh`, physical fields,
schemes, linear solvers, relaxation factors, CPU affinity, one-thread Linux
environment, and external elapsed-time metric for both engines. The
engine-specific outer-control spelling is mapped to one equivalent predictor /
pressure-corrector cycle rather than claimed byte-identical.

On exact commit `3d84b33f2406b143e6349ea6a9e9438c029a324f`, the accepted
Fixed-1,000 track measured medians of `85.170 s` for Ferrum and `31.335 s` for
OpenFOAM, with a paired Ferrum/OpenFOAM median ratio of `2.752571`. The separate
`U,p = 1e-5` time-to-accuracy track measured `129.680 s` versus `38.735 s` and
a paired ratio of `3.335669`; Ferrum reached the threshold in 986 outer steps
versus 995 for OpenFOAM. The full-field relative L2 differences were
`0.525295%` for `U` and `0.926074%` for `p`, both below the `2%` gate.

Residual stopping remains optional: the TTA track supplies `residualControl`,
while the Fixed-1,000 track omits it and proves exact execution of the longer
budget. These results identify Cylinder as a remaining performance hotspot and
do not support a general all-case speedup claim.

## Numerical benchmark

The behavioral reference is the official
`tutorials/incompressibleFluid/cylinder` case installed by OpenFOAM Foundation
13 package `20260407`, build `13-441953dfbb42`. The Ferrum dictionaries and
mesh in this repository were newly authored or generated for FerrumCFD. No
OpenFOAM case or runtime tool is distributed here. See the repository's
`THIRD_PARTY_NOTICES.md` for external-reference provenance.

The official 5388-cell run records, after 5000 iterations, `Re=1`, final
`Cd=10.6558580`, `Cl=4.6316142e-11`, final local continuity `2.200139e-10`, and
final global continuity `-1.2322084e-11`. These are provenance targets, not
results produced by either new checked-in case. `comparison.toml` selects drag,
lift, and continuity and records a 15% relative drag tolerance plus `1e-6`
absolute tolerances for the near-zero quantities, justified by the coarse smoke
mesh.

There is no useful closed-form solution for this finite-domain viscous cylinder
problem, so the documented numerical benchmark is used instead. Current
limitations are the finite outer boundary and steady laminar model. Runtime is
not a C3 acceptance value because it depends on hardware; C4 records it under
the controlled same-Linux protocol linked above.
