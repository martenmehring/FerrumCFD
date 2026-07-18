# Scalar CPU Performance Foundation

Date: 2026-07-16
Latest measurement: 2026-07-17

This diagnostic establishes a release-only Ferrum performance profile for the
existing `laminarPipe` and `planeChannel` regression cases. Cargo build time is
recorded separately. Solver timings come from a prebuilt
`target/release/ferrumRun.exe`; compilation is never included.

## Measurement Policy

Two profiles are intentionally separate:

- `fixed` forces exactly 10 pipe and 500 channel SIMPLE iterations, so code
  changes are compared at identical work;
- `converged` copies each tutorial into `target/`, overlays external
  `residualControl` dictionaries, and measures time until the generic solver
  satisfies those criteria.

The overlays live under `validation/profiles/`. They are not solver defaults,
fallbacks, or analytic stopping criteria.

## First Hot-path Result

The initial profile showed that a convection-divergence diagnostic was
repeated inside every SIMPLE iteration although it was only needed in the final
report. Moving it to finalization changed no equation, boundary condition,
relaxation factor, linear-solver control, or convergence criterion.

| Case | Cells | SIMPLE | Before [s] | After [s] | Speedup | Reduction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Laminar pipe | 4608 | 10 | 64.7464 | 16.6408 | 3.89x | 74.30% |
| Plane channel | 2000 | 500 | 1134.8209 | 354.9458 | 3.20x | 68.72% |

These are one-run diagnostics used to select the next hot path, not five-run
cross-version medians.

## CSR Sparsity Reuse

The momentum CSR topology is now built once from the runtime mesh. All three
component matrices and all SIMPLE iterations share immutable row offsets and
column indices; only coefficients and right-hand sides are rebuilt. Equation
relaxation updates diagonal values in place instead of reconstructing CSR rows.

Finer timing then showed that the former `momentumAssemblySeconds` label was
mostly gradient reconstruction. In the CSR-only diagnostic, actual matrix fill
was only `0.0322 s` of `24.4961 s` for the pipe and `0.6677 s` of `474.2500 s`
for the channel. CSR reuse is therefore useful infrastructure, but no material
end-to-end speedup is attributed to it alone.

## Gradient Geometry Cache

Gauss-linear gradients previously recomputed mesh-only interpolation geometry
and eagerly formatted error-context strings for every face on every call.
Ferrum now computes internal-face owner weights, boundary normal distances, and
inverse cell volumes once during solver setup. Runtime arithmetic retains the
same face order and checks non-finite results, while diagnostic strings are
created only on an actual error path.

Back-to-back one-run diagnostics on the same host were:

| Case | CSR-only [s] | With gradient cache [s] | Observed speedup |
| --- | ---: | ---: | ---: |
| Laminar pipe, 10 SIMPLE | 24.4961 | 2.1789 | 11.24x |
| Plane channel, 500 SIMPLE | 474.2500 | 11.5608 | 41.02x |

## Pre-pressure Stable Fixed-work Profile

This checkpoint used one warmup and the median of five measured release runs
after momentum CSR and gradient caching, but before pressure-path reuse. Every
measured run executed the same forced iteration count.

| Case | SIMPLE | Median total [s] | Momentum gradients [s] | Momentum matrix fill [s] | Momentum solve [s] | Pressure solve [s] |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Laminar pipe | 10 | 1.3847 | 0.0112 | 0.0264 | 0.1363 | 0.4960 |
| Plane channel | 500 | 8.8470 | 0.2888 | 0.5451 | 1.2649 | 4.9861 |

Compared with the first recorded fixed-work diagnostic before any of the
hot-path changes in this report, that checkpoint was:

| Case | Original basis [s] | Pre-pressure checkpoint [s] | Speedup | Reduction |
| --- | ---: | ---: | ---: | ---: |
| Laminar pipe, 10 SIMPLE | 64.7464 | 1.3847 | 46.76x | 97.86% |
| Plane channel, 500 SIMPLE | 1134.8209 | 8.8470 | 128.27x | 99.22% |

The original basis values are single-run diagnostics, while the checkpoint
values are five-run medians. The table therefore records the total observed
project improvement, not a statistically controlled per-change A/B result.

For every subsequent optimization, this report records two comparisons:

1. the five-run release median immediately before versus immediately after the
   isolated change, which is the individual optimization result;
2. the original basis above versus the newest combined five-run median, which
   is the cumulative project result.

Both comparisons use the same fixed SIMPLE counts and must preserve all
numerical regression observables.

Across all five runs per case, SIMPLE counts, stop reasons, linear iteration
counts, final continuity, final momentum and pressure residuals, and velocity
and pressure field summaries were identical. The same numerical values are
also bit-identical to the pre-cache fixed-work report.

The fixed pipe has no `residualControl`, so its outer status is
`not-evaluated`, not a numerical failure. The channel has configured criteria
but does not reach them within the forced 500 iterations, so its status is
`not-reached`.

## Pressure Matrix Reuse

The next controlled baseline was recorded immediately before this change. The
pressure matrix then switched from rebuilding `BTreeMap` rows and CSR storage
on every pressure correction to one mesh-topology pattern plus reusable matrix
value and right-hand-side buffers. Pressure reference elimination now updates
the existing values in place and retains the same equation.

| Case | Before total [s] | After total [s] | Total ratio | Assembly before [s] | Assembly after [s] | Assembly speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Laminar pipe | 1.8243 | 1.5336 | 1.19x | 0.04846 | 0.00937 | 5.17x |
| Plane channel | 7.8778 | 9.1923 | 0.86x | 0.72908 | 0.20954 | 3.48x |

All numerical observables and iteration counts were exactly identical. The
targeted assembly improved in both cases. The channel end-to-end ratio is not
accepted as a solver regression or speedup measurement because every unchanged
major phase was about 20% slower during the after run. A repeat was affected by
even higher system load. This result is retained rather than replaced with a
more favorable sample.

## Reusable PCG And IC(0)

The pressure PCG path now retains its work vectors across pressure solves. The
IC(0) symbolic lower-triangle structure and dependency graph are built once
from CSR topology; each equation refactors only numerical values into retained
storage. Forward and backward substitutions write into reusable buffers.

Against the immediately preceding pressure-reuse median:

| Case | Before total [s] | After total [s] | Total ratio | Pressure solve before [s] | Pressure solve after [s] | Pressure-solve speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Laminar pipe | 1.5336 | 1.5897 | 0.96x | 0.5163 | 0.4812 | 1.07x |
| Plane channel | 9.1923 | 6.9013 | 1.33x | 5.5903 | 3.6967 | 1.51x |

The targeted pressure-solve phase improved in both cases. Host-load variance is
again visible in the pipe total, so the project-level acceptance uses the
baseline immediately before both pressure changes:

| Case | Before both pressure changes [s] | Current combined [s] | Combined speedup | Combined reduction |
| --- | ---: | ---: | ---: | ---: |
| Laminar pipe | 1.8243 | 1.5897 | 1.15x | 12.86% |
| Plane channel | 7.8778 | 6.9013 | 1.14x | 12.40% |

All five runs preserved SIMPLE counts, stop reasons, momentum and pressure
linear iterations, continuity, residuals, and velocity/pressure summaries
exactly.

## Scalar Solve Buffer Reuse

The generic scalar-solve wrapper previously allocated new zero-initial,
matrix-product, and residual vectors before and after every linear solve. One
solver-sized workspace now retains those buffers across all momentum components
and pressure corrections. The pressure loop also passes the latest stored
solution directly to the next non-orthogonal correction instead of cloning a
separate `pressure_guess` field.

The immediate one-warmup/five-measured-run comparison was:

| Case | Before total [s] | After total [s] | Observed speedup | Observed reduction |
| --- | ---: | ---: | ---: | ---: |
| Laminar pipe, 10 SIMPLE | 2.4271 | 2.2228 | 1.09x | 8.42% |
| Plane channel, 500 SIMPLE | 10.3636 | 8.1320 | 1.27x | 21.53% |

The corresponding linear-solve phase medians were:

| Case | Momentum before/after [s] | Momentum ratio | Pressure before/after [s] | Pressure ratio |
| --- | ---: | ---: | ---: | ---: |
| Laminar pipe | 0.20936 / 0.19325 | 1.08x | 0.66654 / 0.63122 | 1.06x |
| Plane channel | 1.72412 / 1.39680 | 1.23x | 5.36625 / 4.41618 | 1.22x |

System load was materially higher and more variable than in the preceding-day
PCG/IC(0) checkpoint. For example, the five pre-change channel solver times
ranged from `8.7139 s` to `13.8518 s`; the post-change times ranged from
`7.5020 s` to `8.7041 s`. The table therefore records the observed adjacent A/B
medians, not an allocation-only hardware-isolated speedup claim.

After timing fields were removed, all ten post-change fixed-work JSON reports
were byte-identical to the pre-change numerical report, including complete
SIMPLE history. A dedicated unit test also verifies that workspace buffer
addresses remain unchanged across successive scalar solves.

## Pressure Matrix Scale And Skew Gate

The reusable PCG/IC(0) path now has a deterministic Rust integration gate for
conservative finite-volume pressure systems in the same size range as the
existing Gmsh medium and fine pipe meshes. A second heterogeneous coefficient
field uses the same CSR topology, forcing the retained IC(0) symbolic structure
to refactor its numerical values. Its result must be exactly equal to a fresh
PCG/IC(0) workspace.

| Case | Rows | NNZ | Shear angle | First iterations | Refactored iterations | True relative residual | Relative solution error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Medium | 13,824 | 68,648 | 0.00 deg | 98 | 129 | 9.828534e-10 | 2.057937e-11 |
| Fine | 38,912 | 193,744 | 0.00 deg | 146 | 213 | 1.602064e-9 | 2.780666e-11 |
| Skewed | 12,288 | 60,992 | 56.31 deg | 100 | 118 | 3.183826e-10 | 6.540778e-12 |

Acceptance limits are `1e-8` for independently recomputed `||b-Ax||/||b||`
and `1e-6` for relative solution error. All cases converged, and every
refactored solution, iteration count, and reported residual was exactly equal
to the corresponding fresh-workspace solve. The existing lower-level test
separately proves that retained PCG and IC(0) buffer addresses do not change.

This gate changes no production solver code, equation, or case input. It
therefore has no new speedup claim; the adjacent and cumulative performance
figures below remain the latest solver measurements.

## Pressure PCG Kernel Profile

The reusable PCG workspace now offers an opt-in profiled solve that executes
the same numerical routine and separately records preconditioner updates,
matrix-vector products, preconditioner applications, and vector operations.
The normal unprofiled linear API remains available. The SIMPLE pressure path
exports the profile through console, JSON, and Markdown reports; the benchmark
driver retains medians for every field.

An optimized release run of the accepted pressure-matrix gate produced:

| Case | PCG total [s] | IC(0) update [s] | Matrix-vector [s] | IC(0) apply [s] | Vector operations [s] | Dominant share |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Medium | 0.09316 | 0.00042 | 0.01934 | 0.05043 | 0.02284 | IC(0) apply 54.1% |
| Fine | 0.51070 | 0.00124 | 0.10455 | 0.28154 | 0.12306 | IC(0) apply 55.1% |
| Skewed | 0.11016 | 0.00043 | 0.02359 | 0.05798 | 0.02805 | IC(0) apply 52.6% |

The instrumented fixed-work profile used one warmup and five measured release
runs. Each timing field below is its own median, so component medians are not
expected to sum exactly to the independently selected total median.

| Case | Pressure solve [s] | PCG total [s] | IC(0) update [s] | Matrix-vector [s] | IC(0) apply [s] | Vector operations [s] |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Laminar pipe, 10 SIMPLE | 0.57909 | 0.57700 | 0.00193 | 0.16138 | 0.27453 | 0.14268 |
| Plane channel, 500 SIMPLE | 5.63256 | 5.58489 | 0.04209 | 1.30411 | 2.61110 | 1.58499 |

A separate one-run convergence diagnostic retained the existing `207` pipe
and `545` channel SIMPLE counts:

| Case | Pressure solve [s] | PCG total [s] | IC(0) update share | Matrix-vector share | IC(0) apply share | Vector-operation share |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Laminar pipe | 18.26843 | 18.19932 | 0.35% | 26.54% | 47.57% | 25.24% |
| Plane channel | 7.40552 | 7.32743 | 0.71% | 25.50% | 45.84% | 27.22% |

The convergence wall times were affected by current host load and do not
replace the stable timing table below. They are used only for within-run phase
shares. In all three matrix gates, profiled and fresh PCG/IC(0) results remain
exactly equal. Fixed and converged pipe/channel reports are also byte-equivalent
to their pre-profile numerical reports after removing only case paths and
timing fields.

The profile rejects IC(0) numerical refactorization as the next optimization:
it accounts for less than one percent in every measured workload. The bounded
next target is the repeated IC(0) forward/backward application, beginning with
a contiguous symbolic dependency layout that preserves operation order. If
that does not materially improve both regression cases, the next pressure
algorithm candidate is multigrid rather than further factor-build tuning.

## Contiguous IC(0) Application Layout

The backward IC(0) substitution previously stored one `Vec` per matrix row.
It now uses one row-offset array and one contiguous `(dependent row, factor
slot)` array. Entries are generated and consumed in exactly the previous row
and factor order. The forward substitution was already contiguous and remains
arithmetically unchanged.

For the accepted pressure matrices, row-metadata storage changes from one
24-byte `Vec` descriptor per row to one 8-byte offset per row plus one terminal
offset. This excludes the unchanged logical entry payload and allocator
metadata:

| Case | Rows | Nested row metadata | Flat offsets | Metadata reduction |
| --- | ---: | ---: | ---: | ---: |
| Medium | 13,824 | 331,776 B | 110,600 B | 221,176 B |
| Fine | 38,912 | 933,888 B | 311,304 B | 622,584 B |
| Skewed | 12,288 | 294,912 B | 98,312 B | 196,600 B |

The flat representation also replaces up to one small dependency allocation
per non-empty row with one entry allocation. A unit test verifies offsets,
entry order, retained buffer addresses, and exact equality against a fresh
PCG/IC(0) solve.

Two sequential fixed-work after-batches showed the host-load limitation again.
The first after-batch improved the targeted median in both cases; a later
Clippy-clean rebuild ran every major channel phase slower:

| Case | Before IC(0) apply [s] | After batch A [s] | Ratio | After-final batch B [s] | Ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| Laminar pipe | 0.22621 | 0.21859 | 1.03x | 0.21901 | 1.03x |
| Plane channel | 2.07216 | 2.02295 | 1.02x | 2.18433 | 0.95x |

The corresponding solver-total medians were `1.73193 -> 1.49522 -> 1.36362 s`
for the pipe and `8.30388 -> 8.31979 -> 8.80259 s` for the channel. These
blocks cannot support an end-to-end speedup claim because unchanged phases
moved with host load.

To isolate the layout, an ignored release diagnostic alternates old nested and
new flat applications inside the same process on the same `38,912`-row IC(0)
factors and residual. Each run takes nine alternating samples of 64
applications and reports each layout's sample median. Three independent
executions produced:

| Execution | Nested median [s] | Flat median [s] | Flat speedup |
| --- | ---: | ---: | ---: |
| 1 | 0.056892 | 0.050984 | 1.1159x |
| 2 | 0.048243 | 0.044881 | 1.0749x |
| 3 | 0.064684 | 0.055616 | 1.1630x |

The median speedup is `1.1159x`. Every nested/flat output is bit-identical.
All ten pipe/channel before/after numerical reports are also exactly equal
after removing timing fields. The layout is therefore accepted for its
isolated application improvement and deterministic memory reduction, while
the published cumulative end-to-end speedups remain unchanged.

## GAMG Pressure Foundation And SIMPLE Integration

The pressure-algorithm foundation is implemented at matrix level and connected
to the symmetric SIMPLE pressure equation. It is based on the OpenFOAM
Foundation 13
[`GAMGSolver`](https://cpp.openfoam.org/v13/classFoam_1_1GAMGSolver.html),
[`GAMGSolver::Vcycle`](https://cpp.openfoam.org/v13/GAMGSolverSolve_8C_source.html),
and
[`pairGAMGAgglomeration`](https://cpp.openfoam.org/v13/pairGAMGAgglomerate_8C_source.html)
contracts rather than on a case-specific pressure shortcut.

The implementation is independently authored Rust over Ferrum's CSR API. The
official OpenFOAM links document behavior, controls, and defaults used for
compatibility; no OpenFOAM C++ source or derived source file is included in the
MIT-licensed Ferrum crates.

The Rust foundation provides:

- reusable CSR hierarchy, transfer addressing, matrices, vectors, and
  coarsest-level PCG/IC(0) storage;
- `algebraicPair` weights from the maximum absolute symmetric face coefficient,
  alternating traversal direction, and OpenFOAM-style assignment of unmatched
  cells to the strongest neighbouring cluster;
- `faceAreaPair` weights from the runtime face-area vectors using OpenFOAM's
  axis-weighted magnitude and summed coarse-face weights;
- coarse-operator coefficient summation, summation restriction, injection
  prolongation, pre/post/finest smoothing, correction scaling, and V-cycles;
- OpenFOAM 13 cycle defaults and explicit `fvSolution` mapping for `smoother`,
  agglomeration, tolerances, iteration limits, cache policy, level size, sweep
  controls, correction controls, and coarsest-level policy;
- one cached hierarchy per SIMPLE pressure topology with coefficient refreshes
  for subsequent pressure equations;
- console, JSON, and Markdown reporting of the effective GAMG controls;
- explicit errors for unsupported controls and for momentum GAMG. No PCG,
  smoother, agglomerator, or interpolation fallback is applied.

The current matrix gate uses a `24 x 20` general Poisson CSR system. GAMG
converges to an absolute residual tolerance of `1e-10`; its solution agrees
with the constructed exact field and an independent PCG/IC(0) result within
`1e-8`. A second `20 x 16` gate updates coefficients while preserving topology
and verifies hierarchy/vector allocation reuse. Pair ordering, OpenFOAM's
unmatched-cell behavior, defaults, and unsupported-control failures have
separate tests. The default iterative PCG/IC(0) coarsest-level solve has its
own convergence gate. Mesh-weight and SIMPLE pressure-correction tests cover
`faceAreaPair`. Four CLI tests verify dictionary mapping, resolved runtime
options, pressure-only selection, and fail-closed validation.

```powershell
cargo test -p ferrum-mesh linear::gamg -- --nocapture
cargo test -p ferrum-mesh face_area_pair -- --nocapture
cargo test -p ferrum-mesh runs_minimal_simple_pressure_correction_with_face_area_gamg -- --nocapture
cargo test -p ferrum-cli gamg -- --nocapture
```

The targeted result is `11/11` GAMG module tests, all three `face_area_pair`
filter matches, the full SIMPLE pressure-correction test, and `5/5` CLI GAMG
tests passing.

### Paired Release Diagnostics

The first end-to-end gate used the same release executable and one measured
run per solver. It is diagnostic evidence, not a five-run median claim. Fixed
work retained 10 pipe and 500 channel SIMPLE iterations:

| Case | PCG total [s] | GAMG total [s] | GAMG / PCG | PCG pressure [s] | GAMG pressure [s] | PCG linear iterations | GAMG V-cycles |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Laminar pipe | 1.16797 | 10.94488 | 9.37x | 0.38380 | 9.81945 | 2,358 | 7,276 |
| Plane channel | 7.08944 | 9.37210 | 1.32x | 3.78331 | 6.32187 | 46,289 | 11,069 |

The converged profiles then exercised the complete outer-control path:

| Case | SIMPLE | PCG total [s] | GAMG total [s] | GAMG / PCG | PCG pressure [s] | GAMG pressure [s] |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Laminar pipe | 207 | 14.02633 | 135.91862 | 9.69x | 9.28206 | 131.35149 |
| Plane channel | 545 | 8.00961 | 9.53562 | 1.19x | 4.35427 | 6.43488 |

Both GAMG runs report `converged=true`; every momentum and pressure linear
solve converged. Compared with PCG, absolute final deltas are at most
`3.99e-11` for velocity L2, `5.72e-10` for pressure L2, and `3.70e-18` for
continuity L2. The solver choice therefore preserves the accepted numerical
result at this reporting precision.

A post-integration audit repeated both converged cases after translating GAMG
`relTol` from the OpenFOAM-normalized LDU L1 criterion to the internal L2
limit. SIMPLE counts (207/545), pressure iteration counts, convergence flags,
and the numerical delta bounds above were unchanged. Host load changed strongly
between the sequential one-run samples (PCG 35.82/18.77 s and GAMG
223.77/12.59 s for pipe/channel), so those audit timings are retained only as
correctness artifacts under `target/benchmarks/gamg_simple_integration/final-*`
and do not replace the paired timing table.

The performance gate does not support a speedup claim. `faceAreaPair` with
`symGaussSeidel` is the valid tested configuration. A diagnostic
`GaussSeidel` profile reached the pipe's linear iteration ceiling and was
rejected; `algebraicPair` was slower on the fixed pipe. PCG/IC(0) remains the
current tutorial and performance default while GAMG remains explicitly
selectable for further profiling.

## GAMG Cycle Profile And Cached Diagonal Smoothing

`--profileGamg` now exposes aggregate and per-level cycle timings without
changing the case dictionary or the numerical solve path. The unprofiled path
does not perform phase clock reads. A parity test executes profiled and
unprofiled workspaces independently and requires bit-identical solutions,
reported residuals, iteration counts, and convergence state.

The first fixed-work release profile used 10 pipe and 500 channel SIMPLE
iterations. It showed that hierarchy construction, matrix refresh, and the
coarsest solve were not the bottleneck. Smoothing consumed `8.06872 s` of the
pipe's `9.89299 s` V-cycle time (`81.56%`) and `3.45983 s` of the channel's
`4.36961 s` (`79.18%`). The pipe performs about `25,904` weighted cell-sweeps
per cycle versus `11,119` for the channel, and its finest CSR matrix has about
`6.58` nonzeros per row versus `4.88`. This explains why its V-cycle costs
substantially more without attributing the difference to hierarchy setup.

The existing GAMG workspace already retained each row's diagonal slot. The
smoother now consumes that layout directly, traversing the entries before and
after the diagonal in the same CSR order instead of searching for the diagonal
inside every sweep. GAMG validates exactly one diagonal entry per row and fails
explicitly otherwise. A dedicated test compares forward and reverse old/new
sweeps bit for bit.

The controlled release A/B used one warmup and five measured runs for each
kernel. Every table value is the median of its own field:

| Case | V-cycles | Smoothing before/after [s] | Smoothing speedup | V-cycle before/after [s] | V-cycle speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| Laminar pipe | 7,276 | 6.53312 / 5.78680 | 1.13x | 8.19654 / 7.64469 | 1.07x |
| Plane channel | 11,069 | 3.60824 / 2.91453 | 1.24x | 4.63997 / 3.91068 | 1.19x |

The targeted smoothing phase improved by `11.42%` for the pipe and `19.23%`
for the channel. The complete V-cycle improved by `6.73%` and `15.72%`.

End-to-end and pressure-solve medians improved in both independent cases:

| Case | Solver before/after [s] | Total speedup | Pressure before/after [s] | Pressure speedup | Current smoothing share |
| --- | ---: | ---: | ---: | ---: | ---: |
| Laminar pipe | 9.65350 / 9.22446 | 1.05x | 8.68195 / 8.19197 | 1.06x | 75.70% |
| Plane channel | 9.00659 / 8.15759 | 1.10x | 6.05422 / 5.26869 | 1.15x | 74.53% |

All five runs per case retained exactly the same SIMPLE counts, pressure
V-cycle counts, continuity, momentum residual, pressure residual, velocity
summary, and pressure summary. The complete solver reduction is `4.44%` for
the pipe and `9.43%` for the channel. This optimization therefore passes both
the isolated kernel gate and the two-case end-to-end gate.

## Current Total Improvement

The requested cumulative comparison from the original recorded basis to the
newest combined five-run median is:

| Case | Original basis [s] | Current combined [s] | Total speedup | Total reduction |
| --- | ---: | ---: | ---: | ---: |
| Laminar pipe, 10 SIMPLE | 64.7464 | 2.2228 | 29.13x | 96.57% |
| Plane channel, 500 SIMPLE | 1134.8209 | 8.1320 | 139.55x | 99.28% |

The original basis remains a single-run historical diagnostic; the current
values are medians of five release runs. The current absolute medians are slower
than the preceding-day `1.5897 s`/`6.9013 s` checkpoint despite less allocation,
which demonstrates the cross-session host-load sensitivity. Future optimization
entries continue to report both the immediately preceding baseline and this
cumulative basis rather than selecting the most favorable historical run.

## Time To Convergence

The external convergence profiles produced:

| Case | Maximum SIMPLE | Executed SIMPLE | U initial / tolerance | p initial / tolerance | Solver time [s] |
| --- | ---: | ---: | ---: | ---: | ---: |
| Laminar pipe | 250 | 207 | 9.986584e-4 / 1e-3 | 2.618382e-5 / 1e-2 | 15.9451 |
| Plane channel | 600 | 545 | 9.974216e-6 / 1e-5 | 4.210369e-8 / 1e-5 | 8.1579 |

Both report `converged=true`. Every momentum-component and pressure linear
solve also converged. Outer convergence and linear convergence remain separate
report fields. The iteration counts and all final numerical observables are
exactly unchanged from the preceding PCG/IC(0) convergence runs. After removing
only working-case paths and timing fields, both complete numerical reports were
byte-identical. Relative to that immediately preceding checkpoint, the pipe
changed from `16.8136 s` to `15.9451 s` (`1.05x`), while the channel changed
from `7.3885 s` to `8.1579 s` (`0.91x`) under the variable host load.

## External Accuracy Check

Analytic and OpenFOAM processing remained outside the solver and its cases.
Ferrum values use SI; OpenFOAM kinematic pressure was converted to Pa. Pipe
mean velocity is volume-weighted from the stored cell field. Channel cells have
uniform volume, so their arithmetic and volume-weighted means are identical.

| Case/source | SIMPLE | Outer converged | Mean U [m/s] | Mean-U error | DeltaP from mean U [Pa] | Execution [s] |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| Pipe analytic | n/a | n/a | 0.02000000 | 0.0000% | 1.6032000 | n/a |
| Pipe Ferrum | 207 | yes | 0.01996467 | -0.1767% | 1.6003677 | 15.9451 |
| Pipe OpenFOAM 13 | 254 | yes | 0.01998286 | -0.0857% | 1.6018259 | 8.7685 |
| Channel analytic | n/a | n/a | 0.02000000 | 0.0000% | 0.6012000 | n/a |
| Channel Ferrum | 545 | yes | 0.02003293 | +0.1646% | 0.6021898 | 8.1579 |
| Channel OpenFOAM 13 | 600 | no | 0.02008722 | +0.4361% | 0.6038217 | 3.4659 |

The matched OpenFOAM pipe run used the same mesh, schemes, and outer tolerances.
The OpenFOAM channel reached its 600-step budget with `Ux` and `p` below their
tolerances, but `Uy` remained about `4.69e-3`; its row is a fixed-budget result,
not an outer-convergence claim.

Named-patch owner-cell pressure differences are retained as field diagnostics,
but they are not used as the primary full-length analytic metric because owner
centres do not lie on the physical boundary planes.

## Next Performance Target

Pressure linear solves still dominate converged runtime. GAMG integration,
mesh-geometric `faceAreaPair`, per-equation controls, hierarchy reuse,
per-level cycle profiling, and the first measured smoother-layout optimization
are complete. Smoothing remains `74.53%` to `75.70%` of the current V-cycle,
while scaling remains about `18%` to `19%`. Any further GAMG performance leaf must first isolate
the remaining smoother row-kernel cost and preserve the same CSR and sweep
order; case tolerances or cycle counts are not performance knobs. PCG/IC(0)
remains the current default and an explicit selectable solver rather than a
hidden fallback. The next solver-capability task is tracked separately in the
roadmap.
Parallel CPU and GPU work must retain the same operator and convergence
contracts.

Reproduction commands:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File validation\scripts\incompressibleFluid\run_cpu_performance_baseline.ps1 -RunProfile fixed -PressureSolver pcg
powershell -NoProfile -ExecutionPolicy Bypass -File validation\scripts\incompressibleFluid\run_cpu_performance_baseline.ps1 -RunProfile fixed -PressureSolver gamg
powershell -NoProfile -ExecutionPolicy Bypass -File validation\scripts\incompressibleFluid\run_cpu_performance_baseline.ps1 -RunProfile converged -PressureSolver gamg -WarmupRuns 0 -MeasuredRuns 1 -RequireConverged
cargo test -p ferrum-mesh --test pressure_mesh_gate -- --nocapture
cargo test --release -p ferrum-mesh --test pressure_mesh_gate -- --nocapture
cargo test --release -p ferrum-mesh benchmarks_flat_ic0_dependency_layout_against_nested_rows -- --ignored --nocapture
```

Generated logs, working cases, fields, JSON, and Markdown remain below
`target/benchmarks`; only this stable summary is versioned.
