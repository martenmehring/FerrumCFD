# Weighted least-squares direct-gradient parity

Status: accepted direct cellwise evidence recorded on 2026-07-31. This closes
the direct-gradient leaf of **F-CYL-SPATIAL-ACCURACY**. It is not evidence for
boundary-condition-resolved wall pressure, reconstructed wall forces, grid
convergence, complete Cylinder parity, or performance.

## Compared implementation and environment

The frozen comparison binds exact merged Ferrum commit
`7dc7a4968fc11c7ac9aa99baf6a719a7f39fc50a` (tree
`d7e9ed8d81e6253d9bc005a56642357f8349118f`) to:

- Rust `1.94.0` on `x86_64-unknown-linux-gnu`;
- OpenFOAM Foundation 13 build `13-441953dfbb42`,
  `linux64GccDPInt32Opt`;
- Ubuntu 22.04.5 under WSL2;
- one frozen set of manufactured fields and generated meshes; and
- tensor-component mapping `[0, 3, 6, 1, 4, 7, 2, 5, 8]` from OpenFOAM tensor
  output to Ferrum's row-major gradient representation.

The Git commit-object SHA256 is
`f901059af5e6571b3a26ca3fadefe85df1524e42ed8563f309753aaab13451a4`;
the tree-object SHA256 is
`0d44571ad71a6d9881bf43dccd9f2c10b930e0d6b77a60c0a28100b3bd5567d3`.

Ferrum and OpenFOAM reconstructed `grad(p)` and `grad(U)` independently from
the same canonical mesh and field inputs. The comparison used the four
fixtures below:

| Fixture | Cells | Pressure system | Geometry / boundary |
| --- | ---: | --- | --- |
| `open-orth-empty` | 48 | open | orthogonal / `empty` |
| `open-skew-symmetry` | 960 | open | skewed / `symmetryPlane` |
| `closed-orth-empty` | 1,024 | closed | orthogonal / `empty` |
| `closed-skew-wedge` | 64 | closed | skewed / `wedge` |

No OpenFOAM source, case, executable, launcher, or raw result is tracked in
FerrumCFD. The generated cases and complete sealed artifact remain under the
ignored local `target/validation/wls-6d-v7` evidence root.

## Accepted metrics

Every gate below retained its frozen threshold. L2 and Linf values are
cellwise norms over the complete fixture, not selected probes.

### `open-orth-empty`

Input-formula Linf was `4.44089209850063e-16`; centre Linf was
`3.33066907387547e-16`.

| Comparison | Field | L2 | L2 gate | Linf | Linf gate |
| --- | --- | ---: | ---: | ---: | ---: |
| Ferrum vs analytic | `grad(p)` | `1.17538663714276e-15` | `1e-11` | `4.44089209850063e-15` | `5e-11` |
| Ferrum vs analytic | `grad(U)` | `1.01481262939566e-15` | `1e-11` | `6.66133814775094e-15` | `5e-11` |
| Foundation 13 vs analytic | `grad(p)` | `7.19745078007501e-16` | `1e-11` | `2.22044604925031e-15` | `5e-11` |
| Foundation 13 vs analytic | `grad(U)` | `5.31371826553176e-16` | `1e-11` | `3.5527136788005e-15` | `5e-11` |
| Ferrum vs Foundation 13 | `grad(p)` | `1.21900064352807e-15` | `2e-11` | `4.44089209850063e-15` | `1e-10` |
| Ferrum vs Foundation 13 | `grad(U)` | `7.94224408063269e-16` | `2e-11` | `4.88498130835069e-15` | `1e-10` |

### `open-skew-symmetry`

Input-formula Linf and centre Linf were both
`4.44089209850063e-16`.

| Comparison | Field | L2 | L2 gate | Linf | Linf gate |
| --- | --- | ---: | ---: | ---: | ---: |
| Ferrum vs analytic | `grad(p)` | `1.55150409774673e-15` | `2e-11` | `1.1508780811565e-14` | `1e-10` |
| Ferrum vs analytic | `grad(U)` | `7.35880024056621e-16` | `2e-11` | `6.77236045021345e-15` | `1e-10` |
| Foundation 13 vs analytic | `grad(p)` | `1.20241145556075e-15` | `2e-11` | `1.04674674012625e-14` | `1e-10` |
| Foundation 13 vs analytic | `grad(U)` | `5.45992351974251e-16` | `2e-11` | `5.99520433297585e-15` | `1e-10` |
| Ferrum vs Foundation 13 | `grad(p)` | `9.84712872636091e-16` | `5e-11` | `7.54951656745106e-15` | `2e-10` |
| Ferrum vs Foundation 13 | `grad(U)` | `4.83675292779672e-16` | `5e-11` | `3.88578058618805e-15` | `2e-10` |

### `closed-orth-empty`

Input-formula Linf was `2.22044604925031e-16`; centre Linf was
`4.44089209850063e-16`.

| Comparison | Field | L2 | L2 gate | Linf | Linf gate |
| --- | --- | ---: | ---: | ---: | ---: |
| Ferrum vs analytic | `grad(p)` | `0.00863124863819235` | `0.1` | `0.05154855196491498` | `0.15` |
| Ferrum vs analytic | `grad(U)` | `7.71923341028839e-16` | `1e-11` | `6.88338275267597e-15` | `5e-11` |
| Foundation 13 vs analytic | `grad(p)` | `0.008631248638192288` | `0.1` | `0.051548551964912884` | `0.15` |
| Foundation 13 vs analytic | `grad(U)` | `1.57660366342472e-16` | `1e-11` | `1.55431223447522e-15` | `5e-11` |
| Ferrum vs Foundation 13 | `grad(p)` | `3.92601896953693e-15` | `2e-10` | `2.37587727269783e-14` | `1e-9` |
| Ferrum vs Foundation 13 | `grad(U)` | `7.73267898453237e-16` | `2e-11` | `6.43929354282591e-15` | `1e-10` |

### `closed-skew-wedge`

Input-formula Linf was `5.55111512312578e-17`; centre Linf was
`3.33066907387547e-16`.

| Comparison | Field | L2 | L2 gate | Linf | Linf gate |
| --- | --- | ---: | ---: | ---: | ---: |
| Ferrum vs analytic | `grad(p)` | `0.022755435571827833` | `0.1` | `0.10395984297196148` | `0.15` |
| Ferrum vs analytic | `grad(U)` | `2.31383504740464e-16` | `2e-11` | `1.55431223447522e-15` | `1e-10` |
| Foundation 13 vs analytic | `grad(p)` | `0.022755435571827756` | `0.1` | `0.10395984297196012` | `0.15` |
| Foundation 13 vs analytic | `grad(U)` | `2.03938452963963e-16` | `2e-11` | `1.11022302462516e-15` | `1e-10` |
| Ferrum vs Foundation 13 | `grad(p)` | `1.15970719576806e-15` | `2e-10` | `5.32907051820075e-15` | `1e-9` |
| Ferrum vs Foundation 13 | `grad(U)` | `1.1965135650164e-16` | `5e-11` | `1.11022302462516e-15` | `2e-10` |

The closed-system analytic pressure error is a boundary-constrained
reconstruction measure rather than a cross-engine disagreement. Ferrum and
Foundation 13 agree there to approximately `1e-15` while independently
exhibiting the same finite-boundary analytic error.

## Sealed provenance

The accepted artifact records these SHA256 values:

| Item | SHA256 |
| --- | --- |
| Protocol | `f98fbe8f441220b2ad83fc9c34360650b1c9b70f39efaf7da98f30eeb41831f7` |
| Comparator | `d58c6f88fbfc90b322daa7a742f660586d35fdc75c9fc86a6de0847e1db18ef7` |
| Run manifest | `58bf15ad5886fc203002fdf35bec366c21517939f8311b24ecc541028073c444` |
| Ferrum probe source | `4d87fc61be6c15eea93d687e6c80b954058cb505fcc6e7dcbf961022324ec086` |
| Linux Ferrum probe | `feaa3b13c9f0b5a2b7844d76c311737e179eeeacdb7e71bf6fc004323c2e2e61` |
| Canonical input manifest | `005fd71034634b0f527cd0c74aeb90b1524e5b5964f8866ff188122a88497d72` |
| OpenFOAM output manifest | `d183a7a217eaf0e235f73ec2ec436f44d28a128e0b15749e0626087791f869f5` |
| OpenFOAM provenance | `9a15aef8e24e06572287d717d621d9187e4eb0f723023c74b1fcf85c4f3794df` |
| Rust provenance | `b75985398e5509a90a7d5d68f8ec6690675f703c5cfa11893b04fc9c3fceca69` |
| Environment | `f25020109f3f7b652bf17592b695f266273a8cd16f3a988ce1e256ad7890d5ab` |
| Final `result.json` | `1a2fd36cab8c010878e25e4c9946f09f66f01a7722c67f515530e0b29ed17c20` |

## Rejected predecessors

Rejected runs remain rejected evidence:

- **v3** failed one generic analytic pressure gate at
  `0.21471920749717682 > 0.15`. The unnamed failure was initially, and
  incorrectly, attributed to the closed orthogonal fixture.
- **v4** refined that fixture from 16 by 16 to 32 by 32 cells on the basis of
  that attribution, without relaxing a threshold. The same generic failure
  remained; post-rejection diagnostics then localized it to wedge pressure.
- **v5** refined the wedge from 32 to 64 cells. Pressure passed, while the
  continuous radial affine velocity oracle was not invariant under the finite
  wedge transform. Ferrum and Foundation 13 nevertheless agreed to
  approximately `1e-15`, so this was a fixture-contract defect rather than an
  engine discrepancy.
- **v6** replaced only that wedge velocity with the invariant axial affine
  field but failed before metrics because the protocol carried a stale probe
  source pin.
- **v7** corrected the source pin before generating fresh outputs and passed
  every unchanged numerical gate.

No rejected result was reclassified, and no numerical threshold was loosened.
