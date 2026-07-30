# Cylinder same-Linux parity

Status: accepted C4 evidence recorded on 2026-07-30. This is a narrow serial
CPU comparison for one steady `Re = 1` Cylinder configuration, not a general
FerrumCFD performance claim.

## Compared configuration

The comparison archived exact Ferrum source commit
`3d84b33f2406b143e6349ea6a9e9438c029a324f` (tree
`e75fb0c9172bcadc261fe5826ca6da9334b4172a`) and built it on Linux with
Rust `1.94.0` and `-C target-cpu=native`. Ferrum and a separately installed
OpenFOAM Foundation 13 build (`13-441953dfbb42`, `linux64GccDPInt32Opt`) ran
inside the same Ubuntu 22.04 WSL2 ext4 environment, pinned to logical CPU `2`
of an Intel Core i7-1165G7. Both processes used a one-thread environment and
the same external GNU `time` elapsed metric. Compilation and all force
post-processing were excluded from measured time.

The installed official Foundation 13 Cylinder tutorial generated one
5,388-cell mesh outside the repository. That one `polyMesh` was copied to
both temporary cases. The canonical file hashes were:

| File | SHA256 |
| --- | --- |
| `points` | `4ae33eb62d0fc1bb2b6f6e6ab43234bc87389c1b3bda39bd2ad85e70c8bb47c1` |
| `faces` | `6111ef554cacccf9945a082b8d6952f0d4d2ec009dbcd1fa4e9f4e79c6d24726` |
| `owner` | `1a8db68035dfab16339cfdcf15067018073ef786fbba99f257ec9d1e70539975` |
| `neighbour` | `3e9a4ca17a03512199c6537ba6bb2d0f249467f725fc1fae42385296307645eb` |
| `boundary` | `5a65c6db02c789cebc44fd5775d3ff2f21b3128001bd505a14f3144d3d8026e0` |

Both cases used the same initial `U` and `p`, `Uinlet = 0.015 m/s`,
`nu = 1.5e-5 m2/s`, finite-volume schemes, pressure `PCG/DIC`, velocity
`smoothSolver/symGaussSeidel`, absolute linear tolerance `1e-9`, zero linear
`relTol`, pressure/velocity relaxation `0.3/0.7`, and one corrected
non-orthogonal pressure pass. The outer-control dictionaries are necessarily
engine-specific rather than byte-identical: Ferrum uses its supported `SIMPLE`
block; Foundation 13 uses a semantically matched `PIMPLE` block with one outer
corrector, one pressure corrector, and `momentumPredictor yes`.

No OpenFOAM source, case, executable, or comparison launcher is tracked in
FerrumCFD. Only this result and its external-reference provenance are retained.

## Protocol

Each timing track used one warmup and six measured pairs with alternating
AB/BA order. Timing cases wrote no time directories and ran no function
objects.

- **Fixed work:** both engines ran exactly 1,000 outer steps. No
  `residualControl` block was present. Ferrum also received equal explicit
  minimum and maximum SIMPLE iteration bounds.
- **Time to residual accuracy (TTA):** an unmeasured discovery run used the
  common `U,p = 1e-5` residual target. OpenFOAM writes a final state when its
  residual controller ends a run, so timing that discovery directly would
  charge write I/O only to OpenFOAM. The measured phase therefore replayed
  each discovered iteration count without residual control or writes. This
  preserves the discovered work-to-threshold while keeping the timed process
  contract symmetric.

The fixed-work and TTA tracks answer different questions and must not be
combined into one speed number.

## Timing results

Primary values are external process elapsed seconds. MAD is the median
absolute deviation across the six measured runs or paired ratios.

| Track | Ferrum median [s] | Ferrum MAD [s] | OpenFOAM median [s] | OpenFOAM MAD [s] | Paired Ferrum/OpenFOAM median | Paired MAD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fixed 1,000 | 85.170 | 0.840 | 31.335 | 5.170 | 2.752571 | 0.495953 |
| TTA `1e-5` | 129.680 | 15.230 | 38.735 | 9.535 | 3.335669 | 0.608799 |

Observed paired Ferrum/OpenFOAM ratios ranged from `2.122564` to `3.296440`
for Fixed-1,000 and from `2.479727` to `4.041439` for TTA. These ranges expose
the laptop/WSL variability; the paired median and MAD remain the primary
statistics.

All six fixed-work runs for each engine executed exactly 1,000 steps. All six
TTA replays executed exactly 986 Ferrum steps and 995 OpenFOAM steps. Ferrum
therefore reached the selected residual threshold nine outer steps earlier,
but its Cylinder step remained substantially more expensive: Ferrum consumed
about `2.75x` the OpenFOAM elapsed time at fixed work and `3.34x` in the TTA
track. This case is an optimization target; it disproves an all-case Ferrum
speedup claim.

## Physics and field parity

The unmeasured residual-controlled discovery produced the following final
observables:

| Engine | Steps | Cd | Cl | Local continuity | Global continuity |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ferrum | 986 | 10.6272235558 | -3.493474e-9 | 6.332751e-12 | -1.107258e-13 |
| OpenFOAM Foundation 13 | 995 | 10.6436129968 | -1.025552e-8 | 2.012036e-11 | 2.122538e-13 |

The drag-coefficient difference is `0.153984%`. With the byte-identical mesh
establishing common cell ordering and the fixed-zero outlet establishing the
same pressure gauge, the full internal-field differences are:

| Field | Ferrum vs OpenFOAM relative L2 | Gate |
| --- | ---: | ---: |
| `U` | 0.525295% | <= 2% |
| `p` | 0.926074% | <= 2% |

The force, continuity, and field gates all pass. The comparison does not claim
bit-identical cross-engine fields.

## Residual-control behavior

Residual stopping is optional. The TTA discovery demonstrates automatic
termination when a `residualControl` block is supplied. The separate
Fixed-1,000 track omits that block and demonstrates that both engines honor the
full explicit iteration budget even though the selected TTA threshold would
have been reached earlier.

## External evidence and provenance notes

The ignored local evidence directories are:

- `target/benchmarks/c4-linux-cylinder-native-3d84b33-v1` for the TTA timing,
  discovery physics, and the non-claim Fixed-200 diagnostic;
- `target/benchmarks/c4-linux-cylinder-fixed1000-native-3d84b33-v1` for the
  accepted Fixed-1,000 track.

The TTA artifact predates at-run harness self-sealing. Its already sealed
discovery fields were checked post-run by the preserved field verifier with
SHA256
`61205d570ca46f2db32b89f85ab53a3116e9f52c32b8dae7f864a3e20cb0b286`;
the artifact manifest was then regenerated. This post-run field gate is not
presented as at-run timing-harness provenance.

The Fixed-1,000 artifact sealed the exact executing harness and field verifier
before benchmark execution. Their SHA256 values are respectively
`e34b8508e51519702a7c931ec9f2986a8e202e2a2445ccca314b417ae8d2255a`
and
`61205d570ca46f2db32b89f85ab53a3116e9f52c32b8dae7f864a3e20cb0b286`.
All 14 engine runs, summaries, metadata, and raw logs completed. A fixed-only
export path initially treated the intentionally absent optional
`verification/` directory as a `pipefail`; the preserved workspace and copied
metadata/raw trees were byte-compared, and the final artifact checksum was
completed without changing or rerunning a measurement.
