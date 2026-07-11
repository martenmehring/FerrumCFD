# FerrumCFD

FerrumCFD is an early Rust CFD platform focused on native, backend-aware
finite-volume solvers and reproducible comparison with OpenFOAM 13 and
analytical references.

The first executable application module is `incompressibleFluid`. Its current
CPU implementation covers steady laminar SIMPLE cases and has validation
bundles for a 3D circular pipe and a true 2D plane channel.

```text
ferrumRun -solver incompressibleFluid -case <case>
```

`incompressibleFluid` is the permanent public module name. Laminar flow is a
model regime, while SIMPLE, SIMPLEC, PISO, and PIMPLE are algorithms selected
by the case. There is no separate algorithm-named solver executable.

## Quick Start

Requirements are a current Rust toolchain and PowerShell. Gmsh and a local
OpenFOAM 13 installation under WSL are optional and needed only to regenerate
meshes or external reference runs.

```powershell
cargo test --workspace

cargo run -p ferrum-cli --bin checkFerrumMesh -- `
  -case tutorials\incompressibleFluid\laminarPipe\ferrum\case

cargo run -p ferrum-run --bin ferrumRun -- `
  -solver incompressibleFluid `
  -case tutorials\incompressibleFluid\laminarPipe\ferrum\case `
  --preflight `
  --planJson target\ferrumRunPlan.json

cargo run -p ferrum-run --bin ferrumRun -- `
  -solver incompressibleFluid `
  -case tutorials\incompressibleFluid\laminarPipe\ferrum\case `
  --maxSimpleIterations 2
```

The `solver incompressibleFluid;` entry is already present in the curated
Ferrum cases, so `-solver incompressibleFluid` may be omitted there. Keeping it
in examples makes module selection explicit.

## Repository Layout

```text
applications/    public runners, runtime modules, and utilities
src/             reusable Rust mesh, finite-volume, I/O, and model libraries
tutorials/       independent Ferrum, OpenFOAM 13, and reference case bundles
validation/      comparison automation and stable validation contracts
test/            cross-package test contracts
docs/            user guide, architecture, roadmap, and benchmark reports
target/          generated build and validation artifacts
```

`ferrumRun` is the single-region runner. The planned `ferrumMultiRun` is the
coupled multi-region runner: one case, several regions, one module per region,
and a shared phase-oriented coupling loop. Its design includes CPU, GPU, mixed
CPU/GPU, and multi-GPU resource placement. It is intentionally not a batch or
parameter-sweep tool.

Both runners share one backend contract. After the selected steady and
transient incompressible SIMPLE/SIMPLEC/PISO/PIMPLE cases pass on the serial CPU
reference backend, `ferrumRun` advances through threaded CPU, distributed CPU,
single-GPU, and multi-GPU acceptance. `ferrumMultiRun` reuses those kernels for
coupled regions.

## Documentation

- [User Guide](docs/user-guide.md)
- [Architecture](docs/architecture.md)
- [Solver Roadmap](docs/solver-roadmap.md)
- [Validation Script Policy](docs/development/script-policy.md)
- [Benchmark Reports](docs/benchmarks)
- [Changelog](CHANGELOG.md)
- [Security Policy](SECURITY.md)

Local tutorial `README.md` files stay beside their cases because they describe
how that specific validation bundle is structured and reproduced. Project
manuals and design documents live under `docs/`. Standard repository metadata
(`LICENSE`, `CHANGELOG.md`, and `SECURITY.md`) remains at the root for GitHub
discovery.

## Validation Automation

PowerShell files under `validation/scripts/` are reproducibility and developer
tools, not public solver commands. For example, the matched Ferrum/OpenFOAM 13
pipe comparison is run with:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File validation\scripts\incompressibleFluid\run_laminar_simple_matched_time_benchmark.ps1 `
  -MatchedTimeSeconds 100
```

Generated meshes, fields, logs, plots, and reports belong below `target/`.

## License

FerrumCFD is licensed under the [MIT License](LICENSE).
