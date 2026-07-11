# ferrumMultiRun

`ferrumMultiRun` is the planned coupled multi-region dispatcher. It follows
the OpenFOAM 13 `foamMultiRun` responsibility: one case, multiple named
regions, one runtime-selectable solver module per region, and one coordinated
time/coupling loop. It is not a parameter-sweep or unrelated-case batch tool.

The implementation contract requires:

- a `regionSolvers`-style region-to-module registry;
- no `-solver` option; all module selection comes from the region registry;
- a capability/dependency phase graph with synchronization at every coupled
  data dependency and concurrency only for independent tasks;
- a global time-step constrained by global controls and active transient
  regions, with explicit mixed steady/transient operation;
- convergence only when all participating region criteria are satisfied;
- backend assignment per region and compute stage;
- CPU sockets/cores/workers, process ranks, domain decomposition, halo/ghost
  exchange, and conservative region-interface communication;
- CPU, GPU, mixed CPU/GPU, and multi-GPU resource planning without
  oversubscription;
- explicit data residency and host/device transfers, backend capability checks
  including `f64`, deterministic reductions, and parity/conservation tests.

The shared lifecycle and backend APIs are designed with `ferrumRun` and must
be stable before this dispatcher is implemented. Executable multi-region
support is a prerequisite for Driver 6.
