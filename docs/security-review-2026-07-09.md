# Codex Security Review - 2026-07-09

This review maps the 29 findings exported in
`codex-security-findings_ferrumcfd.csv` to the current working tree. The scan
was produced for commit `d5e9a9ea40e7374ada719c0a2a83d79389d7fde9`; the
review includes later solver work and the uncommitted fixes listed below.

Status meanings:

- **Fixed**: the reported source-to-sink path is rejected or handled safely.
- **Policy**: expected local CLI behavior under `SECURITY.md`; hosted wrappers
  must enforce quotas and filesystem isolation.
- **Open**: a real robustness issue remains and is tracked as follow-up work.

| # | Finding | Status | Current control or evidence |
| ---: | --- | --- | --- |
| 0 | Solver preflight can exhaust memory on sparse cell labels | Fixed | `PolyMesh::validate` rejects overflowing, sparse, and non-dense labels before geometry or field allocation; regression test covers a billion-cell label. |
| 1 | Property preflight can leak files via constant symlinks | Fixed | Property files and region directories are inspected without following symlinks; property symlinks are rejected. |
| 2 | checkFerrumMesh now allocates from untrusted sparse cell labels | Fixed | Region and base-mesh label validation runs before cell-to-zone allocation. |
| 3 | Unbounded non-orthogonal correctors can DoS solver | Policy | The `+ 1` overflow is checked. Iteration counts remain operator controls by design; services must enforce CPU/wall-time quotas. |
| 4 | Laminar SIMPLE convergence ignores pressure-field drop | Fixed | Benchmark pressure-drop acceptance no longer participates in solver convergence. `converged=true` requires configured residual controls, continuity, and successful linear solves. |
| 5 | Unbounded fvSolution maxIter enables CPU denial of service | Policy | `maxIter` is intentional numerical configuration, matching OpenFOAM behavior. Hosted execution must constrain cases or process resources rather than silently changing solver settings. |
| 6 | Laminar SIMPLE trusts patch ranges and can panic/OOM | Fixed | Mesh and runtime validation use checked patch ends, face bounds, boundary coverage, and overlap checks before treatment-vector access. |
| 7 | Near-zero solver pivots can produce non-finite solves | Fixed | Iterative solvers reject invalid pivots and non-finite updates; regression coverage exercises a tiny Jacobi pivot. |
| 8 | Unsafe gmsh.exe auto-discovery in Downloads | Fixed | Mesh-study tooling now accepts an explicit executable or resolves `gmsh` from `PATH`; Downloads discovery was removed. |
| 9 | Untrusted gmsh.exe auto-discovery in script | Fixed | Import tooling uses the same explicit-path/`PATH` policy. |
| 10 | Unbounded nonuniform field value loading can exhaust memory | Fixed for the reported numeric-tail amplification | Numeric materialization stops at the declared scalar count and rejects extra numeric values. The broader whole-file parsing concern is tracked in finding 18. |
| 11 | Unchecked byte-total sum can panic on crafted mesh labels | Fixed | Cell labels are validated first and byte totals use checked accumulation. |
| 12 | Unbounded runner dry-run preview can exhaust resources | Policy | Preview length is an explicit CLI operator option with a small default. Hosted wrappers must not expose an unrestricted value. |
| 13 | Unrestricted --planJson path can clobber writable files | Policy | An explicit CLI output path is expected local behavior. Hosted wrappers must supply an isolated approved output path. |
| 14 | Quadratic fvSolution preflight can DoS untrusted cases | Fixed | Solver-field membership uses precomputed sets instead of repeated all-pairs scans. |
| 15 | Unbounded recursive fvSchemes/fvSolution parsing can DoS | Fixed | Dictionary parsing rejects nesting beyond 128 levels; a regression test covers excessive nesting. |
| 16 | Quadratic field-boundary validation can DoS checkFerrumMesh | Fixed | Boundary lookup uses a hash map, reducing matching to linear expected time. |
| 17 | checkFerrumMesh panics on crafted polyMesh owner label | Fixed | Checked, dense cell-label validation precedes geometry allocation and indexing. |
| 18 | Unbounded initial-field parsing can exhaust memory | Open | Field files are still read and tokenized completely, with additional owned token/value storage. Replace this with a streaming or load-policy-aware parser without imposing an arbitrary CFD field-size cap. |
| 19 | Case initializer follows symlinks when writing templates | Fixed | Initializer directories and files use `symlink_metadata` and reject symlink targets, including with `--force`. |
| 20 | checkFerrumMesh can be crashed by crafted region mesh counts | Fixed | List parsers validate the actual list before allocation and no longer reserve from an untrusted declared count. |
| 21 | Unbounded split-region parsing enables DoS | Fixed | Split parsing uses safe list allocation plus checked and bounded patch ranges. |
| 22 | Unvalidated patch type injection in OpenFOAM writer | Fixed | The public writer validates every patch type as an OpenFOAM word before creating output; a syntax-injection regression test is present. |
| 23 | Malformed Gmsh element tag count can panic parser | Fixed | Signed values use checked conversions and checked index arithmetic; a negative-tag-count test returns a parse error. |
| 24 | Diffusion assembly permits non-finite CSR/RHS values | Fixed | Source, coefficient, matrix, and RHS operations reject non-finite results; overflow regression coverage is present. |
| 25 | CG breakdown can be used as a SIMPLE solve result | Fixed | Linear convergence state is preserved through `ScalarSolveReport`; failed momentum or pressure solves prevent SIMPLE convergence and set an invalid-state stop reason. |
| 26 | WSL benchmark script allows shell injection via WorkDir | Fixed | Work directories are constrained below the repository target directory and WSL paths are single-quote escaped before `bash -lc`. |
| 27 | Unchecked run-time write estimate can saturate | Fixed | Write estimates reject non-finite or out-of-range values before integer conversion. |
| 28 | Backend wrapper accepts missing closing brace | Fixed | EOF before the outer closing brace is an error; regression coverage verifies it. |

## Validation

- `cargo test --workspace --all-targets`
- `cargo clippy --workspace --all-targets -- -D warnings`
- PowerShell AST parsing for every script below `scripts/`
- One real `examples/laminar_pipe` SIMPLE iteration using the case's
  `fvSchemes`, `fvSolution`, fields, properties, and `constant/polyMesh`

## Remaining Work

Finding 18 should be addressed by separating field-summary inspection from
full solver-state loading and introducing a streaming OpenFOAM token source.
This keeps memory proportional to data the selected operation actually needs
without an artificial global field-size limit.
