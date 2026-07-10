# Examples

Generated example cases can be written here with `gmshToFerrum`.

Mesh inputs and generated `constant/` or `system/` case output are ignored by
Git because they can be large and may contain private geometry.

Versioned examples:

- `laminar_pipe`: generated circular pipe benchmark for laminar water flow,
  constant wall temperature, Hagen-Poiseuille pressure-loss references, and
  OpenFOAM comparison scripts that record runtime as a benchmark-only artifact.
- `gmsh_pipe`: simple SI pipe `.geo` for Gmsh with a two-layer near-wall prism
  layer setup. It is intended as a shared mesh source for FerrumCFD import and
  later OpenFOAM comparison runs. The mesh-study script creates `coarse`,
  `medium`, and `fine` variants from this same `.geo`.

Useful local test commands:

```powershell
gmshToFerrum path\to\mesh2d.msh -case examples\plate2d -emptyPatch frontAndBack
gmshToFerrum path\to\axisymmetric.msh -case examples\reactor_axi -wedgePatch wedgeMin -wedgePatch wedgeMax
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_gmsh_pipe_import.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_gmsh_pipe_mesh_study.ps1 -SkipOpenFoam
```
