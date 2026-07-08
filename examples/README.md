# Examples

Generated example cases can be written here with `gmshToFerrumFoam`.

Mesh inputs and generated `constant/` or `system/` case output are ignored by
Git because they can be large and may contain private geometry.

Versioned examples:

- `laminar_pipe`: generated circular pipe benchmark for laminar water flow,
  constant wall temperature, Hagen-Poiseuille pressure-loss references, and
  OpenFOAM comparison scripts that record runtime as a benchmark-only artifact.
- `gmsh_pipe`: simple SI pipe `.geo` for Gmsh with a two-layer near-wall prism
  layer setup. It is intended as a shared mesh source for FerrumCFD import and
  later OpenFOAM comparison runs.

Useful local test commands:

```powershell
gmshToFerrumFoam path\to\mesh2d.msh -case examples\plate2d -emptyPatch frontAndBack
gmshToFerrumFoam path\to\axisymmetric.msh -case examples\reactor_axi -wedgePatch wedgeMin -wedgePatch wedgeMax
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_gmsh_pipe_import.ps1
```
