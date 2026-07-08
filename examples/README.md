# Examples

Generated example cases can be written here with `gmshToFerrumFoam`.

Mesh inputs and generated `constant/` or `system/` case output are ignored by
Git because they can be large and may contain private geometry.

Useful local test commands:

```powershell
gmshToFerrumFoam path\to\mesh2d.msh -case examples\plate2d -emptyPatch frontAndBack
gmshToFerrumFoam path\to\axisymmetric.msh -case examples\reactor_axi -wedgePatch wedgeMin -wedgePatch wedgeMax
```
