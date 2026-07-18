# Third-party notices

FerrumCFD is an independent project. Its Rust source code is distributed under
the repository's [MIT License](LICENSE).

## OpenFOAM Foundation 13

OpenFOAM Foundation 13 is an external, optional validation and interoperability
tool. It is not included in FerrumCFD and is not a FerrumCFD runtime dependency.
The independently runnable sibling cases under `tutorials/**/openfoam-v13/`
use OpenFOAM dictionary names, format identifiers, and generated file headers
where those names are required by the external OpenFOAM program or file format.

OpenFOAM is distributed separately under the GNU General Public License v3.0:

- <https://github.com/OpenFOAM/OpenFOAM-13>
- <https://github.com/OpenFOAM/OpenFOAM-13/blob/master/COPYING>

FerrumCFD is not affiliated with or endorsed by the OpenFOAM Foundation or CFD
Direct Ltd. OpenFOAM is a registered trademark of OpenCFD Limited, a producer
of OpenFOAM software. Use of the name in this repository identifies external
compatibility cases and validation evidence only.

## Gmsh

Gmsh is an external, optional mesh-generation tool. FerrumCFD can consume the
documented Gmsh 2.2 ASCII format, and selected tutorial directories include
independently authored `.geo` inputs. Gmsh is not included as a FerrumCFD
runtime dependency.

- <https://gmsh.info/>
- <https://gitlab.onelab.info/gmsh/gmsh>

FerrumCFD is not affiliated with or endorsed by the Gmsh project.
