# Third-party notices

FerrumCFD is an independent project. Its source code is distributed under the
repository's [GNU General Public License v3.0 or later](LICENSE).

## OpenFOAM Foundation 13

OpenFOAM Foundation 13 was used as a separately installed external reference
for recorded interoperability, numerical-validation, and performance results.
It is not a FerrumCFD runtime dependency. FerrumCFD does not bundle OpenFOAM
source code, binaries, tutorial cases, or other artifacts obtained from an
OpenFOAM distribution.

The benchmark documents under `docs/benchmarks/` retain only the external
implementation and version, measurement protocol, and recorded results needed
to interpret the comparison. Reproduction requires users to obtain OpenFOAM
and any corresponding reference inputs independently from its distributor.

OpenFOAM is distributed separately under the GNU General Public License
version 3 or later:

- <https://github.com/OpenFOAM/OpenFOAM-13>
- <https://github.com/OpenFOAM/OpenFOAM-13/blob/master/COPYING>

FerrumCFD is not affiliated with or endorsed by the OpenFOAM Foundation or CFD
Direct Ltd. OpenFOAM is a registered trademark of OpenCFD Limited, a producer
of OpenFOAM software. Use of the name in this repository identifies the
external compatibility target and recorded validation evidence only.

## Gmsh

Gmsh is an external, optional mesh-generation tool. FerrumCFD can consume the
documented Gmsh 2.2 ASCII format, and selected tutorial directories include
independently authored `.geo` inputs. Gmsh is not included as a FerrumCFD
runtime dependency.

- <https://gmsh.info/>
- <https://gitlab.onelab.info/gmsh/gmsh>

FerrumCFD is not affiliated with or endorsed by the Gmsh project.
