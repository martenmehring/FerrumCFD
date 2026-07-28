# Security Policy

FerrumCFD is currently a local command-line CFD application and Rust library.
Case and mesh file contents are treated as untrusted parser input: malformed
OpenFOAM or Gmsh data must return an error and must not panic, follow embedded
symlinks, escape the selected case directory, or allocate from unvalidated
labels and declared counts.

Command-line paths and numerical run controls are operator configuration.
Options such as output paths, `maxIter`, SIMPLE iteration counts, and runner
preview counts are intentionally not capped by the solver. Services, CI jobs,
and shared HPC front ends that expose FerrumCFD to other users must constrain
those operator-controlled arguments and enforce process-level CPU, memory,
wall-time, filesystem, and GPU quotas.

FerrumCFD does not bundle benchmark launchers or external solver runners.
Maintainer measurements must run outside the product repository and must not
turn candidate-controlled paths or output into trusted execution controls.

The supported code is the current default branch. Report suspected
vulnerabilities privately through the repository's GitHub security advisory
interface. Do not include confidential case data in a public issue.
