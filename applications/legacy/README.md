# Legacy Application Package

`ferrumCli` currently contains the executable solver, mesh utilities, case
tools, and benchmark post-processors in one Cargo package. It remains fully
buildable during the repository migration.

Code moves out of this directory only with behavior-parity tests and without
changing the public command names accidentally.
