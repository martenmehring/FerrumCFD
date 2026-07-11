# Legacy Application Package

`ferrumCli` currently contains the shared implementation behind the solver,
mesh utilities, case tools, and benchmark post-processors. The public
`ferrumRun` dispatcher crate already lives under `applications/solvers/`, but
delegates here until the module lifecycle is extracted. The legacy package
remains fully buildable during the repository migration, but it no longer
publishes an application-solver binary. Application execution is owned by
`ferrumRun`.

Code moves out of this directory only with behavior-parity tests and without
changing the public command names accidentally.
