mod case;

use std::env;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use case::{InitCaseOptions, init_case};
use ferrum_mesh::backends::{
    read_backend_config, validate_backend_policy, validate_backend_resources,
};
use ferrum_mesh::check::read_case_summary;
use ferrum_mesh::diffusion::{
    assemble_scalar_diffusion_system, diffusion_assembly_capabilities,
    scalar_diffusion_options_from_field,
};
use ferrum_mesh::fields::{
    FieldBoundaryValidationSummary, FieldFile, InitialFieldSet, read_initial_fields,
    validate_initial_field_boundaries,
};
use ferrum_mesh::flow::{
    ContinuitySummary, FlowBoundarySummary, FlowOperatorSummary, LaminarSimpleIterationSummary,
    LaminarSimpleLinearSolver, LaminarSimpleOptions, LaminarSimplePreconditioner,
    LaminarSimpleReport, LaminarSimpleSolutionSummary, solve_laminar_simple,
};
use ferrum_mesh::foam::{FoamWriteOptions, write_openfoam_case_with_options};
use ferrum_mesh::geometry::{GeometrySummary, summarize_case_geometry};
use ferrum_mesh::gmsh::read_msh22_ascii;
use ferrum_mesh::interfaces::{read_interface_config, validate_interface_config};
use ferrum_mesh::linear::{
    ConjugateGradientOptions, JacobiOptions, conjugate_gradient_solve, jacobi_solve,
    linear_solver_capabilities,
};
use ferrum_mesh::patches::{PatchValidationSummary, validate_case_patches};
use ferrum_mesh::poiseuille::{
    PoiseuilleOptions, poiseuille_diffusion_options, poiseuille_reference,
    summarize_poiseuille_solution,
};
use ferrum_mesh::regions::{
    InterfaceRegistrySummary, InterfaceSummary, build_interface_registry,
    read_region_mesh_summaries, split_regions_by_cell_zones,
};
use ferrum_mesh::runner::{
    SolverRunnerDryRun, SolverRunnerDryRunEvent, SolverRunnerDryRunOptions,
    build_solver_runner_dry_run,
};
use ferrum_mesh::runtime::SolverRuntimeData;
use ferrum_mesh::solver_plan::{
    SolverBackendPlan, SolverCasePlan, SolverFieldPlan, SolverInterfacePlan, SolverMeshPlan,
    SolverNumericsDictionaryPlan, SolverNumericsPlan, SolverPropertiesPlan, SolverRunPlan,
    build_solver_case_plan,
};
use ferrum_mesh::solver_state::SolverStatePlan;

pub fn run_ferrum() -> i32 {
    let args = env::args().skip(1).collect::<Vec<_>>();
    match run_command(CommandMode::Ferrum, args) {
        Ok(()) => 0,
        Err(error) => {
            eprintln!("error: {error}");
            1
        }
    }
}

pub fn run_alias(alias: Alias) -> i32 {
    let args = env::args().skip(1).collect::<Vec<_>>();
    match run_command(CommandMode::Alias(alias), args) {
        Ok(()) => 0,
        Err(error) => {
            eprintln!("error: {error}");
            1
        }
    }
}

#[derive(Clone, Copy)]
pub enum Alias {
    GmshToFerrumFoam,
    CheckFerrumMesh,
    SplitFerrumMeshRegions,
    InitFerrumCase,
    FerrumSolver,
}

enum CommandMode {
    Ferrum,
    Alias(Alias),
}

fn run_command(mode: CommandMode, args: Vec<String>) -> Result<(), String> {
    match mode {
        CommandMode::Ferrum => run_ferrum_subcommand(args),
        CommandMode::Alias(Alias::GmshToFerrumFoam) => gmsh_to_foam(args),
        CommandMode::Alias(Alias::CheckFerrumMesh) => check_mesh(args),
        CommandMode::Alias(Alias::SplitFerrumMeshRegions) => split_mesh_regions(args),
        CommandMode::Alias(Alias::InitFerrumCase) => init_case_command(args),
        CommandMode::Alias(Alias::FerrumSolver) => solve_case(args),
    }
}

fn run_ferrum_subcommand(mut args: Vec<String>) -> Result<(), String> {
    if args.is_empty() || is_help(&args[0]) {
        print_help();
        return Ok(());
    }

    let command = args.remove(0);
    match command.as_str() {
        "gmshToFoam" | "gmshToFerrumFoam" => gmsh_to_foam(args),
        "checkMesh" | "checkFerrumMesh" => check_mesh(args),
        "splitMeshRegions" | "splitFerrumMeshRegions" => split_mesh_regions(args),
        "initCase" | "initFerrumCase" => init_case_command(args),
        "solve" | "solver" | "ferrumSolver" => solve_case(args),
        other => Err(format!("unknown ferrum command '{other}'")),
    }
}

fn init_case_command(args: Vec<String>) -> Result<(), String> {
    if args.is_empty() || args.iter().any(|arg| is_help(arg)) {
        print_init_case_usage();
        return Ok(());
    }

    let options = parse_init_case_args(&args)?;
    let summary = init_case(&options)?;

    println!("Initialized FerrumCFD case: {}", summary.case_dir.display());
    if !summary.created_dirs.is_empty() {
        println!("created directories:");
        for path in &summary.created_dirs {
            println!("  {}", path.display());
        }
    }
    if !summary.written_files.is_empty() {
        println!("written files:");
        for path in &summary.written_files {
            println!("  {}", path.display());
        }
    }
    if !summary.skipped_files.is_empty() {
        println!("skipped existing files:");
        for path in &summary.skipped_files {
            println!("  {}", path.display());
        }
        println!("use --force to overwrite existing template files");
    }

    Ok(())
}

fn gmsh_to_foam(args: Vec<String>) -> Result<(), String> {
    if args.is_empty() || args.iter().any(|arg| is_help(arg)) {
        print_gmsh_to_foam_usage();
        return Ok(());
    }

    let import = parse_gmsh_to_foam_args(&args)?;

    println!("Reading Gmsh mesh: {}", import.mesh_path.display());
    let mesh = read_msh22_ascii(&import.mesh_path).map_err(|error| error.to_string())?;
    println!(
        "Loaded {} points, {} supported volume cells, {} supported boundary faces",
        mesh.points.len(),
        mesh.cells.len(),
        mesh.boundary_faces.len()
    );

    println!("Writing OpenFOAM-like case: {}", import.case_dir.display());
    let summary = write_openfoam_case_with_options(
        &mesh,
        &import.case_dir,
        &import.mesh_path,
        &import.options,
    )
    .map_err(|error| error.to_string())?;

    println!("Wrote constant/polyMesh");
    println!(
        "faces: {} total, {} internal, {} boundary",
        summary.faces, summary.internal_faces, summary.boundary_faces
    );
    if summary.unmatched_boundary_faces > 0
        || summary.duplicate_boundary_faces > 0
        || summary.non_manifold_faces > 0
    {
        println!(
            "warnings: unmatchedBoundaryFaces={}, duplicateBoundaryFaces={}, nonManifoldFaces={}",
            summary.unmatched_boundary_faces,
            summary.duplicate_boundary_faces,
            summary.non_manifold_faces
        );
    }
    for patch in summary
        .patches
        .iter()
        .filter(|patch| patch.patch_type != "patch")
    {
        println!("patch type: {} -> {}", patch.name, patch.patch_type);
    }

    Ok(())
}

fn check_mesh(args: Vec<String>) -> Result<(), String> {
    if args.iter().any(|arg| is_help(arg)) {
        println!("usage: checkFerrumMesh [-case <caseDir>]");
        return Ok(());
    }

    let case_dir = parse_case_dir(&args, PathBuf::from("."))?;
    let summary = read_case_summary(&case_dir).map_err(|error| error.to_string())?;

    println!("Ferrum mesh check");
    println!("case: {}", summary.path.display());
    println!("points: {}", display_count(summary.points));
    println!("cells: {}", display_count(summary.cells));
    println!("faces: {}", display_count(summary.faces));
    println!("internal faces: {}", display_count(summary.internal_faces));
    println!("boundary faces: {}", display_count(summary.boundary_faces));
    println!("patches:");
    for patch in &summary.patches {
        println!("  {patch}");
    }
    println!("face zones:");
    for zone in &summary.face_zones {
        println!("  {zone}");
    }
    println!("cell zones:");
    for zone in &summary.cell_zones {
        println!("  {zone}");
    }

    let interfaces = build_interface_registry(&case_dir).map_err(|error| error.to_string())?;
    print_interface_registry(&interfaces);
    print_interface_config(&case_dir, &interfaces)?;
    print_backend_config(&case_dir)?;
    let fields = read_initial_fields(&case_dir).map_err(|error| error.to_string())?;
    print_initial_fields(&fields);
    print_field_boundary_validation(&case_dir, &fields);
    print_geometry_summary(&case_dir)?;
    print_patch_validation(&case_dir)?;

    let unmatched = summary.unmatched_boundary_faces.unwrap_or(0);
    let duplicate = summary.duplicate_boundary_faces.unwrap_or(0);
    let non_manifold = summary.non_manifold_faces.unwrap_or(0);
    if unmatched == 0 && duplicate == 0 && non_manifold == 0 {
        println!("topology warnings: none");
    } else {
        println!(
            "topology warnings: unmatchedBoundaryFaces={unmatched}, duplicateBoundaryFaces={duplicate}, nonManifoldFaces={non_manifold}"
        );
    }

    let regions = read_region_mesh_summaries(&case_dir).map_err(|error| error.to_string())?;
    if !regions.is_empty() {
        println!("region meshes:");
        for region in regions {
            println!(
                "  {}: points={}, cells={}, faces={}, internal={}, boundary={}, path={}",
                region.name,
                region.points,
                region.cells,
                region.faces,
                region.internal_faces,
                region.boundary_faces,
                region.path.display()
            );
            for patch in &region.patches {
                print_region_patch(patch);
            }
        }
    }

    Ok(())
}

fn split_mesh_regions(args: Vec<String>) -> Result<(), String> {
    if args.iter().any(|arg| is_help(arg)) {
        println!("usage: splitFerrumMeshRegions [-case <caseDir>] [-cellZones]");
        return Ok(());
    }

    let case_dir = parse_case_dir(&args, PathBuf::from("."))?;
    let summary = read_case_summary(&case_dir).map_err(|error| error.to_string())?;
    let use_cell_zones = args
        .iter()
        .any(|arg| arg == "-cellZones" || arg == "--cellZones");

    println!("Ferrum region split preview");
    println!("case: {}", summary.path.display());
    println!(
        "mode: {}",
        if use_cell_zones {
            "cellZones"
        } else {
            "cellZones (default)"
        }
    );
    println!("detected face zones:");
    for zone in &summary.face_zones {
        println!("  {zone}");
    }
    println!("detected regions:");
    for zone in &summary.cell_zones {
        println!("  {zone}");
    }
    let interfaces = build_interface_registry(&case_dir).map_err(|error| error.to_string())?;
    print_interface_registry(&interfaces);

    let split = split_regions_by_cell_zones(&case_dir).map_err(|error| error.to_string())?;
    println!("wrote region meshes:");
    for region in &split.regions {
        println!(
            "  {}: points={}, cells={}, faces={}, internal={}, boundary={}, path={}",
            region.name,
            region.points,
            region.cells,
            region.faces,
            region.internal_faces,
            region.boundary_faces,
            region.path.display()
        );
        for patch in &region.patches {
            print_region_patch(patch);
        }
    }
    Ok(())
}

fn solve_case(args: Vec<String>) -> Result<(), String> {
    if args.iter().any(|arg| is_help(arg)) {
        print_solver_usage();
        return Ok(());
    }

    let options = parse_solver_args(&args)?;
    let plan = build_solver_case_plan(&options.case_dir).map_err(|error| error.to_string())?;
    print_solver_case_plan(&plan);
    if options.runner_dry_run {
        let dry_run = build_solver_runner_dry_run(
            &plan,
            SolverRunnerDryRunOptions {
                max_steps: options.max_runner_steps,
            },
        );
        print_solver_runner_dry_run(&dry_run);
    }
    if let Some(solve) = &options.scalar_diffusion_solve {
        run_scalar_diffusion_solve(&plan, solve)?;
    }
    if let Some(solve) = &options.poiseuille_solve {
        run_poiseuille_solve(&plan, solve)?;
    }
    if let Some(solve) = &options.laminar_simple_solve {
        run_laminar_simple_solve(&plan, solve)?;
    }
    if let Some(path) = options.plan_json {
        write_solver_plan_json(&plan, &path).map_err(|error| {
            format!(
                "could not write solver plan JSON to {} ({error})",
                path.display()
            )
        })?;
        println!("wrote solver plan json: {}", path.display());
    }
    Ok(())
}

fn run_poiseuille_solve(plan: &SolverCasePlan, solve: &PoiseuilleSolveArgs) -> Result<(), String> {
    let options = resolve_poiseuille_options(plan, solve)?;
    let reference = poiseuille_reference(&options).map_err(|error| error.to_string())?;
    let diffusion_options =
        poiseuille_diffusion_options(&options).map_err(|error| error.to_string())?;
    let system = assemble_scalar_diffusion_system(&plan.runtime_data.mesh, &diffusion_options)
        .map_err(|error| error.to_string())?;

    let started = Instant::now();
    let report = match solve.linear_solver {
        ScalarDiffusionLinearSolver::Cg => conjugate_gradient_solve(
            &system.matrix,
            &system.rhs,
            None,
            ConjugateGradientOptions {
                max_iterations: solve.max_iterations,
                tolerance: solve.tolerance,
            },
        ),
        ScalarDiffusionLinearSolver::Jacobi => jacobi_solve(
            &system.matrix,
            &system.rhs,
            None,
            JacobiOptions {
                max_iterations: solve.max_iterations,
                tolerance: solve.tolerance,
                omega: 1.0,
            },
        ),
    }
    .map_err(|error| error.to_string())?;
    let wall_clock_seconds = started.elapsed().as_secs_f64();
    let summary =
        summarize_poiseuille_solution(&plan.runtime_data.mesh, &report.solution, &options)
            .map_err(|error| error.to_string())?;

    println!(
        "poiseuille solve: backend=cpu linearSolver={} cells={} nnz={} pressureDrop={} dynamicViscosity={} length={} diameter={} source={} wallPatches={} iterations={} converged={} residualNorm={} wallClockSeconds={:.6}",
        solve.linear_solver,
        system.stats.cells,
        system.matrix.nnz(),
        format_scientific(options.pressure_drop),
        format_scientific(options.dynamic_viscosity),
        format_scientific(options.length),
        format_scientific(options.diameter),
        format_scientific(reference.source),
        options.wall_patches.join(","),
        report.iterations,
        yes_no(report.converged),
        format_scientific(report.residual_norm),
        wall_clock_seconds
    );
    println!(
        "poiseuille result: meanVelocity={} analyticMeanVelocity={} relativeMeanVelocityError={} flowRate={} analyticFlowRate={} pressureDropFromMean={} minVelocity={} maxVelocity={}",
        format_scientific(summary.mean_velocity),
        format_scientific(summary.analytic_mean_velocity),
        format_scientific(summary.relative_mean_velocity_error),
        format_scientific(summary.flow_rate),
        format_scientific(summary.analytic_flow_rate),
        format_scientific(summary.pressure_drop_from_mean),
        format_scientific(summary.min_velocity),
        format_scientific(summary.max_velocity)
    );
    println!("poiseuille status: no field files written");

    Ok(())
}

fn resolve_poiseuille_options(
    plan: &SolverCasePlan,
    solve: &PoiseuilleSolveArgs,
) -> Result<PoiseuilleOptions, String> {
    let pressure_drop = solve
        .pressure_drop
        .or_else(|| {
            property_number(
                plan,
                "pipeBenchmark",
                Some("flowReference"),
                "expectedDeltaP",
            )
        })
        .ok_or_else(|| {
            "Poiseuille solve requires --pressureDrop or pipeBenchmark.flowReference.expectedDeltaP"
                .to_string()
        })?;
    let dynamic_viscosity = solve
        .dynamic_viscosity
        .or_else(|| property_number(plan, "transportProperties", None, "mu"))
        .or_else(|| property_number(plan, "pipeBenchmark", Some("water"), "mu"))
        .ok_or_else(|| {
            "Poiseuille solve requires --mu or transportProperties.mu/pipeBenchmark.water.mu"
                .to_string()
        })?;
    let length = solve
        .length
        .or_else(|| property_number(plan, "pipeBenchmark", Some("geometry"), "length"))
        .ok_or_else(|| {
            "Poiseuille solve requires --length or pipeBenchmark.geometry.length".to_string()
        })?;
    let diameter = solve
        .diameter
        .or_else(|| property_number(plan, "pipeBenchmark", Some("geometry"), "diameter"))
        .ok_or_else(|| {
            "Poiseuille solve requires --diameter or pipeBenchmark.geometry.diameter".to_string()
        })?;
    let wall_patches = if solve.wall_patches.is_empty() {
        vec!["wall".to_string()]
    } else {
        solve.wall_patches.clone()
    };

    Ok(PoiseuilleOptions {
        pressure_drop,
        dynamic_viscosity,
        length,
        diameter,
        wall_patches,
    })
}

fn run_laminar_simple_solve(
    plan: &SolverCasePlan,
    solve: &LaminarSimpleSolveArgs,
) -> Result<(), String> {
    let fields = read_initial_fields(&plan.case_dir).map_err(|error| error.to_string())?;
    let options = resolve_laminar_simple_options(plan, solve)?;

    let started = Instant::now();
    let report = solve_laminar_simple(&plan.runtime_data, &fields, &options)
        .map_err(|error| error.to_string())?;
    let wall_clock_seconds = started.elapsed().as_secs_f64();

    println!(
        "laminarSimple solve: backend=cpu linearSolver={} momentumLinearSolver={} momentumPreconditioner={} pressureLinearSolver={} pressurePreconditioner={} cells={} faces={} simpleIterations={} minSimpleIterations={} converged={} initialContinuityL2={} finalContinuityL2={} simpleTolerance={} pressureDropTolerance={} fieldChangeTolerance={} momentumResidualNorm={} pressureCorrectionResidualNorm={} momentumLinearIterations={} pressureLinearIterations={} wallClockSeconds={:.6}",
        options.linear_solver,
        options.momentum_linear_solver,
        options.momentum_preconditioner,
        options.pressure_linear_solver,
        options.pressure_preconditioner,
        report.cells,
        report.faces,
        report.simple_iterations,
        options.min_simple_iterations,
        yes_no(report.converged),
        format_scientific(report.initial_continuity.l2_norm),
        format_scientific(report.final_continuity.l2_norm),
        format_scientific(options.simple_tolerance),
        format_scientific(options.pressure_drop_tolerance),
        format_scientific(options.field_change_tolerance),
        format_scientific(report.final_momentum_residual_norm),
        format_scientific(report.final_pressure_correction_residual_norm),
        report.total_momentum_linear_iterations,
        report.total_pressure_linear_iterations,
        wall_clock_seconds
    );
    println!(
        "laminarSimple result: meanVelocity={} analyticMeanVelocity={} relativeMeanVelocityError={} flowRate={} analyticFlowRate={} pressureDropFromMean={} relativePressureDropError={} pressureDropFromField={} minAxialVelocity={} maxAxialVelocity={}",
        format_scientific(report.solution.mean_velocity),
        format_scientific(report.solution.analytic_mean_velocity),
        format_scientific(report.solution.relative_mean_velocity_error),
        format_scientific(report.solution.flow_rate),
        format_scientific(report.solution.analytic_flow_rate),
        format_scientific(report.solution.pressure_drop_from_mean),
        format_scientific(report.solution.relative_pressure_drop_error),
        format_optional_scientific(report.solution.pressure_drop_from_field),
        format_scientific(report.solution.min_axial_velocity),
        format_scientific(report.solution.max_axial_velocity)
    );
    println!(
        "laminarSimple operators: phiMin={} phiMax={} phiSumAbs={} gradPL2={} divPhiUL2={} velocityFixedValueFaces={} velocityZeroGradientFaces={} pressureFixedValueFaces={} pressureZeroGradientFaces={}",
        format_scientific(report.operator_summary.phi_min),
        format_scientific(report.operator_summary.phi_max),
        format_scientific(report.operator_summary.phi_sum_abs),
        format_scientific(report.operator_summary.grad_p_l2_norm),
        format_scientific(report.operator_summary.div_phi_u_l2_norm),
        report.boundary_summary.velocity_fixed_value_faces,
        report.boundary_summary.velocity_zero_gradient_faces,
        report.boundary_summary.pressure_fixed_value_faces,
        report.boundary_summary.pressure_zero_gradient_faces
    );
    println!("laminarSimple status: no field files written");

    if let Some(path) = &solve.report_json {
        write_laminar_simple_report_json(plan, &options, &report, wall_clock_seconds, path)
            .map_err(|error| {
                format!(
                    "could not write laminar SIMPLE report JSON to {} ({error})",
                    path.display()
                )
            })?;
        println!("wrote laminar SIMPLE report json: {}", path.display());
    }
    if let Some(path) = &solve.report_markdown {
        write_laminar_simple_report_markdown(plan, &options, &report, wall_clock_seconds, path)
            .map_err(|error| {
                format!(
                    "could not write laminar SIMPLE report Markdown to {} ({error})",
                    path.display()
                )
            })?;
        println!("wrote laminar SIMPLE report markdown: {}", path.display());
    }

    Ok(())
}

fn resolve_laminar_simple_options(
    plan: &SolverCasePlan,
    solve: &LaminarSimpleSolveArgs,
) -> Result<LaminarSimpleOptions, String> {
    let density = solve
        .density
        .or_else(|| property_number(plan, "transportProperties", None, "rho"))
        .or_else(|| property_number(plan, "pipeBenchmark", Some("water"), "rho"))
        .ok_or_else(|| {
            "Laminar SIMPLE solve requires --rho or transportProperties.rho/pipeBenchmark.water.rho"
                .to_string()
        })?;
    let dynamic_viscosity = solve
        .dynamic_viscosity
        .or_else(|| property_number(plan, "transportProperties", None, "mu"))
        .or_else(|| property_number(plan, "pipeBenchmark", Some("water"), "mu"))
        .ok_or_else(|| {
            "Laminar SIMPLE solve requires --mu or transportProperties.mu/pipeBenchmark.water.mu"
                .to_string()
        })?;
    let pressure_drop = solve
        .pressure_drop
        .or_else(|| {
            property_number(
                plan,
                "pipeBenchmark",
                Some("flowReference"),
                "expectedDeltaP",
            )
        })
        .ok_or_else(|| {
            "Laminar SIMPLE solve requires --pressureDrop or pipeBenchmark.flowReference.expectedDeltaP"
                .to_string()
        })?;
    let length = solve
        .length
        .or_else(|| property_number(plan, "pipeBenchmark", Some("geometry"), "length"))
        .ok_or_else(|| {
            "Laminar SIMPLE solve requires --length or pipeBenchmark.geometry.length".to_string()
        })?;
    let diameter = solve
        .diameter
        .or_else(|| property_number(plan, "pipeBenchmark", Some("geometry"), "diameter"))
        .ok_or_else(|| {
            "Laminar SIMPLE solve requires --diameter or pipeBenchmark.geometry.diameter"
                .to_string()
        })?;

    let linear_tolerance = solve.linear_tolerance.unwrap_or(1.0e-10);
    let max_linear_iterations = solve.max_linear_iterations.unwrap_or(10_000);
    let momentum_linear_tolerance = solve
        .momentum_linear_tolerance
        .or(solve.linear_tolerance)
        .or_else(|| fv_solution_number(plan, "solvers.U", "tolerance"))
        .unwrap_or(1.0e-10);
    let pressure_linear_tolerance = solve
        .pressure_linear_tolerance
        .or(solve.linear_tolerance)
        .or_else(|| fv_solution_number(plan, "solvers.p", "tolerance"))
        .unwrap_or(1.0e-10);
    let momentum_max_linear_iterations = solve
        .momentum_max_linear_iterations
        .or(solve.max_linear_iterations)
        .or_else(|| fv_solution_usize(plan, "solvers.U", "maxIter"))
        .unwrap_or(10_000);
    let pressure_max_linear_iterations = solve
        .pressure_max_linear_iterations
        .or(solve.max_linear_iterations)
        .or_else(|| fv_solution_usize(plan, "solvers.p", "maxIter"))
        .unwrap_or(10_000);
    let linear_solver = solve
        .linear_solver
        .unwrap_or(LaminarSimpleLinearSolver::Jacobi);
    let momentum_linear_solver = solve
        .momentum_linear_solver
        .or(solve.linear_solver)
        .or_else(|| fv_solution_laminar_solver(plan, "solvers.U", "solver"))
        .unwrap_or(LaminarSimpleLinearSolver::Jacobi);
    let pressure_linear_solver = solve
        .pressure_linear_solver
        .or(solve.linear_solver)
        .or_else(|| fv_solution_laminar_solver(plan, "solvers.p", "solver"))
        .unwrap_or(LaminarSimpleLinearSolver::Jacobi);
    let momentum_preconditioner = solve
        .momentum_preconditioner
        .or_else(|| {
            if momentum_linear_solver == LaminarSimpleLinearSolver::Pcg {
                fv_solution_laminar_preconditioner(plan, "solvers.U", "preconditioner")
            } else {
                None
            }
        })
        .unwrap_or(LaminarSimplePreconditioner::None);
    let pressure_preconditioner = solve
        .pressure_preconditioner
        .or_else(|| {
            if pressure_linear_solver == LaminarSimpleLinearSolver::Pcg {
                fv_solution_laminar_preconditioner(plan, "solvers.p", "preconditioner")
            } else {
                None
            }
        })
        .unwrap_or(LaminarSimplePreconditioner::None);
    let min_simple_iterations = solve
        .min_simple_iterations
        .or_else(|| fv_solution_usize(plan, "SIMPLE", "minSimpleIterations"))
        .unwrap_or_else(|| {
            if solve.max_simple_iterations > 1 {
                2
            } else {
                1
            }
        });
    let pressure_drop_tolerance = solve
        .pressure_drop_tolerance
        .or_else(|| fv_solution_number(plan, "SIMPLE", "pressureDropTolerance"))
        .unwrap_or(0.02);
    let field_change_tolerance = solve
        .field_change_tolerance
        .or_else(|| fv_solution_number(plan, "SIMPLE", "fieldChangeTolerance"))
        .unwrap_or(0.01);
    let momentum_residual_control = fv_solution_number(plan, "SIMPLE.residualControl", "U");
    let pressure_residual_control = fv_solution_number(plan, "SIMPLE.residualControl", "p");

    Ok(LaminarSimpleOptions {
        density,
        dynamic_viscosity,
        pressure_drop,
        length,
        diameter,
        inlet_patch: solve.inlet_patch.clone(),
        outlet_patch: solve.outlet_patch.clone(),
        linear_solver,
        momentum_linear_solver,
        pressure_linear_solver,
        momentum_preconditioner,
        pressure_preconditioner,
        linear_tolerance,
        max_linear_iterations,
        momentum_linear_tolerance,
        pressure_linear_tolerance,
        momentum_max_linear_iterations,
        pressure_max_linear_iterations,
        max_simple_iterations: solve.max_simple_iterations,
        min_simple_iterations,
        simple_tolerance: solve.simple_tolerance,
        pressure_drop_tolerance,
        field_change_tolerance,
        momentum_residual_control,
        pressure_residual_control,
        velocity_relaxation: solve
            .velocity_relaxation
            .or_else(|| fv_solution_number(plan, "relaxationFactors.equations", "U"))
            .unwrap_or(0.7),
        pressure_relaxation: solve
            .pressure_relaxation
            .or_else(|| fv_solution_number(plan, "relaxationFactors.fields", "p"))
            .unwrap_or(0.3),
    })
}

fn property_number(
    plan: &SolverCasePlan,
    dictionary: &str,
    section: Option<&str>,
    key: &str,
) -> Option<f64> {
    plan.properties
        .entries
        .iter()
        .find(|entry| {
            entry.dictionary == dictionary
                && entry.section.as_deref() == section
                && entry.key == key
        })
        .and_then(|entry| last_number(&entry.value))
}

fn fv_solution_number(plan: &SolverCasePlan, section: &str, key: &str) -> Option<f64> {
    numerics_dictionary_number(&plan.numerics.fv_solution, section, key)
}

fn fv_solution_laminar_solver(
    plan: &SolverCasePlan,
    section: &str,
    key: &str,
) -> Option<LaminarSimpleLinearSolver> {
    numerics_dictionary_value(&plan.numerics.fv_solution, section, key)
        .and_then(parse_openfoam_laminar_solver)
}

fn fv_solution_laminar_preconditioner(
    plan: &SolverCasePlan,
    section: &str,
    key: &str,
) -> Option<LaminarSimplePreconditioner> {
    numerics_dictionary_value(&plan.numerics.fv_solution, section, key)
        .and_then(parse_openfoam_laminar_preconditioner)
}

fn fv_solution_usize(plan: &SolverCasePlan, section: &str, key: &str) -> Option<usize> {
    numerics_dictionary_usize(&plan.numerics.fv_solution, section, key)
}

fn numerics_dictionary_value<'a>(
    dictionary: &'a SolverNumericsDictionaryPlan,
    section: &str,
    key: &str,
) -> Option<&'a str> {
    dictionary
        .entries
        .iter()
        .find(|entry| entry.section == section && entry.key == key)
        .map(|entry| entry.value.as_str())
}

fn numerics_dictionary_number(
    dictionary: &SolverNumericsDictionaryPlan,
    section: &str,
    key: &str,
) -> Option<f64> {
    dictionary
        .entries
        .iter()
        .find(|entry| entry.section == section && entry.key == key)
        .and_then(|entry| last_number(&entry.value))
}

fn numerics_dictionary_usize(
    dictionary: &SolverNumericsDictionaryPlan,
    section: &str,
    key: &str,
) -> Option<usize> {
    dictionary
        .entries
        .iter()
        .find(|entry| entry.section == section && entry.key == key)
        .and_then(|entry| last_usize(&entry.value))
}

fn last_number(value: &str) -> Option<f64> {
    value.split_whitespace().rev().find_map(|token| {
        token
            .trim_matches(|ch| ch == '[' || ch == ']')
            .parse::<f64>()
            .ok()
    })
}

fn last_usize(value: &str) -> Option<usize> {
    value.split_whitespace().rev().find_map(|token| {
        token
            .trim_matches(|ch| ch == '[' || ch == ']')
            .parse::<usize>()
            .ok()
    })
}

fn parse_openfoam_laminar_solver(value: &str) -> Option<LaminarSimpleLinearSolver> {
    match value.trim() {
        "PCG" | "pcg" => Some(LaminarSimpleLinearSolver::Pcg),
        "CG" | "cg" => Some(LaminarSimpleLinearSolver::Cg),
        "smoothSolver" | "smoothSolver;" | "Jacobi" | "jacobi" => {
            Some(LaminarSimpleLinearSolver::Jacobi)
        }
        _ => None,
    }
}

fn parse_openfoam_laminar_preconditioner(value: &str) -> Option<LaminarSimplePreconditioner> {
    match value.trim() {
        "none" | "None" => Some(LaminarSimplePreconditioner::None),
        "DIC" | "FDIC" | "diagonal" | "Diagonal" => Some(LaminarSimplePreconditioner::Diagonal),
        _ => None,
    }
}

fn run_scalar_diffusion_solve(
    plan: &SolverCasePlan,
    solve: &ScalarDiffusionSolveArgs,
) -> Result<(), String> {
    let fields = read_initial_fields(&plan.case_dir).map_err(|error| error.to_string())?;
    let field = find_field_selection(&fields, &solve.field)?;
    let options = scalar_diffusion_options_from_field(field, solve.diffusivity, solve.source)
        .map_err(|error| error.to_string())?;
    let system = assemble_scalar_diffusion_system(&plan.runtime_data.mesh, &options)
        .map_err(|error| error.to_string())?;
    let initial = runtime_initial_guess(plan, field);

    let started = Instant::now();
    let report = match solve.linear_solver {
        ScalarDiffusionLinearSolver::Cg => conjugate_gradient_solve(
            &system.matrix,
            &system.rhs,
            initial,
            ConjugateGradientOptions {
                max_iterations: solve.max_iterations,
                tolerance: solve.tolerance,
            },
        ),
        ScalarDiffusionLinearSolver::Jacobi => jacobi_solve(
            &system.matrix,
            &system.rhs,
            initial,
            JacobiOptions {
                max_iterations: solve.max_iterations,
                tolerance: solve.tolerance,
                omega: 1.0,
            },
        ),
    }
    .map_err(|error| error.to_string())?;
    let wall_clock_seconds = started.elapsed().as_secs_f64();
    let solution = summarize_scalar_solution(&report.solution);

    println!(
        "scalar diffusion solve: field={} backend=cpu linearSolver={} cells={} nnz={} diffusivity={} source={} fixedValueFaces={} zeroGradientFaces={} constraintFaces={} initialGuess={} iterations={} converged={} residualNorm={} wallClockSeconds={:.6}",
        field_label(field),
        solve.linear_solver,
        system.stats.cells,
        system.matrix.nnz(),
        format_scientific(solve.diffusivity),
        format_scientific(solve.source),
        system.stats.fixed_value_faces,
        system.stats.zero_gradient_faces,
        system.stats.constraint_faces,
        if initial.is_some() { "field" } else { "zero" },
        report.iterations,
        yes_no(report.converged),
        format_scientific(report.residual_norm),
        wall_clock_seconds
    );
    println!(
        "scalar diffusion solution: min={} max={} mean={}",
        format_scientific(solution.min),
        format_scientific(solution.max),
        format_scientific(solution.mean)
    );
    println!("scalar diffusion status: no field files written");

    Ok(())
}

fn find_field_selection<'a>(
    fields: &'a InitialFieldSet,
    selection: &str,
) -> Result<&'a FieldFile, String> {
    let matches = fields
        .fields
        .iter()
        .filter(|field| field_matches_selection(field, selection))
        .collect::<Vec<_>>();

    match matches.as_slice() {
        [field] => Ok(field),
        [] => Err(format!(
            "field '{selection}' was not found below {}",
            fields.case_dir.join("0").display()
        )),
        _ => Err(format!(
            "field '{selection}' is ambiguous; use '<region>/<field>'"
        )),
    }
}

fn field_matches_selection(field: &FieldFile, selection: &str) -> bool {
    if let Some((region, name)) = selection.split_once('/') {
        field.region.as_deref() == Some(region) && field.name == name
    } else {
        field.name == selection
    }
}

fn runtime_initial_guess<'a>(plan: &'a SolverCasePlan, field: &FieldFile) -> Option<&'a [f64]> {
    plan.runtime_data
        .fields
        .iter()
        .find(|buffer| {
            buffer.region == field.region
                && buffer.name == field.name
                && buffer.components == 1
                && buffer.values.len() == plan.runtime_data.mesh.cells
        })
        .map(|buffer| buffer.values.as_slice())
}

fn field_label(field: &FieldFile) -> String {
    if let Some(region) = &field.region {
        format!("{region}/{}", field.name)
    } else {
        field.name.clone()
    }
}

fn summarize_scalar_solution(values: &[f64]) -> ScalarSolutionSummary {
    if values.is_empty() {
        return ScalarSolutionSummary {
            min: 0.0,
            max: 0.0,
            mean: 0.0,
        };
    }

    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    let mut sum = 0.0;
    for value in values {
        min = min.min(*value);
        max = max.max(*value);
        sum += *value;
    }

    ScalarSolutionSummary {
        min,
        max,
        mean: sum / values.len() as f64,
    }
}

fn print_solver_case_plan(plan: &SolverCasePlan) {
    println!("Ferrum solver preflight");
    println!("case: {}", plan.case_dir.display());
    println!(
        "control: application={} startFrom={} startTime={} stopAt={} endTime={} deltaT={} writeControl={} writeInterval={}",
        plan.control.application,
        plan.control.start_from,
        format_optional_number(plan.control.start_time),
        plan.control.stop_at,
        format_optional_number(plan.control.end_time),
        format_optional_number(plan.control.delta_t),
        plan.control.write_control,
        format_optional_number(plan.control.write_interval)
    );
    println!(
        "mesh: dimensionality={} points={} cells={} faces={} internal={} boundary={} patches={}",
        plan.mesh.dimensionality,
        plan.mesh.points,
        plan.mesh.cells,
        plan.mesh.faces,
        plan.mesh.internal_faces,
        plan.mesh.boundary_faces,
        plan.mesh.patches
    );
    println!(
        "special patches: empty={} wedge={} symmetryPlane={}",
        plan.mesh.empty_patches, plan.mesh.wedge_patches, plan.mesh.symmetry_patches
    );
    if plan.mesh.region_meshes.is_empty() {
        println!("region meshes: none");
    } else {
        println!("region meshes:");
        for region in &plan.mesh.region_meshes {
            println!(
                "  {}: cells={} patches={}",
                region.name, region.cells, region.patches
            );
        }
    }
    if plan.fields.fields.is_empty() {
        println!("fields: none");
    } else {
        println!("fields:");
        for field in &plan.fields.fields {
            let name = if let Some(region) = &field.region {
                format!("{region}/{}", field.name)
            } else {
                field.name.clone()
            };
            println!(
                "  {}: class={} boundaryPatches={}",
                name,
                field.class_name.as_deref().unwrap_or("unknown"),
                field.boundary_patches
            );
        }
    }
    print_solver_state_plan(&plan.state);
    print_solver_runtime_data(&plan.runtime_data);
    print_diffusion_assembly_capabilities();
    print_linear_solver_capabilities();
    print_solver_properties(&plan.properties);
    print_solver_numerics_dictionary("fvSchemes", &plan.numerics.fv_schemes);
    print_solver_numerics_dictionary("fvSolution", &plan.numerics.fv_solution);
    println!(
        "interfaces: registry={} discovered={} boundaryFaceZones={} config={} configured={}",
        yes_no(plan.interfaces.registry_available),
        plan.interfaces.discovered_interfaces,
        plan.interfaces.boundary_face_zones,
        yes_no(plan.interfaces.config_present),
        plan.interfaces.configured_interfaces
    );
    print_solver_backend_plan(&plan.backends);
    print_solver_run_plan(&plan.run);
    if plan.warnings.is_empty() {
        println!("preflight warnings: none");
    } else {
        println!("preflight warnings:");
        for warning in &plan.warnings {
            println!("  {warning}");
        }
    }
    println!(
        "solver execution: CPU scalar diffusion, Poiseuille, and laminar SIMPLE kernels are available; GPU equation kernels are planned"
    );
}

fn print_linear_solver_capabilities() {
    let capabilities = linear_solver_capabilities();
    println!(
        "linear solvers: cpuCsr={} cpuJacobi={} cpuCg={} cpuPcg={} cpuDiagonalPreconditioner={} gpuLinearSolvers={}",
        yes_no(capabilities.cpu_csr),
        yes_no(capabilities.cpu_jacobi),
        yes_no(capabilities.cpu_conjugate_gradient),
        yes_no(capabilities.cpu_preconditioned_conjugate_gradient),
        yes_no(capabilities.cpu_diagonal_preconditioner),
        yes_no(capabilities.gpu_linear_solvers)
    );
}

fn print_diffusion_assembly_capabilities() {
    let capabilities = diffusion_assembly_capabilities();
    println!(
        "equation assembly: cpuScalarDiffusion={} cpuPoisson={} fixedValue={} zeroGradient={} gpuAssembly={}",
        yes_no(capabilities.cpu_scalar_diffusion),
        yes_no(capabilities.cpu_poisson),
        yes_no(capabilities.fixed_value_boundary),
        yes_no(capabilities.zero_gradient_boundary),
        yes_no(capabilities.gpu_assembly)
    );
}

fn print_solver_properties(plan: &SolverPropertiesPlan) {
    println!(
        "properties: dictionaries={} entries={}",
        plan.dictionaries.len(),
        plan.entries.len()
    );
    for dictionary in &plan.dictionaries {
        let label = if let Some(region) = &dictionary.region {
            format!("{region}/{}", dictionary.name)
        } else {
            dictionary.name.clone()
        };
        println!(
            "  {}: sections={} entries={}",
            label, dictionary.sections, dictionary.entries
        );
    }
    for entry in &plan.entries {
        let path = if let Some(section) = &entry.section {
            format!("{}.{}.{}", entry.dictionary, section, entry.key)
        } else {
            format!("{}.{}", entry.dictionary, entry.key)
        };
        println!("    {path}={}", entry.value);
    }
}

fn print_solver_state_plan(plan: &SolverStatePlan) {
    let cpu_capable = plan
        .fields
        .iter()
        .filter(|field| field.storage.cpu_capable)
        .count();
    let gpu_capable = plan
        .fields
        .iter()
        .filter(|field| field.storage.gpu_capable)
        .count();
    let bytes_f64 = plan
        .fields
        .iter()
        .filter_map(|field| field.storage.bytes_f64)
        .sum::<usize>();
    let cpu_buffers = plan
        .fields
        .iter()
        .filter(|field| field.cpu_buffer.materializable)
        .count();
    println!(
        "solver state: fields={} cpuCapable={} gpuCapable={} cpuBuffers={} bytesF64={}",
        plan.fields.len(),
        cpu_capable,
        gpu_capable,
        cpu_buffers,
        bytes_f64
    );
    for field in &plan.fields {
        let name = if let Some(region) = &field.region {
            format!("{region}/{}", field.name)
        } else {
            field.name.clone()
        };
        println!(
            "  {}: class={} kind={} meshCells={} internal={} values={} expected={} valid={} components={} scalarSlots={} bytesF64={} uniform={} loadedScalars={} boundaryPatches={}/{} cpu={} gpu={} storage={} cpuBuffer={} cpuBufferStatus={}",
            name,
            field.class_name.as_deref().unwrap_or("unknown"),
            field.kind,
            format_optional_usize(field.mesh_cells),
            field.internal_field.kind,
            format_optional_usize(field.internal_field.value_count),
            format_optional_usize(field.internal_field.expected_count),
            format_optional_bool(field.internal_field.valid_count),
            format_optional_usize(field.storage.components),
            format_optional_usize(field.storage.scalar_slots),
            format_optional_usize(field.storage.bytes_f64),
            format_optional_f64_list(field.internal_field.uniform_components.as_deref()),
            format_optional_usize(
                field
                    .internal_field
                    .nonuniform_values
                    .as_ref()
                    .map(Vec::len)
            ),
            field.boundary_patches,
            format_optional_usize(field.mesh_boundary_patches),
            yes_no(field.storage.cpu_capable),
            yes_no(field.storage.gpu_capable),
            field.storage.status,
            yes_no(field.cpu_buffer.materializable),
            field.cpu_buffer.status
        );
    }
    for warning in &plan.warnings {
        println!("solver state warning: {warning}");
    }
}

fn print_solver_runtime_data(runtime: &SolverRuntimeData) {
    let field_scalars = runtime
        .fields
        .iter()
        .map(|field| field.scalar_slots)
        .sum::<usize>();
    let field_bytes = runtime
        .fields
        .iter()
        .map(|field| field.bytes_f64)
        .sum::<usize>();
    println!(
        "runtime data: meshGeometry=yes fields={} fieldScalars={} fieldBytesF64={} warnings={}",
        runtime.fields.len(),
        field_scalars,
        field_bytes,
        runtime.warnings.len()
    );
    println!(
        "runtime mesh: points={} cells={} faces={} internal={} boundary={} ownerLabels={} neighbourLabels={} patches={}",
        runtime.mesh.points,
        runtime.mesh.cells,
        runtime.mesh.faces,
        runtime.mesh.internal_faces,
        runtime.mesh.boundary_faces,
        runtime.mesh.owner.len(),
        runtime
            .mesh
            .neighbour
            .iter()
            .filter(|cell| cell.is_some())
            .count(),
        runtime.mesh.patches.len()
    );
    println!(
        "runtime geometry arrays: cellCentres={} faceCentres={} faceAreaVectors={} cellVolumes={} totalVolume={} minCellVolume={} maxCellVolume={} minFaceArea={} maxFaceArea={} nonPositiveCellVolumes={}",
        runtime.mesh.cell_centres.len(),
        runtime.mesh.face_centres.len(),
        runtime.mesh.face_area_vectors.len(),
        runtime.mesh.cell_volumes.len(),
        format_scientific(runtime.mesh.total_cell_volume),
        format_scientific(runtime.mesh.min_cell_volume),
        format_scientific(runtime.mesh.max_cell_volume),
        format_scientific(runtime.mesh.min_face_area),
        format_scientific(runtime.mesh.max_face_area),
        runtime.mesh.non_positive_cell_volumes
    );
    println!("runtime patches:");
    for patch in &runtime.mesh.patches {
        println!(
            "  {}: type={} faces={} startFace={}",
            patch.name, patch.patch_type, patch.faces, patch.start_face
        );
    }
    if runtime.fields.is_empty() {
        println!("runtime field buffers: none");
    } else {
        println!("runtime field buffers:");
        for field in &runtime.fields {
            let name = if let Some(region) = &field.region {
                format!("{region}/{}", field.name)
            } else {
                field.name.clone()
            };
            println!(
                "  {}: kind={} components={} scalarSlots={} bytesF64={} values={}",
                name,
                field.kind,
                field.components,
                field.scalar_slots,
                field.bytes_f64,
                field.values.len()
            );
        }
    }
    for warning in &runtime.warnings {
        println!("runtime data warning: {warning}");
    }
}

fn print_solver_numerics_dictionary(name: &str, plan: &SolverNumericsDictionaryPlan) {
    println!(
        "{}: present={} sections={} entries={}",
        name,
        yes_no(plan.present),
        plan.sections.len(),
        plan.entries.len()
    );
    for entry in &plan.entries {
        println!("  {}.{}={}", entry.section, entry.key, entry.value);
    }
}

fn print_solver_backend_plan(plan: &SolverBackendPlan) {
    println!(
        "backend plan: config={} default={} usesCpu={} usesGpu={} mixed={}",
        yes_no(plan.config_present),
        plan.default,
        plan.uses_cpu,
        plan.uses_gpu,
        plan.mixed_execution
    );
    println!(
        "cpu resources: cpus={} coresPerCpu={} threads={} threadPinning={} numa={}",
        plan.cpu.cpus,
        plan.cpu.cores_per_cpu,
        plan.cpu.threads,
        plan.cpu.thread_pinning,
        plan.cpu.numa
    );
    println!(
        "gpu resources: backend={} devices={} multiGpu={} precision={}",
        plan.gpu.backend,
        format_devices(&plan.gpu.devices),
        plan.gpu.multi_gpu,
        plan.gpu.precision
    );
    if plan.stages.is_empty() {
        println!("backend stages: default only");
        return;
    }

    println!("backend stages:");
    for stage in &plan.stages {
        println!("  {}.{}={}", stage.section, stage.step, stage.choice);
    }
}

fn print_solver_run_plan(plan: &SolverRunPlan) {
    println!(
        "run schedule: stopAt={} startTime={} endTime={} deltaT={} estimatedSteps={} writeControl={} writeInterval={} estimatedWrites={}",
        plan.stop_at,
        format_optional_number(plan.start_time),
        format_optional_number(plan.end_time),
        format_optional_number(plan.delta_t),
        format_optional_usize(plan.estimated_steps),
        plan.write_control,
        format_optional_number(plan.write_interval),
        format_optional_usize(plan.estimated_write_events)
    );
    if plan.stages.is_empty() {
        println!("run stages: none");
        return;
    }

    println!("run stages:");
    for stage in &plan.stages {
        println!(
            "  {}.{}={} ({})",
            stage.section, stage.step, stage.choice, stage.source
        );
    }
}

fn print_solver_runner_dry_run(dry_run: &SolverRunnerDryRun) {
    println!(
        "runner dry-run: plannedSteps={} previewSteps={} maxPreviewSteps={} stageCount={} previewWriteEvents={} truncated={}",
        format_optional_usize(dry_run.planned_steps),
        dry_run.preview_steps,
        dry_run.max_steps,
        dry_run.stage_count,
        dry_run.preview_write_events,
        dry_run.truncated
    );
    print_solver_runner_state(&dry_run.state);
    println!(
        "runner runtime: cpuRequested={} cpuHandle={} cpuLinearSolvers={} cpuKernels={} cpuThreads={} gpuRequested={} gpuHandle={} gpuLinearSolvers={} gpuKernels={} gpuBackend={} gpuDevices={} gpuPrecision={}",
        yes_no(dry_run.runtime.cpu.requested),
        dry_run.runtime.cpu.handle,
        yes_no(dry_run.runtime.cpu.linear_solvers_available),
        yes_no(dry_run.runtime.cpu.kernels_available),
        dry_run.runtime.cpu.threads,
        yes_no(dry_run.runtime.gpu.requested),
        dry_run.runtime.gpu.handle,
        yes_no(dry_run.runtime.gpu.linear_solvers_available),
        yes_no(dry_run.runtime.gpu.kernels_available),
        dry_run.runtime.gpu.backend,
        format_devices(&dry_run.runtime.gpu.devices),
        dry_run.runtime.gpu.precision
    );
    for warning in &dry_run.runtime.warnings {
        println!("runner runtime warning: {warning}");
    }
    for warning in &dry_run.warnings {
        println!("runner dry-run warning: {warning}");
    }
    for event in &dry_run.events {
        match event {
            SolverRunnerDryRunEvent::StepStart { step, time } => {
                println!("  step {step}: time={}", format_optional_number(*time));
            }
            SolverRunnerDryRunEvent::Stage {
                step,
                section,
                stage,
                choice,
                source,
                dispatch,
            } => {
                println!(
                    "    step {step} stage {section}.{stage}: backend={choice} source={source} runtimeTarget={} runtimeHandle={} executable={} status={}",
                    dispatch.target,
                    dispatch.handle,
                    yes_no(dispatch.executable),
                    dispatch.status
                );
            }
            SolverRunnerDryRunEvent::Write { step, time } => {
                println!(
                    "    step {step} write: time={} action=planned-output",
                    format_optional_number(*time)
                );
            }
        }
    }
    println!("runner dry-run status: no fields updated; no equations solved");
}

fn print_solver_runner_state(plan: &SolverStatePlan) {
    let cpu_capable = plan
        .fields
        .iter()
        .filter(|field| field.storage.cpu_capable)
        .count();
    let gpu_capable = plan
        .fields
        .iter()
        .filter(|field| field.storage.gpu_capable)
        .count();
    let bytes_f64 = plan
        .fields
        .iter()
        .filter_map(|field| field.storage.bytes_f64)
        .sum::<usize>();
    let cpu_buffers = plan
        .fields
        .iter()
        .filter(|field| field.cpu_buffer.materializable)
        .count();
    println!(
        "runner state: fields={} cpuCapable={} gpuCapable={} cpuBuffers={} bytesF64={}",
        plan.fields.len(),
        cpu_capable,
        gpu_capable,
        cpu_buffers,
        bytes_f64
    );
    for field in &plan.fields {
        let name = if let Some(region) = &field.region {
            format!("{region}/{}", field.name)
        } else {
            field.name.clone()
        };
        println!(
            "  field {}: kind={} internal={} values={} expected={} components={} scalarSlots={} bytesF64={} uniform={} loadedScalars={} cpu={} gpu={} storage={} cpuBuffer={} cpuBufferStatus={}",
            name,
            field.kind,
            field.internal_field.kind,
            format_optional_usize(field.internal_field.value_count),
            format_optional_usize(field.internal_field.expected_count),
            format_optional_usize(field.storage.components),
            format_optional_usize(field.storage.scalar_slots),
            format_optional_usize(field.storage.bytes_f64),
            format_optional_f64_list(field.internal_field.uniform_components.as_deref()),
            format_optional_usize(
                field
                    .internal_field
                    .nonuniform_values
                    .as_ref()
                    .map(Vec::len)
            ),
            yes_no(field.storage.cpu_capable),
            yes_no(field.storage.gpu_capable),
            field.storage.status,
            yes_no(field.cpu_buffer.materializable),
            field.cpu_buffer.status
        );
    }
    for warning in &plan.warnings {
        println!("runner state warning: {warning}");
    }
}

fn write_solver_plan_json(plan: &SolverCasePlan, path: &Path) -> std::io::Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    writeln!(writer, "{{")?;
    write_json_key(&mut writer, 2, "caseDir")?;
    write_json_string(&mut writer, &plan.case_dir.display().to_string())?;
    writeln!(writer, ",")?;
    write_json_control(&mut writer, plan)?;
    writeln!(writer, ",")?;
    write_json_mesh(&mut writer, &plan.mesh)?;
    writeln!(writer, ",")?;
    write_json_fields(&mut writer, &plan.fields)?;
    writeln!(writer, ",")?;
    write_json_solver_state(&mut writer, &plan.state)?;
    writeln!(writer, ",")?;
    write_json_runtime_data(&mut writer, &plan.runtime_data)?;
    writeln!(writer, ",")?;
    write_json_properties(&mut writer, &plan.properties)?;
    writeln!(writer, ",")?;
    write_json_numerics(&mut writer, &plan.numerics)?;
    writeln!(writer, ",")?;
    write_json_interfaces(&mut writer, &plan.interfaces)?;
    writeln!(writer, ",")?;
    write_json_backends(&mut writer, &plan.backends)?;
    writeln!(writer, ",")?;
    write_json_run_plan(&mut writer, &plan.run)?;
    writeln!(writer, ",")?;
    write_json_key(&mut writer, 2, "warnings")?;
    write_json_string_array(&mut writer, &plan.warnings)?;
    writeln!(writer)?;
    writeln!(writer, "}}")?;

    writer.flush()
}

fn write_laminar_simple_report_json(
    plan: &SolverCasePlan,
    options: &LaminarSimpleOptions,
    report: &LaminarSimpleReport,
    wall_clock_seconds: f64,
    path: &Path,
) -> std::io::Result<()> {
    ensure_parent_dir(path)?;
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    writeln!(writer, "{{")?;
    write_json_key(&mut writer, 2, "caseDir")?;
    write_json_string(&mut writer, &plan.case_dir.display().to_string())?;
    writeln!(writer, ",")?;
    write_json_key(&mut writer, 2, "solver")?;
    write_json_string(&mut writer, "laminarSimple")?;
    writeln!(writer, ",")?;
    write_json_key(&mut writer, 2, "backend")?;
    write_json_string(&mut writer, "cpu")?;
    writeln!(writer, ",")?;
    write_json_laminar_simple_options(&mut writer, options)?;
    writeln!(writer, ",")?;
    write_json_key(&mut writer, 2, "mesh")?;
    writeln!(writer, "{{")?;
    write_json_number_field(&mut writer, 4, "cells", report.cells)?;
    writeln!(writer, ",")?;
    write_json_number_field(&mut writer, 4, "faces", report.faces)?;
    writeln!(writer, ",")?;
    write_json_number_field(&mut writer, 4, "internalFaces", report.internal_faces)?;
    writeln!(writer, ",")?;
    write_json_number_field(&mut writer, 4, "boundaryFaces", report.boundary_faces)?;
    writeln!(writer)?;
    write_indent(&mut writer, 2)?;
    writeln!(writer, "}},")?;
    write_json_key(&mut writer, 2, "solve")?;
    writeln!(writer, "{{")?;
    write_json_number_field(&mut writer, 4, "simpleIterations", report.simple_iterations)?;
    writeln!(writer, ",")?;
    write_json_bool_field(&mut writer, 4, "converged", report.converged)?;
    writeln!(writer, ",")?;
    write_json_number_field(
        &mut writer,
        4,
        "momentumLinearIterations",
        report.total_momentum_linear_iterations,
    )?;
    writeln!(writer, ",")?;
    write_json_number_field(
        &mut writer,
        4,
        "pressureLinearIterations",
        report.total_pressure_linear_iterations,
    )?;
    writeln!(writer, ",")?;
    write_json_key(&mut writer, 4, "wallClockSeconds")?;
    write_json_optional_number(&mut writer, Some(wall_clock_seconds))?;
    writeln!(writer, ",")?;
    write_json_key(&mut writer, 4, "finalMomentumResidualNorm")?;
    write_json_optional_number(&mut writer, Some(report.final_momentum_residual_norm))?;
    writeln!(writer, ",")?;
    write_json_key(&mut writer, 4, "finalPressureCorrectionResidualNorm")?;
    write_json_optional_number(
        &mut writer,
        Some(report.final_pressure_correction_residual_norm),
    )?;
    writeln!(writer)?;
    write_indent(&mut writer, 2)?;
    writeln!(writer, "}},")?;
    write_json_key(&mut writer, 2, "continuity")?;
    writeln!(writer, "{{")?;
    write_json_key(&mut writer, 4, "initial")?;
    write_json_continuity_summary(&mut writer, 4, &report.initial_continuity)?;
    writeln!(writer, ",")?;
    write_json_key(&mut writer, 4, "final")?;
    write_json_continuity_summary(&mut writer, 4, &report.final_continuity)?;
    writeln!(writer)?;
    write_indent(&mut writer, 2)?;
    writeln!(writer, "}},")?;
    write_json_operator_summary(&mut writer, &report.operator_summary)?;
    writeln!(writer, ",")?;
    write_json_boundary_summary(&mut writer, &report.boundary_summary)?;
    writeln!(writer, ",")?;
    write_json_solution_summary(&mut writer, &report.solution)?;
    writeln!(writer, ",")?;
    write_json_laminar_simple_history(&mut writer, &report.history)?;
    writeln!(writer)?;
    writeln!(writer, "}}")?;

    writer.flush()
}

fn write_json_laminar_simple_options(
    writer: &mut impl Write,
    options: &LaminarSimpleOptions,
) -> std::io::Result<()> {
    write_json_key(writer, 2, "options")?;
    writeln!(writer, "{{")?;
    write_json_string_field(
        writer,
        4,
        "linearSolver",
        &options.linear_solver.to_string(),
    )?;
    writeln!(writer, ",")?;
    write_json_string_field(
        writer,
        4,
        "momentumLinearSolver",
        &options.momentum_linear_solver.to_string(),
    )?;
    writeln!(writer, ",")?;
    write_json_string_field(
        writer,
        4,
        "momentumPreconditioner",
        &options.momentum_preconditioner.to_string(),
    )?;
    writeln!(writer, ",")?;
    write_json_string_field(
        writer,
        4,
        "pressureLinearSolver",
        &options.pressure_linear_solver.to_string(),
    )?;
    writeln!(writer, ",")?;
    write_json_string_field(
        writer,
        4,
        "pressurePreconditioner",
        &options.pressure_preconditioner.to_string(),
    )?;
    writeln!(writer, ",")?;
    write_json_key(writer, 4, "density")?;
    write_json_optional_number(writer, Some(options.density))?;
    writeln!(writer, ",")?;
    write_json_key(writer, 4, "dynamicViscosity")?;
    write_json_optional_number(writer, Some(options.dynamic_viscosity))?;
    writeln!(writer, ",")?;
    write_json_key(writer, 4, "pressureDrop")?;
    write_json_optional_number(writer, Some(options.pressure_drop))?;
    writeln!(writer, ",")?;
    write_json_key(writer, 4, "length")?;
    write_json_optional_number(writer, Some(options.length))?;
    writeln!(writer, ",")?;
    write_json_key(writer, 4, "diameter")?;
    write_json_optional_number(writer, Some(options.diameter))?;
    writeln!(writer, ",")?;
    write_json_string_field(writer, 4, "inletPatch", &options.inlet_patch)?;
    writeln!(writer, ",")?;
    write_json_string_field(writer, 4, "outletPatch", &options.outlet_patch)?;
    writeln!(writer, ",")?;
    write_json_number_field(
        writer,
        4,
        "maxSimpleIterations",
        options.max_simple_iterations,
    )?;
    writeln!(writer, ",")?;
    write_json_number_field(
        writer,
        4,
        "minSimpleIterations",
        options.min_simple_iterations,
    )?;
    writeln!(writer, ",")?;
    write_json_key(writer, 4, "simpleTolerance")?;
    write_json_optional_number(writer, Some(options.simple_tolerance))?;
    writeln!(writer, ",")?;
    write_json_key(writer, 4, "pressureDropTolerance")?;
    write_json_optional_number(writer, Some(options.pressure_drop_tolerance))?;
    writeln!(writer, ",")?;
    write_json_key(writer, 4, "fieldChangeTolerance")?;
    write_json_optional_number(writer, Some(options.field_change_tolerance))?;
    writeln!(writer, ",")?;
    write_json_key(writer, 4, "momentumResidualControl")?;
    write_json_optional_number(writer, options.momentum_residual_control)?;
    writeln!(writer, ",")?;
    write_json_key(writer, 4, "pressureResidualControl")?;
    write_json_optional_number(writer, options.pressure_residual_control)?;
    writeln!(writer, ",")?;
    write_json_number_field(
        writer,
        4,
        "maxLinearIterations",
        options.max_linear_iterations,
    )?;
    writeln!(writer, ",")?;
    write_json_number_field(
        writer,
        4,
        "momentumMaxLinearIterations",
        options.momentum_max_linear_iterations,
    )?;
    writeln!(writer, ",")?;
    write_json_number_field(
        writer,
        4,
        "pressureMaxLinearIterations",
        options.pressure_max_linear_iterations,
    )?;
    writeln!(writer, ",")?;
    write_json_key(writer, 4, "linearTolerance")?;
    write_json_optional_number(writer, Some(options.linear_tolerance))?;
    writeln!(writer, ",")?;
    write_json_key(writer, 4, "momentumLinearTolerance")?;
    write_json_optional_number(writer, Some(options.momentum_linear_tolerance))?;
    writeln!(writer, ",")?;
    write_json_key(writer, 4, "pressureLinearTolerance")?;
    write_json_optional_number(writer, Some(options.pressure_linear_tolerance))?;
    writeln!(writer, ",")?;
    write_json_key(writer, 4, "velocityRelaxation")?;
    write_json_optional_number(writer, Some(options.velocity_relaxation))?;
    writeln!(writer, ",")?;
    write_json_key(writer, 4, "pressureRelaxation")?;
    write_json_optional_number(writer, Some(options.pressure_relaxation))?;
    writeln!(writer)?;
    write_indent(writer, 2)?;
    write!(writer, "}}")
}

fn write_json_continuity_summary(
    writer: &mut impl Write,
    indent: usize,
    summary: &ContinuitySummary,
) -> std::io::Result<()> {
    writeln!(writer, "{{")?;
    write_json_key(writer, indent + 2, "l2Norm")?;
    write_json_optional_number(writer, Some(summary.l2_norm))?;
    writeln!(writer, ",")?;
    write_json_key(writer, indent + 2, "maxAbs")?;
    write_json_optional_number(writer, Some(summary.max_abs))?;
    writeln!(writer, ",")?;
    write_json_key(writer, indent + 2, "sumAbs")?;
    write_json_optional_number(writer, Some(summary.sum_abs))?;
    writeln!(writer, ",")?;
    write_json_key(writer, indent + 2, "globalSum")?;
    write_json_optional_number(writer, Some(summary.global_sum))?;
    writeln!(writer)?;
    write_indent(writer, indent)?;
    write!(writer, "}}")
}

fn write_json_operator_summary(
    writer: &mut impl Write,
    summary: &FlowOperatorSummary,
) -> std::io::Result<()> {
    write_json_key(writer, 2, "operators")?;
    writeln!(writer, "{{")?;
    write_json_key(writer, 4, "phiMin")?;
    write_json_optional_number(writer, Some(summary.phi_min))?;
    writeln!(writer, ",")?;
    write_json_key(writer, 4, "phiMax")?;
    write_json_optional_number(writer, Some(summary.phi_max))?;
    writeln!(writer, ",")?;
    write_json_key(writer, 4, "phiSumAbs")?;
    write_json_optional_number(writer, Some(summary.phi_sum_abs))?;
    writeln!(writer, ",")?;
    write_json_key(writer, 4, "gradPL2Norm")?;
    write_json_optional_number(writer, Some(summary.grad_p_l2_norm))?;
    writeln!(writer, ",")?;
    write_json_key(writer, 4, "divPhiUL2Norm")?;
    write_json_optional_number(writer, Some(summary.div_phi_u_l2_norm))?;
    writeln!(writer)?;
    write_indent(writer, 2)?;
    write!(writer, "}}")
}

fn write_json_boundary_summary(
    writer: &mut impl Write,
    summary: &FlowBoundarySummary,
) -> std::io::Result<()> {
    write_json_key(writer, 2, "boundaries")?;
    writeln!(writer, "{{")?;
    write_json_number_field(
        writer,
        4,
        "velocityFixedValueFaces",
        summary.velocity_fixed_value_faces,
    )?;
    writeln!(writer, ",")?;
    write_json_number_field(
        writer,
        4,
        "velocityZeroGradientFaces",
        summary.velocity_zero_gradient_faces,
    )?;
    writeln!(writer, ",")?;
    write_json_number_field(
        writer,
        4,
        "velocityConstraintFaces",
        summary.velocity_constraint_faces,
    )?;
    writeln!(writer, ",")?;
    write_json_number_field(
        writer,
        4,
        "pressureFixedValueFaces",
        summary.pressure_fixed_value_faces,
    )?;
    writeln!(writer, ",")?;
    write_json_number_field(
        writer,
        4,
        "pressureZeroGradientFaces",
        summary.pressure_zero_gradient_faces,
    )?;
    writeln!(writer, ",")?;
    write_json_number_field(
        writer,
        4,
        "pressureConstraintFaces",
        summary.pressure_constraint_faces,
    )?;
    writeln!(writer)?;
    write_indent(writer, 2)?;
    write!(writer, "}}")
}

fn write_json_solution_summary(
    writer: &mut impl Write,
    summary: &LaminarSimpleSolutionSummary,
) -> std::io::Result<()> {
    write_json_key(writer, 2, "solution")?;
    writeln!(writer, "{{")?;
    write_json_key(writer, 4, "meanVelocity")?;
    write_json_optional_number(writer, Some(summary.mean_velocity))?;
    writeln!(writer, ",")?;
    write_json_key(writer, 4, "analyticMeanVelocity")?;
    write_json_optional_number(writer, Some(summary.analytic_mean_velocity))?;
    writeln!(writer, ",")?;
    write_json_key(writer, 4, "relativeMeanVelocityError")?;
    write_json_optional_number(writer, Some(summary.relative_mean_velocity_error))?;
    writeln!(writer, ",")?;
    write_json_key(writer, 4, "flowRate")?;
    write_json_optional_number(writer, Some(summary.flow_rate))?;
    writeln!(writer, ",")?;
    write_json_key(writer, 4, "analyticFlowRate")?;
    write_json_optional_number(writer, Some(summary.analytic_flow_rate))?;
    writeln!(writer, ",")?;
    write_json_key(writer, 4, "pressureDropFromMean")?;
    write_json_optional_number(writer, Some(summary.pressure_drop_from_mean))?;
    writeln!(writer, ",")?;
    write_json_key(writer, 4, "relativePressureDropError")?;
    write_json_optional_number(writer, Some(summary.relative_pressure_drop_error))?;
    writeln!(writer, ",")?;
    write_json_key(writer, 4, "pressureDropFromField")?;
    write_json_optional_number(writer, summary.pressure_drop_from_field)?;
    writeln!(writer, ",")?;
    write_json_key(writer, 4, "minAxialVelocity")?;
    write_json_optional_number(writer, Some(summary.min_axial_velocity))?;
    writeln!(writer, ",")?;
    write_json_key(writer, 4, "maxAxialVelocity")?;
    write_json_optional_number(writer, Some(summary.max_axial_velocity))?;
    writeln!(writer)?;
    write_indent(writer, 2)?;
    write!(writer, "}}")
}

fn write_json_laminar_simple_history(
    writer: &mut impl Write,
    history: &[LaminarSimpleIterationSummary],
) -> std::io::Result<()> {
    write_json_key(writer, 2, "history")?;
    writeln!(writer, "[")?;
    for (index, item) in history.iter().enumerate() {
        write_indent(writer, 4)?;
        writeln!(writer, "{{")?;
        write_json_number_field(writer, 6, "iteration", item.iteration)?;
        writeln!(writer, ",")?;
        write_json_key(writer, 6, "continuityBefore")?;
        write_json_continuity_summary(writer, 6, &item.continuity_before)?;
        writeln!(writer, ",")?;
        write_json_key(writer, 6, "continuityAfter")?;
        write_json_continuity_summary(writer, 6, &item.continuity_after)?;
        writeln!(writer, ",")?;
        write_json_bool_field(
            writer,
            6,
            "pressureCorrectionAccepted",
            item.pressure_correction_accepted,
        )?;
        writeln!(writer, ",")?;
        write_json_number_field(
            writer,
            6,
            "momentumLinearIterations",
            item.momentum_linear_iterations,
        )?;
        writeln!(writer, ",")?;
        write_json_number_field(
            writer,
            6,
            "pressureLinearIterations",
            item.pressure_linear_iterations,
        )?;
        writeln!(writer, ",")?;
        write_json_key(writer, 6, "momentumResidualNorm")?;
        write_json_optional_number(writer, Some(item.momentum_residual_norm))?;
        writeln!(writer, ",")?;
        write_json_key(writer, 6, "pressureCorrectionResidualNorm")?;
        write_json_optional_number(writer, Some(item.pressure_correction_residual_norm))?;
        writeln!(writer, ",")?;
        write_json_key(writer, 6, "relativePressureDropError")?;
        write_json_optional_number(writer, Some(item.relative_pressure_drop_error))?;
        writeln!(writer, ",")?;
        write_json_key(writer, 6, "relativeVelocityChangeL2")?;
        write_json_optional_number(writer, Some(item.relative_velocity_change_l2))?;
        writeln!(writer, ",")?;
        write_json_key(writer, 6, "relativePressureChangeL2")?;
        write_json_optional_number(writer, Some(item.relative_pressure_change_l2))?;
        writeln!(writer, ",")?;
        write_json_key(writer, 6, "momentumUpdateScale")?;
        write_json_optional_number(writer, Some(item.momentum_update_scale))?;
        writeln!(writer, ",")?;
        write_json_key(writer, 6, "pressureCorrectionUpdateScale")?;
        write_json_optional_number(writer, Some(item.pressure_correction_update_scale))?;
        writeln!(writer)?;
        write_indent(writer, 4)?;
        if index + 1 == history.len() {
            writeln!(writer, "}}")?;
        } else {
            writeln!(writer, "}},")?;
        }
    }
    write_indent(writer, 2)?;
    write!(writer, "]")
}

fn write_laminar_simple_report_markdown(
    plan: &SolverCasePlan,
    options: &LaminarSimpleOptions,
    report: &LaminarSimpleReport,
    wall_clock_seconds: f64,
    path: &Path,
) -> std::io::Result<()> {
    ensure_parent_dir(path)?;
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    writeln!(writer, "# Laminar SIMPLE Benchmark")?;
    writeln!(writer)?;
    writeln!(writer, "Case: `{}`", plan.case_dir.display())?;
    writeln!(writer)?;
    writeln!(writer, "## Inputs")?;
    writeln!(writer)?;
    writeln!(writer, "| Quantity | Value |")?;
    writeln!(writer, "| --- | ---: |")?;
    writeln!(
        writer,
        "| Density [kg/m3] | {} |",
        format_scientific(options.density)
    )?;
    writeln!(
        writer,
        "| Dynamic viscosity [Pa s] | {} |",
        format_scientific(options.dynamic_viscosity)
    )?;
    writeln!(
        writer,
        "| Length [m] | {} |",
        format_scientific(options.length)
    )?;
    writeln!(
        writer,
        "| Diameter [m] | {} |",
        format_scientific(options.diameter)
    )?;
    writeln!(
        writer,
        "| Analytic deltaP [Pa] | {} |",
        format_scientific(options.pressure_drop)
    )?;
    writeln!(
        writer,
        "| Momentum linear solver | {} |",
        options.momentum_linear_solver
    )?;
    writeln!(
        writer,
        "| Momentum preconditioner | {} |",
        options.momentum_preconditioner
    )?;
    writeln!(
        writer,
        "| Momentum linear tolerance | {} |",
        format_scientific(options.momentum_linear_tolerance)
    )?;
    writeln!(
        writer,
        "| Momentum max linear iterations | {} |",
        options.momentum_max_linear_iterations
    )?;
    writeln!(
        writer,
        "| Pressure linear solver | {} |",
        options.pressure_linear_solver
    )?;
    writeln!(
        writer,
        "| Pressure preconditioner | {} |",
        options.pressure_preconditioner
    )?;
    writeln!(
        writer,
        "| Pressure linear tolerance | {} |",
        format_scientific(options.pressure_linear_tolerance)
    )?;
    writeln!(
        writer,
        "| Pressure max linear iterations | {} |",
        options.pressure_max_linear_iterations
    )?;
    writeln!(
        writer,
        "| Min SIMPLE iterations | {} |",
        options.min_simple_iterations
    )?;
    writeln!(
        writer,
        "| SIMPLE continuity tolerance | {} |",
        format_scientific(options.simple_tolerance)
    )?;
    writeln!(
        writer,
        "| Pressure-drop tolerance | {} |",
        format_percent(options.pressure_drop_tolerance)
    )?;
    writeln!(
        writer,
        "| Field-change tolerance | {} |",
        format_percent(options.field_change_tolerance)
    )?;
    writeln!(
        writer,
        "| U residualControl | {} |",
        format_optional_scientific(options.momentum_residual_control)
    )?;
    writeln!(
        writer,
        "| p residualControl | {} |",
        format_optional_scientific(options.pressure_residual_control)
    )?;
    writeln!(writer)?;
    writeln!(writer, "## Result")?;
    writeln!(writer)?;
    writeln!(writer, "| Quantity | Value |")?;
    writeln!(writer, "| --- | ---: |")?;
    writeln!(
        writer,
        "| SIMPLE iterations | {} |",
        report.simple_iterations
    )?;
    writeln!(writer, "| Converged | {} |", yes_no(report.converged))?;
    writeln!(
        writer,
        "| Final continuity L2 | {} |",
        format_scientific(report.final_continuity.l2_norm)
    )?;
    writeln!(
        writer,
        "| Momentum residual norm | {} |",
        format_scientific(report.final_momentum_residual_norm)
    )?;
    writeln!(
        writer,
        "| Pressure-correction residual norm | {} |",
        format_scientific(report.final_pressure_correction_residual_norm)
    )?;
    writeln!(
        writer,
        "| Wall clock [s] | {} |",
        format_scientific(wall_clock_seconds)
    )?;
    writeln!(
        writer,
        "| Mean velocity [m/s] | {} |",
        format_scientific(report.solution.mean_velocity)
    )?;
    writeln!(
        writer,
        "| Analytic mean velocity [m/s] | {} |",
        format_scientific(report.solution.analytic_mean_velocity)
    )?;
    writeln!(
        writer,
        "| Relative mean-velocity error | {} |",
        format_percent(report.solution.relative_mean_velocity_error)
    )?;
    writeln!(
        writer,
        "| DeltaP from mean [Pa] | {} |",
        format_scientific(report.solution.pressure_drop_from_mean)
    )?;
    writeln!(
        writer,
        "| Relative pressure-drop error | {} |",
        format_percent(report.solution.relative_pressure_drop_error)
    )?;
    writeln!(
        writer,
        "| DeltaP from pressure field [Pa] | {} |",
        format_optional_scientific(report.solution.pressure_drop_from_field)
    )?;
    writeln!(writer)?;
    writeln!(writer, "## Iterations")?;
    writeln!(writer)?;
    writeln!(
        writer,
        "| Iteration | Continuity before | Continuity after | Pressure correction | Momentum residual | Pressure residual | Pressure-drop error | U change | p change | Momentum scale | Pressure scale |"
    )?;
    writeln!(
        writer,
        "| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )?;
    for item in &report.history {
        writeln!(
            writer,
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |",
            item.iteration,
            format_scientific(item.continuity_before.l2_norm),
            format_scientific(item.continuity_after.l2_norm),
            if item.pressure_correction_accepted {
                "accepted"
            } else {
                "skipped"
            },
            format_scientific(item.momentum_residual_norm),
            format_scientific(item.pressure_correction_residual_norm),
            format_percent(item.relative_pressure_drop_error),
            format_percent(item.relative_velocity_change_l2),
            format_percent(item.relative_pressure_change_l2),
            format_percent(item.momentum_update_scale),
            format_percent(item.pressure_correction_update_scale)
        )?;
    }

    writer.flush()
}

fn ensure_parent_dir(path: &Path) -> std::io::Result<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }
    Ok(())
}

fn write_json_control(writer: &mut impl Write, plan: &SolverCasePlan) -> std::io::Result<()> {
    write_json_key(writer, 2, "control")?;
    writeln!(writer, "{{")?;
    write_json_key(writer, 4, "application")?;
    write_json_string(writer, &plan.control.application)?;
    writeln!(writer, ",")?;
    write_json_key(writer, 4, "startFrom")?;
    write_json_string(writer, &plan.control.start_from)?;
    writeln!(writer, ",")?;
    write_json_key(writer, 4, "startTime")?;
    write_json_optional_number(writer, plan.control.start_time)?;
    writeln!(writer, ",")?;
    write_json_key(writer, 4, "stopAt")?;
    write_json_string(writer, &plan.control.stop_at)?;
    writeln!(writer, ",")?;
    write_json_key(writer, 4, "endTime")?;
    write_json_optional_number(writer, plan.control.end_time)?;
    writeln!(writer, ",")?;
    write_json_key(writer, 4, "deltaT")?;
    write_json_optional_number(writer, plan.control.delta_t)?;
    writeln!(writer, ",")?;
    write_json_key(writer, 4, "writeControl")?;
    write_json_string(writer, &plan.control.write_control)?;
    writeln!(writer, ",")?;
    write_json_key(writer, 4, "writeInterval")?;
    write_json_optional_number(writer, plan.control.write_interval)?;
    writeln!(writer)?;
    write_indent(writer, 2)?;
    write!(writer, "}}")
}

fn write_json_mesh(writer: &mut impl Write, plan: &SolverMeshPlan) -> std::io::Result<()> {
    write_json_key(writer, 2, "mesh")?;
    writeln!(writer, "{{")?;
    write_json_key(writer, 4, "dimensionality")?;
    write_json_string(writer, &plan.dimensionality.to_string())?;
    writeln!(writer, ",")?;
    write_json_number_field(writer, 4, "points", plan.points)?;
    writeln!(writer, ",")?;
    write_json_number_field(writer, 4, "cells", plan.cells)?;
    writeln!(writer, ",")?;
    write_json_number_field(writer, 4, "faces", plan.faces)?;
    writeln!(writer, ",")?;
    write_json_number_field(writer, 4, "internalFaces", plan.internal_faces)?;
    writeln!(writer, ",")?;
    write_json_number_field(writer, 4, "boundaryFaces", plan.boundary_faces)?;
    writeln!(writer, ",")?;
    write_json_number_field(writer, 4, "patches", plan.patches)?;
    writeln!(writer, ",")?;
    write_json_number_field(writer, 4, "emptyPatches", plan.empty_patches)?;
    writeln!(writer, ",")?;
    write_json_number_field(writer, 4, "wedgePatches", plan.wedge_patches)?;
    writeln!(writer, ",")?;
    write_json_number_field(writer, 4, "symmetryPatches", plan.symmetry_patches)?;
    writeln!(writer, ",")?;
    write_json_key(writer, 4, "regionMeshes")?;
    writeln!(writer, "[")?;
    for (index, region) in plan.region_meshes.iter().enumerate() {
        write_indent(writer, 6)?;
        writeln!(writer, "{{")?;
        write_json_key(writer, 8, "name")?;
        write_json_string(writer, &region.name)?;
        writeln!(writer, ",")?;
        write_json_number_field(writer, 8, "cells", region.cells)?;
        writeln!(writer, ",")?;
        write_json_number_field(writer, 8, "patches", region.patches)?;
        writeln!(writer)?;
        write_indent(writer, 6)?;
        if index + 1 == plan.region_meshes.len() {
            writeln!(writer, "}}")?;
        } else {
            writeln!(writer, "}},")?;
        }
    }
    write_indent(writer, 4)?;
    writeln!(writer, "]")?;
    write_indent(writer, 2)?;
    write!(writer, "}}")
}

fn write_json_fields(writer: &mut impl Write, plan: &SolverFieldPlan) -> std::io::Result<()> {
    write_json_key(writer, 2, "fields")?;
    writeln!(writer, "[")?;
    for (index, field) in plan.fields.iter().enumerate() {
        write_indent(writer, 4)?;
        writeln!(writer, "{{")?;
        write_json_key(writer, 6, "region")?;
        write_json_optional_string(writer, field.region.as_deref())?;
        writeln!(writer, ",")?;
        write_json_key(writer, 6, "name")?;
        write_json_string(writer, &field.name)?;
        writeln!(writer, ",")?;
        write_json_key(writer, 6, "className")?;
        write_json_optional_string(writer, field.class_name.as_deref())?;
        writeln!(writer, ",")?;
        write_json_number_field(writer, 6, "boundaryPatches", field.boundary_patches)?;
        writeln!(writer)?;
        write_indent(writer, 4)?;
        if index + 1 == plan.fields.len() {
            writeln!(writer, "}}")?;
        } else {
            writeln!(writer, "}},")?;
        }
    }
    write_indent(writer, 2)?;
    write!(writer, "]")
}

fn write_json_solver_state(writer: &mut impl Write, plan: &SolverStatePlan) -> std::io::Result<()> {
    write_json_key(writer, 2, "state")?;
    writeln!(writer, "{{")?;
    write_json_key(writer, 4, "fields")?;
    writeln!(writer, "[")?;
    for (index, field) in plan.fields.iter().enumerate() {
        write_indent(writer, 6)?;
        writeln!(writer, "{{")?;
        write_json_key(writer, 8, "region")?;
        write_json_optional_string(writer, field.region.as_deref())?;
        writeln!(writer, ",")?;
        write_json_key(writer, 8, "name")?;
        write_json_string(writer, &field.name)?;
        writeln!(writer, ",")?;
        write_json_key(writer, 8, "className")?;
        write_json_optional_string(writer, field.class_name.as_deref())?;
        writeln!(writer, ",")?;
        write_json_key(writer, 8, "kind")?;
        write_json_string(writer, &field.kind.to_string())?;
        writeln!(writer, ",")?;
        write_json_key(writer, 8, "dimensions")?;
        if let Some(dimensions) = &field.dimensions {
            write_json_string_array(writer, dimensions)?;
        } else {
            write!(writer, "null")?;
        }
        writeln!(writer, ",")?;
        write_json_key(writer, 8, "meshCells")?;
        write_json_optional_usize(writer, field.mesh_cells)?;
        writeln!(writer, ",")?;
        write_json_key(writer, 8, "meshFaces")?;
        write_json_optional_usize(writer, field.mesh_faces)?;
        writeln!(writer, ",")?;
        write_json_key(writer, 8, "internalField")?;
        writeln!(writer, "{{")?;
        write_json_key(writer, 10, "kind")?;
        write_json_string(writer, &field.internal_field.kind.to_string())?;
        writeln!(writer, ",")?;
        write_json_key(writer, 10, "valueCount")?;
        write_json_optional_usize(writer, field.internal_field.value_count)?;
        writeln!(writer, ",")?;
        write_json_key(writer, 10, "expectedCount")?;
        write_json_optional_usize(writer, field.internal_field.expected_count)?;
        writeln!(writer, ",")?;
        write_json_key(writer, 10, "validCount")?;
        write_json_optional_bool(writer, field.internal_field.valid_count)?;
        writeln!(writer, ",")?;
        write_json_key(writer, 10, "uniformComponents")?;
        write_json_optional_f64_array(writer, field.internal_field.uniform_components.as_deref())?;
        writeln!(writer, ",")?;
        write_json_key(writer, 10, "loadedScalars")?;
        write_json_optional_usize(
            writer,
            field
                .internal_field
                .nonuniform_values
                .as_ref()
                .map(Vec::len),
        )?;
        writeln!(writer)?;
        write_indent(writer, 8)?;
        writeln!(writer, "}},")?;
        write_json_number_field(writer, 8, "boundaryPatches", field.boundary_patches)?;
        writeln!(writer, ",")?;
        write_json_key(writer, 8, "meshBoundaryPatches")?;
        write_json_optional_usize(writer, field.mesh_boundary_patches)?;
        writeln!(writer, ",")?;
        write_json_key(writer, 8, "storage")?;
        writeln!(writer, "{{")?;
        write_json_bool_field(writer, 10, "cpuCapable", field.storage.cpu_capable)?;
        writeln!(writer, ",")?;
        write_json_bool_field(writer, 10, "gpuCapable", field.storage.gpu_capable)?;
        writeln!(writer, ",")?;
        write_json_key(writer, 10, "components")?;
        write_json_optional_usize(writer, field.storage.components)?;
        writeln!(writer, ",")?;
        write_json_key(writer, 10, "scalarSlots")?;
        write_json_optional_usize(writer, field.storage.scalar_slots)?;
        writeln!(writer, ",")?;
        write_json_key(writer, 10, "bytesF64")?;
        write_json_optional_usize(writer, field.storage.bytes_f64)?;
        writeln!(writer, ",")?;
        write_json_key(writer, 10, "status")?;
        write_json_string(writer, &field.storage.status.to_string())?;
        writeln!(writer)?;
        write_indent(writer, 8)?;
        writeln!(writer, "}},")?;
        write_json_key(writer, 8, "cpuBuffer")?;
        writeln!(writer, "{{")?;
        write_json_bool_field(
            writer,
            10,
            "materializable",
            field.cpu_buffer.materializable,
        )?;
        writeln!(writer, ",")?;
        write_json_key(writer, 10, "scalarSlots")?;
        write_json_optional_usize(writer, field.cpu_buffer.scalar_slots)?;
        writeln!(writer, ",")?;
        write_json_key(writer, 10, "bytesF64")?;
        write_json_optional_usize(writer, field.cpu_buffer.bytes_f64)?;
        writeln!(writer, ",")?;
        write_json_key(writer, 10, "status")?;
        write_json_string(writer, &field.cpu_buffer.status.to_string())?;
        writeln!(writer)?;
        write_indent(writer, 8)?;
        writeln!(writer, "}}")?;
        write_indent(writer, 6)?;
        if index + 1 == plan.fields.len() {
            writeln!(writer, "}}")?;
        } else {
            writeln!(writer, "}},")?;
        }
    }
    write_indent(writer, 4)?;
    writeln!(writer, "],")?;
    write_json_key(writer, 4, "warnings")?;
    write_json_string_array(writer, &plan.warnings)?;
    writeln!(writer)?;
    write_indent(writer, 2)?;
    write!(writer, "}}")
}

fn write_json_runtime_data(
    writer: &mut impl Write,
    runtime: &SolverRuntimeData,
) -> std::io::Result<()> {
    write_json_key(writer, 2, "runtimeData")?;
    writeln!(writer, "{{")?;
    write_json_key(writer, 4, "mesh")?;
    writeln!(writer, "{{")?;
    write_json_number_field(writer, 6, "points", runtime.mesh.points)?;
    writeln!(writer, ",")?;
    write_json_number_field(writer, 6, "cells", runtime.mesh.cells)?;
    writeln!(writer, ",")?;
    write_json_number_field(writer, 6, "faces", runtime.mesh.faces)?;
    writeln!(writer, ",")?;
    write_json_number_field(writer, 6, "internalFaces", runtime.mesh.internal_faces)?;
    writeln!(writer, ",")?;
    write_json_number_field(writer, 6, "boundaryFaces", runtime.mesh.boundary_faces)?;
    writeln!(writer, ",")?;
    write_json_number_field(writer, 6, "ownerLabels", runtime.mesh.owner.len())?;
    writeln!(writer, ",")?;
    write_json_number_field(
        writer,
        6,
        "neighbourLabels",
        runtime
            .mesh
            .neighbour
            .iter()
            .filter(|cell| cell.is_some())
            .count(),
    )?;
    writeln!(writer, ",")?;
    write_json_number_field(writer, 6, "cellCentres", runtime.mesh.cell_centres.len())?;
    writeln!(writer, ",")?;
    write_json_number_field(writer, 6, "faceCentres", runtime.mesh.face_centres.len())?;
    writeln!(writer, ",")?;
    write_json_number_field(
        writer,
        6,
        "faceAreaVectors",
        runtime.mesh.face_area_vectors.len(),
    )?;
    writeln!(writer, ",")?;
    write_json_number_field(writer, 6, "cellVolumes", runtime.mesh.cell_volumes.len())?;
    writeln!(writer, ",")?;
    write_json_key(writer, 6, "totalCellVolume")?;
    write_json_optional_number(writer, Some(runtime.mesh.total_cell_volume))?;
    writeln!(writer, ",")?;
    write_json_key(writer, 6, "minCellVolume")?;
    write_json_optional_number(writer, Some(runtime.mesh.min_cell_volume))?;
    writeln!(writer, ",")?;
    write_json_key(writer, 6, "maxCellVolume")?;
    write_json_optional_number(writer, Some(runtime.mesh.max_cell_volume))?;
    writeln!(writer, ",")?;
    write_json_key(writer, 6, "minFaceArea")?;
    write_json_optional_number(writer, Some(runtime.mesh.min_face_area))?;
    writeln!(writer, ",")?;
    write_json_key(writer, 6, "maxFaceArea")?;
    write_json_optional_number(writer, Some(runtime.mesh.max_face_area))?;
    writeln!(writer, ",")?;
    write_json_number_field(
        writer,
        6,
        "nonPositiveCellVolumes",
        runtime.mesh.non_positive_cell_volumes,
    )?;
    writeln!(writer, ",")?;
    write_json_key(writer, 6, "patches")?;
    writeln!(writer, "[")?;
    for (index, patch) in runtime.mesh.patches.iter().enumerate() {
        write_indent(writer, 8)?;
        writeln!(writer, "{{")?;
        write_json_string_field(writer, 10, "name", &patch.name)?;
        writeln!(writer, ",")?;
        write_json_string_field(writer, 10, "type", &patch.patch_type)?;
        writeln!(writer, ",")?;
        write_json_number_field(writer, 10, "startFace", patch.start_face)?;
        writeln!(writer, ",")?;
        write_json_number_field(writer, 10, "faces", patch.faces)?;
        writeln!(writer)?;
        write_indent(writer, 8)?;
        if index + 1 == runtime.mesh.patches.len() {
            writeln!(writer, "}}")?;
        } else {
            writeln!(writer, "}},")?;
        }
    }
    write_indent(writer, 6)?;
    writeln!(writer, "]")?;
    write_indent(writer, 4)?;
    writeln!(writer, "}},")?;
    write_json_key(writer, 4, "fields")?;
    writeln!(writer, "[")?;
    for (index, field) in runtime.fields.iter().enumerate() {
        write_indent(writer, 6)?;
        writeln!(writer, "{{")?;
        write_json_key(writer, 8, "region")?;
        write_json_optional_string(writer, field.region.as_deref())?;
        writeln!(writer, ",")?;
        write_json_string_field(writer, 8, "name", &field.name)?;
        writeln!(writer, ",")?;
        write_json_key(writer, 8, "kind")?;
        write_json_string(writer, &field.kind.to_string())?;
        writeln!(writer, ",")?;
        write_json_number_field(writer, 8, "components", field.components)?;
        writeln!(writer, ",")?;
        write_json_number_field(writer, 8, "scalarSlots", field.scalar_slots)?;
        writeln!(writer, ",")?;
        write_json_number_field(writer, 8, "bytesF64", field.bytes_f64)?;
        writeln!(writer)?;
        write_indent(writer, 6)?;
        if index + 1 == runtime.fields.len() {
            writeln!(writer, "}}")?;
        } else {
            writeln!(writer, "}},")?;
        }
    }
    write_indent(writer, 4)?;
    writeln!(writer, "],")?;
    write_json_key(writer, 4, "warnings")?;
    write_json_string_array(writer, &runtime.warnings)?;
    writeln!(writer)?;
    write_indent(writer, 2)?;
    write!(writer, "}}")
}

fn write_json_properties(
    writer: &mut impl Write,
    plan: &SolverPropertiesPlan,
) -> std::io::Result<()> {
    write_json_key(writer, 2, "properties")?;
    writeln!(writer, "{{")?;
    write_json_key(writer, 4, "dictionaries")?;
    writeln!(writer, "[")?;
    for (index, dictionary) in plan.dictionaries.iter().enumerate() {
        write_indent(writer, 6)?;
        writeln!(writer, "{{")?;
        write_json_key(writer, 8, "name")?;
        write_json_string(writer, &dictionary.name)?;
        writeln!(writer, ",")?;
        write_json_key(writer, 8, "region")?;
        write_json_optional_string(writer, dictionary.region.as_deref())?;
        writeln!(writer, ",")?;
        write_json_number_field(writer, 8, "sections", dictionary.sections)?;
        writeln!(writer, ",")?;
        write_json_number_field(writer, 8, "entries", dictionary.entries)?;
        writeln!(writer)?;
        write_indent(writer, 6)?;
        if index + 1 == plan.dictionaries.len() {
            writeln!(writer, "}}")?;
        } else {
            writeln!(writer, "}},")?;
        }
    }
    write_indent(writer, 4)?;
    writeln!(writer, "],")?;
    write_json_key(writer, 4, "entries")?;
    writeln!(writer, "[")?;
    for (index, entry) in plan.entries.iter().enumerate() {
        write_indent(writer, 6)?;
        writeln!(writer, "{{")?;
        write_json_key(writer, 8, "dictionary")?;
        write_json_string(writer, &entry.dictionary)?;
        writeln!(writer, ",")?;
        write_json_key(writer, 8, "section")?;
        write_json_optional_string(writer, entry.section.as_deref())?;
        writeln!(writer, ",")?;
        write_json_key(writer, 8, "key")?;
        write_json_string(writer, &entry.key)?;
        writeln!(writer, ",")?;
        write_json_key(writer, 8, "value")?;
        write_json_string(writer, &entry.value)?;
        writeln!(writer)?;
        write_indent(writer, 6)?;
        if index + 1 == plan.entries.len() {
            writeln!(writer, "}}")?;
        } else {
            writeln!(writer, "}},")?;
        }
    }
    write_indent(writer, 4)?;
    writeln!(writer, "]")?;
    write_indent(writer, 2)?;
    write!(writer, "}}")
}

fn write_json_numerics(writer: &mut impl Write, plan: &SolverNumericsPlan) -> std::io::Result<()> {
    write_json_key(writer, 2, "numerics")?;
    writeln!(writer, "{{")?;
    write_json_numerics_dictionary(writer, 4, "fvSchemes", &plan.fv_schemes)?;
    writeln!(writer, ",")?;
    write_json_numerics_dictionary(writer, 4, "fvSolution", &plan.fv_solution)?;
    writeln!(writer)?;
    write_indent(writer, 2)?;
    write!(writer, "}}")
}

fn write_json_numerics_dictionary(
    writer: &mut impl Write,
    indent: usize,
    name: &str,
    plan: &SolverNumericsDictionaryPlan,
) -> std::io::Result<()> {
    write_json_key(writer, indent, name)?;
    writeln!(writer, "{{")?;
    write_json_key(writer, indent + 2, "present")?;
    write!(writer, "{}", plan.present)?;
    writeln!(writer, ",")?;
    write_json_key(writer, indent + 2, "sections")?;
    writeln!(writer, "[")?;
    for (index, section) in plan.sections.iter().enumerate() {
        write_indent(writer, indent + 4)?;
        writeln!(writer, "{{")?;
        write_json_key(writer, indent + 6, "path")?;
        write_json_string(writer, &section.path)?;
        writeln!(writer, ",")?;
        write_json_number_field(writer, indent + 6, "entries", section.entries)?;
        writeln!(writer)?;
        write_indent(writer, indent + 4)?;
        if index + 1 == plan.sections.len() {
            writeln!(writer, "}}")?;
        } else {
            writeln!(writer, "}},")?;
        }
    }
    write_indent(writer, indent + 2)?;
    writeln!(writer, "],")?;
    write_json_key(writer, indent + 2, "entries")?;
    writeln!(writer, "[")?;
    for (index, entry) in plan.entries.iter().enumerate() {
        write_indent(writer, indent + 4)?;
        writeln!(writer, "{{")?;
        write_json_key(writer, indent + 6, "section")?;
        write_json_string(writer, &entry.section)?;
        writeln!(writer, ",")?;
        write_json_key(writer, indent + 6, "key")?;
        write_json_string(writer, &entry.key)?;
        writeln!(writer, ",")?;
        write_json_key(writer, indent + 6, "value")?;
        write_json_string(writer, &entry.value)?;
        writeln!(writer)?;
        write_indent(writer, indent + 4)?;
        if index + 1 == plan.entries.len() {
            writeln!(writer, "}}")?;
        } else {
            writeln!(writer, "}},")?;
        }
    }
    write_indent(writer, indent + 2)?;
    writeln!(writer, "]")?;
    write_indent(writer, indent)?;
    write!(writer, "}}")
}

fn write_json_interfaces(
    writer: &mut impl Write,
    plan: &SolverInterfacePlan,
) -> std::io::Result<()> {
    write_json_key(writer, 2, "interfaces")?;
    writeln!(writer, "{{")?;
    write_json_bool_field(writer, 4, "registryAvailable", plan.registry_available)?;
    writeln!(writer, ",")?;
    write_json_number_field(
        writer,
        4,
        "discoveredInterfaces",
        plan.discovered_interfaces,
    )?;
    writeln!(writer, ",")?;
    write_json_number_field(writer, 4, "boundaryFaceZones", plan.boundary_face_zones)?;
    writeln!(writer, ",")?;
    write_json_bool_field(writer, 4, "configPresent", plan.config_present)?;
    writeln!(writer, ",")?;
    write_json_number_field(
        writer,
        4,
        "configuredInterfaces",
        plan.configured_interfaces,
    )?;
    writeln!(writer)?;
    write_indent(writer, 2)?;
    write!(writer, "}}")
}

fn write_json_run_plan(writer: &mut impl Write, plan: &SolverRunPlan) -> std::io::Result<()> {
    write_json_key(writer, 2, "run")?;
    writeln!(writer, "{{")?;
    write_json_key(writer, 4, "stopAt")?;
    write_json_string(writer, &plan.stop_at)?;
    writeln!(writer, ",")?;
    write_json_key(writer, 4, "startTime")?;
    write_json_optional_number(writer, plan.start_time)?;
    writeln!(writer, ",")?;
    write_json_key(writer, 4, "endTime")?;
    write_json_optional_number(writer, plan.end_time)?;
    writeln!(writer, ",")?;
    write_json_key(writer, 4, "deltaT")?;
    write_json_optional_number(writer, plan.delta_t)?;
    writeln!(writer, ",")?;
    write_json_key(writer, 4, "estimatedSteps")?;
    write_json_optional_usize(writer, plan.estimated_steps)?;
    writeln!(writer, ",")?;
    write_json_key(writer, 4, "writeControl")?;
    write_json_string(writer, &plan.write_control)?;
    writeln!(writer, ",")?;
    write_json_key(writer, 4, "writeInterval")?;
    write_json_optional_number(writer, plan.write_interval)?;
    writeln!(writer, ",")?;
    write_json_key(writer, 4, "estimatedWriteEvents")?;
    write_json_optional_usize(writer, plan.estimated_write_events)?;
    writeln!(writer, ",")?;
    write_json_key(writer, 4, "stages")?;
    writeln!(writer, "[")?;
    for (index, stage) in plan.stages.iter().enumerate() {
        write_indent(writer, 6)?;
        writeln!(writer, "{{")?;
        write_json_key(writer, 8, "section")?;
        write_json_string(writer, &stage.section)?;
        writeln!(writer, ",")?;
        write_json_key(writer, 8, "step")?;
        write_json_string(writer, &stage.step)?;
        writeln!(writer, ",")?;
        write_json_key(writer, 8, "choice")?;
        write_json_string(writer, &stage.choice.to_string())?;
        writeln!(writer, ",")?;
        write_json_key(writer, 8, "source")?;
        write_json_string(writer, &stage.source.to_string())?;
        writeln!(writer)?;
        write_indent(writer, 6)?;
        if index + 1 == plan.stages.len() {
            writeln!(writer, "}}")?;
        } else {
            writeln!(writer, "}},")?;
        }
    }
    write_indent(writer, 4)?;
    writeln!(writer, "]")?;
    write_indent(writer, 2)?;
    write!(writer, "}}")
}

fn write_json_backends(writer: &mut impl Write, plan: &SolverBackendPlan) -> std::io::Result<()> {
    write_json_key(writer, 2, "backends")?;
    writeln!(writer, "{{")?;
    write_json_bool_field(writer, 4, "configPresent", plan.config_present)?;
    writeln!(writer, ",")?;
    write_json_key(writer, 4, "default")?;
    write_json_string(writer, &plan.default.to_string())?;
    writeln!(writer, ",")?;
    write_json_bool_field(writer, 4, "usesCpu", plan.uses_cpu)?;
    writeln!(writer, ",")?;
    write_json_bool_field(writer, 4, "usesGpu", plan.uses_gpu)?;
    writeln!(writer, ",")?;
    write_json_bool_field(writer, 4, "mixedExecution", plan.mixed_execution)?;
    writeln!(writer, ",")?;
    write_json_key(writer, 4, "cpu")?;
    writeln!(writer, "{{")?;
    write_json_string_field(writer, 6, "cpus", &plan.cpu.cpus)?;
    writeln!(writer, ",")?;
    write_json_string_field(writer, 6, "coresPerCpu", &plan.cpu.cores_per_cpu)?;
    writeln!(writer, ",")?;
    write_json_string_field(writer, 6, "threads", &plan.cpu.threads)?;
    writeln!(writer, ",")?;
    write_json_string_field(writer, 6, "threadPinning", &plan.cpu.thread_pinning)?;
    writeln!(writer, ",")?;
    write_json_string_field(writer, 6, "numa", &plan.cpu.numa)?;
    writeln!(writer)?;
    write_indent(writer, 4)?;
    writeln!(writer, "}},")?;
    write_json_key(writer, 4, "gpu")?;
    writeln!(writer, "{{")?;
    write_json_string_field(writer, 6, "backend", &plan.gpu.backend)?;
    writeln!(writer, ",")?;
    write_json_key(writer, 6, "devices")?;
    write_json_string_array(writer, &plan.gpu.devices)?;
    writeln!(writer, ",")?;
    write_json_string_field(writer, 6, "multiGpu", &plan.gpu.multi_gpu)?;
    writeln!(writer, ",")?;
    write_json_string_field(writer, 6, "precision", &plan.gpu.precision)?;
    writeln!(writer)?;
    write_indent(writer, 4)?;
    writeln!(writer, "}},")?;
    write_json_key(writer, 4, "stages")?;
    writeln!(writer, "[")?;
    for (index, stage) in plan.stages.iter().enumerate() {
        write_indent(writer, 6)?;
        writeln!(writer, "{{")?;
        write_json_key(writer, 8, "section")?;
        write_json_string(writer, &stage.section)?;
        writeln!(writer, ",")?;
        write_json_key(writer, 8, "step")?;
        write_json_string(writer, &stage.step)?;
        writeln!(writer, ",")?;
        write_json_key(writer, 8, "choice")?;
        write_json_string(writer, &stage.choice.to_string())?;
        writeln!(writer)?;
        write_indent(writer, 6)?;
        if index + 1 == plan.stages.len() {
            writeln!(writer, "}}")?;
        } else {
            writeln!(writer, "}},")?;
        }
    }
    write_indent(writer, 4)?;
    writeln!(writer, "]")?;
    write_indent(writer, 2)?;
    write!(writer, "}}")
}

fn write_json_key(writer: &mut impl Write, indent: usize, key: &str) -> std::io::Result<()> {
    write_indent(writer, indent)?;
    write_json_string(writer, key)?;
    write!(writer, ": ")
}

fn write_json_number_field(
    writer: &mut impl Write,
    indent: usize,
    key: &str,
    value: usize,
) -> std::io::Result<()> {
    write_json_key(writer, indent, key)?;
    write!(writer, "{value}")
}

fn write_json_bool_field(
    writer: &mut impl Write,
    indent: usize,
    key: &str,
    value: bool,
) -> std::io::Result<()> {
    write_json_key(writer, indent, key)?;
    write!(writer, "{value}")
}

fn write_json_string_field(
    writer: &mut impl Write,
    indent: usize,
    key: &str,
    value: &str,
) -> std::io::Result<()> {
    write_json_key(writer, indent, key)?;
    write_json_string(writer, value)
}

fn write_json_optional_number(writer: &mut impl Write, value: Option<f64>) -> std::io::Result<()> {
    match value {
        Some(value) if value.is_finite() => write!(writer, "{value}"),
        Some(value) => write_json_string(writer, &value.to_string()),
        None => write!(writer, "null"),
    }
}

fn write_json_optional_usize(writer: &mut impl Write, value: Option<usize>) -> std::io::Result<()> {
    match value {
        Some(value) => write!(writer, "{value}"),
        None => write!(writer, "null"),
    }
}

fn write_json_optional_bool(writer: &mut impl Write, value: Option<bool>) -> std::io::Result<()> {
    match value {
        Some(value) => write!(writer, "{value}"),
        None => write!(writer, "null"),
    }
}

fn write_json_optional_f64_array(
    writer: &mut impl Write,
    values: Option<&[f64]>,
) -> std::io::Result<()> {
    let Some(values) = values else {
        return write!(writer, "null");
    };
    write!(writer, "[")?;
    for (index, value) in values.iter().enumerate() {
        if index > 0 {
            write!(writer, ", ")?;
        }
        if value.is_finite() {
            write!(writer, "{value}")?;
        } else {
            write_json_string(writer, &value.to_string())?;
        }
    }
    write!(writer, "]")
}

fn write_json_optional_string(writer: &mut impl Write, value: Option<&str>) -> std::io::Result<()> {
    match value {
        Some(value) => write_json_string(writer, value),
        None => write!(writer, "null"),
    }
}

fn write_json_string_array(writer: &mut impl Write, values: &[String]) -> std::io::Result<()> {
    write!(writer, "[")?;
    for (index, value) in values.iter().enumerate() {
        if index > 0 {
            write!(writer, ", ")?;
        }
        write_json_string(writer, value)?;
    }
    write!(writer, "]")
}

fn write_json_string(writer: &mut impl Write, value: &str) -> std::io::Result<()> {
    write!(writer, "\"")?;
    for ch in value.chars() {
        match ch {
            '"' => write!(writer, "\\\"")?,
            '\\' => write!(writer, "\\\\")?,
            '\n' => write!(writer, "\\n")?,
            '\r' => write!(writer, "\\r")?,
            '\t' => write!(writer, "\\t")?,
            ch if ch.is_control() => write!(writer, "\\u{:04x}", ch as u32)?,
            ch => write!(writer, "{ch}")?,
        }
    }
    write!(writer, "\"")
}

fn write_indent(writer: &mut impl Write, indent: usize) -> std::io::Result<()> {
    for _ in 0..indent {
        write!(writer, " ")?;
    }
    Ok(())
}

fn print_interface_registry(registry: &InterfaceRegistrySummary) {
    if !registry.interfaces.is_empty() {
        println!("interfaces:");
        for interface in &registry.interfaces {
            print_interface(interface);
        }
    }
    if registry.same_region_face_zone_faces > 0 || registry.unknown_region_face_zone_faces > 0 {
        println!(
            "interface registry notes: sameRegionFaceZoneFaces={}, unknownRegionFaceZoneFaces={}",
            registry.same_region_face_zone_faces, registry.unknown_region_face_zone_faces
        );
    }
}

fn print_interface(interface: &InterfaceSummary) {
    println!(
        "  {}: {} <-> {} faces={} mesh({}->{}={}, {}->{}={}) zone({}->{}={}, {}->{}={}) flipped={}",
        interface.name,
        interface.region_a,
        interface.region_b,
        interface.faces,
        interface.region_a,
        interface.region_b,
        interface.mesh_a_to_b_faces,
        interface.region_b,
        interface.region_a,
        interface.mesh_b_to_a_faces,
        interface.region_a,
        interface.region_b,
        interface.zone_a_to_b_faces,
        interface.region_b,
        interface.region_a,
        interface.zone_b_to_a_faces,
        interface.flipped_faces
    );
}

fn print_interface_config(
    case_dir: &Path,
    registry: &InterfaceRegistrySummary,
) -> Result<(), String> {
    let Some(config) = read_interface_config(case_dir).map_err(|error| error.to_string())? else {
        println!("interface config: none (no constant/interfaces)");
        return Ok(());
    };

    let validation = validate_interface_config(&config, registry);
    if validation.entries.is_empty() {
        println!("interface config: no configured entries");
    } else {
        println!("interface config:");
        for entry in &validation.entries {
            println!(
                "  {}: faceZone={} sign={}->{} model={} meshFaces={}",
                entry.name,
                entry.face_zone,
                entry.positive_from,
                entry.positive_to,
                entry.model,
                display_count(entry.mesh_faces)
            );
        }
    }
    for warning in &validation.warnings {
        println!("interface config warning: {warning}");
    }

    Ok(())
}

fn print_backend_config(case_dir: &Path) -> Result<(), String> {
    let Some(config) = read_backend_config(case_dir).map_err(|error| error.to_string())? else {
        println!("backend config: none (no system/ferrumBackends)");
        return Ok(());
    };

    println!(
        "backend config: default={} cpuCpus={} cpuCoresPerCpu={} cpuThreads={} cpuPinning={} cpuNuma={} gpuBackend={} gpuDevices={} multiGpu={} precision={}",
        config.default,
        config.cpu.cpus,
        config.cpu.cores_per_cpu,
        config.cpu.threads,
        config.cpu.thread_pinning,
        config.cpu.numa,
        config.gpu.backend,
        format_devices(&config.gpu.devices),
        config.gpu.multi_gpu,
        config.gpu.precision
    );
    for section in &config.sections {
        if section.entries.is_empty() {
            continue;
        }

        let selections = section
            .entries
            .iter()
            .map(|entry| format!("{}={}", entry.step, entry.choice))
            .collect::<Vec<_>>()
            .join(", ");
        println!("  {}: {}", section.name, selections);
    }
    let validation = validate_backend_resources(&config);
    println!(
        "backend resources: usesCpu={} usesGpu={} mixed={}",
        validation.uses_cpu, validation.uses_gpu, validation.mixed_execution
    );
    for warning in &validation.warnings {
        println!("backend resource warning: {warning}");
    }
    let policy_validation = validate_backend_policy(&config);
    for warning in &policy_validation.warnings {
        println!("backend policy warning: {warning}");
    }

    Ok(())
}

fn format_devices(devices: &[String]) -> String {
    if devices.len() == 1 {
        return devices[0].clone();
    }
    format!("({})", devices.join(" "))
}

fn format_optional_number(value: Option<f64>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "missing".to_string())
}

fn format_optional_usize(value: Option<usize>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "missing".to_string())
}

fn format_optional_bool(value: Option<bool>) -> &'static str {
    match value {
        Some(true) => "yes",
        Some(false) => "no",
        None => "missing",
    }
}

fn format_optional_f64_list(values: Option<&[f64]>) -> String {
    let Some(values) = values else {
        return "missing".to_string();
    };
    if values.is_empty() {
        return "empty".to_string();
    }
    format!(
        "({})",
        values
            .iter()
            .map(|value| value.to_string())
            .collect::<Vec<_>>()
            .join(" ")
    )
}

fn yes_no(value: bool) -> &'static str {
    if value { "yes" } else { "no" }
}

fn print_initial_fields(fields: &InitialFieldSet) {
    if fields.fields.is_empty() {
        println!("initial fields: none");
        return;
    }

    println!("initial fields:");
    for field in &fields.fields {
        print_initial_field(field);
    }
}

fn print_initial_field(field: &FieldFile) {
    let display_name = if let Some(region) = &field.region {
        format!("{region}/{}", field.name)
    } else {
        field.name.clone()
    };
    let dimensions = field
        .dimensions
        .as_ref()
        .map(|values| format!("[{}]", values.join(" ")))
        .unwrap_or_else(|| "unknown".to_string());
    let internal = field
        .internal_field
        .as_ref()
        .map(|value| value.to_string())
        .unwrap_or_else(|| "missing".to_string());

    println!(
        "  {}: class={} dimensions={} internal={} boundaryPatches={}",
        display_name,
        field.class_name.as_deref().unwrap_or("unknown"),
        dimensions,
        internal,
        field.boundary_patches.len()
    );
    for patch in &field.boundary_patches {
        if let Some(value) = &patch.value {
            println!(
                "    patch {} type={} value={}",
                patch.name,
                patch.patch_type.as_deref().unwrap_or("unknown"),
                value
            );
        } else {
            println!(
                "    patch {} type={}",
                patch.name,
                patch.patch_type.as_deref().unwrap_or("unknown")
            );
        }
    }
}

fn print_field_boundary_validation(case_dir: &Path, fields: &InitialFieldSet) {
    let summary = validate_initial_field_boundaries(case_dir, fields);
    print_field_boundary_validation_summary(&summary);
}

fn print_field_boundary_validation_summary(summary: &FieldBoundaryValidationSummary) {
    if summary.fields == 0 {
        return;
    }

    println!(
        "field boundary validation: fields={} warnings={}",
        summary.fields,
        summary.warnings.len()
    );
    for warning in &summary.warnings {
        println!("field boundary warning: {warning}");
    }
}

fn print_geometry_summary(case_dir: &Path) -> Result<(), String> {
    let geometry = summarize_case_geometry(case_dir).map_err(|error| error.to_string())?;
    print_geometry(&geometry);
    Ok(())
}

fn print_geometry(geometry: &GeometrySummary) {
    println!(
        "geometry: cells={} faces={} totalVolume={} minCellVolume={} maxCellVolume={} nonPositiveCellVolumes={}",
        geometry.cells,
        geometry.faces,
        format_scientific(geometry.total_cell_volume),
        format_scientific(geometry.min_cell_volume),
        format_scientific(geometry.max_cell_volume),
        geometry.non_positive_cell_volumes
    );
    println!(
        "geometry faces: minArea={} maxArea={} totalBoundaryArea={}",
        format_scientific(geometry.min_face_area),
        format_scientific(geometry.max_face_area),
        format_scientific(geometry.total_boundary_area)
    );
}

fn print_patch_validation(case_dir: &Path) -> Result<(), String> {
    let summary = validate_case_patches(case_dir).map_err(|error| error.to_string())?;
    print_patch_validation_summary(&summary);
    Ok(())
}

fn print_patch_validation_summary(summary: &PatchValidationSummary) {
    println!(
        "patch validation: patches={} empty={} wedge={} symmetryPlane={} warnings={}",
        summary.patches,
        summary.empty_patches,
        summary.wedge_patches,
        summary.symmetry_patches,
        summary.warnings.len()
    );
    for warning in &summary.warnings {
        println!("patch validation warning: {warning}");
    }
}

fn format_scientific(value: f64) -> String {
    format!("{value:.6e}")
}

fn format_optional_scientific(value: Option<f64>) -> String {
    value
        .map(format_scientific)
        .unwrap_or_else(|| "n/a".to_string())
}

fn format_percent(value: f64) -> String {
    format!("{:.3}%", value * 100.0)
}

fn print_region_patch(patch: &ferrum_mesh::regions::RegionPatchSummary) {
    if patch.source_flipped_faces > 0 {
        println!(
            "    patch {} type={} faces={} startFace={} sourceFlippedFaces={}",
            patch.name, patch.patch_type, patch.faces, patch.start_face, patch.source_flipped_faces
        );
    } else {
        println!(
            "    patch {} type={} faces={} startFace={}",
            patch.name, patch.patch_type, patch.faces, patch.start_face
        );
    }
}

fn parse_case_dir(args: &[String], default: PathBuf) -> Result<PathBuf, String> {
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "-case" | "--case" => {
                let value = args
                    .get(index + 1)
                    .ok_or_else(|| "-case requires a directory".to_string())?;
                return Ok(PathBuf::from(value));
            }
            _ => index += 1,
        }
    }
    Ok(default)
}

#[derive(Debug)]
struct SolverArgs {
    case_dir: PathBuf,
    plan_json: Option<PathBuf>,
    runner_dry_run: bool,
    max_runner_steps: usize,
    scalar_diffusion_solve: Option<ScalarDiffusionSolveArgs>,
    poiseuille_solve: Option<PoiseuilleSolveArgs>,
    laminar_simple_solve: Option<LaminarSimpleSolveArgs>,
}

#[derive(Debug)]
struct ScalarDiffusionSolveArgs {
    field: String,
    diffusivity: f64,
    source: f64,
    linear_solver: ScalarDiffusionLinearSolver,
    tolerance: f64,
    max_iterations: usize,
}

#[derive(Debug)]
struct PoiseuilleSolveArgs {
    pressure_drop: Option<f64>,
    dynamic_viscosity: Option<f64>,
    length: Option<f64>,
    diameter: Option<f64>,
    wall_patches: Vec<String>,
    linear_solver: ScalarDiffusionLinearSolver,
    tolerance: f64,
    max_iterations: usize,
}

#[derive(Debug)]
struct LaminarSimpleSolveArgs {
    density: Option<f64>,
    dynamic_viscosity: Option<f64>,
    pressure_drop: Option<f64>,
    length: Option<f64>,
    diameter: Option<f64>,
    inlet_patch: String,
    outlet_patch: String,
    linear_solver: Option<LaminarSimpleLinearSolver>,
    momentum_linear_solver: Option<LaminarSimpleLinearSolver>,
    pressure_linear_solver: Option<LaminarSimpleLinearSolver>,
    momentum_preconditioner: Option<LaminarSimplePreconditioner>,
    pressure_preconditioner: Option<LaminarSimplePreconditioner>,
    linear_tolerance: Option<f64>,
    max_linear_iterations: Option<usize>,
    momentum_linear_tolerance: Option<f64>,
    pressure_linear_tolerance: Option<f64>,
    momentum_max_linear_iterations: Option<usize>,
    pressure_max_linear_iterations: Option<usize>,
    max_simple_iterations: usize,
    min_simple_iterations: Option<usize>,
    simple_tolerance: f64,
    pressure_drop_tolerance: Option<f64>,
    field_change_tolerance: Option<f64>,
    velocity_relaxation: Option<f64>,
    pressure_relaxation: Option<f64>,
    report_json: Option<PathBuf>,
    report_markdown: Option<PathBuf>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ScalarDiffusionLinearSolver {
    Cg,
    Jacobi,
}

struct ScalarSolutionSummary {
    min: f64,
    max: f64,
    mean: f64,
}

impl std::fmt::Display for ScalarDiffusionLinearSolver {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cg => formatter.write_str("cg"),
            Self::Jacobi => formatter.write_str("jacobi"),
        }
    }
}

fn parse_solver_args(args: &[String]) -> Result<SolverArgs, String> {
    let mut case_dir = PathBuf::from(".");
    let mut plan_json = None;
    let mut runner_dry_run = false;
    let mut max_runner_steps = SolverRunnerDryRunOptions::default().max_steps;
    let mut scalar_diffusion_field = None;
    let mut scalar_diffusion_option_seen = false;
    let mut poiseuille_solve = false;
    let mut poiseuille_option_seen = false;
    let mut laminar_simple_solve = false;
    let mut laminar_simple_option_seen = false;
    let mut density = None;
    let mut pressure_drop = None;
    let mut dynamic_viscosity = None;
    let mut length = None;
    let mut diameter = None;
    let mut inlet_patch = "inlet".to_string();
    let mut outlet_patch = "outlet".to_string();
    let mut wall_patches = Vec::new();
    let mut linear_solve_option_seen = false;
    let mut laminar_linear_solver = None;
    let mut momentum_linear_solver = None;
    let mut pressure_linear_solver = None;
    let mut momentum_preconditioner = None;
    let mut pressure_preconditioner = None;
    let mut scalar_diffusion_diffusivity = 1.0;
    let mut scalar_diffusion_source = 0.0;
    let mut scalar_diffusion_linear_solver = ScalarDiffusionLinearSolver::Cg;
    let mut scalar_diffusion_tolerance = 1.0e-10;
    let mut scalar_diffusion_max_iterations = 10_000;
    let mut laminar_linear_tolerance = None;
    let mut laminar_max_linear_iterations = None;
    let mut momentum_linear_tolerance = None;
    let mut pressure_linear_tolerance = None;
    let mut momentum_max_linear_iterations = None;
    let mut pressure_max_linear_iterations = None;
    let mut max_simple_iterations = 1;
    let mut min_simple_iterations = None;
    let mut simple_tolerance = 1.0e-8;
    let mut pressure_drop_tolerance = None;
    let mut field_change_tolerance = None;
    let mut velocity_relaxation = None;
    let mut pressure_relaxation = None;
    let mut solve_report_json = None;
    let mut solve_report_markdown = None;
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "-case" | "--case" => {
                case_dir = PathBuf::from(
                    args.get(index + 1)
                        .ok_or_else(|| "-case requires a directory".to_string())?,
                );
                index += 2;
            }
            "-preflight" | "--preflight" | "-dryRun" | "--dry-run" => {
                index += 1;
            }
            "-planJson" | "--planJson" | "-plan-json" | "--plan-json" => {
                let path = args
                    .get(index + 1)
                    .ok_or_else(|| "--planJson requires a file path".to_string())?;
                plan_json = Some(PathBuf::from(path));
                index += 2;
            }
            "-runnerDryRun" | "--runnerDryRun" | "-runner-dry-run" | "--runner-dry-run" => {
                runner_dry_run = true;
                index += 1;
            }
            "-maxRunnerSteps" | "--maxRunnerSteps" | "-max-runner-steps" | "--max-runner-steps" => {
                let value = args
                    .get(index + 1)
                    .ok_or_else(|| "--maxRunnerSteps requires a positive integer".to_string())?;
                max_runner_steps = value.parse::<usize>().map_err(|_| {
                    format!("invalid --maxRunnerSteps value '{value}'; expected a positive integer")
                })?;
                if max_runner_steps == 0 {
                    return Err("--maxRunnerSteps must be greater than zero".to_string());
                }
                index += 2;
            }
            "-solveScalarDiffusion"
            | "--solveScalarDiffusion"
            | "-solve-scalar-diffusion"
            | "--solve-scalar-diffusion" => {
                let field = args
                    .get(index + 1)
                    .ok_or_else(|| "--solveScalarDiffusion requires a field name".to_string())?;
                if field.trim().is_empty() {
                    return Err("--solveScalarDiffusion field name must not be empty".to_string());
                }
                scalar_diffusion_field = Some(field.to_string());
                index += 2;
            }
            "-solvePoiseuille" | "--solvePoiseuille" | "-solve-poiseuille"
            | "--solve-poiseuille" => {
                poiseuille_solve = true;
                index += 1;
            }
            "-solveLaminarSimple"
            | "--solveLaminarSimple"
            | "-solve-laminar-simple"
            | "--solve-laminar-simple" => {
                laminar_simple_solve = true;
                index += 1;
            }
            "-diffusivity" | "--diffusivity" => {
                let value = args
                    .get(index + 1)
                    .ok_or_else(|| "--diffusivity requires a positive number".to_string())?;
                scalar_diffusion_diffusivity = parse_positive_f64_arg("--diffusivity", value)?;
                scalar_diffusion_option_seen = true;
                index += 2;
            }
            "-source" | "--source" => {
                let value = args
                    .get(index + 1)
                    .ok_or_else(|| "--source requires a finite number".to_string())?;
                scalar_diffusion_source = parse_finite_f64_arg("--source", value)?;
                scalar_diffusion_option_seen = true;
                index += 2;
            }
            "-pressureDrop" | "--pressureDrop" | "-pressure-drop" | "--pressure-drop" => {
                let value = args.get(index + 1).ok_or_else(|| {
                    "--pressureDrop requires a positive pressure drop in Pa".to_string()
                })?;
                pressure_drop = Some(parse_positive_f64_arg("--pressureDrop", value)?);
                poiseuille_option_seen = true;
                index += 2;
            }
            "-rho" | "--rho" => {
                let value = args
                    .get(index + 1)
                    .ok_or_else(|| "--rho requires a positive density in kg/m3".to_string())?;
                density = Some(parse_positive_f64_arg("--rho", value)?);
                laminar_simple_option_seen = true;
                index += 2;
            }
            "-mu" | "--mu" => {
                let value = args.get(index + 1).ok_or_else(|| {
                    "--mu requires a positive dynamic viscosity in Pa s".to_string()
                })?;
                dynamic_viscosity = Some(parse_positive_f64_arg("--mu", value)?);
                poiseuille_option_seen = true;
                index += 2;
            }
            "-length" | "--length" => {
                let value = args
                    .get(index + 1)
                    .ok_or_else(|| "--length requires a positive pipe length in m".to_string())?;
                length = Some(parse_positive_f64_arg("--length", value)?);
                poiseuille_option_seen = true;
                index += 2;
            }
            "-diameter" | "--diameter" => {
                let value = args.get(index + 1).ok_or_else(|| {
                    "--diameter requires a positive pipe diameter in m".to_string()
                })?;
                diameter = Some(parse_positive_f64_arg("--diameter", value)?);
                poiseuille_option_seen = true;
                index += 2;
            }
            "-inletPatch" | "--inletPatch" | "-inlet-patch" | "--inlet-patch" => {
                let value = args
                    .get(index + 1)
                    .ok_or_else(|| "--inletPatch requires a patch name".to_string())?;
                if value.trim().is_empty() {
                    return Err("--inletPatch patch name must not be empty".to_string());
                }
                inlet_patch = value.to_string();
                laminar_simple_option_seen = true;
                index += 2;
            }
            "-outletPatch" | "--outletPatch" | "-outlet-patch" | "--outlet-patch" => {
                let value = args
                    .get(index + 1)
                    .ok_or_else(|| "--outletPatch requires a patch name".to_string())?;
                if value.trim().is_empty() {
                    return Err("--outletPatch patch name must not be empty".to_string());
                }
                outlet_patch = value.to_string();
                laminar_simple_option_seen = true;
                index += 2;
            }
            "-wallPatch" | "--wallPatch" | "-wall-patch" | "--wall-patch" => {
                let value = args
                    .get(index + 1)
                    .ok_or_else(|| "--wallPatch requires a patch name".to_string())?;
                if value.trim().is_empty() {
                    return Err("--wallPatch patch name must not be empty".to_string());
                }
                wall_patches.push(value.to_string());
                poiseuille_option_seen = true;
                index += 2;
            }
            "-linearSolver" | "--linearSolver" | "-linear-solver" | "--linear-solver" => {
                let value = args.get(index + 1).ok_or_else(|| {
                    "--linearSolver requires 'cg', 'pcg', or 'jacobi'".to_string()
                })?;
                scalar_diffusion_linear_solver = parse_scalar_diffusion_linear_solver(value)?;
                laminar_linear_solver = Some(parse_laminar_simple_linear_solver(value)?);
                linear_solve_option_seen = true;
                index += 2;
            }
            "-momentumLinearSolver"
            | "--momentumLinearSolver"
            | "-momentum-linear-solver"
            | "--momentum-linear-solver" => {
                let value = args.get(index + 1).ok_or_else(|| {
                    "--momentumLinearSolver requires 'cg', 'pcg', or 'jacobi'".to_string()
                })?;
                momentum_linear_solver = Some(parse_laminar_simple_linear_solver(value)?);
                laminar_simple_option_seen = true;
                index += 2;
            }
            "-pressureLinearSolver"
            | "--pressureLinearSolver"
            | "-pressure-linear-solver"
            | "--pressure-linear-solver" => {
                let value = args.get(index + 1).ok_or_else(|| {
                    "--pressureLinearSolver requires 'cg', 'pcg', or 'jacobi'".to_string()
                })?;
                pressure_linear_solver = Some(parse_laminar_simple_linear_solver(value)?);
                laminar_simple_option_seen = true;
                index += 2;
            }
            "-momentumPreconditioner"
            | "--momentumPreconditioner"
            | "-momentum-preconditioner"
            | "--momentum-preconditioner" => {
                let value = args.get(index + 1).ok_or_else(|| {
                    "--momentumPreconditioner requires 'none', 'diagonal', or 'DIC'".to_string()
                })?;
                momentum_preconditioner = Some(parse_laminar_simple_preconditioner(value)?);
                laminar_simple_option_seen = true;
                index += 2;
            }
            "-pressurePreconditioner"
            | "--pressurePreconditioner"
            | "-pressure-preconditioner"
            | "--pressure-preconditioner" => {
                let value = args.get(index + 1).ok_or_else(|| {
                    "--pressurePreconditioner requires 'none', 'diagonal', or 'DIC'".to_string()
                })?;
                pressure_preconditioner = Some(parse_laminar_simple_preconditioner(value)?);
                laminar_simple_option_seen = true;
                index += 2;
            }
            "-solveTolerance" | "--solveTolerance" | "-solve-tolerance" | "--solve-tolerance" => {
                let value = args
                    .get(index + 1)
                    .ok_or_else(|| "--solveTolerance requires a non-negative number".to_string())?;
                scalar_diffusion_tolerance = parse_non_negative_f64_arg("--solveTolerance", value)?;
                laminar_linear_tolerance = Some(scalar_diffusion_tolerance);
                linear_solve_option_seen = true;
                index += 2;
            }
            "-maxIterations" | "--maxIterations" | "-max-iterations" | "--max-iterations" => {
                let value = args
                    .get(index + 1)
                    .ok_or_else(|| "--maxIterations requires a positive integer".to_string())?;
                scalar_diffusion_max_iterations =
                    parse_positive_usize_arg("--maxIterations", value)?;
                laminar_max_linear_iterations = Some(scalar_diffusion_max_iterations);
                linear_solve_option_seen = true;
                index += 2;
            }
            "-momentumSolveTolerance"
            | "--momentumSolveTolerance"
            | "-momentum-solve-tolerance"
            | "--momentum-solve-tolerance" => {
                let value = args.get(index + 1).ok_or_else(|| {
                    "--momentumSolveTolerance requires a non-negative number".to_string()
                })?;
                momentum_linear_tolerance = Some(parse_non_negative_f64_arg(
                    "--momentumSolveTolerance",
                    value,
                )?);
                laminar_simple_option_seen = true;
                index += 2;
            }
            "-pressureSolveTolerance"
            | "--pressureSolveTolerance"
            | "-pressure-solve-tolerance"
            | "--pressure-solve-tolerance" => {
                let value = args.get(index + 1).ok_or_else(|| {
                    "--pressureSolveTolerance requires a non-negative number".to_string()
                })?;
                pressure_linear_tolerance = Some(parse_non_negative_f64_arg(
                    "--pressureSolveTolerance",
                    value,
                )?);
                laminar_simple_option_seen = true;
                index += 2;
            }
            "-momentumMaxIterations"
            | "--momentumMaxIterations"
            | "-momentum-max-iterations"
            | "--momentum-max-iterations" => {
                let value = args.get(index + 1).ok_or_else(|| {
                    "--momentumMaxIterations requires a positive integer".to_string()
                })?;
                momentum_max_linear_iterations =
                    Some(parse_positive_usize_arg("--momentumMaxIterations", value)?);
                laminar_simple_option_seen = true;
                index += 2;
            }
            "-pressureMaxIterations"
            | "--pressureMaxIterations"
            | "-pressure-max-iterations"
            | "--pressure-max-iterations" => {
                let value = args.get(index + 1).ok_or_else(|| {
                    "--pressureMaxIterations requires a positive integer".to_string()
                })?;
                pressure_max_linear_iterations =
                    Some(parse_positive_usize_arg("--pressureMaxIterations", value)?);
                laminar_simple_option_seen = true;
                index += 2;
            }
            "-maxSimpleIterations"
            | "--maxSimpleIterations"
            | "-max-simple-iterations"
            | "--max-simple-iterations" => {
                let value = args.get(index + 1).ok_or_else(|| {
                    "--maxSimpleIterations requires a positive integer".to_string()
                })?;
                max_simple_iterations = parse_positive_usize_arg("--maxSimpleIterations", value)?;
                laminar_simple_option_seen = true;
                index += 2;
            }
            "-minSimpleIterations"
            | "--minSimpleIterations"
            | "-min-simple-iterations"
            | "--min-simple-iterations" => {
                let value = args.get(index + 1).ok_or_else(|| {
                    "--minSimpleIterations requires a positive integer".to_string()
                })?;
                min_simple_iterations =
                    Some(parse_positive_usize_arg("--minSimpleIterations", value)?);
                laminar_simple_option_seen = true;
                index += 2;
            }
            "-simpleTolerance" | "--simpleTolerance" | "-simple-tolerance"
            | "--simple-tolerance" => {
                let value = args.get(index + 1).ok_or_else(|| {
                    "--simpleTolerance requires a non-negative number".to_string()
                })?;
                simple_tolerance = parse_non_negative_f64_arg("--simpleTolerance", value)?;
                laminar_simple_option_seen = true;
                index += 2;
            }
            "-pressureDropTolerance"
            | "--pressureDropTolerance"
            | "-pressure-drop-tolerance"
            | "--pressure-drop-tolerance" => {
                let value = args.get(index + 1).ok_or_else(|| {
                    "--pressureDropTolerance requires a non-negative relative tolerance".to_string()
                })?;
                pressure_drop_tolerance = Some(parse_non_negative_f64_arg(
                    "--pressureDropTolerance",
                    value,
                )?);
                laminar_simple_option_seen = true;
                index += 2;
            }
            "-fieldChangeTolerance"
            | "--fieldChangeTolerance"
            | "-field-change-tolerance"
            | "--field-change-tolerance" => {
                let value = args.get(index + 1).ok_or_else(|| {
                    "--fieldChangeTolerance requires a non-negative relative tolerance".to_string()
                })?;
                field_change_tolerance =
                    Some(parse_non_negative_f64_arg("--fieldChangeTolerance", value)?);
                laminar_simple_option_seen = true;
                index += 2;
            }
            "-velocityRelaxation"
            | "--velocityRelaxation"
            | "-velocity-relaxation"
            | "--velocity-relaxation" => {
                let value = args.get(index + 1).ok_or_else(|| {
                    "--velocityRelaxation requires a number in (0, 1]".to_string()
                })?;
                velocity_relaxation = Some(parse_relaxation_arg("--velocityRelaxation", value)?);
                laminar_simple_option_seen = true;
                index += 2;
            }
            "-pressureRelaxation"
            | "--pressureRelaxation"
            | "-pressure-relaxation"
            | "--pressure-relaxation" => {
                let value = args.get(index + 1).ok_or_else(|| {
                    "--pressureRelaxation requires a number in (0, 1]".to_string()
                })?;
                pressure_relaxation = Some(parse_relaxation_arg("--pressureRelaxation", value)?);
                laminar_simple_option_seen = true;
                index += 2;
            }
            "-solveReportJson"
            | "--solveReportJson"
            | "-solve-report-json"
            | "--solve-report-json" => {
                let path = args
                    .get(index + 1)
                    .ok_or_else(|| "--solveReportJson requires a file path".to_string())?;
                solve_report_json = Some(PathBuf::from(path));
                laminar_simple_option_seen = true;
                index += 2;
            }
            "-solveReportMarkdown"
            | "--solveReportMarkdown"
            | "-solve-report-markdown"
            | "--solve-report-markdown" => {
                let path = args
                    .get(index + 1)
                    .ok_or_else(|| "--solveReportMarkdown requires a file path".to_string())?;
                solve_report_markdown = Some(PathBuf::from(path));
                laminar_simple_option_seen = true;
                index += 2;
            }
            other => return Err(format!("unknown ferrumSolver option '{other}'")),
        }
    }
    let scalar_diffusion_solve = scalar_diffusion_field.map(|field| ScalarDiffusionSolveArgs {
        field,
        diffusivity: scalar_diffusion_diffusivity,
        source: scalar_diffusion_source,
        linear_solver: scalar_diffusion_linear_solver,
        tolerance: scalar_diffusion_tolerance,
        max_iterations: scalar_diffusion_max_iterations,
    });
    if scalar_diffusion_solve.is_none() && scalar_diffusion_option_seen {
        return Err(
            "scalar diffusion solve options require --solveScalarDiffusion <field>".to_string(),
        );
    }
    let poiseuille_solve = if poiseuille_solve {
        Some(PoiseuilleSolveArgs {
            pressure_drop,
            dynamic_viscosity,
            length,
            diameter,
            wall_patches,
            linear_solver: scalar_diffusion_linear_solver,
            tolerance: scalar_diffusion_tolerance,
            max_iterations: scalar_diffusion_max_iterations,
        })
    } else {
        None
    };
    let laminar_simple_solve = if laminar_simple_solve {
        Some(LaminarSimpleSolveArgs {
            density,
            dynamic_viscosity,
            pressure_drop,
            length,
            diameter,
            inlet_patch,
            outlet_patch,
            linear_solver: laminar_linear_solver,
            momentum_linear_solver,
            pressure_linear_solver,
            momentum_preconditioner,
            pressure_preconditioner,
            linear_tolerance: laminar_linear_tolerance,
            max_linear_iterations: laminar_max_linear_iterations,
            momentum_linear_tolerance,
            pressure_linear_tolerance,
            momentum_max_linear_iterations,
            pressure_max_linear_iterations,
            max_simple_iterations,
            min_simple_iterations,
            simple_tolerance,
            pressure_drop_tolerance,
            field_change_tolerance,
            velocity_relaxation,
            pressure_relaxation,
            report_json: solve_report_json,
            report_markdown: solve_report_markdown,
        })
    } else {
        None
    };
    if poiseuille_solve.is_none() && laminar_simple_solve.is_none() && poiseuille_option_seen {
        return Err("Poiseuille solve options require --solvePoiseuille".to_string());
    }
    if laminar_simple_solve.is_none() && laminar_simple_option_seen {
        return Err("Laminar SIMPLE solve options require --solveLaminarSimple".to_string());
    }
    if scalar_diffusion_solve.is_none()
        && poiseuille_solve.is_none()
        && laminar_simple_solve.is_none()
        && linear_solve_option_seen
    {
        return Err(
            "linear solve options require --solveScalarDiffusion <field>, --solvePoiseuille, or --solveLaminarSimple"
                .to_string(),
        );
    }
    let executable_solve_count = scalar_diffusion_solve.is_some() as usize
        + poiseuille_solve.is_some() as usize
        + laminar_simple_solve.is_some() as usize;
    if executable_solve_count > 1 {
        return Err(
            "--solveScalarDiffusion, --solvePoiseuille, and --solveLaminarSimple cannot be combined in one command yet"
                .to_string(),
        );
    }
    Ok(SolverArgs {
        case_dir,
        plan_json,
        runner_dry_run,
        max_runner_steps,
        scalar_diffusion_solve,
        poiseuille_solve,
        laminar_simple_solve,
    })
}

fn parse_scalar_diffusion_linear_solver(
    value: &str,
) -> Result<ScalarDiffusionLinearSolver, String> {
    match value {
        "cg" | "CG" | "pcg" | "PCG" => Ok(ScalarDiffusionLinearSolver::Cg),
        "jacobi" | "Jacobi" => Ok(ScalarDiffusionLinearSolver::Jacobi),
        other => Err(format!(
            "invalid --linearSolver value '{other}'; expected 'cg' or 'jacobi'"
        )),
    }
}

fn parse_laminar_simple_linear_solver(value: &str) -> Result<LaminarSimpleLinearSolver, String> {
    parse_openfoam_laminar_solver(value).ok_or_else(|| {
        format!("invalid laminar SIMPLE linear solver '{value}'; expected 'cg', 'pcg', or 'jacobi'")
    })
}

fn parse_laminar_simple_preconditioner(value: &str) -> Result<LaminarSimplePreconditioner, String> {
    parse_openfoam_laminar_preconditioner(value).ok_or_else(|| {
        format!(
            "invalid laminar SIMPLE preconditioner '{value}'; expected 'none', 'diagonal', or 'DIC'"
        )
    })
}

fn parse_positive_usize_arg(label: &str, value: &str) -> Result<usize, String> {
    let parsed = value
        .parse::<usize>()
        .map_err(|_| format!("invalid {label} value '{value}'; expected a positive integer"))?;
    if parsed == 0 {
        return Err(format!("{label} must be greater than zero"));
    }
    Ok(parsed)
}

fn parse_finite_f64_arg(label: &str, value: &str) -> Result<f64, String> {
    let parsed = value
        .parse::<f64>()
        .map_err(|_| format!("invalid {label} value '{value}'; expected a finite number"))?;
    if !parsed.is_finite() {
        return Err(format!("{label} must be finite"));
    }
    Ok(parsed)
}

fn parse_positive_f64_arg(label: &str, value: &str) -> Result<f64, String> {
    let parsed = parse_finite_f64_arg(label, value)?;
    if parsed <= 0.0 {
        return Err(format!("{label} must be greater than zero"));
    }
    Ok(parsed)
}

fn parse_non_negative_f64_arg(label: &str, value: &str) -> Result<f64, String> {
    let parsed = parse_finite_f64_arg(label, value)?;
    if parsed < 0.0 {
        return Err(format!("{label} must be non-negative"));
    }
    Ok(parsed)
}

fn parse_relaxation_arg(label: &str, value: &str) -> Result<f64, String> {
    let parsed = parse_finite_f64_arg(label, value)?;
    if parsed <= 0.0 || parsed > 1.0 {
        return Err(format!("{label} must be in (0, 1]"));
    }
    Ok(parsed)
}

fn parse_init_case_args(args: &[String]) -> Result<InitCaseOptions, String> {
    let case_dir = PathBuf::from(
        args.first()
            .ok_or_else(|| "initFerrumCase requires a case directory".to_string())?,
    );
    let mut force = false;
    let mut regions = Vec::new();

    let mut index = 1;
    while index < args.len() {
        match args[index].as_str() {
            "--force" | "-force" => {
                force = true;
                index += 1;
            }
            "-region" | "--region" => {
                let region = args
                    .get(index + 1)
                    .ok_or_else(|| "-region requires a region name".to_string())?;
                regions.push(validate_case_name(region, "region")?);
                index += 2;
            }
            "-regions" | "--regions" => {
                let value = args
                    .get(index + 1)
                    .ok_or_else(|| "-regions requires a comma-separated region list".to_string())?;
                for region in value
                    .split(',')
                    .map(str::trim)
                    .filter(|value| !value.is_empty())
                {
                    regions.push(validate_case_name(region, "region")?);
                }
                index += 2;
            }
            other => return Err(format!("unknown initFerrumCase option '{other}'")),
        }
    }

    regions.sort();
    regions.dedup();

    Ok(InitCaseOptions {
        case_dir,
        force,
        regions,
    })
}

fn validate_case_name(value: &str, label: &str) -> Result<String, String> {
    if value.is_empty() {
        return Err(format!("{label} name must not be empty"));
    }
    if !is_openfoam_word(value) {
        return Err(format!("invalid {label} name '{value}'"));
    }
    Ok(value.to_string())
}

struct GmshToFoamArgs {
    mesh_path: PathBuf,
    case_dir: PathBuf,
    options: FoamWriteOptions,
}

fn parse_gmsh_to_foam_args(args: &[String]) -> Result<GmshToFoamArgs, String> {
    let mesh_path = PathBuf::from(
        args.first()
            .ok_or_else(|| "gmshToFerrumFoam requires a mesh path".to_string())?,
    );
    let mut case_dir = PathBuf::from(".");
    let mut options = FoamWriteOptions::default();

    let mut index = 1;
    while index < args.len() {
        match args[index].as_str() {
            "-case" | "--case" => {
                case_dir = PathBuf::from(
                    args.get(index + 1)
                        .ok_or_else(|| "-case requires a directory".to_string())?,
                );
                index += 2;
            }
            "-emptyPatch" | "--emptyPatch" => {
                let patch = args
                    .get(index + 1)
                    .ok_or_else(|| "-emptyPatch requires a patch name".to_string())?;
                options.set_patch_type(patch, "empty");
                index += 2;
            }
            "-wedgePatch" | "--wedgePatch" => {
                let patch = args
                    .get(index + 1)
                    .ok_or_else(|| "-wedgePatch requires a patch name".to_string())?;
                options.set_patch_type(patch, "wedge");
                index += 2;
            }
            "-symmetryPatch" | "--symmetryPatch" => {
                let patch = args
                    .get(index + 1)
                    .ok_or_else(|| "-symmetryPatch requires a patch name".to_string())?;
                options.set_patch_type(patch, "symmetryPlane");
                index += 2;
            }
            "-patchType" | "--patchType" => {
                let first = args.get(index + 1).ok_or_else(|| {
                    "-patchType requires '<patch>=<type>' or '<patch> <type>'".to_string()
                })?;
                if let Some((patch, patch_type)) = first.split_once('=') {
                    set_validated_patch_type(&mut options, patch, patch_type)?;
                    index += 2;
                } else {
                    let patch_type = args.get(index + 2).ok_or_else(|| {
                        "-patchType requires '<patch>=<type>' or '<patch> <type>'".to_string()
                    })?;
                    set_validated_patch_type(&mut options, first, patch_type)?;
                    index += 3;
                }
            }
            other => return Err(format!("unknown gmshToFerrumFoam option '{other}'")),
        }
    }

    Ok(GmshToFoamArgs {
        mesh_path,
        case_dir,
        options,
    })
}

fn set_validated_patch_type(
    options: &mut FoamWriteOptions,
    patch: &str,
    patch_type: &str,
) -> Result<(), String> {
    if patch.trim().is_empty() {
        return Err("patch name must not be empty".to_string());
    }
    if !is_openfoam_word(patch_type) {
        return Err(format!("invalid OpenFOAM patch type '{patch_type}'"));
    }
    options.set_patch_type(patch, patch_type);
    Ok(())
}

fn is_openfoam_word(value: &str) -> bool {
    !value.is_empty()
        && value
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || ch == '_' || ch == '-')
}

fn display_count(value: Option<usize>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "unknown".to_string())
}

fn is_help(arg: &str) -> bool {
    arg == "-h" || arg == "--help" || arg == "help"
}

fn print_help() {
    println!("FerrumCFD mesh tools");
    println!();
    println!("usage:");
    println!("  ferrum initCase <caseDir> [--region <name> ...] [--force]");
    println!("  ferrum gmshToFoam <mesh.msh> [-case <caseDir>] [patch type options]");
    println!("  ferrum checkMesh [-case <caseDir>]");
    println!("  ferrum splitMeshRegions [-case <caseDir>] [-cellZones]");
    println!("  ferrum solve [-case <caseDir>] [--preflight] [--planJson <file>] [--runnerDryRun]");
    println!();
    println!("aliases:");
    println!("  initFerrumCase <caseDir> [--region <name> ...] [--force]");
    println!("  gmshToFerrumFoam <mesh.msh> [-case <caseDir>] [patch type options]");
    println!("  checkFerrumMesh [-case <caseDir>]");
    println!("  splitFerrumMeshRegions [-case <caseDir>] [-cellZones]");
    println!("  ferrumSolver [-case <caseDir>] [--preflight] [--planJson <file>] [--runnerDryRun]");
    println!();
    print_patch_type_options();
}

fn print_init_case_usage() {
    println!("usage: initFerrumCase <caseDir> [--region <name> ...] [--regions a,b] [--force]");
    println!();
    println!("creates an OpenFOAM-like FerrumCFD case skeleton:");
    println!("  0/");
    println!("  constant/");
    println!("  constant/polyMesh/");
    println!("  constant/interfaces");
    println!("  system/controlDict");
    println!("  system/fvSchemes");
    println!("  system/fvSolution");
    println!("  system/ferrumBackends");
}

fn print_solver_usage() {
    println!(
        "usage: ferrumSolver [-case <caseDir>] [--preflight] [--planJson <file>] [--runnerDryRun] [--maxRunnerSteps <n>] [--solveScalarDiffusion <field>|--solvePoiseuille|--solveLaminarSimple]"
    );
    println!();
    println!("reads a FerrumCFD/OpenFOAM-like case and prints the solver preflight plan:");
    println!("  system/controlDict");
    println!("  system/fvSchemes");
    println!("  system/fvSolution");
    println!("  system/ferrumBackends");
    println!("  constant/polyMesh");
    println!("  constant/<property dictionaries>");
    println!("  constant/interfaces");
    println!("  0/<fields>");
    println!();
    println!("options:");
    println!("  --planJson <file>    also write the solver-neutral plan as JSON");
    println!("  --runnerDryRun       preview the future solver runner without solving equations");
    println!("  --maxRunnerSteps <n> limit runner dry-run preview steps (default: 3)");
    println!("  --solveScalarDiffusion <field> assemble and solve one CPU scalar diffusion system");
    println!("  --solvePoiseuille    solve a source-driven axial Stokes/Poiseuille benchmark");
    println!("  --solveLaminarSimple solve the first laminar incompressible SIMPLE path");
    println!(
        "  --diffusivity <v>    scalar diffusion coefficient for --solveScalarDiffusion (default: 1)"
    );
    println!(
        "  --source <v>         uniform volume source for --solveScalarDiffusion (default: 0)"
    );
    println!(
        "  --linearSolver <s>   cg, pcg, or jacobi for executable solves (default: cg; laminar SIMPLE default: fvSolution or jacobi)"
    );
    println!(
        "  --momentumLinearSolver <s> override laminar SIMPLE momentum solver (cg, pcg, or jacobi)"
    );
    println!(
        "  --pressureLinearSolver <s> override laminar SIMPLE pressure-correction solver (cg, pcg, or jacobi)"
    );
    println!(
        "  --momentumPreconditioner <s> override laminar SIMPLE U preconditioner (none, diagonal, DIC)"
    );
    println!(
        "  --pressurePreconditioner <s> override laminar SIMPLE p preconditioner (none, diagonal, DIC)"
    );
    println!(
        "  --momentumSolveTolerance <v> override laminar SIMPLE U solve tolerance (default: --solveTolerance, fvSolution solvers.U.tolerance, or 1e-10)"
    );
    println!(
        "  --pressureSolveTolerance <v> override laminar SIMPLE p solve tolerance (default: --solveTolerance, fvSolution solvers.p.tolerance, or 1e-10)"
    );
    println!("  --momentumMaxIterations <n> override laminar SIMPLE U linear iteration cap");
    println!("  --pressureMaxIterations <n> override laminar SIMPLE p linear iteration cap");
    println!("  --pressureDrop <Pa>  pressure drop for --solvePoiseuille/--solveLaminarSimple");
    println!("  --rho <kg/m3>        density for --solveLaminarSimple");
    println!("  --mu <Pa.s>          dynamic viscosity for --solvePoiseuille/--solveLaminarSimple");
    println!("  --length <m>         pipe length for --solvePoiseuille/--solveLaminarSimple");
    println!("  --diameter <m>       pipe diameter for --solvePoiseuille/--solveLaminarSimple");
    println!("  --wallPatch <name>   wall patch for --solvePoiseuille (default: wall)");
    println!("  --inletPatch <name>  inlet patch for --solveLaminarSimple (default: inlet)");
    println!("  --outletPatch <name> outlet patch for --solveLaminarSimple (default: outlet)");
    println!(
        "  --maxSimpleIterations <n> SIMPLE iteration cap for --solveLaminarSimple (default: 1)"
    );
    println!(
        "  --minSimpleIterations <n> minimum SIMPLE iterations before convergence (default: 1 for one-step runs, otherwise 2)"
    );
    println!(
        "  --simpleTolerance <v> SIMPLE continuity tolerance for --solveLaminarSimple (default: 1e-8)"
    );
    println!(
        "  --pressureDropTolerance <v> relative Hagen-Poiseuille pressure-drop tolerance for --solveLaminarSimple (default: 0.02)"
    );
    println!(
        "  --fieldChangeTolerance <v> relative U/p field-change tolerance for --solveLaminarSimple (default: 0.01)"
    );
    println!(
        "  --velocityRelaxation <v> override U relaxation for --solveLaminarSimple (default: fvSolution relaxationFactors.equations.U or 0.7)"
    );
    println!(
        "  --pressureRelaxation <v> override p relaxation for --solveLaminarSimple (default: fvSolution relaxationFactors.fields.p or 0.3)"
    );
    println!("  --solveReportJson <file> write --solveLaminarSimple JSON report");
    println!("  --solveReportMarkdown <file> write --solveLaminarSimple Markdown report");
    println!("  --solveTolerance <v> absolute residual tolerance (default: 1e-10)");
    println!("  --maxIterations <n>  linear solver iteration cap (default: 10000)");
    println!();
    println!(
        "CPU scalar diffusion, Poiseuille, and a first laminar SIMPLE path are available; GPU equation kernels are planned"
    );
}

fn print_gmsh_to_foam_usage() {
    println!("usage: gmshToFerrumFoam <mesh.msh> [-case <caseDir>] [patch type options]");
    println!();
    print_patch_type_options();
}

fn print_patch_type_options() {
    println!("patch type options:");
    println!("  -emptyPatch <patch>          write patch type 'empty' for 2D front/back patches");
    println!(
        "  -wedgePatch <patch>          write patch type 'wedge' for axisymmetric wedge patches"
    );
    println!("  -symmetryPatch <patch>       write patch type 'symmetryPlane'");
    println!("  -patchType <patch>=<type>    write any OpenFOAM-compatible patch type");
}

#[allow(dead_code)]
fn normalize_case_path(path: &Path) -> PathBuf {
    path.to_path_buf()
}

#[cfg(test)]
mod tests {
    use super::{
        ScalarDiffusionLinearSolver, SolverNumericsDictionaryPlan, numerics_dictionary_number,
        numerics_dictionary_usize, numerics_dictionary_value,
        parse_openfoam_laminar_preconditioner, parse_openfoam_laminar_solver, parse_solver_args,
        write_json_solver_state, write_json_string,
    };
    use ferrum_mesh::flow::{LaminarSimpleLinearSolver, LaminarSimplePreconditioner};
    use ferrum_mesh::solver_state::{
        SolverStateCpuBufferPlan, SolverStateCpuBufferStatus, SolverStateFieldKind,
        SolverStateFieldPlan, SolverStateInternalFieldPlan, SolverStatePlan,
        SolverStateStoragePlan, SolverStateStorageStatus, SolverStateValueKind,
    };
    use std::path::PathBuf;

    #[test]
    fn parses_solver_plan_json_option() {
        let args = vec![
            "-case".to_string(),
            "cases/reactor".to_string(),
            "--preflight".to_string(),
            "--planJson".to_string(),
            "system/solverPlan.json".to_string(),
            "--runnerDryRun".to_string(),
            "--maxRunnerSteps".to_string(),
            "4".to_string(),
        ];

        let parsed = parse_solver_args(&args).expect("solver args should parse");

        assert_eq!(parsed.case_dir, PathBuf::from("cases/reactor"));
        assert_eq!(
            parsed.plan_json,
            Some(PathBuf::from("system/solverPlan.json"))
        );
        assert!(parsed.runner_dry_run);
        assert_eq!(parsed.max_runner_steps, 4);
    }

    #[test]
    fn rejects_zero_runner_preview_steps() {
        let args = vec!["--maxRunnerSteps".to_string(), "0".to_string()];

        let error = parse_solver_args(&args).expect_err("zero preview steps should fail");

        assert!(error.contains("greater than zero"));
    }

    #[test]
    fn parses_scalar_diffusion_solve_options() {
        let args = vec![
            "-case".to_string(),
            "examples/laminar_pipe".to_string(),
            "--solveScalarDiffusion".to_string(),
            "T".to_string(),
            "--diffusivity".to_string(),
            "0.598".to_string(),
            "--source".to_string(),
            "2.5".to_string(),
            "--linearSolver".to_string(),
            "jacobi".to_string(),
            "--solveTolerance".to_string(),
            "1e-8".to_string(),
            "--maxIterations".to_string(),
            "123".to_string(),
        ];

        let parsed = parse_solver_args(&args).expect("solver args should parse");
        let solve = parsed
            .scalar_diffusion_solve
            .expect("scalar diffusion solve args");

        assert_eq!(solve.field, "T");
        assert_eq!(solve.diffusivity, 0.598);
        assert_eq!(solve.source, 2.5);
        assert_eq!(solve.linear_solver, ScalarDiffusionLinearSolver::Jacobi);
        assert_eq!(solve.tolerance, 1e-8);
        assert_eq!(solve.max_iterations, 123);
    }

    #[test]
    fn rejects_scalar_diffusion_options_without_field() {
        let args = vec!["--diffusivity".to_string(), "1.0".to_string()];

        let error =
            parse_solver_args(&args).expect_err("diffusivity without solve field should fail");

        assert!(error.contains("--solveScalarDiffusion"));
    }

    #[test]
    fn parses_poiseuille_solve_options() {
        let args = vec![
            "--solvePoiseuille".to_string(),
            "--pressureDrop".to_string(),
            "1.6032".to_string(),
            "--mu".to_string(),
            "0.001002".to_string(),
            "--length".to_string(),
            "1.0".to_string(),
            "--diameter".to_string(),
            "0.02".to_string(),
            "--wallPatch".to_string(),
            "pipeWall".to_string(),
            "--linearSolver".to_string(),
            "cg".to_string(),
        ];

        let parsed = parse_solver_args(&args).expect("solver args should parse");
        let solve = parsed.poiseuille_solve.expect("poiseuille solve args");

        assert_eq!(solve.pressure_drop, Some(1.6032));
        assert_eq!(solve.dynamic_viscosity, Some(0.001002));
        assert_eq!(solve.length, Some(1.0));
        assert_eq!(solve.diameter, Some(0.02));
        assert_eq!(solve.wall_patches, vec!["pipeWall"]);
        assert_eq!(solve.linear_solver, ScalarDiffusionLinearSolver::Cg);
    }

    #[test]
    fn parses_laminar_simple_solve_options() {
        let args = vec![
            "--solveLaminarSimple".to_string(),
            "--rho".to_string(),
            "998.2".to_string(),
            "--mu".to_string(),
            "0.001002".to_string(),
            "--pressureDrop".to_string(),
            "1.6032".to_string(),
            "--length".to_string(),
            "1.0".to_string(),
            "--diameter".to_string(),
            "0.02".to_string(),
            "--maxSimpleIterations".to_string(),
            "7".to_string(),
            "--minSimpleIterations".to_string(),
            "3".to_string(),
            "--simpleTolerance".to_string(),
            "1e-7".to_string(),
            "--pressureDropTolerance".to_string(),
            "0.015".to_string(),
            "--fieldChangeTolerance".to_string(),
            "0.005".to_string(),
            "--velocityRelaxation".to_string(),
            "0.6".to_string(),
            "--pressureRelaxation".to_string(),
            "0.2".to_string(),
            "--solveReportJson".to_string(),
            "target/simple.json".to_string(),
            "--solveReportMarkdown".to_string(),
            "target/simple.md".to_string(),
        ];

        let parsed = parse_solver_args(&args).expect("solver args should parse");
        let solve = parsed
            .laminar_simple_solve
            .expect("laminar SIMPLE solve args");

        assert_eq!(solve.density, Some(998.2));
        assert_eq!(solve.dynamic_viscosity, Some(0.001002));
        assert_eq!(solve.pressure_drop, Some(1.6032));
        assert_eq!(solve.length, Some(1.0));
        assert_eq!(solve.diameter, Some(0.02));
        assert_eq!(solve.linear_solver, None);
        assert_eq!(solve.momentum_linear_solver, None);
        assert_eq!(solve.pressure_linear_solver, None);
        assert_eq!(solve.momentum_preconditioner, None);
        assert_eq!(solve.pressure_preconditioner, None);
        assert_eq!(solve.linear_tolerance, None);
        assert_eq!(solve.max_linear_iterations, None);
        assert_eq!(solve.momentum_linear_tolerance, None);
        assert_eq!(solve.pressure_linear_tolerance, None);
        assert_eq!(solve.momentum_max_linear_iterations, None);
        assert_eq!(solve.pressure_max_linear_iterations, None);
        assert_eq!(solve.max_simple_iterations, 7);
        assert_eq!(solve.min_simple_iterations, Some(3));
        assert_eq!(solve.simple_tolerance, 1e-7);
        assert_eq!(solve.pressure_drop_tolerance, Some(0.015));
        assert_eq!(solve.field_change_tolerance, Some(0.005));
        assert_eq!(solve.velocity_relaxation, Some(0.6));
        assert_eq!(solve.pressure_relaxation, Some(0.2));
        assert_eq!(solve.report_json, Some(PathBuf::from("target/simple.json")));
        assert_eq!(
            solve.report_markdown,
            Some(PathBuf::from("target/simple.md"))
        );
    }

    #[test]
    fn parses_laminar_simple_split_linear_solvers() {
        let args = vec![
            "--solveLaminarSimple".to_string(),
            "--linearSolver".to_string(),
            "pcg".to_string(),
            "--momentumLinearSolver".to_string(),
            "cg".to_string(),
            "--pressureLinearSolver".to_string(),
            "pcg".to_string(),
            "--pressurePreconditioner".to_string(),
            "DIC".to_string(),
        ];

        let parsed = parse_solver_args(&args).expect("solver args should parse");
        let solve = parsed
            .laminar_simple_solve
            .expect("laminar SIMPLE solve args");

        assert_eq!(solve.linear_solver, Some(LaminarSimpleLinearSolver::Pcg));
        assert_eq!(
            solve.momentum_linear_solver,
            Some(LaminarSimpleLinearSolver::Cg)
        );
        assert_eq!(
            solve.pressure_linear_solver,
            Some(LaminarSimpleLinearSolver::Pcg)
        );
        assert_eq!(
            solve.pressure_preconditioner,
            Some(LaminarSimplePreconditioner::Diagonal)
        );
    }

    #[test]
    fn parses_laminar_simple_split_linear_controls() {
        let args = vec![
            "--solveLaminarSimple".to_string(),
            "--solveTolerance".to_string(),
            "1e-6".to_string(),
            "--maxIterations".to_string(),
            "200".to_string(),
            "--momentumSolveTolerance".to_string(),
            "1e-7".to_string(),
            "--pressureSolveTolerance".to_string(),
            "1e-9".to_string(),
            "--momentumMaxIterations".to_string(),
            "300".to_string(),
            "--pressureMaxIterations".to_string(),
            "400".to_string(),
        ];

        let parsed = parse_solver_args(&args).expect("solver args should parse");
        let solve = parsed
            .laminar_simple_solve
            .expect("laminar SIMPLE solve args");

        assert_eq!(solve.linear_tolerance, Some(1e-6));
        assert_eq!(solve.max_linear_iterations, Some(200));
        assert_eq!(solve.momentum_linear_tolerance, Some(1e-7));
        assert_eq!(solve.pressure_linear_tolerance, Some(1e-9));
        assert_eq!(solve.momentum_max_linear_iterations, Some(300));
        assert_eq!(solve.pressure_max_linear_iterations, Some(400));
    }

    #[test]
    fn parses_laminar_simple_relaxation_as_case_defaults_when_not_overridden() {
        let args = vec!["--solveLaminarSimple".to_string()];

        let parsed = parse_solver_args(&args).expect("solver args should parse");
        let solve = parsed
            .laminar_simple_solve
            .expect("laminar SIMPLE solve args");

        assert_eq!(solve.velocity_relaxation, None);
        assert_eq!(solve.pressure_relaxation, None);
    }

    #[test]
    fn reads_numerics_dictionary_numbers() {
        let dictionary = SolverNumericsDictionaryPlan {
            present: true,
            sections: Vec::new(),
            entries: vec![
                ferrum_mesh::solver_plan::SolverNumericsEntryPlan {
                    section: "relaxationFactors.fields".to_string(),
                    key: "p".to_string(),
                    value: "0.3".to_string(),
                },
                ferrum_mesh::solver_plan::SolverNumericsEntryPlan {
                    section: "relaxationFactors.equations".to_string(),
                    key: "U".to_string(),
                    value: "0.7".to_string(),
                },
                ferrum_mesh::solver_plan::SolverNumericsEntryPlan {
                    section: "solvers.p".to_string(),
                    key: "solver".to_string(),
                    value: "PCG".to_string(),
                },
                ferrum_mesh::solver_plan::SolverNumericsEntryPlan {
                    section: "solvers.p".to_string(),
                    key: "preconditioner".to_string(),
                    value: "DIC".to_string(),
                },
                ferrum_mesh::solver_plan::SolverNumericsEntryPlan {
                    section: "solvers.p".to_string(),
                    key: "maxIter".to_string(),
                    value: "250".to_string(),
                },
            ],
        };

        assert_eq!(
            numerics_dictionary_number(&dictionary, "relaxationFactors.equations", "U"),
            Some(0.7)
        );
        assert_eq!(
            numerics_dictionary_number(&dictionary, "relaxationFactors.fields", "p"),
            Some(0.3)
        );
        assert_eq!(
            numerics_dictionary_number(&dictionary, "relaxationFactors.fields", "U"),
            None
        );
        assert_eq!(
            numerics_dictionary_usize(&dictionary, "solvers.p", "maxIter"),
            Some(250)
        );
        assert_eq!(
            numerics_dictionary_value(&dictionary, "solvers.p", "solver")
                .and_then(parse_openfoam_laminar_solver),
            Some(LaminarSimpleLinearSolver::Pcg)
        );
        assert_eq!(
            numerics_dictionary_value(&dictionary, "solvers.p", "preconditioner")
                .and_then(parse_openfoam_laminar_preconditioner),
            Some(LaminarSimplePreconditioner::Diagonal)
        );
    }

    #[test]
    fn rejects_mixed_executable_solves() {
        let args = vec![
            "--solveScalarDiffusion".to_string(),
            "T".to_string(),
            "--solvePoiseuille".to_string(),
        ];

        let error = parse_solver_args(&args).expect_err("mixed executable solves should fail");

        assert!(error.contains("cannot be combined"));
    }

    #[test]
    fn escapes_json_strings() {
        let mut output = Vec::new();

        write_json_string(&mut output, "a\"b\\c\n\t").expect("json string should write");

        assert_eq!(
            String::from_utf8(output).expect("valid utf8"),
            "\"a\\\"b\\\\c\\n\\t\""
        );
    }

    #[test]
    fn writes_solver_state_cpu_buffer_json() {
        let plan = SolverStatePlan {
            fields: vec![SolverStateFieldPlan {
                region: None,
                name: "p".to_string(),
                class_name: Some("volScalarField".to_string()),
                kind: SolverStateFieldKind::VolScalar,
                dimensions: Some(vec![
                    "0".to_string(),
                    "2".to_string(),
                    "-2".to_string(),
                    "0".to_string(),
                    "0".to_string(),
                    "0".to_string(),
                    "0".to_string(),
                ]),
                mesh_cells: Some(4),
                mesh_faces: Some(5),
                internal_field: SolverStateInternalFieldPlan {
                    kind: SolverStateValueKind::Uniform,
                    value_count: Some(4),
                    expected_count: Some(4),
                    valid_count: Some(true),
                    uniform_components: Some(vec![0.0]),
                    nonuniform_values: None,
                },
                boundary_patches: 1,
                mesh_boundary_patches: Some(1),
                storage: SolverStateStoragePlan {
                    cpu_capable: true,
                    gpu_capable: true,
                    components: Some(1),
                    scalar_slots: Some(4),
                    bytes_f64: Some(32),
                    status: SolverStateStorageStatus::Loaded,
                },
                cpu_buffer: SolverStateCpuBufferPlan {
                    materializable: true,
                    scalar_slots: Some(4),
                    bytes_f64: Some(32),
                    status: SolverStateCpuBufferStatus::UniformReady,
                },
            }],
            warnings: Vec::new(),
        };
        let mut output = Vec::new();

        write_json_solver_state(&mut output, &plan).expect("solver state json should write");
        let json = String::from_utf8(output).expect("valid utf8");

        assert!(json.contains("\"cpuBuffer\""));
        assert!(json.contains("\"materializable\": true"));
        assert!(json.contains("\"status\": \"uniform-ready\""));
    }
}
