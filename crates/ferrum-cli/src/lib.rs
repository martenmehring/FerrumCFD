mod case;

use std::env;
use std::fs::File;
use std::io::{BufWriter, Error, ErrorKind, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Instant;

use case::{InitCaseOptions, init_case};
use ferrum_mesh::Point3;
use ferrum_mesh::backends::{
    read_backend_config, validate_backend_policy, validate_backend_resources,
};
use ferrum_mesh::check::read_case_summary;
use ferrum_mesh::diffusion::{
    assemble_scalar_diffusion_system, diffusion_assembly_capabilities,
    scalar_diffusion_options_from_field,
};
use ferrum_mesh::fields::{
    FieldBoundaryValidationSummary, FieldFile, FieldValueSummary, InitialFieldSet,
    read_fields_from_directory, read_initial_fields, validate_initial_field_boundaries,
};
use ferrum_mesh::flow::{
    ContinuitySummary, FaceFluxDiagnosticSummary, FlowBoundarySummary, FlowOperatorSummary,
    LaminarSimpleConvectionScheme, LaminarSimpleFieldSummary, LaminarSimpleGradientScheme,
    LaminarSimpleInterpolationScheme, LaminarSimpleIterationSummary, LaminarSimpleLaplacianScheme,
    LaminarSimpleLinearSolver, LaminarSimpleOptions, LaminarSimplePreconditioner,
    LaminarSimpleReport, LaminarSimpleResidualControlSummary, LaminarSimpleSchemes,
    LaminarSimpleSnGradScheme, LaminarSimpleStopReason, LinearSolveSummary,
    MatrixDiagnosticSummary, PressureAssemblyDiagnostics, ScalarDiagnosticSummary,
    VectorDiagnosticSummary, solve_laminar_simple, solve_laminar_simple_with_observer,
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
    LaminarPipeBenchmarkOptions, LaminarPipeBenchmarkSummary, LaminarPlaneChannelBenchmarkOptions,
    LaminarPlaneChannelBenchmarkSummary, PipeAxis, PoiseuilleOptions, poiseuille_diffusion_options,
    poiseuille_reference, summarize_laminar_pipe_solution,
    summarize_laminar_plane_channel_solution, summarize_poiseuille_solution,
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
use ferrum_mesh::solver_state::{
    SolverStateFieldKind, SolverStatePlan, build_solver_state_plan, materialize_cpu_buffer,
};

const OPENFOAM_DEFAULT_LDU_TOLERANCE: f64 = 1.0e-6;
const OPENFOAM_DEFAULT_LDU_MAX_ITERATIONS: usize = 1_000;

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
    FerrumPipeBenchmark,
    FerrumPlaneChannelBenchmark,
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
        CommandMode::Alias(Alias::FerrumPipeBenchmark) => pipe_benchmark(args),
        CommandMode::Alias(Alias::FerrumPlaneChannelBenchmark) => plane_channel_benchmark(args),
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
        "pipeBenchmark" | "ferrumPipeBenchmark" => pipe_benchmark(args),
        "planeChannelBenchmark" | "ferrumPlaneChannelBenchmark" => plane_channel_benchmark(args),
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

#[derive(Debug)]
struct PipeBenchmarkArgs {
    case_dir: PathBuf,
    fields_dir: PathBuf,
    options: LaminarPipeBenchmarkOptions,
    out_json: Option<PathBuf>,
    out_markdown: Option<PathBuf>,
}

fn pipe_benchmark(args: Vec<String>) -> Result<(), String> {
    if args.is_empty() || args.iter().any(|arg| is_help(arg)) {
        print_pipe_benchmark_usage();
        return Ok(());
    }
    let args = parse_pipe_benchmark_args(&args)?;
    let (plan, velocity, pressure) = read_benchmark_fields(&args.case_dir, &args.fields_dir)?;
    let summary = summarize_laminar_pipe_solution(
        &plan.runtime_data.mesh,
        &velocity,
        &pressure,
        &args.options,
    )
    .map_err(|error| error.to_string())?;

    println!(
        "pipeBenchmark result: meanVelocity={} analyticMeanVelocity={} relativeMeanVelocityError={} flowRate={} analyticFlowRate={} pressureDropFromMean={} relativePressureDropFromMeanError={} pressureDropFromOwnerCells={} relativePressureDropFromOwnerCellsError={} minVelocity={} maxVelocity={}",
        format_scientific(summary.mean_velocity),
        format_scientific(summary.analytic_mean_velocity),
        format_scientific(summary.relative_mean_velocity_error),
        format_scientific(summary.flow_rate),
        format_scientific(summary.analytic_flow_rate),
        format_scientific(summary.pressure_drop_from_mean),
        format_scientific(summary.relative_pressure_drop_from_mean_error),
        format_scientific(summary.pressure_drop_from_owner_cells),
        format_scientific(summary.relative_pressure_drop_from_owner_cells_error),
        format_scientific(summary.min_velocity),
        format_scientific(summary.max_velocity),
    );

    if let Some(path) = &args.out_json {
        write_pipe_benchmark_json(&args, &summary, path).map_err(|error| {
            format!(
                "could not write pipe benchmark JSON to {} ({error})",
                path.display()
            )
        })?;
        println!("wrote pipe benchmark json: {}", path.display());
    }
    if let Some(path) = &args.out_markdown {
        write_pipe_benchmark_markdown(&args, &summary, path).map_err(|error| {
            format!(
                "could not write pipe benchmark Markdown to {} ({error})",
                path.display()
            )
        })?;
        println!("wrote pipe benchmark markdown: {}", path.display());
    }
    Ok(())
}

fn read_benchmark_fields(
    case_dir: &Path,
    fields_dir: &Path,
) -> Result<(SolverCasePlan, Vec<Point3>, Vec<f64>), String> {
    let plan = build_solver_case_plan(case_dir).map_err(|error| error.to_string())?;
    let fields =
        read_fields_from_directory(case_dir, fields_dir).map_err(|error| error.to_string())?;
    let state = build_solver_state_plan(case_dir, &fields);
    let velocity_values = benchmark_field_buffer(&state, "U", SolverStateFieldKind::VolVector)?;
    let pressure = benchmark_field_buffer(&state, "p", SolverStateFieldKind::VolScalar)?;
    if velocity_values.len() % 3 != 0 {
        return Err(format!(
            "benchmark U field has {} scalar values, expected a multiple of 3",
            velocity_values.len()
        ));
    }
    let velocity = velocity_values
        .chunks_exact(3)
        .map(|value| Point3 {
            x: value[0],
            y: value[1],
            z: value[2],
        })
        .collect::<Vec<_>>();
    Ok((plan, velocity, pressure))
}

fn benchmark_field_buffer(
    state: &SolverStatePlan,
    name: &str,
    kind: SolverStateFieldKind,
) -> Result<Vec<f64>, String> {
    let available = state
        .fields
        .iter()
        .map(|field| {
            format!(
                "{}:class={}:kind={}",
                field.name,
                field.class_name.as_deref().unwrap_or("missing"),
                field.kind
            )
        })
        .collect::<Vec<_>>()
        .join(", ");
    let field = state
        .fields
        .iter()
        .find(|field| field.region.is_none() && field.name == name && field.kind == kind)
        .ok_or_else(|| {
            format!(
                "benchmark field '{name}' with kind {kind} was not found; parsed fields: [{}]",
                available
            )
        })?;
    materialize_cpu_buffer(field).ok_or_else(|| {
        format!(
            "benchmark field '{name}' could not be materialized ({})",
            field.cpu_buffer.status
        )
    })
}

fn parse_pipe_benchmark_args(args: &[String]) -> Result<PipeBenchmarkArgs, String> {
    let mut case_dir = PathBuf::from(".");
    let mut fields_dir = None;
    let mut pressure_drop = None;
    let mut dynamic_viscosity = None;
    let mut length = None;
    let mut diameter = None;
    let mut inlet_patch = "inlet".to_string();
    let mut outlet_patch = "outlet".to_string();
    let mut axis = PipeAxis::X;
    let mut out_json = None;
    let mut out_markdown = None;
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
            "-fields" | "--fields" => {
                fields_dir =
                    Some(PathBuf::from(args.get(index + 1).ok_or_else(|| {
                        "--fields requires a time/field directory".to_string()
                    })?));
                index += 2;
            }
            "-pressureDrop" | "--pressureDrop" | "-pressure-drop" | "--pressure-drop" => {
                pressure_drop = Some(parse_positive_f64_arg(
                    "--pressureDrop",
                    args.get(index + 1)
                        .ok_or_else(|| "--pressureDrop requires Pa".to_string())?,
                )?);
                index += 2;
            }
            "-mu" | "--mu" => {
                dynamic_viscosity = Some(parse_positive_f64_arg(
                    "--mu",
                    args.get(index + 1)
                        .ok_or_else(|| "--mu requires Pa s".to_string())?,
                )?);
                index += 2;
            }
            "-length" | "--length" => {
                length = Some(parse_positive_f64_arg(
                    "--length",
                    args.get(index + 1)
                        .ok_or_else(|| "--length requires m".to_string())?,
                )?);
                index += 2;
            }
            "-diameter" | "--diameter" => {
                diameter = Some(parse_positive_f64_arg(
                    "--diameter",
                    args.get(index + 1)
                        .ok_or_else(|| "--diameter requires m".to_string())?,
                )?);
                index += 2;
            }
            "-inletPatch" | "--inletPatch" | "-inlet-patch" | "--inlet-patch" => {
                inlet_patch = required_non_empty_arg(args, index, "--inletPatch")?;
                index += 2;
            }
            "-outletPatch" | "--outletPatch" | "-outlet-patch" | "--outlet-patch" => {
                outlet_patch = required_non_empty_arg(args, index, "--outletPatch")?;
                index += 2;
            }
            "-axis" | "--axis" => {
                axis = match args
                    .get(index + 1)
                    .ok_or_else(|| "--axis requires x, y, or z".to_string())?
                    .to_ascii_lowercase()
                    .as_str()
                {
                    "x" => PipeAxis::X,
                    "y" => PipeAxis::Y,
                    "z" => PipeAxis::Z,
                    other => return Err(format!("invalid --axis '{other}'; expected x, y, or z")),
                };
                index += 2;
            }
            "-outJson" | "--outJson" | "-out-json" | "--out-json" => {
                out_json = Some(PathBuf::from(
                    args.get(index + 1)
                        .ok_or_else(|| "--outJson requires a file".to_string())?,
                ));
                index += 2;
            }
            "-outMarkdown" | "--outMarkdown" | "-out-markdown" | "--out-markdown" => {
                out_markdown =
                    Some(PathBuf::from(args.get(index + 1).ok_or_else(|| {
                        "--outMarkdown requires a file".to_string()
                    })?));
                index += 2;
            }
            other => return Err(format!("unknown ferrumPipeBenchmark option '{other}'")),
        }
    }

    Ok(PipeBenchmarkArgs {
        case_dir,
        fields_dir: fields_dir
            .ok_or_else(|| "ferrumPipeBenchmark requires --fields".to_string())?,
        options: LaminarPipeBenchmarkOptions {
            pressure_drop: pressure_drop
                .ok_or_else(|| "ferrumPipeBenchmark requires --pressureDrop".to_string())?,
            dynamic_viscosity: dynamic_viscosity
                .ok_or_else(|| "ferrumPipeBenchmark requires --mu".to_string())?,
            length: length.ok_or_else(|| "ferrumPipeBenchmark requires --length".to_string())?,
            diameter: diameter
                .ok_or_else(|| "ferrumPipeBenchmark requires --diameter".to_string())?,
            inlet_patch,
            outlet_patch,
            axis,
        },
        out_json,
        out_markdown,
    })
}

fn required_non_empty_arg(args: &[String], index: usize, flag: &str) -> Result<String, String> {
    let value = args
        .get(index + 1)
        .ok_or_else(|| format!("{flag} requires a value"))?;
    if value.trim().is_empty() {
        return Err(format!("{flag} must not be empty"));
    }
    Ok(value.to_string())
}

fn write_pipe_benchmark_json(
    args: &PipeBenchmarkArgs,
    summary: &LaminarPipeBenchmarkSummary,
    path: &Path,
) -> std::io::Result<()> {
    ensure_parent_dir(path)?;
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(writer, "{{")?;
    write_json_string_field(&mut writer, 2, "benchmark", "laminarPipeHagenPoiseuille")?;
    writeln!(writer, ",")?;
    write_json_string_field(
        &mut writer,
        2,
        "caseDir",
        &args.case_dir.display().to_string(),
    )?;
    writeln!(writer, ",")?;
    write_json_string_field(
        &mut writer,
        2,
        "fieldsDir",
        &args.fields_dir.display().to_string(),
    )?;
    writeln!(writer, ",")?;
    write_json_key(&mut writer, 2, "inputs")?;
    writeln!(writer, "{{")?;
    write_json_key(&mut writer, 4, "pressureDrop")?;
    write_json_optional_number(&mut writer, Some(args.options.pressure_drop))?;
    writeln!(writer, ",")?;
    write_json_key(&mut writer, 4, "dynamicViscosity")?;
    write_json_optional_number(&mut writer, Some(args.options.dynamic_viscosity))?;
    writeln!(writer, ",")?;
    write_json_key(&mut writer, 4, "length")?;
    write_json_optional_number(&mut writer, Some(args.options.length))?;
    writeln!(writer, ",")?;
    write_json_key(&mut writer, 4, "diameter")?;
    write_json_optional_number(&mut writer, Some(args.options.diameter))?;
    writeln!(writer, ",")?;
    write_json_string_field(&mut writer, 4, "inletPatch", &args.options.inlet_patch)?;
    writeln!(writer, ",")?;
    write_json_string_field(&mut writer, 4, "outletPatch", &args.options.outlet_patch)?;
    writeln!(writer, ",")?;
    write_json_string_field(&mut writer, 4, "axis", pipe_axis_name(args.options.axis))?;
    writeln!(writer)?;
    write_indent(&mut writer, 2)?;
    writeln!(writer, "}},")?;
    write_json_key(&mut writer, 2, "solution")?;
    writeln!(writer, "{{")?;
    write_pipe_benchmark_summary_json(&mut writer, summary)?;
    writeln!(writer)?;
    write_indent(&mut writer, 2)?;
    writeln!(writer, "}}")?;
    writeln!(writer, "}}")?;
    writer.flush()
}

fn write_pipe_benchmark_summary_json(
    writer: &mut impl Write,
    summary: &LaminarPipeBenchmarkSummary,
) -> std::io::Result<()> {
    let values = [
        ("minVelocity", summary.min_velocity),
        ("maxVelocity", summary.max_velocity),
        ("meanVelocity", summary.mean_velocity),
        ("flowRate", summary.flow_rate),
        ("analyticMeanVelocity", summary.analytic_mean_velocity),
        ("analyticFlowRate", summary.analytic_flow_rate),
        ("pressureDropFromMean", summary.pressure_drop_from_mean),
        (
            "pressureDropFromOwnerCells",
            summary.pressure_drop_from_owner_cells,
        ),
        (
            "relativeMeanVelocityError",
            summary.relative_mean_velocity_error,
        ),
        (
            "relativePressureDropFromMeanError",
            summary.relative_pressure_drop_from_mean_error,
        ),
        (
            "relativePressureDropFromOwnerCellsError",
            summary.relative_pressure_drop_from_owner_cells_error,
        ),
    ];
    for (index, (key, value)) in values.iter().enumerate() {
        write_json_key(writer, 4, key)?;
        write_json_optional_number(writer, Some(*value))?;
        if index + 1 != values.len() {
            writeln!(writer, ",")?;
        }
    }
    Ok(())
}

fn write_pipe_benchmark_markdown(
    args: &PipeBenchmarkArgs,
    summary: &LaminarPipeBenchmarkSummary,
    path: &Path,
) -> std::io::Result<()> {
    ensure_parent_dir(path)?;
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(writer, "# Laminar Pipe Benchmark")?;
    writeln!(writer)?;
    writeln!(writer, "Case: `{}`", args.case_dir.display())?;
    writeln!(writer, "Fields: `{}`", args.fields_dir.display())?;
    writeln!(writer)?;
    writeln!(writer, "| Quantity | Value |")?;
    writeln!(writer, "| --- | ---: |")?;
    writeln!(writer, "| Axis | {} |", pipe_axis_name(args.options.axis))?;
    writeln!(
        writer,
        "| Analytic deltaP [Pa] | {} |",
        format_scientific(args.options.pressure_drop)
    )?;
    writeln!(
        writer,
        "| Mean velocity [m/s] | {} |",
        format_scientific(summary.mean_velocity)
    )?;
    writeln!(
        writer,
        "| Analytic mean velocity [m/s] | {} |",
        format_scientific(summary.analytic_mean_velocity)
    )?;
    writeln!(
        writer,
        "| Mean velocity error | {} |",
        format_percent(summary.relative_mean_velocity_error)
    )?;
    writeln!(
        writer,
        "| DeltaP from mean velocity [Pa] | {} |",
        format_scientific(summary.pressure_drop_from_mean)
    )?;
    writeln!(
        writer,
        "| DeltaP from owner cells [Pa] | {} |",
        format_scientific(summary.pressure_drop_from_owner_cells)
    )?;
    writeln!(
        writer,
        "| Owner-cell deltaP error | {} |",
        format_percent(summary.relative_pressure_drop_from_owner_cells_error)
    )?;
    writer.flush()
}

#[derive(Debug)]
struct PlaneChannelBenchmarkArgs {
    case_dir: PathBuf,
    fields_dir: PathBuf,
    options: LaminarPlaneChannelBenchmarkOptions,
    pressure_scale: f64,
    out_json: Option<PathBuf>,
    out_markdown: Option<PathBuf>,
}

fn plane_channel_benchmark(args: Vec<String>) -> Result<(), String> {
    if args.is_empty() || args.iter().any(|arg| is_help(arg)) {
        print_plane_channel_benchmark_usage();
        return Ok(());
    }
    let args = parse_plane_channel_benchmark_args(&args)?;
    let (plan, velocity, mut pressure) = read_benchmark_fields(&args.case_dir, &args.fields_dir)?;
    for value in &mut pressure {
        *value *= args.pressure_scale;
    }
    let summary = summarize_laminar_plane_channel_solution(
        &plan.runtime_data.mesh,
        &velocity,
        &pressure,
        &args.options,
    )
    .map_err(|error| error.to_string())?;

    println!(
        "planeChannelBenchmark result: meanVelocity={} analyticMeanVelocity={} relativeMeanVelocityError={} flowRate={} flowRatePerUnitDepth={} pressureDropFromMean={} relativePressureDropFromMeanError={} pressureDropFromOwnerCells={} relativePressureDropFromOwnerCellsError={} minVelocity={} maxVelocity={}",
        format_scientific(summary.mean_velocity),
        format_scientific(summary.analytic_mean_velocity),
        format_scientific(summary.relative_mean_velocity_error),
        format_scientific(summary.flow_rate),
        format_scientific(summary.flow_rate_per_unit_depth),
        format_scientific(summary.pressure_drop_from_mean),
        format_scientific(summary.relative_pressure_drop_from_mean_error),
        format_scientific(summary.pressure_drop_from_owner_cells),
        format_scientific(summary.relative_pressure_drop_from_owner_cells_error),
        format_scientific(summary.min_velocity),
        format_scientific(summary.max_velocity),
    );

    if let Some(path) = &args.out_json {
        write_plane_channel_benchmark_json(&args, &summary, path).map_err(|error| {
            format!(
                "could not write plane-channel benchmark JSON to {} ({error})",
                path.display()
            )
        })?;
        println!("wrote plane-channel benchmark json: {}", path.display());
    }
    if let Some(path) = &args.out_markdown {
        write_plane_channel_benchmark_markdown(&args, &summary, path).map_err(|error| {
            format!(
                "could not write plane-channel benchmark Markdown to {} ({error})",
                path.display()
            )
        })?;
        println!("wrote plane-channel benchmark markdown: {}", path.display());
    }
    Ok(())
}

fn parse_plane_channel_benchmark_args(
    args: &[String],
) -> Result<PlaneChannelBenchmarkArgs, String> {
    let mut case_dir = PathBuf::from(".");
    let mut fields_dir = None;
    let mut pressure_drop = None;
    let mut dynamic_viscosity = None;
    let mut length = None;
    let mut gap = None;
    let mut depth = None;
    let mut inlet_patch = "inlet".to_string();
    let mut outlet_patch = "outlet".to_string();
    let mut axis = PipeAxis::X;
    let mut pressure_scale = 1.0;
    let mut out_json = None;
    let mut out_markdown = None;
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
            "-fields" | "--fields" => {
                fields_dir =
                    Some(PathBuf::from(args.get(index + 1).ok_or_else(|| {
                        "--fields requires a time/field directory".to_string()
                    })?));
                index += 2;
            }
            "-pressureDrop" | "--pressureDrop" | "-pressure-drop" | "--pressure-drop" => {
                pressure_drop = Some(parse_positive_f64_arg(
                    "--pressureDrop",
                    args.get(index + 1)
                        .ok_or_else(|| "--pressureDrop requires Pa".to_string())?,
                )?);
                index += 2;
            }
            "-mu" | "--mu" => {
                dynamic_viscosity = Some(parse_positive_f64_arg(
                    "--mu",
                    args.get(index + 1)
                        .ok_or_else(|| "--mu requires Pa s".to_string())?,
                )?);
                index += 2;
            }
            "-length" | "--length" => {
                length = Some(parse_positive_f64_arg(
                    "--length",
                    args.get(index + 1)
                        .ok_or_else(|| "--length requires m".to_string())?,
                )?);
                index += 2;
            }
            "-gap" | "--gap" => {
                gap = Some(parse_positive_f64_arg(
                    "--gap",
                    args.get(index + 1)
                        .ok_or_else(|| "--gap requires m".to_string())?,
                )?);
                index += 2;
            }
            "-depth" | "--depth" => {
                depth = Some(parse_positive_f64_arg(
                    "--depth",
                    args.get(index + 1)
                        .ok_or_else(|| "--depth requires m".to_string())?,
                )?);
                index += 2;
            }
            "-inletPatch" | "--inletPatch" | "-inlet-patch" | "--inlet-patch" => {
                inlet_patch = required_non_empty_arg(args, index, "--inletPatch")?;
                index += 2;
            }
            "-outletPatch" | "--outletPatch" | "-outlet-patch" | "--outlet-patch" => {
                outlet_patch = required_non_empty_arg(args, index, "--outletPatch")?;
                index += 2;
            }
            "-axis" | "--axis" => {
                axis = parse_pipe_axis(
                    args.get(index + 1)
                        .ok_or_else(|| "--axis requires x, y, or z".to_string())?,
                )?;
                index += 2;
            }
            "-pressureScale" | "--pressureScale" | "-pressure-scale" | "--pressure-scale" => {
                pressure_scale = parse_positive_f64_arg(
                    "--pressureScale",
                    args.get(index + 1)
                        .ok_or_else(|| "--pressureScale requires a positive factor".to_string())?,
                )?;
                index += 2;
            }
            "-outJson" | "--outJson" | "-out-json" | "--out-json" => {
                out_json = Some(PathBuf::from(
                    args.get(index + 1)
                        .ok_or_else(|| "--outJson requires a file".to_string())?,
                ));
                index += 2;
            }
            "-outMarkdown" | "--outMarkdown" | "-out-markdown" | "--out-markdown" => {
                out_markdown =
                    Some(PathBuf::from(args.get(index + 1).ok_or_else(|| {
                        "--outMarkdown requires a file".to_string()
                    })?));
                index += 2;
            }
            other => {
                return Err(format!(
                    "unknown ferrumPlaneChannelBenchmark option '{other}'"
                ));
            }
        }
    }

    Ok(PlaneChannelBenchmarkArgs {
        case_dir,
        fields_dir: fields_dir
            .ok_or_else(|| "ferrumPlaneChannelBenchmark requires --fields".to_string())?,
        options: LaminarPlaneChannelBenchmarkOptions {
            pressure_drop: pressure_drop
                .ok_or_else(|| "ferrumPlaneChannelBenchmark requires --pressureDrop".to_string())?,
            dynamic_viscosity: dynamic_viscosity
                .ok_or_else(|| "ferrumPlaneChannelBenchmark requires --mu".to_string())?,
            length: length
                .ok_or_else(|| "ferrumPlaneChannelBenchmark requires --length".to_string())?,
            gap: gap.ok_or_else(|| "ferrumPlaneChannelBenchmark requires --gap".to_string())?,
            depth: depth
                .ok_or_else(|| "ferrumPlaneChannelBenchmark requires --depth".to_string())?,
            inlet_patch,
            outlet_patch,
            axis,
        },
        pressure_scale,
        out_json,
        out_markdown,
    })
}

fn parse_pipe_axis(value: &str) -> Result<PipeAxis, String> {
    match value.to_ascii_lowercase().as_str() {
        "x" => Ok(PipeAxis::X),
        "y" => Ok(PipeAxis::Y),
        "z" => Ok(PipeAxis::Z),
        other => Err(format!("invalid --axis '{other}'; expected x, y, or z")),
    }
}

fn write_plane_channel_benchmark_json(
    args: &PlaneChannelBenchmarkArgs,
    summary: &LaminarPlaneChannelBenchmarkSummary,
    path: &Path,
) -> std::io::Result<()> {
    ensure_parent_dir(path)?;
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(writer, "{{")?;
    write_json_string_field(&mut writer, 2, "benchmark", "laminarPlanePoiseuille")?;
    writeln!(writer, ",")?;
    write_json_string_field(
        &mut writer,
        2,
        "caseDir",
        &args.case_dir.display().to_string(),
    )?;
    writeln!(writer, ",")?;
    write_json_string_field(
        &mut writer,
        2,
        "fieldsDir",
        &args.fields_dir.display().to_string(),
    )?;
    writeln!(writer, ",")?;
    write_json_key(&mut writer, 2, "inputs")?;
    writeln!(writer, "{{")?;
    for (key, value) in [
        ("pressureDrop", args.options.pressure_drop),
        ("dynamicViscosity", args.options.dynamic_viscosity),
        ("length", args.options.length),
        ("gap", args.options.gap),
        ("depth", args.options.depth),
        ("pressureScale", args.pressure_scale),
    ]
    .iter()
    {
        write_json_key(&mut writer, 4, key)?;
        write_json_optional_number(&mut writer, Some(*value))?;
        writeln!(writer, ",")?;
    }
    write_json_string_field(&mut writer, 4, "inletPatch", &args.options.inlet_patch)?;
    writeln!(writer, ",")?;
    write_json_string_field(&mut writer, 4, "outletPatch", &args.options.outlet_patch)?;
    writeln!(writer, ",")?;
    write_json_string_field(&mut writer, 4, "axis", pipe_axis_name(args.options.axis))?;
    writeln!(writer)?;
    write_indent(&mut writer, 2)?;
    writeln!(writer, "}},")?;
    write_json_key(&mut writer, 2, "solution")?;
    writeln!(writer, "{{")?;
    let values = [
        ("minVelocity", summary.min_velocity),
        ("maxVelocity", summary.max_velocity),
        ("meanVelocity", summary.mean_velocity),
        ("flowRate", summary.flow_rate),
        ("flowRatePerUnitDepth", summary.flow_rate_per_unit_depth),
        ("analyticMeanVelocity", summary.analytic_mean_velocity),
        ("analyticFlowRate", summary.analytic_flow_rate),
        (
            "analyticFlowRatePerUnitDepth",
            summary.analytic_flow_rate_per_unit_depth,
        ),
        ("pressureDropFromMean", summary.pressure_drop_from_mean),
        (
            "pressureDropFromOwnerCells",
            summary.pressure_drop_from_owner_cells,
        ),
        (
            "relativeMeanVelocityError",
            summary.relative_mean_velocity_error,
        ),
        (
            "relativePressureDropFromMeanError",
            summary.relative_pressure_drop_from_mean_error,
        ),
        (
            "relativePressureDropFromOwnerCellsError",
            summary.relative_pressure_drop_from_owner_cells_error,
        ),
    ];
    for (index, (key, value)) in values.iter().enumerate() {
        write_json_key(&mut writer, 4, key)?;
        write_json_optional_number(&mut writer, Some(*value))?;
        if index + 1 != values.len() {
            writeln!(writer, ",")?;
        }
    }
    writeln!(writer)?;
    write_indent(&mut writer, 2)?;
    writeln!(writer, "}}")?;
    writeln!(writer, "}}")?;
    writer.flush()
}

fn write_plane_channel_benchmark_markdown(
    args: &PlaneChannelBenchmarkArgs,
    summary: &LaminarPlaneChannelBenchmarkSummary,
    path: &Path,
) -> std::io::Result<()> {
    ensure_parent_dir(path)?;
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(writer, "# Laminar Plane-Channel Benchmark")?;
    writeln!(writer)?;
    writeln!(writer, "Case: `{}`", args.case_dir.display())?;
    writeln!(writer, "Fields: `{}`", args.fields_dir.display())?;
    writeln!(writer)?;
    writeln!(writer, "| Quantity | Value |")?;
    writeln!(writer, "| --- | ---: |")?;
    writeln!(writer, "| Axis | {} |", pipe_axis_name(args.options.axis))?;
    writeln!(
        writer,
        "| Analytic deltaP [Pa] | {} |",
        format_scientific(args.options.pressure_drop)
    )?;
    writeln!(
        writer,
        "| Mean velocity [m/s] | {} |",
        format_scientific(summary.mean_velocity)
    )?;
    writeln!(
        writer,
        "| Analytic mean velocity [m/s] | {} |",
        format_scientific(summary.analytic_mean_velocity)
    )?;
    writeln!(
        writer,
        "| Mean velocity error | {} |",
        format_percent(summary.relative_mean_velocity_error)
    )?;
    writeln!(
        writer,
        "| DeltaP from mean velocity [Pa] | {} |",
        format_scientific(summary.pressure_drop_from_mean)
    )?;
    writeln!(
        writer,
        "| DeltaP from mean velocity error | {} |",
        format_percent(summary.relative_pressure_drop_from_mean_error)
    )?;
    writeln!(
        writer,
        "| DeltaP from owner cells [Pa] | {} |",
        format_scientific(summary.pressure_drop_from_owner_cells)
    )?;
    writeln!(
        writer,
        "| Owner-cell deltaP error | {} |",
        format_percent(summary.relative_pressure_drop_from_owner_cells_error)
    )?;
    writeln!(
        writer,
        "| Flow rate per unit depth [m2/s] | {} |",
        format_scientific(summary.flow_rate_per_unit_depth)
    )?;
    writer.flush()
}

fn pipe_axis_name(axis: PipeAxis) -> &'static str {
    match axis {
        PipeAxis::X => "x",
        PipeAxis::Y => "y",
        PipeAxis::Z => "z",
    }
}

fn resolve_poiseuille_options(
    plan: &SolverCasePlan,
    solve: &PoiseuilleSolveArgs,
) -> Result<PoiseuilleOptions, String> {
    let pressure_drop = solve
        .pressure_drop
        .ok_or_else(|| "Poiseuille solve requires --pressureDrop".to_string())?;
    let dynamic_viscosity = solve
        .dynamic_viscosity
        .or_else(|| property_number(plan, "transportProperties", None, "mu"))
        .ok_or_else(|| "Poiseuille solve requires --mu or transportProperties.mu".to_string())?;
    let length = solve
        .length
        .ok_or_else(|| "Poiseuille solve requires --length".to_string())?;
    let diameter = solve
        .diameter
        .ok_or_else(|| "Poiseuille solve requires --diameter".to_string())?;
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
    let report = if solve.solve_verbose {
        let mut printed_header = false;
        let mut print_iteration = |item: &LaminarSimpleIterationSummary| {
            if !printed_header {
                println!(
                    "laminarSimple residual history (OpenFOAM-style initial/final residuals; linear and outer convergence are separate):"
                );
                printed_header = true;
            }
            println!(
                "  SIMPLE {:>4}: U initial={} final={} linearConverged={} linearIterations={} | p initial={} final={} linearConverged={} linearIterations={} | continuityL2={} | residualControl={}",
                item.iteration,
                format_scientific(item.momentum_initial_normalized_residual_norm),
                format_scientific(item.momentum_normalized_residual_norm),
                yes_no(item.momentum_linear_converged),
                item.momentum_linear_iterations,
                format_scientific(item.pressure_correction_initial_normalized_residual_norm),
                format_scientific(item.pressure_correction_normalized_residual_norm),
                yes_no(item.pressure_linear_converged),
                item.pressure_linear_iterations,
                format_scientific(item.continuity_after.l2_norm),
                residual_control_state(item.residual_control),
            );
            let _ = std::io::stdout().flush();
        };
        solve_laminar_simple_with_observer(
            &plan.runtime_data,
            &fields,
            &options,
            Some(&mut print_iteration),
        )
        .map_err(|error| error.to_string())?
    } else {
        solve_laminar_simple(&plan.runtime_data, &fields, &options)
            .map_err(|error| error.to_string())?
    };
    let wall_clock_seconds = started.elapsed().as_secs_f64();

    println!(
        "laminarSimple solve: backend=cpu linearSolver={} momentumLinearSolver={} momentumPreconditioner={} pressureLinearSolver={} pressurePreconditioner={} divPhiU=\"{}\" gradP=\"{}\" gradU=\"{}\" laplacian=\"{}\" snGrad=\"{}\" interpolation=\"{}\" pRefCell={} pRefValue={} nonOrthogonalCorrectors={} consistent={} stopReason={} cells={} faces={} simpleIterations={} minSimpleIterations={} converged={} residualControl={} initialContinuityL2={} finalContinuityL2={} momentumInitialResidual={} momentumFinalResidual={} momentumResidualNorm={} pressureInitialResidual={} pressureFinalResidual={} pressureResidualNorm={} momentumLinearIterations={} pressureLinearIterations={} wallClockSeconds={:.6}",
        options.linear_solver,
        options.momentum_linear_solver,
        options.momentum_preconditioner,
        options.pressure_linear_solver,
        options.pressure_preconditioner,
        options.schemes.div_phi_u,
        options.schemes.grad_p,
        options.schemes.grad_u,
        options.schemes.laplacian,
        options.schemes.sn_grad,
        options.schemes.interpolation,
        options
            .pressure_reference_cell
            .map(|value| value.to_string())
            .unwrap_or_else(|| "n/a".to_string()),
        format_scientific(options.pressure_reference_value),
        options.non_orthogonal_correctors,
        yes_no(options.simple_consistent),
        report.stop_reason,
        report.cells,
        report.faces,
        report.simple_iterations,
        options.min_simple_iterations,
        yes_no(report.converged),
        residual_control_state(report.residual_control),
        format_scientific(report.initial_continuity.l2_norm),
        format_scientific(report.final_continuity.l2_norm),
        format_scientific(report.final_momentum_initial_normalized_residual_norm),
        format_scientific(report.final_momentum_normalized_residual_norm),
        format_scientific(report.final_momentum_residual_norm),
        format_scientific(report.final_pressure_correction_initial_normalized_residual_norm),
        format_scientific(report.final_pressure_correction_normalized_residual_norm),
        format_scientific(report.final_pressure_correction_residual_norm),
        report.total_momentum_linear_iterations,
        report.total_pressure_linear_iterations,
        wall_clock_seconds
    );
    println!(
        "laminarSimple residualControl: state={} checked={} satisfied={} U(tolerance={},initial={},satisfied={}) p(tolerance={},initial={},satisfied={})",
        residual_control_state(report.residual_control),
        yes_no(report.residual_control.checked),
        yes_no(report.residual_control.satisfied),
        format_optional_scientific(options.momentum_residual_control),
        format_scientific(report.final_momentum_initial_normalized_residual_norm),
        format_optional_bool(report.residual_control.momentum_satisfied),
        format_optional_scientific(options.pressure_residual_control),
        format_scientific(report.final_pressure_correction_initial_normalized_residual_norm),
        format_optional_bool(report.residual_control.pressure_satisfied),
    );
    println!(
        "laminarSimple linearSolves: finalMomentumConverged={} finalPressureConverged={} momentumPredictors={} momentumNonConvergedPredictors={} momentumComponentSolves={} momentumComponentNonConvergedSolves={} pressureCorrectionSolves={} pressureCorrectionNonConvergedSolves={} maxMomentumIterationsPerSimple={} maxPressureIterationsPerSimple={} avgMomentumIterationsPerSimple={} avgPressureIterationsPerSimple={}",
        yes_no(report.linear_solve_summary.final_momentum_linear_converged),
        yes_no(report.linear_solve_summary.final_pressure_linear_converged),
        report.linear_solve_summary.momentum_predictors,
        report
            .linear_solve_summary
            .momentum_non_converged_predictors,
        report.linear_solve_summary.momentum_component_solves,
        report
            .linear_solve_summary
            .momentum_component_non_converged_solves,
        report.linear_solve_summary.pressure_correction_solves,
        report
            .linear_solve_summary
            .pressure_correction_non_converged_solves,
        report
            .linear_solve_summary
            .max_momentum_linear_iterations_per_simple,
        report
            .linear_solve_summary
            .max_pressure_linear_iterations_per_simple,
        format_scientific(
            report
                .linear_solve_summary
                .average_momentum_linear_iterations_per_simple,
        ),
        format_scientific(
            report
                .linear_solve_summary
                .average_pressure_linear_iterations_per_simple,
        )
    );
    println!(
        "laminarSimple fields: velocityMinMagnitude={} velocityMaxMagnitude={} velocityL2={} velocityXMin={} velocityXMax={} velocityYMin={} velocityYMax={} velocityZMin={} velocityZMax={} pressureMin={} pressureMax={} pressureL2={}",
        format_scientific(report.fields.velocity.min_magnitude),
        format_scientific(report.fields.velocity.max_magnitude),
        format_scientific(report.fields.velocity.l2_norm),
        format_scientific(report.fields.velocity.x_min),
        format_scientific(report.fields.velocity.x_max),
        format_scientific(report.fields.velocity.y_min),
        format_scientific(report.fields.velocity.y_max),
        format_scientific(report.fields.velocity.z_min),
        format_scientific(report.fields.velocity.z_max),
        format_scientific(report.fields.pressure.min),
        format_scientific(report.fields.pressure.max),
        format_scientific(report.fields.pressure.l2_norm)
    );
    let mut residual_csv_path = solve.solve_residual_csv.clone();
    if let Some(path) = &solve.solve_residual_csv {
        write_laminar_simple_residual_csv(&report, path).map_err(|error| {
            format!(
                "could not write laminar SIMPLE residual history CSV to {} ({error})",
                path.display()
            )
        })?;
        println!("wrote laminar SIMPLE residual CSV: {}", path.display());
    }
    if let Some(plot_path) = &solve.solve_residual_plot {
        if residual_csv_path.is_none() {
            residual_csv_path = Some(plot_path.with_extension("csv"));
        }
        if let Some(csv_path) = &residual_csv_path {
            if solve.solve_residual_csv.is_none() {
                write_laminar_simple_residual_csv(&report, csv_path).map_err(|error| {
                    format!(
                        "could not write temporary residual history CSV to {} ({error})",
                        csv_path.display()
                    )
                })?;
                println!(
                    "wrote temporary residual CSV for plotting: {}",
                    csv_path.display()
                );
            }
            match write_laminar_simple_residual_plot(csv_path, plot_path) {
                Ok(()) => {
                    println!(
                        "wrote laminar SIMPLE residual plot: {}",
                        plot_path.display()
                    )
                }
                Err(error) => {
                    println!(
                        "laminarSimple residual plot warning: {} (CSV: {})",
                        error,
                        csv_path.display()
                    )
                }
            }
        }
    }
    println!(
        "laminarSimple operators: phiMin={} phiMax={} phiSumAbs={} gradPL2={} hbyAL2={} divPhiUL2={} velocityFixedValueFaces={} velocityZeroGradientFaces={} velocityInletOutletFaces={} pressureFixedValueFaces={} pressureZeroGradientFaces={}",
        format_scientific(report.operator_summary.phi_min),
        format_scientific(report.operator_summary.phi_max),
        format_scientific(report.operator_summary.phi_sum_abs),
        format_scientific(report.operator_summary.grad_p_l2_norm),
        format_scientific(report.operator_summary.hby_a_l2_norm),
        format_scientific(report.operator_summary.div_phi_u_l2_norm),
        report.boundary_summary.velocity_fixed_value_faces,
        report.boundary_summary.velocity_zero_gradient_faces,
        report.boundary_summary.velocity_inlet_outlet_faces,
        report.boundary_summary.pressure_fixed_value_faces,
        report.boundary_summary.pressure_zero_gradient_faces
    );
    if let Some(output_dir) = &solve.write_final_fields {
        write_laminar_simple_fields(&fields, &report, output_dir).map_err(|error| {
            format!(
                "could not write laminar SIMPLE fields to {} ({error})",
                output_dir.display()
            )
        })?;
        println!(
            "wrote laminar SIMPLE final fields: {}",
            output_dir.display()
        );
    } else {
        println!("laminarSimple status: no field files written");
    }

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
    print_laminar_simple_convergence_feedback(&report, &options);

    Ok(())
}

fn write_laminar_simple_residual_csv(
    report: &LaminarSimpleReport,
    path: &Path,
) -> std::io::Result<()> {
    ensure_parent_dir(path)?;
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    writeln!(
        writer,
        "iteration,continuityBeforeL2,continuityAfterL2,momentumInitialResidualNormalized,momentumFinalResidualNormalized,momentumResidualNorm,pressureInitialResidualNormalized,pressureFinalResidualNormalized,pressureResidualNorm,residualControlConfigured,residualControlChecked,residualControlSatisfied,pressureCorrectionAccepted,momentumLinearIterations,momentumLinearConverged,pressureLinearIterations,pressureLinearConverged,relativeVelocityChangeL2,relativePressureChangeL2"
    )?;
    for item in &report.history {
        writeln!(
            writer,
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            item.iteration,
            item.continuity_before.l2_norm,
            item.continuity_after.l2_norm,
            item.momentum_initial_normalized_residual_norm,
            item.momentum_normalized_residual_norm,
            item.momentum_residual_norm,
            item.pressure_correction_initial_normalized_residual_norm,
            item.pressure_correction_normalized_residual_norm,
            item.pressure_correction_residual_norm,
            item.residual_control.configured,
            item.residual_control.checked,
            item.residual_control.satisfied,
            item.pressure_correction_accepted,
            item.momentum_linear_iterations,
            item.momentum_linear_converged,
            item.pressure_linear_iterations,
            item.pressure_linear_converged,
            item.relative_velocity_change_l2,
            item.relative_pressure_change_l2
        )?;
    }

    writer.flush()
}

struct SimpleIterationEstimate {
    primary_metric: &'static str,
    additional_iterations: usize,
    geometric_ratio: f64,
}

fn estimate_simple_iterations_to_convergence(
    history: &[LaminarSimpleIterationSummary],
    options: &LaminarSimpleOptions,
) -> Option<SimpleIterationEstimate> {
    let mut candidates = Vec::new();
    if let Some(target) = options.momentum_residual_control {
        let momentum_values: Vec<f64> = history
            .iter()
            .map(|item| item.momentum_initial_normalized_residual_norm)
            .collect();
        if let Some(estimate) = estimate_iterations_to_convergence(&momentum_values, target)
            && estimate.additional_iterations > 0
        {
            candidates.push((
                "momentum residual",
                estimate.additional_iterations,
                estimate.geometric_ratio,
            ));
        }
    }

    if let Some(target) = options.pressure_residual_control {
        let pressure_values: Vec<f64> = history
            .iter()
            .map(|item| item.pressure_correction_initial_normalized_residual_norm)
            .collect();
        if let Some(estimate) = estimate_iterations_to_convergence(&pressure_values, target)
            && estimate.additional_iterations > 0
        {
            candidates.push((
                "pressure residual",
                estimate.additional_iterations,
                estimate.geometric_ratio,
            ));
        }
    }

    candidates
        .into_iter()
        .max_by_key(|candidate| candidate.1)
        .map(|(metric, iterations, ratio)| SimpleIterationEstimate {
            primary_metric: metric,
            additional_iterations: iterations,
            geometric_ratio: ratio,
        })
}

struct ConvergenceRatioEstimate {
    additional_iterations: usize,
    geometric_ratio: f64,
}

fn estimate_iterations_to_convergence(
    values: &[f64],
    target: f64,
) -> Option<ConvergenceRatioEstimate> {
    if !target.is_finite() || target <= 0.0 || values.len() < 3 {
        return None;
    }

    let values: Vec<f64> = values
        .iter()
        .copied()
        .filter(|value| value.is_finite() && *value > 0.0)
        .collect();
    if values.len() < 3 {
        return None;
    }

    let tail = if values.len() > 10 {
        &values[values.len() - 10..]
    } else {
        values.as_slice()
    };
    let mut ratios = Vec::new();
    for pair in tail.windows(2) {
        let previous = pair[0];
        let current = pair[1];
        if previous <= 0.0 || current <= 0.0 || !current.is_finite() || !previous.is_finite() {
            continue;
        }
        let ratio = current / previous;
        if ratio.is_finite() && ratio > 0.0 && ratio < 1.0 {
            ratios.push(ratio);
        }
    }
    if ratios.is_empty() {
        return None;
    }

    let geometric_ratio = ratios
        .iter()
        .fold(1.0, |acc, ratio| acc * ratio)
        .powf(1.0 / ratios.len() as f64);
    if geometric_ratio <= 0.0 || geometric_ratio >= 1.0 {
        return None;
    }

    let latest = *tail.last()?;
    if latest <= target {
        return Some(ConvergenceRatioEstimate {
            additional_iterations: 0,
            geometric_ratio,
        });
    }

    let estimate = (target / latest).ln() / geometric_ratio.ln();
    if !estimate.is_finite() || estimate <= 0.0 {
        return None;
    }

    Some(ConvergenceRatioEstimate {
        additional_iterations: estimate.ceil().max(1.0) as usize,
        geometric_ratio,
    })
}

fn print_laminar_simple_convergence_feedback(
    report: &LaminarSimpleReport,
    options: &LaminarSimpleOptions,
) {
    if report.converged {
        return;
    }

    match report.stop_reason {
        LaminarSimpleStopReason::MaxIterationsReached => {
            let reached_budget = options.max_simple_iterations == report.simple_iterations;
            let budget_message = if reached_budget {
                format!(
                    "iteration budget reached ({})",
                    options.max_simple_iterations
                )
            } else {
                "iteration budget stopped".to_string()
            };
            println!("laminarSimple convergence note: {budget_message}.");

            if report.history.len() >= 2 {
                let previous = &report.history[report.history.len() - 2];
                let latest = &report.history[report.history.len() - 1];
                let momentum_ratio = residual_ratio(
                    previous.momentum_initial_normalized_residual_norm,
                    latest.momentum_initial_normalized_residual_norm,
                );
                let pressure_ratio = residual_ratio(
                    previous.pressure_correction_initial_normalized_residual_norm,
                    latest.pressure_correction_initial_normalized_residual_norm,
                );
                let continuity_ratio = residual_ratio(
                    previous.continuity_after.l2_norm,
                    latest.continuity_after.l2_norm,
                );
                println!(
                    "  last-iteration trend (iter {} -> {}): momentum {} | pressure {} | continuity {}",
                    previous.iteration,
                    latest.iteration,
                    format_ratio(momentum_ratio),
                    format_ratio(pressure_ratio),
                    format_ratio(continuity_ratio)
                );
            } else {
                println!("  last-iteration trend: not enough samples");
            }

            println!(
                "  suggestion: if the initial residuals are still decreasing, increase the controlDict endTime or --maxSimpleIterations budget."
            );

            if let Some(estimate) =
                estimate_simple_iterations_to_convergence(&report.history, options)
            {
                println!(
                    "  trend estimate: if current geometric decay persists, add about {} SIMPLE iteration(s) for convergence (targeted on {} criterion; last ratio {:.4}).",
                    estimate.additional_iterations,
                    estimate.primary_metric,
                    estimate.geometric_ratio
                );
            }
        }
        LaminarSimpleStopReason::ConvergenceCriteriaNotConfigured => {
            if report.simple_iterations > 0
                && let (Some(momentum), Some(pressure)) = (
                    report
                        .history
                        .last()
                        .map(|item| item.momentum_initial_normalized_residual_norm),
                    report
                        .history
                        .last()
                        .map(|item| item.pressure_correction_initial_normalized_residual_norm),
                )
            {
                println!(
                    "  final SIMPLE-iteration initial residual U={}",
                    format_scientific(momentum)
                );
                println!(
                    "  final SIMPLE-iteration initial residual p={}",
                    format_scientific(pressure)
                );
            }
            println!(
                "laminarSimple convergence note: no active convergence criteria (no residualControl in fvSolution)."
            );
            println!(
                "  to stop early, set SIMPLE.residualControl U/p in system/fvSolution. Benchmark acceptance is evaluated externally."
            );
            if options.max_simple_iterations > 1 {
                println!(
                    "  run note: you requested --maxSimpleIterations {}, if this case was still improving, increase it and keep a convergence check enabled.",
                    options.max_simple_iterations
                );
            }
            if let Some(estimate) =
                estimate_simple_iterations_to_convergence(&report.history, options)
            {
                println!(
                    "  trend estimate: if current geometric decay persists, add about {} SIMPLE iteration(s) for the {} criterion (last ratio {:.4}).",
                    estimate.additional_iterations,
                    estimate.primary_metric,
                    estimate.geometric_ratio
                );
            }
        }
        LaminarSimpleStopReason::MomentumSolverInvalidState => {
            println!(
                "laminarSimple convergence note: momentum equation linear solve entered invalid state."
            );
        }
        LaminarSimpleStopReason::PressureSolverInvalidState => {
            println!(
                "laminarSimple convergence note: pressure equation linear solve entered invalid state."
            );
        }
        LaminarSimpleStopReason::SolverInvalidState => {
            println!(
                "laminarSimple convergence note: solver encountered a non-finite field/state."
            );
        }
        LaminarSimpleStopReason::Converged => {}
    }
}

fn residual_ratio(previous: f64, latest: f64) -> Option<f64> {
    if !previous.is_finite() || !latest.is_finite() || previous.abs() <= f64::EPSILON {
        return None;
    }
    Some(latest / previous)
}

fn format_ratio(value: Option<f64>) -> String {
    value
        .map(format_scientific)
        .unwrap_or_else(|| "n/a".to_string())
}

fn write_laminar_simple_residual_plot(csv_path: &Path, plot_path: &Path) -> std::io::Result<()> {
    let wants_svg = plot_path
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("svg"));
    if wants_svg {
        return write_laminar_simple_residual_plot_svg(csv_path, plot_path);
    }

    let python = locate_python_interpreter().ok_or_else(|| {
        Error::new(
            ErrorKind::NotFound,
            "python is required to generate the residual plot, but could not be found",
        )
    })?;
    ensure_parent_dir(plot_path)?;

    const SCRIPT: &str = r#"
import csv
import sys

import matplotlib.pyplot as plt

from pathlib import Path

csv_path = Path(sys.argv[1])
plot_path = Path(sys.argv[2])

with csv_path.open("r", newline="") as handle:
    rows = list(csv.DictReader(handle))

if not rows:
    raise RuntimeError("no residual history rows in CSV")

iterations = [float(row["iteration"]) for row in rows]
momentum = [float(row["momentumInitialResidualNormalized"]) for row in rows]
pressure = [float(row["pressureInitialResidualNormalized"]) for row in rows]
continuity = [float(row["continuityAfterL2"]) for row in rows]

plt.figure(figsize=(8, 5))
plt.plot(iterations, continuity, label="Continuity L2")
plt.plot(iterations, momentum, label="U initial residual")
plt.plot(iterations, pressure, label="p initial residual")
plt.yscale("log")
plt.xlabel("SIMPLE iteration")
plt.ylabel("Residual / Continuity metric")
plt.legend()
plt.grid(True, which="both", alpha=0.25)
plt.tight_layout()
plt.savefig(plot_path, dpi=150)
"#;

    let status = Command::new(python)
        .arg("-c")
        .arg(SCRIPT)
        .arg(csv_path)
        .arg(plot_path)
        .status()?;

    if !status.success() {
        return Err(Error::other(format!(
            "python plotting failed with status {status}"
        )));
    }

    Ok(())
}

fn write_laminar_simple_residual_plot_svg(
    csv_path: &Path,
    plot_path: &Path,
) -> std::io::Result<()> {
    #[derive(Default)]
    struct ParsedCsvRow {
        iteration: f64,
        continuity: f64,
        momentum: f64,
        pressure: f64,
    }

    let raw = std::fs::read_to_string(csv_path)?;
    let mut rows = Vec::new();
    for (index, line) in raw.lines().enumerate() {
        if index == 0 || line.trim().is_empty() {
            continue;
        }
        let values: Vec<&str> = line.split(',').collect();
        if values.len() < 9 {
            continue;
        }

        let parse = |idx: usize| -> f64 {
            values
                .get(idx)
                .and_then(|value| value.parse::<f64>().ok())
                .unwrap_or(f64::NAN)
        };

        let row = ParsedCsvRow {
            iteration: parse(0),
            continuity: parse(2),
            momentum: parse(3),
            pressure: parse(6),
        };
        if row.iteration.is_finite()
            && row.continuity.is_finite()
            && row.momentum.is_finite()
            && row.pressure.is_finite()
        {
            rows.push(row);
        }
    }

    if rows.is_empty() {
        return Err(Error::other("no residual history rows in CSV"));
    }

    let width = 800.0f64;
    let height = 500.0f64;
    let left = 70.0f64;
    let right = 20.0f64;
    let top = 20.0f64;
    let bottom = 60.0f64;
    let plot_width = width - left - right;
    let plot_height = height - top - bottom;
    let plot_top = top;
    let plot_bottom = top + plot_height;

    let y_values: Vec<f64> = rows
        .iter()
        .flat_map(|row| [row.continuity, row.momentum, row.pressure])
        .filter(|value| value.is_finite() && *value > 0.0)
        .collect();
    let y_min = y_values
        .iter()
        .map(|value| value.log10())
        .fold(f64::INFINITY, |acc, value| acc.min(value));
    let y_max = y_values
        .iter()
        .map(|value| value.log10())
        .fold(f64::NEG_INFINITY, |acc, value| acc.max(value));
    let y_display_min = y_min.floor() - 1.0;
    let y_display_max = y_max.ceil() + 1.0;
    let y_span = (y_display_max - y_display_min).max(1.0);

    let min_iteration = rows.first().map(|row| row.iteration).unwrap_or(0.0);
    let max_iteration = rows.last().map(|row| row.iteration).unwrap_or(1.0);
    let iteration_span = (max_iteration - min_iteration).max(1.0);

    fn map_x(iteration: f64, min_iteration: f64, span: f64, left: f64, width: f64) -> f64 {
        left + ((iteration - min_iteration) / span) * width
    }

    fn map_y(value: f64, min_log: f64, span: f64, top: f64, bottom: f64) -> f64 {
        let y = value.log10();
        let clamped = y.clamp(min_log, min_log + span);
        bottom - (clamped - min_log) / span * (bottom - top)
    }

    let polyline_points = |series: &[ParsedCsvRow], selector: fn(&ParsedCsvRow) -> f64| {
        let mut points = String::new();
        for row in series {
            let y = map_y(selector(row), y_display_min, y_span, plot_top, plot_bottom);
            let x = map_x(
                row.iteration,
                min_iteration,
                iteration_span,
                left,
                plot_width,
            );
            points.push_str(&format!("{x:.3},{y:.3} "));
        }
        points.trim_end().to_string()
    };

    let momentum_points = polyline_points(&rows, |row| row.momentum);
    let pressure_points = polyline_points(&rows, |row| row.pressure);
    let continuity_points = polyline_points(&rows, |row| row.continuity);

    let y_tick_start = y_display_min as i32;
    let y_tick_end = y_display_max as i32;
    let y_ticks = (y_tick_start..=y_tick_end)
        .map(|tick| {
            let y = plot_bottom - ((tick as f64 - y_display_min) / y_span) * (plot_bottom - plot_top);
            format!("<line x1=\"{:.3}\" y1=\"{:.3}\" x2=\"{:.3}\" y2=\"{:.3}\" stroke=\"#ddd\" stroke-width=\"1\"/>\n           <text x=\"{:.3}\" y=\"{:.3}\" font-size=\"10\" fill=\"#444\">1e{}</text>",
                left, y, left + plot_width, y, left - 12.0, y + 4.0, tick)
        })
        .collect::<Vec<_>>()
        .join("\n");

    let x_ticks = 5usize;
    let x_tick_lines = (0..=x_ticks)
        .map(|idx| {
            let fraction = idx as f64 / x_ticks as f64;
            let iter = min_iteration + fraction * iteration_span;
            let x = left + fraction * plot_width;
            let label = format!("{:.0}", iter.round());
            format!(
                "<line x1=\"{x:.3}\" y1=\"{top:.3}\" x2=\"{x:.3}\" y2=\"{:.3}\" stroke=\"#ddd\" stroke-width=\"1\"/>\n           <text x=\"{:.3}\" y=\"{:.3}\" font-size=\"10\" fill=\"#444\" text-anchor=\"middle\">{}</text>",
                plot_bottom + 3.0,
                x - 2.0,
                plot_bottom + 15.0,
                label
            )
        })
        .collect::<Vec<_>>()
        .join("\n");

    let mut svg = String::new();
    svg.push_str(&format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {width} {height}\" width=\"{width}\" height=\"{height}\">\n",
        width = width,
        height = height
    ));
    svg.push_str(&format!(
        "  <rect x=\"0\" y=\"0\" width=\"{width}\" height=\"{height}\" fill=\"#fff\"/>\n",
        width = width,
        height = height
    ));
    svg.push_str(&format!(
        "  <rect x=\"{left}\" y=\"{top}\" width=\"{plot_width}\" height=\"{plot_height}\" fill=\"none\" stroke=\"#888\" stroke-width=\"1\"/>\n",
        left = left,
        top = top,
        plot_width = plot_width,
        plot_height = plot_height
    ));
    svg.push_str("  <g stroke=\"none\" fill=\"#666\" font-family=\"Arial,sans-serif\">\n");
    svg.push_str(&format!("    {y_ticks}\n"));
    svg.push_str(&format!("    {x_tick_lines}\n"));
    svg.push_str(&format!(
        "    <text x=\"{:.1}\" y=\"{:.1}\" font-size=\"12\" text-anchor=\"middle\">Iteration</text>\n",
        left + plot_width / 2.0,
        plot_bottom + 40.0
    ));
    svg.push_str(&format!(
        "    <text transform=\"translate(16,{:.1}) rotate(-90)\" font-size=\"12\" text-anchor=\"middle\">Residual (log10 scale)</text>\n",
        (plot_top + plot_bottom) / 2.0
    ));
    svg.push_str("  </g>\n");
    svg.push_str("  <g fill=\"none\" stroke-width=\"2\">\n");
    svg.push_str(&format!(
        "    <polyline points=\"{momentum_points}\" stroke=\"#1f77b4\" />\n"
    ));
    svg.push_str(&format!(
        "    <polyline points=\"{pressure_points}\" stroke=\"#ff7f0e\" />\n"
    ));
    svg.push_str(&format!(
        "    <polyline points=\"{continuity_points}\" stroke=\"#2ca02c\" />\n"
    ));
    svg.push_str("  </g>\n");
    svg.push_str("  <g font-family=\"Arial,sans-serif\" font-size=\"11\" fill=\"#333\">\n");
    svg.push_str(&format!(
        "    <text x=\"{:.1}\" y=\"{:.1}\">U initial residual</text>\n",
        left + plot_width - 10.0,
        plot_top + 20.0
    ));
    svg.push_str(&format!(
        "    <text x=\"{:.1}\" y=\"{:.1}\">p initial residual</text>\n",
        left + plot_width - 10.0,
        plot_top + 38.0
    ));
    svg.push_str(&format!(
        "    <text x=\"{:.1}\" y=\"{:.1}\">Continuity L2</text>\n",
        left + plot_width - 10.0,
        plot_top + 56.0
    ));
    svg.push_str(&format!(
        "    <text x=\"{:.1}\" y=\"{:.1}\" fill=\"#1f77b4\">Momentum</text>\n",
        left + plot_width + 10.0,
        plot_top + 20.0
    ));
    svg.push_str(&format!(
        "    <text x=\"{:.1}\" y=\"{:.1}\" fill=\"#ff7f0e\">Pressure</text>\n",
        left + plot_width + 10.0,
        plot_top + 38.0
    ));
    svg.push_str(&format!(
        "    <text x=\"{:.1}\" y=\"{:.1}\" fill=\"#2ca02c\">Continuity</text>\n",
        left + plot_width + 10.0,
        plot_top + 56.0
    ));
    svg.push_str("  </g>\n");
    svg.push_str("</svg>\n");

    ensure_parent_dir(plot_path)?;
    std::fs::write(plot_path, svg)?;
    Ok(())
}

fn locate_python_interpreter() -> Option<&'static str> {
    ["python", "python3"].into_iter().find(|command| {
        Command::new(command)
            .arg("--version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_ok_and(|status| status.success())
    })
}

fn resolve_laminar_simple_options(
    plan: &SolverCasePlan,
    solve: &LaminarSimpleSolveArgs,
) -> Result<LaminarSimpleOptions, String> {
    if !plan.numerics.fv_solution.present {
        return Err("Laminar SIMPLE solve requires OpenFOAM-style system/fvSolution".to_string());
    }
    let density = solve
        .density
        .or_else(|| property_number(plan, "transportProperties", None, "rho"))
        .ok_or_else(|| {
            "Laminar SIMPLE solve requires --rho or transportProperties.rho".to_string()
        })?;
    let dynamic_viscosity = solve
        .dynamic_viscosity
        .or_else(|| property_number(plan, "transportProperties", None, "mu"))
        .ok_or_else(|| {
            "Laminar SIMPLE solve requires --mu or transportProperties.mu".to_string()
        })?;
    let momentum_case_tolerance = fv_solution_number(plan, "solvers.U", "tolerance")?;
    let pressure_case_tolerance = fv_solution_number(plan, "solvers.p", "tolerance")?;
    let momentum_case_max_iterations = fv_solution_usize(plan, "solvers.U", "maxIter")?;
    let pressure_case_max_iterations = fv_solution_usize(plan, "solvers.p", "maxIter")?;

    let linear_tolerance = solve
        .linear_tolerance
        .or(momentum_case_tolerance)
        .unwrap_or(OPENFOAM_DEFAULT_LDU_TOLERANCE);
    let max_linear_iterations = solve
        .max_linear_iterations
        .unwrap_or(OPENFOAM_DEFAULT_LDU_MAX_ITERATIONS);
    let momentum_linear_tolerance = solve
        .momentum_linear_tolerance
        .or(solve.linear_tolerance)
        .or(momentum_case_tolerance)
        .unwrap_or(OPENFOAM_DEFAULT_LDU_TOLERANCE);
    let pressure_linear_tolerance = solve
        .pressure_linear_tolerance
        .or(solve.linear_tolerance)
        .or(pressure_case_tolerance)
        .unwrap_or(OPENFOAM_DEFAULT_LDU_TOLERANCE);
    let momentum_max_linear_iterations = solve
        .momentum_max_linear_iterations
        .or(solve.max_linear_iterations)
        .or(momentum_case_max_iterations)
        .unwrap_or(OPENFOAM_DEFAULT_LDU_MAX_ITERATIONS);
    let pressure_max_linear_iterations = solve
        .pressure_max_linear_iterations
        .or(solve.max_linear_iterations)
        .or(pressure_case_max_iterations)
        .unwrap_or(OPENFOAM_DEFAULT_LDU_MAX_ITERATIONS);
    let momentum_linear_solver = match solve.momentum_linear_solver.or(solve.linear_solver) {
        Some(solver) => solver,
        None => required_fv_solution_laminar_solver(plan, "solvers.U")?,
    };
    let pressure_linear_solver = match solve.pressure_linear_solver.or(solve.linear_solver) {
        Some(solver) => solver,
        None => required_fv_solution_laminar_solver(plan, "solvers.p")?,
    };
    validate_openfoam_linear_controls(plan, "solvers.U", momentum_linear_solver)?;
    validate_openfoam_linear_controls(plan, "solvers.p", pressure_linear_solver)?;
    let linear_solver = solve.linear_solver.unwrap_or(momentum_linear_solver);
    let momentum_preconditioner = resolve_laminar_preconditioner(
        plan,
        "solvers.U",
        momentum_linear_solver,
        solve.momentum_preconditioner,
    )?;
    let pressure_preconditioner = resolve_laminar_preconditioner(
        plan,
        "solvers.p",
        pressure_linear_solver,
        solve.pressure_preconditioner,
    )?;
    let max_simple_iterations = solve
        .max_simple_iterations
        .or(plan.run.estimated_steps)
        .filter(|iterations| *iterations > 0)
        .ok_or_else(|| {
            "Laminar SIMPLE requires --maxSimpleIterations or a positive controlDict endTime/deltaT iteration count"
                .to_string()
        })?;
    let min_simple_iterations = solve
        .min_simple_iterations
        .or(fv_solution_usize(plan, "SIMPLE", "minSimpleIterations")?)
        .unwrap_or(if max_simple_iterations > 1 { 2 } else { 1 });
    validate_laminar_residual_control_dictionary(&plan.numerics.fv_solution)?;
    let momentum_residual_control = fv_solution_single_scalar(plan, "SIMPLE.residualControl", "U")?;
    let pressure_residual_control = fv_solution_single_scalar(plan, "SIMPLE.residualControl", "p")?;
    let pressure_reference_cell = solve
        .pressure_reference_cell
        .or(fv_solution_usize(plan, "SIMPLE", "pRefCell")?);
    let pressure_reference_value = solve
        .pressure_reference_value
        .or(fv_solution_number(plan, "SIMPLE", "pRefValue")?)
        .unwrap_or(0.0);
    let non_orthogonal_correctors = solve
        .non_orthogonal_correctors
        .or(fv_solution_usize(
            plan,
            "SIMPLE",
            "nNonOrthogonalCorrectors",
        )?)
        .unwrap_or(0);
    let simple_consistent = solve
        .simple_consistent
        .or(fv_solution_bool(plan, "SIMPLE", "consistent")?)
        .unwrap_or(false);
    let schemes = resolve_laminar_simple_schemes(plan)?;

    Ok(LaminarSimpleOptions {
        density,
        dynamic_viscosity,
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
        max_simple_iterations,
        min_simple_iterations,
        momentum_residual_control,
        pressure_residual_control,
        pressure_reference_cell,
        pressure_reference_value,
        non_orthogonal_correctors,
        simple_consistent,
        velocity_relaxation: solve
            .velocity_relaxation
            .or(fv_solution_number(
                plan,
                "relaxationFactors.equations",
                "U",
            )?)
            .unwrap_or(1.0),
        pressure_relaxation: solve
            .pressure_relaxation
            .or(fv_solution_number(plan, "relaxationFactors.fields", "p")?)
            .unwrap_or(1.0),
        schemes,
    })
}

fn resolve_laminar_simple_schemes(plan: &SolverCasePlan) -> Result<LaminarSimpleSchemes, String> {
    if !plan.numerics.fv_schemes.present {
        return Err("Laminar SIMPLE solve requires OpenFOAM-style system/fvSchemes".to_string());
    }

    let sn_grad = parse_laminar_simple_sn_grad_scheme(required_fv_scheme(
        plan,
        "snGradSchemes",
        "default",
        None,
    )?)?;
    let laplacian = fv_schemes_value(plan, "laplacianSchemes", "laplacian(nu,U)")
        .or_else(|| fv_schemes_value(plan, "laplacianSchemes", "laplacian(nuEff,U)"))
        .or_else(|| fv_schemes_value(plan, "laplacianSchemes", "default"))
        .ok_or_else(|| {
            "fvSchemes laplacianSchemes requires laplacian(nu,U), laplacian(nuEff,U), or default"
                .to_string()
        })
        .and_then(|value| parse_laminar_simple_laplacian_scheme(value, sn_grad))?;

    Ok(LaminarSimpleSchemes {
        grad_p: parse_laminar_simple_gradient_scheme(required_fv_scheme(
            plan,
            "gradSchemes",
            "grad(p)",
            Some("default"),
        )?)?,
        grad_u: parse_laminar_simple_gradient_scheme(required_fv_scheme(
            plan,
            "gradSchemes",
            "grad(U)",
            Some("default"),
        )?)?,
        div_phi_u: parse_laminar_simple_convection_scheme(required_fv_scheme(
            plan,
            "divSchemes",
            "div(phi,U)",
            Some("default"),
        )?)?,
        laplacian,
        interpolation: parse_laminar_simple_interpolation_scheme(required_fv_scheme(
            plan,
            "interpolationSchemes",
            "default",
            None,
        )?)?,
        sn_grad,
    })
}

fn fv_schemes_value<'a>(plan: &'a SolverCasePlan, section: &str, key: &str) -> Option<&'a str> {
    numerics_dictionary_value(&plan.numerics.fv_schemes, section, key)
}

fn required_fv_scheme<'a>(
    plan: &'a SolverCasePlan,
    section: &str,
    key: &str,
    fallback_key: Option<&str>,
) -> Result<&'a str, String> {
    fv_schemes_value(plan, section, key)
        .or_else(|| fallback_key.and_then(|fallback| fv_schemes_value(plan, section, fallback)))
        .ok_or_else(|| match fallback_key {
            Some(fallback) => {
                format!("fvSchemes {section} requires {key} or {fallback}")
            }
            None => format!("fvSchemes {section} requires {key}"),
        })
}

fn parse_laminar_simple_gradient_scheme(
    value: &str,
) -> Result<LaminarSimpleGradientScheme, String> {
    let tokens = normalized_scheme_tokens(value);
    if scheme_tokens_are(&tokens, &["gauss", "linear"]) {
        Ok(LaminarSimpleGradientScheme::GaussLinear)
    } else {
        Err(format!(
            "unsupported laminar SIMPLE grad scheme '{value}'; currently supported: Gauss linear"
        ))
    }
}

fn parse_laminar_simple_convection_scheme(
    value: &str,
) -> Result<LaminarSimpleConvectionScheme, String> {
    let tokens = normalized_scheme_tokens(value);
    let tokens = strip_bounded_scheme_prefix(&tokens);
    if scheme_tokens_are(tokens, &["gauss", "upwind"]) {
        Ok(LaminarSimpleConvectionScheme::GaussUpwind)
    } else if scheme_token_is(tokens, 0, "gauss") && scheme_token_is(tokens, 1, "linearupwind") {
        Ok(LaminarSimpleConvectionScheme::GaussLinearUpwind)
    } else if scheme_tokens_are(tokens, &["none"]) {
        Err(
            "laminar SIMPLE requires divSchemes.div(phi,U); divSchemes default none is not executable"
                .to_string(),
        )
    } else {
        Err(format!(
            "unsupported laminar SIMPLE div(phi,U) scheme '{value}'; currently supported: Gauss upwind or Gauss linearUpwind grad(U)"
        ))
    }
}

fn parse_laminar_simple_interpolation_scheme(
    value: &str,
) -> Result<LaminarSimpleInterpolationScheme, String> {
    let tokens = normalized_scheme_tokens(value);
    if scheme_tokens_are(&tokens, &["linear"]) {
        Ok(LaminarSimpleInterpolationScheme::Linear)
    } else {
        Err(format!(
            "unsupported laminar SIMPLE interpolation scheme '{value}'; currently supported: linear"
        ))
    }
}

fn parse_laminar_simple_sn_grad_scheme(value: &str) -> Result<LaminarSimpleSnGradScheme, String> {
    let tokens = normalized_scheme_tokens(value);
    if scheme_tokens_are(&tokens, &["corrected"]) {
        Ok(LaminarSimpleSnGradScheme::Corrected)
    } else if scheme_tokens_are(&tokens, &["orthogonal"]) {
        Ok(LaminarSimpleSnGradScheme::Orthogonal)
    } else if scheme_tokens_are(&tokens, &["uncorrected"]) {
        Ok(LaminarSimpleSnGradScheme::Uncorrected)
    } else {
        Err(format!(
            "unsupported laminar SIMPLE snGrad scheme '{value}'; currently supported: corrected, orthogonal, uncorrected"
        ))
    }
}

fn parse_laminar_simple_laplacian_scheme(
    value: &str,
    sn_grad: LaminarSimpleSnGradScheme,
) -> Result<LaminarSimpleLaplacianScheme, String> {
    let tokens = normalized_scheme_tokens(value);
    let tokens = strip_bounded_scheme_prefix(&tokens);
    if !scheme_token_is(tokens, 0, "gauss")
        || !scheme_token_is(tokens, 1, "linear")
        || tokens.len() > 3
    {
        return Err(format!(
            "unsupported laminar SIMPLE laplacian scheme '{value}'; currently supported: Gauss linear corrected/orthogonal/uncorrected"
        ));
    }

    match tokens.get(2).map(String::as_str) {
        Some("corrected") => Ok(LaminarSimpleLaplacianScheme::GaussLinearCorrected),
        Some("orthogonal") => Ok(LaminarSimpleLaplacianScheme::GaussLinearOrthogonal),
        Some("uncorrected") => Ok(LaminarSimpleLaplacianScheme::GaussLinearUncorrected),
        Some(other) => Err(format!(
            "unsupported laminar SIMPLE laplacian correction '{other}' in scheme '{value}'"
        )),
        None => Ok(laplacian_from_sn_grad(sn_grad)),
    }
}

fn laplacian_from_sn_grad(sn_grad: LaminarSimpleSnGradScheme) -> LaminarSimpleLaplacianScheme {
    match sn_grad {
        LaminarSimpleSnGradScheme::Corrected => LaminarSimpleLaplacianScheme::GaussLinearCorrected,
        LaminarSimpleSnGradScheme::Orthogonal => {
            LaminarSimpleLaplacianScheme::GaussLinearOrthogonal
        }
        LaminarSimpleSnGradScheme::Uncorrected => {
            LaminarSimpleLaplacianScheme::GaussLinearUncorrected
        }
    }
}

fn normalized_scheme_tokens(value: &str) -> Vec<String> {
    value
        .split_whitespace()
        .map(|token| token.trim_matches(';').to_ascii_lowercase())
        .filter(|token| !token.is_empty())
        .collect()
}

fn strip_bounded_scheme_prefix(tokens: &[String]) -> &[String] {
    if tokens.first().is_some_and(|token| token == "bounded") {
        &tokens[1..]
    } else {
        tokens
    }
}

fn scheme_tokens_are(tokens: &[String], expected: &[&str]) -> bool {
    tokens.len() == expected.len()
        && tokens
            .iter()
            .zip(expected)
            .all(|(token, expected)| token == expected)
}

fn scheme_token_is(tokens: &[String], index: usize, expected: &str) -> bool {
    tokens.get(index).is_some_and(|token| token == expected)
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

fn fv_solution_number(
    plan: &SolverCasePlan,
    section: &str,
    key: &str,
) -> Result<Option<f64>, String> {
    let Some(value) = numerics_dictionary_value(&plan.numerics.fv_solution, section, key) else {
        return Ok(None);
    };
    last_number(value).map(Some).ok_or_else(|| {
        format!("fvSolution {section}.{key} must contain a numeric value, got '{value}'")
    })
}

fn fv_solution_single_scalar(
    plan: &SolverCasePlan,
    section: &str,
    key: &str,
) -> Result<Option<f64>, String> {
    let Some(value) = numerics_dictionary_value(&plan.numerics.fv_solution, section, key) else {
        return Ok(None);
    };
    let tokens = value
        .trim()
        .trim_end_matches(';')
        .split_whitespace()
        .collect::<Vec<_>>();
    if tokens.len() != 1 {
        return Err(format!(
            "fvSolution {section}.{key} must be one scalar value, got '{value}'"
        ));
    }
    tokens[0]
        .parse::<f64>()
        .map(Some)
        .map_err(|_| format!("fvSolution {section}.{key} must be one scalar value, got '{value}'"))
}

fn validate_laminar_residual_control_dictionary(
    dictionary: &SolverNumericsDictionaryPlan,
) -> Result<(), String> {
    const SECTION: &str = "SIMPLE.residualControl";
    if let Some(nested) = dictionary
        .sections
        .iter()
        .find(|section| section.path.starts_with(&format!("{SECTION}.")))
    {
        return Err(format!(
            "fvSolution {SECTION} entries must be single scalar values for steady SIMPLE convergence; nested dictionary '{}' is not valid OpenFOAM Foundation syntax here",
            nested.path
        ));
    }

    if let Some(entry) = dictionary
        .entries
        .iter()
        .find(|entry| entry.section == SECTION && entry.key != "U" && entry.key != "p")
    {
        return Err(format!(
            "fvSolution {SECTION}.{} is not supported by laminarSimple; supported solved fields are U and p",
            entry.key
        ));
    }

    Ok(())
}

fn fv_solution_bool(
    plan: &SolverCasePlan,
    section: &str,
    key: &str,
) -> Result<Option<bool>, String> {
    let Some(value) = numerics_dictionary_value(&plan.numerics.fv_solution, section, key) else {
        return Ok(None);
    };
    parse_bool_value(value).map(Some).ok_or_else(|| {
        format!("fvSolution {section}.{key} must be true/false or yes/no, got '{value}'")
    })
}

fn required_fv_solution_laminar_solver(
    plan: &SolverCasePlan,
    section: &str,
) -> Result<LaminarSimpleLinearSolver, String> {
    let value = numerics_dictionary_value(&plan.numerics.fv_solution, section, "solver")
        .ok_or_else(|| format!("fvSolution {section} requires a solver entry"))?;
    if value.trim().trim_end_matches(';') == "smoothSolver" {
        let smoother = numerics_dictionary_value(&plan.numerics.fv_solution, section, "smoother")
            .ok_or_else(|| {
            format!("fvSolution {section} smoothSolver requires a smoother entry")
        })?;
        return match smoother.trim().trim_end_matches(';') {
            "GaussSeidel" | "gaussSeidel" => Ok(LaminarSimpleLinearSolver::GaussSeidel),
            "symGaussSeidel" => Ok(LaminarSimpleLinearSolver::SymGaussSeidel),
            other => Err(format!(
                "unsupported fvSolution {section} smoother '{other}'; Ferrum currently supports GaussSeidel and symGaussSeidel"
            )),
        };
    }
    parse_openfoam_laminar_solver(value)
}

fn resolve_laminar_preconditioner(
    plan: &SolverCasePlan,
    section: &str,
    solver: LaminarSimpleLinearSolver,
    explicit: Option<LaminarSimplePreconditioner>,
) -> Result<LaminarSimplePreconditioner, String> {
    if let Some(preconditioner) = explicit {
        return Ok(preconditioner);
    }
    if !matches!(
        solver,
        LaminarSimpleLinearSolver::Pcg | LaminarSimpleLinearSolver::BiCgStab
    ) {
        return Ok(LaminarSimplePreconditioner::None);
    }

    let value = numerics_dictionary_value(&plan.numerics.fv_solution, section, "preconditioner")
        .ok_or_else(|| format!("fvSolution {section} requires a preconditioner entry"))?;
    parse_openfoam_laminar_preconditioner(value)
}

fn validate_openfoam_linear_controls(
    plan: &SolverCasePlan,
    section: &str,
    solver: LaminarSimpleLinearSolver,
) -> Result<(), String> {
    if let Some(relative_tolerance) = fv_solution_number(plan, section, "relTol")?
        && relative_tolerance != 0.0
    {
        return Err(format!(
            "fvSolution {section}.relTol={relative_tolerance} is not implemented yet; Ferrum refuses to ignore a non-zero OpenFOAM relative tolerance"
        ));
    }
    if let Some(min_iterations) = fv_solution_usize(plan, section, "minIter")?
        && min_iterations != 0
    {
        return Err(format!(
            "fvSolution {section}.minIter={min_iterations} is not implemented yet; Ferrum refuses to ignore it"
        ));
    }
    if matches!(
        solver,
        LaminarSimpleLinearSolver::GaussSeidel | LaminarSimpleLinearSolver::SymGaussSeidel
    ) && let Some(sweeps) = fv_solution_usize(plan, section, "nSweeps")?
        && sweeps != 1
    {
        return Err(format!(
            "fvSolution {section}.nSweeps={sweeps} is not implemented yet; Ferrum currently matches the OpenFOAM default nSweeps=1"
        ));
    }
    Ok(())
}

fn fv_solution_usize(
    plan: &SolverCasePlan,
    section: &str,
    key: &str,
) -> Result<Option<usize>, String> {
    let Some(value) = numerics_dictionary_value(&plan.numerics.fv_solution, section, key) else {
        return Ok(None);
    };
    value
        .trim()
        .trim_end_matches(';')
        .parse::<usize>()
        .ok()
        .map(Some)
        .ok_or_else(|| {
            format!("fvSolution {section}.{key} must contain a non-negative integer, got '{value}'")
        })
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

#[cfg(test)]
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

#[cfg(test)]
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

#[cfg(test)]
fn last_usize(value: &str) -> Option<usize> {
    value.split_whitespace().rev().find_map(|token| {
        token
            .trim_matches(|ch| ch == '[' || ch == ']')
            .parse::<usize>()
            .ok()
    })
}

fn parse_openfoam_laminar_solver(value: &str) -> Result<LaminarSimpleLinearSolver, String> {
    match value.trim().trim_end_matches(';') {
        "PBiCG" | "PBiCGStab" | "BiCGStab" | "bicgstab" => Ok(LaminarSimpleLinearSolver::BiCgStab),
        "GaussSeidel" | "gaussSeidel" | "gauss-seidel" => {
            Ok(LaminarSimpleLinearSolver::GaussSeidel)
        }
        "symGaussSeidel" | "sym-gauss-seidel" => Ok(LaminarSimpleLinearSolver::SymGaussSeidel),
        "smoothSolver" => Err(
            "OpenFOAM smoothSolver requires a smoother entry in fvSolution; use GaussSeidel or symGaussSeidel for a direct CLI override"
                .to_string(),
        ),
        "PCG" | "pcg" => Ok(LaminarSimpleLinearSolver::Pcg),
        "CG" | "cg" => Ok(LaminarSimpleLinearSolver::Cg),
        "Jacobi" | "jacobi" => Ok(LaminarSimpleLinearSolver::Jacobi),
        other => Err(format!(
            "unsupported OpenFOAM linear solver '{other}'; no fallback was applied"
        )),
    }
}

fn parse_openfoam_laminar_preconditioner(
    value: &str,
) -> Result<LaminarSimplePreconditioner, String> {
    match value.trim().trim_end_matches(';') {
        "none" | "None" => Ok(LaminarSimplePreconditioner::None),
        "DIC" | "FDIC" | "incompleteCholesky" | "ic0" | "IC0" => {
            Ok(LaminarSimplePreconditioner::IncompleteCholesky)
        }
        "diagonal" | "Diagonal" => Ok(LaminarSimplePreconditioner::Diagonal),
        "DILU" => Err(
            "OpenFOAM DILU is not implemented yet; refusing to substitute a diagonal preconditioner"
                .to_string(),
        ),
        other => Err(format!(
            "unsupported OpenFOAM preconditioner '{other}'; no fallback was applied"
        )),
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
    print_openfoam_case_compatibility_warnings(&plan.warnings);
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

fn print_openfoam_case_compatibility_warnings(warnings: &[String]) {
    let compatibility_warnings: Vec<String> = warnings
        .iter()
        .filter_map(|warning| {
            warning
                .strip_prefix("openFOAM compatibility: ")
                .map(std::string::ToString::to_string)
        })
        .collect();
    if compatibility_warnings.is_empty() {
        println!("openFOAM compatibility: case layout and required fields look present");
        return;
    }

    println!(
        "openFOAM compatibility: {} item(s) to check",
        compatibility_warnings.len()
    );
    for message in compatibility_warnings {
        println!("  {}", message);
    }
}

fn print_linear_solver_capabilities() {
    let capabilities = linear_solver_capabilities();
    println!(
        "linear solvers: cpuCsr={} cpuJacobi={} cpuGaussSeidel={} cpuSymGaussSeidel={} cpuCg={} cpuPcg={} cpuBiCgStab={} cpuDiagonalPreconditioner={} cpuIncompleteCholeskyPreconditioner={} gpuLinearSolvers={}",
        yes_no(capabilities.cpu_csr),
        yes_no(capabilities.cpu_jacobi),
        yes_no(capabilities.cpu_gauss_seidel),
        yes_no(capabilities.cpu_symmetric_gauss_seidel),
        yes_no(capabilities.cpu_conjugate_gradient),
        yes_no(capabilities.cpu_preconditioned_conjugate_gradient),
        yes_no(capabilities.cpu_bicgstab),
        yes_no(capabilities.cpu_diagonal_preconditioner),
        yes_no(capabilities.cpu_incomplete_cholesky_preconditioner),
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
        .try_fold(0usize, |total, bytes| total.checked_add(bytes));
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
        format_optional_usize(bytes_f64)
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
        .try_fold(0usize, |total, bytes| total.checked_add(bytes));
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
        format_optional_usize(bytes_f64)
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
    write_json_key(&mut writer, 4, "stopReason")?;
    write_json_string(&mut writer, &report.stop_reason.to_string())?;
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
    write_json_key(&mut writer, 4, "finalMomentumInitialResidual")?;
    write_json_optional_number(
        &mut writer,
        Some(report.final_momentum_initial_normalized_residual_norm),
    )?;
    writeln!(writer, ",")?;
    write_json_key(&mut writer, 4, "finalMomentumResidualNorm")?;
    write_json_optional_number(&mut writer, Some(report.final_momentum_residual_norm))?;
    writeln!(writer, ",")?;
    write_json_key(&mut writer, 4, "finalMomentumNormalizedResidualNorm")?;
    write_json_optional_number(
        &mut writer,
        Some(report.final_momentum_normalized_residual_norm),
    )?;
    writeln!(writer, ",")?;
    write_json_key(&mut writer, 4, "finalPressureCorrectionInitialResidual")?;
    write_json_optional_number(
        &mut writer,
        Some(report.final_pressure_correction_initial_normalized_residual_norm),
    )?;
    writeln!(writer, ",")?;
    write_json_key(&mut writer, 4, "finalPressureCorrectionResidualNorm")?;
    write_json_optional_number(
        &mut writer,
        Some(report.final_pressure_correction_residual_norm),
    )?;
    writeln!(writer, ",")?;
    write_json_key(
        &mut writer,
        4,
        "finalPressureCorrectionNormalizedResidualNorm",
    )?;
    write_json_optional_number(
        &mut writer,
        Some(report.final_pressure_correction_normalized_residual_norm),
    )?;
    writeln!(writer, ",")?;
    write_json_bool_field(
        &mut writer,
        4,
        "finalMomentumLinearConverged",
        report.linear_solve_summary.final_momentum_linear_converged,
    )?;
    writeln!(writer, ",")?;
    write_json_bool_field(
        &mut writer,
        4,
        "finalPressureLinearConverged",
        report.linear_solve_summary.final_pressure_linear_converged,
    )?;
    writeln!(writer)?;
    write_indent(&mut writer, 2)?;
    writeln!(writer, "}},")?;
    write_json_residual_control_summary(
        &mut writer,
        options,
        &report.residual_control,
        report.final_momentum_initial_normalized_residual_norm,
        report.final_pressure_correction_initial_normalized_residual_norm,
    )?;
    writeln!(writer, ",")?;
    write_json_linear_solve_summary(&mut writer, &report.linear_solve_summary)?;
    writeln!(writer, ",")?;
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
    write_json_pressure_assembly_diagnostics(&mut writer, report.pressure_assembly.as_ref())?;
    writeln!(writer, ",")?;
    write_json_field_summary(&mut writer, &report.fields)?;
    writeln!(writer, ",")?;
    write_json_laminar_simple_history(&mut writer, &report.history)?;
    writeln!(writer)?;
    writeln!(writer, "}}")?;

    writer.flush()
}

fn write_json_residual_control_summary(
    writer: &mut impl Write,
    options: &LaminarSimpleOptions,
    summary: &LaminarSimpleResidualControlSummary,
    momentum_initial_residual: f64,
    pressure_initial_residual: f64,
) -> std::io::Result<()> {
    write_json_key(writer, 2, "residualControl")?;
    writeln!(writer, "{{")?;
    write_json_bool_field(writer, 4, "configured", summary.configured)?;
    writeln!(writer, ",")?;
    write_json_bool_field(writer, 4, "checked", summary.checked)?;
    writeln!(writer, ",")?;
    write_json_bool_field(writer, 4, "satisfied", summary.satisfied)?;
    writeln!(writer, ",")?;
    write_json_key(writer, 4, "U")?;
    writeln!(writer, "{{")?;
    write_json_key(writer, 6, "tolerance")?;
    write_json_optional_number(writer, options.momentum_residual_control)?;
    writeln!(writer, ",")?;
    write_json_key(writer, 6, "initialResidual")?;
    write_json_optional_number(writer, Some(momentum_initial_residual))?;
    writeln!(writer, ",")?;
    write_json_key(writer, 6, "satisfied")?;
    write_json_optional_bool(writer, summary.momentum_satisfied)?;
    writeln!(writer)?;
    write_indent(writer, 4)?;
    writeln!(writer, "}},")?;
    write_json_key(writer, 4, "p")?;
    writeln!(writer, "{{")?;
    write_json_key(writer, 6, "tolerance")?;
    write_json_optional_number(writer, options.pressure_residual_control)?;
    writeln!(writer, ",")?;
    write_json_key(writer, 6, "initialResidual")?;
    write_json_optional_number(writer, Some(pressure_initial_residual))?;
    writeln!(writer, ",")?;
    write_json_key(writer, 6, "satisfied")?;
    write_json_optional_bool(writer, summary.pressure_satisfied)?;
    writeln!(writer)?;
    write_indent(writer, 4)?;
    writeln!(writer, "}}")?;
    write_indent(writer, 2)?;
    write!(writer, "}}")
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
    write_json_key(writer, 4, "momentumResidualControl")?;
    write_json_optional_number(writer, options.momentum_residual_control)?;
    writeln!(writer, ",")?;
    write_json_key(writer, 4, "pressureResidualControl")?;
    write_json_optional_number(writer, options.pressure_residual_control)?;
    writeln!(writer, ",")?;
    write_json_key(writer, 4, "pressureReferenceCell")?;
    write_json_optional_number(
        writer,
        options.pressure_reference_cell.map(|value| value as f64),
    )?;
    writeln!(writer, ",")?;
    write_json_key(writer, 4, "pressureReferenceValue")?;
    write_json_optional_number(writer, Some(options.pressure_reference_value))?;
    writeln!(writer, ",")?;
    write_json_number_field(
        writer,
        4,
        "nonOrthogonalCorrectors",
        options.non_orthogonal_correctors,
    )?;
    writeln!(writer, ",")?;
    write_json_bool_field(writer, 4, "consistent", options.simple_consistent)?;
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
    writeln!(writer, ",")?;
    write_json_laminar_simple_schemes(writer, 4, &options.schemes)?;
    writeln!(writer)?;
    write_indent(writer, 2)?;
    write!(writer, "}}")
}

fn write_json_laminar_simple_schemes(
    writer: &mut impl Write,
    indent: usize,
    schemes: &LaminarSimpleSchemes,
) -> std::io::Result<()> {
    write_json_key(writer, indent, "schemes")?;
    writeln!(writer, "{{")?;
    write_json_string_field(writer, indent + 2, "gradP", &schemes.grad_p.to_string())?;
    writeln!(writer, ",")?;
    write_json_string_field(writer, indent + 2, "gradU", &schemes.grad_u.to_string())?;
    writeln!(writer, ",")?;
    write_json_string_field(
        writer,
        indent + 2,
        "divPhiU",
        &schemes.div_phi_u.to_string(),
    )?;
    writeln!(writer, ",")?;
    write_json_string_field(
        writer,
        indent + 2,
        "laplacian",
        &schemes.laplacian.to_string(),
    )?;
    writeln!(writer, ",")?;
    write_json_string_field(
        writer,
        indent + 2,
        "interpolation",
        &schemes.interpolation.to_string(),
    )?;
    writeln!(writer, ",")?;
    write_json_string_field(writer, indent + 2, "snGrad", &schemes.sn_grad.to_string())?;
    writeln!(writer)?;
    write_indent(writer, indent)?;
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
    write_json_key(writer, 4, "hbyAL2Norm")?;
    write_json_optional_number(writer, Some(summary.hby_a_l2_norm))?;
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
        "velocityInletOutletFaces",
        summary.velocity_inlet_outlet_faces,
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

fn write_json_linear_solve_summary(
    writer: &mut impl Write,
    summary: &LinearSolveSummary,
) -> std::io::Result<()> {
    write_json_key(writer, 2, "linearSolves")?;
    writeln!(writer, "{{")?;
    write_json_number_field(writer, 4, "momentumPredictors", summary.momentum_predictors)?;
    writeln!(writer, ",")?;
    write_json_number_field(
        writer,
        4,
        "momentumNonConvergedPredictors",
        summary.momentum_non_converged_predictors,
    )?;
    writeln!(writer, ",")?;
    write_json_number_field(
        writer,
        4,
        "momentumComponentSolves",
        summary.momentum_component_solves,
    )?;
    writeln!(writer, ",")?;
    write_json_number_field(
        writer,
        4,
        "momentumComponentNonConvergedSolves",
        summary.momentum_component_non_converged_solves,
    )?;
    writeln!(writer, ",")?;
    write_json_number_field(
        writer,
        4,
        "pressureCorrectionSolves",
        summary.pressure_correction_solves,
    )?;
    writeln!(writer, ",")?;
    write_json_number_field(
        writer,
        4,
        "pressureCorrectionNonConvergedSolves",
        summary.pressure_correction_non_converged_solves,
    )?;
    writeln!(writer, ",")?;
    write_json_number_field(
        writer,
        4,
        "maxMomentumLinearIterationsPerSimple",
        summary.max_momentum_linear_iterations_per_simple,
    )?;
    writeln!(writer, ",")?;
    write_json_number_field(
        writer,
        4,
        "maxPressureLinearIterationsPerSimple",
        summary.max_pressure_linear_iterations_per_simple,
    )?;
    writeln!(writer, ",")?;
    write_json_key(writer, 4, "averageMomentumLinearIterationsPerSimple")?;
    write_json_optional_number(
        writer,
        Some(summary.average_momentum_linear_iterations_per_simple),
    )?;
    writeln!(writer, ",")?;
    write_json_key(writer, 4, "averagePressureLinearIterationsPerSimple")?;
    write_json_optional_number(
        writer,
        Some(summary.average_pressure_linear_iterations_per_simple),
    )?;
    writeln!(writer, ",")?;
    write_json_bool_field(
        writer,
        4,
        "finalMomentumLinearConverged",
        summary.final_momentum_linear_converged,
    )?;
    writeln!(writer, ",")?;
    write_json_bool_field(
        writer,
        4,
        "finalPressureLinearConverged",
        summary.final_pressure_linear_converged,
    )?;
    writeln!(writer)?;
    write_indent(writer, 2)?;
    write!(writer, "}}")
}

fn write_json_pressure_assembly_diagnostics(
    writer: &mut impl Write,
    diagnostics: Option<&PressureAssemblyDiagnostics>,
) -> std::io::Result<()> {
    write_json_key(writer, 2, "pressureAssembly")?;
    let Some(diagnostics) = diagnostics else {
        return write!(writer, "null");
    };

    writeln!(writer, "{{")?;
    write_json_scalar_diagnostic_summary(writer, 4, "rAU", &diagnostics.r_au)?;
    writeln!(writer, ",")?;
    write_json_scalar_diagnostic_summary(writer, 4, "rAtU", &diagnostics.r_at_u)?;
    writeln!(writer, ",")?;
    write_json_vector_diagnostic_summary(writer, 4, "HbyA", &diagnostics.hby_a)?;
    writeln!(writer, ",")?;
    write_json_face_flux_diagnostic_summary(
        writer,
        4,
        "phiHbyABeforeAdjust",
        &diagnostics.phi_hby_a_before_adjust,
    )?;
    writeln!(writer, ",")?;
    write_json_face_flux_diagnostic_summary(
        writer,
        4,
        "phiHbyAAfterAdjust",
        &diagnostics.phi_hby_a_after_adjust,
    )?;
    writeln!(writer, ",")?;
    write_json_scalar_diagnostic_summary(
        writer,
        4,
        "pressureSource",
        &diagnostics.pressure_source,
    )?;
    writeln!(writer, ",")?;
    write_json_face_flux_diagnostic_summary(
        writer,
        4,
        "pressureEquationFlux",
        &diagnostics.pressure_equation_flux,
    )?;
    writeln!(writer, ",")?;
    write_json_matrix_diagnostic_summary(
        writer,
        4,
        "pressureMatrix",
        &diagnostics.pressure_matrix,
    )?;
    writeln!(writer, ",")?;
    write_json_face_flux_diagnostic_summary(writer, 4, "pressureFlux", &diagnostics.pressure_flux)?;
    writeln!(writer, ",")?;
    write_json_face_flux_diagnostic_summary(writer, 4, "correctedPhi", &diagnostics.corrected_phi)?;
    writeln!(writer)?;
    write_indent(writer, 2)?;
    write!(writer, "}}")
}

fn write_json_scalar_diagnostic_summary(
    writer: &mut impl Write,
    indent: usize,
    key: &str,
    summary: &ScalarDiagnosticSummary,
) -> std::io::Result<()> {
    write_json_key(writer, indent, key)?;
    writeln!(writer, "{{")?;
    write_json_key(writer, indent + 2, "min")?;
    write_json_optional_number(writer, Some(summary.min))?;
    writeln!(writer, ",")?;
    write_json_key(writer, indent + 2, "max")?;
    write_json_optional_number(writer, Some(summary.max))?;
    writeln!(writer, ",")?;
    write_json_key(writer, indent + 2, "l2Norm")?;
    write_json_optional_number(writer, Some(summary.l2_norm))?;
    writeln!(writer, ",")?;
    write_json_key(writer, indent + 2, "sum")?;
    write_json_optional_number(writer, Some(summary.sum))?;
    writeln!(writer, ",")?;
    write_json_key(writer, indent + 2, "sumAbs")?;
    write_json_optional_number(writer, Some(summary.sum_abs))?;
    writeln!(writer)?;
    write_indent(writer, indent)?;
    write!(writer, "}}")
}

fn write_json_vector_diagnostic_summary(
    writer: &mut impl Write,
    indent: usize,
    key: &str,
    summary: &VectorDiagnosticSummary,
) -> std::io::Result<()> {
    write_json_key(writer, indent, key)?;
    writeln!(writer, "{{")?;
    write_json_key(writer, indent + 2, "minMagnitude")?;
    write_json_optional_number(writer, Some(summary.min_magnitude))?;
    writeln!(writer, ",")?;
    write_json_key(writer, indent + 2, "maxMagnitude")?;
    write_json_optional_number(writer, Some(summary.max_magnitude))?;
    writeln!(writer, ",")?;
    write_json_key(writer, indent + 2, "l2Norm")?;
    write_json_optional_number(writer, Some(summary.l2_norm))?;
    writeln!(writer, ",")?;
    write_json_key(writer, indent + 2, "xMin")?;
    write_json_optional_number(writer, Some(summary.x_min))?;
    writeln!(writer, ",")?;
    write_json_key(writer, indent + 2, "xMax")?;
    write_json_optional_number(writer, Some(summary.x_max))?;
    writeln!(writer, ",")?;
    write_json_key(writer, indent + 2, "yMin")?;
    write_json_optional_number(writer, Some(summary.y_min))?;
    writeln!(writer, ",")?;
    write_json_key(writer, indent + 2, "yMax")?;
    write_json_optional_number(writer, Some(summary.y_max))?;
    writeln!(writer, ",")?;
    write_json_key(writer, indent + 2, "zMin")?;
    write_json_optional_number(writer, Some(summary.z_min))?;
    writeln!(writer, ",")?;
    write_json_key(writer, indent + 2, "zMax")?;
    write_json_optional_number(writer, Some(summary.z_max))?;
    writeln!(writer)?;
    write_indent(writer, indent)?;
    write!(writer, "}}")
}

fn write_json_face_flux_diagnostic_summary(
    writer: &mut impl Write,
    indent: usize,
    key: &str,
    summary: &FaceFluxDiagnosticSummary,
) -> std::io::Result<()> {
    write_json_key(writer, indent, key)?;
    writeln!(writer, "{{")?;
    write_json_key(writer, indent + 2, "min")?;
    write_json_optional_number(writer, Some(summary.min))?;
    writeln!(writer, ",")?;
    write_json_key(writer, indent + 2, "max")?;
    write_json_optional_number(writer, Some(summary.max))?;
    writeln!(writer, ",")?;
    write_json_key(writer, indent + 2, "l2Norm")?;
    write_json_optional_number(writer, Some(summary.l2_norm))?;
    writeln!(writer, ",")?;
    write_json_key(writer, indent + 2, "sum")?;
    write_json_optional_number(writer, Some(summary.sum))?;
    writeln!(writer, ",")?;
    write_json_key(writer, indent + 2, "sumAbs")?;
    write_json_optional_number(writer, Some(summary.sum_abs))?;
    writeln!(writer, ",")?;
    write_json_key(writer, indent + 2, "internalSumAbs")?;
    write_json_optional_number(writer, Some(summary.internal_sum_abs))?;
    writeln!(writer, ",")?;
    write_json_key(writer, indent + 2, "boundarySum")?;
    write_json_optional_number(writer, Some(summary.boundary_sum))?;
    writeln!(writer, ",")?;
    write_json_key(writer, indent + 2, "boundarySumAbs")?;
    write_json_optional_number(writer, Some(summary.boundary_sum_abs))?;
    writeln!(writer)?;
    write_indent(writer, indent)?;
    write!(writer, "}}")
}

fn write_json_matrix_diagnostic_summary(
    writer: &mut impl Write,
    indent: usize,
    key: &str,
    summary: &MatrixDiagnosticSummary,
) -> std::io::Result<()> {
    write_json_key(writer, indent, key)?;
    writeln!(writer, "{{")?;
    write_json_number_field(writer, indent + 2, "rows", summary.rows)?;
    writeln!(writer, ",")?;
    write_json_number_field(writer, indent + 2, "cols", summary.cols)?;
    writeln!(writer, ",")?;
    write_json_number_field(writer, indent + 2, "nonzeros", summary.nonzeros)?;
    writeln!(writer, ",")?;
    write_json_key(writer, indent + 2, "diagonalMin")?;
    write_json_optional_number(writer, Some(summary.diagonal_min))?;
    writeln!(writer, ",")?;
    write_json_key(writer, indent + 2, "diagonalMax")?;
    write_json_optional_number(writer, Some(summary.diagonal_max))?;
    writeln!(writer, ",")?;
    write_json_key(writer, indent + 2, "diagonalSumAbs")?;
    write_json_optional_number(writer, Some(summary.diagonal_sum_abs))?;
    writeln!(writer, ",")?;
    write_json_key(writer, indent + 2, "offDiagonalSumAbs")?;
    write_json_optional_number(writer, Some(summary.off_diagonal_sum_abs))?;
    writeln!(writer, ",")?;
    write_json_key(writer, indent + 2, "maxRowSumAbs")?;
    write_json_optional_number(writer, Some(summary.max_row_sum_abs))?;
    writeln!(writer, ",")?;
    write_json_key(writer, indent + 2, "maxRowOffDiagonalSumAbs")?;
    write_json_optional_number(writer, Some(summary.max_row_off_diagonal_sum_abs))?;
    writeln!(writer)?;
    write_indent(writer, indent)?;
    write!(writer, "}}")
}

fn write_json_field_summary(
    writer: &mut impl Write,
    summary: &LaminarSimpleFieldSummary,
) -> std::io::Result<()> {
    write_json_key(writer, 2, "fields")?;
    writeln!(writer, "{{")?;
    write_json_vector_diagnostic_summary(writer, 4, "velocity", &summary.velocity)?;
    writeln!(writer, ",")?;
    write_json_scalar_diagnostic_summary(writer, 4, "pressure", &summary.pressure)?;
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
        write_json_bool_field(
            writer,
            6,
            "momentumLinearConverged",
            item.momentum_linear_converged,
        )?;
        writeln!(writer, ",")?;
        write_json_key(writer, 6, "momentumComponentLinearConverged")?;
        write_json_bool_array(writer, &item.momentum_component_linear_converged)?;
        writeln!(writer, ",")?;
        write_json_number_field(
            writer,
            6,
            "pressureLinearIterations",
            item.pressure_linear_iterations,
        )?;
        writeln!(writer, ",")?;
        write_json_bool_field(
            writer,
            6,
            "pressureLinearConverged",
            item.pressure_linear_converged,
        )?;
        writeln!(writer, ",")?;
        write_json_number_field(
            writer,
            6,
            "pressureLinearSolves",
            item.pressure_linear_solves,
        )?;
        writeln!(writer, ",")?;
        write_json_number_field(
            writer,
            6,
            "pressureLinearNonConvergedSolves",
            item.pressure_linear_non_converged_solves,
        )?;
        writeln!(writer, ",")?;
        write_json_key(writer, 6, "momentumInitialResidual")?;
        write_json_optional_number(writer, Some(item.momentum_initial_normalized_residual_norm))?;
        writeln!(writer, ",")?;
        write_json_key(writer, 6, "momentumResidualNorm")?;
        write_json_optional_number(writer, Some(item.momentum_residual_norm))?;
        writeln!(writer, ",")?;
        write_json_key(writer, 6, "momentumNormalizedResidualNorm")?;
        write_json_optional_number(writer, Some(item.momentum_normalized_residual_norm))?;
        writeln!(writer, ",")?;
        write_json_key(writer, 6, "momentumComponentResidualNorms")?;
        write_json_optional_f64_array(writer, Some(&item.momentum_component_residual_norms))?;
        writeln!(writer, ",")?;
        write_json_key(writer, 6, "momentumComponentInitialResiduals")?;
        write_json_optional_f64_array(
            writer,
            Some(&item.momentum_component_initial_normalized_residual_norms),
        )?;
        writeln!(writer, ",")?;
        write_json_key(writer, 6, "momentumComponentNormalizedResidualNorms")?;
        write_json_optional_f64_array(
            writer,
            Some(&item.momentum_component_normalized_residual_norms),
        )?;
        writeln!(writer, ",")?;
        write_json_key(writer, 6, "momentumDiagonalMin")?;
        write_json_optional_number(writer, Some(item.momentum_diagonal_min))?;
        writeln!(writer, ",")?;
        write_json_key(writer, 6, "momentumDiagonalMax")?;
        write_json_optional_number(writer, Some(item.momentum_diagonal_max))?;
        writeln!(writer, ",")?;
        write_json_key(writer, 6, "momentumH1Min")?;
        write_json_optional_number(writer, Some(item.momentum_h1_min))?;
        writeln!(writer, ",")?;
        write_json_key(writer, 6, "momentumH1Max")?;
        write_json_optional_number(writer, Some(item.momentum_h1_max))?;
        writeln!(writer, ",")?;
        write_json_key(writer, 6, "pressureCorrectionInitialResidual")?;
        write_json_optional_number(
            writer,
            Some(item.pressure_correction_initial_normalized_residual_norm),
        )?;
        writeln!(writer, ",")?;
        write_json_key(writer, 6, "pressureCorrectionResidualNorm")?;
        write_json_optional_number(writer, Some(item.pressure_correction_residual_norm))?;
        writeln!(writer, ",")?;
        write_json_key(writer, 6, "pressureCorrectionNormalizedResidualNorm")?;
        write_json_optional_number(
            writer,
            Some(item.pressure_correction_normalized_residual_norm),
        )?;
        writeln!(writer, ",")?;
        write_json_bool_field(
            writer,
            6,
            "residualControlConfigured",
            item.residual_control.configured,
        )?;
        writeln!(writer, ",")?;
        write_json_bool_field(
            writer,
            6,
            "residualControlChecked",
            item.residual_control.checked,
        )?;
        writeln!(writer, ",")?;
        write_json_bool_field(
            writer,
            6,
            "residualControlSatisfied",
            item.residual_control.satisfied,
        )?;
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
        writeln!(writer, ",")?;
        write_json_key(writer, 6, "adjustPhiGlobalFluxBefore")?;
        write_json_optional_number(writer, Some(item.adjust_phi_global_flux_before))?;
        writeln!(writer, ",")?;
        write_json_key(writer, 6, "adjustPhiGlobalFluxAfter")?;
        write_json_optional_number(writer, Some(item.adjust_phi_global_flux_after))?;
        writeln!(writer, ",")?;
        write_json_number_field(
            writer,
            6,
            "adjustPhiAdjustedFaces",
            item.adjust_phi_adjusted_faces,
        )?;
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

    writeln!(writer, "# Laminar SIMPLE Report")?;
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
        "| U residualControl | {} |",
        format_optional_scientific(options.momentum_residual_control)
    )?;
    writeln!(
        writer,
        "| p residualControl | {} |",
        format_optional_scientific(options.pressure_residual_control)
    )?;
    writeln!(
        writer,
        "| pRefCell | {} |",
        options
            .pressure_reference_cell
            .map(|value| value.to_string())
            .unwrap_or_else(|| "n/a".to_string())
    )?;
    writeln!(
        writer,
        "| pRefValue [Pa] | {} |",
        format_scientific(options.pressure_reference_value)
    )?;
    writeln!(
        writer,
        "| Non-orthogonal correctors | {} |",
        options.non_orthogonal_correctors
    )?;
    writeln!(writer, "| grad(p) scheme | {} |", options.schemes.grad_p)?;
    writeln!(writer, "| grad(U) scheme | {} |", options.schemes.grad_u)?;
    writeln!(
        writer,
        "| div(phi,U) scheme | {} |",
        options.schemes.div_phi_u
    )?;
    writeln!(
        writer,
        "| laplacian scheme | {} |",
        options.schemes.laplacian
    )?;
    writeln!(
        writer,
        "| interpolation scheme | {} |",
        options.schemes.interpolation
    )?;
    writeln!(writer, "| snGrad scheme | {} |", options.schemes.sn_grad)?;
    writeln!(
        writer,
        "| Consistent SIMPLE | {} |",
        yes_no(options.simple_consistent)
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
    writeln!(writer, "| Stop reason | {} |", report.stop_reason)?;
    writeln!(
        writer,
        "| residualControl state | {} |",
        residual_control_state(report.residual_control)
    )?;
    writeln!(
        writer,
        "| U residualControl satisfied | {} |",
        format_optional_bool(report.residual_control.momentum_satisfied)
    )?;
    writeln!(
        writer,
        "| p residualControl satisfied | {} |",
        format_optional_bool(report.residual_control.pressure_satisfied)
    )?;
    writeln!(
        writer,
        "| Final momentum linear converged | {} |",
        yes_no(report.linear_solve_summary.final_momentum_linear_converged)
    )?;
    writeln!(
        writer,
        "| Final pressure linear converged | {} |",
        yes_no(report.linear_solve_summary.final_pressure_linear_converged)
    )?;
    writeln!(
        writer,
        "| Final continuity L2 | {} |",
        format_scientific(report.final_continuity.l2_norm)
    )?;
    writeln!(
        writer,
        "| Final SIMPLE iteration U initial residual | {} |",
        format_scientific(report.final_momentum_initial_normalized_residual_norm)
    )?;
    writeln!(
        writer,
        "| Final SIMPLE iteration U final residual | {} |",
        format_scientific(report.final_momentum_normalized_residual_norm)
    )?;
    writeln!(
        writer,
        "| Momentum raw L2 residual norm | {} |",
        format_scientific(report.final_momentum_residual_norm)
    )?;
    writeln!(
        writer,
        "| Final SIMPLE iteration p initial residual | {} |",
        format_scientific(report.final_pressure_correction_initial_normalized_residual_norm)
    )?;
    writeln!(
        writer,
        "| Final SIMPLE iteration p final residual | {} |",
        format_scientific(report.final_pressure_correction_normalized_residual_norm)
    )?;
    writeln!(
        writer,
        "| Pressure-correction raw L2 residual norm | {} |",
        format_scientific(report.final_pressure_correction_residual_norm)
    )?;
    writeln!(
        writer,
        "| HbyA L2 norm | {} |",
        format_scientific(report.operator_summary.hby_a_l2_norm)
    )?;
    writeln!(
        writer,
        "| Wall clock [s] | {} |",
        format_scientific(wall_clock_seconds)
    )?;
    writeln!(
        writer,
        "| Velocity magnitude min | {} |",
        format_scientific(report.fields.velocity.min_magnitude)
    )?;
    writeln!(
        writer,
        "| Velocity magnitude max | {} |",
        format_scientific(report.fields.velocity.max_magnitude)
    )?;
    writeln!(
        writer,
        "| Velocity L2 norm | {} |",
        format_scientific(report.fields.velocity.l2_norm)
    )?;
    writeln!(
        writer,
        "| Pressure min [Pa] | {} |",
        format_scientific(report.fields.pressure.min)
    )?;
    writeln!(
        writer,
        "| Pressure max [Pa] | {} |",
        format_scientific(report.fields.pressure.max)
    )?;
    writeln!(
        writer,
        "| Pressure L2 norm | {} |",
        format_scientific(report.fields.pressure.l2_norm)
    )?;
    writeln!(writer)?;
    writeln!(writer, "## Linear Solve Profile")?;
    writeln!(writer)?;
    writeln!(writer, "| Quantity | Value |")?;
    writeln!(writer, "| --- | ---: |")?;
    writeln!(
        writer,
        "| Momentum predictors | {} |",
        report.linear_solve_summary.momentum_predictors
    )?;
    writeln!(
        writer,
        "| Momentum non-converged predictors | {} |",
        report
            .linear_solve_summary
            .momentum_non_converged_predictors
    )?;
    writeln!(
        writer,
        "| Momentum component solves | {} |",
        report.linear_solve_summary.momentum_component_solves
    )?;
    writeln!(
        writer,
        "| Momentum component non-converged solves | {} |",
        report
            .linear_solve_summary
            .momentum_component_non_converged_solves
    )?;
    writeln!(
        writer,
        "| Pressure correction solves | {} |",
        report.linear_solve_summary.pressure_correction_solves
    )?;
    writeln!(
        writer,
        "| Pressure correction non-converged solves | {} |",
        report
            .linear_solve_summary
            .pressure_correction_non_converged_solves
    )?;
    writeln!(
        writer,
        "| Max momentum linear iterations per SIMPLE | {} |",
        report
            .linear_solve_summary
            .max_momentum_linear_iterations_per_simple
    )?;
    writeln!(
        writer,
        "| Max pressure linear iterations per SIMPLE | {} |",
        report
            .linear_solve_summary
            .max_pressure_linear_iterations_per_simple
    )?;
    writeln!(
        writer,
        "| Average momentum linear iterations per SIMPLE | {} |",
        format_scientific(
            report
                .linear_solve_summary
                .average_momentum_linear_iterations_per_simple,
        )
    )?;
    writeln!(
        writer,
        "| Average pressure linear iterations per SIMPLE | {} |",
        format_scientific(
            report
                .linear_solve_summary
                .average_pressure_linear_iterations_per_simple,
        )
    )?;
    writeln!(writer)?;
    if let Some(diagnostics) = &report.pressure_assembly {
        writeln!(writer, "## Pressure Assembly Diagnostics")?;
        writeln!(writer)?;
        writeln!(writer, "| Field | Min | Max | L2 | Sum abs |")?;
        writeln!(writer, "| --- | ---: | ---: | ---: | ---: |")?;
        writeln!(
            writer,
            "| rAU | {} | {} | {} | {} |",
            format_scientific(diagnostics.r_au.min),
            format_scientific(diagnostics.r_au.max),
            format_scientific(diagnostics.r_au.l2_norm),
            format_scientific(diagnostics.r_au.sum_abs)
        )?;
        writeln!(
            writer,
            "| rAtU | {} | {} | {} | {} |",
            format_scientific(diagnostics.r_at_u.min),
            format_scientific(diagnostics.r_at_u.max),
            format_scientific(diagnostics.r_at_u.l2_norm),
            format_scientific(diagnostics.r_at_u.sum_abs)
        )?;
        writeln!(
            writer,
            "| pressureSource | {} | {} | {} | {} |",
            format_scientific(diagnostics.pressure_source.min),
            format_scientific(diagnostics.pressure_source.max),
            format_scientific(diagnostics.pressure_source.l2_norm),
            format_scientific(diagnostics.pressure_source.sum_abs)
        )?;
        writeln!(writer)?;
        writeln!(
            writer,
            "| Vector | Magnitude min | Magnitude max | L2 | x min/max |"
        )?;
        writeln!(writer, "| --- | ---: | ---: | ---: | --- |")?;
        writeln!(
            writer,
            "| HbyA | {} | {} | {} | {} / {} |",
            format_scientific(diagnostics.hby_a.min_magnitude),
            format_scientific(diagnostics.hby_a.max_magnitude),
            format_scientific(diagnostics.hby_a.l2_norm),
            format_scientific(diagnostics.hby_a.x_min),
            format_scientific(diagnostics.hby_a.x_max)
        )?;
        writeln!(writer)?;
        writeln!(
            writer,
            "| Matrix | Rows | Nonzeros | Diagonal min/max | Off-diagonal sum abs | Max row sum abs |"
        )?;
        writeln!(writer, "| --- | ---: | ---: | --- | ---: | ---: |")?;
        writeln!(
            writer,
            "| pressureMatrix | {} | {} | {} / {} | {} | {} |",
            diagnostics.pressure_matrix.rows,
            diagnostics.pressure_matrix.nonzeros,
            format_scientific(diagnostics.pressure_matrix.diagonal_min),
            format_scientific(diagnostics.pressure_matrix.diagonal_max),
            format_scientific(diagnostics.pressure_matrix.off_diagonal_sum_abs),
            format_scientific(diagnostics.pressure_matrix.max_row_sum_abs)
        )?;
        writeln!(writer)?;
        writeln!(
            writer,
            "| Face flux | Boundary sum | Boundary sum abs | Total sum abs | L2 |"
        )?;
        writeln!(writer, "| --- | ---: | ---: | ---: | ---: |")?;
        write_markdown_face_flux_diagnostic(
            &mut writer,
            "phiHbyA before adjust",
            &diagnostics.phi_hby_a_before_adjust,
        )?;
        write_markdown_face_flux_diagnostic(
            &mut writer,
            "phiHbyA after adjust",
            &diagnostics.phi_hby_a_after_adjust,
        )?;
        write_markdown_face_flux_diagnostic(
            &mut writer,
            "pressureEquationFlux",
            &diagnostics.pressure_equation_flux,
        )?;
        write_markdown_face_flux_diagnostic(
            &mut writer,
            "pressureFlux",
            &diagnostics.pressure_flux,
        )?;
        write_markdown_face_flux_diagnostic(
            &mut writer,
            "correctedPhi",
            &diagnostics.corrected_phi,
        )?;
        writeln!(writer)?;
    }
    writeln!(writer, "## Iterations")?;
    writeln!(writer)?;
    writeln!(
        writer,
        "| Iteration | Continuity before | Continuity after | Pressure correction | U initial | U final | U linear iter/ok | U components initial | U components final | p initial | p final | p linear iter/ok | p-solves/nonconv | residualControl | A diag min/max | H1 min/max | adjustPhi before/after/faces | U change | p change |"
    )?;
    writeln!(
        writer,
        "| ---: | ---: | ---: | --- | ---: | ---: | --- | --- | --- | ---: | ---: | --- | --- | --- | --- | --- | --- | ---: | ---: |"
    )?;
    for item in &report.history {
        writeln!(
            writer,
            "| {} | {} | {} | {} | {} | {} | {} / {} | {} | {} | {} | {} | {} / {} | {} / {} | {} | {} / {} | {} / {} | {} / {} / {} | {} | {} |",
            item.iteration,
            format_scientific(item.continuity_before.l2_norm),
            format_scientific(item.continuity_after.l2_norm),
            if item.pressure_correction_accepted {
                "accepted"
            } else {
                "skipped"
            },
            format_scientific(item.momentum_initial_normalized_residual_norm),
            format_scientific(item.momentum_normalized_residual_norm),
            item.momentum_linear_iterations,
            yes_no(item.momentum_linear_converged),
            format_triplet(item.momentum_component_initial_normalized_residual_norms),
            format_triplet(item.momentum_component_normalized_residual_norms),
            format_scientific(item.pressure_correction_initial_normalized_residual_norm),
            format_scientific(item.pressure_correction_normalized_residual_norm),
            item.pressure_linear_iterations,
            yes_no(item.pressure_linear_converged),
            item.pressure_linear_solves,
            item.pressure_linear_non_converged_solves,
            residual_control_state(item.residual_control),
            format_scientific(item.momentum_diagonal_min),
            format_scientific(item.momentum_diagonal_max),
            format_scientific(item.momentum_h1_min),
            format_scientific(item.momentum_h1_max),
            format_scientific(item.adjust_phi_global_flux_before),
            format_scientific(item.adjust_phi_global_flux_after),
            item.adjust_phi_adjusted_faces,
            format_percent(item.relative_velocity_change_l2),
            format_percent(item.relative_pressure_change_l2)
        )?;
    }

    writer.flush()
}

fn write_markdown_face_flux_diagnostic(
    writer: &mut impl Write,
    label: &str,
    summary: &FaceFluxDiagnosticSummary,
) -> std::io::Result<()> {
    writeln!(
        writer,
        "| {} | {} | {} | {} | {} |",
        label,
        format_scientific(summary.boundary_sum),
        format_scientific(summary.boundary_sum_abs),
        format_scientific(summary.sum_abs),
        format_scientific(summary.l2_norm)
    )
}

fn write_laminar_simple_fields(
    fields: &InitialFieldSet,
    report: &LaminarSimpleReport,
    output_dir: &Path,
) -> std::io::Result<()> {
    if report.final_velocity.len() != report.cells {
        return Err(invalid_field_data(format!(
            "final U field has {} cells, expected {}",
            report.final_velocity.len(),
            report.cells
        )));
    }
    if report.final_pressure.len() != report.cells {
        return Err(invalid_field_data(format!(
            "final p field has {} cells, expected {}",
            report.final_pressure.len(),
            report.cells
        )));
    }
    if !report
        .final_velocity
        .iter()
        .all(|value| value.x.is_finite() && value.y.is_finite() && value.z.is_finite())
    {
        return Err(invalid_field_data(
            "final U field contains non-finite values".to_string(),
        ));
    }
    if !report.final_pressure.iter().all(|value| value.is_finite()) {
        return Err(invalid_field_data(
            "final p field contains non-finite values".to_string(),
        ));
    }

    std::fs::create_dir_all(output_dir)?;
    let location = openfoam_output_location(output_dir);
    let velocity_field = solver_initial_field(fields, "U", "volVectorField")?;
    let pressure_field = solver_initial_field(fields, "p", "volScalarField")?;

    write_openfoam_vector_field(
        &output_dir.join("U"),
        velocity_field,
        &location,
        &report.final_velocity,
    )?;
    write_openfoam_scalar_field(
        &output_dir.join("p"),
        pressure_field,
        &location,
        &report.final_pressure,
    )
}

fn solver_initial_field<'a>(
    fields: &'a InitialFieldSet,
    name: &str,
    class_name: &str,
) -> std::io::Result<&'a FieldFile> {
    let field = fields
        .fields
        .iter()
        .find(|field| field.region.is_none() && field.name == name)
        .ok_or_else(|| {
            invalid_field_data(format!(
                "field '{}' was not found below {}",
                name,
                fields.case_dir.join("0").display()
            ))
        })?;
    if field.class_name.as_deref() != Some(class_name) {
        return Err(invalid_field_data(format!(
            "field '{}' has class '{}', expected '{}'",
            name,
            field.class_name.as_deref().unwrap_or("unknown"),
            class_name
        )));
    }
    Ok(field)
}

fn write_openfoam_vector_field(
    path: &Path,
    source_field: &FieldFile,
    location: &str,
    values: &[Point3],
) -> std::io::Result<()> {
    ensure_parent_dir(path)?;
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    write_openfoam_field_header(
        &mut writer,
        source_field
            .class_name
            .as_deref()
            .unwrap_or("volVectorField"),
        location,
        "U",
    )?;
    writeln!(writer)?;
    write_openfoam_dimensions(&mut writer, source_field, "0 1 -1 0 0 0 0")?;
    writeln!(writer)?;
    writeln!(writer, "internalField nonuniform List<vector>")?;
    writeln!(writer, "{}", values.len())?;
    writeln!(writer, "(")?;
    for value in values {
        writeln!(
            writer,
            "    ({} {} {})",
            format_openfoam_f64(value.x),
            format_openfoam_f64(value.y),
            format_openfoam_f64(value.z)
        )?;
    }
    writeln!(writer, ");")?;
    writeln!(writer)?;
    write_openfoam_boundary_field(&mut writer, source_field)?;

    writer.flush()
}

fn write_openfoam_scalar_field(
    path: &Path,
    source_field: &FieldFile,
    location: &str,
    values: &[f64],
) -> std::io::Result<()> {
    ensure_parent_dir(path)?;
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    write_openfoam_field_header(
        &mut writer,
        source_field
            .class_name
            .as_deref()
            .unwrap_or("volScalarField"),
        location,
        "p",
    )?;
    writeln!(writer)?;
    write_openfoam_dimensions(&mut writer, source_field, "1 -1 -2 0 0 0 0")?;
    writeln!(writer)?;
    writeln!(writer, "internalField nonuniform List<scalar>")?;
    writeln!(writer, "{}", values.len())?;
    writeln!(writer, "(")?;
    for value in values {
        writeln!(writer, "    {}", format_openfoam_f64(*value))?;
    }
    writeln!(writer, ");")?;
    writeln!(writer)?;
    write_openfoam_boundary_field(&mut writer, source_field)?;

    writer.flush()
}

fn write_openfoam_field_header(
    writer: &mut impl Write,
    class_name: &str,
    location: &str,
    object: &str,
) -> std::io::Result<()> {
    writeln!(writer, "FoamFile")?;
    writeln!(writer, "{{")?;
    writeln!(writer, "    version 2.0;")?;
    writeln!(writer, "    format ascii;")?;
    writeln!(writer, "    class {class_name};")?;
    writeln!(
        writer,
        "    location \"{}\";",
        escape_openfoam_string(location)
    )?;
    writeln!(writer, "    object {object};")?;
    writeln!(writer, "}}")
}

fn write_openfoam_dimensions(
    writer: &mut impl Write,
    field: &FieldFile,
    fallback: &str,
) -> std::io::Result<()> {
    let dimensions = field
        .dimensions
        .as_ref()
        .filter(|values| !values.is_empty())
        .map(|values| values.join(" "))
        .unwrap_or_else(|| fallback.to_string());
    writeln!(writer, "dimensions [{dimensions}];")
}

fn write_openfoam_boundary_field(
    writer: &mut impl Write,
    field: &FieldFile,
) -> std::io::Result<()> {
    writeln!(writer, "boundaryField")?;
    writeln!(writer, "{{")?;
    for patch in &field.boundary_patches {
        writeln!(writer, "    {}", patch.name)?;
        writeln!(writer, "    {{")?;
        if let Some(patch_type) = &patch.patch_type {
            writeln!(writer, "        type {patch_type};")?;
        }
        if let Some(inlet_value) = &patch.inlet_value {
            write_openfoam_field_value(writer, "inletValue", inlet_value, 8)?;
        }
        if let Some(value) = &patch.value {
            write_openfoam_field_value(writer, "value", value, 8)?;
        }
        writeln!(writer, "    }}")?;
    }
    writeln!(writer, "}}")
}

fn write_openfoam_field_value(
    writer: &mut impl Write,
    keyword: &str,
    value: &FieldValueSummary,
    indent: usize,
) -> std::io::Result<()> {
    match value {
        FieldValueSummary::Uniform(value) => {
            write_indent(writer, indent)?;
            writeln!(writer, "{keyword} uniform {};", value.trim())
        }
        FieldValueSummary::Other(value) => {
            write_indent(writer, indent)?;
            writeln!(writer, "{keyword} {};", value.trim().trim_end_matches(';'))
        }
        FieldValueSummary::NonUniform {
            value_type,
            count,
            values,
        } => {
            let value_type = value_type.as_deref().ok_or_else(|| {
                invalid_field_data(format!(
                    "boundary entry '{keyword}' has nonuniform data without a value type"
                ))
            })?;
            let components = openfoam_nonuniform_components(value_type).ok_or_else(|| {
                invalid_field_data(format!(
                    "boundary entry '{keyword}' uses unsupported nonuniform type '{value_type}'"
                ))
            })?;
            let values = values.as_ref().ok_or_else(|| {
                invalid_field_data(format!(
                    "boundary entry '{keyword}' has nonuniform {value_type} data without loaded numeric values"
                ))
            })?;
            let count = count.unwrap_or(values.len() / components);
            if values.len() != count * components {
                return Err(invalid_field_data(format!(
                    "boundary entry '{keyword}' has {} scalar values, expected {} for {} {} entries",
                    values.len(),
                    count * components,
                    count,
                    value_type
                )));
            }

            write_indent(writer, indent)?;
            writeln!(writer, "{keyword} nonuniform {value_type}")?;
            write_indent(writer, indent)?;
            writeln!(writer, "{count}")?;
            write_indent(writer, indent)?;
            writeln!(writer, "(")?;
            for entry in values.chunks(components) {
                write_indent(writer, indent + 4)?;
                if components == 1 {
                    writeln!(writer, "{}", format_openfoam_f64(entry[0]))?;
                } else {
                    writeln!(
                        writer,
                        "({} {} {})",
                        format_openfoam_f64(entry[0]),
                        format_openfoam_f64(entry[1]),
                        format_openfoam_f64(entry[2])
                    )?;
                }
            }
            write_indent(writer, indent)?;
            writeln!(writer, ");")
        }
    }
}

fn openfoam_nonuniform_components(value_type: &str) -> Option<usize> {
    match value_type {
        "List<scalar>" | "scalarField" | "Field<scalar>" => Some(1),
        "List<vector>" | "vectorField" | "Field<vector>" => Some(3),
        _ => None,
    }
}

fn openfoam_output_location(output_dir: &Path) -> String {
    output_dir
        .file_name()
        .map(|name| name.to_string_lossy().to_string())
        .unwrap_or_else(|| output_dir.display().to_string())
}

fn escape_openfoam_string(value: &str) -> String {
    value.replace('\\', "\\\\").replace('"', "\\\"")
}

fn format_openfoam_f64(value: f64) -> String {
    if value == 0.0 {
        "0".to_string()
    } else {
        format!("{value:.16e}")
    }
}

fn invalid_field_data(message: String) -> Error {
    Error::new(ErrorKind::InvalidData, message)
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

fn write_json_bool_array(writer: &mut impl Write, values: &[bool]) -> std::io::Result<()> {
    write!(writer, "[")?;
    for (index, value) in values.iter().enumerate() {
        if index > 0 {
            write!(writer, ", ")?;
        }
        write!(writer, "{value}")?;
    }
    write!(writer, "]")
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

fn residual_control_state(summary: LaminarSimpleResidualControlSummary) -> &'static str {
    if !summary.configured {
        "not-configured"
    } else if !summary.checked {
        "not-checked"
    } else if summary.satisfied {
        "satisfied"
    } else {
        "not-satisfied"
    }
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

fn format_triplet(values: [f64; 3]) -> String {
    format!(
        "{} / {} / {}",
        format_scientific(values[0]),
        format_scientific(values[1]),
        format_scientific(values[2])
    )
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
    max_simple_iterations: Option<usize>,
    min_simple_iterations: Option<usize>,
    pressure_reference_cell: Option<usize>,
    pressure_reference_value: Option<f64>,
    non_orthogonal_correctors: Option<usize>,
    simple_consistent: Option<bool>,
    velocity_relaxation: Option<f64>,
    pressure_relaxation: Option<f64>,
    solve_verbose: bool,
    solve_residual_csv: Option<PathBuf>,
    solve_residual_plot: Option<PathBuf>,
    report_json: Option<PathBuf>,
    report_markdown: Option<PathBuf>,
    write_final_fields: Option<PathBuf>,
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
    let mut shared_flow_option_seen = false;
    let mut density = None;
    let mut pressure_drop = None;
    let mut dynamic_viscosity = None;
    let mut length = None;
    let mut diameter = None;
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
    let mut scalar_diffusion_linear_solver_error = None;
    let mut scalar_diffusion_tolerance = 1.0e-10;
    let mut scalar_diffusion_max_iterations = 10_000;
    let mut laminar_linear_tolerance = None;
    let mut laminar_max_linear_iterations = None;
    let mut momentum_linear_tolerance = None;
    let mut pressure_linear_tolerance = None;
    let mut momentum_max_linear_iterations = None;
    let mut pressure_max_linear_iterations = None;
    let mut max_simple_iterations = None;
    let mut min_simple_iterations = None;
    let mut pressure_reference_cell = None;
    let mut pressure_reference_value = None;
    let mut non_orthogonal_correctors = None;
    let mut simple_consistent = None;
    let mut velocity_relaxation = None;
    let mut pressure_relaxation = None;
    let mut solve_verbose = false;
    let mut solve_residual_csv = None;
    let mut solve_residual_plot = None;
    let mut solve_report_json = None;
    let mut solve_report_markdown = None;
    let mut write_final_fields = None;
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
                shared_flow_option_seen = true;
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
                    "--linearSolver requires 'bicgstab', 'gaussSeidel', 'cg', 'pcg', or 'jacobi'"
                        .to_string()
                })?;
                match parse_scalar_diffusion_linear_solver(value) {
                    Ok(solver) => {
                        scalar_diffusion_linear_solver = solver;
                        scalar_diffusion_linear_solver_error = None;
                    }
                    Err(error) => scalar_diffusion_linear_solver_error = Some(error),
                }
                laminar_linear_solver = Some(parse_laminar_simple_linear_solver(value)?);
                linear_solve_option_seen = true;
                index += 2;
            }
            "-momentumLinearSolver"
            | "--momentumLinearSolver"
            | "-momentum-linear-solver"
            | "--momentum-linear-solver" => {
                let value = args.get(index + 1).ok_or_else(|| {
                    "--momentumLinearSolver requires 'bicgstab', 'gaussSeidel', 'cg', 'pcg', or 'jacobi'"
                        .to_string()
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
                    "--pressureLinearSolver requires 'bicgstab', 'gaussSeidel', 'cg', 'pcg', or 'jacobi'"
                        .to_string()
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
                    "--momentumPreconditioner requires 'none', 'diagonal', 'DIC', or 'incompleteCholesky'"
                        .to_string()
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
                    "--pressurePreconditioner requires 'none', 'diagonal', 'DIC', or 'incompleteCholesky'"
                        .to_string()
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
                max_simple_iterations =
                    Some(parse_positive_usize_arg("--maxSimpleIterations", value)?);
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
            "-nNonOrthogonalCorrectors"
            | "--nNonOrthogonalCorrectors"
            | "-n-non-orthogonal-correctors"
            | "--n-non-orthogonal-correctors" => {
                let value = args.get(index + 1).ok_or_else(|| {
                    "--nNonOrthogonalCorrectors requires a non-negative integer".to_string()
                })?;
                non_orthogonal_correctors = Some(parse_non_negative_usize_arg(
                    "--nNonOrthogonalCorrectors",
                    value,
                )?);
                laminar_simple_option_seen = true;
                index += 2;
            }
            "-simpleConsistent"
            | "--simpleConsistent"
            | "-simple-consistent"
            | "--simple-consistent" => {
                if args
                    .get(index + 1)
                    .is_some_and(|value| !value.starts_with('-'))
                {
                    simple_consistent =
                        Some(parse_bool_arg("--simpleConsistent", &args[index + 1])?);
                    index += 2;
                } else {
                    simple_consistent = Some(true);
                    index += 1;
                }
                laminar_simple_option_seen = true;
            }
            "-noSimpleConsistent"
            | "--noSimpleConsistent"
            | "-no-simple-consistent"
            | "--no-simple-consistent" => {
                simple_consistent = Some(false);
                laminar_simple_option_seen = true;
                index += 1;
            }
            "-pRefCell" | "--pRefCell" | "-p-ref-cell" | "--p-ref-cell" => {
                let value = args
                    .get(index + 1)
                    .ok_or_else(|| "--pRefCell requires a non-negative integer".to_string())?;
                pressure_reference_cell = Some(parse_non_negative_usize_arg("--pRefCell", value)?);
                laminar_simple_option_seen = true;
                index += 2;
            }
            "-pRefValue" | "--pRefValue" | "-p-ref-value" | "--p-ref-value" => {
                let value = args
                    .get(index + 1)
                    .ok_or_else(|| "--pRefValue requires a finite pressure value".to_string())?;
                pressure_reference_value = Some(parse_finite_f64_arg("--pRefValue", value)?);
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
            "-solveVerbose" | "--solveVerbose" | "-solve-verbose" | "--solve-verbose" => {
                solve_verbose = true;
                laminar_simple_option_seen = true;
                index += 1;
            }
            "-solveResidualCsv"
            | "--solveResidualCsv"
            | "-solve-residual-csv"
            | "--solve-residual-csv" => {
                let path = args
                    .get(index + 1)
                    .ok_or_else(|| "--solveResidualCsv requires a file path".to_string())?;
                solve_residual_csv = Some(PathBuf::from(path));
                laminar_simple_option_seen = true;
                index += 2;
            }
            "-solveResidualPlot"
            | "--solveResidualPlot"
            | "-solve-residual-plot"
            | "--solve-residual-plot" => {
                let path = args.get(index + 1).ok_or_else(|| {
                    "--solveResidualPlot requires an output image path".to_string()
                })?;
                solve_residual_plot = Some(PathBuf::from(path));
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
            "-writeFinalFields"
            | "--writeFinalFields"
            | "-write-final-fields"
            | "--write-final-fields" => {
                let path = args
                    .get(index + 1)
                    .ok_or_else(|| "--writeFinalFields requires an output directory".to_string())?;
                write_final_fields = Some(PathBuf::from(path));
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
            pressure_reference_cell,
            pressure_reference_value,
            non_orthogonal_correctors,
            simple_consistent,
            velocity_relaxation,
            pressure_relaxation,
            solve_verbose,
            solve_residual_csv,
            solve_residual_plot,
            report_json: solve_report_json,
            report_markdown: solve_report_markdown,
            write_final_fields,
        })
    } else {
        None
    };
    if poiseuille_solve.is_none() && poiseuille_option_seen {
        return Err("Poiseuille solve options require --solvePoiseuille".to_string());
    }
    if poiseuille_solve.is_none() && laminar_simple_solve.is_none() && shared_flow_option_seen {
        return Err("--mu requires --solvePoiseuille or --solveLaminarSimple".to_string());
    }
    if (scalar_diffusion_solve.is_some() || poiseuille_solve.is_some())
        && let Some(error) = scalar_diffusion_linear_solver_error
    {
        return Err(error);
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
    parse_openfoam_laminar_solver(value)
}

fn parse_laminar_simple_preconditioner(value: &str) -> Result<LaminarSimplePreconditioner, String> {
    parse_openfoam_laminar_preconditioner(value)
}

fn parse_bool_value(value: &str) -> Option<bool> {
    match value.trim_matches(';') {
        "true" | "True" | "TRUE" | "yes" | "Yes" | "YES" | "on" | "On" | "ON" | "1" => Some(true),
        "false" | "False" | "FALSE" | "no" | "No" | "NO" | "off" | "Off" | "OFF" | "0" => {
            Some(false)
        }
        _ => None,
    }
}

fn parse_bool_arg(label: &str, value: &str) -> Result<bool, String> {
    parse_bool_value(value)
        .ok_or_else(|| format!("invalid {label} value '{value}'; expected true/false or yes/no"))
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

fn parse_non_negative_usize_arg(label: &str, value: &str) -> Result<usize, String> {
    value
        .parse::<usize>()
        .map_err(|_| format!("invalid {label} value '{value}'; expected a non-negative integer"))
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
    println!("  ferrum pipeBenchmark -case <caseDir> --fields <timeDir> [reference options]");
    println!(
        "  ferrum planeChannelBenchmark -case <caseDir> --fields <timeDir> [reference options]"
    );
    println!();
    println!("aliases:");
    println!("  initFerrumCase <caseDir> [--region <name> ...] [--force]");
    println!("  gmshToFerrumFoam <mesh.msh> [-case <caseDir>] [patch type options]");
    println!("  checkFerrumMesh [-case <caseDir>]");
    println!("  splitFerrumMeshRegions [-case <caseDir>] [-cellZones]");
    println!("  ferrumSolver [-case <caseDir>] [--preflight] [--planJson <file>] [--runnerDryRun]");
    println!("  ferrumPipeBenchmark -case <caseDir> --fields <timeDir> [reference options]");
    println!(
        "  ferrumPlaneChannelBenchmark -case <caseDir> --fields <timeDir> [reference options]"
    );
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
        "  --linearSolver <s>   cg, pcg, gaussSeidel, symGaussSeidel, bicgstab, or jacobi for executable solves (default: cg; laminar SIMPLE reads fvSolution)"
    );
    println!(
        "  --momentumLinearSolver <s> override laminar SIMPLE momentum solver (bicgstab, gaussSeidel, symGaussSeidel, cg, pcg, or jacobi)"
    );
    println!(
        "  --pressureLinearSolver <s> override laminar SIMPLE pressure solver (bicgstab, gaussSeidel, symGaussSeidel, cg, pcg, or jacobi)"
    );
    println!(
        "  --momentumPreconditioner <s> override laminar SIMPLE U preconditioner (none, diagonal; DIC requires PCG/SPD)"
    );
    println!(
        "  --pressurePreconditioner <s> override laminar SIMPLE p preconditioner (none, diagonal, DIC/incompleteCholesky)"
    );
    println!(
        "  --momentumSolveTolerance <v> override laminar SIMPLE U solve tolerance (default: --solveTolerance, fvSolution solvers.U.tolerance, or OpenFOAM 1e-6)"
    );
    println!(
        "  --pressureSolveTolerance <v> override laminar SIMPLE p solve tolerance (default: --solveTolerance, fvSolution solvers.p.tolerance, or OpenFOAM 1e-6)"
    );
    println!(
        "  --momentumMaxIterations <n> override laminar SIMPLE U linear iteration cap (default: fvSolution or OpenFOAM 1000)"
    );
    println!(
        "  --pressureMaxIterations <n> override laminar SIMPLE p linear iteration cap (default: fvSolution or OpenFOAM 1000)"
    );
    println!("  --pressureDrop <Pa>  pressure drop for --solvePoiseuille");
    println!("  --rho <kg/m3>        density for --solveLaminarSimple");
    println!("  --mu <Pa.s>          dynamic viscosity for --solvePoiseuille/--solveLaminarSimple");
    println!("  --length <m>         pipe length for --solvePoiseuille");
    println!("  --diameter <m>       pipe diameter for --solvePoiseuille");
    println!("  --wallPatch <name>   wall patch for --solvePoiseuille (default: wall)");
    println!(
        "  --maxSimpleIterations <n> override SIMPLE iteration count (default: controlDict endTime/deltaT)"
    );
    println!(
        "  --minSimpleIterations <n> minimum SIMPLE iterations before convergence (default: 1 for one-step runs, otherwise 2)"
    );
    println!(
        "  --nNonOrthogonalCorrectors <n> override SIMPLE nNonOrthogonalCorrectors (default: fvSolution or 0)"
    );
    println!(
        "  --simpleConsistent [bool] enable OpenFOAM-style SIMPLE consistent rAtU correction (default: fvSolution SIMPLE.consistent or false)"
    );
    println!("  --pRefCell <n>       pressure reference cell for closed-pressure cases");
    println!("  --pRefValue <Pa>     pressure reference value (default: fvSolution or 0)");
    println!(
        "  --velocityRelaxation <v> override U relaxation for --solveLaminarSimple (default: fvSolution relaxationFactors.equations.U or no relaxation)"
    );
    println!(
        "  --pressureRelaxation <v> override p relaxation for --solveLaminarSimple (default: fvSolution relaxationFactors.fields.p or no relaxation)"
    );
    println!(
        "  --solveVerbose print per-iteration initial/final residuals and linear/outer convergence"
    );
    println!("  --solveResidualCsv <file> write SIMPLE residual history as CSV");
    println!("  --solveResidualPlot <file> render residual plot image from CSV data");
    println!("  --solveReportJson <file> write --solveLaminarSimple JSON report");
    println!("  --solveReportMarkdown <file> write --solveLaminarSimple Markdown report");
    println!(
        "  --writeFinalFields <dir> write final U and p fields to an OpenFOAM-like time directory"
    );
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

fn print_pipe_benchmark_usage() {
    println!(
        "usage: ferrumPipeBenchmark -case <caseDir> --fields <timeDir> --pressureDrop <Pa> --mu <Pa.s> --length <m> --diameter <m> [options]"
    );
    println!();
    println!("post-processes stored U/p fields outside the generic SIMPLE solver");
    println!("  --axis <x|y|z>       axial velocity component (default: x)");
    println!("  --inletPatch <name>  inlet patch for pressure sampling (default: inlet)");
    println!("  --outletPatch <name> outlet patch for pressure sampling (default: outlet)");
    println!("  --outJson <file>     write benchmark JSON");
    println!("  --outMarkdown <file> write benchmark Markdown");
}

fn print_plane_channel_benchmark_usage() {
    println!(
        "usage: ferrumPlaneChannelBenchmark -case <caseDir> --fields <timeDir> --pressureDrop <Pa> --mu <Pa.s> --length <m> --gap <m> --depth <m> [options]"
    );
    println!();
    println!("post-processes stored U/p fields outside the generic SIMPLE solver");
    println!("  --axis <x|y|z>       axial velocity component (default: x)");
    println!("  --inletPatch <name>  inlet patch for pressure sampling (default: inlet)");
    println!("  --outletPatch <name> outlet patch for pressure sampling (default: outlet)");
    println!("  --pressureScale <v>  multiply stored p before SI comparison (default: 1)");
    println!("  --outJson <file>     write benchmark JSON");
    println!("  --outMarkdown <file> write benchmark Markdown");
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
        ContinuitySummary, LaminarSimpleIterationSummary, LaminarSimpleOptions,
        LaminarSimpleResidualControlSummary, LaminarSimpleSchemes, ScalarDiffusionLinearSolver,
        SolverNumericsDictionaryPlan, estimate_iterations_to_convergence,
        estimate_simple_iterations_to_convergence, numerics_dictionary_number,
        numerics_dictionary_usize, numerics_dictionary_value,
        parse_laminar_simple_convection_scheme, parse_laminar_simple_gradient_scheme,
        parse_laminar_simple_laplacian_scheme, parse_laminar_simple_sn_grad_scheme,
        parse_openfoam_laminar_preconditioner, parse_openfoam_laminar_solver,
        parse_pipe_benchmark_args, parse_plane_channel_benchmark_args, parse_solver_args,
        resolve_laminar_simple_options, validate_laminar_residual_control_dictionary,
        write_json_solver_state, write_json_string,
    };
    use ferrum_mesh::backends::BackendChoice;
    use ferrum_mesh::control::ControlDict;
    use ferrum_mesh::flow::{
        LaminarSimpleConvectionScheme, LaminarSimpleGradientScheme, LaminarSimpleLaplacianScheme,
        LaminarSimpleLinearSolver, LaminarSimplePreconditioner, LaminarSimpleSnGradScheme,
    };
    use ferrum_mesh::poiseuille::PipeAxis;
    use ferrum_mesh::runtime::{SolverRuntimeData, SolverRuntimeMeshData};
    use ferrum_mesh::solver_plan::{
        SolverBackendPlan, SolverCasePlan, SolverCpuResourcePlan, SolverDimensionality,
        SolverFieldPlan, SolverGpuResourcePlan, SolverInterfacePlan, SolverMeshPlan,
        SolverNumericsPlan, SolverPropertiesPlan, SolverPropertyEntryPlan, SolverRunPlan,
    };
    use ferrum_mesh::solver_state::{
        SolverStateCpuBufferPlan, SolverStateCpuBufferStatus, SolverStateFieldKind,
        SolverStateFieldPlan, SolverStateInternalFieldPlan, SolverStatePlan,
        SolverStateStoragePlan, SolverStateStorageStatus, SolverStateValueKind,
    };
    use std::path::PathBuf;

    #[test]
    fn parses_external_pipe_benchmark_options() {
        let args = vec![
            "-case".to_string(),
            "examples/pipe".to_string(),
            "--fields".to_string(),
            "target/fields/100".to_string(),
            "--pressureDrop".to_string(),
            "1.6032".to_string(),
            "--mu".to_string(),
            "0.001002".to_string(),
            "--length".to_string(),
            "1".to_string(),
            "--diameter".to_string(),
            "0.02".to_string(),
            "--axis".to_string(),
            "z".to_string(),
            "--inletPatch".to_string(),
            "feed".to_string(),
            "--outletPatch".to_string(),
            "product".to_string(),
            "--outJson".to_string(),
            "target/pipe.json".to_string(),
        ];

        let parsed = parse_pipe_benchmark_args(&args).expect("pipe benchmark args should parse");

        assert_eq!(parsed.case_dir, PathBuf::from("examples/pipe"));
        assert_eq!(parsed.fields_dir, PathBuf::from("target/fields/100"));
        assert_eq!(parsed.options.pressure_drop, 1.6032);
        assert_eq!(parsed.options.dynamic_viscosity, 0.001002);
        assert_eq!(parsed.options.length, 1.0);
        assert_eq!(parsed.options.diameter, 0.02);
        assert_eq!(parsed.options.axis, PipeAxis::Z);
        assert_eq!(parsed.options.inlet_patch, "feed");
        assert_eq!(parsed.options.outlet_patch, "product");
        assert_eq!(parsed.out_json, Some(PathBuf::from("target/pipe.json")));
    }

    #[test]
    fn parses_external_plane_channel_benchmark_options() {
        let args = vec![
            "-case".to_string(),
            "target/cases/channel".to_string(),
            "--fields".to_string(),
            "target/fields/545".to_string(),
            "--pressureDrop".to_string(),
            "0.6012".to_string(),
            "--mu".to_string(),
            "0.001002".to_string(),
            "--length".to_string(),
            "1".to_string(),
            "--gap".to_string(),
            "0.02".to_string(),
            "--depth".to_string(),
            "0.001".to_string(),
            "--axis".to_string(),
            "x".to_string(),
            "--pressureScale".to_string(),
            "998.2".to_string(),
        ];

        let parsed = parse_plane_channel_benchmark_args(&args)
            .expect("plane-channel benchmark args should parse");

        assert_eq!(parsed.case_dir, PathBuf::from("target/cases/channel"));
        assert_eq!(parsed.fields_dir, PathBuf::from("target/fields/545"));
        assert_eq!(parsed.options.pressure_drop, 0.6012);
        assert_eq!(parsed.options.gap, 0.02);
        assert_eq!(parsed.options.depth, 0.001);
        assert_eq!(parsed.options.axis, PipeAxis::X);
        assert_eq!(parsed.pressure_scale, 998.2);
    }

    #[test]
    fn external_pipe_benchmark_requires_stored_fields() {
        let error = parse_pipe_benchmark_args(&[
            "--pressureDrop".to_string(),
            "1.6032".to_string(),
            "--mu".to_string(),
            "0.001002".to_string(),
            "--length".to_string(),
            "1".to_string(),
            "--diameter".to_string(),
            "0.02".to_string(),
        ])
        .expect_err("pipe post-processing without stored fields must fail");

        assert!(error.contains("requires --fields"));
    }

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
            "--maxSimpleIterations".to_string(),
            "7".to_string(),
            "--minSimpleIterations".to_string(),
            "3".to_string(),
            "--nNonOrthogonalCorrectors".to_string(),
            "2".to_string(),
            "--simpleConsistent".to_string(),
            "--pRefCell".to_string(),
            "12".to_string(),
            "--pRefValue".to_string(),
            "101325".to_string(),
            "--velocityRelaxation".to_string(),
            "0.6".to_string(),
            "--pressureRelaxation".to_string(),
            "0.2".to_string(),
            "--solveReportJson".to_string(),
            "target/simple.json".to_string(),
            "--solveReportMarkdown".to_string(),
            "target/simple.md".to_string(),
            "--writeFinalFields".to_string(),
            "target/simpleFields/1".to_string(),
        ];

        let parsed = parse_solver_args(&args).expect("solver args should parse");
        let solve = parsed
            .laminar_simple_solve
            .expect("laminar SIMPLE solve args");

        assert_eq!(solve.density, Some(998.2));
        assert_eq!(solve.dynamic_viscosity, Some(0.001002));
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
        assert_eq!(solve.max_simple_iterations, Some(7));
        assert_eq!(solve.min_simple_iterations, Some(3));
        assert_eq!(solve.non_orthogonal_correctors, Some(2));
        assert_eq!(solve.simple_consistent, Some(true));
        assert_eq!(solve.pressure_reference_cell, Some(12));
        assert_eq!(solve.pressure_reference_value, Some(101325.0));
        assert_eq!(solve.velocity_relaxation, Some(0.6));
        assert_eq!(solve.pressure_relaxation, Some(0.2));
        assert_eq!(solve.report_json, Some(PathBuf::from("target/simple.json")));
        assert_eq!(
            solve.report_markdown,
            Some(PathBuf::from("target/simple.md"))
        );
        assert_eq!(
            solve.write_final_fields,
            Some(PathBuf::from("target/simpleFields/1"))
        );
    }

    #[test]
    fn parses_laminar_simple_residual_reporting_options() {
        let args = vec![
            "--solveLaminarSimple".to_string(),
            "--solveVerbose".to_string(),
            "--solveResidualCsv".to_string(),
            "target/simple-residuals.csv".to_string(),
            "--solveResidualPlot".to_string(),
            "target/simple-residuals.png".to_string(),
            "--solveReportJson".to_string(),
            "target/simple.json".to_string(),
        ];

        let parsed = parse_solver_args(&args).expect("solver args should parse");
        let solve = parsed
            .laminar_simple_solve
            .expect("laminar SIMPLE solve args");

        assert!(solve.solve_verbose);
        assert_eq!(
            solve.solve_residual_csv,
            Some(PathBuf::from("target/simple-residuals.csv"))
        );
        assert_eq!(
            solve.solve_residual_plot,
            Some(PathBuf::from("target/simple-residuals.png"))
        );
        assert_eq!(solve.report_json, Some(PathBuf::from("target/simple.json")));
    }

    #[test]
    fn parses_laminar_simple_split_linear_solvers() {
        let args = vec![
            "--solveLaminarSimple".to_string(),
            "--linearSolver".to_string(),
            "bicgstab".to_string(),
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

        assert_eq!(
            solve.linear_solver,
            Some(LaminarSimpleLinearSolver::BiCgStab)
        );
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
            Some(LaminarSimplePreconditioner::IncompleteCholesky)
        );
    }

    #[test]
    fn rejects_bicgstab_for_scalar_diffusion_generic_solver() {
        let args = vec![
            "--solveScalarDiffusion".to_string(),
            "T".to_string(),
            "--linearSolver".to_string(),
            "bicgstab".to_string(),
        ];

        let error =
            parse_solver_args(&args).expect_err("bicgstab is not a scalar-diffusion solver");

        assert!(error.contains("expected 'cg' or 'jacobi'"));
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
        assert_eq!(solve.simple_consistent, None);
    }

    #[test]
    fn laminar_simple_resolves_without_pipe_benchmark_inputs() {
        let plan = laminar_simple_test_plan(1000.0, 0.001002);
        let args = vec![
            "--solveLaminarSimple".to_string(),
            "--rho".to_string(),
            "1000".to_string(),
            "--mu".to_string(),
            "0.001002".to_string(),
        ];
        let parsed = parse_solver_args(&args).expect("solver args should parse");
        let solve = parsed
            .laminar_simple_solve
            .expect("laminar SIMPLE solve args");
        let options = resolve_laminar_simple_options(&plan, &solve)
            .expect("laminar options should resolve without pipeBenchmark");

        assert_eq!(options.density, 1000.0);
        assert_eq!(options.dynamic_viscosity, 0.001002);
        assert_eq!(
            options.momentum_linear_solver,
            LaminarSimpleLinearSolver::SymGaussSeidel
        );
        assert_eq!(
            options.pressure_linear_solver,
            LaminarSimpleLinearSolver::Pcg
        );
        assert_eq!(options.max_simple_iterations, 100);
    }

    #[test]
    fn laminar_simple_rejects_pipe_benchmark_cli_options() {
        let error = parse_solver_args(&[
            "--solveLaminarSimple".to_string(),
            "--pressureDrop".to_string(),
            "1.6032".to_string(),
        ])
        .expect_err("pipe benchmark inputs must stay outside the generic SIMPLE solve");

        assert!(error.contains("Poiseuille solve options require --solvePoiseuille"));
    }

    #[test]
    fn laminar_simple_uses_openfoam_ldu_defaults_when_controls_are_absent() {
        let mut plan = laminar_simple_test_plan(1000.0, 0.001002);
        plan.numerics
            .fv_solution
            .entries
            .retain(|entry| !matches!(entry.key.as_str(), "tolerance" | "maxIter"));
        let parsed = parse_solver_args(&["--solveLaminarSimple".to_string()])
            .expect("solver args should parse");
        let solve = parsed
            .laminar_simple_solve
            .expect("laminar SIMPLE solve args");
        let options = resolve_laminar_simple_options(&plan, &solve)
            .expect("OpenFOAM defaults should resolve");

        assert_eq!(options.momentum_linear_tolerance, 1.0e-6);
        assert_eq!(options.pressure_linear_tolerance, 1.0e-6);
        assert_eq!(options.momentum_max_linear_iterations, 1_000);
        assert_eq!(options.pressure_max_linear_iterations, 1_000);
    }

    #[test]
    fn laminar_simple_rejects_unimplemented_nonzero_relative_tolerance() {
        let mut plan = laminar_simple_test_plan(1000.0, 0.001002);
        plan.numerics
            .fv_solution
            .entries
            .push(ferrum_mesh::solver_plan::SolverNumericsEntryPlan {
                section: "solvers.U".to_string(),
                key: "relTol".to_string(),
                value: "0.1".to_string(),
            });
        let parsed = parse_solver_args(&["--solveLaminarSimple".to_string()])
            .expect("solver args should parse");
        let solve = parsed
            .laminar_simple_solve
            .expect("laminar SIMPLE solve args");
        let error = resolve_laminar_simple_options(&plan, &solve)
            .expect_err("non-zero relTol must not be ignored");

        assert!(error.contains("relTol=0.1 is not implemented"));
    }

    #[test]
    fn laminar_simple_ignores_case_level_benchmark_toggles() {
        let mut plan = laminar_simple_test_plan(1000.0, 0.001002);
        plan.numerics.fv_solution.present = true;
        plan.numerics.fv_solution.entries.extend([
            ferrum_mesh::solver_plan::SolverNumericsEntryPlan {
                section: "SIMPLE".to_string(),
                key: "pressureDropTolerance".to_string(),
                value: "0.001".to_string(),
            },
            ferrum_mesh::solver_plan::SolverNumericsEntryPlan {
                section: "SIMPLE".to_string(),
                key: "fieldChangeTolerance".to_string(),
                value: "0.001".to_string(),
            },
            ferrum_mesh::solver_plan::SolverNumericsEntryPlan {
                section: "SIMPLE".to_string(),
                key: "benchmarkConvergence".to_string(),
                value: "true".to_string(),
            },
            ferrum_mesh::solver_plan::SolverNumericsEntryPlan {
                section: "SIMPLE".to_string(),
                key: "minSimpleIterations".to_string(),
                value: "3".to_string(),
            },
        ]);

        let args = vec!["--solveLaminarSimple".to_string()];
        let parsed = parse_solver_args(&args).expect("solver args should parse");
        let solve = parsed
            .laminar_simple_solve
            .expect("laminar SIMPLE solve args");
        let options =
            resolve_laminar_simple_options(&plan, &solve).expect("laminar options should resolve");

        assert_eq!(options.min_simple_iterations, 3);
    }

    fn minimal_laminar_simple_options_for_estimate() -> LaminarSimpleOptions {
        LaminarSimpleOptions {
            density: 1000.0,
            dynamic_viscosity: 0.001,
            linear_solver: LaminarSimpleLinearSolver::Cg,
            momentum_linear_solver: LaminarSimpleLinearSolver::Cg,
            pressure_linear_solver: LaminarSimpleLinearSolver::Cg,
            momentum_preconditioner: LaminarSimplePreconditioner::None,
            pressure_preconditioner: LaminarSimplePreconditioner::None,
            linear_tolerance: 1.0e-10,
            max_linear_iterations: 10_000,
            momentum_linear_tolerance: 1.0e-10,
            pressure_linear_tolerance: 1.0e-10,
            momentum_max_linear_iterations: 10_000,
            pressure_max_linear_iterations: 10_000,
            max_simple_iterations: 100,
            min_simple_iterations: 1,
            momentum_residual_control: Some(1.0e-5),
            pressure_residual_control: Some(1.0e-5),
            pressure_reference_cell: None,
            pressure_reference_value: 0.0,
            non_orthogonal_correctors: 0,
            simple_consistent: false,
            velocity_relaxation: 0.7,
            pressure_relaxation: 0.3,
            schemes: LaminarSimpleSchemes::default(),
        }
    }

    fn build_laminar_simple_iteration_summary(
        iteration: usize,
        continuity_after: f64,
        momentum_norm: f64,
        pressure_norm: f64,
    ) -> LaminarSimpleIterationSummary {
        LaminarSimpleIterationSummary {
            iteration,
            continuity_before: ContinuitySummary {
                l2_norm: continuity_after * 2.0,
                ..ContinuitySummary::default()
            },
            continuity_after: ContinuitySummary {
                l2_norm: continuity_after,
                ..ContinuitySummary::default()
            },
            pressure_correction_accepted: true,
            momentum_linear_iterations: 0,
            momentum_linear_converged: true,
            momentum_component_linear_converged: [true, true, true],
            pressure_linear_iterations: 0,
            pressure_linear_converged: true,
            pressure_linear_solves: 0,
            pressure_linear_non_converged_solves: 0,
            momentum_initial_normalized_residual_norm: momentum_norm,
            momentum_residual_norm: 0.0,
            momentum_normalized_residual_norm: momentum_norm,
            momentum_component_initial_normalized_residual_norms: [momentum_norm, 0.0, 0.0],
            momentum_component_residual_norms: [0.0, 0.0, 0.0],
            momentum_component_normalized_residual_norms: [0.0, 0.0, 0.0],
            momentum_diagonal_min: 0.0,
            momentum_diagonal_max: 0.0,
            momentum_h1_min: 0.0,
            momentum_h1_max: 0.0,
            pressure_correction_initial_normalized_residual_norm: pressure_norm,
            pressure_correction_residual_norm: 0.0,
            pressure_correction_normalized_residual_norm: pressure_norm,
            residual_control: LaminarSimpleResidualControlSummary::default(),
            relative_velocity_change_l2: 0.0,
            relative_pressure_change_l2: 0.0,
            momentum_update_scale: 0.0,
            pressure_correction_update_scale: 0.0,
            adjust_phi_global_flux_before: 0.0,
            adjust_phi_global_flux_after: 0.0,
            adjust_phi_adjusted_faces: 0,
        }
    }

    #[test]
    fn estimates_additional_simple_iterations_from_monotone_history() {
        let options = minimal_laminar_simple_options_for_estimate();
        let history = vec![
            build_laminar_simple_iteration_summary(1, 1.0, 1.0e-2, 1.0e-2),
            build_laminar_simple_iteration_summary(2, 0.5, 5.0e-3, 5.0e-3),
            build_laminar_simple_iteration_summary(3, 0.25, 2.5e-3, 2.5e-3),
        ];

        let estimate = estimate_simple_iterations_to_convergence(&history, &options)
            .expect("iteration estimate should be available");

        assert!(estimate.additional_iterations > 0);
    }

    #[test]
    fn estimates_are_not_available_without_decay() {
        let options = minimal_laminar_simple_options_for_estimate();
        let history = vec![
            build_laminar_simple_iteration_summary(1, 1.0, 1.0e-2, 1.0e-2),
            build_laminar_simple_iteration_summary(2, 1.1, 2.0e-2, 2.0e-2),
            build_laminar_simple_iteration_summary(3, 1.2, 2.1e-2, 2.1e-2),
        ];

        assert!(estimate_simple_iterations_to_convergence(&history, &options).is_none());
    }

    #[test]
    fn estimate_iterations_to_convergence_requires_decay() {
        let history = vec![1.0e-2, 5.0e-3, 2.5e-3];
        let estimate = estimate_iterations_to_convergence(&history, 1.0e-5)
            .expect("estimate should exist for geometric decay");

        assert_eq!(estimate.geometric_ratio, 0.5);
        assert!(estimate.additional_iterations >= 1);
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
            parse_openfoam_laminar_solver(
                numerics_dictionary_value(&dictionary, "solvers.p", "solver").unwrap()
            )
            .unwrap(),
            LaminarSimpleLinearSolver::Pcg
        );
        assert!(parse_openfoam_laminar_solver("smoothSolver").is_err());
        assert_eq!(
            parse_openfoam_laminar_solver("symGaussSeidel").unwrap(),
            LaminarSimpleLinearSolver::SymGaussSeidel
        );
        assert_eq!(
            parse_openfoam_laminar_preconditioner(
                numerics_dictionary_value(&dictionary, "solvers.p", "preconditioner").unwrap()
            )
            .unwrap(),
            LaminarSimplePreconditioner::IncompleteCholesky
        );
        assert!(parse_openfoam_laminar_preconditioner("DILU").is_err());
    }

    #[test]
    fn rejects_dictionary_form_for_simple_residual_control() {
        let dictionary = SolverNumericsDictionaryPlan {
            present: true,
            sections: vec![ferrum_mesh::solver_plan::SolverNumericsSectionPlan {
                path: "SIMPLE.residualControl.U".to_string(),
                entries: 2,
            }],
            entries: Vec::new(),
        };

        let error = validate_laminar_residual_control_dictionary(&dictionary)
            .expect_err("nested residualControl must fail");

        assert!(error.contains("single scalar"));
        assert!(error.contains("SIMPLE.residualControl.U"));
    }

    #[test]
    fn rejects_unsolved_fields_in_simple_residual_control() {
        let dictionary = SolverNumericsDictionaryPlan {
            present: true,
            sections: Vec::new(),
            entries: vec![ferrum_mesh::solver_plan::SolverNumericsEntryPlan {
                section: "SIMPLE.residualControl".to_string(),
                key: "T".to_string(),
                value: "1e-5".to_string(),
            }],
        };

        let error = validate_laminar_residual_control_dictionary(&dictionary)
            .expect_err("unsolved residualControl field must fail");

        assert!(error.contains("supported solved fields are U and p"));
    }

    fn laminar_simple_test_plan(density: f64, dynamic_viscosity: f64) -> SolverCasePlan {
        SolverCasePlan {
            case_dir: PathBuf::from("case"),
            control: ControlDict {
                path: PathBuf::from("controlDict"),
                application: "ferrumSolver".to_string(),
                start_from: "startTime".to_string(),
                start_time: Some(0.0),
                stop_at: "endTime".to_string(),
                end_time: Some(100.0),
                delta_t: Some(1.0),
                write_control: "timeStep".to_string(),
                write_interval: None,
            },
            mesh: SolverMeshPlan {
                points: 0,
                cells: 0,
                faces: 0,
                internal_faces: 0,
                boundary_faces: 0,
                patches: 0,
                empty_patches: 0,
                wedge_patches: 0,
                symmetry_patches: 0,
                dimensionality: SolverDimensionality::ThreeD,
                region_meshes: Vec::new(),
            },
            fields: SolverFieldPlan { fields: Vec::new() },
            state: ferrum_mesh::solver_state::SolverStatePlan {
                fields: Vec::new(),
                warnings: Vec::new(),
            },
            runtime_data: SolverRuntimeData {
                mesh: SolverRuntimeMeshData {
                    points: 0,
                    cells: 0,
                    faces: 0,
                    internal_faces: 0,
                    boundary_faces: 0,
                    owner: Vec::new(),
                    neighbour: Vec::new(),
                    patches: Vec::new(),
                    face_centres: Vec::new(),
                    face_area_vectors: Vec::new(),
                    cell_centres: Vec::new(),
                    cell_volumes: Vec::new(),
                    min_face_area: 0.0,
                    max_face_area: 0.0,
                    min_cell_volume: 0.0,
                    max_cell_volume: 0.0,
                    total_cell_volume: 0.0,
                    non_positive_cell_volumes: 0,
                },
                fields: Vec::new(),
                warnings: Vec::new(),
            },
            properties: SolverPropertiesPlan {
                dictionaries: Vec::new(),
                entries: vec![
                    SolverPropertyEntryPlan {
                        dictionary: "transportProperties".to_string(),
                        section: None,
                        key: "rho".to_string(),
                        value: format!("{density}"),
                    },
                    SolverPropertyEntryPlan {
                        dictionary: "transportProperties".to_string(),
                        section: None,
                        key: "mu".to_string(),
                        value: format!("{dynamic_viscosity}"),
                    },
                ],
            },
            numerics: SolverNumericsPlan {
                fv_schemes: SolverNumericsDictionaryPlan {
                    present: true,
                    sections: Vec::new(),
                    entries: vec![
                        ferrum_mesh::solver_plan::SolverNumericsEntryPlan {
                            section: "gradSchemes".to_string(),
                            key: "default".to_string(),
                            value: "Gauss linear".to_string(),
                        },
                        ferrum_mesh::solver_plan::SolverNumericsEntryPlan {
                            section: "divSchemes".to_string(),
                            key: "div(phi,U)".to_string(),
                            value: "Gauss upwind".to_string(),
                        },
                        ferrum_mesh::solver_plan::SolverNumericsEntryPlan {
                            section: "laplacianSchemes".to_string(),
                            key: "default".to_string(),
                            value: "Gauss linear corrected".to_string(),
                        },
                        ferrum_mesh::solver_plan::SolverNumericsEntryPlan {
                            section: "interpolationSchemes".to_string(),
                            key: "default".to_string(),
                            value: "linear".to_string(),
                        },
                        ferrum_mesh::solver_plan::SolverNumericsEntryPlan {
                            section: "snGradSchemes".to_string(),
                            key: "default".to_string(),
                            value: "corrected".to_string(),
                        },
                    ],
                },
                fv_solution: SolverNumericsDictionaryPlan {
                    present: true,
                    sections: Vec::new(),
                    entries: vec![
                        ferrum_mesh::solver_plan::SolverNumericsEntryPlan {
                            section: "solvers.U".to_string(),
                            key: "solver".to_string(),
                            value: "smoothSolver".to_string(),
                        },
                        ferrum_mesh::solver_plan::SolverNumericsEntryPlan {
                            section: "solvers.U".to_string(),
                            key: "smoother".to_string(),
                            value: "symGaussSeidel".to_string(),
                        },
                        ferrum_mesh::solver_plan::SolverNumericsEntryPlan {
                            section: "solvers.U".to_string(),
                            key: "tolerance".to_string(),
                            value: "1e-10".to_string(),
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
                            key: "tolerance".to_string(),
                            value: "1e-10".to_string(),
                        },
                    ],
                },
            },
            interfaces: SolverInterfacePlan {
                registry_available: false,
                discovered_interfaces: 0,
                boundary_face_zones: 0,
                config_present: false,
                configured_interfaces: 0,
            },
            backends: SolverBackendPlan {
                config_present: false,
                default: BackendChoice::Cpu,
                uses_cpu: true,
                uses_gpu: true,
                mixed_execution: true,
                cpu: SolverCpuResourcePlan {
                    cpus: "auto".to_string(),
                    cores_per_cpu: "auto".to_string(),
                    threads: "auto".to_string(),
                    thread_pinning: "off".to_string(),
                    numa: "auto".to_string(),
                },
                gpu: SolverGpuResourcePlan {
                    backend: "auto".to_string(),
                    devices: vec!["auto".to_string()],
                    multi_gpu: "auto".to_string(),
                    precision: "f64".to_string(),
                },
                stages: Vec::new(),
            },
            run: SolverRunPlan {
                stop_at: "endTime".to_string(),
                start_time: Some(0.0),
                end_time: Some(100.0),
                delta_t: Some(1.0),
                estimated_steps: Some(100),
                write_control: "timeStep".to_string(),
                write_interval: None,
                estimated_write_events: None,
                stages: Vec::new(),
            },
            warnings: Vec::new(),
        }
    }

    #[test]
    fn parses_laminar_simple_fv_schemes_subset() {
        assert_eq!(
            parse_laminar_simple_gradient_scheme("Gauss linear").expect("grad scheme"),
            LaminarSimpleGradientScheme::GaussLinear
        );
        assert_eq!(
            parse_laminar_simple_convection_scheme("Gauss upwind").expect("upwind scheme"),
            LaminarSimpleConvectionScheme::GaussUpwind
        );
        assert_eq!(
            parse_laminar_simple_convection_scheme("Gauss linearUpwind grad(U)")
                .expect("linearUpwind scheme"),
            LaminarSimpleConvectionScheme::GaussLinearUpwind
        );
        assert_eq!(
            parse_laminar_simple_sn_grad_scheme("corrected").expect("snGrad scheme"),
            LaminarSimpleSnGradScheme::Corrected
        );
        assert_eq!(
            parse_laminar_simple_laplacian_scheme(
                "Gauss linear corrected",
                LaminarSimpleSnGradScheme::Orthogonal,
            )
            .expect("laplacian scheme"),
            LaminarSimpleLaplacianScheme::GaussLinearCorrected
        );
        assert!(
            parse_laminar_simple_convection_scheme("none")
                .expect_err("default none must not be executable")
                .contains("divSchemes.div(phi,U)")
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
