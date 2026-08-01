mod case;

use std::collections::{BTreeMap, btree_map::Entry};
use std::env;
use std::fmt;
use std::fs::File;
use std::io::{BufWriter, Error, ErrorKind, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use case::{InitCaseOptions, init_case};
use ferrum_finite_volume::boundary_forces::{
    CellVelocityGradientComponents, MAX_RELATIVE_AREA_VECTOR_IMBALANCE_TOLERANCE,
    NoSlipWallForceOptions, PressureFieldKind, PressureReference, ReferenceArea,
    WallFaceForceContribution, WallForceCoefficients,
    integrate_stationary_no_slip_zero_gradient_pressure_wall_forces_with_reconstructed_gradient,
};
use ferrum_finite_volume::continuity::{NormalizedContinuity, normalize_steady_continuity};
use ferrum_mesh::Point3;
use ferrum_mesh::backends::{
    read_backend_config, validate_backend_policy, validate_backend_resources,
};
use ferrum_mesh::check::read_case_summary;
use ferrum_mesh::control::{ControlDict, read_control_dict};
use ferrum_mesh::diffusion::{
    assemble_scalar_diffusion_system, diffusion_assembly_capabilities,
    scalar_diffusion_options_from_field,
};
use ferrum_mesh::fields::{
    FieldBoundaryValidationSummary, FieldFile, FieldLoadPolicy, FieldValueSummary, InitialFieldSet,
    read_initial_fields_with_policy, validate_initial_field_boundaries,
};
use ferrum_mesh::flow::{
    AlgebraicResidualRowDiagnostic, ContinuitySummary, FaceFluxDiagnosticSummary,
    FlowBoundarySummary, FlowOperatorSummary, IncidentFaceDiagnosticSummary,
    LaminarSimpleConvectionScheme, LaminarSimpleFieldSummary, LaminarSimpleGradientScheme,
    LaminarSimpleInterpolationScheme, LaminarSimpleIterationSummary, LaminarSimpleLaplacianScheme,
    LaminarSimpleLinearSolver, LaminarSimpleMomentumExecution, LaminarSimpleOptions,
    LaminarSimplePreconditioner, LaminarSimpleReport, LaminarSimpleResidualControlSummary,
    LaminarSimpleSchemes, LaminarSimpleSnGradScheme, LaminarSimpleStopReason,
    LinearNonConvergenceSnapshot, LinearSolveConvergenceSummary, LinearSolveSummary,
    MatrixDiagnosticSummary, PressureAssemblyDiagnostics, ScalarDiagnosticSummary,
    VectorDiagnosticSummary, reconstruct_laminar_gradients_from_fields, solve_laminar_simple,
    solve_laminar_simple_profiled_pcg, solve_laminar_simple_profiled_pcg_with_observer,
    solve_laminar_simple_with_observer,
};
use ferrum_mesh::foam::{FoamWriteOptions, write_openfoam_case_with_options};
use ferrum_mesh::geometry::{GeometrySummary, summarize_case_geometry};
use ferrum_mesh::gmsh::read_msh22_ascii;
use ferrum_mesh::interfaces::{read_interface_config, validate_interface_config};
use ferrum_mesh::linear::{
    ConjugateGradientOptions, GamgAgglomerator, GamgKernelTiming, GamgOptions, GamgOuterSolver,
    GamgSmoother, JacobiOptions, conjugate_gradient_solve, jacobi_solve,
    linear_solver_capabilities,
};
use ferrum_mesh::patches::{PatchValidationSummary, validate_case_patches};
use ferrum_mesh::regions::{
    InterfaceRegistrySummary, InterfaceSummary, build_interface_registry,
    read_region_mesh_summaries, split_regions_by_cell_zones,
};
use ferrum_mesh::runner::{
    MAX_RUNNER_DRY_RUN_STEPS, SolverRunnerDryRun, SolverRunnerDryRunEvent,
    SolverRunnerDryRunOptions, build_solver_runner_dry_run,
};
use ferrum_mesh::runtime::{SolverRuntimeData, SolverRuntimeMeshData};
use ferrum_mesh::safe_output::{SafeOutputEntry, SafeOutputRoot};
use ferrum_mesh::solver_plan::{
    SolverBackendPlan, SolverCasePlan, SolverFieldPlan, SolverInterfacePlan, SolverMeshPlan,
    SolverNumericsDictionaryPlan, SolverNumericsPlan, SolverPropertiesPlan, SolverRunPlan,
    build_solver_case_plan_with_policy,
};
use ferrum_mesh::solver_state::SolverStatePlan;

const FERRUM_DEFAULT_LDU_TOLERANCE: f64 = 1.0e-6;
const FERRUM_DEFAULT_LDU_MAX_ITERATIONS: usize = 1_000;
const FERRUM_MAX_CASE_LDU_MAX_ITERATIONS: usize = FERRUM_DEFAULT_LDU_MAX_ITERATIONS;
const FERRUM_MAX_CASE_SIMPLE_ITERATIONS: usize = 10_000;
const WALL_FORCE_TRACTION_METHOD: &str = "reconstructedGradientFullDeviatoric";
const WALL_FORCE_TRACTION_METHOD_VERSION: usize = 1;
const WALL_FORCE_FORCE_CONVENTION: &str = "fluidOnBody";
const WALL_FORCE_AREA_VECTOR_ORIENTATION: &str = "outwardFromFluid";
const WALL_FORCE_PRESSURE_FACE_TREATMENT: &str = "zeroGradientOwner";
const WALL_FACE_LOADS_CSV_SCHEMA_VERSION: usize = 1;
const WALL_FORCE_PRESSURE_FIELD_KIND: &str = "kinematic";
const WALL_FORCE_PRESSURE_REFERENCE_MODE: &str = "areaVectorBalancedMean";

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
    GmshToFerrum,
    CheckFerrumMesh,
    SplitFerrumMeshRegions,
    InitFerrumCase,
    FerrumRun,
}

enum CommandMode {
    Ferrum,
    Alias(Alias),
}

fn run_command(mode: CommandMode, args: Vec<String>) -> Result<(), String> {
    match mode {
        CommandMode::Ferrum => run_ferrum_subcommand(args),
        CommandMode::Alias(Alias::GmshToFerrum) => gmsh_to_ferrum(args),
        CommandMode::Alias(Alias::CheckFerrumMesh) => check_mesh(args),
        CommandMode::Alias(Alias::SplitFerrumMeshRegions) => split_mesh_regions(args),
        CommandMode::Alias(Alias::InitFerrumCase) => init_case_command(args),
        CommandMode::Alias(Alias::FerrumRun) => run_solver_module(args),
    }
}

fn run_ferrum_subcommand(mut args: Vec<String>) -> Result<(), String> {
    if args.is_empty() || is_help(&args[0]) {
        print_help();
        return Ok(());
    }

    let command = args.remove(0);
    match command.as_str() {
        "gmshToFerrum" => gmsh_to_ferrum(args),
        "checkFerrumMesh" => check_mesh(args),
        "splitFerrumMeshRegions" => split_mesh_regions(args),
        "initFerrumCase" => init_case_command(args),
        "run" | "ferrumRun" => run_solver_module(args),
        "solve" => solve_case(args),
        "gmshToFoam" | "gmshToFerrumFoam" => Err(format!(
            "command '{command}' was replaced by 'gmshToFerrum'"
        )),
        "checkMesh" => Err("command 'checkMesh' was replaced by 'checkFerrumMesh'".to_string()),
        "splitMeshRegions" => {
            Err("command 'splitMeshRegions' was replaced by 'splitFerrumMeshRegions'".to_string())
        }
        "initCase" => Err("command 'initCase' was replaced by 'initFerrumCase'".to_string()),
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

fn gmsh_to_ferrum(args: Vec<String>) -> Result<(), String> {
    if args.is_empty() || args.iter().any(|arg| is_help(arg)) {
        print_gmsh_to_ferrum_usage();
        return Ok(());
    }

    let import = parse_gmsh_to_ferrum_args(&args)?;

    println!("Reading Gmsh mesh: {}", import.mesh_path.display());
    let mesh = read_msh22_ascii(&import.mesh_path).map_err(|error| error.to_string())?;
    println!(
        "Loaded {} points, {} supported volume cells, {} supported boundary faces",
        mesh.points.len(),
        mesh.cells.len(),
        mesh.boundary_faces.len()
    );

    println!("Writing Ferrum case: {}", import.case_dir.display());
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
    let fields = read_initial_fields_with_policy(&case_dir, FieldLoadPolicy::Summary)
        .map_err(|error| error.to_string())?;
    print_initial_fields(&fields);
    print_field_boundary_validation(&case_dir, &fields)?;
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

#[derive(Debug)]
struct FerrumRunArgs {
    solver: Option<String>,
    forwarded_args: Vec<String>,
    execute: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SolverSelectionSource {
    Cli,
    ControlDict,
}

impl std::fmt::Display for SolverSelectionSource {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cli => formatter.write_str("cli"),
            Self::ControlDict => formatter.write_str("controlDict"),
        }
    }
}

#[derive(Debug)]
struct SolverDispatch {
    module: String,
    source: SolverSelectionSource,
}

fn run_solver_module(args: Vec<String>) -> Result<(), String> {
    if args.iter().any(|arg| is_help(arg)) {
        print_ferrum_run_usage();
        return Ok(());
    }

    let FerrumRunArgs {
        solver,
        forwarded_args,
        execute,
    } = parse_ferrum_run_args(&args)?;

    let dispatch = match solver {
        Some(solver) => resolve_solver_dispatch(Some(solver), None)?,
        None => {
            let case_dir = parse_case_dir(&forwarded_args, PathBuf::from("."))?;
            let control = read_control_dict(&case_dir).map_err(|error| error.to_string())?;
            resolve_solver_dispatch(None, Some(&control))?
        }
    };

    solve_case_with_contract(
        forwarded_args,
        execute.then_some(("incompressibleFluid", "SIMPLE")),
        Some(dispatch),
        if execute {
            SolverInvocation::IncompressibleFluidExecute
        } else {
            SolverInvocation::IncompressibleFluidPlanOnly
        },
    )
}

fn resolve_solver_dispatch(
    cli_solver: Option<String>,
    control: Option<&ControlDict>,
) -> Result<SolverDispatch, String> {
    let (module, source) = if let Some(module) = cli_solver {
        (module, SolverSelectionSource::Cli)
    } else {
        let control = control.ok_or_else(|| "solver selection is missing".to_string())?;
        if control.application.as_deref() != Some("ferrumRun") {
            return Err(
                "controlDict solver fallback requires the explicit Ferrum marker 'application ferrumRun;'; pass '-solver incompressibleFluid' only for an intentional interoperability run"
                    .to_string(),
            );
        }
        let module = control.solver.clone().ok_or_else(|| {
            "solver not specified; add 'solver incompressibleFluid;' to system/controlDict or pass '-solver incompressibleFluid'"
                .to_string()
        })?;
        (module, SolverSelectionSource::ControlDict)
    };

    if module != "incompressibleFluid" {
        return Err(format!(
            "unknown Ferrum solver module '{module}'; the first executable module is 'incompressibleFluid'"
        ));
    }

    Ok(SolverDispatch { module, source })
}

fn parse_ferrum_run_args(args: &[String]) -> Result<FerrumRunArgs, String> {
    let mut solver = None;
    let mut forwarded_args = Vec::new();
    let mut index = 0;

    while index < args.len() {
        let arg = &args[index];
        if matches!(arg.as_str(), "-solver" | "--solver") {
            if solver.is_some() {
                return Err("-solver may be specified only once".to_string());
            }
            let value = args
                .get(index + 1)
                .filter(|value| !value.starts_with('-'))
                .ok_or_else(|| "-solver requires a module name".to_string())?;
            solver = Some(value.to_string());
            index += 2;
            continue;
        }
        if let Some(value) = arg.strip_prefix("--solver=") {
            if solver.is_some() {
                return Err("-solver may be specified only once".to_string());
            }
            if value.is_empty() {
                return Err("--solver requires a module name".to_string());
            }
            solver = Some(value.to_string());
            index += 1;
            continue;
        }
        if is_utility_execution_selector(arg) {
            return Err(format!(
                "'{arg}' is a developer utility selector; use it through 'ferrum solve', not ferrumRun"
            ));
        }

        forwarded_args.push(arg.clone());
        index += 1;
    }

    let execute = !forwarded_args.iter().any(|arg| {
        matches!(
            arg.as_str(),
            "-preflight"
                | "--preflight"
                | "-dryRun"
                | "--dry-run"
                | "-runnerDryRun"
                | "--runnerDryRun"
                | "-runner-dry-run"
                | "--runner-dry-run"
        )
    });

    Ok(FerrumRunArgs {
        solver,
        forwarded_args,
        execute,
    })
}

fn is_utility_execution_selector(arg: &str) -> bool {
    matches!(
        arg,
        "-solveScalarDiffusion"
            | "--solveScalarDiffusion"
            | "-solve-scalar-diffusion"
            | "--solve-scalar-diffusion"
    )
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SolverInvocation {
    Utility,
    IncompressibleFluidPlanOnly,
    IncompressibleFluidExecute,
}

fn solve_case(args: Vec<String>) -> Result<(), String> {
    solve_case_with_contract(args, None, None, SolverInvocation::Utility)
}

fn solve_case_with_contract(
    args: Vec<String>,
    required_module_section: Option<(&str, &str)>,
    dispatch: Option<SolverDispatch>,
    invocation: SolverInvocation,
) -> Result<(), String> {
    if args.iter().any(|arg| is_help(arg)) {
        print_solver_usage();
        return Ok(());
    }

    let options = match invocation {
        SolverInvocation::Utility => parse_solver_args(&args)?,
        SolverInvocation::IncompressibleFluidPlanOnly => {
            parse_incompressible_fluid_plan_args(&args)?
        }
        SolverInvocation::IncompressibleFluidExecute => parse_incompressible_fluid_args(&args)?,
    };
    if options.scalar_diffusion_solve.is_some() && options.laminar_simple_solve.is_some() {
        return Err(
            "scalar diffusion and incompressibleFluid solves cannot share one-shot initial field buffers in one invocation"
                .to_string(),
        );
    }
    let field_policy =
        if options.scalar_diffusion_solve.is_some() || options.laminar_simple_solve.is_some() {
            FieldLoadPolicy::Full
        } else {
            FieldLoadPolicy::Summary
        };
    let mut plan = build_solver_case_plan_with_policy(&options.case_dir, field_policy)
        .map_err(|error| error.to_string())?;
    if let Some((module, required_section)) = required_module_section {
        validate_module_execution_contract(&plan, module, required_section)?;
    }
    print_solver_case_plan(&plan, dispatch.as_ref());
    if options.runner_dry_run {
        let dry_run = build_solver_runner_dry_run(
            &plan,
            SolverRunnerDryRunOptions {
                max_steps: options.max_runner_steps,
            },
        )
        .map_err(|error| error.to_string())?;
        print_solver_runner_dry_run(&dry_run);
    }
    if let Some(solve) = &options.scalar_diffusion_solve {
        run_scalar_diffusion_solve(&mut plan, solve)?;
    }
    if let Some(solve) = &options.laminar_simple_solve {
        run_laminar_simple_solve(&mut plan, solve)?;
    }
    if let Some(path) = options.plan_json {
        write_solver_plan_json(&plan, dispatch.as_ref(), &path).map_err(|error| {
            format!(
                "could not write solver plan JSON to {} ({error})",
                path.display()
            )
        })?;
        println!("wrote solver plan json: {}", path.display());
    }
    Ok(())
}

fn validate_module_execution_contract(
    plan: &SolverCasePlan,
    module: &str,
    required_section: &str,
) -> Result<(), String> {
    if module != "incompressibleFluid" || required_section != "SIMPLE" {
        return Err(format!(
            "no executable contract is registered for module '{module}' and algorithm section '{required_section}'"
        ));
    }

    let simple_sections = plan
        .numerics
        .fv_solution
        .sections
        .iter()
        .filter(|section| section.path == "SIMPLE")
        .count();
    if simple_sections != 1 {
        return Err(format!(
            "solver module 'incompressibleFluid' currently requires exactly one 'SIMPLE' section in system/fvSolution, found {simple_sections}"
        ));
    }

    let conflicting_algorithms = plan
        .numerics
        .fv_solution
        .sections
        .iter()
        .filter(|section| matches!(section.path.as_str(), "PISO" | "PIMPLE"))
        .map(|section| section.path.as_str())
        .collect::<Vec<_>>();
    if !conflicting_algorithms.is_empty() {
        return Err(format!(
            "solver module 'incompressibleFluid' cannot execute the current SIMPLE kernel while {} is also configured; PISO/PIMPLE execution is not implemented yet",
            conflicting_algorithms.join(" and ")
        ));
    }

    let ddt_schemes = plan
        .numerics
        .fv_schemes
        .entries
        .iter()
        .filter(|entry| entry.section == "ddtSchemes" && entry.key == "default")
        .map(|entry| entry.value.as_str())
        .collect::<Vec<_>>();
    if ddt_schemes.len() != 1 {
        return Err(format!(
            "solver module 'incompressibleFluid' requires exactly one ddtSchemes.default entry, found {}",
            ddt_schemes.len()
        ));
    }
    if ddt_schemes[0] != "steadyState" {
        return Err(format!(
            "solver module 'incompressibleFluid' currently executes SIMPLE only with ddtSchemes.default=steadyState, found {}; transient PISO/PIMPLE execution is not implemented yet",
            ddt_schemes[0]
        ));
    }

    validate_laminar_transport_regime(plan)?;

    Ok(())
}

fn validate_laminar_transport_regime(plan: &SolverCasePlan) -> Result<(), String> {
    let regime_dictionaries = plan
        .properties
        .dictionaries
        .iter()
        .filter(|dictionary| {
            dictionary.region.is_none()
                && matches!(
                    dictionary.name.as_str(),
                    "momentumTransport" | "turbulenceProperties"
                )
        })
        .map(|dictionary| dictionary.name.as_str())
        .collect::<Vec<_>>();

    if regime_dictionaries.is_empty() {
        // Ferrum's current compatibility cases predate momentumTransport and
        // are explicitly treated as laminar until FerrumFile v1 is available.
        return Ok(());
    }
    if regime_dictionaries.len() != 1 {
        return Err(format!(
            "laminar incompressible execution requires one transport-regime dictionary, found {}",
            regime_dictionaries.join(" and ")
        ));
    }

    let dictionary = regime_dictionaries[0];
    let simulation_types = plan
        .properties
        .entries
        .iter()
        .filter(|entry| {
            entry.dictionary == dictionary
                && entry.section.is_none()
                && entry.key == "simulationType"
        })
        .map(|entry| entry.value.as_str())
        .collect::<Vec<_>>();
    if simulation_types.len() != 1 {
        return Err(format!(
            "constant/{dictionary} requires exactly one top-level simulationType for laminar execution, found {}",
            simulation_types.len()
        ));
    }
    if simulation_types[0] != "laminar" {
        return Err(format!(
            "the current incompressibleFluid kernel requires simulationType laminar, found '{}' in constant/{dictionary}; RAS/LES execution is not implemented yet",
            simulation_types[0]
        ));
    }

    Ok(())
}

fn run_laminar_simple_solve(
    plan: &mut SolverCasePlan,
    solve: &LaminarSimpleSolveArgs,
) -> Result<(), String> {
    let options = resolve_laminar_simple_options(plan, solve)?;
    validate_wall_face_loads_output_destination(solve)?;
    if let Some(wall_forces) = &solve.wall_forces {
        validate_wall_force_boundary_conditions(&plan.initial_fields, &wall_forces.patch_names)?;
    }

    let started = Instant::now();
    let report = if solve.solve_verbose {
        let mut printed_header = false;
        let mut print_iteration = |item: &LaminarSimpleIterationSummary| {
            if !printed_header {
                println!(
                    "incompressibleFluid residual history (Ferrum initial/final residuals; linear and outer convergence are separate):"
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
        if solve.profile_pcg {
            solve_laminar_simple_profiled_pcg_with_observer(
                &mut plan.runtime_data,
                &plan.initial_fields,
                &options,
                Some(&mut print_iteration),
            )
            .map_err(|error| error.to_string())?
        } else {
            solve_laminar_simple_with_observer(
                &mut plan.runtime_data,
                &plan.initial_fields,
                &options,
                Some(&mut print_iteration),
            )
            .map_err(|error| error.to_string())?
        }
    } else if solve.profile_pcg {
        solve_laminar_simple_profiled_pcg(&mut plan.runtime_data, &plan.initial_fields, &options)
            .map_err(|error| error.to_string())?
    } else {
        solve_laminar_simple(&mut plan.runtime_data, &plan.initial_fields, &options)
            .map_err(|error| error.to_string())?
    };
    let wall_clock_seconds = started.elapsed().as_secs_f64();
    let post_processing = build_laminar_simple_post_processing(plan, solve, &options, &report)?;

    if let Some(continuity_errors) = &post_processing.continuity_errors {
        print_continuity_errors(continuity_errors);
    }
    if let Some(wall_forces) = &post_processing.wall_forces {
        print_wall_force_coefficients(wall_forces);
        print_wall_force_method(wall_forces);
    }

    println!(
        "incompressibleFluid solve: backend=cpu momentumExecution={} linearSolver={} momentumLinearSolver={} momentumPreconditioner={} momentumRelTol={} pressureLinearSolver={} pressurePreconditioner={} pressureRelTol={} divPhiU=\"{}\" gradP=\"{}\" gradU=\"{}\" laplacian=\"{}\" snGrad=\"{}\" interpolation=\"{}\" pRefCell={} pRefValue={} nonOrthogonalCorrectors={} consistent={} stopReason={} cells={} faces={} simpleIterations={} minSimpleIterations={} converged={} residualControl={} initialContinuityL2={} finalContinuityL2={} momentumInitialResidual={} momentumFinalResidual={} momentumResidualNorm={} pressureInitialResidual={} pressureFinalResidual={} pressureResidualNorm={} momentumLinearIterations={} pressureLinearIterations={} wallClockSeconds={:.6}",
        options.momentum_execution,
        options.linear_solver,
        options.momentum_linear_solver,
        options.momentum_preconditioner,
        format_scientific(options.momentum_linear_relative_tolerance),
        options.pressure_linear_solver,
        options.pressure_preconditioner,
        format_scientific(options.pressure_linear_relative_tolerance),
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
    if let Some(gamg) = options.pressure_gamg_options {
        println!("{}", format_pressure_gamg_console(&gamg));
    }
    println!(
        "incompressibleFluid residualControl: state={} checked={} satisfied={} U(tolerance={},initial={},satisfied={}) p(tolerance={},initial={},satisfied={})",
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
        "incompressibleFluid outerConvergence: status={} configured={} evaluated={} converged={} reason={}",
        outer_convergence_status(&report),
        yes_no(report.residual_control.configured),
        yes_no(report.residual_control.checked),
        yes_no(report.converged),
        report.stop_reason,
    );
    println!(
        "incompressibleFluid linearSolves: finalMomentumConverged={} finalPressureConverged={} momentumPredictors={} momentumNonConvergedPredictors={} momentumComponentSolves={} momentumComponentNonConvergedSolves={} pressureCorrectionSolves={} pressureCorrectionNonConvergedSolves={} maxMomentumIterationsPerSimple={} maxPressureIterationsPerSimple={} avgMomentumIterationsPerSimple={} avgPressureIterationsPerSimple={}",
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
        "incompressibleFluid timing: solverTotalSeconds={:.6} driverMeasuredSeconds={:.6} setupSeconds={:.6} iterationSetupSeconds={:.6} operatorEvaluationSeconds={:.6} momentumAssemblySeconds={:.6} momentumGradientSeconds={:.6} momentumMatrixFillSeconds={:.6} momentumLinearSolveSeconds={:.6} pressureCouplingSetupSeconds={:.6} pressureAssemblySeconds={:.6} pressureLinearSolveSeconds={:.6} fieldCorrectionSeconds={:.6} finalizationSeconds={:.6} otherSolverWorkSeconds={:.6}",
        report.timing.solver_total_seconds,
        wall_clock_seconds,
        report.timing.setup_seconds,
        report.timing.iteration_setup_seconds,
        report.timing.operator_evaluation_seconds,
        report.timing.momentum_assembly_seconds,
        report.timing.momentum_gradient_seconds,
        report.timing.momentum_matrix_fill_seconds,
        report.timing.momentum_linear_solve_seconds,
        report.timing.pressure_coupling_setup_seconds,
        report.timing.pressure_assembly_seconds,
        report.timing.pressure_linear_solve_seconds,
        report.timing.field_correction_seconds,
        report.timing.finalization_seconds,
        report.timing.other_solver_work_seconds,
    );
    if solve.profile_pcg {
        println!(
            "incompressibleFluid pressurePcgKernel: totalSeconds={:.6} preconditionerUpdateSeconds={:.6} matrixVectorSeconds={:.6} preconditionerApplicationSeconds={:.6} vectorOperationSeconds={:.6} otherSeconds={:.6} matrixVectorProducts={} preconditionerApplications={}",
            report.timing.pressure_pcg_total_seconds,
            report.timing.pressure_preconditioner_update_seconds,
            report.timing.pressure_matrix_vector_seconds,
            report.timing.pressure_preconditioner_application_seconds,
            report.timing.pressure_vector_operation_seconds,
            report.timing.pressure_pcg_other_seconds,
            report.timing.pressure_matrix_vector_products,
            report.timing.pressure_preconditioner_applications,
        );
    }
    if let Some(profile) = &report.timing.pressure_gamg_profile {
        println!(
            "incompressibleFluid pressureGamgProfile: totalSeconds={:.6} hierarchyBuildSeconds={:.6} hierarchyRebuildSeconds={:.6} hierarchyDiagnosticSeconds={:.6} matrixRefreshSeconds={:.6} finestResidualSeconds={:.6} vCycleSeconds={:.6} restrictionSeconds={:.6} prolongationSeconds={:.6} smoothingSeconds={:.6} scalingSeconds={:.6} coarseResidualSeconds={:.6} correctionSeconds={:.6} coarsestSolveSeconds={:.6} vCycleOtherSeconds={:.6} otherSeconds={:.6} solves={} vCycles={} levels={} {}",
            profile.total_seconds,
            profile.hierarchy_build_seconds,
            profile.hierarchy_rebuild_seconds,
            profile.hierarchy_diagnostic_seconds,
            profile.matrix_refresh_seconds,
            profile.finest_residual_seconds,
            profile.v_cycle_seconds,
            profile.restriction_seconds(),
            profile.prolongation_seconds(),
            profile.smoothing_seconds(),
            profile.scaling_seconds(),
            profile.coarse_residual_seconds(),
            profile.correction_seconds(),
            profile.coarsest_solve_seconds(),
            profile.v_cycle_other_seconds(),
            profile.other_seconds,
            profile.solves,
            profile.v_cycles,
            profile.levels.len(),
            gamg_outer_profile_counters(profile),
        );
        for level in &profile.levels {
            println!(
                "incompressibleFluid pressureGamgLevel: level={} cells={} nonzeros={} matrixRefreshSeconds={:.6} restrictionSeconds={:.6} prolongationSeconds={:.6} smoothingSeconds={:.6} scalingSeconds={:.6} residualSeconds={:.6} correctionSeconds={:.6} coarsestSolveSeconds={:.6} restrictionCalls={} prolongationCalls={} smoothingCalls={} smoothingSweeps={} scalingCalls={} residualEvaluations={} correctionUpdates={} coarsestSolves={} coarsestIterations={}",
                level.level,
                level.cells,
                level.nonzeros,
                level.matrix_refresh_seconds,
                level.restriction_seconds,
                level.prolongation_seconds,
                level.smoothing_seconds,
                level.scaling_seconds,
                level.residual_seconds,
                level.correction_seconds,
                level.coarsest_solve_seconds,
                level.restriction_calls,
                level.prolongation_calls,
                level.smoothing_calls,
                level.smoothing_sweeps,
                level.scaling_calls,
                level.residual_evaluations,
                level.correction_updates,
                level.coarsest_solves,
                level.coarsest_iterations,
            );
        }
        if let Some(hierarchy) = &profile.hierarchy {
            let grid_complexity = hierarchy.grid_complexity_terms();
            let operator_complexity = hierarchy.operator_complexity_terms();
            println!(
                "incompressibleFluid pressureGamgHierarchy: smootherPassesPerSweep={} directSolveCoarsest={} gridComplexityNumeratorDecimal={} gridComplexityDenominatorDecimal={} operatorComplexityNumeratorDecimal={} operatorComplexityDenominatorDecimal={} nnzWeightedSmoothingWorkDecimal={} nnzWeightedSparseWorkDecimal={} levels={} transfers={}",
                hierarchy.smoother_passes_per_sweep,
                yes_no(hierarchy.direct_solve_coarsest),
                format_optional_u128(grid_complexity.map(|terms| terms.0)),
                format_optional_u128(grid_complexity.map(|terms| terms.1)),
                format_optional_u128(operator_complexity.map(|terms| terms.0)),
                format_optional_u128(operator_complexity.map(|terms| terms.1)),
                format_optional_u128(profile.nnz_weighted_smoothing_work()),
                format_optional_u128(profile.nnz_weighted_sparse_work()),
                hierarchy.levels.len(),
                hierarchy.transfers.len(),
            );
            for transfer in &hierarchy.transfers {
                println!(
                    "incompressibleFluid pressureGamgTransfer: fineLevel={} coarseLevel={} fineCells={} coarseCells={} singletonFineCells={} unmatchedFineCells={} minAggregateSize={} maxAggregateSize={}",
                    transfer.fine_level,
                    transfer.coarse_level,
                    transfer.fine_cells,
                    transfer.coarse_cells,
                    transfer.singleton_fine_cells,
                    transfer.unmatched_fine_cells,
                    transfer.min_aggregate_size,
                    transfer.max_aggregate_size,
                );
                for bin in &transfer.aggregate_size_histogram {
                    println!(
                        "incompressibleFluid pressureGamgAggregateSize: fineLevel={} coarseLevel={} aggregateSize={} aggregateCount={}",
                        transfer.fine_level,
                        transfer.coarse_level,
                        bin.aggregate_size,
                        bin.aggregate_count,
                    );
                }
            }
        } else {
            println!("incompressibleFluid pressureGamgHierarchy: available=no");
        }
    }
    println!(
        "incompressibleFluid fields: velocityMinMagnitude={} velocityMaxMagnitude={} velocityL2={} velocityXMin={} velocityXMax={} velocityYMin={} velocityYMax={} velocityZMin={} velocityZMax={} pressureMin={} pressureMax={} pressureL2={}",
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
    let needs_output_root = solve.solve_residual_csv.is_some()
        || solve.solve_residual_plot.is_some()
        || solve.report_json.is_some()
        || solve.report_markdown.is_some()
        || solve.write_final_fields.is_some()
        || solve
            .wall_forces
            .as_ref()
            .and_then(|request| request.face_loads_csv.as_ref())
            .is_some();
    let output_root = if needs_output_root {
        let path = env::current_dir()
            .map_err(|error| format!("could not resolve the solver output root ({error})"))?;
        Some(SafeOutputRoot::open_existing(&path).map_err(|error| {
            format!(
                "could not safely open solver output root {} ({error})",
                path.display()
            )
        })?)
    } else {
        None
    };
    let output_root = output_root.as_ref();

    if let Some(path) = solve
        .wall_forces
        .as_ref()
        .and_then(|request| request.face_loads_csv.as_ref())
    {
        let wall_forces = post_processing.wall_forces.as_ref().ok_or_else(|| {
            "internal error: wall-face CSV was requested without reconstructed wall forces"
                .to_string()
        })?;
        write_laminar_simple_wall_face_loads_csv(
            &plan.runtime_data.mesh,
            &options,
            wall_forces,
            output_root.expect("output requested"),
            path,
        )
        .map_err(|error| {
            format!(
                "could not write per-face wall loads CSV to {} ({error})",
                path.display()
            )
        })?;
        println!("wrote per-face wall loads CSV: {}", path.display());
    }

    let mut residual_csv_path = solve.solve_residual_csv.clone();
    if let Some(path) = &solve.solve_residual_csv {
        write_laminar_simple_residual_csv(&report, output_root.expect("output requested"), path)
            .map_err(|error| {
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
                write_laminar_simple_residual_csv(
                    &report,
                    output_root.expect("output requested"),
                    csv_path,
                )
                .map_err(|error| {
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
            match write_laminar_simple_residual_plot(
                output_root.expect("output requested"),
                csv_path,
                plot_path,
            ) {
                Ok(()) => {
                    println!(
                        "wrote laminar SIMPLE residual plot: {}",
                        plot_path.display()
                    )
                }
                Err(error) => {
                    println!(
                        "incompressibleFluid residual plot warning: {} (CSV: {})",
                        error,
                        csv_path.display()
                    )
                }
            }
        }
    }
    println!(
        "incompressibleFluid operators: phiMin={} phiMax={} phiSumAbs={} gradPL2={} hbyAL2={} divPhiUL2={} velocityFixedValueFaces={} velocityZeroGradientFaces={} velocityInletOutletFaces={} pressureFixedValueFaces={} pressureZeroGradientFaces={}",
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
        write_laminar_simple_fields(
            &plan.initial_fields,
            &report,
            output_root.expect("output requested"),
            output_dir,
        )
        .map_err(|error| {
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
        println!("incompressibleFluid status: no field files written");
    }

    if let Some(path) = &solve.report_json {
        write_laminar_simple_report_json(
            plan,
            &options,
            &report,
            &post_processing,
            LaminarSimpleReportRunMetadata {
                profile_pcg: solve.profile_pcg,
                wall_clock_seconds,
            },
            output_root.expect("output requested"),
            path,
        )
        .map_err(|error| {
            format!(
                "could not write laminar SIMPLE report JSON to {} ({error})",
                path.display()
            )
        })?;
        println!("wrote laminar SIMPLE report json: {}", path.display());
    }
    if let Some(path) = &solve.report_markdown {
        write_laminar_simple_report_markdown(
            plan,
            &options,
            &report,
            &post_processing,
            LaminarSimpleReportRunMetadata {
                profile_pcg: solve.profile_pcg,
                wall_clock_seconds,
            },
            output_root.expect("output requested"),
            path,
        )
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

fn validate_wall_face_loads_output_destination(
    solve: &LaminarSimpleSolveArgs,
) -> Result<(), String> {
    let Some(path) = solve
        .wall_forces
        .as_ref()
        .and_then(|request| request.face_loads_csv.as_ref())
    else {
        return Ok(());
    };
    let root = env::current_dir()
        .map_err(|error| format!("could not resolve the solver output root ({error})"))?;
    let output_root = SafeOutputRoot::open_existing(&root).map_err(|error| {
        format!(
            "could not safely open solver output root {} ({error})",
            root.display()
        )
    })?;
    validate_wall_face_loads_output_destination_in_root(solve, path, &output_root)
}

fn validate_wall_face_loads_output_destination_in_root(
    solve: &LaminarSimpleSolveArgs,
    wall_face_loads_path: &Path,
    output_root: &SafeOutputRoot,
) -> Result<(), String> {
    let wall_face_loads =
        validated_output_collision_key(output_root, wall_face_loads_path, "--wallFaceLoadsCsv")?;
    let mut other_files = Vec::new();
    other_files
        .try_reserve_exact(7)
        .map_err(|_| "could not allocate solver output collision checks".to_string())?;
    if let Some(path) = solve.solve_residual_csv.as_deref() {
        other_files.push(("--solveResidualCsv", path.to_path_buf()));
    } else if let Some(plot) = solve.solve_residual_plot.as_deref() {
        other_files.push(("automatic residual CSV", plot.with_extension("csv")));
    }
    if let Some(path) = solve.solve_residual_plot.as_deref() {
        other_files.push(("--solveResidualPlot", path.to_path_buf()));
    }
    if let Some(path) = solve.report_json.as_deref() {
        other_files.push(("--solveReportJson", path.to_path_buf()));
    }
    if let Some(path) = solve.report_markdown.as_deref() {
        other_files.push(("--solveReportMarkdown", path.to_path_buf()));
    }
    if let Some(directory) = solve.write_final_fields.as_deref() {
        other_files.push((
            "--writeFinalFields U",
            fallible_join_output_path(directory, "U")
                .map_err(|error| format!("could not validate final U output path ({error})"))?,
        ));
        other_files.push((
            "--writeFinalFields p",
            fallible_join_output_path(directory, "p")
                .map_err(|error| format!("could not validate final p output path ({error})"))?,
        ));
        let directory_key =
            validated_output_collision_key(output_root, directory, "--writeFinalFields")?;
        if wall_face_loads == directory_key
            || output_key_is_prefix(&wall_face_loads, &directory_key)
        {
            return Err(format!(
                "--wallFaceLoadsCsv path '{}' collides with --writeFinalFields directory '{}'",
                wall_face_loads_path.display(),
                directory.display()
            ));
        }
    }

    for (label, path) in other_files {
        let other = validated_output_collision_key(output_root, &path, label)?;
        if wall_face_loads == other
            || output_key_is_prefix(&wall_face_loads, &other)
            || output_key_is_prefix(&other, &wall_face_loads)
        {
            return Err(format!(
                "--wallFaceLoadsCsv path '{}' collides with {label} path '{}'",
                wall_face_loads_path.display(),
                path.display()
            ));
        }
    }
    Ok(())
}

fn validated_output_collision_key(
    output_root: &SafeOutputRoot,
    path: &Path,
    label: &str,
) -> Result<Vec<String>, String> {
    let relative = relative_output_path(output_root, path)
        .and_then(|relative| {
            output_root.validate_file_path(&relative)?;
            Ok(relative)
        })
        .map_err(|error| format!("invalid {label} output path '{}' ({error})", path.display()))?;
    let components = relative.components().count();
    let mut key = Vec::new();
    key.try_reserve_exact(components)
        .map_err(|_| format!("could not allocate {label} output path validation"))?;
    for component in relative.components() {
        let std::path::Component::Normal(component) = component else {
            return Err(format!("invalid {label} output path '{}'", path.display()));
        };
        let component = component.to_str().ok_or_else(|| {
            format!(
                "invalid non-Unicode {label} output path '{}'",
                path.display()
            )
        })?;
        #[cfg(windows)]
        key.push(component.to_lowercase());
        #[cfg(not(windows))]
        key.push(component.to_string());
    }
    Ok(key)
}

fn output_key_is_prefix(prefix: &[String], value: &[String]) -> bool {
    prefix.len() < value.len() && prefix.iter().zip(value).all(|(left, right)| left == right)
}

fn build_laminar_simple_post_processing(
    plan: &SolverCasePlan,
    solve: &LaminarSimpleSolveArgs,
    options: &LaminarSimpleOptions,
    report: &LaminarSimpleReport,
) -> Result<LaminarSimplePostProcessing, String> {
    let total_volume = plan.runtime_data.mesh.total_cell_volume;
    let continuity_errors = plan
        .run
        .delta_t
        .map(|delta_t| {
            let final_error =
                normalize_steady_continuity(report.final_continuity, total_volume, delta_t)
                    .map_err(|error| {
                        format!("could not normalize final continuity errors ({error})")
                    })?;
            if report.history.len() != report.simple_iterations {
                return Err(format!(
                    "could not normalize cumulative continuity errors: history has {} completed iterations but report records {}",
                    report.history.len(),
                    report.simple_iterations
                ));
            }
            let mut cumulative_global = 0.0;
            for item in &report.history {
                let normalized =
                    normalize_steady_continuity(item.continuity_after, total_volume, delta_t)
                        .map_err(|error| {
                            format!(
                                "could not normalize continuity errors after SIMPLE iteration {} ({error})",
                                item.iteration
                            )
                        })?;
                cumulative_global += normalized.global;
                if !cumulative_global.is_finite() {
                    return Err(format!(
                        "cumulative normalized global continuity error became non-finite after SIMPLE iteration {}",
                        item.iteration
                    ));
                }
            }
            Ok(ContinuityErrorsReport {
                final_error,
                cumulative_global,
                delta_t,
                total_volume,
            })
        })
        .transpose()?;

    let wall_forces = solve
        .wall_forces
        .as_ref()
        .map(|request| -> Result<WallForceReport, String> {
            let patch_names = request
                .patch_names
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>();
            let gradients = reconstruct_laminar_gradients_from_fields(
                &plan.runtime_data,
                &plan.initial_fields,
                &report.final_velocity,
                &report.final_pressure,
                options.schemes,
            )
            .map_err(|error| {
                format!("could not reconstruct final fields for selected wall forces ({error})")
            })?;
            let resolved =
                integrate_stationary_no_slip_zero_gradient_pressure_wall_forces_with_reconstructed_gradient(
                &plan.runtime_data.mesh,
                &report.final_velocity,
                &report.final_pressure,
                CellVelocityGradientComponents {
                    grad_ux: &gradients.velocity_component_gradients[0],
                    grad_uy: &gradients.velocity_component_gradients[1],
                    grad_uz: &gradients.velocity_component_gradients[2],
                },
                &patch_names,
                NoSlipWallForceOptions {
                    pressure_kind: PressureFieldKind::Kinematic,
                    pressure_reference: PressureReference::AreaVectorBalancedMean {
                        relative_area_vector_imbalance_tolerance:
                            MAX_RELATIVE_AREA_VECTOR_IMBALANCE_TOLERANCE,
                    },
                    density: options.density,
                    dynamic_viscosity: options.dynamic_viscosity,
                    reference_speed: request.reference_speed,
                    reference_area: ReferenceArea::Explicit(request.reference_area),
                    drag_direction: Point3 {
                        x: 1.0,
                        y: 0.0,
                        z: 0.0,
                    },
                    lift_direction: Point3 {
                        x: 0.0,
                        y: 1.0,
                        z: 0.0,
                    },
                },
            )
            .map_err(|error| format!("could not integrate selected wall forces ({error})"))?;
            Ok(WallForceReport {
                patch_names: request.patch_names.clone(),
                reference_speed: request.reference_speed,
                velocity_gradient_scheme: options.schemes.grad_u,
                coefficients: resolved.coefficients,
                faces: resolved.faces,
            })
        })
        .transpose()?;

    Ok(LaminarSimplePostProcessing {
        continuity_errors,
        wall_forces,
    })
}

fn validate_wall_force_boundary_conditions(
    fields: &InitialFieldSet,
    patch_names: &[String],
) -> Result<(), String> {
    let velocity = required_initial_field(fields, "U")?;
    let pressure = required_initial_field(fields, "p")?;
    for patch_name in patch_names {
        validate_wall_force_patch_condition(velocity, patch_name, "noSlip")?;
        validate_wall_force_patch_condition(pressure, patch_name, "zeroGradient")?;
    }
    Ok(())
}

fn required_initial_field<'a>(
    fields: &'a InitialFieldSet,
    field_name: &str,
) -> Result<&'a FieldFile, String> {
    let mut matches = fields
        .fields
        .iter()
        .filter(|field| field.region.is_none() && field.name == field_name);
    let field = matches.next().ok_or_else(|| {
        format!("wall-force integration requires the initial field 0/{field_name}")
    })?;
    if matches.next().is_some() {
        return Err(format!(
            "wall-force integration requires exactly one default-region 0/{field_name} field"
        ));
    }
    Ok(field)
}

fn validate_wall_force_patch_condition(
    field: &FieldFile,
    patch_name: &str,
    required_type: &str,
) -> Result<(), String> {
    let mut matches = field
        .boundary_patches
        .iter()
        .filter(|patch| patch.name == patch_name);
    let patch = matches.next().ok_or_else(|| {
        format!(
            "wall-force patch '{patch_name}' is missing from 0/{} boundaryField",
            field.name
        )
    })?;
    if matches.next().is_some() {
        return Err(format!(
            "wall-force patch '{patch_name}' occurs more than once in 0/{} boundaryField",
            field.name
        ));
    }
    if patch.patch_type.as_deref() != Some(required_type) {
        return Err(format!(
            "wall-force patch '{patch_name}' requires 0/{} type {required_type}, found {}",
            field.name,
            patch.patch_type.as_deref().unwrap_or("missing")
        ));
    }
    Ok(())
}

fn print_continuity_errors(summary: &ContinuityErrorsReport) {
    println!(
        "incompressibleFluid continuityErrors: local={} global={} cumulativeGlobal={} deltaT={} totalVolume={}",
        format_scientific(summary.final_error.local),
        format_scientific(summary.final_error.global),
        format_scientific(summary.cumulative_global),
        format_scientific(summary.delta_t),
        format_scientific(summary.total_volume),
    );
}

fn print_wall_force_coefficients(report: &WallForceReport) {
    let value = &report.coefficients;
    println!(
        "incompressibleFluid wallForces: patches={} selectedPatches={} selectedFaces={} pressureFx={} pressureFy={} pressureFz={} viscousFx={} viscousFy={} viscousFz={} totalFx={} totalFy={} totalFz={} dragPressure={} dragViscous={} dragTotal={} liftPressure={} liftViscous={} liftTotal={} referenceArea={} referenceSpeed={}",
        report.patch_names.join(","),
        value.selected_patches,
        value.selected_faces,
        format_scientific(value.pressure_force.x),
        format_scientific(value.pressure_force.y),
        format_scientific(value.pressure_force.z),
        format_scientific(value.viscous_force.x),
        format_scientific(value.viscous_force.y),
        format_scientific(value.viscous_force.z),
        format_scientific(value.total_force.x),
        format_scientific(value.total_force.y),
        format_scientific(value.total_force.z),
        format_scientific(value.drag.pressure),
        format_scientific(value.drag.viscous),
        format_scientific(value.drag.total),
        format_scientific(value.lift.pressure),
        format_scientific(value.lift.viscous),
        format_scientific(value.lift.total),
        format_scientific(value.resolved_reference_area),
        format_scientific(report.reference_speed),
    );
}

fn print_wall_force_method(report: &WallForceReport) {
    println!(
        "incompressibleFluid wallForceMethod: tractionMethod={} tractionMethodVersion={} forceConvention={} faceAreaVectorOrientation={} pressureFaceTreatment={} gradU=\"{}\"",
        WALL_FORCE_TRACTION_METHOD,
        WALL_FORCE_TRACTION_METHOD_VERSION,
        WALL_FORCE_FORCE_CONVENTION,
        WALL_FORCE_AREA_VECTOR_ORIENTATION,
        WALL_FORCE_PRESSURE_FACE_TREATMENT,
        report.velocity_gradient_scheme,
    );
}

fn write_laminar_simple_wall_face_loads_csv(
    mesh: &SolverRuntimeMeshData,
    options: &LaminarSimpleOptions,
    report: &WallForceReport,
    output_root: &SafeOutputRoot,
    path: &Path,
) -> std::io::Result<()> {
    validate_wall_face_loads_metadata(options, report)?;
    visit_wall_face_load_rows(mesh, report, |_patch, _centre, _face| Ok(()))?;

    let relative = relative_output_path(output_root, path)?;
    output_root.validate_file_path(&relative)?;
    validate_replace_output_entry(output_root, &relative)?;

    let file = open_replace_output_file(output_root, path)?;
    let mut writer = BufWriter::new(file);
    writeln!(
        writer,
        "wallFaceLoadsSchemaVersion,tractionMethod,tractionMethodVersion,forceConvention,faceAreaVectorOrientation,pressureFieldKind,pressureReferenceMode,pressureReferenceRelativeAreaVectorImbalanceTolerance,pressureFaceTreatment,velocityGradientScheme,densityKgPerM3,dynamicViscosityPaS,referenceSpeedMPerS,referenceAreaM2,referenceDynamicPressurePa,resolvedPressureReferenceM2PerS2,patch,faceIndex,ownerCell,faceCenterXM,faceCenterYM,faceCenterZM,areaVectorXM2,areaVectorYM2,areaVectorZM2,boundaryPressureM2PerS2,resolvedDynamicPressurePa,pressureTractionOnBodyXPa,pressureTractionOnBodyYPa,pressureTractionOnBodyZPa,viscousTractionOnBodyXPa,viscousTractionOnBodyYPa,viscousTractionOnBodyZPa,wallShearTractionOnBodyXPa,wallShearTractionOnBodyYPa,wallShearTractionOnBodyZPa,pressureForceOnBodyXN,pressureForceOnBodyYN,pressureForceOnBodyZN,viscousForceOnBodyXN,viscousForceOnBodyYN,viscousForceOnBodyZN,totalForceOnBodyXN,totalForceOnBodyYN,totalForceOnBodyZN"
    )?;
    visit_wall_face_load_rows(mesh, report, |patch, centre, face| {
        write!(writer, "{WALL_FACE_LOADS_CSV_SCHEMA_VERSION},")?;
        write_csv_text_field(&mut writer, WALL_FORCE_TRACTION_METHOD)?;
        write!(writer, ",{WALL_FORCE_TRACTION_METHOD_VERSION},")?;
        write_csv_text_field(&mut writer, WALL_FORCE_FORCE_CONVENTION)?;
        writer.write_all(b",")?;
        write_csv_text_field(&mut writer, WALL_FORCE_AREA_VECTOR_ORIENTATION)?;
        writer.write_all(b",")?;
        write_csv_text_field(&mut writer, WALL_FORCE_PRESSURE_FIELD_KIND)?;
        writer.write_all(b",")?;
        write_csv_text_field(&mut writer, WALL_FORCE_PRESSURE_REFERENCE_MODE)?;
        write!(writer, ",{},", MAX_RELATIVE_AREA_VECTOR_IMBALANCE_TOLERANCE)?;
        write_csv_text_field(&mut writer, WALL_FORCE_PRESSURE_FACE_TREATMENT)?;
        writer.write_all(b",")?;
        write_csv_text_field(&mut writer, &report.velocity_gradient_scheme.to_string())?;
        write!(
            writer,
            ",{},{},{},{},{},{},",
            options.density,
            options.dynamic_viscosity,
            report.reference_speed,
            report.coefficients.resolved_reference_area,
            report.coefficients.reference_dynamic_pressure,
            report.coefficients.resolved_pressure_reference,
        )?;
        write_csv_text_field(&mut writer, patch)?;
        writeln!(
            writer,
            ",{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            face.face_index,
            face.owner_cell,
            centre.x,
            centre.y,
            centre.z,
            face.area_vector.x,
            face.area_vector.y,
            face.area_vector.z,
            face.boundary_pressure,
            face.resolved_dynamic_pressure,
            face.pressure_traction_on_body.x,
            face.pressure_traction_on_body.y,
            face.pressure_traction_on_body.z,
            face.viscous_traction_on_body.x,
            face.viscous_traction_on_body.y,
            face.viscous_traction_on_body.z,
            face.wall_shear_traction_on_body.x,
            face.wall_shear_traction_on_body.y,
            face.wall_shear_traction_on_body.z,
            face.pressure_force_on_body.x,
            face.pressure_force_on_body.y,
            face.pressure_force_on_body.z,
            face.viscous_force_on_body.x,
            face.viscous_force_on_body.y,
            face.viscous_force_on_body.z,
            face.total_force_on_body.x,
            face.total_force_on_body.y,
            face.total_force_on_body.z,
        )
    })?;
    writer.flush()
}

fn validate_wall_face_loads_metadata(
    options: &LaminarSimpleOptions,
    report: &WallForceReport,
) -> std::io::Result<()> {
    if !options.density.is_finite()
        || options.density <= 0.0
        || !options.dynamic_viscosity.is_finite()
        || options.dynamic_viscosity <= 0.0
        || !report.reference_speed.is_finite()
        || report.reference_speed <= 0.0
        || report.patch_names.is_empty()
        || report.coefficients.selected_patches != report.patch_names.len()
        || report.coefficients.selected_faces != report.faces.len()
    {
        return Err(invalid_field_data(
            "per-face wall loads metadata is inconsistent".to_string(),
        ));
    }
    for value in [
        report.coefficients.selected_area,
        report.coefficients.resolved_pressure_reference,
        report.coefficients.resolved_reference_area,
        report.coefficients.reference_dynamic_pressure,
    ] {
        if !value.is_finite() {
            return Err(invalid_field_data(
                "per-face wall loads metadata contains a non-finite value".to_string(),
            ));
        }
    }
    if report.coefficients.selected_area <= 0.0
        || report.coefficients.resolved_reference_area <= 0.0
        || report.coefficients.reference_dynamic_pressure <= 0.0
    {
        return Err(invalid_field_data(
            "per-face wall loads metadata contains a non-positive scale".to_string(),
        ));
    }
    Ok(())
}

fn visit_wall_face_load_rows(
    mesh: &SolverRuntimeMeshData,
    report: &WallForceReport,
    mut visitor: impl FnMut(&str, Point3, &WallFaceForceContribution) -> std::io::Result<()>,
) -> std::io::Result<()> {
    if mesh.owner.len() != mesh.faces
        || mesh.neighbour.len() != mesh.faces
        || mesh.face_centres.len() != mesh.faces
        || mesh.face_area_vectors.len() != mesh.faces
    {
        return Err(invalid_field_data(
            "runtime mesh arrays do not match the face count".to_string(),
        ));
    }

    let mut row = 0usize;
    let mut pressure_force = CsvCompensatedPoint::default();
    let mut viscous_force = CsvCompensatedPoint::default();
    for (patch_position, patch_name) in report.patch_names.iter().enumerate() {
        if report.patch_names[..patch_position]
            .iter()
            .any(|previous| previous == patch_name)
        {
            return Err(invalid_field_data(format!(
                "per-face wall loads patch '{patch_name}' is selected more than once"
            )));
        }
        let mut matches = mesh
            .patches
            .iter()
            .filter(|patch| patch.name == *patch_name);
        let patch = matches.next().ok_or_else(|| {
            invalid_field_data(format!(
                "per-face wall loads patch '{patch_name}' is missing from the runtime mesh"
            ))
        })?;
        if matches.next().is_some() {
            return Err(invalid_field_data(format!(
                "per-face wall loads patch '{patch_name}' occurs more than once in the runtime mesh"
            )));
        }
        if patch.patch_type != "wall" {
            return Err(invalid_field_data(format!(
                "per-face wall loads patch '{patch_name}' has mesh type '{}', expected wall",
                patch.patch_type
            )));
        }
        let end = patch.start_face.checked_add(patch.faces).ok_or_else(|| {
            invalid_field_data(format!(
                "per-face wall loads patch '{patch_name}' face range overflows"
            ))
        })?;
        if patch.start_face < mesh.internal_faces || end > mesh.faces {
            return Err(invalid_field_data(format!(
                "per-face wall loads patch '{patch_name}' has an invalid boundary-face range"
            )));
        }

        for expected_face in patch.start_face..end {
            let face = report.faces.get(row).ok_or_else(|| {
                invalid_field_data(format!(
                    "per-face wall loads ended before patch '{patch_name}' face {expected_face}"
                ))
            })?;
            if face.face_index != expected_face {
                return Err(invalid_field_data(format!(
                    "per-face wall loads row {row} has face {}, expected {expected_face}",
                    face.face_index
                )));
            }
            let expected_owner = mesh.owner[expected_face];
            if face.owner_cell != expected_owner || expected_owner >= mesh.cells {
                return Err(invalid_field_data(format!(
                    "per-face wall loads face {expected_face} has an inconsistent owner cell"
                )));
            }
            if mesh.neighbour[expected_face].is_some() {
                return Err(invalid_field_data(format!(
                    "per-face wall loads face {expected_face} is not a boundary face"
                )));
            }
            let centre = mesh.face_centres[expected_face];
            let area_vector = mesh.face_area_vectors[expected_face];
            if !point_is_finite(centre)
                || !point_bits_equal(face.area_vector, area_vector)
                || !wall_face_contribution_is_finite(face)
            {
                return Err(invalid_field_data(format!(
                    "per-face wall loads face {expected_face} contains invalid geometry or values"
                )));
            }
            let total = Point3 {
                x: face.pressure_force_on_body.x + face.viscous_force_on_body.x,
                y: face.pressure_force_on_body.y + face.viscous_force_on_body.y,
                z: face.pressure_force_on_body.z + face.viscous_force_on_body.z,
            };
            if !point_bits_equal(total, face.total_force_on_body) {
                return Err(invalid_field_data(format!(
                    "per-face wall loads face {expected_face} has an inconsistent total force"
                )));
            }
            pressure_force.add(face.pressure_force_on_body);
            viscous_force.add(face.viscous_force_on_body);
            visitor(patch_name, centre, face)?;
            row = row.checked_add(1).ok_or_else(|| {
                invalid_field_data("per-face wall loads row count overflows".to_string())
            })?;
        }
    }

    if row != report.faces.len() {
        return Err(invalid_field_data(format!(
            "per-face wall loads contains {} rows after the selected patch ranges",
            report.faces.len() - row
        )));
    }
    let pressure_force = pressure_force.total();
    let viscous_force = viscous_force.total();
    let total_force = Point3 {
        x: pressure_force.x + viscous_force.x,
        y: pressure_force.y + viscous_force.y,
        z: pressure_force.z + viscous_force.z,
    };
    if !point_bits_equal(pressure_force, report.coefficients.pressure_force)
        || !point_bits_equal(viscous_force, report.coefficients.viscous_force)
        || !point_bits_equal(total_force, report.coefficients.total_force)
    {
        return Err(invalid_field_data(
            "per-face wall loads do not reproduce the aggregate wall forces".to_string(),
        ));
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Default)]
struct CsvCompensatedSum {
    sum: f64,
    correction: f64,
}

impl CsvCompensatedSum {
    fn add(&mut self, value: f64) {
        let next = self.sum + value;
        if self.sum.abs() >= value.abs() {
            self.correction += (self.sum - next) + value;
        } else {
            self.correction += (value - next) + self.sum;
        }
        self.sum = next;
    }

    fn total(self) -> f64 {
        self.sum + self.correction
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct CsvCompensatedPoint {
    x: CsvCompensatedSum,
    y: CsvCompensatedSum,
    z: CsvCompensatedSum,
}

impl CsvCompensatedPoint {
    fn add(&mut self, value: Point3) {
        self.x.add(value.x);
        self.y.add(value.y);
        self.z.add(value.z);
    }

    fn total(self) -> Point3 {
        Point3 {
            x: self.x.total(),
            y: self.y.total(),
            z: self.z.total(),
        }
    }
}

fn point_is_finite(value: Point3) -> bool {
    value.x.is_finite() && value.y.is_finite() && value.z.is_finite()
}

fn point_bits_equal(left: Point3, right: Point3) -> bool {
    left.x.to_bits() == right.x.to_bits()
        && left.y.to_bits() == right.y.to_bits()
        && left.z.to_bits() == right.z.to_bits()
}

fn wall_face_contribution_is_finite(face: &WallFaceForceContribution) -> bool {
    face.boundary_pressure.is_finite()
        && face.resolved_dynamic_pressure.is_finite()
        && point_is_finite(face.area_vector)
        && point_is_finite(face.pressure_traction_on_body)
        && point_is_finite(face.viscous_traction_on_body)
        && point_is_finite(face.wall_shear_traction_on_body)
        && point_is_finite(face.pressure_force_on_body)
        && point_is_finite(face.viscous_force_on_body)
        && point_is_finite(face.total_force_on_body)
}

fn write_csv_text_field(writer: &mut impl Write, value: &str) -> std::io::Result<()> {
    writer.write_all(b"\"")?;
    if value
        .chars()
        .next()
        .is_some_and(|first| matches!(first, '=' | '+' | '-' | '@' | '\t' | '\r'))
    {
        writer.write_all(b"'")?;
    }
    for character in value.chars() {
        if character == '"' {
            writer.write_all(b"\"\"")?;
        } else {
            let mut encoded = [0u8; 4];
            writer.write_all(character.encode_utf8(&mut encoded).as_bytes())?;
        }
    }
    writer.write_all(b"\"")
}

fn write_laminar_simple_residual_csv(
    report: &LaminarSimpleReport,
    output_root: &SafeOutputRoot,
    path: &Path,
) -> std::io::Result<()> {
    let file = open_replace_output_file(output_root, path)?;
    let mut writer = BufWriter::new(file);

    writeln!(
        writer,
        "iteration,continuityBeforeL2,continuityAfterL2,momentumInitialResidualNormalized,momentumFinalResidualNormalized,momentumResidualNorm,pressureInitialResidualNormalized,pressureFinalResidualNormalized,pressureResidualNorm,residualControlConfigured,residualControlChecked,residualControlSatisfied,pressureCorrectionAccepted,momentumLinearIterations,momentumLinearConverged,pressureLinearIterations,pressureLinearConverged,relativeVelocityChangeL2,relativePressureChangeL2,momentumComponentLinearSolveDiagnostics,pressureCorrectionLinearSolveDiagnostics"
    )?;
    for item in &report.history {
        let pressure_linear_solves = pressure_linear_solve_summaries(item)?;
        writeln!(
            writer,
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
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
            item.relative_pressure_change_l2,
            format_momentum_linear_solve_diagnostics(&item.momentum_component_linear_solves, ";"),
            format_pressure_linear_solve_diagnostics(pressure_linear_solves, ";")
        )?;
    }

    writer.flush()
}

fn pressure_linear_solve_summaries(
    item: &LaminarSimpleIterationSummary,
) -> std::io::Result<&[LinearSolveConvergenceSummary]> {
    item.pressure_correction_linear_solves
        .get(..item.pressure_linear_solves)
        .ok_or_else(|| {
            Error::new(
                ErrorKind::InvalidData,
                format!(
                    "SIMPLE iteration {} reports {} pressure solves, exceeding telemetry capacity {}",
                    item.iteration,
                    item.pressure_linear_solves,
                    item.pressure_correction_linear_solves.len()
                ),
            )
        })
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
        println!(
            "incompressibleFluid SIMPLE outer convergence: CONVERGED after {} iteration(s); all configured residualControl criteria are satisfied.",
            report.simple_iterations
        );
        return;
    }

    match report.stop_reason {
        LaminarSimpleStopReason::MaxIterationsReached => {
            println!("incompressibleFluid SIMPLE outer convergence: NOT REACHED.");
            println!(
                "  reason: maximum SIMPLE iteration count reached before all configured residualControl criteria were satisfied."
            );
            print_outer_residual_comparison(
                "U",
                report.final_momentum_initial_normalized_residual_norm,
                options.momentum_residual_control,
            );
            print_outer_residual_comparison(
                "p",
                report.final_pressure_correction_initial_normalized_residual_norm,
                options.pressure_residual_control,
            );
            let reached_budget = options.max_simple_iterations == report.simple_iterations;
            let budget_message = if reached_budget {
                format!(
                    "iteration budget reached ({})",
                    options.max_simple_iterations
                )
            } else {
                "iteration budget stopped".to_string()
            };
            println!("incompressibleFluid convergence note: {budget_message}.");

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
            println!("incompressibleFluid SIMPLE outer convergence: NOT EVALUATED.");
            println!(
                "  reason: residualControl is not configured in system/fvSolution; the run stopped at its configured iteration limit."
            );
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
                "incompressibleFluid convergence note: no active convergence criteria (no residualControl in fvSolution)."
            );
            println!(
                "  to stop early, set SIMPLE.residualControl U/p in system/fvSolution. Solution acceptance is evaluated externally."
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
                "incompressibleFluid convergence note: momentum equation linear solve entered invalid state."
            );
        }
        LaminarSimpleStopReason::PressureSolverInvalidState => {
            println!(
                "incompressibleFluid convergence note: pressure equation linear solve entered invalid state."
            );
        }
        LaminarSimpleStopReason::SolverInvalidState => {
            println!(
                "incompressibleFluid convergence note: solver encountered a non-finite field/state."
            );
        }
        LaminarSimpleStopReason::Converged => {}
    }
}

fn print_outer_residual_comparison(field: &str, initial_residual: f64, tolerance: Option<f64>) {
    match tolerance {
        Some(tolerance) => println!(
            "  {field}: initialResidual={} {} tolerance={}",
            format_scientific(initial_residual),
            if initial_residual < tolerance {
                "<"
            } else {
                ">="
            },
            format_scientific(tolerance),
        ),
        None => println!("  {field}: residualControl not configured"),
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

fn write_laminar_simple_residual_plot(
    output_root: &SafeOutputRoot,
    csv_path: &Path,
    plot_path: &Path,
) -> std::io::Result<()> {
    let wants_svg = plot_path
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("svg"));
    if !wants_svg {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "native residual plots require an output path with the .svg extension",
        ));
    }

    write_laminar_simple_residual_plot_svg(output_root, csv_path, plot_path)
}

fn write_laminar_simple_residual_plot_svg(
    output_root: &SafeOutputRoot,
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

    let mut file = open_replace_output_file(output_root, plot_path)?;
    file.write_all(svg.as_bytes())
}

fn resolve_laminar_simple_options(
    plan: &SolverCasePlan,
    solve: &LaminarSimpleSolveArgs,
) -> Result<LaminarSimpleOptions, String> {
    if !plan.numerics.fv_solution.present {
        return Err("Laminar SIMPLE solve requires FOAM dictionary system/fvSolution".to_string());
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
    let momentum_linear_relative_tolerance =
        fv_solution_number(plan, "solvers.U", "relTol")?.unwrap_or(0.0);
    let pressure_linear_relative_tolerance =
        fv_solution_number(plan, "solvers.p", "relTol")?.unwrap_or(0.0);
    let momentum_case_max_iterations = fv_solution_usize(plan, "solvers.U", "maxIter")?;
    let pressure_case_max_iterations = fv_solution_usize(plan, "solvers.p", "maxIter")?;
    if solve.momentum_max_linear_iterations.is_none() && solve.max_linear_iterations.is_none() {
        validate_fv_solution_max_iterations("solvers.U", momentum_case_max_iterations)?;
    }
    if solve.pressure_max_linear_iterations.is_none() && solve.max_linear_iterations.is_none() {
        validate_fv_solution_max_iterations("solvers.p", pressure_case_max_iterations)?;
    }

    let linear_tolerance = solve
        .linear_tolerance
        .or(momentum_case_tolerance)
        .unwrap_or(FERRUM_DEFAULT_LDU_TOLERANCE);
    let max_linear_iterations = solve
        .max_linear_iterations
        .unwrap_or(FERRUM_DEFAULT_LDU_MAX_ITERATIONS);
    let momentum_linear_tolerance = solve
        .momentum_linear_tolerance
        .or(solve.linear_tolerance)
        .or(momentum_case_tolerance)
        .unwrap_or(FERRUM_DEFAULT_LDU_TOLERANCE);
    let pressure_linear_tolerance = solve
        .pressure_linear_tolerance
        .or(solve.linear_tolerance)
        .or(pressure_case_tolerance)
        .unwrap_or(FERRUM_DEFAULT_LDU_TOLERANCE);
    let momentum_max_linear_iterations = solve
        .momentum_max_linear_iterations
        .or(solve.max_linear_iterations)
        .or(momentum_case_max_iterations)
        .unwrap_or(FERRUM_DEFAULT_LDU_MAX_ITERATIONS);
    let pressure_max_linear_iterations = solve
        .pressure_max_linear_iterations
        .or(solve.max_linear_iterations)
        .or(pressure_case_max_iterations)
        .unwrap_or(FERRUM_DEFAULT_LDU_MAX_ITERATIONS);
    let momentum_linear_solver = match solve.momentum_linear_solver.or(solve.linear_solver) {
        Some(solver) => solver,
        None => required_fv_solution_laminar_solver(plan, "solvers.U")?,
    };
    let pressure_linear_solver = match solve.pressure_linear_solver.or(solve.linear_solver) {
        Some(solver) => solver,
        None => required_fv_solution_laminar_solver(plan, "solvers.p")?,
    };
    if momentum_linear_solver == LaminarSimpleLinearSolver::Gamg {
        return Err(
            "GAMG is implemented for the symmetric SIMPLE pressure equation only; select a nonsymmetric momentum solver in solvers.U"
                .to_string(),
        );
    }
    if solve.profile_gamg && pressure_linear_solver != LaminarSimpleLinearSolver::Gamg {
        return Err("--profileGamg requires solvers.p.solver GAMG".to_string());
    }
    if solve.profile_pcg && pressure_linear_solver != LaminarSimpleLinearSolver::Pcg {
        return Err("--profilePcg requires solvers.p.solver PCG".to_string());
    }
    validate_openfoam_linear_controls(plan, "solvers.U", momentum_linear_solver)?;
    validate_openfoam_linear_controls(plan, "solvers.p", pressure_linear_solver)?;
    let pressure_gamg_options = if pressure_linear_solver == LaminarSimpleLinearSolver::Gamg {
        let mut options = openfoam_gamg_options(plan, "solvers.p")?;
        options.max_iterations = pressure_max_linear_iterations;
        options.tolerance = pressure_linear_tolerance;
        Some(options)
    } else {
        None
    };
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
    let case_derived_simple_iterations = solve.max_simple_iterations.is_none();
    let max_simple_iterations = solve
        .max_simple_iterations
        .or(plan.run.estimated_steps)
        .filter(|iterations| *iterations > 0)
        .ok_or_else(|| {
            "Laminar SIMPLE requires --maxSimpleIterations or a positive controlDict endTime/deltaT iteration count"
                .to_string()
        })?;
    if case_derived_simple_iterations && max_simple_iterations > FERRUM_MAX_CASE_SIMPLE_ITERATIONS {
        return Err(format!(
            "controlDict-derived SIMPLE iteration count {max_simple_iterations} exceeds Ferrum's case-file safety cap of {FERRUM_MAX_CASE_SIMPLE_ITERATIONS}; use --maxSimpleIterations for trusted cases that need a higher limit"
        ));
    }
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
        momentum_execution: solve
            .momentum_execution
            .unwrap_or(LaminarSimpleMomentumExecution::Serial),
        pressure_linear_solver,
        momentum_preconditioner,
        pressure_preconditioner,
        pressure_gamg_options,
        profile_gamg: solve.profile_gamg,
        linear_tolerance,
        max_linear_iterations,
        momentum_linear_tolerance,
        pressure_linear_tolerance,
        momentum_linear_relative_tolerance,
        pressure_linear_relative_tolerance,
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
        return Err("Laminar SIMPLE solve requires FOAM dictionary system/fvSchemes".to_string());
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

    let gradient_schemes = GradientSchemeAliasIndex::new(&plan.numerics.fv_schemes);
    let grad_p = gradient_schemes.resolve("grad(p)", Some("default"))?;
    let grad_u = gradient_schemes.resolve("grad(U)", Some("default"))?;

    Ok(LaminarSimpleSchemes {
        grad_p: parse_laminar_simple_gradient_scheme(&grad_p)?,
        grad_u: parse_laminar_simple_gradient_scheme(&grad_u)?,
        div_phi_u: resolve_laminar_simple_convection_scheme_with_index(plan, &gradient_schemes)?,
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
    } else if scheme_tokens_are(&tokens, &["leastsquares"]) {
        Ok(LaminarSimpleGradientScheme::LeastSquares)
    } else if tokens.len() == 4
        && scheme_token_is(&tokens, 0, "celllimited")
        && scheme_token_is(&tokens, 1, "gauss")
        && scheme_token_is(&tokens, 2, "linear")
    {
        let coefficient = tokens[3].parse::<f64>().map_err(|_| {
            format!(
                "cellLimited Gauss linear coefficient must be finite and in [0, 1], got '{}'",
                tokens[3]
            )
        })?;
        if !coefficient.is_finite() || !(0.0..=1.0).contains(&coefficient) {
            return Err(format!(
                "cellLimited Gauss linear coefficient must be finite and in [0, 1], got '{}'",
                tokens[3]
            ));
        }
        let coefficient = if coefficient == 0.0 { 0.0 } else { coefficient };
        Ok(LaminarSimpleGradientScheme::CellLimitedGaussLinear(
            coefficient,
        ))
    } else {
        Err(format!(
            "unsupported laminar SIMPLE grad scheme '{value}'; currently supported: Gauss linear, leastSquares, or cellLimited Gauss linear k"
        ))
    }
}

#[derive(Clone, Copy)]
struct IndexedGradientScheme<'a> {
    value: &'a str,
    count: usize,
    id: usize,
}

struct GradientSchemeAliasIndex<'a> {
    entries: BTreeMap<&'a str, IndexedGradientScheme<'a>>,
}

impl<'a> GradientSchemeAliasIndex<'a> {
    fn new(dictionary: &'a SolverNumericsDictionaryPlan) -> Self {
        // The ordered index gives attacker-controlled names a deterministic
        // logarithmic lookup bound without changing the accepted alias depth.
        let mut entries = BTreeMap::new();
        for entry in dictionary
            .entries
            .iter()
            .filter(|entry| entry.section == "gradSchemes")
        {
            let next_id = entries.len();
            match entries.entry(entry.key.as_str()) {
                Entry::Vacant(slot) => {
                    slot.insert(IndexedGradientScheme {
                        value: entry.value.as_str(),
                        count: 1,
                        id: next_id,
                    });
                }
                Entry::Occupied(mut slot) => slot.get_mut().count += 1,
            }
        }
        Self { entries }
    }

    fn resolve(&self, key: &str, fallback_key: Option<&str>) -> Result<String, String> {
        let section = "gradSchemes";
        let selected_key = self
            .entries
            .get_key_value(key)
            .map(|(stored, _)| *stored)
            .or_else(|| {
                fallback_key.and_then(|fallback| {
                    self.entries
                        .get_key_value(fallback)
                        .map(|(stored, _)| *stored)
                })
            })
            .ok_or_else(|| match fallback_key {
                Some(fallback) => format!("fvSchemes {section} requires {key} or {fallback}"),
                None => format!("fvSchemes {section} requires {key}"),
            })?;

        let mut current = selected_key;
        let mut visited = vec![false; self.entries.len()];
        let mut path = Vec::new();
        loop {
            let Some((stored_key, entry)) = self.entries.get_key_value(current) else {
                return Err(format!(
                    "fvSchemes gradSchemes alias '${current}' references a missing entry"
                ));
            };
            if visited[entry.id] {
                path.push(*stored_key);
                return Err(format!(
                    "fvSchemes gradSchemes alias cycle: {}",
                    path.join(" -> ")
                ));
            }
            visited[entry.id] = true;
            path.push(*stored_key);
            if entry.count != 1 {
                return Err(format!(
                    "fvSchemes gradSchemes entry '{current}' must be unique, found {}",
                    entry.count
                ));
            }

            let raw_value = entry.value;
            let value = raw_value.trim().trim_end_matches(';').trim();
            if !value.contains('$') {
                return Ok(value.to_string());
            }
            let mut tokens = value.split_whitespace();
            let Some(token) = tokens.next() else {
                return Err(format!(
                    "fvSchemes gradSchemes alias in '{current}' must be one token, got '{raw_value}'"
                ));
            };
            if tokens.next().is_some() {
                return Err(format!(
                    "fvSchemes gradSchemes alias in '{current}' must be one token, got '{raw_value}'"
                ));
            }
            let Some(reference) = token.strip_prefix('$') else {
                return Err(format!(
                    "fvSchemes gradSchemes entry '{current}' contains an embedded alias '{token}'"
                ));
            };
            if reference.is_empty() {
                return Err(format!(
                    "fvSchemes gradSchemes entry '{current}' contains a bare '$' alias"
                ));
            }
            if reference.contains('$')
                || reference.contains('.')
                || reference.contains('/')
                || reference.contains(':')
                || reference.contains('{')
                || reference.contains('}')
            {
                return Err(format!(
                    "fvSchemes gradSchemes alias '{token}' in '{current}' is not an exact same-section reference"
                ));
            }
            current = reference;
        }
    }
}

#[cfg(test)]
fn resolved_gradient_scheme_value(
    plan: &SolverCasePlan,
    key: &str,
    fallback_key: Option<&str>,
) -> Result<String, String> {
    GradientSchemeAliasIndex::new(&plan.numerics.fv_schemes).resolve(key, fallback_key)
}

fn parse_laminar_simple_convection_scheme(
    value: &str,
) -> Result<LaminarSimpleConvectionScheme, String> {
    let tokens = normalized_scheme_tokens(value);
    if scheme_tokens_are(&tokens, &["gauss", "upwind"]) {
        Ok(LaminarSimpleConvectionScheme::GaussUpwind)
    } else if scheme_tokens_are(&tokens, &["gauss", "linearupwind", "grad(u)"]) {
        Ok(LaminarSimpleConvectionScheme::GaussLinearUpwind)
    } else if scheme_tokens_are(&tokens, &["none"]) {
        Err(
            "laminar SIMPLE requires divSchemes.div(phi,U); divSchemes default none is not executable"
                .to_string(),
        )
    } else {
        Err(format!(
            "unsupported laminar SIMPLE div(phi,U) scheme '{value}'; currently supported: Gauss upwind, Gauss linearUpwind grad(U), or bounded Gauss linearUpwind limited"
        ))
    }
}

fn resolve_laminar_simple_convection_scheme_with_index(
    plan: &SolverCasePlan,
    gradient_schemes: &GradientSchemeAliasIndex<'_>,
) -> Result<LaminarSimpleConvectionScheme, String> {
    let value = required_fv_scheme(plan, "divSchemes", "div(phi,U)", Some("default"))?;
    let tokens = normalized_scheme_tokens(value);
    if scheme_tokens_are(&tokens, &["bounded", "gauss", "linearupwind", "limited"]) {
        let limited = gradient_schemes.resolve("limited", None)?;
        return Ok(LaminarSimpleConvectionScheme::BoundedGaussLinearUpwind(
            parse_laminar_simple_gradient_scheme(&limited)?,
        ));
    }
    parse_laminar_simple_convection_scheme(value)
}

#[cfg(test)]
fn resolve_laminar_simple_convection_scheme(
    plan: &SolverCasePlan,
) -> Result<LaminarSimpleConvectionScheme, String> {
    let gradient_schemes = GradientSchemeAliasIndex::new(&plan.numerics.fv_schemes);
    resolve_laminar_simple_convection_scheme_with_index(plan, &gradient_schemes)
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
    if !scheme_token_is(&tokens, 0, "gauss")
        || !scheme_token_is(&tokens, 1, "linear")
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
            "fvSolution {SECTION} entries must be single scalar values for steady SIMPLE convergence; nested dictionary '{}' is not valid FOAM dictionary syntax here",
            nested.path
        ));
    }

    if let Some(entry) = dictionary
        .entries
        .iter()
        .find(|entry| entry.section == SECTION && entry.key != "U" && entry.key != "p")
    {
        return Err(format!(
            "fvSolution {SECTION}.{} is not supported by incompressibleFluid SIMPLE; supported solved fields are U and p",
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

fn openfoam_gamg_options(plan: &SolverCasePlan, section: &str) -> Result<GamgOptions, String> {
    let smoother = numerics_dictionary_value(&plan.numerics.fv_solution, section, "smoother")
        .ok_or_else(|| format!("fvSolution {section} GAMG requires a smoother entry"))?;
    let smoother = match smoother.trim().trim_end_matches(';') {
        "GaussSeidel" | "gaussSeidel" => GamgSmoother::GaussSeidel,
        "symGaussSeidel" => GamgSmoother::SymGaussSeidel,
        other => {
            return Err(format!(
                "unsupported fvSolution {section} GAMG smoother '{other}'; the matrix foundation supports GaussSeidel and symGaussSeidel"
            ));
        }
    };
    let agglomerator = match numerics_dictionary_value(
        &plan.numerics.fv_solution,
        section,
        "agglomerator",
    )
    .map(|value| value.trim().trim_end_matches(';'))
    .unwrap_or("faceAreaPair")
    {
        "algebraicPair" => GamgAgglomerator::AlgebraicPair,
        "faceAreaPair" => GamgAgglomerator::FaceAreaPair,
        other => {
            return Err(format!(
                "unsupported fvSolution {section} GAMG agglomerator '{other}'; no agglomerator fallback was applied"
            ));
        }
    };
    let outer_solver = match numerics_dictionary_value(
        &plan.numerics.fv_solution,
        section,
        "outerSolver",
    )
    .map(|value| value.trim().trim_end_matches(';'))
    .unwrap_or("standalone")
    {
        "standalone" => GamgOuterSolver::Standalone,
        "FCG" => GamgOuterSolver::FlexibleCg,
        other => {
            return Err(format!(
                "unsupported fvSolution {section} GAMG outerSolver '{other}'; Ferrum supports exactly standalone and FCG"
            ));
        }
    };

    let mut options = GamgOptions {
        agglomerator,
        smoother,
        outer_solver,
        ..GamgOptions::default()
    };
    options.max_iterations =
        fv_solution_usize(plan, section, "maxIter")?.unwrap_or(FERRUM_DEFAULT_LDU_MAX_ITERATIONS);
    options.min_iterations = fv_solution_usize(plan, section, "minIter")?.unwrap_or(0);
    options.tolerance =
        fv_solution_number(plan, section, "tolerance")?.unwrap_or(FERRUM_DEFAULT_LDU_TOLERANCE);
    options.relative_tolerance = fv_solution_number(plan, section, "relTol")?.unwrap_or(0.0);
    options.cache_agglomeration =
        fv_solution_bool(plan, section, "cacheAgglomeration")?.unwrap_or(true);
    options.n_cells_in_coarsest_level =
        fv_solution_usize(plan, section, "nCellsInCoarsestLevel")?.unwrap_or(10);
    options.merge_levels = fv_solution_usize(plan, section, "mergeLevels")?.unwrap_or(1);
    options.n_pre_sweeps = fv_solution_usize(plan, section, "nPreSweeps")?.unwrap_or(0);
    options.pre_sweeps_level_multiplier =
        fv_solution_usize(plan, section, "preSweepsLevelMultiplier")?.unwrap_or(1);
    options.max_pre_sweeps = fv_solution_usize(plan, section, "maxPreSweeps")?.unwrap_or(4);
    options.n_post_sweeps = fv_solution_usize(plan, section, "nPostSweeps")?.unwrap_or(2);
    options.post_sweeps_level_multiplier =
        fv_solution_usize(plan, section, "postSweepsLevelMultiplier")?.unwrap_or(1);
    options.max_post_sweeps = fv_solution_usize(plan, section, "maxPostSweeps")?.unwrap_or(4);
    options.n_finest_sweeps = fv_solution_usize(plan, section, "nFinestSweeps")?.unwrap_or(2);
    options.interpolate_correction =
        fv_solution_bool(plan, section, "interpolateCorrection")?.unwrap_or(false);
    options.scale_correction = fv_solution_bool(plan, section, "scaleCorrection")?.unwrap_or(true);
    options.direct_solve_coarsest =
        fv_solution_bool(plan, section, "directSolveCoarsest")?.unwrap_or(false);

    if !options.tolerance.is_finite() || options.tolerance < 0.0 {
        return Err(format!(
            "fvSolution {section}.tolerance must be finite and non-negative, got {}",
            options.tolerance
        ));
    }
    if !options.relative_tolerance.is_finite() || options.relative_tolerance < 0.0 {
        return Err(format!(
            "fvSolution {section}.relTol must be finite and non-negative, got {}",
            options.relative_tolerance
        ));
    }
    if options.n_cells_in_coarsest_level == 0 {
        return Err(format!(
            "fvSolution {section}.nCellsInCoarsestLevel must be positive"
        ));
    }
    if options.merge_levels == 0 {
        return Err(format!("fvSolution {section}.mergeLevels must be positive"));
    }
    Ok(options)
}

fn gamg_outer_solver_name(solver: &GamgOuterSolver) -> &'static str {
    match solver {
        GamgOuterSolver::Standalone => "standalone",
        GamgOuterSolver::FlexibleCg => "FCG",
    }
}

fn format_pressure_gamg_console(gamg: &GamgOptions) -> String {
    format!(
        "incompressibleFluid pressureGAMG: agglomerator={} smoother={} cacheAgglomeration={} nCellsInCoarsestLevel={} mergeLevels={} minIter={} maxIter={} tolerance={} relTol={} nPreSweeps={} nPostSweeps={} nFinestSweeps={} interpolateCorrection={} scaleCorrection={} directSolveCoarsest={} outerSolver={}",
        gamg.agglomerator,
        gamg.smoother,
        yes_no(gamg.cache_agglomeration),
        gamg.n_cells_in_coarsest_level,
        gamg.merge_levels,
        gamg.min_iterations,
        gamg.max_iterations,
        format_scientific(gamg.tolerance),
        format_scientific(gamg.relative_tolerance),
        gamg.n_pre_sweeps,
        gamg.n_post_sweeps,
        gamg.n_finest_sweeps,
        yes_no(gamg.interpolate_correction),
        yes_no(gamg.scale_correction),
        yes_no(gamg.direct_solve_coarsest),
        gamg_outer_solver_name(&gamg.outer_solver),
    )
}

fn gamg_outer_profile_counters(profile: &GamgKernelTiming) -> String {
    format!(
        "outerMatrixVectorProducts={} outerReductions={}",
        profile.outer_matrix_vector_products, profile.outer_reductions
    )
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
    if solver == LaminarSimpleLinearSolver::Gamg {
        openfoam_gamg_options(plan, section)?;
        return Ok(());
    }
    if let Some(relative_tolerance) = fv_solution_number(plan, section, "relTol")?
        && (!relative_tolerance.is_finite() || relative_tolerance < 0.0)
    {
        return Err(format!(
            "fvSolution {section}.relTol must be finite and non-negative, got {relative_tolerance}"
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
            "fvSolution {section}.nSweeps={sweeps} is not implemented yet; Ferrum currently supports nSweeps=1"
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

fn validate_fv_solution_max_iterations(
    section: &str,
    max_iterations: Option<usize>,
) -> Result<(), String> {
    if let Some(max_iterations) = max_iterations
        && max_iterations > FERRUM_MAX_CASE_LDU_MAX_ITERATIONS
    {
        return Err(format!(
            "fvSolution {section}.maxIter={max_iterations} exceeds Ferrum's case-file safety cap of {FERRUM_MAX_CASE_LDU_MAX_ITERATIONS}; use --maxIterations or a field-specific CLI override for trusted cases that need a higher limit"
        ));
    }
    Ok(())
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
            "smoothSolver requires a smoother entry in fvSolution; use GaussSeidel or symGaussSeidel for a direct CLI override"
                .to_string(),
        ),
        "PCG" | "pcg" => Ok(LaminarSimpleLinearSolver::Pcg),
        "GAMG" | "gamg" => Ok(LaminarSimpleLinearSolver::Gamg),
        "CG" | "cg" => Ok(LaminarSimpleLinearSolver::Cg),
        "Jacobi" | "jacobi" => Ok(LaminarSimpleLinearSolver::Jacobi),
        other => Err(format!(
            "unsupported fvSolution linear solver '{other}'; no fallback was applied"
        )),
    }
}

fn parse_openfoam_laminar_preconditioner(
    value: &str,
) -> Result<LaminarSimplePreconditioner, String> {
    match value.trim().trim_end_matches(';') {
        "none" | "None" => Ok(LaminarSimplePreconditioner::None),
        "DIC" => Ok(LaminarSimplePreconditioner::Dic),
        "FDIC" => Ok(LaminarSimplePreconditioner::Fdic),
        "incompleteCholesky" | "ic0" | "IC0" => Ok(LaminarSimplePreconditioner::IncompleteCholesky),
        "diagonal" | "Diagonal" => Ok(LaminarSimplePreconditioner::Diagonal),
        "DILU" => Err(
            "DILU is not implemented yet; refusing to substitute a diagonal preconditioner"
                .to_string(),
        ),
        other => Err(format!(
            "unsupported fvSolution preconditioner '{other}'; no fallback was applied"
        )),
    }
}

fn run_scalar_diffusion_solve(
    plan: &mut SolverCasePlan,
    solve: &ScalarDiffusionSolveArgs,
) -> Result<(), String> {
    let field = find_field_selection(&plan.initial_fields, &solve.field)?;
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
                && buffer
                    .values
                    .as_ref()
                    .is_some_and(|values| values.len() == plan.runtime_data.mesh.cells)
        })
        .and_then(|buffer| buffer.values.as_deref())
}

#[derive(Clone, Copy)]
struct QualifiedName<'a> {
    region: Option<&'a str>,
    name: &'a str,
}

impl fmt::Display for QualifiedName<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(region) = self.region {
            write!(formatter, "{region}/")?;
        }
        formatter.write_str(self.name)
    }
}

fn qualified_name<'a>(region: Option<&'a str>, name: &'a str) -> QualifiedName<'a> {
    QualifiedName { region, name }
}

fn field_label(field: &FieldFile) -> QualifiedName<'_> {
    qualified_name(field.region.as_deref(), &field.name)
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

fn print_solver_case_plan(plan: &SolverCasePlan, dispatch: Option<&SolverDispatch>) {
    println!("Ferrum solver preflight");
    println!("case: {}", plan.case_dir.display());
    println!(
        "control: application={} solver={} startFrom={} startTime={} stopAt={} endTime={} deltaT={} writeControl={} writeInterval={}",
        plan.control.application.as_deref().unwrap_or("missing"),
        plan.control.solver.as_deref().unwrap_or("missing"),
        plan.control.start_from,
        format_optional_number(plan.control.start_time),
        plan.control.stop_at,
        format_optional_number(plan.control.end_time),
        format_optional_number(plan.control.delta_t),
        plan.control.write_control,
        format_optional_number(plan.control.write_interval)
    );
    if let Some(dispatch) = dispatch {
        println!(
            "dispatch: module={} source={}",
            dispatch.module, dispatch.source
        );
    }
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
            println!(
                "  {}: class={} boundaryPatches={}",
                qualified_name(field.region.as_deref(), &field.name),
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
        "solver execution: the incompressibleFluid steady laminar SIMPLE CPU kernel is available; scalar diffusion remains a developer utility; GPU equation kernels are planned"
    );
}

fn print_openfoam_case_compatibility_warnings(warnings: &[String]) {
    let compatibility_count = warnings
        .iter()
        .filter(|warning| warning.starts_with("Ferrum case compatibility: "))
        .count();
    if compatibility_count == 0 {
        println!("Ferrum case compatibility: case layout and required fields look present");
        return;
    }

    println!(
        "Ferrum case compatibility: {} item(s) to check",
        compatibility_count
    );
    for message in warnings
        .iter()
        .filter_map(|warning| warning.strip_prefix("Ferrum case compatibility: "))
    {
        println!("  {}", message);
    }
}

fn print_linear_solver_capabilities() {
    let capabilities = linear_solver_capabilities();
    println!(
        "linear solvers: cpuCsr={} cpuJacobi={} cpuGaussSeidel={} cpuSymGaussSeidel={} cpuCg={} cpuPcg={} cpuBiCgStab={} cpuGamg={} cpuDiagonalPreconditioner={} cpuIncompleteCholeskyPreconditioner={} cpuDicPreconditioner={} cpuFdicPreconditioner={} gpuLinearSolvers={}",
        yes_no(capabilities.cpu_csr),
        yes_no(capabilities.cpu_jacobi),
        yes_no(capabilities.cpu_gauss_seidel),
        yes_no(capabilities.cpu_symmetric_gauss_seidel),
        yes_no(capabilities.cpu_conjugate_gradient),
        yes_no(capabilities.cpu_preconditioned_conjugate_gradient),
        yes_no(capabilities.cpu_bicgstab),
        yes_no(capabilities.cpu_gamg),
        yes_no(capabilities.cpu_diagonal_preconditioner),
        yes_no(capabilities.cpu_incomplete_cholesky_preconditioner),
        yes_no(capabilities.cpu_dic_preconditioner),
        yes_no(capabilities.cpu_fdic_preconditioner),
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
        println!(
            "  {}: sections={} entries={}",
            qualified_name(dictionary.region.as_deref(), &dictionary.name),
            dictionary.sections,
            dictionary.entries
        );
    }
    for entry in &plan.entries {
        if let Some(section) = &entry.section {
            println!(
                "    {}.{}.{}={}",
                entry.dictionary, section, entry.key, entry.value
            );
        } else {
            println!("    {}.{}={}", entry.dictionary, entry.key, entry.value);
        }
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
        println!(
            "  {}: class={} kind={} meshCells={} internal={} values={} expected={} valid={} components={} scalarSlots={} bytesF64={} uniform={} loadedScalars={} boundaryPatches={}/{} cpu={} gpu={} storage={} cpuBuffer={} cpuBufferStatus={}",
            qualified_name(field.region.as_deref(), &field.name),
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
            format_optional_usize(field.internal_field.loaded_scalars),
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
            println!(
                "  {}: kind={} components={} scalarSlots={} bytesF64={} values={}",
                qualified_name(field.region.as_deref(), &field.name),
                field.kind,
                field.components,
                field.scalar_slots,
                field.bytes_f64,
                field.values.as_ref().map_or(0, Vec::len)
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
        println!(
            "  field {}: kind={} internal={} values={} expected={} components={} scalarSlots={} bytesF64={} uniform={} loadedScalars={} cpu={} gpu={} storage={} cpuBuffer={} cpuBufferStatus={}",
            qualified_name(field.region.as_deref(), &field.name),
            field.kind,
            field.internal_field.kind,
            format_optional_usize(field.internal_field.value_count),
            format_optional_usize(field.internal_field.expected_count),
            format_optional_usize(field.storage.components),
            format_optional_usize(field.storage.scalar_slots),
            format_optional_usize(field.storage.bytes_f64),
            format_optional_f64_list(field.internal_field.uniform_components.as_deref()),
            format_optional_usize(field.internal_field.loaded_scalars),
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

fn write_solver_plan_json(
    plan: &SolverCasePlan,
    dispatch: Option<&SolverDispatch>,
    path: &Path,
) -> std::io::Result<()> {
    let trusted_root = env::current_dir()?;
    write_solver_plan_json_in_root(plan, dispatch, &trusted_root, path)
}

fn write_solver_plan_json_in_root(
    plan: &SolverCasePlan,
    dispatch: Option<&SolverDispatch>,
    trusted_root: &Path,
    path: &Path,
) -> std::io::Result<()> {
    let output = SafeOutputRoot::open_existing(trusted_root)?;
    let file = open_create_new_operator_output_file(&output, path)?;
    let mut writer = BufWriter::new(file);

    writeln!(writer, "{{")?;
    write_json_number_field(&mut writer, 2, "schemaVersion", 2)?;
    writeln!(writer, ",")?;
    write_json_key(&mut writer, 2, "caseDir")?;
    write_json_string(&mut writer, &plan.case_dir.display().to_string())?;
    writeln!(writer, ",")?;
    write_json_control(&mut writer, plan)?;
    writeln!(writer, ",")?;
    write_json_dispatch(&mut writer, dispatch)?;
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

#[derive(Clone, Copy, Debug, PartialEq)]
struct LaminarSimpleReportRunMetadata {
    profile_pcg: bool,
    wall_clock_seconds: f64,
}

fn write_laminar_simple_report_json(
    plan: &SolverCasePlan,
    options: &LaminarSimpleOptions,
    report: &LaminarSimpleReport,
    post_processing: &LaminarSimplePostProcessing,
    run: LaminarSimpleReportRunMetadata,
    output_root: &SafeOutputRoot,
    path: &Path,
) -> std::io::Result<()> {
    let file = open_replace_output_file(output_root, path)?;
    let mut writer = BufWriter::new(file);

    writeln!(writer, "{{")?;
    write_json_number_field(&mut writer, 2, "schemaVersion", 3)?;
    writeln!(writer, ",")?;
    write_json_key(&mut writer, 2, "caseDir")?;
    write_json_string(&mut writer, &plan.case_dir.display().to_string())?;
    writeln!(writer, ",")?;
    write_json_key(&mut writer, 2, "solver")?;
    write_json_string(&mut writer, "incompressibleFluid")?;
    writeln!(writer, ",")?;
    write_json_key(&mut writer, 2, "algorithm")?;
    write_json_string(&mut writer, "SIMPLE")?;
    writeln!(writer, ",")?;
    write_json_key(&mut writer, 2, "regime")?;
    write_json_string(&mut writer, "laminar")?;
    writeln!(writer, ",")?;
    write_json_key(&mut writer, 2, "backend")?;
    write_json_string(&mut writer, "cpu")?;
    writeln!(writer, ",")?;
    write_json_laminar_simple_options(&mut writer, options, run.profile_pcg)?;
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
    write_json_optional_number(&mut writer, Some(run.wall_clock_seconds))?;
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
    write_json_outer_convergence(&mut writer, report)?;
    writeln!(writer, ",")?;
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
    write_json_linear_nonconvergence_diagnostics(&mut writer, report)?;
    writeln!(writer, ",")?;
    write_json_laminar_simple_timing_summary(&mut writer, report, run.wall_clock_seconds)?;
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
    write_json_continuity_errors(&mut writer, post_processing.continuity_errors.as_ref())?;
    writeln!(writer, ",")?;
    write_json_wall_forces(&mut writer, post_processing.wall_forces.as_ref())?;
    writeln!(writer, ",")?;
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

fn write_json_outer_convergence(
    writer: &mut impl Write,
    report: &LaminarSimpleReport,
) -> std::io::Result<()> {
    write_json_key(writer, 2, "outerConvergence")?;
    writeln!(writer, "{{")?;
    write_json_key(writer, 4, "status")?;
    write_json_string(writer, outer_convergence_status(report))?;
    writeln!(writer, ",")?;
    write_json_bool_field(writer, 4, "configured", report.residual_control.configured)?;
    writeln!(writer, ",")?;
    write_json_bool_field(writer, 4, "evaluated", report.residual_control.checked)?;
    writeln!(writer, ",")?;
    write_json_bool_field(writer, 4, "converged", report.converged)?;
    writeln!(writer, ",")?;
    write_json_key(writer, 4, "reason")?;
    write_json_string(writer, &report.stop_reason.to_string())?;
    writeln!(writer)?;
    write_indent(writer, 2)?;
    write!(writer, "}}")
}

fn write_json_laminar_simple_timing_summary(
    writer: &mut impl Write,
    report: &LaminarSimpleReport,
    driver_measured_seconds: f64,
) -> std::io::Result<()> {
    write_json_key(writer, 2, "timing")?;
    writeln!(writer, "{{")?;
    let fields = [
        ("solverTotalSeconds", report.timing.solver_total_seconds),
        ("driverMeasuredSeconds", driver_measured_seconds),
        ("setupSeconds", report.timing.setup_seconds),
        (
            "iterationSetupSeconds",
            report.timing.iteration_setup_seconds,
        ),
        (
            "operatorEvaluationSeconds",
            report.timing.operator_evaluation_seconds,
        ),
        (
            "momentumAssemblySeconds",
            report.timing.momentum_assembly_seconds,
        ),
        (
            "momentumGradientSeconds",
            report.timing.momentum_gradient_seconds,
        ),
        (
            "momentumMatrixFillSeconds",
            report.timing.momentum_matrix_fill_seconds,
        ),
        (
            "momentumLinearSolveSeconds",
            report.timing.momentum_linear_solve_seconds,
        ),
        (
            "pressureCouplingSetupSeconds",
            report.timing.pressure_coupling_setup_seconds,
        ),
        (
            "pressureAssemblySeconds",
            report.timing.pressure_assembly_seconds,
        ),
        (
            "pressureLinearSolveSeconds",
            report.timing.pressure_linear_solve_seconds,
        ),
        (
            "pressurePcgTotalSeconds",
            report.timing.pressure_pcg_total_seconds,
        ),
        (
            "pressurePreconditionerUpdateSeconds",
            report.timing.pressure_preconditioner_update_seconds,
        ),
        (
            "pressureMatrixVectorSeconds",
            report.timing.pressure_matrix_vector_seconds,
        ),
        (
            "pressurePreconditionerApplicationSeconds",
            report.timing.pressure_preconditioner_application_seconds,
        ),
        (
            "pressureVectorOperationSeconds",
            report.timing.pressure_vector_operation_seconds,
        ),
        (
            "pressurePcgOtherSeconds",
            report.timing.pressure_pcg_other_seconds,
        ),
        (
            "pressureMatrixVectorProducts",
            report.timing.pressure_matrix_vector_products as f64,
        ),
        (
            "pressurePreconditionerApplications",
            report.timing.pressure_preconditioner_applications as f64,
        ),
        (
            "fieldCorrectionSeconds",
            report.timing.field_correction_seconds,
        ),
        ("finalizationSeconds", report.timing.finalization_seconds),
        (
            "otherSolverWorkSeconds",
            report.timing.other_solver_work_seconds,
        ),
    ];
    for (name, value) in fields {
        write_json_key(writer, 4, name)?;
        write_json_optional_number(writer, Some(value))?;
        writeln!(writer, ",")?;
    }
    write_json_gamg_profile(writer, 4, report.timing.pressure_gamg_profile.as_ref())?;
    writeln!(writer)?;
    write_indent(writer, 2)?;
    write!(writer, "}}")
}

fn write_json_gamg_profile(
    writer: &mut impl Write,
    indent: usize,
    profile: Option<&GamgKernelTiming>,
) -> std::io::Result<()> {
    write_json_key(writer, indent, "pressureGamgProfile")?;
    let Some(profile) = profile else {
        return write!(writer, "null");
    };
    writeln!(writer, "{{")?;
    let nested = indent + 2;
    let seconds = [
        ("totalSeconds", profile.total_seconds),
        ("hierarchyBuildSeconds", profile.hierarchy_build_seconds),
        ("hierarchyRebuildSeconds", profile.hierarchy_rebuild_seconds),
        (
            "hierarchyDiagnosticSeconds",
            profile.hierarchy_diagnostic_seconds,
        ),
        ("matrixRefreshSeconds", profile.matrix_refresh_seconds),
        ("finestResidualSeconds", profile.finest_residual_seconds),
        ("vCycleSeconds", profile.v_cycle_seconds),
        ("restrictionSeconds", profile.restriction_seconds()),
        ("prolongationSeconds", profile.prolongation_seconds()),
        ("smoothingSeconds", profile.smoothing_seconds()),
        ("scalingSeconds", profile.scaling_seconds()),
        ("coarseResidualSeconds", profile.coarse_residual_seconds()),
        ("correctionSeconds", profile.correction_seconds()),
        ("coarsestSolveSeconds", profile.coarsest_solve_seconds()),
        ("vCycleOtherSeconds", profile.v_cycle_other_seconds()),
        ("otherSeconds", profile.other_seconds),
    ];
    for (name, value) in seconds {
        write_json_key(writer, nested, name)?;
        write_json_optional_number(writer, Some(value))?;
        writeln!(writer, ",")?;
    }
    let counters = [
        ("hierarchyBuilds", profile.hierarchy_builds),
        ("hierarchyRebuilds", profile.hierarchy_rebuilds),
        ("matrixRefreshes", profile.matrix_refreshes),
        (
            "finestResidualEvaluations",
            profile.finest_residual_evaluations,
        ),
        (
            "outerMatrixVectorProducts",
            profile.outer_matrix_vector_products,
        ),
        ("outerReductions", profile.outer_reductions),
        ("solves", profile.solves),
        ("vCycles", profile.v_cycles),
    ];
    for (name, value) in counters {
        write_json_number_field(writer, nested, name, value)?;
        writeln!(writer, ",")?;
    }
    write_json_gamg_hierarchy(writer, nested, profile)?;
    writeln!(writer, ",")?;
    write_json_key(writer, nested, "levels")?;
    writeln!(writer, "[")?;
    for (index, level) in profile.levels.iter().enumerate() {
        let level_indent = nested + 2;
        write_indent(writer, level_indent)?;
        writeln!(writer, "{{")?;
        write_json_number_field(writer, level_indent + 2, "level", level.level)?;
        writeln!(writer, ",")?;
        write_json_number_field(writer, level_indent + 2, "cells", level.cells)?;
        writeln!(writer, ",")?;
        write_json_number_field(writer, level_indent + 2, "nonzeros", level.nonzeros)?;
        writeln!(writer, ",")?;
        let level_seconds = [
            ("matrixRefreshSeconds", level.matrix_refresh_seconds),
            ("restrictionSeconds", level.restriction_seconds),
            ("prolongationSeconds", level.prolongation_seconds),
            ("smoothingSeconds", level.smoothing_seconds),
            ("scalingSeconds", level.scaling_seconds),
            ("residualSeconds", level.residual_seconds),
            ("correctionSeconds", level.correction_seconds),
            ("coarsestSolveSeconds", level.coarsest_solve_seconds),
        ];
        for (name, value) in level_seconds {
            write_json_key(writer, level_indent + 2, name)?;
            write_json_optional_number(writer, Some(value))?;
            writeln!(writer, ",")?;
        }
        let level_counters = [
            ("matrixRefreshes", level.matrix_refreshes),
            ("restrictionCalls", level.restriction_calls),
            ("prolongationCalls", level.prolongation_calls),
            ("smoothingCalls", level.smoothing_calls),
            ("smoothingSweeps", level.smoothing_sweeps),
            ("scalingCalls", level.scaling_calls),
            ("residualEvaluations", level.residual_evaluations),
            ("correctionUpdates", level.correction_updates),
            ("coarsestSolves", level.coarsest_solves),
            ("coarsestIterations", level.coarsest_iterations),
        ];
        for (counter_index, (name, value)) in level_counters.iter().enumerate() {
            write_json_number_field(writer, level_indent + 2, name, *value)?;
            if counter_index + 1 < level_counters.len() {
                writeln!(writer, ",")?;
            } else {
                writeln!(writer)?;
            }
        }
        write_indent(writer, level_indent)?;
        if index + 1 < profile.levels.len() {
            writeln!(writer, "}},")?;
        } else {
            writeln!(writer, "}}")?;
        }
    }
    write_indent(writer, nested)?;
    writeln!(writer, "]")?;
    write_indent(writer, indent)?;
    write!(writer, "}}")
}

fn write_json_gamg_hierarchy(
    writer: &mut impl Write,
    indent: usize,
    profile: &GamgKernelTiming,
) -> std::io::Result<()> {
    write_json_key(writer, indent, "hierarchy")?;
    let Some(hierarchy) = profile.hierarchy.as_ref() else {
        return write!(writer, "null");
    };

    writeln!(writer, "{{")?;
    let nested = indent + 2;
    write_json_number_field(
        writer,
        nested,
        "smootherPassesPerSweep",
        hierarchy.smoother_passes_per_sweep,
    )?;
    writeln!(writer, ",")?;
    write_json_bool_field(
        writer,
        nested,
        "directSolveCoarsest",
        hierarchy.direct_solve_coarsest,
    )?;
    writeln!(writer, ",")?;

    let grid_complexity = hierarchy.grid_complexity_terms();
    let operator_complexity = hierarchy.operator_complexity_terms();
    for (name, value) in [
        (
            "gridComplexityNumeratorDecimal",
            grid_complexity.map(|terms| terms.0),
        ),
        (
            "gridComplexityDenominatorDecimal",
            grid_complexity.map(|terms| terms.1),
        ),
        (
            "operatorComplexityNumeratorDecimal",
            operator_complexity.map(|terms| terms.0),
        ),
        (
            "operatorComplexityDenominatorDecimal",
            operator_complexity.map(|terms| terms.1),
        ),
        (
            "nnzWeightedSmoothingWorkDecimal",
            profile.nnz_weighted_smoothing_work(),
        ),
        (
            "nnzWeightedSparseWorkDecimal",
            profile.nnz_weighted_sparse_work(),
        ),
    ] {
        write_json_key(writer, nested, name)?;
        write_json_optional_u128_decimal(writer, value)?;
        writeln!(writer, ",")?;
    }

    write_json_key(writer, nested, "levels")?;
    writeln!(writer, "[")?;
    for (index, level) in hierarchy.levels.iter().enumerate() {
        let item_indent = nested + 2;
        write_indent(writer, item_indent)?;
        writeln!(writer, "{{")?;
        write_json_number_field(writer, item_indent + 2, "level", level.level)?;
        writeln!(writer, ",")?;
        write_json_number_field(writer, item_indent + 2, "cells", level.cells)?;
        writeln!(writer, ",")?;
        write_json_number_field(writer, item_indent + 2, "nonzeros", level.nonzeros)?;
        writeln!(writer)?;
        write_indent(writer, item_indent)?;
        if index + 1 < hierarchy.levels.len() {
            writeln!(writer, "}},")?;
        } else {
            writeln!(writer, "}}")?;
        }
    }
    write_indent(writer, nested)?;
    writeln!(writer, "],")?;

    write_json_key(writer, nested, "transfers")?;
    writeln!(writer, "[")?;
    for (index, transfer) in hierarchy.transfers.iter().enumerate() {
        let item_indent = nested + 2;
        write_indent(writer, item_indent)?;
        writeln!(writer, "{{")?;
        for (name, value) in [
            ("fineLevel", transfer.fine_level),
            ("coarseLevel", transfer.coarse_level),
            ("fineCells", transfer.fine_cells),
            ("coarseCells", transfer.coarse_cells),
            ("singletonFineCells", transfer.singleton_fine_cells),
            ("unmatchedFineCells", transfer.unmatched_fine_cells),
            ("minAggregateSize", transfer.min_aggregate_size),
            ("maxAggregateSize", transfer.max_aggregate_size),
        ] {
            write_json_number_field(writer, item_indent + 2, name, value)?;
            writeln!(writer, ",")?;
        }
        write_json_key(writer, item_indent + 2, "aggregateSizeHistogram")?;
        writeln!(writer, "[")?;
        for (bin_index, bin) in transfer.aggregate_size_histogram.iter().enumerate() {
            let bin_indent = item_indent + 4;
            write_indent(writer, bin_indent)?;
            writeln!(writer, "{{")?;
            write_json_number_field(writer, bin_indent + 2, "aggregateSize", bin.aggregate_size)?;
            writeln!(writer, ",")?;
            write_json_number_field(
                writer,
                bin_indent + 2,
                "aggregateCount",
                bin.aggregate_count,
            )?;
            writeln!(writer)?;
            write_indent(writer, bin_indent)?;
            if bin_index + 1 < transfer.aggregate_size_histogram.len() {
                writeln!(writer, "}},")?;
            } else {
                writeln!(writer, "}}")?;
            }
        }
        write_indent(writer, item_indent + 2)?;
        writeln!(writer, "]")?;
        write_indent(writer, item_indent)?;
        if index + 1 < hierarchy.transfers.len() {
            writeln!(writer, "}},")?;
        } else {
            writeln!(writer, "}}")?;
        }
    }
    write_indent(writer, nested)?;
    writeln!(writer, "]")?;
    write_indent(writer, indent)?;
    write!(writer, "}}")
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
    profile_pcg: bool,
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
        "momentumExecution",
        &options.momentum_execution.to_string(),
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
    write_json_pressure_gamg_options(writer, 4, options.pressure_gamg_options)?;
    writeln!(writer, ",")?;
    write_json_bool_field(writer, 4, "profileGamg", options.profile_gamg)?;
    writeln!(writer, ",")?;
    write_json_bool_field(writer, 4, "profilePcg", profile_pcg)?;
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
    write_json_key(writer, 4, "momentumLinearRelativeTolerance")?;
    write_json_optional_number(writer, Some(options.momentum_linear_relative_tolerance))?;
    writeln!(writer, ",")?;
    write_json_key(writer, 4, "pressureLinearTolerance")?;
    write_json_optional_number(writer, Some(options.pressure_linear_tolerance))?;
    writeln!(writer, ",")?;
    write_json_key(writer, 4, "pressureLinearRelativeTolerance")?;
    write_json_optional_number(writer, Some(options.pressure_linear_relative_tolerance))?;
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

fn write_json_pressure_gamg_options(
    writer: &mut impl Write,
    indent: usize,
    options: Option<GamgOptions>,
) -> std::io::Result<()> {
    write_json_key(writer, indent, "pressureGamg")?;
    let Some(options) = options else {
        return write!(writer, "null");
    };
    writeln!(writer, "{{")?;
    let nested = indent + 2;
    write_json_string_field(
        writer,
        nested,
        "outerSolver",
        gamg_outer_solver_name(&options.outer_solver),
    )?;
    writeln!(writer, ",")?;
    write_json_string_field(
        writer,
        nested,
        "agglomerator",
        &options.agglomerator.to_string(),
    )?;
    writeln!(writer, ",")?;
    write_json_string_field(writer, nested, "smoother", &options.smoother.to_string())?;
    writeln!(writer, ",")?;
    write_json_bool_field(
        writer,
        nested,
        "cacheAgglomeration",
        options.cache_agglomeration,
    )?;
    writeln!(writer, ",")?;
    write_json_number_field(
        writer,
        nested,
        "nCellsInCoarsestLevel",
        options.n_cells_in_coarsest_level,
    )?;
    writeln!(writer, ",")?;
    write_json_number_field(writer, nested, "mergeLevels", options.merge_levels)?;
    writeln!(writer, ",")?;
    write_json_number_field(writer, nested, "minIter", options.min_iterations)?;
    writeln!(writer, ",")?;
    write_json_number_field(writer, nested, "maxIter", options.max_iterations)?;
    writeln!(writer, ",")?;
    write_json_key(writer, nested, "tolerance")?;
    write_json_optional_number(writer, Some(options.tolerance))?;
    writeln!(writer, ",")?;
    write_json_key(writer, nested, "relTol")?;
    write_json_optional_number(writer, Some(options.relative_tolerance))?;
    writeln!(writer, ",")?;
    write_json_number_field(writer, nested, "nPreSweeps", options.n_pre_sweeps)?;
    writeln!(writer, ",")?;
    write_json_number_field(
        writer,
        nested,
        "preSweepsLevelMultiplier",
        options.pre_sweeps_level_multiplier,
    )?;
    writeln!(writer, ",")?;
    write_json_number_field(writer, nested, "maxPreSweeps", options.max_pre_sweeps)?;
    writeln!(writer, ",")?;
    write_json_number_field(writer, nested, "nPostSweeps", options.n_post_sweeps)?;
    writeln!(writer, ",")?;
    write_json_number_field(
        writer,
        nested,
        "postSweepsLevelMultiplier",
        options.post_sweeps_level_multiplier,
    )?;
    writeln!(writer, ",")?;
    write_json_number_field(writer, nested, "maxPostSweeps", options.max_post_sweeps)?;
    writeln!(writer, ",")?;
    write_json_number_field(writer, nested, "nFinestSweeps", options.n_finest_sweeps)?;
    writeln!(writer, ",")?;
    write_json_bool_field(
        writer,
        nested,
        "interpolateCorrection",
        options.interpolate_correction,
    )?;
    writeln!(writer, ",")?;
    write_json_bool_field(writer, nested, "scaleCorrection", options.scale_correction)?;
    writeln!(writer, ",")?;
    write_json_bool_field(
        writer,
        nested,
        "directSolveCoarsest",
        options.direct_solve_coarsest,
    )?;
    writeln!(writer)?;
    write_indent(writer, indent)?;
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

fn write_json_continuity_errors(
    writer: &mut impl Write,
    summary: Option<&ContinuityErrorsReport>,
) -> std::io::Result<()> {
    write_json_key(writer, 2, "continuityErrors")?;
    let Some(summary) = summary else {
        return write!(writer, "null");
    };
    writeln!(writer, "{{")?;
    write_json_key(writer, 4, "local")?;
    write_json_optional_number(writer, Some(summary.final_error.local))?;
    writeln!(writer, ",")?;
    write_json_key(writer, 4, "global")?;
    write_json_optional_number(writer, Some(summary.final_error.global))?;
    writeln!(writer, ",")?;
    write_json_key(writer, 4, "cumulativeGlobal")?;
    write_json_optional_number(writer, Some(summary.cumulative_global))?;
    writeln!(writer, ",")?;
    write_json_key(writer, 4, "deltaT")?;
    write_json_optional_number(writer, Some(summary.delta_t))?;
    writeln!(writer, ",")?;
    write_json_key(writer, 4, "totalVolume")?;
    write_json_optional_number(writer, Some(summary.total_volume))?;
    writeln!(writer)?;
    write_indent(writer, 2)?;
    write!(writer, "}}")
}

fn write_json_wall_forces(
    writer: &mut impl Write,
    report: Option<&WallForceReport>,
) -> std::io::Result<()> {
    write_json_key(writer, 2, "wallForces")?;
    let Some(report) = report else {
        return write!(writer, "null");
    };
    let value = &report.coefficients;
    writeln!(writer, "{{")?;
    write_json_key(writer, 4, "patches")?;
    write_json_string_array(writer, &report.patch_names)?;
    writeln!(writer, ",")?;
    write_json_number_field(writer, 4, "selectedPatches", value.selected_patches)?;
    writeln!(writer, ",")?;
    write_json_number_field(writer, 4, "selectedFaces", value.selected_faces)?;
    writeln!(writer, ",")?;
    write_json_key(writer, 4, "pressureFieldKind")?;
    write_json_string(writer, "kinematic")?;
    writeln!(writer, ",")?;
    write_json_key(writer, 4, "pressureReference")?;
    write_json_string(writer, "areaVectorBalancedMean")?;
    writeln!(writer, ",")?;
    write_json_key(writer, 4, "tractionMethod")?;
    write_json_string(writer, WALL_FORCE_TRACTION_METHOD)?;
    writeln!(writer, ",")?;
    write_json_number_field(
        writer,
        4,
        "tractionMethodVersion",
        WALL_FORCE_TRACTION_METHOD_VERSION,
    )?;
    writeln!(writer, ",")?;
    write_json_key(writer, 4, "forceConvention")?;
    write_json_string(writer, WALL_FORCE_FORCE_CONVENTION)?;
    writeln!(writer, ",")?;
    write_json_key(writer, 4, "faceAreaVectorOrientation")?;
    write_json_string(writer, WALL_FORCE_AREA_VECTOR_ORIENTATION)?;
    writeln!(writer, ",")?;
    write_json_key(writer, 4, "pressureFaceTreatment")?;
    write_json_string(writer, WALL_FORCE_PRESSURE_FACE_TREATMENT)?;
    writeln!(writer, ",")?;
    write_json_key(writer, 4, "velocityGradientScheme")?;
    write_json_string(writer, &report.velocity_gradient_scheme.to_string())?;
    writeln!(writer, ",")?;
    write_json_key(writer, 4, "resolvedPressureReference")?;
    write_json_optional_number(writer, Some(value.resolved_pressure_reference))?;
    writeln!(writer, ",")?;
    write_json_key(writer, 4, "referenceSpeed")?;
    write_json_optional_number(writer, Some(report.reference_speed))?;
    writeln!(writer, ",")?;
    write_json_key(writer, 4, "referenceArea")?;
    write_json_optional_number(writer, Some(value.resolved_reference_area))?;
    writeln!(writer, ",")?;
    write_json_key(writer, 4, "referenceDynamicPressure")?;
    write_json_optional_number(writer, Some(value.reference_dynamic_pressure))?;
    writeln!(writer, ",")?;
    write_json_key(writer, 4, "selectedArea")?;
    write_json_optional_number(writer, Some(value.selected_area))?;
    writeln!(writer, ",")?;
    write_json_point(writer, 4, "areaVectorSum", value.area_vector_sum)?;
    writeln!(writer, ",")?;
    write_json_point(writer, 4, "pressureForce", value.pressure_force)?;
    writeln!(writer, ",")?;
    write_json_point(writer, 4, "viscousForce", value.viscous_force)?;
    writeln!(writer, ",")?;
    write_json_point(writer, 4, "totalForce", value.total_force)?;
    writeln!(writer, ",")?;
    write_json_directional_coefficient(writer, 4, "drag", value.drag)?;
    writeln!(writer, ",")?;
    write_json_directional_coefficient(writer, 4, "lift", value.lift)?;
    writeln!(writer)?;
    write_indent(writer, 2)?;
    write!(writer, "}}")
}

fn write_json_point(
    writer: &mut impl Write,
    indent: usize,
    key: &str,
    point: Point3,
) -> std::io::Result<()> {
    write_json_key(writer, indent, key)?;
    writeln!(writer, "{{")?;
    write_json_key(writer, indent + 2, "x")?;
    write_json_optional_number(writer, Some(point.x))?;
    writeln!(writer, ",")?;
    write_json_key(writer, indent + 2, "y")?;
    write_json_optional_number(writer, Some(point.y))?;
    writeln!(writer, ",")?;
    write_json_key(writer, indent + 2, "z")?;
    write_json_optional_number(writer, Some(point.z))?;
    writeln!(writer)?;
    write_indent(writer, indent)?;
    write!(writer, "}}")
}

fn write_json_directional_coefficient(
    writer: &mut impl Write,
    indent: usize,
    key: &str,
    value: ferrum_finite_volume::boundary_forces::DirectionalCoefficient,
) -> std::io::Result<()> {
    write_json_key(writer, indent, key)?;
    writeln!(writer, "{{")?;
    write_json_key(writer, indent + 2, "pressure")?;
    write_json_optional_number(writer, Some(value.pressure))?;
    writeln!(writer, ",")?;
    write_json_key(writer, indent + 2, "viscous")?;
    write_json_optional_number(writer, Some(value.viscous))?;
    writeln!(writer, ",")?;
    write_json_key(writer, indent + 2, "total")?;
    write_json_optional_number(writer, Some(value.total))?;
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

fn write_json_linear_nonconvergence_diagnostics(
    writer: &mut impl Write,
    report: &LaminarSimpleReport,
) -> std::io::Result<()> {
    write_json_key(writer, 2, "linearNonConvergenceDiagnostics")?;
    writeln!(writer, "{{")?;
    write_json_key(writer, 4, "firstMomentum")?;
    write_json_linear_nonconvergence_snapshot(
        writer,
        4,
        report.first_momentum_linear_nonconvergence.as_ref(),
    )?;
    writeln!(writer, ",")?;
    write_json_key(writer, 4, "firstPressure")?;
    write_json_linear_nonconvergence_snapshot(
        writer,
        4,
        report.first_pressure_linear_nonconvergence.as_ref(),
    )?;
    writeln!(writer)?;
    write_indent(writer, 2)?;
    write!(writer, "}}")
}

fn write_json_linear_nonconvergence_snapshot(
    writer: &mut impl Write,
    indent: usize,
    snapshot: Option<&LinearNonConvergenceSnapshot>,
) -> std::io::Result<()> {
    let Some(snapshot) = snapshot else {
        return write!(writer, "null");
    };
    writeln!(writer, "{{")?;
    write_json_number_field(
        writer,
        indent + 2,
        "simpleIteration",
        snapshot.simple_iteration,
    )?;
    writeln!(writer, ",")?;
    write_json_number_field(writer, indent + 2, "solveOrdinal", snapshot.solve_ordinal)?;
    writeln!(writer, ",")?;
    write_json_string_field(writer, indent + 2, "solver", &snapshot.solver.to_string())?;
    writeln!(writer, ",")?;
    write_json_string_field(
        writer,
        indent + 2,
        "preconditioner",
        &snapshot.preconditioner.to_string(),
    )?;
    writeln!(writer, ",")?;
    write_json_bool_field(
        writer,
        indent + 2,
        "linearIterateAccepted",
        snapshot.linear_iterate_accepted,
    )?;
    writeln!(writer, ",")?;
    write_json_key(writer, indent + 2, "solve")?;
    write_json_linear_solve_diagnostic(writer, indent + 2, &snapshot.solve)?;
    writeln!(writer, ",")?;
    write_json_key(writer, indent + 2, "continuityBefore")?;
    write_json_continuity_summary(writer, indent + 2, &snapshot.continuity_before)?;
    writeln!(writer, ",")?;
    write_json_key(writer, indent + 2, "continuityStar")?;
    write_json_optional_continuity_summary(writer, indent + 2, snapshot.continuity_star.as_ref())?;
    writeln!(writer, ",")?;
    write_json_key(writer, indent + 2, "continuityAfter")?;
    write_json_optional_continuity_summary(writer, indent + 2, snapshot.continuity_after.as_ref())?;
    writeln!(writer, ",")?;
    write_json_key(writer, indent + 2, "worstAlgebraicResidualRow")?;
    write_json_algebraic_residual_row(writer, indent + 2, &snapshot.worst_algebraic_residual_row)?;
    writeln!(writer)?;
    write_indent(writer, indent)?;
    write!(writer, "}}")
}

fn write_json_optional_continuity_summary(
    writer: &mut impl Write,
    indent: usize,
    summary: Option<&ContinuitySummary>,
) -> std::io::Result<()> {
    match summary {
        Some(summary) => write_json_continuity_summary(writer, indent, summary),
        None => write!(writer, "null"),
    }
}

fn write_json_algebraic_residual_row(
    writer: &mut impl Write,
    indent: usize,
    diagnostic: &AlgebraicResidualRowDiagnostic,
) -> std::io::Result<()> {
    writeln!(writer, "{{")?;
    write_json_bool_field(
        writer,
        indent + 2,
        "representable",
        diagnostic.representable,
    )?;
    writeln!(writer, ",")?;
    write_json_key(writer, indent + 2, "row")?;
    write_json_optional_usize(writer, diagnostic.row)?;
    for (name, value) in [
        ("cellVolume", diagnostic.cell_volume),
        ("diagonal", diagnostic.diagonal),
        ("rowSumAbs", diagnostic.row_sum_abs),
        ("offDiagonalSumAbs", diagnostic.off_diagonal_sum_abs),
        ("rhs", diagnostic.rhs),
        ("matrixProduct", diagnostic.matrix_product),
        ("residual", diagnostic.residual),
        ("initialValue", diagnostic.initial_value),
        ("candidateValue", diagnostic.candidate_value),
    ] {
        writeln!(writer, ",")?;
        write_json_key(writer, indent + 2, name)?;
        write_json_optional_number(writer, value)?;
    }
    writeln!(writer, ",")?;
    write_json_key(writer, indent + 2, "incidentFaces")?;
    write_json_incident_face_diagnostic_summary(writer, indent + 2, &diagnostic.incident_faces)?;
    writeln!(writer)?;
    write_indent(writer, indent)?;
    write!(writer, "}}")
}

fn write_json_incident_face_diagnostic_summary(
    writer: &mut impl Write,
    indent: usize,
    summary: &IncidentFaceDiagnosticSummary,
) -> std::io::Result<()> {
    writeln!(writer, "{{")?;
    write_json_bool_field(writer, indent + 2, "representable", summary.representable)?;
    writeln!(writer, ",")?;
    write_json_number_field(writer, indent + 2, "count", summary.count)?;
    for (name, value) in [
        ("signedSum", summary.signed_sum),
        ("sumAbs", summary.sum_abs),
        ("min", summary.min),
        ("max", summary.max),
        ("maxAbsFaceFlux", summary.max_abs_face_flux),
    ] {
        writeln!(writer, ",")?;
        write_json_key(writer, indent + 2, name)?;
        write_json_optional_number(writer, value)?;
    }
    for (name, value) in [
        ("maxAbsFaceIndex", summary.max_abs_face_index),
        ("maxAbsFaceOwner", summary.max_abs_face_owner),
        ("maxAbsFaceNeighbour", summary.max_abs_face_neighbour),
        ("maxAbsFacePatchIndex", summary.max_abs_face_patch_index),
    ] {
        writeln!(writer, ",")?;
        write_json_key(writer, indent + 2, name)?;
        write_json_optional_usize(writer, value)?;
    }
    writeln!(writer)?;
    write_indent(writer, indent)?;
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
        let pressure_linear_solves = pressure_linear_solve_summaries(item)?;
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
        writeln!(writer, ",")?;
        write_json_key(writer, 6, "momentumComponentLinearSolves")?;
        write_json_momentum_linear_solve_diagnostics(
            writer,
            6,
            &item.momentum_component_linear_solves,
        )?;
        writeln!(writer, ",")?;
        write_json_key(writer, 6, "pressureCorrectionLinearSolves")?;
        write_json_pressure_linear_solve_diagnostics(writer, 6, pressure_linear_solves)?;
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

fn write_json_linear_solve_diagnostic(
    writer: &mut impl Write,
    indent: usize,
    summary: &LinearSolveConvergenceSummary,
) -> std::io::Result<()> {
    writeln!(writer, "{{")?;
    write_json_number_field(writer, indent + 2, "iterations", summary.iterations)?;
    writeln!(writer, ",")?;
    write_json_bool_field(writer, indent + 2, "converged", summary.converged)?;
    writeln!(writer, ",")?;
    write_json_key(writer, indent + 2, "initialNormalizedResidual")?;
    write_json_optional_number(writer, Some(summary.initial_normalized_residual_norm))?;
    writeln!(writer, ",")?;
    write_json_key(writer, indent + 2, "residualNorm")?;
    write_json_optional_number(writer, Some(summary.residual_norm))?;
    writeln!(writer, ",")?;
    write_json_key(writer, indent + 2, "normalizedResidual")?;
    write_json_optional_number(writer, Some(summary.normalized_residual_norm))?;
    writeln!(writer, ",")?;
    write_json_key(writer, indent + 2, "effectiveNormalizedTolerance")?;
    write_json_optional_number(writer, Some(summary.effective_normalized_tolerance))?;
    writeln!(writer, ",")?;
    write_json_key(writer, indent + 2, "stopReason")?;
    write_json_string(writer, &summary.stop_reason.to_string())?;
    writeln!(writer)?;
    write_indent(writer, indent)?;
    write!(writer, "}}")
}

fn write_json_momentum_linear_solve_diagnostics(
    writer: &mut impl Write,
    indent: usize,
    summaries: &[LinearSolveConvergenceSummary; 3],
) -> std::io::Result<()> {
    writeln!(writer, "[")?;
    for (index, (component, summary)) in ["x", "y", "z"].iter().zip(summaries).enumerate() {
        write_indent(writer, indent + 2)?;
        writeln!(writer, "{{")?;
        write_json_key(writer, indent + 4, "component")?;
        write_json_string(writer, component)?;
        writeln!(writer, ",")?;
        write_json_key(writer, indent + 4, "solve")?;
        write_json_linear_solve_diagnostic(writer, indent + 4, summary)?;
        writeln!(writer)?;
        write_indent(writer, indent + 2)?;
        if index + 1 == summaries.len() {
            writeln!(writer, "}}")?;
        } else {
            writeln!(writer, "}},")?;
        }
    }
    write_indent(writer, indent)?;
    write!(writer, "]")
}

fn write_json_pressure_linear_solve_diagnostics(
    writer: &mut impl Write,
    indent: usize,
    summaries: &[LinearSolveConvergenceSummary],
) -> std::io::Result<()> {
    writeln!(writer, "[")?;
    for (index, summary) in summaries.iter().enumerate() {
        write_indent(writer, indent + 2)?;
        writeln!(writer, "{{")?;
        write_json_number_field(writer, indent + 4, "correction", index + 1)?;
        writeln!(writer, ",")?;
        write_json_key(writer, indent + 4, "solve")?;
        write_json_linear_solve_diagnostic(writer, indent + 4, summary)?;
        writeln!(writer)?;
        write_indent(writer, indent + 2)?;
        if index + 1 == summaries.len() {
            writeln!(writer, "}}")?;
        } else {
            writeln!(writer, "}},")?;
        }
    }
    write_indent(writer, indent)?;
    write!(writer, "]")
}

fn write_markdown_gamg_profile(
    writer: &mut impl Write,
    profile: &GamgKernelTiming,
) -> std::io::Result<()> {
    writeln!(writer)?;
    writeln!(writer, "## Pressure GAMG Cycle Profile")?;
    writeln!(writer)?;
    writeln!(writer, "| Quantity | Value |")?;
    writeln!(writer, "| --- | ---: |")?;
    let seconds = [
        ("Profiled GAMG total [s]", profile.total_seconds),
        ("Hierarchy build [s]", profile.hierarchy_build_seconds),
        ("Hierarchy rebuild [s]", profile.hierarchy_rebuild_seconds),
        (
            "Hierarchy diagnostics [s]",
            profile.hierarchy_diagnostic_seconds,
        ),
        ("Matrix refresh [s]", profile.matrix_refresh_seconds),
        ("Finest residual [s]", profile.finest_residual_seconds),
        ("V-cycles [s]", profile.v_cycle_seconds),
        ("Restriction [s]", profile.restriction_seconds()),
        ("Prolongation [s]", profile.prolongation_seconds()),
        ("Smoothing [s]", profile.smoothing_seconds()),
        ("Scaling [s]", profile.scaling_seconds()),
        ("Coarse residual [s]", profile.coarse_residual_seconds()),
        ("Correction work [s]", profile.correction_seconds()),
        ("Coarsest solve [s]", profile.coarsest_solve_seconds()),
        ("Other V-cycle work [s]", profile.v_cycle_other_seconds()),
        ("Other profiled work [s]", profile.other_seconds),
    ];
    for (quantity, value) in seconds {
        writeln!(writer, "| {quantity} | {} |", format_scientific(value))?;
    }
    writeln!(writer, "| Pressure solves | {} |", profile.solves)?;
    writeln!(writer, "| V-cycles | {} |", profile.v_cycles)?;
    writeln!(
        writer,
        "| Outer matrix-vector products | {} |",
        profile.outer_matrix_vector_products
    )?;
    writeln!(
        writer,
        "| Outer reductions | {} |",
        profile.outer_reductions
    )?;
    writeln!(writer, "| Levels | {} |", profile.levels.len())?;
    writeln!(writer)?;
    writeln!(
        writer,
        "| Level | Cells | NNZ | Refresh [s] | Restrict [s] | Prolong [s] | Smooth [s] | Scale [s] | Residual [s] | Correction [s] | Coarsest [s] | Smooth sweeps | Coarsest iterations |"
    )?;
    writeln!(
        writer,
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )?;
    for level in &profile.levels {
        writeln!(
            writer,
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |",
            level.level,
            level.cells,
            level.nonzeros,
            format_scientific(level.matrix_refresh_seconds),
            format_scientific(level.restriction_seconds),
            format_scientific(level.prolongation_seconds),
            format_scientific(level.smoothing_seconds),
            format_scientific(level.scaling_seconds),
            format_scientific(level.residual_seconds),
            format_scientific(level.correction_seconds),
            format_scientific(level.coarsest_solve_seconds),
            level.smoothing_sweeps,
            level.coarsest_iterations,
        )?;
    }
    write_markdown_gamg_hierarchy(writer, profile)?;
    Ok(())
}

fn write_markdown_gamg_hierarchy(
    writer: &mut impl Write,
    profile: &GamgKernelTiming,
) -> std::io::Result<()> {
    writeln!(writer)?;
    writeln!(writer, "## Pressure GAMG Hierarchy Diagnostics")?;
    writeln!(writer)?;
    let Some(hierarchy) = profile.hierarchy.as_ref() else {
        return writeln!(writer, "Static hierarchy diagnostics were not recorded.");
    };

    let grid_complexity = hierarchy.grid_complexity_terms();
    let operator_complexity = hierarchy.operator_complexity_terms();
    writeln!(writer, "| Quantity | Exact value |")?;
    writeln!(writer, "| --- | ---: |")?;
    writeln!(
        writer,
        "| Smoother passes per sweep | {} |",
        hierarchy.smoother_passes_per_sweep
    )?;
    writeln!(
        writer,
        "| Direct coarsest solve | {} |",
        yes_no(hierarchy.direct_solve_coarsest)
    )?;
    writeln!(
        writer,
        "| Grid complexity terms | {} / {} |",
        format_optional_u128(grid_complexity.map(|terms| terms.0)),
        format_optional_u128(grid_complexity.map(|terms| terms.1)),
    )?;
    writeln!(
        writer,
        "| Operator complexity terms | {} / {} |",
        format_optional_u128(operator_complexity.map(|terms| terms.0)),
        format_optional_u128(operator_complexity.map(|terms| terms.1)),
    )?;
    writeln!(
        writer,
        "| NNZ-weighted smoothing work | {} |",
        format_optional_u128(profile.nnz_weighted_smoothing_work()),
    )?;
    writeln!(
        writer,
        "| NNZ-weighted sparse work | {} |",
        format_optional_u128(profile.nnz_weighted_sparse_work()),
    )?;

    writeln!(writer)?;
    writeln!(
        writer,
        "| Fine level | Coarse level | Fine cells | Coarse cells | Singleton fine cells | Unmatched fine cells | Min aggregate size | Max aggregate size |"
    )?;
    writeln!(
        writer,
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )?;
    for transfer in &hierarchy.transfers {
        writeln!(
            writer,
            "| {} | {} | {} | {} | {} | {} | {} | {} |",
            transfer.fine_level,
            transfer.coarse_level,
            transfer.fine_cells,
            transfer.coarse_cells,
            transfer.singleton_fine_cells,
            transfer.unmatched_fine_cells,
            transfer.min_aggregate_size,
            transfer.max_aggregate_size,
        )?;
    }

    writeln!(writer)?;
    writeln!(
        writer,
        "| Fine level | Coarse level | Aggregate size | Aggregate count |"
    )?;
    writeln!(writer, "| ---: | ---: | ---: | ---: |")?;
    for transfer in &hierarchy.transfers {
        for bin in &transfer.aggregate_size_histogram {
            writeln!(
                writer,
                "| {} | {} | {} | {} |",
                transfer.fine_level, transfer.coarse_level, bin.aggregate_size, bin.aggregate_count,
            )?;
        }
    }
    Ok(())
}

fn write_laminar_simple_report_markdown(
    plan: &SolverCasePlan,
    options: &LaminarSimpleOptions,
    report: &LaminarSimpleReport,
    post_processing: &LaminarSimplePostProcessing,
    run: LaminarSimpleReportRunMetadata,
    output_root: &SafeOutputRoot,
    path: &Path,
) -> std::io::Result<()> {
    let file = open_replace_output_file(output_root, path)?;
    let mut writer = BufWriter::new(file);

    writeln!(writer, "# incompressibleFluid Solver Report")?;
    writeln!(writer)?;
    writeln!(
        writer,
        "Case: {}",
        escape_markdown_literal(&plan.case_dir.display().to_string())
    )?;
    writeln!(writer)?;
    writeln!(writer, "- Schema version: `3`")?;
    writeln!(writer, "- Module: `incompressibleFluid`")?;
    writeln!(writer, "- Algorithm: `SIMPLE`")?;
    writeln!(writer, "- Regime: `laminar`")?;
    writeln!(writer, "- Backend: `cpu`")?;
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
        "| Momentum execution | {} |",
        options.momentum_execution
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
        "| Momentum linear relative tolerance | {} |",
        format_scientific(options.momentum_linear_relative_tolerance)
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
        "| Pressure linear relative tolerance | {} |",
        format_scientific(options.pressure_linear_relative_tolerance)
    )?;
    writeln!(
        writer,
        "| Pressure max linear iterations | {} |",
        options.pressure_max_linear_iterations
    )?;
    if let Some(gamg) = options.pressure_gamg_options {
        writeln!(
            writer,
            "| GAMG outer solver | {} |",
            gamg_outer_solver_name(&gamg.outer_solver)
        )?;
        writeln!(writer, "| GAMG agglomerator | {} |", gamg.agglomerator)?;
        writeln!(writer, "| GAMG smoother | {} |", gamg.smoother)?;
        writeln!(
            writer,
            "| GAMG cache agglomeration | {} |",
            yes_no(gamg.cache_agglomeration)
        )?;
        writeln!(
            writer,
            "| GAMG cells in coarsest level | {} |",
            gamg.n_cells_in_coarsest_level
        )?;
        writeln!(writer, "| GAMG merge levels | {} |", gamg.merge_levels)?;
        writeln!(writer, "| GAMG min iterations | {} |", gamg.min_iterations)?;
        writeln!(
            writer,
            "| GAMG relative tolerance | {} |",
            format_scientific(gamg.relative_tolerance)
        )?;
        writeln!(writer, "| GAMG pre sweeps | {} |", gamg.n_pre_sweeps)?;
        writeln!(writer, "| GAMG post sweeps | {} |", gamg.n_post_sweeps)?;
        writeln!(writer, "| GAMG finest sweeps | {} |", gamg.n_finest_sweeps)?;
        writeln!(
            writer,
            "| GAMG interpolate correction | {} |",
            yes_no(gamg.interpolate_correction)
        )?;
        writeln!(
            writer,
            "| GAMG scale correction | {} |",
            yes_no(gamg.scale_correction)
        )?;
        writeln!(
            writer,
            "| GAMG direct coarsest solve | {} |",
            yes_no(gamg.direct_solve_coarsest)
        )?;
    }
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
        "| Outer convergence status | {} |",
        outer_convergence_status(report)
    )?;
    writeln!(
        writer,
        "| Outer convergence evaluated | {} |",
        yes_no(report.residual_control.checked)
    )?;
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
        format_scientific(run.wall_clock_seconds)
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
    writeln!(writer, "## Continuity errors")?;
    writeln!(writer)?;
    if let Some(continuity_errors) = &post_processing.continuity_errors {
        writeln!(writer, "| Quantity | Value |")?;
        writeln!(writer, "| --- | ---: |")?;
        writeln!(
            writer,
            "| Final local error | {} |",
            format_scientific(continuity_errors.final_error.local)
        )?;
        writeln!(
            writer,
            "| Final global error | {} |",
            format_scientific(continuity_errors.final_error.global)
        )?;
        writeln!(
            writer,
            "| Cumulative global error | {} |",
            format_scientific(continuity_errors.cumulative_global)
        )?;
        writeln!(
            writer,
            "| deltaT | {} |",
            format_scientific(continuity_errors.delta_t)
        )?;
        writeln!(
            writer,
            "| Total volume | {} |",
            format_scientific(continuity_errors.total_volume)
        )?;
    } else {
        writeln!(
            writer,
            "Unavailable because `controlDict` does not define `deltaT`."
        )?;
    }
    if let Some(wall_forces) = &post_processing.wall_forces {
        let value = &wall_forces.coefficients;
        writeln!(writer)?;
        writeln!(writer, "## Wall forces")?;
        writeln!(writer)?;
        writeln!(writer, "| Quantity | Value |")?;
        writeln!(writer, "| --- | ---: |")?;
        writeln!(
            writer,
            "| Patches | {} |",
            escape_markdown_literal(&wall_forces.patch_names.join(","))
        )?;
        writeln!(writer, "| Selected patches | {} |", value.selected_patches)?;
        writeln!(writer, "| Selected faces | {} |", value.selected_faces)?;
        writeln!(writer, "| Pressure field kind | kinematic |")?;
        writeln!(writer, "| Pressure reference | area-vector balanced mean |")?;
        writeln!(writer, "| Traction method | {WALL_FORCE_TRACTION_METHOD} |")?;
        writeln!(
            writer,
            "| Traction method version | {WALL_FORCE_TRACTION_METHOD_VERSION} |"
        )?;
        writeln!(
            writer,
            "| Force convention | {WALL_FORCE_FORCE_CONVENTION} |"
        )?;
        writeln!(
            writer,
            "| Face area-vector orientation | {WALL_FORCE_AREA_VECTOR_ORIENTATION} |"
        )?;
        writeln!(
            writer,
            "| Pressure face treatment | {WALL_FORCE_PRESSURE_FACE_TREATMENT} |"
        )?;
        writeln!(
            writer,
            "| Velocity gradient scheme | {} |",
            wall_forces.velocity_gradient_scheme
        )?;
        writeln!(
            writer,
            "| Selected area | {} |",
            format_scientific(value.selected_area)
        )?;
        writeln!(
            writer,
            "| Reference speed | {} |",
            format_scientific(wall_forces.reference_speed)
        )?;
        writeln!(
            writer,
            "| Reference area | {} |",
            format_scientific(value.resolved_reference_area)
        )?;
        writeln!(
            writer,
            "| Resolved pressure reference | {} |",
            format_scientific(value.resolved_pressure_reference)
        )?;
        writeln!(
            writer,
            "| Pressure force x/y/z | {} / {} / {} |",
            format_scientific(value.pressure_force.x),
            format_scientific(value.pressure_force.y),
            format_scientific(value.pressure_force.z)
        )?;
        writeln!(
            writer,
            "| Viscous force x/y/z | {} / {} / {} |",
            format_scientific(value.viscous_force.x),
            format_scientific(value.viscous_force.y),
            format_scientific(value.viscous_force.z)
        )?;
        writeln!(
            writer,
            "| Total force x/y/z | {} / {} / {} |",
            format_scientific(value.total_force.x),
            format_scientific(value.total_force.y),
            format_scientific(value.total_force.z)
        )?;
        writeln!(
            writer,
            "| Drag pressure/viscous/total | {} / {} / {} |",
            format_scientific(value.drag.pressure),
            format_scientific(value.drag.viscous),
            format_scientific(value.drag.total)
        )?;
        writeln!(
            writer,
            "| Lift pressure/viscous/total | {} / {} / {} |",
            format_scientific(value.lift.pressure),
            format_scientific(value.lift.viscous),
            format_scientific(value.lift.total)
        )?;
    }
    writeln!(writer)?;
    writeln!(writer, "## Timing Profile")?;
    writeln!(writer)?;
    writeln!(writer, "| Phase | Seconds |")?;
    writeln!(writer, "| --- | ---: |")?;
    let timing_rows = [
        ("Solver total", report.timing.solver_total_seconds),
        ("Driver measured", run.wall_clock_seconds),
        ("Setup", report.timing.setup_seconds),
        ("Iteration setup", report.timing.iteration_setup_seconds),
        (
            "Operator evaluation",
            report.timing.operator_evaluation_seconds,
        ),
        (
            "Momentum matrix assembly",
            report.timing.momentum_assembly_seconds,
        ),
        (
            "Momentum gradient reconstruction",
            report.timing.momentum_gradient_seconds,
        ),
        (
            "Momentum matrix fill",
            report.timing.momentum_matrix_fill_seconds,
        ),
        (
            "Momentum linear solves",
            report.timing.momentum_linear_solve_seconds,
        ),
        (
            "Pressure coupling setup",
            report.timing.pressure_coupling_setup_seconds,
        ),
        (
            "Pressure matrix assembly",
            report.timing.pressure_assembly_seconds,
        ),
        (
            "Pressure linear solves",
            report.timing.pressure_linear_solve_seconds,
        ),
        ("Field correction", report.timing.field_correction_seconds),
        ("Finalization", report.timing.finalization_seconds),
        ("Other solver work", report.timing.other_solver_work_seconds),
    ];
    for (phase, seconds) in timing_rows {
        writeln!(writer, "| {phase} | {} |", format_scientific(seconds))?;
    }
    if run.profile_pcg {
        writeln!(writer)?;
        writeln!(writer, "## Pressure PCG Kernel Profile")?;
        writeln!(writer)?;
        writeln!(writer, "| Quantity | Value |")?;
        writeln!(writer, "| --- | ---: |")?;
        let pressure_kernel_rows = [
            ("PCG total [s]", report.timing.pressure_pcg_total_seconds),
            (
                "Preconditioner update [s]",
                report.timing.pressure_preconditioner_update_seconds,
            ),
            (
                "Matrix-vector products [s]",
                report.timing.pressure_matrix_vector_seconds,
            ),
            (
                "Preconditioner applications [s]",
                report.timing.pressure_preconditioner_application_seconds,
            ),
            (
                "Vector operations [s]",
                report.timing.pressure_vector_operation_seconds,
            ),
            (
                "Other PCG work [s]",
                report.timing.pressure_pcg_other_seconds,
            ),
        ];
        for (quantity, value) in pressure_kernel_rows {
            writeln!(writer, "| {quantity} | {} |", format_scientific(value))?;
        }
        writeln!(
            writer,
            "| Matrix-vector product calls | {} |",
            report.timing.pressure_matrix_vector_products
        )?;
        writeln!(
            writer,
            "| Preconditioner application calls | {} |",
            report.timing.pressure_preconditioner_applications
        )?;
    }
    if let Some(profile) = &report.timing.pressure_gamg_profile {
        write_markdown_gamg_profile(&mut writer, profile)?;
    }
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
        "| Iteration | Continuity before | Continuity after | Pressure correction | U initial | U final | U linear iter/ok | U components initial | U components final | p initial | p final | p linear iter/ok | p-solves/nonconv | residualControl | A diag min/max | H1 min/max | adjustPhi before/after/faces | U change | p change | U solve diagnostics | p solve diagnostics |"
    )?;
    writeln!(
        writer,
        "| ---: | ---: | ---: | --- | ---: | ---: | --- | --- | --- | ---: | ---: | --- | --- | --- | --- | --- | --- | ---: | ---: | --- | --- |"
    )?;
    for item in &report.history {
        let pressure_linear_solves = pressure_linear_solve_summaries(item)?;
        writeln!(
            writer,
            "| {} | {} | {} | {} | {} | {} | {} / {} | {} | {} | {} | {} | {} / {} | {} / {} | {} | {} / {} | {} / {} | {} / {} / {} | {} | {} | {} | {} |",
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
            format_percent(item.relative_pressure_change_l2),
            format_momentum_linear_solve_diagnostics(
                &item.momentum_component_linear_solves,
                "<br>"
            ),
            format_pressure_linear_solve_diagnostics(pressure_linear_solves, "<br>")
        )?;
    }

    writer.flush()
}

fn escape_markdown_literal(value: &str) -> String {
    use std::fmt::Write as _;

    let mut escaped = String::with_capacity(value.len());
    for character in value.chars() {
        if character.is_ascii_alphanumeric() || matches!(character, ' ' | ',' | '-') {
            escaped.push(character);
        } else {
            write!(&mut escaped, "&#{};", u32::from(character))
                .expect("writing a numeric entity to String cannot fail");
        }
    }
    escaped
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
    output_root: &SafeOutputRoot,
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

    let velocity_field = solver_initial_field(fields, "U", "volVectorField")?;
    let pressure_field = solver_initial_field(fields, "p", "volScalarField")?;
    let relative_output_dir = relative_output_path(output_root, output_dir)?;
    let location = openfoam_output_location(output_dir)?;
    let velocity_path = fallible_join_output_path(&relative_output_dir, "U")?;
    let pressure_path = fallible_join_output_path(&relative_output_dir, "p")?;
    output_root.validate_file_path(&velocity_path)?;
    output_root.validate_file_path(&pressure_path)?;
    validate_replace_output_entry(output_root, &velocity_path)?;
    validate_replace_output_entry(output_root, &pressure_path)?;
    output_root.ensure_dir(&relative_output_dir)?;

    write_openfoam_vector_field(
        output_root,
        &velocity_path,
        velocity_field,
        &location,
        &report.final_velocity,
    )?;
    write_openfoam_scalar_field(
        output_root,
        &pressure_path,
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
    output_root: &SafeOutputRoot,
    path: &Path,
    source_field: &FieldFile,
    location: &str,
    values: &[Point3],
) -> std::io::Result<()> {
    let file = open_replace_output_file(output_root, path)?;
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
    output_root: &SafeOutputRoot,
    path: &Path,
    source_field: &FieldFile,
    location: &str,
    values: &[f64],
) -> std::io::Result<()> {
    let file = open_replace_output_file(output_root, path)?;
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

fn openfoam_output_location(output_dir: &Path) -> std::io::Result<String> {
    let source = output_dir.file_name().unwrap_or(output_dir.as_os_str());
    let value = source
        .to_str()
        .ok_or_else(|| Error::from(ErrorKind::InvalidInput))?;
    let mut location = String::new();
    location
        .try_reserve_exact(value.len())
        .map_err(|_| Error::from(ErrorKind::OutOfMemory))?;
    location.push_str(value);
    Ok(location)
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

fn relative_output_path(output_root: &SafeOutputRoot, path: &Path) -> std::io::Result<PathBuf> {
    let relative = if path.is_absolute() {
        path.strip_prefix(output_root.path()).map_err(|_| {
            Error::new(
                ErrorKind::PermissionDenied,
                "output path must remain below the trusted output root",
            )
        })?
    } else {
        path
    };
    let reserve = relative
        .as_os_str()
        .len()
        .checked_add(1)
        .ok_or_else(|| Error::from(ErrorKind::OutOfMemory))?;
    let mut owned = PathBuf::new();
    owned
        .try_reserve(reserve)
        .map_err(|_| Error::from(ErrorKind::OutOfMemory))?;
    owned.push(relative);
    Ok(owned)
}

fn fallible_join_output_path(base: &Path, leaf: &str) -> std::io::Result<PathBuf> {
    let reserve = base
        .as_os_str()
        .len()
        .checked_add(leaf.len())
        .and_then(|size| size.checked_add(1))
        .ok_or_else(|| Error::from(ErrorKind::OutOfMemory))?;
    let mut path = PathBuf::new();
    path.try_reserve(reserve)
        .map_err(|_| Error::from(ErrorKind::OutOfMemory))?;
    path.push(base);
    path.push(leaf);
    Ok(path)
}

fn open_replace_output_file(output_root: &SafeOutputRoot, path: &Path) -> std::io::Result<File> {
    let relative = relative_output_path(output_root, path)?;
    output_root.validate_file_path(&relative)?;
    if let Some(parent) = relative.parent()
        && !parent.as_os_str().is_empty()
    {
        output_root.ensure_dir(parent)?;
    }
    output_root.open_replace_regular(&relative)
}

fn open_create_new_operator_output_file(
    default_root: &SafeOutputRoot,
    path: &Path,
) -> std::io::Result<File> {
    if !path.is_absolute() || path.starts_with(default_root.path()) {
        let relative = relative_output_path(default_root, path)?;
        return default_root.open_create_new(&relative);
    }

    SafeOutputRoot::open_create_new_trusted_absolute(path)
}

fn validate_replace_output_entry(
    output_root: &SafeOutputRoot,
    relative: &Path,
) -> std::io::Result<()> {
    let entry = match output_root.entry(relative) {
        Ok(entry) => entry,
        Err(error) if error.kind() == ErrorKind::NotFound => None,
        Err(error) => return Err(error),
    };
    match entry {
        None | Some(SafeOutputEntry::File) => Ok(()),
        Some(SafeOutputEntry::Directory | SafeOutputEntry::Other) => Err(Error::new(
            ErrorKind::InvalidInput,
            "output path is not a regular file",
        )),
    }
}

fn write_json_control(writer: &mut impl Write, plan: &SolverCasePlan) -> std::io::Result<()> {
    write_json_key(writer, 2, "control")?;
    writeln!(writer, "{{")?;
    write_json_key(writer, 4, "application")?;
    write_json_optional_string(writer, plan.control.application.as_deref())?;
    writeln!(writer, ",")?;
    write_json_key(writer, 4, "solver")?;
    write_json_optional_string(writer, plan.control.solver.as_deref())?;
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

fn write_json_dispatch(
    writer: &mut impl Write,
    dispatch: Option<&SolverDispatch>,
) -> std::io::Result<()> {
    write_json_key(writer, 2, "dispatch")?;
    match dispatch {
        Some(dispatch) => {
            writeln!(writer, "{{")?;
            write_json_key(writer, 4, "module")?;
            write_json_string(writer, &dispatch.module)?;
            writeln!(writer, ",")?;
            write_json_key(writer, 4, "source")?;
            write_json_string(writer, &dispatch.source.to_string())?;
            writeln!(writer)?;
            write_indent(writer, 2)?;
            write!(writer, "}}")
        }
        None => write!(writer, "null"),
    }
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
        write_json_optional_usize(writer, field.internal_field.loaded_scalars)?;
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

fn write_json_optional_u128_decimal(
    writer: &mut impl Write,
    value: Option<u128>,
) -> std::io::Result<()> {
    match value {
        Some(value) => write_json_string(writer, &value.to_string()),
        None => write!(writer, "null"),
    }
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

        print!("  {}: ", section.name);
        for (index, entry) in section.entries.iter().enumerate() {
            if index != 0 {
                print!(", ");
            }
            print!("{}={}", entry.step, entry.choice);
        }
        println!();
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

#[derive(Clone, Copy)]
struct DeviceList<'a>(&'a [String]);

impl fmt::Display for DeviceList<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let [device] = self.0 {
            return formatter.write_str(device);
        }
        formatter.write_str("(")?;
        for (index, device) in self.0.iter().enumerate() {
            if index != 0 {
                formatter.write_str(" ")?;
            }
            formatter.write_str(device)?;
        }
        formatter.write_str(")")
    }
}

fn format_devices(devices: &[String]) -> DeviceList<'_> {
    DeviceList(devices)
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

fn format_optional_u128(value: Option<u128>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "n/a".to_string())
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

fn outer_convergence_status(report: &LaminarSimpleReport) -> &'static str {
    outer_convergence_status_for_reason(report.stop_reason)
}

fn outer_convergence_status_for_reason(reason: LaminarSimpleStopReason) -> &'static str {
    match reason {
        LaminarSimpleStopReason::Converged => "converged",
        LaminarSimpleStopReason::ConvergenceCriteriaNotConfigured => "not-evaluated",
        LaminarSimpleStopReason::MaxIterationsReached => "not-reached",
        LaminarSimpleStopReason::MomentumSolverInvalidState
        | LaminarSimpleStopReason::PressureSolverInvalidState
        | LaminarSimpleStopReason::SolverInvalidState => "invalid",
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

fn print_field_boundary_validation(
    case_dir: &Path,
    fields: &InitialFieldSet,
) -> Result<(), String> {
    let summary =
        validate_initial_field_boundaries(case_dir, fields).map_err(|error| error.to_string())?;
    print_field_boundary_validation_summary(&summary);
    Ok(())
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

fn format_linear_solve_diagnostic(label: &str, summary: &LinearSolveConvergenceSummary) -> String {
    format!(
        "{label}:iter={}/ok={}/initial={}/residual={}/final={}/effectiveTol={}/reason={}",
        summary.iterations,
        summary.converged,
        format_scientific(summary.initial_normalized_residual_norm),
        format_scientific(summary.residual_norm),
        format_scientific(summary.normalized_residual_norm),
        format_scientific(summary.effective_normalized_tolerance),
        summary.stop_reason
    )
}

fn format_momentum_linear_solve_diagnostics(
    summaries: &[LinearSolveConvergenceSummary; 3],
    separator: &str,
) -> String {
    ["x", "y", "z"]
        .iter()
        .zip(summaries)
        .map(|(label, summary)| format_linear_solve_diagnostic(label, summary))
        .collect::<Vec<_>>()
        .join(separator)
}

fn format_pressure_linear_solve_diagnostics(
    summaries: &[LinearSolveConvergenceSummary],
    separator: &str,
) -> String {
    summaries
        .iter()
        .enumerate()
        .map(|(index, summary)| format_linear_solve_diagnostic(&(index + 1).to_string(), summary))
        .collect::<Vec<_>>()
        .join(separator)
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
struct LaminarSimpleSolveArgs {
    density: Option<f64>,
    dynamic_viscosity: Option<f64>,
    linear_solver: Option<LaminarSimpleLinearSolver>,
    momentum_linear_solver: Option<LaminarSimpleLinearSolver>,
    momentum_execution: Option<LaminarSimpleMomentumExecution>,
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
    profile_gamg: bool,
    profile_pcg: bool,
    solve_verbose: bool,
    solve_residual_csv: Option<PathBuf>,
    solve_residual_plot: Option<PathBuf>,
    report_json: Option<PathBuf>,
    report_markdown: Option<PathBuf>,
    write_final_fields: Option<PathBuf>,
    wall_forces: Option<WallForceSolveArgs>,
}

#[derive(Clone, Debug, PartialEq)]
struct WallForceSolveArgs {
    patch_names: Vec<String>,
    reference_speed: f64,
    reference_area: f64,
    face_loads_csv: Option<PathBuf>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct ContinuityErrorsReport {
    final_error: NormalizedContinuity,
    cumulative_global: f64,
    delta_t: f64,
    total_volume: f64,
}

#[derive(Clone, Debug)]
struct WallForceReport {
    patch_names: Vec<String>,
    reference_speed: f64,
    velocity_gradient_scheme: LaminarSimpleGradientScheme,
    coefficients: WallForceCoefficients,
    faces: Vec<WallFaceForceContribution>,
}

#[derive(Clone, Debug)]
struct LaminarSimplePostProcessing {
    continuity_errors: Option<ContinuityErrorsReport>,
    wall_forces: Option<WallForceReport>,
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
    parse_solver_args_for_invocation(args, SolverInvocation::Utility)
}

fn parse_incompressible_fluid_args(args: &[String]) -> Result<SolverArgs, String> {
    parse_solver_args_for_invocation(args, SolverInvocation::IncompressibleFluidExecute)
}

fn parse_incompressible_fluid_plan_args(args: &[String]) -> Result<SolverArgs, String> {
    parse_solver_args_for_invocation(args, SolverInvocation::IncompressibleFluidPlanOnly)
}

fn parse_solver_args_for_invocation(
    args: &[String],
    invocation: SolverInvocation,
) -> Result<SolverArgs, String> {
    let mut case_dir = PathBuf::from(".");
    let mut plan_json = None;
    let mut inspection_only = false;
    let mut runner_dry_run = false;
    let mut max_runner_steps = SolverRunnerDryRunOptions::default().max_steps;
    let mut scalar_diffusion_field = None;
    let mut scalar_diffusion_option_seen = false;
    let mut laminar_simple_option_seen = false;
    let mut shared_flow_option_seen = false;
    let mut density = None;
    let mut dynamic_viscosity = None;
    let mut linear_solve_option_seen = false;
    let mut laminar_linear_solver = None;
    let mut momentum_linear_solver = None;
    let mut momentum_execution = None;
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
    let mut profile_gamg = false;
    let mut profile_pcg = false;
    let mut solve_verbose = false;
    let mut solve_residual_csv = None;
    let mut solve_residual_plot = None;
    let mut solve_report_json = None;
    let mut solve_report_markdown = None;
    let mut write_final_fields = None;
    let mut wall_force_patches = None;
    let mut force_reference_speed = None;
    let mut force_reference_area = None;
    let mut wall_face_loads_csv = None;
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
                inspection_only = true;
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
                inspection_only = true;
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
                if max_runner_steps > MAX_RUNNER_DRY_RUN_STEPS {
                    return Err(format!(
                        "--maxRunnerSteps must not exceed {MAX_RUNNER_DRY_RUN_STEPS}"
                    ));
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
            "--momentumExecution" | "--momentum-execution" => {
                let value = args.get(index + 1).ok_or_else(|| {
                    "--momentumExecution requires 'serial' or 'parallel-components'".to_string()
                })?;
                momentum_execution = Some(parse_laminar_simple_momentum_execution(value)?);
                laminar_simple_option_seen = true;
                index += 2;
            }
            "-pressureLinearSolver"
            | "--pressureLinearSolver"
            | "-pressure-linear-solver"
            | "--pressure-linear-solver" => {
                let value = args.get(index + 1).ok_or_else(|| {
                    "--pressureLinearSolver requires 'bicgstab', 'gaussSeidel', 'symGaussSeidel', 'cg', 'pcg', 'GAMG', or 'jacobi'"
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
                    "--momentumPreconditioner requires 'none', 'diagonal', 'DIC', 'FDIC', 'ic0', or 'incompleteCholesky'"
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
                    "--pressurePreconditioner requires 'none', 'diagonal', 'DIC', 'FDIC', 'ic0', or 'incompleteCholesky'"
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
                let path = args
                    .get(index + 1)
                    .ok_or_else(|| "--solveResidualPlot requires an output SVG path".to_string())?;
                solve_residual_plot = Some(PathBuf::from(path));
                laminar_simple_option_seen = true;
                index += 2;
            }
            "-profileGamg" | "--profileGamg" | "-profile-gamg" | "--profile-gamg" => {
                profile_gamg = true;
                laminar_simple_option_seen = true;
                index += 1;
            }
            "-profilePcg" | "--profilePcg" | "-profile-pcg" | "--profile-pcg" => {
                profile_pcg = true;
                laminar_simple_option_seen = true;
                index += 1;
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
            "-wallForcePatches"
            | "--wallForcePatches"
            | "-wall-force-patches"
            | "--wall-force-patches" => {
                if wall_force_patches.is_some() {
                    return Err("--wallForcePatches may be specified only once".to_string());
                }
                let value = args.get(index + 1).ok_or_else(|| {
                    "--wallForcePatches requires comma-separated patch names".to_string()
                })?;
                wall_force_patches = Some(parse_wall_force_patch_names(value)?);
                laminar_simple_option_seen = true;
                index += 2;
            }
            "-forceReferenceSpeed"
            | "--forceReferenceSpeed"
            | "-force-reference-speed"
            | "--force-reference-speed" => {
                if force_reference_speed.is_some() {
                    return Err("--forceReferenceSpeed may be specified only once".to_string());
                }
                let value = args.get(index + 1).ok_or_else(|| {
                    "--forceReferenceSpeed requires a positive finite speed".to_string()
                })?;
                force_reference_speed =
                    Some(parse_positive_f64_arg("--forceReferenceSpeed", value)?);
                laminar_simple_option_seen = true;
                index += 2;
            }
            "-forceReferenceArea"
            | "--forceReferenceArea"
            | "-force-reference-area"
            | "--force-reference-area" => {
                if force_reference_area.is_some() {
                    return Err("--forceReferenceArea may be specified only once".to_string());
                }
                let value = args.get(index + 1).ok_or_else(|| {
                    "--forceReferenceArea requires a positive finite area".to_string()
                })?;
                force_reference_area = Some(parse_positive_f64_arg("--forceReferenceArea", value)?);
                laminar_simple_option_seen = true;
                index += 2;
            }
            "-wallFaceLoadsCsv"
            | "--wallFaceLoadsCsv"
            | "-wall-face-loads-csv"
            | "--wall-face-loads-csv" => {
                if wall_face_loads_csv.is_some() {
                    return Err("--wallFaceLoadsCsv may be specified only once".to_string());
                }
                let value = args
                    .get(index + 1)
                    .ok_or_else(|| "--wallFaceLoadsCsv requires a file path".to_string())?;
                if value.trim().is_empty() {
                    return Err("--wallFaceLoadsCsv requires a non-empty file path".to_string());
                }
                wall_face_loads_csv = Some(PathBuf::from(value));
                laminar_simple_option_seen = true;
                index += 2;
            }
            other => return Err(format!("unknown ferrum solve option '{other}'")),
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
    let wall_forces = match (
        wall_force_patches,
        force_reference_speed,
        force_reference_area,
    ) {
        (None, None, None) if wall_face_loads_csv.is_none() => None,
        (None, None, None) => {
            return Err(
                "--wallFaceLoadsCsv requires --wallForcePatches, --forceReferenceSpeed, and --forceReferenceArea"
                    .to_string(),
            );
        }
        (Some(patch_names), Some(reference_speed), Some(reference_area)) => {
            Some(WallForceSolveArgs {
                patch_names,
                reference_speed,
                reference_area,
                face_loads_csv: wall_face_loads_csv,
            })
        }
        _ => {
            return Err(
                "--wallForcePatches, --forceReferenceSpeed, and --forceReferenceArea must be supplied together"
                    .to_string(),
            );
        }
    };
    let laminar_simple_solve = if invocation == SolverInvocation::IncompressibleFluidExecute {
        Some(LaminarSimpleSolveArgs {
            density,
            dynamic_viscosity,
            linear_solver: laminar_linear_solver,
            momentum_linear_solver,
            momentum_execution,
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
            profile_gamg,
            profile_pcg,
            solve_verbose,
            solve_residual_csv,
            solve_residual_plot,
            report_json: solve_report_json,
            report_markdown: solve_report_markdown,
            write_final_fields,
            wall_forces,
        })
    } else {
        None
    };
    if inspection_only && (scalar_diffusion_solve.is_some() || laminar_simple_solve.is_some()) {
        return Err(
            "inspection modes (--preflight/--runnerDryRun) cannot be combined with equation solves"
                .to_string(),
        );
    }
    if laminar_simple_solve.is_none() && shared_flow_option_seen {
        return Err("--mu requires ferrumRun -solver incompressibleFluid".to_string());
    }
    if scalar_diffusion_solve.is_some()
        && let Some(error) = scalar_diffusion_linear_solver_error
    {
        return Err(error);
    }
    if laminar_simple_solve.is_none() && laminar_simple_option_seen {
        return Err(
            "incompressible flow options require ferrumRun -solver incompressibleFluid".to_string(),
        );
    }
    if scalar_diffusion_solve.is_none()
        && laminar_simple_solve.is_none()
        && linear_solve_option_seen
    {
        return Err(
            "linear solve options require ferrum solve --solveScalarDiffusion <field>, or ferrumRun -solver incompressibleFluid"
                .to_string(),
        );
    }
    let executable_solve_count =
        scalar_diffusion_solve.is_some() as usize + laminar_simple_solve.is_some() as usize;
    if executable_solve_count > 1 {
        return Err(
            "developer utility solves cannot be combined with an incompressibleFluid application run"
                .to_string(),
        );
    }
    Ok(SolverArgs {
        case_dir,
        plan_json,
        runner_dry_run,
        max_runner_steps,
        scalar_diffusion_solve,
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

fn parse_laminar_simple_momentum_execution(
    value: &str,
) -> Result<LaminarSimpleMomentumExecution, String> {
    match value {
        "serial" => Ok(LaminarSimpleMomentumExecution::Serial),
        "parallel-components" => Ok(LaminarSimpleMomentumExecution::ParallelComponents),
        other => Err(format!(
            "invalid --momentumExecution value '{other}'; expected 'serial' or 'parallel-components'"
        )),
    }
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

fn parse_wall_force_patch_names(value: &str) -> Result<Vec<String>, String> {
    if value.is_empty() {
        return Err("--wallForcePatches requires at least one patch name".to_string());
    }
    let mut patch_names = Vec::new();
    for raw_name in value.split(',') {
        let name = raw_name.trim();
        if name.is_empty() {
            return Err(
                "--wallForcePatches must not contain empty comma-separated patch names".to_string(),
            );
        }
        if patch_names.iter().any(|existing| existing == name) {
            return Err(format!(
                "--wallForcePatches patch '{name}' is selected more than once"
            ));
        }
        patch_names.push(name.to_string());
    }
    Ok(patch_names)
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

struct GmshToFerrumArgs {
    mesh_path: PathBuf,
    case_dir: PathBuf,
    options: FoamWriteOptions,
}

fn parse_gmsh_to_ferrum_args(args: &[String]) -> Result<GmshToFerrumArgs, String> {
    let mesh_path = PathBuf::from(
        args.first()
            .ok_or_else(|| "gmshToFerrum requires a mesh path".to_string())?,
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
            other => return Err(format!("unknown gmshToFerrum option '{other}'")),
        }
    }

    Ok(GmshToFerrumArgs {
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
        return Err(format!("invalid supported patch type '{patch_type}'"));
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
    println!("FerrumCFD command-line tools");
    println!();
    println!("usage:");
    println!("  ferrum initFerrumCase <caseDir> [--region <name> ...] [--force]");
    println!("  ferrum gmshToFerrum <mesh.msh> [-case <caseDir>] [patch type options]");
    println!("  ferrum checkFerrumMesh [-case <caseDir>]");
    println!("  ferrum splitFerrumMeshRegions [-case <caseDir>] [-cellZones]");
    println!("  ferrum run -solver incompressibleFluid [-case <caseDir>] [run options]");
    println!("  ferrum solve [-case <caseDir>] [--preflight] [--planJson <file>] [--runnerDryRun]");
    println!();
    println!("aliases:");
    println!("  initFerrumCase <caseDir> [--region <name> ...] [--force]");
    println!("  gmshToFerrum <mesh.msh> [-case <caseDir>] [patch type options]");
    println!("  checkFerrumMesh [-case <caseDir>]");
    println!("  splitFerrumMeshRegions [-case <caseDir>] [-cellZones]");
    println!("  ferrumRun -solver incompressibleFluid [-case <caseDir>] [run options]");

    println!();
    print_patch_type_options();
}

fn print_ferrum_run_usage() {
    println!(
        "usage: ferrumRun [-solver incompressibleFluid] [-case <caseDir>] [--preflight|--runnerDryRun] [run options]"
    );
    println!();
    println!("runs one FerrumCFD case through a runtime-selectable solver module");
    println!("the solver may be supplied on the command line or as this controlDict entry:");
    println!("  solver incompressibleFluid;");
    println!();
    println!("current executable contract:");
    println!("  module: incompressibleFluid");
    println!("  coupling: SIMPLE section required for execution");
    println!("  regime: laminar");
    println!("  backend: CPU equation kernels; GPU kernels are planned");
    println!();
    println!(
        "SIMPLE, SIMPLEC, PISO, PIMPLE, and laminar/turbulent model choices belong in the case, not in the public solver name"
    );
    println!("--preflight and --runnerDryRun inspect the case without executing equations");
    println!();
    println!("incompressibleFluid run options:");
    println!("  --rho <kg/m3>                    override case density");
    println!("  --mu <Pa.s>                      override case dynamic viscosity");
    println!("  --linearSolver <name>            override both equation solvers");
    println!("  --momentumLinearSolver <name>    override the U solver");
    println!("  --momentumExecution <mode>       serial or parallel-components U solves");
    println!("  --pressureLinearSolver <name>    override the p solver");
    println!("  --momentumPreconditioner <name>  override the U preconditioner");
    println!("  --pressurePreconditioner <name>  override the p preconditioner");
    println!("  --maxSimpleIterations <n>        override the SIMPLE iteration budget");
    println!("  --minSimpleIterations <n>        set the minimum SIMPLE iterations");
    println!("  --nNonOrthogonalCorrectors <n>   override pressure correctors");
    println!("  --simpleConsistent [bool]        select consistent SIMPLE/SIMPLEC behavior");
    println!("  --pRefCell <n>                   set the pressure reference cell");
    println!("  --pRefValue <Pa>                 set the pressure reference value");
    println!("  --velocityRelaxation <v>         override U relaxation");
    println!("  --pressureRelaxation <v>         override p relaxation");
    println!("  --solveVerbose                   print per-iteration residuals");
    println!("  --solveResidualCsv <file>        write residual history CSV");
    println!("  --solveResidualPlot <file.svg>   render a native residual SVG");
    println!("  --profileGamg                    profile GAMG phases and levels (diagnostic)");
    println!("  --profilePcg                     profile pressure PCG kernels (diagnostic)");
    println!("  --solveReportJson <file>         write the solver report as JSON");
    println!("  --solveReportMarkdown <file>     write the solver report as Markdown");
    println!("  --writeFinalFields <dir>         write final U and p fields");
    println!("  --wallForcePatches <a,b>         integrate selected noSlip wall patches");
    println!("  --forceReferenceSpeed <m/s>      positive force-coefficient speed");
    println!("  --forceReferenceArea <m2>        positive force-coefficient area");
    println!("  --wallFaceLoadsCsv <file>        write optional per-face wall loads CSV");
    println!("  wall-force options are opt-in and must be supplied together");
}

fn print_init_case_usage() {
    println!("usage: initFerrumCase <caseDir> [--region <name> ...] [--regions a,b] [--force]");
    println!();
    println!("creates a FerrumCFD case skeleton:");
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
        "usage: ferrum solve [-case <caseDir>] [--preflight] [--planJson <file>] [--runnerDryRun] [--maxRunnerSteps <n>] [--solveScalarDiffusion <field>]"
    );
    println!();
    println!("developer-only scalar-diffusion equation utility; application flow uses ferrumRun");
    println!();
    println!("options:");
    println!("  --planJson <file>    also write the solver-neutral plan as JSON");
    println!("  --runnerDryRun       preview the future solver runner without solving equations");
    println!("  inspection modes cannot be combined with --solveScalarDiffusion");
    println!(
        "  --maxRunnerSteps <n> limit runner dry-run preview steps (default: 3, max: {MAX_RUNNER_DRY_RUN_STEPS})"
    );
    println!("  --solveScalarDiffusion <field> assemble and solve one CPU scalar diffusion system");

    println!("  --solveTolerance <v> absolute residual tolerance (default: 1e-10)");
    println!("  --maxIterations <n>  linear solver iteration cap (default: 10000)");
}

fn print_gmsh_to_ferrum_usage() {
    println!("usage: gmshToFerrum <mesh.msh> [-case <caseDir>] [patch type options]");
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
    println!("  -patchType <patch>=<type>    write any supported patch type");
}

#[allow(dead_code)]
fn normalize_case_path(path: &Path) -> PathBuf {
    path.to_path_buf()
}

#[cfg(test)]
mod tests {
    use super::{
        ContinuityErrorsReport, ContinuitySummary, FERRUM_MAX_CASE_SIMPLE_ITERATIONS,
        LaminarSimpleIterationSummary, LaminarSimpleOptions, LaminarSimplePostProcessing,
        LaminarSimpleReportRunMetadata, LaminarSimpleResidualControlSummary, LaminarSimpleSchemes,
        MAX_RUNNER_DRY_RUN_STEPS, SafeOutputRoot, ScalarDiffusionLinearSolver,
        SolverNumericsDictionaryPlan, SolverSelectionSource, WallForceReport,
        build_laminar_simple_post_processing, escape_markdown_literal,
        estimate_iterations_to_convergence, estimate_simple_iterations_to_convergence,
        format_optional_u128, format_pressure_gamg_console, gamg_outer_profile_counters,
        numerics_dictionary_number, numerics_dictionary_usize, numerics_dictionary_value,
        openfoam_gamg_options, outer_convergence_status_for_reason, parse_ferrum_run_args,
        parse_incompressible_fluid_args, parse_incompressible_fluid_plan_args,
        parse_laminar_simple_convection_scheme, parse_laminar_simple_gradient_scheme,
        parse_laminar_simple_laplacian_scheme, parse_laminar_simple_sn_grad_scheme,
        parse_openfoam_laminar_preconditioner, parse_openfoam_laminar_solver, parse_solver_args,
        resolve_laminar_simple_convection_scheme, resolve_laminar_simple_options,
        resolve_laminar_simple_schemes, resolve_solver_dispatch, resolved_gradient_scheme_value,
        run_ferrum_subcommand, validate_laminar_residual_control_dictionary,
        validate_module_execution_contract, validate_wall_face_loads_output_destination_in_root,
        validate_wall_force_boundary_conditions, write_json_gamg_hierarchy,
        write_json_optional_u128_decimal, write_json_solver_state, write_json_string,
        write_laminar_simple_fields, write_laminar_simple_report_json,
        write_laminar_simple_report_markdown, write_laminar_simple_residual_csv,
        write_laminar_simple_residual_plot, write_laminar_simple_wall_face_loads_csv,
        write_solver_plan_json_in_root,
    };
    use ferrum_mesh::backends::BackendChoice;
    use ferrum_mesh::control::ControlDict;
    use ferrum_mesh::flow::{
        AlgebraicResidualRowDiagnostic, IncidentFaceDiagnosticSummary,
        LaminarSimpleConvectionScheme, LaminarSimpleGradientScheme, LaminarSimpleLaplacianScheme,
        LaminarSimpleLinearSolver, LaminarSimpleMomentumExecution, LaminarSimplePreconditioner,
        LaminarSimpleSnGradScheme, LaminarSimpleStopReason, LinearNonConvergenceSnapshot,
        LinearSolveConvergenceSummary, LinearSolveStopReason,
    };
    use ferrum_mesh::linear::{
        GamgAgglomerator, GamgAggregateSizeBin, GamgHierarchyDiagnostics,
        GamgHierarchyLevelDiagnostics, GamgKernelTiming, GamgLevelTiming, GamgOptions,
        GamgOuterSolver, GamgSmoother, GamgTransferDiagnostics,
    };
    use ferrum_mesh::runtime::{SolverRuntimeData, SolverRuntimeMeshData, SolverRuntimePatchRange};
    use ferrum_mesh::solver_plan::{
        SolverBackendPlan, SolverCasePlan, SolverCpuResourcePlan, SolverDimensionality,
        SolverFieldPlan, SolverGpuResourcePlan, SolverInterfacePlan, SolverMeshPlan,
        SolverNumericsEntryPlan, SolverNumericsPlan, SolverPropertiesPlan,
        SolverPropertyDictionaryPlan, SolverPropertyEntryPlan, SolverRunPlan,
    };
    use ferrum_mesh::solver_state::{
        SolverStateCpuBufferPlan, SolverStateCpuBufferStatus, SolverStateFieldKind,
        SolverStateFieldPlan, SolverStateInternalFieldPlan, SolverStatePlan,
        SolverStateStoragePlan, SolverStateStorageStatus, SolverStateValueKind,
    };
    use std::io::ErrorKind;
    use std::path::{Path, PathBuf};

    use ferrum_finite_volume::boundary_forces::{
        DirectionalCoefficient, WallFaceForceContribution, WallForceCoefficients,
    };

    #[test]
    fn outer_convergence_status_distinguishes_missing_and_unmet_criteria() {
        assert_eq!(
            outer_convergence_status_for_reason(LaminarSimpleStopReason::Converged),
            "converged"
        );
        assert_eq!(
            outer_convergence_status_for_reason(
                LaminarSimpleStopReason::ConvergenceCriteriaNotConfigured,
            ),
            "not-evaluated"
        );
        assert_eq!(
            outer_convergence_status_for_reason(LaminarSimpleStopReason::MaxIterationsReached),
            "not-reached"
        );
        assert_eq!(
            outer_convergence_status_for_reason(LaminarSimpleStopReason::SolverInvalidState),
            "invalid"
        );
    }

    #[test]
    fn parses_canonical_ferrum_run_module() {
        let parsed = parse_ferrum_run_args(&[
            "-solver".to_string(),
            "incompressibleFluid".to_string(),
            "-case".to_string(),
            "cases/pipe".to_string(),
            "--maxSimpleIterations".to_string(),
            "2".to_string(),
        ])
        .expect("canonical ferrumRun args should parse");

        assert_eq!(parsed.solver.as_deref(), Some("incompressibleFluid"));
        assert_eq!(
            parsed.forwarded_args,
            vec!["-case", "cases/pipe", "--maxSimpleIterations", "2"]
        );
        assert!(parsed.execute);
    }

    #[test]
    fn ferrum_run_preflight_does_not_execute() {
        let parsed = parse_ferrum_run_args(&[
            "--solver=incompressibleFluid".to_string(),
            "--preflight".to_string(),
        ])
        .expect("ferrumRun preflight args should parse");

        assert!(!parsed.execute);
        assert_eq!(parsed.forwarded_args, vec!["--preflight"]);
    }

    #[test]
    fn inspection_modes_do_not_select_or_mix_with_solver_kernels() {
        for mode in [
            "-preflight",
            "--preflight",
            "-dryRun",
            "--dry-run",
            "-runnerDryRun",
            "--runnerDryRun",
            "-runner-dry-run",
            "--runner-dry-run",
        ] {
            let parsed = parse_incompressible_fluid_plan_args(&[mode.to_string()])
                .expect("incompressibleFluid plan-only args should parse");

            assert!(parsed.laminar_simple_solve.is_none(), "mode: {mode}");
            assert!(parsed.scalar_diffusion_solve.is_none(), "mode: {mode}");

            let error = parse_solver_args(&[
                mode.to_string(),
                "--solveScalarDiffusion".to_string(),
                "T".to_string(),
            ])
            .expect_err("inspection and scalar solve modes must be mutually exclusive");
            assert!(
                error.contains("cannot be combined"),
                "mode: {mode}: {error}"
            );
        }
    }

    #[test]
    fn ferrum_run_rejects_duplicate_solver_selection() {
        let error = parse_ferrum_run_args(&[
            "-solver".to_string(),
            "incompressibleFluid".to_string(),
            "--solver=incompressibleFluid".to_string(),
        ])
        .expect_err("duplicate solver selection should fail");

        assert!(error.contains("only once"));
    }

    #[test]
    fn removed_simple_selector_is_not_a_utility_mode() {
        let error = parse_solver_args(&["--solveLaminarSimple".to_string()])
            .expect_err("removed public selector must stay unavailable");

        assert!(error.contains("unknown ferrum solve option"));
    }

    #[test]
    fn control_solver_fallback_requires_explicit_ferrum_application() {
        let mut plan = laminar_simple_test_plan(1000.0, 0.001);
        plan.control.application = None;

        let error = resolve_solver_dispatch(None, Some(&plan.control))
            .expect_err("unmarked controlDict must not select a Ferrum solver");

        assert!(error.contains("application ferrumRun"));
    }

    #[test]
    fn solver_dispatch_records_selection_source() {
        let plan = laminar_simple_test_plan(1000.0, 0.001);
        let from_control = resolve_solver_dispatch(None, Some(&plan.control))
            .expect("marked Ferrum controlDict should select its solver");
        let from_cli = resolve_solver_dispatch(Some("incompressibleFluid".to_string()), None)
            .expect("explicit CLI solver should be accepted");

        assert_eq!(from_control.source, SolverSelectionSource::ControlDict);
        assert_eq!(from_cli.source, SolverSelectionSource::Cli);
    }

    #[test]
    fn ferrum_branded_workflow_commands_are_canonical() {
        for command in [
            "initFerrumCase",
            "gmshToFerrum",
            "checkFerrumMesh",
            "splitFerrumMeshRegions",
            "run",
        ] {
            assert!(
                run_ferrum_subcommand(vec![command.to_string(), "--help".to_string()]).is_ok(),
                "canonical command {command} should dispatch"
            );
        }
    }

    #[test]
    fn legacy_openfoam_style_command_names_report_their_replacements() {
        for (legacy, replacement) in [
            ("initCase", "initFerrumCase"),
            ("gmshToFoam", "gmshToFerrum"),
            ("gmshToFerrumFoam", "gmshToFerrum"),
            ("checkMesh", "checkFerrumMesh"),
            ("splitMeshRegions", "splitFerrumMeshRegions"),
        ] {
            let error = run_ferrum_subcommand(vec![legacy.to_string()])
                .expect_err("legacy command should not dispatch");
            assert!(
                error.contains(replacement),
                "legacy command {legacy} should point to {replacement}, got {error}"
            );
        }
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
    fn max_runner_steps_cap_is_exact() {
        let accepted = parse_solver_args(&[
            "--maxRunnerSteps".to_string(),
            MAX_RUNNER_DRY_RUN_STEPS.to_string(),
        ])
        .expect("exact runner preview cap should parse");
        assert_eq!(accepted.max_runner_steps, MAX_RUNNER_DRY_RUN_STEPS);

        for rejected in [MAX_RUNNER_DRY_RUN_STEPS + 1, usize::MAX] {
            let error = parse_solver_args(&["--maxRunnerSteps".to_string(), rejected.to_string()])
                .expect_err("runner preview above the cap must fail");
            assert!(error.contains("must not exceed 1000"));
        }
    }

    #[test]
    fn parses_scalar_diffusion_solve_options() {
        let args = vec![
            "-case".to_string(),
            "tutorials/incompressibleFluid/laminarPipe/ferrum/case".to_string(),
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
    fn parses_laminar_simple_solve_options() {
        let args = vec![
            "--rho".to_string(),
            "998.2".to_string(),
            "--mu".to_string(),
            "0.001002".to_string(),
            "--momentumExecution".to_string(),
            "parallel-components".to_string(),
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
            "--profileGamg".to_string(),
            "--profilePcg".to_string(),
            "--solveReportJson".to_string(),
            "target/simple.json".to_string(),
            "--solveReportMarkdown".to_string(),
            "target/simple.md".to_string(),
            "--writeFinalFields".to_string(),
            "target/simpleFields/1".to_string(),
        ];

        let parsed = parse_incompressible_fluid_args(&args).expect("solver args should parse");
        let solve = parsed
            .laminar_simple_solve
            .expect("laminar SIMPLE solve args");

        assert_eq!(solve.density, Some(998.2));
        assert_eq!(solve.dynamic_viscosity, Some(0.001002));
        assert_eq!(solve.linear_solver, None);
        assert_eq!(solve.momentum_linear_solver, None);
        assert_eq!(
            solve.momentum_execution,
            Some(LaminarSimpleMomentumExecution::ParallelComponents)
        );
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
        assert!(solve.profile_gamg);
        assert!(solve.profile_pcg);
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
    fn parses_atomic_wall_force_options_for_incompressible_execution() {
        let parsed = parse_incompressible_fluid_args(&[
            "--wallForcePatches".to_string(),
            " cylinder, innerWall ".to_string(),
            "--forceReferenceSpeed".to_string(),
            "0.015".to_string(),
            "--forceReferenceArea".to_string(),
            "1e-6".to_string(),
        ])
        .expect("complete wall-force options should parse");
        let wall_forces = parsed
            .laminar_simple_solve
            .expect("incompressible solve")
            .wall_forces
            .expect("wall-force request");

        assert_eq!(wall_forces.patch_names, ["cylinder", "innerWall"]);
        assert_eq!(wall_forces.reference_speed, 0.015);
        assert_eq!(wall_forces.reference_area, 1.0e-6);
        assert_eq!(wall_forces.face_loads_csv, None);
    }

    #[test]
    fn wall_face_loads_csv_is_atomic_optional_and_execution_only() {
        for flag in ["--wallFaceLoadsCsv", "--wall-face-loads-csv"] {
            let parsed = parse_incompressible_fluid_args(&[
                "--wallForcePatches".to_string(),
                "cylinder".to_string(),
                "--forceReferenceSpeed".to_string(),
                "0.015".to_string(),
                "--forceReferenceArea".to_string(),
                "1e-6".to_string(),
                flag.to_string(),
                "nested/loads.csv".to_string(),
            ])
            .expect("complete per-face wall-load request should parse");
            assert_eq!(
                parsed
                    .laminar_simple_solve
                    .expect("incompressible solve")
                    .wall_forces
                    .expect("wall-force request")
                    .face_loads_csv,
                Some(PathBuf::from("nested/loads.csv"))
            );
        }

        let without_wall_forces = ["--wallFaceLoadsCsv", "loads.csv"]
            .into_iter()
            .map(str::to_string)
            .collect::<Vec<_>>();
        let error = parse_incompressible_fluid_args(&without_wall_forces)
            .expect_err("CSV output without wall forces must fail");
        assert!(error.contains("requires --wallForcePatches"));
        assert!(parse_solver_args(&without_wall_forces).is_err());
        assert!(parse_incompressible_fluid_plan_args(&without_wall_forces).is_err());

        for args in [
            vec!["--wallFaceLoadsCsv"],
            vec!["--wallFaceLoadsCsv", ""],
            vec![
                "--wallForcePatches",
                "cylinder",
                "--forceReferenceSpeed",
                "1",
                "--forceReferenceArea",
                "1",
                "--wallFaceLoadsCsv",
                "first.csv",
                "--wallFaceLoadsCsv",
                "second.csv",
            ],
        ] {
            let args = args.into_iter().map(str::to_string).collect::<Vec<_>>();
            parse_incompressible_fluid_args(&args)
                .expect_err("missing, empty, or duplicate CSV path must fail");
        }
    }

    #[test]
    fn wall_face_loads_csv_rejects_output_namespace_collisions_before_solve() {
        let base = output_test_dir("wall-face-loads-collisions");
        std::fs::create_dir_all(&base).expect("output root should be created");
        let output_root = SafeOutputRoot::open_existing(&base).expect("output root should open");

        for extra in [
            vec!["--solveResidualCsv", "same.csv"],
            vec!["--solveResidualPlot", "plots/history.svg"],
            vec!["--writeFinalFields", "final"],
            vec!["--solveReportJson", "reports/report.json"],
        ] {
            let wall_path = match extra[0] {
                "--solveResidualCsv" => "same.csv",
                "--solveResidualPlot" => "plots/history.csv",
                "--writeFinalFields" => "final/U",
                "--solveReportJson" => "reports",
                _ => unreachable!(),
            };
            let mut args = vec![
                "--wallForcePatches",
                "cylinder",
                "--forceReferenceSpeed",
                "1",
                "--forceReferenceArea",
                "1",
                "--wallFaceLoadsCsv",
                wall_path,
            ];
            args.extend(extra);
            let args = args.into_iter().map(str::to_string).collect::<Vec<_>>();
            let solve = parse_incompressible_fluid_args(&args)
                .expect("colliding outputs should pass syntax parsing")
                .laminar_simple_solve
                .expect("incompressible solve");
            let path = solve
                .wall_forces
                .as_ref()
                .and_then(|request| request.face_loads_csv.as_deref())
                .expect("wall-face CSV path");
            let error =
                validate_wall_face_loads_output_destination_in_root(&solve, path, &output_root)
                    .expect_err("colliding output paths must fail before the solve");
            assert!(error.contains("collides"), "unexpected error: {error}");
        }

        let args = [
            "--wallForcePatches",
            "cylinder",
            "--forceReferenceSpeed",
            "1",
            "--forceReferenceArea",
            "1",
            "--wallFaceLoadsCsv",
            "final/audit.csv",
            "--writeFinalFields",
            "final",
            "--solveReportJson",
            "reports/report.json",
        ]
        .into_iter()
        .map(str::to_string)
        .collect::<Vec<_>>();
        let solve = parse_incompressible_fluid_args(&args)
            .expect("non-colliding outputs should parse")
            .laminar_simple_solve
            .expect("incompressible solve");
        let path = solve
            .wall_forces
            .as_ref()
            .and_then(|request| request.face_loads_csv.as_deref())
            .expect("wall-face CSV path");
        validate_wall_face_loads_output_destination_in_root(&solve, path, &output_root)
            .expect("separate output paths must remain valid");

        let _ = std::fs::remove_dir_all(base);
    }

    #[test]
    fn wall_force_options_reject_partial_duplicate_empty_and_nonpositive_inputs() {
        for args in [
            vec!["--wallForcePatches", "cylinder"],
            vec!["--forceReferenceSpeed", "1", "--forceReferenceArea", "1"],
            vec![
                "--wallForcePatches",
                "cylinder",
                "--forceReferenceSpeed",
                "1",
            ],
        ] {
            let args = args.into_iter().map(str::to_string).collect::<Vec<_>>();
            let error = parse_incompressible_fluid_args(&args)
                .expect_err("partial wall-force options must fail");
            assert!(error.contains("must be supplied together"));
        }

        for patches in ["", "cylinder,", "cylinder,,wall", "cylinder,cylinder"] {
            let error = parse_incompressible_fluid_args(&[
                "--wallForcePatches".to_string(),
                patches.to_string(),
                "--forceReferenceSpeed".to_string(),
                "1".to_string(),
                "--forceReferenceArea".to_string(),
                "1".to_string(),
            ])
            .expect_err("invalid patch list must fail");
            assert!(
                error.contains("patch") || error.contains("Patch"),
                "unexpected error: {error}"
            );
        }

        for (flag, value) in [
            ("--forceReferenceSpeed", "0"),
            ("--forceReferenceSpeed", "NaN"),
            ("--forceReferenceArea", "-1"),
            ("--forceReferenceArea", "inf"),
        ] {
            let mut args = vec![
                "--wallForcePatches".to_string(),
                "cylinder".to_string(),
                "--forceReferenceSpeed".to_string(),
                "1".to_string(),
                "--forceReferenceArea".to_string(),
                "1".to_string(),
            ];
            let position = args
                .iter()
                .position(|arg| arg == flag)
                .expect("fixture flag");
            args[position + 1] = value.to_string();
            parse_incompressible_fluid_args(&args)
                .expect_err("non-positive or non-finite reference value must fail");
        }
    }

    #[test]
    fn wall_force_options_are_execution_only() {
        let args = [
            "--wallForcePatches".to_string(),
            "cylinder".to_string(),
            "--forceReferenceSpeed".to_string(),
            "1".to_string(),
            "--forceReferenceArea".to_string(),
            "1".to_string(),
        ];

        assert!(parse_solver_args(&args).is_err());
        assert!(parse_incompressible_fluid_plan_args(&args).is_err());
    }

    #[test]
    fn parses_laminar_simple_residual_reporting_options() {
        let args = vec![
            "--solveVerbose".to_string(),
            "--solveResidualCsv".to_string(),
            "target/simple-residuals.csv".to_string(),
            "--solveResidualPlot".to_string(),
            "target/simple-residuals.svg".to_string(),
            "--solveReportJson".to_string(),
            "target/simple.json".to_string(),
        ];

        let parsed = parse_incompressible_fluid_args(&args).expect("solver args should parse");
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
            Some(PathBuf::from("target/simple-residuals.svg"))
        );
        assert_eq!(solve.report_json, Some(PathBuf::from("target/simple.json")));
    }

    #[test]
    fn writes_laminar_simple_residual_plot_as_native_svg() {
        let unique = format!(
            "{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system clock should be after Unix epoch")
                .as_nanos()
        );
        let csv_path = std::env::temp_dir().join(format!("ferrum-residuals-{unique}.csv"));
        let svg_path = std::env::temp_dir().join(format!("ferrum-residuals-{unique}.svg"));
        std::fs::write(
            &csv_path,
            "iteration,unused,continuityAfterL2,momentumInitialResidualNormalized,unused,unused,pressureInitialResidualNormalized,unused,unused\n1,0,1e-4,1e-3,0,0,1e-2,0,0\n",
        )
        .expect("residual CSV fixture should be written");

        let output_root = SafeOutputRoot::open_existing(&std::env::temp_dir())
            .expect("temporary output root should open");
        write_laminar_simple_residual_plot(&output_root, &csv_path, &svg_path)
            .expect("native SVG residual plot should be written");
        let svg = std::fs::read_to_string(&svg_path).expect("SVG should be readable");

        let _ = std::fs::remove_file(&csv_path);
        let _ = std::fs::remove_file(&svg_path);

        assert!(svg.starts_with("<svg"));
        assert!(svg.contains("U initial residual"));
        assert!(svg.contains("p initial residual"));
        assert!(svg.contains("Continuity L2"));
    }

    #[test]
    fn accepts_mixed_case_svg_residual_plot_extension() {
        let unique = format!(
            "{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system clock should be after Unix epoch")
                .as_nanos()
        );
        let csv_path = std::env::temp_dir().join(format!("ferrum-residuals-{unique}.csv"));
        let svg_path = std::env::temp_dir().join(format!("ferrum-residuals-{unique}.SVG"));
        std::fs::write(
            &csv_path,
            "iteration,unused,continuityAfterL2,momentumInitialResidualNormalized,unused,unused,pressureInitialResidualNormalized,unused,unused\n1,0,1e-4,1e-3,0,0,1e-2,0,0\n",
        )
        .expect("residual CSV fixture should be written");

        let output_root = SafeOutputRoot::open_existing(&std::env::temp_dir())
            .expect("temporary output root should open");
        write_laminar_simple_residual_plot(&output_root, &csv_path, &svg_path)
            .expect("mixed-case SVG extension should use native rendering");
        let svg = std::fs::read_to_string(&svg_path).expect("SVG should be readable");

        let _ = std::fs::remove_file(&csv_path);
        let _ = std::fs::remove_file(&svg_path);

        assert!(svg.starts_with("<svg"));
    }

    #[test]
    fn rejects_non_svg_residual_plot_without_creating_output() {
        let unique = format!(
            "{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system clock should be after Unix epoch")
                .as_nanos()
        );
        let csv_path = std::env::temp_dir().join(format!("ferrum-residuals-{unique}.csv"));
        let png_path = std::env::temp_dir().join(format!("ferrum-residuals-{unique}.png"));

        let output_root = SafeOutputRoot::open_existing(&std::env::temp_dir())
            .expect("temporary output root should open");
        let error = write_laminar_simple_residual_plot(&output_root, &csv_path, &png_path)
            .expect_err("non-SVG residual plots must be rejected");

        assert_eq!(error.kind(), ErrorKind::InvalidInput);
        assert_eq!(
            error.to_string(),
            "native residual plots require an output path with the .svg extension"
        );
        assert!(!png_path.exists());
    }

    #[test]
    fn simple_report_and_field_outputs_replace_regular_files() {
        let base = output_test_dir("replace");
        std::fs::create_dir_all(base.join("fields/1"))
            .expect("field output directory should be created");
        std::fs::write(base.join("fields/1/U"), "old U").expect("old U should be written");
        std::fs::write(base.join("fields/1/p"), "old p").expect("old p should be written");
        std::fs::create_dir_all(base.join("reports"))
            .expect("report output directory should be created");
        std::fs::write(base.join("reports/residual.csv"), "old report")
            .expect("old report should be written");
        let output_root = SafeOutputRoot::open_existing(&base).expect("output root should open");
        let report = output_test_report();

        write_laminar_simple_residual_csv(&report, &output_root, Path::new("reports/residual.csv"))
            .expect("regular report should be replaced");
        write_laminar_simple_fields(
            &output_test_fields(),
            &report,
            &output_root,
            Path::new("fields/1"),
        )
        .expect("regular field outputs should be replaced");

        assert!(
            std::fs::read_to_string(base.join("reports/residual.csv"))
                .expect("report should be readable")
                .starts_with("iteration,continuityBeforeL2")
        );
        assert!(
            std::fs::read_to_string(base.join("fields/1/U"))
                .expect("U should be readable")
                .contains("internalField nonuniform List<vector>")
        );
        assert!(
            std::fs::read_to_string(base.join("fields/1/p"))
                .expect("p should be readable")
                .contains("internalField nonuniform List<scalar>")
        );
        let _ = std::fs::remove_dir_all(base);
    }

    #[test]
    fn simple_reports_include_relative_tolerances_and_every_linear_solve_diagnostic() {
        let base = output_test_dir("linear-reltol-reporting");
        std::fs::create_dir_all(&base).expect("output root should be created");
        let output_root = SafeOutputRoot::open_existing(&base).expect("output root should open");
        let plan = laminar_simple_test_plan(1000.0, 0.001);
        let mut options = minimal_laminar_simple_options_for_estimate();
        options.momentum_linear_relative_tolerance = 0.125;
        options.pressure_linear_relative_tolerance = 1.25;
        let mut report = output_test_report();
        let relative_solve = LinearSolveConvergenceSummary {
            iterations: 3,
            converged: true,
            initial_normalized_residual_norm: 0.5,
            residual_norm: 0.01,
            normalized_residual_norm: 0.05,
            effective_normalized_tolerance: 0.125,
            stop_reason: LinearSolveStopReason::RelativeTolerance,
        };
        let absolute_solve = LinearSolveConvergenceSummary {
            iterations: 2,
            converged: true,
            initial_normalized_residual_norm: 0.25,
            residual_norm: 0.001,
            normalized_residual_norm: 0.01,
            effective_normalized_tolerance: 0.02,
            stop_reason: LinearSolveStopReason::AbsoluteTolerance,
        };
        let mut iteration = build_laminar_simple_iteration_summary(1, 0.1, 0.5, 0.25);
        iteration.momentum_component_linear_solves = [
            relative_solve,
            absolute_solve,
            LinearSolveConvergenceSummary {
                stop_reason: LinearSolveStopReason::ExactZero,
                converged: true,
                ..LinearSolveConvergenceSummary::default()
            },
        ];
        iteration.pressure_linear_solves = 2;
        iteration.pressure_correction_linear_solves[0] = relative_solve;
        iteration.pressure_correction_linear_solves[1] = absolute_solve;
        report.simple_iterations = 1;
        report.history.push(iteration);

        write_laminar_simple_report_json(
            &plan,
            &options,
            &report,
            &output_test_post_processing(),
            LaminarSimpleReportRunMetadata {
                profile_pcg: false,
                wall_clock_seconds: 0.0,
            },
            &output_root,
            Path::new("report.json"),
        )
        .expect("JSON report should be written");
        write_laminar_simple_report_markdown(
            &plan,
            &options,
            &report,
            &output_test_post_processing(),
            LaminarSimpleReportRunMetadata {
                profile_pcg: false,
                wall_clock_seconds: 0.0,
            },
            &output_root,
            Path::new("report.md"),
        )
        .expect("Markdown report should be written");
        write_laminar_simple_report_json(
            &plan,
            &options,
            &report,
            &output_test_post_processing(),
            LaminarSimpleReportRunMetadata {
                profile_pcg: true,
                wall_clock_seconds: 0.0,
            },
            &output_root,
            Path::new("profiled-report.json"),
        )
        .expect("profiled JSON report should be written");
        write_laminar_simple_report_markdown(
            &plan,
            &options,
            &report,
            &output_test_post_processing(),
            LaminarSimpleReportRunMetadata {
                profile_pcg: true,
                wall_clock_seconds: 0.0,
            },
            &output_root,
            Path::new("profiled-report.md"),
        )
        .expect("profiled Markdown report should be written");
        write_laminar_simple_residual_csv(&report, &output_root, Path::new("residual.csv"))
            .expect("CSV report should be written");

        let json = std::fs::read_to_string(base.join("report.json"))
            .expect("JSON report should be readable");
        let markdown = std::fs::read_to_string(base.join("report.md"))
            .expect("Markdown report should be readable");
        let profiled_json = std::fs::read_to_string(base.join("profiled-report.json"))
            .expect("profiled JSON report should be readable");
        let profiled_markdown = std::fs::read_to_string(base.join("profiled-report.md"))
            .expect("profiled Markdown report should be readable");
        let csv = std::fs::read_to_string(base.join("residual.csv"))
            .expect("CSV report should be readable");
        assert!(json.contains("\"momentumLinearRelativeTolerance\": 0.125"));
        assert!(json.contains("\"momentumExecution\": \"serial\""));
        assert!(json.contains("\"pressureLinearRelativeTolerance\": 1.25"));
        assert!(json.contains("\"profilePcg\": false"));
        assert!(!markdown.contains("## Pressure PCG Kernel Profile"));
        assert!(profiled_json.contains("\"profilePcg\": true"));
        assert!(profiled_markdown.contains("## Pressure PCG Kernel Profile"));
        assert!(json.contains("\"momentumComponentLinearSolves\""));
        assert!(json.contains("\"pressureCorrectionLinearSolves\""));
        assert!(json.contains("\"correction\": 1"));
        assert!(json.contains("\"correction\": 2"));
        assert!(json.contains("\"effectiveNormalizedTolerance\": 0.125"));
        assert!(json.contains("\"stopReason\": \"RelativeTolerance\""));
        assert!(markdown.contains("| Momentum linear relative tolerance | 1.250000e-1 |"));
        assert!(markdown.contains("| Momentum execution | serial |"));
        assert!(markdown.contains("| Pressure linear relative tolerance | 1.250000e0 |"));
        assert!(markdown.contains("| U solve diagnostics | p solve diagnostics |"));
        assert!(markdown.contains(
            "x:iter=3/ok=true/initial=5.000000e-1/residual=1.000000e-2/final=5.000000e-2/effectiveTol=1.250000e-1/reason=RelativeTolerance"
        ));
        assert!(markdown.contains(
            "1:iter=3/ok=true/initial=5.000000e-1/residual=1.000000e-2/final=5.000000e-2/effectiveTol=1.250000e-1/reason=RelativeTolerance"
        ));
        let mut csv_lines = csv.lines();
        let csv_header = csv_lines.next().expect("CSV header");
        let csv_row = csv_lines.next().expect("CSV history row");
        assert!(csv_header.ends_with(
            "momentumComponentLinearSolveDiagnostics,pressureCorrectionLinearSolveDiagnostics"
        ));
        assert_eq!(csv_header.split(',').count(), csv_row.split(',').count());
        assert!(csv_row.contains(
            "x:iter=3/ok=true/initial=5.000000e-1/residual=1.000000e-2/final=5.000000e-2/effectiveTol=1.250000e-1/reason=RelativeTolerance"
        ));
        assert!(csv_row.contains(
            "2:iter=2/ok=true/initial=2.500000e-1/residual=1.000000e-3/final=1.000000e-2/effectiveTol=2.000000e-2/reason=AbsoluteTolerance"
        ));

        let _ = std::fs::remove_dir_all(base);
    }

    #[test]
    fn simple_json_report_exposes_fixed_size_first_nonconvergence_diagnostics() {
        let empty_report = output_test_report();
        let mut empty_diagnostics = Vec::new();
        super::write_json_linear_nonconvergence_diagnostics(&mut empty_diagnostics, &empty_report)
            .expect("empty diagnostics JSON");
        assert_eq!(
            String::from_utf8(empty_diagnostics).expect("UTF-8 diagnostics"),
            concat!(
                "  \"linearNonConvergenceDiagnostics\": {\n",
                "    \"firstMomentum\": null,\n",
                "    \"firstPressure\": null\n",
                "  }"
            )
        );

        let mut nonrepresentable_row = Vec::new();
        super::write_json_algebraic_residual_row(
            &mut nonrepresentable_row,
            0,
            &AlgebraicResidualRowDiagnostic::default(),
        )
        .expect("nonrepresentable row JSON");
        assert_eq!(
            String::from_utf8(nonrepresentable_row).expect("UTF-8 row diagnostic"),
            concat!(
                "{\n",
                "  \"representable\": false,\n",
                "  \"row\": null,\n",
                "  \"cellVolume\": null,\n",
                "  \"diagonal\": null,\n",
                "  \"rowSumAbs\": null,\n",
                "  \"offDiagonalSumAbs\": null,\n",
                "  \"rhs\": null,\n",
                "  \"matrixProduct\": null,\n",
                "  \"residual\": null,\n",
                "  \"initialValue\": null,\n",
                "  \"candidateValue\": null,\n",
                "  \"incidentFaces\": {\n",
                "    \"representable\": false,\n",
                "    \"count\": 0,\n",
                "    \"signedSum\": null,\n",
                "    \"sumAbs\": null,\n",
                "    \"min\": null,\n",
                "    \"max\": null,\n",
                "    \"maxAbsFaceFlux\": null,\n",
                "    \"maxAbsFaceIndex\": null,\n",
                "    \"maxAbsFaceOwner\": null,\n",
                "    \"maxAbsFaceNeighbour\": null,\n",
                "    \"maxAbsFacePatchIndex\": null\n",
                "  }\n",
                "}"
            )
        );

        let base = output_test_dir("first-linear-nonconvergence");
        std::fs::create_dir_all(&base).expect("output root should be created");
        let output_root = SafeOutputRoot::open_existing(&base).expect("output root should open");
        let plan = laminar_simple_test_plan(1000.0, 0.001);
        let options = minimal_laminar_simple_options_for_estimate();
        let populated_snapshot = LinearNonConvergenceSnapshot {
            simple_iteration: 7,
            solve_ordinal: 2,
            solver: LaminarSimpleLinearSolver::Pcg,
            preconditioner: LaminarSimplePreconditioner::Dic,
            solve: LinearSolveConvergenceSummary {
                iterations: 1000,
                converged: false,
                initial_normalized_residual_norm: 1.0,
                residual_norm: 0.25,
                normalized_residual_norm: 0.125,
                effective_normalized_tolerance: 1.0e-9,
                stop_reason: LinearSolveStopReason::MaxIterations,
            },
            linear_iterate_accepted: true,
            continuity_before: ContinuitySummary {
                l2_norm: 1.0,
                ..ContinuitySummary::default()
            },
            continuity_star: None,
            continuity_after: None,
            worst_algebraic_residual_row: AlgebraicResidualRowDiagnostic {
                representable: true,
                row: Some(42),
                cell_volume: Some(0.5),
                diagonal: Some(4.0),
                row_sum_abs: Some(6.0),
                off_diagonal_sum_abs: Some(2.0),
                rhs: Some(3.0),
                matrix_product: Some(2.5),
                residual: Some(0.5),
                initial_value: Some(0.0),
                candidate_value: Some(1.0),
                incident_faces: IncidentFaceDiagnosticSummary {
                    representable: true,
                    count: 3,
                    signed_sum: Some(0.25),
                    sum_abs: Some(1.25),
                    min: Some(-0.5),
                    max: Some(0.75),
                    max_abs_face_index: Some(9),
                    max_abs_face_owner: Some(42),
                    max_abs_face_neighbour: Some(43),
                    max_abs_face_patch_index: None,
                    max_abs_face_flux: Some(0.75),
                },
            },
        };
        let mut populated_snapshot_json = Vec::new();
        super::write_json_linear_nonconvergence_snapshot(
            &mut populated_snapshot_json,
            0,
            Some(&populated_snapshot),
        )
        .expect("populated snapshot JSON");
        assert_eq!(
            String::from_utf8(populated_snapshot_json).expect("UTF-8 populated snapshot"),
            concat!(
                "{\n",
                "  \"simpleIteration\": 7,\n",
                "  \"solveOrdinal\": 2,\n",
                "  \"solver\": \"pcg\",\n",
                "  \"preconditioner\": \"DIC\",\n",
                "  \"linearIterateAccepted\": true,\n",
                "  \"solve\": {\n",
                "    \"iterations\": 1000,\n",
                "    \"converged\": false,\n",
                "    \"initialNormalizedResidual\": 1,\n",
                "    \"residualNorm\": 0.25,\n",
                "    \"normalizedResidual\": 0.125,\n",
                "    \"effectiveNormalizedTolerance\": 0.000000001,\n",
                "    \"stopReason\": \"MaxIterations\"\n",
                "  },\n",
                "  \"continuityBefore\": {\n",
                "    \"l2Norm\": 1,\n",
                "    \"maxAbs\": 0,\n",
                "    \"sumAbs\": 0,\n",
                "    \"globalSum\": 0\n",
                "  },\n",
                "  \"continuityStar\": null,\n",
                "  \"continuityAfter\": null,\n",
                "  \"worstAlgebraicResidualRow\": {\n",
                "    \"representable\": true,\n",
                "    \"row\": 42,\n",
                "    \"cellVolume\": 0.5,\n",
                "    \"diagonal\": 4,\n",
                "    \"rowSumAbs\": 6,\n",
                "    \"offDiagonalSumAbs\": 2,\n",
                "    \"rhs\": 3,\n",
                "    \"matrixProduct\": 2.5,\n",
                "    \"residual\": 0.5,\n",
                "    \"initialValue\": 0,\n",
                "    \"candidateValue\": 1,\n",
                "    \"incidentFaces\": {\n",
                "      \"representable\": true,\n",
                "      \"count\": 3,\n",
                "      \"signedSum\": 0.25,\n",
                "      \"sumAbs\": 1.25,\n",
                "      \"min\": -0.5,\n",
                "      \"max\": 0.75,\n",
                "      \"maxAbsFaceFlux\": 0.75,\n",
                "      \"maxAbsFaceIndex\": 9,\n",
                "      \"maxAbsFaceOwner\": 42,\n",
                "      \"maxAbsFaceNeighbour\": 43,\n",
                "      \"maxAbsFacePatchIndex\": null\n",
                "    }\n",
                "  }\n",
                "}"
            )
        );

        let mut report = output_test_report();
        report.first_pressure_linear_nonconvergence = Some(populated_snapshot);

        write_laminar_simple_report_json(
            &plan,
            &options,
            &report,
            &output_test_post_processing(),
            LaminarSimpleReportRunMetadata {
                profile_pcg: false,
                wall_clock_seconds: 0.0,
            },
            &output_root,
            Path::new("report.json"),
        )
        .expect("JSON report should be written");

        let json = std::fs::read_to_string(base.join("report.json"))
            .expect("JSON report should be readable");
        assert!(json.contains("\"linearNonConvergenceDiagnostics\""));
        assert!(json.contains("\"firstMomentum\": null"));
        assert!(json.contains("\"firstPressure\": {"));
        assert!(json.contains("\"simpleIteration\": 7"));
        assert!(json.contains("\"solveOrdinal\": 2"));
        assert!(json.contains("\"linearIterateAccepted\": true"));
        assert!(json.contains("\"worstAlgebraicResidualRow\""));
        assert!(json.contains("\"row\": 42"));
        assert!(json.contains("\"maxAbsFacePatchIndex\": null"));

        let _ = std::fs::remove_dir_all(base);
    }

    #[test]
    fn gamg_hierarchy_json_contract_is_exact_ordered_and_decimal_encoded() {
        let profile = hierarchy_profile_fixture();
        let mut bytes = Vec::new();
        write_json_gamg_hierarchy(&mut bytes, 0, &profile)
            .expect("hierarchy JSON should be written");
        let json = String::from_utf8(bytes).expect("hierarchy JSON should be UTF-8");

        assert_eq!(
            json,
            concat!(
                "\"hierarchy\": {\n",
                "  \"smootherPassesPerSweep\": 2,\n",
                "  \"directSolveCoarsest\": true,\n",
                "  \"gridComplexityNumeratorDecimal\": \"14\",\n",
                "  \"gridComplexityDenominatorDecimal\": \"8\",\n",
                "  \"operatorComplexityNumeratorDecimal\": \"38\",\n",
                "  \"operatorComplexityDenominatorDecimal\": \"24\",\n",
                "  \"nnzWeightedSmoothingWorkDecimal\": \"224\",\n",
                "  \"nnzWeightedSparseWorkDecimal\": \"878\",\n",
                "  \"levels\": [\n",
                "    {\n",
                "      \"level\": 0,\n",
                "      \"cells\": 8,\n",
                "      \"nonzeros\": 24\n",
                "    },\n",
                "    {\n",
                "      \"level\": 1,\n",
                "      \"cells\": 4,\n",
                "      \"nonzeros\": 10\n",
                "    },\n",
                "    {\n",
                "      \"level\": 2,\n",
                "      \"cells\": 2,\n",
                "      \"nonzeros\": 4\n",
                "    }\n",
                "  ],\n",
                "  \"transfers\": [\n",
                "    {\n",
                "      \"fineLevel\": 0,\n",
                "      \"coarseLevel\": 1,\n",
                "      \"fineCells\": 8,\n",
                "      \"coarseCells\": 4,\n",
                "      \"singletonFineCells\": 0,\n",
                "      \"unmatchedFineCells\": 0,\n",
                "      \"minAggregateSize\": 2,\n",
                "      \"maxAggregateSize\": 2,\n",
                "      \"aggregateSizeHistogram\": [\n",
                "        {\n",
                "          \"aggregateSize\": 2,\n",
                "          \"aggregateCount\": 4\n",
                "        }\n",
                "      ]\n",
                "    },\n",
                "    {\n",
                "      \"fineLevel\": 1,\n",
                "      \"coarseLevel\": 2,\n",
                "      \"fineCells\": 4,\n",
                "      \"coarseCells\": 2,\n",
                "      \"singletonFineCells\": 1,\n",
                "      \"unmatchedFineCells\": 1,\n",
                "      \"minAggregateSize\": 1,\n",
                "      \"maxAggregateSize\": 3,\n",
                "      \"aggregateSizeHistogram\": [\n",
                "        {\n",
                "          \"aggregateSize\": 1,\n",
                "          \"aggregateCount\": 1\n",
                "        },\n",
                "        {\n",
                "          \"aggregateSize\": 3,\n",
                "          \"aggregateCount\": 1\n",
                "        }\n",
                "      ]\n",
                "    }\n",
                "  ]\n",
                "}"
            )
        );
    }

    #[test]
    fn gamg_hierarchy_json_uses_null_for_unavailable_exact_counts() {
        let profile = GamgKernelTiming {
            hierarchy: Some(std::sync::Arc::new(GamgHierarchyDiagnostics {
                levels: vec![GamgHierarchyLevelDiagnostics {
                    level: 0,
                    cells: 0,
                    nonzeros: 0,
                }],
                transfers: Vec::new(),
                smoother_passes_per_sweep: 1,
                direct_solve_coarsest: false,
            })),
            ..GamgKernelTiming::default()
        };
        let mut bytes = Vec::new();
        write_json_gamg_hierarchy(&mut bytes, 0, &profile)
            .expect("unavailable hierarchy counts should still serialize");
        let json = String::from_utf8(bytes).expect("hierarchy JSON should be UTF-8");

        assert_eq!(
            json,
            concat!(
                "\"hierarchy\": {\n",
                "  \"smootherPassesPerSweep\": 1,\n",
                "  \"directSolveCoarsest\": false,\n",
                "  \"gridComplexityNumeratorDecimal\": null,\n",
                "  \"gridComplexityDenominatorDecimal\": null,\n",
                "  \"operatorComplexityNumeratorDecimal\": null,\n",
                "  \"operatorComplexityDenominatorDecimal\": null,\n",
                "  \"nnzWeightedSmoothingWorkDecimal\": null,\n",
                "  \"nnzWeightedSparseWorkDecimal\": null,\n",
                "  \"levels\": [\n",
                "    {\n",
                "      \"level\": 0,\n",
                "      \"cells\": 0,\n",
                "      \"nonzeros\": 0\n",
                "    }\n",
                "  ],\n",
                "  \"transfers\": [\n",
                "  ]\n",
                "}"
            )
        );
        assert_eq!(json.matches(": null").count(), 6);
    }

    #[test]
    fn gamg_exact_count_formatting_is_lossless_at_u128_max_and_explicit_when_unavailable() {
        let mut bytes = Vec::new();
        write_json_optional_u128_decimal(&mut bytes, Some(u128::MAX))
            .expect("u128 max should serialize as a decimal JSON string");
        assert_eq!(
            String::from_utf8(bytes).expect("decimal JSON should be UTF-8"),
            "\"340282366920938463463374607431768211455\""
        );

        let mut bytes = Vec::new();
        write_json_optional_u128_decimal(&mut bytes, None)
            .expect("unavailable count should serialize as null");
        assert_eq!(
            String::from_utf8(bytes).expect("null JSON should be UTF-8"),
            "null"
        );
        assert_eq!(
            format_optional_u128(Some(u128::MAX)),
            "340282366920938463463374607431768211455"
        );
        assert_eq!(format_optional_u128(None), "n/a");
    }

    #[test]
    fn gamg_reports_expose_resolved_outer_solver() {
        let base = output_test_dir("gamg-outer-solver-reporting");
        std::fs::create_dir_all(&base).expect("output root should be created");
        let output_root = SafeOutputRoot::open_existing(&base).expect("output root should open");
        let plan = laminar_simple_test_plan(1000.0, 0.001);
        let mut options = minimal_laminar_simple_options_for_estimate();
        options.pressure_linear_solver = LaminarSimpleLinearSolver::Gamg;
        options.pressure_gamg_options = Some(GamgOptions {
            outer_solver: GamgOuterSolver::FlexibleCg,
            ..GamgOptions::default()
        });
        let mut report = output_test_report();
        report.timing.pressure_gamg_profile = Some(hierarchy_profile_fixture());

        write_laminar_simple_report_json(
            &plan,
            &options,
            &report,
            &output_test_post_processing(),
            LaminarSimpleReportRunMetadata {
                profile_pcg: false,
                wall_clock_seconds: 0.0,
            },
            &output_root,
            Path::new("report.json"),
        )
        .expect("JSON report should be written");
        write_laminar_simple_report_markdown(
            &plan,
            &options,
            &report,
            &output_test_post_processing(),
            LaminarSimpleReportRunMetadata {
                profile_pcg: false,
                wall_clock_seconds: 0.0,
            },
            &output_root,
            Path::new("report.md"),
        )
        .expect("Markdown report should be written");

        let json = std::fs::read_to_string(base.join("report.json"))
            .expect("JSON report should be readable");
        let markdown = std::fs::read_to_string(base.join("report.md"))
            .expect("Markdown report should be readable");
        assert!(json.contains("\"outerSolver\": \"FCG\""));
        assert!(json.contains("\"outerMatrixVectorProducts\": 17"));
        assert!(json.contains("\"outerReductions\": 23"));
        assert!(json.contains("\"hierarchyDiagnosticSeconds\": 0.125"));
        assert!(json.contains("\"nnzWeightedSmoothingWorkDecimal\": \"224\""));
        assert!(json.contains("\"nnzWeightedSparseWorkDecimal\": \"878\""));
        assert!(json.contains("\"coarsestIterations\": 7"));
        assert!(markdown.contains("| GAMG outer solver | FCG |"));
        assert!(markdown.contains("| Outer matrix-vector products | 17 |"));
        assert!(markdown.contains("| Outer reductions | 23 |"));
        assert!(markdown.contains("| Hierarchy diagnostics [s] | 1.250000e-1 |"));
        assert!(markdown.contains("| Coarsest iterations |"));
        assert!(markdown.contains("| Grid complexity terms | 14 / 8 |"));
        assert!(markdown.contains("| Operator complexity terms | 38 / 24 |"));
        assert!(markdown.contains("| NNZ-weighted smoothing work | 224 |"));
        assert!(markdown.contains("| NNZ-weighted sparse work | 878 |"));
        assert!(markdown.contains("| 1 | 2 | 4 | 2 | 1 | 1 | 1 | 3 |"));
        assert!(markdown.contains("| 1 | 2 | 3 | 1 |"));
        assert_eq!(
            gamg_outer_profile_counters(
                report
                    .timing
                    .pressure_gamg_profile
                    .as_ref()
                    .expect("profile should remain available")
            ),
            "outerMatrixVectorProducts=17 outerReductions=23"
        );

        let _ = std::fs::remove_dir_all(base);
    }

    #[test]
    fn simple_report_writers_reject_pressure_telemetry_count_overflow_without_panicking() {
        let base = output_test_dir("linear-telemetry-count");
        std::fs::create_dir_all(&base).expect("output root should be created");
        let output_root = SafeOutputRoot::open_existing(&base).expect("output root should open");
        let plan = laminar_simple_test_plan(1000.0, 0.001);
        let options = minimal_laminar_simple_options_for_estimate();
        let mut report = output_test_report();
        let mut iteration = build_laminar_simple_iteration_summary(1, 0.1, 0.2, 0.3);
        iteration.pressure_linear_solves = iteration.pressure_correction_linear_solves.len() + 1;
        report.history.push(iteration);

        for error in [
            write_laminar_simple_residual_csv(&report, &output_root, Path::new("overflow.csv"))
                .expect_err("CSV must reject invalid pressure telemetry count"),
            write_laminar_simple_report_json(
                &plan,
                &options,
                &report,
                &output_test_post_processing(),
                LaminarSimpleReportRunMetadata {
                    profile_pcg: false,
                    wall_clock_seconds: 0.0,
                },
                &output_root,
                Path::new("overflow.json"),
            )
            .expect_err("JSON must reject invalid pressure telemetry count"),
            write_laminar_simple_report_markdown(
                &plan,
                &options,
                &report,
                &output_test_post_processing(),
                LaminarSimpleReportRunMetadata {
                    profile_pcg: false,
                    wall_clock_seconds: 0.0,
                },
                &output_root,
                Path::new("overflow.md"),
            )
            .expect_err("Markdown must reject invalid pressure telemetry count"),
        ] {
            assert_eq!(error.kind(), ErrorKind::InvalidData);
            assert!(error.to_string().contains("exceeding telemetry capacity"));
        }

        let _ = std::fs::remove_dir_all(base);
    }

    #[test]
    fn simple_report_output_rejects_root_escape() {
        let base = output_test_dir("escape-root");
        let outside = output_test_dir("escape-outside");
        std::fs::create_dir_all(&base).expect("output root should be created");
        std::fs::create_dir_all(&outside).expect("outside root should be created");
        let output_root = SafeOutputRoot::open_existing(&base).expect("output root should open");
        let report = output_test_report();

        let parent_error =
            write_laminar_simple_residual_csv(&report, &output_root, Path::new("../escaped.csv"))
                .expect_err("parent traversal must be rejected");
        let absolute_error =
            write_laminar_simple_residual_csv(&report, &output_root, &outside.join("escaped.csv"))
                .expect_err("absolute root escape must be rejected");

        assert_eq!(parent_error.kind(), ErrorKind::InvalidInput);
        assert_eq!(absolute_error.kind(), ErrorKind::PermissionDenied);
        assert!(
            !base
                .parent()
                .unwrap_or(Path::new(""))
                .join("escaped.csv")
                .exists()
        );
        assert!(!outside.join("escaped.csv").exists());
        let _ = std::fs::remove_dir_all(base);
        let _ = std::fs::remove_dir_all(outside);
    }

    #[test]
    fn simple_report_output_detaches_external_hard_link() {
        let base = output_test_dir("report-hard-link");
        let outside = output_test_dir("report-hard-link-outside");
        std::fs::create_dir_all(&base).expect("output root should be created");
        std::fs::create_dir_all(&outside).expect("outside root should be created");
        let target = outside.join("target.csv");
        std::fs::write(&target, "external").expect("outside target should be written");
        std::fs::hard_link(&target, base.join("report.csv"))
            .expect("hard link fixture should be created");
        let output_root = SafeOutputRoot::open_existing(&base).expect("output root should open");

        write_laminar_simple_residual_csv(
            &output_test_report(),
            &output_root,
            Path::new("report.csv"),
        )
        .expect("local hard-link entry should be replaced");

        assert_eq!(
            std::fs::read_to_string(&target).expect("outside target should be readable"),
            "external"
        );
        assert!(
            std::fs::read_to_string(base.join("report.csv"))
                .expect("replacement report should be readable")
                .starts_with("iteration,continuityBeforeL2")
        );
        let _ = std::fs::remove_dir_all(base);
        let _ = std::fs::remove_dir_all(outside);
    }

    #[cfg(any(unix, windows))]
    #[test]
    fn simple_report_output_rejects_final_symlink_without_external_mutation() {
        let base = output_test_dir("report-symlink");
        let outside = output_test_dir("report-symlink-outside");
        std::fs::create_dir_all(&base).expect("output root should be created");
        std::fs::create_dir_all(&outside).expect("outside root should be created");
        let target = outside.join("target.csv");
        let link = base.join("report.csv");
        std::fs::write(&target, "external").expect("outside target should be written");
        if create_output_file_symlink(&target, &link).is_err() {
            let _ = std::fs::remove_dir_all(base);
            let _ = std::fs::remove_dir_all(outside);
            return;
        }
        let output_root = SafeOutputRoot::open_existing(&base).expect("output root should open");

        write_laminar_simple_residual_csv(
            &output_test_report(),
            &output_root,
            Path::new("report.csv"),
        )
        .expect_err("final report symlink must be rejected");

        assert_eq!(
            std::fs::read_to_string(&target).expect("outside target should be readable"),
            "external"
        );
        let _ = std::fs::remove_dir_all(base);
        let _ = std::fs::remove_dir_all(outside);
    }

    #[cfg(any(unix, windows))]
    #[test]
    fn simple_field_outputs_preflight_both_files_before_replacement() {
        let base = output_test_dir("field-symlink");
        let outside = output_test_dir("field-symlink-outside");
        std::fs::create_dir_all(base.join("fields"))
            .expect("field output directory should be created");
        std::fs::create_dir_all(&outside).expect("outside root should be created");
        std::fs::write(base.join("fields/U"), "old U").expect("old U should be written");
        let target = outside.join("target-p");
        let link = base.join("fields/p");
        std::fs::write(&target, "external p").expect("outside target should be written");
        if create_output_file_symlink(&target, &link).is_err() {
            let _ = std::fs::remove_dir_all(base);
            let _ = std::fs::remove_dir_all(outside);
            return;
        }
        let output_root = SafeOutputRoot::open_existing(&base).expect("output root should open");

        write_laminar_simple_fields(
            &output_test_fields(),
            &output_test_report(),
            &output_root,
            Path::new("fields"),
        )
        .expect_err("p symlink must reject the field output set");

        assert_eq!(
            std::fs::read_to_string(base.join("fields/U")).expect("U should be readable"),
            "old U"
        );
        assert_eq!(
            std::fs::read_to_string(&target).expect("outside target should be readable"),
            "external p"
        );
        let _ = std::fs::remove_dir_all(base);
        let _ = std::fs::remove_dir_all(outside);
    }

    #[test]
    fn simple_field_outputs_validate_sources_before_creating_directories() {
        let base = output_test_dir("field-source-preflight");
        std::fs::create_dir_all(&base).expect("output root should be created");
        let output_root = SafeOutputRoot::open_existing(&base).expect("output root should open");
        let mut fields = output_test_fields();
        fields
            .fields
            .retain(|field| field.name != "p" || field.region.is_some());

        write_laminar_simple_fields(
            &fields,
            &output_test_report(),
            &output_root,
            Path::new("new/1"),
        )
        .expect_err("missing p field must reject the output set");

        assert!(!base.join("new").exists());
        let _ = std::fs::remove_dir_all(base);
    }

    #[cfg(unix)]
    #[test]
    fn simple_report_output_rejects_fifo_without_blocking() {
        let base = output_test_dir("report-fifo");
        std::fs::create_dir_all(&base).expect("output root should be created");
        let fifo = base.join("report.csv");
        let status = std::process::Command::new("mkfifo")
            .arg(&fifo)
            .status()
            .expect("mkfifo should run");
        assert!(status.success(), "mkfifo should create the fixture");
        let output_root = SafeOutputRoot::open_existing(&base).expect("output root should open");

        write_laminar_simple_residual_csv(
            &output_test_report(),
            &output_root,
            Path::new("report.csv"),
        )
        .expect_err("FIFO report path must be rejected");

        assert!(fifo.exists());
        let _ = std::fs::remove_dir_all(base);
    }

    #[cfg(unix)]
    #[test]
    fn simple_field_output_rejects_symlinked_ancestor() {
        let base = output_test_dir("field-ancestor");
        let outside = output_test_dir("field-ancestor-outside");
        std::fs::create_dir_all(&base).expect("output root should be created");
        std::fs::create_dir_all(&outside).expect("outside root should be created");
        std::os::unix::fs::symlink(&outside, base.join("linked"))
            .expect("symlink fixture should be created");
        let output_root = SafeOutputRoot::open_existing(&base).expect("output root should open");

        write_laminar_simple_fields(
            &output_test_fields(),
            &output_test_report(),
            &output_root,
            Path::new("linked/1"),
        )
        .expect_err("symlinked output ancestor must be rejected");

        assert!(!outside.join("1/U").exists());
        assert!(!outside.join("1/p").exists());
        let _ = std::fs::remove_dir_all(base);
        let _ = std::fs::remove_dir_all(outside);
    }

    #[cfg(windows)]
    #[test]
    fn simple_field_output_rejects_windows_junction_ancestor() {
        let base = output_test_dir("field-junction");
        let outside = output_test_dir("field-junction-outside");
        std::fs::create_dir_all(&base).expect("output root should be created");
        std::fs::create_dir_all(&outside).expect("outside root should be created");
        let junction = base.join("linked");
        if create_output_junction(&outside, &junction).is_err() {
            let _ = std::fs::remove_dir_all(base);
            let _ = std::fs::remove_dir_all(outside);
            return;
        }
        let output_root = SafeOutputRoot::open_existing(&base).expect("output root should open");

        write_laminar_simple_fields(
            &output_test_fields(),
            &output_test_report(),
            &output_root,
            Path::new("linked/1"),
        )
        .expect_err("junction output ancestor must be rejected");

        assert!(!outside.join("1/U").exists());
        assert!(!outside.join("1/p").exists());
        let _ = std::fs::remove_dir(&junction);
        let _ = std::fs::remove_dir_all(base);
        let _ = std::fs::remove_dir_all(outside);
    }

    fn output_test_fields() -> ferrum_mesh::fields::InitialFieldSet {
        ferrum_mesh::fields::InitialFieldSet {
            case_dir: PathBuf::from("case"),
            fields: vec![
                ferrum_mesh::fields::FieldFile {
                    path: PathBuf::from("case/0/U"),
                    region: None,
                    name: "U".to_string(),
                    class_name: Some("volVectorField".to_string()),
                    dimensions: Some(vec![
                        "0".to_string(),
                        "1".to_string(),
                        "-1".to_string(),
                        "0".to_string(),
                        "0".to_string(),
                        "0".to_string(),
                        "0".to_string(),
                    ]),
                    internal_field: None,
                    boundary_patches: Vec::new(),
                },
                ferrum_mesh::fields::FieldFile {
                    path: PathBuf::from("case/0/p"),
                    region: None,
                    name: "p".to_string(),
                    class_name: Some("volScalarField".to_string()),
                    dimensions: Some(vec![
                        "1".to_string(),
                        "-1".to_string(),
                        "-2".to_string(),
                        "0".to_string(),
                        "0".to_string(),
                        "0".to_string(),
                        "0".to_string(),
                    ]),
                    internal_field: None,
                    boundary_patches: Vec::new(),
                },
            ],
        }
    }

    fn wall_force_test_fields(
        velocity_type: Option<&str>,
        pressure_type: Option<&str>,
    ) -> ferrum_mesh::fields::InitialFieldSet {
        ferrum_mesh::fields::InitialFieldSet {
            case_dir: PathBuf::from("case"),
            fields: vec![
                ferrum_mesh::fields::FieldFile {
                    path: PathBuf::from("case/0/U"),
                    region: None,
                    name: "U".to_string(),
                    class_name: Some("volVectorField".to_string()),
                    dimensions: None,
                    internal_field: None,
                    boundary_patches: vec![ferrum_mesh::fields::FieldBoundaryPatch {
                        name: "cylinder".to_string(),
                        patch_type: velocity_type.map(str::to_string),
                        inlet_value: None,
                        value: None,
                    }],
                },
                ferrum_mesh::fields::FieldFile {
                    path: PathBuf::from("case/0/p"),
                    region: None,
                    name: "p".to_string(),
                    class_name: Some("volScalarField".to_string()),
                    dimensions: None,
                    internal_field: None,
                    boundary_patches: vec![ferrum_mesh::fields::FieldBoundaryPatch {
                        name: "cylinder".to_string(),
                        patch_type: pressure_type.map(str::to_string),
                        inlet_value: None,
                        value: None,
                    }],
                },
            ],
        }
    }

    #[test]
    fn wall_force_bridge_requires_exact_no_slip_and_zero_gradient_field_conditions() {
        let selected = ["cylinder".to_string()];
        validate_wall_force_boundary_conditions(
            &wall_force_test_fields(Some("noSlip"), Some("zeroGradient")),
            &selected,
        )
        .expect("supported wall-force boundary conditions should pass");

        for (velocity_type, pressure_type, expected) in [
            (Some("fixedValue"), Some("zeroGradient"), "type noSlip"),
            (Some("noSlip"), Some("fixedValue"), "type zeroGradient"),
            (None, Some("zeroGradient"), "type noSlip"),
            (Some("noSlip"), None, "type zeroGradient"),
        ] {
            let error = validate_wall_force_boundary_conditions(
                &wall_force_test_fields(velocity_type, pressure_type),
                &selected,
            )
            .expect_err("unsupported wall-force boundary condition must fail");
            assert!(error.contains(expected), "unexpected error: {error}");
        }
    }

    fn output_test_report() -> ferrum_mesh::flow::LaminarSimpleReport {
        ferrum_mesh::flow::LaminarSimpleReport {
            cells: 1,
            faces: 0,
            internal_faces: 0,
            boundary_faces: 0,
            simple_iterations: 0,
            converged: false,
            stop_reason: LaminarSimpleStopReason::MaxIterationsReached,
            initial_continuity: Default::default(),
            final_continuity: Default::default(),
            final_momentum_initial_normalized_residual_norm: 0.0,
            final_momentum_residual_norm: 0.0,
            final_momentum_normalized_residual_norm: 0.0,
            final_pressure_correction_initial_normalized_residual_norm: 0.0,
            final_pressure_correction_residual_norm: 0.0,
            final_pressure_correction_normalized_residual_norm: 0.0,
            residual_control: Default::default(),
            total_momentum_linear_iterations: 0,
            total_pressure_linear_iterations: 0,
            linear_solve_summary: Default::default(),
            operator_summary: Default::default(),
            boundary_summary: Default::default(),
            pressure_assembly: None,
            first_momentum_linear_nonconvergence: None,
            first_pressure_linear_nonconvergence: None,
            timing: Default::default(),
            fields: Default::default(),
            final_velocity: vec![ferrum_mesh::Point3 {
                x: 1.0,
                y: 0.0,
                z: 0.0,
            }],
            final_pressure: vec![1.0],
            history: Vec::new(),
        }
    }

    fn output_test_post_processing() -> LaminarSimplePostProcessing {
        LaminarSimplePostProcessing {
            continuity_errors: Some(ContinuityErrorsReport {
                final_error: Default::default(),
                cumulative_global: 0.0,
                delta_t: 1.0,
                total_volume: 1.0,
            }),
            wall_forces: None,
        }
    }

    fn output_test_post_processing_with_forces() -> LaminarSimplePostProcessing {
        LaminarSimplePostProcessing {
            continuity_errors: Some(ContinuityErrorsReport {
                final_error: ferrum_finite_volume::continuity::NormalizedContinuity {
                    local: 1.25e-7,
                    global: -2.5e-9,
                },
                cumulative_global: -3.75e-9,
                delta_t: 0.5,
                total_volume: 4.0,
            }),
            wall_forces: Some(WallForceReport {
                patch_names: vec!["cylinder".to_string()],
                reference_speed: 0.015,
                velocity_gradient_scheme: LaminarSimpleGradientScheme::CellLimitedGaussLinear(1.0),
                coefficients: WallForceCoefficients {
                    pressure_force: ferrum_mesh::Point3 {
                        x: 1.0,
                        y: 2.0,
                        z: 3.0,
                    },
                    viscous_force: ferrum_mesh::Point3 {
                        x: 0.5,
                        y: -0.25,
                        z: 0.0,
                    },
                    total_force: ferrum_mesh::Point3 {
                        x: 1.5,
                        y: 1.75,
                        z: 3.0,
                    },
                    drag: DirectionalCoefficient {
                        pressure: 4.0,
                        viscous: 2.0,
                        total: 6.0,
                    },
                    lift: DirectionalCoefficient {
                        pressure: 1.0,
                        viscous: -0.5,
                        total: 0.5,
                    },
                    selected_patches: 1,
                    selected_faces: 16,
                    selected_area: 0.25,
                    area_vector_sum: ferrum_mesh::Point3 {
                        x: 0.0,
                        y: 0.0,
                        z: 0.0,
                    },
                    resolved_pressure_reference: 1.25,
                    resolved_reference_area: 1.0e-6,
                    reference_dynamic_pressure: 0.1125,
                },
                faces: Vec::new(),
            }),
        }
    }

    fn wall_face_csv_contribution(
        face_index: usize,
        owner_cell: usize,
        area_vector: ferrum_mesh::Point3,
        pressure_force_x: f64,
        viscous_force_x: f64,
    ) -> WallFaceForceContribution {
        WallFaceForceContribution {
            face_index,
            owner_cell,
            area_vector,
            boundary_pressure: 0.123_456_789_012_345_66 + face_index as f64,
            resolved_dynamic_pressure: pressure_force_x,
            pressure_traction_on_body: ferrum_mesh::Point3 {
                x: pressure_force_x,
                y: 0.0,
                z: 0.0,
            },
            viscous_traction_on_body: ferrum_mesh::Point3 {
                x: viscous_force_x,
                y: 0.25 * face_index as f64,
                z: 0.0,
            },
            wall_shear_traction_on_body: ferrum_mesh::Point3 {
                x: 0.0,
                y: 0.25 * face_index as f64,
                z: 0.0,
            },
            pressure_force_on_body: ferrum_mesh::Point3 {
                x: pressure_force_x,
                y: 0.0,
                z: 0.0,
            },
            viscous_force_on_body: ferrum_mesh::Point3 {
                x: viscous_force_x,
                y: 0.0,
                z: 0.0,
            },
            total_force_on_body: ferrum_mesh::Point3 {
                x: pressure_force_x + viscous_force_x,
                y: 0.0,
                z: 0.0,
            },
        }
    }

    fn wall_face_csv_fixture() -> (SolverRuntimeMeshData, WallForceReport) {
        let area_vectors = vec![
            ferrum_mesh::Point3 {
                x: 1.0,
                y: 0.0,
                z: 0.0,
            },
            ferrum_mesh::Point3 {
                x: 0.0,
                y: 1.0,
                z: 0.0,
            },
            ferrum_mesh::Point3 {
                x: -1.0,
                y: 0.0,
                z: 0.0,
            },
            ferrum_mesh::Point3 {
                x: 0.0,
                y: -1.0,
                z: 0.0,
            },
        ];
        let mesh = SolverRuntimeMeshData {
            points: 8,
            cells: 4,
            faces: 4,
            internal_faces: 0,
            boundary_faces: 4,
            owner: vec![0, 1, 2, 3],
            neighbour: vec![None; 4],
            patches: vec![
                SolverRuntimePatchRange {
                    name: "first".to_string(),
                    patch_type: "wall".to_string(),
                    start_face: 0,
                    faces: 2,
                },
                SolverRuntimePatchRange {
                    name: "=SUM(1,2)\"wall".to_string(),
                    patch_type: "wall".to_string(),
                    start_face: 2,
                    faces: 2,
                },
            ],
            face_centres: (0..4)
                .map(|index| ferrum_mesh::Point3 {
                    x: index as f64,
                    y: index as f64 + 0.5,
                    z: 0.0,
                })
                .collect(),
            face_area_vectors: area_vectors.clone(),
            cell_centres: vec![
                ferrum_mesh::Point3 {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                };
                4
            ],
            cell_volumes: vec![1.0; 4],
            min_face_area: 1.0,
            max_face_area: 1.0,
            min_cell_volume: 1.0,
            max_cell_volume: 1.0,
            total_cell_volume: 4.0,
            non_positive_cell_volumes: 0,
        };
        let faces = vec![
            wall_face_csv_contribution(2, 2, area_vectors[2], 3.0, 0.5),
            wall_face_csv_contribution(3, 3, area_vectors[3], 4.0, 0.25),
            wall_face_csv_contribution(0, 0, area_vectors[0], 1.0, -0.5),
            wall_face_csv_contribution(1, 1, area_vectors[1], 2.0, -0.25),
        ];
        let report = WallForceReport {
            patch_names: vec!["=SUM(1,2)\"wall".to_string(), "first".to_string()],
            reference_speed: 0.015,
            velocity_gradient_scheme: LaminarSimpleGradientScheme::CellLimitedGaussLinear(1.0),
            coefficients: WallForceCoefficients {
                pressure_force: ferrum_mesh::Point3 {
                    x: 10.0,
                    y: 0.0,
                    z: 0.0,
                },
                viscous_force: ferrum_mesh::Point3 {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                },
                total_force: ferrum_mesh::Point3 {
                    x: 10.0,
                    y: 0.0,
                    z: 0.0,
                },
                drag: DirectionalCoefficient {
                    pressure: 1.0,
                    viscous: 0.0,
                    total: 1.0,
                },
                lift: DirectionalCoefficient {
                    pressure: 0.0,
                    viscous: 0.0,
                    total: 0.0,
                },
                selected_patches: 2,
                selected_faces: 4,
                selected_area: 4.0,
                area_vector_sum: ferrum_mesh::Point3 {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                },
                resolved_pressure_reference: 1.25,
                resolved_reference_area: 1.0e-6,
                reference_dynamic_pressure: 0.1125,
            },
            faces,
        };
        (mesh, report)
    }

    #[test]
    fn wall_face_loads_csv_is_round_trip_deterministic_and_confined() {
        let base = output_test_dir("wall-face-loads-csv");
        std::fs::create_dir_all(&base).expect("output root should be created");
        let output_root = SafeOutputRoot::open_existing(&base).expect("output root should open");
        let options = minimal_laminar_simple_options_for_estimate();
        let (mesh, report) = wall_face_csv_fixture();

        write_laminar_simple_wall_face_loads_csv(
            &mesh,
            &options,
            &report,
            &output_root,
            Path::new("nested/loads.csv"),
        )
        .expect("wall-face CSV should be written");
        write_laminar_simple_wall_face_loads_csv(
            &mesh,
            &options,
            &report,
            &output_root,
            Path::new("repeat.csv"),
        )
        .expect("repeated wall-face CSV should be written");

        let bytes = std::fs::read(base.join("nested/loads.csv")).expect("read wall-face CSV");
        assert_eq!(
            bytes,
            std::fs::read(base.join("repeat.csv")).expect("read repeated wall-face CSV")
        );
        let csv = String::from_utf8(bytes.clone()).expect("wall-face CSV is UTF-8");
        let lines = csv.lines().collect::<Vec<_>>();
        assert_eq!(lines.len(), 5);
        assert!(lines[0].starts_with("wallFaceLoadsSchemaVersion,tractionMethod,"));
        assert!(lines[0].ends_with("totalForceOnBodyZN"));
        assert!(lines[1].contains("\"'=SUM(1,2)\"\"wall\",2,2,"));
        assert!(lines[2].contains("\"'=SUM(1,2)\"\"wall\",3,3,"));
        assert!(lines[3].contains("\"first\",0,0,"));
        assert!(lines[4].contains("\"first\",1,1,"));
        assert!(csv.contains("0.12345678901234566"));

        let outside = base
            .parent()
            .expect("test root parent")
            .join("escaped-wall-face-loads.csv");
        let _ = std::fs::remove_file(&outside);
        let error = write_laminar_simple_wall_face_loads_csv(
            &mesh,
            &options,
            &report,
            &output_root,
            &outside,
        )
        .expect_err("wall-face CSV must remain below its safe output root");
        assert_eq!(error.kind(), ErrorKind::PermissionDenied);
        assert!(!outside.exists());

        let mut invalid = report;
        invalid.faces[0].face_index = 3;
        let error = write_laminar_simple_wall_face_loads_csv(
            &mesh,
            &options,
            &invalid,
            &output_root,
            Path::new("nested/loads.csv"),
        )
        .expect_err("invalid row order must fail before replacing the output");
        assert_eq!(error.kind(), ErrorKind::InvalidData);
        assert_eq!(
            std::fs::read(base.join("nested/loads.csv")).expect("read preserved wall-face CSV"),
            bytes
        );

        let _ = std::fs::remove_dir_all(base);
    }

    #[test]
    fn markdown_report_escapes_untrusted_wall_force_patch_names() {
        let base = output_test_dir("wall-force-markdown-escaping");
        std::fs::create_dir_all(&base).expect("output root should be created");
        let output_root = SafeOutputRoot::open_existing(&base).expect("output root should open");
        let mut plan = laminar_simple_test_plan(1000.0, 0.001);
        plan.case_dir = PathBuf::from("case`![case](https://attacker.example/case.png)");
        let mut post_processing = output_test_post_processing_with_forces();
        post_processing
            .wall_forces
            .as_mut()
            .expect("wall-force fixture should be present")
            .patch_names = vec![
            "![image](https://attacker.example/pixel.png)".to_string(),
            "[label](https://attacker.example/)".to_string(),
            "https://attacker.example/@user#1".to_string(),
        ];

        write_laminar_simple_report_markdown(
            &plan,
            &minimal_laminar_simple_options_for_estimate(),
            &output_test_report(),
            &post_processing,
            LaminarSimpleReportRunMetadata {
                profile_pcg: false,
                wall_clock_seconds: 0.0,
            },
            &output_root,
            Path::new("report.md"),
        )
        .expect("Markdown report should be written");

        let markdown = std::fs::read_to_string(base.join("report.md"))
            .expect("Markdown report should be readable");
        let case_line = markdown
            .lines()
            .find(|line| line.starts_with("Case: "))
            .expect("case line");
        assert!(case_line.contains("case&#96;&#33;&#91;case&#93;&#40;https&#58;&#47;&#47;"));
        assert!(!case_line.contains('`'));
        assert!(!case_line.contains("https://"));

        let patch_rows = markdown
            .lines()
            .filter(|line| line.starts_with("| Patches |"))
            .collect::<Vec<_>>();
        assert_eq!(patch_rows.len(), 1);
        let patch_row = patch_rows[0];
        assert_eq!(
            patch_row
                .chars()
                .filter(|character| *character == '|')
                .count(),
            3
        );
        assert!(patch_row.contains("&#33;&#91;image&#93;&#40;https&#58;&#47;&#47;"));
        assert!(!patch_row.contains("!["));
        assert!(!patch_row.contains("]("));
        assert!(!patch_row.contains("https://"));

        assert_eq!(
            escape_markdown_literal("cylinder,innerWall-1"),
            "cylinder,innerWall-1"
        );
        let controls = escape_markdown_literal("<script>|`\\*_~[]()!https://x@y#1\r\n\t\u{1b}\0");
        for entity in [
            "&#60;", "&#62;", "&#124;", "&#96;", "&#92;", "&#42;", "&#95;", "&#126;", "&#91;",
            "&#93;", "&#40;", "&#41;", "&#33;", "&#58;", "&#47;", "&#64;", "&#35;", "&#13;",
            "&#10;", "&#9;", "&#27;", "&#0;",
        ] {
            assert!(
                controls.contains(entity),
                "missing entity {entity} in {controls}"
            );
        }

        write_laminar_simple_report_json(
            &plan,
            &minimal_laminar_simple_options_for_estimate(),
            &output_test_report(),
            &post_processing,
            LaminarSimpleReportRunMetadata {
                profile_pcg: false,
                wall_clock_seconds: 0.0,
            },
            &output_root,
            Path::new("report.json"),
        )
        .expect("JSON report should be written");
        let json = std::fs::read_to_string(base.join("report.json"))
            .expect("JSON report should be readable");
        assert!(json.contains("![image](https://attacker.example/pixel.png)"));
        assert!(json.contains("[label](https://attacker.example/)"));
        assert!(json.contains("https://attacker.example/@user#1"));

        let _ = std::fs::remove_dir_all(base);
    }

    #[test]
    fn continuity_bridge_uses_final_and_completed_iteration_foundation_semantics() {
        let mut plan = laminar_simple_test_plan(1000.0, 0.001);
        plan.run.delta_t = Some(0.5);
        plan.runtime_data.mesh.total_cell_volume = 2.0;
        let solve = parse_incompressible_fluid_args(&[])
            .expect("default incompressible args")
            .laminar_simple_solve
            .expect("incompressible solve");
        let options = minimal_laminar_simple_options_for_estimate();
        let mut report = output_test_report();
        report.simple_iterations = 2;
        report.final_continuity = ContinuitySummary {
            l2_norm: 1.0,
            max_abs: 1.0,
            sum_abs: 4.0,
            global_sum: -1.0,
        };
        let mut first = build_laminar_simple_iteration_summary(1, 0.0, 0.0, 0.0);
        first.continuity_after = ContinuitySummary {
            l2_norm: 1.0,
            max_abs: 1.0,
            sum_abs: 3.0,
            global_sum: 2.0,
        };
        let mut second = build_laminar_simple_iteration_summary(2, 0.0, 0.0, 0.0);
        second.continuity_after = report.final_continuity;
        report.history = vec![first, second];

        let post = build_laminar_simple_post_processing(&plan, &solve, &options, &report)
            .expect("continuity bridge should normalize valid summaries");

        let continuity = post
            .continuity_errors
            .expect("deltaT should enable normalized continuity");
        assert_eq!(continuity.final_error.local, 1.0);
        assert_eq!(continuity.final_error.global, -0.25);
        assert_eq!(continuity.cumulative_global, 0.25);
        assert_eq!(continuity.delta_t, 0.5);
        assert_eq!(continuity.total_volume, 2.0);
    }

    #[test]
    fn missing_delta_t_keeps_normalized_continuity_optional() {
        let mut plan = laminar_simple_test_plan(1000.0, 0.001);
        plan.run.delta_t = None;
        let solve = parse_incompressible_fluid_args(&[])
            .expect("default incompressible args")
            .laminar_simple_solve
            .expect("incompressible solve");
        let options = minimal_laminar_simple_options_for_estimate();
        let report = output_test_report();

        let post = build_laminar_simple_post_processing(&plan, &solve, &options, &report)
            .expect("missing deltaT must preserve the existing solve contract");

        assert!(post.continuity_errors.is_none());
        assert!(post.wall_forces.is_none());
    }

    #[test]
    fn no_wall_force_request_does_not_reconstruct_invalid_final_fields() {
        let mut plan = laminar_simple_test_plan(1000.0, 0.001);
        plan.run.delta_t = None;
        let solve = parse_incompressible_fluid_args(&[])
            .expect("default incompressible args")
            .laminar_simple_solve
            .expect("incompressible solve");
        assert!(solve.wall_forces.is_none());
        let options = minimal_laminar_simple_options_for_estimate();
        let mut report = output_test_report();
        report.final_velocity.clear();
        report.final_pressure.clear();

        let post = build_laminar_simple_post_processing(&plan, &solve, &options, &report)
            .expect("no wall-force request must not reconstruct invalid final fields");

        assert!(post.wall_forces.is_none());
    }

    #[test]
    fn report_schema_three_serializes_continuity_and_optional_wall_forces() {
        let base = output_test_dir("post-processing-report");
        std::fs::create_dir_all(&base).expect("output root should be created");
        let output_root = SafeOutputRoot::open_existing(&base).expect("output root should open");
        let plan = laminar_simple_test_plan(1000.0, 0.001);
        let options = minimal_laminar_simple_options_for_estimate();
        let report = output_test_report();
        let post = output_test_post_processing_with_forces();

        write_laminar_simple_report_json(
            &plan,
            &options,
            &report,
            &post,
            LaminarSimpleReportRunMetadata {
                profile_pcg: false,
                wall_clock_seconds: 0.0,
            },
            &output_root,
            Path::new("report.json"),
        )
        .expect("JSON report should be written");
        write_laminar_simple_report_markdown(
            &plan,
            &options,
            &report,
            &post,
            LaminarSimpleReportRunMetadata {
                profile_pcg: false,
                wall_clock_seconds: 0.0,
            },
            &output_root,
            Path::new("report.md"),
        )
        .expect("Markdown report should be written");

        let json = std::fs::read_to_string(base.join("report.json"))
            .expect("JSON report should be readable");
        let markdown = std::fs::read_to_string(base.join("report.md"))
            .expect("Markdown report should be readable");
        assert!(json.contains("\"schemaVersion\": 3"));
        assert!(json.contains("\"continuityErrors\""));
        assert!(json.contains("\"cumulativeGlobal\": -0.00000000375"));
        assert!(json.contains("\"wallForces\""));
        assert!(json.contains("\"pressureFieldKind\": \"kinematic\""));
        assert!(json.contains("\"tractionMethod\": \"reconstructedGradientFullDeviatoric\""));
        assert!(json.contains("\"tractionMethodVersion\": 1"));
        assert!(json.contains("\"forceConvention\": \"fluidOnBody\""));
        assert!(json.contains("\"faceAreaVectorOrientation\": \"outwardFromFluid\""));
        assert!(json.contains("\"pressureFaceTreatment\": \"zeroGradientOwner\""));
        assert!(json.contains("\"velocityGradientScheme\": \"cellLimited Gauss linear 1\""));
        assert!(json.contains("\"selectedFaces\": 16"));
        assert!(json.contains("\"total\": 6"));
        assert!(markdown.contains("- Schema version: `3`"));
        assert!(markdown.contains("## Continuity errors"));
        assert!(markdown.contains("## Wall forces"));
        assert!(markdown.contains("| Traction method | reconstructedGradientFullDeviatoric |"));
        assert!(markdown.contains("| Traction method version | 1 |"));
        assert!(markdown.contains("| Force convention | fluidOnBody |"));
        assert!(markdown.contains("| Face area-vector orientation | outwardFromFluid |"));
        assert!(markdown.contains("| Pressure face treatment | zeroGradientOwner |"));
        assert!(markdown.contains("| Velocity gradient scheme | cellLimited Gauss linear 1 |"));
        assert!(
            markdown
                .contains("| Drag pressure/viscous/total | 4.000000e0 / 2.000000e0 / 6.000000e0 |")
        );

        let _ = std::fs::remove_dir_all(base);
    }

    fn hierarchy_profile_fixture() -> GamgKernelTiming {
        GamgKernelTiming {
            hierarchy_diagnostic_seconds: 0.125,
            finest_residual_evaluations: 5,
            outer_matrix_vector_products: 17,
            outer_reductions: 23,
            levels: vec![
                GamgLevelTiming {
                    level: 0,
                    cells: 8,
                    nonzeros: 24,
                    smoothing_sweeps: 3,
                    scaling_calls: 1,
                    residual_evaluations: 2,
                    ..GamgLevelTiming::default()
                },
                GamgLevelTiming {
                    level: 1,
                    cells: 4,
                    nonzeros: 10,
                    smoothing_sweeps: 4,
                    scaling_calls: 2,
                    residual_evaluations: 3,
                    ..GamgLevelTiming::default()
                },
                GamgLevelTiming {
                    level: 2,
                    cells: 2,
                    nonzeros: 4,
                    residual_evaluations: 1,
                    coarsest_solves: 1,
                    coarsest_iterations: 7,
                    ..GamgLevelTiming::default()
                },
            ],
            hierarchy: Some(std::sync::Arc::new(GamgHierarchyDiagnostics {
                levels: vec![
                    GamgHierarchyLevelDiagnostics {
                        level: 0,
                        cells: 8,
                        nonzeros: 24,
                    },
                    GamgHierarchyLevelDiagnostics {
                        level: 1,
                        cells: 4,
                        nonzeros: 10,
                    },
                    GamgHierarchyLevelDiagnostics {
                        level: 2,
                        cells: 2,
                        nonzeros: 4,
                    },
                ],
                transfers: vec![
                    GamgTransferDiagnostics {
                        fine_level: 0,
                        coarse_level: 1,
                        fine_cells: 8,
                        coarse_cells: 4,
                        singleton_fine_cells: 0,
                        unmatched_fine_cells: 0,
                        min_aggregate_size: 2,
                        max_aggregate_size: 2,
                        aggregate_size_histogram: vec![GamgAggregateSizeBin {
                            aggregate_size: 2,
                            aggregate_count: 4,
                        }],
                    },
                    GamgTransferDiagnostics {
                        fine_level: 1,
                        coarse_level: 2,
                        fine_cells: 4,
                        coarse_cells: 2,
                        singleton_fine_cells: 1,
                        unmatched_fine_cells: 1,
                        min_aggregate_size: 1,
                        max_aggregate_size: 3,
                        aggregate_size_histogram: vec![
                            GamgAggregateSizeBin {
                                aggregate_size: 1,
                                aggregate_count: 1,
                            },
                            GamgAggregateSizeBin {
                                aggregate_size: 3,
                                aggregate_count: 1,
                            },
                        ],
                    },
                ],
                smoother_passes_per_sweep: 2,
                direct_solve_coarsest: true,
            })),
            ..GamgKernelTiming::default()
        }
    }

    fn output_test_dir(label: &str) -> PathBuf {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock should be after Unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "ferrum-cli-output-{label}-{}-{unique}",
            std::process::id()
        ))
    }

    #[cfg(unix)]
    fn create_output_file_symlink(target: &Path, link: &Path) -> std::io::Result<()> {
        std::os::unix::fs::symlink(target, link)
    }

    #[cfg(windows)]
    fn create_output_file_symlink(target: &Path, link: &Path) -> std::io::Result<()> {
        std::os::windows::fs::symlink_file(target, link)
    }

    #[cfg(windows)]
    fn create_output_junction(target: &Path, link: &Path) -> std::io::Result<()> {
        let status = std::process::Command::new("cmd")
            .arg("/C")
            .arg("mklink")
            .arg("/J")
            .arg(link)
            .arg(target)
            .status()?;
        if status.success() {
            Ok(())
        } else {
            Err(std::io::Error::other("mklink /J failed"))
        }
    }

    #[test]
    fn parses_laminar_simple_split_linear_solvers() {
        let args = vec![
            "--linearSolver".to_string(),
            "bicgstab".to_string(),
            "--momentumLinearSolver".to_string(),
            "cg".to_string(),
            "--pressureLinearSolver".to_string(),
            "pcg".to_string(),
            "--pressurePreconditioner".to_string(),
            "DIC".to_string(),
        ];

        let parsed = parse_incompressible_fluid_args(&args).expect("solver args should parse");
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
            Some(LaminarSimplePreconditioner::Dic)
        );
    }

    #[test]
    fn parses_laminar_simple_momentum_execution_aliases_and_rejects_invalid_values() {
        for flag in ["--momentumExecution", "--momentum-execution"] {
            let parsed = parse_incompressible_fluid_args(&[
                flag.to_string(),
                "parallel-components".to_string(),
            ])
            .expect("parallel momentum execution should parse");
            let solve = parsed
                .laminar_simple_solve
                .expect("laminar SIMPLE solve args");
            assert_eq!(
                solve.momentum_execution,
                Some(LaminarSimpleMomentumExecution::ParallelComponents)
            );
        }

        let parsed = parse_incompressible_fluid_args(&[
            "--momentum-execution".to_string(),
            "serial".to_string(),
        ])
        .expect("serial momentum execution should parse");
        assert_eq!(
            parsed
                .laminar_simple_solve
                .expect("laminar SIMPLE solve args")
                .momentum_execution,
            Some(LaminarSimpleMomentumExecution::Serial)
        );

        let missing = parse_incompressible_fluid_args(&["--momentumExecution".to_string()])
            .expect_err("missing momentum execution value must fail");
        assert_eq!(
            missing,
            "--momentumExecution requires 'serial' or 'parallel-components'"
        );

        let invalid = parse_incompressible_fluid_args(&[
            "--momentumExecution".to_string(),
            "parallel".to_string(),
        ])
        .expect_err("invalid momentum execution value must fail");
        assert_eq!(
            invalid,
            "invalid --momentumExecution value 'parallel'; expected 'serial' or 'parallel-components'"
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

        let parsed = parse_incompressible_fluid_args(&args).expect("solver args should parse");
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
        let args = Vec::new();

        let parsed = parse_incompressible_fluid_args(&args).expect("solver args should parse");
        let solve = parsed
            .laminar_simple_solve
            .expect("laminar SIMPLE solve args");

        assert_eq!(solve.velocity_relaxation, None);
        assert_eq!(solve.pressure_relaxation, None);
        assert_eq!(solve.simple_consistent, None);
    }

    #[test]
    fn laminar_simple_resolves_from_standard_fluid_inputs() {
        let plan = laminar_simple_test_plan(1000.0, 0.001002);
        let args = vec![
            "--rho".to_string(),
            "1000".to_string(),
            "--mu".to_string(),
            "0.001002".to_string(),
        ];
        let parsed = parse_incompressible_fluid_args(&args).expect("solver args should parse");
        let solve = parsed
            .laminar_simple_solve
            .expect("laminar SIMPLE solve args");
        let options = resolve_laminar_simple_options(&plan, &solve)
            .expect("laminar options should resolve from incompressible inputs");

        assert_eq!(options.density, 1000.0);
        assert_eq!(options.dynamic_viscosity, 0.001002);
        assert_eq!(
            options.momentum_linear_solver,
            LaminarSimpleLinearSolver::SymGaussSeidel
        );
        assert_eq!(
            options.momentum_execution,
            LaminarSimpleMomentumExecution::Serial
        );
        assert_eq!(
            options.pressure_linear_solver,
            LaminarSimpleLinearSolver::Pcg
        );
        assert_eq!(options.max_simple_iterations, 100);
    }

    #[test]
    fn laminar_simple_resolves_explicit_parallel_momentum_execution() {
        let plan = laminar_simple_test_plan(1000.0, 0.001002);
        let parsed = parse_incompressible_fluid_args(&[
            "--momentumExecution".to_string(),
            "parallel-components".to_string(),
        ])
        .expect("solver args should parse");
        let solve = parsed
            .laminar_simple_solve
            .expect("laminar SIMPLE solve args");
        let options = resolve_laminar_simple_options(&plan, &solve)
            .expect("parallel momentum execution should resolve");

        assert_eq!(
            options.momentum_execution,
            LaminarSimpleMomentumExecution::ParallelComponents
        );
    }

    #[test]
    fn laminar_simple_uses_ferrum_ldu_defaults_when_controls_are_absent() {
        let mut plan = laminar_simple_test_plan(1000.0, 0.001002);
        plan.numerics
            .fv_solution
            .entries
            .retain(|entry| !matches!(entry.key.as_str(), "tolerance" | "maxIter"));
        let parsed = parse_incompressible_fluid_args(&[]).expect("solver args should parse");
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
    fn laminar_simple_rejects_untrusted_case_max_iter_above_safety_cap() {
        let mut plan = laminar_simple_test_plan(1000.0, 0.001002);
        plan.numerics
            .fv_solution
            .entries
            .push(SolverNumericsEntryPlan {
                section: "solvers.U".to_string(),
                key: "maxIter".to_string(),
                value: "100000000".to_string(),
            });
        let parsed = parse_incompressible_fluid_args(&[]).expect("solver args should parse");
        let solve = parsed
            .laminar_simple_solve
            .expect("laminar SIMPLE solve args");

        let error = resolve_laminar_simple_options(&plan, &solve)
            .expect_err("oversized case maxIter must fail closed");

        assert!(error.contains("solvers.U.maxIter=100000000 exceeds"));
        assert!(error.contains("safety cap of 1000"));
    }

    #[test]
    fn laminar_simple_allows_cli_iteration_override_when_case_max_iter_exceeds_cap() {
        let mut plan = laminar_simple_test_plan(1000.0, 0.001002);
        plan.numerics
            .fv_solution
            .entries
            .push(SolverNumericsEntryPlan {
                section: "solvers.U".to_string(),
                key: "maxIter".to_string(),
                value: "100000000".to_string(),
            });
        let parsed =
            parse_incompressible_fluid_args(&["--maxIterations".to_string(), "2500".to_string()])
                .expect("solver args should parse");
        let solve = parsed
            .laminar_simple_solve
            .expect("laminar SIMPLE solve args");
        let options =
            resolve_laminar_simple_options(&plan, &solve).expect("CLI cap should be trusted");

        assert_eq!(options.momentum_max_linear_iterations, 2_500);
        assert_eq!(options.pressure_max_linear_iterations, 2_500);
    }

    #[test]
    fn laminar_simple_control_dict_iteration_cap_is_exact() {
        let parsed = parse_incompressible_fluid_args(&[]).expect("solver args should parse");
        let solve = parsed
            .laminar_simple_solve
            .expect("laminar SIMPLE solve args");

        let mut accepted = laminar_simple_test_plan(1000.0, 0.001002);
        accepted.run.estimated_steps = Some(FERRUM_MAX_CASE_SIMPLE_ITERATIONS);
        let options = resolve_laminar_simple_options(&accepted, &solve)
            .expect("exact controlDict SIMPLE cap should resolve");
        assert_eq!(
            options.max_simple_iterations,
            FERRUM_MAX_CASE_SIMPLE_ITERATIONS
        );

        let mut rejected = laminar_simple_test_plan(1000.0, 0.001002);
        rejected.run.estimated_steps = Some(FERRUM_MAX_CASE_SIMPLE_ITERATIONS + 1);
        let error = resolve_laminar_simple_options(&rejected, &solve)
            .expect_err("controlDict SIMPLE iterations above the cap must fail");
        assert!(error.contains("case-file safety cap of 10000"));
    }

    #[test]
    fn laminar_simple_accepts_two_thousand_control_dict_iterations() {
        let parsed = parse_incompressible_fluid_args(&[]).expect("solver args should parse");
        let solve = parsed
            .laminar_simple_solve
            .expect("laminar SIMPLE solve args");
        let mut plan = laminar_simple_test_plan(1000.0, 0.001002);
        plan.control.start_time = Some(0.0);
        plan.control.end_time = Some(2_000.0);
        plan.control.delta_t = Some(1.0);
        plan.run.start_time = Some(0.0);
        plan.run.end_time = Some(2_000.0);
        plan.run.delta_t = Some(1.0);
        plan.run.estimated_steps = Some(2_000);

        let options = resolve_laminar_simple_options(&plan, &solve)
            .expect("2,000 controlDict-derived SIMPLE iterations should resolve");

        assert_eq!(options.max_simple_iterations, 2_000);
    }

    #[test]
    fn laminar_simple_cli_override_bypasses_case_iteration_cap() {
        let trusted_iterations = FERRUM_MAX_CASE_SIMPLE_ITERATIONS + 1;
        let parsed = parse_incompressible_fluid_args(&[
            "--maxSimpleIterations".to_string(),
            trusted_iterations.to_string(),
        ])
        .expect("trusted SIMPLE override should parse");
        let solve = parsed
            .laminar_simple_solve
            .expect("laminar SIMPLE solve args");
        let mut plan = laminar_simple_test_plan(1000.0, 0.001002);
        plan.run.estimated_steps = Some(trusted_iterations);

        let options = resolve_laminar_simple_options(&plan, &solve)
            .expect("trusted CLI SIMPLE override should bypass the case cap");

        assert_eq!(options.max_simple_iterations, trusted_iterations);
    }

    #[test]
    fn laminar_simple_resolves_relative_tolerances_independently_of_absolute_cli_overrides() {
        let mut plan = laminar_simple_test_plan(1000.0, 0.001002);
        plan.numerics.fv_solution.entries.extend([
            ferrum_mesh::solver_plan::SolverNumericsEntryPlan {
                section: "solvers.U".to_string(),
                key: "relTol".to_string(),
                value: "0.1".to_string(),
            },
            ferrum_mesh::solver_plan::SolverNumericsEntryPlan {
                section: "solvers.p".to_string(),
                key: "relTol".to_string(),
                value: "1.25".to_string(),
            },
        ]);
        let parsed = parse_incompressible_fluid_args(&[
            "--momentumSolveTolerance".to_string(),
            "1e-7".to_string(),
            "--pressureSolveTolerance".to_string(),
            "1e-9".to_string(),
        ])
        .expect("solver args should parse");
        let solve = parsed
            .laminar_simple_solve
            .expect("laminar SIMPLE solve args");
        let options = resolve_laminar_simple_options(&plan, &solve)
            .expect("finite non-negative relTol must resolve");

        assert_eq!(options.momentum_linear_tolerance, 1.0e-7);
        assert_eq!(options.pressure_linear_tolerance, 1.0e-9);
        assert_eq!(options.momentum_linear_relative_tolerance, 0.1);
        assert_eq!(options.pressure_linear_relative_tolerance, 1.25);
    }

    #[test]
    fn laminar_simple_relative_tolerance_defaults_to_zero_and_rejects_invalid_values() {
        let parsed = parse_incompressible_fluid_args(&[]).expect("solver args should parse");
        let solve = parsed
            .laminar_simple_solve
            .expect("laminar SIMPLE solve args");
        let defaults =
            resolve_laminar_simple_options(&laminar_simple_test_plan(1000.0, 0.001002), &solve)
                .expect("missing relTol must preserve the zero default");
        assert_eq!(defaults.momentum_linear_relative_tolerance, 0.0);
        assert_eq!(defaults.pressure_linear_relative_tolerance, 0.0);

        for value in ["-1", "NaN", "inf", "-inf"] {
            let mut plan = laminar_simple_test_plan(1000.0, 0.001002);
            plan.numerics.fv_solution.entries.push(
                ferrum_mesh::solver_plan::SolverNumericsEntryPlan {
                    section: "solvers.U".to_string(),
                    key: "relTol".to_string(),
                    value: value.to_string(),
                },
            );
            let error = resolve_laminar_simple_options(&plan, &solve)
                .expect_err("invalid relTol must fail closed");
            assert!(
                error.contains("fvSolution solvers.U.relTol must be finite and non-negative"),
                "unexpected error for {value}: {error}"
            );
        }
    }

    #[test]
    fn laminar_simple_ignores_unsupported_case_level_keys() {
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
                key: "unsupportedCaseOption".to_string(),
                value: "true".to_string(),
            },
            ferrum_mesh::solver_plan::SolverNumericsEntryPlan {
                section: "SIMPLE".to_string(),
                key: "minSimpleIterations".to_string(),
                value: "3".to_string(),
            },
        ]);

        let args = Vec::new();
        let parsed = parse_incompressible_fluid_args(&args).expect("solver args should parse");
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
            momentum_execution: LaminarSimpleMomentumExecution::Serial,
            pressure_linear_solver: LaminarSimpleLinearSolver::Cg,
            momentum_preconditioner: LaminarSimplePreconditioner::None,
            pressure_preconditioner: LaminarSimplePreconditioner::None,
            pressure_gamg_options: None,
            profile_gamg: false,
            linear_tolerance: 1.0e-10,
            max_linear_iterations: 10_000,
            momentum_linear_tolerance: 1.0e-10,
            pressure_linear_tolerance: 1.0e-10,
            momentum_linear_relative_tolerance: 0.0,
            pressure_linear_relative_tolerance: 0.0,
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
            momentum_component_linear_solves: Default::default(),
            pressure_correction_linear_solves: Default::default(),
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
            parse_openfoam_laminar_solver("GAMG").unwrap(),
            LaminarSimpleLinearSolver::Gamg
        );
        assert_eq!(
            parse_openfoam_laminar_preconditioner(
                numerics_dictionary_value(&dictionary, "solvers.p", "preconditioner").unwrap()
            )
            .unwrap(),
            LaminarSimplePreconditioner::Dic
        );
        assert_eq!(
            parse_openfoam_laminar_preconditioner("FDIC").unwrap(),
            LaminarSimplePreconditioner::Fdic
        );
        assert_eq!(
            parse_openfoam_laminar_preconditioner("ic0").unwrap(),
            LaminarSimplePreconditioner::IncompleteCholesky
        );
        assert!(parse_openfoam_laminar_preconditioner("DILU").is_err());
    }

    #[test]
    fn maps_openfoam_gamg_dictionary_controls_without_substitution() {
        let mut plan = laminar_simple_test_plan(998.2, 1.002e-3);
        plan.numerics
            .fv_solution
            .entries
            .retain(|entry| entry.section != "solvers.p");
        for (key, value) in [
            ("solver", "GAMG"),
            ("smoother", "symGaussSeidel"),
            ("agglomerator", "algebraicPair"),
            ("outerSolver", "FCG"),
            ("maxIter", "73"),
            ("minIter", "2"),
            ("tolerance", "2e-9"),
            ("relTol", "0.075"),
            ("cacheAgglomeration", "false"),
            ("nCellsInCoarsestLevel", "12"),
            ("mergeLevels", "1"),
            ("nPreSweeps", "1"),
            ("preSweepsLevelMultiplier", "2"),
            ("maxPreSweeps", "5"),
            ("nPostSweeps", "3"),
            ("postSweepsLevelMultiplier", "2"),
            ("maxPostSweeps", "6"),
            ("nFinestSweeps", "4"),
            ("interpolateCorrection", "true"),
            ("scaleCorrection", "false"),
            ("directSolveCoarsest", "true"),
        ] {
            plan.numerics
                .fv_solution
                .entries
                .push(SolverNumericsEntryPlan {
                    section: "solvers.p".to_string(),
                    key: key.to_string(),
                    value: value.to_string(),
                });
        }

        let options = openfoam_gamg_options(&plan, "solvers.p").expect("GAMG controls should map");

        assert_eq!(options.agglomerator, GamgAgglomerator::AlgebraicPair);
        assert_eq!(options.smoother, GamgSmoother::SymGaussSeidel);
        assert_eq!(options.outer_solver, GamgOuterSolver::FlexibleCg);
        assert_eq!(options.max_iterations, 73);
        assert_eq!(options.min_iterations, 2);
        assert_eq!(options.tolerance, 2.0e-9);
        assert_eq!(options.relative_tolerance, 0.075);
        assert!(!options.cache_agglomeration);
        assert_eq!(options.n_cells_in_coarsest_level, 12);
        assert_eq!(options.merge_levels, 1);
        assert_eq!(options.n_pre_sweeps, 1);
        assert_eq!(options.pre_sweeps_level_multiplier, 2);
        assert_eq!(options.max_pre_sweeps, 5);
        assert_eq!(options.n_post_sweeps, 3);
        assert_eq!(options.post_sweeps_level_multiplier, 2);
        assert_eq!(options.max_post_sweeps, 6);
        assert_eq!(options.n_finest_sweeps, 4);
        assert!(options.interpolate_correction);
        assert!(!options.scale_correction);
        assert!(options.direct_solve_coarsest);
    }

    #[test]
    fn resolves_gamg_pressure_controls_into_simple_runtime_options() {
        let mut plan = laminar_simple_test_plan(998.2, 1.002e-3);
        plan.numerics
            .fv_solution
            .entries
            .retain(|entry| entry.section != "solvers.p");
        for (key, value) in [
            ("solver", "GAMG"),
            ("smoother", "symGaussSeidel"),
            ("agglomerator", "faceAreaPair"),
            ("outerSolver", "FCG"),
            ("maxIter", "73"),
            ("minIter", "2"),
            ("tolerance", "2e-9"),
            ("relTol", "0.075"),
            ("cacheAgglomeration", "true"),
            ("nCellsInCoarsestLevel", "12"),
            ("mergeLevels", "1"),
            ("nPreSweeps", "1"),
            ("nPostSweeps", "3"),
            ("nFinestSweeps", "4"),
            ("interpolateCorrection", "true"),
        ] {
            plan.numerics
                .fv_solution
                .entries
                .push(SolverNumericsEntryPlan {
                    section: "solvers.p".to_string(),
                    key: key.to_string(),
                    value: value.to_string(),
                });
        }
        let parsed = parse_incompressible_fluid_args(&[
            "--pressureSolveTolerance".to_string(),
            "5e-9".to_string(),
            "--pressureMaxIterations".to_string(),
            "41".to_string(),
            "--profileGamg".to_string(),
        ])
        .expect("solver args should parse");
        let solve = parsed
            .laminar_simple_solve
            .expect("laminar SIMPLE solve args");

        let options = resolve_laminar_simple_options(&plan, &solve)
            .expect("GAMG pressure options should resolve");
        let gamg = options
            .pressure_gamg_options
            .expect("resolved pressure GAMG controls");

        assert_eq!(
            options.pressure_linear_solver,
            LaminarSimpleLinearSolver::Gamg
        );
        assert_eq!(
            options.pressure_preconditioner,
            LaminarSimplePreconditioner::None
        );
        assert_eq!(options.pressure_linear_tolerance, 5.0e-9);
        assert_eq!(options.pressure_linear_relative_tolerance, 0.075);
        assert_eq!(options.pressure_max_linear_iterations, 41);
        assert!(options.profile_gamg);
        assert_eq!(gamg.agglomerator, GamgAgglomerator::FaceAreaPair);
        assert_eq!(gamg.smoother, GamgSmoother::SymGaussSeidel);
        assert_eq!(gamg.outer_solver, GamgOuterSolver::FlexibleCg);
        assert_eq!(gamg.tolerance, 5.0e-9);
        assert_eq!(gamg.max_iterations, 41);
        assert_eq!(gamg.min_iterations, 2);
        assert_eq!(gamg.relative_tolerance, 0.075);
        assert_eq!(gamg.n_cells_in_coarsest_level, 12);
        assert_eq!(gamg.n_pre_sweeps, 1);
        assert_eq!(gamg.n_post_sweeps, 3);
        assert_eq!(gamg.n_finest_sweeps, 4);
        assert!(gamg.interpolate_correction);
    }

    #[test]
    fn rejects_gamg_profile_without_gamg_pressure_solver() {
        let plan = laminar_simple_test_plan(998.2, 1.002e-3);
        let parsed = parse_incompressible_fluid_args(&["--profileGamg".to_string()])
            .expect("solver args should parse");
        let solve = parsed
            .laminar_simple_solve
            .expect("laminar SIMPLE solve args");

        let error = resolve_laminar_simple_options(&plan, &solve)
            .expect_err("profiling PCG as GAMG must fail");

        assert!(error.contains("requires solvers.p.solver GAMG"));
    }

    #[test]
    fn profile_pcg_defaults_off_and_accepts_all_cli_aliases() {
        let parsed = parse_incompressible_fluid_args(&[]).expect("empty solver args should parse");
        assert!(
            !parsed
                .laminar_simple_solve
                .expect("default laminar SIMPLE solve args")
                .profile_pcg
        );

        for alias in [
            "-profilePcg",
            "--profilePcg",
            "-profile-pcg",
            "--profile-pcg",
        ] {
            let parsed = parse_incompressible_fluid_args(&[alias.to_string()])
                .expect("PCG profile alias should parse");
            assert!(
                parsed
                    .laminar_simple_solve
                    .expect("laminar SIMPLE solve args")
                    .profile_pcg
            );
        }
    }

    #[test]
    fn rejects_pcg_profile_without_pcg_pressure_solver() {
        let plan = laminar_simple_test_plan(998.2, 1.002e-3);
        let parsed = parse_incompressible_fluid_args(&[
            "--pressureLinearSolver".to_string(),
            "cg".to_string(),
            "--profilePcg".to_string(),
        ])
        .expect("solver args should parse");
        let solve = parsed
            .laminar_simple_solve
            .expect("laminar SIMPLE solve args");

        let error = resolve_laminar_simple_options(&plan, &solve)
            .expect_err("profiling a non-PCG pressure solver must fail");

        assert!(error.contains("--profilePcg requires solvers.p.solver PCG"));
    }

    #[test]
    fn rejects_mixed_pcg_and_gamg_profile_modes() {
        let plan = laminar_simple_test_plan(998.2, 1.002e-3);
        let parsed = parse_incompressible_fluid_args(&[
            "--pressureLinearSolver".to_string(),
            "pcg".to_string(),
            "--profilePcg".to_string(),
            "--profileGamg".to_string(),
        ])
        .expect("solver args should parse");
        let solve = parsed
            .laminar_simple_solve
            .expect("laminar SIMPLE solve args");

        let error = resolve_laminar_simple_options(&plan, &solve)
            .expect_err("mixed pressure profile modes must fail");

        assert!(error.contains("--profileGamg requires solvers.p.solver GAMG"));
    }

    #[test]
    fn rejects_gamg_for_nonsymmetric_simple_momentum_equation() {
        let mut plan = laminar_simple_test_plan(998.2, 1.002e-3);
        let momentum_solver = plan
            .numerics
            .fv_solution
            .entries
            .iter_mut()
            .find(|entry| entry.section == "solvers.U" && entry.key == "solver")
            .expect("momentum solver entry");
        momentum_solver.value = "GAMG".to_string();
        let parsed = parse_incompressible_fluid_args(&[]).expect("solver args should parse");
        let solve = parsed
            .laminar_simple_solve
            .expect("laminar SIMPLE solve args");

        let error = resolve_laminar_simple_options(&plan, &solve)
            .expect_err("momentum GAMG must be rejected");

        assert!(error.contains("symmetric SIMPLE pressure equation only"));
        assert!(error.contains("nonsymmetric momentum solver"));
    }

    #[test]
    fn rejects_incomplete_or_unknown_gamg_controls_without_fallback() {
        let mut plan = laminar_simple_test_plan(998.2, 1.002e-3);
        plan.numerics
            .fv_solution
            .entries
            .retain(|entry| entry.section != "solvers.p");
        plan.numerics
            .fv_solution
            .entries
            .push(SolverNumericsEntryPlan {
                section: "solvers.p".to_string(),
                key: "solver".to_string(),
                value: "GAMG".to_string(),
            });

        let missing_smoother =
            openfoam_gamg_options(&plan, "solvers.p").expect_err("GAMG without smoother must fail");
        assert!(missing_smoother.contains("requires a smoother"));

        plan.numerics
            .fv_solution
            .entries
            .push(SolverNumericsEntryPlan {
                section: "solvers.p".to_string(),
                key: "smoother".to_string(),
                value: "GaussSeidel".to_string(),
            });
        plan.numerics
            .fv_solution
            .entries
            .push(SolverNumericsEntryPlan {
                section: "solvers.p".to_string(),
                key: "agglomerator".to_string(),
                value: "unknownPair".to_string(),
            });

        let unknown_agglomerator = openfoam_gamg_options(&plan, "solvers.p")
            .expect_err("unknown GAMG agglomerator must fail");
        assert!(unknown_agglomerator.contains("no agglomerator fallback was applied"));
    }

    #[test]
    fn gamg_outer_solver_defaults_to_standalone_and_rejects_unknown_values() {
        let mut plan = laminar_simple_test_plan(998.2, 1.002e-3);
        plan.numerics
            .fv_solution
            .entries
            .retain(|entry| entry.section != "solvers.p");
        for (key, value) in [("solver", "GAMG"), ("smoother", "GaussSeidel")] {
            plan.numerics
                .fv_solution
                .entries
                .push(SolverNumericsEntryPlan {
                    section: "solvers.p".to_string(),
                    key: key.to_string(),
                    value: value.to_string(),
                });
        }

        let defaults = openfoam_gamg_options(&plan, "solvers.p")
            .expect("omitted outerSolver should use the standalone GAMG solve");
        assert_eq!(defaults.outer_solver, GamgOuterSolver::Standalone);

        plan.numerics
            .fv_solution
            .entries
            .push(SolverNumericsEntryPlan {
                section: "solvers.p".to_string(),
                key: "outerSolver".to_string(),
                value: "standalone".to_string(),
            });
        let explicit = openfoam_gamg_options(&plan, "solvers.p")
            .expect("the exact standalone outerSolver value should be accepted");
        assert_eq!(explicit.outer_solver, GamgOuterSolver::Standalone);

        plan.numerics
            .fv_solution
            .entries
            .last_mut()
            .expect("outerSolver entry")
            .value = "fcg".to_string();
        let error = openfoam_gamg_options(&plan, "solvers.p")
            .expect_err("outerSolver values are case-sensitive and must not fall back");
        assert!(error.contains("outerSolver 'fcg'"));
        assert!(error.contains("exactly standalone and FCG"));
    }

    #[test]
    fn pressure_gamg_console_keeps_legacy_prefix_and_appends_outer_solver() {
        let options = GamgOptions {
            outer_solver: GamgOuterSolver::FlexibleCg,
            ..GamgOptions::default()
        };

        let line = format_pressure_gamg_console(&options);

        assert!(line.starts_with("incompressibleFluid pressureGAMG: agglomerator="));
        assert!(line.ends_with(" outerSolver=FCG"));
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
                application: Some("ferrumRun".to_string()),
                solver: Some("incompressibleFluid".to_string()),
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
            initial_fields: ferrum_mesh::fields::InitialFieldSet {
                case_dir: PathBuf::from("case"),
                fields: Vec::new(),
            },
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
                            section: "ddtSchemes".to_string(),
                            key: "default".to_string(),
                            value: "steadyState".to_string(),
                        },
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
                    sections: vec![ferrum_mesh::solver_plan::SolverNumericsSectionPlan {
                        path: "SIMPLE".to_string(),
                        entries: 0,
                    }],
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
    fn current_incompressible_execution_contract_is_unambiguously_steady_simple() {
        let plan = laminar_simple_test_plan(1000.0, 0.001);
        validate_module_execution_contract(&plan, "incompressibleFluid", "SIMPLE")
            .expect("steady SIMPLE plan should pass");
    }

    #[test]
    fn current_incompressible_execution_rejects_competing_pimple_section() {
        let mut plan = laminar_simple_test_plan(1000.0, 0.001);
        plan.numerics.fv_solution.sections.push(
            ferrum_mesh::solver_plan::SolverNumericsSectionPlan {
                path: "PIMPLE".to_string(),
                entries: 0,
            },
        );

        let error = validate_module_execution_contract(&plan, "incompressibleFluid", "SIMPLE")
            .expect_err("competing PIMPLE configuration must fail");

        assert!(error.contains("PISO/PIMPLE execution is not implemented"));
    }

    #[test]
    fn current_incompressible_execution_rejects_transient_ddt_scheme() {
        let mut plan = laminar_simple_test_plan(1000.0, 0.001);
        plan.numerics
            .fv_schemes
            .entries
            .iter_mut()
            .find(|entry| entry.section == "ddtSchemes" && entry.key == "default")
            .expect("test plan ddt scheme")
            .value = "Euler".to_string();

        let error = validate_module_execution_contract(&plan, "incompressibleFluid", "SIMPLE")
            .expect_err("transient ddt scheme must fail");

        assert!(error.contains("ddtSchemes.default=steadyState"));
    }

    #[test]
    fn current_incompressible_execution_accepts_explicit_laminar_regime() {
        let mut plan = laminar_simple_test_plan(1000.0, 0.001);
        plan.properties
            .dictionaries
            .push(SolverPropertyDictionaryPlan {
                name: "momentumTransport".to_string(),
                region: None,
                sections: 0,
                entries: 1,
            });
        plan.properties.entries.push(SolverPropertyEntryPlan {
            dictionary: "momentumTransport".to_string(),
            section: None,
            key: "simulationType".to_string(),
            value: "laminar".to_string(),
        });

        validate_module_execution_contract(&plan, "incompressibleFluid", "SIMPLE")
            .expect("explicit laminar momentum transport should pass");
    }

    #[test]
    fn current_incompressible_execution_rejects_ras_regime() {
        let mut plan = laminar_simple_test_plan(1000.0, 0.001);
        plan.properties
            .dictionaries
            .push(SolverPropertyDictionaryPlan {
                name: "momentumTransport".to_string(),
                region: None,
                sections: 0,
                entries: 1,
            });
        plan.properties.entries.push(SolverPropertyEntryPlan {
            dictionary: "momentumTransport".to_string(),
            section: None,
            key: "simulationType".to_string(),
            value: "RAS".to_string(),
        });

        let error = validate_module_execution_contract(&plan, "incompressibleFluid", "SIMPLE")
            .expect_err("RAS must not run through the laminar kernel");

        assert!(error.contains("requires simulationType laminar"));
        assert!(error.contains("RAS/LES execution is not implemented"));
    }

    #[test]
    fn solver_plan_json_records_effective_dispatch() {
        let plan = laminar_simple_test_plan(1000.0, 0.001);
        let dispatch = resolve_solver_dispatch(Some("incompressibleFluid".to_string()), None)
            .expect("explicit solver selection");
        let base = std::env::temp_dir().join(format!(
            "ferrum-plan-dispatch-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system clock should be after Unix epoch")
                .as_nanos()
        ));
        std::fs::create_dir_all(&base).expect("test root should be created");
        let path = base.join("nested/plan.json");

        write_solver_plan_json_in_root(&plan, Some(&dispatch), &base, &path)
            .expect("solver plan JSON should be written");
        let json = std::fs::read_to_string(&path).expect("solver plan JSON should be readable");
        let _ = std::fs::remove_dir_all(&base);

        assert!(json.contains("\"schemaVersion\": 2"));
        assert!(json.contains("\"dispatch\""));
        assert!(json.contains("\"module\": \"incompressibleFluid\""));
        assert!(json.contains("\"source\": \"cli\""));
    }

    #[test]
    fn solver_plan_json_does_not_clobber_existing_file() {
        let plan = laminar_simple_test_plan(1000.0, 0.001);
        let base = std::env::temp_dir().join(format!(
            "ferrum-plan-existing-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system clock should be after Unix epoch")
                .as_nanos()
        ));
        std::fs::create_dir_all(&base).expect("test root should be created");
        let path = base.join("plan.json");
        std::fs::write(&path, "do-not-clobber").expect("existing file should be created");

        let error = write_solver_plan_json_in_root(&plan, None, &base, &path)
            .expect_err("existing plan path must not be clobbered");
        let contents = std::fs::read_to_string(&path).expect("existing file should be readable");
        let _ = std::fs::remove_dir_all(&base);

        assert_eq!(error.kind(), std::io::ErrorKind::AlreadyExists);
        assert_eq!(contents, "do-not-clobber");
    }

    #[cfg(unix)]
    #[test]
    fn solver_plan_json_rejects_symlink_path() {
        let plan = laminar_simple_test_plan(1000.0, 0.001);
        let base = std::env::temp_dir().join(format!(
            "ferrum-plan-symlink-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system clock should be after Unix epoch")
                .as_nanos()
        ));
        std::fs::create_dir_all(&base).expect("temp directory should be created");
        let target = base.join("target.json");
        let link = base.join("plan.json");
        std::fs::write(&target, "do-not-clobber").expect("target should be created");
        std::os::unix::fs::symlink(&target, &link).expect("symlink should be created");

        let error = write_solver_plan_json_in_root(&plan, None, &base, &link)
            .expect_err("symlink plan path must not be followed");
        let contents = std::fs::read_to_string(&target).expect("target should be readable");
        let _ = std::fs::remove_dir_all(&base);

        assert_eq!(error.kind(), std::io::ErrorKind::AlreadyExists);
        assert_eq!(contents, "do-not-clobber");
    }

    #[test]
    fn solver_plan_json_rejects_relative_parent_escape() {
        let plan = laminar_simple_test_plan(1000.0, 0.001);
        let base = std::env::temp_dir().join(format!(
            "ferrum-plan-root-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system clock should be after Unix epoch")
                .as_nanos()
        ));
        let trusted = base.join("trusted");
        std::fs::create_dir_all(&trusted).expect("trusted root should be created");

        let parent_error =
            write_solver_plan_json_in_root(&plan, None, &trusted, Path::new("../escaped.json"))
                .expect_err("parent traversal must be rejected");

        assert_eq!(parent_error.kind(), std::io::ErrorKind::InvalidInput);
        assert!(!base.join("escaped.json").exists());
        let _ = std::fs::remove_dir_all(base);
    }

    #[test]
    fn solver_plan_json_writes_absolute_path_outside_working_directory() {
        let plan = laminar_simple_test_plan(1000.0, 0.001);
        let base = output_test_dir("plan-absolute");
        let trusted = base.join("trusted");
        std::fs::create_dir_all(&trusted).expect("trusted root should be created");
        let path = base.join("outside/nested/plan.json");

        write_solver_plan_json_in_root(&plan, None, &trusted, &path)
            .expect("trusted absolute plan path should be written outside the working directory");

        let json = std::fs::read_to_string(&path).expect("solver plan should be readable");
        assert!(json.contains("\"schemaVersion\": 2"));
        assert!(json.contains("\"caseDir\": \"case\""));
        assert!(base.join("outside/nested").is_dir());
        let _ = std::fs::remove_dir_all(base);
    }

    #[test]
    fn solver_plan_json_absolute_path_does_not_clobber_existing_file() {
        let plan = laminar_simple_test_plan(1000.0, 0.001);
        let base = output_test_dir("plan-absolute-existing");
        let trusted = base.join("trusted");
        let outside = base.join("outside");
        std::fs::create_dir_all(&trusted).expect("trusted root should be created");
        std::fs::create_dir_all(&outside).expect("outside root should be created");
        let path = outside.join("plan.json");
        std::fs::write(&path, "do-not-clobber").expect("existing plan should be written");

        let error = write_solver_plan_json_in_root(&plan, None, &trusted, &path)
            .expect_err("existing absolute plan path must not be clobbered");

        assert_eq!(error.kind(), std::io::ErrorKind::AlreadyExists);
        assert_eq!(
            std::fs::read_to_string(&path).expect("existing plan should remain readable"),
            "do-not-clobber"
        );
        let _ = std::fs::remove_dir_all(base);
    }

    #[cfg(any(unix, windows))]
    #[test]
    fn solver_plan_json_absolute_path_rejects_final_symlink() {
        let plan = laminar_simple_test_plan(1000.0, 0.001);
        let base = output_test_dir("plan-absolute-symlink");
        let trusted = base.join("trusted");
        let outside = base.join("outside");
        let target_root = base.join("target");
        std::fs::create_dir_all(&trusted).expect("trusted root should be created");
        std::fs::create_dir_all(&outside).expect("outside root should be created");
        std::fs::create_dir_all(&target_root).expect("target root should be created");
        let target = target_root.join("target.json");
        let link = outside.join("plan.json");
        std::fs::write(&target, "do-not-clobber").expect("target should be written");
        if create_output_file_symlink(&target, &link).is_err() {
            let _ = std::fs::remove_dir_all(base);
            return;
        }

        write_solver_plan_json_in_root(&plan, None, &trusted, &link)
            .expect_err("absolute plan symlink must not be followed");

        assert_eq!(
            std::fs::read_to_string(&target).expect("target should remain readable"),
            "do-not-clobber"
        );
        let _ = std::fs::remove_dir_all(base);
    }

    #[cfg(windows)]
    #[test]
    fn solver_plan_json_absolute_path_rejects_windows_ads() {
        let plan = laminar_simple_test_plan(1000.0, 0.001);
        let base = output_test_dir("plan-absolute-ads");
        let trusted = base.join("trusted");
        let outside = base.join("outside");
        std::fs::create_dir_all(&trusted).expect("trusted root should be created");
        std::fs::create_dir_all(&outside).expect("outside root should be created");
        let path = outside.join("plan.json:stream");

        let error = write_solver_plan_json_in_root(&plan, None, &trusted, &path)
            .expect_err("absolute plan ADS path must be rejected");

        assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput);
        assert!(!outside.join("plan.json").exists());
        let _ = std::fs::remove_dir_all(base);
    }

    #[cfg(windows)]
    #[test]
    fn solver_plan_json_rejects_windows_reparse_ancestor() {
        use std::os::windows::fs::symlink_dir;

        let plan = laminar_simple_test_plan(1000.0, 0.001);
        let base = std::env::temp_dir().join(format!(
            "ferrum-plan-reparse-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system clock should be after Unix epoch")
                .as_nanos()
        ));
        let trusted = base.join("trusted");
        let outside = base.join("outside");
        std::fs::create_dir_all(&trusted).expect("trusted root should be created");
        std::fs::create_dir_all(&outside).expect("outside root should be created");
        if symlink_dir(&outside, trusted.join("linked")).is_err() {
            let _ = std::fs::remove_dir_all(base);
            return;
        }

        assert!(
            write_solver_plan_json_in_root(&plan, None, &trusted, Path::new("linked/plan.json"),)
                .is_err()
        );
        assert!(!outside.join("plan.json").exists());
        let _ = std::fs::remove_dir_all(base);
    }

    #[test]
    fn parses_laminar_simple_fv_schemes_subset() {
        assert_eq!(
            parse_laminar_simple_gradient_scheme("Gauss linear").expect("grad scheme"),
            LaminarSimpleGradientScheme::GaussLinear
        );
        assert_eq!(
            parse_laminar_simple_gradient_scheme("leastSquares").expect("least-squares scheme"),
            LaminarSimpleGradientScheme::LeastSquares
        );
        assert_eq!(
            LaminarSimpleGradientScheme::LeastSquares.to_string(),
            "leastSquares"
        );
        assert_eq!(
            parse_laminar_simple_gradient_scheme("cellLimited Gauss linear 0.5")
                .expect("limited grad scheme"),
            LaminarSimpleGradientScheme::CellLimitedGaussLinear(0.5)
        );
        assert_eq!(
            LaminarSimpleGradientScheme::CellLimitedGaussLinear(0.5).to_string(),
            "cellLimited Gauss linear 0.5"
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
    fn bounded_linear_upwind_syntax_is_exact_and_canonical() {
        let mut plan = laminar_simple_test_plan(1.0, 1.0);
        plan.numerics
            .fv_schemes
            .entries
            .iter_mut()
            .find(|entry| entry.section == "divSchemes" && entry.key == "div(phi,U)")
            .expect("div scheme")
            .value = "bounded Gauss linearUpwind limited".to_string();
        plan.numerics
            .fv_schemes
            .entries
            .push(SolverNumericsEntryPlan {
                section: "gradSchemes".to_string(),
                key: "limited".to_string(),
                value: "cellLimited Gauss linear 1".to_string(),
            });
        let scheme = resolve_laminar_simple_convection_scheme(&plan).expect("bounded scheme");
        assert_eq!(scheme.to_string(), "bounded Gauss linearUpwind limited");

        for invalid in [
            "bounded Gauss linearUpwind",
            "bounded Gauss linearUpwind limited trailing",
            "bounded Gauss upwind",
        ] {
            plan.numerics
                .fv_schemes
                .entries
                .iter_mut()
                .find(|entry| entry.section == "divSchemes" && entry.key == "div(phi,U)")
                .expect("div scheme")
                .value = invalid.to_string();
            assert!(resolve_laminar_simple_convection_scheme(&plan).is_err());
        }

        assert!(
            parse_laminar_simple_laplacian_scheme(
                "bounded Gauss linear corrected",
                LaminarSimpleSnGradScheme::Corrected,
            )
            .is_err()
        );
    }

    #[test]
    fn bounded_limited_gradient_requires_a_valid_unique_reference() {
        let mut plan = laminar_simple_test_plan(1.0, 1.0);
        plan.numerics
            .fv_schemes
            .entries
            .iter_mut()
            .find(|entry| entry.section == "divSchemes" && entry.key == "div(phi,U)")
            .expect("div scheme")
            .value = "bounded Gauss linearUpwind limited".to_string();
        assert!(resolve_laminar_simple_convection_scheme(&plan).is_err());

        plan.numerics
            .fv_schemes
            .entries
            .push(SolverNumericsEntryPlan {
                section: "gradSchemes".to_string(),
                key: "limited".to_string(),
                value: "$selected".to_string(),
            });
        plan.numerics
            .fv_schemes
            .entries
            .push(SolverNumericsEntryPlan {
                section: "gradSchemes".to_string(),
                key: "selected".to_string(),
                value: "cellLimited Gauss linear 0.5".to_string(),
            });
        assert!(resolve_laminar_simple_convection_scheme(&plan).is_ok());

        for invalid in ["$missing", "$limited", "$selected extra", "Gauss cubic"] {
            plan.numerics
                .fv_schemes
                .entries
                .iter_mut()
                .find(|entry| entry.section == "gradSchemes" && entry.key == "limited")
                .expect("limited gradient")
                .value = invalid.to_string();
            assert!(resolve_laminar_simple_convection_scheme(&plan).is_err());
        }
        plan.numerics
            .fv_schemes
            .entries
            .push(SolverNumericsEntryPlan {
                section: "gradSchemes".to_string(),
                key: "limited".to_string(),
                value: "Gauss linear".to_string(),
            });
        assert!(resolve_laminar_simple_convection_scheme(&plan).is_err());
    }

    #[test]
    fn gradient_scheme_parser_rejects_invalid_coefficients_and_arity() {
        for value in [
            "leastSquares trailing",
            "weightedLeastSquares",
            "cellLimited leastSquares 1",
            "cellLimited Gauss linear",
            "cellLimited Gauss linear 0.5 trailing",
            "cellLimited Gauss linear -0.1",
            "cellLimited Gauss linear 1.1",
            "cellLimited Gauss linear NaN",
            "cellLimited Gauss linear inf",
            "cellLimited Gauss cubic 0.5",
        ] {
            assert!(
                parse_laminar_simple_gradient_scheme(value).is_err(),
                "'{value}' must be rejected"
            );
        }
    }

    fn set_gradient_scheme_value(plan: &mut SolverCasePlan, key: &str, value: String) {
        if let Some(entry) = plan
            .numerics
            .fv_schemes
            .entries
            .iter_mut()
            .find(|entry| entry.section == "gradSchemes" && entry.key == key)
        {
            entry.value = value;
        } else {
            plan.numerics
                .fv_schemes
                .entries
                .push(SolverNumericsEntryPlan {
                    section: "gradSchemes".to_string(),
                    key: key.to_string(),
                    value,
                });
        }
    }

    fn append_gradient_alias_chain(
        plan: &mut SolverCasePlan,
        start: &str,
        prefix: &str,
        aliases: usize,
        terminal: &str,
    ) {
        assert!(aliases > 0);
        set_gradient_scheme_value(plan, start, format!("${prefix}0"));
        plan.numerics
            .fv_schemes
            .entries
            .extend((0..aliases).map(|index| SolverNumericsEntryPlan {
                section: "gradSchemes".to_string(),
                key: format!("{prefix}{index}"),
                value: if index + 1 == aliases {
                    terminal.to_string()
                } else {
                    format!("${prefix}{}", index + 1)
                },
            }));
    }

    #[test]
    fn gradient_alias_chains_have_no_artificial_depth_limit() {
        for aliases in [65, 4_096] {
            let mut plan = laminar_simple_test_plan(1.0, 1.0);
            let prefix = format!("long{aliases}_");
            append_gradient_alias_chain(&mut plan, "default", &prefix, aliases, "Gauss linear");

            assert_eq!(
                resolved_gradient_scheme_value(&plan, "grad(U)", Some("default"))
                    .expect("valid long alias chain"),
                "Gauss linear"
            );
        }
    }

    #[test]
    fn late_gradient_alias_failures_preserve_exact_semantics() {
        let mut cyclic = laminar_simple_test_plan(1.0, 1.0);
        append_gradient_alias_chain(&mut cyclic, "default", "cycle", 96, "$cycle32");
        let cycle_error = resolved_gradient_scheme_value(&cyclic, "grad(U)", Some("default"))
            .expect_err("late cycle must fail");
        assert!(cycle_error.starts_with("fvSchemes gradSchemes alias cycle: default -> cycle0"));
        assert!(cycle_error.ends_with("cycle95 -> cycle32"));

        let mut missing = laminar_simple_test_plan(1.0, 1.0);
        append_gradient_alias_chain(&mut missing, "default", "missing", 96, "$missingTail");
        assert_eq!(
            resolved_gradient_scheme_value(&missing, "grad(U)", Some("default"))
                .expect_err("late missing alias must fail"),
            "fvSchemes gradSchemes alias '$missingTail' references a missing entry"
        );

        let mut duplicate = laminar_simple_test_plan(1.0, 1.0);
        append_gradient_alias_chain(&mut duplicate, "default", "duplicate", 96, "Gauss linear");
        duplicate
            .numerics
            .fv_schemes
            .entries
            .push(SolverNumericsEntryPlan {
                section: "gradSchemes".to_string(),
                key: "duplicate95".to_string(),
                value: "Gauss linear".to_string(),
            });
        assert_eq!(
            resolved_gradient_scheme_value(&duplicate, "grad(U)", Some("default"))
                .expect_err("late duplicate alias must fail"),
            "fvSchemes gradSchemes entry 'duplicate95' must be unique, found 2"
        );
    }

    #[test]
    fn gradient_alias_index_is_lazy_exact_and_same_section() {
        let mut plan = laminar_simple_test_plan(1.0, 1.0);
        plan.numerics.fv_schemes.entries.extend([
            SolverNumericsEntryPlan {
                section: "gradSchemes".to_string(),
                key: "unreachableDuplicate".to_string(),
                value: "$".to_string(),
            },
            SolverNumericsEntryPlan {
                section: "gradSchemes".to_string(),
                key: "unreachableDuplicate".to_string(),
                value: "$missing".to_string(),
            },
            SolverNumericsEntryPlan {
                section: "divSchemes".to_string(),
                key: "default".to_string(),
                value: "$broken".to_string(),
            },
            SolverNumericsEntryPlan {
                section: "divSchemes".to_string(),
                key: "foreignOnly".to_string(),
                value: "Gauss linear".to_string(),
            },
        ]);
        assert_eq!(
            resolved_gradient_scheme_value(&plan, "grad(U)", Some("default"))
                .expect("unreachable and other-section entries must be ignored"),
            "Gauss linear"
        );

        set_gradient_scheme_value(&mut plan, "default", "$foreignOnly".to_string());
        assert_eq!(
            resolved_gradient_scheme_value(&plan, "grad(U)", Some("default"))
                .expect_err("other-section entry must not satisfy an alias"),
            "fvSchemes gradSchemes alias '$foreignOnly' references a missing entry"
        );
    }

    #[test]
    fn requested_gradient_duplicate_precedes_valid_fallback() {
        let mut plan = laminar_simple_test_plan(1.0, 1.0);
        for value in ["Gauss linear", "cellLimited Gauss linear 1"] {
            plan.numerics
                .fv_schemes
                .entries
                .push(SolverNumericsEntryPlan {
                    section: "gradSchemes".to_string(),
                    key: "grad(U)".to_string(),
                    value: value.to_string(),
                });
        }

        assert_eq!(
            resolved_gradient_scheme_value(&plan, "grad(U)", Some("default"))
                .expect_err("requested duplicate must not fall back"),
            "fvSchemes gradSchemes entry 'grad(U)' must be unique, found 2"
        );
    }

    #[test]
    fn laminar_scheme_entrypoint_shares_gradient_alias_index() {
        let mut plan = laminar_simple_test_plan(1.0, 1.0);
        set_gradient_scheme_value(&mut plan, "grad(p)", "$shared".to_string());
        set_gradient_scheme_value(&mut plan, "grad(U)", "$velocity".to_string());
        set_gradient_scheme_value(&mut plan, "velocity", "$shared".to_string());
        set_gradient_scheme_value(&mut plan, "limited", "$shared".to_string());
        set_gradient_scheme_value(
            &mut plan,
            "shared",
            "cellLimited Gauss linear 1".to_string(),
        );
        plan.numerics
            .fv_schemes
            .entries
            .iter_mut()
            .find(|entry| entry.section == "divSchemes" && entry.key == "div(phi,U)")
            .expect("div scheme")
            .value = "bounded Gauss linearUpwind limited".to_string();

        let schemes = resolve_laminar_simple_schemes(&plan).expect("shared alias index");
        assert_eq!(
            schemes.grad_p,
            LaminarSimpleGradientScheme::CellLimitedGaussLinear(1.0)
        );
        assert_eq!(
            schemes.grad_u,
            LaminarSimpleGradientScheme::CellLimitedGaussLinear(1.0)
        );
        assert_eq!(
            schemes.div_phi_u,
            LaminarSimpleConvectionScheme::BoundedGaussLinearUpwind(
                LaminarSimpleGradientScheme::CellLimitedGaussLinear(1.0)
            )
        );
    }

    #[test]
    fn gradient_scheme_aliases_are_exact_unique_and_same_section() {
        let mut passing = laminar_simple_test_plan(1.0, 1.0);
        passing
            .numerics
            .fv_schemes
            .entries
            .push(SolverNumericsEntryPlan {
                section: "gradSchemes".to_string(),
                key: "cylinderGradient".to_string(),
                value: "cellLimited Gauss linear 1".to_string(),
            });
        passing
            .numerics
            .fv_schemes
            .entries
            .iter_mut()
            .find(|entry| entry.section == "gradSchemes" && entry.key == "default")
            .expect("default gradient")
            .value = "$cylinderGradient".to_string();
        assert_eq!(
            resolved_gradient_scheme_value(&passing, "grad(U)", Some("default"))
                .expect("same-section alias"),
            "cellLimited Gauss linear 1"
        );

        for reference in [
            "$missing",
            "$default extra",
            "Gauss$default",
            "$",
            "$divSchemes.default",
            "${default}",
        ] {
            let mut plan = laminar_simple_test_plan(1.0, 1.0);
            plan.numerics
                .fv_schemes
                .entries
                .iter_mut()
                .find(|entry| entry.section == "gradSchemes" && entry.key == "default")
                .expect("default gradient")
                .value = reference.to_string();
            assert!(
                resolved_gradient_scheme_value(&plan, "grad(U)", Some("default")).is_err(),
                "unsupported reference '{reference}' must fail"
            );
        }

        let mut duplicate = laminar_simple_test_plan(1.0, 1.0);
        duplicate
            .numerics
            .fv_schemes
            .entries
            .push(SolverNumericsEntryPlan {
                section: "gradSchemes".to_string(),
                key: "default".to_string(),
                value: "Gauss linear".to_string(),
            });
        assert!(resolved_gradient_scheme_value(&duplicate, "grad(U)", Some("default")).is_err());

        let mut cyclic = laminar_simple_test_plan(1.0, 1.0);
        cyclic
            .numerics
            .fv_schemes
            .entries
            .iter_mut()
            .find(|entry| entry.section == "gradSchemes" && entry.key == "default")
            .expect("default gradient")
            .value = "$a".to_string();
        cyclic.numerics.fv_schemes.entries.extend([
            SolverNumericsEntryPlan {
                section: "gradSchemes".to_string(),
                key: "a".to_string(),
                value: "$b".to_string(),
            },
            SolverNumericsEntryPlan {
                section: "gradSchemes".to_string(),
                key: "b".to_string(),
                value: "$a".to_string(),
            },
        ]);
        assert!(resolved_gradient_scheme_value(&cyclic, "grad(U)", Some("default")).is_err());
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
                    loaded_scalars: None,
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
