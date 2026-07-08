mod case;

use std::env;
use std::path::{Path, PathBuf};

use case::{InitCaseOptions, init_case};
use ferrum_mesh::check::read_case_summary;
use ferrum_mesh::foam::{FoamWriteOptions, write_openfoam_case_with_options};
use ferrum_mesh::gmsh::read_msh22_ascii;
use ferrum_mesh::interfaces::{read_interface_config, validate_interface_config};
use ferrum_mesh::regions::{
    InterfaceRegistrySummary, InterfaceSummary, build_interface_registry,
    read_region_mesh_summaries, split_regions_by_cell_zones,
};

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
        "Loaded {} points, {} hex cells, {} boundary quads",
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
    println!();
    println!("aliases:");
    println!("  initFerrumCase <caseDir> [--region <name> ...] [--force]");
    println!("  gmshToFerrumFoam <mesh.msh> [-case <caseDir>] [patch type options]");
    println!("  checkFerrumMesh [-case <caseDir>]");
    println!("  splitFerrumMeshRegions [-case <caseDir>] [-cellZones]");
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
