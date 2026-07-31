use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::time::{SystemTime, UNIX_EPOCH};

use ferrum_mesh::cylinder_ogrid::{CylinderOGridPreset, generate_cylinder_ogrid};
use ferrum_mesh::foam::{FoamWriteOptions, write_openfoam_case_with_options};
use ferrum_mesh::geometry::{CartesianAxis, PolyMeshQualityOptions, summarize_poly_mesh_quality};
use ferrum_mesh::gmsh::read_msh22_ascii;
use ferrum_mesh::gmsh_write::write_msh22_ascii;
use ferrum_mesh::poly_mesh::PolyMesh;

const REFERENCE_CD: f64 = 10.655_858_0;
const MAX_REFERENCE_CD_RELATIVE_ERROR: f64 = 0.15;
const MAX_REFINEMENT_CD_RELATIVE_DRIFT: f64 = 0.05;
const MAX_ABSOLUTE_CL: f64 = 1.0e-6;
const MAX_ABSOLUTE_CONTINUITY_ERROR: f64 = 1.0e-6;

struct TemporaryRoot(PathBuf);

impl TemporaryRoot {
    fn new() -> Self {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock after epoch")
            .as_nanos();
        let path =
            std::env::temp_dir().join(format!("ferrum-cylinder-c3-{}-{nonce}", std::process::id()));
        fs::create_dir(&path).expect("create C3 temporary root");
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TemporaryRoot {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[derive(Debug)]
struct RunEvidence {
    cells: usize,
    simple_iterations: usize,
    stop_reason: String,
    local_continuity: f64,
    global_continuity: f64,
    cumulative_global_continuity: f64,
    cylinder_faces: usize,
    cd: f64,
    cl: f64,
    final_u: Vec<u8>,
    final_p: Vec<u8>,
}

#[test]
#[ignore = "release-only C3 gate generates and solves the 5,376- and 21,504-cell meshes"]
fn cylinder_c3_coarse_fine_physical_acceptance() {
    assert_reference_manifest();
    let temporary = TemporaryRoot::new();

    eprintln!("C3 gate: starting Coarse repetition 1/2");
    let coarse_first = run_preset(temporary.path(), CylinderOGridPreset::Coarse, "coarse-a");
    eprintln!(
        "C3 gate: Coarse repetition 1/2 passed (iterations={}, localContinuity={}, globalContinuity={}, cumulativeGlobalContinuity={}, Cd={}, Cl={})",
        coarse_first.simple_iterations,
        coarse_first.local_continuity,
        coarse_first.global_continuity,
        coarse_first.cumulative_global_continuity,
        coarse_first.cd,
        coarse_first.cl
    );
    eprintln!("C3 gate: starting Coarse repetition 2/2");
    let coarse_second = run_preset(temporary.path(), CylinderOGridPreset::Coarse, "coarse-b");
    assert_repeated_run_equal(&coarse_first, &coarse_second);
    eprintln!("C3 gate: Coarse deterministic repetition passed");

    eprintln!("C3 gate: starting Fine");
    let fine = run_preset(temporary.path(), CylinderOGridPreset::Fine, "fine");
    let refinement_drift = (coarse_first.cd - fine.cd).abs() / fine.cd.abs();
    assert!(
        refinement_drift <= MAX_REFINEMENT_CD_RELATIVE_DRIFT,
        "Coarse/Fine Cd drift {refinement_drift} exceeds {MAX_REFINEMENT_CD_RELATIVE_DRIFT}: coarse={}, fine={}",
        coarse_first.cd,
        fine.cd
    );
    eprintln!(
        "C3 gate: Fine and refinement passed (iterations={}, localContinuity={}, globalContinuity={}, cumulativeGlobalContinuity={}, Cd={}, Cl={}, drift={})",
        fine.simple_iterations,
        fine.local_continuity,
        fine.global_continuity,
        fine.cumulative_global_continuity,
        fine.cd,
        fine.cl,
        refinement_drift
    );
}

fn run_preset(root: &Path, preset: CylinderOGridPreset, label: &str) -> RunEvidence {
    let repository = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..");
    let source_case = repository.join("tutorials/incompressibleFluid/cylinder/ferrum/case");
    let profile = repository.join("validation/profiles/incompressibleFluid/cylinder/c3/system");
    let run_root = root.join(label);
    let case = run_root.join("case");
    fs::create_dir_all(&run_root).expect("create preset run root");

    let generated = generate_cylinder_ogrid(&preset.config()).expect("generate C3 O-grid");
    let msh = run_root.join("cylinder.msh");
    write_msh22_ascii(&msh, &generated).expect("write C3 neutral mesh");
    let imported = read_msh22_ascii(&msh).expect("read C3 neutral mesh");

    let mut write_options = FoamWriteOptions::default();
    write_options.set_patch_type("cylinder", "wall");
    write_options.set_patch_type("frontAndBack", "empty");
    let written = write_openfoam_case_with_options(&imported, &case, &msh, &write_options)
        .expect("write C3 polyMesh");
    assert_eq!(written.cells, generated.cells.len());
    assert_eq!(written.unmatched_boundary_faces, 0);
    assert_eq!(written.duplicate_boundary_faces, 0);
    assert_eq!(written.non_manifold_faces, 0);

    copy_tree(&source_case.join("0"), &case.join("0"));
    copy_file(
        &source_case.join("constant/transportProperties"),
        &case.join("constant/transportProperties"),
    );
    copy_file(
        &source_case.join("system/ferrumBackends"),
        &case.join("system/ferrumBackends"),
    );
    copy_file(
        &source_case.join("system/fvSchemes"),
        &case.join("system/fvSchemes"),
    );
    copy_file(
        &profile.join("controlDict"),
        &case.join("system/controlDict"),
    );
    copy_file(&profile.join("fvSolution"), &case.join("system/fvSolution"));

    let poly_mesh = PolyMesh::read(&case.join("constant/polyMesh")).expect("read C3 polyMesh");
    let expected = match preset {
        CylinderOGridPreset::Coarse => (5_376, 128),
        CylinderOGridPreset::Fine => (21_504, 256),
        CylinderOGridPreset::LegacySmoke => panic!("LegacySmoke is not a C3 production mesh"),
    };
    assert_eq!(poly_mesh.cell_count(), expected.0);
    assert_mesh_quality(&poly_mesh, preset);

    let output = Command::new(env!("CARGO_BIN_EXE_ferrumRun"))
        .current_dir(&run_root)
        .arg("-solver")
        .arg("incompressibleFluid")
        .arg("-case")
        .arg(&case)
        .arg("--minSimpleIterations")
        .arg("10")
        .arg("--wallForcePatches")
        .arg("cylinder")
        .arg("--forceReferenceSpeed")
        .arg("0.015")
        .arg("--forceReferenceArea")
        .arg("1e-6")
        .arg("--solveReportJson")
        .arg("report.json")
        .arg("--solveReportMarkdown")
        .arg("report.md")
        .arg("--writeFinalFields")
        .arg("final")
        .output()
        .expect("run C3 cylinder case");
    assert_success(&output, preset);

    let stdout = String::from_utf8(output.stdout).expect("C3 stdout is UTF-8");
    let solve = evidence_line(&stdout, "incompressibleFluid solve:");
    let outer = evidence_line(&stdout, "incompressibleFluid outerConvergence:");
    let linear = evidence_line(&stdout, "incompressibleFluid linearSolves:");
    let continuity = evidence_line(&stdout, "incompressibleFluid continuityErrors:");
    let forces = evidence_line(&stdout, "incompressibleFluid wallForces:");
    let force_method = evidence_line(&stdout, "incompressibleFluid wallForceMethod:");

    assert_eq!(token(solve, "converged"), "yes");
    assert_eq!(token(solve, "stopReason"), "Converged");
    assert_eq!(token(outer, "configured"), "yes");
    assert_eq!(token(outer, "evaluated"), "yes");
    assert_eq!(token(outer, "converged"), "yes");
    assert_eq!(token(linear, "finalMomentumConverged"), "yes");
    assert_eq!(token(linear, "finalPressureConverged"), "yes");
    assert_eq!(parse_usize(linear, "momentumNonConvergedPredictors"), 0);
    assert_eq!(
        parse_usize(linear, "momentumComponentNonConvergedSolves"),
        0
    );
    assert_eq!(
        parse_usize(linear, "pressureCorrectionNonConvergedSolves"),
        0
    );

    let simple_iterations = parse_usize(solve, "simpleIterations");
    assert!((10..=5_000).contains(&simple_iterations));
    let local_continuity = parse_f64(continuity, "local");
    let global_continuity = parse_f64(continuity, "global");
    let cumulative_global_continuity = parse_f64(continuity, "cumulativeGlobal");
    assert!(
        local_continuity.abs() <= MAX_ABSOLUTE_CONTINUITY_ERROR,
        "{preset:?} normalized local continuity {local_continuity} exceeds {MAX_ABSOLUTE_CONTINUITY_ERROR} after {simple_iterations} SIMPLE iterations (global={global_continuity}, cumulativeGlobal={cumulative_global_continuity})"
    );
    assert!(
        global_continuity.abs() <= MAX_ABSOLUTE_CONTINUITY_ERROR,
        "{preset:?} normalized global continuity {global_continuity} exceeds {MAX_ABSOLUTE_CONTINUITY_ERROR} after {simple_iterations} SIMPLE iterations (local={local_continuity}, cumulativeGlobal={cumulative_global_continuity})"
    );
    assert!(
        cumulative_global_continuity.is_finite(),
        "{preset:?} cumulative normalized global continuity is non-finite after {simple_iterations} SIMPLE iterations"
    );

    assert_eq!(token(forces, "patches"), "cylinder");
    assert_eq!(parse_usize(forces, "selectedPatches"), 1);
    assert_eq!(parse_usize(forces, "selectedFaces"), expected.1);
    assert_eq!(
        token(force_method, "tractionMethod"),
        "reconstructedGradientFullDeviatoric"
    );
    assert_eq!(parse_usize(force_method, "tractionMethodVersion"), 1);
    assert_eq!(token(force_method, "forceConvention"), "fluidOnBody");
    assert_eq!(
        token(force_method, "faceAreaVectorOrientation"),
        "outwardFromFluid"
    );
    assert_eq!(
        token(force_method, "pressureFaceTreatment"),
        "zeroGradientOwner"
    );
    let cd = parse_f64(forces, "dragTotal");
    let cl = parse_f64(forces, "liftTotal");
    let reference_error = (cd - REFERENCE_CD).abs() / REFERENCE_CD;
    assert!(
        reference_error <= MAX_REFERENCE_CD_RELATIVE_ERROR,
        "{preset:?} Cd={cd} differs from {REFERENCE_CD} by {reference_error}"
    );
    assert!(cl.abs() <= MAX_ABSOLUTE_CL, "{preset:?} Cl={cl}");

    let json = fs::read_to_string(run_root.join("report.json")).expect("read C3 JSON report");
    let markdown = fs::read_to_string(run_root.join("report.md")).expect("read C3 Markdown report");
    assert!(json.contains("\"continuityErrors\""));
    assert!(json.contains("\"wallForces\""));
    assert!(json.contains("\"tractionMethod\": \"reconstructedGradientFullDeviatoric\""));
    assert!(json.contains("\"tractionMethodVersion\": 1"));
    assert!(json.contains("\"forceConvention\": \"fluidOnBody\""));
    assert!(json.contains("\"faceAreaVectorOrientation\": \"outwardFromFluid\""));
    assert!(json.contains("\"pressureFaceTreatment\": \"zeroGradientOwner\""));
    assert!(json.contains("\"velocityGradientScheme\": \"cellLimited Gauss linear 1\""));
    assert!(markdown.contains("Continuity errors"));
    assert!(markdown.contains("Wall forces"));
    assert!(markdown.contains("| Traction method | reconstructedGradientFullDeviatoric |"));
    assert!(markdown.contains("| Force convention | fluidOnBody |"));
    assert!(markdown.contains("| Pressure face treatment | zeroGradientOwner |"));
    assert!(markdown.contains("| Velocity gradient scheme | cellLimited Gauss linear 1 |"));

    RunEvidence {
        cells: expected.0,
        simple_iterations,
        stop_reason: token(solve, "stopReason").to_string(),
        local_continuity,
        global_continuity,
        cumulative_global_continuity,
        cylinder_faces: expected.1,
        cd,
        cl,
        final_u: fs::read(run_root.join("final/U")).expect("read final U"),
        final_p: fs::read(run_root.join("final/p")).expect("read final p"),
    }
}

fn assert_mesh_quality(poly_mesh: &PolyMesh, preset: CylinderOGridPreset) {
    let quality = summarize_poly_mesh_quality(
        poly_mesh,
        &PolyMeshQualityOptions {
            empty_extrusion_axis: Some(CartesianAxis::Z),
        },
    )
    .expect("summarize C3 mesh quality");
    assert!(quality.problematic_face_indices.is_empty());
    assert!(quality.problematic_cell_indices.is_empty());
    assert!(quality.non_finite_face_geometry_indices.is_empty());
    assert!(quality.non_positive_face_area_indices.is_empty());
    assert!(quality.non_finite_cell_geometry_indices.is_empty());
    assert!(quality.non_positive_cell_volume_indices.is_empty());
    assert!(
        quality
            .max_internal_non_orthogonality_degrees
            .is_some_and(|value| value <= 50.0),
        "{preset:?} non-orthogonality gate failed: {quality:?}"
    );
    assert!(
        quality
            .max_normalized_internal_skewness
            .is_some_and(|value| value <= 0.55),
        "{preset:?} skewness gate failed: {quality:?}"
    );
    assert!(
        quality
            .max_active_edge_aspect_ratio
            .is_some_and(|value| value <= 4.0),
        "{preset:?} aspect-ratio gate failed: {quality:?}"
    );
}

fn assert_repeated_run_equal(first: &RunEvidence, second: &RunEvidence) {
    assert_eq!(first.cells, second.cells);
    assert_eq!(first.simple_iterations, second.simple_iterations);
    assert_eq!(first.stop_reason, second.stop_reason);
    assert_eq!(first.cylinder_faces, second.cylinder_faces);
    for (left, right) in [
        (first.local_continuity, second.local_continuity),
        (first.global_continuity, second.global_continuity),
        (
            first.cumulative_global_continuity,
            second.cumulative_global_continuity,
        ),
        (first.cd, second.cd),
        (first.cl, second.cl),
    ] {
        assert_eq!(left.to_bits(), right.to_bits());
    }
    assert_eq!(first.final_u, second.final_u);
    assert_eq!(first.final_p, second.final_p);
}

fn assert_reference_manifest() {
    let repository = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..");
    let manifest = fs::read_to_string(
        repository.join("tutorials/incompressibleFluid/cylinder/comparison.toml"),
    )
    .expect("read Cylinder comparison manifest");
    assert!(manifest.contains("drag_coefficient_relative = 0.15"));
    assert!(manifest.contains("lift_coefficient_absolute = 1.0e-6"));
    assert!(manifest.contains("local_continuity_absolute = 1.0e-6"));
    assert!(manifest.contains("global_continuity_absolute = 1.0e-6"));
    assert!(manifest.contains("10.6558580"));
}

fn copy_tree(source: &Path, destination: &Path) {
    fs::create_dir_all(destination).expect("create copied directory");
    for entry in fs::read_dir(source).expect("read copied directory") {
        let entry = entry.expect("read copied entry");
        let target = destination.join(entry.file_name());
        if entry.file_type().expect("read copied entry type").is_dir() {
            copy_tree(&entry.path(), &target);
        } else {
            copy_file(&entry.path(), &target);
        }
    }
}

fn copy_file(source: &Path, destination: &Path) {
    if let Some(parent) = destination.parent() {
        fs::create_dir_all(parent).expect("create copied file parent");
    }
    fs::copy(source, destination).expect("copy C3 case input");
}

fn assert_success(output: &Output, preset: CylinderOGridPreset) {
    assert!(
        output.status.success(),
        "{preset:?} C3 solve failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

fn evidence_line<'a>(stdout: &'a str, prefix: &str) -> &'a str {
    stdout
        .lines()
        .find(|line| line.starts_with(prefix))
        .unwrap_or_else(|| panic!("missing evidence line {prefix:?}\n{stdout}"))
}

fn token<'a>(line: &'a str, key: &str) -> &'a str {
    let prefix = format!("{key}=");
    line.split_whitespace()
        .find_map(|part| part.strip_prefix(&prefix))
        .unwrap_or_else(|| panic!("missing token {key:?} in {line:?}"))
}

fn parse_usize(line: &str, key: &str) -> usize {
    token(line, key)
        .parse()
        .unwrap_or_else(|_| panic!("token {key:?} is not usize in {line:?}"))
}

fn parse_f64(line: &str, key: &str) -> f64 {
    let value = token(line, key)
        .parse::<f64>()
        .unwrap_or_else(|_| panic!("token {key:?} is not f64 in {line:?}"));
    assert!(value.is_finite(), "token {key:?} is not finite in {line:?}");
    value
}
