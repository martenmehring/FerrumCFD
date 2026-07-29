use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::time::{SystemTime, UNIX_EPOCH};

struct Cleanup {
    paths: Vec<PathBuf>,
    empty_dirs: Vec<PathBuf>,
}

impl Drop for Cleanup {
    fn drop(&mut self) {
        for path in &self.paths {
            let _ = fs::remove_dir_all(path);
        }
        for path in &self.empty_dirs {
            let _ = fs::remove_dir(path);
        }
    }
}

fn copy_tree(source: &Path, destination: &Path) {
    fs::create_dir_all(destination).expect("create temporary case directory");
    for entry in fs::read_dir(source).expect("read packaged lid-driven cavity case") {
        let entry = entry.expect("read lid-driven cavity entry");
        let target = destination.join(entry.file_name());
        if entry.file_type().expect("read entry type").is_dir() {
            copy_tree(&entry.path(), &target);
        } else {
            fs::copy(entry.path(), target).expect("copy lid-driven cavity case file");
        }
    }
}

fn run_case(case: &Path, extra: &[String]) -> Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_ferrumRun"));
    command
        .arg("-solver")
        .arg("incompressibleFluid")
        .arg("-case")
        .arg(case)
        .args(extra);
    command
        .output()
        .expect("run ferrumRun lid-driven cavity case")
}

fn stdout(output: &Output) -> String {
    String::from_utf8(output.stdout.clone()).expect("ferrumRun stdout is UTF-8")
}

fn nonuniform_scalars(field: &str) -> Vec<f64> {
    let (_, list) = field
        .split_once("internalField nonuniform List<scalar>")
        .expect("nonuniform scalar field");
    let (_, values) = list.split_once('(').expect("scalar list start");
    let (values, _) = values.split_once(");").expect("scalar list end");
    values
        .lines()
        .filter_map(|line| {
            let value = line.trim();
            (!value.is_empty()).then(|| value.parse::<f64>().expect("finite scalar value"))
        })
        .collect()
}

fn nonuniform_vectors(field: &str) -> Vec<[f64; 3]> {
    let (_, list) = field
        .split_once("internalField nonuniform List<vector>")
        .expect("nonuniform vector field");
    let (_, values) = list.split_once('(').expect("vector list start");
    let (values, _) = values.split_once(");").expect("vector list end");
    values
        .lines()
        .filter_map(|line| {
            let value = line.trim();
            if value.is_empty() {
                return None;
            }
            let components = value
                .trim_start_matches('(')
                .trim_end_matches(')')
                .split_whitespace()
                .map(|component| component.parse::<f64>().expect("finite vector component"))
                .collect::<Vec<_>>();
            assert_eq!(components.len(), 3, "vector has exactly three components");
            Some([components[0], components[1], components[2]])
        })
        .collect()
}

fn json_has_exact_field(document: &str, key: &str, value: &str) -> bool {
    let with_comma = format!("\"{key}\": {value},");
    let without_comma = format!("\"{key}\": {value}");
    document.lines().any(|line| {
        let line = line.trim();
        line == with_comma || line == without_comma
    })
}

fn numeric_evidence(output: &str, name: &str) -> f64 {
    let prefix = format!("{name}=");
    output
        .lines()
        .flat_map(str::split_whitespace)
        .find_map(|item| item.strip_prefix(&prefix))
        .unwrap_or_else(|| panic!("missing numeric evidence: {name}"))
        .parse::<f64>()
        .unwrap_or_else(|_| panic!("numeric evidence is not a number: {name}"))
}

fn assert_close(left: f64, right: f64, tolerance: f64, label: &str) {
    let error = (left - right).abs();
    let scale = left.abs().max(right.abs()).max(1.0);
    assert!(
        error <= tolerance * scale,
        "{label} differs: left={left:e}, right={right:e}, error={error:e}"
    );
}

#[test]
fn packaged_lid_driven_cavity_exercises_closed_pressure_reference_end_to_end() {
    let package = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let tutorial = package.join("../../../tutorials/incompressibleFluid/lidDrivenCavity");
    let source = tutorial.join("ferrum/case");
    let mesh_source = tutorial.join("shared/geometry/lid_cavity_2x2x1.msh");
    assert!(source.join("constant/polyMesh/faces").is_file());
    assert!(mesh_source.is_file(), "canonical Gmsh source is packaged");

    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock after epoch")
        .as_nanos();
    let temporary_root = std::env::temp_dir().join(format!(
        "ferrum-lid-driven-cavity-{}-{nonce}",
        std::process::id()
    ));
    let case_three = temporary_root.join("pRef3");
    let case_zero = temporary_root.join("pRef0");
    let artifact_relative = PathBuf::from(format!(
        "target/lid-driven-cavity-tests/{}-{nonce}",
        std::process::id()
    ));
    let artifacts = package.join(&artifact_relative);
    let _cleanup = Cleanup {
        paths: vec![temporary_root.clone(), artifacts.clone()],
        empty_dirs: vec![package.join("target/lid-driven-cavity-tests")],
    };
    copy_tree(&source, &case_three);
    copy_tree(&source, &case_zero);

    let zero_solution_path = case_zero.join("system/fvSolution");
    let zero_solution =
        fs::read_to_string(&zero_solution_path).expect("read zero-gauge fvSolution");
    assert_eq!(
        zero_solution.matches("pRefValue 3;").count(),
        1,
        "packaged case has one deliberate nonzero pressure reference"
    );
    fs::write(
        &zero_solution_path,
        zero_solution.replacen("pRefValue 3;", "pRefValue 0;", 1),
    )
    .expect("write zero-gauge fvSolution");

    let preflight = run_case(&source, &["--preflight".to_string()]);
    assert!(
        preflight.status.success(),
        "preflight failed: {}",
        String::from_utf8_lossy(&preflight.stderr)
    );
    let preflight_stdout = stdout(&preflight);
    assert!(preflight_stdout.contains("mesh: dimensionality=2d-empty points=18 cells=4 faces=20"));
    assert!(preflight_stdout.contains("SIMPLE.pRefCell=0"));
    assert!(preflight_stdout.contains("SIMPLE.pRefValue=3"));

    let run = |case: &Path, label: &str| {
        let output_relative = artifact_relative.join(label);
        let solve = run_case(
            case,
            &[
                "--maxSimpleIterations".to_string(),
                "2".to_string(),
                "--solveReportJson".to_string(),
                output_relative.join("report.json").display().to_string(),
                "--writeFinalFields".to_string(),
                output_relative.join("fields").display().to_string(),
            ],
        );
        assert!(
            solve.status.success(),
            "{label} smoke solve failed: {}\n{}",
            String::from_utf8_lossy(&solve.stderr),
            String::from_utf8_lossy(&solve.stdout)
        );
        (stdout(&solve), package.join(output_relative))
    };

    let (stdout_three, artifacts_three) = run(&case_three, "pRef3");
    let (stdout_zero, artifacts_zero) = run(&case_zero, "pRef0");

    let evidence = stdout_three
        .lines()
        .find(|line| line.starts_with("incompressibleFluid solve:"))
        .expect("missing solve evidence");
    for expected in [
        "cells=4",
        "faces=20",
        "simpleIterations=2",
        "pRefCell=0",
        "pRefValue=3.000000e0",
        "nonOrthogonalCorrectors=1",
    ] {
        assert!(
            evidence.contains(expected),
            "missing solve evidence: {expected}"
        );
    }

    let report_three =
        fs::read_to_string(artifacts_three.join("report.json")).expect("read pRef3 solve report");
    for (key, value) in [
        ("cells", "4"),
        ("faces", "20"),
        ("pressureReferenceCell", "0"),
        ("pressureReferenceValue", "3"),
        ("nonOrthogonalCorrectors", "1"),
        ("velocityFixedValueFaces", "8"),
        ("velocityConstraintFaces", "8"),
        ("pressureFixedValueFaces", "0"),
        ("pressureZeroGradientFaces", "8"),
        ("pressureConstraintFaces", "8"),
        ("pressureCorrectionSolves", "4"),
        ("pressureCorrectionNonConvergedSolves", "0"),
    ] {
        assert!(
            json_has_exact_field(&report_three, key, value),
            "missing exact report evidence: {key}={value}"
        );
    }
    assert_eq!(
        report_three
            .lines()
            .filter(|line| line.trim() == "\"pressureCorrectionAccepted\": true,")
            .count(),
        2
    );
    assert_eq!(
        report_three
            .lines()
            .filter(|line| line.trim() == "\"adjustPhiAdjustedFaces\": 0,")
            .count(),
        2
    );

    let report_zero =
        fs::read_to_string(artifacts_zero.join("report.json")).expect("read pRef0 solve report");
    assert!(json_has_exact_field(
        &report_zero,
        "pressureReferenceValue",
        "0"
    ));

    for metric in [
        "finalContinuityL2",
        "phiMin",
        "phiMax",
        "phiSumAbs",
        "gradPL2",
        "hbyAL2",
        "divPhiUL2",
    ] {
        let value_three = numeric_evidence(&stdout_three, metric);
        let value_zero = numeric_evidence(&stdout_zero, metric);
        assert!(value_three.is_finite(), "pRef3 {metric} is finite");
        assert!(value_zero.is_finite(), "pRef0 {metric} is finite");
        assert_close(value_three, value_zero, 1.0e-12, metric);
    }

    let fields_three = artifacts_three.join("fields");
    let fields_zero = artifacts_zero.join("fields");
    let pressure_three = nonuniform_scalars(
        &fs::read_to_string(fields_three.join("p")).expect("read pRef3 pressure field"),
    );
    let pressure_zero = nonuniform_scalars(
        &fs::read_to_string(fields_zero.join("p")).expect("read pRef0 pressure field"),
    );
    assert_eq!(pressure_three.len(), 4);
    assert_eq!(pressure_zero.len(), 4);
    assert_close(pressure_three[0], 3.0, 1.0e-12, "pRef3 anchor");
    assert_close(pressure_zero[0], 0.0, 1.0e-12, "pRef0 anchor");
    for (index, (&three, &zero)) in pressure_three.iter().zip(&pressure_zero).enumerate() {
        assert!(three.is_finite());
        assert!(zero.is_finite());
        assert_close(
            three - zero,
            3.0,
            1.0e-12,
            &format!("pressure[{index}] gauge shift"),
        );
    }

    let velocity_three = nonuniform_vectors(
        &fs::read_to_string(fields_three.join("U")).expect("read pRef3 velocity field"),
    );
    let velocity_zero = nonuniform_vectors(
        &fs::read_to_string(fields_zero.join("U")).expect("read pRef0 velocity field"),
    );
    assert_eq!(velocity_three.len(), 4);
    assert_eq!(velocity_zero.len(), 4);
    for (cell, (three, zero)) in velocity_three.iter().zip(&velocity_zero).enumerate() {
        for component in 0..3 {
            assert!(three[component].is_finite());
            assert!(zero[component].is_finite());
            assert_close(
                three[component],
                zero[component],
                1.0e-13,
                &format!("velocity[{cell}][{component}] gauge invariance"),
            );
        }
    }
}
