use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::time::{SystemTime, UNIX_EPOCH};

fn copy_tree(source: &Path, destination: &Path) {
    fs::create_dir_all(destination).expect("create temporary case directory");
    for entry in fs::read_dir(source).expect("read packaged cylinder case") {
        let entry = entry.expect("read cylinder case entry");
        let target = destination.join(entry.file_name());
        if entry.file_type().expect("read entry type").is_dir() {
            copy_tree(&entry.path(), &target);
        } else {
            fs::copy(entry.path(), target).expect("copy cylinder case file");
        }
    }
}

fn run_case(case: &Path, extra: &[&str]) -> Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_ferrumRun"));
    command
        .arg("-solver")
        .arg("incompressibleFluid")
        .arg("-case")
        .arg(case)
        .args(extra);
    command.output().expect("run ferrumRun cylinder case")
}

fn stdout(output: &Output) -> String {
    String::from_utf8(output.stdout.clone()).expect("ferrumRun stdout is UTF-8")
}

#[test]
fn packaged_cylinder_preflight_and_two_iteration_smoke() {
    let package = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let source = package.join("../../../tutorials/incompressibleFluid/cylinder/ferrum/case");
    assert!(source.join("constant/polyMesh/faces").is_file());

    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock after epoch")
        .as_nanos();
    let temporary =
        std::env::temp_dir().join(format!("ferrum-cylinder-{}-{nonce}", std::process::id()));
    copy_tree(&source, &temporary);

    let preflight = run_case(&source, &["--preflight"]);
    assert!(
        preflight.status.success(),
        "preflight failed: {}",
        String::from_utf8_lossy(&preflight.stderr)
    );
    let preflight_stdout = stdout(&preflight);
    assert!(preflight_stdout.contains("dispatch: module=incompressibleFluid"));
    assert!(preflight_stdout.contains("gradSchemes.grad(U)=$limited"));
    assert!(preflight_stdout.contains("divSchemes.div(phi,U)=bounded Gauss linearUpwind limited"));

    let solve = run_case(&temporary, &["--maxSimpleIterations", "2"]);
    let _ = fs::remove_dir_all(&temporary);
    assert!(
        solve.status.success(),
        "smoke solve failed: {}",
        String::from_utf8_lossy(&solve.stderr)
    );
    let solve_stdout = stdout(&solve);
    let evidence = solve_stdout
        .lines()
        .find(|line| line.starts_with("incompressibleFluid solve:"))
        .expect("missing solve evidence");
    assert!(evidence.contains("simpleIterations=2"));
    assert!(evidence.contains("divPhiU=\"bounded Gauss linearUpwind limited\""));
    assert!(evidence.contains("gradU=\"cellLimited Gauss linear 1\""));
    let continuity = evidence
        .split_whitespace()
        .find_map(|item| item.strip_prefix("finalContinuityL2="))
        .expect("missing final continuity evidence")
        .parse::<f64>()
        .expect("final continuity is numeric");
    assert!(continuity.is_finite());
}
