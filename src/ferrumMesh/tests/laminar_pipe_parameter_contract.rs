use std::fs;
use std::path::{Path, PathBuf};

fn tutorial_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tutorials/incompressibleFluid/laminarPipe")
}

fn read(relative: &str) -> String {
    let path = tutorial_root().join(relative);
    fs::read_to_string(&path).unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()))
}

fn toml_number(text: &str, section: Option<&str>, key: &str) -> f64 {
    let mut current = None;
    for raw in text.lines() {
        let line = raw.split('#').next().unwrap_or("").trim();
        if line.starts_with('[') && line.ends_with(']') {
            current = Some(&line[1..line.len() - 1]);
            continue;
        }
        if current == section {
            if let Some((candidate, value)) = line.split_once('=') {
                if candidate.trim() == key {
                    return value.trim().parse().unwrap_or_else(|_| panic!("{key} must be a finite number"));
                }
            }
        }
    }
    panic!("missing parameter {key}");
}

fn scalar_after(text: &str, marker: &str) -> f64 {
    let tail = text.find(marker).map(|at| &text[at + marker.len()..])
        .unwrap_or_else(|| panic!("missing scalar marker {marker}"));
    tail.split_whitespace().next().unwrap_or("").trim_end_matches([';', ',']).parse()
        .unwrap_or_else(|_| panic!("invalid scalar after {marker}"))
}

fn initial_velocity(text: &str) -> f64 {
    let field = text.find("internalField").map(|at| &text[at..])
        .unwrap_or_else(|| panic!("missing internalField"));
    scalar_after(field, "(")
}

fn assert_close(label: &str, actual: f64, expected: f64, relative_tolerance: f64) {
    assert!(actual.is_finite(), "{label} is not finite");
    let scale = expected.abs().max(1.0e-30);
    assert!((actual - expected).abs() <= relative_tolerance * scale,
        "{label} drifted: found {actual:.17e}, expected {expected:.17e}");
}

#[test]
fn bundled_laminar_pipe_parameters_match_the_shared_contract() {
    let contract = read("shared/physicalParameters.toml");
    let length = toml_number(&contract, Some("geometry"), "length_m");
    let diameter = toml_number(&contract, Some("geometry"), "diameter_m");
    let rho = toml_number(&contract, Some("material"), "density_kg_per_m3");
    let mu = toml_number(&contract, Some("material"), "dynamic_viscosity_pa_s");
    let nu = toml_number(&contract, Some("material"), "kinematic_viscosity_m2_per_s");
    let velocity = toml_number(&contract, Some("flow_reference"), "mean_velocity_m_per_s");
    let pressure_drop = toml_number(&contract, Some("analytical_reference"), "pressure_drop_pa");

    for (name, value) in [("length", length), ("diameter", diameter), ("density", rho),
        ("dynamic viscosity", mu), ("kinematic viscosity", nu), ("mean velocity", velocity),
        ("pressure drop", pressure_drop)] {
        assert!(value.is_finite() && value > 0.0, "{name} must be finite and positive");
    }
    assert_close("contract nu = mu/rho", nu, mu / rho, 1.0e-14);
    assert_close("contract Hagen-Poiseuille pressure drop", pressure_drop,
        32.0 * mu * velocity * length / diameter.powi(2), 1.0e-14);

    let ferrum = read("ferrum/case/constant/transportProperties");
    assert_close("Ferrum density", scalar_after(&ferrum, "rho [1 -3 0 0 0 0 0]"), rho, 1.0e-14);
    assert_close("Ferrum dynamic viscosity", scalar_after(&ferrum, "mu [1 -1 -1 0 0 0 0]"), mu, 1.0e-14);
    let ferrum_nu = scalar_after(&ferrum, "nu [0 2 -1 0 0 0 0]");
    assert_close("Ferrum kinematic viscosity", ferrum_nu, nu, 1.0e-5);

    let openfoam = read("openfoam-v13/case/constant/physicalProperties");
    // The bundled runtime dictionaries retain their established decimal precision.
    assert_close("OpenFOAM kinematic viscosity", scalar_after(&openfoam, "nu [0 2 -1 0 0 0 0]"), nu, 2.0e-9);

    let geometry = read("shared/geometry/pipe_prism2.geo");
    assert_close("Gmsh radius", scalar_after(&geometry, "radius = {"), diameter / 2.0, 1.0e-14);
    assert_close("Gmsh length", scalar_after(&geometry, "length = {"), length, 1.0e-14);

    for velocity_file in ["ferrum/case/0/U", "openfoam-v13/case/0/U"] {
        let field = read(velocity_file);
        assert_close(velocity_file, initial_velocity(&field), velocity, 1.0e-14);
    }

    let analytical = read("analytical/pipeBenchmark");
    assert_close("analytical length", scalar_after(&analytical, "length [0 1 0 0 0 0 0]"), length, 1.0e-14);
    assert_close("analytical diameter", scalar_after(&analytical, "diameter [0 1 0 0 0 0 0]"), diameter, 1.0e-14);
    assert_close("analytical density", scalar_after(&analytical, "rho [1 -3 0 0 0 0 0]"), rho, 1.0e-14);
    assert_close("analytical dynamic viscosity", scalar_after(&analytical, "mu [1 -1 -1 0 0 0 0]"), mu, 1.0e-14);
    assert_close("analytical mean velocity", scalar_after(&analytical, "meanVelocity [0 1 -1 0 0 0 0]"), velocity, 1.0e-14);
    assert_close("analytical pressure drop", scalar_after(&analytical, "expectedDeltaP [1 -1 -2 0 0 0 0]"), pressure_drop, 1.0e-14);
}

#[test]
fn comparison_references_metadata_without_becoming_a_runtime_dependency() {
    let comparison = read("comparison.toml");
    assert!(comparison.contains("physical_parameters = \"shared/physicalParameters.toml\""));
    for duplicate in ["length_m", "diameter_m", "density_kg_per_m3", "dynamic_viscosity_pa_s",
        "mean_velocity_m_per_s", "pressure_drop_pa"] {
        assert!(!comparison.contains(duplicate), "comparison.toml duplicates authoritative {duplicate}");
    }
    for runtime_file in ["ferrum/case/system/controlDict", "openfoam-v13/case/system/controlDict",
        "ferrum/case/constant/transportProperties", "openfoam-v13/case/constant/physicalProperties"] {
        let contents = read(runtime_file);
        assert!(!contents.contains("physicalParameters.toml") && !contents.contains("comparison.toml"),
            "{runtime_file} must remain independent of comparison metadata");
    }
}
