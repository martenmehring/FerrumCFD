use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use ferrum_mesh::dictionary::tokenize;
use ferrum_mesh::fields::{FieldValueSummary, read_initial_fields};
use ferrum_mesh::geometry::compute_poly_mesh_geometry;
use ferrum_mesh::poly_mesh::PolyMesh;

const FULL: f64 = 1.0e-14;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn text(path: &Path) -> String {
    fs::read_to_string(path).unwrap_or_else(|error| panic!("read {}: {error}", path.display()))
}

fn toml_values(input: &str) -> BTreeMap<String, String> {
    let mut section = String::new();
    let mut values = BTreeMap::new();
    for raw in input.lines() {
        let line = raw.split('#').next().unwrap_or_default().trim();
        if line.starts_with('[') && line.ends_with(']') {
            section = line[1..line.len() - 1].to_owned();
        } else if let Some((key, value)) = line.split_once('=') {
            let key = key.trim();
            let name = if section.is_empty() {
                key.to_owned()
            } else {
                format!("{section}.{key}")
            };
            values.insert(name, value.trim().trim_matches('"').to_owned());
        }
    }
    values
}

fn value<'a>(values: &'a BTreeMap<String, String>, key: &str) -> &'a str {
    values.get(key).unwrap_or_else(|| panic!("missing {key}"))
}

fn number(values: &BTreeMap<String, String>, key: &str) -> f64 {
    value(values, key)
        .parse()
        .unwrap_or_else(|error| panic!("parse {key}: {error}"))
}

fn assert_close(label: &str, actual: f64, expected: f64, relative: f64, absolute: f64) {
    assert!(
        actual.is_finite() && expected.is_finite(),
        "{label} must be finite"
    );
    let error = (actual - expected).abs();
    let limit = absolute + relative * actual.abs().max(expected.abs());
    assert!(
        error <= limit,
        "{label}: {actual:.17e} != {expected:.17e} (error {error:.3e}, limit {limit:.3e})"
    );
}

fn foam_scalar(input: &str, key: &str) -> f64 {
    let tokens = tokenize(input);
    let position = tokens
        .iter()
        .position(|token| token.value == key)
        .unwrap_or_else(|| panic!("missing Foam key {key}"));
    tokens[position + 1..]
        .iter()
        .take_while(|token| token.value != ";")
        .filter_map(|token| token.value.parse().ok())
        .last()
        .unwrap_or_else(|| panic!("missing numeric value for Foam key {key}"))
}

fn foam_word<'a>(input: &'a str, key: &str) -> &'a str {
    let start = input.find(key).unwrap_or_else(|| panic!("missing {key}")) + key.len();
    input[start..]
        .trim_start()
        .split(|character: char| character.is_whitespace() || character == ';')
        .next()
        .unwrap_or_else(|| panic!("missing word for {key}"))
}

#[test]
fn laminar_pipe_physical_contract_and_inlet_profiles_do_not_drift() {
    let tutorial = root().join("tutorials/incompressibleFluid/laminarPipe");
    let shared = toml_values(&text(&tutorial.join("shared/physicalParameters.toml")));
    let comparison = toml_values(&text(&tutorial.join("comparison.toml")));

    assert_eq!(value(&shared, "schema_version"), "1");
    assert_eq!(value(&shared, "unit_system"), "SI");
    assert_eq!(value(&comparison, "schema_version"), "2");
    assert_eq!(
        value(&comparison, "physical_parameters"),
        "shared/physicalParameters.toml"
    );
    for key in ["id", "title", "regime"] {
        assert_eq!(
            value(&shared, &format!("case.{key}")),
            value(&comparison, if key == "id" { "case_id" } else { key })
        );
    }
    assert_eq!(value(&shared, "geometry.axis"), "x");
    assert_eq!(
        value(&shared, "flow.inlet_velocity_profile"),
        "parabolicFullyDeveloped"
    );
    assert_eq!(value(&shared, "pressure_loss.model"), "Hagen-Poiseuille");
    assert_eq!(value(&comparison, "reference.model"), "Hagen-Poiseuille");
    assert_eq!(value(&shared, "pressure_loss.minor_losses"), "false");
    assert!(!value(&shared, "provenance.description").trim().is_empty());
    assert!(!value(&shared, "provenance.source").trim().is_empty());

    let length = number(&shared, "geometry.length_m");
    let diameter = number(&shared, "geometry.diameter_m");
    let temperature = number(&shared, "fluid.reference_temperature_k");
    let rho = number(&shared, "fluid.density_kg_per_m3");
    let mu = number(&shared, "fluid.dynamic_viscosity_pa_s");
    let nu = number(&shared, "fluid.kinematic_viscosity_m2_per_s");
    let mean = number(&shared, "flow.mean_velocity_m_per_s");
    let delta_p = number(&shared, "pressure_loss.delta_p_pa");
    for (label, quantity) in [
        ("length", length),
        ("diameter", diameter),
        ("temperature", temperature),
        ("rho", rho),
        ("mu", mu),
        ("nu", nu),
        ("mean velocity", mean),
        ("pressure drop", delta_p),
    ] {
        assert!(
            quantity.is_finite() && quantity > 0.0,
            "{label} must be finite and positive"
        );
    }
    assert_close("exact nu", nu, mu / rho, FULL, 0.0);
    assert_close(
        "Hagen-Poiseuille pressure drop",
        delta_p,
        32.0 * mu * mean * length / diameter.powi(2),
        FULL,
        0.0,
    );

    let ferrum_properties = text(&tutorial.join("ferrum/case/constant/transportProperties"));
    assert_close(
        "Ferrum rho",
        foam_scalar(&ferrum_properties, "rho"),
        rho,
        FULL,
        0.0,
    );
    assert_close(
        "Ferrum mu",
        foam_scalar(&ferrum_properties, "mu"),
        mu,
        FULL,
        0.0,
    );
    assert_close(
        "rounded Ferrum nu",
        foam_scalar(&ferrum_properties, "nu"),
        nu,
        1.0e-5,
        0.0,
    );
    let openfoam_properties = text(&tutorial.join("openfoam-v13/case/constant/physicalProperties"));
    assert_close(
        "OpenFOAM nu",
        foam_scalar(&openfoam_properties, "nu"),
        nu,
        2.0e-9,
        0.0,
    );

    let geometry = text(&tutorial.join("shared/geometry/pipe_prism2.geo"));
    assert_close(
        "Gmsh radius",
        assignment(&geometry, "radius"),
        diameter / 2.0,
        FULL,
        0.0,
    );
    assert_close(
        "Gmsh length",
        assignment(&geometry, "length"),
        length,
        FULL,
        0.0,
    );

    let analytical = text(&tutorial.join("analytical/pipeBenchmark"));
    for (label, key, expected) in [
        ("analytical length", "length", length),
        ("analytical diameter", "diameter", diameter),
        (
            "analytical temperature",
            "referenceTemperature",
            temperature,
        ),
        ("analytical rho", "rho", rho),
        ("analytical mu", "mu", mu),
        ("analytical mean", "meanVelocity", mean),
        ("analytical pressure", "expectedDeltaP", delta_p),
    ] {
        assert_close(label, foam_scalar(&analytical, key), expected, FULL, 0.0);
    }
    assert_close(
        "provenance temperature",
        number(&shared, "provenance.reference_temperature_k"),
        foam_scalar(&analytical, "referenceTemperature"),
        FULL,
        0.0,
    );
    assert_eq!(
        foam_word(&analytical, "inletVelocityProfile"),
        value(&shared, "flow.inlet_velocity_profile")
    );
    assert_eq!(
        foam_word(&analytical, "pressureLossModel"),
        "HagenPoiseuille"
    );
    assert_eq!(foam_word(&analytical, "minorLosses"), "off");
    let scale = foam_scalar(&analytical, "inletVelocityScale");

    check_case_profile(&tutorial.join("ferrum/case"), mean, diameter / 2.0, scale);
    check_case_profile(
        &tutorial.join("openfoam-v13/case"),
        mean,
        diameter / 2.0,
        scale,
    );
}

fn assignment(input: &str, name: &str) -> f64 {
    let line = input
        .lines()
        .find(|line| line.trim_start().starts_with(&format!("{name} =")))
        .unwrap_or_else(|| panic!("missing Gmsh assignment {name}"));
    line.split_once('=')
        .unwrap()
        .1
        .split([',', ';'])
        .next()
        .unwrap()
        .trim_matches(|c: char| c == '{' || c.is_whitespace())
        .parse()
        .unwrap()
}

fn check_case_profile(case: &Path, mean: f64, radius: f64, scale: f64) {
    let fields = read_initial_fields(case)
        .unwrap_or_else(|error| panic!("read fields {}: {error}", case.display()));
    let velocity = fields
        .fields
        .iter()
        .find(|field| field.name == "U" && field.region.is_none())
        .expect("U field");
    let internal = match velocity.internal_field.as_ref().expect("U internalField") {
        FieldValueSummary::Uniform(value) => value,
        _ => panic!("U internalField must be uniform"),
    };
    let components: Vec<f64> = internal
        .trim_matches(|c| c == '(' || c == ')')
        .split_whitespace()
        .map(|part| part.parse().unwrap())
        .collect();
    assert_close("U internal x", components[0], mean, FULL, 0.0);
    assert_eq!(&components[1..], &[0.0, 0.0]);

    let inlet = velocity
        .boundary_patches
        .iter()
        .find(|patch| patch.name == "inlet")
        .expect("inlet field patch");
    assert_eq!(inlet.patch_type.as_deref(), Some("fixedValue"));
    let (declared, values) = match inlet.value.as_ref().expect("inlet value") {
        FieldValueSummary::NonUniform {
            value_type,
            count: Some(count),
            values: Some(values),
        } => {
            assert_eq!(value_type.as_deref(), Some("List<vector>"));
            (*count, values)
        }
        other => panic!("inlet must contain loaded nonuniform vectors, got {other}"),
    };
    let mesh = PolyMesh::read(&case.join("constant/polyMesh")).unwrap();
    let patch = mesh
        .patches
        .iter()
        .find(|patch| patch.name == "inlet")
        .expect("inlet mesh patch");
    assert_eq!(declared, patch.faces);
    assert_eq!(values.len(), patch.faces * 3);
    let geometry = compute_poly_mesh_geometry(&mesh).unwrap();
    let mut weighted_velocity = 0.0;
    let mut total_area = 0.0;
    for (local, vector) in values.chunks_exact(3).enumerate() {
        assert!(vector.iter().all(|component| component.is_finite()));
        assert!(vector[0] >= 0.0);
        assert_eq!(vector[1], 0.0);
        assert_eq!(vector[2], 0.0);
        let face = patch.start_face + local;
        let centre = geometry.face_centres[face];
        let area_vector = geometry.face_area_vectors[face];
        let area = (area_vector.x.powi(2) + area_vector.y.powi(2) + area_vector.z.powi(2)).sqrt();
        let radial_squared = centre.y.powi(2) + centre.z.powi(2);
        let expected = scale * 2.0 * mean * (1.0 - radial_squared / radius.powi(2));
        assert_close("inlet profile", vector[0], expected, 1.0e-8, 1.0e-12);
        weighted_velocity += vector[0] * area;
        total_area += area;
    }
    assert_close(
        "area-weighted inlet mean",
        weighted_velocity / total_area,
        mean,
        1.0e-9,
        0.0,
    );
}
