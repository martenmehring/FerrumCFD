use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use ferrum_mesh::dictionary::tokenize;
use ferrum_mesh::fields::{FieldFile, FieldValueSummary, read_initial_fields};
use ferrum_mesh::geometry::compute_poly_mesh_geometry;
use ferrum_mesh::poly_mesh::PolyMesh;
use ferrum_mesh::properties::{PropertyDictionary, PropertyEntry, read_case_properties};

type CheckResult<T = ()> = Result<T, String>;

const LENGTH: f64 = 1.0;
const DIAMETER: f64 = 0.02;
const RHO: f64 = 998.2;
const MU: f64 = 0.001002;
const EXACT_NU: f64 = 1.0038068513323983e-6;
const FERRUM_NU: f64 = 1.0038e-6;
const MEAN_U: f64 = 0.02;
const EPS: f64 = 1.0e-12;

#[derive(Debug)]
struct FlatToml {
    tables: BTreeMap<String, BTreeMap<String, String>>,
}

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn tutorial() -> PathBuf {
    root().join("tutorials/incompressibleFluid/laminarPipe")
}

fn case(name: &str) -> PathBuf {
    tutorial().join(name).join("case")
}

fn text(path: &Path) -> String {
    fs::read_to_string(path).unwrap_or_else(|error| panic!("read {}: {error}", path.display()))
}

fn strip_comment(line: &str) -> CheckResult<&str> {
    let mut quoted = false;
    let mut escaped = false;
    for (index, ch) in line.char_indices() {
        if escaped {
            escaped = false;
        } else if quoted && ch == '\\' {
            escaped = true;
        } else if ch == '"' {
            quoted = !quoted;
        } else if ch == '#' && !quoted {
            return Ok(&line[..index]);
        }
    }
    if quoted || escaped {
        return Err("unterminated quoted string".into());
    }
    Ok(line)
}

fn assignment(line: &str) -> CheckResult<(&str, &str)> {
    let mut quoted = false;
    let mut escaped = false;
    let mut equals = None;
    for (index, ch) in line.char_indices() {
        if escaped {
            escaped = false;
        } else if quoted && ch == '\\' {
            escaped = true;
        } else if ch == '"' {
            quoted = !quoted;
        } else if ch == '=' && !quoted && equals.replace(index).is_some() {
            return Err("multiple assignment operators".into());
        }
    }
    let index = equals.ok_or_else(|| "missing assignment operator".to_string())?;
    let key = line[..index].trim();
    let value = line[index + 1..].trim();
    if key.is_empty()
        || value.is_empty()
        || !key
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
    {
        return Err("malformed assignment".into());
    }
    Ok((key, value))
}

fn parse_flat_toml(input: &str) -> CheckResult<FlatToml> {
    let mut tables = BTreeMap::from([("root".to_string(), BTreeMap::new())]);
    let mut current = "root".to_string();
    let mut seen_tables = BTreeSet::from(["root".to_string()]);
    for (number, raw) in input.lines().enumerate() {
        let line = strip_comment(raw)
            .map_err(|error| format!("line {}: {error}", number + 1))?
            .trim();
        if line.is_empty() {
            continue;
        }
        if line.starts_with('[') {
            if !line.ends_with(']') || line.starts_with("[[") {
                return Err(format!("line {}: malformed table header", number + 1));
            }
            let table = line[1..line.len() - 1].trim();
            if table.is_empty()
                || !table
                    .chars()
                    .all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
                || !seen_tables.insert(table.to_string())
            {
                return Err(format!("line {}: invalid or repeated table", number + 1));
            }
            current = table.to_string();
            tables.insert(current.clone(), BTreeMap::new());
            continue;
        }
        let (key, value) =
            assignment(line).map_err(|error| format!("line {}: {error}", number + 1))?;
        if tables
            .get_mut(&current)
            .expect("current table exists")
            .insert(key.to_string(), value.to_string())
            .is_some()
        {
            return Err(format!("line {}: duplicate assignment", number + 1));
        }
    }
    Ok(FlatToml { tables })
}

fn exact_entries(entries: &BTreeMap<String, String>, expected: &[(&str, &str)]) -> CheckResult {
    let keys = entries.keys().map(String::as_str).collect::<BTreeSet<_>>();
    let expected_keys = expected
        .iter()
        .map(|(key, _)| *key)
        .collect::<BTreeSet<_>>();
    if keys != expected_keys {
        return Err(format!("key set mismatch: {keys:?} != {expected_keys:?}"));
    }
    for (key, value) in expected {
        if entries.get(*key).map(String::as_str) != Some(*value) {
            return Err(format!("unexpected value for {key}"));
        }
    }
    Ok(())
}

fn close(actual: f64, expected: f64, tolerance: f64) -> CheckResult {
    if !actual.is_finite() || (actual - expected).abs() > tolerance {
        Err(format!("{actual} differs from {expected}"))
    } else {
        Ok(())
    }
}

fn validate_metadata(comparison: &str, physical: &str) -> CheckResult {
    let comparison = parse_flat_toml(comparison)?;
    if comparison
        .tables
        .keys()
        .map(String::as_str)
        .collect::<BTreeSet<_>>()
        != BTreeSet::from([
            "root",
            "implementations",
            "physics",
            "reference",
            "comparison",
        ])
    {
        return Err("comparison table set mismatch".into());
    }
    exact_entries(
        &comparison.tables["root"],
        &[
            ("schema_version", "2"),
            ("case_id", "\"incompressibleFluid.laminarPipe\""),
            ("module", "\"incompressibleFluid\""),
            ("readiness_driver", "\"steadyIncompressible\""),
            ("algorithm", "\"SIMPLE\""),
            ("regime", "\"laminar\""),
            ("title", "\"Laminar circular-pipe flow\""),
            ("physical_parameters", "\"shared/physicalParameters.toml\""),
        ],
    )?;
    exact_entries(
        &comparison.tables["implementations"],
        &[
            ("ferrum_case", "\"ferrum/case\""),
            ("openfoam_v13_case", "\"openfoam-v13/case\""),
            ("reference", "\"analytical/pipeBenchmark\""),
        ],
    )?;
    exact_entries(&comparison.tables["physics"], &[("unit_system", "\"SI\"")])?;
    exact_entries(
        &comparison.tables["reference"],
        &[
            ("kind", "\"analytical\""),
            ("model", "\"Hagen-Poiseuille\""),
        ],
    )?;
    exact_entries(
        &comparison.tables["comparison"],
        &[
            ("sample_pressure_on", "[\"inlet\", \"outlet\"]"),
            (
                "compare",
                "[\"pressureDrop\", \"meanVelocity\", \"flowRate\", \"velocityProfile\"]",
            ),
        ],
    )?;

    let physical = parse_flat_toml(physical)?;
    if physical
        .tables
        .keys()
        .map(String::as_str)
        .collect::<BTreeSet<_>>()
        != BTreeSet::from(["root"])
    {
        return Err("physical metadata must contain only root scalars".into());
    }
    exact_entries(
        &physical.tables["root"],
        &[
            ("schema_version", "1"),
            ("id", "\"incompressibleFluid.laminarPipe\""),
            ("title", "\"Laminar circular-pipe flow\""),
            ("regime", "\"laminar\""),
            (
                "provenance",
                "\"Hagen-Poiseuille: deltaP = 32*mu*L*meanU/D^2\"",
            ),
            ("length_m", "1.0"),
            ("diameter_m", "0.02"),
            ("axis", "\"x\""),
            ("reference_temperature_k", "293.15"),
            ("density_kg_per_m3", "998.2"),
            ("dynamic_viscosity_pa_s", "0.001002"),
            ("kinematic_viscosity_m2_per_s", "1.0038068513323983e-6"),
            ("mean_velocity_m_per_s", "0.02"),
            ("inlet_velocity_profile", "\"parabolicFullyDeveloped\""),
            ("pressure_loss_model", "\"Hagen-Poiseuille\""),
            ("pressure_drop_pa", "1.6032"),
            ("minor_losses", "false"),
        ],
    )?;
    let values = &physical.tables["root"];
    let number = |key: &str| -> CheckResult<f64> {
        values[key]
            .parse::<f64>()
            .map_err(|_| format!("{key} is not numeric"))
    };
    for key in [
        "length_m",
        "diameter_m",
        "reference_temperature_k",
        "density_kg_per_m3",
        "dynamic_viscosity_pa_s",
        "kinematic_viscosity_m2_per_s",
        "mean_velocity_m_per_s",
        "pressure_drop_pa",
    ] {
        let value = number(key)?;
        if !value.is_finite() || value <= 0.0 {
            return Err(format!("{key} must be finite and positive"));
        }
    }
    close(number("kinematic_viscosity_m2_per_s")?, MU / RHO, EPS)?;
    close(
        number("pressure_drop_pa")?,
        32.0 * MU * MEAN_U * LENGTH / DIAMETER.powi(2),
        EPS,
    )
}

fn entry_map(dictionary: &PropertyDictionary) -> CheckResult<BTreeMap<&str, &PropertyEntry>> {
    if dictionary.region.is_some() || !dictionary.sections.is_empty() {
        return Err("unexpected property region or section".into());
    }
    let mut entries = BTreeMap::new();
    for entry in &dictionary.entries {
        if entries.insert(entry.key.as_str(), entry).is_some() {
            return Err(format!("duplicate property key {}", entry.key));
        }
    }
    Ok(entries)
}

fn assert_dimensioned(
    entry: &PropertyEntry,
    dimensions: [i32; 7],
    expected: f64,
    tolerance: f64,
) -> CheckResult {
    if entry.value.len() != 10 || entry.value[0] != "[" || entry.value[8] != "]" {
        return Err(format!("malformed dimensioned entry {}", entry.key));
    }
    let parsed = entry.value[1..8]
        .iter()
        .map(|value| {
            value
                .parse::<i32>()
                .map_err(|_| "bad dimension".to_string())
        })
        .collect::<CheckResult<Vec<_>>>()?;
    if parsed.as_slice() != dimensions {
        return Err(format!("wrong dimensions for {}", entry.key));
    }
    let value = entry.value[9]
        .parse::<f64>()
        .map_err(|_| format!("bad value for {}", entry.key))?;
    close(value, expected, tolerance)
}

fn validate_properties(name: &str, dictionaries: &[PropertyDictionary]) -> CheckResult {
    let mut by_name = BTreeMap::new();
    for dictionary in dictionaries {
        if by_name
            .insert(dictionary.name.as_str(), dictionary)
            .is_some()
        {
            return Err("duplicate property dictionary".into());
        }
    }
    if name == "ferrum" {
        if by_name.keys().copied().collect::<BTreeSet<_>>()
            != BTreeSet::from(["transportProperties"])
        {
            return Err("Ferrum property dictionary set mismatch".into());
        }
        let entries = entry_map(by_name["transportProperties"])?;
        if entries.keys().copied().collect::<BTreeSet<_>>()
            != BTreeSet::from(["Cp", "Pr", "k", "mu", "nu", "rho", "transportModel"])
            || entries["transportModel"].value != ["Newtonian"]
        {
            return Err("Ferrum property semantics mismatch".into());
        }
        assert_dimensioned(entries["rho"], [1, -3, 0, 0, 0, 0, 0], RHO, EPS)?;
        assert_dimensioned(entries["mu"], [1, -1, -1, 0, 0, 0, 0], MU, EPS)?;
        assert_dimensioned(entries["nu"], [0, 2, -1, 0, 0, 0, 0], FERRUM_NU, EPS)?;
        assert_dimensioned(entries["Cp"], [0, 2, -2, -1, 0, 0, 0], 4182.0, EPS)?;
        assert_dimensioned(entries["k"], [1, 1, -3, -1, 0, 0, 0], 0.598, EPS)?;
        assert_dimensioned(entries["Pr"], [0, 0, 0, 0, 0, 0, 0], 7.01, EPS)
    } else {
        if by_name.keys().copied().collect::<BTreeSet<_>>()
            != BTreeSet::from(["momentumTransport", "physicalProperties"])
        {
            return Err("OpenFOAM property dictionary set mismatch".into());
        }
        let momentum = entry_map(by_name["momentumTransport"])?;
        if momentum.keys().copied().collect::<BTreeSet<_>>() != BTreeSet::from(["simulationType"])
            || momentum["simulationType"].value != ["laminar"]
        {
            return Err("momentum transport mismatch".into());
        }
        let physical = entry_map(by_name["physicalProperties"])?;
        if physical.keys().copied().collect::<BTreeSet<_>>()
            != BTreeSet::from(["nu", "viscosityModel"])
            || physical["viscosityModel"].value != ["constant"]
        {
            return Err("physical properties mismatch".into());
        }
        assert_dimensioned(physical["nu"], [0, 2, -1, 0, 0, 0, 0], EXACT_NU, EPS)
    }
}

fn field<'a>(fields: &'a [FieldFile], name: &str) -> CheckResult<&'a FieldFile> {
    let matching = fields
        .iter()
        .filter(|candidate| candidate.name == name && candidate.region.is_none())
        .collect::<Vec<_>>();
    if matching.len() != 1 {
        return Err(format!("expected exactly one root-region {name} field"));
    }
    Ok(matching[0])
}

fn dimensions(field: &FieldFile) -> CheckResult<Vec<&str>> {
    field
        .dimensions
        .as_ref()
        .map(|items| items.iter().map(String::as_str).collect())
        .ok_or_else(|| format!("{} has no dimensions", field.name))
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

fn validate_case(name: &str, openfoam: bool) -> CheckResult {
    let case = case(name);
    validate_properties(
        name,
        &read_case_properties(&case).map_err(|error| error.to_string())?,
    )?;
    let mesh =
        PolyMesh::read(&case.join("constant/polyMesh")).map_err(|error| error.to_string())?;
    if mesh.points.len() != 4_825
        || mesh.cell_count() != 4_608
        || mesh.faces.len() != 14_016
        || mesh.owner.len() != 14_016
        || mesh.neighbour.len() != 12_864
    {
        return Err("mesh count contract mismatch".into());
    }
    let patches = mesh
        .patches
        .iter()
        .map(|patch| {
            (
                patch.name.as_str(),
                (patch.patch_type.as_str(), patch.faces, patch.start_face),
            )
        })
        .collect::<BTreeMap<_, _>>();
    if patches
        != BTreeMap::from([
            ("inlet", ("patch", 192, 12_864)),
            ("outlet", ("patch", 192, 13_056)),
            ("wall", ("wall", 768, 13_248)),
        ])
    {
        return Err("mesh patch contract mismatch".into());
    }
    let geometry = compute_poly_mesh_geometry(&mesh).map_err(|error| error.to_string())?;
    if geometry
        .cell_volumes
        .iter()
        .any(|volume| !volume.is_finite() || *volume <= 0.0)
    {
        return Err("cell volumes must be finite and positive".into());
    }

    let loaded = read_initial_fields(&case).map_err(|error| error.to_string())?;
    let velocity = field(&loaded.fields, "U")?;
    let pressure = field(&loaded.fields, "p")?;
    if velocity.class_name.as_deref() != Some("volVectorField")
        || dimensions(velocity)? != ["0", "1", "-1", "0", "0", "0", "0"]
        || pressure.class_name.as_deref() != Some("volScalarField")
    {
        return Err("field identity mismatch".into());
    }
    let expected_p = if openfoam {
        ["0", "2", "-2", "0", "0", "0", "0"]
    } else {
        ["1", "-1", "-2", "0", "0", "0", "0"]
    };
    if dimensions(pressure)? != expected_p {
        return Err("pressure dimensions mismatch".into());
    }
    let FieldValueSummary::Uniform(internal) = velocity
        .internal_field
        .as_ref()
        .ok_or_else(|| "missing U internal field".to_string())?
    else {
        return Err("U internal field must be uniform".into());
    };
    let components = internal
        .trim_matches(|ch: char| ch == '(' || ch == ')' || ch.is_whitespace())
        .split_whitespace()
        .map(|part| {
            part.parse::<f64>()
                .map_err(|_| "bad U component".to_string())
        })
        .collect::<CheckResult<Vec<_>>>()?;
    if components.len() != 3 {
        return Err("U internal field arity mismatch".into());
    }
    close(components[0], MEAN_U, EPS)?;
    close(components[1], 0.0, 0.0)?;
    close(components[2], 0.0, 0.0)?;

    let inlet = velocity
        .boundary_patches
        .iter()
        .find(|patch| patch.name == "inlet")
        .ok_or_else(|| "missing U inlet".to_string())?;
    if inlet.patch_type.as_deref() != Some("fixedValue") {
        return Err("U inlet must be fixedValue".into());
    }
    let (declared, values) = match inlet.value.as_ref() {
        Some(FieldValueSummary::NonUniform {
            value_type,
            count: Some(count),
            values: Some(values),
        }) if value_type.as_deref() == Some("List<vector>") => (*count, values),
        _ => return Err("U inlet must contain loaded nonuniform vectors".into()),
    };
    if declared != 192 || values.len() != 576 {
        return Err("U inlet vector count mismatch".into());
    }
    let scale = foam_scalar(
        &text(&tutorial().join("analytical/pipeBenchmark")),
        "inletVelocityScale",
    );
    let inlet_patch = mesh
        .patches
        .iter()
        .find(|patch| patch.name == "inlet")
        .expect("validated inlet patch");
    let mut weighted_velocity = 0.0;
    let mut total_area = 0.0;
    for (local, vector) in values.chunks_exact(3).enumerate() {
        if vector.iter().any(|component| !component.is_finite())
            || vector[0] < 0.0
            || vector[1] != 0.0
            || vector[2] != 0.0
        {
            return Err("invalid inlet velocity vector".into());
        }
        let face = inlet_patch.start_face + local;
        let centre = geometry.face_centres[face];
        let area_vector = geometry.face_area_vectors[face];
        let area = (area_vector.x.powi(2) + area_vector.y.powi(2) + area_vector.z.powi(2)).sqrt();
        let radial_squared = centre.y.powi(2) + centre.z.powi(2);
        let expected = scale * 2.0 * MEAN_U * (1.0 - radial_squared / (DIAMETER / 2.0).powi(2));
        close(vector[0], expected, 1.0e-10)?;
        weighted_velocity += vector[0] * area;
        total_area += area;
    }
    close(weighted_velocity / total_area, MEAN_U, 1.0e-10)
}

#[test]
fn laminar_pipe_metadata_is_single_source_and_physically_consistent() {
    let comparison = text(&tutorial().join("comparison.toml"));
    let physical = text(&tutorial().join("shared/physicalParameters.toml"));
    validate_metadata(&comparison, &physical).unwrap();
}

#[test]
fn laminar_pipe_cases_preserve_the_shared_contract() {
    validate_case("ferrum", false).unwrap();
    validate_case("openfoam-v13", true).unwrap();
}

#[test]
fn laminar_pipe_contract_rejects_duplicate_wrong_and_nonfinite_metadata() {
    let comparison = text(&tutorial().join("comparison.toml"));
    let physical = text(&tutorial().join("shared/physicalParameters.toml"));
    assert!(validate_metadata(&format!("{comparison}\nschema_version = 2\n"), &physical).is_err());
    assert!(
        validate_metadata(
            &comparison.replace("model = \"Hagen-Poiseuille\"", "model = \"other\""),
            &physical,
        )
        .is_err()
    );
    assert!(
        validate_metadata(
            &comparison,
            &physical.replace("pressure_drop_pa = 1.6032", "pressure_drop_pa = NaN"),
        )
        .is_err()
    );
    assert!(validate_metadata(&comparison, &format!("{physical}\n[extra]\nkey = 1\n"),).is_err());
    for bad in [
        "key value",
        "key = 1\nkey = 2",
        "[a]\nx=1\n[a]\ny=2",
        "[broken",
        "x = \"unterminated",
    ] {
        assert!(parse_flat_toml(bad).is_err(), "accepted {bad:?}");
    }
}

#[test]
fn laminar_pipe_property_dimension_mutations_are_rejected() {
    let mut entry = PropertyEntry {
        key: "nu".into(),
        value: ["[", "0", "2", "0", "0", "0", "0", "0", "]", "1.0038e-6"]
            .map(str::to_string)
            .to_vec(),
    };
    assert!(assert_dimensioned(&entry, [0, 2, -1, 0, 0, 0, 0], FERRUM_NU, EPS).is_err());
    entry.value = ["[", "0", "2", "-1", "0", "0", "0", "0", "]", "NaN"]
        .map(str::to_string)
        .to_vec();
    assert!(assert_dimensioned(&entry, [0, 2, -1, 0, 0, 0, 0], FERRUM_NU, EPS).is_err());
}
