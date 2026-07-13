use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use ferrum_mesh::Point3;
use ferrum_mesh::fields::{FieldBoundaryPatch, FieldFile, FieldValueSummary, read_initial_fields};
use ferrum_mesh::geometry::compute_poly_mesh_geometry;
use ferrum_mesh::poly_mesh::{BoundaryPatch, PolyMesh};
use ferrum_mesh::properties::{PropertyDictionary, PropertyEntry, read_case_properties};

type CheckResult<T = ()> = Result<T, String>;

const L: f64 = 1.0;
const H: f64 = 0.02;
const W: f64 = 0.001;
const RHO: f64 = 998.2;
const MU: f64 = 0.001002;
const MEAN_U: f64 = 0.02;
const DELTA_P: f64 = 0.6012;
const NATIVE_NU: f64 = 1.0038e-6;
const EXACT_NU: f64 = 1.0038068513323983e-6;
const EPS: f64 = 1.0e-12;

#[derive(Debug)]
struct FlatToml {
    tables: BTreeMap<String, BTreeMap<String, String>>,
}

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn case(name: &str) -> PathBuf {
    root()
        .join("tutorials/incompressibleFluid/planeChannel")
        .join(name)
        .join("case")
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

fn parse_flat_toml(text: &str) -> CheckResult<FlatToml> {
    let mut tables = BTreeMap::from([("root".to_string(), BTreeMap::new())]);
    let mut current = "root".to_string();
    let mut seen_tables = BTreeSet::from(["root".to_string()]);
    for (number, raw) in text.lines().enumerate() {
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

fn validate_metadata(comparison: &str, physical: &str) -> CheckResult {
    let comparison = parse_flat_toml(comparison)?;
    let table_names = comparison
        .tables
        .keys()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    let expected_tables = BTreeSet::from([
        "root",
        "implementations",
        "physics",
        "reference",
        "comparison",
    ]);
    if table_names != expected_tables {
        return Err("comparison table set mismatch".into());
    }
    exact_entries(
        &comparison.tables["root"],
        &[
            ("schema_version", "2"),
            ("case_id", "\"incompressibleFluid.planeChannel\""),
            ("module", "\"incompressibleFluid\""),
            ("readiness_driver", "\"steadyIncompressible\""),
            ("algorithm", "\"SIMPLE\""),
            ("regime", "\"laminar\""),
            ("title", "\"Laminar plane-channel flow\""),
            ("physical_parameters", "\"shared/physicalParameters.toml\""),
        ],
    )?;
    exact_entries(
        &comparison.tables["implementations"],
        &[
            ("ferrum_case", "\"ferrum/case\""),
            ("openfoam_v13_case", "\"openfoam-v13/case\""),
            ("reference", "\"analytical/planeChannelBenchmark\""),
        ],
    )?;
    exact_entries(&comparison.tables["physics"], &[("unit_system", "\"SI\"")])?;
    exact_entries(
        &comparison.tables["reference"],
        &[
            ("kind", "\"analytical\""),
            ("model", "\"Plane-Poiseuille\""),
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
            ("id", "\"incompressibleFluid.planeChannel\""),
            ("title", "\"Laminar plane-channel flow\""),
            ("regime", "\"laminar\""),
            (
                "provenance",
                "\"Plane-Poiseuille: deltaP = 12*mu*L*meanU/H^2\"",
            ),
            ("length_m", "1.0"),
            ("gap_m", "0.02"),
            ("depth_m", "0.001"),
            ("streamwise_axis", "\"x\""),
            ("wall_normal_axis", "\"y\""),
            ("density_kg_per_m3", "998.2"),
            ("dynamic_viscosity_pa_s", "0.001002"),
            ("kinematic_viscosity_m2_per_s", "1.0038068513323983e-6"),
            ("mean_velocity_m_per_s", "0.02"),
            ("pressure_drop_pa", "0.6012"),
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
        "gap_m",
        "depth_m",
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
        12.0 * MU * L * MEAN_U / H.powi(2),
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
    key: &str,
    dimensions: [i32; 7],
    expected: f64,
    tolerance: f64,
) -> CheckResult {
    if entry.key != key || entry.value.len() != 10 || entry.value[0] != "[" || entry.value[8] != "]"
    {
        return Err(format!("malformed dimensioned entry {key}"));
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
        return Err(format!("wrong dimensions for {key}"));
    }
    let value = entry.value[9]
        .parse::<f64>()
        .map_err(|_| format!("bad value for {key}"))?;
    if !value.is_finite() || (value - expected).abs() > tolerance {
        return Err(format!("wrong value for {key}"));
    }
    Ok(())
}

fn validate_properties(case_name: &str, dictionaries: &[PropertyDictionary]) -> CheckResult {
    let mut by_name = BTreeMap::new();
    for dictionary in dictionaries {
        if by_name
            .insert(dictionary.name.as_str(), dictionary)
            .is_some()
        {
            return Err("duplicate property dictionary".into());
        }
    }
    if case_name == "ferrum" {
        if by_name.keys().copied().collect::<BTreeSet<_>>()
            != BTreeSet::from(["transportProperties"])
        {
            return Err("Ferrum property dictionary set mismatch".into());
        }
        let entries = entry_map(by_name["transportProperties"])?;
        if entries.keys().copied().collect::<BTreeSet<_>>()
            != BTreeSet::from(["transportModel", "rho", "mu", "nu"])
            || entries["transportModel"].value != ["Newtonian"]
        {
            return Err("Ferrum property semantics mismatch".into());
        }
        assert_dimensioned(entries["rho"], "rho", [1, -3, 0, 0, 0, 0, 0], RHO, 0.0)?;
        assert_dimensioned(entries["mu"], "mu", [1, -1, -1, 0, 0, 0, 0], MU, 0.0)?;
        assert_dimensioned(
            entries["nu"],
            "nu",
            [0, 2, -1, 0, 0, 0, 0],
            NATIVE_NU,
            1.0e-15,
        )?;
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
            != BTreeSet::from(["viscosityModel", "nu"])
            || physical["viscosityModel"].value != ["constant"]
        {
            return Err("physical properties mismatch".into());
        }
        assert_dimensioned(
            physical["nu"],
            "nu",
            [0, 2, -1, 0, 0, 0, 0],
            EXACT_NU,
            1.0e-18,
        )?;
    }
    Ok(())
}

fn validate_foam_header(text: &str, object: &str) -> CheckResult {
    let start = text
        .find("FoamFile")
        .ok_or_else(|| "missing FoamFile".to_string())?;
    let open = text[start..]
        .find('{')
        .ok_or_else(|| "missing header open".to_string())?
        + start;
    let close = text[open..]
        .find('}')
        .ok_or_else(|| "missing header close".to_string())?
        + open;
    let mut entries = BTreeMap::new();
    for statement in text[open + 1..close].split(';') {
        let tokens = statement.split_whitespace().collect::<Vec<_>>();
        if tokens.is_empty() {
            continue;
        }
        if tokens.len() != 2
            || entries
                .insert(tokens[0], tokens[1].trim_matches('"'))
                .is_some()
        {
            return Err("malformed or duplicate FoamFile entry".into());
        }
    }
    let expected = BTreeMap::from([
        ("version", "2.0"),
        ("format", "ascii"),
        ("class", "dictionary"),
        ("location", "constant"),
        ("object", object),
    ]);
    if entries != expected {
        return Err("FoamFile header mismatch".into());
    }
    Ok(())
}

fn uniform(value: &Option<FieldValueSummary>) -> CheckResult<Vec<f64>> {
    let Some(FieldValueSummary::Uniform(value)) = value else {
        return Err("expected uniform field value".into());
    };
    value
        .trim_matches(|ch: char| ch == '(' || ch == ')' || ch.is_whitespace())
        .split_whitespace()
        .map(|token| {
            token
                .parse::<f64>()
                .map_err(|_| "invalid field scalar".to_string())
        })
        .collect()
}

fn patches(field: &FieldFile) -> CheckResult<BTreeMap<&str, &FieldBoundaryPatch>> {
    if field.boundary_patches.len() != 5 {
        return Err("field must have five boundary patches".into());
    }
    let mut patches = BTreeMap::new();
    for patch in &field.boundary_patches {
        if patches.insert(patch.name.as_str(), patch).is_some() {
            return Err("duplicate field patch".into());
        }
    }
    if patches.keys().copied().collect::<BTreeSet<_>>()
        != BTreeSet::from(["inlet", "outlet", "wall", "front", "back"])
    {
        return Err("field patch set mismatch".into());
    }
    Ok(patches)
}

fn validate_u(field: &FieldFile) -> CheckResult {
    let dimensions = field
        .dimensions
        .as_ref()
        .map(|values| values.iter().map(String::as_str).collect::<Vec<_>>());
    if field.name != "U"
        || field.class_name.as_deref() != Some("volVectorField")
        || field.region.is_some()
        || dimensions.as_deref() != Some(&["0", "1", "-1", "0", "0", "0", "0"])
    {
        return Err("U identity mismatch".into());
    }
    let values = uniform(&field.internal_field)?;
    if values.len() != 3 || values.iter().any(|value| !value.is_finite()) {
        return Err("U internal arity or finiteness mismatch".into());
    }
    close(values[0], MEAN_U, 0.0)?;
    close(values[1], 0.0, 0.0)?;
    close(values[2], 0.0, 0.0)?;
    let patches = patches(field)?;
    for (name, kind) in [
        ("inlet", "zeroGradient"),
        ("outlet", "zeroGradient"),
        ("wall", "noSlip"),
        ("front", "empty"),
        ("back", "empty"),
    ] {
        let patch = patches[name];
        if patch.patch_type.as_deref() != Some(kind)
            || patch.value.is_some()
            || patch.inlet_value.is_some()
        {
            return Err(format!("U patch {name} mismatch"));
        }
    }
    Ok(())
}

fn validate_p(field: &FieldFile, openfoam: bool) -> CheckResult {
    let dimensions: &[&str] = if openfoam {
        &["0", "2", "-2", "0", "0", "0", "0"]
    } else {
        &["1", "-1", "-2", "0", "0", "0", "0"]
    };
    let actual_dimensions = field
        .dimensions
        .as_ref()
        .map(|values| values.iter().map(String::as_str).collect::<Vec<_>>());
    if field.name != "p"
        || field.class_name.as_deref() != Some("volScalarField")
        || field.region.is_some()
        || actual_dimensions.as_deref() != Some(dimensions)
    {
        return Err("p identity mismatch".into());
    }
    let values = uniform(&field.internal_field)?;
    if values.len() != 1 || !values[0].is_finite() {
        return Err("p internal scalar arity or finiteness mismatch".into());
    }
    let scale = if openfoam { RHO } else { 1.0 };
    close(values[0] * scale, DELTA_P / 2.0, EPS)?;
    let patches = patches(field)?;
    for (name, kind, expected) in [
        ("inlet", "fixedValue", Some(DELTA_P)),
        ("outlet", "fixedValue", Some(0.0)),
        ("wall", "zeroGradient", None),
        ("front", "empty", None),
        ("back", "empty", None),
    ] {
        let patch = patches[name];
        if patch.patch_type.as_deref() != Some(kind) || patch.inlet_value.is_some() {
            return Err(format!("p patch {name} type/presence mismatch"));
        }
        match expected {
            Some(expected) => {
                let values = uniform(&patch.value)?;
                if values.len() != 1 || !values[0].is_finite() {
                    return Err(format!("p patch {name} scalar arity mismatch"));
                }
                close(values[0] * scale, expected, EPS)?;
            }
            None if patch.value.is_none() => {}
            None => return Err(format!("unexpected p value on {name}")),
        }
    }
    Ok(())
}

fn validate_fields(fields: &[FieldFile], openfoam: bool) -> CheckResult {
    if fields.len() != 2 || fields.iter().any(|field| field.region.is_some()) {
        return Err("expected exactly two root-region fields".into());
    }
    let mut by_name = BTreeMap::new();
    for field in fields {
        if by_name.insert(field.name.as_str(), field).is_some() {
            return Err("duplicate field name".into());
        }
    }
    if by_name.keys().copied().collect::<BTreeSet<_>>() != BTreeSet::from(["U", "p"]) {
        return Err("field name set mismatch".into());
    }
    validate_u(by_name["U"])?;
    validate_p(by_name["p"], openfoam)
}

fn raw_normal(mesh: &PolyMesh, face_index: usize) -> CheckResult<Point3> {
    let face = mesh
        .faces
        .get(face_index)
        .ok_or_else(|| "missing face".to_string())?;
    if face.len() < 3 {
        return Err("face has fewer than three vertices".into());
    }
    let origin = mesh.points[face[0]];
    let mut normal = Point3 {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };
    for index in 1..face.len() - 1 {
        let a = mesh.points[face[index]];
        let b = mesh.points[face[index + 1]];
        normal.x +=
            ((a.y - origin.y) * (b.z - origin.z) - (a.z - origin.z) * (b.y - origin.y)) / 2.0;
        normal.y +=
            ((a.z - origin.z) * (b.x - origin.x) - (a.x - origin.x) * (b.z - origin.z)) / 2.0;
        normal.z +=
            ((a.x - origin.x) * (b.y - origin.y) - (a.y - origin.y) * (b.x - origin.x)) / 2.0;
    }
    Ok(normal)
}

fn validate_mesh(mesh: &PolyMesh) -> CheckResult {
    mesh.validate().map_err(|error| error.to_string())?;
    if mesh.patches.len() != 5 {
        return Err("mesh must contain five patches".into());
    }
    let names = mesh
        .patches
        .iter()
        .map(|patch| patch.name.as_str())
        .collect::<BTreeSet<_>>();
    if names.len() != 5 || names != BTreeSet::from(["inlet", "outlet", "wall", "front", "back"]) {
        return Err("mesh patch name set mismatch".into());
    }
    let mut by_name = BTreeMap::new();
    for patch in &mesh.patches {
        by_name.insert(patch.name.as_str(), patch);
    }
    for (name, kind) in [
        ("inlet", "patch"),
        ("outlet", "patch"),
        ("wall", "wall"),
        ("front", "empty"),
        ("back", "empty"),
    ] {
        if by_name[name].patch_type != kind || by_name[name].faces == 0 {
            return Err(format!("mesh patch {name} type/range mismatch"));
        }
    }
    let mut ranges = mesh.patches.iter().collect::<Vec<_>>();
    ranges.sort_by_key(|patch| patch.start_face);
    let mut next = mesh.neighbour.len();
    for patch in ranges {
        if patch.start_face != next {
            return Err("boundary ranges contain a gap or overlap".into());
        }
        next = next
            .checked_add(patch.faces)
            .ok_or_else(|| "patch range overflow".to_string())?;
    }
    if next != mesh.faces.len() {
        return Err("boundary ranges leave unclaimed faces".into());
    }
    let geometry = compute_poly_mesh_geometry(mesh).map_err(|error| error.to_string())?;
    if geometry
        .cell_volumes
        .iter()
        .any(|volume| !volume.is_finite() || *volume <= 0.0)
    {
        return Err("cell volumes must be finite and positive".into());
    }
    let mut positive_wall = false;
    let mut negative_wall = false;
    for patch in &mesh.patches {
        for face_index in patch.start_face..patch.start_face + patch.faces {
            let centre = geometry.face_centres[face_index];
            let normal = geometry.face_area_vectors[face_index];
            let raw = raw_normal(mesh, face_index)?;
            match patch.name.as_str() {
                "inlet" if centre.x.abs() <= EPS && normal.x < 0.0 => {}
                "outlet" if (centre.x - L).abs() <= EPS && normal.x > 0.0 => {}
                "wall" if (centre.y - H / 2.0).abs() <= EPS && normal.y > 0.0 => {
                    positive_wall = true
                }
                "wall" if (centre.y + H / 2.0).abs() <= EPS && normal.y < 0.0 => {
                    negative_wall = true
                }
                "front" if (centre.z + W / 2.0).abs() <= EPS && normal.z < 0.0 && raw.z < 0.0 => {}
                "back" if (centre.z - W / 2.0).abs() <= EPS && normal.z > 0.0 && raw.z > 0.0 => {}
                _ => return Err(format!("patch {} plane or normal mismatch", patch.name)),
            }
        }
    }
    if !positive_wall || !negative_wall {
        return Err("both wall planes must be present".into());
    }
    Ok(())
}

fn fresh_mesh() -> PolyMesh {
    PolyMesh::read(&case("ferrum").join("constant/polyMesh")).expect("read fresh Ferrum mesh")
}

fn close(actual: f64, expected: f64, tolerance: f64) -> CheckResult {
    if !actual.is_finite() || (actual - expected).abs() > tolerance {
        Err(format!("{actual} differs from {expected}"))
    } else {
        Ok(())
    }
}

#[test]
fn plane_channel_positive_contract() {
    let directory = root().join("tutorials/incompressibleFluid/planeChannel");
    let comparison = fs::read_to_string(directory.join("comparison.toml")).unwrap();
    let physical = fs::read_to_string(directory.join("shared/physicalParameters.toml")).unwrap();
    validate_metadata(&comparison, &physical).unwrap();
    for (name, openfoam) in [("ferrum", false), ("openfoam-v13", true)] {
        let case = case(name);
        let properties = read_case_properties(&case).unwrap();
        validate_properties(
            if openfoam { "openfoam-v13" } else { "ferrum" },
            &properties,
        )
        .unwrap();
        let fields = read_initial_fields(&case).unwrap();
        validate_fields(&fields.fields, openfoam).unwrap();
        validate_mesh(&PolyMesh::read(&case.join("constant/polyMesh")).unwrap()).unwrap();
    }
    for object in ["momentumTransport", "physicalProperties"] {
        let text = fs::read_to_string(case("openfoam-v13").join("constant").join(object)).unwrap();
        validate_foam_header(&text, object).unwrap();
        assert!(
            validate_foam_header(
                &text.replacen("class dictionary;", "class scalar;", 1),
                object
            )
            .is_err()
        );
        assert!(
            validate_foam_header(
                &text.replacen("format ascii;", "format ascii; extra bad;", 1),
                object
            )
            .is_err()
        );
        assert!(
            validate_foam_header(
                &text.replacen(&format!("object {object};"), "object substituted;", 1),
                object
            )
            .is_err()
        );
    }
}

#[test]
fn metadata_mutation_probes_are_rejected() {
    let directory = root().join("tutorials/incompressibleFluid/planeChannel");
    let comparison = fs::read_to_string(directory.join("comparison.toml")).unwrap();
    let physical = fs::read_to_string(directory.join("shared/physicalParameters.toml")).unwrap();
    assert!(validate_metadata(&format!("{comparison}\n[extra]\nkey = 1\n"), &physical).is_err());
    assert!(
        validate_metadata(
            &comparison.replace(
                "[reference]",
                "[reference]\npressure_drop_pa = 0.6012\n[new]"
            ),
            &physical
        )
        .is_err()
    );
    assert!(
        validate_metadata(
            &comparison.replace("model = \"Plane-Poiseuille\"", "model = \"other\""),
            &physical
        )
        .is_err()
    );
    assert!(parse_flat_toml("title = \"# retained\" # ignored\n").is_ok());
    for bad in [
        "key value",
        "key = 1\nkey = 2",
        "x = 1\n[root]\nx = 2",
        "[a]\nx=1\n[a]\ny=2",
        "[broken",
        "x = \"unterminated",
    ] {
        assert!(parse_flat_toml(bad).is_err(), "accepted {bad:?}");
    }
}

#[test]
fn wrong_reference_model_is_rejected() {
    let directory = root().join("tutorials/incompressibleFluid/planeChannel");
    let comparison = fs::read_to_string(directory.join("comparison.toml")).unwrap();
    let physical = fs::read_to_string(directory.join("shared/physicalParameters.toml")).unwrap();
    let wrong = comparison.replace(
        "model = \"Plane-Poiseuille\"",
        "model = \"Hagen-Poiseuille\"",
    );
    assert!(validate_metadata(&wrong, &physical).is_err());
}

#[test]
fn wrong_property_dimensions_are_rejected() {
    let mut entry = PropertyEntry {
        key: "nu".into(),
        value: ["[", "0", "2", "0", "0", "0", "0", "0", "]", "1.0038e-6"]
            .map(str::to_string)
            .to_vec(),
    };
    assert!(assert_dimensioned(&entry, "nu", [0, 2, -1, 0, 0, 0, 0], NATIVE_NU, EPS).is_err());
    entry.value = ["[", "0", "2", "-1", "0", "0", "0", "0", "]", "NaN"]
        .map(str::to_string)
        .to_vec();
    assert!(assert_dimensioned(&entry, "nu", [0, 2, -1, 0, 0, 0, 0], NATIVE_NU, EPS).is_err());
    entry.value[9] = "2e-6".into();
    assert!(assert_dimensioned(&entry, "nu", [0, 2, -1, 0, 0, 0, 0], NATIVE_NU, EPS).is_err());
}

#[test]
fn field_mutation_probes_are_rejected() {
    let load = || read_initial_fields(&case("ferrum")).unwrap();
    let mut fields = load();
    let p = fields
        .fields
        .iter_mut()
        .find(|field| field.name == "p")
        .unwrap();
    p.internal_field = Some(FieldValueSummary::Uniform("(0.3006 9)".into()));
    assert!(validate_fields(&fields.fields, false).is_err());
    let mut fields = load();
    fields.fields[0].name = "bad".into();
    assert!(validate_fields(&fields.fields, false).is_err());
    let mut fields = load();
    fields.fields[0].class_name = Some("wrongClass".into());
    assert!(validate_fields(&fields.fields, false).is_err());
    let mut fields = load();
    fields.fields[0].region = Some("fluid".into());
    assert!(validate_fields(&fields.fields, false).is_err());
    let mut fields = load();
    let u = fields
        .fields
        .iter_mut()
        .find(|field| field.name == "U")
        .unwrap();
    u.internal_field = Some(FieldValueSummary::Uniform("(0.02 0)".into()));
    assert!(validate_fields(&fields.fields, false).is_err());
    let mut fields = load();
    let p = fields
        .fields
        .iter_mut()
        .find(|field| field.name == "p")
        .unwrap();
    p.internal_field = Some(FieldValueSummary::Uniform("0.4".into()));
    assert!(validate_fields(&fields.fields, false).is_err());
    let mut fields = load();
    let u = fields
        .fields
        .iter_mut()
        .find(|field| field.name == "U")
        .unwrap();
    u.boundary_patches[0].patch_type = Some("fixedValue".into());
    assert!(validate_fields(&fields.fields, false).is_err());
}

#[test]
fn reversed_front_winding_is_rejected() {
    let mut mesh = fresh_mesh();
    let front_face = mesh
        .patches
        .iter()
        .find(|patch| patch.name == "front")
        .unwrap()
        .start_face;
    mesh.faces[front_face].reverse();
    let error = validate_mesh(&mesh).unwrap_err();
    assert!(
        error.contains("plane or normal"),
        "unexpected error: {error}"
    );
}

#[test]
fn mesh_mutation_probes_are_rejected() {
    let mut mesh = fresh_mesh();
    let inlet = mesh
        .patches
        .iter()
        .find(|patch| patch.name == "inlet")
        .unwrap();
    mesh.patches.push(BoundaryPatch {
        name: "inlet".into(),
        patch_type: inlet.patch_type.clone(),
        faces: 1,
        start_face: inlet.start_face,
    });
    assert!(validate_mesh(&mesh).is_err());

    let mut mesh = fresh_mesh();
    mesh.patches[0].start_face += 1;
    assert!(validate_mesh(&mesh).is_err());

    let mut mesh = fresh_mesh();
    let wall = mesh
        .patches
        .iter_mut()
        .find(|patch| patch.name == "wall")
        .unwrap();
    wall.patch_type = "patch".into();
    assert!(validate_mesh(&mesh).is_err());

    let mut mesh = fresh_mesh();
    let front = mesh
        .patches
        .iter_mut()
        .find(|patch| patch.name == "front")
        .unwrap();
    front.patch_type = "wall".into();
    assert!(validate_mesh(&mesh).is_err());

    let mut mesh = fresh_mesh();
    let inlet_face = mesh
        .patches
        .iter()
        .find(|patch| patch.name == "inlet")
        .unwrap()
        .start_face;
    let point = mesh.faces[inlet_face][0];
    mesh.points[point].x += 1.0e-4;
    assert!(validate_mesh(&mesh).is_err());
}
