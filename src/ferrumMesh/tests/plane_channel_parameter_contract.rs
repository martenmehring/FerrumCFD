use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use ferrum_mesh::fields::{FieldBoundaryPatch, FieldFile, FieldValueSummary, read_initial_fields};
use ferrum_mesh::geometry::compute_poly_mesh_geometry;
use ferrum_mesh::poly_mesh::{BoundaryPatch, PolyMesh};
use ferrum_mesh::properties::{PropertyDictionary, PropertyEntry, read_case_properties};
use ferrum_mesh::{MeshError, Point3, Result};

const ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../..");
const BASE: &str = "tutorials/incompressibleFluid/planeChannel";
const L: f64 = 1.0;
const H: f64 = 0.02;
const W: f64 = 0.001;
const RHO: f64 = 998.2;
const MU: f64 = 0.001002;
const MEAN_U: f64 = 0.02;
const DELTA_P: f64 = 0.6012;

fn invalid(message: impl Into<String>) -> MeshError {
    MeshError::InvalidInput(message.into())
}
fn case(relative: &str) -> PathBuf {
    Path::new(ROOT).join(BASE).join(relative).join("case")
}
fn close(a: f64, b: f64, tolerance: f64) -> bool {
    (a - b).abs() <= tolerance
}

/// A deliberately small strict TOML reader for comparison metadata. Comments
/// begin only outside quoted strings; reopening tables and duplicate keys fail.
fn parse_flat_toml(text: &str) -> Result<BTreeMap<String, BTreeMap<String, String>>> {
    let mut result = BTreeMap::new();
    result.insert("root".into(), BTreeMap::new());
    let mut table = "root".to_string();
    for (number, source) in text.lines().enumerate() {
        let mut quoted = false;
        let mut escaped = false;
        let mut clean = String::new();
        for ch in source.chars() {
            if ch == '#' && !quoted {
                break;
            }
            clean.push(ch);
            if ch == '"' && !escaped {
                quoted = !quoted;
            }
            escaped = ch == '\\' && !escaped;
            if ch != '\\' {
                escaped = false;
            }
        }
        if quoted {
            return Err(invalid(format!(
                "unterminated quote on line {}",
                number + 1
            )));
        }
        let line = clean.trim();
        if line.is_empty() {
            continue;
        }
        if line.starts_with('[') {
            if !line.ends_with(']')
                || line.matches('[').count() != 1
                || line.matches(']').count() != 1
            {
                return Err(invalid("malformed table header"));
            }
            table = line[1..line.len() - 1].trim().to_string();
            if table.is_empty() || result.insert(table.clone(), BTreeMap::new()).is_some() {
                return Err(invalid("repeated or empty table"));
            }
            continue;
        }
        let Some((key, value)) = line.split_once('=') else {
            return Err(invalid("malformed assignment"));
        };
        let (key, value) = (key.trim(), value.trim());
        if key.is_empty() || value.is_empty() || value.matches('"').count() % 2 != 0 {
            return Err(invalid("malformed assignment"));
        }
        if result
            .get_mut(&table)
            .unwrap()
            .insert(key.into(), value.into())
            .is_some()
        {
            return Err(invalid("duplicate assignment"));
        }
    }
    Ok(result)
}

fn exact_entries(map: &BTreeMap<String, String>, expected: &[&str]) -> Result<()> {
    let actual: BTreeSet<_> = map.keys().map(String::as_str).collect();
    let expected: BTreeSet<_> = expected.iter().copied().collect();
    if actual != expected {
        return Err(invalid(format!("key set mismatch: {actual:?}")));
    }
    Ok(())
}

fn unquote(value: &str) -> &str {
    value
        .strip_prefix('"')
        .and_then(|v| v.strip_suffix('"'))
        .unwrap_or(value)
}
fn number(map: &BTreeMap<String, String>, key: &str) -> Result<f64> {
    map.get(key)
        .ok_or_else(|| invalid(format!("missing {key}")))?
        .parse()
        .map_err(|_| invalid(format!("invalid {key}")))
}

fn validate_metadata(text: &str) -> Result<()> {
    let tables = parse_flat_toml(text)?;
    let names: BTreeSet<_> = tables.keys().map(String::as_str).collect();
    if names
        != BTreeSet::from([
            "root",
            "implementations",
            "physics",
            "reference",
            "comparison",
        ])
    {
        return Err(invalid("comparison table-name set mismatch"));
    }
    let root = &tables["root"];
    exact_entries(
        root,
        &[
            "schema_version",
            "case_id",
            "module",
            "readiness_driver",
            "algorithm",
            "regime",
            "title",
            "physical_parameters",
        ],
    )?;
    let expected = [
        ("schema_version", "2"),
        ("case_id", "incompressibleFluid.planeChannel"),
        ("module", "incompressibleFluid"),
        ("readiness_driver", "steadyIncompressible"),
        ("algorithm", "SIMPLE"),
        ("regime", "laminar"),
        ("title", "Laminar plane-channel flow"),
        ("physical_parameters", "shared/physicalParameters.toml"),
    ];
    for (key, value) in expected {
        if unquote(&root[key]) != value {
            return Err(invalid(format!("wrong {key}")));
        }
    }
    exact_entries(
        &tables["implementations"],
        &["ferrum_case", "openfoam_v13_case", "reference"],
    )?;
    for (key, expected) in [
        ("ferrum_case", "ferrum/case"),
        ("openfoam_v13_case", "openfoam-v13/case"),
        ("reference", "analytical/planeChannelBenchmark"),
    ] {
        if unquote(&tables["implementations"][key]) != expected {
            return Err(invalid(format!("wrong implementation path {key}")));
        }
    }
    exact_entries(
        &tables["physics"],
        &[
            "unit_system",
            "length_m",
            "gap_m",
            "depth_m",
            "density_kg_per_m3",
            "dynamic_viscosity_pa_s",
            "mean_velocity_m_per_s",
        ],
    )?;
    exact_entries(&tables["reference"], &["kind", "model", "pressure_drop_pa"])?;
    exact_entries(&tables["comparison"], &["sample_pressure_on", "compare"])?;
    let physics = &tables["physics"];
    if unquote(&physics["unit_system"]) != "SI"
        || number(physics, "length_m")? != L
        || number(physics, "gap_m")? != H
        || number(physics, "depth_m")? != W
        || number(physics, "density_kg_per_m3")? != RHO
        || number(physics, "dynamic_viscosity_pa_s")? != MU
        || number(physics, "mean_velocity_m_per_s")? != MEAN_U
        || number(&tables["reference"], "pressure_drop_pa")? != DELTA_P
    {
        return Err(invalid("comparison physical constants mismatch"));
    }
    if unquote(&tables["reference"]["kind"]) != "analytical"
        || unquote(&tables["reference"]["model"]) != "Plane-Poiseuille"
    {
        return Err(invalid("wrong reference model"));
    }
    if tables["comparison"]["sample_pressure_on"] != "[\"inlet\", \"outlet\"]"
        || tables["comparison"]["compare"]
            != "[\"pressureDrop\", \"meanVelocity\", \"flowRate\", \"velocityProfile\"]"
    {
        return Err(invalid("wrong comparison ordering"));
    }
    Ok(())
}

fn validate_shared_metadata(text: &str) -> Result<()> {
    let parsed = parse_flat_toml(text)?;
    if parsed.keys().map(String::as_str).collect::<BTreeSet<_>>() != BTreeSet::from(["root"]) {
        return Err(invalid("shared metadata contains tables"));
    }
    let root = &parsed["root"];
    exact_entries(
        root,
        &[
            "schema_version",
            "case_id",
            "title",
            "regime",
            "provenance",
            "length_m",
            "gap_m",
            "depth_m",
            "streamwise_axis",
            "wall_normal_axis",
            "density_kg_per_m3",
            "dynamic_viscosity_pa_s",
            "kinematic_viscosity_m2_per_s",
            "mean_velocity_m_per_s",
            "pressure_drop_pa",
            "pressure_drop_formula",
        ],
    )?;
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
        let v = number(root, key)?;
        if !v.is_finite() || v <= 0.0 {
            return Err(invalid("non-positive physical constant"));
        }
    }
    if number(root, "schema_version")? != 1.0
        || unquote(&root["case_id"]) != "incompressibleFluid.planeChannel"
        || unquote(&root["title"]) != "Laminar plane-channel flow"
        || unquote(&root["regime"]) != "laminar"
        || unquote(&root["provenance"])
            != "Canonical SI inputs for the Ferrum/OpenFOAM/analytical comparison"
        || unquote(&root["streamwise_axis"]) != "x"
        || unquote(&root["wall_normal_axis"]) != "y"
    {
        return Err(invalid("shared identity mismatch"));
    }
    if number(root, "length_m")? != L
        || number(root, "gap_m")? != H
        || number(root, "depth_m")? != W
        || number(root, "density_kg_per_m3")? != RHO
        || number(root, "dynamic_viscosity_pa_s")? != MU
        || number(root, "mean_velocity_m_per_s")? != MEAN_U
        || number(root, "pressure_drop_pa")? != DELTA_P
    {
        return Err(invalid("shared constant mismatch"));
    }
    if !close(
        number(root, "kinematic_viscosity_m2_per_s")?,
        MU / RHO,
        1e-18,
    ) || !close(12.0 * MU * L * MEAN_U / (H * H), DELTA_P, 1e-14)
        || unquote(&root["pressure_drop_formula"]) != "12*mu*L*meanU/H^2"
    {
        return Err(invalid("physical formula mismatch"));
    }
    Ok(())
}

fn assert_dimensioned(
    entry: &PropertyEntry,
    dimensions: [i32; 7],
    value: f64,
    tolerance: f64,
) -> Result<()> {
    if entry.value.len() != 10 || entry.value[0] != "[" || entry.value[8] != "]" {
        return Err(invalid("malformed dimensioned value"));
    }
    for (token, expected) in entry.value[1..8].iter().zip(dimensions) {
        if token.parse::<i32>().ok() != Some(expected) {
            return Err(invalid("wrong dimensions"));
        }
    }
    let actual: f64 = entry.value[9]
        .parse()
        .map_err(|_| invalid("invalid property value"))?;
    if !actual.is_finite() || !close(actual, value, tolerance) {
        return Err(invalid("wrong property value"));
    }
    Ok(())
}

fn validate_properties(case_name: &str, dictionaries: &[PropertyDictionary]) -> Result<()> {
    let expected_name = if case_name == "ferrum" {
        "transportProperties"
    } else {
        "physicalProperties"
    };
    let dictionary = dictionaries
        .iter()
        .find(|d| d.name == expected_name)
        .ok_or_else(|| invalid("missing property dictionary"))?;
    if dictionary.region.is_some() || !dictionary.sections.is_empty() {
        return Err(invalid("unexpected property structure"));
    }
    let keys: BTreeSet<_> = dictionary.entries.iter().map(|e| e.key.as_str()).collect();
    if case_name == "ferrum" {
        if dictionary.entries.len() != 4
            || keys != BTreeSet::from(["transportModel", "rho", "mu", "nu"])
        {
            return Err(invalid("Ferrum property keys"));
        }
        if dictionary
            .entries
            .iter()
            .find(|e| e.key == "transportModel")
            .unwrap()
            .value
            != ["Newtonian"]
        {
            return Err(invalid("wrong transport model"));
        }
        assert_dimensioned(
            dictionary.entries.iter().find(|e| e.key == "rho").unwrap(),
            [1, -3, 0, 0, 0, 0, 0],
            RHO,
            0.0,
        )?;
        assert_dimensioned(
            dictionary.entries.iter().find(|e| e.key == "mu").unwrap(),
            [1, -1, -1, 0, 0, 0, 0],
            MU,
            0.0,
        )?;
    } else if dictionary.entries.len() != 2
        || keys != BTreeSet::from(["viscosityModel", "nu"])
        || dictionary
            .entries
            .iter()
            .find(|e| e.key == "viscosityModel")
            .unwrap()
            .value
            != ["constant"]
    {
        return Err(invalid("OpenFOAM property keys/model"));
    }
    assert_dimensioned(
        dictionary.entries.iter().find(|e| e.key == "nu").unwrap(),
        [0, 2, -1, 0, 0, 0, 0],
        MU / RHO,
        1e-11,
    )?;
    Ok(())
}

fn uniform(value: &Option<FieldValueSummary>) -> Result<Vec<f64>> {
    let FieldValueSummary::Uniform(text) = value
        .as_ref()
        .ok_or_else(|| invalid("missing uniform value"))?
    else {
        return Err(invalid("not uniform"));
    };
    text.trim_matches(|c| c == '(' || c == ')')
        .split_whitespace()
        .map(|v| v.parse().map_err(|_| invalid("invalid field scalar")))
        .collect()
}
fn validate_u(field: &FieldFile) -> Result<()> {
    if field.name != "U"
        || field.region.is_some()
        || field.class_name.as_deref() != Some("volVectorField")
        || field.dimensions.as_deref()
            != Some(&["0", "1", "-1", "0", "0", "0", "0"].map(str::to_string))
    {
        return Err(invalid("U header"));
    }
    let values = uniform(&field.internal_field)?;
    if values.len() != 3 || values.iter().any(|v| !v.is_finite()) || values != [MEAN_U, 0.0, 0.0] {
        return Err(invalid("U internal arity/value"));
    }
    validate_boundaries(field, false)
}
fn validate_p(field: &FieldFile, openfoam: bool) -> Result<()> {
    let dims = if openfoam {
        ["0", "2", "-2", "0", "0", "0", "0"]
    } else {
        ["1", "-1", "-2", "0", "0", "0", "0"]
    };
    if field.name != "p"
        || field.region.is_some()
        || field.class_name.as_deref() != Some("volScalarField")
        || field.dimensions.as_deref() != Some(&dims.map(str::to_string))
    {
        return Err(invalid("p header"));
    }
    let values = uniform(&field.internal_field)?;
    if values.len() != 1 || !values[0].is_finite() {
        return Err(invalid("p internal scalar arity"));
    }
    validate_boundaries(field, true)?;
    let scale = if openfoam { RHO } else { 1.0 };
    if !close(values[0] * scale, DELTA_P / 2.0, 1e-9) {
        return Err(invalid("p internal value"));
    }
    let expected = [("inlet", DELTA_P), ("outlet", 0.0)];
    for (name, want) in expected {
        let patch = field
            .boundary_patches
            .iter()
            .find(|p| p.name == name)
            .unwrap();
        let v = uniform(&patch.value)?;
        if v.len() != 1 || !v[0].is_finite() || !close(v[0] * scale, want, 1e-9) {
            return Err(invalid("p patch scalar/value"));
        }
    }
    Ok(())
}
fn validate_boundaries(field: &FieldFile, pressure: bool) -> Result<()> {
    if field.boundary_patches.len() != 5 {
        return Err(invalid("boundary count"));
    }
    let expected = [
        (
            "inlet",
            if pressure {
                "fixedValue"
            } else {
                "zeroGradient"
            },
        ),
        (
            "outlet",
            if pressure {
                "fixedValue"
            } else {
                "zeroGradient"
            },
        ),
        ("wall", if pressure { "zeroGradient" } else { "noSlip" }),
        ("front", "empty"),
        ("back", "empty"),
    ];
    for (name, kind) in expected {
        let p = field
            .boundary_patches
            .iter()
            .find(|p| p.name == name)
            .ok_or_else(|| invalid("missing field patch"))?;
        if p.patch_type.as_deref() != Some(kind)
            || p.inlet_value.is_some()
            || (!pressure && p.value.is_some())
            || (pressure && matches!(name, "inlet" | "outlet") != p.value.is_some())
        {
            return Err(invalid("boundary drift"));
        }
    }
    Ok(())
}
fn validate_fields(case_name: &str, fields: &[FieldFile]) -> Result<()> {
    if fields.len() != 2
        || fields.iter().any(|f| f.region.is_some())
        || fields
            .iter()
            .map(|f| f.name.as_str())
            .collect::<BTreeSet<_>>()
            != BTreeSet::from(["U", "p"])
    {
        return Err(invalid("field set"));
    }
    validate_u(fields.iter().find(|f| f.name == "U").unwrap())?;
    validate_p(
        fields.iter().find(|f| f.name == "p").unwrap(),
        case_name == "openfoam-v13",
    )
}

fn raw_normal(mesh: &PolyMesh, face: usize) -> Result<Point3> {
    let ids = &mesh.faces[face];
    if ids.len() < 3 {
        return Err(invalid("short face"));
    }
    let mut n = Point3 {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };
    for i in 0..ids.len() {
        let a = mesh.points[ids[i]];
        let b = mesh.points[ids[(i + 1) % ids.len()]];
        n.x += (a.y - b.y) * (a.z + b.z);
        n.y += (a.z - b.z) * (a.x + b.x);
        n.z += (a.x - b.x) * (a.y + b.y);
    }
    Ok(n)
}
fn validate_mesh(mesh: &PolyMesh) -> Result<()> {
    mesh.validate()?;
    if mesh.patches.len() != 5 {
        return Err(invalid("patch count"));
    }
    let names: BTreeSet<_> = mesh.patches.iter().map(|p| p.name.as_str()).collect();
    if names.len() != 5 || names != BTreeSet::from(["inlet", "outlet", "wall", "front", "back"]) {
        return Err(invalid("patch names"));
    }
    let geometry = compute_poly_mesh_geometry(mesh)?;
    if geometry
        .cell_volumes
        .iter()
        .any(|v| !v.is_finite() || *v <= 0.0)
    {
        return Err(invalid("cell volume"));
    }
    let mut next = mesh.neighbour.len();
    let mut positive_wall = false;
    let mut negative_wall = false;
    for patch in &mesh.patches {
        if patch.start_face != next || patch.faces == 0 {
            return Err(invalid("patch range gap"));
        }
        next += patch.faces;
        let kind = match patch.name.as_str() {
            "front" | "back" => "empty",
            "wall" => "wall",
            _ => "patch",
        };
        if patch.patch_type != kind {
            return Err(invalid("patch type"));
        }
        for face in patch.start_face..next {
            let c = geometry.face_centres[face];
            let n = geometry.face_area_vectors[face];
            let raw = raw_normal(mesh, face)?;
            let ok = match patch.name.as_str() {
                "inlet" => close(c.x, 0.0, 1e-12) && n.x < 0.0,
                "outlet" => close(c.x, L, 1e-12) && n.x > 0.0,
                "wall" => {
                    if close(c.y, H / 2.0, 1e-12) {
                        positive_wall = true;
                        n.y > 0.0
                    } else if close(c.y, -H / 2.0, 1e-12) {
                        negative_wall = true;
                        n.y < 0.0
                    } else {
                        false
                    }
                }
                "front" => close(c.z, -W / 2.0, 1e-12) && n.z < 0.0 && raw.z < 0.0,
                "back" => close(c.z, W / 2.0, 1e-12) && n.z > 0.0 && raw.z > 0.0,
                _ => false,
            };
            if !ok {
                return Err(invalid(format!(
                    "boundary plane/orientation: {}",
                    patch.name
                )));
            }
        }
    }
    if next != mesh.faces.len() || !positive_wall || !negative_wall {
        return Err(invalid("coverage/wall planes"));
    }
    Ok(())
}
fn validate_foam_header(text: &str, object: &str) -> Result<()> {
    let start = text
        .find("FoamFile")
        .ok_or_else(|| invalid("missing FoamFile"))?;
    let open = text[start..].find('{').ok_or_else(|| invalid("header"))? + start;
    let close = text[open..].find('}').ok_or_else(|| invalid("header"))? + open;
    let body = &text[open + 1..close];
    let mut entries = BTreeMap::new();
    for statement in body.split(';').map(str::trim).filter(|s| !s.is_empty()) {
        let mut parts = statement.split_whitespace();
        let key = parts.next().unwrap();
        let value = parts.next().ok_or_else(|| invalid("header statement"))?;
        if parts.next().is_some() || entries.insert(key, value.trim_matches('"')).is_some() {
            return Err(invalid("header duplicate/malformed"));
        }
    }
    exact_entries(
        &entries
            .into_iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect(),
        &["version", "format", "class", "location", "object"],
    )?;
    if body.contains("version 2.0;")
        && body.contains("format ascii;")
        && body.contains("class dictionary;")
        && body.contains("location \"constant\";")
        && body.contains(&format!("object {object};"))
    {
        Ok(())
    } else {
        Err(invalid("wrong FoamFile header"))
    }
}
fn validate_foam_dictionary(text: &str, object: &str) -> Result<()> {
    validate_foam_header(text, object)?;
    let header_end = text
        .find('}')
        .ok_or_else(|| invalid("unterminated FoamFile header"))?;
    let statements: Vec<_> = text[header_end + 1..]
        .split(';')
        .map(str::trim)
        .filter(|statement| !statement.is_empty())
        .collect();
    match object {
        "momentumTransport" if statements == ["simulationType laminar"] => Ok(()),
        "physicalProperties"
            if statements.len() == 2 && statements[0] == "viscosityModel constant" =>
        {
            let normalized = statements[1].replace('[', "[ ").replace(']', " ]");
            let tokens: Vec<_> = normalized.split_whitespace().collect();
            let entry = PropertyEntry {
                key: tokens.first().copied().unwrap_or_default().into(),
                value: tokens.iter().skip(1).map(|token| (*token).into()).collect(),
            };
            if entry.key != "nu" {
                return Err(invalid("wrong physicalProperties key"));
            }
            assert_dimensioned(&entry, [0, 2, -1, 0, 0, 0, 0], MU / RHO, 1e-11)
        }
        _ => Err(invalid("wrong OpenFOAM dictionary body")),
    }
}
fn clone_field(field: &FieldFile) -> FieldFile {
    FieldFile {
        path: field.path.clone(),
        region: field.region.clone(),
        name: field.name.clone(),
        class_name: field.class_name.clone(),
        dimensions: field.dimensions.clone(),
        internal_field: field.internal_field.clone(),
        boundary_patches: field
            .boundary_patches
            .iter()
            .map(|patch| FieldBoundaryPatch {
                name: patch.name.clone(),
                patch_type: patch.patch_type.clone(),
                inlet_value: patch.inlet_value.clone(),
                value: patch.value.clone(),
            })
            .collect(),
    }
}
fn clone_mesh(m: &PolyMesh) -> PolyMesh {
    PolyMesh {
        path: m.path.clone(),
        points: m.points.clone(),
        faces: m.faces.clone(),
        owner: m.owner.clone(),
        neighbour: m.neighbour.clone(),
        patches: m
            .patches
            .iter()
            .map(|p| BoundaryPatch {
                name: p.name.clone(),
                patch_type: p.patch_type.clone(),
                faces: p.faces,
                start_face: p.start_face,
            })
            .collect(),
    }
}

#[test]
fn plane_channel_contract_is_complete() -> Result<()> {
    let base = Path::new(ROOT).join(BASE);
    validate_metadata(&fs::read_to_string(base.join("comparison.toml"))?)?;
    validate_shared_metadata(&fs::read_to_string(
        base.join("shared/physicalParameters.toml"),
    )?)?;
    for name in ["ferrum", "openfoam-v13"] {
        let dir = case(name);
        validate_properties(name, &read_case_properties(&dir)?)?;
        validate_fields(name, &read_initial_fields(&dir)?.fields)?;
        validate_mesh(&PolyMesh::read(&dir.join("constant/polyMesh"))?)?;
    }
    for object in ["momentumTransport", "physicalProperties"] {
        let text = fs::read_to_string(case("openfoam-v13").join("constant").join(object))?;
        validate_foam_dictionary(&text, object)?;
    }
    Ok(())
}

#[test]
fn comparison_table_and_model_mutations_are_rejected() -> Result<()> {
    let text = fs::read_to_string(Path::new(ROOT).join(BASE).join("comparison.toml"))?;
    assert!(validate_metadata(&(text.clone() + "\n[extra]\nvalue=1\n")).is_err());
    assert!(
        validate_metadata(&text.replace(
            "pressure_drop_pa = 0.6012",
            "[relocated]\npressure_drop_pa = 0.6012"
        ))
        .is_err()
    );
    assert!(validate_metadata(&text.replace("Plane-Poiseuille", "wrong")).is_err());
    assert!(validate_metadata(&text.replace("ferrum/case", "bogus/case")).is_err());
    assert!(validate_metadata(&text.replace("length_m = 1.0", "length_m = 2.0")).is_err());
    assert!(
        validate_metadata(&text.replace("pressure_drop_pa = 0.6012", "pressure_drop_pa = 1.0"))
            .is_err()
    );
    Ok(())
}

#[test]
fn wrong_reference_model_is_rejected() -> Result<()> {
    comparison_table_and_model_mutations_are_rejected()
}

#[test]
fn wrong_property_dimensions_are_rejected() {
    let bad_dimensions = PropertyEntry {
        key: "nu".into(),
        value: vec![
            "[",
            "1",
            "2",
            "-1",
            "0",
            "0",
            "0",
            "0",
            "]",
            "1.0038068513323983e-6",
        ]
        .into_iter()
        .map(str::to_string)
        .collect(),
    };
    assert!(assert_dimensioned(&bad_dimensions, [0, 2, -1, 0, 0, 0, 0], MU / RHO, 1e-11).is_err());
    let bad_value = PropertyEntry {
        key: "nu".into(),
        value: vec!["[", "0", "2", "-1", "0", "0", "0", "0", "]", "NaN"]
            .into_iter()
            .map(str::to_string)
            .collect(),
    };
    assert!(assert_dimensioned(&bad_value, [0, 2, -1, 0, 0, 0, 0], MU / RHO, 1e-11).is_err());
}

#[test]
fn field_mutation_probes_are_rejected() -> Result<()> {
    let fields = read_initial_fields(&case("ferrum"))?.fields;
    let source_p = fields.iter().find(|field| field.name == "p").unwrap();

    let mut vector_p = clone_field(source_p);
    vector_p.internal_field = Some(FieldValueSummary::Uniform("(0.3006 0 0)".into()));
    assert!(validate_p(&vector_p, false).is_err());

    let mut wrong_name = clone_field(source_p);
    wrong_name.name = "q".into();
    assert!(validate_p(&wrong_name, false).is_err());
    let mut wrong_class = clone_field(source_p);
    wrong_class.class_name = Some("volVectorField".into());
    assert!(validate_p(&wrong_class, false).is_err());
    let mut wrong_region = clone_field(source_p);
    wrong_region.region = Some("fluid".into());
    assert!(validate_p(&wrong_region, false).is_err());
    let mut wrong_pressure = clone_field(source_p);
    wrong_pressure.internal_field = Some(FieldValueSummary::Uniform("0.4".into()));
    assert!(validate_p(&wrong_pressure, false).is_err());
    let mut boundary_drift = clone_field(source_p);
    boundary_drift.boundary_patches[0].patch_type = Some("zeroGradient".into());
    assert!(validate_p(&boundary_drift, false).is_err());
    Ok(())
}

#[test]
fn duplicate_property_and_non_laminar_dictionary_are_rejected() -> Result<()> {
    let mut dictionaries = read_case_properties(&case("ferrum"))?;
    let dictionary = dictionaries
        .iter_mut()
        .find(|dictionary| dictionary.name == "transportProperties")
        .unwrap();
    dictionary.entries.push(dictionary.entries[0].clone());
    assert!(validate_properties("ferrum", &dictionaries).is_err());

    let text = fs::read_to_string(case("openfoam-v13").join("constant/momentumTransport"))?;
    assert!(
        validate_foam_dictionary(&text.replace("laminar", "RAS"), "momentumTransport").is_err()
    );
    Ok(())
}

#[test]
fn reversed_front_winding_is_rejected() -> Result<()> {
    let original = PolyMesh::read(&case("ferrum").join("constant/polyMesh"))?;
    let mut mesh = clone_mesh(&original);
    let face = mesh
        .patches
        .iter()
        .find(|p| p.name == "front")
        .unwrap()
        .start_face;
    mesh.faces[face].reverse();
    assert!(validate_mesh(&mesh).is_err());
    Ok(())
}

#[test]
fn mesh_mutation_probes_are_rejected() -> Result<()> {
    let original = PolyMesh::read(&case("ferrum").join("constant/polyMesh"))?;
    let mut six = clone_mesh(&original);
    let inlet = &six.patches[0];
    six.patches.push(BoundaryPatch {
        name: "inlet".into(),
        patch_type: inlet.patch_type.clone(),
        faces: inlet.faces,
        start_face: inlet.start_face,
    });
    assert!(validate_mesh(&six).is_err());
    let mut gap = clone_mesh(&original);
    gap.patches[1].start_face += 1;
    assert!(validate_mesh(&gap).is_err());
    let mut kind = clone_mesh(&original);
    kind.patches
        .iter_mut()
        .find(|p| p.name == "front")
        .unwrap()
        .patch_type = "wall".into();
    assert!(validate_mesh(&kind).is_err());
    let mut plane = clone_mesh(&original);
    let f = plane
        .patches
        .iter()
        .find(|p| p.name == "inlet")
        .unwrap()
        .start_face;
    let point = plane.faces[f][0];
    plane.points[point].x += 1e-4;
    assert!(validate_mesh(&plane).is_err());
    Ok(())
}
