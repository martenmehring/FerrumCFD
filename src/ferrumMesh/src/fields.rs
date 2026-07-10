use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

use crate::dictionary::{TokenCursor, tokenize};
use crate::poly_mesh::PolyMesh;
use crate::{MeshError, Result};

#[derive(Debug)]
pub struct InitialFieldSet {
    pub case_dir: PathBuf,
    pub fields: Vec<FieldFile>,
}

#[derive(Debug)]
pub struct FieldFile {
    pub path: PathBuf,
    pub region: Option<String>,
    pub name: String,
    pub class_name: Option<String>,
    pub dimensions: Option<Vec<String>>,
    pub internal_field: Option<FieldValueSummary>,
    pub boundary_patches: Vec<FieldBoundaryPatch>,
}

#[derive(Clone, Debug)]
pub enum FieldValueSummary {
    Uniform(String),
    NonUniform {
        value_type: Option<String>,
        count: Option<usize>,
        values: Option<Vec<f64>>,
    },
    Other(String),
}

#[derive(Debug)]
pub struct FieldBoundaryPatch {
    pub name: String,
    pub patch_type: Option<String>,
    pub inlet_value: Option<FieldValueSummary>,
    pub value: Option<FieldValueSummary>,
}

#[derive(Debug)]
pub struct FieldBoundaryValidationSummary {
    pub fields: usize,
    pub warnings: Vec<String>,
}

impl std::fmt::Display for FieldValueSummary {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Uniform(value) => write!(formatter, "uniform {value}"),
            Self::NonUniform {
                value_type,
                count,
                values,
            } => {
                write!(
                    formatter,
                    "nonuniform {} count={} loadedScalars={}",
                    value_type.as_deref().unwrap_or("unknown"),
                    count
                        .map(|value| value.to_string())
                        .unwrap_or_else(|| "unknown".to_string()),
                    values
                        .as_ref()
                        .map(|values| values.len().to_string())
                        .unwrap_or_else(|| "unknown".to_string())
                )
            }
            Self::Other(value) => formatter.write_str(value),
        }
    }
}

pub fn read_initial_fields(case_dir: &Path) -> Result<InitialFieldSet> {
    let fields_dir = case_dir.join("0");
    if !fields_dir.exists() {
        return Ok(InitialFieldSet {
            case_dir: case_dir.to_path_buf(),
            fields: Vec::new(),
        });
    }

    let mut fields = Vec::new();
    read_field_files_in_dir(&fields_dir, None, &mut fields)?;

    for entry in fs::read_dir(&fields_dir)? {
        let entry = entry?;
        let path = entry.path();
        let file_type = entry.file_type()?;
        if file_type.is_symlink() || !file_type.is_dir() {
            continue;
        }

        let region = entry.file_name().to_string_lossy().to_string();
        read_field_files_in_dir(&path, Some(region), &mut fields)?;
    }

    fields.sort_by(|left, right| {
        left.region
            .cmp(&right.region)
            .then(left.name.cmp(&right.name))
            .then(left.path.cmp(&right.path))
    });

    Ok(InitialFieldSet {
        case_dir: case_dir.to_path_buf(),
        fields,
    })
}

pub fn read_fields_from_directory(case_dir: &Path, fields_dir: &Path) -> Result<InitialFieldSet> {
    let metadata = fs::symlink_metadata(fields_dir).map_err(|error| {
        MeshError::InvalidInput(format!(
            "could not inspect field directory {} ({error})",
            fields_dir.display()
        ))
    })?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(MeshError::InvalidInput(format!(
            "field directory must be a real directory, not a symlink: {}",
            fields_dir.display()
        )));
    }

    let mut fields = Vec::new();
    read_field_files_in_dir(fields_dir, None, &mut fields)?;
    fields.sort_by(|left, right| left.name.cmp(&right.name).then(left.path.cmp(&right.path)));
    Ok(InitialFieldSet {
        case_dir: case_dir.to_path_buf(),
        fields,
    })
}

pub fn validate_initial_field_boundaries(
    case_dir: &Path,
    fields: &InitialFieldSet,
) -> FieldBoundaryValidationSummary {
    let mut validator = FieldBoundaryValidator::new(case_dir);
    for field in &fields.fields {
        validator.validate_field(field);
    }

    FieldBoundaryValidationSummary {
        fields: fields.fields.len(),
        warnings: validator.warnings,
    }
}

fn read_field_files_in_dir(
    dir: &Path,
    region: Option<String>,
    fields: &mut Vec<FieldFile>,
) -> Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        let file_type = entry.file_type()?;
        if file_type.is_symlink() {
            return Err(MeshError::InvalidInput(format!(
                "initial field symlinks are not allowed: {}",
                path.display()
            )));
        }
        if !file_type.is_file() {
            continue;
        }

        fields.push(read_field_file(&path, region.clone())?);
    }
    Ok(())
}

fn read_field_file(path: &Path, region: Option<String>) -> Result<FieldFile> {
    let content = fs::read_to_string(path).map_err(|error| {
        MeshError::InvalidInput(format!("could not read {} ({error})", path.display()))
    })?;
    parse_field_file_str(&content, path, region)
}

fn parse_field_file_str(content: &str, path: &Path, region: Option<String>) -> Result<FieldFile> {
    let tokens = tokenize(content);
    let mut cursor = TokenCursor::new(path, tokens);
    let mut class_name = None;
    let mut object_name = None;
    let mut dimensions = None;
    let mut internal_field = None;
    let mut boundary_patches = Vec::new();

    while let Some(token) = cursor.peek() {
        match token {
            "FoamFile" => {
                cursor.next_required()?;
                let metadata = parse_foam_file(&mut cursor)?;
                class_name = metadata.class_name;
                object_name = metadata.object_name;
            }
            "dimensions" => dimensions = Some(parse_dimensions(&mut cursor)?),
            "internalField" => {
                cursor.next_required()?;
                internal_field = Some(parse_field_value_tokens(
                    cursor.read_value_until_semicolon()?,
                ));
            }
            "boundaryField" => {
                cursor.next_required()?;
                boundary_patches = parse_boundary_field(&mut cursor)?;
            }
            _ => cursor.skip_value_or_block()?,
        }
    }

    Ok(FieldFile {
        path: path.to_path_buf(),
        region,
        name: object_name.unwrap_or_else(|| {
            path.file_name()
                .map(|name| name.to_string_lossy().to_string())
                .unwrap_or_else(|| "unknown".to_string())
        }),
        class_name,
        dimensions,
        internal_field,
        boundary_patches,
    })
}

struct FoamFileMetadata {
    class_name: Option<String>,
    object_name: Option<String>,
}

fn parse_foam_file(cursor: &mut TokenCursor) -> Result<FoamFileMetadata> {
    cursor.expect("{")?;
    let mut class_name = None;
    let mut object_name = None;

    while !cursor.peek_is("}")? {
        let key = cursor.next_required()?;
        let value = cursor.read_value_until_semicolon()?;
        match key.as_str() {
            "class" => class_name = value.first().cloned(),
            "object" => object_name = value.first().cloned(),
            _ => {}
        }
    }
    cursor.expect("}")?;

    Ok(FoamFileMetadata {
        class_name,
        object_name,
    })
}

fn parse_dimensions(cursor: &mut TokenCursor) -> Result<Vec<String>> {
    cursor.next_required()?;
    if cursor.peek() != Some("[") {
        return cursor.read_value_until_semicolon();
    }

    cursor.expect("[")?;
    let mut values = Vec::new();
    while !cursor.peek_is("]")? {
        values.push(cursor.next_required()?);
    }
    cursor.expect("]")?;
    cursor.expect_optional(";")?;
    Ok(values)
}

fn parse_boundary_field(cursor: &mut TokenCursor) -> Result<Vec<FieldBoundaryPatch>> {
    cursor.expect("{")?;
    let mut patches = Vec::new();

    while !cursor.peek_is("}")? {
        let name = cursor.next_required()?;
        cursor.expect("{")?;
        patches.push(parse_boundary_patch(cursor, name)?);
    }
    cursor.expect("}")?;

    Ok(patches)
}

fn parse_boundary_patch(cursor: &mut TokenCursor, name: String) -> Result<FieldBoundaryPatch> {
    let mut patch_type = None;
    let mut inlet_value = None;
    let mut value = None;

    while !cursor.peek_is("}")? {
        let key = cursor.next_required()?;
        match key.as_str() {
            "type" => {
                patch_type = cursor.read_value_until_semicolon()?.first().cloned();
            }
            "value" => {
                value = Some(parse_field_value_tokens(
                    cursor.read_value_until_semicolon()?,
                ));
            }
            "inletValue" => {
                inlet_value = Some(parse_field_value_tokens(
                    cursor.read_value_until_semicolon()?,
                ));
            }
            _ => cursor.skip_value_or_block()?,
        }
    }
    cursor.expect("}")?;

    Ok(FieldBoundaryPatch {
        name,
        patch_type,
        inlet_value,
        value,
    })
}

fn parse_field_value_tokens(tokens: Vec<String>) -> FieldValueSummary {
    let Some(kind) = tokens.first() else {
        return FieldValueSummary::Other("empty".to_string());
    };

    match kind.as_str() {
        "uniform" => FieldValueSummary::Uniform(join_tokens(&tokens[1..])),
        "nonuniform" => parse_nonuniform_field_value(&tokens),
        _ => FieldValueSummary::Other(join_tokens(&tokens)),
    }
}

fn parse_nonuniform_field_value(tokens: &[String]) -> FieldValueSummary {
    let value_type = tokens.get(1).cloned();
    let count_index = tokens
        .iter()
        .enumerate()
        .skip(2)
        .find_map(|(index, token)| token.parse::<usize>().ok().map(|count| (index, count)));
    let count = count_index.map(|(_, count)| count);
    let values = count_index.and_then(|(index, count)| {
        parse_nonuniform_numeric_values(tokens, index + 1, &value_type, count)
    });

    FieldValueSummary::NonUniform {
        value_type,
        count,
        values,
    }
}

fn parse_nonuniform_numeric_values(
    tokens: &[String],
    start_index: usize,
    value_type: &Option<String>,
    count: usize,
) -> Option<Vec<f64>> {
    let components = nonuniform_components_for_type(value_type.as_deref())?;
    let expected_values = count.checked_mul(components)?;
    let mut values = Vec::new();
    for token in &tokens[start_index..] {
        if matches!(token.as_str(), "(" | ")" | "[" | "]") {
            continue;
        }
        if values.len() == expected_values {
            return None;
        }
        values.push(token.parse::<f64>().ok()?);
    }
    if values.len() == expected_values {
        Some(values)
    } else {
        None
    }
}

fn nonuniform_components_for_type(value_type: Option<&str>) -> Option<usize> {
    match value_type {
        Some("List<scalar>") | Some("scalarField") | Some("Field<scalar>") => Some(1),
        Some("List<vector>") | Some("vectorField") | Some("Field<vector>") => Some(3),
        _ => None,
    }
}

fn join_tokens(tokens: &[String]) -> String {
    if tokens.is_empty() {
        return "empty".to_string();
    }
    tokens.join(" ")
}

struct FieldBoundaryValidator<'a> {
    case_dir: &'a Path,
    mesh_cache: HashMap<Option<String>, Result<PolyMesh>>,
    warnings: Vec<String>,
}

impl<'a> FieldBoundaryValidator<'a> {
    fn new(case_dir: &'a Path) -> Self {
        Self {
            case_dir,
            mesh_cache: HashMap::new(),
            warnings: Vec::new(),
        }
    }

    fn validate_field(&mut self, field: &FieldFile) {
        let region = field.region.clone();
        let mesh = self.mesh_for_region(region.clone());
        let Some(mesh) = mesh else {
            return;
        };

        let mut field_warnings = Vec::new();
        validate_field_boundary_patches(field, mesh, &mut field_warnings);
        self.warnings.extend(field_warnings);
    }

    fn mesh_for_region(&mut self, region: Option<String>) -> Option<&PolyMesh> {
        if !self.mesh_cache.contains_key(&region) {
            let mesh_path = if let Some(region) = &region {
                self.case_dir.join("constant").join(region).join("polyMesh")
            } else {
                self.case_dir.join("constant").join("polyMesh")
            };
            let mesh = PolyMesh::read(&mesh_path);
            self.mesh_cache.insert(region.clone(), mesh);
        }

        match self
            .mesh_cache
            .get(&region)
            .expect("mesh cache entry exists")
        {
            Ok(mesh) => Some(mesh),
            Err(error) => {
                let label = region
                    .as_deref()
                    .map(|region| format!("region '{region}'"))
                    .unwrap_or_else(|| "base mesh".to_string());
                self.warnings
                    .push(format!("could not validate fields for {label}: {error}"));
                None
            }
        }
    }
}

fn validate_field_boundary_patches(field: &FieldFile, mesh: &PolyMesh, warnings: &mut Vec<String>) {
    let field_label = field_label(field);
    let mut field_patches = HashMap::with_capacity(field.boundary_patches.len());
    let mut seen = HashSet::new();
    for patch in &field.boundary_patches {
        if !seen.insert(patch.name.as_str()) {
            warnings.push(format!(
                "field '{}' has duplicate boundaryField entry '{}'",
                field_label, patch.name
            ));
        }
        field_patches.entry(patch.name.as_str()).or_insert(patch);
    }

    let mesh_patch_names = mesh
        .patches
        .iter()
        .map(|patch| patch.name.as_str())
        .collect::<HashSet<_>>();
    for patch in &mesh.patches {
        let Some(field_patch) = field_patches.get(patch.name.as_str()).copied() else {
            warnings.push(format!(
                "field '{}' is missing boundaryField entry for mesh patch '{}'",
                field_label, patch.name
            ));
            continue;
        };

        validate_special_patch_field_type(field, field_patch, &patch.patch_type, warnings);
    }

    for field_patch in &field.boundary_patches {
        if !mesh_patch_names.contains(field_patch.name.as_str()) {
            warnings.push(format!(
                "field '{}' has boundaryField entry '{}' that is not a mesh patch",
                field_label, field_patch.name
            ));
        }
    }
}

fn validate_special_patch_field_type(
    field: &FieldFile,
    field_patch: &FieldBoundaryPatch,
    mesh_patch_type: &str,
    warnings: &mut Vec<String>,
) {
    let expected = match mesh_patch_type {
        "empty" => "empty",
        "wedge" => "wedge",
        "symmetryPlane" => "symmetryPlane",
        _ => return,
    };

    if field_patch.patch_type.as_deref() != Some(expected) {
        warnings.push(format!(
            "field '{}' patch '{}' should use boundary type '{}' for mesh patch type '{}', found '{}'",
            field_label(field),
            field_patch.name,
            expected,
            mesh_patch_type,
            field_patch.patch_type.as_deref().unwrap_or("missing")
        ));
    }
}

fn field_label(field: &FieldFile) -> String {
    if let Some(region) = &field.region {
        format!("{region}/{}", field.name)
    } else {
        field.name.clone()
    }
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use crate::Point3;
    use crate::poly_mesh::{BoundaryPatch, PolyMesh};

    use super::{
        FieldBoundaryPatch, FieldFile, FieldValueSummary, parse_field_file_str,
        validate_field_boundary_patches,
    };

    #[test]
    fn parses_scalar_field_with_boundary_field() {
        let content = r#"
        FoamFile
        {
            version 2.0;
            format ascii;
            class volScalarField;
            object p;
        }

        dimensions [0 2 -2 0 0 0 0];
        internalField uniform 0;

        boundaryField
        {
            inlet
            {
                type fixedValue;
                value uniform 10;
            }
            outlet
            {
                type zeroGradient;
            }
        }
        "#;

        let field = parse_field_file_str(content, Path::new("0/p"), None).unwrap();
        assert_eq!(field.name, "p");
        assert_eq!(field.class_name.as_deref(), Some("volScalarField"));
        assert_eq!(field.dimensions.unwrap().len(), 7);
        assert_eq!(field.boundary_patches.len(), 2);
        assert_eq!(
            field.boundary_patches[0].patch_type.as_deref(),
            Some("fixedValue")
        );
    }

    #[test]
    fn parses_vector_uniform_values() {
        let content = r#"
        FoamFile
        {
            class volVectorField;
            object U;
        }

        dimensions [0 1 -1 0 0 0 0];
        internalField uniform (0 0 0);
        boundaryField
        {
            inlet
            {
                type fixedValue;
                value uniform (1 0 0);
            }
        }
        "#;

        let field = parse_field_file_str(content, Path::new("0/U"), None).unwrap();
        match field.internal_field.unwrap() {
            FieldValueSummary::Uniform(value) => assert_eq!(value, "( 0 0 0 )"),
            other => panic!("unexpected field value: {other:?}"),
        }
    }

    #[test]
    fn parses_inlet_value_boundary_entry() {
        let content = r#"
        FoamFile
        {
            class volVectorField;
            object U;
        }

        dimensions [0 1 -1 0 0 0 0];
        internalField uniform (0 0 0);
        boundaryField
        {
            outlet
            {
                type inletOutlet;
                inletValue uniform (1 0 0);
                value uniform (0 0 0);
            }
        }
        "#;

        let field = parse_field_file_str(content, Path::new("0/U"), None).unwrap();
        let patch = &field.boundary_patches[0];

        assert_eq!(patch.patch_type.as_deref(), Some("inletOutlet"));
        match patch.inlet_value.as_ref().expect("inletValue") {
            FieldValueSummary::Uniform(value) => assert_eq!(value, "( 1 0 0 )"),
            other => panic!("unexpected inlet value: {other:?}"),
        }
    }

    #[test]
    fn parses_nonuniform_summary() {
        let content = r#"
        FoamFile
        {
            class volScalarField;
            object T;
        }

        dimensions [0 0 0 1 0 0 0];
        internalField nonuniform List<scalar>
        3
        (
            300
            310
            320
        );
        boundaryField {}
        "#;

        let field =
            parse_field_file_str(content, Path::new("0/T"), Some("fluid".to_string())).unwrap();
        assert_eq!(field.region.as_deref(), Some("fluid"));
        match field.internal_field.unwrap() {
            FieldValueSummary::NonUniform {
                value_type,
                count,
                values,
            } => {
                assert_eq!(value_type.as_deref(), Some("List<scalar>"));
                assert_eq!(count, Some(3));
                assert_eq!(values, Some(vec![300.0, 310.0, 320.0]));
            }
            other => panic!("unexpected field value: {other:?}"),
        }
    }

    #[test]
    fn parses_nonuniform_vector_values() {
        let content = r#"
        FoamFile
        {
            class volVectorField;
            object U;
        }

        dimensions [0 1 -1 0 0 0 0];
        internalField nonuniform List<vector>
        2
        (
            (1 0 0)
            (0 1 0)
        );
        boundaryField {}
        "#;

        let field = parse_field_file_str(content, Path::new("0/U"), None).unwrap();
        match field.internal_field.unwrap() {
            FieldValueSummary::NonUniform {
                value_type,
                count,
                values,
            } => {
                assert_eq!(value_type.as_deref(), Some("List<vector>"));
                assert_eq!(count, Some(2));
                assert_eq!(values, Some(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]));
            }
            other => panic!("unexpected field value: {other:?}"),
        }
    }

    #[test]
    fn rejects_nonuniform_numeric_tail_beyond_declared_count() {
        let content = r#"
        FoamFile { class volScalarField; object p; }
        internalField nonuniform List<scalar> 1 ( 1 2 3 );
        boundaryField { }
        "#;

        let field = parse_field_file_str(content, Path::new("0/p"), None).unwrap();

        match field.internal_field {
            Some(FieldValueSummary::NonUniform { count, values, .. }) => {
                assert_eq!(count, Some(1));
                assert_eq!(values, None);
            }
            other => panic!("unexpected field value: {other:?}"),
        }
    }

    #[test]
    fn validates_special_field_patch_types() {
        let field = test_field(vec![
            FieldBoundaryPatch {
                name: "inlet".to_string(),
                patch_type: Some("fixedValue".to_string()),
                inlet_value: None,
                value: None,
            },
            FieldBoundaryPatch {
                name: "front".to_string(),
                patch_type: Some("empty".to_string()),
                inlet_value: None,
                value: None,
            },
        ]);
        let mesh = test_mesh(vec![
            BoundaryPatch {
                name: "inlet".to_string(),
                patch_type: "patch".to_string(),
                faces: 1,
                start_face: 0,
            },
            BoundaryPatch {
                name: "front".to_string(),
                patch_type: "empty".to_string(),
                faces: 1,
                start_face: 1,
            },
        ]);
        let mut warnings = Vec::new();

        validate_field_boundary_patches(&field, &mesh, &mut warnings);

        assert!(warnings.is_empty());
    }

    #[test]
    fn warns_for_missing_extra_and_wrong_special_field_patches() {
        let field = test_field(vec![
            FieldBoundaryPatch {
                name: "front".to_string(),
                patch_type: Some("zeroGradient".to_string()),
                inlet_value: None,
                value: None,
            },
            FieldBoundaryPatch {
                name: "unused".to_string(),
                patch_type: Some("patch".to_string()),
                inlet_value: None,
                value: None,
            },
        ]);
        let mesh = test_mesh(vec![
            BoundaryPatch {
                name: "inlet".to_string(),
                patch_type: "patch".to_string(),
                faces: 1,
                start_face: 0,
            },
            BoundaryPatch {
                name: "front".to_string(),
                patch_type: "empty".to_string(),
                faces: 1,
                start_face: 1,
            },
        ]);
        let mut warnings = Vec::new();

        validate_field_boundary_patches(&field, &mesh, &mut warnings);

        assert_eq!(warnings.len(), 3);
    }

    fn test_field(boundary_patches: Vec<FieldBoundaryPatch>) -> FieldFile {
        FieldFile {
            path: PathBuf::from("0/p"),
            region: None,
            name: "p".to_string(),
            class_name: Some("volScalarField".to_string()),
            dimensions: None,
            internal_field: None,
            boundary_patches,
        }
    }

    fn test_mesh(patches: Vec<BoundaryPatch>) -> PolyMesh {
        PolyMesh {
            path: PathBuf::from("polyMesh"),
            points: vec![
                Point3 {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                },
                Point3 {
                    x: 1.0,
                    y: 0.0,
                    z: 0.0,
                },
                Point3 {
                    x: 0.0,
                    y: 1.0,
                    z: 0.0,
                },
            ],
            faces: vec![vec![0, 1, 2], vec![0, 2, 1]],
            owner: vec![0, 0],
            neighbour: Vec::new(),
            patches,
        }
    }
}
