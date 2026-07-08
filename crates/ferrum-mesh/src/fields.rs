use std::fs;
use std::path::{Path, PathBuf};

use crate::dictionary::{TokenCursor, tokenize};
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
    },
    Other(String),
}

#[derive(Debug)]
pub struct FieldBoundaryPatch {
    pub name: String,
    pub patch_type: Option<String>,
    pub value: Option<FieldValueSummary>,
}

impl std::fmt::Display for FieldValueSummary {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Uniform(value) => write!(formatter, "uniform {value}"),
            Self::NonUniform { value_type, count } => {
                write!(
                    formatter,
                    "nonuniform {} count={}",
                    value_type.as_deref().unwrap_or("unknown"),
                    count
                        .map(|value| value.to_string())
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
        if !path.is_dir() {
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

fn read_field_files_in_dir(
    dir: &Path,
    region: Option<String>,
    fields: &mut Vec<FieldFile>,
) -> Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_file() {
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
            _ => cursor.skip_value_or_block()?,
        }
    }
    cursor.expect("}")?;

    Ok(FieldBoundaryPatch {
        name,
        patch_type,
        value,
    })
}

fn parse_field_value_tokens(tokens: Vec<String>) -> FieldValueSummary {
    let Some(kind) = tokens.first() else {
        return FieldValueSummary::Other("empty".to_string());
    };

    match kind.as_str() {
        "uniform" => FieldValueSummary::Uniform(join_tokens(&tokens[1..])),
        "nonuniform" => FieldValueSummary::NonUniform {
            value_type: tokens.get(1).cloned(),
            count: tokens.iter().find_map(|token| token.parse::<usize>().ok()),
        },
        _ => FieldValueSummary::Other(join_tokens(&tokens)),
    }
}

fn join_tokens(tokens: &[String]) -> String {
    if tokens.is_empty() {
        return "empty".to_string();
    }
    tokens.join(" ")
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::{FieldValueSummary, parse_field_file_str};

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
            FieldValueSummary::NonUniform { value_type, count } => {
                assert_eq!(value_type.as_deref(), Some("List<scalar>"));
                assert_eq!(count, Some(3));
            }
            other => panic!("unexpected field value: {other:?}"),
        }
    }
}
