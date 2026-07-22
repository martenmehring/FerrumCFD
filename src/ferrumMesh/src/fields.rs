use std::cmp::Ordering;
use std::fs::{self, File};
#[cfg(test)]
use std::io::Cursor;
use std::io::{self, BufRead, BufReader};
use std::mem::size_of;
use std::path::{Path, PathBuf};

use crate::dictionary::{
    MAX_DICTIONARY_NESTING, MAX_TOKEN_BYTES, Token, TokenProvenance, TokenSource,
};
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FieldLoadPolicy {
    Summary,
    Full,
}

const MAX_RETAINED_FIELD_VALUE_BYTES: usize = MAX_TOKEN_BYTES;

pub(crate) fn nonuniform_value_type_components(value_type: Option<&str>) -> Option<usize> {
    match value_type {
        Some("List<scalar>" | "scalarField" | "Field<scalar>") => Some(1),
        Some("List<vector>" | "vectorField" | "Field<vector>") => Some(3),
        _ => None,
    }
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
    read_initial_fields_with_policy(case_dir, FieldLoadPolicy::Full)
}

pub fn read_initial_fields_with_policy(
    case_dir: &Path,
    policy: FieldLoadPolicy,
) -> Result<InitialFieldSet> {
    let owned_case_dir = try_path_buf(case_dir, "initial field case path allocation failed")?;
    let fields_dir = case_dir.join("0");
    let fields_metadata = match fs::symlink_metadata(&fields_dir) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            return Ok(InitialFieldSet {
                case_dir: owned_case_dir,
                fields: Vec::new(),
            });
        }
        Err(_) => {
            return Err(field_path_error(
                &fields_dir,
                "could not inspect initial field directory",
            )?);
        }
    };
    if fields_metadata.file_type().is_symlink() || !fields_metadata.is_dir() {
        return Err(field_path_error(
            &fields_dir,
            "initial field directory must be a real directory, not a symlink",
        )?);
    }

    let mut fields = Vec::new();
    read_field_files_in_dir(&fields_dir, None, policy, &mut fields)?;

    let region_entries = path_io(
        fs::read_dir(&fields_dir),
        &fields_dir,
        "could not read initial field directory",
    )?;
    for entry in region_entries {
        let entry = path_io(
            entry,
            &fields_dir,
            "could not read initial field directory entry",
        )?;
        let path = entry.path();
        let file_type = path_io(
            entry.file_type(),
            &path,
            "could not inspect initial field entry",
        )?;
        if file_type.is_symlink() {
            return Err(field_path_error(
                &path,
                "initial field symlinks are not allowed",
            )?);
        }
        if !file_type.is_dir() {
            continue;
        }

        let file_name = entry.file_name();
        let region = file_name.to_str().ok_or_else(|| MeshError::Parse {
            line: 1,
            message: "initial field region name is not valid UTF-8".to_owned(),
        })?;
        read_field_files_in_dir(&path, Some(region), policy, &mut fields)?;
    }

    fields.sort_by(|left, right| {
        left.region
            .cmp(&right.region)
            .then(left.name.cmp(&right.name))
            .then(left.path.cmp(&right.path))
    });

    Ok(InitialFieldSet {
        case_dir: owned_case_dir,
        fields,
    })
}

pub fn read_fields_from_directory(case_dir: &Path, fields_dir: &Path) -> Result<InitialFieldSet> {
    read_fields_from_directory_with_policy(case_dir, fields_dir, FieldLoadPolicy::Full)
}

pub fn read_fields_from_directory_with_policy(
    case_dir: &Path,
    fields_dir: &Path,
    policy: FieldLoadPolicy,
) -> Result<InitialFieldSet> {
    let metadata = path_io(
        fs::symlink_metadata(fields_dir),
        fields_dir,
        "could not inspect initial field directory",
    )?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(field_path_error(
            fields_dir,
            "initial field directory must be a real directory, not a symlink",
        )?);
    }

    let mut fields = Vec::new();
    read_field_files_in_dir(fields_dir, None, policy, &mut fields)?;
    fields.sort_by(|left, right| left.name.cmp(&right.name).then(left.path.cmp(&right.path)));
    Ok(InitialFieldSet {
        case_dir: try_path_buf(case_dir, "initial field case path allocation failed")?,
        fields,
    })
}

pub fn validate_initial_field_boundaries(
    case_dir: &Path,
    fields: &InitialFieldSet,
) -> Result<FieldBoundaryValidationSummary> {
    validate_canonical_field_order(&fields.fields)?;
    let mut validator = FieldBoundaryValidator::new(case_dir);
    for field in &fields.fields {
        validator.validate_field(field)?;
    }

    Ok(FieldBoundaryValidationSummary {
        fields: fields.fields.len(),
        warnings: validator.warnings,
    })
}

fn read_field_files_in_dir(
    dir: &Path,
    region: Option<&str>,
    policy: FieldLoadPolicy,
    fields: &mut Vec<FieldFile>,
) -> Result<()> {
    let entries = path_io(
        fs::read_dir(dir),
        dir,
        "could not read initial field directory",
    )?;
    for entry in entries {
        let entry = path_io(entry, dir, "could not read initial field directory entry")?;
        let path = entry.path();
        let file_type = path_io(
            entry.file_type(),
            &path,
            "could not inspect initial field entry",
        )?;
        if file_type.is_symlink() {
            return Err(field_path_error(
                &path,
                "initial field symlinks are not allowed",
            )?);
        }
        if !file_type.is_file() {
            continue;
        }

        fields.try_reserve(1).map_err(|_| MeshError::Parse {
            line: 1,
            message: "initial field table allocation failed".to_owned(),
        })?;
        fields.push(read_field_file(&path, region, policy)?);
    }
    Ok(())
}

fn read_field_file(
    path: &Path,
    region: Option<&str>,
    policy: FieldLoadPolicy,
) -> Result<FieldFile> {
    let file = path_io(File::open(path), path, "could not open initial field file")?;
    let metadata = path_io(
        file.metadata(),
        path,
        "could not read initial field metadata",
    )?;
    if !metadata.is_file() {
        return Err(field_path_error(
            path,
            "initial field input is not a regular file",
        )?);
    }
    let exact_total_bytes = match usize::try_from(metadata.len()) {
        Ok(value) => value,
        Err(_) => {
            return Err(field_path_error(
                path,
                "initial field byte length exceeds this platform",
            )?);
        }
    };
    let region = match region {
        Some(value) => Some(try_string(value, "initial field region allocation failed")?),
        None => None,
    };
    parse_field_file_reader(
        BufReader::new(file),
        exact_total_bytes,
        path,
        region,
        policy,
    )
}

#[cfg(test)]
fn parse_field_file_str(content: &str, path: &Path, region: Option<String>) -> Result<FieldFile> {
    parse_field_file_str_with_policy(content, path, region, FieldLoadPolicy::Full)
}

#[cfg(test)]
fn parse_field_file_str_with_policy(
    content: &str,
    path: &Path,
    region: Option<String>,
    policy: FieldLoadPolicy,
) -> Result<FieldFile> {
    parse_field_file_reader(
        Cursor::new(content.as_bytes()),
        content.len(),
        path,
        region,
        policy,
    )
}

fn parse_field_file_reader<R: BufRead>(
    reader: R,
    exact_total_bytes: usize,
    path: &Path,
    region: Option<String>,
    policy: FieldLoadPolicy,
) -> Result<FieldFile> {
    let mut source = TokenSource::new(path, reader, exact_total_bytes)?;
    let mut class_name = None;
    let mut object_name = None;
    let mut dimensions = None;
    let mut internal_field = None;
    let mut boundary_patches = Vec::new();

    while let Some(token) = source.peek()? {
        #[derive(Clone, Copy)]
        enum Action {
            FoamFile,
            Dimensions,
            InternalField,
            BoundaryField,
            Other,
            Invalid,
        }
        let action = match token.provenance {
            TokenProvenance::Ordinary => match token.value.as_str() {
                "FoamFile" => Action::FoamFile,
                "dimensions" => Action::Dimensions,
                "internalField" => Action::InternalField,
                "boundaryField" => Action::BoundaryField,
                _ => Action::Other,
            },
            TokenProvenance::Quoted => Action::Other,
            TokenProvenance::Structural => Action::Invalid,
        };
        match action {
            Action::FoamFile => {
                source.next_required()?;
                let metadata = parse_foam_file(&mut source)?;
                class_name = metadata.class_name;
                object_name = metadata.object_name;
            }
            Action::Dimensions => dimensions = Some(parse_dimensions(&mut source)?),
            Action::InternalField => {
                source.next_required()?;
                internal_field = Some(parse_field_value(&mut source, policy)?);
            }
            Action::BoundaryField => {
                source.next_required()?;
                boundary_patches = parse_boundary_field(&mut source, policy)?;
            }
            Action::Other => {
                source.next_required()?;
                source.discard_exact_value_or_block()?;
            }
            Action::Invalid => {
                return source.reject_current_as("structural token cannot be a field key");
            }
        }
    }

    let name = match object_name {
        Some(value) => value,
        None => {
            let fallback = path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("unknown");
            try_string(fallback, "initial field name allocation failed")?
        }
    };

    Ok(FieldFile {
        path: try_path_buf(path, "initial field path allocation failed")?,
        region,
        name,
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

fn parse_foam_file<R: BufRead>(source: &mut TokenSource<R>) -> Result<FoamFileMetadata> {
    expect_structural(source, "{", "unexpected dictionary token")?;
    let mut class_name = None;
    let mut object_name = None;

    loop {
        #[derive(Clone, Copy)]
        enum Action {
            End,
            Class,
            Object,
            Other,
            Invalid,
        }
        let action = match source.peek()? {
            None => Action::Invalid,
            Some(token) if structural(token, "}") => Action::End,
            Some(token) if token.provenance == TokenProvenance::Structural => Action::Invalid,
            Some(token)
                if token.provenance == TokenProvenance::Ordinary && token.value == "class" =>
            {
                Action::Class
            }
            Some(token)
                if token.provenance == TokenProvenance::Ordinary && token.value == "object" =>
            {
                Action::Object
            }
            Some(_) => Action::Other,
        };
        match action {
            Action::End => break,
            Action::Invalid => return reject(source, "invalid FoamFile entry"),
            Action::Class | Action::Object => {
                source.next_required()?;
                let scalar = take_scalar_entry(source, "invalid FoamFile scalar entry")?;
                if matches!(action, Action::Class) {
                    class_name = Some(scalar);
                } else {
                    object_name = Some(scalar);
                }
            }
            Action::Other => {
                source.next_required()?;
                source.discard_exact_value_or_block()?;
            }
        }
    }
    expect_structural(source, "}", "unexpected dictionary token")?;

    Ok(FoamFileMetadata {
        class_name,
        object_name,
    })
}

fn parse_dimensions<R: BufRead>(source: &mut TokenSource<R>) -> Result<Vec<String>> {
    let key = source.next_required()?;
    if key.provenance != TokenProvenance::Ordinary || key.value != "dimensions" {
        return source.reject_line_as(key.line, "expected dimensions entry");
    }
    expect_structural(source, "[", "expected dimensions opener")?;
    let mut values = Vec::new();
    if values.try_reserve(7).is_err() {
        return reject(source, "dimensions allocation failed");
    }
    let closer_line = loop {
        let token = source.next_required()?;
        if structural(&token, "]") {
            break token.line;
        }
        if token.provenance != TokenProvenance::Ordinary
            || !token.value.parse::<f64>().is_ok_and(f64::is_finite)
        {
            return source.reject_line_as(token.line, "expected finite dimensions exponent");
        }
        if values.len() == 7 {
            return source.reject_line_as(
                token.line,
                "dimensions must contain exactly 5 or 7 exponents",
            );
        }
        values.push(token.value);
    };
    match values.len() {
        5 => {
            let mut current = String::new();
            if current.try_reserve(1).is_err() {
                return source.reject_line_as(closer_line, "dimensions allocation failed");
            }
            current.push('0');
            let mut luminous_intensity = String::new();
            if luminous_intensity.try_reserve(1).is_err() {
                return source.reject_line_as(closer_line, "dimensions allocation failed");
            }
            luminous_intensity.push('0');
            values.push(current);
            values.push(luminous_intensity);
        }
        7 => {}
        _ => {
            return source.reject_line_as(
                closer_line,
                "dimensions must contain exactly 5 or 7 exponents",
            );
        }
    }
    expect_current_structural(source, ";", "dimensions entry is missing a semicolon")?;
    Ok(values)
}

fn parse_boundary_field<R: BufRead>(
    source: &mut TokenSource<R>,
    policy: FieldLoadPolicy,
) -> Result<Vec<FieldBoundaryPatch>> {
    expect_structural(source, "{", "unexpected dictionary token")?;
    let mut patches = Vec::new();

    loop {
        #[derive(Clone, Copy)]
        enum Action {
            End,
            Patch,
            Invalid,
        }
        let action = match source.peek()? {
            Some(token) if structural(token, "}") => Action::End,
            Some(token) if token.provenance != TokenProvenance::Structural => Action::Patch,
            _ => Action::Invalid,
        };
        match action {
            Action::End => break,
            Action::Invalid => return reject(source, "invalid boundary patch header"),
            Action::Patch => {
                if patches.try_reserve(1).is_err() {
                    return reject(source, "boundary patch allocation failed");
                }
                let name = source.next_required()?;
                if !source.peek()?.is_some_and(|token| structural(token, "{")) {
                    return source.reject_line_as(name.line, "invalid boundary patch header");
                }
                source.next_required()?;
                let patch = parse_boundary_patch(source, name.value, policy)?;
                patches.push(patch);
            }
        }
    }
    expect_structural(source, "}", "unexpected dictionary token")?;

    Ok(patches)
}

fn parse_boundary_patch<R: BufRead>(
    source: &mut TokenSource<R>,
    name: String,
    policy: FieldLoadPolicy,
) -> Result<FieldBoundaryPatch> {
    let mut patch_type = None;
    let mut inlet_value = None;
    let mut value = None;

    loop {
        #[derive(Clone, Copy)]
        enum Action {
            End,
            Type,
            Value,
            InletValue,
            Other,
            Invalid,
        }
        let action = match source.peek()? {
            None => Action::Invalid,
            Some(token) if structural(token, "}") => Action::End,
            Some(token) if token.provenance == TokenProvenance::Structural => Action::Invalid,
            Some(token) if token.provenance == TokenProvenance::Ordinary => {
                match token.value.as_str() {
                    "type" => Action::Type,
                    "value" => Action::Value,
                    "inletValue" => Action::InletValue,
                    _ => Action::Other,
                }
            }
            Some(_) => Action::Other,
        };
        match action {
            Action::End => break,
            Action::Invalid => return reject(source, "invalid boundary patch entry"),
            Action::Type => {
                source.next_required()?;
                patch_type = Some(take_scalar_entry(source, "invalid patch type")?);
            }
            Action::Value => {
                source.next_required()?;
                value = Some(parse_field_value(source, policy)?);
            }
            Action::InletValue => {
                source.next_required()?;
                inlet_value = Some(parse_field_value(source, policy)?);
            }
            Action::Other => {
                source.next_required()?;
                source.discard_semicolon_terminated_value_or_block()?;
            }
        }
    }
    expect_structural(source, "}", "unexpected dictionary token")?;

    Ok(FieldBoundaryPatch {
        name,
        patch_type,
        inlet_value,
        value,
    })
}

fn structural(token: &Token, value: &str) -> bool {
    token.provenance == TokenProvenance::Structural && token.value == value
}
fn opener(token: &Token) -> bool {
    token.provenance == TokenProvenance::Structural
        && matches!(token.value.as_str(), "(" | "[" | "{")
}
fn closer(token: &Token) -> bool {
    token.provenance == TokenProvenance::Structural
        && matches!(token.value.as_str(), ")" | "]" | "}")
}
fn matching(open: char, close: &str) -> bool {
    matches!((open, close), ('(', ")") | ('[', "]") | ('{', "}"))
}
fn reject<R: BufRead, T>(source: &mut TokenSource<R>, detail: &'static str) -> Result<T> {
    source.reject_current_as(detail)
}

fn expect_structural<R: BufRead>(
    source: &mut TokenSource<R>,
    expected: &str,
    detail: &'static str,
) -> Result<()> {
    let token = source.next_required()?;
    if structural(&token, expected) {
        Ok(())
    } else {
        source.reject_line_as(token.line, detail)
    }
}

fn expect_current_structural<R: BufRead>(
    source: &mut TokenSource<R>,
    expected: &str,
    detail: &'static str,
) -> Result<()> {
    if !source
        .peek()?
        .is_some_and(|token| structural(token, expected))
    {
        return reject(source, detail);
    }
    source.next_required()?;
    Ok(())
}

fn take_scalar_entry<R: BufRead>(
    source: &mut TokenSource<R>,
    detail: &'static str,
) -> Result<String> {
    let token = source.next_required()?;
    if token.provenance != TokenProvenance::Ordinary
        || !source.peek()?.is_some_and(|next| structural(next, ";"))
    {
        return source.reject_line_as(token.line, detail);
    }
    source.next_required()?;
    Ok(token.value)
}

fn track_field_delimiter(
    token: &Token,
    stack: &mut [char; MAX_DICTIONARY_NESTING],
    depth: &mut usize,
) -> std::result::Result<(), &'static str> {
    if opener(token) {
        if *depth == MAX_DICTIONARY_NESTING {
            return Err("field value nesting limit exceeded");
        }
        stack[*depth] = token.value.as_bytes()[0] as char;
        *depth = (*depth)
            .checked_add(1)
            .ok_or("field value nesting counter overflow")?;
    } else if closer(token) {
        let top = (*depth)
            .checked_sub(1)
            .ok_or("unexpected field value closing delimiter")?;
        if !matching(stack[top], token.value.as_str()) {
            return Err("mismatched field value delimiter");
        }
        *depth = top;
    }
    Ok(())
}

fn append_joined<R: BufRead>(
    source: &mut TokenSource<R>,
    value: &mut String,
    token: Token,
) -> Result<()> {
    let additional = match token.value.len().checked_add(1) {
        Some(value) => value,
        None => return source.reject_line_as(token.line, "field value length overflow"),
    };
    let retained = match value.len().checked_add(additional) {
        Some(retained) => retained,
        None => return source.reject_line_as(token.line, "field value length overflow"),
    };
    if retained > MAX_RETAINED_FIELD_VALUE_BYTES {
        return source.reject_line_as(token.line, "retained field value exceeds byte limit");
    }
    if value.try_reserve(additional).is_err() {
        return source.reject_line_as(token.line, "field value allocation failed");
    }
    value.push(' ');
    value.push_str(&token.value);
    Ok(())
}

fn read_other_value<R: BufRead>(source: &mut TokenSource<R>) -> Result<String> {
    let first = source.next_required()?;
    if first.provenance == TokenProvenance::Structural && !opener(&first) {
        return source.reject_line_as(first.line, "invalid field value");
    }
    let mut stack = ['\0'; MAX_DICTIONARY_NESTING];
    let mut depth = 0usize;
    if let Err(detail) = track_field_delimiter(&first, &mut stack, &mut depth) {
        return source.reject_line_as(first.line, detail);
    }
    let mut value = first.value;
    loop {
        let token = source.next_required()?;
        if depth == 0 && structural(&token, ";") {
            return Ok(value);
        }
        if depth == 0 && closer(&token) {
            return source.reject_line_as(token.line, "field value is missing a semicolon");
        }
        if let Err(detail) = track_field_delimiter(&token, &mut stack, &mut depth) {
            return source.reject_line_as(token.line, detail);
        }
        append_joined(source, &mut value, token)?;
    }
}

fn read_uniform_value<R: BufRead>(source: &mut TokenSource<R>) -> Result<String> {
    let first = source.next_required()?;
    if structural(&first, ";") {
        return source.reject_line_as(first.line, "uniform value is missing");
    }
    if first.provenance == TokenProvenance::Structural && !opener(&first) {
        return source.reject_line_as(first.line, "invalid uniform value");
    }

    let first_is_opener = opener(&first);
    let first_open = if first_is_opener {
        Some(first.value.as_bytes()[0] as char)
    } else {
        None
    };
    let mut value = first.value;
    if let Some(first_open) = first_open {
        let mut stack = ['\0'; MAX_DICTIONARY_NESTING];
        let mut depth = 1usize;
        stack[0] = first_open;
        while depth != 0 {
            let token = source.next_required()?;
            if let Err(detail) = track_field_delimiter(&token, &mut stack, &mut depth) {
                return source.reject_line_as(token.line, detail);
            }
            append_joined(source, &mut value, token)?;
        }
    }

    if !source.peek()?.is_some_and(|token| structural(token, ";")) {
        return reject(source, "uniform value has trailing tokens");
    }
    source.next_required()?;
    Ok(value)
}

fn parse_field_value<R: BufRead>(
    source: &mut TokenSource<R>,
    policy: FieldLoadPolicy,
) -> Result<FieldValueSummary> {
    #[derive(Clone, Copy)]
    enum Action {
        Uniform,
        NonUniform,
        Other,
        Invalid,
    }
    let action = {
        match source.peek()? {
            Some(token)
                if token.provenance == TokenProvenance::Ordinary && token.value == "uniform" =>
            {
                Action::Uniform
            }
            Some(token)
                if token.provenance == TokenProvenance::Ordinary && token.value == "nonuniform" =>
            {
                Action::NonUniform
            }
            Some(token) if token.provenance == TokenProvenance::Structural => Action::Invalid,
            Some(_) => Action::Other,
            None => return reject(source, "field value is missing"),
        }
    };
    match action {
        Action::Invalid => reject(source, "invalid field value"),
        Action::Other => Ok(FieldValueSummary::Other(read_other_value(source)?)),
        Action::Uniform => {
            source.next_required()?;
            Ok(FieldValueSummary::Uniform(read_uniform_value(source)?))
        }
        Action::NonUniform => {
            source.next_required()?;
            parse_nonuniform(source, policy)
        }
    }
}

fn nonuniform_layout(count: usize, components: usize) -> Option<(usize, usize)> {
    let expected = count.checked_mul(components)?;
    let tuple_delimiters = if components == 3 {
        count.checked_mul(2)?
    } else {
        0
    };
    // This is a conservative physical-byte lower bound, not a token/byte unit
    // conversion: every required numeric token and structural delimiter has
    // at least one source byte. Whitespace and longer numerics only increase
    // the real byte requirement. Keeping the lower bound conservative also
    // preserves more specific grammar errors for malformed short lists.
    let minimum_encoded_bytes = expected.checked_add(tuple_delimiters)?.checked_add(2)?;
    Some((expected, minimum_encoded_bytes))
}

fn take_finite_numeric<R: BufRead>(source: &mut TokenSource<R>) -> Result<f64> {
    let token = source.next_required()?;
    if token.provenance != TokenProvenance::Ordinary {
        return source.reject_line_as(token.line, "invalid nonuniform numeric value");
    }
    match token.value.parse::<f64>() {
        Ok(value) if value.is_finite() => Ok(value),
        _ => source.reject_line_as(token.line, "invalid nonuniform numeric value"),
    }
}

fn parse_nonuniform<R: BufRead>(
    source: &mut TokenSource<R>,
    policy: FieldLoadPolicy,
) -> Result<FieldValueSummary> {
    let value_type = source.next_required()?;
    let components = if value_type.provenance == TokenProvenance::Ordinary {
        nonuniform_value_type_components(Some(value_type.value.as_str()))
    } else {
        None
    };
    let Some(components) = components else {
        return source.reject_line_as(value_type.line, "unsupported nonuniform value type");
    };

    let count_token = source.next_required()?;
    if count_token.provenance != TokenProvenance::Ordinary {
        return source.reject_line_as(count_token.line, "invalid nonuniform count");
    }
    let count = match count_token.value.parse::<usize>() {
        Ok(value) => value,
        Err(_) => return source.reject_line_as(count_token.line, "invalid nonuniform count"),
    };
    let (expected, minimum_encoded_bytes) = match nonuniform_layout(count, components) {
        Some(value) => value,
        None => return source.reject_line_as(count_token.line, "nonuniform count overflow"),
    };

    expect_structural(source, "(", "expected nonuniform list opener")?;
    if source.checked_remaining_bytes()? < minimum_encoded_bytes {
        return source.reject_line_as(count_token.line, "nonuniform count exceeds remaining input");
    }

    let mut values = match policy {
        FieldLoadPolicy::Summary => None,
        FieldLoadPolicy::Full => {
            let Some(retained_bytes) = expected.checked_mul(size_of::<f64>()) else {
                return source.reject_line_as(count_token.line, "nonuniform storage size overflow");
            };
            if retained_bytes > MAX_RETAINED_FIELD_VALUE_BYTES {
                return source.reject_line_as(
                    count_token.line,
                    "nonuniform retained value storage exceeds byte limit",
                );
            }
            let mut values = Vec::new();
            if values.try_reserve_exact(expected).is_err() {
                return source
                    .reject_line_as(count_token.line, "nonuniform value allocation failed");
            }
            Some(values)
        }
    };

    for _ in 0..count {
        if components == 3 {
            expect_structural(source, "(", "expected vector tuple opener")?;
        }
        for _ in 0..components {
            let value = take_finite_numeric(source)?;
            if let Some(values) = values.as_mut() {
                values.push(value);
            }
        }
        if components == 3 {
            expect_structural(source, ")", "expected vector tuple closer")?;
        }
    }
    expect_structural(source, ")", "expected nonuniform list closer")?;
    expect_current_structural(source, ";", "nonuniform value is missing a semicolon")?;

    Ok(FieldValueSummary::NonUniform {
        value_type: Some(value_type.value),
        count: Some(count),
        values,
    })
}

fn try_string(value: &str, detail: &'static str) -> Result<String> {
    let mut owned = String::new();
    owned
        .try_reserve(value.len())
        .map_err(|_| MeshError::Parse {
            line: 1,
            message: detail.to_owned(),
        })?;
    owned.push_str(value);
    Ok(owned)
}

fn field_path_error(path: &Path, detail: &'static str) -> Result<MeshError> {
    let rendered_path = path.to_str().unwrap_or("<non-UTF-8 initial field path>");
    let capacity = rendered_path
        .len()
        .checked_add(2)
        .and_then(|length| length.checked_add(detail.len()))
        .ok_or_else(|| MeshError::Io(io::ErrorKind::OutOfMemory.into()))?;
    let mut message = String::new();
    message
        .try_reserve(capacity)
        .map_err(|_| MeshError::Io(io::ErrorKind::OutOfMemory.into()))?;
    message.push_str(rendered_path);
    message.push_str(": ");
    message.push_str(detail);
    Ok(MeshError::Parse { line: 1, message })
}

fn path_io<T>(result: io::Result<T>, path: &Path, detail: &'static str) -> Result<T> {
    match result {
        Ok(value) => Ok(value),
        Err(_) => Err(field_path_error(path, detail)?),
    }
}

fn try_path_buf(path: &Path, detail: &'static str) -> Result<PathBuf> {
    let mut owned = PathBuf::new();
    owned
        .try_reserve(path.as_os_str().len())
        .map_err(|_| MeshError::Parse {
            line: 1,
            message: detail.to_owned(),
        })?;
    owned.push(path);
    Ok(owned)
}

struct FieldBoundaryValidator<'a> {
    case_dir: &'a Path,
    current_region: Option<RegionKey>,
    current_mesh: Option<Result<PolyMesh>>,
    warnings: Vec<String>,
}

#[derive(Debug, Eq, PartialEq)]
enum RegionKey {
    Base,
    Named(String),
}

impl RegionKey {
    fn matches(&self, region: Option<&str>) -> bool {
        match (self, region) {
            (Self::Base, None) => true,
            (Self::Named(current), Some(candidate)) => current == candidate,
            _ => false,
        }
    }

    fn from_region(region: Option<&str>) -> Result<Self> {
        match region {
            None => Ok(Self::Base),
            Some(region) => Ok(Self::Named(try_string(
                region,
                "field validation region allocation failed",
            )?)),
        }
    }
}

impl<'a> FieldBoundaryValidator<'a> {
    fn new(case_dir: &'a Path) -> Self {
        Self {
            case_dir,
            current_region: None,
            current_mesh: None,
            warnings: Vec::new(),
        }
    }

    fn validate_field(&mut self, field: &FieldFile) -> Result<()> {
        self.select_region(field.region.as_deref())?;
        let Some(Ok(mesh)) = self.current_mesh.as_ref() else {
            return Ok(());
        };
        validate_field_boundary_patches(field, mesh, &mut self.warnings)
    }

    fn select_region(&mut self, region: Option<&str>) -> Result<()> {
        if self
            .current_region
            .as_ref()
            .is_some_and(|current| current.matches(region))
        {
            return Ok(());
        }

        // Release the previous region before constructing or loading its successor.
        self.current_mesh = None;
        self.current_region = None;

        let key = RegionKey::from_region(region)?;
        let mesh_path = field_mesh_path(self.case_dir, region)?;
        let mesh = PolyMesh::read(&mesh_path);
        if mesh.is_err() {
            match region {
                Some(region) => push_warning_parts(
                    &mut self.warnings,
                    &[
                        "could not validate fields for region '",
                        region,
                        "': mesh unavailable",
                    ],
                )?,
                None => push_warning_parts(
                    &mut self.warnings,
                    &["could not validate fields for base mesh: mesh unavailable"],
                )?,
            }
        }
        self.current_region = Some(key);
        self.current_mesh = Some(mesh);
        Ok(())
    }
}

fn validate_canonical_field_order(fields: &[FieldFile]) -> Result<()> {
    for pair in fields.windows(2) {
        let left = &pair[0];
        let right = &pair[1];
        let ordering = left
            .region
            .as_deref()
            .cmp(&right.region.as_deref())
            .then(left.name.cmp(&right.name))
            .then(left.path.cmp(&right.path));
        if ordering == Ordering::Greater {
            return Err(MeshError::InvalidInput(try_string(
                "initial fields must be canonically sorted by region, name, and path",
                "field ordering error allocation failed",
            )?));
        }
    }
    Ok(())
}

fn field_mesh_path(case_dir: &Path, region: Option<&str>) -> Result<PathBuf> {
    let mut path = try_path_buf(case_dir, "field validation mesh path allocation failed")?;
    let region_len = region.map_or(0, str::len);
    let additional = "constant"
        .len()
        .checked_add("polyMesh".len())
        .and_then(|value| value.checked_add(region_len))
        .and_then(|value| value.checked_add(3))
        .ok_or_else(|| {
            MeshError::InvalidInput("field validation mesh path length overflow".to_owned())
        })?;
    path.try_reserve(additional).map_err(|_| {
        MeshError::InvalidInput("field validation mesh path allocation failed".to_owned())
    })?;
    path.push("constant");
    if let Some(region) = region {
        path.push(region);
    }
    path.push("polyMesh");
    Ok(path)
}

fn push_warning_parts(warnings: &mut Vec<String>, parts: &[&str]) -> Result<()> {
    let length = parts
        .iter()
        .try_fold(0usize, |length, part| length.checked_add(part.len()));
    let Some(length) = length else {
        return Err(MeshError::InvalidInput(
            "field validation warning length overflow".to_owned(),
        ));
    };
    warnings.try_reserve(1).map_err(|_| {
        MeshError::InvalidInput("field validation warning table allocation failed".to_owned())
    })?;
    let mut warning = String::new();
    warning.try_reserve(length).map_err(|_| {
        MeshError::InvalidInput("field validation warning allocation failed".to_owned())
    })?;
    for part in parts {
        warning.push_str(part);
    }
    warnings.push(warning);
    Ok(())
}

fn push_field_warning(warnings: &mut Vec<String>, field: &FieldFile, tail: &[&str]) -> Result<()> {
    match field.region.as_deref() {
        Some(region) => {
            let mut parts = Vec::new();
            parts
                .try_reserve(tail.len().checked_add(4).ok_or_else(|| {
                    MeshError::InvalidInput("field validation warning part overflow".to_owned())
                })?)
                .map_err(|_| {
                    MeshError::InvalidInput(
                        "field validation warning part allocation failed".to_owned(),
                    )
                })?;
            parts.extend_from_slice(&["field '", region, "/", field.name.as_str()]);
            parts.extend_from_slice(tail);
            push_warning_parts(warnings, &parts)
        }
        None => {
            let mut parts = Vec::new();
            parts
                .try_reserve(tail.len().checked_add(2).ok_or_else(|| {
                    MeshError::InvalidInput("field validation warning part overflow".to_owned())
                })?)
                .map_err(|_| {
                    MeshError::InvalidInput(
                        "field validation warning part allocation failed".to_owned(),
                    )
                })?;
            parts.extend_from_slice(&["field '", field.name.as_str()]);
            parts.extend_from_slice(tail);
            push_warning_parts(warnings, &parts)
        }
    }
}

fn validate_field_boundary_patches(
    field: &FieldFile,
    mesh: &PolyMesh,
    warnings: &mut Vec<String>,
) -> Result<()> {
    for (index, patch) in field.boundary_patches.iter().enumerate() {
        if field.boundary_patches[..index]
            .iter()
            .any(|seen| seen.name == patch.name)
        {
            push_field_warning(
                warnings,
                field,
                &["' has duplicate boundaryField entry '", &patch.name, "'"],
            )?;
        }
    }

    for patch in &mesh.patches {
        let Some(field_patch) = field
            .boundary_patches
            .iter()
            .find(|candidate| candidate.name == patch.name)
        else {
            push_field_warning(
                warnings,
                field,
                &[
                    "' is missing boundaryField entry for mesh patch '",
                    &patch.name,
                    "'",
                ],
            )?;
            continue;
        };

        validate_special_patch_field_type(field, field_patch, &patch.patch_type, warnings)?;
    }

    for field_patch in &field.boundary_patches {
        if !mesh
            .patches
            .iter()
            .any(|patch| patch.name == field_patch.name)
        {
            push_field_warning(
                warnings,
                field,
                &[
                    "' has boundaryField entry '",
                    &field_patch.name,
                    "' that is not a mesh patch",
                ],
            )?;
        }
    }
    Ok(())
}

fn validate_special_patch_field_type(
    field: &FieldFile,
    field_patch: &FieldBoundaryPatch,
    mesh_patch_type: &str,
    warnings: &mut Vec<String>,
) -> Result<()> {
    let expected = match mesh_patch_type {
        "empty" => "empty",
        "wedge" => "wedge",
        "symmetryPlane" => "symmetryPlane",
        _ => return Ok(()),
    };

    if field_patch.patch_type.as_deref() != Some(expected) {
        push_field_warning(
            warnings,
            field,
            &[
                "' patch '",
                &field_patch.name,
                "' should use boundary type '",
                expected,
                "' for mesh patch type '",
                mesh_patch_type,
                "', found '",
                field_patch.patch_type.as_deref().unwrap_or("missing"),
                "'",
            ],
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::io::{self, BufReader, Cursor};
    use std::path::{Path, PathBuf};
    use std::time::{SystemTime, UNIX_EPOCH};

    use crate::MeshError;
    use crate::Point3;
    use crate::dictionary::MAX_DICTIONARY_TOKENS;
    use crate::poly_mesh::{BoundaryPatch, PolyMesh};

    use super::{
        FieldBoundaryPatch, FieldFile, FieldLoadPolicy, FieldValueSummary, InitialFieldSet,
        MAX_DICTIONARY_NESTING, MAX_RETAINED_FIELD_VALUE_BYTES, nonuniform_layout,
        parse_field_file_reader, parse_field_file_str, parse_field_file_str_with_policy,
        read_field_file, read_fields_from_directory_with_policy, read_initial_fields_with_policy,
        validate_field_boundary_patches, validate_initial_field_boundaries,
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

        let nonuniform_content = r#"
        FoamFile { class volScalarField; object T; }
        internalField nonuniform List<scalar> 3 (300 310 320);
        boundaryField {}
        "#;
        let summary = parse_field_file_str_with_policy(
            nonuniform_content,
            Path::new("0/T"),
            Some("fluid".to_string()),
            FieldLoadPolicy::Summary,
        )
        .unwrap();
        assert!(matches!(
            summary.internal_field,
            Some(FieldValueSummary::NonUniform {
                value_type: Some(ref value_type),
                count: Some(3),
                values: None,
            }) if value_type == "List<scalar>"
        ));

        let buffered = parse_field_file_reader(
            BufReader::with_capacity(1, Cursor::new(nonuniform_content.as_bytes())),
            nonuniform_content.len(),
            Path::new("0/T"),
            Some("fluid".to_string()),
            FieldLoadPolicy::Full,
        )
        .unwrap();
        assert!(matches!(
            buffered.internal_field,
            Some(FieldValueSummary::NonUniform {
                values: Some(ref values),
                ..
            }) if values == &[300.0, 310.0, 320.0]
        ));

        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "ferrum-fields-streaming-{}-{stamp}",
            std::process::id()
        ));
        fs::write(&path, nonuniform_content).unwrap();
        let from_file = read_field_file(&path, Some("fluid"), FieldLoadPolicy::Full).unwrap();
        fs::remove_file(&path).unwrap();
        assert!(matches!(
            from_file.internal_field,
            Some(FieldValueSummary::NonUniform {
                values: Some(ref values),
                ..
            }) if values == &[300.0, 310.0, 320.0]
        ));
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

        let nonuniform_boundary = r#"
        FoamFile { class volVectorField; object U; }
        internalField uniform (0 0 0);
        boundaryField
        {
            outlet
            {
                type inletOutlet;
                inletValue nonuniform List<vector> 1 ((1 0 0));
                value nonuniform List<vector> 1 ((0 0 0));
            }
        }
        "#;
        let summary = parse_field_file_str_with_policy(
            nonuniform_boundary,
            Path::new("0/U"),
            None,
            FieldLoadPolicy::Summary,
        )
        .unwrap();
        let patch = &summary.boundary_patches[0];
        assert!(matches!(
            patch.inlet_value,
            Some(FieldValueSummary::NonUniform { values: None, .. })
        ));
        assert!(matches!(
            patch.value,
            Some(FieldValueSummary::NonUniform { values: None, .. })
        ));

        let malformed = r#"
        FoamFile { class volVectorField; object U; }
        internalField uniform (0 0 0);
        boundaryField
        {
            outlet
            {
                type inletOutlet;
                inletValue nonuniform List<vector> 2 ((1 0 0));
            }
        }
        "#;
        let summary_error = parse_field_file_str_with_policy(
            malformed,
            Path::new("0/U"),
            None,
            FieldLoadPolicy::Summary,
        )
        .unwrap_err();
        let full_error = parse_field_file_str_with_policy(
            malformed,
            Path::new("0/U"),
            None,
            FieldLoadPolicy::Full,
        )
        .unwrap_err();
        assert_eq!(summary_error.to_string(), full_error.to_string());
    }

    #[test]
    fn unknown_openfoam_boundary_entries_are_ignored_under_both_load_policies() {
        let content = r#"
        FoamFile { class volScalarField; object p; }
        dimensions [0 2 -2 0 0 0 0];
        internalField uniform 0;
        boundaryField
        {
            fixedGradientPatch
            {
                type fixedGradient;
                gradient uniform 0;
                value uniform 11;
            }
            mixedPatch
            {
                type mixed;
                refValue uniform 0;
                refGradient uniform 0;
                valueFraction uniform 1;
                inletValue uniform 2;
                value uniform 3;
            }
            totalPressurePatch
            {
                type totalPressure;
                p0 uniform 1e5;
                value uniform 4;
            }
            nonuniformAuxiliaryPatch
            {
                type custom;
                profile nonuniform List<scalar> 2 (1 2);
                value uniform 5;
            }
            quotedKeyPatch
            {
                type custom;
                "value" uniform 99;
                value uniform 6;
            }
            provenancePatch
            {
                type custom;
                ignored innocent value uniform 88;
                quotedTerminator alpha ";" beta;
                value uniform 7;
            }
        }
        "#;

        for policy in [FieldLoadPolicy::Summary, FieldLoadPolicy::Full] {
            let field =
                parse_field_file_str_with_policy(content, Path::new("0/p"), None, policy).unwrap();
            assert_eq!(field.boundary_patches.len(), 6);
            let expected = [
                ("fixedGradient", "11", None),
                ("mixed", "3", Some("2")),
                ("totalPressure", "4", None),
                ("custom", "5", None),
                ("custom", "6", None),
                ("custom", "7", None),
            ];
            for (patch, (patch_type, value, inlet_value)) in
                field.boundary_patches.iter().zip(expected)
            {
                assert_eq!(patch.patch_type.as_deref(), Some(patch_type));
                assert!(matches!(
                    patch.value,
                    Some(FieldValueSummary::Uniform(ref actual)) if actual == value
                ));
                match inlet_value {
                    Some(expected) => assert!(matches!(
                        patch.inlet_value,
                        Some(FieldValueSummary::Uniform(ref actual)) if actual == expected
                    )),
                    None => assert!(patch.inlet_value.is_none()),
                }
            }
        }

        assert_parse_error(
            "FoamFile { class volScalarField; object p; }\n\
             internalField uniform 0;\n\
             boundaryField { outlet { type fixedGradient; gradient uniform 0 } }",
            3,
            "dictionary value is missing a semicolon",
        );
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

        let summary = parse_field_file_str_with_policy(
            content,
            Path::new("0/U"),
            None,
            FieldLoadPolicy::Summary,
        )
        .unwrap();
        assert!(matches!(
            summary.internal_field,
            Some(FieldValueSummary::NonUniform {
                value_type: Some(ref value_type),
                count: Some(3),
                values: None,
            }) if value_type == "List<scalar>"
        ));
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
    fn parses_all_supported_nonuniform_type_aliases() {
        let cases = [
            ("volScalarField", "p", "List<scalar>", "1 2", 2),
            ("volScalarField", "p", "scalarField", "1 2", 2),
            ("volScalarField", "p", "Field<scalar>", "1 2", 2),
            ("volVectorField", "U", "List<vector>", "(1 2 3) (4 5 6)", 6),
            ("volVectorField", "U", "vectorField", "(1 2 3) (4 5 6)", 6),
            ("volVectorField", "U", "Field<vector>", "(1 2 3) (4 5 6)", 6),
        ];

        for (class_name, object, value_type, values, scalar_count) in cases {
            let content = format!(
                "FoamFile {{ class {class_name}; object {object}; }}\n\
                 internalField nonuniform {value_type} 2 ({values});\n\
                 boundaryField {{}}\n"
            );
            let path = Path::new("0/field");
            let full =
                parse_field_file_str_with_policy(&content, path, None, FieldLoadPolicy::Full)
                    .unwrap();
            let summary =
                parse_field_file_str_with_policy(&content, path, None, FieldLoadPolicy::Summary)
                    .unwrap();

            assert!(matches!(
                full.internal_field,
                Some(FieldValueSummary::NonUniform {
                    value_type: Some(ref parsed_type),
                    count: Some(2),
                    values: Some(ref parsed_values),
                }) if parsed_type == value_type && parsed_values.len() == scalar_count
            ));
            assert!(matches!(
                summary.internal_field,
                Some(FieldValueSummary::NonUniform {
                    value_type: Some(ref parsed_type),
                    count: Some(2),
                    values: None,
                }) if parsed_type == value_type
            ));
        }

        for unsupported in [
            r#""scalarField""#,
            r#""Field<scalar>""#,
            "List<label>",
            "List<tensor>",
            "Field<scalar",
        ] {
            let content = format!(
                "FoamFile {{ class volScalarField; object p; }}\n\
                 internalField nonuniform {unsupported} 1 (7);\n\
                 boundaryField {{}}\n"
            );
            assert_parse_error(&content, 2, "unsupported nonuniform value type");
        }
    }

    #[test]
    fn field_directives_fail_closed_without_activating_quoted_spellings() {
        for directive in [
            "#include \"initialConditions\"",
            "#includeFunc residuals",
            "#include \"initialConditions\";",
            "#includeFunc residuals;",
        ] {
            let content = format!(
                "FoamFile {{ class volScalarField; object p; }}\n\
                 {directive}\n\
                 internalField uniform 0;\n\
                 boundaryField {{}}\n"
            );
            assert_parse_error(&content, 2, "unsupported dictionary directive");
        }

        for quoted in [r##""#include" inert;"##, r##""#includeFunc" inert;"##] {
            let content = format!(
                "FoamFile {{ class volScalarField; object p; }}\n\
                 {quoted}\n\
                 internalField uniform 0;\n\
                 boundaryField {{}}\n"
            );
            let field = parse_field_file_str(&content, Path::new("0/p"), None)
                .expect("quoted directive spelling must remain inert data");
            assert_eq!(field.name, "p");
        }

        assert_parse_error(
            "FoamFile { class volScalarField; object p; }\n\
             unknown { #include \"initialConditions\"; }\n\
             internalField uniform 0;\n\
             boundaryField {}\n",
            2,
            "unsupported dictionary directive",
        );
    }

    #[test]
    fn rejects_nonuniform_numeric_tail_beyond_declared_count() {
        let content = r#"
        FoamFile { class volScalarField; object p; }
        internalField nonuniform List<scalar> 1 ( 1 2 3 );
        boundaryField { }
        "#;

        let error = parse_field_file_str(content, Path::new("0/p"), None).unwrap_err();
        assert_eq!(
            error.to_string(),
            "line 3: 0/p: expected nonuniform list closer"
        );
    }

    #[test]
    fn field_exact_one_and_provenance_are_safe() {
        let content = r#"FoamFile
{
    "class" inert;
    class volScalarField;
    "object" inert;
    object p;
}
"FoamFile" ignored;
unknownScalar inert;
unknownGroup (alpha [beta {gamma}]);
unknownBlock { nested inert; }
"internalField" ignored;
internalField "uniform";
"boundaryField" ignored;
boundaryField
{
    inlet
    {
        "type" inert;
        type fixedValue;
        "value" inert;
        value "uniform";
    }
}
"#;

        let field = parse_field_file_str(content, Path::new("0/p"), None).unwrap();
        assert_eq!(field.class_name.as_deref(), Some("volScalarField"));
        assert_eq!(field.name, "p");
        assert_eq!(field.boundary_patches.len(), 1);
        assert_eq!(
            field.boundary_patches[0].patch_type.as_deref(),
            Some("fixedValue")
        );
        assert!(matches!(
            field.boundary_patches[0].value,
            Some(FieldValueSummary::Other(ref value)) if value == "uniform"
        ));
        assert!(matches!(
            field.internal_field,
            Some(FieldValueSummary::Other(ref value)) if value == "uniform"
        ));

        assert_parse_error(
            ";\nFoamFile { class volScalarField; object p; }",
            1,
            "structural token cannot be a field key",
        );
        assert_parse_error(
            "FoamFile { class volScalarField; object p; }\nunknown inert\ninternalField uniform 0;\n",
            3,
            "dictionary value is missing a semicolon",
        );
        assert_parse_error(
            "FoamFile { class volScalarField; object p; }\nunknown (inert)\ninternalField uniform 0;\n",
            3,
            "dictionary value is missing a semicolon",
        );
        assert_parse_error(
            "FoamFile { class volScalarField; object p; }\n#includeFunc residuals\ninternalField uniform 0;\n",
            2,
            "unsupported dictionary directive",
        );

        let repetitions = MAX_DICTIONARY_TOKENS / 3 + 1;
        assert!(repetitions * 3 > MAX_DICTIONARY_TOKENS);
        let mut many = String::from("FoamFile { class volScalarField; object p; }\n");
        many.push_str(&"x 0;\n".repeat(repetitions));
        many.push_str("internalField uniform 0;\nboundaryField {}\n");
        let streamed = parse_field_file_str_with_policy(
            &many,
            Path::new("0/p"),
            None,
            FieldLoadPolicy::Summary,
        )
        .unwrap();
        assert!(matches!(
            streamed.internal_field,
            Some(FieldValueSummary::Uniform(ref value)) if value == "0"
        ));
        assert!(streamed.boundary_patches.is_empty());
    }

    #[test]
    fn field_value_planning_rejects_exact_offender_atomically() {
        assert_parse_error(
            "FoamFile { class volScalarField; object p; }\ninternalField uniform;\n",
            2,
            "uniform value is missing",
        );
        assert_parse_error(
            "FoamFile { class volVectorField; object U; }\ninternalField uniform\n(1 2 3)\ntrailing;\n",
            4,
            "uniform value has trailing tokens",
        );
        assert_parse_error(
            "FoamFile { class volVectorField; object U; }\ninternalField uniform (1 2];\n",
            2,
            "mismatched dictionary delimiter",
        );

        let mut too_deep = String::from("internalField uniform ");
        too_deep.push_str(&"(".repeat(MAX_DICTIONARY_NESTING + 1));
        too_deep.push('0');
        too_deep.push_str(&")".repeat(MAX_DICTIONARY_NESTING + 1));
        too_deep.push(';');
        assert_parse_error(&too_deep, 1, "dictionary nesting limit exceeded");

        let malformed = "FoamFile { class volVectorField; object U; }\ninternalField uniform\n(1 2 3)\ntrailing;\n";
        let string_error = parse_field_file_str(malformed, Path::new("0/U"), None).unwrap_err();
        let buffered_error = parse_field_file_reader(
            BufReader::with_capacity(1, Cursor::new(malformed.as_bytes())),
            malformed.len(),
            Path::new("0/U"),
            None,
            FieldLoadPolicy::Full,
        )
        .unwrap_err();
        assert_eq!(buffered_error.to_string(), string_error.to_string());

        let retained_tokens = MAX_RETAINED_FIELD_VALUE_BYTES / 2 + 1;
        let mut retained_flood =
            String::from("FoamFile { class volScalarField; object p; }\ninternalField ");
        retained_flood.push_str(&"a ".repeat(retained_tokens));
        retained_flood.push_str(";\n");
        assert_parse_error(
            &retained_flood,
            2,
            "retained field value exceeds byte limit",
        );
    }

    #[test]
    fn nonuniform_values_are_capped_and_fail_closed() {
        let zero = parse_field_file_str(
            "FoamFile { class volScalarField; object p; }\ninternalField nonuniform List<scalar> 0 ();\nboundaryField {}\n",
            Path::new("0/p"),
            None,
        )
        .unwrap();
        assert!(matches!(
            zero.internal_field,
            Some(FieldValueSummary::NonUniform {
                count: Some(0),
                values: Some(ref values),
                ..
            }) if values.is_empty()
        ));

        assert_parse_error(
            "FoamFile { class volScalarField; object p; }\ninternalField nonuniform List<scalar> 1 (NaN);\n",
            2,
            "invalid nonuniform numeric value",
        );
        assert_parse_error(
            "FoamFile { class volScalarField; object p; }\ninternalField nonuniform;\n",
            2,
            "unsupported nonuniform value type",
        );
        assert_parse_error(
            "FoamFile { class volScalarField; object p; }\ninternalField nonuniform List<scalar> \"1\" (1);\n",
            2,
            "invalid nonuniform count",
        );
        assert_parse_error(
            "FoamFile { class volVectorField; object U; }\ninternalField nonuniform List<vector> 1 (1 2 3);\n",
            2,
            "expected vector tuple opener",
        );
        assert_parse_error(
            "FoamFile { class volVectorField; object U; }\ninternalField nonuniform List<vector> 1 ((1 2));\n",
            2,
            "invalid nonuniform numeric value",
        );
        assert_parse_error(
            "FoamFile { class volVectorField; object U; }\ninternalField nonuniform List<vector> 1 ((1 2 3 4));\n",
            2,
            "expected vector tuple closer",
        );
        assert_parse_error(
            "FoamFile { class volVectorField; object U; }\ninternalField nonuniform List<vector> 1 (((1 2 3)));\n",
            2,
            "invalid nonuniform numeric value",
        );
        let overflow = format!(
            "FoamFile {{ class volVectorField; object U; }}\ninternalField nonuniform List<vector> {} ();\n",
            usize::MAX
        );
        assert_parse_error(&overflow, 2, "nonuniform count overflow");

        let count = 1_000_001usize;
        assert_eq!(nonuniform_layout(count, 1), Some((count, count + 2)));
        let oversized = format!(
            "FoamFile {{ class volScalarField; object p; }}\ninternalField nonuniform List<scalar> {count} ();\n"
        );
        assert_parse_error(&oversized, 2, "nonuniform count exceeds remaining input");

        for exact in [
            "FoamFile { class volScalarField; object p; }\ninternalField nonuniform List<scalar> 1 (0);",
            "FoamFile { class volVectorField; object U; }\ninternalField nonuniform List<vector> 1 ((0 0 0));",
        ] {
            parse_field_file_str(exact, Path::new("0/exact"), None)
                .expect("compact exact-fit nonuniform list must parse");
        }
        for impossible in [
            "FoamFile { class volScalarField; object p; }\ninternalField nonuniform List<scalar> 2 (0);",
            "FoamFile { class volVectorField; object U; }\ninternalField nonuniform List<vector> 2 ((0 0 0));",
        ] {
            assert_parse_error(impossible, 2, "nonuniform count exceeds remaining input");
        }
        let retained_limit_count = MAX_RETAINED_FIELD_VALUE_BYTES / size_of::<f64>();
        let over_retained_limit = retained_limit_count + 1;
        let over_retained_values = "0 ".repeat(over_retained_limit);
        let over_retained = format!(
            "FoamFile {{ class volScalarField; object p; }}\ninternalField nonuniform List<scalar> {over_retained_limit} ({over_retained_values});\n"
        );
        assert_parse_error(
            &over_retained,
            2,
            "nonuniform retained value storage exceeds byte limit",
        );
        let summary = parse_field_file_str_with_policy(
            &over_retained,
            Path::new("0/p"),
            None,
            FieldLoadPolicy::Summary,
        )
        .unwrap();
        assert!(matches!(
            summary.internal_field,
            Some(FieldValueSummary::NonUniform { values: None, .. })
        ));

        assert_eq!(nonuniform_layout(usize::MAX, 3), None);
    }

    #[test]
    fn dimensions_accept_exactly_five_exponents_and_normalize_to_seven() {
        let field = parse_field_file_str(
            "FoamFile { class volScalarField; object p; }\n\
             dimensions [0 1 -1 0 0];\n\
             internalField uniform 0;\n\
             boundaryField {}\n",
            Path::new("0/p"),
            None,
        )
        .unwrap();

        assert_eq!(
            field.dimensions,
            Some(vec![
                "0".to_string(),
                "1".to_string(),
                "-1".to_string(),
                "0".to_string(),
                "0".to_string(),
                "0".to_string(),
                "0".to_string(),
            ])
        );
    }

    #[test]
    fn dimensions_accept_exactly_seven_exponents() {
        let field = parse_field_file_str(
            "FoamFile { class volScalarField; object p; }\n\
             dimensions [0 1 -1 0 0 2 -2];\n\
             internalField uniform 0;\n\
             boundaryField {}\n",
            Path::new("0/p"),
            None,
        )
        .unwrap();

        assert_eq!(
            field.dimensions,
            Some(vec![
                "0".to_string(),
                "1".to_string(),
                "-1".to_string(),
                "0".to_string(),
                "0".to_string(),
                "2".to_string(),
                "-2".to_string(),
            ])
        );
    }

    #[test]
    fn dimensions_reject_four_six_and_eight_exponents_without_accepting_a_prefix() {
        assert_parse_error(
            "FoamFile { class volScalarField; object p; }\n\
             dimensions [0 1 -1 0];\n\
             internalField uniform 8;\n\
             boundaryField {}\n",
            2,
            "dimensions must contain exactly 5 or 7 exponents",
        );
        assert_parse_error(
            "FoamFile { class volScalarField; object p; }\n\
             dimensions [0 1 -1 0 0 0];\n\
             internalField uniform 9;\n\
             boundaryField {}\n",
            2,
            "dimensions must contain exactly 5 or 7 exponents",
        );
        assert_parse_error(
            "FoamFile { class volScalarField; object p; }\n\
             dimensions [0 1 -1 0 0 0 0 8];\n\
             internalField uniform 10;\n\
             boundaryField {}\n",
            2,
            "dimensions must contain exactly 5 or 7 exponents",
        );
    }

    #[test]
    fn dimensions_reject_quoted_exponents() {
        assert_parse_error(
            "FoamFile { class volScalarField; object p; }\n\
             dimensions [0 1 \"-1\" 0 0];\n",
            2,
            "expected finite dimensions exponent",
        );
    }

    #[test]
    fn dimensions_reject_non_finite_exponents() {
        assert_parse_error(
            "FoamFile { class volScalarField; object p; }\n\
             dimensions [0 1 NaN 0 0];\n",
            2,
            "expected finite dimensions exponent",
        );
    }

    #[test]
    fn dimensions_patches_and_mesh_cache_fail_atomically() {
        let quoted = parse_field_file_str(
            "FoamFile { class volScalarField; object p; }\n\"dimensions\" ignored;\ninternalField uniform 0;\nboundaryField {}\n",
            Path::new("0/p"),
            None,
        )
        .unwrap();
        assert_eq!(quoted.dimensions, None);

        assert_parse_error(
            "FoamFile { class volScalarField; object p; }\ndimensions [0 1 nope 0 0 0 0];\n",
            2,
            "expected finite dimensions exponent",
        );
        assert_parse_error(
            "FoamFile { class volScalarField; object p; }\ndimensions [0 1 2\n3 4 5\n];\n",
            4,
            "dimensions must contain exactly 5 or 7 exponents",
        );
        assert_parse_error(
            "FoamFile { class volScalarField; object p; }\ndimensions [0 1 2 3 4 5 6]\ninternalField uniform 0;\n",
            3,
            "dimensions entry is missing a semicolon",
        );
        assert_parse_error(
            "FoamFile { class volScalarField; object p; }\nboundaryField { { type fixedValue; } }\n",
            2,
            "invalid boundary patch header",
        );

        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let missing_case = std::env::temp_dir().join(format!(
            "ferrum-fields-missing-mesh-{}-{stamp}",
            std::process::id()
        ));
        let base_p = test_field(Vec::new());
        let mut base_q = test_field(Vec::new());
        base_q.name = "q".to_string();
        base_q.path = PathBuf::from("0/q");
        let mut region_p = test_field(Vec::new());
        region_p.region = Some("fluid".to_string());
        region_p.path = PathBuf::from("0/fluid/p");
        let fields = InitialFieldSet {
            case_dir: missing_case.clone(),
            fields: vec![base_p, base_q, region_p],
        };
        let summary = validate_initial_field_boundaries(&missing_case, &fields).unwrap();
        assert_eq!(summary.fields, 3);
        assert_eq!(summary.warnings.len(), 2);
        assert!(summary.warnings[0].contains("base mesh"));
        assert!(summary.warnings[1].contains("region 'fluid'"));

        let mut later = test_field(Vec::new());
        later.region = Some("zeta".to_string());
        let mut earlier = test_field(Vec::new());
        earlier.region = Some("alpha".to_string());
        let unsorted = InitialFieldSet {
            case_dir: missing_case.clone(),
            fields: vec![later, earlier],
        };
        let error = validate_initial_field_boundaries(&missing_case, &unsorted).unwrap_err();
        assert_eq!(
            error.to_string(),
            "initial fields must be canonically sorted by region, name, and path"
        );

        let io_root = std::env::temp_dir().join(format!(
            "ferrum-fields-io-contract-{}-{stamp}",
            std::process::id()
        ));
        fs::create_dir_all(&io_root).unwrap();

        let missing_file = io_root.join("missing-field");
        let error = read_field_file(&missing_file, None, FieldLoadPolicy::Summary).unwrap_err();
        assert_path_error(error, &missing_file, "could not open initial field file");

        let missing_directory = io_root.join("missing-directory");
        let error = read_fields_from_directory_with_policy(
            &io_root,
            &missing_directory,
            FieldLoadPolicy::Summary,
        )
        .unwrap_err();
        assert_path_error(
            error,
            &missing_directory,
            "could not inspect initial field directory",
        );

        let invalid_case = io_root.join("invalid-case");
        fs::create_dir_all(&invalid_case).unwrap();
        let invalid_zero = invalid_case.join("0");
        fs::write(&invalid_zero, b"not a directory").unwrap();
        let error =
            read_initial_fields_with_policy(&invalid_case, FieldLoadPolicy::Summary).unwrap_err();
        assert_path_error(
            error,
            &invalid_zero,
            "initial field directory must be a real directory, not a symlink",
        );

        let symlink_case = io_root.join("symlink-case");
        let symlink_target = io_root.join("symlink-target");
        fs::create_dir_all(&symlink_case).unwrap();
        fs::create_dir_all(&symlink_target).unwrap();
        let symlink_zero = symlink_case.join("0");
        if create_directory_symlink(&symlink_target, &symlink_zero).is_ok() {
            let error = read_initial_fields_with_policy(&symlink_case, FieldLoadPolicy::Summary)
                .unwrap_err();
            assert_path_error(
                error,
                &symlink_zero,
                "initial field directory must be a real directory, not a symlink",
            );
        }

        fs::remove_dir_all(&io_root).unwrap();
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

        validate_field_boundary_patches(&field, &mesh, &mut warnings).unwrap();

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

        validate_field_boundary_patches(&field, &mesh, &mut warnings).unwrap();

        assert_eq!(warnings.len(), 3);
    }

    fn assert_parse_error(content: &str, line: usize, detail: &str) {
        let error = parse_field_file_str(content, Path::new("0/p"), None).unwrap_err();
        match error {
            MeshError::Parse {
                line: actual_line,
                message,
            } => {
                assert_eq!(actual_line, line);
                assert_eq!(message, format!("0/p: {detail}"));
                assert_eq!(message.matches("0/p").count(), 1);
            }
            other => panic!("expected parse error, got {other}"),
        }
    }

    fn assert_path_error(error: MeshError, path: &Path, detail: &str) {
        match error {
            MeshError::Parse { line, message } => {
                let rendered = path.to_str().unwrap();
                assert_eq!(line, 1);
                assert_eq!(message, format!("{rendered}: {detail}"));
                assert_eq!(message.matches(rendered).count(), 1);
            }
            other => panic!("expected path-aware parse error, got {other}"),
        }
    }

    #[cfg(unix)]
    fn create_directory_symlink(target: &Path, link: &Path) -> io::Result<()> {
        std::os::unix::fs::symlink(target, link)
    }

    #[cfg(windows)]
    fn create_directory_symlink(target: &Path, link: &Path) -> io::Result<()> {
        std::os::windows::fs::symlink_dir(target, link)
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
