use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

use crate::dictionary::{
    MAX_DICTIONARY_NESTING, MAX_DICTIONARY_PAYLOAD_BYTES, MAX_DICTIONARY_TOKENS, Token,
    TokenCursor, TokenProvenance, tokenize,
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
    let mut cursor = tokenize(path, content)?.into_cursor();
    let mut class_name = None;
    let mut object_name = None;
    let mut dimensions = None;
    let mut internal_field = None;
    let mut boundary_patches = Vec::new();

    while let Some(token) = cursor.peek()? {
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
                cursor.next_required()?;
                let metadata = parse_foam_file(&mut cursor)?;
                class_name = metadata.class_name;
                object_name = metadata.object_name;
            }
            Action::Dimensions => dimensions = Some(parse_dimensions(&mut cursor)?),
            Action::InternalField => {
                cursor.next_required()?;
                internal_field = Some(parse_field_value(&mut cursor)?);
            }
            Action::BoundaryField => {
                cursor.next_required()?;
                boundary_patches = parse_boundary_field(&mut cursor)?;
            }
            Action::Other => {
                cursor.next_required()?;
                skip_exact_one(&mut cursor)?;
            }
            Action::Invalid => {
                return cursor.reject_current_as("structural token cannot be a field key");
            }
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

    loop {
        #[derive(Clone, Copy)]
        enum Action {
            End,
            Class,
            Object,
            Other,
            Invalid,
        }
        let action = match cursor.peek()? {
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
            Action::Invalid => return reject(cursor, "invalid FoamFile entry"),
            Action::Class | Action::Object => {
                cursor.next_required()?;
                let scalar = take_scalar_entry(cursor, "invalid FoamFile scalar entry")?;
                if matches!(action, Action::Class) {
                    class_name = Some(scalar);
                } else {
                    object_name = Some(scalar);
                }
            }
            Action::Other => {
                cursor.next_required()?;
                skip_exact_one(cursor)?;
            }
        }
    }
    cursor.expect("}")?;

    Ok(FoamFileMetadata {
        class_name,
        object_name,
    })
}

fn parse_dimensions(cursor: &mut TokenCursor) -> Result<Vec<String>> {
    {
        let tokens = cursor.remaining_tokens()?;
        if tokens.first().is_none_or(|token| {
            token.provenance != TokenProvenance::Ordinary || token.value != "dimensions"
        }) {
            return reject_at(cursor, 0, "expected dimensions entry");
        }
        if tokens.get(1).is_none_or(|token| !structural(token, "[")) {
            return reject_at(cursor, 1, "expected dimensions opener");
        }
        for index in 2..9 {
            let valid = tokens.get(index).is_some_and(|token| {
                token.provenance == TokenProvenance::Ordinary
                    && token.value.parse::<f64>().is_ok_and(f64::is_finite)
            });
            if !valid {
                return reject_at(cursor, index, "expected finite dimensions exponent");
            }
        }
        if tokens.get(9).is_none_or(|token| !structural(token, "]")) {
            return reject_at(cursor, 9, "expected dimensions closer");
        }
        if tokens.get(10).is_none_or(|token| !structural(token, ";")) {
            return reject_at(cursor, 10, "dimensions entry is missing a semicolon");
        }
    }

    cursor.next_required()?;
    cursor.next_required()?;
    let mut values = Vec::new();
    if values.try_reserve(7).is_err() {
        return reject(cursor, "dimensions allocation failed");
    }
    for _ in 0..7 {
        values.push(cursor.next_required()?.value);
    }
    cursor.next_required()?;
    cursor.next_required()?;
    Ok(values)
}

fn parse_boundary_field(cursor: &mut TokenCursor) -> Result<Vec<FieldBoundaryPatch>> {
    cursor.expect("{")?;
    let mut patches = Vec::new();

    loop {
        #[derive(Clone, Copy)]
        enum Action {
            End,
            Patch,
            Invalid,
        }
        let action = match cursor.remaining_tokens()? {
            [token, ..] if structural(token, "}") => Action::End,
            [name, open, ..]
                if name.provenance != TokenProvenance::Structural && structural(open, "{") =>
            {
                Action::Patch
            }
            _ => Action::Invalid,
        };
        match action {
            Action::End => break,
            Action::Invalid => return reject(cursor, "invalid boundary patch header"),
            Action::Patch => {
                if patches.try_reserve(1).is_err() {
                    return reject(cursor, "boundary patch allocation failed");
                }
                let name = cursor.next_required()?.value;
                cursor.expect("{")?;
                let patch = parse_boundary_patch(cursor, name)?;
                patches.push(patch);
            }
        }
    }
    cursor.expect("}")?;

    Ok(patches)
}

fn parse_boundary_patch(cursor: &mut TokenCursor, name: String) -> Result<FieldBoundaryPatch> {
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
        let action = match cursor.peek()? {
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
            Action::Invalid => return reject(cursor, "invalid boundary patch entry"),
            Action::Type => {
                cursor.next_required()?;
                patch_type = Some(take_scalar_entry(cursor, "invalid patch type")?);
            }
            Action::Value => {
                cursor.next_required()?;
                value = Some(parse_field_value(cursor)?);
            }
            Action::InletValue => {
                cursor.next_required()?;
                inlet_value = Some(parse_field_value(cursor)?);
            }
            Action::Other => {
                cursor.next_required()?;
                skip_exact_one(cursor)?;
            }
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
fn reject<T>(cursor: &mut TokenCursor, detail: &'static str) -> Result<T> {
    cursor.reject_current_as(detail)
}
fn reject_at<T>(cursor: &mut TokenCursor, index: usize, detail: &'static str) -> Result<T> {
    cursor.reject_at_as(index, detail)
}

fn balanced_end(
    tokens: &[Token],
    start: usize,
) -> std::result::Result<usize, (usize, &'static str)> {
    let mut stack = ['\0'; MAX_DICTIONARY_NESTING];
    let mut depth = 0usize;
    let mut index = start;
    loop {
        let token = tokens
            .get(index)
            .ok_or((index, "unterminated dictionary group"))?;
        if opener(token) {
            if depth == MAX_DICTIONARY_NESTING {
                return Err((index, "dictionary nesting limit exceeded"));
            }
            stack[depth] = token.value.as_bytes()[0] as char;
            depth = depth
                .checked_add(1)
                .ok_or((index, "dictionary nesting counter overflow"))?;
        } else if closer(token) {
            let top = depth
                .checked_sub(1)
                .ok_or((index, "unexpected dictionary closing delimiter"))?;
            if !matching(stack[top], token.value.as_str()) {
                return Err((index, "mismatched dictionary delimiter"));
            }
            depth = top;
            if depth == 0 {
                return index
                    .checked_add(1)
                    .ok_or((index, "dictionary index overflow"));
            }
        }
        index = index
            .checked_add(1)
            .ok_or((index, "dictionary index overflow"))?;
    }
}

fn skip_exact_one(cursor: &mut TokenCursor) -> Result<()> {
    cursor.skip_exact_value_or_block()
}

fn take_scalar_entry(cursor: &mut TokenCursor, detail: &'static str) -> Result<String> {
    let tokens = cursor.remaining_tokens()?;
    if tokens.len() < 2
        || tokens[0].provenance != TokenProvenance::Ordinary
        || !structural(&tokens[1], ";")
    {
        return reject(cursor, detail);
    }
    let value = cursor.next_required()?.value;
    cursor.next_required()?;
    Ok(value)
}

fn value_end(tokens: &[Token]) -> std::result::Result<usize, (usize, &'static str)> {
    let mut stack = ['\0'; MAX_DICTIONARY_NESTING];
    let mut depth = 0usize;
    for (index, token) in tokens.iter().enumerate() {
        if depth == 0 && structural(token, ";") {
            return if index == 0 {
                Err((0, "field value is missing"))
            } else {
                Ok(index)
            };
        }
        if opener(token) {
            if depth == MAX_DICTIONARY_NESTING {
                return Err((index, "field value nesting limit exceeded"));
            }
            stack[depth] = token.value.as_bytes()[0] as char;
            depth = depth
                .checked_add(1)
                .ok_or((index, "field value nesting counter overflow"))?;
        } else if closer(token) {
            let top = depth
                .checked_sub(1)
                .ok_or((index, "unexpected field value closing delimiter"))?;
            if !matching(stack[top], token.value.as_str()) {
                return Err((index, "mismatched field value delimiter"));
            }
            depth = top;
        }
    }
    Err((tokens.len(), "field value is missing a semicolon"))
}

fn joined_value(cursor: &mut TokenCursor, start: usize, end: usize) -> Result<String> {
    let additional = {
        let tokens = cursor.remaining_tokens()?;
        let mut bytes = Some(0usize);
        for token in &tokens[start + 1..end] {
            bytes = bytes
                .and_then(|n| n.checked_add(1))
                .and_then(|n| n.checked_add(token.value.len()));
        }
        bytes
    };
    let Some(additional) = additional else {
        return reject(cursor, "field value length overflow");
    };
    for _ in 0..start {
        cursor.next_required()?;
    }
    cursor.try_reserve_current_value(additional)?;
    let mut value = cursor.next_required()?.value;
    for _ in start + 1..end {
        value.push(' ');
        value.push_str(&cursor.next_required()?.value);
    }
    cursor.next_required()?;
    Ok(value)
}

fn parse_field_value(cursor: &mut TokenCursor) -> Result<FieldValueSummary> {
    #[derive(Clone, Copy)]
    enum Action {
        Uniform,
        NonUniform,
        Other,
        Invalid,
    }
    let action = {
        let tokens = cursor.remaining_tokens()?;
        match tokens.first() {
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
            None => return reject(cursor, "field value is missing"),
        }
    };
    let end = {
        let tokens = cursor.remaining_tokens()?;
        match value_end(tokens) {
            Ok(v) => v,
            Err((i, d)) => return reject_at(cursor, i, d),
        }
    };
    match action {
        Action::Invalid => reject(cursor, "invalid field value"),
        Action::Other => Ok(FieldValueSummary::Other(joined_value(cursor, 0, end)?)),
        Action::Uniform => {
            let valid_end = {
                let tokens = cursor.remaining_tokens()?;
                if end < 2 {
                    return reject_at(cursor, end, "uniform value is missing");
                }
                if opener(&tokens[1]) {
                    match balanced_end(tokens, 1) {
                        Ok(v) => v,
                        Err((i, d)) => return reject_at(cursor, i, d),
                    }
                } else if tokens[1].provenance != TokenProvenance::Structural {
                    2
                } else {
                    return reject_at(cursor, 1, "invalid uniform value");
                }
            };
            if valid_end != end {
                return reject_at(cursor, valid_end, "uniform value has trailing tokens");
            }
            Ok(FieldValueSummary::Uniform(joined_value(cursor, 1, end)?))
        }
        Action::NonUniform => parse_nonuniform(cursor, end),
    }
}

fn parse_nonuniform(cursor: &mut TokenCursor, end: usize) -> Result<FieldValueSummary> {
    let (count, expected) = {
        let tokens = cursor.remaining_tokens()?;
        if end < 5 {
            return reject_at(cursor, end, "incomplete nonuniform value");
        }
        let components = if tokens[1].provenance == TokenProvenance::Ordinary
            && tokens[1].value == "List<scalar>"
        {
            1
        } else if tokens[1].provenance == TokenProvenance::Ordinary
            && tokens[1].value == "List<vector>"
        {
            3
        } else {
            return reject_at(cursor, 1, "unsupported nonuniform value type");
        };
        if tokens[2].provenance != TokenProvenance::Ordinary {
            return reject_at(cursor, 2, "invalid nonuniform count");
        }
        let count = match tokens[2].value.parse::<usize>() {
            Ok(v) => v,
            Err(_) => return reject_at(cursor, 2, "invalid nonuniform count"),
        };
        let expected = match count.checked_mul(components) {
            Some(v) => v,
            None => return reject_at(cursor, 2, "nonuniform count overflow"),
        };
        if count > MAX_DICTIONARY_TOKENS
            || expected > MAX_DICTIONARY_TOKENS
            || expected > MAX_DICTIONARY_PAYLOAD_BYTES
        {
            return reject_at(cursor, 2, "nonuniform value limit exceeded");
        }
        if !structural(&tokens[3], "(") {
            return reject_at(cursor, 3, "expected nonuniform list opener");
        }
        let mut index = 4usize;
        for _ in 0..count {
            if components == 3 {
                if tokens.get(index).is_none_or(|t| !structural(t, "(")) {
                    return reject_at(cursor, index, "expected vector tuple opener");
                }
                index = match index.checked_add(1) {
                    Some(v) => v,
                    None => return reject_at(cursor, index, "nonuniform index overflow"),
                };
            }
            for _ in 0..components {
                let token = match tokens.get(index) {
                    Some(v) => v,
                    None => return reject_at(cursor, index, "missing nonuniform numeric value"),
                };
                if token.provenance != TokenProvenance::Ordinary {
                    return reject_at(cursor, index, "invalid nonuniform numeric value");
                }
                index = match index.checked_add(1) {
                    Some(v) => v,
                    None => return reject_at(cursor, index, "nonuniform index overflow"),
                };
            }
            if components == 3 {
                if tokens.get(index).is_none_or(|t| !structural(t, ")")) {
                    return reject_at(cursor, index, "expected vector tuple closer");
                }
                index = match index.checked_add(1) {
                    Some(v) => v,
                    None => return reject_at(cursor, index, "nonuniform index overflow"),
                };
            }
        }
        if tokens.get(index).is_none_or(|t| !structural(t, ")")) {
            return reject_at(cursor, index, "expected nonuniform list closer");
        }
        index = match index.checked_add(1) {
            Some(v) => v,
            None => return reject_at(cursor, index, "nonuniform index overflow"),
        };
        if index != end {
            return reject_at(cursor, index, "nonuniform value has trailing tokens");
        }
        (count, expected)
    };
    let mut values = Vec::new();
    if values.try_reserve(expected).is_err() {
        return reject(cursor, "nonuniform value allocation failed");
    }
    let invalid_numeric = {
        let tokens = cursor.remaining_tokens()?;
        let mut invalid = None;
        for (index, token) in tokens.iter().enumerate().take(end).skip(4) {
            if token.provenance != TokenProvenance::Ordinary {
                continue;
            }
            match token.value.parse::<f64>() {
                Ok(number) if number.is_finite() => values.push(number),
                _ => {
                    invalid = Some(index);
                    break;
                }
            }
        }
        invalid
    };
    if let Some(index) = invalid_numeric {
        return reject_at(cursor, index, "invalid nonuniform numeric value");
    }
    if values.len() != expected {
        return reject(cursor, "nonuniform numeric count mismatch");
    }
    let mut type_value = None;
    for index in 0..end {
        let token = cursor.next_required()?;
        if index == 1 {
            type_value = Some(token.value);
        }
    }
    cursor.next_required()?;
    Ok(FieldValueSummary::NonUniform {
        value_type: type_value,
        count: Some(count),
        values: Some(values),
    })
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

        let cached = lookup_mesh_cache(&self.mesh_cache, &region, &mut self.warnings)?;
        match cached {
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

fn lookup_mesh_cache<'a>(
    cache: &'a HashMap<Option<String>, Result<PolyMesh>>,
    region: &Option<String>,
    warnings: &mut Vec<String>,
) -> Option<&'a Result<PolyMesh>> {
    if let Some(cached) = cache.get(region) {
        return Some(cached);
    }
    if warnings.try_reserve(1).is_err() {
        return None;
    }
    let detail = "could not validate fields: mesh cache entry is unavailable";
    let mut warning = String::new();
    if warning.try_reserve(detail.len()).is_err() {
        return None;
    }
    warning.push_str(detail);
    warnings.push(warning);
    None
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
    use std::collections::HashMap;
    use std::path::{Path, PathBuf};

    use crate::Point3;
    use crate::poly_mesh::{BoundaryPatch, PolyMesh};
    use crate::{MeshError, Result};

    use super::{
        FieldBoundaryPatch, FieldFile, FieldValueSummary, MAX_DICTIONARY_NESTING,
        MAX_DICTIONARY_TOKENS, lookup_mesh_cache, parse_field_file_str,
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

        let count = MAX_DICTIONARY_TOKENS + 1;
        let oversized = format!(
            "FoamFile {{ class volScalarField; object p; }}\ninternalField nonuniform List<scalar> {count} ();\n"
        );
        assert_parse_error(&oversized, 2, "nonuniform value limit exceeded");
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
            "expected finite dimensions exponent",
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

        let cache: HashMap<Option<String>, Result<PolyMesh>> = HashMap::new();
        let mut warnings = vec!["sentinel".to_string()];
        assert!(lookup_mesh_cache(&cache, &None, &mut warnings).is_none());
        assert_eq!(
            warnings,
            vec![
                "sentinel",
                "could not validate fields: mesh cache entry is unavailable"
            ]
        );
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
