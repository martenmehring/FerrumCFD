use std::ffi::{OsStr, OsString};
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

use crate::case_input::CaseInput;
use crate::dictionary::{MAX_DICTIONARY_NESTING, Token, TokenCursor, TokenProvenance, tokenize};
use crate::{MeshError, Result};

pub const MAX_PROPERTY_REGIONS: usize = 256;
pub const MAX_PROPERTY_DICTIONARIES: usize = 1_024;
pub const MAX_PROPERTY_DISCOVERY_ENTRIES: usize = 4_096;
pub const MAX_RETAINED_PROPERTY_CONTENT_BYTES: usize = 16 * 1024 * 1024;

#[derive(Clone, Copy)]
struct PropertyLimits {
    regions: usize,
    dictionaries: usize,
    discovery_entries: usize,
    retained_content_bytes: usize,
}

const PROPERTY_LIMITS: PropertyLimits = PropertyLimits {
    regions: MAX_PROPERTY_REGIONS,
    dictionaries: MAX_PROPERTY_DICTIONARIES,
    discovery_entries: MAX_PROPERTY_DISCOVERY_ENTRIES,
    retained_content_bytes: MAX_RETAINED_PROPERTY_CONTENT_BYTES,
};

struct PropertyBudget {
    limits: PropertyLimits,
    regions: usize,
    dictionaries: usize,
    discovery_entries: usize,
    retained_content_bytes: usize,
}

impl PropertyBudget {
    fn new(limits: PropertyLimits) -> Self {
        Self {
            limits,
            regions: 0,
            dictionaries: 0,
            discovery_entries: 0,
            retained_content_bytes: 0,
        }
    }

    fn add_region(&mut self, path: &Path) -> Result<()> {
        self.regions = checked_bounded_add(
            self.regions,
            1,
            self.limits.regions,
            path,
            "property region count",
        )?;
        Ok(())
    }

    fn add_dictionary(&mut self, path: &Path) -> Result<()> {
        self.dictionaries = checked_bounded_add(
            self.dictionaries,
            1,
            self.limits.dictionaries,
            path,
            "property dictionary count",
        )?;
        Ok(())
    }

    fn add_discovery_entry(&mut self, path: &Path) -> Result<()> {
        self.discovery_entries = checked_bounded_add(
            self.discovery_entries,
            1,
            self.limits.discovery_entries,
            path,
            "property discovery entry count",
        )?;
        Ok(())
    }

    fn retain_content(&mut self, path: &Path, bytes: usize) -> Result<()> {
        self.retained_content_bytes = checked_bounded_add(
            self.retained_content_bytes,
            bytes,
            self.limits.retained_content_bytes,
            path,
            "retained property content bytes",
        )?;
        Ok(())
    }
}

fn try_copy_string(value: &str) -> Result<String> {
    let mut copy = String::new();
    copy.try_reserve_exact(value.len())
        .map_err(|_| MeshError::OutOfMemory)?;
    copy.push_str(value);
    Ok(copy)
}

fn try_copy_path(path: &Path) -> Result<PathBuf> {
    let source = path.as_os_str();
    let mut copy = OsString::new();
    copy.try_reserve_exact(source.len())
        .map_err(|_| MeshError::OutOfMemory)?;
    copy.push(source);
    Ok(PathBuf::from(copy))
}

fn try_join_path(parent: &Path, child: &OsStr) -> Result<PathBuf> {
    let additional = child.len().checked_add(1).ok_or(MeshError::OutOfMemory)?;
    let mut joined = try_copy_path(parent)?;
    joined
        .try_reserve_exact(additional)
        .map_err(|_| MeshError::OutOfMemory)?;
    joined.push(child);
    Ok(joined)
}

fn try_path_display(path: &Path) -> Result<String> {
    let bytes = path.as_os_str().as_encoded_bytes();
    let capacity = lossy_utf8_length(bytes)?;
    let mut display = String::new();
    display
        .try_reserve_exact(capacity)
        .map_err(|_| MeshError::OutOfMemory)?;
    append_lossy_utf8(&mut display, bytes);
    Ok(display)
}

fn lossy_utf8_length(mut bytes: &[u8]) -> Result<usize> {
    let mut length = 0usize;
    loop {
        match std::str::from_utf8(bytes) {
            Ok(valid) => {
                return length
                    .checked_add(valid.len())
                    .ok_or(MeshError::OutOfMemory);
            }
            Err(error) => {
                let valid = error.valid_up_to();
                length = length
                    .checked_add(valid)
                    .and_then(|value| value.checked_add(char::REPLACEMENT_CHARACTER.len_utf8()))
                    .ok_or(MeshError::OutOfMemory)?;
                let invalid = error.error_len().unwrap_or(bytes.len() - valid);
                let consumed = valid.checked_add(invalid).ok_or(MeshError::OutOfMemory)?;
                bytes = &bytes[consumed..];
            }
        }
    }
}

fn append_lossy_utf8(output: &mut String, mut bytes: &[u8]) {
    loop {
        match std::str::from_utf8(bytes) {
            Ok(valid) => {
                output.push_str(valid);
                return;
            }
            Err(error) => {
                let valid = error.valid_up_to();
                // SAFETY: `valid_up_to` is the length of a valid UTF-8 prefix.
                output.push_str(unsafe { std::str::from_utf8_unchecked(&bytes[..valid]) });
                output.push(char::REPLACEMENT_CHARACTER);
                let invalid = error.error_len().unwrap_or(bytes.len() - valid);
                bytes = &bytes[valid + invalid..];
            }
        }
    }
}

#[derive(Default)]
struct CountingWriter {
    len: usize,
}

impl fmt::Write for CountingWriter {
    fn write_str(&mut self, value: &str) -> fmt::Result {
        self.len = self.len.checked_add(value.len()).ok_or(fmt::Error)?;
        Ok(())
    }
}

fn try_invalid_input(arguments: fmt::Arguments<'_>) -> Result<MeshError> {
    let mut counter = CountingWriter::default();
    fmt::write(&mut counter, arguments).map_err(|_| MeshError::OutOfMemory)?;
    let mut message = String::new();
    message
        .try_reserve_exact(counter.len)
        .map_err(|_| MeshError::OutOfMemory)?;
    fmt::write(&mut message, arguments).map_err(|_| MeshError::OutOfMemory)?;
    Ok(MeshError::InvalidInput(message))
}

fn property_path_error(prefix: &str, path: &Path) -> Result<MeshError> {
    let display = try_path_display(path)?;
    try_invalid_input(format_args!("{prefix}{display}"))
}

fn property_io_error(path: &Path, source: &std::io::Error) -> MeshError {
    let display = match try_path_display(path) {
        Ok(display) => display,
        Err(error) => return error,
    };
    match try_invalid_input(format_args!("could not inspect {display} ({source})")) {
        Ok(error) | Err(error) => error,
    }
}

#[derive(Debug)]
pub struct PropertyDictionary {
    pub path: PathBuf,
    pub region: Option<String>,
    pub name: String,
    pub entries: Vec<PropertyEntry>,
    pub sections: Vec<PropertySection>,
}

#[derive(Clone, Debug)]
pub struct PropertySection {
    pub name: String,
    pub entries: Vec<PropertyEntry>,
    pub sections: Vec<PropertySection>,
}

#[derive(Clone, Debug)]
pub struct PropertyEntry {
    pub key: String,
    pub value: Vec<String>,
}

#[derive(Debug)]
pub struct PropertyValidation {
    pub warnings: Vec<String>,
}

pub fn read_case_properties(case_dir: &Path) -> Result<Vec<PropertyDictionary>> {
    read_case_properties_with_limits(case_dir, PROPERTY_LIMITS)
}

fn read_case_properties_with_limits(
    case_dir: &Path,
    limits: PropertyLimits,
) -> Result<Vec<PropertyDictionary>> {
    let constant_dir = try_join_path(case_dir, OsStr::new("constant"))?;
    if !try_path_is_real_directory(&constant_dir)? {
        return Ok(Vec::new());
    }

    let input = CaseInput::new(case_dir);
    let mut dictionaries = Vec::new();
    let mut budget = PropertyBudget::new(limits);
    read_property_dictionaries_in_dir(
        &constant_dir,
        None,
        true,
        &input,
        &mut budget,
        &mut dictionaries,
    )?;

    dictionaries.sort_unstable_by(|left, right| {
        left.region
            .cmp(&right.region)
            .then(left.name.cmp(&right.name))
            .then(left.path.cmp(&right.path))
    });
    Ok(dictionaries)
}

pub fn format_property_value(value: &[String]) -> String {
    if value.is_empty() {
        return "empty".to_string();
    }
    value.join(" ")
}

pub fn validate_properties(dictionaries: &[PropertyDictionary]) -> PropertyValidation {
    let mut warnings = Vec::new();
    if dictionaries.is_empty() {
        warnings.push("no constant property dictionaries found".to_string());
        return PropertyValidation { warnings };
    }

    for dictionary in dictionaries {
        if dictionary.entries.is_empty() && dictionary.sections.is_empty() {
            warnings.push(format!(
                "property dictionary '{}' has no entries",
                dictionary_label(dictionary)
            ));
        }
        validate_dimensioned_entries(
            &dictionary_label(dictionary),
            &dictionary.entries,
            &mut warnings,
        );
        validate_section_entries(
            &dictionary_label(dictionary),
            &dictionary.sections,
            &mut warnings,
        );
    }

    PropertyValidation { warnings }
}

fn read_property_dictionaries_in_dir(
    dir: &Path,
    region: Option<&str>,
    discover_regions: bool,
    input: &CaseInput,
    budget: &mut PropertyBudget,
    dictionaries: &mut Vec<PropertyDictionary>,
) -> Result<()> {
    let entries = fs::read_dir(dir).map_err(|error| property_io_error(dir, &error))?;
    for entry in entries {
        let entry = entry.map_err(|error| property_io_error(dir, &error))?;
        let file_name = entry.file_name();
        let path = try_join_path(dir, &file_name)?;
        budget.add_discovery_entry(&path)?;
        let file_type = entry
            .file_type()
            .map_err(|error| property_io_error(&path, &error))?;
        if file_type.is_symlink() {
            return Err(property_path_error(
                "property dictionary symlinks are not allowed: ",
                &path,
            )?);
        }
        if discover_regions && file_type.is_dir() {
            let Some(child_region) = file_name.to_str() else {
                return Err(property_path_error(
                    "property dictionary region name is not valid UTF-8: ",
                    &path,
                )?);
            };
            let poly_mesh = try_join_path(&path, OsStr::new("polyMesh"))?;
            if child_region != "polyMesh" && try_path_is_real_directory(&poly_mesh)? {
                budget.add_region(&path)?;
                read_property_dictionaries_in_dir(
                    &path,
                    Some(child_region),
                    false,
                    input,
                    budget,
                    dictionaries,
                )?;
            }
            continue;
        }
        if !file_type.is_file() || !is_property_dictionary_file(&file_name) {
            continue;
        }

        let Some(name) = file_name.to_str() else {
            return Err(property_path_error(
                "property dictionary name is not valid UTF-8: ",
                &path,
            )?);
        };
        budget.add_dictionary(&path)?;
        dictionaries
            .try_reserve(1)
            .map_err(|_| MeshError::OutOfMemory)?;
        let logical = property_logical_path(region, name)?;
        let dictionary = read_property_dictionary(input, &logical, &path, region, budget)?;
        dictionaries.push(dictionary);
    }
    Ok(())
}

fn property_logical_path(region: Option<&str>, name: &str) -> Result<String> {
    let region_bytes = match region {
        Some(value) => value.len().checked_add(1).ok_or(MeshError::OutOfMemory)?,
        None => 0,
    };
    let capacity = "constant/"
        .len()
        .checked_add(region_bytes)
        .and_then(|length| length.checked_add(name.len()))
        .ok_or(MeshError::OutOfMemory)?;
    let mut logical = String::new();
    logical
        .try_reserve_exact(capacity)
        .map_err(|_| MeshError::OutOfMemory)?;
    logical.push_str("constant/");
    if let Some(region) = region {
        logical.push_str(region);
        logical.push('/');
    }
    logical.push_str(name);
    Ok(logical)
}

fn is_property_dictionary_file(file_name: &OsStr) -> bool {
    let Some(name) = file_name.to_str() else {
        return false;
    };

    matches!(
        name,
        "transportProperties"
            | "physicalProperties"
            | "momentumTransport"
            | "turbulenceProperties"
            | "thermophysicalProperties"
    )
}

fn try_path_is_real_directory(path: &Path) -> Result<bool> {
    match fs::symlink_metadata(path) {
        Ok(metadata) => Ok(metadata.file_type().is_dir()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(property_io_error(path, &error)),
    }
}

fn read_property_dictionary(
    input: &CaseInput,
    logical: &str,
    path: &Path,
    region: Option<&str>,
    budget: &mut PropertyBudget,
) -> Result<PropertyDictionary> {
    let content = input.required(logical)?;
    budget.retain_content(path, content.len())?;
    parse_property_dictionary_str(&content, path, copy_region(region)?)
}

fn copy_region(region: Option<&str>) -> Result<Option<String>> {
    let Some(region) = region else {
        return Ok(None);
    };
    Ok(Some(try_copy_string(region)?))
}

fn parse_property_dictionary_str(
    content: &str,
    path: &Path,
    region: Option<String>,
) -> Result<PropertyDictionary> {
    let dictionary_path = try_copy_path(path)?;
    let name = match path.file_name().and_then(OsStr::to_str) {
        Some(name) => try_copy_string(name)?,
        None if path.file_name().is_none() => try_copy_string("unknown")?,
        None => {
            return Err(property_path_error(
                "property dictionary name is not valid UTF-8: ",
                path,
            )?);
        }
    };
    let mut cursor = tokenize(path, content)?.into_cursor();
    let mut entries = Vec::new();
    let mut sections = Vec::new();

    while let Some(token) = cursor.peek()? {
        if structural(token, ";") {
            cursor.next_required()?;
            continue;
        }
        if ordinary(token, "FoamFile") {
            cursor.next_required()?;
            cursor.skip_braced_block()?;
            continue;
        }

        let key = take_name(&mut cursor)?;
        if cursor.peek()?.is_some_and(|token| structural(token, "{")) {
            if sections.try_reserve(1).is_err() {
                return cursor.reject_current_as("property section allocation failed");
            }
            let section = parse_property_section(&mut cursor, key, 1)?;
            sections.push(section);
        } else {
            if entries.try_reserve(1).is_err() {
                return cursor.reject_current_as("property entry allocation failed");
            }
            let entry = PropertyEntry {
                key,
                value: cursor.read_provenance_preserving_bare_entry()?,
            };
            entries.push(entry);
        }
    }

    Ok(PropertyDictionary {
        path: dictionary_path,
        region,
        name,
        entries,
        sections,
    })
}

fn parse_property_section(
    cursor: &mut TokenCursor,
    name: String,
    depth: usize,
) -> Result<PropertySection> {
    if depth > MAX_DICTIONARY_NESTING {
        let display = try_path_display(cursor.path())?;
        return Err(try_invalid_input(format_args!(
            "property dictionary nesting exceeds {MAX_DICTIONARY_NESTING} levels in {display}"
        ))?);
    }
    cursor.expect("{")?;
    let mut entries = Vec::new();
    let mut sections = Vec::new();

    while !cursor.peek()?.is_some_and(|token| structural(token, "}")) {
        if cursor.peek()?.is_some_and(|token| structural(token, ";")) {
            cursor.next_required()?;
            continue;
        }

        let key = take_name(cursor)?;
        if cursor.peek()?.is_some_and(|token| structural(token, "{")) {
            if sections.try_reserve(1).is_err() {
                return cursor.reject_current_as("property section allocation failed");
            }
            let next_depth = depth.checked_add(1).ok_or(MeshError::OutOfMemory)?;
            let section = parse_property_section(cursor, key, next_depth)?;
            sections.push(section);
        } else {
            if entries.try_reserve(1).is_err() {
                return cursor.reject_current_as("property entry allocation failed");
            }
            let entry = PropertyEntry {
                key,
                value: cursor.read_provenance_preserving_bare_entry()?,
            };
            entries.push(entry);
        }
    }
    cursor.expect("}")?;
    cursor.expect_optional(";")?;

    Ok(PropertySection {
        name,
        entries,
        sections,
    })
}

fn structural(token: &Token, expected: &str) -> bool {
    token.provenance == TokenProvenance::Structural && token.value == expected
}

fn ordinary(token: &Token, expected: &str) -> bool {
    token.provenance == TokenProvenance::Ordinary && token.value == expected
}

fn take_name(cursor: &mut TokenCursor) -> Result<String> {
    if cursor
        .peek()?
        .is_some_and(|token| token.provenance == TokenProvenance::Structural)
    {
        return cursor.reject_current_as("property name must not be structural punctuation");
    }
    let quoted = cursor
        .peek()?
        .is_some_and(|token| token.provenance == TokenProvenance::Quoted);
    if quoted {
        cursor.try_reserve_current_value(2)?;
    }
    let mut token = cursor.next_required()?;
    if quoted {
        token.value.insert(0, '"');
        token.value.push('"');
    }
    Ok(token.value)
}

fn validate_section_entries(
    dictionary: &str,
    sections: &[PropertySection],
    warnings: &mut Vec<String>,
) {
    for section in sections {
        let label = format!("{dictionary}.{}", section.name);
        validate_dimensioned_entries(&label, &section.entries, warnings);
        validate_section_entries(&label, &section.sections, warnings);
    }
}

fn validate_dimensioned_entries(
    label: &str,
    entries: &[PropertyEntry],
    warnings: &mut Vec<String>,
) {
    for entry in entries {
        if entry.value.first().map(String::as_str) != Some("[") {
            continue;
        }

        let Some(end) = entry.value.iter().position(|value| value == "]") else {
            warnings.push(format!(
                "{label}.{} has an unterminated dimension vector",
                entry.key
            ));
            continue;
        };

        if end != 8 {
            warnings.push(format!(
                "{label}.{} dimension vector has {} entries; expected 7",
                entry.key,
                end.saturating_sub(1)
            ));
        }
        if entry.value.len() <= end + 1 {
            warnings.push(format!("{label}.{} has dimensions but no value", entry.key));
        }
    }
}

fn dictionary_label(dictionary: &PropertyDictionary) -> String {
    if let Some(region) = &dictionary.region {
        format!("{region}/{}", dictionary.name)
    } else {
        dictionary.name.clone()
    }
}

fn checked_bounded_add(
    current: usize,
    additional: usize,
    limit: usize,
    path: &Path,
    label: &str,
) -> Result<usize> {
    let Some(next) = current.checked_add(additional) else {
        return Err(property_limit_error(path, label, limit)?);
    };
    if next > limit {
        return Err(property_limit_error(path, label, limit)?);
    }
    Ok(next)
}

fn property_limit_error(path: &Path, label: &str, limit: usize) -> Result<MeshError> {
    let display = try_path_display(path)?;
    try_invalid_input(format_args!(
        "{label} exceeds limit {limit} while reading {display}"
    ))
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::{
        MAX_PROPERTY_DICTIONARIES, MAX_PROPERTY_DISCOVERY_ENTRIES, MAX_PROPERTY_REGIONS,
        MAX_RETAINED_PROPERTY_CONTENT_BYTES, PropertyLimits, format_property_value,
        parse_property_dictionary_str, read_case_properties, read_case_properties_with_limits,
        validate_properties,
    };

    struct TestCaseDir {
        path: PathBuf,
    }

    impl TestCaseDir {
        fn new(name: &str) -> Self {
            let unique = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let path = std::env::temp_dir().join(format!(
                "ferrum-properties-{name}-{}-{unique}",
                std::process::id()
            ));
            fs::create_dir_all(path.join("constant")).unwrap();
            Self { path }
        }
    }

    impl Drop for TestCaseDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.path);
        }
    }

    #[test]
    fn reads_only_known_property_dictionary_names() {
        let case = TestCaseDir::new("allowlist");
        fs::write(
            case.path.join("constant/transportProperties"),
            "nu [0 2 -1 0 0 0 0] 1e-05;",
        )
        .unwrap();
        fs::write(
            case.path.join("constant/leakDict"),
            "secret token should not be parsed;",
        )
        .unwrap();

        let dictionaries = read_case_properties(&case.path).unwrap();

        assert_eq!(dictionaries.len(), 1);
        assert_eq!(dictionaries[0].name, "transportProperties");
    }

    #[cfg(unix)]
    #[test]
    fn rejects_symlinked_property_dictionaries_before_reading() {
        let case = TestCaseDir::new("symlink");
        let secret = case.path.join("secret-outside-case");
        fs::write(&secret, "nu [0 2 -1 0 0 0 0] leaked;").unwrap();
        std::os::unix::fs::symlink(&secret, case.path.join("constant/transportProperties"))
            .unwrap();

        let error = read_case_properties(&case.path).unwrap_err();

        assert!(
            error
                .to_string()
                .contains("property dictionary symlinks are not allowed"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn rejects_oversized_property_dictionary_before_parsing() {
        let case = TestCaseDir::new("oversized");
        let path = case.path.join("constant/transportProperties");
        fs::File::create(&path)
            .unwrap()
            .set_len(16 * 1024 * 1024 + 1)
            .unwrap();

        let error = read_case_properties(&case.path).unwrap_err();
        let message = error.to_string();

        assert!(message.contains("constant/transportProperties"));
        assert_eq!(message.matches("constant/transportProperties").count(), 1);
    }

    #[test]
    fn reads_region_property_through_capability_scope() {
        let case = TestCaseDir::new("region");
        fs::create_dir_all(case.path.join("constant/fluid/polyMesh")).unwrap();
        fs::write(
            case.path.join("constant/fluid/transportProperties"),
            "nu [0 2 -1 0 0 0 0] 1e-05;",
        )
        .unwrap();

        let dictionaries = read_case_properties(&case.path).unwrap();

        assert_eq!(dictionaries.len(), 1);
        assert_eq!(dictionaries[0].region.as_deref(), Some("fluid"));
        assert_eq!(dictionaries[0].name, "transportProperties");
    }

    #[test]
    fn property_region_count_cap_is_exact() {
        let case = TestCaseDir::new("region-count-cap");
        let limits = PropertyLimits {
            regions: 1,
            dictionaries: MAX_PROPERTY_DICTIONARIES,
            discovery_entries: MAX_PROPERTY_DISCOVERY_ENTRIES,
            retained_content_bytes: MAX_RETAINED_PROPERTY_CONTENT_BYTES,
        };
        fs::create_dir_all(case.path.join("constant/region0/polyMesh")).unwrap();

        assert!(read_case_properties_with_limits(&case.path, limits).is_ok());

        fs::create_dir_all(case.path.join("constant/region1/polyMesh")).unwrap();
        let message = read_case_properties_with_limits(&case.path, limits)
            .unwrap_err()
            .to_string();
        let root = case.path.display().to_string();

        assert!(message.contains("property region count exceeds limit 1"));
        assert_eq!(message.matches(&root).count(), 1);
    }

    #[test]
    fn property_dictionary_count_cap_is_exact() {
        let case = TestCaseDir::new("dictionary-count-cap");
        let limits = PropertyLimits {
            regions: MAX_PROPERTY_REGIONS,
            dictionaries: 1,
            discovery_entries: MAX_PROPERTY_DISCOVERY_ENTRIES,
            retained_content_bytes: MAX_RETAINED_PROPERTY_CONTENT_BYTES,
        };
        fs::write(case.path.join("constant/transportProperties"), "nu 1;").unwrap();

        assert_eq!(
            read_case_properties_with_limits(&case.path, limits)
                .unwrap()
                .len(),
            1
        );

        fs::write(case.path.join("constant/physicalProperties"), "rho 1;").unwrap();
        let message = read_case_properties_with_limits(&case.path, limits)
            .unwrap_err()
            .to_string();
        let root = case.path.display().to_string();

        assert!(message.contains("property dictionary count exceeds limit 1"));
        assert_eq!(message.matches(&root).count(), 1);
    }

    #[test]
    fn property_discovery_entry_cap_is_aggregate_and_exact() {
        let case = TestCaseDir::new("discovery-entry-cap");
        let region = case.path.join("constant/region");
        let limits = PropertyLimits {
            regions: MAX_PROPERTY_REGIONS,
            dictionaries: MAX_PROPERTY_DICTIONARIES,
            discovery_entries: 2,
            retained_content_bytes: MAX_RETAINED_PROPERTY_CONTENT_BYTES,
        };
        fs::create_dir_all(region.join("polyMesh")).unwrap();

        assert!(read_case_properties_with_limits(&case.path, limits).is_ok());

        fs::write(region.join("ignored-entry"), "ignored").unwrap();
        let message = read_case_properties_with_limits(&case.path, limits)
            .unwrap_err()
            .to_string();
        let root = case.path.display().to_string();

        assert!(message.contains("property discovery entry count exceeds limit 2"));
        assert_eq!(message.matches(&root).count(), 1);
    }

    #[test]
    fn retained_property_content_byte_cap_is_exact() {
        let case = TestCaseDir::new("retained-content-cap");
        let first = case.path.join("constant/transportProperties");
        let second = case.path.join("constant/physicalProperties");
        let limits = PropertyLimits {
            regions: MAX_PROPERTY_REGIONS,
            dictionaries: MAX_PROPERTY_DICTIONARIES,
            discovery_entries: MAX_PROPERTY_DISCOVERY_ENTRIES,
            retained_content_bytes: 2,
        };
        fs::write(&first, ";").unwrap();
        fs::write(&second, ";").unwrap();

        assert_eq!(
            read_case_properties_with_limits(&case.path, limits)
                .unwrap()
                .len(),
            2
        );

        fs::write(&second, ";;").unwrap();
        let message = read_case_properties_with_limits(&case.path, limits)
            .unwrap_err()
            .to_string();
        let root = case.path.display().to_string();

        assert!(message.contains("retained property content bytes exceeds limit 2"));
        assert_eq!(message.matches(&root).count(), 1);
    }

    #[cfg(unix)]
    #[test]
    fn invalid_utf8_property_name_has_single_lossy_path_diagnostic() {
        use std::ffi::OsString;
        use std::os::unix::ffi::OsStringExt;

        let path = PathBuf::from(OsString::from_vec(
            b"constant/transportProperties-\xff".to_vec(),
        ));
        let message = parse_property_dictionary_str("", &path, None)
            .unwrap_err()
            .to_string();

        assert!(message.starts_with("property dictionary name is not valid UTF-8: "));
        assert_eq!(message.matches("constant/transportProperties-").count(), 1);
        assert_eq!(message.matches(char::REPLACEMENT_CHARACTER).count(), 1);
    }

    #[cfg(unix)]
    #[test]
    fn ignores_symlinked_constant_directory() {
        let case = TestCaseDir::new("constant-symlink");
        fs::remove_dir_all(case.path.join("constant")).unwrap();
        let outside = case.path.join("outside-constant");
        fs::create_dir_all(&outside).unwrap();
        fs::write(
            outside.join("transportProperties"),
            "nu [0 2 -1 0 0 0 0] leaked;",
        )
        .unwrap();
        std::os::unix::fs::symlink(&outside, case.path.join("constant")).unwrap();

        let dictionaries = read_case_properties(&case.path).unwrap();

        assert!(dictionaries.is_empty());
    }

    #[test]
    fn parses_dimensioned_transport_properties() {
        let content = r#"
        FoamFile
        {
            class dictionary;
            object transportProperties;
        }

        transportModel Newtonian;
        nu [0 2 -1 0 0 0 0] 1e-05;
        rho [1 -3 0 0 0 0 0] 1.2;
        "#;

        let dictionary =
            parse_property_dictionary_str(content, Path::new("transportProperties"), None).unwrap();

        assert_eq!(dictionary.name, "transportProperties");
        assert_eq!(dictionary.entries.len(), 3);
        assert_eq!(dictionary.entries[0].key, "transportModel");
        assert_eq!(
            format_property_value(&dictionary.entries[1].value),
            "[ 0 2 -1 0 0 0 0 ] 1e-05"
        );
    }

    #[test]
    fn parses_nested_property_sections() {
        let content = r#"
        mixture
        {
            specie
            {
                molWeight 18;
            }
            thermodynamics
            {
                Cp [0 2 -2 -1 0 0 0] 4180;
            }
        }
        "#;

        let dictionary = parse_property_dictionary_str(
            content,
            Path::new("thermophysicalProperties"),
            Some("fluid".to_string()),
        )
        .unwrap();

        assert_eq!(dictionary.region.as_deref(), Some("fluid"));
        assert_eq!(dictionary.sections[0].name, "mixture");
        assert_eq!(dictionary.sections[0].sections.len(), 2);
    }

    #[test]
    fn validates_dimension_vector_shape() {
        let dictionary = parse_property_dictionary_str(
            "nu [0 2 -1] 1e-05;",
            Path::new("transportProperties"),
            None,
        )
        .unwrap();

        let validation = validate_properties(&[dictionary]);

        assert_eq!(validation.warnings.len(), 1);
        assert!(validation.warnings[0].contains("expected 7"));
    }

    #[test]
    fn warns_for_empty_property_dictionary() {
        let dictionary =
            parse_property_dictionary_str("", Path::new("transportProperties"), None).unwrap();

        let validation = validate_properties(&[dictionary]);

        assert_eq!(validation.warnings.len(), 1);
        assert!(validation.warnings[0].contains("has no entries"));
    }

    #[test]
    fn quoted_property_keys_and_brackets_are_inert() {
        let dictionary = parse_property_dictionary_str(
            r#"
            "FoamFile" { class dictionary; }
            "nu" "[" 0 2 -1 0 0 0 0 "]" 1e-05;
            nu "[" 0 2 -1 0 0 0 0 "]" 1e-05;
            marker ";";
            ";" inert;
            rho [1 -3 0 0 0 0 0] 1.2;
            "#,
            Path::new("transportProperties"),
            None,
        )
        .unwrap();

        assert_eq!(dictionary.sections[0].name, "\"FoamFile\"");
        assert_eq!(dictionary.entries[0].key, "\"nu\"");
        assert_eq!(dictionary.entries[0].value[0], "\"[\"");
        assert_eq!(dictionary.entries[0].value[8], "\"]\"");
        assert_eq!(dictionary.entries[1].value[0], "\"[\"");
        assert_eq!(dictionary.entries[2].value, ["\";\""]);
        assert_eq!(dictionary.entries[3].key, "\";\"");
        assert!(validate_properties(&[dictionary]).warnings.is_empty());
    }

    #[test]
    fn quoted_closer_does_not_close_property_section() {
        let dictionary = parse_property_dictionary_str(
            r#"section { "}" inert; after 1; }"#,
            Path::new("transportProperties"),
            None,
        )
        .unwrap();

        assert_eq!(dictionary.sections[0].entries.len(), 2);
        assert_eq!(dictionary.sections[0].entries[0].key, "\"}\"");
        assert_eq!(dictionary.sections[0].entries[1].key, "after");
    }
}
