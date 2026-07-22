use std::fs;
use std::path::{Path, PathBuf};

use crate::dictionary::{MAX_DICTIONARY_NESTING, Token, TokenCursor, TokenProvenance, tokenize};
use crate::{MeshError, Result};

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
    let constant_dir = case_dir.join("constant");
    if !try_path_is_real_directory(&constant_dir)? {
        return Ok(Vec::new());
    }

    let mut dictionaries = Vec::new();
    read_property_dictionaries_in_dir(&constant_dir, None, &mut dictionaries)?;

    for entry in fs::read_dir(&constant_dir)? {
        let entry = entry?;
        let path = entry.path();
        let file_type = entry.file_type()?;
        if file_type.is_symlink() || !file_type.is_dir() {
            continue;
        }

        let region = entry.file_name().to_string_lossy().to_string();
        if region == "polyMesh" || !try_path_is_real_directory(&path.join("polyMesh"))? {
            continue;
        }
        read_property_dictionaries_in_dir(&path, Some(region), &mut dictionaries)?;
    }

    dictionaries.sort_by(|left, right| {
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
    region: Option<String>,
    dictionaries: &mut Vec<PropertyDictionary>,
) -> Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        let file_type = entry.file_type()?;
        if file_type.is_symlink() {
            return Err(MeshError::InvalidInput(format!(
                "property dictionary symlinks are not allowed: {}",
                path.display()
            )));
        }
        if !file_type.is_file() || !is_property_dictionary_file(&path) {
            continue;
        }

        dictionaries.push(read_property_dictionary(&path, region.clone())?);
    }
    Ok(())
}

fn is_property_dictionary_file(path: &Path) -> bool {
    let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
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
        Err(error) => Err(MeshError::InvalidInput(format!(
            "could not inspect {} ({error})",
            path.display()
        ))),
    }
}

fn read_property_dictionary(path: &Path, region: Option<String>) -> Result<PropertyDictionary> {
    let content = fs::read_to_string(path).map_err(|error| {
        MeshError::InvalidInput(format!("could not read {} ({error})", path.display()))
    })?;
    parse_property_dictionary_str(&content, path, region)
}

fn parse_property_dictionary_str(
    content: &str,
    path: &Path,
    region: Option<String>,
) -> Result<PropertyDictionary> {
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
            sections.push(parse_property_section(&mut cursor, key, 1)?);
        } else {
            entries.push(PropertyEntry {
                key,
                value: cursor.read_provenance_preserving_bare_entry()?,
            });
        }
    }

    Ok(PropertyDictionary {
        path: path.to_path_buf(),
        region,
        name: path
            .file_name()
            .map(|name| name.to_string_lossy().to_string())
            .unwrap_or_else(|| "unknown".to_string()),
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
        return Err(MeshError::InvalidInput(format!(
            "property dictionary nesting exceeds {MAX_DICTIONARY_NESTING} levels in {}",
            cursor.path().display()
        )));
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
            sections.push(parse_property_section(cursor, key, depth + 1)?);
        } else {
            entries.push(PropertyEntry {
                key,
                value: cursor.read_provenance_preserving_bare_entry()?,
            });
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

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::{
        format_property_value, parse_property_dictionary_str, read_case_properties,
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
