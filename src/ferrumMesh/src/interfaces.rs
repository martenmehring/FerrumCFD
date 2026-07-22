use std::fs;
use std::path::{Path, PathBuf};

use crate::dictionary::{MAX_DICTIONARY_NESTING, Token, TokenCursor, TokenProvenance, tokenize};
use crate::regions::InterfaceRegistrySummary;
use crate::{MeshError, Result};

#[derive(Debug)]
pub struct InterfaceConfig {
    pub path: PathBuf,
    pub entries: Vec<InterfaceConfigEntry>,
}

#[derive(Debug)]
pub struct InterfaceConfigEntry {
    pub name: String,
    pub regions: [String; 2],
    pub face_zone: String,
    pub orientation: InterfaceOrientation,
    pub model: String,
}

#[derive(Debug)]
pub struct InterfaceOrientation {
    pub positive_from: String,
    pub positive_to: String,
}

#[derive(Debug)]
pub struct InterfaceConfigValidation {
    pub entries: Vec<ValidatedInterfaceConfigEntry>,
    pub warnings: Vec<String>,
}

#[derive(Debug)]
pub struct ValidatedInterfaceConfigEntry {
    pub name: String,
    pub face_zone: String,
    pub positive_from: String,
    pub positive_to: String,
    pub model: String,
    pub mesh_faces: Option<usize>,
}

pub fn read_interface_config(case_dir: &Path) -> Result<Option<InterfaceConfig>> {
    let path = case_dir.join("constant").join("interfaces");
    if !path.exists() {
        return Ok(None);
    }

    let content = fs::read_to_string(&path).map_err(|error| {
        MeshError::InvalidInput(format!("could not read {} ({error})", path.display()))
    })?;
    let mut config = parse_interface_config_str(&content, &path)?;
    config.path = path;
    Ok(Some(config))
}

pub fn validate_interface_config(
    config: &InterfaceConfig,
    registry: &InterfaceRegistrySummary,
) -> InterfaceConfigValidation {
    let mut entries = Vec::new();
    let mut warnings = Vec::new();

    for entry in &config.entries {
        let matched = registry.interfaces.iter().find(|interface| {
            interface.name == entry.face_zone
                && same_region_pair(&interface.region_a, &interface.region_b, &entry.regions)
        });

        if let Some(interface) = matched {
            entries.push(ValidatedInterfaceConfigEntry {
                name: entry.name.clone(),
                face_zone: entry.face_zone.clone(),
                positive_from: entry.orientation.positive_from.clone(),
                positive_to: entry.orientation.positive_to.clone(),
                model: entry.model.clone(),
                mesh_faces: Some(interface.faces),
            });
            continue;
        }

        entries.push(ValidatedInterfaceConfigEntry {
            name: entry.name.clone(),
            face_zone: entry.face_zone.clone(),
            positive_from: entry.orientation.positive_from.clone(),
            positive_to: entry.orientation.positive_to.clone(),
            model: entry.model.clone(),
            mesh_faces: None,
        });

        if let Some(interface) = registry
            .interfaces
            .iter()
            .find(|interface| interface.name == entry.face_zone)
        {
            warnings.push(format!(
                "interface '{}' references faceZone '{}' for regions {}<->{}, but mesh has {}<->{}",
                entry.name,
                entry.face_zone,
                entry.regions[0],
                entry.regions[1],
                interface.region_a,
                interface.region_b
            ));
            continue;
        }

        if let Some(boundary) = registry
            .boundary_face_zones
            .iter()
            .find(|zone| zone.name == entry.face_zone)
        {
            warnings.push(format!(
                "interface '{}' references boundary faceZone '{}' on region '{}'",
                entry.name, entry.face_zone, boundary.region
            ));
            continue;
        }

        warnings.push(format!(
            "interface '{}' references missing faceZone '{}'",
            entry.name, entry.face_zone
        ));
    }

    InterfaceConfigValidation { entries, warnings }
}

fn parse_interface_config_str(content: &str, path: &Path) -> Result<InterfaceConfig> {
    let mut cursor = tokenize(path, content)?.into_cursor();
    let mut entries = Vec::new();
    let mut found_interfaces = false;

    while let Some(token) = cursor.peek()? {
        if token.provenance == TokenProvenance::Structural {
            return cursor.reject_current_as("unexpected structural token at dictionary root");
        }

        match (token.provenance, token.value.as_str()) {
            (TokenProvenance::Ordinary, "FoamFile") => {
                cursor.next_required()?;
                cursor.skip_braced_block()?;
            }
            (TokenProvenance::Ordinary, "interfaces") => {
                if found_interfaces {
                    return cursor.reject_current_as("duplicate ordinary 'interfaces' block");
                }
                cursor.next_required()?;
                cursor.expect("{")?;
                entries = parse_interfaces_block(&mut cursor)?;
                found_interfaces = true;
            }
            _ => {
                cursor.next_required()?;
                cursor.skip_exact_value_or_block()?;
            }
        }
    }

    if !found_interfaces {
        return cursor.reject_current_as("missing ordinary 'interfaces' block");
    }

    let path = copy_path(&mut cursor)?;
    Ok(InterfaceConfig { path, entries })
}

fn parse_interfaces_block(cursor: &mut TokenCursor) -> Result<Vec<InterfaceConfigEntry>> {
    let mut entries = Vec::new();

    while !peek_structural(cursor, "}")? {
        let Some(token) = cursor.peek()? else {
            return cursor.reject_current_as("unterminated ordinary 'interfaces' block");
        };
        if token.provenance != TokenProvenance::Ordinary {
            return cursor.reject_current_as("interface name must be an ordinary token");
        }
        if entries
            .iter()
            .any(|entry: &InterfaceConfigEntry| entry.name == token.value)
        {
            return cursor.reject_current_as("duplicate interface name");
        }
        if entries.try_reserve(1).is_err() {
            return cursor.reject_current_as("interface entry allocation failed");
        }

        let name = cursor.next_required()?.value;
        cursor.expect("{")?;
        entries.push(parse_interface_entry(cursor, name)?);
    }
    cursor.expect("}")?;

    Ok(entries)
}

fn parse_interface_entry(cursor: &mut TokenCursor, name: String) -> Result<InterfaceConfigEntry> {
    preflight_orientation(cursor)?;

    let mut regions = None;
    let mut face_zone = None;
    let mut orientation = None;
    let mut model = None;

    while !peek_structural(cursor, "}")? {
        let Some(token) = cursor.peek()? else {
            return cursor.reject_current_as("unterminated interface entry");
        };
        if token.provenance == TokenProvenance::Structural {
            return cursor.reject_current_as("interface key must not be structural punctuation");
        }

        if token.provenance == TokenProvenance::Ordinary {
            let duplicate = match token.value.as_str() {
                "regions" => regions.is_some(),
                "faceZone" => face_zone.is_some(),
                "orientation" => orientation.is_some(),
                "model" => model.is_some(),
                _ => false,
            };
            if duplicate {
                return cursor.reject_current_as("duplicate interface entry key");
            }
        }

        let key = cursor.next_required()?;
        match (key.provenance, key.value.as_str()) {
            (TokenProvenance::Ordinary, "regions") => regions = Some(read_regions(cursor)?),
            (TokenProvenance::Ordinary, "faceZone") => {
                face_zone = Some(read_ordinary_scalar(
                    cursor,
                    "faceZone value must be an ordinary token",
                )?);
                cursor.expect(";")?;
            }
            (TokenProvenance::Ordinary, "orientation") => {
                orientation = Some(read_ordinary_scalar(
                    cursor,
                    "orientation value must be an ordinary token",
                )?);
                cursor.expect(";")?;
            }
            (TokenProvenance::Ordinary, "model") => {
                model = Some(read_ordinary_scalar(
                    cursor,
                    "model value must be an ordinary token",
                )?);
                cursor.expect(";")?;
            }
            _ => cursor.skip_exact_value_or_block()?,
        }
    }
    let Some(region_tokens) = regions else {
        return cursor.reject_current_as("missing ordinary 'regions' in interface entry");
    };
    let Some(raw_orientation) = orientation else {
        return cursor.reject_current_as("missing ordinary 'orientation' in interface entry");
    };
    let Some(face_zone) = face_zone else {
        return cursor.reject_current_as("missing ordinary 'faceZone' in interface entry");
    };
    let orientation = parse_orientation(cursor, &raw_orientation, &region_tokens)?;
    let model = match model {
        Some(token) => token.value,
        None => copy_string(cursor, "none", "default interface model allocation failed")?,
    };
    let [first_region, second_region] = region_tokens;
    let regions = [first_region.value, second_region.value];
    cursor.expect("}")?;

    Ok(InterfaceConfigEntry {
        name,
        regions,
        face_zone: face_zone.value,
        orientation,
        model,
    })
}

fn read_regions(cursor: &mut TokenCursor) -> Result<[Token; 2]> {
    cursor.expect("(")?;
    let first = read_ordinary_scalar(cursor, "regions must contain exactly two ordinary entries")?;
    if cursor.peek()?.is_some_and(|token| {
        token.provenance == TokenProvenance::Ordinary && token.value == first.value
    }) {
        return cursor.reject_current_as("regions must name two different ordinary entries");
    }
    let second = read_ordinary_scalar(cursor, "regions must contain exactly two ordinary entries")?;
    if !peek_structural(cursor, ")")? {
        return cursor.reject_current_as("regions must contain exactly two ordinary entries");
    }
    cursor.expect(")")?;
    cursor.expect(";")?;

    Ok([first, second])
}

fn read_ordinary_scalar(cursor: &mut TokenCursor, detail: &'static str) -> Result<Token> {
    if !cursor
        .peek()?
        .is_some_and(|token| token.provenance == TokenProvenance::Ordinary)
    {
        return cursor.reject_current_as(detail);
    }
    cursor.next_required()
}

fn peek_structural(cursor: &mut TokenCursor, value: &str) -> Result<bool> {
    Ok(cursor.peek()?.is_some_and(|token| {
        token.provenance == TokenProvenance::Structural && token.value == value
    }))
}

fn parse_orientation(
    cursor: &mut TokenCursor,
    orientation: &Token,
    regions: &[Token; 2],
) -> Result<InterfaceOrientation> {
    let (from, to) =
        if orientation_matches(&orientation.value, &regions[0].value, &regions[1].value) {
            (&regions[0].value, &regions[1].value)
        } else if orientation_matches(&orientation.value, &regions[1].value, &regions[0].value) {
            (&regions[1].value, &regions[0].value)
        } else {
            return cursor
                .reject_current_as("orientation must match the declared ordinary region order");
        };

    Ok(InterfaceOrientation {
        positive_from: copy_string(
            cursor,
            from,
            "interface orientation source allocation failed",
        )?,
        positive_to: copy_string(
            cursor,
            to,
            "interface orientation destination allocation failed",
        )?,
    })
}

fn same_region_pair(left: &str, right: &str, pair: &[String; 2]) -> bool {
    (left == pair[0] && right == pair[1]) || (left == pair[1] && right == pair[0])
}

fn orientation_matches(value: &str, from: &str, to: &str) -> bool {
    value
        .strip_prefix(from)
        .and_then(|suffix| suffix.strip_prefix("_to_"))
        == Some(to)
}

fn copy_string(cursor: &mut TokenCursor, value: &str, detail: &'static str) -> Result<String> {
    let mut copy = String::new();
    if copy.try_reserve(value.len()).is_err() {
        return cursor.reject_current_as(detail);
    }
    copy.push_str(value);
    Ok(copy)
}

fn copy_path(cursor: &mut TokenCursor) -> Result<PathBuf> {
    let required = cursor.path().as_os_str().len();
    let mut copy = PathBuf::new();
    if copy.try_reserve(required).is_err() {
        return cursor.reject_current_as("interface path allocation failed");
    }
    copy.push(cursor.path());
    Ok(copy)
}

fn preflight_orientation(cursor: &mut TokenCursor) -> Result<()> {
    let invalid_offset = invalid_orientation_offset(cursor.remaining_tokens()?);
    if let Some(offset) = invalid_offset {
        return cursor.reject_at_as(
            offset,
            "orientation must match the declared ordinary region order",
        );
    }
    Ok(())
}

fn invalid_orientation_offset(tokens: &[Token]) -> Option<usize> {
    let mut index = 0usize;
    let mut regions = None;
    let mut orientation = None;
    let mut seen_regions = false;
    let mut seen_face_zone = false;
    let mut seen_orientation = false;
    let mut seen_model = false;

    loop {
        let key = tokens.get(index)?;
        if structural(key, "}") {
            break;
        }
        if key.provenance == TokenProvenance::Structural {
            return None;
        }
        index = index.checked_add(1)?;

        match (key.provenance, key.value.as_str()) {
            (TokenProvenance::Ordinary, "regions") => {
                if seen_regions {
                    return None;
                }
                seen_regions = true;
                let open = token_at(tokens, index, 0)?;
                let first = token_at(tokens, index, 1)?;
                let second = token_at(tokens, index, 2)?;
                let close = token_at(tokens, index, 3)?;
                let terminator = token_at(tokens, index, 4)?;
                if !structural(open, "(")
                    || first.provenance != TokenProvenance::Ordinary
                    || second.provenance != TokenProvenance::Ordinary
                    || first.value == second.value
                    || !structural(close, ")")
                    || !structural(terminator, ";")
                {
                    return None;
                }
                regions = Some((first.value.as_str(), second.value.as_str()));
                index = index.checked_add(5)?;
            }
            (TokenProvenance::Ordinary, "faceZone") => {
                if seen_face_zone {
                    return None;
                }
                seen_face_zone = true;
                index = preflight_scalar(tokens, index)?.0;
            }
            (TokenProvenance::Ordinary, "orientation") => {
                if seen_orientation {
                    return None;
                }
                seen_orientation = true;
                let (next, value_index) = preflight_scalar(tokens, index)?;
                orientation = Some((value_index, tokens.get(value_index)?.value.as_str()));
                index = next;
            }
            (TokenProvenance::Ordinary, "model") => {
                if seen_model {
                    return None;
                }
                seen_model = true;
                index = preflight_scalar(tokens, index)?.0;
            }
            _ => index = preflight_skip_exact(tokens, index)?,
        }
    }

    let ((first, second), (offset, value)) = (regions?, orientation?);
    if orientation_matches(value, first, second) || orientation_matches(value, second, first) {
        None
    } else {
        Some(offset)
    }
}

fn preflight_scalar(tokens: &[Token], index: usize) -> Option<(usize, usize)> {
    let value = tokens.get(index)?;
    if value.provenance != TokenProvenance::Ordinary {
        return None;
    }
    let terminator_index = index.checked_add(1)?;
    if !structural(tokens.get(terminator_index)?, ";") {
        return None;
    }
    Some((terminator_index.checked_add(1)?, index))
}

fn preflight_skip_exact(tokens: &[Token], index: usize) -> Option<usize> {
    let first = tokens.get(index)?;
    if structural(first, ";") || closer(first) {
        return None;
    }
    let braced = structural(first, "{");
    let end = if opener(first) {
        let mut stack = ['\0'; MAX_DICTIONARY_NESTING];
        let mut depth = 0usize;
        let mut end = None;
        for (relative, token) in tokens.get(index..)?.iter().enumerate() {
            if !track_delimiter(token, &mut stack, &mut depth) {
                return None;
            }
            if depth == 0 {
                end = index.checked_add(relative)?.checked_add(1);
                break;
            }
        }
        end?
    } else {
        index.checked_add(1)?
    };
    let has_terminator = tokens.get(end).is_some_and(|token| structural(token, ";"));
    if !braced && !has_terminator {
        return None;
    }
    end.checked_add(usize::from(has_terminator))
}

fn token_at(tokens: &[Token], index: usize, offset: usize) -> Option<&Token> {
    tokens.get(index.checked_add(offset)?)
}

fn structural(token: &Token, value: &str) -> bool {
    token.provenance == TokenProvenance::Structural && token.value == value
}

fn opener(token: &Token) -> bool {
    token.provenance == TokenProvenance::Structural
        && matches!(token.value.as_str(), "{" | "(" | "[")
}

fn closer(token: &Token) -> bool {
    token.provenance == TokenProvenance::Structural
        && matches!(token.value.as_str(), "}" | ")" | "]")
}

fn track_delimiter(
    token: &Token,
    stack: &mut [char; MAX_DICTIONARY_NESTING],
    depth: &mut usize,
) -> bool {
    if token.provenance != TokenProvenance::Structural {
        return true;
    }
    let delimiter = match token.value.as_str() {
        "{" => '{',
        "(" => '(',
        "[" => '[',
        "}" => '}',
        ")" => ')',
        "]" => ']',
        _ => return true,
    };
    if matches!(delimiter, '{' | '(' | '[') {
        if *depth == MAX_DICTIONARY_NESTING {
            return false;
        }
        stack[*depth] = delimiter;
        let Some(next) = depth.checked_add(1) else {
            return false;
        };
        *depth = next;
        return true;
    }
    let Some(top) = depth.checked_sub(1) else {
        return false;
    };
    let matches = matches!(
        (stack[top], delimiter),
        ('{', '}') | ('(', ')') | ('[', ']')
    );
    if matches {
        *depth = top;
    }
    matches
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::parse_interface_config_str;
    use crate::MeshError;

    const FIXTURE: &str = "fixture/interfaces";

    fn assert_parse(content: &str, line: usize, detail: &str) {
        let error = parse_interface_config_str(content, Path::new(FIXTURE)).unwrap_err();
        let expected_message = format!("{FIXTURE}: {detail}");
        match &error {
            MeshError::Parse {
                line: actual_line,
                message,
            } => {
                assert_eq!(*actual_line, line);
                assert_eq!(message, &expected_message);
            }
            other => panic!("expected Parse error, got {other:?}"),
        }
        assert_eq!(
            error.to_string(),
            format!("line {line}: {expected_message}")
        );
        assert_eq!(error.to_string().matches(FIXTURE).count(), 1);
    }

    fn entry_with_regions(regions: &str) -> String {
        format!(
            "interfaces {{ membrane {{ regions {regions} faceZone membrane_wall; orientation fluid_to_solid; }} }}"
        )
    }

    #[test]
    fn parses_positive_orientation_as_sign_convention() {
        let content = r#"
        "FoamFile" { ignored interfaces; }
        FoamFile
        {
            version 2.0;
            class dictionary;
            object interfaces;
        }

        interfaces
        {
            reactor_wall
            {
                regions (fluid solid);
                faceZone wall_interface;
                orientation fluid_to_solid;
                model heatTransfer;
            }
        }
        "#;

        let config = parse_interface_config_str(content, Path::new(FIXTURE)).unwrap();
        assert_eq!(config.path, Path::new(FIXTURE));
        assert_eq!(config.entries.len(), 1);
        assert_eq!(config.entries[0].orientation.positive_from, "fluid");
        assert_eq!(config.entries[0].orientation.positive_to, "solid");
        assert_eq!(config.entries[0].model, "heatTransfer");
    }

    #[test]
    fn accepts_reversed_positive_orientation() {
        let content = r#"
        interfaces
        {
            membrane
            {
                orientation permeate_to_retentate;
                "model" ignored;
                faceZone membrane_wall;
                regions (retentate permeate);
            }
        }
        "#;

        let config = parse_interface_config_str(content, Path::new(FIXTURE)).unwrap();
        assert_eq!(config.entries[0].orientation.positive_from, "permeate");
        assert_eq!(config.entries[0].orientation.positive_to, "retentate");
        assert_eq!(config.entries[0].model, "none");
    }

    #[test]
    fn requires_exactly_one_ordinary_interfaces_block() {
        assert_parse(
            "FoamFile { class dictionary; }",
            1,
            "missing ordinary 'interfaces' block",
        );
        assert_parse(
            r#""interfaces" { ignored value; }"#,
            1,
            "missing ordinary 'interfaces' block",
        );
        assert_parse(
            "interfaces {} \"interfaces\" {} interfaces {}",
            1,
            "duplicate ordinary 'interfaces' block",
        );
    }

    #[test]
    fn rejects_duplicate_interface_names_and_required_keys() {
        assert_parse(
            "interfaces { shared { regions (fluid solid); faceZone first; orientation fluid_to_solid; } shared { regions (fluid solid); faceZone second; orientation fluid_to_solid; } }",
            1,
            "duplicate interface name",
        );

        for duplicate in [
            "regions (fluid solid); regions (fluid solid);",
            "faceZone first; faceZone second;",
            "orientation fluid_to_solid; orientation solid_to_fluid;",
            "model heatTransfer; model none;",
        ] {
            let content = format!(
                "interfaces {{ shared {{ regions (fluid solid); faceZone first; orientation fluid_to_solid; {duplicate} }} }}"
            );
            assert_parse(&content, 1, "duplicate interface entry key");
        }
    }

    #[test]
    fn regions_are_exactly_two_ordinary_tokens_with_a_semicolon() {
        for regions in [
            "();",
            "(fluid);",
            "(fluid solid gas);",
            "(fluid \"solid\");",
        ] {
            assert_parse(
                &entry_with_regions(regions),
                1,
                "regions must contain exactly two ordinary entries",
            );
        }

        assert_parse(
            &entry_with_regions("(fluid fluid);"),
            1,
            "regions must name two different ordinary entries",
        );
        assert_parse(
            &entry_with_regions("(fluid solid)"),
            1,
            "unexpected dictionary token",
        );
    }

    #[test]
    fn required_keys_and_scalar_values_are_provenance_safe_and_terminated() {
        for (entry, detail) in [
            (
                r#""regions" (fluid solid); faceZone membrane_wall; orientation fluid_to_solid;"#,
                "missing ordinary 'regions' in interface entry",
            ),
            (
                r#"regions (fluid solid); "faceZone" membrane_wall; orientation fluid_to_solid;"#,
                "missing ordinary 'faceZone' in interface entry",
            ),
            (
                r#"regions (fluid solid); faceZone membrane_wall; "orientation" fluid_to_solid;"#,
                "missing ordinary 'orientation' in interface entry",
            ),
        ] {
            assert_parse(
                &format!("interfaces {{ membrane {{ {entry} }} }}"),
                1,
                detail,
            );
        }

        for (entry, detail) in [
            (
                r#"regions (fluid solid); faceZone "membrane_wall"; orientation fluid_to_solid;"#,
                "faceZone value must be an ordinary token",
            ),
            (
                r#"regions (fluid solid); faceZone membrane_wall; orientation "fluid_to_solid";"#,
                "orientation value must be an ordinary token",
            ),
            (
                r#"regions (fluid solid); faceZone membrane_wall; orientation fluid_to_solid; model "transport";"#,
                "model value must be an ordinary token",
            ),
        ] {
            assert_parse(
                &format!("interfaces {{ membrane {{ {entry} }} }}"),
                1,
                detail,
            );
        }

        for entry in [
            "regions (fluid solid); faceZone membrane_wall orientation fluid_to_solid;",
            "regions (fluid solid); faceZone membrane_wall; orientation fluid_to_solid",
            "regions (fluid solid); faceZone membrane_wall; orientation fluid_to_solid; model transport",
        ] {
            assert_parse(
                &format!("interfaces {{ membrane {{ {entry} }} }}"),
                1,
                "unexpected dictionary token",
            );
        }

        assert_parse(
            "interfaces {\nmembrane {\norientation wrong_order;\nregions (fluid solid);\nfaceZone membrane_wall;\n}\n}",
            3,
            "orientation must match the declared ordinary region order",
        );

        let quoted_optional = parse_interface_config_str(
            r#"interfaces { membrane { regions (fluid solid); faceZone membrane_wall; orientation fluid_to_solid; "model" transport; } }"#,
            Path::new(FIXTURE),
        )
        .unwrap();
        assert_eq!(quoted_optional.entries[0].model, "none");
    }

    #[test]
    fn exact_unknown_skip_preserves_following_required_sentinels() {
        let content = r#"
        preamble { nested (one [two]); }
        interfaces
        {
            membrane
            {
                ignoredScalar 17;
                ignoredGroup (alpha [beta gamma]);
                ignoredBlock { nested { value 1; } }
                regions (retentate permeate);
                faceZone membrane_wall;
                orientation retentate_to_permeate;
                model transport;
            }
        }
        trailer finished;
        "#;
        let config = parse_interface_config_str(content, Path::new(FIXTURE)).unwrap();
        assert_eq!(config.entries.len(), 1);
        assert_eq!(config.entries[0].regions, ["retentate", "permeate"]);
        assert_eq!(config.entries[0].face_zone, "membrane_wall");
        assert_eq!(config.entries[0].model, "transport");

        assert_parse(
            "interfaces { membrane { ignored innocent faceZone hijacked; regions (fluid solid); faceZone membrane_wall; orientation fluid_to_solid; } }",
            1,
            "dictionary value is missing a semicolon",
        );
        assert_parse(
            "interfaces { membrane { ignoredGroup (alpha beta) regions (fluid solid); faceZone membrane_wall; orientation fluid_to_solid; } }",
            1,
            "dictionary value is missing a semicolon",
        );
    }

    #[test]
    fn structural_punctuation_cannot_be_a_name_or_key() {
        assert_parse(
            "interfaces { ; }",
            1,
            "interface name must be an ordinary token",
        );
        assert_parse(
            r#"interfaces { "membrane" {} }"#,
            1,
            "interface name must be an ordinary token",
        );
        assert_parse(
            "interfaces { membrane { ; } }",
            1,
            "interface key must not be structural punctuation",
        );
        assert_parse(
            "interfaces { membrane { regions (fluid solid); faceZone membrane_wall; orientation fluid_to_solid; } } ]",
            1,
            "dictionary nesting counter underflow",
        );
    }
}
