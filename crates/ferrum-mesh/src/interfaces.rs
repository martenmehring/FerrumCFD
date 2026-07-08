use std::fs;
use std::path::{Path, PathBuf};

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
    let tokens = tokenize(content);
    let mut cursor = TokenCursor::new(path, tokens);
    let mut entries = Vec::new();

    while let Some(token) = cursor.peek() {
        match token {
            "FoamFile" => {
                cursor.next_required()?;
                cursor.skip_braced_block()?;
            }
            "interfaces" => {
                cursor.next_required()?;
                cursor.expect("{")?;
                entries = parse_interfaces_block(&mut cursor)?;
            }
            _ => {
                cursor.next_required()?;
            }
        }
    }

    Ok(InterfaceConfig {
        path: path.to_path_buf(),
        entries,
    })
}

fn parse_interfaces_block(cursor: &mut TokenCursor) -> Result<Vec<InterfaceConfigEntry>> {
    let mut entries = Vec::new();

    while !cursor.peek_is("}")? {
        let name = cursor.next_required()?;
        cursor.expect("{")?;
        entries.push(parse_interface_entry(cursor, name)?);
    }
    cursor.expect("}")?;

    Ok(entries)
}

fn parse_interface_entry(cursor: &mut TokenCursor, name: String) -> Result<InterfaceConfigEntry> {
    let mut regions = None;
    let mut face_zone = None;
    let mut orientation = None;
    let mut model = None;

    while !cursor.peek_is("}")? {
        let key = cursor.next_required()?;
        match key.as_str() {
            "regions" => regions = Some(read_regions(cursor)?),
            "faceZone" => {
                face_zone = Some(cursor.next_required()?);
                cursor.expect_optional(";")?;
            }
            "orientation" => {
                orientation = Some(cursor.next_required()?);
                cursor.expect_optional(";")?;
            }
            "model" => {
                model = Some(cursor.next_required()?);
                cursor.expect_optional(";")?;
            }
            _ => cursor.skip_value_or_block()?,
        }
    }
    cursor.expect("}")?;

    let regions = regions.ok_or_else(|| missing_key(cursor.path(), &name, "regions"))?;
    let raw_orientation =
        orientation.ok_or_else(|| missing_key(cursor.path(), &name, "orientation"))?;
    let orientation = parse_orientation(&raw_orientation, &regions, cursor.path())?;
    let face_zone = face_zone.ok_or_else(|| missing_key(cursor.path(), &name, "faceZone"))?;

    Ok(InterfaceConfigEntry {
        name,
        regions,
        face_zone,
        orientation,
        model: model.unwrap_or_else(|| "none".to_string()),
    })
}

fn read_regions(cursor: &mut TokenCursor) -> Result<[String; 2]> {
    cursor.expect("(")?;
    let mut values = Vec::new();
    while !cursor.peek_is(")")? {
        values.push(cursor.next_required()?);
    }
    cursor.expect(")")?;
    cursor.expect_optional(";")?;

    if values.len() != 2 {
        return Err(MeshError::InvalidInput(format!(
            "regions must contain exactly two entries in {}",
            cursor.path().display()
        )));
    }

    Ok([values.remove(0), values.remove(0)])
}

fn parse_orientation(
    value: &str,
    regions: &[String; 2],
    path: &Path,
) -> Result<InterfaceOrientation> {
    let first_to_second = format!("{}_to_{}", regions[0], regions[1]);
    if value == first_to_second {
        return Ok(InterfaceOrientation {
            positive_from: regions[0].clone(),
            positive_to: regions[1].clone(),
        });
    }

    let second_to_first = format!("{}_to_{}", regions[1], regions[0]);
    if value == second_to_first {
        return Ok(InterfaceOrientation {
            positive_from: regions[1].clone(),
            positive_to: regions[0].clone(),
        });
    }

    Err(MeshError::InvalidInput(format!(
        "orientation '{}' in {} must be '{}' or '{}'",
        value,
        path.display(),
        first_to_second,
        second_to_first
    )))
}

fn same_region_pair(left: &str, right: &str, pair: &[String; 2]) -> bool {
    (left == pair[0] && right == pair[1]) || (left == pair[1] && right == pair[0])
}

fn missing_key(path: &Path, entry: &str, key: &str) -> MeshError {
    MeshError::InvalidInput(format!(
        "missing '{key}' in interface '{entry}' in {}",
        path.display()
    ))
}

fn tokenize(content: &str) -> Vec<Token> {
    let mut tokens = Vec::new();
    for (line_index, line) in content.lines().enumerate() {
        let mut current = String::new();
        let mut chars = line.chars().peekable();

        while let Some(ch) = chars.next() {
            if ch == '/' && chars.peek() == Some(&'/') {
                break;
            }

            if ch == '"' {
                current.push(ch);
                for quoted in chars.by_ref() {
                    current.push(quoted);
                    if quoted == '"' {
                        break;
                    }
                }
                continue;
            }

            if ch.is_whitespace() {
                push_token(&mut tokens, &mut current, line_index + 1);
                continue;
            }

            if matches!(ch, '{' | '}' | '(' | ')' | ';') {
                push_token(&mut tokens, &mut current, line_index + 1);
                tokens.push(Token {
                    value: ch.to_string(),
                    line: line_index + 1,
                });
                continue;
            }

            current.push(ch);
        }
        push_token(&mut tokens, &mut current, line_index + 1);
    }
    tokens
}

fn push_token(tokens: &mut Vec<Token>, current: &mut String, line: usize) {
    if current.is_empty() {
        return;
    }

    tokens.push(Token {
        value: current.trim_matches('"').to_string(),
        line,
    });
    current.clear();
}

#[derive(Clone)]
struct Token {
    value: String,
    line: usize,
}

struct TokenCursor {
    path: PathBuf,
    tokens: Vec<Token>,
    index: usize,
}

impl TokenCursor {
    fn new(path: &Path, tokens: Vec<Token>) -> Self {
        Self {
            path: path.to_path_buf(),
            tokens,
            index: 0,
        }
    }

    fn path(&self) -> &Path {
        &self.path
    }

    fn peek(&self) -> Option<&str> {
        self.tokens
            .get(self.index)
            .map(|token| token.value.as_str())
    }

    fn peek_is(&self, expected: &str) -> Result<bool> {
        Ok(self.peek_required()? == expected)
    }

    fn expect(&mut self, expected: &str) -> Result<()> {
        let token = self.next_token()?;
        if token.value == expected {
            Ok(())
        } else {
            Err(MeshError::Parse {
                line: token.line,
                message: format!("expected '{expected}' but found '{}'", token.value),
            })
        }
    }

    fn expect_optional(&mut self, expected: &str) -> Result<()> {
        if self.peek() == Some(expected) {
            self.next_required()?;
        }
        Ok(())
    }

    fn next_required(&mut self) -> Result<String> {
        Ok(self.next_token()?.value)
    }

    fn peek_required(&self) -> Result<&str> {
        self.tokens
            .get(self.index)
            .map(|token| token.value.as_str())
            .ok_or_else(|| {
                MeshError::InvalidInput(format!("unexpected EOF in {}", self.path.display()))
            })
    }

    fn next_token(&mut self) -> Result<Token> {
        let token = self.tokens.get(self.index).cloned().ok_or_else(|| {
            MeshError::InvalidInput(format!("unexpected EOF in {}", self.path.display()))
        })?;
        self.index += 1;
        Ok(token)
    }

    fn skip_braced_block(&mut self) -> Result<()> {
        self.expect("{")?;
        let mut depth = 1;
        while depth > 0 {
            let token = self.next_required()?;
            match token.as_str() {
                "{" => depth += 1,
                "}" => depth -= 1,
                _ => {}
            }
        }
        Ok(())
    }

    fn skip_value_or_block(&mut self) -> Result<()> {
        if self.peek() == Some("{") {
            self.skip_braced_block()?;
            return Ok(());
        }

        while let Some(token) = self.peek() {
            if token == ";" {
                self.next_required()?;
                break;
            }
            if token == "}" {
                break;
            }
            if token == "{" {
                self.skip_braced_block()?;
                break;
            }
            self.next_required()?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::parse_interface_config_str;

    #[test]
    fn parses_positive_orientation_as_sign_convention() {
        let content = r#"
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

        let config = parse_interface_config_str(content, Path::new("interfaces")).unwrap();
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
                regions (retentate permeate);
                faceZone membrane_wall;
                orientation permeate_to_retentate;
            }
        }
        "#;

        let config = parse_interface_config_str(content, Path::new("interfaces")).unwrap();
        assert_eq!(config.entries[0].orientation.positive_from, "permeate");
        assert_eq!(config.entries[0].orientation.positive_to, "retentate");
        assert_eq!(config.entries[0].model, "none");
    }
}
