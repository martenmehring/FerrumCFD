use std::fs;
use std::path::{Path, PathBuf};

use crate::dictionary::{TokenCursor, tokenize};
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

    while let Some(token) = cursor.peek()? {
        match token.value.as_str() {
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

    while cursor.peek()?.is_none_or(|token| token.value != "}") {
        let name = cursor.next_required()?.value;
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

    while cursor.peek()?.is_none_or(|token| token.value != "}") {
        let key = cursor.next_required()?;
        match key.value.as_str() {
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
    let orientation = parse_orientation(&raw_orientation.value, &regions, cursor.path())?;
    let face_zone = face_zone.ok_or_else(|| missing_key(cursor.path(), &name, "faceZone"))?;

    Ok(InterfaceConfigEntry {
        name,
        regions,
        face_zone: face_zone.value,
        orientation,
        model: model.map_or_else(|| "none".to_string(), |token| token.value),
    })
}

fn read_regions(cursor: &mut TokenCursor) -> Result<[String; 2]> {
    cursor.expect("(")?;
    let mut values = Vec::new();
    while cursor.peek()?.is_none_or(|token| token.value != ")") {
        values.push(cursor.next_required()?.value);
    }
    cursor.expect(")")?;
    cursor.expect_optional(";")?;

    if values.len() != 2 {
        return Err(MeshError::InvalidInput(format!(
            "regions must contain exactly two entries in {}",
            cursor.path().display()
        )));
    }

    Ok([values[0].clone(), values[1].clone()])
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
