use std::fs;
use std::path::{Path, PathBuf};

use crate::dictionary::{TokenCursor, tokenize};
use crate::{MeshError, Result};

#[derive(Debug)]
pub struct ControlDict {
    pub path: PathBuf,
    pub application: String,
    pub start_from: String,
    pub start_time: Option<f64>,
    pub stop_at: String,
    pub end_time: Option<f64>,
    pub delta_t: Option<f64>,
    pub write_control: String,
    pub write_interval: Option<f64>,
}

pub fn read_control_dict(case_dir: &Path) -> Result<ControlDict> {
    let path = case_dir.join("system").join("controlDict");
    let content = fs::read_to_string(&path).map_err(|error| {
        MeshError::InvalidInput(format!(
            "could not read {}; run initFerrumCase first ({error})",
            path.display()
        ))
    })?;
    let mut control = parse_control_dict_str(&content, &path)?;
    control.path = path;
    Ok(control)
}

fn parse_control_dict_str(content: &str, path: &Path) -> Result<ControlDict> {
    let tokens = tokenize(content);
    let mut cursor = TokenCursor::new(path, tokens);
    let mut builder = ControlDictBuilder::new(path);

    while let Some(token) = cursor.peek() {
        if token == "FoamFile" {
            cursor.next_required()?;
            cursor.skip_braced_block()?;
            continue;
        }

        let key = cursor.next_required()?;
        let values = cursor.read_value_until_semicolon()?;
        match key.as_str() {
            "application" => builder.application = Some(single_value(&values, "application")?),
            "startFrom" => builder.start_from = Some(single_value(&values, "startFrom")?),
            "startTime" => builder.start_time = Some(number_value(&values, "startTime", path)?),
            "stopAt" => builder.stop_at = Some(single_value(&values, "stopAt")?),
            "endTime" => builder.end_time = Some(number_value(&values, "endTime", path)?),
            "deltaT" => builder.delta_t = Some(number_value(&values, "deltaT", path)?),
            "writeControl" => builder.write_control = Some(single_value(&values, "writeControl")?),
            "writeInterval" => {
                builder.write_interval = Some(number_value(&values, "writeInterval", path)?);
            }
            _ => {}
        }
    }

    builder.finish()
}

fn single_value(values: &[String], label: &str) -> Result<String> {
    if values.len() == 1 {
        return Ok(values[0].clone());
    }

    Err(MeshError::InvalidInput(format!(
        "controlDict entry '{label}' must be a single value"
    )))
}

fn number_value(values: &[String], label: &str, path: &Path) -> Result<f64> {
    let value = single_value(values, label)?;
    value.parse::<f64>().map_err(|_| {
        MeshError::InvalidInput(format!(
            "controlDict entry '{label}' in {} must be numeric",
            path.display()
        ))
    })
}

struct ControlDictBuilder {
    path: PathBuf,
    application: Option<String>,
    start_from: Option<String>,
    start_time: Option<f64>,
    stop_at: Option<String>,
    end_time: Option<f64>,
    delta_t: Option<f64>,
    write_control: Option<String>,
    write_interval: Option<f64>,
}

impl ControlDictBuilder {
    fn new(path: &Path) -> Self {
        Self {
            path: path.to_path_buf(),
            application: None,
            start_from: None,
            start_time: None,
            stop_at: None,
            end_time: None,
            delta_t: None,
            write_control: None,
            write_interval: None,
        }
    }

    fn finish(self) -> Result<ControlDict> {
        Ok(ControlDict {
            path: self.path,
            application: self
                .application
                .unwrap_or_else(|| "ferrumSolver".to_string()),
            start_from: self.start_from.unwrap_or_else(|| "startTime".to_string()),
            start_time: self.start_time,
            stop_at: self.stop_at.unwrap_or_else(|| "endTime".to_string()),
            end_time: self.end_time,
            delta_t: self.delta_t,
            write_control: self.write_control.unwrap_or_else(|| "timeStep".to_string()),
            write_interval: self.write_interval,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::parse_control_dict_str;

    #[test]
    fn parses_basic_control_dict() {
        let content = r#"
        FoamFile
        {
            version 2.0;
            class dictionary;
            object controlDict;
        }

        application ferrumSolver;
        startFrom startTime;
        startTime 0;
        stopAt endTime;
        endTime 10;
        deltaT 0.05;
        writeControl timeStep;
        writeInterval 20;
        "#;

        let control = parse_control_dict_str(content, Path::new("controlDict")).unwrap();
        assert_eq!(control.application, "ferrumSolver");
        assert_eq!(control.start_from, "startTime");
        assert_eq!(control.start_time, Some(0.0));
        assert_eq!(control.end_time, Some(10.0));
        assert_eq!(control.delta_t, Some(0.05));
        assert_eq!(control.write_interval, Some(20.0));
    }

    #[test]
    fn uses_openfoam_like_defaults_for_missing_optional_values() {
        let control = parse_control_dict_str("", Path::new("controlDict")).unwrap();
        assert_eq!(control.application, "ferrumSolver");
        assert_eq!(control.start_from, "startTime");
        assert_eq!(control.stop_at, "endTime");
        assert_eq!(control.write_control, "timeStep");
    }
}
