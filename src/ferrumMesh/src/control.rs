use std::path::{Path, PathBuf};

use crate::dictionary::{TokenCursor, TokenProvenance, tokenize};
use crate::{MeshError, Result};

#[derive(Debug)]
pub struct ControlDict {
    pub path: PathBuf,
    pub application: Option<String>,
    pub solver: Option<String>,
    pub start_from: String,
    pub start_time: Option<f64>,
    pub stop_at: String,
    pub end_time: Option<f64>,
    pub delta_t: Option<f64>,
    pub write_control: String,
    pub write_interval: Option<f64>,
}

#[derive(Debug)]
pub struct ControlValidation {
    pub warnings: Vec<String>,
}

pub fn read_control_dict(case_dir: &Path) -> Result<ControlDict> {
    let path = case_dir.join("system").join("controlDict");
    let content = crate::case_input::CaseInput::new(case_dir)
        .required("system/controlDict")
        .map_err(append_control_guidance)?;
    let mut control = parse_control_dict_str(&content, &path)?;
    control.path = path;
    Ok(control)
}

fn append_control_guidance(error: MeshError) -> MeshError {
    let MeshError::InvalidInput(mut message) = error else {
        return error;
    };
    const GUIDANCE: &str = "; run initFerrumCase first";
    if message.try_reserve(GUIDANCE.len()).is_err() {
        return MeshError::OutOfMemory;
    }
    message.push_str(GUIDANCE);
    MeshError::InvalidInput(message)
}

pub fn validate_control_dict(control: &ControlDict) -> ControlValidation {
    let mut warnings = Vec::new();

    if control.application.is_none() {
        warnings.push("missing application".to_string());
    }

    if !matches!(
        control.start_from.as_str(),
        "startTime" | "firstTime" | "latestTime"
    ) {
        warnings.push(format!(
            "startFrom '{}' is not recognized",
            control.start_from
        ));
    }
    if control.start_from == "startTime" && control.start_time.is_none() {
        warnings.push("startFrom startTime requires startTime".to_string());
    }
    if let Some(start_time) = control.start_time {
        validate_finite_number("startTime", start_time, &mut warnings);
    }

    if !matches!(
        control.stop_at.as_str(),
        "endTime" | "writeNow" | "noWriteNow" | "nextWrite"
    ) {
        warnings.push(format!("stopAt '{}' is not recognized", control.stop_at));
    }
    if control.stop_at == "endTime" && control.end_time.is_none() {
        warnings.push("stopAt endTime requires endTime".to_string());
    }
    if let Some(end_time) = control.end_time {
        validate_finite_number("endTime", end_time, &mut warnings);
    }
    if let (Some(start_time), Some(end_time)) = (control.start_time, control.end_time)
        && control.stop_at == "endTime"
        && end_time < start_time
    {
        warnings.push(format!(
            "endTime {end_time} is earlier than startTime {start_time}"
        ));
    }

    match control.delta_t {
        Some(delta_t) if delta_t.is_finite() && delta_t > 0.0 => {}
        Some(delta_t) => warnings.push(format!(
            "deltaT must be positive and finite, found {delta_t}"
        )),
        None => warnings.push("missing deltaT".to_string()),
    }

    if !matches!(
        control.write_control.as_str(),
        "timeStep" | "runTime" | "adjustableRunTime" | "cpuTime" | "clockTime" | "none"
    ) {
        warnings.push(format!(
            "writeControl '{}' is not recognized",
            control.write_control
        ));
    }
    if control.write_control != "none" {
        match control.write_interval {
            Some(write_interval) if write_interval.is_finite() && write_interval > 0.0 => {}
            Some(write_interval) => warnings.push(format!(
                "writeInterval must be positive and finite, found {write_interval}"
            )),
            None => warnings.push("writeControl requires writeInterval".to_string()),
        }
    }

    ControlValidation { warnings }
}

fn parse_control_dict_str(content: &str, path: &Path) -> Result<ControlDict> {
    let mut cursor = tokenize(path, content)?.into_cursor();
    let mut builder = ControlDictBuilder::new(path);

    while let Some(token) = cursor.peek()? {
        if token.value == "FoamFile" && token.provenance == TokenProvenance::Ordinary {
            cursor.next_required()?;
            cursor.skip_braced_block()?;
            continue;
        }

        let key = cursor.next_required()?;
        if key.provenance == TokenProvenance::Structural {
            return Err(MeshError::InvalidInput(format!(
                "unexpected dictionary token in {}",
                path.display()
            )));
        }
        if key.provenance != TokenProvenance::Ordinary
            || !matches!(
                key.value.as_str(),
                "application"
                    | "solver"
                    | "startFrom"
                    | "startTime"
                    | "stopAt"
                    | "endTime"
                    | "deltaT"
                    | "writeControl"
                    | "writeInterval"
            )
        {
            skip_control_value(&mut cursor)?;
            continue;
        }
        let values = cursor.read_value_until_semicolon()?;
        match key.value.as_str() {
            "application" => builder.application = Some(single_value(&values, "application")?),
            "solver" => builder.solver = Some(single_value(&values, "solver")?),
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

fn skip_control_value(cursor: &mut TokenCursor) -> Result<()> {
    cursor.skip_exact_value_or_block()
}

fn validate_finite_number(label: &str, value: f64, warnings: &mut Vec<String>) {
    if !value.is_finite() {
        warnings.push(format!("{label} must be finite, found {value}"));
    }
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
    solver: Option<String>,
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
            solver: None,
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
            application: self.application,
            solver: self.solver,
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

    use super::{parse_control_dict_str, validate_control_dict};

    #[test]
    fn parses_basic_control_dict() {
        let content = r#"
        FoamFile
        {
            version 2.0;
            class dictionary;
            object controlDict;
        }

        application ferrumRun;
        solver incompressibleFluid;
        startFrom startTime;
        startTime 0;
        stopAt endTime;
        endTime 10;
        deltaT 0.05;
        writeControl timeStep;
        writeInterval 20;
        "#;

        let control = parse_control_dict_str(content, Path::new("controlDict")).unwrap();
        assert_eq!(control.application.as_deref(), Some("ferrumRun"));
        assert_eq!(control.solver.as_deref(), Some("incompressibleFluid"));
        assert_eq!(control.start_from, "startTime");
        assert_eq!(control.start_time, Some(0.0));
        assert_eq!(control.end_time, Some(10.0));
        assert_eq!(control.delta_t, Some(0.05));
        assert_eq!(control.write_interval, Some(20.0));
    }

    #[test]
    fn uses_openfoam_like_defaults_for_missing_optional_values() {
        let control = parse_control_dict_str("", Path::new("controlDict")).unwrap();
        assert_eq!(control.application, None);
        assert_eq!(control.solver, None);
        assert_eq!(control.start_from, "startTime");
        assert_eq!(control.stop_at, "endTime");
        assert_eq!(control.write_control, "timeStep");
    }

    #[test]
    fn validates_basic_control_dict() {
        let control = parse_control_dict_str(
            r#"
            application ferrumRun;
            solver incompressibleFluid;
            startFrom startTime;
            startTime 0;
            stopAt endTime;
            endTime 1;
            deltaT 0.1;
            writeControl timeStep;
            writeInterval 1;
            "#,
            Path::new("controlDict"),
        )
        .unwrap();

        let validation = validate_control_dict(&control);

        assert!(validation.warnings.is_empty());
    }

    #[test]
    fn warns_for_invalid_time_controls() {
        let control = parse_control_dict_str(
            r#"
            startFrom invalidStart;
            startTime 2;
            stopAt endTime;
            endTime 1;
            deltaT 0;
            writeControl strange;
            writeInterval -1;
            "#,
            Path::new("controlDict"),
        )
        .unwrap();

        let validation = validate_control_dict(&control);

        assert!(
            validation
                .warnings
                .iter()
                .any(|warning| warning.contains("startFrom"))
        );
        assert!(
            validation
                .warnings
                .iter()
                .any(|warning| warning.contains("endTime 1 is earlier"))
        );
        assert!(
            validation
                .warnings
                .iter()
                .any(|warning| warning.contains("deltaT"))
        );
        assert!(
            validation
                .warnings
                .iter()
                .any(|warning| warning.contains("writeControl"))
        );
    }

    #[test]
    fn quoted_and_unknown_values_preserve_control_sentinels() {
        let control = parse_control_dict_str(
            r#"
            "application" ignored; application ferrumRun;
            unknown { deltaT 99; } deltaT 0.25;
            list (application swallowed); solver fluid;
            quotedSemi ";"; writeInterval 4;
            "#,
            Path::new("controlDict"),
        )
        .unwrap();

        assert_eq!(control.application.as_deref(), Some("ferrumRun"));
        assert_eq!(control.solver.as_deref(), Some("fluid"));
        assert_eq!(control.delta_t, Some(0.25));
        assert_eq!(control.write_interval, Some(4.0));
    }

    #[test]
    fn unknown_entry_cannot_consume_a_structural_delimiter_as_its_value() {
        let error = parse_control_dict_str("unknown ;", Path::new("controlDict")).unwrap_err();
        assert!(error.to_string().contains("dictionary value is missing"));
    }

    #[test]
    fn openfoam_directives_fail_with_a_specific_source_location() {
        for directive in [
            "#include \"initialConditions\"",
            "#includeFunc residuals",
            "#include \"initialConditions\";",
            "#includeFunc residuals;",
        ] {
            let content = format!("{directive}\napplication ferrumRun;\n");
            let error = parse_control_dict_str(&content, Path::new("controlDict"))
                .expect_err("unresolved directives must fail closed");
            assert_eq!(
                error.to_string(),
                "line 1: controlDict: unsupported dictionary directive"
            );
        }

        for quoted in [r##""#include" inert;"##, r##""#includeFunc" inert;"##] {
            let content = format!("{quoted}\napplication ferrumRun;\n");
            let control = parse_control_dict_str(&content, Path::new("controlDict"))
                .expect("quoted directive spelling must remain inert data");
            assert_eq!(control.application.as_deref(), Some("ferrumRun"));
        }

        let nested = "unknown { #include \"initialConditions\"; }\napplication ferrumRun;\n";
        let error = parse_control_dict_str(nested, Path::new("controlDict"))
            .expect_err("discarded blocks cannot hide unsupported directives");
        assert_eq!(
            error.to_string(),
            "line 1: controlDict: unsupported dictionary directive"
        );
    }

    #[test]
    fn quoted_and_unknown_control_value_matrix_preserves_sentinels() {
        for key in ["unknown", r#""unknown""#] {
            for value in [
                "scalar;",
                r#""quoted scalar";"#,
                "(hidden values);",
                "[hidden values];",
                "{ hidden values; }",
                "{ hidden values; };",
            ] {
                let content = format!("{key} {value} application ferrumRun; deltaT 0.125;");
                let control = parse_control_dict_str(&content, Path::new("controlDict")).unwrap();
                assert_eq!(control.application.as_deref(), Some("ferrumRun"));
                assert_eq!(control.delta_t, Some(0.125));
            }
        }
    }

    #[test]
    fn unterminated_control_values_cannot_redispatch_reserved_keys() {
        for content in [
            "unknown scalar application ferrumRun;",
            r#""unknown" "scalar" application ferrumRun;"#,
            "unknown (hidden values) application ferrumRun;",
            "unknown [hidden values] application ferrumRun;",
        ] {
            let error = parse_control_dict_str(content, Path::new("controlDict"))
                .expect_err("missing structural semicolon must fail before the sentinel");
            assert!(
                error
                    .to_string()
                    .contains("dictionary value is missing a semicolon"),
                "unexpected error for {content:?}: {error}"
            );
        }
    }

    #[test]
    fn known_control_scalars_require_structural_semicolons() {
        for content in [
            "application ferrumRun solver incompressibleFluid;",
            "deltaT 0.5 writeInterval 4;",
        ] {
            let error = parse_control_dict_str(content, Path::new("controlDict"))
                .expect_err("known scalar without semicolon must not redispatch its sentinel");
            assert!(
                error.to_string().contains("must be a single value"),
                "unexpected error for {content:?}: {error}"
            );
        }
    }

    #[test]
    fn structural_control_key_fails_closed() {
        let error = parse_control_dict_str("; application ferrumRun;", Path::new("controlDict"))
            .unwrap_err();
        assert!(error.to_string().contains("unexpected dictionary token"));
    }
}
