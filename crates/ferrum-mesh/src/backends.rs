use std::fs;
use std::path::{Path, PathBuf};

use crate::dictionary::{TokenCursor, tokenize};
use crate::{MeshError, Result};

#[derive(Debug)]
pub struct BackendConfig {
    pub path: PathBuf,
    pub default: BackendChoice,
    pub sections: Vec<BackendSection>,
    pub cpu: CpuConfig,
    pub gpu: GpuConfig,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BackendChoice {
    Cpu,
    Gpu,
    Auto,
}

#[derive(Debug)]
pub struct BackendSection {
    pub name: String,
    pub entries: Vec<BackendSelection>,
}

#[derive(Debug)]
pub struct BackendSelection {
    pub step: String,
    pub choice: BackendChoice,
}

#[derive(Debug)]
pub struct CpuConfig {
    pub threads: String,
    pub thread_pinning: String,
    pub numa: String,
}

#[derive(Debug)]
pub struct GpuConfig {
    pub backend: String,
    pub devices: Vec<String>,
    pub multi_gpu: String,
    pub precision: String,
}

impl Default for CpuConfig {
    fn default() -> Self {
        Self {
            threads: "auto".to_string(),
            thread_pinning: "off".to_string(),
            numa: "auto".to_string(),
        }
    }
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            backend: "auto".to_string(),
            devices: vec!["auto".to_string()],
            multi_gpu: "auto".to_string(),
            precision: "f64".to_string(),
        }
    }
}

impl std::fmt::Display for BackendChoice {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cpu => formatter.write_str("cpu"),
            Self::Gpu => formatter.write_str("gpu"),
            Self::Auto => formatter.write_str("auto"),
        }
    }
}

pub fn read_backend_config(case_dir: &Path) -> Result<Option<BackendConfig>> {
    let path = case_dir.join("system").join("ferrumBackends");
    if !path.exists() {
        return Ok(None);
    }

    let content = fs::read_to_string(&path).map_err(|error| {
        MeshError::InvalidInput(format!("could not read {} ({error})", path.display()))
    })?;
    let mut config = parse_backend_config_str(&content, &path)?;
    config.path = path;
    Ok(Some(config))
}

fn parse_backend_config_str(content: &str, path: &Path) -> Result<BackendConfig> {
    let tokens = tokenize(content);
    let mut cursor = TokenCursor::new(path, tokens);
    let mut builder = BackendConfigBuilder::new(path);

    while let Some(token) = cursor.peek() {
        if token == "FoamFile" {
            cursor.next_required()?;
            cursor.skip_braced_block()?;
            continue;
        }

        if token == "ferrumBackends" && cursor.peek_next() == Some("{") {
            cursor.next_required()?;
            cursor.expect("{")?;
            parse_backend_entries(&mut cursor, &mut builder, true)?;
            continue;
        }

        parse_backend_entry(&mut cursor, &mut builder)?;
    }

    builder.finish()
}

fn parse_backend_entries(
    cursor: &mut TokenCursor,
    builder: &mut BackendConfigBuilder,
    stop_at_close: bool,
) -> Result<()> {
    while cursor.peek().is_some() {
        if stop_at_close && cursor.peek_is("}")? {
            cursor.expect("}")?;
            return Ok(());
        }
        parse_backend_entry(cursor, builder)?;
    }
    Ok(())
}

fn parse_backend_entry(cursor: &mut TokenCursor, builder: &mut BackendConfigBuilder) -> Result<()> {
    let key = cursor.next_required()?;
    match key.as_str() {
        "default" => {
            let choice = parse_backend_choice(&cursor.next_required()?, cursor.path())?;
            cursor.expect_optional(";")?;
            builder.default = Some(choice);
        }
        "gpu" => {
            cursor.expect("{")?;
            builder.gpu = parse_gpu_block(cursor)?;
        }
        "cpu" => {
            cursor.expect("{")?;
            builder.cpu = parse_cpu_block(cursor)?;
        }
        _ => {
            cursor.expect("{")?;
            builder.sections.push(parse_backend_section(cursor, key)?);
        }
    }
    Ok(())
}

fn parse_backend_section(cursor: &mut TokenCursor, name: String) -> Result<BackendSection> {
    let mut entries = Vec::new();

    while !cursor.peek_is("}")? {
        let step = cursor.next_required()?;
        let choice = parse_backend_choice(&cursor.next_required()?, cursor.path())?;
        cursor.expect_optional(";")?;
        entries.push(BackendSelection { step, choice });
    }
    cursor.expect("}")?;

    Ok(BackendSection { name, entries })
}

fn parse_cpu_block(cursor: &mut TokenCursor) -> Result<CpuConfig> {
    let mut cpu = CpuConfig::default();

    while !cursor.peek_is("}")? {
        let key = cursor.next_required()?;
        let values = cursor.read_value_until_semicolon()?;
        match key.as_str() {
            "threads" => {
                let value = single_value(&values, "CPU threads", cursor.path())?;
                validate_auto_or_positive_integer(&value, "CPU threads", cursor.path())?;
                cpu.threads = value;
            }
            "threadPinning" => {
                let value = single_value(&values, "CPU threadPinning", cursor.path())?;
                validate_auto_on_off(&value, "CPU threadPinning", cursor.path())?;
                cpu.thread_pinning = value;
            }
            "numa" => {
                let value = single_value(&values, "CPU numa", cursor.path())?;
                validate_auto_on_off(&value, "CPU numa", cursor.path())?;
                cpu.numa = value;
            }
            _ => {}
        }
    }
    cursor.expect("}")?;

    Ok(cpu)
}

fn parse_gpu_block(cursor: &mut TokenCursor) -> Result<GpuConfig> {
    let mut gpu = GpuConfig::default();

    while !cursor.peek_is("}")? {
        let key = cursor.next_required()?;
        let values = cursor.read_value_until_semicolon()?;
        match key.as_str() {
            "backend" => {
                let value = single_value(&values, "GPU backend", cursor.path())?;
                validate_gpu_backend(&value, cursor.path())?;
                gpu.backend = value;
            }
            "device" => {
                let value = single_value(&values, "GPU device", cursor.path())?;
                validate_word_or_number(&value, "GPU device", cursor.path())?;
                gpu.devices = vec![value];
            }
            "devices" => {
                let devices = value_list(&values, "GPU devices", cursor.path())?;
                for device in &devices {
                    validate_word_or_number(device, "GPU device", cursor.path())?;
                }
                gpu.devices = devices;
            }
            "multiGpu" => {
                let value = single_value(&values, "GPU multiGpu", cursor.path())?;
                validate_auto_on_off(&value, "GPU multiGpu", cursor.path())?;
                gpu.multi_gpu = value;
            }
            "precision" => {
                let value = single_value(&values, "GPU precision", cursor.path())?;
                validate_precision(&value, cursor.path())?;
                gpu.precision = value;
            }
            _ => {}
        }
    }
    cursor.expect("}")?;

    Ok(gpu)
}

fn parse_backend_choice(value: &str, path: &Path) -> Result<BackendChoice> {
    match value {
        "cpu" => Ok(BackendChoice::Cpu),
        "gpu" => Ok(BackendChoice::Gpu),
        "auto" => Ok(BackendChoice::Auto),
        _ => Err(MeshError::InvalidInput(format!(
            "invalid backend choice '{}' in {}; expected cpu, gpu, or auto",
            value,
            path.display()
        ))),
    }
}

fn single_value(values: &[String], label: &str, path: &Path) -> Result<String> {
    if values.len() == 1 {
        return Ok(values[0].clone());
    }

    Err(MeshError::InvalidInput(format!(
        "{label} in {} must be a single value",
        path.display()
    )))
}

fn value_list(values: &[String], label: &str, path: &Path) -> Result<Vec<String>> {
    if values.is_empty() {
        return Err(MeshError::InvalidInput(format!(
            "{label} in {} must not be empty",
            path.display()
        )));
    }

    if values.first().map(String::as_str) == Some("(")
        && values.last().map(String::as_str) == Some(")")
    {
        let values = values[1..values.len() - 1].to_vec();
        if values.is_empty() {
            return Err(MeshError::InvalidInput(format!(
                "{label} in {} must not be empty",
                path.display()
            )));
        }
        return Ok(values);
    }

    if values.len() == 1 {
        return Ok(vec![values[0].clone()]);
    }

    Err(MeshError::InvalidInput(format!(
        "{label} in {} must be a single value or parenthesized list",
        path.display()
    )))
}

fn validate_auto_or_positive_integer(value: &str, label: &str, path: &Path) -> Result<()> {
    if value == "auto" {
        return Ok(());
    }

    let parsed = value.parse::<usize>().map_err(|_| {
        MeshError::InvalidInput(format!(
            "invalid {label} '{}' in {}; expected auto or a positive integer",
            value,
            path.display()
        ))
    })?;
    if parsed > 0 {
        return Ok(());
    }

    Err(MeshError::InvalidInput(format!(
        "invalid {label} '{}' in {}; expected auto or a positive integer",
        value,
        path.display()
    )))
}

fn validate_auto_on_off(value: &str, label: &str, path: &Path) -> Result<()> {
    match value {
        "auto" | "on" | "off" => Ok(()),
        _ => Err(MeshError::InvalidInput(format!(
            "invalid {label} '{}' in {}; expected auto, on, or off",
            value,
            path.display()
        ))),
    }
}

fn validate_gpu_backend(value: &str, path: &Path) -> Result<()> {
    match value {
        "auto" | "wgpu" | "cuda" | "hip" => Ok(()),
        _ => Err(MeshError::InvalidInput(format!(
            "invalid GPU backend '{}' in {}; expected auto, wgpu, cuda, or hip",
            value,
            path.display()
        ))),
    }
}

fn validate_precision(value: &str, path: &Path) -> Result<()> {
    match value {
        "auto" | "f32" | "f64" => Ok(()),
        _ => Err(MeshError::InvalidInput(format!(
            "invalid GPU precision '{}' in {}; expected auto, f32, or f64",
            value,
            path.display()
        ))),
    }
}

fn validate_word_or_number(value: &str, label: &str, path: &Path) -> Result<()> {
    if value
        .chars()
        .all(|ch| ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' || ch == ':' || ch == '.')
    {
        return Ok(());
    }

    Err(MeshError::InvalidInput(format!(
        "invalid {label} '{}' in {}",
        value,
        path.display()
    )))
}

struct BackendConfigBuilder {
    path: PathBuf,
    default: Option<BackendChoice>,
    sections: Vec<BackendSection>,
    cpu: CpuConfig,
    gpu: GpuConfig,
}

impl BackendConfigBuilder {
    fn new(path: &Path) -> Self {
        Self {
            path: path.to_path_buf(),
            default: None,
            sections: Vec::new(),
            cpu: CpuConfig::default(),
            gpu: GpuConfig::default(),
        }
    }

    fn finish(self) -> Result<BackendConfig> {
        Ok(BackendConfig {
            path: self.path,
            default: self.default.unwrap_or(BackendChoice::Cpu),
            sections: self.sections,
            cpu: self.cpu,
            gpu: self.gpu,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::{BackendChoice, parse_backend_config_str};

    #[test]
    fn parses_template_backend_config() {
        let content = r#"
        FoamFile
        {
            version 2.0;
            class dictionary;
            object ferrumBackends;
        }

        default cpu;

        mesh
        {
            import cpu;
            checks cpu;
        }

        cpu
        {
            threads auto;
            threadPinning off;
            numa auto;
        }

        flow
        {
            nonlinearSolve gpu;
            residual auto;
            jacobian auto;
            linearSolve gpu;
        }

        gpu
        {
            backend wgpu;
            devices (0 1);
            multiGpu auto;
            precision f64;
        }
        "#;

        let config = parse_backend_config_str(content, Path::new("ferrumBackends")).unwrap();
        assert_eq!(config.default, BackendChoice::Cpu);
        assert_eq!(config.sections.len(), 2);
        assert_eq!(config.cpu.threads, "auto");
        assert_eq!(config.sections[1].entries[0].step, "nonlinearSolve");
        assert_eq!(config.sections[1].entries[0].choice, BackendChoice::Gpu);
        assert_eq!(config.gpu.backend, "wgpu");
        assert_eq!(config.gpu.devices, vec!["0".to_string(), "1".to_string()]);
        assert_eq!(config.gpu.multi_gpu, "auto");
        assert_eq!(config.gpu.precision, "f64");
    }

    #[test]
    fn parses_outer_ferrum_backends_block() {
        let content = r#"
        ferrumBackends
        {
            default auto;
            flow
            {
                residual gpu;
            }
        }
        "#;

        let config = parse_backend_config_str(content, Path::new("ferrumBackends")).unwrap();
        assert_eq!(config.default, BackendChoice::Auto);
        assert_eq!(config.sections[0].name, "flow");
        assert_eq!(config.sections[0].entries[0].choice, BackendChoice::Gpu);
    }

    #[test]
    fn rejects_unknown_backend_choice() {
        let content = r#"
        flow
        {
            residual quantum;
        }
        "#;

        let error = parse_backend_config_str(content, Path::new("ferrumBackends")).unwrap_err();
        assert!(error.to_string().contains("expected cpu, gpu, or auto"));
    }

    #[test]
    fn rejects_zero_cpu_threads() {
        let content = r#"
        cpu
        {
            threads 0;
        }
        "#;

        let error = parse_backend_config_str(content, Path::new("ferrumBackends")).unwrap_err();
        assert!(error.to_string().contains("positive integer"));
    }
}
