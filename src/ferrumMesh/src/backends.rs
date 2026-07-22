use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

const MAX_GPU_DEVICE_COUNT: usize = 32;
const MAX_GPU_DEVICE_LIST_BYTES: usize = 1024;
const MAX_BACKEND_SECTION_COUNT: usize = 256;
const MAX_BACKEND_STAGE_COUNT: usize = 4096;
const MAX_BACKEND_POLICY_PAYLOAD_BYTES: usize = 1024 * 1024;
const MAX_BACKEND_VALIDATION_WARNINGS: usize = 256;
const MAX_BACKEND_VALIDATION_WARNING_BYTES: usize = 64 * 1024;
const BACKEND_VALIDATION_TRUNCATED_WARNING: &str =
    "backend validation warning output was truncated at configured safety limits";

use crate::dictionary::{TokenCursor, TokenProvenance, tokenize};
use crate::{MeshError, Result};

#[derive(Debug)]
pub struct BackendConfig {
    pub path: PathBuf,
    pub default: BackendChoice,
    pub sections: Vec<BackendSection>,
    pub cpu: CpuConfig,
    pub gpu: GpuConfig,
    pub cpu_explicit: bool,
    pub gpu_explicit: bool,
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
    pub cpus: String,
    pub cores_per_cpu: String,
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

#[derive(Debug)]
pub struct BackendResourceValidation {
    pub uses_cpu: bool,
    pub uses_gpu: bool,
    pub mixed_execution: bool,
    pub warnings: Vec<String>,
}

#[derive(Debug)]
pub struct BackendPolicyValidation {
    pub warnings: Vec<String>,
}

impl Default for CpuConfig {
    fn default() -> Self {
        Self {
            cpus: "auto".to_string(),
            cores_per_cpu: "auto".to_string(),
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
    let Some(content) =
        crate::case_input::CaseInput::new(case_dir).optional("system/ferrumBackends")?
    else {
        return Ok(None);
    };
    let mut config = parse_backend_config_str(&content, &path)?;
    config.path = path;
    Ok(Some(config))
}

pub fn validate_backend_resources(config: &BackendConfig) -> BackendResourceValidation {
    let mut uses_cpu = choice_can_use_cpu(config.default);
    let mut uses_gpu = choice_can_use_gpu(config.default);

    for section in &config.sections {
        for entry in &section.entries {
            uses_cpu |= choice_can_use_cpu(entry.choice);
            uses_gpu |= choice_can_use_gpu(entry.choice);
        }
    }

    let mixed_execution = uses_cpu && uses_gpu;
    let mut warnings = BackendWarningCollector::production();
    if uses_cpu && !config.cpu_explicit {
        warnings.push(&[BackendWarningPart::Text(
            "CPU execution is selected or possible, but no explicit cpu resource block was provided",
        )]);
    }
    if uses_gpu && !config.gpu_explicit {
        warnings.push(&[BackendWarningPart::Text(
            "GPU execution is selected or possible, but no explicit gpu resource block was provided",
        )]);
    }
    if mixed_execution && (!config.cpu_explicit || !config.gpu_explicit) {
        warnings.push(&[BackendWarningPart::Text(
            "mixed CPU/GPU execution should specify both cpu and gpu resources explicitly",
        )]);
    }

    BackendResourceValidation {
        uses_cpu,
        uses_gpu,
        mixed_execution,
        warnings: warnings.finish(),
    }
}

pub fn validate_backend_policy(config: &BackendConfig) -> BackendPolicyValidation {
    let mut warnings = BackendWarningCollector::production();
    if warn_duplicate_sections(config, &mut warnings)
        && warn_duplicate_steps(config, &mut warnings)
        && warn_unknown_builtin_policy_entries(config, &mut warnings)
    {
        warn_inconsistent_resource_policy(config, &mut warnings);
    }

    BackendPolicyValidation {
        warnings: warnings.finish(),
    }
}

#[derive(Clone, Copy)]
struct BackendWarningLimits {
    count: usize,
    payload_bytes: usize,
}

const BACKEND_WARNING_LIMITS: BackendWarningLimits = BackendWarningLimits {
    count: MAX_BACKEND_VALIDATION_WARNINGS,
    payload_bytes: MAX_BACKEND_VALIDATION_WARNING_BYTES,
};

enum BackendWarningPart<'a> {
    Text(&'a str),
    Number(usize),
}

struct BackendWarningCollector {
    warnings: Vec<String>,
    payload_bytes: usize,
    limits: BackendWarningLimits,
    truncated: bool,
}

impl BackendWarningCollector {
    fn production() -> Self {
        Self::with_limits(BACKEND_WARNING_LIMITS)
    }

    fn with_limits(limits: BackendWarningLimits) -> Self {
        Self {
            warnings: Vec::new(),
            payload_bytes: 0,
            limits,
            truncated: false,
        }
    }

    fn push(&mut self, parts: &[BackendWarningPart<'_>]) -> bool {
        if self.truncated {
            return false;
        }

        let Some(message_bytes) = parts.iter().try_fold(0usize, |bytes, part| {
            let part_bytes = match part {
                BackendWarningPart::Text(text) => text.len(),
                BackendWarningPart::Number(value) => decimal_usize_len(*value),
            };
            bytes.checked_add(part_bytes)
        }) else {
            self.truncate();
            return false;
        };
        let Some(next_payload_bytes) = self.payload_bytes.checked_add(message_bytes) else {
            self.truncate();
            return false;
        };
        let Some(next_count) = self.warnings.len().checked_add(1) else {
            self.truncate();
            return false;
        };
        if next_count > self.limits.count || next_payload_bytes > self.limits.payload_bytes {
            self.truncate();
            return false;
        }

        let mut message = String::new();
        if message.try_reserve_exact(message_bytes).is_err() {
            self.truncate();
            return false;
        }
        for part in parts {
            match part {
                BackendWarningPart::Text(text) => message.push_str(text),
                BackendWarningPart::Number(value) => push_decimal_usize(&mut message, *value),
            }
        }
        if self.warnings.try_reserve(1).is_err() {
            self.truncate();
            return false;
        }
        self.warnings.push(message);
        self.payload_bytes = next_payload_bytes;
        true
    }

    fn truncate(&mut self) {
        if self.truncated {
            return;
        }
        self.truncated = true;

        let sentinel_bytes = BACKEND_VALIDATION_TRUNCATED_WARNING.len();
        if self.limits.count == 0 || sentinel_bytes > self.limits.payload_bytes {
            self.warnings.clear();
            self.payload_bytes = 0;
            return;
        }

        let mut sentinel = String::new();
        if sentinel.try_reserve_exact(sentinel_bytes).is_err() {
            return;
        }
        sentinel.push_str(BACKEND_VALIDATION_TRUNCATED_WARNING);

        while self.warnings.len() >= self.limits.count
            || self
                .payload_bytes
                .checked_add(sentinel_bytes)
                .is_none_or(|bytes| bytes > self.limits.payload_bytes)
        {
            let Some(removed) = self.warnings.pop() else {
                break;
            };
            let Some(payload_bytes) = self.payload_bytes.checked_sub(removed.len()) else {
                self.warnings.clear();
                self.payload_bytes = 0;
                return;
            };
            self.payload_bytes = payload_bytes;
        }
        let Some(next_payload_bytes) = self.payload_bytes.checked_add(sentinel_bytes) else {
            return;
        };
        if self.warnings.len() < self.warnings.capacity() || self.warnings.try_reserve(1).is_ok() {
            self.warnings.push(sentinel);
            self.payload_bytes = next_payload_bytes;
        }
    }

    fn finish(self) -> Vec<String> {
        self.warnings
    }
}

fn decimal_usize_len(mut value: usize) -> usize {
    let mut digits = 1;
    while value >= 10 {
        value /= 10;
        digits += 1;
    }
    digits
}

fn push_decimal_usize(target: &mut String, mut value: usize) {
    let mut digits = [0u8; 40];
    let mut index = digits.len();
    loop {
        index -= 1;
        digits[index] = b'0' + (value % 10) as u8;
        value /= 10;
        if value == 0 {
            break;
        }
    }
    for digit in &digits[index..] {
        target.push(char::from(*digit));
    }
}

fn try_copy_backend_string(value: &str) -> Result<String> {
    let mut owned = String::new();
    owned
        .try_reserve_exact(value.len())
        .map_err(|_| MeshError::OutOfMemory)?;
    owned.push_str(value);
    Ok(owned)
}

fn backend_limit_error(message: &'static str) -> Result<MeshError> {
    Ok(MeshError::InvalidInput(try_copy_backend_string(message)?))
}

#[derive(Default)]
struct BackendPolicyBudget {
    sections: usize,
    stages: usize,
    payload_bytes: usize,
}

impl BackendPolicyBudget {
    fn add_section(&mut self, name_bytes: usize) -> Result<()> {
        let sections = self.sections.checked_add(1).ok_or(MeshError::OutOfMemory)?;
        if sections > MAX_BACKEND_SECTION_COUNT {
            return Err(backend_limit_error(
                "backend policy section count exceeds maximum of 256",
            )?);
        }
        let payload_bytes = self
            .payload_bytes
            .checked_add(name_bytes)
            .ok_or(MeshError::OutOfMemory)?;
        if payload_bytes > MAX_BACKEND_POLICY_PAYLOAD_BYTES {
            return Err(backend_limit_error(
                "backend policy retained payload exceeds maximum of 1048576 bytes",
            )?);
        }
        self.sections = sections;
        self.payload_bytes = payload_bytes;
        Ok(())
    }

    fn add_stage(&mut self, step_bytes: usize) -> Result<()> {
        let stages = self.stages.checked_add(1).ok_or(MeshError::OutOfMemory)?;
        if stages > MAX_BACKEND_STAGE_COUNT {
            return Err(backend_limit_error(
                "backend policy stage count exceeds maximum of 4096",
            )?);
        }
        let payload_bytes = self
            .payload_bytes
            .checked_add(step_bytes)
            .ok_or(MeshError::OutOfMemory)?;
        if payload_bytes > MAX_BACKEND_POLICY_PAYLOAD_BYTES {
            return Err(backend_limit_error(
                "backend policy retained payload exceeds maximum of 1048576 bytes",
            )?);
        }
        self.stages = stages;
        self.payload_bytes = payload_bytes;
        Ok(())
    }
}

fn choice_can_use_cpu(choice: BackendChoice) -> bool {
    matches!(choice, BackendChoice::Cpu | BackendChoice::Auto)
}

fn warn_duplicate_sections(config: &BackendConfig, warnings: &mut BackendWarningCollector) -> bool {
    let mut counts = HashMap::<&str, usize>::new();
    if counts.try_reserve(config.sections.len()).is_err() {
        warnings.truncate();
        return false;
    }
    for section in &config.sections {
        let count = counts.entry(section.name.as_str()).or_insert(0);
        let Some(next_count) = count.checked_add(1) else {
            warnings.truncate();
            return false;
        };
        *count = next_count;
    }

    for (section, count) in counts {
        if count > 1
            && !warnings.push(&[
                BackendWarningPart::Text("backend section '"),
                BackendWarningPart::Text(section),
                BackendWarningPart::Text("' appears "),
                BackendWarningPart::Number(count),
                BackendWarningPart::Text(
                    " times; merge duplicate sections to avoid ambiguous stage policy",
                ),
            ])
        {
            return false;
        }
    }
    true
}

fn warn_duplicate_steps(config: &BackendConfig, warnings: &mut BackendWarningCollector) -> bool {
    for section in &config.sections {
        let mut seen = HashSet::<&str>::new();
        if seen.try_reserve(section.entries.len()).is_err() {
            warnings.truncate();
            return false;
        }
        for entry in &section.entries {
            if !seen.insert(entry.step.as_str())
                && !warnings.push(&[
                    BackendWarningPart::Text("backend stage '"),
                    BackendWarningPart::Text(&section.name),
                    BackendWarningPart::Text("."),
                    BackendWarningPart::Text(&entry.step),
                    BackendWarningPart::Text("' is configured more than once"),
                ])
            {
                return false;
            }
        }
    }
    true
}

fn warn_unknown_builtin_policy_entries(
    config: &BackendConfig,
    warnings: &mut BackendWarningCollector,
) -> bool {
    for section in &config.sections {
        let Some(known_steps) = known_backend_steps(&section.name) else {
            if !warnings.push(&[
                BackendWarningPart::Text("backend section '"),
                BackendWarningPart::Text(&section.name),
                BackendWarningPart::Text(
                    "' is not a known built-in section; custom sections are allowed but are not consumed by current solver preflight",
                ),
            ]) {
                return false;
            }
            continue;
        };

        for entry in &section.entries {
            if !known_steps.contains(&entry.step.as_str())
                && !warnings.push(&[
                    BackendWarningPart::Text("backend stage '"),
                    BackendWarningPart::Text(&section.name),
                    BackendWarningPart::Text("."),
                    BackendWarningPart::Text(&entry.step),
                    BackendWarningPart::Text("' is not a known built-in stage"),
                ])
            {
                return false;
            }
        }
    }
    true
}

fn warn_inconsistent_resource_policy(
    config: &BackendConfig,
    warnings: &mut BackendWarningCollector,
) -> bool {
    let numeric_devices = config
        .gpu
        .devices
        .iter()
        .filter(|device| device.as_str() != "auto")
        .count();
    if numeric_devices > 1
        && config.gpu.multi_gpu == "off"
        && !warnings.push(&[
            BackendWarningPart::Text("gpu.devices selects "),
            BackendWarningPart::Number(numeric_devices),
            BackendWarningPart::Text(" devices but gpu.multiGpu is off"),
        ])
    {
        return false;
    }
    if config.gpu.multi_gpu == "on"
        && config.gpu.devices.iter().any(|device| device == "auto")
        && !warnings.push(&[BackendWarningPart::Text(
            "gpu.multiGpu is on but gpu.devices contains auto; list explicit device ids for reproducible multi-GPU runs",
        )])
    {
        return false;
    }

    if let (Some(cpus), Some(cores_per_cpu), Some(threads)) = (
        parse_explicit_usize(&config.cpu.cpus),
        parse_explicit_usize(&config.cpu.cores_per_cpu),
        parse_explicit_usize(&config.cpu.threads),
    ) {
        let Some(declared_cores) = cpus.checked_mul(cores_per_cpu) else {
            return warnings.push(&[BackendWarningPart::Text(
                "cpu declared physical core budget cpus*coresPerCpu exceeds supported numeric range",
            )]);
        };
        if declared_cores > 0
            && threads > declared_cores
            && !warnings.push(&[
                BackendWarningPart::Text("cpu.threads="),
                BackendWarningPart::Number(threads),
                BackendWarningPart::Text(
                    " exceeds declared physical core budget cpus*coresPerCpu=",
                ),
                BackendWarningPart::Number(declared_cores),
            ])
        {
            return false;
        }
    }
    true
}

fn known_backend_steps(section: &str) -> Option<&'static [&'static str]> {
    let steps = match section {
        "mesh" => ["import", "checks"].as_slice(),
        "interfaces" => ["flux", "coupling", "sourceTerms"].as_slice(),
        "flow" => [
            "nonlinearSolve",
            "residual",
            "jacobian",
            "linearSolve",
            "pressureCorrection",
        ]
        .as_slice(),
        "chemistry" => ["nonlinearSolve", "residual", "jacobian", "odeSolve"].as_slice(),
        "heat" => ["nonlinearSolve", "residual", "jacobian", "linearSolve"].as_slice(),
        "species" => ["nonlinearSolve", "residual", "jacobian", "linearSolve"].as_slice(),
        _ => return None,
    };

    Some(steps)
}

fn parse_explicit_usize(value: &str) -> Option<usize> {
    if value == "auto" {
        return None;
    }
    value.parse::<usize>().ok()
}

fn choice_can_use_gpu(choice: BackendChoice) -> bool {
    matches!(choice, BackendChoice::Gpu | BackendChoice::Auto)
}

fn parse_backend_config_str(content: &str, path: &Path) -> Result<BackendConfig> {
    let mut cursor = tokenize(path, content)?.into_cursor();
    let mut builder = BackendConfigBuilder::new(path);

    while let Some(token) = cursor.peek()? {
        if token.value == "FoamFile" && token.provenance == TokenProvenance::Ordinary {
            cursor.next_required()?;
            cursor.skip_braced_block()?;
            continue;
        }

        if token.value == "ferrumBackends"
            && token.provenance == TokenProvenance::Ordinary
            && cursor.peek_next()?.is_some_and(|token| {
                token.value == "{" && token.provenance == TokenProvenance::Structural
            })
        {
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
    while cursor.peek()?.is_some() {
        if stop_at_close
            && cursor.peek()?.is_some_and(|token| {
                token.value == "}" && token.provenance == TokenProvenance::Structural
            })
        {
            cursor.expect("}")?;
            return Ok(());
        }
        parse_backend_entry(cursor, builder)?;
    }
    if stop_at_close {
        return Err(MeshError::InvalidInput(format!(
            "missing closing '}}' for ferrumBackends block in {}",
            cursor.path().display()
        )));
    }
    Ok(())
}

fn parse_backend_entry(cursor: &mut TokenCursor, builder: &mut BackendConfigBuilder) -> Result<()> {
    let key = cursor.next_required()?;
    if key.provenance == TokenProvenance::Structural {
        return Err(MeshError::InvalidInput(format!(
            "unexpected dictionary token in {}",
            cursor.path().display()
        )));
    }
    if key.provenance != TokenProvenance::Ordinary {
        return skip_backend_value(cursor);
    }
    match key.value.as_str() {
        "default" => {
            if cursor
                .peek()?
                .is_some_and(|choice| choice.provenance == TokenProvenance::Structural)
            {
                return Err(MeshError::InvalidInput(format!(
                    "unexpected dictionary token in {}",
                    cursor.path().display()
                )));
            }
            let ordinary_choice = cursor
                .peek()?
                .is_some_and(|choice| choice.provenance == TokenProvenance::Ordinary);
            if !ordinary_choice {
                skip_backend_value(cursor)?;
                return Ok(());
            }
            let choice = cursor.next_required()?;
            let choice = parse_backend_choice(&choice.value, cursor.path())?;
            cursor.expect(";")?;
            builder.default = Some(choice);
        }
        "gpu" => {
            cursor.expect("{")?;
            builder.gpu = parse_gpu_block(cursor)?;
            builder.gpu_explicit = true;
        }
        "cpu" => {
            cursor.expect("{")?;
            builder.cpu = parse_cpu_block(cursor)?;
            builder.cpu_explicit = true;
        }
        name if known_backend_steps(name).is_some()
            || cursor.peek()?.is_some_and(|token| {
                token.value == "{" && token.provenance == TokenProvenance::Structural
            }) =>
        {
            cursor.expect("{")?;
            builder.budget.add_section(key.value.len())?;
            let section = parse_backend_section(cursor, key.value, &mut builder.budget)?;
            builder
                .sections
                .try_reserve(1)
                .map_err(|_| MeshError::OutOfMemory)?;
            builder.sections.push(section);
        }
        _ => skip_backend_value(cursor)?,
    }
    Ok(())
}

fn parse_backend_section(
    cursor: &mut TokenCursor,
    name: String,
    budget: &mut BackendPolicyBudget,
) -> Result<BackendSection> {
    let mut entries = Vec::new();
    let known_steps = known_backend_steps(&name);

    while cursor
        .peek()?
        .is_none_or(|token| token.value != "}" || token.provenance != TokenProvenance::Structural)
    {
        let step = cursor.next_required()?;
        if step.provenance == TokenProvenance::Structural {
            return Err(MeshError::InvalidInput(format!(
                "unexpected dictionary token in {}",
                cursor.path().display()
            )));
        }
        if step.provenance != TokenProvenance::Ordinary {
            skip_backend_value(cursor)?;
            continue;
        }
        if cursor.peek()?.is_some_and(|token| {
            token.provenance == TokenProvenance::Structural
                && matches!(token.value.as_str(), "{" | "(" | "[")
        }) {
            if known_steps
                .as_ref()
                .is_some_and(|steps| steps.contains(&step.value.as_str()))
            {
                return Err(MeshError::InvalidInput(format!(
                    "unexpected dictionary token in {}",
                    cursor.path().display()
                )));
            }
            skip_backend_value(cursor)?;
            continue;
        }
        if cursor
            .peek()?
            .is_some_and(|choice| choice.provenance == TokenProvenance::Structural)
        {
            return Err(MeshError::InvalidInput(format!(
                "unexpected dictionary token in {}",
                cursor.path().display()
            )));
        }
        let ordinary_choice = cursor
            .peek()?
            .is_some_and(|choice| choice.provenance == TokenProvenance::Ordinary);
        if !ordinary_choice {
            skip_backend_value(cursor)?;
            continue;
        }
        let choice = cursor.next_required()?;
        let choice = parse_backend_choice(&choice.value, cursor.path())?;
        cursor.expect(";")?;
        budget.add_stage(step.value.len())?;
        entries.try_reserve(1).map_err(|_| MeshError::OutOfMemory)?;
        entries.push(BackendSelection {
            step: step.value,
            choice,
        });
    }
    cursor.expect("}")?;

    Ok(BackendSection { name, entries })
}

fn parse_cpu_block(cursor: &mut TokenCursor) -> Result<CpuConfig> {
    let mut cpu = CpuConfig::default();

    while cursor
        .peek()?
        .is_none_or(|token| token.value != "}" || token.provenance != TokenProvenance::Structural)
    {
        let key = cursor.next_required()?;
        if key.provenance == TokenProvenance::Structural {
            return Err(MeshError::InvalidInput(format!(
                "unexpected dictionary token in {}",
                cursor.path().display()
            )));
        }
        if key.provenance != TokenProvenance::Ordinary
            || !matches!(
                key.value.as_str(),
                "cpus" | "coresPerCpu" | "threads" | "threadPinning" | "numa"
            )
        {
            skip_backend_value(cursor)?;
            continue;
        }
        let values = cursor.read_value_until_semicolon()?;
        match key.value.as_str() {
            "cpus" => {
                let value = single_value(values, "CPU cpus", cursor.path())?;
                validate_auto_or_positive_integer(&value, "CPU cpus", cursor.path())?;
                cpu.cpus = value;
            }
            "coresPerCpu" => {
                let value = single_value(values, "CPU coresPerCpu", cursor.path())?;
                validate_auto_or_positive_integer(&value, "CPU coresPerCpu", cursor.path())?;
                cpu.cores_per_cpu = value;
            }
            "threads" => {
                let value = single_value(values, "CPU threads", cursor.path())?;
                validate_auto_or_positive_integer(&value, "CPU threads", cursor.path())?;
                cpu.threads = value;
            }
            "threadPinning" => {
                let value = single_value(values, "CPU threadPinning", cursor.path())?;
                validate_auto_on_off(&value, "CPU threadPinning", cursor.path())?;
                cpu.thread_pinning = value;
            }
            "numa" => {
                let value = single_value(values, "CPU numa", cursor.path())?;
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

    while cursor
        .peek()?
        .is_none_or(|token| token.value != "}" || token.provenance != TokenProvenance::Structural)
    {
        let key = cursor.next_required()?;
        if key.provenance == TokenProvenance::Structural {
            return Err(MeshError::InvalidInput(format!(
                "unexpected dictionary token in {}",
                cursor.path().display()
            )));
        }
        if key.provenance != TokenProvenance::Ordinary
            || !matches!(
                key.value.as_str(),
                "backend" | "device" | "devices" | "multiGpu" | "precision"
            )
        {
            skip_backend_value(cursor)?;
            continue;
        }
        let structurally_grouped = cursor.peek()?.is_some_and(|token| {
            token.value == "(" && token.provenance == TokenProvenance::Structural
        });
        let raw_quoted_open = cursor
            .peek()?
            .is_some_and(|token| token.value == "(" && token.provenance == TokenProvenance::Quoted);
        let raw_quoted_item = cursor
            .peek_next()?
            .is_some_and(|token| token.provenance == TokenProvenance::Quoted);
        let values = cursor.read_value_until_semicolon()?;
        match key.value.as_str() {
            "backend" => {
                let value = single_value(values, "GPU backend", cursor.path())?;
                validate_gpu_backend(&value, cursor.path())?;
                gpu.backend = value;
            }
            "device" => {
                let value = single_value(values, "GPU device", cursor.path())?;
                validate_gpu_device_list(std::slice::from_ref(&value), cursor.path())?;
                validate_word_or_number(&value, "GPU device", cursor.path())?;
                let mut devices = Vec::new();
                devices
                    .try_reserve_exact(1)
                    .map_err(|_| MeshError::OutOfMemory)?;
                devices.push(value);
                gpu.devices = devices;
            }
            "devices" => {
                let raw_quoted_parentheses = !structurally_grouped
                    && raw_quoted_open
                    && raw_quoted_item
                    && values.len() == 3
                    && values.first().map(String::as_str) == Some("(")
                    && values.last().map(String::as_str) == Some(")");
                let devices = value_list(
                    values,
                    structurally_grouped,
                    raw_quoted_parentheses,
                    "GPU devices",
                    cursor.path(),
                )?;
                gpu.devices = devices;
            }
            "multiGpu" => {
                let value = single_value(values, "GPU multiGpu", cursor.path())?;
                validate_auto_on_off(&value, "GPU multiGpu", cursor.path())?;
                gpu.multi_gpu = value;
            }
            "precision" => {
                let value = single_value(values, "GPU precision", cursor.path())?;
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

fn single_value(mut values: Vec<String>, label: &str, path: &Path) -> Result<String> {
    if values.len() == 1 {
        return values.pop().ok_or_else(|| {
            MeshError::InvalidInput(format!(
                "{label} in {} must be a single value",
                path.display()
            ))
        });
    }

    Err(MeshError::InvalidInput(format!(
        "{label} in {} must be a single value",
        path.display()
    )))
}

fn value_list(
    mut values: Vec<String>,
    structurally_grouped: bool,
    raw_quoted_parentheses: bool,
    label: &str,
    path: &Path,
) -> Result<Vec<String>> {
    if values.is_empty() {
        return Err(MeshError::InvalidInput(format!(
            "{label} in {} must not be empty",
            path.display()
        )));
    }

    let (start, end) = if structurally_grouped
        && values.first().map(String::as_str) == Some("(")
        && values.last().map(String::as_str) == Some(")")
    {
        let end = values.len().checked_sub(1).ok_or(MeshError::OutOfMemory)?;
        if end <= 1 {
            return Err(MeshError::InvalidInput(format!(
                "{label} in {} must not be empty",
                path.display()
            )));
        }
        (1, end)
    } else if values.first().map(String::as_str) == Some("(")
        && values.last().map(String::as_str) == Some(")")
    {
        (0, values.len())
    } else if values.len() == 1 {
        (0, 1)
    } else {
        return Err(MeshError::InvalidInput(format!(
            "{label} in {} must be a single value or parenthesized list",
            path.display()
        )));
    };

    let selected = &values[start..end];
    validate_gpu_device_list(selected, path)?;
    for (index, device) in selected.iter().enumerate() {
        if raw_quoted_parentheses && (index == 0 || index + 1 == selected.len()) {
            continue;
        }
        validate_word_or_number(device, "GPU device", path)?;
    }

    if start == 0 && end == values.len() {
        return Ok(values);
    }
    let selected_count = end.checked_sub(start).ok_or(MeshError::OutOfMemory)?;
    let mut selected_values = Vec::new();
    selected_values
        .try_reserve_exact(selected_count)
        .map_err(|_| MeshError::OutOfMemory)?;
    for value in values.drain(start..end) {
        selected_values.push(value);
    }
    Ok(selected_values)
}

fn skip_backend_value(cursor: &mut TokenCursor) -> Result<()> {
    cursor.skip_exact_value_or_block()
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

fn validate_gpu_device_list(devices: &[String], path: &Path) -> Result<()> {
    if devices.len() > MAX_GPU_DEVICE_COUNT {
        return Err(MeshError::InvalidInput(format!(
            "GPU devices in {} lists {} entries; maximum is {}",
            path.display(),
            devices.len(),
            MAX_GPU_DEVICE_COUNT
        )));
    }

    let Some(separators) = devices.len().checked_sub(1) else {
        return Err(backend_limit_error("GPU devices must not be empty")?);
    };
    let bytes = devices.iter().try_fold(separators, |bytes, device| {
        bytes
            .checked_add(device.len())
            .ok_or(MeshError::OutOfMemory)
    })?;
    if bytes > MAX_GPU_DEVICE_LIST_BYTES {
        return Err(MeshError::InvalidInput(format!(
            "GPU devices in {} uses {} bytes; maximum is {}",
            path.display(),
            bytes,
            MAX_GPU_DEVICE_LIST_BYTES
        )));
    }

    Ok(())
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
    cpu_explicit: bool,
    gpu_explicit: bool,
    budget: BackendPolicyBudget,
}

impl BackendConfigBuilder {
    fn new(path: &Path) -> Self {
        Self {
            path: path.to_path_buf(),
            default: None,
            sections: Vec::new(),
            cpu: CpuConfig::default(),
            gpu: GpuConfig::default(),
            cpu_explicit: false,
            gpu_explicit: false,
            budget: BackendPolicyBudget::default(),
        }
    }

    fn finish(self) -> Result<BackendConfig> {
        Ok(BackendConfig {
            path: self.path,
            default: self.default.unwrap_or(BackendChoice::Cpu),
            sections: self.sections,
            cpu: self.cpu,
            gpu: self.gpu,
            cpu_explicit: self.cpu_explicit,
            gpu_explicit: self.gpu_explicit,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::{
        BACKEND_VALIDATION_TRUNCATED_WARNING, BackendChoice, BackendConfig, BackendSection,
        BackendSelection, CpuConfig, GpuConfig, MAX_BACKEND_POLICY_PAYLOAD_BYTES,
        MAX_BACKEND_SECTION_COUNT, MAX_BACKEND_STAGE_COUNT, MAX_BACKEND_VALIDATION_WARNING_BYTES,
        MAX_BACKEND_VALIDATION_WARNINGS, MAX_GPU_DEVICE_COUNT, MAX_GPU_DEVICE_LIST_BYTES,
        parse_backend_config_str, validate_backend_policy,
    };

    fn policy_config(sections: Vec<BackendSection>) -> BackendConfig {
        BackendConfig {
            path: Path::new("ferrumBackends").to_path_buf(),
            default: BackendChoice::Cpu,
            sections,
            cpu: CpuConfig::default(),
            gpu: GpuConfig::default(),
            cpu_explicit: false,
            gpu_explicit: false,
        }
    }

    fn repeated_known_step_config(entry_count: usize) -> BackendConfig {
        let entries = (0..entry_count)
            .map(|_| BackendSelection {
                step: "residual".to_string(),
                choice: BackendChoice::Cpu,
            })
            .collect();
        policy_config(vec![BackendSection {
            name: "flow".to_string(),
            entries,
        }])
    }

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
            cpus auto;
            coresPerCpu auto;
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
        assert!(config.cpu_explicit);
        assert!(config.gpu_explicit);
        assert_eq!(config.cpu.cpus, "auto");
        assert_eq!(config.cpu.cores_per_cpu, "auto");
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
            cpu
            {
                cpus auto;
                coresPerCpu auto;
                threads auto;
            }
            gpu
            {
                backend auto;
                devices (auto);
            }
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
    fn rejects_unclosed_outer_ferrum_backends_block() {
        let error =
            parse_backend_config_str("ferrumBackends { default cpu;", Path::new("ferrumBackends"))
                .expect_err("missing closing brace must fail");

        assert!(error.to_string().contains("unclosed dictionary delimiter"));
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

    #[test]
    fn warns_when_mixed_policy_missing_explicit_resources() {
        let content = r#"
        default cpu;
        flow
        {
            residual gpu;
        }
        "#;

        let config = parse_backend_config_str(content, Path::new("ferrumBackends")).unwrap();
        let validation = super::validate_backend_resources(&config);
        assert!(validation.mixed_execution);
        assert_eq!(validation.warnings.len(), 3);
    }

    #[test]
    fn warns_for_duplicate_and_unknown_backend_stages() {
        let content = r#"
        flow
        {
            residual gpu;
            residual cpu;
            linearSlove gpu;
        }
        customModel
        {
            thing gpu;
        }
        "#;

        let config = parse_backend_config_str(content, Path::new("ferrumBackends")).unwrap();
        let validation = super::validate_backend_policy(&config);

        assert!(
            validation
                .warnings
                .iter()
                .any(|warning| warning.contains("flow.residual"))
        );
        assert!(
            validation
                .warnings
                .iter()
                .any(|warning| warning.contains("linearSlove"))
        );
        assert!(
            validation
                .warnings
                .iter()
                .any(|warning| warning.contains("customModel"))
        );
    }

    #[test]
    fn accepts_interface_backend_stages_as_builtin() {
        let content = r#"
        interfaces
        {
            flux gpu;
            coupling cpu;
            sourceTerms auto;
        }
        "#;

        let config = parse_backend_config_str(content, Path::new("ferrumBackends")).unwrap();
        let validation = super::validate_backend_policy(&config);

        assert!(
            validation
                .warnings
                .iter()
                .all(|warning| !warning.contains("interfaces"))
        );
    }

    #[test]
    fn warns_for_inconsistent_resource_policy() {
        let content = r#"
        cpu
        {
            cpus 2;
            coresPerCpu 4;
            threads 16;
        }
        gpu
        {
            devices (0 1);
            multiGpu off;
        }
        "#;

        let config = parse_backend_config_str(content, Path::new("ferrumBackends")).unwrap();
        let validation = super::validate_backend_policy(&config);

        assert!(
            validation
                .warnings
                .iter()
                .any(|warning| warning.contains("gpu.devices selects 2 devices"))
        );
        assert!(
            validation
                .warnings
                .iter()
                .any(|warning| warning.contains("cpu.threads=16"))
        );
    }

    #[test]
    fn backend_warning_payload_cap_is_exact_and_one_over_has_no_prefix() {
        const PREFIX: &str = "backend section '";
        const SUFFIX: &str = "' is not a known built-in section; custom sections are allowed but are not consumed by current solver preflight";
        let fixed_bytes = PREFIX.len() + SUFFIX.len();
        let exact_name = "x".repeat(MAX_BACKEND_VALIDATION_WARNING_BYTES - fixed_bytes);
        let exact = validate_backend_policy(&policy_config(vec![BackendSection {
            name: exact_name,
            entries: Vec::new(),
        }]));
        assert_eq!(exact.warnings.len(), 1);
        assert_eq!(
            exact.warnings[0].len(),
            MAX_BACKEND_VALIDATION_WARNING_BYTES
        );
        assert!(exact.warnings[0].starts_with(PREFIX));
        assert!(exact.warnings[0].ends_with(SUFFIX));

        let one_over_name = "x".repeat(MAX_BACKEND_VALIDATION_WARNING_BYTES - fixed_bytes + 1);
        let one_over = validate_backend_policy(&policy_config(vec![BackendSection {
            name: one_over_name,
            entries: Vec::new(),
        }]));
        assert_eq!(one_over.warnings.len(), 1);
        assert_eq!(one_over.warnings[0], BACKEND_VALIDATION_TRUNCATED_WARNING);
        assert!(!one_over.warnings[0].starts_with(PREFIX));
    }

    #[test]
    fn backend_warning_count_cap_is_exact_for_duplicate_steps() {
        const DUPLICATE_WARNING: &str =
            "backend stage 'flow.residual' is configured more than once";
        let exact = validate_backend_policy(&repeated_known_step_config(
            MAX_BACKEND_VALIDATION_WARNINGS + 1,
        ));
        assert_eq!(exact.warnings.len(), MAX_BACKEND_VALIDATION_WARNINGS);
        assert!(
            exact
                .warnings
                .iter()
                .all(|warning| warning == DUPLICATE_WARNING)
        );

        let one_over = validate_backend_policy(&repeated_known_step_config(
            MAX_BACKEND_VALIDATION_WARNINGS + 2,
        ));
        assert_eq!(one_over.warnings.len(), MAX_BACKEND_VALIDATION_WARNINGS);
        assert_eq!(
            one_over.warnings.last().map(String::as_str),
            Some(BACKEND_VALIDATION_TRUNCATED_WARNING)
        );
        assert!(
            one_over.warnings[..one_over.warnings.len() - 1]
                .iter()
                .all(|warning| warning == DUPLICATE_WARNING)
        );
    }

    #[test]
    fn backend_warning_count_cap_bounds_many_unknown_sections_without_prefix() {
        let sections = (0..=MAX_BACKEND_VALIDATION_WARNINGS)
            .map(|index| BackendSection {
                name: format!("unknown-{index}"),
                entries: Vec::new(),
            })
            .collect();
        let validation = validate_backend_policy(&policy_config(sections));

        assert_eq!(validation.warnings.len(), MAX_BACKEND_VALIDATION_WARNINGS);
        assert_eq!(
            validation.warnings.last().map(String::as_str),
            Some(BACKEND_VALIDATION_TRUNCATED_WARNING)
        );
        assert!(
            validation.warnings[..validation.warnings.len() - 1]
                .iter()
                .all(|warning| warning.ends_with(
                    "' is not a known built-in section; custom sections are allowed but are not consumed by current solver preflight"
                ))
        );
        assert!(
            validation
                .warnings
                .iter()
                .all(|warning| !warning.contains("unknown-256"))
        );
        assert!(
            validation.warnings.iter().map(String::len).sum::<usize>()
                <= MAX_BACKEND_VALIDATION_WARNING_BYTES
        );
    }

    #[test]
    fn backend_section_count_cap_is_exact_and_one_over_returns_no_prefix() {
        let exact_input = (0..MAX_BACKEND_SECTION_COUNT)
            .map(|index| format!("custom{index} {{}}"))
            .collect::<Vec<_>>()
            .join(" ");
        let exact = parse_backend_config_str(&exact_input, Path::new("ferrumBackends")).unwrap();
        assert_eq!(exact.sections.len(), MAX_BACKEND_SECTION_COUNT);

        let one_over_input = format!("{exact_input} custom{} {{}}", MAX_BACKEND_SECTION_COUNT);
        let error = parse_backend_config_str(&one_over_input, Path::new("ferrumBackends"))
            .expect_err("one section over the retained limit must reject the whole config");
        assert_eq!(
            error.to_string(),
            "backend policy section count exceeds maximum of 256"
        );
    }

    #[test]
    fn backend_stage_count_cap_is_exact_and_one_over_returns_no_prefix() {
        let exact_entries = "residual cpu; ".repeat(MAX_BACKEND_STAGE_COUNT);
        let exact_input = format!("flow {{ {exact_entries} }}");
        let exact = parse_backend_config_str(&exact_input, Path::new("ferrumBackends")).unwrap();
        assert_eq!(exact.sections[0].entries.len(), MAX_BACKEND_STAGE_COUNT);

        let one_over_input = format!("flow {{ {exact_entries} residual cpu; }}");
        let error = parse_backend_config_str(&one_over_input, Path::new("ferrumBackends"))
            .expect_err("one stage over the retained limit must reject the whole config");
        assert_eq!(
            error.to_string(),
            "backend policy stage count exceeds maximum of 4096"
        );
    }

    #[test]
    fn backend_policy_payload_cap_is_exact_and_one_over_returns_no_prefix() {
        let first_name = "a".repeat(MAX_BACKEND_POLICY_PAYLOAD_BYTES / 2);
        let second_name = "b".repeat(MAX_BACKEND_POLICY_PAYLOAD_BYTES - first_name.len());
        let exact_input = format!("{first_name} {{}} {second_name} {{}}");
        let exact = parse_backend_config_str(&exact_input, Path::new("ferrumBackends")).unwrap();
        assert_eq!(exact.sections.len(), 2);
        assert_eq!(
            exact
                .sections
                .iter()
                .map(|section| section.name.len())
                .sum::<usize>(),
            MAX_BACKEND_POLICY_PAYLOAD_BYTES
        );

        let one_over_input = format!("{first_name} {{}} {second_name}x {{}}");
        let error = parse_backend_config_str(&one_over_input, Path::new("ferrumBackends"))
            .expect_err("one retained byte over the limit must reject the whole config");
        assert_eq!(
            error.to_string(),
            "backend policy retained payload exceeds maximum of 1048576 bytes"
        );
    }

    #[test]
    fn singular_gpu_device_uses_the_same_exact_payload_cap_as_devices() {
        let exact_device = "d".repeat(MAX_GPU_DEVICE_LIST_BYTES);
        let exact = parse_backend_config_str(
            &format!("gpu {{ device {exact_device}; }}"),
            Path::new("ferrumBackends"),
        )
        .unwrap();
        assert_eq!(exact.gpu.devices, vec![exact_device]);

        let one_over_device = "d".repeat(MAX_GPU_DEVICE_LIST_BYTES + 1);
        let error = parse_backend_config_str(
            &format!("gpu {{ device ok; device {one_over_device}; }}"),
            Path::new("ferrumBackends"),
        )
        .expect_err("the singular alias must not bypass the device payload limit");
        let display = error.to_string();
        assert!(display.contains("uses 1025 bytes; maximum is 1024"));
        assert!(!display.contains(&one_over_device));

        let invalid_one_over_device = "!".repeat(MAX_GPU_DEVICE_LIST_BYTES + 1);
        let error = parse_backend_config_str(
            &format!("gpu {{ device {invalid_one_over_device}; }}"),
            Path::new("ferrumBackends"),
        )
        .expect_err("the payload cap must run before attacker-controlled word diagnostics");
        let display = error.to_string();
        assert!(display.contains("uses 1025 bytes; maximum is 1024"));
        assert!(!display.contains(&invalid_one_over_device));
    }

    #[test]
    fn plural_gpu_device_count_and_payload_caps_are_exact_before_collection() {
        let exact_devices = (0..MAX_GPU_DEVICE_COUNT)
            .map(|index| index.to_string())
            .collect::<Vec<_>>()
            .join(" ");
        let exact = parse_backend_config_str(
            &format!("gpu {{ devices ({exact_devices}); }}"),
            Path::new("ferrumBackends"),
        )
        .unwrap();
        assert_eq!(exact.gpu.devices.len(), MAX_GPU_DEVICE_COUNT);

        let one_over_devices = format!("{exact_devices} {}", MAX_GPU_DEVICE_COUNT);
        let error = parse_backend_config_str(
            &format!("gpu {{ devices ({one_over_devices}); }}"),
            Path::new("ferrumBackends"),
        )
        .expect_err("the collection must reject before growing past its count cap");
        assert!(
            error
                .to_string()
                .contains("lists 33 entries; maximum is 32")
        );

        let exact_payload = "p".repeat(MAX_GPU_DEVICE_LIST_BYTES);
        let exact = parse_backend_config_str(
            &format!("gpu {{ devices ({exact_payload}); }}"),
            Path::new("ferrumBackends"),
        )
        .unwrap();
        assert_eq!(exact.gpu.devices, vec![exact_payload]);

        let one_over_payload = "p".repeat(MAX_GPU_DEVICE_LIST_BYTES + 1);
        let error = parse_backend_config_str(
            &format!("gpu {{ devices ({one_over_payload}); }}"),
            Path::new("ferrumBackends"),
        )
        .expect_err("the collection must reject before copying an oversized payload");
        assert!(
            error
                .to_string()
                .contains("uses 1025 bytes; maximum is 1024")
        );
    }

    #[test]
    fn rejects_oversized_gpu_device_lists() {
        let too_many_devices = (0..=super::MAX_GPU_DEVICE_COUNT)
            .map(|device| device.to_string())
            .collect::<Vec<_>>()
            .join(" ");
        let error = parse_backend_config_str(
            &format!("gpu {{ devices ({too_many_devices}); }}"),
            Path::new("ferrumBackends"),
        )
        .unwrap_err();
        assert!(error.to_string().contains("maximum"));

        let oversized_device = "a".repeat(super::MAX_GPU_DEVICE_LIST_BYTES + 1);
        let error = parse_backend_config_str(
            &format!("gpu {{ devices ({oversized_device}); }}"),
            Path::new("ferrumBackends"),
        )
        .unwrap_err();
        assert!(error.to_string().contains("maximum"));
    }

    #[test]
    fn quoted_device_parentheses_remain_device_values() {
        for (device, expected) in [
            ("0", ["(", "0", ")"]),
            ("1", ["(", "1", ")"]),
            ("auto", ["(", "auto", ")"]),
        ] {
            let config = parse_backend_config_str(
                &format!(r#"gpu {{ devices "(" "{device}" ")"; }}"#),
                Path::new("ferrumBackends"),
            )
            .unwrap();
            assert_eq!(config.gpu.devices, expected);
        }
    }

    #[test]
    fn unknown_and_quoted_entries_preserve_backend_sentinels() {
        let config = parse_backend_config_str(
            r#"
            cpu { "threads" 99; threads 7; mystery { swallowed gpu; } numa off; }
            gpu { "precision" f32; precision f64; mystery (0 1); multiGpu on; }
            flow {
                mystery { swallowed gpu; }
                "quotedStep" { swallowed gpu; };
                residual "gpu"; jacobian gpu;
                "jacobian" gpu;
            }
            heat { "quotedStep" { swallowed gpu; } residual cpu; }
            "default" gpu; default cpu;
            "#,
            Path::new("ferrumBackends"),
        )
        .unwrap();
        assert_eq!(config.cpu.threads, "7");
        assert_eq!(config.cpu.numa, "off");
        assert_eq!(config.gpu.precision, "f64");
        assert_eq!(config.gpu.multi_gpu, "on");
        assert_eq!(config.sections[0].entries.len(), 1);
        assert_eq!(config.sections[0].entries[0].step, "jacobian");
        assert_eq!(config.sections[0].entries[0].choice, BackendChoice::Gpu);
        assert_eq!(config.sections[1].entries.len(), 1);
        assert_eq!(config.sections[1].entries[0].step, "residual");
        assert_eq!(config.sections[1].entries[0].choice, BackendChoice::Cpu);
        assert_eq!(config.default, BackendChoice::Cpu);
    }

    #[test]
    fn rejects_non_value_delimiters_and_invalid_quoted_device_parentheses() {
        for content in ["default { choice gpu; }", "flow { residual }"] {
            let error = parse_backend_config_str(content, Path::new("ferrumBackends")).unwrap_err();
            assert!(error.to_string().contains("unexpected dictionary token"));
        }

        for content in [
            r#"gpu { devices "(" auto ")"; }"#,
            r#"gpu { devices "(" ")"; }"#,
            r#"gpu { devices "(" "0" "1" ")"; }"#,
            r#"gpu { devices "(" ( 0 ) ")"; }"#,
            r#"gpu { devices "(" "invalid device" ")"; }"#,
            r#"gpu { devices "(" "0"; }"#,
        ] {
            let error = parse_backend_config_str(content, Path::new("ferrumBackends")).unwrap_err();
            assert!(error.to_string().contains("GPU device"));
        }
    }

    #[test]
    fn rejects_braced_value_for_known_section_step() {
        let error = parse_backend_config_str(
            "flow { residual { gpu; } pressureCorrection cpu; }",
            Path::new("ferrumBackends"),
        )
        .unwrap_err();
        assert!(error.to_string().contains("unexpected dictionary token"));
    }

    #[test]
    fn rejects_typed_list_for_known_section_step_without_consuming_sentinel() {
        let error = parse_backend_config_str(
            "flow { residual (gpu); pressureCorrection cpu; }",
            Path::new("ferrumBackends"),
        )
        .unwrap_err();
        assert_eq!(
            error.to_string(),
            "unexpected dictionary token in ferrumBackends"
        );
    }

    #[test]
    fn rejects_bare_multi_device_value_and_structural_key() {
        let bare = parse_backend_config_str("gpu { devices 0 1; }", Path::new("ferrumBackends"))
            .unwrap_err();
        assert!(
            bare.to_string()
                .contains("single value or parenthesized list")
        );

        let structural =
            parse_backend_config_str("; cpu;", Path::new("ferrumBackends")).unwrap_err();
        assert!(
            structural
                .to_string()
                .contains("unexpected dictionary token")
        );
    }

    #[test]
    fn section_braced_entries_preserve_residual_or_fail_closed() {
        for key in ["mystery", r#""mystery""#] {
            for terminator in ["", ";"] {
                let content =
                    format!("flow {{ {key} {{ swallowed gpu; }}{terminator} residual cpu; }}");
                let config =
                    parse_backend_config_str(&content, Path::new("ferrumBackends")).unwrap();
                assert_eq!(config.sections.len(), 1);
                assert_eq!(config.sections[0].entries.len(), 1);
                assert_eq!(config.sections[0].entries[0].step, "residual");
                assert_eq!(config.sections[0].entries[0].choice, BackendChoice::Cpu);
            }
        }

        let error = parse_backend_config_str(
            "flow { residual { swallowed gpu; } pressureCorrection cpu; }",
            Path::new("ferrumBackends"),
        )
        .unwrap_err();
        assert!(error.to_string().contains("unexpected dictionary token"));
    }

    #[test]
    fn unknown_resource_blocks_preserve_ordinary_sentinels() {
        let config = parse_backend_config_str(
            "cpu { mystery { threads 99; }; threads 3; } gpu { mystery { precision f32; } backend cuda; }",
            Path::new("ferrumBackends"),
        )
        .unwrap();
        assert_eq!(config.cpu.threads, "3");
        assert_eq!(config.gpu.backend, "cuda");
        assert_eq!(config.gpu.precision, "f64");
    }

    #[test]
    fn structural_keys_fail_closed_in_every_backend_context() {
        for content in [
            "; default cpu;",
            "cpu { ; threads 3; }",
            "gpu { ; precision f64; }",
            "flow { ; residual cpu; }",
        ] {
            let error = parse_backend_config_str(content, Path::new("ferrumBackends")).unwrap_err();
            assert!(error.to_string().contains("unexpected dictionary token"));
        }
    }

    #[test]
    fn exact_unknown_backend_values_preserve_reserved_sentinels() {
        for key in ["mystery", r#""mystery""#] {
            for (value, ordinary_custom_step) in [
                ("gpu;", true),
                (r#""ignored";"#, false),
                ("(ignored nested);", false),
                ("[ignored nested];", false),
                ("{ default gpu; }", false),
                ("{ default gpu; };", false),
            ] {
                // An ordinary top-level name followed by a brace is an intentional custom
                // backend section, not an unknown value. Such sections do not use the skip
                // path, so the optional block terminator is covered by the quoted-key and
                // nested-resource cases below.
                if !(key == "mystery" && value == "{ default gpu; };") {
                    let top = parse_backend_config_str(
                        &format!("{key} {value} default cpu;"),
                        Path::new("ferrumBackends"),
                    )
                    .unwrap();
                    assert_eq!(top.default, BackendChoice::Cpu);
                }

                let cpu = parse_backend_config_str(
                    &format!("cpu {{ {key} {value} threads 3; }}"),
                    Path::new("ferrumBackends"),
                )
                .unwrap();
                assert_eq!(cpu.cpu.threads, "3");

                let gpu = parse_backend_config_str(
                    &format!("gpu {{ {key} {value} precision f32; }}"),
                    Path::new("ferrumBackends"),
                )
                .unwrap();
                assert_eq!(gpu.gpu.precision, "f32");

                let section = parse_backend_config_str(
                    &format!("flow {{ {key} {value} residual cpu; }}"),
                    Path::new("ferrumBackends"),
                )
                .unwrap();
                let entries = &section.sections[0].entries;
                let sentinel = entries.last().unwrap();
                assert_eq!(sentinel.step, "residual");
                assert_eq!(sentinel.choice, BackendChoice::Cpu);
                let expected = usize::from(key == "mystery" && ordinary_custom_step) + 1;
                assert_eq!(entries.len(), expected);
            }
        }
    }

    #[test]
    fn unterminated_backend_values_cannot_redispatch_reserved_keys() {
        for content in [
            "mystery ignored default cpu;",
            r#""mystery" "ignored" default cpu;"#,
            "mystery (ignored) default cpu;",
            "mystery [ignored] default cpu;",
            "cpu { mystery ignored threads 3; }",
            "gpu { mystery (ignored) precision f32; }",
            "flow { mystery [ignored] residual cpu; }",
        ] {
            let error = parse_backend_config_str(content, Path::new("ferrumBackends"))
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
    fn backend_choices_require_structural_semicolons() {
        for content in [
            "default cpu flow { residual gpu; }",
            "flow { residual cpu pressureCorrection gpu; }",
        ] {
            let error = parse_backend_config_str(content, Path::new("ferrumBackends"))
                .expect_err("ordinary backend choice without semicolon must fail");
            assert!(
                error.to_string().contains("unexpected dictionary token"),
                "unexpected error for {content:?}: {error}"
            );
        }

        let config = parse_backend_config_str(
            r#"
            default "gpu";
            default cpu;
            flow { residual "gpu"; residual cpu; }
            "#,
            Path::new("ferrumBackends"),
        )
        .unwrap();
        assert_eq!(config.default, BackendChoice::Cpu);
        assert_eq!(config.sections[0].entries.len(), 1);
        assert_eq!(config.sections[0].entries[0].choice, BackendChoice::Cpu);
    }
}
