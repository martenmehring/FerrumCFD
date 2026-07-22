use std::fmt::Write as _;
use std::mem::size_of;

use crate::backends::BackendChoice;
use crate::linear::linear_solver_capabilities;
use crate::solver_plan::{SolverCasePlan, SolverRunPlan, SolverRunStageSource};
use crate::solver_state::{
    SolverStateCpuBufferPlan, SolverStateFieldPlan, SolverStateInternalFieldPlan, SolverStatePlan,
    SolverStateStoragePlan,
};
use crate::{MeshError, Result};

pub const MAX_RUNNER_DRY_RUN_STEPS: usize = 1_000;
pub const MAX_RUNNER_DRY_RUN_EVENTS: usize = 100_000;
pub const MAX_RUNNER_DRY_RUN_STRING_BYTES: usize = 1024 * 1024;
pub const MAX_RUNNER_DRY_RUN_PAYLOAD_BYTES: usize = 16 * 1024 * 1024;

#[derive(Clone, Copy, Debug)]
pub struct SolverRunnerDryRunOptions {
    pub max_steps: usize,
}

#[derive(Debug)]
pub struct SolverRunnerDryRun {
    pub planned_steps: Option<usize>,
    pub preview_steps: usize,
    pub max_steps: usize,
    pub stage_count: usize,
    pub preview_write_events: usize,
    pub truncated: bool,
    pub state: SolverStatePlan,
    pub runtime: SolverRuntimePlan,
    pub events: Vec<SolverRunnerDryRunEvent>,
    pub warnings: Vec<String>,
}

#[derive(Debug)]
pub enum SolverRunnerDryRunEvent {
    StepStart {
        step: usize,
        time: Option<f64>,
    },
    Stage {
        step: usize,
        section: String,
        stage: String,
        choice: BackendChoice,
        source: SolverRunStageSource,
        dispatch: SolverRuntimeDispatch,
    },
    Write {
        step: usize,
        time: Option<f64>,
    },
}

#[derive(Clone, Debug)]
pub struct SolverRuntimePlan {
    pub cpu: SolverCpuRuntimeHandle,
    pub gpu: SolverGpuRuntimeHandle,
    pub warnings: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct SolverCpuRuntimeHandle {
    pub requested: bool,
    pub handle: String,
    pub cpus: String,
    pub cores_per_cpu: String,
    pub threads: String,
    pub thread_pinning: String,
    pub numa: String,
    pub linear_solvers_available: bool,
    pub kernels_available: bool,
}

#[derive(Clone, Debug)]
pub struct SolverGpuRuntimeHandle {
    pub requested: bool,
    pub handle: String,
    pub backend: String,
    pub devices: Vec<String>,
    pub multi_gpu: String,
    pub precision: String,
    pub linear_solvers_available: bool,
    pub kernels_available: bool,
}

#[derive(Clone, Debug)]
pub struct SolverRuntimeDispatch {
    pub target: SolverRuntimeTarget,
    pub handle: String,
    pub executable: bool,
    pub status: SolverRuntimeDispatchStatus,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SolverRuntimeTarget {
    Cpu,
    Gpu,
    Auto,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SolverRuntimeDispatchStatus {
    PlannedOnly,
    GpuRuntimeUnavailable,
    AutoPolicyUnresolved,
}

impl Default for SolverRunnerDryRunOptions {
    fn default() -> Self {
        Self { max_steps: 3 }
    }
}

impl std::fmt::Display for SolverRuntimeTarget {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cpu => formatter.write_str("cpu"),
            Self::Gpu => formatter.write_str("gpu"),
            Self::Auto => formatter.write_str("auto"),
        }
    }
}

impl std::fmt::Display for SolverRuntimeDispatchStatus {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PlannedOnly => formatter.write_str("planned-only"),
            Self::GpuRuntimeUnavailable => formatter.write_str("gpu-runtime-unavailable"),
            Self::AutoPolicyUnresolved => formatter.write_str("auto-policy-unresolved"),
        }
    }
}

fn try_copy_runner_text(value: &str) -> Result<String> {
    let mut owned = String::new();
    owned
        .try_reserve_exact(value.len())
        .map_err(|_| MeshError::OutOfMemory)?;
    owned.push_str(value);
    Ok(owned)
}

fn runner_invalid_input(message: &str) -> Result<MeshError> {
    Ok(MeshError::InvalidInput(try_copy_runner_text(message)?))
}

fn decimal_len(value: usize) -> usize {
    if value == 0 {
        1
    } else {
        value.ilog10() as usize + 1
    }
}

fn runner_max_steps_error() -> Result<MeshError> {
    const PREFIX: &str = "runner dry-run max steps must be between 1 and ";
    let len = PREFIX
        .len()
        .checked_add(decimal_len(MAX_RUNNER_DRY_RUN_STEPS))
        .ok_or(MeshError::OutOfMemory)?;
    let mut message = String::new();
    message
        .try_reserve_exact(len)
        .map_err(|_| MeshError::OutOfMemory)?;
    message.push_str(PREFIX);
    write!(&mut message, "{MAX_RUNNER_DRY_RUN_STEPS}").map_err(|_| MeshError::OutOfMemory)?;
    Ok(MeshError::InvalidInput(message))
}

fn runner_event_cap_error(event_count: usize) -> Result<MeshError> {
    const PREFIX: &str = "runner dry-run would create ";
    const MIDDLE: &str = " events, exceeding the safety cap of ";
    let len = PREFIX
        .len()
        .checked_add(decimal_len(event_count))
        .and_then(|len| len.checked_add(MIDDLE.len()))
        .and_then(|len| len.checked_add(decimal_len(MAX_RUNNER_DRY_RUN_EVENTS)))
        .ok_or(MeshError::OutOfMemory)?;
    let mut message = String::new();
    message
        .try_reserve_exact(len)
        .map_err(|_| MeshError::OutOfMemory)?;
    message.push_str(PREFIX);
    write!(&mut message, "{event_count}").map_err(|_| MeshError::OutOfMemory)?;
    message.push_str(MIDDLE);
    write!(&mut message, "{MAX_RUNNER_DRY_RUN_EVENTS}").map_err(|_| MeshError::OutOfMemory)?;
    Ok(MeshError::InvalidInput(message))
}

pub fn build_solver_runner_dry_run(
    plan: &SolverCasePlan,
    options: SolverRunnerDryRunOptions,
) -> Result<SolverRunnerDryRun> {
    if options.max_steps == 0 || options.max_steps > MAX_RUNNER_DRY_RUN_STEPS {
        return Err(runner_max_steps_error()?);
    }
    let max_steps = options.max_steps;
    let planned_steps = plan.run.estimated_steps;
    let preview_steps = planned_steps
        .map(|steps| steps.min(max_steps))
        .unwrap_or_default();
    let truncated = planned_steps
        .map(|steps| steps > preview_steps)
        .unwrap_or(false);
    let mut warnings = Vec::new();
    if planned_steps.is_none() {
        warnings
            .try_reserve_exact(1)
            .map_err(|_| MeshError::OutOfMemory)?;
        warnings.push(try_owned_runner_string(
            "time loop cannot be expanded because the run plan has no finite estimated step count",
        )?);
    }
    let events_without_writes = preview_steps
        .checked_mul(
            plan.run
                .stages
                .len()
                .checked_add(1)
                .ok_or(MeshError::OutOfMemory)?,
        )
        .ok_or(MeshError::OutOfMemory)?;
    let mut write_events = 0usize;
    for step in 1..=preview_steps {
        if is_write_due(&plan.run, step, step_time(&plan.run, step)) {
            write_events = write_events.checked_add(1).ok_or(MeshError::OutOfMemory)?;
        }
    }
    let event_count = events_without_writes
        .checked_add(write_events)
        .ok_or(MeshError::OutOfMemory)?;
    if event_count > MAX_RUNNER_DRY_RUN_EVENTS {
        return Err(runner_event_cap_error(event_count)?);
    }

    preflight_dry_run_payload(plan, preview_steps, event_count, warnings.len())?;

    let runtime = build_solver_runtime_plan(plan)?;
    let state = try_clone_solver_state_plan(&plan.state)?;
    let mut events = Vec::new();
    events
        .try_reserve_exact(event_count)
        .map_err(|_| MeshError::OutOfMemory)?;
    for step in 1..=preview_steps {
        let time = step_time(&plan.run, step);
        events.push(SolverRunnerDryRunEvent::StepStart { step, time });

        for stage in &plan.run.stages {
            events.push(SolverRunnerDryRunEvent::Stage {
                step,
                section: try_owned_runner_string(&stage.section)?,
                stage: try_owned_runner_string(&stage.step)?,
                choice: stage.choice,
                source: stage.source,
                dispatch: resolve_runtime_dispatch(stage.choice, &runtime)?,
            });
        }

        if is_write_due(&plan.run, step, time) {
            events.push(SolverRunnerDryRunEvent::Write { step, time });
        }
    }

    Ok(SolverRunnerDryRun {
        planned_steps,
        preview_steps,
        max_steps,
        stage_count: plan.run.stages.len(),
        preview_write_events: write_events,
        truncated,
        state,
        runtime,
        events,
        warnings,
    })
}

#[derive(Default)]
struct DryRunPayloadBudget {
    bytes: usize,
}

impl DryRunPayloadBudget {
    fn add_bytes(&mut self, additional: usize) -> Result<()> {
        self.bytes = self
            .bytes
            .checked_add(additional)
            .ok_or(MeshError::OutOfMemory)?;
        if self.bytes > MAX_RUNNER_DRY_RUN_PAYLOAD_BYTES {
            return Err(runner_invalid_input(
                "runner dry-run payload exceeds the safety cap",
            )?);
        }
        Ok(())
    }

    fn add_vec_backing<T>(&mut self, len: usize) -> Result<()> {
        self.add_bytes(
            len.checked_mul(size_of::<T>())
                .ok_or(MeshError::OutOfMemory)?,
        )
    }

    fn add_string(&mut self, value: &str) -> Result<()> {
        self.add_repeated_string(value, 1)
    }

    fn add_repeated_string(&mut self, value: &str, copies: usize) -> Result<()> {
        if copies == 0 {
            return Ok(());
        }
        if value.len() > MAX_RUNNER_DRY_RUN_STRING_BYTES {
            return Err(runner_invalid_input(
                "runner dry-run string exceeds the per-string safety cap",
            )?);
        }
        self.add_bytes(
            value
                .len()
                .checked_mul(copies)
                .ok_or(MeshError::OutOfMemory)?,
        )
    }

    fn add_generated_string(&mut self, len: usize, copies: usize) -> Result<()> {
        if copies == 0 {
            return Ok(());
        }
        if len > MAX_RUNNER_DRY_RUN_STRING_BYTES {
            return Err(runner_invalid_input(
                "runner dry-run generated string exceeds the per-string safety cap",
            )?);
        }
        self.add_bytes(len.checked_mul(copies).ok_or(MeshError::OutOfMemory)?)
    }
}

fn preflight_dry_run_payload(
    plan: &SolverCasePlan,
    preview_steps: usize,
    event_count: usize,
    warning_count: usize,
) -> Result<usize> {
    let mut budget = DryRunPayloadBudget::default();
    budget.add_vec_backing::<SolverRunnerDryRunEvent>(event_count)?;
    budget.add_vec_backing::<String>(warning_count)?;
    if warning_count != 0 {
        budget.add_string(
            "time loop cannot be expanded because the run plan has no finite estimated step count",
        )?;
    }

    let cpu_handle_len = cpu_handle_len(plan)?;
    budget.add_generated_string(cpu_handle_len, 1)?;
    budget.add_string(&plan.backends.cpu.cpus)?;
    budget.add_string(&plan.backends.cpu.cores_per_cpu)?;
    budget.add_string(&plan.backends.cpu.threads)?;
    budget.add_string(&plan.backends.cpu.thread_pinning)?;
    budget.add_string(&plan.backends.cpu.numa)?;

    let gpu_handle_len = gpu_handle_len(plan)?;
    budget.add_generated_string(gpu_handle_len, 1)?;
    budget.add_string(&plan.backends.gpu.backend)?;
    budget.add_vec_backing::<String>(plan.backends.gpu.devices.len())?;
    for device in &plan.backends.gpu.devices {
        budget.add_string(device)?;
    }
    budget.add_string(&plan.backends.gpu.multi_gpu)?;
    budget.add_string(&plan.backends.gpu.precision)?;
    if plan.backends.uses_gpu {
        budget.add_vec_backing::<String>(1)?;
        budget.add_string(
            "GPU execution is selected or possible, but executable GPU solver kernels are not implemented yet",
        )?;
    }

    preflight_solver_state_payload(&plan.state, &mut budget)?;

    for stage in &plan.run.stages {
        budget.add_repeated_string(&stage.section, preview_steps)?;
        budget.add_repeated_string(&stage.step, preview_steps)?;
        let dispatch_len = match stage.choice {
            BackendChoice::Cpu => cpu_handle_len,
            BackendChoice::Gpu => gpu_handle_len,
            BackendChoice::Auto => "auto-policy".len(),
        };
        budget.add_generated_string(dispatch_len, preview_steps)?;
    }

    Ok(budget.bytes)
}

fn preflight_solver_state_payload(
    state: &SolverStatePlan,
    budget: &mut DryRunPayloadBudget,
) -> Result<()> {
    budget.add_vec_backing::<SolverStateFieldPlan>(state.fields.len())?;
    for field in &state.fields {
        if let Some(region) = field.region.as_deref() {
            budget.add_string(region)?;
        }
        budget.add_string(&field.name)?;
        if let Some(class_name) = field.class_name.as_deref() {
            budget.add_string(class_name)?;
        }
        if let Some(dimensions) = field.dimensions.as_deref() {
            budget.add_vec_backing::<String>(dimensions.len())?;
            for dimension in dimensions {
                budget.add_string(dimension)?;
            }
        }
        if let Some(components) = field.internal_field.uniform_components.as_deref() {
            budget.add_vec_backing::<f64>(components.len())?;
        }
    }
    budget.add_vec_backing::<String>(state.warnings.len())?;
    for warning in &state.warnings {
        budget.add_string(warning)?;
    }
    Ok(())
}

fn cpu_handle_len(plan: &SolverCasePlan) -> Result<usize> {
    "cpu:cpus="
        .len()
        .checked_add(plan.backends.cpu.cpus.len())
        .and_then(|len| len.checked_add(":coresPerCpu=".len()))
        .and_then(|len| len.checked_add(plan.backends.cpu.cores_per_cpu.len()))
        .and_then(|len| len.checked_add(":threads=".len()))
        .and_then(|len| len.checked_add(plan.backends.cpu.threads.len()))
        .ok_or(MeshError::OutOfMemory)
}

fn gpu_handle_len(plan: &SolverCasePlan) -> Result<usize> {
    let mut len = "gpu:"
        .len()
        .checked_add(plan.backends.gpu.backend.len())
        .and_then(|len| len.checked_add(":devices=".len()))
        .ok_or(MeshError::OutOfMemory)?;
    for (index, device) in plan.backends.gpu.devices.iter().enumerate() {
        if index != 0 {
            len = len.checked_add(1).ok_or(MeshError::OutOfMemory)?;
        }
        len = len
            .checked_add(device.len())
            .ok_or(MeshError::OutOfMemory)?;
    }
    Ok(len)
}

fn build_cpu_handle(plan: &SolverCasePlan) -> Result<String> {
    let len = cpu_handle_len(plan)?;
    if len > MAX_RUNNER_DRY_RUN_STRING_BYTES {
        return Err(runner_invalid_input(
            "runner dry-run generated string exceeds the per-string safety cap",
        )?);
    }
    let mut handle = String::new();
    handle
        .try_reserve_exact(len)
        .map_err(|_| MeshError::OutOfMemory)?;
    handle.push_str("cpu:cpus=");
    handle.push_str(&plan.backends.cpu.cpus);
    handle.push_str(":coresPerCpu=");
    handle.push_str(&plan.backends.cpu.cores_per_cpu);
    handle.push_str(":threads=");
    handle.push_str(&plan.backends.cpu.threads);
    Ok(handle)
}

fn build_gpu_handle(plan: &SolverCasePlan) -> Result<String> {
    let len = gpu_handle_len(plan)?;
    if len > MAX_RUNNER_DRY_RUN_STRING_BYTES {
        return Err(runner_invalid_input(
            "runner dry-run generated string exceeds the per-string safety cap",
        )?);
    }
    let mut handle = String::new();
    handle
        .try_reserve_exact(len)
        .map_err(|_| MeshError::OutOfMemory)?;
    handle.push_str("gpu:");
    handle.push_str(&plan.backends.gpu.backend);
    handle.push_str(":devices=");
    for (index, device) in plan.backends.gpu.devices.iter().enumerate() {
        if index != 0 {
            handle.push(',');
        }
        handle.push_str(device);
    }
    Ok(handle)
}

fn try_owned_runner_string(value: &str) -> Result<String> {
    if value.len() > MAX_RUNNER_DRY_RUN_STRING_BYTES {
        return Err(runner_invalid_input(
            "runner dry-run string exceeds the per-string safety cap",
        )?);
    }
    try_copy_runner_text(value)
}

fn try_clone_runner_strings(values: &[String]) -> Result<Vec<String>> {
    let mut cloned = Vec::new();
    cloned
        .try_reserve_exact(values.len())
        .map_err(|_| MeshError::OutOfMemory)?;
    for value in values {
        cloned.push(try_owned_runner_string(value)?);
    }
    Ok(cloned)
}

fn try_clone_optional_runner_string(value: Option<&str>) -> Result<Option<String>> {
    value.map(try_owned_runner_string).transpose()
}

fn try_clone_optional_runner_strings(values: Option<&[String]>) -> Result<Option<Vec<String>>> {
    values.map(try_clone_runner_strings).transpose()
}

fn try_clone_optional_f64_values(values: Option<&[f64]>) -> Result<Option<Vec<f64>>> {
    let Some(values) = values else {
        return Ok(None);
    };
    let mut cloned = Vec::new();
    cloned
        .try_reserve_exact(values.len())
        .map_err(|_| MeshError::OutOfMemory)?;
    cloned.extend_from_slice(values);
    Ok(Some(cloned))
}

fn try_clone_solver_state_plan(state: &SolverStatePlan) -> Result<SolverStatePlan> {
    let mut fields = Vec::new();
    fields
        .try_reserve_exact(state.fields.len())
        .map_err(|_| MeshError::OutOfMemory)?;
    for field in &state.fields {
        fields.push(SolverStateFieldPlan {
            region: try_clone_optional_runner_string(field.region.as_deref())?,
            name: try_owned_runner_string(&field.name)?,
            class_name: try_clone_optional_runner_string(field.class_name.as_deref())?,
            kind: field.kind,
            dimensions: try_clone_optional_runner_strings(field.dimensions.as_deref())?,
            mesh_cells: field.mesh_cells,
            mesh_faces: field.mesh_faces,
            internal_field: SolverStateInternalFieldPlan {
                kind: field.internal_field.kind,
                value_count: field.internal_field.value_count,
                expected_count: field.internal_field.expected_count,
                valid_count: field.internal_field.valid_count,
                uniform_components: try_clone_optional_f64_values(
                    field.internal_field.uniform_components.as_deref(),
                )?,
                loaded_scalars: field.internal_field.loaded_scalars,
            },
            boundary_patches: field.boundary_patches,
            mesh_boundary_patches: field.mesh_boundary_patches,
            storage: SolverStateStoragePlan {
                cpu_capable: field.storage.cpu_capable,
                gpu_capable: field.storage.gpu_capable,
                components: field.storage.components,
                scalar_slots: field.storage.scalar_slots,
                bytes_f64: field.storage.bytes_f64,
                status: field.storage.status,
            },
            cpu_buffer: SolverStateCpuBufferPlan {
                materializable: field.cpu_buffer.materializable,
                scalar_slots: field.cpu_buffer.scalar_slots,
                bytes_f64: field.cpu_buffer.bytes_f64,
                status: field.cpu_buffer.status,
            },
        });
    }

    Ok(SolverStatePlan {
        fields,
        warnings: try_clone_runner_strings(&state.warnings)?,
    })
}

fn build_solver_runtime_plan(plan: &SolverCasePlan) -> Result<SolverRuntimePlan> {
    let linear_solvers = linear_solver_capabilities();
    let cpu_handle = build_cpu_handle(plan)?;
    let cpu = SolverCpuRuntimeHandle {
        requested: plan.backends.uses_cpu,
        handle: cpu_handle,
        cpus: try_owned_runner_string(&plan.backends.cpu.cpus)?,
        cores_per_cpu: try_owned_runner_string(&plan.backends.cpu.cores_per_cpu)?,
        threads: try_owned_runner_string(&plan.backends.cpu.threads)?,
        thread_pinning: try_owned_runner_string(&plan.backends.cpu.thread_pinning)?,
        numa: try_owned_runner_string(&plan.backends.cpu.numa)?,
        linear_solvers_available: linear_solvers.cpu_csr
            && linear_solvers.cpu_jacobi
            && linear_solvers.cpu_gauss_seidel
            && linear_solvers.cpu_conjugate_gradient
            && linear_solvers.cpu_bicgstab,
        kernels_available: false,
    };
    let gpu_handle = build_gpu_handle(plan)?;
    let gpu = SolverGpuRuntimeHandle {
        requested: plan.backends.uses_gpu,
        handle: gpu_handle,
        backend: try_owned_runner_string(&plan.backends.gpu.backend)?,
        devices: try_clone_runner_strings(&plan.backends.gpu.devices)?,
        multi_gpu: try_owned_runner_string(&plan.backends.gpu.multi_gpu)?,
        precision: try_owned_runner_string(&plan.backends.gpu.precision)?,
        linear_solvers_available: linear_solvers.gpu_linear_solvers,
        kernels_available: false,
    };

    let mut warnings = Vec::new();
    if gpu.requested {
        warnings
            .try_reserve_exact(1)
            .map_err(|_| MeshError::OutOfMemory)?;
        warnings.push(try_owned_runner_string(
            "GPU execution is selected or possible, but executable GPU solver kernels are not implemented yet",
        )?);
    }

    Ok(SolverRuntimePlan { cpu, gpu, warnings })
}

fn resolve_runtime_dispatch(
    choice: BackendChoice,
    runtime: &SolverRuntimePlan,
) -> Result<SolverRuntimeDispatch> {
    Ok(match choice {
        BackendChoice::Cpu => SolverRuntimeDispatch {
            target: SolverRuntimeTarget::Cpu,
            handle: try_owned_runner_string(&runtime.cpu.handle)?,
            executable: runtime.cpu.kernels_available,
            status: SolverRuntimeDispatchStatus::PlannedOnly,
        },
        BackendChoice::Gpu => SolverRuntimeDispatch {
            target: SolverRuntimeTarget::Gpu,
            handle: try_owned_runner_string(&runtime.gpu.handle)?,
            executable: runtime.gpu.kernels_available,
            status: SolverRuntimeDispatchStatus::GpuRuntimeUnavailable,
        },
        BackendChoice::Auto => SolverRuntimeDispatch {
            target: SolverRuntimeTarget::Auto,
            handle: try_owned_runner_string("auto-policy")?,
            executable: false,
            status: SolverRuntimeDispatchStatus::AutoPolicyUnresolved,
        },
    })
}

fn step_time(run: &SolverRunPlan, step: usize) -> Option<f64> {
    let start_time = run.start_time?;
    let delta_t = run.delta_t?;
    if !start_time.is_finite() || !delta_t.is_finite() {
        return None;
    }
    Some(start_time + delta_t * step as f64)
}

fn is_write_due(run: &SolverRunPlan, step: usize, time: Option<f64>) -> bool {
    if run.write_control == "none" {
        return false;
    }

    let Some(write_interval) = run.write_interval else {
        return false;
    };
    if !write_interval.is_finite() || write_interval <= 0.0 {
        return false;
    }

    match run.write_control.as_str() {
        "timeStep" => {
            let rounded = write_interval.round();
            if (write_interval - rounded).abs() > f64::EPSILON {
                return false;
            }
            let every_steps = rounded as usize;
            every_steps > 0 && step.is_multiple_of(every_steps)
        }
        "runTime" | "adjustableRunTime" => {
            let Some(start_time) = run.start_time else {
                return false;
            };
            let Some(delta_t) = run.delta_t else {
                return false;
            };
            let Some(time) = time else {
                return false;
            };
            let previous_time = if step == 0 {
                start_time
            } else {
                start_time + delta_t * step.saturating_sub(1) as f64
            };
            if !start_time.is_finite() || !previous_time.is_finite() || !time.is_finite() {
                return false;
            }

            let previous_write_index = ((previous_time - start_time) / write_interval).floor();
            let current_write_index = ((time - start_time) / write_interval).floor();
            current_write_index > previous_write_index && current_write_index >= 1.0
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use crate::backends::BackendChoice;
    use crate::control::ControlDict;
    use crate::fields::InitialFieldSet;
    use crate::runtime::{SolverRuntimeData, SolverRuntimeMeshData};
    use crate::solver_plan::{
        SolverBackendPlan, SolverCasePlan, SolverCpuResourcePlan, SolverDimensionality,
        SolverFieldPlan, SolverGpuResourcePlan, SolverInterfacePlan, SolverMeshPlan,
        SolverNumericsDictionaryPlan, SolverNumericsPlan, SolverPropertiesPlan, SolverRunPlan,
        SolverRunStagePlan, SolverRunStageSource,
    };
    use crate::solver_state::SolverStatePlan;

    use super::{
        MAX_RUNNER_DRY_RUN_EVENTS, MAX_RUNNER_DRY_RUN_PAYLOAD_BYTES, MAX_RUNNER_DRY_RUN_STEPS,
        MAX_RUNNER_DRY_RUN_STRING_BYTES, SolverRunnerDryRunEvent, SolverRunnerDryRunOptions,
        build_solver_runner_dry_run, preflight_dry_run_payload,
    };
    use super::{SolverRuntimeDispatchStatus, SolverRuntimeTarget};

    #[test]
    fn expands_capped_time_step_dry_run() {
        let plan = case_plan(Some(5), 1.0, "timeStep", Some(2.0));

        let dry_run =
            build_solver_runner_dry_run(&plan, SolverRunnerDryRunOptions { max_steps: 3 }).unwrap();

        assert_eq!(dry_run.planned_steps, Some(5));
        assert_eq!(dry_run.preview_steps, 3);
        assert_eq!(dry_run.stage_count, 2);
        assert_eq!(dry_run.preview_write_events, 1);
        assert!(dry_run.runtime.cpu.linear_solvers_available);
        assert!(dry_run.runtime.gpu.requested);
        assert!(!dry_run.runtime.gpu.linear_solvers_available);
        assert!(
            dry_run
                .runtime
                .warnings
                .iter()
                .any(|warning| warning.contains("GPU execution"))
        );
        let gpu_stage = dry_run
            .events
            .iter()
            .find_map(|event| match event {
                SolverRunnerDryRunEvent::Stage {
                    stage, dispatch, ..
                } if stage == "residual" => Some(dispatch),
                _ => None,
            })
            .expect("gpu residual dispatch");
        assert_eq!(gpu_stage.target, SolverRuntimeTarget::Gpu);
        assert_eq!(
            gpu_stage.status,
            SolverRuntimeDispatchStatus::GpuRuntimeUnavailable
        );
        assert!(!gpu_stage.executable);
        assert!(dry_run.truncated);
        assert!(matches!(
            dry_run.events.first(),
            Some(SolverRunnerDryRunEvent::StepStart { step: 1, .. })
        ));
        assert!(
            dry_run
                .events
                .iter()
                .any(|event| matches!(event, SolverRunnerDryRunEvent::Write { step: 2, .. }))
        );
    }

    #[test]
    fn reports_unknown_time_loop_without_stage_expansion() {
        let plan = case_plan(None, 1.0, "timeStep", Some(1.0));

        let dry_run =
            build_solver_runner_dry_run(&plan, SolverRunnerDryRunOptions { max_steps: 3 }).unwrap();

        assert_eq!(dry_run.planned_steps, None);
        assert_eq!(dry_run.preview_steps, 0);
        assert!(dry_run.events.is_empty());
        assert!(
            dry_run
                .warnings
                .iter()
                .any(|warning| warning.contains("cannot be expanded"))
        );
    }

    #[test]
    fn runner_preview_step_cap_is_exact() {
        let accepted_plan = case_plan(Some(MAX_RUNNER_DRY_RUN_STEPS), 1.0, "timeStep", Some(1.0));
        let accepted = build_solver_runner_dry_run(
            &accepted_plan,
            SolverRunnerDryRunOptions {
                max_steps: MAX_RUNNER_DRY_RUN_STEPS,
            },
        )
        .expect("exact preview step cap should succeed");
        assert_eq!(accepted.preview_steps, MAX_RUNNER_DRY_RUN_STEPS);

        let rejected_plan = case_plan(
            Some(MAX_RUNNER_DRY_RUN_STEPS + 1),
            1.0,
            "timeStep",
            Some(1.0),
        );

        let error = build_solver_runner_dry_run(
            &rejected_plan,
            SolverRunnerDryRunOptions {
                max_steps: MAX_RUNNER_DRY_RUN_STEPS + 1,
            },
        )
        .unwrap_err();

        assert!(error.to_string().contains("max steps must be between"));
    }

    #[test]
    fn runner_direct_zero_steps_fail_closed() {
        let plan = case_plan(Some(1), 1.0, "timeStep", Some(1.0));

        let error = build_solver_runner_dry_run(&plan, SolverRunnerDryRunOptions { max_steps: 0 })
            .unwrap_err();

        assert_eq!(
            error.to_string(),
            "runner dry-run max steps must be between 1 and 1000"
        );
    }

    #[test]
    fn runner_event_cap_fails_before_growth() {
        let mut plan = case_plan(Some(MAX_RUNNER_DRY_RUN_STEPS), 1.0, "timeStep", None);
        plan.run.stages = (0..MAX_RUNNER_DRY_RUN_EVENTS / MAX_RUNNER_DRY_RUN_STEPS)
            .map(|index| SolverRunStagePlan {
                section: format!("flow{index}"),
                step: "residual".to_string(),
                choice: BackendChoice::Cpu,
                source: SolverRunStageSource::Configured,
            })
            .collect();

        let error = build_solver_runner_dry_run(
            &plan,
            SolverRunnerDryRunOptions {
                max_steps: MAX_RUNNER_DRY_RUN_STEPS,
            },
        )
        .unwrap_err();

        assert_eq!(
            error.to_string(),
            "runner dry-run would create 101000 events, exceeding the safety cap of 100000"
        );
    }

    #[test]
    fn runner_string_cap_is_exact() {
        let mut plan = case_plan(Some(1), 1.0, "timeStep", None);
        plan.run.stages.truncate(1);
        plan.run.stages[0].section = "s".repeat(MAX_RUNNER_DRY_RUN_STRING_BYTES);

        let accepted =
            build_solver_runner_dry_run(&plan, SolverRunnerDryRunOptions { max_steps: 1 })
                .expect("an output string exactly at the cap should succeed");
        let accepted_section = accepted.events.iter().find_map(|event| match event {
            SolverRunnerDryRunEvent::Stage { section, .. } => Some(section),
            _ => None,
        });
        assert_eq!(
            accepted_section.map(String::len),
            Some(MAX_RUNNER_DRY_RUN_STRING_BYTES)
        );

        plan.run.stages[0].section.push('s');
        let error = build_solver_runner_dry_run(&plan, SolverRunnerDryRunOptions { max_steps: 1 })
            .unwrap_err();
        assert!(error.to_string().contains("per-string safety cap"));
    }

    #[test]
    fn runner_aggregate_payload_cap_is_exact() {
        const STEPS: usize = 16;
        let mut plan = case_plan(Some(STEPS), 1.0, "timeStep", None);
        plan.run.stages.truncate(1);
        plan.run.stages[0].section.clear();
        plan.state.warnings.push(String::new());

        let event_count = STEPS * 2;
        let base = preflight_dry_run_payload(&plan, STEPS, event_count, 0)
            .expect("small baseline payload should fit");
        let remaining = MAX_RUNNER_DRY_RUN_PAYLOAD_BYTES - base;
        let repeated_bytes = remaining / STEPS;
        let final_bytes = remaining % STEPS;
        assert!(repeated_bytes <= MAX_RUNNER_DRY_RUN_STRING_BYTES);
        plan.run.stages[0].section = "p".repeat(repeated_bytes);
        plan.state.warnings[0] = "w".repeat(final_bytes);

        assert_eq!(
            preflight_dry_run_payload(&plan, STEPS, event_count, 0).unwrap(),
            MAX_RUNNER_DRY_RUN_PAYLOAD_BYTES
        );
        let accepted =
            build_solver_runner_dry_run(&plan, SolverRunnerDryRunOptions { max_steps: STEPS })
                .expect("aggregate payload exactly at the cap should succeed");
        assert_eq!(accepted.preview_steps, STEPS);

        plan.state.warnings[0].push('w');
        let error =
            build_solver_runner_dry_run(&plan, SolverRunnerDryRunOptions { max_steps: STEPS })
                .unwrap_err();
        assert!(error.to_string().contains("payload exceeds the safety cap"));
    }

    #[test]
    fn runner_rejects_large_repeated_plan_payload_without_partial_prefix() {
        let mut plan = case_plan(Some(17), 1.0, "timeStep", None);
        plan.run.stages.truncate(1);
        plan.run.stages[0].section = "q".repeat(MAX_RUNNER_DRY_RUN_STRING_BYTES);
        let original_section = plan.run.stages[0].section.as_bytes().to_vec();

        let error = build_solver_runner_dry_run(&plan, SolverRunnerDryRunOptions { max_steps: 17 })
            .unwrap_err();

        assert!(error.to_string().contains("payload exceeds the safety cap"));
        assert_eq!(plan.run.stages[0].section.as_bytes(), original_section);
    }

    #[test]
    fn runner_rejects_generated_backend_handle_before_event_build() {
        let mut plan = case_plan(Some(1), 1.0, "timeStep", None);
        plan.backends.gpu.devices = vec!["d".repeat(MAX_RUNNER_DRY_RUN_STRING_BYTES)];
        let original_device = plan.backends.gpu.devices[0].as_bytes().to_vec();

        let error = build_solver_runner_dry_run(&plan, SolverRunnerDryRunOptions { max_steps: 1 })
            .unwrap_err();

        assert!(
            error
                .to_string()
                .contains("generated string exceeds the per-string safety cap")
        );
        assert_eq!(plan.backends.gpu.devices[0].as_bytes(), original_device);
    }

    fn case_plan(
        estimated_steps: Option<usize>,
        delta_t: f64,
        write_control: &str,
        write_interval: Option<f64>,
    ) -> SolverCasePlan {
        SolverCasePlan {
            case_dir: PathBuf::from("case"),
            control: ControlDict {
                path: PathBuf::from("controlDict"),
                application: Some("ferrumRun".to_string()),
                solver: Some("incompressibleFluid".to_string()),
                start_from: "startTime".to_string(),
                start_time: Some(0.0),
                stop_at: "endTime".to_string(),
                end_time: Some(estimated_steps.unwrap_or(0) as f64 * delta_t),
                delta_t: Some(delta_t),
                write_control: write_control.to_string(),
                write_interval,
            },
            mesh: SolverMeshPlan {
                points: 0,
                cells: 0,
                faces: 0,
                internal_faces: 0,
                boundary_faces: 0,
                patches: 0,
                empty_patches: 0,
                wedge_patches: 0,
                symmetry_patches: 0,
                dimensionality: SolverDimensionality::ThreeD,
                region_meshes: Vec::new(),
            },
            fields: SolverFieldPlan { fields: Vec::new() },
            initial_fields: InitialFieldSet {
                case_dir: PathBuf::from("case"),
                fields: Vec::new(),
            },
            state: SolverStatePlan {
                fields: Vec::new(),
                warnings: Vec::new(),
            },
            runtime_data: SolverRuntimeData {
                mesh: SolverRuntimeMeshData {
                    points: 0,
                    cells: 0,
                    faces: 0,
                    internal_faces: 0,
                    boundary_faces: 0,
                    owner: Vec::new(),
                    neighbour: Vec::new(),
                    patches: Vec::new(),
                    face_centres: Vec::new(),
                    face_area_vectors: Vec::new(),
                    cell_centres: Vec::new(),
                    cell_volumes: Vec::new(),
                    min_face_area: 0.0,
                    max_face_area: 0.0,
                    min_cell_volume: 0.0,
                    max_cell_volume: 0.0,
                    total_cell_volume: 0.0,
                    non_positive_cell_volumes: 0,
                },
                fields: Vec::new(),
                warnings: Vec::new(),
            },
            properties: SolverPropertiesPlan {
                dictionaries: Vec::new(),
                entries: Vec::new(),
            },
            numerics: SolverNumericsPlan {
                fv_schemes: empty_numerics_dictionary(),
                fv_solution: empty_numerics_dictionary(),
            },
            interfaces: SolverInterfacePlan {
                registry_available: false,
                discovered_interfaces: 0,
                boundary_face_zones: 0,
                config_present: false,
                configured_interfaces: 0,
            },
            backends: SolverBackendPlan {
                config_present: false,
                default: BackendChoice::Cpu,
                uses_cpu: true,
                uses_gpu: true,
                mixed_execution: true,
                cpu: SolverCpuResourcePlan {
                    cpus: "auto".to_string(),
                    cores_per_cpu: "auto".to_string(),
                    threads: "auto".to_string(),
                    thread_pinning: "off".to_string(),
                    numa: "auto".to_string(),
                },
                gpu: SolverGpuResourcePlan {
                    backend: "auto".to_string(),
                    devices: vec!["auto".to_string()],
                    multi_gpu: "auto".to_string(),
                    precision: "f64".to_string(),
                },
                stages: Vec::new(),
            },
            run: SolverRunPlan {
                stop_at: "endTime".to_string(),
                start_time: Some(0.0),
                end_time: Some(estimated_steps.unwrap_or(0) as f64 * delta_t),
                delta_t: Some(delta_t),
                estimated_steps,
                write_control: write_control.to_string(),
                write_interval,
                estimated_write_events: None,
                stages: vec![
                    SolverRunStagePlan {
                        section: "flow".to_string(),
                        step: "residual".to_string(),
                        choice: BackendChoice::Gpu,
                        source: SolverRunStageSource::Configured,
                    },
                    SolverRunStagePlan {
                        section: "flow".to_string(),
                        step: "linearSolve".to_string(),
                        choice: BackendChoice::Cpu,
                        source: SolverRunStageSource::Default,
                    },
                ],
            },
            warnings: Vec::new(),
        }
    }

    fn empty_numerics_dictionary() -> SolverNumericsDictionaryPlan {
        SolverNumericsDictionaryPlan {
            present: false,
            sections: Vec::new(),
            entries: Vec::new(),
        }
    }
}
