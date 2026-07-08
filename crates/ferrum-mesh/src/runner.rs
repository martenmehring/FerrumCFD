use crate::backends::BackendChoice;
use crate::solver_plan::{SolverCasePlan, SolverRunPlan, SolverRunStageSource};

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
    },
    Write {
        step: usize,
        time: Option<f64>,
    },
}

impl Default for SolverRunnerDryRunOptions {
    fn default() -> Self {
        Self { max_steps: 3 }
    }
}

pub fn build_solver_runner_dry_run(
    plan: &SolverCasePlan,
    options: SolverRunnerDryRunOptions,
) -> SolverRunnerDryRun {
    let max_steps = options.max_steps.max(1);
    let planned_steps = plan.run.estimated_steps;
    let preview_steps = planned_steps
        .map(|steps| steps.min(max_steps))
        .unwrap_or_default();
    let truncated = planned_steps
        .map(|steps| steps > preview_steps)
        .unwrap_or(false);
    let mut warnings = Vec::new();
    if planned_steps.is_none() {
        warnings.push(
            "time loop cannot be expanded because the run plan has no finite estimated step count"
                .to_string(),
        );
    }

    let mut events = Vec::new();
    let mut write_events = 0;
    for step in 1..=preview_steps {
        let time = step_time(&plan.run, step);
        events.push(SolverRunnerDryRunEvent::StepStart { step, time });

        for stage in &plan.run.stages {
            events.push(SolverRunnerDryRunEvent::Stage {
                step,
                section: stage.section.clone(),
                stage: stage.step.clone(),
                choice: stage.choice,
                source: stage.source,
            });
        }

        if is_write_due(&plan.run, step, time) {
            write_events += 1;
            events.push(SolverRunnerDryRunEvent::Write { step, time });
        }
    }

    SolverRunnerDryRun {
        planned_steps,
        preview_steps,
        max_steps,
        stage_count: plan.run.stages.len(),
        preview_write_events: write_events,
        truncated,
        events,
        warnings,
    }
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
    use crate::solver_plan::{
        SolverBackendPlan, SolverCasePlan, SolverCpuResourcePlan, SolverDimensionality,
        SolverFieldPlan, SolverGpuResourcePlan, SolverInterfacePlan, SolverMeshPlan,
        SolverNumericsDictionaryPlan, SolverNumericsPlan, SolverPropertiesPlan, SolverRunPlan,
        SolverRunStagePlan, SolverRunStageSource,
    };

    use super::{SolverRunnerDryRunEvent, SolverRunnerDryRunOptions, build_solver_runner_dry_run};

    #[test]
    fn expands_capped_time_step_dry_run() {
        let plan = case_plan(Some(5), 1.0, "timeStep", Some(2.0));

        let dry_run =
            build_solver_runner_dry_run(&plan, SolverRunnerDryRunOptions { max_steps: 3 });

        assert_eq!(dry_run.planned_steps, Some(5));
        assert_eq!(dry_run.preview_steps, 3);
        assert_eq!(dry_run.stage_count, 2);
        assert_eq!(dry_run.preview_write_events, 1);
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
            build_solver_runner_dry_run(&plan, SolverRunnerDryRunOptions { max_steps: 3 });

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
                application: "ferrumSolver".to_string(),
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
