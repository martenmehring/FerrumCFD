use std::collections::BTreeMap;

use crate::fields::{FieldFile, FieldValueSummary, InitialFieldSet};
use crate::linear::{
    CgPreconditioner, ConjugateGradientOptions, CsrMatrix, JacobiOptions,
    PreconditionedConjugateGradientOptions, conjugate_gradient_solve, jacobi_solve, l2_norm,
    preconditioned_conjugate_gradient_solve,
};
use crate::runtime::{SolverRuntimeData, SolverRuntimeMeshData};
use crate::{MeshError, Point3, Result};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LaminarSimpleLinearSolver {
    Cg,
    Jacobi,
    Pcg,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LaminarSimplePreconditioner {
    None,
    Diagonal,
}

#[derive(Clone, Debug)]
pub struct LaminarSimpleOptions {
    pub density: f64,
    pub dynamic_viscosity: f64,
    pub pressure_drop: f64,
    pub length: f64,
    pub diameter: f64,
    pub inlet_patch: String,
    pub outlet_patch: String,
    pub linear_solver: LaminarSimpleLinearSolver,
    pub momentum_linear_solver: LaminarSimpleLinearSolver,
    pub pressure_linear_solver: LaminarSimpleLinearSolver,
    pub momentum_preconditioner: LaminarSimplePreconditioner,
    pub pressure_preconditioner: LaminarSimplePreconditioner,
    pub linear_tolerance: f64,
    pub max_linear_iterations: usize,
    pub momentum_linear_tolerance: f64,
    pub pressure_linear_tolerance: f64,
    pub momentum_max_linear_iterations: usize,
    pub pressure_max_linear_iterations: usize,
    pub max_simple_iterations: usize,
    pub min_simple_iterations: usize,
    pub simple_tolerance: f64,
    pub pressure_drop_tolerance: f64,
    pub field_change_tolerance: f64,
    pub momentum_residual_control: Option<f64>,
    pub pressure_residual_control: Option<f64>,
    pub velocity_relaxation: f64,
    pub pressure_relaxation: f64,
}

#[derive(Clone, Debug)]
pub struct LaminarSimpleReport {
    pub cells: usize,
    pub faces: usize,
    pub internal_faces: usize,
    pub boundary_faces: usize,
    pub simple_iterations: usize,
    pub converged: bool,
    pub initial_continuity: ContinuitySummary,
    pub final_continuity: ContinuitySummary,
    pub final_momentum_residual_norm: f64,
    pub final_pressure_correction_residual_norm: f64,
    pub total_momentum_linear_iterations: usize,
    pub total_pressure_linear_iterations: usize,
    pub operator_summary: FlowOperatorSummary,
    pub boundary_summary: FlowBoundarySummary,
    pub solution: LaminarSimpleSolutionSummary,
    pub history: Vec<LaminarSimpleIterationSummary>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ContinuitySummary {
    pub l2_norm: f64,
    pub max_abs: f64,
    pub sum_abs: f64,
    pub global_sum: f64,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct FlowOperatorSummary {
    pub phi_min: f64,
    pub phi_max: f64,
    pub phi_sum_abs: f64,
    pub grad_p_l2_norm: f64,
    pub div_phi_u_l2_norm: f64,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct FlowBoundarySummary {
    pub velocity_fixed_value_faces: usize,
    pub velocity_zero_gradient_faces: usize,
    pub velocity_constraint_faces: usize,
    pub pressure_fixed_value_faces: usize,
    pub pressure_zero_gradient_faces: usize,
    pub pressure_constraint_faces: usize,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct LaminarSimpleSolutionSummary {
    pub mean_velocity: f64,
    pub analytic_mean_velocity: f64,
    pub relative_mean_velocity_error: f64,
    pub flow_rate: f64,
    pub analytic_flow_rate: f64,
    pub pressure_drop_from_mean: f64,
    pub relative_pressure_drop_error: f64,
    pub pressure_drop_from_field: Option<f64>,
    pub min_axial_velocity: f64,
    pub max_axial_velocity: f64,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct LaminarSimpleIterationSummary {
    pub iteration: usize,
    pub continuity_before: ContinuitySummary,
    pub continuity_after: ContinuitySummary,
    pub pressure_correction_accepted: bool,
    pub momentum_linear_iterations: usize,
    pub pressure_linear_iterations: usize,
    pub momentum_residual_norm: f64,
    pub pressure_correction_residual_norm: f64,
    pub relative_pressure_drop_error: f64,
    pub relative_velocity_change_l2: f64,
    pub relative_pressure_change_l2: f64,
    pub momentum_update_scale: f64,
    pub pressure_correction_update_scale: f64,
}

#[derive(Clone, Debug)]
struct ScalarComponentSystem {
    matrix: CsrMatrix,
    rhs: Vec<f64>,
}

#[derive(Clone, Copy, Debug)]
enum VectorFaceTreatment {
    FixedValue(Point3),
    ZeroGradient,
    Constraint,
}

#[derive(Clone, Copy, Debug)]
enum ScalarFaceTreatment {
    FixedValue(f64),
    ZeroGradient,
    Constraint,
}

impl std::fmt::Display for LaminarSimpleLinearSolver {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cg => formatter.write_str("cg"),
            Self::Jacobi => formatter.write_str("jacobi"),
            Self::Pcg => formatter.write_str("pcg"),
        }
    }
}

impl std::fmt::Display for LaminarSimplePreconditioner {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::None => formatter.write_str("none"),
            Self::Diagonal => formatter.write_str("diagonal"),
        }
    }
}

pub fn solve_laminar_simple(
    runtime: &SolverRuntimeData,
    fields: &InitialFieldSet,
    options: &LaminarSimpleOptions,
) -> Result<LaminarSimpleReport> {
    validate_laminar_simple_options(options)?;
    validate_runtime_mesh(&runtime.mesh)?;

    let velocity_field = find_field(fields, "U", "volVectorField")?;
    let pressure_field = find_field(fields, "p", "volScalarField")?;
    let velocity_boundary = vector_face_treatments(&runtime.mesh, velocity_field)?;
    let pressure_boundary = scalar_face_treatments(&runtime.mesh, pressure_field)?;
    let pressure_correction_boundary = pressure_correction_treatments(&pressure_boundary);
    let boundary_summary =
        summarize_boundaries(&runtime.mesh, &velocity_boundary, &pressure_boundary);

    let mut velocity = runtime_vector_field(runtime, "U")?;
    let mut pressure = runtime_scalar_field(runtime, "p")?;
    let initial_phi = compute_face_flux(&runtime.mesh, &velocity, &velocity_boundary)?;
    let initial_continuity = summarize_continuity(&net_cell_flux(&runtime.mesh, &initial_phi)?);

    let mut history = Vec::new();
    let mut converged = false;
    let mut final_continuity = initial_continuity;
    let mut final_momentum_residual_norm = 0.0;
    let mut final_pressure_correction_residual_norm = 0.0;
    let mut total_momentum_linear_iterations = 0;
    let mut total_pressure_linear_iterations = 0;
    let mut surface_flux = initial_phi;
    let mut final_phi = surface_flux.clone();
    let mut final_grad_p = vec![zero(); runtime.mesh.cells];
    let mut final_convection = vec![zero(); runtime.mesh.cells];

    for iteration in 1..=options.max_simple_iterations {
        let previous_velocity = velocity.clone();
        let previous_pressure = pressure.clone();
        let phi = surface_flux.clone();
        let continuity_before = summarize_continuity(&net_cell_flux(&runtime.mesh, &phi)?);
        let grad_p = scalar_gradient(&runtime.mesh, &pressure, &pressure_boundary)?;
        let convection =
            vector_convection_divergence(&runtime.mesh, &velocity, &velocity_boundary, &phi)?;
        let momentum = solve_momentum_predictor(
            &runtime.mesh,
            &velocity,
            &velocity_boundary,
            &phi,
            &grad_p,
            options,
        )?;
        if !momentum.residual_norm.is_finite() || !points_are_finite(&momentum.velocity) {
            final_phi = phi;
            final_continuity = continuity_before;
            final_grad_p = grad_p;
            final_convection = convection;
            history.push(LaminarSimpleIterationSummary {
                iteration,
                continuity_before,
                continuity_after: final_continuity,
                pressure_correction_accepted: false,
                momentum_linear_iterations: momentum.iterations,
                pressure_linear_iterations: 0,
                momentum_residual_norm: momentum.residual_norm,
                pressure_correction_residual_norm: 0.0,
                relative_pressure_drop_error: relative_pressure_drop_error_from_velocity(
                    &runtime.mesh,
                    &velocity,
                    options,
                )?,
                relative_velocity_change_l2: 0.0,
                relative_pressure_change_l2: 0.0,
                momentum_update_scale: 0.0,
                pressure_correction_update_scale: 0.0,
            });
            break;
        }

        total_momentum_linear_iterations += momentum.iterations;
        final_momentum_residual_norm = momentum.residual_norm;
        let predicted_velocity = momentum.velocity.clone();
        let momentum_update_scale = 1.0;
        if !points_are_finite(&predicted_velocity) {
            final_phi = phi;
            final_continuity = continuity_before;
            final_grad_p = grad_p;
            final_convection = convection;
            history.push(LaminarSimpleIterationSummary {
                iteration,
                continuity_before,
                continuity_after: final_continuity,
                pressure_correction_accepted: false,
                momentum_linear_iterations: momentum.iterations,
                pressure_linear_iterations: 0,
                momentum_residual_norm: momentum.residual_norm,
                pressure_correction_residual_norm: 0.0,
                relative_pressure_drop_error: relative_pressure_drop_error_from_velocity(
                    &runtime.mesh,
                    &velocity,
                    options,
                )?,
                relative_velocity_change_l2: 0.0,
                relative_pressure_change_l2: 0.0,
                momentum_update_scale,
                pressure_correction_update_scale: 0.0,
            });
            break;
        }

        let phi_star = compute_face_flux(&runtime.mesh, &predicted_velocity, &velocity_boundary)?;
        let r_au = reciprocal_momentum_diagonal(
            &runtime.mesh,
            &momentum.diagonal,
            options.velocity_relaxation,
        )?;
        let old_pressure_flux =
            pressure_correction_flux(&runtime.mesh, &pressure, &r_au, &pressure_boundary)?;
        let phi_hby_a = subtract_face_fluxes(&phi_star, &old_pressure_flux)?;
        let net_flux_star = net_cell_flux(&runtime.mesh, &phi_hby_a)?;
        let continuity_star = summarize_continuity(&net_flux_star);
        if !is_finite_continuity(continuity_star) {
            final_phi = phi;
            final_continuity = continuity_before;
            final_grad_p = grad_p;
            final_convection = convection;
            history.push(LaminarSimpleIterationSummary {
                iteration,
                continuity_before,
                continuity_after: final_continuity,
                pressure_correction_accepted: false,
                momentum_linear_iterations: momentum.iterations,
                pressure_linear_iterations: 0,
                momentum_residual_norm: momentum.residual_norm,
                pressure_correction_residual_norm: 0.0,
                relative_pressure_drop_error: relative_pressure_drop_error_from_velocity(
                    &runtime.mesh,
                    &velocity,
                    options,
                )?,
                relative_velocity_change_l2: 0.0,
                relative_pressure_change_l2: 0.0,
                momentum_update_scale,
                pressure_correction_update_scale: 0.0,
            });
            break;
        }
        if continuity_star.l2_norm <= options.simple_tolerance {
            velocity = predicted_velocity;
            final_phi = phi_hby_a;
            surface_flux = final_phi.clone();
            final_continuity = continuity_star;
            final_grad_p = scalar_gradient(&runtime.mesh, &pressure, &pressure_boundary)?;
            final_convection = vector_convection_divergence(
                &runtime.mesh,
                &velocity,
                &velocity_boundary,
                &final_phi,
            )?;
            let relative_pressure_drop_error =
                relative_pressure_drop_error_from_velocity(&runtime.mesh, &velocity, options)?;
            let relative_velocity_change_l2 =
                relative_vector_field_change_l2(&previous_velocity, &velocity);
            let relative_pressure_change_l2 = 0.0;
            history.push(LaminarSimpleIterationSummary {
                iteration,
                continuity_before,
                continuity_after: final_continuity,
                pressure_correction_accepted: false,
                momentum_linear_iterations: momentum.iterations,
                pressure_linear_iterations: 0,
                momentum_residual_norm: momentum.residual_norm,
                pressure_correction_residual_norm: 0.0,
                relative_pressure_drop_error,
                relative_velocity_change_l2,
                relative_pressure_change_l2,
                momentum_update_scale,
                pressure_correction_update_scale: 0.0,
            });
            if laminar_simple_converged(
                iteration,
                final_continuity,
                relative_pressure_drop_error,
                relative_velocity_change_l2,
                relative_pressure_change_l2,
                momentum.residual_norm,
                0.0,
                options,
            ) {
                converged = true;
                break;
            }
            continue;
        }
        let pressure_source = pressure_correction_source(&runtime.mesh, &net_flux_star)?;
        let pressure_system = assemble_variable_scalar_component_system(
            &runtime.mesh,
            &r_au,
            &pressure_source,
            &pressure_boundary,
        )?;
        let pressure_report = match solve_scalar_system(
            &pressure_system.matrix,
            &pressure_system.rhs,
            Some(&pressure),
            options.pressure_linear_solver,
            options.pressure_preconditioner,
            options.pressure_linear_tolerance,
            options.pressure_max_linear_iterations,
        ) {
            Ok(report) => report,
            Err(error) if is_pressure_correction_breakdown(&error) => {
                velocity = predicted_velocity;
                final_pressure_correction_residual_norm = l2_norm(&pressure_system.rhs);
                final_phi = phi_hby_a;
                final_continuity = continuity_star;
                final_grad_p = scalar_gradient(&runtime.mesh, &pressure, &pressure_boundary)?;
                final_convection = vector_convection_divergence(
                    &runtime.mesh,
                    &velocity,
                    &velocity_boundary,
                    &final_phi,
                )?;
                history.push(LaminarSimpleIterationSummary {
                    iteration,
                    continuity_before,
                    continuity_after: final_continuity,
                    pressure_correction_accepted: false,
                    momentum_linear_iterations: momentum.iterations,
                    pressure_linear_iterations: 0,
                    momentum_residual_norm: momentum.residual_norm,
                    pressure_correction_residual_norm: final_pressure_correction_residual_norm,
                    relative_pressure_drop_error: relative_pressure_drop_error_from_velocity(
                        &runtime.mesh,
                        &velocity,
                        options,
                    )?,
                    relative_velocity_change_l2: relative_vector_field_change_l2(
                        &previous_velocity,
                        &velocity,
                    ),
                    relative_pressure_change_l2: 0.0,
                    momentum_update_scale,
                    pressure_correction_update_scale: 0.0,
                });
                break;
            }
            Err(error) => {
                return Err(invalid_input(format!(
                    "laminar SIMPLE pressure correction solve failed: {error}"
                )));
            }
        };
        total_pressure_linear_iterations += pressure_report.iterations;
        final_pressure_correction_residual_norm = pressure_report.residual_norm;

        let pressure_delta = pressure_report
            .solution
            .iter()
            .zip(&pressure)
            .map(|(after, before)| options.pressure_relaxation * (after - before))
            .collect::<Vec<_>>();
        let pressure_correction_gradient = scalar_gradient(
            &runtime.mesh,
            &pressure_delta,
            &pressure_correction_boundary,
        )?;
        let mut corrected_velocity = predicted_velocity.clone();
        correct_velocity(
            &mut corrected_velocity,
            &pressure_correction_gradient,
            &r_au,
            1.0,
        );
        let mut corrected_pressure = pressure.clone();
        for (value, delta) in corrected_pressure.iter_mut().zip(&pressure_delta) {
            *value += delta;
        }

        let pressure_flux = pressure_correction_flux(
            &runtime.mesh,
            &pressure_report.solution,
            &r_au,
            &pressure_boundary,
        )?;
        let corrected_phi = add_face_fluxes(&phi_hby_a, &pressure_flux)?;
        let corrected_continuity =
            summarize_continuity(&net_cell_flux(&runtime.mesh, &corrected_phi)?);
        let pressure_correction_update_scale = 1.0;
        if !is_finite_continuity(corrected_continuity)
            || !points_are_finite(&corrected_velocity)
            || !scalars_are_finite(&corrected_pressure)
            || !corrected_phi.iter().all(|value| value.is_finite())
        {
            velocity = predicted_velocity;
            final_phi = phi_hby_a;
            final_continuity = continuity_star;
            final_grad_p = scalar_gradient(&runtime.mesh, &pressure, &pressure_boundary)?;
            final_convection = vector_convection_divergence(
                &runtime.mesh,
                &velocity,
                &velocity_boundary,
                &final_phi,
            )?;
            history.push(LaminarSimpleIterationSummary {
                iteration,
                continuity_before,
                continuity_after: final_continuity,
                pressure_correction_accepted: false,
                momentum_linear_iterations: momentum.iterations,
                pressure_linear_iterations: pressure_report.iterations,
                momentum_residual_norm: momentum.residual_norm,
                pressure_correction_residual_norm: pressure_report.residual_norm,
                relative_pressure_drop_error: relative_pressure_drop_error_from_velocity(
                    &runtime.mesh,
                    &velocity,
                    options,
                )?,
                relative_velocity_change_l2: relative_vector_field_change_l2(
                    &previous_velocity,
                    &velocity,
                ),
                relative_pressure_change_l2: 0.0,
                momentum_update_scale,
                pressure_correction_update_scale: 0.0,
            });
            break;
        }

        velocity = corrected_velocity;
        pressure = corrected_pressure;
        final_phi = corrected_phi;
        surface_flux = final_phi.clone();
        final_continuity = corrected_continuity;

        final_grad_p = scalar_gradient(&runtime.mesh, &pressure, &pressure_boundary)?;
        final_convection =
            vector_convection_divergence(&runtime.mesh, &velocity, &velocity_boundary, &final_phi)?;

        let relative_pressure_drop_error =
            relative_pressure_drop_error_from_velocity(&runtime.mesh, &velocity, options)?;
        let relative_velocity_change_l2 =
            relative_vector_field_change_l2(&previous_velocity, &velocity);
        let relative_pressure_change_l2 = relative_scalar_field_change_l2_with_reference(
            &previous_pressure,
            &pressure,
            options.pressure_drop.abs(),
        );

        history.push(LaminarSimpleIterationSummary {
            iteration,
            continuity_before,
            continuity_after: final_continuity,
            pressure_correction_accepted: true,
            momentum_linear_iterations: momentum.iterations,
            pressure_linear_iterations: pressure_report.iterations,
            momentum_residual_norm: momentum.residual_norm,
            pressure_correction_residual_norm: pressure_report.residual_norm,
            relative_pressure_drop_error,
            relative_velocity_change_l2,
            relative_pressure_change_l2,
            momentum_update_scale,
            pressure_correction_update_scale,
        });

        if laminar_simple_converged(
            iteration,
            final_continuity,
            relative_pressure_drop_error,
            relative_velocity_change_l2,
            relative_pressure_change_l2,
            momentum.residual_norm,
            pressure_report.residual_norm,
            options,
        ) {
            converged = true;
            break;
        }
    }

    let operator_summary = summarize_operators(&final_phi, &final_grad_p, &final_convection);
    let solution = summarize_laminar_simple_solution(
        &runtime.mesh,
        &velocity,
        &pressure,
        &pressure_boundary,
        options,
    )?;

    Ok(LaminarSimpleReport {
        cells: runtime.mesh.cells,
        faces: runtime.mesh.faces,
        internal_faces: runtime.mesh.internal_faces,
        boundary_faces: runtime.mesh.boundary_faces,
        simple_iterations: history.len(),
        converged,
        initial_continuity,
        final_continuity,
        final_momentum_residual_norm,
        final_pressure_correction_residual_norm,
        total_momentum_linear_iterations,
        total_pressure_linear_iterations,
        operator_summary,
        boundary_summary,
        solution,
        history,
    })
}

struct MomentumPredictorReport {
    velocity: Vec<Point3>,
    diagonal: Vec<f64>,
    iterations: usize,
    residual_norm: f64,
}

struct ScalarSolveReport {
    solution: Vec<f64>,
    iterations: usize,
    residual_norm: f64,
}

fn solve_momentum_predictor(
    mesh: &SolverRuntimeMeshData,
    velocity: &[Point3],
    velocity_boundary: &[VectorFaceTreatment],
    flux: &[f64],
    grad_p: &[Point3],
    options: &LaminarSimpleOptions,
) -> Result<MomentumPredictorReport> {
    let old_components = split_components(velocity);
    let mut solved_components = [Vec::new(), Vec::new(), Vec::new()];
    let mut diagonal = Vec::new();
    let mut total_iterations = 0;
    let mut residual_squared_sum = 0.0;

    for component in 0..3 {
        let volumetric_source = grad_p
            .iter()
            .map(|value| -component_value(*value, component))
            .collect::<Vec<_>>();
        let boundary = scalar_component_boundary(velocity_boundary, component);
        let mut system = assemble_momentum_component_system(
            mesh,
            options.dynamic_viscosity,
            options.density,
            flux,
            &volumetric_source,
            &boundary,
        )?;
        let component_diagonal = relax_scalar_component_equation(
            &mut system,
            &old_components[component],
            options.velocity_relaxation,
        )?;
        if component == 0 {
            diagonal = component_diagonal;
        }
        let report = solve_scalar_system(
            &system.matrix,
            &system.rhs,
            Some(&old_components[component]),
            options.momentum_linear_solver,
            options.momentum_preconditioner,
            options.momentum_linear_tolerance,
            options.momentum_max_linear_iterations,
        )
        .map_err(|error| {
            invalid_input(format!(
                "laminar SIMPLE momentum component {} solve failed: {error}",
                component_name(component)
            ))
        })?;
        total_iterations += report.iterations;
        residual_squared_sum += report.residual_norm * report.residual_norm;
        solved_components[component] = report.solution;
    }

    let solved_velocity = combine_components(
        &solved_components[0],
        &solved_components[1],
        &solved_components[2],
    );

    Ok(MomentumPredictorReport {
        velocity: solved_velocity,
        diagonal,
        iterations: total_iterations,
        residual_norm: residual_squared_sum.sqrt(),
    })
}

fn solve_scalar_system(
    matrix: &CsrMatrix,
    rhs: &[f64],
    initial: Option<&[f64]>,
    solver: LaminarSimpleLinearSolver,
    preconditioner: LaminarSimplePreconditioner,
    tolerance: f64,
    max_iterations: usize,
) -> Result<ScalarSolveReport> {
    let report = match solver {
        LaminarSimpleLinearSolver::Cg => conjugate_gradient_solve(
            matrix,
            rhs,
            initial,
            ConjugateGradientOptions {
                max_iterations,
                tolerance,
            },
        )?,
        LaminarSimpleLinearSolver::Pcg => preconditioned_conjugate_gradient_solve(
            matrix,
            rhs,
            initial,
            PreconditionedConjugateGradientOptions {
                max_iterations,
                tolerance,
                preconditioner: map_cg_preconditioner(preconditioner),
            },
        )?,
        LaminarSimpleLinearSolver::Jacobi => jacobi_solve(
            matrix,
            rhs,
            initial,
            JacobiOptions {
                max_iterations,
                tolerance,
                omega: 1.0,
            },
        )?,
    };
    Ok(ScalarSolveReport {
        solution: report.solution,
        iterations: report.iterations,
        residual_norm: report.residual_norm,
    })
}

fn relax_scalar_component_equation(
    system: &mut ScalarComponentSystem,
    old_solution: &[f64],
    relaxation: f64,
) -> Result<Vec<f64>> {
    if old_solution.len() != system.rhs.len() {
        return Err(invalid_input(format!(
            "equation relaxation expected old solution with {} entries, got {}",
            system.rhs.len(),
            old_solution.len()
        )));
    }
    if !relaxation.is_finite() || relaxation <= 0.0 || relaxation > 1.0 {
        return Err(invalid_input(format!(
            "equation relaxation factor must be in (0, 1], got {relaxation}"
        )));
    }

    let diagonal = system.matrix.diagonal()?;
    if (relaxation - 1.0).abs() <= f64::EPSILON {
        return Ok(diagonal);
    }

    let rhs_scale = (1.0 / relaxation) - 1.0;
    for ((rhs, diagonal), old_value) in system.rhs.iter_mut().zip(&diagonal).zip(old_solution) {
        *rhs += rhs_scale * diagonal * old_value;
    }

    let mut rows = Vec::with_capacity(system.matrix.rows());
    for row in 0..system.matrix.rows() {
        let start = system.matrix.row_offsets()[row];
        let end = system.matrix.row_offsets()[row + 1];
        let mut entries = Vec::with_capacity(end - start);
        for entry in start..end {
            let column = system.matrix.col_indices()[entry];
            let mut value = system.matrix.values()[entry];
            if column == row {
                value /= relaxation;
            }
            entries.push((column, value));
        }
        rows.push(entries);
    }
    system.matrix = CsrMatrix::from_rows(rows, system.matrix.cols())?;

    Ok(diagonal)
}

fn map_cg_preconditioner(preconditioner: LaminarSimplePreconditioner) -> CgPreconditioner {
    match preconditioner {
        LaminarSimplePreconditioner::None => CgPreconditioner::None,
        LaminarSimplePreconditioner::Diagonal => CgPreconditioner::Diagonal,
    }
}

fn is_pressure_correction_breakdown(error: &MeshError) -> bool {
    matches!(
        error,
        MeshError::InvalidInput(message)
            if message.contains("conjugate-gradient denominator is zero")
    )
}

fn is_finite_continuity(summary: ContinuitySummary) -> bool {
    summary.l2_norm.is_finite()
        && summary.max_abs.is_finite()
        && summary.sum_abs.is_finite()
        && summary.global_sum.is_finite()
}

fn laminar_simple_converged(
    iteration: usize,
    continuity: ContinuitySummary,
    relative_pressure_drop_error: f64,
    relative_velocity_change_l2: f64,
    relative_pressure_change_l2: f64,
    momentum_residual_norm: f64,
    pressure_residual_norm: f64,
    options: &LaminarSimpleOptions,
) -> bool {
    iteration >= options.min_simple_iterations
        && continuity.l2_norm <= options.simple_tolerance
        && relative_pressure_drop_error.is_finite()
        && relative_pressure_drop_error.abs() <= options.pressure_drop_tolerance
        && relative_velocity_change_l2.is_finite()
        && relative_velocity_change_l2 <= options.field_change_tolerance
        && relative_pressure_change_l2.is_finite()
        && relative_pressure_change_l2 <= options.field_change_tolerance
        && optional_absolute_residual_within_control(
            momentum_residual_norm,
            options.momentum_residual_control,
        )
        && optional_absolute_residual_within_control(
            pressure_residual_norm,
            options.pressure_residual_control,
        )
}

fn optional_absolute_residual_within_control(residual: f64, tolerance: Option<f64>) -> bool {
    tolerance.is_none_or(|tolerance| residual.is_finite() && residual <= tolerance)
}

fn points_are_finite(values: &[Point3]) -> bool {
    values
        .iter()
        .all(|value| value.x.is_finite() && value.y.is_finite() && value.z.is_finite())
}

fn scalars_are_finite(values: &[f64]) -> bool {
    values.iter().all(|value| value.is_finite())
}

fn relative_vector_field_change_l2(before: &[Point3], after: &[Point3]) -> f64 {
    let mut delta_squared_sum = 0.0;
    let mut value_squared_sum = 0.0;
    for (before, after) in before.iter().zip(after) {
        let dx = after.x - before.x;
        let dy = after.y - before.y;
        let dz = after.z - before.z;
        delta_squared_sum += dx * dx + dy * dy + dz * dz;
        value_squared_sum += after.x * after.x + after.y * after.y + after.z * after.z;
    }
    let value_norm = value_squared_sum.sqrt();
    if value_norm <= f64::EPSILON {
        delta_squared_sum.sqrt()
    } else {
        delta_squared_sum.sqrt() / value_norm
    }
}

fn relative_scalar_field_change_l2_with_reference(
    before: &[f64],
    after: &[f64],
    reference_value: f64,
) -> f64 {
    let mut delta_squared_sum = 0.0;
    let mut value_squared_sum = 0.0;
    for (before, after) in before.iter().zip(after) {
        let delta = *after - *before;
        delta_squared_sum += delta * delta;
        value_squared_sum += *after * *after;
    }
    let delta_norm = delta_squared_sum.sqrt();
    let value_norm = value_squared_sum.sqrt();
    let reference_norm = if reference_value.is_finite() && reference_value > f64::EPSILON {
        reference_value * (after.len() as f64).sqrt()
    } else {
        0.0
    };
    let denominator = value_norm.max(reference_norm);
    if denominator <= f64::EPSILON {
        delta_norm
    } else {
        delta_norm / denominator
    }
}

fn component_name(component: usize) -> &'static str {
    match component {
        0 => "Ux",
        1 => "Uy",
        2 => "Uz",
        _ => "?",
    }
}

fn assemble_momentum_component_system(
    mesh: &SolverRuntimeMeshData,
    diffusivity: f64,
    density: f64,
    flux: &[f64],
    volumetric_source: &[f64],
    boundary: &[ScalarFaceTreatment],
) -> Result<ScalarComponentSystem> {
    if flux.len() != mesh.faces {
        return Err(invalid_input(format!(
            "momentum flux has {} values, expected {} mesh faces",
            flux.len(),
            mesh.faces
        )));
    }
    if volumetric_source.len() != mesh.cells {
        return Err(invalid_input(format!(
            "momentum component source has {} values, expected {} mesh cells",
            volumetric_source.len(),
            mesh.cells
        )));
    }
    if boundary.len() != mesh.faces {
        return Err(invalid_input(format!(
            "momentum component boundary has {} values, expected {} mesh faces",
            boundary.len(),
            mesh.faces
        )));
    }

    let mut rows = vec![BTreeMap::<usize, f64>::new(); mesh.cells];
    let mut rhs = volumetric_source
        .iter()
        .zip(&mesh.cell_volumes)
        .map(|(source, volume)| source * volume)
        .collect::<Vec<_>>();

    for (face_index, treatment) in boundary.iter().enumerate() {
        let owner = mesh.owner[face_index];
        let mass_flux = density * flux[face_index];
        if !mass_flux.is_finite() {
            return Err(invalid_input(format!(
                "momentum face {face_index} mass flux must be finite, got {mass_flux}"
            )));
        }

        if let Some(neighbour) = mesh.neighbour[face_index] {
            let coefficient = face_diffusion_coefficient(
                diffusivity,
                mesh.face_area_vectors[face_index],
                mesh.cell_centres[owner],
                mesh.cell_centres[neighbour],
                face_index,
            )?;
            add_entry(&mut rows[owner], owner, coefficient);
            add_entry(&mut rows[owner], neighbour, -coefficient);
            add_entry(&mut rows[neighbour], neighbour, coefficient);
            add_entry(&mut rows[neighbour], owner, -coefficient);
            add_internal_upwind_convection(&mut rows, owner, neighbour, mass_flux);
            continue;
        }

        match *treatment {
            ScalarFaceTreatment::FixedValue(value) => {
                let coefficient = face_diffusion_coefficient(
                    diffusivity,
                    mesh.face_area_vectors[face_index],
                    mesh.cell_centres[owner],
                    mesh.face_centres[face_index],
                    face_index,
                )?;
                add_entry(&mut rows[owner], owner, coefficient);
                rhs[owner] += coefficient * value;
                add_boundary_upwind_convection(&mut rows, &mut rhs, owner, value, mass_flux);
            }
            ScalarFaceTreatment::ZeroGradient | ScalarFaceTreatment::Constraint => {
                add_entry(&mut rows[owner], owner, mass_flux);
            }
        }
    }

    let matrix_rows = rows
        .into_iter()
        .map(|row| row.into_iter().collect::<Vec<_>>())
        .collect::<Vec<_>>();
    let matrix = CsrMatrix::from_rows(matrix_rows, mesh.cells)?;

    Ok(ScalarComponentSystem { matrix, rhs })
}

fn add_internal_upwind_convection(
    rows: &mut [BTreeMap<usize, f64>],
    owner: usize,
    neighbour: usize,
    mass_flux: f64,
) {
    if mass_flux >= 0.0 {
        add_entry(&mut rows[owner], owner, mass_flux);
        add_entry(&mut rows[neighbour], owner, -mass_flux);
    } else {
        add_entry(&mut rows[owner], neighbour, mass_flux);
        add_entry(&mut rows[neighbour], neighbour, -mass_flux);
    }
}

fn add_boundary_upwind_convection(
    rows: &mut [BTreeMap<usize, f64>],
    rhs: &mut [f64],
    owner: usize,
    value: f64,
    mass_flux: f64,
) {
    if mass_flux < 0.0 {
        rhs[owner] += -mass_flux * value;
    } else {
        add_entry(&mut rows[owner], owner, mass_flux);
    }
}

fn assemble_variable_scalar_component_system(
    mesh: &SolverRuntimeMeshData,
    cell_diffusivity: &[f64],
    volumetric_source: &[f64],
    boundary: &[ScalarFaceTreatment],
) -> Result<ScalarComponentSystem> {
    if cell_diffusivity.len() != mesh.cells {
        return Err(invalid_input(format!(
            "variable scalar component diffusivity has {} values, expected {} mesh cells",
            cell_diffusivity.len(),
            mesh.cells
        )));
    }
    validate_positive_cell_values("variable scalar component diffusivity", cell_diffusivity)?;
    if volumetric_source.len() != mesh.cells {
        return Err(invalid_input(format!(
            "variable scalar component source has {} values, expected {} mesh cells",
            volumetric_source.len(),
            mesh.cells
        )));
    }
    if boundary.len() != mesh.faces {
        return Err(invalid_input(format!(
            "variable scalar component boundary has {} values, expected {} mesh faces",
            boundary.len(),
            mesh.faces
        )));
    }

    let mut rows = vec![BTreeMap::<usize, f64>::new(); mesh.cells];
    let mut rhs = volumetric_source
        .iter()
        .zip(&mesh.cell_volumes)
        .map(|(source, volume)| source * volume)
        .collect::<Vec<_>>();
    for (face_index, treatment) in boundary.iter().enumerate() {
        let owner = mesh.owner[face_index];
        if let Some(neighbour) = mesh.neighbour[face_index] {
            let coefficient = variable_face_diffusion_coefficient(
                mesh,
                cell_diffusivity,
                owner,
                Some(neighbour),
                face_index,
            )?;
            add_entry(&mut rows[owner], owner, coefficient);
            add_entry(&mut rows[owner], neighbour, -coefficient);
            add_entry(&mut rows[neighbour], neighbour, coefficient);
            add_entry(&mut rows[neighbour], owner, -coefficient);
            continue;
        }

        match *treatment {
            ScalarFaceTreatment::FixedValue(value) => {
                let coefficient = variable_face_diffusion_coefficient(
                    mesh,
                    cell_diffusivity,
                    owner,
                    None,
                    face_index,
                )?;
                add_entry(&mut rows[owner], owner, coefficient);
                rhs[owner] += coefficient * value;
            }
            ScalarFaceTreatment::ZeroGradient | ScalarFaceTreatment::Constraint => {}
        }
    }

    let matrix_rows = rows
        .into_iter()
        .map(|row| row.into_iter().collect::<Vec<_>>())
        .collect::<Vec<_>>();
    let matrix = CsrMatrix::from_rows(matrix_rows, mesh.cells)?;

    Ok(ScalarComponentSystem { matrix, rhs })
}

fn compute_face_flux(
    mesh: &SolverRuntimeMeshData,
    velocity: &[Point3],
    boundary: &[VectorFaceTreatment],
) -> Result<Vec<f64>> {
    if velocity.len() != mesh.cells {
        return Err(invalid_input(format!(
            "velocity has {} values, expected {} mesh cells",
            velocity.len(),
            mesh.cells
        )));
    }
    let mut flux = vec![0.0; mesh.faces];
    for face_index in 0..mesh.faces {
        let face_velocity = face_vector_value(mesh, velocity, boundary, face_index);
        flux[face_index] = dot(face_velocity, mesh.face_area_vectors[face_index]);
    }
    Ok(flux)
}

fn pressure_correction_flux(
    mesh: &SolverRuntimeMeshData,
    pressure_correction: &[f64],
    r_au: &[f64],
    boundary: &[ScalarFaceTreatment],
) -> Result<Vec<f64>> {
    if pressure_correction.len() != mesh.cells {
        return Err(invalid_input(format!(
            "pressure correction flux expected {} cell values, got {}",
            mesh.cells,
            pressure_correction.len()
        )));
    }
    if r_au.len() != mesh.cells {
        return Err(invalid_input(format!(
            "pressure correction rAU has {} values, expected {} mesh cells",
            r_au.len(),
            mesh.cells
        )));
    }
    validate_positive_cell_values("pressure correction rAU", r_au)?;
    if boundary.len() != mesh.faces {
        return Err(invalid_input(format!(
            "pressure correction boundary has {} values, expected {} mesh faces",
            boundary.len(),
            mesh.faces
        )));
    }

    let mut flux = vec![0.0; mesh.faces];
    for (face_index, treatment) in boundary.iter().enumerate() {
        let owner = mesh.owner[face_index];
        if let Some(neighbour) = mesh.neighbour[face_index] {
            let coefficient = variable_face_diffusion_coefficient(
                mesh,
                r_au,
                owner,
                Some(neighbour),
                face_index,
            )?;
            flux[face_index] =
                coefficient * (pressure_correction[owner] - pressure_correction[neighbour]);
            continue;
        }

        match *treatment {
            ScalarFaceTreatment::FixedValue(value) => {
                let coefficient =
                    variable_face_diffusion_coefficient(mesh, r_au, owner, None, face_index)?;
                flux[face_index] = coefficient * (pressure_correction[owner] - value);
            }
            ScalarFaceTreatment::ZeroGradient | ScalarFaceTreatment::Constraint => {}
        }
    }
    Ok(flux)
}

fn add_face_fluxes(left: &[f64], right: &[f64]) -> Result<Vec<f64>> {
    if left.len() != right.len() {
        return Err(invalid_input(format!(
            "face flux arrays have different lengths: {} and {}",
            left.len(),
            right.len()
        )));
    }
    Ok(left
        .iter()
        .zip(right)
        .map(|(left, right)| left + right)
        .collect())
}

fn subtract_face_fluxes(left: &[f64], right: &[f64]) -> Result<Vec<f64>> {
    if left.len() != right.len() {
        return Err(invalid_input(format!(
            "face flux arrays have different lengths: {} and {}",
            left.len(),
            right.len()
        )));
    }
    Ok(left
        .iter()
        .zip(right)
        .map(|(left, right)| left - right)
        .collect())
}

fn scalar_gradient(
    mesh: &SolverRuntimeMeshData,
    values: &[f64],
    boundary: &[ScalarFaceTreatment],
) -> Result<Vec<Point3>> {
    if values.len() != mesh.cells {
        return Err(invalid_input(format!(
            "scalar gradient expected {} cell values, got {}",
            mesh.cells,
            values.len()
        )));
    }
    let mut gradient = vec![zero(); mesh.cells];
    for face_index in 0..mesh.faces {
        let owner = mesh.owner[face_index];
        let face_value = face_scalar_value(mesh, values, boundary, face_index);
        let area = mesh.face_area_vectors[face_index];
        add_scaled(&mut gradient[owner], area, face_value);
        if let Some(neighbour) = mesh.neighbour[face_index] {
            add_scaled(&mut gradient[neighbour], area, -face_value);
        }
    }
    for (value, volume) in gradient.iter_mut().zip(&mesh.cell_volumes) {
        if *volume > f64::EPSILON {
            scale(value, 1.0 / volume);
        }
    }
    Ok(gradient)
}

fn vector_convection_divergence(
    mesh: &SolverRuntimeMeshData,
    velocity: &[Point3],
    boundary: &[VectorFaceTreatment],
    flux: &[f64],
) -> Result<Vec<Point3>> {
    if flux.len() != mesh.faces {
        return Err(invalid_input(format!(
            "flux has {} values, expected {} mesh faces",
            flux.len(),
            mesh.faces
        )));
    }
    let mut divergence = vec![zero(); mesh.cells];
    for (face_index, phi) in flux.iter().copied().enumerate() {
        let owner = mesh.owner[face_index];
        let face_velocity = upwind_face_vector_value(mesh, velocity, boundary, face_index, phi);
        add_scaled(&mut divergence[owner], face_velocity, phi);
        if let Some(neighbour) = mesh.neighbour[face_index] {
            add_scaled(&mut divergence[neighbour], face_velocity, -phi);
        }
    }
    Ok(divergence)
}

fn net_cell_flux(mesh: &SolverRuntimeMeshData, flux: &[f64]) -> Result<Vec<f64>> {
    if flux.len() != mesh.faces {
        return Err(invalid_input(format!(
            "flux has {} values, expected {} mesh faces",
            flux.len(),
            mesh.faces
        )));
    }
    let mut net = vec![0.0; mesh.cells];
    for (face_index, phi) in flux.iter().copied().enumerate() {
        let owner = mesh.owner[face_index];
        net[owner] += phi;
        if let Some(neighbour) = mesh.neighbour[face_index] {
            net[neighbour] -= phi;
        }
    }
    Ok(net)
}

fn pressure_correction_source(mesh: &SolverRuntimeMeshData, net_flux: &[f64]) -> Result<Vec<f64>> {
    if net_flux.len() != mesh.cells {
        return Err(invalid_input(format!(
            "pressure correction source expected {} values, got {}",
            mesh.cells,
            net_flux.len()
        )));
    }
    Ok(net_flux
        .iter()
        .zip(&mesh.cell_volumes)
        .map(|(flux, volume)| {
            if *volume > f64::EPSILON {
                -flux / volume
            } else {
                0.0
            }
        })
        .collect())
}

fn reciprocal_momentum_diagonal(
    mesh: &SolverRuntimeMeshData,
    diagonal: &[f64],
    velocity_relaxation: f64,
) -> Result<Vec<f64>> {
    if diagonal.len() != mesh.cells {
        return Err(invalid_input(format!(
            "momentum diagonal has {} values, expected {} mesh cells",
            diagonal.len(),
            mesh.cells
        )));
    }
    diagonal
        .iter()
        .zip(&mesh.cell_volumes)
        .enumerate()
        .map(|(cell, (diagonal, volume))| {
            if !diagonal.is_finite() || *diagonal <= f64::EPSILON {
                return Err(invalid_input(format!(
                    "momentum diagonal for cell {cell} must be positive and finite, got {diagonal}"
                )));
            }
            if !volume.is_finite() || *volume <= f64::EPSILON {
                return Err(invalid_input(format!(
                    "cell volume for cell {cell} must be positive and finite, got {volume}"
                )));
            }
            Ok(velocity_relaxation * volume / diagonal)
        })
        .collect()
}

fn correct_velocity(
    velocity: &mut [Point3],
    pressure_correction_gradient: &[Point3],
    r_au: &[f64],
    pressure_relaxation: f64,
) {
    for ((value, gradient), r_au) in velocity
        .iter_mut()
        .zip(pressure_correction_gradient)
        .zip(r_au)
    {
        if r_au.is_finite() && *r_au > f64::EPSILON {
            let factor = pressure_relaxation * r_au;
            value.x -= factor * gradient.x;
            value.y -= factor * gradient.y;
            value.z -= factor * gradient.z;
        }
    }
}

fn summarize_laminar_simple_solution(
    mesh: &SolverRuntimeMeshData,
    velocity: &[Point3],
    pressure: &[f64],
    pressure_boundary: &[ScalarFaceTreatment],
    options: &LaminarSimpleOptions,
) -> Result<LaminarSimpleSolutionSummary> {
    let cross_section_area = std::f64::consts::PI * options.diameter * options.diameter / 4.0;
    let analytic_mean_velocity = options.pressure_drop * options.diameter * options.diameter
        / (32.0 * options.dynamic_viscosity * options.length);
    let analytic_flow_rate = analytic_mean_velocity * cross_section_area;
    let (mean_velocity, min_axial_velocity, max_axial_velocity) =
        axial_velocity_summary(mesh, velocity)?;
    let flow_rate = mean_velocity * cross_section_area;
    let pressure_drop_from_mean = 32.0 * options.dynamic_viscosity * options.length * mean_velocity
        / (options.diameter * options.diameter);
    let relative_mean_velocity_error = if analytic_mean_velocity.abs() > f64::EPSILON {
        (mean_velocity - analytic_mean_velocity) / analytic_mean_velocity
    } else {
        0.0
    };
    let relative_pressure_drop_error = if options.pressure_drop.abs() > f64::EPSILON {
        (pressure_drop_from_mean - options.pressure_drop) / options.pressure_drop
    } else {
        0.0
    };
    let pressure_drop_from_field =
        pressure_drop_from_field(mesh, pressure, pressure_boundary, options)?;

    Ok(LaminarSimpleSolutionSummary {
        mean_velocity,
        analytic_mean_velocity,
        relative_mean_velocity_error,
        flow_rate,
        analytic_flow_rate,
        pressure_drop_from_mean,
        relative_pressure_drop_error,
        pressure_drop_from_field,
        min_axial_velocity,
        max_axial_velocity,
    })
}

fn relative_pressure_drop_error_from_velocity(
    mesh: &SolverRuntimeMeshData,
    velocity: &[Point3],
    options: &LaminarSimpleOptions,
) -> Result<f64> {
    let pressure_drop_from_mean = pressure_drop_from_mean_velocity(mesh, velocity, options)?;
    if options.pressure_drop.abs() > f64::EPSILON {
        Ok((pressure_drop_from_mean - options.pressure_drop) / options.pressure_drop)
    } else {
        Ok(0.0)
    }
}

fn pressure_drop_from_mean_velocity(
    mesh: &SolverRuntimeMeshData,
    velocity: &[Point3],
    options: &LaminarSimpleOptions,
) -> Result<f64> {
    let (mean_velocity, _, _) = axial_velocity_summary(mesh, velocity)?;
    Ok(
        32.0 * options.dynamic_viscosity * options.length * mean_velocity
            / (options.diameter * options.diameter),
    )
}

fn pressure_drop_from_field(
    mesh: &SolverRuntimeMeshData,
    pressure: &[f64],
    pressure_boundary: &[ScalarFaceTreatment],
    options: &LaminarSimpleOptions,
) -> Result<Option<f64>> {
    Ok(
        patch_owner_average_scalar(mesh, pressure, pressure_boundary, &options.inlet_patch)?
            .zip(patch_owner_average_scalar(
                mesh,
                pressure,
                pressure_boundary,
                &options.outlet_patch,
            )?)
            .map(|(inlet, outlet)| inlet - outlet),
    )
}

fn axial_velocity_summary(
    mesh: &SolverRuntimeMeshData,
    velocity: &[Point3],
) -> Result<(f64, f64, f64)> {
    let mut weighted_sum = 0.0;
    let mut total_volume = 0.0;
    let mut min_axial_velocity = f64::INFINITY;
    let mut max_axial_velocity = f64::NEG_INFINITY;
    for (value, volume) in velocity.iter().zip(&mesh.cell_volumes) {
        weighted_sum += value.x * volume;
        total_volume += volume;
        min_axial_velocity = min_axial_velocity.min(value.x);
        max_axial_velocity = max_axial_velocity.max(value.x);
    }
    if total_volume <= f64::EPSILON {
        return Err(invalid_input(
            "laminar SIMPLE summary requires positive total cell volume".to_string(),
        ));
    }
    Ok((
        weighted_sum / total_volume,
        min_axial_velocity,
        max_axial_velocity,
    ))
}

fn patch_owner_average_scalar(
    mesh: &SolverRuntimeMeshData,
    values: &[f64],
    boundary: &[ScalarFaceTreatment],
    patch_name: &str,
) -> Result<Option<f64>> {
    let Some(patch) = mesh.patches.iter().find(|patch| patch.name == patch_name) else {
        return Ok(None);
    };
    let mut sum = 0.0;
    let mut count = 0;
    for face_index in patch.start_face..patch.start_face + patch.faces {
        if mesh.neighbour[face_index].is_some() {
            continue;
        }
        sum += face_scalar_value(mesh, values, boundary, face_index);
        count += 1;
    }
    if count == 0 {
        Ok(None)
    } else {
        Ok(Some(sum / count as f64))
    }
}

fn summarize_continuity(net_flux: &[f64]) -> ContinuitySummary {
    let mut summary = ContinuitySummary::default();
    for value in net_flux {
        let abs = value.abs();
        summary.max_abs = summary.max_abs.max(abs);
        summary.sum_abs += abs;
        summary.global_sum += value;
    }
    summary.l2_norm = l2_norm(net_flux);
    summary
}

fn summarize_operators(
    phi: &[f64],
    grad_p: &[Point3],
    convection: &[Point3],
) -> FlowOperatorSummary {
    let mut phi_min = f64::INFINITY;
    let mut phi_max = f64::NEG_INFINITY;
    let mut phi_sum_abs = 0.0;
    for value in phi {
        phi_min = phi_min.min(*value);
        phi_max = phi_max.max(*value);
        phi_sum_abs += value.abs();
    }
    if phi.is_empty() {
        phi_min = 0.0;
        phi_max = 0.0;
    }
    FlowOperatorSummary {
        phi_min,
        phi_max,
        phi_sum_abs,
        grad_p_l2_norm: vector_l2_norm(grad_p),
        div_phi_u_l2_norm: vector_l2_norm(convection),
    }
}

fn summarize_boundaries(
    mesh: &SolverRuntimeMeshData,
    velocity: &[VectorFaceTreatment],
    pressure: &[ScalarFaceTreatment],
) -> FlowBoundarySummary {
    let mut summary = FlowBoundarySummary::default();
    for (face, treatment) in velocity.iter().enumerate() {
        if mesh.neighbour[face].is_some() {
            continue;
        }
        match treatment {
            VectorFaceTreatment::FixedValue(_) => summary.velocity_fixed_value_faces += 1,
            VectorFaceTreatment::ZeroGradient => summary.velocity_zero_gradient_faces += 1,
            VectorFaceTreatment::Constraint => summary.velocity_constraint_faces += 1,
        }
    }
    for (face, treatment) in pressure.iter().enumerate() {
        if mesh.neighbour[face].is_some() {
            continue;
        }
        match treatment {
            ScalarFaceTreatment::FixedValue(_) => summary.pressure_fixed_value_faces += 1,
            ScalarFaceTreatment::ZeroGradient => summary.pressure_zero_gradient_faces += 1,
            ScalarFaceTreatment::Constraint => summary.pressure_constraint_faces += 1,
        }
    }
    summary
}

fn vector_face_treatments(
    mesh: &SolverRuntimeMeshData,
    field: &FieldFile,
) -> Result<Vec<VectorFaceTreatment>> {
    let mut treatments = vec![VectorFaceTreatment::ZeroGradient; mesh.faces];
    for patch in &mesh.patches {
        let field_patch = field_patch(field, &patch.name)?;
        let patch_treatments = if is_constraint_patch(&patch.patch_type) {
            vec![VectorFaceTreatment::Constraint; patch.faces]
        } else {
            match field_patch.patch_type.as_deref() {
                Some("fixedValue") => fixed_vector_patch_values(field, field_patch, patch.faces)?,
                Some("noSlip") => vec![VectorFaceTreatment::FixedValue(zero()); patch.faces],
                Some("zeroGradient") => vec![VectorFaceTreatment::ZeroGradient; patch.faces],
                Some("empty" | "wedge" | "symmetryPlane") => {
                    vec![VectorFaceTreatment::Constraint; patch.faces]
                }
                Some(other) => {
                    return Err(invalid_input(format!(
                        "laminar SIMPLE U patch '{}' uses unsupported boundary type '{}'",
                        patch.name, other
                    )));
                }
                None => {
                    return Err(invalid_input(format!(
                        "laminar SIMPLE U patch '{}' has no boundary type",
                        patch.name
                    )));
                }
            }
        };
        for (offset, treatment) in patch_treatments.into_iter().enumerate() {
            treatments[patch.start_face + offset] = treatment;
        }
    }
    Ok(treatments)
}

fn scalar_face_treatments(
    mesh: &SolverRuntimeMeshData,
    field: &FieldFile,
) -> Result<Vec<ScalarFaceTreatment>> {
    let mut treatments = vec![ScalarFaceTreatment::ZeroGradient; mesh.faces];
    for patch in &mesh.patches {
        let field_patch = field_patch(field, &patch.name)?;
        let patch_treatments = if is_constraint_patch(&patch.patch_type) {
            vec![ScalarFaceTreatment::Constraint; patch.faces]
        } else {
            match field_patch.patch_type.as_deref() {
                Some("fixedValue") => fixed_scalar_patch_values(field, field_patch, patch.faces)?,
                Some("zeroGradient") => vec![ScalarFaceTreatment::ZeroGradient; patch.faces],
                Some("empty" | "wedge" | "symmetryPlane") => {
                    vec![ScalarFaceTreatment::Constraint; patch.faces]
                }
                Some(other) => {
                    return Err(invalid_input(format!(
                        "laminar SIMPLE p patch '{}' uses unsupported boundary type '{}'",
                        patch.name, other
                    )));
                }
                None => {
                    return Err(invalid_input(format!(
                        "laminar SIMPLE p patch '{}' has no boundary type",
                        patch.name
                    )));
                }
            }
        };
        for (offset, treatment) in patch_treatments.into_iter().enumerate() {
            treatments[patch.start_face + offset] = treatment;
        }
    }
    Ok(treatments)
}

fn pressure_correction_treatments(pressure: &[ScalarFaceTreatment]) -> Vec<ScalarFaceTreatment> {
    pressure
        .iter()
        .map(|treatment| match treatment {
            ScalarFaceTreatment::FixedValue(_) => ScalarFaceTreatment::FixedValue(0.0),
            ScalarFaceTreatment::ZeroGradient => ScalarFaceTreatment::ZeroGradient,
            ScalarFaceTreatment::Constraint => ScalarFaceTreatment::Constraint,
        })
        .collect()
}

fn fixed_vector_patch_values(
    field: &FieldFile,
    patch: &crate::fields::FieldBoundaryPatch,
    faces: usize,
) -> Result<Vec<VectorFaceTreatment>> {
    let value = patch.value.as_ref().ok_or_else(|| {
        invalid_input(format!(
            "field '{}' patch '{}' fixedValue has no value",
            field_label(field),
            patch.name
        ))
    })?;
    let values = parse_patch_numeric_values(value, 3, faces, field, &patch.name)?;
    Ok(values
        .chunks_exact(3)
        .map(|chunk| {
            VectorFaceTreatment::FixedValue(Point3 {
                x: chunk[0],
                y: chunk[1],
                z: chunk[2],
            })
        })
        .collect())
}

fn fixed_scalar_patch_values(
    field: &FieldFile,
    patch: &crate::fields::FieldBoundaryPatch,
    faces: usize,
) -> Result<Vec<ScalarFaceTreatment>> {
    let value = patch.value.as_ref().ok_or_else(|| {
        invalid_input(format!(
            "field '{}' patch '{}' fixedValue has no value",
            field_label(field),
            patch.name
        ))
    })?;
    let values = parse_patch_numeric_values(value, 1, faces, field, &patch.name)?;
    Ok(values
        .into_iter()
        .map(ScalarFaceTreatment::FixedValue)
        .collect())
}

fn parse_patch_numeric_values(
    value: &FieldValueSummary,
    components: usize,
    faces: usize,
    field: &FieldFile,
    patch: &str,
) -> Result<Vec<f64>> {
    match value {
        FieldValueSummary::Uniform(value) => {
            let values = parse_numbers(value, field, patch)?;
            if values.len() != components {
                return Err(invalid_input(format!(
                    "field '{}' patch '{}' uniform value has {} components, expected {}",
                    field_label(field),
                    patch,
                    values.len(),
                    components
                )));
            }
            let mut expanded = Vec::with_capacity(faces * components);
            for _ in 0..faces {
                expanded.extend(values.iter().copied());
            }
            Ok(expanded)
        }
        FieldValueSummary::NonUniform { values, count, .. } => {
            if count.is_some_and(|count| count != faces) {
                return Err(invalid_input(format!(
                    "field '{}' patch '{}' nonuniform value count is {:?}, expected {}",
                    field_label(field),
                    patch,
                    count,
                    faces
                )));
            }
            let values = values.as_ref().ok_or_else(|| {
                invalid_input(format!(
                    "field '{}' patch '{}' nonuniform numeric values could not be loaded",
                    field_label(field),
                    patch
                ))
            })?;
            let expected = faces * components;
            if values.len() != expected {
                return Err(invalid_input(format!(
                    "field '{}' patch '{}' nonuniform value has {} scalars, expected {}",
                    field_label(field),
                    patch,
                    values.len(),
                    expected
                )));
            }
            Ok(values.clone())
        }
        FieldValueSummary::Other(value) => Err(invalid_input(format!(
            "field '{}' patch '{}' has unsupported value '{}'",
            field_label(field),
            patch,
            value
        ))),
    }
}

fn parse_numbers(value: &str, field: &FieldFile, patch: &str) -> Result<Vec<f64>> {
    value
        .replace(['(', ')'], " ")
        .split_whitespace()
        .map(|token| {
            token.parse::<f64>().map_err(|_| {
                invalid_input(format!(
                    "field '{}' patch '{}' contains non-numeric token '{}'",
                    field_label(field),
                    patch,
                    token
                ))
            })
        })
        .collect()
}

fn field_patch<'a>(
    field: &'a FieldFile,
    patch_name: &str,
) -> Result<&'a crate::fields::FieldBoundaryPatch> {
    field
        .boundary_patches
        .iter()
        .find(|patch| patch.name == patch_name)
        .ok_or_else(|| {
            invalid_input(format!(
                "field '{}' has no boundaryField entry for mesh patch '{}'",
                field_label(field),
                patch_name
            ))
        })
}

fn runtime_vector_field(runtime: &SolverRuntimeData, name: &str) -> Result<Vec<Point3>> {
    let buffer = runtime_field(runtime, name, 3)?;
    Ok(buffer
        .chunks_exact(3)
        .map(|chunk| Point3 {
            x: chunk[0],
            y: chunk[1],
            z: chunk[2],
        })
        .collect())
}

fn runtime_scalar_field(runtime: &SolverRuntimeData, name: &str) -> Result<Vec<f64>> {
    Ok(runtime_field(runtime, name, 1)?.to_vec())
}

fn runtime_field<'a>(
    runtime: &'a SolverRuntimeData,
    name: &str,
    components: usize,
) -> Result<&'a [f64]> {
    let buffer = runtime
        .fields
        .iter()
        .find(|field| {
            field.region.is_none() && field.name == name && field.components == components
        })
        .ok_or_else(|| {
            invalid_input(format!(
                "runtime field '{}' with {} components was not materialized",
                name, components
            ))
        })?;
    let expected = runtime.mesh.cells * components;
    if buffer.values.len() != expected {
        return Err(invalid_input(format!(
            "runtime field '{}' has {} scalars, expected {}",
            name,
            buffer.values.len(),
            expected
        )));
    }
    Ok(buffer.values.as_slice())
}

fn find_field<'a>(
    fields: &'a InitialFieldSet,
    name: &str,
    class_name: &str,
) -> Result<&'a FieldFile> {
    let field = fields
        .fields
        .iter()
        .find(|field| field.region.is_none() && field.name == name)
        .ok_or_else(|| {
            invalid_input(format!(
                "field '{}' was not found below {}",
                name,
                fields.case_dir.join("0").display()
            ))
        })?;
    if field.class_name.as_deref() != Some(class_name) {
        return Err(invalid_input(format!(
            "field '{}' has class '{}', expected '{}'",
            name,
            field.class_name.as_deref().unwrap_or("unknown"),
            class_name
        )));
    }
    Ok(field)
}

fn face_vector_value(
    mesh: &SolverRuntimeMeshData,
    velocity: &[Point3],
    boundary: &[VectorFaceTreatment],
    face_index: usize,
) -> Point3 {
    let owner = mesh.owner[face_index];
    if let Some(neighbour) = mesh.neighbour[face_index] {
        return average(velocity[owner], velocity[neighbour]);
    }
    match boundary[face_index] {
        VectorFaceTreatment::FixedValue(value) => value,
        VectorFaceTreatment::ZeroGradient | VectorFaceTreatment::Constraint => velocity[owner],
    }
}

fn upwind_face_vector_value(
    mesh: &SolverRuntimeMeshData,
    velocity: &[Point3],
    boundary: &[VectorFaceTreatment],
    face_index: usize,
    flux: f64,
) -> Point3 {
    let owner = mesh.owner[face_index];
    if let Some(neighbour) = mesh.neighbour[face_index] {
        return if flux >= 0.0 {
            velocity[owner]
        } else {
            velocity[neighbour]
        };
    }
    match boundary[face_index] {
        VectorFaceTreatment::FixedValue(value) if flux < 0.0 => value,
        VectorFaceTreatment::FixedValue(_) => velocity[owner],
        VectorFaceTreatment::ZeroGradient | VectorFaceTreatment::Constraint => velocity[owner],
    }
}

fn face_scalar_value(
    mesh: &SolverRuntimeMeshData,
    values: &[f64],
    boundary: &[ScalarFaceTreatment],
    face_index: usize,
) -> f64 {
    let owner = mesh.owner[face_index];
    if let Some(neighbour) = mesh.neighbour[face_index] {
        return 0.5 * (values[owner] + values[neighbour]);
    }
    match boundary[face_index] {
        ScalarFaceTreatment::FixedValue(value) => value,
        ScalarFaceTreatment::ZeroGradient | ScalarFaceTreatment::Constraint => values[owner],
    }
}

fn scalar_component_boundary(
    boundary: &[VectorFaceTreatment],
    component: usize,
) -> Vec<ScalarFaceTreatment> {
    boundary
        .iter()
        .map(|treatment| match treatment {
            VectorFaceTreatment::FixedValue(value) => {
                ScalarFaceTreatment::FixedValue(component_value(*value, component))
            }
            VectorFaceTreatment::ZeroGradient => ScalarFaceTreatment::ZeroGradient,
            VectorFaceTreatment::Constraint => ScalarFaceTreatment::Constraint,
        })
        .collect()
}

fn split_components(values: &[Point3]) -> [Vec<f64>; 3] {
    let mut x = Vec::with_capacity(values.len());
    let mut y = Vec::with_capacity(values.len());
    let mut z = Vec::with_capacity(values.len());
    for value in values {
        x.push(value.x);
        y.push(value.y);
        z.push(value.z);
    }
    [x, y, z]
}

fn combine_components(x: &[f64], y: &[f64], z: &[f64]) -> Vec<Point3> {
    x.iter()
        .zip(y)
        .zip(z)
        .map(|((x, y), z)| Point3 {
            x: *x,
            y: *y,
            z: *z,
        })
        .collect()
}

fn component_value(value: Point3, component: usize) -> f64 {
    match component {
        0 => value.x,
        1 => value.y,
        2 => value.z,
        _ => 0.0,
    }
}

fn validate_laminar_simple_options(options: &LaminarSimpleOptions) -> Result<()> {
    if !options.density.is_finite() || options.density <= 0.0 {
        return Err(invalid_input(format!(
            "laminar SIMPLE density must be positive and finite, got {}",
            options.density
        )));
    }
    if !options.dynamic_viscosity.is_finite() || options.dynamic_viscosity <= 0.0 {
        return Err(invalid_input(format!(
            "laminar SIMPLE dynamic viscosity must be positive and finite, got {}",
            options.dynamic_viscosity
        )));
    }
    if !options.pressure_drop.is_finite() || options.pressure_drop <= 0.0 {
        return Err(invalid_input(format!(
            "laminar SIMPLE pressure drop must be positive and finite, got {}",
            options.pressure_drop
        )));
    }
    if !options.length.is_finite() || options.length <= 0.0 {
        return Err(invalid_input(format!(
            "laminar SIMPLE length must be positive and finite, got {}",
            options.length
        )));
    }
    if !options.diameter.is_finite() || options.diameter <= 0.0 {
        return Err(invalid_input(format!(
            "laminar SIMPLE diameter must be positive and finite, got {}",
            options.diameter
        )));
    }
    if options.max_linear_iterations == 0
        || options.momentum_max_linear_iterations == 0
        || options.pressure_max_linear_iterations == 0
        || options.max_simple_iterations == 0
        || options.min_simple_iterations == 0
    {
        return Err(invalid_input(
            "laminar SIMPLE iteration limits must be greater than zero".to_string(),
        ));
    }
    if options.min_simple_iterations > options.max_simple_iterations {
        return Err(invalid_input(format!(
            "laminar SIMPLE minSimpleIterations must not exceed maxSimpleIterations, got {} > {}",
            options.min_simple_iterations, options.max_simple_iterations
        )));
    }
    if !options.linear_tolerance.is_finite() || options.linear_tolerance < 0.0 {
        return Err(invalid_input(format!(
            "laminar SIMPLE linear tolerance must be non-negative and finite, got {}",
            options.linear_tolerance
        )));
    }
    validate_linear_tolerance("momentum", options.momentum_linear_tolerance)?;
    validate_linear_tolerance("pressure", options.pressure_linear_tolerance)?;
    validate_solver_preconditioner(
        "momentum",
        options.momentum_linear_solver,
        options.momentum_preconditioner,
    )?;
    validate_solver_preconditioner(
        "pressure",
        options.pressure_linear_solver,
        options.pressure_preconditioner,
    )?;
    if !options.simple_tolerance.is_finite() || options.simple_tolerance < 0.0 {
        return Err(invalid_input(format!(
            "laminar SIMPLE tolerance must be non-negative and finite, got {}",
            options.simple_tolerance
        )));
    }
    if !options.pressure_drop_tolerance.is_finite() || options.pressure_drop_tolerance < 0.0 {
        return Err(invalid_input(format!(
            "laminar SIMPLE pressure-drop tolerance must be non-negative and finite, got {}",
            options.pressure_drop_tolerance
        )));
    }
    if !options.field_change_tolerance.is_finite() || options.field_change_tolerance < 0.0 {
        return Err(invalid_input(format!(
            "laminar SIMPLE field-change tolerance must be non-negative and finite, got {}",
            options.field_change_tolerance
        )));
    }
    validate_optional_non_negative(
        "laminar SIMPLE U residualControl",
        options.momentum_residual_control,
    )?;
    validate_optional_non_negative(
        "laminar SIMPLE p residualControl",
        options.pressure_residual_control,
    )?;
    validate_relaxation("velocity", options.velocity_relaxation)?;
    validate_relaxation("pressure", options.pressure_relaxation)?;
    if options.inlet_patch.trim().is_empty() || options.outlet_patch.trim().is_empty() {
        return Err(invalid_input(
            "laminar SIMPLE inlet and outlet patch names must not be empty".to_string(),
        ));
    }
    Ok(())
}

fn validate_solver_preconditioner(
    name: &str,
    solver: LaminarSimpleLinearSolver,
    preconditioner: LaminarSimplePreconditioner,
) -> Result<()> {
    if preconditioner != LaminarSimplePreconditioner::None
        && solver != LaminarSimpleLinearSolver::Pcg
    {
        return Err(invalid_input(format!(
            "laminar SIMPLE {name} preconditioner {preconditioner} requires pcg solver, got {solver}"
        )));
    }
    Ok(())
}

fn validate_linear_tolerance(name: &str, value: f64) -> Result<()> {
    if !value.is_finite() || value < 0.0 {
        return Err(invalid_input(format!(
            "laminar SIMPLE {name} linear tolerance must be non-negative and finite, got {value}"
        )));
    }
    Ok(())
}

fn validate_relaxation(name: &str, value: f64) -> Result<()> {
    if !value.is_finite() || value <= 0.0 || value > 1.0 {
        return Err(invalid_input(format!(
            "laminar SIMPLE {name} relaxation must be in (0, 1], got {value}"
        )));
    }
    Ok(())
}

fn validate_optional_non_negative(name: &str, value: Option<f64>) -> Result<()> {
    if let Some(value) = value {
        if !value.is_finite() || value < 0.0 {
            return Err(invalid_input(format!(
                "{name} must be non-negative and finite, got {value}"
            )));
        }
    }
    Ok(())
}

fn validate_runtime_mesh(mesh: &SolverRuntimeMeshData) -> Result<()> {
    if mesh.owner.len() != mesh.faces
        || mesh.neighbour.len() != mesh.faces
        || mesh.face_centres.len() != mesh.faces
        || mesh.face_area_vectors.len() != mesh.faces
    {
        return Err(invalid_input(
            "runtime mesh face arrays are inconsistent".to_string(),
        ));
    }
    if mesh.cell_centres.len() != mesh.cells || mesh.cell_volumes.len() != mesh.cells {
        return Err(invalid_input(
            "runtime mesh cell arrays are inconsistent".to_string(),
        ));
    }
    for (face_index, owner) in mesh.owner.iter().copied().enumerate() {
        if owner >= mesh.cells {
            return Err(invalid_input(format!(
                "runtime mesh face {face_index} owner {owner} exceeds cell count {}",
                mesh.cells
            )));
        }
        if let Some(neighbour) = mesh.neighbour[face_index]
            && neighbour >= mesh.cells
        {
            return Err(invalid_input(format!(
                "runtime mesh face {face_index} neighbour {neighbour} exceeds cell count {}",
                mesh.cells
            )));
        }
    }
    Ok(())
}

fn face_diffusion_coefficient(
    diffusivity: f64,
    area_vector: Point3,
    from: Point3,
    to: Point3,
    face_index: usize,
) -> Result<f64> {
    let area = magnitude(area_vector);
    let distance = distance(from, to);
    if !area.is_finite() || area <= f64::EPSILON {
        return Err(invalid_input(format!(
            "face {face_index} has non-positive area magnitude {area}"
        )));
    }
    if !distance.is_finite() || distance <= f64::EPSILON {
        return Err(invalid_input(format!(
            "face {face_index} has non-positive diffusion distance {distance}"
        )));
    }
    Ok(diffusivity * area / distance)
}

fn variable_face_diffusion_coefficient(
    mesh: &SolverRuntimeMeshData,
    cell_diffusivity: &[f64],
    owner: usize,
    neighbour: Option<usize>,
    face_index: usize,
) -> Result<f64> {
    let diffusivity = if let Some(neighbour) = neighbour {
        0.5 * (cell_diffusivity[owner] + cell_diffusivity[neighbour])
    } else {
        cell_diffusivity[owner]
    };
    let to = if let Some(neighbour) = neighbour {
        mesh.cell_centres[neighbour]
    } else {
        mesh.face_centres[face_index]
    };
    face_diffusion_coefficient(
        diffusivity,
        mesh.face_area_vectors[face_index],
        mesh.cell_centres[owner],
        to,
        face_index,
    )
}

fn validate_positive_cell_values(name: &str, values: &[f64]) -> Result<()> {
    for (index, value) in values.iter().copied().enumerate() {
        if !value.is_finite() || value <= f64::EPSILON {
            return Err(invalid_input(format!(
                "{name} value for cell {index} must be positive and finite, got {value}"
            )));
        }
    }
    Ok(())
}

fn add_entry(row: &mut BTreeMap<usize, f64>, col: usize, value: f64) {
    *row.entry(col).or_insert(0.0) += value;
}

fn add_scaled(target: &mut Point3, value: Point3, scale_value: f64) {
    target.x += value.x * scale_value;
    target.y += value.y * scale_value;
    target.z += value.z * scale_value;
}

fn scale(value: &mut Point3, scale_value: f64) {
    value.x *= scale_value;
    value.y *= scale_value;
    value.z *= scale_value;
}

fn average(left: Point3, right: Point3) -> Point3 {
    Point3 {
        x: 0.5 * (left.x + right.x),
        y: 0.5 * (left.y + right.y),
        z: 0.5 * (left.z + right.z),
    }
}

fn dot(left: Point3, right: Point3) -> f64 {
    left.x * right.x + left.y * right.y + left.z * right.z
}

fn magnitude(value: Point3) -> f64 {
    dot(value, value).sqrt()
}

fn distance(left: Point3, right: Point3) -> f64 {
    let dx = left.x - right.x;
    let dy = left.y - right.y;
    let dz = left.z - right.z;
    (dx * dx + dy * dy + dz * dz).sqrt()
}

fn vector_l2_norm(values: &[Point3]) -> f64 {
    values
        .iter()
        .map(|value| dot(*value, *value))
        .sum::<f64>()
        .sqrt()
}

fn zero() -> Point3 {
    Point3 {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    }
}

fn is_constraint_patch(patch_type: &str) -> bool {
    matches!(patch_type, "empty" | "wedge" | "symmetryPlane")
}

fn field_label(field: &FieldFile) -> String {
    if let Some(region) = &field.region {
        format!("{region}/{}", field.name)
    } else {
        field.name.clone()
    }
}

fn invalid_input(message: String) -> MeshError {
    MeshError::InvalidInput(message)
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use crate::fields::{FieldBoundaryPatch, FieldFile, FieldValueSummary, InitialFieldSet};
    use crate::runtime::{
        SolverRuntimeData, SolverRuntimeFieldBuffer, SolverRuntimeMeshData, SolverRuntimePatchRange,
    };
    use crate::solver_state::SolverStateFieldKind;

    use super::{
        LaminarSimpleLinearSolver, LaminarSimpleOptions, LaminarSimplePreconditioner,
        ScalarFaceTreatment, assemble_momentum_component_system,
        assemble_variable_scalar_component_system, compute_face_flux, net_cell_flux,
        pressure_correction_flux, reciprocal_momentum_diagonal, relax_scalar_component_equation,
        scalar_component_boundary, solve_laminar_simple, upwind_face_vector_value,
        vector_face_treatments,
    };
    use crate::Point3;

    #[test]
    fn computes_owner_oriented_flux() {
        let runtime = two_cell_runtime();
        let fields = two_cell_fields();
        let u_field = fields
            .fields
            .iter()
            .find(|field| field.name == "U")
            .expect("U field");
        let boundary = vector_face_treatments(&runtime.mesh, u_field).expect("U boundary");
        let velocity = vec![
            Point3 {
                x: 1.0,
                y: 0.0,
                z: 0.0,
            },
            Point3 {
                x: 3.0,
                y: 0.0,
                z: 0.0,
            },
        ];

        let flux = compute_face_flux(&runtime.mesh, &velocity, &boundary).expect("flux");

        assert_eq!(flux.len(), 3);
        assert_close(flux[0], 2.0);
        assert_close(flux[1], -1.0);
        assert_close(flux[2], 3.0);
    }

    #[test]
    fn selects_upwind_momentum_face_value_from_flux_direction() {
        let runtime = two_cell_runtime();
        let fields = two_cell_fields();
        let u_field = fields
            .fields
            .iter()
            .find(|field| field.name == "U")
            .expect("U field");
        let boundary = vector_face_treatments(&runtime.mesh, u_field).expect("U boundary");
        let velocity = vec![point(5.0, 0.0, 0.0), point(3.0, 0.0, 0.0)];

        let owner_to_neighbour =
            upwind_face_vector_value(&runtime.mesh, &velocity, &boundary, 0, 1.0);
        let neighbour_to_owner =
            upwind_face_vector_value(&runtime.mesh, &velocity, &boundary, 0, -1.0);
        let inlet_inflow = upwind_face_vector_value(&runtime.mesh, &velocity, &boundary, 1, -1.0);
        let outlet_outflow = upwind_face_vector_value(&runtime.mesh, &velocity, &boundary, 2, 1.0);

        assert_close(owner_to_neighbour.x, 5.0);
        assert_close(neighbour_to_owner.x, 3.0);
        assert_close(inlet_inflow.x, 1.0);
        assert_close(outlet_outflow.x, 3.0);
    }

    #[test]
    fn builds_cell_reciprocal_momentum_diagonal() {
        let runtime = two_cell_runtime();

        let r_au =
            reciprocal_momentum_diagonal(&runtime.mesh, &[2.0, 4.0], 1.0).expect("cell rAU values");

        assert_eq!(r_au.len(), 2);
        assert_close(r_au[0], 0.25);
        assert_close(r_au[1], 0.125);

        let relaxed_r_au = reciprocal_momentum_diagonal(&runtime.mesh, &[2.0, 4.0], 0.5)
            .expect("relaxed cell rAU values");
        assert_close(relaxed_r_au[0], 0.125);
        assert_close(relaxed_r_au[1], 0.0625);
    }

    #[test]
    fn assembles_implicit_upwind_momentum_convection() {
        let runtime = two_cell_runtime();
        let fields = two_cell_fields();
        let u_field = fields
            .fields
            .iter()
            .find(|field| field.name == "U")
            .expect("U field");
        let vector_boundary = vector_face_treatments(&runtime.mesh, u_field).expect("U boundary");
        let boundary = scalar_component_boundary(&vector_boundary, 0);
        let flux = vec![1.0, -2.0, 3.0];
        let source = vec![0.0, 0.0];

        let system =
            assemble_momentum_component_system(&runtime.mesh, 1.0, 2.0, &flux, &source, &boundary)
                .expect("momentum system");

        let diagonal = system.matrix.diagonal().expect("diagonal");
        assert_close(diagonal[0], 8.0);
        assert_close(diagonal[1], 8.0);
        assert_close(system.rhs[0], 8.0);
        assert_close(system.rhs[1], 0.0);

        let owner_column = system.matrix.matvec(&[1.0, 0.0]).expect("owner column");
        let neighbour_column = system.matrix.matvec(&[0.0, 1.0]).expect("neighbour column");
        assert_close(owner_column[0], 8.0);
        assert_close(owner_column[1], -4.0);
        assert_close(neighbour_column[0], -2.0);
        assert_close(neighbour_column[1], 8.0);
    }

    #[test]
    fn equation_relaxation_preserves_original_diagonal_for_rau() {
        let runtime = two_cell_runtime();
        let fields = two_cell_fields();
        let u_field = fields
            .fields
            .iter()
            .find(|field| field.name == "U")
            .expect("U field");
        let vector_boundary = vector_face_treatments(&runtime.mesh, u_field).expect("U boundary");
        let boundary = scalar_component_boundary(&vector_boundary, 0);
        let flux = vec![1.0, -2.0, 3.0];
        let source = vec![0.0, 0.0];
        let mut system =
            assemble_momentum_component_system(&runtime.mesh, 1.0, 2.0, &flux, &source, &boundary)
                .expect("momentum system");

        let original_diagonal =
            relax_scalar_component_equation(&mut system, &[5.0, 3.0], 0.5).expect("relaxed system");

        assert_close(original_diagonal[0], 8.0);
        assert_close(original_diagonal[1], 8.0);
        let relaxed_diagonal = system.matrix.diagonal().expect("relaxed diagonal");
        assert_close(relaxed_diagonal[0], 16.0);
        assert_close(relaxed_diagonal[1], 16.0);
        assert_close(system.rhs[0], 48.0);
        assert_close(system.rhs[1], 24.0);
    }

    #[test]
    fn pressure_correction_flux_matches_variable_laplacian_balance() {
        let runtime = two_cell_runtime();
        let boundary = vec![ScalarFaceTreatment::ZeroGradient; runtime.mesh.faces];
        let r_au = vec![1.0, 2.0];
        let pressure_correction = vec![2.0, 0.0];
        let source = vec![0.0, 0.0];
        let system =
            assemble_variable_scalar_component_system(&runtime.mesh, &r_au, &source, &boundary)
                .expect("variable pressure system");

        let matrix_balance = system
            .matrix
            .matvec(&pressure_correction)
            .expect("matrix balance");
        let face_flux =
            pressure_correction_flux(&runtime.mesh, &pressure_correction, &r_au, &boundary)
                .expect("pressure correction flux");
        let flux_balance = net_cell_flux(&runtime.mesh, &face_flux).expect("flux balance");

        assert_eq!(matrix_balance.len(), flux_balance.len());
        for (matrix_value, flux_value) in matrix_balance.iter().zip(&flux_balance) {
            assert_close(*matrix_value, *flux_value);
        }
    }

    #[test]
    fn runs_minimal_simple_loop_on_two_cells() {
        let runtime = two_cell_runtime();
        let fields = two_cell_fields();
        let options = LaminarSimpleOptions {
            density: 1.0,
            dynamic_viscosity: 1.0,
            pressure_drop: 1.0,
            length: 1.0,
            diameter: 1.0,
            inlet_patch: "inlet".to_string(),
            outlet_patch: "outlet".to_string(),
            linear_solver: LaminarSimpleLinearSolver::Cg,
            momentum_linear_solver: LaminarSimpleLinearSolver::Cg,
            pressure_linear_solver: LaminarSimpleLinearSolver::Cg,
            momentum_preconditioner: LaminarSimplePreconditioner::None,
            pressure_preconditioner: LaminarSimplePreconditioner::None,
            linear_tolerance: 1.0e-10,
            max_linear_iterations: 100,
            momentum_linear_tolerance: 1.0e-10,
            pressure_linear_tolerance: 1.0e-10,
            momentum_max_linear_iterations: 100,
            pressure_max_linear_iterations: 100,
            max_simple_iterations: 3,
            min_simple_iterations: 1,
            simple_tolerance: 1.0e-12,
            pressure_drop_tolerance: 1.0,
            field_change_tolerance: 1.0,
            momentum_residual_control: None,
            pressure_residual_control: None,
            velocity_relaxation: 0.7,
            pressure_relaxation: 0.3,
        };

        let report = solve_laminar_simple(&runtime, &fields, &options).expect("simple report");

        assert_eq!(report.cells, 2);
        assert!(report.simple_iterations > 0);
        assert!(report.solution.mean_velocity.is_finite());
        assert!(report.final_continuity.l2_norm.is_finite());
    }

    fn two_cell_runtime() -> SolverRuntimeData {
        SolverRuntimeData {
            mesh: SolverRuntimeMeshData {
                points: 0,
                cells: 2,
                faces: 3,
                internal_faces: 1,
                boundary_faces: 2,
                owner: vec![0, 0, 1],
                neighbour: vec![Some(1), None, None],
                patches: vec![
                    SolverRuntimePatchRange {
                        name: "inlet".to_string(),
                        patch_type: "patch".to_string(),
                        start_face: 1,
                        faces: 1,
                    },
                    SolverRuntimePatchRange {
                        name: "outlet".to_string(),
                        patch_type: "patch".to_string(),
                        start_face: 2,
                        faces: 1,
                    },
                ],
                face_centres: vec![
                    point(0.5, 0.0, 0.0),
                    point(0.0, 0.0, 0.0),
                    point(1.0, 0.0, 0.0),
                ],
                face_area_vectors: vec![
                    point(1.0, 0.0, 0.0),
                    point(-1.0, 0.0, 0.0),
                    point(1.0, 0.0, 0.0),
                ],
                cell_centres: vec![point(0.25, 0.0, 0.0), point(0.75, 0.0, 0.0)],
                cell_volumes: vec![0.5, 0.5],
                min_face_area: 1.0,
                max_face_area: 1.0,
                min_cell_volume: 0.5,
                max_cell_volume: 0.5,
                total_cell_volume: 1.0,
                non_positive_cell_volumes: 0,
            },
            fields: vec![
                SolverRuntimeFieldBuffer {
                    region: None,
                    name: "U".to_string(),
                    kind: SolverStateFieldKind::VolVector,
                    components: 3,
                    scalar_slots: 6,
                    bytes_f64: 48,
                    values: vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                },
                SolverRuntimeFieldBuffer {
                    region: None,
                    name: "p".to_string(),
                    kind: SolverStateFieldKind::VolScalar,
                    components: 1,
                    scalar_slots: 2,
                    bytes_f64: 16,
                    values: vec![1.0, 0.0],
                },
            ],
            warnings: Vec::new(),
        }
    }

    fn two_cell_fields() -> InitialFieldSet {
        InitialFieldSet {
            case_dir: PathBuf::from("case"),
            fields: vec![
                FieldFile {
                    path: PathBuf::from("case/0/U"),
                    region: None,
                    name: "U".to_string(),
                    class_name: Some("volVectorField".to_string()),
                    dimensions: None,
                    internal_field: None,
                    boundary_patches: vec![
                        FieldBoundaryPatch {
                            name: "inlet".to_string(),
                            patch_type: Some("fixedValue".to_string()),
                            value: Some(FieldValueSummary::Uniform("(1 0 0)".to_string())),
                        },
                        FieldBoundaryPatch {
                            name: "outlet".to_string(),
                            patch_type: Some("zeroGradient".to_string()),
                            value: None,
                        },
                    ],
                },
                FieldFile {
                    path: PathBuf::from("case/0/p"),
                    region: None,
                    name: "p".to_string(),
                    class_name: Some("volScalarField".to_string()),
                    dimensions: None,
                    internal_field: None,
                    boundary_patches: vec![
                        FieldBoundaryPatch {
                            name: "inlet".to_string(),
                            patch_type: Some("zeroGradient".to_string()),
                            value: None,
                        },
                        FieldBoundaryPatch {
                            name: "outlet".to_string(),
                            patch_type: Some("fixedValue".to_string()),
                            value: Some(FieldValueSummary::Uniform("0".to_string())),
                        },
                    ],
                },
            ],
        }
    }

    fn point(x: f64, y: f64, z: f64) -> Point3 {
        Point3 { x, y, z }
    }

    fn assert_close(left: f64, right: f64) {
        assert!(
            (left - right).abs() < 1.0e-12,
            "expected {left} to be close to {right}"
        );
    }
}
