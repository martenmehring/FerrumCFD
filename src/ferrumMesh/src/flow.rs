use std::collections::BTreeMap;

use crate::fields::{FieldFile, FieldValueSummary, InitialFieldSet};
use crate::linear::{
    BiCgStabOptions, CgPreconditioner, ConjugateGradientOptions, CsrMatrix, GaussSeidelOptions,
    JacobiOptions, PreconditionedConjugateGradientOptions, bicgstab_solve,
    conjugate_gradient_solve, gauss_seidel_solve, jacobi_solve, l2_norm,
    preconditioned_conjugate_gradient_solve, symmetric_gauss_seidel_solve,
};
use crate::runtime::{SolverRuntimeData, SolverRuntimeMeshData};
use crate::{MeshError, Point3, Result};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LaminarSimpleLinearSolver {
    BiCgStab,
    Cg,
    GaussSeidel,
    Jacobi,
    Pcg,
    SymGaussSeidel,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LaminarSimplePreconditioner {
    None,
    Diagonal,
    IncompleteCholesky,
}

#[derive(Clone, Copy, Debug)]
pub enum LaminarSimpleGradientScheme {
    GaussLinear,
    CellLimitedGaussLinear(f64),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LaminarSimpleConvectionScheme {
    GaussUpwind,
    GaussLinearUpwind,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LaminarSimpleInterpolationScheme {
    Linear,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LaminarSimpleSnGradScheme {
    Corrected,
    Orthogonal,
    Uncorrected,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LaminarSimpleLaplacianScheme {
    GaussLinearCorrected,
    GaussLinearOrthogonal,
    GaussLinearUncorrected,
}

impl PartialEq for LaminarSimpleGradientScheme {
    fn eq(&self, other: &Self) -> bool {
        match (*self, *other) {
            (Self::GaussLinear, Self::GaussLinear) => true,
            (Self::CellLimitedGaussLinear(left), Self::CellLimitedGaussLinear(right)) => {
                left.to_bits() == right.to_bits()
            }
            _ => false,
        }
    }
}

impl Eq for LaminarSimpleGradientScheme {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LaminarSimpleSchemes {
    pub grad_p: LaminarSimpleGradientScheme,
    pub grad_u: LaminarSimpleGradientScheme,
    pub div_phi_u: LaminarSimpleConvectionScheme,
    pub laplacian: LaminarSimpleLaplacianScheme,
    pub interpolation: LaminarSimpleInterpolationScheme,
    pub sn_grad: LaminarSimpleSnGradScheme,
}

#[derive(Clone, Debug)]
pub struct LaminarSimpleOptions {
    pub density: f64,
    pub dynamic_viscosity: f64,
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
    pub momentum_residual_control: Option<f64>,
    pub pressure_residual_control: Option<f64>,
    pub pressure_reference_cell: Option<usize>,
    pub pressure_reference_value: f64,
    pub non_orthogonal_correctors: usize,
    pub simple_consistent: bool,
    pub velocity_relaxation: f64,
    pub pressure_relaxation: f64,
    pub schemes: LaminarSimpleSchemes,
}

#[derive(Clone, Debug)]
pub struct LaminarSimpleReport {
    pub cells: usize,
    pub faces: usize,
    pub internal_faces: usize,
    pub boundary_faces: usize,
    pub simple_iterations: usize,
    pub converged: bool,
    pub stop_reason: LaminarSimpleStopReason,
    pub initial_continuity: ContinuitySummary,
    pub final_continuity: ContinuitySummary,
    pub final_momentum_initial_normalized_residual_norm: f64,
    pub final_momentum_residual_norm: f64,
    pub final_momentum_normalized_residual_norm: f64,
    pub final_pressure_correction_initial_normalized_residual_norm: f64,
    pub final_pressure_correction_residual_norm: f64,
    pub final_pressure_correction_normalized_residual_norm: f64,
    pub residual_control: LaminarSimpleResidualControlSummary,
    pub total_momentum_linear_iterations: usize,
    pub total_pressure_linear_iterations: usize,
    pub linear_solve_summary: LinearSolveSummary,
    pub operator_summary: FlowOperatorSummary,
    pub boundary_summary: FlowBoundarySummary,
    pub pressure_assembly: Option<PressureAssemblyDiagnostics>,
    pub fields: LaminarSimpleFieldSummary,
    pub final_velocity: Vec<Point3>,
    pub final_pressure: Vec<f64>,
    pub history: Vec<LaminarSimpleIterationSummary>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LaminarSimpleStopReason {
    Converged,
    MaxIterationsReached,
    ConvergenceCriteriaNotConfigured,
    MomentumSolverInvalidState,
    PressureSolverInvalidState,
    SolverInvalidState,
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
    pub hby_a_l2_norm: f64,
    pub div_phi_u_l2_norm: f64,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct FlowBoundarySummary {
    pub velocity_fixed_value_faces: usize,
    pub velocity_zero_gradient_faces: usize,
    pub velocity_inlet_outlet_faces: usize,
    pub velocity_constraint_faces: usize,
    pub pressure_fixed_value_faces: usize,
    pub pressure_zero_gradient_faces: usize,
    pub pressure_constraint_faces: usize,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ScalarDiagnosticSummary {
    pub min: f64,
    pub max: f64,
    pub l2_norm: f64,
    pub sum: f64,
    pub sum_abs: f64,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct VectorDiagnosticSummary {
    pub min_magnitude: f64,
    pub max_magnitude: f64,
    pub l2_norm: f64,
    pub x_min: f64,
    pub x_max: f64,
    pub y_min: f64,
    pub y_max: f64,
    pub z_min: f64,
    pub z_max: f64,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct FaceFluxDiagnosticSummary {
    pub min: f64,
    pub max: f64,
    pub l2_norm: f64,
    pub sum: f64,
    pub sum_abs: f64,
    pub internal_sum_abs: f64,
    pub boundary_sum: f64,
    pub boundary_sum_abs: f64,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct MatrixDiagnosticSummary {
    pub rows: usize,
    pub cols: usize,
    pub nonzeros: usize,
    pub diagonal_min: f64,
    pub diagonal_max: f64,
    pub diagonal_sum_abs: f64,
    pub off_diagonal_sum_abs: f64,
    pub max_row_sum_abs: f64,
    pub max_row_off_diagonal_sum_abs: f64,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct PressureAssemblyDiagnostics {
    pub r_au: ScalarDiagnosticSummary,
    pub r_at_u: ScalarDiagnosticSummary,
    pub hby_a: VectorDiagnosticSummary,
    pub phi_hby_a_before_adjust: FaceFluxDiagnosticSummary,
    pub phi_hby_a_after_adjust: FaceFluxDiagnosticSummary,
    pub pressure_source: ScalarDiagnosticSummary,
    pub pressure_equation_flux: FaceFluxDiagnosticSummary,
    pub pressure_matrix: MatrixDiagnosticSummary,
    pub pressure_flux: FaceFluxDiagnosticSummary,
    pub corrected_phi: FaceFluxDiagnosticSummary,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct LinearSolveSummary {
    pub momentum_predictors: usize,
    pub momentum_non_converged_predictors: usize,
    pub momentum_component_solves: usize,
    pub momentum_component_non_converged_solves: usize,
    pub pressure_correction_solves: usize,
    pub pressure_correction_non_converged_solves: usize,
    pub max_momentum_linear_iterations_per_simple: usize,
    pub max_pressure_linear_iterations_per_simple: usize,
    pub average_momentum_linear_iterations_per_simple: f64,
    pub average_pressure_linear_iterations_per_simple: f64,
    pub final_momentum_linear_converged: bool,
    pub final_pressure_linear_converged: bool,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct LaminarSimpleResidualControlSummary {
    pub configured: bool,
    pub checked: bool,
    pub satisfied: bool,
    pub momentum_satisfied: Option<bool>,
    pub pressure_satisfied: Option<bool>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct LaminarSimpleFieldSummary {
    pub velocity: VectorDiagnosticSummary,
    pub pressure: ScalarDiagnosticSummary,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct LaminarSimpleIterationSummary {
    pub iteration: usize,
    pub continuity_before: ContinuitySummary,
    pub continuity_after: ContinuitySummary,
    pub pressure_correction_accepted: bool,
    pub momentum_linear_iterations: usize,
    pub momentum_linear_converged: bool,
    pub momentum_component_linear_converged: [bool; 3],
    pub pressure_linear_iterations: usize,
    pub pressure_linear_converged: bool,
    pub pressure_linear_solves: usize,
    pub pressure_linear_non_converged_solves: usize,
    pub momentum_initial_normalized_residual_norm: f64,
    pub momentum_residual_norm: f64,
    pub momentum_normalized_residual_norm: f64,
    pub momentum_component_initial_normalized_residual_norms: [f64; 3],
    pub momentum_component_residual_norms: [f64; 3],
    pub momentum_component_normalized_residual_norms: [f64; 3],
    pub momentum_diagonal_min: f64,
    pub momentum_diagonal_max: f64,
    pub momentum_h1_min: f64,
    pub momentum_h1_max: f64,
    pub pressure_correction_initial_normalized_residual_norm: f64,
    pub pressure_correction_residual_norm: f64,
    pub pressure_correction_normalized_residual_norm: f64,
    pub residual_control: LaminarSimpleResidualControlSummary,
    pub relative_velocity_change_l2: f64,
    pub relative_pressure_change_l2: f64,
    pub momentum_update_scale: f64,
    pub pressure_correction_update_scale: f64,
    pub adjust_phi_global_flux_before: f64,
    pub adjust_phi_global_flux_after: f64,
    pub adjust_phi_adjusted_faces: usize,
}

#[derive(Clone, Debug)]
struct ScalarComponentSystem {
    matrix: CsrMatrix,
    rhs: Vec<f64>,
}

#[derive(Clone, Copy, Debug, Default)]
struct AdjustPhiSummary {
    global_flux_before: f64,
    global_flux_after: f64,
    adjusted_faces: usize,
}

#[derive(Clone, Copy, Debug)]
enum VectorFaceTreatment {
    FixedValue(Point3),
    InletOutlet(Point3),
    ZeroGradient,
    Constraint,
}

#[derive(Clone, Copy, Debug)]
enum ScalarFaceTreatment {
    FixedValue(f64),
    FixedGradient(f64),
    InletOutlet(f64),
    ZeroGradient,
    Constraint,
}

impl std::fmt::Display for LaminarSimpleLinearSolver {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BiCgStab => formatter.write_str("bicgstab"),
            Self::Cg => formatter.write_str("cg"),
            Self::GaussSeidel => formatter.write_str("gaussSeidel"),
            Self::Jacobi => formatter.write_str("jacobi"),
            Self::Pcg => formatter.write_str("pcg"),
            Self::SymGaussSeidel => formatter.write_str("symGaussSeidel"),
        }
    }
}

impl std::fmt::Display for LaminarSimplePreconditioner {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::None => formatter.write_str("none"),
            Self::Diagonal => formatter.write_str("diagonal"),
            Self::IncompleteCholesky => formatter.write_str("incompleteCholesky"),
        }
    }
}

impl Default for LaminarSimpleSchemes {
    fn default() -> Self {
        Self {
            grad_p: LaminarSimpleGradientScheme::GaussLinear,
            grad_u: LaminarSimpleGradientScheme::GaussLinear,
            div_phi_u: LaminarSimpleConvectionScheme::GaussUpwind,
            laplacian: LaminarSimpleLaplacianScheme::GaussLinearCorrected,
            interpolation: LaminarSimpleInterpolationScheme::Linear,
            sn_grad: LaminarSimpleSnGradScheme::Corrected,
        }
    }
}

impl LaminarSimpleLaplacianScheme {
    fn uses_non_orthogonal_correction(self) -> bool {
        matches!(self, Self::GaussLinearCorrected)
    }
}

impl std::fmt::Display for LaminarSimpleGradientScheme {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::GaussLinear => formatter.write_str("Gauss linear"),
            Self::CellLimitedGaussLinear(coefficient) => {
                write!(formatter, "cellLimited Gauss linear {coefficient}")
            }
        }
    }
}

impl std::fmt::Display for LaminarSimpleConvectionScheme {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::GaussUpwind => formatter.write_str("Gauss upwind"),
            Self::GaussLinearUpwind => formatter.write_str("Gauss linearUpwind grad(U)"),
        }
    }
}

impl std::fmt::Display for LaminarSimpleInterpolationScheme {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Linear => formatter.write_str("linear"),
        }
    }
}

impl std::fmt::Display for LaminarSimpleSnGradScheme {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Corrected => formatter.write_str("corrected"),
            Self::Orthogonal => formatter.write_str("orthogonal"),
            Self::Uncorrected => formatter.write_str("uncorrected"),
        }
    }
}

impl std::fmt::Display for LaminarSimpleLaplacianScheme {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::GaussLinearCorrected => formatter.write_str("Gauss linear corrected"),
            Self::GaussLinearOrthogonal => formatter.write_str("Gauss linear orthogonal"),
            Self::GaussLinearUncorrected => formatter.write_str("Gauss linear uncorrected"),
        }
    }
}

impl std::fmt::Display for LaminarSimpleStopReason {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Converged => formatter.write_str("Converged"),
            Self::MaxIterationsReached => formatter.write_str("MaxSimpleIterationsReached"),
            Self::ConvergenceCriteriaNotConfigured => {
                formatter.write_str("ConvergenceCriteriaNotConfigured")
            }
            Self::MomentumSolverInvalidState => formatter.write_str("MomentumSolverInvalidState"),
            Self::PressureSolverInvalidState => formatter.write_str("PressureSolverInvalidState"),
            Self::SolverInvalidState => formatter.write_str("SolverInvalidState"),
        }
    }
}

pub fn solve_laminar_simple(
    runtime: &SolverRuntimeData,
    fields: &InitialFieldSet,
    options: &LaminarSimpleOptions,
) -> Result<LaminarSimpleReport> {
    solve_laminar_simple_with_observer(runtime, fields, options, None)
}

pub fn solve_laminar_simple_with_observer(
    runtime: &SolverRuntimeData,
    fields: &InitialFieldSet,
    options: &LaminarSimpleOptions,
    mut on_iteration: Option<&mut dyn FnMut(&LaminarSimpleIterationSummary)>,
) -> Result<LaminarSimpleReport> {
    validate_laminar_simple_options(options)?;
    validate_runtime_mesh(&runtime.mesh)?;

    let velocity_field = find_field(fields, "U", "volVectorField")?;
    let pressure_field = find_field(fields, "p", "volScalarField")?;
    let velocity_boundary = vector_face_treatments(&runtime.mesh, velocity_field)?;
    let pressure_boundary = scalar_face_treatments(&runtime.mesh, pressure_field)?;
    let boundary_summary =
        summarize_boundaries(&runtime.mesh, &velocity_boundary, &pressure_boundary);

    let mut velocity = runtime_vector_field(runtime, "U")?;
    let mut pressure = runtime_scalar_field(runtime, "p")?;
    let initial_phi = compute_face_flux(&runtime.mesh, &velocity, &velocity_boundary)?;
    let initial_continuity = summarize_continuity(&net_cell_flux(&runtime.mesh, &initial_phi)?);

    let mut history = Vec::new();
    let mut converged = false;
    let mut final_continuity = initial_continuity;
    let mut final_momentum_initial_normalized_residual_norm = 0.0;
    let mut final_momentum_residual_norm = 0.0;
    let mut final_momentum_normalized_residual_norm = 0.0;
    let mut final_pressure_correction_initial_normalized_residual_norm = 0.0;
    let mut final_pressure_correction_residual_norm = 0.0;
    let mut final_pressure_correction_normalized_residual_norm = 0.0;
    let mut total_momentum_linear_iterations = 0;
    let mut total_pressure_linear_iterations = 0;
    let mut surface_flux = initial_phi;
    let mut final_phi = surface_flux.clone();
    let mut final_grad_p = vec![zero(); runtime.mesh.cells];
    let mut final_hby_a = vec![zero(); runtime.mesh.cells];
    let mut final_convection = vec![zero(); runtime.mesh.cells];
    let mut final_pressure_assembly = None;
    let mut stop_reason = None;
    let mut emit_iteration = |summary: LaminarSimpleIterationSummary| {
        history.push(summary);
        if let Some(observer) = on_iteration.as_deref_mut() {
            observer(&summary);
        }
    };

    for iteration in 1..=options.max_simple_iterations {
        let previous_velocity = velocity.clone();
        let previous_pressure = pressure.clone();
        let phi = surface_flux.clone();
        let continuity_before = summarize_continuity(&net_cell_flux(&runtime.mesh, &phi)?);
        let grad_p = scalar_gradient(
            &runtime.mesh,
            &pressure,
            &pressure_boundary,
            options.schemes.grad_p,
        )?;
        let convection = vector_convection_divergence(
            &runtime.mesh,
            &velocity,
            &velocity_boundary,
            &phi,
            options.schemes.div_phi_u,
            options.schemes.grad_u,
        )?;
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
            emit_iteration(LaminarSimpleIterationSummary {
                iteration,
                continuity_before,
                continuity_after: final_continuity,
                pressure_correction_accepted: false,
                momentum_linear_iterations: momentum.iterations,
                momentum_linear_converged: momentum.converged,
                momentum_component_linear_converged: momentum.component_converged,
                pressure_linear_iterations: 0,
                pressure_linear_converged: false,
                pressure_linear_solves: 0,
                pressure_linear_non_converged_solves: 0,
                momentum_initial_normalized_residual_norm: momentum
                    .initial_normalized_residual_norm,
                momentum_residual_norm: momentum.residual_norm,
                momentum_normalized_residual_norm: momentum.normalized_residual_norm,
                momentum_component_initial_normalized_residual_norms: momentum
                    .component_initial_normalized_residual_norms,
                momentum_component_residual_norms: momentum.component_residual_norms,
                momentum_component_normalized_residual_norms: momentum
                    .component_normalized_residual_norms,
                momentum_diagonal_min: momentum.diagonal_min,
                momentum_diagonal_max: momentum.diagonal_max,
                momentum_h1_min: momentum.h1_min,
                momentum_h1_max: momentum.h1_max,
                pressure_correction_initial_normalized_residual_norm: f64::NAN,
                pressure_correction_residual_norm: 0.0,
                pressure_correction_normalized_residual_norm: 0.0,
                residual_control: evaluate_laminar_simple_residual_control(
                    Some(momentum.initial_normalized_residual_norm),
                    None,
                    options,
                ),
                relative_velocity_change_l2: 0.0,
                relative_pressure_change_l2: 0.0,
                momentum_update_scale: 0.0,
                pressure_correction_update_scale: 0.0,
                adjust_phi_global_flux_before: 0.0,
                adjust_phi_global_flux_after: 0.0,
                adjust_phi_adjusted_faces: 0,
            });
            stop_reason = Some(LaminarSimpleStopReason::MomentumSolverInvalidState);
            break;
        }

        total_momentum_linear_iterations += momentum.iterations;
        final_momentum_initial_normalized_residual_norm = momentum.initial_normalized_residual_norm;
        final_momentum_residual_norm = momentum.residual_norm;
        final_momentum_normalized_residual_norm = momentum.normalized_residual_norm;
        let predicted_velocity = momentum.velocity.clone();
        let momentum_update_scale = 1.0;
        if !points_are_finite(&predicted_velocity) {
            final_phi = phi;
            final_continuity = continuity_before;
            final_grad_p = grad_p;
            final_convection = convection;
            emit_iteration(LaminarSimpleIterationSummary {
                iteration,
                continuity_before,
                continuity_after: final_continuity,
                pressure_correction_accepted: false,
                momentum_linear_iterations: momentum.iterations,
                momentum_linear_converged: momentum.converged,
                momentum_component_linear_converged: momentum.component_converged,
                pressure_linear_iterations: 0,
                pressure_linear_converged: false,
                pressure_linear_solves: 0,
                pressure_linear_non_converged_solves: 0,
                momentum_initial_normalized_residual_norm: momentum
                    .initial_normalized_residual_norm,
                momentum_residual_norm: momentum.residual_norm,
                momentum_normalized_residual_norm: momentum.normalized_residual_norm,
                momentum_component_initial_normalized_residual_norms: momentum
                    .component_initial_normalized_residual_norms,
                momentum_component_residual_norms: momentum.component_residual_norms,
                momentum_component_normalized_residual_norms: momentum
                    .component_normalized_residual_norms,
                momentum_diagonal_min: momentum.diagonal_min,
                momentum_diagonal_max: momentum.diagonal_max,
                momentum_h1_min: momentum.h1_min,
                momentum_h1_max: momentum.h1_max,
                pressure_correction_initial_normalized_residual_norm: f64::NAN,
                pressure_correction_residual_norm: 0.0,
                pressure_correction_normalized_residual_norm: 0.0,
                residual_control: evaluate_laminar_simple_residual_control(
                    Some(momentum.initial_normalized_residual_norm),
                    None,
                    options,
                ),
                relative_velocity_change_l2: 0.0,
                relative_pressure_change_l2: 0.0,
                momentum_update_scale,
                pressure_correction_update_scale: 0.0,
                adjust_phi_global_flux_before: 0.0,
                adjust_phi_global_flux_after: 0.0,
                adjust_phi_adjusted_faces: 0,
            });
            stop_reason = Some(LaminarSimpleStopReason::MomentumSolverInvalidState);
            break;
        }

        let r_au = reciprocal_momentum_diagonal(
            &runtime.mesh,
            &momentum.diagonal,
            options.velocity_relaxation,
        )?;
        let r_at_u = if options.simple_consistent {
            consistent_reciprocal_momentum_diagonal(&runtime.mesh, &r_au, &momentum.h1)?
        } else {
            r_au.clone()
        };
        let mut hby_a = hby_a_from_predicted_velocity(&predicted_velocity, &grad_p, &r_au)?;
        let mut phi_hby_a = compute_phi_hby_a(&runtime.mesh, &hby_a, &velocity_boundary)?;
        let phi_hby_a_before_adjust_summary = summarize_face_fluxes(&runtime.mesh, &phi_hby_a);
        if options.simple_consistent {
            let consistent_phi_correction = consistent_phi_hby_a_pressure_correction(
                &runtime.mesh,
                &pressure,
                &pressure_boundary,
                &r_au,
                &r_at_u,
            )?;
            phi_hby_a = add_face_fluxes(&phi_hby_a, &consistent_phi_correction)?;
            hby_a = consistent_hby_a_from_base(&hby_a, &grad_p, &r_au, &r_at_u)?;
        }
        let adjust_phi_summary = adjust_phi_hby_a(
            &runtime.mesh,
            &velocity_boundary,
            &pressure_boundary,
            &mut phi_hby_a,
        )?;
        let phi_hby_a_after_adjust_summary = summarize_face_fluxes(&runtime.mesh, &phi_hby_a);
        let net_flux_star = net_cell_flux(&runtime.mesh, &phi_hby_a)?;
        let continuity_star = summarize_continuity(&net_flux_star);
        if !is_finite_continuity(continuity_star) {
            final_phi = phi;
            final_continuity = continuity_before;
            final_grad_p = grad_p;
            final_hby_a = hby_a;
            final_convection = convection;
            emit_iteration(LaminarSimpleIterationSummary {
                iteration,
                continuity_before,
                continuity_after: final_continuity,
                pressure_correction_accepted: false,
                momentum_linear_iterations: momentum.iterations,
                momentum_linear_converged: momentum.converged,
                momentum_component_linear_converged: momentum.component_converged,
                pressure_linear_iterations: 0,
                pressure_linear_converged: false,
                pressure_linear_solves: 0,
                pressure_linear_non_converged_solves: 0,
                momentum_initial_normalized_residual_norm: momentum
                    .initial_normalized_residual_norm,
                momentum_residual_norm: momentum.residual_norm,
                momentum_normalized_residual_norm: momentum.normalized_residual_norm,
                momentum_component_initial_normalized_residual_norms: momentum
                    .component_initial_normalized_residual_norms,
                momentum_component_residual_norms: momentum.component_residual_norms,
                momentum_component_normalized_residual_norms: momentum
                    .component_normalized_residual_norms,
                momentum_diagonal_min: momentum.diagonal_min,
                momentum_diagonal_max: momentum.diagonal_max,
                momentum_h1_min: momentum.h1_min,
                momentum_h1_max: momentum.h1_max,
                pressure_correction_initial_normalized_residual_norm: f64::NAN,
                pressure_correction_residual_norm: 0.0,
                pressure_correction_normalized_residual_norm: 0.0,
                residual_control: evaluate_laminar_simple_residual_control(
                    Some(momentum.initial_normalized_residual_norm),
                    None,
                    options,
                ),
                relative_velocity_change_l2: 0.0,
                relative_pressure_change_l2: 0.0,
                momentum_update_scale,
                pressure_correction_update_scale: 0.0,
                adjust_phi_global_flux_before: adjust_phi_summary.global_flux_before,
                adjust_phi_global_flux_after: adjust_phi_summary.global_flux_after,
                adjust_phi_adjusted_faces: adjust_phi_summary.adjusted_faces,
            });
            stop_reason = Some(LaminarSimpleStopReason::SolverInvalidState);
            break;
        }
        let constrained_pressure_boundary = constrained_pressure_treatments(
            &runtime.mesh,
            &pressure_boundary,
            &velocity_boundary,
            &predicted_velocity,
            &phi_hby_a,
            &r_at_u,
        )?;
        let mut pressure_report = None;
        let mut first_pressure_initial_normalized_residual_norm = None;
        let mut pressure_guess = pressure.clone();
        let mut pressure_linear_iterations_this_simple = 0;
        let mut pressure_linear_converged_this_simple = true;
        let mut pressure_linear_solves_this_simple = 0;
        let mut pressure_linear_non_converged_solves_this_simple = 0;
        let mut pressure_source_summary = ScalarDiagnosticSummary::default();
        let mut pressure_equation_flux_summary = FaceFluxDiagnosticSummary::default();
        let mut pressure_matrix_summary = MatrixDiagnosticSummary::default();
        let apply_non_orthogonal_correction =
            options.schemes.laplacian.uses_non_orthogonal_correction();
        let pressure_solve_count = if apply_non_orthogonal_correction {
            options
                .non_orthogonal_correctors
                .checked_add(1)
                .ok_or_else(|| {
                    invalid_input(
                        "laminar SIMPLE nNonOrthogonalCorrectors overflows the pressure solve count"
                            .to_string(),
                    )
                })?
        } else {
            1
        };
        for _ in 0..pressure_solve_count {
            let pressure_equation_flux =
                if apply_non_orthogonal_correction && options.non_orthogonal_correctors > 0 {
                    let non_orthogonal_flux = non_orthogonal_pressure_flux_correction(
                        &runtime.mesh,
                        &pressure_guess,
                        &r_at_u,
                        &constrained_pressure_boundary,
                        options.schemes.grad_p,
                    )?;
                    add_face_fluxes(&phi_hby_a, &non_orthogonal_flux)?
                } else {
                    phi_hby_a.clone()
                };
            pressure_equation_flux_summary =
                summarize_face_fluxes(&runtime.mesh, &pressure_equation_flux);
            let pressure_source = pressure_correction_source(
                &runtime.mesh,
                &net_cell_flux(&runtime.mesh, &pressure_equation_flux)?,
            )?;
            pressure_source_summary = summarize_scalars(&pressure_source);
            let mut pressure_system = assemble_variable_scalar_component_system(
                &runtime.mesh,
                &r_at_u,
                &pressure_source,
                &constrained_pressure_boundary,
            )?;
            apply_pressure_reference(
                &mut pressure_system,
                &runtime.mesh,
                &constrained_pressure_boundary,
                options,
            )?;
            pressure_matrix_summary = summarize_csr_matrix(&pressure_system.matrix)?;
            let pressure_system_rhs_norm = l2_norm(&pressure_system.rhs);
            let initial_pressure = pressure_report
                .as_ref()
                .map(|report: &ScalarSolveReport| report.solution.as_slice())
                .unwrap_or(&pressure);
            pressure_linear_solves_this_simple += 1;
            let report = match solve_scalar_system(
                &pressure_system.matrix,
                &pressure_system.rhs,
                Some(initial_pressure),
                options.pressure_linear_solver,
                options.pressure_preconditioner,
                options.pressure_linear_tolerance,
                options.pressure_max_linear_iterations,
            ) {
                Ok(report) => report,
                Err(error) if is_pressure_correction_breakdown(&error) => {
                    pressure_linear_converged_this_simple = false;
                    pressure_linear_non_converged_solves_this_simple += 1;
                    velocity = predicted_velocity.clone();
                    final_pressure_correction_residual_norm = pressure_system_rhs_norm;
                    final_pressure_correction_normalized_residual_norm = normalized_residual_norm(
                        pressure_system_rhs_norm,
                        pressure_system_rhs_norm,
                    );
                    final_phi = phi_hby_a.clone();
                    final_continuity = continuity_star;
                    final_hby_a = hby_a.clone();
                    final_grad_p = scalar_gradient(
                        &runtime.mesh,
                        &pressure,
                        &constrained_pressure_boundary,
                        options.schemes.grad_p,
                    )?;
                    final_convection = vector_convection_divergence(
                        &runtime.mesh,
                        &velocity,
                        &velocity_boundary,
                        &final_phi,
                        options.schemes.div_phi_u,
                        options.schemes.grad_u,
                    )?;
                    emit_iteration(LaminarSimpleIterationSummary {
                        iteration,
                        continuity_before,
                        continuity_after: final_continuity,
                        pressure_correction_accepted: false,
                        momentum_linear_iterations: momentum.iterations,
                        momentum_linear_converged: momentum.converged,
                        momentum_component_linear_converged: momentum.component_converged,
                        pressure_linear_iterations: 0,
                        pressure_linear_converged: false,
                        pressure_linear_solves: pressure_linear_solves_this_simple,
                        pressure_linear_non_converged_solves:
                            pressure_linear_non_converged_solves_this_simple,
                        momentum_initial_normalized_residual_norm: momentum
                            .initial_normalized_residual_norm,
                        momentum_residual_norm: momentum.residual_norm,
                        momentum_normalized_residual_norm: momentum.normalized_residual_norm,
                        momentum_component_initial_normalized_residual_norms: momentum
                            .component_initial_normalized_residual_norms,
                        momentum_component_residual_norms: momentum.component_residual_norms,
                        momentum_component_normalized_residual_norms: momentum
                            .component_normalized_residual_norms,
                        momentum_diagonal_min: momentum.diagonal_min,
                        momentum_diagonal_max: momentum.diagonal_max,
                        momentum_h1_min: momentum.h1_min,
                        momentum_h1_max: momentum.h1_max,
                        pressure_correction_initial_normalized_residual_norm: f64::NAN,
                        pressure_correction_residual_norm: final_pressure_correction_residual_norm,
                        pressure_correction_normalized_residual_norm:
                            final_pressure_correction_normalized_residual_norm,
                        residual_control: evaluate_laminar_simple_residual_control(
                            Some(momentum.initial_normalized_residual_norm),
                            None,
                            options,
                        ),
                        relative_velocity_change_l2: relative_vector_field_change_l2(
                            &previous_velocity,
                            &velocity,
                        ),
                        relative_pressure_change_l2: 0.0,
                        momentum_update_scale,
                        pressure_correction_update_scale: 0.0,
                        adjust_phi_global_flux_before: adjust_phi_summary.global_flux_before,
                        adjust_phi_global_flux_after: adjust_phi_summary.global_flux_after,
                        adjust_phi_adjusted_faces: adjust_phi_summary.adjusted_faces,
                    });
                    stop_reason = Some(LaminarSimpleStopReason::PressureSolverInvalidState);
                    break;
                }
                Err(error) => {
                    return Err(invalid_input(format!(
                        "laminar SIMPLE pressure correction solve failed: {error}"
                    )));
                }
            };
            if !report.converged {
                pressure_linear_converged_this_simple = false;
                pressure_linear_non_converged_solves_this_simple += 1;
            }
            first_pressure_initial_normalized_residual_norm
                .get_or_insert(report.initial_normalized_residual_norm);
            total_pressure_linear_iterations += report.iterations;
            pressure_linear_iterations_this_simple += report.iterations;
            final_pressure_correction_residual_norm = report.residual_norm;
            final_pressure_correction_normalized_residual_norm = report.normalized_residual_norm;
            pressure_guess = report.solution.clone();
            pressure_report = Some(report);
        }
        let Some(pressure_report) = pressure_report else {
            break;
        };
        let pressure_initial_normalized_residual_norm =
            first_pressure_initial_normalized_residual_norm
                .unwrap_or(pressure_report.initial_normalized_residual_norm);
        final_pressure_correction_initial_normalized_residual_norm =
            pressure_initial_normalized_residual_norm;
        let pressure_delta = pressure_report
            .solution
            .iter()
            .zip(&pressure)
            .map(|(after, before)| options.pressure_relaxation * (after - before))
            .collect::<Vec<_>>();
        let mut corrected_pressure = pressure.clone();
        for (value, delta) in corrected_pressure.iter_mut().zip(&pressure_delta) {
            *value += delta;
        }
        let corrected_pressure_gradient = scalar_gradient(
            &runtime.mesh,
            &corrected_pressure,
            &constrained_pressure_boundary,
            options.schemes.grad_p,
        )?;
        let corrected_velocity =
            velocity_from_hby_a(&runtime.mesh, &hby_a, &corrected_pressure_gradient, &r_at_u)?;

        let pressure_flux = pressure_equation_flux(
            &runtime.mesh,
            &pressure_report.solution,
            &r_at_u,
            &constrained_pressure_boundary,
        )?;
        let corrected_phi = subtract_face_fluxes(&phi_hby_a, &pressure_flux)?;
        final_pressure_assembly = Some(PressureAssemblyDiagnostics {
            r_au: summarize_scalars(&r_au),
            r_at_u: summarize_scalars(&r_at_u),
            hby_a: summarize_vectors(&hby_a),
            phi_hby_a_before_adjust: phi_hby_a_before_adjust_summary,
            phi_hby_a_after_adjust: phi_hby_a_after_adjust_summary,
            pressure_source: pressure_source_summary,
            pressure_equation_flux: pressure_equation_flux_summary,
            pressure_matrix: pressure_matrix_summary,
            pressure_flux: summarize_face_fluxes(&runtime.mesh, &pressure_flux),
            corrected_phi: summarize_face_fluxes(&runtime.mesh, &corrected_phi),
        });
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
            final_hby_a = hby_a;
            final_grad_p = scalar_gradient(
                &runtime.mesh,
                &pressure,
                &pressure_boundary,
                options.schemes.grad_p,
            )?;
            final_convection = vector_convection_divergence(
                &runtime.mesh,
                &velocity,
                &velocity_boundary,
                &final_phi,
                options.schemes.div_phi_u,
                options.schemes.grad_u,
            )?;
            emit_iteration(LaminarSimpleIterationSummary {
                iteration,
                continuity_before,
                continuity_after: final_continuity,
                pressure_correction_accepted: false,
                momentum_linear_iterations: momentum.iterations,
                momentum_linear_converged: momentum.converged,
                momentum_component_linear_converged: momentum.component_converged,
                pressure_linear_iterations: pressure_linear_iterations_this_simple,
                pressure_linear_converged: pressure_linear_converged_this_simple,
                pressure_linear_solves: pressure_linear_solves_this_simple,
                pressure_linear_non_converged_solves:
                    pressure_linear_non_converged_solves_this_simple,
                momentum_initial_normalized_residual_norm: momentum
                    .initial_normalized_residual_norm,
                momentum_residual_norm: momentum.residual_norm,
                momentum_normalized_residual_norm: momentum.normalized_residual_norm,
                momentum_component_initial_normalized_residual_norms: momentum
                    .component_initial_normalized_residual_norms,
                momentum_component_residual_norms: momentum.component_residual_norms,
                momentum_component_normalized_residual_norms: momentum
                    .component_normalized_residual_norms,
                momentum_diagonal_min: momentum.diagonal_min,
                momentum_diagonal_max: momentum.diagonal_max,
                momentum_h1_min: momentum.h1_min,
                momentum_h1_max: momentum.h1_max,
                pressure_correction_initial_normalized_residual_norm:
                    pressure_initial_normalized_residual_norm,
                pressure_correction_residual_norm: pressure_report.residual_norm,
                pressure_correction_normalized_residual_norm: pressure_report
                    .normalized_residual_norm,
                residual_control: evaluate_laminar_simple_residual_control(
                    Some(momentum.initial_normalized_residual_norm),
                    Some(pressure_initial_normalized_residual_norm),
                    options,
                ),
                relative_velocity_change_l2: relative_vector_field_change_l2(
                    &previous_velocity,
                    &velocity,
                ),
                relative_pressure_change_l2: 0.0,
                momentum_update_scale,
                pressure_correction_update_scale: 0.0,
                adjust_phi_global_flux_before: adjust_phi_summary.global_flux_before,
                adjust_phi_global_flux_after: adjust_phi_summary.global_flux_after,
                adjust_phi_adjusted_faces: adjust_phi_summary.adjusted_faces,
            });
            stop_reason = Some(LaminarSimpleStopReason::SolverInvalidState);
            break;
        }

        velocity = corrected_velocity;
        pressure = corrected_pressure;
        final_phi = corrected_phi;
        surface_flux = final_phi.clone();
        final_continuity = corrected_continuity;
        final_hby_a = hby_a;

        final_grad_p = scalar_gradient(
            &runtime.mesh,
            &pressure,
            &constrained_pressure_boundary,
            options.schemes.grad_p,
        )?;
        final_convection = vector_convection_divergence(
            &runtime.mesh,
            &velocity,
            &velocity_boundary,
            &final_phi,
            options.schemes.div_phi_u,
            options.schemes.grad_u,
        )?;

        let relative_velocity_change_l2 =
            relative_vector_field_change_l2(&previous_velocity, &velocity);
        let relative_pressure_change_l2 =
            relative_scalar_field_change_l2(&previous_pressure, &pressure);
        let residual_control = evaluate_laminar_simple_residual_control(
            Some(momentum.initial_normalized_residual_norm),
            Some(pressure_initial_normalized_residual_norm),
            options,
        );

        emit_iteration(LaminarSimpleIterationSummary {
            iteration,
            continuity_before,
            continuity_after: final_continuity,
            pressure_correction_accepted: true,
            momentum_linear_iterations: momentum.iterations,
            momentum_linear_converged: momentum.converged,
            momentum_component_linear_converged: momentum.component_converged,
            pressure_linear_iterations: pressure_linear_iterations_this_simple,
            pressure_linear_converged: pressure_linear_converged_this_simple,
            pressure_linear_solves: pressure_linear_solves_this_simple,
            pressure_linear_non_converged_solves: pressure_linear_non_converged_solves_this_simple,
            momentum_initial_normalized_residual_norm: momentum.initial_normalized_residual_norm,
            momentum_residual_norm: momentum.residual_norm,
            momentum_normalized_residual_norm: momentum.normalized_residual_norm,
            momentum_component_initial_normalized_residual_norms: momentum
                .component_initial_normalized_residual_norms,
            momentum_component_residual_norms: momentum.component_residual_norms,
            momentum_component_normalized_residual_norms: momentum
                .component_normalized_residual_norms,
            momentum_diagonal_min: momentum.diagonal_min,
            momentum_diagonal_max: momentum.diagonal_max,
            momentum_h1_min: momentum.h1_min,
            momentum_h1_max: momentum.h1_max,
            pressure_correction_initial_normalized_residual_norm:
                pressure_initial_normalized_residual_norm,
            pressure_correction_residual_norm: pressure_report.residual_norm,
            pressure_correction_normalized_residual_norm: pressure_report.normalized_residual_norm,
            residual_control,
            relative_velocity_change_l2,
            relative_pressure_change_l2,
            momentum_update_scale,
            pressure_correction_update_scale,
            adjust_phi_global_flux_before: adjust_phi_summary.global_flux_before,
            adjust_phi_global_flux_after: adjust_phi_summary.global_flux_after,
            adjust_phi_adjusted_faces: adjust_phi_summary.adjusted_faces,
        });

        if laminar_simple_converged(iteration, residual_control, options) {
            converged = true;
            stop_reason = Some(LaminarSimpleStopReason::Converged);
            break;
        }
    }

    let stop_reason = stop_reason.unwrap_or_else(|| {
        if !converged
            && options.momentum_residual_control.is_none()
            && options.pressure_residual_control.is_none()
        {
            LaminarSimpleStopReason::ConvergenceCriteriaNotConfigured
        } else {
            LaminarSimpleStopReason::MaxIterationsReached
        }
    });

    let operator_summary =
        summarize_operators(&final_phi, &final_grad_p, &final_hby_a, &final_convection);
    let fields = LaminarSimpleFieldSummary {
        velocity: summarize_vectors(&velocity),
        pressure: summarize_scalars(&pressure),
    };

    Ok(LaminarSimpleReport {
        cells: runtime.mesh.cells,
        faces: runtime.mesh.faces,
        internal_faces: runtime.mesh.internal_faces,
        boundary_faces: runtime.mesh.boundary_faces,
        simple_iterations: history.len(),
        converged,
        stop_reason,
        initial_continuity,
        final_continuity,
        final_momentum_initial_normalized_residual_norm,
        final_momentum_residual_norm,
        final_momentum_normalized_residual_norm,
        final_pressure_correction_initial_normalized_residual_norm,
        final_pressure_correction_residual_norm,
        final_pressure_correction_normalized_residual_norm,
        residual_control: history
            .last()
            .map(|item| item.residual_control)
            .unwrap_or_default(),
        total_momentum_linear_iterations,
        total_pressure_linear_iterations,
        linear_solve_summary: summarize_linear_solves(&history),
        operator_summary,
        boundary_summary,
        pressure_assembly: final_pressure_assembly,
        fields,
        final_velocity: velocity,
        final_pressure: pressure,
        history,
    })
}

struct MomentumPredictorReport {
    velocity: Vec<Point3>,
    diagonal: Vec<f64>,
    h1: Vec<f64>,
    iterations: usize,
    converged: bool,
    initial_normalized_residual_norm: f64,
    residual_norm: f64,
    normalized_residual_norm: f64,
    component_initial_normalized_residual_norms: [f64; 3],
    component_residual_norms: [f64; 3],
    component_normalized_residual_norms: [f64; 3],
    component_converged: [bool; 3],
    diagonal_min: f64,
    diagonal_max: f64,
    h1_min: f64,
    h1_max: f64,
}

struct ScalarSolveReport {
    solution: Vec<f64>,
    iterations: usize,
    converged: bool,
    initial_normalized_residual_norm: f64,
    residual_norm: f64,
    normalized_residual_norm: f64,
}

struct MomentumEquation {
    components: Vec<ScalarComponentSystem>,
    diagonal: Vec<f64>,
    h1: Vec<f64>,
    diagonal_min: f64,
    diagonal_max: f64,
    h1_min: f64,
    h1_max: f64,
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
    let equation = assemble_momentum_equation(
        mesh,
        velocity_boundary,
        flux,
        grad_p,
        &old_components,
        options,
    )?;
    let mut solved_components = [Vec::new(), Vec::new(), Vec::new()];
    let mut total_iterations = 0;
    let mut residual_squared_sum = 0.0;
    let mut component_initial_normalized_residual_norms = [0.0; 3];
    let mut component_residual_norms = [0.0; 3];
    let mut component_normalized_residual_norms = [0.0; 3];
    let mut component_converged = [false; 3];

    for (component, system) in equation.components.iter().enumerate() {
        if component >= 3 {
            return Err(invalid_input(format!(
                "laminar SIMPLE momentum equation has unexpected component index {component}"
            )));
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
        component_initial_normalized_residual_norms[component] =
            report.initial_normalized_residual_norm;
        component_residual_norms[component] = report.residual_norm;
        component_normalized_residual_norms[component] = report.normalized_residual_norm;
        component_converged[component] = report.converged;
        solved_components[component] = report.solution;
    }

    let solved_velocity = combine_components(
        &solved_components[0],
        &solved_components[1],
        &solved_components[2],
    );

    Ok(MomentumPredictorReport {
        velocity: solved_velocity,
        diagonal: equation.diagonal,
        h1: equation.h1,
        iterations: total_iterations,
        converged: component_converged.iter().all(|value| *value),
        initial_normalized_residual_norm: component_initial_normalized_residual_norms
            .iter()
            .copied()
            .fold(0.0, f64::max),
        residual_norm: residual_squared_sum.sqrt(),
        normalized_residual_norm: component_normalized_residual_norms
            .iter()
            .copied()
            .fold(0.0, f64::max),
        component_initial_normalized_residual_norms,
        component_residual_norms,
        component_normalized_residual_norms,
        component_converged,
        diagonal_min: equation.diagonal_min,
        diagonal_max: equation.diagonal_max,
        h1_min: equation.h1_min,
        h1_max: equation.h1_max,
    })
}

fn assemble_momentum_equation(
    mesh: &SolverRuntimeMeshData,
    velocity_boundary: &[VectorFaceTreatment],
    flux: &[f64],
    grad_p: &[Point3],
    old_components: &[Vec<f64>; 3],
    options: &LaminarSimpleOptions,
) -> Result<MomentumEquation> {
    let mut components = Vec::with_capacity(3);
    let mut diagonal = Vec::new();
    let mut h1 = Vec::new();
    let component_gradients = if matches!(
        options.schemes.div_phi_u,
        LaminarSimpleConvectionScheme::GaussLinearUpwind
    ) {
        Some([
            scalar_gradient(
                mesh,
                &old_components[0],
                &scalar_component_boundary(velocity_boundary, 0),
                options.schemes.grad_u,
            )?,
            scalar_gradient(
                mesh,
                &old_components[1],
                &scalar_component_boundary(velocity_boundary, 1),
                options.schemes.grad_u,
            )?,
            scalar_gradient(
                mesh,
                &old_components[2],
                &scalar_component_boundary(velocity_boundary, 2),
                options.schemes.grad_u,
            )?,
        ])
    } else {
        None
    };

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
            &old_components[component],
            component_gradients
                .as_ref()
                .map(|gradients| gradients[component].as_slice()),
            options.schemes.div_phi_u,
        )?;
        let component_diagonal = relax_scalar_component_equation(
            &mut system,
            &old_components[component],
            options.velocity_relaxation,
        )?;
        if component == 0 {
            diagonal = component_diagonal;
            h1 = momentum_h1_from_matrix(mesh, &system.matrix)?;
        }
        components.push(system);
    }

    let (diagonal_min, diagonal_max) = coefficient_range(&diagonal);
    let (h1_min, h1_max) = coefficient_range(&h1);

    Ok(MomentumEquation {
        components,
        diagonal,
        h1,
        diagonal_min,
        diagonal_max,
        h1_min,
        h1_max,
    })
}

fn momentum_h1_from_matrix(mesh: &SolverRuntimeMeshData, matrix: &CsrMatrix) -> Result<Vec<f64>> {
    if matrix.rows() != mesh.cells {
        return Err(invalid_input(format!(
            "momentum H1 expected {} matrix rows, got {}",
            mesh.cells,
            matrix.rows()
        )));
    }
    let mut h1 = vec![0.0; mesh.cells];
    for (row, h1_value) in h1.iter_mut().enumerate() {
        let volume = mesh.cell_volumes[row];
        if !volume.is_finite() || volume <= f64::EPSILON {
            return Err(invalid_input(format!(
                "momentum H1 cell {row} has non-positive volume {volume}"
            )));
        }
        let start = matrix.row_offsets()[row];
        let end = matrix.row_offsets()[row + 1];
        let mut off_diagonal_sum = 0.0;
        for entry in start..end {
            let column = matrix.col_indices()[entry];
            let coefficient = matrix.values()[entry];
            if column != row && coefficient.is_finite() && coefficient < 0.0 {
                off_diagonal_sum += -coefficient;
            }
        }
        *h1_value = off_diagonal_sum / volume;
    }
    Ok(h1)
}

fn coefficient_range(values: &[f64]) -> (f64, f64) {
    let mut minimum = f64::INFINITY;
    let mut maximum = f64::NEG_INFINITY;
    for value in values.iter().copied().filter(|value| value.is_finite()) {
        minimum = minimum.min(value);
        maximum = maximum.max(value);
    }
    if minimum.is_finite() && maximum.is_finite() {
        (minimum, maximum)
    } else {
        (0.0, 0.0)
    }
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
    let zero_initial = vec![0.0; rhs.len()];
    let initial_values = initial.unwrap_or(&zero_initial);
    let initial_ax = matrix.matvec(initial_values)?;
    let normalisation_factor =
        openfoam_normalisation_factor(matrix, rhs, initial_values, &initial_ax)?;
    let initial_residual = rhs
        .iter()
        .zip(&initial_ax)
        .map(|(source, matrix_value)| source - matrix_value)
        .collect::<Vec<_>>();
    let initial_residual_norm = l2_norm(&initial_residual);
    let initial_normalized_residual_norm = l1_norm(&initial_residual) / normalisation_factor;

    if initial_normalized_residual_norm < tolerance {
        return Ok(ScalarSolveReport {
            solution: initial_values.to_vec(),
            iterations: 0,
            converged: true,
            initial_normalized_residual_norm,
            residual_norm: initial_residual_norm,
            normalized_residual_norm: initial_normalized_residual_norm,
        });
    }

    // Ferrum's current CSR kernels stop on L2. This conservative conversion
    // guarantees the OpenFOAM L1-normalised tolerance before reporting success.
    let component_count = rhs.len().max(1) as f64;
    let solver_tolerance = tolerance * normalisation_factor / component_count.sqrt();
    let report = match solver {
        LaminarSimpleLinearSolver::BiCgStab => bicgstab_solve(
            matrix,
            rhs,
            initial,
            BiCgStabOptions {
                max_iterations,
                tolerance: solver_tolerance,
                preconditioner: map_cg_preconditioner(preconditioner),
            },
        )?,
        LaminarSimpleLinearSolver::Cg => conjugate_gradient_solve(
            matrix,
            rhs,
            initial,
            ConjugateGradientOptions {
                max_iterations,
                tolerance: solver_tolerance,
            },
        )?,
        LaminarSimpleLinearSolver::GaussSeidel => gauss_seidel_solve(
            matrix,
            rhs,
            initial,
            GaussSeidelOptions {
                max_iterations,
                tolerance: solver_tolerance,
                omega: 1.0,
            },
        )?,
        LaminarSimpleLinearSolver::SymGaussSeidel => symmetric_gauss_seidel_solve(
            matrix,
            rhs,
            initial,
            GaussSeidelOptions {
                max_iterations,
                tolerance: solver_tolerance,
                omega: 1.0,
            },
        )?,
        LaminarSimpleLinearSolver::Pcg => preconditioned_conjugate_gradient_solve(
            matrix,
            rhs,
            initial,
            PreconditionedConjugateGradientOptions {
                max_iterations,
                tolerance: solver_tolerance,
                preconditioner: map_cg_preconditioner(preconditioner),
            },
        )?,
        LaminarSimpleLinearSolver::Jacobi => jacobi_solve(
            matrix,
            rhs,
            initial,
            JacobiOptions {
                max_iterations,
                tolerance: solver_tolerance,
                omega: 1.0,
            },
        )?,
    };
    let final_ax = matrix.matvec(&report.solution)?;
    let final_residual = rhs
        .iter()
        .zip(final_ax)
        .map(|(source, matrix_value)| source - matrix_value)
        .collect::<Vec<_>>();
    let final_normalized_residual_norm = l1_norm(&final_residual) / normalisation_factor;
    Ok(ScalarSolveReport {
        solution: report.solution,
        iterations: report.iterations,
        converged: final_normalized_residual_norm < tolerance,
        initial_normalized_residual_norm,
        residual_norm: report.residual_norm,
        normalized_residual_norm: final_normalized_residual_norm,
    })
}

fn openfoam_normalisation_factor(
    matrix: &CsrMatrix,
    source: &[f64],
    solution: &[f64],
    matrix_solution: &[f64],
) -> Result<f64> {
    if source.len() != matrix.rows() || matrix_solution.len() != matrix.rows() {
        return Err(invalid_input(format!(
            "OpenFOAM residual normalisation expected {} source and matrix-product entries, got {} and {}",
            matrix.rows(),
            source.len(),
            matrix_solution.len()
        )));
    }
    if solution.len() != matrix.cols() {
        return Err(invalid_input(format!(
            "OpenFOAM residual normalisation expected {} solution entries, got {}",
            matrix.cols(),
            solution.len()
        )));
    }

    let average_solution = if solution.is_empty() {
        0.0
    } else {
        solution.iter().sum::<f64>() / solution.len() as f64
    };
    let mut factor = 0.0;
    for row in 0..matrix.rows() {
        let start = matrix.row_offsets()[row];
        let end = matrix.row_offsets()[row + 1];
        let row_sum = matrix.values()[start..end].iter().sum::<f64>();
        let reference = row_sum * average_solution;
        factor += (matrix_solution[row] - reference).abs() + (source[row] - reference).abs();
    }

    if !factor.is_finite() {
        return Err(invalid_input(
            "OpenFOAM residual normalisation factor is not finite".to_string(),
        ));
    }
    Ok(factor.max(1.0e-20))
}

fn l1_norm(values: &[f64]) -> f64 {
    values.iter().map(|value| value.abs()).sum()
}

fn normalized_residual_norm(residual_norm: f64, reference_norm: f64) -> f64 {
    if reference_norm.is_finite() && reference_norm > f64::EPSILON {
        residual_norm / reference_norm
    } else {
        residual_norm
    }
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

fn apply_pressure_reference(
    system: &mut ScalarComponentSystem,
    mesh: &SolverRuntimeMeshData,
    pressure_boundary: &[ScalarFaceTreatment],
    options: &LaminarSimpleOptions,
) -> Result<()> {
    let pressure_needs_reference = !pressure_boundary
        .iter()
        .any(|treatment| matches!(treatment, ScalarFaceTreatment::FixedValue(_)));
    if !pressure_needs_reference && options.pressure_reference_cell.is_none() {
        return Ok(());
    }

    let reference_cell = options.pressure_reference_cell.unwrap_or(0);
    if reference_cell >= mesh.cells {
        return Err(invalid_input(format!(
            "laminar SIMPLE pRefCell {reference_cell} exceeds mesh cell count {}",
            mesh.cells
        )));
    }
    let reference_value = options.pressure_reference_value;
    let mut rows = csr_rows(&system.matrix);
    for (row_index, row) in rows.iter_mut().enumerate() {
        if row_index == reference_cell {
            row.clear();
            row.push((reference_cell, 1.0));
            system.rhs[row_index] = reference_value;
            continue;
        }
        if let Some(position) = row.iter().position(|(column, _)| *column == reference_cell) {
            let (_, coefficient) = row.remove(position);
            system.rhs[row_index] -= coefficient * reference_value;
        }
    }
    system.matrix = CsrMatrix::from_rows(rows, system.matrix.cols())?;
    Ok(())
}

fn csr_rows(matrix: &CsrMatrix) -> Vec<Vec<(usize, f64)>> {
    let mut rows = Vec::with_capacity(matrix.rows());
    for row in 0..matrix.rows() {
        let start = matrix.row_offsets()[row];
        let end = matrix.row_offsets()[row + 1];
        rows.push(
            (start..end)
                .map(|entry| (matrix.col_indices()[entry], matrix.values()[entry]))
                .collect(),
        );
    }
    rows
}

fn map_cg_preconditioner(preconditioner: LaminarSimplePreconditioner) -> CgPreconditioner {
    match preconditioner {
        LaminarSimplePreconditioner::None => CgPreconditioner::None,
        LaminarSimplePreconditioner::Diagonal => CgPreconditioner::Diagonal,
        LaminarSimplePreconditioner::IncompleteCholesky => CgPreconditioner::IncompleteCholesky,
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

fn summarize_linear_solves(history: &[LaminarSimpleIterationSummary]) -> LinearSolveSummary {
    let momentum_predictors = history.len();
    let pressure_correction_solves = history
        .iter()
        .map(|item| item.pressure_linear_solves)
        .sum::<usize>();
    let total_momentum_iterations = history
        .iter()
        .map(|item| item.momentum_linear_iterations)
        .sum::<usize>();
    let total_pressure_iterations = history
        .iter()
        .map(|item| item.pressure_linear_iterations)
        .sum::<usize>();

    LinearSolveSummary {
        momentum_predictors,
        momentum_non_converged_predictors: history
            .iter()
            .filter(|item| !item.momentum_linear_converged)
            .count(),
        momentum_component_solves: momentum_predictors * 3,
        momentum_component_non_converged_solves: history
            .iter()
            .flat_map(|item| item.momentum_component_linear_converged)
            .filter(|converged| !*converged)
            .count(),
        pressure_correction_solves,
        pressure_correction_non_converged_solves: history
            .iter()
            .map(|item| item.pressure_linear_non_converged_solves)
            .sum(),
        max_momentum_linear_iterations_per_simple: history
            .iter()
            .map(|item| item.momentum_linear_iterations)
            .max()
            .unwrap_or(0),
        max_pressure_linear_iterations_per_simple: history
            .iter()
            .map(|item| item.pressure_linear_iterations)
            .max()
            .unwrap_or(0),
        average_momentum_linear_iterations_per_simple: if momentum_predictors > 0 {
            total_momentum_iterations as f64 / momentum_predictors as f64
        } else {
            0.0
        },
        average_pressure_linear_iterations_per_simple: if momentum_predictors > 0 {
            total_pressure_iterations as f64 / momentum_predictors as f64
        } else {
            0.0
        },
        final_momentum_linear_converged: history
            .last()
            .is_some_and(|item| item.momentum_linear_converged),
        final_pressure_linear_converged: history
            .last()
            .is_some_and(|item| item.pressure_linear_converged),
    }
}

fn laminar_simple_converged(
    iteration: usize,
    residual_control: LaminarSimpleResidualControlSummary,
    options: &LaminarSimpleOptions,
) -> bool {
    if iteration < options.min_simple_iterations {
        return false;
    }

    residual_control.checked && residual_control.satisfied
}

fn evaluate_laminar_simple_residual_control(
    momentum_initial_residual: Option<f64>,
    pressure_initial_residual: Option<f64>,
    options: &LaminarSimpleOptions,
) -> LaminarSimpleResidualControlSummary {
    let momentum_satisfied = options.momentum_residual_control.map(|tolerance| {
        momentum_initial_residual
            .is_some_and(|residual| residual.is_finite() && residual < tolerance)
    });
    let pressure_satisfied = options.pressure_residual_control.map(|tolerance| {
        pressure_initial_residual
            .is_some_and(|residual| residual.is_finite() && residual < tolerance)
    });
    let configured =
        options.momentum_residual_control.is_some() || options.pressure_residual_control.is_some();
    let checked = (options.momentum_residual_control.is_some()
        && momentum_initial_residual.is_some())
        || (options.pressure_residual_control.is_some() && pressure_initial_residual.is_some());
    let satisfied = configured
        && options
            .momentum_residual_control
            .is_none_or(|_| momentum_satisfied == Some(true))
        && options
            .pressure_residual_control
            .is_none_or(|_| pressure_satisfied == Some(true));

    LaminarSimpleResidualControlSummary {
        configured,
        checked,
        satisfied,
        momentum_satisfied,
        pressure_satisfied,
    }
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

fn component_name(component: usize) -> &'static str {
    match component {
        0 => "Ux",
        1 => "Uy",
        2 => "Uz",
        _ => "?",
    }
}

#[allow(clippy::too_many_arguments)]
fn assemble_momentum_component_system(
    mesh: &SolverRuntimeMeshData,
    diffusivity: f64,
    density: f64,
    flux: &[f64],
    volumetric_source: &[f64],
    boundary: &[ScalarFaceTreatment],
    old_values: &[f64],
    old_gradient: Option<&[Point3]>,
    convection_scheme: LaminarSimpleConvectionScheme,
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
    if old_values.len() != mesh.cells {
        return Err(invalid_input(format!(
            "momentum component old values have {} values, expected {} mesh cells",
            old_values.len(),
            mesh.cells
        )));
    }
    if let Some(gradient) = old_gradient
        && gradient.len() != mesh.cells
    {
        return Err(invalid_input(format!(
            "momentum component old gradient has {} values, expected {} mesh cells",
            gradient.len(),
            mesh.cells
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
            add_internal_convection(
                &mut rows,
                &mut rhs,
                mesh,
                old_values,
                old_gradient,
                owner,
                neighbour,
                face_index,
                mass_flux,
                convection_scheme,
            );
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
                add_boundary_convection(
                    &mut rows,
                    &mut rhs,
                    mesh,
                    old_values,
                    old_gradient,
                    owner,
                    face_index,
                    value,
                    mass_flux,
                    convection_scheme,
                );
            }
            ScalarFaceTreatment::InletOutlet(value) if mass_flux < 0.0 => {
                let coefficient = face_diffusion_coefficient(
                    diffusivity,
                    mesh.face_area_vectors[face_index],
                    mesh.cell_centres[owner],
                    mesh.face_centres[face_index],
                    face_index,
                )?;
                add_entry(&mut rows[owner], owner, coefficient);
                rhs[owner] += coefficient * value;
                add_boundary_convection(
                    &mut rows,
                    &mut rhs,
                    mesh,
                    old_values,
                    old_gradient,
                    owner,
                    face_index,
                    value,
                    mass_flux,
                    convection_scheme,
                );
            }
            ScalarFaceTreatment::FixedGradient(_)
            | ScalarFaceTreatment::InletOutlet(_)
            | ScalarFaceTreatment::ZeroGradient
            | ScalarFaceTreatment::Constraint => {
                add_boundary_extrapolated_convection(
                    &mut rows,
                    &mut rhs,
                    mesh,
                    old_values,
                    old_gradient,
                    owner,
                    face_index,
                    mass_flux,
                    convection_scheme,
                );
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

#[allow(clippy::too_many_arguments)]
fn add_internal_convection(
    rows: &mut [BTreeMap<usize, f64>],
    rhs: &mut [f64],
    mesh: &SolverRuntimeMeshData,
    old_values: &[f64],
    old_gradient: Option<&[Point3]>,
    owner: usize,
    neighbour: usize,
    face_index: usize,
    mass_flux: f64,
    scheme: LaminarSimpleConvectionScheme,
) {
    add_internal_upwind_convection(rows, owner, neighbour, mass_flux);
    if matches!(scheme, LaminarSimpleConvectionScheme::GaussLinearUpwind) {
        let Some(gradient) = old_gradient else {
            return;
        };
        let correction = linear_upwind_scalar_correction(
            mesh,
            old_values,
            gradient,
            owner,
            Some(neighbour),
            face_index,
            mass_flux,
        );
        rhs[owner] -= mass_flux * correction;
        rhs[neighbour] += mass_flux * correction;
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

#[allow(clippy::too_many_arguments)]
fn add_boundary_convection(
    rows: &mut [BTreeMap<usize, f64>],
    rhs: &mut [f64],
    mesh: &SolverRuntimeMeshData,
    old_values: &[f64],
    old_gradient: Option<&[Point3]>,
    owner: usize,
    face_index: usize,
    value: f64,
    mass_flux: f64,
    scheme: LaminarSimpleConvectionScheme,
) {
    add_boundary_upwind_convection(rows, rhs, owner, value, mass_flux);
    if mass_flux >= 0.0
        && matches!(scheme, LaminarSimpleConvectionScheme::GaussLinearUpwind)
        && let Some(gradient) = old_gradient
    {
        let correction = linear_upwind_scalar_correction(
            mesh, old_values, gradient, owner, None, face_index, mass_flux,
        );
        rhs[owner] -= mass_flux * correction;
    }
}

#[allow(clippy::too_many_arguments)]
fn add_boundary_extrapolated_convection(
    rows: &mut [BTreeMap<usize, f64>],
    rhs: &mut [f64],
    mesh: &SolverRuntimeMeshData,
    old_values: &[f64],
    old_gradient: Option<&[Point3]>,
    owner: usize,
    face_index: usize,
    mass_flux: f64,
    scheme: LaminarSimpleConvectionScheme,
) {
    add_entry(&mut rows[owner], owner, mass_flux);
    if matches!(scheme, LaminarSimpleConvectionScheme::GaussLinearUpwind)
        && let Some(gradient) = old_gradient
    {
        let correction = linear_upwind_scalar_correction(
            mesh, old_values, gradient, owner, None, face_index, mass_flux,
        );
        rhs[owner] -= mass_flux * correction;
    }
}

fn linear_upwind_scalar_correction(
    mesh: &SolverRuntimeMeshData,
    _old_values: &[f64],
    gradient: &[Point3],
    owner: usize,
    neighbour: Option<usize>,
    face_index: usize,
    mass_flux: f64,
) -> f64 {
    let upwind = if mass_flux >= 0.0 {
        owner
    } else {
        neighbour.unwrap_or(owner)
    };
    let cell_centre = mesh.cell_centres[upwind];
    let face_centre = mesh.face_centres[face_index];
    let delta = Point3 {
        x: face_centre.x - cell_centre.x,
        y: face_centre.y - cell_centre.y,
        z: face_centre.z - cell_centre.z,
    };
    dot(gradient[upwind], delta)
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
            ScalarFaceTreatment::InletOutlet(value) => {
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
            ScalarFaceTreatment::FixedGradient(gradient) => {
                let flux = fixed_gradient_pressure_flux(
                    mesh,
                    cell_diffusivity,
                    owner,
                    face_index,
                    gradient,
                )?;
                rhs[owner] -= flux;
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
    for (face_index, face_flux) in flux.iter_mut().enumerate() {
        let face_velocity = face_vector_value(mesh, velocity, boundary, face_index);
        *face_flux = dot(face_velocity, mesh.face_area_vectors[face_index]);
    }
    Ok(flux)
}

fn compute_phi_hby_a(
    mesh: &SolverRuntimeMeshData,
    hby_a: &[Point3],
    velocity_boundary: &[VectorFaceTreatment],
) -> Result<Vec<f64>> {
    compute_face_flux(mesh, hby_a, velocity_boundary)
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
    validate_non_negative_cell_values("pressure correction rAU", r_au)?;
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
            ScalarFaceTreatment::InletOutlet(value) => {
                let coefficient =
                    variable_face_diffusion_coefficient(mesh, r_au, owner, None, face_index)?;
                flux[face_index] = coefficient * (pressure_correction[owner] - value);
            }
            ScalarFaceTreatment::FixedGradient(gradient) => {
                flux[face_index] =
                    fixed_gradient_pressure_flux(mesh, r_au, owner, face_index, gradient)?;
            }
            ScalarFaceTreatment::ZeroGradient | ScalarFaceTreatment::Constraint => {}
        }
    }
    Ok(flux)
}

fn pressure_equation_flux(
    mesh: &SolverRuntimeMeshData,
    pressure: &[f64],
    r_au: &[f64],
    boundary: &[ScalarFaceTreatment],
) -> Result<Vec<f64>> {
    Ok(pressure_correction_flux(mesh, pressure, r_au, boundary)?
        .into_iter()
        .map(|flux| -flux)
        .collect())
}

fn consistent_phi_hby_a_pressure_correction(
    mesh: &SolverRuntimeMeshData,
    pressure: &[f64],
    boundary: &[ScalarFaceTreatment],
    r_au: &[f64],
    r_at_u: &[f64],
) -> Result<Vec<f64>> {
    if r_au.len() != mesh.cells || r_at_u.len() != mesh.cells {
        return Err(invalid_input(format!(
            "consistent SIMPLE rAU/rAtU expected {} cell values, got rAU={} rAtU={}",
            mesh.cells,
            r_au.len(),
            r_at_u.len()
        )));
    }
    let mut delta = Vec::with_capacity(mesh.cells);
    for (cell, (r_au, r_at_u)) in r_au.iter().zip(r_at_u).enumerate() {
        let value = r_at_u - r_au;
        if !value.is_finite() || value < -f64::EPSILON {
            return Err(invalid_input(format!(
                "consistent SIMPLE rAtU-rAU for cell {cell} must be non-negative and finite, got {value}"
            )));
        }
        delta.push(value.max(0.0));
    }

    Ok(pressure_correction_flux(mesh, pressure, &delta, boundary)?
        .into_iter()
        .map(|flux| -flux)
        .collect())
}

fn non_orthogonal_pressure_flux_correction(
    mesh: &SolverRuntimeMeshData,
    pressure: &[f64],
    r_at_u: &[f64],
    boundary: &[ScalarFaceTreatment],
    gradient_scheme: LaminarSimpleGradientScheme,
) -> Result<Vec<f64>> {
    if pressure.len() != mesh.cells {
        return Err(invalid_input(format!(
            "non-orthogonal pressure correction expected {} cell values, got {}",
            mesh.cells,
            pressure.len()
        )));
    }
    if r_at_u.len() != mesh.cells {
        return Err(invalid_input(format!(
            "non-orthogonal pressure correction rAtU has {} values, expected {} mesh cells",
            r_at_u.len(),
            mesh.cells
        )));
    }
    validate_positive_cell_values("non-orthogonal pressure correction rAtU", r_at_u)?;
    if boundary.len() != mesh.faces {
        return Err(invalid_input(format!(
            "non-orthogonal pressure correction boundary has {} values, expected {} mesh faces",
            boundary.len(),
            mesh.faces
        )));
    }

    let pressure_gradient = scalar_gradient(mesh, pressure, boundary, gradient_scheme)?;
    let mut flux = vec![0.0; mesh.faces];
    for (face_index, treatment) in boundary.iter().enumerate() {
        let owner = mesh.owner[face_index];
        if let Some(neighbour) = mesh.neighbour[face_index] {
            let diffusivity = 0.5 * (r_at_u[owner] + r_at_u[neighbour]);
            let face_gradient = average(pressure_gradient[owner], pressure_gradient[neighbour]);
            let non_orthogonal_area = non_orthogonal_area_vector(
                mesh.face_area_vectors[face_index],
                mesh.cell_centres[owner],
                mesh.cell_centres[neighbour],
                face_index,
            )?;
            flux[face_index] = -diffusivity * dot(face_gradient, non_orthogonal_area);
            continue;
        }

        if matches!(
            treatment,
            ScalarFaceTreatment::FixedValue(_)
                | ScalarFaceTreatment::FixedGradient(_)
                | ScalarFaceTreatment::InletOutlet(_)
        ) {
            let non_orthogonal_area = non_orthogonal_area_vector(
                mesh.face_area_vectors[face_index],
                mesh.cell_centres[owner],
                mesh.face_centres[face_index],
                face_index,
            )?;
            flux[face_index] = -r_at_u[owner] * dot(pressure_gradient[owner], non_orthogonal_area);
        }
    }
    Ok(flux)
}

fn non_orthogonal_area_vector(
    area_vector: Point3,
    from: Point3,
    to: Point3,
    face_index: usize,
) -> Result<Point3> {
    let distance = distance(from, to);
    if !distance.is_finite() || distance <= f64::EPSILON {
        return Err(invalid_input(format!(
            "face {face_index} has non-positive non-orthogonal correction distance {distance}"
        )));
    }
    let direction = Point3 {
        x: (to.x - from.x) / distance,
        y: (to.y - from.y) / distance,
        z: (to.z - from.z) / distance,
    };
    let orthogonal_area = dot(area_vector, direction);
    Ok(Point3 {
        x: area_vector.x - orthogonal_area * direction.x,
        y: area_vector.y - orthogonal_area * direction.y,
        z: area_vector.z - orthogonal_area * direction.z,
    })
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
    scheme: LaminarSimpleGradientScheme,
) -> Result<Vec<Point3>> {
    if values.len() != mesh.cells {
        return Err(invalid_input(format!(
            "scalar gradient expected {} cell values, got {}",
            mesh.cells,
            values.len()
        )));
    }
    if boundary.len() != mesh.faces {
        return Err(invalid_input(format!(
            "scalar gradient expected {} boundary treatments, got {}",
            mesh.faces,
            boundary.len()
        )));
    }
    for (cell, value) in values.iter().copied().enumerate() {
        require_finite(value, format!("scalar gradient cell {cell} value"))?;
    }
    let mut gradient = vec![zero(); mesh.cells];
    for face_index in 0..mesh.faces {
        let owner = mesh.owner[face_index];
        let face_value = face_scalar_value(mesh, values, boundary, face_index)?;
        let area = mesh.face_area_vectors[face_index];
        checked_add_scaled(&mut gradient[owner], area, face_value, face_index, owner)?;
        if let Some(neighbour) = mesh.neighbour[face_index] {
            checked_add_scaled(
                &mut gradient[neighbour],
                area,
                -face_value,
                face_index,
                neighbour,
            )?;
        }
    }
    for (cell, (value, volume)) in gradient.iter_mut().zip(&mesh.cell_volumes).enumerate() {
        if !volume.is_finite() || *volume <= f64::EPSILON {
            return Err(invalid_input(format!(
                "scalar gradient cell {cell} has non-positive or non-finite volume {volume}"
            )));
        }
        checked_scale(value, 1.0 / volume, format!("scalar gradient cell {cell}"))?;
    }
    match scheme {
        LaminarSimpleGradientScheme::GaussLinear => Ok(gradient),
        LaminarSimpleGradientScheme::CellLimitedGaussLinear(coefficient) => {
            limit_scalar_gradient(mesh, values, boundary, gradient, coefficient)
        }
    }
}

fn vector_convection_divergence(
    mesh: &SolverRuntimeMeshData,
    velocity: &[Point3],
    boundary: &[VectorFaceTreatment],
    flux: &[f64],
    scheme: LaminarSimpleConvectionScheme,
    gradient_scheme: LaminarSimpleGradientScheme,
) -> Result<Vec<Point3>> {
    if flux.len() != mesh.faces {
        return Err(invalid_input(format!(
            "flux has {} values, expected {} mesh faces",
            flux.len(),
            mesh.faces
        )));
    }
    let gradients = if matches!(scheme, LaminarSimpleConvectionScheme::GaussLinearUpwind) {
        Some(vector_component_gradients(
            mesh,
            velocity,
            boundary,
            gradient_scheme,
        )?)
    } else {
        None
    };
    let mut divergence = vec![zero(); mesh.cells];
    for (face_index, phi) in flux.iter().copied().enumerate() {
        let owner = mesh.owner[face_index];
        let face_velocity = convection_face_vector_value(
            mesh,
            velocity,
            boundary,
            face_index,
            phi,
            scheme,
            gradients.as_ref(),
        );
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

fn adjust_phi_hby_a(
    mesh: &SolverRuntimeMeshData,
    velocity_boundary: &[VectorFaceTreatment],
    pressure_boundary: &[ScalarFaceTreatment],
    phi_hby_a: &mut [f64],
) -> Result<AdjustPhiSummary> {
    if velocity_boundary.len() != mesh.faces
        || pressure_boundary.len() != mesh.faces
        || phi_hby_a.len() != mesh.faces
    {
        return Err(invalid_input(format!(
            "adjustPhi inputs must match mesh face count {}",
            mesh.faces
        )));
    }

    let global_flux_before = boundary_global_flux(mesh, phi_hby_a);
    if !global_flux_before.is_finite() {
        return Err(invalid_input(format!(
            "adjustPhi global flux must be finite, got {global_flux_before}"
        )));
    }

    let mut adjustable_area = 0.0;
    let mut adjusted_faces = 0;
    for face_index in 0..mesh.faces {
        if mesh.neighbour[face_index].is_some()
            || !is_adjustable_pressure_open_face(
                velocity_boundary[face_index],
                pressure_boundary[face_index],
                phi_hby_a[face_index],
            )
        {
            continue;
        }
        let area = magnitude(mesh.face_area_vectors[face_index]);
        if area.is_finite() && area > f64::EPSILON {
            adjustable_area += area;
            adjusted_faces += 1;
        }
    }

    if adjusted_faces == 0 || adjustable_area <= f64::EPSILON {
        return Ok(AdjustPhiSummary {
            global_flux_before,
            global_flux_after: global_flux_before,
            adjusted_faces: 0,
        });
    }

    for face_index in 0..mesh.faces {
        if mesh.neighbour[face_index].is_some()
            || !is_adjustable_pressure_open_face(
                velocity_boundary[face_index],
                pressure_boundary[face_index],
                phi_hby_a[face_index],
            )
        {
            continue;
        }
        let area = magnitude(mesh.face_area_vectors[face_index]);
        if area.is_finite() && area > f64::EPSILON {
            phi_hby_a[face_index] += -global_flux_before * area / adjustable_area;
        }
    }

    Ok(AdjustPhiSummary {
        global_flux_before,
        global_flux_after: boundary_global_flux(mesh, phi_hby_a),
        adjusted_faces,
    })
}

fn is_adjustable_pressure_open_face(
    velocity: VectorFaceTreatment,
    pressure: ScalarFaceTreatment,
    flux: f64,
) -> bool {
    matches!(
        velocity,
        VectorFaceTreatment::ZeroGradient | VectorFaceTreatment::InletOutlet(_) if flux >= 0.0
    ) && matches!(pressure, ScalarFaceTreatment::FixedValue(_))
}

fn boundary_global_flux(mesh: &SolverRuntimeMeshData, flux: &[f64]) -> f64 {
    let mut global_flux = 0.0;
    for (face_index, value) in flux.iter().copied().enumerate() {
        if mesh.neighbour[face_index].is_none() {
            global_flux += value;
        }
    }
    global_flux
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

fn consistent_reciprocal_momentum_diagonal(
    mesh: &SolverRuntimeMeshData,
    r_au: &[f64],
    h1: &[f64],
) -> Result<Vec<f64>> {
    if r_au.len() != mesh.cells {
        return Err(invalid_input(format!(
            "consistent SIMPLE rAU has {} values, expected {} mesh cells",
            r_au.len(),
            mesh.cells
        )));
    }
    if h1.len() != mesh.cells {
        return Err(invalid_input(format!(
            "consistent SIMPLE H1 has {} values, expected {} mesh cells",
            h1.len(),
            mesh.cells
        )));
    }
    r_au
        .iter()
        .zip(h1)
        .enumerate()
        .map(|(cell, (r_au, h1))| {
            if !r_au.is_finite() || *r_au <= f64::EPSILON {
                return Err(invalid_input(format!(
                    "consistent SIMPLE rAU for cell {cell} must be positive and finite, got {r_au}"
                )));
            }
            if !h1.is_finite() || *h1 < 0.0 {
                return Err(invalid_input(format!(
                    "consistent SIMPLE H1 for cell {cell} must be non-negative and finite, got {h1}"
                )));
            }
            let denominator = (1.0 / r_au) - h1;
            if !denominator.is_finite() || denominator <= f64::EPSILON {
                return Err(invalid_input(format!(
                    "consistent SIMPLE rAtU denominator for cell {cell} must be positive and finite, got {denominator}"
                )));
            }
            Ok(1.0 / denominator)
        })
        .collect()
}

fn hby_a_from_predicted_velocity(
    predicted_velocity: &[Point3],
    pressure_gradient: &[Point3],
    r_at_u: &[f64],
) -> Result<Vec<Point3>> {
    if predicted_velocity.len() != pressure_gradient.len()
        || predicted_velocity.len() != r_at_u.len()
    {
        return Err(invalid_input(format!(
            "HbyA inputs have incompatible sizes: U={} grad(p)={} rAtU={}",
            predicted_velocity.len(),
            pressure_gradient.len(),
            r_at_u.len()
        )));
    }

    Ok(predicted_velocity
        .iter()
        .zip(pressure_gradient)
        .zip(r_at_u)
        .map(|((velocity, gradient), r_at_u)| Point3 {
            x: velocity.x + r_at_u * gradient.x,
            y: velocity.y + r_at_u * gradient.y,
            z: velocity.z + r_at_u * gradient.z,
        })
        .collect())
}

fn consistent_hby_a_from_base(
    hby_a: &[Point3],
    pressure_gradient: &[Point3],
    r_au: &[f64],
    r_at_u: &[f64],
) -> Result<Vec<Point3>> {
    if hby_a.len() != pressure_gradient.len()
        || hby_a.len() != r_au.len()
        || hby_a.len() != r_at_u.len()
    {
        return Err(invalid_input(format!(
            "consistent SIMPLE HbyA inputs have incompatible sizes: HbyA={} grad(p)={} rAU={} rAtU={}",
            hby_a.len(),
            pressure_gradient.len(),
            r_au.len(),
            r_at_u.len()
        )));
    }

    Ok(hby_a
        .iter()
        .zip(pressure_gradient)
        .zip(r_au)
        .zip(r_at_u)
        .map(|(((hby_a, gradient), r_au), r_at_u)| {
            let delta = r_au - r_at_u;
            Point3 {
                x: hby_a.x - delta * gradient.x,
                y: hby_a.y - delta * gradient.y,
                z: hby_a.z - delta * gradient.z,
            }
        })
        .collect())
}

fn velocity_from_hby_a(
    mesh: &SolverRuntimeMeshData,
    hby_a: &[Point3],
    pressure_gradient: &[Point3],
    r_at_u: &[f64],
) -> Result<Vec<Point3>> {
    if hby_a.len() != mesh.cells
        || pressure_gradient.len() != mesh.cells
        || r_at_u.len() != mesh.cells
    {
        return Err(invalid_input(format!(
            "HbyA velocity correction expected {} cell values, got HbyA={} grad(p)={} rAtU={}",
            mesh.cells,
            hby_a.len(),
            pressure_gradient.len(),
            r_at_u.len()
        )));
    }

    Ok(hby_a
        .iter()
        .zip(pressure_gradient)
        .zip(r_at_u)
        .map(|((hby_a, gradient), r_at_u)| Point3 {
            x: hby_a.x - r_at_u * gradient.x,
            y: hby_a.y - r_at_u * gradient.y,
            z: hby_a.z - r_at_u * gradient.z,
        })
        .collect())
}

fn relative_scalar_field_change_l2(before: &[f64], after: &[f64]) -> f64 {
    let mut delta_squared_sum = 0.0;
    let mut value_squared_sum = 0.0;
    for (before, after) in before.iter().zip(after) {
        let delta = *after - *before;
        delta_squared_sum += delta * delta;
        value_squared_sum += after * after;
    }
    let value_norm = value_squared_sum.sqrt();
    if value_norm <= f64::EPSILON {
        delta_squared_sum.sqrt()
    } else {
        delta_squared_sum.sqrt() / value_norm
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
    hby_a: &[Point3],
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
        hby_a_l2_norm: vector_l2_norm(hby_a),
        div_phi_u_l2_norm: vector_l2_norm(convection),
    }
}

fn summarize_scalars(values: &[f64]) -> ScalarDiagnosticSummary {
    if values.is_empty() {
        return ScalarDiagnosticSummary::default();
    }

    let mut summary = ScalarDiagnosticSummary {
        min: f64::INFINITY,
        max: f64::NEG_INFINITY,
        ..ScalarDiagnosticSummary::default()
    };
    let mut sum_squares = 0.0;
    for value in values {
        summary.min = summary.min.min(*value);
        summary.max = summary.max.max(*value);
        summary.sum += value;
        summary.sum_abs += value.abs();
        sum_squares += value * value;
    }
    summary.l2_norm = sum_squares.sqrt();
    summary
}

fn summarize_vectors(values: &[Point3]) -> VectorDiagnosticSummary {
    if values.is_empty() {
        return VectorDiagnosticSummary::default();
    }

    let mut summary = VectorDiagnosticSummary {
        min_magnitude: f64::INFINITY,
        max_magnitude: f64::NEG_INFINITY,
        x_min: f64::INFINITY,
        x_max: f64::NEG_INFINITY,
        y_min: f64::INFINITY,
        y_max: f64::NEG_INFINITY,
        z_min: f64::INFINITY,
        z_max: f64::NEG_INFINITY,
        ..VectorDiagnosticSummary::default()
    };
    let mut sum_squares = 0.0;
    for value in values {
        let magnitude = magnitude(*value);
        summary.min_magnitude = summary.min_magnitude.min(magnitude);
        summary.max_magnitude = summary.max_magnitude.max(magnitude);
        summary.x_min = summary.x_min.min(value.x);
        summary.x_max = summary.x_max.max(value.x);
        summary.y_min = summary.y_min.min(value.y);
        summary.y_max = summary.y_max.max(value.y);
        summary.z_min = summary.z_min.min(value.z);
        summary.z_max = summary.z_max.max(value.z);
        sum_squares += dot(*value, *value);
    }
    summary.l2_norm = sum_squares.sqrt();
    summary
}

fn summarize_face_fluxes(
    mesh: &SolverRuntimeMeshData,
    values: &[f64],
) -> FaceFluxDiagnosticSummary {
    if values.is_empty() {
        return FaceFluxDiagnosticSummary::default();
    }

    let mut summary = FaceFluxDiagnosticSummary {
        min: f64::INFINITY,
        max: f64::NEG_INFINITY,
        ..FaceFluxDiagnosticSummary::default()
    };
    let mut sum_squares = 0.0;
    for (face_index, value) in values.iter().enumerate() {
        summary.min = summary.min.min(*value);
        summary.max = summary.max.max(*value);
        summary.sum += value;
        summary.sum_abs += value.abs();
        sum_squares += value * value;
        if mesh
            .neighbour
            .get(face_index)
            .and_then(|neighbour| *neighbour)
            .is_some()
        {
            summary.internal_sum_abs += value.abs();
        } else {
            summary.boundary_sum += value;
            summary.boundary_sum_abs += value.abs();
        }
    }
    summary.l2_norm = sum_squares.sqrt();
    summary
}

fn summarize_csr_matrix(matrix: &CsrMatrix) -> Result<MatrixDiagnosticSummary> {
    let diagonal = matrix.diagonal()?;
    let mut summary = MatrixDiagnosticSummary {
        rows: matrix.rows(),
        cols: matrix.cols(),
        nonzeros: matrix.nnz(),
        diagonal_min: f64::INFINITY,
        diagonal_max: f64::NEG_INFINITY,
        ..MatrixDiagnosticSummary::default()
    };
    for value in &diagonal {
        summary.diagonal_min = summary.diagonal_min.min(*value);
        summary.diagonal_max = summary.diagonal_max.max(*value);
        summary.diagonal_sum_abs += value.abs();
    }
    if diagonal.is_empty() {
        summary.diagonal_min = 0.0;
        summary.diagonal_max = 0.0;
    }

    for row in 0..matrix.rows() {
        let mut row_sum_abs = 0.0;
        let mut row_off_diagonal_sum_abs = 0.0;
        for entry in matrix.row_offsets()[row]..matrix.row_offsets()[row + 1] {
            let value_abs = matrix.values()[entry].abs();
            row_sum_abs += value_abs;
            if matrix.col_indices()[entry] != row {
                row_off_diagonal_sum_abs += value_abs;
                summary.off_diagonal_sum_abs += value_abs;
            }
        }
        summary.max_row_sum_abs = summary.max_row_sum_abs.max(row_sum_abs);
        summary.max_row_off_diagonal_sum_abs = summary
            .max_row_off_diagonal_sum_abs
            .max(row_off_diagonal_sum_abs);
    }

    Ok(summary)
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
            VectorFaceTreatment::InletOutlet(_) => summary.velocity_inlet_outlet_faces += 1,
            VectorFaceTreatment::ZeroGradient => summary.velocity_zero_gradient_faces += 1,
            VectorFaceTreatment::Constraint => summary.velocity_constraint_faces += 1,
        }
    }
    for (face, treatment) in pressure.iter().enumerate() {
        if mesh.neighbour[face].is_some() {
            continue;
        }
        match treatment {
            ScalarFaceTreatment::FixedValue(_) | ScalarFaceTreatment::InletOutlet(_) => {
                summary.pressure_fixed_value_faces += 1;
            }
            ScalarFaceTreatment::FixedGradient(_) | ScalarFaceTreatment::ZeroGradient => {
                summary.pressure_zero_gradient_faces += 1;
            }
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
                Some("inletOutlet" | "pressureInletOutletVelocity") => {
                    inlet_outlet_vector_patch_values(field, field_patch, patch.faces)?
                }
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
                Some("fixedFluxPressure") => {
                    vec![ScalarFaceTreatment::FixedGradient(0.0); patch.faces]
                }
                Some("inletOutlet") => {
                    inlet_outlet_scalar_patch_values(field, field_patch, patch.faces)?
                }
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

fn constrained_pressure_treatments(
    mesh: &SolverRuntimeMeshData,
    pressure_boundary: &[ScalarFaceTreatment],
    velocity_boundary: &[VectorFaceTreatment],
    velocity: &[Point3],
    phi_hby_a: &[f64],
    r_au: &[f64],
) -> Result<Vec<ScalarFaceTreatment>> {
    if pressure_boundary.len() != mesh.faces || velocity_boundary.len() != mesh.faces {
        return Err(invalid_input(format!(
            "pressure constraint boundary sizes must match mesh faces ({})",
            mesh.faces
        )));
    }
    if velocity.len() != mesh.cells || phi_hby_a.len() != mesh.faces || r_au.len() != mesh.cells {
        return Err(invalid_input(
            "pressure constraint field sizes do not match runtime mesh".to_string(),
        ));
    }

    let mut constrained = pressure_boundary.to_vec();
    for face_index in 0..mesh.faces {
        if mesh.neighbour[face_index].is_some() {
            continue;
        }
        if !is_pressure_gradient_constrained_boundary(
            pressure_boundary[face_index],
            velocity_boundary[face_index],
            phi_hby_a[face_index],
        ) {
            continue;
        }
        let owner = mesh.owner[face_index];
        let area = magnitude(mesh.face_area_vectors[face_index]);
        let face_r_au = r_au[owner];
        if !area.is_finite() || area <= f64::EPSILON {
            return Err(invalid_input(format!(
                "fixedFluxPressure face {face_index} has non-positive area magnitude {area}"
            )));
        }
        if !face_r_au.is_finite() || face_r_au <= f64::EPSILON {
            return Err(invalid_input(format!(
                "fixedFluxPressure face {face_index} has invalid rAU {face_r_au}"
            )));
        }
        let u_flux = prescribed_boundary_velocity_flux(
            mesh,
            velocity,
            velocity_boundary,
            face_index,
            phi_hby_a[face_index],
        )
        .unwrap_or_else(|| {
            dot(
                face_vector_value(mesh, velocity, velocity_boundary, face_index),
                mesh.face_area_vectors[face_index],
            )
        });
        let gradient = (phi_hby_a[face_index] - u_flux) / (area * face_r_au);
        constrained[face_index] = ScalarFaceTreatment::FixedGradient(gradient);
    }
    Ok(constrained)
}

fn is_pressure_gradient_constrained_boundary(
    pressure: ScalarFaceTreatment,
    velocity: VectorFaceTreatment,
    flux: f64,
) -> bool {
    match pressure {
        ScalarFaceTreatment::FixedGradient(_) => true,
        ScalarFaceTreatment::ZeroGradient => prescribed_velocity_boundary_is_active(velocity, flux),
        ScalarFaceTreatment::InletOutlet(_) => flux < 0.0,
        ScalarFaceTreatment::FixedValue(_) | ScalarFaceTreatment::Constraint => false,
    }
}

fn prescribed_velocity_boundary_is_active(velocity: VectorFaceTreatment, flux: f64) -> bool {
    match velocity {
        VectorFaceTreatment::FixedValue(_) => true,
        VectorFaceTreatment::InletOutlet(_) => flux < 0.0,
        VectorFaceTreatment::ZeroGradient | VectorFaceTreatment::Constraint => false,
    }
}

fn prescribed_boundary_velocity_flux(
    mesh: &SolverRuntimeMeshData,
    velocity: &[Point3],
    boundary: &[VectorFaceTreatment],
    face_index: usize,
    flux: f64,
) -> Option<f64> {
    let value = match boundary[face_index] {
        VectorFaceTreatment::FixedValue(value) => value,
        VectorFaceTreatment::InletOutlet(value) if flux < 0.0 => value,
        VectorFaceTreatment::InletOutlet(_)
        | VectorFaceTreatment::ZeroGradient
        | VectorFaceTreatment::Constraint => return None,
    };
    if mesh.neighbour[face_index].is_some() {
        return None;
    }
    if velocity.len() != mesh.cells {
        return None;
    }
    Some(dot(value, mesh.face_area_vectors[face_index]))
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

fn inlet_outlet_vector_patch_values(
    field: &FieldFile,
    patch: &crate::fields::FieldBoundaryPatch,
    faces: usize,
) -> Result<Vec<VectorFaceTreatment>> {
    let value = patch
        .inlet_value
        .as_ref()
        .or(patch.value.as_ref())
        .ok_or_else(|| {
            invalid_input(format!(
                "field '{}' patch '{}' inletOutlet boundary has neither inletValue nor value",
                field_label(field),
                patch.name
            ))
        })?;
    let values = parse_patch_numeric_values(value, 3, faces, field, &patch.name)?;
    Ok(values
        .chunks_exact(3)
        .map(|chunk| {
            VectorFaceTreatment::InletOutlet(Point3 {
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

fn inlet_outlet_scalar_patch_values(
    field: &FieldFile,
    patch: &crate::fields::FieldBoundaryPatch,
    faces: usize,
) -> Result<Vec<ScalarFaceTreatment>> {
    let value = patch
        .inlet_value
        .as_ref()
        .or(patch.value.as_ref())
        .ok_or_else(|| {
            invalid_input(format!(
                "field '{}' patch '{}' inletOutlet boundary has neither inletValue nor value",
                field_label(field),
                patch.name
            ))
        })?;
    let values = parse_patch_numeric_values(value, 1, faces, field, &patch.name)?;
    Ok(values
        .into_iter()
        .map(ScalarFaceTreatment::InletOutlet)
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
        VectorFaceTreatment::InletOutlet(value) => {
            let owner_flux = dot(velocity[owner], mesh.face_area_vectors[face_index]);
            if owner_flux < 0.0 {
                value
            } else {
                velocity[owner]
            }
        }
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
        VectorFaceTreatment::InletOutlet(value) if flux < 0.0 => value,
        VectorFaceTreatment::InletOutlet(_) => velocity[owner],
        VectorFaceTreatment::ZeroGradient | VectorFaceTreatment::Constraint => velocity[owner],
    }
}

fn convection_face_vector_value(
    mesh: &SolverRuntimeMeshData,
    velocity: &[Point3],
    boundary: &[VectorFaceTreatment],
    face_index: usize,
    flux: f64,
    scheme: LaminarSimpleConvectionScheme,
    gradients: Option<&[Vec<Point3>; 3]>,
) -> Point3 {
    let upwind = upwind_face_vector_value(mesh, velocity, boundary, face_index, flux);
    if !matches!(scheme, LaminarSimpleConvectionScheme::GaussLinearUpwind) {
        return upwind;
    }
    let Some(gradients) = gradients else {
        return upwind;
    };

    let owner = mesh.owner[face_index];
    let upwind_cell = if let Some(neighbour) = mesh.neighbour[face_index] {
        if flux >= 0.0 { owner } else { neighbour }
    } else if flux < 0.0 {
        return upwind;
    } else {
        owner
    };
    let cell_centre = mesh.cell_centres[upwind_cell];
    let face_centre = mesh.face_centres[face_index];
    let delta = Point3 {
        x: face_centre.x - cell_centre.x,
        y: face_centre.y - cell_centre.y,
        z: face_centre.z - cell_centre.z,
    };

    Point3 {
        x: upwind.x + dot(gradients[0][upwind_cell], delta),
        y: upwind.y + dot(gradients[1][upwind_cell], delta),
        z: upwind.z + dot(gradients[2][upwind_cell], delta),
    }
}

fn face_scalar_value(
    mesh: &SolverRuntimeMeshData,
    values: &[f64],
    boundary: &[ScalarFaceTreatment],
    face_index: usize,
) -> Result<f64> {
    let owner = mesh.owner[face_index];
    if let Some(neighbour) = mesh.neighbour[face_index] {
        let weight = gauss_linear_owner_weight(mesh, owner, neighbour, face_index)?;
        let owner_part = checked_product(
            weight,
            values[owner],
            format!("internal face {face_index} owner interpolation"),
        )?;
        let neighbour_part = checked_product(
            1.0 - weight,
            values[neighbour],
            format!("internal face {face_index} neighbour interpolation"),
        )?;
        return require_finite(
            owner_part + neighbour_part,
            format!("internal face {face_index} interpolated value"),
        );
    }
    let value = match boundary[face_index] {
        ScalarFaceTreatment::FixedValue(value) => value,
        ScalarFaceTreatment::FixedGradient(gradient) => {
            let distance = boundary_normal_distance(mesh, owner, face_index);
            require_finite(
                distance,
                format!("boundary face {face_index} normal distance"),
            )?;
            let increment = checked_product(
                gradient,
                distance,
                format!("boundary face {face_index} fixed-gradient extrapolation"),
            )?;
            require_finite(
                values[owner] + increment,
                format!("boundary face {face_index} fixed-gradient value"),
            )?
        }
        ScalarFaceTreatment::InletOutlet(value) => value,
        ScalarFaceTreatment::ZeroGradient | ScalarFaceTreatment::Constraint => values[owner],
    };
    require_finite(value, format!("boundary face {face_index} effective value"))
}

const V_GREAT: f64 = f64::MAX / 10.0;

fn gauss_linear_owner_weight(
    mesh: &SolverRuntimeMeshData,
    owner: usize,
    neighbour: usize,
    face_index: usize,
) -> Result<f64> {
    let face = mesh.face_centres[face_index];
    let owner_delta = checked_delta(
        face,
        mesh.cell_centres[owner],
        format!("internal face {face_index} owner-centre delta"),
    )?;
    let neighbour_delta = checked_delta(
        mesh.cell_centres[neighbour],
        face,
        format!("internal face {face_index} neighbour-centre delta"),
    )?;
    let area = mesh.face_area_vectors[face_index];
    require_finite_point(area, format!("internal face {face_index} area vector"))?;
    let sfd_owner = checked_dot(
        area,
        owner_delta,
        format!("internal face {face_index} projected owner distance"),
    )?
    .abs();
    let sfd_neighbour = checked_dot(
        area,
        neighbour_delta,
        format!("internal face {face_index} projected neighbour distance"),
    )?
    .abs();
    let projected_sum = require_finite(
        sfd_owner + sfd_neighbour,
        format!("internal face {face_index} projected distance sum"),
    )?;

    let weight = if sfd_neighbour / V_GREAT < projected_sum {
        sfd_neighbour / projected_sum
    } else {
        let owner_distance = checked_magnitude(
            owner_delta,
            format!("internal face {face_index} Euclidean owner distance"),
        )?;
        let neighbour_distance = checked_magnitude(
            neighbour_delta,
            format!("internal face {face_index} Euclidean neighbour distance"),
        )?;
        let distance_sum = require_finite(
            owner_distance + neighbour_distance,
            format!("internal face {face_index} Euclidean distance sum"),
        )?;
        if distance_sum <= 0.0 {
            return Err(invalid_input(format!(
                "internal face {face_index} has zero projected and Euclidean centre distance"
            )));
        }
        neighbour_distance / distance_sum
    };
    let weight = require_finite(weight, format!("internal face {face_index} linear weight"))?;
    if !(0.0..=1.0).contains(&weight) {
        return Err(invalid_input(format!(
            "internal face {face_index} linear weight {weight} is outside [0, 1]"
        )));
    }
    Ok(weight)
}

fn limit_scalar_gradient(
    mesh: &SolverRuntimeMeshData,
    values: &[f64],
    boundary: &[ScalarFaceTreatment],
    mut gradient: Vec<Point3>,
    coefficient: f64,
) -> Result<Vec<Point3>> {
    if !coefficient.is_finite() || !(0.0..=1.0).contains(&coefficient) {
        return Err(invalid_input(format!(
            "cellLimited gradient coefficient must be finite and in [0, 1], got {coefficient}"
        )));
    }
    if coefficient == 0.0 {
        return Ok(gradient);
    }

    let mut minima = values.to_vec();
    let mut maxima = values.to_vec();
    for face_index in 0..mesh.faces {
        let owner = mesh.owner[face_index];
        if let Some(neighbour) = mesh.neighbour[face_index] {
            minima[owner] = minima[owner].min(values[neighbour]);
            maxima[owner] = maxima[owner].max(values[neighbour]);
            minima[neighbour] = minima[neighbour].min(values[owner]);
            maxima[neighbour] = maxima[neighbour].max(values[owner]);
        } else {
            let boundary_value = face_scalar_value(mesh, values, boundary, face_index)?;
            minima[owner] = minima[owner].min(boundary_value);
            maxima[owner] = maxima[owner].max(boundary_value);
        }
    }

    for cell in 0..mesh.cells {
        let maximum_delta = checked_subtraction(
            maxima[cell],
            values[cell],
            format!("cellLimited cell {cell} maximum extrema delta"),
        )?;
        let minimum_delta = checked_subtraction(
            minima[cell],
            values[cell],
            format!("cellLimited cell {cell} minimum extrema delta"),
        )?;
        let span = checked_subtraction(
            maxima[cell],
            minima[cell],
            format!("cellLimited cell {cell} extrema span"),
        )?;
        let widening = if coefficient == 1.0 {
            0.0
        } else {
            let widening_numerator = checked_product(
                span,
                1.0 - coefficient,
                format!("cellLimited cell {cell} widening numerator"),
            )?;
            require_finite(
                widening_numerator / coefficient,
                format!("cellLimited cell {cell} widening term"),
            )?
        };
        let widened_maximum = require_finite(
            maximum_delta + widening,
            format!("cellLimited cell {cell} widened maximum delta"),
        )?;
        let widened_minimum = require_finite(
            minimum_delta - widening,
            format!("cellLimited cell {cell} widened minimum delta"),
        )?;
        let mut limiter: f64 = 1.0;

        for face_index in 0..mesh.faces {
            if mesh.owner[face_index] != cell && mesh.neighbour[face_index] != Some(cell) {
                continue;
            }
            let delta = checked_delta(
                mesh.face_centres[face_index],
                mesh.cell_centres[cell],
                format!("cellLimited cell {cell} face {face_index} centre delta"),
            )?;
            let extrapolation = checked_dot(
                gradient[cell],
                delta,
                format!("cellLimited cell {cell} face {face_index} extrapolation"),
            )?;
            let ratio = if extrapolation > widened_maximum && extrapolation > 0.0 {
                widened_maximum / extrapolation
            } else if extrapolation < widened_minimum && extrapolation < 0.0 {
                widened_minimum / extrapolation
            } else {
                1.0
            };
            let ratio = require_finite(
                ratio,
                format!("cellLimited cell {cell} face {face_index} limiter ratio"),
            )?;
            limiter = limiter.min(ratio.clamp(0.0, 1.0));
            require_finite(limiter, format!("cellLimited cell {cell} final limiter"))?;
        }
        checked_scale(
            &mut gradient[cell],
            limiter,
            format!("cellLimited cell {cell} limited gradient"),
        )?;
    }
    Ok(gradient)
}

fn require_finite(value: f64, context: String) -> Result<f64> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(invalid_input(format!("{context} is non-finite ({value})")))
    }
}

fn require_finite_point(value: Point3, context: String) -> Result<Point3> {
    require_finite(value.x, format!("{context} x component"))?;
    require_finite(value.y, format!("{context} y component"))?;
    require_finite(value.z, format!("{context} z component"))?;
    Ok(value)
}

fn checked_subtraction(left: f64, right: f64, context: String) -> Result<f64> {
    require_finite(left, format!("{context} left operand"))?;
    require_finite(right, format!("{context} right operand"))?;
    require_finite(left - right, context)
}

fn checked_product(left: f64, right: f64, context: String) -> Result<f64> {
    require_finite(left, format!("{context} left operand"))?;
    require_finite(right, format!("{context} right operand"))?;
    require_finite(left * right, context)
}

fn checked_delta(left: Point3, right: Point3, context: String) -> Result<Point3> {
    require_finite_point(left, format!("{context} left point"))?;
    require_finite_point(right, format!("{context} right point"))?;
    Ok(Point3 {
        x: checked_subtraction(left.x, right.x, format!("{context} x component"))?,
        y: checked_subtraction(left.y, right.y, format!("{context} y component"))?,
        z: checked_subtraction(left.z, right.z, format!("{context} z component"))?,
    })
}

fn checked_dot(left: Point3, right: Point3, context: String) -> Result<f64> {
    let x = checked_product(left.x, right.x, format!("{context} x product"))?;
    let y = checked_product(left.y, right.y, format!("{context} y product"))?;
    let z = checked_product(left.z, right.z, format!("{context} z product"))?;
    let xy = require_finite(x + y, format!("{context} x-y sum"))?;
    require_finite(xy + z, context)
}

fn checked_magnitude(value: Point3, context: String) -> Result<f64> {
    require_finite_point(value, context.clone())?;
    require_finite(value.x.hypot(value.y).hypot(value.z), context)
}

fn checked_add_scaled(
    target: &mut Point3,
    value: Point3,
    scale_value: f64,
    face_index: usize,
    cell: usize,
) -> Result<()> {
    let context = format!("scalar gradient face {face_index} cell {cell} accumulation");
    let x = checked_product(value.x, scale_value, format!("{context} x product"))?;
    let y = checked_product(value.y, scale_value, format!("{context} y product"))?;
    let z = checked_product(value.z, scale_value, format!("{context} z product"))?;
    target.x = require_finite(target.x + x, format!("{context} x sum"))?;
    target.y = require_finite(target.y + y, format!("{context} y sum"))?;
    target.z = require_finite(target.z + z, format!("{context} z sum"))?;
    Ok(())
}

fn checked_scale(value: &mut Point3, factor: f64, context: String) -> Result<()> {
    value.x = checked_product(value.x, factor, format!("{context} x component"))?;
    value.y = checked_product(value.y, factor, format!("{context} y component"))?;
    value.z = checked_product(value.z, factor, format!("{context} z component"))?;
    Ok(())
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
            VectorFaceTreatment::InletOutlet(value) => {
                ScalarFaceTreatment::InletOutlet(component_value(*value, component))
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

fn vector_component_gradients(
    mesh: &SolverRuntimeMeshData,
    velocity: &[Point3],
    boundary: &[VectorFaceTreatment],
    scheme: LaminarSimpleGradientScheme,
) -> Result<[Vec<Point3>; 3]> {
    let components = split_components(velocity);
    Ok([
        scalar_gradient(
            mesh,
            &components[0],
            &scalar_component_boundary(boundary, 0),
            scheme,
        )?,
        scalar_gradient(
            mesh,
            &components[1],
            &scalar_component_boundary(boundary, 1),
            scheme,
        )?,
        scalar_gradient(
            mesh,
            &components[2],
            &scalar_component_boundary(boundary, 2),
            scheme,
        )?,
    ])
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
    if !options.pressure_reference_value.is_finite() {
        return Err(invalid_input(format!(
            "laminar SIMPLE pressure reference value must be finite, got {}",
            options.pressure_reference_value
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
    Ok(())
}

fn validate_solver_preconditioner(
    name: &str,
    solver: LaminarSimpleLinearSolver,
    preconditioner: LaminarSimplePreconditioner,
) -> Result<()> {
    if preconditioner != LaminarSimplePreconditioner::None
        && !matches!(
            solver,
            LaminarSimpleLinearSolver::Pcg | LaminarSimpleLinearSolver::BiCgStab
        )
    {
        return Err(invalid_input(format!(
            "laminar SIMPLE {name} preconditioner {preconditioner} requires pcg or bicgstab solver, got {solver}"
        )));
    }
    if preconditioner == LaminarSimplePreconditioner::IncompleteCholesky
        && solver != LaminarSimpleLinearSolver::Pcg
    {
        return Err(invalid_input(format!(
            "laminar SIMPLE {name} incompleteCholesky preconditioner requires pcg on an SPD matrix, got {solver}"
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
    if let Some(value) = value
        && (!value.is_finite() || value < 0.0)
    {
        return Err(invalid_input(format!(
            "{name} must be non-negative and finite, got {value}"
        )));
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
    for patch in &mesh.patches {
        let end_face = patch.start_face.checked_add(patch.faces).ok_or_else(|| {
            invalid_input(format!(
                "runtime mesh patch '{}' face range overflows",
                patch.name
            ))
        })?;
        if patch.start_face < mesh.internal_faces || end_face > mesh.faces {
            return Err(invalid_input(format!(
                "runtime mesh patch '{}' range {}..{} is outside boundary face range {}..{}",
                patch.name, patch.start_face, end_face, mesh.internal_faces, mesh.faces
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
    if !area.is_finite() || area <= f64::EPSILON {
        return Err(invalid_input(format!(
            "face {face_index} has non-positive area magnitude {area}"
        )));
    }
    let delta = Point3 {
        x: to.x - from.x,
        y: to.y - from.y,
        z: to.z - from.z,
    };
    let projected_distance = (dot(delta, area_vector) / area).abs();
    if !projected_distance.is_finite() || projected_distance <= f64::EPSILON {
        return Err(invalid_input(format!(
            "face {face_index} has non-positive projected diffusion distance {projected_distance}"
        )));
    }
    Ok(diffusivity * area / projected_distance)
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

fn fixed_gradient_pressure_flux(
    mesh: &SolverRuntimeMeshData,
    cell_diffusivity: &[f64],
    owner: usize,
    face_index: usize,
    gradient: f64,
) -> Result<f64> {
    if !gradient.is_finite() {
        return Err(invalid_input(format!(
            "fixed-gradient pressure face {face_index} has non-finite gradient {gradient}"
        )));
    }
    let area = magnitude(mesh.face_area_vectors[face_index]);
    if !area.is_finite() || area <= f64::EPSILON {
        return Err(invalid_input(format!(
            "fixed-gradient pressure face {face_index} has non-positive area magnitude {area}"
        )));
    }
    Ok(-cell_diffusivity[owner] * area * gradient)
}

fn boundary_normal_distance(mesh: &SolverRuntimeMeshData, owner: usize, face_index: usize) -> f64 {
    let area = mesh.face_area_vectors[face_index];
    let area_magnitude = magnitude(area);
    if area_magnitude <= f64::EPSILON {
        return distance(mesh.cell_centres[owner], mesh.face_centres[face_index]);
    }
    let centre_to_face = Point3 {
        x: mesh.face_centres[face_index].x - mesh.cell_centres[owner].x,
        y: mesh.face_centres[face_index].y - mesh.cell_centres[owner].y,
        z: mesh.face_centres[face_index].z - mesh.cell_centres[owner].z,
    };
    (dot(centre_to_face, area) / area_magnitude).abs()
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

fn validate_non_negative_cell_values(name: &str, values: &[f64]) -> Result<()> {
    for (index, value) in values.iter().copied().enumerate() {
        if !value.is_finite() || value < 0.0 {
            return Err(invalid_input(format!(
                "{name} value for cell {index} must be non-negative and finite, got {value}"
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
        LaminarSimpleConvectionScheme, LaminarSimpleGradientScheme, LaminarSimpleLinearSolver,
        LaminarSimpleOptions, LaminarSimplePreconditioner, LaminarSimpleSchemes,
        ScalarFaceTreatment, VectorFaceTreatment, adjust_phi_hby_a, apply_pressure_reference,
        assemble_momentum_component_system, assemble_momentum_equation,
        assemble_variable_scalar_component_system, compute_face_flux, compute_phi_hby_a,
        consistent_reciprocal_momentum_diagonal, constrained_pressure_treatments,
        face_diffusion_coefficient, hby_a_from_predicted_velocity, limit_scalar_gradient,
        net_cell_flux, non_orthogonal_pressure_flux_correction, normalized_residual_norm,
        pressure_correction_flux, reciprocal_momentum_diagonal, relax_scalar_component_equation,
        scalar_component_boundary, scalar_gradient, solve_laminar_simple, split_components,
        subtract_face_fluxes, upwind_face_vector_value, vector_face_treatments,
        velocity_from_hby_a,
    };
    use crate::Point3;

    #[test]
    fn vector_momentum_norm_aggregates_residuals_before_normalizing() {
        let residuals = [5.0e-9, 1.0e-7, 1.0e-7];
        let reference_norms = [2.5e-4, 1.0e-8, 1.0e-8];

        let residual_norm = residuals
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt();
        let reference_norm = reference_norms
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt();
        let combined = normalized_residual_norm(residual_norm, reference_norm);
        let componentwise_combined = residuals
            .iter()
            .zip(reference_norms)
            .map(|(residual, reference)| normalized_residual_norm(*residual, reference).powi(2))
            .sum::<f64>()
            .sqrt();

        assert_close(combined, residual_norm / reference_norm);
        assert!(combined < 1.0e-3);
        assert!(componentwise_combined > 10.0);
    }

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
    fn diffusion_coefficient_uses_projected_normal_distance() {
        let coefficient = face_diffusion_coefficient(
            2.0,
            Point3 {
                x: 1.0,
                y: 0.0,
                z: 0.0,
            },
            point(0.0, 0.0, 0.0),
            point(1.0, 1.0, 0.0),
            0,
        )
        .expect("coefficient");

        assert_close(coefficient, 2.0);
    }

    #[test]
    fn gauss_linear_interpolation_uses_projected_geometry_weights() {
        let mut runtime = two_cell_runtime();
        let boundary = vec![
            ScalarFaceTreatment::ZeroGradient,
            ScalarFaceTreatment::ZeroGradient,
            ScalarFaceTreatment::ZeroGradient,
        ];

        let midpoint = scalar_gradient(
            &runtime.mesh,
            &[2.0, 10.0],
            &boundary,
            LaminarSimpleGradientScheme::GaussLinear,
        )
        .expect("midpoint gradient");
        assert_close(midpoint[0].x, 8.0);
        assert_close(midpoint[1].x, 8.0);

        runtime.mesh.face_centres[0].x = 0.4;

        let gradient = scalar_gradient(
            &runtime.mesh,
            &[2.0, 10.0],
            &boundary,
            LaminarSimpleGradientScheme::GaussLinear,
        )
        .expect("geometry-weighted gradient");

        // The internal face value is 0.7*2 + 0.3*10 = 4.4, rather than 6.0.
        assert_close(gradient[0].x, 4.8);
        assert_close(gradient[1].x, 11.2);

        runtime.mesh.face_area_vectors[0] = point(0.0, 1.0, 0.0);
        let euclidean_fallback = scalar_gradient(
            &runtime.mesh,
            &[2.0, 10.0],
            &boundary,
            LaminarSimpleGradientScheme::GaussLinear,
        )
        .expect("Euclidean fallback gradient");
        assert_close(euclidean_fallback[0].y, 8.8);
        assert_close(euclidean_fallback[1].y, -8.8);
    }

    #[test]
    fn cell_limited_gradient_removes_local_overshoot() {
        let mesh = three_cell_line_mesh();
        let boundary = vec![
            ScalarFaceTreatment::ZeroGradient,
            ScalarFaceTreatment::ZeroGradient,
            ScalarFaceTreatment::ZeroGradient,
            ScalarFaceTreatment::ZeroGradient,
        ];
        let values = [0.0, 1.0, 1.0];
        let raw = scalar_gradient(
            &mesh,
            &values,
            &boundary,
            LaminarSimpleGradientScheme::GaussLinear,
        )
        .expect("raw gradient");
        let limited = scalar_gradient(
            &mesh,
            &values,
            &boundary,
            LaminarSimpleGradientScheme::CellLimitedGaussLinear(1.0),
        )
        .expect("limited gradient");

        assert!(raw[1].x > 0.0);
        assert_close(limited[1].x, 0.0);
        let extrapolated = values[1] + limited[1].x * 0.5;
        assert!((0.0..=1.0).contains(&extrapolated));
    }

    #[test]
    fn cell_limited_zero_and_tiny_coefficients_are_stable() {
        let runtime = two_cell_runtime();
        let boundary = vec![
            ScalarFaceTreatment::ZeroGradient,
            ScalarFaceTreatment::ZeroGradient,
            ScalarFaceTreatment::ZeroGradient,
        ];
        let raw = vec![point(3.0, -2.0, 1.0), point(-4.0, 5.0, -6.0)];
        let unchanged =
            limit_scalar_gradient(&runtime.mesh, &[1.0, 1.0], &boundary, raw.clone(), 0.0)
                .expect("k=0 must bypass limiting");
        for (actual, expected) in unchanged.iter().zip(&raw) {
            assert_close(actual.x, expected.x);
            assert_close(actual.y, expected.y);
            assert_close(actual.z, expected.z);
        }

        let tiny = limit_scalar_gradient(
            &runtime.mesh,
            &[1.0, 1.0],
            &boundary,
            raw,
            f64::from_bits(1),
        )
        .expect("subnormal k with zero extrema span must remain finite");
        assert!(
            tiny.iter()
                .all(|value| value.x.is_finite() && value.y.is_finite() && value.z.is_finite())
        );

        let line_mesh = three_cell_line_mesh();
        let line_boundary = vec![ScalarFaceTreatment::ZeroGradient; 4];
        let nearly_one = f64::from_bits(1.0f64.to_bits() - 1);
        let stable_widening = limit_scalar_gradient(
            &line_mesh,
            &[-f64::MAX / 2.0, 0.0, f64::MAX / 2.0],
            &line_boundary,
            vec![point(0.0, 0.0, 0.0); 3],
            nearly_one,
        )
        .expect("finite symmetric widening must not overflow an intermediate quotient");
        assert!(
            stable_widening
                .iter()
                .all(|value| value.x == 0.0 && value.y == 0.0 && value.z == 0.0)
        );
    }

    #[test]
    fn gradient_guards_non_finite_and_overflowing_arithmetic() {
        let runtime = two_cell_runtime();
        let boundary = vec![
            ScalarFaceTreatment::ZeroGradient,
            ScalarFaceTreatment::ZeroGradient,
            ScalarFaceTreatment::ZeroGradient,
        ];
        assert!(
            scalar_gradient(
                &runtime.mesh,
                &[f64::NAN, 0.0],
                &boundary,
                LaminarSimpleGradientScheme::GaussLinear,
            )
            .is_err()
        );

        let mut infinite = two_cell_runtime();
        infinite.mesh.face_centres[0].x = f64::INFINITY;
        assert!(
            scalar_gradient(
                &infinite.mesh,
                &[0.0, 1.0],
                &boundary,
                LaminarSimpleGradientScheme::GaussLinear,
            )
            .is_err()
        );

        let mut subtraction_overflow = two_cell_runtime();
        subtraction_overflow.mesh.face_centres[0].x = f64::MAX;
        subtraction_overflow.mesh.cell_centres[0].x = -f64::MAX;
        assert!(
            scalar_gradient(
                &subtraction_overflow.mesh,
                &[0.0, 1.0],
                &boundary,
                LaminarSimpleGradientScheme::GaussLinear,
            )
            .is_err()
        );

        let widening_overflow = limit_scalar_gradient(
            &runtime.mesh,
            &[-f64::MAX / 2.0, f64::MAX / 2.0],
            &boundary,
            vec![point(0.0, 0.0, 0.0); 2],
            f64::from_bits(1),
        )
        .expect_err("finite extrema whose widening overflows must be rejected");
        assert!(widening_overflow.to_string().contains("widening"));

        let mut extrapolation_mesh = two_cell_runtime().mesh;
        extrapolation_mesh.face_centres[0] = point(1.0, 1.0, 1.0);
        extrapolation_mesh.cell_centres[0] = point(0.0, 0.0, 0.0);
        let extrapolation_overflow = limit_scalar_gradient(
            &extrapolation_mesh,
            &[0.0, 1.0],
            &boundary,
            vec![point(f64::MAX, f64::MAX, f64::MAX), point(0.0, 0.0, 0.0)],
            1.0,
        )
        .expect_err("overflowing extrapolation must be rejected");
        assert!(extrapolation_overflow.to_string().contains("extrapolation"));
    }

    #[test]
    fn phi_hby_a_uses_velocity_boundary_constraints() {
        let runtime = two_cell_runtime();
        let fields = two_cell_fields();
        let u_field = fields
            .fields
            .iter()
            .find(|field| field.name == "U")
            .expect("U field");
        let boundary = vector_face_treatments(&runtime.mesh, u_field).expect("U boundary");
        let hby_a = vec![point(10.0, 0.0, 0.0), point(20.0, 0.0, 0.0)];

        let flux = compute_phi_hby_a(&runtime.mesh, &hby_a, &boundary).expect("phiHbyA");

        assert_close(flux[0], 15.0);
        assert_close(flux[1], -1.0);
        assert_close(flux[2], 20.0);
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
    fn inlet_outlet_velocity_uses_inlet_value_only_for_backflow() {
        let runtime = two_cell_runtime();
        let boundary = vec![
            VectorFaceTreatment::ZeroGradient,
            VectorFaceTreatment::ZeroGradient,
            VectorFaceTreatment::InletOutlet(point(-2.0, 0.0, 0.0)),
        ];
        let velocity = vec![point(5.0, 0.0, 0.0), point(3.0, 0.0, 0.0)];

        let outflow = upwind_face_vector_value(&runtime.mesh, &velocity, &boundary, 2, 1.0);
        let backflow = upwind_face_vector_value(&runtime.mesh, &velocity, &boundary, 2, -1.0);

        assert_close(outflow.x, 3.0);
        assert_close(backflow.x, -2.0);
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
    fn consistent_simple_r_at_u_uses_h1_term() {
        let runtime = two_cell_runtime();
        let r_at_u =
            consistent_reciprocal_momentum_diagonal(&runtime.mesh, &[0.25, 0.125], &[1.0, 2.0])
                .expect("consistent rAtU values");
        let delta = [r_at_u[0] - 0.25, r_at_u[1] - 0.125];

        assert_close(r_at_u[0], 1.0 / 3.0);
        assert_close(r_at_u[1], 1.0 / 6.0);
        assert!(delta[0] > 0.0);
        assert!(delta[1] > 0.0);
    }

    #[test]
    fn hby_a_velocity_correction_matches_pressure_delta_form() {
        let runtime = two_cell_runtime();
        let predicted = vec![point(5.0, 0.0, 0.0), point(3.0, 0.0, 0.0)];
        let old_grad_p = vec![point(2.0, 0.0, 0.0), point(1.0, 0.0, 0.0)];
        let corrected_grad_p = vec![point(3.0, 0.0, 0.0), point(5.0, 0.0, 0.0)];
        let r_at_u = vec![0.25, 0.125];

        let hby_a =
            hby_a_from_predicted_velocity(&predicted, &old_grad_p, &r_at_u).expect("HbyA field");
        let corrected = velocity_from_hby_a(&runtime.mesh, &hby_a, &corrected_grad_p, &r_at_u)
            .expect("corrected velocity");

        assert_close(corrected[0].x, 5.0 - 0.25 * (3.0 - 2.0));
        assert_close(corrected[1].x, 3.0 - 0.125 * (5.0 - 1.0));
        assert_close(hby_a[0].x, 5.5);
        assert_close(hby_a[1].x, 3.125);
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
        let old_values = vec![5.0, 3.0];

        let system = assemble_momentum_component_system(
            &runtime.mesh,
            1.0,
            2.0,
            &flux,
            &source,
            &boundary,
            &old_values,
            None,
            LaminarSimpleConvectionScheme::GaussUpwind,
        )
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
    fn linear_upwind_adds_deferred_momentum_correction() {
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
        let old_values = vec![5.0, 3.0];
        let old_gradient = vec![point(4.0, 0.0, 0.0), point(4.0, 0.0, 0.0)];

        let upwind = assemble_momentum_component_system(
            &runtime.mesh,
            1.0,
            2.0,
            &flux,
            &source,
            &boundary,
            &old_values,
            None,
            LaminarSimpleConvectionScheme::GaussUpwind,
        )
        .expect("upwind momentum system");
        let linear_upwind = assemble_momentum_component_system(
            &runtime.mesh,
            1.0,
            2.0,
            &flux,
            &source,
            &boundary,
            &old_values,
            Some(&old_gradient),
            LaminarSimpleConvectionScheme::GaussLinearUpwind,
        )
        .expect("linearUpwind momentum system");

        assert_eq!(
            upwind.matrix.row_offsets(),
            linear_upwind.matrix.row_offsets()
        );
        assert_eq!(
            upwind.matrix.col_indices(),
            linear_upwind.matrix.col_indices()
        );
        assert_close(linear_upwind.rhs[0], upwind.rhs[0] - 2.0);
        assert_close(linear_upwind.rhs[1], upwind.rhs[1] - 4.0);
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
        let old_values = vec![5.0, 3.0];
        let mut system = assemble_momentum_component_system(
            &runtime.mesh,
            1.0,
            2.0,
            &flux,
            &source,
            &boundary,
            &old_values,
            None,
            LaminarSimpleConvectionScheme::GaussUpwind,
        )
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
    fn assembles_momentum_equation_with_component_systems_and_h1() {
        let runtime = two_cell_runtime();
        let fields = two_cell_fields();
        let u_field = fields
            .fields
            .iter()
            .find(|field| field.name == "U")
            .expect("U field");
        let vector_boundary = vector_face_treatments(&runtime.mesh, u_field).expect("U boundary");
        let velocity = vec![point(5.0, 0.0, 0.0), point(3.0, 0.0, 0.0)];
        let old_components = split_components(&velocity);
        let flux = vec![1.0, -2.0, 3.0];
        let grad_p = vec![point(2.0, 0.0, 0.0), point(1.0, 0.0, 0.0)];
        let options = minimal_laminar_options();

        let equation = assemble_momentum_equation(
            &runtime.mesh,
            &vector_boundary,
            &flux,
            &grad_p,
            &old_components,
            &options,
        )
        .expect("momentum equation");

        assert_eq!(equation.components.len(), 3);
        assert_eq!(equation.diagonal.len(), runtime.mesh.cells);
        assert_eq!(equation.h1.len(), runtime.mesh.cells);
        assert!(equation.diagonal_min > 0.0);
        assert!(equation.diagonal_max >= equation.diagonal_min);
        assert!(equation.h1_max >= equation.h1_min);
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
    fn pressure_correction_flux_accepts_zero_consistent_delta() {
        let runtime = two_cell_runtime();
        let boundary = vec![ScalarFaceTreatment::ZeroGradient; runtime.mesh.faces];
        let flux = pressure_correction_flux(&runtime.mesh, &[2.0, 0.0], &[0.0, 0.0], &boundary)
            .expect("zero consistent delta flux");

        for value in flux {
            assert_close(value, 0.0);
        }
    }

    #[test]
    fn pressure_equation_flux_matches_openfoam_phi_correction_sign() {
        let runtime = two_cell_runtime();
        let boundary = vec![ScalarFaceTreatment::ZeroGradient; runtime.mesh.faces];
        let r_au = vec![1.0, 1.0];
        let pressure = vec![1.0, 0.0];

        let correction_flux = pressure_correction_flux(&runtime.mesh, &pressure, &r_au, &boundary)
            .expect("pressure correction flux");
        let phi_hby_a = correction_flux.iter().map(|flux| -flux).collect::<Vec<_>>();
        let p_eqn_flux = super::pressure_equation_flux(&runtime.mesh, &pressure, &r_au, &boundary)
            .expect("pEqn flux");
        let corrected_with_openfoam_form =
            subtract_face_fluxes(&phi_hby_a, &p_eqn_flux).expect("corrected phi");
        let corrected_with_balance_form =
            super::add_face_fluxes(&phi_hby_a, &correction_flux).expect("corrected phi");

        for (openfoam_form, balance_form) in corrected_with_openfoam_form
            .iter()
            .zip(corrected_with_balance_form)
        {
            assert_close(*openfoam_form, balance_form);
        }
        for value in net_cell_flux(&runtime.mesh, &corrected_with_openfoam_form)
            .expect("corrected continuity")
        {
            assert_close(value, 0.0);
        }
    }

    #[test]
    fn consistent_simple_pressure_correction_uses_r_at_u_delta() {
        let runtime = two_cell_runtime();
        let boundary = vec![ScalarFaceTreatment::ZeroGradient; runtime.mesh.faces];
        let pressure = vec![1.0, 0.0];
        let r_au = vec![1.0, 1.0];
        let r_at_u = vec![1.5, 1.5];

        let correction = super::consistent_phi_hby_a_pressure_correction(
            &runtime.mesh,
            &pressure,
            &boundary,
            &r_au,
            &r_at_u,
        )
        .expect("consistent phi correction");
        let direct_delta_flux =
            pressure_correction_flux(&runtime.mesh, &pressure, &[0.5, 0.5], &boundary)
                .expect("delta pressure flux");

        for (correction, direct) in correction.iter().zip(direct_delta_flux) {
            assert_close(*correction, -direct);
        }
    }

    #[test]
    fn consistent_simple_hby_a_matches_openfoam_update() {
        let hby_a = vec![point(4.0, 0.0, 0.0), point(2.0, 0.0, 0.0)];
        let grad_p = vec![point(3.0, 0.0, 0.0), point(5.0, 0.0, 0.0)];
        let r_au = vec![1.0, 0.5];
        let r_at_u = vec![1.5, 0.75];

        let consistent = super::consistent_hby_a_from_base(&hby_a, &grad_p, &r_au, &r_at_u)
            .expect("consistent HbyA");

        assert_close(consistent[0].x, 5.5);
        assert_close(consistent[1].x, 3.25);
    }

    #[test]
    fn adjust_phi_balances_pressure_open_boundary_only() {
        let runtime = two_cell_runtime();
        let fields = two_cell_fields();
        let u_field = fields
            .fields
            .iter()
            .find(|field| field.name == "U")
            .expect("U field");
        let velocity_boundary = vector_face_treatments(&runtime.mesh, u_field).expect("U boundary");
        let mut pressure_boundary = vec![ScalarFaceTreatment::ZeroGradient; runtime.mesh.faces];
        pressure_boundary[2] = ScalarFaceTreatment::FixedValue(0.0);
        let mut phi_hby_a = vec![0.5, -1.0, 0.0];

        let summary = adjust_phi_hby_a(
            &runtime.mesh,
            &velocity_boundary,
            &pressure_boundary,
            &mut phi_hby_a,
        )
        .expect("adjust phi");

        assert_eq!(summary.adjusted_faces, 1);
        assert_close(summary.global_flux_before, -1.0);
        assert_close(summary.global_flux_after, 0.0);
        assert_close(phi_hby_a[0], 0.5);
        assert_close(phi_hby_a[1], -1.0);
        assert_close(phi_hby_a[2], 1.0);
    }

    #[test]
    fn adjust_phi_does_not_change_fixed_velocity_open_boundary() {
        let runtime = two_cell_runtime();
        let mut velocity_boundary = vec![VectorFaceTreatment::ZeroGradient; runtime.mesh.faces];
        velocity_boundary[2] = VectorFaceTreatment::FixedValue(point(1.0, 0.0, 0.0));
        let mut pressure_boundary = vec![ScalarFaceTreatment::ZeroGradient; runtime.mesh.faces];
        pressure_boundary[2] = ScalarFaceTreatment::FixedValue(0.0);
        let mut phi_hby_a = vec![0.0, 0.0, 1.0];

        let summary = adjust_phi_hby_a(
            &runtime.mesh,
            &velocity_boundary,
            &pressure_boundary,
            &mut phi_hby_a,
        )
        .expect("adjust phi");

        assert_eq!(summary.adjusted_faces, 0);
        assert_close(summary.global_flux_before, 1.0);
        assert_close(summary.global_flux_after, 1.0);
        assert_close(phi_hby_a[2], 1.0);
    }

    #[test]
    fn adjust_phi_uses_inlet_outlet_only_for_outflow() {
        let runtime = two_cell_runtime();
        let mut velocity_boundary = vec![VectorFaceTreatment::ZeroGradient; runtime.mesh.faces];
        velocity_boundary[2] = VectorFaceTreatment::InletOutlet(point(-2.0, 0.0, 0.0));
        let mut pressure_boundary = vec![ScalarFaceTreatment::ZeroGradient; runtime.mesh.faces];
        pressure_boundary[2] = ScalarFaceTreatment::FixedValue(0.0);
        let mut outflow_phi = vec![0.0, 0.0, 1.0];

        let outflow_summary = adjust_phi_hby_a(
            &runtime.mesh,
            &velocity_boundary,
            &pressure_boundary,
            &mut outflow_phi,
        )
        .expect("adjust outflow");
        assert_eq!(outflow_summary.adjusted_faces, 1);
        assert_close(outflow_summary.global_flux_after, 0.0);

        let mut backflow_phi = vec![0.0, 0.0, -1.0];
        let backflow_summary = adjust_phi_hby_a(
            &runtime.mesh,
            &velocity_boundary,
            &pressure_boundary,
            &mut backflow_phi,
        )
        .expect("adjust backflow");
        assert_eq!(backflow_summary.adjusted_faces, 0);
        assert_close(backflow_summary.global_flux_after, -1.0);
    }

    #[test]
    fn non_orthogonal_pressure_flux_is_zero_on_orthogonal_two_cell_mesh() {
        let runtime = two_cell_runtime();
        let boundary = vec![ScalarFaceTreatment::ZeroGradient; runtime.mesh.faces];
        let flux = non_orthogonal_pressure_flux_correction(
            &runtime.mesh,
            &[1.0, 0.0],
            &[1.0, 1.0],
            &boundary,
            LaminarSimpleGradientScheme::GaussLinear,
        )
        .expect("non-orthogonal correction flux");

        assert_eq!(flux.len(), runtime.mesh.faces);
        for value in flux {
            assert_close(value, 0.0);
        }
    }

    #[test]
    fn pressure_reference_anchors_closed_pressure_system() {
        let runtime = two_cell_runtime();
        let boundary = vec![ScalarFaceTreatment::ZeroGradient; runtime.mesh.faces];
        let r_au = vec![1.0, 1.0];
        let source = vec![0.0, 0.0];
        let mut system =
            assemble_variable_scalar_component_system(&runtime.mesh, &r_au, &source, &boundary)
                .expect("closed pressure system");
        let mut options = minimal_laminar_options();
        options.pressure_reference_cell = Some(1);
        options.pressure_reference_value = 7.0;

        apply_pressure_reference(&mut system, &runtime.mesh, &boundary, &options)
            .expect("pressure reference");
        let solution = system.matrix.matvec(&[7.0, 7.0]).expect("matvec");

        assert_close(solution[0], system.rhs[0]);
        assert_close(solution[1], system.rhs[1]);
    }

    #[test]
    fn fixed_flux_pressure_gradient_makes_pressure_flux_match_boundary_u() {
        let runtime = two_cell_runtime();
        let fields = two_cell_fields();
        let u_field = fields
            .fields
            .iter()
            .find(|field| field.name == "U")
            .expect("U field");
        let velocity_boundary = vector_face_treatments(&runtime.mesh, u_field).expect("U boundary");
        let velocity = vec![point(0.0, 0.0, 0.0), point(0.0, 0.0, 0.0)];
        let mut pressure_boundary = vec![ScalarFaceTreatment::ZeroGradient; runtime.mesh.faces];
        pressure_boundary[1] = ScalarFaceTreatment::FixedGradient(0.0);
        let phi_hby_a = vec![0.0, -0.25, 0.0];
        let r_au = vec![2.0, 2.0];

        let constrained = constrained_pressure_treatments(
            &runtime.mesh,
            &pressure_boundary,
            &velocity_boundary,
            &velocity,
            &phi_hby_a,
            &r_au,
        )
        .expect("constrained pressure");
        let pressure_flux =
            pressure_correction_flux(&runtime.mesh, &[0.0, 0.0], &r_au, &constrained)
                .expect("pressure flux");

        assert_close(phi_hby_a[1] + pressure_flux[1], -1.0);
    }

    #[test]
    fn zero_gradient_pressure_on_fixed_velocity_boundary_is_constrained() {
        let runtime = two_cell_runtime();
        let mut velocity_boundary = vec![VectorFaceTreatment::ZeroGradient; runtime.mesh.faces];
        velocity_boundary[1] = VectorFaceTreatment::FixedValue(point(1.0, 0.0, 0.0));
        let pressure_boundary = vec![ScalarFaceTreatment::ZeroGradient; runtime.mesh.faces];
        let velocity = vec![point(0.0, 0.0, 0.0), point(0.0, 0.0, 0.0)];
        let phi_hby_a = vec![0.0, -0.25, 0.0];
        let r_au = vec![2.0, 2.0];

        let constrained = constrained_pressure_treatments(
            &runtime.mesh,
            &pressure_boundary,
            &velocity_boundary,
            &velocity,
            &phi_hby_a,
            &r_au,
        )
        .expect("constrained pressure");
        let pressure_flux =
            pressure_correction_flux(&runtime.mesh, &[0.0, 0.0], &r_au, &constrained)
                .expect("pressure flux");

        assert!(matches!(
            constrained[1],
            ScalarFaceTreatment::FixedGradient(_)
        ));
        assert_close(phi_hby_a[1] + pressure_flux[1], -1.0);
    }

    #[test]
    fn zero_gradient_pressure_on_open_outflow_remains_unconstrained() {
        let runtime = two_cell_runtime();
        let mut velocity_boundary = vec![VectorFaceTreatment::ZeroGradient; runtime.mesh.faces];
        velocity_boundary[2] = VectorFaceTreatment::InletOutlet(point(-2.0, 0.0, 0.0));
        let pressure_boundary = vec![ScalarFaceTreatment::ZeroGradient; runtime.mesh.faces];
        let velocity = vec![point(0.0, 0.0, 0.0), point(1.0, 0.0, 0.0)];
        let phi_hby_a = vec![0.0, 0.0, 1.0];
        let r_au = vec![1.0, 1.0];

        let constrained = constrained_pressure_treatments(
            &runtime.mesh,
            &pressure_boundary,
            &velocity_boundary,
            &velocity,
            &phi_hby_a,
            &r_au,
        )
        .expect("constrained pressure");

        assert!(matches!(constrained[2], ScalarFaceTreatment::ZeroGradient));
    }

    #[test]
    fn vector_treatments_accept_pressure_inlet_outlet_velocity() {
        let runtime = two_cell_runtime();
        let fields = two_cell_fields_with_pressure_inlet_outlet_velocity();
        let u_field = fields
            .fields
            .iter()
            .find(|field| field.name == "U")
            .expect("U field");
        let velocity_boundary = vector_face_treatments(&runtime.mesh, u_field)
            .expect("pressureInletOutletVelocity boundary");

        assert!(matches!(
            velocity_boundary[2],
            VectorFaceTreatment::InletOutlet(_)
        ));
    }

    #[test]
    fn pressure_inlet_outlet_velocity_alias_is_backflow_sensitive_for_pressure_constraint() {
        let runtime = two_cell_runtime();
        let fields = two_cell_fields_with_pressure_inlet_outlet_velocity();
        let u_field = fields
            .fields
            .iter()
            .find(|field| field.name == "U")
            .expect("U field");
        let velocity_boundary = vector_face_treatments(&runtime.mesh, u_field)
            .expect("pressureInletOutletVelocity boundary");
        let velocity = vec![point(0.0, 0.0, 0.0), point(0.0, 0.0, 0.0)];

        let mut pressure_boundary = vec![ScalarFaceTreatment::ZeroGradient; runtime.mesh.faces];
        pressure_boundary[2] = ScalarFaceTreatment::ZeroGradient;
        let phi_inflow = vec![0.0, 0.0, -1.0];
        let phi_outflow = vec![0.0, 0.0, 1.0];
        let r_au = vec![1.0, 1.0];

        let constrained_inflow = constrained_pressure_treatments(
            &runtime.mesh,
            &pressure_boundary,
            &velocity_boundary,
            &velocity,
            &phi_inflow,
            &r_au,
        )
        .expect("constrained pressure (inflow)");
        let constrained_outflow = constrained_pressure_treatments(
            &runtime.mesh,
            &pressure_boundary,
            &velocity_boundary,
            &velocity,
            &phi_outflow,
            &r_au,
        )
        .expect("constrained pressure (outflow)");

        assert!(matches!(
            constrained_inflow[2],
            ScalarFaceTreatment::FixedGradient(_)
        ));
        assert!(matches!(
            constrained_outflow[2],
            ScalarFaceTreatment::ZeroGradient
        ));
    }

    #[test]
    fn inlet_outlet_pressure_inflow_is_constrained() {
        let runtime = two_cell_runtime();
        let mut velocity_boundary = vec![VectorFaceTreatment::ZeroGradient; runtime.mesh.faces];
        velocity_boundary[2] = VectorFaceTreatment::FixedValue(point(-2.0, 0.0, 0.0));
        let mut pressure_boundary = vec![ScalarFaceTreatment::ZeroGradient; runtime.mesh.faces];
        pressure_boundary[2] = ScalarFaceTreatment::InletOutlet(0.0);
        let velocity = vec![point(0.0, 0.0, 0.0), point(0.0, 0.0, 0.0)];
        let phi_hby_a = vec![0.0, 0.0, -1.0];
        let r_au = vec![1.0, 1.0];

        let constrained = constrained_pressure_treatments(
            &runtime.mesh,
            &pressure_boundary,
            &velocity_boundary,
            &velocity,
            &phi_hby_a,
            &r_au,
        )
        .expect("constrained pressure");

        assert!(matches!(
            constrained[2],
            ScalarFaceTreatment::FixedGradient(_)
        ));
    }

    #[test]
    fn inlet_outlet_pressure_outflow_remains_unconstrained() {
        let runtime = two_cell_runtime();
        let mut velocity_boundary = vec![VectorFaceTreatment::ZeroGradient; runtime.mesh.faces];
        velocity_boundary[2] = VectorFaceTreatment::FixedValue(point(-2.0, 0.0, 0.0));
        let mut pressure_boundary = vec![ScalarFaceTreatment::ZeroGradient; runtime.mesh.faces];
        pressure_boundary[2] = ScalarFaceTreatment::InletOutlet(0.0);
        let velocity = vec![point(0.0, 0.0, 0.0), point(0.0, 0.0, 0.0)];
        let phi_hby_a = vec![0.0, 0.0, 1.0];
        let r_au = vec![1.0, 1.0];

        let constrained = constrained_pressure_treatments(
            &runtime.mesh,
            &pressure_boundary,
            &velocity_boundary,
            &velocity,
            &phi_hby_a,
            &r_au,
        )
        .expect("constrained pressure");

        assert!(matches!(
            constrained[2],
            ScalarFaceTreatment::InletOutlet(_)
        ));
    }

    #[test]
    fn runs_minimal_simple_loop_on_two_cells() {
        let runtime = two_cell_runtime();
        let fields = two_cell_fields();
        let options = minimal_laminar_options();

        let report = solve_laminar_simple(&runtime, &fields, &options).expect("simple report");

        assert_eq!(report.cells, 2);
        assert!(report.simple_iterations > 0);
        assert!(report.fields.velocity.l2_norm.is_finite());
        assert!(report.fields.pressure.l2_norm.is_finite());
        assert!(report.final_continuity.l2_norm.is_finite());
        assert!(report.operator_summary.hby_a_l2_norm.is_finite());
        assert_eq!(report.final_velocity.len(), runtime.mesh.cells);
        assert_eq!(report.final_pressure.len(), runtime.mesh.cells);
    }

    #[test]
    fn laminar_simple_converged_requires_residual_controls() {
        let mut options = minimal_laminar_options();
        options.momentum_residual_control = Some(1.0e-6);
        options.pressure_residual_control = Some(1.0e-6);
        let satisfied =
            super::evaluate_laminar_simple_residual_control(Some(1.0e-7), Some(1.0e-7), &options);

        assert!(satisfied.checked);
        assert!(satisfied.satisfied);
        assert!(super::laminar_simple_converged(2, satisfied, &options));

        let pressure_not_satisfied =
            super::evaluate_laminar_simple_residual_control(Some(1.0e-7), Some(1.0e-5), &options);
        assert!(!pressure_not_satisfied.satisfied);
        assert!(!super::laminar_simple_converged(
            2,
            pressure_not_satisfied,
            &options,
        ));

        options.momentum_residual_control = None;
        options.pressure_residual_control = None;
        let not_configured =
            super::evaluate_laminar_simple_residual_control(Some(0.0), Some(0.0), &options);
        assert!(!not_configured.configured);
        assert!(!super::laminar_simple_converged(
            2,
            not_configured,
            &options,
        ));
    }

    #[test]
    fn residual_control_uses_strict_openfoam_absolute_tolerance() {
        let mut options = minimal_laminar_options();
        options.momentum_residual_control = Some(1.0e-3);

        let equal = super::evaluate_laminar_simple_residual_control(Some(1.0e-3), None, &options);
        let below = super::evaluate_laminar_simple_residual_control(Some(9.9e-4), None, &options);

        assert_eq!(equal.momentum_satisfied, Some(false));
        assert_eq!(below.momentum_satisfied, Some(true));
    }

    #[test]
    fn openfoam_residual_normalisation_matches_reference_formula() {
        let matrix = crate::linear::CsrMatrix::from_rows(
            vec![vec![(0, 2.0), (1, -1.0)], vec![(0, -1.0), (1, 2.0)]],
            2,
        )
        .expect("matrix");
        let source = [1.0, 0.0];
        let solution = [0.0, 0.0];
        let matrix_solution = matrix.matvec(&solution).expect("matrix product");

        let factor =
            super::openfoam_normalisation_factor(&matrix, &source, &solution, &matrix_solution)
                .expect("normalisation factor");
        let residual = source
            .iter()
            .zip(matrix_solution)
            .map(|(source, matrix_value)| source - matrix_value)
            .collect::<Vec<_>>();

        assert_close(factor, 1.0);
        assert_close(super::l1_norm(&residual) / factor, 1.0);
    }

    fn minimal_laminar_options() -> LaminarSimpleOptions {
        LaminarSimpleOptions {
            density: 1.0,
            dynamic_viscosity: 1.0,
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
            momentum_residual_control: None,
            pressure_residual_control: None,
            pressure_reference_cell: None,
            pressure_reference_value: 0.0,
            non_orthogonal_correctors: 0,
            simple_consistent: false,
            velocity_relaxation: 0.7,
            pressure_relaxation: 0.3,
            schemes: LaminarSimpleSchemes::default(),
        }
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

    fn three_cell_line_mesh() -> SolverRuntimeMeshData {
        SolverRuntimeMeshData {
            points: 0,
            cells: 3,
            faces: 4,
            internal_faces: 2,
            boundary_faces: 2,
            owner: vec![0, 1, 0, 2],
            neighbour: vec![Some(1), Some(2), None, None],
            patches: Vec::new(),
            face_centres: vec![
                point(1.0, 0.0, 0.0),
                point(2.0, 0.0, 0.0),
                point(0.0, 0.0, 0.0),
                point(3.0, 0.0, 0.0),
            ],
            face_area_vectors: vec![
                point(1.0, 0.0, 0.0),
                point(1.0, 0.0, 0.0),
                point(-1.0, 0.0, 0.0),
                point(1.0, 0.0, 0.0),
            ],
            cell_centres: vec![
                point(0.5, 0.0, 0.0),
                point(1.5, 0.0, 0.0),
                point(2.5, 0.0, 0.0),
            ],
            cell_volumes: vec![1.0; 3],
            min_face_area: 1.0,
            max_face_area: 1.0,
            min_cell_volume: 1.0,
            max_cell_volume: 1.0,
            total_cell_volume: 3.0,
            non_positive_cell_volumes: 0,
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
                            inlet_value: None,
                            value: Some(FieldValueSummary::Uniform("(1 0 0)".to_string())),
                        },
                        FieldBoundaryPatch {
                            name: "outlet".to_string(),
                            patch_type: Some("zeroGradient".to_string()),
                            inlet_value: None,
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
                            inlet_value: None,
                            value: None,
                        },
                        FieldBoundaryPatch {
                            name: "outlet".to_string(),
                            patch_type: Some("fixedValue".to_string()),
                            inlet_value: None,
                            value: Some(FieldValueSummary::Uniform("0".to_string())),
                        },
                    ],
                },
            ],
        }
    }

    fn two_cell_fields_with_pressure_inlet_outlet_velocity() -> InitialFieldSet {
        let mut fields = two_cell_fields();
        let fields_u = fields
            .fields
            .iter()
            .position(|field| field.name == "U")
            .expect("U field");
        let modified_u = fields.fields.get_mut(fields_u).expect("U field");
        modified_u.boundary_patches[1].patch_type = Some("pressureInletOutletVelocity".to_string());
        modified_u.boundary_patches[1].inlet_value =
            Some(FieldValueSummary::Uniform("(0.5 0 0)".to_string()));
        modified_u.boundary_patches[1].value =
            Some(FieldValueSummary::Uniform("(1 0 0)".to_string()));
        fields
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
