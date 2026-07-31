use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};
use std::thread::{self, JoinHandle};
use std::time::Instant;

use crate::fields::{FieldFile, FieldValueSummary, InitialFieldSet};
use crate::linear::gamg::{NormalizedL1GamgSolveControls, OPENFOAM_RELATIVE_TOLERANCE_SMALL};
use crate::linear::{
    BiCgStabOptions, CgPreconditioner, ConjugateGradientOptions, CsrMatrix, CsrSparsityPattern,
    GamgAgglomerator, GamgFacePairWeight, GamgKernelTiming, GamgOptions, GamgSolveControls,
    GamgWorkspace, GaussSeidelOptions, IterativeSolveTermination, JacobiOptions,
    NormalizedL1PcgSolveControls, PcgKernelTiming, PreconditionedConjugateGradientWorkspace,
    PreparedPcgInitialTiming, PreparedPcgResidual, bicgstab_solve, conjugate_gradient_solve,
    gauss_seidel_solve, jacobi_solve, l2_norm, symmetric_gauss_seidel_solve,
};
use crate::runtime::{SolverRuntimeData, SolverRuntimeMeshData};
use crate::solver_state::SolverStateFieldKind;
use crate::{MeshError, Point3, Result};

mod compact_faces;

#[path = "gradient.rs"]
mod gradient;

use compact_faces::CompactSimpleFaceAddressing;
use gradient::{
    ScalarGradientGeometry, WeightedLeastSquaresGeometry, gauss_linear_owner_weight,
    scalar_gradient_with_scheme_and_addressing,
    vector_component_gradients_with_scheme_and_addressing,
};

#[cfg(test)]
use gradient::{
    limit_scalar_gradient_with_addressing, scalar_gradient_with_geometry_and_addressing,
};

#[derive(Clone, Copy, Debug)]
struct SimpleUpdateMetrics {
    velocity_change_l2: f64,
    pressure_change_l2: f64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LaminarSimpleLinearSolver {
    BiCgStab,
    Cg,
    Gamg,
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
    Dic,
    Fdic,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum LaminarSimpleMomentumExecution {
    #[default]
    Serial,
    ParallelComponents,
}

#[derive(Clone, Copy, Debug)]
pub enum LaminarSimpleGradientScheme {
    GaussLinear,
    LeastSquares,
    CellLimitedGaussLinear(f64),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LaminarSimpleConvectionScheme {
    GaussUpwind,
    GaussLinearUpwind,
    BoundedGaussLinearUpwind(LaminarSimpleGradientScheme),
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
            (Self::LeastSquares, Self::LeastSquares) => true,
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

/// Cell-centred gradients reconstructed from the loaded laminar initial fields.
///
/// `velocity_component_gradients[0]`, `[1]`, and `[2]` are the gradients of
/// `Ux`, `Uy`, and `Uz`, respectively. They are present only when the selected
/// convection scheme consumes `grad(U)`.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct LaminarInitialGradientProbe {
    pub pressure_gradient: Vec<Point3>,
    pub velocity_component_gradients: Option<[Vec<Point3>; 3]>,
}

/// Cell-centred gradients reconstructed from an explicit laminar field state.
///
/// `velocity_component_gradients[0]`, `[1]`, and `[2]` are the gradients of
/// `Ux`, `Uy`, and `Uz`, respectively. Unlike [`LaminarInitialGradientProbe`],
/// this post-processing result always contains `grad(U)`, including when the
/// convection scheme itself is upwind and does not consume a velocity
/// gradient.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct LaminarFieldGradients {
    pub pressure_gradient: Vec<Point3>,
    pub velocity_component_gradients: [Vec<Point3>; 3],
}

pub const MAX_NON_ORTHOGONAL_CORRECTORS: usize = 20;
pub const MAX_PRESSURE_CORRECTION_LINEAR_SOLVES: usize = MAX_NON_ORTHOGONAL_CORRECTORS + 1;

#[derive(Clone, Debug)]
pub struct LaminarSimpleOptions {
    pub density: f64,
    pub dynamic_viscosity: f64,
    pub linear_solver: LaminarSimpleLinearSolver,
    pub momentum_linear_solver: LaminarSimpleLinearSolver,
    pub momentum_execution: LaminarSimpleMomentumExecution,
    pub pressure_linear_solver: LaminarSimpleLinearSolver,
    pub momentum_preconditioner: LaminarSimplePreconditioner,
    pub pressure_preconditioner: LaminarSimplePreconditioner,
    pub pressure_gamg_options: Option<GamgOptions>,
    pub profile_gamg: bool,
    pub linear_tolerance: f64,
    pub max_linear_iterations: usize,
    pub momentum_linear_tolerance: f64,
    pub pressure_linear_tolerance: f64,
    pub momentum_linear_relative_tolerance: f64,
    pub pressure_linear_relative_tolerance: f64,
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
    pub timing: LaminarSimpleTimingSummary,
    pub fields: LaminarSimpleFieldSummary,
    pub final_velocity: Vec<Point3>,
    pub final_pressure: Vec<f64>,
    #[cfg(test)]
    pub final_phi: Vec<f64>,
    pub history: Vec<LaminarSimpleIterationSummary>,
}

#[derive(Clone, Debug, Default)]
pub struct LaminarSimpleTimingSummary {
    pub solver_total_seconds: f64,
    pub setup_seconds: f64,
    pub iteration_setup_seconds: f64,
    pub operator_evaluation_seconds: f64,
    pub momentum_assembly_seconds: f64,
    pub momentum_gradient_seconds: f64,
    pub momentum_matrix_fill_seconds: f64,
    pub momentum_linear_solve_seconds: f64,
    pub pressure_coupling_setup_seconds: f64,
    pub pressure_assembly_seconds: f64,
    pub pressure_linear_solve_seconds: f64,
    pub pressure_pcg_total_seconds: f64,
    pub pressure_preconditioner_update_seconds: f64,
    pub pressure_matrix_vector_seconds: f64,
    pub pressure_preconditioner_application_seconds: f64,
    pub pressure_vector_operation_seconds: f64,
    pub pressure_pcg_other_seconds: f64,
    pub pressure_matrix_vector_products: usize,
    pub pressure_preconditioner_applications: usize,
    pub pressure_gamg_profile: Option<GamgKernelTiming>,
    pub field_correction_seconds: f64,
    pub finalization_seconds: f64,
    pub other_solver_work_seconds: f64,
}

#[derive(Default)]
struct LaminarSimpleTimingAccumulator {
    setup_seconds: f64,
    iteration_setup_seconds: f64,
    operator_evaluation_seconds: f64,
    momentum_assembly_seconds: f64,
    momentum_gradient_seconds: f64,
    momentum_matrix_fill_seconds: f64,
    momentum_linear_solve_seconds: f64,
    pressure_coupling_setup_seconds: f64,
    pressure_assembly_seconds: f64,
    pressure_linear_solve_seconds: f64,
    pressure_pcg_total_seconds: f64,
    pressure_preconditioner_update_seconds: f64,
    pressure_matrix_vector_seconds: f64,
    pressure_preconditioner_application_seconds: f64,
    pressure_vector_operation_seconds: f64,
    pressure_pcg_other_seconds: f64,
    pressure_matrix_vector_products: usize,
    pressure_preconditioner_applications: usize,
    pressure_gamg_profile: Option<GamgKernelTiming>,
    field_correction_seconds: f64,
    finalization_seconds: f64,
}

impl LaminarSimpleTimingAccumulator {
    fn finish(self, solver_total_seconds: f64) -> LaminarSimpleTimingSummary {
        let accounted_seconds = self.setup_seconds
            + self.iteration_setup_seconds
            + self.operator_evaluation_seconds
            + self.momentum_assembly_seconds
            + self.momentum_linear_solve_seconds
            + self.pressure_coupling_setup_seconds
            + self.pressure_assembly_seconds
            + self.pressure_linear_solve_seconds
            + self.field_correction_seconds
            + self.finalization_seconds;
        LaminarSimpleTimingSummary {
            solver_total_seconds,
            setup_seconds: self.setup_seconds,
            iteration_setup_seconds: self.iteration_setup_seconds,
            operator_evaluation_seconds: self.operator_evaluation_seconds,
            momentum_assembly_seconds: self.momentum_assembly_seconds,
            momentum_gradient_seconds: self.momentum_gradient_seconds,
            momentum_matrix_fill_seconds: self.momentum_matrix_fill_seconds,
            momentum_linear_solve_seconds: self.momentum_linear_solve_seconds,
            pressure_coupling_setup_seconds: self.pressure_coupling_setup_seconds,
            pressure_assembly_seconds: self.pressure_assembly_seconds,
            pressure_linear_solve_seconds: self.pressure_linear_solve_seconds,
            pressure_pcg_total_seconds: self.pressure_pcg_total_seconds,
            pressure_preconditioner_update_seconds: self.pressure_preconditioner_update_seconds,
            pressure_matrix_vector_seconds: self.pressure_matrix_vector_seconds,
            pressure_preconditioner_application_seconds: self
                .pressure_preconditioner_application_seconds,
            pressure_vector_operation_seconds: self.pressure_vector_operation_seconds,
            pressure_pcg_other_seconds: self.pressure_pcg_other_seconds,
            pressure_matrix_vector_products: self.pressure_matrix_vector_products,
            pressure_preconditioner_applications: self.pressure_preconditioner_applications,
            pressure_gamg_profile: self.pressure_gamg_profile,
            field_correction_seconds: self.field_correction_seconds,
            finalization_seconds: self.finalization_seconds,
            other_solver_work_seconds: (solver_total_seconds - accounted_seconds).max(0.0),
        }
    }

    fn add_pressure_pcg_timing(&mut self, timing: PcgKernelTiming) {
        self.pressure_pcg_total_seconds += timing.total_seconds;
        self.pressure_preconditioner_update_seconds += timing.preconditioner_update_seconds;
        self.pressure_matrix_vector_seconds += timing.matrix_vector_seconds;
        self.pressure_preconditioner_application_seconds +=
            timing.preconditioner_application_seconds;
        self.pressure_vector_operation_seconds += timing.vector_operation_seconds;
        self.pressure_pcg_other_seconds += timing.other_seconds;
        self.pressure_matrix_vector_products += timing.matrix_vector_products;
        self.pressure_preconditioner_applications += timing.preconditioner_applications;
    }

    fn add_pressure_gamg_timing(&mut self, timing: GamgKernelTiming) -> Result<()> {
        if let Some(profile) = &mut self.pressure_gamg_profile {
            profile.accumulate(&timing)
        } else {
            self.pressure_gamg_profile = Some(timing);
            Ok(())
        }
    }

    fn add_pressure_gamg_hierarchy_build(&mut self, seconds: f64) {
        let profile = self
            .pressure_gamg_profile
            .get_or_insert_with(GamgKernelTiming::default);
        profile.add_hierarchy_build(seconds);
    }
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

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum LinearSolveStopReason {
    #[default]
    NotRun,
    ExactZero,
    AbsoluteTolerance,
    RelativeTolerance,
    MaxIterations,
    Breakdown,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct LinearSolveConvergenceSummary {
    pub iterations: usize,
    pub converged: bool,
    pub initial_normalized_residual_norm: f64,
    pub residual_norm: f64,
    pub normalized_residual_norm: f64,
    pub effective_normalized_tolerance: f64,
    pub stop_reason: LinearSolveStopReason,
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
    pub momentum_component_linear_solves: [LinearSolveConvergenceSummary; 3],
    pub pressure_correction_linear_solves:
        [LinearSolveConvergenceSummary; MAX_PRESSURE_CORRECTION_LINEAR_SOLVES],
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

fn flow_try_vec<T>(capacity: usize) -> Result<Vec<T>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(capacity)
        .map_err(|_| MeshError::OutOfMemory)?;
    Ok(values)
}

fn flow_try_filled_vec<T: Clone>(length: usize, value: T) -> Result<Vec<T>> {
    let mut values = flow_try_vec(length)?;
    values.resize(length, value);
    Ok(values)
}

fn resolve_pressure_inlet_outlet(
    boundary: &[ScalarFaceTreatment],
    flux: &[f64],
) -> Result<Vec<ScalarFaceTreatment>> {
    if boundary.len() != flux.len() {
        return Err(invalid_input(format!(
            "pressure inletOutlet resolution requires one flux per face, got {} treatments and {} fluxes",
            boundary.len(),
            flux.len()
        )));
    }
    for (face_index, (treatment, face_flux)) in boundary.iter().zip(flux).enumerate() {
        if matches!(treatment, ScalarFaceTreatment::InletOutlet(_)) && !face_flux.is_finite() {
            return Err(invalid_input(format!(
                "pressure inletOutlet face {face_index} has non-finite decision flux {face_flux}"
            )));
        }
    }
    let mut resolved = flow_try_vec(boundary.len())?;
    for (treatment, face_flux) in boundary.iter().zip(flux) {
        resolved.push(match *treatment {
            ScalarFaceTreatment::InletOutlet(value) if *face_flux < 0.0 => {
                ScalarFaceTreatment::FixedValue(value)
            }
            ScalarFaceTreatment::InletOutlet(_) => ScalarFaceTreatment::ZeroGradient,
            treatment => treatment,
        });
    }
    Ok(resolved)
}

impl std::fmt::Display for LaminarSimpleLinearSolver {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BiCgStab => formatter.write_str("bicgstab"),
            Self::Cg => formatter.write_str("cg"),
            Self::Gamg => formatter.write_str("GAMG"),
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
            Self::Dic => formatter.write_str("DIC"),
            Self::Fdic => formatter.write_str("FDIC"),
        }
    }
}

impl std::fmt::Display for LaminarSimpleMomentumExecution {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Serial => formatter.write_str("serial"),
            Self::ParallelComponents => formatter.write_str("parallel-components"),
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
            Self::LeastSquares => formatter.write_str("leastSquares"),
            Self::CellLimitedGaussLinear(coefficient) => {
                write!(formatter, "cellLimited Gauss linear {coefficient}")
            }
        }
    }
}

impl LaminarSimpleGradientScheme {
    fn uses_weighted_least_squares(self) -> bool {
        matches!(self, Self::LeastSquares)
    }
}

impl std::fmt::Display for LaminarSimpleConvectionScheme {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::GaussUpwind => formatter.write_str("Gauss upwind"),
            Self::GaussLinearUpwind => formatter.write_str("Gauss linearUpwind grad(U)"),
            Self::BoundedGaussLinearUpwind(_) => {
                formatter.write_str("bounded Gauss linearUpwind limited")
            }
        }
    }
}

impl LaminarSimpleConvectionScheme {
    fn uses_linear_upwind(self) -> bool {
        matches!(
            self,
            Self::GaussLinearUpwind | Self::BoundedGaussLinearUpwind(_)
        )
    }

    fn is_bounded(self) -> bool {
        matches!(self, Self::BoundedGaussLinearUpwind(_))
    }

    fn gradient_scheme(self, fallback: LaminarSimpleGradientScheme) -> LaminarSimpleGradientScheme {
        match self {
            Self::BoundedGaussLinearUpwind(scheme) => scheme,
            _ => fallback,
        }
    }
}

impl LaminarSimpleSchemes {
    fn uses_weighted_least_squares(self) -> bool {
        self.grad_p.uses_weighted_least_squares()
            || (self.div_phi_u.uses_linear_upwind()
                && self
                    .div_phi_u
                    .gradient_scheme(self.grad_u)
                    .uses_weighted_least_squares())
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

impl std::fmt::Display for LinearSolveStopReason {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotRun => formatter.write_str("NotRun"),
            Self::ExactZero => formatter.write_str("ExactZero"),
            Self::AbsoluteTolerance => formatter.write_str("AbsoluteTolerance"),
            Self::RelativeTolerance => formatter.write_str("RelativeTolerance"),
            Self::MaxIterations => formatter.write_str("MaxIterations"),
            Self::Breakdown => formatter.write_str("Breakdown"),
        }
    }
}

/// Reconstructs `grad(p)` and `grad(U)` from the loaded initial fields without
/// consuming or mutating their one-shot runtime payloads.
///
/// The probe uses the same field-boundary resolution, face-flux construction,
/// geometry caches, and gradient dispatchers as [`solve_laminar_simple`]. It is
/// intended for diagnostics and cross-engine validation; it does not assemble
/// equations or advance SIMPLE.
pub fn reconstruct_laminar_initial_gradients(
    runtime: &SolverRuntimeData,
    fields: &InitialFieldSet,
    schemes: LaminarSimpleSchemes,
) -> Result<LaminarInitialGradientProbe> {
    validate_runtime_mesh(&runtime.mesh)?;

    let velocity_field = find_field(fields, "U", "volVectorField")?;
    let pressure_field = find_field(fields, "p", "volScalarField")?;
    let velocity_boundary = vector_face_treatments(&runtime.mesh, velocity_field)?;
    let pressure_boundary = scalar_face_treatments(&runtime.mesh, pressure_field)?;

    let mesh_cache = LaminarSimpleMeshCache::from_mesh(&runtime.mesh, schemes)?;
    let (velocity, pressure) = snapshot_runtime_initial_fields(runtime)?;
    let flux = compute_face_flux_with_addressing(
        &runtime.mesh,
        &mesh_cache.face_addressing,
        &velocity,
        &velocity_boundary,
    )?;
    let continuity = summarize_continuity(&net_cell_flux_with_addressing(
        &runtime.mesh,
        &mesh_cache.face_addressing,
        &flux,
    )?);
    checked_update_metrics(&velocity, pressure, &flux, continuity).ok_or_else(|| {
        invalid_input(
            "laminar initial-gradient probe field metrics exceed the finite numeric range"
                .to_string(),
        )
    })?;
    let resolved_pressure_boundary = resolve_pressure_inlet_outlet(&pressure_boundary, &flux)?;

    let pressure_gradient = scalar_gradient_with_scheme_and_addressing(
        &runtime.mesh,
        &mesh_cache.scalar_gradient,
        mesh_cache.weighted_least_squares.as_ref(),
        &mesh_cache.face_addressing,
        pressure,
        &resolved_pressure_boundary,
        schemes.grad_p,
    )?;
    let velocity_component_gradients = if schemes.div_phi_u.uses_linear_upwind() {
        Some(vector_component_gradients_with_scheme_and_addressing(
            &runtime.mesh,
            &mesh_cache.scalar_gradient,
            mesh_cache.weighted_least_squares.as_ref(),
            &mesh_cache.face_addressing,
            &velocity,
            &velocity_boundary,
            &flux,
            schemes.div_phi_u.gradient_scheme(schemes.grad_u),
        )?)
    } else {
        None
    };

    Ok(LaminarInitialGradientProbe {
        pressure_gradient,
        velocity_component_gradients,
    })
}

/// Reconstructs `grad(p)` and `grad(U)` from an explicit cell-centred field
/// state without consuming or mutating the runtime or field definitions.
///
/// This is the post-processing counterpart of
/// [`reconstruct_laminar_initial_gradients`]. A caller may pass the final
/// fields from [`LaminarSimpleReport`] after [`solve_laminar_simple`] has
/// consumed the runtime's one-shot initial payloads. Boundary treatments,
/// face fluxes, and every gradient scheme are resolved through the same
/// production dispatchers used by the solver.
///
/// The weighted-least-squares geometry needed only by this request is built
/// locally. Selecting `leastSquares` for `grad(U)` while convection remains
/// upwind therefore adds no setup or execution work to the SIMPLE hot path.
pub fn reconstruct_laminar_gradients_from_fields(
    runtime: &SolverRuntimeData,
    field_definitions: &InitialFieldSet,
    velocity: &[Point3],
    pressure: &[f64],
    schemes: LaminarSimpleSchemes,
) -> Result<LaminarFieldGradients> {
    validate_runtime_mesh(&runtime.mesh)?;
    if velocity.len() != runtime.mesh.cells || pressure.len() != runtime.mesh.cells {
        return Err(invalid_input(format!(
            "laminar gradient reconstruction expected {} cell values, got U={} p={}",
            runtime.mesh.cells,
            velocity.len(),
            pressure.len()
        )));
    }

    let velocity_field = find_field(field_definitions, "U", "volVectorField")?;
    let pressure_field = find_field(field_definitions, "p", "volScalarField")?;
    let velocity_boundary = vector_face_treatments(&runtime.mesh, velocity_field)?;
    let pressure_boundary = scalar_face_treatments(&runtime.mesh, pressure_field)?;

    let scalar_gradient = ScalarGradientGeometry::from_mesh(&runtime.mesh)?;
    let face_addressing = CompactSimpleFaceAddressing::from_mesh(&runtime.mesh)?;
    let weighted_least_squares = (schemes.grad_p.uses_weighted_least_squares()
        || schemes.grad_u.uses_weighted_least_squares())
    .then(|| {
        WeightedLeastSquaresGeometry::from_mesh(&runtime.mesh, &scalar_gradient, &face_addressing)
    })
    .transpose()?;

    let flux = compute_face_flux_with_addressing(
        &runtime.mesh,
        &face_addressing,
        velocity,
        &velocity_boundary,
    )?;
    let continuity = summarize_continuity(&net_cell_flux_with_addressing(
        &runtime.mesh,
        &face_addressing,
        &flux,
    )?);
    checked_update_metrics(velocity, pressure, &flux, continuity).ok_or_else(|| {
        invalid_input(
            "laminar gradient reconstruction field metrics exceed the finite numeric range"
                .to_string(),
        )
    })?;
    let resolved_pressure_boundary = resolve_pressure_inlet_outlet(&pressure_boundary, &flux)?;

    let pressure_gradient = scalar_gradient_with_scheme_and_addressing(
        &runtime.mesh,
        &scalar_gradient,
        weighted_least_squares.as_ref(),
        &face_addressing,
        pressure,
        &resolved_pressure_boundary,
        schemes.grad_p,
    )?;
    let velocity_component_gradients = vector_component_gradients_with_scheme_and_addressing(
        &runtime.mesh,
        &scalar_gradient,
        weighted_least_squares.as_ref(),
        &face_addressing,
        velocity,
        &velocity_boundary,
        &flux,
        schemes.grad_u,
    )?;

    Ok(LaminarFieldGradients {
        pressure_gradient,
        velocity_component_gradients,
    })
}

/// Runs the laminar SIMPLE solver from the initial `U` and `p` payloads stored
/// in `runtime`.
///
/// # One-shot initial payloads
///
/// After setup has validated both fields, this call takes ownership of their
/// runtime payloads before the first solver iteration. Those payloads are not
/// restored, including when a later solver operation returns an error. A
/// second call with the same `SolverRuntimeData` therefore fails
/// deterministically because the initial payloads have already been consumed.
/// Build a fresh solver plan and runtime for each retry or parameter-sweep run.
pub fn solve_laminar_simple(
    runtime: &mut SolverRuntimeData,
    fields: &InitialFieldSet,
    options: &LaminarSimpleOptions,
) -> Result<LaminarSimpleReport> {
    solve_laminar_simple_with_observer(runtime, fields, options, None)
}

/// Runs the laminar SIMPLE solver and reports each completed iteration to an
/// optional observer.
///
/// This is the observer-enabled form of [`solve_laminar_simple`] and has the
/// same one-shot ownership contract: after setup validates `U` and `p`, their
/// runtime payloads are consumed before the first iteration and are not
/// restored after a later error. Use a fresh plan and runtime for every retry
/// or parameter-sweep run.
pub fn solve_laminar_simple_with_observer(
    runtime: &mut SolverRuntimeData,
    fields: &InitialFieldSet,
    options: &LaminarSimpleOptions,
    on_iteration: Option<&mut dyn FnMut(&LaminarSimpleIterationSummary)>,
) -> Result<LaminarSimpleReport> {
    #[cfg(test)]
    return solve_laminar_simple_driven(runtime, fields, options, on_iteration, false, None, None);

    #[cfg(not(test))]
    solve_laminar_simple_driven(runtime, fields, options, on_iteration, false)
}

/// Runs the laminar SIMPLE solver with pressure-PCG kernel profiling enabled.
///
/// This diagnostic entry point executes the same solver operations as
/// [`solve_laminar_simple`] and differs only by collecting per-phase PCG clock
/// readings and counters. It requires the pressure solver to be `PCG`.
pub fn solve_laminar_simple_profiled_pcg(
    runtime: &mut SolverRuntimeData,
    fields: &InitialFieldSet,
    options: &LaminarSimpleOptions,
) -> Result<LaminarSimpleReport> {
    solve_laminar_simple_profiled_pcg_with_observer(runtime, fields, options, None)
}

/// Observer-enabled form of [`solve_laminar_simple_profiled_pcg`].
pub fn solve_laminar_simple_profiled_pcg_with_observer(
    runtime: &mut SolverRuntimeData,
    fields: &InitialFieldSet,
    options: &LaminarSimpleOptions,
    on_iteration: Option<&mut dyn FnMut(&LaminarSimpleIterationSummary)>,
) -> Result<LaminarSimpleReport> {
    #[cfg(test)]
    return solve_laminar_simple_driven(runtime, fields, options, on_iteration, true, None, None);

    #[cfg(not(test))]
    solve_laminar_simple_driven(runtime, fields, options, on_iteration, true)
}

#[cfg(test)]
type PressureReportDriver<'a> =
    &'a mut dyn FnMut(usize, &mut ScalarSolveReport, &[Point3], &[f64], ContinuitySummary);

#[cfg(test)]
type PressureBoundaryDriver<'a> =
    &'a mut dyn FnMut(usize, &mut Vec<f64>, Option<&[ScalarFaceTreatment]>);

fn solve_laminar_simple_driven(
    runtime: &mut SolverRuntimeData,
    fields: &InitialFieldSet,
    options: &LaminarSimpleOptions,
    mut on_iteration: Option<&mut dyn FnMut(&LaminarSimpleIterationSummary)>,
    profile_pcg: bool,
    #[cfg(test)] mut drive_pressure_report: Option<PressureReportDriver<'_>>,
    #[cfg(test)] mut drive_pressure_boundary: Option<PressureBoundaryDriver<'_>>,
) -> Result<LaminarSimpleReport> {
    let solver_started = Instant::now();
    let setup_started = Instant::now();
    validate_laminar_simple_options(options)?;
    if profile_pcg && options.pressure_linear_solver != LaminarSimpleLinearSolver::Pcg {
        return Err(invalid_input(
            "laminar SIMPLE PCG profiling requires the PCG pressure solver".to_string(),
        ));
    }
    validate_runtime_mesh(&runtime.mesh)?;

    let velocity_field = find_field(fields, "U", "volVectorField")?;
    let pressure_field = find_field(fields, "p", "volScalarField")?;
    let velocity_boundary = vector_face_treatments(&runtime.mesh, velocity_field)?;
    let pressure_boundary = scalar_face_treatments(&runtime.mesh, pressure_field)?;
    let boundary_summary =
        summarize_boundaries(&runtime.mesh, &velocity_boundary, &pressure_boundary);
    let mesh_cache = LaminarSimpleMeshCache::from_mesh(&runtime.mesh, options.schemes)?;
    let mut pressure_system = ScalarComponentSystem {
        matrix: CsrMatrix::from_pattern(
            &mesh_cache.momentum.sparsity,
            vec![0.0; mesh_cache.momentum.sparsity.nnz()],
        )?,
        rhs: vec![0.0; runtime.mesh.cells],
    };
    let mut pressure_pcg_workspace =
        if options.pressure_linear_solver == LaminarSimpleLinearSolver::Pcg {
            Some(PreconditionedConjugateGradientWorkspace::new(
                &pressure_system.matrix,
                map_cg_preconditioner(options.pressure_preconditioner),
            )?)
        } else {
            None
        };
    let mut pressure_gamg_workspace = None;
    let mut scalar_solve_workspace = ScalarSolveWorkspace::new(runtime.mesh.cells);
    let mut momentum_component_executor = match options.momentum_execution {
        LaminarSimpleMomentumExecution::Serial => None,
        LaminarSimpleMomentumExecution::ParallelComponents => {
            Some(MomentumComponentExecutor::new(runtime.mesh.cells)?)
        }
    };

    let (mut velocity, mut pressure) = take_runtime_initial_fields(runtime)?;
    let initial_phi = compute_face_flux_with_addressing(
        &runtime.mesh,
        &mesh_cache.face_addressing,
        &velocity,
        &velocity_boundary,
    )?;
    let initial_continuity = summarize_continuity(&net_cell_flux_with_addressing(
        &runtime.mesh,
        &mesh_cache.face_addressing,
        &initial_phi,
    )?);
    checked_update_metrics(&velocity, &pressure, &initial_phi, initial_continuity).ok_or_else(
        || {
            invalid_input(
                "laminar SIMPLE initial field metrics exceed the finite numeric range".to_string(),
            )
        },
    )?;
    let mut timing = LaminarSimpleTimingAccumulator {
        setup_seconds: setup_started.elapsed().as_secs_f64(),
        ..LaminarSimpleTimingAccumulator::default()
    };

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
    let initial_pressure_boundary =
        resolve_pressure_inlet_outlet(&pressure_boundary, &surface_flux)?;
    let mut final_grad_p = scalar_gradient_with_scheme_and_addressing(
        &runtime.mesh,
        &mesh_cache.scalar_gradient,
        mesh_cache.weighted_least_squares.as_ref(),
        &mesh_cache.face_addressing,
        &pressure,
        &initial_pressure_boundary,
        options.schemes.grad_p,
    )?;
    let mut final_hby_a = vec![zero(); runtime.mesh.cells];
    let mut final_pressure_assembly = None;
    let mut stop_reason = None;
    let mut emit_iteration = |summary: LaminarSimpleIterationSummary| {
        history.push(summary);
        if let Some(observer) = on_iteration.as_deref_mut() {
            observer(&summary);
        }
    };

    for iteration in 1..=options.max_simple_iterations {
        let accepted_final_residuals = (
            final_momentum_initial_normalized_residual_norm,
            final_momentum_residual_norm,
            final_momentum_normalized_residual_norm,
            final_pressure_correction_initial_normalized_residual_norm,
            final_pressure_correction_residual_norm,
            final_pressure_correction_normalized_residual_norm,
        );
        let iteration_setup_started = Instant::now();
        let previous_velocity = velocity.clone();
        let previous_pressure = pressure.clone();
        let phi = surface_flux.clone();
        let carried_pressure_boundary = resolve_pressure_inlet_outlet(&pressure_boundary, &phi)?;
        let continuity_before = summarize_continuity(&net_cell_flux_with_addressing(
            &runtime.mesh,
            &mesh_cache.face_addressing,
            &phi,
        )?);
        timing.iteration_setup_seconds += iteration_setup_started.elapsed().as_secs_f64();
        let operator_evaluation_started = Instant::now();
        let grad_p = scalar_gradient_with_scheme_and_addressing(
            &runtime.mesh,
            &mesh_cache.scalar_gradient,
            mesh_cache.weighted_least_squares.as_ref(),
            &mesh_cache.face_addressing,
            &pressure,
            &carried_pressure_boundary,
            options.schemes.grad_p,
        )?;
        timing.operator_evaluation_seconds += operator_evaluation_started.elapsed().as_secs_f64();
        let momentum = solve_momentum_predictor(
            &runtime.mesh,
            &mesh_cache,
            MomentumPredictorFields {
                velocity: &velocity,
                velocity_boundary: &velocity_boundary,
                flux: &phi,
                grad_p: &grad_p,
            },
            options,
            &mut scalar_solve_workspace,
            momentum_component_executor.as_mut(),
        )?;
        timing.momentum_assembly_seconds += momentum.assembly_seconds;
        timing.momentum_gradient_seconds += momentum.gradient_seconds;
        timing.momentum_matrix_fill_seconds += momentum.matrix_fill_seconds;
        timing.momentum_linear_solve_seconds += momentum.linear_solve_seconds;
        if !momentum.residual_norm.is_finite() || !points_are_finite(&momentum.velocity) {
            final_phi = phi;
            final_continuity = continuity_before;
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
                momentum_component_linear_solves: momentum.component_linear_solves,
                pressure_correction_linear_solves: [LinearSolveConvergenceSummary::default();
                    MAX_PRESSURE_CORRECTION_LINEAR_SOLVES],
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
                momentum_component_linear_solves: momentum.component_linear_solves,
                pressure_correction_linear_solves: [LinearSolveConvergenceSummary::default();
                    MAX_PRESSURE_CORRECTION_LINEAR_SOLVES],
            });
            stop_reason = Some(LaminarSimpleStopReason::MomentumSolverInvalidState);
            break;
        }

        let pressure_coupling_setup_started = Instant::now();
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
        let mut phi_hby_a = compute_phi_hby_a_with_addressing(
            &runtime.mesh,
            &mesh_cache.face_addressing,
            &hby_a,
            &velocity_boundary,
        )?;
        #[cfg(test)]
        if let Some(driver) = drive_pressure_boundary.as_deref_mut() {
            driver(iteration, &mut phi_hby_a, None);
        }
        let coupling_pressure_boundary =
            resolve_pressure_inlet_outlet(&pressure_boundary, &phi_hby_a)?;
        #[cfg(test)]
        if let Some(driver) = drive_pressure_boundary.as_deref_mut() {
            driver(iteration, &mut phi_hby_a, Some(&coupling_pressure_boundary));
        }
        let phi_hby_a_before_adjust_summary = summarize_face_fluxes(&runtime.mesh, &phi_hby_a);
        let adjust_phi_summary = adjust_phi_hby_a_with_pressure_geometry_and_addressing(
            &runtime.mesh,
            &mesh_cache.pressure_geometry,
            &mesh_cache.face_addressing,
            &velocity_boundary,
            &coupling_pressure_boundary,
            &mut phi_hby_a,
        )?;
        let phi_hby_a_after_adjust_summary = summarize_face_fluxes(&runtime.mesh, &phi_hby_a);
        if options.simple_consistent {
            let consistent_phi_correction =
                consistent_phi_hby_a_pressure_correction_with_geometry_and_addressing(
                    &runtime.mesh,
                    &mesh_cache.pressure_geometry,
                    &mesh_cache.face_addressing,
                    &pressure,
                    &coupling_pressure_boundary,
                    &r_au,
                    &r_at_u,
                )?;
            phi_hby_a = add_face_fluxes(&phi_hby_a, &consistent_phi_correction)?;
            hby_a = consistent_hby_a_from_base(&hby_a, &grad_p, &r_au, &r_at_u)?;
        }
        let net_flux_star =
            net_cell_flux_with_addressing(&runtime.mesh, &mesh_cache.face_addressing, &phi_hby_a)?;
        let continuity_star = summarize_continuity(&net_flux_star);
        if !is_finite_continuity(continuity_star) {
            final_phi = phi;
            final_continuity = continuity_before;
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
                momentum_component_linear_solves: momentum.component_linear_solves,
                pressure_correction_linear_solves: [LinearSolveConvergenceSummary::default();
                    MAX_PRESSURE_CORRECTION_LINEAR_SOLVES],
            });
            stop_reason = Some(LaminarSimpleStopReason::SolverInvalidState);
            break;
        }
        let constrained_pressure_boundary =
            constrained_pressure_treatments_with_geometry_and_addressing(
                &runtime.mesh,
                &mesh_cache.pressure_geometry,
                &mesh_cache.face_addressing,
                &coupling_pressure_boundary,
                &velocity_boundary,
                &predicted_velocity,
                &phi_hby_a,
                &r_at_u,
            )?;
        let mut pressure_report = None;
        let mut first_pressure_initial_normalized_residual_norm = None;
        let mut pressure_linear_iterations_this_simple = 0;
        let mut pressure_linear_converged_this_simple = true;
        let mut pressure_linear_solves_this_simple = 0;
        let mut pressure_linear_non_converged_solves_this_simple = 0;
        let mut pressure_correction_linear_solves =
            [LinearSolveConvergenceSummary::default(); MAX_PRESSURE_CORRECTION_LINEAR_SOLVES];
        let mut pressure_source_summary = ScalarDiagnosticSummary::default();
        let mut pressure_equation_flux_summary = FaceFluxDiagnosticSummary::default();
        let mut pressure_matrix_summary = MatrixDiagnosticSummary::default();
        let mut last_pressure_equation_flux = None;
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
        timing.pressure_coupling_setup_seconds +=
            pressure_coupling_setup_started.elapsed().as_secs_f64();
        for pressure_solve_index in 0..pressure_solve_count {
            let pressure_assembly_started = Instant::now();
            let initial_pressure = pressure_report
                .as_ref()
                .map(|report: &ScalarSolveReport| report.solution.as_slice())
                .unwrap_or(&pressure);
            let pressure_equation_flux =
                if apply_non_orthogonal_correction && options.non_orthogonal_correctors > 0 {
                    let non_orthogonal_flux =
                        non_orthogonal_pressure_flux_correction_with_geometry_and_addressing(
                            &runtime.mesh,
                            &mesh_cache.pressure_geometry,
                            &mesh_cache.scalar_gradient,
                            mesh_cache.weighted_least_squares.as_ref(),
                            &mesh_cache.face_addressing,
                            initial_pressure,
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
                &net_cell_flux_with_addressing(
                    &runtime.mesh,
                    &mesh_cache.face_addressing,
                    &pressure_equation_flux,
                )?,
            )?;
            pressure_source_summary = summarize_scalars(&pressure_source);
            assemble_variable_scalar_component_system_into_with_pressure_geometry_and_addressing(
                &runtime.mesh,
                &mesh_cache.momentum,
                &mesh_cache.pressure_geometry,
                &mesh_cache.face_addressing,
                &r_at_u,
                &pressure_source,
                &constrained_pressure_boundary,
                &mut pressure_system,
            )?;
            apply_pressure_reference(
                &mut pressure_system,
                &runtime.mesh,
                &constrained_pressure_boundary,
                options,
            )?;
            pressure_matrix_summary = summarize_csr_matrix(&pressure_system.matrix)?;
            pressure_linear_solves_this_simple += 1;
            timing.pressure_assembly_seconds += pressure_assembly_started.elapsed().as_secs_f64();
            let pressure_solve_started = Instant::now();
            if options.pressure_linear_solver == LaminarSimpleLinearSolver::Gamg
                && pressure_gamg_workspace.is_none()
            {
                let hierarchy_build_started = options.profile_gamg.then(Instant::now);
                let gamg_options = options.pressure_gamg_options.ok_or_else(|| {
                    invalid_input(
                        "laminar SIMPLE pressure GAMG requires resolved GAMG options".to_string(),
                    )
                })?;
                pressure_gamg_workspace = Some(match gamg_options.agglomerator {
                    GamgAgglomerator::AlgebraicPair => {
                        GamgWorkspace::new(&pressure_system.matrix, gamg_options)?
                    }
                    GamgAgglomerator::FaceAreaPair => GamgWorkspace::new_with_face_area_weights(
                        &pressure_system.matrix,
                        gamg_options,
                        &mesh_cache.gamg_face_area_weights,
                    )?,
                });
                if let Some(started) = hierarchy_build_started {
                    timing.add_pressure_gamg_hierarchy_build(started.elapsed().as_secs_f64());
                }
            }
            let pressure_solve_result = solve_scalar_system_with_workspaces(
                &pressure_system.matrix,
                &pressure_system.rhs,
                Some(initial_pressure),
                ScalarSolveControls {
                    solver: options.pressure_linear_solver,
                    preconditioner: options.pressure_preconditioner,
                    tolerance: options.pressure_linear_tolerance,
                    relative_tolerance: options.pressure_linear_relative_tolerance,
                    max_iterations: options.pressure_max_linear_iterations,
                    gamg_options: options.pressure_gamg_options,
                    profile_gamg: options.profile_gamg,
                    profile_pcg,
                },
                &mut scalar_solve_workspace,
                pressure_pcg_workspace.as_mut(),
                pressure_gamg_workspace.as_mut(),
            );
            timing.pressure_linear_solve_seconds += pressure_solve_started.elapsed().as_secs_f64();
            let mut report = match pressure_solve_result {
                Ok(report) => report,
                Err(error) => {
                    return Err(invalid_input(format!(
                        "laminar SIMPLE pressure correction solve failed: {error}"
                    )));
                }
            };
            #[cfg(test)]
            if let Some(driver) = drive_pressure_report.as_deref_mut() {
                driver(
                    pressure_linear_solves_this_simple,
                    &mut report,
                    &predicted_velocity,
                    &phi_hby_a,
                    continuity_star,
                );
            }
            if let Some(pcg_timing) = report.pcg_timing {
                timing.add_pressure_pcg_timing(pcg_timing);
            }
            if let Some(gamg_timing) = report.gamg_timing.take() {
                timing.add_pressure_gamg_timing(gamg_timing)?;
            }
            pressure_correction_linear_solves[pressure_solve_index] =
                LinearSolveConvergenceSummary::from(&report);
            if let Some(pressure_stop_reason) = pressure_solver_stop_reason(report.termination) {
                pressure_linear_converged_this_simple = false;
                pressure_linear_non_converged_solves_this_simple += 1;
                total_pressure_linear_iterations += report.iterations;
                pressure_linear_iterations_this_simple += report.iterations;
                final_pressure_correction_initial_normalized_residual_norm =
                    report.initial_normalized_residual_norm;
                final_pressure_correction_residual_norm = report.residual_norm;
                final_pressure_correction_normalized_residual_norm =
                    report.normalized_residual_norm;
                final_phi = phi.clone();
                final_continuity = continuity_before;
                emit_iteration(LaminarSimpleIterationSummary {
                    iteration,
                    continuity_before,
                    continuity_after: final_continuity,
                    pressure_correction_accepted: false,
                    momentum_linear_iterations: momentum.iterations,
                    momentum_linear_converged: momentum.converged,
                    momentum_component_linear_converged: momentum.component_converged,
                    pressure_linear_iterations: pressure_linear_iterations_this_simple,
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
                    pressure_correction_initial_normalized_residual_norm: report
                        .initial_normalized_residual_norm,
                    pressure_correction_residual_norm: report.residual_norm,
                    pressure_correction_normalized_residual_norm: report.normalized_residual_norm,
                    residual_control: evaluate_laminar_simple_residual_control(
                        Some(momentum.initial_normalized_residual_norm),
                        Some(report.initial_normalized_residual_norm),
                        options,
                    ),
                    relative_velocity_change_l2: 0.0,
                    relative_pressure_change_l2: 0.0,
                    momentum_update_scale,
                    pressure_correction_update_scale: 0.0,
                    adjust_phi_global_flux_before: adjust_phi_summary.global_flux_before,
                    adjust_phi_global_flux_after: adjust_phi_summary.global_flux_after,
                    adjust_phi_adjusted_faces: adjust_phi_summary.adjusted_faces,
                    momentum_component_linear_solves: momentum.component_linear_solves,
                    pressure_correction_linear_solves,
                });
                stop_reason = Some(pressure_stop_reason);
                pressure_report = None;
                break;
            }
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
            last_pressure_equation_flux = Some(pressure_equation_flux);
            pressure_report = Some(report);
        }
        let Some(pressure_report) = pressure_report else {
            break;
        };
        let final_pressure_equation_flux = last_pressure_equation_flux.ok_or_else(|| {
            invalid_input(
                "laminar SIMPLE completed a pressure solve without retaining its equation flux"
                    .to_string(),
            )
        })?;
        let field_correction_started = Instant::now();
        let pressure_initial_normalized_residual_norm =
            first_pressure_initial_normalized_residual_norm
                .unwrap_or(pressure_report.initial_normalized_residual_norm);
        final_pressure_correction_initial_normalized_residual_norm =
            pressure_initial_normalized_residual_norm;
        let pressure_solution_is_representable =
            checked_l2_norm(pressure_report.solution.iter().copied()).is_some();
        let pressure_solution = if pressure_solution_is_representable {
            pressure_report.solution.as_slice()
        } else {
            pressure.as_slice()
        };
        let pressure_delta = pressure_solution
            .iter()
            .zip(&pressure)
            .map(|(after, before)| options.pressure_relaxation * (after - before))
            .collect::<Vec<_>>();
        let mut corrected_pressure = pressure.clone();
        for (value, delta) in corrected_pressure.iter_mut().zip(&pressure_delta) {
            *value += delta;
        }
        let corrected_pressure_gradient = scalar_gradient_with_scheme_and_addressing(
            &runtime.mesh,
            &mesh_cache.scalar_gradient,
            mesh_cache.weighted_least_squares.as_ref(),
            &mesh_cache.face_addressing,
            &corrected_pressure,
            &constrained_pressure_boundary,
            options.schemes.grad_p,
        )?;
        let corrected_velocity =
            velocity_from_hby_a(&runtime.mesh, &hby_a, &corrected_pressure_gradient, &r_at_u)?;

        let pressure_flux = pressure_equation_flux_with_geometry_and_addressing(
            &runtime.mesh,
            &mesh_cache.pressure_geometry,
            &mesh_cache.face_addressing,
            pressure_solution,
            &r_at_u,
            &constrained_pressure_boundary,
        )?;
        let corrected_phi =
            corrected_pressure_equation_flux(&final_pressure_equation_flux, &pressure_flux)?;
        let pressure_assembly = PressureAssemblyDiagnostics {
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
        };
        let corrected_continuity = summarize_continuity(&net_cell_flux_with_addressing(
            &runtime.mesh,
            &mesh_cache.face_addressing,
            &corrected_phi,
        )?);
        let pressure_correction_update_scale = 1.0;
        let candidate_update_metrics = pressure_solution_is_representable
            .then(|| {
                checked_candidate_update_metrics(
                    &previous_velocity,
                    &previous_pressure,
                    &phi,
                    &corrected_velocity,
                    &corrected_pressure,
                    &corrected_phi,
                    corrected_continuity,
                )
            })
            .flatten();
        if candidate_update_metrics.is_none()
            || !is_finite_continuity(corrected_continuity)
            || !points_are_finite(&corrected_velocity)
            || !scalars_are_finite(&corrected_pressure)
            || !corrected_phi.iter().all(|value| value.is_finite())
        {
            velocity = previous_velocity;
            pressure = previous_pressure;
            (
                final_momentum_initial_normalized_residual_norm,
                final_momentum_residual_norm,
                final_momentum_normalized_residual_norm,
                final_pressure_correction_initial_normalized_residual_norm,
                final_pressure_correction_residual_norm,
                final_pressure_correction_normalized_residual_norm,
            ) = accepted_final_residuals;
            final_phi = phi;
            final_continuity = continuity_before;
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
                relative_velocity_change_l2: 0.0,
                relative_pressure_change_l2: 0.0,
                momentum_update_scale,
                pressure_correction_update_scale: 0.0,
                adjust_phi_global_flux_before: adjust_phi_summary.global_flux_before,
                adjust_phi_global_flux_after: adjust_phi_summary.global_flux_after,
                adjust_phi_adjusted_faces: adjust_phi_summary.adjusted_faces,
                momentum_component_linear_solves: momentum.component_linear_solves,
                pressure_correction_linear_solves,
            });
            stop_reason = Some(LaminarSimpleStopReason::SolverInvalidState);
            break;
        }

        let accepted_update_metrics = candidate_update_metrics.ok_or_else(|| {
            invalid_input(String::from(
                "laminar SIMPLE accepted update metrics exceed the finite numeric range",
            ))
        })?;
        velocity = corrected_velocity;
        pressure = corrected_pressure;
        final_phi = corrected_phi;
        surface_flux = final_phi.clone();
        final_continuity = corrected_continuity;
        final_hby_a = hby_a;
        final_pressure_assembly = Some(pressure_assembly);

        final_grad_p = corrected_pressure_gradient;

        let relative_velocity_change_l2 = accepted_update_metrics.velocity_change_l2;
        let relative_pressure_change_l2 = accepted_update_metrics.pressure_change_l2;
        let residual_control = evaluate_laminar_simple_residual_control(
            Some(momentum.initial_normalized_residual_norm),
            Some(pressure_initial_normalized_residual_norm),
            options,
        );
        timing.field_correction_seconds += field_correction_started.elapsed().as_secs_f64();

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
            momentum_component_linear_solves: momentum.component_linear_solves,
            pressure_correction_linear_solves,
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

    let finalization_started = Instant::now();
    let final_convection = vector_convection_divergence_with_addressing(
        &runtime.mesh,
        &mesh_cache.scalar_gradient,
        mesh_cache.weighted_least_squares.as_ref(),
        &mesh_cache.face_addressing,
        &velocity,
        &velocity_boundary,
        &final_phi,
        options.schemes.div_phi_u,
        options.schemes.grad_u,
    )?;
    let operator_summary =
        summarize_operators(&final_phi, &final_grad_p, &final_hby_a, &final_convection);
    let fields = LaminarSimpleFieldSummary {
        velocity: summarize_vectors(&velocity),
        pressure: summarize_scalars(&pressure),
    };
    let linear_solve_summary = summarize_linear_solves(&history);
    if let Some(executor) = momentum_component_executor.as_mut() {
        executor.shutdown()?;
    }
    timing.finalization_seconds = finalization_started.elapsed().as_secs_f64();
    let timing = timing.finish(solver_started.elapsed().as_secs_f64());

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
        linear_solve_summary,
        operator_summary,
        boundary_summary,
        pressure_assembly: final_pressure_assembly,
        timing,
        fields,
        final_velocity: velocity,
        final_pressure: pressure,
        #[cfg(test)]
        final_phi,
        history,
    })
}

fn pressure_solver_stop_reason(
    termination: IterativeSolveTermination,
) -> Option<LaminarSimpleStopReason> {
    matches!(termination, IterativeSolveTermination::Breakdown)
        .then_some(LaminarSimpleStopReason::PressureSolverInvalidState)
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
    component_linear_solves: [LinearSolveConvergenceSummary; 3],
    diagonal_min: f64,
    diagonal_max: f64,
    h1_min: f64,
    h1_max: f64,
    assembly_seconds: f64,
    gradient_seconds: f64,
    matrix_fill_seconds: f64,
    linear_solve_seconds: f64,
}

struct MomentumPredictorFields<'a> {
    velocity: &'a [Point3],
    velocity_boundary: &'a [VectorFaceTreatment],
    flux: &'a [f64],
    grad_p: &'a [Point3],
}

struct ScalarSolveReport {
    solution: Vec<f64>,
    iterations: usize,
    converged: bool,
    termination: IterativeSolveTermination,
    initial_normalized_residual_norm: f64,
    residual_norm: f64,
    normalized_residual_norm: f64,
    effective_normalized_tolerance: f64,
    stop_reason: LinearSolveStopReason,
    pcg_timing: Option<PcgKernelTiming>,
    gamg_timing: Option<GamgKernelTiming>,
}

impl From<&ScalarSolveReport> for LinearSolveConvergenceSummary {
    fn from(report: &ScalarSolveReport) -> Self {
        let stop_reason = match report.termination {
            IterativeSolveTermination::Breakdown => LinearSolveStopReason::Breakdown,
            IterativeSolveTermination::MaxIterations if !report.converged => {
                LinearSolveStopReason::MaxIterations
            }
            _ => report.stop_reason,
        };
        Self {
            iterations: report.iterations,
            converged: report.converged,
            initial_normalized_residual_norm: report.initial_normalized_residual_norm,
            residual_norm: report.residual_norm,
            normalized_residual_norm: report.normalized_residual_norm,
            effective_normalized_tolerance: report.effective_normalized_tolerance,
            stop_reason,
        }
    }
}

#[derive(Clone, Copy)]
struct ScalarSolveControls {
    solver: LaminarSimpleLinearSolver,
    preconditioner: LaminarSimplePreconditioner,
    tolerance: f64,
    relative_tolerance: f64,
    max_iterations: usize,
    gamg_options: Option<GamgOptions>,
    profile_gamg: bool,
    profile_pcg: bool,
}

// OpenFOAM Foundation's solver-performance convergence check treats relTol as
// enabled only strictly above `small` for double precision.
fn linear_relative_tolerance_is_active(
    _solver: LaminarSimpleLinearSolver,
    relative_tolerance: f64,
) -> bool {
    relative_tolerance > OPENFOAM_RELATIVE_TOLERANCE_SMALL
}

fn effective_normalized_tolerance_and_reason(
    solver: LaminarSimpleLinearSolver,
    absolute_tolerance: f64,
    relative_tolerance: f64,
    initial_normalized_residual: f64,
) -> (f64, LinearSolveStopReason) {
    if linear_relative_tolerance_is_active(solver, relative_tolerance) {
        let relative_limit = relative_tolerance * initial_normalized_residual;
        if relative_limit > absolute_tolerance {
            return (relative_limit, LinearSolveStopReason::RelativeTolerance);
        }
    }
    (absolute_tolerance, LinearSolveStopReason::AbsoluteTolerance)
}

#[cfg(test)]
fn effective_non_gamg_normalized_tolerance(
    absolute_tolerance: f64,
    relative_tolerance: f64,
    initial_normalized_residual: f64,
) -> f64 {
    effective_normalized_tolerance_and_reason(
        LaminarSimpleLinearSolver::Cg,
        absolute_tolerance,
        relative_tolerance,
        initial_normalized_residual,
    )
    .0
}

struct ScalarSolveWorkspace {
    zero_initial: Vec<f64>,
    matrix_product: Vec<f64>,
    residual: Vec<f64>,
}

impl ScalarSolveWorkspace {
    fn new(size: usize) -> Self {
        Self {
            zero_initial: vec![0.0; size],
            matrix_product: vec![0.0; size],
            residual: vec![0.0; size],
        }
    }

    fn try_new(size: usize) -> Result<Self> {
        fn try_zeroed(size: usize) -> Result<Vec<f64>> {
            let mut values = Vec::new();
            values
                .try_reserve_exact(size)
                .map_err(|_| MeshError::OutOfMemory)?;
            values.resize(size, 0.0);
            Ok(values)
        }

        Ok(Self {
            zero_initial: try_zeroed(size)?,
            matrix_product: try_zeroed(size)?,
            residual: try_zeroed(size)?,
        })
    }

    #[cfg(test)]
    fn identity(&self) -> ScalarSolveWorkspaceIdentity {
        ScalarSolveWorkspaceIdentity {
            zero_initial: (
                self.zero_initial.as_ptr() as usize,
                self.zero_initial.capacity(),
            ),
            matrix_product: (
                self.matrix_product.as_ptr() as usize,
                self.matrix_product.capacity(),
            ),
            residual: (self.residual.as_ptr() as usize, self.residual.capacity()),
        }
    }

    fn validate(&self, matrix: &CsrMatrix, rhs: &[f64]) -> Result<()> {
        let size = self.zero_initial.len();
        if matrix.rows() != size
            || matrix.cols() != size
            || rhs.len() != size
            || self.matrix_product.len() != size
            || self.residual.len() != size
        {
            return Err(invalid_input(format!(
                "scalar solve workspace size {size} does not match matrix {}x{}, rhs {}, matrix-product {}, and residual {}",
                matrix.rows(),
                matrix.cols(),
                rhs.len(),
                self.matrix_product.len(),
                self.residual.len()
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ScalarSolveWorkspaceIdentity {
    zero_initial: (usize, usize),
    matrix_product: (usize, usize),
    residual: (usize, usize),
}

struct MomentumComponentSolveJob {
    component: usize,
    system: ScalarComponentSystem,
    initial: Vec<f64>,
    controls: ScalarSolveControls,
    #[cfg(test)]
    action: MomentumComponentTestAction,
}

enum MomentumComponentWorkerFailure {
    Solve(MeshError),
    Panicked,
}

type MomentumComponentCommand = Option<MomentumComponentSolveJob>;
type MomentumComponentWorkerResult =
    std::result::Result<ScalarSolveReport, MomentumComponentWorkerFailure>;

struct MomentumComponentOutcome {
    component: usize,
    result: MomentumComponentWorkerResult,
    #[cfg(test)]
    workspace_identity: ScalarSolveWorkspaceIdentity,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum MomentumComponentTestAction {
    Solve,
    Error(&'static str),
    Panic,
}

struct MomentumComponentExecutor {
    command_senders: Vec<SyncSender<MomentumComponentCommand>>,
    result_receiver: Receiver<MomentumComponentOutcome>,
    handles: Vec<Option<JoinHandle<()>>>,
    finished: bool,
    #[cfg(test)]
    test_actions: [MomentumComponentTestAction; 3],
    #[cfg(test)]
    workspace_observations: [Vec<ScalarSolveWorkspaceIdentity>; 3],
}

impl MomentumComponentExecutor {
    fn new(size: usize) -> Result<Self> {
        let workspaces = [
            ScalarSolveWorkspace::try_new(size)?,
            ScalarSolveWorkspace::try_new(size)?,
            ScalarSolveWorkspace::try_new(size)?,
        ];
        let (result_sender, result_receiver) = sync_channel(3);
        let mut command_senders = Vec::with_capacity(3);
        let mut handles = Vec::with_capacity(3);

        for (component, workspace) in workspaces.into_iter().enumerate() {
            let (command_sender, command_receiver) = sync_channel(1);
            let worker_result_sender = result_sender.clone();
            let worker_name = format!(
                "ferrum-momentum-{}",
                ["X", "Y", "Z"]
                    .get(component)
                    .expect("three fixed momentum components")
            );
            let spawn_result = thread::Builder::new().name(worker_name).spawn(move || {
                run_momentum_component_worker(
                    component,
                    command_receiver,
                    worker_result_sender,
                    workspace,
                );
            });
            match spawn_result {
                Ok(handle) => {
                    command_senders.push(command_sender);
                    handles.push(Some(handle));
                }
                Err(error) => {
                    for sender in &command_senders {
                        let _ = sender.send(None);
                    }
                    for handle in handles.iter_mut().filter_map(Option::take) {
                        let _ = handle.join();
                    }
                    return Err(MeshError::Io(error));
                }
            }
        }
        drop(result_sender);

        Ok(Self {
            command_senders,
            result_receiver,
            handles,
            finished: false,
            #[cfg(test)]
            test_actions: [MomentumComponentTestAction::Solve; 3],
            #[cfg(test)]
            workspace_observations: std::array::from_fn(|_| Vec::new()),
        })
    }

    fn solve(
        &mut self,
        components: Vec<ScalarComponentSystem>,
        old_components: [Vec<f64>; 3],
        controls: ScalarSolveControls,
    ) -> Result<[ScalarSolveReport; 3]> {
        if self.finished {
            return Err(invalid_input(
                "laminar SIMPLE momentum component executor is already shut down".to_string(),
            ));
        }
        if components.len() != 3 {
            return Err(invalid_input(format!(
                "laminar SIMPLE momentum equation expected 3 components, got {}",
                components.len()
            )));
        }
        if self.command_senders.len() != 3 || self.handles.len() != 3 {
            return Err(invalid_input(
                "laminar SIMPLE momentum component executor has an invalid worker count"
                    .to_string(),
            ));
        }

        let systems: [ScalarComponentSystem; 3] =
            components.try_into().map_err(|components: Vec<_>| {
                invalid_input(format!(
                    "laminar SIMPLE momentum equation expected 3 components, got {}",
                    components.len()
                ))
            })?;
        let mut outcomes: [Option<MomentumComponentWorkerResult>; 3] = [None, None, None];
        let mut dispatched = 0;
        for (component, (system, initial)) in systems.into_iter().zip(old_components).enumerate() {
            let job = MomentumComponentSolveJob {
                component,
                system,
                initial,
                controls,
                #[cfg(test)]
                action: self.test_actions[component],
            };
            if self.command_senders[component].send(Some(job)).is_err() {
                outcomes[component] = Some(Err(MomentumComponentWorkerFailure::Solve(
                    invalid_input(format!(
                        "laminar SIMPLE momentum component {} worker command channel disconnected",
                        component_name(component)
                    )),
                )));
            } else {
                dispatched += 1;
            }
        }

        for _ in 0..dispatched {
            let outcome = self.result_receiver.recv().map_err(|_| {
                let missing = outcomes
                    .iter()
                    .position(Option::is_none)
                    .map(component_name)
                    .unwrap_or("unknown");
                invalid_input(format!(
                    "laminar SIMPLE momentum component {missing} worker result channel disconnected"
                ))
            })?;
            let component = outcome.component;
            #[cfg(test)]
            let workspace_identity = outcome.workspace_identity;
            store_momentum_component_worker_result(&mut outcomes, component, outcome.result)?;
            #[cfg(test)]
            self.workspace_observations[component].push(workspace_identity);
        }

        let mut reports: [Option<ScalarSolveReport>; 3] = [None, None, None];
        for component in 0..3 {
            match outcomes[component].take().ok_or_else(|| {
                invalid_input(format!(
                    "laminar SIMPLE momentum component {} worker returned no result",
                    component_name(component)
                ))
            })? {
                Ok(report) => {
                    reports[component] = Some(report);
                }
                Err(MomentumComponentWorkerFailure::Solve(error)) => {
                    return Err(invalid_input(format!(
                        "laminar SIMPLE momentum component {} solve failed: {error}",
                        component_name(component)
                    )));
                }
                Err(MomentumComponentWorkerFailure::Panicked) => {
                    return Err(invalid_input(format!(
                        "laminar SIMPLE momentum component {} worker panicked",
                        component_name(component)
                    )));
                }
            }
        }

        Ok([
            reports[0]
                .take()
                .expect("validated momentum component X report"),
            reports[1]
                .take()
                .expect("validated momentum component Y report"),
            reports[2]
                .take()
                .expect("validated momentum component Z report"),
        ])
    }

    fn shutdown(&mut self) -> Result<()> {
        if self.finished {
            return Ok(());
        }
        self.finished = true;

        let mut first_error = None;
        for component in 0..self.command_senders.len() {
            if self.command_senders[component].send(None).is_err() && first_error.is_none() {
                first_error = Some(invalid_input(format!(
                    "laminar SIMPLE momentum component {} worker shutdown channel disconnected",
                    component_name(component)
                )));
            }
        }
        self.command_senders.clear();

        for component in 0..self.handles.len() {
            if let Some(handle) = self.handles[component].take()
                && handle.join().is_err()
                && first_error.is_none()
            {
                first_error = Some(invalid_input(format!(
                    "laminar SIMPLE momentum component {} worker join failed",
                    component_name(component)
                )));
            }
        }

        match first_error {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }

    #[cfg(test)]
    fn set_test_actions(&mut self, actions: [MomentumComponentTestAction; 3]) {
        self.test_actions = actions;
    }
}

fn store_momentum_component_worker_result(
    outcomes: &mut [Option<MomentumComponentWorkerResult>; 3],
    component: usize,
    result: MomentumComponentWorkerResult,
) -> Result<()> {
    if component >= 3 {
        return Err(invalid_input(format!(
            "laminar SIMPLE momentum worker returned unexpected component index {component}"
        )));
    }
    if outcomes[component].is_some() {
        return Err(invalid_input(format!(
            "laminar SIMPLE momentum component {} worker returned a duplicate result",
            component_name(component)
        )));
    }
    outcomes[component] = Some(result);
    Ok(())
}

impl Drop for MomentumComponentExecutor {
    fn drop(&mut self) {
        let _ = self.shutdown();
    }
}

fn run_momentum_component_worker(
    component: usize,
    command_receiver: Receiver<MomentumComponentCommand>,
    result_sender: SyncSender<MomentumComponentOutcome>,
    mut workspace: ScalarSolveWorkspace,
) {
    let worker_result = catch_unwind(AssertUnwindSafe(|| {
        while let Ok(command) = command_receiver.recv() {
            let Some(job) = command else {
                break;
            };
            let result = if job.component != component {
                Err(MomentumComponentWorkerFailure::Solve(invalid_input(
                    format!(
                        "momentum worker {} received component {}",
                        component_name(component),
                        job.component
                    ),
                )))
            } else {
                #[cfg(test)]
                {
                    match job.action {
                        MomentumComponentTestAction::Solve => solve_scalar_system_with_workspaces(
                            &job.system.matrix,
                            &job.system.rhs,
                            Some(&job.initial),
                            job.controls,
                            &mut workspace,
                            None,
                            None,
                        )
                        .map_err(MomentumComponentWorkerFailure::Solve),
                        MomentumComponentTestAction::Error(message) => {
                            Err(MomentumComponentWorkerFailure::Solve(invalid_input(
                                message.to_string(),
                            )))
                        }
                        MomentumComponentTestAction::Panic => {
                            panic!("injected momentum component worker panic")
                        }
                    }
                }
                #[cfg(not(test))]
                {
                    solve_scalar_system_with_workspaces(
                        &job.system.matrix,
                        &job.system.rhs,
                        Some(&job.initial),
                        job.controls,
                        &mut workspace,
                        None,
                        None,
                    )
                    .map_err(MomentumComponentWorkerFailure::Solve)
                }
            };
            let outcome = MomentumComponentOutcome {
                component,
                result,
                #[cfg(test)]
                workspace_identity: workspace.identity(),
            };
            if result_sender.send(outcome).is_err() {
                break;
            }
        }
    }));

    if worker_result.is_err() {
        let _ = result_sender.send(MomentumComponentOutcome {
            component,
            result: Err(MomentumComponentWorkerFailure::Panicked),
            #[cfg(test)]
            workspace_identity: workspace.identity(),
        });
    }
}

struct MomentumEquation {
    components: Vec<ScalarComponentSystem>,
    diagonal: Vec<f64>,
    h1: Vec<f64>,
    diagonal_min: f64,
    diagonal_max: f64,
    h1_min: f64,
    h1_max: f64,
    gradient_seconds: f64,
    matrix_fill_seconds: f64,
}

#[derive(Clone, Copy, Debug, Default)]
struct MomentumFaceCsrSlots {
    owner_neighbour: Option<usize>,
    neighbour_owner: Option<usize>,
}

struct MomentumCsrPattern {
    sparsity: CsrSparsityPattern,
    diagonal_slots: Vec<usize>,
    face_slots: Vec<MomentumFaceCsrSlots>,
}

#[derive(Clone, Copy, Debug)]
struct PressureFaceGeometry {
    area_magnitude: f64,
    projected_distance: f64,
    centre_distance: f64,
    non_orthogonal_area_vector: Point3,
}

struct PressureGeometryCache {
    faces: Vec<PressureFaceGeometry>,
}

struct LaminarSimpleMeshCache {
    momentum: MomentumCsrPattern,
    scalar_gradient: ScalarGradientGeometry,
    weighted_least_squares: Option<WeightedLeastSquaresGeometry>,
    pressure_geometry: PressureGeometryCache,
    gamg_face_area_weights: Vec<GamgFacePairWeight>,
    face_addressing: CompactSimpleFaceAddressing,
}

impl LaminarSimpleMeshCache {
    fn from_mesh(mesh: &SolverRuntimeMeshData, schemes: LaminarSimpleSchemes) -> Result<Self> {
        let momentum = MomentumCsrPattern::from_mesh(mesh)?;
        let scalar_gradient = ScalarGradientGeometry::from_mesh(mesh)?;
        let face_addressing = CompactSimpleFaceAddressing::from_mesh(mesh)?;
        let weighted_least_squares = schemes
            .uses_weighted_least_squares()
            .then(|| {
                WeightedLeastSquaresGeometry::from_mesh(mesh, &scalar_gradient, &face_addressing)
            })
            .transpose()?;
        let pressure_geometry = PressureGeometryCache::from_mesh(mesh);
        let gamg_face_area_weights = gamg_face_area_pair_weights(mesh)?;
        debug_assert_eq!(
            face_addressing.internal_faces().len() + face_addressing.boundary_faces().len(),
            mesh.faces
        );
        Ok(Self {
            momentum,
            scalar_gradient,
            weighted_least_squares,
            pressure_geometry,
            gamg_face_area_weights,
            face_addressing,
        })
    }
}

impl PressureGeometryCache {
    fn from_mesh(mesh: &SolverRuntimeMeshData) -> Self {
        let mut faces = Vec::with_capacity(mesh.faces);
        for face_index in 0..mesh.faces {
            let owner = mesh.owner[face_index];
            let from = mesh.cell_centres[owner];
            let to = mesh.neighbour[face_index]
                .map(|neighbour| mesh.cell_centres[neighbour])
                .unwrap_or(mesh.face_centres[face_index]);
            let area_vector = mesh.face_area_vectors[face_index];
            let area_magnitude = magnitude(area_vector);
            let delta = Point3 {
                x: to.x - from.x,
                y: to.y - from.y,
                z: to.z - from.z,
            };
            let projected_distance = (dot(delta, area_vector) / area_magnitude).abs();
            let centre_distance = distance(from, to);
            let direction = Point3 {
                x: (to.x - from.x) / centre_distance,
                y: (to.y - from.y) / centre_distance,
                z: (to.z - from.z) / centre_distance,
            };
            let orthogonal_area = dot(area_vector, direction);
            let non_orthogonal_area_vector = Point3 {
                x: area_vector.x - orthogonal_area * direction.x,
                y: area_vector.y - orthogonal_area * direction.y,
                z: area_vector.z - orthogonal_area * direction.z,
            };
            faces.push(PressureFaceGeometry {
                area_magnitude,
                projected_distance,
                centre_distance,
                non_orthogonal_area_vector,
            });
        }
        Self { faces }
    }

    fn face(&self, face_index: usize) -> PressureFaceGeometry {
        self.faces[face_index]
    }
}

fn gamg_face_area_pair_weights(mesh: &SolverRuntimeMeshData) -> Result<Vec<GamgFacePairWeight>> {
    let mut weights = Vec::with_capacity(mesh.internal_faces);
    for face_index in 0..mesh.faces {
        let Some(neighbour) = mesh.neighbour[face_index] else {
            continue;
        };
        let area_vector = mesh.face_area_vectors[face_index];
        let area = magnitude(area_vector);
        if !area.is_finite() || area <= f64::EPSILON {
            return Err(invalid_input(format!(
                "GAMG faceAreaPair internal face {face_index} has invalid area magnitude {area}"
            )));
        }
        let area_root = area.sqrt();
        let weighted_direction = Point3 {
            x: area_vector.x / area_root,
            y: 1.01 * area_vector.y / area_root,
            z: 1.02 * area_vector.z / area_root,
        };
        weights.push(GamgFacePairWeight::new(
            mesh.owner[face_index],
            neighbour,
            magnitude(weighted_direction),
        )?);
    }
    Ok(weights)
}

impl MomentumCsrPattern {
    fn from_mesh(mesh: &SolverRuntimeMeshData) -> Result<Self> {
        let mut row_columns = (0..mesh.cells).map(|cell| vec![cell]).collect::<Vec<_>>();
        for face_index in 0..mesh.faces {
            let owner = mesh.owner[face_index];
            if let Some(neighbour) = mesh.neighbour[face_index] {
                row_columns[owner].push(neighbour);
                row_columns[neighbour].push(owner);
            }
        }

        let mut row_offsets = Vec::with_capacity(mesh.cells + 1);
        let mut col_indices = Vec::new();
        row_offsets.push(0);
        for columns in &mut row_columns {
            columns.sort_unstable();
            columns.dedup();
            col_indices.extend_from_slice(columns);
            row_offsets.push(col_indices.len());
        }

        let sparsity = CsrSparsityPattern::new(mesh.cells, mesh.cells, row_offsets, col_indices)?;
        let diagonal_slots = (0..mesh.cells)
            .map(|cell| momentum_csr_slot(&sparsity, cell, cell))
            .collect::<Result<Vec<_>>>()?;
        let face_slots = (0..mesh.faces)
            .map(|face_index| {
                let owner = mesh.owner[face_index];
                let Some(neighbour) = mesh.neighbour[face_index] else {
                    return Ok(MomentumFaceCsrSlots::default());
                };
                Ok(MomentumFaceCsrSlots {
                    owner_neighbour: Some(momentum_csr_slot(&sparsity, owner, neighbour)?),
                    neighbour_owner: Some(momentum_csr_slot(&sparsity, neighbour, owner)?),
                })
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            sparsity,
            diagonal_slots,
            face_slots,
        })
    }

    fn diagonal_slot(&self, cell: usize) -> usize {
        self.diagonal_slots[cell]
    }

    fn internal_slots(&self, face_index: usize) -> (usize, usize) {
        let slots = self.face_slots[face_index];
        (
            slots
                .owner_neighbour
                .expect("validated internal momentum face owner-neighbour slot"),
            slots
                .neighbour_owner
                .expect("validated internal momentum face neighbour-owner slot"),
        )
    }
}

fn momentum_csr_slot(pattern: &CsrSparsityPattern, row: usize, column: usize) -> Result<usize> {
    let start = pattern.row_offsets()[row];
    let end = pattern.row_offsets()[row + 1];
    pattern.col_indices()[start..end]
        .binary_search(&column)
        .map(|offset| start + offset)
        .map_err(|_| {
            invalid_input(format!(
                "momentum CSR pattern row {row} has no column {column}"
            ))
        })
}

fn solve_momentum_predictor(
    mesh: &SolverRuntimeMeshData,
    mesh_cache: &LaminarSimpleMeshCache,
    fields: MomentumPredictorFields<'_>,
    options: &LaminarSimpleOptions,
    scalar_solve_workspace: &mut ScalarSolveWorkspace,
    momentum_component_executor: Option<&mut MomentumComponentExecutor>,
) -> Result<MomentumPredictorReport> {
    let MomentumPredictorFields {
        velocity,
        velocity_boundary,
        flux,
        grad_p,
    } = fields;
    let old_components = split_components(velocity);
    let assembly_started = Instant::now();
    let equation = assemble_momentum_equation(
        mesh,
        mesh_cache,
        velocity,
        velocity_boundary,
        flux,
        grad_p,
        &old_components,
        options,
    )?;
    let assembly_seconds = assembly_started.elapsed().as_secs_f64();
    let mut solved_components = [Vec::new(), Vec::new(), Vec::new()];
    let mut total_iterations = 0;
    let mut residual_squared_sum = 0.0;
    let mut component_initial_normalized_residual_norms = [0.0; 3];
    let mut component_residual_norms = [0.0; 3];
    let mut component_normalized_residual_norms = [0.0; 3];
    let mut component_converged = [false; 3];
    let mut component_linear_solves = [LinearSolveConvergenceSummary::default(); 3];
    let linear_solve_started = Instant::now();
    let controls = ScalarSolveControls {
        solver: options.momentum_linear_solver,
        preconditioner: options.momentum_preconditioner,
        tolerance: options.momentum_linear_tolerance,
        relative_tolerance: options.momentum_linear_relative_tolerance,
        max_iterations: options.momentum_max_linear_iterations,
        gamg_options: None,
        profile_gamg: false,
        profile_pcg: false,
    };
    let reports = if let Some(executor) = momentum_component_executor {
        executor.solve(equation.components, old_components, controls)?
    } else {
        solve_momentum_components_serial(
            &equation.components,
            &old_components,
            controls,
            scalar_solve_workspace,
        )?
    };

    for (component, report) in reports.into_iter().enumerate() {
        total_iterations += report.iterations;
        residual_squared_sum += report.residual_norm * report.residual_norm;
        component_initial_normalized_residual_norms[component] =
            report.initial_normalized_residual_norm;
        component_residual_norms[component] = report.residual_norm;
        component_normalized_residual_norms[component] = report.normalized_residual_norm;
        component_converged[component] = report.converged;
        component_linear_solves[component] = LinearSolveConvergenceSummary::from(&report);
        solved_components[component] = report.solution;
    }
    let linear_solve_seconds = linear_solve_started.elapsed().as_secs_f64();

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
        component_linear_solves,
        diagonal_min: equation.diagonal_min,
        diagonal_max: equation.diagonal_max,
        h1_min: equation.h1_min,
        h1_max: equation.h1_max,
        assembly_seconds,
        gradient_seconds: equation.gradient_seconds,
        matrix_fill_seconds: equation.matrix_fill_seconds,
        linear_solve_seconds,
    })
}

fn solve_momentum_components_serial(
    components: &[ScalarComponentSystem],
    old_components: &[Vec<f64>; 3],
    controls: ScalarSolveControls,
    scalar_solve_workspace: &mut ScalarSolveWorkspace,
) -> Result<[ScalarSolveReport; 3]> {
    if components.len() != 3 {
        return Err(invalid_input(format!(
            "laminar SIMPLE momentum equation expected 3 components, got {}",
            components.len()
        )));
    }
    let mut reports: [Option<ScalarSolveReport>; 3] = [None, None, None];
    for (component, system) in components.iter().enumerate() {
        let report = solve_scalar_system_with_workspaces(
            &system.matrix,
            &system.rhs,
            Some(&old_components[component]),
            controls,
            scalar_solve_workspace,
            None,
            None,
        )
        .map_err(|error| {
            invalid_input(format!(
                "laminar SIMPLE momentum component {} solve failed: {error}",
                component_name(component)
            ))
        })?;
        reports[component] = Some(report);
    }
    Ok([
        reports[0]
            .take()
            .expect("validated serial momentum component X report"),
        reports[1]
            .take()
            .expect("validated serial momentum component Y report"),
        reports[2]
            .take()
            .expect("validated serial momentum component Z report"),
    ])
}

#[allow(clippy::too_many_arguments)]
fn assemble_momentum_equation(
    mesh: &SolverRuntimeMeshData,
    mesh_cache: &LaminarSimpleMeshCache,
    velocity: &[Point3],
    velocity_boundary: &[VectorFaceTreatment],
    flux: &[f64],
    grad_p: &[Point3],
    old_components: &[Vec<f64>; 3],
    options: &LaminarSimpleOptions,
) -> Result<MomentumEquation> {
    let mut components = Vec::with_capacity(3);
    let mut diagonal = Vec::new();
    let mut h1 = Vec::new();
    let gradient_started = Instant::now();
    let component_gradients = if options.schemes.div_phi_u.uses_linear_upwind() {
        let gradient_scheme = options
            .schemes
            .div_phi_u
            .gradient_scheme(options.schemes.grad_u);
        Some(vector_component_gradients_with_scheme_and_addressing(
            mesh,
            &mesh_cache.scalar_gradient,
            mesh_cache.weighted_least_squares.as_ref(),
            &mesh_cache.face_addressing,
            velocity,
            velocity_boundary,
            flux,
            gradient_scheme,
        )?)
    } else {
        None
    };
    let gradient_seconds = gradient_started.elapsed().as_secs_f64();

    let matrix_fill_started = Instant::now();
    for component in 0..3 {
        let volumetric_source = grad_p
            .iter()
            .map(|value| -component_value(*value, component))
            .collect::<Vec<_>>();
        let boundary = scalar_component_boundary(velocity_boundary, component);
        let mut system = assemble_momentum_component_system_with_addressing(
            mesh,
            &mesh_cache.face_addressing,
            &mesh_cache.momentum,
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
    let matrix_fill_seconds = matrix_fill_started.elapsed().as_secs_f64();

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
        gradient_seconds,
        matrix_fill_seconds,
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

fn solve_scalar_system_with_workspaces(
    matrix: &CsrMatrix,
    rhs: &[f64],
    initial: Option<&[f64]>,
    controls: ScalarSolveControls,
    scalar_workspace: &mut ScalarSolveWorkspace,
    pcg_workspace: Option<&mut PreconditionedConjugateGradientWorkspace>,
    gamg_workspace: Option<&mut GamgWorkspace>,
) -> Result<ScalarSolveReport> {
    if !controls.relative_tolerance.is_finite() || controls.relative_tolerance < 0.0 {
        return Err(invalid_input(format!(
            "scalar solve relTol must be finite and non-negative, got {}",
            controls.relative_tolerance
        )));
    }
    if controls.solver == LaminarSimpleLinearSolver::Gamg {
        let gamg_options = controls.gamg_options.ok_or_else(|| {
            invalid_input("GAMG solve requires resolved GAMG options".to_string())
        })?;
        if controls.relative_tolerance.to_bits() != gamg_options.relative_tolerance.to_bits() {
            return Err(invalid_input(format!(
                "GAMG scalar solve relTol sources must match exactly, got controls={} and options={}",
                controls.relative_tolerance, gamg_options.relative_tolerance
            )));
        }
    }
    scalar_workspace.validate(matrix, rhs)?;
    let ScalarSolveWorkspace {
        zero_initial,
        matrix_product,
        residual,
    } = scalar_workspace;
    let initial_values = initial.unwrap_or(zero_initial);
    let (
        normalisation_factor,
        initial_residual_norm,
        initial_l1_residual_norm,
        prepared_pcg_residual,
        prepared_pcg_timing,
    ) = if controls.solver == LaminarSimpleLinearSolver::Pcg {
        let profile_prepared_pcg = controls.profile_pcg;
        let prepared_total_started = profile_prepared_pcg.then(Instant::now);
        let prepared_matrix_vector_started = profile_prepared_pcg.then(Instant::now);
        matrix.matvec_into(initial_values, matrix_product)?;
        let prepared_matrix_vector_seconds = prepared_matrix_vector_started
            .map(|started| started.elapsed().as_secs_f64())
            .unwrap_or(0.0);
        let normalisation_factor =
            ldu_l1_residual_normalisation_factor(matrix, rhs, initial_values, matrix_product)?;
        let prepared_residual_started = profile_prepared_pcg.then(Instant::now);
        let prepared_residual =
            PreparedPcgResidual::write_from_rhs_and_matrix_product(residual, rhs, matrix_product)?;
        let prepared_vector_operation_seconds = prepared_residual_started
            .map(|started| started.elapsed().as_secs_f64())
            .unwrap_or(0.0);
        let prepared_pcg_timing = PreparedPcgInitialTiming {
            total_seconds: prepared_total_started
                .map(|started| started.elapsed().as_secs_f64())
                .unwrap_or(0.0),
            matrix_vector_seconds: prepared_matrix_vector_seconds,
            vector_operation_seconds: prepared_vector_operation_seconds,
        };
        (
            normalisation_factor,
            prepared_residual.residual_squared().sqrt(),
            prepared_residual.initial_l1_residual_norm(),
            Some(prepared_residual),
            prepared_pcg_timing,
        )
    } else {
        matrix.matvec_into(initial_values, matrix_product)?;
        let normalisation_factor =
            ldu_l1_residual_normalisation_factor(matrix, rhs, initial_values, matrix_product)?;
        for ((residual_value, source), matrix_value) in
            residual.iter_mut().zip(rhs).zip(matrix_product.iter())
        {
            *residual_value = source - matrix_value;
        }
        let initial_residual_norm = l2_norm(residual);
        let initial_l1_residual_norm = l1_norm(residual);
        (
            normalisation_factor,
            initial_residual_norm,
            initial_l1_residual_norm,
            None::<PreparedPcgResidual<'_>>,
            PreparedPcgInitialTiming::default(),
        )
    };
    let initial_normalized_residual_norm = initial_l1_residual_norm / normalisation_factor;
    let (effective_normalized_tolerance, dominant_tolerance_reason) =
        effective_normalized_tolerance_and_reason(
            controls.solver,
            controls.tolerance,
            controls.relative_tolerance,
            initial_normalized_residual_norm,
        );

    let gamg_min_iterations = controls
        .gamg_options
        .map(|options| options.min_iterations)
        .unwrap_or(0);
    let initially_converged = if controls.solver == LaminarSimpleLinearSolver::Gamg {
        initial_l1_residual_norm == 0.0
    } else {
        initial_normalized_residual_norm < effective_normalized_tolerance
    };
    if initially_converged
        && gamg_min_iterations == 0
        && controls.solver != LaminarSimpleLinearSolver::Pcg
    {
        return Ok(ScalarSolveReport {
            solution: initial_values.to_vec(),
            iterations: 0,
            converged: true,
            termination: IterativeSolveTermination::Converged,
            initial_normalized_residual_norm,
            residual_norm: initial_residual_norm,
            normalized_residual_norm: initial_normalized_residual_norm,
            effective_normalized_tolerance,
            stop_reason: if controls.solver == LaminarSimpleLinearSolver::Gamg
                && initial_l1_residual_norm == 0.0
            {
                LinearSolveStopReason::ExactZero
            } else {
                dominant_tolerance_reason
            },
            pcg_timing: None,
            gamg_timing: None,
        });
    }

    // Ferrum's current CSR kernels stop on L2. This conservative conversion
    // guarantees the LDU L1-normalised residual tolerance before reporting success.
    let component_count = rhs.len().max(1) as f64;
    let l2_absolute_or_effective_tolerance = if controls.solver == LaminarSimpleLinearSolver::Gamg {
        controls.tolerance
    } else {
        effective_normalized_tolerance
    };
    let solver_tolerance = strict_l2_tolerance_for_l1_limit(
        l2_absolute_or_effective_tolerance * normalisation_factor,
        component_count,
    );
    let (report, pcg_timing, gamg_timing) = match controls.solver {
        LaminarSimpleLinearSolver::BiCgStab => (
            bicgstab_solve(
                matrix,
                rhs,
                initial,
                BiCgStabOptions {
                    max_iterations: controls.max_iterations,
                    tolerance: solver_tolerance,
                    preconditioner: map_cg_preconditioner(controls.preconditioner),
                },
            )?,
            None,
            None,
        ),
        LaminarSimpleLinearSolver::Cg => (
            conjugate_gradient_solve(
                matrix,
                rhs,
                initial,
                ConjugateGradientOptions {
                    max_iterations: controls.max_iterations,
                    tolerance: solver_tolerance,
                },
            )?,
            None,
            None,
        ),
        LaminarSimpleLinearSolver::Gamg => {
            let gamg_options = controls
                .gamg_options
                .expect("GAMG options were validated before workspace mutation");
            let workspace = gamg_workspace.ok_or_else(|| {
                invalid_input("GAMG solve requires a matching hierarchy workspace".to_string())
            })?;
            let relative_solver_tolerance = if linear_relative_tolerance_is_active(
                LaminarSimpleLinearSolver::Gamg,
                gamg_options.relative_tolerance,
            ) {
                strict_l2_tolerance_for_l1_limit(
                    gamg_options.relative_tolerance * initial_l1_residual_norm,
                    component_count,
                )
            } else {
                0.0
            };
            let solve_controls = NormalizedL1GamgSolveControls {
                normalization_factor: normalisation_factor,
                tolerance: controls.tolerance,
                relative_tolerance: gamg_options.relative_tolerance,
                l2_controls: GamgSolveControls {
                    max_iterations: controls.max_iterations,
                    min_iterations: gamg_options.min_iterations,
                    tolerance: solver_tolerance.max(relative_solver_tolerance),
                    relative_tolerance: 0.0,
                },
            };
            if controls.profile_gamg {
                let profiled = workspace.solve_normalized_l1_with_controls_profiled(
                    matrix,
                    rhs,
                    initial,
                    solve_controls,
                )?;
                (profiled.report, None, Some(profiled.timing))
            } else {
                (
                    workspace.solve_normalized_l1_with_controls(
                        matrix,
                        rhs,
                        initial,
                        solve_controls,
                    )?,
                    None,
                    None,
                )
            }
        }
        LaminarSimpleLinearSolver::GaussSeidel => (
            gauss_seidel_solve(
                matrix,
                rhs,
                initial,
                GaussSeidelOptions {
                    max_iterations: controls.max_iterations,
                    tolerance: solver_tolerance,
                    omega: 1.0,
                },
            )?,
            None,
            None,
        ),
        LaminarSimpleLinearSolver::SymGaussSeidel => (
            symmetric_gauss_seidel_solve(
                matrix,
                rhs,
                initial,
                GaussSeidelOptions {
                    max_iterations: controls.max_iterations,
                    tolerance: solver_tolerance,
                    omega: 1.0,
                },
            )?,
            None,
            None,
        ),
        LaminarSimpleLinearSolver::Pcg => {
            let pcg_controls = NormalizedL1PcgSolveControls {
                max_iterations: controls.max_iterations,
                normalization_factor: normalisation_factor,
                tolerance: controls.tolerance,
                relative_tolerance: controls.relative_tolerance,
                preconditioner: map_cg_preconditioner(controls.preconditioner),
            };
            let prepared_residual = prepared_pcg_residual.ok_or_else(|| {
                invalid_input("PCG solve requires a prepared initial residual".to_string())
            })?;
            if controls.profile_pcg {
                if let Some(workspace) = pcg_workspace {
                    let profiled = workspace.solve_normalized_l1_prepared_profiled(
                        matrix,
                        rhs,
                        initial,
                        prepared_residual,
                        prepared_pcg_timing,
                        pcg_controls,
                    )?;
                    (profiled.report, Some(profiled.timing), None)
                } else {
                    let mut workspace = PreconditionedConjugateGradientWorkspace::new(
                        matrix,
                        pcg_controls.preconditioner,
                    )?;
                    let profiled = workspace.solve_normalized_l1_prepared_profiled(
                        matrix,
                        rhs,
                        initial,
                        prepared_residual,
                        prepared_pcg_timing,
                        pcg_controls,
                    )?;
                    (profiled.report, Some(profiled.timing), None)
                }
            } else if let Some(workspace) = pcg_workspace {
                (
                    workspace.solve_normalized_l1_prepared(
                        matrix,
                        rhs,
                        initial,
                        prepared_residual,
                        pcg_controls,
                    )?,
                    None,
                    None,
                )
            } else {
                let mut workspace = PreconditionedConjugateGradientWorkspace::new(
                    matrix,
                    pcg_controls.preconditioner,
                )?;
                (
                    workspace.solve_normalized_l1_prepared(
                        matrix,
                        rhs,
                        initial,
                        prepared_residual,
                        pcg_controls,
                    )?,
                    None,
                    None,
                )
            }
        }
        LaminarSimpleLinearSolver::Jacobi => (
            jacobi_solve(
                matrix,
                rhs,
                initial,
                JacobiOptions {
                    max_iterations: controls.max_iterations,
                    tolerance: solver_tolerance,
                    omega: 1.0,
                },
            )?,
            None,
            None,
        ),
    };
    matrix.matvec_into(&report.solution, matrix_product)?;
    for ((residual_value, source), matrix_value) in
        residual.iter_mut().zip(rhs).zip(matrix_product.iter())
    {
        *residual_value = source - matrix_value;
    }
    let final_l1_residual_norm = l1_norm(residual);
    let final_normalized_residual_norm = final_l1_residual_norm / normalisation_factor;
    let relative_tolerance = controls
        .gamg_options
        .map(|options| options.relative_tolerance)
        .unwrap_or(controls.relative_tolerance);
    let relative_tolerance_active =
        linear_relative_tolerance_is_active(controls.solver, relative_tolerance);
    let gamg_exact_zero = controls.solver == LaminarSimpleLinearSolver::Gamg
        && final_l1_residual_norm == 0.0
        && report.termination == IterativeSolveTermination::Converged
        && report.iterations == 0;
    let converged = gamg_exact_zero
        || final_normalized_residual_norm < controls.tolerance
        || (relative_tolerance_active
            && final_normalized_residual_norm
                < relative_tolerance * initial_normalized_residual_norm);
    let termination = if converged {
        IterativeSolveTermination::Converged
    } else if report.termination == IterativeSolveTermination::Breakdown {
        IterativeSolveTermination::Breakdown
    } else {
        IterativeSolveTermination::MaxIterations
    };
    let stop_reason = if gamg_exact_zero {
        LinearSolveStopReason::ExactZero
    } else if converged {
        dominant_tolerance_reason
    } else if termination == IterativeSolveTermination::Breakdown {
        LinearSolveStopReason::Breakdown
    } else {
        LinearSolveStopReason::MaxIterations
    };
    Ok(ScalarSolveReport {
        solution: report.solution,
        iterations: report.iterations,
        converged,
        termination,
        initial_normalized_residual_norm,
        residual_norm: report.residual_norm,
        normalized_residual_norm: final_normalized_residual_norm,
        effective_normalized_tolerance,
        stop_reason,
        pcg_timing,
        gamg_timing,
    })
}

fn strict_l2_tolerance_for_l1_limit(l1_limit: f64, component_count: f64) -> f64 {
    let l2_limit = l1_limit / component_count.sqrt();
    if l2_limit.is_finite() && l2_limit > 0.0 {
        l2_limit.next_down()
    } else {
        l2_limit
    }
}

fn ldu_l1_residual_normalisation_factor(
    matrix: &CsrMatrix,
    source: &[f64],
    solution: &[f64],
    matrix_solution: &[f64],
) -> Result<f64> {
    if source.len() != matrix.rows() || matrix_solution.len() != matrix.rows() {
        return Err(invalid_input(format!(
            "LDU L1 residual normalisation expected {} source and matrix-product entries, got {} and {}",
            matrix.rows(),
            source.len(),
            matrix_solution.len()
        )));
    }
    if solution.len() != matrix.cols() {
        return Err(invalid_input(format!(
            "LDU L1 residual normalisation expected {} solution entries, got {}",
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
            "LDU L1 residual normalisation factor is not finite".to_string(),
        ));
    }
    Ok(factor + OPENFOAM_RELATIVE_TOLERANCE_SMALL)
}

fn l1_norm(values: &[f64]) -> f64 {
    values.iter().map(|value| value.abs()).sum()
}

#[cfg(test)]
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

    for row in 0..system.matrix.rows() {
        let start = system.matrix.row_offsets()[row];
        let end = system.matrix.row_offsets()[row + 1];
        let diagonal_entry = (start..end)
            .find(|entry| system.matrix.col_indices()[*entry] == row)
            .ok_or_else(|| invalid_input(format!("row {row} has no diagonal entry")))?;
        system.matrix.values_mut()[diagonal_entry] /= relaxation;
    }

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
    for row_index in 0..system.matrix.rows() {
        let start = system.matrix.row_offsets()[row_index];
        let end = system.matrix.row_offsets()[row_index + 1];
        if row_index == reference_cell {
            let diagonal_entry = (start..end)
                .find(|entry| system.matrix.col_indices()[*entry] == reference_cell)
                .ok_or_else(|| {
                    invalid_input(format!(
                        "pressure reference row {reference_cell} has no diagonal entry"
                    ))
                })?;
            system.matrix.values_mut()[start..end].fill(0.0);
            system.matrix.values_mut()[diagonal_entry] = 1.0;
            system.rhs[row_index] = reference_value;
            continue;
        }
        if let Some(entry) =
            (start..end).find(|entry| system.matrix.col_indices()[*entry] == reference_cell)
        {
            let coefficient = system.matrix.values()[entry];
            system.matrix.values_mut()[entry] = 0.0;
            system.rhs[row_index] -= coefficient * reference_value;
        }
    }
    system.matrix.validate_values()?;
    Ok(())
}

fn map_cg_preconditioner(preconditioner: LaminarSimplePreconditioner) -> CgPreconditioner {
    match preconditioner {
        LaminarSimplePreconditioner::None => CgPreconditioner::None,
        LaminarSimplePreconditioner::Diagonal => CgPreconditioner::Diagonal,
        LaminarSimplePreconditioner::IncompleteCholesky => CgPreconditioner::IncompleteCholesky,
        LaminarSimplePreconditioner::Dic => CgPreconditioner::Dic,
        LaminarSimplePreconditioner::Fdic => CgPreconditioner::Fdic,
    }
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

fn checked_update_metrics(
    velocity: &[Point3],
    pressure: &[f64],
    phi: &[f64],
    continuity: ContinuitySummary,
) -> Option<()> {
    checked_l2_norm(
        velocity
            .iter()
            .flat_map(|value| [value.x, value.y, value.z]),
    )?;
    checked_gauge_invariant_scalar_l2(pressure)?;
    checked_l2_norm(phi.iter().copied())?;
    is_finite_continuity(continuity).then_some(())
}

#[allow(clippy::too_many_arguments)]
fn checked_candidate_update_metrics(
    previous_velocity: &[Point3],
    previous_pressure: &[f64],
    previous_phi: &[f64],
    velocity: &[Point3],
    pressure: &[f64],
    phi: &[f64],
    continuity: ContinuitySummary,
) -> Option<SimpleUpdateMetrics> {
    checked_update_metrics(velocity, pressure, phi, continuity)?;
    let velocity_change_l2 = checked_relative_vector_field_change_l2(previous_velocity, velocity)?;
    let pressure_change_l2 =
        checked_relative_gauge_scalar_field_change_l2(previous_pressure, pressure)?;
    checked_relative_scalar_field_change_l2(previous_phi, phi)?;
    Some(SimpleUpdateMetrics {
        velocity_change_l2,
        pressure_change_l2,
    })
}

fn checked_gauge_invariant_scalar_l2(values: &[f64]) -> Option<f64> {
    let reference = values.first().copied().unwrap_or(0.0);
    checked_l2_norm(values.iter().map(|value| *value - reference))
}

#[derive(Clone, Copy, Debug)]
struct CheckedScaledSumSquares {
    scale: f64,
    scaled_sum_squares: f64,
}

impl CheckedScaledSumSquares {
    fn new() -> Self {
        Self {
            scale: 0.0,
            scaled_sum_squares: 1.0,
        }
    }

    fn add(&mut self, value: f64) -> Option<()> {
        if !value.is_finite() {
            return None;
        }
        let absolute = value.abs();
        if absolute == 0.0 {
            return Some(());
        }
        if self.scale < absolute {
            let ratio = self.scale / absolute;
            self.scaled_sum_squares = 1.0 + self.scaled_sum_squares * ratio * ratio;
            self.scale = absolute;
        } else {
            let ratio = absolute / self.scale;
            self.scaled_sum_squares += ratio * ratio;
        }
        self.scaled_sum_squares.is_finite().then_some(())
    }

    fn finish(self) -> Option<f64> {
        if self.scale == 0.0 {
            return Some(0.0);
        }
        let norm = self.scale * self.scaled_sum_squares.sqrt();
        norm.is_finite().then_some(norm)
    }
}

fn checked_l2_norm(values: impl IntoIterator<Item = f64>) -> Option<f64> {
    let mut accumulator = CheckedScaledSumSquares::new();
    for value in values {
        accumulator.add(value)?;
    }
    accumulator.finish()
}

fn points_are_finite(values: &[Point3]) -> bool {
    values
        .iter()
        .all(|value| value.x.is_finite() && value.y.is_finite() && value.z.is_finite())
}

fn scalars_are_finite(values: &[f64]) -> bool {
    values.iter().all(|value| value.is_finite())
}

fn checked_relative_vector_field_change_l2(before: &[Point3], after: &[Point3]) -> Option<f64> {
    if before.len() != after.len() {
        return None;
    }
    let mut delta_norm = CheckedScaledSumSquares::new();
    for (before, after) in before.iter().zip(after) {
        let dx = after.x - before.x;
        let dy = after.y - before.y;
        let dz = after.z - before.z;
        delta_norm.add(dx)?;
        delta_norm.add(dy)?;
        delta_norm.add(dz)?;
    }
    let delta_norm = delta_norm.finish()?;
    let before_norm = checked_l2_norm(before.iter().flat_map(|value| [value.x, value.y, value.z]))?;
    let after_norm = checked_l2_norm(after.iter().flat_map(|value| [value.x, value.y, value.z]))?;
    checked_relative_change_ratio(delta_norm, before_norm, after_norm)
}

fn checked_relative_scalar_field_change_l2(before: &[f64], after: &[f64]) -> Option<f64> {
    if before.len() != after.len() {
        return None;
    }
    let mut delta_norm = CheckedScaledSumSquares::new();
    for (before, after) in before.iter().zip(after) {
        let delta = *after - *before;
        delta_norm.add(delta)?;
    }
    let delta_norm = delta_norm.finish()?;
    let before_norm = checked_l2_norm(before.iter().copied())?;
    let after_norm = checked_l2_norm(after.iter().copied())?;
    checked_relative_change_ratio(delta_norm, before_norm, after_norm)
}

fn checked_relative_gauge_scalar_field_change_l2(before: &[f64], after: &[f64]) -> Option<f64> {
    if before.len() != after.len() {
        return None;
    }
    let before_reference = before.first().copied().unwrap_or(0.0);
    let after_reference = after.first().copied().unwrap_or(0.0);
    let mut delta_norm = CheckedScaledSumSquares::new();
    for (before, after) in before.iter().zip(after) {
        let before_centered = *before - before_reference;
        let after_centered = *after - after_reference;
        let delta = after_centered - before_centered;
        if !before_centered.is_finite() || !after_centered.is_finite() {
            return None;
        }
        delta_norm.add(delta)?;
    }
    let delta_norm = delta_norm.finish()?;
    let before_norm = checked_gauge_invariant_scalar_l2(before)?;
    let after_norm = checked_gauge_invariant_scalar_l2(after)?;
    checked_relative_change_ratio(delta_norm, before_norm, after_norm)
}

fn checked_relative_change_ratio(
    delta_norm: f64,
    before_norm: f64,
    after_norm: f64,
) -> Option<f64> {
    let scale = before_norm.max(after_norm);
    if scale == 0.0 {
        return (delta_norm == 0.0).then_some(0.0);
    }
    let ratio = delta_norm / scale;
    ratio.is_finite().then_some(ratio)
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
#[cfg(test)]
fn assemble_momentum_component_system(
    mesh: &SolverRuntimeMeshData,
    momentum_csr_pattern: &MomentumCsrPattern,
    diffusivity: f64,
    density: f64,
    flux: &[f64],
    volumetric_source: &[f64],
    boundary: &[ScalarFaceTreatment],
    old_values: &[f64],
    old_gradient: Option<&[Point3]>,
    convection_scheme: LaminarSimpleConvectionScheme,
) -> Result<ScalarComponentSystem> {
    let face_addressing = CompactSimpleFaceAddressing::from_mesh(mesh)?;
    assemble_momentum_component_system_with_addressing(
        mesh,
        &face_addressing,
        momentum_csr_pattern,
        diffusivity,
        density,
        flux,
        volumetric_source,
        boundary,
        old_values,
        old_gradient,
        convection_scheme,
    )
}

#[allow(clippy::too_many_arguments)]
fn assemble_momentum_component_system_with_addressing(
    mesh: &SolverRuntimeMeshData,
    face_addressing: &CompactSimpleFaceAddressing,
    momentum_csr_pattern: &MomentumCsrPattern,
    diffusivity: f64,
    density: f64,
    flux: &[f64],
    volumetric_source: &[f64],
    boundary: &[ScalarFaceTreatment],
    old_values: &[f64],
    old_gradient: Option<&[Point3]>,
    convection_scheme: LaminarSimpleConvectionScheme,
) -> Result<ScalarComponentSystem> {
    if momentum_csr_pattern.sparsity.rows() != mesh.cells
        || momentum_csr_pattern.sparsity.cols() != mesh.cells
    {
        return Err(invalid_input(format!(
            "momentum CSR pattern is {}x{}, expected {}x{}",
            momentum_csr_pattern.sparsity.rows(),
            momentum_csr_pattern.sparsity.cols(),
            mesh.cells,
            mesh.cells
        )));
    }
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

    let mut values = vec![0.0; momentum_csr_pattern.sparsity.nnz()];
    let mut rhs = volumetric_source
        .iter()
        .zip(&mesh.cell_volumes)
        .map(|(source, volume)| source * volume)
        .collect::<Vec<_>>();

    for (face_index, treatment) in boundary.iter().enumerate() {
        let owner = face_addressing.owner(face_index);
        let mass_flux = density * flux[face_index];
        if !mass_flux.is_finite() {
            return Err(invalid_input(format!(
                "momentum face {face_index} mass flux must be finite, got {mass_flux}"
            )));
        }

        if let Some(neighbour) = face_addressing.neighbour(face_index) {
            let coefficient = face_diffusion_coefficient(
                diffusivity,
                mesh.face_area_vectors[face_index],
                mesh.cell_centres[owner],
                mesh.cell_centres[neighbour],
                face_index,
            )?;
            let (owner_neighbour_slot, neighbour_owner_slot) =
                momentum_csr_pattern.internal_slots(face_index);
            add_csr_value(
                &mut values,
                momentum_csr_pattern.diagonal_slot(owner),
                coefficient,
            );
            add_csr_value(&mut values, owner_neighbour_slot, -coefficient);
            add_csr_value(
                &mut values,
                momentum_csr_pattern.diagonal_slot(neighbour),
                coefficient,
            );
            add_csr_value(&mut values, neighbour_owner_slot, -coefficient);
            add_internal_convection(
                &mut values,
                &mut rhs,
                momentum_csr_pattern,
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
                add_csr_value(
                    &mut values,
                    momentum_csr_pattern.diagonal_slot(owner),
                    coefficient,
                );
                rhs[owner] += coefficient * value;
                add_boundary_convection(
                    &mut values,
                    &mut rhs,
                    momentum_csr_pattern,
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
                add_csr_value(
                    &mut values,
                    momentum_csr_pattern.diagonal_slot(owner),
                    coefficient,
                );
                rhs[owner] += coefficient * value;
                add_boundary_convection(
                    &mut values,
                    &mut rhs,
                    momentum_csr_pattern,
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
                    &mut values,
                    &mut rhs,
                    momentum_csr_pattern,
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

    if convection_scheme.is_bounded() {
        let net_flux = net_cell_flux_with_addressing(mesh, face_addressing, flux)?;
        for (cell, cell_flux) in net_flux.into_iter().enumerate() {
            let correction = checked_product(
                density,
                cell_flux,
                format!("bounded momentum cell {cell} net mass flux"),
            )?;
            add_csr_value(
                &mut values,
                momentum_csr_pattern.diagonal_slot(cell),
                -correction,
            );
        }
    }

    let matrix = CsrMatrix::from_pattern(&momentum_csr_pattern.sparsity, values)?;

    Ok(ScalarComponentSystem { matrix, rhs })
}

fn add_internal_upwind_convection(
    values: &mut [f64],
    momentum_csr_pattern: &MomentumCsrPattern,
    owner: usize,
    neighbour: usize,
    face_index: usize,
    mass_flux: f64,
) {
    let (owner_neighbour_slot, neighbour_owner_slot) =
        momentum_csr_pattern.internal_slots(face_index);
    if mass_flux >= 0.0 {
        add_csr_value(values, momentum_csr_pattern.diagonal_slot(owner), mass_flux);
        add_csr_value(values, neighbour_owner_slot, -mass_flux);
    } else {
        add_csr_value(values, owner_neighbour_slot, mass_flux);
        add_csr_value(
            values,
            momentum_csr_pattern.diagonal_slot(neighbour),
            -mass_flux,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn add_internal_convection(
    values: &mut [f64],
    rhs: &mut [f64],
    momentum_csr_pattern: &MomentumCsrPattern,
    mesh: &SolverRuntimeMeshData,
    old_values: &[f64],
    old_gradient: Option<&[Point3]>,
    owner: usize,
    neighbour: usize,
    face_index: usize,
    mass_flux: f64,
    scheme: LaminarSimpleConvectionScheme,
) {
    add_internal_upwind_convection(
        values,
        momentum_csr_pattern,
        owner,
        neighbour,
        face_index,
        mass_flux,
    );
    if scheme.uses_linear_upwind() {
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
    values: &mut [f64],
    rhs: &mut [f64],
    momentum_csr_pattern: &MomentumCsrPattern,
    owner: usize,
    value: f64,
    mass_flux: f64,
) {
    if mass_flux < 0.0 {
        rhs[owner] += -mass_flux * value;
    } else {
        add_csr_value(values, momentum_csr_pattern.diagonal_slot(owner), mass_flux);
    }
}

#[allow(clippy::too_many_arguments)]
fn add_boundary_convection(
    values: &mut [f64],
    rhs: &mut [f64],
    momentum_csr_pattern: &MomentumCsrPattern,
    mesh: &SolverRuntimeMeshData,
    old_values: &[f64],
    old_gradient: Option<&[Point3]>,
    owner: usize,
    face_index: usize,
    value: f64,
    mass_flux: f64,
    scheme: LaminarSimpleConvectionScheme,
) {
    add_boundary_upwind_convection(values, rhs, momentum_csr_pattern, owner, value, mass_flux);
    if mass_flux >= 0.0
        && scheme.uses_linear_upwind()
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
    values: &mut [f64],
    rhs: &mut [f64],
    momentum_csr_pattern: &MomentumCsrPattern,
    mesh: &SolverRuntimeMeshData,
    old_values: &[f64],
    old_gradient: Option<&[Point3]>,
    owner: usize,
    face_index: usize,
    mass_flux: f64,
    scheme: LaminarSimpleConvectionScheme,
) {
    add_csr_value(values, momentum_csr_pattern.diagonal_slot(owner), mass_flux);
    if scheme.uses_linear_upwind()
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

#[cfg(test)]
fn assemble_variable_scalar_component_system(
    mesh: &SolverRuntimeMeshData,
    cell_diffusivity: &[f64],
    volumetric_source: &[f64],
    boundary: &[ScalarFaceTreatment],
) -> Result<ScalarComponentSystem> {
    let pattern = MomentumCsrPattern::from_mesh(mesh)?;
    let pressure_geometry = PressureGeometryCache::from_mesh(mesh);
    let mut system = ScalarComponentSystem {
        matrix: CsrMatrix::from_pattern(&pattern.sparsity, vec![0.0; pattern.sparsity.nnz()])?,
        rhs: vec![0.0; mesh.cells],
    };
    assemble_variable_scalar_component_system_into_with_pressure_geometry(
        mesh,
        &pattern,
        &pressure_geometry,
        cell_diffusivity,
        volumetric_source,
        boundary,
        &mut system,
    )?;
    Ok(system)
}

#[cfg(test)]
fn assemble_variable_scalar_component_system_into_with_pressure_geometry(
    mesh: &SolverRuntimeMeshData,
    pattern: &MomentumCsrPattern,
    pressure_geometry: &PressureGeometryCache,
    cell_diffusivity: &[f64],
    volumetric_source: &[f64],
    boundary: &[ScalarFaceTreatment],
    system: &mut ScalarComponentSystem,
) -> Result<()> {
    let face_addressing = CompactSimpleFaceAddressing::from_mesh(mesh)?;
    assemble_variable_scalar_component_system_into_with_pressure_geometry_and_addressing(
        mesh,
        pattern,
        pressure_geometry,
        &face_addressing,
        cell_diffusivity,
        volumetric_source,
        boundary,
        system,
    )
}

#[allow(clippy::too_many_arguments)]
fn assemble_variable_scalar_component_system_into_with_pressure_geometry_and_addressing(
    mesh: &SolverRuntimeMeshData,
    pattern: &MomentumCsrPattern,
    pressure_geometry: &PressureGeometryCache,
    face_addressing: &CompactSimpleFaceAddressing,
    cell_diffusivity: &[f64],
    volumetric_source: &[f64],
    boundary: &[ScalarFaceTreatment],
    system: &mut ScalarComponentSystem,
) -> Result<()> {
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
    if system.rhs.len() != mesh.cells || !system.matrix.shares_sparsity_with(&pattern.sparsity) {
        return Err(invalid_input(
            "variable scalar component workspace does not match the runtime mesh sparsity"
                .to_string(),
        ));
    }

    system.matrix.values_mut().fill(0.0);
    for ((rhs, source), volume) in system
        .rhs
        .iter_mut()
        .zip(volumetric_source)
        .zip(&mesh.cell_volumes)
    {
        *rhs = source * volume;
    }
    for (face_index, treatment) in boundary.iter().enumerate() {
        let owner = face_addressing.owner(face_index);
        if let Some(neighbour) = face_addressing.neighbour(face_index) {
            let coefficient = cached_variable_face_diffusion_coefficient(
                pressure_geometry,
                cell_diffusivity,
                owner,
                Some(neighbour),
                face_index,
            )?;
            let (owner_neighbour, neighbour_owner) = pattern.internal_slots(face_index);
            add_csr_value(
                system.matrix.values_mut(),
                pattern.diagonal_slot(owner),
                coefficient,
            );
            add_csr_value(system.matrix.values_mut(), owner_neighbour, -coefficient);
            add_csr_value(
                system.matrix.values_mut(),
                pattern.diagonal_slot(neighbour),
                coefficient,
            );
            add_csr_value(system.matrix.values_mut(), neighbour_owner, -coefficient);
            continue;
        }

        match *treatment {
            ScalarFaceTreatment::FixedValue(value) => {
                let coefficient = cached_variable_face_diffusion_coefficient(
                    pressure_geometry,
                    cell_diffusivity,
                    owner,
                    None,
                    face_index,
                )?;
                add_csr_value(
                    system.matrix.values_mut(),
                    pattern.diagonal_slot(owner),
                    coefficient,
                );
                system.rhs[owner] += coefficient * value;
            }
            ScalarFaceTreatment::InletOutlet(value) => {
                let coefficient = cached_variable_face_diffusion_coefficient(
                    pressure_geometry,
                    cell_diffusivity,
                    owner,
                    None,
                    face_index,
                )?;
                add_csr_value(
                    system.matrix.values_mut(),
                    pattern.diagonal_slot(owner),
                    coefficient,
                );
                system.rhs[owner] += coefficient * value;
            }
            ScalarFaceTreatment::FixedGradient(gradient) => {
                let flux = cached_fixed_gradient_pressure_flux(
                    pressure_geometry,
                    cell_diffusivity,
                    owner,
                    face_index,
                    gradient,
                )?;
                system.rhs[owner] -= flux;
            }
            ScalarFaceTreatment::ZeroGradient | ScalarFaceTreatment::Constraint => {}
        }
    }

    system.matrix.validate_values()?;
    Ok(())
}

#[cfg(test)]
fn assemble_variable_scalar_component_system_into(
    mesh: &SolverRuntimeMeshData,
    pattern: &MomentumCsrPattern,
    cell_diffusivity: &[f64],
    volumetric_source: &[f64],
    boundary: &[ScalarFaceTreatment],
    system: &mut ScalarComponentSystem,
) -> Result<()> {
    let pressure_geometry = PressureGeometryCache::from_mesh(mesh);
    assemble_variable_scalar_component_system_into_with_pressure_geometry(
        mesh,
        pattern,
        &pressure_geometry,
        cell_diffusivity,
        volumetric_source,
        boundary,
        system,
    )
}

#[cfg(test)]
fn compute_face_flux(
    mesh: &SolverRuntimeMeshData,
    velocity: &[Point3],
    boundary: &[VectorFaceTreatment],
) -> Result<Vec<f64>> {
    let face_addressing = CompactSimpleFaceAddressing::from_mesh(mesh)?;
    compute_face_flux_with_addressing(mesh, &face_addressing, velocity, boundary)
}

fn compute_face_flux_with_addressing(
    mesh: &SolverRuntimeMeshData,
    face_addressing: &CompactSimpleFaceAddressing,
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
    let mut flux = flow_try_filled_vec(mesh.faces, 0.0)?;
    for (face_index, face_flux) in flux.iter_mut().enumerate() {
        let face_velocity = face_vector_value_with_addressing(
            mesh,
            face_addressing,
            velocity,
            boundary,
            face_index,
        );
        *face_flux = dot(face_velocity, mesh.face_area_vectors[face_index]);
    }
    Ok(flux)
}

#[cfg(test)]
fn compute_phi_hby_a(
    mesh: &SolverRuntimeMeshData,
    hby_a: &[Point3],
    velocity_boundary: &[VectorFaceTreatment],
) -> Result<Vec<f64>> {
    compute_face_flux(mesh, hby_a, velocity_boundary)
}

fn compute_phi_hby_a_with_addressing(
    mesh: &SolverRuntimeMeshData,
    face_addressing: &CompactSimpleFaceAddressing,
    hby_a: &[Point3],
    velocity_boundary: &[VectorFaceTreatment],
) -> Result<Vec<f64>> {
    compute_face_flux_with_addressing(mesh, face_addressing, hby_a, velocity_boundary)
}

#[cfg(test)]
fn pressure_correction_flux_with_geometry(
    mesh: &SolverRuntimeMeshData,
    pressure_geometry: &PressureGeometryCache,
    pressure_correction: &[f64],
    r_au: &[f64],
    boundary: &[ScalarFaceTreatment],
) -> Result<Vec<f64>> {
    let face_addressing = CompactSimpleFaceAddressing::from_mesh(mesh)?;
    pressure_correction_flux_with_geometry_and_addressing(
        mesh,
        pressure_geometry,
        &face_addressing,
        pressure_correction,
        r_au,
        boundary,
    )
}

fn pressure_correction_flux_with_geometry_and_addressing(
    mesh: &SolverRuntimeMeshData,
    pressure_geometry: &PressureGeometryCache,
    face_addressing: &CompactSimpleFaceAddressing,
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
        let owner = face_addressing.owner(face_index);
        if let Some(neighbour) = face_addressing.neighbour(face_index) {
            let coefficient = cached_variable_face_diffusion_coefficient(
                pressure_geometry,
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
                let coefficient = cached_variable_face_diffusion_coefficient(
                    pressure_geometry,
                    r_au,
                    owner,
                    None,
                    face_index,
                )?;
                flux[face_index] = coefficient * (pressure_correction[owner] - value);
            }
            ScalarFaceTreatment::InletOutlet(value) => {
                let coefficient = cached_variable_face_diffusion_coefficient(
                    pressure_geometry,
                    r_au,
                    owner,
                    None,
                    face_index,
                )?;
                flux[face_index] = coefficient * (pressure_correction[owner] - value);
            }
            ScalarFaceTreatment::FixedGradient(gradient) => {
                flux[face_index] = cached_fixed_gradient_pressure_flux(
                    pressure_geometry,
                    r_au,
                    owner,
                    face_index,
                    gradient,
                )?;
            }
            ScalarFaceTreatment::ZeroGradient | ScalarFaceTreatment::Constraint => {}
        }
    }
    Ok(flux)
}

#[cfg(test)]
fn pressure_correction_flux(
    mesh: &SolverRuntimeMeshData,
    pressure_correction: &[f64],
    r_au: &[f64],
    boundary: &[ScalarFaceTreatment],
) -> Result<Vec<f64>> {
    let pressure_geometry = PressureGeometryCache::from_mesh(mesh);
    pressure_correction_flux_with_geometry(
        mesh,
        &pressure_geometry,
        pressure_correction,
        r_au,
        boundary,
    )
}

#[cfg(test)]
fn pressure_equation_flux_with_geometry(
    mesh: &SolverRuntimeMeshData,
    pressure_geometry: &PressureGeometryCache,
    pressure: &[f64],
    r_au: &[f64],
    boundary: &[ScalarFaceTreatment],
) -> Result<Vec<f64>> {
    Ok(
        pressure_correction_flux_with_geometry(mesh, pressure_geometry, pressure, r_au, boundary)?
            .into_iter()
            .map(|flux| -flux)
            .collect(),
    )
}

fn pressure_equation_flux_with_geometry_and_addressing(
    mesh: &SolverRuntimeMeshData,
    pressure_geometry: &PressureGeometryCache,
    face_addressing: &CompactSimpleFaceAddressing,
    pressure: &[f64],
    r_au: &[f64],
    boundary: &[ScalarFaceTreatment],
) -> Result<Vec<f64>> {
    Ok(pressure_correction_flux_with_geometry_and_addressing(
        mesh,
        pressure_geometry,
        face_addressing,
        pressure,
        r_au,
        boundary,
    )?
    .into_iter()
    .map(|flux| -flux)
    .collect())
}

#[cfg(test)]
fn pressure_equation_flux(
    mesh: &SolverRuntimeMeshData,
    pressure: &[f64],
    r_au: &[f64],
    boundary: &[ScalarFaceTreatment],
) -> Result<Vec<f64>> {
    let pressure_geometry = PressureGeometryCache::from_mesh(mesh);
    pressure_equation_flux_with_geometry(mesh, &pressure_geometry, pressure, r_au, boundary)
}

#[cfg(test)]
fn consistent_phi_hby_a_pressure_correction_with_geometry(
    mesh: &SolverRuntimeMeshData,
    pressure_geometry: &PressureGeometryCache,
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

    Ok(
        pressure_correction_flux_with_geometry(
            mesh,
            pressure_geometry,
            pressure,
            &delta,
            boundary,
        )?
        .into_iter()
        .map(|flux| -flux)
        .collect(),
    )
}

fn consistent_phi_hby_a_pressure_correction_with_geometry_and_addressing(
    mesh: &SolverRuntimeMeshData,
    pressure_geometry: &PressureGeometryCache,
    face_addressing: &CompactSimpleFaceAddressing,
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

    Ok(pressure_correction_flux_with_geometry_and_addressing(
        mesh,
        pressure_geometry,
        face_addressing,
        pressure,
        &delta,
        boundary,
    )?
    .into_iter()
    .map(|flux| -flux)
    .collect())
}

#[cfg(test)]
fn consistent_phi_hby_a_pressure_correction(
    mesh: &SolverRuntimeMeshData,
    pressure: &[f64],
    boundary: &[ScalarFaceTreatment],
    r_au: &[f64],
    r_at_u: &[f64],
) -> Result<Vec<f64>> {
    let pressure_geometry = PressureGeometryCache::from_mesh(mesh);
    consistent_phi_hby_a_pressure_correction_with_geometry(
        mesh,
        &pressure_geometry,
        pressure,
        boundary,
        r_au,
        r_at_u,
    )
}

#[cfg(test)]
fn non_orthogonal_pressure_flux_correction_with_geometry(
    mesh: &SolverRuntimeMeshData,
    pressure_geometry: &PressureGeometryCache,
    scalar_gradient_geometry: &ScalarGradientGeometry,
    pressure: &[f64],
    r_at_u: &[f64],
    boundary: &[ScalarFaceTreatment],
    gradient_scheme: LaminarSimpleGradientScheme,
) -> Result<Vec<f64>> {
    let face_addressing = CompactSimpleFaceAddressing::from_mesh(mesh)?;
    let weighted_least_squares = gradient_scheme
        .uses_weighted_least_squares()
        .then(|| {
            WeightedLeastSquaresGeometry::from_mesh(
                mesh,
                scalar_gradient_geometry,
                &face_addressing,
            )
        })
        .transpose()?;
    non_orthogonal_pressure_flux_correction_with_geometry_and_addressing(
        mesh,
        pressure_geometry,
        scalar_gradient_geometry,
        weighted_least_squares.as_ref(),
        &face_addressing,
        pressure,
        r_at_u,
        boundary,
        gradient_scheme,
    )
}

#[allow(clippy::too_many_arguments)]
fn non_orthogonal_pressure_flux_correction_with_geometry_and_addressing(
    mesh: &SolverRuntimeMeshData,
    pressure_geometry: &PressureGeometryCache,
    scalar_gradient_geometry: &ScalarGradientGeometry,
    weighted_least_squares: Option<&WeightedLeastSquaresGeometry>,
    face_addressing: &CompactSimpleFaceAddressing,
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

    let pressure_gradient = scalar_gradient_with_scheme_and_addressing(
        mesh,
        scalar_gradient_geometry,
        weighted_least_squares,
        face_addressing,
        pressure,
        boundary,
        gradient_scheme,
    )?;
    let mut flux = vec![0.0; mesh.faces];
    for (face_index, treatment) in boundary.iter().enumerate() {
        let owner = face_addressing.owner(face_index);
        if let Some(neighbour) = face_addressing.neighbour(face_index) {
            let diffusivity = 0.5 * (r_at_u[owner] + r_at_u[neighbour]);
            let face_gradient = average(pressure_gradient[owner], pressure_gradient[neighbour]);
            let non_orthogonal_area =
                cached_non_orthogonal_area_vector(pressure_geometry, face_index)?;
            flux[face_index] = -diffusivity * dot(face_gradient, non_orthogonal_area);
            continue;
        }

        if matches!(
            treatment,
            ScalarFaceTreatment::FixedValue(_)
                | ScalarFaceTreatment::FixedGradient(_)
                | ScalarFaceTreatment::InletOutlet(_)
        ) {
            let non_orthogonal_area =
                cached_non_orthogonal_area_vector(pressure_geometry, face_index)?;
            flux[face_index] = -r_at_u[owner] * dot(pressure_gradient[owner], non_orthogonal_area);
        }
    }
    Ok(flux)
}

#[cfg(test)]
fn non_orthogonal_pressure_flux_correction(
    mesh: &SolverRuntimeMeshData,
    scalar_gradient_geometry: &ScalarGradientGeometry,
    pressure: &[f64],
    r_at_u: &[f64],
    boundary: &[ScalarFaceTreatment],
    gradient_scheme: LaminarSimpleGradientScheme,
) -> Result<Vec<f64>> {
    let pressure_geometry = PressureGeometryCache::from_mesh(mesh);
    non_orthogonal_pressure_flux_correction_with_geometry(
        mesh,
        &pressure_geometry,
        scalar_gradient_geometry,
        pressure,
        r_at_u,
        boundary,
        gradient_scheme,
    )
}

fn cached_non_orthogonal_area_vector(
    pressure_geometry: &PressureGeometryCache,
    face_index: usize,
) -> Result<Point3> {
    let geometry = pressure_geometry.face(face_index);
    let distance = geometry.centre_distance;
    if !distance.is_finite() || distance <= f64::EPSILON {
        return Err(invalid_input(format!(
            "face {face_index} has non-positive non-orthogonal correction distance {distance}"
        )));
    }
    Ok(geometry.non_orthogonal_area_vector)
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

fn corrected_pressure_equation_flux(
    pressure_equation_flux: &[f64],
    pressure_flux: &[f64],
) -> Result<Vec<f64>> {
    subtract_face_fluxes(pressure_equation_flux, pressure_flux)
}

#[cfg(test)]
fn scalar_gradient(
    mesh: &SolverRuntimeMeshData,
    values: &[f64],
    boundary: &[ScalarFaceTreatment],
    scheme: LaminarSimpleGradientScheme,
) -> Result<Vec<Point3>> {
    let geometry = ScalarGradientGeometry::from_mesh(mesh)?;
    let face_addressing = CompactSimpleFaceAddressing::from_mesh(mesh)?;
    let weighted_least_squares = scheme
        .uses_weighted_least_squares()
        .then(|| WeightedLeastSquaresGeometry::from_mesh(mesh, &geometry, &face_addressing))
        .transpose()?;
    scalar_gradient_with_scheme_and_addressing(
        mesh,
        &geometry,
        weighted_least_squares.as_ref(),
        &face_addressing,
        values,
        boundary,
        scheme,
    )
}

#[cfg(test)]
fn scalar_gradient_with_geometry(
    mesh: &SolverRuntimeMeshData,
    geometry: &ScalarGradientGeometry,
    values: &[f64],
    boundary: &[ScalarFaceTreatment],
    scheme: LaminarSimpleGradientScheme,
) -> Result<Vec<Point3>> {
    let face_addressing = CompactSimpleFaceAddressing::from_mesh(mesh)?;
    let weighted_least_squares = scheme
        .uses_weighted_least_squares()
        .then(|| WeightedLeastSquaresGeometry::from_mesh(mesh, geometry, &face_addressing))
        .transpose()?;
    scalar_gradient_with_scheme_and_addressing(
        mesh,
        geometry,
        weighted_least_squares.as_ref(),
        &face_addressing,
        values,
        boundary,
        scheme,
    )
}

#[cfg(test)]
fn vector_convection_divergence(
    mesh: &SolverRuntimeMeshData,
    velocity: &[Point3],
    boundary: &[VectorFaceTreatment],
    flux: &[f64],
    scheme: LaminarSimpleConvectionScheme,
    gradient_scheme: LaminarSimpleGradientScheme,
) -> Result<Vec<Point3>> {
    let scalar_gradient_geometry = ScalarGradientGeometry::from_mesh(mesh)?;
    let face_addressing = CompactSimpleFaceAddressing::from_mesh(mesh)?;
    let effective_gradient_scheme = scheme.gradient_scheme(gradient_scheme);
    let weighted_least_squares = (scheme.uses_linear_upwind()
        && effective_gradient_scheme.uses_weighted_least_squares())
    .then(|| {
        WeightedLeastSquaresGeometry::from_mesh(mesh, &scalar_gradient_geometry, &face_addressing)
    })
    .transpose()?;
    vector_convection_divergence_with_addressing(
        mesh,
        &scalar_gradient_geometry,
        weighted_least_squares.as_ref(),
        &face_addressing,
        velocity,
        boundary,
        flux,
        scheme,
        gradient_scheme,
    )
}

#[allow(clippy::too_many_arguments)]
fn vector_convection_divergence_with_addressing(
    mesh: &SolverRuntimeMeshData,
    scalar_gradient_geometry: &ScalarGradientGeometry,
    weighted_least_squares: Option<&WeightedLeastSquaresGeometry>,
    face_addressing: &CompactSimpleFaceAddressing,
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
    let gradients = if scheme.uses_linear_upwind() {
        Some(vector_component_gradients_with_scheme_and_addressing(
            mesh,
            scalar_gradient_geometry,
            weighted_least_squares,
            face_addressing,
            velocity,
            boundary,
            flux,
            scheme.gradient_scheme(gradient_scheme),
        )?)
    } else {
        None
    };
    let mut divergence = vec![zero(); mesh.cells];
    for (face_index, phi) in flux.iter().copied().enumerate() {
        if !phi.is_finite() {
            return Err(invalid_input(format!(
                "convection face {face_index} flux must be finite, got {phi}"
            )));
        }
        let owner = face_addressing.owner(face_index);
        let face_velocity = convection_face_vector_value_with_addressing(
            mesh,
            face_addressing,
            velocity,
            boundary,
            face_index,
            phi,
            scheme,
            gradients.as_ref(),
        );
        add_scaled(&mut divergence[owner], face_velocity, phi);
        if let Some(neighbour) = face_addressing.neighbour(face_index) {
            add_scaled(&mut divergence[neighbour], face_velocity, -phi);
        }
    }
    if scheme.is_bounded() {
        let net_flux = net_cell_flux_with_addressing(mesh, face_addressing, flux)?;
        for cell in 0..mesh.cells {
            divergence[cell].x -= checked_product(
                net_flux[cell],
                velocity[cell].x,
                format!("bounded convection cell {cell} x correction"),
            )?;
            divergence[cell].y -= checked_product(
                net_flux[cell],
                velocity[cell].y,
                format!("bounded convection cell {cell} y correction"),
            )?;
            divergence[cell].z -= checked_product(
                net_flux[cell],
                velocity[cell].z,
                format!("bounded convection cell {cell} z correction"),
            )?;
            if !divergence[cell].x.is_finite()
                || !divergence[cell].y.is_finite()
                || !divergence[cell].z.is_finite()
            {
                return Err(invalid_input(format!(
                    "bounded convection cell {cell} correction overflowed"
                )));
            }
        }
    }
    Ok(divergence)
}

#[cfg(test)]
fn net_cell_flux(mesh: &SolverRuntimeMeshData, flux: &[f64]) -> Result<Vec<f64>> {
    let face_addressing = CompactSimpleFaceAddressing::from_mesh(mesh)?;
    net_cell_flux_with_addressing(mesh, &face_addressing, flux)
}

fn net_cell_flux_with_addressing(
    mesh: &SolverRuntimeMeshData,
    face_addressing: &CompactSimpleFaceAddressing,
    flux: &[f64],
) -> Result<Vec<f64>> {
    if flux.len() != mesh.faces {
        return Err(invalid_input(format!(
            "flux has {} values, expected {} mesh faces",
            flux.len(),
            mesh.faces
        )));
    }
    let mut net = flow_try_filled_vec(mesh.cells, 0.0)?;
    for (face_index, phi) in flux.iter().copied().enumerate() {
        if !phi.is_finite() {
            return Err(invalid_input(format!(
                "face {face_index} flux must be finite, got {phi}"
            )));
        }
        let owner = face_addressing.owner(face_index);
        net[owner] += phi;
        if !net[owner].is_finite() {
            return Err(invalid_input(format!(
                "cell {owner} net flux overflowed at face {face_index}"
            )));
        }
        if let Some(neighbour) = face_addressing.neighbour(face_index) {
            net[neighbour] -= phi;
            if !net[neighbour].is_finite() {
                return Err(invalid_input(format!(
                    "cell {neighbour} net flux overflowed at face {face_index}"
                )));
            }
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

#[cfg(test)]
fn adjust_phi_hby_a_with_pressure_geometry(
    mesh: &SolverRuntimeMeshData,
    pressure_geometry: &PressureGeometryCache,
    velocity_boundary: &[VectorFaceTreatment],
    pressure_boundary: &[ScalarFaceTreatment],
    phi_hby_a: &mut [f64],
) -> Result<AdjustPhiSummary> {
    let face_addressing = CompactSimpleFaceAddressing::from_mesh(mesh)?;
    adjust_phi_hby_a_with_pressure_geometry_and_addressing(
        mesh,
        pressure_geometry,
        &face_addressing,
        velocity_boundary,
        pressure_boundary,
        phi_hby_a,
    )
}

fn adjust_phi_hby_a_with_pressure_geometry_and_addressing(
    mesh: &SolverRuntimeMeshData,
    pressure_geometry: &PressureGeometryCache,
    face_addressing: &CompactSimpleFaceAddressing,
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

    let global_flux_before = boundary_global_flux_with_addressing(face_addressing, phi_hby_a);
    if !global_flux_before.is_finite() {
        return Err(invalid_input(format!(
            "adjustPhi global flux must be finite, got {global_flux_before}"
        )));
    }

    let mut adjustable_area = 0.0;
    let mut adjusted_faces = 0;
    for &face_index in face_addressing.boundary_faces() {
        if !is_adjustable_pressure_open_face(
            velocity_boundary[face_index],
            pressure_boundary[face_index],
            phi_hby_a[face_index],
        ) {
            continue;
        }
        let area = pressure_geometry.face(face_index).area_magnitude;
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

    for &face_index in face_addressing.boundary_faces() {
        if !is_adjustable_pressure_open_face(
            velocity_boundary[face_index],
            pressure_boundary[face_index],
            phi_hby_a[face_index],
        ) {
            continue;
        }
        let area = pressure_geometry.face(face_index).area_magnitude;
        if area.is_finite() && area > f64::EPSILON {
            phi_hby_a[face_index] += -global_flux_before * area / adjustable_area;
        }
    }

    Ok(AdjustPhiSummary {
        global_flux_before,
        global_flux_after: boundary_global_flux_with_addressing(face_addressing, phi_hby_a),
        adjusted_faces,
    })
}

#[cfg(test)]
fn adjust_phi_hby_a(
    mesh: &SolverRuntimeMeshData,
    velocity_boundary: &[VectorFaceTreatment],
    pressure_boundary: &[ScalarFaceTreatment],
    phi_hby_a: &mut [f64],
) -> Result<AdjustPhiSummary> {
    let pressure_geometry = PressureGeometryCache::from_mesh(mesh);
    adjust_phi_hby_a_with_pressure_geometry(
        mesh,
        &pressure_geometry,
        velocity_boundary,
        pressure_boundary,
        phi_hby_a,
    )
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

fn boundary_global_flux_with_addressing(
    face_addressing: &CompactSimpleFaceAddressing,
    flux: &[f64],
) -> f64 {
    let mut global_flux = 0.0;
    for &face_index in face_addressing.boundary_faces() {
        global_flux += flux[face_index];
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

fn summarize_continuity(net_flux: &[f64]) -> ContinuitySummary {
    let mut summary = ContinuitySummary::default();
    for value in net_flux {
        let abs = value.abs();
        summary.max_abs = summary.max_abs.max(abs);
        summary.sum_abs += abs;
        summary.global_sum += value;
    }
    summary.l2_norm = checked_l2_norm(net_flux.iter().copied()).unwrap_or(f64::INFINITY);
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
    let mut treatments = flow_try_filled_vec(mesh.faces, VectorFaceTreatment::ZeroGradient)?;
    for patch in &mesh.patches {
        let field_patch = field_patch(field, &patch.name)?;
        let patch_treatments = if is_constraint_patch(&patch.patch_type) {
            flow_try_filled_vec(patch.faces, VectorFaceTreatment::Constraint)?
        } else {
            match field_patch.patch_type.as_deref() {
                Some("fixedValue") => fixed_vector_patch_values(field, field_patch, patch.faces)?,
                Some("noSlip") => {
                    flow_try_filled_vec(patch.faces, VectorFaceTreatment::FixedValue(zero()))?
                }
                Some("inletOutlet" | "pressureInletOutletVelocity") => {
                    inlet_outlet_vector_patch_values(field, field_patch, patch.faces)?
                }
                Some("zeroGradient") => {
                    flow_try_filled_vec(patch.faces, VectorFaceTreatment::ZeroGradient)?
                }
                Some("empty" | "wedge" | "symmetryPlane") => {
                    flow_try_filled_vec(patch.faces, VectorFaceTreatment::Constraint)?
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
    let mut treatments = flow_try_filled_vec(mesh.faces, ScalarFaceTreatment::ZeroGradient)?;
    for patch in &mesh.patches {
        let field_patch = field_patch(field, &patch.name)?;
        let patch_treatments = if is_constraint_patch(&patch.patch_type) {
            flow_try_filled_vec(patch.faces, ScalarFaceTreatment::Constraint)?
        } else {
            match field_patch.patch_type.as_deref() {
                Some("fixedValue") => fixed_scalar_patch_values(field, field_patch, patch.faces)?,
                Some("fixedFluxPressure") => {
                    flow_try_filled_vec(patch.faces, ScalarFaceTreatment::FixedGradient(0.0))?
                }
                Some("inletOutlet") => {
                    inlet_outlet_scalar_patch_values(field, field_patch, patch.faces)?
                }
                Some("zeroGradient") => {
                    flow_try_filled_vec(patch.faces, ScalarFaceTreatment::ZeroGradient)?
                }
                Some("empty" | "wedge" | "symmetryPlane") => {
                    flow_try_filled_vec(patch.faces, ScalarFaceTreatment::Constraint)?
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

#[cfg(test)]
fn constrained_pressure_treatments_with_geometry(
    mesh: &SolverRuntimeMeshData,
    pressure_geometry: &PressureGeometryCache,
    pressure_boundary: &[ScalarFaceTreatment],
    velocity_boundary: &[VectorFaceTreatment],
    velocity: &[Point3],
    phi_hby_a: &[f64],
    r_au: &[f64],
) -> Result<Vec<ScalarFaceTreatment>> {
    let face_addressing = CompactSimpleFaceAddressing::from_mesh(mesh)?;
    constrained_pressure_treatments_with_geometry_and_addressing(
        mesh,
        pressure_geometry,
        &face_addressing,
        pressure_boundary,
        velocity_boundary,
        velocity,
        phi_hby_a,
        r_au,
    )
}

#[allow(clippy::too_many_arguments)]
fn constrained_pressure_treatments_with_geometry_and_addressing(
    mesh: &SolverRuntimeMeshData,
    pressure_geometry: &PressureGeometryCache,
    face_addressing: &CompactSimpleFaceAddressing,
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
    for &face_index in face_addressing.boundary_faces() {
        if !is_pressure_gradient_constrained_boundary(
            pressure_boundary[face_index],
            velocity_boundary[face_index],
            phi_hby_a[face_index],
        ) {
            continue;
        }
        let owner = face_addressing.owner(face_index);
        let area = pressure_geometry.face(face_index).area_magnitude;
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
        let u_flux = prescribed_boundary_velocity_flux_with_addressing(
            mesh,
            face_addressing,
            velocity,
            velocity_boundary,
            face_index,
            phi_hby_a[face_index],
        )
        .unwrap_or_else(|| {
            dot(
                face_vector_value_with_addressing(
                    mesh,
                    face_addressing,
                    velocity,
                    velocity_boundary,
                    face_index,
                ),
                mesh.face_area_vectors[face_index],
            )
        });
        let gradient = (phi_hby_a[face_index] - u_flux) / (area * face_r_au);
        constrained[face_index] = ScalarFaceTreatment::FixedGradient(gradient);
    }
    Ok(constrained)
}

#[cfg(test)]
fn constrained_pressure_treatments(
    mesh: &SolverRuntimeMeshData,
    pressure_boundary: &[ScalarFaceTreatment],
    velocity_boundary: &[VectorFaceTreatment],
    velocity: &[Point3],
    phi_hby_a: &[f64],
    r_au: &[f64],
) -> Result<Vec<ScalarFaceTreatment>> {
    let pressure_geometry = PressureGeometryCache::from_mesh(mesh);
    constrained_pressure_treatments_with_geometry(
        mesh,
        &pressure_geometry,
        pressure_boundary,
        velocity_boundary,
        velocity,
        phi_hby_a,
        r_au,
    )
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

fn prescribed_boundary_velocity_flux_with_addressing(
    mesh: &SolverRuntimeMeshData,
    face_addressing: &CompactSimpleFaceAddressing,
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
    if face_addressing.neighbour(face_index).is_some() {
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
    let mut treatments = flow_try_vec(faces)?;
    for chunk in values.as_slice().chunks_exact(3) {
        treatments.push(VectorFaceTreatment::FixedValue(Point3 {
            x: chunk[0],
            y: chunk[1],
            z: chunk[2],
        }));
    }
    Ok(treatments)
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
    let mut treatments = flow_try_vec(faces)?;
    for chunk in values.as_slice().chunks_exact(3) {
        treatments.push(VectorFaceTreatment::InletOutlet(Point3 {
            x: chunk[0],
            y: chunk[1],
            z: chunk[2],
        }));
    }
    Ok(treatments)
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
    let mut treatments = flow_try_vec(faces)?;
    treatments.extend(
        values
            .as_slice()
            .iter()
            .copied()
            .map(ScalarFaceTreatment::FixedValue),
    );
    Ok(treatments)
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
    let mut treatments = flow_try_vec(faces)?;
    treatments.extend(
        values
            .as_slice()
            .iter()
            .copied()
            .map(ScalarFaceTreatment::InletOutlet),
    );
    Ok(treatments)
}

enum PatchNumericValues<'a> {
    Owned(Vec<f64>),
    Borrowed(&'a [f64]),
}

impl PatchNumericValues<'_> {
    fn as_slice(&self) -> &[f64] {
        match self {
            Self::Owned(values) => values,
            Self::Borrowed(values) => values,
        }
    }
}

fn parse_patch_numeric_values<'a>(
    value: &'a FieldValueSummary,
    components: usize,
    faces: usize,
    field: &FieldFile,
    patch: &str,
) -> Result<PatchNumericValues<'a>> {
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
            let scalar_slots = faces.checked_mul(components).ok_or_else(|| {
                invalid_input(format!(
                    "field '{}' patch '{}' uniform boundary size overflowed",
                    field_label(field),
                    patch
                ))
            })?;
            let mut expanded = flow_try_vec(scalar_slots)?;
            for _ in 0..faces {
                expanded.extend(values.iter().copied());
            }
            Ok(PatchNumericValues::Owned(expanded))
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
            let expected = faces.checked_mul(components).ok_or_else(|| {
                invalid_input(format!(
                    "field '{}' patch '{}' nonuniform boundary size overflowed",
                    field_label(field),
                    patch
                ))
            })?;
            if values.len() != expected {
                return Err(invalid_input(format!(
                    "field '{}' patch '{}' nonuniform value has {} scalars, expected {}",
                    field_label(field),
                    patch,
                    values.len(),
                    expected
                )));
            }
            Ok(PatchNumericValues::Borrowed(values))
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
    let tokens = || {
        value
            .split(|character: char| {
                character.is_whitespace() || character == '(' || character == ')'
            })
            .filter(|token| !token.is_empty())
    };
    let mut values = flow_try_vec(tokens().count())?;
    for token in tokens() {
        values.push(token.parse::<f64>().map_err(|_| {
            invalid_input(format!(
                "field '{}' patch '{}' contains non-numeric token '{}'",
                field_label(field),
                patch,
                token
            ))
        })?);
    }
    Ok(values)
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

fn snapshot_runtime_initial_fields(runtime: &SolverRuntimeData) -> Result<(Vec<Point3>, &[f64])> {
    let velocity_index = runtime_field_index(runtime, "U", SolverStateFieldKind::VolVector, 3)?;
    let pressure_index = runtime_field_index(runtime, "p", SolverStateFieldKind::VolScalar, 1)?;
    validate_runtime_field(runtime, velocity_index, "U", 3)?;
    validate_runtime_field(runtime, pressure_index, "p", 1)?;

    let velocity_scalars = runtime.fields[velocity_index]
        .values
        .as_deref()
        .ok_or_else(|| {
            invalid_input(
                "runtime field 'U' initial payload was already consumed or not loaded".to_string(),
            )
        })?;
    let mut velocity = Vec::new();
    velocity
        .try_reserve_exact(runtime.mesh.cells)
        .map_err(|_| invalid_input("runtime vector field 'U' allocation failed".to_string()))?;
    for chunk in velocity_scalars.chunks_exact(3) {
        velocity.push(Point3 {
            x: chunk[0],
            y: chunk[1],
            z: chunk[2],
        });
    }

    let pressure = runtime.fields[pressure_index]
        .values
        .as_deref()
        .ok_or_else(|| {
            invalid_input(
                "runtime field 'p' initial payload was already consumed or not loaded".to_string(),
            )
        })?;
    Ok((velocity, pressure))
}

fn take_runtime_initial_fields(runtime: &mut SolverRuntimeData) -> Result<(Vec<Point3>, Vec<f64>)> {
    let velocity_index = runtime_field_index(runtime, "U", SolverStateFieldKind::VolVector, 3)?;
    let pressure_index = runtime_field_index(runtime, "p", SolverStateFieldKind::VolScalar, 1)?;
    validate_runtime_field(runtime, velocity_index, "U", 3)?;
    validate_runtime_field(runtime, pressure_index, "p", 1)?;

    // Point3 has no representation guarantee that permits reusing Vec<f64>'s
    // allocation safely. Build the bounded typed representation while the
    // original is still borrowed. Every fallible operation therefore finishes
    // before either one-shot payload is consumed.
    let velocity_scalars = runtime.fields[velocity_index]
        .values
        .as_deref()
        .ok_or_else(|| {
            invalid_input(
                "runtime field 'U' initial payload was already consumed or not loaded".to_string(),
            )
        })?;
    let cells = velocity_scalars.len() / 3;
    let mut vectors = Vec::new();
    vectors
        .try_reserve_exact(cells)
        .map_err(|_| invalid_input("runtime vector field 'U' allocation failed".to_string()))?;
    for chunk in velocity_scalars.chunks_exact(3) {
        vectors.push(Point3 {
            x: chunk[0],
            y: chunk[1],
            z: chunk[2],
        });
    }

    // Pressure is a Vec<f64> already and moves without changing its pointer.
    // If this impossible post-preflight branch is reached, U remains intact.
    let pressure = runtime.fields[pressure_index]
        .values
        .take()
        .ok_or_else(|| {
            invalid_input(
                "runtime field 'p' initial payload was already consumed or not loaded".to_string(),
            )
        })?;
    runtime.fields[velocity_index].values = None;
    Ok((vectors, pressure))
}

fn runtime_field_index(
    runtime: &SolverRuntimeData,
    name: &str,
    kind: SolverStateFieldKind,
    components: usize,
) -> Result<usize> {
    runtime
        .fields
        .iter()
        .position(|field| {
            field.region.is_none()
                && field.name == name
                && field.kind == kind
                && field.components == components
        })
        .ok_or_else(|| {
            invalid_input(format!(
                "runtime field '{}' with {} components was not materialized",
                name, components
            ))
        })
}

fn validate_runtime_field(
    runtime: &SolverRuntimeData,
    index: usize,
    name: &str,
    components: usize,
) -> Result<()> {
    let buffer = &runtime.fields[index];
    let expected = runtime
        .mesh
        .cells
        .checked_mul(components)
        .ok_or_else(|| invalid_input(format!("runtime field '{name}' size overflowed")))?;
    let values = buffer.values.as_ref().ok_or_else(|| {
        invalid_input(format!(
            "runtime field '{name}' initial payload was already consumed or not loaded"
        ))
    })?;
    let expected_bytes = expected
        .checked_mul(std::mem::size_of::<f64>())
        .ok_or_else(|| invalid_input(format!("runtime field '{name}' byte size overflowed")))?;
    if buffer.scalar_slots != expected
        || buffer.bytes_f64 != expected_bytes
        || values.len() != expected
    {
        return Err(invalid_input(format!(
            "runtime field '{name}' has descriptor slots={}, bytes={} and {} scalars, expected slots={expected}, bytes={expected_bytes}",
            buffer.scalar_slots,
            buffer.bytes_f64,
            values.len(),
        )));
    }
    Ok(())
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

fn face_vector_value_with_addressing(
    mesh: &SolverRuntimeMeshData,
    face_addressing: &CompactSimpleFaceAddressing,
    velocity: &[Point3],
    boundary: &[VectorFaceTreatment],
    face_index: usize,
) -> Point3 {
    let owner = face_addressing.owner(face_index);
    if let Some(neighbour) = face_addressing.neighbour(face_index) {
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

#[cfg(test)]
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

fn upwind_face_vector_value_with_addressing(
    face_addressing: &CompactSimpleFaceAddressing,
    velocity: &[Point3],
    boundary: &[VectorFaceTreatment],
    face_index: usize,
    flux: f64,
) -> Point3 {
    let owner = face_addressing.owner(face_index);
    if let Some(neighbour) = face_addressing.neighbour(face_index) {
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

#[allow(clippy::too_many_arguments)]
fn convection_face_vector_value_with_addressing(
    mesh: &SolverRuntimeMeshData,
    face_addressing: &CompactSimpleFaceAddressing,
    velocity: &[Point3],
    boundary: &[VectorFaceTreatment],
    face_index: usize,
    flux: f64,
    scheme: LaminarSimpleConvectionScheme,
    gradients: Option<&[Vec<Point3>; 3]>,
) -> Point3 {
    let upwind = upwind_face_vector_value_with_addressing(
        face_addressing,
        velocity,
        boundary,
        face_index,
        flux,
    );
    if !scheme.uses_linear_upwind() {
        return upwind;
    }
    let Some(gradients) = gradients else {
        return upwind;
    };

    let owner = face_addressing.owner(face_index);
    let upwind_cell = if let Some(neighbour) = face_addressing.neighbour(face_index) {
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

#[cfg(test)]
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

fn face_scalar_value_with_addressing(
    mesh: &SolverRuntimeMeshData,
    face_addressing: &CompactSimpleFaceAddressing,
    values: &[f64],
    boundary: &[ScalarFaceTreatment],
    face_index: usize,
) -> Result<f64> {
    let owner = face_addressing.owner(face_index);
    if let Some(neighbour) = face_addressing.neighbour(face_index) {
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

#[cfg(test)]
fn limit_scalar_gradient(
    mesh: &SolverRuntimeMeshData,
    values: &[f64],
    boundary: &[ScalarFaceTreatment],
    gradient: Vec<Point3>,
    coefficient: f64,
) -> Result<Vec<Point3>> {
    let face_addressing = CompactSimpleFaceAddressing::from_mesh(mesh)?;
    limit_scalar_gradient_with_addressing(
        mesh,
        &face_addressing,
        values,
        boundary,
        gradient,
        coefficient,
    )
}

#[cfg(test)]
fn cell_face_adjacency(mesh: &SolverRuntimeMeshData) -> Result<Vec<Vec<usize>>> {
    if mesh.owner.len() != mesh.faces || mesh.neighbour.len() != mesh.faces {
        return Err(invalid_input(format!(
            "cell-to-face adjacency requires {} owner and neighbour entries, got {} and {}",
            mesh.faces,
            mesh.owner.len(),
            mesh.neighbour.len()
        )));
    }

    let mut adjacency = vec![Vec::new(); mesh.cells];
    for face_index in 0..mesh.faces {
        let owner = mesh.owner[face_index];
        let owner_faces = adjacency.get_mut(owner).ok_or_else(|| {
            invalid_input(format!(
                "face {face_index} owner cell {owner} is outside cell range 0..{}",
                mesh.cells
            ))
        })?;
        owner_faces.push(face_index);

        if let Some(neighbour) = mesh.neighbour[face_index] {
            if neighbour == owner {
                return Err(invalid_input(format!(
                    "face {face_index} has identical owner and neighbour cell {owner}"
                )));
            }
            let neighbour_faces = adjacency.get_mut(neighbour).ok_or_else(|| {
                invalid_input(format!(
                    "face {face_index} neighbour cell {neighbour} is outside cell range 0..{}",
                    mesh.cells
                ))
            })?;
            neighbour_faces.push(face_index);
        }
    }
    Ok(adjacency)
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

#[cfg(test)]
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
    if options.non_orthogonal_correctors > MAX_NON_ORTHOGONAL_CORRECTORS {
        return Err(invalid_input(format!(
            "laminar SIMPLE nNonOrthogonalCorrectors must not exceed {MAX_NON_ORTHOGONAL_CORRECTORS}, got {}",
            options.non_orthogonal_correctors
        )));
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
    validate_linear_relative_tolerance("momentum", options.momentum_linear_relative_tolerance)?;
    validate_linear_relative_tolerance("pressure", options.pressure_linear_relative_tolerance)?;
    if options.momentum_linear_solver == LaminarSimpleLinearSolver::Gamg {
        return Err(invalid_input(
            "laminar SIMPLE GAMG is supported for the symmetric pressure equation only".to_string(),
        ));
    }
    if options.profile_gamg && options.pressure_linear_solver != LaminarSimpleLinearSolver::Gamg {
        return Err(invalid_input(
            "laminar SIMPLE GAMG profiling requires the GAMG pressure solver".to_string(),
        ));
    }
    match (
        options.pressure_linear_solver,
        options.pressure_gamg_options,
    ) {
        (LaminarSimpleLinearSolver::Gamg, None) => {
            return Err(invalid_input(
                "laminar SIMPLE pressure GAMG requires GAMG options".to_string(),
            ));
        }
        (LaminarSimpleLinearSolver::Gamg, Some(gamg_options)) => {
            if options.pressure_linear_relative_tolerance.to_bits()
                != gamg_options.relative_tolerance.to_bits()
            {
                return Err(invalid_input(format!(
                    "laminar SIMPLE pressure GAMG relTol sources must match exactly, got pressure={} and GAMG={}",
                    options.pressure_linear_relative_tolerance, gamg_options.relative_tolerance
                )));
            }
        }
        (_, None) => {}
        (_, Some(_)) => {
            return Err(invalid_input(
                "laminar SIMPLE pressure GAMG options require the GAMG pressure solver".to_string(),
            ));
        }
    }
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
    if matches!(
        preconditioner,
        LaminarSimplePreconditioner::Dic | LaminarSimplePreconditioner::Fdic
    ) && solver != LaminarSimpleLinearSolver::Pcg
    {
        return Err(invalid_input(format!(
            "laminar SIMPLE {name} {preconditioner} preconditioner requires pcg on a symmetric SPD matrix, got {solver}"
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

fn validate_linear_relative_tolerance(name: &str, value: f64) -> Result<()> {
    if !value.is_finite() || value < 0.0 {
        return Err(invalid_input(format!(
            "laminar SIMPLE {name} linear relTol must be non-negative and finite, got {value}"
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

fn cached_variable_face_diffusion_coefficient(
    pressure_geometry: &PressureGeometryCache,
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
    let geometry = pressure_geometry.face(face_index);
    let area = geometry.area_magnitude;
    if !area.is_finite() || area <= f64::EPSILON {
        return Err(invalid_input(format!(
            "face {face_index} has non-positive area magnitude {area}"
        )));
    }
    let projected_distance = geometry.projected_distance;
    if !projected_distance.is_finite() || projected_distance <= f64::EPSILON {
        return Err(invalid_input(format!(
            "face {face_index} has non-positive projected diffusion distance {projected_distance}"
        )));
    }
    Ok(diffusivity * area / projected_distance)
}

fn cached_fixed_gradient_pressure_flux(
    pressure_geometry: &PressureGeometryCache,
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
    let area = pressure_geometry.face(face_index).area_magnitude;
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

fn add_csr_value(values: &mut [f64], slot: usize, value: f64) {
    values[slot] += value;
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
    use crate::linear::{
        CgPreconditioner, CsrMatrix, GamgAgglomerator, GamgOptions, GamgWorkspace,
        IterativeSolveTermination, NormalizedL1PcgSolveControls,
        PreconditionedConjugateGradientOptions, PreconditionedConjugateGradientWorkspace,
        PreparedPcgInitialTiming, PreparedPcgResidual,
    };
    use crate::runtime::{
        SolverRuntimeData, SolverRuntimeFieldBuffer, SolverRuntimeMeshData, SolverRuntimePatchRange,
    };
    use crate::solver_state::SolverStateFieldKind;

    use super::{
        LaminarSimpleConvectionScheme, LaminarSimpleGradientScheme, LaminarSimpleLinearSolver,
        LaminarSimpleMeshCache, LaminarSimpleMomentumExecution, LaminarSimpleOptions,
        LaminarSimplePreconditioner, LaminarSimpleSchemes, LaminarSimpleStopReason,
        LinearSolveStopReason, MAX_NON_ORTHOGONAL_CORRECTORS, MomentumComponentExecutor,
        MomentumComponentTestAction, MomentumComponentWorkerFailure, MomentumCsrPattern,
        ScalarComponentSystem, ScalarFaceTreatment, ScalarGradientGeometry, ScalarSolveControls,
        ScalarSolveReport, ScalarSolveWorkspace, VectorFaceTreatment, adjust_phi_hby_a,
        apply_pressure_reference, assemble_momentum_component_system, assemble_momentum_equation,
        assemble_variable_scalar_component_system, assemble_variable_scalar_component_system_into,
        cell_face_adjacency, compute_face_flux, compute_phi_hby_a,
        consistent_reciprocal_momentum_diagonal, constrained_pressure_treatments,
        face_diffusion_coefficient, hby_a_from_predicted_velocity, limit_scalar_gradient,
        net_cell_flux, non_orthogonal_pressure_flux_correction, normalized_residual_norm,
        pressure_correction_flux, reciprocal_momentum_diagonal, relax_scalar_component_equation,
        scalar_component_boundary, scalar_gradient, solve_laminar_simple,
        solve_momentum_components_serial, split_components, store_momentum_component_worker_result,
        subtract_face_fluxes, upwind_face_vector_value, vector_convection_divergence,
        vector_face_treatments, velocity_from_hby_a, zero,
    };
    use crate::{MeshError, Point3};

    #[test]
    fn dic_fdic_and_ic0_remain_distinct_and_require_their_supported_solver() {
        for (preconditioner, mapped, label) in [
            (
                LaminarSimplePreconditioner::Dic,
                CgPreconditioner::Dic,
                "DIC",
            ),
            (
                LaminarSimplePreconditioner::Fdic,
                CgPreconditioner::Fdic,
                "FDIC",
            ),
            (
                LaminarSimplePreconditioner::IncompleteCholesky,
                CgPreconditioner::IncompleteCholesky,
                "incompleteCholesky",
            ),
        ] {
            assert_eq!(super::map_cg_preconditioner(preconditioner), mapped);
            assert_eq!(preconditioner.to_string(), label);
            super::validate_solver_preconditioner(
                "pressure",
                LaminarSimpleLinearSolver::Pcg,
                preconditioner,
            )
            .expect("symmetric pressure PCG preconditioner should be accepted");
        }

        for preconditioner in [
            LaminarSimplePreconditioner::Dic,
            LaminarSimplePreconditioner::Fdic,
        ] {
            let error = super::validate_solver_preconditioner(
                "pressure",
                LaminarSimpleLinearSolver::BiCgStab,
                preconditioner,
            )
            .expect_err("face-LDU DIC/FDIC must remain restricted to PCG");
            assert!(
                error
                    .to_string()
                    .contains("requires pcg on a symmetric SPD matrix")
            );
        }
    }

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
    fn cell_face_adjacency_preserves_global_order_and_membership() {
        let adjacency = cell_face_adjacency(&three_cell_line_mesh()).expect("adjacency");
        assert_eq!(adjacency, vec![vec![0, 2], vec![0, 1], vec![1, 3]]);

        let mut boundary_only = two_cell_runtime().mesh;
        boundary_only.cells = 1;
        boundary_only.faces = 3;
        boundary_only.owner = vec![0, 0, 0];
        boundary_only.neighbour = vec![None; 3];
        assert_eq!(
            cell_face_adjacency(&boundary_only).expect("boundary-only adjacency"),
            vec![vec![0, 1, 2]]
        );
    }

    #[test]
    fn cell_face_adjacency_rejects_malformed_cell_indices() {
        let mut invalid_owner = two_cell_runtime().mesh;
        invalid_owner.owner[2] = invalid_owner.cells;
        assert!(cell_face_adjacency(&invalid_owner).is_err());

        let mut invalid_neighbour = two_cell_runtime().mesh;
        invalid_neighbour.neighbour[0] = Some(invalid_neighbour.cells);
        assert!(cell_face_adjacency(&invalid_neighbour).is_err());
    }

    #[test]
    fn adjacency_limits_each_cell_to_its_incident_faces() {
        let adjacency = cell_face_adjacency(&three_cell_line_mesh()).expect("adjacency");
        let visits: usize = adjacency.iter().map(Vec::len).sum();
        assert_eq!(visits, 6);
        assert!(visits < 3 * 4);
    }

    #[test]
    fn optimized_cell_limiter_matches_naive_reference() {
        let mesh = three_cell_line_mesh();
        let boundary = vec![ScalarFaceTreatment::ZeroGradient; mesh.faces];
        let values = [0.0, 1.0, 0.25];
        let raw = vec![
            point(2.0, -0.5, 0.25),
            point(-1.5, 0.75, -0.25),
            point(1.25, -1.0, 0.5),
        ];

        for coefficient in [0.0, 1.0, 0.4] {
            let optimized =
                limit_scalar_gradient(&mesh, &values, &boundary, raw.clone(), coefficient)
                    .expect("optimized limiter");
            let reference =
                naive_limit_scalar_gradient(&mesh, &values, &boundary, raw.clone(), coefficient)
                    .expect("naive limiter");
            for (actual, expected) in optimized.iter().zip(reference) {
                assert_eq!(actual.x.to_bits(), expected.x.to_bits());
                assert_eq!(actual.y.to_bits(), expected.y.to_bits());
                assert_eq!(actual.z.to_bits(), expected.z.to_bits());
            }
        }
    }

    fn naive_limit_scalar_gradient(
        mesh: &SolverRuntimeMeshData,
        values: &[f64],
        boundary: &[ScalarFaceTreatment],
        mut gradient: Vec<Point3>,
        coefficient: f64,
    ) -> crate::Result<Vec<Point3>> {
        if coefficient == 0.0 {
            return Ok(gradient);
        }
        let mut minima = values.to_vec();
        let mut maxima = values.to_vec();
        for face in 0..mesh.faces {
            let owner = mesh.owner[face];
            if let Some(neighbour) = mesh.neighbour[face] {
                minima[owner] = minima[owner].min(values[neighbour]);
                maxima[owner] = maxima[owner].max(values[neighbour]);
                minima[neighbour] = minima[neighbour].min(values[owner]);
                maxima[neighbour] = maxima[neighbour].max(values[owner]);
            } else {
                let value = super::face_scalar_value(mesh, values, boundary, face)?;
                minima[owner] = minima[owner].min(value);
                maxima[owner] = maxima[owner].max(value);
            }
        }
        for cell in 0..mesh.cells {
            let maximum_delta = super::checked_subtraction(
                maxima[cell],
                values[cell],
                format!("cellLimited cell {cell} maximum extrema delta"),
            )?;
            let minimum_delta = super::checked_subtraction(
                minima[cell],
                values[cell],
                format!("cellLimited cell {cell} minimum extrema delta"),
            )?;
            let span = super::checked_subtraction(
                maxima[cell],
                minima[cell],
                format!("cellLimited cell {cell} extrema span"),
            )?;
            let widening = if coefficient == 1.0 {
                0.0
            } else {
                let numerator = super::checked_product(
                    span,
                    1.0 - coefficient,
                    format!("cellLimited cell {cell} widening numerator"),
                )?;
                super::require_finite(
                    numerator / coefficient,
                    format!("cellLimited cell {cell} widening term"),
                )?
            };
            let widened_maximum = super::require_finite(
                maximum_delta + widening,
                format!("cellLimited cell {cell} widened maximum delta"),
            )?;
            let widened_minimum = super::require_finite(
                minimum_delta - widening,
                format!("cellLimited cell {cell} widened minimum delta"),
            )?;
            let mut limiter: f64 = 1.0;
            for face in 0..mesh.faces {
                if mesh.owner[face] != cell && mesh.neighbour[face] != Some(cell) {
                    continue;
                }
                let delta = super::checked_delta(
                    mesh.face_centres[face],
                    mesh.cell_centres[cell],
                    format!("cellLimited cell {cell} face {face} centre delta"),
                )?;
                let extrapolation = super::checked_dot(
                    gradient[cell],
                    delta,
                    format!("cellLimited cell {cell} face {face} extrapolation"),
                )?;
                let ratio = if extrapolation > widened_maximum && extrapolation > 0.0 {
                    widened_maximum / extrapolation
                } else if extrapolation < widened_minimum && extrapolation < 0.0 {
                    widened_minimum / extrapolation
                } else {
                    1.0
                };
                limiter = limiter.min(
                    super::require_finite(
                        ratio,
                        format!("cellLimited cell {cell} face {face} limiter ratio"),
                    )?
                    .clamp(0.0, 1.0),
                );
                super::require_finite(limiter, format!("cellLimited cell {cell} final limiter"))?;
            }
            super::checked_scale(
                &mut gradient[cell],
                limiter,
                format!("cellLimited cell {cell} limited gradient"),
            )?;
        }
        Ok(gradient)
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
        let momentum_csr_pattern =
            MomentumCsrPattern::from_mesh(&runtime.mesh).expect("momentum CSR pattern");
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
            &momentum_csr_pattern,
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
        let momentum_csr_pattern =
            MomentumCsrPattern::from_mesh(&runtime.mesh).expect("momentum CSR pattern");
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
            &momentum_csr_pattern,
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
            &momentum_csr_pattern,
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
    fn bounded_linear_upwind_matches_unbounded_for_divergence_free_flux() {
        let runtime = two_cell_runtime();
        let momentum_csr_pattern =
            MomentumCsrPattern::from_mesh(&runtime.mesh).expect("momentum CSR pattern");
        let boundary = vec![ScalarFaceTreatment::ZeroGradient; runtime.mesh.faces];
        let flux = vec![1.0, -1.0, 1.0];
        let source = vec![0.0, 0.0];
        let old_values = vec![5.0, 3.0];
        let old_gradient = vec![point(4.0, 0.0, 0.0); 2];
        let unbounded = assemble_momentum_component_system(
            &runtime.mesh,
            &momentum_csr_pattern,
            1.0,
            2.0,
            &flux,
            &source,
            &boundary,
            &old_values,
            Some(&old_gradient),
            LaminarSimpleConvectionScheme::GaussLinearUpwind,
        )
        .expect("unbounded system");
        let bounded = assemble_momentum_component_system(
            &runtime.mesh,
            &momentum_csr_pattern,
            1.0,
            2.0,
            &flux,
            &source,
            &boundary,
            &old_values,
            Some(&old_gradient),
            LaminarSimpleConvectionScheme::BoundedGaussLinearUpwind(
                LaminarSimpleGradientScheme::GaussLinear,
            ),
        )
        .expect("bounded system");

        assert_eq!(bounded.matrix.row_offsets(), unbounded.matrix.row_offsets());
        assert_eq!(bounded.matrix.col_indices(), unbounded.matrix.col_indices());
        assert_eq!(bounded.matrix.values(), unbounded.matrix.values());
        assert_eq!(bounded.rhs, unbounded.rhs);

        let velocity = vec![point(5.0, 1.0, -2.0), point(3.0, 4.0, 6.0)];
        let vector_boundary = vec![VectorFaceTreatment::ZeroGradient; runtime.mesh.faces];
        let unbounded_divergence = vector_convection_divergence(
            &runtime.mesh,
            &velocity,
            &vector_boundary,
            &flux,
            LaminarSimpleConvectionScheme::GaussLinearUpwind,
            LaminarSimpleGradientScheme::GaussLinear,
        )
        .expect("unbounded divergence");
        let bounded_divergence = vector_convection_divergence(
            &runtime.mesh,
            &velocity,
            &vector_boundary,
            &flux,
            LaminarSimpleConvectionScheme::BoundedGaussLinearUpwind(
                LaminarSimpleGradientScheme::GaussLinear,
            ),
            LaminarSimpleGradientScheme::GaussLinear,
        )
        .expect("bounded divergence");
        for cell in 0..runtime.mesh.cells {
            assert_eq!(bounded_divergence[cell].x, unbounded_divergence[cell].x);
            assert_eq!(bounded_divergence[cell].y, unbounded_divergence[cell].y);
            assert_eq!(bounded_divergence[cell].z, unbounded_divergence[cell].z);
        }
    }

    #[test]
    fn bounded_correction_has_analytic_nonconservative_deltas_and_neutralizes_constants() {
        let runtime = two_cell_runtime();
        let momentum_csr_pattern =
            MomentumCsrPattern::from_mesh(&runtime.mesh).expect("momentum CSR pattern");
        let scalar_boundary = vec![ScalarFaceTreatment::ZeroGradient; runtime.mesh.faces];
        let vector_boundary = vec![VectorFaceTreatment::ZeroGradient; runtime.mesh.faces];
        let flux = vec![1.0, -2.0, 3.0];
        let source = vec![0.0, 0.0];
        let old_values = vec![5.0, 3.0];
        let old_gradient = vec![zero(); 2];
        let unbounded = assemble_momentum_component_system(
            &runtime.mesh,
            &momentum_csr_pattern,
            0.0,
            1.0,
            &flux,
            &source,
            &scalar_boundary,
            &old_values,
            Some(&old_gradient),
            LaminarSimpleConvectionScheme::GaussLinearUpwind,
        )
        .expect("unbounded system");
        let bounded_scheme = LaminarSimpleConvectionScheme::BoundedGaussLinearUpwind(
            LaminarSimpleGradientScheme::GaussLinear,
        );
        let bounded = assemble_momentum_component_system(
            &runtime.mesh,
            &momentum_csr_pattern,
            0.0,
            1.0,
            &flux,
            &source,
            &scalar_boundary,
            &old_values,
            Some(&old_gradient),
            bounded_scheme,
        )
        .expect("bounded system");
        let unbounded_diagonal = unbounded.matrix.diagonal().expect("unbounded diagonal");
        let bounded_diagonal = bounded.matrix.diagonal().expect("bounded diagonal");
        assert_eq!(bounded_diagonal[0] - unbounded_diagonal[0], 1.0);
        assert_eq!(bounded_diagonal[1] - unbounded_diagonal[1], -2.0);
        assert_eq!(bounded.rhs, unbounded.rhs);

        let velocity = vec![point(5.0, 1.0, -2.0), point(3.0, 4.0, 6.0)];
        let unbounded_divergence = vector_convection_divergence(
            &runtime.mesh,
            &velocity,
            &vector_boundary,
            &flux,
            LaminarSimpleConvectionScheme::GaussLinearUpwind,
            LaminarSimpleGradientScheme::GaussLinear,
        )
        .expect("unbounded divergence");
        let bounded_divergence = vector_convection_divergence(
            &runtime.mesh,
            &velocity,
            &vector_boundary,
            &flux,
            bounded_scheme,
            LaminarSimpleGradientScheme::GaussLinear,
        )
        .expect("bounded divergence");
        let net = [-1.0, 2.0];
        for cell in 0..2 {
            assert_eq!(
                bounded_divergence[cell].x - unbounded_divergence[cell].x,
                -net[cell] * velocity[cell].x
            );
            assert_eq!(
                bounded_divergence[cell].y - unbounded_divergence[cell].y,
                -net[cell] * velocity[cell].y
            );
            assert_eq!(
                bounded_divergence[cell].z - unbounded_divergence[cell].z,
                -net[cell] * velocity[cell].z
            );
        }

        let constant = vec![point(7.0, -3.0, 2.0); 2];
        let neutralized = vector_convection_divergence(
            &runtime.mesh,
            &constant,
            &vector_boundary,
            &flux,
            bounded_scheme,
            LaminarSimpleGradientScheme::GaussLinear,
        )
        .expect("constant bounded divergence");
        for value in neutralized {
            assert_eq!(value.x, 0.0);
            assert_eq!(value.y, 0.0);
            assert_eq!(value.z, 0.0);
        }
    }

    #[test]
    fn equation_relaxation_preserves_original_diagonal_for_rau() {
        let runtime = two_cell_runtime();
        let momentum_csr_pattern =
            MomentumCsrPattern::from_mesh(&runtime.mesh).expect("momentum CSR pattern");
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
            &momentum_csr_pattern,
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
        let mesh_cache =
            LaminarSimpleMeshCache::from_mesh(&runtime.mesh, LaminarSimpleSchemes::default())
                .expect("SIMPLE mesh cache");
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
            &mesh_cache,
            &velocity,
            &vector_boundary,
            &flux,
            &grad_p,
            &old_components,
            &options,
        )
        .expect("momentum equation");

        assert_eq!(equation.components.len(), 3);
        assert!(std::ptr::eq(
            equation.components[0].matrix.row_offsets().as_ptr(),
            equation.components[1].matrix.row_offsets().as_ptr(),
        ));
        assert!(std::ptr::eq(
            equation.components[0].matrix.col_indices().as_ptr(),
            equation.components[2].matrix.col_indices().as_ptr(),
        ));
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
    fn pressure_assembly_reuses_csr_topology_and_value_storage() {
        let runtime = two_cell_runtime();
        let pattern = MomentumCsrPattern::from_mesh(&runtime.mesh).expect("pressure CSR pattern");
        let mut workspace = super::ScalarComponentSystem {
            matrix: CsrMatrix::from_pattern(&pattern.sparsity, vec![0.0; pattern.sparsity.nnz()])
                .expect("pressure matrix"),
            rhs: vec![0.0; runtime.mesh.cells],
        };
        let row_offsets = workspace.matrix.row_offsets().as_ptr();
        let col_indices = workspace.matrix.col_indices().as_ptr();
        let values = workspace.matrix.values().as_ptr();
        let boundary = vec![ScalarFaceTreatment::ZeroGradient; runtime.mesh.faces];

        assemble_variable_scalar_component_system_into(
            &runtime.mesh,
            &pattern,
            &[1.0, 2.0],
            &[1.0, 2.0],
            &boundary,
            &mut workspace,
        )
        .expect("first pressure assembly");
        assemble_variable_scalar_component_system_into(
            &runtime.mesh,
            &pattern,
            &[2.0, 4.0],
            &[3.0, 4.0],
            &boundary,
            &mut workspace,
        )
        .expect("second pressure assembly");
        let expected = assemble_variable_scalar_component_system(
            &runtime.mesh,
            &[2.0, 4.0],
            &[3.0, 4.0],
            &boundary,
        )
        .expect("independent pressure assembly");

        assert_eq!(workspace.matrix.row_offsets().as_ptr(), row_offsets);
        assert_eq!(workspace.matrix.col_indices().as_ptr(), col_indices);
        assert_eq!(workspace.matrix.values().as_ptr(), values);
        assert_eq!(workspace.matrix.values(), expected.matrix.values());
        assert_eq!(workspace.rhs, expected.rhs);
    }

    #[test]
    fn pressure_geometry_cache_matches_orthogonal_and_skewed_bit_oracles() {
        for skewed in [false, true] {
            let mut runtime = two_cell_runtime();
            if skewed {
                runtime.mesh.face_centres[0] = point(0.5, 0.125, -0.0625);
                runtime.mesh.face_area_vectors[0] = point(1.25, 0.5, -0.25);
                runtime.mesh.cell_centres[1] = point(0.75, 0.25, -0.125);
            }
            let pressure_geometry = super::PressureGeometryCache::from_mesh(&runtime.mesh);
            let cell_diffusivity = [1.25, 2.75];

            for face_index in 0..runtime.mesh.faces {
                let owner = runtime.mesh.owner[face_index];
                let to = runtime.mesh.neighbour[face_index]
                    .map(|neighbour| runtime.mesh.cell_centres[neighbour])
                    .unwrap_or(runtime.mesh.face_centres[face_index]);
                let from = runtime.mesh.cell_centres[owner];
                let area_vector = runtime.mesh.face_area_vectors[face_index];
                let area = super::magnitude(area_vector);
                let delta = point(to.x - from.x, to.y - from.y, to.z - from.z);
                let projected_distance = (super::dot(delta, area_vector) / area).abs();
                let centre_distance = super::distance(from, to);
                let direction = point(
                    (to.x - from.x) / centre_distance,
                    (to.y - from.y) / centre_distance,
                    (to.z - from.z) / centre_distance,
                );
                let orthogonal_area = super::dot(area_vector, direction);
                let non_orthogonal_area = point(
                    area_vector.x - orthogonal_area * direction.x,
                    area_vector.y - orthogonal_area * direction.y,
                    area_vector.z - orthogonal_area * direction.z,
                );
                let cached = pressure_geometry.face(face_index);

                assert_eq!(cached.area_magnitude.to_bits(), area.to_bits());
                assert_eq!(
                    cached.projected_distance.to_bits(),
                    projected_distance.to_bits()
                );
                assert_eq!(cached.centre_distance.to_bits(), centre_distance.to_bits());
                assert_eq!(
                    cached.non_orthogonal_area_vector.x.to_bits(),
                    non_orthogonal_area.x.to_bits()
                );
                assert_eq!(
                    cached.non_orthogonal_area_vector.y.to_bits(),
                    non_orthogonal_area.y.to_bits()
                );
                assert_eq!(
                    cached.non_orthogonal_area_vector.z.to_bits(),
                    non_orthogonal_area.z.to_bits()
                );

                let diffusivity = runtime.mesh.neighbour[face_index]
                    .map(|neighbour| 0.5 * (cell_diffusivity[owner] + cell_diffusivity[neighbour]))
                    .unwrap_or(cell_diffusivity[owner]);
                let oracle =
                    face_diffusion_coefficient(diffusivity, area_vector, from, to, face_index)
                        .expect("uncached diffusion oracle");
                let actual = super::cached_variable_face_diffusion_coefficient(
                    &pressure_geometry,
                    &cell_diffusivity,
                    owner,
                    runtime.mesh.neighbour[face_index],
                    face_index,
                )
                .expect("cached pressure diffusion coefficient");
                assert_eq!(actual.to_bits(), oracle.to_bits());
            }
        }
    }

    #[test]
    fn pressure_geometry_cache_keeps_boundary_validation_lazy_and_ordered() {
        let mut runtime = two_cell_runtime();
        let invalid_boundary_face = runtime.mesh.internal_faces;
        runtime.mesh.face_area_vectors[invalid_boundary_face] = point(0.0, 1.0, 0.0);
        let pressure_geometry = super::PressureGeometryCache::from_mesh(&runtime.mesh);
        let pattern = MomentumCsrPattern::from_mesh(&runtime.mesh).expect("pressure CSR pattern");
        let mut workspace = super::ScalarComponentSystem {
            matrix: CsrMatrix::from_pattern(&pattern.sparsity, vec![0.0; pattern.sparsity.nnz()])
                .expect("pressure matrix"),
            rhs: vec![0.0; runtime.mesh.cells],
        };
        let source = [0.0, 0.0];
        let diffusivity = [1.0, 1.0];

        for unused in [
            ScalarFaceTreatment::ZeroGradient,
            ScalarFaceTreatment::Constraint,
        ] {
            let mut boundary = vec![ScalarFaceTreatment::ZeroGradient; runtime.mesh.faces];
            boundary[invalid_boundary_face] = unused;
            super::assemble_variable_scalar_component_system_into_with_pressure_geometry(
                &runtime.mesh,
                &pattern,
                &pressure_geometry,
                &diffusivity,
                &source,
                &boundary,
                &mut workspace,
            )
            .expect("unused invalid boundary geometry remains lazy");
        }

        let mut inlet_outlet = vec![ScalarFaceTreatment::ZeroGradient; runtime.mesh.faces];
        inlet_outlet[invalid_boundary_face] = ScalarFaceTreatment::InletOutlet(3.0);
        let mut outflow = vec![0.0; runtime.mesh.faces];
        outflow[invalid_boundary_face] = 1.0;
        let resolved = super::resolve_pressure_inlet_outlet(&inlet_outlet, &outflow)
            .expect("outflow inletOutlet resolution");
        super::assemble_variable_scalar_component_system_into_with_pressure_geometry(
            &runtime.mesh,
            &pattern,
            &pressure_geometry,
            &diffusivity,
            &source,
            &resolved,
            &mut workspace,
        )
        .expect("inletOutlet outflow does not consume invalid geometry");

        let mut fixed_value = vec![ScalarFaceTreatment::ZeroGradient; runtime.mesh.faces];
        fixed_value[invalid_boundary_face] = ScalarFaceTreatment::FixedValue(0.0);
        let error = super::assemble_variable_scalar_component_system_into_with_pressure_geometry(
            &runtime.mesh,
            &pattern,
            &pressure_geometry,
            &diffusivity,
            &source,
            &fixed_value,
            &mut workspace,
        )
        .expect_err("active pressure diffusion must validate projected distance");
        assert_eq!(
            error.to_string(),
            format!("face {invalid_boundary_face} has non-positive projected diffusion distance 0")
        );

        runtime.mesh.face_area_vectors[invalid_boundary_face] = zero();
        let zero_area_geometry = super::PressureGeometryCache::from_mesh(&runtime.mesh);
        let gradient_error = super::cached_fixed_gradient_pressure_flux(
            &zero_area_geometry,
            &diffusivity,
            runtime.mesh.owner[invalid_boundary_face],
            invalid_boundary_face,
            f64::NAN,
        )
        .expect_err("gradient validation must precede cached area validation");
        assert_eq!(
            gradient_error.to_string(),
            format!(
                "fixed-gradient pressure face {invalid_boundary_face} has non-finite gradient NaN"
            )
        );
    }

    #[test]
    fn pressure_geometry_cache_preserves_adjust_constrain_and_nonorth_error_order() {
        let mut runtime = two_cell_runtime();
        let boundary_face = runtime.mesh.internal_faces;
        runtime.mesh.face_area_vectors[boundary_face] = zero();
        let pressure_geometry = super::PressureGeometryCache::from_mesh(&runtime.mesh);
        let mut phi = vec![0.0; runtime.mesh.faces];
        phi[boundary_face] = 1.0;
        let mut velocity_boundary = vec![VectorFaceTreatment::Constraint; runtime.mesh.faces];
        velocity_boundary[boundary_face] = VectorFaceTreatment::ZeroGradient;
        let mut pressure_boundary = vec![ScalarFaceTreatment::Constraint; runtime.mesh.faces];
        pressure_boundary[boundary_face] = ScalarFaceTreatment::FixedValue(0.0);
        let adjust = super::adjust_phi_hby_a_with_pressure_geometry(
            &runtime.mesh,
            &pressure_geometry,
            &velocity_boundary,
            &pressure_boundary,
            &mut phi,
        )
        .expect("adjustPhi silently excludes invalid adjustable area");
        assert_eq!(adjust.adjusted_faces, 0);
        assert_eq!(phi[boundary_face].to_bits(), 1.0f64.to_bits());

        pressure_boundary[boundary_face] = ScalarFaceTreatment::ZeroGradient;
        velocity_boundary[boundary_face] = VectorFaceTreatment::FixedValue(zero());
        let mut invalid_r_au = vec![1.0; runtime.mesh.cells];
        invalid_r_au[runtime.mesh.owner[boundary_face]] = 0.0;
        let constrain_error = super::constrained_pressure_treatments_with_geometry(
            &runtime.mesh,
            &pressure_geometry,
            &pressure_boundary,
            &velocity_boundary,
            &[zero(); 2],
            &phi,
            &invalid_r_au,
        )
        .expect_err("constrainPressure area failure must precede rAU failure");
        assert_eq!(
            constrain_error.to_string(),
            format!("fixedFluxPressure face {boundary_face} has non-positive area magnitude 0")
        );

        let mut valid_runtime = two_cell_runtime();
        let scalar_gradient_geometry =
            ScalarGradientGeometry::from_mesh(&valid_runtime.mesh).expect("gradient geometry");
        valid_runtime.mesh.cell_centres[1] = valid_runtime.mesh.cell_centres[0];
        let coincident_pressure_geometry =
            super::PressureGeometryCache::from_mesh(&valid_runtime.mesh);
        let boundary = vec![ScalarFaceTreatment::ZeroGradient; valid_runtime.mesh.faces];
        let scalar_error = super::non_orthogonal_pressure_flux_correction_with_geometry(
            &valid_runtime.mesh,
            &coincident_pressure_geometry,
            &scalar_gradient_geometry,
            &[f64::NAN, 0.0],
            &[1.0, 1.0],
            &boundary,
            LaminarSimpleGradientScheme::GaussLinear,
        )
        .expect_err("scalar-gradient validation precedes cached face distance");
        assert_eq!(
            scalar_error.to_string(),
            "scalar gradient cell 0 value must be finite, got NaN"
        );

        let distance_error = super::non_orthogonal_pressure_flux_correction_with_geometry(
            &valid_runtime.mesh,
            &coincident_pressure_geometry,
            &scalar_gradient_geometry,
            &[1.0, 0.0],
            &[1.0, 1.0],
            &boundary,
            LaminarSimpleGradientScheme::GaussLinear,
        )
        .expect_err("finite scalar gradient reaches cached centre-distance validation");
        assert_eq!(
            distance_error.to_string(),
            "face 0 has non-positive non-orthogonal correction distance 0"
        );
    }

    #[test]
    fn pressure_geometry_cache_reuses_storage_for_ten_coefficients_and_refreshes_per_mesh() {
        let mut runtime = two_cell_runtime();
        let pattern = MomentumCsrPattern::from_mesh(&runtime.mesh).expect("pressure CSR pattern");
        let pressure_geometry = super::PressureGeometryCache::from_mesh(&runtime.mesh);
        let geometry_pointer = pressure_geometry.faces.as_ptr();
        let geometry_capacity = pressure_geometry.faces.capacity();
        let mut workspace = super::ScalarComponentSystem {
            matrix: CsrMatrix::from_pattern(&pattern.sparsity, vec![0.0; pattern.sparsity.nnz()])
                .expect("pressure matrix"),
            rhs: vec![0.0; runtime.mesh.cells],
        };
        let values_pointer = workspace.matrix.values().as_ptr();
        let boundary = vec![
            ScalarFaceTreatment::ZeroGradient,
            ScalarFaceTreatment::FixedValue(0.0),
            ScalarFaceTreatment::FixedGradient(0.25),
        ];

        for lifecycle in 0..10 {
            let scale = lifecycle as f64 + 1.0;
            super::assemble_variable_scalar_component_system_into_with_pressure_geometry(
                &runtime.mesh,
                &pattern,
                &pressure_geometry,
                &[scale, scale + 0.5],
                &[0.25 * scale, -0.125 * scale],
                &boundary,
                &mut workspace,
            )
            .expect("cached pressure assembly lifecycle");
            assert_eq!(pressure_geometry.faces.as_ptr(), geometry_pointer);
            assert_eq!(pressure_geometry.faces.capacity(), geometry_capacity);
            assert_eq!(workspace.matrix.values().as_ptr(), values_pointer);
        }

        let original = pressure_geometry.face(0);
        runtime.mesh.face_area_vectors[0] = point(2.0, 0.5, 0.0);
        let refreshed = super::PressureGeometryCache::from_mesh(&runtime.mesh);
        assert_eq!(
            pressure_geometry.face(0).area_magnitude.to_bits(),
            original.area_magnitude.to_bits()
        );
        assert_eq!(
            refreshed.face(0).area_magnitude.to_bits(),
            super::magnitude(runtime.mesh.face_area_vectors[0]).to_bits()
        );
        assert_ne!(
            refreshed.face(0).area_magnitude.to_bits(),
            original.area_magnitude.to_bits()
        );
    }

    #[test]
    fn public_pressure_geometry_path_runs_zero_one_and_two_correctors() {
        let fields = two_cell_fields();
        let mut reports = Vec::new();
        for correctors in 0..=2 {
            let mut runtime = two_cell_runtime();
            runtime.mesh.face_centres[0] = point(0.5, 0.05, 0.0);
            runtime.mesh.face_area_vectors[0] = point(1.0, 0.25, 0.0);
            runtime.mesh.cell_centres[1] = point(0.75, 0.1, 0.0);
            let mut options = minimal_laminar_options();
            options.max_simple_iterations = 1;
            options.non_orthogonal_correctors = correctors;

            let report = solve_laminar_simple(&mut runtime, &fields, &options)
                .expect("public skewed SIMPLE solve with cached pressure geometry");
            assert_eq!(report.simple_iterations, 1);
            assert_eq!(report.history[0].pressure_linear_solves, correctors + 1);
            assert!(
                report
                    .final_pressure
                    .iter()
                    .chain(report.final_phi.iter())
                    .all(|value| value.is_finite())
            );
            assert!(report.final_velocity.iter().all(|value| {
                value.x.is_finite() && value.y.is_finite() && value.z.is_finite()
            }));
            reports.push(report);
        }
        assert_eq!(reports.len(), 3);
    }

    #[test]
    fn compact_face_cache_matches_runtime_and_legacy_cell_face_oracles() {
        let runtime = two_cell_runtime();
        let cache =
            LaminarSimpleMeshCache::from_mesh(&runtime.mesh, LaminarSimpleSchemes::default())
                .expect("SIMPLE mesh cache");
        let addressing = &cache.face_addressing;
        let legacy = cell_face_adjacency(&runtime.mesh).expect("legacy adjacency oracle");

        assert_eq!(addressing.faces(), runtime.mesh.faces);
        for face_index in 0..runtime.mesh.faces {
            assert_eq!(addressing.owner(face_index), runtime.mesh.owner[face_index]);
            assert_eq!(
                addressing.neighbour(face_index),
                runtime.mesh.neighbour[face_index]
            );
        }
        assert_eq!(
            addressing.internal_faces(),
            runtime
                .mesh
                .neighbour
                .iter()
                .enumerate()
                .filter_map(|(face, neighbour)| neighbour.map(|_| face))
                .collect::<Vec<_>>()
        );
        assert_eq!(
            addressing.boundary_faces(),
            runtime
                .mesh
                .neighbour
                .iter()
                .enumerate()
                .filter_map(|(face, neighbour)| neighbour.is_none().then_some(face))
                .collect::<Vec<_>>()
        );
        assert!(!addressing.limiter_adjacency_initialized());
        let compact = addressing
            .limiter_cell_adjacency()
            .expect("lazy compact cell adjacency");
        for (cell, expected) in legacy.iter().enumerate() {
            assert_eq!(compact.cell_faces(cell), expected);
        }
        assert!(addressing.limiter_adjacency_initialized());
    }

    #[test]
    fn compact_face_cache_keeps_self_neighbour_failure_limiter_lazy() {
        let mut runtime = two_cell_runtime();
        runtime.mesh.neighbour[0] = Some(runtime.mesh.owner[0]);
        let addressing =
            super::CompactSimpleFaceAddressing::from_mesh(&runtime.mesh).expect("addressing");
        let values = [1.0, 2.0];
        let boundary = vec![ScalarFaceTreatment::ZeroGradient; runtime.mesh.faces];
        let gradient = vec![point(0.5, 0.0, 0.0); runtime.mesh.cells];

        assert!(!addressing.limiter_adjacency_initialized());
        let unchanged = super::limit_scalar_gradient_with_addressing(
            &runtime.mesh,
            &addressing,
            &values,
            &boundary,
            gradient.clone(),
            0.0,
        )
        .expect("zero limiter coefficient must not consume topology validation");
        for (actual, expected) in unchanged.iter().zip(&gradient) {
            assert_eq!(actual.x.to_bits(), expected.x.to_bits());
            assert_eq!(actual.y.to_bits(), expected.y.to_bits());
            assert_eq!(actual.z.to_bits(), expected.z.to_bits());
        }
        assert!(!addressing.limiter_adjacency_initialized());

        let error = super::limit_scalar_gradient_with_addressing(
            &runtime.mesh,
            &addressing,
            &values,
            &boundary,
            gradient.clone(),
            1.0,
        )
        .expect_err("nonzero limiter coefficient consumes topology validation");
        assert_eq!(
            error.to_string(),
            format!(
                "face 0 has identical owner and neighbour cell {}",
                runtime.mesh.owner[0]
            )
        );
        assert!(!addressing.limiter_adjacency_initialized());
        let retry = super::limit_scalar_gradient_with_addressing(
            &runtime.mesh,
            &addressing,
            &values,
            &boundary,
            gradient,
            1.0,
        )
        .expect_err("failed lazy initialization must be retryable");
        assert_eq!(retry.to_string(), error.to_string());
        assert!(!addressing.limiter_adjacency_initialized());
    }

    #[test]
    fn compact_face_cache_reuses_all_storage_for_ten_limiter_lifecycles() {
        let runtime = two_cell_runtime();
        let cache =
            LaminarSimpleMeshCache::from_mesh(&runtime.mesh, LaminarSimpleSchemes::default())
                .expect("SIMPLE mesh cache");
        let storage_identity = cache.face_addressing.storage_identity();
        let values = [1.0, 2.0];
        let boundary = vec![ScalarFaceTreatment::ZeroGradient; runtime.mesh.faces];

        let gauss = super::scalar_gradient_with_geometry_and_addressing(
            &runtime.mesh,
            &cache.scalar_gradient,
            &cache.face_addressing,
            &values,
            &boundary,
            LaminarSimpleGradientScheme::GaussLinear,
        )
        .expect("Gauss gradient must not initialize limiter adjacency");
        assert!(gauss.iter().all(|value| value.x.is_finite()));
        assert!(!cache.face_addressing.limiter_adjacency_initialized());
        let zero = super::scalar_gradient_with_geometry_and_addressing(
            &runtime.mesh,
            &cache.scalar_gradient,
            &cache.face_addressing,
            &values,
            &boundary,
            LaminarSimpleGradientScheme::CellLimitedGaussLinear(0.0),
        )
        .expect("zero limiter must not initialize limiter adjacency");
        assert!(zero.iter().all(|value| value.x.is_finite()));
        assert!(!cache.face_addressing.limiter_adjacency_initialized());

        let mut adjacency_identity = None;
        for lifecycle in 0..10 {
            let coefficient = (lifecycle + 1) as f64 / 10.0;
            let actual = super::scalar_gradient_with_geometry_and_addressing(
                &runtime.mesh,
                &cache.scalar_gradient,
                &cache.face_addressing,
                &values,
                &boundary,
                LaminarSimpleGradientScheme::CellLimitedGaussLinear(coefficient),
            )
            .expect("cached cell-limited gradient");
            let oracle = super::scalar_gradient_with_geometry(
                &runtime.mesh,
                &cache.scalar_gradient,
                &values,
                &boundary,
                LaminarSimpleGradientScheme::CellLimitedGaussLinear(coefficient),
            )
            .expect("independent compact-addressing oracle");

            for (actual, expected) in actual.iter().zip(&oracle) {
                assert_eq!(actual.x.to_bits(), expected.x.to_bits());
                assert_eq!(actual.y.to_bits(), expected.y.to_bits());
                assert_eq!(actual.z.to_bits(), expected.z.to_bits());
            }
            assert_eq!(cache.face_addressing.storage_identity(), storage_identity);
            let current_adjacency_identity = cache
                .face_addressing
                .limiter_cell_adjacency()
                .expect("initialized limiter adjacency")
                .storage_identity();
            if let Some(expected) = adjacency_identity {
                assert_eq!(current_adjacency_identity, expected);
            } else {
                adjacency_identity = Some(current_adjacency_identity);
            }
        }
        assert!(cache.face_addressing.limiter_adjacency_initialized());
    }

    #[test]
    fn public_compact_face_path_runs_zero_one_and_two_correctors() {
        let fields = two_cell_fields();
        for correctors in 0..=2 {
            let mut runtime = two_cell_runtime();
            let mut options = minimal_laminar_options();
            options.max_simple_iterations = 1;
            options.non_orthogonal_correctors = correctors;
            options.schemes.grad_p = LaminarSimpleGradientScheme::CellLimitedGaussLinear(1.0);
            options.schemes.grad_u = LaminarSimpleGradientScheme::CellLimitedGaussLinear(1.0);

            let report = solve_laminar_simple(&mut runtime, &fields, &options)
                .expect("public SIMPLE solve with compact face addressing");
            assert_eq!(report.simple_iterations, 1);
            assert_eq!(report.history[0].pressure_linear_solves, correctors + 1);
            assert!(
                report
                    .final_pressure
                    .iter()
                    .chain(report.final_phi.iter())
                    .all(|value| value.is_finite())
            );
        }
    }

    #[test]
    fn least_squares_cache_is_selection_driven_and_gauss_dispatch_stays_bit_identical() {
        let runtime = two_cell_runtime();
        let default_cache =
            LaminarSimpleMeshCache::from_mesh(&runtime.mesh, LaminarSimpleSchemes::default())
                .expect("default mesh cache");
        assert!(default_cache.weighted_least_squares.is_none());

        let values = [2.0, 10.0];
        let boundary = vec![ScalarFaceTreatment::ZeroGradient; runtime.mesh.faces];
        let direct = super::scalar_gradient_with_geometry_and_addressing(
            &runtime.mesh,
            &default_cache.scalar_gradient,
            &default_cache.face_addressing,
            &values,
            &boundary,
            LaminarSimpleGradientScheme::GaussLinear,
        )
        .expect("direct Gauss gradient");
        let dispatched = super::scalar_gradient_with_scheme_and_addressing(
            &runtime.mesh,
            &default_cache.scalar_gradient,
            None,
            &default_cache.face_addressing,
            &values,
            &boundary,
            LaminarSimpleGradientScheme::GaussLinear,
        )
        .expect("scheme-dispatched Gauss gradient");
        for (actual, expected) in dispatched.iter().zip(&direct) {
            assert_eq!(actual.x.to_bits(), expected.x.to_bits());
            assert_eq!(actual.y.to_bits(), expected.y.to_bits());
            assert_eq!(actual.z.to_bits(), expected.z.to_bits());
        }

        let extruded = two_cell_runtime_with_empty_sides();
        let mut schemes = LaminarSimpleSchemes {
            grad_p: LaminarSimpleGradientScheme::LeastSquares,
            ..LaminarSimpleSchemes::default()
        };
        let weighted_cache = LaminarSimpleMeshCache::from_mesh(&extruded.mesh, schemes)
            .expect("selected leastSquares cache");
        assert!(weighted_cache.weighted_least_squares.is_some());

        schemes.grad_p = LaminarSimpleGradientScheme::GaussLinear;
        schemes.grad_u = LaminarSimpleGradientScheme::LeastSquares;
        schemes.div_phi_u = LaminarSimpleConvectionScheme::GaussUpwind;
        let unused_velocity_cache = LaminarSimpleMeshCache::from_mesh(&extruded.mesh, schemes)
            .expect("upwind does not consume grad(U)");
        assert!(unused_velocity_cache.weighted_least_squares.is_none());

        schemes.div_phi_u = LaminarSimpleConvectionScheme::GaussLinearUpwind;
        let selected_velocity_cache = LaminarSimpleMeshCache::from_mesh(&extruded.mesh, schemes)
            .expect("linearUpwind consumes grad(U)");
        assert!(selected_velocity_cache.weighted_least_squares.is_some());
    }

    #[test]
    fn public_initial_gradient_probe_is_non_consuming_and_matches_production_dispatch() {
        let mut runtime = two_cell_runtime_with_empty_sides();
        let fields = two_cell_fields_with_empty_sides();
        let schemes = LaminarSimpleSchemes {
            grad_p: LaminarSimpleGradientScheme::LeastSquares,
            grad_u: LaminarSimpleGradientScheme::LeastSquares,
            div_phi_u: LaminarSimpleConvectionScheme::GaussLinearUpwind,
            ..LaminarSimpleSchemes::default()
        };
        let before = format!("{runtime:?}");
        let fields_before = format!("{fields:?}");
        let value_pointers = runtime
            .fields
            .iter()
            .map(|field| field.values.as_ref().map(Vec::as_ptr))
            .collect::<Vec<_>>();
        let value_capacities = runtime
            .fields
            .iter()
            .map(|field| field.values.as_ref().map(Vec::capacity))
            .collect::<Vec<_>>();

        let probe = super::reconstruct_laminar_initial_gradients(&runtime, &fields, schemes)
            .expect("public leastSquares initial-gradient probe");
        let repeated = super::reconstruct_laminar_initial_gradients(&runtime, &fields, schemes)
            .expect("repeated public leastSquares initial-gradient probe");

        assert_eq!(format!("{runtime:?}"), before);
        assert_eq!(format!("{fields:?}"), fields_before);
        assert_eq!(
            runtime
                .fields
                .iter()
                .map(|field| field.values.as_ref().map(Vec::as_ptr))
                .collect::<Vec<_>>(),
            value_pointers
        );
        assert_eq!(
            runtime
                .fields
                .iter()
                .map(|field| field.values.as_ref().map(Vec::capacity))
                .collect::<Vec<_>>(),
            value_capacities
        );

        let velocity_field = super::find_field(&fields, "U", "volVectorField").expect("U field");
        let pressure_field = super::find_field(&fields, "p", "volScalarField").expect("p field");
        let velocity_boundary = super::vector_face_treatments(&runtime.mesh, velocity_field)
            .expect("velocity boundary");
        let pressure_boundary = super::scalar_face_treatments(&runtime.mesh, pressure_field)
            .expect("pressure boundary");
        let cache = LaminarSimpleMeshCache::from_mesh(&runtime.mesh, schemes)
            .expect("production mesh cache");
        let (velocity, pressure) =
            super::snapshot_runtime_initial_fields(&runtime).expect("initial field snapshot");
        let flux = super::compute_face_flux_with_addressing(
            &runtime.mesh,
            &cache.face_addressing,
            &velocity,
            &velocity_boundary,
        )
        .expect("initial face flux");
        let resolved_pressure_boundary =
            super::resolve_pressure_inlet_outlet(&pressure_boundary, &flux)
                .expect("resolved pressure boundary");
        let expected_pressure = super::scalar_gradient_with_scheme_and_addressing(
            &runtime.mesh,
            &cache.scalar_gradient,
            cache.weighted_least_squares.as_ref(),
            &cache.face_addressing,
            pressure,
            &resolved_pressure_boundary,
            schemes.grad_p,
        )
        .expect("production pressure-gradient dispatch");
        let expected_velocity = super::vector_component_gradients_with_scheme_and_addressing(
            &runtime.mesh,
            &cache.scalar_gradient,
            cache.weighted_least_squares.as_ref(),
            &cache.face_addressing,
            &velocity,
            &velocity_boundary,
            &flux,
            schemes.grad_u,
        )
        .expect("production velocity-gradient dispatch");

        assert_point_fields_bit_equal(&probe.pressure_gradient, &expected_pressure);
        assert_point_fields_bit_equal(&probe.pressure_gradient, &repeated.pressure_gradient);
        let actual_velocity = probe
            .velocity_component_gradients
            .as_ref()
            .expect("linearUpwind must request grad(U)");
        let repeated_velocity = repeated
            .velocity_component_gradients
            .as_ref()
            .expect("repeated linearUpwind must request grad(U)");
        for component in 0..3 {
            assert_point_fields_bit_equal(
                &actual_velocity[component],
                &expected_velocity[component],
            );
            assert_point_fields_bit_equal(
                &actual_velocity[component],
                &repeated_velocity[component],
            );
        }

        let mut options = minimal_laminar_options();
        options.max_simple_iterations = 1;
        options.schemes = schemes;
        let report = solve_laminar_simple(&mut runtime, &fields, &options)
            .expect("probe must leave one-shot fields available to the solver");
        assert_eq!(report.simple_iterations, 1);
    }

    #[test]
    fn public_initial_gradient_probe_preserves_gauss_bits_and_upwind_dispatch() {
        let runtime = two_cell_runtime();
        let fields = two_cell_fields();
        let gauss_upwind = LaminarSimpleSchemes::default();
        let upwind_probe =
            super::reconstruct_laminar_initial_gradients(&runtime, &fields, gauss_upwind)
                .expect("Gauss/upwind initial-gradient probe");
        assert!(
            upwind_probe.velocity_component_gradients.is_none(),
            "Gauss upwind must not evaluate grad(U)"
        );

        let schemes = LaminarSimpleSchemes {
            div_phi_u: LaminarSimpleConvectionScheme::GaussLinearUpwind,
            ..gauss_upwind
        };
        let probe = super::reconstruct_laminar_initial_gradients(&runtime, &fields, schemes)
            .expect("Gauss/linearUpwind initial-gradient probe");
        let cache = LaminarSimpleMeshCache::from_mesh(&runtime.mesh, schemes)
            .expect("production Gauss mesh cache");
        let velocity_field = super::find_field(&fields, "U", "volVectorField").expect("U field");
        let pressure_field = super::find_field(&fields, "p", "volScalarField").expect("p field");
        let velocity_boundary = super::vector_face_treatments(&runtime.mesh, velocity_field)
            .expect("velocity boundary");
        let pressure_boundary = super::scalar_face_treatments(&runtime.mesh, pressure_field)
            .expect("pressure boundary");
        let (velocity, pressure) =
            super::snapshot_runtime_initial_fields(&runtime).expect("initial field snapshot");
        let flux = super::compute_face_flux_with_addressing(
            &runtime.mesh,
            &cache.face_addressing,
            &velocity,
            &velocity_boundary,
        )
        .expect("initial face flux");
        let resolved_pressure_boundary =
            super::resolve_pressure_inlet_outlet(&pressure_boundary, &flux)
                .expect("resolved pressure boundary");
        let expected_pressure = super::scalar_gradient_with_scheme_and_addressing(
            &runtime.mesh,
            &cache.scalar_gradient,
            None,
            &cache.face_addressing,
            pressure,
            &resolved_pressure_boundary,
            schemes.grad_p,
        )
        .expect("Gauss pressure-gradient dispatch");
        let expected_velocity = super::vector_component_gradients_with_scheme_and_addressing(
            &runtime.mesh,
            &cache.scalar_gradient,
            None,
            &cache.face_addressing,
            &velocity,
            &velocity_boundary,
            &flux,
            schemes.grad_u,
        )
        .expect("Gauss velocity-gradient dispatch");

        assert_point_fields_bit_equal(&probe.pressure_gradient, &expected_pressure);
        let actual_velocity = probe
            .velocity_component_gradients
            .as_ref()
            .expect("linearUpwind must request grad(U)");
        for component in 0..3 {
            assert_point_fields_bit_equal(
                &actual_velocity[component],
                &expected_velocity[component],
            );
        }
    }

    #[test]
    fn public_field_gradient_reconstruction_runs_after_consumed_upwind_solve() {
        let mut runtime = two_cell_runtime_with_empty_sides();
        let fields = two_cell_fields_with_empty_sides();
        let mut options = minimal_laminar_options();
        options.max_simple_iterations = 1;
        options.schemes.grad_p = LaminarSimpleGradientScheme::GaussLinear;
        options.schemes.grad_u = LaminarSimpleGradientScheme::LeastSquares;
        options.schemes.div_phi_u = LaminarSimpleConvectionScheme::GaussUpwind;

        let solver_cache = LaminarSimpleMeshCache::from_mesh(&runtime.mesh, options.schemes)
            .expect("upwind solver cache");
        assert!(
            solver_cache.weighted_least_squares.is_none(),
            "upwind SIMPLE must not build an unused grad(U) WLS cache"
        );
        let report =
            solve_laminar_simple(&mut runtime, &fields, &options).expect("one-step upwind solve");
        assert!(
            runtime
                .fields
                .iter()
                .filter(|field| field.name == "U" || field.name == "p")
                .all(|field| field.values.is_none()),
            "SIMPLE must have consumed the one-shot U/p payloads"
        );

        let runtime_before = format!("{runtime:?}");
        let fields_before = format!("{fields:?}");
        let velocity_pointer = report.final_velocity.as_ptr();
        let velocity_capacity = report.final_velocity.capacity();
        let pressure_pointer = report.final_pressure.as_ptr();
        let pressure_capacity = report.final_pressure.capacity();
        let gradients = super::reconstruct_laminar_gradients_from_fields(
            &runtime,
            &fields,
            &report.final_velocity,
            &report.final_pressure,
            options.schemes,
        )
        .expect("post-solve upwind leastSquares gradients");
        let repeated = super::reconstruct_laminar_gradients_from_fields(
            &runtime,
            &fields,
            &report.final_velocity,
            &report.final_pressure,
            options.schemes,
        )
        .expect("repeated post-solve upwind leastSquares gradients");
        let mut linear_upwind_schemes = options.schemes;
        linear_upwind_schemes.div_phi_u = LaminarSimpleConvectionScheme::GaussLinearUpwind;
        let production_oracle = super::reconstruct_laminar_gradients_from_fields(
            &runtime,
            &fields,
            &report.final_velocity,
            &report.final_pressure,
            linear_upwind_schemes,
        )
        .expect("linearUpwind production-dispatch oracle");

        assert_eq!(format!("{runtime:?}"), runtime_before);
        assert_eq!(format!("{fields:?}"), fields_before);
        assert_eq!(report.final_velocity.as_ptr(), velocity_pointer);
        assert_eq!(report.final_velocity.capacity(), velocity_capacity);
        assert_eq!(report.final_pressure.as_ptr(), pressure_pointer);
        assert_eq!(report.final_pressure.capacity(), pressure_capacity);
        assert_point_fields_bit_equal(&gradients.pressure_gradient, &repeated.pressure_gradient);
        assert!(
            gradients
                .velocity_component_gradients
                .iter()
                .flatten()
                .any(|value| value.x != 0.0 || value.y != 0.0 || value.z != 0.0),
            "the final nonuniform velocity field must produce a nonzero grad(U)"
        );
        for component in 0..3 {
            assert_point_fields_bit_equal(
                &gradients.velocity_component_gradients[component],
                &repeated.velocity_component_gradients[component],
            );
            assert_point_fields_bit_equal(
                &gradients.velocity_component_gradients[component],
                &production_oracle.velocity_component_gradients[component],
            );
            assert_eq!(
                gradients.velocity_component_gradients[component].len(),
                runtime.mesh.cells
            );
        }
    }

    #[test]
    fn public_field_gradient_reconstruction_matches_initial_production_dispatch_bits() {
        let runtime = two_cell_runtime_with_empty_sides();
        let fields = two_cell_fields_with_empty_sides();
        let schemes = LaminarSimpleSchemes {
            grad_p: LaminarSimpleGradientScheme::LeastSquares,
            grad_u: LaminarSimpleGradientScheme::LeastSquares,
            div_phi_u: LaminarSimpleConvectionScheme::GaussLinearUpwind,
            ..LaminarSimpleSchemes::default()
        };
        let (velocity, pressure) =
            super::snapshot_runtime_initial_fields(&runtime).expect("initial field snapshot");
        let initial = super::reconstruct_laminar_initial_gradients(&runtime, &fields, schemes)
            .expect("initial production-dispatch gradients");
        let explicit = super::reconstruct_laminar_gradients_from_fields(
            &runtime, &fields, &velocity, pressure, schemes,
        )
        .expect("explicit production-dispatch gradients");

        assert_point_fields_bit_equal(&explicit.pressure_gradient, &initial.pressure_gradient);
        let initial_velocity = initial
            .velocity_component_gradients
            .expect("linearUpwind initial grad(U)");
        for (explicit_component, initial_component) in explicit
            .velocity_component_gradients
            .iter()
            .zip(&initial_velocity)
        {
            assert_point_fields_bit_equal(explicit_component, initial_component);
        }
    }

    #[test]
    fn public_field_gradient_reconstruction_rejects_bad_fields_without_mutation() {
        let runtime = two_cell_runtime_with_empty_sides();
        let fields = two_cell_fields_with_empty_sides();
        let (velocity, pressure) =
            super::snapshot_runtime_initial_fields(&runtime).expect("initial field snapshot");
        let schemes = LaminarSimpleSchemes::default();
        let runtime_before = format!("{runtime:?}");
        let fields_before = format!("{fields:?}");

        let short_error = super::reconstruct_laminar_gradients_from_fields(
            &runtime,
            &fields,
            &velocity[..1],
            pressure,
            schemes,
        )
        .expect_err("short velocity field must fail");
        assert!(short_error.to_string().contains("expected 2 cell values"));

        let mut non_finite_velocity = velocity;
        non_finite_velocity[0].x = f64::NAN;
        let finite_error = super::reconstruct_laminar_gradients_from_fields(
            &runtime,
            &fields,
            &non_finite_velocity,
            pressure,
            schemes,
        )
        .expect_err("non-finite velocity field must fail");
        assert!(finite_error.to_string().contains("finite"));
        assert!(matches!(
            super::flow_try_vec::<u8>(usize::MAX),
            Err(MeshError::OutOfMemory)
        ));
        assert_eq!(format!("{runtime:?}"), runtime_before);
        assert_eq!(format!("{fields:?}"), fields_before);
    }

    #[test]
    fn public_initial_gradient_probe_uses_bounded_embedded_gradient_scheme() {
        let runtime = two_cell_runtime_with_empty_sides();
        let fields = two_cell_fields_with_empty_sides();
        let schemes = LaminarSimpleSchemes {
            grad_u: LaminarSimpleGradientScheme::GaussLinear,
            div_phi_u: LaminarSimpleConvectionScheme::BoundedGaussLinearUpwind(
                LaminarSimpleGradientScheme::LeastSquares,
            ),
            ..LaminarSimpleSchemes::default()
        };

        let probe = super::reconstruct_laminar_initial_gradients(&runtime, &fields, schemes)
            .expect("bounded embedded leastSquares initial-gradient probe");
        let cache = LaminarSimpleMeshCache::from_mesh(&runtime.mesh, schemes)
            .expect("bounded embedded leastSquares production cache");
        assert!(cache.weighted_least_squares.is_some());
        let velocity_field = super::find_field(&fields, "U", "volVectorField").expect("U field");
        let velocity_boundary = super::vector_face_treatments(&runtime.mesh, velocity_field)
            .expect("velocity boundary");
        let (velocity, _) =
            super::snapshot_runtime_initial_fields(&runtime).expect("initial field snapshot");
        let flux = super::compute_face_flux_with_addressing(
            &runtime.mesh,
            &cache.face_addressing,
            &velocity,
            &velocity_boundary,
        )
        .expect("initial face flux");
        let expected = super::vector_component_gradients_with_scheme_and_addressing(
            &runtime.mesh,
            &cache.scalar_gradient,
            cache.weighted_least_squares.as_ref(),
            &cache.face_addressing,
            &velocity,
            &velocity_boundary,
            &flux,
            LaminarSimpleGradientScheme::LeastSquares,
        )
        .expect("embedded leastSquares production dispatch");
        let actual = probe
            .velocity_component_gradients
            .as_ref()
            .expect("bounded linearUpwind must request grad(U)");
        for component in 0..3 {
            assert_point_fields_bit_equal(&actual[component], &expected[component]);
        }
    }

    #[test]
    fn public_initial_gradient_probe_failure_preserves_runtime_and_fields() {
        let runtime = two_cell_runtime();
        let fields = two_cell_fields();
        let schemes = LaminarSimpleSchemes {
            grad_p: LaminarSimpleGradientScheme::LeastSquares,
            ..LaminarSimpleSchemes::default()
        };
        let runtime_before = format!("{runtime:?}");
        let fields_before = format!("{fields:?}");
        let value_pointers = runtime
            .fields
            .iter()
            .map(|field| field.values.as_ref().map(Vec::as_ptr))
            .collect::<Vec<_>>();
        let value_capacities = runtime
            .fields
            .iter()
            .map(|field| field.values.as_ref().map(Vec::capacity))
            .collect::<Vec<_>>();

        let error = super::reconstruct_laminar_initial_gradients(&runtime, &fields, schemes)
            .expect_err("rank-deficient 3D leastSquares probe must fail");

        assert!(error.to_string().contains("weighted least-squares"));
        assert_eq!(format!("{runtime:?}"), runtime_before);
        assert_eq!(format!("{fields:?}"), fields_before);
        assert_eq!(
            runtime
                .fields
                .iter()
                .map(|field| field.values.as_ref().map(Vec::as_ptr))
                .collect::<Vec<_>>(),
            value_pointers
        );
        assert_eq!(
            runtime
                .fields
                .iter()
                .map(|field| field.values.as_ref().map(Vec::capacity))
                .collect::<Vec<_>>(),
            value_capacities
        );
    }

    #[test]
    fn public_least_squares_paths_run_zero_one_and_two_correctors_without_fallback() {
        let fields = two_cell_fields_with_empty_sides();
        for correctors in 0..=2 {
            let mut runtime = two_cell_runtime_with_empty_sides();
            let mut options = minimal_laminar_options();
            options.max_simple_iterations = 1;
            options.non_orthogonal_correctors = correctors;
            options.schemes.grad_p = LaminarSimpleGradientScheme::LeastSquares;
            options.schemes.grad_u = LaminarSimpleGradientScheme::LeastSquares;
            options.schemes.div_phi_u = LaminarSimpleConvectionScheme::GaussLinearUpwind;

            let report = solve_laminar_simple(&mut runtime, &fields, &options)
                .expect("public leastSquares SIMPLE solve");
            assert_eq!(report.simple_iterations, 1);
            assert_eq!(report.history[0].pressure_linear_solves, correctors + 1);
            assert!(
                report
                    .final_pressure
                    .iter()
                    .chain(report.final_phi.iter())
                    .all(|value| value.is_finite())
            );
            assert!(report.final_velocity.iter().all(|value| {
                value.x.is_finite() && value.y.is_finite() && value.z.is_finite()
            }));
        }

        let runtime = two_cell_runtime_with_empty_sides();
        let boundary = vec![
            ScalarFaceTreatment::ZeroGradient,
            ScalarFaceTreatment::FixedValue(0.0),
            ScalarFaceTreatment::FixedValue(2.0),
            ScalarFaceTreatment::Constraint,
            ScalarFaceTreatment::Constraint,
            ScalarFaceTreatment::Constraint,
            ScalarFaceTreatment::Constraint,
            ScalarFaceTreatment::Constraint,
            ScalarFaceTreatment::Constraint,
            ScalarFaceTreatment::Constraint,
            ScalarFaceTreatment::Constraint,
        ];
        let gradient = scalar_gradient(
            &runtime.mesh,
            &[0.5, 1.5],
            &boundary,
            LaminarSimpleGradientScheme::LeastSquares,
        )
        .expect("public scalar leastSquares gradient");
        for value in gradient {
            assert_close(value.x, 2.0);
            assert_eq!(value.y.to_bits(), 0.0f64.to_bits());
            assert_eq!(value.z.to_bits(), 0.0f64.to_bits());
        }
    }

    #[test]
    fn public_least_squares_corrector_matrix_covers_open_closed_pressure_systems_and_geometry() {
        for closed in [false, true] {
            for skewed in [false, true] {
                for correctors in 0..=2 {
                    let mut runtime = four_cell_planar_runtime(skewed);
                    let fields = four_cell_planar_fields(closed);
                    let pressure_has_fixed_value = fields
                        .fields
                        .iter()
                        .find(|field| field.name == "p")
                        .expect("p field")
                        .boundary_patches
                        .iter()
                        .any(|patch| patch.patch_type.as_deref() == Some("fixedValue"));
                    assert_eq!(pressure_has_fixed_value, !closed);
                    let has_non_orthogonal_internal_face =
                        (0..runtime.mesh.internal_faces).any(|face| {
                            let owner = runtime.mesh.owner[face];
                            let neighbour =
                                runtime.mesh.neighbour[face].expect("internal neighbour");
                            let delta = point(
                                runtime.mesh.cell_centres[neighbour].x
                                    - runtime.mesh.cell_centres[owner].x,
                                runtime.mesh.cell_centres[neighbour].y
                                    - runtime.mesh.cell_centres[owner].y,
                                runtime.mesh.cell_centres[neighbour].z
                                    - runtime.mesh.cell_centres[owner].z,
                            );
                            let area = runtime.mesh.face_area_vectors[face];
                            let cross = point(
                                delta.y * area.z - delta.z * area.y,
                                delta.z * area.x - delta.x * area.z,
                                delta.x * area.y - delta.y * area.x,
                            );
                            cross.x != 0.0 || cross.y != 0.0 || cross.z != 0.0
                        });
                    assert_eq!(has_non_orthogonal_internal_face, skewed);
                    let has_skewed_internal_face = (0..runtime.mesh.internal_faces).any(|face| {
                        let owner = runtime.mesh.owner[face];
                        let neighbour = runtime.mesh.neighbour[face].expect("internal neighbour");
                        let owner_centre = runtime.mesh.cell_centres[owner];
                        let neighbour_centre = runtime.mesh.cell_centres[neighbour];
                        let face_centre = runtime.mesh.face_centres[face];
                        let delta = point(
                            neighbour_centre.x - owner_centre.x,
                            neighbour_centre.y - owner_centre.y,
                            neighbour_centre.z - owner_centre.z,
                        );
                        let owner_to_face = point(
                            face_centre.x - owner_centre.x,
                            face_centre.y - owner_centre.y,
                            face_centre.z - owner_centre.z,
                        );
                        let delta_squared =
                            delta.x * delta.x + delta.y * delta.y + delta.z * delta.z;
                        let projection = (owner_to_face.x * delta.x
                            + owner_to_face.y * delta.y
                            + owner_to_face.z * delta.z)
                            / delta_squared;
                        let offset = point(
                            owner_to_face.x - projection * delta.x,
                            owner_to_face.y - projection * delta.y,
                            owner_to_face.z - projection * delta.z,
                        );
                        offset.x != 0.0 || offset.y != 0.0 || offset.z != 0.0
                    });
                    assert_eq!(has_skewed_internal_face, skewed);
                    let mut options = minimal_laminar_options();
                    options.max_simple_iterations = 1;
                    options.non_orthogonal_correctors = correctors;
                    options.pressure_reference_cell = closed.then_some(0);
                    assert_eq!(options.pressure_reference_cell.is_some(), closed);
                    options.schemes.grad_p = LaminarSimpleGradientScheme::LeastSquares;
                    options.schemes.grad_u = LaminarSimpleGradientScheme::LeastSquares;
                    options.schemes.div_phi_u = LaminarSimpleConvectionScheme::GaussLinearUpwind;

                    let report = solve_laminar_simple(&mut runtime, &fields, &options)
                        .unwrap_or_else(|error| {
                            panic!(
                                "leastSquares matrix failed for closed={closed}, skewed={skewed}, correctors={correctors}: {error}"
                            )
                        });
                    assert_eq!(report.simple_iterations, 1);
                    assert_eq!(
                        report.stop_reason,
                        LaminarSimpleStopReason::ConvergenceCriteriaNotConfigured
                    );
                    let iteration = &report.history[0];
                    assert!(iteration.pressure_correction_accepted);
                    assert_eq!(iteration.pressure_linear_solves, correctors + 1);
                    assert!(iteration.momentum_linear_converged);
                    assert!(
                        iteration
                            .momentum_component_linear_converged
                            .iter()
                            .all(|converged| *converged)
                    );
                    assert!(iteration.pressure_linear_converged);
                    assert_eq!(iteration.pressure_linear_non_converged_solves, 0);
                    assert!(
                        iteration
                            .momentum_component_linear_solves
                            .iter()
                            .all(|solve| solve.converged
                                && solve.stop_reason != LinearSolveStopReason::Breakdown)
                    );
                    assert!(
                        iteration.pressure_correction_linear_solves
                            [..iteration.pressure_linear_solves]
                            .iter()
                            .all(|solve| solve.converged
                                && solve.stop_reason != LinearSolveStopReason::Breakdown)
                    );
                    assert_eq!(
                        report
                            .linear_solve_summary
                            .momentum_component_non_converged_solves,
                        0
                    );
                    assert_eq!(
                        report
                            .linear_solve_summary
                            .pressure_correction_non_converged_solves,
                        0
                    );
                    assert!(report.final_pressure.iter().all(|value| value.is_finite()));
                    assert!(report.final_velocity.iter().all(|value| {
                        value.x.is_finite() && value.y.is_finite() && value.z.is_finite()
                    }));
                }
            }
        }
    }

    #[test]
    fn least_squares_geometry_failure_precedes_runtime_field_consumption() {
        let mut runtime = two_cell_runtime();
        let fields = two_cell_fields();
        let mut options = minimal_laminar_options();
        options.schemes.grad_p = LaminarSimpleGradientScheme::LeastSquares;
        let error = solve_laminar_simple(&mut runtime, &fields, &options)
            .expect_err("rank-deficient selected WLS geometry must fail");
        assert!(error.to_string().contains("weighted least-squares"));
        assert!(
            runtime.fields.iter().all(|field| field.values.is_some()),
            "failed WLS cache construction must not consume runtime fields"
        );
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
    fn non_orthogonal_corrected_flux_matches_final_matrix_residual() {
        let mut runtime = two_cell_runtime();
        runtime.mesh.face_area_vectors[0] = point(1.0, 0.5, 0.0);
        runtime.mesh.cell_volumes = vec![0.4, 0.6];
        runtime.mesh.min_cell_volume = 0.4;
        runtime.mesh.max_cell_volume = 0.6;
        let boundary = vec![ScalarFaceTreatment::ZeroGradient; runtime.mesh.faces];
        let geometry = ScalarGradientGeometry::from_mesh(&runtime.mesh)
            .expect("skewed scalar gradient geometry");
        let r_au = vec![1.0, 2.0];
        let pressure = vec![1.25, -0.5];
        let phi_hby_a = vec![0.4, -0.1, 0.2];

        let non_orthogonal_flux = non_orthogonal_pressure_flux_correction(
            &runtime.mesh,
            &geometry,
            &pressure,
            &r_au,
            &boundary,
            LaminarSimpleGradientScheme::GaussLinear,
        )
        .expect("skewed non-orthogonal correction flux");
        assert!(
            non_orthogonal_flux
                .iter()
                .any(|value| value.abs() > 1.0e-12),
            "skewed regression mesh must exercise a non-zero explicit correction"
        );
        let final_equation_flux =
            super::add_face_fluxes(&phi_hby_a, &non_orthogonal_flux).expect("final equation flux");
        let source = super::pressure_correction_source(
            &runtime.mesh,
            &net_cell_flux(&runtime.mesh, &final_equation_flux).expect("equation net flux"),
        )
        .expect("pressure source");
        let system =
            assemble_variable_scalar_component_system(&runtime.mesh, &r_au, &source, &boundary)
                .expect("skewed pressure system");
        let matrix_balance = system
            .matrix
            .matvec(&pressure)
            .expect("pressure matrix balance");
        let pressure_flux =
            super::pressure_equation_flux(&runtime.mesh, &pressure, &r_au, &boundary)
                .expect("pressure equation flux");

        let corrected =
            super::corrected_pressure_equation_flux(&final_equation_flux, &pressure_flux)
                .expect("non-orthogonal corrected flux");
        let corrected_balance =
            net_cell_flux(&runtime.mesh, &corrected).expect("corrected flux balance");
        for ((actual, matrix), rhs) in corrected_balance
            .iter()
            .zip(&matrix_balance)
            .zip(&system.rhs)
        {
            assert_close(*actual, matrix - rhs);
        }

        let omitted_non_orthogonal = subtract_face_fluxes(&phi_hby_a, &pressure_flux)
            .expect("legacy incomplete corrected flux");
        let omitted_balance = net_cell_flux(&runtime.mesh, &omitted_non_orthogonal)
            .expect("legacy incomplete flux balance");
        assert!(
            omitted_balance
                .iter()
                .zip(&matrix_balance)
                .zip(&system.rhs)
                .any(|((actual, matrix), rhs)| (actual - (matrix - rhs)).abs() > 1.0e-12),
            "omitting the final explicit non-orthogonal flux must leave a continuity defect"
        );
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
        let scalar_gradient_geometry =
            ScalarGradientGeometry::from_mesh(&runtime.mesh).expect("scalar gradient geometry");
        let boundary = vec![ScalarFaceTreatment::ZeroGradient; runtime.mesh.faces];
        let flux = non_orthogonal_pressure_flux_correction(
            &runtime.mesh,
            &scalar_gradient_geometry,
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
        let row_offsets = system.matrix.row_offsets().as_ptr();
        let col_indices = system.matrix.col_indices().as_ptr();
        let values = system.matrix.values().as_ptr();
        let mut pcg_workspace = PreconditionedConjugateGradientWorkspace::new(
            &system.matrix,
            CgPreconditioner::IncompleteCholesky,
        )
        .expect("closed pressure PCG workspace");

        apply_pressure_reference(&mut system, &runtime.mesh, &boundary, &options)
            .expect("pressure reference");
        let solution = system.matrix.matvec(&[7.0, 7.0]).expect("matvec");
        let report = pcg_workspace
            .solve(
                &system.matrix,
                &system.rhs,
                None,
                PreconditionedConjugateGradientOptions {
                    max_iterations: 8,
                    tolerance: 1.0e-12,
                    preconditioner: CgPreconditioner::IncompleteCholesky,
                },
            )
            .expect("referenced pressure solve");

        assert_close(solution[0], system.rhs[0]);
        assert_close(solution[1], system.rhs[1]);
        assert!(report.converged);
        assert_close(report.solution[0], 7.0);
        assert_close(report.solution[1], 7.0);
        assert_eq!(system.matrix.row_offsets().as_ptr(), row_offsets);
        assert_eq!(system.matrix.col_indices().as_ptr(), col_indices);
        assert_eq!(system.matrix.values().as_ptr(), values);
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
    fn pressure_inlet_outlet_backflow_survives_constrain_pressure_as_fixed_value() {
        let runtime = two_cell_runtime();
        let mut velocity_boundary = vec![VectorFaceTreatment::ZeroGradient; runtime.mesh.faces];
        velocity_boundary[2] = VectorFaceTreatment::InletOutlet(point(-2.0, 0.0, 0.0));
        let mut pressure_boundary = vec![ScalarFaceTreatment::ZeroGradient; runtime.mesh.faces];
        pressure_boundary[2] = ScalarFaceTreatment::InletOutlet(3.5);
        let velocity = vec![point(0.0, 0.0, 0.0), point(0.0, 0.0, 0.0)];
        let phi_hby_a = vec![0.0, 0.0, -1.0];
        let r_au = vec![1.0, 1.0];
        let resolved = super::resolve_pressure_inlet_outlet(&pressure_boundary, &phi_hby_a)
            .expect("resolved pressure");

        let constrained = constrained_pressure_treatments(
            &runtime.mesh,
            &resolved,
            &velocity_boundary,
            &velocity,
            &phi_hby_a,
            &r_au,
        )
        .expect("constrained pressure");

        assert!(matches!(
            constrained[2],
            ScalarFaceTreatment::FixedValue(value) if value == 3.5
        ));
    }

    #[test]
    fn pressure_inlet_outlet_outflow_survives_constrain_pressure_as_zero_gradient() {
        let runtime = two_cell_runtime();
        let mut velocity_boundary = vec![VectorFaceTreatment::ZeroGradient; runtime.mesh.faces];
        velocity_boundary[2] = VectorFaceTreatment::InletOutlet(point(-2.0, 0.0, 0.0));
        let mut pressure_boundary = vec![ScalarFaceTreatment::ZeroGradient; runtime.mesh.faces];
        pressure_boundary[2] = ScalarFaceTreatment::InletOutlet(0.0);
        let velocity = vec![point(0.0, 0.0, 0.0), point(0.0, 0.0, 0.0)];
        let phi_hby_a = vec![0.0, 0.0, 1.0];
        let r_au = vec![1.0, 1.0];
        let resolved = super::resolve_pressure_inlet_outlet(&pressure_boundary, &phi_hby_a)
            .expect("resolved pressure");

        let constrained = constrained_pressure_treatments(
            &runtime.mesh,
            &resolved,
            &velocity_boundary,
            &velocity,
            &phi_hby_a,
            &r_au,
        )
        .expect("constrained pressure");

        assert!(matches!(constrained[2], ScalarFaceTreatment::ZeroGradient));
    }

    #[test]
    fn pressure_inlet_outlet_resolver_uses_exact_flux_sign_including_signed_zero() {
        let boundary = [
            ScalarFaceTreatment::InletOutlet(1.0),
            ScalarFaceTreatment::InletOutlet(2.0),
            ScalarFaceTreatment::InletOutlet(3.0),
            ScalarFaceTreatment::InletOutlet(4.0),
            ScalarFaceTreatment::FixedGradient(5.0),
        ];
        let resolved = super::resolve_pressure_inlet_outlet(
            &boundary,
            &[-f64::MIN_POSITIVE, -0.0, 0.0, 1.0, 6.0],
        )
        .expect("finite fluxes resolve");

        assert!(matches!(
            resolved[0],
            ScalarFaceTreatment::FixedValue(value) if value == 1.0
        ));
        assert!(matches!(resolved[1], ScalarFaceTreatment::ZeroGradient));
        assert!(matches!(resolved[2], ScalarFaceTreatment::ZeroGradient));
        assert!(matches!(resolved[3], ScalarFaceTreatment::ZeroGradient));
        assert!(matches!(
            resolved[4],
            ScalarFaceTreatment::FixedGradient(value) if value == 5.0
        ));
    }

    #[test]
    fn pressure_inlet_outlet_resolver_rejects_non_finite_decision_flux() {
        for flux in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let error = super::resolve_pressure_inlet_outlet(
                &[ScalarFaceTreatment::InletOutlet(7.0)],
                &[flux],
            )
            .expect_err("non-finite decision flux must fail");
            assert!(matches!(
                error,
                MeshError::InvalidInput(message)
                    if message.contains("non-finite decision flux")
            ));
        }
    }

    #[test]
    fn pressure_inlet_outlet_resolved_treatments_match_local_numerical_oracles() {
        let runtime = two_cell_runtime();
        let geometry = ScalarGradientGeometry::from_mesh(&runtime.mesh).expect("gradient geometry");
        let values = [1.25, -0.5];
        let diffusivity = [1.0, 2.0];
        let source = [0.25, -0.75];
        for (decision_flux, explicit) in [
            (-1.0, ScalarFaceTreatment::FixedValue(3.5)),
            (1.0, ScalarFaceTreatment::ZeroGradient),
        ] {
            let mut mixed = vec![ScalarFaceTreatment::ZeroGradient; runtime.mesh.faces];
            mixed[2] = ScalarFaceTreatment::InletOutlet(3.5);
            let mut flux = vec![0.0; runtime.mesh.faces];
            flux[2] = decision_flux;
            let resolved = super::resolve_pressure_inlet_outlet(&mixed, &flux)
                .expect("resolved mixed pressure");
            let mut oracle = vec![ScalarFaceTreatment::ZeroGradient; runtime.mesh.faces];
            oracle[2] = explicit;

            let actual_system = assemble_variable_scalar_component_system(
                &runtime.mesh,
                &diffusivity,
                &source,
                &resolved,
            )
            .expect("resolved system");
            let oracle_system = assemble_variable_scalar_component_system(
                &runtime.mesh,
                &diffusivity,
                &source,
                &oracle,
            )
            .expect("oracle system");
            assert_eq!(actual_system.matrix.values(), oracle_system.matrix.values());
            assert_eq!(actual_system.rhs, oracle_system.rhs);

            let actual_gradient = super::scalar_gradient_with_geometry(
                &runtime.mesh,
                &geometry,
                &values,
                &resolved,
                LaminarSimpleGradientScheme::GaussLinear,
            )
            .expect("resolved gradient");
            let oracle_gradient = super::scalar_gradient_with_geometry(
                &runtime.mesh,
                &geometry,
                &values,
                &oracle,
                LaminarSimpleGradientScheme::GaussLinear,
            )
            .expect("oracle gradient");
            for (actual, expected) in actual_gradient.iter().zip(&oracle_gradient) {
                assert_eq!(actual.x.to_bits(), expected.x.to_bits());
                assert_eq!(actual.y.to_bits(), expected.y.to_bits());
                assert_eq!(actual.z.to_bits(), expected.z.to_bits());
            }
            assert_eq!(
                super::pressure_equation_flux(&runtime.mesh, &values, &diffusivity, &resolved)
                    .expect("resolved pressure flux"),
                super::pressure_equation_flux(&runtime.mesh, &values, &diffusivity, &oracle)
                    .expect("oracle pressure flux")
            );
        }
    }

    #[test]
    fn pressure_reference_tracks_resolved_inlet_outlet_state() {
        let runtime = two_cell_runtime();
        let mut mixed = vec![ScalarFaceTreatment::ZeroGradient; runtime.mesh.faces];
        mixed[2] = ScalarFaceTreatment::InletOutlet(4.0);
        let options = minimal_laminar_options();
        let make_system = || {
            assemble_variable_scalar_component_system(
                &runtime.mesh,
                &[1.0, 1.0],
                &[0.0, 0.0],
                &vec![ScalarFaceTreatment::ZeroGradient; runtime.mesh.faces],
            )
            .expect("pressure system")
        };

        let open = super::resolve_pressure_inlet_outlet(&mixed, &[0.0, 0.0, -1.0])
            .expect("backflow resolution");
        let mut open_system = make_system();
        let open_values = open_system.matrix.values().to_vec();
        let open_rhs = open_system.rhs.clone();
        apply_pressure_reference(&mut open_system, &runtime.mesh, &open, &options)
            .expect("open pressure system");
        assert_eq!(open_system.matrix.values(), open_values);
        assert_eq!(open_system.rhs, open_rhs);

        let closed = super::resolve_pressure_inlet_outlet(&mixed, &[0.0, 0.0, 1.0])
            .expect("outflow resolution");
        let mut closed_system = make_system();
        let closed_values = closed_system.matrix.values().to_vec();
        apply_pressure_reference(&mut closed_system, &runtime.mesh, &closed, &options)
            .expect("closed pressure system");
        assert_ne!(closed_system.matrix.values(), closed_values);
    }

    #[test]
    fn simple_entrypoint_tracks_pressure_inlet_outlet_reversal_each_step() {
        let mut runtime = two_cell_runtime();
        let mut fields = two_cell_fields();
        fields.fields[0].boundary_patches[1].patch_type = Some("inletOutlet".to_string());
        fields.fields[0].boundary_patches[1].inlet_value =
            Some(FieldValueSummary::Uniform("(0 0 0)".to_string()));
        fields.fields[1].boundary_patches[1].patch_type = Some("inletOutlet".to_string());
        fields.fields[1].boundary_patches[1].inlet_value =
            Some(FieldValueSummary::Uniform("3.5".to_string()));
        fields.fields[1].boundary_patches[1].value = None;
        let mut options = minimal_laminar_options();
        options.max_simple_iterations = 3;
        let signs = [1.0, -1.0, 1.0];
        let mut resolved_steps = 0;
        let mut drive_reversal =
            |iteration: usize,
             phi_hby_a: &mut Vec<f64>,
             resolved: Option<&[ScalarFaceTreatment]>| {
                let sign = signs[iteration - 1];
                if let Some(resolved) = resolved {
                    if sign < 0.0 {
                        assert!(matches!(
                            resolved[2],
                            ScalarFaceTreatment::FixedValue(value) if value == 3.5
                        ));
                    } else {
                        assert!(matches!(resolved[2], ScalarFaceTreatment::ZeroGradient));
                    }
                    resolved_steps += 1;
                } else {
                    phi_hby_a[2] = sign;
                }
            };

        let report = super::solve_laminar_simple_driven(
            &mut runtime,
            &fields,
            &options,
            None,
            false,
            None,
            Some(&mut drive_reversal),
        )
        .expect("three-step pressure inletOutlet reversal");

        assert_eq!(report.history.len(), 3);
        assert!(
            report
                .history
                .iter()
                .all(|step| step.pressure_correction_accepted)
        );
        assert_eq!(resolved_steps, 3);
    }

    #[test]
    fn runs_minimal_simple_loop_on_two_cells() {
        let mut runtime = two_cell_runtime();
        let fields = two_cell_fields();
        let options = minimal_laminar_options();

        let report = solve_laminar_simple(&mut runtime, &fields, &options).expect("simple report");

        assert_eq!(report.cells, 2);
        assert!(report.simple_iterations > 0);
        assert!(report.fields.velocity.l2_norm.is_finite());
        assert!(report.fields.pressure.l2_norm.is_finite());
        assert!(report.final_continuity.l2_norm.is_finite());
        assert!(report.operator_summary.hby_a_l2_norm.is_finite());
        assert_eq!(report.final_velocity.len(), runtime.mesh.cells);
        assert_eq!(report.final_pressure.len(), runtime.mesh.cells);
        let timing_values = [
            report.timing.solver_total_seconds,
            report.timing.setup_seconds,
            report.timing.iteration_setup_seconds,
            report.timing.operator_evaluation_seconds,
            report.timing.momentum_assembly_seconds,
            report.timing.momentum_linear_solve_seconds,
            report.timing.pressure_coupling_setup_seconds,
            report.timing.pressure_assembly_seconds,
            report.timing.pressure_linear_solve_seconds,
            report.timing.field_correction_seconds,
            report.timing.finalization_seconds,
            report.timing.other_solver_work_seconds,
        ];
        assert!(
            timing_values
                .iter()
                .all(|seconds| seconds.is_finite() && *seconds >= 0.0)
        );
        let pressure_kernel_timing_values = [
            report.timing.pressure_pcg_total_seconds,
            report.timing.pressure_preconditioner_update_seconds,
            report.timing.pressure_matrix_vector_seconds,
            report.timing.pressure_preconditioner_application_seconds,
            report.timing.pressure_vector_operation_seconds,
            report.timing.pressure_pcg_other_seconds,
        ];
        assert!(
            pressure_kernel_timing_values
                .iter()
                .all(|seconds| seconds.is_finite() && *seconds >= 0.0)
        );
        assert!(
            report.timing.pressure_pcg_total_seconds
                <= report.timing.pressure_linear_solve_seconds + 1.0e-9
        );
        let phase_total: f64 = timing_values[1..].iter().sum();
        assert!(phase_total <= report.timing.solver_total_seconds + 1.0e-9);

        let second = solve_laminar_simple(&mut runtime, &fields, &options)
            .expect_err("initial fields are one-shot runtime payloads");
        assert!(matches!(
            second,
            MeshError::InvalidInput(message)
                if message
                    == "runtime field 'U' initial payload was already consumed or not loaded"
        ));
    }

    #[test]
    fn dic_and_fdic_run_through_public_simple_entrypoint_bit_identically() {
        let mut reports = Vec::new();
        for preconditioner in [
            LaminarSimplePreconditioner::Dic,
            LaminarSimplePreconditioner::Fdic,
        ] {
            let mut runtime = two_cell_runtime();
            let fields = two_cell_fields();
            let mut options = minimal_laminar_options();
            options.pressure_linear_solver = LaminarSimpleLinearSolver::Pcg;
            options.pressure_preconditioner = preconditioner;

            reports.push(
                solve_laminar_simple(&mut runtime, &fields, &options)
                    .expect("public SIMPLE pressure PCG with face-LDU preconditioner"),
            );
        }

        let dic = &reports[0];
        let fdic = &reports[1];
        assert!(dic.total_pressure_linear_iterations > 0);
        assert_eq!(
            dic.total_pressure_linear_iterations,
            fdic.total_pressure_linear_iterations
        );
        assert_eq!(
            dic.final_pressure
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            fdic.final_pressure
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
        assert_eq!(
            dic.final_velocity
                .iter()
                .flat_map(|value| [value.x.to_bits(), value.y.to_bits(), value.z.to_bits()])
                .collect::<Vec<_>>(),
            fdic.final_velocity
                .iter()
                .flat_map(|value| [value.x.to_bits(), value.y.to_bits(), value.z.to_bits()])
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn consistent_simple_runs_through_public_entrypoint_and_reports_r_at_u() {
        fn assert_face_flux_bits(
            actual: super::FaceFluxDiagnosticSummary,
            expected: super::FaceFluxDiagnosticSummary,
        ) {
            assert_eq!(actual.min.to_bits(), expected.min.to_bits());
            assert_eq!(actual.max.to_bits(), expected.max.to_bits());
            assert_eq!(actual.l2_norm.to_bits(), expected.l2_norm.to_bits());
            assert_eq!(actual.sum.to_bits(), expected.sum.to_bits());
            assert_eq!(actual.sum_abs.to_bits(), expected.sum_abs.to_bits());
            assert_eq!(
                actual.internal_sum_abs.to_bits(),
                expected.internal_sum_abs.to_bits()
            );
            assert_eq!(
                actual.boundary_sum.to_bits(),
                expected.boundary_sum.to_bits()
            );
            assert_eq!(
                actual.boundary_sum_abs.to_bits(),
                expected.boundary_sum_abs.to_bits()
            );
        }

        let fields = two_cell_fields();
        let mut options = minimal_laminar_options();
        options.max_simple_iterations = 1;
        options.momentum_linear_tolerance = 1.0e-3;
        let mut baseline_runtime = two_cell_runtime();
        let baseline = solve_laminar_simple(&mut baseline_runtime, &fields, &options)
            .expect("baseline SIMPLE report");

        let mut consistent_runtime = two_cell_runtime();
        let mut consistent_options = options;
        consistent_options.simple_consistent = true;
        let consistent =
            solve_laminar_simple(&mut consistent_runtime, &fields, &consistent_options)
                .expect("consistent SIMPLE report");

        let baseline_assembly = baseline
            .pressure_assembly
            .expect("baseline pressure assembly");
        let consistent_assembly = consistent
            .pressure_assembly
            .expect("consistent pressure assembly");

        assert_eq!(
            baseline_assembly.r_au.l2_norm.to_bits(),
            baseline_assembly.r_at_u.l2_norm.to_bits()
        );
        assert_eq!(
            baseline_assembly.r_au.min.to_bits(),
            baseline_assembly.r_at_u.min.to_bits()
        );
        assert_eq!(
            baseline_assembly.r_au.max.to_bits(),
            baseline_assembly.r_at_u.max.to_bits()
        );
        assert_ne!(
            consistent_assembly.r_au.l2_norm.to_bits(),
            consistent_assembly.r_at_u.l2_norm.to_bits()
        );
        assert!(consistent_assembly.r_at_u.min > consistent_assembly.r_au.min);
        assert!(consistent_assembly.r_at_u.max > consistent_assembly.r_au.max);
        assert_face_flux_bits(
            consistent_assembly.phi_hby_a_before_adjust,
            baseline_assembly.phi_hby_a_before_adjust,
        );
        assert_face_flux_bits(
            consistent_assembly.phi_hby_a_after_adjust,
            baseline_assembly.phi_hby_a_after_adjust,
        );
        assert_ne!(
            consistent_assembly.hby_a.l2_norm.to_bits(),
            baseline_assembly.hby_a.l2_norm.to_bits()
        );
        assert_ne!(
            consistent_assembly.pressure_source.l2_norm.to_bits(),
            baseline_assembly.pressure_source.l2_norm.to_bits()
        );
        assert_ne!(
            consistent_assembly
                .pressure_matrix
                .diagonal_sum_abs
                .to_bits(),
            baseline_assembly.pressure_matrix.diagonal_sum_abs.to_bits()
        );
        assert!(
            consistent
                .final_velocity
                .iter()
                .zip(&baseline.final_velocity)
                .any(|(actual, expected)| {
                    actual.x.to_bits() != expected.x.to_bits()
                        || actual.y.to_bits() != expected.y.to_bits()
                        || actual.z.to_bits() != expected.z.to_bits()
                })
        );

        assert_eq!(consistent.history.len(), consistent.simple_iterations);
        assert!(consistent.simple_iterations > 0);
        assert!(
            consistent.history.iter().all(|iteration| {
                iteration.pressure_correction_accepted
                    && iteration.momentum_linear_converged
                    && iteration.pressure_linear_converged
                    && iteration.pressure_linear_solves > 0
                    && iteration
                        .momentum_component_linear_solves
                        .iter()
                        .all(|solve| {
                            solve.stop_reason != super::LinearSolveStopReason::NotRun
                                && solve.effective_normalized_tolerance.is_finite()
                        })
                    && iteration.pressure_correction_linear_solves
                        [..iteration.pressure_linear_solves]
                        .iter()
                        .all(|solve| {
                            solve.stop_reason != super::LinearSolveStopReason::NotRun
                                && solve.effective_normalized_tolerance.is_finite()
                        })
            }),
            "{:#?}",
            consistent.history
        );
        assert_eq!(
            consistent.stop_reason,
            LaminarSimpleStopReason::ConvergenceCriteriaNotConfigured
        );
        assert_eq!(
            consistent
                .linear_solve_summary
                .momentum_component_non_converged_solves,
            0
        );
        assert_eq!(
            consistent
                .linear_solve_summary
                .pressure_correction_non_converged_solves,
            0
        );
        assert!(
            consistent
                .final_velocity
                .iter()
                .all(|value| { value.x.is_finite() && value.y.is_finite() && value.z.is_finite() })
        );
        assert!(
            consistent
                .final_pressure
                .iter()
                .all(|value| value.is_finite())
        );
        assert!(consistent.final_phi.iter().all(|value| value.is_finite()));
        assert!(consistent.final_continuity.l2_norm.is_finite());
    }

    #[test]
    fn rejects_excessive_non_orthogonal_correctors() {
        let mut runtime = two_cell_runtime();
        let fields = two_cell_fields();
        let mut options = minimal_laminar_options();
        options.non_orthogonal_correctors = MAX_NON_ORTHOGONAL_CORRECTORS + 1;

        let error = solve_laminar_simple(&mut runtime, &fields, &options)
            .expect_err("excessive pressure correctors must be rejected");

        assert!(matches!(
            error,
            MeshError::InvalidInput(message)
                if message
                    == format!(
                        "laminar SIMPLE nNonOrthogonalCorrectors must not exceed {}, got {}",
                        MAX_NON_ORTHOGONAL_CORRECTORS,
                        MAX_NON_ORTHOGONAL_CORRECTORS + 1
                    )
        ));
    }

    #[test]
    fn missing_pressure_payload_does_not_consume_velocity() {
        let mut runtime = two_cell_runtime();
        runtime.fields[1].values = None;
        let original_velocity = runtime.fields[0]
            .values
            .as_ref()
            .expect("velocity payload")
            .as_ptr();

        let error =
            solve_laminar_simple(&mut runtime, &two_cell_fields(), &minimal_laminar_options())
                .expect_err("missing pressure must fail before consuming velocity");

        assert!(matches!(
            error,
            MeshError::InvalidInput(message)
                if message
                    == "runtime field 'p' initial payload was already consumed or not loaded"
        ));
        let velocity = runtime.fields[0]
            .values
            .as_ref()
            .expect("velocity must remain available");
        assert_eq!(velocity.as_ptr(), original_velocity);
        assert_eq!(velocity, &[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn later_solver_error_leaves_one_shot_payloads_consumed() {
        let mut runtime = two_cell_runtime();
        let fields = two_cell_fields();
        let mut options = minimal_laminar_options();
        options.pressure_linear_solver = LaminarSimpleLinearSolver::Gamg;
        options.pressure_gamg_options = Some(GamgOptions {
            max_iterations: options.pressure_max_linear_iterations,
            tolerance: options.pressure_linear_tolerance,
            n_cells_in_coarsest_level: 1,
            agglomerator: GamgAgglomerator::FaceAreaPair,
            direct_solve_coarsest: true,
            ..GamgOptions::default()
        });
        options.profile_gamg = true;
        options.non_orthogonal_correctors = 1;

        let mut pressure_solves = 0;
        let error = {
            let mut invalidate_second_profile =
                |solve: usize,
                 report: &mut super::ScalarSolveReport,
                 _predicted_velocity: &[Point3],
                 _phi_hby_a: &[f64],
                 _continuity_star: super::ContinuitySummary| {
                    pressure_solves += 1;
                    if solve == 2 {
                        report.gamg_timing = Some(super::GamgKernelTiming::default());
                    }
                };
            super::solve_laminar_simple_driven(
                &mut runtime,
                &fields,
                &options,
                None,
                false,
                Some(&mut invalidate_second_profile),
                None,
            )
            .expect_err("a changed later GAMG profile must fail after payload transfer")
        };

        assert_eq!(pressure_solves, 2);
        assert!(matches!(
            error,
            MeshError::InvalidInput(message)
                if message == "GAMG profile hierarchy changed from 2 to 0 levels"
        ));
        assert!(runtime.fields.iter().all(|field| field.values.is_none()));

        let second = solve_laminar_simple(&mut runtime, &fields, &options)
            .expect_err("a failed one-shot run must not restore consumed payloads");
        assert!(matches!(
            second,
            MeshError::InvalidInput(message)
                if message
                    == "runtime field 'U' initial payload was already consumed or not loaded"
        ));
    }

    #[test]
    fn nonuniform_boundary_numeric_values_are_borrowed() {
        let field_set = two_cell_fields();
        let field = &field_set.fields[0];
        let value = FieldValueSummary::NonUniform {
            value_type: Some("List<vector>".to_string()),
            count: Some(1),
            values: Some(vec![1.0, 2.0, 3.0]),
        };
        let source = match &value {
            FieldValueSummary::NonUniform {
                values: Some(values),
                ..
            } => values.as_ptr(),
            _ => unreachable!(),
        };

        let parsed = super::parse_patch_numeric_values(&value, 3, 1, field, "inlet")
            .expect("nonuniform boundary values");
        match parsed {
            super::PatchNumericValues::Borrowed(values) => {
                assert_eq!(values.as_ptr(), source);
                assert_eq!(values, &[1.0, 2.0, 3.0]);
            }
            super::PatchNumericValues::Owned(_) => {
                panic!("nonuniform boundary values were copied")
            }
        }
    }

    #[test]
    fn face_area_pair_weights_follow_openfoam_axis_weighting() {
        let mut runtime = two_cell_runtime();
        runtime.mesh.face_area_vectors[0] = point(0.0, 4.0, 0.0);

        let weights =
            super::gamg_face_area_pair_weights(&runtime.mesh).expect("faceAreaPair mesh weights");

        assert_eq!(weights.len(), 1);
        assert_eq!(weights[0].cells(), (0, 1));
        assert_close(weights[0].weight(), 2.02);
    }

    #[test]
    fn runs_minimal_simple_pressure_correction_with_face_area_gamg() {
        let mut plain_runtime = two_cell_runtime();
        let mut profiled_runtime = two_cell_runtime();
        let fields = two_cell_fields();
        let mut options = minimal_laminar_options();
        options.pressure_linear_solver = LaminarSimpleLinearSolver::Gamg;
        options.pressure_gamg_options = Some(GamgOptions {
            max_iterations: options.pressure_max_linear_iterations,
            tolerance: options.pressure_linear_tolerance,
            n_cells_in_coarsest_level: 1,
            agglomerator: GamgAgglomerator::FaceAreaPair,
            n_pre_sweeps: 1,
            interpolate_correction: true,
            direct_solve_coarsest: false,
            ..GamgOptions::default()
        });
        let plain = solve_laminar_simple(&mut plain_runtime, &fields, &options)
            .expect("plain faceAreaPair interpolation SIMPLE report");

        options.profile_gamg = true;
        let report = solve_laminar_simple(&mut profiled_runtime, &fields, &options)
            .expect("profiled faceAreaPair interpolation SIMPLE report");

        assert!(report.simple_iterations > 0);
        assert!(report.total_pressure_linear_iterations > 0);
        assert_eq!(
            report
                .linear_solve_summary
                .pressure_correction_non_converged_solves,
            0
        );
        assert!(report.final_continuity.l2_norm.is_finite());
        assert!(report.fields.pressure.l2_norm.is_finite());
        let profile = report
            .timing
            .pressure_gamg_profile
            .as_ref()
            .expect("pressure GAMG profile");
        assert_eq!(
            profile.solves,
            report.linear_solve_summary.pressure_correction_solves
        );
        assert_eq!(profile.v_cycles, report.total_pressure_linear_iterations);
        assert_eq!(profile.hierarchy_builds, 1);
        assert_eq!(plain.simple_iterations, report.simple_iterations);
        assert_eq!(plain.stop_reason, report.stop_reason);
        assert_eq!(
            plain.total_pressure_linear_iterations,
            report.total_pressure_linear_iterations
        );
        for (plain, profiled) in plain.final_pressure.iter().zip(&report.final_pressure) {
            assert_eq!(plain.to_bits(), profiled.to_bits());
        }
        for (plain, profiled) in plain.final_velocity.iter().zip(&report.final_velocity) {
            assert_eq!(plain.x.to_bits(), profiled.x.to_bits());
            assert_eq!(plain.y.to_bits(), profiled.y.to_bits());
            assert_eq!(plain.z.to_bits(), profiled.z.to_bits());
        }
        for (plain, profiled) in plain.final_phi.iter().zip(&report.final_phi) {
            assert_eq!(plain.to_bits(), profiled.to_bits());
        }
    }

    #[test]
    fn pressure_pcg_profiling_is_opt_in_and_bit_identical() {
        let mut plain_runtime = two_cell_runtime();
        let mut profiled_runtime = two_cell_runtime();
        let fields = two_cell_fields();
        let mut plain_options = minimal_laminar_options();
        plain_options.pressure_linear_solver = LaminarSimpleLinearSolver::Pcg;
        plain_options.pressure_preconditioner = LaminarSimplePreconditioner::IncompleteCholesky;

        let plain = solve_laminar_simple(&mut plain_runtime, &fields, &plain_options)
            .expect("plain SIMPLE PCG report");
        let profiled = super::solve_laminar_simple_profiled_pcg(
            &mut profiled_runtime,
            &fields,
            &plain_options,
        )
        .expect("profiled SIMPLE PCG report");

        assert_eq!(plain.timing.pressure_pcg_total_seconds.to_bits(), 0);
        assert_eq!(
            plain
                .timing
                .pressure_preconditioner_update_seconds
                .to_bits(),
            0
        );
        assert_eq!(plain.timing.pressure_matrix_vector_seconds.to_bits(), 0);
        assert_eq!(
            plain
                .timing
                .pressure_preconditioner_application_seconds
                .to_bits(),
            0
        );
        assert_eq!(plain.timing.pressure_vector_operation_seconds.to_bits(), 0);
        assert_eq!(plain.timing.pressure_pcg_other_seconds.to_bits(), 0);
        assert_eq!(plain.timing.pressure_matrix_vector_products, 0);
        assert_eq!(plain.timing.pressure_preconditioner_applications, 0);

        assert!(profiled.timing.pressure_pcg_total_seconds >= 0.0);
        assert!(profiled.timing.pressure_preconditioner_update_seconds >= 0.0);
        assert!(profiled.timing.pressure_matrix_vector_products > 0);
        assert!(
            profiled.timing.pressure_preconditioner_applications
                <= profiled.timing.pressure_matrix_vector_products
        );
        assert!(
            profiled.timing.pressure_pcg_total_seconds
                <= profiled.timing.pressure_linear_solve_seconds + 1.0e-9
        );

        assert_eq!(plain.simple_iterations, profiled.simple_iterations);
        assert_eq!(plain.converged, profiled.converged);
        assert_eq!(plain.stop_reason, profiled.stop_reason);
        assert_eq!(
            plain.total_momentum_linear_iterations,
            profiled.total_momentum_linear_iterations
        );
        assert_eq!(
            plain.total_pressure_linear_iterations,
            profiled.total_pressure_linear_iterations
        );
        assert_eq!(
            plain.final_momentum_normalized_residual_norm.to_bits(),
            profiled.final_momentum_normalized_residual_norm.to_bits()
        );
        assert_eq!(
            plain
                .final_pressure_correction_normalized_residual_norm
                .to_bits(),
            profiled
                .final_pressure_correction_normalized_residual_norm
                .to_bits()
        );
        for (plain, profiled) in plain.final_pressure.iter().zip(&profiled.final_pressure) {
            assert_eq!(plain.to_bits(), profiled.to_bits());
        }
        for (plain, profiled) in plain.final_velocity.iter().zip(&profiled.final_velocity) {
            assert_eq!(plain.x.to_bits(), profiled.x.to_bits());
            assert_eq!(plain.y.to_bits(), profiled.y.to_bits());
            assert_eq!(plain.z.to_bits(), profiled.z.to_bits());
        }
        for (plain, profiled) in plain.final_phi.iter().zip(&profiled.final_phi) {
            assert_eq!(plain.to_bits(), profiled.to_bits());
        }
        assert_eq!(plain.history.len(), profiled.history.len());
        for (plain, profiled) in plain.history.iter().zip(&profiled.history) {
            assert_eq!(plain.iteration, profiled.iteration);
            assert_eq!(
                plain.pressure_linear_iterations,
                profiled.pressure_linear_iterations
            );
            assert_eq!(
                plain.pressure_correction_normalized_residual_norm.to_bits(),
                profiled
                    .pressure_correction_normalized_residual_norm
                    .to_bits()
            );
            assert_eq!(
                plain.momentum_normalized_residual_norm.to_bits(),
                profiled.momentum_normalized_residual_norm.to_bits()
            );
        }
    }

    #[test]
    fn profiled_pcg_api_rejects_other_pressure_solvers_without_consuming_fields() {
        let mut runtime = two_cell_runtime();
        let fields = two_cell_fields();
        let mut options = minimal_laminar_options();
        options.pressure_linear_solver = LaminarSimpleLinearSolver::Cg;
        let velocity_before = runtime.fields[0]
            .values
            .as_ref()
            .expect("velocity payload before rejection");
        let pressure_before = runtime.fields[1]
            .values
            .as_ref()
            .expect("pressure payload before rejection");
        let velocity_ptr = velocity_before.as_ptr();
        let pressure_ptr = pressure_before.as_ptr();
        let velocity_values = velocity_before.clone();
        let pressure_values = pressure_before.clone();

        let error = super::solve_laminar_simple_profiled_pcg(&mut runtime, &fields, &options)
            .expect_err("profiled PCG API must reject a CG pressure solver");

        match error {
            MeshError::InvalidInput(message) => assert_eq!(
                message,
                "laminar SIMPLE PCG profiling requires the PCG pressure solver"
            ),
            other => panic!("unexpected profiling rejection: {other}"),
        }
        let velocity_after = runtime.fields[0]
            .values
            .as_ref()
            .expect("velocity payload after rejection");
        let pressure_after = runtime.fields[1]
            .values
            .as_ref()
            .expect("pressure payload after rejection");
        assert_eq!(velocity_after.as_ptr(), velocity_ptr);
        assert_eq!(pressure_after.as_ptr(), pressure_ptr);
        assert_eq!(velocity_after, &velocity_values);
        assert_eq!(pressure_after, &pressure_values);
    }

    #[test]
    fn checked_update_metrics_reject_finite_arithmetic_overflow() {
        let velocity = [Point3 {
            x: f64::MAX,
            y: f64::MAX,
            z: 0.0,
        }];

        assert!(
            super::checked_update_metrics(
                &velocity,
                &[0.0],
                &[0.0],
                super::ContinuitySummary::default(),
            )
            .is_none()
        );
    }

    #[test]
    fn checked_scaled_sum_squares_is_finite_and_overflow_exact() {
        assert_eq!(super::checked_l2_norm([0.0, -0.0]), Some(0.0));
        assert_eq!(super::checked_l2_norm([3.0, 4.0]), Some(5.0));

        let subnormal = f64::from_bits(1);
        let subnormal_norm = super::checked_l2_norm([subnormal, subnormal])
            .expect("subnormal norm remains representable");
        assert!(subnormal_norm.is_finite());
        assert!(subnormal_norm > 0.0);

        let large = f64::MAX / 4.0;
        let large_norm = super::checked_l2_norm([large, -large])
            .expect("large finite norm remains representable");
        assert!(large_norm.is_finite());
        assert!(large_norm > large);

        assert!(super::checked_l2_norm([f64::MAX, f64::MAX]).is_none());
        assert!(super::checked_l2_norm([f64::NAN]).is_none());
        assert!(super::checked_l2_norm([f64::INFINITY]).is_none());
    }

    #[test]
    fn pressure_metrics_ignore_a_large_uniform_gauge_shift() {
        let velocity = [point(1.0, 0.0, 0.0), point(2.0, 0.0, 0.0)];
        let phi = [1.0, 2.0];
        let before_pressure = [0.0, 1.0];
        let after_pressure = [1.0e12, 1.0e12 + 1.0];
        let continuity = super::ContinuitySummary::default();
        super::checked_update_metrics(&velocity, &before_pressure, &phi, continuity)
            .expect("initial metrics");
        let shifted = super::checked_candidate_update_metrics(
            &velocity,
            &before_pressure,
            &phi,
            &velocity,
            &after_pressure,
            &phi,
            continuity,
        )
        .expect("shifted metrics");

        assert_eq!(
            super::checked_gauge_invariant_scalar_l2(&before_pressure)
                .unwrap()
                .to_bits(),
            super::checked_gauge_invariant_scalar_l2(&after_pressure)
                .unwrap()
                .to_bits()
        );
        assert_eq!(shifted.pressure_change_l2.to_bits(), 0.0_f64.to_bits());
        assert_eq!(shifted.velocity_change_l2.to_bits(), 0.0_f64.to_bits());
    }

    #[test]
    fn large_finite_simple_update_commits() {
        let mut runtime = two_cell_runtime();
        let mut fields = two_cell_fields();
        fields.fields[1].boundary_patches[1].patch_type = Some("zeroGradient".to_string());
        fields.fields[1].boundary_patches[1].value = None;
        let mut options = minimal_laminar_options();
        options.max_simple_iterations = 1;
        options.pressure_reference_cell = Some(0);
        let initial_velocity = [point(1.0, 0.0, 0.0), point(1.0, 0.0, 0.0)];
        let mut force_large_correction =
            |_solve: usize,
             report: &mut super::ScalarSolveReport,
             _predicted_velocity: &[Point3],
             _phi_hby_a: &[f64],
             _continuity_star: super::ContinuitySummary| {
                report.solution.copy_from_slice(&[1.0e12, -1.0e12]);
            };

        let report = super::solve_laminar_simple_driven(
            &mut runtime,
            &fields,
            &options,
            None,
            false,
            Some(&mut force_large_correction),
            None,
        )
        .expect("large finite update");

        assert_eq!(
            report.stop_reason,
            LaminarSimpleStopReason::ConvergenceCriteriaNotConfigured
        );
        assert!(report.pressure_assembly.is_some());
        assert_eq!(report.simple_iterations, 1);
        assert!(report.history[0].pressure_correction_accepted);
        assert_eq!(report.history[0].pressure_correction_update_scale, 1.0);
        assert!(
            report
                .final_velocity
                .iter()
                .all(|value| value.x.is_finite())
        );
        assert!(
            report
                .final_velocity
                .iter()
                .zip(initial_velocity)
                .any(|(actual, initial)| actual.x.to_bits() != initial.x.to_bits())
        );
        let expected_pressure: [f64; 2] = [0.7 * 1.0 + 0.3 * 1.0e12, -0.3 * 1.0e12];
        assert_eq!(report.final_pressure.len(), expected_pressure.len());
        for (actual, expected) in report.final_pressure.iter().zip(expected_pressure) {
            assert_eq!(actual.to_bits(), expected.to_bits());
        }
        let committed_pressure_span = report.final_pressure[0] - report.final_pressure[1];
        assert_eq!(
            committed_pressure_span.to_bits(),
            (expected_pressure[0] - expected_pressure[1]).to_bits()
        );
        assert!(committed_pressure_span > 1_048_576.0);
        assert!(committed_pressure_span > 64.0);
        assert!(report.final_phi.iter().all(|value| value.is_finite()));
        assert!(report.final_continuity.l2_norm.is_finite());
        assert!(report.final_continuity.l2_norm > 0.0);
    }

    #[test]
    fn repeated_growing_and_sign_reversing_finite_updates_commit() {
        let mut runtime = two_cell_runtime();
        let mut fields = two_cell_fields();
        fields.fields[1].boundary_patches[1].patch_type = Some("zeroGradient".to_string());
        fields.fields[1].boundary_patches[1].value = None;
        let mut options = minimal_laminar_options();
        options.max_simple_iterations = 4;
        options.pressure_reference_cell = Some(0);
        let seeded_pressure_difference = 0.7 * 1.0 + 0.3 * 200.0;
        let first_reversal_difference = -(2.2 / 0.3) * seeded_pressure_difference;
        let second_reversal_difference = -1.5 * first_reversal_difference;
        let third_reversal_difference = -1.5 * second_reversal_difference;
        let forced_solutions = [
            [100.0, -100.0],
            [
                0.5 * first_reversal_difference,
                -0.5 * first_reversal_difference,
            ],
            [
                0.5 * second_reversal_difference,
                -0.5 * second_reversal_difference,
            ],
            [
                0.5 * third_reversal_difference,
                -0.5 * third_reversal_difference,
            ],
        ];
        let mut iteration = 0;
        let mut drive_corrections =
            |_solve: usize,
             report: &mut super::ScalarSolveReport,
             _predicted_velocity: &[Point3],
             _phi_hby_a: &[f64],
             _continuity_star: super::ContinuitySummary| {
                report
                    .solution
                    .copy_from_slice(&forced_solutions[iteration]);
                iteration += 1;
            };

        let report = super::solve_laminar_simple_driven(
            &mut runtime,
            &fields,
            &options,
            None,
            false,
            Some(&mut drive_corrections),
            None,
        )
        .expect("repeated finite updates");

        assert_eq!(iteration, options.max_simple_iterations);
        assert_eq!(report.simple_iterations, options.max_simple_iterations);
        assert_eq!(
            report.stop_reason,
            LaminarSimpleStopReason::ConvergenceCriteriaNotConfigured
        );
        assert!(report.history.iter().all(|summary| {
            summary.pressure_correction_accepted
                && summary.pressure_correction_update_scale == 1.0
                && summary.relative_velocity_change_l2.is_finite()
                && summary.relative_pressure_change_l2.is_finite()
        }));
        assert!(
            report
                .history
                .iter()
                .skip(1)
                .all(|summary| summary.relative_pressure_change_l2 >= 1.5),
            "{:?}",
            report
                .history
                .iter()
                .map(|summary| summary.relative_pressure_change_l2)
                .collect::<Vec<_>>()
        );
        assert_eq!(
            report
                .history
                .iter()
                .skip(1)
                .filter(|summary| summary.relative_pressure_change_l2 >= 1.5)
                .count(),
            3
        );
        assert_eq!(
            report
                .history
                .iter()
                .filter(
                    |summary| summary.continuity_after.l2_norm > summary.continuity_before.l2_norm
                )
                .count(),
            2
        );
        assert!(forced_solutions.windows(2).all(|updates| {
            updates[0][0].is_sign_positive() != updates[1][0].is_sign_positive()
                && updates[1][0].abs() > updates[0][0].abs()
        }));
        let mut expected_pressure = [1.0, 0.0];
        for forced in forced_solutions {
            for (value, target) in expected_pressure.iter_mut().zip(forced) {
                *value += options.pressure_relaxation * (target - *value);
            }
        }
        for (actual, expected) in report.final_pressure.iter().zip(expected_pressure) {
            assert_eq!(actual.to_bits(), expected.to_bits());
        }
        assert!(
            report
                .final_velocity
                .iter()
                .all(|value| value.x.is_finite())
        );
        assert!(report.final_phi.iter().all(|value| value.is_finite()));
        assert!(report.final_continuity.l2_norm.is_finite());
    }

    #[test]
    fn non_finite_and_overflowing_simple_updates_roll_back_atomically() {
        for invalid_solution in [
            [f64::NAN, 0.0],
            [f64::INFINITY, 0.0],
            [f64::NEG_INFINITY, 0.0],
            [f64::MAX, -f64::MAX],
        ] {
            let mut runtime = two_cell_runtime();
            let fields = two_cell_fields();
            let options = minimal_laminar_options();
            let initial_velocity = [point(1.0, 0.0, 0.0), point(1.0, 0.0, 0.0)];
            let initial_pressure = [1.0, 0.0];
            let velocity_boundary =
                super::vector_face_treatments(&runtime.mesh, &fields.fields[0]).unwrap();
            let initial_phi =
                compute_face_flux(&runtime.mesh, &initial_velocity, &velocity_boundary).unwrap();
            let initial_continuity =
                super::summarize_continuity(&net_cell_flux(&runtime.mesh, &initial_phi).unwrap());
            let mut drive_invalid =
                |_solve: usize,
                 report: &mut super::ScalarSolveReport,
                 _predicted_velocity: &[Point3],
                 _phi_hby_a: &[f64],
                 _continuity_star: super::ContinuitySummary| {
                    report.solution.copy_from_slice(&invalid_solution);
                };

            let report = super::solve_laminar_simple_driven(
                &mut runtime,
                &fields,
                &options,
                None,
                false,
                Some(&mut drive_invalid),
                None,
            )
            .expect("invalid candidate report");

            assert_eq!(
                report.stop_reason,
                LaminarSimpleStopReason::SolverInvalidState
            );
            for (actual, expected) in report.final_velocity.iter().zip(initial_velocity) {
                assert_eq!(actual.x.to_bits(), expected.x.to_bits());
                assert_eq!(actual.y.to_bits(), expected.y.to_bits());
                assert_eq!(actual.z.to_bits(), expected.z.to_bits());
            }
            assert_eq!(report.final_pressure, initial_pressure);
            assert_eq!(report.final_phi, initial_phi);
            assert_eq!(
                format!("{:?}", report.final_continuity),
                format!("{initial_continuity:?}")
            );
            let rejected = report.history.last().unwrap();
            assert!(!rejected.pressure_correction_accepted);
            assert_eq!(rejected.pressure_correction_update_scale, 0.0);
            assert_eq!(
                format!("{:?}", rejected.continuity_after),
                format!("{initial_continuity:?}")
            );
        }
    }

    #[test]
    fn invalid_second_update_preserves_last_accepted_state_and_diagnostics() {
        fn assert_f64_bits(actual: f64, expected: f64) {
            assert_eq!(actual.to_bits(), expected.to_bits());
        }

        fn assert_continuity_bits(
            actual: super::ContinuitySummary,
            expected: super::ContinuitySummary,
        ) {
            assert_f64_bits(actual.l2_norm, expected.l2_norm);
            assert_f64_bits(actual.max_abs, expected.max_abs);
            assert_f64_bits(actual.sum_abs, expected.sum_abs);
            assert_f64_bits(actual.global_sum, expected.global_sum);
        }

        fn assert_scalar_diagnostics_bits(
            actual: super::ScalarDiagnosticSummary,
            expected: super::ScalarDiagnosticSummary,
        ) {
            assert_f64_bits(actual.min, expected.min);
            assert_f64_bits(actual.max, expected.max);
            assert_f64_bits(actual.l2_norm, expected.l2_norm);
            assert_f64_bits(actual.sum, expected.sum);
            assert_f64_bits(actual.sum_abs, expected.sum_abs);
        }

        fn assert_vector_diagnostics_bits(
            actual: super::VectorDiagnosticSummary,
            expected: super::VectorDiagnosticSummary,
        ) {
            assert_f64_bits(actual.min_magnitude, expected.min_magnitude);
            assert_f64_bits(actual.max_magnitude, expected.max_magnitude);
            assert_f64_bits(actual.l2_norm, expected.l2_norm);
            assert_f64_bits(actual.x_min, expected.x_min);
            assert_f64_bits(actual.x_max, expected.x_max);
            assert_f64_bits(actual.y_min, expected.y_min);
            assert_f64_bits(actual.y_max, expected.y_max);
            assert_f64_bits(actual.z_min, expected.z_min);
            assert_f64_bits(actual.z_max, expected.z_max);
        }

        fn assert_face_flux_diagnostics_bits(
            actual: super::FaceFluxDiagnosticSummary,
            expected: super::FaceFluxDiagnosticSummary,
        ) {
            assert_f64_bits(actual.min, expected.min);
            assert_f64_bits(actual.max, expected.max);
            assert_f64_bits(actual.l2_norm, expected.l2_norm);
            assert_f64_bits(actual.sum, expected.sum);
            assert_f64_bits(actual.sum_abs, expected.sum_abs);
            assert_f64_bits(actual.internal_sum_abs, expected.internal_sum_abs);
            assert_f64_bits(actual.boundary_sum, expected.boundary_sum);
            assert_f64_bits(actual.boundary_sum_abs, expected.boundary_sum_abs);
        }

        fn assert_matrix_diagnostics_bits(
            actual: super::MatrixDiagnosticSummary,
            expected: super::MatrixDiagnosticSummary,
        ) {
            assert_eq!(actual.rows, expected.rows);
            assert_eq!(actual.cols, expected.cols);
            assert_eq!(actual.nonzeros, expected.nonzeros);
            assert_f64_bits(actual.diagonal_min, expected.diagonal_min);
            assert_f64_bits(actual.diagonal_max, expected.diagonal_max);
            assert_f64_bits(actual.diagonal_sum_abs, expected.diagonal_sum_abs);
            assert_f64_bits(actual.off_diagonal_sum_abs, expected.off_diagonal_sum_abs);
            assert_f64_bits(actual.max_row_sum_abs, expected.max_row_sum_abs);
            assert_f64_bits(
                actual.max_row_off_diagonal_sum_abs,
                expected.max_row_off_diagonal_sum_abs,
            );
        }

        fn run(max_iterations: usize, invalidate_second: bool) -> super::LaminarSimpleReport {
            let mut runtime = two_cell_runtime();
            let fields = two_cell_fields();
            let mut options = minimal_laminar_options();
            options.max_simple_iterations = max_iterations;
            let mut iteration = 0;
            let mut drive =
                |_solve: usize,
                 report: &mut super::ScalarSolveReport,
                 _predicted_velocity: &[Point3],
                 _phi_hby_a: &[f64],
                 _continuity_star: super::ContinuitySummary| {
                    iteration += 1;
                    if iteration == 1 {
                        report.solution.copy_from_slice(&[2.0, -2.0]);
                    } else if invalidate_second {
                        report.solution.copy_from_slice(&[f64::MAX, -f64::MAX]);
                    }
                };
            super::solve_laminar_simple_driven(
                &mut runtime,
                &fields,
                &options,
                None,
                false,
                Some(&mut drive),
                None,
            )
            .expect("driven SIMPLE report")
        }

        let accepted = run(1, false);
        let rejected = run(2, true);

        assert_eq!(
            rejected.stop_reason,
            LaminarSimpleStopReason::SolverInvalidState
        );
        for (actual, expected) in rejected.final_velocity.iter().zip(&accepted.final_velocity) {
            assert_eq!(actual.x.to_bits(), expected.x.to_bits());
            assert_eq!(actual.y.to_bits(), expected.y.to_bits());
            assert_eq!(actual.z.to_bits(), expected.z.to_bits());
        }
        for (actual, expected) in rejected.final_pressure.iter().zip(&accepted.final_pressure) {
            assert_f64_bits(*actual, *expected);
        }
        for (actual, expected) in rejected.final_phi.iter().zip(&accepted.final_phi) {
            assert_f64_bits(*actual, *expected);
        }
        assert_continuity_bits(rejected.final_continuity, accepted.final_continuity);
        assert_vector_diagnostics_bits(rejected.fields.velocity, accepted.fields.velocity);
        assert_scalar_diagnostics_bits(rejected.fields.pressure, accepted.fields.pressure);
        assert_f64_bits(
            rejected.operator_summary.phi_min,
            accepted.operator_summary.phi_min,
        );
        assert_f64_bits(
            rejected.operator_summary.phi_max,
            accepted.operator_summary.phi_max,
        );
        assert_f64_bits(
            rejected.operator_summary.phi_sum_abs,
            accepted.operator_summary.phi_sum_abs,
        );
        assert_f64_bits(
            rejected.operator_summary.grad_p_l2_norm,
            accepted.operator_summary.grad_p_l2_norm,
        );
        assert_f64_bits(
            rejected.operator_summary.hby_a_l2_norm,
            accepted.operator_summary.hby_a_l2_norm,
        );
        assert_f64_bits(
            rejected.operator_summary.div_phi_u_l2_norm,
            accepted.operator_summary.div_phi_u_l2_norm,
        );
        let rejected_assembly = rejected.pressure_assembly.unwrap();
        let accepted_assembly = accepted.pressure_assembly.unwrap();
        assert_scalar_diagnostics_bits(rejected_assembly.r_au, accepted_assembly.r_au);
        assert_scalar_diagnostics_bits(rejected_assembly.r_at_u, accepted_assembly.r_at_u);
        assert_vector_diagnostics_bits(rejected_assembly.hby_a, accepted_assembly.hby_a);
        assert_face_flux_diagnostics_bits(
            rejected_assembly.phi_hby_a_before_adjust,
            accepted_assembly.phi_hby_a_before_adjust,
        );
        assert_face_flux_diagnostics_bits(
            rejected_assembly.phi_hby_a_after_adjust,
            accepted_assembly.phi_hby_a_after_adjust,
        );
        assert_scalar_diagnostics_bits(
            rejected_assembly.pressure_source,
            accepted_assembly.pressure_source,
        );
        assert_face_flux_diagnostics_bits(
            rejected_assembly.pressure_equation_flux,
            accepted_assembly.pressure_equation_flux,
        );
        assert_matrix_diagnostics_bits(
            rejected_assembly.pressure_matrix,
            accepted_assembly.pressure_matrix,
        );
        assert_face_flux_diagnostics_bits(
            rejected_assembly.pressure_flux,
            accepted_assembly.pressure_flux,
        );
        assert_face_flux_diagnostics_bits(
            rejected_assembly.corrected_phi,
            accepted_assembly.corrected_phi,
        );
        assert_eq!(
            rejected
                .final_momentum_initial_normalized_residual_norm
                .to_bits(),
            accepted
                .final_momentum_initial_normalized_residual_norm
                .to_bits()
        );
        assert_eq!(
            rejected.final_momentum_residual_norm.to_bits(),
            accepted.final_momentum_residual_norm.to_bits()
        );
        assert_eq!(
            rejected.final_momentum_normalized_residual_norm.to_bits(),
            accepted.final_momentum_normalized_residual_norm.to_bits()
        );
        assert_eq!(
            rejected
                .final_pressure_correction_initial_normalized_residual_norm
                .to_bits(),
            accepted
                .final_pressure_correction_initial_normalized_residual_norm
                .to_bits()
        );
        assert_eq!(
            rejected.final_pressure_correction_residual_norm.to_bits(),
            accepted.final_pressure_correction_residual_norm.to_bits()
        );
        assert_eq!(
            rejected
                .final_pressure_correction_normalized_residual_norm
                .to_bits(),
            accepted
                .final_pressure_correction_normalized_residual_norm
                .to_bits()
        );
        let summary = rejected.history.last().unwrap();
        assert!(!summary.pressure_correction_accepted);
        assert_eq!(summary.pressure_correction_update_scale, 0.0);
        assert_continuity_bits(summary.continuity_after, accepted.final_continuity);
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
    fn residual_control_uses_strict_absolute_tolerance() {
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

        let factor = super::ldu_l1_residual_normalisation_factor(
            &matrix,
            &source,
            &solution,
            &matrix_solution,
        )
        .expect("normalisation factor");
        let residual = source
            .iter()
            .zip(matrix_solution)
            .map(|(source, matrix_value)| source - matrix_value)
            .collect::<Vec<_>>();

        assert_close(factor, 1.0);
        assert_close(super::l1_norm(&residual) / factor, 1.0);
    }

    #[test]
    fn openfoam_residual_normalisation_adds_small_without_flooring() {
        let matrix = crate::linear::CsrMatrix::from_rows(vec![vec![(0, 1.0)]], 1).expect("matrix");
        let solution = [0.0];
        let matrix_solution = [0.0];

        for raw_factor in [0.0, 0.5e-20, 1.0e-12] {
            let factor = super::ldu_l1_residual_normalisation_factor(
                &matrix,
                &[raw_factor],
                &solution,
                &matrix_solution,
            )
            .expect("normalisation factor");
            let expected = raw_factor + 1.0e-20;

            assert_eq!(
                factor.to_bits(),
                expected.to_bits(),
                "raw factor {raw_factor:e}"
            );
        }
    }

    #[test]
    fn ldu_l1_limit_is_conservatively_translated_to_a_strict_l2_limit() {
        let tolerance = super::strict_l2_tolerance_for_l1_limit(1.0, 4.0);

        assert_eq!(tolerance, 0.5_f64.next_down());
        assert!(4.0_f64.sqrt() * tolerance < 1.0);
    }

    fn solve_non_gamg_relative_probe(
        solver: LaminarSimpleLinearSolver,
        tolerance: f64,
        relative_tolerance: f64,
    ) -> super::ScalarSolveReport {
        let matrix = CsrMatrix::from_rows(vec![vec![(0, 2.0)], vec![(1, 4.0)]], 2).expect("matrix");
        let mut workspace = super::ScalarSolveWorkspace::new(2);
        super::solve_scalar_system_with_workspaces(
            &matrix,
            &[2.0, 8.0],
            None,
            super::ScalarSolveControls {
                solver,
                preconditioner: LaminarSimplePreconditioner::None,
                tolerance,
                relative_tolerance,
                max_iterations: 16,
                gamg_options: None,
                profile_gamg: false,
                profile_pcg: false,
            },
            &mut workspace,
            None,
            None,
        )
        .expect("non-GAMG relative probe")
    }

    #[test]
    fn non_gamg_relative_tolerance_uses_strict_openfoam_boundaries_for_every_solver() {
        let solvers = [
            LaminarSimpleLinearSolver::BiCgStab,
            LaminarSimpleLinearSolver::Cg,
            LaminarSimpleLinearSolver::GaussSeidel,
            LaminarSimpleLinearSolver::SymGaussSeidel,
            LaminarSimpleLinearSolver::Pcg,
            LaminarSimpleLinearSolver::Jacobi,
        ];
        for solver in solvers {
            let zero_rel = solve_non_gamg_relative_probe(solver, 1.0, 0.0);
            assert_eq!(
                zero_rel.initial_normalized_residual_norm.to_bits(),
                1.0f64.to_bits()
            );
            assert!(
                zero_rel.iterations > 0,
                "{solver} accepted absolute equality"
            );
            assert!(zero_rel.converged, "{solver} failed the absolute probe");
            assert_eq!(
                zero_rel.stop_reason,
                super::LinearSolveStopReason::AbsoluteTolerance
            );

            let absolute_next = solve_non_gamg_relative_probe(solver, 1.0f64.next_up(), 0.0);
            assert_eq!(absolute_next.iterations, 0, "{solver} missed abs next_up");
            assert_eq!(
                absolute_next.effective_normalized_tolerance.to_bits(),
                1.0f64.next_up().to_bits()
            );

            let relative_equal = solve_non_gamg_relative_probe(solver, 0.0, 1.0);
            assert!(
                relative_equal.iterations > 0,
                "{solver} accepted relative equality"
            );
            assert!(
                relative_equal.converged,
                "{solver} failed the relative probe"
            );
            assert_eq!(
                relative_equal.stop_reason,
                super::LinearSolveStopReason::RelativeTolerance
            );

            let relative_next = solve_non_gamg_relative_probe(solver, 0.0, 1.0f64.next_up());
            assert_eq!(relative_next.iterations, 0, "{solver} missed rel next_up");
            assert_eq!(
                relative_next.stop_reason,
                super::LinearSolveStopReason::RelativeTolerance
            );

            let greater_than_one = solve_non_gamg_relative_probe(solver, 0.0, 2.0);
            assert_eq!(
                greater_than_one.iterations, 0,
                "{solver} artificially capped relTol above one"
            );
            assert_eq!(
                greater_than_one.effective_normalized_tolerance.to_bits(),
                2.0f64.to_bits()
            );
        }
    }

    #[test]
    fn pcg_flow_entrypoint_reuses_prepared_residual_and_matches_direct_profiled_solve() {
        let matrix =
            CsrMatrix::from_rows(vec![vec![(0, 4.0), (1, 1.0)], vec![(0, 1.0), (1, 3.0)]], 2)
                .expect("flow PCG matrix");
        let rhs = [1.0, 2.0];
        let initial = [0.0, 0.0];
        let matrix_product = matrix.matvec(&initial).expect("initial matrix product");
        let normalization_factor =
            super::ldu_l1_residual_normalisation_factor(&matrix, &rhs, &initial, &matrix_product)
                .expect("normalization factor");
        let tolerance = (0.75 / normalization_factor).next_up();

        let mut scalar_workspace = super::ScalarSolveWorkspace::new(2);
        let mut flow_workspace =
            PreconditionedConjugateGradientWorkspace::new(&matrix, CgPreconditioner::None)
                .expect("flow PCG workspace");
        let flow_report = super::solve_scalar_system_with_workspaces(
            &matrix,
            &rhs,
            Some(&initial),
            super::ScalarSolveControls {
                solver: LaminarSimpleLinearSolver::Pcg,
                preconditioner: LaminarSimplePreconditioner::None,
                tolerance,
                relative_tolerance: 0.0,
                max_iterations: 2,
                gamg_options: None,
                profile_gamg: false,
                profile_pcg: true,
            },
            &mut scalar_workspace,
            Some(&mut flow_workspace),
            None,
        )
        .expect("flow prepared PCG solve");

        let mut direct_workspace =
            PreconditionedConjugateGradientWorkspace::new(&matrix, CgPreconditioner::None)
                .expect("direct PCG workspace");
        let mut direct_prepared_values = vec![0.0; rhs.len()];
        let direct_prepared_residual = PreparedPcgResidual::write_from_rhs_and_matrix_product(
            &mut direct_prepared_values,
            &rhs,
            &matrix_product,
        )
        .expect("direct prepared residual");
        let direct = direct_workspace
            .solve_normalized_l1_prepared_profiled(
                &matrix,
                &rhs,
                Some(&initial),
                direct_prepared_residual,
                PreparedPcgInitialTiming::default(),
                NormalizedL1PcgSolveControls {
                    max_iterations: 2,
                    normalization_factor,
                    tolerance,
                    relative_tolerance: 0.0,
                    preconditioner: CgPreconditioner::None,
                },
            )
            .expect("direct prepared PCG solve");

        assert_eq!(flow_report.iterations, direct.report.iterations);
        assert_eq!(flow_report.termination, direct.report.termination);
        assert_eq!(
            flow_report.residual_norm.to_bits(),
            direct.report.residual_norm.to_bits()
        );
        assert_eq!(
            flow_report
                .solution
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            direct
                .report
                .solution
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
        let timing = flow_report.pcg_timing.expect("flow PCG timing");
        assert_eq!(
            timing.matrix_vector_products,
            flow_report.iterations + 1,
            "the prepared flow matvec must be counted once without a duplicate initial A*x"
        );
        assert_eq!(timing.preconditioner_applications, flow_report.iterations);
        assert_eq!(
            direct.timing.matrix_vector_products,
            direct.report.iterations + 1
        );
    }

    #[test]
    fn non_gamg_effective_tolerance_selects_absolute_or_relative_limit_without_a_cap() {
        assert_eq!(
            super::effective_non_gamg_normalized_tolerance(0.2, 0.5, 0.25).to_bits(),
            0.2f64.to_bits()
        );
        assert_eq!(
            super::effective_non_gamg_normalized_tolerance(0.1, 0.5, 0.25).to_bits(),
            0.125f64.to_bits()
        );
        assert_eq!(
            super::effective_normalized_tolerance_and_reason(
                LaminarSimpleLinearSolver::Cg,
                0.125,
                0.5,
                0.25,
            ),
            (0.125, super::LinearSolveStopReason::AbsoluteTolerance)
        );
        assert_eq!(
            super::effective_non_gamg_normalized_tolerance(
                0.1,
                super::OPENFOAM_RELATIVE_TOLERANCE_SMALL,
                0.25,
            )
            .to_bits(),
            0.1f64.to_bits()
        );
        assert_eq!(
            super::effective_non_gamg_normalized_tolerance(0.0, 8.0, 0.25).to_bits(),
            2.0f64.to_bits()
        );
    }

    #[test]
    fn every_linear_solver_uses_the_strict_openfoam_relative_tolerance_boundary() {
        assert_eq!(
            super::OPENFOAM_RELATIVE_TOLERANCE_SMALL.to_bits(),
            1.0e-20_f64.to_bits()
        );
        let below_openfoam_small = super::OPENFOAM_RELATIVE_TOLERANCE_SMALL.next_down();
        let above_openfoam_small = super::OPENFOAM_RELATIVE_TOLERANCE_SMALL.next_up();

        for solver in [
            LaminarSimpleLinearSolver::Gamg,
            LaminarSimpleLinearSolver::BiCgStab,
            LaminarSimpleLinearSolver::Cg,
            LaminarSimpleLinearSolver::GaussSeidel,
            LaminarSimpleLinearSolver::SymGaussSeidel,
            LaminarSimpleLinearSolver::Pcg,
            LaminarSimpleLinearSolver::Jacobi,
        ] {
            assert!(!super::linear_relative_tolerance_is_active(
                solver,
                below_openfoam_small
            ));
            assert!(!super::linear_relative_tolerance_is_active(
                solver,
                super::OPENFOAM_RELATIVE_TOLERANCE_SMALL
            ));
            assert!(super::linear_relative_tolerance_is_active(
                solver,
                above_openfoam_small
            ));
        }
        assert!(!super::linear_relative_tolerance_is_active(
            LaminarSimpleLinearSolver::Gamg,
            0.0
        ));
        assert_eq!(
            super::effective_normalized_tolerance_and_reason(
                LaminarSimpleLinearSolver::Gamg,
                0.0,
                super::OPENFOAM_RELATIVE_TOLERANCE_SMALL,
                1.0,
            ),
            (0.0, super::LinearSolveStopReason::AbsoluteTolerance)
        );
        assert_eq!(
            super::effective_normalized_tolerance_and_reason(
                LaminarSimpleLinearSolver::Gamg,
                0.0,
                above_openfoam_small,
                1.0,
            ),
            (
                above_openfoam_small,
                super::LinearSolveStopReason::RelativeTolerance,
            )
        );
    }

    #[test]
    fn scalar_relative_tolerance_rejects_invalid_values_before_workspace_mutation() {
        let matrix = CsrMatrix::from_rows(vec![vec![(0, 1.0)]], 1).expect("matrix");
        for relative_tolerance in [-1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut workspace = super::ScalarSolveWorkspace::new(1);
            let pointers = (
                workspace.zero_initial.as_ptr(),
                workspace.matrix_product.as_ptr(),
                workspace.residual.as_ptr(),
            );
            let error = super::solve_scalar_system_with_workspaces(
                &matrix,
                &[1.0],
                None,
                super::ScalarSolveControls {
                    solver: LaminarSimpleLinearSolver::Cg,
                    preconditioner: LaminarSimplePreconditioner::None,
                    tolerance: 0.0,
                    relative_tolerance,
                    max_iterations: 1,
                    gamg_options: None,
                    profile_gamg: false,
                    profile_pcg: false,
                },
                &mut workspace,
                None,
                None,
            )
            .err()
            .expect("invalid relTol must fail closed");
            assert_eq!(
                error.to_string(),
                format!(
                    "scalar solve relTol must be finite and non-negative, got {relative_tolerance}"
                )
            );
            assert_eq!(
                (
                    workspace.zero_initial.as_ptr(),
                    workspace.matrix_product.as_ptr(),
                    workspace.residual.as_ptr(),
                ),
                pointers
            );
            assert_eq!(workspace.matrix_product, [0.0]);
            assert_eq!(workspace.residual, [0.0]);
        }
    }

    #[test]
    fn mismatched_gamg_relative_tolerance_fails_before_workspace_mutation() {
        let matrix = CsrMatrix::from_rows(
            vec![vec![(0, 2.0), (1, -1.0)], vec![(0, -1.0), (1, 2.0)]],
            2,
        )
        .expect("symmetric matrix");
        let rhs = [1.0, 0.0];
        let gamg_options = GamgOptions {
            max_iterations: 2,
            tolerance: 1.0e-12,
            relative_tolerance: 0.25,
            n_cells_in_coarsest_level: 1,
            direct_solve_coarsest: true,
            ..GamgOptions::default()
        };
        let mut rejected_scalar_workspace = super::ScalarSolveWorkspace::new(2);
        let scalar_pointers = (
            rejected_scalar_workspace.zero_initial.as_ptr(),
            rejected_scalar_workspace.matrix_product.as_ptr(),
            rejected_scalar_workspace.residual.as_ptr(),
        );
        let mut rejected_gamg_workspace =
            GamgWorkspace::new(&matrix, gamg_options).expect("rejected workspace");
        let mut pristine_gamg_workspace =
            GamgWorkspace::new(&matrix, gamg_options).expect("pristine workspace");

        let error = super::solve_scalar_system_with_workspaces(
            &matrix,
            &rhs,
            None,
            super::ScalarSolveControls {
                solver: LaminarSimpleLinearSolver::Gamg,
                preconditioner: LaminarSimplePreconditioner::None,
                tolerance: gamg_options.tolerance,
                relative_tolerance: 0.5,
                max_iterations: gamg_options.max_iterations,
                gamg_options: Some(gamg_options),
                profile_gamg: false,
                profile_pcg: false,
            },
            &mut rejected_scalar_workspace,
            None,
            Some(&mut rejected_gamg_workspace),
        )
        .err()
        .expect("mismatched GAMG relTol sources must fail closed");
        assert_eq!(
            error.to_string(),
            "GAMG scalar solve relTol sources must match exactly, got controls=0.5 and options=0.25"
        );
        assert_eq!(
            (
                rejected_scalar_workspace.zero_initial.as_ptr(),
                rejected_scalar_workspace.matrix_product.as_ptr(),
                rejected_scalar_workspace.residual.as_ptr(),
            ),
            scalar_pointers
        );
        assert_eq!(rejected_scalar_workspace.matrix_product, [0.0, 0.0]);
        assert_eq!(rejected_scalar_workspace.residual, [0.0, 0.0]);

        let accepted_controls = super::ScalarSolveControls {
            solver: LaminarSimpleLinearSolver::Gamg,
            preconditioner: LaminarSimplePreconditioner::None,
            tolerance: gamg_options.tolerance,
            relative_tolerance: gamg_options.relative_tolerance,
            max_iterations: gamg_options.max_iterations,
            gamg_options: Some(gamg_options),
            profile_gamg: false,
            profile_pcg: false,
        };
        let after_rejection = super::solve_scalar_system_with_workspaces(
            &matrix,
            &rhs,
            None,
            accepted_controls,
            &mut rejected_scalar_workspace,
            None,
            Some(&mut rejected_gamg_workspace),
        )
        .expect("workspace rejected before mutation must remain usable");
        let mut pristine_scalar_workspace = super::ScalarSolveWorkspace::new(2);
        let pristine = super::solve_scalar_system_with_workspaces(
            &matrix,
            &rhs,
            None,
            accepted_controls,
            &mut pristine_scalar_workspace,
            None,
            Some(&mut pristine_gamg_workspace),
        )
        .expect("pristine control solve");

        assert_eq!(after_rejection.iterations, pristine.iterations);
        assert_eq!(after_rejection.termination, pristine.termination);
        assert_eq!(after_rejection.stop_reason, pristine.stop_reason);
        assert_eq!(
            after_rejection.effective_normalized_tolerance.to_bits(),
            pristine.effective_normalized_tolerance.to_bits()
        );
        for (actual, expected) in after_rejection.solution.iter().zip(&pristine.solution) {
            assert_eq!(actual.to_bits(), expected.to_bits());
        }
        assert_eq!(
            after_rejection.residual_norm.to_bits(),
            pristine.residual_norm.to_bits()
        );
        assert_eq!(
            after_rejection.normalized_residual_norm.to_bits(),
            pristine.normalized_residual_norm.to_bits()
        );
    }

    #[test]
    fn gamg_zero_relative_tolerance_wrapper_is_bit_identical_to_normalized_l1_entrypoint() {
        let matrix = CsrMatrix::from_rows(
            vec![vec![(0, 4.0), (1, -1.0)], vec![(0, -1.0), (1, 3.0)]],
            2,
        )
        .expect("symmetric matrix");
        let rhs = [1.0, 2.0];
        let initial = [0.25, -0.25];
        let gamg_options = GamgOptions {
            max_iterations: 3,
            tolerance: 1.0e-12,
            relative_tolerance: 0.0,
            n_cells_in_coarsest_level: 1,
            direct_solve_coarsest: true,
            ..GamgOptions::default()
        };
        let mut matrix_product = [0.0; 2];
        matrix
            .matvec_into(&initial, &mut matrix_product)
            .expect("initial matrix product");
        let normalization_factor =
            super::ldu_l1_residual_normalisation_factor(&matrix, &rhs, &initial, &matrix_product)
                .expect("normalization factor");
        let initial_l1 = rhs
            .iter()
            .zip(matrix_product)
            .map(|(source, product)| (source - product).abs())
            .sum::<f64>();
        let direct_controls = crate::linear::gamg::NormalizedL1GamgSolveControls {
            normalization_factor,
            tolerance: gamg_options.tolerance,
            relative_tolerance: 0.0,
            l2_controls: crate::linear::GamgSolveControls {
                max_iterations: gamg_options.max_iterations,
                min_iterations: gamg_options.min_iterations,
                tolerance: super::strict_l2_tolerance_for_l1_limit(
                    gamg_options.tolerance * normalization_factor,
                    2.0,
                ),
                relative_tolerance: 0.0,
            },
        };
        let mut direct_workspace =
            GamgWorkspace::new(&matrix, gamg_options).expect("direct workspace");
        let direct = direct_workspace
            .solve_normalized_l1_with_controls(&matrix, &rhs, Some(&initial), direct_controls)
            .expect("direct normalized-L1 solve");

        let mut wrapper_scalar_workspace = super::ScalarSolveWorkspace::new(2);
        let mut wrapper_gamg_workspace =
            GamgWorkspace::new(&matrix, gamg_options).expect("wrapper workspace");
        let wrapper = super::solve_scalar_system_with_workspaces(
            &matrix,
            &rhs,
            Some(&initial),
            super::ScalarSolveControls {
                solver: LaminarSimpleLinearSolver::Gamg,
                preconditioner: LaminarSimplePreconditioner::None,
                tolerance: gamg_options.tolerance,
                relative_tolerance: 0.0,
                max_iterations: gamg_options.max_iterations,
                gamg_options: Some(gamg_options),
                profile_gamg: false,
                profile_pcg: false,
            },
            &mut wrapper_scalar_workspace,
            None,
            Some(&mut wrapper_gamg_workspace),
        )
        .expect("wrapper normalized-L1 solve");

        assert_eq!(wrapper.iterations, direct.iterations);
        assert_eq!(wrapper.termination, direct.termination);
        assert_eq!(
            wrapper.initial_normalized_residual_norm.to_bits(),
            (initial_l1 / normalization_factor).to_bits()
        );
        assert_eq!(
            wrapper.residual_norm.to_bits(),
            direct.residual_norm.to_bits()
        );
        for (actual, expected) in wrapper.solution.iter().zip(&direct.solution) {
            assert_eq!(actual.to_bits(), expected.to_bits());
        }
        assert_eq!(
            wrapper.effective_normalized_tolerance.to_bits(),
            gamg_options.tolerance.to_bits()
        );
        assert_eq!(
            wrapper.stop_reason,
            match direct.termination {
                IterativeSolveTermination::Converged => {
                    super::LinearSolveStopReason::AbsoluteTolerance
                }
                IterativeSolveTermination::MaxIterations => {
                    super::LinearSolveStopReason::MaxIterations
                }
                IterativeSolveTermination::Breakdown => super::LinearSolveStopReason::Breakdown,
            }
        );
    }

    #[test]
    fn gamg_absolute_normalized_l1_stops_before_conservative_l2() {
        let cell_count = 8;
        let rows = vec![
            vec![(0, 5.2), (1, -0.7), (3, -0.4)],
            vec![(0, -0.7), (1, 4.7), (2, -0.6)],
            vec![(1, -0.6), (2, 4.9), (3, -0.8), (5, -0.3)],
            vec![(0, -0.4), (2, -0.8), (3, 5.4), (4, -0.5)],
            vec![(3, -0.5), (4, 4.6), (5, -0.9), (7, -0.2)],
            vec![(2, -0.3), (4, -0.9), (5, 5.1), (6, -0.6)],
            vec![(5, -0.6), (6, 4.8), (7, -0.7)],
            vec![(4, -0.2), (6, -0.7), (7, 4.3)],
        ];
        let expected = [0.3, -0.2, 0.8, 1.4, -0.7, 0.5, 1.1, -0.4];
        let initial = [0.1, 0.4, -0.3, 0.9, -0.2, 0.7, 0.2, -0.1];
        let mut rhs = vec![0.0; cell_count];
        let mut initial_product = vec![0.0; cell_count];
        let mut row_sums = vec![0.0; cell_count];
        for (row, entries) in rows.iter().enumerate() {
            for &(column, coefficient) in entries {
                rhs[row] += coefficient * expected[column];
                initial_product[row] += coefficient * initial[column];
                row_sums[row] += coefficient;
            }
        }
        let average = initial.iter().sum::<f64>() / cell_count as f64;
        let mut source_term = 0.0;
        let mut matrix_term = 0.0;
        let mut factor = 0.0;
        let mut initial_l1 = 0.0;
        let mut initial_l2_squared = 0.0;
        for row in 0..cell_count {
            let source_contribution = (rhs[row] - row_sums[row] * average).abs();
            let matrix_contribution = (initial_product[row] - row_sums[row] * average).abs();
            source_term += source_contribution;
            matrix_term += matrix_contribution;
            factor += matrix_contribution + source_contribution;
            let residual = rhs[row] - initial_product[row];
            initial_l1 += residual.abs();
            initial_l2_squared += residual * residual;
        }
        assert!(source_term > 0.0);
        assert!(matrix_term > 0.0);
        assert!(initial_l2_squared > 0.0);
        let matrix = CsrMatrix::from_rows(rows, cell_count).expect("nontrivial SPD matrix");
        let gamg_options = GamgOptions {
            max_iterations: 2,
            tolerance: initial_l1 / factor,
            relative_tolerance: 0.0,
            n_cells_in_coarsest_level: 1,
            direct_solve_coarsest: true,
            ..GamgOptions::default()
        };
        let mut scalar_workspace = super::ScalarSolveWorkspace::new(cell_count);
        let mut gamg_workspace = GamgWorkspace::new(&matrix, gamg_options).expect("GAMG workspace");

        let report = super::solve_scalar_system_with_workspaces(
            &matrix,
            &rhs,
            Some(&initial),
            super::ScalarSolveControls {
                solver: LaminarSimpleLinearSolver::Gamg,
                preconditioner: LaminarSimplePreconditioner::None,
                tolerance: gamg_options.tolerance,
                relative_tolerance: gamg_options.relative_tolerance,
                max_iterations: gamg_options.max_iterations,
                gamg_options: Some(gamg_options),
                profile_gamg: true,
                profile_pcg: false,
            },
            &mut scalar_workspace,
            None,
            Some(&mut gamg_workspace),
        )
        .expect("GAMG absolute probe");

        assert_eq!(report.iterations, 1);
        assert_eq!(report.termination, IterativeSolveTermination::Converged);
        assert_eq!(
            report.initial_normalized_residual_norm.to_bits(),
            (initial_l1 / factor).to_bits()
        );
        let mut final_l1 = 0.0;
        let mut final_l2_squared = 0.0;
        for (row, rhs_value) in rhs.iter().enumerate() {
            let mut product = 0.0;
            for entry in matrix.row_offsets()[row]..matrix.row_offsets()[row + 1] {
                product += matrix.values()[entry] * report.solution[matrix.col_indices()[entry]];
            }
            let residual = rhs_value - product;
            final_l1 += residual.abs();
            final_l2_squared += residual * residual;
        }
        let boundary = final_l1 / factor;
        assert_eq!(
            report.normalized_residual_norm.to_bits(),
            boundary.to_bits()
        );
        assert_eq!(
            report.residual_norm.to_bits(),
            final_l2_squared.sqrt().to_bits()
        );
        let profile = report.gamg_timing.as_ref().expect("GAMG profile");
        assert_eq!(profile.solves, 1);
        assert_eq!(profile.v_cycles, report.iterations);
        assert_eq!(profile.finest_residual_evaluations, 2);
        assert!(report.pcg_timing.is_none());

        let accepted_options = GamgOptions {
            tolerance: boundary.next_up(),
            max_iterations: 2,
            ..gamg_options
        };
        let mut accepted_scalar_workspace = super::ScalarSolveWorkspace::new(cell_count);
        let mut accepted_gamg_workspace =
            GamgWorkspace::new(&matrix, accepted_options).expect("accepted GAMG workspace");
        let accepted = super::solve_scalar_system_with_workspaces(
            &matrix,
            &rhs,
            Some(&initial),
            super::ScalarSolveControls {
                solver: LaminarSimpleLinearSolver::Gamg,
                preconditioner: LaminarSimplePreconditioner::None,
                tolerance: accepted_options.tolerance,
                relative_tolerance: accepted_options.relative_tolerance,
                max_iterations: accepted_options.max_iterations,
                gamg_options: Some(accepted_options),
                profile_gamg: true,
                profile_pcg: false,
            },
            &mut accepted_scalar_workspace,
            None,
            Some(&mut accepted_gamg_workspace),
        )
        .expect("accepted GAMG absolute solve");
        assert_eq!(accepted.termination, IterativeSolveTermination::Converged);
        assert_eq!(accepted.iterations, 1);
        assert_eq!(accepted.solution.len(), report.solution.len());
        for (accepted_value, probe_value) in accepted.solution.iter().zip(&report.solution) {
            assert_eq!(accepted_value.to_bits(), probe_value.to_bits());
        }
        assert_eq!(
            accepted.residual_norm.to_bits(),
            report.residual_norm.to_bits()
        );
        let accepted_timing = accepted.gamg_timing.as_ref().expect("accepted timing");
        assert_eq!(
            (
                accepted_timing.hierarchy_builds,
                accepted_timing.hierarchy_rebuilds,
                accepted_timing.matrix_refreshes,
                accepted_timing.finest_residual_evaluations,
                accepted_timing.solves,
                accepted_timing.v_cycles,
            ),
            (0, 0, 1, 2, 1, 1)
        );
        assert_eq!(
            accepted_timing
                .levels
                .iter()
                .map(|level| (level.level, level.cells, level.nonzeros))
                .collect::<Vec<_>>(),
            [(0, 8, 28), (1, 4, 10), (2, 2, 4), (3, 1, 1)]
        );
        for level in &accepted_timing.levels {
            let expected = match level.level {
                0 => (1, 1, 1, 1, 2, 1, 0, 1, 0),
                1 => (1, 1, 1, 1, 2, 1, 0, 1, 0),
                2 => (1, 1, 1, 2, 4, 0, 1, 1, 0),
                3 => (1, 0, 0, 0, 0, 0, 0, 0, 1),
                _ => unreachable!("literal four-level hierarchy"),
            };
            assert_eq!(
                (
                    level.matrix_refreshes,
                    level.restriction_calls,
                    level.prolongation_calls,
                    level.smoothing_calls,
                    level.smoothing_sweeps,
                    level.scaling_calls,
                    level.residual_evaluations,
                    level.correction_updates,
                    level.coarsest_solves,
                ),
                expected
            );
            assert!(
                level.matrix_refresh_seconds.is_finite() && level.matrix_refresh_seconds >= 0.0
            );
            for seconds in [
                level.restriction_seconds,
                level.prolongation_seconds,
                level.smoothing_seconds,
                level.scaling_seconds,
                level.residual_seconds,
                level.correction_seconds,
                level.coarsest_solve_seconds,
            ] {
                assert!(seconds.is_finite() && seconds >= 0.0);
            }
        }
        for seconds in [
            accepted_timing.total_seconds,
            accepted_timing.hierarchy_build_seconds,
            accepted_timing.hierarchy_rebuild_seconds,
            accepted_timing.matrix_refresh_seconds,
            accepted_timing.finest_residual_seconds,
            accepted_timing.v_cycle_seconds,
            accepted_timing.other_seconds,
        ] {
            assert!(seconds.is_finite() && seconds >= 0.0);
        }
        assert!(accepted.pcg_timing.is_none());
        let equality_options = GamgOptions {
            tolerance: boundary,
            ..accepted_options
        };
        let mut equality_scalar_workspace = super::ScalarSolveWorkspace::new(cell_count);
        let mut equality_gamg_workspace =
            GamgWorkspace::new(&matrix, equality_options).expect("equality GAMG workspace");
        let equality = super::solve_scalar_system_with_workspaces(
            &matrix,
            &rhs,
            Some(&initial),
            super::ScalarSolveControls {
                solver: LaminarSimpleLinearSolver::Gamg,
                preconditioner: LaminarSimplePreconditioner::None,
                tolerance: equality_options.tolerance,
                relative_tolerance: equality_options.relative_tolerance,
                max_iterations: equality_options.max_iterations,
                gamg_options: Some(equality_options),
                profile_gamg: true,
                profile_pcg: false,
            },
            &mut equality_scalar_workspace,
            None,
            Some(&mut equality_gamg_workspace),
        )
        .expect("equality GAMG absolute solve");
        assert_eq!(equality.iterations, 2);
        assert_eq!(equality.termination, IterativeSolveTermination::Converged);
        let equality_timing = equality.gamg_timing.expect("equality timing");
        assert_eq!(equality_timing.finest_residual_evaluations, 3);

        let legacy_limit =
            (accepted_options.tolerance * factor / (cell_count as f64).sqrt()).next_down();
        let mut legacy_workspace =
            GamgWorkspace::new(&matrix, gamg_options).expect("legacy GAMG workspace");
        let legacy = legacy_workspace
            .solve_with_controls_profiled(
                &matrix,
                &rhs,
                Some(&initial),
                crate::linear::gamg::GamgSolveControls {
                    max_iterations: 2,
                    min_iterations: 0,
                    tolerance: legacy_limit,
                    relative_tolerance: 0.0,
                },
            )
            .expect("legacy L2 solve");
        assert_eq!(legacy.report.iterations, 2);
        assert_eq!(
            legacy.report.termination,
            IterativeSolveTermination::Converged
        );
        assert_eq!(legacy.timing.finest_residual_evaluations, 3);
        for (equality_value, legacy_value) in equality.solution.iter().zip(&legacy.report.solution)
        {
            assert_eq!(equality_value.to_bits(), legacy_value.to_bits());
        }
        assert_eq!(
            equality.residual_norm.to_bits(),
            legacy.report.residual_norm.to_bits()
        );
    }

    #[test]
    fn gamg_relative_normalized_l1_stops_before_conservative_l2() {
        let rows = vec![
            vec![(0, 4.0), (1, -1.0), (3, -0.5)],
            vec![(0, -1.0), (1, 4.5), (2, -0.8)],
            vec![(1, -0.8), (2, 3.8), (3, -0.7)],
            vec![(0, -0.5), (2, -0.7), (3, 4.2), (4, -0.9)],
            vec![(3, -0.9), (4, 4.1), (5, -0.6)],
            vec![(4, -0.6), (5, 3.9), (6, -0.8)],
            vec![(5, -0.8), (6, 4.4), (7, -0.7)],
            vec![(6, -0.7), (7, 3.7)],
        ];
        let matrix = CsrMatrix::from_rows(rows, 8).expect("nontrivial SPD matrix");
        let reference = [0.7, -0.4, 1.2, 0.3, -0.8, 1.5, 0.6, -0.2];
        let initial = [-0.1, 0.5, 0.2, -0.6, 0.9, 0.1, -0.3, 0.4];
        let mut rhs = [0.0; 8];
        let mut initial_product = [0.0; 8];
        let mut row_sums = [0.0; 8];
        for (row, rhs_value) in rhs.iter_mut().enumerate() {
            for entry in matrix.row_offsets()[row]..matrix.row_offsets()[row + 1] {
                let column = matrix.col_indices()[entry];
                let coefficient = matrix.values()[entry];
                *rhs_value += coefficient * reference[column];
                initial_product[row] += coefficient * initial[column];
                row_sums[row] += coefficient;
            }
        }
        let average = initial.iter().sum::<f64>() / 8.0;
        let mut source_term = 0.0;
        let mut matrix_term = 0.0;
        let mut factor = 0.0;
        let mut initial_l1 = 0.0;
        let mut initial_l2_squared = 0.0;
        for row in 0..8 {
            let source_contribution = (rhs[row] - row_sums[row] * average).abs();
            let matrix_contribution = (initial_product[row] - row_sums[row] * average).abs();
            source_term += source_contribution;
            matrix_term += matrix_contribution;
            factor += matrix_contribution + source_contribution;
            let residual = rhs[row] - initial_product[row];
            initial_l1 += residual.abs();
            initial_l2_squared += residual * residual;
        }
        assert!(source_term > 0.0);
        assert!(matrix_term > 0.0);
        assert!(initial_l2_squared > 0.0);
        let probe_options = GamgOptions {
            max_iterations: 2,
            tolerance: f64::MIN_POSITIVE,
            relative_tolerance: 1.0,
            n_cells_in_coarsest_level: 1,
            direct_solve_coarsest: true,
            ..GamgOptions::default()
        };
        let mut probe_scalar_workspace = super::ScalarSolveWorkspace::new(8);
        let mut probe_gamg_workspace =
            GamgWorkspace::new(&matrix, probe_options).expect("probe GAMG workspace");
        let probe = super::solve_scalar_system_with_workspaces(
            &matrix,
            &rhs,
            Some(&initial),
            super::ScalarSolveControls {
                solver: LaminarSimpleLinearSolver::Gamg,
                preconditioner: LaminarSimplePreconditioner::None,
                tolerance: probe_options.tolerance,
                relative_tolerance: probe_options.relative_tolerance,
                max_iterations: probe_options.max_iterations,
                gamg_options: Some(probe_options),
                profile_gamg: false,
                profile_pcg: false,
            },
            &mut probe_scalar_workspace,
            None,
            Some(&mut probe_gamg_workspace),
        )
        .expect("relative probe");
        assert_eq!(probe.iterations, 1);
        assert_eq!(probe.termination, IterativeSolveTermination::Converged);
        assert_eq!(
            probe.initial_normalized_residual_norm.to_bits(),
            (initial_l1 / factor).to_bits()
        );
        let mut final_l1 = 0.0;
        let mut final_l2_squared = 0.0;
        for (row, rhs_value) in rhs.iter().enumerate() {
            let mut product = 0.0;
            for entry in matrix.row_offsets()[row]..matrix.row_offsets()[row + 1] {
                product += matrix.values()[entry] * probe.solution[matrix.col_indices()[entry]];
            }
            let residual = rhs_value - product;
            final_l1 += residual.abs();
            final_l2_squared += residual * residual;
        }
        let relative_boundary = (final_l1 / factor) / (initial_l1 / factor);
        assert_eq!(
            probe.normalized_residual_norm.to_bits(),
            (final_l1 / factor).to_bits()
        );
        assert_eq!(
            probe.residual_norm.to_bits(),
            final_l2_squared.sqrt().to_bits()
        );

        let accepted_options = GamgOptions {
            max_iterations: 2,
            relative_tolerance: relative_boundary.next_up(),
            ..probe_options
        };
        let mut accepted_scalar_workspace = super::ScalarSolveWorkspace::new(8);
        let mut accepted_gamg_workspace =
            GamgWorkspace::new(&matrix, accepted_options).expect("accepted GAMG workspace");
        let accepted = super::solve_scalar_system_with_workspaces(
            &matrix,
            &rhs,
            Some(&initial),
            super::ScalarSolveControls {
                solver: LaminarSimpleLinearSolver::Gamg,
                preconditioner: LaminarSimplePreconditioner::None,
                tolerance: accepted_options.tolerance,
                relative_tolerance: accepted_options.relative_tolerance,
                max_iterations: accepted_options.max_iterations,
                gamg_options: Some(accepted_options),
                profile_gamg: true,
                profile_pcg: false,
            },
            &mut accepted_scalar_workspace,
            None,
            Some(&mut accepted_gamg_workspace),
        )
        .expect("accepted GAMG relative solve");
        assert_eq!(accepted.termination, IterativeSolveTermination::Converged);
        assert_eq!(accepted.iterations, 1);
        for (accepted_value, probe_value) in accepted.solution.iter().zip(&probe.solution) {
            assert_eq!(accepted_value.to_bits(), probe_value.to_bits());
        }
        assert_eq!(
            accepted.residual_norm.to_bits(),
            probe.residual_norm.to_bits()
        );
        let timing = accepted.gamg_timing.expect("profiled GAMG timing");
        assert_eq!(timing.solves, 1);
        assert_eq!(timing.v_cycles, 1);
        assert_eq!(timing.finest_residual_evaluations, 2);
        assert_eq!(
            (
                timing.hierarchy_builds,
                timing.hierarchy_rebuilds,
                timing.matrix_refreshes,
            ),
            (0, 0, 1)
        );
        assert_eq!(
            timing
                .levels
                .iter()
                .map(|level| (level.level, level.cells, level.nonzeros))
                .collect::<Vec<_>>(),
            [(0, 8, 24), (1, 4, 10), (2, 2, 4), (3, 1, 1)]
        );
        for level in &timing.levels {
            let expected = match level.level {
                0 => (1, 1, 1, 1, 2, 1, 0, 1, 0),
                1 => (1, 1, 1, 1, 2, 1, 0, 1, 0),
                2 => (1, 1, 1, 2, 4, 0, 1, 1, 0),
                3 => (1, 0, 0, 0, 0, 0, 0, 0, 1),
                _ => unreachable!("literal four-level hierarchy"),
            };
            assert_eq!(
                (
                    level.matrix_refreshes,
                    level.restriction_calls,
                    level.prolongation_calls,
                    level.smoothing_calls,
                    level.smoothing_sweeps,
                    level.scaling_calls,
                    level.residual_evaluations,
                    level.correction_updates,
                    level.coarsest_solves,
                ),
                expected
            );
            for seconds in [
                level.matrix_refresh_seconds,
                level.restriction_seconds,
                level.prolongation_seconds,
                level.smoothing_seconds,
                level.scaling_seconds,
                level.residual_seconds,
                level.correction_seconds,
                level.coarsest_solve_seconds,
            ] {
                assert!(seconds.is_finite() && seconds >= 0.0);
            }
        }
        for seconds in [
            timing.total_seconds,
            timing.hierarchy_build_seconds,
            timing.hierarchy_rebuild_seconds,
            timing.matrix_refresh_seconds,
            timing.finest_residual_seconds,
            timing.v_cycle_seconds,
            timing.other_seconds,
        ] {
            assert!(seconds.is_finite() && seconds >= 0.0);
        }
        assert!(accepted.pcg_timing.is_none());

        let equality_options = GamgOptions {
            relative_tolerance: relative_boundary,
            ..accepted_options
        };
        let mut equality_scalar_workspace = super::ScalarSolveWorkspace::new(8);
        let mut equality_gamg_workspace =
            GamgWorkspace::new(&matrix, equality_options).expect("equality GAMG workspace");
        let equality = super::solve_scalar_system_with_workspaces(
            &matrix,
            &rhs,
            Some(&initial),
            super::ScalarSolveControls {
                solver: LaminarSimpleLinearSolver::Gamg,
                preconditioner: LaminarSimplePreconditioner::None,
                tolerance: equality_options.tolerance,
                relative_tolerance: equality_options.relative_tolerance,
                max_iterations: equality_options.max_iterations,
                gamg_options: Some(equality_options),
                profile_gamg: true,
                profile_pcg: false,
            },
            &mut equality_scalar_workspace,
            None,
            Some(&mut equality_gamg_workspace),
        )
        .expect("equality GAMG relative solve");
        assert_eq!(equality.iterations, 2);
        assert_eq!(equality.termination, IterativeSolveTermination::Converged);
        assert!(
            equality.normalized_residual_norm
                < relative_boundary * equality.initial_normalized_residual_norm
        );
        let equality_timing = equality.gamg_timing.expect("equality timing");
        assert_eq!(equality_timing.finest_residual_evaluations, 3);

        let legacy_absolute_limit =
            (accepted_options.tolerance * factor / 8.0_f64.sqrt()).next_down();
        let legacy_relative_limit =
            (accepted_options.relative_tolerance * initial_l1 / 8.0_f64.sqrt()).next_down();
        let legacy_limit = legacy_absolute_limit.max(legacy_relative_limit);
        let mut legacy_workspace =
            GamgWorkspace::new(&matrix, probe_options).expect("legacy GAMG workspace");
        let legacy = legacy_workspace
            .solve_with_controls_profiled(
                &matrix,
                &rhs,
                Some(&initial),
                crate::linear::gamg::GamgSolveControls {
                    max_iterations: 2,
                    min_iterations: 0,
                    tolerance: legacy_limit,
                    relative_tolerance: 0.0,
                },
            )
            .expect("legacy L2 solve");
        assert_eq!(legacy.report.iterations, 2);
        assert_eq!(
            legacy.report.termination,
            IterativeSolveTermination::Converged
        );
        assert_eq!(legacy.timing.finest_residual_evaluations, 3);
        for (equality_value, legacy_value) in equality.solution.iter().zip(&legacy.report.solution)
        {
            assert_eq!(equality_value.to_bits(), legacy_value.to_bits());
        }
        assert_eq!(
            equality.residual_norm.to_bits(),
            legacy.report.residual_norm.to_bits()
        );
    }

    #[test]
    fn scalar_solve_workspace_reuses_outer_residual_buffers() {
        let matrix =
            CsrMatrix::from_rows(vec![vec![(0, 2.0)], vec![(1, 4.0)]], 2).expect("diagonal matrix");
        let controls = super::ScalarSolveControls {
            solver: LaminarSimpleLinearSolver::SymGaussSeidel,
            preconditioner: LaminarSimplePreconditioner::None,
            tolerance: 1.0e-12,
            relative_tolerance: 0.0,
            max_iterations: 4,
            gamg_options: None,
            profile_gamg: false,
            profile_pcg: false,
        };
        let mut workspace = super::ScalarSolveWorkspace::new(2);
        let zero_initial_ptr = workspace.zero_initial.as_ptr();
        let matrix_product_ptr = workspace.matrix_product.as_ptr();
        let residual_ptr = workspace.residual.as_ptr();

        let first = super::solve_scalar_system_with_workspaces(
            &matrix,
            &[2.0, 8.0],
            None,
            controls,
            &mut workspace,
            None,
            None,
        )
        .expect("first scalar solve");
        let second = super::solve_scalar_system_with_workspaces(
            &matrix,
            &[4.0, 4.0],
            None,
            controls,
            &mut workspace,
            None,
            None,
        )
        .expect("second scalar solve");

        assert_eq!(first.solution, [1.0, 2.0]);
        assert_eq!(second.solution, [2.0, 1.0]);
        assert_eq!(first.initial_normalized_residual_norm, 1.0);
        assert_eq!(second.initial_normalized_residual_norm, 1.0);
        assert_eq!(first.normalized_residual_norm, 0.0);
        assert_eq!(second.normalized_residual_norm, 0.0);
        assert_eq!(workspace.zero_initial.as_ptr(), zero_initial_ptr);
        assert_eq!(workspace.matrix_product.as_ptr(), matrix_product_ptr);
        assert_eq!(workspace.residual.as_ptr(), residual_ptr);
    }

    #[test]
    fn scalar_solve_preserves_breakdown_and_max_iterations_termination() {
        let controls = |max_iterations| super::ScalarSolveControls {
            solver: LaminarSimpleLinearSolver::Cg,
            preconditioner: LaminarSimplePreconditioner::None,
            tolerance: 1.0e-12,
            relative_tolerance: 0.0,
            max_iterations,
            gamg_options: None,
            profile_gamg: false,
            profile_pcg: false,
        };

        let zero_matrix = CsrMatrix::from_rows(vec![Vec::new()], 1).expect("zero matrix");
        let mut workspace = super::ScalarSolveWorkspace::new(1);
        let breakdown = super::solve_scalar_system_with_workspaces(
            &zero_matrix,
            &[1.0],
            None,
            controls(1),
            &mut workspace,
            None,
            None,
        )
        .expect("finite singular solve report");
        assert_eq!(breakdown.termination, IterativeSolveTermination::Breakdown);
        assert!(!breakdown.converged);

        let identity = CsrMatrix::from_rows(vec![vec![(0, 1.0)]], 1).expect("identity matrix");
        let exhausted = super::solve_scalar_system_with_workspaces(
            &identity,
            &[1.0],
            None,
            controls(0),
            &mut workspace,
            None,
            None,
        )
        .expect("budget-exhausted solve report");
        assert_eq!(
            exhausted.termination,
            IterativeSolveTermination::MaxIterations
        );
        assert!(!exhausted.converged);
    }

    #[test]
    fn pressure_path_rejects_breakdown_but_not_iteration_exhaustion() {
        let mut runtime = two_cell_runtime();
        let initial_pressure = runtime.fields[1]
            .values
            .clone()
            .expect("initial pressure payload");
        let fields = two_cell_fields();
        let initial_velocity = [point(1.0, 0.0, 0.0), point(1.0, 0.0, 0.0)];
        let velocity_boundary = super::vector_face_treatments(&runtime.mesh, &fields.fields[0])
            .expect("velocity boundary");
        let initial_phi = compute_face_flux(&runtime.mesh, &initial_velocity, &velocity_boundary)
            .expect("initial phi");
        let initial_continuity =
            super::summarize_continuity(&net_cell_flux(&runtime.mesh, &initial_phi).unwrap());
        let mut options = minimal_laminar_options();
        options.pressure_linear_solver = LaminarSimpleLinearSolver::Pcg;
        options.non_orthogonal_correctors = 1;

        let mut attempted_pressure_solves = 0;
        let breakdown = {
            let mut drive_later_breakdown =
                |solve: usize,
                 report: &mut super::ScalarSolveReport,
                 _predicted_velocity: &[Point3],
                 _phi_hby_a: &[f64],
                 _continuity_star: super::ContinuitySummary| {
                    attempted_pressure_solves += 1;
                    if solve == 2 {
                        report.converged = false;
                        report.termination = IterativeSolveTermination::Breakdown;
                    }
                };
            super::solve_laminar_simple_driven(
                &mut runtime,
                &fields,
                &options,
                None,
                true,
                Some(&mut drive_later_breakdown),
                None,
            )
            .expect("driven pressure-breakdown report")
        };

        assert_eq!(attempted_pressure_solves, 2);
        assert_eq!(
            breakdown.stop_reason,
            LaminarSimpleStopReason::PressureSolverInvalidState
        );
        assert_eq!(breakdown.simple_iterations, 1);
        assert_eq!(breakdown.final_pressure, initial_pressure);
        assert_eq!(breakdown.final_velocity.len(), initial_velocity.len());
        for (actual, expected) in breakdown.final_velocity.iter().zip(&initial_velocity) {
            assert_eq!(
                (actual.x, actual.y, actual.z),
                (expected.x, expected.y, expected.z)
            );
        }
        assert_eq!(breakdown.final_phi, initial_phi);
        assert_eq!(
            breakdown.final_continuity.l2_norm,
            initial_continuity.l2_norm
        );
        assert_eq!(
            breakdown.final_continuity.max_abs,
            initial_continuity.max_abs
        );
        assert_eq!(
            breakdown.final_continuity.sum_abs,
            initial_continuity.sum_abs
        );
        assert_eq!(
            breakdown.final_continuity.global_sum,
            initial_continuity.global_sum
        );
        let rejected = breakdown
            .history
            .last()
            .expect("rejected iteration summary");
        assert!(!rejected.pressure_correction_accepted);
        assert_eq!(rejected.pressure_linear_solves, 2);
        assert_eq!(rejected.pressure_linear_non_converged_solves, 1);
        assert!(rejected.pressure_linear_iterations > 0);
        assert!(
            rejected
                .pressure_correction_initial_normalized_residual_norm
                .is_finite()
        );
        assert!(rejected.pressure_correction_residual_norm.is_finite());
        assert!(
            rejected
                .pressure_correction_normalized_residual_norm
                .is_finite()
        );
        assert_eq!(
            rejected.continuity_after.l2_norm,
            initial_continuity.l2_norm
        );
        assert!(breakdown.timing.pressure_matrix_vector_products > 0);

        let mut drive_exhaustion =
            |solve: usize,
             report: &mut super::ScalarSolveReport,
             _predicted_velocity: &[Point3],
             _phi_hby_a: &[f64],
             _continuity_star: super::ContinuitySummary| {
                if solve == 2 {
                    report.converged = false;
                    report.termination = IterativeSolveTermination::MaxIterations;
                }
            };
        let mut exhaustion_runtime = two_cell_runtime();
        let exhausted = super::solve_laminar_simple_driven(
            &mut exhaustion_runtime,
            &fields,
            &options,
            None,
            false,
            Some(&mut drive_exhaustion),
            None,
        )
        .expect("driven iteration-budget-exhausted SIMPLE report");
        assert_ne!(
            exhausted.stop_reason,
            LaminarSimpleStopReason::PressureSolverInvalidState
        );
        assert_eq!(exhausted.simple_iterations, options.max_simple_iterations);
        assert!(
            exhausted
                .linear_solve_summary
                .pressure_correction_non_converged_solves
                > 0
        );
    }

    #[test]
    fn simple_scalar_path_requires_explicit_gamg_workspace_without_pcg_fallback() {
        let matrix = CsrMatrix::from_rows(
            vec![vec![(0, 2.0), (1, -1.0)], vec![(0, -1.0), (1, 2.0)]],
            2,
        )
        .expect("symmetric test matrix");
        let controls = super::ScalarSolveControls {
            solver: LaminarSimpleLinearSolver::Gamg,
            preconditioner: LaminarSimplePreconditioner::None,
            tolerance: 1.0e-10,
            relative_tolerance: 0.0,
            max_iterations: 100,
            gamg_options: Some(GamgOptions {
                n_cells_in_coarsest_level: 1,
                ..GamgOptions::default()
            }),
            profile_gamg: false,
            profile_pcg: false,
        };
        let mut workspace = super::ScalarSolveWorkspace::new(2);

        let error = super::solve_scalar_system_with_workspaces(
            &matrix,
            &[1.0, 1.0],
            None,
            controls,
            &mut workspace,
            None,
            None,
        )
        .err()
        .expect("GAMG without a workspace must fail");

        assert!(error.to_string().contains("requires a matching hierarchy"));
    }

    fn momentum_component_test_systems() -> Vec<ScalarComponentSystem> {
        (0..3)
            .map(|component| ScalarComponentSystem {
                matrix: CsrMatrix::from_rows(
                    vec![vec![(0, 4.0), (1, -1.0)], vec![(0, -1.0), (1, 3.0)]],
                    2,
                )
                .expect("two-cell momentum component matrix"),
                rhs: vec![1.0 + component as f64, 2.0 - 0.25 * component as f64],
            })
            .collect()
    }

    #[test]
    fn momentum_execution_defaults_to_serial_and_displays_exactly() {
        assert_eq!(
            LaminarSimpleMomentumExecution::default(),
            LaminarSimpleMomentumExecution::Serial
        );
        assert_eq!(LaminarSimpleMomentumExecution::Serial.to_string(), "serial");
        assert_eq!(
            LaminarSimpleMomentumExecution::ParallelComponents.to_string(),
            "parallel-components"
        );
    }

    fn momentum_component_test_initials() -> [Vec<f64>; 3] {
        [vec![0.125, -0.25], vec![-0.375, 0.5], vec![0.625, -0.75]]
    }

    fn momentum_component_test_controls(
        solver: LaminarSimpleLinearSolver,
        preconditioner: LaminarSimplePreconditioner,
    ) -> ScalarSolveControls {
        ScalarSolveControls {
            solver,
            preconditioner,
            tolerance: 1.0e-10,
            relative_tolerance: 0.0,
            max_iterations: 100,
            gamg_options: None,
            profile_gamg: false,
            profile_pcg: false,
        }
    }

    fn assert_scalar_solve_reports_bit_equal(left: &ScalarSolveReport, right: &ScalarSolveReport) {
        assert_eq!(
            left.solution
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            right
                .solution
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
        assert_eq!(left.iterations, right.iterations);
        assert_eq!(left.converged, right.converged);
        assert_eq!(left.termination, right.termination);
        assert_eq!(
            left.initial_normalized_residual_norm.to_bits(),
            right.initial_normalized_residual_norm.to_bits()
        );
        assert_eq!(left.residual_norm.to_bits(), right.residual_norm.to_bits());
        assert_eq!(
            left.normalized_residual_norm.to_bits(),
            right.normalized_residual_norm.to_bits()
        );
        assert_eq!(
            left.effective_normalized_tolerance.to_bits(),
            right.effective_normalized_tolerance.to_bits()
        );
        assert_eq!(left.stop_reason, right.stop_reason);
        assert_eq!(left.pcg_timing.is_some(), right.pcg_timing.is_some());
        assert_eq!(left.gamg_timing.is_some(), right.gamg_timing.is_some());
    }

    #[test]
    fn parallel_momentum_components_reuse_three_named_worker_workspaces_for_ten_lifecycles() {
        let mut executor =
            MomentumComponentExecutor::new(2).expect("three component workers must start");
        assert_eq!(executor.command_senders.len(), 3);
        assert_eq!(
            executor
                .handles
                .iter()
                .map(|handle| {
                    handle
                        .as_ref()
                        .and_then(|handle| handle.thread().name())
                        .expect("named momentum worker")
                        .to_string()
                })
                .collect::<Vec<_>>(),
            [
                "ferrum-momentum-X",
                "ferrum-momentum-Y",
                "ferrum-momentum-Z"
            ]
        );
        let controls = momentum_component_test_controls(
            LaminarSimpleLinearSolver::Cg,
            LaminarSimplePreconditioner::None,
        );
        let mut serial_workspace = ScalarSolveWorkspace::new(2);

        for _ in 0..10 {
            let systems = momentum_component_test_systems();
            let initials = momentum_component_test_initials();
            let serial = solve_momentum_components_serial(
                &systems,
                &initials,
                controls,
                &mut serial_workspace,
            )
            .expect("serial component oracle");
            let parallel = executor
                .solve(systems, initials, controls)
                .expect("parallel component solve");
            for component in 0..3 {
                assert_scalar_solve_reports_bit_equal(&serial[component], &parallel[component]);
            }
        }

        for observations in &executor.workspace_observations {
            assert_eq!(observations.len(), 10);
            assert!(
                observations
                    .iter()
                    .all(|identity| *identity == observations[0])
            );
        }
        executor.shutdown().expect("deterministic worker shutdown");
        assert!(executor.finished);
        assert!(executor.handles.iter().all(Option::is_none));
    }

    #[test]
    fn momentum_worker_results_store_out_of_order_but_select_errors_in_component_order() {
        let mut outcomes = [None, None, None];
        store_momentum_component_worker_result(
            &mut outcomes,
            2,
            Err(MomentumComponentWorkerFailure::Solve(super::invalid_input(
                "Z failure".to_string(),
            ))),
        )
        .expect("Z arrival");
        store_momentum_component_worker_result(
            &mut outcomes,
            0,
            Err(MomentumComponentWorkerFailure::Solve(super::invalid_input(
                "X failure".to_string(),
            ))),
        )
        .expect("X arrival");
        store_momentum_component_worker_result(
            &mut outcomes,
            1,
            Err(MomentumComponentWorkerFailure::Panicked),
        )
        .expect("Y arrival");
        assert!(outcomes.iter().all(Option::is_some));

        let duplicate = store_momentum_component_worker_result(
            &mut outcomes,
            0,
            Err(MomentumComponentWorkerFailure::Panicked),
        )
        .expect_err("duplicate X result must fail");
        assert_eq!(
            duplicate.to_string(),
            "laminar SIMPLE momentum component Ux worker returned a duplicate result"
        );
        let unexpected = store_momentum_component_worker_result(
            &mut outcomes,
            3,
            Err(MomentumComponentWorkerFailure::Panicked),
        )
        .expect_err("unexpected component must fail");
        assert_eq!(
            unexpected.to_string(),
            "laminar SIMPLE momentum worker returned unexpected component index 3"
        );
    }

    #[test]
    fn parallel_momentum_components_prioritize_x_error_before_y_panic_and_z_error() {
        let mut executor = MomentumComponentExecutor::new(2).expect("component executor");
        executor.set_test_actions([
            MomentumComponentTestAction::Error("injected X error"),
            MomentumComponentTestAction::Panic,
            MomentumComponentTestAction::Error("injected Z error"),
        ]);
        let error = executor
            .solve(
                momentum_component_test_systems(),
                momentum_component_test_initials(),
                momentum_component_test_controls(
                    LaminarSimpleLinearSolver::Cg,
                    LaminarSimplePreconditioner::None,
                ),
            )
            .err()
            .expect("injected component failures must fail");
        assert_eq!(
            error.to_string(),
            "laminar SIMPLE momentum component Ux solve failed: injected X error"
        );
    }

    #[test]
    fn parallel_momentum_components_report_y_panic_before_z_error() {
        let mut executor = MomentumComponentExecutor::new(2).expect("component executor");
        executor.set_test_actions([
            MomentumComponentTestAction::Solve,
            MomentumComponentTestAction::Panic,
            MomentumComponentTestAction::Error("injected Z error"),
        ]);
        let error = executor
            .solve(
                momentum_component_test_systems(),
                momentum_component_test_initials(),
                momentum_component_test_controls(
                    LaminarSimpleLinearSolver::Cg,
                    LaminarSimplePreconditioner::None,
                ),
            )
            .err()
            .expect("injected component panic must fail");
        assert_eq!(
            error.to_string(),
            "laminar SIMPLE momentum component Uy worker panicked"
        );
    }

    #[test]
    fn parallel_momentum_components_match_serial_for_all_supported_solver_preconditioner_pairs() {
        let combinations = [
            (
                LaminarSimpleLinearSolver::BiCgStab,
                LaminarSimplePreconditioner::None,
            ),
            (
                LaminarSimpleLinearSolver::Cg,
                LaminarSimplePreconditioner::None,
            ),
            (
                LaminarSimpleLinearSolver::GaussSeidel,
                LaminarSimplePreconditioner::None,
            ),
            (
                LaminarSimpleLinearSolver::SymGaussSeidel,
                LaminarSimplePreconditioner::None,
            ),
            (
                LaminarSimpleLinearSolver::Jacobi,
                LaminarSimplePreconditioner::None,
            ),
            (
                LaminarSimpleLinearSolver::Pcg,
                LaminarSimplePreconditioner::None,
            ),
            (
                LaminarSimpleLinearSolver::Pcg,
                LaminarSimplePreconditioner::Diagonal,
            ),
            (
                LaminarSimpleLinearSolver::Pcg,
                LaminarSimplePreconditioner::IncompleteCholesky,
            ),
            (
                LaminarSimpleLinearSolver::Pcg,
                LaminarSimplePreconditioner::Dic,
            ),
            (
                LaminarSimpleLinearSolver::Pcg,
                LaminarSimplePreconditioner::Fdic,
            ),
        ];

        for (solver, preconditioner) in combinations {
            let systems = momentum_component_test_systems();
            let initials = momentum_component_test_initials();
            let controls = momentum_component_test_controls(solver, preconditioner);
            let mut serial_workspace = ScalarSolveWorkspace::new(2);
            let serial = solve_momentum_components_serial(
                &systems,
                &initials,
                controls,
                &mut serial_workspace,
            )
            .unwrap_or_else(|error| {
                panic!("{solver}/{preconditioner} serial solve failed: {error}")
            });
            let mut executor = MomentumComponentExecutor::new(2).expect("component executor");
            let parallel = executor
                .solve(systems, initials, controls)
                .unwrap_or_else(|error| {
                    panic!("{solver}/{preconditioner} parallel solve failed: {error}")
                });
            for component in 0..3 {
                assert_scalar_solve_reports_bit_equal(&serial[component], &parallel[component]);
            }
            executor.shutdown().expect("component executor shutdown");
        }
    }

    fn assert_public_serial_parallel_bit_parity(mut options: LaminarSimpleOptions) {
        let fields = two_cell_fields();
        let mut serial_runtime = two_cell_runtime();
        let mut parallel_runtime = serial_runtime.clone();

        options.momentum_execution = LaminarSimpleMomentumExecution::Serial;
        let mut serial = solve_laminar_simple(&mut serial_runtime, &fields, &options)
            .expect("serial public SIMPLE solve");
        options.momentum_execution = LaminarSimpleMomentumExecution::ParallelComponents;
        let mut parallel = solve_laminar_simple(&mut parallel_runtime, &fields, &options)
            .expect("parallel public SIMPLE solve");

        serial.timing = super::LaminarSimpleTimingSummary::default();
        parallel.timing = super::LaminarSimpleTimingSummary::default();
        assert_eq!(format!("{serial:?}"), format!("{parallel:?}"));
        assert_eq!(
            format!("{serial_runtime:?}"),
            format!("{parallel_runtime:?}")
        );
    }

    #[test]
    fn public_parallel_momentum_matches_serial_across_schemes_and_skewed_geometry() {
        for scheme in [
            LaminarSimpleConvectionScheme::GaussUpwind,
            LaminarSimpleConvectionScheme::GaussLinearUpwind,
            LaminarSimpleConvectionScheme::BoundedGaussLinearUpwind(
                LaminarSimpleGradientScheme::GaussLinear,
            ),
        ] {
            let mut options = minimal_laminar_options();
            options.max_simple_iterations = 2;
            options.schemes.div_phi_u = scheme;
            assert_public_serial_parallel_bit_parity(options);
        }

        let mut options = minimal_laminar_options();
        options.max_simple_iterations = 2;
        options.non_orthogonal_correctors = 1;
        let fields = two_cell_fields();
        let mut serial_runtime = two_cell_runtime();
        serial_runtime.mesh.face_centres[0].y = 0.05;
        serial_runtime.mesh.cell_centres[1].y = 0.2;
        let mut parallel_runtime = serial_runtime.clone();
        options.momentum_execution = LaminarSimpleMomentumExecution::Serial;
        let mut serial = solve_laminar_simple(&mut serial_runtime, &fields, &options)
            .expect("serial skewed SIMPLE solve");
        options.momentum_execution = LaminarSimpleMomentumExecution::ParallelComponents;
        let mut parallel = solve_laminar_simple(&mut parallel_runtime, &fields, &options)
            .expect("parallel skewed SIMPLE solve");
        serial.timing = super::LaminarSimpleTimingSummary::default();
        parallel.timing = super::LaminarSimpleTimingSummary::default();
        assert_eq!(format!("{serial:?}"), format!("{parallel:?}"));
    }

    #[test]
    fn parallel_workspace_capacity_failure_is_fallible() {
        let error = ScalarSolveWorkspace::try_new(usize::MAX)
            .err()
            .expect("capacity overflow must fail without a process abort");
        assert!(matches!(error, MeshError::OutOfMemory));
    }

    fn minimal_laminar_options() -> LaminarSimpleOptions {
        LaminarSimpleOptions {
            density: 1.0,
            dynamic_viscosity: 1.0,
            linear_solver: LaminarSimpleLinearSolver::Cg,
            momentum_linear_solver: LaminarSimpleLinearSolver::Cg,
            momentum_execution: LaminarSimpleMomentumExecution::Serial,
            pressure_linear_solver: LaminarSimpleLinearSolver::Cg,
            momentum_preconditioner: LaminarSimplePreconditioner::None,
            pressure_preconditioner: LaminarSimplePreconditioner::None,
            pressure_gamg_options: None,
            profile_gamg: false,
            linear_tolerance: 1.0e-10,
            max_linear_iterations: 100,
            momentum_linear_tolerance: 1.0e-10,
            pressure_linear_tolerance: 1.0e-10,
            momentum_linear_relative_tolerance: 0.0,
            pressure_linear_relative_tolerance: 0.0,
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

    #[test]
    fn laminar_simple_options_reject_invalid_equation_relative_tolerances() {
        for value in [-1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut momentum = minimal_laminar_options();
            momentum.momentum_linear_relative_tolerance = value;
            let momentum_error = super::validate_laminar_simple_options(&momentum)
                .expect_err("invalid momentum relTol must fail");
            assert_eq!(
                momentum_error.to_string(),
                format!(
                    "laminar SIMPLE momentum linear relTol must be non-negative and finite, got {value}"
                )
            );

            let mut pressure = minimal_laminar_options();
            pressure.pressure_linear_relative_tolerance = value;
            let pressure_error = super::validate_laminar_simple_options(&pressure)
                .expect_err("invalid pressure relTol must fail");
            assert_eq!(
                pressure_error.to_string(),
                format!(
                    "laminar SIMPLE pressure linear relTol must be non-negative and finite, got {value}"
                )
            );
        }
    }

    #[test]
    fn public_gamg_relative_tolerance_mismatch_rejects_before_runtime_mutation() {
        for momentum_execution in [
            LaminarSimpleMomentumExecution::Serial,
            LaminarSimpleMomentumExecution::ParallelComponents,
        ] {
            let mut runtime = two_cell_runtime();
            let before = runtime.clone();
            let value_pointers = runtime
                .fields
                .iter()
                .map(|field| field.values.as_ref().map(Vec::as_ptr))
                .collect::<Vec<_>>();
            let mut options = minimal_laminar_options();
            options.momentum_execution = momentum_execution;
            options.pressure_linear_solver = LaminarSimpleLinearSolver::Gamg;
            options.pressure_linear_relative_tolerance = 0.5;
            options.pressure_gamg_options = Some(GamgOptions {
                max_iterations: options.pressure_max_linear_iterations,
                tolerance: options.pressure_linear_tolerance,
                relative_tolerance: 0.25,
                n_cells_in_coarsest_level: 1,
                direct_solve_coarsest: true,
                ..GamgOptions::default()
            });

            let error = solve_laminar_simple(&mut runtime, &two_cell_fields(), &options)
                .expect_err("public GAMG relTol mismatch must fail closed");

            assert_eq!(
                error.to_string(),
                "laminar SIMPLE pressure GAMG relTol sources must match exactly, got pressure=0.5 and GAMG=0.25"
            );
            assert_eq!(format!("{runtime:?}"), format!("{before:?}"));
            assert_eq!(
                runtime
                    .fields
                    .iter()
                    .map(|field| field.values.as_ref().map(Vec::as_ptr))
                    .collect::<Vec<_>>(),
                value_pointers
            );
        }
    }

    #[test]
    fn active_relative_tolerances_are_reported_through_warm_nonorthogonal_simple_solves() {
        let mut runtime = two_cell_runtime();
        runtime.mesh.face_centres[0].y = 0.05;
        runtime.mesh.cell_centres[1].y = 0.2;
        let mut options = minimal_laminar_options();
        options.max_simple_iterations = 1;
        options.momentum_linear_relative_tolerance = 0.5;
        options.pressure_linear_relative_tolerance = 0.5;
        options.non_orthogonal_correctors = 1;

        let report = solve_laminar_simple(&mut runtime, &two_cell_fields(), &options)
            .expect("active relTol SIMPLE integration solve");

        assert_eq!(report.history.len(), 1);
        let iteration = &report.history[0];
        assert_eq!(iteration.pressure_linear_solves, 2);
        assert!(
            iteration
                .momentum_component_linear_solves
                .iter()
                .all(|solve| solve.stop_reason != super::LinearSolveStopReason::NotRun)
        );
        let pressure_solves =
            &iteration.pressure_correction_linear_solves[..iteration.pressure_linear_solves];
        assert_eq!(pressure_solves.len(), 2);
        assert!(
            pressure_solves
                .iter()
                .all(|solve| solve.stop_reason != super::LinearSolveStopReason::NotRun)
        );
        assert!(
            pressure_solves
                .iter()
                .all(|solve| solve.effective_normalized_tolerance.is_finite())
        );
        assert!(
            iteration
                .momentum_component_linear_solves
                .iter()
                .chain(pressure_solves)
                .any(|solve| solve.stop_reason == super::LinearSolveStopReason::RelativeTolerance)
        );
        assert!(
            pressure_solves[1]
                .initial_normalized_residual_norm
                .is_finite()
        );
        assert_ne!(
            pressure_solves[0]
                .initial_normalized_residual_norm
                .to_bits(),
            pressure_solves[1]
                .initial_normalized_residual_norm
                .to_bits(),
            "the second corrected pressure solve must report its warm-started system"
        );
        for solve in iteration
            .momentum_component_linear_solves
            .iter()
            .chain(pressure_solves)
        {
            let expected = options
                .momentum_linear_tolerance
                .max(0.5 * solve.initial_normalized_residual_norm);
            assert_eq!(
                solve.effective_normalized_tolerance.to_bits(),
                expected.to_bits()
            );
        }
    }

    fn four_cell_planar_runtime(skewed: bool) -> SolverRuntimeData {
        let cell_centres = if skewed {
            vec![
                point(0.24, 0.22, 0.0),
                point(0.76, 0.29, 0.0),
                point(0.18, 0.77, 0.0),
                point(0.82, 0.83, 0.0),
            ]
        } else {
            vec![
                point(0.25, 0.25, 0.0),
                point(0.75, 0.25, 0.0),
                point(0.25, 0.75, 0.0),
                point(0.75, 0.75, 0.0),
            ]
        };
        let mut face_centres = if skewed {
            vec![
                point(0.51, 0.28, 0.0),
                point(0.48, 0.79, 0.0),
                point(0.23, 0.49, 0.0),
                point(0.78, 0.56, 0.0),
                point(0.0, 0.22, 0.0),
                point(0.0, 0.77, 0.0),
                point(1.0, 0.29, 0.0),
                point(1.0, 0.83, 0.0),
                point(0.24, 0.0, 0.0),
                point(0.76, 0.0, 0.0),
                point(0.18, 1.0, 0.0),
                point(0.82, 1.0, 0.0),
            ]
        } else {
            vec![
                point(0.5, 0.25, 0.0),
                point(0.5, 0.75, 0.0),
                point(0.25, 0.5, 0.0),
                point(0.75, 0.5, 0.0),
                point(0.0, 0.25, 0.0),
                point(0.0, 0.75, 0.0),
                point(1.0, 0.25, 0.0),
                point(1.0, 0.75, 0.0),
                point(0.25, 0.0, 0.0),
                point(0.75, 0.0, 0.0),
                point(0.25, 1.0, 0.0),
                point(0.75, 1.0, 0.0),
            ]
        };
        for centre in &cell_centres {
            face_centres.push(point(centre.x, centre.y, 0.5));
            face_centres.push(point(centre.x, centre.y, -0.5));
        }

        SolverRuntimeData {
            mesh: SolverRuntimeMeshData {
                points: 0,
                cells: 4,
                faces: 20,
                internal_faces: 4,
                boundary_faces: 16,
                owner: vec![0, 2, 0, 1, 0, 2, 1, 3, 0, 1, 2, 3, 0, 0, 1, 1, 2, 2, 3, 3],
                neighbour: vec![
                    Some(1),
                    Some(3),
                    Some(2),
                    Some(3),
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                ],
                patches: vec![
                    SolverRuntimePatchRange {
                        name: "inlet".to_string(),
                        patch_type: "patch".to_string(),
                        start_face: 4,
                        faces: 2,
                    },
                    SolverRuntimePatchRange {
                        name: "outlet".to_string(),
                        patch_type: "patch".to_string(),
                        start_face: 6,
                        faces: 2,
                    },
                    SolverRuntimePatchRange {
                        name: "walls".to_string(),
                        patch_type: "wall".to_string(),
                        start_face: 8,
                        faces: 4,
                    },
                    SolverRuntimePatchRange {
                        name: "frontAndBack".to_string(),
                        patch_type: "empty".to_string(),
                        start_face: 12,
                        faces: 8,
                    },
                ],
                face_centres,
                face_area_vectors: vec![
                    point(1.0, 0.0, 0.0),
                    point(1.0, 0.0, 0.0),
                    point(0.0, 1.0, 0.0),
                    point(0.0, 1.0, 0.0),
                    point(-1.0, 0.0, 0.0),
                    point(-1.0, 0.0, 0.0),
                    point(1.0, 0.0, 0.0),
                    point(1.0, 0.0, 0.0),
                    point(0.0, -1.0, 0.0),
                    point(0.0, -1.0, 0.0),
                    point(0.0, 1.0, 0.0),
                    point(0.0, 1.0, 0.0),
                    point(0.0, 0.0, 1.0),
                    point(0.0, 0.0, -1.0),
                    point(0.0, 0.0, 1.0),
                    point(0.0, 0.0, -1.0),
                    point(0.0, 0.0, 1.0),
                    point(0.0, 0.0, -1.0),
                    point(0.0, 0.0, 1.0),
                    point(0.0, 0.0, -1.0),
                ],
                cell_centres,
                cell_volumes: vec![0.25; 4],
                min_face_area: 1.0,
                max_face_area: 1.0,
                min_cell_volume: 0.25,
                max_cell_volume: 0.25,
                total_cell_volume: 1.0,
                non_positive_cell_volumes: 0,
            },
            fields: vec![
                SolverRuntimeFieldBuffer {
                    region: None,
                    name: "U".to_string(),
                    kind: SolverStateFieldKind::VolVector,
                    components: 3,
                    scalar_slots: 12,
                    bytes_f64: 96,
                    values: Some(vec![
                        1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                    ]),
                },
                SolverRuntimeFieldBuffer {
                    region: None,
                    name: "p".to_string(),
                    kind: SolverStateFieldKind::VolScalar,
                    components: 1,
                    scalar_slots: 4,
                    bytes_f64: 32,
                    values: Some(vec![1.0, 0.0, 1.0, 0.0]),
                },
            ],
            warnings: Vec::new(),
        }
    }

    fn four_cell_planar_fields(closed: bool) -> InitialFieldSet {
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
                        FieldBoundaryPatch {
                            name: "walls".to_string(),
                            patch_type: Some("noSlip".to_string()),
                            inlet_value: None,
                            value: None,
                        },
                        FieldBoundaryPatch {
                            name: "frontAndBack".to_string(),
                            patch_type: Some("empty".to_string()),
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
                            patch_type: Some(if closed {
                                "zeroGradient".to_string()
                            } else {
                                "fixedValue".to_string()
                            }),
                            inlet_value: None,
                            value: (!closed).then(|| FieldValueSummary::Uniform("0".to_string())),
                        },
                        FieldBoundaryPatch {
                            name: "walls".to_string(),
                            patch_type: Some("zeroGradient".to_string()),
                            inlet_value: None,
                            value: None,
                        },
                        FieldBoundaryPatch {
                            name: "frontAndBack".to_string(),
                            patch_type: Some("empty".to_string()),
                            inlet_value: None,
                            value: None,
                        },
                    ],
                },
            ],
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
                    values: Some(vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
                },
                SolverRuntimeFieldBuffer {
                    region: None,
                    name: "p".to_string(),
                    kind: SolverStateFieldKind::VolScalar,
                    components: 1,
                    scalar_slots: 2,
                    bytes_f64: 16,
                    values: Some(vec![1.0, 0.0]),
                },
            ],
            warnings: Vec::new(),
        }
    }

    fn two_cell_runtime_with_empty_sides() -> SolverRuntimeData {
        let mut runtime = two_cell_runtime();
        runtime.mesh.faces = 11;
        runtime.mesh.boundary_faces = 10;
        runtime.mesh.owner.extend([0, 0, 0, 0, 1, 1, 1, 1]);
        runtime.mesh.neighbour.extend([None; 8]);
        runtime.mesh.face_centres.extend([
            point(0.25, 0.5, 0.0),
            point(0.25, -0.5, 0.0),
            point(0.25, 0.0, 0.5),
            point(0.25, 0.0, -0.5),
            point(0.75, 0.5, 0.0),
            point(0.75, -0.5, 0.0),
            point(0.75, 0.0, 0.5),
            point(0.75, 0.0, -0.5),
        ]);
        runtime.mesh.face_area_vectors.extend([
            point(0.0, 1.0, 0.0),
            point(0.0, -1.0, 0.0),
            point(0.0, 0.0, 1.0),
            point(0.0, 0.0, -1.0),
            point(0.0, 1.0, 0.0),
            point(0.0, -1.0, 0.0),
            point(0.0, 0.0, 1.0),
            point(0.0, 0.0, -1.0),
        ]);
        runtime.mesh.patches.push(SolverRuntimePatchRange {
            name: "frontAndBack".to_string(),
            patch_type: "empty".to_string(),
            start_face: 3,
            faces: 8,
        });
        runtime
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

    fn two_cell_fields_with_empty_sides() -> InitialFieldSet {
        let mut fields = two_cell_fields();
        for field in &mut fields.fields {
            field.boundary_patches.push(FieldBoundaryPatch {
                name: "frontAndBack".to_string(),
                patch_type: Some("empty".to_string()),
                inlet_value: None,
                value: None,
            });
        }
        fields
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

    fn assert_point_fields_bit_equal(actual: &[Point3], expected: &[Point3]) {
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert_eq!(actual.x.to_bits(), expected.x.to_bits());
            assert_eq!(actual.y.to_bits(), expected.y.to_bits());
            assert_eq!(actual.z.to_bits(), expected.z.to_bits());
        }
    }

    fn assert_close(left: f64, right: f64) {
        assert!(
            (left - right).abs() < 1.0e-12,
            "expected {left} to be close to {right}"
        );
    }
}
