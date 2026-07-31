use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::time::Instant;

use super::{
    CgPreconditioner, CsrMatrix, CsrSparsityPattern, IterativeSolveReport,
    IterativeSolveTermination, PreconditionedConjugateGradientOptions,
    PreconditionedConjugateGradientWorkspace, dot, dot_product_is_singular,
    gauss_seidel_sweep_with_cached_diagonal, invalid_input, l2_norm,
    validate_iterative_solve_input,
};
use crate::Result;

const MAX_LEVELS: usize = 50;
const COARSEST_MAX_ITERATIONS: usize = 1_000;
const SCALE_STABILISER: f64 = 1.0e-300;
pub(crate) const OPENFOAM_RELATIVE_TOLERANCE_SMALL: f64 = 1.0e-20;
/// Caps dense coarsest storage at 512 KiB and its cubic factorisation at
/// roughly 17 million elimination steps, keeping both costs predictable.
const MAX_DENSE_COARSEST_CELLS: usize = 256;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GamgAgglomerator {
    AlgebraicPair,
    FaceAreaPair,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GamgSmoother {
    GaussSeidel,
    SymGaussSeidel,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GamgOuterSolver {
    Standalone,
    FlexibleCg,
}

impl std::fmt::Display for GamgAgglomerator {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AlgebraicPair => formatter.write_str("algebraicPair"),
            Self::FaceAreaPair => formatter.write_str("faceAreaPair"),
        }
    }
}

impl std::fmt::Display for GamgSmoother {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::GaussSeidel => formatter.write_str("GaussSeidel"),
            Self::SymGaussSeidel => formatter.write_str("symGaussSeidel"),
        }
    }
}

impl std::fmt::Display for GamgOuterSolver {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Standalone => formatter.write_str("standalone"),
            Self::FlexibleCg => formatter.write_str("FCG"),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct GamgOptions {
    pub max_iterations: usize,
    pub min_iterations: usize,
    pub tolerance: f64,
    pub relative_tolerance: f64,
    pub cache_agglomeration: bool,
    pub n_cells_in_coarsest_level: usize,
    pub merge_levels: usize,
    pub agglomerator: GamgAgglomerator,
    pub smoother: GamgSmoother,
    pub outer_solver: GamgOuterSolver,
    pub n_pre_sweeps: usize,
    pub pre_sweeps_level_multiplier: usize,
    pub max_pre_sweeps: usize,
    pub n_post_sweeps: usize,
    pub post_sweeps_level_multiplier: usize,
    pub max_post_sweeps: usize,
    pub n_finest_sweeps: usize,
    pub interpolate_correction: bool,
    pub scale_correction: bool,
    pub direct_solve_coarsest: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct GamgSolveControls {
    pub max_iterations: usize,
    pub min_iterations: usize,
    pub tolerance: f64,
    pub relative_tolerance: f64,
}

#[derive(Clone, Copy)]
struct FcgLinearSystem<'a> {
    matrix: &'a CsrMatrix,
    rhs: &'a [f64],
}

#[derive(Clone, Copy)]
struct FcgInitialNorms {
    l1: f64,
    l2: f64,
}

#[derive(Clone, Copy, Debug)]
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) struct NormalizedL1GamgSolveControls {
    pub(crate) normalization_factor: f64,
    pub(crate) tolerance: f64,
    pub(crate) relative_tolerance: f64,
    pub(crate) l2_controls: GamgSolveControls,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct GamgLevelTiming {
    pub level: usize,
    pub cells: usize,
    pub nonzeros: usize,
    pub matrix_refresh_seconds: f64,
    pub restriction_seconds: f64,
    pub prolongation_seconds: f64,
    pub smoothing_seconds: f64,
    pub scaling_seconds: f64,
    pub residual_seconds: f64,
    pub correction_seconds: f64,
    pub coarsest_solve_seconds: f64,
    pub matrix_refreshes: usize,
    pub restriction_calls: usize,
    pub prolongation_calls: usize,
    pub smoothing_calls: usize,
    pub smoothing_sweeps: usize,
    pub scaling_calls: usize,
    pub residual_evaluations: usize,
    pub correction_updates: usize,
    pub coarsest_solves: usize,
    pub coarsest_iterations: usize,
}

impl GamgLevelTiming {
    fn new(level: usize, matrix: &CsrMatrix) -> Self {
        Self {
            level,
            cells: matrix.rows(),
            nonzeros: matrix.nnz(),
            ..Self::default()
        }
    }

    fn phase_seconds(self) -> f64 {
        self.restriction_seconds
            + self.prolongation_seconds
            + self.smoothing_seconds
            + self.scaling_seconds
            + self.residual_seconds
            + self.correction_seconds
            + self.coarsest_solve_seconds
    }

    fn validate_metadata(&self, other: Self) -> Result<()> {
        if self.level != other.level || self.cells != other.cells || self.nonzeros != other.nonzeros
        {
            return Err(invalid_input(format!(
                "GAMG profile hierarchy changed at level {}: expected cells={} nonzeros={}, got level={} cells={} nonzeros={}",
                self.level, self.cells, self.nonzeros, other.level, other.cells, other.nonzeros
            )));
        }
        Ok(())
    }

    fn accumulate_unchecked(&mut self, other: Self) {
        self.restriction_seconds += other.restriction_seconds;
        self.matrix_refresh_seconds += other.matrix_refresh_seconds;
        self.prolongation_seconds += other.prolongation_seconds;
        self.smoothing_seconds += other.smoothing_seconds;
        self.scaling_seconds += other.scaling_seconds;
        self.residual_seconds += other.residual_seconds;
        self.correction_seconds += other.correction_seconds;
        self.coarsest_solve_seconds += other.coarsest_solve_seconds;
        self.matrix_refreshes += other.matrix_refreshes;
        self.restriction_calls += other.restriction_calls;
        self.prolongation_calls += other.prolongation_calls;
        self.smoothing_calls += other.smoothing_calls;
        self.smoothing_sweeps += other.smoothing_sweeps;
        self.scaling_calls += other.scaling_calls;
        self.residual_evaluations += other.residual_evaluations;
        self.correction_updates += other.correction_updates;
        self.coarsest_solves += other.coarsest_solves;
        self.coarsest_iterations += other.coarsest_iterations;
    }
}

/// One sorted aggregate-size histogram bin in a profiled GAMG hierarchy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GamgAggregateSizeBin {
    pub aggregate_size: usize,
    pub aggregate_count: usize,
}

/// Static diagnostics for one fine-to-coarse transfer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GamgTransferDiagnostics {
    pub fine_level: usize,
    pub coarse_level: usize,
    pub fine_cells: usize,
    pub coarse_cells: usize,
    pub singleton_fine_cells: usize,
    pub unmatched_fine_cells: usize,
    pub min_aggregate_size: usize,
    pub max_aggregate_size: usize,
    pub aggregate_size_histogram: Vec<GamgAggregateSizeBin>,
}

/// Static matrix shape for one GAMG hierarchy level.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GamgHierarchyLevelDiagnostics {
    pub level: usize,
    pub cells: usize,
    pub nonzeros: usize,
}

/// Static hierarchy data collected only by profiled solves.
///
/// Integer counts are authoritative. Complexity helpers return exact
/// numerator and denominator terms without floating-point division.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GamgHierarchyDiagnostics {
    pub levels: Vec<GamgHierarchyLevelDiagnostics>,
    pub transfers: Vec<GamgTransferDiagnostics>,
    pub smoother_passes_per_sweep: usize,
    pub direct_solve_coarsest: bool,
}

impl GamgHierarchyDiagnostics {
    /// Returns `(sum(level cells), finest cells)` for grid complexity.
    pub fn grid_complexity_terms(&self) -> Option<(u128, u128)> {
        let finest = self.levels.first()?.cells as u128;
        if finest == 0 {
            return None;
        }
        let numerator = self
            .levels
            .iter()
            .try_fold(0u128, |sum, level| sum.checked_add(level.cells as u128))?;
        Some((numerator, finest))
    }

    /// Returns `(sum(level nonzeros), finest nonzeros)` for operator complexity.
    pub fn operator_complexity_terms(&self) -> Option<(u128, u128)> {
        let finest = self.levels.first()?.nonzeros as u128;
        if finest == 0 {
            return None;
        }
        let numerator = self
            .levels
            .iter()
            .try_fold(0u128, |sum, level| sum.checked_add(level.nonzeros as u128))?;
        Some((numerator, finest))
    }
}

#[derive(Clone, Debug, Default)]
pub struct GamgKernelTiming {
    pub total_seconds: f64,
    pub hierarchy_build_seconds: f64,
    pub hierarchy_rebuild_seconds: f64,
    pub hierarchy_diagnostic_seconds: f64,
    pub matrix_refresh_seconds: f64,
    pub finest_residual_seconds: f64,
    pub v_cycle_seconds: f64,
    pub other_seconds: f64,
    pub hierarchy_builds: usize,
    pub hierarchy_rebuilds: usize,
    pub matrix_refreshes: usize,
    pub finest_residual_evaluations: usize,
    pub solves: usize,
    pub v_cycles: usize,
    /// Matrix-vector products performed by the opt-in FCG outer iteration.
    pub outer_matrix_vector_products: usize,
    /// Logical vector reductions performed by FCG iterations. This excludes
    /// initial residual normalization and reductions inside a GAMG V-cycle.
    pub outer_reductions: usize,
    pub levels: Vec<GamgLevelTiming>,
    pub hierarchy: Option<Arc<GamgHierarchyDiagnostics>>,
}

impl GamgKernelTiming {
    fn from_matrices(matrices: &[CsrMatrix]) -> Self {
        Self {
            levels: matrices
                .iter()
                .enumerate()
                .map(|(level, matrix)| GamgLevelTiming::new(level, matrix))
                .collect(),
            ..Self::default()
        }
    }

    fn from_hierarchy(matrices: &[CsrMatrix], hierarchy: Arc<GamgHierarchyDiagnostics>) -> Self {
        let mut timing = Self::from_matrices(matrices);
        timing.hierarchy = Some(hierarchy);
        timing
    }

    pub fn add_hierarchy_build(&mut self, seconds: f64) {
        self.total_seconds += seconds;
        self.hierarchy_build_seconds += seconds;
        self.hierarchy_builds += 1;
    }

    pub fn restriction_seconds(&self) -> f64 {
        self.levels
            .iter()
            .map(|level| level.restriction_seconds)
            .sum()
    }

    pub fn prolongation_seconds(&self) -> f64 {
        self.levels
            .iter()
            .map(|level| level.prolongation_seconds)
            .sum()
    }

    pub fn smoothing_seconds(&self) -> f64 {
        self.levels
            .iter()
            .map(|level| level.smoothing_seconds)
            .sum()
    }

    pub fn scaling_seconds(&self) -> f64 {
        self.levels.iter().map(|level| level.scaling_seconds).sum()
    }

    pub fn coarse_residual_seconds(&self) -> f64 {
        self.levels.iter().map(|level| level.residual_seconds).sum()
    }

    pub fn correction_seconds(&self) -> f64 {
        self.levels
            .iter()
            .map(|level| level.correction_seconds)
            .sum()
    }

    pub fn coarsest_solve_seconds(&self) -> f64 {
        self.levels
            .iter()
            .map(|level| level.coarsest_solve_seconds)
            .sum()
    }

    pub fn v_cycle_other_seconds(&self) -> f64 {
        let accounted = self
            .levels
            .iter()
            .copied()
            .map(GamgLevelTiming::phase_seconds)
            .sum::<f64>();
        (self.v_cycle_seconds - accounted).max(0.0)
    }

    /// Returns an NNZ-weighted smoothing-work proxy.
    ///
    /// This deliberately weights every logical smoother pass by the complete
    /// level NNZ. It is a stable hierarchy/work comparison unit, not a claim
    /// that the cached-diagonal kernel physically reloads the diagonal entry.
    pub fn nnz_weighted_smoothing_work(&self) -> Option<u128> {
        let hierarchy = self.hierarchy.as_ref()?;
        if hierarchy.levels.len() != self.levels.len() {
            return None;
        }
        self.levels
            .iter()
            .zip(&hierarchy.levels)
            .try_fold(0u128, |sum, (timing, level)| {
                if timing.level != level.level
                    || timing.cells != level.cells
                    || timing.nonzeros != level.nonzeros
                {
                    return None;
                }
                let visits = (timing.smoothing_sweeps as u128)
                    .checked_mul(hierarchy.smoother_passes_per_sweep as u128)?
                    .checked_mul(level.nonzeros as u128)?;
                sum.checked_add(visits)
            })
    }

    /// Returns an NNZ-weighted sparse-work proxy for profiled solver work.
    ///
    /// The sum includes smoothing, level/finest residual products, correction
    /// scaling products, and FCG outer matrix-vector products. Transfer maps,
    /// coefficient refresh, and vector-only reductions remain separate.
    pub fn nnz_weighted_sparse_work(&self) -> Option<u128> {
        let hierarchy = self.hierarchy.as_ref()?;
        if hierarchy.levels.len() != self.levels.len() {
            return None;
        }
        let level_visits =
            self.levels
                .iter()
                .zip(&hierarchy.levels)
                .try_fold(0u128, |sum, (timing, level)| {
                    if timing.level != level.level
                        || timing.cells != level.cells
                        || timing.nonzeros != level.nonzeros
                    {
                        return None;
                    }
                    let smoothing_passes = (timing.smoothing_sweeps as u128)
                        .checked_mul(hierarchy.smoother_passes_per_sweep as u128)?;
                    let sparse_passes = smoothing_passes
                        .checked_add(timing.residual_evaluations as u128)?
                        .checked_add(timing.scaling_calls as u128)?;
                    let visits = sparse_passes.checked_mul(level.nonzeros as u128)?;
                    sum.checked_add(visits)
                })?;
        let finest_matrix_products = (self.finest_residual_evaluations as u128)
            .checked_add(self.outer_matrix_vector_products as u128)?;
        let finest_residual_work =
            finest_matrix_products.checked_mul(hierarchy.levels.first()?.nonzeros as u128)?;
        level_visits.checked_add(finest_residual_work)
    }

    pub fn accumulate(&mut self, other: &Self) -> Result<()> {
        let initialize_levels = self.levels.is_empty();
        if initialize_levels {
            if self.hierarchy.is_some() && self.hierarchy != other.hierarchy {
                return Err(invalid_input(
                    "GAMG profile static hierarchy diagnostics changed during accumulation"
                        .to_string(),
                ));
            }
        } else if self.levels.len() != other.levels.len() {
            return Err(invalid_input(format!(
                "GAMG profile hierarchy changed from {} to {} levels",
                self.levels.len(),
                other.levels.len()
            )));
        } else {
            if self.hierarchy != other.hierarchy {
                return Err(invalid_input(
                    "GAMG profile static hierarchy diagnostics changed during accumulation"
                        .to_string(),
                ));
            }
            for (level, other_level) in self.levels.iter().zip(&other.levels) {
                level.validate_metadata(*other_level)?;
            }
        }

        if initialize_levels {
            self.levels = other.levels.clone();
            self.hierarchy = other.hierarchy.clone();
        } else {
            for (level, other_level) in self.levels.iter_mut().zip(&other.levels) {
                level.accumulate_unchecked(*other_level);
            }
        }
        self.total_seconds += other.total_seconds;
        self.hierarchy_build_seconds += other.hierarchy_build_seconds;
        self.hierarchy_rebuild_seconds += other.hierarchy_rebuild_seconds;
        self.hierarchy_diagnostic_seconds += other.hierarchy_diagnostic_seconds;
        self.matrix_refresh_seconds += other.matrix_refresh_seconds;
        self.finest_residual_seconds += other.finest_residual_seconds;
        self.v_cycle_seconds += other.v_cycle_seconds;
        self.other_seconds += other.other_seconds;
        self.hierarchy_builds += other.hierarchy_builds;
        self.hierarchy_rebuilds += other.hierarchy_rebuilds;
        self.matrix_refreshes += other.matrix_refreshes;
        self.finest_residual_evaluations += other.finest_residual_evaluations;
        self.solves += other.solves;
        self.v_cycles += other.v_cycles;
        self.outer_matrix_vector_products += other.outer_matrix_vector_products;
        self.outer_reductions += other.outer_reductions;
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct ProfiledGamgSolveReport {
    pub report: IterativeSolveReport,
    pub timing: GamgKernelTiming,
}

impl From<GamgOptions> for GamgSolveControls {
    fn from(options: GamgOptions) -> Self {
        Self {
            max_iterations: options.max_iterations,
            min_iterations: options.min_iterations,
            tolerance: options.tolerance,
            relative_tolerance: options.relative_tolerance,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GamgFacePairWeight {
    first_cell: usize,
    second_cell: usize,
    weight: f64,
}

impl GamgFacePairWeight {
    pub fn new(first_cell: usize, second_cell: usize, weight: f64) -> Result<Self> {
        if first_cell == second_cell {
            return Err(invalid_input(format!(
                "GAMG faceAreaPair connection must join different cells, got {first_cell} twice"
            )));
        }
        if !weight.is_finite() || weight <= 0.0 {
            return Err(invalid_input(format!(
                "GAMG faceAreaPair weight must be positive and finite, got {weight}"
            )));
        }
        Ok(Self {
            first_cell,
            second_cell,
            weight,
        })
    }

    pub fn cells(self) -> (usize, usize) {
        (self.first_cell, self.second_cell)
    }

    pub fn weight(self) -> f64 {
        self.weight
    }
}

impl Default for GamgOptions {
    fn default() -> Self {
        Self {
            max_iterations: 1_000,
            min_iterations: 0,
            tolerance: 1.0e-10,
            relative_tolerance: 0.0,
            cache_agglomeration: true,
            n_cells_in_coarsest_level: 10,
            merge_levels: 1,
            // Matrix-only callers have no face geometry, so the convenience
            // default selects algebraicPair explicitly.
            agglomerator: GamgAgglomerator::AlgebraicPair,
            smoother: GamgSmoother::GaussSeidel,
            outer_solver: GamgOuterSolver::Standalone,
            n_pre_sweeps: 0,
            pre_sweeps_level_multiplier: 1,
            max_pre_sweeps: 4,
            n_post_sweeps: 2,
            post_sweeps_level_multiplier: 1,
            max_post_sweeps: 4,
            n_finest_sweeps: 2,
            interpolate_correction: false,
            scale_correction: true,
            direct_solve_coarsest: false,
        }
    }
}

#[derive(Clone)]
enum GamgAgglomerationSource {
    Algebraic,
    FaceArea(Arc<[GamgFacePairWeight]>),
}

#[derive(Clone, Debug)]
struct GamgTransfer {
    fine_to_coarse: Vec<usize>,
    fine_entry_to_coarse_entry: Vec<usize>,
}

struct GamgInterpolationScratch<'a> {
    fine: &'a mut [f64],
    coarse_correction: &'a mut [f64],
    coarse_diagonal: &'a mut [f64],
}

fn fcg_mmax_one_direction(
    preconditioned_residual: &[f64],
    previous: Option<(&[f64], &[f64])>,
    direction: &mut [f64],
) -> Result<Option<f64>> {
    if direction.len() != preconditioned_residual.len() {
        return Err(invalid_input(format!(
            "GAMG FCG direction length mismatch: expected {}, got {}",
            preconditioned_residual.len(),
            direction.len()
        )));
    }
    if let Some((previous_direction, previous_normalized_matrix_direction)) = previous {
        if previous_direction.len() != direction.len()
            || previous_normalized_matrix_direction.len() != direction.len()
        {
            return Err(invalid_input(format!(
                "GAMG FCG previous-direction length mismatch: expected {}, got direction={} normalizedMatrixDirection={}",
                direction.len(),
                previous_direction.len(),
                previous_normalized_matrix_direction.len()
            )));
        }
        let orthogonalisation = dot(
            preconditioned_residual,
            previous_normalized_matrix_direction,
        );
        if !orthogonalisation.is_finite() {
            return Err(invalid_input(
                "GAMG FCG orthogonalisation product is not finite".to_string(),
            ));
        }
        for row in 0..direction.len() {
            direction[row] =
                preconditioned_residual[row] + (-orthogonalisation) * previous_direction[row];
        }
        Ok(Some(orthogonalisation))
    } else {
        direction.copy_from_slice(preconditioned_residual);
        Ok(None)
    }
}

fn validate_fcg_preconditioner_output(values: &[f64]) -> Result<()> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err(invalid_input(
            "GAMG FCG preconditioner output is not finite".to_string(),
        ));
    }
    Ok(())
}

impl GamgTransfer {
    fn restrict_sum(&self, fine: &[f64], coarse: &mut [f64]) -> Result<()> {
        if fine.len() != self.fine_to_coarse.len() {
            return Err(invalid_input(format!(
                "GAMG restriction expected {} fine entries, got {}",
                self.fine_to_coarse.len(),
                fine.len()
            )));
        }
        coarse.fill(0.0);
        for (fine_value, &coarse_index) in fine.iter().zip(&self.fine_to_coarse) {
            let coarse_len = coarse.len();
            let Some(coarse_value) = coarse.get_mut(coarse_index) else {
                return Err(invalid_input(format!(
                    "GAMG restriction coarse index {coarse_index} is out of range {coarse_len}"
                )));
            };
            *coarse_value += fine_value;
        }
        Ok(())
    }

    fn prolong_injection(&self, coarse: &[f64], fine: &mut [f64]) -> Result<()> {
        if fine.len() != self.fine_to_coarse.len() {
            return Err(invalid_input(format!(
                "GAMG prolongation expected {} fine entries, got {}",
                self.fine_to_coarse.len(),
                fine.len()
            )));
        }
        for (fine_value, &coarse_index) in fine.iter_mut().zip(&self.fine_to_coarse) {
            *fine_value = *coarse.get(coarse_index).ok_or_else(|| {
                invalid_input(format!(
                    "GAMG prolongation coarse index {coarse_index} is out of range {}",
                    coarse.len()
                ))
            })?;
        }
        Ok(())
    }

    fn interpolate_correction(
        &self,
        matrix: &CsrMatrix,
        diagonal_values: &[f64],
        coarse: &[f64],
        fine: &mut [f64],
        scratch: GamgInterpolationScratch<'_>,
    ) -> Result<()> {
        let GamgInterpolationScratch {
            fine: fine_scratch,
            coarse_correction: coarse_correction_scratch,
            coarse_diagonal: coarse_diagonal_scratch,
        } = scratch;
        let rows = matrix.rows();
        if fine.len() != rows
            || fine_scratch.len() != rows
            || diagonal_values.len() != rows
            || self.fine_to_coarse.len() != rows
        {
            return Err(invalid_input(format!(
                "GAMG correction interpolation fine shape mismatch: matrix={rows} mapping={} diagonal={} correction={} scratch={}",
                self.fine_to_coarse.len(),
                diagonal_values.len(),
                fine.len(),
                fine_scratch.len()
            )));
        }
        if coarse_correction_scratch.len() != coarse.len()
            || coarse_diagonal_scratch.len() != coarse.len()
        {
            return Err(invalid_input(format!(
                "GAMG correction interpolation coarse shape mismatch: correction={} correctionScratch={} diagonalScratch={}",
                coarse.len(),
                coarse_correction_scratch.len(),
                coarse_diagonal_scratch.len()
            )));
        }
        for (row, &coarse_index) in self.fine_to_coarse.iter().enumerate() {
            if coarse_index >= coarse.len() {
                return Err(invalid_input(format!(
                    "GAMG correction interpolation coarse index {coarse_index} for fine row {row} is out of range {}",
                    coarse.len()
                )));
            }
        }

        // Match the Foundation GAMG interpolation order: form the complete
        // off-diagonal product from the injected field before replacing any
        // fine value with its diagonal interpolation.
        for row in 0..rows {
            let mut off_diagonal = 0.0;
            for entry in matrix.row_offsets()[row]..matrix.row_offsets()[row + 1] {
                let column = matrix.col_indices()[entry];
                if column != row {
                    off_diagonal += matrix.values()[entry] * fine[column];
                }
            }
            if !off_diagonal.is_finite() {
                return Err(invalid_input(format!(
                    "GAMG correction interpolation off-diagonal product at row {row} is not finite"
                )));
            }
            let diagonal = diagonal_values[row];
            if !diagonal.is_finite() || diagonal == 0.0 {
                return Err(invalid_input(format!(
                    "GAMG correction interpolation diagonal at row {row} is invalid: {diagonal}"
                )));
            }
            let interpolated = -off_diagonal / diagonal;
            if !interpolated.is_finite() {
                return Err(invalid_input(format!(
                    "GAMG correction interpolation value at row {row} is not finite"
                )));
            }
            fine_scratch[row] = interpolated;
        }

        coarse_correction_scratch.fill(0.0);
        coarse_diagonal_scratch.fill(0.0);
        for row in 0..rows {
            let coarse_index = self.fine_to_coarse[row];
            let diagonal = diagonal_values[row];
            let weighted = diagonal * fine_scratch[row];
            if !weighted.is_finite() {
                return Err(invalid_input(format!(
                    "GAMG correction interpolation weighted value at row {row} is not finite"
                )));
            }
            coarse_correction_scratch[coarse_index] += weighted;
            coarse_diagonal_scratch[coarse_index] += diagonal;
            if !coarse_correction_scratch[coarse_index].is_finite()
                || !coarse_diagonal_scratch[coarse_index].is_finite()
            {
                return Err(invalid_input(format!(
                    "GAMG correction interpolation aggregate {coarse_index} is not finite"
                )));
            }
        }
        for coarse_index in 0..coarse.len() {
            let diagonal = coarse_diagonal_scratch[coarse_index];
            if !diagonal.is_finite() || diagonal == 0.0 {
                return Err(invalid_input(format!(
                    "GAMG correction interpolation aggregate diagonal {coarse_index} is invalid: {diagonal}"
                )));
            }
            let correction =
                coarse[coarse_index] - coarse_correction_scratch[coarse_index] / diagonal;
            if !correction.is_finite() {
                return Err(invalid_input(format!(
                    "GAMG correction interpolation aggregate correction {coarse_index} is not finite"
                )));
            }
            coarse_correction_scratch[coarse_index] = correction;
        }
        for row in 0..rows {
            let corrected = fine_scratch[row] + coarse_correction_scratch[self.fine_to_coarse[row]];
            if !corrected.is_finite() {
                return Err(invalid_input(format!(
                    "GAMG correction interpolation corrected value at row {row} is not finite"
                )));
            }
            fine_scratch[row] = corrected;
        }
        fine.copy_from_slice(fine_scratch);
        Ok(())
    }

    fn agglomerate_values(&self, fine: &[f64], coarse: &mut [f64]) -> Result<()> {
        if fine.len() != self.fine_entry_to_coarse_entry.len() {
            return Err(invalid_input(format!(
                "GAMG coefficient agglomeration expected {} fine entries, got {}",
                self.fine_entry_to_coarse_entry.len(),
                fine.len()
            )));
        }
        coarse.fill(0.0);
        for (&fine_value, &coarse_entry) in fine.iter().zip(&self.fine_entry_to_coarse_entry) {
            let coarse_len = coarse.len();
            let coarse_value = coarse.get_mut(coarse_entry).ok_or_else(|| {
                invalid_input(format!(
                    "GAMG coarse matrix entry {coarse_entry} is out of range {coarse_len}"
                ))
            })?;
            *coarse_value += fine_value;
        }
        Ok(())
    }
}

fn build_hierarchy_diagnostics(
    matrices: &[CsrMatrix],
    transfers: &[GamgTransfer],
    options: GamgOptions,
) -> Result<GamgHierarchyDiagnostics> {
    if matrices.len() != transfers.len() + 1 {
        return Err(invalid_input(format!(
            "GAMG profile hierarchy expected one fewer transfers than levels, got {} levels and {} transfers",
            matrices.len(),
            transfers.len()
        )));
    }

    let levels = matrices
        .iter()
        .enumerate()
        .map(|(level, matrix)| GamgHierarchyLevelDiagnostics {
            level,
            cells: matrix.rows(),
            nonzeros: matrix.nnz(),
        })
        .collect();
    let mut transfer_diagnostics = Vec::with_capacity(transfers.len());
    for (fine_level, transfer) in transfers.iter().enumerate() {
        let coarse_level = fine_level + 1;
        let fine_cells = matrices[fine_level].rows();
        let coarse_cells = matrices[coarse_level].rows();
        if transfer.fine_to_coarse.len() != fine_cells {
            return Err(invalid_input(format!(
                "GAMG profile transfer {fine_level}->{coarse_level} expected {fine_cells} fine cells, got {}",
                transfer.fine_to_coarse.len()
            )));
        }

        let mut aggregate_sizes = vec![0usize; coarse_cells];
        for (fine_cell, &coarse_cell) in transfer.fine_to_coarse.iter().enumerate() {
            let aggregate_size = aggregate_sizes.get_mut(coarse_cell).ok_or_else(|| {
                invalid_input(format!(
                    "GAMG profile transfer {fine_level}->{coarse_level} maps fine cell {fine_cell} to out-of-range coarse cell {coarse_cell} of {coarse_cells}"
                ))
            })?;
            *aggregate_size = aggregate_size.checked_add(1).ok_or_else(|| {
                invalid_input(format!(
                    "GAMG profile transfer {fine_level}->{coarse_level} aggregate {coarse_cell} size overflow"
                ))
            })?;
        }
        if aggregate_sizes.contains(&0) {
            return Err(invalid_input(format!(
                "GAMG profile transfer {fine_level}->{coarse_level} contains an empty aggregate"
            )));
        }

        let singleton_fine_cells = aggregate_sizes.iter().filter(|&&size| size == 1).count();
        // Pair agglomeration seeds every non-singleton aggregate with one
        // matched pair. Remaining members were unmatched by the greedy pass
        // and subsequently attached to an existing aggregate.
        let unmatched_fine_cells = aggregate_sizes
            .iter()
            .try_fold(0usize, |sum, &size| {
                sum.checked_add(if size == 1 { 1 } else { size - 2 })
            })
            .ok_or_else(|| {
                invalid_input(format!(
                    "GAMG profile transfer {fine_level}->{coarse_level} unmatched-cell count overflow"
                ))
            })?;
        let min_aggregate_size = aggregate_sizes.iter().copied().min().unwrap_or(0);
        let max_aggregate_size = aggregate_sizes.iter().copied().max().unwrap_or(0);
        let mut histogram = BTreeMap::<usize, usize>::new();
        for aggregate_size in aggregate_sizes {
            let count = histogram.entry(aggregate_size).or_default();
            *count = count.checked_add(1).ok_or_else(|| {
                invalid_input(format!(
                    "GAMG profile transfer {fine_level}->{coarse_level} aggregate histogram overflow"
                ))
            })?;
        }
        let aggregate_size_histogram = histogram
            .into_iter()
            .map(|(aggregate_size, aggregate_count)| GamgAggregateSizeBin {
                aggregate_size,
                aggregate_count,
            })
            .collect();

        transfer_diagnostics.push(GamgTransferDiagnostics {
            fine_level,
            coarse_level,
            fine_cells,
            coarse_cells,
            singleton_fine_cells,
            unmatched_fine_cells,
            min_aggregate_size,
            max_aggregate_size,
            aggregate_size_histogram,
        });
    }

    Ok(GamgHierarchyDiagnostics {
        levels,
        transfers: transfer_diagnostics,
        smoother_passes_per_sweep: match options.smoother {
            GamgSmoother::GaussSeidel => 1,
            GamgSmoother::SymGaussSeidel => 2,
        },
        direct_solve_coarsest: options.direct_solve_coarsest,
    })
}

pub struct GamgWorkspace {
    options: GamgOptions,
    agglomeration_source: GamgAgglomerationSource,
    finest_sparsity: CsrSparsityPattern,
    matrices: Vec<CsrMatrix>,
    transfers: Vec<GamgTransfer>,
    diagonal_slots: Vec<Vec<usize>>,
    diagonal_values: Vec<Vec<f64>>,
    corrections: Vec<Vec<f64>>,
    sources: Vec<Vec<f64>>,
    residuals: Vec<Vec<f64>>,
    products: Vec<Vec<f64>>,
    pre_smoothed: Vec<Vec<f64>>,
    fcg_residual: Vec<f64>,
    fcg_preconditioned_residual: Vec<f64>,
    fcg_direction: Vec<f64>,
    fcg_matrix_direction: Vec<f64>,
    fcg_previous_direction: Vec<f64>,
    fcg_previous_matrix_direction: Vec<f64>,
    coarsest_pcg: Option<PreconditionedConjugateGradientWorkspace>,
    profiled_hierarchy: Option<Arc<GamgHierarchyDiagnostics>>,
    has_solved: bool,
}

impl GamgWorkspace {
    pub fn new(matrix: &CsrMatrix, options: GamgOptions) -> Result<Self> {
        if options.agglomerator != GamgAgglomerator::AlgebraicPair {
            return Err(invalid_input(
                "GAMG faceAreaPair requires explicit mesh face weights; use GamgWorkspace::new_with_face_area_weights"
                    .to_string(),
            ));
        }
        Self::build(matrix, options, GamgAgglomerationSource::Algebraic)
    }

    pub fn new_with_face_area_weights(
        matrix: &CsrMatrix,
        options: GamgOptions,
        face_weights: &[GamgFacePairWeight],
    ) -> Result<Self> {
        if options.agglomerator != GamgAgglomerator::FaceAreaPair {
            return Err(invalid_input(
                "GAMG face-area weights require agglomerator faceAreaPair; no agglomerator substitution was applied"
                    .to_string(),
            ));
        }
        Self::build(
            matrix,
            options,
            GamgAgglomerationSource::FaceArea(Arc::from(face_weights)),
        )
    }

    fn build(
        matrix: &CsrMatrix,
        options: GamgOptions,
        agglomeration_source: GamgAgglomerationSource,
    ) -> Result<Self> {
        validate_options(options)?;
        validate_gamg_matrix(matrix)?;
        if options.merge_levels != 1 {
            return Err(invalid_input(format!(
                "GAMG mergeLevels={} is not implemented by the matrix foundation; no level-combination fallback was applied",
                options.merge_levels
            )));
        }
        Self::build_validated(matrix, options, agglomeration_source)
    }

    fn build_validated(
        matrix: &CsrMatrix,
        options: GamgOptions,
        agglomeration_source: GamgAgglomerationSource,
    ) -> Result<Self> {
        Self::build_validated_with_finest_diagonal(matrix, options, agglomeration_source, None)
    }

    fn build_validated_with_finest_diagonal(
        matrix: &CsrMatrix,
        options: GamgOptions,
        agglomeration_source: GamgAgglomerationSource,
        finest_diagonal_slots: Option<Vec<usize>>,
    ) -> Result<Self> {
        let finest_sparsity = matrix.sparsity_pattern();
        let mut matrices = vec![matrix.clone()];
        let mut transfers = Vec::new();
        let mut forward = true;
        let mut face_edges = match &agglomeration_source {
            GamgAgglomerationSource::Algebraic => None,
            GamgAgglomerationSource::FaceArea(weights) => Some(face_pair_edges(matrix, weights)?),
        };

        while matrices.len() < MAX_LEVELS {
            let fine = matrices.last().expect("GAMG always has a finest matrix");
            let (fine_to_coarse, n_coarse_cells) = if let Some(edges) = &face_edges {
                pair_map_from_edges(fine.rows(), edges, forward)?
            } else {
                algebraic_pair_map(fine, forward)?
            };
            forward = !forward;
            if n_coarse_cells < options.n_cells_in_coarsest_level || n_coarse_cells >= fine.rows() {
                break;
            }
            let next_face_edges = face_edges
                .as_ref()
                .map(|edges| agglomerate_pair_edges(edges, &fine_to_coarse, n_coarse_cells));
            let (transfer, coarse) = build_coarse_matrix(fine, fine_to_coarse, n_coarse_cells)?;
            transfers.push(transfer);
            matrices.push(coarse);
            face_edges = next_face_edges;
        }

        if transfers.is_empty() {
            return Err(invalid_input(format!(
                "GAMG created no coarse level for {} rows with nCellsInCoarsestLevel={}; choose another solver or reduce nCellsInCoarsestLevel",
                matrix.rows(),
                options.n_cells_in_coarsest_level
            )));
        }

        let mut diagonal_slots = Vec::with_capacity(matrices.len());
        if let Some(slots) = finest_diagonal_slots {
            diagonal_slots.push(slots);
        } else {
            diagonal_slots.push(super::csr_diagonal_slots(&matrices[0])?);
        }
        for matrix in &matrices[1..] {
            diagonal_slots.push(super::csr_diagonal_slots(matrix)?);
        }
        let mut diagonal_values =
            level_vectors(&matrices.iter().map(CsrMatrix::rows).collect::<Vec<_>>());
        for level in 0..matrices.len() {
            refresh_diagonal_values(
                &matrices[level],
                &diagonal_slots[level],
                &mut diagonal_values[level],
            )?;
        }
        let level_sizes = matrices.iter().map(CsrMatrix::rows).collect::<Vec<_>>();
        let corrections = level_vectors(&level_sizes);
        let sources = level_vectors(&level_sizes);
        let residuals = level_vectors(&level_sizes);
        let products = level_vectors(&level_sizes);
        let pre_smoothed = level_vectors(&level_sizes);
        // Standalone GAMG is the default and must not pay the per-cell memory
        // cost of the opt-in FCG outer iteration.
        let fcg_rows = if options.outer_solver == GamgOuterSolver::FlexibleCg {
            matrix.rows()
        } else {
            0
        };
        let fcg_residual = vec![0.0; fcg_rows];
        let fcg_preconditioned_residual = vec![0.0; fcg_rows];
        let fcg_direction = vec![0.0; fcg_rows];
        let fcg_matrix_direction = vec![0.0; fcg_rows];
        let fcg_previous_direction = vec![0.0; fcg_rows];
        let fcg_previous_matrix_direction = vec![0.0; fcg_rows];
        let coarsest_pcg = if options.direct_solve_coarsest {
            None
        } else {
            Some(PreconditionedConjugateGradientWorkspace::new(
                matrices.last().expect("GAMG has a coarsest matrix"),
                CgPreconditioner::IncompleteCholesky,
            )?)
        };

        Ok(Self {
            options,
            agglomeration_source,
            finest_sparsity,
            matrices,
            transfers,
            diagonal_slots,
            diagonal_values,
            corrections,
            sources,
            residuals,
            products,
            pre_smoothed,
            fcg_residual,
            fcg_preconditioned_residual,
            fcg_direction,
            fcg_matrix_direction,
            fcg_previous_direction,
            fcg_previous_matrix_direction,
            coarsest_pcg,
            profiled_hierarchy: None,
            has_solved: false,
        })
    }

    pub fn level_count(&self) -> usize {
        self.matrices.len()
    }

    pub fn level_sizes(&self) -> Vec<usize> {
        self.matrices.iter().map(CsrMatrix::rows).collect()
    }

    pub fn solve(
        &mut self,
        matrix: &CsrMatrix,
        rhs: &[f64],
        initial: Option<&[f64]>,
    ) -> Result<IterativeSolveReport> {
        self.solve_with_controls(matrix, rhs, initial, self.options.into())
    }

    pub fn solve_with_controls(
        &mut self,
        matrix: &CsrMatrix,
        rhs: &[f64],
        initial: Option<&[f64]>,
        controls: GamgSolveControls,
    ) -> Result<IterativeSolveReport> {
        let mut timing = GamgKernelTiming::default();
        self.solve_with_controls_internal::<false, false>(
            matrix,
            rhs,
            initial,
            controls,
            &mut timing,
        )
    }

    pub fn solve_with_controls_profiled(
        &mut self,
        matrix: &CsrMatrix,
        rhs: &[f64],
        initial: Option<&[f64]>,
        controls: GamgSolveControls,
    ) -> Result<ProfiledGamgSolveReport> {
        let started = Instant::now();
        let mut timing = GamgKernelTiming::default();
        let report = self.solve_with_controls_internal::<true, false>(
            matrix,
            rhs,
            initial,
            controls,
            &mut timing,
        )?;
        if self.profiled_hierarchy.is_none() {
            self.profiled_hierarchy = timing.hierarchy.clone();
        }
        timing.total_seconds = started.elapsed().as_secs_f64();
        let accounted_seconds = timing.hierarchy_rebuild_seconds
            + timing.hierarchy_diagnostic_seconds
            + timing.matrix_refresh_seconds
            + timing.finest_residual_seconds
            + timing.v_cycle_seconds;
        timing.other_seconds = (timing.total_seconds - accounted_seconds).max(0.0);
        Ok(ProfiledGamgSolveReport { report, timing })
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn solve_normalized_l1_with_controls(
        &mut self,
        matrix: &CsrMatrix,
        rhs: &[f64],
        initial: Option<&[f64]>,
        controls: NormalizedL1GamgSolveControls,
    ) -> Result<IterativeSolveReport> {
        let mut timing = GamgKernelTiming::default();
        self.solve_normalized_l1_with_controls_internal::<false, false>(
            matrix,
            rhs,
            initial,
            controls,
            &mut timing,
        )
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn solve_normalized_l1_with_controls_profiled(
        &mut self,
        matrix: &CsrMatrix,
        rhs: &[f64],
        initial: Option<&[f64]>,
        controls: NormalizedL1GamgSolveControls,
    ) -> Result<ProfiledGamgSolveReport> {
        let started = Instant::now();
        let mut timing = GamgKernelTiming::default();
        let report = self.solve_normalized_l1_with_controls_internal::<true, false>(
            matrix,
            rhs,
            initial,
            controls,
            &mut timing,
        )?;
        if self.profiled_hierarchy.is_none() {
            self.profiled_hierarchy = timing.hierarchy.clone();
        }
        timing.total_seconds = started.elapsed().as_secs_f64();
        let accounted_seconds = timing.hierarchy_rebuild_seconds
            + timing.hierarchy_diagnostic_seconds
            + timing.matrix_refresh_seconds
            + timing.finest_residual_seconds
            + timing.v_cycle_seconds;
        timing.other_seconds = (timing.total_seconds - accounted_seconds).max(0.0);
        Ok(ProfiledGamgSolveReport { report, timing })
    }

    #[cfg_attr(not(test), allow(dead_code))]
    fn solve_normalized_l1_with_controls_internal<
        const PROFILE: bool,
        const SCANNED_DIAGONAL: bool,
    >(
        &mut self,
        matrix: &CsrMatrix,
        rhs: &[f64],
        initial: Option<&[f64]>,
        controls: NormalizedL1GamgSolveControls,
        timing: &mut GamgKernelTiming,
    ) -> Result<IterativeSolveReport> {
        let finest_diagonal_slots = validate_normalized_l1_call(matrix, rhs, initial, controls)?;

        if !self.options.cache_agglomeration && self.has_solved {
            let rebuild_started = profile_started::<PROFILE>();
            *self = Self::build_validated_with_finest_diagonal(
                matrix,
                self.options,
                self.agglomeration_source.clone(),
                Some(finest_diagonal_slots),
            )?;
            add_profile_elapsed::<PROFILE>(&mut timing.hierarchy_rebuild_seconds, rebuild_started);
            if PROFILE {
                timing.hierarchy_rebuilds += 1;
            }
        }
        if PROFILE {
            let hierarchy_rebuild_seconds = timing.hierarchy_rebuild_seconds;
            let hierarchy_rebuilds = timing.hierarchy_rebuilds;
            let (hierarchy, hierarchy_diagnostic_seconds) = match &self.profiled_hierarchy {
                Some(hierarchy) => (Arc::clone(hierarchy), 0.0),
                None => {
                    let diagnostic_started = profile_started::<PROFILE>();
                    let hierarchy = Arc::new(build_hierarchy_diagnostics(
                        &self.matrices,
                        &self.transfers,
                        self.options,
                    )?);
                    (hierarchy, profile_elapsed(diagnostic_started))
                }
            };
            *timing = GamgKernelTiming::from_hierarchy(&self.matrices, hierarchy);
            timing.hierarchy_diagnostic_seconds = hierarchy_diagnostic_seconds;
            timing.hierarchy_rebuild_seconds = hierarchy_rebuild_seconds;
            timing.hierarchy_rebuilds = hierarchy_rebuilds;
            timing.solves = 1;
        }

        let refresh_started = profile_started::<PROFILE>();
        self.refresh_matrix_values::<PROFILE>(matrix, timing)?;
        add_profile_elapsed::<PROFILE>(&mut timing.matrix_refresh_seconds, refresh_started);
        if PROFILE {
            timing.matrix_refreshes += 1;
        }

        let mut solution = initial
            .map(<[f64]>::to_vec)
            .unwrap_or_else(|| vec![0.0; rhs.len()]);
        let residual_started = profile_started::<PROFILE>();
        let (initial_l1, initial_l2_squared) =
            self.update_finest_residual_with_norms(&solution, rhs)?;
        add_profile_elapsed::<PROFILE>(&mut timing.finest_residual_seconds, residual_started);
        if PROFILE {
            timing.finest_residual_evaluations += 1;
        }
        let mut residual_norm = initial_l2_squared.sqrt();
        if controls.l2_controls.min_iterations == 0 && initial_l1 == 0.0 {
            self.has_solved = true;
            return Ok(IterativeSolveReport {
                solution,
                iterations: 0,
                residual_norm,
                converged: true,
                termination: IterativeSolveTermination::Converged,
            });
        }

        if self.options.outer_solver == GamgOuterSolver::FlexibleCg {
            return self.solve_flexible_cg_with_controls_internal::<PROFILE, SCANNED_DIAGONAL, _>(
                FcgLinearSystem { matrix, rhs },
                solution,
                FcgInitialNorms {
                    l1: initial_l1,
                    l2: residual_norm,
                },
                controls.l2_controls,
                timing,
                |current_l1, _current_l2| {
                    normalized_l1_has_converged(current_l1, initial_l1, controls)
                },
            );
        }

        let iteration_limit = controls
            .l2_controls
            .max_iterations
            .max(controls.l2_controls.min_iterations)
            .max(1);
        for iteration in 1..=iteration_limit {
            let cycle_started = profile_started::<PROFILE>();
            self.v_cycle::<PROFILE, SCANNED_DIAGONAL>(
                &mut solution,
                rhs,
                controls.l2_controls,
                timing,
            )?;
            add_profile_elapsed::<PROFILE>(&mut timing.v_cycle_seconds, cycle_started);
            if PROFILE {
                timing.v_cycles += 1;
            }
            let residual_started = profile_started::<PROFILE>();
            let (current_l1, current_l2_squared) =
                self.update_finest_residual_with_norms(&solution, rhs)?;
            add_profile_elapsed::<PROFILE>(&mut timing.finest_residual_seconds, residual_started);
            if PROFILE {
                timing.finest_residual_evaluations += 1;
            }
            residual_norm = current_l2_squared.sqrt();
            if iteration >= controls.l2_controls.min_iterations
                && normalized_l1_has_converged(current_l1, initial_l1, controls)
            {
                self.has_solved = true;
                return Ok(IterativeSolveReport {
                    solution,
                    iterations: iteration,
                    residual_norm,
                    converged: true,
                    termination: IterativeSolveTermination::Converged,
                });
            }
        }

        self.has_solved = true;
        Ok(IterativeSolveReport {
            solution,
            iterations: iteration_limit,
            residual_norm,
            converged: false,
            termination: IterativeSolveTermination::MaxIterations,
        })
    }

    fn solve_flexible_cg_with_controls_internal<
        const PROFILE: bool,
        const SCANNED_DIAGONAL: bool,
        F,
    >(
        &mut self,
        system: FcgLinearSystem<'_>,
        mut solution: Vec<f64>,
        initial_norms: FcgInitialNorms,
        controls: GamgSolveControls,
        timing: &mut GamgKernelTiming,
        mut has_converged: F,
    ) -> Result<IterativeSolveReport>
    where
        F: FnMut(f64, f64) -> bool,
    {
        self.fcg_residual.copy_from_slice(&self.residuals[0]);
        let mut current_l1 = initial_norms.l1;
        let mut residual_norm = initial_norms.l2;
        let iteration_limit = controls.max_iterations.max(controls.min_iterations).max(1);

        for iteration in 1..=iteration_limit {
            let residual = std::mem::take(&mut self.fcg_residual);
            let mut preconditioned = std::mem::take(&mut self.fcg_preconditioned_residual);
            preconditioned.fill(0.0);
            self.residuals[0].copy_from_slice(&residual);
            let cycle_started = profile_started::<PROFILE>();
            let cycle_result = self.v_cycle::<PROFILE, SCANNED_DIAGONAL>(
                &mut preconditioned,
                &residual,
                controls,
                timing,
            );
            self.fcg_residual = residual;
            self.fcg_preconditioned_residual = preconditioned;
            cycle_result?;
            add_profile_elapsed::<PROFILE>(&mut timing.v_cycle_seconds, cycle_started);
            if PROFILE {
                timing.v_cycles += 1;
            }

            validate_fcg_preconditioner_output(&self.fcg_preconditioned_residual)?;

            // Preserve GAMG minIter semantics for an exact-zero residual. The
            // standalone path executes no-op V-cycles until minIter is met;
            // FCG cannot form a non-zero Krylov direction in that state.
            if current_l1 == 0.0 {
                let residual_started = profile_started::<PROFILE>();
                let (raw_l1, squared_l2) =
                    self.update_finest_residual_with_norms(&solution, system.rhs)?;
                add_profile_elapsed::<PROFILE>(
                    &mut timing.finest_residual_seconds,
                    residual_started,
                );
                if PROFILE {
                    timing.finest_residual_evaluations += 1;
                    timing.outer_reductions += 2;
                }
                self.fcg_residual.copy_from_slice(&self.residuals[0]);
                current_l1 = raw_l1;
                residual_norm = squared_l2.sqrt();
                if iteration >= controls.min_iterations && has_converged(current_l1, residual_norm)
                {
                    self.has_solved = true;
                    return Ok(IterativeSolveReport {
                        solution,
                        iterations: iteration,
                        residual_norm,
                        converged: true,
                        termination: IterativeSolveTermination::Converged,
                    });
                }
                continue;
            }

            let previous = (iteration > 1).then_some((
                self.fcg_previous_direction.as_slice(),
                self.fcg_previous_matrix_direction.as_slice(),
            ));
            if fcg_mmax_one_direction(
                &self.fcg_preconditioned_residual,
                previous,
                &mut self.fcg_direction,
            )?
            .is_some()
                && PROFILE
            {
                timing.outer_reductions += 1;
            }

            system
                .matrix
                .matvec_into(&self.fcg_direction, &mut self.fcg_matrix_direction)?;
            if PROFILE {
                timing.outer_matrix_vector_products += 1;
            }
            let step_numerator = dot(&self.fcg_direction, &self.fcg_residual);
            let curvature = dot(&self.fcg_direction, &self.fcg_matrix_direction);
            if PROFILE {
                // Two explicit dot products and four L2 norms in the two
                // scaled singularity checks below.
                timing.outer_reductions += 6;
            }
            if !step_numerator.is_finite() {
                return Err(invalid_input(
                    "GAMG FCG step numerator is not finite".to_string(),
                ));
            }
            if !curvature.is_finite() {
                return Err(invalid_input(
                    "GAMG FCG curvature is not finite; matrix is likely not SPD".to_string(),
                ));
            }
            let numerator_is_singular =
                dot_product_is_singular(step_numerator, &self.fcg_direction, &self.fcg_residual);
            let curvature_is_singular =
                dot_product_is_singular(curvature, &self.fcg_direction, &self.fcg_matrix_direction);
            if step_numerator <= 0.0
                || numerator_is_singular
                || curvature <= 0.0
                || curvature_is_singular
            {
                self.has_solved = true;
                return Ok(IterativeSolveReport {
                    solution,
                    iterations: iteration,
                    residual_norm,
                    converged: false,
                    termination: IterativeSolveTermination::Breakdown,
                });
            }

            let alpha = step_numerator / curvature;
            if !alpha.is_finite() {
                return Err(invalid_input(
                    "GAMG FCG step length is not finite".to_string(),
                ));
            }
            for (value, direction) in solution.iter_mut().zip(&self.fcg_direction) {
                *value += alpha * direction;
                if !value.is_finite() {
                    return Err(invalid_input("GAMG FCG solution is not finite".to_string()));
                }
            }
            let residual_started = profile_started::<PROFILE>();
            let (raw_l1, squared_l2) =
                self.update_finest_residual_with_norms(&solution, system.rhs)?;
            add_profile_elapsed::<PROFILE>(&mut timing.finest_residual_seconds, residual_started);
            if PROFILE {
                timing.finest_residual_evaluations += 1;
                timing.outer_reductions += 2;
            }
            self.fcg_residual.copy_from_slice(&self.residuals[0]);
            current_l1 = raw_l1;
            residual_norm = squared_l2.sqrt();
            if !current_l1.is_finite() || !residual_norm.is_finite() {
                return Err(invalid_input(
                    "GAMG FCG residual norm is not finite".to_string(),
                ));
            }
            if iteration >= controls.min_iterations && has_converged(current_l1, residual_norm) {
                self.has_solved = true;
                return Ok(IterativeSolveReport {
                    solution,
                    iterations: iteration,
                    residual_norm,
                    converged: true,
                    termination: IterativeSolveTermination::Converged,
                });
            }

            self.fcg_previous_direction
                .copy_from_slice(&self.fcg_direction);
            for row in 0..self.fcg_previous_matrix_direction.len() {
                self.fcg_previous_matrix_direction[row] =
                    self.fcg_matrix_direction[row] / curvature;
                if !self.fcg_previous_matrix_direction[row].is_finite() {
                    return Err(invalid_input(
                        "GAMG FCG normalized matrix direction is not finite".to_string(),
                    ));
                }
            }
        }

        self.has_solved = true;
        Ok(IterativeSolveReport {
            solution,
            iterations: iteration_limit,
            residual_norm,
            converged: false,
            termination: IterativeSolveTermination::MaxIterations,
        })
    }

    fn solve_with_controls_internal<const PROFILE: bool, const SCANNED_DIAGONAL: bool>(
        &mut self,
        matrix: &CsrMatrix,
        rhs: &[f64],
        initial: Option<&[f64]>,
        controls: GamgSolveControls,
        timing: &mut GamgKernelTiming,
    ) -> Result<IterativeSolveReport> {
        validate_solve_controls(controls)?;
        validate_iterative_solve_input(matrix, rhs, initial, controls.tolerance)?;
        validate_gamg_matrix(matrix)?;

        if !self.options.cache_agglomeration && self.has_solved {
            let rebuild_started = profile_started::<PROFILE>();
            *self = Self::build(matrix, self.options, self.agglomeration_source.clone())?;
            add_profile_elapsed::<PROFILE>(&mut timing.hierarchy_rebuild_seconds, rebuild_started);
            if PROFILE {
                timing.hierarchy_rebuilds += 1;
            }
        }
        if PROFILE {
            let hierarchy_rebuild_seconds = timing.hierarchy_rebuild_seconds;
            let hierarchy_rebuilds = timing.hierarchy_rebuilds;
            let (hierarchy, hierarchy_diagnostic_seconds) = match &self.profiled_hierarchy {
                Some(hierarchy) => (Arc::clone(hierarchy), 0.0),
                None => {
                    let diagnostic_started = profile_started::<PROFILE>();
                    let hierarchy = Arc::new(build_hierarchy_diagnostics(
                        &self.matrices,
                        &self.transfers,
                        self.options,
                    )?);
                    (hierarchy, profile_elapsed(diagnostic_started))
                }
            };
            *timing = GamgKernelTiming::from_hierarchy(&self.matrices, hierarchy);
            timing.hierarchy_diagnostic_seconds = hierarchy_diagnostic_seconds;
            timing.hierarchy_rebuild_seconds = hierarchy_rebuild_seconds;
            timing.hierarchy_rebuilds = hierarchy_rebuilds;
            timing.solves = 1;
        }

        let refresh_started = profile_started::<PROFILE>();
        self.refresh_matrix_values::<PROFILE>(matrix, timing)?;
        add_profile_elapsed::<PROFILE>(&mut timing.matrix_refresh_seconds, refresh_started);
        if PROFILE {
            timing.matrix_refreshes += 1;
        }

        let mut solution = initial
            .map(<[f64]>::to_vec)
            .unwrap_or_else(|| vec![0.0; rhs.len()]);
        let residual_started = profile_started::<PROFILE>();
        self.update_finest_residual(&solution, rhs)?;
        add_profile_elapsed::<PROFILE>(&mut timing.finest_residual_seconds, residual_started);
        if PROFILE {
            timing.finest_residual_evaluations += 1;
        }
        let initial_residual_norm = l2_norm(&self.residuals[0]);
        let mut residual_norm = initial_residual_norm;
        if controls.min_iterations == 0
            && has_converged(residual_norm, initial_residual_norm, controls)
        {
            self.has_solved = true;
            return Ok(IterativeSolveReport {
                solution,
                iterations: 0,
                residual_norm,
                converged: true,
                termination: IterativeSolveTermination::Converged,
            });
        }

        if self.options.outer_solver == GamgOuterSolver::FlexibleCg {
            let initial_l1 = self.residuals[0]
                .iter()
                .map(|value| value.abs())
                .sum::<f64>();
            return self.solve_flexible_cg_with_controls_internal::<PROFILE, SCANNED_DIAGONAL, _>(
                FcgLinearSystem { matrix, rhs },
                solution,
                FcgInitialNorms {
                    l1: initial_l1,
                    l2: residual_norm,
                },
                controls,
                timing,
                |_current_l1, current_l2| {
                    has_converged(current_l2, initial_residual_norm, controls)
                },
            );
        }

        let iteration_limit = controls.max_iterations.max(controls.min_iterations).max(1);
        for iteration in 1..=iteration_limit {
            let cycle_started = profile_started::<PROFILE>();
            self.v_cycle::<PROFILE, SCANNED_DIAGONAL>(&mut solution, rhs, controls, timing)?;
            add_profile_elapsed::<PROFILE>(&mut timing.v_cycle_seconds, cycle_started);
            if PROFILE {
                timing.v_cycles += 1;
            }
            let residual_started = profile_started::<PROFILE>();
            self.update_finest_residual(&solution, rhs)?;
            add_profile_elapsed::<PROFILE>(&mut timing.finest_residual_seconds, residual_started);
            if PROFILE {
                timing.finest_residual_evaluations += 1;
            }
            residual_norm = l2_norm(&self.residuals[0]);
            if iteration >= controls.min_iterations
                && has_converged(residual_norm, initial_residual_norm, controls)
            {
                self.has_solved = true;
                return Ok(IterativeSolveReport {
                    solution,
                    iterations: iteration,
                    residual_norm,
                    converged: true,
                    termination: IterativeSolveTermination::Converged,
                });
            }
        }

        self.has_solved = true;
        Ok(IterativeSolveReport {
            solution,
            iterations: iteration_limit,
            residual_norm,
            converged: false,
            termination: IterativeSolveTermination::MaxIterations,
        })
    }

    fn refresh_matrix_values<const PROFILE: bool>(
        &mut self,
        matrix: &CsrMatrix,
        timing: &mut GamgKernelTiming,
    ) -> Result<()> {
        if matrix.rows() != self.finest_sparsity.rows()
            || matrix.cols() != self.finest_sparsity.cols()
            || matrix.row_offsets() != self.finest_sparsity.row_offsets()
            || matrix.col_indices() != self.finest_sparsity.col_indices()
        {
            return Err(invalid_input(
                "GAMG workspace does not match matrix sparsity".to_string(),
            ));
        }
        let finest_started = profile_started::<PROFILE>();
        self.matrices[0]
            .values_mut()
            .copy_from_slice(matrix.values());
        refresh_diagonal_values(
            &self.matrices[0],
            &self.diagonal_slots[0],
            &mut self.diagonal_values[0],
        )?;
        if PROFILE {
            timing.levels[0].matrix_refresh_seconds += profile_elapsed(finest_started);
            timing.levels[0].matrix_refreshes += 1;
        }
        for level in 0..self.transfers.len() {
            let level_started = profile_started::<PROFILE>();
            let (fine_levels, coarse_levels) = self.matrices.split_at_mut(level + 1);
            self.transfers[level]
                .agglomerate_values(fine_levels[level].values(), coarse_levels[0].values_mut())?;
            refresh_diagonal_values(
                &self.matrices[level + 1],
                &self.diagonal_slots[level + 1],
                &mut self.diagonal_values[level + 1],
            )?;
            if PROFILE {
                timing.levels[level + 1].matrix_refresh_seconds += profile_elapsed(level_started);
                timing.levels[level + 1].matrix_refreshes += 1;
            }
        }
        Ok(())
    }

    fn update_finest_residual(&mut self, solution: &[f64], rhs: &[f64]) -> Result<()> {
        self.matrices[0].matvec_into(solution, &mut self.products[0])?;
        for ((residual, source), product) in
            self.residuals[0].iter_mut().zip(rhs).zip(&self.products[0])
        {
            *residual = source - product;
        }
        Ok(())
    }

    #[cfg_attr(not(test), allow(dead_code))]
    fn update_finest_residual_with_norms(
        &mut self,
        solution: &[f64],
        rhs: &[f64],
    ) -> Result<(f64, f64)> {
        self.matrices[0].matvec_into(solution, &mut self.products[0])?;
        let mut raw_l1 = 0.0;
        let mut squared_l2 = 0.0;
        for ((residual, source), product) in
            self.residuals[0].iter_mut().zip(rhs).zip(&self.products[0])
        {
            *residual = source - product;
            raw_l1 += residual.abs();
            squared_l2 += *residual * *residual;
        }
        Ok((raw_l1, squared_l2))
    }

    fn v_cycle<const PROFILE: bool, const SCANNED_DIAGONAL: bool>(
        &mut self,
        solution: &mut [f64],
        rhs: &[f64],
        controls: GamgSolveControls,
        timing: &mut GamgKernelTiming,
    ) -> Result<()> {
        let coarsest = self.matrices.len() - 1;
        let restriction_started = profile_started::<PROFILE>();
        self.transfers[0].restrict_sum(&self.residuals[0], &mut self.sources[1])?;
        if PROFILE {
            timing.levels[0].restriction_seconds += profile_elapsed(restriction_started);
            timing.levels[0].restriction_calls += 1;
        }

        for level in 1..coarsest {
            let correction_started = profile_started::<PROFILE>();
            self.corrections[level].fill(0.0);
            if PROFILE {
                timing.levels[level].correction_seconds += profile_elapsed(correction_started);
                timing.levels[level].correction_updates += 1;
            }
            let level_index = level - 1;
            let pre_sweeps = sweep_count(
                self.options.n_pre_sweeps,
                self.options.pre_sweeps_level_multiplier,
                self.options.max_pre_sweeps,
                level_index,
            );
            if pre_sweeps > 0 {
                let smoothing_started = profile_started::<PROFILE>();
                smooth::<SCANNED_DIAGONAL>(
                    &self.matrices[level],
                    &self.diagonal_slots[level],
                    &self.diagonal_values[level],
                    &self.sources[level],
                    &mut self.corrections[level],
                    self.options.smoother,
                    pre_sweeps,
                )?;
                if PROFILE {
                    timing.levels[level].smoothing_seconds += profile_elapsed(smoothing_started);
                    timing.levels[level].smoothing_calls += 1;
                    timing.levels[level].smoothing_sweeps += pre_sweeps;
                }
                if self.options.scale_correction && level < coarsest - 1 {
                    let scaling_started = profile_started::<PROFILE>();
                    scale_correction(
                        &self.matrices[level],
                        &self.diagonal_slots[level],
                        &self.sources[level],
                        &mut self.corrections[level],
                        &mut self.products[level],
                    )?;
                    if PROFILE {
                        timing.levels[level].scaling_seconds += profile_elapsed(scaling_started);
                        timing.levels[level].scaling_calls += 1;
                    }
                }
                let residual_started = profile_started::<PROFILE>();
                self.matrices[level]
                    .matvec_into(&self.corrections[level], &mut self.products[level])?;
                for ((residual, source), product) in self.residuals[level]
                    .iter_mut()
                    .zip(&self.sources[level])
                    .zip(&self.products[level])
                {
                    *residual = source - product;
                }
                if PROFILE {
                    timing.levels[level].residual_seconds += profile_elapsed(residual_started);
                    timing.levels[level].residual_evaluations += 1;
                }
                let restriction_started = profile_started::<PROFILE>();
                self.transfers[level]
                    .restrict_sum(&self.residuals[level], &mut self.sources[level + 1])?;
                if PROFILE {
                    timing.levels[level].restriction_seconds +=
                        profile_elapsed(restriction_started);
                    timing.levels[level].restriction_calls += 1;
                }
            } else {
                let restriction_started = profile_started::<PROFILE>();
                let (fine_sources, coarse_sources) = self.sources.split_at_mut(level + 1);
                self.transfers[level].restrict_sum(&fine_sources[level], &mut coarse_sources[0])?;
                if PROFILE {
                    timing.levels[level].restriction_seconds +=
                        profile_elapsed(restriction_started);
                    timing.levels[level].restriction_calls += 1;
                }
            }
        }

        let coarsest_started = profile_started::<PROFILE>();
        let coarsest_iterations = self.solve_coarsest_level(coarsest, controls)?;
        if PROFILE {
            timing.levels[coarsest].coarsest_solve_seconds += profile_elapsed(coarsest_started);
            timing.levels[coarsest].coarsest_solves += 1;
            timing.levels[coarsest].coarsest_iterations += coarsest_iterations;
        }

        for level in (1..coarsest).rev() {
            let level_index = level - 1;
            if self.options.n_pre_sweeps > 0 {
                let correction_started = profile_started::<PROFILE>();
                self.pre_smoothed[level].copy_from_slice(&self.corrections[level]);
                if PROFILE {
                    timing.levels[level].correction_seconds += profile_elapsed(correction_started);
                    timing.levels[level].correction_updates += 1;
                }
            }
            let prolongation_started = profile_started::<PROFILE>();
            let (fine_corrections, coarse_corrections) = self.corrections.split_at_mut(level + 1);
            self.transfers[level]
                .prolong_injection(&coarse_corrections[0], &mut fine_corrections[level])?;
            if self.options.interpolate_correction {
                let (fine_products, coarse_products) = self.products.split_at_mut(level + 1);
                let (_, coarse_residuals) = self.residuals.split_at_mut(level + 1);
                self.transfers[level].interpolate_correction(
                    &self.matrices[level],
                    &self.diagonal_values[level],
                    &coarse_corrections[0],
                    &mut fine_corrections[level],
                    GamgInterpolationScratch {
                        fine: &mut fine_products[level],
                        coarse_correction: &mut coarse_residuals[0],
                        coarse_diagonal: &mut coarse_products[0],
                    },
                )?;
            }
            if PROFILE {
                timing.levels[level].prolongation_seconds += profile_elapsed(prolongation_started);
                timing.levels[level].prolongation_calls += 1;
            }
            if self.options.scale_correction
                && (self.options.interpolate_correction || level < coarsest - 1)
            {
                let scaling_started = profile_started::<PROFILE>();
                scale_correction(
                    &self.matrices[level],
                    &self.diagonal_slots[level],
                    &self.sources[level],
                    &mut self.corrections[level],
                    &mut self.products[level],
                )?;
                if PROFILE {
                    timing.levels[level].scaling_seconds += profile_elapsed(scaling_started);
                    timing.levels[level].scaling_calls += 1;
                }
            }
            if self.options.n_pre_sweeps > 0 {
                let correction_started = profile_started::<PROFILE>();
                for (correction, pre_smoothed) in self.corrections[level]
                    .iter_mut()
                    .zip(&self.pre_smoothed[level])
                {
                    *correction += pre_smoothed;
                }
                if PROFILE {
                    timing.levels[level].correction_seconds += profile_elapsed(correction_started);
                    timing.levels[level].correction_updates += 1;
                }
            }
            let post_sweeps = sweep_count(
                self.options.n_post_sweeps,
                self.options.post_sweeps_level_multiplier,
                self.options.max_post_sweeps,
                level_index,
            );
            let smoothing_started = profile_started::<PROFILE>();
            smooth::<SCANNED_DIAGONAL>(
                &self.matrices[level],
                &self.diagonal_slots[level],
                &self.diagonal_values[level],
                &self.sources[level],
                &mut self.corrections[level],
                self.options.smoother,
                post_sweeps,
            )?;
            if PROFILE {
                timing.levels[level].smoothing_seconds += profile_elapsed(smoothing_started);
                timing.levels[level].smoothing_calls += 1;
                timing.levels[level].smoothing_sweeps += post_sweeps;
            }
        }

        let prolongation_started = profile_started::<PROFILE>();
        let (finest_correction, coarse_corrections) = self.corrections.split_at_mut(1);
        self.transfers[0].prolong_injection(&coarse_corrections[0], &mut finest_correction[0])?;
        if self.options.interpolate_correction {
            let (finest_products, coarse_products) = self.products.split_at_mut(1);
            let (_, coarse_residuals) = self.residuals.split_at_mut(1);
            self.transfers[0].interpolate_correction(
                &self.matrices[0],
                &self.diagonal_values[0],
                &coarse_corrections[0],
                &mut finest_correction[0],
                GamgInterpolationScratch {
                    fine: &mut finest_products[0],
                    coarse_correction: &mut coarse_residuals[0],
                    coarse_diagonal: &mut coarse_products[0],
                },
            )?;
        }
        if PROFILE {
            timing.levels[0].prolongation_seconds += profile_elapsed(prolongation_started);
            timing.levels[0].prolongation_calls += 1;
        }
        if self.options.scale_correction {
            let scaling_started = profile_started::<PROFILE>();
            scale_correction(
                &self.matrices[0],
                &self.diagonal_slots[0],
                &self.residuals[0],
                &mut self.corrections[0],
                &mut self.products[0],
            )?;
            if PROFILE {
                timing.levels[0].scaling_seconds += profile_elapsed(scaling_started);
                timing.levels[0].scaling_calls += 1;
            }
        }
        let correction_started = profile_started::<PROFILE>();
        for (value, correction) in solution.iter_mut().zip(&self.corrections[0]) {
            *value += correction;
        }
        if PROFILE {
            timing.levels[0].correction_seconds += profile_elapsed(correction_started);
            timing.levels[0].correction_updates += 1;
        }
        let smoothing_started = profile_started::<PROFILE>();
        let result = smooth::<SCANNED_DIAGONAL>(
            &self.matrices[0],
            &self.diagonal_slots[0],
            &self.diagonal_values[0],
            rhs,
            solution,
            self.options.smoother,
            self.options.n_finest_sweeps,
        );
        if PROFILE {
            timing.levels[0].smoothing_seconds += profile_elapsed(smoothing_started);
            timing.levels[0].smoothing_calls += 1;
            timing.levels[0].smoothing_sweeps += self.options.n_finest_sweeps;
        }
        result
    }

    fn solve_coarsest_level(
        &mut self,
        coarsest: usize,
        controls: GamgSolveControls,
    ) -> Result<usize> {
        if self.options.direct_solve_coarsest {
            dense_lu_solve(
                &self.matrices[coarsest],
                &self.sources[coarsest],
                &mut self.corrections[coarsest],
            )?;
            Ok(0)
        } else {
            let initial_norm = l2_norm(&self.sources[coarsest]);
            let tolerance = controls
                .tolerance
                .max(controls.relative_tolerance * initial_norm);
            let report = self
                .coarsest_pcg
                .as_mut()
                .expect("iterative GAMG coarsest solver has a PCG workspace")
                .solve(
                    &self.matrices[coarsest],
                    &self.sources[coarsest],
                    None,
                    PreconditionedConjugateGradientOptions {
                        max_iterations: COARSEST_MAX_ITERATIONS,
                        tolerance,
                        preconditioner: CgPreconditioner::IncompleteCholesky,
                    },
                )?;
            let iterations = report.iterations;
            self.corrections[coarsest].copy_from_slice(&report.solution);
            Ok(iterations)
        }
    }
}

pub fn gamg_solve(
    matrix: &CsrMatrix,
    rhs: &[f64],
    initial: Option<&[f64]>,
    options: GamgOptions,
) -> Result<IterativeSolveReport> {
    let mut workspace = GamgWorkspace::new(matrix, options)?;
    workspace.solve(matrix, rhs, initial)
}

fn validate_options(options: GamgOptions) -> Result<()> {
    validate_solve_controls(options.into())?;
    if options.n_cells_in_coarsest_level == 0 {
        return Err(invalid_input(
            "GAMG nCellsInCoarsestLevel must be positive".to_string(),
        ));
    }
    if options.merge_levels == 0 {
        return Err(invalid_input(
            "GAMG mergeLevels must be positive".to_string(),
        ));
    }
    Ok(())
}

fn validate_solve_controls(controls: GamgSolveControls) -> Result<()> {
    if !controls.tolerance.is_finite() || controls.tolerance < 0.0 {
        return Err(invalid_input(format!(
            "GAMG tolerance must be finite and non-negative, got {}",
            controls.tolerance
        )));
    }
    if !controls.relative_tolerance.is_finite() || controls.relative_tolerance < 0.0 {
        return Err(invalid_input(format!(
            "GAMG relTol must be finite and non-negative, got {}",
            controls.relative_tolerance
        )));
    }
    Ok(())
}

#[cfg_attr(not(test), allow(dead_code))]
fn validate_normalized_l1_call(
    matrix: &CsrMatrix,
    rhs: &[f64],
    initial: Option<&[f64]>,
    controls: NormalizedL1GamgSolveControls,
) -> Result<Vec<usize>> {
    if !controls.normalization_factor.is_finite() || controls.normalization_factor <= 0.0 {
        return Err(invalid_input(format!(
            "GAMG normalized-L1 factor must be finite and positive, got {}",
            controls.normalization_factor
        )));
    }
    if !controls.tolerance.is_finite() || controls.tolerance < 0.0 {
        return Err(invalid_input(format!(
            "GAMG normalized-L1 tolerance must be finite and non-negative, got {}",
            controls.tolerance
        )));
    }
    if !controls.relative_tolerance.is_finite() || controls.relative_tolerance < 0.0 {
        return Err(invalid_input(format!(
            "GAMG normalized-L1 relTol must be finite and non-negative, got {}",
            controls.relative_tolerance
        )));
    }
    validate_solve_controls(controls.l2_controls)?;
    if rhs.len() != matrix.rows() {
        return Err(invalid_input(format!(
            "iterative solve expected rhs with {} entries, got {}",
            matrix.rows(),
            rhs.len()
        )));
    }
    if let Some(initial) = initial
        && initial.len() != matrix.cols()
    {
        return Err(invalid_input(format!(
            "iterative solve expected initial guess with {} entries, got {}",
            matrix.cols(),
            initial.len()
        )));
    }
    if let Some((index, value)) = rhs
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(invalid_input(format!(
            "iterative solve rhs entry {index} must be finite, got {value}"
        )));
    }
    if let Some((index, value)) = initial.and_then(|values| {
        values
            .iter()
            .copied()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
    }) {
        return Err(invalid_input(format!(
            "iterative solve initial entry {index} must be finite, got {value}"
        )));
    }
    validate_gamg_matrix_with_diagonal_slots(matrix)
}

fn validate_gamg_matrix(matrix: &CsrMatrix) -> Result<()> {
    if matrix.rows() != matrix.cols() {
        return Err(invalid_input(format!(
            "GAMG pressure foundation requires a square matrix, got {}x{}",
            matrix.rows(),
            matrix.cols()
        )));
    }
    if matrix.rows() == 0 {
        return Err(invalid_input(
            "GAMG pressure foundation requires at least one matrix row".to_string(),
        ));
    }
    let mut entries = BTreeMap::<(usize, usize), f64>::new();
    let mut diagonal_counts = vec![0usize; matrix.rows()];
    for (row, diagonal_count) in diagonal_counts.iter_mut().enumerate() {
        for entry in matrix.row_offsets()[row]..matrix.row_offsets()[row + 1] {
            if matrix.col_indices()[entry] == row {
                *diagonal_count += 1;
            }
            *entries
                .entry((row, matrix.col_indices()[entry]))
                .or_default() += matrix.values()[entry];
        }
    }
    for (row, &diagonal_count) in diagonal_counts.iter().enumerate() {
        if diagonal_count != 1 {
            return Err(invalid_input(format!(
                "GAMG row {row} must have exactly one diagonal entry, got {}",
                diagonal_count
            )));
        }
        let diagonal = entries.get(&(row, row)).copied().unwrap_or_default();
        if !diagonal.is_finite() || diagonal == 0.0 {
            return Err(invalid_input(format!(
                "GAMG row {row} has invalid diagonal value {diagonal}"
            )));
        }
    }
    for (&(row, column), &value) in &entries {
        if row == column {
            continue;
        }
        let transpose = entries.get(&(column, row)).copied().unwrap_or_default();
        let scale = value.abs().max(transpose.abs()).max(1.0);
        if (value - transpose).abs() > 64.0 * f64::EPSILON * scale {
            return Err(invalid_input(format!(
                "GAMG pressure foundation requires a symmetric matrix; A[{row},{column}]={value} differs from A[{column},{row}]={transpose}"
            )));
        }
    }
    Ok(())
}

fn validate_gamg_matrix_with_diagonal_slots(matrix: &CsrMatrix) -> Result<Vec<usize>> {
    if matrix.rows() != matrix.cols() {
        return Err(invalid_input(format!(
            "GAMG pressure foundation requires a square matrix, got {}x{}",
            matrix.rows(),
            matrix.cols()
        )));
    }
    if matrix.rows() == 0 {
        return Err(invalid_input(
            "GAMG pressure foundation requires at least one matrix row".to_string(),
        ));
    }
    let mut entries = BTreeMap::<(usize, usize), f64>::new();
    let mut diagonal_counts = vec![0usize; matrix.rows()];
    let mut diagonal_slots = vec![0usize; matrix.rows()];
    for (row, diagonal_count) in diagonal_counts.iter_mut().enumerate() {
        for entry in matrix.row_offsets()[row]..matrix.row_offsets()[row + 1] {
            if matrix.col_indices()[entry] == row {
                *diagonal_count += 1;
                diagonal_slots[row] = entry;
            }
            *entries
                .entry((row, matrix.col_indices()[entry]))
                .or_default() += matrix.values()[entry];
        }
    }
    for (row, &diagonal_count) in diagonal_counts.iter().enumerate() {
        if diagonal_count != 1 {
            return Err(invalid_input(format!(
                "GAMG row {row} must have exactly one diagonal entry, got {}",
                diagonal_count
            )));
        }
        let diagonal = entries.get(&(row, row)).copied().unwrap_or_default();
        if !diagonal.is_finite() || diagonal == 0.0 {
            return Err(invalid_input(format!(
                "GAMG row {row} has invalid diagonal value {diagonal}"
            )));
        }
    }
    for (&(row, column), &value) in &entries {
        if row == column {
            continue;
        }
        let transpose = entries.get(&(column, row)).copied().unwrap_or_default();
        let scale = value.abs().max(transpose.abs()).max(1.0);
        if (value - transpose).abs() > 64.0 * f64::EPSILON * scale {
            return Err(invalid_input(format!(
                "GAMG pressure foundation requires a symmetric matrix; A[{row},{column}]={value} differs from A[{column},{row}]={transpose}"
            )));
        }
    }
    Ok(diagonal_slots)
}

fn level_vectors(level_sizes: &[usize]) -> Vec<Vec<f64>> {
    level_sizes.iter().map(|&size| vec![0.0; size]).collect()
}

fn has_converged(residual: f64, initial: f64, controls: GamgSolveControls) -> bool {
    residual <= controls.tolerance
        || (controls.relative_tolerance > 0.0 && residual <= controls.relative_tolerance * initial)
}

#[cfg_attr(not(test), allow(dead_code))]
fn normalized_l1_has_converged(
    current_l1: f64,
    initial_l1: f64,
    controls: NormalizedL1GamgSolveControls,
) -> bool {
    current_l1 / controls.normalization_factor < controls.tolerance
        || (controls.relative_tolerance > OPENFOAM_RELATIVE_TOLERANCE_SMALL
            && current_l1 / controls.normalization_factor
                < controls.relative_tolerance * (initial_l1 / controls.normalization_factor))
}

#[inline]
fn profile_started<const PROFILE: bool>() -> Option<Instant> {
    if PROFILE { Some(Instant::now()) } else { None }
}

#[inline]
fn profile_elapsed(started: Option<Instant>) -> f64 {
    started
        .map(|started| started.elapsed().as_secs_f64())
        .unwrap_or(0.0)
}

#[inline]
fn add_profile_elapsed<const PROFILE: bool>(target: &mut f64, started: Option<Instant>) {
    if PROFILE {
        *target += profile_elapsed(started);
    }
}

fn sweep_count(base: usize, multiplier: usize, maximum: usize, level: usize) -> usize {
    base.saturating_add(multiplier.saturating_mul(level))
        .min(maximum)
}

fn refresh_diagonal_values(
    matrix: &CsrMatrix,
    diagonal_slots: &[usize],
    diagonal_values: &mut [f64],
) -> Result<()> {
    debug_assert_eq!(diagonal_slots.len(), matrix.rows());
    debug_assert_eq!(diagonal_values.len(), matrix.rows());
    for row in 0..matrix.rows() {
        let diagonal = matrix.values()[diagonal_slots[row]];
        if !diagonal.is_finite() || diagonal == 0.0 {
            return Err(invalid_input(format!(
                "GAMG row {row} has invalid diagonal value {diagonal}"
            )));
        }
        diagonal_values[row] = diagonal;
    }
    Ok(())
}

fn smooth<const SCANNED_DIAGONAL: bool>(
    matrix: &CsrMatrix,
    diagonal_slots: &[usize],
    diagonal_values: &[f64],
    rhs: &[f64],
    solution: &mut [f64],
    smoother: GamgSmoother,
    sweeps: usize,
) -> Result<()> {
    for _ in 0..sweeps {
        if SCANNED_DIAGONAL {
            #[cfg(test)]
            scanned_diagonal_sweep(matrix, rhs, solution, 0..matrix.rows())?;
            #[cfg(not(test))]
            unreachable!("scanned diagonal sweeps are a test oracle");
        } else {
            gauss_seidel_sweep_with_cached_diagonal(
                matrix,
                diagonal_slots,
                diagonal_values,
                rhs,
                solution,
                0..matrix.rows(),
            )?;
        }
        if smoother == GamgSmoother::SymGaussSeidel {
            if SCANNED_DIAGONAL {
                #[cfg(test)]
                scanned_diagonal_sweep(matrix, rhs, solution, (0..matrix.rows()).rev())?;
                #[cfg(not(test))]
                unreachable!("scanned diagonal sweeps are a test oracle");
            } else {
                gauss_seidel_sweep_with_cached_diagonal(
                    matrix,
                    diagonal_slots,
                    diagonal_values,
                    rhs,
                    solution,
                    (0..matrix.rows()).rev(),
                )?;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
fn scanned_diagonal_sweep(
    matrix: &CsrMatrix,
    rhs: &[f64],
    solution: &mut [f64],
    rows: impl IntoIterator<Item = usize>,
) -> Result<()> {
    for row in rows {
        let start = matrix.row_offsets()[row];
        let end = matrix.row_offsets()[row + 1];
        let mut diagonal = None;
        let mut off_diagonal_sum = 0.0;
        for entry in start..end {
            let column = matrix.col_indices()[entry];
            let value = matrix.values()[entry];
            if column == row {
                diagonal = Some(value);
            } else {
                off_diagonal_sum += value * solution[column];
            }
        }
        let diagonal = diagonal.ok_or_else(|| {
            invalid_input(format!("row {row} has no diagonal entry for Gauss-Seidel"))
        })?;
        if !diagonal.is_finite() || diagonal == 0.0 {
            return Err(invalid_input(format!(
                "row {row} has invalid Gauss-Seidel diagonal value {diagonal}"
            )));
        }
        let raw = (rhs[row] - off_diagonal_sum) / diagonal;
        if !raw.is_finite() {
            return Err(invalid_input(format!(
                "Gauss-Seidel update for row {row} is not finite"
            )));
        }
        solution[row] = raw;
    }
    Ok(())
}

fn scale_correction(
    matrix: &CsrMatrix,
    diagonal_slots: &[usize],
    source: &[f64],
    correction: &mut [f64],
    product: &mut [f64],
) -> Result<()> {
    matrix.matvec_into(correction, product)?;
    let numerator = dot(source, correction);
    let denominator = dot(product, correction);
    let stabilised_denominator = if denominator.abs() < SCALE_STABILISER {
        if denominator.is_sign_negative() {
            -SCALE_STABILISER
        } else {
            SCALE_STABILISER
        }
    } else {
        denominator
    };
    let factor = numerator / stabilised_denominator;
    if !factor.is_finite() {
        return Err(invalid_input(
            "GAMG correction scaling factor is not finite".to_string(),
        ));
    }
    for row in 0..correction.len() {
        let diagonal = matrix.values()[diagonal_slots[row]];
        correction[row] =
            factor * correction[row] + (source[row] - factor * product[row]) / diagonal;
        if !correction[row].is_finite() {
            return Err(invalid_input(format!(
                "GAMG scaled correction at row {row} is not finite"
            )));
        }
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct PairEdge {
    lower: usize,
    upper: usize,
    weight: f64,
}

fn algebraic_pair_map(matrix: &CsrMatrix, forward: bool) -> Result<(Vec<usize>, usize)> {
    let mut weights = BTreeMap::<(usize, usize), f64>::new();
    for row in 0..matrix.rows() {
        for entry in matrix.row_offsets()[row]..matrix.row_offsets()[row + 1] {
            let column = matrix.col_indices()[entry];
            if row == column {
                continue;
            }
            let pair = if row < column {
                (row, column)
            } else {
                (column, row)
            };
            let weight = matrix.values()[entry].abs();
            weights
                .entry(pair)
                .and_modify(|current| *current = current.max(weight))
                .or_insert(weight);
        }
    }
    let edges = weights
        .into_iter()
        .map(|((lower, upper), weight)| PairEdge {
            lower,
            upper,
            weight,
        })
        .collect::<Vec<_>>();
    pair_map_from_edges(matrix.rows(), &edges, forward)
}

fn face_pair_edges(
    matrix: &CsrMatrix,
    face_weights: &[GamgFacePairWeight],
) -> Result<Vec<PairEdge>> {
    let matrix_pairs = matrix_connection_pairs(matrix);
    let mut weighted_pairs = BTreeSet::new();
    let mut edges = Vec::with_capacity(face_weights.len());
    for (index, connection) in face_weights.iter().copied().enumerate() {
        let (first_cell, second_cell) = connection.cells();
        if first_cell >= matrix.rows() || second_cell >= matrix.rows() {
            return Err(invalid_input(format!(
                "GAMG faceAreaPair connection {index} uses cells {first_cell} and {second_cell}, but the matrix has {} rows",
                matrix.rows()
            )));
        }
        let pair = ordered_pair(first_cell, second_cell);
        if !matrix_pairs.contains(&pair) {
            return Err(invalid_input(format!(
                "GAMG faceAreaPair connection {index} for cells {} and {} is absent from the matrix sparsity",
                pair.0, pair.1
            )));
        }
        weighted_pairs.insert(pair);
        edges.push(PairEdge {
            lower: pair.0,
            upper: pair.1,
            weight: connection.weight(),
        });
    }
    if let Some((lower, upper)) = matrix_pairs.difference(&weighted_pairs).next() {
        return Err(invalid_input(format!(
            "GAMG faceAreaPair has no mesh weight for matrix connection {lower}-{upper}"
        )));
    }
    Ok(edges)
}

fn matrix_connection_pairs(matrix: &CsrMatrix) -> BTreeSet<(usize, usize)> {
    let mut pairs = BTreeSet::new();
    for row in 0..matrix.rows() {
        for entry in matrix.row_offsets()[row]..matrix.row_offsets()[row + 1] {
            let column = matrix.col_indices()[entry];
            if row != column {
                pairs.insert(ordered_pair(row, column));
            }
        }
    }
    pairs
}

fn ordered_pair(first: usize, second: usize) -> (usize, usize) {
    if first < second {
        (first, second)
    } else {
        (second, first)
    }
}

fn agglomerate_pair_edges(
    fine_edges: &[PairEdge],
    fine_to_coarse: &[usize],
    n_coarse_cells: usize,
) -> Vec<PairEdge> {
    let mut coarse_edges = Vec::<PairEdge>::new();
    let mut coarse_slots = BTreeMap::<(usize, usize), usize>::new();
    for edge in fine_edges {
        let first = fine_to_coarse[edge.lower];
        let second = fine_to_coarse[edge.upper];
        if first == second {
            continue;
        }
        let pair = ordered_pair(first, second);
        if let Some(&slot) = coarse_slots.get(&pair) {
            coarse_edges[slot].weight += edge.weight;
        } else {
            coarse_slots.insert(pair, coarse_edges.len());
            coarse_edges.push(PairEdge {
                lower: pair.0,
                upper: pair.1,
                weight: edge.weight,
            });
        }
    }
    debug_assert!(
        coarse_edges
            .iter()
            .all(|edge| edge.lower < n_coarse_cells && edge.upper < n_coarse_cells)
    );
    coarse_edges
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ExternalNeighbourCountMethod {
    MembershipIntersection,
    UnionScan,
}

fn external_neighbour_count(
    lower: usize,
    upper: usize,
    lower_neighbours: &BTreeSet<usize>,
    upper_neighbours: &BTreeSet<usize>,
) -> (usize, ExternalNeighbourCountMethod) {
    let (smaller, larger) = if lower_neighbours.len() <= upper_neighbours.len() {
        (lower_neighbours, upper_neighbours)
    } else {
        (upper_neighbours, lower_neighbours)
    };
    let membership_steps = smaller.len().saturating_mul(
        larger
            .len()
            .max(1)
            .ilog2()
            .saturating_add(1)
            .try_into()
            .unwrap_or(usize::MAX),
    );
    let union_steps = lower_neighbours
        .len()
        .saturating_add(upper_neighbours.len());

    if membership_steps < union_steps {
        let shared_neighbours = smaller
            .iter()
            .filter(|neighbour| larger.contains(neighbour))
            .count();
        (
            lower_neighbours.len() + upper_neighbours.len() - shared_neighbours - 2,
            ExternalNeighbourCountMethod::MembershipIntersection,
        )
    } else {
        (
            lower_neighbours
                .union(upper_neighbours)
                .filter(|&&cell| cell != lower && cell != upper)
                .count(),
            ExternalNeighbourCountMethod::UnionScan,
        )
    }
}

fn pair_map_from_edges(
    n_cells: usize,
    edges: &[PairEdge],
    forward: bool,
) -> Result<(Vec<usize>, usize)> {
    let mut cell_edges = vec![Vec::<usize>::new(); n_cells];
    let mut cell_neighbours = vec![BTreeSet::<usize>::new(); n_cells];
    for (edge_index, edge) in edges.iter().enumerate() {
        if edge.lower >= n_cells || edge.upper >= n_cells || edge.lower == edge.upper {
            return Err(invalid_input(format!(
                "GAMG pair edge {edge_index} has invalid cells {} and {} for {n_cells} rows",
                edge.lower, edge.upper
            )));
        }
        if !edge.weight.is_finite() || edge.weight <= 0.0 {
            return Err(invalid_input(format!(
                "GAMG pair edge {edge_index} has invalid weight {}",
                edge.weight
            )));
        }
        cell_edges[edge.lower].push(edge_index);
        cell_edges[edge.upper].push(edge_index);
        cell_neighbours[edge.lower].insert(edge.upper);
        cell_neighbours[edge.upper].insert(edge.lower);
    }
    let external_neighbour_counts = edges
        .iter()
        .map(|edge| {
            let lower_neighbours = &cell_neighbours[edge.lower];
            let upper_neighbours = &cell_neighbours[edge.upper];
            external_neighbour_count(edge.lower, edge.upper, lower_neighbours, upper_neighbours).0
        })
        .collect::<Vec<_>>();

    let mut coarse_map = vec![usize::MAX; n_cells];
    let mut n_coarse = 0usize;
    for offset in 0..n_cells {
        let cell = if forward {
            offset
        } else {
            n_cells - offset - 1
        };
        if coarse_map[cell] != usize::MAX {
            continue;
        }

        let mut match_edge = None;
        for &edge_index in &cell_edges[cell] {
            let edge = edges[edge_index];
            if coarse_map[edge.lower] == usize::MAX
                && coarse_map[edge.upper] == usize::MAX
                && match_edge.is_none_or(|current| {
                    pair_edge_is_preferred(
                        edge_index,
                        current,
                        edges,
                        &external_neighbour_counts,
                        forward,
                    )
                })
            {
                match_edge = Some(edge_index);
            }
        }

        if let Some(edge_index) = match_edge {
            let edge = edges[edge_index];
            coarse_map[edge.lower] = n_coarse;
            coarse_map[edge.upper] = n_coarse;
            n_coarse += 1;
        } else {
            let mut cluster_edge: Option<usize> = None;
            for &edge_index in &cell_edges[cell] {
                let edge = edges[edge_index];
                if cluster_edge.is_none_or(|current| {
                    edge.weight > edges[current].weight
                        || (edge.weight == edges[current].weight
                            && pair_endpoints_are_preferred(edge, edges[current], forward))
                }) {
                    cluster_edge = Some(edge_index);
                }
            }
            if let Some(edge_index) = cluster_edge {
                let edge = edges[edge_index];
                let neighbour = if edge.lower == cell {
                    edge.upper
                } else {
                    edge.lower
                };
                let cluster = coarse_map[neighbour];
                if cluster != usize::MAX {
                    coarse_map[cell] = cluster;
                }
            }
        }
    }

    for offset in 0..n_cells {
        let cell = if forward {
            offset
        } else {
            n_cells - offset - 1
        };
        if coarse_map[cell] == usize::MAX {
            coarse_map[cell] = n_coarse;
            n_coarse += 1;
        }
    }
    if !forward {
        for coarse_cell in &mut coarse_map {
            *coarse_cell = n_coarse - 1 - *coarse_cell;
        }
    }
    Ok((coarse_map, n_coarse))
}

fn pair_edge_is_preferred(
    candidate: usize,
    current: usize,
    edges: &[PairEdge],
    external_neighbour_counts: &[usize],
    forward: bool,
) -> bool {
    let candidate_edge = edges[candidate];
    let current_edge = edges[current];
    candidate_edge.weight > current_edge.weight
        || (candidate_edge.weight == current_edge.weight
            && (external_neighbour_counts[candidate] < external_neighbour_counts[current]
                || (external_neighbour_counts[candidate] == external_neighbour_counts[current]
                    && pair_endpoints_are_preferred(candidate_edge, current_edge, forward))))
}

fn pair_endpoints_are_preferred(candidate: PairEdge, current: PairEdge, forward: bool) -> bool {
    let candidate_pair = ordered_pair(candidate.lower, candidate.upper);
    let current_pair = ordered_pair(current.lower, current.upper);
    if forward {
        candidate_pair < current_pair
    } else {
        candidate_pair > current_pair
    }
}

fn build_coarse_matrix(
    fine: &CsrMatrix,
    fine_to_coarse: Vec<usize>,
    n_coarse: usize,
) -> Result<(GamgTransfer, CsrMatrix)> {
    let mut coarse_columns = vec![BTreeMap::<usize, usize>::new(); n_coarse];
    for fine_row in 0..fine.rows() {
        let coarse_row = fine_to_coarse[fine_row];
        for entry in fine.row_offsets()[fine_row]..fine.row_offsets()[fine_row + 1] {
            let coarse_column = fine_to_coarse[fine.col_indices()[entry]];
            coarse_columns[coarse_row].insert(coarse_column, 0);
        }
    }
    for (row, columns) in coarse_columns.iter_mut().enumerate() {
        columns.insert(row, 0);
    }

    let mut row_offsets = Vec::with_capacity(n_coarse + 1);
    let mut col_indices = Vec::new();
    row_offsets.push(0);
    for columns in &coarse_columns {
        col_indices.extend(columns.keys().copied());
        row_offsets.push(col_indices.len());
    }
    let pattern = CsrSparsityPattern::new(n_coarse, n_coarse, row_offsets, col_indices)?;
    let mut slot_lookup = vec![BTreeMap::<usize, usize>::new(); n_coarse];
    for (row, lookup) in slot_lookup.iter_mut().enumerate() {
        for slot in pattern.row_offsets()[row]..pattern.row_offsets()[row + 1] {
            lookup.insert(pattern.col_indices()[slot], slot);
        }
    }

    let mut fine_entry_to_coarse_entry = Vec::with_capacity(fine.nnz());
    for fine_row in 0..fine.rows() {
        let coarse_row = fine_to_coarse[fine_row];
        for entry in fine.row_offsets()[fine_row]..fine.row_offsets()[fine_row + 1] {
            let coarse_column = fine_to_coarse[fine.col_indices()[entry]];
            fine_entry_to_coarse_entry.push(
                *slot_lookup[coarse_row]
                    .get(&coarse_column)
                    .expect("GAMG coarse slot was created from every fine entry"),
            );
        }
    }
    let transfer = GamgTransfer {
        fine_to_coarse,
        fine_entry_to_coarse_entry,
    };
    let mut coarse = CsrMatrix::from_pattern(&pattern, vec![0.0; pattern.nnz()])?;
    transfer.agglomerate_values(fine.values(), coarse.values_mut())?;
    Ok((transfer, coarse))
}

fn dense_lu_solve(matrix: &CsrMatrix, rhs: &[f64], solution: &mut [f64]) -> Result<()> {
    let n = matrix.rows();
    if rhs.len() != n || solution.len() != n {
        return Err(invalid_input(format!(
            "GAMG direct coarsest solve expected {n} entries, got rhs={} solution={}",
            rhs.len(),
            solution.len()
        )));
    }
    let dense_len = checked_dense_storage_len(n)?;
    let mut dense = Vec::new();
    dense.try_reserve_exact(dense_len).map_err(|_| {
        invalid_input(format!(
            "GAMG direct coarsest solve could not allocate dense storage for {n} rows"
        ))
    })?;
    dense.resize(dense_len, 0.0);
    for row in 0..n {
        for entry in matrix.row_offsets()[row]..matrix.row_offsets()[row + 1] {
            dense[row * n + matrix.col_indices()[entry]] += matrix.values()[entry];
        }
    }
    solution.copy_from_slice(rhs);

    for pivot_column in 0..n {
        let mut pivot_row = pivot_column;
        let mut pivot_magnitude = dense[pivot_column * n + pivot_column].abs();
        for row in pivot_column + 1..n {
            let magnitude = dense[row * n + pivot_column].abs();
            if magnitude > pivot_magnitude {
                pivot_row = row;
                pivot_magnitude = magnitude;
            }
        }
        if !pivot_magnitude.is_finite() || pivot_magnitude <= f64::EPSILON {
            return Err(invalid_input(format!(
                "GAMG direct coarsest solve has a singular pivot in column {pivot_column}"
            )));
        }
        if pivot_row != pivot_column {
            for column in 0..n {
                dense.swap(pivot_column * n + column, pivot_row * n + column);
            }
            solution.swap(pivot_column, pivot_row);
        }
        let pivot = dense[pivot_column * n + pivot_column];
        for row in pivot_column + 1..n {
            let factor = dense[row * n + pivot_column] / pivot;
            dense[row * n + pivot_column] = 0.0;
            for column in pivot_column + 1..n {
                dense[row * n + column] -= factor * dense[pivot_column * n + column];
            }
            solution[row] -= factor * solution[pivot_column];
        }
    }

    for row in (0..n).rev() {
        let mut value = solution[row];
        for column in row + 1..n {
            value -= dense[row * n + column] * solution[column];
        }
        solution[row] = value / dense[row * n + row];
        if !solution[row].is_finite() {
            return Err(invalid_input(format!(
                "GAMG direct coarsest solution at row {row} is not finite"
            )));
        }
    }
    Ok(())
}

fn checked_dense_storage_len(n: usize) -> Result<usize> {
    let dense_len = n.checked_mul(n).ok_or_else(|| {
        invalid_input(format!(
            "GAMG direct coarsest dense storage size overflow for {n} rows"
        ))
    })?;
    if n > MAX_DENSE_COARSEST_CELLS {
        return Err(invalid_input(format!(
            "GAMG direct coarsest solve supports at most {MAX_DENSE_COARSEST_CELLS} actual coarsest cells, got {n}"
        )));
    }
    Ok(dense_len)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::{
        ExternalNeighbourCountMethod, GamgAgglomerator, GamgFacePairWeight,
        GamgInterpolationScratch, GamgKernelTiming, GamgOptions, GamgOuterSolver, GamgSmoother,
        GamgSolveControls, GamgTransfer, GamgWorkspace, MAX_DENSE_COARSEST_CELLS,
        NormalizedL1GamgSolveControls, PairEdge, algebraic_pair_map, checked_dense_storage_len,
        dense_lu_solve, external_neighbour_count, fcg_mmax_one_direction, gamg_solve,
        pair_map_from_edges, validate_fcg_preconditioner_output,
    };
    use crate::linear::{
        CgPreconditioner, CsrMatrix, PreconditionedConjugateGradientOptions,
        preconditioned_conjugate_gradient_solve,
    };

    #[test]
    fn openfoam_13_cycle_defaults_are_preserved() {
        let options = GamgOptions::default();

        assert!(options.cache_agglomeration);
        assert_eq!(options.n_cells_in_coarsest_level, 10);
        assert_eq!(options.merge_levels, 1);
        assert_eq!(options.n_pre_sweeps, 0);
        assert_eq!(options.pre_sweeps_level_multiplier, 1);
        assert_eq!(options.max_pre_sweeps, 4);
        assert_eq!(options.n_post_sweeps, 2);
        assert_eq!(options.post_sweeps_level_multiplier, 1);
        assert_eq!(options.max_post_sweeps, 4);
        assert_eq!(options.n_finest_sweeps, 2);
        assert!(!options.interpolate_correction);
        assert!(options.scale_correction);
        assert!(!options.direct_solve_coarsest);
        assert_eq!(options.outer_solver, GamgOuterSolver::Standalone);
        assert_eq!(options.agglomerator, GamgAgglomerator::AlgebraicPair);
        assert_eq!(options.smoother, GamgSmoother::GaussSeidel);
    }

    #[test]
    fn interpolate_correction_matches_foundation_13_ldu_face_order_oracle_and_coarse_means() {
        let order_sentinel = f64::EPSILON / 2.0;
        let matrix = CsrMatrix::from_rows(
            vec![
                vec![(0, 4.0), (1, 1.0), (2, order_sentinel), (3, -1.0)],
                vec![(0, 1.0), (1, 5.0)],
                vec![(0, order_sentinel), (2, 3.0)],
                vec![(0, -1.0), (3, 6.0)],
            ],
            4,
        )
        .expect("interpolation oracle matrix");
        let transfer = GamgTransfer {
            fine_to_coarse: vec![0, 0, 1, 1],
            fine_entry_to_coarse_entry: Vec::new(),
        };
        let diagonal = [4.0, 5.0, 3.0, 6.0];
        let coarse = [1.0, 1.0];
        let mut fine = [1.0; 4];
        let original = fine;
        let mut fine_scratch = [0.0; 4];
        let mut coarse_correction_scratch = [0.0; 2];
        let mut coarse_diagonal_scratch = [0.0; 2];

        transfer
            .interpolate_correction(
                &matrix,
                &diagonal,
                &coarse,
                &mut fine,
                GamgInterpolationScratch {
                    fine: &mut fine_scratch,
                    coarse_correction: &mut coarse_correction_scratch,
                    coarse_diagonal: &mut coarse_diagonal_scratch,
                },
            )
            .expect("Foundation interpolation");

        // Independent Foundation-13 LDU-face-order oracle. This intentionally
        // does not reuse the CSR row traversal from interpolate_correction.
        let faces = [
            (0_usize, 1_usize, 1.0),
            (0, 2, order_sentinel),
            (0, 3, -1.0),
        ];
        let mut off_diagonal = [0.0; 4];
        for (lower, upper, coefficient) in faces {
            off_diagonal[lower] += coefficient * original[upper];
            off_diagonal[upper] += coefficient * original[lower];
        }
        let mut reordered_off_diagonal = [0.0; 4];
        for (lower, upper, coefficient) in [faces[0], faces[2], faces[1]] {
            reordered_off_diagonal[lower] += coefficient * original[upper];
            reordered_off_diagonal[upper] += coefficient * original[lower];
        }
        assert_eq!(off_diagonal[0].to_bits(), 0.0_f64.to_bits());
        assert_eq!(
            reordered_off_diagonal[0].to_bits(),
            order_sentinel.to_bits()
        );
        let mut expected = [0.0; 4];
        for row in 0..4 {
            expected[row] = -off_diagonal[row] / diagonal[row];
        }
        let mut weighted_sum = [0.0; 2];
        let mut diagonal_sum = [0.0; 2];
        for row in 0..4 {
            let aggregate = transfer.fine_to_coarse[row];
            weighted_sum[aggregate] += diagonal[row] * expected[row];
            diagonal_sum[aggregate] += diagonal[row];
        }
        for aggregate in 0..2 {
            weighted_sum[aggregate] =
                coarse[aggregate] - weighted_sum[aggregate] / diagonal_sum[aggregate];
        }
        for row in 0..4 {
            expected[row] += weighted_sum[transfer.fine_to_coarse[row]];
        }

        let sealed_foundation_bits = [
            0x3ff1_c71c_71c7_1c72,
            0x3fed_27d2_7d27_d27e,
            0x3fec_71c7_1c71_c71c,
            0x3ff0_e38e_38e3_8e39,
        ];
        for ((actual, expected), sealed) in fine.iter().zip(expected).zip(sealed_foundation_bits) {
            assert_eq!(actual.to_bits(), expected.to_bits());
            assert_eq!(actual.to_bits(), sealed);
        }
        for (aggregate, coarse_value) in coarse.iter().enumerate() {
            let mut weighted = 0.0;
            let mut weight = 0.0;
            for row in 0..4 {
                if transfer.fine_to_coarse[row] == aggregate {
                    weighted += diagonal[row] * fine[row];
                    weight += diagonal[row];
                }
            }
            assert!((weighted / weight - coarse_value).abs() <= 2.0 * f64::EPSILON);
        }
    }

    #[test]
    fn interpolate_correction_failures_do_not_write_the_fine_field() {
        let matrix = CsrMatrix::from_rows(
            vec![vec![(0, 1.0), (1, -1.0)], vec![(0, -1.0), (1, -1.0)]],
            2,
        )
        .expect("zero aggregate diagonal matrix");
        let transfer = GamgTransfer {
            fine_to_coarse: vec![0, 0],
            fine_entry_to_coarse_entry: Vec::new(),
        };
        let mut fine = [2.0, 2.0];
        let before = fine;
        let mut fine_scratch = [0.0; 2];
        let mut coarse_correction_scratch = [0.0; 1];
        let mut coarse_diagonal_scratch = [0.0; 1];
        let error = transfer
            .interpolate_correction(
                &matrix,
                &[1.0, -1.0],
                &[0.5],
                &mut fine,
                GamgInterpolationScratch {
                    fine: &mut fine_scratch,
                    coarse_correction: &mut coarse_correction_scratch,
                    coarse_diagonal: &mut coarse_diagonal_scratch,
                },
            )
            .expect_err("zero aggregate diagonal must fail");
        assert_eq!(
            error.to_string(),
            "GAMG correction interpolation aggregate diagonal 0 is invalid: 0"
        );
        assert_eq!(fine, before);
        assert!(fine_scratch.iter().all(|value| value.is_finite()));
        assert!(
            coarse_correction_scratch
                .iter()
                .all(|value| value.is_finite())
        );
        assert!(
            coarse_diagonal_scratch
                .iter()
                .all(|value| value.is_finite())
        );

        let mut short_scratch = [0.0; 1];
        let shape_error = transfer
            .interpolate_correction(
                &matrix,
                &[1.0, -1.0],
                &[0.5],
                &mut fine,
                GamgInterpolationScratch {
                    fine: &mut short_scratch,
                    coarse_correction: &mut coarse_correction_scratch,
                    coarse_diagonal: &mut coarse_diagonal_scratch,
                },
            )
            .expect_err("short scratch must fail");
        assert_eq!(
            shape_error.to_string(),
            "GAMG correction interpolation fine shape mismatch: matrix=2 mapping=2 diagonal=2 correction=2 scratch=1"
        );
        assert_eq!(fine, before);

        let bad_mapping = GamgTransfer {
            fine_to_coarse: vec![0, 1],
            fine_entry_to_coarse_entry: Vec::new(),
        };
        let mapping_error = bad_mapping
            .interpolate_correction(
                &matrix,
                &[1.0, -1.0],
                &[0.5],
                &mut fine,
                GamgInterpolationScratch {
                    fine: &mut fine_scratch,
                    coarse_correction: &mut coarse_correction_scratch,
                    coarse_diagonal: &mut coarse_diagonal_scratch,
                },
            )
            .expect_err("out-of-range aggregate must fail");
        assert_eq!(
            mapping_error.to_string(),
            "GAMG correction interpolation coarse index 1 for fine row 1 is out of range 1"
        );
        assert_eq!(fine, before);

        let mut long_coarse_scratch = [0.0; 2];
        let coarse_shape_error = transfer
            .interpolate_correction(
                &matrix,
                &[1.0, -1.0],
                &[0.5],
                &mut fine,
                GamgInterpolationScratch {
                    fine: &mut fine_scratch,
                    coarse_correction: &mut long_coarse_scratch,
                    coarse_diagonal: &mut coarse_diagonal_scratch,
                },
            )
            .expect_err("coarse scratch mismatch must fail");
        assert_eq!(
            coarse_shape_error.to_string(),
            "GAMG correction interpolation coarse shape mismatch: correction=1 correctionScratch=2 diagonalScratch=1"
        );
        assert_eq!(fine, before);

        let overflow_matrix =
            CsrMatrix::from_rows(vec![vec![(0, 1.0), (1, 0.5)], vec![(0, 0.5), (1, 1.0)]], 2)
                .expect("late correction overflow matrix");
        let mut overflow_fine = [f64::MAX, f64::MAX];
        let overflow_before = overflow_fine;
        let late_error = transfer
            .interpolate_correction(
                &overflow_matrix,
                &[1.0, 1.0],
                &[f64::MAX],
                &mut overflow_fine,
                GamgInterpolationScratch {
                    fine: &mut fine_scratch,
                    coarse_correction: &mut coarse_correction_scratch,
                    coarse_diagonal: &mut coarse_diagonal_scratch,
                },
            )
            .expect_err("late aggregate correction overflow must fail");
        assert_eq!(
            late_error.to_string(),
            "GAMG correction interpolation aggregate correction 0 is not finite"
        );
        assert_eq!(overflow_fine, overflow_before);
    }

    #[test]
    fn interpolate_correction_failed_solve_is_externally_atomic_and_retry_safe() {
        let invalid = CsrMatrix::from_rows(
            vec![vec![(0, 1.0), (1, -0.125)], vec![(0, -0.125), (1, -1.0)]],
            2,
        )
        .expect("interpolation failure matrix");
        let valid = CsrMatrix::from_rows(
            vec![vec![(0, 2.0), (1, -0.125)], vec![(0, -0.125), (1, 2.0)]],
            2,
        )
        .expect("interpolation retry matrix");
        let rhs = [1.0, -0.5];
        let initial = [0.25, -0.125];
        let initial_before = initial;
        let options = GamgOptions {
            max_iterations: 1,
            min_iterations: 1,
            tolerance: 0.0,
            relative_tolerance: 0.0,
            n_cells_in_coarsest_level: 1,
            interpolate_correction: true,
            direct_solve_coarsest: true,
            ..GamgOptions::default()
        };
        let controls: GamgSolveControls = options.into();
        let mut workspace = GamgWorkspace::new(&invalid, options).expect("failure-path workspace");

        let error = workspace
            .solve_with_controls(&invalid, &rhs, Some(&initial), controls)
            .expect_err("zero fine-level aggregate diagonal must fail");
        assert_eq!(
            error.to_string(),
            "GAMG correction interpolation aggregate diagonal 0 is invalid: 0"
        );
        assert_eq!(initial, initial_before);
        assert!(!workspace.has_solved);

        let retried = workspace
            .solve_with_controls(&valid, &rhs, Some(&initial), controls)
            .expect("failed workspace must remain retry-safe");
        let mut fresh = GamgWorkspace::new(&valid, options).expect("fresh retry workspace");
        let expected = fresh
            .solve_with_controls(&valid, &rhs, Some(&initial), controls)
            .expect("fresh retry solve");
        assert_eq!(retried.iterations, expected.iterations);
        assert_eq!(retried.converged, expected.converged);
        assert_eq!(retried.termination, expected.termination);
        assert_eq!(
            retried.residual_norm.to_bits(),
            expected.residual_norm.to_bits()
        );
        for (actual, expected) in retried.solution.iter().zip(expected.solution) {
            assert_eq!(actual.to_bits(), expected.to_bits());
        }
    }

    #[test]
    fn interpolate_correction_true_preserves_plain_profiled_and_agglomerator_routes() {
        let matrix = poisson_grid(4, 4, 1.0);
        let rhs = (0..matrix.rows())
            .map(|row| 1.0 + row as f64 * 0.03125)
            .collect::<Vec<_>>();
        let options = GamgOptions {
            max_iterations: 2,
            min_iterations: 2,
            tolerance: 0.0,
            relative_tolerance: 0.0,
            n_cells_in_coarsest_level: 2,
            n_pre_sweeps: 1,
            interpolate_correction: true,
            direct_solve_coarsest: false,
            ..GamgOptions::default()
        };
        let controls = GamgSolveControls {
            max_iterations: 2,
            min_iterations: 2,
            tolerance: 0.0,
            relative_tolerance: 0.0,
        };

        let mut plain_workspace =
            GamgWorkspace::new(&matrix, options).expect("plain interpolation workspace");
        let plain = plain_workspace
            .solve_with_controls(&matrix, &rhs, None, controls)
            .expect("plain interpolation solve");
        let mut profiled_workspace =
            GamgWorkspace::new(&matrix, options).expect("profiled interpolation workspace");
        let profiled = profiled_workspace
            .solve_with_controls_profiled(&matrix, &rhs, None, controls)
            .expect("profiled interpolation solve");
        assert_eq!(plain.iterations, 2);
        assert_eq!(profiled.report.iterations, 2);
        assert_eq!(plain.converged, profiled.report.converged);
        assert_eq!(plain.termination, profiled.report.termination);
        assert_eq!(
            plain.residual_norm.to_bits(),
            profiled.report.residual_norm.to_bits()
        );
        for (plain, profiled) in plain.solution.iter().zip(&profiled.report.solution) {
            assert_eq!(plain.to_bits(), profiled.to_bits());
        }
        assert_eq!(profiled.timing.v_cycles, 2);
        assert_eq!(profiled.timing.solves, 1);
        let coarsest = profiled_workspace.level_count() - 1;
        for (level, timing) in profiled.timing.levels.iter().enumerate() {
            assert_eq!(
                timing.prolongation_calls,
                if level < coarsest { 2 } else { 0 }
            );
        }
        assert!(
            profiled_workspace.pre_smoothed[1..coarsest]
                .iter()
                .flatten()
                .any(|value| *value != 0.0)
        );
        assert_eq!(profiled.timing.levels[1].smoothing_calls, 4);
        assert_eq!(profiled.timing.levels[1].smoothing_sweeps, 6);

        let normalized = NormalizedL1GamgSolveControls {
            normalization_factor: 1.0,
            tolerance: 0.0,
            relative_tolerance: 0.0,
            l2_controls: controls,
        };
        let mut normalized_plain_workspace =
            GamgWorkspace::new(&matrix, options).expect("normalized plain workspace");
        let normalized_plain = normalized_plain_workspace
            .solve_normalized_l1_with_controls(&matrix, &rhs, None, normalized)
            .expect("normalized plain interpolation solve");
        let mut normalized_profiled_workspace =
            GamgWorkspace::new(&matrix, options).expect("normalized profiled workspace");
        let normalized_profiled = normalized_profiled_workspace
            .solve_normalized_l1_with_controls_profiled(&matrix, &rhs, None, normalized)
            .expect("normalized profiled interpolation solve");
        assert_eq!(normalized_plain.iterations, 2);
        assert_eq!(normalized_profiled.report.iterations, 2);
        assert_eq!(
            normalized_plain.residual_norm.to_bits(),
            normalized_profiled.report.residual_norm.to_bits()
        );
        for (plain, profiled) in normalized_plain
            .solution
            .iter()
            .zip(&normalized_profiled.report.solution)
        {
            assert_eq!(plain.to_bits(), profiled.to_bits());
        }
        assert_eq!(normalized_profiled.timing.v_cycles, 2);
        for (level, timing) in normalized_profiled.timing.levels.iter().enumerate() {
            assert_eq!(
                timing.prolongation_calls,
                if level < coarsest { 2 } else { 0 }
            );
        }

        let face_options = GamgOptions {
            agglomerator: GamgAgglomerator::FaceAreaPair,
            max_iterations: 1,
            min_iterations: 1,
            ..options
        };
        let mut face_workspace = GamgWorkspace::new_with_face_area_weights(
            &matrix,
            face_options,
            &grid_face_weights(4, 4),
        )
        .expect("face-area interpolation workspace");
        let face_report = face_workspace
            .solve(&matrix, &rhs, None)
            .expect("face-area interpolation solve");
        assert_eq!(face_report.iterations, 1);
        assert!(face_report.solution.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn interpolate_correction_entrypoints_have_sealed_one_cycle_sentinels() {
        let matrix = poisson_grid(4, 4, 1.0);
        let rhs = (0..matrix.rows())
            .map(|row| 1.0 + row as f64 * 0.03125)
            .collect::<Vec<_>>();
        let true_options = GamgOptions {
            max_iterations: 1,
            min_iterations: 1,
            tolerance: 0.0,
            relative_tolerance: 0.0,
            n_cells_in_coarsest_level: 2,
            n_pre_sweeps: 1,
            interpolate_correction: true,
            direct_solve_coarsest: false,
            ..GamgOptions::default()
        };
        let false_options = GamgOptions {
            interpolate_correction: false,
            ..true_options
        };
        let controls: GamgSolveControls = true_options.into();
        let normalized = NormalizedL1GamgSolveControls {
            normalization_factor: 1.0,
            tolerance: 0.0,
            relative_tolerance: 0.0,
            l2_controls: controls,
        };

        let mut true_l2_workspace =
            GamgWorkspace::new(&matrix, true_options).expect("true L2 sentinel workspace");
        let true_l2 = true_l2_workspace
            .solve_with_controls(&matrix, &rhs, None, controls)
            .expect("true L2 sentinel");
        let mut true_l1_workspace =
            GamgWorkspace::new(&matrix, true_options).expect("true L1 sentinel workspace");
        let true_l1 = true_l1_workspace
            .solve_normalized_l1_with_controls(&matrix, &rhs, None, normalized)
            .expect("true normalized-L1 sentinel");
        let mut false_l2_workspace =
            GamgWorkspace::new(&matrix, false_options).expect("false L2 sentinel workspace");
        let false_l2 = false_l2_workspace
            .solve_with_controls(&matrix, &rhs, None, controls)
            .expect("false L2 sentinel");
        let mut false_l1_workspace =
            GamgWorkspace::new(&matrix, false_options).expect("false L1 sentinel workspace");
        let false_l1 = false_l1_workspace
            .solve_normalized_l1_with_controls(&matrix, &rhs, None, normalized)
            .expect("false normalized-L1 sentinel");

        let true_solution_bits = [
            0x3fed_65c3_64a4_ed0c,
            0x3ff4_ffcb_6f65_2de2,
            0x3ff5_3eaa_ab73_5ff8,
            0x3fee_aac9_af10_98b4,
            0x3ff5_e5ca_324e_c053,
            0x3fff_a791_ad90_f3e2,
            0x4000_0834_913f_7538,
            0x3ff6_b883_80c0_e5e9,
            0x3ff7_3848_3e91_daec,
            0x4000_b82a_2e4f_89a9,
            0x4000_eb58_2ccd_e933,
            0x3ff8_096d_350c_42d2,
            0x3ff1_7550_11a8_753d,
            0x3ff8_90da_1327_1ce4,
            0x3ff8_df82_d9a5_d090,
            0x3ff2_1a3c_03ac_84d8,
        ];
        let false_solution_bits = [
            0x3feb_c563_3c8a_b6e4,
            0x3ff3_cb12_87a8_b142,
            0x3ff4_5986_8679_2184,
            0x3fed_ca7a_9e49_9b93,
            0x3ff4_d8d2_b729_8c92,
            0x3ffe_4f65_290f_b100,
            0x3fff_07aa_f463_a1d7,
            0x3ff6_3cc6_1dcf_394f,
            0x3ff6_8502_494b_a76a,
            0x4000_4797_e759_b324,
            0x4000_93fc_2283_62ca,
            0x3ff7_b1bf_fdc5_c09c,
            0x3ff1_2cc2_2974_923a,
            0x3ff8_2fb2_055d_e446,
            0x3ff8_8e7a_f7a9_6b5a,
            0x3ff1_f00e_bd5b_cafe,
        ];
        for ((true_l2, true_l1), expected) in true_l2
            .solution
            .iter()
            .zip(&true_l1.solution)
            .zip(true_solution_bits)
        {
            assert_eq!(true_l2.to_bits(), expected);
            assert_eq!(true_l1.to_bits(), expected);
        }
        for ((false_l2, false_l1), expected) in false_l2
            .solution
            .iter()
            .zip(&false_l1.solution)
            .zip(false_solution_bits)
        {
            assert_eq!(false_l2.to_bits(), expected);
            assert_eq!(false_l1.to_bits(), expected);
        }
        assert_eq!(true_l2.residual_norm.to_bits(), 0x3fa2_e97e_52b5_abb0);
        assert_eq!(true_l1.residual_norm.to_bits(), 0x3fa2_e97e_52b5_abb0);
        assert_eq!(false_l2.residual_norm.to_bits(), 0x3fcb_cf71_dd54_76ae);
        assert_eq!(false_l1.residual_norm.to_bits(), 0x3fcb_cf71_dd54_76ae);
        assert_ne!(true_solution_bits, false_solution_bits);
    }

    #[test]
    fn interpolate_correction_scales_coarsest_child_and_reuses_scratch() {
        let matrix = poisson_grid(8, 8, 1.0);
        let rhs = vec![1.0; matrix.rows()];
        let options = GamgOptions {
            max_iterations: 1,
            min_iterations: 1,
            tolerance: 0.0,
            relative_tolerance: 0.0,
            n_cells_in_coarsest_level: 2,
            interpolate_correction: true,
            scale_correction: true,
            direct_solve_coarsest: true,
            ..GamgOptions::default()
        };
        let controls: GamgSolveControls = options.into();
        let mut workspace =
            GamgWorkspace::new(&matrix, options).expect("interpolation lifecycle workspace");
        assert!(workspace.level_count() >= 3);
        let product_allocations = workspace
            .products
            .iter()
            .map(|values| (values.as_ptr(), values.len(), values.capacity()))
            .collect::<Vec<_>>();
        let residual_allocations = workspace
            .residuals
            .iter()
            .map(|values| (values.as_ptr(), values.len(), values.capacity()))
            .collect::<Vec<_>>();

        for lifecycle in 0..10 {
            let scaled = poisson_grid(8, 8, 1.0 + lifecycle as f64 * 0.015625);
            let report = workspace
                .solve_with_controls(&scaled, &rhs, None, controls)
                .expect("reused interpolation solve");
            let mut fresh =
                GamgWorkspace::new(&scaled, options).expect("fresh interpolation workspace");
            let expected = fresh
                .solve_with_controls(&scaled, &rhs, None, controls)
                .expect("fresh interpolation solve");
            assert_eq!(report.iterations, expected.iterations);
            assert_eq!(
                report.residual_norm.to_bits(),
                expected.residual_norm.to_bits()
            );
            for (actual, expected) in report.solution.iter().zip(expected.solution) {
                assert_eq!(actual.to_bits(), expected.to_bits());
            }
            assert_eq!(
                workspace
                    .products
                    .iter()
                    .map(|values| (values.as_ptr(), values.len(), values.capacity()))
                    .collect::<Vec<_>>(),
                product_allocations
            );
            assert_eq!(
                workspace
                    .residuals
                    .iter()
                    .map(|values| (values.as_ptr(), values.len(), values.capacity()))
                    .collect::<Vec<_>>(),
                residual_allocations
            );
        }

        let mut profiled_workspace =
            GamgWorkspace::new(&matrix, options).expect("profiled scaling workspace");
        let profiled = profiled_workspace
            .solve_with_controls_profiled(&matrix, &rhs, None, controls)
            .expect("profiled scaling solve");
        let coarsest = profiled_workspace.level_count() - 1;
        assert_eq!(profiled.timing.levels[coarsest - 1].scaling_calls, 1);

        let injection_options = GamgOptions {
            interpolate_correction: false,
            ..options
        };
        let mut injection_workspace =
            GamgWorkspace::new(&matrix, injection_options).expect("injection workspace");
        let injection_profiled = injection_workspace
            .solve_with_controls_profiled(&matrix, &rhs, None, injection_options.into())
            .expect("profiled injection solve");
        let injection_coarsest = injection_workspace.level_count() - 1;
        assert_eq!(
            injection_profiled.timing.levels[injection_coarsest - 1].scaling_calls,
            0
        );

        let unscaled_options = GamgOptions {
            scale_correction: false,
            ..options
        };
        let mut unscaled_workspace =
            GamgWorkspace::new(&matrix, unscaled_options).expect("unscaled workspace");
        let unscaled_profiled = unscaled_workspace
            .solve_with_controls_profiled(&matrix, &rhs, None, unscaled_options.into())
            .expect("unscaled interpolation solve");
        assert!(
            unscaled_profiled
                .timing
                .levels
                .iter()
                .all(|level| level.scaling_calls == 0)
        );

        let rebuild_options = GamgOptions {
            cache_agglomeration: false,
            ..options
        };
        let mut rebuild_workspace =
            GamgWorkspace::new(&matrix, rebuild_options).expect("rebuild workspace");
        rebuild_workspace
            .solve(&matrix, &rhs, None)
            .expect("initial rebuild solve");
        let rebuilt = rebuild_workspace
            .solve_with_controls_profiled(&matrix, &rhs, None, rebuild_options.into())
            .expect("second rebuild solve");
        assert_eq!(rebuilt.timing.hierarchy_rebuilds, 1);
        assert!(
            rebuilt
                .report
                .solution
                .iter()
                .all(|value| value.is_finite())
        );
    }

    #[test]
    fn normalized_l1_uses_strict_absolute_and_relative_boundaries() {
        let matrix = CsrMatrix::from_rows(
            vec![
                vec![(0, 2.0), (1, -1.0)],
                vec![(0, -1.0), (1, 2.0), (2, -1.0)],
                vec![(1, -1.0), (2, 2.0), (3, -1.0)],
                vec![(2, -1.0), (3, 2.0), (4, -1.0)],
                vec![(3, -1.0), (4, 2.0), (5, -1.0)],
                vec![(4, -1.0), (5, 2.0), (6, -1.0)],
                vec![(5, -1.0), (6, 2.0), (7, -1.0)],
                vec![(6, -1.0), (7, 2.0)],
            ],
            8,
        )
        .expect("explicit boundary matrix");
        let rhs = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let options = GamgOptions {
            max_iterations: 1,
            n_cells_in_coarsest_level: 2,
            direct_solve_coarsest: true,
            ..GamgOptions::default()
        };
        let mut prepass_workspace =
            GamgWorkspace::new(&matrix, options).expect("prepass workspace");
        let prepass = prepass_workspace
            .solve(&matrix, &rhs, None)
            .expect("one-cycle prepass");
        let initial_raw_l1 = rhs.iter().map(|value| value.abs()).sum::<f64>();
        assert_eq!(initial_raw_l1.to_bits(), 1.0f64.to_bits());
        assert_eq!((initial_raw_l1 / 1.0).to_bits(), 1.0f64.to_bits());
        let mut raw_l1 = 0.0;
        let mut squared_l2 = 0.0;
        for (row, source) in rhs.iter().enumerate() {
            let mut product = 0.0;
            for entry in matrix.row_offsets()[row]..matrix.row_offsets()[row + 1] {
                product += matrix.values()[entry] * prepass.solution[matrix.col_indices()[entry]];
            }
            let residual = source - product;
            raw_l1 += residual.abs();
            squared_l2 += residual * residual;
        }
        assert_eq!(squared_l2.sqrt().to_bits(), prepass.residual_norm.to_bits());
        assert!(raw_l1.is_finite() && raw_l1 > 0.0);
        let next = raw_l1.next_up();
        assert!(next.is_finite() && next > 0.0 && next > raw_l1);
        assert_eq!(next.to_bits(), raw_l1.to_bits() + 1);
        let mut two_cycle_solution_bits: Option<[u64; 8]> = None;
        let mut equality_legs = 0;
        let mut equality_comparisons = 0;

        for relative in [false, true] {
            for profiled in [false, true] {
                for (boundary, expected_iterations) in [(raw_l1, 2), (next, 1)] {
                    let controls = NormalizedL1GamgSolveControls {
                        normalization_factor: 1.0,
                        tolerance: if relative { 0.0 } else { boundary },
                        relative_tolerance: if relative { boundary } else { 0.0 },
                        l2_controls: super::GamgSolveControls {
                            max_iterations: 2,
                            min_iterations: 0,
                            tolerance: 0.0,
                            relative_tolerance: 0.0,
                        },
                    };
                    let mut workspace =
                        GamgWorkspace::new(&matrix, options).expect("boundary workspace");
                    let (report, timing) = if profiled {
                        let profiled = workspace
                            .solve_normalized_l1_with_controls_profiled(
                                &matrix, &rhs, None, controls,
                            )
                            .expect("profiled normalized solve");
                        (profiled.report, Some(profiled.timing))
                    } else {
                        (
                            workspace
                                .solve_normalized_l1_with_controls(&matrix, &rhs, None, controls)
                                .expect("plain normalized solve"),
                            None,
                        )
                    };
                    assert_eq!(report.iterations, expected_iterations);
                    assert!(report.converged);
                    assert_eq!(
                        report.termination,
                        super::IterativeSolveTermination::Converged
                    );
                    if expected_iterations == 1 {
                        for (actual, expected) in
                            report.solution.iter().zip(prepass.solution.iter())
                        {
                            assert_eq!(actual.to_bits(), expected.to_bits());
                        }
                    } else {
                        equality_legs += 1;
                        let actual_bits =
                            std::array::from_fn(|index| report.solution[index].to_bits());
                        if let Some(expected_bits) = two_cycle_solution_bits {
                            assert_eq!(actual_bits, expected_bits);
                            equality_comparisons += 1;
                        } else {
                            two_cycle_solution_bits = Some(actual_bits);
                        }
                    }
                    let mut report_squared_l2 = 0.0;
                    for (row, source) in rhs.iter().enumerate() {
                        let mut product = 0.0;
                        for entry in matrix.row_offsets()[row]..matrix.row_offsets()[row + 1] {
                            product += matrix.values()[entry]
                                * report.solution[matrix.col_indices()[entry]];
                        }
                        let residual = source - product;
                        report_squared_l2 += residual * residual;
                    }
                    assert_eq!(
                        report.residual_norm.to_bits(),
                        report_squared_l2.sqrt().to_bits()
                    );
                    if let Some(timing) = timing {
                        assert_eq!(timing.hierarchy_builds, 0);
                        assert_eq!(timing.hierarchy_rebuilds, 0);
                        assert_eq!(timing.matrix_refreshes, 1);
                        assert_eq!(timing.finest_residual_evaluations, expected_iterations + 1);
                        assert_eq!(timing.solves, 1);
                        assert_eq!(timing.v_cycles, expected_iterations);
                        let level_tuples = timing
                            .levels
                            .iter()
                            .map(|level| {
                                (
                                    level.level,
                                    level.cells,
                                    level.nonzeros,
                                    level.matrix_refreshes,
                                    level.restriction_calls,
                                    level.prolongation_calls,
                                    level.smoothing_calls,
                                    level.smoothing_sweeps,
                                    level.scaling_calls,
                                    level.residual_evaluations,
                                    level.correction_updates,
                                    level.coarsest_solves,
                                )
                            })
                            .collect::<Vec<_>>();
                        assert_eq!(
                            level_tuples,
                            vec![
                                (
                                    0,
                                    8,
                                    22,
                                    1,
                                    expected_iterations,
                                    expected_iterations,
                                    expected_iterations,
                                    2 * expected_iterations,
                                    expected_iterations,
                                    0,
                                    expected_iterations,
                                    0,
                                ),
                                (
                                    1,
                                    4,
                                    10,
                                    1,
                                    expected_iterations,
                                    expected_iterations,
                                    expected_iterations,
                                    2 * expected_iterations,
                                    0,
                                    0,
                                    expected_iterations,
                                    0,
                                ),
                                (2, 2, 4, 1, 0, 0, 0, 0, 0, 0, 0, expected_iterations),
                            ]
                        );
                    }
                    assert_eq!(workspace.level_sizes(), vec![8, 4, 2]);
                }
            }
        }
        assert!(two_cycle_solution_bits.is_some());
        assert_eq!(equality_legs, 4);
        assert_eq!(equality_comparisons, 3);
    }

    #[test]
    fn normalized_l1_entrypoint_uses_strict_openfoam_reltol_small_boundary() {
        assert_eq!(
            super::OPENFOAM_RELATIVE_TOLERANCE_SMALL.to_bits(),
            1.0e-20_f64.to_bits()
        );
        let matrix = CsrMatrix::from_rows(
            vec![vec![(0, 2.0), (1, -1.0)], vec![(0, -1.0), (1, 2.0)]],
            2,
        )
        .expect("two-cell matrix");
        let rhs = [1.0, 1.0];
        let options = GamgOptions {
            max_iterations: 2,
            min_iterations: 0,
            n_cells_in_coarsest_level: 1,
            n_post_sweeps: 0,
            n_finest_sweeps: 0,
            scale_correction: false,
            direct_solve_coarsest: true,
            ..GamgOptions::default()
        };
        let solve_controls = |relative_tolerance| NormalizedL1GamgSolveControls {
            normalization_factor: 1.0,
            tolerance: 0.0,
            relative_tolerance,
            l2_controls: super::GamgSolveControls {
                max_iterations: 2,
                min_iterations: 0,
                tolerance: 0.0,
                relative_tolerance: 0.0,
            },
        };

        let mut equality_workspace =
            GamgWorkspace::new(&matrix, options).expect("equality workspace");
        let equality = equality_workspace
            .solve_normalized_l1_with_controls_profiled(
                &matrix,
                &rhs,
                None,
                solve_controls(super::OPENFOAM_RELATIVE_TOLERANCE_SMALL),
            )
            .expect("equality solve");
        assert_eq!(equality.report.iterations, 2);
        assert!(!equality.report.converged);
        assert_eq!(
            equality.report.termination,
            super::IterativeSolveTermination::MaxIterations
        );
        assert_eq!(equality.report.residual_norm.to_bits(), 0.0f64.to_bits());
        assert_eq!(equality.timing.v_cycles, 2);
        assert_eq!(equality.timing.finest_residual_evaluations, 3);

        let mut active_workspace = GamgWorkspace::new(&matrix, options).expect("active workspace");
        let active = active_workspace
            .solve_normalized_l1_with_controls_profiled(
                &matrix,
                &rhs,
                None,
                solve_controls(super::OPENFOAM_RELATIVE_TOLERANCE_SMALL.next_up()),
            )
            .expect("next-up solve");
        assert_eq!(active.report.iterations, 1);
        assert!(active.report.converged);
        assert_eq!(
            active.report.termination,
            super::IterativeSolveTermination::Converged
        );
        assert_eq!(active.report.residual_norm.to_bits(), 0.0f64.to_bits());
        assert_eq!(active.timing.v_cycles, 1);
        assert_eq!(active.timing.finest_residual_evaluations, 2);
        for (left, right) in equality
            .report
            .solution
            .iter()
            .zip(active.report.solution.iter())
        {
            assert_eq!(left.to_bits(), right.to_bits());
        }
    }

    #[test]
    fn public_l2_convergence_keeps_inclusive_boundaries() {
        let matrix = CsrMatrix::from_rows(
            vec![
                vec![(0, 2.0), (1, -1.0)],
                vec![(0, -1.0), (1, 2.0), (2, -1.0)],
                vec![(1, -1.0), (2, 2.0), (3, -1.0)],
                vec![(2, -1.0), (3, 2.0), (4, -1.0)],
                vec![(3, -1.0), (4, 2.0), (5, -1.0)],
                vec![(4, -1.0), (5, 2.0), (6, -1.0)],
                vec![(5, -1.0), (6, 2.0), (7, -1.0)],
                vec![(6, -1.0), (7, 2.0)],
            ],
            8,
        )
        .expect("explicit boundary matrix");
        let rhs = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let options = GamgOptions {
            max_iterations: 1,
            n_cells_in_coarsest_level: 2,
            direct_solve_coarsest: true,
            ..GamgOptions::default()
        };
        let mut prepass_workspace =
            GamgWorkspace::new(&matrix, options).expect("prepass workspace");
        let prepass = prepass_workspace
            .solve(&matrix, &rhs, None)
            .expect("one-cycle prepass");
        let mut squared_l2 = 0.0;
        for (row, source) in rhs.iter().enumerate() {
            let mut product = 0.0;
            for entry in matrix.row_offsets()[row]..matrix.row_offsets()[row + 1] {
                product += matrix.values()[entry] * prepass.solution[matrix.col_indices()[entry]];
            }
            let residual = source - product;
            squared_l2 += residual * residual;
        }
        let boundary = squared_l2.sqrt();
        assert_eq!(boundary.to_bits(), prepass.residual_norm.to_bits());
        let initial_l2 = 1.0;

        for relative in [false, true] {
            for profiled in [false, true] {
                let controls = super::GamgSolveControls {
                    max_iterations: 2,
                    min_iterations: 0,
                    tolerance: if relative { 0.0 } else { boundary },
                    relative_tolerance: if relative { boundary / initial_l2 } else { 0.0 },
                };
                assert_eq!(
                    (controls.relative_tolerance * initial_l2).to_bits(),
                    if relative {
                        boundary.to_bits()
                    } else {
                        0.0f64.to_bits()
                    }
                );
                let mut workspace =
                    GamgWorkspace::new(&matrix, options).expect("boundary workspace");
                let (report, timing) = if profiled {
                    let profiled = workspace
                        .solve_with_controls_profiled(&matrix, &rhs, None, controls)
                        .expect("profiled L2 solve");
                    (profiled.report, Some(profiled.timing))
                } else {
                    (
                        workspace
                            .solve_with_controls(&matrix, &rhs, None, controls)
                            .expect("plain L2 solve"),
                        None,
                    )
                };
                assert_eq!(report.iterations, 1);
                assert!(report.converged);
                assert_eq!(
                    report.termination,
                    super::IterativeSolveTermination::Converged
                );
                assert_eq!(report.residual_norm.to_bits(), boundary.to_bits());
                for (actual, expected) in report.solution.iter().zip(&prepass.solution) {
                    assert_eq!(actual.to_bits(), expected.to_bits());
                }
                if let Some(timing) = timing {
                    assert_eq!(timing.hierarchy_builds, 0);
                    assert_eq!(timing.hierarchy_rebuilds, 0);
                    assert_eq!(timing.matrix_refreshes, 1);
                    assert_eq!(timing.finest_residual_evaluations, 2);
                    assert_eq!(timing.solves, 1);
                    assert_eq!(timing.v_cycles, 1);
                    let level_tuples = timing
                        .levels
                        .iter()
                        .map(|level| {
                            (
                                level.level,
                                level.cells,
                                level.nonzeros,
                                level.matrix_refreshes,
                                level.restriction_calls,
                                level.prolongation_calls,
                                level.smoothing_calls,
                                level.smoothing_sweeps,
                                level.scaling_calls,
                                level.residual_evaluations,
                                level.correction_updates,
                                level.coarsest_solves,
                            )
                        })
                        .collect::<Vec<_>>();
                    assert_eq!(
                        level_tuples,
                        vec![
                            (0, 8, 22, 1, 1, 1, 1, 2, 1, 0, 1, 0),
                            (1, 4, 10, 1, 1, 1, 1, 2, 0, 0, 1, 0),
                            (2, 2, 4, 1, 0, 0, 0, 0, 0, 0, 0, 1),
                        ]
                    );
                }
                assert_eq!(workspace.level_sizes(), vec![8, 4, 2]);
            }
        }
    }

    #[test]
    fn invalid_normalized_l1_controls_fail_before_solve_mutation() {
        #[derive(Clone, Debug, PartialEq, Eq)]
        struct UsizeVecSnap {
            len: usize,
            capacity: usize,
            values: Vec<usize>,
        }

        #[derive(Clone, Debug, PartialEq, Eq)]
        struct BitsVecSnap {
            len: usize,
            capacity: usize,
            bits: Vec<u64>,
        }

        #[derive(Clone, Debug, PartialEq, Eq)]
        struct PairVecSnap {
            len: usize,
            capacity: usize,
            values: Vec<(usize, usize)>,
        }

        #[derive(Clone, Debug, PartialEq, Eq)]
        struct NestedUsizeSnap {
            len: usize,
            capacity: usize,
            vectors: Vec<(usize, UsizeVecSnap)>,
        }

        #[derive(Clone, Debug, PartialEq, Eq)]
        struct NestedBitsSnap {
            len: usize,
            capacity: usize,
            vectors: Vec<(usize, BitsVecSnap)>,
        }

        #[derive(Clone, Debug, PartialEq, Eq)]
        struct NestedPairSnap {
            len: usize,
            capacity: usize,
            vectors: Vec<(usize, PairVecSnap)>,
        }

        #[derive(Clone, Debug, PartialEq, Eq)]
        struct SparsityValueSnap {
            rows: usize,
            cols: usize,
            row_offsets_len: usize,
            row_offsets: Vec<usize>,
            col_indices_len: usize,
            col_indices: Vec<usize>,
        }

        #[derive(Clone, Debug, PartialEq, Eq)]
        struct MatrixValueSnap {
            sparsity: SparsityValueSnap,
            values: BitsVecSnap,
        }

        #[derive(Clone, Debug, PartialEq, Eq)]
        enum AgglomerationValueSnap {
            Algebraic,
            FaceArea {
                len: usize,
                weights: Vec<(usize, usize, u64)>,
            },
        }

        #[derive(Clone, Debug, PartialEq, Eq)]
        struct TransferValueSnap {
            fine_to_coarse: UsizeVecSnap,
            fine_entry_to_coarse_entry: UsizeVecSnap,
        }

        #[derive(Clone, Debug, PartialEq, Eq)]
        struct IncompleteCholeskyValueSnap {
            sparsity: SparsityValueSnap,
            lower_row_offsets: UsizeVecSnap,
            lower_columns: UsizeVecSnap,
            matrix_slots: UsizeVecSnap,
            diagonal_factor_slots: UsizeVecSnap,
            update_pairs: NestedPairSnap,
            dependent_row_offsets: UsizeVecSnap,
            dependent_entries: PairVecSnap,
            factors: BitsVecSnap,
        }

        #[derive(Clone, Debug, PartialEq, Eq)]
        enum PreconditionerValueSnap {
            None,
            Diagonal {
                matrix_slots: UsizeVecSnap,
                inverse: BitsVecSnap,
            },
            IncompleteCholesky(Box<IncompleteCholeskyValueSnap>),
        }

        #[derive(Clone, Debug, PartialEq, Eq)]
        struct PcgValueSnap {
            sparsity: SparsityValueSnap,
            preconditioner_kind: CgPreconditioner,
            preconditioner: PreconditionerValueSnap,
            residual: BitsVecSnap,
            preconditioned_residual: BitsVecSnap,
            direction: BitsVecSnap,
            matrix_direction: BitsVecSnap,
            preconditioner_scratch: BitsVecSnap,
        }

        #[derive(Clone, Debug, PartialEq, Eq)]
        struct WorkspaceValueSnap {
            option_iterations: (usize, usize, u64, u64),
            option_hierarchy: (bool, usize, usize, GamgAgglomerator),
            option_outer_solver: GamgOuterSolver,
            option_smoothing: (GamgSmoother, usize, usize, usize, usize, usize, usize),
            option_correction: (usize, bool, bool, bool),
            agglomeration: AgglomerationValueSnap,
            finest_sparsity: SparsityValueSnap,
            matrices_len: usize,
            matrices_capacity: usize,
            matrices: Vec<(usize, MatrixValueSnap)>,
            transfers_len: usize,
            transfers_capacity: usize,
            transfers: Vec<(usize, TransferValueSnap)>,
            diagonal_slots: NestedUsizeSnap,
            diagonal_values: NestedBitsSnap,
            corrections: NestedBitsSnap,
            sources: NestedBitsSnap,
            residuals: NestedBitsSnap,
            products: NestedBitsSnap,
            pre_smoothed: NestedBitsSnap,
            fcg_residual: BitsVecSnap,
            fcg_preconditioned_residual: BitsVecSnap,
            fcg_direction: BitsVecSnap,
            fcg_matrix_direction: BitsVecSnap,
            fcg_previous_direction: BitsVecSnap,
            fcg_previous_matrix_direction: BitsVecSnap,
            coarsest_pcg: Option<PcgValueSnap>,
            profiled_hierarchy: Option<super::GamgHierarchyDiagnostics>,
            has_solved: bool,
        }

        #[derive(Clone, Debug, PartialEq, Eq)]
        struct LevelTimingValueSnap {
            metadata: (usize, usize, usize),
            seconds_bits: [u64; 8],
            counters: [usize; 10],
        }

        #[derive(Clone, Debug, PartialEq, Eq)]
        struct TimingValueSnap {
            seconds_bits: [u64; 8],
            counters: [usize; 8],
            levels_len: usize,
            levels_capacity: usize,
            levels: Vec<(usize, LevelTimingValueSnap)>,
            hierarchy: Option<(usize, super::GamgHierarchyDiagnostics)>,
        }

        #[derive(Clone, Debug)]
        struct FullSnapshot {
            workspace: WorkspaceValueSnap,
            timing: TimingValueSnap,
            initial: BitsVecSnap,
            usize_arcs: Vec<(String, std::sync::Arc<[usize]>)>,
            face_arcs: Vec<std::sync::Arc<[GamgFacePairWeight]>>,
            profiled_hierarchy: Option<std::sync::Arc<super::GamgHierarchyDiagnostics>>,
        }

        let usize_vec_snap = |values: &Vec<usize>| UsizeVecSnap {
            len: values.len(),
            capacity: values.capacity(),
            values: values.clone(),
        };
        let bits_vec_snap = |values: &Vec<f64>| BitsVecSnap {
            len: values.len(),
            capacity: values.capacity(),
            bits: values.iter().map(|value| value.to_bits()).collect(),
        };
        let pair_vec_snap = |values: &Vec<(usize, usize)>| PairVecSnap {
            len: values.len(),
            capacity: values.capacity(),
            values: values.clone(),
        };
        let nested_usize_snap = |vectors: &Vec<Vec<usize>>| NestedUsizeSnap {
            len: vectors.len(),
            capacity: vectors.capacity(),
            vectors: vectors
                .iter()
                .enumerate()
                .map(|(index, values)| (index, usize_vec_snap(values)))
                .collect(),
        };
        let nested_bits_snap = |vectors: &Vec<Vec<f64>>| NestedBitsSnap {
            len: vectors.len(),
            capacity: vectors.capacity(),
            vectors: vectors
                .iter()
                .enumerate()
                .map(|(index, values)| (index, bits_vec_snap(values)))
                .collect(),
        };
        let nested_pair_snap = |vectors: &Vec<Vec<(usize, usize)>>| NestedPairSnap {
            len: vectors.len(),
            capacity: vectors.capacity(),
            vectors: vectors
                .iter()
                .enumerate()
                .map(|(index, values)| (index, pair_vec_snap(values)))
                .collect(),
        };
        let sparsity_snap =
            |sparsity: &crate::linear::CsrSparsityPattern,
             label: &str,
             arcs: &mut Vec<(String, std::sync::Arc<[usize]>)>| {
                arcs.push((format!("{label}.row_offsets"), sparsity.row_offsets.clone()));
                arcs.push((format!("{label}.col_indices"), sparsity.col_indices.clone()));
                SparsityValueSnap {
                    rows: sparsity.rows,
                    cols: sparsity.cols,
                    row_offsets_len: sparsity.row_offsets.len(),
                    row_offsets: sparsity.row_offsets.to_vec(),
                    col_indices_len: sparsity.col_indices.len(),
                    col_indices: sparsity.col_indices.to_vec(),
                }
            };
        let timing_snap = |timing: &super::GamgKernelTiming| TimingValueSnap {
            seconds_bits: [
                timing.total_seconds.to_bits(),
                timing.hierarchy_build_seconds.to_bits(),
                timing.hierarchy_rebuild_seconds.to_bits(),
                timing.hierarchy_diagnostic_seconds.to_bits(),
                timing.matrix_refresh_seconds.to_bits(),
                timing.finest_residual_seconds.to_bits(),
                timing.v_cycle_seconds.to_bits(),
                timing.other_seconds.to_bits(),
            ],
            counters: [
                timing.hierarchy_builds,
                timing.hierarchy_rebuilds,
                timing.matrix_refreshes,
                timing.finest_residual_evaluations,
                timing.solves,
                timing.v_cycles,
                timing.outer_matrix_vector_products,
                timing.outer_reductions,
            ],
            levels_len: timing.levels.len(),
            levels_capacity: timing.levels.capacity(),
            levels: timing
                .levels
                .iter()
                .enumerate()
                .map(|(index, level)| {
                    (
                        index,
                        LevelTimingValueSnap {
                            metadata: (level.level, level.cells, level.nonzeros),
                            seconds_bits: [
                                level.matrix_refresh_seconds.to_bits(),
                                level.restriction_seconds.to_bits(),
                                level.prolongation_seconds.to_bits(),
                                level.smoothing_seconds.to_bits(),
                                level.scaling_seconds.to_bits(),
                                level.residual_seconds.to_bits(),
                                level.correction_seconds.to_bits(),
                                level.coarsest_solve_seconds.to_bits(),
                            ],
                            counters: [
                                level.matrix_refreshes,
                                level.restriction_calls,
                                level.prolongation_calls,
                                level.smoothing_calls,
                                level.smoothing_sweeps,
                                level.scaling_calls,
                                level.residual_evaluations,
                                level.correction_updates,
                                level.coarsest_solves,
                                level.coarsest_iterations,
                            ],
                        },
                    )
                })
                .collect(),
            hierarchy: timing.hierarchy.as_ref().map(|hierarchy| {
                (
                    std::sync::Arc::as_ptr(hierarchy) as usize,
                    (**hierarchy).clone(),
                )
            }),
        };
        let snapshot = |workspace: &GamgWorkspace,
                        timing: &super::GamgKernelTiming,
                        initial: &Vec<f64>| {
            let mut usize_arcs = Vec::<(String, std::sync::Arc<[usize]>)>::new();
            let mut face_arcs = Vec::<std::sync::Arc<[GamgFacePairWeight]>>::new();

            let agglomeration = match &workspace.agglomeration_source {
                super::GamgAgglomerationSource::Algebraic => AgglomerationValueSnap::Algebraic,
                super::GamgAgglomerationSource::FaceArea(weights) => {
                    face_arcs.push(weights.clone());
                    AgglomerationValueSnap::FaceArea {
                        len: weights.len(),
                        weights: weights
                            .iter()
                            .map(|weight| {
                                let (first, second) = weight.cells();
                                (first, second, weight.weight().to_bits())
                            })
                            .collect(),
                    }
                }
            };
            let finest_sparsity = sparsity_snap(
                &workspace.finest_sparsity,
                "finest_sparsity",
                &mut usize_arcs,
            );
            let matrices = workspace
                .matrices
                .iter()
                .enumerate()
                .map(|(index, matrix)| {
                    let sparsity = sparsity_snap(
                        &matrix.sparsity_pattern(),
                        &format!("matrix[{index}]"),
                        &mut usize_arcs,
                    );
                    (
                        index,
                        MatrixValueSnap {
                            sparsity,
                            values: bits_vec_snap(&matrix.values),
                        },
                    )
                })
                .collect();
            let transfers = workspace
                .transfers
                .iter()
                .enumerate()
                .map(|(index, transfer)| {
                    (
                        index,
                        TransferValueSnap {
                            fine_to_coarse: usize_vec_snap(&transfer.fine_to_coarse),
                            fine_entry_to_coarse_entry: usize_vec_snap(
                                &transfer.fine_entry_to_coarse_entry,
                            ),
                        },
                    )
                })
                .collect();
            let coarsest_pcg = workspace.coarsest_pcg.as_ref().map(|pcg| {
                let sparsity =
                    sparsity_snap(&pcg.sparsity, "coarsest_pcg.sparsity", &mut usize_arcs);
                let preconditioner = match &pcg.preconditioner {
                    crate::linear::ReusablePreconditioner::None => PreconditionerValueSnap::None,
                    crate::linear::ReusablePreconditioner::Diagonal {
                        matrix_slots,
                        inverse,
                    } => PreconditionerValueSnap::Diagonal {
                        matrix_slots: usize_vec_snap(matrix_slots),
                        inverse: bits_vec_snap(inverse),
                    },
                    crate::linear::ReusablePreconditioner::IncompleteCholesky(ic) => {
                        let ic_sparsity = sparsity_snap(
                            &ic.sparsity,
                            "coarsest_pcg.ic.sparsity",
                            &mut usize_arcs,
                        );
                        PreconditionerValueSnap::IncompleteCholesky(Box::new(
                            IncompleteCholeskyValueSnap {
                                sparsity: ic_sparsity,
                                lower_row_offsets: usize_vec_snap(&ic.lower_row_offsets),
                                lower_columns: usize_vec_snap(&ic.lower_columns),
                                matrix_slots: usize_vec_snap(&ic.matrix_slots),
                                diagonal_factor_slots: usize_vec_snap(&ic.diagonal_factor_slots),
                                update_pairs: nested_pair_snap(&ic.update_pairs),
                                dependent_row_offsets: usize_vec_snap(&ic.dependent_row_offsets),
                                dependent_entries: pair_vec_snap(&ic.dependent_entries),
                                factors: bits_vec_snap(&ic.factors),
                            },
                        ))
                    }
                };
                PcgValueSnap {
                    sparsity,
                    preconditioner_kind: pcg.preconditioner_kind,
                    preconditioner,
                    residual: bits_vec_snap(&pcg.residual),
                    preconditioned_residual: bits_vec_snap(&pcg.preconditioned_residual),
                    direction: bits_vec_snap(&pcg.direction),
                    matrix_direction: bits_vec_snap(&pcg.matrix_direction),
                    preconditioner_scratch: bits_vec_snap(&pcg.preconditioner_scratch),
                }
            });
            FullSnapshot {
                workspace: WorkspaceValueSnap {
                    option_iterations: (
                        workspace.options.max_iterations,
                        workspace.options.min_iterations,
                        workspace.options.tolerance.to_bits(),
                        workspace.options.relative_tolerance.to_bits(),
                    ),
                    option_hierarchy: (
                        workspace.options.cache_agglomeration,
                        workspace.options.n_cells_in_coarsest_level,
                        workspace.options.merge_levels,
                        workspace.options.agglomerator,
                    ),
                    option_outer_solver: workspace.options.outer_solver,
                    option_smoothing: (
                        workspace.options.smoother,
                        workspace.options.n_pre_sweeps,
                        workspace.options.pre_sweeps_level_multiplier,
                        workspace.options.max_pre_sweeps,
                        workspace.options.n_post_sweeps,
                        workspace.options.post_sweeps_level_multiplier,
                        workspace.options.max_post_sweeps,
                    ),
                    option_correction: (
                        workspace.options.n_finest_sweeps,
                        workspace.options.interpolate_correction,
                        workspace.options.scale_correction,
                        workspace.options.direct_solve_coarsest,
                    ),
                    agglomeration,
                    finest_sparsity,
                    matrices_len: workspace.matrices.len(),
                    matrices_capacity: workspace.matrices.capacity(),
                    matrices,
                    transfers_len: workspace.transfers.len(),
                    transfers_capacity: workspace.transfers.capacity(),
                    transfers,
                    diagonal_slots: nested_usize_snap(&workspace.diagonal_slots),
                    diagonal_values: nested_bits_snap(&workspace.diagonal_values),
                    corrections: nested_bits_snap(&workspace.corrections),
                    sources: nested_bits_snap(&workspace.sources),
                    residuals: nested_bits_snap(&workspace.residuals),
                    products: nested_bits_snap(&workspace.products),
                    pre_smoothed: nested_bits_snap(&workspace.pre_smoothed),
                    fcg_residual: bits_vec_snap(&workspace.fcg_residual),
                    fcg_preconditioned_residual: bits_vec_snap(
                        &workspace.fcg_preconditioned_residual,
                    ),
                    fcg_direction: bits_vec_snap(&workspace.fcg_direction),
                    fcg_matrix_direction: bits_vec_snap(&workspace.fcg_matrix_direction),
                    fcg_previous_direction: bits_vec_snap(&workspace.fcg_previous_direction),
                    fcg_previous_matrix_direction: bits_vec_snap(
                        &workspace.fcg_previous_matrix_direction,
                    ),
                    coarsest_pcg,
                    profiled_hierarchy: workspace.profiled_hierarchy.as_deref().cloned(),
                    has_solved: workspace.has_solved,
                },
                timing: timing_snap(timing),
                initial: bits_vec_snap(initial),
                usize_arcs,
                face_arcs,
                profiled_hierarchy: workspace.profiled_hierarchy.clone(),
            }
        };

        let assert_same_snapshot = |before: &FullSnapshot, after: &FullSnapshot| {
            assert_eq!(after.workspace, before.workspace);
            assert_eq!(after.timing, before.timing);
            assert_eq!(after.initial, before.initial);
            assert_eq!(after.usize_arcs.len(), before.usize_arcs.len());
            for ((before_label, before_arc), (after_label, after_arc)) in
                before.usize_arcs.iter().zip(&after.usize_arcs)
            {
                assert_eq!(after_label, before_label);
                assert!(
                    std::sync::Arc::ptr_eq(before_arc, after_arc),
                    "Arc identity changed for {before_label}"
                );
            }
            for first in 0..before.usize_arcs.len() {
                for second in 0..before.usize_arcs.len() {
                    assert_eq!(
                        std::sync::Arc::ptr_eq(
                            &before.usize_arcs[first].1,
                            &before.usize_arcs[second].1,
                        ),
                        std::sync::Arc::ptr_eq(
                            &after.usize_arcs[first].1,
                            &after.usize_arcs[second].1,
                        ),
                        "Arc sharing relation changed for {} and {}",
                        before.usize_arcs[first].0,
                        before.usize_arcs[second].0,
                    );
                }
            }
            assert_eq!(after.face_arcs.len(), before.face_arcs.len());
            for (before_arc, after_arc) in before.face_arcs.iter().zip(&after.face_arcs) {
                assert!(std::sync::Arc::ptr_eq(before_arc, after_arc));
            }
            match (&before.profiled_hierarchy, &after.profiled_hierarchy) {
                (Some(before), Some(after)) => assert!(std::sync::Arc::ptr_eq(before, after)),
                (None, None) => {}
                _ => panic!("profiled hierarchy cache presence changed"),
            }
        };

        let matrix = CsrMatrix::from_rows(
            vec![
                vec![(0, 2.0), (1, -1.0)],
                vec![(0, -1.0), (1, 2.0), (2, -1.0)],
                vec![(1, -1.0), (2, 2.0), (3, -1.0)],
                vec![(2, -1.0), (3, 2.0)],
            ],
            4,
        )
        .expect("invalid-controls matrix");
        let rhs = [1.0, 0.0, 0.0, 0.0];
        let face_weights = [
            GamgFacePairWeight::new(0, 1, 4.0).expect("face 0-1"),
            GamgFacePairWeight::new(1, 2, 3.0).expect("face 1-2"),
            GamgFacePairWeight::new(2, 3, 2.0).expect("face 2-3"),
        ];
        let options = GamgOptions {
            max_iterations: 1,
            cache_agglomeration: false,
            n_cells_in_coarsest_level: 2,
            agglomerator: GamgAgglomerator::FaceAreaPair,
            direct_solve_coarsest: false,
            ..GamgOptions::default()
        };
        let mut workspace =
            GamgWorkspace::new_with_face_area_weights(&matrix, options, &face_weights)
                .expect("invalid workspace");
        workspace
            .solve(&matrix, &rhs, None)
            .expect("prime uncached FaceArea/PCG workspace");
        assert!(workspace.has_solved);

        let mut initial = Vec::with_capacity(11);
        initial.extend_from_slice(&[0.25, -0.5, 0.75, -1.0]);
        let seeded_timing = || super::GamgKernelTiming {
            total_seconds: 1.0,
            hierarchy_build_seconds: 2.0,
            hierarchy_rebuild_seconds: 3.0,
            hierarchy_diagnostic_seconds: 3.5,
            matrix_refresh_seconds: 4.0,
            finest_residual_seconds: 5.0,
            v_cycle_seconds: 6.0,
            other_seconds: 7.0,
            hierarchy_builds: 11,
            hierarchy_rebuilds: 12,
            matrix_refreshes: 13,
            finest_residual_evaluations: 14,
            solves: 15,
            v_cycles: 16,
            outer_matrix_vector_products: 17,
            outer_reductions: 18,
            levels: vec![
                super::GamgLevelTiming {
                    level: 0,
                    cells: 4,
                    nonzeros: 10,
                    matrix_refresh_seconds: 21.0,
                    restriction_seconds: 22.0,
                    prolongation_seconds: 23.0,
                    smoothing_seconds: 24.0,
                    scaling_seconds: 25.0,
                    residual_seconds: 26.0,
                    correction_seconds: 27.0,
                    coarsest_solve_seconds: 28.0,
                    matrix_refreshes: 31,
                    restriction_calls: 32,
                    prolongation_calls: 33,
                    smoothing_calls: 34,
                    smoothing_sweeps: 35,
                    scaling_calls: 36,
                    residual_evaluations: 37,
                    correction_updates: 38,
                    coarsest_solves: 39,
                    coarsest_iterations: 40,
                },
                super::GamgLevelTiming {
                    level: 1,
                    cells: 2,
                    nonzeros: 4,
                    matrix_refresh_seconds: 41.0,
                    restriction_seconds: 42.0,
                    prolongation_seconds: 43.0,
                    smoothing_seconds: 44.0,
                    scaling_seconds: 45.0,
                    residual_seconds: 46.0,
                    correction_seconds: 47.0,
                    coarsest_solve_seconds: 48.0,
                    matrix_refreshes: 51,
                    restriction_calls: 52,
                    prolongation_calls: 53,
                    smoothing_calls: 54,
                    smoothing_sweeps: 55,
                    scaling_calls: 56,
                    residual_evaluations: 57,
                    correction_updates: 58,
                    coarsest_solves: 59,
                    coarsest_iterations: 60,
                },
            ],
            hierarchy: None,
        };

        let ic_variant = snapshot(&workspace, &seeded_timing(), &initial);
        assert!(matches!(
            ic_variant.workspace.coarsest_pcg,
            Some(PcgValueSnap {
                preconditioner: PreconditionerValueSnap::IncompleteCholesky(_),
                ..
            })
        ));

        let none_options = GamgOptions {
            direct_solve_coarsest: true,
            ..options
        };
        let mut none_workspace =
            GamgWorkspace::new_with_face_area_weights(&matrix, none_options, &face_weights)
                .expect("None-PCG snapshot workspace");
        none_workspace
            .solve(&matrix, &rhs, None)
            .expect("prime None-PCG workspace");
        let none_variant = snapshot(&none_workspace, &seeded_timing(), &initial);
        assert!(none_variant.workspace.coarsest_pcg.is_none());

        let mut unpreconditioned_workspace =
            GamgWorkspace::new_with_face_area_weights(&matrix, options, &face_weights)
                .expect("unpreconditioned PCG snapshot workspace");
        let unpreconditioned_coarsest = unpreconditioned_workspace
            .matrices
            .last()
            .expect("unpreconditioned coarsest matrix")
            .clone();
        unpreconditioned_workspace.coarsest_pcg = Some(
            crate::linear::PreconditionedConjugateGradientWorkspace::new(
                &unpreconditioned_coarsest,
                CgPreconditioner::None,
            )
            .expect("unpreconditioned PCG workspace"),
        );
        let unpreconditioned_variant =
            snapshot(&unpreconditioned_workspace, &seeded_timing(), &initial);
        assert!(matches!(
            unpreconditioned_variant.workspace.coarsest_pcg,
            Some(PcgValueSnap {
                preconditioner_kind: CgPreconditioner::None,
                preconditioner: PreconditionerValueSnap::None,
                ..
            })
        ));

        let mut diagonal_workspace =
            GamgWorkspace::new_with_face_area_weights(&matrix, options, &face_weights)
                .expect("Diagonal-PCG snapshot workspace");
        let coarsest_matrix = diagonal_workspace
            .matrices
            .last()
            .expect("coarsest matrix")
            .clone();
        diagonal_workspace.coarsest_pcg = Some(
            crate::linear::PreconditionedConjugateGradientWorkspace::new(
                &coarsest_matrix,
                CgPreconditioner::Diagonal,
            )
            .expect("Diagonal PCG workspace"),
        );
        let diagonal_variant = snapshot(&diagonal_workspace, &seeded_timing(), &initial);
        assert!(matches!(
            diagonal_variant.workspace.coarsest_pcg,
            Some(PcgValueSnap {
                preconditioner: PreconditionerValueSnap::Diagonal { .. },
                ..
            })
        ));

        macro_rules! assert_normalized_invalid {
            ($controls:expr, $expected:expr) => {
                for route in 0..4 {
                    let controls = $controls;
                    let expected = ($expected).clone();
                    let mut timing = seeded_timing();
                    let before = snapshot(&workspace, &timing, &initial);
                    let error = match route {
                        0 => workspace
                            .solve_normalized_l1_with_controls(
                                &matrix,
                                &rhs,
                                Some(initial.as_slice()),
                                controls,
                            )
                            .expect_err("normalized public plain route must reject"),
                        1 => workspace
                            .solve_normalized_l1_with_controls_profiled(
                                &matrix,
                                &rhs,
                                Some(initial.as_slice()),
                                controls,
                            )
                            .expect_err("normalized public profiled route must reject"),
                        2 => workspace
                            .solve_normalized_l1_with_controls_internal::<false, false>(
                                &matrix,
                                &rhs,
                                Some(initial.as_slice()),
                                controls,
                                &mut timing,
                            )
                            .expect_err("normalized internal false route must reject"),
                        3 => workspace
                            .solve_normalized_l1_with_controls_internal::<true, false>(
                                &matrix,
                                &rhs,
                                Some(initial.as_slice()),
                                controls,
                                &mut timing,
                            )
                            .expect_err("normalized internal true route must reject"),
                        _ => unreachable!("exactly four normalized routes"),
                    };
                    assert_eq!(error.to_string(), expected, "normalized route={route}");
                    let crate::MeshError::InvalidInput(payload) = error else {
                        panic!("normalized route={route} returned the wrong error variant");
                    };
                    assert_eq!(payload, expected, "normalized route={route}");
                    let after = snapshot(&workspace, &timing, &initial);
                    assert_same_snapshot(&before, &after);
                }
            };
        }

        macro_rules! assert_legacy_invalid {
            ($controls:expr, $expected:expr) => {
                for route in 0..4 {
                    let controls = $controls;
                    let expected = ($expected).clone();
                    let mut timing = seeded_timing();
                    let before = snapshot(&workspace, &timing, &initial);
                    let error = match route {
                        0 => workspace
                            .solve_with_controls(&matrix, &rhs, Some(initial.as_slice()), controls)
                            .expect_err("legacy public plain route must reject"),
                        1 => workspace
                            .solve_with_controls_profiled(
                                &matrix,
                                &rhs,
                                Some(initial.as_slice()),
                                controls,
                            )
                            .expect_err("legacy public profiled route must reject"),
                        2 => workspace
                            .solve_with_controls_internal::<false, false>(
                                &matrix,
                                &rhs,
                                Some(initial.as_slice()),
                                controls,
                                &mut timing,
                            )
                            .expect_err("legacy internal false route must reject"),
                        3 => workspace
                            .solve_with_controls_internal::<true, false>(
                                &matrix,
                                &rhs,
                                Some(initial.as_slice()),
                                controls,
                                &mut timing,
                            )
                            .expect_err("legacy internal true route must reject"),
                        _ => unreachable!("exactly four legacy routes"),
                    };
                    assert_eq!(error.to_string(), expected, "legacy route={route}");
                    let crate::MeshError::InvalidInput(payload) = error else {
                        panic!("legacy route={route} returned the wrong error variant");
                    };
                    assert_eq!(payload, expected, "legacy route={route}");
                    let after = snapshot(&workspace, &timing, &initial);
                    assert_same_snapshot(&before, &after);
                }
            };
        }

        for value in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_normalized_invalid!(
                NormalizedL1GamgSolveControls {
                    normalization_factor: value,
                    tolerance: 0.0,
                    relative_tolerance: 0.0,
                    l2_controls: super::GamgSolveControls {
                        max_iterations: 1,
                        min_iterations: 0,
                        tolerance: 0.0,
                        relative_tolerance: 0.0,
                    },
                },
                format!("GAMG normalized-L1 factor must be finite and positive, got {value}")
            );
        }
        for value in [-1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_normalized_invalid!(
                NormalizedL1GamgSolveControls {
                    normalization_factor: 1.0,
                    tolerance: value,
                    relative_tolerance: 0.0,
                    l2_controls: super::GamgSolveControls {
                        max_iterations: 1,
                        min_iterations: 0,
                        tolerance: 0.0,
                        relative_tolerance: 0.0,
                    },
                },
                format!(
                    "GAMG normalized-L1 tolerance must be finite and non-negative, got {value}"
                )
            );
            assert_normalized_invalid!(
                NormalizedL1GamgSolveControls {
                    normalization_factor: 1.0,
                    tolerance: 0.0,
                    relative_tolerance: value,
                    l2_controls: super::GamgSolveControls {
                        max_iterations: 1,
                        min_iterations: 0,
                        tolerance: 0.0,
                        relative_tolerance: 0.0,
                    },
                },
                format!("GAMG normalized-L1 relTol must be finite and non-negative, got {value}")
            );
        }
        for value in [-1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let nested_tolerance = super::GamgSolveControls {
                max_iterations: 1,
                min_iterations: 0,
                tolerance: value,
                relative_tolerance: 0.0,
            };
            let tolerance_error =
                format!("GAMG tolerance must be finite and non-negative, got {value}");
            assert_normalized_invalid!(
                NormalizedL1GamgSolveControls {
                    normalization_factor: 1.0,
                    tolerance: 0.0,
                    relative_tolerance: 0.0,
                    l2_controls: nested_tolerance,
                },
                tolerance_error.clone()
            );
            assert_legacy_invalid!(nested_tolerance, tolerance_error);

            let nested_relative = super::GamgSolveControls {
                max_iterations: 1,
                min_iterations: 0,
                tolerance: 0.0,
                relative_tolerance: value,
            };
            let relative_error =
                format!("GAMG relTol must be finite and non-negative, got {value}");
            assert_normalized_invalid!(
                NormalizedL1GamgSolveControls {
                    normalization_factor: 1.0,
                    tolerance: 0.0,
                    relative_tolerance: 0.0,
                    l2_controls: nested_relative,
                },
                relative_error.clone()
            );
            assert_legacy_invalid!(nested_relative, relative_error);
        }

        #[derive(Clone, Copy, Debug)]
        enum ValidRoute {
            NormalizedPlain,
            NormalizedProfiled,
            LegacyPlain,
            LegacyProfiled,
        }
        let valid_normalized = NormalizedL1GamgSolveControls {
            normalization_factor: 1.0,
            tolerance: 0.0,
            relative_tolerance: 0.0,
            l2_controls: super::GamgSolveControls {
                max_iterations: 2,
                min_iterations: 2,
                tolerance: 0.0,
                relative_tolerance: 0.0,
            },
        };
        let valid_legacy = valid_normalized.l2_controls;
        let build_primed = || {
            let mut fresh =
                GamgWorkspace::new_with_face_area_weights(&matrix, options, &face_weights)
                    .expect("clean retry workspace");
            fresh
                .solve(&matrix, &rhs, None)
                .expect("prime clean retry workspace");
            fresh
        };
        let run_valid = |target: &mut GamgWorkspace,
                         route: ValidRoute|
         -> crate::Result<(
            super::IterativeSolveReport,
            Option<super::GamgKernelTiming>,
        )> {
            match route {
                ValidRoute::NormalizedPlain => Ok((
                    target.solve_normalized_l1_with_controls(
                        &matrix,
                        &rhs,
                        Some(initial.as_slice()),
                        valid_normalized,
                    )?,
                    None,
                )),
                ValidRoute::NormalizedProfiled => {
                    let profiled = target.solve_normalized_l1_with_controls_profiled(
                        &matrix,
                        &rhs,
                        Some(initial.as_slice()),
                        valid_normalized,
                    )?;
                    Ok((profiled.report, Some(profiled.timing)))
                }
                ValidRoute::LegacyPlain => Ok((
                    target.solve_with_controls(
                        &matrix,
                        &rhs,
                        Some(initial.as_slice()),
                        valid_legacy,
                    )?,
                    None,
                )),
                ValidRoute::LegacyProfiled => {
                    let profiled = target.solve_with_controls_profiled(
                        &matrix,
                        &rhs,
                        Some(initial.as_slice()),
                        valid_legacy,
                    )?;
                    Ok((profiled.report, Some(profiled.timing)))
                }
            }
        };
        let report_bits = |report: &super::IterativeSolveReport| {
            (
                report
                    .solution
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                report.iterations,
                report.residual_norm.to_bits(),
                report.converged,
                report.termination,
            )
        };
        let logical_timing = |timing: &super::GamgKernelTiming| {
            (
                [
                    timing.hierarchy_builds,
                    timing.hierarchy_rebuilds,
                    timing.matrix_refreshes,
                    timing.finest_residual_evaluations,
                    timing.solves,
                    timing.v_cycles,
                    timing.outer_matrix_vector_products,
                    timing.outer_reductions,
                ],
                timing
                    .levels
                    .iter()
                    .map(|level| {
                        (
                            [level.level, level.cells, level.nonzeros],
                            [
                                level.matrix_refreshes,
                                level.restriction_calls,
                                level.prolongation_calls,
                                level.smoothing_calls,
                                level.smoothing_sweeps,
                                level.scaling_calls,
                                level.residual_evaluations,
                                level.correction_updates,
                                level.coarsest_solves,
                                level.coarsest_iterations,
                            ],
                        )
                    })
                    .collect::<Vec<_>>(),
                timing.hierarchy.as_deref().cloned(),
            )
        };
        let assert_finite_timing = |timing: &super::GamgKernelTiming| {
            for seconds in [
                timing.total_seconds,
                timing.hierarchy_build_seconds,
                timing.hierarchy_rebuild_seconds,
                timing.hierarchy_diagnostic_seconds,
                timing.matrix_refresh_seconds,
                timing.finest_residual_seconds,
                timing.v_cycle_seconds,
                timing.other_seconds,
            ] {
                assert!(seconds.is_finite() && seconds >= 0.0);
            }
            for level in &timing.levels {
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
        };

        for route in [
            ValidRoute::NormalizedPlain,
            ValidRoute::NormalizedProfiled,
            ValidRoute::LegacyPlain,
            ValidRoute::LegacyProfiled,
        ] {
            let (live_report, live_timing) =
                run_valid(&mut workspace, route).expect("live post-invalid retry");
            let mut clean_workspace = build_primed();
            let (clean_report, clean_timing) =
                run_valid(&mut clean_workspace, route).expect("clean retry baseline");
            let default_clean = super::GamgKernelTiming::default();
            let clean_timing_ref = clean_timing.as_ref().unwrap_or(&default_clean);
            let clean_state = snapshot(&clean_workspace, clean_timing_ref, &initial);
            let (repeated_report, repeated_timing) =
                run_valid(&mut clean_workspace, route).expect("repeated retry baseline");
            let default_repeated = super::GamgKernelTiming::default();
            let repeated_timing_ref = repeated_timing.as_ref().unwrap_or(&default_repeated);
            let repeated_state = snapshot(&clean_workspace, repeated_timing_ref, &initial);

            assert_eq!(report_bits(&live_report), report_bits(&clean_report));
            assert_eq!(report_bits(&live_report), report_bits(&repeated_report));

            let default_live = super::GamgKernelTiming::default();
            let live_timing_ref = live_timing.as_ref().unwrap_or(&default_live);
            let live_state = snapshot(&workspace, live_timing_ref, &initial);
            assert_eq!(live_state.workspace, clean_state.workspace);
            assert_eq!(live_state.initial, clean_state.initial);
            assert_eq!(live_state.workspace, repeated_state.workspace);
            assert_eq!(live_state.initial, repeated_state.initial);
            assert_eq!(
                logical_timing(live_timing_ref),
                logical_timing(clean_timing_ref)
            );
            assert_eq!(
                logical_timing(live_timing_ref),
                logical_timing(repeated_timing_ref)
            );

            if live_timing.is_some() {
                let logical = logical_timing(live_timing_ref);
                assert_eq!(logical.0, [0, 1, 1, 3, 1, 2, 0, 0]);
                assert_eq!(
                    logical.1,
                    vec![
                        ([0, 4, 10], [1, 2, 2, 2, 4, 2, 0, 2, 0, 0]),
                        ([1, 2, 4], [1, 0, 0, 0, 0, 0, 0, 0, 2, 13]),
                    ]
                );
                assert!(logical.2.is_some());
                assert_finite_timing(live_timing_ref);
                assert_finite_timing(clean_timing_ref);
                assert_finite_timing(repeated_timing_ref);
            } else {
                assert_eq!(logical_timing(live_timing_ref), ([0; 8], vec![], None));
            }
        }
    }

    #[test]
    fn normalized_l1_solve_preserves_gamg_lifecycle_and_l2_report() {
        let matrix = CsrMatrix::from_rows(
            vec![vec![(0, 2.0), (1, -1.0)], vec![(0, -1.0), (1, 2.0)]],
            2,
        )
        .expect("lifecycle matrix");
        let exact_rhs = [1.0, 1.0];
        let rhs = [1.0, 0.0];
        let exact_initial = [1.0, 1.0];
        let options = GamgOptions {
            max_iterations: 2,
            min_iterations: 0,
            tolerance: 0.0,
            relative_tolerance: 0.0,
            cache_agglomeration: true,
            n_cells_in_coarsest_level: 1,
            merge_levels: 1,
            agglomerator: GamgAgglomerator::AlgebraicPair,
            smoother: GamgSmoother::GaussSeidel,
            outer_solver: GamgOuterSolver::Standalone,
            n_pre_sweeps: 0,
            pre_sweeps_level_multiplier: 1,
            max_pre_sweeps: 0,
            n_post_sweeps: 0,
            post_sweeps_level_multiplier: 1,
            max_post_sweeps: 0,
            n_finest_sweeps: 0,
            interpolate_correction: false,
            scale_correction: true,
            direct_solve_coarsest: true,
        };
        let exact_controls = NormalizedL1GamgSolveControls {
            normalization_factor: 1.0,
            tolerance: 0.0,
            relative_tolerance: 0.0,
            l2_controls: super::GamgSolveControls {
                max_iterations: 4,
                min_iterations: 0,
                tolerance: 0.0,
                relative_tolerance: 0.0,
            },
        };
        let first_controls = NormalizedL1GamgSolveControls {
            normalization_factor: 1.0,
            tolerance: 1.0,
            relative_tolerance: 0.0,
            l2_controls: super::GamgSolveControls {
                max_iterations: 4,
                min_iterations: 1,
                tolerance: 0.0,
                relative_tolerance: 0.0,
            },
        };
        let minimum_controls = NormalizedL1GamgSolveControls {
            normalization_factor: 1.0,
            tolerance: 1.0,
            relative_tolerance: 0.0,
            l2_controls: super::GamgSolveControls {
                max_iterations: 1,
                min_iterations: 2,
                tolerance: 0.0,
                relative_tolerance: 0.0,
            },
        };
        let maximum_controls = NormalizedL1GamgSolveControls {
            normalization_factor: 1.0,
            tolerance: 0.0,
            relative_tolerance: 0.0,
            l2_controls: super::GamgSolveControls {
                max_iterations: 2,
                min_iterations: 0,
                tolerance: 0.0,
                relative_tolerance: 0.0,
            },
        };
        macro_rules! assert_success_workspace {
            ($workspace:expr) => {{
                let workspace: &GamgWorkspace = $workspace;
                assert_eq!(workspace.level_sizes(), vec![2, 1]);
                assert!(workspace.has_solved);
                assert_eq!(
                    (
                        (
                            workspace.options.max_iterations,
                            workspace.options.min_iterations,
                            workspace.options.tolerance.to_bits(),
                            workspace.options.relative_tolerance.to_bits(),
                        ),
                        (
                            workspace.options.cache_agglomeration,
                            workspace.options.n_cells_in_coarsest_level,
                            workspace.options.merge_levels,
                            workspace.options.agglomerator,
                        ),
                        (
                            workspace.options.smoother,
                            workspace.options.n_pre_sweeps,
                            workspace.options.pre_sweeps_level_multiplier,
                            workspace.options.max_pre_sweeps,
                            workspace.options.n_post_sweeps,
                            workspace.options.post_sweeps_level_multiplier,
                            workspace.options.max_post_sweeps,
                        ),
                        (
                            workspace.options.n_finest_sweeps,
                            workspace.options.interpolate_correction,
                            workspace.options.scale_correction,
                            workspace.options.direct_solve_coarsest,
                        ),
                    ),
                    (
                        (2, 0, 0.0f64.to_bits(), 0.0f64.to_bits()),
                        (true, 1, 1, GamgAgglomerator::AlgebraicPair),
                        (GamgSmoother::GaussSeidel, 0, 1, 0, 0, 1, 0),
                        (0, false, true, true),
                    )
                );
            }};
        }
        macro_rules! assert_singular_workspace {
            ($workspace:expr) => {{
                let workspace: &GamgWorkspace = $workspace;
                assert_eq!(workspace.level_sizes(), vec![4, 2]);
                assert!(!workspace.has_solved);
                assert_eq!(
                    (
                        (
                            workspace.options.max_iterations,
                            workspace.options.min_iterations,
                            workspace.options.tolerance.to_bits(),
                            workspace.options.relative_tolerance.to_bits(),
                        ),
                        (
                            workspace.options.cache_agglomeration,
                            workspace.options.n_cells_in_coarsest_level,
                            workspace.options.merge_levels,
                            workspace.options.agglomerator,
                        ),
                        (
                            workspace.options.smoother,
                            workspace.options.n_pre_sweeps,
                            workspace.options.pre_sweeps_level_multiplier,
                            workspace.options.max_pre_sweeps,
                            workspace.options.n_post_sweeps,
                            workspace.options.post_sweeps_level_multiplier,
                            workspace.options.max_post_sweeps,
                        ),
                        (
                            workspace.options.n_finest_sweeps,
                            workspace.options.interpolate_correction,
                            workspace.options.scale_correction,
                            workspace.options.direct_solve_coarsest,
                        ),
                    ),
                    (
                        (2, 0, 0.0f64.to_bits(), 0.0f64.to_bits()),
                        (true, 2, 1, GamgAgglomerator::AlgebraicPair),
                        (GamgSmoother::GaussSeidel, 0, 1, 0, 0, 1, 0),
                        (0, false, true, true),
                    )
                );
            }};
        }

        // 1/10: exact-zero, plain internal engine.
        {
            let mut workspace =
                GamgWorkspace::new(&matrix, options).expect("plain exact-zero workspace");
            let mut timing = super::GamgKernelTiming::from_matrices(&workspace.matrices);
            let report = workspace
                .solve_normalized_l1_with_controls_internal::<false, false>(
                    &matrix,
                    &exact_rhs,
                    Some(&exact_initial),
                    exact_controls,
                    &mut timing,
                )
                .expect("plain exact-zero solve");
            let mut squared_l2 = 0.0;
            for (row, source) in exact_rhs.iter().enumerate() {
                let mut product = 0.0;
                for entry in matrix.row_offsets()[row]..matrix.row_offsets()[row + 1] {
                    product +=
                        matrix.values()[entry] * report.solution[matrix.col_indices()[entry]];
                }
                let residual = source - product;
                assert_eq!(residual.to_bits(), 0.0f64.to_bits());
                squared_l2 += residual * residual;
            }
            assert_eq!(
                report
                    .solution
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                exact_initial
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>()
            );
            assert_eq!(report.residual_norm.to_bits(), squared_l2.sqrt().to_bits());
            assert_eq!(report.iterations, 0);
            assert!(report.converged);
            assert_eq!(
                report.termination,
                super::IterativeSolveTermination::Converged
            );
            assert_eq!(workspace.level_sizes(), vec![2, 1]);
            assert!(workspace.has_solved);
            assert_success_workspace!(&workspace);
            assert_eq!(
                (
                    (
                        workspace.options.max_iterations,
                        workspace.options.min_iterations,
                        workspace.options.tolerance.to_bits(),
                        workspace.options.relative_tolerance.to_bits(),
                    ),
                    (
                        workspace.options.cache_agglomeration,
                        workspace.options.n_cells_in_coarsest_level,
                        workspace.options.merge_levels,
                        workspace.options.agglomerator,
                    ),
                    (
                        workspace.options.smoother,
                        workspace.options.n_pre_sweeps,
                        workspace.options.pre_sweeps_level_multiplier,
                        workspace.options.max_pre_sweeps,
                        workspace.options.n_post_sweeps,
                        workspace.options.post_sweeps_level_multiplier,
                        workspace.options.max_post_sweeps,
                    ),
                    (
                        workspace.options.n_finest_sweeps,
                        workspace.options.interpolate_correction,
                        workspace.options.scale_correction,
                        workspace.options.direct_solve_coarsest,
                    ),
                ),
                (
                    (2, 0, 0.0f64.to_bits(), 0.0f64.to_bits()),
                    (true, 1, 1, GamgAgglomerator::AlgebraicPair),
                    (GamgSmoother::GaussSeidel, 0, 1, 0, 0, 1, 0),
                    (0, false, true, true),
                )
            );
            assert_eq!(
                (
                    timing.hierarchy_builds,
                    timing.hierarchy_rebuilds,
                    timing.matrix_refreshes,
                    timing.finest_residual_evaluations,
                    timing.solves,
                    timing.v_cycles,
                ),
                (0, 0, 0, 0, 0, 0)
            );
            assert_eq!(
                timing
                    .levels
                    .iter()
                    .map(|level| (
                        level.level,
                        level.cells,
                        level.nonzeros,
                        level.matrix_refreshes,
                        level.restriction_calls,
                        level.prolongation_calls,
                        level.smoothing_calls,
                        level.smoothing_sweeps,
                        level.scaling_calls,
                        level.residual_evaluations,
                        level.correction_updates,
                        level.coarsest_solves,
                    ))
                    .collect::<Vec<_>>(),
                vec![
                    (0, 2, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0),
                    (1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0),
                ]
            );
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
            for level in &timing.levels {
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
        }

        // 2/10: exact-zero, profiled public entrypoint.
        {
            let mut workspace =
                GamgWorkspace::new(&matrix, options).expect("profiled exact-zero workspace");
            let profiled = workspace
                .solve_normalized_l1_with_controls_profiled(
                    &matrix,
                    &exact_rhs,
                    Some(&exact_initial),
                    exact_controls,
                )
                .expect("profiled exact-zero solve");
            let report = &profiled.report;
            let mut squared_l2 = 0.0;
            for (row, source) in exact_rhs.iter().enumerate() {
                let mut product = 0.0;
                for entry in matrix.row_offsets()[row]..matrix.row_offsets()[row + 1] {
                    product +=
                        matrix.values()[entry] * report.solution[matrix.col_indices()[entry]];
                }
                let residual = source - product;
                assert_eq!(residual.to_bits(), 0.0f64.to_bits());
                squared_l2 += residual * residual;
            }
            assert_eq!(
                report
                    .solution
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                exact_initial
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>()
            );
            assert_eq!(report.residual_norm.to_bits(), squared_l2.sqrt().to_bits());
            assert_eq!(report.iterations, 0);
            assert!(report.converged);
            assert_eq!(
                report.termination,
                super::IterativeSolveTermination::Converged
            );
            assert_eq!(workspace.level_sizes(), vec![2, 1]);
            assert!(workspace.has_solved);
            assert_success_workspace!(&workspace);
            assert_eq!(
                (
                    profiled.timing.hierarchy_builds,
                    profiled.timing.hierarchy_rebuilds,
                    profiled.timing.matrix_refreshes,
                    profiled.timing.finest_residual_evaluations,
                    profiled.timing.solves,
                    profiled.timing.v_cycles,
                ),
                (0, 0, 1, 1, 1, 0)
            );
            assert_eq!(
                profiled
                    .timing
                    .levels
                    .iter()
                    .map(|level| (
                        level.level,
                        level.cells,
                        level.nonzeros,
                        level.matrix_refreshes,
                        level.restriction_calls,
                        level.prolongation_calls,
                        level.smoothing_calls,
                        level.smoothing_sweeps,
                        level.scaling_calls,
                        level.residual_evaluations,
                        level.correction_updates,
                        level.coarsest_solves,
                    ))
                    .collect::<Vec<_>>(),
                vec![
                    (0, 2, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0),
                    (1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0),
                ]
            );
            for seconds in [
                profiled.timing.total_seconds,
                profiled.timing.hierarchy_build_seconds,
                profiled.timing.hierarchy_rebuild_seconds,
                profiled.timing.matrix_refresh_seconds,
                profiled.timing.finest_residual_seconds,
                profiled.timing.v_cycle_seconds,
                profiled.timing.other_seconds,
            ] {
                assert!(seconds.is_finite() && seconds >= 0.0);
            }
            for level in &profiled.timing.levels {
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
        }

        // 3/10: first eligible iteration, plain internal engine.
        {
            let mut workspace =
                GamgWorkspace::new(&matrix, options).expect("plain first-eligible workspace");
            let mut timing = super::GamgKernelTiming::from_matrices(&workspace.matrices);
            let report = workspace
                .solve_normalized_l1_with_controls_internal::<false, false>(
                    &matrix,
                    &rhs,
                    None,
                    first_controls,
                    &mut timing,
                )
                .expect("plain first-eligible solve");
            let mut expected = vec![0.0; rhs.len()];
            let mut expected_residual = vec![0.0; rhs.len()];
            for (row, source) in rhs.iter().enumerate() {
                let mut product = 0.0;
                for entry in matrix.row_offsets()[row]..matrix.row_offsets()[row + 1] {
                    product += matrix.values()[entry] * expected[matrix.col_indices()[entry]];
                }
                expected_residual[row] = source - product;
            }
            for _ in 0..1 {
                let mut coarse_diagonal = 0.0;
                for row in 0..matrix.rows() {
                    for entry in matrix.row_offsets()[row]..matrix.row_offsets()[row + 1] {
                        coarse_diagonal += matrix.values()[entry];
                    }
                }
                let coarse_source = expected_residual.iter().copied().sum::<f64>();
                let coarse_value = coarse_source / coarse_diagonal;
                let mut correction = vec![coarse_value; rhs.len()];
                let mut product = vec![0.0; rhs.len()];
                for (row, product_value) in product.iter_mut().enumerate().take(matrix.rows()) {
                    for entry in matrix.row_offsets()[row]..matrix.row_offsets()[row + 1] {
                        *product_value +=
                            matrix.values()[entry] * correction[matrix.col_indices()[entry]];
                    }
                }
                let mut numerator = 0.0;
                let mut denominator = 0.0;
                for index in 0..rhs.len() {
                    numerator += expected_residual[index] * correction[index];
                    denominator += product[index] * correction[index];
                }
                let denominator = if denominator.abs() < 1.0e-300_f64 {
                    if denominator.is_sign_negative() {
                        -1.0e-300_f64
                    } else {
                        1.0e-300_f64
                    }
                } else {
                    denominator
                };
                let factor = numerator / denominator;
                for row in 0..matrix.rows() {
                    let mut diagonal = None;
                    for entry in matrix.row_offsets()[row]..matrix.row_offsets()[row + 1] {
                        if matrix.col_indices()[entry] == row {
                            diagonal = Some(matrix.values()[entry]);
                        }
                    }
                    correction[row] = factor * correction[row]
                        + (expected_residual[row] - factor * product[row])
                            / diagonal.expect("oracle diagonal");
                    expected[row] += correction[row];
                }
                for (row, source) in rhs.iter().enumerate() {
                    let mut matrix_value = 0.0;
                    for entry in matrix.row_offsets()[row]..matrix.row_offsets()[row + 1] {
                        matrix_value +=
                            matrix.values()[entry] * expected[matrix.col_indices()[entry]];
                    }
                    expected_residual[row] = source - matrix_value;
                }
            }
            let squared_l2 = expected_residual
                .iter()
                .map(|residual| residual * residual)
                .sum::<f64>();
            assert_eq!(
                report
                    .solution
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                expected
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>()
            );
            assert_eq!(report.residual_norm.to_bits(), squared_l2.sqrt().to_bits());
            assert_eq!(report.iterations, 1);
            assert!(report.converged);
            assert_eq!(
                report.termination,
                super::IterativeSolveTermination::Converged
            );
            assert_eq!(workspace.level_sizes(), vec![2, 1]);
            assert!(workspace.has_solved);
            assert_success_workspace!(&workspace);
            assert_eq!(
                (
                    timing.hierarchy_builds,
                    timing.hierarchy_rebuilds,
                    timing.matrix_refreshes,
                    timing.finest_residual_evaluations,
                    timing.solves,
                    timing.v_cycles,
                ),
                (0, 0, 0, 0, 0, 0)
            );
            assert_eq!(
                timing
                    .levels
                    .iter()
                    .map(|level| (
                        level.level,
                        level.cells,
                        level.nonzeros,
                        level.matrix_refreshes,
                        level.restriction_calls,
                        level.prolongation_calls,
                        level.smoothing_calls,
                        level.smoothing_sweeps,
                        level.scaling_calls,
                        level.residual_evaluations,
                        level.correction_updates,
                        level.coarsest_solves,
                    ))
                    .collect::<Vec<_>>(),
                vec![
                    (0, 2, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0),
                    (1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0),
                ]
            );
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
            for level in &timing.levels {
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
        }

        // 4/10: first eligible iteration, profiled public entrypoint.
        {
            let mut workspace =
                GamgWorkspace::new(&matrix, options).expect("profiled first-eligible workspace");
            let profiled = workspace
                .solve_normalized_l1_with_controls_profiled(&matrix, &rhs, None, first_controls)
                .expect("profiled first-eligible solve");
            let report = &profiled.report;
            let mut expected = vec![0.0; rhs.len()];
            let mut expected_residual = vec![0.0; rhs.len()];
            for (row, source) in rhs.iter().enumerate() {
                let mut product = 0.0;
                for entry in matrix.row_offsets()[row]..matrix.row_offsets()[row + 1] {
                    product += matrix.values()[entry] * expected[matrix.col_indices()[entry]];
                }
                expected_residual[row] = source - product;
            }
            for _ in 0..1 {
                let mut coarse_diagonal = 0.0;
                for row in 0..matrix.rows() {
                    for entry in matrix.row_offsets()[row]..matrix.row_offsets()[row + 1] {
                        coarse_diagonal += matrix.values()[entry];
                    }
                }
                let coarse_source = expected_residual.iter().copied().sum::<f64>();
                let coarse_value = coarse_source / coarse_diagonal;
                let mut correction = vec![coarse_value; rhs.len()];
                let mut product = vec![0.0; rhs.len()];
                for (row, product_value) in product.iter_mut().enumerate().take(matrix.rows()) {
                    for entry in matrix.row_offsets()[row]..matrix.row_offsets()[row + 1] {
                        *product_value +=
                            matrix.values()[entry] * correction[matrix.col_indices()[entry]];
                    }
                }
                let mut numerator = 0.0;
                let mut denominator = 0.0;
                for index in 0..rhs.len() {
                    numerator += expected_residual[index] * correction[index];
                    denominator += product[index] * correction[index];
                }
                let denominator = if denominator.abs() < 1.0e-300_f64 {
                    if denominator.is_sign_negative() {
                        -1.0e-300_f64
                    } else {
                        1.0e-300_f64
                    }
                } else {
                    denominator
                };
                let factor = numerator / denominator;
                for row in 0..matrix.rows() {
                    let mut diagonal = None;
                    for entry in matrix.row_offsets()[row]..matrix.row_offsets()[row + 1] {
                        if matrix.col_indices()[entry] == row {
                            diagonal = Some(matrix.values()[entry]);
                        }
                    }
                    correction[row] = factor * correction[row]
                        + (expected_residual[row] - factor * product[row])
                            / diagonal.expect("oracle diagonal");
                    expected[row] += correction[row];
                }
                for (row, source) in rhs.iter().enumerate() {
                    let mut matrix_value = 0.0;
                    for entry in matrix.row_offsets()[row]..matrix.row_offsets()[row + 1] {
                        matrix_value +=
                            matrix.values()[entry] * expected[matrix.col_indices()[entry]];
                    }
                    expected_residual[row] = source - matrix_value;
                }
            }
            let squared_l2 = expected_residual
                .iter()
                .map(|residual| residual * residual)
                .sum::<f64>();
            assert_eq!(
                report
                    .solution
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                expected
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>()
            );
            assert_eq!(report.residual_norm.to_bits(), squared_l2.sqrt().to_bits());
            assert_eq!(report.iterations, 1);
            assert!(report.converged);
            assert_eq!(
                report.termination,
                super::IterativeSolveTermination::Converged
            );
            assert_eq!(workspace.level_sizes(), vec![2, 1]);
            assert!(workspace.has_solved);
            assert_success_workspace!(&workspace);
            assert_eq!(
                (
                    profiled.timing.hierarchy_builds,
                    profiled.timing.hierarchy_rebuilds,
                    profiled.timing.matrix_refreshes,
                    profiled.timing.finest_residual_evaluations,
                    profiled.timing.solves,
                    profiled.timing.v_cycles,
                ),
                (0, 0, 1, 2, 1, 1)
            );
            assert_eq!(
                profiled
                    .timing
                    .levels
                    .iter()
                    .map(|level| (
                        level.level,
                        level.cells,
                        level.nonzeros,
                        level.matrix_refreshes,
                        level.restriction_calls,
                        level.prolongation_calls,
                        level.smoothing_calls,
                        level.smoothing_sweeps,
                        level.scaling_calls,
                        level.residual_evaluations,
                        level.correction_updates,
                        level.coarsest_solves,
                    ))
                    .collect::<Vec<_>>(),
                vec![
                    (0, 2, 4, 1, 1, 1, 1, 0, 1, 0, 1, 0),
                    (1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1),
                ]
            );
            for seconds in [
                profiled.timing.total_seconds,
                profiled.timing.hierarchy_build_seconds,
                profiled.timing.hierarchy_rebuild_seconds,
                profiled.timing.matrix_refresh_seconds,
                profiled.timing.finest_residual_seconds,
                profiled.timing.v_cycle_seconds,
                profiled.timing.other_seconds,
            ] {
                assert!(seconds.is_finite() && seconds >= 0.0);
            }
            for level in &profiled.timing.levels {
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
        }

        // 5/10: min_iterations > 1, plain internal engine.
        {
            let mut workspace =
                GamgWorkspace::new(&matrix, options).expect("plain minimum workspace");
            let mut timing = super::GamgKernelTiming::from_matrices(&workspace.matrices);
            let report = workspace
                .solve_normalized_l1_with_controls_internal::<false, false>(
                    &matrix,
                    &rhs,
                    None,
                    minimum_controls,
                    &mut timing,
                )
                .expect("plain minimum solve");
            let mut expected = vec![0.0; rhs.len()];
            let mut expected_residual = vec![0.0; rhs.len()];
            for (row, source) in rhs.iter().enumerate() {
                let mut product = 0.0;
                for entry in matrix.row_offsets()[row]..matrix.row_offsets()[row + 1] {
                    product += matrix.values()[entry] * expected[matrix.col_indices()[entry]];
                }
                expected_residual[row] = source - product;
            }
            for _ in 0..2 {
                let mut coarse_diagonal = 0.0;
                for row in 0..matrix.rows() {
                    for entry in matrix.row_offsets()[row]..matrix.row_offsets()[row + 1] {
                        coarse_diagonal += matrix.values()[entry];
                    }
                }
                let coarse_source = expected_residual.iter().copied().sum::<f64>();
                let coarse_value = coarse_source / coarse_diagonal;
                let mut correction = vec![coarse_value; rhs.len()];
                let mut product = vec![0.0; rhs.len()];
                for (row, product_value) in product.iter_mut().enumerate().take(matrix.rows()) {
                    for entry in matrix.row_offsets()[row]..matrix.row_offsets()[row + 1] {
                        *product_value +=
                            matrix.values()[entry] * correction[matrix.col_indices()[entry]];
                    }
                }
                let mut numerator = 0.0;
                let mut denominator = 0.0;
                for index in 0..rhs.len() {
                    numerator += expected_residual[index] * correction[index];
                    denominator += product[index] * correction[index];
                }
                let denominator = if denominator.abs() < 1.0e-300_f64 {
                    if denominator.is_sign_negative() {
                        -1.0e-300_f64
                    } else {
                        1.0e-300_f64
                    }
                } else {
                    denominator
                };
                let factor = numerator / denominator;
                for row in 0..matrix.rows() {
                    let mut diagonal = None;
                    for entry in matrix.row_offsets()[row]..matrix.row_offsets()[row + 1] {
                        if matrix.col_indices()[entry] == row {
                            diagonal = Some(matrix.values()[entry]);
                        }
                    }
                    correction[row] = factor * correction[row]
                        + (expected_residual[row] - factor * product[row])
                            / diagonal.expect("oracle diagonal");
                    expected[row] += correction[row];
                }
                for (row, source) in rhs.iter().enumerate() {
                    let mut matrix_value = 0.0;
                    for entry in matrix.row_offsets()[row]..matrix.row_offsets()[row + 1] {
                        matrix_value +=
                            matrix.values()[entry] * expected[matrix.col_indices()[entry]];
                    }
                    expected_residual[row] = source - matrix_value;
                }
            }
            let squared_l2 = expected_residual
                .iter()
                .map(|residual| residual * residual)
                .sum::<f64>();
            assert_eq!(
                report
                    .solution
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                expected
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>()
            );
            assert_eq!(report.residual_norm.to_bits(), squared_l2.sqrt().to_bits());
            assert_eq!(report.iterations, 2);
            assert!(report.converged);
            assert_eq!(
                report.termination,
                super::IterativeSolveTermination::Converged
            );
            assert_eq!(workspace.level_sizes(), vec![2, 1]);
            assert!(workspace.has_solved);
            assert_success_workspace!(&workspace);
            assert_eq!(
                (
                    timing.hierarchy_builds,
                    timing.hierarchy_rebuilds,
                    timing.matrix_refreshes,
                    timing.finest_residual_evaluations,
                    timing.solves,
                    timing.v_cycles,
                ),
                (0, 0, 0, 0, 0, 0)
            );
            assert_eq!(
                timing
                    .levels
                    .iter()
                    .map(|level| (
                        level.level,
                        level.cells,
                        level.nonzeros,
                        level.matrix_refreshes,
                        level.restriction_calls,
                        level.prolongation_calls,
                        level.smoothing_calls,
                        level.smoothing_sweeps,
                        level.scaling_calls,
                        level.residual_evaluations,
                        level.correction_updates,
                        level.coarsest_solves,
                    ))
                    .collect::<Vec<_>>(),
                vec![
                    (0, 2, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0),
                    (1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0),
                ]
            );
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
            for level in &timing.levels {
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
        }

        // 6/10: min_iterations > 1, profiled public entrypoint.
        {
            let mut workspace =
                GamgWorkspace::new(&matrix, options).expect("profiled minimum workspace");
            let profiled = workspace
                .solve_normalized_l1_with_controls_profiled(&matrix, &rhs, None, minimum_controls)
                .expect("profiled minimum solve");
            let report = &profiled.report;
            let mut expected = vec![0.0; rhs.len()];
            let mut expected_residual = vec![0.0; rhs.len()];
            for (row, source) in rhs.iter().enumerate() {
                let mut product = 0.0;
                for entry in matrix.row_offsets()[row]..matrix.row_offsets()[row + 1] {
                    product += matrix.values()[entry] * expected[matrix.col_indices()[entry]];
                }
                expected_residual[row] = source - product;
            }
            for _ in 0..2 {
                let mut coarse_diagonal = 0.0;
                for row in 0..matrix.rows() {
                    for entry in matrix.row_offsets()[row]..matrix.row_offsets()[row + 1] {
                        coarse_diagonal += matrix.values()[entry];
                    }
                }
                let coarse_source = expected_residual.iter().copied().sum::<f64>();
                let coarse_value = coarse_source / coarse_diagonal;
                let mut correction = vec![coarse_value; rhs.len()];
                let mut product = vec![0.0; rhs.len()];
                for (row, product_value) in product.iter_mut().enumerate().take(matrix.rows()) {
                    for entry in matrix.row_offsets()[row]..matrix.row_offsets()[row + 1] {
                        *product_value +=
                            matrix.values()[entry] * correction[matrix.col_indices()[entry]];
                    }
                }
                let mut numerator = 0.0;
                let mut denominator = 0.0;
                for index in 0..rhs.len() {
                    numerator += expected_residual[index] * correction[index];
                    denominator += product[index] * correction[index];
                }
                let denominator = if denominator.abs() < 1.0e-300_f64 {
                    if denominator.is_sign_negative() {
                        -1.0e-300_f64
                    } else {
                        1.0e-300_f64
                    }
                } else {
                    denominator
                };
                let factor = numerator / denominator;
                for row in 0..matrix.rows() {
                    let mut diagonal = None;
                    for entry in matrix.row_offsets()[row]..matrix.row_offsets()[row + 1] {
                        if matrix.col_indices()[entry] == row {
                            diagonal = Some(matrix.values()[entry]);
                        }
                    }
                    correction[row] = factor * correction[row]
                        + (expected_residual[row] - factor * product[row])
                            / diagonal.expect("oracle diagonal");
                    expected[row] += correction[row];
                }
                for (row, source) in rhs.iter().enumerate() {
                    let mut matrix_value = 0.0;
                    for entry in matrix.row_offsets()[row]..matrix.row_offsets()[row + 1] {
                        matrix_value +=
                            matrix.values()[entry] * expected[matrix.col_indices()[entry]];
                    }
                    expected_residual[row] = source - matrix_value;
                }
            }
            let squared_l2 = expected_residual
                .iter()
                .map(|residual| residual * residual)
                .sum::<f64>();
            assert_eq!(
                report
                    .solution
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                expected
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>()
            );
            assert_eq!(report.residual_norm.to_bits(), squared_l2.sqrt().to_bits());
            assert_eq!(report.iterations, 2);
            assert!(report.converged);
            assert_eq!(
                report.termination,
                super::IterativeSolveTermination::Converged
            );
            assert_eq!(workspace.level_sizes(), vec![2, 1]);
            assert!(workspace.has_solved);
            assert_success_workspace!(&workspace);
            assert_eq!(
                (
                    profiled.timing.hierarchy_builds,
                    profiled.timing.hierarchy_rebuilds,
                    profiled.timing.matrix_refreshes,
                    profiled.timing.finest_residual_evaluations,
                    profiled.timing.solves,
                    profiled.timing.v_cycles,
                ),
                (0, 0, 1, 3, 1, 2)
            );
            assert_eq!(
                profiled
                    .timing
                    .levels
                    .iter()
                    .map(|level| (
                        level.level,
                        level.cells,
                        level.nonzeros,
                        level.matrix_refreshes,
                        level.restriction_calls,
                        level.prolongation_calls,
                        level.smoothing_calls,
                        level.smoothing_sweeps,
                        level.scaling_calls,
                        level.residual_evaluations,
                        level.correction_updates,
                        level.coarsest_solves,
                    ))
                    .collect::<Vec<_>>(),
                vec![
                    (0, 2, 4, 1, 2, 2, 2, 0, 2, 0, 2, 0),
                    (1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 2),
                ]
            );
            for seconds in [
                profiled.timing.total_seconds,
                profiled.timing.hierarchy_build_seconds,
                profiled.timing.hierarchy_rebuild_seconds,
                profiled.timing.matrix_refresh_seconds,
                profiled.timing.finest_residual_seconds,
                profiled.timing.v_cycle_seconds,
                profiled.timing.other_seconds,
            ] {
                assert!(seconds.is_finite() && seconds >= 0.0);
            }
            for level in &profiled.timing.levels {
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
        }

        // 7/10: exhausted iteration budget, plain internal engine.
        {
            let mut workspace =
                GamgWorkspace::new(&matrix, options).expect("plain maximum workspace");
            let mut timing = super::GamgKernelTiming::from_matrices(&workspace.matrices);
            let report = workspace
                .solve_normalized_l1_with_controls_internal::<false, false>(
                    &matrix,
                    &rhs,
                    None,
                    maximum_controls,
                    &mut timing,
                )
                .expect("plain maximum solve");
            let mut expected = vec![0.0; rhs.len()];
            let mut expected_residual = vec![0.0; rhs.len()];
            for (row, source) in rhs.iter().enumerate() {
                let mut product = 0.0;
                for entry in matrix.row_offsets()[row]..matrix.row_offsets()[row + 1] {
                    product += matrix.values()[entry] * expected[matrix.col_indices()[entry]];
                }
                expected_residual[row] = source - product;
            }
            for _ in 0..2 {
                let mut coarse_diagonal = 0.0;
                for row in 0..matrix.rows() {
                    for entry in matrix.row_offsets()[row]..matrix.row_offsets()[row + 1] {
                        coarse_diagonal += matrix.values()[entry];
                    }
                }
                let coarse_source = expected_residual.iter().copied().sum::<f64>();
                let coarse_value = coarse_source / coarse_diagonal;
                let mut correction = vec![coarse_value; rhs.len()];
                let mut product = vec![0.0; rhs.len()];
                for (row, product_value) in product.iter_mut().enumerate().take(matrix.rows()) {
                    for entry in matrix.row_offsets()[row]..matrix.row_offsets()[row + 1] {
                        *product_value +=
                            matrix.values()[entry] * correction[matrix.col_indices()[entry]];
                    }
                }
                let mut numerator = 0.0;
                let mut denominator = 0.0;
                for index in 0..rhs.len() {
                    numerator += expected_residual[index] * correction[index];
                    denominator += product[index] * correction[index];
                }
                let denominator = if denominator.abs() < 1.0e-300_f64 {
                    if denominator.is_sign_negative() {
                        -1.0e-300_f64
                    } else {
                        1.0e-300_f64
                    }
                } else {
                    denominator
                };
                let factor = numerator / denominator;
                for row in 0..matrix.rows() {
                    let mut diagonal = None;
                    for entry in matrix.row_offsets()[row]..matrix.row_offsets()[row + 1] {
                        if matrix.col_indices()[entry] == row {
                            diagonal = Some(matrix.values()[entry]);
                        }
                    }
                    correction[row] = factor * correction[row]
                        + (expected_residual[row] - factor * product[row])
                            / diagonal.expect("oracle diagonal");
                    expected[row] += correction[row];
                }
                for (row, source) in rhs.iter().enumerate() {
                    let mut matrix_value = 0.0;
                    for entry in matrix.row_offsets()[row]..matrix.row_offsets()[row + 1] {
                        matrix_value +=
                            matrix.values()[entry] * expected[matrix.col_indices()[entry]];
                    }
                    expected_residual[row] = source - matrix_value;
                }
            }
            let squared_l2 = expected_residual
                .iter()
                .map(|residual| residual * residual)
                .sum::<f64>();
            assert_eq!(
                report
                    .solution
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                expected
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>()
            );
            assert_eq!(report.residual_norm.to_bits(), squared_l2.sqrt().to_bits());
            assert_eq!(report.iterations, 2);
            assert!(!report.converged);
            assert_eq!(
                report.termination,
                super::IterativeSolveTermination::MaxIterations
            );
            assert_eq!(workspace.level_sizes(), vec![2, 1]);
            assert!(workspace.has_solved);
            assert_success_workspace!(&workspace);
            assert_eq!(
                (
                    timing.hierarchy_builds,
                    timing.hierarchy_rebuilds,
                    timing.matrix_refreshes,
                    timing.finest_residual_evaluations,
                    timing.solves,
                    timing.v_cycles,
                ),
                (0, 0, 0, 0, 0, 0)
            );
            assert_eq!(
                timing
                    .levels
                    .iter()
                    .map(|level| (
                        level.level,
                        level.cells,
                        level.nonzeros,
                        level.matrix_refreshes,
                        level.restriction_calls,
                        level.prolongation_calls,
                        level.smoothing_calls,
                        level.smoothing_sweeps,
                        level.scaling_calls,
                        level.residual_evaluations,
                        level.correction_updates,
                        level.coarsest_solves,
                    ))
                    .collect::<Vec<_>>(),
                vec![
                    (0, 2, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0),
                    (1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0),
                ]
            );
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
            for level in &timing.levels {
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
        }

        // 8/10: exhausted iteration budget, profiled public entrypoint.
        {
            let mut workspace =
                GamgWorkspace::new(&matrix, options).expect("profiled maximum workspace");
            let profiled = workspace
                .solve_normalized_l1_with_controls_profiled(&matrix, &rhs, None, maximum_controls)
                .expect("profiled maximum solve");
            let report = &profiled.report;
            let mut expected = vec![0.0; rhs.len()];
            let mut expected_residual = vec![0.0; rhs.len()];
            for (row, source) in rhs.iter().enumerate() {
                let mut product = 0.0;
                for entry in matrix.row_offsets()[row]..matrix.row_offsets()[row + 1] {
                    product += matrix.values()[entry] * expected[matrix.col_indices()[entry]];
                }
                expected_residual[row] = source - product;
            }
            for _ in 0..2 {
                let mut coarse_diagonal = 0.0;
                for row in 0..matrix.rows() {
                    for entry in matrix.row_offsets()[row]..matrix.row_offsets()[row + 1] {
                        coarse_diagonal += matrix.values()[entry];
                    }
                }
                let coarse_source = expected_residual.iter().copied().sum::<f64>();
                let coarse_value = coarse_source / coarse_diagonal;
                let mut correction = vec![coarse_value; rhs.len()];
                let mut product = vec![0.0; rhs.len()];
                for (row, product_value) in product.iter_mut().enumerate().take(matrix.rows()) {
                    for entry in matrix.row_offsets()[row]..matrix.row_offsets()[row + 1] {
                        *product_value +=
                            matrix.values()[entry] * correction[matrix.col_indices()[entry]];
                    }
                }
                let mut numerator = 0.0;
                let mut denominator = 0.0;
                for index in 0..rhs.len() {
                    numerator += expected_residual[index] * correction[index];
                    denominator += product[index] * correction[index];
                }
                let denominator = if denominator.abs() < 1.0e-300_f64 {
                    if denominator.is_sign_negative() {
                        -1.0e-300_f64
                    } else {
                        1.0e-300_f64
                    }
                } else {
                    denominator
                };
                let factor = numerator / denominator;
                for row in 0..matrix.rows() {
                    let mut diagonal = None;
                    for entry in matrix.row_offsets()[row]..matrix.row_offsets()[row + 1] {
                        if matrix.col_indices()[entry] == row {
                            diagonal = Some(matrix.values()[entry]);
                        }
                    }
                    correction[row] = factor * correction[row]
                        + (expected_residual[row] - factor * product[row])
                            / diagonal.expect("oracle diagonal");
                    expected[row] += correction[row];
                }
                for (row, source) in rhs.iter().enumerate() {
                    let mut matrix_value = 0.0;
                    for entry in matrix.row_offsets()[row]..matrix.row_offsets()[row + 1] {
                        matrix_value +=
                            matrix.values()[entry] * expected[matrix.col_indices()[entry]];
                    }
                    expected_residual[row] = source - matrix_value;
                }
            }
            let squared_l2 = expected_residual
                .iter()
                .map(|residual| residual * residual)
                .sum::<f64>();
            assert_eq!(
                report
                    .solution
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                expected
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>()
            );
            assert_eq!(report.residual_norm.to_bits(), squared_l2.sqrt().to_bits());
            assert_eq!(report.iterations, 2);
            assert!(!report.converged);
            assert_eq!(
                report.termination,
                super::IterativeSolveTermination::MaxIterations
            );
            assert_eq!(workspace.level_sizes(), vec![2, 1]);
            assert!(workspace.has_solved);
            assert_success_workspace!(&workspace);
            assert_eq!(
                (
                    profiled.timing.hierarchy_builds,
                    profiled.timing.hierarchy_rebuilds,
                    profiled.timing.matrix_refreshes,
                    profiled.timing.finest_residual_evaluations,
                    profiled.timing.solves,
                    profiled.timing.v_cycles,
                ),
                (0, 0, 1, 3, 1, 2)
            );
            assert_eq!(
                profiled
                    .timing
                    .levels
                    .iter()
                    .map(|level| (
                        level.level,
                        level.cells,
                        level.nonzeros,
                        level.matrix_refreshes,
                        level.restriction_calls,
                        level.prolongation_calls,
                        level.smoothing_calls,
                        level.smoothing_sweeps,
                        level.scaling_calls,
                        level.residual_evaluations,
                        level.correction_updates,
                        level.coarsest_solves,
                    ))
                    .collect::<Vec<_>>(),
                vec![
                    (0, 2, 4, 1, 2, 2, 2, 0, 2, 0, 2, 0),
                    (1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 2),
                ]
            );
            for seconds in [
                profiled.timing.total_seconds,
                profiled.timing.hierarchy_build_seconds,
                profiled.timing.hierarchy_rebuild_seconds,
                profiled.timing.matrix_refresh_seconds,
                profiled.timing.finest_residual_seconds,
                profiled.timing.v_cycle_seconds,
                profiled.timing.other_seconds,
            ] {
                assert!(seconds.is_finite() && seconds >= 0.0);
            }
            for level in &profiled.timing.levels {
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
        }

        let singular_matrix = CsrMatrix::from_rows(
            vec![
                vec![(0, 1.0), (1, -1.0)],
                vec![(0, -1.0), (1, 2.0), (2, -1.0)],
                vec![(1, -1.0), (2, 2.0), (3, -1.0)],
                vec![(2, -1.0), (3, 1.0)],
            ],
            4,
        )
        .expect("singular lifecycle matrix");
        let singular_rhs = [0.0; 4];
        let singular_options = GamgOptions {
            n_cells_in_coarsest_level: 2,
            ..options
        };
        let singular_controls = NormalizedL1GamgSolveControls {
            normalization_factor: 1.0,
            tolerance: 0.0,
            relative_tolerance: 0.0,
            l2_controls: super::GamgSolveControls {
                max_iterations: 1,
                min_iterations: 1,
                tolerance: 0.0,
                relative_tolerance: 0.0,
            },
        };

        // 9/10: singular direct coarsest solve, plain internal engine.
        {
            let mut workspace = GamgWorkspace::new(&singular_matrix, singular_options)
                .expect("plain singular workspace");
            assert_singular_workspace!(&workspace);
            let mut initial = Vec::<f64>::with_capacity(9);
            initial.extend_from_slice(&[0.0; 4]);
            let before_finest_row_arc = workspace.finest_sparsity.row_offsets.clone();
            let before_finest_column_arc = workspace.finest_sparsity.col_indices.clone();
            let before_matrix_row_arcs = workspace
                .matrices
                .iter()
                .map(|matrix| matrix.row_offsets.clone())
                .collect::<Vec<_>>();
            let before_matrix_column_arcs = workspace
                .matrices
                .iter()
                .map(|matrix| matrix.col_indices.clone())
                .collect::<Vec<_>>();
            let before_state = (
                (
                    (
                        workspace.options.max_iterations,
                        workspace.options.min_iterations,
                        workspace.options.tolerance.to_bits(),
                        workspace.options.relative_tolerance.to_bits(),
                    ),
                    (
                        workspace.options.cache_agglomeration,
                        workspace.options.n_cells_in_coarsest_level,
                        workspace.options.merge_levels,
                        workspace.options.agglomerator,
                    ),
                    (
                        workspace.options.smoother,
                        workspace.options.n_pre_sweeps,
                        workspace.options.pre_sweeps_level_multiplier,
                        workspace.options.max_pre_sweeps,
                        workspace.options.n_post_sweeps,
                        workspace.options.post_sweeps_level_multiplier,
                        workspace.options.max_post_sweeps,
                    ),
                    (
                        workspace.options.n_finest_sweeps,
                        workspace.options.interpolate_correction,
                        workspace.options.scale_correction,
                        workspace.options.direct_solve_coarsest,
                    ),
                ),
                matches!(
                    workspace.agglomeration_source,
                    super::GamgAgglomerationSource::Algebraic
                ),
                (
                    workspace.finest_sparsity.rows,
                    workspace.finest_sparsity.cols,
                    workspace.finest_sparsity.row_offsets.len(),
                    workspace.finest_sparsity.row_offsets.to_vec(),
                    workspace.finest_sparsity.col_indices.len(),
                    workspace.finest_sparsity.col_indices.to_vec(),
                ),
                (
                    workspace.matrices.len(),
                    workspace.matrices.capacity(),
                    workspace
                        .matrices
                        .iter()
                        .enumerate()
                        .map(|(index, matrix)| {
                            (
                                index,
                                matrix.rows,
                                matrix.cols,
                                matrix.row_offsets.len(),
                                matrix.row_offsets.to_vec(),
                                matrix.col_indices.len(),
                                matrix.col_indices.to_vec(),
                                matrix.values.len(),
                                matrix.values.capacity(),
                                matrix
                                    .values
                                    .iter()
                                    .map(|value| value.to_bits())
                                    .collect::<Vec<_>>(),
                            )
                        })
                        .collect::<Vec<_>>(),
                ),
                (
                    workspace.transfers.len(),
                    workspace.transfers.capacity(),
                    workspace
                        .transfers
                        .iter()
                        .enumerate()
                        .map(|(index, transfer)| {
                            (
                                index,
                                (
                                    transfer.fine_to_coarse.len(),
                                    transfer.fine_to_coarse.capacity(),
                                    transfer.fine_to_coarse.clone(),
                                ),
                                (
                                    transfer.fine_entry_to_coarse_entry.len(),
                                    transfer.fine_entry_to_coarse_entry.capacity(),
                                    transfer.fine_entry_to_coarse_entry.clone(),
                                ),
                            )
                        })
                        .collect::<Vec<_>>(),
                ),
                (
                    (
                        workspace.diagonal_slots.len(),
                        workspace.diagonal_slots.capacity(),
                        workspace
                            .diagonal_slots
                            .iter()
                            .enumerate()
                            .map(|(index, values)| {
                                (index, values.len(), values.capacity(), values.clone())
                            })
                            .collect::<Vec<_>>(),
                    ),
                    (
                        workspace.corrections.len(),
                        workspace.corrections.capacity(),
                        workspace
                            .corrections
                            .iter()
                            .enumerate()
                            .map(|(index, values)| {
                                (
                                    index,
                                    values.len(),
                                    values.capacity(),
                                    values
                                        .iter()
                                        .map(|value| value.to_bits())
                                        .collect::<Vec<_>>(),
                                )
                            })
                            .collect::<Vec<_>>(),
                    ),
                    (
                        workspace.sources.len(),
                        workspace.sources.capacity(),
                        workspace
                            .sources
                            .iter()
                            .enumerate()
                            .map(|(index, values)| {
                                (
                                    index,
                                    values.len(),
                                    values.capacity(),
                                    values
                                        .iter()
                                        .map(|value| value.to_bits())
                                        .collect::<Vec<_>>(),
                                )
                            })
                            .collect::<Vec<_>>(),
                    ),
                    (
                        workspace.residuals.len(),
                        workspace.residuals.capacity(),
                        workspace
                            .residuals
                            .iter()
                            .enumerate()
                            .map(|(index, values)| {
                                (
                                    index,
                                    values.len(),
                                    values.capacity(),
                                    values
                                        .iter()
                                        .map(|value| value.to_bits())
                                        .collect::<Vec<_>>(),
                                )
                            })
                            .collect::<Vec<_>>(),
                    ),
                    (
                        workspace.products.len(),
                        workspace.products.capacity(),
                        workspace
                            .products
                            .iter()
                            .enumerate()
                            .map(|(index, values)| {
                                (
                                    index,
                                    values.len(),
                                    values.capacity(),
                                    values
                                        .iter()
                                        .map(|value| value.to_bits())
                                        .collect::<Vec<_>>(),
                                )
                            })
                            .collect::<Vec<_>>(),
                    ),
                    (
                        workspace.pre_smoothed.len(),
                        workspace.pre_smoothed.capacity(),
                        workspace
                            .pre_smoothed
                            .iter()
                            .enumerate()
                            .map(|(index, values)| {
                                (
                                    index,
                                    values.len(),
                                    values.capacity(),
                                    values
                                        .iter()
                                        .map(|value| value.to_bits())
                                        .collect::<Vec<_>>(),
                                )
                            })
                            .collect::<Vec<_>>(),
                    ),
                ),
                workspace.coarsest_pcg.is_none(),
                workspace.has_solved,
                workspace.level_sizes(),
                (
                    initial.len(),
                    initial.capacity(),
                    initial
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>(),
                ),
            );
            assert!(std::sync::Arc::ptr_eq(
                &before_finest_row_arc,
                &before_matrix_row_arcs[0]
            ));
            assert!(std::sync::Arc::ptr_eq(
                &before_finest_column_arc,
                &before_matrix_column_arcs[0]
            ));

            let mut timing = super::GamgKernelTiming::from_matrices(&workspace.matrices);
            let error = workspace
                .solve_normalized_l1_with_controls_internal::<false, false>(
                    &singular_matrix,
                    &singular_rhs,
                    Some(initial.as_slice()),
                    singular_controls,
                    &mut timing,
                )
                .expect_err("plain singular solve must fail");
            let expected_error =
                "GAMG direct coarsest solve has a singular pivot in column 1".to_string();
            assert_eq!(error.to_string(), expected_error);
            let crate::MeshError::InvalidInput(payload) = error else {
                panic!("plain singular solve returned the wrong error variant");
            };
            assert_eq!(payload, expected_error);

            let after_finest_row_arc = workspace.finest_sparsity.row_offsets.clone();
            let after_finest_column_arc = workspace.finest_sparsity.col_indices.clone();
            let after_matrix_row_arcs = workspace
                .matrices
                .iter()
                .map(|matrix| matrix.row_offsets.clone())
                .collect::<Vec<_>>();
            let after_matrix_column_arcs = workspace
                .matrices
                .iter()
                .map(|matrix| matrix.col_indices.clone())
                .collect::<Vec<_>>();
            let after_state = (
                (
                    (
                        workspace.options.max_iterations,
                        workspace.options.min_iterations,
                        workspace.options.tolerance.to_bits(),
                        workspace.options.relative_tolerance.to_bits(),
                    ),
                    (
                        workspace.options.cache_agglomeration,
                        workspace.options.n_cells_in_coarsest_level,
                        workspace.options.merge_levels,
                        workspace.options.agglomerator,
                    ),
                    (
                        workspace.options.smoother,
                        workspace.options.n_pre_sweeps,
                        workspace.options.pre_sweeps_level_multiplier,
                        workspace.options.max_pre_sweeps,
                        workspace.options.n_post_sweeps,
                        workspace.options.post_sweeps_level_multiplier,
                        workspace.options.max_post_sweeps,
                    ),
                    (
                        workspace.options.n_finest_sweeps,
                        workspace.options.interpolate_correction,
                        workspace.options.scale_correction,
                        workspace.options.direct_solve_coarsest,
                    ),
                ),
                matches!(
                    workspace.agglomeration_source,
                    super::GamgAgglomerationSource::Algebraic
                ),
                (
                    workspace.finest_sparsity.rows,
                    workspace.finest_sparsity.cols,
                    workspace.finest_sparsity.row_offsets.len(),
                    workspace.finest_sparsity.row_offsets.to_vec(),
                    workspace.finest_sparsity.col_indices.len(),
                    workspace.finest_sparsity.col_indices.to_vec(),
                ),
                (
                    workspace.matrices.len(),
                    workspace.matrices.capacity(),
                    workspace
                        .matrices
                        .iter()
                        .enumerate()
                        .map(|(index, matrix)| {
                            (
                                index,
                                matrix.rows,
                                matrix.cols,
                                matrix.row_offsets.len(),
                                matrix.row_offsets.to_vec(),
                                matrix.col_indices.len(),
                                matrix.col_indices.to_vec(),
                                matrix.values.len(),
                                matrix.values.capacity(),
                                matrix
                                    .values
                                    .iter()
                                    .map(|value| value.to_bits())
                                    .collect::<Vec<_>>(),
                            )
                        })
                        .collect::<Vec<_>>(),
                ),
                (
                    workspace.transfers.len(),
                    workspace.transfers.capacity(),
                    workspace
                        .transfers
                        .iter()
                        .enumerate()
                        .map(|(index, transfer)| {
                            (
                                index,
                                (
                                    transfer.fine_to_coarse.len(),
                                    transfer.fine_to_coarse.capacity(),
                                    transfer.fine_to_coarse.clone(),
                                ),
                                (
                                    transfer.fine_entry_to_coarse_entry.len(),
                                    transfer.fine_entry_to_coarse_entry.capacity(),
                                    transfer.fine_entry_to_coarse_entry.clone(),
                                ),
                            )
                        })
                        .collect::<Vec<_>>(),
                ),
                (
                    (
                        workspace.diagonal_slots.len(),
                        workspace.diagonal_slots.capacity(),
                        workspace
                            .diagonal_slots
                            .iter()
                            .enumerate()
                            .map(|(index, values)| {
                                (index, values.len(), values.capacity(), values.clone())
                            })
                            .collect::<Vec<_>>(),
                    ),
                    (
                        workspace.corrections.len(),
                        workspace.corrections.capacity(),
                        workspace
                            .corrections
                            .iter()
                            .enumerate()
                            .map(|(index, values)| {
                                (
                                    index,
                                    values.len(),
                                    values.capacity(),
                                    values
                                        .iter()
                                        .map(|value| value.to_bits())
                                        .collect::<Vec<_>>(),
                                )
                            })
                            .collect::<Vec<_>>(),
                    ),
                    (
                        workspace.sources.len(),
                        workspace.sources.capacity(),
                        workspace
                            .sources
                            .iter()
                            .enumerate()
                            .map(|(index, values)| {
                                (
                                    index,
                                    values.len(),
                                    values.capacity(),
                                    values
                                        .iter()
                                        .map(|value| value.to_bits())
                                        .collect::<Vec<_>>(),
                                )
                            })
                            .collect::<Vec<_>>(),
                    ),
                    (
                        workspace.residuals.len(),
                        workspace.residuals.capacity(),
                        workspace
                            .residuals
                            .iter()
                            .enumerate()
                            .map(|(index, values)| {
                                (
                                    index,
                                    values.len(),
                                    values.capacity(),
                                    values
                                        .iter()
                                        .map(|value| value.to_bits())
                                        .collect::<Vec<_>>(),
                                )
                            })
                            .collect::<Vec<_>>(),
                    ),
                    (
                        workspace.products.len(),
                        workspace.products.capacity(),
                        workspace
                            .products
                            .iter()
                            .enumerate()
                            .map(|(index, values)| {
                                (
                                    index,
                                    values.len(),
                                    values.capacity(),
                                    values
                                        .iter()
                                        .map(|value| value.to_bits())
                                        .collect::<Vec<_>>(),
                                )
                            })
                            .collect::<Vec<_>>(),
                    ),
                    (
                        workspace.pre_smoothed.len(),
                        workspace.pre_smoothed.capacity(),
                        workspace
                            .pre_smoothed
                            .iter()
                            .enumerate()
                            .map(|(index, values)| {
                                (
                                    index,
                                    values.len(),
                                    values.capacity(),
                                    values
                                        .iter()
                                        .map(|value| value.to_bits())
                                        .collect::<Vec<_>>(),
                                )
                            })
                            .collect::<Vec<_>>(),
                    ),
                ),
                workspace.coarsest_pcg.is_none(),
                workspace.has_solved,
                workspace.level_sizes(),
                (
                    initial.len(),
                    initial.capacity(),
                    initial
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>(),
                ),
            );
            assert_eq!(after_state, before_state);
            assert_singular_workspace!(&workspace);
            assert!(std::sync::Arc::ptr_eq(
                &before_finest_row_arc,
                &after_finest_row_arc
            ));
            assert!(std::sync::Arc::ptr_eq(
                &before_finest_column_arc,
                &after_finest_column_arc
            ));
            for (before, after) in before_matrix_row_arcs.iter().zip(&after_matrix_row_arcs) {
                assert!(std::sync::Arc::ptr_eq(before, after));
            }
            for (before, after) in before_matrix_column_arcs
                .iter()
                .zip(&after_matrix_column_arcs)
            {
                assert!(std::sync::Arc::ptr_eq(before, after));
            }
            assert_eq!(
                (
                    timing.hierarchy_builds,
                    timing.hierarchy_rebuilds,
                    timing.matrix_refreshes,
                    timing.finest_residual_evaluations,
                    timing.solves,
                    timing.v_cycles,
                ),
                (0, 0, 0, 0, 0, 0)
            );
            assert_eq!(
                timing
                    .levels
                    .iter()
                    .map(|level| (
                        level.level,
                        level.cells,
                        level.nonzeros,
                        level.matrix_refreshes,
                        level.restriction_calls,
                        level.prolongation_calls,
                        level.smoothing_calls,
                        level.smoothing_sweeps,
                        level.scaling_calls,
                        level.residual_evaluations,
                        level.correction_updates,
                        level.coarsest_solves,
                    ))
                    .collect::<Vec<_>>(),
                vec![
                    (0, 4, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0),
                    (1, 2, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0),
                ]
            );
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
            for level in &timing.levels {
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
        }

        // 10/10: singular direct coarsest solve, profiled internal engine.
        {
            let mut workspace = GamgWorkspace::new(&singular_matrix, singular_options)
                .expect("profiled singular workspace");
            assert_singular_workspace!(&workspace);
            let mut initial = Vec::<f64>::with_capacity(9);
            initial.extend_from_slice(&[0.0; 4]);
            let before_options = (
                (
                    workspace.options.max_iterations,
                    workspace.options.min_iterations,
                    workspace.options.tolerance.to_bits(),
                    workspace.options.relative_tolerance.to_bits(),
                ),
                (
                    workspace.options.cache_agglomeration,
                    workspace.options.n_cells_in_coarsest_level,
                    workspace.options.merge_levels,
                    workspace.options.agglomerator,
                ),
                (
                    workspace.options.smoother,
                    workspace.options.n_pre_sweeps,
                    workspace.options.pre_sweeps_level_multiplier,
                    workspace.options.max_pre_sweeps,
                    workspace.options.n_post_sweeps,
                    workspace.options.post_sweeps_level_multiplier,
                    workspace.options.max_post_sweeps,
                ),
                (
                    workspace.options.n_finest_sweeps,
                    workspace.options.interpolate_correction,
                    workspace.options.scale_correction,
                    workspace.options.direct_solve_coarsest,
                ),
            );
            let before_agglomeration = matches!(
                workspace.agglomeration_source,
                super::GamgAgglomerationSource::Algebraic
            );
            let before_finest = (
                workspace.finest_sparsity.rows,
                workspace.finest_sparsity.cols,
                workspace.finest_sparsity.row_offsets.len(),
                workspace.finest_sparsity.row_offsets.to_vec(),
                workspace.finest_sparsity.col_indices.len(),
                workspace.finest_sparsity.col_indices.to_vec(),
            );
            let before_finest_row_arc = workspace.finest_sparsity.row_offsets.clone();
            let before_finest_column_arc = workspace.finest_sparsity.col_indices.clone();
            let before_matrices = (
                workspace.matrices.len(),
                workspace.matrices.capacity(),
                workspace
                    .matrices
                    .iter()
                    .enumerate()
                    .map(|(index, matrix)| {
                        (
                            index,
                            matrix.rows,
                            matrix.cols,
                            matrix.row_offsets.len(),
                            matrix.row_offsets.to_vec(),
                            matrix.col_indices.len(),
                            matrix.col_indices.to_vec(),
                            matrix.values.len(),
                            matrix.values.capacity(),
                            matrix
                                .values
                                .iter()
                                .map(|value| value.to_bits())
                                .collect::<Vec<_>>(),
                        )
                    })
                    .collect::<Vec<_>>(),
            );
            let before_matrix_row_arcs = workspace
                .matrices
                .iter()
                .map(|matrix| matrix.row_offsets.clone())
                .collect::<Vec<_>>();
            let before_matrix_column_arcs = workspace
                .matrices
                .iter()
                .map(|matrix| matrix.col_indices.clone())
                .collect::<Vec<_>>();
            let before_transfers = (
                workspace.transfers.len(),
                workspace.transfers.capacity(),
                workspace
                    .transfers
                    .iter()
                    .enumerate()
                    .map(|(index, transfer)| {
                        (
                            index,
                            (
                                transfer.fine_to_coarse.len(),
                                transfer.fine_to_coarse.capacity(),
                                transfer.fine_to_coarse.clone(),
                            ),
                            (
                                transfer.fine_entry_to_coarse_entry.len(),
                                transfer.fine_entry_to_coarse_entry.capacity(),
                                transfer.fine_entry_to_coarse_entry.clone(),
                            ),
                        )
                    })
                    .collect::<Vec<_>>(),
            );
            let before_diagonal = (
                workspace.diagonal_slots.len(),
                workspace.diagonal_slots.capacity(),
                workspace
                    .diagonal_slots
                    .iter()
                    .enumerate()
                    .map(|(index, values)| (index, values.len(), values.capacity(), values.clone()))
                    .collect::<Vec<_>>(),
            );
            let before_corrections = (
                workspace.corrections.len(),
                workspace.corrections.capacity(),
                workspace
                    .corrections
                    .iter()
                    .enumerate()
                    .map(|(index, values)| {
                        (
                            index,
                            values.len(),
                            values.capacity(),
                            values
                                .iter()
                                .map(|value| value.to_bits())
                                .collect::<Vec<_>>(),
                        )
                    })
                    .collect::<Vec<_>>(),
            );
            let before_sources = (
                workspace.sources.len(),
                workspace.sources.capacity(),
                workspace
                    .sources
                    .iter()
                    .enumerate()
                    .map(|(index, values)| {
                        (
                            index,
                            values.len(),
                            values.capacity(),
                            values
                                .iter()
                                .map(|value| value.to_bits())
                                .collect::<Vec<_>>(),
                        )
                    })
                    .collect::<Vec<_>>(),
            );
            let before_residuals = (
                workspace.residuals.len(),
                workspace.residuals.capacity(),
                workspace
                    .residuals
                    .iter()
                    .enumerate()
                    .map(|(index, values)| {
                        (
                            index,
                            values.len(),
                            values.capacity(),
                            values
                                .iter()
                                .map(|value| value.to_bits())
                                .collect::<Vec<_>>(),
                        )
                    })
                    .collect::<Vec<_>>(),
            );
            let before_products = (
                workspace.products.len(),
                workspace.products.capacity(),
                workspace
                    .products
                    .iter()
                    .enumerate()
                    .map(|(index, values)| {
                        (
                            index,
                            values.len(),
                            values.capacity(),
                            values
                                .iter()
                                .map(|value| value.to_bits())
                                .collect::<Vec<_>>(),
                        )
                    })
                    .collect::<Vec<_>>(),
            );
            let before_pre_smoothed = (
                workspace.pre_smoothed.len(),
                workspace.pre_smoothed.capacity(),
                workspace
                    .pre_smoothed
                    .iter()
                    .enumerate()
                    .map(|(index, values)| {
                        (
                            index,
                            values.len(),
                            values.capacity(),
                            values
                                .iter()
                                .map(|value| value.to_bits())
                                .collect::<Vec<_>>(),
                        )
                    })
                    .collect::<Vec<_>>(),
            );
            let before_pcg_none = workspace.coarsest_pcg.is_none();
            let before_has_solved = workspace.has_solved;
            let before_levels = workspace.level_sizes();
            let before_initial = (
                initial.len(),
                initial.capacity(),
                initial
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
            );
            assert!(std::sync::Arc::ptr_eq(
                &before_finest_row_arc,
                &before_matrix_row_arcs[0]
            ));
            assert!(std::sync::Arc::ptr_eq(
                &before_finest_column_arc,
                &before_matrix_column_arcs[0]
            ));

            let mut timing = super::GamgKernelTiming::from_matrices(&workspace.matrices);
            let error = workspace
                .solve_normalized_l1_with_controls_internal::<true, false>(
                    &singular_matrix,
                    &singular_rhs,
                    Some(initial.as_slice()),
                    singular_controls,
                    &mut timing,
                )
                .expect_err("profiled singular solve must fail");
            let expected_error =
                "GAMG direct coarsest solve has a singular pivot in column 1".to_string();
            assert_eq!(error.to_string(), expected_error);
            let crate::MeshError::InvalidInput(payload) = error else {
                panic!("profiled singular solve returned the wrong error variant");
            };
            assert_eq!(payload, expected_error);

            assert_eq!(
                (
                    (
                        workspace.options.max_iterations,
                        workspace.options.min_iterations,
                        workspace.options.tolerance.to_bits(),
                        workspace.options.relative_tolerance.to_bits(),
                    ),
                    (
                        workspace.options.cache_agglomeration,
                        workspace.options.n_cells_in_coarsest_level,
                        workspace.options.merge_levels,
                        workspace.options.agglomerator,
                    ),
                    (
                        workspace.options.smoother,
                        workspace.options.n_pre_sweeps,
                        workspace.options.pre_sweeps_level_multiplier,
                        workspace.options.max_pre_sweeps,
                        workspace.options.n_post_sweeps,
                        workspace.options.post_sweeps_level_multiplier,
                        workspace.options.max_post_sweeps,
                    ),
                    (
                        workspace.options.n_finest_sweeps,
                        workspace.options.interpolate_correction,
                        workspace.options.scale_correction,
                        workspace.options.direct_solve_coarsest,
                    ),
                ),
                before_options
            );
            assert_eq!(
                matches!(
                    workspace.agglomeration_source,
                    super::GamgAgglomerationSource::Algebraic
                ),
                before_agglomeration
            );
            assert_eq!(
                (
                    workspace.finest_sparsity.rows,
                    workspace.finest_sparsity.cols,
                    workspace.finest_sparsity.row_offsets.len(),
                    workspace.finest_sparsity.row_offsets.to_vec(),
                    workspace.finest_sparsity.col_indices.len(),
                    workspace.finest_sparsity.col_indices.to_vec(),
                ),
                before_finest
            );
            assert_eq!(
                (
                    workspace.matrices.len(),
                    workspace.matrices.capacity(),
                    workspace
                        .matrices
                        .iter()
                        .enumerate()
                        .map(|(index, matrix)| (
                            index,
                            matrix.rows,
                            matrix.cols,
                            matrix.row_offsets.len(),
                            matrix.row_offsets.to_vec(),
                            matrix.col_indices.len(),
                            matrix.col_indices.to_vec(),
                            matrix.values.len(),
                            matrix.values.capacity(),
                            matrix
                                .values
                                .iter()
                                .map(|value| value.to_bits())
                                .collect::<Vec<_>>(),
                        ))
                        .collect::<Vec<_>>(),
                ),
                before_matrices
            );
            assert_eq!(
                (
                    workspace.transfers.len(),
                    workspace.transfers.capacity(),
                    workspace
                        .transfers
                        .iter()
                        .enumerate()
                        .map(|(index, transfer)| (
                            index,
                            (
                                transfer.fine_to_coarse.len(),
                                transfer.fine_to_coarse.capacity(),
                                transfer.fine_to_coarse.clone(),
                            ),
                            (
                                transfer.fine_entry_to_coarse_entry.len(),
                                transfer.fine_entry_to_coarse_entry.capacity(),
                                transfer.fine_entry_to_coarse_entry.clone(),
                            ),
                        ))
                        .collect::<Vec<_>>(),
                ),
                before_transfers
            );
            assert_eq!(
                (
                    workspace.diagonal_slots.len(),
                    workspace.diagonal_slots.capacity(),
                    workspace
                        .diagonal_slots
                        .iter()
                        .enumerate()
                        .map(|(index, values)| (
                            index,
                            values.len(),
                            values.capacity(),
                            values.clone(),
                        ))
                        .collect::<Vec<_>>(),
                ),
                before_diagonal
            );
            assert_eq!(
                (
                    workspace.corrections.len(),
                    workspace.corrections.capacity(),
                    workspace
                        .corrections
                        .iter()
                        .enumerate()
                        .map(|(index, values)| (
                            index,
                            values.len(),
                            values.capacity(),
                            values
                                .iter()
                                .map(|value| value.to_bits())
                                .collect::<Vec<_>>(),
                        ))
                        .collect::<Vec<_>>(),
                ),
                before_corrections
            );
            assert_eq!(
                (
                    workspace.sources.len(),
                    workspace.sources.capacity(),
                    workspace
                        .sources
                        .iter()
                        .enumerate()
                        .map(|(index, values)| (
                            index,
                            values.len(),
                            values.capacity(),
                            values
                                .iter()
                                .map(|value| value.to_bits())
                                .collect::<Vec<_>>(),
                        ))
                        .collect::<Vec<_>>(),
                ),
                before_sources
            );
            assert_eq!(
                (
                    workspace.residuals.len(),
                    workspace.residuals.capacity(),
                    workspace
                        .residuals
                        .iter()
                        .enumerate()
                        .map(|(index, values)| (
                            index,
                            values.len(),
                            values.capacity(),
                            values
                                .iter()
                                .map(|value| value.to_bits())
                                .collect::<Vec<_>>(),
                        ))
                        .collect::<Vec<_>>(),
                ),
                before_residuals
            );
            assert_eq!(
                (
                    workspace.products.len(),
                    workspace.products.capacity(),
                    workspace
                        .products
                        .iter()
                        .enumerate()
                        .map(|(index, values)| (
                            index,
                            values.len(),
                            values.capacity(),
                            values
                                .iter()
                                .map(|value| value.to_bits())
                                .collect::<Vec<_>>(),
                        ))
                        .collect::<Vec<_>>(),
                ),
                before_products
            );
            assert_eq!(
                (
                    workspace.pre_smoothed.len(),
                    workspace.pre_smoothed.capacity(),
                    workspace
                        .pre_smoothed
                        .iter()
                        .enumerate()
                        .map(|(index, values)| (
                            index,
                            values.len(),
                            values.capacity(),
                            values
                                .iter()
                                .map(|value| value.to_bits())
                                .collect::<Vec<_>>(),
                        ))
                        .collect::<Vec<_>>(),
                ),
                before_pre_smoothed
            );
            assert_eq!(workspace.coarsest_pcg.is_none(), before_pcg_none);
            assert_eq!(workspace.has_solved, before_has_solved);
            assert_eq!(workspace.level_sizes(), before_levels);
            assert_singular_workspace!(&workspace);
            assert_eq!(
                (
                    initial.len(),
                    initial.capacity(),
                    initial
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>(),
                ),
                before_initial
            );
            assert!(std::sync::Arc::ptr_eq(
                &before_finest_row_arc,
                &workspace.finest_sparsity.row_offsets
            ));
            assert!(std::sync::Arc::ptr_eq(
                &before_finest_column_arc,
                &workspace.finest_sparsity.col_indices
            ));
            for (before, after) in before_matrix_row_arcs
                .iter()
                .zip(workspace.matrices.iter().map(|matrix| &matrix.row_offsets))
            {
                assert!(std::sync::Arc::ptr_eq(before, after));
            }
            for (before, after) in before_matrix_column_arcs
                .iter()
                .zip(workspace.matrices.iter().map(|matrix| &matrix.col_indices))
            {
                assert!(std::sync::Arc::ptr_eq(before, after));
            }
            assert_eq!(
                (
                    timing.hierarchy_builds,
                    timing.hierarchy_rebuilds,
                    timing.matrix_refreshes,
                    timing.finest_residual_evaluations,
                    timing.solves,
                    timing.v_cycles,
                ),
                (0, 0, 1, 1, 1, 0)
            );
            assert_eq!(
                timing
                    .levels
                    .iter()
                    .map(|level| (
                        level.level,
                        level.cells,
                        level.nonzeros,
                        level.matrix_refreshes,
                        level.restriction_calls,
                        level.prolongation_calls,
                        level.smoothing_calls,
                        level.smoothing_sweeps,
                        level.scaling_calls,
                        level.residual_evaluations,
                        level.correction_updates,
                        level.coarsest_solves,
                    ))
                    .collect::<Vec<_>>(),
                vec![
                    (0, 4, 10, 1, 1, 0, 0, 0, 0, 0, 0, 0),
                    (1, 2, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0),
                ]
            );
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
            for level in &timing.levels {
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
        }
    }
    #[test]
    fn profiled_normalized_l1_solve_is_bit_identical() {
        let matrix = CsrMatrix::from_rows(
            vec![
                vec![(0, 4.0), (1, -1.0), (2, -0.5)],
                vec![(0, -1.0), (1, 3.5), (3, -1.0)],
                vec![(0, -0.5), (2, 3.0), (3, -1.0), (4, -0.25)],
                vec![(1, -1.0), (2, -1.0), (3, 4.0), (5, -0.5)],
                vec![(2, -0.25), (4, 2.5), (5, -1.0), (6, -0.5)],
                vec![(3, -0.5), (4, -1.0), (5, 3.5), (7, -0.75)],
                vec![(4, -0.5), (6, 2.75), (7, -1.0)],
                vec![(5, -0.75), (6, -1.0), (7, 3.0)],
            ],
            8,
        )
        .expect("parity matrix");
        let reference = [0.5, 1.0, -0.25, 2.0, 1.5, -1.0, 0.75, 1.25];
        let mut rhs = vec![0.0; 8];
        for (row, source) in rhs.iter_mut().enumerate() {
            for entry in matrix.row_offsets()[row]..matrix.row_offsets()[row + 1] {
                *source += matrix.values()[entry] * reference[matrix.col_indices()[entry]];
            }
        }
        let options = GamgOptions {
            max_iterations: 2,
            n_cells_in_coarsest_level: 2,
            direct_solve_coarsest: true,
            ..GamgOptions::default()
        };
        let controls = NormalizedL1GamgSolveControls {
            normalization_factor: rhs.iter().map(|value| value.abs()).sum(),
            tolerance: 0.0,
            relative_tolerance: 0.0,
            l2_controls: super::GamgSolveControls {
                max_iterations: 2,
                min_iterations: 0,
                tolerance: 0.0,
                relative_tolerance: 0.0,
            },
        };
        let mut plain_workspace = GamgWorkspace::new(&matrix, options).expect("plain workspace");
        let plain = plain_workspace
            .solve_normalized_l1_with_controls(&matrix, &rhs, None, controls)
            .expect("plain parity solve");
        let mut profiled_workspace =
            GamgWorkspace::new(&matrix, options).expect("profiled workspace");
        let profiled = profiled_workspace
            .solve_normalized_l1_with_controls_profiled(&matrix, &rhs, None, controls)
            .expect("profiled parity solve");

        assert_eq!(plain.iterations, 2);
        assert_eq!(profiled.report.iterations, 2);
        assert!(!plain.converged);
        assert!(!profiled.report.converged);
        assert_eq!(
            plain.termination,
            super::IterativeSolveTermination::MaxIterations
        );
        assert_eq!(
            profiled.report.termination,
            super::IterativeSolveTermination::MaxIterations
        );
        assert_eq!(plain_workspace.level_sizes(), vec![8, 4, 2]);
        assert_eq!(profiled_workspace.level_sizes(), vec![8, 4, 2]);
        assert!(plain_workspace.has_solved);
        assert!(profiled_workspace.has_solved);
        assert_eq!(
            (
                (
                    plain_workspace.options.max_iterations,
                    plain_workspace.options.min_iterations,
                    plain_workspace.options.tolerance.to_bits(),
                    plain_workspace.options.relative_tolerance.to_bits(),
                ),
                (
                    plain_workspace.options.cache_agglomeration,
                    plain_workspace.options.n_cells_in_coarsest_level,
                    plain_workspace.options.merge_levels,
                    plain_workspace.options.agglomerator,
                ),
                (
                    plain_workspace.options.smoother,
                    plain_workspace.options.n_pre_sweeps,
                    plain_workspace.options.pre_sweeps_level_multiplier,
                    plain_workspace.options.max_pre_sweeps,
                    plain_workspace.options.n_post_sweeps,
                    plain_workspace.options.post_sweeps_level_multiplier,
                    plain_workspace.options.max_post_sweeps,
                ),
                (
                    plain_workspace.options.n_finest_sweeps,
                    plain_workspace.options.interpolate_correction,
                    plain_workspace.options.scale_correction,
                    plain_workspace.options.direct_solve_coarsest,
                ),
            ),
            (
                (2, 0, 1.0e-10f64.to_bits(), 0.0f64.to_bits()),
                (true, 2, 1, GamgAgglomerator::AlgebraicPair),
                (GamgSmoother::GaussSeidel, 0, 1, 4, 2, 1, 4),
                (2, false, true, true),
            )
        );
        assert_eq!(
            (
                (
                    profiled_workspace.options.max_iterations,
                    profiled_workspace.options.min_iterations,
                    profiled_workspace.options.tolerance.to_bits(),
                    profiled_workspace.options.relative_tolerance.to_bits(),
                ),
                (
                    profiled_workspace.options.cache_agglomeration,
                    profiled_workspace.options.n_cells_in_coarsest_level,
                    profiled_workspace.options.merge_levels,
                    profiled_workspace.options.agglomerator,
                ),
                (
                    profiled_workspace.options.smoother,
                    profiled_workspace.options.n_pre_sweeps,
                    profiled_workspace.options.pre_sweeps_level_multiplier,
                    profiled_workspace.options.max_pre_sweeps,
                    profiled_workspace.options.n_post_sweeps,
                    profiled_workspace.options.post_sweeps_level_multiplier,
                    profiled_workspace.options.max_post_sweeps,
                ),
                (
                    profiled_workspace.options.n_finest_sweeps,
                    profiled_workspace.options.interpolate_correction,
                    profiled_workspace.options.scale_correction,
                    profiled_workspace.options.direct_solve_coarsest,
                ),
            ),
            (
                (2, 0, 1.0e-10f64.to_bits(), 0.0f64.to_bits()),
                (true, 2, 1, GamgAgglomerator::AlgebraicPair),
                (GamgSmoother::GaussSeidel, 0, 1, 4, 2, 1, 4),
                (2, false, true, true),
            )
        );
        assert_eq!(
            plain.residual_norm.to_bits(),
            profiled.report.residual_norm.to_bits()
        );
        assert_eq!(plain.solution.len(), 8);
        assert_eq!(profiled.report.solution.len(), 8);
        assert_eq!(
            plain
                .solution
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            profiled
                .report
                .solution
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
        let mut squared_l2 = 0.0;
        for (row, source) in rhs.iter().enumerate() {
            let mut product = 0.0;
            for entry in matrix.row_offsets()[row]..matrix.row_offsets()[row + 1] {
                product += matrix.values()[entry] * plain.solution[matrix.col_indices()[entry]];
            }
            let residual = source - product;
            squared_l2 += residual * residual;
        }
        assert_eq!(plain.residual_norm.to_bits(), squared_l2.sqrt().to_bits());
        assert_eq!(
            (
                profiled.timing.hierarchy_builds,
                profiled.timing.hierarchy_rebuilds,
                profiled.timing.matrix_refreshes,
                profiled.timing.finest_residual_evaluations,
                profiled.timing.solves,
                profiled.timing.v_cycles,
            ),
            (0, 0, 1, 3, 1, 2)
        );
        assert_eq!(
            profiled
                .timing
                .levels
                .iter()
                .map(|level| (
                    level.level,
                    level.cells,
                    level.nonzeros,
                    level.matrix_refreshes,
                    level.restriction_calls,
                    level.prolongation_calls,
                    level.smoothing_calls,
                    level.smoothing_sweeps,
                    level.scaling_calls,
                    level.residual_evaluations,
                    level.correction_updates,
                    level.coarsest_solves,
                ))
                .collect::<Vec<_>>(),
            vec![
                (0, 8, 28, 1, 2, 2, 2, 4, 2, 0, 2, 0),
                (1, 4, 10, 1, 2, 2, 2, 4, 0, 0, 2, 0),
                (2, 2, 4, 1, 0, 0, 0, 0, 0, 0, 0, 2),
            ]
        );
        for seconds in [
            profiled.timing.total_seconds,
            profiled.timing.hierarchy_build_seconds,
            profiled.timing.hierarchy_rebuild_seconds,
            profiled.timing.matrix_refresh_seconds,
            profiled.timing.finest_residual_seconds,
            profiled.timing.v_cycle_seconds,
            profiled.timing.other_seconds,
        ] {
            assert!(seconds.is_finite() && seconds >= 0.0);
        }
        for level in &profiled.timing.levels {
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
    }

    #[test]
    fn algebraic_pair_agglomeration_is_deterministic_and_alternates_order() {
        let matrix = poisson_grid(8, 8, 1.0);
        let forward = algebraic_pair_map(&matrix, true).expect("forward map");
        let reverse = algebraic_pair_map(&matrix, false).expect("reverse map");
        let forward_again = algebraic_pair_map(&matrix, true).expect("repeat forward map");

        assert_eq!(forward, forward_again);
        assert_eq!(forward.1, reverse.1);
        assert_ne!(forward.0, reverse.0);
        assert!(forward.1 < matrix.rows());
    }

    #[test]
    fn algebraic_pair_adds_unmatched_cell_to_best_neighbour_cluster() {
        let matrix = CsrMatrix::from_rows(
            vec![
                vec![(0, 2.0), (1, -1.0)],
                vec![(0, -1.0), (1, 2.0), (2, -1.0)],
                vec![(1, -1.0), (2, 2.0)],
            ],
            3,
        )
        .expect("three-cell chain");

        let (coarse_map, n_coarse) =
            algebraic_pair_map(&matrix, true).expect("three-cell pair map");

        assert_eq!(coarse_map, vec![0, 0, 0]);
        assert_eq!(n_coarse, 1);
    }

    #[test]
    fn face_area_pair_uses_the_strongest_mesh_connection() {
        let edges = vec![
            PairEdge {
                lower: 0,
                upper: 1,
                weight: 1.0,
            },
            PairEdge {
                lower: 0,
                upper: 2,
                weight: 10.0,
            },
            PairEdge {
                lower: 1,
                upper: 3,
                weight: 2.0,
            },
        ];

        let (coarse_map, n_coarse) =
            pair_map_from_edges(4, &edges, true).expect("face-area pair map");

        assert_eq!(coarse_map[0], coarse_map[2]);
        assert_eq!(coarse_map[1], coarse_map[3]);
        assert_ne!(coarse_map[0], coarse_map[1]);
        assert_eq!(n_coarse, 2);
    }

    #[test]
    fn equal_weight_pair_prefers_narrower_external_stencil() {
        let edges = vec![
            PairEdge {
                lower: 0,
                upper: 2,
                weight: 10.0,
            },
            PairEdge {
                lower: 2,
                upper: 3,
                weight: 1.0,
            },
            PairEdge {
                lower: 2,
                upper: 4,
                weight: 1.0,
            },
            PairEdge {
                lower: 0,
                upper: 1,
                weight: 10.0,
            },
        ];

        let (coarse_map, n_coarse) =
            pair_map_from_edges(5, &edges, true).expect("equal-weight pair map");

        assert_eq!(coarse_map[0], coarse_map[1]);
        assert_ne!(coarse_map[0], coarse_map[2]);
        assert_eq!(coarse_map[2], coarse_map[3]);
        assert_eq!(coarse_map[2], coarse_map[4]);
        assert_eq!(n_coarse, 2);
    }

    #[test]
    fn external_neighbour_count_matches_union_oracle_for_all_small_graphs() {
        for n_cells in 2usize..=5 {
            let possible_edges = (0..n_cells)
                .flat_map(|lower| ((lower + 1)..n_cells).map(move |upper| (lower, upper)))
                .collect::<Vec<_>>();
            for mask in 1usize..(1usize << possible_edges.len()) {
                let mut neighbours = vec![BTreeSet::<usize>::new(); n_cells];
                let mut edges = Vec::new();
                for (bit, &(lower, upper)) in possible_edges.iter().enumerate() {
                    if mask & (1usize << bit) != 0 {
                        neighbours[lower].insert(upper);
                        neighbours[upper].insert(lower);
                        edges.push((lower, upper));
                    }
                }
                for (lower, upper) in edges {
                    let expected = neighbours[lower]
                        .union(&neighbours[upper])
                        .filter(|&&cell| cell != lower && cell != upper)
                        .count();
                    let actual = external_neighbour_count(
                        lower,
                        upper,
                        &neighbours[lower],
                        &neighbours[upper],
                    )
                    .0;
                    assert_eq!(actual, expected, "graph mask {mask}, edge {lower}-{upper}");
                }
            }
        }
    }

    #[test]
    fn shared_neighbour_triangle_prefers_the_narrower_stencil() {
        let edges = [
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
            (1, 2),
            (1, 3),
            (1, 4),
            (2, 5),
        ]
        .into_iter()
        .map(|(lower, upper)| PairEdge {
            lower,
            upper,
            weight: 1.0,
        })
        .collect::<Vec<_>>();

        let (coarse_map, _) =
            pair_map_from_edges(6, &edges, true).expect("shared-neighbour pair map");
        assert_eq!(coarse_map[0], coarse_map[1]);
        assert_ne!(coarse_map[0], coarse_map[2]);
    }

    #[test]
    fn parallel_and_reversed_edges_preserve_the_pair_map() {
        let unique = vec![
            PairEdge {
                lower: 0,
                upper: 1,
                weight: 4.0,
            },
            PairEdge {
                lower: 0,
                upper: 2,
                weight: 4.0,
            },
            PairEdge {
                lower: 1,
                upper: 2,
                weight: 1.0,
            },
            PairEdge {
                lower: 2,
                upper: 3,
                weight: 2.0,
            },
        ];
        let mut repeated = unique.clone();
        repeated.extend([
            PairEdge {
                lower: 1,
                upper: 0,
                weight: 4.0,
            },
            PairEdge {
                lower: 0,
                upper: 1,
                weight: 3.0,
            },
        ]);

        for forward in [true, false] {
            assert_eq!(
                pair_map_from_edges(4, &repeated, forward).expect("repeated pair map"),
                pair_map_from_edges(4, &unique, forward).expect("unique pair map")
            );
        }
    }

    #[test]
    fn external_neighbour_count_adapts_between_star_and_dense_graphs() {
        let n_star = 4097usize;
        let hub = (1..n_star).collect::<BTreeSet<_>>();
        let leaf = BTreeSet::from([0usize]);
        let (star_count, star_method) = external_neighbour_count(0, 1, &hub, &leaf);
        assert_eq!(star_count, n_star - 2);
        assert_eq!(
            star_method,
            ExternalNeighbourCountMethod::MembershipIntersection
        );

        let n_dense = 64usize;
        let first = (0..n_dense).filter(|&cell| cell != 0).collect();
        let second = (0..n_dense).filter(|&cell| cell != 1).collect();
        let (dense_count, dense_method) = external_neighbour_count(0, 1, &first, &second);
        assert_eq!(dense_count, n_dense - 2);
        assert_eq!(dense_method, ExternalNeighbourCountMethod::UnionScan);
    }

    #[test]
    fn pair_map_is_invariant_to_equal_weight_edge_permutation() {
        let edges = vec![
            PairEdge {
                lower: 0,
                upper: 2,
                weight: 10.0,
            },
            PairEdge {
                lower: 2,
                upper: 3,
                weight: 1.0,
            },
            PairEdge {
                lower: 2,
                upper: 4,
                weight: 1.0,
            },
            PairEdge {
                lower: 0,
                upper: 1,
                weight: 10.0,
            },
        ];
        let mut reversed = edges.clone();
        reversed.reverse();

        for forward in [true, false] {
            let expected = pair_map_from_edges(5, &edges, forward).expect("ordered pair map");
            let actual = pair_map_from_edges(5, &reversed, forward).expect("reversed pair map");

            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn stronger_pair_weight_wins_over_stencil_tiebreak() {
        let edges = vec![
            PairEdge {
                lower: 0,
                upper: 1,
                weight: 10.0,
            },
            PairEdge {
                lower: 0,
                upper: 2,
                weight: 11.0,
            },
            PairEdge {
                lower: 1,
                upper: 5,
                weight: 9.0,
            },
            PairEdge {
                lower: 2,
                upper: 3,
                weight: 1.0,
            },
            PairEdge {
                lower: 2,
                upper: 4,
                weight: 1.0,
            },
        ];

        let (coarse_map, _) =
            pair_map_from_edges(6, &edges, true).expect("stronger-weight pair map");

        assert_eq!(coarse_map[0], coarse_map[2]);
        assert_ne!(coarse_map[0], coarse_map[1]);
        assert_eq!(coarse_map[1], coarse_map[5]);
    }

    #[test]
    fn gamg_converges_to_pcg_on_a_general_poisson_csr_matrix() {
        let matrix = poisson_grid(24, 20, 1.0);
        let expected = (0..matrix.rows())
            .map(|row| 0.25 + (row % 17) as f64 / 17.0)
            .collect::<Vec<_>>();
        let rhs = matrix.matvec(&expected).expect("Poisson rhs");
        let options = GamgOptions {
            max_iterations: 80,
            tolerance: 1.0e-10,
            direct_solve_coarsest: true,
            ..GamgOptions::default()
        };

        let gamg = gamg_solve(&matrix, &rhs, None, options).expect("GAMG solve");
        let pcg = preconditioned_conjugate_gradient_solve(
            &matrix,
            &rhs,
            None,
            PreconditionedConjugateGradientOptions {
                max_iterations: 1_000,
                tolerance: 1.0e-12,
                preconditioner: CgPreconditioner::IncompleteCholesky,
            },
        )
        .expect("PCG parity solve");

        assert!(gamg.converged, "GAMG residual={}", gamg.residual_norm);
        assert!(pcg.converged, "PCG residual={}", pcg.residual_norm);
        assert_close(&gamg.solution, &expected, 1.0e-8);
        assert_close(&gamg.solution, &pcg.solution, 1.0e-8);
    }

    #[test]
    fn gamg_default_iterative_coarsest_solver_converges() {
        let matrix = poisson_grid(18, 14, 1.0);
        let expected = (0..matrix.rows())
            .map(|row| 0.5 + (row % 11) as f64 / 11.0)
            .collect::<Vec<_>>();
        let rhs = matrix.matvec(&expected).expect("Poisson rhs");
        let options = GamgOptions {
            max_iterations: 80,
            tolerance: 1.0e-10,
            ..GamgOptions::default()
        };

        let report = gamg_solve(&matrix, &rhs, None, options)
            .expect("GAMG solve with iterative coarsest solver");

        assert!(report.converged, "GAMG residual={}", report.residual_norm);
        assert_close(&report.solution, &expected, 1.0e-8);
    }

    #[test]
    fn legitimate_dense_coarsest_solve_preserves_the_expected_solution() {
        let matrix =
            CsrMatrix::from_rows(vec![vec![(0, 4.0), (1, 1.0)], vec![(0, 2.0), (1, 3.0)]], 2)
                .expect("small direct matrix");
        let mut solution = vec![0.0; 2];

        dense_lu_solve(&matrix, &[6.0, 8.0], &mut solution).expect("bounded dense solve");

        assert_close(&solution, &[1.0, 2.0], 1.0e-12);
    }

    #[test]
    fn dense_coarsest_rejects_actual_sparse_matrix_above_the_limit() {
        let n = MAX_DENSE_COARSEST_CELLS + 1;
        let matrix = diagonal_matrix(n);
        let mut solution = vec![0.0; n];

        let error = dense_lu_solve(&matrix, &vec![1.0; n], &mut solution)
            .expect_err("oversized actual coarsest matrix must fail")
            .to_string();

        assert!(error.contains("actual coarsest cells"));
    }

    #[test]
    fn dense_storage_size_overflow_is_reported_without_allocation() {
        let error = checked_dense_storage_len(usize::MAX)
            .expect_err("overflow-sized dense matrix must fail")
            .to_string();

        assert!(error.contains("overflow"));
    }

    #[test]
    fn configured_threshold_cannot_bypass_actual_coarsest_limit() {
        let threshold = MAX_DENSE_COARSEST_CELLS + 1;
        let matrix = tridiagonal_matrix(4 * threshold);
        let options = GamgOptions {
            n_cells_in_coarsest_level: threshold,
            direct_solve_coarsest: true,
            ..GamgOptions::default()
        };
        let mut workspace = GamgWorkspace::new(&matrix, options).expect("bounded hierarchy");
        let rhs = vec![1.0; matrix.rows()];

        let error = workspace
            .solve(&matrix, &rhs, None)
            .expect_err("actual direct coarsest matrix must be bounded")
            .to_string();

        assert!(error.contains("actual coarsest cells"));
        assert_eq!(workspace.level_sizes().last(), Some(&threshold));
    }

    #[test]
    fn profiled_gamg_preserves_solution_order_and_reports_each_level() {
        let matrix = poisson_grid(18, 14, 1.0);
        let expected = (0..matrix.rows())
            .map(|row| 0.5 + (row % 11) as f64 / 11.0)
            .collect::<Vec<_>>();
        let rhs = matrix.matvec(&expected).expect("Poisson rhs");
        let options = GamgOptions {
            max_iterations: 80,
            tolerance: 1.0e-10,
            direct_solve_coarsest: true,
            ..GamgOptions::default()
        };
        let mut regular_workspace = GamgWorkspace::new(&matrix, options).expect("regular GAMG");
        let mut profiled_workspace = GamgWorkspace::new(&matrix, options).expect("profiled GAMG");

        let regular = regular_workspace
            .solve_with_controls(&matrix, &rhs, None, options.into())
            .expect("regular solve");
        let profiled = profiled_workspace
            .solve_with_controls_profiled(&matrix, &rhs, None, options.into())
            .expect("profiled solve");

        assert_eq!(profiled.report.solution, regular.solution);
        assert_eq!(profiled.report.iterations, regular.iterations);
        assert_eq!(
            profiled.report.residual_norm.to_bits(),
            regular.residual_norm.to_bits()
        );
        assert_eq!(profiled.report.converged, regular.converged);
        assert_eq!(profiled.timing.solves, 1);
        assert_eq!(profiled.timing.v_cycles, profiled.report.iterations);
        assert_eq!(profiled.timing.matrix_refreshes, 1);
        assert_eq!(
            profiled.timing.finest_residual_evaluations,
            profiled.report.iterations + 1
        );
        assert_eq!(
            profiled.timing.levels.len(),
            profiled_workspace.level_count()
        );
        assert_eq!(profiled.timing.levels[0].cells, matrix.rows());
        assert_eq!(
            profiled
                .timing
                .levels
                .last()
                .expect("coarsest level")
                .coarsest_solves,
            profiled.report.iterations
        );
        assert!(profiled.timing.total_seconds >= profiled.timing.v_cycle_seconds);

        let mut accumulated = profiled.timing.clone();
        accumulated
            .accumulate(&profiled.timing)
            .expect("matching hierarchy profiles accumulate");
        assert_eq!(accumulated.solves, 2);
        assert_eq!(accumulated.v_cycles, 2 * profiled.report.iterations);
    }

    #[test]
    fn face_area_pair_workspace_solves_and_rebuilds_without_cache() {
        let matrix = poisson_grid(18, 14, 1.0);
        let expected = (0..matrix.rows())
            .map(|row| 0.75 + (row % 13) as f64 / 13.0)
            .collect::<Vec<_>>();
        let rhs = matrix.matvec(&expected).expect("Poisson rhs");
        let options = GamgOptions {
            max_iterations: 80,
            tolerance: 1.0e-10,
            agglomerator: GamgAgglomerator::FaceAreaPair,
            cache_agglomeration: false,
            direct_solve_coarsest: true,
            ..GamgOptions::default()
        };
        let face_weights = grid_face_weights(18, 14);
        let mut workspace =
            GamgWorkspace::new_with_face_area_weights(&matrix, options, &face_weights)
                .expect("faceAreaPair GAMG workspace");

        let first = workspace
            .solve(&matrix, &rhs, None)
            .expect("first faceAreaPair solve");
        let second = workspace
            .solve(&matrix, &rhs, Some(&first.solution))
            .expect("rebuilt faceAreaPair solve");

        assert!(first.converged, "first residual={}", first.residual_norm);
        assert!(second.converged, "second residual={}", second.residual_norm);
        assert_close(&first.solution, &expected, 1.0e-8);
        assert_close(&second.solution, &expected, 1.0e-8);
    }

    #[test]
    fn cached_workspace_reuses_hierarchy_for_updated_coefficients() {
        let first = poisson_grid(20, 16, 1.0);
        let options = GamgOptions {
            max_iterations: 3,
            min_iterations: 3,
            tolerance: 0.0,
            relative_tolerance: 0.0,
            n_cells_in_coarsest_level: 4,
            direct_solve_coarsest: true,
            ..GamgOptions::default()
        };
        let mut workspace = GamgWorkspace::new(&first, options).expect("GAMG workspace");
        let mut scanned_reference =
            GamgWorkspace::new(&first, options).expect("scanned-reference GAMG workspace");
        assert!(
            workspace.level_count() >= 3,
            "lifecycle proof requires multiple coarse levels"
        );
        let level_sizes = workspace.level_sizes();
        let correction_pointers = workspace
            .corrections
            .iter()
            .map(|level| level.as_ptr())
            .collect::<Vec<_>>();
        let diagonal_allocations = workspace
            .diagonal_values
            .iter()
            .map(|level| (level.as_ptr(), level.len(), level.capacity()))
            .collect::<Vec<_>>();

        for refresh in 0..10 {
            let matrix = poisson_grid(20, 16, 1.0 + refresh as f64 * 0.125);
            let expected = (0..matrix.rows())
                .map(|row| 0.5 + ((row * 17 + refresh * 11) % 29) as f64 / 23.0)
                .collect::<Vec<_>>();
            let rhs = matrix.matvec(&expected).expect("updated rhs");
            let mut timing = GamgKernelTiming::default();
            let report = workspace
                .solve_with_controls_internal::<true, false>(
                    &matrix,
                    &rhs,
                    None,
                    options.into(),
                    &mut timing,
                )
                .expect("updated GAMG solve");
            let mut reference_timing = GamgKernelTiming::default();
            let reference_report = scanned_reference
                .solve_with_controls_internal::<true, true>(
                    &matrix,
                    &rhs,
                    None,
                    options.into(),
                    &mut reference_timing,
                )
                .expect("scanned-reference GAMG solve");

            assert_eq!(workspace.level_count(), level_sizes.len());
            for (level, ((level_matrix, slots), cached)) in workspace
                .matrices
                .iter()
                .zip(&workspace.diagonal_slots)
                .zip(&workspace.diagonal_values)
                .enumerate()
            {
                assert_eq!(cached.len(), level_matrix.rows());
                for row in 0..level_matrix.rows() {
                    let scanned = (level_matrix.row_offsets()[row]
                        ..level_matrix.row_offsets()[row + 1])
                        .filter(|&entry| level_matrix.col_indices()[entry] == row)
                        .map(|entry| level_matrix.values()[entry])
                        .collect::<Vec<_>>();
                    assert_eq!(
                        scanned.len(),
                        1,
                        "level {level} row {row} must have one scanned diagonal"
                    );
                    assert_eq!(slots[row], {
                        (level_matrix.row_offsets()[row]..level_matrix.row_offsets()[row + 1])
                            .find(|&entry| level_matrix.col_indices()[entry] == row)
                            .expect("scanned diagonal slot")
                    });
                    assert_eq!(
                        cached[row].to_bits(),
                        scanned[0].to_bits(),
                        "refresh {refresh}, level {level}, row {row}"
                    );
                }
            }
            assert_eq!(
                report
                    .solution
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                reference_report
                    .solution
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>()
            );
            assert_eq!(report.iterations, reference_report.iterations);
            assert_eq!(
                report.residual_norm.to_bits(),
                reference_report.residual_norm.to_bits()
            );
            assert_eq!(report.converged, reference_report.converged);
            assert_eq!(report.termination, reference_report.termination);
            assert_eq!(timing.hierarchy_builds, reference_timing.hierarchy_builds);
            assert_eq!(timing.hierarchy_builds, 0);
            assert_eq!(
                timing.hierarchy_rebuilds,
                reference_timing.hierarchy_rebuilds
            );
            assert_eq!(timing.matrix_refreshes, reference_timing.matrix_refreshes);
            assert_eq!(
                timing.finest_residual_evaluations,
                reference_timing.finest_residual_evaluations
            );
            assert_eq!(timing.solves, reference_timing.solves);
            assert_eq!(timing.v_cycles, reference_timing.v_cycles);
            assert_eq!(timing.levels.len(), reference_timing.levels.len());
            for (cached_level, scanned_level) in timing.levels.iter().zip(&reference_timing.levels)
            {
                assert_eq!(cached_level.level, scanned_level.level);
                assert_eq!(cached_level.cells, scanned_level.cells);
                assert_eq!(cached_level.nonzeros, scanned_level.nonzeros);
                assert_eq!(
                    cached_level.matrix_refreshes,
                    scanned_level.matrix_refreshes
                );
                assert_eq!(
                    cached_level.restriction_calls,
                    scanned_level.restriction_calls
                );
                assert_eq!(
                    cached_level.prolongation_calls,
                    scanned_level.prolongation_calls
                );
                assert_eq!(cached_level.smoothing_calls, scanned_level.smoothing_calls);
                assert_eq!(
                    cached_level.smoothing_sweeps,
                    scanned_level.smoothing_sweeps
                );
                assert_eq!(cached_level.scaling_calls, scanned_level.scaling_calls);
                assert_eq!(
                    cached_level.residual_evaluations,
                    scanned_level.residual_evaluations
                );
                assert_eq!(
                    cached_level.correction_updates,
                    scanned_level.correction_updates
                );
                assert_eq!(cached_level.coarsest_solves, scanned_level.coarsest_solves);
            }
            assert_eq!(
                workspace
                    .diagonal_values
                    .iter()
                    .map(|level| (level.as_ptr(), level.len(), level.capacity()))
                    .collect::<Vec<_>>(),
                diagonal_allocations
            );
        }
        assert_eq!(workspace.level_sizes(), level_sizes);
        assert_eq!(
            workspace
                .corrections
                .iter()
                .map(|level| level.as_ptr())
                .collect::<Vec<_>>(),
            correction_pointers
        );
    }

    #[test]
    fn unsupported_gamg_controls_fail_without_substitution() {
        let matrix = poisson_grid(12, 12, 1.0);
        let face_area_error = GamgWorkspace::new(
            &matrix,
            GamgOptions {
                agglomerator: GamgAgglomerator::FaceAreaPair,
                ..GamgOptions::default()
            },
        )
        .err()
        .expect("faceAreaPair must require geometry")
        .to_string();
        let merge_error = GamgWorkspace::new(
            &matrix,
            GamgOptions {
                merge_levels: 2,
                ..GamgOptions::default()
            },
        )
        .err()
        .expect("mergeLevels=2 must not be ignored")
        .to_string();

        assert!(face_area_error.contains("faceAreaPair"));
        assert!(merge_error.contains("no level-combination fallback"));
    }

    #[test]
    fn gamg_rejects_duplicate_diagonal_entries_required_by_cached_smoothing() {
        let matrix = CsrMatrix::from_rows(
            vec![
                vec![(0, 1.0), (0, 1.0), (1, -1.0)],
                vec![(0, -1.0), (1, 2.0)],
            ],
            2,
        )
        .expect("CSR matrix with duplicate diagonal entry");

        let error = GamgWorkspace::new(
            &matrix,
            GamgOptions {
                n_cells_in_coarsest_level: 1,
                ..GamgOptions::default()
            },
        )
        .err()
        .expect("duplicate GAMG diagonal must fail")
        .to_string();

        assert!(error.contains("exactly one diagonal entry"));
    }

    mod ldu_b1_spec {
        use std::collections::BTreeMap;

        use super::super::gauss_seidel_sweep_with_cached_diagonal;
        use crate::linear::CsrMatrix;

        #[derive(Debug)]
        struct SpecLduLevel {
            lower_addr: Vec<usize>,
            upper_addr: Vec<usize>,
            lower_csr: Vec<Option<usize>>,
            upper_csr: Vec<Option<usize>>,
            lower: Vec<f64>,
            upper: Vec<f64>,
            owner_start: Vec<usize>,
            b_prime: Vec<f64>,
        }

        #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
        struct Counters {
            rhs_copies: usize,
            rows: usize,
        }

        type Allocation = (*const (), usize, usize);

        #[derive(Debug, PartialEq)]
        struct LevelState {
            lower_addr: Vec<usize>,
            upper_addr: Vec<usize>,
            lower_csr: Vec<Option<usize>>,
            upper_csr: Vec<Option<usize>>,
            lower: Vec<f64>,
            upper: Vec<f64>,
            owner_start: Vec<usize>,
            b_prime: Vec<f64>,
            allocations: [Allocation; 8],
        }

        impl SpecLduLevel {
            fn new(matrix: &CsrMatrix) -> Self {
                let mut pairs = BTreeMap::<(usize, usize), (Vec<usize>, Vec<usize>)>::new();
                for row in 0..matrix.rows() {
                    for slot in matrix.row_offsets()[row]..matrix.row_offsets()[row + 1] {
                        let column = matrix.col_indices()[slot];
                        if row == column {
                            continue;
                        }
                        let (owner, neighbour) = if row < column {
                            (row, column)
                        } else {
                            (column, row)
                        };
                        let occurrences = pairs.entry((owner, neighbour)).or_default();
                        if row == owner {
                            occurrences.1.push(slot);
                        } else {
                            occurrences.0.push(slot);
                        }
                    }
                }
                let face_count = pairs
                    .values()
                    .map(|(lower, upper)| lower.len().max(upper.len()))
                    .sum();
                let mut level = Self {
                    lower_addr: Vec::with_capacity(face_count),
                    upper_addr: Vec::with_capacity(face_count),
                    lower_csr: Vec::with_capacity(face_count),
                    upper_csr: Vec::with_capacity(face_count),
                    lower: vec![0.0; face_count],
                    upper: vec![0.0; face_count],
                    owner_start: vec![0; matrix.rows() + 1],
                    b_prime: vec![0.0; matrix.rows()],
                };
                for ((owner, neighbour), (lower, upper)) in pairs {
                    for occurrence in 0..lower.len().max(upper.len()) {
                        level.lower_addr.push(owner);
                        level.upper_addr.push(neighbour);
                        level.lower_csr.push(lower.get(occurrence).copied());
                        level.upper_csr.push(upper.get(occurrence).copied());
                    }
                }
                for &owner in &level.lower_addr {
                    level.owner_start[owner + 1] += 1;
                }
                for cell in 0..matrix.rows() {
                    level.owner_start[cell + 1] += level.owner_start[cell];
                }
                level.refresh(matrix);
                level
            }

            fn refresh(&mut self, matrix: &CsrMatrix) {
                for face in 0..self.lower.len() {
                    self.lower[face] =
                        self.lower_csr[face].map_or(0.0, |slot| matrix.values()[slot]);
                    self.upper[face] =
                        self.upper_csr[face].map_or(0.0, |slot| matrix.values()[slot]);
                }
            }

            fn allocations(&self) -> [Allocation; 8] {
                [
                    vec_allocation(&self.lower_addr),
                    vec_allocation(&self.upper_addr),
                    vec_allocation(&self.lower_csr),
                    vec_allocation(&self.upper_csr),
                    vec_allocation(&self.lower),
                    vec_allocation(&self.upper),
                    vec_allocation(&self.owner_start),
                    vec_allocation(&self.b_prime),
                ]
            }

            fn state(&self) -> LevelState {
                LevelState {
                    lower_addr: self.lower_addr.clone(),
                    upper_addr: self.upper_addr.clone(),
                    lower_csr: self.lower_csr.clone(),
                    upper_csr: self.upper_csr.clone(),
                    lower: self.lower.clone(),
                    upper: self.upper.clone(),
                    owner_start: self.owner_start.clone(),
                    b_prime: self.b_prime.clone(),
                    allocations: self.allocations(),
                }
            }
        }

        fn vec_allocation<T>(values: &Vec<T>) -> Allocation {
            (values.as_ptr().cast(), values.len(), values.capacity())
        }

        fn diagonal(matrix: &CsrMatrix) -> Result<Vec<f64>, String> {
            let mut result = Vec::with_capacity(matrix.rows());
            for row in 0..matrix.rows() {
                let values = (matrix.row_offsets()[row]..matrix.row_offsets()[row + 1])
                    .filter(|&slot| matrix.col_indices()[slot] == row)
                    .map(|slot| matrix.values()[slot])
                    .collect::<Vec<_>>();
                if values.len() != 1 {
                    return Err(format!(
                        "LDU B1 diagonal row {row} must contain exactly one entry, got {}",
                        values.len()
                    ));
                }
                if !values[0].is_finite() || values[0] == 0.0 {
                    return Err(format!(
                        "LDU B1 diagonal row {row} must be finite and non-zero"
                    ));
                }
                result.push(values[0]);
            }
            Ok(result)
        }

        fn sweep(
            level: &mut SpecLduLevel,
            diagonal: &[f64],
            rhs: &[f64],
            psi: &mut [f64],
            counters: &mut Counters,
        ) -> Result<(), String> {
            if diagonal.len() != psi.len() {
                return Err(format!(
                    "LDU B1 diagonal length mismatch: expected {}, got {}",
                    psi.len(),
                    diagonal.len()
                ));
            }
            if rhs.len() != psi.len() {
                return Err(format!(
                    "LDU B1 rhs length mismatch: expected {}, got {}",
                    psi.len(),
                    rhs.len()
                ));
            }
            if level.b_prime.len() != psi.len() {
                return Err(format!(
                    "LDU B1 bPrime length mismatch: expected {}, got {}",
                    psi.len(),
                    level.b_prime.len()
                ));
            }
            for (row, value) in diagonal.iter().enumerate() {
                if !value.is_finite() || *value == 0.0 {
                    return Err(format!(
                        "LDU B1 diagonal row {row} must be finite and non-zero"
                    ));
                }
            }
            level.b_prime.copy_from_slice(rhs);
            counters.rhs_copies += 1;
            half_sweep(level, diagonal, psi, 0..psi.len(), counters)?;
            half_sweep(level, diagonal, psi, (0..psi.len()).rev(), counters)
        }

        fn half_sweep<I: Iterator<Item = usize>>(
            level: &mut SpecLduLevel,
            diagonal: &[f64],
            psi: &mut [f64],
            cells: I,
            counters: &mut Counters,
        ) -> Result<(), String> {
            for cell in cells {
                let mut psii = level.b_prime[cell];
                for face in level.owner_start[cell]..level.owner_start[cell + 1] {
                    psii -= level.upper[face] * psi[level.upper_addr[face]];
                }
                psii /= diagonal[cell];
                if !psii.is_finite() {
                    return Err(format!("LDU B1 update row {cell} is not finite"));
                }
                for face in level.owner_start[cell]..level.owner_start[cell + 1] {
                    level.b_prime[level.upper_addr[face]] -= level.lower[face] * psii;
                }
                psi[cell] = psii;
                counters.rows += 1;
            }
            Ok(())
        }

        fn matrix(rows: Vec<Vec<(usize, f64)>>) -> CsrMatrix {
            let count = rows.len();
            CsrMatrix::from_rows(rows, count).expect("B1 fixture")
        }

        fn main_fixture() -> CsrMatrix {
            matrix(vec![
                vec![
                    (0, 4.0),
                    (1, -1.0000000000000002),
                    (1, -0.25),
                    (2, -0.12500000000000003),
                ],
                vec![
                    (0, -0.9999999999999999),
                    (0, -0.125),
                    (1, 3.0),
                    (2, -0.5000000000000001),
                ],
                vec![
                    (0, -0.12499999999999999),
                    (1, -0.49999999999999994),
                    (2, 2.5),
                ],
            ])
        }

        #[test]
        fn ldu_b1_topology_is_deterministic_for_noncanonical_directed_duplicates() {
            let canonical = main_fixture();
            let reordered = matrix(vec![
                vec![
                    (2, -0.12500000000000003),
                    (1, -0.25),
                    (0, 4.0),
                    (1, -1.0000000000000002),
                ],
                vec![
                    (2, -0.5000000000000001),
                    (1, 3.0),
                    (0, -0.125),
                    (0, -0.9999999999999999),
                ],
                vec![
                    (2, 2.5),
                    (1, -0.49999999999999994),
                    (0, -0.12499999999999999),
                ],
            ]);
            let left = SpecLduLevel::new(&canonical);
            let right = SpecLduLevel::new(&reordered);
            assert_eq!(left.lower_addr, right.lower_addr);
            assert_eq!(left.upper_addr, right.upper_addr);
            assert_eq!(left.owner_start, right.owner_start);
            assert_ne!(left.upper_csr, right.upper_csr);
            assert_eq!(right.upper[0].to_bits(), (-0.25f64).to_bits());
        }

        #[test]
        fn ldu_b1_preserves_missing_reciprocal_terms_without_merging() {
            let mut csr = matrix(vec![
                vec![(0, 2.0), (1, -1.0), (1, -2.0)],
                vec![(0, -3.0), (1, 2.0)],
            ]);
            let mut level = SpecLduLevel::new(&csr);
            assert_eq!(level.lower.len(), 2);
            assert_eq!(
                level.lower_csr.iter().filter(|slot| slot.is_none()).count(),
                1
            );
            assert_eq!(level.lower[1].to_bits(), 0.0f64.to_bits());
            assert_eq!(level.upper, [-1.0, -2.0]);
            for lifecycle in 0..10 {
                for (slot, value) in csr.values_mut().iter_mut().enumerate() {
                    *value = (lifecycle * 10 + slot + 1) as f64;
                }
                level.refresh(&csr);
                assert_eq!(level.lower[1].to_bits(), 0.0f64.to_bits());
            }
        }

        #[test]
        fn ldu_b1_refresh_is_in_place_across_three_levels_and_ten_lifecycles() {
            for size in [2, 3, 7] {
                let rows = (0..size)
                    .map(|row| {
                        let mut values = vec![(row, 4.0)];
                        if row + 1 < size {
                            values.push((row + 1, -1.0));
                        }
                        if row > 0 {
                            values.push((row - 1, -1.0));
                        }
                        values
                    })
                    .collect();
                let mut csr = matrix(rows);
                let mut level = SpecLduLevel::new(&csr);
                let allocations = level.allocations();
                let topology = (
                    level.lower_addr.clone(),
                    level.upper_addr.clone(),
                    level.lower_csr.clone(),
                    level.upper_csr.clone(),
                    level.owner_start.clone(),
                );
                for lifecycle in 0..10 {
                    for (slot, value) in csr.values_mut().iter_mut().enumerate() {
                        *value = (lifecycle * 100 + slot + 1) as f64;
                    }
                    let before_refresh = level.allocations();
                    level.refresh(&csr);
                    assert_eq!(level.allocations(), before_refresh);
                    assert_eq!(level.allocations(), allocations);
                    assert_eq!(
                        (
                            level.lower_addr.clone(),
                            level.upper_addr.clone(),
                            level.lower_csr.clone(),
                            level.upper_csr.clone(),
                            level.owner_start.clone(),
                        ),
                        topology
                    );
                    for face in 0..level.lower.len() {
                        assert_eq!(
                            level.lower[face].to_bits(),
                            level.lower_csr[face]
                                .map_or(0.0, |slot| csr.values()[slot])
                                .to_bits()
                        );
                        assert_eq!(
                            level.upper[face].to_bits(),
                            level.upper_csr[face]
                                .map_or(0.0, |slot| csr.values()[slot])
                                .to_bits()
                        );
                    }
                    let diagonal = diagonal(&csr).unwrap();
                    let rhs = vec![1.0; size];
                    let mut psi = vec![0.25; size];
                    let mut counters = Counters::default();
                    let before_sweep = level.allocations();
                    sweep(&mut level, &diagonal, &rhs, &mut psi, &mut counters).unwrap();
                    assert_eq!(level.allocations(), before_sweep);
                    assert_eq!(level.allocations(), allocations);
                    assert_eq!(
                        (
                            level.lower_addr.clone(),
                            level.upper_addr.clone(),
                            level.lower_csr.clone(),
                            level.upper_csr.clone(),
                            level.owner_start.clone(),
                        ),
                        topology
                    );
                    for face in 0..level.lower.len() {
                        assert_eq!(
                            level.lower[face].to_bits(),
                            level.lower_csr[face]
                                .map_or(0.0, |slot| csr.values()[slot])
                                .to_bits()
                        );
                        assert_eq!(
                            level.upper[face].to_bits(),
                            level.upper_csr[face]
                                .map_or(0.0, |slot| csr.values()[slot])
                                .to_bits()
                        );
                    }
                    assert_eq!(
                        counters,
                        Counters {
                            rhs_copies: 1,
                            rows: 2 * size
                        }
                    );
                }
            }
        }

        #[test]
        fn ldu_b1_symgs_matches_openfoam_v13_forward_backward_bit_oracle() {
            let csr = main_fixture();
            let mut level = SpecLduLevel::new(&csr);
            let diagonal = diagonal(&csr).unwrap();
            let mut psi = vec![0.25, -0.5, 0.75];
            level.b_prime.copy_from_slice(&[1.0, -2.0, 0.5]);
            let mut counters = Counters::default();
            half_sweep(&mut level, &diagonal, &mut psi, 0..3, &mut counters).unwrap();
            assert_eq!(
                psi.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                [
                    4593108669964484606,
                    13826009807593319083,
                    4592325231279306616
                ]
            );
            assert_eq!(
                level
                    .b_prime
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                [
                    4607182418800017408,
                    13834464319003164672,
                    4598459626552994475
                ]
            );
            half_sweep(&mut level, &diagonal, &mut psi, (0..3).rev(), &mut counters).unwrap();
            assert_eq!(
                psi.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                [
                    4589294781764422130,
                    13826996631496043907,
                    4592325231279306616
                ]
            );
            assert_eq!(
                level
                    .b_prime
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                [
                    4607182418800017408,
                    13834138746738232525,
                    13807295973087021176
                ]
            );
        }

        #[test]
        fn ldu_b1_symgs_distinguishes_face_order_and_bprime_reset_sentinels() {
            let csr = main_fixture();
            let diagonal = diagonal(&csr).unwrap();
            let mut level = SpecLduLevel::new(&csr);
            let mut psi = vec![0.25, -0.5, 0.75];
            let mut counters = Counters::default();
            half_sweep_with_rhs(
                &mut level,
                &diagonal,
                &[1.0, -2.0, 0.5],
                &mut psi,
                0..3,
                &mut counters,
            );
            level.b_prime.copy_from_slice(&[1.0, -2.0, 0.5]);
            half_sweep(&mut level, &diagonal, &mut psi, (0..3).rev(), &mut counters).unwrap();
            assert_eq!(
                psi.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                [
                    4588567540340219356,
                    13827251815928054852,
                    4596373779694328218
                ]
            );
            assert_eq!(
                level
                    .b_prime
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                [
                    4607182418800017408,
                    13834762506556617523,
                    4596036009722275433
                ]
            );
            let wrong = (
                psi.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
                level
                    .b_prime
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
            );
            let mut correct_level = SpecLduLevel::new(&csr);
            let mut correct_psi = vec![0.25, -0.5, 0.75];
            sweep(
                &mut correct_level,
                &diagonal,
                &[1.0, -2.0, 0.5],
                &mut correct_psi,
                &mut Counters::default(),
            )
            .unwrap();
            assert_ne!(
                wrong,
                (
                    correct_psi
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>(),
                    correct_level
                        .b_prime
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>()
                )
            );

            let (canonical_psi, canonical_bprime) = run_order_fixture(false);
            let (reversed_psi, reversed_bprime) = run_order_fixture(true);
            assert_eq!(
                canonical_psi,
                [
                    13907265759572411461,
                    13826200974869677124,
                    4594572341561366938
                ]
            );
            assert_eq!(
                canonical_bprime,
                [
                    4607182418800017408,
                    13834241774413728973,
                    4590744298198977742
                ]
            );
            assert_eq!(
                reversed_psi,
                [
                    13907265759572411460,
                    13826200974869677124,
                    4594572341561366938
                ]
            );
            assert_eq!(
                reversed_bprime,
                [
                    4607182418800017408,
                    13834241774413728972,
                    4590744298198977742
                ]
            );
        }

        fn half_sweep_with_rhs(
            level: &mut SpecLduLevel,
            diagonal: &[f64],
            rhs: &[f64],
            psi: &mut [f64],
            cells: impl Iterator<Item = usize>,
            counters: &mut Counters,
        ) {
            level.b_prime.copy_from_slice(rhs);
            half_sweep(level, diagonal, psi, cells, counters).unwrap();
        }

        fn order_fixture() -> CsrMatrix {
            matrix(vec![
                vec![
                    (0, 4.0),
                    (1, -1048576.0),
                    (1, -2f64.powi(-33)),
                    (2, -2f64.powi(-33)),
                ],
                vec![
                    (0, -2f64.powi(-20)),
                    (0, -2f64.powi(-21)),
                    (1, 3.0),
                    (2, -0.5),
                ],
                vec![(0, -2f64.powi(-22)), (1, -0.5), (2, 2.5)],
            ])
        }

        fn run_order_fixture(reverse: bool) -> ([u64; 3], [u64; 3]) {
            let csr = order_fixture();
            let mut level = SpecLduLevel::new(&csr);
            if reverse {
                level.lower_addr[0..3].reverse();
                level.upper_addr[0..3].reverse();
                level.lower_csr[0..3].reverse();
                level.upper_csr[0..3].reverse();
                level.lower[0..3].reverse();
                level.upper[0..3].reverse();
            }
            let mut psi = [0.25, 1.0, 1.0];
            sweep(
                &mut level,
                &diagonal(&csr).unwrap(),
                &[1.0, -2.0, 0.5],
                &mut psi,
                &mut Counters::default(),
            )
            .unwrap();
            (
                psi.map(f64::to_bits),
                std::array::from_fn(|i| level.b_prime[i].to_bits()),
            )
        }

        #[test]
        fn ldu_b1_symgs_matches_cached_csr_kernel_with_nonzero_initial() {
            for csr in [
                main_fixture(),
                matrix(vec![
                    vec![(0, 2.0), (1, -1.0)],
                    vec![(1, 3.0), (2, -0.5)],
                    vec![(1, -0.25), (2, 2.5)],
                ]),
            ] {
                let diagonal = diagonal(&csr).unwrap();
                let diagonal_slots = (0..csr.rows())
                    .map(|row| {
                        (csr.row_offsets()[row]..csr.row_offsets()[row + 1])
                            .find(|&slot| csr.col_indices()[slot] == row)
                            .unwrap()
                    })
                    .collect::<Vec<_>>();
                let rhs = vec![1.0, -2.0, 0.5];
                let initial = vec![0.25, -0.5, 0.75];
                let mut ldu = initial.clone();
                let mut level = SpecLduLevel::new(&csr);
                sweep(
                    &mut level,
                    &diagonal,
                    &rhs,
                    &mut ldu,
                    &mut Counters::default(),
                )
                .unwrap();
                let mut cached = initial;
                gauss_seidel_sweep_with_cached_diagonal(
                    &csr,
                    &diagonal_slots,
                    &diagonal,
                    &rhs,
                    &mut cached,
                    0..csr.rows(),
                )
                .unwrap();
                gauss_seidel_sweep_with_cached_diagonal(
                    &csr,
                    &diagonal_slots,
                    &diagonal,
                    &rhs,
                    &mut cached,
                    (0..csr.rows()).rev(),
                )
                .unwrap();
                for (actual, expected) in ldu.iter().zip(cached) {
                    assert!((actual - expected).abs() <= 64.0 * f64::EPSILON);
                }
            }
        }

        #[test]
        fn ldu_b1_dimension_and_diagonal_failures_are_exact_and_non_mutating() {
            let csr = main_fixture();
            for (diagonal, rhs, expected) in [
                (
                    vec![4.0, 3.0],
                    vec![1.0, -2.0, 0.5],
                    "LDU B1 diagonal length mismatch: expected 3, got 2",
                ),
                (
                    vec![4.0, 3.0, 2.5],
                    vec![1.0],
                    "LDU B1 rhs length mismatch: expected 3, got 1",
                ),
            ] {
                let mut level = SpecLduLevel::new(&csr);
                let mut psi = vec![0.25, -0.5, 0.75];
                let before_psi = psi.clone();
                let before_level = level.state();
                let mut counters = Counters::default();
                assert_eq!(
                    sweep(&mut level, &diagonal, &rhs, &mut psi, &mut counters).unwrap_err(),
                    expected
                );
                assert_eq!(psi, before_psi);
                assert_eq!(level.state(), before_level);
                assert_eq!(counters, Counters::default());
            }
            let mut level = SpecLduLevel::new(&csr);
            level.b_prime.pop();
            let mut psi = vec![0.25, -0.5, 0.75];
            let before_psi = psi.clone();
            let before_level = level.state();
            let mut counters = Counters::default();
            assert_eq!(
                sweep(
                    &mut level,
                    &[4.0, 3.0, 2.5],
                    &[1.0, -2.0, 0.5],
                    &mut psi,
                    &mut counters
                )
                .unwrap_err(),
                "LDU B1 bPrime length mismatch: expected 3, got 2"
            );
            assert_eq!(psi, before_psi);
            assert_eq!(level.state(), before_level);
            assert_eq!(counters, Counters::default());
            for bad in [0.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
                let mut level = SpecLduLevel::new(&csr);
                let mut psi = vec![0.25, -0.5, 0.75];
                let before_psi = psi.clone();
                let before_level = level.state();
                let mut counters = Counters::default();
                assert_eq!(
                    sweep(
                        &mut level,
                        &[bad, 3.0, 2.5],
                        &[1.0, -2.0, 0.5],
                        &mut psi,
                        &mut counters
                    )
                    .unwrap_err(),
                    "LDU B1 diagonal row 0 must be finite and non-zero"
                );
                assert_eq!(psi, before_psi);
                assert_eq!(level.state(), before_level);
                assert_eq!(counters, Counters::default());
            }
        }

        #[test]
        fn ldu_b1_nonfinite_update_fails_before_psi_write() {
            let csr = matrix(vec![
                vec![(0, f64::MIN_POSITIVE), (1, -1.0)],
                vec![(0, -1.0), (1, 2.0)],
            ]);
            let mut level = SpecLduLevel::new(&csr);
            let mut psi = vec![1.0, 1.0];
            let before_psi = psi.clone();
            let mut expected_level = level.state();
            expected_level.b_prime = vec![f64::MAX, 0.0];
            let mut counters = Counters::default();
            assert_eq!(
                sweep(
                    &mut level,
                    &[f64::MIN_POSITIVE, 2.0],
                    &[f64::MAX, 0.0],
                    &mut psi,
                    &mut counters
                )
                .unwrap_err(),
                "LDU B1 update row 0 is not finite"
            );
            assert_eq!(psi, before_psi);
            assert_eq!(level.state(), expected_level);
            assert_eq!(
                counters,
                Counters {
                    rhs_copies: 1,
                    rows: 0
                }
            );
        }
    }

    #[test]
    fn flexible_cg_profiled_and_plain_solve_match_pcg_on_spd_grid() {
        let matrix = poisson_grid(3, 3, 1.0);
        let rhs = vec![1.0; matrix.rows()];
        let options = GamgOptions {
            outer_solver: GamgOuterSolver::FlexibleCg,
            max_iterations: 40,
            tolerance: 1.0e-12,
            n_cells_in_coarsest_level: 2,
            smoother: GamgSmoother::SymGaussSeidel,
            direct_solve_coarsest: true,
            ..GamgOptions::default()
        };
        let controls = NormalizedL1GamgSolveControls {
            normalization_factor: 1.0,
            tolerance: 1.0e-12,
            relative_tolerance: 0.0,
            l2_controls: options.into(),
        };
        let mut plain_workspace =
            GamgWorkspace::new(&matrix, options).expect("plain FCG workspace");
        let plain = plain_workspace
            .solve_normalized_l1_with_controls(&matrix, &rhs, None, controls)
            .expect("plain FCG solve");
        let mut profiled_workspace =
            GamgWorkspace::new(&matrix, options).expect("profiled FCG workspace");
        let profiled = profiled_workspace
            .solve_normalized_l1_with_controls_profiled(&matrix, &rhs, None, controls)
            .expect("profiled FCG solve");
        let pcg = preconditioned_conjugate_gradient_solve(
            &matrix,
            &rhs,
            None,
            PreconditionedConjugateGradientOptions {
                max_iterations: 40,
                tolerance: 1.0e-12,
                preconditioner: CgPreconditioner::IncompleteCholesky,
            },
        )
        .expect("PCG oracle");

        assert!(plain.converged, "{plain:?}");
        assert_eq!(plain.iterations, profiled.report.iterations);
        assert_eq!(
            plain.residual_norm.to_bits(),
            profiled.report.residual_norm.to_bits()
        );
        assert_eq!(plain.solution.len(), profiled.report.solution.len());
        for (plain_value, profiled_value) in plain.solution.iter().zip(&profiled.report.solution) {
            assert_eq!(plain_value.to_bits(), profiled_value.to_bits());
        }
        assert_eq!(profiled.timing.v_cycles, profiled.report.iterations);
        assert_eq!(
            profiled.timing.outer_matrix_vector_products,
            profiled.report.iterations
        );
        assert_eq!(
            profiled.timing.finest_residual_evaluations,
            profiled.report.iterations + 1
        );
        assert_eq!(
            profiled.timing.outer_reductions,
            9 * profiled.report.iterations - 1
        );
        assert_close(&plain.solution, &pcg.solution, 1.0e-10);
    }

    #[test]
    fn flexible_cg_mmax_one_direction_matches_two_step_nonlinear_oracle() {
        let matrix = poisson_grid(3, 3, 1.0);
        let rhs = (0..matrix.rows())
            .map(|row| 1.0 + row as f64 / 8.0)
            .collect::<Vec<_>>();
        let options = GamgOptions {
            outer_solver: GamgOuterSolver::FlexibleCg,
            max_iterations: 2,
            min_iterations: 2,
            tolerance: 0.0,
            relative_tolerance: 0.0,
            n_cells_in_coarsest_level: 2,
            smoother: GamgSmoother::SymGaussSeidel,
            direct_solve_coarsest: true,
            ..GamgOptions::default()
        };
        let controls = options.into();

        // Build the expected two-step result independently around the same
        // nonlinear GAMG V-cycle that the public FCG solve uses as M^-1.
        // The outer recurrence below deliberately does not call the production
        // direction helper, so numerator/history/true-residual wiring remains
        // independently checked.
        let oracle_options = GamgOptions {
            outer_solver: GamgOuterSolver::Standalone,
            ..options
        };
        let mut oracle_workspace =
            GamgWorkspace::new(&matrix, oracle_options).expect("oracle GAMG workspace");
        let mut timing = GamgKernelTiming::default();
        let mut solution = vec![0.0; matrix.rows()];
        let mut residual = rhs.clone();
        let mut previous_direction = vec![0.0; matrix.rows()];
        let mut previous_normalized_matrix_direction = vec![0.0; matrix.rows()];
        let mut product = vec![0.0; matrix.rows()];
        let mut observed_nonzero_truncation = false;

        for iteration in 0..2 {
            let mut preconditioned = vec![0.0; matrix.rows()];
            oracle_workspace.residuals[0].copy_from_slice(&residual);
            oracle_workspace
                .v_cycle::<false, false>(&mut preconditioned, &residual, controls, &mut timing)
                .expect("oracle GAMG V-cycle");
            validate_fcg_preconditioner_output(&preconditioned)
                .expect("finite oracle preconditioner output");

            let mut direction = preconditioned.clone();
            if iteration > 0 {
                let truncation = super::dot(&preconditioned, &previous_normalized_matrix_direction);
                observed_nonzero_truncation |= truncation != 0.0;
                for row in 0..direction.len() {
                    direction[row] -= truncation * previous_direction[row];
                }
            }
            matrix
                .matvec_into(&direction, &mut product)
                .expect("oracle direction matvec");
            let numerator = super::dot(&direction, &residual);
            let curvature = super::dot(&direction, &product);
            assert!(numerator > 0.0 && curvature > 0.0);
            let alpha = numerator / curvature;
            for (value, direction) in solution.iter_mut().zip(&direction) {
                *value += alpha * direction;
            }

            previous_direction.copy_from_slice(&direction);
            for (value, matrix_direction) in previous_normalized_matrix_direction
                .iter_mut()
                .zip(&product)
            {
                *value = matrix_direction / curvature;
            }
            matrix
                .matvec_into(&solution, &mut product)
                .expect("oracle true-residual matvec");
            for ((value, source), matrix_value) in residual.iter_mut().zip(&rhs).zip(&product) {
                *value = source - matrix_value;
            }
        }
        assert!(
            observed_nonzero_truncation,
            "oracle must exercise the mmax=1 history term"
        );

        let mut production_workspace =
            GamgWorkspace::new(&matrix, options).expect("production FCG workspace");
        let report = production_workspace
            .solve(&matrix, &rhs, None)
            .expect("production two-step FCG solve");
        let mut profiled_workspace =
            GamgWorkspace::new(&matrix, options).expect("profiled production FCG workspace");
        let profiled = profiled_workspace
            .solve_with_controls_profiled(&matrix, &rhs, None, controls)
            .expect("profiled production two-step FCG solve");

        assert_eq!(report.iterations, 2);
        assert_eq!(profiled.report.iterations, 2);
        assert_eq!(report.solution.len(), solution.len());
        for ((actual, profiled), expected) in report
            .solution
            .iter()
            .zip(&profiled.report.solution)
            .zip(solution)
        {
            assert_eq!(actual.to_bits(), expected.to_bits());
            assert_eq!(profiled.to_bits(), expected.to_bits());
        }
        assert_eq!(
            report.residual_norm.to_bits(),
            super::l2_norm(&residual).to_bits()
        );
        assert_eq!(
            profiled.report.residual_norm.to_bits(),
            report.residual_norm.to_bits()
        );
    }

    #[test]
    fn flexible_cg_exact_zero_honours_minimum_cycles_and_profile_counts() {
        let matrix = poisson_grid(2, 2, 1.0);
        let rhs = vec![0.0; matrix.rows()];
        let options = GamgOptions {
            outer_solver: GamgOuterSolver::FlexibleCg,
            max_iterations: 2,
            min_iterations: 2,
            tolerance: 1.0e-12,
            n_cells_in_coarsest_level: 2,
            smoother: GamgSmoother::SymGaussSeidel,
            direct_solve_coarsest: true,
            ..GamgOptions::default()
        };
        let controls = NormalizedL1GamgSolveControls {
            normalization_factor: 1.0,
            tolerance: 1.0e-12,
            relative_tolerance: 0.0,
            l2_controls: options.into(),
        };
        let mut workspace = GamgWorkspace::new(&matrix, options).expect("zero FCG workspace");
        let profiled = workspace
            .solve_normalized_l1_with_controls_profiled(&matrix, &rhs, None, controls)
            .expect("zero FCG solve");

        assert!(profiled.report.converged);
        assert_eq!(profiled.report.iterations, 2);
        assert_eq!(profiled.report.residual_norm.to_bits(), 0.0f64.to_bits());
        assert_eq!(profiled.timing.v_cycles, 2);
        assert_eq!(profiled.timing.outer_matrix_vector_products, 0);
        assert_eq!(profiled.timing.outer_reductions, 4);
        assert_eq!(profiled.timing.finest_residual_evaluations, 3);
    }

    #[test]
    fn flexible_cg_normalized_l1_keeps_strict_equality_boundary() {
        let matrix = poisson_grid(3, 3, 1.0);
        let mut rhs = vec![0.0; matrix.rows()];
        rhs[0] = 1.0;
        let options = GamgOptions {
            outer_solver: GamgOuterSolver::FlexibleCg,
            max_iterations: 1,
            min_iterations: 1,
            tolerance: 0.0,
            n_cells_in_coarsest_level: 2,
            smoother: GamgSmoother::SymGaussSeidel,
            direct_solve_coarsest: true,
            ..GamgOptions::default()
        };
        let controls = |tolerance| NormalizedL1GamgSolveControls {
            normalization_factor: 1.0,
            tolerance,
            relative_tolerance: 0.0,
            l2_controls: options.into(),
        };
        let mut probe_workspace = GamgWorkspace::new(&matrix, options).expect("probe workspace");
        let probe = probe_workspace
            .solve_normalized_l1_with_controls(&matrix, &rhs, None, controls(0.0))
            .expect("one-step FCG probe");
        let product = matrix
            .matvec(&probe.solution)
            .expect("probe residual matvec");
        let exact_l1 = rhs
            .iter()
            .zip(product)
            .map(|(source, product)| (source - product).abs())
            .sum::<f64>();
        assert!(exact_l1 > 0.0 && exact_l1.is_finite());

        let mut equal_workspace = GamgWorkspace::new(&matrix, options).expect("equal workspace");
        let equal = equal_workspace
            .solve_normalized_l1_with_controls(&matrix, &rhs, None, controls(exact_l1))
            .expect("equality-boundary FCG solve");
        let mut next_workspace = GamgWorkspace::new(&matrix, options).expect("next workspace");
        let next = next_workspace
            .solve_normalized_l1_with_controls(&matrix, &rhs, None, controls(exact_l1.next_up()))
            .expect("next-up FCG solve");

        assert!(
            !equal.converged,
            "equality must not satisfy strict normalized L1"
        );
        assert!(next.converged, "next_up must satisfy strict normalized L1");
        assert_eq!(equal.iterations, 1);
        assert_eq!(next.iterations, 1);
        for (equal_value, next_value) in equal.solution.iter().zip(next.solution) {
            assert_eq!(equal_value.to_bits(), next_value.to_bits());
        }
    }

    #[test]
    fn flexible_cg_reports_nonpositive_spd_products_as_breakdown() {
        for (rows, rhs, expected_numerator_positive, expected_curvature_positive) in [
            (
                vec![vec![(0, -4.0), (1, -4.0)], vec![(0, -4.0), (1, -3.0)]],
                vec![-2.0, 0.5],
                false,
                true,
            ),
            (
                vec![vec![(0, -4.0), (1, -4.0)], vec![(0, -4.0), (1, 0.5)]],
                vec![-2.0, 0.5],
                true,
                false,
            ),
        ] {
            let matrix = CsrMatrix::from_rows(rows, 2).expect("symmetric breakdown matrix");
            let options = GamgOptions {
                outer_solver: GamgOuterSolver::FlexibleCg,
                max_iterations: 1,
                tolerance: 0.0,
                relative_tolerance: 0.0,
                n_cells_in_coarsest_level: 1,
                smoother: GamgSmoother::GaussSeidel,
                n_finest_sweeps: 1,
                scale_correction: false,
                direct_solve_coarsest: true,
                ..GamgOptions::default()
            };
            let controls = options.into();
            let mut probe = GamgWorkspace::new(
                &matrix,
                GamgOptions {
                    outer_solver: GamgOuterSolver::Standalone,
                    ..options
                },
            )
            .expect("breakdown probe workspace");
            probe.residuals[0].copy_from_slice(&rhs);
            let mut direction = vec![0.0; matrix.rows()];
            probe
                .v_cycle::<false, false>(
                    &mut direction,
                    &rhs,
                    controls,
                    &mut GamgKernelTiming::default(),
                )
                .expect("breakdown probe V-cycle");
            let matrix_direction = matrix.matvec(&direction).expect("breakdown probe matvec");
            let numerator = super::dot(&direction, &rhs);
            let curvature = super::dot(&direction, &matrix_direction);
            assert_eq!(numerator > 0.0, expected_numerator_positive);
            assert_eq!(curvature > 0.0, expected_curvature_positive);

            let mut workspace = GamgWorkspace::new(&matrix, options).expect("indefinite workspace");
            let report = workspace
                .solve(&matrix, &rhs, None)
                .expect("indefinite FCG returns a breakdown report");

            assert!(!report.converged);
            assert_eq!(report.iterations, 1);
            assert_eq!(
                report.termination,
                super::IterativeSolveTermination::Breakdown
            );
            assert!(report.solution.iter().all(|value| value.to_bits() == 0));
        }

        let matrix = CsrMatrix::from_rows(
            vec![vec![(0, 2.0), (1, -1.0)], vec![(0, -1.0), (1, 2.0)]],
            2,
        )
        .expect("finite overflow matrix");
        let rhs = vec![1.0e308, 0.0];
        let initial = vec![0.0, 0.0];
        let options = GamgOptions {
            outer_solver: GamgOuterSolver::FlexibleCg,
            max_iterations: 1,
            tolerance: 0.0,
            relative_tolerance: 0.0,
            n_cells_in_coarsest_level: 1,
            n_finest_sweeps: 0,
            scale_correction: false,
            direct_solve_coarsest: true,
            ..GamgOptions::default()
        };
        let mut workspace = GamgWorkspace::new(&matrix, options).expect("overflow workspace");
        let error = workspace
            .solve(&matrix, &rhs, Some(&initial))
            .expect_err("non-finite FCG numerator must fail closed");
        assert_eq!(error.to_string(), "GAMG FCG step numerator is not finite");
        assert_eq!(rhs, vec![1.0e308, 0.0]);
        assert_eq!(initial, vec![0.0, 0.0]);
    }

    #[test]
    fn flexible_cg_validation_is_fail_closed_and_non_mutating() {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let values = [1.0, bad];
            let error = validate_fcg_preconditioner_output(&values)
                .expect_err("non-finite FCG preconditioner output must fail");
            assert_eq!(
                error.to_string(),
                "GAMG FCG preconditioner output is not finite"
            );
            assert_eq!(values[0].to_bits(), 1.0f64.to_bits());
            assert_eq!(values[1].to_bits(), bad.to_bits());
        }

        let mut direction = [3.0, 4.0];
        let before = direction.map(f64::to_bits);
        let error = fcg_mmax_one_direction(&[1.0], None, &mut direction)
            .expect_err("length mismatch must fail before direction mutation");
        assert!(error.to_string().contains("direction length mismatch"));
        assert_eq!(direction.map(f64::to_bits), before);

        let error =
            fcg_mmax_one_direction(&[1.0, 2.0], Some((&[1.0], &[1.0, 2.0])), &mut direction)
                .expect_err("history length mismatch must fail before direction mutation");
        assert!(
            error
                .to_string()
                .contains("previous-direction length mismatch")
        );
        assert_eq!(direction.map(f64::to_bits), before);
    }

    #[test]
    fn flexible_cg_workspace_reuses_buffers_and_resets_history_across_coefficients() {
        let first_matrix = poisson_grid(3, 3, 1.0);
        let rhs = (0..first_matrix.rows())
            .map(|row| 1.0 + row as f64 / 8.0)
            .collect::<Vec<_>>();
        let standalone = GamgWorkspace::new(
            &first_matrix,
            GamgOptions {
                n_cells_in_coarsest_level: 2,
                ..GamgOptions::default()
            },
        )
        .expect("standalone GAMG workspace");
        for buffer in [
            &standalone.fcg_residual,
            &standalone.fcg_preconditioned_residual,
            &standalone.fcg_direction,
            &standalone.fcg_matrix_direction,
            &standalone.fcg_previous_direction,
            &standalone.fcg_previous_matrix_direction,
        ] {
            assert!(buffer.is_empty());
            assert_eq!(buffer.capacity(), 0);
        }
        let options = GamgOptions {
            outer_solver: GamgOuterSolver::FlexibleCg,
            max_iterations: 40,
            tolerance: 1.0e-11,
            n_cells_in_coarsest_level: 2,
            smoother: GamgSmoother::SymGaussSeidel,
            direct_solve_coarsest: true,
            ..GamgOptions::default()
        };
        let mut workspace =
            GamgWorkspace::new(&first_matrix, options).expect("reused FCG workspace");
        let pointers = [
            workspace.fcg_residual.as_ptr(),
            workspace.fcg_preconditioned_residual.as_ptr(),
            workspace.fcg_direction.as_ptr(),
            workspace.fcg_matrix_direction.as_ptr(),
            workspace.fcg_previous_direction.as_ptr(),
            workspace.fcg_previous_matrix_direction.as_ptr(),
        ];

        for lifecycle in 0..10 {
            let matrix = poisson_grid(3, 3, 1.0 + lifecycle as f64 / 16.0);
            let reused = workspace
                .solve(&matrix, &rhs, None)
                .expect("reused FCG coefficient lifecycle");
            let mut fresh_workspace =
                GamgWorkspace::new(&matrix, options).expect("fresh FCG workspace");
            let fresh = fresh_workspace
                .solve(&matrix, &rhs, None)
                .expect("fresh FCG coefficient lifecycle");

            assert!(reused.converged && fresh.converged);
            assert_eq!(reused.iterations, fresh.iterations);
            assert_eq!(
                reused.residual_norm.to_bits(),
                fresh.residual_norm.to_bits()
            );
            for (reused_value, fresh_value) in reused.solution.iter().zip(fresh.solution) {
                assert_eq!(reused_value.to_bits(), fresh_value.to_bits());
            }
            assert_eq!(
                [
                    workspace.fcg_residual.as_ptr(),
                    workspace.fcg_preconditioned_residual.as_ptr(),
                    workspace.fcg_direction.as_ptr(),
                    workspace.fcg_matrix_direction.as_ptr(),
                    workspace.fcg_previous_direction.as_ptr(),
                    workspace.fcg_previous_matrix_direction.as_ptr(),
                ],
                pointers
            );
        }
    }

    #[test]
    fn hierarchy_diagnostics_report_exact_shapes_histograms_and_complexity_terms() {
        let matrices = vec![
            tridiagonal_matrix(8),
            tridiagonal_matrix(4),
            tridiagonal_matrix(2),
        ];
        let transfers = vec![
            GamgTransfer {
                fine_to_coarse: vec![0, 0, 1, 1, 2, 2, 3, 3],
                fine_entry_to_coarse_entry: Vec::new(),
            },
            GamgTransfer {
                fine_to_coarse: vec![0, 0, 1, 1],
                fine_entry_to_coarse_entry: Vec::new(),
            },
        ];
        let options = GamgOptions {
            smoother: GamgSmoother::SymGaussSeidel,
            direct_solve_coarsest: true,
            ..GamgOptions::default()
        };

        let diagnostics = super::build_hierarchy_diagnostics(&matrices, &transfers, options)
            .expect("exact hierarchy diagnostics");

        assert_eq!(
            diagnostics.levels,
            vec![
                super::GamgHierarchyLevelDiagnostics {
                    level: 0,
                    cells: 8,
                    nonzeros: 22,
                },
                super::GamgHierarchyLevelDiagnostics {
                    level: 1,
                    cells: 4,
                    nonzeros: 10,
                },
                super::GamgHierarchyLevelDiagnostics {
                    level: 2,
                    cells: 2,
                    nonzeros: 4,
                },
            ]
        );
        assert_eq!(
            diagnostics.transfers,
            vec![
                super::GamgTransferDiagnostics {
                    fine_level: 0,
                    coarse_level: 1,
                    fine_cells: 8,
                    coarse_cells: 4,
                    singleton_fine_cells: 0,
                    unmatched_fine_cells: 0,
                    min_aggregate_size: 2,
                    max_aggregate_size: 2,
                    aggregate_size_histogram: vec![super::GamgAggregateSizeBin {
                        aggregate_size: 2,
                        aggregate_count: 4,
                    }],
                },
                super::GamgTransferDiagnostics {
                    fine_level: 1,
                    coarse_level: 2,
                    fine_cells: 4,
                    coarse_cells: 2,
                    singleton_fine_cells: 0,
                    unmatched_fine_cells: 0,
                    min_aggregate_size: 2,
                    max_aggregate_size: 2,
                    aggregate_size_histogram: vec![super::GamgAggregateSizeBin {
                        aggregate_size: 2,
                        aggregate_count: 2,
                    }],
                },
            ]
        );
        assert_eq!(diagnostics.grid_complexity_terms(), Some((14, 8)));
        assert_eq!(diagnostics.operator_complexity_terms(), Some((36, 22)));
        assert_eq!(diagnostics.smoother_passes_per_sweep, 2);
        assert!(diagnostics.direct_solve_coarsest);
    }

    #[test]
    fn hierarchy_diagnostics_reject_structural_mismatches_without_mutating_inputs() {
        let matrix_snapshot = |matrices: &[CsrMatrix]| {
            matrices
                .iter()
                .map(|matrix| {
                    (
                        matrix.rows(),
                        matrix.cols(),
                        matrix.row_offsets().to_vec(),
                        matrix.col_indices().to_vec(),
                        matrix
                            .values()
                            .iter()
                            .map(|value| value.to_bits())
                            .collect::<Vec<_>>(),
                    )
                })
                .collect::<Vec<_>>()
        };
        let transfer_snapshot = |transfers: &[GamgTransfer]| {
            transfers
                .iter()
                .map(|transfer| {
                    (
                        transfer.fine_to_coarse.clone(),
                        transfer.fine_entry_to_coarse_entry.clone(),
                    )
                })
                .collect::<Vec<_>>()
        };
        let assert_rejected = |matrices: Vec<CsrMatrix>,
                               transfers: Vec<GamgTransfer>,
                               expected: &str| {
            let matrices_before = matrix_snapshot(&matrices);
            let transfers_before = transfer_snapshot(&transfers);
            let error =
                super::build_hierarchy_diagnostics(&matrices, &transfers, GamgOptions::default())
                    .expect_err("invalid hierarchy diagnostics must fail closed");
            assert_eq!(error.to_string(), expected);
            assert_eq!(matrix_snapshot(&matrices), matrices_before);
            assert_eq!(transfer_snapshot(&transfers), transfers_before);
        };

        assert_rejected(
            vec![tridiagonal_matrix(4), tridiagonal_matrix(2)],
            Vec::new(),
            "GAMG profile hierarchy expected one fewer transfers than levels, got 2 levels and 0 transfers",
        );
        assert_rejected(
            vec![tridiagonal_matrix(4), tridiagonal_matrix(2)],
            vec![GamgTransfer {
                fine_to_coarse: vec![0, 0, 1],
                fine_entry_to_coarse_entry: vec![17, 19],
            }],
            "GAMG profile transfer 0->1 expected 4 fine cells, got 3",
        );
        assert_rejected(
            vec![tridiagonal_matrix(4), tridiagonal_matrix(2)],
            vec![GamgTransfer {
                fine_to_coarse: vec![0, 0, 1, 2],
                fine_entry_to_coarse_entry: vec![23],
            }],
            "GAMG profile transfer 0->1 maps fine cell 3 to out-of-range coarse cell 2 of 2",
        );
        assert_rejected(
            vec![tridiagonal_matrix(4), tridiagonal_matrix(3)],
            vec![GamgTransfer {
                fine_to_coarse: vec![0, 0, 1, 1],
                fine_entry_to_coarse_entry: vec![29, 31, 37],
            }],
            "GAMG profile transfer 0->1 contains an empty aggregate",
        );
    }

    #[test]
    fn hierarchy_diagnostics_distinguish_odd_attached_and_singleton_cells() {
        let attached = super::build_hierarchy_diagnostics(
            &[tridiagonal_matrix(5), tridiagonal_matrix(2)],
            &[GamgTransfer {
                fine_to_coarse: vec![0, 0, 0, 1, 1],
                fine_entry_to_coarse_entry: Vec::new(),
            }],
            GamgOptions::default(),
        )
        .expect("odd attached aggregate diagnostics");
        let attached_transfer = &attached.transfers[0];
        assert_eq!(attached_transfer.singleton_fine_cells, 0);
        assert_eq!(attached_transfer.unmatched_fine_cells, 1);
        assert_eq!(attached_transfer.min_aggregate_size, 2);
        assert_eq!(attached_transfer.max_aggregate_size, 3);
        assert_eq!(
            attached_transfer.aggregate_size_histogram,
            vec![
                super::GamgAggregateSizeBin {
                    aggregate_size: 2,
                    aggregate_count: 1,
                },
                super::GamgAggregateSizeBin {
                    aggregate_size: 3,
                    aggregate_count: 1,
                },
            ]
        );

        let singleton = super::build_hierarchy_diagnostics(
            &[tridiagonal_matrix(5), tridiagonal_matrix(3)],
            &[GamgTransfer {
                fine_to_coarse: vec![0, 0, 1, 1, 2],
                fine_entry_to_coarse_entry: Vec::new(),
            }],
            GamgOptions::default(),
        )
        .expect("odd singleton aggregate diagnostics");
        let singleton_transfer = &singleton.transfers[0];
        assert_eq!(singleton_transfer.singleton_fine_cells, 1);
        assert_eq!(singleton_transfer.unmatched_fine_cells, 1);
        assert_eq!(singleton_transfer.min_aggregate_size, 1);
        assert_eq!(singleton_transfer.max_aggregate_size, 2);
        assert_eq!(
            singleton_transfer.aggregate_size_histogram,
            vec![
                super::GamgAggregateSizeBin {
                    aggregate_size: 1,
                    aggregate_count: 1,
                },
                super::GamgAggregateSizeBin {
                    aggregate_size: 2,
                    aggregate_count: 2,
                },
            ]
        );
    }

    #[test]
    fn hierarchy_weighted_work_applies_smoother_and_fcg_matrix_product_factors() {
        let matrices = vec![tridiagonal_matrix(8), tridiagonal_matrix(4)];
        let transfers = vec![GamgTransfer {
            fine_to_coarse: vec![0, 0, 1, 1, 2, 2, 3, 3],
            fine_entry_to_coarse_entry: Vec::new(),
        }];
        let mut gauss_seidel = timing_with_hierarchy(
            &matrices,
            &transfers,
            GamgOptions {
                smoother: GamgSmoother::GaussSeidel,
                ..GamgOptions::default()
            },
        );
        gauss_seidel.levels[0].smoothing_sweeps = 2;
        gauss_seidel.levels[0].residual_evaluations = 1;
        gauss_seidel.levels[0].scaling_calls = 1;
        gauss_seidel.levels[1].smoothing_sweeps = 3;
        gauss_seidel.levels[1].residual_evaluations = 2;
        gauss_seidel.levels[1].scaling_calls = 1;
        gauss_seidel.finest_residual_evaluations = 2;
        gauss_seidel.outer_matrix_vector_products = 3;

        let mut symmetric = timing_with_hierarchy(
            &matrices,
            &transfers,
            GamgOptions {
                smoother: GamgSmoother::SymGaussSeidel,
                ..GamgOptions::default()
            },
        );
        symmetric.levels.clone_from(&gauss_seidel.levels);
        symmetric.finest_residual_evaluations = gauss_seidel.finest_residual_evaluations;
        symmetric.outer_matrix_vector_products = gauss_seidel.outer_matrix_vector_products;

        assert_eq!(gauss_seidel.nnz_weighted_smoothing_work(), Some(74));
        assert_eq!(symmetric.nnz_weighted_smoothing_work(), Some(148));
        assert_eq!(gauss_seidel.nnz_weighted_sparse_work(), Some(258));
        assert_eq!(symmetric.nnz_weighted_sparse_work(), Some(332));
    }

    #[test]
    fn hierarchy_profile_is_observational_and_records_coarsest_iterations() {
        let matrix = poisson_grid(4, 4, 1.0);
        let rhs = (0..matrix.rows())
            .map(|row| 1.0 + row as f64 / matrix.rows() as f64)
            .collect::<Vec<_>>();
        let base_options = GamgOptions {
            max_iterations: 2,
            min_iterations: 2,
            tolerance: 0.0,
            relative_tolerance: 0.0,
            n_cells_in_coarsest_level: 2,
            direct_solve_coarsest: false,
            ..GamgOptions::default()
        };
        let mut plain_workspace =
            GamgWorkspace::new(&matrix, base_options).expect("plain hierarchy workspace");
        let mut plain_timing = GamgKernelTiming::default();
        let plain = plain_workspace
            .solve_with_controls_internal::<false, false>(
                &matrix,
                &rhs,
                None,
                base_options.into(),
                &mut plain_timing,
            )
            .expect("plain hierarchy solve");
        assert!(plain_timing.hierarchy.is_none());
        assert!(plain_timing.levels.is_empty());
        assert_eq!(plain_timing.solves, 0);

        let mut iterative_workspace =
            GamgWorkspace::new(&matrix, base_options).expect("profiled iterative workspace");
        let iterative = iterative_workspace
            .solve_with_controls_profiled(&matrix, &rhs, None, base_options.into())
            .expect("profiled iterative solve");
        assert_eq!(plain.iterations, iterative.report.iterations);
        assert_eq!(plain.converged, iterative.report.converged);
        assert_eq!(plain.termination, iterative.report.termination);
        assert_eq!(
            plain.residual_norm.to_bits(),
            iterative.report.residual_norm.to_bits()
        );
        assert_eq!(
            plain
                .solution
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            iterative
                .report
                .solution
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
        let iterative_hierarchy = iterative
            .timing
            .hierarchy
            .as_ref()
            .expect("profiled solve must report its static hierarchy");
        assert!(!iterative_hierarchy.direct_solve_coarsest);
        let iterative_coarsest = iterative
            .timing
            .levels
            .last()
            .expect("iterative coarsest level");
        assert_eq!(
            iterative_coarsest.coarsest_solves,
            iterative.report.iterations
        );
        assert!(iterative_coarsest.coarsest_iterations >= iterative_coarsest.coarsest_solves);
        assert!(
            iterative.timing.levels[..iterative.timing.levels.len() - 1]
                .iter()
                .all(|level| level.coarsest_iterations == 0)
        );

        let direct_options = GamgOptions {
            direct_solve_coarsest: true,
            ..base_options
        };
        let mut direct_workspace =
            GamgWorkspace::new(&matrix, direct_options).expect("profiled direct workspace");
        let direct = direct_workspace
            .solve_with_controls_profiled(&matrix, &rhs, None, direct_options.into())
            .expect("profiled direct solve");
        assert!(
            direct
                .timing
                .hierarchy
                .as_ref()
                .expect("direct hierarchy diagnostics")
                .direct_solve_coarsest
        );
        let direct_coarsest = direct.timing.levels.last().expect("direct coarsest level");
        assert_eq!(direct_coarsest.coarsest_solves, direct.report.iterations);
        assert_eq!(direct_coarsest.coarsest_iterations, 0);
    }

    #[test]
    fn profiled_hierarchy_cache_is_success_bound_and_reused_across_ten_coefficients() {
        let matrix = poisson_grid(4, 4, 1.0);
        let rhs = (0..matrix.rows())
            .map(|row| 1.0 + row as f64 / matrix.rows() as f64)
            .collect::<Vec<_>>();
        let options = GamgOptions {
            max_iterations: 2,
            min_iterations: 2,
            tolerance: 0.0,
            relative_tolerance: 0.0,
            cache_agglomeration: true,
            n_cells_in_coarsest_level: 2,
            ..GamgOptions::default()
        };

        let mut failed_workspace =
            GamgWorkspace::new(&matrix, options).expect("failed-profile workspace");
        let error = failed_workspace
            .solve_with_controls_profiled(&matrix, &rhs[..rhs.len() - 1], None, options.into())
            .expect_err("invalid profiled solve must fail before caching diagnostics");
        assert_eq!(
            error.to_string(),
            "iterative solve expected rhs with 16 entries, got 15"
        );
        assert!(failed_workspace.profiled_hierarchy.is_none());

        let mut plain_workspace =
            GamgWorkspace::new(&matrix, options).expect("unprofiled hierarchy workspace");
        plain_workspace
            .solve_with_controls(&matrix, &rhs, None, options.into())
            .expect("unprofiled hierarchy solve");
        assert!(plain_workspace.profiled_hierarchy.is_none());

        let mut workspace =
            GamgWorkspace::new(&matrix, options).expect("profiled hierarchy workspace");
        let first = workspace
            .solve_with_controls_profiled(&matrix, &rhs, None, options.into())
            .expect("first profiled hierarchy solve");
        let cached = workspace
            .profiled_hierarchy
            .as_ref()
            .expect("successful profiled solve caches diagnostics")
            .clone();
        assert!(std::sync::Arc::ptr_eq(
            first
                .timing
                .hierarchy
                .as_ref()
                .expect("first timing hierarchy"),
            &cached,
        ));
        assert!(first.timing.hierarchy_diagnostic_seconds.is_finite());
        assert!(first.timing.hierarchy_diagnostic_seconds >= 0.0);

        for lifecycle in 1..10 {
            let matrix = poisson_grid(4, 4, 1.0 + lifecycle as f64 / 32.0);
            let profiled = workspace
                .solve_with_controls_profiled(&matrix, &rhs, None, options.into())
                .expect("cached profiled coefficient lifecycle");
            assert!(std::sync::Arc::ptr_eq(
                workspace
                    .profiled_hierarchy
                    .as_ref()
                    .expect("workspace hierarchy cache"),
                &cached,
            ));
            assert!(std::sync::Arc::ptr_eq(
                profiled
                    .timing
                    .hierarchy
                    .as_ref()
                    .expect("profile timing hierarchy"),
                &cached,
            ));
            assert_eq!(
                profiled.timing.hierarchy_diagnostic_seconds.to_bits(),
                0.0f64.to_bits(),
            );
        }

        let cached_before_failure = workspace
            .profiled_hierarchy
            .as_ref()
            .expect("cache before failed retry")
            .clone();
        workspace
            .solve_with_controls_profiled(&matrix, &rhs[..rhs.len() - 1], None, options.into())
            .expect_err("invalid retry must not replace cached diagnostics");
        assert!(std::sync::Arc::ptr_eq(
            workspace
                .profiled_hierarchy
                .as_ref()
                .expect("cache after failed retry"),
            &cached_before_failure,
        ));

        let uncached_options = GamgOptions {
            cache_agglomeration: false,
            ..options
        };
        let mut uncached_workspace = GamgWorkspace::new(&matrix, uncached_options)
            .expect("uncached profiled hierarchy workspace");
        let uncached_first = uncached_workspace
            .solve_with_controls_profiled(&matrix, &rhs, None, uncached_options.into())
            .expect("first uncached profiled solve");
        let uncached_first_hierarchy = uncached_first
            .timing
            .hierarchy
            .as_ref()
            .expect("first uncached timing hierarchy")
            .clone();
        assert!(std::sync::Arc::ptr_eq(
            uncached_workspace
                .profiled_hierarchy
                .as_ref()
                .expect("first uncached workspace hierarchy"),
            &uncached_first_hierarchy,
        ));
        let second_matrix = poisson_grid(4, 4, 1.25);
        let uncached_second = uncached_workspace
            .solve_with_controls_profiled(&second_matrix, &rhs, None, uncached_options.into())
            .expect("second uncached profiled solve");
        let uncached_second_hierarchy = uncached_second
            .timing
            .hierarchy
            .as_ref()
            .expect("second uncached timing hierarchy");
        assert!(!std::sync::Arc::ptr_eq(
            &uncached_first_hierarchy,
            uncached_second_hierarchy,
        ));
        assert!(std::sync::Arc::ptr_eq(
            uncached_workspace
                .profiled_hierarchy
                .as_ref()
                .expect("second uncached workspace hierarchy"),
            uncached_second_hierarchy,
        ));
        assert_eq!(uncached_second.timing.hierarchy_rebuilds, 1);
        assert!(
            uncached_second
                .timing
                .hierarchy_diagnostic_seconds
                .is_finite()
        );
        assert!(uncached_second.timing.hierarchy_diagnostic_seconds >= 0.0);
    }

    #[test]
    fn hierarchy_timing_accumulation_rejects_static_mismatch_before_mutation() {
        let matrices = vec![tridiagonal_matrix(4), tridiagonal_matrix(2)];
        let transfers = vec![GamgTransfer {
            fine_to_coarse: vec![0, 0, 1, 1],
            fine_entry_to_coarse_entry: Vec::new(),
        }];
        let mut accumulated = timing_with_hierarchy(
            &matrices,
            &transfers,
            GamgOptions {
                smoother: GamgSmoother::GaussSeidel,
                ..GamgOptions::default()
            },
        );
        accumulated.total_seconds = 11.0;
        accumulated.levels[1].coarsest_iterations = 7;
        let mut compatible = timing_with_hierarchy(
            &matrices,
            &transfers,
            GamgOptions {
                smoother: GamgSmoother::GaussSeidel,
                ..GamgOptions::default()
            },
        );
        compatible.total_seconds = 2.0;
        compatible.levels[1].coarsest_iterations = 5;
        accumulated
            .accumulate(&compatible)
            .expect("matching hierarchy must accumulate");
        assert_eq!(accumulated.total_seconds.to_bits(), 13.0f64.to_bits());
        assert_eq!(accumulated.levels[1].coarsest_iterations, 12);

        let mismatched = timing_with_hierarchy(
            &matrices,
            &transfers,
            GamgOptions {
                smoother: GamgSmoother::SymGaussSeidel,
                ..GamgOptions::default()
            },
        );
        let before_hierarchy = accumulated.hierarchy.clone();
        let before_total = accumulated.total_seconds;
        let before_coarsest_iterations = accumulated.levels[1].coarsest_iterations;

        let error = accumulated
            .accumulate(&mismatched)
            .expect_err("static hierarchy mismatch must fail closed");

        assert_eq!(
            error.to_string(),
            "GAMG profile static hierarchy diagnostics changed during accumulation"
        );
        assert_eq!(accumulated.hierarchy, before_hierarchy);
        assert_eq!(accumulated.total_seconds.to_bits(), before_total.to_bits());
        assert_eq!(
            accumulated.levels[1].coarsest_iterations,
            before_coarsest_iterations
        );

        let mut late_level_mismatch = compatible.clone();
        late_level_mismatch.hierarchy = accumulated.hierarchy.clone();
        late_level_mismatch.levels[0].smoothing_calls = 99;
        late_level_mismatch.levels[1].cells += 1;
        let before_debug = format!("{accumulated:?}");
        let before_levels_pointer = accumulated.levels.as_ptr();
        let before_hierarchy_pointer = accumulated.hierarchy.as_ref().map(std::sync::Arc::as_ptr);

        let error = accumulated
            .accumulate(&late_level_mismatch)
            .expect_err("later-level metadata mismatch must fail before any accumulation");

        assert_eq!(
            error.to_string(),
            "GAMG profile hierarchy changed at level 1: expected cells=2 nonzeros=4, got level=1 cells=3 nonzeros=4"
        );
        assert_eq!(format!("{accumulated:?}"), before_debug);
        assert_eq!(accumulated.levels.as_ptr(), before_levels_pointer);
        assert_eq!(
            accumulated.hierarchy.as_ref().map(std::sync::Arc::as_ptr),
            before_hierarchy_pointer,
        );
    }

    #[test]
    fn hierarchy_complexity_and_weighted_visit_helpers_fail_closed() {
        let empty = super::GamgHierarchyDiagnostics {
            levels: Vec::new(),
            transfers: Vec::new(),
            smoother_passes_per_sweep: 1,
            direct_solve_coarsest: false,
        };
        assert_eq!(empty.grid_complexity_terms(), None);
        assert_eq!(empty.operator_complexity_terms(), None);

        let zero_finest = super::GamgHierarchyDiagnostics {
            levels: vec![super::GamgHierarchyLevelDiagnostics {
                level: 0,
                cells: 0,
                nonzeros: 0,
            }],
            transfers: Vec::new(),
            smoother_passes_per_sweep: 1,
            direct_solve_coarsest: false,
        };
        assert_eq!(zero_finest.grid_complexity_terms(), None);
        assert_eq!(zero_finest.operator_complexity_terms(), None);

        let mut mismatched = GamgKernelTiming {
            levels: vec![super::GamgLevelTiming {
                level: 0,
                cells: 1,
                nonzeros: 1,
                smoothing_sweeps: 1,
                ..super::GamgLevelTiming::default()
            }],
            hierarchy: Some(std::sync::Arc::new(super::GamgHierarchyDiagnostics {
                levels: Vec::new(),
                transfers: Vec::new(),
                smoother_passes_per_sweep: 1,
                direct_solve_coarsest: false,
            })),
            ..GamgKernelTiming::default()
        };
        assert_eq!(mismatched.nnz_weighted_smoothing_work(), None);
        assert_eq!(mismatched.nnz_weighted_sparse_work(), None);

        mismatched.hierarchy = Some(std::sync::Arc::new(super::GamgHierarchyDiagnostics {
            levels: vec![super::GamgHierarchyLevelDiagnostics {
                level: 0,
                cells: 2,
                nonzeros: 1,
            }],
            transfers: Vec::new(),
            smoother_passes_per_sweep: 1,
            direct_solve_coarsest: false,
        }));
        assert_eq!(mismatched.nnz_weighted_smoothing_work(), None);
        assert_eq!(mismatched.nnz_weighted_sparse_work(), None);

        mismatched.hierarchy = Some(std::sync::Arc::new(super::GamgHierarchyDiagnostics {
            levels: vec![super::GamgHierarchyLevelDiagnostics {
                level: 0,
                cells: 1,
                nonzeros: usize::MAX,
            }],
            transfers: Vec::new(),
            smoother_passes_per_sweep: 2,
            direct_solve_coarsest: false,
        }));
        mismatched.levels[0].nonzeros = usize::MAX;
        mismatched.levels[0].smoothing_sweeps = usize::MAX;
        assert_eq!(mismatched.nnz_weighted_smoothing_work(), None);
        assert_eq!(mismatched.nnz_weighted_sparse_work(), None);
    }

    fn timing_with_hierarchy(
        matrices: &[CsrMatrix],
        transfers: &[GamgTransfer],
        options: GamgOptions,
    ) -> GamgKernelTiming {
        let hierarchy = std::sync::Arc::new(
            super::build_hierarchy_diagnostics(matrices, transfers, options)
                .expect("test hierarchy diagnostics"),
        );
        GamgKernelTiming::from_hierarchy(matrices, hierarchy)
    }

    fn poisson_grid(nx: usize, ny: usize, scale: f64) -> CsrMatrix {
        let mut rows = Vec::with_capacity(nx * ny);
        for y in 0..ny {
            for x in 0..nx {
                let row = y * nx + x;
                let mut entries = vec![(row, 4.0 * scale)];
                if x > 0 {
                    entries.push((row - 1, -scale));
                }
                if x + 1 < nx {
                    entries.push((row + 1, -scale));
                }
                if y > 0 {
                    entries.push((row - nx, -scale));
                }
                if y + 1 < ny {
                    entries.push((row + nx, -scale));
                }
                entries.sort_by_key(|(column, _)| *column);
                rows.push(entries);
            }
        }
        CsrMatrix::from_rows(rows, nx * ny).expect("Poisson grid")
    }

    fn diagonal_matrix(n: usize) -> CsrMatrix {
        CsrMatrix::from_rows((0..n).map(|row| vec![(row, 1.0)]).collect(), n)
            .expect("diagonal matrix")
    }

    fn tridiagonal_matrix(n: usize) -> CsrMatrix {
        let rows = (0..n)
            .map(|row| {
                let mut entries = Vec::with_capacity(3);
                if row > 0 {
                    entries.push((row - 1, -1.0));
                }
                entries.push((row, 3.0));
                if row + 1 < n {
                    entries.push((row + 1, -1.0));
                }
                entries
            })
            .collect();
        CsrMatrix::from_rows(rows, n).expect("tridiagonal matrix")
    }

    fn grid_face_weights(nx: usize, ny: usize) -> Vec<GamgFacePairWeight> {
        let mut weights = Vec::new();
        for y in 0..ny {
            for x in 0..nx {
                let cell = y * nx + x;
                if x + 1 < nx {
                    weights.push(
                        GamgFacePairWeight::new(cell, cell + 1, 1.0)
                            .expect("horizontal face weight"),
                    );
                }
                if y + 1 < ny {
                    weights.push(
                        GamgFacePairWeight::new(cell, cell + nx, 1.01)
                            .expect("vertical face weight"),
                    );
                }
            }
        }
        weights
    }

    fn assert_close(actual: &[f64], expected: &[f64], tolerance: f64) {
        assert_eq!(actual.len(), expected.len());
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= tolerance,
                "entry {index}: actual={actual} expected={expected} tolerance={tolerance}"
            );
        }
    }
}
