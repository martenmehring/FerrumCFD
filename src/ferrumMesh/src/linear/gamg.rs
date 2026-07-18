use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::time::Instant;

use super::{
    CgPreconditioner, CsrMatrix, CsrSparsityPattern, IterativeSolveReport,
    PreconditionedConjugateGradientOptions, PreconditionedConjugateGradientWorkspace, dot,
    gauss_seidel_sweep_with_cached_diagonal, invalid_input, l2_norm,
    validate_iterative_solve_input,
};
use crate::Result;

const MAX_LEVELS: usize = 50;
const COARSEST_MAX_ITERATIONS: usize = 1_000;
const SCALE_STABILISER: f64 = 1.0e-300;

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

    fn accumulate(&mut self, other: Self) -> Result<()> {
        if self.level != other.level || self.cells != other.cells || self.nonzeros != other.nonzeros
        {
            return Err(invalid_input(format!(
                "GAMG profile hierarchy changed at level {}: expected cells={} nonzeros={}, got level={} cells={} nonzeros={}",
                self.level, self.cells, self.nonzeros, other.level, other.cells, other.nonzeros
            )));
        }
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
        Ok(())
    }
}

#[derive(Clone, Debug, Default)]
pub struct GamgKernelTiming {
    pub total_seconds: f64,
    pub hierarchy_build_seconds: f64,
    pub hierarchy_rebuild_seconds: f64,
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
    pub levels: Vec<GamgLevelTiming>,
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

    pub fn accumulate(&mut self, other: &Self) -> Result<()> {
        if self.levels.is_empty() {
            self.levels = other.levels.clone();
        } else if self.levels.len() != other.levels.len() {
            return Err(invalid_input(format!(
                "GAMG profile hierarchy changed from {} to {} levels",
                self.levels.len(),
                other.levels.len()
            )));
        } else {
            for (level, other_level) in self.levels.iter_mut().zip(&other.levels) {
                level.accumulate(*other_level)?;
            }
        }
        self.total_seconds += other.total_seconds;
        self.hierarchy_build_seconds += other.hierarchy_build_seconds;
        self.hierarchy_rebuild_seconds += other.hierarchy_rebuild_seconds;
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

pub struct GamgWorkspace {
    options: GamgOptions,
    agglomeration_source: GamgAgglomerationSource,
    finest_sparsity: CsrSparsityPattern,
    matrices: Vec<CsrMatrix>,
    transfers: Vec<GamgTransfer>,
    diagonal_slots: Vec<Vec<usize>>,
    corrections: Vec<Vec<f64>>,
    sources: Vec<Vec<f64>>,
    residuals: Vec<Vec<f64>>,
    products: Vec<Vec<f64>>,
    pre_smoothed: Vec<Vec<f64>>,
    coarsest_pcg: Option<PreconditionedConjugateGradientWorkspace>,
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
        if options.interpolate_correction {
            return Err(invalid_input(
                "GAMG interpolateCorrection=true is not implemented by the matrix foundation; no injection fallback was applied"
                    .to_string(),
            ));
        }

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

        let diagonal_slots = matrices
            .iter()
            .map(super::csr_diagonal_slots)
            .collect::<Result<Vec<_>>>()?;
        let level_sizes = matrices.iter().map(CsrMatrix::rows).collect::<Vec<_>>();
        let corrections = level_vectors(&level_sizes);
        let sources = level_vectors(&level_sizes);
        let residuals = level_vectors(&level_sizes);
        let products = level_vectors(&level_sizes);
        let pre_smoothed = level_vectors(&level_sizes);
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
            corrections,
            sources,
            residuals,
            products,
            pre_smoothed,
            coarsest_pcg,
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
        self.solve_with_controls_internal::<false>(matrix, rhs, initial, controls, &mut timing)
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
        let report =
            self.solve_with_controls_internal::<true>(matrix, rhs, initial, controls, &mut timing)?;
        timing.total_seconds = started.elapsed().as_secs_f64();
        let accounted_seconds = timing.hierarchy_rebuild_seconds
            + timing.matrix_refresh_seconds
            + timing.finest_residual_seconds
            + timing.v_cycle_seconds;
        timing.other_seconds = (timing.total_seconds - accounted_seconds).max(0.0);
        Ok(ProfiledGamgSolveReport { report, timing })
    }

    fn solve_with_controls_internal<const PROFILE: bool>(
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
            *timing = GamgKernelTiming::from_matrices(&self.matrices);
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
            });
        }

        let iteration_limit = controls.max_iterations.max(controls.min_iterations).max(1);
        for iteration in 1..=iteration_limit {
            let cycle_started = profile_started::<PROFILE>();
            self.v_cycle::<PROFILE>(&mut solution, rhs, controls, timing)?;
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
                });
            }
        }

        self.has_solved = true;
        Ok(IterativeSolveReport {
            solution,
            iterations: iteration_limit,
            residual_norm,
            converged: false,
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
        if PROFILE {
            timing.levels[0].matrix_refresh_seconds += profile_elapsed(finest_started);
            timing.levels[0].matrix_refreshes += 1;
        }
        for level in 0..self.transfers.len() {
            let level_started = profile_started::<PROFILE>();
            let (fine_levels, coarse_levels) = self.matrices.split_at_mut(level + 1);
            self.transfers[level]
                .agglomerate_values(fine_levels[level].values(), coarse_levels[0].values_mut())?;
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

    fn v_cycle<const PROFILE: bool>(
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
                smooth(
                    &self.matrices[level],
                    &self.diagonal_slots[level],
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
        self.solve_coarsest_level(coarsest, controls)?;
        if PROFILE {
            timing.levels[coarsest].coarsest_solve_seconds += profile_elapsed(coarsest_started);
            timing.levels[coarsest].coarsest_solves += 1;
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
            if PROFILE {
                timing.levels[level].prolongation_seconds += profile_elapsed(prolongation_started);
                timing.levels[level].prolongation_calls += 1;
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
            smooth(
                &self.matrices[level],
                &self.diagonal_slots[level],
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
        let result = smooth(
            &self.matrices[0],
            &self.diagonal_slots[0],
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

    fn solve_coarsest_level(&mut self, coarsest: usize, controls: GamgSolveControls) -> Result<()> {
        if self.options.direct_solve_coarsest {
            dense_lu_solve(
                &self.matrices[coarsest],
                &self.sources[coarsest],
                &mut self.corrections[coarsest],
            )
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
            self.corrections[coarsest].copy_from_slice(&report.solution);
            Ok(())
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

fn level_vectors(level_sizes: &[usize]) -> Vec<Vec<f64>> {
    level_sizes.iter().map(|&size| vec![0.0; size]).collect()
}

fn has_converged(residual: f64, initial: f64, controls: GamgSolveControls) -> bool {
    residual <= controls.tolerance
        || (controls.relative_tolerance > 0.0 && residual <= controls.relative_tolerance * initial)
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

fn smooth(
    matrix: &CsrMatrix,
    diagonal_slots: &[usize],
    rhs: &[f64],
    solution: &mut [f64],
    smoother: GamgSmoother,
    sweeps: usize,
) -> Result<()> {
    for _ in 0..sweeps {
        gauss_seidel_sweep_with_cached_diagonal(
            matrix,
            diagonal_slots,
            rhs,
            solution,
            0..matrix.rows(),
        )?;
        if smoother == GamgSmoother::SymGaussSeidel {
            gauss_seidel_sweep_with_cached_diagonal(
                matrix,
                diagonal_slots,
                rhs,
                solution,
                (0..matrix.rows()).rev(),
            )?;
        }
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

fn pair_map_from_edges(
    n_cells: usize,
    edges: &[PairEdge],
    forward: bool,
) -> Result<(Vec<usize>, usize)> {
    let mut cell_edges = vec![Vec::<usize>::new(); n_cells];
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
    }

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
        let mut max_weight = f64::NEG_INFINITY;
        for &edge_index in &cell_edges[cell] {
            let edge = edges[edge_index];
            if coarse_map[edge.lower] == usize::MAX
                && coarse_map[edge.upper] == usize::MAX
                && edge.weight > max_weight
            {
                match_edge = Some(edge_index);
                max_weight = edge.weight;
            }
        }

        if let Some(edge_index) = match_edge {
            let edge = edges[edge_index];
            coarse_map[edge.lower] = n_coarse;
            coarse_map[edge.upper] = n_coarse;
            n_coarse += 1;
        } else {
            let mut cluster_edge = None;
            let mut cluster_weight = f64::NEG_INFINITY;
            for &edge_index in &cell_edges[cell] {
                let edge = edges[edge_index];
                if edge.weight > cluster_weight {
                    cluster_edge = Some(edge_index);
                    cluster_weight = edge.weight;
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
    let mut dense = vec![0.0; n * n];
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

#[cfg(test)]
mod tests {
    use super::{
        GamgAgglomerator, GamgFacePairWeight, GamgOptions, GamgSmoother, GamgWorkspace, PairEdge,
        algebraic_pair_map, gamg_solve, pair_map_from_edges,
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
        assert_eq!(options.agglomerator, GamgAgglomerator::AlgebraicPair);
        assert_eq!(options.smoother, GamgSmoother::GaussSeidel);
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
        let second = poisson_grid(20, 16, 1.75);
        let options = GamgOptions {
            max_iterations: 80,
            tolerance: 1.0e-10,
            direct_solve_coarsest: true,
            ..GamgOptions::default()
        };
        let mut workspace = GamgWorkspace::new(&first, options).expect("GAMG workspace");
        let level_sizes = workspace.level_sizes();
        let correction_pointers = workspace
            .corrections
            .iter()
            .map(|level| level.as_ptr())
            .collect::<Vec<_>>();
        let expected = vec![1.0; first.rows()];

        let first_rhs = first.matvec(&expected).expect("first rhs");
        let first_report = workspace
            .solve(&first, &first_rhs, None)
            .expect("first GAMG solve");
        let second_rhs = second.matvec(&expected).expect("second rhs");
        let second_report = workspace
            .solve(&second, &second_rhs, None)
            .expect("second GAMG solve");

        assert!(first_report.converged);
        assert!(second_report.converged);
        assert_eq!(workspace.level_sizes(), level_sizes);
        assert_eq!(
            workspace
                .corrections
                .iter()
                .map(|level| level.as_ptr())
                .collect::<Vec<_>>(),
            correction_pointers
        );
        assert_close(&second_report.solution, &expected, 1.0e-8);
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
