use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::time::Instant;

#[cfg(test)]
use std::cell::RefCell;

use super::{
    CgPreconditioner, CsrMatrix, CsrSparsityPattern, IterativeSolveReport,
    IterativeSolveTermination, PreconditionedConjugateGradientOptions,
    PreconditionedConjugateGradientWorkspace, dot, gauss_seidel_sweep_with_cached_diagonal,
    invalid_input, l2_norm, validate_iterative_solve_input,
};
use crate::Result;

const MAX_LEVELS: usize = 50;
const COARSEST_MAX_ITERATIONS: usize = 1_000;
const SCALE_STABILISER: f64 = 1.0e-300;
/// Caps dense coarsest storage at 512 KiB and its cubic factorisation at
/// roughly 17 million elimination steps, keeping both costs predictable.
const MAX_DENSE_COARSEST_CELLS: usize = 256;

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum B2PublicCallKind {
    L2Plain,
    L2Profiled,
    NormalizedL1Plain,
    NormalizedL1Profiled,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum B2TraceEvent {
    PublicCall(B2PublicCallKind),
    HierarchyRebuild,
    LduBuild { level: usize },
    CsrRefresh { level: usize },
    DiagonalRefresh { level: usize },
    LduRefresh { level: usize },
    LduSweep { level: usize },
    CsrSweep { level: usize, symmetric: bool },
    CoarseDirect { level: usize },
    CoarsePcg { level: usize },
}

#[cfg(test)]
std::thread_local! {
    static B2_TRACE: RefCell<Option<Vec<B2TraceEvent>>> = const { RefCell::new(None) };
}

#[cfg(test)]
#[derive(Debug)]
struct B2TraceGuard;

#[cfg(test)]
impl B2TraceGuard {
    fn new() -> Self {
        B2_TRACE.with(|trace| {
            let mut trace = trace.borrow_mut();
            assert!(trace.is_none(), "B2 trace guards must not be nested");
            *trace = Some(Vec::new());
        });
        Self
    }

    fn events(&self) -> Vec<B2TraceEvent> {
        B2_TRACE.with(|trace| {
            trace
                .borrow()
                .as_ref()
                .expect("B2 trace guard owns an active trace")
                .clone()
        })
    }
}

#[cfg(test)]
impl Drop for B2TraceGuard {
    fn drop(&mut self) {
        B2_TRACE.with(|trace| {
            trace.borrow_mut().take();
        });
    }
}

#[cfg(test)]
fn b2_trace(event: B2TraceEvent) {
    B2_TRACE.with(|trace| {
        if let Some(events) = trace.borrow_mut().as_mut() {
            events.push(event);
        }
    });
}

#[cfg(test)]
fn b2_trace_sweeps(level: usize, ldu: bool, symmetric: bool, sweeps: usize) {
    for _ in 0..sweeps {
        b2_trace(if ldu {
            B2TraceEvent::LduSweep { level }
        } else {
            B2TraceEvent::CsrSweep { level, symmetric }
        });
    }
}

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
    diagonal_values: Vec<Vec<f64>>,
    ldu_levels: Option<Vec<GamgLduLevel>>,
    corrections: Vec<Vec<f64>>,
    sources: Vec<Vec<f64>>,
    residuals: Vec<Vec<f64>>,
    products: Vec<Vec<f64>>,
    pre_smoothed: Vec<Vec<f64>>,
    coarsest_pcg: Option<PreconditionedConjugateGradientWorkspace>,
    has_solved: bool,
}

#[derive(Debug)]
struct GamgLduLevel {
    lower_addr: Vec<usize>,
    upper_addr: Vec<usize>,
    lower_csr: Vec<Option<usize>>,
    upper_csr: Vec<Option<usize>>,
    lower: Vec<f64>,
    upper: Vec<f64>,
    owner_start: Vec<usize>,
    b_prime: Vec<f64>,
}

impl GamgLduLevel {
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
        let mut result = Self {
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
                result.lower_addr.push(owner);
                result.upper_addr.push(neighbour);
                result.lower_csr.push(lower.get(occurrence).copied());
                result.upper_csr.push(upper.get(occurrence).copied());
            }
        }
        for &owner in &result.lower_addr {
            result.owner_start[owner + 1] += 1;
        }
        for cell in 0..matrix.rows() {
            result.owner_start[cell + 1] += result.owner_start[cell];
        }
        result.refresh(matrix);
        result
    }

    fn refresh(&mut self, matrix: &CsrMatrix) {
        for face in 0..self.lower.len() {
            self.lower[face] = self.lower_csr[face].map_or(0.0, |slot| matrix.values()[slot]);
            self.upper[face] = self.upper_csr[face].map_or(0.0, |slot| matrix.values()[slot]);
        }
    }
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
        let coarsest_pcg = if options.direct_solve_coarsest {
            None
        } else {
            Some(PreconditionedConjugateGradientWorkspace::new(
                matrices.last().expect("GAMG has a coarsest matrix"),
                CgPreconditioner::IncompleteCholesky,
            )?)
        };
        let ldu_levels = if options.smoother == GamgSmoother::SymGaussSeidel {
            #[cfg(test)]
            let levels = matrices
                .iter()
                .enumerate()
                .map(|(level, matrix)| {
                    b2_trace(B2TraceEvent::LduBuild { level });
                    GamgLduLevel::new(matrix)
                })
                .collect();
            #[cfg(not(test))]
            let levels = matrices.iter().map(GamgLduLevel::new).collect();
            Some(levels)
        } else {
            None
        };

        Ok(Self {
            options,
            agglomeration_source,
            finest_sparsity,
            matrices,
            transfers,
            diagonal_slots,
            diagonal_values,
            ldu_levels,
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
        #[cfg(test)]
        b2_trace(B2TraceEvent::PublicCall(B2PublicCallKind::L2Plain));
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
        #[cfg(test)]
        b2_trace(B2TraceEvent::PublicCall(B2PublicCallKind::L2Profiled));
        let started = Instant::now();
        let mut timing = GamgKernelTiming::default();
        let report = self.solve_with_controls_internal::<true, false>(
            matrix,
            rhs,
            initial,
            controls,
            &mut timing,
        )?;
        timing.total_seconds = started.elapsed().as_secs_f64();
        let accounted_seconds = timing.hierarchy_rebuild_seconds
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
        #[cfg(test)]
        b2_trace(B2TraceEvent::PublicCall(
            B2PublicCallKind::NormalizedL1Plain,
        ));
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
        #[cfg(test)]
        b2_trace(B2TraceEvent::PublicCall(
            B2PublicCallKind::NormalizedL1Profiled,
        ));
        let started = Instant::now();
        let mut timing = GamgKernelTiming::default();
        let report = self.solve_normalized_l1_with_controls_internal::<true, false>(
            matrix,
            rhs,
            initial,
            controls,
            &mut timing,
        )?;
        timing.total_seconds = started.elapsed().as_secs_f64();
        let accounted_seconds = timing.hierarchy_rebuild_seconds
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
            #[cfg(test)]
            b2_trace(B2TraceEvent::HierarchyRebuild);
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
            *timing = GamgKernelTiming::from_matrices(&self.matrices);
            timing.hierarchy_rebuild_seconds = hierarchy_rebuild_seconds;
            timing.hierarchy_rebuilds = hierarchy_rebuilds;
            timing.solves = 1;
        }

        let refresh_started = profile_started::<PROFILE>();
        self.refresh_matrix_values::<PROFILE, SCANNED_DIAGONAL>(matrix, timing)?;
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
            #[cfg(test)]
            b2_trace(B2TraceEvent::HierarchyRebuild);
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
        self.refresh_matrix_values::<PROFILE, SCANNED_DIAGONAL>(matrix, timing)?;
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

    fn refresh_matrix_values<const PROFILE: bool, const SCANNED_DIAGONAL: bool>(
        &mut self,
        matrix: &CsrMatrix,
        timing: &mut GamgKernelTiming,
    ) -> Result<()> {
        if !SCANNED_DIAGONAL && self.options.smoother == GamgSmoother::SymGaussSeidel {
            let levels = self.ldu_levels.as_ref().ok_or_else(|| {
                invalid_input("GAMG symGaussSeidel LDU hierarchy is missing".to_string())
            })?;
            if levels.len() != self.matrices.len() {
                return Err(invalid_input(format!(
                    "GAMG symGaussSeidel LDU hierarchy has {} levels, expected {}",
                    levels.len(),
                    self.matrices.len()
                )));
            }
        }
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
        #[cfg(test)]
        b2_trace(B2TraceEvent::CsrRefresh { level: 0 });
        refresh_diagonal_values(
            &self.matrices[0],
            &self.diagonal_slots[0],
            &mut self.diagonal_values[0],
        )?;
        #[cfg(test)]
        b2_trace(B2TraceEvent::DiagonalRefresh { level: 0 });
        if PROFILE {
            timing.levels[0].matrix_refresh_seconds += profile_elapsed(finest_started);
            timing.levels[0].matrix_refreshes += 1;
        }
        for level in 0..self.transfers.len() {
            let level_started = profile_started::<PROFILE>();
            let (fine_levels, coarse_levels) = self.matrices.split_at_mut(level + 1);
            self.transfers[level]
                .agglomerate_values(fine_levels[level].values(), coarse_levels[0].values_mut())?;
            #[cfg(test)]
            b2_trace(B2TraceEvent::CsrRefresh { level: level + 1 });
            refresh_diagonal_values(
                &self.matrices[level + 1],
                &self.diagonal_slots[level + 1],
                &mut self.diagonal_values[level + 1],
            )?;
            #[cfg(test)]
            b2_trace(B2TraceEvent::DiagonalRefresh { level: level + 1 });
            if PROFILE {
                timing.levels[level + 1].matrix_refresh_seconds += profile_elapsed(level_started);
                timing.levels[level + 1].matrix_refreshes += 1;
            }
        }
        if !SCANNED_DIAGONAL && self.options.smoother == GamgSmoother::SymGaussSeidel {
            let levels = self
                .ldu_levels
                .as_mut()
                .expect("validated GAMG SymGS LDU hierarchy");
            let mut level = 0;
            while level < self.matrices.len() {
                let level_started = profile_started::<PROFILE>();
                levels[level].refresh(&self.matrices[level]);
                #[cfg(test)]
                b2_trace(B2TraceEvent::LduRefresh { level });
                if PROFILE {
                    timing.levels[level].matrix_refresh_seconds += profile_elapsed(level_started);
                }
                level += 1;
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
                #[cfg(test)]
                b2_trace_sweeps(
                    level,
                    !SCANNED_DIAGONAL && self.ldu_levels.is_some(),
                    self.options.smoother == GamgSmoother::SymGaussSeidel,
                    pre_sweeps,
                );
                smooth::<SCANNED_DIAGONAL>(
                    &self.matrices[level],
                    (&self.diagonal_slots[level], &self.diagonal_values[level]),
                    if SCANNED_DIAGONAL {
                        None
                    } else {
                        self.ldu_levels
                            .as_mut()
                            .and_then(|levels| levels.get_mut(level))
                    },
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
            #[cfg(test)]
            b2_trace_sweeps(
                level,
                !SCANNED_DIAGONAL && self.ldu_levels.is_some(),
                self.options.smoother == GamgSmoother::SymGaussSeidel,
                post_sweeps,
            );
            smooth::<SCANNED_DIAGONAL>(
                &self.matrices[level],
                (&self.diagonal_slots[level], &self.diagonal_values[level]),
                if SCANNED_DIAGONAL {
                    None
                } else {
                    self.ldu_levels
                        .as_mut()
                        .and_then(|levels| levels.get_mut(level))
                },
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
        #[cfg(test)]
        b2_trace_sweeps(
            0,
            !SCANNED_DIAGONAL && self.ldu_levels.is_some(),
            self.options.smoother == GamgSmoother::SymGaussSeidel,
            self.options.n_finest_sweeps,
        );
        let result = smooth::<SCANNED_DIAGONAL>(
            &self.matrices[0],
            (&self.diagonal_slots[0], &self.diagonal_values[0]),
            if SCANNED_DIAGONAL {
                None
            } else {
                self.ldu_levels
                    .as_mut()
                    .and_then(|levels| levels.get_mut(0))
            },
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
            let result = dense_lu_solve(
                &self.matrices[coarsest],
                &self.sources[coarsest],
                &mut self.corrections[coarsest],
            );
            #[cfg(test)]
            if result.is_ok() {
                b2_trace(B2TraceEvent::CoarseDirect { level: coarsest });
            }
            result
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
            #[cfg(test)]
            b2_trace(B2TraceEvent::CoarsePcg { level: coarsest });
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
        || (controls.relative_tolerance > 0.0
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
    diagonal: (&[usize], &[f64]),
    ldu_level: Option<&mut GamgLduLevel>,
    rhs: &[f64],
    solution: &mut [f64],
    smoother: GamgSmoother,
    sweeps: usize,
) -> Result<()> {
    let (diagonal_slots, diagonal_values) = diagonal;
    if !SCANNED_DIAGONAL && smoother == GamgSmoother::SymGaussSeidel {
        let level = ldu_level.ok_or_else(|| {
            invalid_input("GAMG symGaussSeidel LDU hierarchy is missing".to_string())
        })?;
        for _ in 0..sweeps {
            ldu_sym_gauss_seidel_sweep_unchecked(level, diagonal_values, rhs, solution)?;
        }
        return Ok(());
    }
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
fn ldu_sym_gauss_seidel_sweep(
    level: &mut GamgLduLevel,
    diagonal: &[f64],
    rhs: &[f64],
    solution: &mut [f64],
) -> Result<()> {
    validate_ldu_sweep_inputs(level, diagonal, rhs, solution)?;
    ldu_sym_gauss_seidel_sweep_unchecked(level, diagonal, rhs, solution)
}

fn ldu_sym_gauss_seidel_sweep_unchecked(
    level: &mut GamgLduLevel,
    diagonal: &[f64],
    rhs: &[f64],
    solution: &mut [f64],
) -> Result<()> {
    level.b_prime.copy_from_slice(rhs);
    ldu_sym_gauss_seidel_half_unchecked(level, diagonal, solution, 0..solution.len())?;
    ldu_sym_gauss_seidel_half_unchecked(level, diagonal, solution, (0..solution.len()).rev())
}

#[cfg(test)]
fn ldu_sym_gauss_seidel_half(
    level: &mut GamgLduLevel,
    diagonal: &[f64],
    solution: &mut [f64],
    cells: impl Iterator<Item = usize>,
) -> Result<()> {
    validate_ldu_half_inputs(level, diagonal, solution)?;
    ldu_sym_gauss_seidel_half_unchecked(level, diagonal, solution, cells)
}

fn ldu_sym_gauss_seidel_half_unchecked(
    level: &mut GamgLduLevel,
    diagonal: &[f64],
    solution: &mut [f64],
    cells: impl Iterator<Item = usize>,
) -> Result<()> {
    for cell in cells {
        let mut psii = level.b_prime[cell];
        for face in level.owner_start[cell]..level.owner_start[cell + 1] {
            psii -= level.upper[face] * solution[level.upper_addr[face]];
        }
        psii /= diagonal[cell];
        if !psii.is_finite() {
            return Err(invalid_input(format!(
                "Gauss-Seidel update for row {cell} is not finite"
            )));
        }
        for face in level.owner_start[cell]..level.owner_start[cell + 1] {
            level.b_prime[level.upper_addr[face]] -= level.lower[face] * psii;
        }
        solution[cell] = psii;
    }
    Ok(())
}

#[cfg(test)]
fn validate_ldu_sweep_inputs(
    level: &GamgLduLevel,
    diagonal: &[f64],
    rhs: &[f64],
    solution: &[f64],
) -> Result<()> {
    validate_ldu_dimensions_and_coefficients(level, diagonal, solution)?;
    let rows = level.b_prime.len();
    if rhs.len() != rows {
        return Err(invalid_input(format!(
            "GAMG symGaussSeidel LDU expected {rows} rhs entries, got {}",
            rhs.len()
        )));
    }
    if let Some((row, value)) = rhs
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(invalid_input(format!(
            "GAMG symGaussSeidel LDU rhs entry {row} must be finite, got {value}"
        )));
    }
    Ok(())
}

#[cfg(test)]
fn validate_ldu_half_inputs(
    level: &GamgLduLevel,
    diagonal: &[f64],
    solution: &[f64],
) -> Result<()> {
    validate_ldu_dimensions_and_coefficients(level, diagonal, solution)?;
    if let Some((row, value)) = level
        .b_prime
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(invalid_input(format!(
            "GAMG symGaussSeidel LDU bPrime entry {row} must be finite, got {value}"
        )));
    }
    Ok(())
}

#[cfg(test)]
fn validate_ldu_dimensions_and_coefficients(
    level: &GamgLduLevel,
    diagonal: &[f64],
    solution: &[f64],
) -> Result<()> {
    let rows = level.b_prime.len();
    if diagonal.len() != rows {
        return Err(invalid_input(format!(
            "GAMG symGaussSeidel LDU expected {rows} diagonal entries, got {}",
            diagonal.len()
        )));
    }
    if solution.len() != rows {
        return Err(invalid_input(format!(
            "GAMG symGaussSeidel LDU expected {rows} solution entries, got {}",
            solution.len()
        )));
    }
    let expected_owner_start = rows.checked_add(1).ok_or_else(|| {
        invalid_input("GAMG symGaussSeidel LDU ownerStart length overflow".to_string())
    })?;
    if level.owner_start.len() != expected_owner_start {
        return Err(invalid_input(format!(
            "GAMG symGaussSeidel LDU expected {expected_owner_start} ownerStart entries, got {}",
            level.owner_start.len()
        )));
    }
    let faces = level.lower.len();
    for (name, len) in [
        ("lowerAddr", level.lower_addr.len()),
        ("upperAddr", level.upper_addr.len()),
        ("lowerCsr", level.lower_csr.len()),
        ("upperCsr", level.upper_csr.len()),
        ("upper", level.upper.len()),
    ] {
        if len != faces {
            return Err(invalid_input(format!(
                "GAMG symGaussSeidel LDU expected {faces} {name} entries, got {len}"
            )));
        }
    }
    if level.owner_start.first().copied() != Some(0) {
        return Err(invalid_input(
            "GAMG symGaussSeidel LDU ownerStart must begin at 0".to_string(),
        ));
    }
    if level.owner_start.last().copied() != Some(faces) {
        return Err(invalid_input(format!(
            "GAMG symGaussSeidel LDU ownerStart must end at {faces}, got {}",
            level.owner_start.last().copied().unwrap_or_default()
        )));
    }
    for row in 0..rows {
        let start = level.owner_start[row];
        let end = level.owner_start[row + 1];
        if start > end || end > faces {
            return Err(invalid_input(format!(
                "GAMG symGaussSeidel LDU ownerStart row {row} has invalid range {start}..{end} for {faces} faces"
            )));
        }
        for face in start..end {
            if level.lower_addr[face] != row {
                return Err(invalid_input(format!(
                    "GAMG symGaussSeidel LDU face {face} has owner {}, expected {row}",
                    level.lower_addr[face]
                )));
            }
        }
    }
    for face in 0..faces {
        let owner = level.lower_addr[face];
        let neighbour = level.upper_addr[face];
        if owner >= rows || neighbour >= rows || owner >= neighbour {
            return Err(invalid_input(format!(
                "GAMG symGaussSeidel LDU face {face} has invalid owner/neighbour {owner}/{neighbour} for {rows} rows"
            )));
        }
        for (name, value) in [("lower", level.lower[face]), ("upper", level.upper[face])] {
            if !value.is_finite() {
                return Err(invalid_input(format!(
                    "GAMG symGaussSeidel LDU {name} coefficient {face} must be finite, got {value}"
                )));
            }
        }
    }
    for (row, value) in diagonal.iter().copied().enumerate() {
        if !value.is_finite() || value == 0.0 {
            return Err(invalid_input(format!(
                "GAMG symGaussSeidel LDU diagonal row {row} must be finite and non-zero, got {value}"
            )));
        }
    }
    if let Some((row, value)) = solution
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(invalid_input(format!(
            "GAMG symGaussSeidel LDU solution entry {row} must be finite, got {value}"
        )));
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
    use super::{
        GamgAgglomerator, GamgFacePairWeight, GamgKernelTiming, GamgOptions, GamgSmoother,
        GamgWorkspace, MAX_DENSE_COARSEST_CELLS, NormalizedL1GamgSolveControls, PairEdge,
        algebraic_pair_map, checked_dense_storage_len, dense_lu_solve, gamg_solve,
        pair_map_from_edges,
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
        struct OptionalUsizeVecSnap {
            len: usize,
            capacity: usize,
            values: Vec<Option<usize>>,
        }

        #[derive(Clone, Debug, PartialEq, Eq)]
        struct LduLevelValueSnap {
            lower_addr: UsizeVecSnap,
            upper_addr: UsizeVecSnap,
            lower_csr: OptionalUsizeVecSnap,
            upper_csr: OptionalUsizeVecSnap,
            lower: BitsVecSnap,
            upper: BitsVecSnap,
            owner_start: UsizeVecSnap,
            b_prime: BitsVecSnap,
            allocations: [(usize, usize, usize); 8],
        }

        #[derive(Clone, Debug, PartialEq, Eq)]
        struct LduLevelsValueSnap {
            len: usize,
            capacity: usize,
            allocations: (usize, usize, usize),
            levels: Vec<(usize, LduLevelValueSnap)>,
        }

        #[derive(Clone, Debug, PartialEq, Eq)]
        struct WorkspaceValueSnap {
            option_iterations: (usize, usize, u64, u64),
            option_hierarchy: (bool, usize, usize, GamgAgglomerator),
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
            diagonal_value_allocations: Vec<(usize, usize, usize)>,
            ldu_levels: Option<LduLevelsValueSnap>,
            corrections: NestedBitsSnap,
            sources: NestedBitsSnap,
            residuals: NestedBitsSnap,
            products: NestedBitsSnap,
            pre_smoothed: NestedBitsSnap,
            coarsest_pcg: Option<PcgValueSnap>,
            has_solved: bool,
        }

        #[derive(Clone, Debug, PartialEq, Eq)]
        struct LevelTimingValueSnap {
            metadata: (usize, usize, usize),
            seconds_bits: [u64; 8],
            counters: [usize; 9],
        }

        #[derive(Clone, Debug, PartialEq, Eq)]
        struct TimingValueSnap {
            seconds_bits: [u64; 7],
            counters: [usize; 6],
            levels_len: usize,
            levels_capacity: usize,
            levels: Vec<(usize, LevelTimingValueSnap)>,
        }

        #[derive(Clone, Debug)]
        struct FullSnapshot {
            workspace: WorkspaceValueSnap,
            timing: TimingValueSnap,
            initial: BitsVecSnap,
            usize_arcs: Vec<(String, std::sync::Arc<[usize]>)>,
            face_arcs: Vec<std::sync::Arc<[GamgFacePairWeight]>>,
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
        let optional_usize_snap = |values: &Vec<Option<usize>>| OptionalUsizeVecSnap {
            len: values.len(),
            capacity: values.capacity(),
            values: values.clone(),
        };
        let ldu_level_snap = |level: &super::GamgLduLevel| LduLevelValueSnap {
            lower_addr: usize_vec_snap(&level.lower_addr),
            upper_addr: usize_vec_snap(&level.upper_addr),
            lower_csr: optional_usize_snap(&level.lower_csr),
            upper_csr: optional_usize_snap(&level.upper_csr),
            lower: bits_vec_snap(&level.lower),
            upper: bits_vec_snap(&level.upper),
            owner_start: usize_vec_snap(&level.owner_start),
            b_prime: bits_vec_snap(&level.b_prime),
            allocations: [
                (
                    level.lower_addr.as_ptr() as usize,
                    level.lower_addr.len(),
                    level.lower_addr.capacity(),
                ),
                (
                    level.upper_addr.as_ptr() as usize,
                    level.upper_addr.len(),
                    level.upper_addr.capacity(),
                ),
                (
                    level.lower_csr.as_ptr() as usize,
                    level.lower_csr.len(),
                    level.lower_csr.capacity(),
                ),
                (
                    level.upper_csr.as_ptr() as usize,
                    level.upper_csr.len(),
                    level.upper_csr.capacity(),
                ),
                (
                    level.lower.as_ptr() as usize,
                    level.lower.len(),
                    level.lower.capacity(),
                ),
                (
                    level.upper.as_ptr() as usize,
                    level.upper.len(),
                    level.upper.capacity(),
                ),
                (
                    level.owner_start.as_ptr() as usize,
                    level.owner_start.len(),
                    level.owner_start.capacity(),
                ),
                (
                    level.b_prime.as_ptr() as usize,
                    level.b_prime.len(),
                    level.b_prime.capacity(),
                ),
            ],
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
                            ],
                        },
                    )
                })
                .collect(),
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
            let ldu_levels = workspace
                .ldu_levels
                .as_ref()
                .map(|levels| LduLevelsValueSnap {
                    len: levels.len(),
                    capacity: levels.capacity(),
                    allocations: (levels.as_ptr() as usize, levels.len(), levels.capacity()),
                    levels: levels
                        .iter()
                        .enumerate()
                        .map(|(index, level)| (index, ldu_level_snap(level)))
                        .collect(),
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
                    diagonal_value_allocations: workspace
                        .diagonal_values
                        .iter()
                        .map(|values| (values.as_ptr() as usize, values.len(), values.capacity()))
                        .collect(),
                    ldu_levels,
                    corrections: nested_bits_snap(&workspace.corrections),
                    sources: nested_bits_snap(&workspace.sources),
                    residuals: nested_bits_snap(&workspace.residuals),
                    products: nested_bits_snap(&workspace.products),
                    pre_smoothed: nested_bits_snap(&workspace.pre_smoothed),
                    coarsest_pcg,
                    has_solved: workspace.has_solved,
                },
                timing: timing_snap(timing),
                initial: bits_vec_snap(initial),
                usize_arcs,
                face_arcs,
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
        };

        let without_allocation_addresses = |mut workspace: WorkspaceValueSnap| {
            for allocation in &mut workspace.diagonal_value_allocations {
                allocation.0 = 0;
            }
            if let Some(levels) = &mut workspace.ldu_levels {
                levels.allocations.0 = 0;
                for (_, level) in &mut levels.levels {
                    for allocation in &mut level.allocations {
                        allocation.0 = 0;
                    }
                }
            }
            workspace
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
            smoother: GamgSmoother::SymGaussSeidel,
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
                },
            ],
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
                ],
                timing
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
                    .collect::<Vec<_>>(),
            )
        };
        let assert_finite_timing = |timing: &super::GamgKernelTiming| {
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
            assert_eq!(
                without_allocation_addresses(live_state.workspace.clone()),
                without_allocation_addresses(clean_state.workspace.clone())
            );
            assert_eq!(live_state.initial, clean_state.initial);
            assert_eq!(
                without_allocation_addresses(live_state.workspace),
                without_allocation_addresses(repeated_state.workspace)
            );
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
                assert_eq!(
                    logical_timing(live_timing_ref),
                    (
                        [0, 1, 1, 3, 1, 2],
                        vec![
                            (0, 4, 10, 1, 2, 2, 2, 4, 2, 0, 2, 0),
                            (1, 2, 4, 1, 0, 0, 0, 0, 0, 0, 0, 2),
                        ],
                    )
                );
                assert_finite_timing(live_timing_ref);
                assert_finite_timing(clean_timing_ref);
                assert_finite_timing(repeated_timing_ref);
            } else {
                assert_eq!(logical_timing(live_timing_ref), ([0; 6], vec![]));
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

    mod ldu_b2 {
        use std::collections::BTreeSet;

        use super::super::{
            B2PublicCallKind, B2TraceEvent, B2TraceGuard, GamgAgglomerator, GamgFacePairWeight,
            GamgKernelTiming, GamgLduLevel, GamgOptions, GamgSmoother, GamgSolveControls,
            GamgWorkspace, NormalizedL1GamgSolveControls, gauss_seidel_sweep_with_cached_diagonal,
            ldu_sym_gauss_seidel_half, ldu_sym_gauss_seidel_sweep, matrix_connection_pairs,
        };
        use crate::linear::{CsrMatrix, IterativeSolveReport, IterativeSolveTermination};

        fn cell(x: usize, y: usize, z: usize) -> usize {
            x + 2 * y + 4 * z
        }

        fn matrix(scale: f64) -> CsrMatrix {
            CsrMatrix::from_rows(
                (0..2)
                    .flat_map(|z| (0..2).flat_map(move |y| (0..2).map(move |x| (x, y, z))))
                    .map(|(x, y, z)| {
                        let row = cell(x, y, z);
                        let mut entries = vec![(row, 8.0 * scale)];
                        if x > 0 {
                            entries.push((cell(x - 1, y, z), -scale));
                        }
                        if x + 1 < 2 {
                            entries.push((cell(x + 1, y, z), -scale));
                        }
                        if y > 0 {
                            entries.push((cell(x, y - 1, z), -scale));
                        }
                        if y + 1 < 2 {
                            entries.push((cell(x, y + 1, z), -scale));
                        }
                        if z > 0 {
                            entries.push((cell(x, y, z - 1), -scale));
                        }
                        if z + 1 < 2 {
                            entries.push((cell(x, y, z + 1), -scale));
                        }
                        entries
                    })
                    .collect(),
                8,
            )
            .expect("B2 matrix")
        }

        fn chain_matrix(rows: usize, scale: f64) -> CsrMatrix {
            CsrMatrix::from_rows(
                (0..rows)
                    .map(|row| {
                        let mut entries = Vec::new();
                        if row > 0 {
                            entries.push((row - 1, -scale));
                        }
                        entries.push((row, 4.0 * scale));
                        if row + 1 < rows {
                            entries.push((row + 1, -scale));
                        }
                        entries
                    })
                    .collect(),
                rows,
            )
            .expect("B2 chain matrix")
        }

        fn main_fixture() -> CsrMatrix {
            CsrMatrix::from_rows(
                vec![
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
                ],
                3,
            )
            .expect("B2 main fixture")
        }

        fn reordered_main_fixture() -> CsrMatrix {
            CsrMatrix::from_rows(
                vec![
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
                ],
                3,
            )
            .expect("B2 reordered fixture")
        }

        fn missing_signed_zero_fixture() -> CsrMatrix {
            CsrMatrix::from_rows(
                vec![
                    vec![(0, 2.0), (1, -1.0), (1, -0.0)],
                    vec![(0, -3.0), (1, 2.0)],
                ],
                2,
            )
            .expect("B2 missing-side fixture")
        }

        fn order_fixture() -> CsrMatrix {
            CsrMatrix::from_rows(
                vec![
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
                ],
                3,
            )
            .expect("B2 face-order fixture")
        }

        fn diagonal_cache(csr: &CsrMatrix) -> (Vec<usize>, Vec<f64>) {
            let slots = (0..csr.rows())
                .map(|row| {
                    (csr.row_offsets()[row]..csr.row_offsets()[row + 1])
                        .find(|&slot| csr.col_indices()[slot] == row)
                        .expect("B2 diagonal slot")
                })
                .collect::<Vec<_>>();
            let values = slots.iter().map(|&slot| csr.values()[slot]).collect();
            (slots, values)
        }

        fn reverse_first_owner_faces(level: &mut GamgLduLevel) {
            let faces = level.owner_start[0]..level.owner_start[1];
            level.lower_addr[faces.clone()].reverse();
            level.upper_addr[faces.clone()].reverse();
            level.lower_csr[faces.clone()].reverse();
            level.upper_csr[faces.clone()].reverse();
            level.lower[faces.clone()].reverse();
            level.upper[faces].reverse();
        }

        fn assert_cached_csr_parity(csr: &CsrMatrix) {
            let (diagonal_slots, diagonal) = diagonal_cache(csr);
            let rhs = (0..csr.rows())
                .map(|row| 1.0 - row as f64 * 0.375)
                .collect::<Vec<_>>();
            let initial = (0..csr.rows())
                .map(|row| 0.25 + row as f64 * 0.125)
                .collect::<Vec<_>>();
            let mut ldu = initial.clone();
            let mut cached = initial;
            ldu_sym_gauss_seidel_sweep(&mut GamgLduLevel::new(csr), &diagonal, &rhs, &mut ldu)
                .expect("B2 LDU sweep");
            gauss_seidel_sweep_with_cached_diagonal(
                csr,
                &diagonal_slots,
                &diagonal,
                &rhs,
                &mut cached,
                0..csr.rows(),
            )
            .expect("B2 cached CSR forward sweep");
            gauss_seidel_sweep_with_cached_diagonal(
                csr,
                &diagonal_slots,
                &diagonal,
                &rhs,
                &mut cached,
                (0..csr.rows()).rev(),
            )
            .expect("B2 cached CSR backward sweep");
            for (row, (&actual, &expected)) in ldu.iter().zip(&cached).enumerate() {
                let scale = actual.abs().max(expected.abs()).max(1.0);
                assert!(
                    (actual - expected).abs() <= 64.0 * f64::EPSILON * scale,
                    "B2 LDU/CSR mismatch at row {row}: {actual} != {expected}"
                );
            }
        }

        fn face_weights() -> Vec<GamgFacePairWeight> {
            let mut result = Vec::with_capacity(12);
            let mut ordinal = 0usize;
            for z in 0..2 {
                for y in 0..2 {
                    for x in 0..2 {
                        let first = cell(x, y, z);
                        for second in [
                            (x + 1 < 2).then(|| cell(x + 1, y, z)),
                            (y + 1 < 2).then(|| cell(x, y + 1, z)),
                            (z + 1 < 2).then(|| cell(x, y, z + 1)),
                        ]
                        .into_iter()
                        .flatten()
                        {
                            ordinal += 1;
                            result.push(
                                GamgFacePairWeight::new(first, second, 1.0 + ordinal as f64 / 16.0)
                                    .expect("positive nonuniform B2 face weight"),
                            );
                        }
                    }
                }
            }
            result
        }

        fn options(
            direct: bool,
            cache: bool,
            agglomerator: GamgAgglomerator,
            smoother: GamgSmoother,
        ) -> GamgOptions {
            GamgOptions {
                max_iterations: 1,
                min_iterations: 1,
                tolerance: 0.0,
                relative_tolerance: 0.0,
                cache_agglomeration: cache,
                n_cells_in_coarsest_level: 2,
                merge_levels: 1,
                agglomerator,
                smoother,
                n_pre_sweeps: 0,
                pre_sweeps_level_multiplier: 0,
                max_pre_sweeps: 0,
                n_post_sweeps: 1,
                post_sweeps_level_multiplier: 0,
                max_post_sweeps: 1,
                n_finest_sweeps: 1,
                interpolate_correction: false,
                scale_correction: false,
                direct_solve_coarsest: direct,
            }
        }

        fn workspace(
            matrix: &CsrMatrix,
            options: GamgOptions,
            weights: &[GamgFacePairWeight],
        ) -> GamgWorkspace {
            match options.agglomerator {
                GamgAgglomerator::AlgebraicPair => GamgWorkspace::new(matrix, options),
                GamgAgglomerator::FaceAreaPair => {
                    GamgWorkspace::new_with_face_area_weights(matrix, options, weights)
                }
            }
            .expect("B2 workspace")
        }

        fn allocations(level: &GamgLduLevel) -> [(usize, usize, usize); 8] {
            [
                (
                    level.lower_addr.as_ptr() as usize,
                    level.lower_addr.len(),
                    level.lower_addr.capacity(),
                ),
                (
                    level.upper_addr.as_ptr() as usize,
                    level.upper_addr.len(),
                    level.upper_addr.capacity(),
                ),
                (
                    level.lower_csr.as_ptr() as usize,
                    level.lower_csr.len(),
                    level.lower_csr.capacity(),
                ),
                (
                    level.upper_csr.as_ptr() as usize,
                    level.upper_csr.len(),
                    level.upper_csr.capacity(),
                ),
                (
                    level.lower.as_ptr() as usize,
                    level.lower.len(),
                    level.lower.capacity(),
                ),
                (
                    level.upper.as_ptr() as usize,
                    level.upper.len(),
                    level.upper.capacity(),
                ),
                (
                    level.owner_start.as_ptr() as usize,
                    level.owner_start.len(),
                    level.owner_start.capacity(),
                ),
                (
                    level.b_prime.as_ptr() as usize,
                    level.b_prime.len(),
                    level.b_prime.capacity(),
                ),
            ]
        }

        fn controls() -> GamgSolveControls {
            GamgSolveControls {
                max_iterations: 1,
                min_iterations: 1,
                tolerance: 0.0,
                relative_tolerance: 0.0,
            }
        }

        fn normalized_controls() -> NormalizedL1GamgSolveControls {
            NormalizedL1GamgSolveControls {
                normalization_factor: 1.0,
                tolerance: 0.0,
                relative_tolerance: 0.0,
                l2_controls: controls(),
            }
        }

        fn rhs() -> [f64; 8] {
            [1.0, -0.75, 0.5, 1.25, -1.5, 0.625, -0.375, 0.875]
        }

        fn initial() -> [f64; 8] {
            [0.25, -0.125, 0.375, -0.25, 0.5, -0.375, 0.125, -0.5]
        }

        fn assert_report_literal(report: &IterativeSolveReport) {
            assert_eq!(report.solution.len(), 8);
            assert_eq!(report.iterations, 1);
            assert!(!report.converged);
            assert_eq!(report.termination, IterativeSolveTermination::MaxIterations);
            assert!(report.residual_norm.is_finite());
            assert!(report.residual_norm > 0.0);
            assert!(report.solution.iter().all(|value| value.is_finite()));
        }

        fn assert_report_bits_equal(
            actual: &IterativeSolveReport,
            expected: &IterativeSolveReport,
        ) {
            assert_eq!(actual.iterations, expected.iterations);
            assert_eq!(actual.converged, expected.converged);
            assert_eq!(actual.termination, expected.termination);
            assert_eq!(
                actual.residual_norm.to_bits(),
                expected.residual_norm.to_bits()
            );
            assert_eq!(
                actual
                    .solution
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                expected
                    .solution
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>()
            );
        }

        fn assert_report_close(actual: &IterativeSolveReport, expected: &IterativeSolveReport) {
            assert_eq!(actual.iterations, expected.iterations);
            assert_eq!(actual.converged, expected.converged);
            assert_eq!(actual.termination, expected.termination);
            let residual_scale = actual
                .residual_norm
                .abs()
                .max(expected.residual_norm.abs())
                .max(1.0);
            assert!(
                (actual.residual_norm - expected.residual_norm).abs()
                    <= 64.0 * f64::EPSILON * residual_scale
            );
            for (actual, expected) in actual.solution.iter().zip(&expected.solution) {
                let scale = actual.abs().max(expected.abs()).max(1.0);
                assert!((actual - expected).abs() <= 64.0 * f64::EPSILON * scale);
            }
        }

        fn assert_seconds(value: f64) {
            assert!(value.is_finite());
            assert!(value >= 0.0);
        }

        fn assert_profile_literal(timing: &GamgKernelTiming, rebuilds: usize) {
            assert_eq!(timing.hierarchy_builds, 0);
            assert_eq!(timing.hierarchy_rebuilds, rebuilds);
            assert_eq!(timing.matrix_refreshes, 1);
            assert_eq!(timing.finest_residual_evaluations, 2);
            assert_eq!(timing.solves, 1);
            assert_eq!(timing.v_cycles, 1);
            assert_eq!(timing.levels.len(), 3);
            assert_eq!(
                timing
                    .levels
                    .iter()
                    .map(|level| (level.level, level.cells, level.nonzeros))
                    .collect::<Vec<_>>(),
                [(0, 8, 32), (1, 4, 12), (2, 2, 4)]
            );
            let expected_counters = [
                (1, 1, 1, 1, 1, 0, 0, 1, 0),
                (1, 1, 1, 1, 1, 0, 0, 1, 0),
                (1, 0, 0, 0, 0, 0, 0, 0, 1),
            ];
            for (level, expected) in timing.levels.iter().zip(expected_counters) {
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
                    assert_seconds(seconds);
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
                timing.restriction_seconds(),
                timing.prolongation_seconds(),
                timing.smoothing_seconds(),
                timing.scaling_seconds(),
                timing.coarse_residual_seconds(),
                timing.correction_seconds(),
                timing.coarsest_solve_seconds(),
                timing.v_cycle_other_seconds(),
            ] {
                assert_seconds(seconds);
            }
            assert_eq!(timing.hierarchy_build_seconds.to_bits(), 0.0f64.to_bits());
            if rebuilds == 0 {
                assert_eq!(timing.hierarchy_rebuild_seconds.to_bits(), 0.0f64.to_bits());
            }
        }

        fn assert_complete_face_weights(matrix: &CsrMatrix, weights: &[GamgFacePairWeight]) {
            assert_eq!(weights.len(), 12);
            assert!(weights.iter().all(|weight| weight.weight() > 0.0));
            assert_eq!(
                weights
                    .iter()
                    .map(|weight| weight.weight().to_bits())
                    .collect::<BTreeSet<_>>()
                    .len(),
                12
            );
            let weighted_pairs = weights
                .iter()
                .map(|weight| {
                    let (first, second) = weight.cells();
                    if first < second {
                        (first, second)
                    } else {
                        (second, first)
                    }
                })
                .collect::<BTreeSet<_>>();
            assert_eq!(weighted_pairs, matrix_connection_pairs(matrix));
        }

        fn assert_ldu_values(workspace: &GamgWorkspace) {
            let levels = workspace.ldu_levels.as_ref().expect("B2 LDU levels");
            assert_eq!(levels.len(), workspace.matrices.len());
            for (level, matrix) in levels.iter().zip(&workspace.matrices) {
                for face in 0..level.lower.len() {
                    assert_eq!(
                        level.lower[face].to_bits(),
                        level.lower_csr[face]
                            .map_or(0.0, |slot| matrix.values()[slot])
                            .to_bits()
                    );
                    assert_eq!(
                        level.upper[face].to_bits(),
                        level.upper_csr[face]
                            .map_or(0.0, |slot| matrix.values()[slot])
                            .to_bits()
                    );
                }
            }
        }

        fn expected_solve_events(
            kind: B2PublicCallKind,
            direct: bool,
            rebuild: bool,
        ) -> Vec<B2TraceEvent> {
            let mut expected = vec![B2TraceEvent::PublicCall(kind)];
            if rebuild {
                expected.push(B2TraceEvent::HierarchyRebuild);
                expected.extend((0..3).map(|level| B2TraceEvent::LduBuild { level }));
            }
            for level in 0..3 {
                expected.push(B2TraceEvent::CsrRefresh { level });
                expected.push(B2TraceEvent::DiagonalRefresh { level });
            }
            expected.extend((0..3).map(|level| B2TraceEvent::LduRefresh { level }));
            expected.push(if direct {
                B2TraceEvent::CoarseDirect { level: 2 }
            } else {
                B2TraceEvent::CoarsePcg { level: 2 }
            });
            expected.push(B2TraceEvent::LduSweep { level: 1 });
            expected.push(B2TraceEvent::LduSweep { level: 0 });
            expected
        }

        fn expected_scanned_solve_events(direct: bool, iterations: usize) -> Vec<B2TraceEvent> {
            let mut expected = Vec::new();
            for level in 0..3 {
                expected.push(B2TraceEvent::CsrRefresh { level });
                expected.push(B2TraceEvent::DiagonalRefresh { level });
            }
            for _ in 0..iterations {
                expected.push(if direct {
                    B2TraceEvent::CoarseDirect { level: 2 }
                } else {
                    B2TraceEvent::CoarsePcg { level: 2 }
                });
                expected.push(B2TraceEvent::CsrSweep {
                    level: 1,
                    symmetric: true,
                });
                expected.push(B2TraceEvent::CsrSweep {
                    level: 0,
                    symmetric: true,
                });
            }
            expected
        }

        fn assert_scanned_solve_suffix(
            events: &[B2TraceEvent],
            direct: bool,
            calls: usize,
            iterations: usize,
        ) {
            let expected = (0..calls)
                .flat_map(|_| expected_scanned_solve_events(direct, iterations))
                .collect::<Vec<_>>();
            assert_eq!(events, expected);
            assert_eq!(
                events
                    .iter()
                    .filter(|event| matches!(event, B2TraceEvent::PublicCall(_)))
                    .count(),
                0
            );
            assert_eq!(
                events
                    .iter()
                    .filter(|event| matches!(event, B2TraceEvent::CsrRefresh { .. }))
                    .count(),
                3 * calls
            );
            assert_eq!(
                events
                    .iter()
                    .filter(|event| matches!(event, B2TraceEvent::DiagonalRefresh { .. }))
                    .count(),
                3 * calls
            );
            assert_eq!(
                events
                    .iter()
                    .filter(|event| matches!(
                        event,
                        B2TraceEvent::CsrSweep {
                            symmetric: true,
                            ..
                        }
                    ))
                    .count(),
                2 * iterations * calls
            );
            assert!(!events.iter().any(|event| matches!(
                event,
                B2TraceEvent::CsrSweep {
                    symmetric: false,
                    ..
                }
            )));
            assert!(!events.iter().any(|event| matches!(
                event,
                B2TraceEvent::LduBuild { .. }
                    | B2TraceEvent::LduRefresh { .. }
                    | B2TraceEvent::LduSweep { .. }
            )));
            let direct_count = events
                .iter()
                .filter(|event| matches!(event, B2TraceEvent::CoarseDirect { level: 2 }))
                .count();
            let pcg_count = events
                .iter()
                .filter(|event| matches!(event, B2TraceEvent::CoarsePcg { level: 2 }))
                .count();
            assert_eq!(direct_count, usize::from(direct) * iterations * calls);
            assert_eq!(pcg_count, usize::from(!direct) * iterations * calls);
        }

        fn assert_configuration_trace(events: &[B2TraceEvent], direct: bool, cache: bool) {
            let starts = events
                .iter()
                .enumerate()
                .filter_map(|(index, event)| {
                    matches!(event, B2TraceEvent::PublicCall(_)).then_some(index)
                })
                .collect::<Vec<_>>();
            assert_eq!(starts.len(), 8);
            assert_eq!(
                &events[..starts[0]],
                (0..4)
                    .flat_map(|_| (0..3).map(|level| B2TraceEvent::LduBuild { level }))
                    .collect::<Vec<_>>()
            );
            let kinds = [
                B2PublicCallKind::L2Plain,
                B2PublicCallKind::L2Plain,
                B2PublicCallKind::L2Profiled,
                B2PublicCallKind::L2Profiled,
                B2PublicCallKind::NormalizedL1Plain,
                B2PublicCallKind::NormalizedL1Plain,
                B2PublicCallKind::NormalizedL1Profiled,
                B2PublicCallKind::NormalizedL1Profiled,
            ];
            for (call, (&start, kind)) in starts.iter().zip(kinds).enumerate() {
                let end = starts.get(call + 1).copied().unwrap_or(events.len());
                assert_eq!(
                    &events[start..end],
                    expected_solve_events(kind, direct, !cache && call % 2 == 1)
                );
            }
        }

        type Allocation = (usize, usize, usize);
        type OptionsState = (
            (
                usize,
                usize,
                u64,
                u64,
                bool,
                usize,
                usize,
                GamgAgglomerator,
                GamgSmoother,
                usize,
            ),
            (usize, usize, usize, usize, usize, usize, bool, bool, bool),
        );
        type LduSemanticLevel = (
            Vec<usize>,
            Vec<usize>,
            Vec<Option<usize>>,
            Vec<Option<usize>>,
            Vec<u64>,
            Vec<u64>,
            Vec<usize>,
            Vec<u64>,
        );
        type TimingLevelSignature = (usize, usize, usize, [usize; 9]);
        type TimingSignature = (
            usize,
            usize,
            usize,
            usize,
            usize,
            usize,
            Vec<TimingLevelSignature>,
        );

        #[derive(Clone, Debug, PartialEq, Eq)]
        struct UsizeState {
            allocation: Allocation,
            values: Vec<usize>,
        }

        #[derive(Clone, Debug, PartialEq, Eq)]
        struct OptionalUsizeState {
            allocation: Allocation,
            values: Vec<Option<usize>>,
        }

        #[derive(Clone, Debug, PartialEq, Eq)]
        struct F64State {
            allocation: Allocation,
            bits: Vec<u64>,
        }

        #[derive(Clone, Debug, PartialEq, Eq)]
        struct LduState {
            lower_addr: UsizeState,
            upper_addr: UsizeState,
            lower_csr: OptionalUsizeState,
            upper_csr: OptionalUsizeState,
            lower: F64State,
            upper: F64State,
            owner_start: UsizeState,
            b_prime: F64State,
        }

        #[derive(Clone, Debug, PartialEq, Eq)]
        struct MatrixState {
            rows: usize,
            cols: usize,
            row_arc: usize,
            column_arc: usize,
            row_offsets: Vec<usize>,
            columns: Vec<usize>,
            values: F64State,
        }

        #[derive(Clone, Debug, PartialEq, Eq)]
        struct WorkspaceState {
            options: OptionsState,
            finest: (usize, usize, usize, usize, Vec<usize>, Vec<usize>),
            matrices: Vec<MatrixState>,
            transfers: Vec<(UsizeState, UsizeState)>,
            diagonal_slots: Vec<UsizeState>,
            diagonal_values: Vec<F64State>,
            ldu_outer: Option<Allocation>,
            ldu: Option<Vec<LduState>>,
            corrections: Vec<F64State>,
            sources: Vec<F64State>,
            residuals: Vec<F64State>,
            products: Vec<F64State>,
            pre_smoothed: Vec<F64State>,
            pcg_vectors: Option<[F64State; 5]>,
            pcg_sparsity: Option<(usize, usize, Vec<usize>, Vec<usize>)>,
            has_solved: bool,
        }

        fn allocation<T>(values: &Vec<T>) -> Allocation {
            (values.as_ptr() as usize, values.len(), values.capacity())
        }

        fn usize_state(values: &Vec<usize>) -> UsizeState {
            UsizeState {
                allocation: allocation(values),
                values: values.clone(),
            }
        }

        fn optional_usize_state(values: &Vec<Option<usize>>) -> OptionalUsizeState {
            OptionalUsizeState {
                allocation: allocation(values),
                values: values.clone(),
            }
        }

        fn f64_state(values: &Vec<f64>) -> F64State {
            F64State {
                allocation: allocation(values),
                bits: values.iter().map(|value| value.to_bits()).collect(),
            }
        }

        fn ldu_state(level: &GamgLduLevel) -> LduState {
            LduState {
                lower_addr: usize_state(&level.lower_addr),
                upper_addr: usize_state(&level.upper_addr),
                lower_csr: optional_usize_state(&level.lower_csr),
                upper_csr: optional_usize_state(&level.upper_csr),
                lower: f64_state(&level.lower),
                upper: f64_state(&level.upper),
                owner_start: usize_state(&level.owner_start),
                b_prime: f64_state(&level.b_prime),
            }
        }

        fn ldu_states(workspace: &GamgWorkspace) -> Option<Vec<LduState>> {
            workspace
                .ldu_levels
                .as_ref()
                .map(|levels| levels.iter().map(ldu_state).collect())
        }

        fn workspace_state(workspace: &GamgWorkspace) -> WorkspaceState {
            let pcg_vectors = workspace.coarsest_pcg.as_ref().map(|pcg| {
                [
                    f64_state(&pcg.residual),
                    f64_state(&pcg.preconditioned_residual),
                    f64_state(&pcg.direction),
                    f64_state(&pcg.matrix_direction),
                    f64_state(&pcg.preconditioner_scratch),
                ]
            });
            let pcg_sparsity = workspace.coarsest_pcg.as_ref().map(|pcg| {
                (
                    pcg.sparsity.row_offsets.as_ptr() as usize,
                    pcg.sparsity.col_indices.as_ptr() as usize,
                    pcg.sparsity.row_offsets.to_vec(),
                    pcg.sparsity.col_indices.to_vec(),
                )
            });
            WorkspaceState {
                options: (
                    (
                        workspace.options.max_iterations,
                        workspace.options.min_iterations,
                        workspace.options.tolerance.to_bits(),
                        workspace.options.relative_tolerance.to_bits(),
                        workspace.options.cache_agglomeration,
                        workspace.options.n_cells_in_coarsest_level,
                        workspace.options.merge_levels,
                        workspace.options.agglomerator,
                        workspace.options.smoother,
                        workspace.options.n_pre_sweeps,
                    ),
                    (
                        workspace.options.pre_sweeps_level_multiplier,
                        workspace.options.max_pre_sweeps,
                        workspace.options.n_post_sweeps,
                        workspace.options.post_sweeps_level_multiplier,
                        workspace.options.max_post_sweeps,
                        workspace.options.n_finest_sweeps,
                        workspace.options.interpolate_correction,
                        workspace.options.scale_correction,
                        workspace.options.direct_solve_coarsest,
                    ),
                ),
                finest: (
                    workspace.finest_sparsity.row_offsets.as_ptr() as usize,
                    workspace.finest_sparsity.col_indices.as_ptr() as usize,
                    workspace.finest_sparsity.rows,
                    workspace.finest_sparsity.cols,
                    workspace.finest_sparsity.row_offsets.to_vec(),
                    workspace.finest_sparsity.col_indices.to_vec(),
                ),
                matrices: workspace
                    .matrices
                    .iter()
                    .map(|matrix| MatrixState {
                        rows: matrix.rows,
                        cols: matrix.cols,
                        row_arc: matrix.row_offsets.as_ptr() as usize,
                        column_arc: matrix.col_indices.as_ptr() as usize,
                        row_offsets: matrix.row_offsets.to_vec(),
                        columns: matrix.col_indices.to_vec(),
                        values: f64_state(&matrix.values),
                    })
                    .collect(),
                transfers: workspace
                    .transfers
                    .iter()
                    .map(|transfer| {
                        (
                            usize_state(&transfer.fine_to_coarse),
                            usize_state(&transfer.fine_entry_to_coarse_entry),
                        )
                    })
                    .collect(),
                diagonal_slots: workspace.diagonal_slots.iter().map(usize_state).collect(),
                diagonal_values: workspace.diagonal_values.iter().map(f64_state).collect(),
                ldu_outer: workspace.ldu_levels.as_ref().map(allocation),
                ldu: ldu_states(workspace),
                corrections: workspace.corrections.iter().map(f64_state).collect(),
                sources: workspace.sources.iter().map(f64_state).collect(),
                residuals: workspace.residuals.iter().map(f64_state).collect(),
                products: workspace.products.iter().map(f64_state).collect(),
                pre_smoothed: workspace.pre_smoothed.iter().map(f64_state).collect(),
                pcg_vectors,
                pcg_sparsity,
                has_solved: workspace.has_solved,
            }
        }

        fn without_allocation_addresses(mut state: WorkspaceState) -> WorkspaceState {
            state.finest.0 = 0;
            state.finest.1 = 0;
            for matrix in &mut state.matrices {
                matrix.row_arc = 0;
                matrix.column_arc = 0;
                matrix.values.allocation.0 = 0;
            }
            for (fine_to_coarse, entry_to_coarse) in &mut state.transfers {
                fine_to_coarse.allocation.0 = 0;
                entry_to_coarse.allocation.0 = 0;
            }
            for slots in &mut state.diagonal_slots {
                slots.allocation.0 = 0;
            }
            for values in &mut state.diagonal_values {
                values.allocation.0 = 0;
            }
            if let Some(outer) = &mut state.ldu_outer {
                outer.0 = 0;
            }
            if let Some(levels) = &mut state.ldu {
                for level in levels {
                    level.lower_addr.allocation.0 = 0;
                    level.upper_addr.allocation.0 = 0;
                    level.lower_csr.allocation.0 = 0;
                    level.upper_csr.allocation.0 = 0;
                    level.lower.allocation.0 = 0;
                    level.upper.allocation.0 = 0;
                    level.owner_start.allocation.0 = 0;
                    level.b_prime.allocation.0 = 0;
                }
            }
            for buffers in [
                &mut state.corrections,
                &mut state.sources,
                &mut state.residuals,
                &mut state.products,
                &mut state.pre_smoothed,
            ] {
                for buffer in buffers {
                    buffer.allocation.0 = 0;
                }
            }
            if let Some(vectors) = &mut state.pcg_vectors {
                for vector in vectors {
                    vector.allocation.0 = 0;
                }
            }
            if let Some(sparsity) = &mut state.pcg_sparsity {
                sparsity.0 = 0;
                sparsity.1 = 0;
            }
            state
        }

        fn allocation_signature(
            workspace: &GamgWorkspace,
        ) -> (Vec<Allocation>, Vec<[Allocation; 8]>) {
            (
                workspace.diagonal_values.iter().map(allocation).collect(),
                workspace
                    .ldu_levels
                    .as_ref()
                    .expect("SymGS LDU hierarchy")
                    .iter()
                    .map(allocations)
                    .collect(),
            )
        }

        fn ldu_semantic_signature(workspace: &GamgWorkspace) -> Option<Vec<LduSemanticLevel>> {
            workspace.ldu_levels.as_ref().map(|levels| {
                levels
                    .iter()
                    .map(|level| {
                        (
                            level.lower_addr.clone(),
                            level.upper_addr.clone(),
                            level.lower_csr.clone(),
                            level.upper_csr.clone(),
                            level.lower.iter().map(|value| value.to_bits()).collect(),
                            level.upper.iter().map(|value| value.to_bits()).collect(),
                            level.owner_start.clone(),
                            level.b_prime.iter().map(|value| value.to_bits()).collect(),
                        )
                    })
                    .collect()
            })
        }

        fn matrix_value_bits(workspace: &GamgWorkspace) -> Vec<Vec<u64>> {
            workspace
                .matrices
                .iter()
                .map(|matrix| {
                    matrix
                        .values()
                        .iter()
                        .map(|value| value.to_bits())
                        .collect()
                })
                .collect()
        }

        fn diagonal_value_bits(workspace: &GamgWorkspace) -> Vec<Vec<u64>> {
            workspace
                .diagonal_values
                .iter()
                .map(|values| values.iter().map(|value| value.to_bits()).collect())
                .collect()
        }

        fn assert_invalid(error: crate::MeshError, expected: &str) {
            assert_eq!(error.to_string(), expected);
            let crate::MeshError::InvalidInput(payload) = error else {
                panic!("expected InvalidInput, got {error:?}");
            };
            assert_eq!(payload, expected);
        }

        fn timing_signature(timing: &GamgKernelTiming) -> TimingSignature {
            (
                timing.hierarchy_builds,
                timing.hierarchy_rebuilds,
                timing.matrix_refreshes,
                timing.finest_residual_evaluations,
                timing.solves,
                timing.v_cycles,
                timing
                    .levels
                    .iter()
                    .map(|level| {
                        (
                            level.level,
                            level.cells,
                            level.nonzeros,
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
                            ],
                        )
                    })
                    .collect(),
            )
        }

        fn sentinelize_ldu(workspace: &mut GamgWorkspace) {
            for (level_index, level) in workspace
                .ldu_levels
                .as_mut()
                .expect("SymGS LDU hierarchy")
                .iter_mut()
                .enumerate()
            {
                level.lower_addr.fill(usize::MAX - level_index);
                level.upper_addr.fill(usize::MAX - level_index);
                level.lower_csr.fill(None);
                level.upper_csr.fill(None);
                level.lower.fill(1000.0 + level_index as f64);
                level.upper.fill(-1000.0 - level_index as f64);
                level.owner_start.fill(usize::MAX - level_index);
                level.b_prime.fill(2000.0 + level_index as f64);
            }
        }

        fn singular_matrix() -> CsrMatrix {
            CsrMatrix::from_rows(
                vec![
                    vec![(0, 1.0), (1, -1.0)],
                    vec![(0, -1.0), (1, 2.0), (2, -1.0)],
                    vec![(1, -1.0), (2, 2.0), (3, -1.0)],
                    vec![(2, -1.0), (3, 1.0)],
                ],
                4,
            )
            .expect("singular B2 matrix")
        }

        #[test]
        fn ldu_b2_builds_one_level_per_gamg_matrix_only_for_symgs() {
            let csr = matrix(1.0);
            let weights = face_weights();
            assert_complete_face_weights(&csr, &weights);
            let trace = B2TraceGuard::new();
            let sym = workspace(
                &csr,
                options(
                    true,
                    true,
                    GamgAgglomerator::AlgebraicPair,
                    GamgSmoother::SymGaussSeidel,
                ),
                &weights,
            );
            assert_eq!(sym.ldu_levels.as_ref().unwrap().len(), sym.matrices.len());
            assert_eq!(sym.level_sizes(), [8, 4, 2]);
            let gauss = workspace(
                &csr,
                options(
                    true,
                    true,
                    GamgAgglomerator::AlgebraicPair,
                    GamgSmoother::GaussSeidel,
                ),
                &weights,
            );
            assert!(gauss.ldu_levels.is_none());
            assert_eq!(
                trace.events(),
                (0..3)
                    .map(|level| B2TraceEvent::LduBuild { level })
                    .collect::<Vec<_>>()
            );
            drop(trace);

            let canonical = GamgLduLevel::new(&main_fixture());
            assert_eq!(canonical.lower_addr, [0, 0, 0, 1]);
            assert_eq!(canonical.upper_addr, [1, 1, 2, 2]);
            assert_eq!(canonical.owner_start, [0, 3, 4, 4]);
            assert_eq!(canonical.lower_csr, [Some(4), Some(5), Some(8), Some(9)]);
            assert_eq!(canonical.upper_csr, [Some(1), Some(2), Some(3), Some(7)]);
            assert_eq!(
                canonical
                    .lower
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                [
                    -0.9999999999999999,
                    -0.125,
                    -0.12499999999999999,
                    -0.49999999999999994,
                ]
                .map(f64::to_bits)
            );
            assert_eq!(
                canonical
                    .upper
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                [
                    -1.0000000000000002,
                    -0.25,
                    -0.12500000000000003,
                    -0.5000000000000001,
                ]
                .map(f64::to_bits)
            );

            let reordered = GamgLduLevel::new(&reordered_main_fixture());
            assert_eq!(reordered.lower_addr, [0, 0, 0, 1]);
            assert_eq!(reordered.upper_addr, [1, 1, 2, 2]);
            assert_eq!(reordered.owner_start, [0, 3, 4, 4]);
            assert_eq!(reordered.lower_csr, [Some(6), Some(7), Some(10), Some(9)]);
            assert_eq!(reordered.upper_csr, [Some(1), Some(3), Some(0), Some(4)]);
            assert_eq!(
                reordered
                    .lower
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                [
                    -0.125,
                    -0.9999999999999999,
                    -0.12499999999999999,
                    -0.49999999999999994,
                ]
                .map(f64::to_bits)
            );
            assert_eq!(
                reordered
                    .upper
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                [
                    -0.25,
                    -1.0000000000000002,
                    -0.12500000000000003,
                    -0.5000000000000001,
                ]
                .map(f64::to_bits)
            );

            let mut missing = GamgLduLevel::new(&missing_signed_zero_fixture());
            assert_eq!(missing.lower_addr, [0, 0]);
            assert_eq!(missing.upper_addr, [1, 1]);
            assert_eq!(missing.owner_start, [0, 2, 2]);
            assert_eq!(missing.lower_csr, [Some(3), None]);
            assert_eq!(missing.upper_csr, [Some(1), Some(2)]);
            assert_eq!(missing.lower[1].to_bits(), 0.0f64.to_bits());
            assert_eq!(missing.upper[1].to_bits(), (-0.0f64).to_bits());
            let refreshed = missing_signed_zero_fixture();
            missing.refresh(&refreshed);
            assert_eq!(missing.lower[1].to_bits(), 0.0f64.to_bits());
            assert_eq!(missing.upper[1].to_bits(), (-0.0f64).to_bits());

            let mut missing_hierarchy = workspace(
                &csr,
                options(
                    true,
                    true,
                    GamgAgglomerator::AlgebraicPair,
                    GamgSmoother::SymGaussSeidel,
                ),
                &weights,
            );
            missing_hierarchy.ldu_levels = None;
            let before = workspace_state(&missing_hierarchy);
            let mut timing = GamgKernelTiming::default();
            let error = missing_hierarchy
                .solve_with_controls_internal::<false, false>(
                    &csr,
                    &rhs(),
                    None,
                    controls(),
                    &mut timing,
                )
                .expect_err("missing LDU hierarchy must fail");
            assert_invalid(error, "GAMG symGaussSeidel LDU hierarchy is missing");
            assert_eq!(workspace_state(&missing_hierarchy), before);
            assert_eq!(
                timing_signature(&timing),
                timing_signature(&GamgKernelTiming::default())
            );

            let mut short_hierarchy = workspace(
                &csr,
                options(
                    true,
                    true,
                    GamgAgglomerator::AlgebraicPair,
                    GamgSmoother::SymGaussSeidel,
                ),
                &weights,
            );
            short_hierarchy.ldu_levels.as_mut().unwrap().pop();
            let before = workspace_state(&short_hierarchy);
            let expected = short_hierarchy.matrices.len();
            let got = short_hierarchy.ldu_levels.as_ref().unwrap().len();
            let mut timing = GamgKernelTiming::default();
            let error = short_hierarchy
                .solve_with_controls_internal::<false, false>(
                    &csr,
                    &rhs(),
                    None,
                    controls(),
                    &mut timing,
                )
                .expect_err("short LDU hierarchy must fail");
            assert_invalid(
                error,
                &format!("GAMG symGaussSeidel LDU hierarchy has {got} levels, expected {expected}"),
            );
            assert_eq!(workspace_state(&short_hierarchy), before);
            assert_eq!(
                timing_signature(&timing),
                timing_signature(&GamgKernelTiming::default())
            );
        }

        #[test]
        fn ldu_b2_refreshes_all_levels_in_place_across_ten_solve_lifecycles() {
            let weights = face_weights();
            let trace = B2TraceGuard::new();
            let mut workspace = workspace(
                &matrix(1.0),
                options(
                    true,
                    true,
                    GamgAgglomerator::AlgebraicPair,
                    GamgSmoother::SymGaussSeidel,
                ),
                &weights,
            );
            let pinned = workspace
                .ldu_levels
                .as_ref()
                .unwrap()
                .iter()
                .map(allocations)
                .collect::<Vec<_>>();
            let pinned_outer = workspace.ldu_levels.as_ref().map(allocation);
            let pinned_diagonals = workspace
                .diagonal_values
                .iter()
                .map(|values| (values.as_ptr() as usize, values.len(), values.capacity()))
                .collect::<Vec<_>>();
            let pinned_topology = workspace
                .ldu_levels
                .as_ref()
                .unwrap()
                .iter()
                .map(|level| {
                    (
                        level.lower_addr.clone(),
                        level.upper_addr.clone(),
                        level.lower_csr.clone(),
                        level.upper_csr.clone(),
                        level.owner_start.clone(),
                    )
                })
                .collect::<Vec<_>>();
            for cycle in 1..=10 {
                let csr = matrix(1.0 + f64::from(cycle) / 32.0);
                let report = workspace
                    .solve_with_controls(&csr, &rhs(), Some(&initial()), controls())
                    .unwrap();
                assert_report_literal(&report);
                assert_ldu_values(&workspace);
                assert_eq!(workspace.ldu_levels.as_ref().map(allocation), pinned_outer);
                for (level, expected) in workspace.ldu_levels.as_ref().unwrap().iter().zip(&pinned)
                {
                    assert_eq!(allocations(level), *expected);
                }
                assert_eq!(
                    workspace
                        .diagonal_values
                        .iter()
                        .map(|values| (values.as_ptr() as usize, values.len(), values.capacity()))
                        .collect::<Vec<_>>(),
                    pinned_diagonals
                );
                assert_eq!(
                    workspace
                        .ldu_levels
                        .as_ref()
                        .unwrap()
                        .iter()
                        .map(|level| {
                            (
                                level.lower_addr.clone(),
                                level.upper_addr.clone(),
                                level.lower_csr.clone(),
                                level.upper_csr.clone(),
                                level.owner_start.clone(),
                            )
                        })
                        .collect::<Vec<_>>(),
                    pinned_topology
                );
                for ((matrix, slots), values) in workspace
                    .matrices
                    .iter()
                    .zip(&workspace.diagonal_slots)
                    .zip(&workspace.diagonal_values)
                {
                    assert_eq!(
                        values
                            .iter()
                            .map(|value| value.to_bits())
                            .collect::<Vec<_>>(),
                        slots
                            .iter()
                            .map(|&slot| matrix.values()[slot].to_bits())
                            .collect::<Vec<_>>()
                    );
                }
            }
            let events = trace.events();
            assert_eq!(
                events
                    .iter()
                    .filter(|event| matches!(
                        event,
                        B2TraceEvent::PublicCall(B2PublicCallKind::L2Plain)
                    ))
                    .count(),
                10
            );
            for expected in [
                B2TraceEvent::HierarchyRebuild,
                B2TraceEvent::CoarsePcg { level: 2 },
            ] {
                assert!(!events.contains(&expected));
            }
            assert_eq!(
                events
                    .iter()
                    .filter(|event| matches!(event, B2TraceEvent::LduBuild { .. }))
                    .count(),
                3
            );
            assert_eq!(
                events
                    .iter()
                    .filter(|event| matches!(event, B2TraceEvent::CsrRefresh { .. }))
                    .count(),
                30
            );
            assert_eq!(
                events
                    .iter()
                    .filter(|event| matches!(event, B2TraceEvent::DiagonalRefresh { .. }))
                    .count(),
                30
            );
            assert_eq!(
                events
                    .iter()
                    .filter(|event| matches!(event, B2TraceEvent::LduRefresh { .. }))
                    .count(),
                30
            );
            assert_eq!(
                events
                    .iter()
                    .filter(|event| matches!(event, B2TraceEvent::CoarseDirect { level: 2 }))
                    .count(),
                10
            );
            assert_eq!(
                events
                    .iter()
                    .filter(|event| matches!(event, B2TraceEvent::LduSweep { .. }))
                    .count(),
                20
            );
        }

        #[test]
        fn ldu_b2_rebuilds_levels_exactly_when_cache_agglomeration_is_disabled() {
            let weights = face_weights();
            let trace = B2TraceGuard::new();
            let opts = options(
                true,
                false,
                GamgAgglomerator::AlgebraicPair,
                GamgSmoother::SymGaussSeidel,
            );
            let mut workspace = workspace(&matrix(1.0), opts, &weights);
            let first = workspace
                .solve_with_controls_profiled(&matrix(1.0), &rhs(), Some(&initial()), controls())
                .unwrap();
            let first_allocations = allocation_signature(&workspace);
            let second = workspace
                .solve_with_controls_profiled(&matrix(1.125), &rhs(), Some(&initial()), controls())
                .unwrap();
            assert_profile_literal(&first.timing, 0);
            assert_profile_literal(&second.timing, 1);
            let mut expected = (0..3)
                .map(|level| B2TraceEvent::LduBuild { level })
                .collect::<Vec<_>>();
            expected.extend(expected_solve_events(
                B2PublicCallKind::L2Profiled,
                true,
                false,
            ));
            expected.extend(expected_solve_events(
                B2PublicCallKind::L2Profiled,
                true,
                true,
            ));
            assert_eq!(trace.events(), expected);
            drop(trace);

            let rebuilt_matrix = matrix(1.125);
            let mut fresh = self::workspace(&rebuilt_matrix, opts, &weights);
            let expected = fresh
                .solve_with_controls_profiled(&rebuilt_matrix, &rhs(), Some(&initial()), controls())
                .expect("fresh uncached lifecycle");
            assert_report_bits_equal(&second.report, &expected.report);
            assert_eq!(matrix_value_bits(&workspace), matrix_value_bits(&fresh));
            assert_eq!(diagonal_value_bits(&workspace), diagonal_value_bits(&fresh));
            assert_eq!(
                ldu_semantic_signature(&workspace),
                ldu_semantic_signature(&fresh)
            );
            assert_eq!(
                workspace.ldu_levels.as_ref().unwrap().len(),
                workspace.matrices.len()
            );
            assert_ne!(allocation_signature(&workspace), first_allocations);
        }

        #[test]
        fn ldu_b2_symgs_matches_openfoam_v13_forward_backward_bit_oracle() {
            let csr = main_fixture();
            let mut level = GamgLduLevel::new(&csr);
            let diagonal = [4.0, 3.0, 2.5];
            let rhs = [1.0, -2.0, 0.5];
            let mut psi = [0.25, -0.5, 0.75];
            level.b_prime.copy_from_slice(&rhs);
            ldu_sym_gauss_seidel_half(&mut level, &diagonal, &mut psi, 0..3).unwrap();
            assert_eq!(
                psi.map(f64::to_bits),
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
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                [
                    4607182418800017408,
                    13834464319003164672,
                    4598459626552994475
                ]
            );
            ldu_sym_gauss_seidel_half(&mut level, &diagonal, &mut psi, (0..3).rev()).unwrap();
            assert_eq!(
                psi.map(f64::to_bits),
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
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                [
                    4607182418800017408,
                    13834138746738232525,
                    13807295973087021176
                ]
            );

            let expected = (
                psi.map(f64::to_bits),
                level
                    .b_prime
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
            );
            let mut wrapper_level = GamgLduLevel::new(&csr);
            let mut wrapper_psi = [0.25, -0.5, 0.75];
            ldu_sym_gauss_seidel_sweep(&mut wrapper_level, &diagonal, &rhs, &mut wrapper_psi)
                .unwrap();
            assert_eq!(wrapper_psi.map(f64::to_bits), expected.0);
            assert_eq!(
                wrapper_level
                    .b_prime
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                expected.1
            );

            let mut reset_level = GamgLduLevel::new(&csr);
            let mut reset_psi = [0.25, -0.5, 0.75];
            reset_level.b_prime.copy_from_slice(&rhs);
            ldu_sym_gauss_seidel_half(&mut reset_level, &diagonal, &mut reset_psi, 0..3).unwrap();
            reset_level.b_prime.copy_from_slice(&rhs);
            ldu_sym_gauss_seidel_half(&mut reset_level, &diagonal, &mut reset_psi, (0..3).rev())
                .unwrap();
            let wrong_reset = (
                reset_psi.map(f64::to_bits),
                reset_level
                    .b_prime
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
            );
            assert_eq!(
                wrong_reset.0,
                [
                    4588567540340219356,
                    13827251815928054852,
                    4596373779694328218
                ]
            );
            assert_eq!(
                wrong_reset.1,
                [
                    4607182418800017408,
                    13834762506556617523,
                    4596036009722275433
                ]
            );
            assert_ne!(wrong_reset, expected);

            let order_csr = order_fixture();
            let (_, order_diagonal) = diagonal_cache(&order_csr);
            let mut canonical_level = GamgLduLevel::new(&order_csr);
            let mut canonical_psi = [0.25, 1.0, 1.0];
            ldu_sym_gauss_seidel_sweep(
                &mut canonical_level,
                &order_diagonal,
                &rhs,
                &mut canonical_psi,
            )
            .unwrap();
            let canonical_order = (
                canonical_psi.map(f64::to_bits),
                canonical_level
                    .b_prime
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
            );
            assert_eq!(
                canonical_order.0,
                [
                    13907265759572411461,
                    13826200974869677124,
                    4594572341561366938
                ]
            );
            assert_eq!(
                canonical_order.1,
                [
                    4607182418800017408,
                    13834241774413728973,
                    4590744298198977742
                ]
            );

            let mut reversed_level = GamgLduLevel::new(&order_csr);
            reverse_first_owner_faces(&mut reversed_level);
            let mut reversed_psi = [0.25, 1.0, 1.0];
            ldu_sym_gauss_seidel_sweep(
                &mut reversed_level,
                &order_diagonal,
                &rhs,
                &mut reversed_psi,
            )
            .unwrap();
            let reversed_order = (
                reversed_psi.map(f64::to_bits),
                reversed_level
                    .b_prime
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
            );
            assert_eq!(
                reversed_order.0,
                [
                    13907265759572411460,
                    13826200974869677124,
                    4594572341561366938
                ]
            );
            assert_eq!(
                reversed_order.1,
                [
                    4607182418800017408,
                    13834241774413728972,
                    4590744298198977742
                ]
            );
            assert_ne!(reversed_order, canonical_order);
        }

        #[test]
        fn ldu_b2_symgs_matches_cached_csr_from_same_rhs_and_nonzero_initial() {
            for csr in [
                matrix(1.0),
                main_fixture(),
                reordered_main_fixture(),
                missing_signed_zero_fixture(),
            ] {
                assert_cached_csr_parity(&csr);
            }
            for size in [2, 3, 7] {
                assert_cached_csr_parity(&chain_matrix(size, 1.0));
            }
            for size in [2, 3, 7] {
                let csr = chain_matrix(size, 1.0);
                let (slots, diagonal) = diagonal_cache(&csr);
                let rhs = (0..size)
                    .map(|row| 0.75 + (row * 5 + 1) as f64 / 32.0)
                    .collect::<Vec<_>>();
                let mut ldu = (0..size)
                    .map(|row| -0.375 + row as f64 / 64.0)
                    .collect::<Vec<_>>();
                let mut cached = ldu.clone();
                ldu_sym_gauss_seidel_sweep(&mut GamgLduLevel::new(&csr), &diagonal, &rhs, &mut ldu)
                    .unwrap();
                gauss_seidel_sweep_with_cached_diagonal(
                    &csr,
                    &slots,
                    &diagonal,
                    &rhs,
                    &mut cached,
                    0..size,
                )
                .unwrap();
                gauss_seidel_sweep_with_cached_diagonal(
                    &csr,
                    &slots,
                    &diagonal,
                    &rhs,
                    &mut cached,
                    (0..size).rev(),
                )
                .unwrap();
                assert_eq!(
                    ldu.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
                    cached
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>()
                );
            }
        }

        #[test]
        fn ldu_b2_gauss_seidel_keeps_cached_csr_route() {
            let csr = matrix(1.0);
            let weights = face_weights();
            let trace = B2TraceGuard::new();
            let mut workspace = workspace(
                &csr,
                options(
                    true,
                    true,
                    GamgAgglomerator::AlgebraicPair,
                    GamgSmoother::GaussSeidel,
                ),
                &weights,
            );
            let profiled = workspace
                .solve_with_controls_profiled(&csr, &rhs(), Some(&initial()), controls())
                .unwrap();
            assert!(workspace.ldu_levels.is_none());
            assert_profile_literal(&profiled.timing, 0);
            let events = trace.events();
            assert!(!events.iter().any(|event| matches!(
                event,
                B2TraceEvent::LduBuild { .. }
                    | B2TraceEvent::LduRefresh { .. }
                    | B2TraceEvent::LduSweep { .. }
            )));
            assert_eq!(
                events
                    .iter()
                    .filter(|event| matches!(
                        event,
                        B2TraceEvent::CsrSweep {
                            symmetric: false,
                            ..
                        }
                    ))
                    .count(),
                2
            );
            assert!(events.contains(&B2TraceEvent::CoarseDirect { level: 2 }));
            drop(trace);

            for profiled in [false, true] {
                let opts = options(
                    true,
                    true,
                    GamgAgglomerator::AlgebraicPair,
                    GamgSmoother::GaussSeidel,
                );
                let mut cached = self::workspace(&csr, opts, &weights);
                let mut scanned = self::workspace(&csr, opts, &weights);
                let mut cached_timing = GamgKernelTiming::default();
                let mut scanned_timing = GamgKernelTiming::default();
                let (actual, expected) = if profiled {
                    (
                        cached
                            .solve_with_controls_internal::<true, false>(
                                &csr,
                                &rhs(),
                                Some(&initial()),
                                controls(),
                                &mut cached_timing,
                            )
                            .unwrap(),
                        scanned
                            .solve_with_controls_internal::<true, true>(
                                &csr,
                                &rhs(),
                                Some(&initial()),
                                controls(),
                                &mut scanned_timing,
                            )
                            .unwrap(),
                    )
                } else {
                    (
                        cached
                            .solve_with_controls_internal::<false, false>(
                                &csr,
                                &rhs(),
                                Some(&initial()),
                                controls(),
                                &mut cached_timing,
                            )
                            .unwrap(),
                        scanned
                            .solve_with_controls_internal::<false, true>(
                                &csr,
                                &rhs(),
                                Some(&initial()),
                                controls(),
                                &mut scanned_timing,
                            )
                            .unwrap(),
                    )
                };
                assert_report_bits_equal(&actual, &expected);
                assert_eq!(
                    timing_signature(&cached_timing),
                    timing_signature(&scanned_timing)
                );
                assert!(cached.ldu_levels.is_none());
                assert!(scanned.ldu_levels.is_none());
            }
        }

        #[test]
        fn ldu_b2_public_l2_entrypoint_matches_cached_csr_reference() {
            let csr = matrix(1.0);
            let weights = face_weights();
            let opts = options(
                true,
                true,
                GamgAgglomerator::AlgebraicPair,
                GamgSmoother::SymGaussSeidel,
            );
            let trace = B2TraceGuard::new();
            let mut ldu = workspace(&csr, opts, &weights);
            let mut csr_reference = workspace(&csr, opts, &weights);
            let actual = ldu
                .solve_with_controls(&csr, &rhs(), Some(&initial()), controls())
                .unwrap();
            let expected = csr_reference
                .solve_with_controls_internal::<false, true>(
                    &csr,
                    &rhs(),
                    Some(&initial()),
                    controls(),
                    &mut GamgKernelTiming::default(),
                )
                .unwrap();
            assert_report_literal(&actual);
            assert_report_close(&actual, &expected);
            let events = trace.events();
            assert_eq!(
                events
                    .iter()
                    .filter(|event| matches!(event, B2TraceEvent::LduSweep { .. }))
                    .count(),
                2
            );
            assert_eq!(
                events
                    .iter()
                    .filter(|event| matches!(
                        event,
                        B2TraceEvent::CsrSweep {
                            symmetric: true,
                            ..
                        }
                    ))
                    .count(),
                2
            );
            drop(trace);

            let two_controls = GamgSolveControls {
                max_iterations: 2,
                min_iterations: 2,
                tolerance: 0.0,
                relative_tolerance: 0.0,
            };
            let mut production_plain = workspace(&csr, opts, &weights);
            let mut production_plain_timing = GamgKernelTiming::default();
            let production_plain_report = production_plain
                .solve_with_controls_internal::<false, false>(
                    &csr,
                    &rhs(),
                    Some(&initial()),
                    two_controls,
                    &mut production_plain_timing,
                )
                .expect("plain production L2 solve");
            let mut production_profiled = workspace(&csr, opts, &weights);
            let production_profiled_report = production_profiled
                .solve_with_controls_profiled(&csr, &rhs(), Some(&initial()), two_controls)
                .expect("profiled production L2 solve");

            let mut scanned = workspace(&csr, opts, &weights);
            let mut sentinel = workspace(&csr, opts, &weights);
            let mut scanned_profiled_workspace = workspace(&csr, opts, &weights);
            let mut profiled_sentinel = workspace(&csr, opts, &weights);
            sentinelize_ldu(&mut sentinel);
            sentinelize_ldu(&mut profiled_sentinel);
            let sentinel_before = ldu_states(&sentinel);
            let sentinel_outer = sentinel.ldu_levels.as_ref().map(allocation);
            let profiled_sentinel_before = ldu_states(&profiled_sentinel);
            let profiled_sentinel_outer = profiled_sentinel.ldu_levels.as_ref().map(allocation);
            let mut scanned_timing = GamgKernelTiming::default();
            let mut sentinel_timing = GamgKernelTiming::default();
            let mut scanned_profiled_timing = GamgKernelTiming::default();
            let mut profiled_sentinel_timing = GamgKernelTiming::default();
            let scanned_trace = B2TraceGuard::new();
            let scanned_report = scanned
                .solve_with_controls_internal::<false, true>(
                    &csr,
                    &rhs(),
                    Some(&initial()),
                    two_controls,
                    &mut scanned_timing,
                )
                .expect("plain scanned L2 solve");
            let sentinel_report = sentinel
                .solve_with_controls_internal::<false, true>(
                    &csr,
                    &rhs(),
                    Some(&initial()),
                    two_controls,
                    &mut sentinel_timing,
                )
                .expect("plain sentinel scanned L2 solve");
            let scanned_profiled = scanned_profiled_workspace
                .solve_with_controls_internal::<true, true>(
                    &csr,
                    &rhs(),
                    Some(&initial()),
                    two_controls,
                    &mut scanned_profiled_timing,
                )
                .expect("profiled scanned L2 solve");
            let sentinel_profiled = profiled_sentinel
                .solve_with_controls_internal::<true, true>(
                    &csr,
                    &rhs(),
                    Some(&initial()),
                    two_controls,
                    &mut profiled_sentinel_timing,
                )
                .expect("profiled sentinel scanned L2 solve");
            let scanned_events = scanned_trace.events();
            assert_scanned_solve_suffix(&scanned_events, true, 4, 2);
            drop(scanned_trace);

            for report in [
                &production_plain_report,
                &production_profiled_report.report,
                &scanned_report,
                &sentinel_report,
                &scanned_profiled,
                &sentinel_profiled,
            ] {
                assert_eq!(report.iterations, 2);
            }
            assert_report_close(&production_plain_report, &scanned_report);
            assert_report_close(&production_profiled_report.report, &scanned_profiled);
            assert_report_bits_equal(&sentinel_report, &scanned_report);
            assert_report_bits_equal(&sentinel_profiled, &scanned_profiled);
            assert_eq!(ldu_states(&sentinel), sentinel_before);
            assert_eq!(sentinel.ldu_levels.as_ref().map(allocation), sentinel_outer);
            assert_eq!(ldu_states(&profiled_sentinel), profiled_sentinel_before);
            assert_eq!(
                profiled_sentinel.ldu_levels.as_ref().map(allocation),
                profiled_sentinel_outer
            );
            assert_eq!(
                timing_signature(&sentinel_timing),
                timing_signature(&scanned_timing)
            );
            assert_eq!(
                timing_signature(&profiled_sentinel_timing),
                timing_signature(&scanned_profiled_timing)
            );
            assert_eq!(
                timing_signature(&production_plain_timing),
                timing_signature(&scanned_timing)
            );
            assert_eq!(
                timing_signature(&production_profiled_report.timing),
                timing_signature(&scanned_profiled_timing)
            );
        }

        #[test]
        fn ldu_b2_public_normalized_l1_entrypoints_preserve_lifecycle_and_profile() {
            let matrix_a = matrix(1.0);
            let matrix_b = matrix(1.125);
            let weights = face_weights();
            assert_complete_face_weights(&matrix_a, &weights);
            let mut all_events = Vec::new();
            let mut configurations = 0usize;

            for direct in [false, true] {
                for cache in [false, true] {
                    for agglomerator in [
                        GamgAgglomerator::AlgebraicPair,
                        GamgAgglomerator::FaceAreaPair,
                    ] {
                        configurations += 1;
                        let opts =
                            options(direct, cache, agglomerator, GamgSmoother::SymGaussSeidel);
                        let trace = B2TraceGuard::new();
                        let mut plain_l2 = workspace(&matrix_a, opts, &weights);
                        let mut profiled_l2 = workspace(&matrix_a, opts, &weights);
                        let mut plain_l1 = workspace(&matrix_a, opts, &weights);
                        let mut profiled_l1 = workspace(&matrix_a, opts, &weights);

                        let plain_l2_a = plain_l2
                            .solve_with_controls(&matrix_a, &rhs(), Some(&initial()), controls())
                            .unwrap();
                        let plain_l2_b = plain_l2
                            .solve_with_controls(&matrix_b, &rhs(), Some(&initial()), controls())
                            .unwrap();
                        let profiled_l2_a = profiled_l2
                            .solve_with_controls_profiled(
                                &matrix_a,
                                &rhs(),
                                Some(&initial()),
                                controls(),
                            )
                            .unwrap();
                        let profiled_l2_b = profiled_l2
                            .solve_with_controls_profiled(
                                &matrix_b,
                                &rhs(),
                                Some(&initial()),
                                controls(),
                            )
                            .unwrap();
                        let plain_l1_a = plain_l1
                            .solve_normalized_l1_with_controls(
                                &matrix_a,
                                &rhs(),
                                Some(&initial()),
                                normalized_controls(),
                            )
                            .unwrap();
                        let plain_l1_b = plain_l1
                            .solve_normalized_l1_with_controls(
                                &matrix_b,
                                &rhs(),
                                Some(&initial()),
                                normalized_controls(),
                            )
                            .unwrap();
                        let profiled_l1_a = profiled_l1
                            .solve_normalized_l1_with_controls_profiled(
                                &matrix_a,
                                &rhs(),
                                Some(&initial()),
                                normalized_controls(),
                            )
                            .unwrap();
                        let profiled_l1_b = profiled_l1
                            .solve_normalized_l1_with_controls_profiled(
                                &matrix_b,
                                &rhs(),
                                Some(&initial()),
                                normalized_controls(),
                            )
                            .unwrap();

                        for report in [
                            &plain_l2_a,
                            &plain_l2_b,
                            &profiled_l2_a.report,
                            &profiled_l2_b.report,
                            &plain_l1_a,
                            &plain_l1_b,
                            &profiled_l1_a.report,
                            &profiled_l1_b.report,
                        ] {
                            assert_report_literal(report);
                        }
                        for report in [&profiled_l2_a.report, &plain_l1_a, &profiled_l1_a.report] {
                            assert_report_bits_equal(report, &plain_l2_a);
                        }
                        for report in [&profiled_l2_b.report, &plain_l1_b, &profiled_l1_b.report] {
                            assert_report_bits_equal(report, &plain_l2_b);
                        }
                        assert_ne!(
                            plain_l2_a
                                .solution
                                .iter()
                                .map(|value| value.to_bits())
                                .collect::<Vec<_>>(),
                            plain_l2_b
                                .solution
                                .iter()
                                .map(|value| value.to_bits())
                                .collect::<Vec<_>>()
                        );
                        assert_profile_literal(&profiled_l2_a.timing, 0);
                        assert_profile_literal(&profiled_l2_b.timing, usize::from(!cache));
                        assert_profile_literal(&profiled_l1_a.timing, 0);
                        assert_profile_literal(&profiled_l1_b.timing, usize::from(!cache));
                        for workspace in [&plain_l2, &profiled_l2, &plain_l1, &profiled_l1] {
                            assert!(workspace.has_solved);
                            assert_eq!(workspace.level_sizes(), [8, 4, 2]);
                            assert_ldu_values(workspace);
                        }

                        let events = trace.events();
                        assert_configuration_trace(&events, direct, cache);
                        all_events.extend(events);
                    }
                }
            }

            assert_eq!(configurations, 8);
            let public_calls = all_events
                .iter()
                .filter_map(|event| match event {
                    B2TraceEvent::PublicCall(kind) => Some(*kind),
                    _ => None,
                })
                .collect::<Vec<_>>();
            assert_eq!(public_calls.len(), 64);
            for kind in [
                B2PublicCallKind::L2Plain,
                B2PublicCallKind::L2Profiled,
                B2PublicCallKind::NormalizedL1Plain,
                B2PublicCallKind::NormalizedL1Profiled,
            ] {
                assert_eq!(
                    public_calls
                        .iter()
                        .filter(|actual| **actual == kind)
                        .count(),
                    16
                );
            }
            assert_eq!(
                all_events
                    .iter()
                    .filter(|event| matches!(event, B2TraceEvent::HierarchyRebuild))
                    .count(),
                16
            );
            assert_eq!(
                all_events
                    .iter()
                    .filter(|event| matches!(event, B2TraceEvent::LduBuild { .. }))
                    .count(),
                144
            );
            for expected in [
                B2TraceEvent::CsrRefresh { level: usize::MAX },
                B2TraceEvent::DiagonalRefresh { level: usize::MAX },
                B2TraceEvent::LduRefresh { level: usize::MAX },
            ] {
                let count = all_events
                    .iter()
                    .filter(|event| {
                        matches!(
                            (event, expected),
                            (
                                B2TraceEvent::CsrRefresh { .. },
                                B2TraceEvent::CsrRefresh { .. }
                            ) | (
                                B2TraceEvent::DiagonalRefresh { .. },
                                B2TraceEvent::DiagonalRefresh { .. }
                            ) | (
                                B2TraceEvent::LduRefresh { .. },
                                B2TraceEvent::LduRefresh { .. }
                            )
                        )
                    })
                    .count();
                assert_eq!(count, 192);
            }
            assert_eq!(
                all_events
                    .iter()
                    .filter(|event| matches!(event, B2TraceEvent::LduSweep { .. }))
                    .count(),
                128
            );
            assert!(
                !all_events
                    .iter()
                    .any(|event| matches!(event, B2TraceEvent::CsrSweep { .. }))
            );
            assert_eq!(
                all_events
                    .iter()
                    .filter(|event| matches!(event, B2TraceEvent::CoarseDirect { level: 2 }))
                    .count(),
                32
            );
            assert_eq!(
                all_events
                    .iter()
                    .filter(|event| matches!(event, B2TraceEvent::CoarsePcg { level: 2 }))
                    .count(),
                32
            );

            let opts = options(
                true,
                true,
                GamgAgglomerator::AlgebraicPair,
                GamgSmoother::SymGaussSeidel,
            );
            let two_l2 = GamgSolveControls {
                max_iterations: 2,
                min_iterations: 2,
                tolerance: 0.0,
                relative_tolerance: 0.0,
            };
            let two_normalized = NormalizedL1GamgSolveControls {
                normalization_factor: 1.0,
                tolerance: 0.0,
                relative_tolerance: 0.0,
                l2_controls: two_l2,
            };
            let mut production_plain = workspace(&matrix_a, opts, &weights);
            let mut production_plain_timing = GamgKernelTiming::default();
            let production_plain_report = production_plain
                .solve_normalized_l1_with_controls_internal::<false, false>(
                    &matrix_a,
                    &rhs(),
                    Some(&initial()),
                    two_normalized,
                    &mut production_plain_timing,
                )
                .expect("plain production normalized-L1 solve");
            let mut production_profiled = workspace(&matrix_a, opts, &weights);
            let production_profiled_report = production_profiled
                .solve_normalized_l1_with_controls_profiled(
                    &matrix_a,
                    &rhs(),
                    Some(&initial()),
                    two_normalized,
                )
                .expect("profiled production normalized-L1 solve");

            let mut plain = workspace(&matrix_a, opts, &weights);
            let mut plain_sentinel = workspace(&matrix_a, opts, &weights);
            let mut profiled = workspace(&matrix_a, opts, &weights);
            let mut profiled_sentinel = workspace(&matrix_a, opts, &weights);
            sentinelize_ldu(&mut plain_sentinel);
            sentinelize_ldu(&mut profiled_sentinel);
            let plain_sentinel_before = ldu_states(&plain_sentinel);
            let plain_sentinel_outer = plain_sentinel.ldu_levels.as_ref().map(allocation);
            let profiled_sentinel_before = ldu_states(&profiled_sentinel);
            let profiled_sentinel_outer = profiled_sentinel.ldu_levels.as_ref().map(allocation);
            let mut plain_timing = GamgKernelTiming::default();
            let mut plain_sentinel_timing = GamgKernelTiming::default();
            let mut profiled_timing = GamgKernelTiming::default();
            let mut profiled_sentinel_timing = GamgKernelTiming::default();
            let scanned_trace = B2TraceGuard::new();
            let plain_report = plain
                .solve_normalized_l1_with_controls_internal::<false, true>(
                    &matrix_a,
                    &rhs(),
                    Some(&initial()),
                    two_normalized,
                    &mut plain_timing,
                )
                .expect("plain scanned normalized-L1 solve");
            let plain_sentinel_report = plain_sentinel
                .solve_normalized_l1_with_controls_internal::<false, true>(
                    &matrix_a,
                    &rhs(),
                    Some(&initial()),
                    two_normalized,
                    &mut plain_sentinel_timing,
                )
                .expect("plain sentinel scanned normalized-L1 solve");
            let profiled_report = profiled
                .solve_normalized_l1_with_controls_internal::<true, true>(
                    &matrix_a,
                    &rhs(),
                    Some(&initial()),
                    two_normalized,
                    &mut profiled_timing,
                )
                .expect("profiled scanned normalized-L1 solve");
            let profiled_sentinel_report = profiled_sentinel
                .solve_normalized_l1_with_controls_internal::<true, true>(
                    &matrix_a,
                    &rhs(),
                    Some(&initial()),
                    two_normalized,
                    &mut profiled_sentinel_timing,
                )
                .expect("profiled sentinel scanned normalized-L1 solve");
            let scanned_events = scanned_trace.events();
            assert_scanned_solve_suffix(&scanned_events, true, 4, 2);
            drop(scanned_trace);

            for report in [
                &production_plain_report,
                &production_profiled_report.report,
                &plain_report,
                &plain_sentinel_report,
                &profiled_report,
                &profiled_sentinel_report,
            ] {
                assert_eq!(report.iterations, 2);
            }
            assert_report_close(&production_plain_report, &plain_report);
            assert_report_close(&production_profiled_report.report, &profiled_report);
            assert_report_bits_equal(&plain_sentinel_report, &plain_report);
            assert_report_bits_equal(&profiled_report, &plain_report);
            assert_report_bits_equal(&profiled_sentinel_report, &profiled_report);
            assert_eq!(ldu_states(&plain_sentinel), plain_sentinel_before);
            assert_eq!(
                plain_sentinel.ldu_levels.as_ref().map(allocation),
                plain_sentinel_outer
            );
            assert_eq!(
                timing_signature(&plain_sentinel_timing),
                timing_signature(&plain_timing)
            );
            assert_eq!(ldu_states(&profiled_sentinel), profiled_sentinel_before);
            assert_eq!(
                profiled_sentinel.ldu_levels.as_ref().map(allocation),
                profiled_sentinel_outer
            );
            assert_eq!(
                timing_signature(&profiled_sentinel_timing),
                timing_signature(&profiled_timing)
            );
            assert_eq!(
                timing_signature(&production_plain_timing),
                timing_signature(&plain_timing)
            );
            assert_eq!(
                timing_signature(&production_profiled_report.timing),
                timing_signature(&profiled_timing)
            );
        }

        #[test]
        fn ldu_b2_rejects_invalid_dimensions_diagonals_and_nonfinite_updates() {
            let csr = matrix(1.0);
            let weights = face_weights();
            let mut workspace = workspace(
                &csr,
                options(
                    true,
                    true,
                    GamgAgglomerator::AlgebraicPair,
                    GamgSmoother::SymGaussSeidel,
                ),
                &weights,
            );
            let valid_rhs = rhs().to_vec();
            let mut assert_workspace_failure =
                |bad_matrix: &CsrMatrix,
                 bad_rhs: Vec<f64>,
                 initial: Option<Vec<f64>>,
                 expected: &str| {
                    let before = workspace_state(&workspace);
                    let error = workspace
                        .solve_with_controls(bad_matrix, &bad_rhs, initial.as_deref(), controls())
                        .expect_err("invalid workspace input must fail");
                    assert_invalid(error, expected);
                    assert_eq!(workspace_state(&workspace), before);
                };

            assert_workspace_failure(
                &csr,
                vec![1.0],
                None,
                "iterative solve expected rhs with 8 entries, got 1",
            );
            assert_workspace_failure(
                &csr,
                valid_rhs.clone(),
                Some(vec![0.0]),
                "iterative solve expected initial guess with 8 entries, got 1",
            );

            let diagonal_slot = (csr.row_offsets()[0]..csr.row_offsets()[1])
                .find(|&slot| csr.col_indices()[slot] == 0)
                .unwrap();
            let mut zero_diagonal = csr.clone();
            zero_diagonal.values_mut()[diagonal_slot] = 0.0;
            assert_workspace_failure(
                &zero_diagonal,
                valid_rhs.clone(),
                None,
                "GAMG row 0 has invalid diagonal value 0",
            );
            let mut nan_diagonal = csr.clone();
            nan_diagonal.values_mut()[diagonal_slot] = f64::NAN;
            assert_workspace_failure(
                &nan_diagonal,
                valid_rhs.clone(),
                None,
                "GAMG row 0 has invalid diagonal value NaN",
            );
            let mut infinite_diagonal = csr.clone();
            infinite_diagonal.values_mut()[diagonal_slot] = f64::INFINITY;
            assert_workspace_failure(
                &infinite_diagonal,
                valid_rhs.clone(),
                None,
                "GAMG row 0 has invalid diagonal value inf",
            );
            let mut negative_infinite_diagonal = csr.clone();
            negative_infinite_diagonal.values_mut()[diagonal_slot] = f64::NEG_INFINITY;
            assert_workspace_failure(
                &negative_infinite_diagonal,
                valid_rhs.clone(),
                None,
                "GAMG row 0 has invalid diagonal value -inf",
            );
            let source_rows = (0..csr.rows())
                .map(|row| {
                    (csr.row_offsets()[row]..csr.row_offsets()[row + 1])
                        .map(|slot| (csr.col_indices()[slot], csr.values()[slot]))
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let mut missing_rows = source_rows.clone();
            missing_rows[0].retain(|(column, _)| *column != 0);
            let missing_diagonal = CsrMatrix::from_rows(missing_rows, 8).unwrap();
            assert_workspace_failure(
                &missing_diagonal,
                valid_rhs.clone(),
                None,
                "GAMG row 0 must have exactly one diagonal entry, got 0",
            );
            let mut duplicate_rows = source_rows;
            duplicate_rows[0].push((0, 1.0));
            let duplicate_diagonal = CsrMatrix::from_rows(duplicate_rows, 8).unwrap();
            assert_workspace_failure(
                &duplicate_diagonal,
                valid_rhs.clone(),
                None,
                "GAMG row 0 must have exactly one diagonal entry, got 2",
            );
            let diagonal_only =
                CsrMatrix::from_rows((0..8).map(|row| vec![(row, 1.0)]).collect(), 8).unwrap();
            assert_workspace_failure(
                &diagonal_only,
                valid_rhs.clone(),
                None,
                "GAMG workspace does not match matrix sparsity",
            );
            workspace
                .solve_with_controls(&csr, &valid_rhs, Some(&initial()), controls())
                .expect("valid retry after pre-mutation failures");

            let direct_matrix = chain_matrix(2, 1.0);
            let diagonal = vec![4.0; 2];
            let direct_rhs = vec![1.0, -0.5];
            let direct_initial = vec![0.25, -0.125];
            let direct_failure = |mut level: GamgLduLevel,
                                  bad_diagonal: Vec<f64>,
                                  bad_rhs: Vec<f64>,
                                  mut solution: Vec<f64>,
                                  expected: &str| {
                let level_before = ldu_state(&level);
                let solution_before = solution
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>();
                let error =
                    ldu_sym_gauss_seidel_sweep(&mut level, &bad_diagonal, &bad_rhs, &mut solution)
                        .expect_err("invalid direct LDU input must fail");
                assert_invalid(error, expected);
                assert_eq!(ldu_state(&level), level_before);
                assert_eq!(
                    solution
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>(),
                    solution_before
                );
            };
            direct_failure(
                GamgLduLevel::new(&direct_matrix),
                vec![4.0],
                direct_rhs.clone(),
                direct_initial.clone(),
                "GAMG symGaussSeidel LDU expected 2 diagonal entries, got 1",
            );
            direct_failure(
                GamgLduLevel::new(&direct_matrix),
                diagonal.clone(),
                vec![1.0],
                direct_initial.clone(),
                "GAMG symGaussSeidel LDU expected 2 rhs entries, got 1",
            );
            direct_failure(
                GamgLduLevel::new(&direct_matrix),
                diagonal.clone(),
                direct_rhs.clone(),
                vec![0.0],
                "GAMG symGaussSeidel LDU expected 2 solution entries, got 1",
            );
            direct_failure(
                GamgLduLevel::new(&direct_matrix),
                vec![0.0, 4.0],
                direct_rhs.clone(),
                direct_initial.clone(),
                "GAMG symGaussSeidel LDU diagonal row 0 must be finite and non-zero, got 0",
            );
            direct_failure(
                GamgLduLevel::new(&direct_matrix),
                vec![f64::NAN, 4.0],
                direct_rhs.clone(),
                direct_initial.clone(),
                "GAMG symGaussSeidel LDU diagonal row 0 must be finite and non-zero, got NaN",
            );
            direct_failure(
                GamgLduLevel::new(&direct_matrix),
                diagonal.clone(),
                vec![f64::INFINITY, 0.0],
                direct_initial.clone(),
                "GAMG symGaussSeidel LDU rhs entry 0 must be finite, got inf",
            );
            direct_failure(
                GamgLduLevel::new(&direct_matrix),
                diagonal.clone(),
                direct_rhs.clone(),
                vec![f64::NAN, 0.0],
                "GAMG symGaussSeidel LDU solution entry 0 must be finite, got NaN",
            );

            let mut malformed = GamgLduLevel::new(&direct_matrix);
            malformed.lower_addr.pop();
            direct_failure(
                malformed,
                diagonal.clone(),
                direct_rhs.clone(),
                direct_initial.clone(),
                "GAMG symGaussSeidel LDU expected 1 lowerAddr entries, got 0",
            );
            let mut malformed = GamgLduLevel::new(&direct_matrix);
            malformed.upper_addr.pop();
            direct_failure(
                malformed,
                diagonal.clone(),
                direct_rhs.clone(),
                direct_initial.clone(),
                "GAMG symGaussSeidel LDU expected 1 upperAddr entries, got 0",
            );
            let mut malformed = GamgLduLevel::new(&direct_matrix);
            malformed.lower_csr.pop();
            direct_failure(
                malformed,
                diagonal.clone(),
                direct_rhs.clone(),
                direct_initial.clone(),
                "GAMG symGaussSeidel LDU expected 1 lowerCsr entries, got 0",
            );
            let mut malformed = GamgLduLevel::new(&direct_matrix);
            malformed.upper_csr.pop();
            direct_failure(
                malformed,
                diagonal.clone(),
                direct_rhs.clone(),
                direct_initial.clone(),
                "GAMG symGaussSeidel LDU expected 1 upperCsr entries, got 0",
            );
            let mut malformed = GamgLduLevel::new(&direct_matrix);
            malformed.upper.pop();
            direct_failure(
                malformed,
                diagonal.clone(),
                direct_rhs.clone(),
                direct_initial.clone(),
                "GAMG symGaussSeidel LDU expected 1 upper entries, got 0",
            );
            let mut malformed = GamgLduLevel::new(&direct_matrix);
            malformed.owner_start.pop();
            direct_failure(
                malformed,
                diagonal.clone(),
                direct_rhs.clone(),
                direct_initial.clone(),
                "GAMG symGaussSeidel LDU expected 3 ownerStart entries, got 2",
            );
            let mut malformed = GamgLduLevel::new(&direct_matrix);
            malformed.owner_start[0] = 1;
            direct_failure(
                malformed,
                diagonal.clone(),
                direct_rhs.clone(),
                direct_initial.clone(),
                "GAMG symGaussSeidel LDU ownerStart must begin at 0",
            );
            let mut malformed = GamgLduLevel::new(&direct_matrix);
            *malformed.owner_start.last_mut().unwrap() = 0;
            direct_failure(
                malformed,
                diagonal.clone(),
                direct_rhs.clone(),
                direct_initial.clone(),
                "GAMG symGaussSeidel LDU ownerStart must end at 1, got 0",
            );
            let mut malformed = GamgLduLevel::new(&direct_matrix);
            malformed.owner_start[1] = 2;
            direct_failure(
                malformed,
                diagonal.clone(),
                direct_rhs.clone(),
                direct_initial.clone(),
                "GAMG symGaussSeidel LDU ownerStart row 0 has invalid range 0..2 for 1 faces",
            );
            let mut malformed = GamgLduLevel::new(&direct_matrix);
            malformed.lower_addr[0] = 1;
            direct_failure(
                malformed,
                diagonal.clone(),
                direct_rhs.clone(),
                direct_initial.clone(),
                "GAMG symGaussSeidel LDU face 0 has owner 1, expected 0",
            );
            let mut malformed = GamgLduLevel::new(&direct_matrix);
            malformed.upper_addr[0] = 0;
            direct_failure(
                malformed,
                diagonal.clone(),
                direct_rhs.clone(),
                direct_initial.clone(),
                "GAMG symGaussSeidel LDU face 0 has invalid owner/neighbour 0/0 for 2 rows",
            );
            let mut malformed = GamgLduLevel::new(&direct_matrix);
            malformed.lower[0] = f64::NAN;
            direct_failure(
                malformed,
                diagonal.clone(),
                direct_rhs.clone(),
                direct_initial.clone(),
                "GAMG symGaussSeidel LDU lower coefficient 0 must be finite, got NaN",
            );
            let mut malformed = GamgLduLevel::new(&direct_matrix);
            malformed.upper[0] = f64::INFINITY;
            direct_failure(
                malformed,
                diagonal.clone(),
                direct_rhs.clone(),
                direct_initial.clone(),
                "GAMG symGaussSeidel LDU upper coefficient 0 must be finite, got inf",
            );
            let mut malformed = GamgLduLevel::new(&direct_matrix);
            malformed.b_prime[0] = f64::INFINITY;
            let malformed_before = ldu_state(&malformed);
            let mut half_solution = direct_initial.clone();
            let half_before = half_solution.clone();
            let error =
                ldu_sym_gauss_seidel_half(&mut malformed, &diagonal, &mut half_solution, 0..2)
                    .expect_err("nonfinite bPrime must fail before half sweep");
            assert_invalid(
                error,
                "GAMG symGaussSeidel LDU bPrime entry 0 must be finite, got inf",
            );
            assert_eq!(ldu_state(&malformed), malformed_before);
            assert_eq!(half_solution, half_before);

            let overflow_matrix = CsrMatrix::from_rows(
                vec![
                    vec![(0, 4.0), (1, -1.0)],
                    vec![(0, -1.0), (1, f64::MIN_POSITIVE), (2, -1.0)],
                    vec![(1, -1.0), (2, 4.0)],
                ],
                3,
            )
            .unwrap();
            let mut level = GamgLduLevel::new(&overflow_matrix);
            let pinned = allocations(&level);
            let level_before = ldu_state(&level);
            let mut solution = vec![0.0; 3];
            let error = ldu_sym_gauss_seidel_sweep(
                &mut level,
                &[4.0, f64::MIN_POSITIVE, 4.0],
                &[1.0, 8.0, 0.0],
                &mut solution,
            )
            .expect_err("finite nonfinite-update fixture must fail");
            assert_invalid(error, "Gauss-Seidel update for row 1 is not finite");
            assert_eq!(
                solution
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                vec![0.25f64.to_bits(), 0.0f64.to_bits(), 0.0f64.to_bits()]
            );
            assert_eq!(allocations(&level), pinned);
            assert_eq!(
                level
                    .b_prime
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                vec![1.0f64.to_bits(), 8.25f64.to_bits(), 0.0f64.to_bits()]
            );
            let level_after = ldu_state(&level);
            assert_eq!(level_after.lower_addr, level_before.lower_addr);
            assert_eq!(level_after.upper_addr, level_before.upper_addr);
            assert_eq!(level_after.lower_csr, level_before.lower_csr);
            assert_eq!(level_after.upper_csr, level_before.upper_csr);
            assert_eq!(level_after.lower, level_before.lower);
            assert_eq!(level_after.upper, level_before.upper);
            assert_eq!(level_after.owner_start, level_before.owner_start);

            let mut healed = vec![0.0; 3];
            ldu_sym_gauss_seidel_sweep(
                &mut level,
                &[4.0, f64::MIN_POSITIVE, 4.0],
                &[1.0, -0.25, 0.0],
                &mut healed,
            )
            .expect("direct LDU retry must heal");
            let mut fresh_level = GamgLduLevel::new(&overflow_matrix);
            let mut expected = vec![0.0; 3];
            ldu_sym_gauss_seidel_sweep(
                &mut fresh_level,
                &[4.0, f64::MIN_POSITIVE, 4.0],
                &[1.0, -0.25, 0.0],
                &mut expected,
            )
            .unwrap();
            assert_eq!(
                healed
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                expected
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>()
            );
            assert_eq!(allocations(&level), pinned);
        }

        #[test]
        fn ldu_b2_rejects_nonsymmetric_matrix_before_workspace_state_changes() {
            let csr = matrix(1.0);
            let weights = face_weights();
            let mut workspace = workspace(
                &csr,
                options(
                    true,
                    true,
                    GamgAgglomerator::AlgebraicPair,
                    GamgSmoother::SymGaussSeidel,
                ),
                &weights,
            );
            let before = workspace_state(&workspace);
            let mut bad = csr.clone();
            bad.values_mut()[1] = -2.0;
            let error = workspace
                .solve_with_controls(&bad, &rhs(), None, controls())
                .expect_err("nonsymmetric matrix must fail before mutation");
            assert_invalid(
                error,
                "GAMG pressure foundation requires a symmetric matrix; A[0,1]=-2 differs from A[1,0]=-1",
            );
            assert_eq!(workspace_state(&workspace), before);
            workspace
                .solve_with_controls(&csr, &rhs(), Some(&initial()), controls())
                .expect("valid retry after symmetry failure");
        }

        #[test]
        fn ldu_b2_preserves_singular_and_refresh_failure_contracts() {
            let weights = face_weights();
            let good = chain_matrix(4, 1.0);
            let singular = singular_matrix();
            let singular_rhs = [0.0; 4];
            let l2 = controls();
            let normalized = normalized_controls();
            for direct in [true, false] {
                for use_normalized_l1 in [false, true] {
                    let opts = options(
                        direct,
                        true,
                        GamgAgglomerator::AlgebraicPair,
                        GamgSmoother::SymGaussSeidel,
                    );
                    let mut workspace = workspace(&good, opts, &weights);
                    let pinned = allocation_signature(&workspace);
                    let mut timing = GamgKernelTiming::default();
                    let error = if use_normalized_l1 {
                        workspace
                            .solve_normalized_l1_with_controls_internal::<true, false>(
                                &singular,
                                &singular_rhs,
                                None,
                                normalized,
                                &mut timing,
                            )
                            .expect_err("singular normalized-L1 solve must fail")
                    } else {
                        workspace
                            .solve_with_controls_internal::<true, false>(
                                &singular,
                                &singular_rhs,
                                None,
                                l2,
                                &mut timing,
                            )
                            .expect_err("singular L2 solve must fail")
                    };
                    let expected = if direct {
                        "GAMG direct coarsest solve has a singular pivot in column 1"
                    } else {
                        "incomplete Cholesky preconditioner row 1 has non-positive pivot square 0"
                    };
                    assert_invalid(error, expected);
                    assert_eq!(allocation_signature(&workspace), pinned);
                    assert!(!workspace.has_solved);
                    let failed_state = without_allocation_addresses(workspace_state(&workspace));
                    let failed_timing = timing_signature(&timing);

                    let mut repeated = self::workspace(&good, opts, &weights);
                    let repeated_pinned = allocation_signature(&repeated);
                    let mut repeated_timing = GamgKernelTiming::default();
                    let repeated_error = if use_normalized_l1 {
                        repeated
                            .solve_normalized_l1_with_controls_internal::<true, false>(
                                &singular,
                                &singular_rhs,
                                None,
                                normalized,
                                &mut repeated_timing,
                            )
                            .expect_err("repeated singular normalized-L1 solve must fail")
                    } else {
                        repeated
                            .solve_with_controls_internal::<true, false>(
                                &singular,
                                &singular_rhs,
                                None,
                                l2,
                                &mut repeated_timing,
                            )
                            .expect_err("repeated singular L2 solve must fail")
                    };
                    assert_invalid(repeated_error, expected);
                    assert_eq!(allocation_signature(&repeated), repeated_pinned);
                    assert!(!repeated.has_solved);
                    assert_eq!(
                        without_allocation_addresses(workspace_state(&repeated)),
                        failed_state
                    );
                    assert_eq!(timing_signature(&repeated_timing), failed_timing);
                    let retry = workspace
                        .solve_with_controls(&good, &[1.0; 4], None, l2)
                        .expect("valid retry after singular failure");
                    assert_eq!(retry.iterations, 1);
                    assert_eq!(allocation_signature(&workspace), pinned);
                }
            }

            let good = chain_matrix(16, 1.0);
            let overflow = chain_matrix(16, f64::MAX / 5.0);
            assert!(overflow.values().iter().all(|value| value.is_finite()));
            let opts = options(
                true,
                true,
                GamgAgglomerator::AlgebraicPair,
                GamgSmoother::SymGaussSeidel,
            );
            let mut workspace = workspace(&good, opts, &weights);
            assert!(workspace.matrices.len() >= 3);
            let before_matrix = matrix_value_bits(&workspace);
            let before_diagonal = diagonal_value_bits(&workspace);
            let before_ldu = ldu_states(&workspace);
            let pinned = allocation_signature(&workspace);
            let pinned_outer = workspace.ldu_levels.as_ref().map(allocation);
            let mut timing = GamgKernelTiming::default();
            let error = workspace
                .solve_with_controls_internal::<true, false>(
                    &overflow,
                    &[1.0; 16],
                    None,
                    controls(),
                    &mut timing,
                )
                .expect_err("finite same-sparsity coarse overflow must fail");
            assert_invalid(error, "GAMG row 0 has invalid diagonal value inf");
            assert_eq!(allocation_signature(&workspace), pinned);
            assert_eq!(workspace.ldu_levels.as_ref().map(allocation), pinned_outer);
            assert_eq!(ldu_states(&workspace), before_ldu);
            assert_eq!(
                workspace.matrices[0]
                    .values()
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                overflow
                    .values()
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>()
            );
            assert!(
                workspace.matrices[1]
                    .values()
                    .iter()
                    .any(|value| !value.is_finite())
            );
            assert_eq!(
                workspace.diagonal_values[0]
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                (0..16)
                    .map(|row| overflow.values()[workspace.diagonal_slots[0][row]].to_bits())
                    .collect::<Vec<_>>()
            );
            assert_eq!(
                workspace.diagonal_values[1]
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                before_diagonal[1]
            );
            assert_eq!(matrix_value_bits(&workspace)[2..], before_matrix[2..]);
            assert_eq!(diagonal_value_bits(&workspace)[2..], before_diagonal[2..]);
            assert_eq!(timing.matrix_refreshes, 0);
            assert_eq!(timing.levels[0].matrix_refreshes, 1);
            assert_eq!(timing.levels[1].matrix_refreshes, 0);
            assert!(
                timing.levels[2..]
                    .iter()
                    .all(|level| level.matrix_refreshes == 0)
            );
            assert!(!workspace.has_solved);

            let healed_matrix = chain_matrix(16, 1.25);
            let healed_rhs = (0..16)
                .map(|row| 1.0 + row as f64 / 41.0)
                .collect::<Vec<_>>();
            let healed = workspace
                .solve_with_controls(&healed_matrix, &healed_rhs, None, controls())
                .expect("same workspace must heal after coarse overflow");
            assert_eq!(allocation_signature(&workspace), pinned);
            assert_eq!(workspace.ldu_levels.as_ref().map(allocation), pinned_outer);
            let mut fresh = self::workspace(&healed_matrix, opts, &weights);
            let expected = fresh
                .solve_with_controls(&healed_matrix, &healed_rhs, None, controls())
                .unwrap();
            assert_report_bits_equal(&healed, &expected);
            assert_eq!(matrix_value_bits(&workspace), matrix_value_bits(&fresh));
            assert_eq!(diagonal_value_bits(&workspace), diagonal_value_bits(&fresh));
            assert_eq!(
                ldu_semantic_signature(&workspace),
                ldu_semantic_signature(&fresh)
            );
        }
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
