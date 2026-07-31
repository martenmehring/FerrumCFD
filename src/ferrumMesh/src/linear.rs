use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Instant;

use crate::{MeshError, Result};

pub mod gamg;
pub use gamg::{
    GamgAgglomerator, GamgAggregateSizeBin, GamgFacePairWeight, GamgHierarchyDiagnostics,
    GamgHierarchyLevelDiagnostics, GamgKernelTiming, GamgLevelTiming, GamgOptions, GamgOuterSolver,
    GamgSmoother, GamgSolveControls, GamgTransferDiagnostics, GamgWorkspace,
    ProfiledGamgSolveReport, gamg_solve,
};

#[derive(Clone, Debug)]
pub struct CsrMatrix {
    rows: usize,
    cols: usize,
    row_offsets: Arc<[usize]>,
    col_indices: Arc<[usize]>,
    values: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct CsrSparsityPattern {
    rows: usize,
    cols: usize,
    row_offsets: Arc<[usize]>,
    col_indices: Arc<[usize]>,
}

#[derive(Clone, Copy, Debug)]
pub struct LinearSolverCapabilities {
    pub cpu_csr: bool,
    pub cpu_jacobi: bool,
    pub cpu_gauss_seidel: bool,
    pub cpu_symmetric_gauss_seidel: bool,
    pub cpu_conjugate_gradient: bool,
    pub cpu_preconditioned_conjugate_gradient: bool,
    pub cpu_bicgstab: bool,
    pub cpu_gamg: bool,
    pub cpu_diagonal_preconditioner: bool,
    pub cpu_incomplete_cholesky_preconditioner: bool,
    pub cpu_dic_preconditioner: bool,
    pub cpu_fdic_preconditioner: bool,
    pub gpu_linear_solvers: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct JacobiOptions {
    pub max_iterations: usize,
    pub tolerance: f64,
    pub omega: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct GaussSeidelOptions {
    pub max_iterations: usize,
    pub tolerance: f64,
    pub omega: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct ConjugateGradientOptions {
    pub max_iterations: usize,
    pub tolerance: f64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CgPreconditioner {
    None,
    Diagonal,
    IncompleteCholesky,
    Dic,
    Fdic,
}

#[derive(Clone, Copy, Debug)]
pub struct PreconditionedConjugateGradientOptions {
    pub max_iterations: usize,
    pub tolerance: f64,
    pub preconditioner: CgPreconditioner,
}

#[derive(Clone, Copy, Debug)]
pub struct BiCgStabOptions {
    pub max_iterations: usize,
    pub tolerance: f64,
    pub preconditioner: CgPreconditioner,
}

#[derive(Clone, Debug)]
pub struct IterativeSolveReport {
    pub solution: Vec<f64>,
    pub iterations: usize,
    pub residual_norm: f64,
    pub converged: bool,
    pub termination: IterativeSolveTermination,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IterativeSolveTermination {
    Converged,
    MaxIterations,
    Breakdown,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct PcgKernelTiming {
    pub total_seconds: f64,
    pub preconditioner_update_seconds: f64,
    pub matrix_vector_seconds: f64,
    pub preconditioner_application_seconds: f64,
    pub vector_operation_seconds: f64,
    pub other_seconds: f64,
    pub matrix_vector_products: usize,
    pub preconditioner_applications: usize,
}

#[derive(Clone, Debug)]
pub struct ProfiledIterativeSolveReport {
    pub report: IterativeSolveReport,
    pub timing: PcgKernelTiming,
}

impl CsrMatrix {
    pub fn new(
        rows: usize,
        cols: usize,
        row_offsets: Vec<usize>,
        col_indices: Vec<usize>,
        values: Vec<f64>,
    ) -> Result<Self> {
        let pattern = CsrSparsityPattern::new(rows, cols, row_offsets, col_indices)?;
        Self::from_pattern(&pattern, values)
    }

    pub fn from_pattern(pattern: &CsrSparsityPattern, values: Vec<f64>) -> Result<Self> {
        validate_csr_values(pattern.nnz(), &values)?;
        Ok(Self {
            rows: pattern.rows,
            cols: pattern.cols,
            row_offsets: Arc::clone(&pattern.row_offsets),
            col_indices: Arc::clone(&pattern.col_indices),
            values,
        })
    }

    pub fn from_rows(rows: Vec<Vec<(usize, f64)>>, cols: usize) -> Result<Self> {
        let mut row_offsets = Vec::with_capacity(rows.len() + 1);
        let mut col_indices = Vec::new();
        let mut values = Vec::new();
        row_offsets.push(0);
        for row in &rows {
            for &(col, value) in row {
                col_indices.push(col);
                values.push(value);
            }
            row_offsets.push(col_indices.len());
        }
        Self::new(rows.len(), cols, row_offsets, col_indices, values)
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    pub fn row_offsets(&self) -> &[usize] {
        &self.row_offsets
    }

    pub fn col_indices(&self) -> &[usize] {
        &self.col_indices
    }

    pub fn values(&self) -> &[f64] {
        &self.values
    }

    pub fn values_mut(&mut self) -> &mut [f64] {
        &mut self.values
    }

    pub fn shares_sparsity_with(&self, pattern: &CsrSparsityPattern) -> bool {
        self.rows == pattern.rows
            && self.cols == pattern.cols
            && Arc::ptr_eq(&self.row_offsets, &pattern.row_offsets)
            && Arc::ptr_eq(&self.col_indices, &pattern.col_indices)
    }

    pub fn sparsity_pattern(&self) -> CsrSparsityPattern {
        CsrSparsityPattern {
            rows: self.rows,
            cols: self.cols,
            row_offsets: Arc::clone(&self.row_offsets),
            col_indices: Arc::clone(&self.col_indices),
        }
    }

    pub fn validate_values(&self) -> Result<()> {
        validate_csr_values(self.nnz(), &self.values)
    }

    pub fn matvec(&self, x: &[f64]) -> Result<Vec<f64>> {
        let mut y = vec![0.0; self.rows];
        self.matvec_into(x, &mut y)?;
        Ok(y)
    }

    pub fn matvec_into(&self, x: &[f64], y: &mut [f64]) -> Result<()> {
        if x.len() != self.cols {
            return Err(invalid_input(format!(
                "CSR matvec expected x with {} entries, got {}",
                self.cols,
                x.len()
            )));
        }
        if y.len() != self.rows {
            return Err(invalid_input(format!(
                "CSR matvec expected y with {} entries, got {}",
                self.rows,
                y.len()
            )));
        }

        for (row, output) in y.iter_mut().enumerate() {
            let start = self.row_offsets[row];
            let end = self.row_offsets[row + 1];
            let mut sum = 0.0;
            for entry in start..end {
                sum += self.values[entry] * x[self.col_indices[entry]];
            }
            *output = sum;
        }

        Ok(())
    }

    pub fn diagonal(&self) -> Result<Vec<f64>> {
        if self.rows != self.cols {
            return Err(invalid_input(format!(
                "diagonal extraction requires a square matrix, got {}x{}",
                self.rows, self.cols
            )));
        }

        let mut diagonal = vec![None; self.rows];
        for (row, slot) in diagonal.iter_mut().enumerate() {
            let start = self.row_offsets[row];
            let end = self.row_offsets[row + 1];
            for entry in start..end {
                if self.col_indices[entry] == row {
                    *slot = Some(self.values[entry]);
                    break;
                }
            }
        }

        diagonal
            .into_iter()
            .enumerate()
            .map(|(row, value)| {
                value.ok_or_else(|| invalid_input(format!("row {row} has no diagonal entry")))
            })
            .collect()
    }
}

impl CsrSparsityPattern {
    pub fn new(
        rows: usize,
        cols: usize,
        row_offsets: Vec<usize>,
        col_indices: Vec<usize>,
    ) -> Result<Self> {
        validate_csr_pattern(rows, cols, &row_offsets, &col_indices)?;
        Ok(Self {
            rows,
            cols,
            row_offsets: row_offsets.into(),
            col_indices: col_indices.into(),
        })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn nnz(&self) -> usize {
        self.col_indices.len()
    }

    pub fn row_offsets(&self) -> &[usize] {
        &self.row_offsets
    }

    pub fn col_indices(&self) -> &[usize] {
        &self.col_indices
    }
}

impl Default for JacobiOptions {
    fn default() -> Self {
        Self {
            max_iterations: 1_000,
            tolerance: 1.0e-10,
            omega: 1.0,
        }
    }
}

impl Default for GaussSeidelOptions {
    fn default() -> Self {
        Self {
            max_iterations: 1_000,
            tolerance: 1.0e-10,
            omega: 1.0,
        }
    }
}

impl Default for ConjugateGradientOptions {
    fn default() -> Self {
        Self {
            max_iterations: 1_000,
            tolerance: 1.0e-10,
        }
    }
}

impl Default for PreconditionedConjugateGradientOptions {
    fn default() -> Self {
        Self {
            max_iterations: 1_000,
            tolerance: 1.0e-10,
            preconditioner: CgPreconditioner::Diagonal,
        }
    }
}

impl Default for BiCgStabOptions {
    fn default() -> Self {
        Self {
            max_iterations: 1_000,
            tolerance: 1.0e-10,
            preconditioner: CgPreconditioner::Diagonal,
        }
    }
}

pub fn linear_solver_capabilities() -> LinearSolverCapabilities {
    LinearSolverCapabilities {
        cpu_csr: true,
        cpu_jacobi: true,
        cpu_gauss_seidel: true,
        cpu_symmetric_gauss_seidel: true,
        cpu_conjugate_gradient: true,
        cpu_preconditioned_conjugate_gradient: true,
        cpu_bicgstab: true,
        cpu_gamg: true,
        cpu_diagonal_preconditioner: true,
        cpu_incomplete_cholesky_preconditioner: true,
        cpu_dic_preconditioner: true,
        cpu_fdic_preconditioner: true,
        gpu_linear_solvers: false,
    }
}

pub fn residual(matrix: &CsrMatrix, x: &[f64], rhs: &[f64]) -> Result<Vec<f64>> {
    if rhs.len() != matrix.rows {
        return Err(invalid_input(format!(
            "residual expected rhs with {} entries, got {}",
            matrix.rows,
            rhs.len()
        )));
    }
    let ax = matrix.matvec(x)?;
    Ok(rhs
        .iter()
        .zip(ax)
        .map(|(rhs_value, ax_value)| rhs_value - ax_value)
        .collect())
}

pub fn l2_norm(values: &[f64]) -> f64 {
    dot(values, values).sqrt()
}

pub fn jacobi_solve(
    matrix: &CsrMatrix,
    rhs: &[f64],
    initial: Option<&[f64]>,
    options: JacobiOptions,
) -> Result<IterativeSolveReport> {
    validate_iterative_solve_input(matrix, rhs, initial, options.tolerance)?;
    if !options.omega.is_finite() || options.omega <= 0.0 {
        return Err(invalid_input(format!(
            "Jacobi omega must be positive and finite, got {}",
            options.omega
        )));
    }

    let diagonal = matrix.diagonal()?;
    for (row, value) in diagonal.iter().enumerate() {
        if !value.is_finite() || *value == 0.0 {
            return Err(invalid_input(format!(
                "row {row} has an invalid Jacobi diagonal entry {value}"
            )));
        }
    }

    let mut x = initial
        .map(|values| values.to_vec())
        .unwrap_or_else(|| vec![0.0; rhs.len()]);
    let mut residual_norm = l2_norm(&residual(matrix, &x, rhs)?);
    if residual_norm <= options.tolerance {
        return Ok(IterativeSolveReport {
            solution: x,
            iterations: 0,
            residual_norm,
            converged: true,
            termination: IterativeSolveTermination::Converged,
        });
    }

    let mut next = x.clone();
    for iteration in 1..=options.max_iterations {
        for row in 0..matrix.rows {
            let start = matrix.row_offsets[row];
            let end = matrix.row_offsets[row + 1];
            let mut off_diagonal_sum = 0.0;
            for entry in start..end {
                let col = matrix.col_indices[entry];
                if col != row {
                    off_diagonal_sum += matrix.values[entry] * x[col];
                }
            }
            let raw = (rhs[row] - off_diagonal_sum) / diagonal[row];
            if !raw.is_finite() {
                return Err(invalid_input(format!(
                    "Jacobi update for row {row} is not finite"
                )));
            }
            next[row] = (1.0 - options.omega) * x[row] + options.omega * raw;
        }

        std::mem::swap(&mut x, &mut next);
        residual_norm = l2_norm(&residual(matrix, &x, rhs)?);
        if residual_norm <= options.tolerance {
            return Ok(IterativeSolveReport {
                solution: x,
                iterations: iteration,
                residual_norm,
                converged: true,
                termination: IterativeSolveTermination::Converged,
            });
        }
    }

    Ok(IterativeSolveReport {
        solution: x,
        iterations: options.max_iterations,
        residual_norm,
        converged: false,
        termination: IterativeSolveTermination::MaxIterations,
    })
}

pub fn gauss_seidel_solve(
    matrix: &CsrMatrix,
    rhs: &[f64],
    initial: Option<&[f64]>,
    options: GaussSeidelOptions,
) -> Result<IterativeSolveReport> {
    validate_iterative_solve_input(matrix, rhs, initial, options.tolerance)?;
    if !options.omega.is_finite() || options.omega <= 0.0 {
        return Err(invalid_input(format!(
            "Gauss-Seidel omega must be positive and finite, got {}",
            options.omega
        )));
    }

    let mut x = initial
        .map(|values| values.to_vec())
        .unwrap_or_else(|| vec![0.0; rhs.len()]);
    let mut residual_norm = l2_norm(&residual(matrix, &x, rhs)?);
    if residual_norm <= options.tolerance {
        return Ok(IterativeSolveReport {
            solution: x,
            iterations: 0,
            residual_norm,
            converged: true,
            termination: IterativeSolveTermination::Converged,
        });
    }

    for iteration in 1..=options.max_iterations {
        gauss_seidel_sweep(matrix, rhs, &mut x, options.omega, 0..matrix.rows)?;

        residual_norm = l2_norm(&residual(matrix, &x, rhs)?);
        if residual_norm <= options.tolerance {
            return Ok(IterativeSolveReport {
                solution: x,
                iterations: iteration,
                residual_norm,
                converged: true,
                termination: IterativeSolveTermination::Converged,
            });
        }
    }

    Ok(IterativeSolveReport {
        solution: x,
        iterations: options.max_iterations,
        residual_norm,
        converged: false,
        termination: IterativeSolveTermination::MaxIterations,
    })
}

pub fn symmetric_gauss_seidel_solve(
    matrix: &CsrMatrix,
    rhs: &[f64],
    initial: Option<&[f64]>,
    options: GaussSeidelOptions,
) -> Result<IterativeSolveReport> {
    validate_iterative_solve_input(matrix, rhs, initial, options.tolerance)?;
    if !options.omega.is_finite() || options.omega <= 0.0 {
        return Err(invalid_input(format!(
            "symmetric Gauss-Seidel omega must be positive and finite, got {}",
            options.omega
        )));
    }

    let mut x = initial
        .map(|values| values.to_vec())
        .unwrap_or_else(|| vec![0.0; rhs.len()]);
    let mut residual_norm = l2_norm(&residual(matrix, &x, rhs)?);
    if residual_norm <= options.tolerance {
        return Ok(IterativeSolveReport {
            solution: x,
            iterations: 0,
            residual_norm,
            converged: true,
            termination: IterativeSolveTermination::Converged,
        });
    }

    for iteration in 1..=options.max_iterations {
        gauss_seidel_sweep(matrix, rhs, &mut x, options.omega, 0..matrix.rows)?;
        gauss_seidel_sweep(matrix, rhs, &mut x, options.omega, (0..matrix.rows).rev())?;

        residual_norm = l2_norm(&residual(matrix, &x, rhs)?);
        if residual_norm <= options.tolerance {
            return Ok(IterativeSolveReport {
                solution: x,
                iterations: iteration,
                residual_norm,
                converged: true,
                termination: IterativeSolveTermination::Converged,
            });
        }
    }

    Ok(IterativeSolveReport {
        solution: x,
        iterations: options.max_iterations,
        residual_norm,
        converged: false,
        termination: IterativeSolveTermination::MaxIterations,
    })
}

fn gauss_seidel_sweep(
    matrix: &CsrMatrix,
    rhs: &[f64],
    x: &mut [f64],
    omega: f64,
    rows: impl IntoIterator<Item = usize>,
) -> Result<()> {
    for row in rows {
        let start = matrix.row_offsets[row];
        let end = matrix.row_offsets[row + 1];
        let mut diagonal = None;
        let mut off_diagonal_sum = 0.0;
        for entry in start..end {
            let column = matrix.col_indices[entry];
            let value = matrix.values[entry];
            if column == row {
                diagonal = Some(value);
            } else {
                off_diagonal_sum += value * x[column];
            }
        }
        let Some(diagonal) = diagonal else {
            return Err(invalid_input(format!(
                "row {row} has no diagonal entry for Gauss-Seidel"
            )));
        };
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
        x[row] = (1.0 - omega) * x[row] + omega * raw;
    }
    Ok(())
}

fn gauss_seidel_sweep_with_cached_diagonal(
    matrix: &CsrMatrix,
    diagonal_slots: &[usize],
    diagonal_values: &[f64],
    rhs: &[f64],
    x: &mut [f64],
    rows: impl IntoIterator<Item = usize>,
) -> Result<()> {
    debug_assert_eq!(diagonal_slots.len(), matrix.rows);
    debug_assert_eq!(diagonal_values.len(), matrix.rows);
    for row in rows {
        let start = matrix.row_offsets[row];
        let end = matrix.row_offsets[row + 1];
        let diagonal_slot = diagonal_slots[row];
        debug_assert!(diagonal_slot >= start && diagonal_slot < end);
        debug_assert_eq!(matrix.col_indices[diagonal_slot], row);

        let mut off_diagonal_sum = 0.0;
        for entry in start..diagonal_slot {
            off_diagonal_sum += matrix.values[entry] * x[matrix.col_indices[entry]];
        }
        for entry in diagonal_slot + 1..end {
            off_diagonal_sum += matrix.values[entry] * x[matrix.col_indices[entry]];
        }

        let diagonal = diagonal_values[row];
        let raw = (rhs[row] - off_diagonal_sum) / diagonal;
        if !raw.is_finite() {
            return Err(invalid_input(format!(
                "Gauss-Seidel update for row {row} is not finite"
            )));
        }
        x[row] = raw;
    }
    Ok(())
}

pub fn conjugate_gradient_solve(
    matrix: &CsrMatrix,
    rhs: &[f64],
    initial: Option<&[f64]>,
    options: ConjugateGradientOptions,
) -> Result<IterativeSolveReport> {
    validate_iterative_solve_input(matrix, rhs, initial, options.tolerance)?;

    let mut x = initial
        .map(|values| values.to_vec())
        .unwrap_or_else(|| vec![0.0; rhs.len()]);
    let mut r = residual(matrix, &x, rhs)?;
    let mut residual_squared = dot(&r, &r);
    let mut residual_norm = residual_squared.sqrt();
    if residual_norm <= options.tolerance {
        return Ok(IterativeSolveReport {
            solution: x,
            iterations: 0,
            residual_norm,
            converged: true,
            termination: IterativeSolveTermination::Converged,
        });
    }

    let mut p = r.clone();
    for iteration in 1..=options.max_iterations {
        let ap = matrix.matvec(&p)?;
        let denominator = dot(&p, &ap);
        if !denominator.is_finite() {
            return Err(invalid_input(
                "conjugate-gradient denominator is not finite; matrix is likely not SPD"
                    .to_string(),
            ));
        }
        if dot_product_is_singular(denominator, &p, &ap) {
            return Ok(IterativeSolveReport {
                solution: x,
                iterations: iteration,
                residual_norm,
                converged: false,
                termination: IterativeSolveTermination::Breakdown,
            });
        }

        let alpha = residual_squared / denominator;
        for row in 0..x.len() {
            x[row] += alpha * p[row];
            r[row] -= alpha * ap[row];
        }

        let next_residual_squared = dot(&r, &r);
        residual_norm = next_residual_squared.sqrt();
        if residual_norm <= options.tolerance {
            return Ok(IterativeSolveReport {
                solution: x,
                iterations: iteration,
                residual_norm,
                converged: true,
                termination: IterativeSolveTermination::Converged,
            });
        }

        let beta = next_residual_squared / residual_squared;
        for row in 0..p.len() {
            p[row] = r[row] + beta * p[row];
        }
        residual_squared = next_residual_squared;
    }

    Ok(IterativeSolveReport {
        solution: x,
        iterations: options.max_iterations,
        residual_norm,
        converged: false,
        termination: IterativeSolveTermination::MaxIterations,
    })
}

#[inline(always)]
fn profile_start<const PROFILE: bool>() -> Option<Instant> {
    if PROFILE { Some(Instant::now()) } else { None }
}

#[inline(always)]
fn profile_elapsed(started: Option<Instant>) -> f64 {
    started
        .map(|started| started.elapsed().as_secs_f64())
        .unwrap_or(0.0)
}

pub struct PreconditionedConjugateGradientWorkspace {
    sparsity: CsrSparsityPattern,
    preconditioner_kind: CgPreconditioner,
    preconditioner: ReusablePreconditioner,
    face_ldu_preconditioner: Option<FaceLduPreconditioner>,
    residual: Vec<f64>,
    preconditioned_residual: Vec<f64>,
    direction: Vec<f64>,
    matrix_direction: Vec<f64>,
    preconditioner_scratch: Vec<f64>,
}

impl PreconditionedConjugateGradientWorkspace {
    pub fn new(matrix: &CsrMatrix, preconditioner: CgPreconditioner) -> Result<Self> {
        if matrix.rows() != matrix.cols() {
            return Err(invalid_input(format!(
                "preconditioned conjugate-gradient workspace requires a square matrix, got {}x{}",
                matrix.rows(),
                matrix.cols()
            )));
        }
        let rows = matrix.rows();
        let preconditioner_kind = preconditioner;
        let (preconditioner, face_ldu_preconditioner) = match preconditioner_kind {
            CgPreconditioner::Dic => (
                ReusablePreconditioner::None,
                Some(FaceLduPreconditioner::new(
                    matrix,
                    FaceLduPreconditionerKind::Dic,
                )?),
            ),
            CgPreconditioner::Fdic => (
                ReusablePreconditioner::None,
                Some(FaceLduPreconditioner::new(
                    matrix,
                    FaceLduPreconditionerKind::Fdic,
                )?),
            ),
            preconditioner => (ReusablePreconditioner::new(matrix, preconditioner)?, None),
        };
        Ok(Self {
            sparsity: matrix.sparsity_pattern(),
            preconditioner_kind,
            preconditioner,
            face_ldu_preconditioner,
            residual: vec![0.0; rows],
            preconditioned_residual: vec![0.0; rows],
            direction: vec![0.0; rows],
            matrix_direction: vec![0.0; rows],
            preconditioner_scratch: vec![0.0; rows],
        })
    }

    pub fn solve(
        &mut self,
        matrix: &CsrMatrix,
        rhs: &[f64],
        initial: Option<&[f64]>,
        options: PreconditionedConjugateGradientOptions,
    ) -> Result<IterativeSolveReport> {
        let mut timing = PcgKernelTiming::default();
        self.solve_internal::<false>(matrix, rhs, initial, options, &mut timing)
    }

    pub fn solve_profiled(
        &mut self,
        matrix: &CsrMatrix,
        rhs: &[f64],
        initial: Option<&[f64]>,
        options: PreconditionedConjugateGradientOptions,
    ) -> Result<ProfiledIterativeSolveReport> {
        let started = Instant::now();
        let mut timing = PcgKernelTiming::default();
        let report = self.solve_internal::<true>(matrix, rhs, initial, options, &mut timing)?;
        timing.total_seconds = started.elapsed().as_secs_f64();
        let accounted_seconds = timing.preconditioner_update_seconds
            + timing.matrix_vector_seconds
            + timing.preconditioner_application_seconds
            + timing.vector_operation_seconds;
        timing.other_seconds = (timing.total_seconds - accounted_seconds).max(0.0);
        Ok(ProfiledIterativeSolveReport { report, timing })
    }

    fn solve_internal<const PROFILE: bool>(
        &mut self,
        matrix: &CsrMatrix,
        rhs: &[f64],
        initial: Option<&[f64]>,
        options: PreconditionedConjugateGradientOptions,
        timing: &mut PcgKernelTiming,
    ) -> Result<IterativeSolveReport> {
        validate_iterative_solve_input(matrix, rhs, initial, options.tolerance)?;
        if options.preconditioner != self.preconditioner_kind {
            return Err(invalid_input(format!(
                "PCG workspace preconditioner {:?} does not match requested {:?}",
                self.preconditioner_kind, options.preconditioner
            )));
        }
        if !matrix.shares_sparsity_with(&self.sparsity) {
            return Err(invalid_input(
                "PCG workspace does not match matrix sparsity".to_string(),
            ));
        }

        let preconditioner_update_started = profile_start::<PROFILE>();
        if let Some(preconditioner) = &mut self.face_ldu_preconditioner {
            preconditioner.refactor(matrix)?;
        } else {
            self.preconditioner.update(matrix)?;
        }
        timing.preconditioner_update_seconds += profile_elapsed(preconditioner_update_started);
        let solution_setup_started = profile_start::<PROFILE>();
        let mut solution = initial
            .map(|values| values.to_vec())
            .unwrap_or_else(|| vec![0.0; rhs.len()]);
        timing.vector_operation_seconds += profile_elapsed(solution_setup_started);

        let matrix_vector_started = profile_start::<PROFILE>();
        matrix.matvec_into(&solution, &mut self.matrix_direction)?;
        timing.matrix_vector_seconds += profile_elapsed(matrix_vector_started);
        if PROFILE {
            timing.matrix_vector_products += 1;
        }

        let residual_setup_started = profile_start::<PROFILE>();
        for ((residual, source), matrix_value) in self
            .residual
            .iter_mut()
            .zip(rhs)
            .zip(&self.matrix_direction)
        {
            *residual = source - matrix_value;
        }
        let mut residual_norm = l2_norm(&self.residual);
        timing.vector_operation_seconds += profile_elapsed(residual_setup_started);
        if residual_norm <= options.tolerance {
            return Ok(IterativeSolveReport {
                solution,
                iterations: 0,
                residual_norm,
                converged: true,
                termination: IterativeSolveTermination::Converged,
            });
        }

        let preconditioner_application_started = profile_start::<PROFILE>();
        if let Some(preconditioner) = &self.face_ldu_preconditioner {
            preconditioner.apply_into(&self.residual, &mut self.preconditioned_residual)?;
        } else {
            self.preconditioner.apply_into(
                &self.residual,
                &mut self.preconditioner_scratch,
                &mut self.preconditioned_residual,
            )?;
        }
        timing.preconditioner_application_seconds +=
            profile_elapsed(preconditioner_application_started);
        if PROFILE {
            timing.preconditioner_applications += 1;
        }

        let residual_product_started = profile_start::<PROFILE>();
        let mut residual_product = dot(&self.residual, &self.preconditioned_residual);
        timing.vector_operation_seconds += profile_elapsed(residual_product_started);
        if !residual_product.is_finite() {
            return Err(invalid_input(
                "preconditioned conjugate-gradient residual product is not finite".to_string(),
            ));
        }
        let singularity_check_started = profile_start::<PROFILE>();
        let residual_product_is_singular = dot_product_is_singular(
            residual_product,
            &self.residual,
            &self.preconditioned_residual,
        );
        timing.vector_operation_seconds += profile_elapsed(singularity_check_started);
        if residual_product_is_singular {
            return Ok(IterativeSolveReport {
                solution,
                iterations: 0,
                residual_norm,
                converged: false,
                termination: IterativeSolveTermination::Breakdown,
            });
        }

        let direction_setup_started = profile_start::<PROFILE>();
        self.direction
            .copy_from_slice(&self.preconditioned_residual);
        timing.vector_operation_seconds += profile_elapsed(direction_setup_started);
        for iteration in 1..=options.max_iterations {
            let matrix_vector_started = profile_start::<PROFILE>();
            matrix.matvec_into(&self.direction, &mut self.matrix_direction)?;
            timing.matrix_vector_seconds += profile_elapsed(matrix_vector_started);
            if PROFILE {
                timing.matrix_vector_products += 1;
            }

            let denominator_started = profile_start::<PROFILE>();
            let denominator = dot(&self.direction, &self.matrix_direction);
            timing.vector_operation_seconds += profile_elapsed(denominator_started);
            if !denominator.is_finite() {
                return Err(invalid_input(
                    "preconditioned conjugate-gradient denominator is not finite; matrix is likely not SPD"
                        .to_string(),
                ));
            }
            let singularity_check_started = profile_start::<PROFILE>();
            let denominator_is_singular =
                dot_product_is_singular(denominator, &self.direction, &self.matrix_direction);
            timing.vector_operation_seconds += profile_elapsed(singularity_check_started);
            if denominator_is_singular {
                return Ok(IterativeSolveReport {
                    solution,
                    iterations: iteration,
                    residual_norm,
                    converged: false,
                    termination: IterativeSolveTermination::Breakdown,
                });
            }

            let alpha = residual_product / denominator;
            let solution_update_started = profile_start::<PROFILE>();
            for (row, solution_value) in solution.iter_mut().enumerate() {
                *solution_value += alpha * self.direction[row];
                self.residual[row] -= alpha * self.matrix_direction[row];
            }
            timing.vector_operation_seconds += profile_elapsed(solution_update_started);

            let residual_norm_started = profile_start::<PROFILE>();
            residual_norm = l2_norm(&self.residual);
            timing.vector_operation_seconds += profile_elapsed(residual_norm_started);
            if residual_norm <= options.tolerance {
                return Ok(IterativeSolveReport {
                    solution,
                    iterations: iteration,
                    residual_norm,
                    converged: true,
                    termination: IterativeSolveTermination::Converged,
                });
            }

            let preconditioner_application_started = profile_start::<PROFILE>();
            if let Some(preconditioner) = &self.face_ldu_preconditioner {
                preconditioner.apply_into(&self.residual, &mut self.preconditioned_residual)?;
            } else {
                self.preconditioner.apply_into(
                    &self.residual,
                    &mut self.preconditioner_scratch,
                    &mut self.preconditioned_residual,
                )?;
            }
            timing.preconditioner_application_seconds +=
                profile_elapsed(preconditioner_application_started);
            if PROFILE {
                timing.preconditioner_applications += 1;
            }

            let residual_product_started = profile_start::<PROFILE>();
            let next_residual_product = dot(&self.residual, &self.preconditioned_residual);
            timing.vector_operation_seconds += profile_elapsed(residual_product_started);
            if !next_residual_product.is_finite() {
                return Err(invalid_input(
                    "preconditioned conjugate-gradient residual product is not finite".to_string(),
                ));
            }
            let singularity_check_started = profile_start::<PROFILE>();
            let next_residual_product_is_singular = dot_product_is_singular(
                next_residual_product,
                &self.residual,
                &self.preconditioned_residual,
            );
            timing.vector_operation_seconds += profile_elapsed(singularity_check_started);
            if next_residual_product_is_singular {
                return Ok(IterativeSolveReport {
                    solution,
                    iterations: iteration,
                    residual_norm,
                    converged: false,
                    termination: IterativeSolveTermination::Breakdown,
                });
            }
            let beta = next_residual_product / residual_product;
            let direction_update_started = profile_start::<PROFILE>();
            for row in 0..self.direction.len() {
                self.direction[row] =
                    self.preconditioned_residual[row] + beta * self.direction[row];
            }
            timing.vector_operation_seconds += profile_elapsed(direction_update_started);
            residual_product = next_residual_product;
        }

        Ok(IterativeSolveReport {
            solution,
            iterations: options.max_iterations,
            residual_norm,
            converged: false,
            termination: IterativeSolveTermination::MaxIterations,
        })
    }
}

pub fn preconditioned_conjugate_gradient_solve(
    matrix: &CsrMatrix,
    rhs: &[f64],
    initial: Option<&[f64]>,
    options: PreconditionedConjugateGradientOptions,
) -> Result<IterativeSolveReport> {
    let mut workspace =
        PreconditionedConjugateGradientWorkspace::new(matrix, options.preconditioner)?;
    workspace.solve(matrix, rhs, initial, options)
}

pub fn bicgstab_solve(
    matrix: &CsrMatrix,
    rhs: &[f64],
    initial: Option<&[f64]>,
    options: BiCgStabOptions,
) -> Result<IterativeSolveReport> {
    validate_iterative_solve_input(matrix, rhs, initial, options.tolerance)?;

    let preconditioner = BuiltPreconditioner::build(matrix, options.preconditioner)?;
    let mut x = initial
        .map(|values| values.to_vec())
        .unwrap_or_else(|| vec![0.0; rhs.len()]);
    let mut r = residual(matrix, &x, rhs)?;
    let r_hat = r.clone();
    let mut residual_norm = l2_norm(&r);
    if residual_norm <= options.tolerance {
        return Ok(IterativeSolveReport {
            solution: x,
            iterations: 0,
            residual_norm,
            converged: true,
            termination: IterativeSolveTermination::Converged,
        });
    }

    let mut rho_old = 1.0;
    let mut alpha = 1.0;
    let mut omega = 1.0;
    let mut v = vec![0.0; rhs.len()];
    let mut p = vec![0.0; rhs.len()];

    for iteration in 1..=options.max_iterations {
        let rho = dot(&r_hat, &r);
        if !rho.is_finite() {
            return Err(invalid_input(
                "BiCGStab residual product is not finite".to_string(),
            ));
        }
        if dot_product_is_singular(rho, &r_hat, &r) {
            return Ok(IterativeSolveReport {
                solution: x,
                iterations: iteration,
                residual_norm,
                converged: false,
                termination: IterativeSolveTermination::Breakdown,
            });
        }

        let beta = (rho / rho_old) * (alpha / omega);
        for row in 0..p.len() {
            p[row] = r[row] + beta * (p[row] - omega * v[row]);
        }

        let p_hat = preconditioner.apply(&p)?;
        v = matrix.matvec(&p_hat)?;
        let alpha_denominator = dot(&r_hat, &v);
        if !alpha_denominator.is_finite() {
            return Err(invalid_input(
                "BiCGStab alpha denominator is not finite".to_string(),
            ));
        }
        if dot_product_is_singular(alpha_denominator, &r_hat, &v) {
            return Ok(IterativeSolveReport {
                solution: x,
                iterations: iteration,
                residual_norm,
                converged: false,
                termination: IterativeSolveTermination::Breakdown,
            });
        }
        alpha = rho / alpha_denominator;

        let mut s = vec![0.0; r.len()];
        for row in 0..s.len() {
            s[row] = r[row] - alpha * v[row];
        }
        let s_norm = l2_norm(&s);
        if s_norm <= options.tolerance {
            for row in 0..x.len() {
                x[row] += alpha * p_hat[row];
            }
            return Ok(IterativeSolveReport {
                solution: x,
                iterations: iteration,
                residual_norm: s_norm,
                converged: true,
                termination: IterativeSolveTermination::Converged,
            });
        }

        let s_hat = preconditioner.apply(&s)?;
        let t = matrix.matvec(&s_hat)?;
        let omega_denominator = dot(&t, &t);
        if !omega_denominator.is_finite() {
            return Err(invalid_input(
                "BiCGStab omega denominator is not finite".to_string(),
            ));
        }
        if omega_denominator == 0.0 {
            return Ok(IterativeSolveReport {
                solution: x,
                iterations: iteration,
                residual_norm,
                converged: false,
                termination: IterativeSolveTermination::Breakdown,
            });
        }
        omega = dot(&t, &s) / omega_denominator;
        if !omega.is_finite() {
            return Err(invalid_input("BiCGStab omega is not finite".to_string()));
        }
        if omega == 0.0 {
            return Ok(IterativeSolveReport {
                solution: x,
                iterations: iteration,
                residual_norm,
                converged: false,
                termination: IterativeSolveTermination::Breakdown,
            });
        }

        for row in 0..x.len() {
            x[row] += alpha * p_hat[row] + omega * s_hat[row];
            r[row] = s[row] - omega * t[row];
        }
        residual_norm = l2_norm(&r);
        if residual_norm <= options.tolerance {
            return Ok(IterativeSolveReport {
                solution: x,
                iterations: iteration,
                residual_norm,
                converged: true,
                termination: IterativeSolveTermination::Converged,
            });
        }
        rho_old = rho;
    }

    Ok(IterativeSolveReport {
        solution: x,
        iterations: options.max_iterations,
        residual_norm,
        converged: false,
        termination: IterativeSolveTermination::MaxIterations,
    })
}

enum BuiltPreconditioner {
    None,
    Diagonal(Vec<f64>),
    IncompleteCholesky(Box<IncompleteCholeskyPreconditioner>),
    FaceLdu(Box<FaceLduPreconditioner>),
}

impl BuiltPreconditioner {
    fn build(matrix: &CsrMatrix, kind: CgPreconditioner) -> Result<Self> {
        match kind {
            CgPreconditioner::None => Ok(Self::None),
            CgPreconditioner::Diagonal => {
                let diagonal = matrix.diagonal()?;
                let mut inverse = Vec::with_capacity(diagonal.len());
                for (row, value) in diagonal.iter().copied().enumerate() {
                    if !value.is_finite() || value == 0.0 {
                        return Err(invalid_input(format!(
                            "row {row} has invalid diagonal preconditioner value {value}"
                        )));
                    }
                    let inverse_value = 1.0 / value;
                    if !inverse_value.is_finite() {
                        return Err(invalid_input(format!(
                            "row {row} diagonal preconditioner inverse is not finite for value {value}"
                        )));
                    }
                    inverse.push(inverse_value);
                }
                Ok(Self::Diagonal(inverse))
            }
            CgPreconditioner::IncompleteCholesky => Ok(Self::IncompleteCholesky(Box::new(
                IncompleteCholeskyPreconditioner::build(matrix)?,
            ))),
            CgPreconditioner::Dic => Ok(Self::FaceLdu(Box::new(FaceLduPreconditioner::build(
                matrix,
                FaceLduPreconditionerKind::Dic,
            )?))),
            CgPreconditioner::Fdic => Ok(Self::FaceLdu(Box::new(FaceLduPreconditioner::build(
                matrix,
                FaceLduPreconditionerKind::Fdic,
            )?))),
        }
    }

    fn apply(&self, residual: &[f64]) -> Result<Vec<f64>> {
        match self {
            Self::None => Ok(residual.to_vec()),
            Self::Diagonal(inverse) => Ok(residual
                .iter()
                .zip(inverse)
                .map(|(value, inverse)| value * inverse)
                .collect()),
            Self::IncompleteCholesky(preconditioner) => preconditioner.apply(residual),
            Self::FaceLdu(preconditioner) => preconditioner.apply(residual),
        }
    }
}

enum ReusablePreconditioner {
    None,
    Diagonal {
        matrix_slots: Vec<usize>,
        inverse: Vec<f64>,
    },
    IncompleteCholesky(IncompleteCholeskyPreconditioner),
}

impl ReusablePreconditioner {
    fn new(matrix: &CsrMatrix, kind: CgPreconditioner) -> Result<Self> {
        match kind {
            CgPreconditioner::None => Ok(Self::None),
            CgPreconditioner::Diagonal => Ok(Self::Diagonal {
                matrix_slots: csr_diagonal_slots(matrix)?,
                inverse: vec![0.0; matrix.rows()],
            }),
            CgPreconditioner::IncompleteCholesky => Ok(Self::IncompleteCholesky(
                IncompleteCholeskyPreconditioner::new(matrix)?,
            )),
            CgPreconditioner::Dic | CgPreconditioner::Fdic => Err(invalid_input(
                "DIC and FDIC reusable state is owned by the PCG face-LDU workspace".to_string(),
            )),
        }
    }

    fn update(&mut self, matrix: &CsrMatrix) -> Result<()> {
        match self {
            Self::None => Ok(()),
            Self::Diagonal {
                matrix_slots,
                inverse,
            } => {
                for (row, (slot, inverse)) in matrix_slots.iter().zip(inverse).enumerate() {
                    let value = matrix.values[*slot];
                    if !value.is_finite() || value == 0.0 {
                        return Err(invalid_input(format!(
                            "row {row} has invalid diagonal preconditioner value {value}"
                        )));
                    }
                    *inverse = 1.0 / value;
                    if !inverse.is_finite() {
                        return Err(invalid_input(format!(
                            "row {row} diagonal preconditioner inverse is not finite for value {value}"
                        )));
                    }
                }
                Ok(())
            }
            Self::IncompleteCholesky(preconditioner) => preconditioner.refactor(matrix),
        }
    }

    fn apply_into(&self, residual: &[f64], scratch: &mut [f64], output: &mut [f64]) -> Result<()> {
        if output.len() != residual.len() || scratch.len() != residual.len() {
            return Err(invalid_input(format!(
                "preconditioner workspace lengths must match residual length {}, got scratch={} output={}",
                residual.len(),
                scratch.len(),
                output.len()
            )));
        }
        match self {
            Self::None => {
                output.copy_from_slice(residual);
                Ok(())
            }
            Self::Diagonal { inverse, .. } => {
                for ((output, value), inverse) in output.iter_mut().zip(residual).zip(inverse) {
                    *output = value * inverse;
                }
                Ok(())
            }
            Self::IncompleteCholesky(preconditioner) => {
                preconditioner.apply_into(residual, scratch, output)
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum FaceLduPreconditionerKind {
    Dic,
    Fdic,
}

struct SymmetricFaceLduPattern {
    sparsity: CsrSparsityPattern,
    diagonal_slots: Vec<usize>,
    lower_addr: Vec<usize>,
    upper_addr: Vec<usize>,
    owner_start: Vec<usize>,
    lower_matrix_slots: Vec<usize>,
    upper_matrix_slots: Vec<usize>,
}

impl SymmetricFaceLduPattern {
    fn new(matrix: &CsrMatrix) -> Result<Self> {
        if matrix.rows != matrix.cols {
            return Err(invalid_input(format!(
                "symmetric face-LDU preconditioner requires a square matrix, got {}x{}",
                matrix.rows, matrix.cols
            )));
        }

        let mut row_slots = Vec::with_capacity(matrix.rows);
        for row in 0..matrix.rows {
            let mut slots = BTreeMap::new();
            for entry in matrix.row_offsets[row]..matrix.row_offsets[row + 1] {
                let column = matrix.col_indices[entry];
                if slots.insert(column, entry).is_some() {
                    return Err(invalid_input(format!(
                        "symmetric face-LDU preconditioner row {row} has duplicate column {column}"
                    )));
                }
            }
            if !slots.contains_key(&row) {
                return Err(invalid_input(format!(
                    "symmetric face-LDU preconditioner row {row} has no diagonal entry"
                )));
            }
            row_slots.push(slots);
        }

        for (row, slots) in row_slots.iter().enumerate() {
            for &column in slots.keys() {
                if row != column && !row_slots[column].contains_key(&row) {
                    return Err(invalid_input(format!(
                        "symmetric face-LDU preconditioner entry ({row},{column}) has no transpose entry ({column},{row})"
                    )));
                }
            }
        }

        let diagonal_slots = row_slots
            .iter()
            .enumerate()
            .map(|(row, slots)| {
                slots.get(&row).copied().ok_or_else(|| {
                    invalid_input(format!(
                        "symmetric face-LDU preconditioner row {row} has no diagonal entry"
                    ))
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let mut lower_addr = Vec::new();
        let mut upper_addr = Vec::new();
        let mut owner_start = Vec::with_capacity(matrix.rows + 1);
        let mut lower_matrix_slots = Vec::new();
        let mut upper_matrix_slots = Vec::new();
        for lower in 0..matrix.rows {
            owner_start.push(lower_addr.len());
            for (&upper, &upper_slot) in &row_slots[lower] {
                if upper <= lower {
                    continue;
                }
                let lower_slot = row_slots[upper].get(&lower).copied().ok_or_else(|| {
                    invalid_input(format!(
                        "symmetric face-LDU preconditioner entry ({lower},{upper}) has no transpose entry ({upper},{lower})"
                    ))
                })?;
                lower_addr.push(lower);
                upper_addr.push(upper);
                upper_matrix_slots.push(upper_slot);
                lower_matrix_slots.push(lower_slot);
            }
        }
        owner_start.push(lower_addr.len());

        Ok(Self {
            sparsity: matrix.sparsity_pattern(),
            diagonal_slots,
            lower_addr,
            upper_addr,
            owner_start,
            lower_matrix_slots,
            upper_matrix_slots,
        })
    }
}

struct FaceLduPreconditioner {
    kind: FaceLduPreconditionerKind,
    pattern: SymmetricFaceLduPattern,
    upper_coefficients: Vec<f64>,
    reciprocal_diagonal: Vec<f64>,
    fdic_upper_scaled_by_upper_diagonal: Vec<f64>,
    fdic_upper_scaled_by_lower_diagonal: Vec<f64>,
    valid: bool,
}

impl FaceLduPreconditioner {
    fn build(matrix: &CsrMatrix, kind: FaceLduPreconditionerKind) -> Result<Self> {
        let mut preconditioner = Self::new(matrix, kind)?;
        preconditioner.refactor(matrix)?;
        Ok(preconditioner)
    }

    fn new(matrix: &CsrMatrix, kind: FaceLduPreconditionerKind) -> Result<Self> {
        let pattern = SymmetricFaceLduPattern::new(matrix)?;
        let faces = pattern.upper_addr.len();
        let cells = pattern.diagonal_slots.len();
        let (fdic_upper_scaled_by_upper_diagonal, fdic_upper_scaled_by_lower_diagonal) =
            if kind == FaceLduPreconditionerKind::Fdic {
                (vec![0.0; faces], vec![0.0; faces])
            } else {
                (Vec::new(), Vec::new())
            };
        Ok(Self {
            kind,
            pattern,
            upper_coefficients: vec![0.0; faces],
            reciprocal_diagonal: vec![0.0; cells],
            fdic_upper_scaled_by_upper_diagonal,
            fdic_upper_scaled_by_lower_diagonal,
            valid: false,
        })
    }

    fn refactor(&mut self, matrix: &CsrMatrix) -> Result<()> {
        self.valid = false;
        if !matrix.shares_sparsity_with(&self.pattern.sparsity) {
            return Err(invalid_input(
                "symmetric face-LDU preconditioner does not match matrix sparsity".to_string(),
            ));
        }

        for (cell, (&slot, diagonal)) in self
            .pattern
            .diagonal_slots
            .iter()
            .zip(&mut self.reciprocal_diagonal)
            .enumerate()
        {
            let value = matrix.values[slot];
            if !value.is_finite() || value <= 0.0 {
                return Err(invalid_input(format!(
                    "symmetric face-LDU preconditioner cell {cell} has invalid diagonal pivot {value}"
                )));
            }
            *diagonal = value;
        }

        for face in 0..self.pattern.upper_addr.len() {
            let upper_value = matrix.values[self.pattern.upper_matrix_slots[face]];
            let lower_value = matrix.values[self.pattern.lower_matrix_slots[face]];
            if !upper_value.is_finite() || !lower_value.is_finite() {
                return Err(invalid_input(format!(
                    "symmetric face-LDU preconditioner face {face} has non-finite coefficients upper={upper_value} lower={lower_value}"
                )));
            }
            if upper_value.to_bits() != lower_value.to_bits() {
                let lower = self.pattern.lower_addr[face];
                let upper = self.pattern.upper_addr[face];
                return Err(invalid_input(format!(
                    "symmetric face-LDU preconditioner requires bitwise symmetric coefficients; A[{lower},{upper}]={upper_value} differs from A[{upper},{lower}]={lower_value}"
                )));
            }
            self.upper_coefficients[face] = upper_value;
        }

        for lower in 0..self.pattern.diagonal_slots.len() {
            for face in self.pattern.owner_start[lower]..self.pattern.owner_start[lower + 1] {
                debug_assert_eq!(self.pattern.lower_addr[face], lower);
                let upper = self.pattern.upper_addr[face];
                let pivot = self.reciprocal_diagonal[lower];
                if !pivot.is_finite() || pivot <= 0.0 {
                    return Err(invalid_input(format!(
                        "symmetric face-LDU preconditioner cell {lower} has invalid DIC pivot {pivot}"
                    )));
                }
                let coefficient = self.upper_coefficients[face];
                let square = coefficient * coefficient;
                if !square.is_finite() {
                    return Err(invalid_input(format!(
                        "symmetric face-LDU preconditioner face {face} coefficient square is not finite"
                    )));
                }
                let contribution = square / pivot;
                if !contribution.is_finite() {
                    return Err(invalid_input(format!(
                        "symmetric face-LDU preconditioner face {face} diagonal contribution is not finite"
                    )));
                }
                let updated = self.reciprocal_diagonal[upper] - contribution;
                if !updated.is_finite() || updated <= 0.0 {
                    return Err(invalid_input(format!(
                        "symmetric face-LDU preconditioner cell {upper} has non-positive DIC pivot {updated}"
                    )));
                }
                self.reciprocal_diagonal[upper] = updated;
            }
        }

        for (cell, diagonal) in self.reciprocal_diagonal.iter_mut().enumerate() {
            if !diagonal.is_finite() || *diagonal <= 0.0 {
                return Err(invalid_input(format!(
                    "symmetric face-LDU preconditioner cell {cell} has invalid DIC pivot {diagonal}"
                )));
            }
            let reciprocal = 1.0 / *diagonal;
            if !reciprocal.is_finite() || reciprocal <= 0.0 {
                return Err(invalid_input(format!(
                    "symmetric face-LDU preconditioner cell {cell} reciprocal pivot is invalid for value {diagonal}"
                )));
            }
            *diagonal = reciprocal;
        }

        if self.kind == FaceLduPreconditionerKind::Fdic {
            for face in 0..self.pattern.upper_addr.len() {
                let lower = self.pattern.lower_addr[face];
                let upper = self.pattern.upper_addr[face];
                let coefficient = self.upper_coefficients[face];
                let upper_scaled = self.reciprocal_diagonal[upper] * coefficient;
                let lower_scaled = self.reciprocal_diagonal[lower] * coefficient;
                if !upper_scaled.is_finite() || !lower_scaled.is_finite() {
                    return Err(invalid_input(format!(
                        "symmetric face-LDU FDIC face {face} cached multiplier is not finite"
                    )));
                }
                self.fdic_upper_scaled_by_upper_diagonal[face] = upper_scaled;
                self.fdic_upper_scaled_by_lower_diagonal[face] = lower_scaled;
            }
        }

        self.valid = true;
        Ok(())
    }

    fn apply(&self, residual: &[f64]) -> Result<Vec<f64>> {
        let mut output = vec![0.0; residual.len()];
        self.apply_into(residual, &mut output)?;
        Ok(output)
    }

    fn apply_into(&self, residual: &[f64], output: &mut [f64]) -> Result<()> {
        if !self.valid {
            return Err(invalid_input(
                "symmetric face-LDU preconditioner has no valid numerical factorization"
                    .to_string(),
            ));
        }
        if residual.len() != self.reciprocal_diagonal.len() || output.len() != residual.len() {
            return Err(invalid_input(format!(
                "symmetric face-LDU apply expected {} entries, got residual={} output={}",
                self.reciprocal_diagonal.len(),
                residual.len(),
                output.len()
            )));
        }

        for ((output, residual), reciprocal) in output
            .iter_mut()
            .zip(residual)
            .zip(&self.reciprocal_diagonal)
        {
            *output = reciprocal * residual;
        }
        match self.kind {
            FaceLduPreconditionerKind::Dic => {
                for face in 0..self.pattern.upper_addr.len() {
                    let lower = self.pattern.lower_addr[face];
                    let upper = self.pattern.upper_addr[face];
                    output[upper] -= self.reciprocal_diagonal[upper]
                        * self.upper_coefficients[face]
                        * output[lower];
                }
                for face in (0..self.pattern.upper_addr.len()).rev() {
                    let lower = self.pattern.lower_addr[face];
                    let upper = self.pattern.upper_addr[face];
                    output[lower] -= self.reciprocal_diagonal[lower]
                        * self.upper_coefficients[face]
                        * output[upper];
                }
            }
            FaceLduPreconditionerKind::Fdic => {
                for face in 0..self.pattern.upper_addr.len() {
                    let lower = self.pattern.lower_addr[face];
                    let upper = self.pattern.upper_addr[face];
                    output[upper] -= self.fdic_upper_scaled_by_upper_diagonal[face] * output[lower];
                }
                for face in (0..self.pattern.upper_addr.len()).rev() {
                    let lower = self.pattern.lower_addr[face];
                    let upper = self.pattern.upper_addr[face];
                    output[lower] -= self.fdic_upper_scaled_by_lower_diagonal[face] * output[upper];
                }
            }
        }
        Ok(())
    }
}

struct IncompleteCholeskyPreconditioner {
    sparsity: CsrSparsityPattern,
    lower_row_offsets: Vec<usize>,
    lower_columns: Vec<usize>,
    matrix_slots: Vec<usize>,
    diagonal_factor_slots: Vec<usize>,
    update_pairs: Vec<Vec<(usize, usize)>>,
    dependent_row_offsets: Vec<usize>,
    dependent_entries: Vec<(usize, usize)>,
    factors: Vec<f64>,
}

impl IncompleteCholeskyPreconditioner {
    fn build(matrix: &CsrMatrix) -> Result<Self> {
        let mut preconditioner = Self::new(matrix)?;
        preconditioner.refactor(matrix)?;
        Ok(preconditioner)
    }

    fn new(matrix: &CsrMatrix) -> Result<Self> {
        if matrix.rows != matrix.cols {
            return Err(invalid_input(format!(
                "incomplete Cholesky preconditioner requires a square matrix, got {}x{}",
                matrix.rows, matrix.cols
            )));
        }

        let mut lower_matrix_slots = vec![BTreeMap::<usize, usize>::new(); matrix.rows];
        for (row, lower_row) in lower_matrix_slots.iter_mut().enumerate() {
            for entry in matrix.row_offsets[row]..matrix.row_offsets[row + 1] {
                let column = matrix.col_indices[entry];
                if column <= row {
                    lower_row.insert(column, entry);
                }
            }
            if !lower_row.contains_key(&row) {
                return Err(invalid_input(format!(
                    "incomplete Cholesky preconditioner row {row} has no diagonal entry"
                )));
            }
        }

        let mut lower_row_offsets = Vec::with_capacity(matrix.rows + 1);
        let mut lower_columns = Vec::new();
        let mut matrix_slots = Vec::new();
        let mut diagonal_factor_slots = Vec::with_capacity(matrix.rows);
        let mut factor_lookup = vec![BTreeMap::<usize, usize>::new(); matrix.rows];
        lower_row_offsets.push(0);
        for (row, entries) in lower_matrix_slots.iter().enumerate() {
            for (&column, &matrix_slot) in entries {
                let factor_slot = lower_columns.len();
                lower_columns.push(column);
                matrix_slots.push(matrix_slot);
                factor_lookup[row].insert(column, factor_slot);
            }
            diagonal_factor_slots.push(*factor_lookup[row].get(&row).ok_or_else(|| {
                invalid_input(format!(
                    "incomplete Cholesky preconditioner row {row} has no diagonal entry"
                ))
            })?);
            lower_row_offsets.push(lower_columns.len());
        }

        let mut update_pairs = vec![Vec::new(); lower_columns.len()];
        let mut dependent_counts = vec![0usize; matrix.rows];
        for row in 0..matrix.rows {
            let start = lower_row_offsets[row];
            let diagonal = diagonal_factor_slots[row];
            for factor_slot in start..diagonal {
                let column = lower_columns[factor_slot];
                for (row_factor_slot, &shared) in lower_columns
                    .iter()
                    .enumerate()
                    .take(factor_slot)
                    .skip(start)
                {
                    if shared >= column {
                        break;
                    }
                    if let Some(&column_factor_slot) = factor_lookup[column].get(&shared) {
                        update_pairs[factor_slot].push((row_factor_slot, column_factor_slot));
                    }
                }
                dependent_counts[column] =
                    dependent_counts[column].checked_add(1).ok_or_else(|| {
                        invalid_input(
                            "incomplete Cholesky dependent-entry count overflow".to_string(),
                        )
                    })?;
            }
        }

        let mut dependent_row_offsets = Vec::with_capacity(matrix.rows + 1);
        dependent_row_offsets.push(0usize);
        for count in dependent_counts {
            let next = dependent_row_offsets
                .last()
                .copied()
                .unwrap_or_default()
                .checked_add(count)
                .ok_or_else(|| {
                    invalid_input("incomplete Cholesky dependent-offset count overflow".to_string())
                })?;
            dependent_row_offsets.push(next);
        }
        let mut dependent_entries =
            vec![(0usize, 0usize); dependent_row_offsets.last().copied().unwrap_or_default()];
        let mut dependent_write_slots = dependent_row_offsets[..matrix.rows].to_vec();
        for row in 0..matrix.rows {
            let start = lower_row_offsets[row];
            let diagonal = diagonal_factor_slots[row];
            for (factor_slot, &column) in
                lower_columns.iter().enumerate().take(diagonal).skip(start)
            {
                let entry = dependent_write_slots[column];
                dependent_entries[entry] = (row, factor_slot);
                dependent_write_slots[column] += 1;
            }
        }
        debug_assert!(
            dependent_write_slots
                .iter()
                .zip(&dependent_row_offsets[1..])
                .all(|(written, end)| written == end)
        );

        Ok(Self {
            sparsity: matrix.sparsity_pattern(),
            lower_row_offsets,
            lower_columns,
            matrix_slots,
            diagonal_factor_slots,
            update_pairs,
            dependent_row_offsets,
            dependent_entries,
            factors: vec![0.0; factor_lookup.iter().map(BTreeMap::len).sum()],
        })
    }

    fn refactor(&mut self, matrix: &CsrMatrix) -> Result<()> {
        if !matrix.shares_sparsity_with(&self.sparsity) {
            return Err(invalid_input(
                "incomplete Cholesky preconditioner does not match matrix sparsity".to_string(),
            ));
        }
        for (factor, matrix_slot) in self.factors.iter_mut().zip(&self.matrix_slots) {
            *factor = matrix.values[*matrix_slot];
        }

        for row in 0..matrix.rows {
            let start = self.lower_row_offsets[row];
            let diagonal_slot = self.diagonal_factor_slots[row];
            for factor_slot in start..diagonal_slot {
                let column = self.lower_columns[factor_slot];
                let mut value = self.factors[factor_slot];
                for &(row_factor_slot, column_factor_slot) in &self.update_pairs[factor_slot] {
                    value -= self.factors[row_factor_slot] * self.factors[column_factor_slot];
                }
                let pivot = self.factors[self.diagonal_factor_slots[column]];
                if !pivot.is_finite() || pivot <= 0.0 {
                    return Err(invalid_input(format!(
                        "incomplete Cholesky preconditioner row {column} has invalid pivot {pivot}"
                    )));
                }
                self.factors[factor_slot] = value / pivot;
            }

            let mut diagonal = self.factors[diagonal_slot];
            for factor_slot in start..diagonal_slot {
                let value = self.factors[factor_slot];
                diagonal -= value * value;
            }
            if !diagonal.is_finite() || diagonal <= 0.0 {
                return Err(invalid_input(format!(
                    "incomplete Cholesky preconditioner row {row} has non-positive pivot square {diagonal}"
                )));
            }
            self.factors[diagonal_slot] = diagonal.sqrt();
        }
        Ok(())
    }

    fn apply(&self, residual: &[f64]) -> Result<Vec<f64>> {
        let mut scratch = vec![0.0; residual.len()];
        let mut output = vec![0.0; residual.len()];
        self.apply_into(residual, &mut scratch, &mut output)?;
        Ok(output)
    }

    fn apply_into(&self, residual: &[f64], scratch: &mut [f64], output: &mut [f64]) -> Result<()> {
        if residual.len() != self.diagonal_factor_slots.len()
            || scratch.len() != residual.len()
            || output.len() != residual.len()
        {
            return Err(invalid_input(format!(
                "incomplete Cholesky apply expected {} entries, got residual={} scratch={} output={}",
                self.diagonal_factor_slots.len(),
                residual.len(),
                scratch.len(),
                output.len()
            )));
        }
        for row in 0..residual.len() {
            let mut sum = residual[row];
            let start = self.lower_row_offsets[row];
            let diagonal_slot = self.diagonal_factor_slots[row];
            for factor_slot in start..diagonal_slot {
                sum -= self.factors[factor_slot] * scratch[self.lower_columns[factor_slot]];
            }
            scratch[row] = sum / self.factors[diagonal_slot];
        }

        for row in (0..residual.len()).rev() {
            let mut sum = scratch[row];
            let dependent_start = self.dependent_row_offsets[row];
            let dependent_end = self.dependent_row_offsets[row + 1];
            for &(dependent_row, factor_slot) in
                &self.dependent_entries[dependent_start..dependent_end]
            {
                sum -= self.factors[factor_slot] * output[dependent_row];
            }
            output[row] = sum / self.factors[self.diagonal_factor_slots[row]];
        }
        Ok(())
    }
}

fn csr_diagonal_slots(matrix: &CsrMatrix) -> Result<Vec<usize>> {
    if matrix.rows != matrix.cols {
        return Err(invalid_input(format!(
            "diagonal extraction requires a square matrix, got {}x{}",
            matrix.rows, matrix.cols
        )));
    }
    (0..matrix.rows)
        .map(|row| {
            (matrix.row_offsets[row]..matrix.row_offsets[row + 1])
                .find(|entry| matrix.col_indices[*entry] == row)
                .ok_or_else(|| invalid_input(format!("row {row} has no diagonal entry")))
        })
        .collect()
}

fn validate_csr_pattern(
    rows: usize,
    cols: usize,
    row_offsets: &[usize],
    col_indices: &[usize],
) -> Result<()> {
    if row_offsets.len() != rows + 1 {
        return Err(invalid_input(format!(
            "CSR row_offsets length must be rows + 1 ({}), got {}",
            rows + 1,
            row_offsets.len()
        )));
    }
    if row_offsets.first().copied() != Some(0) {
        return Err(invalid_input(
            "CSR row_offsets must start with zero".to_string(),
        ));
    }
    if row_offsets.last().copied() != Some(col_indices.len()) {
        return Err(invalid_input(format!(
            "CSR final row offset must equal nnz {}, got {}",
            col_indices.len(),
            row_offsets.last().copied().unwrap_or_default()
        )));
    }
    for (row, offsets) in row_offsets.windows(2).enumerate() {
        if offsets[0] > offsets[1] {
            return Err(invalid_input(format!(
                "CSR row_offsets must be monotonic; row {row} has {} > {}",
                offsets[0], offsets[1]
            )));
        }
    }
    for (entry, &col) in col_indices.iter().enumerate() {
        if col >= cols {
            return Err(invalid_input(format!(
                "CSR column index out of range at entry {entry}: {col} >= {cols}"
            )));
        }
    }
    Ok(())
}

fn validate_csr_values(expected_nnz: usize, values: &[f64]) -> Result<()> {
    if values.len() != expected_nnz {
        return Err(invalid_input(format!(
            "CSR values length {} does not match sparsity nnz {}",
            values.len(),
            expected_nnz
        )));
    }
    if let Some((entry, value)) = values
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(invalid_input(format!(
            "CSR value at entry {entry} must be finite, got {value}"
        )));
    }

    Ok(())
}

fn validate_iterative_solve_input(
    matrix: &CsrMatrix,
    rhs: &[f64],
    initial: Option<&[f64]>,
    tolerance: f64,
) -> Result<()> {
    if matrix.rows != matrix.cols {
        return Err(invalid_input(format!(
            "iterative solvers require a square matrix, got {}x{}",
            matrix.rows, matrix.cols
        )));
    }
    if rhs.len() != matrix.rows {
        return Err(invalid_input(format!(
            "iterative solve expected rhs with {} entries, got {}",
            matrix.rows,
            rhs.len()
        )));
    }
    if let Some(initial) = initial
        && initial.len() != matrix.cols
    {
        return Err(invalid_input(format!(
            "iterative solve expected initial guess with {} entries, got {}",
            matrix.cols,
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
    if !tolerance.is_finite() || tolerance < 0.0 {
        return Err(invalid_input(format!(
            "iterative solve tolerance must be finite and non-negative, got {tolerance}"
        )));
    }
    Ok(())
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left_value, right_value)| left_value * right_value)
        .sum()
}

pub(super) fn dot_product_is_singular(value: f64, left: &[f64], right: &[f64]) -> bool {
    let scale = l2_norm(left) * l2_norm(right);
    scale == 0.0 || !scale.is_finite() || value.abs() <= f64::EPSILON * scale
}

fn invalid_input(message: String) -> MeshError {
    MeshError::InvalidInput(message)
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use super::{
        BiCgStabOptions, CgPreconditioner, ConjugateGradientOptions, CsrMatrix, CsrSparsityPattern,
        FaceLduPreconditioner, FaceLduPreconditionerKind, GaussSeidelOptions,
        IncompleteCholeskyPreconditioner, IterativeSolveTermination, JacobiOptions,
        PreconditionedConjugateGradientOptions, PreconditionedConjugateGradientWorkspace,
        ReusablePreconditioner, bicgstab_solve, conjugate_gradient_solve, csr_diagonal_slots,
        gauss_seidel_solve, gauss_seidel_sweep, gauss_seidel_sweep_with_cached_diagonal,
        jacobi_solve, preconditioned_conjugate_gradient_solve, residual,
        symmetric_gauss_seidel_solve,
    };

    #[test]
    fn multiplies_csr_matrix_vector() {
        let matrix = poisson3();

        let result = matrix.matvec(&[1.0, 1.0, 1.0]).expect("matvec");

        assert_close(&result, &[1.0, 0.0, 1.0], 1.0e-14);
        assert_eq!(matrix.rows(), 3);
        assert_eq!(matrix.cols(), 3);
        assert_eq!(matrix.nnz(), 7);
    }

    #[test]
    fn rejects_invalid_csr_layout() {
        let short_offsets = CsrMatrix::new(2, 2, vec![0, 1], vec![0], vec![1.0]);
        assert!(short_offsets.is_err());

        let bad_column = CsrMatrix::new(1, 1, vec![0, 1], vec![1], vec![1.0]);
        assert!(bad_column.is_err());

        let bad_nnz = CsrMatrix::new(1, 1, vec![0, 2], vec![0], vec![1.0]);
        assert!(bad_nnz.is_err());

        let non_finite = CsrMatrix::new(1, 1, vec![0, 1], vec![0], vec![f64::NAN]);
        assert!(non_finite.is_err());
    }

    #[test]
    fn builds_matrices_that_share_an_immutable_sparsity_pattern() {
        let pattern = CsrSparsityPattern::new(2, 2, vec![0, 2, 4], vec![0, 1, 0, 1])
            .expect("sparsity pattern");
        let first =
            CsrMatrix::from_pattern(&pattern, vec![2.0, -1.0, -1.0, 2.0]).expect("first matrix");
        let second =
            CsrMatrix::from_pattern(&pattern, vec![3.0, -1.0, -1.0, 3.0]).expect("second matrix");

        assert!(std::sync::Arc::ptr_eq(
            &first.row_offsets,
            &second.row_offsets
        ));
        assert!(std::sync::Arc::ptr_eq(
            &first.col_indices,
            &second.col_indices
        ));
        assert!(first.shares_sparsity_with(&pattern));
        assert!(second.shares_sparsity_with(&pattern));
        assert_eq!(first.matvec(&[1.0, 1.0]).expect("first matvec"), [1.0, 1.0]);
        assert_eq!(
            second.matvec(&[1.0, 1.0]).expect("second matvec"),
            [2.0, 2.0]
        );
    }

    #[test]
    fn jacobi_solves_diagonal_system() {
        let matrix =
            CsrMatrix::from_rows(vec![vec![(0, 4.0)], vec![(1, 2.0)]], 2).expect("diagonal matrix");
        let report = jacobi_solve(
            &matrix,
            &[8.0, 6.0],
            None,
            JacobiOptions {
                max_iterations: 4,
                tolerance: 1.0e-12,
                omega: 1.0,
            },
        )
        .expect("jacobi solve");

        assert!(report.converged);
        assert_eq!(report.iterations, 1);
        assert_close(&report.solution, &[2.0, 3.0], 1.0e-14);
        assert!(report.residual_norm <= 1.0e-12);
    }

    #[test]
    fn jacobi_rejects_non_finite_update_from_tiny_pivot() {
        let matrix =
            CsrMatrix::new(1, 1, vec![0, 1], vec![0], vec![1.0e-320]).expect("finite matrix");

        let error = jacobi_solve(
            &matrix,
            &[1.0],
            None,
            JacobiOptions {
                max_iterations: 1,
                tolerance: 0.0,
                omega: 1.0,
            },
        )
        .expect_err("non-finite update must fail");

        assert!(error.to_string().contains("not finite"));
    }

    #[test]
    fn gauss_seidel_solves_nonsymmetric_system() {
        let matrix =
            CsrMatrix::from_rows(vec![vec![(0, 4.0), (1, -1.0)], vec![(0, 2.0), (1, 3.0)]], 2)
                .expect("nonsymmetric matrix");
        let report = gauss_seidel_solve(
            &matrix,
            &[3.0, 8.0],
            None,
            GaussSeidelOptions {
                max_iterations: 64,
                tolerance: 1.0e-12,
                omega: 1.0,
            },
        )
        .expect("gauss-seidel solve");

        assert!(report.converged);
        assert_close(&report.solution, &[17.0 / 14.0, 13.0 / 7.0], 1.0e-10);
        assert!(report.residual_norm <= 1.0e-12);
    }

    #[test]
    fn symmetric_gauss_seidel_solves_nonsymmetric_system() {
        let matrix =
            CsrMatrix::from_rows(vec![vec![(0, 4.0), (1, -1.0)], vec![(0, 2.0), (1, 3.0)]], 2)
                .expect("nonsymmetric matrix");
        let report = symmetric_gauss_seidel_solve(
            &matrix,
            &[3.0, 8.0],
            None,
            GaussSeidelOptions {
                max_iterations: 64,
                tolerance: 1.0e-12,
                omega: 1.0,
            },
        )
        .expect("symmetric Gauss-Seidel solve");

        assert!(report.converged);
        assert_close(&report.solution, &[17.0 / 14.0, 13.0 / 7.0], 1.0e-10);
        assert!(report.residual_norm <= 1.0e-12);
    }

    #[test]
    fn cached_diagonal_gauss_seidel_preserves_the_scan_order_bit_for_bit() {
        let matrix = CsrMatrix::from_rows(
            vec![
                vec![(0, 4.0), (1, -1.0), (2, 0.25)],
                vec![(0, -2.0), (1, 5.0), (2, -1.0)],
                vec![(0, 0.5), (1, -1.5), (2, 3.0)],
            ],
            3,
        )
        .expect("matrix with first, middle, and last diagonal slots");
        let diagonal_slots = csr_diagonal_slots(&matrix).expect("diagonal slots");
        let diagonal_values = diagonal_slots
            .iter()
            .map(|&slot| matrix.values()[slot])
            .collect::<Vec<_>>();
        let rhs = [1.25, -0.5, 2.75];
        let mut scanned = [0.125, -0.25, 0.75];
        let mut cached = scanned;

        for _ in 0..3 {
            gauss_seidel_sweep(&matrix, &rhs, &mut scanned, 1.0, 0..matrix.rows())
                .expect("forward scan sweep");
            gauss_seidel_sweep_with_cached_diagonal(
                &matrix,
                &diagonal_slots,
                &diagonal_values,
                &rhs,
                &mut cached,
                0..matrix.rows(),
            )
            .expect("forward cached sweep");
            assert_eq!(
                scanned.map(f64::to_bits),
                cached.map(f64::to_bits),
                "forward sweep changed floating-point order"
            );

            gauss_seidel_sweep(&matrix, &rhs, &mut scanned, 1.0, (0..matrix.rows()).rev())
                .expect("reverse scan sweep");
            gauss_seidel_sweep_with_cached_diagonal(
                &matrix,
                &diagonal_slots,
                &diagonal_values,
                &rhs,
                &mut cached,
                (0..matrix.rows()).rev(),
            )
            .expect("reverse cached sweep");
            assert_eq!(
                scanned.map(f64::to_bits),
                cached.map(f64::to_bits),
                "reverse sweep changed floating-point order"
            );
        }
    }

    #[test]
    fn conjugate_gradient_solves_poisson_system() {
        let matrix = poisson3();
        let report = conjugate_gradient_solve(
            &matrix,
            &[1.0, 0.0, 1.0],
            None,
            ConjugateGradientOptions {
                max_iterations: 8,
                tolerance: 1.0e-12,
            },
        )
        .expect("cg solve");

        assert!(report.converged);
        assert!(report.iterations <= 2);
        assert_close(&report.solution, &[1.0, 1.0, 1.0], 1.0e-12);
        assert!(report.residual_norm <= 1.0e-12);
    }

    #[test]
    fn conjugate_gradient_is_invariant_to_small_matrix_scale() {
        let scale = 1.0e-20;
        let matrix = CsrMatrix::from_rows(
            vec![
                vec![(0, 4.0 * scale), (1, scale)],
                vec![(0, scale), (1, 3.0 * scale)],
            ],
            2,
        )
        .expect("scaled SPD matrix");
        let report = conjugate_gradient_solve(
            &matrix,
            &[scale, 2.0 * scale],
            None,
            ConjugateGradientOptions {
                max_iterations: 8,
                tolerance: 1.0e-32,
            },
        )
        .expect("scaled CG solve");

        assert!(report.converged);
        assert_close(&report.solution, &[1.0 / 11.0, 7.0 / 11.0], 1.0e-12);
    }

    #[test]
    fn preconditioned_conjugate_gradient_solves_diagonal_system() {
        let matrix = CsrMatrix::from_rows(vec![vec![(0, 4.0)], vec![(1, 2.0)]], 2).expect("matrix");
        let report = preconditioned_conjugate_gradient_solve(
            &matrix,
            &[8.0, 6.0],
            None,
            PreconditionedConjugateGradientOptions {
                max_iterations: 4,
                tolerance: 1.0e-12,
                preconditioner: CgPreconditioner::Diagonal,
            },
        )
        .expect("pcg solve");

        assert!(report.converged);
        assert_eq!(report.iterations, 1);
        assert_close(&report.solution, &[2.0, 3.0], 1.0e-14);
        assert!(report.residual_norm <= 1.0e-12);
    }

    #[test]
    fn dic_and_fdic_match_hand_derived_face_ldu_oracle_bit_for_bit() {
        let matrix = dic_oracle_matrix();
        let dic =
            FaceLduPreconditioner::build(&matrix, FaceLduPreconditionerKind::Dic).expect("DIC");
        let fdic =
            FaceLduPreconditioner::build(&matrix, FaceLduPreconditionerKind::Fdic).expect("FDIC");

        assert_eq!(dic.pattern.lower_addr, [0, 1]);
        assert_eq!(dic.pattern.upper_addr, [1, 2]);
        assert_eq!(dic.pattern.owner_start, [0, 1, 2, 2]);
        assert_close(
            &dic.reciprocal_diagonal,
            &[1.0 / 4.0, 4.0 / 15.0, 15.0 / 41.0],
            1.0e-15,
        );
        assert_close(
            &fdic.fdic_upper_scaled_by_upper_diagonal,
            &[-4.0 / 15.0, -15.0 / 41.0],
            1.0e-15,
        );
        assert_close(
            &fdic.fdic_upper_scaled_by_lower_diagonal,
            &[-1.0 / 4.0, -4.0 / 15.0],
            1.0e-15,
        );

        let residual = [1.0, 2.0, 3.0];
        let dic_output = dic.apply(&residual).expect("DIC application");
        let fdic_output = fdic.apply(&residual).expect("FDIC application");
        assert_close(
            &dic_output,
            &[20.0 / 41.0, 39.0 / 41.0, 54.0 / 41.0],
            1.0e-15,
        );
        assert_eq!(
            dic_output
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            fdic_output
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn capabilities_advertise_distinct_dic_fdic_and_ic0_preconditioners() {
        let capabilities = super::linear_solver_capabilities();
        assert!(capabilities.cpu_incomplete_cholesky_preconditioner);
        assert!(capabilities.cpu_dic_preconditioner);
        assert!(capabilities.cpu_fdic_preconditioner);
    }

    #[test]
    fn face_ldu_structure_is_canonical_and_rejects_ambiguous_csr() {
        let unordered = CsrMatrix::from_rows(
            vec![
                vec![(1, -1.0), (0, 4.0)],
                vec![(2, -1.0), (1, 4.0), (0, -1.0)],
                vec![(2, 3.0), (1, -1.0)],
            ],
            3,
        )
        .expect("unordered symmetric matrix");
        let canonical = FaceLduPreconditioner::new(&unordered, FaceLduPreconditionerKind::Dic)
            .expect("canonical face-LDU pattern");
        assert_eq!(canonical.pattern.diagonal_slots, [1, 3, 5]);
        assert_eq!(canonical.pattern.lower_addr, [0, 1]);
        assert_eq!(canonical.pattern.upper_addr, [1, 2]);
        assert_eq!(canonical.pattern.owner_start, [0, 1, 2, 2]);

        let non_square =
            CsrMatrix::new(1, 2, vec![0, 1], vec![0], vec![1.0]).expect("non-square matrix");
        let error = FaceLduPreconditioner::new(&non_square, FaceLduPreconditionerKind::Dic)
            .err()
            .expect("non-square face-LDU must fail");
        assert!(error.to_string().contains("requires a square matrix"));

        let duplicate = CsrMatrix::new(1, 1, vec![0, 2], vec![0, 0], vec![1.0, 1.0])
            .expect("duplicate-column CSR");
        let error = FaceLduPreconditioner::new(&duplicate, FaceLduPreconditionerKind::Dic)
            .err()
            .expect("duplicate face-LDU column must fail");
        assert!(error.to_string().contains("duplicate column 0"));

        let missing_diagonal =
            CsrMatrix::new(2, 2, vec![0, 1, 3], vec![1, 0, 1], vec![-1.0, -1.0, 2.0])
                .expect("CSR without first diagonal");
        let error = FaceLduPreconditioner::new(&missing_diagonal, FaceLduPreconditionerKind::Dic)
            .err()
            .expect("missing face-LDU diagonal must fail");
        assert!(error.to_string().contains("row 0 has no diagonal entry"));

        let missing_transpose =
            CsrMatrix::new(2, 2, vec![0, 2, 3], vec![0, 1, 1], vec![2.0, -1.0, 2.0])
                .expect("CSR without transpose");
        let error = FaceLduPreconditioner::new(&missing_transpose, FaceLduPreconditionerKind::Dic)
            .err()
            .expect("missing face-LDU transpose must fail");
        assert!(error.to_string().contains("has no transpose entry"));
    }

    #[test]
    fn face_ldu_numeric_failures_are_fail_closed_and_recoverable() {
        let matrix = dic_oracle_matrix();
        let pattern = matrix.sparsity_pattern();
        let asymmetric =
            CsrMatrix::from_pattern(&pattern, vec![4.0, -1.0, -2.0, 4.0, -1.0, -1.0, 3.0])
                .expect("asymmetric values");
        let mut non_finite =
            CsrMatrix::from_pattern(&pattern, matrix.values().to_vec()).expect("shared matrix");
        non_finite.values_mut()[1] = f64::NAN;
        for kind in [CgPreconditioner::Dic, CgPreconditioner::Fdic] {
            let mut workspace = PreconditionedConjugateGradientWorkspace::new(&matrix, kind)
                .expect("face-LDU workspace");
            let options = PreconditionedConjugateGradientOptions {
                max_iterations: 8,
                tolerance: 1.0e-12,
                preconditioner: kind,
            };

            let error = workspace
                .solve(&asymmetric, &[1.0, 0.0, 1.0], None, options)
                .expect_err("bitwise asymmetric matrix must fail");
            assert!(error.to_string().contains("bitwise symmetric"));
            assert!(
                !workspace
                    .face_ldu_preconditioner
                    .as_ref()
                    .expect("face-LDU state")
                    .valid
            );

            let recovered = workspace
                .solve(&matrix, &[1.0, 0.0, 1.0], None, options)
                .expect("valid retry after failed refactor");
            assert!(recovered.converged);
            assert!(
                workspace
                    .face_ldu_preconditioner
                    .as_ref()
                    .expect("face-LDU state")
                    .valid
            );

            let error = workspace
                .solve(&non_finite, &[1.0, 0.0, 1.0], None, options)
                .expect_err("non-finite face coefficient must fail");
            assert!(error.to_string().contains("non-finite coefficients"));
            assert!(
                !workspace
                    .face_ldu_preconditioner
                    .as_ref()
                    .expect("face-LDU state")
                    .valid
            );

            let recovered = workspace
                .solve(&matrix, &[1.0, 0.0, 1.0], None, options)
                .expect("valid retry after non-finite refactor");
            assert!(recovered.converged);
        }

        let indefinite =
            CsrMatrix::from_rows(vec![vec![(0, 1.0), (1, 2.0)], vec![(0, 2.0), (1, 1.0)]], 2)
                .expect("indefinite symmetric matrix");
        let error = FaceLduPreconditioner::build(&indefinite, FaceLduPreconditionerKind::Dic)
            .err()
            .expect("non-positive DIC pivot must fail");
        assert!(error.to_string().contains("non-positive DIC pivot"));

        let reciprocal_overflow = CsrMatrix::from_rows(vec![vec![(0, f64::from_bits(1))]], 1)
            .expect("subnormal diagonal");
        let error =
            FaceLduPreconditioner::build(&reciprocal_overflow, FaceLduPreconditionerKind::Dic)
                .err()
                .expect("non-finite reciprocal must fail");
        assert!(error.to_string().contains("reciprocal pivot is invalid"));

        let square_overflow = CsrMatrix::from_rows(
            vec![
                vec![(0, f64::MAX), (1, f64::MAX)],
                vec![(0, f64::MAX), (1, f64::MAX)],
            ],
            2,
        )
        .expect("finite overflowing matrix");
        let error = FaceLduPreconditioner::build(&square_overflow, FaceLduPreconditionerKind::Fdic)
            .err()
            .expect("coefficient-square overflow must fail");
        assert!(
            error
                .to_string()
                .contains("coefficient square is not finite")
        );
    }

    #[test]
    fn reusable_dic_and_fdic_keep_allocations_stable_for_ten_lifecycles() {
        let first_matrix = dic_oracle_matrix();
        let pattern = first_matrix.sparsity_pattern();
        let exact = [1.0, -0.5, 2.0];

        for kind in [CgPreconditioner::Dic, CgPreconditioner::Fdic] {
            let mut workspace = PreconditionedConjugateGradientWorkspace::new(&first_matrix, kind)
                .expect("face-LDU PCG workspace");
            let pointers = face_ldu_workspace_pointers(&workspace);
            for lifecycle in 0..10 {
                let shift = lifecycle as f64 * 0.125;
                let matrix = CsrMatrix::from_pattern(
                    &pattern,
                    vec![
                        4.0 + shift,
                        -1.0,
                        -1.0,
                        4.0 + shift,
                        -1.0,
                        -1.0,
                        3.0 + shift,
                    ],
                )
                .expect("shared lifecycle matrix");
                let rhs = matrix.matvec(&exact).expect("manufactured rhs");
                let options = PreconditionedConjugateGradientOptions {
                    max_iterations: 16,
                    tolerance: 1.0e-12,
                    preconditioner: kind,
                };
                let reused = workspace
                    .solve(&matrix, &rhs, None, options)
                    .expect("reused face-LDU solve");
                let fresh = preconditioned_conjugate_gradient_solve(&matrix, &rhs, None, options)
                    .expect("fresh face-LDU solve");
                assert!(reused.converged);
                assert_eq!(reused.iterations, fresh.iterations);
                assert_eq!(
                    reused.residual_norm.to_bits(),
                    fresh.residual_norm.to_bits()
                );
                assert_eq!(
                    reused
                        .solution
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>(),
                    fresh
                        .solution
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>()
                );
                assert_eq!(face_ldu_workspace_pointers(&workspace), pointers);
            }
        }
    }

    #[test]
    fn dic_and_fdic_profiled_and_plain_pcg_are_bit_identical() {
        let matrix = dic_oracle_matrix();
        let rhs = [1.0, 0.0, 1.0];
        for kind in [CgPreconditioner::Dic, CgPreconditioner::Fdic] {
            let options = PreconditionedConjugateGradientOptions {
                max_iterations: 16,
                tolerance: 1.0e-12,
                preconditioner: kind,
            };
            let mut plain_workspace = PreconditionedConjugateGradientWorkspace::new(&matrix, kind)
                .expect("plain face-LDU workspace");
            let mut profiled_workspace =
                PreconditionedConjugateGradientWorkspace::new(&matrix, kind)
                    .expect("profiled face-LDU workspace");
            let plain = plain_workspace
                .solve(&matrix, &rhs, None, options)
                .expect("plain face-LDU PCG");
            let profiled = profiled_workspace
                .solve_profiled(&matrix, &rhs, None, options)
                .expect("profiled face-LDU PCG");

            assert!(plain.converged);
            assert_eq!(plain.iterations, profiled.report.iterations);
            assert_eq!(plain.termination, profiled.report.termination);
            assert_eq!(
                plain.residual_norm.to_bits(),
                profiled.report.residual_norm.to_bits()
            );
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
            assert_eq!(
                profiled.timing.matrix_vector_products,
                profiled.report.iterations + 1
            );
            assert_eq!(
                profiled.timing.preconditioner_applications,
                profiled.report.iterations
            );
        }
    }

    #[test]
    fn built_dic_and_fdic_precondition_bicgstab_on_symmetric_matrix() {
        let matrix = dic_oracle_matrix();
        let exact = [1.0, -0.5, 2.0];
        let rhs = matrix.matvec(&exact).expect("manufactured rhs");
        for kind in [CgPreconditioner::Dic, CgPreconditioner::Fdic] {
            let report = bicgstab_solve(
                &matrix,
                &rhs,
                None,
                BiCgStabOptions {
                    max_iterations: 16,
                    tolerance: 1.0e-12,
                    preconditioner: kind,
                },
            )
            .expect("built face-LDU BiCGStab solve");
            assert!(report.converged);
            assert_close(&report.solution, &exact, 1.0e-11);
        }
    }

    #[test]
    fn incomplete_cholesky_pcg_solves_spd_poisson_system() {
        let matrix = poisson3();
        let report = preconditioned_conjugate_gradient_solve(
            &matrix,
            &[1.0, 0.0, 1.0],
            None,
            PreconditionedConjugateGradientOptions {
                max_iterations: 4,
                tolerance: 1.0e-12,
                preconditioner: CgPreconditioner::IncompleteCholesky,
            },
        )
        .expect("ic0 pcg solve");

        assert!(report.converged);
        assert!(report.iterations <= 1);
        assert_close(&report.solution, &[1.0, 1.0, 1.0], 1.0e-12);
        assert!(report.residual_norm <= 1.0e-12);
    }

    #[test]
    fn reusable_ic0_pcg_workspace_refactors_without_reallocating_work_buffers() {
        let first_matrix = poisson3();
        let pattern = first_matrix.sparsity_pattern();
        let second_matrix =
            CsrMatrix::from_pattern(&pattern, vec![3.0, -1.0, -1.0, 3.0, -1.0, -1.0, 3.0])
                .expect("second SPD matrix");
        let options = PreconditionedConjugateGradientOptions {
            max_iterations: 8,
            tolerance: 1.0e-12,
            preconditioner: CgPreconditioner::IncompleteCholesky,
        };
        let mut workspace = PreconditionedConjugateGradientWorkspace::new(
            &first_matrix,
            CgPreconditioner::IncompleteCholesky,
        )
        .expect("reusable PCG workspace");
        let residual_ptr = workspace.residual.as_ptr();
        let direction_ptr = workspace.direction.as_ptr();
        let (factor_ptr, dependent_entries_ptr) = match &workspace.preconditioner {
            ReusablePreconditioner::IncompleteCholesky(preconditioner) => {
                assert_eq!(preconditioner.dependent_row_offsets, [0, 1, 2, 2]);
                assert_eq!(
                    preconditioner
                        .dependent_entries
                        .iter()
                        .map(|(row, _)| *row)
                        .collect::<Vec<_>>(),
                    [1, 2]
                );
                (
                    preconditioner.factors.as_ptr(),
                    preconditioner.dependent_entries.as_ptr(),
                )
            }
            _ => panic!("expected IC(0) workspace"),
        };

        let first = workspace
            .solve(&first_matrix, &[1.0, 0.0, 1.0], None, options)
            .expect("first workspace solve");
        let profiled = workspace
            .solve_profiled(&second_matrix, &[1.0, 0.0, 1.0], None, options)
            .expect("profiled refactored workspace solve");
        let reused = profiled.report;
        let fresh = preconditioned_conjugate_gradient_solve(
            &second_matrix,
            &[1.0, 0.0, 1.0],
            None,
            options,
        )
        .expect("fresh PCG solve");

        assert!(first.converged);
        assert!(reused.converged);
        assert_eq!(reused.iterations, fresh.iterations);
        assert_eq!(reused.residual_norm, fresh.residual_norm);
        assert_close(&reused.solution, &fresh.solution, 0.0);
        assert_eq!(
            profiled.timing.matrix_vector_products,
            reused.iterations + 1
        );
        assert_eq!(
            profiled.timing.preconditioner_applications,
            reused.iterations
        );
        assert!(profiled.timing.total_seconds >= 0.0);
        assert!(profiled.timing.preconditioner_update_seconds >= 0.0);
        assert!(profiled.timing.matrix_vector_seconds >= 0.0);
        assert!(profiled.timing.preconditioner_application_seconds >= 0.0);
        assert!(profiled.timing.vector_operation_seconds >= 0.0);
        assert!(profiled.timing.other_seconds >= 0.0);
        assert_eq!(workspace.residual.as_ptr(), residual_ptr);
        assert_eq!(workspace.direction.as_ptr(), direction_ptr);
        let (current_factor_ptr, current_dependent_entries_ptr) = match &workspace.preconditioner {
            ReusablePreconditioner::IncompleteCholesky(preconditioner) => (
                preconditioner.factors.as_ptr(),
                preconditioner.dependent_entries.as_ptr(),
            ),
            _ => panic!("expected IC(0) workspace"),
        };
        assert_eq!(current_factor_ptr, factor_ptr);
        assert_eq!(current_dependent_entries_ptr, dependent_entries_ptr);
    }

    #[test]
    #[ignore = "release-only IC(0) dependency-layout A/B diagnostic"]
    fn benchmarks_flat_ic0_dependency_layout_against_nested_rows() {
        const NX: usize = 256;
        const NY: usize = 152;
        const ROUNDS: usize = 9;
        const APPLICATIONS_PER_SAMPLE: usize = 64;

        let matrix = poisson_grid(NX, NY);
        let preconditioner =
            IncompleteCholeskyPreconditioner::build(&matrix).expect("fine-grid IC(0)");
        let nested_dependencies = nested_dependencies(&preconditioner);
        let residual = (0..matrix.rows())
            .map(|row| ((row % 29) + 1) as f64 / 29.0)
            .collect::<Vec<_>>();
        let mut flat_scratch = vec![0.0; matrix.rows()];
        let mut flat_output = vec![0.0; matrix.rows()];
        let mut nested_scratch = vec![0.0; matrix.rows()];
        let mut nested_output = vec![0.0; matrix.rows()];

        preconditioner
            .apply_into(&residual, &mut flat_scratch, &mut flat_output)
            .expect("flat IC(0) application");
        apply_ic0_with_nested_dependencies(
            &preconditioner,
            &nested_dependencies,
            &residual,
            &mut nested_scratch,
            &mut nested_output,
        );
        assert_close(&flat_output, &nested_output, 0.0);

        let mut flat_samples = Vec::with_capacity(ROUNDS);
        let mut nested_samples = Vec::with_capacity(ROUNDS);
        for round in 0..ROUNDS {
            if round % 2 == 0 {
                nested_samples.push(measure_nested_ic0_applications(
                    &preconditioner,
                    &nested_dependencies,
                    &residual,
                    &mut nested_scratch,
                    &mut nested_output,
                    APPLICATIONS_PER_SAMPLE,
                ));
                flat_samples.push(measure_flat_ic0_applications(
                    &preconditioner,
                    &residual,
                    &mut flat_scratch,
                    &mut flat_output,
                    APPLICATIONS_PER_SAMPLE,
                ));
            } else {
                flat_samples.push(measure_flat_ic0_applications(
                    &preconditioner,
                    &residual,
                    &mut flat_scratch,
                    &mut flat_output,
                    APPLICATIONS_PER_SAMPLE,
                ));
                nested_samples.push(measure_nested_ic0_applications(
                    &preconditioner,
                    &nested_dependencies,
                    &residual,
                    &mut nested_scratch,
                    &mut nested_output,
                    APPLICATIONS_PER_SAMPLE,
                ));
            }
        }
        assert_close(&flat_output, &nested_output, 0.0);

        let flat_median = median(&mut flat_samples);
        let nested_median = median(&mut nested_samples);
        println!(
            "ic0-layout-ab rows={} applicationsPerSample={} rounds={} nestedMedianSeconds={nested_median:.6} flatMedianSeconds={flat_median:.6} speedup={:.4}",
            matrix.rows(),
            APPLICATIONS_PER_SAMPLE,
            ROUNDS,
            nested_median / flat_median,
        );
    }

    #[test]
    fn incomplete_cholesky_pcg_is_invariant_to_small_matrix_scale() {
        let scale = 1.0e-20;
        let matrix = CsrMatrix::from_rows(
            vec![
                vec![(0, 4.0 * scale), (1, scale)],
                vec![(0, scale), (1, 3.0 * scale)],
            ],
            2,
        )
        .expect("scaled SPD matrix");
        let report = preconditioned_conjugate_gradient_solve(
            &matrix,
            &[scale, 2.0 * scale],
            None,
            PreconditionedConjugateGradientOptions {
                max_iterations: 8,
                tolerance: 1.0e-32,
                preconditioner: CgPreconditioner::IncompleteCholesky,
            },
        )
        .expect("scaled IC0 PCG solve");

        assert!(report.converged);
        assert_close(&report.solution, &[1.0 / 11.0, 7.0 / 11.0], 1.0e-12);
    }

    #[test]
    fn incomplete_cholesky_rejects_non_positive_pivot() {
        let matrix =
            CsrMatrix::from_rows(vec![vec![(0, 1.0), (1, 2.0)], vec![(0, 2.0), (1, 1.0)]], 2)
                .expect("matrix");
        let error = preconditioned_conjugate_gradient_solve(
            &matrix,
            &[1.0, 1.0],
            None,
            PreconditionedConjugateGradientOptions {
                max_iterations: 4,
                tolerance: 1.0e-12,
                preconditioner: CgPreconditioner::IncompleteCholesky,
            },
        )
        .expect_err("ic0 must reject indefinite pivots");

        assert!(error.to_string().contains("non-positive pivot"));
    }

    #[test]
    fn bicgstab_solves_nonsymmetric_system() {
        let matrix =
            CsrMatrix::from_rows(vec![vec![(0, 4.0), (1, 1.0)], vec![(0, 2.0), (1, 3.0)]], 2)
                .expect("matrix");
        let report = bicgstab_solve(
            &matrix,
            &[1.0, 1.0],
            None,
            BiCgStabOptions {
                max_iterations: 8,
                tolerance: 1.0e-12,
                preconditioner: CgPreconditioner::Diagonal,
            },
        )
        .expect("bicgstab solve");

        assert!(report.converged);
        assert!(report.iterations <= 2);
        assert_close(&report.solution, &[0.2, 0.2], 1.0e-12);
        assert!(report.residual_norm <= 1.0e-12);
    }

    #[test]
    fn conjugate_gradient_reports_breakdown_as_not_converged() {
        let matrix = CsrMatrix::from_rows(vec![vec![(0, 0.0)]], 1).expect("matrix");
        let report = conjugate_gradient_solve(
            &matrix,
            &[1.0],
            None,
            ConjugateGradientOptions {
                max_iterations: 10,
                tolerance: 1.0e-12,
            },
        )
        .expect("breakdown should return a non-converged report");

        assert!(!report.converged);
        assert_eq!(report.termination, IterativeSolveTermination::Breakdown);
        assert_eq!(report.iterations, 1);
        assert_close(&report.solution, &[0.0], 1.0e-14);
        assert_eq!(report.residual_norm, 1.0);
    }

    #[test]
    fn conjugate_gradient_reports_iteration_budget_and_convergence() {
        let matrix = CsrMatrix::from_rows(vec![vec![(0, 2.0)]], 1).expect("matrix");
        let exhausted = conjugate_gradient_solve(
            &matrix,
            &[1.0],
            None,
            ConjugateGradientOptions {
                max_iterations: 0,
                tolerance: 1.0e-12,
            },
        )
        .expect("zero-budget solve");
        assert_eq!(
            exhausted.termination,
            IterativeSolveTermination::MaxIterations
        );
        assert!(!exhausted.converged);

        let converged = conjugate_gradient_solve(
            &matrix,
            &[1.0],
            None,
            ConjugateGradientOptions {
                max_iterations: 1,
                tolerance: 1.0e-12,
            },
        )
        .expect("converged solve");
        assert_eq!(converged.termination, IterativeSolveTermination::Converged);
        assert!(converged.converged);
    }

    #[test]
    fn unpreconditioned_pcg_reports_zero_matrix_breakdown() {
        let matrix = CsrMatrix::from_rows(vec![vec![(0, 0.0)]], 1).expect("matrix");
        let report = preconditioned_conjugate_gradient_solve(
            &matrix,
            &[1.0],
            None,
            PreconditionedConjugateGradientOptions {
                max_iterations: 10,
                tolerance: 1.0e-12,
                preconditioner: CgPreconditioner::None,
            },
        )
        .expect("breakdown report");

        assert_eq!(report.termination, IterativeSolveTermination::Breakdown);
        assert!(!report.converged);
        assert_eq!(report.iterations, 1);
        assert_eq!(report.residual_norm, 1.0);
    }

    #[test]
    fn reports_residual_as_rhs_minus_matrix_vector() {
        let matrix = poisson3();

        let result = residual(&matrix, &[0.5, 1.0, 0.5], &[1.0, 0.0, 1.0]).expect("residual");

        assert_close(&result, &[1.0, -1.0, 1.0], 1.0e-14);
    }

    fn nested_dependencies(
        preconditioner: &IncompleteCholeskyPreconditioner,
    ) -> Vec<Vec<(usize, usize)>> {
        (0..preconditioner.diagonal_factor_slots.len())
            .map(|row| {
                preconditioner.dependent_entries[preconditioner.dependent_row_offsets[row]
                    ..preconditioner.dependent_row_offsets[row + 1]]
                    .to_vec()
            })
            .collect()
    }

    fn apply_ic0_with_nested_dependencies(
        preconditioner: &IncompleteCholeskyPreconditioner,
        dependencies: &[Vec<(usize, usize)>],
        residual: &[f64],
        scratch: &mut [f64],
        output: &mut [f64],
    ) {
        assert_eq!(residual.len(), preconditioner.diagonal_factor_slots.len());
        assert_eq!(scratch.len(), residual.len());
        assert_eq!(output.len(), residual.len());
        assert_eq!(dependencies.len(), residual.len());
        for row in 0..residual.len() {
            let mut sum = residual[row];
            let start = preconditioner.lower_row_offsets[row];
            let diagonal_slot = preconditioner.diagonal_factor_slots[row];
            for factor_slot in start..diagonal_slot {
                sum -= preconditioner.factors[factor_slot]
                    * scratch[preconditioner.lower_columns[factor_slot]];
            }
            scratch[row] = sum / preconditioner.factors[diagonal_slot];
        }
        for row in (0..residual.len()).rev() {
            let mut sum = scratch[row];
            for &(dependent_row, factor_slot) in &dependencies[row] {
                sum -= preconditioner.factors[factor_slot] * output[dependent_row];
            }
            output[row] = sum / preconditioner.factors[preconditioner.diagonal_factor_slots[row]];
        }
    }

    fn measure_flat_ic0_applications(
        preconditioner: &IncompleteCholeskyPreconditioner,
        residual: &[f64],
        scratch: &mut [f64],
        output: &mut [f64],
        applications: usize,
    ) -> f64 {
        let started = Instant::now();
        for _ in 0..applications {
            preconditioner
                .apply_into(std::hint::black_box(residual), scratch, output)
                .expect("flat IC(0) benchmark application");
            std::hint::black_box(&*output);
        }
        started.elapsed().as_secs_f64()
    }

    fn measure_nested_ic0_applications(
        preconditioner: &IncompleteCholeskyPreconditioner,
        dependencies: &[Vec<(usize, usize)>],
        residual: &[f64],
        scratch: &mut [f64],
        output: &mut [f64],
        applications: usize,
    ) -> f64 {
        let started = Instant::now();
        for _ in 0..applications {
            apply_ic0_with_nested_dependencies(
                preconditioner,
                dependencies,
                std::hint::black_box(residual),
                scratch,
                output,
            );
            std::hint::black_box(&*output);
        }
        started.elapsed().as_secs_f64()
    }

    fn median(samples: &mut [f64]) -> f64 {
        samples.sort_by(f64::total_cmp);
        samples[samples.len() / 2]
    }

    fn dic_oracle_matrix() -> CsrMatrix {
        CsrMatrix::new(
            3,
            3,
            vec![0, 2, 5, 7],
            vec![0, 1, 0, 1, 2, 1, 2],
            vec![4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 3.0],
        )
        .expect("DIC oracle matrix")
    }

    fn face_ldu_workspace_pointers(
        workspace: &PreconditionedConjugateGradientWorkspace,
    ) -> [usize; 17] {
        let preconditioner = workspace
            .face_ldu_preconditioner
            .as_ref()
            .expect("face-LDU preconditioner");
        [
            preconditioner.pattern.sparsity.row_offsets.as_ptr() as usize,
            preconditioner.pattern.sparsity.col_indices.as_ptr() as usize,
            preconditioner.pattern.diagonal_slots.as_ptr() as usize,
            preconditioner.pattern.lower_addr.as_ptr() as usize,
            preconditioner.pattern.upper_addr.as_ptr() as usize,
            preconditioner.pattern.owner_start.as_ptr() as usize,
            preconditioner.pattern.lower_matrix_slots.as_ptr() as usize,
            preconditioner.pattern.upper_matrix_slots.as_ptr() as usize,
            preconditioner.upper_coefficients.as_ptr() as usize,
            preconditioner.reciprocal_diagonal.as_ptr() as usize,
            preconditioner.fdic_upper_scaled_by_upper_diagonal.as_ptr() as usize,
            preconditioner.fdic_upper_scaled_by_lower_diagonal.as_ptr() as usize,
            workspace.residual.as_ptr() as usize,
            workspace.preconditioned_residual.as_ptr() as usize,
            workspace.direction.as_ptr() as usize,
            workspace.matrix_direction.as_ptr() as usize,
            workspace.preconditioner_scratch.as_ptr() as usize,
        ]
    }

    fn poisson_grid(nx: usize, ny: usize) -> CsrMatrix {
        let cells = nx.checked_mul(ny).expect("Poisson grid size");
        let mut rows = Vec::with_capacity(cells);
        for cell in 0..cells {
            let i = cell % nx;
            let j = cell / nx;
            let mut row = Vec::with_capacity(5);
            if j > 0 {
                row.push((cell - nx, -1.0));
            }
            if i > 0 {
                row.push((cell - 1, -1.0));
            }
            row.push((cell, 5.0));
            if i + 1 < nx {
                row.push((cell + 1, -1.0));
            }
            if j + 1 < ny {
                row.push((cell + nx, -1.0));
            }
            rows.push(row);
        }
        CsrMatrix::from_rows(rows, cells).expect("Poisson grid matrix")
    }

    fn poisson3() -> CsrMatrix {
        CsrMatrix::new(
            3,
            3,
            vec![0, 2, 5, 7],
            vec![0, 1, 0, 1, 2, 1, 2],
            vec![2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0],
        )
        .expect("poisson matrix")
    }

    fn assert_close(actual: &[f64], expected: &[f64], tolerance: f64) {
        assert_eq!(actual.len(), expected.len());
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= tolerance,
                "entry {index}: expected {expected}, got {actual}"
            );
        }
    }
}
