use std::collections::BTreeMap;

use crate::{MeshError, Result};

#[derive(Clone, Debug)]
pub struct CsrMatrix {
    rows: usize,
    cols: usize,
    row_offsets: Vec<usize>,
    col_indices: Vec<usize>,
    values: Vec<f64>,
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
    pub cpu_diagonal_preconditioner: bool,
    pub cpu_incomplete_cholesky_preconditioner: bool,
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
}

impl CsrMatrix {
    pub fn new(
        rows: usize,
        cols: usize,
        row_offsets: Vec<usize>,
        col_indices: Vec<usize>,
        values: Vec<f64>,
    ) -> Result<Self> {
        validate_csr(rows, cols, &row_offsets, &col_indices, &values)?;
        Ok(Self {
            rows,
            cols,
            row_offsets,
            col_indices,
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
        cpu_diagonal_preconditioner: true,
        cpu_incomplete_cholesky_preconditioner: true,
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
            });
        }
    }

    Ok(IterativeSolveReport {
        solution: x,
        iterations: options.max_iterations,
        residual_norm,
        converged: false,
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
            });
        }
    }

    Ok(IterativeSolveReport {
        solution: x,
        iterations: options.max_iterations,
        residual_norm,
        converged: false,
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
            });
        }
    }

    Ok(IterativeSolveReport {
        solution: x,
        iterations: options.max_iterations,
        residual_norm,
        converged: false,
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
                iterations: iteration.saturating_sub(1),
                residual_norm,
                converged: false,
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
    })
}

pub fn preconditioned_conjugate_gradient_solve(
    matrix: &CsrMatrix,
    rhs: &[f64],
    initial: Option<&[f64]>,
    options: PreconditionedConjugateGradientOptions,
) -> Result<IterativeSolveReport> {
    validate_iterative_solve_input(matrix, rhs, initial, options.tolerance)?;

    let preconditioner = BuiltPreconditioner::build(matrix, options.preconditioner)?;
    let mut x = initial
        .map(|values| values.to_vec())
        .unwrap_or_else(|| vec![0.0; rhs.len()]);
    let mut r = residual(matrix, &x, rhs)?;
    let mut residual_norm = l2_norm(&r);
    if residual_norm <= options.tolerance {
        return Ok(IterativeSolveReport {
            solution: x,
            iterations: 0,
            residual_norm,
            converged: true,
        });
    }

    let mut z = preconditioner.apply(&r);
    let mut rz = dot(&r, &z);
    if !rz.is_finite() {
        return Err(invalid_input(
            "preconditioned conjugate-gradient residual product is not finite".to_string(),
        ));
    }
    if dot_product_is_singular(rz, &r, &z) {
        return Ok(IterativeSolveReport {
            solution: x,
            iterations: 0,
            residual_norm,
            converged: false,
        });
    }

    let mut p = z.clone();
    for iteration in 1..=options.max_iterations {
        let ap = matrix.matvec(&p)?;
        let denominator = dot(&p, &ap);
        if !denominator.is_finite() {
            return Err(invalid_input(
                "preconditioned conjugate-gradient denominator is not finite; matrix is likely not SPD"
                    .to_string(),
            ));
        }
        if dot_product_is_singular(denominator, &p, &ap) {
            return Ok(IterativeSolveReport {
                solution: x,
                iterations: iteration.saturating_sub(1),
                residual_norm,
                converged: false,
            });
        }

        let alpha = rz / denominator;
        for row in 0..x.len() {
            x[row] += alpha * p[row];
            r[row] -= alpha * ap[row];
        }

        residual_norm = l2_norm(&r);
        if residual_norm <= options.tolerance {
            return Ok(IterativeSolveReport {
                solution: x,
                iterations: iteration,
                residual_norm,
                converged: true,
            });
        }

        z = preconditioner.apply(&r);
        let next_rz = dot(&r, &z);
        if !next_rz.is_finite() {
            return Err(invalid_input(
                "preconditioned conjugate-gradient residual product is not finite".to_string(),
            ));
        }
        if dot_product_is_singular(next_rz, &r, &z) {
            return Ok(IterativeSolveReport {
                solution: x,
                iterations: iteration,
                residual_norm,
                converged: false,
            });
        }
        let beta = next_rz / rz;
        for row in 0..p.len() {
            p[row] = z[row] + beta * p[row];
        }
        rz = next_rz;
    }

    Ok(IterativeSolveReport {
        solution: x,
        iterations: options.max_iterations,
        residual_norm,
        converged: false,
    })
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
                iterations: iteration.saturating_sub(1),
                residual_norm,
                converged: false,
            });
        }

        let beta = (rho / rho_old) * (alpha / omega);
        for row in 0..p.len() {
            p[row] = r[row] + beta * (p[row] - omega * v[row]);
        }

        let p_hat = preconditioner.apply(&p);
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
                iterations: iteration.saturating_sub(1),
                residual_norm,
                converged: false,
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
            });
        }

        let s_hat = preconditioner.apply(&s);
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
                iterations: iteration.saturating_sub(1),
                residual_norm,
                converged: false,
            });
        }
        omega = dot(&t, &s) / omega_denominator;
        if !omega.is_finite() {
            return Err(invalid_input("BiCGStab omega is not finite".to_string()));
        }
        if omega == 0.0 {
            return Ok(IterativeSolveReport {
                solution: x,
                iterations: iteration.saturating_sub(1),
                residual_norm,
                converged: false,
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
            });
        }
        rho_old = rho;
    }

    Ok(IterativeSolveReport {
        solution: x,
        iterations: options.max_iterations,
        residual_norm,
        converged: false,
    })
}

enum BuiltPreconditioner {
    None,
    Diagonal(Vec<f64>),
    IncompleteCholesky(IncompleteCholeskyPreconditioner),
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
            CgPreconditioner::IncompleteCholesky => Ok(Self::IncompleteCholesky(
                IncompleteCholeskyPreconditioner::build(matrix)?,
            )),
        }
    }

    fn apply(&self, residual: &[f64]) -> Vec<f64> {
        match self {
            Self::None => residual.to_vec(),
            Self::Diagonal(inverse) => residual
                .iter()
                .zip(inverse)
                .map(|(value, inverse)| value * inverse)
                .collect(),
            Self::IncompleteCholesky(preconditioner) => preconditioner.apply(residual),
        }
    }
}

struct IncompleteCholeskyPreconditioner {
    lower_rows: Vec<Vec<(usize, f64)>>,
    lower_columns: Vec<Vec<(usize, f64)>>,
}

impl IncompleteCholeskyPreconditioner {
    fn build(matrix: &CsrMatrix) -> Result<Self> {
        if matrix.rows != matrix.cols {
            return Err(invalid_input(format!(
                "incomplete Cholesky preconditioner requires a square matrix, got {}x{}",
                matrix.rows, matrix.cols
            )));
        }

        let mut lower = vec![BTreeMap::<usize, f64>::new(); matrix.rows];
        for (row, lower_row) in lower.iter_mut().enumerate() {
            for entry in matrix.row_offsets[row]..matrix.row_offsets[row + 1] {
                let column = matrix.col_indices[entry];
                if column <= row {
                    lower_row.insert(column, matrix.values[entry]);
                }
            }
            if !lower_row.contains_key(&row) {
                return Err(invalid_input(format!(
                    "incomplete Cholesky preconditioner row {row} has no diagonal entry"
                )));
            }
        }

        for row in 0..matrix.rows {
            let off_diagonal_columns = lower[row]
                .keys()
                .copied()
                .filter(|column| *column < row)
                .collect::<Vec<_>>();
            for column in off_diagonal_columns {
                let mut value = *lower[row].get(&column).unwrap_or(&0.0);
                for (&shared, &row_value) in lower[row].range(..column) {
                    if let Some(column_value) = lower[column].get(&shared) {
                        value -= row_value * column_value;
                    }
                }
                let pivot = *lower[column].get(&column).ok_or_else(|| {
                    invalid_input(format!(
                        "incomplete Cholesky preconditioner row {column} has no computed pivot"
                    ))
                })?;
                if !pivot.is_finite() || pivot <= 0.0 {
                    return Err(invalid_input(format!(
                        "incomplete Cholesky preconditioner row {column} has invalid pivot {pivot}"
                    )));
                }
                lower[row].insert(column, value / pivot);
            }

            let mut diagonal = *lower[row].get(&row).unwrap_or(&0.0);
            for (&column, &value) in lower[row].range(..row) {
                let _ = column;
                diagonal -= value * value;
            }
            if !diagonal.is_finite() || diagonal <= 0.0 {
                return Err(invalid_input(format!(
                    "incomplete Cholesky preconditioner row {row} has non-positive pivot square {diagonal}"
                )));
            }
            lower[row].insert(row, diagonal.sqrt());
        }

        let lower_rows = lower
            .into_iter()
            .map(|row| row.into_iter().collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let mut lower_columns = vec![Vec::<(usize, f64)>::new(); matrix.rows];
        for (row, entries) in lower_rows.iter().enumerate() {
            for &(column, value) in entries {
                if column < row {
                    lower_columns[column].push((row, value));
                }
            }
        }

        Ok(Self {
            lower_rows,
            lower_columns,
        })
    }

    fn apply(&self, residual: &[f64]) -> Vec<f64> {
        let mut y = vec![0.0; residual.len()];
        for row in 0..residual.len() {
            let mut sum = residual[row];
            let mut diagonal = 1.0;
            for &(column, value) in &self.lower_rows[row] {
                if column < row {
                    sum -= value * y[column];
                } else if column == row {
                    diagonal = value;
                }
            }
            y[row] = sum / diagonal;
        }

        let mut x = vec![0.0; residual.len()];
        for row in (0..residual.len()).rev() {
            let mut sum = y[row];
            for &(dependent_row, value) in &self.lower_columns[row] {
                sum -= value * x[dependent_row];
            }
            let diagonal = self.lower_rows[row]
                .iter()
                .find_map(|(column, value)| (*column == row).then_some(*value))
                .unwrap_or(1.0);
            x[row] = sum / diagonal;
        }
        x
    }
}

fn validate_csr(
    rows: usize,
    cols: usize,
    row_offsets: &[usize],
    col_indices: &[usize],
    values: &[f64],
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
    if col_indices.len() != values.len() {
        return Err(invalid_input(format!(
            "CSR col_indices length {} does not match values length {}",
            col_indices.len(),
            values.len()
        )));
    }
    if row_offsets.last().copied() != Some(values.len()) {
        return Err(invalid_input(format!(
            "CSR final row offset must equal nnz {}, got {}",
            values.len(),
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

fn dot_product_is_singular(value: f64, left: &[f64], right: &[f64]) -> bool {
    let scale = l2_norm(left) * l2_norm(right);
    scale == 0.0 || !scale.is_finite() || value.abs() <= f64::EPSILON * scale
}

fn invalid_input(message: String) -> MeshError {
    MeshError::InvalidInput(message)
}

#[cfg(test)]
mod tests {
    use super::{
        BiCgStabOptions, CgPreconditioner, ConjugateGradientOptions, CsrMatrix, GaussSeidelOptions,
        JacobiOptions, PreconditionedConjugateGradientOptions, bicgstab_solve,
        conjugate_gradient_solve, gauss_seidel_solve, jacobi_solve,
        preconditioned_conjugate_gradient_solve, residual, symmetric_gauss_seidel_solve,
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
        assert_eq!(report.iterations, 0);
        assert_close(&report.solution, &[0.0], 1.0e-14);
        assert_eq!(report.residual_norm, 1.0);
    }

    #[test]
    fn reports_residual_as_rhs_minus_matrix_vector() {
        let matrix = poisson3();

        let result = residual(&matrix, &[0.5, 1.0, 0.5], &[1.0, 0.0, 1.0]).expect("residual");

        assert_close(&result, &[1.0, -1.0, 1.0], 1.0e-14);
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
