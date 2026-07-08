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
    pub cpu_conjugate_gradient: bool,
    pub gpu_linear_solvers: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct JacobiOptions {
    pub max_iterations: usize,
    pub tolerance: f64,
    pub omega: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct ConjugateGradientOptions {
    pub max_iterations: usize,
    pub tolerance: f64,
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

impl Default for ConjugateGradientOptions {
    fn default() -> Self {
        Self {
            max_iterations: 1_000,
            tolerance: 1.0e-10,
        }
    }
}

pub fn linear_solver_capabilities() -> LinearSolverCapabilities {
    LinearSolverCapabilities {
        cpu_csr: true,
        cpu_jacobi: true,
        cpu_conjugate_gradient: true,
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
        if *value == 0.0 {
            return Err(invalid_input(format!(
                "row {row} has a zero diagonal entry"
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
        if !denominator.is_finite() || denominator == 0.0 {
            return Err(invalid_input(
                "conjugate-gradient denominator is zero; matrix is likely singular or not SPD"
                    .to_string(),
            ));
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

fn invalid_input(message: String) -> MeshError {
    MeshError::InvalidInput(message)
}

#[cfg(test)]
mod tests {
    use super::{
        ConjugateGradientOptions, CsrMatrix, JacobiOptions, conjugate_gradient_solve, jacobi_solve,
        residual,
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
