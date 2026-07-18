use std::f64::consts::PI;
use std::time::Instant;

use ferrum_mesh::linear::{
    CgPreconditioner, CsrMatrix, PreconditionedConjugateGradientOptions,
    PreconditionedConjugateGradientWorkspace, l2_norm, preconditioned_conjugate_gradient_solve,
    residual,
};

#[derive(Clone, Copy)]
struct PressureMeshCase {
    name: &'static str,
    nx: usize,
    ny: usize,
    shear: f64,
}

#[test]
fn reusable_ic0_pressure_mesh_gate_covers_medium_fine_and_skewed_systems() {
    // Medium and fine row counts track the existing Gmsh pipe meshes closely.
    let cases = [
        PressureMeshCase {
            name: "medium",
            nx: 128,
            ny: 108,
            shear: 0.0,
        },
        PressureMeshCase {
            name: "fine",
            nx: 256,
            ny: 152,
            shear: 0.0,
        },
        PressureMeshCase {
            name: "skewed",
            nx: 128,
            ny: 96,
            shear: 1.5,
        },
    ];

    for case in cases {
        run_pressure_mesh_case(case);
    }
}

fn run_pressure_mesh_case(case: PressureMeshCase) {
    let started = Instant::now();
    let first_matrix = build_pressure_matrix(case, 0.0);
    let pattern = first_matrix.sparsity_pattern();
    let second_unshared = build_pressure_matrix(case, 0.65);
    assert_eq!(first_matrix.row_offsets(), second_unshared.row_offsets());
    assert_eq!(first_matrix.col_indices(), second_unshared.col_indices());
    let second_matrix = CsrMatrix::from_pattern(&pattern, second_unshared.values().to_vec())
        .expect("second pressure matrix with shared topology");

    let exact = exact_pressure(case);
    let first_rhs = first_matrix.matvec(&exact).expect("first pressure rhs");
    let second_rhs = second_matrix.matvec(&exact).expect("second pressure rhs");
    let options = PreconditionedConjugateGradientOptions {
        max_iterations: 2_000,
        tolerance: 1.0e-10,
        preconditioner: CgPreconditioner::IncompleteCholesky,
    };
    let mut workspace = PreconditionedConjugateGradientWorkspace::new(
        &first_matrix,
        CgPreconditioner::IncompleteCholesky,
    )
    .expect("reusable pressure PCG/IC(0) workspace");

    let first = workspace
        .solve(&first_matrix, &first_rhs, None, options)
        .expect("first reusable pressure solve");
    let profiled = workspace
        .solve_profiled(&second_matrix, &second_rhs, None, options)
        .expect("profiled refactored reusable pressure solve");
    let refactored = profiled.report;
    let fresh = preconditioned_conjugate_gradient_solve(&second_matrix, &second_rhs, None, options)
        .expect("fresh pressure solve");

    assert!(first.converged, "{} first pressure solve", case.name);
    assert!(
        refactored.converged,
        "{} refactored pressure solve",
        case.name
    );
    assert!(fresh.converged, "{} fresh pressure solve", case.name);
    assert_eq!(refactored.iterations, fresh.iterations, "{}", case.name);
    assert_eq!(
        refactored.residual_norm, fresh.residual_norm,
        "{}",
        case.name
    );
    assert_eq!(refactored.solution, fresh.solution, "{}", case.name);
    assert_eq!(
        profiled.timing.matrix_vector_products,
        refactored.iterations + 1,
        "{}",
        case.name
    );
    assert_eq!(
        profiled.timing.preconditioner_applications, refactored.iterations,
        "{}",
        case.name
    );

    let first_relative_residual =
        relative_true_residual(&first_matrix, &first.solution, &first_rhs);
    let refactored_relative_residual =
        relative_true_residual(&second_matrix, &refactored.solution, &second_rhs);
    let first_relative_error = relative_solution_error(&first.solution, &exact);
    let refactored_relative_error = relative_solution_error(&refactored.solution, &exact);
    assert!(
        first_relative_residual <= 1.0e-8,
        "{} first true relative residual {}",
        case.name,
        first_relative_residual
    );
    assert!(
        refactored_relative_residual <= 1.0e-8,
        "{} refactored true relative residual {}",
        case.name,
        refactored_relative_residual
    );
    assert!(
        first_relative_error <= 1.0e-6,
        "{} first relative solution error {}",
        case.name,
        first_relative_error
    );
    assert!(
        refactored_relative_error <= 1.0e-6,
        "{} refactored relative solution error {}",
        case.name,
        refactored_relative_error
    );

    println!(
        "pressure-mesh-gate case={} rows={} nnz={} shearDegrees={:.3} firstIterations={} refactoredIterations={} trueRelativeResidual={:.6e} relativeSolutionError={:.6e} pcgTotalSeconds={:.6} preconditionerUpdateSeconds={:.6} matrixVectorSeconds={:.6} preconditionerApplicationSeconds={:.6} vectorOperationSeconds={:.6} otherSeconds={:.6} matrixVectorProducts={} preconditionerApplications={} elapsedSeconds={:.6}",
        case.name,
        first_matrix.rows(),
        first_matrix.nnz(),
        case.shear.atan().to_degrees(),
        first.iterations,
        refactored.iterations,
        refactored_relative_residual,
        refactored_relative_error,
        profiled.timing.total_seconds,
        profiled.timing.preconditioner_update_seconds,
        profiled.timing.matrix_vector_seconds,
        profiled.timing.preconditioner_application_seconds,
        profiled.timing.vector_operation_seconds,
        profiled.timing.other_seconds,
        profiled.timing.matrix_vector_products,
        profiled.timing.preconditioner_applications,
        started.elapsed().as_secs_f64(),
    );
}

fn build_pressure_matrix(case: PressureMeshCase, heterogeneity: f64) -> CsrMatrix {
    let cells = case.nx.checked_mul(case.ny).expect("pressure grid size");
    let dx = 1.0 / case.nx as f64;
    let dy = 1.0 / case.ny as f64;
    let x_transmissibility = dy * (1.0 + case.shear * case.shear) / dx;
    let y_transmissibility = dx / dy;
    let diffusivity = (0..cells)
        .map(|cell| {
            let (i, j) = cell_coordinates(cell, case.nx);
            let x = (i as f64 + 0.5) * dx;
            let y = (j as f64 + 0.5) * dy;
            let shape = 0.45 * (2.0 * PI * x).sin() * (PI * y).cos() + 0.35 * (3.0 * PI * y).cos();
            let value = 1.0 + heterogeneity * shape;
            assert!(value > 0.0 && value.is_finite());
            value
        })
        .collect::<Vec<_>>();

    let mut rows = Vec::with_capacity(cells);
    for cell in 0..cells {
        let (i, j) = cell_coordinates(cell, case.nx);
        let mut row = Vec::with_capacity(5);
        let mut diagonal = 0.0;

        for (neighbour, base) in [
            (
                i.checked_sub(1).map(|x| cell_index(x, j, case.nx)),
                x_transmissibility,
            ),
            (
                (i + 1 < case.nx).then(|| cell_index(i + 1, j, case.nx)),
                x_transmissibility,
            ),
            (
                j.checked_sub(1).map(|y| cell_index(i, y, case.nx)),
                y_transmissibility,
            ),
            (
                (j + 1 < case.ny).then(|| cell_index(i, j + 1, case.nx)),
                y_transmissibility,
            ),
        ] {
            if let Some(neighbour) = neighbour {
                let coefficient = base * harmonic_mean(diffusivity[cell], diffusivity[neighbour]);
                diagonal += coefficient;
                row.push((neighbour, -coefficient));
            } else {
                diagonal += 2.0 * base * diffusivity[cell];
            }
        }

        row.push((cell, diagonal));
        row.sort_unstable_by_key(|(column, _)| *column);
        rows.push(row);
    }

    CsrMatrix::from_rows(rows, cells).expect("conservative pressure matrix")
}

fn exact_pressure(case: PressureMeshCase) -> Vec<f64> {
    let dx = 1.0 / case.nx as f64;
    let dy = 1.0 / case.ny as f64;
    (0..case.nx * case.ny)
        .map(|cell| {
            let (i, j) = cell_coordinates(cell, case.nx);
            let x = (i as f64 + 0.5) * dx;
            let y = (j as f64 + 0.5) * dy;
            (PI * x).sin() * (PI * y).sin() + 0.1 * (3.0 * PI * x).sin() * (PI * y).sin()
        })
        .collect()
}

fn relative_true_residual(matrix: &CsrMatrix, solution: &[f64], rhs: &[f64]) -> f64 {
    let true_residual = residual(matrix, solution, rhs).expect("true pressure residual");
    l2_norm(&true_residual) / l2_norm(rhs).max(f64::MIN_POSITIVE)
}

fn relative_solution_error(solution: &[f64], exact: &[f64]) -> f64 {
    let difference = solution
        .iter()
        .zip(exact)
        .map(|(actual, expected)| actual - expected)
        .collect::<Vec<_>>();
    l2_norm(&difference) / l2_norm(exact).max(f64::MIN_POSITIVE)
}

fn harmonic_mean(left: f64, right: f64) -> f64 {
    2.0 * left * right / (left + right)
}

fn cell_coordinates(cell: usize, nx: usize) -> (usize, usize) {
    (cell % nx, cell / nx)
}

fn cell_index(i: usize, j: usize, nx: usize) -> usize {
    j * nx + i
}
