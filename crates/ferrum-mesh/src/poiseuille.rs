use crate::diffusion::{
    ScalarBoundaryCondition, ScalarDiffusionOptions, ScalarPatchBoundaryCondition,
};
use crate::runtime::SolverRuntimeMeshData;
use crate::{MeshError, Result};

#[derive(Clone, Debug)]
pub struct PoiseuilleOptions {
    pub pressure_drop: f64,
    pub dynamic_viscosity: f64,
    pub length: f64,
    pub diameter: f64,
    pub wall_patches: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct PoiseuilleReference {
    pub source: f64,
    pub cross_section_area: f64,
    pub mean_velocity: f64,
    pub flow_rate: f64,
}

#[derive(Clone, Debug)]
pub struct PoiseuilleSolutionSummary {
    pub min_velocity: f64,
    pub max_velocity: f64,
    pub mean_velocity: f64,
    pub flow_rate: f64,
    pub analytic_mean_velocity: f64,
    pub analytic_flow_rate: f64,
    pub pressure_drop_from_mean: f64,
    pub mean_velocity_error: f64,
    pub relative_mean_velocity_error: f64,
}

pub fn poiseuille_diffusion_options(options: &PoiseuilleOptions) -> Result<ScalarDiffusionOptions> {
    validate_poiseuille_options(options)?;
    Ok(ScalarDiffusionOptions {
        diffusivity: options.dynamic_viscosity,
        source: options.pressure_drop / options.length,
        default_boundary: ScalarBoundaryCondition::ZeroGradient,
        patch_boundary_conditions: options
            .wall_patches
            .iter()
            .map(|patch| ScalarPatchBoundaryCondition {
                patch: patch.clone(),
                condition: ScalarBoundaryCondition::FixedValue(0.0),
            })
            .collect(),
    })
}

pub fn poiseuille_reference(options: &PoiseuilleOptions) -> Result<PoiseuilleReference> {
    validate_poiseuille_options(options)?;
    let cross_section_area = std::f64::consts::PI * options.diameter * options.diameter / 4.0;
    let mean_velocity = options.pressure_drop * options.diameter * options.diameter
        / (32.0 * options.dynamic_viscosity * options.length);
    Ok(PoiseuilleReference {
        source: options.pressure_drop / options.length,
        cross_section_area,
        mean_velocity,
        flow_rate: mean_velocity * cross_section_area,
    })
}

pub fn summarize_poiseuille_solution(
    mesh: &SolverRuntimeMeshData,
    solution: &[f64],
    options: &PoiseuilleOptions,
) -> Result<PoiseuilleSolutionSummary> {
    if solution.len() != mesh.cells {
        return Err(invalid_input(format!(
            "Poiseuille solution has {} values, expected {} mesh cells",
            solution.len(),
            mesh.cells
        )));
    }
    if mesh.cell_volumes.len() != mesh.cells {
        return Err(invalid_input(format!(
            "runtime mesh cell_volumes length {} does not match cells {}",
            mesh.cell_volumes.len(),
            mesh.cells
        )));
    }

    let reference = poiseuille_reference(options)?;
    let mut min_velocity = f64::INFINITY;
    let mut max_velocity = f64::NEG_INFINITY;
    let mut weighted_sum = 0.0;
    let mut total_volume = 0.0;
    for (velocity, volume) in solution.iter().zip(&mesh.cell_volumes) {
        if !velocity.is_finite() {
            return Err(invalid_input(
                "Poiseuille solution contains a non-finite value".to_string(),
            ));
        }
        if !volume.is_finite() || *volume < 0.0 {
            return Err(invalid_input(format!(
                "runtime mesh cell volume must be finite and non-negative, got {volume}"
            )));
        }
        min_velocity = min_velocity.min(*velocity);
        max_velocity = max_velocity.max(*velocity);
        weighted_sum += velocity * volume;
        total_volume += volume;
    }
    if total_volume <= 0.0 || !total_volume.is_finite() {
        return Err(invalid_input(format!(
            "runtime mesh total volume must be positive and finite, got {total_volume}"
        )));
    }

    let mean_velocity = weighted_sum / total_volume;
    let flow_rate = mean_velocity * reference.cross_section_area;
    let pressure_drop_from_mean = 32.0 * options.dynamic_viscosity * options.length * mean_velocity
        / (options.diameter * options.diameter);
    let mean_velocity_error = mean_velocity - reference.mean_velocity;
    let relative_mean_velocity_error = if reference.mean_velocity.abs() > f64::EPSILON {
        mean_velocity_error / reference.mean_velocity
    } else {
        0.0
    };

    Ok(PoiseuilleSolutionSummary {
        min_velocity,
        max_velocity,
        mean_velocity,
        flow_rate,
        analytic_mean_velocity: reference.mean_velocity,
        analytic_flow_rate: reference.flow_rate,
        pressure_drop_from_mean,
        mean_velocity_error,
        relative_mean_velocity_error,
    })
}

fn validate_poiseuille_options(options: &PoiseuilleOptions) -> Result<()> {
    if !options.pressure_drop.is_finite() || options.pressure_drop <= 0.0 {
        return Err(invalid_input(format!(
            "Poiseuille pressure drop must be positive and finite, got {}",
            options.pressure_drop
        )));
    }
    if !options.dynamic_viscosity.is_finite() || options.dynamic_viscosity <= 0.0 {
        return Err(invalid_input(format!(
            "Poiseuille dynamic viscosity must be positive and finite, got {}",
            options.dynamic_viscosity
        )));
    }
    if !options.length.is_finite() || options.length <= 0.0 {
        return Err(invalid_input(format!(
            "Poiseuille length must be positive and finite, got {}",
            options.length
        )));
    }
    if !options.diameter.is_finite() || options.diameter <= 0.0 {
        return Err(invalid_input(format!(
            "Poiseuille diameter must be positive and finite, got {}",
            options.diameter
        )));
    }
    if options.wall_patches.is_empty() {
        return Err(invalid_input(
            "Poiseuille solve requires at least one wall patch".to_string(),
        ));
    }
    for patch in &options.wall_patches {
        if patch.trim().is_empty() {
            return Err(invalid_input(
                "Poiseuille wall patch name must not be empty".to_string(),
            ));
        }
    }
    Ok(())
}

fn invalid_input(message: String) -> MeshError {
    MeshError::InvalidInput(message)
}

#[cfg(test)]
mod tests {
    use crate::Point3;
    use crate::runtime::{SolverRuntimeMeshData, SolverRuntimePatchRange};

    use super::{
        PoiseuilleOptions, poiseuille_diffusion_options, poiseuille_reference,
        summarize_poiseuille_solution,
    };

    #[test]
    fn builds_source_driven_diffusion_options() {
        let options = test_options();

        let diffusion = poiseuille_diffusion_options(&options).expect("diffusion options");

        assert_eq!(diffusion.diffusivity, 0.001);
        assert_eq!(diffusion.source, 2.0);
        assert_eq!(diffusion.patch_boundary_conditions.len(), 1);
        assert_eq!(diffusion.patch_boundary_conditions[0].patch, "wall");
    }

    #[test]
    fn computes_hagen_poiseuille_reference() {
        let options = test_options();

        let reference = poiseuille_reference(&options).expect("reference");

        assert_close(reference.source, 2.0, 1.0e-14);
        assert_close(reference.mean_velocity, 0.00625, 1.0e-14);
        assert_close(
            reference.flow_rate,
            0.00625 * std::f64::consts::PI * 0.01 * 0.01 / 4.0,
            1.0e-14,
        );
    }

    #[test]
    fn summarizes_volume_weighted_solution_against_reference() {
        let mesh = two_cell_mesh();
        let options = test_options();

        let summary =
            summarize_poiseuille_solution(&mesh, &[0.005, 0.0075], &options).expect("summary");

        assert_close(summary.mean_velocity, 0.00625, 1.0e-14);
        assert_close(summary.mean_velocity_error, 0.0, 1.0e-14);
        assert_close(summary.relative_mean_velocity_error, 0.0, 1.0e-14);
        assert_close(
            summary.pressure_drop_from_mean,
            options.pressure_drop,
            1.0e-14,
        );
    }

    fn test_options() -> PoiseuilleOptions {
        PoiseuilleOptions {
            pressure_drop: 2.0,
            dynamic_viscosity: 0.001,
            length: 1.0,
            diameter: 0.01,
            wall_patches: vec!["wall".to_string()],
        }
    }

    fn two_cell_mesh() -> SolverRuntimeMeshData {
        SolverRuntimeMeshData {
            points: 0,
            cells: 2,
            faces: 0,
            internal_faces: 0,
            boundary_faces: 0,
            owner: Vec::new(),
            neighbour: Vec::new(),
            patches: vec![SolverRuntimePatchRange {
                name: "wall".to_string(),
                patch_type: "wall".to_string(),
                start_face: 0,
                faces: 0,
            }],
            face_centres: Vec::new(),
            face_area_vectors: Vec::new(),
            cell_centres: vec![point(0.0, 0.0, 0.0), point(1.0, 0.0, 0.0)],
            cell_volumes: vec![1.0, 1.0],
            min_face_area: 0.0,
            max_face_area: 0.0,
            min_cell_volume: 1.0,
            max_cell_volume: 1.0,
            total_cell_volume: 2.0,
            non_positive_cell_volumes: 0,
        }
    }

    fn point(x: f64, y: f64, z: f64) -> Point3 {
        Point3 { x, y, z }
    }

    fn assert_close(left: f64, right: f64, tolerance: f64) {
        assert!(
            (left - right).abs() <= tolerance,
            "expected {left} to be close to {right}"
        );
    }
}
