use crate::diffusion::{
    ScalarBoundaryCondition, ScalarDiffusionOptions, ScalarPatchBoundaryCondition,
};
use crate::runtime::SolverRuntimeMeshData;
use crate::{MeshError, Point3, Result};

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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PipeAxis {
    X,
    Y,
    Z,
}

#[derive(Clone, Debug)]
pub struct LaminarPipeBenchmarkOptions {
    pub pressure_drop: f64,
    pub dynamic_viscosity: f64,
    pub length: f64,
    pub diameter: f64,
    pub inlet_patch: String,
    pub outlet_patch: String,
    pub axis: PipeAxis,
}

#[derive(Clone, Debug)]
pub struct LaminarPipeBenchmarkSummary {
    pub min_velocity: f64,
    pub max_velocity: f64,
    pub mean_velocity: f64,
    pub flow_rate: f64,
    pub analytic_mean_velocity: f64,
    pub analytic_flow_rate: f64,
    pub pressure_drop_from_mean: f64,
    pub pressure_drop_from_owner_cells: f64,
    pub relative_mean_velocity_error: f64,
    pub relative_pressure_drop_from_mean_error: f64,
    pub relative_pressure_drop_from_owner_cells_error: f64,
}

#[derive(Clone, Debug)]
pub struct LaminarPlaneChannelBenchmarkOptions {
    pub pressure_drop: f64,
    pub dynamic_viscosity: f64,
    pub length: f64,
    pub gap: f64,
    pub depth: f64,
    pub inlet_patch: String,
    pub outlet_patch: String,
    pub axis: PipeAxis,
}

#[derive(Clone, Debug)]
pub struct LaminarPlaneChannelBenchmarkSummary {
    pub min_velocity: f64,
    pub max_velocity: f64,
    pub mean_velocity: f64,
    pub flow_rate: f64,
    pub flow_rate_per_unit_depth: f64,
    pub analytic_mean_velocity: f64,
    pub analytic_flow_rate: f64,
    pub analytic_flow_rate_per_unit_depth: f64,
    pub pressure_drop_from_mean: f64,
    pub pressure_drop_from_owner_cells: f64,
    pub relative_mean_velocity_error: f64,
    pub relative_pressure_drop_from_mean_error: f64,
    pub relative_pressure_drop_from_owner_cells_error: f64,
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
    Ok(hagen_poiseuille_reference(
        options.pressure_drop,
        options.dynamic_viscosity,
        options.length,
        options.diameter,
    ))
}

fn hagen_poiseuille_reference(
    pressure_drop: f64,
    dynamic_viscosity: f64,
    length: f64,
    diameter: f64,
) -> PoiseuilleReference {
    let cross_section_area = std::f64::consts::PI * diameter * diameter / 4.0;
    let mean_velocity = pressure_drop * diameter * diameter / (32.0 * dynamic_viscosity * length);
    PoiseuilleReference {
        source: pressure_drop / length,
        cross_section_area,
        mean_velocity,
        flow_rate: mean_velocity * cross_section_area,
    }
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

pub fn summarize_laminar_pipe_solution(
    mesh: &SolverRuntimeMeshData,
    velocity: &[Point3],
    pressure: &[f64],
    options: &LaminarPipeBenchmarkOptions,
) -> Result<LaminarPipeBenchmarkSummary> {
    validate_laminar_pipe_benchmark_options(options)?;
    if velocity.len() != mesh.cells || pressure.len() != mesh.cells {
        return Err(invalid_input(format!(
            "laminar pipe benchmark expected {} cells, got U={} p={}",
            mesh.cells,
            velocity.len(),
            pressure.len()
        )));
    }
    if mesh.cell_volumes.len() != mesh.cells {
        return Err(invalid_input(format!(
            "runtime mesh cell_volumes length {} does not match cells {}",
            mesh.cell_volumes.len(),
            mesh.cells
        )));
    }

    let mut min_velocity = f64::INFINITY;
    let mut max_velocity = f64::NEG_INFINITY;
    let mut weighted_sum = 0.0;
    let mut total_volume = 0.0;
    for (value, volume) in velocity.iter().zip(&mesh.cell_volumes) {
        let component = pipe_axis_component(*value, options.axis);
        if !component.is_finite() || !volume.is_finite() || *volume < 0.0 {
            return Err(invalid_input(
                "laminar pipe benchmark fields and cell volumes must be finite".to_string(),
            ));
        }
        min_velocity = min_velocity.min(component);
        max_velocity = max_velocity.max(component);
        weighted_sum += component * volume;
        total_volume += volume;
    }
    if !total_volume.is_finite() || total_volume <= 0.0 {
        return Err(invalid_input(format!(
            "runtime mesh total volume must be positive and finite, got {total_volume}"
        )));
    }
    if pressure.iter().any(|value| !value.is_finite()) {
        return Err(invalid_input(
            "laminar pipe benchmark pressure contains a non-finite value".to_string(),
        ));
    }

    let reference = hagen_poiseuille_reference(
        options.pressure_drop,
        options.dynamic_viscosity,
        options.length,
        options.diameter,
    );
    let mean_velocity = weighted_sum / total_volume;
    let flow_rate = mean_velocity * reference.cross_section_area;
    let pressure_drop_from_mean = 32.0 * options.dynamic_viscosity * options.length * mean_velocity
        / (options.diameter * options.diameter);
    let inlet_pressure = patch_owner_cell_average(mesh, pressure, &options.inlet_patch)?;
    let outlet_pressure = patch_owner_cell_average(mesh, pressure, &options.outlet_patch)?;
    let pressure_drop_from_owner_cells = inlet_pressure - outlet_pressure;

    Ok(LaminarPipeBenchmarkSummary {
        min_velocity,
        max_velocity,
        mean_velocity,
        flow_rate,
        analytic_mean_velocity: reference.mean_velocity,
        analytic_flow_rate: reference.flow_rate,
        pressure_drop_from_mean,
        pressure_drop_from_owner_cells,
        relative_mean_velocity_error: relative_error(mean_velocity, reference.mean_velocity),
        relative_pressure_drop_from_mean_error: relative_error(
            pressure_drop_from_mean,
            options.pressure_drop,
        ),
        relative_pressure_drop_from_owner_cells_error: relative_error(
            pressure_drop_from_owner_cells,
            options.pressure_drop,
        ),
    })
}

pub fn summarize_laminar_plane_channel_solution(
    mesh: &SolverRuntimeMeshData,
    velocity: &[Point3],
    pressure: &[f64],
    options: &LaminarPlaneChannelBenchmarkOptions,
) -> Result<LaminarPlaneChannelBenchmarkSummary> {
    validate_laminar_plane_channel_benchmark_options(options)?;
    if velocity.len() != mesh.cells || pressure.len() != mesh.cells {
        return Err(invalid_input(format!(
            "laminar plane-channel benchmark expected {} cells, got U={} p={}",
            mesh.cells,
            velocity.len(),
            pressure.len()
        )));
    }
    if mesh.cell_volumes.len() != mesh.cells {
        return Err(invalid_input(format!(
            "runtime mesh cell_volumes length {} does not match cells {}",
            mesh.cell_volumes.len(),
            mesh.cells
        )));
    }

    let mut min_velocity = f64::INFINITY;
    let mut max_velocity = f64::NEG_INFINITY;
    let mut weighted_sum = 0.0;
    let mut total_volume = 0.0;
    for (value, volume) in velocity.iter().zip(&mesh.cell_volumes) {
        let component = pipe_axis_component(*value, options.axis);
        if !component.is_finite() || !volume.is_finite() || *volume < 0.0 {
            return Err(invalid_input(
                "laminar plane-channel benchmark fields and cell volumes must be finite"
                    .to_string(),
            ));
        }
        min_velocity = min_velocity.min(component);
        max_velocity = max_velocity.max(component);
        weighted_sum += component * volume;
        total_volume += volume;
    }
    if !total_volume.is_finite() || total_volume <= 0.0 {
        return Err(invalid_input(format!(
            "runtime mesh total volume must be positive and finite, got {total_volume}"
        )));
    }
    if pressure.iter().any(|value| !value.is_finite()) {
        return Err(invalid_input(
            "laminar plane-channel benchmark pressure contains a non-finite value".to_string(),
        ));
    }

    let analytic_mean_velocity = options.pressure_drop * options.gap * options.gap
        / (12.0 * options.dynamic_viscosity * options.length);
    let mean_velocity = weighted_sum / total_volume;
    let cross_section_area = options.gap * options.depth;
    let flow_rate = mean_velocity * cross_section_area;
    let flow_rate_per_unit_depth = mean_velocity * options.gap;
    let analytic_flow_rate = analytic_mean_velocity * cross_section_area;
    let analytic_flow_rate_per_unit_depth = analytic_mean_velocity * options.gap;
    let pressure_drop_from_mean = 12.0 * options.dynamic_viscosity * options.length * mean_velocity
        / (options.gap * options.gap);
    let inlet_pressure = patch_owner_cell_average(mesh, pressure, &options.inlet_patch)?;
    let outlet_pressure = patch_owner_cell_average(mesh, pressure, &options.outlet_patch)?;
    let pressure_drop_from_owner_cells = inlet_pressure - outlet_pressure;

    Ok(LaminarPlaneChannelBenchmarkSummary {
        min_velocity,
        max_velocity,
        mean_velocity,
        flow_rate,
        flow_rate_per_unit_depth,
        analytic_mean_velocity,
        analytic_flow_rate,
        analytic_flow_rate_per_unit_depth,
        pressure_drop_from_mean,
        pressure_drop_from_owner_cells,
        relative_mean_velocity_error: relative_error(mean_velocity, analytic_mean_velocity),
        relative_pressure_drop_from_mean_error: relative_error(
            pressure_drop_from_mean,
            options.pressure_drop,
        ),
        relative_pressure_drop_from_owner_cells_error: relative_error(
            pressure_drop_from_owner_cells,
            options.pressure_drop,
        ),
    })
}

fn patch_owner_cell_average(
    mesh: &SolverRuntimeMeshData,
    values: &[f64],
    patch_name: &str,
) -> Result<f64> {
    let patch = mesh
        .patches
        .iter()
        .find(|patch| patch.name == patch_name)
        .ok_or_else(|| invalid_input(format!("mesh patch '{patch_name}' was not found")))?;
    let end_face = patch
        .start_face
        .checked_add(patch.faces)
        .ok_or_else(|| invalid_input(format!("mesh patch '{patch_name}' face range overflows")))?;
    if end_face > mesh.faces || end_face > mesh.owner.len() || end_face > mesh.neighbour.len() {
        return Err(invalid_input(format!(
            "mesh patch '{patch_name}' face range {}..{} exceeds mesh faces {}",
            patch.start_face, end_face, mesh.faces
        )));
    }

    let mut sum = 0.0;
    let mut count = 0usize;
    for face_index in patch.start_face..end_face {
        if mesh.neighbour[face_index].is_some() {
            continue;
        }
        let owner = mesh.owner[face_index];
        let value = values.get(owner).ok_or_else(|| {
            invalid_input(format!(
                "mesh patch '{patch_name}' face {face_index} references missing owner cell {owner}"
            ))
        })?;
        sum += value;
        count += 1;
    }
    if count == 0 {
        return Err(invalid_input(format!(
            "mesh patch '{patch_name}' has no boundary owner cells"
        )));
    }
    Ok(sum / count as f64)
}

fn pipe_axis_component(value: Point3, axis: PipeAxis) -> f64 {
    match axis {
        PipeAxis::X => value.x,
        PipeAxis::Y => value.y,
        PipeAxis::Z => value.z,
    }
}

fn relative_error(value: f64, reference: f64) -> f64 {
    (value - reference) / reference
}

fn validate_poiseuille_options(options: &PoiseuilleOptions) -> Result<()> {
    validate_hagen_poiseuille_inputs(
        options.pressure_drop,
        options.dynamic_viscosity,
        options.length,
        options.diameter,
    )?;
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

fn validate_hagen_poiseuille_inputs(
    pressure_drop: f64,
    dynamic_viscosity: f64,
    length: f64,
    diameter: f64,
) -> Result<()> {
    if !pressure_drop.is_finite() || pressure_drop <= 0.0 {
        return Err(invalid_input(format!(
            "Poiseuille pressure drop must be positive and finite, got {}",
            pressure_drop
        )));
    }
    if !dynamic_viscosity.is_finite() || dynamic_viscosity <= 0.0 {
        return Err(invalid_input(format!(
            "Poiseuille dynamic viscosity must be positive and finite, got {}",
            dynamic_viscosity
        )));
    }
    if !length.is_finite() || length <= 0.0 {
        return Err(invalid_input(format!(
            "Poiseuille length must be positive and finite, got {}",
            length
        )));
    }
    if !diameter.is_finite() || diameter <= 0.0 {
        return Err(invalid_input(format!(
            "Poiseuille diameter must be positive and finite, got {}",
            diameter
        )));
    }
    Ok(())
}

fn validate_laminar_pipe_benchmark_options(options: &LaminarPipeBenchmarkOptions) -> Result<()> {
    validate_hagen_poiseuille_inputs(
        options.pressure_drop,
        options.dynamic_viscosity,
        options.length,
        options.diameter,
    )?;
    if options.inlet_patch.trim().is_empty() || options.outlet_patch.trim().is_empty() {
        return Err(invalid_input(
            "laminar pipe benchmark inlet and outlet patch names must not be empty".to_string(),
        ));
    }
    Ok(())
}

fn validate_laminar_plane_channel_benchmark_options(
    options: &LaminarPlaneChannelBenchmarkOptions,
) -> Result<()> {
    for (name, value) in [
        ("pressure drop", options.pressure_drop),
        ("dynamic viscosity", options.dynamic_viscosity),
        ("length", options.length),
        ("gap", options.gap),
        ("depth", options.depth),
    ] {
        if !value.is_finite() || value <= 0.0 {
            return Err(invalid_input(format!(
                "laminar plane-channel benchmark {name} must be positive and finite, got {value}"
            )));
        }
    }
    if options.inlet_patch.trim().is_empty() || options.outlet_patch.trim().is_empty() {
        return Err(invalid_input(
            "laminar plane-channel benchmark inlet and outlet patch names must not be empty"
                .to_string(),
        ));
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
        LaminarPipeBenchmarkOptions, LaminarPlaneChannelBenchmarkOptions, PipeAxis,
        PoiseuilleOptions, poiseuille_diffusion_options, poiseuille_reference,
        summarize_laminar_pipe_solution, summarize_laminar_plane_channel_solution,
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

    #[test]
    fn summarizes_external_laminar_pipe_fields() {
        let mesh = two_cell_pipe_mesh();
        let options = LaminarPipeBenchmarkOptions {
            pressure_drop: 2.0,
            dynamic_viscosity: 0.001,
            length: 1.0,
            diameter: 0.01,
            inlet_patch: "inlet".to_string(),
            outlet_patch: "outlet".to_string(),
            axis: PipeAxis::X,
        };

        let summary = summarize_laminar_pipe_solution(
            &mesh,
            &[point(0.005, 0.0, 0.0), point(0.0075, 0.0, 0.0)],
            &[2.0, 0.0],
            &options,
        )
        .expect("laminar pipe summary");

        assert_close(summary.mean_velocity, 0.00625, 1.0e-14);
        assert_close(summary.pressure_drop_from_mean, 2.0, 1.0e-14);
        assert_close(summary.pressure_drop_from_owner_cells, 2.0, 1.0e-14);
        assert_close(summary.relative_mean_velocity_error, 0.0, 1.0e-14);
    }

    #[test]
    fn summarizes_external_laminar_plane_channel_fields() {
        let mesh = two_cell_pipe_mesh();
        let options = LaminarPlaneChannelBenchmarkOptions {
            pressure_drop: 0.6012,
            dynamic_viscosity: 0.001002,
            length: 1.0,
            gap: 0.02,
            depth: 0.001,
            inlet_patch: "inlet".to_string(),
            outlet_patch: "outlet".to_string(),
            axis: PipeAxis::X,
        };

        let summary = summarize_laminar_plane_channel_solution(
            &mesh,
            &[point(0.015, 0.0, 0.0), point(0.025, 0.0, 0.0)],
            &[0.6012, 0.0],
            &options,
        )
        .expect("laminar plane-channel summary");

        assert_close(summary.mean_velocity, 0.02, 1.0e-14);
        assert_close(summary.analytic_mean_velocity, 0.02, 1.0e-14);
        assert_close(summary.flow_rate, 4.0e-7, 1.0e-18);
        assert_close(summary.pressure_drop_from_mean, 0.6012, 1.0e-14);
        assert_close(summary.pressure_drop_from_owner_cells, 0.6012, 1.0e-14);
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

    fn two_cell_pipe_mesh() -> SolverRuntimeMeshData {
        let mut mesh = two_cell_mesh();
        mesh.faces = 2;
        mesh.boundary_faces = 2;
        mesh.owner = vec![0, 1];
        mesh.neighbour = vec![None, None];
        mesh.patches = vec![
            SolverRuntimePatchRange {
                name: "inlet".to_string(),
                patch_type: "patch".to_string(),
                start_face: 0,
                faces: 1,
            },
            SolverRuntimePatchRange {
                name: "outlet".to_string(),
                patch_type: "patch".to_string(),
                start_face: 1,
                faces: 1,
            },
        ];
        mesh
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
