use std::collections::BTreeMap;

use crate::fields::{FieldFile, FieldValueSummary};
use crate::linear::CsrMatrix;
use crate::runtime::SolverRuntimeMeshData;
use crate::{MeshError, Point3, Result};

#[derive(Clone, Copy, Debug)]
pub struct DiffusionAssemblyCapabilities {
    pub cpu_scalar_diffusion: bool,
    pub cpu_poisson: bool,
    pub fixed_value_boundary: bool,
    pub zero_gradient_boundary: bool,
    pub gpu_assembly: bool,
}

#[derive(Clone, Debug)]
pub struct ScalarDiffusionOptions {
    pub diffusivity: f64,
    pub source: f64,
    pub default_boundary: ScalarBoundaryCondition,
    pub patch_boundary_conditions: Vec<ScalarPatchBoundaryCondition>,
}

#[derive(Clone, Debug)]
pub struct ScalarPatchBoundaryCondition {
    pub patch: String,
    pub condition: ScalarBoundaryCondition,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ScalarBoundaryCondition {
    FixedValue(f64),
    ZeroGradient,
}

#[derive(Clone, Debug)]
pub struct ScalarDiffusionSystem {
    pub matrix: CsrMatrix,
    pub rhs: Vec<f64>,
    pub stats: ScalarDiffusionAssemblyStats,
}

#[derive(Clone, Debug, Default)]
pub struct ScalarDiffusionAssemblyStats {
    pub cells: usize,
    pub internal_faces: usize,
    pub fixed_value_faces: usize,
    pub zero_gradient_faces: usize,
    pub constraint_faces: usize,
    pub min_coefficient: f64,
    pub max_coefficient: f64,
}

impl Default for ScalarDiffusionOptions {
    fn default() -> Self {
        Self {
            diffusivity: 1.0,
            source: 0.0,
            default_boundary: ScalarBoundaryCondition::ZeroGradient,
            patch_boundary_conditions: Vec::new(),
        }
    }
}

pub fn diffusion_assembly_capabilities() -> DiffusionAssemblyCapabilities {
    DiffusionAssemblyCapabilities {
        cpu_scalar_diffusion: true,
        cpu_poisson: true,
        fixed_value_boundary: true,
        zero_gradient_boundary: true,
        gpu_assembly: false,
    }
}

pub fn scalar_diffusion_options_from_field(
    field: &FieldFile,
    diffusivity: f64,
    source: f64,
) -> Result<ScalarDiffusionOptions> {
    if field.class_name.as_deref() != Some("volScalarField") {
        return Err(invalid_input(format!(
            "scalar diffusion requires a volScalarField, field '{}' has class '{}'",
            field_label(field),
            field.class_name.as_deref().unwrap_or("unknown")
        )));
    }

    let mut patch_boundary_conditions = Vec::new();
    for patch in &field.boundary_patches {
        let Some(patch_type) = patch.patch_type.as_deref() else {
            return Err(invalid_input(format!(
                "field '{}' patch '{}' has no boundary type",
                field_label(field),
                patch.name
            )));
        };

        match patch_type {
            "fixedValue" => {
                let value = patch.value.as_ref().ok_or_else(|| {
                    invalid_input(format!(
                        "field '{}' patch '{}' uses fixedValue without a value",
                        field_label(field),
                        patch.name
                    ))
                })?;
                patch_boundary_conditions.push(ScalarPatchBoundaryCondition {
                    patch: patch.name.clone(),
                    condition: ScalarBoundaryCondition::FixedValue(parse_uniform_scalar_value(
                        value,
                        field,
                        &patch.name,
                    )?),
                });
            }
            "zeroGradient" => {
                patch_boundary_conditions.push(ScalarPatchBoundaryCondition {
                    patch: patch.name.clone(),
                    condition: ScalarBoundaryCondition::ZeroGradient,
                });
            }
            "empty" | "wedge" | "symmetryPlane" => {}
            other => {
                return Err(invalid_input(format!(
                    "field '{}' patch '{}' uses unsupported scalar diffusion boundary type '{}'",
                    field_label(field),
                    patch.name,
                    other
                )));
            }
        }
    }

    Ok(ScalarDiffusionOptions {
        diffusivity,
        source,
        default_boundary: ScalarBoundaryCondition::ZeroGradient,
        patch_boundary_conditions,
    })
}

pub fn assemble_scalar_diffusion_system(
    mesh: &SolverRuntimeMeshData,
    options: &ScalarDiffusionOptions,
) -> Result<ScalarDiffusionSystem> {
    validate_diffusion_input(mesh, options)?;

    let mut rows = vec![BTreeMap::<usize, f64>::new(); mesh.cells];
    let mut rhs = mesh
        .cell_volumes
        .iter()
        .map(|volume| options.source * volume)
        .collect::<Vec<_>>();
    let boundary_conditions = boundary_conditions_by_face(mesh, options)?;
    let mut stats = ScalarDiffusionAssemblyStats {
        cells: mesh.cells,
        min_coefficient: f64::INFINITY,
        ..ScalarDiffusionAssemblyStats::default()
    };

    for (face_index, boundary_condition) in boundary_conditions.iter().enumerate() {
        let owner = mesh.owner[face_index];
        if let Some(neighbour) = mesh.neighbour[face_index] {
            let coefficient = face_diffusion_coefficient(
                options.diffusivity,
                mesh.face_area_vectors[face_index],
                mesh.cell_centres[owner],
                mesh.cell_centres[neighbour],
                face_index,
            )?;
            add_entry(&mut rows[owner], owner, coefficient);
            add_entry(&mut rows[owner], neighbour, -coefficient);
            add_entry(&mut rows[neighbour], neighbour, coefficient);
            add_entry(&mut rows[neighbour], owner, -coefficient);
            stats.internal_faces += 1;
            stats.record_coefficient(coefficient);
            continue;
        }

        match *boundary_condition {
            FaceBoundaryTreatment::FixedValue(value) => {
                let coefficient = face_diffusion_coefficient(
                    options.diffusivity,
                    mesh.face_area_vectors[face_index],
                    mesh.cell_centres[owner],
                    mesh.face_centres[face_index],
                    face_index,
                )?;
                add_entry(&mut rows[owner], owner, coefficient);
                rhs[owner] += coefficient * value;
                stats.fixed_value_faces += 1;
                stats.record_coefficient(coefficient);
            }
            FaceBoundaryTreatment::ZeroGradient => {
                stats.zero_gradient_faces += 1;
            }
            FaceBoundaryTreatment::Constraint => {
                stats.constraint_faces += 1;
            }
        }
    }

    stats.finish();
    let matrix_rows = rows
        .into_iter()
        .map(|row| row.into_iter().collect::<Vec<_>>())
        .collect::<Vec<_>>();
    let matrix = CsrMatrix::from_rows(matrix_rows, mesh.cells)?;

    Ok(ScalarDiffusionSystem { matrix, rhs, stats })
}

impl ScalarDiffusionAssemblyStats {
    fn record_coefficient(&mut self, coefficient: f64) {
        self.min_coefficient = self.min_coefficient.min(coefficient);
        self.max_coefficient = self.max_coefficient.max(coefficient);
    }

    fn finish(&mut self) {
        if self.min_coefficient == f64::INFINITY {
            self.min_coefficient = 0.0;
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum FaceBoundaryTreatment {
    FixedValue(f64),
    ZeroGradient,
    Constraint,
}

fn validate_diffusion_input(
    mesh: &SolverRuntimeMeshData,
    options: &ScalarDiffusionOptions,
) -> Result<()> {
    if !options.diffusivity.is_finite() || options.diffusivity <= 0.0 {
        return Err(invalid_input(format!(
            "scalar diffusion diffusivity must be positive and finite, got {}",
            options.diffusivity
        )));
    }
    if !options.source.is_finite() {
        return Err(invalid_input(format!(
            "scalar diffusion source must be finite, got {}",
            options.source
        )));
    }
    match options.default_boundary {
        ScalarBoundaryCondition::FixedValue(value) if !value.is_finite() => {
            return Err(invalid_input(format!(
                "default fixedValue boundary must be finite, got {value}"
            )));
        }
        _ => {}
    }
    for condition in &options.patch_boundary_conditions {
        if condition.patch.trim().is_empty() {
            return Err(invalid_input(
                "patch boundary condition name must not be empty".to_string(),
            ));
        }
        if let ScalarBoundaryCondition::FixedValue(value) = condition.condition
            && !value.is_finite()
        {
            return Err(invalid_input(format!(
                "fixedValue boundary for patch '{}' must be finite, got {value}",
                condition.patch
            )));
        }
    }

    if mesh.owner.len() != mesh.faces {
        return Err(invalid_input(format!(
            "runtime mesh owner length {} does not match faces {}",
            mesh.owner.len(),
            mesh.faces
        )));
    }
    if mesh.neighbour.len() != mesh.faces {
        return Err(invalid_input(format!(
            "runtime mesh neighbour length {} does not match faces {}",
            mesh.neighbour.len(),
            mesh.faces
        )));
    }
    if mesh.face_centres.len() != mesh.faces {
        return Err(invalid_input(format!(
            "runtime mesh face_centres length {} does not match faces {}",
            mesh.face_centres.len(),
            mesh.faces
        )));
    }
    if mesh.face_area_vectors.len() != mesh.faces {
        return Err(invalid_input(format!(
            "runtime mesh face_area_vectors length {} does not match faces {}",
            mesh.face_area_vectors.len(),
            mesh.faces
        )));
    }
    if mesh.cell_centres.len() != mesh.cells {
        return Err(invalid_input(format!(
            "runtime mesh cell_centres length {} does not match cells {}",
            mesh.cell_centres.len(),
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

    for (face_index, &owner) in mesh.owner.iter().enumerate() {
        if owner >= mesh.cells {
            return Err(invalid_input(format!(
                "runtime mesh face {face_index} owner {owner} is out of range for {} cells",
                mesh.cells
            )));
        }
        if let Some(neighbour) = mesh.neighbour[face_index]
            && neighbour >= mesh.cells
        {
            return Err(invalid_input(format!(
                "runtime mesh face {face_index} neighbour {neighbour} is out of range for {} cells",
                mesh.cells
            )));
        }
    }
    for patch in &mesh.patches {
        let end_face = patch.start_face.checked_add(patch.faces).ok_or_else(|| {
            invalid_input(format!(
                "patch '{}' face range overflows: startFace={} faces={}",
                patch.name, patch.start_face, patch.faces
            ))
        })?;
        if end_face > mesh.faces {
            return Err(invalid_input(format!(
                "patch '{}' face range {}..{} exceeds mesh faces {}",
                patch.name, patch.start_face, end_face, mesh.faces
            )));
        }
    }
    for (cell, volume) in mesh.cell_volumes.iter().enumerate() {
        if !volume.is_finite() || *volume < 0.0 {
            return Err(invalid_input(format!(
                "runtime mesh cell {cell} volume must be finite and non-negative, got {volume}"
            )));
        }
    }

    Ok(())
}

fn boundary_conditions_by_face(
    mesh: &SolverRuntimeMeshData,
    options: &ScalarDiffusionOptions,
) -> Result<Vec<FaceBoundaryTreatment>> {
    let mut conditions = vec![FaceBoundaryTreatment::ZeroGradient; mesh.faces];
    for patch in &mesh.patches {
        let treatment = if is_constraint_patch(&patch.patch_type) {
            if let Some(ScalarBoundaryCondition::FixedValue(_)) =
                patch_condition(&patch.name, options)
            {
                return Err(invalid_input(format!(
                    "constraint patch '{}' of type '{}' cannot use fixedValue in scalar diffusion assembly",
                    patch.name, patch.patch_type
                )));
            }
            FaceBoundaryTreatment::Constraint
        } else {
            match patch_condition(&patch.name, options).unwrap_or(options.default_boundary) {
                ScalarBoundaryCondition::FixedValue(value) => {
                    FaceBoundaryTreatment::FixedValue(value)
                }
                ScalarBoundaryCondition::ZeroGradient => FaceBoundaryTreatment::ZeroGradient,
            }
        };

        for (face_index, condition) in conditions
            .iter_mut()
            .enumerate()
            .skip(patch.start_face)
            .take(patch.faces)
        {
            if mesh.neighbour[face_index].is_none() {
                *condition = treatment;
            }
        }
    }
    Ok(conditions)
}

fn patch_condition(
    patch: &str,
    options: &ScalarDiffusionOptions,
) -> Option<ScalarBoundaryCondition> {
    options
        .patch_boundary_conditions
        .iter()
        .rev()
        .find(|condition| condition.patch == patch)
        .map(|condition| condition.condition)
}

fn is_constraint_patch(patch_type: &str) -> bool {
    matches!(patch_type, "empty" | "wedge" | "symmetryPlane")
}

fn face_diffusion_coefficient(
    diffusivity: f64,
    area_vector: Point3,
    from: Point3,
    to: Point3,
    face_index: usize,
) -> Result<f64> {
    let area = magnitude(area_vector);
    let distance = distance(from, to);
    if !area.is_finite() || area <= f64::EPSILON {
        return Err(invalid_input(format!(
            "face {face_index} has non-positive area magnitude {area}"
        )));
    }
    if !distance.is_finite() || distance <= f64::EPSILON {
        return Err(invalid_input(format!(
            "face {face_index} has non-positive diffusion distance {distance}"
        )));
    }
    Ok(diffusivity * area / distance)
}

fn add_entry(row: &mut BTreeMap<usize, f64>, col: usize, value: f64) {
    *row.entry(col).or_insert(0.0) += value;
}

fn magnitude(vector: Point3) -> f64 {
    (vector.x * vector.x + vector.y * vector.y + vector.z * vector.z).sqrt()
}

fn distance(left: Point3, right: Point3) -> f64 {
    let dx = left.x - right.x;
    let dy = left.y - right.y;
    let dz = left.z - right.z;
    (dx * dx + dy * dy + dz * dz).sqrt()
}

fn parse_uniform_scalar_value(
    value: &FieldValueSummary,
    field: &FieldFile,
    patch: &str,
) -> Result<f64> {
    let FieldValueSummary::Uniform(value) = value else {
        return Err(invalid_input(format!(
            "field '{}' patch '{}' fixedValue must be uniform scalar for scalar diffusion",
            field_label(field),
            patch
        )));
    };

    let values = value
        .replace(['(', ')'], " ")
        .split_whitespace()
        .map(|token| {
            token.parse::<f64>().map_err(|_| {
                invalid_input(format!(
                    "field '{}' patch '{}' fixedValue contains non-numeric token '{}'",
                    field_label(field),
                    patch,
                    token
                ))
            })
        })
        .collect::<Result<Vec<_>>>()?;
    if values.len() != 1 {
        return Err(invalid_input(format!(
            "field '{}' patch '{}' fixedValue must contain exactly one scalar, got {} values",
            field_label(field),
            patch,
            values.len()
        )));
    }
    Ok(values[0])
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
    use crate::Point3;
    use crate::fields::{FieldBoundaryPatch, FieldFile, FieldValueSummary};
    use crate::linear::{ConjugateGradientOptions, conjugate_gradient_solve};
    use crate::runtime::{SolverRuntimeMeshData, SolverRuntimePatchRange};

    use super::{
        ScalarBoundaryCondition, ScalarDiffusionOptions, ScalarPatchBoundaryCondition,
        assemble_scalar_diffusion_system, scalar_diffusion_options_from_field,
    };

    #[test]
    fn assembles_and_solves_two_cell_dirichlet_diffusion() {
        let mesh = two_cell_line_mesh();
        let system = assemble_scalar_diffusion_system(
            &mesh,
            &ScalarDiffusionOptions {
                diffusivity: 1.0,
                source: 0.0,
                default_boundary: ScalarBoundaryCondition::ZeroGradient,
                patch_boundary_conditions: vec![
                    ScalarPatchBoundaryCondition {
                        patch: "left".to_string(),
                        condition: ScalarBoundaryCondition::FixedValue(1.0),
                    },
                    ScalarPatchBoundaryCondition {
                        patch: "right".to_string(),
                        condition: ScalarBoundaryCondition::FixedValue(0.0),
                    },
                ],
            },
        )
        .expect("scalar diffusion system");

        assert_eq!(system.stats.cells, 2);
        assert_eq!(system.stats.internal_faces, 1);
        assert_eq!(system.stats.fixed_value_faces, 2);
        assert_eq!(system.stats.zero_gradient_faces, 0);
        assert_close(system.stats.min_coefficient, 1.0, 1.0e-14);
        assert_close(system.stats.max_coefficient, 2.0, 1.0e-14);
        assert_eq!(system.matrix.rows(), 2);
        assert_eq!(system.matrix.cols(), 2);
        assert_eq!(system.matrix.nnz(), 4);
        assert_close_vec(
            &system.matrix.matvec(&[1.0, 1.0]).expect("matvec"),
            &[2.0, 2.0],
            1.0e-14,
        );
        assert_close_vec(&system.rhs, &[2.0, 0.0], 1.0e-14);

        let report = conjugate_gradient_solve(
            &system.matrix,
            &system.rhs,
            None,
            ConjugateGradientOptions {
                max_iterations: 8,
                tolerance: 1.0e-12,
            },
        )
        .expect("cg solve");

        assert!(report.converged);
        assert_close_vec(&report.solution, &[0.75, 0.25], 1.0e-12);
    }

    #[test]
    fn applies_uniform_volume_source_to_rhs() {
        let mesh = two_cell_line_mesh();
        let system = assemble_scalar_diffusion_system(
            &mesh,
            &ScalarDiffusionOptions {
                diffusivity: 2.0,
                source: 3.0,
                default_boundary: ScalarBoundaryCondition::ZeroGradient,
                patch_boundary_conditions: Vec::new(),
            },
        )
        .expect("scalar diffusion system");

        assert_eq!(system.stats.internal_faces, 1);
        assert_eq!(system.stats.fixed_value_faces, 0);
        assert_eq!(system.stats.zero_gradient_faces, 2);
        assert_close_vec(&system.rhs, &[3.0, 3.0], 1.0e-14);
    }

    #[test]
    fn treats_empty_wedge_and_symmetry_as_constraints() {
        for patch_type in ["empty", "wedge", "symmetryPlane"] {
            let mut mesh = two_cell_line_mesh();
            mesh.patches[0].patch_type = patch_type.to_string();

            let system = assemble_scalar_diffusion_system(
                &mesh,
                &ScalarDiffusionOptions {
                    diffusivity: 1.0,
                    source: 0.0,
                    default_boundary: ScalarBoundaryCondition::FixedValue(1.0),
                    patch_boundary_conditions: Vec::new(),
                },
            )
            .expect("scalar diffusion system");

            assert_eq!(system.stats.constraint_faces, 1);
            assert_eq!(system.stats.fixed_value_faces, 1);
        }
    }

    #[test]
    fn rejects_fixed_value_on_constraint_patch() {
        let mut mesh = two_cell_line_mesh();
        mesh.patches[0].patch_type = "wedge".to_string();

        let error = assemble_scalar_diffusion_system(
            &mesh,
            &ScalarDiffusionOptions {
                diffusivity: 1.0,
                source: 0.0,
                default_boundary: ScalarBoundaryCondition::ZeroGradient,
                patch_boundary_conditions: vec![ScalarPatchBoundaryCondition {
                    patch: "left".to_string(),
                    condition: ScalarBoundaryCondition::FixedValue(1.0),
                }],
            },
        )
        .expect_err("fixedValue on wedge should fail");

        assert!(error.to_string().contains("constraint patch"));
    }

    #[test]
    fn builds_diffusion_options_from_scalar_field_boundaries() {
        let field = scalar_field(vec![
            FieldBoundaryPatch {
                name: "inlet".to_string(),
                patch_type: Some("fixedValue".to_string()),
                value: Some(FieldValueSummary::Uniform("293.15".to_string())),
            },
            FieldBoundaryPatch {
                name: "outlet".to_string(),
                patch_type: Some("zeroGradient".to_string()),
                value: None,
            },
            FieldBoundaryPatch {
                name: "front".to_string(),
                patch_type: Some("empty".to_string()),
                value: None,
            },
        ]);

        let options =
            scalar_diffusion_options_from_field(&field, 0.5, 2.0).expect("diffusion options");

        assert_eq!(options.diffusivity, 0.5);
        assert_eq!(options.source, 2.0);
        assert_eq!(options.patch_boundary_conditions.len(), 2);
        assert_eq!(
            options.patch_boundary_conditions[0].condition,
            ScalarBoundaryCondition::FixedValue(293.15)
        );
        assert_eq!(
            options.patch_boundary_conditions[1].condition,
            ScalarBoundaryCondition::ZeroGradient
        );
    }

    #[test]
    fn rejects_non_scalar_fixed_value_field_boundary() {
        let field = scalar_field(vec![FieldBoundaryPatch {
            name: "wall".to_string(),
            patch_type: Some("fixedValue".to_string()),
            value: Some(FieldValueSummary::Uniform("( 1 0 0 )".to_string())),
        }]);

        let error = scalar_diffusion_options_from_field(&field, 1.0, 0.0)
            .expect_err("vector fixedValue should fail");

        assert!(error.to_string().contains("exactly one scalar"));
    }

    fn two_cell_line_mesh() -> SolverRuntimeMeshData {
        SolverRuntimeMeshData {
            points: 0,
            cells: 2,
            faces: 3,
            internal_faces: 1,
            boundary_faces: 2,
            owner: vec![0, 0, 1],
            neighbour: vec![None, Some(1), None],
            patches: vec![
                SolverRuntimePatchRange {
                    name: "left".to_string(),
                    patch_type: "patch".to_string(),
                    start_face: 0,
                    faces: 1,
                },
                SolverRuntimePatchRange {
                    name: "right".to_string(),
                    patch_type: "patch".to_string(),
                    start_face: 2,
                    faces: 1,
                },
            ],
            face_centres: vec![
                point(0.0, 0.5, 0.5),
                point(1.0, 0.5, 0.5),
                point(2.0, 0.5, 0.5),
            ],
            face_area_vectors: vec![
                point(-1.0, 0.0, 0.0),
                point(1.0, 0.0, 0.0),
                point(1.0, 0.0, 0.0),
            ],
            cell_centres: vec![point(0.5, 0.5, 0.5), point(1.5, 0.5, 0.5)],
            cell_volumes: vec![1.0, 1.0],
            min_face_area: 1.0,
            max_face_area: 1.0,
            min_cell_volume: 1.0,
            max_cell_volume: 1.0,
            total_cell_volume: 2.0,
            non_positive_cell_volumes: 0,
        }
    }

    fn point(x: f64, y: f64, z: f64) -> Point3 {
        Point3 { x, y, z }
    }

    fn scalar_field(boundary_patches: Vec<FieldBoundaryPatch>) -> FieldFile {
        FieldFile {
            path: "0/T".into(),
            region: None,
            name: "T".to_string(),
            class_name: Some("volScalarField".to_string()),
            dimensions: None,
            internal_field: None,
            boundary_patches,
        }
    }

    fn assert_close(left: f64, right: f64, tolerance: f64) {
        assert!(
            (left - right).abs() <= tolerance,
            "expected {left} to be close to {right}"
        );
    }

    fn assert_close_vec(left: &[f64], right: &[f64], tolerance: f64) {
        assert_eq!(left.len(), right.len());
        for (index, (&left, &right)) in left.iter().zip(right).enumerate() {
            assert!(
                (left - right).abs() <= tolerance,
                "entry {index}: expected {right}, got {left}"
            );
        }
    }
}
