use std::mem::size_of;
use std::path::Path;

use crate::geometry::compute_poly_mesh_geometry;
use crate::poly_mesh::PolyMesh;
use crate::solver_state::{SolverStateFieldKind, SolverStatePlan, materialize_cpu_buffer};
use crate::{Point3, Result};

#[derive(Clone, Debug)]
pub struct SolverRuntimeData {
    pub mesh: SolverRuntimeMeshData,
    pub fields: Vec<SolverRuntimeFieldBuffer>,
    pub warnings: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct SolverRuntimeMeshData {
    pub points: usize,
    pub cells: usize,
    pub faces: usize,
    pub internal_faces: usize,
    pub boundary_faces: usize,
    pub owner: Vec<usize>,
    pub neighbour: Vec<Option<usize>>,
    pub patches: Vec<SolverRuntimePatchRange>,
    pub face_centres: Vec<Point3>,
    pub face_area_vectors: Vec<Point3>,
    pub cell_centres: Vec<Point3>,
    pub cell_volumes: Vec<f64>,
    pub min_face_area: f64,
    pub max_face_area: f64,
    pub min_cell_volume: f64,
    pub max_cell_volume: f64,
    pub total_cell_volume: f64,
    pub non_positive_cell_volumes: usize,
}

#[derive(Clone, Debug)]
pub struct SolverRuntimePatchRange {
    pub name: String,
    pub patch_type: String,
    pub start_face: usize,
    pub faces: usize,
}

#[derive(Clone, Debug)]
pub struct SolverRuntimeFieldBuffer {
    pub region: Option<String>,
    pub name: String,
    pub kind: SolverStateFieldKind,
    pub components: usize,
    pub scalar_slots: usize,
    pub bytes_f64: usize,
    pub values: Vec<f64>,
}

pub fn build_solver_runtime_data(
    case_dir: &Path,
    mesh: &PolyMesh,
    state: &SolverStatePlan,
) -> Result<SolverRuntimeData> {
    let runtime_mesh = build_solver_runtime_mesh(mesh)?;
    let mut warnings = Vec::new();
    let mut fields = Vec::new();

    for field in &state.fields {
        let label = runtime_field_label(field.region.as_deref(), &field.name);
        if !field.cpu_buffer.materializable {
            warnings.push(format!(
                "field '{label}' is not materializable as a CPU f64 buffer: {}",
                field.cpu_buffer.status
            ));
            continue;
        }

        let Some(values) = materialize_cpu_buffer(field) else {
            warnings.push(format!(
                "field '{label}' was marked materializable but no CPU buffer could be built"
            ));
            continue;
        };
        let Some(components) = field.storage.components else {
            warnings.push(format!(
                "field '{label}' has no component count for runtime buffer"
            ));
            continue;
        };
        let scalar_slots = values.len();
        let bytes_f64 = scalar_slots.saturating_mul(size_of::<f64>());
        if field
            .cpu_buffer
            .scalar_slots
            .is_some_and(|expected| expected != scalar_slots)
        {
            warnings.push(format!(
                "field '{label}' runtime buffer has {scalar_slots} scalars, expected {}",
                field.cpu_buffer.scalar_slots.unwrap_or_default()
            ));
            continue;
        }

        fields.push(SolverRuntimeFieldBuffer {
            region: field.region.clone(),
            name: field.name.clone(),
            kind: field.kind,
            components,
            scalar_slots,
            bytes_f64,
            values,
        });
    }

    if fields.is_empty() && !state.fields.is_empty() {
        warnings.push(format!(
            "no runtime field buffers were built for {}",
            case_dir.display()
        ));
    }

    Ok(SolverRuntimeData {
        mesh: runtime_mesh,
        fields,
        warnings,
    })
}

fn build_solver_runtime_mesh(mesh: &PolyMesh) -> Result<SolverRuntimeMeshData> {
    let geometry = compute_poly_mesh_geometry(mesh)?;
    let mut neighbour = vec![None; mesh.faces.len()];
    for (face_index, cell) in mesh.neighbour.iter().copied().enumerate() {
        neighbour[face_index] = Some(cell);
    }

    let mut min_face_area = f64::INFINITY;
    let mut max_face_area = 0.0_f64;
    for area_vector in &geometry.face_area_vectors {
        let area = vector_magnitude(*area_vector);
        min_face_area = min_face_area.min(area);
        max_face_area = max_face_area.max(area);
    }
    if geometry.face_area_vectors.is_empty() {
        min_face_area = 0.0;
    }

    let mut min_cell_volume = f64::INFINITY;
    let mut max_cell_volume = 0.0_f64;
    let mut total_cell_volume = 0.0_f64;
    for volume in &geometry.cell_volumes {
        min_cell_volume = min_cell_volume.min(*volume);
        max_cell_volume = max_cell_volume.max(*volume);
        total_cell_volume += *volume;
    }
    if geometry.cell_volumes.is_empty() {
        min_cell_volume = 0.0;
    }

    Ok(SolverRuntimeMeshData {
        points: mesh.points.len(),
        cells: mesh.cell_count(),
        faces: mesh.faces.len(),
        internal_faces: mesh.neighbour.len(),
        boundary_faces: mesh.faces.len().saturating_sub(mesh.neighbour.len()),
        owner: mesh.owner.clone(),
        neighbour,
        patches: mesh
            .patches
            .iter()
            .map(|patch| SolverRuntimePatchRange {
                name: patch.name.clone(),
                patch_type: patch.patch_type.clone(),
                start_face: patch.start_face,
                faces: patch.faces,
            })
            .collect(),
        face_centres: geometry.face_centres,
        face_area_vectors: geometry.face_area_vectors,
        cell_centres: geometry.cell_centres,
        cell_volumes: geometry.cell_volumes,
        min_face_area,
        max_face_area,
        min_cell_volume,
        max_cell_volume,
        total_cell_volume,
        non_positive_cell_volumes: geometry.non_positive_cell_volumes,
    })
}

fn runtime_field_label(region: Option<&str>, name: &str) -> String {
    if let Some(region) = region {
        format!("{region}/{name}")
    } else {
        name.to_string()
    }
}

fn vector_magnitude(vector: Point3) -> f64 {
    (vector.x * vector.x + vector.y * vector.y + vector.z * vector.z).sqrt()
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use crate::Point3;
    use crate::poly_mesh::{BoundaryPatch, PolyMesh};
    use crate::solver_state::{
        SolverStateCpuBufferPlan, SolverStateCpuBufferStatus, SolverStateFieldKind,
        SolverStateFieldPlan, SolverStateInternalFieldPlan, SolverStatePlan,
        SolverStateStoragePlan, SolverStateStorageStatus, SolverStateValueKind,
    };

    use super::build_solver_runtime_data;

    #[test]
    fn builds_runtime_mesh_geometry_and_uniform_field_buffer() {
        let mesh = unit_cube_mesh();
        let state = SolverStatePlan {
            fields: vec![SolverStateFieldPlan {
                region: None,
                name: "p".to_string(),
                class_name: Some("volScalarField".to_string()),
                kind: SolverStateFieldKind::VolScalar,
                dimensions: None,
                mesh_cells: Some(1),
                mesh_faces: Some(6),
                internal_field: SolverStateInternalFieldPlan {
                    kind: SolverStateValueKind::Uniform,
                    value_count: Some(1),
                    expected_count: Some(1),
                    valid_count: Some(true),
                    uniform_components: Some(vec![7.0]),
                    nonuniform_values: None,
                },
                boundary_patches: 1,
                mesh_boundary_patches: Some(1),
                storage: SolverStateStoragePlan {
                    cpu_capable: true,
                    gpu_capable: true,
                    components: Some(1),
                    scalar_slots: Some(1),
                    bytes_f64: Some(8),
                    status: SolverStateStorageStatus::Loaded,
                },
                cpu_buffer: SolverStateCpuBufferPlan {
                    materializable: true,
                    scalar_slots: Some(1),
                    bytes_f64: Some(8),
                    status: SolverStateCpuBufferStatus::UniformReady,
                },
            }],
            warnings: Vec::new(),
        };

        let runtime = build_solver_runtime_data(Path::new("case"), &mesh, &state)
            .expect("runtime data should build");

        assert_eq!(runtime.mesh.cells, 1);
        assert_eq!(runtime.mesh.faces, 6);
        assert_eq!(runtime.mesh.owner, vec![0; 6]);
        assert_eq!(runtime.mesh.neighbour, vec![None; 6]);
        assert_eq!(runtime.mesh.cell_centres.len(), 1);
        assert_eq!(runtime.mesh.face_centres.len(), 6);
        assert_eq!(runtime.mesh.face_area_vectors.len(), 6);
        assert_close(runtime.mesh.cell_volumes[0], 1.0);
        assert_eq!(runtime.fields.len(), 1);
        assert_eq!(runtime.fields[0].name, "p");
        assert_eq!(runtime.fields[0].values, vec![7.0]);
        assert!(runtime.warnings.is_empty());
    }

    fn unit_cube_mesh() -> PolyMesh {
        PolyMesh {
            path: PathBuf::from("polyMesh"),
            points: vec![
                Point3 {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                },
                Point3 {
                    x: 1.0,
                    y: 0.0,
                    z: 0.0,
                },
                Point3 {
                    x: 1.0,
                    y: 1.0,
                    z: 0.0,
                },
                Point3 {
                    x: 0.0,
                    y: 1.0,
                    z: 0.0,
                },
                Point3 {
                    x: 0.0,
                    y: 0.0,
                    z: 1.0,
                },
                Point3 {
                    x: 1.0,
                    y: 0.0,
                    z: 1.0,
                },
                Point3 {
                    x: 1.0,
                    y: 1.0,
                    z: 1.0,
                },
                Point3 {
                    x: 0.0,
                    y: 1.0,
                    z: 1.0,
                },
            ],
            faces: vec![
                vec![0, 3, 2, 1],
                vec![4, 5, 6, 7],
                vec![0, 1, 5, 4],
                vec![1, 2, 6, 5],
                vec![2, 3, 7, 6],
                vec![3, 0, 4, 7],
            ],
            owner: vec![0; 6],
            neighbour: Vec::new(),
            patches: vec![BoundaryPatch {
                name: "walls".to_string(),
                patch_type: "wall".to_string(),
                faces: 6,
                start_face: 0,
            }],
        }
    }

    fn assert_close(left: f64, right: f64) {
        assert!(
            (left - right).abs() < 1e-12,
            "expected {left} to be close to {right}"
        );
    }
}
