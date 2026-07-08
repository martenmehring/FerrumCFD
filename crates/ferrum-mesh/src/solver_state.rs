use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::fields::{FieldFile, FieldValueSummary, InitialFieldSet};
use crate::poly_mesh::PolyMesh;

#[derive(Clone, Debug)]
pub struct SolverStatePlan {
    pub fields: Vec<SolverStateFieldPlan>,
    pub warnings: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct SolverStateFieldPlan {
    pub region: Option<String>,
    pub name: String,
    pub class_name: Option<String>,
    pub kind: SolverStateFieldKind,
    pub dimensions: Option<Vec<String>>,
    pub mesh_cells: Option<usize>,
    pub mesh_faces: Option<usize>,
    pub internal_field: SolverStateInternalFieldPlan,
    pub boundary_patches: usize,
    pub mesh_boundary_patches: Option<usize>,
    pub storage: SolverStateStoragePlan,
}

#[derive(Clone, Debug)]
pub struct SolverStateInternalFieldPlan {
    pub kind: SolverStateValueKind,
    pub value_count: Option<usize>,
    pub expected_count: Option<usize>,
    pub valid_count: Option<bool>,
}

#[derive(Clone, Debug)]
pub struct SolverStateStoragePlan {
    pub cpu_capable: bool,
    pub gpu_capable: bool,
    pub status: SolverStateStorageStatus,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SolverStateFieldKind {
    VolScalar,
    VolVector,
    SurfaceScalar,
    Other,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SolverStateValueKind {
    Uniform,
    NonUniform,
    Other,
    Missing,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SolverStateStorageStatus {
    Loaded,
    UnsupportedClass,
}

impl std::fmt::Display for SolverStateFieldKind {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::VolScalar => formatter.write_str("volScalarField"),
            Self::VolVector => formatter.write_str("volVectorField"),
            Self::SurfaceScalar => formatter.write_str("surfaceScalarField"),
            Self::Other => formatter.write_str("other"),
        }
    }
}

impl std::fmt::Display for SolverStateValueKind {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Uniform => formatter.write_str("uniform"),
            Self::NonUniform => formatter.write_str("nonuniform"),
            Self::Other => formatter.write_str("other"),
            Self::Missing => formatter.write_str("missing"),
        }
    }
}

impl std::fmt::Display for SolverStateStorageStatus {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Loaded => formatter.write_str("loaded"),
            Self::UnsupportedClass => formatter.write_str("unsupported-class"),
        }
    }
}

pub fn build_solver_state_plan(case_dir: &Path, fields: &InitialFieldSet) -> SolverStatePlan {
    let mut mesh_cache = HashMap::<Option<String>, MeshCacheEntry>::new();
    let mut state_fields = Vec::new();
    let mut warnings = Vec::new();

    for field in &fields.fields {
        let mesh = mesh_for_field(case_dir, field, &mut mesh_cache, &mut warnings);
        state_fields.push(build_state_field(field, mesh, &mut warnings));
    }

    SolverStatePlan {
        fields: state_fields,
        warnings,
    }
}

fn build_state_field(
    field: &FieldFile,
    mesh: Option<&PolyMesh>,
    warnings: &mut Vec<String>,
) -> SolverStateFieldPlan {
    let label = field_label(field);
    let kind = SolverStateFieldKind::from_class(field.class_name.as_deref());
    let mesh_cells = mesh.map(PolyMesh::cell_count);
    let mesh_faces = mesh.map(|mesh| mesh.faces.len());
    let expected_count = expected_internal_count(kind, mesh);

    validate_dimensions(field, &label, warnings);
    let internal_field = build_internal_field_plan(field, kind, expected_count, &label, warnings);
    let storage = build_storage_plan(kind, &label, warnings);

    SolverStateFieldPlan {
        region: field.region.clone(),
        name: field.name.clone(),
        class_name: field.class_name.clone(),
        kind,
        dimensions: field.dimensions.clone(),
        mesh_cells,
        mesh_faces,
        internal_field,
        boundary_patches: field.boundary_patches.len(),
        mesh_boundary_patches: mesh.map(|mesh| mesh.patches.len()),
        storage,
    }
}

fn mesh_for_field<'a>(
    case_dir: &Path,
    field: &FieldFile,
    mesh_cache: &'a mut HashMap<Option<String>, MeshCacheEntry>,
    warnings: &mut Vec<String>,
) -> Option<&'a PolyMesh> {
    let region = field.region.clone();
    if !mesh_cache.contains_key(&region) {
        let path = poly_mesh_path(case_dir, region.as_deref());
        let entry = match PolyMesh::read(&path) {
            Ok(mesh) => MeshCacheEntry::Mesh(mesh),
            Err(error) => MeshCacheEntry::Error(format!("{error}")),
        };
        mesh_cache.insert(region.clone(), entry);
    }

    match mesh_cache
        .get(&region)
        .expect("mesh cache entry exists after insertion")
    {
        MeshCacheEntry::Mesh(mesh) => Some(mesh),
        MeshCacheEntry::Error(error) => {
            let label = region
                .as_deref()
                .map(|region| format!("region '{region}'"))
                .unwrap_or_else(|| "base mesh".to_string());
            warnings.push(format!(
                "could not build solver-state field storage for {label}: {error}"
            ));
            None
        }
    }
}

fn poly_mesh_path(case_dir: &Path, region: Option<&str>) -> PathBuf {
    if let Some(region) = region {
        case_dir.join("constant").join(region).join("polyMesh")
    } else {
        case_dir.join("constant").join("polyMesh")
    }
}

fn validate_dimensions(field: &FieldFile, label: &str, warnings: &mut Vec<String>) {
    match &field.dimensions {
        Some(dimensions) if dimensions.len() == 7 => {}
        Some(dimensions) => warnings.push(format!(
            "field '{label}' dimensions should contain 7 entries, found {}",
            dimensions.len()
        )),
        None => warnings.push(format!("field '{label}' has no dimensions entry")),
    }
}

fn build_internal_field_plan(
    field: &FieldFile,
    kind: SolverStateFieldKind,
    expected_count: Option<usize>,
    label: &str,
    warnings: &mut Vec<String>,
) -> SolverStateInternalFieldPlan {
    match &field.internal_field {
        Some(FieldValueSummary::Uniform(_)) => SolverStateInternalFieldPlan {
            kind: SolverStateValueKind::Uniform,
            value_count: expected_count,
            expected_count,
            valid_count: expected_count.map(|_| true),
        },
        Some(FieldValueSummary::NonUniform { count, .. }) => {
            let valid_count = count.zip(expected_count).map(|(count, expected)| {
                if count != expected {
                    warnings.push(format!(
                        "field '{label}' internalField count {count} does not match expected {} for {}",
                        expected, kind
                    ));
                    false
                } else {
                    true
                }
            });
            if count.is_none() {
                warnings.push(format!(
                    "field '{label}' has nonuniform internalField without a readable value count"
                ));
            }
            SolverStateInternalFieldPlan {
                kind: SolverStateValueKind::NonUniform,
                value_count: *count,
                expected_count,
                valid_count,
            }
        }
        Some(FieldValueSummary::Other(_)) => SolverStateInternalFieldPlan {
            kind: SolverStateValueKind::Other,
            value_count: None,
            expected_count,
            valid_count: None,
        },
        None => {
            warnings.push(format!("field '{label}' has no internalField entry"));
            SolverStateInternalFieldPlan {
                kind: SolverStateValueKind::Missing,
                value_count: None,
                expected_count,
                valid_count: Some(false),
            }
        }
    }
}

fn expected_internal_count(kind: SolverStateFieldKind, mesh: Option<&PolyMesh>) -> Option<usize> {
    let mesh = mesh?;
    match kind {
        SolverStateFieldKind::VolScalar | SolverStateFieldKind::VolVector => {
            Some(mesh.cell_count())
        }
        SolverStateFieldKind::SurfaceScalar => Some(mesh.faces.len()),
        SolverStateFieldKind::Other => None,
    }
}

fn build_storage_plan(
    kind: SolverStateFieldKind,
    label: &str,
    warnings: &mut Vec<String>,
) -> SolverStateStoragePlan {
    match kind {
        SolverStateFieldKind::VolScalar
        | SolverStateFieldKind::VolVector
        | SolverStateFieldKind::SurfaceScalar => SolverStateStoragePlan {
            cpu_capable: true,
            gpu_capable: true,
            status: SolverStateStorageStatus::Loaded,
        },
        SolverStateFieldKind::Other => {
            warnings.push(format!(
                "field '{label}' has unsupported class for solver-state storage"
            ));
            SolverStateStoragePlan {
                cpu_capable: false,
                gpu_capable: false,
                status: SolverStateStorageStatus::UnsupportedClass,
            }
        }
    }
}

fn field_label(field: &FieldFile) -> String {
    if let Some(region) = &field.region {
        format!("{region}/{}", field.name)
    } else {
        field.name.clone()
    }
}

impl SolverStateFieldKind {
    fn from_class(class_name: Option<&str>) -> Self {
        match class_name {
            Some("volScalarField") => Self::VolScalar,
            Some("volVectorField") => Self::VolVector,
            Some("surfaceScalarField") => Self::SurfaceScalar,
            _ => Self::Other,
        }
    }
}

enum MeshCacheEntry {
    Mesh(PolyMesh),
    Error(String),
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use crate::Point3;
    use crate::fields::{FieldBoundaryPatch, FieldFile, FieldValueSummary};
    use crate::poly_mesh::{BoundaryPatch, PolyMesh};

    use super::{
        SolverStateFieldKind, SolverStateStorageStatus, SolverStateValueKind, build_state_field,
    };

    #[test]
    fn accepts_uniform_vol_scalar_as_cell_sized_state() {
        let field = field(
            "p",
            "volScalarField",
            Some(FieldValueSummary::Uniform("0".to_string())),
        );
        let mesh = mesh(4);
        let mut warnings = Vec::new();

        let state = build_state_field(&field, Some(&mesh), &mut warnings);

        assert_eq!(state.kind, SolverStateFieldKind::VolScalar);
        assert_eq!(state.mesh_cells, Some(4));
        assert_eq!(state.internal_field.kind, SolverStateValueKind::Uniform);
        assert_eq!(state.internal_field.value_count, Some(4));
        assert_eq!(state.internal_field.valid_count, Some(true));
        assert!(state.storage.cpu_capable);
        assert!(state.storage.gpu_capable);
        assert_eq!(state.storage.status, SolverStateStorageStatus::Loaded);
        assert!(warnings.is_empty());
    }

    #[test]
    fn warns_for_wrong_nonuniform_volume_count() {
        let field = field(
            "U",
            "volVectorField",
            Some(FieldValueSummary::NonUniform {
                value_type: Some("List<vector>".to_string()),
                count: Some(3),
            }),
        );
        let mesh = mesh(4);
        let mut warnings = Vec::new();

        let state = build_state_field(&field, Some(&mesh), &mut warnings);

        assert_eq!(state.kind, SolverStateFieldKind::VolVector);
        assert_eq!(state.internal_field.value_count, Some(3));
        assert_eq!(state.internal_field.expected_count, Some(4));
        assert_eq!(state.internal_field.valid_count, Some(false));
        assert!(
            warnings
                .iter()
                .any(|warning| warning.contains("does not match expected 4"))
        );
    }

    #[test]
    fn marks_unknown_field_classes_as_unsupported() {
        let field = field(
            "alpha",
            "pointScalarField",
            Some(FieldValueSummary::Uniform("0".to_string())),
        );
        let mesh = mesh(4);
        let mut warnings = Vec::new();

        let state = build_state_field(&field, Some(&mesh), &mut warnings);

        assert_eq!(state.kind, SolverStateFieldKind::Other);
        assert!(!state.storage.cpu_capable);
        assert!(!state.storage.gpu_capable);
        assert_eq!(
            state.storage.status,
            SolverStateStorageStatus::UnsupportedClass
        );
        assert!(
            warnings
                .iter()
                .any(|warning| warning.contains("unsupported class"))
        );
    }

    fn field(name: &str, class_name: &str, internal_field: Option<FieldValueSummary>) -> FieldFile {
        FieldFile {
            path: PathBuf::from(name),
            region: None,
            name: name.to_string(),
            class_name: Some(class_name.to_string()),
            dimensions: Some(vec![
                "0".to_string(),
                "0".to_string(),
                "0".to_string(),
                "0".to_string(),
                "0".to_string(),
                "0".to_string(),
                "0".to_string(),
            ]),
            internal_field,
            boundary_patches: vec![FieldBoundaryPatch {
                name: "wall".to_string(),
                patch_type: Some("zeroGradient".to_string()),
                value: None,
            }],
        }
    }

    fn mesh(cells: usize) -> PolyMesh {
        PolyMesh {
            path: PathBuf::from("polyMesh"),
            points: vec![Point3 {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            }],
            faces: vec![vec![0]; cells + 1],
            owner: (0..cells).collect(),
            neighbour: Vec::new(),
            patches: vec![BoundaryPatch {
                name: "wall".to_string(),
                patch_type: "patch".to_string(),
                faces: 1,
                start_face: cells,
            }],
        }
    }
}
