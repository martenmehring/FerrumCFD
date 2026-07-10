use std::collections::HashMap;
use std::mem::size_of;
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
    pub cpu_buffer: SolverStateCpuBufferPlan,
}

#[derive(Clone, Debug)]
pub struct SolverStateInternalFieldPlan {
    pub kind: SolverStateValueKind,
    pub value_count: Option<usize>,
    pub expected_count: Option<usize>,
    pub valid_count: Option<bool>,
    pub uniform_components: Option<Vec<f64>>,
    pub nonuniform_values: Option<Vec<f64>>,
}

#[derive(Clone, Debug)]
pub struct SolverStateStoragePlan {
    pub cpu_capable: bool,
    pub gpu_capable: bool,
    pub components: Option<usize>,
    pub scalar_slots: Option<usize>,
    pub bytes_f64: Option<usize>,
    pub status: SolverStateStorageStatus,
}

#[derive(Clone, Debug)]
pub struct SolverStateCpuBufferPlan {
    pub materializable: bool,
    pub scalar_slots: Option<usize>,
    pub bytes_f64: Option<usize>,
    pub status: SolverStateCpuBufferStatus,
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SolverStateCpuBufferStatus {
    UniformReady,
    NonUniformReady,
    NonUniformDataNotLoaded,
    UnsupportedClass,
    InvalidShape,
    MissingInternalField,
    UnsupportedInternalField,
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

impl std::fmt::Display for SolverStateCpuBufferStatus {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UniformReady => formatter.write_str("uniform-ready"),
            Self::NonUniformReady => formatter.write_str("nonuniform-ready"),
            Self::NonUniformDataNotLoaded => formatter.write_str("nonuniform-data-not-loaded"),
            Self::UnsupportedClass => formatter.write_str("unsupported-class"),
            Self::InvalidShape => formatter.write_str("invalid-shape"),
            Self::MissingInternalField => formatter.write_str("missing-internal-field"),
            Self::UnsupportedInternalField => formatter.write_str("unsupported-internal-field"),
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
    let storage = build_storage_plan(kind, expected_count, &label, warnings);
    let cpu_buffer = build_cpu_buffer_plan(&internal_field, &storage);

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
        cpu_buffer,
    }
}

pub fn materialize_cpu_buffer(field: &SolverStateFieldPlan) -> Option<Vec<f64>> {
    if !field.cpu_buffer.materializable {
        return None;
    }

    match field.cpu_buffer.status {
        SolverStateCpuBufferStatus::UniformReady => materialize_uniform_cpu_buffer(field),
        SolverStateCpuBufferStatus::NonUniformReady => {
            field.internal_field.nonuniform_values.clone()
        }
        _ => None,
    }
}

pub fn materialize_uniform_cpu_buffer(field: &SolverStateFieldPlan) -> Option<Vec<f64>> {
    if field.cpu_buffer.status != SolverStateCpuBufferStatus::UniformReady {
        return None;
    }

    let components = field.internal_field.uniform_components.as_deref()?;
    let value_count = field.internal_field.expected_count?;
    let scalar_slots = field.cpu_buffer.scalar_slots?;
    let mut buffer = Vec::new();
    buffer.try_reserve_exact(scalar_slots).ok()?;
    for _ in 0..value_count {
        buffer.extend_from_slice(components);
    }
    Some(buffer)
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
        Some(FieldValueSummary::Uniform(value)) => {
            let uniform_components = parse_uniform_components(value);
            match (&uniform_components, components_per_value(kind)) {
                (Some(values), Some(expected_components))
                    if values.len() != expected_components =>
                {
                    warnings.push(format!(
                        "field '{label}' uniform internalField has {} components, expected {expected_components} for {kind}",
                        values.len()
                    ));
                }
                (None, Some(_)) => warnings.push(format!(
                    "field '{label}' uniform internalField value could not be parsed as numeric components"
                )),
                _ => {}
            }

            SolverStateInternalFieldPlan {
                kind: SolverStateValueKind::Uniform,
                value_count: expected_count,
                expected_count,
                valid_count: expected_count.map(|_| true),
                uniform_components,
                nonuniform_values: None,
            }
        }
        Some(FieldValueSummary::NonUniform {
            value_type,
            count,
            values,
        }) => {
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
            validate_nonuniform_values(
                kind,
                value_type.as_deref(),
                *count,
                values.as_deref(),
                label,
                warnings,
            );
            SolverStateInternalFieldPlan {
                kind: SolverStateValueKind::NonUniform,
                value_count: *count,
                expected_count,
                valid_count,
                uniform_components: None,
                nonuniform_values: values.clone(),
            }
        }
        Some(FieldValueSummary::Other(_)) => SolverStateInternalFieldPlan {
            kind: SolverStateValueKind::Other,
            value_count: None,
            expected_count,
            valid_count: None,
            uniform_components: None,
            nonuniform_values: None,
        },
        None => {
            warnings.push(format!("field '{label}' has no internalField entry"));
            SolverStateInternalFieldPlan {
                kind: SolverStateValueKind::Missing,
                value_count: None,
                expected_count,
                valid_count: Some(false),
                uniform_components: None,
                nonuniform_values: None,
            }
        }
    }
}

fn validate_nonuniform_values(
    kind: SolverStateFieldKind,
    value_type: Option<&str>,
    count: Option<usize>,
    values: Option<&[f64]>,
    label: &str,
    warnings: &mut Vec<String>,
) {
    let Some(values) = values else {
        if count.is_some() && nonuniform_value_type_is_supported(value_type) {
            warnings.push(format!(
                "field '{label}' nonuniform internalField numeric values could not be loaded"
            ));
        }
        return;
    };

    let Some(components) = components_per_value(kind) else {
        return;
    };
    let Some(count) = count else {
        return;
    };
    let Some(expected_values) = count.checked_mul(components) else {
        warnings.push(format!(
            "field '{label}' nonuniform internalField value storage size overflowed"
        ));
        return;
    };
    if values.len() != expected_values {
        warnings.push(format!(
            "field '{label}' nonuniform internalField loaded {} scalar values, expected {expected_values}",
            values.len()
        ));
    }
}

fn nonuniform_value_type_is_supported(value_type: Option<&str>) -> bool {
    matches!(
        value_type,
        Some("List<scalar>")
            | Some("scalarField")
            | Some("Field<scalar>")
            | Some("List<vector>")
            | Some("vectorField")
            | Some("Field<vector>")
    )
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
    expected_count: Option<usize>,
    label: &str,
    warnings: &mut Vec<String>,
) -> SolverStateStoragePlan {
    match kind {
        SolverStateFieldKind::VolScalar
        | SolverStateFieldKind::VolVector
        | SolverStateFieldKind::SurfaceScalar => {
            let components = components_per_value(kind);
            let scalar_slots = components
                .zip(expected_count)
                .and_then(|(components, count)| count.checked_mul(components));
            let bytes_f64 = scalar_slots.and_then(|slots| slots.checked_mul(size_of::<f64>()));
            if expected_count.is_some() && scalar_slots.is_none() {
                warnings.push(format!(
                    "field '{label}' storage size overflowed while estimating scalar slots"
                ));
            }

            SolverStateStoragePlan {
                cpu_capable: true,
                gpu_capable: true,
                components,
                scalar_slots,
                bytes_f64,
                status: SolverStateStorageStatus::Loaded,
            }
        }
        SolverStateFieldKind::Other => {
            warnings.push(format!(
                "field '{label}' has unsupported class for solver-state storage"
            ));
            SolverStateStoragePlan {
                cpu_capable: false,
                gpu_capable: false,
                components: None,
                scalar_slots: None,
                bytes_f64: None,
                status: SolverStateStorageStatus::UnsupportedClass,
            }
        }
    }
}

fn build_cpu_buffer_plan(
    internal_field: &SolverStateInternalFieldPlan,
    storage: &SolverStateStoragePlan,
) -> SolverStateCpuBufferPlan {
    let status = if storage.status != SolverStateStorageStatus::Loaded {
        SolverStateCpuBufferStatus::UnsupportedClass
    } else {
        match internal_field.kind {
            SolverStateValueKind::Uniform => {
                let valid_shape = matches!(internal_field.valid_count, Some(true))
                    && internal_field.expected_count.is_some()
                    && storage.scalar_slots.is_some()
                    && storage.bytes_f64.is_some()
                    && storage
                        .components
                        .zip(internal_field.uniform_components.as_ref())
                        .is_some_and(|(expected_components, components)| {
                            components.len() == expected_components
                        });
                if valid_shape {
                    SolverStateCpuBufferStatus::UniformReady
                } else {
                    SolverStateCpuBufferStatus::InvalidShape
                }
            }
            SolverStateValueKind::NonUniform => {
                let valid_shape = matches!(internal_field.valid_count, Some(true))
                    && storage.scalar_slots.is_some()
                    && storage.bytes_f64.is_some()
                    && storage
                        .scalar_slots
                        .zip(internal_field.nonuniform_values.as_ref())
                        .is_some_and(|(expected_scalars, values)| values.len() == expected_scalars);
                if valid_shape {
                    SolverStateCpuBufferStatus::NonUniformReady
                } else if internal_field.valid_count == Some(false)
                    || internal_field.nonuniform_values.is_some()
                {
                    SolverStateCpuBufferStatus::InvalidShape
                } else {
                    SolverStateCpuBufferStatus::NonUniformDataNotLoaded
                }
            }
            SolverStateValueKind::Other => SolverStateCpuBufferStatus::UnsupportedInternalField,
            SolverStateValueKind::Missing => SolverStateCpuBufferStatus::MissingInternalField,
        }
    };

    SolverStateCpuBufferPlan {
        materializable: matches!(
            status,
            SolverStateCpuBufferStatus::UniformReady | SolverStateCpuBufferStatus::NonUniformReady
        ),
        scalar_slots: storage.scalar_slots,
        bytes_f64: storage.bytes_f64,
        status,
    }
}

fn components_per_value(kind: SolverStateFieldKind) -> Option<usize> {
    match kind {
        SolverStateFieldKind::VolScalar | SolverStateFieldKind::SurfaceScalar => Some(1),
        SolverStateFieldKind::VolVector => Some(3),
        SolverStateFieldKind::Other => None,
    }
}

fn parse_uniform_components(value: &str) -> Option<Vec<f64>> {
    let cleaned = value.replace(['(', ')'], " ");
    let mut values = Vec::new();
    for token in cleaned.split_whitespace() {
        values.push(token.parse::<f64>().ok()?);
    }
    if values.is_empty() {
        None
    } else {
        Some(values)
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
        SolverStateCpuBufferStatus, SolverStateFieldKind, SolverStateStorageStatus,
        SolverStateValueKind, build_state_field, materialize_cpu_buffer,
        materialize_uniform_cpu_buffer,
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
        assert_eq!(state.internal_field.uniform_components, Some(vec![0.0]));
        assert_eq!(state.internal_field.nonuniform_values, None);
        assert!(state.storage.cpu_capable);
        assert!(state.storage.gpu_capable);
        assert_eq!(state.storage.components, Some(1));
        assert_eq!(state.storage.scalar_slots, Some(4));
        assert_eq!(state.storage.bytes_f64, Some(32));
        assert_eq!(state.storage.status, SolverStateStorageStatus::Loaded);
        assert!(state.cpu_buffer.materializable);
        assert_eq!(state.cpu_buffer.scalar_slots, Some(4));
        assert_eq!(state.cpu_buffer.bytes_f64, Some(32));
        assert_eq!(
            state.cpu_buffer.status,
            SolverStateCpuBufferStatus::UniformReady
        );
        assert!(warnings.is_empty());
    }

    #[test]
    fn estimates_vector_storage_slots_from_mesh_cells() {
        let field = field(
            "U",
            "volVectorField",
            Some(FieldValueSummary::Uniform("( 1 2 3 )".to_string())),
        );
        let mesh = mesh(4);
        let mut warnings = Vec::new();

        let state = build_state_field(&field, Some(&mesh), &mut warnings);

        assert_eq!(state.kind, SolverStateFieldKind::VolVector);
        assert_eq!(
            state.internal_field.uniform_components,
            Some(vec![1.0, 2.0, 3.0])
        );
        assert_eq!(state.storage.components, Some(3));
        assert_eq!(state.storage.scalar_slots, Some(12));
        assert_eq!(state.storage.bytes_f64, Some(96));
        assert_eq!(
            state.cpu_buffer.status,
            SolverStateCpuBufferStatus::UniformReady
        );
        assert!(warnings.is_empty());
    }

    #[test]
    fn materializes_uniform_scalar_cpu_buffer() {
        let field = field(
            "p",
            "volScalarField",
            Some(FieldValueSummary::Uniform("7".to_string())),
        );
        let mesh = mesh(4);
        let mut warnings = Vec::new();

        let state = build_state_field(&field, Some(&mesh), &mut warnings);
        let buffer = materialize_cpu_buffer(&state).expect("uniform scalar materializes");

        assert_eq!(buffer, vec![7.0, 7.0, 7.0, 7.0]);
        assert_eq!(
            materialize_uniform_cpu_buffer(&state),
            Some(vec![7.0, 7.0, 7.0, 7.0])
        );
    }

    #[test]
    fn materializes_uniform_vector_cpu_buffer() {
        let field = field(
            "U",
            "volVectorField",
            Some(FieldValueSummary::Uniform("( 1 2 3 )".to_string())),
        );
        let mesh = mesh(3);
        let mut warnings = Vec::new();

        let state = build_state_field(&field, Some(&mesh), &mut warnings);
        let buffer = materialize_cpu_buffer(&state).expect("uniform vector materializes");

        assert_eq!(buffer, vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn warns_for_wrong_uniform_component_count() {
        let field = field(
            "U",
            "volVectorField",
            Some(FieldValueSummary::Uniform("( 1 2 )".to_string())),
        );
        let mesh = mesh(4);
        let mut warnings = Vec::new();

        let state = build_state_field(&field, Some(&mesh), &mut warnings);

        assert_eq!(state.storage.components, Some(3));
        assert_eq!(
            state.internal_field.uniform_components,
            Some(vec![1.0, 2.0])
        );
        assert!(!state.cpu_buffer.materializable);
        assert_eq!(
            state.cpu_buffer.status,
            SolverStateCpuBufferStatus::InvalidShape
        );
        assert!(materialize_uniform_cpu_buffer(&state).is_none());
        assert!(
            warnings
                .iter()
                .any(|warning| warning.contains("expected 3"))
        );
    }

    #[test]
    fn warns_for_wrong_nonuniform_volume_count() {
        let field = field(
            "U",
            "volVectorField",
            Some(FieldValueSummary::NonUniform {
                value_type: Some("List<vector>".to_string()),
                count: Some(3),
                values: Some(vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0]),
            }),
        );
        let mesh = mesh(4);
        let mut warnings = Vec::new();

        let state = build_state_field(&field, Some(&mesh), &mut warnings);

        assert_eq!(state.kind, SolverStateFieldKind::VolVector);
        assert_eq!(state.internal_field.value_count, Some(3));
        assert_eq!(state.internal_field.expected_count, Some(4));
        assert_eq!(state.internal_field.valid_count, Some(false));
        assert!(!state.cpu_buffer.materializable);
        assert_eq!(
            state.cpu_buffer.status,
            SolverStateCpuBufferStatus::InvalidShape
        );
        assert!(materialize_cpu_buffer(&state).is_none());
        assert!(
            warnings
                .iter()
                .any(|warning| warning.contains("does not match expected 4"))
        );
    }

    #[test]
    fn materializes_nonuniform_scalar_cpu_buffer() {
        let field = field(
            "T",
            "volScalarField",
            Some(FieldValueSummary::NonUniform {
                value_type: Some("List<scalar>".to_string()),
                count: Some(4),
                values: Some(vec![300.0, 310.0, 320.0, 330.0]),
            }),
        );
        let mesh = mesh(4);
        let mut warnings = Vec::new();

        let state = build_state_field(&field, Some(&mesh), &mut warnings);

        assert_eq!(state.internal_field.kind, SolverStateValueKind::NonUniform);
        assert_eq!(
            state.internal_field.nonuniform_values,
            Some(vec![300.0, 310.0, 320.0, 330.0])
        );
        assert!(state.cpu_buffer.materializable);
        assert_eq!(
            state.cpu_buffer.status,
            SolverStateCpuBufferStatus::NonUniformReady
        );
        assert_eq!(
            materialize_cpu_buffer(&state),
            Some(vec![300.0, 310.0, 320.0, 330.0])
        );
        assert!(materialize_uniform_cpu_buffer(&state).is_none());
        assert!(warnings.is_empty());
    }

    #[test]
    fn materializes_nonuniform_vector_cpu_buffer() {
        let field = field(
            "U",
            "volVectorField",
            Some(FieldValueSummary::NonUniform {
                value_type: Some("List<vector>".to_string()),
                count: Some(3),
                values: Some(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]),
            }),
        );
        let mesh = mesh(3);
        let mut warnings = Vec::new();

        let state = build_state_field(&field, Some(&mesh), &mut warnings);

        assert_eq!(
            state.cpu_buffer.status,
            SolverStateCpuBufferStatus::NonUniformReady
        );
        assert_eq!(
            materialize_cpu_buffer(&state),
            Some(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
        );
        assert!(warnings.is_empty());
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
        assert!(!state.cpu_buffer.materializable);
        assert_eq!(
            state.cpu_buffer.status,
            SolverStateCpuBufferStatus::UnsupportedClass
        );
        assert!(
            warnings
                .iter()
                .any(|warning| warning.contains("unsupported class"))
        );
    }

    #[test]
    fn keeps_valid_nonuniform_cpu_buffer_unmaterialized_until_values_are_loaded() {
        let field = field(
            "phi",
            "surfaceScalarField",
            Some(FieldValueSummary::NonUniform {
                value_type: Some("List<scalar>".to_string()),
                count: Some(5),
                values: None,
            }),
        );
        let mesh = mesh(4);
        let mut warnings = Vec::new();

        let state = build_state_field(&field, Some(&mesh), &mut warnings);

        assert_eq!(state.internal_field.valid_count, Some(true));
        assert!(!state.cpu_buffer.materializable);
        assert_eq!(
            state.cpu_buffer.status,
            SolverStateCpuBufferStatus::NonUniformDataNotLoaded
        );
        assert_eq!(state.cpu_buffer.scalar_slots, Some(5));
        assert_eq!(state.cpu_buffer.bytes_f64, Some(40));
        assert!(materialize_uniform_cpu_buffer(&state).is_none());
        assert!(materialize_cpu_buffer(&state).is_none());
        assert!(
            warnings
                .iter()
                .any(|warning| warning.contains("could not be loaded"))
        );
    }

    #[test]
    fn rejects_nonuniform_cpu_buffer_with_wrong_loaded_shape() {
        let field = field(
            "U",
            "volVectorField",
            Some(FieldValueSummary::NonUniform {
                value_type: Some("List<vector>".to_string()),
                count: Some(4),
                values: Some(vec![1.0, 0.0, 0.0]),
            }),
        );
        let mesh = mesh(4);
        let mut warnings = Vec::new();

        let state = build_state_field(&field, Some(&mesh), &mut warnings);

        assert!(!state.cpu_buffer.materializable);
        assert_eq!(
            state.cpu_buffer.status,
            SolverStateCpuBufferStatus::InvalidShape
        );
        assert!(materialize_cpu_buffer(&state).is_none());
        assert!(
            warnings
                .iter()
                .any(|warning| warning.contains("loaded 3 scalar values"))
        );
    }

    #[test]
    fn keeps_unknown_nonuniform_types_as_unmaterialized() {
        let field = field(
            "alpha",
            "volScalarField",
            Some(FieldValueSummary::NonUniform {
                value_type: Some("List<label>".to_string()),
                count: Some(4),
                values: None,
            }),
        );
        let mesh = mesh(4);
        let mut warnings = Vec::new();

        let state = build_state_field(&field, Some(&mesh), &mut warnings);

        assert!(!state.cpu_buffer.materializable);
        assert_eq!(
            state.cpu_buffer.status,
            SolverStateCpuBufferStatus::NonUniformDataNotLoaded
        );
        assert!(warnings.is_empty());
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
                inlet_value: None,
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
