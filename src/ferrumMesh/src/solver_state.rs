use std::cmp::Ordering;
use std::mem::size_of;
use std::path::{Path, PathBuf};

use crate::fields::{
    FieldFile, FieldValueSummary, InitialFieldSet, nonuniform_value_type_components,
};
use crate::poly_mesh::PolyMesh;
use crate::{MeshError, Result};

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
    pub loaded_scalars: Option<usize>,
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
    /// True only when `materialize_cpu_buffer` can build the buffer from this
    /// self-contained state plan. Valid nonuniform payloads remain owned by
    /// `InitialFieldSet` until their one-shot transfer into runtime data.
    pub materializable: bool,
    pub scalar_slots: Option<usize>,
    pub bytes_f64: Option<usize>,
    pub status: SolverStateCpuBufferStatus,
}

#[derive(Debug)]
pub(crate) struct SolverStateFieldDescriptors {
    pub kind: SolverStateFieldKind,
    pub internal_field: SolverStateInternalFieldPlan,
    pub storage: SolverStateStoragePlan,
    pub cpu_buffer: SolverStateCpuBufferPlan,
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
    UnsupportedNonUniform,
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
    /// Shape and source payload are valid for one-shot runtime transfer. The
    /// payload is intentionally not cloned into `SolverStateFieldPlan`.
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
            // Keep the existing text/JSON value stable while retaining enough
            // internal provenance to classify an unsupported declaration.
            Self::UnsupportedNonUniform => formatter.write_str("nonuniform"),
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

pub fn build_solver_state_plan(
    case_dir: &Path,
    fields: &InitialFieldSet,
) -> Result<SolverStatePlan> {
    let mut state_fields = Vec::new();
    state_fields
        .try_reserve_exact(fields.fields.len())
        .map_err(|_| MeshError::InvalidInput("solver-state field allocation failed".to_string()))?;
    let mut warnings = Vec::new();
    let mut current_region: Option<Option<&str>> = None;
    let mut current_mesh: Option<Result<PolyMesh>> = None;
    let mut previous_field: Option<&FieldFile> = None;

    for field in &fields.fields {
        let region = field.region.as_deref();
        if previous_field.is_some_and(|previous| {
            previous
                .region
                .as_deref()
                .cmp(&region)
                .then(previous.name.cmp(&field.name))
                .then(previous.path.cmp(&field.path))
                == Ordering::Greater
        }) {
            return Err(MeshError::InvalidInput(
                "initial fields are not in canonical region, name, and path order".to_string(),
            ));
        }
        previous_field = Some(field);

        if current_region != Some(region) {
            // Assignment evaluates its right-hand side before dropping the old
            // value. Clear the window first so two full region meshes are never
            // live while the successor is read.
            drop(current_mesh.take());
            current_mesh = Some(PolyMesh::read(&poly_mesh_path(case_dir, region)));
            current_region = Some(region);
        }

        let mesh = match current_mesh.as_ref() {
            Some(Ok(mesh)) => Some(mesh),
            Some(Err(error)) => {
                let label = region
                    .map(|region| format!("region '{region}'"))
                    .unwrap_or_else(|| "base mesh".to_string());
                push_warning(
                    &mut warnings,
                    format!("could not build solver-state field storage for {label}: {error}"),
                )?;
                None
            }
            None => {
                return Err(MeshError::InvalidInput(
                    "solver-state region mesh window is unavailable".to_string(),
                ));
            }
        };
        state_fields.push(build_state_field(field, mesh, &mut warnings)?);
    }

    Ok(SolverStatePlan {
        fields: state_fields,
        warnings,
    })
}

fn build_state_field(
    field: &FieldFile,
    mesh: Option<&PolyMesh>,
    warnings: &mut Vec<String>,
) -> Result<SolverStateFieldPlan> {
    let label = field_label(field)?;
    let mesh_cells = mesh.map(PolyMesh::cell_count);
    let mesh_faces = mesh.map(|mesh| mesh.faces.len());

    validate_dimensions(field, &label, warnings)?;
    let descriptors = derive_field_descriptors(field, mesh_cells, mesh_faces)?;
    validate_internal_field_descriptor(field, &descriptors, &label, warnings)?;
    validate_storage_descriptor(&descriptors, &label, warnings)?;

    Ok(SolverStateFieldPlan {
        region: try_clone_optional_string(field.region.as_deref())?,
        name: try_clone_string(&field.name, "solver-state field name allocation failed")?,
        class_name: try_clone_optional_string(field.class_name.as_deref())?,
        kind: descriptors.kind,
        dimensions: try_clone_optional_string_vec(field.dimensions.as_deref())?,
        mesh_cells,
        mesh_faces,
        internal_field: descriptors.internal_field,
        boundary_patches: field.boundary_patches.len(),
        mesh_boundary_patches: mesh.map(|mesh| mesh.patches.len()),
        storage: descriptors.storage,
        cpu_buffer: descriptors.cpu_buffer,
    })
}

pub(crate) fn derive_field_descriptors(
    field: &FieldFile,
    mesh_cells: Option<usize>,
    mesh_faces: Option<usize>,
) -> Result<SolverStateFieldDescriptors> {
    let kind = SolverStateFieldKind::from_class(field.class_name.as_deref());
    let expected_count = expected_internal_count(kind, mesh_cells, mesh_faces);
    let internal_field = derive_internal_field_plan(field, expected_count)?;
    let storage = derive_storage_plan(kind, expected_count);
    let cpu_buffer = build_cpu_buffer_plan(&internal_field, &storage);
    Ok(SolverStateFieldDescriptors {
        kind,
        internal_field,
        storage,
        cpu_buffer,
    })
}

pub fn materialize_cpu_buffer(field: &SolverStateFieldPlan) -> Option<Vec<f64>> {
    if !field.cpu_buffer.materializable {
        return None;
    }

    match field.cpu_buffer.status {
        SolverStateCpuBufferStatus::UniformReady => materialize_uniform_cpu_buffer(field),
        SolverStateCpuBufferStatus::NonUniformReady => None,
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

fn poly_mesh_path(case_dir: &Path, region: Option<&str>) -> PathBuf {
    if let Some(region) = region {
        case_dir.join("constant").join(region).join("polyMesh")
    } else {
        case_dir.join("constant").join("polyMesh")
    }
}

fn validate_dimensions(field: &FieldFile, label: &str, warnings: &mut Vec<String>) -> Result<()> {
    match &field.dimensions {
        Some(dimensions) if dimensions.len() == 7 => {}
        Some(dimensions) => push_warning(
            warnings,
            format!(
                "field '{label}' dimensions should contain 7 entries, found {}",
                dimensions.len()
            ),
        )?,
        None => push_warning(warnings, format!("field '{label}' has no dimensions entry"))?,
    }
    Ok(())
}

fn derive_internal_field_plan(
    field: &FieldFile,
    expected_count: Option<usize>,
) -> Result<SolverStateInternalFieldPlan> {
    match &field.internal_field {
        Some(FieldValueSummary::Uniform(value)) => {
            let uniform_components = parse_uniform_components(value)?;
            Ok(SolverStateInternalFieldPlan {
                kind: SolverStateValueKind::Uniform,
                value_count: expected_count,
                expected_count,
                valid_count: expected_count.map(|_| true),
                uniform_components,
                loaded_scalars: None,
            })
        }
        Some(FieldValueSummary::NonUniform {
            value_type,
            count,
            values,
        }) => {
            let supported_value_type =
                nonuniform_value_type_components(value_type.as_deref()).is_some();
            let valid_count = count
                .zip(expected_count)
                .map(|(count, expected)| count == expected);
            Ok(SolverStateInternalFieldPlan {
                kind: if supported_value_type {
                    SolverStateValueKind::NonUniform
                } else {
                    SolverStateValueKind::UnsupportedNonUniform
                },
                value_count: *count,
                expected_count,
                valid_count,
                uniform_components: None,
                loaded_scalars: supported_value_type
                    .then(|| values.as_ref().map(Vec::len))
                    .flatten(),
            })
        }
        Some(FieldValueSummary::Other(_)) => Ok(SolverStateInternalFieldPlan {
            kind: SolverStateValueKind::Other,
            value_count: None,
            expected_count,
            valid_count: None,
            uniform_components: None,
            loaded_scalars: None,
        }),
        None => Ok(SolverStateInternalFieldPlan {
            kind: SolverStateValueKind::Missing,
            value_count: None,
            expected_count,
            valid_count: Some(false),
            uniform_components: None,
            loaded_scalars: None,
        }),
    }
}

fn validate_internal_field_descriptor(
    field: &FieldFile,
    descriptors: &SolverStateFieldDescriptors,
    label: &str,
    warnings: &mut Vec<String>,
) -> Result<()> {
    match &field.internal_field {
        Some(FieldValueSummary::Uniform(_)) => {
            match (
                &descriptors.internal_field.uniform_components,
                components_per_value(descriptors.kind),
            ) {
                (Some(values), Some(expected_components))
                    if values.len() != expected_components =>
                {
                    push_warning(
                        warnings,
                        format!(
                            "field '{label}' uniform internalField has {} components, expected {expected_components} for {}",
                            values.len(),
                            descriptors.kind
                        ),
                    )?;
                }
                (None, Some(_)) => push_warning(
                    warnings,
                    format!(
                        "field '{label}' uniform internalField value could not be parsed as numeric components"
                    ),
                )?,
                _ => {}
            }
        }
        Some(FieldValueSummary::NonUniform {
            value_type,
            count,
            values,
        }) => {
            let expected_count = descriptors.internal_field.expected_count;
            if let (Some(count), Some(expected)) = (count, expected_count)
                && count != &expected
            {
                push_warning(
                    warnings,
                    format!(
                        "field '{label}' internalField count {count} does not match expected {expected} for {}",
                        descriptors.kind
                    ),
                )?;
            }
            if count.is_none() {
                push_warning(
                    warnings,
                    format!(
                        "field '{label}' has nonuniform internalField without a readable value count"
                    ),
                )?;
            }
            validate_nonuniform_values(
                descriptors.kind,
                value_type.as_deref(),
                *count,
                values.as_deref(),
                label,
                warnings,
            )?;
        }
        Some(FieldValueSummary::Other(_)) => {}
        None => push_warning(
            warnings,
            format!("field '{label}' has no internalField entry"),
        )?,
    }
    Ok(())
}

fn validate_nonuniform_values(
    kind: SolverStateFieldKind,
    value_type: Option<&str>,
    count: Option<usize>,
    values: Option<&[f64]>,
    label: &str,
    warnings: &mut Vec<String>,
) -> Result<()> {
    if nonuniform_value_type_components(value_type).is_none() {
        push_warning(
            warnings,
            format!("field '{label}' has an unsupported nonuniform value type"),
        )?;
        return Ok(());
    }

    let Some(values) = values else {
        return Ok(());
    };

    let Some(components) = components_per_value(kind) else {
        return Ok(());
    };
    let Some(count) = count else {
        return Ok(());
    };
    let Some(expected_values) = count.checked_mul(components) else {
        push_warning(
            warnings,
            format!("field '{label}' nonuniform internalField value storage size overflowed"),
        )?;
        return Ok(());
    };
    if values.len() != expected_values {
        push_warning(
            warnings,
            format!(
                "field '{label}' nonuniform internalField loaded {} scalar values, expected {expected_values}",
                values.len()
            ),
        )?;
    }
    Ok(())
}

fn expected_internal_count(
    kind: SolverStateFieldKind,
    mesh_cells: Option<usize>,
    mesh_faces: Option<usize>,
) -> Option<usize> {
    match kind {
        SolverStateFieldKind::VolScalar | SolverStateFieldKind::VolVector => mesh_cells,
        SolverStateFieldKind::SurfaceScalar => mesh_faces,
        SolverStateFieldKind::Other => None,
    }
}

fn derive_storage_plan(
    kind: SolverStateFieldKind,
    expected_count: Option<usize>,
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
            SolverStateStoragePlan {
                cpu_capable: true,
                gpu_capable: true,
                components,
                scalar_slots,
                bytes_f64,
                status: SolverStateStorageStatus::Loaded,
            }
        }
        SolverStateFieldKind::Other => SolverStateStoragePlan {
            cpu_capable: false,
            gpu_capable: false,
            components: None,
            scalar_slots: None,
            bytes_f64: None,
            status: SolverStateStorageStatus::UnsupportedClass,
        },
    }
}

fn validate_storage_descriptor(
    descriptors: &SolverStateFieldDescriptors,
    label: &str,
    warnings: &mut Vec<String>,
) -> Result<()> {
    if descriptors.kind == SolverStateFieldKind::Other {
        push_warning(
            warnings,
            format!("field '{label}' has unsupported class for solver-state storage"),
        )?;
    } else if descriptors.internal_field.expected_count.is_some()
        && descriptors.storage.scalar_slots.is_none()
    {
        push_warning(
            warnings,
            format!("field '{label}' storage size overflowed while estimating scalar slots"),
        )?;
    }
    Ok(())
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
                        .zip(internal_field.loaded_scalars)
                        .is_some_and(|(expected_scalars, loaded_scalars)| {
                            loaded_scalars == expected_scalars
                        });
                if valid_shape {
                    SolverStateCpuBufferStatus::NonUniformReady
                } else if internal_field.valid_count == Some(false)
                    || internal_field.loaded_scalars.is_some()
                {
                    SolverStateCpuBufferStatus::InvalidShape
                } else {
                    SolverStateCpuBufferStatus::NonUniformDataNotLoaded
                }
            }
            SolverStateValueKind::UnsupportedNonUniform => {
                SolverStateCpuBufferStatus::UnsupportedInternalField
            }
            SolverStateValueKind::Other => SolverStateCpuBufferStatus::UnsupportedInternalField,
            SolverStateValueKind::Missing => SolverStateCpuBufferStatus::MissingInternalField,
        }
    };

    SolverStateCpuBufferPlan {
        materializable: status == SolverStateCpuBufferStatus::UniformReady,
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

const MAX_UNIFORM_COMPONENTS: usize = 3;

fn parse_uniform_components(value: &str) -> Result<Option<Vec<f64>>> {
    let mut values = Vec::new();
    for token in value
        .split(|character: char| character.is_whitespace() || matches!(character, '(' | ')'))
        .filter(|token| !token.is_empty())
    {
        if values.len() == MAX_UNIFORM_COMPONENTS {
            return Ok(None);
        }
        let Ok(number) = token.parse::<f64>() else {
            return Ok(None);
        };
        if !number.is_finite() {
            return Ok(None);
        }
        values.try_reserve(1).map_err(|_| {
            MeshError::InvalidInput("uniform component allocation failed".to_string())
        })?;
        values.push(number);
    }
    if values.is_empty() {
        Ok(None)
    } else {
        Ok(Some(values))
    }
}

fn field_label(field: &FieldFile) -> Result<String> {
    if let Some(region) = &field.region {
        let length = region
            .len()
            .checked_add(1)
            .and_then(|length| length.checked_add(field.name.len()))
            .ok_or_else(|| MeshError::InvalidInput("field label size overflowed".to_string()))?;
        let mut label = String::new();
        label
            .try_reserve_exact(length)
            .map_err(|_| MeshError::InvalidInput("field label allocation failed".to_string()))?;
        label.push_str(region);
        label.push('/');
        label.push_str(&field.name);
        Ok(label)
    } else {
        try_clone_string(&field.name, "field label allocation failed")
    }
}

fn push_warning(warnings: &mut Vec<String>, warning: String) -> Result<()> {
    warnings.try_reserve(1).map_err(|_| {
        MeshError::InvalidInput("solver-state warning allocation failed".to_string())
    })?;
    warnings.push(warning);
    Ok(())
}

fn try_clone_string(value: &str, failure: &str) -> Result<String> {
    let mut cloned = String::new();
    cloned
        .try_reserve_exact(value.len())
        .map_err(|_| MeshError::InvalidInput(failure.to_string()))?;
    cloned.push_str(value);
    Ok(cloned)
}

fn try_clone_optional_string(value: Option<&str>) -> Result<Option<String>> {
    value
        .map(|value| try_clone_string(value, "solver-state string allocation failed"))
        .transpose()
}

fn try_clone_optional_string_vec(values: Option<&[String]>) -> Result<Option<Vec<String>>> {
    let Some(values) = values else {
        return Ok(None);
    };
    let mut cloned = Vec::new();
    cloned.try_reserve_exact(values.len()).map_err(|_| {
        MeshError::InvalidInput("solver-state dimensions allocation failed".to_string())
    })?;
    for value in values {
        cloned.push(try_clone_string(
            value,
            "solver-state dimension allocation failed",
        )?);
    }
    Ok(Some(cloned))
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

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use crate::Point3;
    use crate::fields::{FieldBoundaryPatch, FieldFile, FieldValueSummary, InitialFieldSet};
    use crate::poly_mesh::{BoundaryPatch, PolyMesh};

    use super::{
        SolverStateCpuBufferStatus, SolverStateFieldKind, SolverStateStorageStatus,
        SolverStateValueKind, build_state_field as try_build_state_field, materialize_cpu_buffer,
        materialize_uniform_cpu_buffer,
    };

    fn build_state_field(
        field: &FieldFile,
        mesh: Option<&PolyMesh>,
        warnings: &mut Vec<String>,
    ) -> super::SolverStateFieldPlan {
        try_build_state_field(field, mesh, warnings).expect("solver-state field should build")
    }

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
        assert_eq!(state.internal_field.loaded_scalars, None);
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
        let materialized = materialize_cpu_buffer(&state);

        assert_eq!(state.cpu_buffer.materializable, materialized.is_some());
        let buffer = materialized.expect("uniform scalar materializes");
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
        let materialized = materialize_cpu_buffer(&state);

        assert_eq!(state.cpu_buffer.materializable, materialized.is_some());
        let buffer = materialized.expect("uniform vector materializes");
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
        assert_eq!(
            state.cpu_buffer.materializable,
            materialize_cpu_buffer(&state).is_some()
        );
        assert!(materialize_uniform_cpu_buffer(&state).is_none());
        assert!(
            warnings
                .iter()
                .any(|warning| warning.contains("expected 3"))
        );
    }

    #[test]
    fn rejects_uniform_component_lists_larger_than_openfoam_shapes() {
        let field = field(
            "U",
            "volVectorField",
            Some(FieldValueSummary::Uniform("( 1 2 3 4 5 6 )".to_string())),
        );
        let mesh = mesh(4);
        let mut warnings = Vec::new();

        let state = build_state_field(&field, Some(&mesh), &mut warnings);

        assert_eq!(state.internal_field.uniform_components, None);
        assert!(!state.cpu_buffer.materializable);
        assert_eq!(
            state.cpu_buffer.status,
            SolverStateCpuBufferStatus::InvalidShape
        );
        assert!(materialize_cpu_buffer(&state).is_none());
        assert!(materialize_uniform_cpu_buffer(&state).is_none());
        assert!(
            warnings
                .iter()
                .any(|warning| warning.contains("could not be parsed"))
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
    fn records_nonuniform_scalar_payload_without_cloning_it() {
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
        assert_eq!(state.internal_field.loaded_scalars, Some(4));
        assert!(!state.cpu_buffer.materializable);
        assert_eq!(
            state.cpu_buffer.status,
            SolverStateCpuBufferStatus::NonUniformReady
        );
        assert!(materialize_cpu_buffer(&state).is_none());
        assert!(materialize_uniform_cpu_buffer(&state).is_none());
        assert!(warnings.is_empty());
    }

    #[test]
    fn records_nonuniform_vector_payload_without_cloning_it() {
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
        assert_eq!(state.internal_field.loaded_scalars, Some(9));
        assert!(!state.cpu_buffer.materializable);
        assert!(materialize_cpu_buffer(&state).is_none());
        assert!(materialize_uniform_cpu_buffer(&state).is_none());
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
        assert!(warnings.is_empty());
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
            SolverStateCpuBufferStatus::UnsupportedInternalField
        );
        assert!(
            warnings
                .iter()
                .any(|warning| warning.contains("unsupported nonuniform value type"))
        );
    }

    #[test]
    fn solver_state_requires_full_canonical_field_order() {
        let fields = InitialFieldSet {
            case_dir: PathBuf::from("missing-case"),
            fields: vec![
                field(
                    "z",
                    "volScalarField",
                    Some(FieldValueSummary::Uniform("0".to_string())),
                ),
                field(
                    "a",
                    "volScalarField",
                    Some(FieldValueSummary::Uniform("0".to_string())),
                ),
            ],
        };

        let error = super::build_solver_state_plan(Path::new("missing-case"), &fields)
            .expect_err("non-canonical field names must fail before reuse of the mesh window");
        assert_eq!(
            error.to_string(),
            "initial fields are not in canonical region, name, and path order"
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
