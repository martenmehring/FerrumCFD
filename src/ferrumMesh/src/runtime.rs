use std::mem::size_of;
use std::path::Path;

use crate::fields::{FieldLoadPolicy, FieldValueSummary, InitialFieldSet};
use crate::geometry::compute_poly_mesh_geometry;
use crate::poly_mesh::PolyMesh;
use crate::solver_state::{
    SolverStateCpuBufferStatus, SolverStateFieldKind, SolverStateFieldPlan, SolverStatePlan,
    SolverStateStorageStatus, SolverStateValueKind,
};
use crate::{MeshError, Point3, Result};

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
    pub values: Option<Vec<f64>>,
}

pub fn build_solver_runtime_data(
    case_dir: &Path,
    mesh: &PolyMesh,
    state: &SolverStatePlan,
    initial_fields: &mut InitialFieldSet,
    policy: FieldLoadPolicy,
) -> Result<SolverRuntimeData> {
    preflight_runtime_fields(mesh, state, initial_fields)?;
    let runtime_mesh = build_solver_runtime_mesh(mesh)?;
    let mut warnings = Vec::new();
    let mut fields = Vec::new();
    fields.try_reserve_exact(state.fields.len()).map_err(|_| {
        MeshError::InvalidInput("runtime field descriptor allocation failed".to_string())
    })?;
    let mut nonuniform_sources = Vec::new();
    nonuniform_sources
        .try_reserve_exact(state.fields.len())
        .map_err(|_| {
            MeshError::InvalidInput("runtime transfer table allocation failed".to_string())
        })?;

    for (source_index, field) in state.fields.iter().enumerate() {
        let label = try_runtime_field_label(field.region.as_deref(), &field.name)?;
        let Some(components) = field.storage.components else {
            if policy == FieldLoadPolicy::Full {
                push_runtime_warning(
                    &mut warnings,
                    format!("field '{label}' has no component count for runtime buffer"),
                )?;
            }
            continue;
        };
        let Some(scalar_slots) = field.storage.scalar_slots else {
            if policy == FieldLoadPolicy::Full {
                push_runtime_warning(
                    &mut warnings,
                    format!("field '{label}' has no scalar-slot count for runtime buffer"),
                )?;
            }
            continue;
        };
        let Some(bytes_f64) = scalar_slots.checked_mul(size_of::<f64>()) else {
            return Err(MeshError::InvalidInput(format!(
                "field '{label}' runtime byte size overflowed"
            )));
        };

        let (values, nonuniform_source) = match policy {
            FieldLoadPolicy::Summary => (None, None),
            FieldLoadPolicy::Full => match field.cpu_buffer.status {
                SolverStateCpuBufferStatus::UniformReady => {
                    (Some(materialize_uniform(field, scalar_slots)?), None)
                }
                SolverStateCpuBufferStatus::NonUniformReady => (None, Some(source_index)),
                _ => {
                    push_runtime_warning(
                        &mut warnings,
                        format!(
                            "field '{label}' is not materializable as a CPU f64 buffer: {}",
                            field.cpu_buffer.status
                        ),
                    )?;
                    continue;
                }
            },
        };

        let region = try_clone_optional_string(field.region.as_deref())?;
        let name = try_clone_runtime_string(&field.name)?;
        fields.push(SolverRuntimeFieldBuffer {
            region,
            name,
            kind: field.kind,
            components,
            scalar_slots,
            bytes_f64,
            values,
        });
        nonuniform_sources.push(nonuniform_source);
    }

    if policy == FieldLoadPolicy::Full && fields.is_empty() && !state.fields.is_empty() {
        push_runtime_warning(
            &mut warnings,
            format!(
                "no runtime field buffers were built for {}",
                case_dir.display()
            ),
        )?;
    }

    if policy == FieldLoadPolicy::Full {
        // Recheck every commit slot immediately before mutation. This makes an
        // invariant regression an ordinary pre-commit error rather than a
        // partial transfer or an indexing panic.
        for (runtime_index, source_index) in nonuniform_sources.iter().copied().enumerate() {
            let Some(source_index) = source_index else {
                continue;
            };
            let source_ready = initial_fields
                .fields
                .get(source_index)
                .and_then(|source| source.internal_field.as_ref())
                .is_some_and(|value| {
                    matches!(
                        value,
                        FieldValueSummary::NonUniform {
                            values: Some(_),
                            ..
                        }
                    )
                });
            if fields.get(runtime_index).is_none() || !source_ready {
                return Err(MeshError::InvalidInput(
                    "runtime nonuniform commit invariant changed before transfer".to_string(),
                ));
            }
        }

        // Every operation above is fallible; this commit pass is deliberately
        // allocation-free. Preflight proved every marked source has the exact
        // shape, so moving the Vec cannot fail and preserves its allocation.
        for (runtime_index, source_index) in nonuniform_sources.iter().copied().enumerate() {
            let Some(source_index) = source_index else {
                continue;
            };
            let moved = initial_fields
                .fields
                .get_mut(source_index)
                .and_then(|source| source.internal_field.as_mut())
                .and_then(|value| match value {
                    FieldValueSummary::NonUniform { values, .. } => values.take(),
                    _ => None,
                });
            debug_assert!(
                moved.is_some(),
                "preflighted nonuniform payload disappeared"
            );
            if let Some(values) = moved {
                fields[runtime_index].values = Some(values);
            }
        }
        discard_remaining_internal_payloads(initial_fields);
    }

    Ok(SolverRuntimeData {
        mesh: runtime_mesh,
        fields,
        warnings,
    })
}

fn preflight_runtime_fields(
    mesh: &PolyMesh,
    state: &SolverStatePlan,
    initial_fields: &InitialFieldSet,
) -> Result<()> {
    if state.fields.len() != initial_fields.fields.len() {
        return Err(MeshError::InvalidInput(format!(
            "runtime field identity mismatch: state has {} fields, initial fields have {}",
            state.fields.len(),
            initial_fields.fields.len()
        )));
    }

    for (planned, source) in state.fields.iter().zip(&initial_fields.fields) {
        let planned_label = try_runtime_field_label(planned.region.as_deref(), &planned.name)?;
        let source_label = try_runtime_field_label(source.region.as_deref(), &source.name)?;
        if planned.region != source.region
            || planned.name != source.name
            || planned.class_name != source.class_name
            || planned.dimensions != source.dimensions
            || planned.boundary_patches != source.boundary_patches.len()
        {
            return Err(MeshError::InvalidInput(format!(
                "runtime field identity mismatch: planned '{}', source '{}'",
                planned_label, source_label
            )));
        }
        preflight_runtime_descriptor(mesh, planned, source, &planned_label)?;
    }
    Ok(())
}

fn preflight_runtime_descriptor(
    mesh: &PolyMesh,
    planned: &SolverStateFieldPlan,
    source: &crate::fields::FieldFile,
    label: &str,
) -> Result<()> {
    let expected_kind = runtime_kind_from_class(source.class_name.as_deref());
    if planned.kind != expected_kind {
        return Err(runtime_descriptor_mismatch(label, "field kind"));
    }
    if planned.region.is_none()
        && (planned.mesh_cells != Some(mesh.cell_count())
            || planned.mesh_faces != Some(mesh.faces.len())
            || planned.mesh_boundary_patches != Some(mesh.patches.len()))
    {
        return Err(runtime_descriptor_mismatch(label, "base mesh shape"));
    }

    let expected_components = runtime_components(expected_kind);
    let expected_count = match expected_kind {
        SolverStateFieldKind::VolScalar | SolverStateFieldKind::VolVector => planned.mesh_cells,
        SolverStateFieldKind::SurfaceScalar => planned.mesh_faces,
        SolverStateFieldKind::Other => None,
    };
    let expected_slots = expected_components
        .zip(expected_count)
        .map(|(components, count)| {
            count.checked_mul(components).ok_or_else(|| {
                MeshError::InvalidInput(format!(
                    "field '{label}' runtime scalar-slot count overflowed"
                ))
            })
        })
        .transpose()?;
    let expected_bytes = expected_slots
        .map(|slots| {
            slots.checked_mul(size_of::<f64>()).ok_or_else(|| {
                MeshError::InvalidInput(format!("field '{label}' runtime byte size overflowed"))
            })
        })
        .transpose()?;
    let expected_storage_status = if expected_kind == SolverStateFieldKind::Other {
        SolverStateStorageStatus::UnsupportedClass
    } else {
        SolverStateStorageStatus::Loaded
    };
    let storage_capable = expected_storage_status == SolverStateStorageStatus::Loaded;
    if planned.storage.components != expected_components
        || planned.storage.scalar_slots != expected_slots
        || planned.storage.bytes_f64 != expected_bytes
        || planned.storage.status != expected_storage_status
        || planned.storage.cpu_capable != storage_capable
        || planned.storage.gpu_capable != storage_capable
        || planned.cpu_buffer.scalar_slots != expected_slots
        || planned.cpu_buffer.bytes_f64 != expected_bytes
    {
        return Err(runtime_descriptor_mismatch(
            label,
            "storage/cpu shape or byte metadata",
        ));
    }

    let (internal_kind, value_count, valid_count, uniform_components, loaded_scalars, status) =
        expected_internal_descriptor(source, expected_kind, expected_count, expected_slots)?;
    if planned.internal_field.kind != internal_kind
        || planned.internal_field.value_count != value_count
        || planned.internal_field.expected_count != expected_count
        || planned.internal_field.valid_count != valid_count
        || planned.internal_field.uniform_components != uniform_components
        || planned.internal_field.loaded_scalars != loaded_scalars
        || planned.cpu_buffer.status != status
        || planned.cpu_buffer.materializable
            != matches!(
                status,
                SolverStateCpuBufferStatus::UniformReady
                    | SolverStateCpuBufferStatus::NonUniformReady
            )
    {
        return Err(runtime_descriptor_mismatch(
            label,
            "internal-field status or source provenance",
        ));
    }
    Ok(())
}

type ExpectedInternalDescriptor = (
    SolverStateValueKind,
    Option<usize>,
    Option<bool>,
    Option<Vec<f64>>,
    Option<usize>,
    SolverStateCpuBufferStatus,
);

fn expected_internal_descriptor(
    source: &crate::fields::FieldFile,
    kind: SolverStateFieldKind,
    expected_count: Option<usize>,
    expected_slots: Option<usize>,
) -> Result<ExpectedInternalDescriptor> {
    if kind == SolverStateFieldKind::Other {
        let internal_kind = source_internal_kind(source);
        let value_count = source_nonuniform_count(source);
        return Ok((
            internal_kind,
            value_count,
            None,
            source_uniform_components(source)?,
            source_loaded_scalars(source),
            SolverStateCpuBufferStatus::UnsupportedClass,
        ));
    }

    match &source.internal_field {
        Some(FieldValueSummary::Uniform(value)) => {
            let uniform = parse_runtime_uniform_components(value)?;
            let valid_count = expected_count.map(|_| true);
            let ready = expected_count.is_some()
                && expected_slots.is_some()
                && runtime_components(kind)
                    .zip(uniform.as_ref())
                    .is_some_and(|(components, values)| values.len() == components);
            Ok((
                SolverStateValueKind::Uniform,
                expected_count,
                valid_count,
                uniform,
                None,
                if ready {
                    SolverStateCpuBufferStatus::UniformReady
                } else {
                    SolverStateCpuBufferStatus::InvalidShape
                },
            ))
        }
        Some(FieldValueSummary::NonUniform {
            value_type,
            count,
            values,
        }) => {
            let supported = runtime_nonuniform_type_is_supported(value_type.as_deref());
            let valid_count = count
                .zip(expected_count)
                .map(|(count, expected)| count == expected);
            let loaded_scalars = supported.then(|| values.as_ref().map(Vec::len)).flatten();
            let status = if !supported {
                SolverStateCpuBufferStatus::UnsupportedInternalField
            } else if valid_count == Some(true)
                && loaded_scalars.is_some_and(|loaded| Some(loaded) == expected_slots)
            {
                SolverStateCpuBufferStatus::NonUniformReady
            } else if valid_count == Some(false) || loaded_scalars.is_some() {
                SolverStateCpuBufferStatus::InvalidShape
            } else {
                SolverStateCpuBufferStatus::NonUniformDataNotLoaded
            };
            Ok((
                if supported {
                    SolverStateValueKind::NonUniform
                } else {
                    SolverStateValueKind::UnsupportedNonUniform
                },
                *count,
                valid_count,
                None,
                loaded_scalars,
                status,
            ))
        }
        Some(FieldValueSummary::Other(_)) => Ok((
            SolverStateValueKind::Other,
            None,
            None,
            None,
            None,
            SolverStateCpuBufferStatus::UnsupportedInternalField,
        )),
        None => Ok((
            SolverStateValueKind::Missing,
            None,
            Some(false),
            None,
            None,
            SolverStateCpuBufferStatus::MissingInternalField,
        )),
    }
}

fn source_internal_kind(source: &crate::fields::FieldFile) -> SolverStateValueKind {
    match &source.internal_field {
        Some(FieldValueSummary::Uniform(_)) => SolverStateValueKind::Uniform,
        Some(FieldValueSummary::NonUniform { value_type, .. })
            if runtime_nonuniform_type_is_supported(value_type.as_deref()) =>
        {
            SolverStateValueKind::NonUniform
        }
        Some(FieldValueSummary::NonUniform { .. }) => SolverStateValueKind::UnsupportedNonUniform,
        Some(FieldValueSummary::Other(_)) => SolverStateValueKind::Other,
        None => SolverStateValueKind::Missing,
    }
}

fn source_nonuniform_count(source: &crate::fields::FieldFile) -> Option<usize> {
    match &source.internal_field {
        Some(FieldValueSummary::NonUniform { count, .. }) => *count,
        _ => None,
    }
}

fn source_uniform_components(source: &crate::fields::FieldFile) -> Result<Option<Vec<f64>>> {
    match &source.internal_field {
        Some(FieldValueSummary::Uniform(value)) => parse_runtime_uniform_components(value),
        _ => Ok(None),
    }
}

fn source_loaded_scalars(source: &crate::fields::FieldFile) -> Option<usize> {
    match &source.internal_field {
        Some(FieldValueSummary::NonUniform {
            value_type, values, ..
        }) if runtime_nonuniform_type_is_supported(value_type.as_deref()) => {
            values.as_ref().map(Vec::len)
        }
        _ => None,
    }
}

fn parse_runtime_uniform_components(value: &str) -> Result<Option<Vec<f64>>> {
    let mut values = Vec::new();
    for token in value
        .split(|character: char| character.is_whitespace() || matches!(character, '(' | ')'))
        .filter(|token| !token.is_empty())
    {
        let Ok(number) = token.parse::<f64>() else {
            return Ok(None);
        };
        if !number.is_finite() {
            return Ok(None);
        }
        values.try_reserve(1).map_err(|_| {
            MeshError::InvalidInput("runtime uniform preflight allocation failed".to_string())
        })?;
        values.push(number);
    }
    Ok((!values.is_empty()).then_some(values))
}

fn runtime_kind_from_class(class_name: Option<&str>) -> SolverStateFieldKind {
    match class_name {
        Some("volScalarField") => SolverStateFieldKind::VolScalar,
        Some("volVectorField") => SolverStateFieldKind::VolVector,
        Some("surfaceScalarField") => SolverStateFieldKind::SurfaceScalar,
        _ => SolverStateFieldKind::Other,
    }
}

fn runtime_components(kind: SolverStateFieldKind) -> Option<usize> {
    match kind {
        SolverStateFieldKind::VolScalar | SolverStateFieldKind::SurfaceScalar => Some(1),
        SolverStateFieldKind::VolVector => Some(3),
        SolverStateFieldKind::Other => None,
    }
}

fn runtime_nonuniform_type_is_supported(value_type: Option<&str>) -> bool {
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

fn runtime_descriptor_mismatch(label: &str, detail: &str) -> MeshError {
    MeshError::InvalidInput(format!(
        "field '{label}' runtime descriptor mismatch: {detail}"
    ))
}

fn materialize_uniform(field: &SolverStateFieldPlan, scalar_slots: usize) -> Result<Vec<f64>> {
    let components = field
        .internal_field
        .uniform_components
        .as_deref()
        .ok_or_else(|| {
            MeshError::InvalidInput("uniform runtime components are missing".to_string())
        })?;
    if components.is_empty() || !scalar_slots.is_multiple_of(components.len()) {
        return Err(MeshError::InvalidInput(
            "uniform runtime shape is invalid".to_string(),
        ));
    }
    let mut values = Vec::new();
    values
        .try_reserve_exact(scalar_slots)
        .map_err(|_| MeshError::InvalidInput("uniform runtime allocation failed".to_string()))?;
    for _ in 0..scalar_slots / components.len() {
        values.extend_from_slice(components);
    }
    Ok(values)
}

fn discard_remaining_internal_payloads(fields: &mut InitialFieldSet) {
    for field in &mut fields.fields {
        if let Some(FieldValueSummary::NonUniform { values, .. }) = field.internal_field.as_mut() {
            *values = None;
        }
    }
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

fn try_runtime_field_label(region: Option<&str>, name: &str) -> Result<String> {
    if let Some(region) = region {
        let length = region
            .len()
            .checked_add(1)
            .and_then(|length| length.checked_add(name.len()))
            .ok_or_else(|| {
                MeshError::InvalidInput("runtime field label length overflowed".to_string())
            })?;
        let mut label = String::new();
        label.try_reserve_exact(length).map_err(|_| {
            MeshError::InvalidInput("runtime field label allocation failed".to_string())
        })?;
        label.push_str(region);
        label.push('/');
        label.push_str(name);
        Ok(label)
    } else {
        try_clone_runtime_string(name)
    }
}

fn push_runtime_warning(warnings: &mut Vec<String>, warning: String) -> Result<()> {
    warnings
        .try_reserve(1)
        .map_err(|_| MeshError::InvalidInput("runtime warning allocation failed".to_string()))?;
    warnings.push(warning);
    Ok(())
}

fn try_clone_runtime_string(value: &str) -> Result<String> {
    let mut cloned = String::new();
    cloned
        .try_reserve_exact(value.len())
        .map_err(|_| MeshError::InvalidInput("runtime string allocation failed".to_string()))?;
    cloned.push_str(value);
    Ok(cloned)
}

fn try_clone_optional_string(value: Option<&str>) -> Result<Option<String>> {
    value.map(try_clone_runtime_string).transpose()
}

fn vector_magnitude(vector: Point3) -> f64 {
    (vector.x * vector.x + vector.y * vector.y + vector.z * vector.z).sqrt()
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use crate::Point3;
    use crate::fields::{FieldFile, FieldLoadPolicy, FieldValueSummary, InitialFieldSet};
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
            fields: vec![
                SolverStateFieldPlan {
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
                        loaded_scalars: None,
                    },
                    boundary_patches: 0,
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
                },
                SolverStateFieldPlan {
                    region: None,
                    name: "T".to_string(),
                    class_name: Some("volScalarField".to_string()),
                    kind: SolverStateFieldKind::VolScalar,
                    dimensions: None,
                    mesh_cells: Some(1),
                    mesh_faces: Some(6),
                    internal_field: SolverStateInternalFieldPlan {
                        kind: SolverStateValueKind::NonUniform,
                        value_count: Some(1),
                        expected_count: Some(1),
                        valid_count: Some(true),
                        uniform_components: None,
                        loaded_scalars: Some(1),
                    },
                    boundary_patches: 0,
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
                        status: SolverStateCpuBufferStatus::NonUniformReady,
                    },
                },
            ],
            warnings: Vec::new(),
        };
        let mut full_fields = initial_fields();
        let source_pointer = match &full_fields.fields[1].internal_field {
            Some(FieldValueSummary::NonUniform {
                values: Some(values),
                ..
            }) => values.as_ptr(),
            _ => panic!("nonuniform fixture missing"),
        };

        let runtime = build_solver_runtime_data(
            Path::new("case"),
            &mesh,
            &state,
            &mut full_fields,
            FieldLoadPolicy::Full,
        )
        .expect("runtime data should build");

        assert_eq!(runtime.mesh.cells, 1);
        assert_eq!(runtime.mesh.faces, 6);
        assert_eq!(runtime.mesh.owner, vec![0; 6]);
        assert_eq!(runtime.mesh.neighbour, vec![None; 6]);
        assert_eq!(runtime.mesh.cell_centres.len(), 1);
        assert_eq!(runtime.mesh.face_centres.len(), 6);
        assert_eq!(runtime.mesh.face_area_vectors.len(), 6);
        assert_close(runtime.mesh.cell_volumes[0], 1.0);
        assert_eq!(runtime.fields.len(), 2);
        assert_eq!(runtime.fields[0].name, "p");
        assert_eq!(runtime.fields[0].values, Some(vec![7.0]));
        assert_eq!(runtime.fields[1].values.as_deref(), Some(&[300.0][..]));
        assert_eq!(
            runtime.fields[1].values.as_ref().map(Vec::as_ptr),
            Some(source_pointer)
        );
        assert!(matches!(
            full_fields.fields[1].internal_field,
            Some(FieldValueSummary::NonUniform { values: None, .. })
        ));
        assert!(runtime.warnings.is_empty());

        let mut summary_fields = initial_fields();
        let summary = build_solver_runtime_data(
            Path::new("case"),
            &mesh,
            &state,
            &mut summary_fields,
            FieldLoadPolicy::Summary,
        )
        .expect("summary runtime should build");
        assert!(summary.fields.iter().all(|field| field.values.is_none()));
        assert!(summary.warnings.is_empty());

        let mut inconsistent = state.clone();
        inconsistent.fields[1].storage.scalar_slots = Some(2);
        let mut untouched = initial_fields();
        let original_pointer = match &untouched.fields[1].internal_field {
            Some(FieldValueSummary::NonUniform {
                values: Some(values),
                ..
            }) => values.as_ptr(),
            _ => panic!("nonuniform fixture missing"),
        };
        let error = build_solver_runtime_data(
            Path::new("case"),
            &mesh,
            &inconsistent,
            &mut untouched,
            FieldLoadPolicy::Full,
        )
        .expect_err("inconsistent storage/cpu descriptors must fail before transfer");
        assert!(error.to_string().contains("runtime descriptor mismatch"));
        let preserved = match &untouched.fields[1].internal_field {
            Some(FieldValueSummary::NonUniform {
                values: Some(values),
                ..
            }) => values,
            _ => panic!("preflight failure consumed the source payload"),
        };
        assert_eq!(preserved.as_ptr(), original_pointer);
        assert_eq!(preserved, &[300.0]);
    }

    fn initial_fields() -> InitialFieldSet {
        InitialFieldSet {
            case_dir: PathBuf::from("case"),
            fields: vec![
                FieldFile {
                    path: PathBuf::from("case/0/p"),
                    region: None,
                    name: "p".to_string(),
                    class_name: Some("volScalarField".to_string()),
                    dimensions: None,
                    internal_field: Some(FieldValueSummary::Uniform("7".to_string())),
                    boundary_patches: Vec::new(),
                },
                FieldFile {
                    path: PathBuf::from("case/0/T"),
                    region: None,
                    name: "T".to_string(),
                    class_name: Some("volScalarField".to_string()),
                    dimensions: None,
                    internal_field: Some(FieldValueSummary::NonUniform {
                        value_type: Some("List<scalar>".to_string()),
                        count: Some(1),
                        values: Some(vec![300.0]),
                    }),
                    boundary_patches: Vec::new(),
                },
            ],
        }
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
