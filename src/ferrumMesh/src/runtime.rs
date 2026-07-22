use std::mem::size_of;
use std::path::Path;

use crate::fields::{FieldLoadPolicy, FieldValueSummary, InitialFieldSet};
use crate::geometry::compute_poly_mesh_geometry;
use crate::poly_mesh::PolyMesh;
use crate::solver_state::{
    SolverStateCpuBufferStatus, SolverStateFieldKind, SolverStateFieldPlan, SolverStatePlan,
    derive_field_descriptors,
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
            push_runtime_warning(
                &mut warnings,
                format!("field '{label}' has no component count for runtime buffer"),
            )?;
            continue;
        };
        let Some(scalar_slots) = field.storage.scalar_slots else {
            push_runtime_warning(
                &mut warnings,
                format!("field '{label}' has no scalar-slot count for runtime buffer"),
            )?;
            continue;
        };
        let Some(bytes_f64) = scalar_slots.checked_mul(size_of::<f64>()) else {
            return Err(MeshError::InvalidInput(format!(
                "field '{label}' runtime byte size overflowed"
            )));
        };

        let (values, nonuniform_source) = match field.cpu_buffer.status {
            SolverStateCpuBufferStatus::UniformReady => match policy {
                FieldLoadPolicy::Summary => (None, None),
                FieldLoadPolicy::Full => (Some(materialize_uniform(field, scalar_slots)?), None),
            },
            SolverStateCpuBufferStatus::NonUniformReady => match policy {
                FieldLoadPolicy::Summary => (None, None),
                FieldLoadPolicy::Full => (None, Some(source_index)),
            },
            SolverStateCpuBufferStatus::NonUniformDataNotLoaded
                if policy == FieldLoadPolicy::Summary =>
            {
                (None, None)
            }
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

    if fields.is_empty() && !state.fields.is_empty() {
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
    let (mesh_cells, mesh_faces) = if planned.region.is_none() {
        if planned.mesh_cells != Some(mesh.cell_count())
            || planned.mesh_faces != Some(mesh.faces.len())
            || planned.mesh_boundary_patches != Some(mesh.patches.len())
        {
            return Err(runtime_descriptor_mismatch(label, "base mesh shape"));
        }
        (Some(mesh.cell_count()), Some(mesh.faces.len()))
    } else {
        // Region meshes were read while the plan was built and are not retained
        // at runtime. Their sealed shape is therefore authoritative here; the
        // runtime base mesh must never be substituted for a region shape.
        (planned.mesh_cells, planned.mesh_faces)
    };
    let expected = derive_field_descriptors(source, mesh_cells, mesh_faces)?;
    if expected.kind != SolverStateFieldKind::Other
        && expected.internal_field.expected_count.is_some()
    {
        if expected.storage.scalar_slots.is_none() {
            return Err(MeshError::InvalidInput(format!(
                "field '{label}' runtime scalar-slot count overflowed"
            )));
        }
        if expected.storage.bytes_f64.is_none() {
            return Err(MeshError::InvalidInput(format!(
                "field '{label}' runtime byte size overflowed"
            )));
        }
    }

    if planned.kind != expected.kind {
        return Err(runtime_descriptor_mismatch(label, "field kind"));
    }
    if planned.storage.components != expected.storage.components
        || planned.storage.scalar_slots != expected.storage.scalar_slots
        || planned.storage.bytes_f64 != expected.storage.bytes_f64
        || planned.storage.status != expected.storage.status
        || planned.storage.cpu_capable != expected.storage.cpu_capable
        || planned.storage.gpu_capable != expected.storage.gpu_capable
        || planned.cpu_buffer.scalar_slots != expected.cpu_buffer.scalar_slots
        || planned.cpu_buffer.bytes_f64 != expected.cpu_buffer.bytes_f64
    {
        return Err(runtime_descriptor_mismatch(
            label,
            "storage/cpu shape or byte metadata",
        ));
    }

    if planned.internal_field.kind != expected.internal_field.kind
        || planned.internal_field.value_count != expected.internal_field.value_count
        || planned.internal_field.expected_count != expected.internal_field.expected_count
        || planned.internal_field.valid_count != expected.internal_field.valid_count
        || planned.internal_field.uniform_components != expected.internal_field.uniform_components
        || planned.internal_field.loaded_scalars != expected.internal_field.loaded_scalars
        || planned.cpu_buffer.status != expected.cpu_buffer.status
        || planned.cpu_buffer.materializable != expected.cpu_buffer.materializable
    {
        return Err(runtime_descriptor_mismatch(
            label,
            "internal-field status or source provenance",
        ));
    }
    Ok(())
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
        let state = runtime_state();
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

    #[test]
    fn rejects_nonuniform_materializable_claim_before_transfer() {
        let mesh = unit_cube_mesh();
        let mut state = runtime_state();
        state.fields[1].cpu_buffer.materializable = true;
        let mut fields = initial_fields();
        let original_pointer = match &fields.fields[1].internal_field {
            Some(FieldValueSummary::NonUniform {
                values: Some(values),
                ..
            }) => values.as_ptr(),
            _ => panic!("nonuniform fixture missing"),
        };

        let error = build_solver_runtime_data(
            Path::new("case"),
            &mesh,
            &state,
            &mut fields,
            FieldLoadPolicy::Full,
        )
        .expect_err("nonuniform state cannot claim direct materializability");

        assert_eq!(
            error.to_string(),
            "field 'T' runtime descriptor mismatch: internal-field status or source provenance"
        );
        let preserved = match &fields.fields[1].internal_field {
            Some(FieldValueSummary::NonUniform {
                values: Some(values),
                ..
            }) => values,
            _ => panic!("descriptor rejection consumed the nonuniform source"),
        };
        assert_eq!(preserved.as_ptr(), original_pointer);
        assert_eq!(preserved, &[300.0]);
    }

    #[test]
    fn base_and_region_shapes_have_distinct_authority() {
        let mesh = unit_cube_mesh();
        let mut region_state = runtime_state();
        region_state.fields.remove(0);
        let planned = &mut region_state.fields[0];
        planned.region = Some("fluid".to_string());
        planned.mesh_cells = Some(2);
        planned.mesh_faces = Some(12);
        planned.mesh_boundary_patches = Some(2);
        planned.internal_field.value_count = Some(2);
        planned.internal_field.expected_count = Some(2);
        planned.internal_field.loaded_scalars = Some(2);
        planned.storage.scalar_slots = Some(2);
        planned.storage.bytes_f64 = Some(16);
        planned.cpu_buffer.scalar_slots = Some(2);
        planned.cpu_buffer.bytes_f64 = Some(16);

        let mut region_fields = initial_fields();
        region_fields.fields.remove(0);
        let source = &mut region_fields.fields[0];
        source.region = Some("fluid".to_string());
        let Some(FieldValueSummary::NonUniform { count, values, .. }) =
            source.internal_field.as_mut()
        else {
            panic!("nonuniform region fixture missing");
        };
        *count = Some(2);
        *values = Some(vec![300.0, 301.0]);
        let source_pointer = values.as_ref().map(Vec::as_ptr);

        let runtime = build_solver_runtime_data(
            Path::new("case"),
            &mesh,
            &region_state,
            &mut region_fields,
            FieldLoadPolicy::Full,
        )
        .expect("region field must use its planned region shape, not the runtime base mesh");
        assert_eq!(runtime.fields[0].region.as_deref(), Some("fluid"));
        assert_eq!(runtime.fields[0].scalar_slots, 2);
        assert_eq!(
            runtime.fields[0].values.as_ref().map(Vec::as_ptr),
            source_pointer
        );

        let mut base_state = region_state;
        base_state.fields[0].region = None;
        let mut base_fields = initial_fields();
        base_fields.fields.remove(0);
        let source = &mut base_fields.fields[0];
        let Some(FieldValueSummary::NonUniform { count, values, .. }) =
            source.internal_field.as_mut()
        else {
            panic!("nonuniform base fixture missing");
        };
        *count = Some(2);
        *values = Some(vec![300.0, 301.0]);
        let source_pointer = values.as_ref().map(Vec::as_ptr);

        let error = build_solver_runtime_data(
            Path::new("case"),
            &mesh,
            &base_state,
            &mut base_fields,
            FieldLoadPolicy::Full,
        )
        .expect_err("base field shape must be derived from the runtime base mesh");
        assert_eq!(
            error.to_string(),
            "field 'T' runtime descriptor mismatch: base mesh shape"
        );
        let Some(FieldValueSummary::NonUniform {
            values: Some(values),
            ..
        }) = &base_fields.fields[0].internal_field
        else {
            panic!("base shape rejection consumed the source payload");
        };
        assert_eq!(Some(values.as_ptr()), source_pointer);
        assert_eq!(values, &[300.0, 301.0]);
    }

    #[test]
    fn later_source_tampering_preserves_all_nonuniform_allocations() {
        let mesh = unit_cube_mesh();
        let mut state = runtime_state();
        state.fields.remove(0);
        let mut second_plan = state.fields[0].clone();
        state.fields[0].name = "A".to_string();
        second_plan.name = "B".to_string();
        state.fields.push(second_plan);

        let mut fields = initial_fields();
        fields.fields.remove(0);
        let mut second_fields = initial_fields();
        second_fields.fields.remove(0);
        let mut second_source = second_fields.fields.remove(0);
        fields.fields[0].name = "A".to_string();
        second_source.name = "B".to_string();
        let Some(FieldValueSummary::NonUniform { count, .. }) =
            second_source.internal_field.as_mut()
        else {
            panic!("second nonuniform fixture missing");
        };
        *count = Some(2);
        fields.fields.push(second_source);

        let source_pointers = fields
            .fields
            .iter()
            .map(|field| match &field.internal_field {
                Some(FieldValueSummary::NonUniform {
                    values: Some(values),
                    ..
                }) => values.as_ptr(),
                _ => panic!("nonuniform fixture missing"),
            })
            .collect::<Vec<_>>();

        let error = build_solver_runtime_data(
            Path::new("case"),
            &mesh,
            &state,
            &mut fields,
            FieldLoadPolicy::Full,
        )
        .expect_err("tampered later source descriptor must fail before any transfer");
        assert_eq!(
            error.to_string(),
            "field 'B' runtime descriptor mismatch: internal-field status or source provenance"
        );
        for (field, expected_pointer) in fields.fields.iter().zip(source_pointers) {
            let Some(FieldValueSummary::NonUniform {
                values: Some(values),
                ..
            }) = &field.internal_field
            else {
                panic!("all-or-nothing preflight consumed a source payload");
            };
            assert_eq!(values.as_ptr(), expected_pointer);
            assert_eq!(values, &[300.0]);
        }
    }

    #[test]
    fn region_storage_overflow_remains_a_hard_preflight_error() {
        let mesh = unit_cube_mesh();

        let mut vector_state = runtime_state();
        vector_state.fields.remove(0);
        let vector_plan = &mut vector_state.fields[0];
        vector_plan.region = Some("fluid".to_string());
        vector_plan.class_name = Some("volVectorField".to_string());
        vector_plan.kind = SolverStateFieldKind::VolVector;
        vector_plan.mesh_cells = Some(usize::MAX);
        vector_plan.internal_field.value_count = Some(usize::MAX);
        vector_plan.internal_field.expected_count = Some(usize::MAX);
        vector_plan.internal_field.valid_count = Some(true);
        vector_plan.internal_field.loaded_scalars = None;
        vector_plan.storage.components = Some(3);
        vector_plan.storage.scalar_slots = None;
        vector_plan.storage.bytes_f64 = None;
        vector_plan.cpu_buffer.scalar_slots = None;
        vector_plan.cpu_buffer.bytes_f64 = None;
        vector_plan.cpu_buffer.status = SolverStateCpuBufferStatus::NonUniformDataNotLoaded;

        let mut vector_fields = initial_fields();
        vector_fields.fields.remove(0);
        let vector_source = &mut vector_fields.fields[0];
        vector_source.region = Some("fluid".to_string());
        vector_source.class_name = Some("volVectorField".to_string());
        let Some(FieldValueSummary::NonUniform {
            value_type,
            count,
            values,
        }) = vector_source.internal_field.as_mut()
        else {
            panic!("nonuniform vector fixture missing");
        };
        *value_type = Some("List<vector>".to_string());
        *count = Some(usize::MAX);
        *values = None;

        let vector_error = build_solver_runtime_data(
            Path::new("case"),
            &mesh,
            &vector_state,
            &mut vector_fields,
            FieldLoadPolicy::Full,
        )
        .expect_err("vector scalar-slot overflow must fail before runtime construction");
        assert_eq!(
            vector_error.to_string(),
            "field 'fluid/T' runtime scalar-slot count overflowed"
        );

        let mut scalar_state = runtime_state();
        scalar_state.fields.remove(0);
        let scalar_plan = &mut scalar_state.fields[0];
        scalar_plan.region = Some("fluid".to_string());
        scalar_plan.mesh_cells = Some(usize::MAX);
        scalar_plan.internal_field.value_count = Some(usize::MAX);
        scalar_plan.internal_field.expected_count = Some(usize::MAX);
        scalar_plan.internal_field.valid_count = Some(true);
        scalar_plan.internal_field.loaded_scalars = None;
        scalar_plan.storage.scalar_slots = Some(usize::MAX);
        scalar_plan.storage.bytes_f64 = None;
        scalar_plan.cpu_buffer.scalar_slots = Some(usize::MAX);
        scalar_plan.cpu_buffer.bytes_f64 = None;
        scalar_plan.cpu_buffer.status = SolverStateCpuBufferStatus::NonUniformDataNotLoaded;

        let mut scalar_fields = initial_fields();
        scalar_fields.fields.remove(0);
        let scalar_source = &mut scalar_fields.fields[0];
        scalar_source.region = Some("fluid".to_string());
        let Some(FieldValueSummary::NonUniform { count, values, .. }) =
            scalar_source.internal_field.as_mut()
        else {
            panic!("nonuniform scalar fixture missing");
        };
        *count = Some(usize::MAX);
        *values = None;

        let scalar_error = build_solver_runtime_data(
            Path::new("case"),
            &mesh,
            &scalar_state,
            &mut scalar_fields,
            FieldLoadPolicy::Full,
        )
        .expect_err("f64 byte-size overflow must fail before runtime construction");
        assert_eq!(
            scalar_error.to_string(),
            "field 'fluid/T' runtime byte size overflowed"
        );
    }

    #[test]
    fn unsupported_class_with_missing_internal_field_is_consistently_omitted() {
        let mesh = unit_cube_mesh();
        let mut state = runtime_state();
        let planned = &mut state.fields[0];
        planned.class_name = Some("dictionary".to_string());
        planned.kind = SolverStateFieldKind::Other;
        planned.internal_field = SolverStateInternalFieldPlan {
            kind: SolverStateValueKind::Missing,
            value_count: None,
            expected_count: None,
            valid_count: Some(false),
            uniform_components: None,
            loaded_scalars: None,
        };
        planned.storage = SolverStateStoragePlan {
            cpu_capable: false,
            gpu_capable: false,
            components: None,
            scalar_slots: None,
            bytes_f64: None,
            status: SolverStateStorageStatus::UnsupportedClass,
        };
        planned.cpu_buffer = SolverStateCpuBufferPlan {
            materializable: false,
            scalar_slots: None,
            bytes_f64: None,
            status: SolverStateCpuBufferStatus::UnsupportedClass,
        };

        let mut fields = initial_fields();
        fields.fields[0].class_name = Some("dictionary".to_string());
        fields.fields[0].internal_field = None;

        let runtime = build_solver_runtime_data(
            Path::new("case"),
            &mesh,
            &state,
            &mut fields,
            FieldLoadPolicy::Full,
        )
        .expect("unsupported descriptors must be consistently omitted, not misclassified");
        assert_eq!(
            runtime
                .fields
                .iter()
                .map(|field| field.name.as_str())
                .collect::<Vec<_>>(),
            vec!["T"]
        );
        assert_eq!(
            runtime.warnings,
            vec!["field 'p' has no component count for runtime buffer"]
        );
    }

    #[test]
    fn full_policy_warns_and_omits_unloaded_nonuniform_descriptor() {
        let mesh = unit_cube_mesh();
        let mut state = runtime_state();
        state.fields[1].internal_field.loaded_scalars = None;
        state.fields[1].cpu_buffer.status = SolverStateCpuBufferStatus::NonUniformDataNotLoaded;
        let mut fields = initial_fields();
        let Some(FieldValueSummary::NonUniform { values, .. }) =
            fields.fields[1].internal_field.as_mut()
        else {
            panic!("nonuniform summary fixture missing");
        };
        *values = None;

        let runtime = build_solver_runtime_data(
            Path::new("case"),
            &mesh,
            &state,
            &mut fields,
            FieldLoadPolicy::Full,
        )
        .expect("full runtime should omit an unloaded nonuniform payload");

        assert_eq!(
            runtime
                .fields
                .iter()
                .map(|field| field.name.as_str())
                .collect::<Vec<_>>(),
            vec!["p"]
        );
        assert_eq!(
            runtime.warnings,
            vec!["field 'T' is not materializable as a CPU f64 buffer: nonuniform-data-not-loaded"]
        );
    }

    #[test]
    fn summary_and_full_select_and_warn_for_invalid_fields_identically() {
        let mesh = unit_cube_mesh();
        let mut full_state = runtime_state();
        full_state.fields.push(invalid_uniform_vector_state());

        let mut full_fields = initial_fields();
        full_fields.fields.push(invalid_uniform_vector_field());
        let full = build_solver_runtime_data(
            Path::new("case"),
            &mesh,
            &full_state,
            &mut full_fields,
            FieldLoadPolicy::Full,
        )
        .expect("full runtime should reject only the invalid field");

        let mut summary_state = runtime_state();
        summary_state.fields[1].internal_field.loaded_scalars = None;
        summary_state.fields[1].cpu_buffer.status =
            SolverStateCpuBufferStatus::NonUniformDataNotLoaded;
        summary_state.fields.push(invalid_uniform_vector_state());
        let mut summary_fields = initial_fields();
        let Some(FieldValueSummary::NonUniform { values, .. }) =
            summary_fields.fields[1].internal_field.as_mut()
        else {
            panic!("nonuniform summary fixture missing");
        };
        *values = None;
        summary_fields.fields.push(invalid_uniform_vector_field());
        let summary = build_solver_runtime_data(
            Path::new("case"),
            &mesh,
            &summary_state,
            &mut summary_fields,
            FieldLoadPolicy::Summary,
        )
        .expect("summary runtime should reject only the invalid field");

        let full_names = full
            .fields
            .iter()
            .map(|field| field.name.as_str())
            .collect::<Vec<_>>();
        let summary_names = summary
            .fields
            .iter()
            .map(|field| field.name.as_str())
            .collect::<Vec<_>>();
        assert_eq!(full_names, vec!["p", "T"]);
        assert_eq!(summary_names, full_names);
        assert_eq!(
            full.warnings,
            vec!["field 'U' is not materializable as a CPU f64 buffer: invalid-shape"]
        );
        assert_eq!(summary.warnings, full.warnings);
        assert!(full.fields.iter().all(|field| field.values.is_some()));
        assert!(summary.fields.iter().all(|field| field.values.is_none()));
    }

    fn runtime_state() -> SolverStatePlan {
        SolverStatePlan {
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
                        materializable: false,
                        scalar_slots: Some(1),
                        bytes_f64: Some(8),
                        status: SolverStateCpuBufferStatus::NonUniformReady,
                    },
                },
            ],
            warnings: Vec::new(),
        }
    }

    fn invalid_uniform_vector_state() -> SolverStateFieldPlan {
        SolverStateFieldPlan {
            region: None,
            name: "U".to_string(),
            class_name: Some("volVectorField".to_string()),
            kind: SolverStateFieldKind::VolVector,
            dimensions: None,
            mesh_cells: Some(1),
            mesh_faces: Some(6),
            internal_field: SolverStateInternalFieldPlan {
                kind: SolverStateValueKind::Uniform,
                value_count: Some(1),
                expected_count: Some(1),
                valid_count: Some(true),
                uniform_components: Some(vec![1.0, 2.0]),
                loaded_scalars: None,
            },
            boundary_patches: 0,
            mesh_boundary_patches: Some(1),
            storage: SolverStateStoragePlan {
                cpu_capable: true,
                gpu_capable: true,
                components: Some(3),
                scalar_slots: Some(3),
                bytes_f64: Some(24),
                status: SolverStateStorageStatus::Loaded,
            },
            cpu_buffer: SolverStateCpuBufferPlan {
                materializable: false,
                scalar_slots: Some(3),
                bytes_f64: Some(24),
                status: SolverStateCpuBufferStatus::InvalidShape,
            },
        }
    }

    fn invalid_uniform_vector_field() -> FieldFile {
        FieldFile {
            path: PathBuf::from("case/0/U"),
            region: None,
            name: "U".to_string(),
            class_name: Some("volVectorField".to_string()),
            dimensions: None,
            internal_field: Some(FieldValueSummary::Uniform("(1 2)".to_string())),
            boundary_patches: Vec::new(),
        }
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
