use std::path::{Path, PathBuf};

use crate::poly_mesh::PolyMesh;
use crate::{MeshError, Point3, Result};

#[derive(Debug)]
pub struct GeometrySummary {
    pub case_dir: PathBuf,
    pub cells: usize,
    pub faces: usize,
    pub min_face_area: f64,
    pub max_face_area: f64,
    pub total_boundary_area: f64,
    pub min_cell_volume: f64,
    pub max_cell_volume: f64,
    pub total_cell_volume: f64,
    pub non_positive_cell_volumes: usize,
}

#[derive(Clone, Debug)]
pub struct PolyMeshGeometry {
    pub face_centres: Vec<Point3>,
    pub face_area_vectors: Vec<Point3>,
    pub cell_centres: Vec<Point3>,
    pub cell_volumes: Vec<f64>,
    pub non_positive_cell_volumes: usize,
}

/// Cartesian axis occupied only by an `empty`-patch extrusion.
///
/// Selecting an axis does not change the mesh geometry. It only projects that
/// component out while measuring active edge aspect ratios, so a thin or thick
/// 2-D extrusion does not dominate the in-plane metric.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CartesianAxis {
    X,
    Y,
    Z,
}

/// Controls raw polyMesh quality measurements.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PolyMeshQualityOptions {
    pub empty_extrusion_axis: Option<CartesianAxis>,
}

/// Raw geometry measurements and the indices where a measurement is invalid.
///
/// This summary intentionally contains no product acceptance thresholds. The
/// caller can apply case-specific policy without the geometry layer imposing
/// artificial limits.
#[derive(Clone, Debug, PartialEq)]
pub struct PolyMeshQualitySummary {
    pub cells: usize,
    pub faces: usize,
    pub internal_faces: usize,
    pub max_internal_non_orthogonality_degrees: Option<f64>,
    pub max_internal_non_orthogonality_face: Option<usize>,
    pub max_normalized_internal_skewness: Option<f64>,
    pub max_normalized_internal_skewness_face: Option<usize>,
    pub max_active_edge_aspect_ratio: Option<f64>,
    pub max_active_edge_aspect_ratio_cell: Option<usize>,
    pub problematic_face_indices: Vec<usize>,
    pub problematic_cell_indices: Vec<usize>,
    pub non_finite_face_geometry_indices: Vec<usize>,
    pub non_positive_face_area_indices: Vec<usize>,
    pub non_finite_cell_geometry_indices: Vec<usize>,
    pub non_positive_cell_volume_indices: Vec<usize>,
}

pub fn summarize_case_geometry(case_dir: &Path) -> Result<GeometrySummary> {
    let mesh = PolyMesh::read(&case_dir.join("constant").join("polyMesh"))?;
    summarize_poly_mesh_geometry(case_dir, &mesh)
}

pub fn summarize_poly_mesh_geometry(case_dir: &Path, mesh: &PolyMesh) -> Result<GeometrySummary> {
    let geometry = compute_poly_mesh_geometry(mesh)?;
    let mut min_face_area = f64::INFINITY;
    let mut max_face_area = 0.0_f64;
    let mut total_boundary_area = 0.0_f64;

    for (face_index, area_vector) in geometry.face_area_vectors.iter().enumerate() {
        let area = Vec3::from(*area_vector).mag();
        min_face_area = min_face_area.min(area);
        max_face_area = max_face_area.max(area);

        if mesh.neighbour.get(face_index).is_none() {
            total_boundary_area += area;
        }
    }

    let mut min_cell_volume = f64::INFINITY;
    let mut max_cell_volume = 0.0_f64;
    let mut total_cell_volume = 0.0_f64;
    let non_positive_cell_volumes = geometry.non_positive_cell_volumes;

    for volume in geometry.cell_volumes {
        min_cell_volume = min_cell_volume.min(volume);
        max_cell_volume = max_cell_volume.max(volume);
        total_cell_volume += volume;
    }

    if mesh.faces.is_empty() {
        min_face_area = 0.0;
    }
    if mesh.cell_count() == 0 {
        min_cell_volume = 0.0;
    }

    Ok(GeometrySummary {
        case_dir: case_dir.to_path_buf(),
        cells: mesh.cell_count(),
        faces: mesh.faces.len(),
        min_face_area,
        max_face_area,
        total_boundary_area,
        min_cell_volume,
        max_cell_volume,
        total_cell_volume,
        non_positive_cell_volumes,
    })
}

/// Measures polyMesh geometry without applying case-specific quality limits.
///
/// Internal non-orthogonality is the angle between the oriented face-area
/// vector and the owner-to-neighbour centre vector. Internal skewness is the
/// distance from the face centre to that centre line's face-plane intersection,
/// normalized by the owner-to-neighbour distance. The active edge aspect ratio
/// is the longest divided by the shortest positive projected edge in each cell.
pub fn summarize_poly_mesh_quality(
    mesh: &PolyMesh,
    options: &PolyMeshQualityOptions,
) -> Result<PolyMeshQualitySummary> {
    validate_poly_mesh_for_quality(mesh)?;
    let face_geometry = compute_face_geometry(mesh)?;
    let provisional_cell_centres = compute_provisional_cell_centres(mesh, &face_geometry)?;
    let oriented_area_vectors =
        orient_face_area_vectors(mesh, &face_geometry, &provisional_cell_centres)?;
    let (cell_centres, signed_cell_volumes) = compute_cell_geometry(
        mesh,
        &face_geometry,
        &provisional_cell_centres,
        &oriented_area_vectors,
    )?;

    let mut problematic_faces = fallible_flags(mesh.faces.len())?;
    let mut non_finite_faces = fallible_flags(mesh.faces.len())?;
    let mut non_positive_face_areas = fallible_flags(mesh.faces.len())?;
    let mut problematic_cells = fallible_flags(mesh.cell_count())?;
    let mut non_finite_cells = fallible_flags(mesh.cell_count())?;
    let mut non_positive_cell_volumes = fallible_flags(mesh.cell_count())?;

    for (face_index, face) in face_geometry.iter().enumerate() {
        let area = oriented_area_vectors[face_index].stable_mag();
        if !face.centre.is_finite()
            || !oriented_area_vectors[face_index].is_finite()
            || !area.is_finite()
        {
            non_finite_faces[face_index] = true;
            mark_face_and_cells(
                mesh,
                face_index,
                &mut problematic_faces,
                &mut problematic_cells,
            );
        } else if area <= 0.0 {
            non_positive_face_areas[face_index] = true;
            mark_face_and_cells(
                mesh,
                face_index,
                &mut problematic_faces,
                &mut problematic_cells,
            );
        }
    }

    for cell_index in 0..mesh.cell_count() {
        let centre = cell_centres[cell_index];
        let volume = signed_cell_volumes[cell_index];
        if !centre.is_finite() || !volume.is_finite() {
            non_finite_cells[cell_index] = true;
            problematic_cells[cell_index] = true;
        } else if volume <= 0.0 {
            non_positive_cell_volumes[cell_index] = true;
            problematic_cells[cell_index] = true;
        }
    }

    let mut max_non_orthogonality = None;
    let mut max_non_orthogonality_face = None;
    let mut max_skewness = None;
    let mut max_skewness_face = None;

    for face_index in 0..mesh.neighbour.len() {
        let owner = mesh.owner[face_index];
        let neighbour = mesh.neighbour[face_index];
        let owner_centre = cell_centres[owner];
        let neighbour_centre = cell_centres[neighbour];
        let centre_delta = neighbour_centre - owner_centre;
        let centre_distance = centre_delta.stable_mag();
        let area_vector = oriented_area_vectors[face_index];
        let area = area_vector.stable_mag();

        if !owner_centre.is_finite()
            || !neighbour_centre.is_finite()
            || !centre_delta.is_finite()
            || !centre_distance.is_finite()
            || !area_vector.is_finite()
            || !area.is_finite()
            || centre_distance <= 0.0
            || area <= 0.0
        {
            mark_face_and_cells(
                mesh,
                face_index,
                &mut problematic_faces,
                &mut problematic_cells,
            );
            continue;
        }

        let unit_area = area_vector / area;
        let centre_direction = centre_delta / centre_distance;
        let raw_cosine = unit_area.dot(centre_direction);
        if !unit_area.is_finite() || !centre_direction.is_finite() || !raw_cosine.is_finite() {
            mark_face_and_cells(
                mesh,
                face_index,
                &mut problematic_faces,
                &mut problematic_cells,
            );
            continue;
        }

        // Clamping only compensates for round-off around the mathematical
        // cosine domain; it is not a mesh-quality cap.
        let cosine = raw_cosine.clamp(-1.0, 1.0);
        if cosine < 0.0 {
            mark_face_and_cells(
                mesh,
                face_index,
                &mut problematic_faces,
                &mut problematic_cells,
            );
            continue;
        }
        let non_orthogonality = cosine.acos().to_degrees();
        update_indexed_max(
            non_orthogonality,
            face_index,
            &mut max_non_orthogonality,
            &mut max_non_orthogonality_face,
        );
        if cosine == 0.0 {
            mark_face_and_cells(
                mesh,
                face_index,
                &mut problematic_faces,
                &mut problematic_cells,
            );
            continue;
        }

        let owner_to_face = face_geometry[face_index].centre - owner_centre;
        let plane_fraction = unit_area.dot(owner_to_face / centre_distance) / cosine;
        let line_plane_intersection = owner_centre + centre_delta * plane_fraction;
        let skew_distance =
            (face_geometry[face_index].centre - line_plane_intersection).stable_mag();
        let normalized_skewness = skew_distance / centre_distance;

        if !non_orthogonality.is_finite()
            || !plane_fraction.is_finite()
            || !line_plane_intersection.is_finite()
            || !skew_distance.is_finite()
            || !normalized_skewness.is_finite()
        {
            mark_face_and_cells(
                mesh,
                face_index,
                &mut problematic_faces,
                &mut problematic_cells,
            );
            continue;
        }

        update_indexed_max(
            normalized_skewness,
            face_index,
            &mut max_skewness,
            &mut max_skewness_face,
        );
    }

    let mut min_active_edge = fallible_filled(mesh.cell_count(), f64::INFINITY)?;
    let mut max_active_edge = fallible_filled(mesh.cell_count(), 0.0_f64)?;
    let mut active_edge_counts = fallible_filled(mesh.cell_count(), 0usize)?;

    for (face_index, point_indices) in mesh.faces.iter().enumerate() {
        for edge_index in 0..point_indices.len() {
            let start = Vec3::from(mesh.points[point_indices[edge_index]]);
            let end =
                Vec3::from(mesh.points[point_indices[(edge_index + 1) % point_indices.len()]]);
            let edge = end - start;
            let full_length = edge.stable_mag();

            if !edge.is_finite() || !full_length.is_finite() || full_length <= 0.0 {
                mark_face_and_cells(
                    mesh,
                    face_index,
                    &mut problematic_faces,
                    &mut problematic_cells,
                );
                continue;
            }

            let active_edge = edge.without_axis(options.empty_extrusion_axis);
            let active_length = active_edge.stable_mag();
            if !active_length.is_finite() {
                mark_face_and_cells(
                    mesh,
                    face_index,
                    &mut problematic_faces,
                    &mut problematic_cells,
                );
                continue;
            }

            // A positive edge parallel to an explicitly ignored extrusion
            // axis is expected and contributes no in-plane length.
            if active_length <= 0.0 {
                if options.empty_extrusion_axis.is_none() {
                    mark_face_and_cells(
                        mesh,
                        face_index,
                        &mut problematic_faces,
                        &mut problematic_cells,
                    );
                }
                continue;
            }

            update_cell_edge_range(
                mesh.owner[face_index],
                active_length,
                &mut min_active_edge,
                &mut max_active_edge,
                &mut active_edge_counts,
            );
            if let Some(&neighbour) = mesh.neighbour.get(face_index) {
                update_cell_edge_range(
                    neighbour,
                    active_length,
                    &mut min_active_edge,
                    &mut max_active_edge,
                    &mut active_edge_counts,
                );
            }
        }
    }

    let mut max_aspect_ratio = None;
    let mut max_aspect_ratio_cell = None;
    for cell_index in 0..mesh.cell_count() {
        if active_edge_counts[cell_index] == 0
            || !min_active_edge[cell_index].is_finite()
            || !max_active_edge[cell_index].is_finite()
            || min_active_edge[cell_index] <= 0.0
        {
            problematic_cells[cell_index] = true;
            continue;
        }

        let aspect_ratio = max_active_edge[cell_index] / min_active_edge[cell_index];
        if !aspect_ratio.is_finite() {
            problematic_cells[cell_index] = true;
            continue;
        }
        update_indexed_max(
            aspect_ratio,
            cell_index,
            &mut max_aspect_ratio,
            &mut max_aspect_ratio_cell,
        );
    }

    Ok(PolyMeshQualitySummary {
        cells: mesh.cell_count(),
        faces: mesh.faces.len(),
        internal_faces: mesh.neighbour.len(),
        max_internal_non_orthogonality_degrees: max_non_orthogonality,
        max_internal_non_orthogonality_face: max_non_orthogonality_face,
        max_normalized_internal_skewness: max_skewness,
        max_normalized_internal_skewness_face: max_skewness_face,
        max_active_edge_aspect_ratio: max_aspect_ratio,
        max_active_edge_aspect_ratio_cell: max_aspect_ratio_cell,
        problematic_face_indices: indices_from_flags(&problematic_faces)?,
        problematic_cell_indices: indices_from_flags(&problematic_cells)?,
        non_finite_face_geometry_indices: indices_from_flags(&non_finite_faces)?,
        non_positive_face_area_indices: indices_from_flags(&non_positive_face_areas)?,
        non_finite_cell_geometry_indices: indices_from_flags(&non_finite_cells)?,
        non_positive_cell_volume_indices: indices_from_flags(&non_positive_cell_volumes)?,
    })
}

pub fn compute_poly_mesh_geometry(mesh: &PolyMesh) -> Result<PolyMeshGeometry> {
    compute_poly_mesh_geometry_with_volume_validation(mesh, |_, _| Ok(()))
}

pub(crate) fn compute_solver_runtime_geometry(mesh: &PolyMesh) -> Result<PolyMeshGeometry> {
    let geometry =
        compute_poly_mesh_geometry_with_volume_validation(mesh, |cell_index, volume| {
            if !volume.is_finite() || volume <= 0.0 {
                return Err(MeshError::InvalidInput(format!(
                    "cell {cell_index} has invalid oriented volume {volume}"
                )));
            }
            Ok(())
        })?;

    for (face, (centre, area_vector)) in geometry
        .face_centres
        .iter()
        .zip(&geometry.face_area_vectors)
        .enumerate()
    {
        let centre = Vec3::from(*centre);
        let area_vector = Vec3::from(*area_vector);
        let area = area_vector.stable_mag();
        if !centre.is_finite() || !area_vector.is_finite() || !area.is_finite() || area <= 0.0 {
            return Err(MeshError::InvalidInput(format!(
                "face {face} has invalid area geometry"
            )));
        }
    }
    for (cell, centre) in geometry.cell_centres.iter().copied().enumerate() {
        if !Vec3::from(centre).is_finite() {
            return Err(MeshError::InvalidInput(format!(
                "cell {cell} has invalid volume centroid"
            )));
        }
    }

    Ok(geometry)
}

fn compute_poly_mesh_geometry_with_volume_validation(
    mesh: &PolyMesh,
    validate_volume: impl Fn(usize, f64) -> Result<()>,
) -> Result<PolyMeshGeometry> {
    mesh.validate()?;
    let face_geometry = compute_face_geometry(mesh)?;
    let provisional_cell_centres = compute_provisional_cell_centres(mesh, &face_geometry)?;
    let oriented_area_vectors =
        orient_face_area_vectors(mesh, &face_geometry, &provisional_cell_centres)?;
    let (cell_centres, cell_volumes) = compute_cell_geometry(
        mesh,
        &face_geometry,
        &provisional_cell_centres,
        &oriented_area_vectors,
    )?;

    for (cell_index, volume) in cell_volumes.iter().copied().enumerate() {
        validate_volume(cell_index, volume)?;
    }
    let non_positive_cell_volumes = cell_volumes.iter().filter(|volume| **volume <= 0.0).count();

    let mut face_centres = Vec::new();
    face_centres
        .try_reserve_exact(face_geometry.len())
        .map_err(|_| MeshError::OutOfMemory)?;
    let mut face_area_vectors = Vec::new();
    face_area_vectors
        .try_reserve_exact(oriented_area_vectors.len())
        .map_err(|_| MeshError::OutOfMemory)?;
    for (face, area_vector) in face_geometry.iter().zip(oriented_area_vectors) {
        face_centres.push(Point3::from(face.centre));
        face_area_vectors.push(Point3::from(area_vector));
    }

    let mut output_cell_centres = Vec::new();
    output_cell_centres
        .try_reserve_exact(cell_centres.len())
        .map_err(|_| MeshError::OutOfMemory)?;
    let mut output_cell_volumes = Vec::new();
    output_cell_volumes
        .try_reserve_exact(cell_volumes.len())
        .map_err(|_| MeshError::OutOfMemory)?;
    for (centre, volume) in cell_centres.into_iter().zip(cell_volumes) {
        output_cell_centres.push(Point3::from(centre));
        output_cell_volumes.push(volume.abs());
    }

    Ok(PolyMeshGeometry {
        face_centres,
        face_area_vectors,
        cell_centres: output_cell_centres,
        cell_volumes: output_cell_volumes,
        non_positive_cell_volumes,
    })
}

fn compute_cell_geometry(
    mesh: &PolyMesh,
    face_geometry: &[FaceGeometry],
    provisional_cell_centres: &[Vec3],
    oriented_area_vectors: &[Vec3],
) -> Result<(Vec<Vec3>, Vec<f64>)> {
    let mut accumulators = fallible_filled(mesh.cell_count(), CellMomentAccumulator::default())?;

    for (face_index, face) in face_geometry.iter().enumerate() {
        let owner = mesh.owner[face_index];
        accumulate_face_pyramid(
            face.centre,
            provisional_cell_centres[owner],
            oriented_area_vectors[face_index],
            &mut accumulators[owner],
        );

        if let Some(&neighbour) = mesh.neighbour.get(face_index) {
            accumulate_face_pyramid(
                face.centre,
                provisional_cell_centres[neighbour],
                -oriented_area_vectors[face_index],
                &mut accumulators[neighbour],
            );
        }
    }

    let mut cell_centres = Vec::new();
    cell_centres
        .try_reserve_exact(mesh.cell_count())
        .map_err(|_| MeshError::OutOfMemory)?;
    let mut cell_volumes = Vec::new();
    cell_volumes
        .try_reserve_exact(mesh.cell_count())
        .map_err(|_| MeshError::OutOfMemory)?;
    for (reference, accumulator) in provisional_cell_centres.iter().copied().zip(accumulators) {
        let pyramid_volume = accumulator.pyramid_volume.value();
        let volume = pyramid_volume / 3.0;
        let centre = if pyramid_volume == 0.0 {
            reference
        } else {
            reference + accumulator.relative_moment.value() / pyramid_volume
        };
        cell_centres.push(centre);
        cell_volumes.push(volume);
    }
    Ok((cell_centres, cell_volumes))
}

fn accumulate_face_pyramid(
    face_centre: Vec3,
    cell_reference: Vec3,
    outward_area_vector: Vec3,
    accumulator: &mut CellMomentAccumulator,
) {
    let centre_offset = face_centre - cell_reference;
    let pyramid_volume = outward_area_vector.dot(centre_offset);
    accumulator.pyramid_volume.add(pyramid_volume);
    accumulator
        .relative_moment
        .add_scaled(centre_offset, 0.75 * pyramid_volume);
}

fn fallible_flags(len: usize) -> Result<Vec<bool>> {
    fallible_filled(len, false)
}

fn fallible_filled<T: Clone>(len: usize, value: T) -> Result<Vec<T>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|_| MeshError::OutOfMemory)?;
    values.resize(len, value);
    Ok(values)
}

fn indices_from_flags(flags: &[bool]) -> Result<Vec<usize>> {
    let count = flags.iter().filter(|flag| **flag).count();
    let mut indices = Vec::new();
    indices
        .try_reserve_exact(count)
        .map_err(|_| MeshError::OutOfMemory)?;
    indices.extend(
        flags
            .iter()
            .enumerate()
            .filter_map(|(index, flag)| flag.then_some(index)),
    );
    Ok(indices)
}

fn mark_face_and_cells(
    mesh: &PolyMesh,
    face_index: usize,
    problematic_faces: &mut [bool],
    problematic_cells: &mut [bool],
) {
    problematic_faces[face_index] = true;
    problematic_cells[mesh.owner[face_index]] = true;
    if let Some(&neighbour) = mesh.neighbour.get(face_index) {
        problematic_cells[neighbour] = true;
    }
}

fn update_cell_edge_range(
    cell_index: usize,
    length: f64,
    minimum: &mut [f64],
    maximum: &mut [f64],
    counts: &mut [usize],
) {
    minimum[cell_index] = minimum[cell_index].min(length);
    maximum[cell_index] = maximum[cell_index].max(length);
    counts[cell_index] += 1;
}

fn update_indexed_max(
    value: f64,
    index: usize,
    maximum: &mut Option<f64>,
    maximum_index: &mut Option<usize>,
) {
    if maximum.is_none_or(|current| value > current) {
        *maximum = Some(value);
        *maximum_index = Some(index);
    }
}

fn validate_poly_mesh_for_quality(mesh: &PolyMesh) -> Result<()> {
    if mesh.faces.len() != mesh.owner.len() {
        return Err(MeshError::InvalidInput(format!(
            "faces/owner size mismatch in {}",
            mesh.path.display()
        )));
    }
    if mesh.neighbour.len() > mesh.faces.len() {
        return Err(MeshError::InvalidInput(format!(
            "neighbour list is longer than face list in {}",
            mesh.path.display()
        )));
    }

    if let Some(max_cell) = mesh.owner.iter().chain(&mesh.neighbour).copied().max() {
        let cell_count = max_cell.checked_add(1).ok_or_else(|| {
            MeshError::InvalidInput(format!(
                "cell label {max_cell} overflows the cell count in {}",
                mesh.path.display()
            ))
        })?;
        if cell_count > mesh.faces.len() {
            return Err(MeshError::InvalidInput(format!(
                "cell labels in {} imply {cell_count} cells from only {} faces; labels must be dense and bounded by the mesh topology",
                mesh.path.display(),
                mesh.faces.len()
            )));
        }

        let mut seen = fallible_flags(cell_count)?;
        for &cell in mesh.owner.iter().chain(&mesh.neighbour) {
            seen[cell] = true;
        }
        if let Some(missing) = seen.iter().position(|present| !present) {
            return Err(MeshError::InvalidInput(format!(
                "cell labels in {} are sparse; missing cell label {missing}",
                mesh.path.display()
            )));
        }
    }

    let internal_faces = mesh.neighbour.len();
    let mut claimed = fallible_flags(mesh.faces.len() - internal_faces)?;
    for patch in &mesh.patches {
        let end_face = patch.start_face.checked_add(patch.faces).ok_or_else(|| {
            MeshError::InvalidInput(format!(
                "patch '{}' face range overflows in {}",
                patch.name,
                mesh.path.display()
            ))
        })?;
        if patch.start_face < internal_faces || end_face > mesh.faces.len() {
            return Err(MeshError::InvalidInput(format!(
                "patch '{}' range startFace={} nFaces={} is outside boundary face range {}..{} in {}",
                patch.name,
                patch.start_face,
                patch.faces,
                internal_faces,
                mesh.faces.len(),
                mesh.path.display()
            )));
        }
        for face in patch.start_face..end_face {
            let slot = &mut claimed[face - internal_faces];
            if *slot {
                return Err(MeshError::InvalidInput(format!(
                    "boundary face {face} belongs to more than one patch in {}",
                    mesh.path.display()
                )));
            }
            *slot = true;
        }
    }
    Ok(())
}

fn compute_face_geometry(mesh: &PolyMesh) -> Result<Vec<FaceGeometry>> {
    let mut result = Vec::new();
    result
        .try_reserve_exact(mesh.faces.len())
        .map_err(|_| MeshError::OutOfMemory)?;
    let mut points = Vec::new();

    for face in &mesh.faces {
        if face.len() < 3 {
            return Err(MeshError::InvalidInput(format!(
                "face with {} nodes found in {}",
                face.len(),
                mesh.path.display()
            )));
        }

        points.clear();
        if face.len() > points.capacity() {
            points
                .try_reserve_exact(face.len())
                .map_err(|_| MeshError::OutOfMemory)?;
        }
        for &index in face {
            points.push(mesh.points.get(index).copied().ok_or_else(|| {
                MeshError::InvalidInput(format!(
                    "face references missing point {} in {}",
                    index,
                    mesh.path.display()
                ))
            })?);
        }

        result.push(polygon_face_geometry(&points));
    }
    Ok(result)
}

fn compute_provisional_cell_centres(mesh: &PolyMesh, faces: &[FaceGeometry]) -> Result<Vec<Vec3>> {
    let mut sums = fallible_filled(mesh.cell_count(), CompensatedVec3::default())?;
    let mut counts = fallible_filled(mesh.cell_count(), 0usize)?;

    for (face_index, face) in faces.iter().enumerate() {
        let owner = mesh.owner[face_index];
        sums[owner].add(face.centre);
        counts[owner] += 1;

        if let Some(&neighbour) = mesh.neighbour.get(face_index) {
            sums[neighbour].add(face.centre);
            counts[neighbour] += 1;
        }
    }

    let mut centres = Vec::new();
    centres
        .try_reserve_exact(mesh.cell_count())
        .map_err(|_| MeshError::OutOfMemory)?;
    for (sum, count) in sums.into_iter().zip(counts) {
        centres.push(if count == 0 {
            Vec3::default()
        } else {
            sum.value() / count as f64
        });
    }
    Ok(centres)
}

fn orient_face_area_vectors(
    mesh: &PolyMesh,
    faces: &[FaceGeometry],
    cell_centres: &[Vec3],
) -> Result<Vec<Vec3>> {
    let mut oriented = Vec::new();
    oriented
        .try_reserve_exact(faces.len())
        .map_err(|_| MeshError::OutOfMemory)?;
    for (face_index, face) in faces.iter().enumerate() {
        let owner = mesh.owner[face_index];
        let desired_direction = if let Some(&neighbour) = mesh.neighbour.get(face_index) {
            cell_centres[neighbour] - cell_centres[owner]
        } else {
            face.centre - cell_centres[owner]
        };

        oriented.push(if face.area_vector.dot(desired_direction) < 0.0 {
            -face.area_vector
        } else {
            face.area_vector
        });
    }
    Ok(oriented)
}

fn average_point(points: &[Point3]) -> Vec3 {
    let reference = Vec3::from(points[0]);
    let mut relative_sum = CompensatedVec3::default();
    for point in points.iter().copied().skip(1) {
        relative_sum.add(Vec3::from(point) - reference);
    }
    reference + relative_sum.value() / points.len() as f64
}

fn polygon_face_geometry(points: &[Point3]) -> FaceGeometry {
    let reference = average_point(points);
    let mut area_vector = CompensatedVec3::default();
    for edge in 0..points.len() {
        let first = Vec3::from(points[edge]) - reference;
        let second = Vec3::from(points[(edge + 1) % points.len()]) - reference;
        let triangle_area_vector = first.cross(second) * 0.5;
        area_vector.add(triangle_area_vector);
    }

    let area_vector = area_vector.value();
    let area = area_vector.stable_mag();
    let centre = if area == 0.0 {
        reference
    } else {
        let unit_normal = area_vector / area;
        let mut projected_area = CompensatedScalar::default();
        let mut relative_first_moment = CompensatedVec3::default();
        for edge in 0..points.len() {
            let first = Vec3::from(points[edge]) - reference;
            let second = Vec3::from(points[(edge + 1) % points.len()]) - reference;
            let triangle_area_vector = first.cross(second) * 0.5;
            let weight = triangle_area_vector.dot(unit_normal);
            projected_area.add(weight);
            relative_first_moment.add_scaled((first + second) / 3.0, weight);
        }
        let projected_area = projected_area.value();
        if projected_area == 0.0 {
            reference
        } else {
            reference + relative_first_moment.value() / projected_area
        }
    };

    FaceGeometry {
        centre,
        area_vector,
    }
}

#[derive(Clone, Copy, Debug)]
struct FaceGeometry {
    centre: Vec3,
    area_vector: Vec3,
}

#[derive(Clone, Copy, Debug, Default)]
struct CellMomentAccumulator {
    pyramid_volume: CompensatedScalar,
    relative_moment: CompensatedVec3,
}

#[derive(Clone, Copy, Debug, Default)]
struct CompensatedScalar {
    sum: f64,
    correction: f64,
}

impl CompensatedScalar {
    fn add(&mut self, value: f64) {
        let next = self.sum + value;
        let correction = if self.sum.abs() >= value.abs() {
            (self.sum - next) + value
        } else {
            (value - next) + self.sum
        };
        self.sum = next;
        self.correction += correction;
    }

    fn value(self) -> f64 {
        let value = self.sum + self.correction;
        if value == 0.0 { 0.0 } else { value }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct CompensatedVec3 {
    x: CompensatedScalar,
    y: CompensatedScalar,
    z: CompensatedScalar,
}

impl CompensatedVec3 {
    fn add(&mut self, value: Vec3) {
        self.x.add(value.x);
        self.y.add(value.y);
        self.z.add(value.z);
    }

    fn add_scaled(&mut self, value: Vec3, scale: f64) {
        self.add(value * scale);
    }

    fn value(self) -> Vec3 {
        Vec3 {
            x: self.x.value(),
            y: self.y.value(),
            z: self.z.value(),
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct Vec3 {
    x: f64,
    y: f64,
    z: f64,
}

impl Vec3 {
    fn dot(self, rhs: Self) -> f64 {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }

    fn cross(self, rhs: Self) -> Self {
        Self {
            x: self.y * rhs.z - self.z * rhs.y,
            y: self.z * rhs.x - self.x * rhs.z,
            z: self.x * rhs.y - self.y * rhs.x,
        }
    }

    fn mag(self) -> f64 {
        self.dot(self).sqrt()
    }

    fn stable_mag(self) -> f64 {
        self.x.hypot(self.y).hypot(self.z)
    }

    fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
    }

    fn without_axis(self, axis: Option<CartesianAxis>) -> Self {
        match axis {
            Some(CartesianAxis::X) => Self { x: 0.0, ..self },
            Some(CartesianAxis::Y) => Self { y: 0.0, ..self },
            Some(CartesianAxis::Z) => Self { z: 0.0, ..self },
            None => self,
        }
    }
}

impl From<Point3> for Vec3 {
    fn from(value: Point3) -> Self {
        Self {
            x: value.x,
            y: value.y,
            z: value.z,
        }
    }
}

impl From<Vec3> for Point3 {
    fn from(value: Vec3) -> Self {
        Self {
            x: value.x,
            y: value.y,
            z: value.z,
        }
    }
}

impl std::ops::Add for Vec3 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

impl std::ops::AddAssign for Vec3 {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl std::ops::Div<f64> for Vec3 {
    type Output = Self;

    fn div(self, rhs: f64) -> Self::Output {
        Self {
            x: self.x / rhs,
            y: self.y / rhs,
            z: self.z / rhs,
        }
    }
}

impl std::ops::Mul<f64> for Vec3 {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self::Output {
        Self {
            x: self.x * rhs,
            y: self.y * rhs,
            z: self.z * rhs,
        }
    }
}

impl std::ops::Neg for Vec3 {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

impl std::ops::Sub for Vec3 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use crate::poly_mesh::PolyMesh;
    use crate::{MeshError, Point3};

    use super::{
        compute_solver_runtime_geometry, fallible_filled, polygon_face_geometry,
        summarize_poly_mesh_geometry,
    };

    #[test]
    fn computes_unit_cube_geometry() {
        let mesh = PolyMesh {
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
            patches: Vec::new(),
        };

        let summary = summarize_poly_mesh_geometry(Path::new("case"), &mesh).unwrap();
        assert_eq!(summary.cells, 1);
        assert_eq!(summary.faces, 6);
        assert_close(summary.min_face_area, 1.0);
        assert_close(summary.max_face_area, 1.0);
        assert_close(summary.total_boundary_area, 6.0);
        assert_close(summary.total_cell_volume, 1.0);
        assert_eq!(summary.non_positive_cell_volumes, 0);
    }

    #[test]
    fn fallible_geometry_storage_reports_capacity_overflow_as_out_of_memory() {
        assert!(matches!(
            fallible_filled(usize::MAX, 0_u8),
            Err(MeshError::OutOfMemory)
        ));
    }

    #[test]
    fn polygon_face_centre_is_area_weighted_for_a_trapezoid() {
        let points = [
            point(0.0, 0.0, 0.0),
            point(2.0, 0.0, 0.0),
            point(1.0, 1.0, 0.0),
            point(0.0, 1.0, 0.0),
        ];
        let face = polygon_face_geometry(&points);
        let reversed = polygon_face_geometry(&points.into_iter().rev().collect::<Vec<_>>());

        assert_close(face.centre.x, 7.0 / 9.0);
        assert_close(face.centre.y, 4.0 / 9.0);
        assert_close(face.centre.z, 0.0);
        assert_close(face.area_vector.stable_mag(), 1.5);
        assert_close(reversed.centre.x, face.centre.x);
        assert_close(reversed.centre.y, face.centre.y);
        assert_close(reversed.centre.z, face.centre.z);
        assert_close(reversed.area_vector.x, -face.area_vector.x);
        assert_close(reversed.area_vector.y, -face.area_vector.y);
        assert_close(reversed.area_vector.z, -face.area_vector.z);
    }

    #[test]
    fn concave_polygon_uses_projected_signed_area_weights() {
        let points = [
            point(0.0, 0.0, 0.0),
            point(2.0, 0.0, 0.0),
            point(2.0, 1.0, 0.0),
            point(1.0, 1.0, 0.0),
            point(1.0, 2.0, 0.0),
            point(0.0, 2.0, 0.0),
        ];
        let face = polygon_face_geometry(&points);
        let reversed = polygon_face_geometry(&points.into_iter().rev().collect::<Vec<_>>());

        assert_close(face.centre.x, 5.0 / 6.0);
        assert_close(face.centre.y, 5.0 / 6.0);
        assert_close(face.centre.z, 0.0);
        assert_close(face.area_vector.stable_mag(), 3.0);
        assert_close(reversed.centre.x, face.centre.x);
        assert_close(reversed.centre.y, face.centre.y);
        assert_close(reversed.centre.z, face.centre.z);
        assert_close(reversed.area_vector.z, -face.area_vector.z);
    }

    #[test]
    fn extruded_trapezoid_matches_area_and_volume_centroid_oracles() {
        let geometry =
            compute_solver_runtime_geometry(&extruded_trapezoid()).expect("trapezoid prism");
        let centre = geometry.cell_centres[0];

        assert_close(centre.x, 7.0 / 9.0);
        assert_close(centre.y, 4.0 / 9.0);
        assert_close(centre.z, 1.5);
        assert_close(geometry.cell_volumes[0], 4.5);
    }

    #[test]
    fn affine_hexahedron_matches_determinant_and_centroid_oracles() {
        let mesh = PolyMesh {
            path: PathBuf::from("affine-hexahedron/polyMesh"),
            points: vec![
                point(0.2, -0.4, 0.7),
                point(1.5, -0.3, 0.9),
                point(1.3, 0.6, 1.05),
                point(0.0, 0.5, 0.85),
                point(0.45, -0.5, 2.4),
                point(1.75, -0.4, 2.6),
                point(1.55, 0.5, 2.75),
                point(0.25, 0.4, 2.55),
            ],
            faces: hexahedron_faces(),
            owner: vec![0; 6],
            neighbour: Vec::new(),
            patches: Vec::new(),
        };
        let geometry = compute_solver_runtime_geometry(&mesh).expect("affine hexahedron");
        let centre = geometry.cell_centres[0];

        assert_close(centre.x, 0.875);
        assert_close(centre.y, 0.05);
        assert_close(centre.z, 1.725);
        assert_close(geometry.cell_volumes[0], 2.00525);
    }

    #[test]
    fn annular_wedge_prism_matches_closed_centroid_oracle() {
        let mesh = annular_wedge_cell();
        let geometry = compute_solver_runtime_geometry(&mesh).expect("wedge geometry");
        let centre = geometry.cell_centres[0];

        assert_close(centre.x, 0.0625);
        assert_close(centre.y, 0.564_277_236_263_827);
        assert_close(centre.z, 0.0);
        let theta = 2.5_f64.to_radians();
        let expected_volume =
            0.125 * 0.5 * (0.625_f64.powi(2) - 0.5_f64.powi(2)) * (2.0 * theta).sin();
        assert_close(geometry.cell_volumes[0], expected_volume);
    }

    #[test]
    fn wedge_centroid_is_translation_scale_and_face_order_invariant() {
        let baseline = compute_solver_runtime_geometry(&annular_wedge_cell())
            .expect("baseline wedge geometry");

        let mut transformed = annular_wedge_cell();
        for point in &mut transformed.points {
            point.x = 1_000_000.0 + 8.0 * point.x;
            point.y = -2_000_000.0 + 8.0 * point.y;
            point.z = 3_000_000.0 + 8.0 * point.z;
        }
        transformed.faces.reverse();
        transformed.owner.reverse();
        for face in &mut transformed.faces {
            face.reverse();
            face.rotate_left(1);
        }
        let actual =
            compute_solver_runtime_geometry(&transformed).expect("transformed wedge geometry");

        assert_translation_close(
            actual.cell_centres[0].x,
            1_000_000.0 + 8.0 * baseline.cell_centres[0].x,
        );
        assert_translation_close(
            actual.cell_centres[0].y,
            -2_000_000.0 + 8.0 * baseline.cell_centres[0].y,
        );
        assert_translation_close(
            actual.cell_centres[0].z,
            3_000_000.0 + 8.0 * baseline.cell_centres[0].z,
        );
        // Adding million-scale offsets quantizes the stored input coordinates
        // before geometry is reconstructed. Keep the volume check tighter than
        // that input-conditioned round-off, rather than below one input ULP.
        assert_relative_close(
            actual.cell_volumes[0],
            baseline.cell_volumes[0] * 8.0_f64.powi(3),
            2e-9,
        );
    }

    #[test]
    fn solver_geometry_rejects_a_flat_zero_volume_cell() {
        let mut mesh = annular_wedge_cell();
        for point in &mut mesh.points {
            point.x = 0.0;
        }
        assert!(matches!(
            compute_solver_runtime_geometry(&mesh),
            Err(MeshError::InvalidInput(message))
                if message.contains("invalid oriented volume")
        ));
    }

    fn annular_wedge_cell() -> PolyMesh {
        let theta = 2.5_f64.to_radians();
        let c = theta.cos();
        let s = theta.sin();
        let x0 = 0.0;
        let x1 = 0.125;
        let r0 = 0.5;
        let r1 = 0.625;
        PolyMesh {
            path: PathBuf::from("annular-wedge/polyMesh"),
            points: vec![
                point(x0, r0 * c, -r0 * s),
                point(x1, r0 * c, -r0 * s),
                point(x1, r1 * c, -r1 * s),
                point(x0, r1 * c, -r1 * s),
                point(x0, r0 * c, r0 * s),
                point(x1, r0 * c, r0 * s),
                point(x1, r1 * c, r1 * s),
                point(x0, r1 * c, r1 * s),
            ],
            faces: vec![
                vec![0, 4, 7, 3],
                vec![1, 2, 6, 5],
                vec![0, 1, 5, 4],
                vec![3, 7, 6, 2],
                vec![0, 3, 2, 1],
                vec![4, 5, 6, 7],
            ],
            owner: vec![0; 6],
            neighbour: Vec::new(),
            patches: Vec::new(),
        }
    }

    fn extruded_trapezoid() -> PolyMesh {
        PolyMesh {
            path: PathBuf::from("extruded-trapezoid/polyMesh"),
            points: vec![
                point(0.0, 0.0, 0.0),
                point(2.0, 0.0, 0.0),
                point(1.0, 1.0, 0.0),
                point(0.0, 1.0, 0.0),
                point(0.0, 0.0, 3.0),
                point(2.0, 0.0, 3.0),
                point(1.0, 1.0, 3.0),
                point(0.0, 1.0, 3.0),
            ],
            faces: hexahedron_faces(),
            owner: vec![0; 6],
            neighbour: Vec::new(),
            patches: Vec::new(),
        }
    }

    fn hexahedron_faces() -> Vec<Vec<usize>> {
        vec![
            vec![0, 3, 2, 1],
            vec![4, 5, 6, 7],
            vec![0, 1, 5, 4],
            vec![1, 2, 6, 5],
            vec![2, 3, 7, 6],
            vec![3, 0, 4, 7],
        ]
    }

    fn point(x: f64, y: f64, z: f64) -> Point3 {
        Point3 { x, y, z }
    }

    fn assert_close(left: f64, right: f64) {
        assert_close_scaled(left, right, 1e-12);
    }

    fn assert_close_scaled(left: f64, right: f64, tolerance: f64) {
        assert!(
            (left - right).abs() <= tolerance,
            "expected {left:.17e} to be within {tolerance:.3e} of {right:.17e}"
        );
    }

    fn assert_translation_close(left: f64, right: f64) {
        let tolerance = 2.0 * f64::EPSILON * right.abs().max(1.0);
        assert_close_scaled(left, right, tolerance);
    }

    fn assert_relative_close(left: f64, right: f64, relative_tolerance: f64) {
        assert_close_scaled(
            left,
            right,
            relative_tolerance * left.abs().max(right.abs()).max(f64::MIN_POSITIVE),
        );
    }
}
