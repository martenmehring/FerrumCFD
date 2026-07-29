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
    let cell_centres = compute_cell_centres(mesh, &face_geometry)?;
    let oriented_area_vectors = orient_face_area_vectors(mesh, &face_geometry, &cell_centres)?;
    let signed_cell_volumes =
        compute_signed_cell_volumes(mesh, &face_geometry, &cell_centres, &oriented_area_vectors)?;

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
    mesh.validate()?;
    let face_geometry = compute_face_geometry(mesh)?;
    let cell_centres = compute_cell_centres(mesh, &face_geometry)?;
    let oriented_area_vectors = orient_face_area_vectors(mesh, &face_geometry, &cell_centres)?;

    let cell_volumes =
        compute_signed_cell_volumes(mesh, &face_geometry, &cell_centres, &oriented_area_vectors)?;

    let non_positive_cell_volumes = cell_volumes.iter().filter(|volume| **volume <= 0.0).count();

    Ok(PolyMeshGeometry {
        face_centres: face_geometry
            .iter()
            .map(|face| Point3::from(face.centre))
            .collect(),
        face_area_vectors: oriented_area_vectors
            .into_iter()
            .map(Point3::from)
            .collect(),
        cell_centres: cell_centres.into_iter().map(Point3::from).collect(),
        cell_volumes: cell_volumes.into_iter().map(f64::abs).collect(),
        non_positive_cell_volumes,
    })
}

fn compute_signed_cell_volumes(
    mesh: &PolyMesh,
    face_geometry: &[FaceGeometry],
    cell_centres: &[Vec3],
    oriented_area_vectors: &[Vec3],
) -> Result<Vec<f64>> {
    let mut cell_volumes = fallible_filled(mesh.cell_count(), 0.0_f64)?;
    for (face_index, face) in face_geometry.iter().enumerate() {
        let owner = mesh.owner[face_index];
        let owner_centre = cell_centres[owner];
        cell_volumes[owner] +=
            oriented_area_vectors[face_index].dot(face.centre - owner_centre) / 3.0;

        if let Some(&neighbour) = mesh.neighbour.get(face_index) {
            let neighbour_centre = cell_centres[neighbour];
            cell_volumes[neighbour] +=
                (-oriented_area_vectors[face_index]).dot(face.centre - neighbour_centre) / 3.0;
        }
    }
    Ok(cell_volumes)
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

        result.push(FaceGeometry {
            centre: average_point(&points),
            area_vector: polygon_area_vector(&points),
        });
    }
    Ok(result)
}

fn compute_cell_centres(mesh: &PolyMesh, faces: &[FaceGeometry]) -> Result<Vec<Vec3>> {
    let mut sums = fallible_filled(mesh.cell_count(), Vec3::default())?;
    let mut counts = fallible_filled(mesh.cell_count(), 0usize)?;

    for (face_index, face) in faces.iter().enumerate() {
        let owner = mesh.owner[face_index];
        sums[owner] += face.centre;
        counts[owner] += 1;

        if let Some(&neighbour) = mesh.neighbour.get(face_index) {
            sums[neighbour] += face.centre;
            counts[neighbour] += 1;
        }
    }

    let mut centres = Vec::new();
    centres
        .try_reserve_exact(mesh.cell_count())
        .map_err(|_| MeshError::OutOfMemory)?;
    centres.extend(sums.into_iter().zip(counts).map(|(sum, count)| {
        if count == 0 {
            Vec3::default()
        } else {
            sum / count as f64
        }
    }));
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
    let sum = points
        .iter()
        .fold(Vec3::default(), |sum, point| sum + Vec3::from(*point));
    sum / points.len() as f64
}

fn polygon_area_vector(points: &[Point3]) -> Vec3 {
    let origin = Vec3::from(points[0]);
    let mut area = Vec3::default();
    for index in 1..points.len() - 1 {
        let a = Vec3::from(points[index]) - origin;
        let b = Vec3::from(points[index + 1]) - origin;
        area += a.cross(b) * 0.5;
    }
    area
}

#[derive(Clone, Copy, Debug)]
struct FaceGeometry {
    centre: Vec3,
    area_vector: Vec3,
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
        FaceGeometry, PolyMeshGeometry, Vec3, average_point, compute_poly_mesh_geometry,
        fallible_filled, polygon_area_vector, summarize_poly_mesh_geometry,
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
    fn fallible_refactor_is_bit_identical_to_original_geometry_order() {
        let mesh = PolyMesh {
            path: PathBuf::from("polyMesh"),
            points: vec![
                point(0.0, 0.0, 0.0),
                point(1.3, 0.0, 0.0),
                point(1.3, 0.7, 0.0),
                point(0.0, 0.7, 0.0),
                point(0.2, 0.1, 2.1),
                point(1.5, 0.1, 2.1),
                point(1.5, 0.8, 2.1),
                point(0.2, 0.8, 2.1),
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

        let actual = compute_poly_mesh_geometry(&mesh).expect("fallible geometry");
        let original = original_geometry_order(&mesh);

        assert_points_bit_equal(&actual.face_centres, &original.face_centres);
        assert_points_bit_equal(&actual.face_area_vectors, &original.face_area_vectors);
        assert_points_bit_equal(&actual.cell_centres, &original.cell_centres);
        assert_eq!(
            actual
                .cell_volumes
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            original
                .cell_volumes
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
        assert_eq!(
            actual.non_positive_cell_volumes,
            original.non_positive_cell_volumes
        );
    }

    fn original_geometry_order(mesh: &PolyMesh) -> PolyMeshGeometry {
        let face_geometry = mesh
            .faces
            .iter()
            .map(|face| {
                let points = face
                    .iter()
                    .map(|&index| mesh.points[index])
                    .collect::<Vec<_>>();
                FaceGeometry {
                    centre: average_point(&points),
                    area_vector: polygon_area_vector(&points),
                }
            })
            .collect::<Vec<_>>();

        let mut sums = vec![Vec3::default(); mesh.cell_count()];
        let mut counts = vec![0usize; mesh.cell_count()];
        for (face_index, face) in face_geometry.iter().enumerate() {
            let owner = mesh.owner[face_index];
            sums[owner] += face.centre;
            counts[owner] += 1;
            if let Some(&neighbour) = mesh.neighbour.get(face_index) {
                sums[neighbour] += face.centre;
                counts[neighbour] += 1;
            }
        }
        let cell_centres = sums
            .into_iter()
            .zip(counts)
            .map(|(sum, count)| {
                if count == 0 {
                    Vec3::default()
                } else {
                    sum / count as f64
                }
            })
            .collect::<Vec<_>>();

        let oriented_area_vectors = face_geometry
            .iter()
            .enumerate()
            .map(|(face_index, face)| {
                let owner = mesh.owner[face_index];
                let desired_direction = if let Some(&neighbour) = mesh.neighbour.get(face_index) {
                    cell_centres[neighbour] - cell_centres[owner]
                } else {
                    face.centre - cell_centres[owner]
                };
                if face.area_vector.dot(desired_direction) < 0.0 {
                    -face.area_vector
                } else {
                    face.area_vector
                }
            })
            .collect::<Vec<_>>();

        let mut cell_volumes = vec![0.0; mesh.cell_count()];
        for (face_index, face) in face_geometry.iter().enumerate() {
            let owner = mesh.owner[face_index];
            let owner_centre = cell_centres[owner];
            cell_volumes[owner] +=
                oriented_area_vectors[face_index].dot(face.centre - owner_centre) / 3.0;
            if let Some(&neighbour) = mesh.neighbour.get(face_index) {
                let neighbour_centre = cell_centres[neighbour];
                cell_volumes[neighbour] +=
                    (-oriented_area_vectors[face_index]).dot(face.centre - neighbour_centre) / 3.0;
            }
        }
        let non_positive_cell_volumes =
            cell_volumes.iter().filter(|volume| **volume <= 0.0).count();

        PolyMeshGeometry {
            face_centres: face_geometry
                .iter()
                .map(|face| Point3::from(face.centre))
                .collect(),
            face_area_vectors: oriented_area_vectors
                .into_iter()
                .map(Point3::from)
                .collect(),
            cell_centres: cell_centres.into_iter().map(Point3::from).collect(),
            cell_volumes: cell_volumes.into_iter().map(f64::abs).collect(),
            non_positive_cell_volumes,
        }
    }

    fn assert_points_bit_equal(left: &[Point3], right: &[Point3]) {
        assert_eq!(left.len(), right.len());
        for (left, right) in left.iter().zip(right) {
            assert_eq!(left.x.to_bits(), right.x.to_bits());
            assert_eq!(left.y.to_bits(), right.y.to_bits());
            assert_eq!(left.z.to_bits(), right.z.to_bits());
        }
    }

    fn point(x: f64, y: f64, z: f64) -> Point3 {
        Point3 { x, y, z }
    }

    fn assert_close(left: f64, right: f64) {
        assert!(
            (left - right).abs() < 1e-12,
            "expected {left} to be close to {right}"
        );
    }
}
