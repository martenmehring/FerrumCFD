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

pub fn compute_poly_mesh_geometry(mesh: &PolyMesh) -> Result<PolyMeshGeometry> {
    let face_geometry = compute_face_geometry(mesh)?;
    let cell_centres = compute_cell_centres(mesh, &face_geometry);
    let oriented_area_vectors = orient_face_area_vectors(mesh, &face_geometry, &cell_centres);

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

fn compute_face_geometry(mesh: &PolyMesh) -> Result<Vec<FaceGeometry>> {
    mesh.faces
        .iter()
        .map(|face| {
            if face.len() < 3 {
                return Err(MeshError::InvalidInput(format!(
                    "face with {} nodes found in {}",
                    face.len(),
                    mesh.path.display()
                )));
            }

            let points = face
                .iter()
                .map(|&index| {
                    mesh.points.get(index).copied().ok_or_else(|| {
                        MeshError::InvalidInput(format!(
                            "face references missing point {} in {}",
                            index,
                            mesh.path.display()
                        ))
                    })
                })
                .collect::<Result<Vec<_>>>()?;

            Ok(FaceGeometry {
                centre: average_point(&points),
                area_vector: polygon_area_vector(&points),
            })
        })
        .collect()
}

fn compute_cell_centres(mesh: &PolyMesh, faces: &[FaceGeometry]) -> Vec<Vec3> {
    let mut sums = vec![Vec3::default(); mesh.cell_count()];
    let mut counts = vec![0usize; mesh.cell_count()];

    for (face_index, face) in faces.iter().enumerate() {
        let owner = mesh.owner[face_index];
        sums[owner] += face.centre;
        counts[owner] += 1;

        if let Some(&neighbour) = mesh.neighbour.get(face_index) {
            sums[neighbour] += face.centre;
            counts[neighbour] += 1;
        }
    }

    sums.into_iter()
        .zip(counts)
        .map(|(sum, count)| {
            if count == 0 {
                Vec3::default()
            } else {
                sum / count as f64
            }
        })
        .collect()
}

fn orient_face_area_vectors(
    mesh: &PolyMesh,
    faces: &[FaceGeometry],
    cell_centres: &[Vec3],
) -> Vec<Vec3> {
    faces
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
        .collect()
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

    use crate::Point3;
    use crate::poly_mesh::PolyMesh;

    use super::summarize_poly_mesh_geometry;

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

    fn assert_close(left: f64, right: f64) {
        assert!(
            (left - right).abs() < 1e-12,
            "expected {left} to be close to {right}"
        );
    }
}
