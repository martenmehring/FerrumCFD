use std::cell::OnceCell;

use crate::runtime::SolverRuntimeMeshData;
use crate::{MeshError, Result};

use super::invalid_input;

const NO_NEIGHBOUR: usize = usize::MAX;

/// Immutable, cache-friendly SIMPLE face addressing.
///
/// The eager arrays retain the runtime mesh's global face order. Cell
/// adjacency is initialized fallibly only when a nonzero cell-limited
/// gradient first consumes it.
#[derive(Debug)]
pub(super) struct CompactSimpleFaceAddressing {
    cells: usize,
    owner_by_face: Box<[usize]>,
    neighbour_by_face: Box<[usize]>,
    internal_face_indices: Box<[usize]>,
    boundary_face_indices: Box<[usize]>,
    cell_face_adjacency: OnceCell<CompactCellFaceAdjacency>,
    first_self_neighbour: Option<(usize, usize)>,
}

#[derive(Debug)]
pub(super) struct CompactCellFaceAdjacency {
    cell_face_start: Box<[usize]>,
    cell_faces: Box<[usize]>,
}

impl CompactSimpleFaceAddressing {
    pub(super) fn from_mesh(mesh: &SolverRuntimeMeshData) -> Result<Self> {
        if mesh.owner.len() != mesh.faces || mesh.neighbour.len() != mesh.faces {
            return Err(invalid_input(format!(
                "compact face addressing requires {} owner and neighbour entries, got {} and {}",
                mesh.faces,
                mesh.owner.len(),
                mesh.neighbour.len()
            )));
        }

        let mut owner_by_face = try_vec_with_capacity(mesh.faces)?;
        let mut neighbour_by_face = try_vec_with_capacity(mesh.faces)?;
        let mut internal_face_count = 0usize;
        let mut boundary_face_count = 0usize;
        let mut first_self_neighbour = None;

        for face_index in 0..mesh.faces {
            let owner = mesh.owner[face_index];
            if owner >= mesh.cells {
                return Err(invalid_input(format!(
                    "face {face_index} owner cell {owner} is outside cell range 0..{}",
                    mesh.cells
                )));
            }
            owner_by_face.push(owner);

            if let Some(neighbour) = mesh.neighbour[face_index] {
                if neighbour >= mesh.cells {
                    return Err(invalid_input(format!(
                        "face {face_index} neighbour cell {neighbour} is outside cell range 0..{}",
                        mesh.cells
                    )));
                }
                internal_face_count = internal_face_count
                    .checked_add(1)
                    .ok_or(MeshError::OutOfMemory)?;
                if neighbour == owner && first_self_neighbour.is_none() {
                    first_self_neighbour = Some((face_index, owner));
                }
                neighbour_by_face.push(neighbour);
            } else {
                boundary_face_count = boundary_face_count
                    .checked_add(1)
                    .ok_or(MeshError::OutOfMemory)?;
                neighbour_by_face.push(NO_NEIGHBOUR);
            }
        }

        let mut internal_face_indices = try_vec_with_capacity(internal_face_count)?;
        let mut boundary_face_indices = try_vec_with_capacity(boundary_face_count)?;
        for (face_index, &neighbour) in neighbour_by_face.iter().enumerate() {
            if neighbour == NO_NEIGHBOUR {
                boundary_face_indices.push(face_index);
            } else {
                internal_face_indices.push(face_index);
            }
        }

        Ok(Self {
            cells: mesh.cells,
            owner_by_face: owner_by_face.into_boxed_slice(),
            neighbour_by_face: neighbour_by_face.into_boxed_slice(),
            internal_face_indices: internal_face_indices.into_boxed_slice(),
            boundary_face_indices: boundary_face_indices.into_boxed_slice(),
            cell_face_adjacency: OnceCell::new(),
            first_self_neighbour,
        })
    }

    #[inline]
    pub(super) fn faces(&self) -> usize {
        self.owner_by_face.len()
    }

    #[inline]
    pub(super) fn owner(&self, face_index: usize) -> usize {
        self.owner_by_face[face_index]
    }

    #[inline]
    pub(super) fn neighbour(&self, face_index: usize) -> Option<usize> {
        let neighbour = self.neighbour_by_face[face_index];
        (neighbour != NO_NEIGHBOUR).then_some(neighbour)
    }

    #[inline]
    pub(super) fn internal_faces(&self) -> &[usize] {
        &self.internal_face_indices
    }

    #[inline]
    pub(super) fn boundary_faces(&self) -> &[usize] {
        &self.boundary_face_indices
    }

    #[inline]
    pub(super) fn limiter_cell_adjacency(&self) -> Result<&CompactCellFaceAdjacency> {
        self.initialize_limiter_adjacency_with(|| {
            CompactCellFaceAdjacency::from_addressing(
                self.cells,
                &self.owner_by_face,
                &self.neighbour_by_face,
                self.first_self_neighbour,
            )
        })
    }

    fn initialize_limiter_adjacency_with(
        &self,
        build: impl FnOnce() -> Result<CompactCellFaceAdjacency>,
    ) -> Result<&CompactCellFaceAdjacency> {
        if self.cell_face_adjacency.get().is_none() {
            let adjacency = build()?;
            self.cell_face_adjacency
                .set(adjacency)
                .map_err(|_| MeshError::OutOfMemory)?;
        }
        self.cell_face_adjacency.get().ok_or(MeshError::OutOfMemory)
    }

    #[cfg(test)]
    pub(super) fn storage_identity(&self) -> [usize; 4] {
        [
            self.owner_by_face.as_ptr() as usize,
            self.neighbour_by_face.as_ptr() as usize,
            self.internal_face_indices.as_ptr() as usize,
            self.boundary_face_indices.as_ptr() as usize,
        ]
    }

    #[cfg(test)]
    pub(super) fn limiter_adjacency_initialized(&self) -> bool {
        self.cell_face_adjacency.get().is_some()
    }

    #[cfg(test)]
    pub(super) fn inject_limiter_allocation_failure(&self) -> Result<()> {
        self.initialize_limiter_adjacency_with(|| Err(MeshError::OutOfMemory))
            .map(|_| ())
    }
}

impl CompactCellFaceAdjacency {
    fn from_addressing(
        cells: usize,
        owner_by_face: &[usize],
        neighbour_by_face: &[usize],
        first_self_neighbour: Option<(usize, usize)>,
    ) -> Result<Self> {
        if let Some((face_index, owner)) = first_self_neighbour {
            return Err(invalid_input(format!(
                "face {face_index} has identical owner and neighbour cell {owner}"
            )));
        }

        let mut cell_face_counts = try_zeroed_vec(cells)?;
        let mut incidence_count = 0usize;
        for face_index in 0..owner_by_face.len() {
            let owner = owner_by_face[face_index];
            cell_face_counts[owner] = cell_face_counts[owner]
                .checked_add(1)
                .ok_or(MeshError::OutOfMemory)?;
            incidence_count = incidence_count
                .checked_add(1)
                .ok_or(MeshError::OutOfMemory)?;
            let neighbour = neighbour_by_face[face_index];
            if neighbour != NO_NEIGHBOUR {
                cell_face_counts[neighbour] = cell_face_counts[neighbour]
                    .checked_add(1)
                    .ok_or(MeshError::OutOfMemory)?;
                incidence_count = incidence_count
                    .checked_add(1)
                    .ok_or(MeshError::OutOfMemory)?;
            }
        }

        let start_count = cells.checked_add(1).ok_or(MeshError::OutOfMemory)?;
        let mut cell_face_start = try_zeroed_vec(start_count)?;
        for cell in 0..cells {
            cell_face_start[cell + 1] = cell_face_start[cell]
                .checked_add(cell_face_counts[cell])
                .ok_or(MeshError::OutOfMemory)?;
        }
        if cell_face_start[cells] != incidence_count {
            return Err(MeshError::OutOfMemory);
        }

        let mut cell_faces = try_zeroed_vec(incidence_count)?;
        let mut cell_face_write = try_zeroed_vec(cells)?;
        for face_index in 0..owner_by_face.len() {
            write_cell_face(
                &cell_face_start,
                &mut cell_face_write,
                &mut cell_faces,
                owner_by_face[face_index],
                face_index,
            )?;
            let neighbour = neighbour_by_face[face_index];
            if neighbour != NO_NEIGHBOUR {
                write_cell_face(
                    &cell_face_start,
                    &mut cell_face_write,
                    &mut cell_faces,
                    neighbour,
                    face_index,
                )?;
            }
        }

        Ok(Self {
            cell_face_start: cell_face_start.into_boxed_slice(),
            cell_faces: cell_faces.into_boxed_slice(),
        })
    }

    #[inline]
    pub(super) fn cell_faces(&self, cell: usize) -> &[usize] {
        &self.cell_faces[self.cell_face_start[cell]..self.cell_face_start[cell + 1]]
    }

    #[cfg(test)]
    pub(super) fn storage_identity(&self) -> [usize; 2] {
        [
            self.cell_face_start.as_ptr() as usize,
            self.cell_faces.as_ptr() as usize,
        ]
    }
}

fn try_vec_with_capacity<T>(capacity: usize) -> Result<Vec<T>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(capacity)
        .map_err(|_| MeshError::OutOfMemory)?;
    Ok(values)
}

fn try_zeroed_vec(length: usize) -> Result<Vec<usize>> {
    let mut values = try_vec_with_capacity(length)?;
    values.resize(length, 0);
    Ok(values)
}

fn write_cell_face(
    cell_face_start: &[usize],
    cell_face_write: &mut [usize],
    cell_faces: &mut [usize],
    cell: usize,
    face_index: usize,
) -> Result<()> {
    let destination = cell_face_start[cell]
        .checked_add(cell_face_write[cell])
        .ok_or(MeshError::OutOfMemory)?;
    let slot = cell_faces
        .get_mut(destination)
        .ok_or(MeshError::OutOfMemory)?;
    *slot = face_index;
    cell_face_write[cell] = cell_face_write[cell]
        .checked_add(1)
        .ok_or(MeshError::OutOfMemory)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::Point3;
    use crate::runtime::SolverRuntimeMeshData;

    use super::CompactSimpleFaceAddressing;

    fn interleaved_mesh() -> SolverRuntimeMeshData {
        let point = Point3 {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        };
        SolverRuntimeMeshData {
            points: 0,
            cells: 3,
            internal_faces: 2,
            boundary_faces: 2,
            faces: 4,
            owner: vec![0, 2, 1, 2],
            neighbour: vec![Some(1), None, Some(2), None],
            cell_centres: vec![point; 3],
            face_centres: vec![point; 4],
            face_area_vectors: vec![point; 4],
            cell_volumes: vec![1.0; 3],
            patches: Vec::new(),
            min_face_area: 0.0,
            max_face_area: 0.0,
            min_cell_volume: 1.0,
            max_cell_volume: 1.0,
            total_cell_volume: 3.0,
            non_positive_cell_volumes: 0,
        }
    }

    #[test]
    fn compact_addressing_matches_runtime_faces_and_stable_filters() {
        let mesh = interleaved_mesh();
        let addressing = CompactSimpleFaceAddressing::from_mesh(&mesh).expect("addressing");

        assert_eq!(addressing.faces(), mesh.faces);
        for face in 0..mesh.faces {
            assert_eq!(addressing.owner(face), mesh.owner[face]);
            assert_eq!(addressing.neighbour(face), mesh.neighbour[face]);
        }
        assert_eq!(addressing.internal_faces(), &[0, 2]);
        assert_eq!(addressing.boundary_faces(), &[1, 3]);
    }

    #[test]
    fn compact_cell_faces_match_legacy_global_owner_then_neighbour_order() {
        let mesh = interleaved_mesh();
        let addressing = CompactSimpleFaceAddressing::from_mesh(&mesh).expect("addressing");
        assert!(!addressing.limiter_adjacency_initialized());
        let adjacency = addressing
            .limiter_cell_adjacency()
            .expect("lazy cell adjacency");

        assert_eq!(adjacency.cell_faces(0), &[0]);
        assert_eq!(adjacency.cell_faces(1), &[0, 2]);
        assert_eq!(adjacency.cell_faces(2), &[1, 2, 3]);
        assert!(addressing.limiter_adjacency_initialized());
    }

    #[test]
    fn compact_self_neighbour_validation_remains_limiter_lazy() {
        let mut mesh = interleaved_mesh();
        mesh.neighbour[0] = Some(0);
        let addressing = CompactSimpleFaceAddressing::from_mesh(&mesh).expect("addressing");

        assert_eq!(addressing.owner(0), 0);
        assert_eq!(addressing.neighbour(0), Some(0));
        assert!(!addressing.limiter_adjacency_initialized());
        let error = addressing
            .limiter_cell_adjacency()
            .expect_err("self-neighbour must fail only when requested");
        assert_eq!(
            error.to_string(),
            "face 0 has identical owner and neighbour cell 0"
        );
        assert!(!addressing.limiter_adjacency_initialized());
        let retry = addressing
            .limiter_cell_adjacency()
            .expect_err("failed lazy initialization must remain retryable");
        assert_eq!(retry.to_string(), error.to_string());
        assert!(!addressing.limiter_adjacency_initialized());
    }

    #[test]
    fn compact_limiter_allocation_failure_is_atomic_and_retryable() {
        let mesh = interleaved_mesh();
        let addressing = CompactSimpleFaceAddressing::from_mesh(&mesh).expect("addressing");

        assert!(!addressing.limiter_adjacency_initialized());
        let error = addressing
            .inject_limiter_allocation_failure()
            .expect_err("injected allocation failure");
        assert!(matches!(error, crate::MeshError::OutOfMemory));
        assert!(!addressing.limiter_adjacency_initialized());

        let adjacency = addressing
            .limiter_cell_adjacency()
            .expect("retry after allocation failure");
        assert_eq!(adjacency.cell_faces(2), &[1, 2, 3]);
        assert!(addressing.limiter_adjacency_initialized());
    }
}
