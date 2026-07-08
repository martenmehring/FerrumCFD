use std::path::{Path, PathBuf};

use crate::Result;
use crate::poly_mesh::{BoundaryPatch, PolyMesh};

#[derive(Debug)]
pub struct PatchValidationSummary {
    pub case_dir: PathBuf,
    pub patches: usize,
    pub empty_patches: usize,
    pub wedge_patches: usize,
    pub symmetry_patches: usize,
    pub warnings: Vec<String>,
}

pub fn validate_case_patches(case_dir: &Path) -> Result<PatchValidationSummary> {
    let mesh = PolyMesh::read(&case_dir.join("constant").join("polyMesh"))?;
    Ok(validate_poly_mesh_patches(case_dir, &mesh))
}

pub fn validate_poly_mesh_patches(case_dir: &Path, mesh: &PolyMesh) -> PatchValidationSummary {
    let mut summary = PatchValidationSummary {
        case_dir: case_dir.to_path_buf(),
        patches: mesh.patches.len(),
        empty_patches: 0,
        wedge_patches: 0,
        symmetry_patches: 0,
        warnings: Vec::new(),
    };

    for patch in &mesh.patches {
        match patch.patch_type.as_str() {
            "empty" => summary.empty_patches += 1,
            "wedge" => summary.wedge_patches += 1,
            "symmetryPlane" => summary.symmetry_patches += 1,
            _ => {}
        }

        validate_patch_range(mesh, patch, &mut summary.warnings);
        validate_special_patch(patch, &mut summary.warnings);
    }

    if summary.wedge_patches == 1 || summary.wedge_patches % 2 == 1 {
        summary.warnings.push(format!(
            "wedge patches should normally appear in pairs; found {}",
            summary.wedge_patches
        ));
    }

    summary
}

fn validate_patch_range(mesh: &PolyMesh, patch: &BoundaryPatch, warnings: &mut Vec<String>) {
    let Some(end_face) = patch.start_face.checked_add(patch.faces) else {
        warnings.push(format!("patch '{}' face range overflows usize", patch.name));
        return;
    };

    if end_face > mesh.faces.len() {
        warnings.push(format!(
            "patch '{}' range startFace={} nFaces={} exceeds total faces={}",
            patch.name,
            patch.start_face,
            patch.faces,
            mesh.faces.len()
        ));
    }

    if patch.start_face < mesh.neighbour.len() {
        warnings.push(format!(
            "patch '{}' starts inside internal face range: startFace={} internalFaces={}",
            patch.name,
            patch.start_face,
            mesh.neighbour.len()
        ));
    }
}

fn validate_special_patch(patch: &BoundaryPatch, warnings: &mut Vec<String>) {
    if matches!(
        patch.patch_type.as_str(),
        "empty" | "wedge" | "symmetryPlane"
    ) && patch.faces == 0
    {
        warnings.push(format!(
            "special patch '{}' type={} has no faces",
            patch.name, patch.patch_type
        ));
    }
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use crate::Point3;
    use crate::poly_mesh::{BoundaryPatch, PolyMesh};

    use super::validate_poly_mesh_patches;

    #[test]
    fn counts_special_patch_types() {
        let mesh = test_mesh(vec![
            BoundaryPatch {
                name: "front".to_string(),
                patch_type: "empty".to_string(),
                faces: 1,
                start_face: 0,
            },
            BoundaryPatch {
                name: "wedge_min".to_string(),
                patch_type: "wedge".to_string(),
                faces: 1,
                start_face: 1,
            },
            BoundaryPatch {
                name: "wedge_max".to_string(),
                patch_type: "wedge".to_string(),
                faces: 1,
                start_face: 2,
            },
        ]);

        let summary = validate_poly_mesh_patches(Path::new("case"), &mesh);
        assert_eq!(summary.empty_patches, 1);
        assert_eq!(summary.wedge_patches, 2);
        assert!(summary.warnings.is_empty());
    }

    #[test]
    fn warns_for_odd_wedge_count_and_internal_range() {
        let mut mesh = test_mesh(vec![BoundaryPatch {
            name: "wedge_min".to_string(),
            patch_type: "wedge".to_string(),
            faces: 1,
            start_face: 0,
        }]);
        mesh.neighbour = vec![1];

        let summary = validate_poly_mesh_patches(Path::new("case"), &mesh);
        assert_eq!(summary.wedge_patches, 1);
        assert_eq!(summary.warnings.len(), 2);
    }

    fn test_mesh(patches: Vec<BoundaryPatch>) -> PolyMesh {
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
                    x: 0.0,
                    y: 1.0,
                    z: 0.0,
                },
            ],
            faces: vec![vec![0, 1, 2], vec![0, 2, 1], vec![1, 2, 0]],
            owner: vec![0, 0, 0],
            neighbour: Vec::new(),
            patches,
        }
    }
}
