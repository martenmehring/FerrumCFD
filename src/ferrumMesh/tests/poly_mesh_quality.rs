use std::path::PathBuf;

use ferrum_mesh::Point3;
use ferrum_mesh::geometry::{CartesianAxis, PolyMeshQualityOptions, summarize_poly_mesh_quality};
use ferrum_mesh::poly_mesh::PolyMesh;

#[test]
fn orthogonal_two_cell_mesh_has_zero_internal_angle_and_skewness() {
    let mesh = two_unit_cubes();
    let summary = summarize_poly_mesh_quality(&mesh, &PolyMeshQualityOptions::default())
        .expect("valid two-cell quality summary");

    assert_eq!(summary.cells, 2);
    assert_eq!(summary.faces, 11);
    assert_eq!(summary.internal_faces, 1);
    assert_close(
        summary
            .max_internal_non_orthogonality_degrees
            .expect("one internal face"),
        0.0,
    );
    assert_eq!(summary.max_internal_non_orthogonality_face, Some(0));
    assert_close(
        summary
            .max_normalized_internal_skewness
            .expect("one internal face"),
        0.0,
    );
    assert_eq!(summary.max_normalized_internal_skewness_face, Some(0));
    assert_close(
        summary
            .max_active_edge_aspect_ratio
            .expect("active cell edges"),
        1.0,
    );
    assert_eq!(summary.max_active_edge_aspect_ratio_cell, Some(0));
    assert!(summary.problematic_face_indices.is_empty());
    assert!(summary.problematic_cell_indices.is_empty());
}

#[test]
fn skewed_two_cell_mesh_matches_analytic_internal_metrics() {
    let mut mesh = two_unit_cubes();
    for point in &mut mesh.points[8..12] {
        point.y += 1.0;
    }

    let summary = summarize_poly_mesh_quality(&mesh, &PolyMeshQualityOptions::default())
        .expect("valid skewed two-cell quality summary");
    let centre_distance = 1.0_f64.hypot(0.5);

    assert_close(
        summary.max_internal_non_orthogonality_degrees.unwrap(),
        0.5_f64.atan().to_degrees(),
    );
    assert_close(
        summary.max_normalized_internal_skewness.unwrap(),
        0.25 / centre_distance,
    );
    assert_eq!(summary.max_internal_non_orthogonality_face, Some(0));
    assert_eq!(summary.max_normalized_internal_skewness_face, Some(0));
    assert!(summary.problematic_face_indices.is_empty());
    assert!(summary.problematic_cell_indices.is_empty());
}

#[test]
fn coincident_cell_centres_make_internal_metrics_undefined_and_indexed() {
    let mut mesh = two_unit_cubes();
    let replacements = mesh.points[..4].to_vec();
    for (point, replacement) in mesh.points[8..12].iter_mut().zip(replacements) {
        *point = replacement;
    }

    let summary = summarize_poly_mesh_quality(&mesh, &PolyMeshQualityOptions::default())
        .expect("degenerate centre connection is diagnosed");

    assert_eq!(summary.max_internal_non_orthogonality_degrees, None);
    assert_eq!(summary.max_normalized_internal_skewness, None);
    assert!(summary.problematic_face_indices.contains(&0));
    assert_eq!(summary.problematic_cell_indices, vec![0, 1]);
}

#[test]
fn empty_extrusion_axis_is_excluded_only_from_edge_aspect_ratio() {
    let mesh = cuboid(2.0, 1.0, 100.0);
    let full = summarize_poly_mesh_quality(&mesh, &PolyMeshQualityOptions::default())
        .expect("full 3-D quality summary");
    let active = summarize_poly_mesh_quality(
        &mesh,
        &PolyMeshQualityOptions {
            empty_extrusion_axis: Some(CartesianAxis::Z),
        },
    )
    .expect("2-D active quality summary");

    assert_close(full.max_active_edge_aspect_ratio.unwrap(), 100.0);
    assert_close(active.max_active_edge_aspect_ratio.unwrap(), 2.0);
    assert!(active.problematic_face_indices.is_empty());
    assert!(active.problematic_cell_indices.is_empty());
    assert_eq!(
        full.max_internal_non_orthogonality_degrees,
        active.max_internal_non_orthogonality_degrees
    );
    assert_eq!(
        full.max_normalized_internal_skewness,
        active.max_normalized_internal_skewness
    );
}

#[test]
fn non_finite_geometry_is_reported_by_stable_face_and_cell_indices() {
    let mut mesh = cuboid(1.0, 1.0, 1.0);
    mesh.points[0].x = f64::NAN;

    let summary = summarize_poly_mesh_quality(&mesh, &PolyMeshQualityOptions::default())
        .expect("non-finite coordinates are measured rather than policy-rejected");

    assert_eq!(summary.non_finite_face_geometry_indices, vec![0, 2, 5]);
    assert_eq!(summary.non_finite_cell_geometry_indices, vec![0]);
    assert_eq!(summary.problematic_face_indices, vec![0, 2, 5]);
    assert_eq!(summary.problematic_cell_indices, vec![0]);
}

#[test]
fn collapsed_geometry_reports_non_positive_faces_and_cell_volume() {
    let mesh = cuboid(1.0, 1.0, 0.0);

    let summary = summarize_poly_mesh_quality(&mesh, &PolyMeshQualityOptions::default())
        .expect("collapsed geometry summary");

    assert_eq!(summary.non_positive_face_area_indices, vec![2, 3, 4, 5]);
    assert_eq!(summary.non_positive_cell_volume_indices, vec![0]);
    assert_eq!(summary.problematic_face_indices, vec![2, 3, 4, 5]);
    assert_eq!(summary.problematic_cell_indices, vec![0]);
}

#[test]
fn overflowing_cell_label_fails_before_quality_allocations() {
    let mut mesh = cuboid(1.0, 1.0, 1.0);
    mesh.owner[0] = usize::MAX;

    let error = summarize_poly_mesh_quality(&mesh, &PolyMeshQualityOptions::default())
        .expect_err("overflowing labels must fail before allocating cell buffers");

    assert!(error.to_string().contains("overflows the cell count"));
}

fn cuboid(x: f64, y: f64, z: f64) -> PolyMesh {
    PolyMesh {
        path: PathBuf::from("synthetic/polyMesh"),
        points: vec![
            point(0.0, 0.0, 0.0),
            point(x, 0.0, 0.0),
            point(x, y, 0.0),
            point(0.0, y, 0.0),
            point(0.0, 0.0, z),
            point(x, 0.0, z),
            point(x, y, z),
            point(0.0, y, z),
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
    }
}

fn two_unit_cubes() -> PolyMesh {
    PolyMesh {
        path: PathBuf::from("synthetic/polyMesh"),
        points: vec![
            point(0.0, 0.0, 0.0),
            point(0.0, 1.0, 0.0),
            point(0.0, 1.0, 1.0),
            point(0.0, 0.0, 1.0),
            point(1.0, 0.0, 0.0),
            point(1.0, 1.0, 0.0),
            point(1.0, 1.0, 1.0),
            point(1.0, 0.0, 1.0),
            point(2.0, 0.0, 0.0),
            point(2.0, 1.0, 0.0),
            point(2.0, 1.0, 1.0),
            point(2.0, 0.0, 1.0),
        ],
        // OpenFOAM stores every internal face before boundary faces.
        faces: vec![
            vec![4, 5, 6, 7],
            vec![0, 3, 2, 1],
            vec![0, 4, 7, 3],
            vec![1, 2, 6, 5],
            vec![0, 1, 5, 4],
            vec![3, 7, 6, 2],
            vec![8, 9, 10, 11],
            vec![4, 8, 11, 7],
            vec![5, 6, 10, 9],
            vec![4, 5, 9, 8],
            vec![7, 11, 10, 6],
        ],
        owner: vec![0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        neighbour: vec![1],
        patches: Vec::new(),
    }
}

fn point(x: f64, y: f64, z: f64) -> Point3 {
    Point3 { x, y, z }
}

fn assert_close(left: f64, right: f64) {
    assert!(
        (left - right).abs() < 1.0e-12,
        "expected {left} to be close to {right}"
    );
}
