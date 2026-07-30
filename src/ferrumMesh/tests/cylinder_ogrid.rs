use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use ferrum_mesh::cylinder_ogrid::{
    CylinderOGridPreset, CylinderRadialGrading, generate_cylinder_ogrid,
};
use ferrum_mesh::foam::{FoamWriteOptions, write_openfoam_case_with_options};
use ferrum_mesh::geometry::{
    CartesianAxis, PolyMeshQualityOptions, compute_poly_mesh_geometry, summarize_poly_mesh_quality,
};
use ferrum_mesh::gmsh::read_msh22_ascii;
use ferrum_mesh::gmsh_write::{write_msh22_ascii, write_msh22_ascii_to};
use ferrum_mesh::poly_mesh::PolyMesh;
use ferrum_mesh::{Mesh, MeshError};

#[test]
fn presets_have_exact_counts_names_and_patch_partition() {
    for (preset, angular, radial, patch_counts) in preset_cases() {
        let config = preset.config();
        let mesh = generate_cylinder_ogrid(&config).expect("generate cylinder O-grid preset");
        let cells = angular * radial;
        let points = 2 * angular * (radial + 1);

        assert_eq!(config.angular_cells, angular);
        assert_eq!(config.radial_cells, radial);
        assert_eq!(mesh.cells.len(), cells);
        assert_eq!(mesh.points.len(), points);
        assert_eq!(mesh.boundary_faces.len(), 2 * angular + 2 * cells);
        assert_eq!(
            mesh.physical_names
                .iter()
                .map(|physical| (physical.dim, physical.tag, physical.name.as_str()))
                .collect::<Vec<_>>(),
            vec![
                (2, 1, "inlet"),
                (2, 2, "outlet"),
                (2, 3, "cylinder"),
                (2, 4, "frontAndBack"),
                (3, 10, "fluid"),
            ]
        );
        for (tag, expected) in (1..=4).zip(patch_counts) {
            assert_eq!(
                mesh.boundary_faces
                    .iter()
                    .filter(|face| face.physical_tag == tag)
                    .count(),
                expected,
                "wrong face count for preset {preset:?}, tag {tag}"
            );
        }
        assert!(mesh.cells.iter().all(|cell| cell.physical_tag == 10));
        assert!(mesh.unsupported_elements.is_empty());
        assert_sequential_source_ids(&mesh);
    }
}

#[test]
fn legacy_smoke_preserves_checked_in_points_and_seam_contract() {
    let config = CylinderOGridPreset::LegacySmoke.config();
    let mesh = generate_cylinder_ogrid(&config).expect("generate legacy cylinder O-grid");
    let packaged = PolyMesh::read(
        &PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tutorials/incompressibleFluid/cylinder/ferrum/case/constant/polyMesh"),
    )
    .expect("read packaged cylinder polyMesh");

    assert_eq!(mesh.points.len(), packaged.points.len());
    for (generated, checked_in) in mesh.points.iter().zip(&packaged.points) {
        assert_close(generated.x, checked_in.x, 1.0e-14);
        assert_close(generated.y, checked_in.y, 1.0e-14);
        assert_close(generated.z, checked_in.z, 1.0e-14);
        assert_ne!(generated.x.to_bits(), (-0.0_f64).to_bits());
        assert_ne!(generated.y.to_bits(), (-0.0_f64).to_bits());
        assert_ne!(generated.z.to_bits(), (-0.0_f64).to_bits());
    }

    let angular = config.angular_cells;
    assert_ne!(
        mesh.points[0].x.to_bits(),
        mesh.points[angular - 1].x.to_bits()
    );
    assert_ne!(
        mesh.points[0].y.to_bits(),
        mesh.points[angular - 1].y.to_bits()
    );
    let seam_cell = &mesh.cells[angular - 1];
    assert_eq!(seam_cell.nodes[0], angular - 1);
    assert_eq!(seam_cell.nodes[1], 2 * angular - 1);
    assert_eq!(seam_cell.nodes[2], angular);
    assert_eq!(seam_cell.nodes[3], 0);
}

#[test]
fn production_coarse_points_are_bitwise_nested_in_fine() {
    let coarse_config = CylinderOGridPreset::Coarse.config();
    let fine_config = CylinderOGridPreset::Fine.config();
    let coarse = generate_cylinder_ogrid(&coarse_config).expect("generate coarse O-grid");
    let fine = generate_cylinder_ogrid(&fine_config).expect("generate fine O-grid");
    let coarse_layer_points = (coarse_config.radial_cells + 1) * coarse_config.angular_cells;
    let fine_layer_points = (fine_config.radial_cells + 1) * fine_config.angular_cells;

    for layer in 0..2 {
        for radial in 0..=coarse_config.radial_cells {
            for angular in 0..coarse_config.angular_cells {
                let coarse_index =
                    layer * coarse_layer_points + radial * coarse_config.angular_cells + angular;
                let fine_index = layer * fine_layer_points
                    + (2 * radial) * fine_config.angular_cells
                    + 2 * angular;
                let coarse_point = coarse.points[coarse_index];
                let fine_point = fine.points[fine_index];
                assert_eq!(coarse_point.x.to_bits(), fine_point.x.to_bits());
                assert_eq!(coarse_point.y.to_bits(), fine_point.y.to_bits());
                assert_eq!(coarse_point.z.to_bits(), fine_point.z.to_bits());
            }
        }
    }
}

#[test]
fn production_grading_is_continuous_finite_and_strictly_expanding() {
    let config = CylinderOGridPreset::Coarse.config();
    assert_eq!(
        config.radial_grading,
        CylinderRadialGrading::Exponential { ratio: 1000.0 }
    );
    let mesh = generate_cylinder_ogrid(&config).expect("generate coarse O-grid");
    let mut previous_x = mesh.points[0].x;
    let mut previous_width = 0.0;
    for radial in 1..=config.radial_cells {
        let x = mesh.points[radial * config.angular_cells].x;
        let width = x - previous_x;
        assert!(x.is_finite());
        assert!(width > 0.0);
        if radial > 1 {
            assert!(
                width > previous_width,
                "radial cell width did not grow at ring {radial}"
            );
        }
        previous_x = x;
        previous_width = width;
    }
    assert_eq!(previous_x.to_bits(), config.x_max.to_bits());
}

#[test]
fn all_presets_are_mesh_and_byte_deterministic_and_roundtrip_exactly() {
    for (preset, _, _, _) in preset_cases() {
        let first_mesh = generate_cylinder_ogrid(&preset.config())
            .expect("generate first fresh cylinder O-grid");
        let second_mesh = generate_cylinder_ogrid(&preset.config())
            .expect("generate second fresh cylinder O-grid");
        assert_mesh_equal(&first_mesh, &second_mesh);

        let mut first_bytes = Vec::new();
        let mut second_bytes = Vec::new();
        write_msh22_ascii_to(&mut first_bytes, &first_mesh).expect("write first Gmsh output");
        write_msh22_ascii_to(&mut second_bytes, &second_mesh).expect("write second Gmsh output");
        assert_eq!(
            first_bytes, second_bytes,
            "fresh generations differ byte-for-byte for {preset:?}"
        );
        assert!(!first_bytes.windows(2).any(|window| window == b"\r\n"));
        assert!(String::from_utf8_lossy(&first_bytes).contains("3 10 \"fluid\""));

        let path = temporary_path(&format!("cylinder-ogrid-{preset:?}-roundtrip"), "msh");
        write_msh22_ascii(&path, &first_mesh).expect("write Gmsh file");
        let roundtrip = read_msh22_ascii(&path).expect("read generated Gmsh file");
        fs::remove_file(&path).expect("remove generated Gmsh file");
        assert_mesh_equal(&first_mesh, &roundtrip);
    }
}

#[test]
fn gmsh_writer_rejects_trailing_separator_without_clobber() {
    let directory = temporary_path("gmsh-trailing-separator", "dir");
    fs::create_dir(&directory).expect("create temporary output directory");
    let victim = directory.join("victim.msh");
    fs::write(&victim, b"preserve me").expect("write victim file");
    let path = PathBuf::from(format!("{}/", victim.display()));

    assert!(write_msh22_ascii(&path, &gmsh_test_mesh()).is_err());
    assert_eq!(fs::read(&victim).expect("read victim"), b"preserve me");
    fs::remove_dir_all(&directory).expect("remove temporary output directory");
}

#[test]
fn gmsh_writer_rejects_terminal_dot_component_without_clobber() {
    let directory = temporary_path("gmsh-terminal-dot", "dir");
    fs::create_dir(&directory).expect("create temporary output directory");
    let victim = directory.join("victim.msh");
    fs::write(&victim, b"preserve me").expect("write victim file");

    assert!(write_msh22_ascii(&victim.join("."), &gmsh_test_mesh()).is_err());
    assert_eq!(fs::read(&victim).expect("read victim"), b"preserve me");
    fs::remove_dir_all(&directory).expect("remove temporary output directory");
}

#[test]
fn gmsh_writer_rejects_raw_parent_component_before_normalization() {
    let directory = temporary_path("gmsh-parent-component", "dir");
    fs::create_dir(&directory).expect("create temporary output directory");
    let victim = directory.join("victim.msh");
    fs::write(&victim, b"preserve me").expect("write victim file");
    let path = directory.join("missing").join("..").join("victim.msh");

    assert!(write_msh22_ascii(&path, &gmsh_test_mesh()).is_err());
    assert_eq!(fs::read(&victim).expect("read victim"), b"preserve me");
    fs::remove_dir_all(&directory).expect("remove temporary output directory");
}

#[cfg(any(unix, windows))]
#[test]
fn gmsh_writer_rejects_final_symlink_without_clobber() {
    let directory = temporary_path("gmsh-final-symlink", "dir");
    fs::create_dir(&directory).expect("create temporary output directory");
    let victim = directory.join("victim.txt");
    let output = directory.join("mesh.msh");
    fs::write(&victim, b"preserve me").expect("write victim file");
    if create_file_symlink(&victim, &output).is_err() {
        fs::remove_dir_all(&directory).expect("remove temporary output directory");
        return;
    }

    assert!(write_msh22_ascii(&output, &gmsh_test_mesh()).is_err());
    assert_eq!(
        fs::read(&victim).expect("read symlink target"),
        b"preserve me"
    );
    fs::remove_dir_all(&directory).expect("remove temporary output directory");
}

#[test]
fn gmsh_writer_replaces_hardlink_without_modifying_target() {
    let directory = temporary_path("gmsh-hardlink", "dir");
    fs::create_dir(&directory).expect("create temporary output directory");
    let victim = directory.join("victim.txt");
    let output = directory.join("mesh.msh");
    fs::write(&victim, b"preserve me").expect("write victim file");

    fs::hard_link(&victim, &output).expect("create output hard link");
    write_msh22_ascii(&output, &gmsh_test_mesh()).expect("replace hard link safely");
    assert_eq!(
        fs::read(&victim).expect("read hard-link target"),
        b"preserve me"
    );
    assert!(
        fs::read(&output)
            .expect("read replacement output")
            .starts_with(b"$MeshFormat\n")
    );
    fs::remove_dir_all(&directory).expect("remove temporary output directory");
}

#[cfg(any(unix, windows))]
#[test]
fn gmsh_writer_rejects_symlink_or_reparse_parent() {
    let directory = temporary_path("gmsh-linked-parent", "dir");
    let linked_directory = temporary_path("gmsh-linked-parent-alias", "dir");
    fs::create_dir(&directory).expect("create temporary output directory");
    if create_directory_symlink(&directory, &linked_directory).is_err() {
        fs::remove_dir_all(&directory).expect("remove temporary output directory");
        return;
    }

    assert!(write_msh22_ascii(&linked_directory.join("escaped.msh"), &gmsh_test_mesh()).is_err());
    assert!(!directory.join("escaped.msh").exists());
    remove_directory_symlink(&linked_directory).expect("remove output directory symlink");
    fs::remove_dir_all(&directory).expect("remove temporary output directory");
}

#[cfg(unix)]
#[test]
fn gmsh_writer_preserves_existing_access_mode() {
    use std::os::unix::fs::{MetadataExt, PermissionsExt};

    for mode in [0o600, 0o750] {
        let directory = temporary_path(&format!("gmsh-mode-{mode:o}"), "dir");
        fs::create_dir(&directory).expect("create temporary output directory");
        let output = directory.join("mesh.msh");
        fs::write(&output, b"old").expect("write existing output");
        fs::set_permissions(&output, fs::Permissions::from_mode(mode))
            .expect("set existing output mode");

        write_msh22_ascii(&output, &gmsh_test_mesh()).expect("replace output");
        assert_eq!(
            fs::metadata(&output).expect("stat output").mode() & 0o777,
            mode
        );
        fs::remove_dir_all(&directory).expect("remove temporary output directory");
    }
}

#[cfg(windows)]
#[test]
fn gmsh_writer_rejects_trailing_dot_space_and_ads_aliases() {
    let directory = temporary_path("gmsh-windows-alias", "dir");
    fs::create_dir(&directory).expect("create temporary output directory");
    let victim = directory.join("victim.msh");
    fs::write(&victim, b"preserve me").expect("write victim file");

    for alias in ["victim.msh.", "victim.msh ", "victim.msh:stream"] {
        assert!(write_msh22_ascii(&directory.join(alias), &gmsh_test_mesh()).is_err());
        assert_eq!(fs::read(&victim).expect("read victim"), b"preserve me");
    }
    fs::remove_dir_all(&directory).expect("remove temporary output directory");
}

#[test]
fn gmsh_writer_accepts_unambiguous_dot_names() {
    let directory = temporary_path("gmsh-valid-dot-names", "dir");
    fs::create_dir(&directory).expect("create temporary output directory");
    fs::create_dir(directory.join("a..b")).expect("create dotted parent");
    let mesh = gmsh_test_mesh();

    for output in [
        directory.join(".").join("mesh.msh"),
        directory.join(".mesh.msh"),
        directory.join("a..b").join("mesh.msh"),
    ] {
        write_msh22_ascii(&output, &mesh).expect("write unambiguous dotted path");
        assert!(
            fs::read(&output)
                .expect("read output")
                .starts_with(b"$MeshFormat\n")
        );
    }
    fs::remove_dir_all(&directory).expect("remove temporary output directory");
}

#[test]
fn legacy_smoke_roundtrips_with_valid_raw_poly_mesh_quality() {
    assert_preset_roundtrip_quality(CylinderOGridPreset::LegacySmoke, [4, 12, 16, 96], false);
}

#[test]
fn coarse_roundtrips_to_a_valid_quality_gated_poly_mesh() {
    assert_preset_roundtrip_quality(CylinderOGridPreset::Coarse, [96, 32, 128, 10_752], true);
}

#[test]
fn fine_roundtrips_to_a_valid_quality_gated_poly_mesh() {
    assert_preset_roundtrip_quality(CylinderOGridPreset::Fine, [192, 64, 256, 43_008], true);
}

#[test]
fn invalid_configs_fail_without_mutating_a_mesh() {
    let base = CylinderOGridPreset::LegacySmoke.config();

    let mut invalid = base;
    invalid.angular_cells = 15;
    assert_invalid_contains(
        generate_cylinder_ogrid(&invalid),
        "angular_cells must be at least 8 and divisible by 8",
    );

    invalid = base;
    invalid.radial_cells = 0;
    assert_invalid_contains(
        generate_cylinder_ogrid(&invalid),
        "radial_cells must be greater than zero",
    );

    invalid = base;
    invalid.diameter = f64::NAN;
    assert_invalid_contains(generate_cylinder_ogrid(&invalid), "diameter must be finite");

    invalid = base;
    invalid.diameter = f64::from_bits(1);
    assert_invalid_contains(
        generate_cylinder_ogrid(&invalid),
        "derived radius must be finite and greater than zero",
    );

    invalid = base;
    invalid.depth = f64::from_bits(1);
    assert_invalid_contains(
        generate_cylinder_ogrid(&invalid),
        "derived half-depth must be finite and greater than zero",
    );

    invalid = base;
    invalid.x_min = -0.25 * invalid.diameter;
    assert_invalid_contains(
        generate_cylinder_ogrid(&invalid),
        "domain must strictly enclose",
    );

    invalid = base;
    invalid.radial_grading = CylinderRadialGrading::Exponential { ratio: 0.0 };
    assert_invalid_contains(
        generate_cylinder_ogrid(&invalid),
        "grading ratio must be finite and greater than zero",
    );

    invalid = base;
    invalid.radial_cells = usize::MAX;
    assert!(matches!(
        generate_cylinder_ogrid(&invalid),
        Err(MeshError::OutOfMemory)
    ));
}

#[test]
fn gmsh22_writer_rejects_invalid_mesh_data_before_file_creation() {
    let mut mesh = generate_cylinder_ogrid(&CylinderOGridPreset::LegacySmoke.config())
        .expect("generate legacy O-grid");
    mesh.points[0].x = f64::INFINITY;
    let path = temporary_path("invalid-cylinder-ogrid", "msh");
    assert_invalid_contains(write_msh22_ascii(&path, &mesh), "non-finite Gmsh node");
    assert!(!path.exists());

    mesh.points[0].x = 0.0005;
    mesh.boundary_faces[0].nodes[0] = usize::MAX;
    assert_invalid_contains(
        write_msh22_ascii_to(&mut Vec::new(), &mesh),
        "references missing node index",
    );

    mesh.boundary_faces[0].nodes[0] = 0;
    mesh.physical_names[0].name.push('"');
    assert_invalid_contains(
        write_msh22_ascii_to(&mut Vec::new(), &mesh),
        "unsupported quoting",
    );
}

fn preset_cases() -> [(CylinderOGridPreset, usize, usize, [usize; 4]); 3] {
    [
        (CylinderOGridPreset::LegacySmoke, 16, 3, [4, 12, 16, 96]),
        (CylinderOGridPreset::Coarse, 128, 42, [96, 32, 128, 10_752]),
        (CylinderOGridPreset::Fine, 256, 84, [192, 64, 256, 43_008]),
    ]
}

fn assert_preset_roundtrip_quality(
    preset: CylinderOGridPreset,
    patch_counts: [usize; 4],
    enforce_production_gates: bool,
) {
    let generated =
        generate_cylinder_ogrid(&preset.config()).expect("generate quality-gated O-grid");
    let root = temporary_path(&format!("cylinder-{preset:?}"), "case");
    fs::create_dir(&root).expect("create quality roundtrip root");
    let msh_path = root.join("cylinder.msh");
    let case_dir = root.join("ferrum");

    write_msh22_ascii(&msh_path, &generated).expect("write quality-gated Gmsh mesh");
    let imported = read_msh22_ascii(&msh_path).expect("read quality-gated Gmsh mesh");
    assert_mesh_equal(&generated, &imported);

    let mut options = FoamWriteOptions::default();
    options.set_patch_type("cylinder", "wall");
    options.set_patch_type("frontAndBack", "empty");
    let write_summary = write_openfoam_case_with_options(&imported, &case_dir, &msh_path, &options)
        .expect("write quality-gated polyMesh");
    assert_eq!(write_summary.points, generated.points.len());
    assert_eq!(write_summary.cells, generated.cells.len());
    assert_eq!(write_summary.unmatched_boundary_faces, 0);
    assert_eq!(write_summary.duplicate_boundary_faces, 0);
    assert_eq!(write_summary.non_manifold_faces, 0);
    assert_eq!(
        write_summary
            .cell_zones
            .iter()
            .map(|zone| (zone.name.as_str(), zone.physical_tag, zone.cells))
            .collect::<Vec<_>>(),
        vec![("fluid", 10, generated.cells.len())]
    );
    assert_eq!(
        write_summary
            .patches
            .iter()
            .map(|patch| {
                (
                    patch.name.as_str(),
                    patch.patch_type.as_str(),
                    patch.physical_tag,
                    patch.faces,
                )
            })
            .collect::<Vec<_>>(),
        vec![
            ("inlet", "patch", Some(1), patch_counts[0]),
            ("outlet", "patch", Some(2), patch_counts[1]),
            ("cylinder", "wall", Some(3), patch_counts[2]),
            ("frontAndBack", "empty", Some(4), patch_counts[3]),
        ],
        "wrong writer patch contract for {preset:?}"
    );
    assert_eq!(
        write_summary
            .face_zones
            .iter()
            .map(|zone| (zone.name.as_str(), zone.physical_tag, zone.faces))
            .collect::<Vec<_>>(),
        vec![
            ("inlet", 1, patch_counts[0]),
            ("outlet", 2, patch_counts[1]),
            ("cylinder", 3, patch_counts[2]),
            ("frontAndBack", 4, patch_counts[3]),
        ],
        "wrong writer faceZone contract for {preset:?}"
    );

    let poly_mesh =
        PolyMesh::read(&case_dir.join("constant/polyMesh")).expect("read quality-gated polyMesh");
    assert_eq!(poly_mesh.cell_count(), generated.cells.len());
    assert_eq!(
        poly_mesh
            .patches
            .iter()
            .map(|patch| (patch.name.as_str(), patch.patch_type.as_str(), patch.faces))
            .collect::<Vec<_>>(),
        vec![
            ("inlet", "patch", patch_counts[0]),
            ("outlet", "patch", patch_counts[1]),
            ("cylinder", "wall", patch_counts[2]),
            ("frontAndBack", "empty", patch_counts[3]),
        ],
        "wrong readback patch contract for {preset:?}"
    );

    let quality = summarize_poly_mesh_quality(
        &poly_mesh,
        &PolyMeshQualityOptions {
            empty_extrusion_axis: Some(CartesianAxis::Z),
        },
    )
    .expect("measure polyMesh quality");
    assert!(
        quality.problematic_face_indices.is_empty(),
        "{preset:?} has problematic faces: {:?}",
        quality.problematic_face_indices
    );
    assert!(
        quality.problematic_cell_indices.is_empty(),
        "{preset:?} has problematic cells: {:?}",
        quality.problematic_cell_indices
    );
    assert!(quality.non_finite_face_geometry_indices.is_empty());
    assert!(quality.non_positive_face_area_indices.is_empty());
    assert!(quality.non_finite_cell_geometry_indices.is_empty());
    assert!(quality.non_positive_cell_volume_indices.is_empty());
    let non_orthogonality = quality
        .max_internal_non_orthogonality_degrees
        .expect("internal non-orthogonality");
    let skewness = quality
        .max_normalized_internal_skewness
        .expect("internal skewness");
    let aspect_ratio = quality
        .max_active_edge_aspect_ratio
        .expect("active edge aspect ratio");
    assert!(non_orthogonality.is_finite() && non_orthogonality >= 0.0);
    assert!(skewness.is_finite() && skewness >= 0.0);
    assert!(aspect_ratio.is_finite() && aspect_ratio > 0.0);
    if enforce_production_gates {
        assert!(
            non_orthogonality <= 50.0,
            "{preset:?} raw quality: non-orthogonality={non_orthogonality}, normalized-skewness={skewness}, active-edge-aspect={aspect_ratio}; non-orthogonality exceeds 50 degrees"
        );
        assert!(
            skewness <= 0.55,
            "{preset:?} raw quality: non-orthogonality={non_orthogonality}, normalized-skewness={skewness}, active-edge-aspect={aspect_ratio}; normalized skewness exceeds 0.55"
        );
        assert!(
            aspect_ratio <= 4.0,
            "{preset:?} raw quality: non-orthogonality={non_orthogonality}, normalized-skewness={skewness}, active-edge-aspect={aspect_ratio}; active-edge aspect ratio exceeds 4"
        );
    }
    assert_closed_cylinder_area_vectors(&poly_mesh);

    fs::remove_dir_all(&root).expect("remove quality roundtrip root");
}

fn assert_sequential_source_ids(mesh: &Mesh) {
    for (index, face) in mesh.boundary_faces.iter().enumerate() {
        assert_eq!(face.source_id, index + 1);
    }
    for (index, cell) in mesh.cells.iter().enumerate() {
        assert_eq!(cell.source_id, mesh.boundary_faces.len() + index + 1);
    }
}

fn assert_mesh_equal(expected: &Mesh, actual: &Mesh) {
    assert_eq!(expected.points.len(), actual.points.len());
    for (left, right) in expected.points.iter().zip(&actual.points) {
        assert_eq!(left.x.to_bits(), right.x.to_bits());
        assert_eq!(left.y.to_bits(), right.y.to_bits());
        assert_eq!(left.z.to_bits(), right.z.to_bits());
    }
    assert_eq!(expected.cells.len(), actual.cells.len());
    for (left, right) in expected.cells.iter().zip(&actual.cells) {
        assert_eq!(left.source_id, right.source_id);
        assert_eq!(left.physical_tag, right.physical_tag);
        assert_eq!(left.nodes, right.nodes);
    }
    assert_eq!(expected.boundary_faces.len(), actual.boundary_faces.len());
    for (left, right) in expected.boundary_faces.iter().zip(&actual.boundary_faces) {
        assert_eq!(left.source_id, right.source_id);
        assert_eq!(left.physical_tag, right.physical_tag);
        assert_eq!(left.nodes, right.nodes);
    }
    assert_eq!(expected.physical_names.len(), actual.physical_names.len());
    for (left, right) in expected.physical_names.iter().zip(&actual.physical_names) {
        assert_eq!(left.dim, right.dim);
        assert_eq!(left.tag, right.tag);
        assert_eq!(left.name, right.name);
    }
    assert!(actual.unsupported_elements.is_empty());
}

fn assert_closed_cylinder_area_vectors(poly_mesh: &PolyMesh) {
    let geometry = compute_poly_mesh_geometry(poly_mesh).expect("compute cylinder geometry");
    let cylinder = poly_mesh
        .patches
        .iter()
        .find(|patch| patch.name == "cylinder")
        .expect("cylinder patch");
    let mut sum = [0.0_f64; 3];
    let mut total_area = 0.0;
    for area in
        &geometry.face_area_vectors[cylinder.start_face..cylinder.start_face + cylinder.faces]
    {
        sum[0] += area.x;
        sum[1] += area.y;
        sum[2] += area.z;
        total_area += area.x.hypot(area.y).hypot(area.z);
    }
    let closure = sum[0].hypot(sum[1]).hypot(sum[2]);
    assert!(total_area.is_finite() && total_area > 0.0);
    assert!(
        closure <= total_area * 1.0e-10,
        "cylinder area-vector closure {closure} exceeds tolerance for total area {total_area}"
    );
}

fn assert_invalid_contains<T>(result: Result<T, MeshError>, expected: &str) {
    match result {
        Err(MeshError::InvalidInput(message)) => assert!(
            message.contains(expected),
            "expected '{message}' to contain '{expected}'"
        ),
        Err(other) => panic!("expected InvalidInput, got {other}"),
        Ok(_) => panic!("expected InvalidInput containing '{expected}'"),
    }
}

fn assert_close(left: f64, right: f64, tolerance: f64) {
    assert!(
        (left - right).abs() <= tolerance,
        "expected {left} to be within {tolerance} of {right}"
    );
}

fn gmsh_test_mesh() -> Mesh {
    generate_cylinder_ogrid(&CylinderOGridPreset::LegacySmoke.config()).expect("generate test mesh")
}

#[cfg(unix)]
fn create_file_symlink(target: &std::path::Path, link: &std::path::Path) -> std::io::Result<()> {
    std::os::unix::fs::symlink(target, link)
}

#[cfg(windows)]
fn create_file_symlink(target: &std::path::Path, link: &std::path::Path) -> std::io::Result<()> {
    std::os::windows::fs::symlink_file(target, link)
}

#[cfg(unix)]
fn create_directory_symlink(
    target: &std::path::Path,
    link: &std::path::Path,
) -> std::io::Result<()> {
    std::os::unix::fs::symlink(target, link)
}

#[cfg(windows)]
fn create_directory_symlink(
    target: &std::path::Path,
    link: &std::path::Path,
) -> std::io::Result<()> {
    std::os::windows::fs::symlink_dir(target, link)
}

#[cfg(unix)]
fn remove_directory_symlink(link: &std::path::Path) -> std::io::Result<()> {
    fs::remove_file(link)
}

#[cfg(windows)]
fn remove_directory_symlink(link: &std::path::Path) -> std::io::Result<()> {
    fs::remove_dir(link)
}

fn temporary_path(stem: &str, extension: &str) -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock after epoch")
        .as_nanos();
    std::env::temp_dir().join(format!(
        "ferrum-{stem}-{}-{nonce}.{extension}",
        std::process::id()
    ))
}
