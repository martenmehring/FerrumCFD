use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use crate::{Mesh, Result};

#[derive(Clone, Debug)]
pub struct FoamWriteSummary {
    pub case_dir: PathBuf,
    pub points: usize,
    pub cells: usize,
    pub faces: usize,
    pub internal_faces: usize,
    pub boundary_faces: usize,
    pub patches: Vec<PatchSummary>,
    pub face_zones: Vec<FaceZoneSummary>,
    pub cell_zones: Vec<CellZoneSummary>,
    pub unmatched_boundary_faces: usize,
    pub duplicate_boundary_faces: usize,
    pub non_manifold_faces: usize,
}

#[derive(Clone, Debug)]
pub struct PatchSummary {
    pub name: String,
    pub patch_type: String,
    pub physical_tag: Option<i32>,
    pub faces: usize,
    pub start_face: usize,
}

#[derive(Clone, Debug, Default)]
pub struct FoamWriteOptions {
    pub patch_types: HashMap<String, String>,
}

impl FoamWriteOptions {
    pub fn set_patch_type(&mut self, patch_name: impl Into<String>, patch_type: impl Into<String>) {
        self.patch_types
            .insert(sanitize_name(&patch_name.into()), patch_type.into());
    }

    fn patch_type_for(&self, patch_name: &str) -> String {
        self.patch_types
            .get(patch_name)
            .cloned()
            .unwrap_or_else(|| "patch".to_string())
    }
}

#[derive(Clone, Debug)]
pub struct CellZoneSummary {
    pub name: String,
    pub physical_tag: i32,
    pub cells: usize,
}

#[derive(Clone, Debug)]
pub struct FaceZoneSummary {
    pub name: String,
    pub physical_tag: i32,
    pub faces: usize,
}

pub fn write_openfoam_case(
    mesh: &Mesh,
    case_dir: &Path,
    source_path: &Path,
) -> Result<FoamWriteSummary> {
    write_openfoam_case_with_options(mesh, case_dir, source_path, &FoamWriteOptions::default())
}

pub fn write_openfoam_case_with_options(
    mesh: &Mesh,
    case_dir: &Path,
    source_path: &Path,
    options: &FoamWriteOptions,
) -> Result<FoamWriteSummary> {
    let poly_mesh_dir = case_dir.join("constant").join("polyMesh");
    fs::create_dir_all(&poly_mesh_dir)?;
    fs::create_dir_all(case_dir.join("system"))?;

    let topology = build_topology(mesh);
    let ordered = order_faces(mesh, &topology.faces, options);

    write_points(&poly_mesh_dir.join("points"), mesh)?;
    write_faces(
        &poly_mesh_dir.join("faces"),
        &topology.faces,
        &ordered.face_indices,
    )?;
    write_owner(
        &poly_mesh_dir.join("owner"),
        &topology.faces,
        &ordered.face_indices,
    )?;
    write_neighbour(
        &poly_mesh_dir.join("neighbour"),
        &topology.faces,
        &ordered.internal_face_indices,
    )?;
    write_boundary(&poly_mesh_dir.join("boundary"), &ordered.patches)?;
    let face_zones = write_face_zones(
        &poly_mesh_dir.join("faceZones"),
        mesh,
        &topology.faces,
        &ordered,
    )?;
    let cell_zones = write_cell_zones(&poly_mesh_dir.join("cellZones"), mesh)?;
    write_minimal_system_files(case_dir)?;

    let summary = FoamWriteSummary {
        case_dir: case_dir.to_path_buf(),
        points: mesh.points.len(),
        cells: mesh.cells.len(),
        faces: topology.faces.len(),
        internal_faces: ordered.internal_face_indices.len(),
        boundary_faces: ordered.face_indices.len() - ordered.internal_face_indices.len(),
        patches: ordered.patches,
        face_zones,
        cell_zones,
        unmatched_boundary_faces: topology.unmatched_boundary_faces,
        duplicate_boundary_faces: topology.duplicate_boundary_faces,
        non_manifold_faces: topology.non_manifold_faces,
    };
    write_summary(
        &case_dir.join("constant").join("ferrumMeshSummary.txt"),
        &summary,
        mesh,
        source_path,
    )?;
    Ok(summary)
}

fn build_topology(mesh: &Mesh) -> Topology {
    let mut boundary_by_key = HashMap::<FaceKey, BoundaryPatchRef>::new();
    let mut duplicate_boundary_faces = 0;

    for face in &mesh.boundary_faces {
        let key = FaceKey::from_nodes(&face.nodes);
        let patch_name = sanitize_name(&mesh.physical_name(face.physical_tag));
        if boundary_by_key
            .insert(
                key,
                BoundaryPatchRef {
                    name: patch_name,
                    physical_tag: Some(face.physical_tag),
                },
            )
            .is_some()
        {
            duplicate_boundary_faces += 1;
        }
    }

    let mut face_index_by_key = HashMap::<FaceKey, usize>::new();
    let mut faces = Vec::<FaceRecord>::new();
    let mut non_manifold_faces = 0;

    for (cell_index, cell) in mesh.cells.iter().enumerate() {
        for nodes in cell_faces(&cell.nodes) {
            let key = FaceKey::from_nodes(&nodes);
            if let Some(&face_index) = face_index_by_key.get(&key) {
                let face = &mut faces[face_index];
                if face.neighbour.is_some() {
                    non_manifold_faces += 1;
                } else {
                    face.neighbour = Some(cell_index);
                }
            } else {
                face_index_by_key.insert(key.clone(), faces.len());
                faces.push(FaceRecord {
                    nodes,
                    key,
                    owner: cell_index,
                    neighbour: None,
                    patch: None,
                });
            }
        }
    }

    let topology_face_keys = faces
        .iter()
        .map(|face| face.key.clone())
        .collect::<HashSet<_>>();
    let mut unmatched_boundary_faces = 0;
    for key in boundary_by_key.keys() {
        if !topology_face_keys.contains(key) {
            unmatched_boundary_faces += 1;
        }
    }

    for face in &mut faces {
        if face.neighbour.is_none() {
            face.patch = boundary_by_key.get(&face.key).cloned();
            if face.patch.is_none() {
                face.patch = Some(BoundaryPatchRef {
                    name: "defaultFaces".to_string(),
                    physical_tag: None,
                });
            }
        }
    }

    Topology {
        faces,
        unmatched_boundary_faces,
        duplicate_boundary_faces,
        non_manifold_faces,
    }
}

fn order_faces(mesh: &Mesh, faces: &[FaceRecord], options: &FoamWriteOptions) -> OrderedFaces {
    let mut internal_face_indices = Vec::new();
    let mut boundary_by_patch = BTreeMap::<String, Vec<usize>>::new();
    let mut patch_tags = HashMap::<String, Option<i32>>::new();

    for (index, face) in faces.iter().enumerate() {
        if face.neighbour.is_some() {
            internal_face_indices.push(index);
        } else if let Some(patch) = &face.patch {
            boundary_by_patch
                .entry(patch.name.clone())
                .or_default()
                .push(index);
            patch_tags
                .entry(patch.name.clone())
                .or_insert(patch.physical_tag);
        }
    }

    let mut patch_order = Vec::<String>::new();
    for physical in mesh.physical_names.iter().filter(|name| name.dim == 2) {
        let name = sanitize_name(&physical.name);
        if boundary_by_patch.contains_key(&name) && !patch_order.contains(&name) {
            patch_order.push(name);
        }
    }
    for name in boundary_by_patch.keys() {
        if !patch_order.contains(name) {
            patch_order.push(name.clone());
        }
    }

    let mut face_indices = internal_face_indices.clone();
    let mut ordered_label_by_topology_index = HashMap::new();
    for (ordered_label, topology_index) in face_indices.iter().copied().enumerate() {
        ordered_label_by_topology_index.insert(topology_index, ordered_label);
    }
    let mut patches = Vec::new();
    for name in patch_order {
        let start_face = face_indices.len();
        let patch_faces = boundary_by_patch.remove(&name).unwrap_or_default();
        let face_count = patch_faces.len();
        for topology_index in patch_faces {
            let ordered_label = face_indices.len();
            face_indices.push(topology_index);
            ordered_label_by_topology_index.insert(topology_index, ordered_label);
        }
        patches.push(PatchSummary {
            name: name.clone(),
            patch_type: options.patch_type_for(&name),
            physical_tag: patch_tags.get(&name).copied().flatten(),
            faces: face_count,
            start_face,
        });
    }

    OrderedFaces {
        face_indices,
        internal_face_indices,
        ordered_label_by_topology_index,
        patches,
    }
}

fn write_points(path: &Path, mesh: &Mesh) -> Result<()> {
    let mut writer = foam_writer(path, "vectorField", "points")?;
    writeln!(writer, "{}", mesh.points.len())?;
    writeln!(writer, "(")?;
    for point in &mesh.points {
        writeln!(writer, "({:.16} {:.16} {:.16})", point.x, point.y, point.z)?;
    }
    writeln!(writer, ")")?;
    Ok(())
}

fn write_faces(path: &Path, faces: &[FaceRecord], ordered: &[usize]) -> Result<()> {
    let mut writer = foam_writer(path, "faceList", "faces")?;
    writeln!(writer, "{}", ordered.len())?;
    writeln!(writer, "(")?;
    for &index in ordered {
        let nodes = &faces[index].nodes;
        write!(writer, "{}(", nodes.len())?;
        for (node_index, node) in nodes.iter().enumerate() {
            if node_index > 0 {
                write!(writer, " ")?;
            }
            write!(writer, "{node}")?;
        }
        writeln!(writer, ")")?;
    }
    writeln!(writer, ")")?;
    Ok(())
}

fn write_owner(path: &Path, faces: &[FaceRecord], ordered: &[usize]) -> Result<()> {
    let mut writer = foam_writer(path, "labelList", "owner")?;
    writeln!(writer, "{}", ordered.len())?;
    writeln!(writer, "(")?;
    for &index in ordered {
        writeln!(writer, "{}", faces[index].owner)?;
    }
    writeln!(writer, ")")?;
    Ok(())
}

fn write_neighbour(path: &Path, faces: &[FaceRecord], internal: &[usize]) -> Result<()> {
    let mut writer = foam_writer(path, "labelList", "neighbour")?;
    writeln!(writer, "{}", internal.len())?;
    writeln!(writer, "(")?;
    for &index in internal {
        if let Some(neighbour) = faces[index].neighbour {
            writeln!(writer, "{neighbour}")?;
        }
    }
    writeln!(writer, ")")?;
    Ok(())
}

fn write_boundary(path: &Path, patches: &[PatchSummary]) -> Result<()> {
    let mut writer = foam_writer(path, "polyBoundaryMesh", "boundary")?;
    writeln!(writer, "{}", patches.len())?;
    writeln!(writer, "(")?;
    for patch in patches {
        writeln!(writer, "    {}", patch.name)?;
        writeln!(writer, "    {{")?;
        writeln!(writer, "        type {};", patch.patch_type)?;
        writeln!(writer, "        nFaces {};", patch.faces)?;
        writeln!(writer, "        startFace {};", patch.start_face)?;
        writeln!(writer, "    }}")?;
    }
    writeln!(writer, ")")?;
    Ok(())
}

fn write_face_zones(
    path: &Path,
    mesh: &Mesh,
    faces: &[FaceRecord],
    ordered: &OrderedFaces,
) -> Result<Vec<FaceZoneSummary>> {
    let mut topology_index_by_key = HashMap::<FaceKey, usize>::new();
    for (index, face) in faces.iter().enumerate() {
        topology_index_by_key.insert(face.key.clone(), index);
    }

    let mut zones = BTreeMap::<i32, Vec<usize>>::new();
    for face in &mesh.boundary_faces {
        let key = FaceKey::from_nodes(&face.nodes);
        if let Some(&topology_index) = topology_index_by_key.get(&key)
            && let Some(&ordered_index) =
                ordered.ordered_label_by_topology_index.get(&topology_index)
        {
            zones
                .entry(face.physical_tag)
                .or_default()
                .push(ordered_index);
        }
    }

    let mut writer = foam_writer(path, "faceZoneMesh", "faceZones")?;
    writeln!(writer, "{}", zones.len())?;
    writeln!(writer, "(")?;

    let mut summaries = Vec::new();
    for (tag, face_labels) in zones {
        let name = sanitize_name(&mesh.physical_name(tag));
        writeln!(writer, "    {name}")?;
        writeln!(writer, "    {{")?;
        writeln!(writer, "        type faceZone;")?;
        writeln!(writer, "        faceLabels {}", face_labels.len())?;
        writeln!(writer, "        (")?;
        for face_label in &face_labels {
            writeln!(writer, "            {face_label}")?;
        }
        writeln!(writer, "        );")?;
        writeln!(writer, "        flipMap {}", face_labels.len())?;
        writeln!(writer, "        (")?;
        for _ in &face_labels {
            writeln!(writer, "            false")?;
        }
        writeln!(writer, "        );")?;
        writeln!(writer, "    }}")?;

        summaries.push(FaceZoneSummary {
            name,
            physical_tag: tag,
            faces: face_labels.len(),
        });
    }

    writeln!(writer, ")")?;
    Ok(summaries)
}

fn write_cell_zones(path: &Path, mesh: &Mesh) -> Result<Vec<CellZoneSummary>> {
    let mut zones = BTreeMap::<i32, Vec<usize>>::new();
    for (cell_index, cell) in mesh.cells.iter().enumerate() {
        zones.entry(cell.physical_tag).or_default().push(cell_index);
    }

    let mut writer = foam_writer(path, "cellZoneMesh", "cellZones")?;
    writeln!(writer, "{}", zones.len())?;
    writeln!(writer, "(")?;

    let mut summaries = Vec::new();
    for (tag, cells) in zones {
        let name = sanitize_name(&mesh.physical_name(tag));
        writeln!(writer, "    {name}")?;
        writeln!(writer, "    {{")?;
        writeln!(writer, "        type cellZone;")?;
        writeln!(writer, "        cellLabels {}", cells.len())?;
        writeln!(writer, "        (")?;
        for cell in &cells {
            writeln!(writer, "            {cell}")?;
        }
        writeln!(writer, "        );")?;
        writeln!(writer, "    }}")?;

        summaries.push(CellZoneSummary {
            name,
            physical_tag: tag,
            cells: cells.len(),
        });
    }

    writeln!(writer, ")")?;
    Ok(summaries)
}

fn write_minimal_system_files(case_dir: &Path) -> Result<()> {
    let system_dir = case_dir.join("system");

    let mut control = foam_writer_at(
        &system_dir.join("controlDict"),
        "dictionary",
        "controlDict",
        "system",
    )?;
    writeln!(control, "application ferrum;")?;
    writeln!(control, "startFrom startTime;")?;
    writeln!(control, "startTime 0;")?;
    writeln!(control, "endTime 1;")?;
    writeln!(control, "deltaT 1;")?;
    writeln!(control, "writeControl timeStep;")?;
    writeln!(control, "writeInterval 1;")?;

    let mut schemes = foam_writer_at(
        &system_dir.join("fvSchemes"),
        "dictionary",
        "fvSchemes",
        "system",
    )?;
    writeln!(schemes, "ddtSchemes {{ default Euler; }}")?;
    writeln!(schemes, "gradSchemes {{ default Gauss linear; }}")?;
    writeln!(schemes, "divSchemes {{ default none; }}")?;
    writeln!(
        schemes,
        "laplacianSchemes {{ default Gauss linear corrected; }}"
    )?;
    writeln!(schemes, "interpolationSchemes {{ default linear; }}")?;
    writeln!(schemes, "snGradSchemes {{ default corrected; }}")?;

    let mut solution = foam_writer_at(
        &system_dir.join("fvSolution"),
        "dictionary",
        "fvSolution",
        "system",
    )?;
    writeln!(solution, "solvers {{ }}")?;
    writeln!(solution, "SIMPLE {{ nNonOrthogonalCorrectors 0; }}")?;
    Ok(())
}

fn write_summary(
    path: &Path,
    summary: &FoamWriteSummary,
    mesh: &Mesh,
    source_path: &Path,
) -> Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(writer, "FerrumCFD mesh summary")?;
    writeln!(writer, "source={}", source_path.display())?;
    writeln!(writer, "case={}", summary.case_dir.display())?;
    writeln!(writer, "points={}", summary.points)?;
    writeln!(writer, "cells={}", summary.cells)?;
    writeln!(writer, "faces={}", summary.faces)?;
    writeln!(writer, "internal_faces={}", summary.internal_faces)?;
    writeln!(writer, "boundary_faces={}", summary.boundary_faces)?;
    writeln!(
        writer,
        "unmatched_boundary_faces={}",
        summary.unmatched_boundary_faces
    )?;
    writeln!(
        writer,
        "duplicate_boundary_faces={}",
        summary.duplicate_boundary_faces
    )?;
    writeln!(writer, "non_manifold_faces={}", summary.non_manifold_faces)?;
    writeln!(writer)?;
    writeln!(writer, "[patches]")?;
    for patch in &summary.patches {
        let tag = patch
            .physical_tag
            .map(|tag| tag.to_string())
            .unwrap_or_else(|| "-".to_string());
        writeln!(
            writer,
            "{} type={} tag={} faces={} startFace={}",
            patch.name, patch.patch_type, tag, patch.faces, patch.start_face
        )?;
    }
    writeln!(writer)?;
    writeln!(writer, "[face_zones]")?;
    for zone in &summary.face_zones {
        writeln!(
            writer,
            "{} tag={} faces={}",
            zone.name, zone.physical_tag, zone.faces
        )?;
    }
    writeln!(writer)?;
    writeln!(writer, "[cell_zones]")?;
    for zone in &summary.cell_zones {
        writeln!(
            writer,
            "{} tag={} cells={}",
            zone.name, zone.physical_tag, zone.cells
        )?;
    }
    if !mesh.unsupported_elements.is_empty() {
        writeln!(writer)?;
        writeln!(writer, "[unsupported_elements]")?;
        for unsupported in &mesh.unsupported_elements {
            writeln!(
                writer,
                "type={} count={}",
                unsupported.element_type, unsupported.count
            )?;
        }
    }
    Ok(())
}

fn foam_writer(path: &Path, class_name: &str, object: &str) -> Result<BufWriter<File>> {
    foam_writer_at(path, class_name, object, "constant/polyMesh")
}

fn foam_writer_at(
    path: &Path,
    class_name: &str,
    object: &str,
    location: &str,
) -> Result<BufWriter<File>> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(writer, "FoamFile")?;
    writeln!(writer, "{{")?;
    writeln!(writer, "    version 2.0;")?;
    writeln!(writer, "    format ascii;")?;
    writeln!(writer, "    class {class_name};")?;
    writeln!(writer, "    location \"{location}\";")?;
    writeln!(writer, "    object {object};")?;
    writeln!(writer, "}}")?;
    writeln!(writer)?;
    Ok(writer)
}

fn cell_faces(nodes: &[usize]) -> Vec<Vec<usize>> {
    match nodes.len() {
        6 => prism_faces(nodes),
        8 => hex_faces(nodes),
        _ => Vec::new(),
    }
}

fn hex_faces(nodes: &[usize]) -> Vec<Vec<usize>> {
    vec![
        vec![nodes[0], nodes[3], nodes[2], nodes[1]],
        vec![nodes[4], nodes[5], nodes[6], nodes[7]],
        vec![nodes[0], nodes[1], nodes[5], nodes[4]],
        vec![nodes[1], nodes[2], nodes[6], nodes[5]],
        vec![nodes[2], nodes[3], nodes[7], nodes[6]],
        vec![nodes[3], nodes[0], nodes[4], nodes[7]],
    ]
}

fn prism_faces(nodes: &[usize]) -> Vec<Vec<usize>> {
    vec![
        vec![nodes[0], nodes[2], nodes[1]],
        vec![nodes[3], nodes[4], nodes[5]],
        vec![nodes[0], nodes[1], nodes[4], nodes[3]],
        vec![nodes[1], nodes[2], nodes[5], nodes[4]],
        vec![nodes[2], nodes[0], nodes[3], nodes[5]],
    ]
}

fn sanitize_name(name: &str) -> String {
    let sanitized: String = name
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' {
                ch
            } else {
                '_'
            }
        })
        .collect();

    if sanitized.is_empty() {
        "unnamed".to_string()
    } else {
        sanitized
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct FaceKey(Vec<usize>);

impl FaceKey {
    fn from_nodes(nodes: &[usize]) -> Self {
        let mut sorted = nodes.to_vec();
        sorted.sort_unstable();
        Self(sorted)
    }
}

#[derive(Clone, Debug)]
struct BoundaryPatchRef {
    name: String,
    physical_tag: Option<i32>,
}

#[derive(Clone, Debug)]
struct FaceRecord {
    nodes: Vec<usize>,
    key: FaceKey,
    owner: usize,
    neighbour: Option<usize>,
    patch: Option<BoundaryPatchRef>,
}

struct Topology {
    faces: Vec<FaceRecord>,
    unmatched_boundary_faces: usize,
    duplicate_boundary_faces: usize,
    non_manifold_faces: usize,
}

struct OrderedFaces {
    face_indices: Vec<usize>,
    internal_face_indices: Vec<usize>,
    ordered_label_by_topology_index: HashMap<usize, usize>,
    patches: Vec<PatchSummary>,
}
