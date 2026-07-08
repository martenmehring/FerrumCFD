use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use crate::{MeshError, Point3, Result};

#[derive(Debug)]
pub struct RegionSplitSummary {
    pub case_dir: PathBuf,
    pub regions: Vec<RegionSummary>,
}

#[derive(Debug)]
pub struct RegionSummary {
    pub name: String,
    pub path: PathBuf,
    pub points: usize,
    pub cells: usize,
    pub faces: usize,
    pub internal_faces: usize,
    pub boundary_faces: usize,
    pub patches: Vec<RegionPatchSummary>,
}

#[derive(Debug)]
pub struct RegionPatchSummary {
    pub name: String,
    pub patch_type: String,
    pub faces: usize,
    pub start_face: usize,
    pub source_flipped_faces: usize,
}

pub fn split_regions_by_cell_zones(case_dir: &Path) -> Result<RegionSplitSummary> {
    let poly_mesh_dir = case_dir.join("constant").join("polyMesh");
    let mesh = PolyMesh::read(&poly_mesh_dir)?;

    if mesh.cell_zones.is_empty() {
        return Err(MeshError::InvalidInput(format!(
            "no cellZones found in {}",
            poly_mesh_dir.join("cellZones").display()
        )));
    }

    let cell_count = mesh.cell_count();
    let cell_to_zone = build_cell_to_zone(&mesh.cell_zones, cell_count)?;
    let face_zone_by_face = build_face_zone_index(&mesh.face_zones);
    let boundary_by_face = build_boundary_index(&mesh.patches);

    let mut summaries = Vec::new();
    for (zone_index, zone) in mesh.cell_zones.iter().enumerate() {
        let region = build_region_mesh(
            &mesh,
            zone_index,
            zone,
            &cell_to_zone,
            &face_zone_by_face,
            &boundary_by_face,
        )?;
        let region_dir = case_dir
            .join("constant")
            .join(sanitize_name(&zone.name))
            .join("polyMesh");
        fs::create_dir_all(&region_dir)?;
        write_region_poly_mesh(&region_dir, &region)?;

        summaries.push(RegionSummary {
            name: zone.name.clone(),
            path: region_dir,
            points: region.points.len(),
            cells: zone.cells.len(),
            faces: region.faces.len(),
            internal_faces: region.internal_faces,
            boundary_faces: region.faces.len() - region.internal_faces,
            patches: region
                .patches
                .iter()
                .map(|patch| RegionPatchSummary {
                    name: patch.name.clone(),
                    patch_type: patch.patch_type.clone(),
                    faces: patch.faces.len(),
                    start_face: patch.start_face,
                    source_flipped_faces: patch.source_flipped_faces,
                })
                .collect(),
        });
    }

    Ok(RegionSplitSummary {
        case_dir: case_dir.to_path_buf(),
        regions: summaries,
    })
}

pub fn read_region_mesh_summaries(case_dir: &Path) -> Result<Vec<RegionSummary>> {
    let constant_dir = case_dir.join("constant");
    if !constant_dir.exists() {
        return Ok(Vec::new());
    }

    let mut summaries = Vec::new();
    for entry in fs::read_dir(&constant_dir)? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }

        let region_name = entry.file_name().to_string_lossy().to_string();
        if region_name == "polyMesh" {
            continue;
        }

        let poly_mesh_dir = path.join("polyMesh");
        if !poly_mesh_dir.is_dir() {
            continue;
        }

        let mesh = PolyMesh::read(&poly_mesh_dir)?;
        summaries.push(summarize_poly_mesh(region_name, poly_mesh_dir, &mesh));
    }

    summaries.sort_by(|left, right| left.name.cmp(&right.name));
    Ok(summaries)
}

fn summarize_poly_mesh(name: String, path: PathBuf, mesh: &PolyMesh) -> RegionSummary {
    RegionSummary {
        name,
        path,
        points: mesh.points.len(),
        cells: mesh.cell_count(),
        faces: mesh.faces.len(),
        internal_faces: mesh.neighbour.len(),
        boundary_faces: mesh.faces.len().saturating_sub(mesh.neighbour.len()),
        patches: mesh
            .patches
            .iter()
            .map(|patch| RegionPatchSummary {
                name: patch.name.clone(),
                patch_type: patch.patch_type.clone(),
                faces: patch.faces,
                start_face: patch.start_face,
                source_flipped_faces: 0,
            })
            .collect(),
    }
}

fn build_cell_to_zone(cell_zones: &[CellZone], cell_count: usize) -> Result<Vec<Option<usize>>> {
    let mut cell_to_zone: Vec<Option<usize>> = vec![None; cell_count];
    for (zone_index, zone) in cell_zones.iter().enumerate() {
        for &cell in &zone.cells {
            let slot = cell_to_zone.get_mut(cell).ok_or_else(|| {
                MeshError::InvalidInput(format!(
                    "cellZone '{}' references missing cell {}",
                    zone.name, cell
                ))
            })?;
            if let Some(existing) = *slot {
                return Err(MeshError::InvalidInput(format!(
                    "cell {} appears in both '{}' and '{}'",
                    cell, cell_zones[existing].name, zone.name
                )));
            }
            *slot = Some(zone_index);
        }
    }
    Ok(cell_to_zone)
}

fn build_face_zone_index(face_zones: &[FaceZone]) -> HashMap<usize, FaceZoneRef> {
    let mut by_face = HashMap::new();
    for zone in face_zones {
        for entry in &zone.faces {
            by_face.entry(entry.face).or_insert_with(|| FaceZoneRef {
                name: zone.name.clone(),
                flip: entry.flip,
            });
        }
    }
    by_face
}

fn build_boundary_index(patches: &[BoundaryPatch]) -> HashMap<usize, BoundaryPatchRef> {
    let mut by_face = HashMap::new();
    for patch in patches {
        for face in patch.start_face..patch.start_face + patch.faces {
            by_face.insert(
                face,
                BoundaryPatchRef {
                    name: patch.name.clone(),
                    patch_type: patch.patch_type.clone(),
                    face_zone_flip: None,
                },
            );
        }
    }
    by_face
}

fn build_region_mesh(
    mesh: &PolyMesh,
    zone_index: usize,
    zone: &CellZone,
    cell_to_zone: &[Option<usize>],
    face_zone_by_face: &HashMap<usize, FaceZoneRef>,
    boundary_by_face: &HashMap<usize, BoundaryPatchRef>,
) -> Result<RegionMesh> {
    let cell_map = zone
        .cells
        .iter()
        .enumerate()
        .map(|(local, &global)| (global, local))
        .collect::<HashMap<_, _>>();

    let mut point_map = HashMap::<usize, usize>::new();
    let mut points = Vec::<Point3>::new();
    let mut internal_faces = Vec::<RegionFace>::new();
    let mut boundary = BoundaryAccumulator::default();

    for face_index in 0..mesh.faces.len() {
        let owner = mesh.owner[face_index];
        let neighbour = mesh.neighbour.get(face_index).copied();
        let owner_local = cell_map.get(&owner).copied();
        let neighbour_local = neighbour.and_then(|cell| cell_map.get(&cell).copied());

        match (owner_local, neighbour_local) {
            (Some(owner), Some(neighbour)) => {
                internal_faces.push(RegionFace {
                    nodes: map_nodes(
                        &mesh.faces[face_index],
                        &mesh.points,
                        &mut point_map,
                        &mut points,
                    )?,
                    owner,
                    neighbour: Some(neighbour),
                });
            }
            (Some(owner), None) => {
                let patch = region_patch_for_face(
                    mesh,
                    zone_index,
                    neighbour.and_then(|cell| cell_to_zone.get(cell).copied().flatten()),
                    face_index,
                    face_zone_by_face,
                    boundary_by_face,
                );
                boundary.push(
                    patch,
                    RegionFace {
                        nodes: map_nodes(
                            &mesh.faces[face_index],
                            &mesh.points,
                            &mut point_map,
                            &mut points,
                        )?,
                        owner,
                        neighbour: None,
                    },
                );
            }
            (None, Some(owner)) => {
                let patch = region_patch_for_face(
                    mesh,
                    zone_index,
                    Some(cell_to_zone[mesh.owner[face_index]].unwrap_or(zone_index)),
                    face_index,
                    face_zone_by_face,
                    boundary_by_face,
                );
                let mut reversed_nodes = mesh.faces[face_index].clone();
                reversed_nodes.reverse();
                boundary.push(
                    patch,
                    RegionFace {
                        nodes: map_nodes(
                            &reversed_nodes,
                            &mesh.points,
                            &mut point_map,
                            &mut points,
                        )?,
                        owner,
                        neighbour: None,
                    },
                );
            }
            (None, None) => {}
        }
    }

    let mut faces = internal_faces;
    let internal_face_count = faces.len();
    let patches = boundary.into_patches(internal_face_count, &mut faces);

    Ok(RegionMesh {
        points,
        faces,
        internal_faces: internal_face_count,
        patches,
    })
}

fn region_patch_for_face(
    mesh: &PolyMesh,
    zone_index: usize,
    other_zone: Option<usize>,
    face_index: usize,
    face_zone_by_face: &HashMap<usize, FaceZoneRef>,
    boundary_by_face: &HashMap<usize, BoundaryPatchRef>,
) -> BoundaryPatchRef {
    if let Some(patch) = boundary_by_face.get(&face_index) {
        let mut patch = patch.clone();
        if patch.face_zone_flip.is_none() {
            patch.face_zone_flip = face_zone_by_face.get(&face_index).map(|zone| zone.flip);
        }
        return patch;
    }

    if let Some(face_zone) = face_zone_by_face.get(&face_index) {
        return BoundaryPatchRef {
            name: face_zone.name.clone(),
            patch_type: "patch".to_string(),
            face_zone_flip: Some(face_zone.flip),
        };
    }

    let zone_name = sanitize_name(&mesh.cell_zones[zone_index].name);
    let other_name = other_zone
        .and_then(|index| mesh.cell_zones.get(index))
        .map(|zone| sanitize_name(&zone.name))
        .unwrap_or_else(|| "unknown".to_string());
    BoundaryPatchRef {
        name: format!("interface_{zone_name}_to_{other_name}"),
        patch_type: "patch".to_string(),
        face_zone_flip: None,
    }
}

fn map_nodes(
    nodes: &[usize],
    source_points: &[Point3],
    point_map: &mut HashMap<usize, usize>,
    points: &mut Vec<Point3>,
) -> Result<Vec<usize>> {
    nodes
        .iter()
        .map(|&node| {
            if let Some(&local) = point_map.get(&node) {
                return Ok(local);
            }
            let point = *source_points.get(node).ok_or_else(|| {
                MeshError::InvalidInput(format!("face references missing point {node}"))
            })?;
            let local = points.len();
            points.push(point);
            point_map.insert(node, local);
            Ok(local)
        })
        .collect()
}

fn write_region_poly_mesh(path: &Path, region: &RegionMesh) -> Result<()> {
    write_points(&path.join("points"), &region.points)?;
    write_faces(&path.join("faces"), &region.faces)?;
    write_owner(&path.join("owner"), &region.faces)?;
    write_neighbour(
        &path.join("neighbour"),
        &region.faces,
        region.internal_faces,
    )?;
    write_boundary(&path.join("boundary"), &region.patches)?;
    write_empty_zone_file(&path.join("faceZones"), "faceZoneMesh", "faceZones")?;
    write_empty_zone_file(&path.join("cellZones"), "cellZoneMesh", "cellZones")?;
    Ok(())
}

fn write_points(path: &Path, points: &[Point3]) -> Result<()> {
    let mut writer = foam_writer(path, "vectorField", "points")?;
    writeln!(writer, "{}", points.len())?;
    writeln!(writer, "(")?;
    for point in points {
        writeln!(writer, "({:.16} {:.16} {:.16})", point.x, point.y, point.z)?;
    }
    writeln!(writer, ")")?;
    Ok(())
}

fn write_faces(path: &Path, faces: &[RegionFace]) -> Result<()> {
    let mut writer = foam_writer(path, "faceList", "faces")?;
    writeln!(writer, "{}", faces.len())?;
    writeln!(writer, "(")?;
    for face in faces {
        write!(writer, "{}(", face.nodes.len())?;
        for (index, node) in face.nodes.iter().enumerate() {
            if index > 0 {
                write!(writer, " ")?;
            }
            write!(writer, "{node}")?;
        }
        writeln!(writer, ")")?;
    }
    writeln!(writer, ")")?;
    Ok(())
}

fn write_owner(path: &Path, faces: &[RegionFace]) -> Result<()> {
    let mut writer = foam_writer(path, "labelList", "owner")?;
    writeln!(writer, "{}", faces.len())?;
    writeln!(writer, "(")?;
    for face in faces {
        writeln!(writer, "{}", face.owner)?;
    }
    writeln!(writer, ")")?;
    Ok(())
}

fn write_neighbour(path: &Path, faces: &[RegionFace], internal_faces: usize) -> Result<()> {
    let mut writer = foam_writer(path, "labelList", "neighbour")?;
    writeln!(writer, "{internal_faces}")?;
    writeln!(writer, "(")?;
    for face in faces.iter().take(internal_faces) {
        if let Some(neighbour) = face.neighbour {
            writeln!(writer, "{neighbour}")?;
        }
    }
    writeln!(writer, ")")?;
    Ok(())
}

fn write_boundary(path: &Path, patches: &[RegionPatch]) -> Result<()> {
    let mut writer = foam_writer(path, "polyBoundaryMesh", "boundary")?;
    writeln!(writer, "{}", patches.len())?;
    writeln!(writer, "(")?;
    for patch in patches {
        writeln!(writer, "    {}", patch.name)?;
        writeln!(writer, "    {{")?;
        writeln!(writer, "        type {};", patch.patch_type)?;
        writeln!(writer, "        nFaces {};", patch.faces.len())?;
        writeln!(writer, "        startFace {};", patch.start_face)?;
        writeln!(writer, "    }}")?;
    }
    writeln!(writer, ")")?;
    Ok(())
}

fn write_empty_zone_file(path: &Path, class_name: &str, object: &str) -> Result<()> {
    let mut writer = foam_writer(path, class_name, object)?;
    writeln!(writer, "0")?;
    writeln!(writer, "(")?;
    writeln!(writer, ")")?;
    Ok(())
}

fn foam_writer(path: &Path, class_name: &str, object: &str) -> Result<BufWriter<File>> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(writer, "FoamFile")?;
    writeln!(writer, "{{")?;
    writeln!(writer, "    version 2.0;")?;
    writeln!(writer, "    format ascii;")?;
    writeln!(writer, "    class {class_name};")?;
    writeln!(writer, "    location \"constant/polyMesh\";")?;
    writeln!(writer, "    object {object};")?;
    writeln!(writer, "}}")?;
    writeln!(writer)?;
    Ok(writer)
}

#[derive(Default)]
struct BoundaryAccumulator {
    patches: Vec<RegionPatch>,
    by_name: HashMap<String, usize>,
}

impl BoundaryAccumulator {
    fn push(&mut self, patch_ref: BoundaryPatchRef, face: RegionFace) {
        let key = patch_ref.name.clone();
        let index = if let Some(&index) = self.by_name.get(&key) {
            index
        } else {
            let index = self.patches.len();
            self.patches.push(RegionPatch {
                name: patch_ref.name.clone(),
                patch_type: patch_ref.patch_type,
                start_face: 0,
                source_flipped_faces: 0,
                faces: Vec::new(),
            });
            self.by_name.insert(key, index);
            index
        };
        if patch_ref.face_zone_flip.unwrap_or(false) {
            self.patches[index].source_flipped_faces += 1;
        }
        self.patches[index].faces.push(face);
    }

    fn into_patches(
        mut self,
        internal_face_count: usize,
        faces: &mut Vec<RegionFace>,
    ) -> Vec<RegionPatch> {
        let mut start_face = internal_face_count;
        for patch in &mut self.patches {
            patch.start_face = start_face;
            start_face += patch.faces.len();
            faces.extend(patch.faces.iter().cloned());
        }
        self.patches
    }
}

#[derive(Clone)]
struct RegionMesh {
    points: Vec<Point3>,
    faces: Vec<RegionFace>,
    internal_faces: usize,
    patches: Vec<RegionPatch>,
}

#[derive(Clone)]
struct RegionFace {
    nodes: Vec<usize>,
    owner: usize,
    neighbour: Option<usize>,
}

#[derive(Clone)]
struct RegionPatch {
    name: String,
    patch_type: String,
    start_face: usize,
    source_flipped_faces: usize,
    faces: Vec<RegionFace>,
}

#[derive(Clone)]
struct BoundaryPatchRef {
    name: String,
    patch_type: String,
    face_zone_flip: Option<bool>,
}

#[derive(Clone)]
struct FaceZoneRef {
    name: String,
    flip: bool,
}

struct PolyMesh {
    points: Vec<Point3>,
    faces: Vec<Vec<usize>>,
    owner: Vec<usize>,
    neighbour: Vec<usize>,
    patches: Vec<BoundaryPatch>,
    cell_zones: Vec<CellZone>,
    face_zones: Vec<FaceZone>,
}

impl PolyMesh {
    fn read(path: &Path) -> Result<Self> {
        let points = read_points(&path.join("points"))?;
        let faces = read_faces(&path.join("faces"))?;
        let owner = read_label_list(&path.join("owner"))?;
        let neighbour = read_label_list(&path.join("neighbour"))?;
        let patches = read_boundary(&path.join("boundary"))?;
        let cell_zones = read_cell_zones(&path.join("cellZones"))?;
        let face_zones = read_face_zones(&path.join("faceZones"))?;

        if faces.len() != owner.len() {
            return Err(MeshError::InvalidInput(format!(
                "faces/owner size mismatch in {}",
                path.display()
            )));
        }
        if neighbour.len() > faces.len() {
            return Err(MeshError::InvalidInput(format!(
                "neighbour list is longer than face list in {}",
                path.display()
            )));
        }

        Ok(Self {
            points,
            faces,
            owner,
            neighbour,
            patches,
            cell_zones,
            face_zones,
        })
    }

    fn cell_count(&self) -> usize {
        self.owner
            .iter()
            .chain(self.neighbour.iter())
            .copied()
            .max()
            .map(|cell| cell + 1)
            .unwrap_or(0)
    }
}

struct BoundaryPatch {
    name: String,
    patch_type: String,
    faces: usize,
    start_face: usize,
}

struct CellZone {
    name: String,
    cells: Vec<usize>,
}

struct FaceZone {
    name: String,
    faces: Vec<FaceZoneEntry>,
}

struct FaceZoneEntry {
    face: usize,
    flip: bool,
}

fn read_points(path: &Path) -> Result<Vec<Point3>> {
    read_list_entries(path)?
        .into_iter()
        .map(|line| {
            let values = strip_wrapping_parens(&line)
                .split_whitespace()
                .map(str::parse::<f64>)
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(|_| {
                    MeshError::InvalidInput(format!("invalid point in {}", path.display()))
                })?;
            if values.len() != 3 {
                return Err(MeshError::InvalidInput(format!(
                    "point does not have 3 coordinates in {}",
                    path.display()
                )));
            }
            Ok(Point3 {
                x: values[0],
                y: values[1],
                z: values[2],
            })
        })
        .collect()
}

fn read_faces(path: &Path) -> Result<Vec<Vec<usize>>> {
    read_list_entries(path)?
        .into_iter()
        .map(|line| parse_face(&line, path))
        .collect()
}

fn read_label_list(path: &Path) -> Result<Vec<usize>> {
    read_list_entries(path)?
        .into_iter()
        .map(|line| {
            line.parse::<usize>().map_err(|_| {
                MeshError::InvalidInput(format!("invalid label '{}' in {}", line, path.display()))
            })
        })
        .collect()
}

fn read_boundary(path: &Path) -> Result<Vec<BoundaryPatch>> {
    let lines = clean_lines(path)?;
    let mut cursor = DictCursor::after_count_and_open(path, lines)?;
    let mut patches = Vec::new();

    while let Some(name) = cursor.next_entry_name()? {
        cursor.expect("{")?;
        let mut patch_type = None;
        let mut faces = None;
        let mut start_face = None;
        while !cursor.peek_is("}")? {
            let line = cursor.next_required()?;
            if let Some(value) = parse_dict_value(&line, "type") {
                patch_type = Some(value.to_string());
            } else if let Some(value) = parse_dict_value(&line, "nFaces") {
                faces = Some(parse_usize(value, path)?);
            } else if let Some(value) = parse_dict_value(&line, "startFace") {
                start_face = Some(parse_usize(value, path)?);
            }
        }
        cursor.expect("}")?;
        patches.push(BoundaryPatch {
            name,
            patch_type: patch_type.unwrap_or_else(|| "patch".to_string()),
            faces: faces.ok_or_else(|| missing_key(path, "nFaces"))?,
            start_face: start_face.ok_or_else(|| missing_key(path, "startFace"))?,
        });
    }

    Ok(patches)
}

fn read_cell_zones(path: &Path) -> Result<Vec<CellZone>> {
    let lines = clean_lines(path)?;
    let mut cursor = DictCursor::after_count_and_open(path, lines)?;
    let mut zones = Vec::new();

    while let Some(name) = cursor.next_entry_name()? {
        cursor.expect("{")?;
        let mut cells = None;
        while !cursor.peek_is("}")? {
            let line = cursor.next_required()?;
            if line.starts_with("cellLabels ") {
                cells = Some(cursor.read_label_block()?);
            }
        }
        cursor.expect("}")?;
        zones.push(CellZone {
            name,
            cells: cells.ok_or_else(|| missing_key(path, "cellLabels"))?,
        });
    }

    Ok(zones)
}

fn read_face_zones(path: &Path) -> Result<Vec<FaceZone>> {
    let lines = clean_lines(path)?;
    let mut cursor = DictCursor::after_count_and_open(path, lines)?;
    let mut zones = Vec::new();

    while let Some(name) = cursor.next_entry_name()? {
        cursor.expect("{")?;
        let mut faces = None;
        let mut flip_map = None;
        while !cursor.peek_is("}")? {
            let line = cursor.next_required()?;
            if line.starts_with("faceLabels ") {
                faces = Some(cursor.read_label_block()?);
            } else if line.starts_with("flipMap ") {
                flip_map = Some(cursor.read_bool_block()?);
            }
        }
        cursor.expect("}")?;
        let faces = faces.ok_or_else(|| missing_key(path, "faceLabels"))?;
        let flip_map = flip_map.unwrap_or_else(|| vec![false; faces.len()]);
        if flip_map.len() != faces.len() {
            return Err(MeshError::InvalidInput(format!(
                "faceZone '{}' has {} faceLabels but {} flipMap entries in {}",
                name,
                faces.len(),
                flip_map.len(),
                path.display()
            )));
        }
        zones.push(FaceZone {
            name,
            faces: faces
                .into_iter()
                .zip(flip_map)
                .map(|(face, flip)| FaceZoneEntry { face, flip })
                .collect(),
        });
    }

    Ok(zones)
}

fn read_list_entries(path: &Path) -> Result<Vec<String>> {
    let lines = clean_lines(path)?;
    let mut index = lines
        .iter()
        .position(|line| line.parse::<usize>().is_ok())
        .ok_or_else(|| {
            MeshError::InvalidInput(format!("missing list count in {}", path.display()))
        })?;
    let count = parse_usize(&lines[index], path)?;
    index += 1;
    while index < lines.len() && lines[index] != "(" {
        index += 1;
    }
    if index == lines.len() {
        return Err(MeshError::InvalidInput(format!(
            "missing list opening '(' in {}",
            path.display()
        )));
    }
    index += 1;

    let mut entries = Vec::with_capacity(count);
    while index < lines.len() {
        let line = &lines[index];
        if line == ")" || line == ");" {
            break;
        }
        entries.push(line.clone());
        index += 1;
    }

    if entries.len() != count {
        return Err(MeshError::InvalidInput(format!(
            "expected {count} entries but found {} in {}",
            entries.len(),
            path.display()
        )));
    }
    Ok(entries)
}

fn parse_face(line: &str, path: &Path) -> Result<Vec<usize>> {
    let open = line.find('(').ok_or_else(|| {
        MeshError::InvalidInput(format!("invalid face '{}' in {}", line, path.display()))
    })?;
    let close = line.rfind(')').ok_or_else(|| {
        MeshError::InvalidInput(format!("invalid face '{}' in {}", line, path.display()))
    })?;
    let declared = parse_usize(&line[..open], path)?;
    let nodes = line[open + 1..close]
        .split_whitespace()
        .map(|value| parse_usize(value, path))
        .collect::<Result<Vec<_>>>()?;
    if nodes.len() != declared {
        return Err(MeshError::InvalidInput(format!(
            "face declares {declared} nodes but has {} in {}",
            nodes.len(),
            path.display()
        )));
    }
    Ok(nodes)
}

fn clean_lines(path: &Path) -> Result<Vec<String>> {
    let content = fs::read_to_string(path).map_err(|error| {
        MeshError::InvalidInput(format!("could not read {} ({error})", path.display()))
    })?;
    Ok(content
        .lines()
        .map(|line| line.split("//").next().unwrap_or("").trim().to_string())
        .filter(|line| !line.is_empty())
        .collect())
}

fn strip_wrapping_parens(line: &str) -> &str {
    line.trim()
        .trim_start_matches('(')
        .trim_end_matches(')')
        .trim()
}

fn parse_dict_value<'a>(line: &'a str, key: &str) -> Option<&'a str> {
    let rest = line.strip_prefix(key)?.trim();
    Some(rest.trim_end_matches(';').trim())
}

fn parse_usize(value: &str, path: &Path) -> Result<usize> {
    value.trim().parse::<usize>().map_err(|_| {
        MeshError::InvalidInput(format!("invalid label '{}' in {}", value, path.display()))
    })
}

fn missing_key(path: &Path, key: &str) -> MeshError {
    MeshError::InvalidInput(format!("missing '{key}' entry in {}", path.display()))
}

struct DictCursor {
    path: PathBuf,
    lines: Vec<String>,
    index: usize,
}

impl DictCursor {
    fn after_count_and_open(path: &Path, lines: Vec<String>) -> Result<Self> {
        let mut index = lines
            .iter()
            .position(|line| line.parse::<usize>().is_ok())
            .ok_or_else(|| {
                MeshError::InvalidInput(format!("missing dictionary count in {}", path.display()))
            })?;
        index += 1;
        while index < lines.len() && lines[index] != "(" {
            index += 1;
        }
        if index == lines.len() {
            return Err(MeshError::InvalidInput(format!(
                "missing dictionary opening '(' in {}",
                path.display()
            )));
        }
        Ok(Self {
            path: path.to_path_buf(),
            lines,
            index: index + 1,
        })
    }

    fn next_entry_name(&mut self) -> Result<Option<String>> {
        if self.index >= self.lines.len() {
            return Ok(None);
        }
        if self.lines[self.index] == ")" || self.lines[self.index] == ");" {
            return Ok(None);
        }
        Ok(Some(self.next_required()?))
    }

    fn peek_is(&self, expected: &str) -> Result<bool> {
        Ok(self.lines.get(self.index).ok_or_else(|| {
            MeshError::InvalidInput(format!("unexpected EOF in {}", self.path.display()))
        })? == expected)
    }

    fn expect(&mut self, expected: &str) -> Result<()> {
        let line = self.next_required()?;
        if line == expected {
            Ok(())
        } else {
            Err(MeshError::InvalidInput(format!(
                "expected '{}' but found '{}' in {}",
                expected,
                line,
                self.path.display()
            )))
        }
    }

    fn next_required(&mut self) -> Result<String> {
        let line = self.lines.get(self.index).cloned().ok_or_else(|| {
            MeshError::InvalidInput(format!("unexpected EOF in {}", self.path.display()))
        })?;
        self.index += 1;
        Ok(line)
    }

    fn read_label_block(&mut self) -> Result<Vec<usize>> {
        while !self.peek_is("(")? {
            self.index += 1;
        }
        self.expect("(")?;
        let mut labels = Vec::new();
        while !self.peek_is(")")? && !self.peek_is(");")? {
            labels.push(parse_usize(&self.next_required()?, &self.path)?);
        }
        self.index += 1;
        Ok(labels)
    }

    fn read_bool_block(&mut self) -> Result<Vec<bool>> {
        while !self.peek_is("(")? {
            self.index += 1;
        }
        self.expect("(")?;
        let mut values = Vec::new();
        while !self.peek_is(")")? && !self.peek_is(");")? {
            let line = self.next_required()?;
            let value = match line.as_str() {
                "true" => true,
                "false" => false,
                _ => {
                    return Err(MeshError::InvalidInput(format!(
                        "invalid bool '{}' in {}",
                        line,
                        self.path.display()
                    )));
                }
            };
            values.push(value);
        }
        self.index += 1;
        Ok(values)
    }
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
