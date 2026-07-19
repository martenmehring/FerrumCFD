use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;

use crate::{
    BoundaryFace, Cell, Mesh, MeshError, PhysicalName, Point3, Result, UnsupportedElementCount,
};

pub fn read_msh22_ascii(path: &Path) -> Result<Mesh> {
    read_msh22_ascii_with_limits(path, GmshReadLimits::default())
}

/// Resource limits applied while reading an untrusted Gmsh 2.2 ASCII file.
#[derive(Clone, Copy, Debug)]
pub struct GmshReadLimits {
    pub max_input_bytes: u64,
    pub max_physical_names: usize,
    pub max_physical_name_bytes: usize,
    pub max_nodes: usize,
    /// Limits only the `Point3` vector; the sparse lookup grows fallibly per
    /// validated record and is population-bounded by `max_nodes` and `max_input_bytes`.
    pub max_node_point_storage_bytes: usize,
    pub max_elements: usize,
    pub max_element_record_values: usize,
    /// Logical parsed storage for elements and unsupported summaries, excluding
    /// process-memory and allocator overhead.
    pub max_element_storage_bytes: usize,
}

impl Default for GmshReadLimits {
    fn default() -> Self {
        Self {
            max_input_bytes: 1024 * 1024 * 1024,
            max_physical_names: 1_000_000,
            max_physical_name_bytes: 64 * 1024 * 1024,
            max_nodes: 100_000_000,
            max_node_point_storage_bytes: 1024 * 1024 * 1024,
            max_elements: 100_000_000,
            max_element_record_values: 4096,
            max_element_storage_bytes: 1024 * 1024 * 1024,
        }
    }
}

/// Reads a Gmsh 2.2 ASCII file using caller-supplied finite resource limits.
pub fn read_msh22_ascii_with_limits(path: &Path, limits: GmshReadLimits) -> Result<Mesh> {
    let mut file = File::open(path)?;
    let mut input = String::new();
    (&mut file)
        .take(limits.max_input_bytes)
        .read_to_string(&mut input)?;
    let mut extra = [0u8; 1];
    if file.read(&mut extra)? != 0 {
        return Err(MeshError::InvalidInput(format!(
            "Gmsh input exceeds size limit {}",
            limits.max_input_bytes
        )));
    }
    let mut reader = LineReader::new(&input);

    let mut physical_names = Vec::new();
    let mut points = Vec::new();
    let mut node_id_to_index = HashMap::new();
    let mut cells = Vec::new();
    let mut boundary_faces = Vec::new();
    let mut unsupported = HashMap::<i32, usize>::new();

    while let Some(line) = reader.next_optional()? {
        match line.trim() {
            "$MeshFormat" => read_mesh_format(&mut reader)?,
            "$PhysicalNames" => physical_names = read_physical_names(&mut reader, limits)?,
            "$Nodes" => {
                let loaded = read_nodes(&mut reader, limits)?;
                node_id_to_index = loaded.node_id_to_index;
                points = loaded.points;
            }
            "$Elements" => {
                let loaded = read_elements(&mut reader, &node_id_to_index, limits)?;
                cells = loaded.cells;
                boundary_faces = loaded.boundary_faces;
                unsupported = loaded.unsupported;
            }
            _ => {}
        }
    }

    Ok(Mesh {
        points,
        cells,
        boundary_faces,
        physical_names,
        unsupported_elements: {
            let mut summaries = Vec::new();
            summaries
                .try_reserve_exact(unsupported.len())
                .map_err(|_| {
                    reader.parse_error("could not reserve final unsupported-element summaries")
                })?;
            summaries.extend(unsupported.into_iter().map(|(element_type, count)| {
                UnsupportedElementCount {
                    element_type,
                    count,
                }
            }));
            summaries.sort_unstable_by_key(|summary| summary.element_type);
            summaries
        },
    })
}

fn read_mesh_format(reader: &mut LineReader<'_>) -> Result<()> {
    let line = reader.next_required("expected mesh format line")?;
    let parts: Vec<_> = line.split_whitespace().collect();
    if parts.len() < 3 {
        return Err(reader.parse_error("invalid $MeshFormat entry"));
    }

    let version = parts[0];
    let file_type = parts[1];
    if !version.starts_with("2.2") {
        return Err(reader.parse_error(format!(
            "unsupported Gmsh version {version}; expected 2.2 ASCII"
        )));
    }
    if file_type != "0" {
        return Err(reader.parse_error("binary .msh files are not supported yet"));
    }

    reader.expect_marker("$EndMeshFormat")
}

fn read_physical_names(
    reader: &mut LineReader<'_>,
    limits: GmshReadLimits,
) -> Result<Vec<PhysicalName>> {
    let count = reader.parse_count("expected physical-name count")?;
    validate_count(reader, count, limits.max_physical_names, "physical-name")?;
    checked_storage_bytes::<PhysicalName>(reader, count, "physical-name table")?;
    let mut names = Vec::new();
    let mut name_bytes = 0usize;

    for _ in 0..count {
        let line = reader.next_required("expected physical-name entry")?;
        let parsed = parse_physical_name(line, reader.line_number())?;
        name_bytes = name_bytes
            .checked_add(parsed.name.len())
            .ok_or_else(|| reader.parse_error("physical-name text byte count overflows"))?;
        if name_bytes > limits.max_physical_name_bytes {
            return Err(reader.parse_error(format!(
                "physical-name text exceeds limit {}",
                limits.max_physical_name_bytes
            )));
        }
        let mut name = String::new();
        name.try_reserve_exact(parsed.name.len())
            .map_err(|_| reader.parse_error("could not reserve physical-name text storage"))?;
        name.push_str(parsed.name);
        names
            .try_reserve(1)
            .map_err(|_| reader.parse_error("could not grow physical-name table"))?;
        names.push(PhysicalName {
            dim: parsed.dim,
            tag: parsed.tag,
            name,
        });
    }

    reader.expect_marker("$EndPhysicalNames")?;
    Ok(names)
}

struct ParsedPhysicalName<'a> {
    dim: u8,
    tag: i32,
    name: &'a str,
}

fn parse_physical_name(line: &str, line_number: usize) -> Result<ParsedPhysicalName<'_>> {
    let quote_start = line.find('"').ok_or_else(|| MeshError::Parse {
        line: line_number,
        message: "physical name is missing opening quote".to_string(),
    })?;
    let quote_end = line.rfind('"').ok_or_else(|| MeshError::Parse {
        line: line_number,
        message: "physical name is missing closing quote".to_string(),
    })?;
    if quote_end <= quote_start {
        return Err(MeshError::Parse {
            line: line_number,
            message: "invalid quoted physical name".to_string(),
        });
    }

    let mut fields = line[..quote_start].split_whitespace();
    let dim = parse_u8(fields.next(), line_number, "physical dimension")?;
    let tag = parse_i32(fields.next(), line_number, "physical tag")?;
    let name = &line[quote_start + 1..quote_end];

    Ok(ParsedPhysicalName { dim, tag, name })
}

struct LoadedNodes {
    points: Vec<Point3>,
    node_id_to_index: HashMap<usize, usize>,
}

fn read_nodes(reader: &mut LineReader<'_>, limits: GmshReadLimits) -> Result<LoadedNodes> {
    let count = reader.parse_count("expected node count")?;
    validate_count(reader, count, limits.max_nodes, "node")?;
    let storage_bytes = checked_node_storage_bytes(reader, count)?;
    if storage_bytes > limits.max_node_point_storage_bytes {
        return Err(reader.parse_error(format!(
            "node point storage {storage_bytes} bytes exceeds limit {}",
            limits.max_node_point_storage_bytes
        )));
    }
    let mut points = Vec::new();
    let mut node_id_to_index = HashMap::new();

    for _ in 0..count {
        let line = reader.next_required("expected node entry")?;
        let mut fields = line.split_whitespace();
        let node_id = parse_usize(fields.next(), reader.line_number(), "node id")?;
        let x = parse_f64(fields.next(), reader.line_number(), "node x")?;
        let y = parse_f64(fields.next(), reader.line_number(), "node y")?;
        let z = parse_f64(fields.next(), reader.line_number(), "node z")?;
        let index = points.len();

        if node_id_to_index.contains_key(&node_id) {
            return Err(reader.parse_error(format!("duplicate node id {node_id}")));
        }
        points
            .try_reserve(1)
            .map_err(|_| reader.parse_error("could not grow node point storage"))?;
        node_id_to_index
            .try_reserve(1)
            .map_err(|_| reader.parse_error("could not grow node lookup storage"))?;
        node_id_to_index.insert(node_id, index);
        points.push(Point3 { x, y, z });
    }

    reader.expect_marker("$EndNodes")?;
    Ok(LoadedNodes {
        points,
        node_id_to_index,
    })
}

struct LoadedElements {
    cells: Vec<Cell>,
    boundary_faces: Vec<BoundaryFace>,
    unsupported: HashMap<i32, usize>,
}

fn read_elements(
    reader: &mut LineReader<'_>,
    node_id_to_index: &HashMap<usize, usize>,
    limits: GmshReadLimits,
) -> Result<LoadedElements> {
    let count = reader.parse_count("expected element count")?;
    validate_count(reader, count, limits.max_elements, "element")?;
    let mut cells = Vec::new();
    let mut boundary_faces = Vec::new();
    let mut unsupported = HashMap::<i32, usize>::new();
    let mut logical_storage = 0usize;

    for _ in 0..count {
        let line = reader.next_required("expected element entry")?;
        let mut values =
            ElementValues::new(line, reader.line_number(), limits.max_element_record_values);
        let source_id = usize::try_from(values.required("element id")?)
            .map_err(|_| reader.parse_error("element id must be non-negative"))?;
        let element_type = i32::try_from(values.required("element type")?)
            .map_err(|_| reader.parse_error("element type is outside the supported i32 range"))?;
        let tag_count = usize::try_from(values.required("element tag count")?)
            .map_err(|_| reader.parse_error("element tag count must be non-negative"))?;
        let required_prefix_values = 3usize
            .checked_add(tag_count)
            .ok_or_else(|| reader.parse_error("element tag prefix value count overflows"))?;
        if required_prefix_values > limits.max_element_record_values {
            return Err(reader.parse_error(format!(
                "element tag prefix value count {required_prefix_values} exceeds record value limit {}",
                limits.max_element_record_values
            )));
        }
        let mut physical_tag = 0;
        for index in 0..tag_count {
            let tag = values.required("element tag")?;
            if index == 0 {
                physical_tag = i32::try_from(tag)
                    .map_err(|_| reader.parse_error("physical tag is outside the i32 range"))?;
            }
        }

        match element_type {
            2 | 3 => {
                let (arity, name) = if element_type == 2 {
                    (3, "tri3")
                } else {
                    (4, "quad4")
                };
                charge_element::<BoundaryFace>(
                    reader,
                    &mut logical_storage,
                    arity,
                    limits.max_element_storage_bytes,
                )?;
                let nodes = read_element_nodes(&mut values, arity, node_id_to_index, reader, name)?;
                boundary_faces
                    .try_reserve(1)
                    .map_err(|_| reader.parse_error("could not grow boundary-face storage"))?;
                boundary_faces.push(BoundaryFace {
                    source_id,
                    physical_tag,
                    nodes,
                });
            }
            5 | 6 => {
                let (arity, name) = if element_type == 5 {
                    (8, "hex8")
                } else {
                    (6, "prism6")
                };
                charge_element::<Cell>(
                    reader,
                    &mut logical_storage,
                    arity,
                    limits.max_element_storage_bytes,
                )?;
                let nodes = read_element_nodes(&mut values, arity, node_id_to_index, reader, name)?;
                cells
                    .try_reserve(1)
                    .map_err(|_| reader.parse_error("could not grow cell storage"))?;
                cells.push(Cell {
                    source_id,
                    physical_tag,
                    nodes,
                });
            }
            other => {
                values.consume()?;
                if let Some(value) = unsupported.get_mut(&other) {
                    *value = value
                        .checked_add(1)
                        .ok_or_else(|| reader.parse_error("unsupported-element count overflows"))?;
                } else {
                    charge_bytes(
                        reader,
                        &mut logical_storage,
                        std::mem::size_of::<UnsupportedElementCount>(),
                        limits.max_element_storage_bytes,
                    )?;
                    unsupported.try_reserve(1).map_err(|_| {
                        reader.parse_error("could not grow unsupported-element summaries")
                    })?;
                    unsupported.insert(other, 1);
                }
            }
        }
    }

    reader.expect_marker("$EndElements")?;
    Ok(LoadedElements {
        cells,
        boundary_faces,
        unsupported,
    })
}

fn read_element_nodes(
    values: &mut ElementValues<'_>,
    arity: usize,
    node_id_to_index: &HashMap<usize, usize>,
    reader: &LineReader<'_>,
    name: &str,
) -> Result<Vec<usize>> {
    let mut nodes = Vec::new();
    nodes
        .try_reserve_exact(arity)
        .map_err(|_| reader.parse_error("could not reserve element node-index storage"))?;
    for _ in 0..arity {
        let id = values.next().transpose()?.ok_or_else(|| {
            reader.parse_error(format!("{name} element does not have {arity} nodes"))
        })?;
        nodes.push(node_index(id, node_id_to_index, reader.line_number())?);
    }
    if values.next().transpose()?.is_some() {
        return Err(reader.parse_error(format!("{name} element does not have {arity} nodes")));
    }
    Ok(nodes)
}

struct ElementValues<'a> {
    fields: std::str::SplitWhitespace<'a>,
    line: usize,
    seen: usize,
    limit: usize,
}

impl<'a> ElementValues<'a> {
    fn new(line: &'a str, number: usize, limit: usize) -> Self {
        Self {
            fields: line.split_whitespace(),
            line: number,
            seen: 0,
            limit,
        }
    }
    fn next(&mut self) -> Option<Result<i64>> {
        let field = self.fields.next()?;
        self.seen = match self.seen.checked_add(1) {
            Some(value) => value,
            None => return Some(Err(self.error("element record value count overflows"))),
        };
        if self.seen > self.limit {
            return Some(Err(self.error(format!(
                "element record value count exceeds limit {}",
                self.limit
            ))));
        }
        Some(
            field
                .parse()
                .map_err(|_| self.error("invalid integer in element entry")),
        )
    }
    fn required(&mut self, label: &str) -> Result<i64> {
        self.next()
            .transpose()?
            .ok_or_else(|| self.error(format!("missing {label}")))
    }
    fn consume(&mut self) -> Result<()> {
        while self.next().transpose()?.is_some() {}
        Ok(())
    }
    fn error(&self, message: impl Into<String>) -> MeshError {
        MeshError::Parse {
            line: self.line,
            message: message.into(),
        }
    }
}

fn charge_element<T>(
    reader: &LineReader<'_>,
    used: &mut usize,
    nodes: usize,
    limit: usize,
) -> Result<()> {
    let node_bytes = nodes
        .checked_mul(std::mem::size_of::<usize>())
        .ok_or_else(|| reader.parse_error("element node-index storage byte size overflows"))?;
    let bytes = std::mem::size_of::<T>()
        .checked_add(node_bytes)
        .ok_or_else(|| reader.parse_error("element logical storage byte size overflows"))?;
    charge_bytes(reader, used, bytes, limit)
}

fn charge_bytes(
    reader: &LineReader<'_>,
    used: &mut usize,
    bytes: usize,
    limit: usize,
) -> Result<()> {
    let total = used
        .checked_add(bytes)
        .ok_or_else(|| reader.parse_error("element logical storage byte size overflows"))?;
    if total > limit {
        return Err(reader.parse_error(format!(
            "element logical storage {total} bytes exceeds limit {limit}"
        )));
    }
    *used = total;
    Ok(())
}

fn node_index(
    node_id: i64,
    node_id_to_index: &HashMap<usize, usize>,
    line: usize,
) -> Result<usize> {
    let node_id = usize::try_from(node_id).map_err(|_| MeshError::Parse {
        line,
        message: format!("negative or unrepresentable node id {node_id}"),
    })?;

    node_id_to_index
        .get(&node_id)
        .copied()
        .ok_or_else(|| MeshError::Parse {
            line,
            message: format!("unknown node id {node_id}"),
        })
}

fn parse_u8(value: Option<&str>, line: usize, label: &str) -> Result<u8> {
    value
        .ok_or_else(|| MeshError::Parse {
            line,
            message: format!("missing {label}"),
        })?
        .parse()
        .map_err(|_| MeshError::Parse {
            line,
            message: format!("invalid {label}"),
        })
}

fn parse_i32(value: Option<&str>, line: usize, label: &str) -> Result<i32> {
    value
        .ok_or_else(|| MeshError::Parse {
            line,
            message: format!("missing {label}"),
        })?
        .parse()
        .map_err(|_| MeshError::Parse {
            line,
            message: format!("invalid {label}"),
        })
}

fn parse_usize(value: Option<&str>, line: usize, label: &str) -> Result<usize> {
    value
        .ok_or_else(|| MeshError::Parse {
            line,
            message: format!("missing {label}"),
        })?
        .parse()
        .map_err(|_| MeshError::Parse {
            line,
            message: format!("invalid {label}"),
        })
}

fn parse_f64(value: Option<&str>, line: usize, label: &str) -> Result<f64> {
    value
        .ok_or_else(|| MeshError::Parse {
            line,
            message: format!("missing {label}"),
        })?
        .parse()
        .map_err(|_| MeshError::Parse {
            line,
            message: format!("invalid {label}"),
        })
}

struct LineReader<'a> {
    input: &'a str,
    offset: usize,
    line_number: usize,
    remaining_bytes: usize,
}

impl<'a> LineReader<'a> {
    fn new(input: &'a str) -> Self {
        Self {
            input,
            offset: 0,
            line_number: 0,
            remaining_bytes: input.len(),
        }
    }

    fn line_number(&self) -> usize {
        self.line_number
    }

    fn next_optional(&mut self) -> Result<Option<&'a str>> {
        if self.offset == self.input.len() {
            return Ok(None);
        }

        let rest = &self.input[self.offset..];
        let (line_with_cr, consumed) = match rest.find('\n') {
            Some(index) => (&rest[..index], index + 1),
            None => (rest, rest.len()),
        };
        let line = line_with_cr.strip_suffix('\r').unwrap_or(line_with_cr);
        self.offset += consumed;
        self.remaining_bytes -= consumed;
        self.line_number += 1;
        Ok(Some(line))
    }

    fn next_required(&mut self, message: &str) -> Result<&'a str> {
        self.next_optional()?
            .ok_or_else(|| self.parse_error(message.to_string()))
    }

    fn parse_count(&mut self, message: &str) -> Result<usize> {
        let line = self.next_required(message)?;
        line.trim()
            .parse()
            .map_err(|_| self.parse_error(format!("invalid count: {line}")))
    }

    fn expect_marker(&mut self, marker: &str) -> Result<()> {
        let line = self.next_required(&format!("expected {marker}"))?;
        if line.trim() == marker {
            Ok(())
        } else {
            Err(self.parse_error(format!("expected {marker}, found {}", line.trim())))
        }
    }

    fn parse_error(&self, message: impl Into<String>) -> MeshError {
        MeshError::Parse {
            line: self.line_number,
            message: message.into(),
        }
    }
}

fn validate_count(reader: &LineReader<'_>, count: usize, limit: usize, label: &str) -> Result<()> {
    if count > limit {
        return Err(reader.parse_error(format!("{label} count {count} exceeds limit {limit}")));
    }
    if count > reader.remaining_bytes {
        return Err(reader.parse_error(format!(
            "{label} count {count} is implausible for {} remaining input bytes",
            reader.remaining_bytes
        )));
    }
    Ok(())
}

fn checked_storage_bytes<T>(reader: &LineReader<'_>, count: usize, label: &str) -> Result<usize> {
    count
        .checked_mul(std::mem::size_of::<T>())
        .ok_or_else(|| reader.parse_error(format!("{label} byte size overflows")))
}

fn checked_node_storage_bytes(reader: &LineReader<'_>, count: usize) -> Result<usize> {
    checked_storage_bytes::<Point3>(reader, count, "node point storage")
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    use crate::foam::write_openfoam_case;

    use super::*;

    #[test]
    fn reads_prism6_and_tri3_boundary_faces() {
        let mesh_path = unique_temp_path("prism6_tri3", "msh");
        fs::write(&mesh_path, prism_msh()).expect("write test mesh");

        let mesh = read_msh22_ascii(&mesh_path).expect("read prism mesh");

        assert_eq!(mesh.points.len(), 6);
        assert_eq!(mesh.cells.len(), 1);
        assert_eq!(mesh.cells[0].nodes.len(), 6);
        assert_eq!(mesh.boundary_faces.len(), 5);
        assert_eq!(
            mesh.boundary_faces
                .iter()
                .filter(|face| face.nodes.len() == 3)
                .count(),
            2
        );
        assert_eq!(
            mesh.boundary_faces
                .iter()
                .filter(|face| face.nodes.len() == 4)
                .count(),
            3
        );
        assert!(mesh.unsupported_elements.is_empty());

        let _ = fs::remove_file(mesh_path);
    }

    #[test]
    fn writes_openfoam_faces_for_prism6_mesh() {
        let mesh_path = unique_temp_path("prism6_poly_mesh", "msh");
        let case_dir = unique_temp_path("prism6_case", "case");
        fs::write(&mesh_path, prism_msh()).expect("write test mesh");
        let mesh = read_msh22_ascii(&mesh_path).expect("read prism mesh");

        let summary = write_openfoam_case(&mesh, &case_dir, &mesh_path).expect("write foam case");
        assert_eq!(summary.cells, 1);
        assert_eq!(summary.faces, 5);
        assert_eq!(summary.boundary_faces, 5);

        let faces = fs::read_to_string(case_dir.join("constant/polyMesh/faces"))
            .expect("read written faces");
        assert!(faces.contains("3("));
        assert!(faces.contains("4("));

        let _ = fs::remove_file(mesh_path);
        let _ = fs::remove_dir_all(case_dir);
    }

    #[test]
    fn rejects_negative_element_tag_count_without_panicking() {
        let mesh_path = unique_temp_path("negative_tag_count", "msh");
        let content = r#"$MeshFormat
2.2 0 8
$EndMeshFormat
$Nodes
1
1 0 0 0
$EndNodes
$Elements
1
1 15 -1
$EndElements
"#;
        fs::write(&mesh_path, content).expect("write malformed mesh");

        let error = read_msh22_ascii(&mesh_path).expect_err("negative tag count must fail");

        assert!(error.to_string().contains("tag count must be non-negative"));
        let _ = fs::remove_file(mesh_path);
    }

    #[test]
    fn enforces_physical_name_limits_and_accepts_exact_limit() {
        let content = "$PhysicalNames\n1\n2 7 \"wall\"\n$EndPhysicalNames\n";
        let limits = GmshReadLimits {
            max_input_bytes: 1024,
            max_physical_names: 1,
            max_physical_name_bytes: 4,
            max_nodes: 1,
            max_node_point_storage_bytes: 1024,
            max_elements: 1,
            max_element_record_values: 16,
            max_element_storage_bytes: 1024,
        };
        assert_eq!(read_content(content, limits).physical_names.len(), 1);

        let error = read_content_error(
            content,
            GmshReadLimits {
                max_physical_names: 0,
                ..limits
            },
        );
        assert!(error.contains("physical-name count 1 exceeds limit 0"));
        let error = read_content_error(
            content,
            GmshReadLimits {
                max_physical_name_bytes: 3,
                ..limits
            },
        );
        assert!(error.contains("physical-name text exceeds limit 3"));
    }

    #[test]
    fn rejects_over_budget_name_before_owned_name_materialization() {
        let content = "$PhysicalNames\n1\n2 7 \"name-too-long\"\n$EndPhysicalNames\n";
        let error = read_content_error(
            content,
            GmshReadLimits {
                max_physical_names: 1,
                max_physical_name_bytes: 4,
                ..GmshReadLimits::default()
            },
        );

        assert!(error.contains("physical-name text exceeds limit 4"));
    }

    #[test]
    fn rejects_below_limit_physical_name_eof() {
        let error = read_content_error(
            "$PhysicalNames\n2\n",
            GmshReadLimits {
                max_physical_names: 2,
                ..GmshReadLimits::default()
            },
        );
        assert!(error.contains("implausible"));
    }

    #[test]
    fn padded_truncated_physical_names_fail_before_any_declared_count_allocation() {
        let error = read_content_error(
            "$PhysicalNames\n4\n    ",
            GmshReadLimits {
                max_physical_names: 4,
                ..GmshReadLimits::default()
            },
        );
        assert!(error.contains("physical name is missing opening quote"));
        assert!(!error.contains("reserve physical-name table"));
    }

    #[test]
    fn enforces_node_limits_and_accepts_sparse_labels() {
        let content = "$Nodes\n1\n999999999 0 0 0\n$EndNodes\n$Elements\n1\n1 2 0 999999999 999999999 999999999\n$EndElements\n";
        let limits = GmshReadLimits {
            max_nodes: 1,
            ..GmshReadLimits::default()
        };
        let mesh = read_content(content, limits);
        assert_eq!(mesh.points.len(), 1);
        assert_eq!(mesh.boundary_faces[0].nodes, vec![0, 0, 0]);

        let error = read_content_error(
            content,
            GmshReadLimits {
                max_nodes: 0,
                ..limits
            },
        );
        assert!(error.contains("node count 1 exceeds limit 0"));
    }

    #[test]
    fn rejects_duplicate_and_unknown_node_labels() {
        let duplicate = "$Nodes\n2\n4 0 0 0\n4 1 0 0\n$EndNodes\n";
        assert!(
            read_content_error(duplicate, GmshReadLimits::default())
                .contains("duplicate node id 4")
        );

        let unknown = "$Nodes\n1\n4 0 0 0\n$EndNodes\n$Elements\n1\n1 2 0 4 4 5\n$EndElements\n";
        assert!(
            read_content_error(unknown, GmshReadLimits::default()).contains("unknown node id 5")
        );
    }

    #[test]
    fn node_collections_grow_only_for_parsed_records() {
        let mut content = String::from("$Nodes\n15\n");
        for node_id in 1..=15 {
            content.push_str(&format!("{node_id} 0 0 0\n"));
        }
        content.push_str("$EndNodes\n");

        let mut reader = LineReader::new(&content);
        assert_eq!(reader.next_optional().expect("read marker"), Some("$Nodes"));
        let loaded = read_nodes(
            &mut reader,
            GmshReadLimits {
                max_nodes: 15,
                ..GmshReadLimits::default()
            },
        )
        .expect("read nodes across a typical hash-table growth threshold");
        assert_eq!(loaded.points.len(), 15);
        assert_eq!(loaded.node_id_to_index.len(), 15);
    }

    #[test]
    fn padded_truncated_nodes_fail_before_any_declared_count_allocation() {
        let error = read_content_error(
            "$Nodes\n4\n    ",
            GmshReadLimits {
                max_nodes: 4,
                ..GmshReadLimits::default()
            },
        );
        assert!(error.contains("missing node id"));
        assert!(!error.contains("reserve node"));
    }

    #[test]
    fn checked_storage_overflow_is_structured() {
        let reader = LineReader::new("");
        let error = checked_storage_bytes::<Point3>(&reader, usize::MAX, "nodes")
            .expect_err("multiplication must overflow");
        assert!(error.to_string().contains("nodes byte size overflows"));
    }

    #[test]
    fn enforces_node_storage_budget_before_reservation() {
        let one_node = "$Nodes\n1\n1 0 0 0\n$EndNodes\n";
        let exact = checked_node_storage_bytes(&LineReader::new(""), 1)
            .expect("one-node point storage accounting must fit");
        read_content(
            one_node,
            GmshReadLimits {
                max_nodes: 1,
                max_node_point_storage_bytes: exact,
                ..GmshReadLimits::default()
            },
        );

        let error = read_content_error(
            one_node,
            GmshReadLimits {
                max_nodes: 1,
                max_node_point_storage_bytes: exact - 1,
                ..GmshReadLimits::default()
            },
        );
        assert!(error.contains("node point storage"));
        assert!(error.contains("exceeds limit"));
    }

    #[test]
    fn rejects_large_declared_node_storage_without_records() {
        let content = format!("$Nodes\n100\n{}", " ".repeat(100));
        let error = read_content_error(
            &content,
            GmshReadLimits {
                max_nodes: 100,
                max_node_point_storage_bytes: 1,
                ..GmshReadLimits::default()
            },
        );
        assert!(error.contains("node point storage"));
    }

    #[test]
    fn checked_node_point_storage_overflow_is_structured() {
        let reader = LineReader::new("");
        let error = checked_node_storage_bytes(&reader, usize::MAX)
            .expect_err("node point storage must overflow");
        assert!(error.to_string().contains("byte size overflows"));
    }

    #[test]
    fn enforces_actual_input_byte_limit() {
        let content = "$Nodes\n0\n$EndNodes\n";
        let exact_limit = u64::try_from(content.len()).expect("content length fits u64");
        read_content(
            content,
            GmshReadLimits {
                max_input_bytes: exact_limit,
                ..GmshReadLimits::default()
            },
        );

        let error = read_content_error(
            content,
            GmshReadLimits {
                max_input_bytes: exact_limit - 1,
                ..GmshReadLimits::default()
            },
        );
        assert!(error.contains("input exceeds size limit"));
    }

    #[test]
    fn tracks_line_bytes_for_lf_crlf_and_final_line() {
        let mut reader = LineReader::new("a\r\nb\nc");
        assert_eq!(reader.remaining_bytes, 6);
        assert_eq!(reader.next_optional().expect("read line"), Some("a"));
        assert_eq!(reader.remaining_bytes, 3);
        assert_eq!(reader.next_optional().expect("read line"), Some("b"));
        assert_eq!(reader.remaining_bytes, 1);
        assert_eq!(reader.next_optional().expect("read line"), Some("c"));
        assert_eq!(reader.remaining_bytes, 0);
        assert_eq!(reader.next_optional().expect("read eof"), None);
    }

    #[test]
    fn enforces_element_count_and_record_value_limits_at_boundary() {
        let content = "$Elements\n1\n1 15 0\n$EndElements\n";
        read_content(
            content,
            GmshReadLimits {
                max_elements: 1,
                max_element_record_values: 3,
                ..GmshReadLimits::default()
            },
        );
        assert!(
            read_content_error(
                content,
                GmshReadLimits {
                    max_elements: 0,
                    ..GmshReadLimits::default()
                }
            )
            .contains("element count 1 exceeds limit 0")
        );
        assert!(
            read_content_error(
                content,
                GmshReadLimits {
                    max_element_record_values: 2,
                    ..GmshReadLimits::default()
                }
            )
            .contains("record value count exceeds limit 2")
        );
    }

    #[test]
    fn enforces_logical_element_storage_at_boundary() {
        let content = "$Nodes\n1\n1 0 0 0\n$EndNodes\n$Elements\n1\n1 2 0 1 1 1\n$EndElements\n";
        let exact = std::mem::size_of::<BoundaryFace>() + 3 * std::mem::size_of::<usize>();
        read_content(
            content,
            GmshReadLimits {
                max_element_storage_bytes: exact,
                ..GmshReadLimits::default()
            },
        );
        assert!(
            read_content_error(
                content,
                GmshReadLimits {
                    max_element_storage_bytes: exact - 1,
                    ..GmshReadLimits::default()
                }
            )
            .contains("element logical storage")
        );
    }

    #[test]
    fn reads_quad4_hex8_and_sorts_aggregated_unsupported_types() {
        let content = "$Nodes\n8\n1 0 0 0\n2 0 0 0\n3 0 0 0\n4 0 0 0\n5 0 0 0\n6 0 0 0\n7 0 0 0\n8 0 0 0\n$EndNodes\n$Elements\n5\n1 99 0\n2 3 0 1 2 3 4\n3 42 0 7\n4 99 0 8\n5 5 0 1 2 3 4 5 6 7 8\n$EndElements\n";
        let mesh = read_content(content, GmshReadLimits::default());
        assert_eq!(mesh.boundary_faces[0].nodes.len(), 4);
        assert_eq!(mesh.cells[0].nodes.len(), 8);
        assert_eq!(mesh.unsupported_elements.len(), 2);
        assert_eq!(mesh.unsupported_elements[0].element_type, 42);
        assert_eq!(mesh.unsupported_elements[0].count, 1);
        assert_eq!(mesh.unsupported_elements[1].element_type, 99);
        assert_eq!(mesh.unsupported_elements[1].count, 2);
    }

    #[test]
    fn rejects_invalid_and_over_limit_unsupported_record_tails() {
        let invalid = "$Elements\n1\n1 99 0 invalid\n$EndElements\n";
        assert!(
            read_content_error(invalid, GmshReadLimits::default())
                .contains("invalid integer in element entry")
        );

        let over_limit = "$Elements\n1\n1 99 0 7\n$EndElements\n";
        assert!(
            read_content_error(
                over_limit,
                GmshReadLimits {
                    max_element_record_values: 3,
                    ..GmshReadLimits::default()
                }
            )
            .contains("record value count exceeds limit 3")
        );
    }

    #[test]
    fn enforces_default_record_value_limit_for_unsupported_tail() {
        let limit = GmshReadLimits::default().max_element_record_values;
        let record_at_limit = format!("1 99 0{}", " 7".repeat(limit - 3));
        let content_at_limit = format!("$Elements\n1\n{record_at_limit}\n$EndElements\n");
        let mesh = read_content(&content_at_limit, GmshReadLimits::default());
        assert_eq!(mesh.unsupported_elements[0].element_type, 99);
        assert_eq!(mesh.unsupported_elements[0].count, 1);

        let record_over_limit = format!("{record_at_limit} 7");
        let content_over_limit = format!("$Elements\n1\n{record_over_limit}\n$EndElements\n");
        assert!(
            read_content_error(&content_over_limit, GmshReadLimits::default())
                .contains("record value count exceeds limit 4096")
        );
    }

    #[test]
    fn rejects_over_limit_tag_prefix_before_consuming_tags() {
        let content = "$Elements\n1\n1 99 2\n$EndElements\n";
        let error = read_content_error(
            content,
            GmshReadLimits {
                max_element_record_values: 4,
                ..GmshReadLimits::default()
            },
        );
        assert!(error.contains("tag prefix value count 5 exceeds record value limit 4"));
        assert!(!error.contains("missing element tag"));
    }

    fn read_content(content: &str, limits: GmshReadLimits) -> Mesh {
        let path = unique_temp_path("bounded", "msh");
        fs::write(&path, content).expect("write test mesh");
        let result = read_msh22_ascii_with_limits(&path, limits).expect("read test mesh");
        let _ = fs::remove_file(path);
        result
    }

    fn read_content_error(content: &str, limits: GmshReadLimits) -> String {
        let path = unique_temp_path("bounded_error", "msh");
        fs::write(&path, content).expect("write test mesh");
        let error = read_msh22_ascii_with_limits(&path, limits).expect_err("mesh must fail");
        let _ = fs::remove_file(path);
        error.to_string()
    }

    fn unique_temp_path(label: &str, extension: &str) -> std::path::PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time before epoch")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "ferrum_mesh_{label}_{}_{}.{}",
            std::process::id(),
            nanos,
            extension
        ))
    }

    fn prism_msh() -> &'static str {
        r#"$MeshFormat
2.2 0 8
$EndMeshFormat
$PhysicalNames
5
2 1 "inlet"
2 2 "outlet"
2 3 "wall"
3 10 "fluid"
0 99 "ignored"
$EndPhysicalNames
$Nodes
6
1 0 0 0
2 1 0 0
3 0 1 0
4 0 0 1
5 1 0 1
6 0 1 1
$EndNodes
$Elements
6
1 2 2 1 0 1 2 3
2 2 2 2 0 4 5 6
3 3 2 3 0 1 2 5 4
4 3 2 3 0 2 3 6 5
5 3 2 3 0 3 1 4 6
6 6 2 10 0 1 2 3 4 5 6
$EndElements
"#
    }
}
