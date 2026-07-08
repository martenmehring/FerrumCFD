use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Lines};
use std::path::Path;

use crate::{
    BoundaryFace, Cell, Mesh, MeshError, PhysicalName, Point3, Result, UnsupportedElementCount,
};

pub fn read_msh22_ascii(path: &Path) -> Result<Mesh> {
    let file = File::open(path)?;
    let mut reader = LineReader::new(BufReader::new(file).lines());

    let mut physical_names = Vec::new();
    let mut points = Vec::new();
    let mut node_id_to_index: Vec<Option<usize>> = Vec::new();
    let mut cells = Vec::new();
    let mut boundary_faces = Vec::new();
    let mut unsupported = BTreeMap::<i32, usize>::new();

    while let Some(line) = reader.next_optional()? {
        match line.trim() {
            "$MeshFormat" => read_mesh_format(&mut reader)?,
            "$PhysicalNames" => physical_names = read_physical_names(&mut reader)?,
            "$Nodes" => {
                let loaded = read_nodes(&mut reader)?;
                node_id_to_index = loaded.node_id_to_index;
                points = loaded.points;
            }
            "$Elements" => {
                let loaded = read_elements(&mut reader, &node_id_to_index)?;
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
        unsupported_elements: unsupported
            .into_iter()
            .map(|(element_type, count)| UnsupportedElementCount {
                element_type,
                count,
            })
            .collect(),
    })
}

fn read_mesh_format<R: BufRead>(reader: &mut LineReader<R>) -> Result<()> {
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

fn read_physical_names<R: BufRead>(reader: &mut LineReader<R>) -> Result<Vec<PhysicalName>> {
    let count = reader.parse_count("expected physical-name count")?;
    let mut names = Vec::with_capacity(count);

    for _ in 0..count {
        let line = reader.next_required("expected physical-name entry")?;
        names.push(parse_physical_name(&line, reader.line_number())?);
    }

    reader.expect_marker("$EndPhysicalNames")?;
    Ok(names)
}

fn parse_physical_name(line: &str, line_number: usize) -> Result<PhysicalName> {
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
    let name = line[quote_start + 1..quote_end].to_string();

    Ok(PhysicalName { dim, tag, name })
}

struct LoadedNodes {
    points: Vec<Point3>,
    node_id_to_index: Vec<Option<usize>>,
}

fn read_nodes<R: BufRead>(reader: &mut LineReader<R>) -> Result<LoadedNodes> {
    let count = reader.parse_count("expected node count")?;
    let mut points = Vec::with_capacity(count);
    let mut node_id_to_index = vec![None; count + 1];

    for _ in 0..count {
        let line = reader.next_required("expected node entry")?;
        let mut fields = line.split_whitespace();
        let node_id = parse_usize(fields.next(), reader.line_number(), "node id")?;
        let x = parse_f64(fields.next(), reader.line_number(), "node x")?;
        let y = parse_f64(fields.next(), reader.line_number(), "node y")?;
        let z = parse_f64(fields.next(), reader.line_number(), "node z")?;
        let index = points.len();

        if node_id >= node_id_to_index.len() {
            node_id_to_index.resize(node_id + 1, None);
        }
        node_id_to_index[node_id] = Some(index);
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
    unsupported: BTreeMap<i32, usize>,
}

fn read_elements<R: BufRead>(
    reader: &mut LineReader<R>,
    node_id_to_index: &[Option<usize>],
) -> Result<LoadedElements> {
    let count = reader.parse_count("expected element count")?;
    let mut cells = Vec::new();
    let mut boundary_faces = Vec::new();
    let mut unsupported = BTreeMap::<i32, usize>::new();

    for _ in 0..count {
        let line = reader.next_required("expected element entry")?;
        let fields = line
            .split_whitespace()
            .map(str::parse::<i64>)
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|_| reader.parse_error("invalid integer in element entry"))?;

        if fields.len() < 3 {
            return Err(reader.parse_error("element entry is too short"));
        }

        let source_id = fields[0] as usize;
        let element_type = fields[1] as i32;
        let tag_count = fields[2] as usize;
        let node_start = 3 + tag_count;
        if fields.len() < node_start {
            return Err(reader.parse_error("element entry has fewer tags than declared"));
        }
        let physical_tag = if tag_count > 0 { fields[3] as i32 } else { 0 };

        match element_type {
            3 => {
                if fields.len() != node_start + 4 {
                    return Err(reader.parse_error("quad4 element does not have 4 nodes"));
                }
                boundary_faces.push(BoundaryFace {
                    source_id,
                    physical_tag,
                    nodes: [
                        node_index(fields[node_start], node_id_to_index, reader.line_number())?,
                        node_index(
                            fields[node_start + 1],
                            node_id_to_index,
                            reader.line_number(),
                        )?,
                        node_index(
                            fields[node_start + 2],
                            node_id_to_index,
                            reader.line_number(),
                        )?,
                        node_index(
                            fields[node_start + 3],
                            node_id_to_index,
                            reader.line_number(),
                        )?,
                    ],
                });
            }
            5 => {
                if fields.len() != node_start + 8 {
                    return Err(reader.parse_error("hex8 element does not have 8 nodes"));
                }
                cells.push(Cell {
                    source_id,
                    physical_tag,
                    nodes: [
                        node_index(fields[node_start], node_id_to_index, reader.line_number())?,
                        node_index(
                            fields[node_start + 1],
                            node_id_to_index,
                            reader.line_number(),
                        )?,
                        node_index(
                            fields[node_start + 2],
                            node_id_to_index,
                            reader.line_number(),
                        )?,
                        node_index(
                            fields[node_start + 3],
                            node_id_to_index,
                            reader.line_number(),
                        )?,
                        node_index(
                            fields[node_start + 4],
                            node_id_to_index,
                            reader.line_number(),
                        )?,
                        node_index(
                            fields[node_start + 5],
                            node_id_to_index,
                            reader.line_number(),
                        )?,
                        node_index(
                            fields[node_start + 6],
                            node_id_to_index,
                            reader.line_number(),
                        )?,
                        node_index(
                            fields[node_start + 7],
                            node_id_to_index,
                            reader.line_number(),
                        )?,
                    ],
                });
            }
            other => {
                *unsupported.entry(other).or_default() += 1;
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

fn node_index(node_id: i64, node_id_to_index: &[Option<usize>], line: usize) -> Result<usize> {
    if node_id < 0 {
        return Err(MeshError::Parse {
            line,
            message: format!("negative node id {node_id}"),
        });
    }

    node_id_to_index
        .get(node_id as usize)
        .and_then(|index| *index)
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

struct LineReader<R: BufRead> {
    lines: Lines<R>,
    line_number: usize,
}

impl<R: BufRead> LineReader<R> {
    fn new(lines: Lines<R>) -> Self {
        Self {
            lines,
            line_number: 0,
        }
    }

    fn line_number(&self) -> usize {
        self.line_number
    }

    fn next_optional(&mut self) -> Result<Option<String>> {
        match self.lines.next() {
            Some(line) => {
                self.line_number += 1;
                Ok(Some(line?))
            }
            None => Ok(None),
        }
    }

    fn next_required(&mut self, message: &str) -> Result<String> {
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
