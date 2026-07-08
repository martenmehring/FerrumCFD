use std::fs;
use std::path::{Path, PathBuf};

use crate::{MeshError, Point3, Result};

#[derive(Debug)]
pub struct PolyMesh {
    pub path: PathBuf,
    pub points: Vec<Point3>,
    pub faces: Vec<Vec<usize>>,
    pub owner: Vec<usize>,
    pub neighbour: Vec<usize>,
    pub patches: Vec<BoundaryPatch>,
}

#[derive(Debug)]
pub struct BoundaryPatch {
    pub name: String,
    pub patch_type: String,
    pub faces: usize,
    pub start_face: usize,
}

impl PolyMesh {
    pub fn read(path: &Path) -> Result<Self> {
        let points = read_points(&path.join("points"))?;
        let faces = read_faces(&path.join("faces"))?;
        let owner = read_label_list(&path.join("owner"))?;
        let neighbour = read_label_list(&path.join("neighbour"))?;
        let patches = read_boundary(&path.join("boundary"))?;

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
            path: path.to_path_buf(),
            points,
            faces,
            owner,
            neighbour,
            patches,
        })
    }

    pub fn cell_count(&self) -> usize {
        self.owner
            .iter()
            .chain(self.neighbour.iter())
            .copied()
            .max()
            .map(|cell| cell + 1)
            .unwrap_or(0)
    }
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

fn parse_usize(value: &str, path: &Path) -> Result<usize> {
    value.trim().parse::<usize>().map_err(|_| {
        MeshError::InvalidInput(format!("invalid label '{}' in {}", value, path.display()))
    })
}

fn parse_dict_value<'a>(line: &'a str, key: &str) -> Option<&'a str> {
    let rest = line.strip_prefix(key)?.trim();
    Some(rest.trim_end_matches(';').trim())
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
}
