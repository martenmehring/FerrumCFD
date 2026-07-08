pub mod backends;
pub mod check;
pub mod control;
pub mod dictionary;
pub mod fields;
pub mod foam;
pub mod geometry;
pub mod gmsh;
pub mod interfaces;
pub mod numerics;
pub mod patches;
pub mod poly_mesh;
pub mod properties;
pub mod regions;
pub mod solver_plan;

use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct Mesh {
    pub points: Vec<Point3>,
    pub cells: Vec<Cell>,
    pub boundary_faces: Vec<BoundaryFace>,
    pub physical_names: Vec<PhysicalName>,
    pub unsupported_elements: Vec<UnsupportedElementCount>,
}

impl Mesh {
    pub fn physical_name(&self, tag: i32) -> String {
        self.physical_names
            .iter()
            .find(|name| name.tag == tag)
            .map(|name| name.name.clone())
            .unwrap_or_else(|| format!("physical_{tag}"))
    }

    pub fn physical_name_map(&self) -> HashMap<i32, String> {
        self.physical_names
            .iter()
            .map(|name| (name.tag, name.name.clone()))
            .collect()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Point3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

#[derive(Clone, Debug)]
pub struct Cell {
    pub source_id: usize,
    pub physical_tag: i32,
    pub nodes: [usize; 8],
}

#[derive(Clone, Debug)]
pub struct BoundaryFace {
    pub source_id: usize,
    pub physical_tag: i32,
    pub nodes: [usize; 4],
}

#[derive(Clone, Debug)]
pub struct PhysicalName {
    pub dim: u8,
    pub tag: i32,
    pub name: String,
}

#[derive(Clone, Debug)]
pub struct UnsupportedElementCount {
    pub element_type: i32,
    pub count: usize,
}

#[derive(Debug)]
pub enum MeshError {
    Io(std::io::Error),
    Parse { line: usize, message: String },
    InvalidInput(String),
}

impl std::fmt::Display for MeshError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MeshError::Io(error) => write!(f, "{error}"),
            MeshError::Parse { line, message } => write!(f, "line {line}: {message}"),
            MeshError::InvalidInput(message) => write!(f, "{message}"),
        }
    }
}

impl std::error::Error for MeshError {}

impl From<std::io::Error> for MeshError {
    fn from(value: std::io::Error) -> Self {
        MeshError::Io(value)
    }
}

pub type Result<T> = std::result::Result<T, MeshError>;
