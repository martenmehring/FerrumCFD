use std::fs;
use std::path::{Path, PathBuf};

use crate::{MeshError, Result};

#[derive(Debug)]
pub struct CaseSummary {
    pub path: PathBuf,
    pub points: Option<usize>,
    pub cells: Option<usize>,
    pub faces: Option<usize>,
    pub internal_faces: Option<usize>,
    pub boundary_faces: Option<usize>,
    pub unmatched_boundary_faces: Option<usize>,
    pub duplicate_boundary_faces: Option<usize>,
    pub non_manifold_faces: Option<usize>,
    pub patches: Vec<String>,
    pub face_zones: Vec<String>,
    pub cell_zones: Vec<String>,
}

pub fn read_case_summary(case_dir: &Path) -> Result<CaseSummary> {
    let summary_path = case_dir.join("constant").join("ferrumMeshSummary.txt");
    let content = fs::read_to_string(&summary_path).map_err(|error| {
        MeshError::InvalidInput(format!(
            "could not read {}; run gmshToFerrum first ({error})",
            summary_path.display()
        ))
    })?;

    let mut summary = CaseSummary {
        path: case_dir.to_path_buf(),
        points: None,
        cells: None,
        faces: None,
        internal_faces: None,
        boundary_faces: None,
        unmatched_boundary_faces: None,
        duplicate_boundary_faces: None,
        non_manifold_faces: None,
        patches: Vec::new(),
        face_zones: Vec::new(),
        cell_zones: Vec::new(),
    };

    let mut section = "";
    for line in content
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
    {
        match line {
            "[patches]" => {
                section = "patches";
                continue;
            }
            "[cell_zones]" => {
                section = "cell_zones";
                continue;
            }
            "[face_zones]" => {
                section = "face_zones";
                continue;
            }
            "[unsupported_elements]" => {
                section = "unsupported_elements";
                continue;
            }
            _ => {}
        }

        if section == "patches" {
            summary.patches.push(line.to_string());
            continue;
        }
        if section == "face_zones" {
            summary.face_zones.push(line.to_string());
            continue;
        }
        if section == "cell_zones" {
            summary.cell_zones.push(line.to_string());
            continue;
        }

        if let Some((key, value)) = line.split_once('=') {
            let number = value.parse::<usize>().ok();
            match key {
                "points" => summary.points = number,
                "cells" => summary.cells = number,
                "faces" => summary.faces = number,
                "internal_faces" => summary.internal_faces = number,
                "boundary_faces" => summary.boundary_faces = number,
                "unmatched_boundary_faces" => summary.unmatched_boundary_faces = number,
                "duplicate_boundary_faces" => summary.duplicate_boundary_faces = number,
                "non_manifold_faces" => summary.non_manifold_faces = number,
                _ => {}
            }
        }
    }

    Ok(summary)
}
