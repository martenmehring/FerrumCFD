//! Deterministic writer for the supported subset of neutral Gmsh 2.2 ASCII.

use std::collections::HashSet;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use crate::{Mesh, MeshError, PhysicalName, Result};

/// Writes a validated mesh as neutral Gmsh 2.2 ASCII.
pub fn write_msh22_ascii(path: &Path, mesh: &Mesh) -> Result<()> {
    validate_mesh(mesh)?;
    let mut writer = BufWriter::new(File::create(path)?);
    write_validated_msh22_ascii(&mut writer, mesh)?;
    writer.flush()?;
    Ok(())
}

/// Writes a validated mesh to an arbitrary byte sink.
///
/// The output always uses LF line endings and preserves the order of every
/// input vector, making identical meshes byte-for-byte deterministic.
pub fn write_msh22_ascii_to<W: Write>(writer: &mut W, mesh: &Mesh) -> Result<()> {
    validate_mesh(mesh)?;
    write_validated_msh22_ascii(writer, mesh)
}

fn write_validated_msh22_ascii<W: Write>(writer: &mut W, mesh: &Mesh) -> Result<()> {
    writer.write_all(b"$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")?;
    writeln!(writer, "$PhysicalNames")?;
    writeln!(writer, "{}", mesh.physical_names.len())?;
    for physical in &mesh.physical_names {
        writeln!(
            writer,
            "{} {} \"{}\"",
            physical.dim, physical.tag, physical.name
        )?;
    }
    writer.write_all(b"$EndPhysicalNames\n$Nodes\n")?;
    writeln!(writer, "{}", mesh.points.len())?;
    for (index, point) in mesh.points.iter().enumerate() {
        let node_id = index.checked_add(1).ok_or(MeshError::OutOfMemory)?;
        writeln!(writer, "{node_id} {} {} {}", point.x, point.y, point.z)?;
    }
    writer.write_all(b"$EndNodes\n$Elements\n")?;
    let element_count = mesh
        .boundary_faces
        .len()
        .checked_add(mesh.cells.len())
        .ok_or(MeshError::OutOfMemory)?;
    writeln!(writer, "{element_count}")?;
    for face in &mesh.boundary_faces {
        let element_type = match face.nodes.len() {
            3 => 2,
            4 => 3,
            _ => unreachable!("mesh validation accepted only Tri3 and Quad4 faces"),
        };
        write!(
            writer,
            "{} {element_type} 2 {} {}",
            face.source_id, face.physical_tag, face.physical_tag
        )?;
        write_nodes(writer, &face.nodes)?;
    }
    for cell in &mesh.cells {
        let element_type = match cell.nodes.len() {
            6 => 6,
            8 => 5,
            _ => unreachable!("mesh validation accepted only Prism6 and Hex8 cells"),
        };
        write!(
            writer,
            "{} {element_type} 2 {} {}",
            cell.source_id, cell.physical_tag, cell.physical_tag
        )?;
        write_nodes(writer, &cell.nodes)?;
    }
    writer.write_all(b"$EndElements\n")?;
    Ok(())
}

fn write_nodes<W: Write>(writer: &mut W, nodes: &[usize]) -> Result<()> {
    for &node in nodes {
        let node_id = node.checked_add(1).ok_or(MeshError::OutOfMemory)?;
        write!(writer, " {node_id}")?;
    }
    writer.write_all(b"\n")?;
    Ok(())
}

fn validate_mesh(mesh: &Mesh) -> Result<()> {
    if !mesh.unsupported_elements.is_empty() {
        return Err(invalid(
            "cannot write a mesh that contains unsupported-element summaries",
        ));
    }
    validate_physical_names(&mesh.physical_names)?;
    for point in &mesh.points {
        if !point.x.is_finite() || !point.y.is_finite() || !point.z.is_finite() {
            return Err(invalid("cannot write a non-finite Gmsh node"));
        }
    }

    let element_count = mesh
        .boundary_faces
        .len()
        .checked_add(mesh.cells.len())
        .ok_or(MeshError::OutOfMemory)?;
    let mut source_ids = HashSet::new();
    source_ids
        .try_reserve(element_count)
        .map_err(|_| MeshError::OutOfMemory)?;

    for face in &mesh.boundary_faces {
        if !matches!(face.nodes.len(), 3 | 4) {
            return Err(invalid(format!(
                "cannot write boundary face {} with {} nodes",
                face.source_id,
                face.nodes.len()
            )));
        }
        validate_element(
            face.source_id,
            face.physical_tag,
            2,
            &face.nodes,
            mesh,
            &mut source_ids,
        )?;
    }
    for cell in &mesh.cells {
        if !matches!(cell.nodes.len(), 6 | 8) {
            return Err(invalid(format!(
                "cannot write cell {} with {} nodes",
                cell.source_id,
                cell.nodes.len()
            )));
        }
        validate_element(
            cell.source_id,
            cell.physical_tag,
            3,
            &cell.nodes,
            mesh,
            &mut source_ids,
        )?;
    }
    Ok(())
}

fn validate_physical_names(names: &[PhysicalName]) -> Result<()> {
    for (index, physical) in names.iter().enumerate() {
        if physical.dim > 3 {
            return Err(invalid(format!(
                "Gmsh physical dimension {} is outside 0 through 3",
                physical.dim
            )));
        }
        if physical.tag <= 0 {
            return Err(invalid("Gmsh physical tags must be greater than zero"));
        }
        if physical.name.contains(['"', '\n', '\r']) {
            return Err(invalid(format!(
                "Gmsh physical name '{}' contains unsupported quoting or a line break",
                physical.name
            )));
        }
        if names[..index]
            .iter()
            .any(|other| other.dim == physical.dim && other.tag == physical.tag)
        {
            return Err(invalid(format!(
                "duplicate Gmsh physical name key ({}, {})",
                physical.dim, physical.tag
            )));
        }
    }
    Ok(())
}

fn validate_element(
    source_id: usize,
    physical_tag: i32,
    dimension: u8,
    nodes: &[usize],
    mesh: &Mesh,
    source_ids: &mut HashSet<usize>,
) -> Result<()> {
    if source_id == 0 {
        return Err(invalid("Gmsh element ids must be greater than zero"));
    }
    if !source_ids.insert(source_id) {
        return Err(invalid(format!("duplicate Gmsh element id {source_id}")));
    }
    if physical_tag <= 0
        || !mesh
            .physical_names
            .iter()
            .any(|physical| physical.dim == dimension && physical.tag == physical_tag)
    {
        return Err(invalid(format!(
            "Gmsh element {source_id} references unknown {dimension}D physical tag {physical_tag}"
        )));
    }
    for &node in nodes {
        if node >= mesh.points.len() {
            return Err(invalid(format!(
                "Gmsh element {source_id} references missing node index {node}"
            )));
        }
    }
    Ok(())
}

fn invalid(message: impl Into<String>) -> MeshError {
    MeshError::InvalidInput(message.into())
}
