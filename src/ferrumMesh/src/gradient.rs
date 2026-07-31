use crate::runtime::SolverRuntimeMeshData;
use crate::{Point3, Result};

use super::compact_faces::CompactSimpleFaceAddressing;
use super::{
    LaminarSimpleGradientScheme, ScalarFaceTreatment, VectorFaceTreatment,
    boundary_normal_distance, checked_delta, checked_dot, checked_magnitude,
    face_scalar_value_with_addressing, invalid_input, require_finite, require_finite_point,
    scalar_component_boundary, split_components, zero,
};

pub(super) struct ScalarGradientGeometry {
    owner_weights: Vec<Option<f64>>,
    boundary_normal_distances: Vec<Option<f64>>,
    inverse_cell_volumes: Vec<f64>,
}

impl ScalarGradientGeometry {
    pub(super) fn from_mesh(mesh: &SolverRuntimeMeshData) -> Result<Self> {
        let mut owner_weights = Vec::with_capacity(mesh.faces);
        let mut boundary_normal_distances = Vec::with_capacity(mesh.faces);
        for face_index in 0..mesh.faces {
            let owner = mesh.owner[face_index];
            if let Some(neighbour) = mesh.neighbour[face_index] {
                owner_weights.push(Some(gauss_linear_owner_weight(
                    mesh, owner, neighbour, face_index,
                )?));
                boundary_normal_distances.push(None);
            } else {
                owner_weights.push(None);
                let distance = boundary_normal_distance(mesh, owner, face_index);
                if !distance.is_finite() {
                    return Err(invalid_input(format!(
                        "boundary face {face_index} normal distance must be finite, got {distance}"
                    )));
                }
                boundary_normal_distances.push(Some(distance));
            }
        }

        let inverse_cell_volumes = mesh
            .cell_volumes
            .iter()
            .copied()
            .enumerate()
            .map(|(cell, volume)| {
                if !volume.is_finite() || volume <= f64::EPSILON {
                    return Err(invalid_input(format!(
                        "scalar gradient cell {cell} has non-positive or non-finite volume {volume}"
                    )));
                }
                Ok(1.0 / volume)
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            owner_weights,
            boundary_normal_distances,
            inverse_cell_volumes,
        })
    }
}

pub(super) fn scalar_gradient_with_geometry_and_addressing(
    mesh: &SolverRuntimeMeshData,
    geometry: &ScalarGradientGeometry,
    face_addressing: &CompactSimpleFaceAddressing,
    values: &[f64],
    boundary: &[ScalarFaceTreatment],
    scheme: LaminarSimpleGradientScheme,
) -> Result<Vec<Point3>> {
    if values.len() != mesh.cells {
        return Err(invalid_input(format!(
            "scalar gradient expected {} cell values, got {}",
            mesh.cells,
            values.len()
        )));
    }
    if boundary.len() != mesh.faces {
        return Err(invalid_input(format!(
            "scalar gradient expected {} boundary treatments, got {}",
            mesh.faces,
            boundary.len()
        )));
    }
    if geometry.owner_weights.len() != mesh.faces
        || geometry.boundary_normal_distances.len() != mesh.faces
        || geometry.inverse_cell_volumes.len() != mesh.cells
    {
        return Err(invalid_input(
            "scalar gradient geometry does not match the runtime mesh".to_string(),
        ));
    }
    for (cell, value) in values.iter().copied().enumerate() {
        if !value.is_finite() {
            return Err(invalid_input(format!(
                "scalar gradient cell {cell} value must be finite, got {value}"
            )));
        }
    }
    let mut gradient = vec![zero(); mesh.cells];
    if face_addressing.faces() != mesh.faces {
        return Err(invalid_input(
            "compact face addressing does not match the runtime mesh".to_string(),
        ));
    }
    for face_index in 0..mesh.faces {
        let owner = face_addressing.owner(face_index);
        let face_value = cached_face_scalar_value_with_addressing(
            geometry,
            face_addressing,
            values,
            boundary,
            face_index,
        )?;
        let area = mesh.face_area_vectors[face_index];
        add_scalar_gradient_contribution(
            &mut gradient[owner],
            area,
            face_value,
            face_index,
            owner,
        )?;
        if let Some(neighbour) = face_addressing.neighbour(face_index) {
            add_scalar_gradient_contribution(
                &mut gradient[neighbour],
                area,
                -face_value,
                face_index,
                neighbour,
            )?;
        }
    }
    for (cell, (value, inverse_volume)) in gradient
        .iter_mut()
        .zip(&geometry.inverse_cell_volumes)
        .enumerate()
    {
        value.x *= inverse_volume;
        value.y *= inverse_volume;
        value.z *= inverse_volume;
        if !value.x.is_finite() || !value.y.is_finite() || !value.z.is_finite() {
            return Err(invalid_input(format!(
                "scalar gradient cell {cell} scaling produced a non-finite component"
            )));
        }
    }
    match scheme {
        LaminarSimpleGradientScheme::GaussLinear => Ok(gradient),
        LaminarSimpleGradientScheme::CellLimitedGaussLinear(coefficient) => {
            limit_scalar_gradient_with_addressing(
                mesh,
                face_addressing,
                values,
                boundary,
                gradient,
                coefficient,
            )
        }
    }
}

fn cached_face_scalar_value_with_addressing(
    geometry: &ScalarGradientGeometry,
    face_addressing: &CompactSimpleFaceAddressing,
    values: &[f64],
    boundary: &[ScalarFaceTreatment],
    face_index: usize,
) -> Result<f64> {
    let owner = face_addressing.owner(face_index);
    let value = if let Some(neighbour) = face_addressing.neighbour(face_index) {
        let weight = geometry.owner_weights[face_index].ok_or_else(|| {
            invalid_input(format!(
                "internal face {face_index} has no cached interpolation weight"
            ))
        })?;
        let owner_part = weight * values[owner];
        if !owner_part.is_finite() {
            return Err(invalid_input(format!(
                "internal face {face_index} owner interpolation must be finite, got {owner_part}"
            )));
        }
        let neighbour_part = (1.0 - weight) * values[neighbour];
        if !neighbour_part.is_finite() {
            return Err(invalid_input(format!(
                "internal face {face_index} neighbour interpolation must be finite, got {neighbour_part}"
            )));
        }
        owner_part + neighbour_part
    } else {
        match boundary[face_index] {
            ScalarFaceTreatment::FixedValue(value) => value,
            ScalarFaceTreatment::FixedGradient(gradient) => {
                let distance = geometry.boundary_normal_distances[face_index].ok_or_else(|| {
                    invalid_input(format!(
                        "boundary face {face_index} has no cached normal distance"
                    ))
                })?;
                let increment = gradient * distance;
                if !increment.is_finite() {
                    return Err(invalid_input(format!(
                        "boundary face {face_index} fixed-gradient extrapolation must be finite, got {increment}"
                    )));
                }
                values[owner] + increment
            }
            ScalarFaceTreatment::InletOutlet(value) => value,
            ScalarFaceTreatment::ZeroGradient | ScalarFaceTreatment::Constraint => values[owner],
        }
    };
    if !value.is_finite() {
        return Err(invalid_input(format!(
            "face {face_index} effective scalar value must be finite, got {value}"
        )));
    }
    Ok(value)
}

fn add_scalar_gradient_contribution(
    target: &mut Point3,
    area: Point3,
    face_value: f64,
    face_index: usize,
    cell: usize,
) -> Result<()> {
    let x = area.x * face_value;
    let y = area.y * face_value;
    let z = area.z * face_value;
    let next_x = target.x + x;
    let next_y = target.y + y;
    let next_z = target.z + z;
    if !x.is_finite()
        || !y.is_finite()
        || !z.is_finite()
        || !next_x.is_finite()
        || !next_y.is_finite()
        || !next_z.is_finite()
    {
        return Err(invalid_input(format!(
            "scalar gradient face {face_index} cell {cell} accumulation produced a non-finite component"
        )));
    }
    target.x = next_x;
    target.y = next_y;
    target.z = next_z;
    Ok(())
}

const V_GREAT: f64 = f64::MAX / 10.0;

pub(super) fn gauss_linear_owner_weight(
    mesh: &SolverRuntimeMeshData,
    owner: usize,
    neighbour: usize,
    face_index: usize,
) -> Result<f64> {
    let face = mesh.face_centres[face_index];
    let owner_delta = checked_delta(
        face,
        mesh.cell_centres[owner],
        format!("internal face {face_index} owner-centre delta"),
    )?;
    let neighbour_delta = checked_delta(
        mesh.cell_centres[neighbour],
        face,
        format!("internal face {face_index} neighbour-centre delta"),
    )?;
    let area = mesh.face_area_vectors[face_index];
    require_finite_point(area, format!("internal face {face_index} area vector"))?;
    let sfd_owner = checked_dot(
        area,
        owner_delta,
        format!("internal face {face_index} projected owner distance"),
    )?
    .abs();
    let sfd_neighbour = checked_dot(
        area,
        neighbour_delta,
        format!("internal face {face_index} projected neighbour distance"),
    )?
    .abs();
    let projected_sum = require_finite(
        sfd_owner + sfd_neighbour,
        format!("internal face {face_index} projected distance sum"),
    )?;

    let weight = if sfd_neighbour / V_GREAT < projected_sum {
        sfd_neighbour / projected_sum
    } else {
        let owner_distance = checked_magnitude(
            owner_delta,
            format!("internal face {face_index} Euclidean owner distance"),
        )?;
        let neighbour_distance = checked_magnitude(
            neighbour_delta,
            format!("internal face {face_index} Euclidean neighbour distance"),
        )?;
        let distance_sum = require_finite(
            owner_distance + neighbour_distance,
            format!("internal face {face_index} Euclidean distance sum"),
        )?;
        if distance_sum <= 0.0 {
            return Err(invalid_input(format!(
                "internal face {face_index} has zero projected and Euclidean centre distance"
            )));
        }
        neighbour_distance / distance_sum
    };
    let weight = require_finite(weight, format!("internal face {face_index} linear weight"))?;
    if !(0.0..=1.0).contains(&weight) {
        return Err(invalid_input(format!(
            "internal face {face_index} linear weight {weight} is outside [0, 1]"
        )));
    }
    Ok(weight)
}

pub(super) fn limit_scalar_gradient_with_addressing(
    mesh: &SolverRuntimeMeshData,
    face_addressing: &CompactSimpleFaceAddressing,
    values: &[f64],
    boundary: &[ScalarFaceTreatment],
    mut gradient: Vec<Point3>,
    coefficient: f64,
) -> Result<Vec<Point3>> {
    if !coefficient.is_finite() || !(0.0..=1.0).contains(&coefficient) {
        return Err(invalid_input(format!(
            "cellLimited gradient coefficient must be finite and in [0, 1], got {coefficient}"
        )));
    }
    if coefficient == 0.0 {
        return Ok(gradient);
    }

    let cell_face_adjacency = face_addressing.limiter_cell_adjacency()?;

    let mut minima = values.to_vec();
    let mut maxima = values.to_vec();
    for face_index in 0..mesh.faces {
        let owner = face_addressing.owner(face_index);
        if let Some(neighbour) = face_addressing.neighbour(face_index) {
            minima[owner] = minima[owner].min(values[neighbour]);
            maxima[owner] = maxima[owner].max(values[neighbour]);
            minima[neighbour] = minima[neighbour].min(values[owner]);
            maxima[neighbour] = maxima[neighbour].max(values[owner]);
        } else {
            let boundary_value = face_scalar_value_with_addressing(
                mesh,
                face_addressing,
                values,
                boundary,
                face_index,
            )?;
            minima[owner] = minima[owner].min(boundary_value);
            maxima[owner] = maxima[owner].max(boundary_value);
        }
    }

    for cell in 0..mesh.cells {
        let maximum_delta = limiter_checked_cell_subtraction(
            maxima[cell],
            values[cell],
            cell,
            "maximum extrema delta",
        )?;
        let minimum_delta = limiter_checked_cell_subtraction(
            minima[cell],
            values[cell],
            cell,
            "minimum extrema delta",
        )?;
        let span =
            limiter_checked_cell_subtraction(maxima[cell], minima[cell], cell, "extrema span")?;
        let widening = if coefficient == 1.0 {
            0.0
        } else {
            let widening_numerator =
                limiter_checked_cell_product(span, 1.0 - coefficient, cell, "widening numerator")?;
            limiter_require_cell_finite(widening_numerator / coefficient, cell, "widening term")?
        };
        let widened_maximum =
            limiter_require_cell_finite(maximum_delta + widening, cell, "widened maximum delta")?;
        let widened_minimum =
            limiter_require_cell_finite(minimum_delta - widening, cell, "widened minimum delta")?;
        let mut limiter: f64 = 1.0;

        for &face_index in cell_face_adjacency.cell_faces(cell) {
            let delta = limiter_checked_face_delta(
                mesh.face_centres[face_index],
                mesh.cell_centres[cell],
                cell,
                face_index,
            )?;
            let extrapolation = limiter_checked_face_dot(gradient[cell], delta, cell, face_index)?;
            let ratio = if extrapolation > widened_maximum && extrapolation > 0.0 {
                widened_maximum / extrapolation
            } else if extrapolation < widened_minimum && extrapolation < 0.0 {
                widened_minimum / extrapolation
            } else {
                1.0
            };
            let ratio = limiter_require_face_finite(ratio, cell, face_index, "limiter ratio")?;
            limiter = limiter.min(ratio.clamp(0.0, 1.0));
            limiter_require_cell_finite(limiter, cell, "final limiter")?;
        }
        limiter_checked_cell_scale(&mut gradient[cell], limiter, cell)?;
    }
    Ok(gradient)
}

#[inline]
fn limiter_require_cell_finite(value: f64, cell: usize, context: &'static str) -> Result<f64> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(invalid_input(format!(
            "cellLimited cell {cell} {context} is non-finite ({value})"
        )))
    }
}

#[inline]
fn limiter_require_face_finite(
    value: f64,
    cell: usize,
    face: usize,
    context: &'static str,
) -> Result<f64> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(invalid_input(format!(
            "cellLimited cell {cell} face {face} {context} is non-finite ({value})"
        )))
    }
}

#[inline]
fn limiter_checked_cell_subtraction(
    left: f64,
    right: f64,
    cell: usize,
    context: &'static str,
) -> Result<f64> {
    if !left.is_finite() {
        return Err(invalid_input(format!(
            "cellLimited cell {cell} {context} left operand is non-finite ({left})"
        )));
    }
    if !right.is_finite() {
        return Err(invalid_input(format!(
            "cellLimited cell {cell} {context} right operand is non-finite ({right})"
        )));
    }
    limiter_require_cell_finite(left - right, cell, context)
}

#[inline]
fn limiter_checked_cell_product(
    left: f64,
    right: f64,
    cell: usize,
    context: &'static str,
) -> Result<f64> {
    if !left.is_finite() {
        return Err(invalid_input(format!(
            "cellLimited cell {cell} {context} left operand is non-finite ({left})"
        )));
    }
    if !right.is_finite() {
        return Err(invalid_input(format!(
            "cellLimited cell {cell} {context} right operand is non-finite ({right})"
        )));
    }
    limiter_require_cell_finite(left * right, cell, context)
}

#[inline]
fn limiter_checked_face_delta(
    left: Point3,
    right: Point3,
    cell: usize,
    face: usize,
) -> Result<Point3> {
    for (component, value) in [("x", left.x), ("y", left.y), ("z", left.z)] {
        limiter_require_face_finite(
            value,
            cell,
            face,
            match component {
                "x" => "centre delta left point x component",
                "y" => "centre delta left point y component",
                _ => "centre delta left point z component",
            },
        )?;
    }
    for (component, value) in [("x", right.x), ("y", right.y), ("z", right.z)] {
        limiter_require_face_finite(
            value,
            cell,
            face,
            match component {
                "x" => "centre delta right point x component",
                "y" => "centre delta right point y component",
                _ => "centre delta right point z component",
            },
        )?;
    }

    Ok(Point3 {
        x: limiter_require_face_finite(left.x - right.x, cell, face, "centre delta x component")?,
        y: limiter_require_face_finite(left.y - right.y, cell, face, "centre delta y component")?,
        z: limiter_require_face_finite(left.z - right.z, cell, face, "centre delta z component")?,
    })
}

#[inline]
fn limiter_checked_face_product(
    left: f64,
    right: f64,
    cell: usize,
    face: usize,
    context: &'static str,
) -> Result<f64> {
    if !left.is_finite() {
        return Err(invalid_input(format!(
            "cellLimited cell {cell} face {face} {context} left operand is non-finite ({left})"
        )));
    }
    if !right.is_finite() {
        return Err(invalid_input(format!(
            "cellLimited cell {cell} face {face} {context} right operand is non-finite ({right})"
        )));
    }
    limiter_require_face_finite(left * right, cell, face, context)
}

#[inline]
fn limiter_checked_face_dot(left: Point3, right: Point3, cell: usize, face: usize) -> Result<f64> {
    let x = limiter_checked_face_product(left.x, right.x, cell, face, "extrapolation x product")?;
    let y = limiter_checked_face_product(left.y, right.y, cell, face, "extrapolation y product")?;
    let z = limiter_checked_face_product(left.z, right.z, cell, face, "extrapolation z product")?;
    let xy = limiter_require_face_finite(x + y, cell, face, "extrapolation x-y sum")?;
    limiter_require_face_finite(xy + z, cell, face, "extrapolation")
}

#[inline]
fn limiter_checked_cell_scale(value: &mut Point3, factor: f64, cell: usize) -> Result<()> {
    value.x = limiter_checked_cell_product(value.x, factor, cell, "limited gradient x component")?;
    value.y = limiter_checked_cell_product(value.y, factor, cell, "limited gradient y component")?;
    value.z = limiter_checked_cell_product(value.z, factor, cell, "limited gradient z component")?;
    Ok(())
}

pub(super) fn vector_component_gradients_with_addressing(
    mesh: &SolverRuntimeMeshData,
    scalar_gradient_geometry: &ScalarGradientGeometry,
    face_addressing: &CompactSimpleFaceAddressing,
    velocity: &[Point3],
    boundary: &[VectorFaceTreatment],
    scheme: LaminarSimpleGradientScheme,
) -> Result<[Vec<Point3>; 3]> {
    let components = split_components(velocity);
    Ok([
        scalar_gradient_with_geometry_and_addressing(
            mesh,
            scalar_gradient_geometry,
            face_addressing,
            &components[0],
            &scalar_component_boundary(boundary, 0),
            scheme,
        )?,
        scalar_gradient_with_geometry_and_addressing(
            mesh,
            scalar_gradient_geometry,
            face_addressing,
            &components[1],
            &scalar_component_boundary(boundary, 1),
            scheme,
        )?,
        scalar_gradient_with_geometry_and_addressing(
            mesh,
            scalar_gradient_geometry,
            face_addressing,
            &components[2],
            &scalar_component_boundary(boundary, 2),
            scheme,
        )?,
    ])
}
