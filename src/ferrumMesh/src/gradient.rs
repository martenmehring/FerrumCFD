use crate::runtime::SolverRuntimeMeshData;
use crate::{Point3, Result};

use super::compact_faces::CompactSimpleFaceAddressing;
use super::{
    LaminarSimpleGradientScheme, ScalarFaceTreatment, VectorFaceTreatment,
    boundary_normal_distance, checked_delta, checked_dot, checked_magnitude, component_value,
    face_scalar_value_with_addressing, invalid_input, require_finite, require_finite_point, zero,
};

pub(super) struct ScalarGradientGeometry {
    owner_weights: Vec<Option<f64>>,
    boundary_normal_distances: Vec<Option<f64>>,
    inverse_cell_volumes: Vec<f64>,
}

impl ScalarGradientGeometry {
    pub(super) fn from_mesh(mesh: &SolverRuntimeMeshData) -> Result<Self> {
        let mut owner_weights = wls_try_vec(mesh.faces)?;
        let mut boundary_normal_distances = wls_try_vec(mesh.faces)?;
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

        let mut inverse_cell_volumes = wls_try_vec(mesh.cells)?;
        for (cell, volume) in mesh.cell_volumes.iter().copied().enumerate() {
            if !volume.is_finite() || volume <= f64::EPSILON {
                return Err(invalid_input(format!(
                    "scalar gradient cell {cell} has non-positive or non-finite volume {volume}"
                )));
            }
            inverse_cell_volumes.push(1.0 / volume);
        }

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
    let mut gradient = wls_try_filled_vec(mesh.cells, zero())?;
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
        LaminarSimpleGradientScheme::LeastSquares => Err(invalid_input(
            "leastSquares gradient requires the scheme-aware dispatcher".to_string(),
        )),
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

    let mut minima = wls_try_vec(values.len())?;
    minima.extend_from_slice(values);
    let mut maxima = wls_try_vec(values.len())?;
    maxima.extend_from_slice(values);
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
    let components = gradient_try_split_components(velocity)?;
    let boundary_x = gradient_try_scalar_component_boundary(boundary, 0)?;
    let gradient_x = scalar_gradient_with_geometry_and_addressing(
        mesh,
        scalar_gradient_geometry,
        face_addressing,
        &components[0],
        &boundary_x,
        scheme,
    )?;
    drop(boundary_x);

    let boundary_y = gradient_try_scalar_component_boundary(boundary, 1)?;
    let gradient_y = scalar_gradient_with_geometry_and_addressing(
        mesh,
        scalar_gradient_geometry,
        face_addressing,
        &components[1],
        &boundary_y,
        scheme,
    )?;
    drop(boundary_y);

    let boundary_z = gradient_try_scalar_component_boundary(boundary, 2)?;
    let gradient_z = scalar_gradient_with_geometry_and_addressing(
        mesh,
        scalar_gradient_geometry,
        face_addressing,
        &components[2],
        &boundary_z,
        scheme,
    )?;

    Ok([gradient_x, gradient_y, gradient_z])
}

fn gradient_try_split_components(values: &[Point3]) -> Result<[Vec<f64>; 3]> {
    let mut x = wls_try_vec(values.len())?;
    let mut y = wls_try_vec(values.len())?;
    let mut z = wls_try_vec(values.len())?;
    for value in values {
        x.push(value.x);
        y.push(value.y);
        z.push(value.z);
    }
    Ok([x, y, z])
}

fn gradient_try_scalar_component_boundary(
    boundary: &[VectorFaceTreatment],
    component: usize,
) -> Result<Vec<ScalarFaceTreatment>> {
    let mut treatments = wls_try_vec(boundary.len())?;
    for treatment in boundary {
        treatments.push(match treatment {
            VectorFaceTreatment::FixedValue(value) => {
                ScalarFaceTreatment::FixedValue(component_value(*value, component))
            }
            VectorFaceTreatment::InletOutlet(value)
            | VectorFaceTreatment::PressureInletOutletVelocity(value) => {
                ScalarFaceTreatment::InletOutlet(component_value(*value, component))
            }
            VectorFaceTreatment::ZeroGradient => ScalarFaceTreatment::ZeroGradient,
            VectorFaceTreatment::Constraint => ScalarFaceTreatment::Constraint,
        });
    }
    Ok(treatments)
}

const WLS_BASIS_INACTIVE_SQUARED: f64 = 64.0 * f64::EPSILON;
const WLS_BASIS_ACTIVE_SQUARED: f64 = 256.0 * f64::EPSILON;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum WlsSquaredMagnitudeBand {
    Inactive,
    Ambiguous,
    Active,
}

fn wls_squared_magnitude_band(value: f64) -> WlsSquaredMagnitudeBand {
    if !value.is_finite() || value >= WLS_BASIS_ACTIVE_SQUARED {
        WlsSquaredMagnitudeBand::Active
    } else if value > WLS_BASIS_INACTIVE_SQUARED {
        WlsSquaredMagnitudeBand::Ambiguous
    } else {
        WlsSquaredMagnitudeBand::Inactive
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WlsIntrinsicDimension {
    One,
    Two,
    Three,
}

impl WlsIntrinsicDimension {
    fn count(self) -> usize {
        match self {
            Self::One => 1,
            Self::Two => 2,
            Self::Three => 3,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WlsFaceRole {
    Internal,
    RegularBoundary,
    EmptyBoundary,
    WedgeBoundary,
    SymmetryBoundary,
}

#[derive(Clone, Copy, Debug)]
struct WlsIntrinsicBasis {
    dimension: WlsIntrinsicDimension,
    axes: [Point3; 3],
}

#[derive(Clone, Copy, Debug)]
struct WlsFaceSample {
    displacement: Point3,
    owner_weight: f64,
    neighbour_weight: f64,
    role: WlsFaceRole,
}

#[derive(Clone, Copy, Debug)]
struct WlsFaceCoefficient {
    owner_cell: usize,
    neighbour_cell: Option<usize>,
    owner: Point3,
    neighbour: Point3,
    role: WlsFaceRole,
}

#[derive(Clone, Copy, Debug)]
enum WlsVectorConstraint {
    Symmetry { unit_normal: Point3 },
    Wedge { rotation: WlsRotation },
}

#[derive(Clone, Copy, Debug)]
struct WlsRotation {
    vector: Point3,
    cosine: f64,
    inverse_one_plus_cosine: f64,
}

/// Immutable weighted least-squares reconstruction geometry.
///
/// This remains private to the SIMPLE implementation. Construction is
/// fallible, solve-local, and only selected by the explicit `leastSquares`
/// scheme; the existing Gauss path does not build or consume this geometry.
#[cfg_attr(not(test), allow(dead_code))]
#[derive(Debug)]
pub(super) struct WeightedLeastSquaresGeometry {
    cells: usize,
    basis: WlsIntrinsicBasis,
    face_coefficients: Vec<WlsFaceCoefficient>,
    wedge_vector_basis: Option<WlsIntrinsicBasis>,
    wedge_vector_face_coefficients: Option<Vec<WlsFaceCoefficient>>,
    vector_constraints: Vec<Option<WlsVectorConstraint>>,
}

#[derive(Clone, Copy, Debug)]
struct WlsDirectionCandidate {
    unit: Point3,
    source_face: usize,
}

#[derive(Clone, Copy, Debug)]
struct WlsCompensatedMoment {
    sum: [f64; 6],
    correction: [f64; 6],
}

impl WlsCompensatedMoment {
    const ZERO: Self = Self {
        sum: [0.0; 6],
        correction: [0.0; 6],
    };

    fn add(
        &mut self,
        row: usize,
        column: usize,
        value: f64,
        cell: usize,
        face: usize,
    ) -> Result<()> {
        let slot = wls_symmetric_slot(row, column);
        let current = self.sum[slot];
        let next = current + value;
        if !next.is_finite() {
            return Err(invalid_input(format!(
                "weighted least-squares cell {cell} face {face} moment accumulation is non-finite"
            )));
        }
        let compensation = if current.abs() >= value.abs() {
            (current - next) + value
        } else {
            (value - next) + current
        };
        let corrected = self.correction[slot] + compensation;
        if !corrected.is_finite() {
            return Err(invalid_input(format!(
                "weighted least-squares cell {cell} face {face} moment compensation is non-finite"
            )));
        }
        self.sum[slot] = next;
        self.correction[slot] = corrected;
        Ok(())
    }

    fn value(self, row: usize, column: usize, cell: usize) -> Result<f64> {
        let slot = wls_symmetric_slot(row, column);
        let value = self.sum[slot] + self.correction[slot];
        if !value.is_finite() {
            return Err(invalid_input(format!(
                "weighted least-squares cell {cell} corrected moment is non-finite"
            )));
        }
        Ok(value)
    }
}

#[derive(Clone, Copy, Debug)]
struct WlsPivotedLdlt {
    dimension: usize,
    permutation: [usize; 3],
    lower: [[f64; 3]; 3],
    diagonal: [f64; 3],
}

#[cfg_attr(not(test), allow(dead_code))]
impl WeightedLeastSquaresGeometry {
    pub(super) fn from_mesh(
        mesh: &SolverRuntimeMeshData,
        scalar_geometry: &ScalarGradientGeometry,
        face_addressing: &CompactSimpleFaceAddressing,
    ) -> Result<Self> {
        wls_validate_structural_lengths(mesh, scalar_geometry, face_addressing)?;
        let roles = wls_face_roles(mesh, face_addressing)?;
        wls_non_empty_wedge_patch_count(mesh)?;
        let (basis, wedge_vector_basis) = wls_intrinsic_bases(mesh, face_addressing, &roles)?;
        let samples = wls_face_samples(mesh, scalar_geometry, face_addressing, &roles, basis)?;
        let face_coefficients =
            wls_face_coefficients(mesh, face_addressing, basis, &samples, false)?;
        let wedge_vector_face_coefficients = wedge_vector_basis
            .map(|vector_basis| {
                wls_face_coefficients(mesh, face_addressing, vector_basis, &samples, true)
            })
            .transpose()?;
        let vector_constraints = wls_vector_constraints(mesh, &roles)?;

        Ok(Self {
            cells: mesh.cells,
            basis,
            face_coefficients,
            wedge_vector_basis,
            wedge_vector_face_coefficients,
            vector_constraints,
        })
    }

    #[cfg(test)]
    fn storage_identity(&self) -> [usize; 9] {
        let vector = self.wedge_vector_face_coefficients.as_ref();
        [
            self.face_coefficients.as_ptr() as usize,
            self.face_coefficients.len(),
            self.face_coefficients.capacity(),
            vector.map_or(0, |coefficients| coefficients.as_ptr() as usize),
            vector.map_or(0, Vec::len),
            vector.map_or(0, Vec::capacity),
            self.vector_constraints.as_ptr() as usize,
            self.vector_constraints.len(),
            self.vector_constraints.capacity(),
        ]
    }
}

fn wls_validate_structural_lengths(
    mesh: &SolverRuntimeMeshData,
    scalar_geometry: &ScalarGradientGeometry,
    face_addressing: &CompactSimpleFaceAddressing,
) -> Result<()> {
    if mesh.cells == 0 {
        return Err(invalid_input(
            "weighted least-squares geometry requires at least one cell".to_string(),
        ));
    }
    if mesh.owner.len() != mesh.faces
        || mesh.neighbour.len() != mesh.faces
        || mesh.face_centres.len() != mesh.faces
        || mesh.face_area_vectors.len() != mesh.faces
    {
        return Err(invalid_input(format!(
            "weighted least-squares geometry requires {} owner, neighbour, face-centre, and face-area entries",
            mesh.faces
        )));
    }
    if mesh.cell_centres.len() != mesh.cells {
        return Err(invalid_input(format!(
            "weighted least-squares geometry requires {} cell centres, got {}",
            mesh.cells,
            mesh.cell_centres.len()
        )));
    }
    if face_addressing.faces() != mesh.faces {
        return Err(invalid_input(format!(
            "weighted least-squares addressing expected {} faces, got {}",
            mesh.faces,
            face_addressing.faces()
        )));
    }
    if scalar_geometry.owner_weights.len() != mesh.faces {
        return Err(invalid_input(format!(
            "weighted least-squares interpolation geometry expected {} face weights, got {}",
            mesh.faces,
            scalar_geometry.owner_weights.len()
        )));
    }
    for face in 0..mesh.faces {
        if face_addressing.owner(face) != mesh.owner[face]
            || face_addressing.neighbour(face) != mesh.neighbour[face]
        {
            return Err(invalid_input(format!(
                "weighted least-squares addressing does not match runtime mesh face {face}"
            )));
        }
    }
    for (cell, centre) in mesh.cell_centres.iter().copied().enumerate() {
        wls_require_finite_point(centre, || {
            format!("weighted least-squares cell {cell} centre")
        })?;
    }
    Ok(())
}

fn wls_face_roles(
    mesh: &SolverRuntimeMeshData,
    face_addressing: &CompactSimpleFaceAddressing,
) -> Result<Vec<Option<WlsFaceRole>>> {
    let mut roles = wls_try_filled_vec(mesh.faces, None)?;
    for (patch_index, patch) in mesh.patches.iter().enumerate() {
        let end = patch
            .start_face
            .checked_add(patch.faces)
            .ok_or(crate::MeshError::OutOfMemory)?;
        if end > mesh.faces {
            return Err(invalid_input(format!(
                "weighted least-squares patch {patch_index} '{}' range {}..{} exceeds {} faces",
                patch.name, patch.start_face, end, mesh.faces
            )));
        }
        let role = match patch.patch_type.as_str() {
            "empty" => WlsFaceRole::EmptyBoundary,
            "wedge" => WlsFaceRole::WedgeBoundary,
            "symmetryPlane" => WlsFaceRole::SymmetryBoundary,
            _ => WlsFaceRole::RegularBoundary,
        };
        for (face, face_role) in roles
            .iter_mut()
            .enumerate()
            .take(end)
            .skip(patch.start_face)
        {
            if face_addressing.neighbour(face).is_some() {
                return Err(invalid_input(format!(
                    "weighted least-squares patch {patch_index} '{}' contains internal face {face}",
                    patch.name
                )));
            }
            if face_role.replace(role).is_some() {
                return Err(invalid_input(format!(
                    "weighted least-squares boundary face {face} is covered by more than one patch"
                )));
            }
        }
    }

    for (face, face_role) in roles.iter_mut().enumerate() {
        if let Some(neighbour) = face_addressing.neighbour(face) {
            let owner = face_addressing.owner(face);
            if owner == neighbour {
                return Err(invalid_input(format!(
                    "weighted least-squares internal face {face} has identical owner and neighbour cell {owner}"
                )));
            }
            if face_role.is_some() {
                return Err(invalid_input(format!(
                    "weighted least-squares internal face {face} has a boundary patch role"
                )));
            }
            *face_role = Some(WlsFaceRole::Internal);
        } else if face_role.is_none() {
            return Err(invalid_input(format!(
                "weighted least-squares boundary face {face} is not covered by a patch"
            )));
        }
    }
    Ok(roles)
}

fn wls_vector_constraints(
    mesh: &SolverRuntimeMeshData,
    roles: &[Option<WlsFaceRole>],
) -> Result<Vec<Option<WlsVectorConstraint>>> {
    let mut constraints = wls_try_filled_vec(mesh.faces, None)?;
    let mut patch_constraints = wls_try_filled_vec(mesh.patches.len(), None)?;
    let wedge_patch_count = wls_non_empty_wedge_patch_count(mesh)?;
    let mut wedge_patches = wls_try_vec(wedge_patch_count)?;
    for (patch_index, patch) in mesh.patches.iter().enumerate() {
        if patch.faces == 0 {
            continue;
        }
        let constraint = match patch.patch_type.as_str() {
            "symmetryPlane" => Some(WlsVectorConstraint::Symmetry {
                unit_normal: wls_patch_average_unit_normal(mesh, patch_index)?,
            }),
            "wedge" => {
                let patch_normal = wls_patch_average_unit_normal(mesh, patch_index)?;
                let centre_normal = wls_coordinate_plane_normal(patch_normal, patch_index)?;
                let rotation = wls_rotation_between(centre_normal, patch_normal, patch_index)?;
                wedge_patches.push((patch_index, centre_normal, rotation));
                Some(WlsVectorConstraint::Wedge { rotation })
            }
            _ => None,
        };
        patch_constraints[patch_index] = constraint;
    }
    wls_validate_wedge_pair(mesh, &wedge_patches)?;

    for (patch_index, patch) in mesh.patches.iter().enumerate() {
        let Some(constraint) = patch_constraints[patch_index] else {
            continue;
        };
        let end = patch
            .start_face
            .checked_add(patch.faces)
            .ok_or(crate::MeshError::OutOfMemory)?;
        for (face, slot) in constraints
            .iter_mut()
            .enumerate()
            .take(end)
            .skip(patch.start_face)
        {
            let expected_role = match constraint {
                WlsVectorConstraint::Symmetry { .. } => WlsFaceRole::SymmetryBoundary,
                WlsVectorConstraint::Wedge { .. } => WlsFaceRole::WedgeBoundary,
            };
            if roles.get(face).copied().flatten() != Some(expected_role) {
                return Err(invalid_input(format!(
                    "weighted least-squares constraint metadata disagrees with face {face} role"
                )));
            }
            *slot = Some(constraint);
        }
    }
    Ok(constraints)
}

fn wls_non_empty_wedge_patch_count(mesh: &SolverRuntimeMeshData) -> Result<usize> {
    let count = mesh
        .patches
        .iter()
        .filter(|patch| patch.patch_type == "wedge" && patch.faces != 0)
        .count();
    if count != 0 && count != 2 {
        return Err(invalid_input(format!(
            "weighted least-squares wedge topology requires exactly two non-empty wedge patches, got {count}"
        )));
    }
    Ok(count)
}

fn wls_patch_average_unit_normal(
    mesh: &SolverRuntimeMeshData,
    patch_index: usize,
) -> Result<Point3> {
    let patch = &mesh.patches[patch_index];
    if patch.faces == 0 {
        return Err(invalid_input(format!(
            "weighted least-squares constraint patch {patch_index} '{}' has no faces",
            patch.name
        )));
    }
    let end = patch
        .start_face
        .checked_add(patch.faces)
        .ok_or(crate::MeshError::OutOfMemory)?;
    let mut normals = wls_try_vec(patch.faces)?;
    for face in patch.start_face..end {
        let unit = wls_normalize_oriented(mesh.face_area_vectors[face], || {
            format!("weighted least-squares constraint patch {patch_index} face {face} normal")
        })?;
        normals.push((unit, face));
    }
    normals.sort_unstable_by(|(left, left_face), (right, right_face)| {
        left.x
            .total_cmp(&right.x)
            .then_with(|| left.y.total_cmp(&right.y))
            .then_with(|| left.z.total_cmp(&right.z))
            .then_with(|| left_face.cmp(right_face))
    });
    let mut x = WlsCompensatedScalar::ZERO;
    let mut y = WlsCompensatedScalar::ZERO;
    let mut z = WlsCompensatedScalar::ZERO;
    for (normal, _) in &normals {
        x.add(normal.x, patch_index, "normal x")?;
        y.add(normal.y, patch_index, "normal y")?;
        z.add(normal.z, patch_index, "normal z")?;
    }
    let average = wls_normalize_oriented(
        Point3 {
            x: x.value(patch_index, "normal x")?,
            y: y.value(patch_index, "normal y")?,
            z: z.value(patch_index, "normal z")?,
        },
        || {
            format!(
                "weighted least-squares constraint patch {patch_index} '{}' average normal",
                patch.name
            )
        },
    )?;
    for (normal, face) in normals {
        let mismatch = wls_checked_delta(normal, average, || {
            format!("weighted least-squares constraint patch {patch_index} face {face} planarity")
        })?;
        let mismatch_squared = wls_checked_dot(mismatch, mismatch, || {
            format!(
                "weighted least-squares constraint patch {patch_index} face {face} planarity error"
            )
        })?;
        match wls_squared_magnitude_band(mismatch_squared) {
            WlsSquaredMagnitudeBand::Active => {
                return Err(invalid_input(format!(
                    "weighted least-squares constraint patch {patch_index} face {face} is non-planar"
                )));
            }
            WlsSquaredMagnitudeBand::Ambiguous => {
                return Err(invalid_input(format!(
                    "weighted least-squares constraint patch {patch_index} face {face} planarity is numerically ambiguous"
                )));
            }
            WlsSquaredMagnitudeBand::Inactive => {}
        }
    }
    Ok(average)
}

fn wls_coordinate_plane_normal(normal: Point3, patch_index: usize) -> Result<Point3> {
    let components = [normal.x, normal.y, normal.z]
        .map(|component| component.signum() * (component.abs() - 0.5).max(0.0));
    let inferred = wls_normalize_oriented(
        Point3 {
            x: components[0],
            y: components[1],
            z: components[2],
        },
        || format!("weighted least-squares wedge patch {patch_index} centre-plane normal"),
    )?;
    let components = [inferred.x, inferred.y, inferred.z];
    let mut dominant = 0usize;
    for index in 1..3 {
        if components[index].abs() > components[dominant].abs() {
            dominant = index;
        }
    }
    let off_axis_squared = components
        .iter()
        .enumerate()
        .filter(|(index, _)| *index != dominant)
        .map(|(_, component)| component * component)
        .sum::<f64>();
    if !off_axis_squared.is_finite() {
        return Err(invalid_input(format!(
            "weighted least-squares wedge patch {patch_index} centre-plane alignment is non-finite"
        )));
    }
    match wls_squared_magnitude_band(off_axis_squared) {
        WlsSquaredMagnitudeBand::Active => {
            return Err(invalid_input(format!(
                "weighted least-squares wedge patch {patch_index} centre plane does not align with a coordinate plane"
            )));
        }
        WlsSquaredMagnitudeBand::Ambiguous => {
            return Err(invalid_input(format!(
                "weighted least-squares wedge patch {patch_index} centre-plane alignment is numerically ambiguous"
            )));
        }
        WlsSquaredMagnitudeBand::Inactive => {}
    }
    let sign = if components[dominant].is_sign_negative() {
        -1.0
    } else {
        1.0
    };
    Ok(match dominant {
        0 => Point3 {
            x: sign,
            y: 0.0,
            z: 0.0,
        },
        1 => Point3 {
            x: 0.0,
            y: sign,
            z: 0.0,
        },
        _ => Point3 {
            x: 0.0,
            y: 0.0,
            z: sign,
        },
    })
}

fn wls_rotation_between(from: Point3, to: Point3, patch_index: usize) -> Result<WlsRotation> {
    let cosine = wls_checked_dot(from, to, || {
        format!("weighted least-squares wedge patch {patch_index} rotation cosine")
    })?;
    let vector = wls_checked_cross(from, to, || {
        format!("weighted least-squares wedge patch {patch_index} rotation vector")
    })?;
    let sine_squared = wls_checked_dot(vector, vector, || {
        format!("weighted least-squares wedge patch {patch_index} squared rotation sine")
    })?;
    match wls_squared_magnitude_band(sine_squared) {
        WlsSquaredMagnitudeBand::Inactive => {
            return Err(invalid_input(format!(
                "weighted least-squares wedge patch {patch_index} plane aligns with its coordinate centre plane"
            )));
        }
        WlsSquaredMagnitudeBand::Ambiguous => {
            return Err(invalid_input(format!(
                "weighted least-squares wedge patch {patch_index} rotation is numerically ambiguous"
            )));
        }
        WlsSquaredMagnitudeBand::Active => {}
    }
    if !cosine.is_finite() || cosine <= 0.0 {
        return Err(invalid_input(format!(
            "weighted least-squares wedge patch {patch_index} rotation must have a finite positive cosine"
        )));
    }
    let unit_identity_error = (cosine * cosine + sine_squared - 1.0).abs();
    match wls_squared_magnitude_band(unit_identity_error) {
        WlsSquaredMagnitudeBand::Active => {
            return Err(invalid_input(format!(
                "weighted least-squares wedge patch {patch_index} rotation violates the unit-vector identity"
            )));
        }
        WlsSquaredMagnitudeBand::Ambiguous => {
            return Err(invalid_input(format!(
                "weighted least-squares wedge patch {patch_index} unit-vector identity is numerically ambiguous"
            )));
        }
        WlsSquaredMagnitudeBand::Inactive => {}
    }
    let inverse_one_plus_cosine = 1.0 / (1.0 + cosine);
    if !inverse_one_plus_cosine.is_finite() {
        return Err(invalid_input(format!(
            "weighted least-squares wedge patch {patch_index} rotation denominator is invalid"
        )));
    }
    let rotation = WlsRotation {
        vector,
        cosine,
        inverse_one_plus_cosine,
    };
    let mapped_delta = rotation.delta(from, patch_index)?;
    let mapped = wls_checked_add(from, mapped_delta, || {
        format!("weighted least-squares wedge patch {patch_index} mapped centre-plane normal")
    })?;
    let mismatch = wls_checked_delta(mapped, to, || {
        format!("weighted least-squares wedge patch {patch_index} rotation verification")
    })?;
    let mismatch_squared = wls_checked_dot(mismatch, mismatch, || {
        format!("weighted least-squares wedge patch {patch_index} squared rotation mismatch")
    })?;
    match wls_squared_magnitude_band(mismatch_squared) {
        WlsSquaredMagnitudeBand::Active => {
            return Err(invalid_input(format!(
                "weighted least-squares wedge patch {patch_index} rotation does not map the centre-plane normal"
            )));
        }
        WlsSquaredMagnitudeBand::Ambiguous => {
            return Err(invalid_input(format!(
                "weighted least-squares wedge patch {patch_index} rotation mapping is numerically ambiguous"
            )));
        }
        WlsSquaredMagnitudeBand::Inactive => {}
    }
    Ok(rotation)
}

impl WlsRotation {
    fn delta(self, value: Point3, patch_index: usize) -> Result<Point3> {
        wls_require_finite_point(value, || {
            format!("weighted least-squares wedge patch {patch_index} vector value")
        })?;
        let sine_squared = wls_checked_dot(self.vector, self.vector, || {
            format!("weighted least-squares wedge patch {patch_index} squared rotation sine")
        })?;
        let vector_component = wls_checked_dot(self.vector, value, || {
            format!("weighted least-squares wedge patch {patch_index} rotation projection")
        })?;
        let mut delta = wls_checked_cross(self.vector, value, || {
            format!("weighted least-squares wedge patch {patch_index} rotation cross contribution")
        })?;
        wls_add_scaled_point(
            &mut delta,
            self.vector,
            vector_component * self.inverse_one_plus_cosine,
            usize::MAX,
            patch_index,
        )?;
        wls_add_scaled_point(
            &mut delta,
            value,
            -sine_squared * self.inverse_one_plus_cosine,
            usize::MAX,
            patch_index,
        )?;
        wls_require_finite_point(delta, || {
            format!("weighted least-squares wedge patch {patch_index} vector delta")
        })
    }
}

fn wls_validate_wedge_pair(
    mesh: &SolverRuntimeMeshData,
    wedge_patches: &[(usize, Point3, WlsRotation)],
) -> Result<()> {
    if wedge_patches.is_empty() {
        return Ok(());
    }
    if wedge_patches.len() != 2 {
        return Err(invalid_input(format!(
            "weighted least-squares wedge topology requires exactly two non-empty wedge patches, got {}",
            wedge_patches.len()
        )));
    }
    let (first_index, first_centre, first_rotation) = wedge_patches[0];
    let (second_index, second_centre, second_rotation) = wedge_patches[1];
    let centre_sum = wls_checked_add(first_centre, second_centre, || {
        format!(
            "weighted least-squares wedge patches {first_index} and {second_index} centre-plane symmetry"
        )
    })?;
    let centre_error = wls_checked_dot(centre_sum, centre_sum, || {
        format!(
            "weighted least-squares wedge patches {first_index} and {second_index} squared centre-plane symmetry error"
        )
    })?;
    let vector_sum = wls_checked_add(first_rotation.vector, second_rotation.vector, || {
        format!(
            "weighted least-squares wedge patches {first_index} and {second_index} rotation symmetry"
        )
    })?;
    let vector_error = wls_checked_dot(vector_sum, vector_sum, || {
        format!(
            "weighted least-squares wedge patches {first_index} and {second_index} squared rotation-vector symmetry error"
        )
    })?;
    let cosine_error = first_rotation.cosine - second_rotation.cosine;
    wls_require_pair_error(centre_error, first_index, second_index, "centre-plane")?;
    wls_require_pair_error(vector_error, first_index, second_index, "rotation-vector")?;
    wls_require_pair_error(
        cosine_error * cosine_error,
        first_index,
        second_index,
        "rotation-cosine",
    )?;
    let first_owners = wls_wedge_patch_owners(mesh, first_index)?;
    let second_owners = wls_wedge_patch_owners(mesh, second_index)?;
    if first_owners != second_owners {
        return Err(invalid_input(format!(
            "weighted least-squares wedge patches {first_index} and {second_index} do not cover the same owner cells"
        )));
    }
    Ok(())
}

fn wls_require_pair_error(
    squared_error: f64,
    first_patch: usize,
    second_patch: usize,
    property: &str,
) -> Result<()> {
    match wls_squared_magnitude_band(squared_error) {
        WlsSquaredMagnitudeBand::Active => {
            return Err(invalid_input(format!(
                "weighted least-squares wedge patches {first_patch} and {second_patch} {property} values are not symmetric"
            )));
        }
        WlsSquaredMagnitudeBand::Ambiguous => {
            return Err(invalid_input(format!(
                "weighted least-squares wedge patches {first_patch} and {second_patch} {property} symmetry is numerically ambiguous"
            )));
        }
        WlsSquaredMagnitudeBand::Inactive => {}
    }
    Ok(())
}

fn wls_wedge_patch_owners(mesh: &SolverRuntimeMeshData, patch_index: usize) -> Result<Vec<usize>> {
    let patch = &mesh.patches[patch_index];
    let end = patch
        .start_face
        .checked_add(patch.faces)
        .ok_or(crate::MeshError::OutOfMemory)?;
    let mut owners = wls_try_vec(patch.faces)?;
    owners.extend_from_slice(&mesh.owner[patch.start_face..end]);
    owners.sort_unstable();
    if owners.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(invalid_input(format!(
            "weighted least-squares wedge patch {patch_index} contains more than one face for an owner cell"
        )));
    }
    Ok(owners)
}

fn wls_intrinsic_bases(
    mesh: &SolverRuntimeMeshData,
    face_addressing: &CompactSimpleFaceAddressing,
    roles: &[Option<WlsFaceRole>],
) -> Result<(WlsIntrinsicBasis, Option<WlsIntrinsicBasis>)> {
    let mut active_candidates = wls_try_vec(mesh.faces)?;
    let mut vector_candidates = wls_try_vec(mesh.faces)?;
    let mut empty_normals = wls_try_vec(mesh.faces)?;
    let mut wedge_normals = wls_try_vec(mesh.faces)?;
    let mut has_empty = false;
    let mut has_wedge = false;

    for (face, face_role) in roles.iter().copied().enumerate() {
        let role = face_role.ok_or_else(|| {
            invalid_input(format!(
                "weighted least-squares face {face} has no classified role"
            ))
        })?;
        let area = mesh.face_area_vectors[face];
        let area_unit = wls_normalize_direction(area, face, "area vector")?;
        match role {
            WlsFaceRole::EmptyBoundary => {
                has_empty = true;
                empty_normals.push(WlsDirectionCandidate {
                    unit: area_unit,
                    source_face: face,
                });
            }
            WlsFaceRole::WedgeBoundary => {
                has_wedge = true;
                wedge_normals.push(WlsDirectionCandidate {
                    unit: area_unit,
                    source_face: face,
                });
                let displacement = wls_face_displacement(mesh, face_addressing, face, role)?;
                vector_candidates.push(WlsDirectionCandidate {
                    unit: wls_normalize_direction(displacement, face, "displacement")?,
                    source_face: face,
                });
            }
            WlsFaceRole::Internal
            | WlsFaceRole::RegularBoundary
            | WlsFaceRole::SymmetryBoundary => {
                let displacement = wls_face_displacement(mesh, face_addressing, face, role)?;
                let candidate = WlsDirectionCandidate {
                    unit: wls_normalize_direction(displacement, face, "displacement")?,
                    source_face: face,
                };
                active_candidates.push(candidate);
                vector_candidates.push(candidate);
            }
        }
    }

    if has_empty && has_wedge {
        return Err(invalid_input(
            "weighted least-squares intrinsic dimension is ambiguous because empty and wedge patches are mixed"
                .to_string(),
        ));
    }

    let expected = if has_wedge {
        wls_non_empty_wedge_patch_count(mesh)?;
        let wedge_normal_rank = wls_direction_rank(&mut wedge_normals, "wedge constraint normals")?;
        if wedge_normal_rank != 2 {
            return Err(invalid_input(format!(
                "weighted least-squares wedge constraint normals must have rank 2, got {wedge_normal_rank}"
            )));
        }
        WlsIntrinsicDimension::Two
    } else if has_empty {
        let empty_rank = wls_direction_rank(&mut empty_normals, "empty constraint normals")?;
        match empty_rank {
            1 => WlsIntrinsicDimension::Two,
            2 => WlsIntrinsicDimension::One,
            _ => {
                return Err(invalid_input(format!(
                    "weighted least-squares empty constraint normals must have rank 1 or 2, got {empty_rank}"
                )));
            }
        }
    } else {
        WlsIntrinsicDimension::Three
    };

    let axes = wls_build_expected_basis(
        &mut active_candidates,
        expected.count(),
        "active mesh directions",
    )?;
    let basis = WlsIntrinsicBasis {
        dimension: expected,
        axes,
    };
    let wedge_vector_basis = if has_wedge {
        Some(WlsIntrinsicBasis {
            dimension: WlsIntrinsicDimension::Three,
            axes: wls_build_expected_basis(
                &mut vector_candidates,
                WlsIntrinsicDimension::Three.count(),
                "wedge vector directions",
            )?,
        })
    } else {
        None
    };
    Ok((basis, wedge_vector_basis))
}

fn wls_face_samples(
    mesh: &SolverRuntimeMeshData,
    scalar_geometry: &ScalarGradientGeometry,
    face_addressing: &CompactSimpleFaceAddressing,
    roles: &[Option<WlsFaceRole>],
    basis: WlsIntrinsicBasis,
) -> Result<Vec<WlsFaceSample>> {
    let mut samples = wls_try_vec(mesh.faces)?;
    for (face, face_role) in roles.iter().copied().enumerate() {
        let role = face_role.ok_or_else(|| {
            invalid_input(format!(
                "weighted least-squares face {face} has no classified role"
            ))
        })?;
        if role == WlsFaceRole::EmptyBoundary {
            samples.push(WlsFaceSample {
                displacement: zero(),
                owner_weight: 0.0,
                neighbour_weight: 0.0,
                role,
            });
            continue;
        }

        let displacement = wls_face_displacement(mesh, face_addressing, face, role)?;
        let distance_squared = wls_checked_dot(displacement, displacement, || {
            format!("weighted least-squares face {face} displacement square")
        })?;
        if distance_squared <= 0.0 {
            return Err(invalid_input(format!(
                "weighted least-squares face {face} displacement must be nonzero"
            )));
        }
        let area_magnitude = wls_checked_magnitude(mesh.face_area_vectors[face], || {
            format!("weighted least-squares face {face} area magnitude")
        })?;
        if area_magnitude <= 0.0 {
            return Err(invalid_input(format!(
                "weighted least-squares face {face} area magnitude must be positive"
            )));
        }
        let base_weight = area_magnitude / distance_squared;
        if !base_weight.is_finite() || base_weight <= 0.0 {
            return Err(invalid_input(format!(
                "weighted least-squares face {face} base weight must be positive and finite, got {base_weight}"
            )));
        }

        if basis.dimension != WlsIntrinsicDimension::Three && role != WlsFaceRole::WedgeBoundary {
            wls_require_in_active_subspace(displacement, basis, face)?;
        }

        let (owner_weight, neighbour_weight) = if role == WlsFaceRole::Internal {
            let owner_fraction = scalar_geometry.owner_weights[face].ok_or_else(|| {
                invalid_input(format!(
                    "weighted least-squares internal face {face} has no interpolation weight"
                ))
            })?;
            if !owner_fraction.is_finite() || !(0.0..=1.0).contains(&owner_fraction) {
                return Err(invalid_input(format!(
                    "weighted least-squares internal face {face} interpolation weight {owner_fraction} is outside [0, 1]"
                )));
            }
            let owner_weight = (1.0 - owner_fraction) * base_weight;
            let neighbour_weight = owner_fraction * base_weight;
            if !owner_weight.is_finite() || !neighbour_weight.is_finite() {
                return Err(invalid_input(format!(
                    "weighted least-squares internal face {face} split weight is non-finite"
                )));
            }
            (owner_weight, neighbour_weight)
        } else {
            (base_weight, 0.0)
        };

        samples.push(WlsFaceSample {
            displacement,
            owner_weight,
            neighbour_weight,
            role,
        });
    }
    Ok(samples)
}

fn wls_face_coefficients(
    mesh: &SolverRuntimeMeshData,
    face_addressing: &CompactSimpleFaceAddressing,
    basis: WlsIntrinsicBasis,
    samples: &[WlsFaceSample],
    include_wedge: bool,
) -> Result<Vec<WlsFaceCoefficient>> {
    let mut maximum_distance = wls_try_filled_vec(mesh.cells, 0.0f64)?;
    let mut maximum_weight = wls_try_filled_vec(mesh.cells, 0.0f64)?;

    for (face, sample) in samples.iter().copied().enumerate() {
        if wls_sample_is_omitted(sample.role, include_wedge) {
            continue;
        }
        let distance = wls_checked_magnitude(sample.displacement, || {
            format!("weighted least-squares face {face} displacement magnitude")
        })?;
        let owner = face_addressing.owner(face);
        wls_update_cell_scales(
            &mut maximum_distance,
            &mut maximum_weight,
            owner,
            distance,
            sample.owner_weight,
        );
        if let Some(neighbour) = face_addressing.neighbour(face) {
            wls_update_cell_scales(
                &mut maximum_distance,
                &mut maximum_weight,
                neighbour,
                distance,
                sample.neighbour_weight,
            );
        }
    }
    for cell in 0..mesh.cells {
        if maximum_distance[cell] <= 0.0 || maximum_weight[cell] <= 0.0 {
            return Err(invalid_input(format!(
                "weighted least-squares cell {cell} has no positive finite reconstruction scale"
            )));
        }
    }

    let mut moments = wls_try_filled_vec(mesh.cells, WlsCompensatedMoment::ZERO)?;
    let mut observation_counts = wls_try_filled_vec(mesh.cells, 0usize)?;
    for (face, sample) in samples.iter().copied().enumerate() {
        if wls_sample_is_omitted(sample.role, include_wedge) {
            continue;
        }
        let owner = face_addressing.owner(face);
        wls_accumulate_cell_moment(
            &mut moments,
            &mut observation_counts,
            owner,
            face,
            sample.displacement,
            sample.owner_weight,
            maximum_distance[owner],
            maximum_weight[owner],
            basis,
        )?;
        if let Some(neighbour) = face_addressing.neighbour(face) {
            wls_accumulate_cell_moment(
                &mut moments,
                &mut observation_counts,
                neighbour,
                face,
                sample.displacement,
                sample.neighbour_weight,
                maximum_distance[neighbour],
                maximum_weight[neighbour],
                basis,
            )?;
        }
    }

    let mut factors = wls_try_vec(mesh.cells)?;
    let mut moment_diagonal_scales = wls_try_vec(mesh.cells)?;
    for cell in 0..mesh.cells {
        let (factor, diagonal_scale) = wls_factor_cell_moment(
            moments[cell],
            basis.dimension.count(),
            observation_counts[cell],
            cell,
        )?;
        factors.push(factor);
        moment_diagonal_scales.push(diagonal_scale);
    }

    let mut coefficients = wls_try_vec(mesh.faces)?;
    for (face, sample) in samples.iter().copied().enumerate() {
        let owner_cell = face_addressing.owner(face);
        let neighbour_cell = face_addressing.neighbour(face);
        if wls_sample_is_omitted(sample.role, include_wedge) {
            coefficients.push(WlsFaceCoefficient {
                owner_cell,
                neighbour_cell,
                owner: zero(),
                neighbour: zero(),
                role: sample.role,
            });
            continue;
        }
        let owner = owner_cell;
        let owner_vector = wls_reconstruction_vector(
            sample.displacement,
            sample.owner_weight,
            maximum_distance[owner],
            maximum_weight[owner],
            moment_diagonal_scales[owner],
            basis,
            factors[owner],
            owner,
            face,
        )?;
        let neighbour_vector = if let Some(neighbour) = neighbour_cell {
            wls_reconstruction_vector(
                sample.displacement,
                sample.neighbour_weight,
                maximum_distance[neighbour],
                maximum_weight[neighbour],
                moment_diagonal_scales[neighbour],
                basis,
                factors[neighbour],
                neighbour,
                face,
            )?
        } else {
            zero()
        };
        coefficients.push(WlsFaceCoefficient {
            owner_cell,
            neighbour_cell,
            owner: owner_vector,
            neighbour: neighbour_vector,
            role: sample.role,
        });
    }
    Ok(coefficients)
}

fn wls_sample_is_omitted(role: WlsFaceRole, include_wedge: bool) -> bool {
    role == WlsFaceRole::EmptyBoundary || (role == WlsFaceRole::WedgeBoundary && !include_wedge)
}

fn wls_update_cell_scales(
    maximum_distance: &mut [f64],
    maximum_weight: &mut [f64],
    cell: usize,
    distance: f64,
    weight: f64,
) {
    if weight > 0.0 {
        maximum_distance[cell] = maximum_distance[cell].max(distance);
        maximum_weight[cell] = maximum_weight[cell].max(weight);
    }
}

#[allow(clippy::too_many_arguments)]
fn wls_accumulate_cell_moment(
    moments: &mut [WlsCompensatedMoment],
    observation_counts: &mut [usize],
    cell: usize,
    face: usize,
    displacement: Point3,
    weight: f64,
    distance_scale: f64,
    weight_scale: f64,
    basis: WlsIntrinsicBasis,
) -> Result<()> {
    if weight == 0.0 {
        return Ok(());
    }
    if !weight.is_finite() || weight < 0.0 {
        return Err(invalid_input(format!(
            "weighted least-squares cell {cell} face {face} weight must be non-negative and finite, got {weight}"
        )));
    }
    let scaled = wls_scale_point(displacement, 1.0 / distance_scale, || {
        format!("weighted least-squares cell {cell} face {face} scaled displacement")
    })?;
    let projected = wls_project_onto_basis(scaled, basis, face)?;
    let scaled_weight = weight / weight_scale;
    if !scaled_weight.is_finite() || scaled_weight <= 0.0 {
        return Err(invalid_input(format!(
            "weighted least-squares cell {cell} face {face} scaled weight must be positive and finite"
        )));
    }
    let dimension = basis.dimension.count();
    for row in 0..dimension {
        for column in row..dimension {
            let value = scaled_weight * projected[row] * projected[column];
            if !value.is_finite() {
                return Err(invalid_input(format!(
                    "weighted least-squares cell {cell} face {face} moment term is non-finite"
                )));
            }
            moments[cell].add(row, column, value, cell, face)?;
        }
    }
    observation_counts[cell] = observation_counts[cell]
        .checked_add(1)
        .ok_or(crate::MeshError::OutOfMemory)?;
    Ok(())
}

#[allow(clippy::needless_range_loop)]
fn wls_factor_cell_moment(
    moment: WlsCompensatedMoment,
    dimension: usize,
    observations: usize,
    cell: usize,
) -> Result<(WlsPivotedLdlt, f64)> {
    let mut matrix = [[0.0; 3]; 3];
    for row in 0..dimension {
        for column in row..dimension {
            let value = moment.value(row, column, cell)?;
            matrix[row][column] = value;
            matrix[column][row] = value;
        }
    }
    let diagonal_scale = (0..dimension)
        .map(|index| matrix[index][index])
        .fold(0.0f64, f64::max);
    if !diagonal_scale.is_finite() || diagonal_scale <= 0.0 {
        return Err(invalid_input(format!(
            "weighted least-squares cell {cell} moment has no positive finite diagonal scale"
        )));
    }
    for row in matrix.iter_mut().take(dimension) {
        for value in row.iter_mut().take(dimension) {
            *value /= diagonal_scale;
            if !value.is_finite() {
                return Err(invalid_input(format!(
                    "weighted least-squares cell {cell} normalized moment is non-finite"
                )));
            }
        }
    }

    let operation_count = observations
        .checked_mul(12)
        .and_then(|count| count.checked_add(64))
        .ok_or(crate::MeshError::OutOfMemory)?;
    let roundoff = (operation_count as f64) * f64::EPSILON;
    if !roundoff.is_finite() || roundoff >= 0.25 {
        return Err(invalid_input(format!(
            "weighted least-squares cell {cell} stencil is too large for certified factorization"
        )));
    }
    let gamma = roundoff / (1.0 - roundoff);
    let inactive = 8.0 * gamma;
    let active = 32.0 * gamma;
    wls_pivoted_ldlt(matrix, dimension, inactive, active, cell)
        .map(|factor| (factor, diagonal_scale))
}

#[allow(clippy::needless_range_loop)]
fn wls_pivoted_ldlt(
    mut matrix: [[f64; 3]; 3],
    dimension: usize,
    inactive: f64,
    active: f64,
    cell: usize,
) -> Result<WlsPivotedLdlt> {
    let mut permutation = [0usize, 1, 2];
    let mut lower = [[0.0; 3]; 3];
    let mut diagonal = [0.0; 3];
    for (index, row) in lower.iter_mut().enumerate() {
        row[index] = 1.0;
    }

    for pivot_index in 0..dimension {
        let mut selected = pivot_index;
        for candidate in (pivot_index + 1)..dimension {
            let candidate_value = matrix[candidate][candidate];
            let selected_value = matrix[selected][selected];
            if candidate_value > selected_value
                || (candidate_value == selected_value
                    && permutation[candidate] < permutation[selected])
            {
                selected = candidate;
            }
        }
        if selected != pivot_index {
            matrix.swap(selected, pivot_index);
            for row in matrix.iter_mut() {
                row.swap(selected, pivot_index);
            }
            permutation.swap(selected, pivot_index);
            for column in 0..pivot_index {
                let temporary = lower[selected][column];
                lower[selected][column] = lower[pivot_index][column];
                lower[pivot_index][column] = temporary;
            }
        }

        let pivot = matrix[pivot_index][pivot_index];
        if !pivot.is_finite() {
            return Err(invalid_input(format!(
                "weighted least-squares cell {cell} pivot {pivot_index} is non-finite"
            )));
        }
        if pivot < -inactive {
            return Err(invalid_input(format!(
                "weighted least-squares cell {cell} moment is not positive semidefinite at pivot {pivot_index}"
            )));
        }
        if pivot <= inactive {
            return Err(invalid_input(format!(
                "weighted least-squares cell {cell} is locally rank deficient at pivot {pivot_index}"
            )));
        }
        if pivot < active {
            return Err(invalid_input(format!(
                "weighted least-squares cell {cell} local rank is numerically ambiguous at pivot {pivot_index}"
            )));
        }
        diagonal[pivot_index] = pivot;

        for row in (pivot_index + 1)..dimension {
            let multiplier = matrix[row][pivot_index] / pivot;
            if !multiplier.is_finite() {
                return Err(invalid_input(format!(
                    "weighted least-squares cell {cell} factor multiplier is non-finite"
                )));
            }
            lower[row][pivot_index] = multiplier;
        }
        for row in (pivot_index + 1)..dimension {
            for column in row..dimension {
                let reduction = lower[row][pivot_index] * pivot * lower[column][pivot_index];
                let next = matrix[column][row] - reduction;
                if !next.is_finite() {
                    return Err(invalid_input(format!(
                        "weighted least-squares cell {cell} Schur complement is non-finite"
                    )));
                }
                matrix[column][row] = next;
                matrix[row][column] = next;
            }
        }
    }

    Ok(WlsPivotedLdlt {
        dimension,
        permutation,
        lower,
        diagonal,
    })
}

#[allow(clippy::too_many_arguments)]
fn wls_reconstruction_vector(
    displacement: Point3,
    weight: f64,
    distance_scale: f64,
    weight_scale: f64,
    diagonal_scale: f64,
    basis: WlsIntrinsicBasis,
    factor: WlsPivotedLdlt,
    cell: usize,
    face: usize,
) -> Result<Point3> {
    if weight == 0.0 {
        return Ok(zero());
    }
    let scaled = wls_scale_point(displacement, 1.0 / distance_scale, || {
        format!("weighted least-squares cell {cell} face {face} coefficient displacement")
    })?;
    let mut right_hand_side = wls_project_onto_basis(scaled, basis, face)?;
    let scaled_weight = weight / weight_scale;
    for value in right_hand_side.iter_mut().take(basis.dimension.count()) {
        *value *= scaled_weight;
        if !value.is_finite() {
            return Err(invalid_input(format!(
                "weighted least-squares cell {cell} face {face} coefficient right-hand side is non-finite"
            )));
        }
    }
    let mut reduced = factor.solve(right_hand_side, cell)?;
    for value in reduced.iter_mut().take(basis.dimension.count()) {
        *value /= diagonal_scale;
        *value /= distance_scale;
        if !value.is_finite() {
            return Err(invalid_input(format!(
                "weighted least-squares cell {cell} face {face} coefficient is non-finite"
            )));
        }
    }
    wls_lift_from_basis(reduced, basis, cell, face)
}

impl WlsPivotedLdlt {
    #[allow(clippy::needless_range_loop)]
    fn solve(self, right_hand_side: [f64; 3], cell: usize) -> Result<[f64; 3]> {
        let mut permuted = [0.0; 3];
        for (index, value) in permuted.iter_mut().enumerate().take(self.dimension) {
            *value = right_hand_side[self.permutation[index]];
        }
        for row in 0..self.dimension {
            let mut value = permuted[row];
            for column in 0..row {
                value -= self.lower[row][column] * permuted[column];
            }
            if !value.is_finite() {
                return Err(invalid_input(format!(
                    "weighted least-squares cell {cell} forward solve is non-finite"
                )));
            }
            permuted[row] = value;
        }
        for (index, value) in permuted.iter_mut().enumerate().take(self.dimension) {
            *value /= self.diagonal[index];
            if !value.is_finite() {
                return Err(invalid_input(format!(
                    "weighted least-squares cell {cell} diagonal solve is non-finite"
                )));
            }
        }
        for row in (0..self.dimension).rev() {
            let mut value = permuted[row];
            for (column, later) in permuted
                .iter()
                .enumerate()
                .take(self.dimension)
                .skip(row + 1)
            {
                value -= self.lower[column][row] * later;
            }
            if !value.is_finite() {
                return Err(invalid_input(format!(
                    "weighted least-squares cell {cell} backward solve is non-finite"
                )));
            }
            permuted[row] = value;
        }
        let mut solution = [0.0; 3];
        for (index, value) in permuted.iter().copied().enumerate().take(self.dimension) {
            solution[self.permutation[index]] = value;
        }
        Ok(solution)
    }
}

#[cfg_attr(not(test), allow(dead_code))]
pub(super) fn weighted_least_squares_scalar_gradient_from_deltas(
    geometry: &WeightedLeastSquaresGeometry,
    cell_values: &[f64],
    boundary_deltas: &[Option<f64>],
) -> Result<Vec<Point3>> {
    wls_validate_application_lengths(geometry, cell_values.len(), boundary_deltas.len())?;
    for (cell, value) in cell_values.iter().copied().enumerate() {
        if !value.is_finite() {
            return Err(invalid_input(format!(
                "weighted least-squares scalar cell {cell} value must be finite, got {value}"
            )));
        }
    }
    wls_validate_scalar_boundary_deltas(&geometry.face_coefficients, boundary_deltas)?;

    let mut gradient = wls_try_filled_vec(geometry.cells, zero())?;
    for (face, coefficient) in geometry.face_coefficients.iter().copied().enumerate() {
        if coefficient.role == WlsFaceRole::EmptyBoundary {
            continue;
        }
        let owner = coefficient.owner_cell;
        let delta = if let Some(neighbour) = coefficient.neighbour_cell {
            wls_checked_subtraction(cell_values[neighbour], cell_values[owner], || {
                format!("weighted least-squares scalar internal face {face} delta")
            })?
        } else {
            boundary_deltas[face].ok_or_else(|| {
                invalid_input(format!(
                    "weighted least-squares boundary face {face} requires a scalar delta"
                ))
            })?
        };
        if delta == 0.0 {
            continue;
        }
        wls_add_scaled_point(&mut gradient[owner], coefficient.owner, delta, owner, face)?;
        if let Some(neighbour) = coefficient.neighbour_cell {
            wls_add_scaled_point(
                &mut gradient[neighbour],
                coefficient.neighbour,
                delta,
                neighbour,
                face,
            )?;
        }
    }
    Ok(gradient)
}

#[cfg_attr(not(test), allow(dead_code))]
pub(super) fn weighted_least_squares_vector_component_gradients_from_deltas(
    geometry: &WeightedLeastSquaresGeometry,
    cell_values: &[Point3],
    boundary_deltas: &[Option<Point3>],
) -> Result<[Vec<Point3>; 3]> {
    wls_validate_application_lengths(geometry, cell_values.len(), boundary_deltas.len())?;
    for (cell, value) in cell_values.iter().copied().enumerate() {
        wls_require_finite_point(value, || {
            format!("weighted least-squares vector cell {cell} value")
        })?;
    }
    let coefficients = geometry
        .wedge_vector_face_coefficients
        .as_deref()
        .unwrap_or(&geometry.face_coefficients);
    wls_validate_vector_boundary_deltas(coefficients, boundary_deltas)?;

    let mut gradients = [
        wls_try_filled_vec(geometry.cells, zero())?,
        wls_try_filled_vec(geometry.cells, zero())?,
        wls_try_filled_vec(geometry.cells, zero())?,
    ];
    for (face, coefficient) in coefficients.iter().copied().enumerate() {
        if coefficient.role == WlsFaceRole::EmptyBoundary {
            continue;
        }
        let owner = coefficient.owner_cell;
        let delta = if let Some(neighbour) = coefficient.neighbour_cell {
            Point3 {
                x: wls_checked_subtraction(cell_values[neighbour].x, cell_values[owner].x, || {
                    format!("weighted least-squares vector internal face {face} x delta")
                })?,
                y: wls_checked_subtraction(cell_values[neighbour].y, cell_values[owner].y, || {
                    format!("weighted least-squares vector internal face {face} y delta")
                })?,
                z: wls_checked_subtraction(cell_values[neighbour].z, cell_values[owner].z, || {
                    format!("weighted least-squares vector internal face {face} z delta")
                })?,
            }
        } else {
            boundary_deltas[face].ok_or_else(|| {
                invalid_input(format!(
                    "weighted least-squares boundary face {face} requires a vector delta"
                ))
            })?
        };
        let deltas = [delta.x, delta.y, delta.z];
        for (component, value) in deltas.into_iter().enumerate() {
            if value == 0.0 {
                continue;
            }
            wls_add_scaled_point(
                &mut gradients[component][owner],
                coefficient.owner,
                value,
                owner,
                face,
            )?;
            if let Some(neighbour) = coefficient.neighbour_cell {
                wls_add_scaled_point(
                    &mut gradients[component][neighbour],
                    coefficient.neighbour,
                    value,
                    neighbour,
                    face,
                )?;
            }
        }
    }
    Ok(gradients)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn scalar_gradient_with_scheme_and_addressing(
    mesh: &SolverRuntimeMeshData,
    scalar_geometry: &ScalarGradientGeometry,
    weighted_least_squares: Option<&WeightedLeastSquaresGeometry>,
    face_addressing: &CompactSimpleFaceAddressing,
    values: &[f64],
    boundary: &[ScalarFaceTreatment],
    scheme: LaminarSimpleGradientScheme,
) -> Result<Vec<Point3>> {
    match scheme {
        LaminarSimpleGradientScheme::LeastSquares => {
            let geometry = weighted_least_squares.ok_or_else(|| {
                invalid_input(
                    "leastSquares gradient was selected without cached geometry".to_string(),
                )
            })?;
            let boundary_deltas = wls_scalar_boundary_deltas(
                mesh,
                scalar_geometry,
                geometry,
                face_addressing,
                values,
                boundary,
            )?;
            weighted_least_squares_scalar_gradient_from_deltas(geometry, values, &boundary_deltas)
        }
        LaminarSimpleGradientScheme::GaussLinear
        | LaminarSimpleGradientScheme::CellLimitedGaussLinear(_) => {
            scalar_gradient_with_geometry_and_addressing(
                mesh,
                scalar_geometry,
                face_addressing,
                values,
                boundary,
                scheme,
            )
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn vector_component_gradients_with_scheme_and_addressing(
    mesh: &SolverRuntimeMeshData,
    scalar_geometry: &ScalarGradientGeometry,
    weighted_least_squares: Option<&WeightedLeastSquaresGeometry>,
    face_addressing: &CompactSimpleFaceAddressing,
    values: &[Point3],
    boundary: &[VectorFaceTreatment],
    flux: &[f64],
    scheme: LaminarSimpleGradientScheme,
) -> Result<[Vec<Point3>; 3]> {
    match scheme {
        LaminarSimpleGradientScheme::LeastSquares => {
            let geometry = weighted_least_squares.ok_or_else(|| {
                invalid_input(
                    "leastSquares vector gradient was selected without cached geometry".to_string(),
                )
            })?;
            let boundary_deltas =
                wls_vector_boundary_deltas(mesh, geometry, values, boundary, flux)?;
            weighted_least_squares_vector_component_gradients_from_deltas(
                geometry,
                values,
                &boundary_deltas,
            )
        }
        LaminarSimpleGradientScheme::GaussLinear
        | LaminarSimpleGradientScheme::CellLimitedGaussLinear(_) => {
            vector_component_gradients_with_addressing(
                mesh,
                scalar_geometry,
                face_addressing,
                values,
                boundary,
                scheme,
            )
        }
    }
}

fn wls_scalar_boundary_deltas(
    mesh: &SolverRuntimeMeshData,
    scalar_geometry: &ScalarGradientGeometry,
    geometry: &WeightedLeastSquaresGeometry,
    face_addressing: &CompactSimpleFaceAddressing,
    values: &[f64],
    boundary: &[ScalarFaceTreatment],
) -> Result<Vec<Option<f64>>> {
    if values.len() != mesh.cells
        || boundary.len() != mesh.faces
        || scalar_geometry.boundary_normal_distances.len() != mesh.faces
        || geometry.face_coefficients.len() != mesh.faces
        || face_addressing.faces() != mesh.faces
    {
        return Err(invalid_input(
            "leastSquares scalar boundary resolution does not match the runtime mesh".to_string(),
        ));
    }
    let mut deltas = wls_try_filled_vec(mesh.faces, None)?;
    for (face, coefficient) in geometry.face_coefficients.iter().copied().enumerate() {
        if coefficient.neighbour_cell.is_some() || coefficient.role == WlsFaceRole::EmptyBoundary {
            continue;
        }
        let owner = coefficient.owner_cell;
        let delta = match coefficient.role {
            WlsFaceRole::WedgeBoundary | WlsFaceRole::SymmetryBoundary => {
                if !matches!(boundary[face], ScalarFaceTreatment::Constraint) {
                    return Err(invalid_input(format!(
                        "leastSquares scalar constraint face {face} requires a constraint boundary treatment"
                    )));
                }
                0.0
            }
            WlsFaceRole::RegularBoundary => match boundary[face] {
                ScalarFaceTreatment::FixedValue(value) => {
                    wls_checked_subtraction(value, values[owner], || {
                        format!("leastSquares scalar fixed-value face {face} delta")
                    })?
                }
                ScalarFaceTreatment::FixedGradient(gradient) => {
                    let distance = scalar_geometry.boundary_normal_distances[face].ok_or_else(
                        || {
                            invalid_input(format!(
                                "leastSquares scalar fixed-gradient face {face} has no projected distance"
                            ))
                        },
                    )?;
                    let delta = gradient * distance;
                    if !gradient.is_finite()
                        || !distance.is_finite()
                        || distance <= 0.0
                        || !delta.is_finite()
                    {
                        return Err(invalid_input(format!(
                            "leastSquares scalar fixed-gradient face {face} delta is invalid"
                        )));
                    }
                    delta
                }
                ScalarFaceTreatment::ZeroGradient => 0.0,
                ScalarFaceTreatment::InletOutlet(_) => {
                    return Err(invalid_input(format!(
                        "leastSquares scalar inletOutlet face {face} must be resolved against the current flux"
                    )));
                }
                ScalarFaceTreatment::Constraint => {
                    return Err(invalid_input(format!(
                        "leastSquares regular scalar face {face} cannot use a constraint treatment"
                    )));
                }
            },
            WlsFaceRole::Internal | WlsFaceRole::EmptyBoundary => {
                unreachable!("internal and empty faces were handled before scalar delta resolution")
            }
        };
        if !delta.is_finite() {
            return Err(invalid_input(format!(
                "leastSquares scalar boundary face {face} delta must be finite"
            )));
        }
        deltas[face] = Some(if delta == 0.0 { 0.0 } else { delta });
    }
    Ok(deltas)
}

fn wls_vector_boundary_deltas(
    mesh: &SolverRuntimeMeshData,
    geometry: &WeightedLeastSquaresGeometry,
    values: &[Point3],
    boundary: &[VectorFaceTreatment],
    flux: &[f64],
) -> Result<Vec<Option<Point3>>> {
    if values.len() != mesh.cells
        || boundary.len() != mesh.faces
        || flux.len() != mesh.faces
        || geometry.face_coefficients.len() != mesh.faces
        || geometry.vector_constraints.len() != mesh.faces
    {
        return Err(invalid_input(
            "leastSquares vector boundary resolution does not match the runtime mesh".to_string(),
        ));
    }
    let mut deltas = wls_try_filled_vec(mesh.faces, None)?;
    for (face, coefficient) in geometry.face_coefficients.iter().copied().enumerate() {
        if coefficient.neighbour_cell.is_some() || coefficient.role == WlsFaceRole::EmptyBoundary {
            continue;
        }
        let owner = coefficient.owner_cell;
        let owner_value = values[owner];
        wls_require_finite_point(owner_value, || {
            format!("leastSquares vector owner value for face {face}")
        })?;
        let delta = match coefficient.role {
            WlsFaceRole::RegularBoundary => match boundary[face] {
                VectorFaceTreatment::FixedValue(value) => {
                    wls_checked_delta(value, owner_value, || {
                        format!("leastSquares vector fixed-value face {face} delta")
                    })?
                }
                VectorFaceTreatment::InletOutlet(value)
                | VectorFaceTreatment::PressureInletOutletVelocity(value) => {
                    let decision_flux = flux[face];
                    if !decision_flux.is_finite() {
                        return Err(invalid_input(format!(
                            "leastSquares vector inletOutlet face {face} has non-finite decision flux"
                        )));
                    }
                    if decision_flux < 0.0 {
                        wls_checked_delta(value, owner_value, || {
                            format!("leastSquares vector inletOutlet face {face} delta")
                        })?
                    } else {
                        zero()
                    }
                }
                VectorFaceTreatment::ZeroGradient => zero(),
                VectorFaceTreatment::Constraint => {
                    return Err(invalid_input(format!(
                        "leastSquares regular vector face {face} cannot use a constraint treatment"
                    )));
                }
            },
            WlsFaceRole::SymmetryBoundary => {
                if !matches!(boundary[face], VectorFaceTreatment::Constraint) {
                    return Err(invalid_input(format!(
                        "leastSquares symmetry face {face} requires a constraint boundary treatment"
                    )));
                }
                let WlsVectorConstraint::Symmetry { unit_normal } =
                    geometry.vector_constraints[face].ok_or_else(|| {
                        invalid_input(format!(
                            "leastSquares symmetry face {face} has no cached normal"
                        ))
                    })?
                else {
                    return Err(invalid_input(format!(
                        "leastSquares symmetry face {face} has mismatched constraint metadata"
                    )));
                };
                let normal_component = wls_checked_dot(unit_normal, owner_value, || {
                    format!("leastSquares symmetry face {face} normal component")
                })?;
                wls_scale_point(unit_normal, -normal_component, || {
                    format!("leastSquares symmetry face {face} vector delta")
                })?
            }
            WlsFaceRole::WedgeBoundary => {
                if !matches!(boundary[face], VectorFaceTreatment::Constraint) {
                    return Err(invalid_input(format!(
                        "leastSquares wedge face {face} requires a constraint boundary treatment"
                    )));
                }
                let WlsVectorConstraint::Wedge { rotation } = geometry.vector_constraints[face]
                    .ok_or_else(|| {
                        invalid_input(format!(
                            "leastSquares wedge face {face} has no cached transform"
                        ))
                    })?
                else {
                    return Err(invalid_input(format!(
                        "leastSquares wedge face {face} has mismatched constraint metadata"
                    )));
                };
                rotation.delta(owner_value, face)?
            }
            WlsFaceRole::Internal | WlsFaceRole::EmptyBoundary => {
                unreachable!("internal and empty faces were handled before vector delta resolution")
            }
        };
        deltas[face] = Some(wls_require_finite_point(delta, || {
            format!("leastSquares vector boundary face {face} delta")
        })?);
    }
    Ok(deltas)
}

fn wls_validate_application_lengths(
    geometry: &WeightedLeastSquaresGeometry,
    cell_values: usize,
    boundary_deltas: usize,
) -> Result<()> {
    if cell_values != geometry.cells {
        return Err(invalid_input(format!(
            "weighted least-squares application expected {} cell values, got {cell_values}",
            geometry.cells
        )));
    }
    if boundary_deltas != geometry.face_coefficients.len() {
        return Err(invalid_input(format!(
            "weighted least-squares application expected {} boundary-delta slots, got {boundary_deltas}",
            geometry.face_coefficients.len()
        )));
    }
    Ok(())
}

fn wls_validate_scalar_boundary_deltas(
    coefficients: &[WlsFaceCoefficient],
    boundary_deltas: &[Option<f64>],
) -> Result<()> {
    for (face, coefficient) in coefficients.iter().enumerate() {
        if coefficient.neighbour_cell.is_some() || coefficient.role == WlsFaceRole::EmptyBoundary {
            if boundary_deltas[face].is_some() {
                return Err(invalid_input(format!(
                    "weighted least-squares face {face} requires an empty scalar boundary-delta slot"
                )));
            }
            continue;
        }
        let delta = boundary_deltas[face].ok_or_else(|| {
            invalid_input(format!(
                "weighted least-squares boundary face {face} requires a scalar delta"
            ))
        })?;
        if !delta.is_finite() {
            return Err(invalid_input(format!(
                "weighted least-squares boundary face {face} scalar delta must be finite, got {delta}"
            )));
        }
        if matches!(
            coefficient.role,
            WlsFaceRole::WedgeBoundary | WlsFaceRole::SymmetryBoundary
        ) && delta != 0.0
        {
            return Err(invalid_input(format!(
                "weighted least-squares constraint face {face} scalar delta must be exactly zero, got {delta}"
            )));
        }
    }
    Ok(())
}

fn wls_validate_vector_boundary_deltas(
    coefficients: &[WlsFaceCoefficient],
    boundary_deltas: &[Option<Point3>],
) -> Result<()> {
    for (face, coefficient) in coefficients.iter().enumerate() {
        if coefficient.neighbour_cell.is_some() || coefficient.role == WlsFaceRole::EmptyBoundary {
            if boundary_deltas[face].is_some() {
                return Err(invalid_input(format!(
                    "weighted least-squares face {face} requires an empty vector boundary-delta slot"
                )));
            }
            continue;
        }
        let delta = boundary_deltas[face].ok_or_else(|| {
            invalid_input(format!(
                "weighted least-squares boundary face {face} requires a vector delta"
            ))
        })?;
        wls_require_finite_point(delta, || {
            format!("weighted least-squares boundary face {face} vector delta")
        })?;
    }
    Ok(())
}

fn wls_face_displacement(
    mesh: &SolverRuntimeMeshData,
    face_addressing: &CompactSimpleFaceAddressing,
    face: usize,
    role: WlsFaceRole,
) -> Result<Point3> {
    let owner = face_addressing.owner(face);
    if role == WlsFaceRole::Internal {
        let neighbour = face_addressing.neighbour(face).ok_or_else(|| {
            invalid_input(format!(
                "weighted least-squares internal face {face} has no neighbour"
            ))
        })?;
        return wls_checked_delta(
            mesh.cell_centres[neighbour],
            mesh.cell_centres[owner],
            || format!("weighted least-squares internal face {face} centre displacement"),
        );
    }

    let area = mesh.face_area_vectors[face];
    let area_magnitude = wls_checked_magnitude(area, || {
        format!("weighted least-squares boundary face {face} area magnitude")
    })?;
    if area_magnitude <= 0.0 {
        return Err(invalid_input(format!(
            "weighted least-squares boundary face {face} area magnitude must be positive"
        )));
    }
    let normal = wls_scale_point(area, 1.0 / area_magnitude, || {
        format!("weighted least-squares boundary face {face} unit normal")
    })?;
    let raw = wls_checked_delta(mesh.face_centres[face], mesh.cell_centres[owner], || {
        format!("weighted least-squares boundary face {face} centre displacement")
    })?;
    let projected = wls_checked_dot(normal, raw, || {
        format!("weighted least-squares boundary face {face} normal projection")
    })?;
    if projected <= 0.0 {
        return Err(invalid_input(format!(
            "weighted least-squares boundary face {face} has non-positive normal displacement"
        )));
    }
    wls_scale_point(normal, projected, || {
        format!("weighted least-squares boundary face {face} projected displacement")
    })
}

fn wls_direction_rank(
    candidates: &mut [WlsDirectionCandidate],
    context: &'static str,
) -> Result<usize> {
    wls_sort_direction_candidates(candidates);
    let mut axes = [zero(); 3];
    let mut rank = 0usize;
    while rank < 3 {
        let Some((candidate_index, residual, residual_squared)) =
            wls_best_residual(candidates, &axes, rank)?
        else {
            break;
        };
        if residual_squared <= WLS_BASIS_INACTIVE_SQUARED {
            break;
        }
        if residual_squared < WLS_BASIS_ACTIVE_SQUARED {
            return Err(invalid_input(format!(
                "weighted least-squares {context} rank is numerically ambiguous at axis {rank}"
            )));
        }
        let inverse_norm = 1.0 / residual_squared.sqrt();
        axes[rank] = wls_canonicalize_unit(wls_scale_point(residual, inverse_norm, || {
            format!("weighted least-squares {context} axis {rank}")
        })?);
        candidates[candidate_index].unit = zero();
        rank += 1;
    }
    Ok(rank)
}

fn wls_build_expected_basis(
    candidates: &mut [WlsDirectionCandidate],
    expected: usize,
    context: &'static str,
) -> Result<[Point3; 3]> {
    wls_sort_direction_candidates(candidates);
    let mut axes = [zero(); 3];
    for axis in 0..expected {
        let Some((candidate_index, residual, residual_squared)) =
            wls_best_residual(candidates, &axes, axis)?
        else {
            return Err(invalid_input(format!(
                "weighted least-squares {context} are rank deficient: expected {expected}, certified {axis}"
            )));
        };
        if residual_squared <= WLS_BASIS_INACTIVE_SQUARED {
            return Err(invalid_input(format!(
                "weighted least-squares {context} are rank deficient: expected {expected}, certified {axis}"
            )));
        }
        if residual_squared < WLS_BASIS_ACTIVE_SQUARED {
            return Err(invalid_input(format!(
                "weighted least-squares {context} rank is numerically ambiguous at axis {axis}"
            )));
        }
        axes[axis] = wls_canonicalize_unit(wls_scale_point(
            residual,
            1.0 / residual_squared.sqrt(),
            || format!("weighted least-squares {context} axis {axis}"),
        )?);
        candidates[candidate_index].unit = zero();
    }

    if let Some((_, _, residual_squared)) = wls_best_residual(candidates, &axes, expected)? {
        if residual_squared >= WLS_BASIS_ACTIVE_SQUARED {
            return Err(invalid_input(format!(
                "weighted least-squares {context} exceed topology dimension {expected}"
            )));
        }
        if residual_squared > WLS_BASIS_INACTIVE_SQUARED {
            return Err(invalid_input(format!(
                "weighted least-squares {context} topology rank is numerically ambiguous"
            )));
        }
    }
    Ok(axes)
}

fn wls_best_residual(
    candidates: &[WlsDirectionCandidate],
    axes: &[Point3; 3],
    axis_count: usize,
) -> Result<Option<(usize, Point3, f64)>> {
    let mut best = None;
    for (index, candidate) in candidates.iter().copied().enumerate() {
        if candidate.unit.x == 0.0 && candidate.unit.y == 0.0 && candidate.unit.z == 0.0 {
            continue;
        }
        let residual = wls_reorthogonalized_residual(candidate.unit, axes, axis_count)?;
        let residual_squared = wls_checked_dot(residual, residual, || {
            format!(
                "weighted least-squares face {} basis residual square",
                candidate.source_face
            )
        })?;
        match best {
            None => best = Some((index, residual, residual_squared)),
            Some((_, _, best_squared)) if residual_squared > best_squared => {
                best = Some((index, residual, residual_squared));
            }
            _ => {}
        }
    }
    Ok(best)
}

fn wls_reorthogonalized_residual(
    mut residual: Point3,
    axes: &[Point3; 3],
    axis_count: usize,
) -> Result<Point3> {
    for _ in 0..2 {
        for axis in axes.iter().copied().take(axis_count) {
            let projection = wls_checked_dot(residual, axis, || {
                "weighted least-squares basis projection".to_string()
            })?;
            residual = wls_checked_delta(
                residual,
                wls_scale_point(axis, projection, || {
                    "weighted least-squares basis projection vector".to_string()
                })?,
                || "weighted least-squares basis residual".to_string(),
            )?;
        }
    }
    Ok(residual)
}

fn wls_sort_direction_candidates(candidates: &mut [WlsDirectionCandidate]) {
    candidates.sort_unstable_by(|left, right| {
        left.unit
            .x
            .total_cmp(&right.unit.x)
            .then_with(|| left.unit.y.total_cmp(&right.unit.y))
            .then_with(|| left.unit.z.total_cmp(&right.unit.z))
            .then_with(|| left.source_face.cmp(&right.source_face))
    });
}

fn wls_normalize_direction(
    value: Point3,
    source_face: usize,
    context: &'static str,
) -> Result<Point3> {
    wls_require_finite_point(value, || {
        format!("weighted least-squares face {source_face} {context}")
    })?;
    let scale = value.x.abs().max(value.y.abs()).max(value.z.abs());
    if scale == 0.0 {
        return Err(invalid_input(format!(
            "weighted least-squares face {source_face} {context} must be nonzero"
        )));
    }
    let scaled = Point3 {
        x: value.x / scale,
        y: value.y / scale,
        z: value.z / scale,
    };
    let norm = scaled.x.hypot(scaled.y).hypot(scaled.z);
    if !norm.is_finite() || norm <= 0.0 {
        return Err(invalid_input(format!(
            "weighted least-squares face {source_face} {context} norm must be positive and finite"
        )));
    }
    Ok(wls_canonicalize_unit(Point3 {
        x: scaled.x / norm,
        y: scaled.y / norm,
        z: scaled.z / norm,
    }))
}

fn wls_canonicalize_unit(mut value: Point3) -> Point3 {
    let magnitudes = [value.x.abs(), value.y.abs(), value.z.abs()];
    let mut largest = 0usize;
    for index in 1..3 {
        if magnitudes[index] > magnitudes[largest] {
            largest = index;
        }
    }
    let selected = match largest {
        0 => value.x,
        1 => value.y,
        _ => value.z,
    };
    if selected.is_sign_negative() {
        value.x = -value.x;
        value.y = -value.y;
        value.z = -value.z;
    }
    if value.x == 0.0 {
        value.x = 0.0;
    }
    if value.y == 0.0 {
        value.y = 0.0;
    }
    if value.z == 0.0 {
        value.z = 0.0;
    }
    value
}

fn wls_require_in_active_subspace(
    displacement: Point3,
    basis: WlsIntrinsicBasis,
    face: usize,
) -> Result<()> {
    let norm_squared = wls_checked_dot(displacement, displacement, || {
        format!("weighted least-squares face {face} displacement norm square")
    })?;
    let projected = wls_project_onto_basis(displacement, basis, face)?;
    let lifted = wls_lift_from_basis(projected, basis, usize::MAX, face)?;
    let inactive = wls_checked_delta(displacement, lifted, || {
        format!("weighted least-squares face {face} inactive displacement")
    })?;
    let inactive_squared = wls_checked_dot(inactive, inactive, || {
        format!("weighted least-squares face {face} inactive displacement square")
    })? / norm_squared;
    if inactive_squared >= WLS_BASIS_ACTIVE_SQUARED {
        return Err(invalid_input(format!(
            "weighted least-squares face {face} exceeds the declared intrinsic dimension"
        )));
    }
    if inactive_squared > WLS_BASIS_INACTIVE_SQUARED {
        return Err(invalid_input(format!(
            "weighted least-squares face {face} intrinsic projection is numerically ambiguous"
        )));
    }
    Ok(())
}

fn wls_project_onto_basis(
    value: Point3,
    basis: WlsIntrinsicBasis,
    face: usize,
) -> Result<[f64; 3]> {
    let mut projected = [0.0; 3];
    for (axis, output) in basis
        .axes
        .iter()
        .copied()
        .zip(projected.iter_mut())
        .take(basis.dimension.count())
    {
        *output = wls_checked_dot(value, axis, || {
            format!("weighted least-squares face {face} intrinsic projection")
        })?;
    }
    Ok(projected)
}

fn wls_lift_from_basis(
    value: [f64; 3],
    basis: WlsIntrinsicBasis,
    cell: usize,
    face: usize,
) -> Result<Point3> {
    let mut lifted = zero();
    for (axis, coefficient) in basis
        .axes
        .iter()
        .copied()
        .zip(value)
        .take(basis.dimension.count())
    {
        wls_add_scaled_point(&mut lifted, axis, coefficient, cell, face)?;
    }
    Ok(lifted)
}

fn wls_add_scaled_point(
    target: &mut Point3,
    vector: Point3,
    scale: f64,
    cell: usize,
    face: usize,
) -> Result<()> {
    let next_x = target.x + vector.x * scale;
    let next_y = target.y + vector.y * scale;
    let next_z = target.z + vector.z * scale;
    if !next_x.is_finite() || !next_y.is_finite() || !next_z.is_finite() {
        return Err(invalid_input(format!(
            "weighted least-squares cell {cell} face {face} accumulation is non-finite"
        )));
    }
    target.x = next_x;
    target.y = next_y;
    target.z = next_z;
    Ok(())
}

fn wls_symmetric_slot(row: usize, column: usize) -> usize {
    match (row.min(column), row.max(column)) {
        (0, 0) => 0,
        (0, 1) => 1,
        (0, 2) => 2,
        (1, 1) => 3,
        (1, 2) => 4,
        (2, 2) => 5,
        _ => unreachable!("weighted least-squares symmetric index is outside 3x3"),
    }
}

fn wls_checked_delta(
    left: Point3,
    right: Point3,
    context: impl FnOnce() -> String,
) -> Result<Point3> {
    let value = Point3 {
        x: left.x - right.x,
        y: left.y - right.y,
        z: left.z - right.z,
    };
    if !value.x.is_finite() || !value.y.is_finite() || !value.z.is_finite() {
        return Err(invalid_input(format!(
            "{} produced a non-finite component",
            context()
        )));
    }
    Ok(value)
}

fn wls_checked_add(
    left: Point3,
    right: Point3,
    context: impl FnOnce() -> String,
) -> Result<Point3> {
    let value = Point3 {
        x: left.x + right.x,
        y: left.y + right.y,
        z: left.z + right.z,
    };
    if !value.x.is_finite() || !value.y.is_finite() || !value.z.is_finite() {
        return Err(invalid_input(format!(
            "{} produced a non-finite component",
            context()
        )));
    }
    Ok(value)
}

fn wls_checked_subtraction(left: f64, right: f64, context: impl FnOnce() -> String) -> Result<f64> {
    let value = left - right;
    if !value.is_finite() {
        return Err(invalid_input(format!("{} is non-finite", context())));
    }
    Ok(value)
}

fn wls_checked_dot(left: Point3, right: Point3, context: impl FnOnce() -> String) -> Result<f64> {
    let x = left.x * right.x;
    let y = left.y * right.y;
    let z = left.z * right.z;
    let value = (x + y) + z;
    if !x.is_finite() || !y.is_finite() || !z.is_finite() || !value.is_finite() {
        return Err(invalid_input(format!("{} is non-finite", context())));
    }
    Ok(value)
}

fn wls_checked_cross(
    left: Point3,
    right: Point3,
    context: impl FnOnce() -> String,
) -> Result<Point3> {
    let value = Point3 {
        x: left.y * right.z - left.z * right.y,
        y: left.z * right.x - left.x * right.z,
        z: left.x * right.y - left.y * right.x,
    };
    if !value.x.is_finite() || !value.y.is_finite() || !value.z.is_finite() {
        return Err(invalid_input(format!("{} is non-finite", context())));
    }
    Ok(value)
}

fn wls_checked_magnitude(value: Point3, context: impl FnOnce() -> String) -> Result<f64> {
    let magnitude = value.x.hypot(value.y).hypot(value.z);
    if !magnitude.is_finite() {
        return Err(invalid_input(format!("{} is non-finite", context())));
    }
    Ok(magnitude)
}

fn wls_normalize_oriented(value: Point3, context: impl FnOnce() -> String) -> Result<Point3> {
    let context = context();
    wls_require_finite_point(value, || context.clone())?;
    let scale = value.x.abs().max(value.y.abs()).max(value.z.abs());
    if scale == 0.0 {
        return Err(invalid_input(format!("{context} must be nonzero")));
    }
    let scaled = Point3 {
        x: value.x / scale,
        y: value.y / scale,
        z: value.z / scale,
    };
    let magnitude = wls_checked_magnitude(scaled, || format!("{context} magnitude"))?;
    if magnitude == 0.0 {
        return Err(invalid_input(format!(
            "{context} magnitude must be positive"
        )));
    }
    let normalized = Point3 {
        x: scaled.x / magnitude,
        y: scaled.y / magnitude,
        z: scaled.z / magnitude,
    };
    wls_require_finite_point(normalized, || format!("{context} unit vector"))
}

#[derive(Clone, Copy)]
struct WlsCompensatedScalar {
    sum: f64,
    correction: f64,
}

impl WlsCompensatedScalar {
    const ZERO: Self = Self {
        sum: 0.0,
        correction: 0.0,
    };

    fn add(&mut self, value: f64, patch: usize, component: &str) -> Result<()> {
        let next = self.sum + value;
        if !value.is_finite() || !next.is_finite() {
            return Err(invalid_input(format!(
                "weighted least-squares constraint patch {patch} {component} accumulation is non-finite"
            )));
        }
        let compensation = if self.sum.abs() >= value.abs() {
            (self.sum - next) + value
        } else {
            (value - next) + self.sum
        };
        let corrected = self.correction + compensation;
        if !corrected.is_finite() {
            return Err(invalid_input(format!(
                "weighted least-squares constraint patch {patch} {component} compensation is non-finite"
            )));
        }
        self.sum = next;
        self.correction = corrected;
        Ok(())
    }

    fn value(self, patch: usize, component: &str) -> Result<f64> {
        let value = self.sum + self.correction;
        if !value.is_finite() {
            return Err(invalid_input(format!(
                "weighted least-squares constraint patch {patch} corrected {component} is non-finite"
            )));
        }
        Ok(value)
    }
}

fn wls_scale_point(value: Point3, scale: f64, context: impl FnOnce() -> String) -> Result<Point3> {
    let scaled = Point3 {
        x: value.x * scale,
        y: value.y * scale,
        z: value.z * scale,
    };
    if !scale.is_finite() || !scaled.x.is_finite() || !scaled.y.is_finite() || !scaled.z.is_finite()
    {
        return Err(invalid_input(format!("{} is non-finite", context())));
    }
    Ok(scaled)
}

fn wls_require_finite_point(value: Point3, context: impl FnOnce() -> String) -> Result<Point3> {
    if !value.x.is_finite() || !value.y.is_finite() || !value.z.is_finite() {
        return Err(invalid_input(format!(
            "{} must have finite components",
            context()
        )));
    }
    Ok(value)
}

fn wls_try_vec<T>(capacity: usize) -> Result<Vec<T>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(capacity)
        .map_err(|_| crate::MeshError::OutOfMemory)?;
    Ok(values)
}

fn wls_try_filled_vec<T: Clone>(length: usize, value: T) -> Result<Vec<T>> {
    let mut values = wls_try_vec(length)?;
    values.resize(length, value);
    Ok(values)
}

#[cfg(test)]
mod weighted_least_squares_tests {
    use crate::MeshError;
    use crate::runtime::{SolverRuntimeMeshData, SolverRuntimePatchRange};

    use super::*;

    fn point(x: f64, y: f64, z: f64) -> Point3 {
        Point3 { x, y, z }
    }

    fn patch(
        name: &str,
        patch_type: &str,
        start_face: usize,
        faces: usize,
    ) -> SolverRuntimePatchRange {
        SolverRuntimePatchRange {
            name: name.to_string(),
            patch_type: patch_type.to_string(),
            start_face,
            faces,
        }
    }

    fn runtime_mesh(
        cell_centres: Vec<Point3>,
        owner: Vec<usize>,
        neighbour: Vec<Option<usize>>,
        face_centres: Vec<Point3>,
        face_area_vectors: Vec<Point3>,
        patches: Vec<SolverRuntimePatchRange>,
    ) -> SolverRuntimeMeshData {
        let cells = cell_centres.len();
        let faces = owner.len();
        let internal_faces = neighbour.iter().filter(|entry| entry.is_some()).count();
        let boundary_faces = faces - internal_faces;
        SolverRuntimeMeshData {
            points: 0,
            cells,
            faces,
            internal_faces,
            boundary_faces,
            owner,
            neighbour,
            patches,
            face_centres,
            face_area_vectors,
            cell_centres,
            cell_volumes: vec![1.0; cells],
            min_face_area: 1.0,
            max_face_area: 1.0,
            min_cell_volume: 1.0,
            max_cell_volume: 1.0,
            total_cell_volume: cells as f64,
            non_positive_cell_volumes: 0,
        }
    }

    fn one_cell_box(dimension: WlsIntrinsicDimension) -> SolverRuntimeMeshData {
        let half = 0.5;
        let mut face_centres = vec![point(half, 0.0, 0.0), point(-half, 0.0, 0.0)];
        let mut face_areas = vec![point(1.0, 0.0, 0.0), point(-1.0, 0.0, 0.0)];
        let mut regular_faces = match dimension {
            WlsIntrinsicDimension::One => 2,
            WlsIntrinsicDimension::Two | WlsIntrinsicDimension::Three => {
                face_centres.extend([point(0.0, half, 0.0), point(0.0, -half, 0.0)]);
                face_areas.extend([point(0.0, 1.0, 0.0), point(0.0, -1.0, 0.0)]);
                4
            }
        };
        if dimension == WlsIntrinsicDimension::Three {
            face_centres.extend([point(0.0, 0.0, half), point(0.0, 0.0, -half)]);
            face_areas.extend([point(0.0, 0.0, 1.0), point(0.0, 0.0, -1.0)]);
            regular_faces = 6;
        } else if dimension == WlsIntrinsicDimension::Two {
            face_centres.extend([point(0.0, 0.0, half), point(0.0, 0.0, -half)]);
            face_areas.extend([point(0.0, 0.0, 1.0), point(0.0, 0.0, -1.0)]);
        } else {
            face_centres.extend([
                point(0.0, half, 0.0),
                point(0.0, -half, 0.0),
                point(0.0, 0.0, half),
                point(0.0, 0.0, -half),
            ]);
            face_areas.extend([
                point(0.0, 1.0, 0.0),
                point(0.0, -1.0, 0.0),
                point(0.0, 0.0, 1.0),
                point(0.0, 0.0, -1.0),
            ]);
        }
        let faces = face_centres.len();
        let mut patches = vec![patch("physical", "patch", 0, regular_faces)];
        if regular_faces != faces {
            patches.push(patch(
                "frontAndBack",
                "empty",
                regular_faces,
                faces - regular_faces,
            ));
        }
        runtime_mesh(
            vec![zero()],
            vec![0; faces],
            vec![None; faces],
            face_centres,
            face_areas,
            patches,
        )
    }

    fn one_cell_wedge(angle_radians: f64) -> SolverRuntimeMeshData {
        let (sine, cosine) = angle_radians.sin_cos();
        runtime_mesh(
            vec![zero()],
            vec![0; 6],
            vec![None; 6],
            vec![
                point(0.0, 0.5, 0.0),
                point(0.0, -0.5, 0.0),
                point(0.0, 0.0, 0.5),
                point(0.0, 0.0, -0.5),
                point(0.5 * cosine, 0.5 * sine, 0.0),
                point(-0.5 * cosine, 0.5 * sine, 0.0),
            ],
            vec![
                point(0.0, 1.0, 0.0),
                point(0.0, -1.0, 0.0),
                point(0.0, 0.0, 1.0),
                point(0.0, 0.0, -1.0),
                point(cosine, sine, 0.0),
                point(-cosine, sine, 0.0),
            ],
            vec![
                patch("physical", "patch", 0, 4),
                patch("wedgePlus", "wedge", 4, 1),
                patch("wedgeMinus", "wedge", 5, 1),
            ],
        )
    }

    fn build_geometry(
        mesh: &SolverRuntimeMeshData,
    ) -> (
        CompactSimpleFaceAddressing,
        ScalarGradientGeometry,
        WeightedLeastSquaresGeometry,
    ) {
        let addressing = CompactSimpleFaceAddressing::from_mesh(mesh).expect("addressing");
        let scalar_geometry = ScalarGradientGeometry::from_mesh(mesh).expect("scalar geometry");
        let wls = WeightedLeastSquaresGeometry::from_mesh(mesh, &scalar_geometry, &addressing)
            .expect("WLS geometry");
        (addressing, scalar_geometry, wls)
    }

    fn dot(left: Point3, right: Point3) -> f64 {
        (left.x * right.x + left.y * right.y) + left.z * right.z
    }

    fn affine_value(location: Point3, gradient: Point3, intercept: f64) -> f64 {
        intercept + dot(location, gradient)
    }

    fn affine_boundary_deltas(mesh: &SolverRuntimeMeshData, gradient: Point3) -> Vec<Option<f64>> {
        let mut deltas = vec![None; mesh.faces];
        for patch in &mesh.patches {
            if patch.patch_type == "empty" {
                continue;
            }
            for (face, delta) in deltas
                .iter_mut()
                .enumerate()
                .skip(patch.start_face)
                .take(patch.faces)
            {
                if matches!(patch.patch_type.as_str(), "wedge" | "symmetryPlane") {
                    *delta = Some(0.0);
                    continue;
                }
                let owner = mesh.owner[face];
                let displacement = point(
                    mesh.face_centres[face].x - mesh.cell_centres[owner].x,
                    mesh.face_centres[face].y - mesh.cell_centres[owner].y,
                    mesh.face_centres[face].z - mesh.cell_centres[owner].z,
                );
                *delta = Some(dot(displacement, gradient));
            }
        }
        deltas
    }

    fn assert_point_bits(actual: Point3, expected: Point3) {
        assert_eq!(actual.x.to_bits(), expected.x.to_bits(), "x");
        assert_eq!(actual.y.to_bits(), expected.y.to_bits(), "y");
        assert_eq!(actual.z.to_bits(), expected.z.to_bits(), "z");
    }

    fn assert_point_close(actual: Point3, expected: Point3, tolerance: f64) {
        assert!(
            (actual.x - expected.x).abs() <= tolerance,
            "x: {} != {}",
            actual.x,
            expected.x
        );
        assert!(
            (actual.y - expected.y).abs() <= tolerance,
            "y: {} != {}",
            actual.y,
            expected.y
        );
        assert!(
            (actual.z - expected.z).abs() <= tolerance,
            "z: {} != {}",
            actual.z,
            expected.z
        );
    }

    #[test]
    fn wls_constant_and_affine_reproduction_is_exact_in_one_two_and_three_dimensions() {
        let cases = [
            (WlsIntrinsicDimension::One, point(2.0, 0.0, 0.0)),
            (WlsIntrinsicDimension::Two, point(2.0, -0.5, 0.0)),
            (WlsIntrinsicDimension::Three, point(2.0, -0.5, 0.25)),
        ];
        for (dimension, expected) in cases {
            let mesh = one_cell_box(dimension);
            let (_, _, geometry) = build_geometry(&mesh);
            assert_eq!(geometry.basis.dimension, dimension);

            let constant = weighted_least_squares_scalar_gradient_from_deltas(
                &geometry,
                &[3.0],
                &affine_boundary_deltas(&mesh, zero()),
            )
            .expect("constant gradient");
            assert_point_bits(constant[0], zero());

            let values = [affine_value(zero(), expected, 3.0)];
            let gradient = weighted_least_squares_scalar_gradient_from_deltas(
                &geometry,
                &values,
                &affine_boundary_deltas(&mesh, expected),
            )
            .expect("affine gradient");
            assert_point_bits(gradient[0], expected);
        }
    }

    #[test]
    fn wls_rotated_two_dimensional_and_skewed_three_dimensional_fields_match_affine_oracles() {
        let inverse_root_two = 0.5f64.sqrt();
        let first = point(inverse_root_two, inverse_root_two, 0.0);
        let second = point(-inverse_root_two, inverse_root_two, 0.0);
        let rotated_gradient = point(
            2.0 * first.x - 0.5 * second.x,
            2.0 * first.y - 0.5 * second.y,
            0.0,
        );
        let rotated = runtime_mesh(
            vec![zero()],
            vec![0; 6],
            vec![None; 6],
            vec![
                first,
                point(-first.x, -first.y, 0.0),
                second,
                point(-second.x, -second.y, 0.0),
                point(0.0, 0.0, 0.5),
                point(0.0, 0.0, -0.5),
            ],
            vec![
                first,
                point(-first.x, -first.y, 0.0),
                second,
                point(-second.x, -second.y, 0.0),
                point(0.0, 0.0, 1.0),
                point(0.0, 0.0, -1.0),
            ],
            vec![
                patch("physical", "patch", 0, 4),
                patch("frontAndBack", "empty", 4, 2),
            ],
        );
        let (_, _, geometry) = build_geometry(&rotated);
        let result = weighted_least_squares_scalar_gradient_from_deltas(
            &geometry,
            &[1.0],
            &affine_boundary_deltas(&rotated, rotated_gradient),
        )
        .expect("rotated 2D gradient");
        assert_point_close(result[0], rotated_gradient, 2.0e-14);

        let directions = [
            point(0.5, 0.0, 0.0),
            point(0.1, 0.5, 0.0),
            point(0.2, -0.1, 0.5),
        ];
        let mut centres = Vec::new();
        let mut areas = Vec::new();
        for direction in directions {
            centres.push(direction);
            centres.push(point(-direction.x, -direction.y, -direction.z));
            areas.push(direction);
            areas.push(point(-direction.x, -direction.y, -direction.z));
        }
        let skewed = runtime_mesh(
            vec![zero()],
            vec![0; 6],
            vec![None; 6],
            centres,
            areas,
            vec![patch("physical", "patch", 0, 6)],
        );
        let expected = point(1.25, -0.75, 0.5);
        let (_, _, geometry) = build_geometry(&skewed);
        let result = weighted_least_squares_scalar_gradient_from_deltas(
            &geometry,
            &[2.0],
            &affine_boundary_deltas(&skewed, expected),
        )
        .expect("skewed 3D gradient");
        assert_point_close(result[0], expected, 3.0e-13);
    }

    #[test]
    fn wls_power_of_two_geometry_scaling_and_face_permutation_preserve_affine_result() {
        let expected = point(1.0, -0.5, 0.25);
        let mut reference_bits = None;
        for exponent in [-500, 0, 500] {
            let scale = 2.0f64.powi(exponent);
            let area_scale = scale * scale;
            let mut mesh = one_cell_box(WlsIntrinsicDimension::Three);
            for centre in &mut mesh.face_centres {
                centre.x *= scale;
                centre.y *= scale;
                centre.z *= scale;
            }
            for area in &mut mesh.face_area_vectors {
                area.x *= area_scale;
                area.y *= area_scale;
                area.z *= area_scale;
            }
            if exponent == 0 {
                mesh.face_centres.reverse();
                mesh.face_area_vectors.reverse();
                mesh.owner.reverse();
                mesh.neighbour.reverse();
            }
            let addressing = CompactSimpleFaceAddressing::from_mesh(&mesh).expect("addressing");
            let scalar_geometry = ScalarGradientGeometry {
                owner_weights: vec![None; mesh.faces],
                boundary_normal_distances: vec![None; mesh.faces],
                inverse_cell_volumes: vec![1.0],
            };
            let geometry =
                WeightedLeastSquaresGeometry::from_mesh(&mesh, &scalar_geometry, &addressing)
                    .expect("scaled geometry");
            let result = weighted_least_squares_scalar_gradient_from_deltas(
                &geometry,
                &[0.0],
                &affine_boundary_deltas(&mesh, expected),
            )
            .expect("scaled affine gradient");
            assert_point_close(result[0], expected, 2.0e-14);
            let bits = [
                result[0].x.to_bits(),
                result[0].y.to_bits(),
                result[0].z.to_bits(),
            ];
            match reference_bits {
                None => reference_bits = Some(bits),
                Some(reference) => assert_eq!(bits, reference),
            }
        }
    }

    fn two_cell_full_rank() -> SolverRuntimeMeshData {
        runtime_mesh(
            vec![point(-0.75, 0.0, 0.0), point(0.25, 0.0, 0.0)],
            vec![0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            vec![
                Some(1),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            ],
            vec![
                zero(),
                point(-1.25, 0.0, 0.0),
                point(-0.75, 0.5, 0.0),
                point(-0.75, -0.5, 0.0),
                point(-0.75, 0.0, 0.5),
                point(-0.75, 0.0, -0.5),
                point(0.75, 0.0, 0.0),
                point(0.25, 0.5, 0.0),
                point(0.25, -0.5, 0.0),
                point(0.25, 0.0, 0.5),
                point(0.25, 0.0, -0.5),
            ],
            vec![
                point(1.0, 0.0, 0.0),
                point(-1.0, 0.0, 0.0),
                point(0.0, 1.0, 0.0),
                point(0.0, -1.0, 0.0),
                point(0.0, 0.0, 1.0),
                point(0.0, 0.0, -1.0),
                point(1.0, 0.0, 0.0),
                point(0.0, 1.0, 0.0),
                point(0.0, -1.0, 0.0),
                point(0.0, 0.0, 1.0),
                point(0.0, 0.0, -1.0),
            ],
            vec![patch("physical", "patch", 1, 10)],
        )
    }

    #[test]
    fn wls_internal_owner_neighbour_and_boundary_vectors_match_affine_oracle() {
        let mesh = two_cell_full_rank();
        let expected = point(1.5, -0.5, 0.25);
        let (_, scalar_geometry, geometry) = build_geometry(&mesh);
        assert_eq!(
            scalar_geometry.owner_weights[0]
                .expect("internal weight")
                .to_bits(),
            0.25f64.to_bits()
        );
        assert!((geometry.face_coefficients[0].owner.x - 3.0 / 7.0).abs() <= f64::EPSILON);
        assert!((geometry.face_coefficients[0].neighbour.x - 1.0 / 5.0).abs() <= f64::EPSILON);
        let values: Vec<_> = mesh
            .cell_centres
            .iter()
            .copied()
            .map(|centre| affine_value(centre, expected, 2.0))
            .collect();
        let result = weighted_least_squares_scalar_gradient_from_deltas(
            &geometry,
            &values,
            &affine_boundary_deltas(&mesh, expected),
        )
        .expect("two-cell gradient");
        assert_point_close(result[0], expected, 1.0e-14);
        assert_point_close(result[1], expected, 1.0e-14);

        let mut skew_boundary = one_cell_box(WlsIntrinsicDimension::Three);
        skew_boundary.face_centres[0].y = 0.25;
        let (_, _, skew_geometry) = build_geometry(&skew_boundary);
        assert_point_bits(
            skew_geometry.face_coefficients[0].owner,
            point(1.0, 0.0, 0.0),
        );
        let projected_gradient = point(1.0, 2.0, 3.0);
        let mut projected_deltas = affine_boundary_deltas(&skew_boundary, projected_gradient);
        projected_deltas[0] = Some(0.5 * projected_gradient.x);
        let result = weighted_least_squares_scalar_gradient_from_deltas(
            &skew_geometry,
            &[0.0],
            &projected_deltas,
        )
        .expect("projected boundary delta");
        assert_point_bits(result[0], projected_gradient);
    }

    #[test]
    fn wls_empty_wedge_and_symmetry_roles_preserve_intrinsic_dimension_and_scalar_constraints() {
        let empty = one_cell_box(WlsIntrinsicDimension::Two);
        let (_, _, empty_geometry) = build_geometry(&empty);
        assert_eq!(empty_geometry.basis.dimension, WlsIntrinsicDimension::Two);
        for coefficient in &empty_geometry.face_coefficients[4..] {
            assert_eq!(coefficient.role, WlsFaceRole::EmptyBoundary);
            assert_point_bits(coefficient.owner, zero());
        }
        let malformed_empty_slots = weighted_least_squares_scalar_gradient_from_deltas(
            &empty_geometry,
            &[1.0],
            &vec![Some(0.0); empty.faces],
        )
        .expect_err("empty slots must be absent");
        assert!(
            malformed_empty_slots
                .to_string()
                .contains("requires an empty scalar boundary-delta slot")
        );
        let empty_gradient = weighted_least_squares_scalar_gradient_from_deltas(
            &empty_geometry,
            &[1.0],
            &affine_boundary_deltas(&empty, point(1.0, 2.0, 0.0)),
        )
        .expect("empty gradient");
        assert_point_bits(empty_gradient[0], point(1.0, 2.0, 0.0));

        let wedge = runtime_mesh(
            vec![zero()],
            vec![0; 6],
            vec![None; 6],
            vec![
                point(0.5, 0.0, 0.0),
                point(-0.5, 0.0, 0.0),
                point(0.0, 0.0, 0.5),
                point(0.0, 0.0, -0.5),
                point(0.0, 0.5, 0.05),
                point(0.0, -0.5, 0.05),
            ],
            vec![
                point(1.0, 0.0, 0.0),
                point(-1.0, 0.0, 0.0),
                point(0.0, 0.0, 1.0),
                point(0.0, 0.0, -1.0),
                point(0.0, 1.0, 0.1),
                point(0.0, -1.0, 0.1),
            ],
            vec![
                patch("physical", "patch", 0, 4),
                patch("wedgeFront", "wedge", 4, 1),
                patch("wedgeBack", "wedge", 5, 1),
            ],
        );
        let (_, _, wedge_geometry) = build_geometry(&wedge);
        assert_eq!(wedge_geometry.basis.dimension, WlsIntrinsicDimension::Two);
        assert_eq!(
            wedge_geometry
                .wedge_vector_basis
                .expect("wedge vector basis")
                .dimension,
            WlsIntrinsicDimension::Three
        );
        let mut malformed_wedge = wedge.clone();
        malformed_wedge.patches = vec![
            patch("physical", "patch", 0, 4),
            patch("wedgeSides", "wedge", 4, 2),
        ];
        let malformed_addressing =
            CompactSimpleFaceAddressing::from_mesh(&malformed_wedge).expect("addressing");
        let malformed_scalar =
            ScalarGradientGeometry::from_mesh(&malformed_wedge).expect("scalar geometry");
        let malformed_error = WeightedLeastSquaresGeometry::from_mesh(
            &malformed_wedge,
            &malformed_scalar,
            &malformed_addressing,
        )
        .expect_err("one wedge patch must fail");
        assert!(
            malformed_error
                .to_string()
                .contains("requires exactly two non-empty wedge patches")
        );
        let wedge_gradient = weighted_least_squares_scalar_gradient_from_deltas(
            &wedge_geometry,
            &[3.0],
            &affine_boundary_deltas(&wedge, point(1.0, 0.0, -0.5)),
        )
        .expect("wedge gradient");
        assert_point_bits(wedge_gradient[0], point(1.0, 0.0, -0.5));
        let wedge_vector_gradients = [
            point(0.0, 1.0, 0.0),
            point(0.5, 0.0, -0.25),
            point(-0.5, 0.25, 0.5),
        ];
        let wedge_vector_deltas: Vec<_> = wedge
            .face_centres
            .iter()
            .copied()
            .map(|displacement| {
                Some(point(
                    dot(displacement, wedge_vector_gradients[0]),
                    dot(displacement, wedge_vector_gradients[1]),
                    dot(displacement, wedge_vector_gradients[2]),
                ))
            })
            .collect();
        let wedge_vector = weighted_least_squares_vector_component_gradients_from_deltas(
            &wedge_geometry,
            &[zero()],
            &wedge_vector_deltas,
        )
        .expect("wedge vector gradient");
        for component in 0..3 {
            assert_point_close(
                wedge_vector[component][0],
                wedge_vector_gradients[component],
                2.0e-14,
            );
        }

        let mut symmetry = one_cell_box(WlsIntrinsicDimension::Three);
        symmetry.patches = vec![
            patch("symmetry", "symmetryPlane", 0, 1),
            patch("physical", "patch", 1, 5),
        ];
        let (_, _, symmetry_geometry) = build_geometry(&symmetry);
        assert_eq!(
            symmetry_geometry.basis.dimension,
            WlsIntrinsicDimension::Three
        );
        let symmetry_gradient = weighted_least_squares_scalar_gradient_from_deltas(
            &symmetry_geometry,
            &[1.0],
            &vec![Some(0.0); symmetry.faces],
        )
        .expect("symmetry gradient");
        assert_point_bits(symmetry_gradient[0], zero());
    }

    #[test]
    fn wls_boundary_deltas_follow_regular_empty_symmetry_and_wedge_contracts() {
        let regular = one_cell_box(WlsIntrinsicDimension::Three);
        let (addressing, scalar_geometry, geometry) = build_geometry(&regular);
        let values = [2.0];
        let mut scalar_boundary = vec![ScalarFaceTreatment::ZeroGradient; regular.faces];
        scalar_boundary[0] = ScalarFaceTreatment::FixedValue(5.0);
        let deltas = wls_scalar_boundary_deltas(
            &regular,
            &scalar_geometry,
            &geometry,
            &addressing,
            &values,
            &scalar_boundary,
        )
        .expect("fixed-value scalar deltas");
        assert_eq!(
            deltas[0].expect("fixed-value delta").to_bits(),
            3.0f64.to_bits()
        );
        for delta in &deltas[1..] {
            assert_eq!(
                delta.expect("zero-gradient delta").to_bits(),
                0.0f64.to_bits()
            );
        }

        scalar_boundary[0] = ScalarFaceTreatment::FixedGradient(4.0);
        let deltas = wls_scalar_boundary_deltas(
            &regular,
            &scalar_geometry,
            &geometry,
            &addressing,
            &values,
            &scalar_boundary,
        )
        .expect("fixed-gradient scalar deltas");
        assert_eq!(
            deltas[0].expect("fixed-gradient delta").to_bits(),
            2.0f64.to_bits()
        );
        scalar_boundary[0] = ScalarFaceTreatment::InletOutlet(5.0);
        assert!(
            wls_scalar_boundary_deltas(
                &regular,
                &scalar_geometry,
                &geometry,
                &addressing,
                &values,
                &scalar_boundary,
            )
            .expect_err("unresolved scalar inletOutlet must fail")
            .to_string()
            .contains("must be resolved against the current flux")
        );
        scalar_boundary[0] = ScalarFaceTreatment::Constraint;
        assert!(
            wls_scalar_boundary_deltas(
                &regular,
                &scalar_geometry,
                &geometry,
                &addressing,
                &values,
                &scalar_boundary,
            )
            .expect_err("regular scalar constraint must fail")
            .to_string()
            .contains("regular scalar face")
        );

        let empty = one_cell_box(WlsIntrinsicDimension::Two);
        let (addressing, scalar_geometry, geometry) = build_geometry(&empty);
        let deltas = wls_scalar_boundary_deltas(
            &empty,
            &scalar_geometry,
            &geometry,
            &addressing,
            &[1.0],
            &vec![ScalarFaceTreatment::Constraint; empty.faces],
        )
        .expect_err("regular faces cannot masquerade as constraints");
        assert!(deltas.to_string().contains("regular scalar face"));
        let mut empty_boundary = vec![ScalarFaceTreatment::ZeroGradient; empty.faces];
        empty_boundary[4] = ScalarFaceTreatment::Constraint;
        empty_boundary[5] = ScalarFaceTreatment::Constraint;
        let deltas = wls_scalar_boundary_deltas(
            &empty,
            &scalar_geometry,
            &geometry,
            &addressing,
            &[1.0],
            &empty_boundary,
        )
        .expect("empty scalar slots");
        assert!(deltas[4].is_none());
        assert!(deltas[5].is_none());

        let vector_values = [point(1.0, 2.0, 3.0)];
        let mut vector_boundary = vec![VectorFaceTreatment::ZeroGradient; regular.faces];
        vector_boundary[0] = VectorFaceTreatment::FixedValue(point(4.0, 6.0, 8.0));
        let mut flux = vec![0.0; regular.faces];
        let deltas = wls_vector_boundary_deltas(
            &regular,
            &build_geometry(&regular).2,
            &vector_values,
            &vector_boundary,
            &flux,
        )
        .expect("fixed-value vector deltas");
        assert_point_bits(
            deltas[0].expect("fixed-value vector delta"),
            point(3.0, 4.0, 5.0),
        );

        vector_boundary[0] = VectorFaceTreatment::InletOutlet(point(4.0, 6.0, 8.0));
        flux[0] = -f64::from_bits(1);
        let regular_geometry = build_geometry(&regular).2;
        let deltas = wls_vector_boundary_deltas(
            &regular,
            &regular_geometry,
            &vector_values,
            &vector_boundary,
            &flux,
        )
        .expect("negative inlet flux");
        assert_point_bits(deltas[0].expect("inlet delta"), point(3.0, 4.0, 5.0));
        for outlet_flux in [-0.0, 0.0] {
            flux[0] = outlet_flux;
            let deltas = wls_vector_boundary_deltas(
                &regular,
                &regular_geometry,
                &vector_values,
                &vector_boundary,
                &flux,
            )
            .expect("non-negative outlet flux");
            assert_point_bits(deltas[0].expect("outlet delta"), zero());
        }
        flux[0] = f64::NAN;
        assert!(
            wls_vector_boundary_deltas(
                &regular,
                &regular_geometry,
                &vector_values,
                &vector_boundary,
                &flux,
            )
            .expect_err("non-finite inletOutlet flux must fail")
            .to_string()
            .contains("non-finite decision flux")
        );

        let mut symmetry = one_cell_box(WlsIntrinsicDimension::Three);
        let inverse_root_two = 0.5f64.sqrt();
        symmetry.face_area_vectors[0] = point(inverse_root_two, inverse_root_two, 0.0);
        symmetry.face_centres[0] = point(0.5 * inverse_root_two, 0.5 * inverse_root_two, 0.0);
        symmetry.patches = vec![
            patch("symmetry", "symmetryPlane", 0, 1),
            patch("physical", "patch", 1, 5),
        ];
        let (_, _, symmetry_geometry) = build_geometry(&symmetry);
        let mut symmetry_boundary = vec![VectorFaceTreatment::ZeroGradient; symmetry.faces];
        symmetry_boundary[0] = VectorFaceTreatment::Constraint;
        let deltas = wls_vector_boundary_deltas(
            &symmetry,
            &symmetry_geometry,
            &[point(1.0, 0.0, 2.0)],
            &symmetry_boundary,
            &vec![0.0; symmetry.faces],
        )
        .expect("symmetry vector delta");
        assert_point_close(
            deltas[0].expect("symmetry delta"),
            point(-0.5, -0.5, 0.0),
            4.0 * f64::EPSILON,
        );
    }

    #[test]
    fn wls_wedge_rotation_matches_closed_oracles_and_is_scale_invariant() {
        let angle = std::f64::consts::FRAC_PI_6;
        let (sine, cosine) = angle.sin_cos();
        let mesh = one_cell_wedge(angle);
        let (_, scalar_geometry, geometry) = build_geometry(&mesh);
        let owner_value = point(1.0, 2.0, 3.0);
        let mut boundary = vec![VectorFaceTreatment::ZeroGradient; mesh.faces];
        boundary[4] = VectorFaceTreatment::Constraint;
        boundary[5] = VectorFaceTreatment::Constraint;
        let deltas = wls_vector_boundary_deltas(
            &mesh,
            &geometry,
            &[owner_value],
            &boundary,
            &vec![0.0; mesh.faces],
        )
        .expect("wedge vector deltas");
        assert_point_close(
            deltas[4].expect("plus wedge delta"),
            point(cosine - 1.0 - 2.0 * sine, sine + 2.0 * (cosine - 1.0), 0.0),
            8.0 * f64::EPSILON,
        );
        assert_point_close(
            deltas[5].expect("minus wedge delta"),
            point(cosine - 1.0 + 2.0 * sine, -sine + 2.0 * (cosine - 1.0), 0.0),
            8.0 * f64::EPSILON,
        );

        for (face, centre_normal, patch_normal) in [
            (4, point(1.0, 0.0, 0.0), point(cosine, sine, 0.0)),
            (5, point(-1.0, 0.0, 0.0), point(-cosine, sine, 0.0)),
        ] {
            let WlsVectorConstraint::Wedge { rotation } =
                geometry.vector_constraints[face].expect("wedge transform")
            else {
                panic!("face {face} must carry a wedge transform");
            };
            let mapped = wls_checked_add(
                centre_normal,
                rotation.delta(centre_normal, face).expect("normal delta"),
                || "mapped normal".to_string(),
            )
            .expect("mapped normal sum");
            assert_point_close(mapped, patch_normal, 8.0 * f64::EPSILON);
            let transformed = wls_checked_add(
                owner_value,
                rotation.delta(owner_value, face).expect("value delta"),
                || "transformed value".to_string(),
            )
            .expect("transformed value sum");
            assert!(
                (wls_checked_magnitude(transformed, || "transformed norm".to_string())
                    .expect("transformed norm")
                    - wls_checked_magnitude(owner_value, || "owner norm".to_string())
                        .expect("owner norm"))
                .abs()
                    <= 16.0 * f64::EPSILON
            );
            assert_point_close(
                rotation
                    .delta(point(0.0, 0.0, 4.0), face)
                    .expect("axis delta"),
                zero(),
                8.0 * f64::EPSILON,
            );
        }

        let mut scalar_boundary = vec![ScalarFaceTreatment::ZeroGradient; mesh.faces];
        scalar_boundary[4] = ScalarFaceTreatment::Constraint;
        scalar_boundary[5] = ScalarFaceTreatment::Constraint;
        let addressing = CompactSimpleFaceAddressing::from_mesh(&mesh).expect("addressing");
        let scalar_deltas = wls_scalar_boundary_deltas(
            &mesh,
            &scalar_geometry,
            &geometry,
            &addressing,
            &[7.0],
            &scalar_boundary,
        )
        .expect("wedge scalar deltas");
        assert_eq!(
            scalar_deltas[4].expect("plus scalar delta").to_bits(),
            0.0f64.to_bits()
        );
        assert_eq!(
            scalar_deltas[5].expect("minus scalar delta").to_bits(),
            0.0f64.to_bits()
        );

        let mut scaled = mesh.clone();
        for area in &mut scaled.face_area_vectors {
            area.x *= 2.0f64.powi(500);
            area.y *= 2.0f64.powi(500);
            area.z *= 2.0f64.powi(500);
        }
        scaled.patches.swap(1, 2);
        let (_, _, scaled_geometry) = build_geometry(&scaled);
        let scaled_deltas = wls_vector_boundary_deltas(
            &scaled,
            &scaled_geometry,
            &[owner_value],
            &boundary,
            &vec![0.0; scaled.faces],
        )
        .expect("scaled wedge vector deltas");
        assert_point_bits(
            scaled_deltas[4].expect("scaled plus delta"),
            deltas[4].expect("reference plus delta"),
        );
        assert_point_bits(
            scaled_deltas[5].expect("scaled minus delta"),
            deltas[5].expect("reference minus delta"),
        );
    }

    #[test]
    fn wls_wedge_geometry_rejects_degenerate_nonplanar_and_asymmetric_pairs() {
        let aligned = one_cell_wedge(0.0);
        let addressing = CompactSimpleFaceAddressing::from_mesh(&aligned).expect("addressing");
        let scalar_geometry = ScalarGradientGeometry::from_mesh(&aligned).expect("scalar geometry");
        WeightedLeastSquaresGeometry::from_mesh(&aligned, &scalar_geometry, &addressing)
            .expect_err("aligned wedge must fail");
        assert!(
            wls_rotation_between(point(1.0, 0.0, 0.0), point(1.0, 0.0, 0.0), 0,)
                .expect_err("aligned wedge transform must fail")
                .to_string()
                .contains("aligns with its coordinate centre plane")
        );

        let angle = std::f64::consts::FRAC_PI_6;
        let mut asymmetric = one_cell_wedge(angle);
        let (other_sine, other_cosine) = (angle * 0.75).sin_cos();
        asymmetric.face_area_vectors[5] = point(-other_cosine, other_sine, 0.0);
        asymmetric.face_centres[5] = point(-0.5 * other_cosine, 0.5 * other_sine, 0.0);
        let addressing = CompactSimpleFaceAddressing::from_mesh(&asymmetric).expect("addressing");
        let scalar_geometry =
            ScalarGradientGeometry::from_mesh(&asymmetric).expect("scalar geometry");
        assert!(
            WeightedLeastSquaresGeometry::from_mesh(&asymmetric, &scalar_geometry, &addressing,)
                .expect_err("asymmetric wedge must fail")
                .to_string()
                .contains("not symmetric")
        );

        let mut nonplanar = one_cell_box(WlsIntrinsicDimension::Three);
        nonplanar.face_area_vectors[0] = point(1.0, 0.0, 0.0);
        nonplanar.face_area_vectors[1] = point(1.0, 1.0e-5, 0.0);
        nonplanar.patches = vec![
            patch("nonPlanar", "symmetryPlane", 0, 2),
            patch("physical", "patch", 2, 4),
        ];
        assert!(
            wls_patch_average_unit_normal(&nonplanar, 0)
                .expect_err("non-planar patch must fail")
                .to_string()
                .contains("non-planar")
        );
        nonplanar.face_area_vectors[0] = zero();
        assert!(
            wls_patch_average_unit_normal(&nonplanar, 0)
                .expect_err("zero patch normal must fail")
                .to_string()
                .contains("must be nonzero")
        );
        nonplanar.face_area_vectors[0] = point(f64::NAN, 0.0, 0.0);
        assert!(
            wls_patch_average_unit_normal(&nonplanar, 0)
                .expect_err("non-finite patch normal must fail")
                .to_string()
                .contains("finite components")
        );

        let mut misoriented = one_cell_box(WlsIntrinsicDimension::Three);
        misoriented.face_area_vectors[0] = point(-1.0, 0.0, 0.0);
        let addressing = CompactSimpleFaceAddressing::from_mesh(&misoriented).expect("addressing");
        let scalar_geometry =
            ScalarGradientGeometry::from_mesh(&misoriented).expect("scalar geometry");
        assert!(
            WeightedLeastSquaresGeometry::from_mesh(&misoriented, &scalar_geometry, &addressing,)
                .expect_err("non-positive boundary displacement must fail")
                .to_string()
                .contains("non-positive normal displacement")
        );

        let diagonal =
            wls_normalize_oriented(point(0.8, 0.6, 0.0), || "diagonal normal".to_string())
                .expect("diagonal unit normal");
        assert!(
            wls_coordinate_plane_normal(diagonal, 7)
                .expect_err("diagonal centre plane must fail")
                .to_string()
                .contains("does not align with a coordinate plane")
        );

        let mut with_empty_metadata = one_cell_wedge(angle);
        with_empty_metadata.patches.push(patch(
            "unusedWedge",
            "wedge",
            with_empty_metadata.faces,
            0,
        ));
        build_geometry(&with_empty_metadata);

        let mut three_wedges = one_cell_wedge(angle);
        three_wedges.patches = vec![
            patch("physical", "patch", 0, 3),
            patch("thirdWedge", "wedge", 3, 1),
            patch("wedgePlus", "wedge", 4, 1),
            patch("wedgeMinus", "wedge", 5, 1),
        ];
        let three_addressing =
            CompactSimpleFaceAddressing::from_mesh(&three_wedges).expect("addressing");
        let three_scalar_geometry =
            ScalarGradientGeometry::from_mesh(&three_wedges).expect("scalar geometry");
        assert!(
            WeightedLeastSquaresGeometry::from_mesh(
                &three_wedges,
                &three_scalar_geometry,
                &three_addressing,
            )
            .expect_err("three wedge patches must fail at the production constructor preflight")
            .to_string()
            .contains("got 3")
        );

        let (sine, cosine) = angle.sin_cos();
        let first = WlsRotation {
            vector: point(0.0, 0.0, sine),
            cosine,
            inverse_one_plus_cosine: 1.0 / (1.0 + cosine),
        };
        let second = WlsRotation {
            vector: point(0.0, 0.0, -sine),
            cosine,
            inverse_one_plus_cosine: 1.0 / (1.0 + cosine),
        };
        let owner_mesh = runtime_mesh(
            vec![zero(), point(1.0, 0.0, 0.0)],
            vec![0, 1, 1, 0],
            vec![None; 4],
            vec![zero(); 4],
            vec![point(1.0, 0.0, 0.0); 4],
            vec![
                patch("first", "wedge", 0, 2),
                patch("second", "wedge", 2, 2),
            ],
        );
        wls_validate_wedge_pair(
            &owner_mesh,
            &[
                (0, point(1.0, 0.0, 0.0), first),
                (1, point(-1.0, 0.0, 0.0), second),
            ],
        )
        .expect("owner signatures may use different face order");
        let distinct_owner_mesh = runtime_mesh(
            vec![zero(), point(1.0, 0.0, 0.0), point(2.0, 0.0, 0.0)],
            vec![0, 1, 2, 0],
            vec![None; 4],
            vec![zero(); 4],
            vec![point(1.0, 0.0, 0.0); 4],
            vec![
                patch("first", "wedge", 0, 2),
                patch("second", "wedge", 2, 2),
            ],
        );
        assert!(
            wls_validate_wedge_pair(
                &distinct_owner_mesh,
                &[
                    (0, point(1.0, 0.0, 0.0), first),
                    (1, point(-1.0, 0.0, 0.0), second),
                ],
            )
            .expect_err("different unique wedge owner sets must fail")
            .to_string()
            .contains("do not cover the same owner cells")
        );
        let mut duplicate_owner = owner_mesh.clone();
        duplicate_owner.owner[2] = 0;
        assert!(
            wls_validate_wedge_pair(
                &duplicate_owner,
                &[
                    (0, point(1.0, 0.0, 0.0), first),
                    (1, point(-1.0, 0.0, 0.0), second),
                ],
            )
            .expect_err("duplicate wedge owner must fail")
            .to_string()
            .contains("more than one face for an owner cell")
        );
    }

    #[test]
    fn wls_squared_magnitude_guard_bands_have_exact_boundaries() {
        let above_inactive = f64::from_bits(WLS_BASIS_INACTIVE_SQUARED.to_bits() + 1);
        let below_active = f64::from_bits(WLS_BASIS_ACTIVE_SQUARED.to_bits() - 1);

        assert_eq!(
            wls_squared_magnitude_band(WLS_BASIS_INACTIVE_SQUARED),
            WlsSquaredMagnitudeBand::Inactive
        );
        assert_eq!(
            wls_squared_magnitude_band(above_inactive),
            WlsSquaredMagnitudeBand::Ambiguous
        );
        assert_eq!(
            wls_squared_magnitude_band(below_active),
            WlsSquaredMagnitudeBand::Ambiguous
        );
        assert_eq!(
            wls_squared_magnitude_band(WLS_BASIS_ACTIVE_SQUARED),
            WlsSquaredMagnitudeBand::Active
        );
        assert_eq!(
            wls_squared_magnitude_band(f64::NAN),
            WlsSquaredMagnitudeBand::Active
        );
    }

    #[test]
    fn wls_vector_kernel_uses_pre_resolved_constraint_deltas_and_matches_scalar_oracles() {
        let mesh = one_cell_box(WlsIntrinsicDimension::Three);
        let (_, _, geometry) = build_geometry(&mesh);
        let expected = [
            point(1.0, 2.0, 3.0),
            point(-1.0, 0.5, 0.0),
            point(0.25, -0.5, 1.0),
        ];
        let mut vector_deltas = vec![None; mesh.faces];
        let mut scalar_deltas = [
            vec![None; mesh.faces],
            vec![None; mesh.faces],
            vec![None; mesh.faces],
        ];
        for face in 0..mesh.faces {
            let displacement = mesh.face_centres[face];
            let delta = point(
                dot(displacement, expected[0]),
                dot(displacement, expected[1]),
                dot(displacement, expected[2]),
            );
            vector_deltas[face] = Some(delta);
            scalar_deltas[0][face] = Some(delta.x);
            scalar_deltas[1][face] = Some(delta.y);
            scalar_deltas[2][face] = Some(delta.z);
        }
        let vector = weighted_least_squares_vector_component_gradients_from_deltas(
            &geometry,
            &[point(5.0, -2.0, 4.0)],
            &vector_deltas,
        )
        .expect("vector gradients");
        for component in 0..3 {
            let scalar = weighted_least_squares_scalar_gradient_from_deltas(
                &geometry,
                &[component as f64],
                &scalar_deltas[component],
            )
            .expect("scalar oracle");
            assert_point_bits(vector[component][0], scalar[0]);
            assert_point_bits(vector[component][0], expected[component]);
        }
    }

    #[test]
    fn wls_rejects_rank_mismatch_and_lowest_local_stencil_deficiency_without_fallback() {
        let planar = runtime_mesh(
            vec![zero()],
            vec![0; 4],
            vec![None; 4],
            vec![
                point(0.5, 0.0, 0.0),
                point(-0.5, 0.0, 0.0),
                point(0.0, 0.5, 0.0),
                point(0.0, -0.5, 0.0),
            ],
            vec![
                point(1.0, 0.0, 0.0),
                point(-1.0, 0.0, 0.0),
                point(0.0, 1.0, 0.0),
                point(0.0, -1.0, 0.0),
            ],
            vec![patch("physical", "patch", 0, 4)],
        );
        let addressing = CompactSimpleFaceAddressing::from_mesh(&planar).expect("addressing");
        let scalar_geometry = ScalarGradientGeometry::from_mesh(&planar).expect("scalar geometry");
        let error = WeightedLeastSquaresGeometry::from_mesh(&planar, &scalar_geometry, &addressing)
            .expect_err("planar unconstrained mesh must fail");
        assert!(error.to_string().contains("expected 3, certified 2"));

        let deficient = runtime_mesh(
            vec![point(-0.5, 0.0, 0.0), point(0.5, 0.0, 0.0)],
            vec![0, 0, 0, 0, 0, 0, 1],
            vec![Some(1), None, None, None, None, None, None],
            vec![
                zero(),
                point(-1.0, 0.0, 0.0),
                point(-0.5, 0.5, 0.0),
                point(-0.5, -0.5, 0.0),
                point(-0.5, 0.0, 0.5),
                point(-0.5, 0.0, -0.5),
                point(1.0, 0.0, 0.0),
            ],
            vec![
                point(1.0, 0.0, 0.0),
                point(-1.0, 0.0, 0.0),
                point(0.0, 1.0, 0.0),
                point(0.0, -1.0, 0.0),
                point(0.0, 0.0, 1.0),
                point(0.0, 0.0, -1.0),
                point(1.0, 0.0, 0.0),
            ],
            vec![patch("physical", "patch", 1, 6)],
        );
        let addressing = CompactSimpleFaceAddressing::from_mesh(&deficient).expect("addressing");
        let scalar_geometry =
            ScalarGradientGeometry::from_mesh(&deficient).expect("scalar geometry");
        let error =
            WeightedLeastSquaresGeometry::from_mesh(&deficient, &scalar_geometry, &addressing)
                .expect_err("local deficiency must fail");
        assert!(
            error
                .to_string()
                .contains("weighted least-squares cell 1 is locally rank deficient")
        );

        let inactive = 0.125;
        let active = 0.5;
        let diagonal_matrix = |pivot: f64| [[pivot, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]];
        let error = wls_pivoted_ldlt(diagonal_matrix(-0.25), 1, inactive, active, 7)
            .expect_err("negative pivot");
        assert!(error.to_string().contains("not positive semidefinite"));
        let error = wls_pivoted_ldlt(diagonal_matrix(inactive), 1, inactive, active, 7)
            .expect_err("inactive equality");
        assert!(error.to_string().contains("locally rank deficient"));
        let error = wls_pivoted_ldlt(diagonal_matrix(0.25), 1, inactive, active, 7)
            .expect_err("guard interval");
        assert!(error.to_string().contains("numerically ambiguous"));
        wls_pivoted_ldlt(diagonal_matrix(active), 1, inactive, active, 7).expect("active equality");

        let factor = wls_pivoted_ldlt(
            [[4.0, 1.0, 0.0], [1.0, 3.0, 0.0], [0.0, 0.0, 2.0]],
            3,
            1.0e-15,
            1.0e-14,
            0,
        )
        .expect("pivoted factor");
        let solution = factor.solve([9.0, 7.0, 4.0], 0).expect("factor solve");
        assert!((solution[0] - 20.0 / 11.0).abs() <= 4.0 * f64::EPSILON);
        assert!((solution[1] - 19.0 / 11.0).abs() <= 4.0 * f64::EPSILON);
        assert!((solution[2] - 2.0).abs() <= f64::EPSILON);
    }

    #[test]
    fn wls_validation_order_is_structural_then_face_geometry_then_application_data() {
        let valid = one_cell_box(WlsIntrinsicDimension::Three);
        let valid_addressing =
            CompactSimpleFaceAddressing::from_mesh(&valid).expect("valid addressing");
        let valid_scalar =
            ScalarGradientGeometry::from_mesh(&valid).expect("valid scalar geometry");

        let mut structural = valid.clone();
        structural.face_centres.pop();
        let error =
            WeightedLeastSquaresGeometry::from_mesh(&structural, &valid_scalar, &valid_addressing)
                .expect_err("structural mismatch");
        assert!(error.to_string().contains("requires 6 owner, neighbour"));

        let mut invalid_face = valid.clone();
        invalid_face.face_area_vectors[0].x = f64::NAN;
        let error = WeightedLeastSquaresGeometry::from_mesh(
            &invalid_face,
            &valid_scalar,
            &valid_addressing,
        )
        .expect_err("invalid face");
        assert!(error.to_string().contains("face 0 area vector"));

        let topology = two_cell_full_rank();
        let topology_scalar =
            ScalarGradientGeometry::from_mesh(&topology).expect("topology scalar geometry");
        let mut remapped = topology.clone();
        remapped.owner[1] = 1;
        let remapped_addressing =
            CompactSimpleFaceAddressing::from_mesh(&remapped).expect("remapped addressing");
        let error = WeightedLeastSquaresGeometry::from_mesh(
            &topology,
            &topology_scalar,
            &remapped_addressing,
        )
        .expect_err("mismatched addressing");
        assert!(
            error
                .to_string()
                .contains("addressing does not match runtime mesh face 1")
        );

        let geometry =
            WeightedLeastSquaresGeometry::from_mesh(&valid, &valid_scalar, &valid_addressing)
                .expect("valid geometry");
        let error = weighted_least_squares_scalar_gradient_from_deltas(
            &geometry,
            &[],
            &vec![Some(0.0); valid.faces],
        )
        .expect_err("field length mismatch");
        assert!(error.to_string().contains("expected 1 cell values, got 0"));
        let error = weighted_least_squares_scalar_gradient_from_deltas(
            &geometry,
            &[f64::NAN],
            &vec![Some(0.0); valid.faces],
        )
        .expect_err("non-finite cell");
        assert!(error.to_string().contains("scalar cell 0 value"));
        let mut bad_delta = vec![Some(0.0); valid.faces];
        bad_delta[0] = Some(f64::INFINITY);
        let error =
            weighted_least_squares_scalar_gradient_from_deltas(&geometry, &[0.0], &bad_delta)
                .expect_err("non-finite boundary delta");
        assert!(error.to_string().contains("boundary face 0 scalar delta"));

        let internal = two_cell_full_rank();
        let (_, _, internal_geometry) = build_geometry(&internal);
        let mut internal_deltas = affine_boundary_deltas(&internal, zero());
        internal_deltas[0] = Some(0.0);
        let error = weighted_least_squares_scalar_gradient_from_deltas(
            &internal_geometry,
            &[0.0, 0.0],
            &internal_deltas,
        )
        .expect_err("internal delta slot must be absent");
        assert!(
            error
                .to_string()
                .contains("face 0 requires an empty scalar boundary-delta slot")
        );
    }

    #[test]
    fn wls_allocations_are_fallible_and_geometry_storage_is_stable_for_ten_lifecycles() {
        assert!(matches!(
            wls_try_vec::<u8>(usize::MAX),
            Err(MeshError::OutOfMemory)
        ));
        let mesh = one_cell_box(WlsIntrinsicDimension::Three);
        let (_, _, geometry) = build_geometry(&mesh);
        let identity = geometry.storage_identity();
        let expected = point(1.0, -0.5, 0.25);
        let scalar_deltas = affine_boundary_deltas(&mesh, expected);
        let vector_deltas: Vec<_> = scalar_deltas
            .iter()
            .map(|delta| delta.map(|value| point(value, -2.0 * value, 0.5 * value)))
            .collect();
        let scalar_reference =
            weighted_least_squares_scalar_gradient_from_deltas(&geometry, &[2.0], &scalar_deltas)
                .expect("scalar reference");
        let vector_reference = weighted_least_squares_vector_component_gradients_from_deltas(
            &geometry,
            &[point(2.0, -4.0, 1.0)],
            &vector_deltas,
        )
        .expect("vector reference");

        for _ in 0..10 {
            let scalar = weighted_least_squares_scalar_gradient_from_deltas(
                &geometry,
                &[2.0],
                &scalar_deltas,
            )
            .expect("scalar lifecycle");
            let vector = weighted_least_squares_vector_component_gradients_from_deltas(
                &geometry,
                &[point(2.0, -4.0, 1.0)],
                &vector_deltas,
            )
            .expect("vector lifecycle");
            assert_eq!(geometry.storage_identity(), identity);
            assert_point_bits(scalar[0], scalar_reference[0]);
            for component in 0..3 {
                assert_point_bits(vector[component][0], vector_reference[component][0]);
            }
        }
    }
}
