//! Deterministic, body-fitted O-grids for the steady cylinder validation case.
//!
//! Point ordering is extrusion layer, radial ring, then angular index.  The
//! angular seam is periodic and is not duplicated.  Production presets share
//! reduced rational coordinates, so every coarse point is bitwise present in
//! the fine mesh.

use std::f64::consts::{FRAC_1_SQRT_2, TAU};

use crate::{BoundaryFace, Cell, Mesh, MeshError, PhysicalName, Point3, Result};

const INLET_TAG: i32 = 1;
const OUTLET_TAG: i32 = 2;
const CYLINDER_TAG: i32 = 3;
const FRONT_AND_BACK_TAG: i32 = 4;
const FLUID_TAG: i32 = 10;

/// Built-in cylinder O-grid sizes used by validation and production evidence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CylinderOGridPreset {
    /// The checked-in 16 x 3, 48-cell regression geometry.
    LegacySmoke,
    /// A 128 x 42, 5,376-cell production comparison grid.
    Coarse,
    /// A 256 x 84, 21,504-cell production comparison grid.
    Fine,
}

impl CylinderOGridPreset {
    /// Returns the complete, deterministic configuration for this preset.
    pub fn config(self) -> CylinderOGridConfig {
        const DIAMETER: f64 = 0.001;

        match self {
            Self::LegacySmoke => CylinderOGridConfig {
                angular_cells: 16,
                radial_cells: 3,
                diameter: DIAMETER,
                x_min: -4.0 * DIAMETER,
                x_max: 8.0 * DIAMETER,
                y_min: -4.0 * DIAMETER,
                y_max: 4.0 * DIAMETER,
                depth: 0.1 * DIAMETER,
                radial_grading: CylinderRadialGrading::Linear,
                outer_patch_layout: CylinderOuterPatchLayout::LegacyLeftInlet,
            },
            Self::Coarse => production_config(128, 42, DIAMETER),
            Self::Fine => production_config(256, 84, DIAMETER),
        }
    }
}

fn production_config(
    angular_cells: usize,
    radial_cells: usize,
    diameter: f64,
) -> CylinderOGridConfig {
    CylinderOGridConfig {
        angular_cells,
        radial_cells,
        diameter,
        x_min: -100.0 * diameter,
        x_max: 100.0 * diameter,
        y_min: -100.0 * diameter,
        y_max: 100.0 * diameter,
        depth: diameter,
        radial_grading: CylinderRadialGrading::Exponential { ratio: 1000.0 },
        outer_patch_layout: CylinderOuterPatchLayout::RightOutlet,
    }
}

/// Radial interpolation from the cylinder to the finite outer boundary.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CylinderRadialGrading {
    Linear,
    /// Continuous mapping `g(t) = (ratio^t - 1) / (ratio - 1)`.
    Exponential {
        ratio: f64,
    },
}

/// Assignment of the finite outer boundary to inlet and outlet patches.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CylinderOuterPatchLayout {
    /// Preserve the original smoke case: the four left faces are inlet and all
    /// other outer faces are outlet.
    LegacyLeftInlet,
    /// Production case: the right side is outlet; left, top, and bottom are
    /// inlet.
    RightOutlet,
}

/// Complete physical and discretization parameters for a one-cell-thick O-grid.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CylinderOGridConfig {
    pub angular_cells: usize,
    pub radial_cells: usize,
    pub diameter: f64,
    pub x_min: f64,
    pub x_max: f64,
    pub y_min: f64,
    pub y_max: f64,
    pub depth: f64,
    pub radial_grading: CylinderRadialGrading,
    pub outer_patch_layout: CylinderOuterPatchLayout,
}

/// Generates a deterministic body-fitted, extruded Hex8 mesh.
pub fn generate_cylinder_ogrid(config: &CylinderOGridConfig) -> Result<Mesh> {
    let counts = validate_config(config)?;
    let radius = config.diameter * 0.5;
    let half_depth = config.depth * 0.5;

    let mut points = Vec::new();
    points
        .try_reserve_exact(counts.point_count)
        .map_err(|_| MeshError::OutOfMemory)?;

    for layer in 0..2 {
        let z = normalize_zero(if layer == 0 { -half_depth } else { half_depth });
        for radial_index in 0..counts.ring_count {
            let fraction = radial_fraction(radial_index, config.radial_cells);
            let grading = grading_fraction(config.radial_grading, fraction)?;
            for angular_index in 0..config.angular_cells {
                let ray = boundary_ray(config, angular_index);
                let inner_x = radius * ray.x;
                let inner_y = radius * ray.y;
                let (x, y) = if radial_index == 0 {
                    (inner_x, inner_y)
                } else if radial_index == config.radial_cells {
                    (ray.outer_x, ray.outer_y)
                } else {
                    (
                        inner_x + (ray.outer_x - inner_x) * grading,
                        inner_y + (ray.outer_y - inner_y) * grading,
                    )
                };
                let point = Point3 {
                    x: normalize_zero(x),
                    y: normalize_zero(y),
                    z,
                };
                if !point.x.is_finite() || !point.y.is_finite() || !point.z.is_finite() {
                    return Err(invalid("cylinder O-grid produced a non-finite point"));
                }
                points.push(point);
            }
        }
    }

    let mut cells = Vec::new();
    cells
        .try_reserve_exact(counts.cell_count)
        .map_err(|_| MeshError::OutOfMemory)?;
    for radial_index in 0..config.radial_cells {
        for angular_index in 0..config.angular_cells {
            let next_angular = next_angular(angular_index, config.angular_cells);
            let bottom_inner = point_index(
                0,
                radial_index,
                angular_index,
                counts.points_per_layer,
                config.angular_cells,
            )?;
            let bottom_outer = point_index(
                0,
                radial_index + 1,
                angular_index,
                counts.points_per_layer,
                config.angular_cells,
            )?;
            let bottom_outer_next = point_index(
                0,
                radial_index + 1,
                next_angular,
                counts.points_per_layer,
                config.angular_cells,
            )?;
            let bottom_inner_next = point_index(
                0,
                radial_index,
                next_angular,
                counts.points_per_layer,
                config.angular_cells,
            )?;

            validate_positive_cell(
                &points,
                [
                    bottom_inner,
                    bottom_outer,
                    bottom_outer_next,
                    bottom_inner_next,
                ],
                config.depth,
            )?;

            let top_inner = bottom_inner
                .checked_add(counts.points_per_layer)
                .ok_or(MeshError::OutOfMemory)?;
            let top_outer = bottom_outer
                .checked_add(counts.points_per_layer)
                .ok_or(MeshError::OutOfMemory)?;
            let top_outer_next = bottom_outer_next
                .checked_add(counts.points_per_layer)
                .ok_or(MeshError::OutOfMemory)?;
            let top_inner_next = bottom_inner_next
                .checked_add(counts.points_per_layer)
                .ok_or(MeshError::OutOfMemory)?;

            let source_id = counts
                .boundary_face_count
                .checked_add(cells.len())
                .and_then(|index| index.checked_add(1))
                .ok_or(MeshError::OutOfMemory)?;
            cells.push(Cell {
                source_id,
                physical_tag: FLUID_TAG,
                nodes: fallible_nodes([
                    bottom_inner,
                    bottom_outer,
                    bottom_outer_next,
                    bottom_inner_next,
                    top_inner,
                    top_outer,
                    top_outer_next,
                    top_inner_next,
                ])?,
            });
        }
    }

    let mut boundary_faces = Vec::new();
    boundary_faces
        .try_reserve_exact(counts.boundary_face_count)
        .map_err(|_| MeshError::OutOfMemory)?;
    append_outer_patch_faces(&mut boundary_faces, config, counts, OuterPatch::Inlet)?;
    append_outer_patch_faces(&mut boundary_faces, config, counts, OuterPatch::Outlet)?;
    append_cylinder_faces(&mut boundary_faces, config, counts)?;
    append_extrusion_faces(&mut boundary_faces, config, counts)?;

    if boundary_faces.len() != counts.boundary_face_count {
        return Err(invalid(
            "cylinder O-grid boundary-face count violated its checked plan",
        ));
    }

    Ok(Mesh {
        points,
        cells,
        boundary_faces,
        physical_names: physical_names()?,
        unsupported_elements: Vec::new(),
    })
}

#[derive(Clone, Copy)]
struct MeshCounts {
    ring_count: usize,
    points_per_layer: usize,
    point_count: usize,
    cell_count: usize,
    boundary_face_count: usize,
}

fn validate_config(config: &CylinderOGridConfig) -> Result<MeshCounts> {
    if config.angular_cells < 8 || !config.angular_cells.is_multiple_of(8) {
        return Err(invalid(
            "cylinder O-grid angular_cells must be at least 8 and divisible by 8",
        ));
    }
    if config.radial_cells == 0 {
        return Err(invalid(
            "cylinder O-grid radial_cells must be greater than zero",
        ));
    }
    validate_positive_finite(config.diameter, "diameter")?;
    validate_positive_finite(config.depth, "depth")?;
    for (value, name) in [
        (config.x_min, "x_min"),
        (config.x_max, "x_max"),
        (config.y_min, "y_min"),
        (config.y_max, "y_max"),
    ] {
        if !value.is_finite() {
            return Err(invalid(format!("cylinder O-grid {name} must be finite")));
        }
    }

    let radius = config.diameter * 0.5;
    let half_depth = config.depth * 0.5;
    if !radius.is_finite() || radius <= 0.0 {
        return Err(invalid(
            "cylinder O-grid derived radius must be finite and greater than zero",
        ));
    }
    if !half_depth.is_finite() || half_depth <= 0.0 {
        return Err(invalid(
            "cylinder O-grid derived half-depth must be finite and greater than zero",
        ));
    }
    if config.x_min >= -radius
        || config.x_max <= radius
        || config.y_min >= -radius
        || config.y_max <= radius
    {
        return Err(invalid(
            "cylinder O-grid domain must strictly enclose the centered cylinder",
        ));
    }
    if let CylinderRadialGrading::Exponential { ratio } = config.radial_grading {
        validate_positive_finite(ratio, "exponential grading ratio")?;
    }

    let ring_count = config
        .radial_cells
        .checked_add(1)
        .ok_or(MeshError::OutOfMemory)?;
    let points_per_layer = ring_count
        .checked_mul(config.angular_cells)
        .ok_or(MeshError::OutOfMemory)?;
    let point_count = points_per_layer
        .checked_mul(2)
        .ok_or(MeshError::OutOfMemory)?;
    let cell_count = config
        .radial_cells
        .checked_mul(config.angular_cells)
        .ok_or(MeshError::OutOfMemory)?;
    let extrusion_faces = cell_count.checked_mul(2).ok_or(MeshError::OutOfMemory)?;
    let boundary_face_count = config
        .angular_cells
        .checked_mul(2)
        .and_then(|side_faces| side_faces.checked_add(extrusion_faces))
        .ok_or(MeshError::OutOfMemory)?;
    boundary_face_count
        .checked_add(cell_count)
        .ok_or(MeshError::OutOfMemory)?;

    Ok(MeshCounts {
        ring_count,
        points_per_layer,
        point_count,
        cell_count,
        boundary_face_count,
    })
}

fn validate_positive_finite(value: f64, name: &str) -> Result<()> {
    if !value.is_finite() || value <= 0.0 {
        return Err(invalid(format!(
            "cylinder O-grid {name} must be finite and greater than zero"
        )));
    }
    Ok(())
}

fn grading_fraction(grading: CylinderRadialGrading, fraction: f64) -> Result<f64> {
    let value = match grading {
        CylinderRadialGrading::Linear => fraction,
        CylinderRadialGrading::Exponential { ratio } => {
            if ratio == 1.0 {
                fraction
            } else {
                let logarithm = ratio.ln();
                (fraction * logarithm).exp_m1() / logarithm.exp_m1()
            }
        }
    };
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(invalid(
            "cylinder O-grid radial grading produced an invalid fraction",
        ));
    }
    Ok(value)
}

fn radial_fraction(index: usize, count: usize) -> f64 {
    let divisor = greatest_common_divisor(index, count);
    (index / divisor) as f64 / (count / divisor) as f64
}

fn angular_fraction(index: usize, count: usize) -> f64 {
    let divisor = greatest_common_divisor(index, count);
    (index / divisor) as f64 / (count / divisor) as f64
}

fn greatest_common_divisor(mut left: usize, mut right: usize) -> usize {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left.max(1)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OuterSide {
    Left,
    Right,
    Top,
    Bottom,
}

struct BoundaryRay {
    x: f64,
    y: f64,
    outer_x: f64,
    outer_y: f64,
    side: OuterSide,
}

fn boundary_ray(config: &CylinderOGridConfig, angular_index: usize) -> BoundaryRay {
    let (x, y) = unit_direction(angular_index, config.angular_cells);
    let (x_scale, x_boundary, x_side) = if x > 0.0 {
        (config.x_max / x, config.x_max, OuterSide::Right)
    } else if x < 0.0 {
        (config.x_min / x, config.x_min, OuterSide::Left)
    } else {
        (f64::INFINITY, 0.0, OuterSide::Right)
    };
    let (y_scale, y_boundary, y_side) = if y > 0.0 {
        (config.y_max / y, config.y_max, OuterSide::Top)
    } else if y < 0.0 {
        (config.y_min / y, config.y_min, OuterSide::Bottom)
    } else {
        (f64::INFINITY, 0.0, OuterSide::Top)
    };

    if x_scale <= y_scale {
        BoundaryRay {
            x,
            y,
            outer_x: x_boundary,
            outer_y: normalize_zero(if x_scale == y_scale {
                y_boundary
            } else {
                y * x_scale
            }),
            side: x_side,
        }
    } else {
        BoundaryRay {
            x,
            y,
            outer_x: normalize_zero(x * y_scale),
            outer_y: y_boundary,
            side: y_side,
        }
    }
}

fn unit_direction(index: usize, count: usize) -> (f64, f64) {
    let octant_stride = count / 8;
    if index.is_multiple_of(octant_stride) {
        return match index / octant_stride {
            0 => (1.0, 0.0),
            1 => (FRAC_1_SQRT_2, FRAC_1_SQRT_2),
            2 => (0.0, 1.0),
            3 => (-FRAC_1_SQRT_2, FRAC_1_SQRT_2),
            4 => (-1.0, 0.0),
            5 => (-FRAC_1_SQRT_2, -FRAC_1_SQRT_2),
            6 => (0.0, -1.0),
            7 => (FRAC_1_SQRT_2, -FRAC_1_SQRT_2),
            _ => unreachable!("angular index is strictly less than angular cell count"),
        };
    }

    let angle = TAU * angular_fraction(index, count);
    (normalize_zero(angle.cos()), normalize_zero(angle.sin()))
}

fn normalize_zero(value: f64) -> f64 {
    if value == 0.0 { 0.0 } else { value }
}

fn point_index(
    layer: usize,
    radial: usize,
    angular: usize,
    points_per_layer: usize,
    angular_cells: usize,
) -> Result<usize> {
    layer
        .checked_mul(points_per_layer)
        .and_then(|index| {
            radial
                .checked_mul(angular_cells)
                .and_then(|radial_offset| index.checked_add(radial_offset))
        })
        .and_then(|index| index.checked_add(angular))
        .ok_or(MeshError::OutOfMemory)
}

fn next_angular(index: usize, count: usize) -> usize {
    if index + 1 == count { 0 } else { index + 1 }
}

fn validate_positive_cell(points: &[Point3], nodes: [usize; 4], depth: f64) -> Result<()> {
    let mut twice_area = 0.0;
    for index in 0..4 {
        let current = points[nodes[index]];
        let next = points[nodes[(index + 1) % 4]];
        twice_area += current.x * next.y - next.x * current.y;
    }
    let volume = 0.5 * twice_area * depth;
    if !volume.is_finite() || volume <= 0.0 {
        return Err(invalid(
            "cylinder O-grid produced a non-positive or non-finite cell volume",
        ));
    }
    Ok(())
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum OuterPatch {
    Inlet,
    Outlet,
}

fn append_outer_patch_faces(
    faces: &mut Vec<BoundaryFace>,
    config: &CylinderOGridConfig,
    counts: MeshCounts,
    requested_patch: OuterPatch,
) -> Result<()> {
    let radial = config.radial_cells;
    for angular in 0..config.angular_cells {
        let next = next_angular(angular, config.angular_cells);
        let current_side = boundary_ray(config, angular).side;
        let next_side = boundary_ray(config, next).side;
        let patch = outer_patch(config.outer_patch_layout, current_side, next_side);
        if patch != requested_patch {
            continue;
        }
        let bottom = point_index(
            0,
            radial,
            angular,
            counts.points_per_layer,
            config.angular_cells,
        )?;
        let bottom_next = point_index(
            0,
            radial,
            next,
            counts.points_per_layer,
            config.angular_cells,
        )?;
        let top = bottom
            .checked_add(counts.points_per_layer)
            .ok_or(MeshError::OutOfMemory)?;
        let top_next = bottom_next
            .checked_add(counts.points_per_layer)
            .ok_or(MeshError::OutOfMemory)?;
        push_face(
            faces,
            match patch {
                OuterPatch::Inlet => INLET_TAG,
                OuterPatch::Outlet => OUTLET_TAG,
            },
            [bottom, bottom_next, top_next, top],
        )?;
    }
    Ok(())
}

fn outer_patch(
    layout: CylinderOuterPatchLayout,
    current: OuterSide,
    next: OuterSide,
) -> OuterPatch {
    match layout {
        CylinderOuterPatchLayout::LegacyLeftInlet => {
            if current == OuterSide::Left && next == OuterSide::Left {
                OuterPatch::Inlet
            } else {
                OuterPatch::Outlet
            }
        }
        CylinderOuterPatchLayout::RightOutlet => {
            if current == OuterSide::Right && next == OuterSide::Right {
                OuterPatch::Outlet
            } else {
                OuterPatch::Inlet
            }
        }
    }
}

fn append_cylinder_faces(
    faces: &mut Vec<BoundaryFace>,
    config: &CylinderOGridConfig,
    counts: MeshCounts,
) -> Result<()> {
    for angular in 0..config.angular_cells {
        let next = next_angular(angular, config.angular_cells);
        let bottom = point_index(0, 0, angular, counts.points_per_layer, config.angular_cells)?;
        let bottom_next = point_index(0, 0, next, counts.points_per_layer, config.angular_cells)?;
        let top = bottom
            .checked_add(counts.points_per_layer)
            .ok_or(MeshError::OutOfMemory)?;
        let top_next = bottom_next
            .checked_add(counts.points_per_layer)
            .ok_or(MeshError::OutOfMemory)?;
        push_face(faces, CYLINDER_TAG, [bottom_next, bottom, top, top_next])?;
    }
    Ok(())
}

fn append_extrusion_faces(
    faces: &mut Vec<BoundaryFace>,
    config: &CylinderOGridConfig,
    counts: MeshCounts,
) -> Result<()> {
    for layer in 0..2 {
        for radial in 0..config.radial_cells {
            for angular in 0..config.angular_cells {
                let next = next_angular(angular, config.angular_cells);
                let inner = point_index(
                    layer,
                    radial,
                    angular,
                    counts.points_per_layer,
                    config.angular_cells,
                )?;
                let inner_next = point_index(
                    layer,
                    radial,
                    next,
                    counts.points_per_layer,
                    config.angular_cells,
                )?;
                let outer = point_index(
                    layer,
                    radial + 1,
                    angular,
                    counts.points_per_layer,
                    config.angular_cells,
                )?;
                let outer_next = point_index(
                    layer,
                    radial + 1,
                    next,
                    counts.points_per_layer,
                    config.angular_cells,
                )?;
                let nodes = if layer == 0 {
                    [inner, inner_next, outer_next, outer]
                } else {
                    [inner, outer, outer_next, inner_next]
                };
                push_face(faces, FRONT_AND_BACK_TAG, nodes)?;
            }
        }
    }
    Ok(())
}

fn push_face(faces: &mut Vec<BoundaryFace>, physical_tag: i32, nodes: [usize; 4]) -> Result<()> {
    let source_id = faces.len().checked_add(1).ok_or(MeshError::OutOfMemory)?;
    faces.push(BoundaryFace {
        source_id,
        physical_tag,
        nodes: fallible_nodes(nodes)?,
    });
    Ok(())
}

fn fallible_nodes<const N: usize>(nodes: [usize; N]) -> Result<Vec<usize>> {
    let mut result = Vec::new();
    result
        .try_reserve_exact(N)
        .map_err(|_| MeshError::OutOfMemory)?;
    result.extend(nodes);
    Ok(result)
}

fn physical_names() -> Result<Vec<PhysicalName>> {
    let mut names = Vec::new();
    names
        .try_reserve_exact(5)
        .map_err(|_| MeshError::OutOfMemory)?;
    for (dim, tag, name) in [
        (2, INLET_TAG, "inlet"),
        (2, OUTLET_TAG, "outlet"),
        (2, CYLINDER_TAG, "cylinder"),
        (2, FRONT_AND_BACK_TAG, "frontAndBack"),
        (3, FLUID_TAG, "fluid"),
    ] {
        names.push(PhysicalName {
            dim,
            tag,
            name: fallible_string(name)?,
        });
    }
    Ok(names)
}

fn fallible_string(value: &str) -> Result<String> {
    let mut result = String::new();
    result
        .try_reserve_exact(value.len())
        .map_err(|_| MeshError::OutOfMemory)?;
    result.push_str(value);
    Ok(result)
}

fn invalid(message: impl Into<String>) -> MeshError {
    MeshError::InvalidInput(message.into())
}
