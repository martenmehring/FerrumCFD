use ferrum_mesh::runtime::{SolverRuntimeMeshData, SolverRuntimePatchRange};
use ferrum_mesh::{MeshError, Point3, Result};

pub const MAX_RELATIVE_AREA_VECTOR_IMBALANCE_TOLERANCE: f64 = 1.0e-6;

/// Dimensions of the pressure values passed to the force integrator.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PressureFieldKind {
    /// Pressure divided by density, with dimensions L2/T2.
    Kinematic,
    /// Dynamic pressure, with dimensions M/(L T2).
    Dynamic,
}

/// Gauge reference used before integrating pressure forces.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PressureReference {
    /// Explicit reference in the same units as the pressure field.
    Explicit(f64),
    /// Area-weighted mean over a surface whose vector-area sum is balanced.
    /// This is the algebraic condition needed for a uniform pressure offset
    /// to produce no net force; it does not prove topological closure.
    AreaVectorBalancedMean {
        relative_area_vector_imbalance_tolerance: f64,
    },
}

/// Force-coefficient reference area.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ReferenceArea {
    Explicit(f64),
    /// Projected area for an extruded two-dimensional body.
    Extruded2d {
        characteristic_length: f64,
        extrusion_depth: f64,
    },
}

/// Physical and normalization inputs for stationary no-slip walls.
#[derive(Clone, Copy, Debug)]
pub struct NoSlipWallForceOptions {
    pub pressure_kind: PressureFieldKind,
    pub pressure_reference: PressureReference,
    pub density: f64,
    pub dynamic_viscosity: f64,
    pub reference_speed: f64,
    pub reference_area: ReferenceArea,
    pub drag_direction: Point3,
    pub lift_direction: Point3,
}

#[derive(Clone, Copy, Debug)]
pub struct DirectionalCoefficient {
    pub pressure: f64,
    pub viscous: f64,
    pub total: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct WallForceCoefficients {
    pub pressure_force: Point3,
    pub viscous_force: Point3,
    pub total_force: Point3,
    pub drag: DirectionalCoefficient,
    pub lift: DirectionalCoefficient,
    pub selected_patches: usize,
    pub selected_faces: usize,
    pub selected_area: f64,
    pub area_vector_sum: Point3,
    pub resolved_pressure_reference: f64,
    pub resolved_reference_area: f64,
    pub reference_dynamic_pressure: f64,
}

/// Integrates pressure and laminar viscous forces exerted by the fluid on a
/// stationary no-slip body.
///
/// Pressure is sampled from the boundary-face owner and therefore requires
/// `zeroGradient` pressure on every selected patch. Viscous traction uses a
/// one-sided wall-normal reconstruction. For an orthogonal wall-adjacent cell
/// this is exact for a linear Couette profile. Face area vectors must point out
/// of the fluid domain, which is the orientation provided by
/// [`SolverRuntimeMeshData`].
///
/// This first API deliberately supports only stationary `wall` patches. It
/// does not silently approximate moving or non-wall boundary conditions.
pub fn integrate_stationary_no_slip_zero_gradient_pressure_wall_forces(
    mesh: &SolverRuntimeMeshData,
    cell_velocity: &[Point3],
    cell_pressure: &[f64],
    patch_names: &[&str],
    options: NoSlipWallForceOptions,
) -> Result<WallForceCoefficients> {
    let validated = validate_request(mesh, cell_velocity, cell_pressure, patch_names, &options)?;
    let pressure_scale = match options.pressure_kind {
        PressureFieldKind::Kinematic => options.density,
        PressureFieldKind::Dynamic => 1.0,
    };
    let pressure_reference = validated.pressure_reference;

    let mut pressure_force = CompensatedVec3::default();
    let mut viscous_force = CompensatedVec3::default();

    visit_selected_faces(
        mesh,
        patch_names,
        |_face, owner, area_vector, normal, distance| {
            let owner_velocity = Vec3::from(cell_velocity[owner]);
            let face_pressure = scaled_pressure_difference(
                cell_pressure[owner],
                pressure_reference,
                pressure_scale,
            );
            let face_pressure_force = area_vector * face_pressure;

            // grad(U) = (-Uowner) (x) n / dn for a stationary wall. The
            // deviatoric Newtonian traction is
            // mu * [dU/dn + n (dU/dn . n) / 3]. The force on the body is the
            // negative of the traction on the fluid boundary.
            let velocity_normal_derivative = (owner_velocity * -1.0) / distance;
            let deviatoric_traction = (velocity_normal_derivative
                + normal * (velocity_normal_derivative.dot(normal) / 3.0))
                * options.dynamic_viscosity;
            let face_viscous_force = deviatoric_traction * -area_vector.magnitude();
            pressure_force.add(face_pressure_force);
            viscous_force.add(face_viscous_force);
            Ok(())
        },
    )?;

    let pressure_force = pressure_force.total();
    let viscous_force = viscous_force.total();
    let total_force = pressure_force + viscous_force;
    if !pressure_force.is_finite() || !viscous_force.is_finite() || !total_force.is_finite() {
        return Err(invalid(
            "wall-force accumulation exceeded the finite numeric range",
        ));
    }

    let reference_dynamic_pressure =
        (0.5 * options.density * options.reference_speed) * options.reference_speed;
    if !reference_dynamic_pressure.is_finite() || reference_dynamic_pressure <= 0.0 {
        return Err(invalid(
            "wall-force reference dynamic pressure must be positive and finite",
        ));
    }
    let force_denominator = reference_dynamic_pressure * validated.reference_area;
    if !force_denominator.is_finite() || force_denominator <= 0.0 {
        return Err(invalid(
            "wall-force coefficient denominator must be positive and finite",
        ));
    }
    let drag = directional_coefficients(
        pressure_force,
        viscous_force,
        validated.drag_direction,
        force_denominator,
    )?;
    let lift = directional_coefficients(
        pressure_force,
        viscous_force,
        validated.lift_direction,
        force_denominator,
    )?;
    Ok(WallForceCoefficients {
        pressure_force: pressure_force.into(),
        viscous_force: viscous_force.into(),
        total_force: total_force.into(),
        drag,
        lift,
        selected_patches: patch_names.len(),
        selected_faces: validated.selected_faces,
        selected_area: validated.selected_area,
        area_vector_sum: validated.area_vector_sum.into(),
        resolved_pressure_reference: pressure_reference,
        resolved_reference_area: validated.reference_area,
        reference_dynamic_pressure,
    })
}

struct ValidatedRequest {
    selected_faces: usize,
    selected_area: f64,
    area_vector_sum: Vec3,
    pressure_reference: f64,
    reference_area: f64,
    drag_direction: Vec3,
    lift_direction: Vec3,
}

fn validate_request(
    mesh: &SolverRuntimeMeshData,
    cell_velocity: &[Point3],
    cell_pressure: &[f64],
    patch_names: &[&str],
    options: &NoSlipWallForceOptions,
) -> Result<ValidatedRequest> {
    validate_mesh_shape(mesh)?;
    if cell_velocity.len() != mesh.cells || cell_pressure.len() != mesh.cells {
        return Err(invalid(format!(
            "wall-force fields must have one value per cell: mesh={}, velocity={}, pressure={}",
            mesh.cells,
            cell_velocity.len(),
            cell_pressure.len()
        )));
    }
    if patch_names.is_empty() {
        return Err(invalid(
            "wall-force integration requires at least one patch",
        ));
    }
    for (index, name) in patch_names.iter().enumerate() {
        if name.is_empty() {
            return Err(invalid("wall-force patch name must not be empty"));
        }
        if patch_names[..index].contains(name) {
            return Err(invalid(format!(
                "wall-force patch '{}' is selected more than once",
                name
            )));
        }
    }
    validate_selected_patch_ranges(mesh, patch_names)?;
    for (label, value) in [
        ("density", options.density),
        ("dynamic viscosity", options.dynamic_viscosity),
        ("reference speed", options.reference_speed),
    ] {
        if !value.is_finite() || value <= 0.0 {
            return Err(invalid(format!(
                "wall-force {} must be positive and finite, got {}",
                label, value
            )));
        }
    }
    let reference_area = resolve_reference_area(options.reference_area)?;
    let drag_direction = normalized(options.drag_direction, "drag direction")?;
    let lift_direction = normalized(options.lift_direction, "lift direction")?;
    if let PressureReference::Explicit(reference) = options.pressure_reference
        && !reference.is_finite()
    {
        return Err(invalid("wall-force pressure reference must be finite"));
    }

    let mut area_sum = CompensatedSum::default();
    let mut area_vector_sum = CompensatedVec3::default();
    let mut max_abs_pressure = 0.0_f64;
    let mut selected_faces = 0usize;
    visit_selected_faces(
        mesh,
        patch_names,
        |face, owner, area_vector, normal, distance| {
            let velocity = finite_vec(cell_velocity[owner], "cell velocity")?;
            let pressure = cell_pressure[owner];
            if !pressure.is_finite() {
                return Err(invalid(format!(
                    "wall-force owner cell {} pressure is not finite",
                    owner
                )));
            }
            let _ = (face, velocity, normal, distance);
            let area = area_vector.magnitude();
            area_sum.add(area);
            area_vector_sum.add(area_vector);
            max_abs_pressure = max_abs_pressure.max(pressure.abs());
            selected_faces += 1;
            Ok(())
        },
    )?;
    let selected_area = area_sum.total();
    let area_vector_sum = area_vector_sum.total();
    if selected_faces == 0 || !selected_area.is_finite() || selected_area <= 0.0 {
        return Err(invalid(
            "wall-force selected surface has no positive finite area",
        ));
    }
    if !area_vector_sum.is_finite() || !max_abs_pressure.is_finite() {
        return Err(invalid("wall-force selected surface summary is not finite"));
    }
    if let PressureReference::AreaVectorBalancedMean {
        relative_area_vector_imbalance_tolerance,
    } = options.pressure_reference
    {
        if !relative_area_vector_imbalance_tolerance.is_finite()
            || relative_area_vector_imbalance_tolerance <= 0.0
            || relative_area_vector_imbalance_tolerance
                > MAX_RELATIVE_AREA_VECTOR_IMBALANCE_TOLERANCE
        {
            return Err(invalid(
                "wall-force relative area-vector imbalance tolerance must be positive, finite, and at most 1e-6",
            ));
        }
        let closure = area_vector_sum.magnitude();
        if closure > relative_area_vector_imbalance_tolerance * selected_area {
            return Err(invalid(format!(
                "wall-force selected surface is area-vector imbalanced: |sum(Sf)|={} exceeds {} * sum(|Sf|)={}",
                closure, relative_area_vector_imbalance_tolerance, selected_area
            )));
        }
    }

    let pressure_reference = match options.pressure_reference {
        PressureReference::Explicit(reference) => reference,
        PressureReference::AreaVectorBalancedMean { .. } if max_abs_pressure == 0.0 => 0.0,
        PressureReference::AreaVectorBalancedMean { .. } => {
            let mut scaled_pressure_area_sum = CompensatedSum::default();
            visit_selected_faces(
                mesh,
                patch_names,
                |_face, owner, area_vector, _normal, _distance| {
                    scaled_pressure_area_sum
                        .add((cell_pressure[owner] / max_abs_pressure) * area_vector.magnitude());
                    Ok(())
                },
            )?;
            (scaled_pressure_area_sum.total() / selected_area) * max_abs_pressure
        }
    };
    if !pressure_reference.is_finite() {
        return Err(invalid(
            "wall-force resolved pressure reference is not finite",
        ));
    }
    let pressure_scale = match options.pressure_kind {
        PressureFieldKind::Kinematic => options.density,
        PressureFieldKind::Dynamic => 1.0,
    };
    visit_selected_faces(
        mesh,
        patch_names,
        |face, owner, area_vector, normal, distance| {
            let velocity = Vec3::from(cell_velocity[owner]);
            let trial_pressure = scaled_pressure_difference(
                cell_pressure[owner],
                pressure_reference,
                pressure_scale,
            );
            let trial_derivative = (velocity * -1.0) / distance;
            let trial_traction = (trial_derivative + normal * (trial_derivative.dot(normal) / 3.0))
                * options.dynamic_viscosity;
            let pressure_force = area_vector * trial_pressure;
            let viscous_force = trial_traction * -area_vector.magnitude();
            if !trial_pressure.is_finite()
                || !pressure_force.is_finite()
                || !viscous_force.is_finite()
            {
                return Err(invalid(format!(
                    "wall-force face {} contribution exceeds the finite numeric range",
                    face
                )));
            }
            Ok(())
        },
    )?;

    Ok(ValidatedRequest {
        selected_faces,
        selected_area,
        area_vector_sum,
        pressure_reference,
        reference_area,
        drag_direction,
        lift_direction,
    })
}

fn validate_mesh_shape(mesh: &SolverRuntimeMeshData) -> Result<()> {
    if mesh.internal_faces > mesh.faces
        || mesh.boundary_faces != mesh.faces - mesh.internal_faces
        || mesh.owner.len() != mesh.faces
        || mesh.neighbour.len() != mesh.faces
        || mesh.face_centres.len() != mesh.faces
        || mesh.face_area_vectors.len() != mesh.faces
        || mesh.cell_centres.len() != mesh.cells
    {
        return Err(invalid(format!(
            "wall-force mesh shape is inconsistent: cells={}, faces={}, internal={}, boundary={}, owner={}, neighbour={}, faceCentres={}, faceAreas={}, cellCentres={}",
            mesh.cells,
            mesh.faces,
            mesh.internal_faces,
            mesh.boundary_faces,
            mesh.owner.len(),
            mesh.neighbour.len(),
            mesh.face_centres.len(),
            mesh.face_area_vectors.len(),
            mesh.cell_centres.len()
        )));
    }
    Ok(())
}

fn validate_selected_patch_ranges(
    mesh: &SolverRuntimeMeshData,
    patch_names: &[&str],
) -> Result<()> {
    for (index, name) in patch_names.iter().enumerate() {
        let patch = unique_patch(mesh, name)?;
        validate_patch(mesh, patch)?;
        let end = patch.start_face + patch.faces;
        for earlier_name in &patch_names[..index] {
            let earlier = unique_patch(mesh, earlier_name)?;
            let earlier_end = earlier.start_face + earlier.faces;
            if patch.start_face < earlier_end && earlier.start_face < end {
                return Err(invalid(format!(
                    "wall-force patches '{}' and '{}' have overlapping face ranges",
                    earlier.name, patch.name
                )));
            }
        }
    }
    Ok(())
}

fn visit_selected_faces(
    mesh: &SolverRuntimeMeshData,
    patch_names: &[&str],
    mut visitor: impl FnMut(usize, usize, Vec3, Vec3, f64) -> Result<()>,
) -> Result<()> {
    for name in patch_names {
        let patch = unique_patch(mesh, name)?;
        validate_patch(mesh, patch)?;
        for face in patch.start_face..patch.start_face + patch.faces {
            if mesh.neighbour.get(face).is_some_and(Option::is_some) {
                return Err(invalid(format!(
                    "wall-force patch '{}' includes internal face {}",
                    patch.name, face
                )));
            }
            let owner = *mesh
                .owner
                .get(face)
                .ok_or_else(|| invalid(format!("wall-force face {} has no owner", face)))?;
            if owner >= mesh.cells {
                return Err(invalid(format!(
                    "wall-force face {} owner {} is outside {} cells",
                    face, owner, mesh.cells
                )));
            }
            let face_centre = finite_vec(
                *mesh
                    .face_centres
                    .get(face)
                    .ok_or_else(|| invalid(format!("wall-force face {} has no centre", face)))?,
                "face centre",
            )?;
            let owner_centre = finite_vec(
                *mesh.cell_centres.get(owner).ok_or_else(|| {
                    invalid(format!("wall-force owner cell {} has no centre", owner))
                })?,
                "cell centre",
            )?;
            let area_vector = finite_vec(
                *mesh.face_area_vectors.get(face).ok_or_else(|| {
                    invalid(format!("wall-force face {} has no area vector", face))
                })?,
                "face area vector",
            )?;
            let area = area_vector.magnitude();
            if !area.is_finite() || area <= 0.0 {
                return Err(invalid(format!(
                    "wall-force face {} area must be positive and finite",
                    face
                )));
            }
            let normal = area_vector / area;
            let displacement = face_centre - owner_centre;
            let distance = displacement.dot(normal);
            let coordinate_scale = displacement.max_abs_component().max(f64::MIN_POSITIVE);
            let minimum_distance = f64::EPSILON * coordinate_scale;
            if !distance.is_finite() || distance <= minimum_distance {
                return Err(invalid(format!(
                    "wall-force face {} area vector is reversed or its owner-to-wall normal distance is degenerate",
                    face
                )));
            }
            visitor(face, owner, area_vector, normal, distance)?;
        }
    }
    Ok(())
}

fn unique_patch<'a>(
    mesh: &'a SolverRuntimeMeshData,
    name: &str,
) -> Result<&'a SolverRuntimePatchRange> {
    let mut matches = mesh.patches.iter().filter(|patch| patch.name == name);
    let patch = matches
        .next()
        .ok_or_else(|| invalid(format!("wall-force patch '{}' does not exist", name)))?;
    if matches.next().is_some() {
        return Err(invalid(format!(
            "wall-force patch name '{}' is ambiguous in the runtime mesh",
            name
        )));
    }
    Ok(patch)
}

fn validate_patch(mesh: &SolverRuntimeMeshData, patch: &SolverRuntimePatchRange) -> Result<()> {
    if patch.patch_type != "wall" {
        return Err(invalid(format!(
            "wall-force patch '{}' has mesh type '{}', expected 'wall'",
            patch.name, patch.patch_type
        )));
    }
    let end_face = patch.start_face.checked_add(patch.faces).ok_or_else(|| {
        invalid(format!(
            "wall-force patch '{}' face range overflows",
            patch.name
        ))
    })?;
    if patch.start_face < mesh.internal_faces || end_face > mesh.faces {
        return Err(invalid(format!(
            "wall-force patch '{}' face range {}..{} is outside boundary faces {}..{}",
            patch.name, patch.start_face, end_face, mesh.internal_faces, mesh.faces
        )));
    }
    Ok(())
}

fn resolve_reference_area(reference_area: ReferenceArea) -> Result<f64> {
    let area = match reference_area {
        ReferenceArea::Explicit(area) => area,
        ReferenceArea::Extruded2d {
            characteristic_length,
            extrusion_depth,
        } => {
            if !characteristic_length.is_finite() || characteristic_length <= 0.0 {
                return Err(invalid(
                    "wall-force characteristic length must be positive and finite",
                ));
            }
            if !extrusion_depth.is_finite() || extrusion_depth <= 0.0 {
                return Err(invalid(
                    "wall-force extrusion depth must be positive and finite",
                ));
            }
            characteristic_length * extrusion_depth
        }
    };
    if !area.is_finite() || area <= 0.0 {
        return Err(invalid(
            "wall-force reference area must be positive and finite",
        ));
    }
    Ok(area)
}

fn directional_coefficients(
    pressure: Vec3,
    viscous: Vec3,
    direction: Vec3,
    denominator: f64,
) -> Result<DirectionalCoefficient> {
    let pressure = pressure.dot(direction) / denominator;
    let viscous = viscous.dot(direction) / denominator;
    let total = pressure + viscous;
    if !pressure.is_finite() || !viscous.is_finite() || !total.is_finite() {
        return Err(invalid("wall-force coefficient is not finite"));
    }
    Ok(DirectionalCoefficient {
        pressure,
        viscous,
        total,
    })
}

fn scaled_pressure_difference(value: f64, reference: f64, scale: f64) -> f64 {
    let difference = value - reference;
    let scaled = difference * scale;
    if scaled.is_finite() || difference.is_finite() || scale >= 1.0 {
        scaled
    } else {
        value * scale - reference * scale
    }
}

fn finite_vec(value: Point3, label: &str) -> Result<Vec3> {
    let value = Vec3::from(value);
    if !value.is_finite() {
        return Err(invalid(format!("wall-force {} must be finite", label)));
    }
    Ok(value)
}

fn normalized(value: Point3, label: &str) -> Result<Vec3> {
    let value = finite_vec(value, label)?;
    let magnitude = value.magnitude();
    if !magnitude.is_finite() || magnitude <= 0.0 {
        return Err(invalid(format!(
            "wall-force {} must have positive finite magnitude",
            label
        )));
    }
    Ok(value / magnitude)
}

fn invalid(message: impl Into<String>) -> MeshError {
    MeshError::InvalidInput(message.into())
}

#[derive(Clone, Copy, Debug, Default)]
struct CompensatedSum {
    sum: f64,
    correction: f64,
}

impl CompensatedSum {
    fn add(&mut self, value: f64) {
        let next = self.sum + value;
        if self.sum.abs() >= value.abs() {
            self.correction += (self.sum - next) + value;
        } else {
            self.correction += (value - next) + self.sum;
        }
        self.sum = next;
    }

    fn total(self) -> f64 {
        self.sum + self.correction
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct CompensatedVec3 {
    x: CompensatedSum,
    y: CompensatedSum,
    z: CompensatedSum,
}

impl CompensatedVec3 {
    fn add(&mut self, value: Vec3) {
        self.x.add(value.x);
        self.y.add(value.y);
        self.z.add(value.z);
    }

    fn total(self) -> Vec3 {
        Vec3 {
            x: self.x.total(),
            y: self.y.total(),
            z: self.z.total(),
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct Vec3 {
    x: f64,
    y: f64,
    z: f64,
}

impl Vec3 {
    fn dot(self, other: Self) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    fn magnitude(self) -> f64 {
        let scale = self.max_abs_component();
        if scale == 0.0 {
            0.0
        } else {
            let scaled = self / scale;
            scale * scaled.dot(scaled).sqrt()
        }
    }

    fn max_abs_component(self) -> f64 {
        self.x.abs().max(self.y.abs()).max(self.z.abs())
    }

    fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
    }
}

impl From<Point3> for Vec3 {
    fn from(value: Point3) -> Self {
        Self {
            x: value.x,
            y: value.y,
            z: value.z,
        }
    }
}

impl From<Vec3> for Point3 {
    fn from(value: Vec3) -> Self {
        Self {
            x: value.x,
            y: value.y,
            z: value.z,
        }
    }
}

impl std::ops::Add for Vec3 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

impl std::ops::Sub for Vec3 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

impl std::ops::Mul<f64> for Vec3 {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self::Output {
        Self {
            x: self.x * rhs,
            y: self.y * rhs,
            z: self.z * rhs,
        }
    }
}

impl std::ops::Div<f64> for Vec3 {
    type Output = Self;

    fn div(self, rhs: f64) -> Self::Output {
        Self {
            x: self.x / rhs,
            y: self.y / rhs,
            z: self.z / rhs,
        }
    }
}

#[cfg(test)]
mod tests {
    use ferrum_mesh::Point3;
    use ferrum_mesh::geometry::compute_poly_mesh_geometry;
    use ferrum_mesh::poly_mesh::PolyMesh;
    use ferrum_mesh::runtime::{SolverRuntimeMeshData, SolverRuntimePatchRange};

    use super::{
        NoSlipWallForceOptions, PressureFieldKind, PressureReference, ReferenceArea,
        integrate_stationary_no_slip_zero_gradient_pressure_wall_forces as integrate_stationary_no_slip_wall_forces,
    };

    fn point(x: f64, y: f64, z: f64) -> Point3 {
        Point3 { x, y, z }
    }

    fn options(pressure_kind: PressureFieldKind) -> NoSlipWallForceOptions {
        NoSlipWallForceOptions {
            pressure_kind,
            pressure_reference: PressureReference::Explicit(0.0),
            density: 2.0,
            dynamic_viscosity: 0.5,
            reference_speed: 1.0,
            reference_area: ReferenceArea::Explicit(1.0),
            drag_direction: point(1.0, 0.0, 0.0),
            lift_direction: point(0.0, 1.0, 0.0),
        }
    }

    fn boundary_mesh(
        patch_type: &str,
        owners: Vec<usize>,
        face_centres: Vec<Point3>,
        face_area_vectors: Vec<Point3>,
    ) -> SolverRuntimeMeshData {
        let faces = owners.len();
        let cells = owners.iter().copied().max().map_or(0, |owner| owner + 1);
        SolverRuntimeMeshData {
            points: 8,
            cells,
            faces,
            internal_faces: 0,
            boundary_faces: faces,
            owner: owners,
            neighbour: vec![None; faces],
            patches: vec![SolverRuntimePatchRange {
                name: "body".to_string(),
                patch_type: patch_type.to_string(),
                start_face: 0,
                faces,
            }],
            face_centres,
            face_area_vectors,
            cell_centres: vec![point(0.0, 0.0, 0.0); cells],
            cell_volumes: vec![1.0; cells],
            min_face_area: 1.0,
            max_face_area: 1.0,
            min_cell_volume: 1.0,
            max_cell_volume: 1.0,
            total_cell_volume: cells as f64,
            non_positive_cell_volumes: 0,
        }
    }

    fn closed_box_mesh() -> SolverRuntimeMeshData {
        boundary_mesh(
            "wall",
            vec![0, 1, 2, 3, 4, 5],
            vec![
                point(0.5, 0.0, 0.0),
                point(-0.5, 0.0, 0.0),
                point(0.0, 0.5, 0.0),
                point(0.0, -0.5, 0.0),
                point(0.0, 0.0, 0.5),
                point(0.0, 0.0, -0.5),
            ],
            vec![
                point(1.0, 0.0, 0.0),
                point(-1.0, 0.0, 0.0),
                point(0.0, 1.0, 0.0),
                point(0.0, -1.0, 0.0),
                point(0.0, 0.0, 1.0),
                point(0.0, 0.0, -1.0),
            ],
        )
    }

    fn packaged_cylinder_mesh() -> SolverRuntimeMeshData {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let poly_mesh = PolyMesh::read(
            &root.join("tutorials/incompressibleFluid/cylinder/ferrum/case/constant/polyMesh"),
        )
        .expect("packaged cylinder polyMesh");
        let geometry = compute_poly_mesh_geometry(&poly_mesh).expect("packaged cylinder geometry");
        let cells = poly_mesh.cell_count();
        let faces = poly_mesh.faces.len();
        let internal_faces = poly_mesh.neighbour.len();
        let mut neighbour = poly_mesh
            .neighbour
            .iter()
            .copied()
            .map(Some)
            .collect::<Vec<_>>();
        neighbour.resize(faces, None);
        SolverRuntimeMeshData {
            points: poly_mesh.points.len(),
            cells,
            faces,
            internal_faces,
            boundary_faces: faces - internal_faces,
            owner: poly_mesh.owner.clone(),
            neighbour,
            patches: poly_mesh
                .patches
                .iter()
                .map(|patch| SolverRuntimePatchRange {
                    name: patch.name.clone(),
                    patch_type: patch.patch_type.clone(),
                    start_face: patch.start_face,
                    faces: patch.faces,
                })
                .collect(),
            face_centres: geometry.face_centres,
            face_area_vectors: geometry.face_area_vectors,
            cell_centres: geometry.cell_centres,
            cell_volumes: geometry.cell_volumes,
            min_face_area: 0.0,
            max_face_area: 0.0,
            min_cell_volume: 0.0,
            max_cell_volume: 0.0,
            total_cell_volume: 0.0,
            non_positive_cell_volumes: geometry.non_positive_cell_volumes,
        }
    }

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() <= 1.0e-12,
            "expected {expected:.16e}, got {actual:.16e}"
        );
    }

    #[test]
    fn area_vector_balanced_pressure_force_is_gauge_invariant() {
        let mesh = closed_box_mesh();
        let velocity = vec![point(0.0, 0.0, 0.0); 6];
        let pressure = [5.0, 2.0, 4.0, 4.0, 3.0, 3.0];
        let shifted = pressure.map(|value| value + 17.0);
        let mut request = options(PressureFieldKind::Kinematic);
        request.pressure_reference = PressureReference::AreaVectorBalancedMean {
            relative_area_vector_imbalance_tolerance: 1.0e-12,
        };
        let baseline = integrate_stationary_no_slip_wall_forces(
            &mesh,
            &velocity,
            &pressure,
            &["body"],
            request,
        )
        .expect("baseline closed force");
        let shifted = integrate_stationary_no_slip_wall_forces(
            &mesh,
            &velocity,
            &shifted,
            &["body"],
            request,
        )
        .expect("shifted closed force");

        assert_close(baseline.pressure_force.x, shifted.pressure_force.x);
        assert_close(baseline.pressure_force.y, shifted.pressure_force.y);
        assert_close(baseline.drag.pressure, shifted.drag.pressure);
        assert_close(baseline.lift.pressure, shifted.lift.pressure);
    }

    #[test]
    fn closed_constant_pressure_patch_has_zero_net_force() {
        let mesh = closed_box_mesh();
        let velocity = vec![point(0.0, 0.0, 0.0); 6];
        let mut request = options(PressureFieldKind::Dynamic);
        request.pressure_reference = PressureReference::AreaVectorBalancedMean {
            relative_area_vector_imbalance_tolerance: 1.0e-12,
        };
        let report = integrate_stationary_no_slip_wall_forces(
            &mesh,
            &velocity,
            &[3.0; 6],
            &["body"],
            request,
        )
        .expect("closed constant pressure");
        assert_close(report.total_force.x, 0.0);
        assert_close(report.total_force.y, 0.0);
        assert_close(report.total_force.z, 0.0);
    }

    #[test]
    fn large_finite_constant_gauge_does_not_overflow_area_mean() {
        let mesh = closed_box_mesh();
        let velocity = vec![point(0.0, 0.0, 0.0); 6];
        let mut request = options(PressureFieldKind::Kinematic);
        request.pressure_reference = PressureReference::AreaVectorBalancedMean {
            relative_area_vector_imbalance_tolerance: 1.0e-12,
        };
        let pressure = f64::MAX / 4.0;
        let report = integrate_stationary_no_slip_wall_forces(
            &mesh,
            &velocity,
            &[pressure; 6],
            &["body"],
            request,
        )
        .expect("large finite gauge");
        assert_eq!(
            report.resolved_pressure_reference.to_bits(),
            pressure.to_bits()
        );
        assert_close(report.total_force.x, 0.0);
        assert_close(report.total_force.y, 0.0);
        assert_close(report.total_force.z, 0.0);
    }

    #[test]
    fn kinematic_pressure_force_scales_with_density_but_cd_does_not() {
        let mesh = boundary_mesh(
            "wall",
            vec![0],
            vec![point(1.0, 0.0, 0.0)],
            vec![point(1.0, 0.0, 0.0)],
        );
        let mut first = options(PressureFieldKind::Kinematic);
        first.density = 1.0;
        let mut second = first;
        second.density = 2.0;
        let first = integrate_stationary_no_slip_wall_forces(
            &mesh,
            &[point(0.0, 0.0, 0.0)],
            &[5.0],
            &["body"],
            first,
        )
        .expect("first density");
        let second = integrate_stationary_no_slip_wall_forces(
            &mesh,
            &[point(0.0, 0.0, 0.0)],
            &[5.0],
            &["body"],
            second,
        )
        .expect("second density");
        assert_close(second.pressure_force.x, 2.0 * first.pressure_force.x);
        assert_close(second.drag.pressure, first.drag.pressure);
    }

    #[test]
    fn dynamic_pressure_is_not_scaled_twice_by_density() {
        let mesh = boundary_mesh(
            "wall",
            vec![0],
            vec![point(1.0, 0.0, 0.0)],
            vec![point(1.0, 0.0, 0.0)],
        );
        let report = integrate_stationary_no_slip_wall_forces(
            &mesh,
            &[point(0.0, 0.0, 0.0)],
            &[5.0],
            &["body"],
            options(PressureFieldKind::Dynamic),
        )
        .expect("dynamic pressure");
        assert_close(report.pressure_force.x, 5.0);
        assert_close(report.drag.pressure, 5.0);
    }

    #[test]
    fn stationary_couette_wall_matches_analytic_viscous_traction() {
        let mesh = boundary_mesh(
            "wall",
            vec![0],
            vec![point(0.0, 1.0, 0.0)],
            vec![point(0.0, 2.0, 0.0)],
        );
        let report = integrate_stationary_no_slip_wall_forces(
            &mesh,
            &[point(3.0, 0.0, 0.0)],
            &[0.0],
            &["body"],
            options(PressureFieldKind::Dynamic),
        )
        .expect("Couette traction");
        assert_close(report.viscous_force.x, 3.0);
        assert_close(report.drag.viscous, 3.0);
    }

    #[test]
    fn symmetric_cylinder_patch_has_zero_lift_and_componentwise_drag() {
        let mesh = boundary_mesh(
            "wall",
            vec![0, 1, 2, 3],
            vec![
                point(1.0, 0.0, 0.0),
                point(-1.0, 0.0, 0.0),
                point(0.0, 1.0, 0.0),
                point(0.0, -1.0, 0.0),
            ],
            vec![
                point(1.0, 0.0, 0.0),
                point(-1.0, 0.0, 0.0),
                point(0.0, 1.0, 0.0),
                point(0.0, -1.0, 0.0),
            ],
        );
        let report = integrate_stationary_no_slip_wall_forces(
            &mesh,
            &[
                point(0.0, 1.0, 0.0),
                point(0.0, -1.0, 0.0),
                point(1.0, 0.0, 0.0),
                point(1.0, 0.0, 0.0),
            ],
            &[5.0, 2.0, 3.0, 3.0],
            &["body"],
            options(PressureFieldKind::Dynamic),
        )
        .expect("symmetric body");
        assert_close(report.lift.pressure, 0.0);
        assert_close(report.lift.viscous, 0.0);
        assert_close(report.lift.total, 0.0);
        assert!(report.drag.pressure > 0.0);
    }

    #[test]
    fn extruded_depth_scales_force_and_reference_area_not_coefficients() {
        let base_mesh = boundary_mesh(
            "wall",
            vec![0],
            vec![point(1.0, 0.0, 0.0)],
            vec![point(1.0, 0.0, 0.0)],
        );
        let mut deep_mesh = base_mesh.clone();
        deep_mesh.face_area_vectors[0] = point(3.0, 0.0, 0.0);
        let mut base_options = options(PressureFieldKind::Dynamic);
        base_options.reference_area = ReferenceArea::Extruded2d {
            characteristic_length: 1.0,
            extrusion_depth: 1.0,
        };
        let mut deep_options = base_options;
        deep_options.reference_area = ReferenceArea::Extruded2d {
            characteristic_length: 1.0,
            extrusion_depth: 3.0,
        };
        let base = integrate_stationary_no_slip_wall_forces(
            &base_mesh,
            &[point(0.0, 0.0, 0.0)],
            &[4.0],
            &["body"],
            base_options,
        )
        .expect("base depth");
        let deep = integrate_stationary_no_slip_wall_forces(
            &deep_mesh,
            &[point(0.0, 0.0, 0.0)],
            &[4.0],
            &["body"],
            deep_options,
        )
        .expect("deep extrusion");
        assert_close(deep.total_force.x, 3.0 * base.total_force.x);
        assert_close(deep.drag.total, base.drag.total);
    }

    #[test]
    fn balanced_reference_rejects_imbalanced_or_malformed_patch_set() {
        let mut mesh = boundary_mesh(
            "wall",
            vec![0],
            vec![point(1.0, 0.0, 0.0)],
            vec![point(1.0, 0.0, 0.0)],
        );
        let mut request = options(PressureFieldKind::Dynamic);
        request.pressure_reference = PressureReference::AreaVectorBalancedMean {
            relative_area_vector_imbalance_tolerance: 1.0e-12,
        };
        let error = integrate_stationary_no_slip_wall_forces(
            &mesh,
            &[point(0.0, 0.0, 0.0)],
            &[1.0],
            &["body"],
            request,
        )
        .expect_err("open patch must fail");
        assert!(error.to_string().contains("area-vector imbalanced"));

        request.pressure_reference = PressureReference::AreaVectorBalancedMean {
            relative_area_vector_imbalance_tolerance: 1.0,
        };
        let error = integrate_stationary_no_slip_wall_forces(
            &mesh,
            &[point(0.0, 0.0, 0.0)],
            &[1.0],
            &["body"],
            request,
        )
        .expect_err("loose closure tolerance must fail");
        assert!(error.to_string().contains("at most 1e-6"));

        let error = integrate_stationary_no_slip_wall_forces(
            &mesh,
            &[point(0.0, 0.0, 0.0)],
            &[1.0],
            &["body", "body"],
            options(PressureFieldKind::Dynamic),
        )
        .expect_err("duplicate patch must fail");
        assert!(error.to_string().contains("selected more than once"));

        mesh.patches.push(SolverRuntimePatchRange {
            name: "body2".to_string(),
            patch_type: "wall".to_string(),
            start_face: 0,
            faces: 1,
        });
        let error = integrate_stationary_no_slip_wall_forces(
            &mesh,
            &[point(0.0, 0.0, 0.0)],
            &[1.0],
            &["body", "body2"],
            options(PressureFieldKind::Dynamic),
        )
        .expect_err("overlapping patch ranges must fail");
        assert!(error.to_string().contains("overlapping face ranges"));
        mesh.patches.pop();

        mesh.patches.push(mesh.patches[0].clone());
        let error = integrate_stationary_no_slip_wall_forces(
            &mesh,
            &[point(0.0, 0.0, 0.0)],
            &[1.0],
            &["body"],
            options(PressureFieldKind::Dynamic),
        )
        .expect_err("duplicate runtime patch names must fail");
        assert!(error.to_string().contains("ambiguous"));
        mesh.patches.pop();

        mesh.patches[0].patch_type = "patch".to_string();
        let error = integrate_stationary_no_slip_wall_forces(
            &mesh,
            &[point(0.0, 0.0, 0.0)],
            &[1.0],
            &["body"],
            options(PressureFieldKind::Dynamic),
        )
        .expect_err("non-wall patch must fail");
        assert!(error.to_string().contains("expected 'wall'"));
    }

    #[test]
    fn invalid_inputs_fail_before_force_accumulation() {
        let mut mesh = boundary_mesh(
            "wall",
            vec![0],
            vec![point(1.0, 0.0, 0.0)],
            vec![point(1.0, 0.0, 0.0)],
        );
        let error = integrate_stationary_no_slip_wall_forces(
            &mesh,
            &[],
            &[1.0],
            &["body"],
            options(PressureFieldKind::Dynamic),
        )
        .expect_err("wrong field length must fail");
        assert!(error.to_string().contains("one value per cell"));

        let mut malformed_mesh = mesh.clone();
        malformed_mesh.neighbour.clear();
        let error = integrate_stationary_no_slip_wall_forces(
            &malformed_mesh,
            &[point(0.0, 0.0, 0.0)],
            &[1.0],
            &["body"],
            options(PressureFieldKind::Dynamic),
        )
        .expect_err("malformed mesh shape must fail");
        assert!(error.to_string().contains("mesh shape is inconsistent"));

        let mut invalid_options = options(PressureFieldKind::Dynamic);
        invalid_options.density = f64::NAN;
        let error = integrate_stationary_no_slip_wall_forces(
            &mesh,
            &[point(0.0, 0.0, 0.0)],
            &[1.0],
            &["body"],
            invalid_options,
        )
        .expect_err("non-finite density must fail");
        assert!(error.to_string().contains("density"));

        let mut overflow_options = options(PressureFieldKind::Dynamic);
        overflow_options.reference_speed = 1.0e308;
        let error = integrate_stationary_no_slip_wall_forces(
            &mesh,
            &[point(0.0, 0.0, 0.0)],
            &[1.0],
            &["body"],
            overflow_options,
        )
        .expect_err("normalization overflow must fail");
        assert!(error.to_string().contains("dynamic pressure"));

        let mut denominator_options = options(PressureFieldKind::Dynamic);
        denominator_options.reference_area = ReferenceArea::Explicit(f64::MAX);
        denominator_options.reference_speed = 2.0;
        let error = integrate_stationary_no_slip_wall_forces(
            &mesh,
            &[point(0.0, 0.0, 0.0)],
            &[1.0],
            &["body"],
            denominator_options,
        )
        .expect_err("coefficient denominator overflow must fail");
        assert!(error.to_string().contains("coefficient denominator"));

        mesh.face_area_vectors[0] = point(0.0, 0.0, 0.0);
        let error = integrate_stationary_no_slip_wall_forces(
            &mesh,
            &[point(0.0, 0.0, 0.0)],
            &[1.0],
            &["body"],
            options(PressureFieldKind::Dynamic),
        )
        .expect_err("zero area must fail");
        assert!(error.to_string().contains("area must be positive"));
    }

    #[test]
    fn reversed_boundary_area_vector_is_rejected() {
        let mesh = boundary_mesh(
            "wall",
            vec![0],
            vec![point(1.0, 0.0, 0.0)],
            vec![point(-1.0, 0.0, 0.0)],
        );
        let error = integrate_stationary_no_slip_wall_forces(
            &mesh,
            &[point(0.0, 0.0, 0.0)],
            &[1.0],
            &["body"],
            options(PressureFieldKind::Dynamic),
        )
        .expect_err("reversed face normal must fail");
        assert!(error.to_string().contains("area vector is reversed"));
    }

    #[test]
    fn packaged_cylinder_wall_has_valid_runtime_orientation_and_area_balance() {
        let mesh = packaged_cylinder_mesh();
        let mut request = options(PressureFieldKind::Kinematic);
        request.pressure_reference = PressureReference::AreaVectorBalancedMean {
            relative_area_vector_imbalance_tolerance: 1.0e-12,
        };
        request.density = 1.0;
        request.dynamic_viscosity = 1.0;
        request.reference_area = ReferenceArea::Extruded2d {
            characteristic_length: 0.001,
            extrusion_depth: 0.0001,
        };
        let report = integrate_stationary_no_slip_wall_forces(
            &mesh,
            &vec![point(0.0, 0.0, 0.0); mesh.cells],
            &vec![0.0; mesh.cells],
            &["cylinder"],
            request,
        )
        .expect("packaged cylinder wall orientation");
        assert_eq!(report.selected_faces, 16);
        assert_close(report.total_force.x, 0.0);
        assert_close(report.total_force.y, 0.0);
    }

    #[test]
    fn scale_aware_distance_accepts_finite_nanoscale_mesh() {
        let mesh = boundary_mesh(
            "wall",
            vec![0],
            vec![point(1.0e-20, 0.0, 0.0)],
            vec![point(1.0, 0.0, 0.0)],
        );
        let report = integrate_stationary_no_slip_wall_forces(
            &mesh,
            &[point(0.0, 0.0, 0.0)],
            &[1.0],
            &["body"],
            options(PressureFieldKind::Dynamic),
        )
        .expect("finite nanoscale distance");
        assert_close(report.pressure_force.x, 1.0);
    }

    #[test]
    fn stable_norm_accepts_large_area_and_extreme_finite_directions() {
        let mesh = boundary_mesh(
            "wall",
            vec![0],
            vec![point(1.0, 0.0, 0.0)],
            vec![point(1.0e200, 0.0, 0.0)],
        );
        let mut request = options(PressureFieldKind::Dynamic);
        request.reference_area = ReferenceArea::Explicit(1.0e200);
        request.drag_direction = point(1.0e200, 0.0, 0.0);
        request.lift_direction = point(0.0, 1.0e-200, 0.0);
        let report = integrate_stationary_no_slip_wall_forces(
            &mesh,
            &[point(0.0, 0.0, 0.0)],
            &[1.0],
            &["body"],
            request,
        )
        .expect("finite extreme vector scales");
        assert_eq!(report.selected_area.to_bits(), 1.0e200_f64.to_bits());
        assert_close(report.drag.total, 1.0);
        assert_close(report.lift.total, 0.0);
    }

    #[test]
    fn subnormal_wall_distance_with_zero_velocity_has_zero_viscous_force() {
        let mesh = boundary_mesh(
            "wall",
            vec![0],
            vec![point(1.0e-320, 0.0, 0.0)],
            vec![point(1.0, 0.0, 0.0)],
        );
        let report = integrate_stationary_no_slip_wall_forces(
            &mesh,
            &[point(0.0, 0.0, 0.0)],
            &[0.0],
            &["body"],
            options(PressureFieldKind::Dynamic),
        )
        .expect("subnormal distance with exact zero gradient");
        assert_close(report.viscous_force.x, 0.0);
        assert_close(report.viscous_force.y, 0.0);
        assert_close(report.viscous_force.z, 0.0);
    }

    #[test]
    fn kinematic_pressure_scaling_avoids_subtraction_overflow_when_result_is_finite() {
        let mesh = boundary_mesh(
            "wall",
            vec![0],
            vec![point(1.0, 0.0, 0.0)],
            vec![point(1.0, 0.0, 0.0)],
        );
        let mut request = options(PressureFieldKind::Kinematic);
        request.pressure_reference = PressureReference::Explicit(-f64::MAX);
        request.density = f64::MIN_POSITIVE;
        request.reference_area = ReferenceArea::Explicit(f64::MAX);
        let report = integrate_stationary_no_slip_wall_forces(
            &mesh,
            &[point(0.0, 0.0, 0.0)],
            &[f64::MAX],
            &["body"],
            request,
        )
        .expect("scaled finite kinematic pressure difference");
        assert!(report.pressure_force.x.is_finite());
        assert!(report.pressure_force.x > 0.0);
        assert!(report.drag.total.is_finite());
    }
}
