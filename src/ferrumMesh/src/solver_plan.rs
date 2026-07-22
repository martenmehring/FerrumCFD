use std::path::{Path, PathBuf};

use crate::Result;
use crate::backends::{
    BackendChoice, BackendConfig, read_backend_config, validate_backend_policy,
    validate_backend_resources,
};
use crate::control::{ControlDict, read_control_dict, validate_control_dict};
use crate::fields::{
    FieldLoadPolicy, InitialFieldSet, read_initial_fields_with_policy,
    validate_initial_field_boundaries,
};
use crate::interfaces::{read_interface_config, validate_interface_config};
use crate::numerics::{
    NumericsSection, read_fv_schemes, read_fv_solution, validate_fv_schemes, validate_fv_solution,
};
use crate::patches::{PatchValidationSummary, validate_poly_mesh_patches};
use crate::poly_mesh::PolyMesh;
use crate::properties::{PropertyDictionary, PropertySection, read_case_properties};
use crate::regions::{build_interface_registry, read_region_mesh_summaries};
use crate::runtime::{SolverRuntimeData, build_solver_runtime_data};
use crate::solver_state::{SolverStatePlan, build_solver_state_plan};

#[derive(Debug)]
pub struct SolverCasePlan {
    pub case_dir: PathBuf,
    pub control: ControlDict,
    pub mesh: SolverMeshPlan,
    pub fields: SolverFieldPlan,
    pub initial_fields: InitialFieldSet,
    pub state: SolverStatePlan,
    pub runtime_data: SolverRuntimeData,
    pub properties: SolverPropertiesPlan,
    pub numerics: SolverNumericsPlan,
    pub interfaces: SolverInterfacePlan,
    pub backends: SolverBackendPlan,
    pub run: SolverRunPlan,
    pub warnings: Vec<String>,
}

#[derive(Debug)]
pub struct SolverMeshPlan {
    pub points: usize,
    pub cells: usize,
    pub faces: usize,
    pub internal_faces: usize,
    pub boundary_faces: usize,
    pub patches: usize,
    pub empty_patches: usize,
    pub wedge_patches: usize,
    pub symmetry_patches: usize,
    pub dimensionality: SolverDimensionality,
    pub region_meshes: Vec<SolverRegionMeshPlan>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SolverDimensionality {
    ThreeD,
    TwoD,
    Axisymmetric,
    MixedSpecialPatches,
}

#[derive(Debug)]
pub struct SolverRegionMeshPlan {
    pub name: String,
    pub cells: usize,
    pub patches: usize,
}

#[derive(Debug)]
pub struct SolverFieldPlan {
    pub fields: Vec<SolverFieldEntryPlan>,
}

#[derive(Debug)]
pub struct SolverFieldEntryPlan {
    pub region: Option<String>,
    pub name: String,
    pub class_name: Option<String>,
    pub boundary_patches: usize,
}

#[derive(Debug)]
pub struct SolverPropertiesPlan {
    pub dictionaries: Vec<SolverPropertyDictionaryPlan>,
    pub entries: Vec<SolverPropertyEntryPlan>,
}

#[derive(Debug)]
pub struct SolverPropertyDictionaryPlan {
    pub name: String,
    pub region: Option<String>,
    pub sections: usize,
    pub entries: usize,
}

#[derive(Debug)]
pub struct SolverPropertyEntryPlan {
    pub dictionary: String,
    pub section: Option<String>,
    pub key: String,
    pub value: String,
}

#[derive(Debug)]
pub struct SolverNumericsPlan {
    pub fv_schemes: SolverNumericsDictionaryPlan,
    pub fv_solution: SolverNumericsDictionaryPlan,
}

#[derive(Debug)]
pub struct SolverNumericsDictionaryPlan {
    pub present: bool,
    pub sections: Vec<SolverNumericsSectionPlan>,
    pub entries: Vec<SolverNumericsEntryPlan>,
}

#[derive(Debug)]
pub struct SolverNumericsSectionPlan {
    pub path: String,
    pub entries: usize,
}

#[derive(Debug)]
pub struct SolverNumericsEntryPlan {
    pub section: String,
    pub key: String,
    pub value: String,
}

#[derive(Debug)]
pub struct SolverInterfacePlan {
    pub registry_available: bool,
    pub discovered_interfaces: usize,
    pub boundary_face_zones: usize,
    pub config_present: bool,
    pub configured_interfaces: usize,
}

#[derive(Debug)]
pub struct SolverBackendPlan {
    pub config_present: bool,
    pub default: BackendChoice,
    pub uses_cpu: bool,
    pub uses_gpu: bool,
    pub mixed_execution: bool,
    pub cpu: SolverCpuResourcePlan,
    pub gpu: SolverGpuResourcePlan,
    pub stages: Vec<SolverBackendStagePlan>,
}

#[derive(Debug)]
pub struct SolverCpuResourcePlan {
    pub cpus: String,
    pub cores_per_cpu: String,
    pub threads: String,
    pub thread_pinning: String,
    pub numa: String,
}

#[derive(Debug)]
pub struct SolverGpuResourcePlan {
    pub backend: String,
    pub devices: Vec<String>,
    pub multi_gpu: String,
    pub precision: String,
}

#[derive(Debug)]
pub struct SolverBackendStagePlan {
    pub section: String,
    pub step: String,
    pub choice: BackendChoice,
}

#[derive(Debug)]
pub struct SolverRunPlan {
    pub stop_at: String,
    pub start_time: Option<f64>,
    pub end_time: Option<f64>,
    pub delta_t: Option<f64>,
    pub estimated_steps: Option<usize>,
    pub write_control: String,
    pub write_interval: Option<f64>,
    pub estimated_write_events: Option<usize>,
    pub stages: Vec<SolverRunStagePlan>,
}

#[derive(Debug)]
pub struct SolverRunStagePlan {
    pub section: String,
    pub step: String,
    pub choice: BackendChoice,
    pub source: SolverRunStageSource,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SolverRunStageSource {
    Default,
    Configured,
}

impl std::fmt::Display for SolverDimensionality {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ThreeD => formatter.write_str("3d"),
            Self::TwoD => formatter.write_str("2d-empty"),
            Self::Axisymmetric => formatter.write_str("axisymmetric-wedge"),
            Self::MixedSpecialPatches => formatter.write_str("mixed-special-patches"),
        }
    }
}

impl std::fmt::Display for SolverRunStageSource {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Default => formatter.write_str("default"),
            Self::Configured => formatter.write_str("configured"),
        }
    }
}

const BUILT_IN_RUN_STAGES: &[(&str, &str)] = &[
    ("mesh", "checks"),
    ("interfaces", "flux"),
    ("interfaces", "coupling"),
    ("interfaces", "sourceTerms"),
    ("flow", "residual"),
    ("flow", "jacobian"),
    ("flow", "linearSolve"),
    ("flow", "pressureCorrection"),
    ("flow", "nonlinearSolve"),
    ("heat", "residual"),
    ("heat", "jacobian"),
    ("heat", "linearSolve"),
    ("heat", "nonlinearSolve"),
    ("species", "residual"),
    ("species", "jacobian"),
    ("species", "linearSolve"),
    ("species", "nonlinearSolve"),
    ("chemistry", "residual"),
    ("chemistry", "jacobian"),
    ("chemistry", "odeSolve"),
    ("chemistry", "nonlinearSolve"),
];

/// Maximum number of dictionaries, section paths, flattened entries, and
/// validation warnings constructed by the property portion of a solver plan.
pub const MAX_SOLVER_PROPERTY_PLAN_ITEMS: usize = 65_536;
/// Maximum cumulative owned string bytes constructed by the property portion
/// of a solver plan and its validation warnings.
pub const MAX_SOLVER_PROPERTY_PLAN_PAYLOAD_BYTES: usize = 64 * 1024 * 1024;
/// Maximum number of flattened sections, entries, and validation warnings
/// constructed across both numerics dictionaries in one solver plan.
pub const MAX_SOLVER_NUMERICS_PLAN_ITEMS: usize = 65_536;
/// Maximum cumulative copied bytes for flattened numerics section paths,
/// entries, and validation warnings.
pub const MAX_SOLVER_NUMERICS_PLAN_PAYLOAD_BYTES: usize = 64 * 1024 * 1024;
/// Maximum number of flattened backend stages and validation warnings retained
/// by one solver plan.
pub const MAX_SOLVER_BACKEND_PLAN_ITEMS: usize = 16_384;
/// Maximum cumulative copied bytes for backend stages and validation warnings.
pub const MAX_SOLVER_BACKEND_PLAN_PAYLOAD_BYTES: usize = 16 * 1024 * 1024;

#[derive(Clone, Copy)]
struct PropertyPlanLimits {
    items: usize,
    payload_bytes: usize,
}

const PROPERTY_PLAN_LIMITS: PropertyPlanLimits = PropertyPlanLimits {
    items: MAX_SOLVER_PROPERTY_PLAN_ITEMS,
    payload_bytes: MAX_SOLVER_PROPERTY_PLAN_PAYLOAD_BYTES,
};

struct PropertyPlanBudget {
    limits: PropertyPlanLimits,
    items: usize,
    payload_bytes: usize,
}

impl PropertyPlanBudget {
    fn new(limits: PropertyPlanLimits) -> Self {
        Self {
            limits,
            items: 0,
            payload_bytes: 0,
        }
    }

    fn add_item(&mut self, payload_bytes: usize) -> Result<()> {
        let items = self
            .items
            .checked_add(1)
            .ok_or_else(|| property_plan_limit_error("solver property plan item limit exceeded"))?;
        if items > self.limits.items {
            return Err(property_plan_limit_error(
                "solver property plan item limit exceeded",
            ));
        }
        let retained = self.checked_payload(payload_bytes)?;
        self.items = items;
        self.payload_bytes = retained;
        Ok(())
    }

    fn add_temporary_payload(&mut self, payload_bytes: usize) -> Result<()> {
        self.payload_bytes = self.checked_payload(payload_bytes)?;
        Ok(())
    }

    fn checked_payload(&self, payload_bytes: usize) -> Result<usize> {
        let retained = self
            .payload_bytes
            .checked_add(payload_bytes)
            .ok_or_else(|| {
                property_plan_limit_error("solver property plan payload limit exceeded")
            })?;
        if retained > self.limits.payload_bytes {
            return Err(property_plan_limit_error(
                "solver property plan payload limit exceeded",
            ));
        }
        Ok(retained)
    }
}

#[derive(Clone, Copy)]
struct DerivedPlanLimits {
    items: usize,
    payload_bytes: usize,
}

struct DerivedPlanBudget {
    limits: DerivedPlanLimits,
    items: usize,
    payload_bytes: usize,
    item_error: &'static str,
    payload_error: &'static str,
}

impl DerivedPlanBudget {
    fn new(
        limits: DerivedPlanLimits,
        item_error: &'static str,
        payload_error: &'static str,
    ) -> Self {
        Self {
            limits,
            items: 0,
            payload_bytes: 0,
            item_error,
            payload_error,
        }
    }

    fn add_item(&mut self, payload_bytes: usize) -> Result<()> {
        let items = self
            .items
            .checked_add(1)
            .ok_or_else(|| derived_plan_error(self.item_error))?;
        if items > self.limits.items {
            return Err(derived_plan_error(self.item_error));
        }
        let copied = self.checked_payload(payload_bytes)?;
        self.items = items;
        self.payload_bytes = copied;
        Ok(())
    }

    fn add_temporary_payload(&mut self, payload_bytes: usize) -> Result<()> {
        self.payload_bytes = self.checked_payload(payload_bytes)?;
        Ok(())
    }

    fn checked_payload(&self, payload_bytes: usize) -> Result<usize> {
        let copied = self
            .payload_bytes
            .checked_add(payload_bytes)
            .ok_or_else(|| derived_plan_error(self.payload_error))?;
        if copied > self.limits.payload_bytes {
            return Err(derived_plan_error(self.payload_error));
        }
        Ok(copied)
    }
}

/// Builds a solver-ready plan and loads initial-field payloads. Call
/// `build_solver_case_plan_with_policy(..., FieldLoadPolicy::Summary)` only
/// for inspection paths that will not execute the solver.
pub fn build_solver_case_plan(case_dir: &Path) -> Result<SolverCasePlan> {
    build_solver_case_plan_with_policy(case_dir, FieldLoadPolicy::Full)
}

pub fn build_solver_case_plan_with_policy(
    case_dir: &Path,
    field_policy: FieldLoadPolicy,
) -> Result<SolverCasePlan> {
    let control = read_control_dict(case_dir)?;
    let mut warnings = Vec::new();
    let control_validation = validate_control_dict(&control);
    warnings.extend(
        control_validation
            .warnings
            .iter()
            .map(|warning| format!("controlDict: {warning}")),
    );

    let mesh = PolyMesh::read(&case_dir.join("constant").join("polyMesh"))?;
    let patch_validation = validate_poly_mesh_patches(case_dir, &mesh);
    warnings.extend(
        patch_validation
            .warnings
            .iter()
            .map(|warning| format!("patch validation: {warning}")),
    );

    let region_meshes = match read_region_mesh_summaries(case_dir) {
        Ok(regions) => regions
            .into_iter()
            .map(|region| SolverRegionMeshPlan {
                name: region.name,
                cells: region.cells,
                patches: region.patches.len(),
            })
            .collect(),
        Err(error) => {
            warnings.push(format!("could not read region mesh summaries: {error}"));
            Vec::new()
        }
    };

    let mut fields = read_initial_fields_with_policy(case_dir, field_policy)?;
    validate_openfoam_case_structure(case_dir, &fields, &mut warnings);
    let field_validation = validate_initial_field_boundaries(case_dir, &fields)?;
    warnings.extend(
        field_validation
            .warnings
            .iter()
            .map(|warning| format!("field boundary: {warning}")),
    );
    let state = build_solver_state_plan(case_dir, &fields)?;
    warnings.extend(
        state
            .warnings
            .iter()
            .map(|warning| format!("solver state: {warning}")),
    );
    let runtime_data =
        build_solver_runtime_data(case_dir, &mesh, &state, &mut fields, field_policy)?;
    warnings.extend(
        runtime_data
            .warnings
            .iter()
            .map(|warning| format!("runtime data: {warning}")),
    );

    let field_names = unique_field_names(&fields);
    let properties = build_properties_plan(case_dir, &mut warnings)?;
    let numerics = build_numerics_plan(case_dir, &field_names, &mut warnings)?;

    let interface_config = read_interface_config(case_dir)?;
    let config_present = interface_config.is_some();
    let configured_interfaces = interface_config
        .as_ref()
        .map(|config| config.entries.len())
        .unwrap_or(0);

    let interfaces = match build_interface_registry(case_dir) {
        Ok(registry) => {
            if let Some(config) = &interface_config {
                let validation = validate_interface_config(config, &registry);
                warnings.extend(
                    validation
                        .warnings
                        .iter()
                        .map(|warning| format!("interface config: {warning}")),
                );
            }
            SolverInterfacePlan {
                registry_available: true,
                discovered_interfaces: registry.interfaces.len(),
                boundary_face_zones: registry.boundary_face_zones.len(),
                config_present,
                configured_interfaces,
            }
        }
        Err(error) => {
            warnings.push(format!("could not build interface registry: {error}"));
            SolverInterfacePlan {
                registry_available: false,
                discovered_interfaces: 0,
                boundary_face_zones: 0,
                config_present,
                configured_interfaces,
            }
        }
    };

    let backends = build_backend_plan(case_dir, &mut warnings)?;
    let run = build_run_plan(&control, &backends, &mut warnings);

    if patch_validation.empty_patches > 0 && patch_validation.wedge_patches > 0 {
        warnings.push(
            "mesh has both empty and wedge patches; solver dimensionality is ambiguous".to_string(),
        );
    }

    let field_plan = build_solver_field_plan(&fields)?;

    Ok(SolverCasePlan {
        case_dir: case_dir.to_path_buf(),
        control,
        mesh: build_mesh_plan(&mesh, &patch_validation, region_meshes),
        fields: field_plan,
        initial_fields: fields,
        state,
        runtime_data,
        properties,
        numerics,
        interfaces,
        backends,
        run,
        warnings,
    })
}

fn build_solver_field_plan(fields: &InitialFieldSet) -> Result<SolverFieldPlan> {
    let mut entries = Vec::new();
    entries
        .try_reserve_exact(fields.fields.len())
        .map_err(|_| {
            crate::MeshError::InvalidInput("solver field plan allocation failed".to_string())
        })?;
    for field in &fields.fields {
        entries.push(SolverFieldEntryPlan {
            region: try_clone_optional_string(field.region.as_deref())?,
            name: try_clone_plan_string(&field.name)?,
            class_name: try_clone_optional_string(field.class_name.as_deref())?,
            boundary_patches: field.boundary_patches.len(),
        });
    }
    Ok(SolverFieldPlan { fields: entries })
}

fn try_clone_plan_string(value: &str) -> Result<String> {
    let mut cloned = String::new();
    cloned.try_reserve_exact(value.len()).map_err(|_| {
        crate::MeshError::InvalidInput("solver plan string allocation failed".to_string())
    })?;
    cloned.push_str(value);
    Ok(cloned)
}

fn try_clone_optional_string(value: Option<&str>) -> Result<Option<String>> {
    value.map(try_clone_plan_string).transpose()
}

fn validate_openfoam_case_structure(
    case_dir: &Path,
    fields: &crate::fields::InitialFieldSet,
    warnings: &mut Vec<String>,
) {
    if !case_dir.join("system").join("controlDict").exists() {
        warnings.push("OpenFOAM compatibility: missing mandatory system/controlDict".to_string());
    }
    if !case_dir
        .join("constant")
        .join("transportProperties")
        .exists()
    {
        warnings.push(
            "OpenFOAM compatibility: missing mandatory constant/transportProperties".to_string(),
        );
    }
    if !case_dir.join("0").exists() {
        warnings.push(
            "OpenFOAM compatibility: missing mandatory time directory 0 (initial field files)"
                .to_string(),
        );
    }

    let velocity = fields
        .fields
        .iter()
        .find(|field| field.region.is_none() && field.name == "U");
    match velocity {
        Some(field) if field.class_name.as_deref() == Some("volVectorField") => {}
        Some(_) => warnings.push(
            "OpenFOAM compatibility: field 0/U exists but class is not volVectorField".to_string(),
        ),
        None => warnings.push("OpenFOAM compatibility: missing mandatory field 0/U".to_string()),
    }

    let pressure = fields
        .fields
        .iter()
        .find(|field| field.region.is_none() && field.name == "p");
    match pressure {
        Some(field) if field.class_name.as_deref() == Some("volScalarField") => {}
        Some(_) => warnings.push(
            "OpenFOAM compatibility: field 0/p exists but class is not volScalarField".to_string(),
        ),
        None => warnings.push("OpenFOAM compatibility: missing mandatory field 0/p".to_string()),
    }

    if !case_dir.join("system").join("fvSchemes").exists() {
        warnings.push(
            "OpenFOAM compatibility: mandatory system/fvSchemes is missing; laminar SIMPLE execution will reject the case".to_string(),
        );
    }
    if !case_dir.join("system").join("fvSolution").exists() {
        warnings.push(
            "OpenFOAM compatibility: mandatory system/fvSolution is missing; laminar SIMPLE execution will reject the case".to_string(),
        );
    }
    // OpenFOAM-compatible boundary and geometry files are validated by mesh and patch readers elsewhere.

    if !case_dir
        .join("constant")
        .join("polyMesh")
        .join("faces")
        .exists()
        || !case_dir
            .join("constant")
            .join("polyMesh")
            .join("owner")
            .exists()
        || !case_dir
            .join("constant")
            .join("polyMesh")
            .join("neighbour")
            .exists()
    {
        warnings.push(
            "OpenFOAM compatibility: incomplete constant/polyMesh (faces/owner/neighbour)"
                .to_string(),
        );
    }
}

fn build_mesh_plan(
    mesh: &PolyMesh,
    patch_validation: &PatchValidationSummary,
    region_meshes: Vec<SolverRegionMeshPlan>,
) -> SolverMeshPlan {
    SolverMeshPlan {
        points: mesh.points.len(),
        cells: mesh.cell_count(),
        faces: mesh.faces.len(),
        internal_faces: mesh.neighbour.len(),
        boundary_faces: mesh.faces.len().saturating_sub(mesh.neighbour.len()),
        patches: patch_validation.patches,
        empty_patches: patch_validation.empty_patches,
        wedge_patches: patch_validation.wedge_patches,
        symmetry_patches: patch_validation.symmetry_patches,
        dimensionality: classify_dimensionality(patch_validation),
        region_meshes,
    }
}

fn classify_dimensionality(patch_validation: &PatchValidationSummary) -> SolverDimensionality {
    match (
        patch_validation.empty_patches > 0,
        patch_validation.wedge_patches > 0,
    ) {
        (false, false) => SolverDimensionality::ThreeD,
        (true, false) => SolverDimensionality::TwoD,
        (false, true) => SolverDimensionality::Axisymmetric,
        (true, true) => SolverDimensionality::MixedSpecialPatches,
    }
}

fn build_properties_plan(
    case_dir: &Path,
    warnings: &mut Vec<String>,
) -> Result<SolverPropertiesPlan> {
    let dictionaries = read_case_properties(case_dir)?;
    build_properties_plan_from_dictionaries_with_limits(
        &dictionaries,
        warnings,
        PROPERTY_PLAN_LIMITS,
    )
}

fn build_properties_plan_from_dictionaries_with_limits(
    dictionaries: &[PropertyDictionary],
    warnings: &mut Vec<String>,
    limits: PropertyPlanLimits,
) -> Result<SolverPropertiesPlan> {
    let mut budget = PropertyPlanBudget::new(limits);

    let mut dictionary_plans = Vec::new();
    let mut entry_plans = Vec::new();
    for dictionary in dictionaries {
        let label_len = property_dictionary_label_len(dictionary)?;
        budget.add_temporary_payload(label_len)?;
        let label = property_dictionary_label(dictionary)?;
        let (sections, nested_entries) = count_property_sections_and_entries(&dictionary.sections)?;
        let entries = dictionary
            .entries
            .len()
            .checked_add(nested_entries)
            .ok_or_else(|| property_plan_limit_error("solver property plan item limit exceeded"))?;
        let dictionary_payload = dictionary
            .name
            .len()
            .checked_add(dictionary.region.as_ref().map_or(0, String::len))
            .ok_or_else(|| {
                property_plan_limit_error("solver property plan payload limit exceeded")
            })?;
        budget.add_item(dictionary_payload)?;
        dictionary_plans
            .try_reserve(1)
            .map_err(|_| crate::MeshError::OutOfMemory)?;
        dictionary_plans.push(SolverPropertyDictionaryPlan {
            name: try_copy_property_plan_string(&dictionary.name)?,
            region: dictionary
                .region
                .as_deref()
                .map(try_copy_property_plan_string)
                .transpose()?,
            sections,
            entries,
        });
        append_property_entries(
            &label,
            None,
            &dictionary.entries,
            &dictionary.sections,
            &mut entry_plans,
            &mut budget,
        )?;
    }

    let mut property_warnings = Vec::new();
    append_property_validation_warnings(dictionaries, &mut property_warnings, &mut budget)?;
    warnings
        .try_reserve(property_warnings.len())
        .map_err(|_| crate::MeshError::OutOfMemory)?;
    warnings.append(&mut property_warnings);

    Ok(SolverPropertiesPlan {
        dictionaries: dictionary_plans,
        entries: entry_plans,
    })
}

fn property_dictionary_label(dictionary: &PropertyDictionary) -> Result<String> {
    if let Some(region) = &dictionary.region {
        try_join_property_plan_parts(&[region, "/", &dictionary.name])
    } else {
        try_copy_property_plan_string(&dictionary.name)
    }
}

fn property_dictionary_label_len(dictionary: &PropertyDictionary) -> Result<usize> {
    match dictionary.region.as_deref() {
        Some(region) => region
            .len()
            .checked_add(1)
            .and_then(|bytes| bytes.checked_add(dictionary.name.len()))
            .ok_or_else(|| {
                property_plan_limit_error("solver property plan payload limit exceeded")
            }),
        None => Ok(dictionary.name.len()),
    }
}

fn count_property_sections_and_entries(sections: &[PropertySection]) -> Result<(usize, usize)> {
    let mut section_count = 0usize;
    let mut entry_count = 0usize;
    for section in sections {
        section_count = section_count
            .checked_add(1)
            .ok_or_else(|| property_plan_limit_error("solver property plan item limit exceeded"))?;
        entry_count = entry_count
            .checked_add(section.entries.len())
            .ok_or_else(|| property_plan_limit_error("solver property plan item limit exceeded"))?;
        let (nested_sections, nested_entries) =
            count_property_sections_and_entries(&section.sections)?;
        section_count = section_count
            .checked_add(nested_sections)
            .ok_or_else(|| property_plan_limit_error("solver property plan item limit exceeded"))?;
        entry_count = entry_count
            .checked_add(nested_entries)
            .ok_or_else(|| property_plan_limit_error("solver property plan item limit exceeded"))?;
    }
    Ok((section_count, entry_count))
}

fn append_property_entries(
    dictionary: &str,
    section: Option<&str>,
    entries: &[crate::properties::PropertyEntry],
    sections: &[PropertySection],
    entry_plans: &mut Vec<SolverPropertyEntryPlan>,
    budget: &mut PropertyPlanBudget,
) -> Result<()> {
    for entry in entries {
        let formatted_value_len = property_value_plan_len(&entry.value)?;
        let payload = dictionary
            .len()
            .checked_add(section.map_or(0, str::len))
            .and_then(|bytes| bytes.checked_add(entry.key.len()))
            .and_then(|bytes| bytes.checked_add(formatted_value_len))
            .ok_or_else(|| {
                property_plan_limit_error("solver property plan payload limit exceeded")
            })?;
        budget.add_item(payload)?;
        let formatted_value = format_property_value_for_plan(&entry.value, formatted_value_len)?;
        entry_plans
            .try_reserve(1)
            .map_err(|_| crate::MeshError::OutOfMemory)?;
        entry_plans.push(SolverPropertyEntryPlan {
            dictionary: try_copy_property_plan_string(dictionary)?,
            section: section.map(try_copy_property_plan_string).transpose()?,
            key: try_copy_property_plan_string(&entry.key)?,
            value: formatted_value,
        });
    }

    for nested in sections {
        let nested_path_len = match section {
            Some(section) => section
                .len()
                .checked_add(1)
                .and_then(|bytes| bytes.checked_add(nested.name.len()))
                .ok_or_else(|| {
                    property_plan_limit_error("solver property plan payload limit exceeded")
                })?,
            None => nested.name.len(),
        };
        budget.add_item(nested_path_len)?;
        let nested_path = if let Some(section) = section {
            try_join_property_plan_parts(&[section, ".", &nested.name])?
        } else {
            try_copy_property_plan_string(&nested.name)?
        };
        append_property_entries(
            dictionary,
            Some(&nested_path),
            &nested.entries,
            &nested.sections,
            entry_plans,
            budget,
        )?;
    }
    Ok(())
}

fn property_value_plan_len(value: &[String]) -> Result<usize> {
    if value.is_empty() {
        return Ok("empty".len());
    }
    let separators = value
        .len()
        .checked_sub(1)
        .ok_or_else(|| property_plan_limit_error("solver property plan payload limit exceeded"))?;
    value.iter().try_fold(separators, |bytes, part| {
        bytes
            .checked_add(part.len())
            .ok_or_else(|| property_plan_limit_error("solver property plan payload limit exceeded"))
    })
}

fn format_property_value_for_plan(value: &[String], capacity: usize) -> Result<String> {
    if value.is_empty() {
        return try_copy_property_plan_string("empty");
    }
    let mut formatted = String::new();
    formatted
        .try_reserve_exact(capacity)
        .map_err(|_| crate::MeshError::OutOfMemory)?;
    for (index, part) in value.iter().enumerate() {
        if index > 0 {
            formatted.push(' ');
        }
        formatted.push_str(part);
    }
    Ok(formatted)
}

fn append_property_validation_warnings(
    dictionaries: &[PropertyDictionary],
    warnings: &mut Vec<String>,
    budget: &mut PropertyPlanBudget,
) -> Result<()> {
    if dictionaries.is_empty() {
        return push_property_warning(
            warnings,
            budget,
            &["no constant property dictionaries found"],
            None,
        );
    }

    for dictionary in dictionaries {
        let label_len = property_dictionary_label_len(dictionary)?;
        budget.add_temporary_payload(label_len)?;
        let label = property_dictionary_label(dictionary)?;
        if dictionary.entries.is_empty() && dictionary.sections.is_empty() {
            push_property_warning(
                warnings,
                budget,
                &["property dictionary '", &label, "' has no entries"],
                None,
            )?;
        }
        append_dimensioned_property_warnings(&label, &dictionary.entries, warnings, budget)?;
        append_section_property_warnings(&label, &dictionary.sections, warnings, budget)?;
    }
    Ok(())
}

fn append_section_property_warnings(
    dictionary: &str,
    sections: &[PropertySection],
    warnings: &mut Vec<String>,
    budget: &mut PropertyPlanBudget,
) -> Result<()> {
    for section in sections {
        let label_len = dictionary
            .len()
            .checked_add(1)
            .and_then(|bytes| bytes.checked_add(section.name.len()))
            .ok_or_else(|| {
                property_plan_limit_error("solver property plan payload limit exceeded")
            })?;
        budget.add_temporary_payload(label_len)?;
        let label = try_join_property_plan_parts(&[dictionary, ".", &section.name])?;
        append_dimensioned_property_warnings(&label, &section.entries, warnings, budget)?;
        append_section_property_warnings(&label, &section.sections, warnings, budget)?;
    }
    Ok(())
}

fn append_dimensioned_property_warnings(
    label: &str,
    entries: &[crate::properties::PropertyEntry],
    warnings: &mut Vec<String>,
    budget: &mut PropertyPlanBudget,
) -> Result<()> {
    for entry in entries {
        if entry.value.first().map(String::as_str) != Some("[") {
            continue;
        }
        let Some(end) = entry.value.iter().position(|value| value == "]") else {
            push_property_warning(
                warnings,
                budget,
                &[
                    label,
                    ".",
                    &entry.key,
                    " has an unterminated dimension vector",
                ],
                None,
            )?;
            continue;
        };
        if end != 8 {
            let dimension_entries = end.checked_sub(1).ok_or_else(|| {
                property_plan_limit_error("solver property warning construction failed")
            })?;
            push_property_warning(
                warnings,
                budget,
                &[
                    label,
                    ".",
                    &entry.key,
                    " dimension vector has ",
                    " entries; expected 7",
                ],
                Some(dimension_entries),
            )?;
        }
        let value_index = end.checked_add(1).ok_or_else(|| {
            property_plan_limit_error("solver property warning construction failed")
        })?;
        if entry.value.len() <= value_index {
            push_property_warning(
                warnings,
                budget,
                &[label, ".", &entry.key, " has dimensions but no value"],
                None,
            )?;
        }
    }
    Ok(())
}

fn push_property_warning(
    warnings: &mut Vec<String>,
    budget: &mut PropertyPlanBudget,
    parts: &[&str],
    number_before_last_part: Option<usize>,
) -> Result<()> {
    const PREFIX: &str = "properties: ";
    let number_len = number_before_last_part.map_or(0, decimal_usize_len);
    let parts_capacity = parts.iter().try_fold(PREFIX.len(), |bytes, part| {
        bytes
            .checked_add(part.len())
            .ok_or_else(|| property_plan_limit_error("solver property plan payload limit exceeded"))
    })?;
    let capacity = parts_capacity
        .checked_add(number_len)
        .ok_or_else(|| property_plan_limit_error("solver property plan payload limit exceeded"))?;
    budget.add_item(capacity)?;
    let mut warning = String::new();
    warning
        .try_reserve_exact(capacity)
        .map_err(|_| crate::MeshError::OutOfMemory)?;
    warning.push_str(PREFIX);
    if let Some(number) = number_before_last_part {
        let (last, leading) = parts.split_last().ok_or_else(|| {
            property_plan_limit_error("solver property warning construction failed")
        })?;
        for part in leading {
            warning.push_str(part);
        }
        push_decimal_usize(&mut warning, number);
        warning.push_str(last);
    } else {
        for part in parts {
            warning.push_str(part);
        }
    }
    warnings
        .try_reserve(1)
        .map_err(|_| crate::MeshError::OutOfMemory)?;
    warnings.push(warning);
    Ok(())
}

fn try_join_property_plan_parts(parts: &[&str]) -> Result<String> {
    let capacity = parts.iter().try_fold(0usize, |bytes, part| {
        bytes
            .checked_add(part.len())
            .ok_or(crate::MeshError::OutOfMemory)
    })?;
    let mut value = String::new();
    value
        .try_reserve_exact(capacity)
        .map_err(|_| crate::MeshError::OutOfMemory)?;
    for part in parts {
        value.push_str(part);
    }
    Ok(value)
}

fn try_copy_property_plan_string(value: &str) -> Result<String> {
    try_join_property_plan_parts(&[value])
}

fn decimal_usize_len(mut value: usize) -> usize {
    let mut digits = 1;
    while value >= 10 {
        value /= 10;
        digits += 1;
    }
    digits
}

fn push_decimal_usize(target: &mut String, mut value: usize) {
    let mut digits = [0u8; 40];
    let mut index = digits.len();
    loop {
        index -= 1;
        digits[index] = b'0' + (value % 10) as u8;
        value /= 10;
        if value == 0 {
            break;
        }
    }
    for digit in &digits[index..] {
        target.push(char::from(*digit));
    }
}

fn property_plan_limit_error(message: &'static str) -> crate::MeshError {
    let mut owned = String::new();
    if owned.try_reserve_exact(message.len()).is_err() {
        return crate::MeshError::OutOfMemory;
    }
    owned.push_str(message);
    crate::MeshError::InvalidInput(owned)
}

fn unique_field_names(fields: &crate::fields::InitialFieldSet) -> Vec<String> {
    let mut names = fields
        .fields
        .iter()
        .map(|field| field.name.clone())
        .collect::<Vec<_>>();
    names.sort();
    names.dedup();
    names
}

fn build_numerics_plan(
    case_dir: &Path,
    field_names: &[String],
    warnings: &mut Vec<String>,
) -> Result<SolverNumericsPlan> {
    let fv_schemes = read_fv_schemes(case_dir)?;
    let fv_solution = read_fv_solution(case_dir)?;
    let schemes_validation = fv_schemes.as_ref().map(validate_fv_schemes);
    let solution_validation = fv_solution
        .as_ref()
        .map(|solution| validate_fv_solution(solution, field_names));
    let no_warnings: &[String] = &[];
    let schemes_input = fv_schemes.as_ref().map(|schemes| {
        (
            schemes.sections.as_slice(),
            schemes_validation
                .as_ref()
                .map_or(no_warnings, |validation| validation.warnings.as_slice()),
        )
    });
    let solution_input = fv_solution.as_ref().map(|solution| {
        (
            solution.sections.as_slice(),
            solution_validation
                .as_ref()
                .map_or(no_warnings, |validation| validation.warnings.as_slice()),
        )
    });
    build_numerics_plan_from_loaded(
        schemes_input,
        solution_input,
        warnings,
        DerivedPlanLimits {
            items: MAX_SOLVER_NUMERICS_PLAN_ITEMS,
            payload_bytes: MAX_SOLVER_NUMERICS_PLAN_PAYLOAD_BYTES,
        },
    )
}

fn build_numerics_plan_from_loaded(
    fv_schemes: Option<(&[NumericsSection], &[String])>,
    fv_solution: Option<(&[NumericsSection], &[String])>,
    warnings: &mut Vec<String>,
    limits: DerivedPlanLimits,
) -> Result<SolverNumericsPlan> {
    let mut budget = DerivedPlanBudget::new(
        limits,
        "solver numerics plan item limit exceeded",
        "solver numerics plan payload limit exceeded",
    );
    let mut derived_warnings = Vec::new();

    let fv_schemes = match fv_schemes {
        Some((sections, validation_warnings)) => {
            let plan = build_numerics_dictionary_plan_with_budget(true, sections, &mut budget)?;
            append_prefixed_derived_warnings(
                &mut derived_warnings,
                &mut budget,
                "fvSchemes: ",
                validation_warnings,
            )?;
            plan
        }
        None => {
            push_prefixed_derived_warning(
                &mut derived_warnings,
                &mut budget,
                "",
                "no system/fvSchemes found; discretisation plan is empty",
            )?;
            build_numerics_dictionary_plan_with_budget(false, &[], &mut budget)?
        }
    };

    let fv_solution = match fv_solution {
        Some((sections, validation_warnings)) => {
            let plan = build_numerics_dictionary_plan_with_budget(true, sections, &mut budget)?;
            append_prefixed_derived_warnings(
                &mut derived_warnings,
                &mut budget,
                "fvSolution: ",
                validation_warnings,
            )?;
            plan
        }
        None => {
            push_prefixed_derived_warning(
                &mut derived_warnings,
                &mut budget,
                "",
                "no system/fvSolution found; solver settings plan is empty",
            )?;
            build_numerics_dictionary_plan_with_budget(false, &[], &mut budget)?
        }
    };

    append_derived_warnings_atomically(warnings, &mut derived_warnings)?;
    Ok(SolverNumericsPlan {
        fv_schemes,
        fv_solution,
    })
}

fn build_numerics_dictionary_plan_with_budget(
    present: bool,
    sections: &[NumericsSection],
    budget: &mut DerivedPlanBudget,
) -> Result<SolverNumericsDictionaryPlan> {
    let mut section_plans = Vec::new();
    let mut entries = Vec::new();
    append_numerics_sections(sections, None, &mut section_plans, &mut entries, budget)?;
    Ok(SolverNumericsDictionaryPlan {
        present,
        sections: section_plans,
        entries,
    })
}

fn append_numerics_sections(
    sections: &[NumericsSection],
    parent: Option<&str>,
    section_plans: &mut Vec<SolverNumericsSectionPlan>,
    entries: &mut Vec<SolverNumericsEntryPlan>,
    budget: &mut DerivedPlanBudget,
) -> Result<()> {
    for section in sections {
        let path_len = match parent {
            Some(parent) => parent
                .len()
                .checked_add(1)
                .and_then(|bytes| bytes.checked_add(section.name.len()))
                .ok_or_else(|| derived_plan_error("solver numerics plan payload limit exceeded"))?,
            None => section.name.len(),
        };
        budget.add_temporary_payload(path_len)?;
        let path = if let Some(parent) = parent {
            try_join_derived_plan_parts(&[parent, ".", &section.name])?
        } else {
            try_copy_derived_plan_string(&section.name)?
        };

        budget.add_item(path_len)?;
        section_plans
            .try_reserve(1)
            .map_err(|_| crate::MeshError::OutOfMemory)?;
        section_plans.push(SolverNumericsSectionPlan {
            path: try_copy_derived_plan_string(&path)?,
            entries: section.entries.len(),
        });
        for entry in &section.entries {
            let value_len = numerics_value_plan_len(&entry.value)?;
            let payload = path
                .len()
                .checked_add(entry.key.len())
                .and_then(|bytes| bytes.checked_add(value_len))
                .ok_or_else(|| derived_plan_error("solver numerics plan payload limit exceeded"))?;
            budget.add_item(payload)?;
            let value = format_numerics_value_for_plan(&entry.value, value_len)?;
            entries
                .try_reserve(1)
                .map_err(|_| crate::MeshError::OutOfMemory)?;
            entries.push(SolverNumericsEntryPlan {
                section: try_copy_derived_plan_string(&path)?,
                key: try_copy_derived_plan_string(&entry.key)?,
                value,
            });
        }
        append_numerics_sections(
            &section.sections,
            Some(&path),
            section_plans,
            entries,
            budget,
        )?;
    }
    Ok(())
}

fn numerics_value_plan_len(value: &[String]) -> Result<usize> {
    if value.is_empty() {
        return Ok("empty".len());
    }
    let separators = value
        .len()
        .checked_sub(1)
        .ok_or_else(|| derived_plan_error("solver numerics plan payload limit exceeded"))?;
    value.iter().try_fold(separators, |bytes, part| {
        bytes
            .checked_add(part.len())
            .ok_or_else(|| derived_plan_error("solver numerics plan payload limit exceeded"))
    })
}

fn format_numerics_value_for_plan(value: &[String], capacity: usize) -> Result<String> {
    if value.is_empty() {
        return try_copy_derived_plan_string("empty");
    }
    let mut formatted = String::new();
    formatted
        .try_reserve_exact(capacity)
        .map_err(|_| crate::MeshError::OutOfMemory)?;
    for (index, part) in value.iter().enumerate() {
        if index > 0 {
            formatted.push(' ');
        }
        formatted.push_str(part);
    }
    Ok(formatted)
}

fn append_prefixed_derived_warnings(
    derived_warnings: &mut Vec<String>,
    budget: &mut DerivedPlanBudget,
    prefix: &str,
    warnings: &[String],
) -> Result<()> {
    for warning in warnings {
        push_prefixed_derived_warning(derived_warnings, budget, prefix, warning)?;
    }
    Ok(())
}

fn push_prefixed_derived_warning(
    derived_warnings: &mut Vec<String>,
    budget: &mut DerivedPlanBudget,
    prefix: &str,
    warning: &str,
) -> Result<()> {
    let payload = prefix
        .len()
        .checked_add(warning.len())
        .ok_or_else(|| derived_plan_error(budget.payload_error))?;
    budget.add_item(payload)?;
    derived_warnings
        .try_reserve(1)
        .map_err(|_| crate::MeshError::OutOfMemory)?;
    derived_warnings.push(try_join_derived_plan_parts(&[prefix, warning])?);
    Ok(())
}

fn append_derived_warnings_atomically(
    warnings: &mut Vec<String>,
    derived_warnings: &mut Vec<String>,
) -> Result<()> {
    warnings
        .try_reserve(derived_warnings.len())
        .map_err(|_| crate::MeshError::OutOfMemory)?;
    warnings.append(derived_warnings);
    Ok(())
}

fn try_join_derived_plan_parts(parts: &[&str]) -> Result<String> {
    let capacity = parts.iter().try_fold(0usize, |bytes, part| {
        bytes
            .checked_add(part.len())
            .ok_or(crate::MeshError::OutOfMemory)
    })?;
    let mut value = String::new();
    value
        .try_reserve_exact(capacity)
        .map_err(|_| crate::MeshError::OutOfMemory)?;
    for part in parts {
        value.push_str(part);
    }
    Ok(value)
}

fn try_copy_derived_plan_string(value: &str) -> Result<String> {
    try_join_derived_plan_parts(&[value])
}

fn derived_plan_error(message: &'static str) -> crate::MeshError {
    let mut owned = String::new();
    if owned.try_reserve_exact(message.len()).is_err() {
        return crate::MeshError::OutOfMemory;
    }
    owned.push_str(message);
    crate::MeshError::InvalidInput(owned)
}

fn build_backend_plan(case_dir: &Path, warnings: &mut Vec<String>) -> Result<SolverBackendPlan> {
    let limits = DerivedPlanLimits {
        items: MAX_SOLVER_BACKEND_PLAN_ITEMS,
        payload_bytes: MAX_SOLVER_BACKEND_PLAN_PAYLOAD_BYTES,
    };
    let Some(config) = read_backend_config(case_dir)? else {
        return build_default_backend_plan(warnings, limits);
    };

    let resource_validation = validate_backend_resources(&config);
    let policy_validation = validate_backend_policy(&config);
    let validation = BackendPlanValidation {
        uses_cpu: resource_validation.uses_cpu,
        uses_gpu: resource_validation.uses_gpu,
        mixed_execution: resource_validation.mixed_execution,
        resource_warnings: &resource_validation.warnings,
        policy_warnings: &policy_validation.warnings,
    };
    build_backend_plan_from_validated_config(config, validation, warnings, limits)
}

struct BackendPlanValidation<'a> {
    uses_cpu: bool,
    uses_gpu: bool,
    mixed_execution: bool,
    resource_warnings: &'a [String],
    policy_warnings: &'a [String],
}

fn build_backend_plan_from_validated_config(
    config: BackendConfig,
    validation: BackendPlanValidation<'_>,
    warnings: &mut Vec<String>,
    limits: DerivedPlanLimits,
) -> Result<SolverBackendPlan> {
    let mut budget = DerivedPlanBudget::new(
        limits,
        "solver backend plan item limit exceeded",
        "solver backend plan payload limit exceeded",
    );
    let mut derived_warnings = Vec::new();
    append_prefixed_derived_warnings(
        &mut derived_warnings,
        &mut budget,
        "backend resources: ",
        validation.resource_warnings,
    )?;
    append_prefixed_derived_warnings(
        &mut derived_warnings,
        &mut budget,
        "backend policy: ",
        validation.policy_warnings,
    )?;

    let mut stages = Vec::new();
    for section in &config.sections {
        for entry in &section.entries {
            let payload = section
                .name
                .len()
                .checked_add(entry.step.len())
                .ok_or_else(|| derived_plan_error("solver backend plan payload limit exceeded"))?;
            budget.add_item(payload)?;
            stages
                .try_reserve(1)
                .map_err(|_| crate::MeshError::OutOfMemory)?;
            stages.push(SolverBackendStagePlan {
                section: try_copy_derived_plan_string(&section.name)?,
                step: try_copy_derived_plan_string(&entry.step)?,
                choice: entry.choice,
            });
        }
    }

    let plan = SolverBackendPlan {
        config_present: true,
        default: config.default,
        uses_cpu: validation.uses_cpu,
        uses_gpu: validation.uses_gpu,
        mixed_execution: validation.mixed_execution,
        cpu: SolverCpuResourcePlan {
            cpus: config.cpu.cpus,
            cores_per_cpu: config.cpu.cores_per_cpu,
            threads: config.cpu.threads,
            thread_pinning: config.cpu.thread_pinning,
            numa: config.cpu.numa,
        },
        gpu: SolverGpuResourcePlan {
            backend: config.gpu.backend,
            devices: config.gpu.devices,
            multi_gpu: config.gpu.multi_gpu,
            precision: config.gpu.precision,
        },
        stages,
    };
    append_derived_warnings_atomically(warnings, &mut derived_warnings)?;
    Ok(plan)
}

fn build_default_backend_plan(
    warnings: &mut Vec<String>,
    limits: DerivedPlanLimits,
) -> Result<SolverBackendPlan> {
    let mut budget = DerivedPlanBudget::new(
        limits,
        "solver backend plan item limit exceeded",
        "solver backend plan payload limit exceeded",
    );
    let mut derived_warnings = Vec::new();
    push_prefixed_derived_warning(
        &mut derived_warnings,
        &mut budget,
        "",
        "no system/ferrumBackends found; solver plan defaults to CPU",
    )?;
    let cpu = SolverCpuResourcePlan {
        cpus: copy_default_backend_value("auto", &mut budget)?,
        cores_per_cpu: copy_default_backend_value("auto", &mut budget)?,
        threads: copy_default_backend_value("auto", &mut budget)?,
        thread_pinning: copy_default_backend_value("off", &mut budget)?,
        numa: copy_default_backend_value("auto", &mut budget)?,
    };
    let mut devices = Vec::new();
    devices
        .try_reserve_exact(1)
        .map_err(|_| crate::MeshError::OutOfMemory)?;
    devices.push(copy_default_backend_value("auto", &mut budget)?);
    let gpu = SolverGpuResourcePlan {
        backend: copy_default_backend_value("auto", &mut budget)?,
        devices,
        multi_gpu: copy_default_backend_value("auto", &mut budget)?,
        precision: copy_default_backend_value("f64", &mut budget)?,
    };
    let plan = SolverBackendPlan {
        config_present: false,
        default: BackendChoice::Cpu,
        uses_cpu: true,
        uses_gpu: false,
        mixed_execution: false,
        cpu,
        gpu,
        stages: Vec::new(),
    };
    append_derived_warnings_atomically(warnings, &mut derived_warnings)?;
    Ok(plan)
}

fn copy_default_backend_value(
    value: &'static str,
    budget: &mut DerivedPlanBudget,
) -> Result<String> {
    budget.add_item(value.len())?;
    try_copy_derived_plan_string(value)
}

fn build_run_plan(
    control: &ControlDict,
    backends: &SolverBackendPlan,
    warnings: &mut Vec<String>,
) -> SolverRunPlan {
    let end_time = if control.stop_at == "endTime" {
        control.end_time
    } else {
        None
    };
    let estimated_steps = estimate_time_steps(control.start_time, end_time, control.delta_t);
    let estimated_write_events = estimate_write_events(
        control.write_control.as_str(),
        control.write_interval,
        control.start_time,
        end_time,
        estimated_steps,
        warnings,
    );

    SolverRunPlan {
        stop_at: control.stop_at.clone(),
        start_time: control.start_time,
        end_time,
        delta_t: control.delta_t,
        estimated_steps,
        write_control: control.write_control.clone(),
        write_interval: control.write_interval,
        estimated_write_events,
        stages: build_run_stages(backends),
    }
}

fn estimate_time_steps(
    start_time: Option<f64>,
    end_time: Option<f64>,
    delta_t: Option<f64>,
) -> Option<usize> {
    let start_time = start_time?;
    let end_time = end_time?;
    let delta_t = delta_t?;

    if !start_time.is_finite()
        || !end_time.is_finite()
        || !delta_t.is_finite()
        || delta_t <= 0.0
        || end_time < start_time
    {
        return None;
    }

    let duration = end_time - start_time;
    if duration <= f64::EPSILON {
        return Some(0);
    }

    let steps = (duration / delta_t).ceil();
    if !steps.is_finite() || steps > usize::MAX as f64 {
        return None;
    }

    Some(steps as usize)
}

fn estimate_write_events(
    write_control: &str,
    write_interval: Option<f64>,
    start_time: Option<f64>,
    end_time: Option<f64>,
    estimated_steps: Option<usize>,
    warnings: &mut Vec<String>,
) -> Option<usize> {
    if write_control == "none" {
        return Some(0);
    }

    let write_interval = write_interval?;
    if !write_interval.is_finite() || write_interval <= 0.0 {
        return None;
    }

    match write_control {
        "timeStep" => {
            let rounded = write_interval.round();
            if (write_interval - rounded).abs() > f64::EPSILON {
                warnings.push(format!(
                    "run plan: writeControl timeStep expects an integer writeInterval, found {write_interval}"
                ));
                return None;
            }
            let every_steps = rounded as usize;
            if every_steps == 0 {
                return None;
            }
            estimated_steps.map(|steps| steps / every_steps)
        }
        "runTime" | "adjustableRunTime" => {
            let start_time = start_time?;
            let end_time = end_time?;
            if !start_time.is_finite() || !end_time.is_finite() || end_time < start_time {
                return None;
            }
            let writes = ((end_time - start_time) / write_interval).floor();
            if !writes.is_finite() || writes < 0.0 || writes > usize::MAX as f64 {
                warnings.push(
                    "run plan: write-event estimate is outside the supported usize range"
                        .to_string(),
                );
                return None;
            }
            Some(writes as usize)
        }
        _ => None,
    }
}

fn build_run_stages(backends: &SolverBackendPlan) -> Vec<SolverRunStagePlan> {
    BUILT_IN_RUN_STAGES
        .iter()
        .map(|(section, step)| {
            let (choice, source) = resolve_stage_backend(backends, section, step);
            SolverRunStagePlan {
                section: (*section).to_string(),
                step: (*step).to_string(),
                choice,
                source,
            }
        })
        .collect()
}

fn resolve_stage_backend(
    backends: &SolverBackendPlan,
    section: &str,
    step: &str,
) -> (BackendChoice, SolverRunStageSource) {
    for stage in backends.stages.iter().rev() {
        if stage.section == section && stage.step == step {
            return (stage.choice, SolverRunStageSource::Configured);
        }
    }

    (backends.default, SolverRunStageSource::Default)
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::ops::Deref;
    use std::path::Path;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use crate::backends::{
        BackendChoice, BackendConfig, BackendSection, BackendSelection, CpuConfig, GpuConfig,
    };
    use crate::control::ControlDict;
    use crate::fields::{FieldFile, FieldLoadPolicy, FieldValueSummary, InitialFieldSet};
    use crate::numerics::NumericsSection;
    use crate::patches::PatchValidationSummary;
    use crate::properties::{PropertyDictionary, PropertyEntry};
    use crate::solver_state::SolverStateCpuBufferStatus;

    use super::{
        BackendPlanValidation, DerivedPlanLimits, PropertyPlanLimits, SolverBackendPlan,
        SolverBackendStagePlan, SolverCpuResourcePlan, SolverDimensionality, SolverGpuResourcePlan,
        SolverRunStageSource, build_backend_plan_from_validated_config,
        build_numerics_plan_from_loaded, build_properties_plan_from_dictionaries_with_limits,
        build_run_plan, build_solver_case_plan, build_solver_case_plan_with_policy,
        classify_dimensionality, validate_openfoam_case_structure,
    };

    #[test]
    fn classifies_plain_3d_meshes() {
        let summary = patch_summary(0, 0);
        assert_eq!(
            classify_dimensionality(&summary),
            SolverDimensionality::ThreeD
        );
    }

    #[test]
    fn classifies_empty_patches_as_2d() {
        let summary = patch_summary(2, 0);
        assert_eq!(
            classify_dimensionality(&summary),
            SolverDimensionality::TwoD
        );
    }

    #[test]
    fn classifies_wedge_patches_as_axisymmetric() {
        let summary = patch_summary(0, 2);
        assert_eq!(
            classify_dimensionality(&summary),
            SolverDimensionality::Axisymmetric
        );
    }

    #[test]
    fn classifies_mixed_empty_and_wedge_as_ambiguous() {
        let summary = patch_summary(2, 2);
        assert_eq!(
            classify_dimensionality(&summary),
            SolverDimensionality::MixedSpecialPatches
        );
    }

    #[test]
    fn property_plan_count_and_payload_caps_are_exact() {
        let dictionaries = vec![test_property_dictionary(
            "d",
            vec![PropertyEntry {
                key: "k".to_string(),
                value: vec!["v".to_string()],
            }],
        )];
        let exact_limits = PropertyPlanLimits {
            items: 2,
            payload_bytes: 6,
        };
        let mut warnings = Vec::new();

        let exact = build_properties_plan_from_dictionaries_with_limits(
            &dictionaries,
            &mut warnings,
            exact_limits,
        )
        .expect("the exact property-plan count and payload limits must succeed");

        assert_eq!(exact.dictionaries.len(), 1);
        assert_eq!(exact.entries.len(), 1);
        assert_eq!(exact.entries[0].dictionary, "d");
        assert_eq!(exact.entries[0].key, "k");
        assert_eq!(exact.entries[0].value, "v");
        assert!(warnings.is_empty());

        let mut count_warnings = vec!["sentinel".to_string()];
        let count_error = build_properties_plan_from_dictionaries_with_limits(
            &dictionaries,
            &mut count_warnings,
            PropertyPlanLimits {
                items: exact_limits.items - 1,
                payload_bytes: exact_limits.payload_bytes,
            },
        )
        .expect_err("one item over the property-plan count limit must fail");
        assert_eq!(
            count_error.to_string(),
            "solver property plan item limit exceeded"
        );
        assert_eq!(count_warnings, ["sentinel"]);

        let mut payload_warnings = vec!["sentinel".to_string()];
        let payload_error = build_properties_plan_from_dictionaries_with_limits(
            &dictionaries,
            &mut payload_warnings,
            PropertyPlanLimits {
                items: exact_limits.items,
                payload_bytes: exact_limits.payload_bytes - 1,
            },
        )
        .expect_err("one byte over the property-plan payload limit must fail");
        assert_eq!(
            payload_error.to_string(),
            "solver property plan payload limit exceeded"
        );
        assert_eq!(payload_warnings, ["sentinel"]);
    }

    #[test]
    fn property_warning_budget_is_exact_and_failure_appends_no_prefix() {
        let dictionaries = vec![test_property_dictionary("p", Vec::new())];
        let expected_warning = "properties: property dictionary 'p' has no entries";
        let exact_limits = PropertyPlanLimits {
            items: 2,
            payload_bytes: (3 * "p".len()) + expected_warning.len(),
        };
        let mut warnings = vec!["sentinel".to_string()];

        let plan = build_properties_plan_from_dictionaries_with_limits(
            &dictionaries,
            &mut warnings,
            exact_limits,
        )
        .expect("the exact property warning budget must succeed");

        assert_eq!(plan.dictionaries.len(), 1);
        assert!(plan.entries.is_empty());
        assert_eq!(warnings, ["sentinel", expected_warning]);

        let mut rejected_warnings = vec!["sentinel".to_string()];
        let error = build_properties_plan_from_dictionaries_with_limits(
            &dictionaries,
            &mut rejected_warnings,
            PropertyPlanLimits {
                items: exact_limits.items,
                payload_bytes: exact_limits.payload_bytes - 1,
            },
        )
        .expect_err("one warning byte over the shared payload limit must fail");
        assert_eq!(
            error.to_string(),
            "solver property plan payload limit exceeded"
        );
        assert_eq!(rejected_warnings, ["sentinel"]);
    }

    #[test]
    fn property_warning_count_failure_appends_no_partial_prefix() {
        let dictionaries = vec![
            test_property_dictionary("a", Vec::new()),
            test_property_dictionary("b", Vec::new()),
        ];
        let mut warnings = vec!["sentinel".to_string()];

        let error = build_properties_plan_from_dictionaries_with_limits(
            &dictionaries,
            &mut warnings,
            PropertyPlanLimits {
                items: 3,
                payload_bytes: usize::MAX,
            },
        )
        .expect_err("the second warning must exceed the shared item limit");

        assert_eq!(
            error.to_string(),
            "solver property plan item limit exceeded"
        );
        assert_eq!(warnings, ["sentinel"]);
    }

    #[test]
    fn property_dimension_warnings_preserve_exact_text() {
        let dictionaries = vec![test_property_dictionary(
            "transport",
            vec![PropertyEntry {
                key: "nu".to_string(),
                value: vec!["[".to_string(), "0".to_string(), "]".to_string()],
            }],
        )];
        let mut warnings = Vec::new();

        build_properties_plan_from_dictionaries_with_limits(
            &dictionaries,
            &mut warnings,
            PropertyPlanLimits {
                items: usize::MAX,
                payload_bytes: usize::MAX,
            },
        )
        .expect("dimension warning construction should succeed");

        assert_eq!(
            warnings,
            [
                "properties: transport.nu dimension vector has 1 entries; expected 7",
                "properties: transport.nu has dimensions but no value",
            ]
        );
    }

    #[test]
    fn cumulative_numerics_section_paths_have_exact_caps_and_no_prefix() {
        let outer_name = "a".repeat(1_024);
        let inner_name = "b".repeat(1_024);
        let nested_path = format!("{outer_name}.{inner_name}");
        let sections = vec![NumericsSection {
            name: outer_name.clone(),
            entries: Vec::new(),
            sections: vec![NumericsSection {
                name: inner_name,
                entries: Vec::new(),
                sections: Vec::new(),
            }],
        }];
        let exact_limits = DerivedPlanLimits {
            items: 2,
            payload_bytes: (2 * outer_name.len()) + (2 * nested_path.len()),
        };
        let mut warnings = vec!["sentinel".to_string()];

        let exact = build_numerics_plan_from_loaded(
            Some((&sections, &[])),
            Some((&[], &[])),
            &mut warnings,
            exact_limits,
        )
        .expect("the exact cumulative numerics section-path budget must succeed");

        assert_eq!(exact.fv_schemes.sections.len(), 2);
        assert_eq!(exact.fv_schemes.sections[0].path, outer_name);
        assert_eq!(exact.fv_schemes.sections[1].path, nested_path);
        assert_eq!(warnings, ["sentinel"]);

        let mut payload_warnings = vec!["sentinel".to_string()];
        let payload_error = build_numerics_plan_from_loaded(
            Some((&sections, &[])),
            Some((&[], &[])),
            &mut payload_warnings,
            DerivedPlanLimits {
                items: exact_limits.items,
                payload_bytes: exact_limits.payload_bytes - 1,
            },
        )
        .expect_err("one cumulative section-path byte over the limit must fail");
        assert_eq!(
            payload_error.to_string(),
            "solver numerics plan payload limit exceeded"
        );
        assert_eq!(payload_warnings, ["sentinel"]);

        let mut count_warnings = vec!["sentinel".to_string()];
        let count_error = build_numerics_plan_from_loaded(
            Some((&sections, &[])),
            Some((&[], &[])),
            &mut count_warnings,
            DerivedPlanLimits {
                items: exact_limits.items - 1,
                payload_bytes: exact_limits.payload_bytes,
            },
        )
        .expect_err("one cumulative section-path item over the limit must fail");
        assert_eq!(
            count_error.to_string(),
            "solver numerics plan item limit exceeded"
        );
        assert_eq!(count_warnings, ["sentinel"]);
    }

    #[test]
    fn repeated_backend_section_names_have_exact_caps_and_no_prefix() {
        let section_name = "s".repeat(4_096);
        let resource_warnings = vec!["resource warning".to_string()];
        let warning_payload = "backend resources: ".len() + resource_warnings[0].len();
        let stage_payload = (2 * section_name.len()) + "one".len() + "two".len();
        let exact_limits = DerivedPlanLimits {
            items: 3,
            payload_bytes: warning_payload + stage_payload,
        };
        let mut warnings = vec!["sentinel".to_string()];

        let exact = build_backend_plan_from_validated_config(
            test_backend_config(&section_name),
            BackendPlanValidation {
                uses_cpu: true,
                uses_gpu: false,
                mixed_execution: false,
                resource_warnings: &resource_warnings,
                policy_warnings: &[],
            },
            &mut warnings,
            exact_limits,
        )
        .expect("the exact repeated backend-section budget must succeed");

        assert_eq!(exact.stages.len(), 2);
        assert_eq!(exact.stages[0].section, section_name);
        assert_eq!(exact.stages[1].section, section_name);
        assert_eq!(
            warnings,
            ["sentinel", "backend resources: resource warning"]
        );

        let mut payload_warnings = vec!["sentinel".to_string()];
        let payload_error = build_backend_plan_from_validated_config(
            test_backend_config(&section_name),
            BackendPlanValidation {
                uses_cpu: true,
                uses_gpu: false,
                mixed_execution: false,
                resource_warnings: &resource_warnings,
                policy_warnings: &[],
            },
            &mut payload_warnings,
            DerivedPlanLimits {
                items: exact_limits.items,
                payload_bytes: exact_limits.payload_bytes - 1,
            },
        )
        .expect_err("one repeated backend-section byte over the limit must fail");
        assert_eq!(
            payload_error.to_string(),
            "solver backend plan payload limit exceeded"
        );
        assert_eq!(payload_warnings, ["sentinel"]);

        let mut count_warnings = vec!["sentinel".to_string()];
        let count_error = build_backend_plan_from_validated_config(
            test_backend_config(&section_name),
            BackendPlanValidation {
                uses_cpu: true,
                uses_gpu: false,
                mixed_execution: false,
                resource_warnings: &resource_warnings,
                policy_warnings: &[],
            },
            &mut count_warnings,
            DerivedPlanLimits {
                items: exact_limits.items - 1,
                payload_bytes: exact_limits.payload_bytes,
            },
        )
        .expect_err("one repeated backend-section item over the limit must fail");
        assert_eq!(
            count_error.to_string(),
            "solver backend plan item limit exceeded"
        );
        assert_eq!(count_warnings, ["sentinel"]);
    }

    #[test]
    fn builds_time_step_run_plan() {
        let control = control_dict(0.0, 1.0, 0.25, "timeStep", 2.0);
        let backends = backend_plan(
            BackendChoice::Cpu,
            vec![
                ("flow", "residual", BackendChoice::Gpu),
                ("chemistry", "odeSolve", BackendChoice::Gpu),
                ("interfaces", "flux", BackendChoice::Auto),
            ],
        );
        let mut warnings = Vec::new();

        let run = build_run_plan(&control, &backends, &mut warnings);

        assert_eq!(run.estimated_steps, Some(4));
        assert_eq!(run.estimated_write_events, Some(2));
        let flow_residual = run
            .stages
            .iter()
            .find(|stage| stage.section == "flow" && stage.step == "residual")
            .expect("flow residual stage");
        assert_eq!(flow_residual.choice, BackendChoice::Gpu);
        assert_eq!(flow_residual.source, SolverRunStageSource::Configured);
        let flow_linear = run
            .stages
            .iter()
            .find(|stage| stage.section == "flow" && stage.step == "linearSolve")
            .expect("flow linear solve stage");
        assert_eq!(flow_linear.choice, BackendChoice::Cpu);
        assert_eq!(flow_linear.source, SolverRunStageSource::Default);
        let interface_flux = run
            .stages
            .iter()
            .find(|stage| stage.section == "interfaces" && stage.step == "flux")
            .expect("interface flux stage");
        assert_eq!(interface_flux.choice, BackendChoice::Auto);
        assert!(warnings.is_empty());
    }

    #[test]
    fn warns_for_fractional_time_step_write_interval() {
        let control = control_dict(0.0, 1.0, 0.25, "timeStep", 2.5);
        let backends = backend_plan(BackendChoice::Cpu, Vec::new());
        let mut warnings = Vec::new();

        let run = build_run_plan(&control, &backends, &mut warnings);

        assert_eq!(run.estimated_steps, Some(4));
        assert_eq!(run.estimated_write_events, None);
        assert!(
            warnings
                .iter()
                .any(|warning| warning.contains("integer writeInterval"))
        );
    }

    #[test]
    fn warns_for_missing_openfoam_compatibility_requirements() {
        let case_dir = create_temp_case_dir("missing-openfoam-structure");
        let mut warnings = Vec::new();
        let fields = InitialFieldSet {
            case_dir: case_dir.to_path_buf(),
            fields: Vec::new(),
        };

        validate_openfoam_case_structure(&case_dir, &fields, &mut warnings);
        let compatibility = extract_openfoam_warnings(&warnings);

        assert!(has_openfoam_warning(
            &compatibility,
            "missing mandatory system/controlDict"
        ));
        assert!(has_openfoam_warning(
            &compatibility,
            "missing mandatory constant/transportProperties"
        ));
        assert!(has_openfoam_warning(
            &compatibility,
            "missing mandatory time directory 0 (initial field files)"
        ));
        assert!(has_openfoam_warning(
            &compatibility,
            "missing mandatory field 0/U"
        ));
        assert!(has_openfoam_warning(
            &compatibility,
            "missing mandatory field 0/p"
        ));
        assert!(has_openfoam_warning(
            &compatibility,
            "incomplete constant/polyMesh (faces/owner/neighbour)"
        ));

        case_dir.cleanup();
    }

    #[test]
    fn accepts_openfoam_compatible_case_structure() {
        let case_dir = create_temp_case_dir("compatible-openfoam-structure");
        write_file(&case_dir.join("system/controlDict"), "FoamFile {}");
        write_file(
            &case_dir.join("constant/transportProperties"),
            "FoamFile {}",
        );
        write_file(&case_dir.join("system/fvSchemes"), "ddtSchemes {}");
        write_file(&case_dir.join("system/fvSolution"), "solvers {}");
        write_file(&case_dir.join("0/U"), "FoamFile {}");
        write_file(&case_dir.join("0/p"), "FoamFile {}");
        write_file(&case_dir.join("constant/polyMesh/faces"), "");
        write_file(&case_dir.join("constant/polyMesh/owner"), "");
        write_file(&case_dir.join("constant/polyMesh/neighbour"), "");

        let fields = InitialFieldSet {
            case_dir: case_dir.to_path_buf(),
            fields: vec![
                FieldFile {
                    path: case_dir.join("0/U"),
                    region: None,
                    name: "U".to_string(),
                    class_name: Some("volVectorField".to_string()),
                    dimensions: None,
                    internal_field: None,
                    boundary_patches: Vec::new(),
                },
                FieldFile {
                    path: case_dir.join("0/p"),
                    region: None,
                    name: "p".to_string(),
                    class_name: Some("volScalarField".to_string()),
                    dimensions: None,
                    internal_field: None,
                    boundary_patches: Vec::new(),
                },
            ],
        };
        let mut warnings = Vec::new();

        validate_openfoam_case_structure(&case_dir, &fields, &mut warnings);
        let compatibility = extract_openfoam_warnings(&warnings);
        assert!(
            compatibility.is_empty(),
            "expected no OpenFOAM compatibility warnings, got {:?}",
            compatibility
        );

        case_dir.cleanup();
    }

    #[test]
    fn public_solver_plan_is_full_while_explicit_summary_is_inspection_only() {
        let case_dir = create_temp_case_dir("public-field-load-policy");
        write_solver_ready_case(&case_dir);

        let solver_plan = build_solver_case_plan(&case_dir)
            .expect("the public solver-plan builder should produce solver-ready runtime data");
        let velocity = solver_plan
            .runtime_data
            .fields
            .iter()
            .find(|field| field.region.is_none() && field.name == "U")
            .expect("runtime velocity field");
        let pressure = solver_plan
            .runtime_data
            .fields
            .iter()
            .find(|field| field.region.is_none() && field.name == "p")
            .expect("runtime pressure field");
        assert_eq!(velocity.values.as_deref(), Some(&[1.0, 2.0, 3.0][..]));
        assert_eq!(pressure.values.as_deref(), Some(&[7.0][..]));

        let summary_plan = build_solver_case_plan_with_policy(&case_dir, FieldLoadPolicy::Summary)
            .expect("the explicit summary policy should retain inspection descriptors");
        assert_eq!(
            summary_plan.fields.fields.len(),
            solver_plan.fields.fields.len()
        );
        assert!(
            summary_plan
                .initial_fields
                .fields
                .iter()
                .all(|field| matches!(
                    field.internal_field,
                    Some(FieldValueSummary::NonUniform { values: None, .. })
                )),
            "summary plans should retain field metadata without loading payloads"
        );
        assert!(summary_plan.state.fields.iter().all(|field| {
            field.cpu_buffer.status == SolverStateCpuBufferStatus::NonUniformDataNotLoaded
                && !field.cpu_buffer.materializable
        }));
        assert_eq!(
            summary_plan.runtime_data.fields.len(),
            solver_plan.runtime_data.fields.len()
        );
        assert!(
            summary_plan
                .runtime_data
                .fields
                .iter()
                .all(|field| field.values.is_none())
        );
        assert!(summary_plan.runtime_data.warnings.is_empty());

        case_dir.cleanup();
    }

    #[test]
    fn solver_plan_accepts_standard_openfoam_auxiliary_boundary_entries() {
        let case_dir = create_temp_case_dir("openfoam-auxiliary-boundary-entries");
        write_solver_ready_case(&case_dir);
        write_file(
            &case_dir.join("0/U"),
            r#"
            FoamFile { class volVectorField; object U; }
            dimensions [0 1 -1 0 0 0 0];
            internalField nonuniform vectorField 1 ((1 2 3));
            boundaryField
            {
                walls
                {
                    type mixed;
                    refValue uniform (0 0 0);
                    refGradient uniform (0 0 0);
                    valueFraction uniform 1;
                    value uniform (0 0 0);
                }
            }
            "#,
        );
        write_file(
            &case_dir.join("0/p"),
            r#"
            FoamFile { class volScalarField; object p; }
            dimensions [1 -1 -2 0 0 0 0];
            internalField nonuniform scalarField 1 (7);
            boundaryField
            {
                walls
                {
                    type fixedGradient;
                    gradient uniform 0;
                }
            }
            "#,
        );

        let full = build_solver_case_plan(&case_dir)
            .expect("full solver planning must accept OpenFOAM auxiliary patch entries");
        assert_eq!(full.runtime_data.fields.len(), 2);
        assert!(
            full.runtime_data
                .fields
                .iter()
                .all(|field| field.values.is_some())
        );

        let summary = build_solver_case_plan_with_policy(&case_dir, FieldLoadPolicy::Summary)
            .expect("summary planning must accept OpenFOAM auxiliary patch entries");
        assert_eq!(summary.runtime_data.fields.len(), 2);
        assert!(
            summary
                .runtime_data
                .fields
                .iter()
                .all(|field| field.values.is_none())
        );

        case_dir.cleanup();
    }

    fn patch_summary(empty_patches: usize, wedge_patches: usize) -> PatchValidationSummary {
        PatchValidationSummary {
            case_dir: PathBuf::from("case"),
            patches: empty_patches + wedge_patches,
            empty_patches,
            wedge_patches,
            symmetry_patches: 0,
            warnings: Vec::new(),
        }
    }

    fn control_dict(
        start_time: f64,
        end_time: f64,
        delta_t: f64,
        write_control: &str,
        write_interval: f64,
    ) -> ControlDict {
        ControlDict {
            path: PathBuf::from("controlDict"),
            application: Some("ferrumRun".to_string()),
            solver: Some("incompressibleFluid".to_string()),
            start_from: "startTime".to_string(),
            start_time: Some(start_time),
            stop_at: "endTime".to_string(),
            end_time: Some(end_time),
            delta_t: Some(delta_t),
            write_control: write_control.to_string(),
            write_interval: Some(write_interval),
        }
    }

    fn backend_plan(
        default: BackendChoice,
        stages: Vec<(&str, &str, BackendChoice)>,
    ) -> SolverBackendPlan {
        SolverBackendPlan {
            config_present: true,
            default,
            uses_cpu: true,
            uses_gpu: stages
                .iter()
                .any(|(_, _, choice)| matches!(choice, BackendChoice::Gpu | BackendChoice::Auto)),
            mixed_execution: true,
            cpu: SolverCpuResourcePlan {
                cpus: "auto".to_string(),
                cores_per_cpu: "auto".to_string(),
                threads: "auto".to_string(),
                thread_pinning: "off".to_string(),
                numa: "auto".to_string(),
            },
            gpu: SolverGpuResourcePlan {
                backend: "auto".to_string(),
                devices: vec!["auto".to_string()],
                multi_gpu: "auto".to_string(),
                precision: "f64".to_string(),
            },
            stages: stages
                .into_iter()
                .map(|(section, step, choice)| SolverBackendStagePlan {
                    section: section.to_string(),
                    step: step.to_string(),
                    choice,
                })
                .collect(),
        }
    }

    fn test_property_dictionary(name: &str, entries: Vec<PropertyEntry>) -> PropertyDictionary {
        PropertyDictionary {
            path: PathBuf::from(name),
            region: None,
            name: name.to_string(),
            entries,
            sections: Vec::new(),
        }
    }

    fn test_backend_config(section_name: &str) -> BackendConfig {
        BackendConfig {
            path: PathBuf::from("ferrumBackends"),
            default: BackendChoice::Cpu,
            sections: vec![BackendSection {
                name: section_name.to_string(),
                entries: vec![
                    BackendSelection {
                        step: "one".to_string(),
                        choice: BackendChoice::Cpu,
                    },
                    BackendSelection {
                        step: "two".to_string(),
                        choice: BackendChoice::Cpu,
                    },
                ],
            }],
            cpu: CpuConfig::default(),
            gpu: GpuConfig::default(),
            cpu_explicit: true,
            gpu_explicit: false,
        }
    }

    struct TempCaseDir {
        path: PathBuf,
    }

    impl TempCaseDir {
        fn cleanup(self) {
            fs::remove_dir_all(&self.path).expect("temporary case cleanup");
        }
    }

    impl Deref for TempCaseDir {
        type Target = Path;

        fn deref(&self) -> &Self::Target {
            &self.path
        }
    }

    impl Drop for TempCaseDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.path);
        }
    }

    fn create_temp_case_dir(name: &str) -> TempCaseDir {
        let mut path = std::env::temp_dir();
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time should be available")
            .as_nanos();
        path.push(format!("ferrum-openfoam-test-{}-{}", name, nanos));
        fs::create_dir_all(&path).expect("temporary case dir");
        TempCaseDir { path }
    }

    fn write_solver_ready_case(case_dir: &Path) {
        write_file(
            &case_dir.join("system/controlDict"),
            r#"
            FoamFile { class dictionary; object controlDict; }
            application ferrumRun;
            solver incompressibleFluid;
            startFrom startTime;
            startTime 0;
            stopAt endTime;
            endTime 1;
            deltaT 1;
            writeControl timeStep;
            writeInterval 1;
            "#,
        );
        write_file(
            &case_dir.join("system/fvSchemes"),
            r#"
            FoamFile { class dictionary; object fvSchemes; }
            ddtSchemes { default steadyState; }
            gradSchemes { default Gauss linear; }
            divSchemes { default none; }
            laplacianSchemes { default Gauss linear corrected; }
            interpolationSchemes { default linear; }
            snGradSchemes { default corrected; }
            "#,
        );
        write_file(
            &case_dir.join("system/fvSolution"),
            r#"
            FoamFile { class dictionary; object fvSolution; }
            solvers
            {
                p { solver PCG; preconditioner DIC; tolerance 1e-10; relTol 0; }
                U { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-10; relTol 0; }
            }
            SIMPLE { nNonOrthogonalCorrectors 0; consistent false; }
            "#,
        );
        write_file(
            &case_dir.join("constant/transportProperties"),
            r#"
            FoamFile { class dictionary; object transportProperties; }
            transportModel Newtonian;
            nu [0 2 -1 0 0 0 0] 1e-5;
            rho [1 -3 0 0 0 0 0] 1;
            "#,
        );
        write_file(
            &case_dir.join("constant/polyMesh/points"),
            "8\n(\n(0 0 0)\n(1 0 0)\n(1 1 0)\n(0 1 0)\n(0 0 1)\n(1 0 1)\n(1 1 1)\n(0 1 1)\n)\n",
        );
        write_file(
            &case_dir.join("constant/polyMesh/faces"),
            "6\n(\n4(0 3 2 1)\n4(4 5 6 7)\n4(0 1 5 4)\n4(1 2 6 5)\n4(2 3 7 6)\n4(3 0 4 7)\n)\n",
        );
        write_file(
            &case_dir.join("constant/polyMesh/owner"),
            "6\n(\n0\n0\n0\n0\n0\n0\n)\n",
        );
        write_file(&case_dir.join("constant/polyMesh/neighbour"), "0\n(\n)\n");
        write_file(
            &case_dir.join("constant/polyMesh/boundary"),
            "1\n(\nwalls\n{\ntype wall;\nnFaces 6;\nstartFace 0;\n}\n)\n",
        );
        write_file(
            &case_dir.join("0/U"),
            r#"
            FoamFile { class volVectorField; object U; }
            dimensions [0 1 -1 0 0 0 0];
            internalField nonuniform vectorField 1 ((1 2 3));
            boundaryField
            {
                walls { type fixedValue; value uniform (0 0 0); }
            }
            "#,
        );
        write_file(
            &case_dir.join("0/p"),
            r#"
            FoamFile { class volScalarField; object p; }
            dimensions [1 -1 -2 0 0 0 0];
            internalField nonuniform scalarField 1 (7);
            boundaryField
            {
                walls { type zeroGradient; }
            }
            "#,
        );
    }

    fn write_file(path: &Path, content: &str) {
        let parent = path.parent().expect("test file path has parent");
        fs::create_dir_all(parent).expect("test file parent dir");
        fs::write(path, content).expect("test case file");
    }

    fn extract_openfoam_warnings(warnings: &[String]) -> Vec<String> {
        warnings
            .iter()
            .filter_map(|warning| {
                warning
                    .strip_prefix("OpenFOAM compatibility: ")
                    .map(str::to_string)
            })
            .collect()
    }

    fn has_openfoam_warning(warnings: &[String], needle: &str) -> bool {
        warnings.iter().any(|warning| warning.contains(needle))
    }
}
