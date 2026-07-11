use std::path::{Path, PathBuf};

use crate::Result;
use crate::backends::{
    BackendChoice, read_backend_config, validate_backend_policy, validate_backend_resources,
};
use crate::control::{ControlDict, read_control_dict, validate_control_dict};
use crate::fields::{read_initial_fields, validate_initial_field_boundaries};
use crate::interfaces::{read_interface_config, validate_interface_config};
use crate::numerics::{
    NumericsSection, format_numerics_value, read_fv_schemes, read_fv_solution, validate_fv_schemes,
    validate_fv_solution,
};
use crate::patches::{PatchValidationSummary, validate_poly_mesh_patches};
use crate::poly_mesh::PolyMesh;
use crate::properties::{
    PropertyDictionary, PropertySection, format_property_value, read_case_properties,
    validate_properties,
};
use crate::regions::{build_interface_registry, read_region_mesh_summaries};
use crate::runtime::{SolverRuntimeData, build_solver_runtime_data};
use crate::solver_state::{SolverStatePlan, build_solver_state_plan};

#[derive(Debug)]
pub struct SolverCasePlan {
    pub case_dir: PathBuf,
    pub control: ControlDict,
    pub mesh: SolverMeshPlan,
    pub fields: SolverFieldPlan,
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

pub fn build_solver_case_plan(case_dir: &Path) -> Result<SolverCasePlan> {
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

    let fields = read_initial_fields(case_dir)?;
    validate_openfoam_case_structure(case_dir, &fields, &mut warnings);
    let field_validation = validate_initial_field_boundaries(case_dir, &fields);
    warnings.extend(
        field_validation
            .warnings
            .iter()
            .map(|warning| format!("field boundary: {warning}")),
    );
    let state = build_solver_state_plan(case_dir, &fields);
    warnings.extend(
        state
            .warnings
            .iter()
            .map(|warning| format!("solver state: {warning}")),
    );
    let runtime_data = build_solver_runtime_data(case_dir, &mesh, &state)?;
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

    Ok(SolverCasePlan {
        case_dir: case_dir.to_path_buf(),
        control,
        mesh: build_mesh_plan(&mesh, &patch_validation, region_meshes),
        fields: SolverFieldPlan {
            fields: fields
                .fields
                .into_iter()
                .map(|field| SolverFieldEntryPlan {
                    region: field.region,
                    name: field.name,
                    class_name: field.class_name,
                    boundary_patches: field.boundary_patches.len(),
                })
                .collect(),
        },
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
    let validation = validate_properties(&dictionaries);
    warnings.extend(
        validation
            .warnings
            .iter()
            .map(|warning| format!("properties: {warning}")),
    );

    let mut dictionary_plans = Vec::new();
    let mut entry_plans = Vec::new();
    for dictionary in &dictionaries {
        let label = property_dictionary_label(dictionary);
        dictionary_plans.push(SolverPropertyDictionaryPlan {
            name: dictionary.name.clone(),
            region: dictionary.region.clone(),
            sections: count_property_sections(&dictionary.sections),
            entries: count_property_entries(dictionary),
        });
        append_property_entries(
            &label,
            None,
            &dictionary.entries,
            &dictionary.sections,
            &mut entry_plans,
        );
    }

    Ok(SolverPropertiesPlan {
        dictionaries: dictionary_plans,
        entries: entry_plans,
    })
}

fn property_dictionary_label(dictionary: &PropertyDictionary) -> String {
    if let Some(region) = &dictionary.region {
        format!("{region}/{}", dictionary.name)
    } else {
        dictionary.name.clone()
    }
}

fn count_property_sections(sections: &[PropertySection]) -> usize {
    sections
        .iter()
        .map(|section| 1 + count_property_sections(&section.sections))
        .sum()
}

fn count_property_entries(dictionary: &PropertyDictionary) -> usize {
    dictionary.entries.len() + count_section_property_entries(&dictionary.sections)
}

fn count_section_property_entries(sections: &[PropertySection]) -> usize {
    sections
        .iter()
        .map(|section| section.entries.len() + count_section_property_entries(&section.sections))
        .sum()
}

fn append_property_entries(
    dictionary: &str,
    section: Option<&str>,
    entries: &[crate::properties::PropertyEntry],
    sections: &[PropertySection],
    entry_plans: &mut Vec<SolverPropertyEntryPlan>,
) {
    for entry in entries {
        entry_plans.push(SolverPropertyEntryPlan {
            dictionary: dictionary.to_string(),
            section: section.map(str::to_string),
            key: entry.key.clone(),
            value: format_property_value(&entry.value),
        });
    }

    for nested in sections {
        let nested_path = if let Some(section) = section {
            format!("{section}.{}", nested.name)
        } else {
            nested.name.clone()
        };
        append_property_entries(
            dictionary,
            Some(&nested_path),
            &nested.entries,
            &nested.sections,
            entry_plans,
        );
    }
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
    let fv_schemes = match read_fv_schemes(case_dir)? {
        Some(schemes) => {
            let validation = validate_fv_schemes(&schemes);
            warnings.extend(
                validation
                    .warnings
                    .iter()
                    .map(|warning| format!("fvSchemes: {warning}")),
            );
            build_numerics_dictionary_plan(true, &schemes.sections)
        }
        None => {
            warnings.push("no system/fvSchemes found; discretisation plan is empty".to_string());
            build_numerics_dictionary_plan(false, &[])
        }
    };

    let fv_solution = match read_fv_solution(case_dir)? {
        Some(solution) => {
            let validation = validate_fv_solution(&solution, field_names);
            warnings.extend(
                validation
                    .warnings
                    .iter()
                    .map(|warning| format!("fvSolution: {warning}")),
            );
            build_numerics_dictionary_plan(true, &solution.sections)
        }
        None => {
            warnings.push("no system/fvSolution found; solver settings plan is empty".to_string());
            build_numerics_dictionary_plan(false, &[])
        }
    };

    Ok(SolverNumericsPlan {
        fv_schemes,
        fv_solution,
    })
}

fn build_numerics_dictionary_plan(
    present: bool,
    sections: &[NumericsSection],
) -> SolverNumericsDictionaryPlan {
    let mut section_plans = Vec::new();
    let mut entries = Vec::new();
    append_numerics_sections(sections, None, &mut section_plans, &mut entries);
    SolverNumericsDictionaryPlan {
        present,
        sections: section_plans,
        entries,
    }
}

fn append_numerics_sections(
    sections: &[NumericsSection],
    parent: Option<&str>,
    section_plans: &mut Vec<SolverNumericsSectionPlan>,
    entries: &mut Vec<SolverNumericsEntryPlan>,
) {
    for section in sections {
        let path = if let Some(parent) = parent {
            format!("{parent}.{}", section.name)
        } else {
            section.name.clone()
        };

        section_plans.push(SolverNumericsSectionPlan {
            path: path.clone(),
            entries: section.entries.len(),
        });
        for entry in &section.entries {
            entries.push(SolverNumericsEntryPlan {
                section: path.clone(),
                key: entry.key.clone(),
                value: format_numerics_value(&entry.value),
            });
        }
        append_numerics_sections(&section.sections, Some(&path), section_plans, entries);
    }
}

fn build_backend_plan(case_dir: &Path, warnings: &mut Vec<String>) -> Result<SolverBackendPlan> {
    let Some(config) = read_backend_config(case_dir)? else {
        warnings.push("no system/ferrumBackends found; solver plan defaults to CPU".to_string());
        return Ok(SolverBackendPlan {
            config_present: false,
            default: BackendChoice::Cpu,
            uses_cpu: true,
            uses_gpu: false,
            mixed_execution: false,
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
            stages: Vec::new(),
        });
    };

    let resource_validation = validate_backend_resources(&config);
    warnings.extend(
        resource_validation
            .warnings
            .iter()
            .map(|warning| format!("backend resources: {warning}")),
    );
    let policy_validation = validate_backend_policy(&config);
    warnings.extend(
        policy_validation
            .warnings
            .iter()
            .map(|warning| format!("backend policy: {warning}")),
    );

    let stages = config
        .sections
        .iter()
        .flat_map(|section| {
            section
                .entries
                .iter()
                .map(|entry| SolverBackendStagePlan {
                    section: section.name.clone(),
                    step: entry.step.clone(),
                    choice: entry.choice,
                })
                .collect::<Vec<_>>()
        })
        .collect();

    Ok(SolverBackendPlan {
        config_present: true,
        default: config.default,
        uses_cpu: resource_validation.uses_cpu,
        uses_gpu: resource_validation.uses_gpu,
        mixed_execution: resource_validation.mixed_execution,
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
    })
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
    use std::path::Path;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use crate::backends::BackendChoice;
    use crate::control::ControlDict;
    use crate::fields::{FieldFile, InitialFieldSet};
    use crate::patches::PatchValidationSummary;

    use super::{
        SolverBackendPlan, SolverBackendStagePlan, SolverCpuResourcePlan, SolverDimensionality,
        SolverGpuResourcePlan, SolverRunStageSource, build_run_plan, classify_dimensionality,
        validate_openfoam_case_structure,
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
            case_dir: case_dir.clone(),
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

        cleanup_temp_case_dir(&case_dir);
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
            case_dir: case_dir.clone(),
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

        cleanup_temp_case_dir(&case_dir);
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

    fn create_temp_case_dir(name: &str) -> PathBuf {
        let mut path = std::env::temp_dir();
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time should be available")
            .as_nanos();
        path.push(format!("ferrum-openfoam-test-{}-{}", name, nanos));
        fs::create_dir_all(&path).expect("temporary case dir");
        path
    }

    fn write_file(path: &Path, content: &str) {
        let parent = path.parent().expect("test file path has parent");
        fs::create_dir_all(parent).expect("test file parent dir");
        fs::write(path, content).expect("test case file");
    }

    fn cleanup_temp_case_dir(path: &Path) {
        let _ = fs::remove_dir_all(path);
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
