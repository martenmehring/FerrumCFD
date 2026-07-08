use std::path::{Path, PathBuf};

use crate::Result;
use crate::backends::{BackendChoice, read_backend_config, validate_backend_resources};
use crate::control::{ControlDict, read_control_dict};
use crate::fields::{read_initial_fields, validate_initial_field_boundaries};
use crate::interfaces::{read_interface_config, validate_interface_config};
use crate::patches::{PatchValidationSummary, validate_poly_mesh_patches};
use crate::poly_mesh::PolyMesh;
use crate::regions::{build_interface_registry, read_region_mesh_summaries};

#[derive(Debug)]
pub struct SolverCasePlan {
    pub case_dir: PathBuf,
    pub control: ControlDict,
    pub mesh: SolverMeshPlan,
    pub fields: SolverFieldPlan,
    pub interfaces: SolverInterfacePlan,
    pub backends: SolverBackendPlan,
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

pub fn build_solver_case_plan(case_dir: &Path) -> Result<SolverCasePlan> {
    let control = read_control_dict(case_dir)?;
    let mesh = PolyMesh::read(&case_dir.join("constant").join("polyMesh"))?;
    let patch_validation = validate_poly_mesh_patches(case_dir, &mesh);
    let mut warnings = Vec::new();
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
    let field_validation = validate_initial_field_boundaries(case_dir, &fields);
    warnings.extend(
        field_validation
            .warnings
            .iter()
            .map(|warning| format!("field boundary: {warning}")),
    );

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
        interfaces,
        backends,
        warnings,
    })
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

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use crate::patches::PatchValidationSummary;

    use super::{SolverDimensionality, classify_dimensionality};

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
}
