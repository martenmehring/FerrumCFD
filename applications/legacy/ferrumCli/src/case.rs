#[cfg(test)]
use std::fs;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use ferrum_mesh::safe_output::{SafeOutputEntry, SafeOutputRoot};

pub struct InitCaseOptions {
    pub case_dir: PathBuf,
    pub force: bool,
    pub regions: Vec<String>,
}

pub struct InitCaseSummary {
    pub case_dir: PathBuf,
    pub created_dirs: Vec<PathBuf>,
    pub written_files: Vec<PathBuf>,
    pub skipped_files: Vec<PathBuf>,
}

pub fn init_case(options: &InitCaseOptions) -> Result<InitCaseSummary, String> {
    let (output, created_dirs) =
        SafeOutputRoot::create_trusted(&options.case_dir).map_err(|error| {
            format!(
                "could not safely open or create case directory {} ({error})",
                options.case_dir.display()
            )
        })?;
    let mut summary = InitCaseSummary {
        case_dir: options.case_dir.clone(),
        created_dirs,
        written_files: Vec::new(),
        skipped_files: Vec::new(),
    };

    ensure_dir(&output, Path::new("0"), &mut summary)?;
    ensure_dir(&output, Path::new("constant"), &mut summary)?;
    ensure_dir(&output, Path::new("constant/polyMesh"), &mut summary)?;
    ensure_dir(&output, Path::new("system"), &mut summary)?;

    for region in &options.regions {
        ensure_dir(&output, &PathBuf::from("0").join(region), &mut summary)?;
        ensure_dir(
            &output,
            &PathBuf::from("constant").join(region).join("polyMesh"),
            &mut summary,
        )?;
    }

    write_file(
        &output,
        Path::new("README.md"),
        options.force,
        &mut summary,
        |writer| write_case_readme(writer, &options.regions),
    )?;
    write_file(
        &output,
        Path::new("system/controlDict"),
        options.force,
        &mut summary,
        write_control_dict,
    )?;
    write_file(
        &output,
        Path::new("system/fvSchemes"),
        options.force,
        &mut summary,
        write_fv_schemes,
    )?;
    write_file(
        &output,
        Path::new("system/fvSolution"),
        options.force,
        &mut summary,
        write_fv_solution,
    )?;
    write_file(
        &output,
        Path::new("system/ferrumBackends"),
        options.force,
        &mut summary,
        write_ferrum_backends,
    )?;
    write_file(
        &output,
        Path::new("constant/interfaces"),
        options.force,
        &mut summary,
        write_interfaces,
    )?;
    write_file(
        &output,
        Path::new("constant/transportProperties"),
        options.force,
        &mut summary,
        write_transport_properties,
    )?;

    Ok(summary)
}

fn ensure_dir(
    output: &SafeOutputRoot,
    relative: &Path,
    summary: &mut InitCaseSummary,
) -> Result<(), String> {
    let created = output.ensure_dir(relative).map_err(|error| {
        format!(
            "could not safely create case directory {} ({error})",
            output.path().join(relative).display()
        )
    })?;
    summary.created_dirs.extend(created);
    Ok(())
}

fn write_file(
    output: &SafeOutputRoot,
    relative: &Path,
    force: bool,
    summary: &mut InitCaseSummary,
    write: impl FnOnce(&mut BufWriter<File>) -> Result<(), std::io::Error>,
) -> Result<(), String> {
    let display = summary.case_dir.join(relative);
    let existing = output.entry(relative).map_err(|error| {
        format!(
            "could not safely inspect output file {} ({error})",
            display.display()
        )
    })?;
    match existing {
        Some(SafeOutputEntry::File) if !force => {
            summary.skipped_files.push(display);
            return Ok(());
        }
        Some(SafeOutputEntry::Directory | SafeOutputEntry::Other) => {
            return Err(format!(
                "refusing to replace non-regular case template output {}",
                display.display()
            ));
        }
        Some(SafeOutputEntry::File) | None => {}
    }

    let file = if force {
        output.open_replace_regular(relative)
    } else {
        output.open_create_new(relative)
    }
    .map_err(|error| format!("could not safely write {} ({error})", display.display()))?;
    let mut writer = BufWriter::new(file);
    write(&mut writer)
        .map_err(|error| format!("could not write {} ({error})", display.display()))?;
    summary.written_files.push(display);
    Ok(())
}

fn write_case_readme(
    writer: &mut BufWriter<File>,
    regions: &[String],
) -> Result<(), std::io::Error> {
    writeln!(writer, "# FerrumCFD Case")?;
    writeln!(writer)?;
    writeln!(writer, "This case was initialized with `initFerrumCase`.")?;
    writeln!(writer)?;
    writeln!(writer, "Typical workflow:")?;
    writeln!(writer)?;
    writeln!(writer, "```powershell")?;
    writeln!(writer, "gmshToFerrum path\\to\\mesh.msh -case .")?;
    writeln!(writer, "checkFerrumMesh -case .")?;
    writeln!(writer, "splitFerrumMeshRegions -case . -cellZones")?;
    writeln!(
        writer,
        "ferrumRun -solver incompressibleFluid -case . --preflight"
    )?;
    writeln!(writer, "```")?;
    if !regions.is_empty() {
        writeln!(writer)?;
        writeln!(writer, "Configured region folders:")?;
        writeln!(writer)?;
        for region in regions {
            writeln!(writer, "- `{region}`")?;
        }
    }
    Ok(())
}

fn write_control_dict(writer: &mut BufWriter<File>) -> Result<(), std::io::Error> {
    write_foam_header(writer, "dictionary", "controlDict", "system")?;
    writeln!(writer, "application ferrumRun;")?;
    writeln!(writer, "solver incompressibleFluid;")?;
    writeln!(writer, "startFrom startTime;")?;
    writeln!(writer, "startTime 0;")?;
    writeln!(writer, "stopAt endTime;")?;
    writeln!(writer, "endTime 1;")?;
    writeln!(writer, "deltaT 1;")?;
    writeln!(writer, "writeControl timeStep;")?;
    writeln!(writer, "writeInterval 1;")?;
    Ok(())
}

fn write_fv_schemes(writer: &mut BufWriter<File>) -> Result<(), std::io::Error> {
    write_foam_header(writer, "dictionary", "fvSchemes", "system")?;
    writeln!(writer, "ddtSchemes {{ default steadyState; }}")?;
    writeln!(writer, "gradSchemes {{ default Gauss linear; }}")?;
    writeln!(writer, "divSchemes {{ default none; }}")?;
    writeln!(
        writer,
        "laplacianSchemes {{ default Gauss linear corrected; }}"
    )?;
    writeln!(writer, "interpolationSchemes {{ default linear; }}")?;
    writeln!(writer, "snGradSchemes {{ default corrected; }}")?;
    Ok(())
}

fn write_fv_solution(writer: &mut BufWriter<File>) -> Result<(), std::io::Error> {
    write_foam_header(writer, "dictionary", "fvSolution", "system")?;
    writeln!(writer, "solvers {{ }}")?;
    writeln!(writer, "SIMPLE")?;
    writeln!(writer, "{{")?;
    writeln!(writer, "    nNonOrthogonalCorrectors 0;")?;
    writeln!(writer, "    consistent false;")?;
    writeln!(writer, "}}")?;
    writeln!(writer)?;
    writeln!(writer, "relaxationFactors")?;
    writeln!(writer, "{{")?;
    writeln!(writer, "    fields")?;
    writeln!(writer, "    {{")?;
    writeln!(writer, "        p 0.3;")?;
    writeln!(writer, "    }}")?;
    writeln!(writer, "    equations")?;
    writeln!(writer, "    {{")?;
    writeln!(writer, "        U 0.7;")?;
    writeln!(writer, "    }}")?;
    writeln!(writer, "}}")?;
    Ok(())
}

fn write_ferrum_backends(writer: &mut BufWriter<File>) -> Result<(), std::io::Error> {
    write_foam_header(writer, "dictionary", "ferrumBackends", "system")?;
    writeln!(
        writer,
        "// Execution policy only: physics models must remain backend-neutral."
    )?;
    writeln!(writer, "default cpu;")?;
    writeln!(writer)?;
    writeln!(writer, "mesh")?;
    writeln!(writer, "{{")?;
    writeln!(writer, "    import cpu;")?;
    writeln!(writer, "    checks cpu;")?;
    writeln!(writer, "}}")?;
    writeln!(writer)?;
    writeln!(writer, "cpu")?;
    writeln!(writer, "{{")?;
    writeln!(writer, "    cpus auto;")?;
    writeln!(writer, "    coresPerCpu auto;")?;
    writeln!(writer, "    threads auto;")?;
    writeln!(writer, "    threadPinning off;")?;
    writeln!(writer, "    numa auto;")?;
    writeln!(writer, "}}")?;
    writeln!(writer)?;
    writeln!(writer, "flow")?;
    writeln!(writer, "{{")?;
    writeln!(writer, "    nonlinearSolve auto;")?;
    writeln!(writer, "    residual auto;")?;
    writeln!(writer, "    jacobian auto;")?;
    writeln!(writer, "    linearSolve auto;")?;
    writeln!(writer, "    pressureCorrection auto;")?;
    writeln!(writer, "}}")?;
    writeln!(writer)?;
    writeln!(writer, "interfaces")?;
    writeln!(writer, "{{")?;
    writeln!(writer, "    flux auto;")?;
    writeln!(writer, "    coupling auto;")?;
    writeln!(writer, "    sourceTerms auto;")?;
    writeln!(writer, "}}")?;
    writeln!(writer)?;
    writeln!(writer, "chemistry")?;
    writeln!(writer, "{{")?;
    writeln!(writer, "    residual auto;")?;
    writeln!(writer, "    jacobian auto;")?;
    writeln!(writer, "    nonlinearSolve auto;")?;
    writeln!(writer, "    odeSolve auto;")?;
    writeln!(writer, "}}")?;
    writeln!(writer)?;
    writeln!(writer, "gpu")?;
    writeln!(writer, "{{")?;
    writeln!(writer, "    backend auto;")?;
    writeln!(writer, "    devices (auto);")?;
    writeln!(writer, "    multiGpu auto;")?;
    writeln!(writer, "    precision f64;")?;
    writeln!(writer, "}}")?;
    Ok(())
}

fn write_interfaces(writer: &mut BufWriter<File>) -> Result<(), std::io::Error> {
    write_foam_header(writer, "dictionary", "interfaces", "constant")?;
    writeln!(writer, "interfaces")?;
    writeln!(writer, "{{")?;
    writeln!(writer, "    // exampleInterface")?;
    writeln!(writer, "    // {{")?;
    writeln!(writer, "    //     regions (region_a region_b);")?;
    writeln!(writer, "    //     faceZone interface_name;")?;
    writeln!(
        writer,
        "    //     // Sign convention only: reversed physics gives a negative flux."
    )?;
    writeln!(writer, "    //     orientation region_a_to_region_b;")?;
    writeln!(writer, "    //     model none;")?;
    writeln!(writer, "    // }}")?;
    writeln!(writer, "}}")?;
    Ok(())
}

fn write_transport_properties(writer: &mut BufWriter<File>) -> Result<(), std::io::Error> {
    write_foam_header(writer, "dictionary", "transportProperties", "constant")?;
    writeln!(
        writer,
        "// Transport models and material properties will be solver-specific."
    )?;
    Ok(())
}

fn write_foam_header(
    writer: &mut BufWriter<File>,
    class_name: &str,
    object: &str,
    location: &str,
) -> Result<(), std::io::Error> {
    writeln!(writer, "FoamFile")?;
    writeln!(writer, "{{")?;
    writeln!(writer, "    version 2.0;")?;
    writeln!(writer, "    format ascii;")?;
    writeln!(writer, "    class {class_name};")?;
    writeln!(writer, "    location \"{location}\";")?;
    writeln!(writer, "    object {object};")?;
    writeln!(writer, "}}")?;
    writeln!(writer)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn init_case_rejects_symlinked_case_subdirectories() {
        let root = temp_dir("symlinked-case-subdir");
        let case_dir = root.join("case");
        let outside = root.join("outside-system");
        fs::create_dir_all(&case_dir).unwrap();
        fs::create_dir_all(&outside).unwrap();

        if create_directory_symlink(&outside, &case_dir.join("system")).is_err() {
            let _ = fs::remove_dir_all(&root);
            return;
        }

        let error = match init_case(&InitCaseOptions {
            case_dir: case_dir.clone(),
            force: false,
            regions: Vec::new(),
        }) {
            Ok(_) => panic!("init_case accepted a symlinked case subdirectory"),
            Err(error) => error,
        };

        assert!(error.contains("could not safely create case directory"));
        assert!(!outside.join("controlDict").exists());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn init_case_rejects_final_path_symlink_even_with_force() {
        let root = temp_dir("symlinked-case-file");
        let case_dir = root.join("case");
        let outside = root.join("outside-controlDict");
        fs::create_dir_all(case_dir.join("system")).unwrap();
        fs::create_dir_all(case_dir.join("constant")).unwrap();
        fs::create_dir_all(case_dir.join("0")).unwrap();
        fs::write(&outside, "do not clobber").unwrap();

        if create_file_symlink(&outside, &case_dir.join("system").join("controlDict")).is_err() {
            let _ = fs::remove_dir_all(&root);
            return;
        }

        let error = match init_case(&InitCaseOptions {
            case_dir: case_dir.clone(),
            force: true,
            regions: Vec::new(),
        }) {
            Ok(_) => panic!("init_case accepted a final path symlink"),
            Err(error) => error,
        };

        assert!(error.contains("refusing to replace non-regular case template output"));
        assert_eq!(fs::read_to_string(outside).unwrap(), "do not clobber");
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn init_case_force_replaces_regular_templates() {
        let root = temp_dir("force-regular");
        let case_dir = root.join("case");
        fs::create_dir_all(case_dir.join("system")).unwrap();
        fs::create_dir_all(case_dir.join("constant/polyMesh")).unwrap();
        fs::create_dir_all(case_dir.join("0")).unwrap();
        fs::write(case_dir.join("system/controlDict"), "old").unwrap();

        init_case(&InitCaseOptions {
            case_dir: case_dir.clone(),
            force: true,
            regions: Vec::new(),
        })
        .expect("regular templates should be replaceable with force");

        let control = fs::read_to_string(case_dir.join("system/controlDict")).unwrap();
        assert!(control.contains("application ferrumRun;"));
        assert!(!control.contains("old"));
        let _ = fs::remove_dir_all(root);
    }

    fn temp_dir(label: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("ferrum-cli-{label}-{unique}"))
    }

    #[cfg(unix)]
    fn create_directory_symlink(target: &Path, link: &Path) -> std::io::Result<()> {
        std::os::unix::fs::symlink(target, link)
    }

    #[cfg(windows)]
    fn create_directory_symlink(target: &Path, link: &Path) -> std::io::Result<()> {
        std::os::windows::fs::symlink_dir(target, link)
    }

    #[cfg(unix)]
    fn create_file_symlink(target: &Path, link: &Path) -> std::io::Result<()> {
        std::os::unix::fs::symlink(target, link)
    }

    #[cfg(windows)]
    fn create_file_symlink(target: &Path, link: &Path) -> std::io::Result<()> {
        std::os::windows::fs::symlink_file(target, link)
    }
}
