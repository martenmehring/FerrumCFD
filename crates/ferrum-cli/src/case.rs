use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

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
    let mut summary = InitCaseSummary {
        case_dir: options.case_dir.clone(),
        created_dirs: Vec::new(),
        written_files: Vec::new(),
        skipped_files: Vec::new(),
    };

    ensure_dir(&options.case_dir, &mut summary)?;
    ensure_dir(&options.case_dir.join("0"), &mut summary)?;
    ensure_dir(&options.case_dir.join("constant"), &mut summary)?;
    ensure_dir(
        &options.case_dir.join("constant").join("polyMesh"),
        &mut summary,
    )?;
    ensure_dir(&options.case_dir.join("system"), &mut summary)?;

    for region in &options.regions {
        ensure_dir(&options.case_dir.join("0").join(region), &mut summary)?;
        ensure_dir(
            &options
                .case_dir
                .join("constant")
                .join(region)
                .join("polyMesh"),
            &mut summary,
        )?;
    }

    write_file(
        &options.case_dir.join("README.md"),
        options.force,
        &mut summary,
        |writer| write_case_readme(writer, &options.regions),
    )?;
    write_file(
        &options.case_dir.join("system").join("controlDict"),
        options.force,
        &mut summary,
        write_control_dict,
    )?;
    write_file(
        &options.case_dir.join("system").join("fvSchemes"),
        options.force,
        &mut summary,
        write_fv_schemes,
    )?;
    write_file(
        &options.case_dir.join("system").join("fvSolution"),
        options.force,
        &mut summary,
        write_fv_solution,
    )?;
    write_file(
        &options.case_dir.join("system").join("ferrumBackends"),
        options.force,
        &mut summary,
        write_ferrum_backends,
    )?;
    write_file(
        &options.case_dir.join("constant").join("interfaces"),
        options.force,
        &mut summary,
        write_interfaces,
    )?;
    write_file(
        &options
            .case_dir
            .join("constant")
            .join("transportProperties"),
        options.force,
        &mut summary,
        write_transport_properties,
    )?;

    Ok(summary)
}

fn ensure_dir(path: &Path, summary: &mut InitCaseSummary) -> Result<(), String> {
    if !path.exists() {
        fs::create_dir_all(path)
            .map_err(|error| format!("could not create {} ({error})", path.display()))?;
        summary.created_dirs.push(path.to_path_buf());
    }
    Ok(())
}

fn write_file(
    path: &Path,
    force: bool,
    summary: &mut InitCaseSummary,
    write: impl FnOnce(&mut BufWriter<File>) -> Result<(), std::io::Error>,
) -> Result<(), String> {
    if path.exists() && !force {
        summary.skipped_files.push(path.to_path_buf());
        return Ok(());
    }

    let file = File::create(path)
        .map_err(|error| format!("could not write {} ({error})", path.display()))?;
    let mut writer = BufWriter::new(file);
    write(&mut writer).map_err(|error| format!("could not write {} ({error})", path.display()))?;
    summary.written_files.push(path.to_path_buf());
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
    writeln!(writer, "gmshToFerrumFoam path\\to\\mesh.msh -case .")?;
    writeln!(writer, "checkFerrumMesh -case .")?;
    writeln!(writer, "splitFerrumMeshRegions -case . -cellZones")?;
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
    writeln!(writer, "application ferrumSolver;")?;
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
    writeln!(writer, "ddtSchemes {{ default Euler; }}")?;
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
    writeln!(writer, "SIMPLE {{ nNonOrthogonalCorrectors 0; }}")?;
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
