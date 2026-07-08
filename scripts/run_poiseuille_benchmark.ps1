param(
    [string]$CaseRoot = "",
    [string]$OpenFoamWorkDir = "",
    [string]$OpenFoamJson = "",
    [string]$FerrumPlanJson = "",
    [string]$OutFile = "",
    [string]$ReportFile = "",
    [ValidateSet("Auto", "Native", "Wsl")]
    [string]$Mode = "Auto",
    [int]$OpenFoamSteps = 200,
    [switch]$SkipOpenFoam,
    [switch]$RequireOpenFoam,
    [switch]$UseExistingOpenFoamJson,
    [switch]$SkipFerrumSolve,
    [ValidateSet("jacobi", "cg")]
    [string]$FerrumLinearSolver = "cg",
    [double]$FerrumSolveTolerance = 1e-8,
    [int]$FerrumMaxIterations = 20000
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($CaseRoot)) {
    $CaseRoot = Join-Path $RepoRoot "examples\laminar_pipe"
}
if ([string]::IsNullOrWhiteSpace($OpenFoamWorkDir)) {
    $OpenFoamWorkDir = Join-Path $RepoRoot "target\openfoam\laminar_pipe"
}
if ([string]::IsNullOrWhiteSpace($OpenFoamJson)) {
    $OpenFoamJson = Join-Path $RepoRoot "target\benchmarks\laminar_pipe_openfoam.json"
}
if ([string]::IsNullOrWhiteSpace($FerrumPlanJson)) {
    $FerrumPlanJson = Join-Path $RepoRoot "target\benchmarks\laminar_pipe_ferrum_plan.json"
}
if ([string]::IsNullOrWhiteSpace($OutFile)) {
    $OutFile = Join-Path $RepoRoot "target\benchmarks\laminar_pipe_compare.json"
}
if ([string]::IsNullOrWhiteSpace($ReportFile)) {
    $ReportFile = Join-Path $RepoRoot "target\benchmarks\laminar_pipe_compare.md"
}
if ($OpenFoamSteps -le 0) {
    throw "OpenFoamSteps must be positive"
}
if ($FerrumSolveTolerance -le 0.0) {
    throw "FerrumSolveTolerance must be positive"
}
if ($FerrumMaxIterations -le 0) {
    throw "FerrumMaxIterations must be positive"
}

function Test-IsPathUnder([string]$Child, [string]$Parent) {
    $childFull = [System.IO.Path]::GetFullPath($Child)
    $parentFull = ([System.IO.Path]::GetFullPath($Parent)).TrimEnd([System.IO.Path]::DirectorySeparatorChar, [System.IO.Path]::AltDirectorySeparatorChar)
    return $childFull.Equals($parentFull, [System.StringComparison]::OrdinalIgnoreCase) -or
        $childFull.StartsWith($parentFull + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase) -or
        $childFull.StartsWith($parentFull + [System.IO.Path]::AltDirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)
}

$runOpenFoam = Join-Path $PSScriptRoot "run_openfoam_laminar_pipe.ps1"
$compare = Join-Path $PSScriptRoot "compare_laminar_pipe.ps1"
$openFoamJsonForCompare = $OpenFoamJson

if ($SkipOpenFoam) {
    if (!$UseExistingOpenFoamJson) {
        $openFoamJsonForCompare = Join-Path (Split-Path -Parent $OutFile) "laminar_pipe_openfoam_skipped.json"
        if (Test-Path -LiteralPath $openFoamJsonForCompare) {
            $targetRoot = Join-Path $RepoRoot "target"
            if (!(Test-IsPathUnder $openFoamJsonForCompare $targetRoot)) {
                throw "refusing to remove '$openFoamJsonForCompare' because it is outside '$targetRoot'"
            }
            Remove-Item -LiteralPath $openFoamJsonForCompare -Force
        }
    }
    Write-Output "skipping OpenFOAM reference run"
} else {
    Write-Output "running OpenFOAM simpleFoam reference"
    $openFoamArgs = @{
        CaseRoot = $CaseRoot
        WorkDir = $OpenFoamWorkDir
        OutFile = $OpenFoamJson
        Mode = $Mode
        EndTime = $OpenFoamSteps
        WriteInterval = $OpenFoamSteps
    }
    if ($RequireOpenFoam) {
        $openFoamArgs.RequireOpenFoam = $true
    }
    & $runOpenFoam @openFoamArgs
}

Write-Output "running Ferrum/OpenFOAM/analytic Poiseuille comparison"
$compareArgs = @{
    CaseRoot = $CaseRoot
    OpenFoamJson = $openFoamJsonForCompare
    FerrumPlanJson = $FerrumPlanJson
    OutFile = $OutFile
    ReportFile = $ReportFile
    FerrumLinearSolver = $FerrumLinearSolver
    FerrumSolveTolerance = $FerrumSolveTolerance
    FerrumMaxIterations = $FerrumMaxIterations
}
if ($SkipFerrumSolve) {
    $compareArgs.SkipFerrumSolve = $true
}
& $compare @compareArgs

Write-Output "poiseuille benchmark JSON: $OutFile"
Write-Output "poiseuille benchmark report: $ReportFile"
