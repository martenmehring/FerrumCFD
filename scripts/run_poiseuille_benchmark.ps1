param(
    [string]$CaseRoot = "",
    [string]$OpenFoamWorkDir = "",
    [string]$OpenFoamJson = "",
    [string]$FerrumPlanJson = "",
    [string]$OutFile = "",
    [string]$ReportFile = "",
    [string]$BenchmarkProperties = "",
    [ValidateSet("Auto", "Native", "Wsl")]
    [string]$Mode = "Auto",
    [int]$OpenFoamSteps = 200,
    [switch]$SkipOpenFoam,
    [switch]$RequireOpenFoam,
    [switch]$UseFerrumCaseForOpenFoam,
    [switch]$UseExistingOpenFoamJson,
    [switch]$SkipFerrumSolve,
    [ValidateSet("poiseuille", "laminarSimple")]
    [string]$FerrumSolver = "poiseuille",
    [ValidateSet("jacobi", "cg")]
    [string]$FerrumLinearSolver = "cg",
    [double]$FerrumSolveTolerance = 1e-8,
    [int]$FerrumMaxIterations = 20000,
    [int]$FerrumSimpleIterations = 100
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($CaseRoot)) {
    $CaseRoot = Join-Path $RepoRoot "tutorials\steadyIncompressible\laminarPipe\ferrum\case"
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
if ([string]::IsNullOrWhiteSpace($BenchmarkProperties)) {
    $BenchmarkProperties = Join-Path $RepoRoot "tutorials\steadyIncompressible\laminarPipe\analytical\pipeBenchmark"
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
if ($FerrumSimpleIterations -le 0) {
    throw "FerrumSimpleIterations must be positive"
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
    Write-Output "running independent OpenFOAM 13 foamRun/incompressibleFluid reference"
    $openFoamArgs = @{
        WorkDir = $OpenFoamWorkDir
        OutFile = $OpenFoamJson
        BenchmarkProperties = $BenchmarkProperties
        Mode = $Mode
        EndTime = $OpenFoamSteps
        WriteInterval = $OpenFoamSteps
    }
    if ($UseFerrumCaseForOpenFoam) {
        $openFoamArgs.FerrumOverlayCaseRoot = $CaseRoot
    }
    if ($RequireOpenFoam) {
        $openFoamArgs.RequireOpenFoam = $true
    }
    & $runOpenFoam @openFoamArgs
}

Write-Output "running Ferrum/OpenFOAM/analytic $FerrumSolver comparison"
$compareArgs = @{
    CaseRoot = $CaseRoot
    OpenFoamJson = $openFoamJsonForCompare
    FerrumPlanJson = $FerrumPlanJson
    OutFile = $OutFile
    ReportFile = $ReportFile
    BenchmarkProperties = $BenchmarkProperties
    FerrumSolver = $FerrumSolver
    FerrumLinearSolver = $FerrumLinearSolver
    FerrumSolveTolerance = $FerrumSolveTolerance
    FerrumMaxIterations = $FerrumMaxIterations
    FerrumSimpleIterations = $FerrumSimpleIterations
}
if ($SkipFerrumSolve) {
    $compareArgs.SkipFerrumSolve = $true
}
& $compare @compareArgs

Write-Output "$FerrumSolver benchmark JSON: $OutFile"
Write-Output "$FerrumSolver benchmark report: $ReportFile"
