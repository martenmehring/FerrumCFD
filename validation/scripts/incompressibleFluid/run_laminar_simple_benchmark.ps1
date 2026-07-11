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
    [int]$FerrumSimpleIterations = 100,
    [switch]$SkipOpenFoam,
    [switch]$RequireOpenFoam,
    [switch]$UseExistingOpenFoamJson,
    [switch]$SkipFerrumSolve
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))

if ([string]::IsNullOrWhiteSpace($CaseRoot)) {
    $CaseRoot = Join-Path $RepoRoot "tutorials\incompressibleFluid\laminarPipe\ferrum\case"
}
if ([string]::IsNullOrWhiteSpace($OpenFoamWorkDir)) {
    $OpenFoamWorkDir = Join-Path $RepoRoot "target\openfoam\laminar_pipe"
}
if ([string]::IsNullOrWhiteSpace($OpenFoamJson)) {
    $OpenFoamJson = Join-Path $RepoRoot "target\benchmarks\laminar_pipe_openfoam.json"
}
if ([string]::IsNullOrWhiteSpace($FerrumPlanJson)) {
    $FerrumPlanJson = Join-Path $RepoRoot "target\benchmarks\laminar_pipe_laminar_simple_plan.json"
}
if ([string]::IsNullOrWhiteSpace($OutFile)) {
    $OutFile = Join-Path $RepoRoot "target\benchmarks\laminar_pipe_laminar_simple_compare.json"
}
if ([string]::IsNullOrWhiteSpace($ReportFile)) {
    $ReportFile = Join-Path $RepoRoot "target\benchmarks\laminar_pipe_laminar_simple_compare.md"
}
if ([string]::IsNullOrWhiteSpace($BenchmarkProperties)) {
    $BenchmarkProperties = Join-Path $RepoRoot "tutorials\incompressibleFluid\laminarPipe\analytical\pipeBenchmark"
}
if ($OpenFoamSteps -le 0) {
    throw "OpenFoamSteps must be positive"
}
if ($FerrumSimpleIterations -le 0) {
    throw "FerrumSimpleIterations must be positive"
}

$runner = Join-Path $PSScriptRoot "run_poiseuille_benchmark.ps1"
$arguments = @{
    CaseRoot = $CaseRoot
    OpenFoamWorkDir = $OpenFoamWorkDir
    OpenFoamJson = $OpenFoamJson
    FerrumPlanJson = $FerrumPlanJson
    OutFile = $OutFile
    ReportFile = $ReportFile
    BenchmarkProperties = $BenchmarkProperties
    Mode = $Mode
    OpenFoamSteps = $OpenFoamSteps
    FerrumMode = "incompressibleFluid"
    FerrumSimpleIterations = $FerrumSimpleIterations
}
if ($SkipOpenFoam) {
    $arguments.SkipOpenFoam = $true
}
if ($RequireOpenFoam) {
    $arguments.RequireOpenFoam = $true
}
if ($UseExistingOpenFoamJson) {
    $arguments.UseExistingOpenFoamJson = $true
}
if ($SkipFerrumSolve) {
    $arguments.SkipFerrumSolve = $true
}

& $runner @arguments
