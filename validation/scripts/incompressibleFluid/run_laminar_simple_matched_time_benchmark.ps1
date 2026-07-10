param(
    [string]$CaseRoot = "",
    [string]$BenchmarkRoot = "",
    [string]$BenchmarkProperties = "",
    [ValidateSet("Auto", "Native", "Wsl")]
    [string]$Mode = "Auto",
    [int]$MatchedTimeSeconds = 100,
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
if ([string]::IsNullOrWhiteSpace($BenchmarkRoot)) {
    $BenchmarkRoot = Join-Path $RepoRoot "target\benchmarks"
}
if ([string]::IsNullOrWhiteSpace($BenchmarkProperties)) {
    $BenchmarkProperties = Join-Path $RepoRoot "tutorials\incompressibleFluid\laminarPipe\analytical\pipeBenchmark"
}
if (!(Test-Path -LiteralPath $BenchmarkProperties -PathType Leaf)) {
    throw "benchmark properties not found: $BenchmarkProperties"
}
if ($MatchedTimeSeconds -le 0) {
    throw "MatchedTimeSeconds must be positive"
}

$tag = "laminar_pipe_matched_${MatchedTimeSeconds}s"
$openFoamWorkDir = Join-Path $RepoRoot "target\openfoam\$tag"
$openFoamJson = Join-Path $BenchmarkRoot "$tag.openfoam.json"
$ferrumPlanJson = Join-Path $BenchmarkRoot "$tag.ferrum_plan.json"
$outFile = Join-Path $BenchmarkRoot "$tag.compare.json"
$reportFile = Join-Path $BenchmarkRoot "$tag.compare.md"
$runner = Join-Path $PSScriptRoot "run_poiseuille_benchmark.ps1"

Write-Output "running matched laminar SIMPLE benchmark"
Write-Output "OpenFOAM 13: foamRun/incompressibleFluid endTime=$MatchedTimeSeconds deltaT=1"
Write-Output "Ferrum: minSimpleIterations=maxSimpleIterations=$MatchedTimeSeconds"
Write-Output "Note: for steady SIMPLE this is a pseudo-time/iteration budget, not a transient physical-time solve."

$args = @{
    CaseRoot = $CaseRoot
    OpenFoamWorkDir = $openFoamWorkDir
    OpenFoamJson = $openFoamJson
    FerrumPlanJson = $ferrumPlanJson
    OutFile = $outFile
    ReportFile = $reportFile
    BenchmarkProperties = $BenchmarkProperties
    Mode = $Mode
    OpenFoamSteps = $MatchedTimeSeconds
    FerrumSolver = "laminarSimple"
    FerrumSimpleIterations = $MatchedTimeSeconds
}
if ($SkipOpenFoam) {
    $args.SkipOpenFoam = $true
}
if ($RequireOpenFoam) {
    $args.RequireOpenFoam = $true
}
if ($UseExistingOpenFoamJson) {
    $args.UseExistingOpenFoamJson = $true
}
if ($SkipFerrumSolve) {
    $args.SkipFerrumSolve = $true
}

& $runner @args

if (Test-Path -LiteralPath $outFile) {
    $result = Get-Content -LiteralPath $outFile -Raw | ConvertFrom-Json
    Write-Output ""
    Write-Output "matched benchmark summary"
    Write-Output ("analytic deltaP [Pa]: {0:G8}" -f ([double]$result.analytic.deltaPPa))
    if ($null -ne $result.comparison.ferrumDeltaPPa) {
        Write-Output ("Ferrum pressure-field deltaP [Pa]: {0:G8} ({1:P3})" -f ([double]$result.comparison.ferrumDeltaPPa, [double]$result.comparison.ferrumRelativeErrorToAnalytic))
    }
    if ($null -ne $result.comparison.ferrumPressureDropFromMeanPa) {
        Write-Output ("Ferrum mean-U deltaP [Pa]: {0:G8} ({1:P3})" -f ([double]$result.comparison.ferrumPressureDropFromMeanPa, [double]$result.comparison.ferrumPressureDropFromMeanRelativeErrorToAnalytic))
    }
    if ($null -ne $result.comparison.openFoamDeltaPPa) {
        Write-Output ("OpenFOAM deltaP [Pa]: {0:G8} ({1:P3})" -f ([double]$result.comparison.openFoamDeltaPPa, [double]$result.comparison.openFoamRelativeErrorToAnalytic))
    }
    Write-Output ("matched step budget: {0}" -f $result.runBudget.matched)
}

Write-Output "matched benchmark JSON: $outFile"
Write-Output "matched benchmark report: $reportFile"
