param(
    [string]$CaseRoot = "",
    [string]$OpenFoamWorkDir = "",
    [string]$OpenFoamJson = "",
    [string]$FerrumJson = "",
    [string]$FerrumMarkdown = "",
    [string]$FerrumLog = "",
    [string]$OutFile = "",
    [string]$ReportFile = "",
    [ValidateSet("Auto", "Native", "Wsl")]
    [string]$Mode = "Auto",
    [int]$OpenFoamSteps = 200,
    [switch]$SkipOpenFoam,
    [switch]$RequireOpenFoam,
    [switch]$UseExistingOpenFoamJson,
    [ValidateSet("jacobi", "cg")]
    [string]$FerrumLinearSolver = "jacobi",
    [double]$FerrumSolveTolerance = 1e-6,
    [int]$FerrumMaxIterations = 100,
    [int]$FerrumMaxSimpleIterations = 1
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
if ([string]::IsNullOrWhiteSpace($FerrumJson)) {
    $FerrumJson = Join-Path $RepoRoot "target\benchmarks\laminar_pipe_laminar_simple.json"
}
if ([string]::IsNullOrWhiteSpace($FerrumMarkdown)) {
    $FerrumMarkdown = Join-Path $RepoRoot "target\benchmarks\laminar_pipe_laminar_simple.md"
}
if ([string]::IsNullOrWhiteSpace($FerrumLog)) {
    $FerrumLog = Join-Path $RepoRoot "target\benchmarks\laminar_pipe_laminar_simple.log"
}
if ([string]::IsNullOrWhiteSpace($OutFile)) {
    $OutFile = Join-Path $RepoRoot "target\benchmarks\laminar_pipe_laminar_simple_compare.json"
}
if ([string]::IsNullOrWhiteSpace($ReportFile)) {
    $ReportFile = Join-Path $RepoRoot "target\benchmarks\laminar_pipe_laminar_simple_compare.md"
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
if ($FerrumMaxSimpleIterations -le 0) {
    throw "FerrumMaxSimpleIterations must be positive"
}

function Format-F64([double]$Value) {
    return $Value.ToString("G17", [System.Globalization.CultureInfo]::InvariantCulture)
}

function Format-NullableNumber($Value, [string]$Format = "G6") {
    if ($null -eq $Value) {
        return "n/a"
    }
    return ([double]$Value).ToString($Format, [System.Globalization.CultureInfo]::InvariantCulture)
}

function Format-NullablePercent($Value) {
    if ($null -eq $Value) {
        return "n/a"
    }
    return (([double]$Value) * 100.0).ToString("F3", [System.Globalization.CultureInfo]::InvariantCulture) + "%"
}

function Format-CommandLine([string]$Command, [string[]]$Arguments) {
    $parts = New-Object System.Collections.Generic.List[string]
    $parts.Add($Command)
    foreach ($argument in $Arguments) {
        if ($argument -match "\s") {
            $parts.Add('"' + $argument.Replace('"', '\"') + '"')
        } else {
            $parts.Add($argument)
        }
    }
    return ($parts -join " ")
}

function Invoke-FerrumLaminarSimple(
    [string]$CaseRoot,
    [string]$ReportJson,
    [string]$ReportMarkdown,
    [string]$LogPath,
    [string]$LinearSolver,
    [double]$SolveTolerance,
    [int]$MaxIterations,
    [int]$MaxSimpleIterations
) {
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $ReportJson) | Out-Null
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $LogPath) | Out-Null

    $exe = Join-Path $RepoRoot "target\debug\ferrumSolver.exe"
    if (Test-Path -LiteralPath $exe) {
        $command = $exe
        $arguments = @(
            "-case", $CaseRoot,
            "--solveLaminarSimple",
            "--linearSolver", $LinearSolver,
            "--solveTolerance", (Format-F64 $SolveTolerance),
            "--maxIterations", $MaxIterations.ToString([System.Globalization.CultureInfo]::InvariantCulture),
            "--maxSimpleIterations", $MaxSimpleIterations.ToString([System.Globalization.CultureInfo]::InvariantCulture),
            "--solveReportJson", $ReportJson,
            "--solveReportMarkdown", $ReportMarkdown
        )
    } else {
        $command = "cargo"
        $arguments = @(
            "run", "-p", "ferrum-cli", "--bin", "ferrumSolver", "--",
            "-case", $CaseRoot,
            "--solveLaminarSimple",
            "--linearSolver", $LinearSolver,
            "--solveTolerance", (Format-F64 $SolveTolerance),
            "--maxIterations", $MaxIterations.ToString([System.Globalization.CultureInfo]::InvariantCulture),
            "--maxSimpleIterations", $MaxSimpleIterations.ToString([System.Globalization.CultureInfo]::InvariantCulture),
            "--solveReportJson", $ReportJson,
            "--solveReportMarkdown", $ReportMarkdown
        )
    }

    $script:ferrumExitCode = $null
    $elapsed = Measure-Command {
        Push-Location $RepoRoot
        try {
            & $command @arguments *> $LogPath
            $script:ferrumExitCode = $LASTEXITCODE
        } finally {
            Pop-Location
        }
    }
    $exitCode = if ($null -eq $script:ferrumExitCode) { 0 } else { $script:ferrumExitCode }
    if ($exitCode -ne 0) {
        throw "Ferrum laminar SIMPLE solve failed with exit code $exitCode. See $LogPath"
    }

    return [ordered]@{
        command = Format-CommandLine -Command $command -Arguments $arguments
        log = $LogPath
        reportJson = $ReportJson
        reportMarkdown = $ReportMarkdown
        commandWallClockSeconds = $elapsed.TotalSeconds
    }
}

$runOpenFoam = Join-Path $PSScriptRoot "run_openfoam_laminar_pipe.ps1"
$openFoamResult = $null

if ($SkipOpenFoam) {
    Write-Output "skipping OpenFOAM simpleFoam reference"
    if ($UseExistingOpenFoamJson -and (Test-Path -LiteralPath $OpenFoamJson)) {
        $openFoamResult = Get-Content -LiteralPath $OpenFoamJson -Raw | ConvertFrom-Json
    }
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
    if (Test-Path -LiteralPath $OpenFoamJson) {
        $openFoamResult = Get-Content -LiteralPath $OpenFoamJson -Raw | ConvertFrom-Json
    }
}

Write-Output "running Ferrum laminar SIMPLE benchmark"
$ferrumRun = Invoke-FerrumLaminarSimple `
    -CaseRoot $CaseRoot `
    -ReportJson $FerrumJson `
    -ReportMarkdown $FerrumMarkdown `
    -LogPath $FerrumLog `
    -LinearSolver $FerrumLinearSolver `
    -SolveTolerance $FerrumSolveTolerance `
    -MaxIterations $FerrumMaxIterations `
    -MaxSimpleIterations $FerrumMaxSimpleIterations

$ferrum = Get-Content -LiteralPath $FerrumJson -Raw | ConvertFrom-Json
$openFoamPressureLoss = $null
if ($null -ne $openFoamResult -and $null -ne $openFoamResult.openFoam) {
    $openFoamPressureLoss = $openFoamResult.openFoam.pressureLoss
}

$combined = [ordered]@{
    caseDir = $CaseRoot
    benchmark = "laminar-pipe-laminar-simple"
    units = [ordered]@{
        default = "SI"
        pressure = "Pa"
        length = "m"
        velocity = "m/s"
    }
    analytic = [ordered]@{
        pressureLossModel = "HagenPoiseuille"
        pressureDropPa = $ferrum.options.pressureDrop
        meanVelocityMps = $ferrum.solution.analyticMeanVelocity
        flowRateM3s = $ferrum.solution.analyticFlowRate
    }
    ferrum = [ordered]@{
        solver = "laminarSimple"
        status = "ran"
        command = $ferrumRun.command
        log = $ferrumRun.log
        reportJson = $ferrumRun.reportJson
        reportMarkdown = $ferrumRun.reportMarkdown
        commandWallClockSeconds = $ferrumRun.commandWallClockSeconds
        report = $ferrum
    }
    openFoam = $openFoamResult
    comparison = [ordered]@{
        ferrumMeanVelocityRelativeErrorToAnalytic = $ferrum.solution.relativeMeanVelocityError
        ferrumPressureDropRelativeErrorToAnalytic = $ferrum.solution.relativePressureDropError
        ferrumSolveWallClockSeconds = $ferrum.solve.wallClockSeconds
        ferrumCommandWallClockSeconds = $ferrumRun.commandWallClockSeconds
        openFoamPressureDropRelativeErrorToAnalytic = if ($null -ne $openFoamPressureLoss) { $openFoamPressureLoss.relativeErrorToAnalytic } else { $null }
        openFoamWallClockSeconds = if ($null -ne $openFoamResult -and $null -ne $openFoamResult.openFoam) { $openFoamResult.openFoam.wallClockSeconds } else { $null }
        openFoamDeltaPPa = if ($null -ne $openFoamPressureLoss) { $openFoamPressureLoss.deltaPPa } else { $null }
        ferrumPressureDropFromMeanPa = $ferrum.solution.pressureDropFromMean
        ferrumPressureDropFromFieldPa = $ferrum.solution.pressureDropFromField
    }
}

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $OutFile) | Out-Null
$combined | ConvertTo-Json -Depth 32 | Set-Content -LiteralPath $OutFile -Encoding ASCII

$md = New-Object System.Collections.Generic.List[string]
$md.Add("# Laminar Pipe Laminar SIMPLE Benchmark")
$md.Add("")
$md.Add("| Solver | Metric | Value |")
$md.Add("| --- | --- | ---: |")
$md.Add("| Ferrum laminarSimple | cells | $($ferrum.mesh.cells) |")
$md.Add("| Ferrum laminarSimple | SIMPLE iterations | $($ferrum.solve.simpleIterations) |")
$md.Add("| Ferrum laminarSimple | linear solver | $($ferrum.options.linearSolver) |")
$md.Add("| Ferrum laminarSimple | solve wall clock [s] | $(Format-NullableNumber $ferrum.solve.wallClockSeconds 'F6') |")
$md.Add("| Ferrum laminarSimple | mean velocity error | $(Format-NullablePercent $ferrum.solution.relativeMeanVelocityError) |")
$md.Add("| Ferrum laminarSimple | pressure drop error | $(Format-NullablePercent $ferrum.solution.relativePressureDropError) |")
$md.Add("| Ferrum laminarSimple | continuity L2 final | $(Format-NullableNumber $ferrum.continuity.final.l2Norm 'G6') |")
$md.Add("| OpenFOAM simpleFoam | wall clock [s] | $(Format-NullableNumber $combined.comparison.openFoamWallClockSeconds 'F6') |")
$md.Add("| OpenFOAM simpleFoam | pressure drop error | $(Format-NullablePercent $combined.comparison.openFoamPressureDropRelativeErrorToAnalytic) |")
$md.Add("")
$md.Add("Analytic reference: Hagen-Poiseuille, SI units.")
$md.Add("")
$md.Add("- Ferrum report JSON: $FerrumJson")
$md.Add("- Ferrum report Markdown: $FerrumMarkdown")
$md.Add("- OpenFOAM JSON: $OpenFoamJson")

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $ReportFile) | Out-Null
$md | Set-Content -LiteralPath $ReportFile -Encoding ASCII

Write-Output "laminar SIMPLE benchmark JSON: $OutFile"
Write-Output "laminar SIMPLE benchmark report: $ReportFile"
