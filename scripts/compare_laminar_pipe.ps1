param(
    [string]$CaseRoot = "",
    [string]$OpenFoamJson = "",
    [string]$FerrumPlanJson = "",
    [string]$OutFile = "",
    [string]$ReportFile = ""
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($CaseRoot)) {
    $CaseRoot = Join-Path $RepoRoot "examples\laminar_pipe"
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

function Invoke-FerrumPreflight([string]$CaseRoot, [string]$PlanJson) {
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $PlanJson) | Out-Null
    $logPath = Join-Path (Split-Path -Parent $PlanJson) "laminar_pipe_ferrum_preflight.log"
    $exe = Join-Path $RepoRoot "target\debug\ferrumSolver.exe"
    if (Test-Path -LiteralPath $exe) {
        $command = $exe
        $arguments = @("-case", $CaseRoot, "--runnerDryRun", "--maxRunnerSteps", "2", "--planJson", $PlanJson)
    } else {
        $command = "cargo"
        $arguments = @("run", "--bin", "ferrumSolver", "--", "-case", $CaseRoot, "--runnerDryRun", "--maxRunnerSteps", "2", "--planJson", $PlanJson)
    }

    $script:ferrumExitCode = $null
    $elapsed = Measure-Command {
        & $command @arguments *> $logPath
        $script:ferrumExitCode = $LASTEXITCODE
    }
    $exitCode = if ($null -eq $script:ferrumExitCode) { 0 } else { $script:ferrumExitCode }
    if ($exitCode -ne 0) {
        throw "Ferrum preflight failed with exit code $exitCode. See $logPath"
    }

    return [ordered]@{
        command = $command
        log = $logPath
        planJson = $PlanJson
        wallClockSeconds = $elapsed.TotalSeconds
    }
}

function Get-StateSummary($Plan) {
    $fields = @($Plan.state.fields)
    return [ordered]@{
        fields = $fields.Count
        cpuBuffers = @($fields | Where-Object { $_.cpuBuffer.materializable }).Count
        bytesF64 = [int](($fields | ForEach-Object { $_.storage.bytesF64 } | Measure-Object -Sum).Sum)
        warnings = @($Plan.state.warnings)
    }
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

function Read-AnalyticDeltaP([string]$CaseRoot) {
    $path = Join-Path $CaseRoot "constant\pipeBenchmark"
    if (!(Test-Path -LiteralPath $path)) {
        return 1.6032
    }
    $content = Get-Content -LiteralPath $path -Raw
    $match = [regex]::Match($content, "(?m)^\s*expectedDeltaP\s+\[[^\]]+\]\s+([-+0-9.eE]+)\s*;")
    if (!$match.Success) {
        return 1.6032
    }
    return [double]::Parse($match.Groups[1].Value, [System.Globalization.CultureInfo]::InvariantCulture)
}

function Read-PipeBenchmarkMesh([string]$CaseRoot) {
    $path = Join-Path $CaseRoot "constant\pipeBenchmark"
    if (!(Test-Path -LiteralPath $path)) {
        return $null
    }
    $content = Get-Content -LiteralPath $path -Raw
    $result = [ordered]@{ type = "structuredCircularPipe"; axialCells = $null; radialCells = $null; angularSectors = $null; cells = $null }
    foreach ($name in @("axialCells", "radialCells", "angularSectors", "cells")) {
        $match = [regex]::Match($content, "(?m)^\s*$name\s+(\d+)\s*;")
        if ($match.Success) {
            $result[$name] = [int]::Parse($match.Groups[1].Value, [System.Globalization.CultureInfo]::InvariantCulture)
        }
    }
    return [pscustomobject]$result
}

function Write-MarkdownReport($Path, $Result) {
    $comparison = $Result.comparison
    $ferrum = $Result.ferrum
    $openFoam = $Result.openFoam
    $status = $Result.benchmarkStatus
    $mesh = $Result.mesh
    $lines = New-Object System.Collections.Generic.List[string]

    $lines.Add("# Laminar Pipe Benchmark")
    $lines.Add("")
    $lines.Add('All FerrumCFD-facing values are SI. OpenFOAM incompressible pressure is converted from kinematic pressure (`m2/s2`) to pressure (`Pa`) with `rho` before comparison.')
    $lines.Add("")
    $lines.Add("## Status")
    $lines.Add("")
    $lines.Add("| Check | Value |")
    $lines.Add("| --- | --- |")
    $lines.Add("| Ferrum preflight | $($status.ferrumPreflight) |")
    $lines.Add("| OpenFOAM reference | $($status.openFoamReference) |")
    $lines.Add("| Ferrum executable solver comparison | $($status.ferrumSolverComparison) |")
    $lines.Add("")
    if ($null -ne $mesh) {
        $lines.Add("## Mesh")
        $lines.Add("")
        $lines.Add("| Property | Value |")
        $lines.Add("| --- | ---: |")
        $lines.Add("| Type | $($mesh.type) |")
        $lines.Add("| Axial cells | $($mesh.axialCells) |")
        $lines.Add("| Radial cells | $($mesh.radialCells) |")
        $lines.Add("| Angular sectors | $($mesh.angularSectors) |")
        $lines.Add("| Total cells | $($mesh.cells) |")
        $lines.Add("")
    }
    $lines.Add("## Pressure Loss")
    $lines.Add("")
    $lines.Add("| Source | deltaP [Pa] | Relative error to analytic |")
    $lines.Add("| --- | ---: | ---: |")
    $lines.Add("| Analytic Hagen-Poiseuille | $(Format-NullableNumber $Result.analytic.deltaPPa "G8") | 0% |")
    $lines.Add("| OpenFOAM simpleFoam | $(Format-NullableNumber $comparison.openFoamDeltaPPa "G8") | $(Format-NullablePercent $comparison.openFoamRelativeErrorToAnalytic) |")
    $lines.Add("| FerrumCFD solver | n/a | n/a |")
    if ($null -ne $comparison.openFoamPressureLossMethod) {
        $method = $comparison.openFoamPressureLossMethod
        $lines.Add("")
        $lines.Add("OpenFOAM pressure loss sampling: ``$method`` with $($comparison.openFoamInletSamples) inlet and $($comparison.openFoamOutletSamples) outlet samples.")
    }
    $lines.Add("")
    $lines.Add("## Timing")
    $lines.Add("")
    $lines.Add("| Runner | Wall clock [s] | Solver execution time [s] | Steps |")
    $lines.Add("| --- | ---: | ---: | ---: |")
    $lines.Add("| Ferrum preflight | $(Format-NullableNumber $comparison.timing.ferrumPreflightWallClockSeconds "G6") | n/a | $($ferrum.runSchedule.estimatedSteps) planned |")
    if ($null -ne $openFoam) {
        $foamExecution = if ($null -ne $openFoam.foamTiming) { $openFoam.foamTiming.executionTimeSeconds } else { $null }
        $foamSteps = if ($null -ne $Result.openFoamRunControl) { $Result.openFoamRunControl.simulatedSteps } else { "n/a" }
        $lines.Add("| OpenFOAM simpleFoam | $(Format-NullableNumber $comparison.timing.openFoamWallClockSeconds "G6") | $(Format-NullableNumber $foamExecution "G6") | $foamSteps |")
    } else {
        $lines.Add("| OpenFOAM simpleFoam | n/a | n/a | n/a |")
    }
    $lines.Add("")
    $lines.Add("## Notes")
    $lines.Add("")
    foreach ($note in $status.notes) {
        $lines.Add("- $note")
    }

    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Path) | Out-Null
    Set-Content -LiteralPath $Path -Value $lines -Encoding UTF8
}

$analyticDeltaPPa = Read-AnalyticDeltaP $CaseRoot
$ferrumRun = Invoke-FerrumPreflight -CaseRoot $CaseRoot -PlanJson $FerrumPlanJson
$ferrumPlan = Get-Content -LiteralPath $FerrumPlanJson -Raw | ConvertFrom-Json
$openFoam = $null
if (Test-Path -LiteralPath $OpenFoamJson) {
    $openFoam = Get-Content -LiteralPath $OpenFoamJson -Raw | ConvertFrom-Json
    if ($null -ne $openFoam.analytic -and $null -ne $openFoam.analytic.deltaPPa) {
        $analyticDeltaPPa = [double]$openFoam.analytic.deltaPPa
    }
}

$openFoamDeltaPPa = $null
$openFoamRelativeError = $null
$openFoamWallClock = $null
if ($null -ne $openFoam -and $null -ne $openFoam.openFoam.pressureLoss) {
    $openFoamDeltaPPa = [double]$openFoam.openFoam.pressureLoss.deltaPPa
    $openFoamRelativeError = ($openFoamDeltaPPa - $analyticDeltaPPa) / $analyticDeltaPPa
}
if ($null -ne $openFoam) {
    $openFoamWallClock = $openFoam.openFoam.wallClockSeconds
}
$openFoamRunControl = if ($null -ne $openFoam) { $openFoam.runControl } else { $null }
$openFoamPressureLoss = if ($null -ne $openFoam) { $openFoam.openFoam.pressureLoss } else { $null }
$caseMesh = Read-PipeBenchmarkMesh $CaseRoot
$mesh = if ($null -ne $openFoam -and $null -ne $openFoam.mesh) {
    [ordered]@{
        type = "structuredCircularPipe"
        axialCells = $openFoam.mesh.axialCells
        radialCells = $openFoam.mesh.radialCells
        angularSectors = $openFoam.mesh.angularSectors
        cells = $openFoam.mesh.cells
    }
} elseif ($null -ne $caseMesh) {
    [ordered]@{
        type = $caseMesh.type
        axialCells = $caseMesh.axialCells
        radialCells = $caseMesh.radialCells
        angularSectors = $caseMesh.angularSectors
        cells = $caseMesh.cells
    }
} else {
    $null
}

$openFoamReferenceStatus = if ($null -eq $openFoam) {
    "missing"
} elseif ($openFoam.openFoam.available -eq $false) {
    "unavailable"
} elseif ($openFoam.openFoam.exitCode -eq 0) {
    "passed"
} else {
    "failed"
}

$notes = @(
    "The current laminar_pipe mesh is a generated structured circular pipe controlled by scripts/generate_laminar_pipe_case.ps1.",
    "OpenFOAM is generated only under target/openfoam/laminar_pipe for comparison and is not the default FerrumCFD workflow.",
    "OpenFOAM-to-analytic pressure-loss differences are treated as mesh/discretization/setup error at this stage.",
    "FerrumCFD currently contributes preflight timing and field-buffer readiness only; executable solver timing will be added when the flow solver exists."
)

$result = [ordered]@{
    case = "laminar_pipe"
    units = [ordered]@{
        default = "SI"
        length = "m"
        pressure = "Pa"
        temperature = "K"
        velocity = "m/s"
        openFoamPressure = "kinematic m2/s2 converted to Pa"
    }
    analytic = [ordered]@{
        pressureLossModel = "HagenPoiseuille"
        deltaPPa = $analyticDeltaPPa
    }
    mesh = $mesh
    ferrum = [ordered]@{
        mode = "preflight-no-solver"
        executableSolver = $false
        wallClockSeconds = $ferrumRun.wallClockSeconds
        planJson = $ferrumRun.planJson
        log = $ferrumRun.log
        runSchedule = [ordered]@{
            startTime = $ferrumPlan.run.startTime
            endTime = $ferrumPlan.run.endTime
            deltaT = $ferrumPlan.run.deltaT
            estimatedSteps = $ferrumPlan.run.estimatedSteps
            estimatedWrites = $ferrumPlan.run.estimatedWriteEvents
        }
        state = Get-StateSummary $ferrumPlan
    }
    openFoam = if ($null -ne $openFoam) { $openFoam.openFoam } else { $null }
    openFoamRunControl = $openFoamRunControl
    comparison = [ordered]@{
        openFoamDeltaPPa = $openFoamDeltaPPa
        openFoamRelativeErrorToAnalytic = $openFoamRelativeError
        openFoamPressureLossMethod = if ($null -ne $openFoamPressureLoss) { $openFoamPressureLoss.method } else { $null }
        openFoamInletSamples = if ($null -ne $openFoamPressureLoss) { $openFoamPressureLoss.inletSamples } else { $null }
        openFoamOutletSamples = if ($null -ne $openFoamPressureLoss) { $openFoamPressureLoss.outletSamples } else { $null }
        ferrumDeltaPPa = $null
        ferrumRelativeErrorToOpenFoam = $null
        ferrumSolverComparison = "pending executable FerrumCFD flow solver"
        timing = [ordered]@{
            ferrumPreflightWallClockSeconds = $ferrumRun.wallClockSeconds
            openFoamWallClockSeconds = $openFoamWallClock
        }
    }
    benchmarkStatus = [ordered]@{
        ferrumPreflight = "passed"
        openFoamReference = $openFoamReferenceStatus
        ferrumSolverComparison = "pending"
        readyForCiGate = $false
        notes = $notes
    }
}

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $OutFile) | Out-Null
$result | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $OutFile -Encoding UTF8
Write-MarkdownReport -Path $ReportFile -Result $result
Write-Output "wrote laminar pipe comparison: $OutFile"
Write-Output "wrote laminar pipe report: $ReportFile"
