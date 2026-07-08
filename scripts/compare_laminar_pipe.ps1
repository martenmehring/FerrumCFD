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

function Write-MarkdownReport($Path, $Result) {
    $comparison = $Result.comparison
    $ferrum = $Result.ferrum
    $openFoam = $Result.openFoam
    $status = $Result.benchmarkStatus
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
    $lines.Add("## Pressure Loss")
    $lines.Add("")
    $lines.Add("| Source | deltaP [Pa] | Relative error to analytic |")
    $lines.Add("| --- | ---: | ---: |")
    $lines.Add("| Analytic Hagen-Poiseuille | $(Format-NullableNumber $Result.analytic.deltaPPa "G8") | 0% |")
    $lines.Add("| OpenFOAM simpleFoam | $(Format-NullableNumber $comparison.openFoamDeltaPPa "G8") | $(Format-NullablePercent $comparison.openFoamRelativeErrorToAnalytic) |")
    $lines.Add("| FerrumCFD solver | n/a | n/a |")
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

$analyticDeltaPPa = 1.6032
$ferrumRun = Invoke-FerrumPreflight -CaseRoot $CaseRoot -PlanJson $FerrumPlanJson
$ferrumPlan = Get-Content -LiteralPath $FerrumPlanJson -Raw | ConvertFrom-Json
$openFoam = $null
if (Test-Path -LiteralPath $OpenFoamJson) {
    $openFoam = Get-Content -LiteralPath $OpenFoamJson -Raw | ConvertFrom-Json
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

$notes = @(
    "The current laminar_pipe mesh is a very coarse square surrogate, not a resolved circular pipe.",
    "OpenFOAM-to-analytic pressure-loss differences are treated as mesh/model error at this stage.",
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
        openFoamReference = if ($null -ne $openFoam -and $openFoam.openFoam.exitCode -eq 0) { "passed" } elseif ($null -ne $openFoam) { "failed" } else { "missing" }
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
