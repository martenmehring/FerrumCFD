param(
    [string]$CaseRoot = "",
    [string]$OpenFoamJson = "",
    [string]$FerrumPlanJson = "",
    [string]$OutFile = ""
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
}

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $OutFile) | Out-Null
$result | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $OutFile -Encoding UTF8
Write-Output "wrote laminar pipe comparison: $OutFile"
