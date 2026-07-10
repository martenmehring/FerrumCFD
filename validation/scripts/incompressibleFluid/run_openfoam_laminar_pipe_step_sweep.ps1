param(
    [string]$FerrumOverlayCaseRoot = "",
    [string]$StudyRoot = "",
    [string]$BenchmarkProperties = "",
    [ValidateSet("Auto", "Native", "Wsl")]
    [string]$Mode = "Auto",
    [string[]]$OpenFoamSteps = @("100", "200", "400", "800", "1200"),
    [double]$TargetRelativeError = 0.01,
    [switch]$RequireOpenFoam,
    [switch]$UseExistingReports
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))

if ([string]::IsNullOrWhiteSpace($StudyRoot)) {
    $StudyRoot = Join-Path $RepoRoot "target\benchmarks\openfoam_laminar_pipe_step_sweep"
}
if ([string]::IsNullOrWhiteSpace($BenchmarkProperties)) {
    $BenchmarkProperties = Join-Path $RepoRoot "tutorials\incompressibleFluid\laminarPipe\analytical\pipeBenchmark"
}
if (!(Test-Path -LiteralPath $BenchmarkProperties -PathType Leaf)) {
    throw "benchmark properties not found: $BenchmarkProperties"
}
if ($TargetRelativeError -le 0.0) {
    throw "TargetRelativeError must be positive"
}

$stepBudgets = New-Object System.Collections.Generic.List[int]
foreach ($rawValue in $OpenFoamSteps) {
    foreach ($part in ($rawValue -split ",")) {
        $trimmed = $part.Trim()
        if ([string]::IsNullOrWhiteSpace($trimmed)) {
            continue
        }
        $parsed = 0
        if (![int]::TryParse($trimmed, [System.Globalization.NumberStyles]::Integer, [System.Globalization.CultureInfo]::InvariantCulture, [ref]$parsed)) {
            throw "invalid OpenFoamSteps value '$trimmed'; expected a positive integer"
        }
        if ($parsed -le 0) {
            throw "OpenFoamSteps values must be positive"
        }
        $stepBudgets.Add($parsed) | Out-Null
    }
}
if ($stepBudgets.Count -eq 0) {
    throw "OpenFoamSteps must contain at least one positive step count"
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

function Read-JsonFile([string]$Path) {
    if (!(Test-Path -LiteralPath $Path)) {
        return $null
    }
    return Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
}

function New-StepRow([object]$Result, [int]$Steps, [string]$JsonPath, [string]$WorkDir) {
    $pressureLoss = $Result.openFoam.pressureLoss
    return [pscustomobject][ordered]@{
        steps = $Steps
        status = if ($Result.openFoam.exitCode -eq 0) { "passed" } else { "failed" }
        deltaPPa = $pressureLoss.deltaPPa
        relativeErrorToAnalytic = $pressureLoss.relativeErrorToAnalytic
        absoluteRelativeErrorToAnalytic = [Math]::Abs([double]$pressureLoss.relativeErrorToAnalytic)
        sampledDeltaPPa = $pressureLoss.sampledDeltaPPa
        effectiveLengthFraction = $pressureLoss.effectiveLengthFraction
        inletSamples = $pressureLoss.inletSamples
        outletSamples = $pressureLoss.outletSamples
        wallClockSeconds = $Result.openFoam.wallClockSeconds
        executionTimeSeconds = if ($null -ne $Result.openFoam.foamTiming) { $Result.openFoam.foamTiming.executionTimeSeconds } else { $null }
        clockTimeSeconds = if ($null -ne $Result.openFoam.foamTiming) { $Result.openFoam.foamTiming.clockTimeSeconds } else { $null }
        json = $JsonPath
        workDir = $WorkDir
    }
}

function Write-SweepMarkdown($Path, $Rows, $Summary) {
    $lines = New-Object System.Collections.Generic.List[string]
    $lines.Add("# OpenFOAM Laminar Pipe Step Sweep")
    $lines.Add("")
    $lines.Add("Case: ``$($Summary.caseDir)``")
    $lines.Add("")
    $lines.Add('This sweep answers how many steady OpenFOAM 13 `foamRun -solver incompressibleFluid` iterations are needed to reach the target pressure-loss error against Hagen-Poiseuille.')
    $lines.Add("")
    $lines.Add("| Steps | DeltaP [Pa] | Error to analytic | Execution [s] | Wall [s] | Meets target |")
    $lines.Add("| ---: | ---: | ---: | ---: | ---: | --- |")
    foreach ($row in $Rows) {
        $meets = [Math]::Abs([double]$row.relativeErrorToAnalytic) -le [double]$Summary.targetRelativeError
        $lines.Add("| $($row.steps) | $(Format-NullableNumber $row.deltaPPa "G8") | $(Format-NullablePercent $row.relativeErrorToAnalytic) | $(Format-NullableNumber $row.executionTimeSeconds "G6") | $(Format-NullableNumber $row.wallClockSeconds "G6") | $meets |")
    }
    $lines.Add("")
    $lines.Add("## Target")
    $lines.Add("")
    $lines.Add("| Quantity | Value |")
    $lines.Add("| --- | ---: |")
    $lines.Add("| Target relative error | $(Format-NullablePercent $Summary.targetRelativeError) |")
    if ($null -ne $Summary.firstMeetingTarget) {
        $lines.Add("| First meeting target | $($Summary.firstMeetingTarget.steps) steps |")
        $lines.Add("| First meeting target execution [s] | $(Format-NullableNumber $Summary.firstMeetingTarget.executionTimeSeconds "G6") |")
        $lines.Add("| First meeting target wall [s] | $(Format-NullableNumber $Summary.firstMeetingTarget.wallClockSeconds "G6") |")
    } else {
        $lines.Add("| First meeting target | n/a |")
    }
    $lines.Add("")
    $lines.Add("## Files")
    $lines.Add("")
    $lines.Add("- Summary JSON: ``$($Summary.summaryJson)``")
    $lines.Add("- Report: ``$($Summary.reportFile)``")
    $lines.Add("- Study root: ``$($Summary.studyRoot)``")

    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Path) | Out-Null
    Set-Content -LiteralPath $Path -Value $lines -Encoding UTF8
}

$runner = Join-Path $PSScriptRoot "run_openfoam_laminar_pipe.ps1"
$resultRoot = Join-Path $StudyRoot "results"
$caseRootOut = Join-Path $StudyRoot "cases"
New-Item -ItemType Directory -Force -Path $StudyRoot, $resultRoot, $caseRootOut | Out-Null

$rows = New-Object System.Collections.Generic.List[object]
foreach ($steps in $stepBudgets) {
    $name = "foamRun_incompressibleFluid_$steps"
    $workDir = Join-Path $caseRootOut $name
    $outFile = Join-Path $resultRoot "$name.json"
    if ($UseExistingReports -and (Test-Path -LiteralPath $outFile)) {
        Write-Output "using existing OpenFOAM report for $steps steps"
        $result = Read-JsonFile $outFile
    } else {
        Write-Output "running OpenFOAM 13 foamRun/incompressibleFluid for $steps steps"
        $args = @{
            WorkDir = $workDir
            OutFile = $outFile
            BenchmarkProperties = $BenchmarkProperties
            Mode = $Mode
            EndTime = $steps
            WriteInterval = $steps
        }
        if (![string]::IsNullOrWhiteSpace($FerrumOverlayCaseRoot)) {
            $args.FerrumOverlayCaseRoot = $FerrumOverlayCaseRoot
        }
        if ($RequireOpenFoam) {
            $args.RequireOpenFoam = $true
        }
        & $runner @args
        $result = Read-JsonFile $outFile
    }
    if ($null -eq $result) {
        throw "OpenFOAM run $steps did not write $outFile"
    }
    $rows.Add((New-StepRow -Result $result -Steps $steps -JsonPath $outFile -WorkDir $workDir)) | Out-Null
}

$rowArray = @($rows.ToArray() | Sort-Object steps)
$firstMeetingTarget = @($rowArray | Where-Object { $_.absoluteRelativeErrorToAnalytic -le $TargetRelativeError } | Select-Object -First 1)
if ($firstMeetingTarget.Count -gt 0) {
    $firstMeetingTarget = $firstMeetingTarget[0]
} else {
    $firstMeetingTarget = $null
}

$summaryJson = Join-Path $StudyRoot "openfoam_laminar_pipe_step_sweep.json"
$reportFile = Join-Path $StudyRoot "openfoam_laminar_pipe_step_sweep.md"
$summary = [ordered]@{
    caseDir = $CaseRoot
    benchmark = "openfoam-laminar-pipe-step-sweep"
    units = [ordered]@{
        default = "SI"
        pressure = "Pa"
        length = "m"
        velocity = "m/s"
    }
    generatedAt = (Get-Date).ToString("o", [System.Globalization.CultureInfo]::InvariantCulture)
    targetRelativeError = $TargetRelativeError
    benchmarkProperties = $BenchmarkProperties
    openFoamSteps = @($stepBudgets.ToArray())
    firstMeetingTarget = $firstMeetingTarget
    studyRoot = $StudyRoot
    resultRoot = $resultRoot
    summaryJson = $summaryJson
    reportFile = $reportFile
}

$payload = [ordered]@{
    summary = $summary
    rows = $rowArray
}

$payload | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $summaryJson -Encoding UTF8
Write-SweepMarkdown -Path $reportFile -Rows $rowArray -Summary ([pscustomobject]$summary)

if ($null -ne $firstMeetingTarget) {
    Write-Output ("first OpenFOAM run below target: {0} steps, error {1:P3}, execution {2:G6}s, wall {3:G6}s" -f $firstMeetingTarget.steps, $firstMeetingTarget.relativeErrorToAnalytic, $firstMeetingTarget.executionTimeSeconds, $firstMeetingTarget.wallClockSeconds)
} else {
    Write-Output "no OpenFOAM run met the target relative error"
}
Write-Output "OpenFOAM step sweep JSON: $summaryJson"
Write-Output "OpenFOAM step sweep report: $reportFile"
