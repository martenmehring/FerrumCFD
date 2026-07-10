param(
    [string]$StudyRoot = "",
    [string]$BenchmarkProperties = "",
    [ValidateSet("Auto", "Native", "Wsl")]
    [string]$Mode = "Auto",
    [string[]]$VariantName = @(),
    [int]$OpenFoamSteps = 400,
    [int]$FerrumSimpleIterations = 100,
    [switch]$SkipOpenFoam,
    [switch]$RequireOpenFoam,
    [switch]$SkipFerrumSolve,
    [switch]$UseExistingReports
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($StudyRoot)) {
    $StudyRoot = Join-Path $RepoRoot "target\benchmarks\laminar_simple_mesh_study"
}
if ([string]::IsNullOrWhiteSpace($BenchmarkProperties)) {
    $BenchmarkProperties = Join-Path $RepoRoot "tutorials\steadyIncompressible\laminarPipe\analytical\pipeBenchmark"
}
if (!(Test-Path -LiteralPath $BenchmarkProperties -PathType Leaf)) {
    throw "benchmark properties not found: $BenchmarkProperties"
}
if ($OpenFoamSteps -le 0) {
    throw "OpenFoamSteps must be positive"
}
if ($FerrumSimpleIterations -le 0) {
    throw "FerrumSimpleIterations must be positive"
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

function Get-FullPath([string]$Path) {
    return [System.IO.Path]::GetFullPath($Path)
}

function Test-IsPathUnder([string]$Child, [string]$Parent) {
    $childFull = Get-FullPath $Child
    $parentFull = (Get-FullPath $Parent).TrimEnd([System.IO.Path]::DirectorySeparatorChar, [System.IO.Path]::AltDirectorySeparatorChar)
    return $childFull.Equals($parentFull, [System.StringComparison]::OrdinalIgnoreCase) -or
        $childFull.StartsWith($parentFull + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase) -or
        $childFull.StartsWith($parentFull + [System.IO.Path]::AltDirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)
}

function Remove-DirectoryIfExists([string]$Path, [string]$AllowedRoot) {
    if (!(Test-Path -LiteralPath $Path)) {
        return
    }
    if (!(Test-IsPathUnder $Path $AllowedRoot)) {
        throw "refusing to remove '$Path' because it is outside '$AllowedRoot'"
    }
    Remove-Item -LiteralPath $Path -Recurse -Force
}

function Read-JsonFile([string]$Path) {
    if (!(Test-Path -LiteralPath $Path)) {
        return $null
    }
    return Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
}

function Write-StudyMarkdown($Path, $Rows, $Summary) {
    $lines = New-Object System.Collections.Generic.List[string]
    $lines.Add("# Laminar SIMPLE Mesh Study")
    $lines.Add("")
    $lines.Add("FerrumCFD-facing values are SI. Cases are generated under ``$($Summary.studyRoot)`` and OpenFOAM cases are benchmark artifacts only.")
    $lines.Add("")
    $lines.Add("| Quantity | Value |")
    $lines.Add("| --- | ---: |")
    $lines.Add("| Ferrum SIMPLE iterations | $($Summary.ferrumSimpleIterations) |")
    $lines.Add("| OpenFOAM SIMPLE steps | $($Summary.openFoamSteps) |")
    $lines.Add("")
    $lines.Add("## Pressure Loss")
    $lines.Add("")
    $lines.Add("| Variant | Cells | Ferrum p-owner deltaP [Pa] | Ferrum p-owner error | Ferrum mean-U deltaP [Pa] | Ferrum mean-U error | OpenFOAM deltaP [Pa] | OpenFOAM error |")
    $lines.Add("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    foreach ($row in $Rows) {
        $lines.Add("| $($row.variant) | $(Format-NullableNumber $row.mesh.cells "G8") | $(Format-NullableNumber $row.ferrum.pressureDropFromOwnerCellsPa "G8") | $(Format-NullablePercent $row.ferrum.relativePressureDropFromOwnerCellsErrorToAnalytic) | $(Format-NullableNumber $row.ferrum.pressureDropFromMeanPa "G8") | $(Format-NullablePercent $row.ferrum.relativePressureDropFromMeanErrorToAnalytic) | $(Format-NullableNumber $row.openFoam.deltaPPa "G8") | $(Format-NullablePercent $row.openFoam.relativeErrorToAnalytic) |")
    }
    $lines.Add("")
    $lines.Add("## Solver And Timing")
    $lines.Add("")
    $lines.Add("| Variant | Cells | Ferrum iterations | Ferrum converged | Continuity L2 | Momentum normed | p-corr normed | Ferrum solve [s] | OpenFOAM execution [s] | OpenFOAM wall [s] |")
    $lines.Add("| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    foreach ($row in $Rows) {
        $lines.Add("| $($row.variant) | $(Format-NullableNumber $row.mesh.cells "G8") | $(Format-NullableNumber $row.ferrum.iterations "G8") | $($row.ferrum.converged) | $(Format-NullableNumber $row.ferrum.finalContinuityL2 "G6") | $(Format-NullableNumber $row.ferrum.normalizedResidualNorm "G6") | $(Format-NullableNumber $row.ferrum.pressureCorrectionNormalizedResidualNorm "G6") | $(Format-NullableNumber $row.ferrum.solveWallClockSeconds "G6") | $(Format-NullableNumber $row.openFoam.executionTimeSeconds "G6") | $(Format-NullableNumber $row.openFoam.wallClockSeconds "G6") |")
    }
    $lines.Add("")
    $lines.Add("## Files")
    $lines.Add("")
    $lines.Add("- Summary JSON: ``$($Summary.summaryJson)``")
    $lines.Add("- This report: ``$($Summary.reportFile)``")
    $lines.Add("")
    $lines.Add("## Notes")
    $lines.Add("")
    $lines.Add('- Ferrum p-owner deltaP is the direct stored-pressure-field comparison using cells adjacent to inlet/outlet patches.')
    $lines.Add('- Ferrum mean-U deltaP is a Hagen-Poiseuille diagnostic reconstructed from solved mean velocity.')
    $lines.Add('- OpenFOAM pressure is converted from kinematic pressure (`m2/s2`) back to SI pressure (`Pa`) with `rho`.')
    $lines.Add('- Use `-VariantName coarse,medium` while iterating quickly; add `fine` for the full study.')

    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Path) | Out-Null
    Set-Content -LiteralPath $Path -Value $lines -Encoding UTF8
}

$targetRoot = Join-Path $RepoRoot "target"
if (!(Test-IsPathUnder $StudyRoot $targetRoot)) {
    throw "StudyRoot must be inside the repository target directory: $targetRoot"
}

$casesRoot = Join-Path $StudyRoot "cases"
$resultsRoot = Join-Path $StudyRoot "results"
$openFoamRoot = Join-Path $StudyRoot "openfoam"
$logsRoot = Join-Path $StudyRoot "logs"
New-Item -ItemType Directory -Force -Path $casesRoot, $resultsRoot, $openFoamRoot, $logsRoot | Out-Null

$generator = Join-Path $PSScriptRoot "generate_laminar_pipe_case.ps1"
$runOpenFoam = Join-Path $PSScriptRoot "run_openfoam_laminar_pipe.ps1"
$compare = Join-Path $PSScriptRoot "compare_laminar_pipe.ps1"
$sourceSystem = Join-Path $RepoRoot "tutorials\steadyIncompressible\laminarPipe\ferrum\case\system"

$variants = @(
    [pscustomobject][ordered]@{ name = "coarse"; axialCells = 12; radialCells = 4; angularSectors = 24 },
    [pscustomobject][ordered]@{ name = "medium"; axialCells = 24; radialCells = 6; angularSectors = 32 },
    [pscustomobject][ordered]@{ name = "fine"; axialCells = 32; radialCells = 8; angularSectors = 48 }
)

if ($VariantName.Count -gt 0) {
    $wanted = @{}
    foreach ($name in $VariantName) {
        $wanted[$name.ToLowerInvariant()] = $true
    }
    $variants = @($variants | Where-Object { $wanted.ContainsKey($_.name.ToLowerInvariant()) })
    if ($variants.Count -eq 0) {
        throw "none of the requested variants were found"
    }
}

$rows = New-Object System.Collections.Generic.List[object]
foreach ($variant in $variants) {
    $caseRoot = Join-Path $casesRoot $variant.name
    $openFoamWorkDir = Join-Path $openFoamRoot $variant.name
    $openFoamJson = Join-Path $resultsRoot "$($variant.name).openfoam.json"
    $compareJson = Join-Path $resultsRoot "$($variant.name).compare.json"
    $compareReport = Join-Path $resultsRoot "$($variant.name).compare.md"
    $planJson = Join-Path $resultsRoot "$($variant.name).ferrum_plan.json"
    $generateLog = Join-Path $logsRoot "$($variant.name).generate.log"
    $openFoamLog = Join-Path $logsRoot "$($variant.name).openfoam_driver.log"
    $compareLog = Join-Path $logsRoot "$($variant.name).compare.log"

    $comparison = if ($UseExistingReports) { Read-JsonFile $compareJson } else { $null }
    if ($null -eq $comparison) {
        Write-Output "variant $($variant.name): generating $($variant.axialCells)x$($variant.radialCells)x$($variant.angularSectors)"
        Remove-DirectoryIfExists $caseRoot $StudyRoot
        & $generator `
            -CaseRoot $caseRoot `
            -AxialCells $variant.axialCells `
            -RadialCells $variant.radialCells `
            -AngularSectors $variant.angularSectors *> $generateLog

        Copy-Item -LiteralPath $sourceSystem -Destination $caseRoot -Recurse -Force

        if ($SkipOpenFoam) {
            if (Test-Path -LiteralPath $openFoamJson) {
                Remove-Item -LiteralPath $openFoamJson -Force
            }
        } else {
            Write-Output "variant $($variant.name): running OpenFOAM reference"
            $openFoamArgs = @{
                FerrumOverlayCaseRoot = $caseRoot
                WorkDir = $openFoamWorkDir
                OutFile = $openFoamJson
                BenchmarkProperties = $BenchmarkProperties
                Mode = $Mode
                EndTime = $OpenFoamSteps
                WriteInterval = $OpenFoamSteps
            }
            if ($RequireOpenFoam) {
                $openFoamArgs.RequireOpenFoam = $true
            }
            & $runOpenFoam @openFoamArgs *> $openFoamLog
        }

        Write-Output "variant $($variant.name): running Ferrum laminar SIMPLE comparison"
        $compareArgs = @{
            CaseRoot = $caseRoot
            OpenFoamJson = $openFoamJson
            FerrumPlanJson = $planJson
            OutFile = $compareJson
            ReportFile = $compareReport
            BenchmarkProperties = $BenchmarkProperties
            FerrumSolver = "laminarSimple"
            FerrumSimpleIterations = $FerrumSimpleIterations
        }
        if ($SkipFerrumSolve) {
            $compareArgs.SkipFerrumSolve = $true
        }
        & $compare @compareArgs *> $compareLog
        $comparison = Read-JsonFile $compareJson
    } else {
        Write-Output "variant $($variant.name): using existing comparison report"
    }

    if ($null -eq $comparison) {
        throw "missing comparison report for variant $($variant.name): $compareJson"
    }

    $openFoam = Read-JsonFile $openFoamJson
    $pressureLoss = if ($null -ne $openFoam -and $null -ne $openFoam.openFoam.pressureLoss) { $openFoam.openFoam.pressureLoss } else { $null }
    $foamTiming = if ($null -ne $openFoam -and $null -ne $openFoam.openFoam.foamTiming) { $openFoam.openFoam.foamTiming } else { $null }
    $ferrumSolve = $comparison.ferrum.solve

    $rows.Add([pscustomobject][ordered]@{
            variant = $variant.name
            caseRoot = $caseRoot
            mesh = $comparison.mesh
            analytic = $comparison.analytic
            ferrum = [pscustomobject][ordered]@{
                status = $comparison.benchmarkStatus.ferrumSolverComparison
                pressureDropFromOwnerCellsPa = $comparison.comparison.ferrumPressureDropFromOwnerCellsPa
                relativePressureDropFromOwnerCellsErrorToAnalytic = if ($null -ne $comparison.comparison.ferrumPressureDropFromOwnerCellsPa -and $comparison.analytic.deltaPPa -ne 0.0) {
                    (($comparison.comparison.ferrumPressureDropFromOwnerCellsPa - $comparison.analytic.deltaPPa) / $comparison.analytic.deltaPPa)
                } else {
                    $null
                }
                pressureDropFromMeanPa = $comparison.comparison.ferrumPressureDropFromMeanPa
                relativePressureDropFromMeanErrorToAnalytic = $comparison.comparison.ferrumPressureDropFromMeanRelativeErrorToAnalytic
                meanVelocityMps = $comparison.comparison.ferrumMeanVelocityMps
                relativeMeanVelocityErrorToAnalytic = $comparison.comparison.ferrumMeanVelocityRelativeErrorToAnalytic
                solveWallClockSeconds = if ($null -ne $ferrumSolve) { $ferrumSolve.solveWallClockSeconds } else { $null }
                commandWallClockSeconds = if ($null -ne $ferrumSolve) { $ferrumSolve.commandWallClockSeconds } else { $null }
                iterations = if ($null -ne $ferrumSolve) { $ferrumSolve.simpleIterations } else { $null }
                converged = if ($null -ne $ferrumSolve) { $ferrumSolve.converged } else { $null }
                finalContinuityL2 = if ($null -ne $ferrumSolve) { $ferrumSolve.finalContinuityL2 } else { $null }
                normalizedResidualNorm = if ($null -ne $ferrumSolve) { $ferrumSolve.normalizedResidualNorm } else { $null }
                pressureCorrectionNormalizedResidualNorm = if ($null -ne $ferrumSolve) { $ferrumSolve.pressureCorrectionNormalizedResidualNorm } else { $null }
                resultJson = $compareJson
                report = $compareReport
            }
            openFoam = [pscustomobject][ordered]@{
                status = $comparison.benchmarkStatus.openFoamReference
                deltaPPa = if ($null -ne $pressureLoss) { $pressureLoss.deltaPPa } else { $null }
                relativeErrorToAnalytic = if ($null -ne $pressureLoss) { $pressureLoss.relativeErrorToAnalytic } else { $null }
                executionTimeSeconds = if ($null -ne $foamTiming) { $foamTiming.executionTimeSeconds } else { $null }
                wallClockSeconds = if ($null -ne $openFoam) { $openFoam.openFoam.wallClockSeconds } else { $null }
                resultJson = $openFoamJson
            }
            logs = [pscustomobject][ordered]@{
                generate = $generateLog
                openFoamDriver = $openFoamLog
                compare = $compareLog
            }
        }) | Out-Null
}

$summaryJson = Join-Path $StudyRoot "laminar_simple_mesh_study.json"
$reportFile = Join-Path $StudyRoot "laminar_simple_mesh_study.md"
$rowArray = @($rows.ToArray())
$summary = [pscustomobject][ordered]@{
    case = "laminar_pipe"
    generatedAt = (Get-Date).ToString("o", [System.Globalization.CultureInfo]::InvariantCulture)
    studyRoot = $StudyRoot
    openFoamMode = if ($SkipOpenFoam) { "skipped" } else { $Mode }
    openFoamSteps = if ($SkipOpenFoam) { 0 } else { $OpenFoamSteps }
    ferrumSolver = if ($SkipFerrumSolve) { "skipped" } else { "laminarSimple" }
    ferrumSimpleIterations = $FerrumSimpleIterations
    benchmarkProperties = $BenchmarkProperties
    variants = $rowArray
    summaryJson = $summaryJson
    reportFile = $reportFile
}

$summary | ConvertTo-Json -Depth 14 | Set-Content -LiteralPath $summaryJson -Encoding UTF8
Write-StudyMarkdown -Path $reportFile -Rows $rowArray -Summary $summary

Write-Output "wrote laminar SIMPLE mesh-study summary: $summaryJson"
Write-Output "wrote laminar SIMPLE mesh-study report: $reportFile"
