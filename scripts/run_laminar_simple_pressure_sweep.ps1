param(
    [string]$StudyRoot = "",
    [string]$BenchmarkProperties = "",
    [string[]]$VariantName = @("medium", "fine"),
    [string[]]$SimpleIterations = @("50", "100", "200"),
    [switch]$WriteFinalFields,
    [switch]$UseExistingReports
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot

if ([string]::IsNullOrWhiteSpace($StudyRoot)) {
    $StudyRoot = Join-Path $RepoRoot "target\benchmarks\laminar_simple_pressure_sweep"
}
if ([string]::IsNullOrWhiteSpace($BenchmarkProperties)) {
    $BenchmarkProperties = Join-Path $RepoRoot "tutorials\steadyIncompressible\laminarPipe\analytical\pipeBenchmark"
}
if (!(Test-Path -LiteralPath $BenchmarkProperties -PathType Leaf)) {
    throw "benchmark properties not found: $BenchmarkProperties"
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

function ConvertTo-IterationBudgets([string[]]$RawValues) {
    $values = New-Object System.Collections.Generic.List[int]
    foreach ($rawValue in $RawValues) {
        foreach ($part in ($rawValue -split ",")) {
            $trimmed = $part.Trim()
            if ([string]::IsNullOrWhiteSpace($trimmed)) {
                continue
            }
            $parsed = 0
            if (![int]::TryParse($trimmed, [System.Globalization.NumberStyles]::Integer, [System.Globalization.CultureInfo]::InvariantCulture, [ref]$parsed)) {
                throw "invalid SimpleIterations value '$trimmed'; expected a positive integer"
            }
            if ($parsed -le 0) {
                throw "SimpleIterations values must be positive"
            }
            $values.Add($parsed) | Out-Null
        }
    }
    if ($values.Count -eq 0) {
        throw "SimpleIterations must contain at least one positive iteration count"
    }
    return @($values.ToArray())
}

function Write-PressureSweepMarkdown($Path, $Rows, $Summary) {
    $lines = New-Object System.Collections.Generic.List[string]
    $lines.Add("# Laminar SIMPLE Pressure-Field Sweep")
    $lines.Add("")
    $lines.Add('FerrumCFD-facing values are SI. This sweep fixes `minSimpleIterations=maxSimpleIterations` for each row and is Ferrum-only.')
    $lines.Add("")
    $lines.Add("| Variant | Cells | Iterations | p-owner deltaP [Pa] | p-owner error | mean-U deltaP [Pa] | mean-U error | mean U [m/s] | mean-U velocity error |")
    $lines.Add("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    foreach ($row in $Rows) {
        $lines.Add("| $($row.variant) | $(Format-NullableNumber $row.cells "G8") | $($row.iterationBudget) | $(Format-NullableNumber $row.pressureDropFromOwnerCellsPa "G8") | $(Format-NullablePercent $row.relativePressureDropErrorFromOwnerCells) | $(Format-NullableNumber $row.pressureDropFromMeanPa "G8") | $(Format-NullablePercent $row.relativePressureDropErrorFromMean) | $(Format-NullableNumber $row.meanVelocityMps "G8") | $(Format-NullablePercent $row.relativeMeanVelocityError) |")
    }
    $lines.Add("")
    $lines.Add("## Residuals And Timing")
    $lines.Add("")
    $lines.Add("| Variant | Iterations | Converged | Continuity L2 | Momentum normed | p-corr normed | Momentum linear ok | Pressure linear ok | Pressure nonconv solves | Max p iters/SIMPLE | Avg p iters/SIMPLE | Momentum linear iters | Pressure linear iters | Solve [s] |")
    $lines.Add("| --- | ---: | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    foreach ($row in $Rows) {
        $lines.Add("| $($row.variant) | $($row.iterationBudget) | $($row.converged) | $(Format-NullableNumber $row.finalContinuityL2 "G6") | $(Format-NullableNumber $row.momentumNormalizedResidualNorm "G6") | $(Format-NullableNumber $row.pressureCorrectionNormalizedResidualNorm "G6") | $($row.finalMomentumLinearConverged) | $($row.finalPressureLinearConverged) | $(Format-NullableNumber $row.pressureCorrectionNonConvergedSolves "G8") | $(Format-NullableNumber $row.maxPressureLinearIterationsPerSimple "G8") | $(Format-NullableNumber $row.averagePressureLinearIterationsPerSimple "G6") | $(Format-NullableNumber $row.totalMomentumLinearIterations "G8") | $(Format-NullableNumber $row.totalPressureLinearIterations "G8") | $(Format-NullableNumber $row.solverWallClockSeconds "G6") |")
    }
    $lines.Add("")
    $lines.Add("## Pressure Assembly Diagnostics")
    $lines.Add("")
    $lines.Add("| Variant | Iterations | rAU min/max | rAtU min/max | HbyA L2 | Source L2 | phiHbyA boundary before/after | pressureFlux boundary | correctedPhi boundary/abs |")
    $lines.Add("| --- | ---: | --- | --- | ---: | ---: | --- | ---: | --- |")
    foreach ($row in $Rows) {
        $lines.Add("| $($row.variant) | $($row.iterationBudget) | $(Format-NullableNumber $row.pressureAssemblyRAUMin "G6") / $(Format-NullableNumber $row.pressureAssemblyRAUMax "G6") | $(Format-NullableNumber $row.pressureAssemblyRAtUMin "G6") / $(Format-NullableNumber $row.pressureAssemblyRAtUMax "G6") | $(Format-NullableNumber $row.pressureAssemblyHbyAL2 "G6") | $(Format-NullableNumber $row.pressureAssemblySourceL2 "G6") | $(Format-NullableNumber $row.pressureAssemblyPhiHbyABoundaryBefore "G6") / $(Format-NullableNumber $row.pressureAssemblyPhiHbyABoundaryAfter "G6") | $(Format-NullableNumber $row.pressureAssemblyPressureFluxBoundary "G6") | $(Format-NullableNumber $row.pressureAssemblyCorrectedPhiBoundary "G6") / $(Format-NullableNumber $row.pressureAssemblyCorrectedPhiBoundaryAbs "G6") |")
    }
    $lines.Add("")
    $lines.Add("## Files")
    $lines.Add("")
    $lines.Add("- Summary JSON: ``$($Summary.summaryJson)``")
    $lines.Add("- This report: ``$($Summary.reportFile)``")
    $lines.Add("- Study root: ``$($Summary.studyRoot)``")
    $lines.Add("")
    $lines.Add("## Notes")
    $lines.Add("")
    $lines.Add('- p-owner is computed by the external pipe post-processor from the stored pressure field.')
    $lines.Add('- mean-U is reconstructed by the external post-processor using Hagen-Poiseuille; it is not part of SIMPLE convergence.')
    $lines.Add('- The generic SIMPLE JSON remains independent of pipe geometry and analytic reference values.')
    $lines.Add('- Solver wall-clock values are local measurements and should be interpreted together with linear-iteration counts when diagnosing performance.')

    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Path) | Out-Null
    Set-Content -LiteralPath $Path -Value $lines -Encoding UTF8
}

$targetRoot = Join-Path $RepoRoot "target"
if (!(Test-IsPathUnder $StudyRoot $targetRoot)) {
    throw "StudyRoot must be inside the repository target directory: $targetRoot"
}

$iterationBudgets = ConvertTo-IterationBudgets $SimpleIterations
$casesRoot = Join-Path $StudyRoot "cases"
$variantStudiesRoot = Join-Path $StudyRoot "variant_sweeps"
$logsRoot = Join-Path $StudyRoot "logs"
New-Item -ItemType Directory -Force -Path $StudyRoot, $casesRoot, $variantStudiesRoot, $logsRoot | Out-Null

$generator = Join-Path $PSScriptRoot "generate_laminar_pipe_case.ps1"
$iterationSweep = Join-Path $PSScriptRoot "run_laminar_simple_iteration_sweep.ps1"
$sourceSystem = Join-Path $RepoRoot "tutorials\steadyIncompressible\laminarPipe\ferrum\case\system"

$allVariants = @(
    [pscustomobject][ordered]@{ name = "coarse"; axialCells = 12; radialCells = 4; angularSectors = 24 },
    [pscustomobject][ordered]@{ name = "medium"; axialCells = 24; radialCells = 6; angularSectors = 32 },
    [pscustomobject][ordered]@{ name = "fine"; axialCells = 32; radialCells = 8; angularSectors = 48 }
)

$wanted = @{}
foreach ($rawName in $VariantName) {
    foreach ($part in ($rawName -split ",")) {
        $name = $part.Trim()
        if (![string]::IsNullOrWhiteSpace($name)) {
            $wanted[$name.ToLowerInvariant()] = $true
        }
    }
}
if ($wanted.Count -eq 0) {
    throw "VariantName must contain at least one variant"
}
$variants = @($allVariants | Where-Object { $wanted.ContainsKey($_.name.ToLowerInvariant()) })
if ($variants.Count -eq 0) {
    throw "none of the requested variants were found"
}

$rows = New-Object System.Collections.Generic.List[object]
foreach ($variant in $variants) {
    $caseRoot = Join-Path $casesRoot $variant.name
    $variantStudyRoot = Join-Path $variantStudiesRoot $variant.name
    $generateLog = Join-Path $logsRoot "$($variant.name).generate.log"

    if (!$UseExistingReports -or !(Test-Path -LiteralPath $caseRoot)) {
        Write-Output "variant $($variant.name): generating $($variant.axialCells)x$($variant.radialCells)x$($variant.angularSectors)"
        Remove-DirectoryIfExists $caseRoot $StudyRoot
        & $generator `
            -CaseRoot $caseRoot `
            -AxialCells $variant.axialCells `
            -RadialCells $variant.radialCells `
            -AngularSectors $variant.angularSectors *> $generateLog
        Copy-Item -LiteralPath $sourceSystem -Destination $caseRoot -Recurse -Force
    } else {
        Write-Output "variant $($variant.name): using existing generated case"
    }

    Write-Output "variant $($variant.name): running Ferrum pressure sweep"
    $sweepArgs = @{
        CaseRoot = $caseRoot
        StudyRoot = $variantStudyRoot
        BenchmarkProperties = $BenchmarkProperties
        SimpleIterations = ($iterationBudgets | ForEach-Object { $_.ToString([System.Globalization.CultureInfo]::InvariantCulture) })
        RunPipeBenchmark = $true
    }
    if ($WriteFinalFields) {
        $sweepArgs.WriteFinalFields = $true
    }
    if ($UseExistingReports) {
        $sweepArgs.UseExistingReports = $true
    }
    & $iterationSweep @sweepArgs

    $variantSummaryJson = Join-Path $variantStudyRoot "laminar_simple_iteration_sweep.json"
    $variantSummary = Read-JsonFile $variantSummaryJson
    if ($null -eq $variantSummary) {
        throw "missing variant sweep summary for $($variant.name): $variantSummaryJson"
    }
    foreach ($row in @($variantSummary.rows)) {
        $rows.Add([pscustomobject][ordered]@{
                variant = $variant.name
                cells = $row.meshCells
                axialCells = $variant.axialCells
                radialCells = $variant.radialCells
                angularSectors = $variant.angularSectors
                iterationBudget = $row.iterationBudget
                actualSimpleIterations = $row.actualSimpleIterations
                converged = $row.converged
                finalContinuityL2 = $row.finalContinuityL2
                momentumNormalizedResidualNorm = $row.momentumNormalizedResidualNorm
                pressureCorrectionNormalizedResidualNorm = $row.pressureCorrectionNormalizedResidualNorm
                pressureDropFromOwnerCellsPa = $row.pressureDropFromOwnerCellsPa
                relativePressureDropErrorFromOwnerCells = $row.relativePressureDropErrorFromOwnerCells
                pressureDropFromMeanPa = $row.pressureDropFromMeanPa
                relativePressureDropErrorFromMean = $row.relativePressureDropErrorFromMean
                meanVelocityMps = $row.meanVelocityMps
                relativeMeanVelocityError = $row.relativeMeanVelocityError
                totalMomentumLinearIterations = $row.totalMomentumLinearIterations
                totalPressureLinearIterations = $row.totalPressureLinearIterations
                finalMomentumLinearConverged = $row.finalMomentumLinearConverged
                finalPressureLinearConverged = $row.finalPressureLinearConverged
                momentumNonConvergedPredictors = $row.momentumNonConvergedPredictors
                momentumComponentNonConvergedSolves = $row.momentumComponentNonConvergedSolves
                pressureCorrectionSolves = $row.pressureCorrectionSolves
                pressureCorrectionNonConvergedSolves = $row.pressureCorrectionNonConvergedSolves
                maxMomentumLinearIterationsPerSimple = $row.maxMomentumLinearIterationsPerSimple
                maxPressureLinearIterationsPerSimple = $row.maxPressureLinearIterationsPerSimple
                averageMomentumLinearIterationsPerSimple = $row.averageMomentumLinearIterationsPerSimple
                averagePressureLinearIterationsPerSimple = $row.averagePressureLinearIterationsPerSimple
                solverWallClockSeconds = $row.solverWallClockSeconds
                commandWallClockSeconds = $row.commandWallClockSeconds
                pressureAssemblyRAUMin = $row.pressureAssemblyRAUMin
                pressureAssemblyRAUMax = $row.pressureAssemblyRAUMax
                pressureAssemblyRAtUMin = $row.pressureAssemblyRAtUMin
                pressureAssemblyRAtUMax = $row.pressureAssemblyRAtUMax
                pressureAssemblyHbyAL2 = $row.pressureAssemblyHbyAL2
                pressureAssemblySourceL2 = $row.pressureAssemblySourceL2
                pressureAssemblySourceSumAbs = $row.pressureAssemblySourceSumAbs
                pressureAssemblyPhiHbyABoundaryBefore = $row.pressureAssemblyPhiHbyABoundaryBefore
                pressureAssemblyPhiHbyABoundaryAfter = $row.pressureAssemblyPhiHbyABoundaryAfter
                pressureAssemblyPressureEquationFluxBoundary = $row.pressureAssemblyPressureEquationFluxBoundary
                pressureAssemblyPressureFluxBoundary = $row.pressureAssemblyPressureFluxBoundary
                pressureAssemblyCorrectedPhiBoundary = $row.pressureAssemblyCorrectedPhiBoundary
                pressureAssemblyCorrectedPhiBoundaryAbs = $row.pressureAssemblyCorrectedPhiBoundaryAbs
                pressureAssemblyCorrectedPhiSumAbs = $row.pressureAssemblyCorrectedPhiSumAbs
                reportJson = $row.reportJson
                reportMarkdown = $row.reportMarkdown
                log = $row.log
            }) | Out-Null
    }
}

$rowArray = @($rows.ToArray() | Sort-Object `
        @{ Expression = {
                if ($_.variant -eq "coarse") { 0 }
                elseif ($_.variant -eq "medium") { 1 }
                elseif ($_.variant -eq "fine") { 2 }
                else { 99 }
            } },
        iterationBudget)
$summaryJson = Join-Path $StudyRoot "laminar_simple_pressure_sweep.json"
$reportFile = Join-Path $StudyRoot "laminar_simple_pressure_sweep.md"
$summary = [pscustomobject][ordered]@{
    case = "laminar_pipe"
    generatedAt = (Get-Date).ToString("o", [System.Globalization.CultureInfo]::InvariantCulture)
    studyRoot = $StudyRoot
    variants = @($variants | ForEach-Object { $_.name })
    simpleIterations = $iterationBudgets
    writeFinalFields = $WriteFinalFields.IsPresent
    benchmarkProperties = $BenchmarkProperties
    summaryJson = $summaryJson
    reportFile = $reportFile
}

$payload = [ordered]@{
    summary = $summary
    rows = $rowArray
}
$payload | ConvertTo-Json -Depth 14 | Set-Content -LiteralPath $summaryJson -Encoding UTF8
Write-PressureSweepMarkdown -Path $reportFile -Rows $rowArray -Summary $summary

Write-Output "wrote laminar SIMPLE pressure sweep summary: $summaryJson"
Write-Output "wrote laminar SIMPLE pressure sweep report: $reportFile"
