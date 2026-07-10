param(
    [string]$StudyRoot = "",
    [string]$BenchmarkProperties = "",
    [ValidateSet("Auto", "Native", "Wsl")]
    [string]$Mode = "Auto",
    [string[]]$VariantName = @(),
    [int]$OpenFoamSteps = 200,
    [switch]$SkipOpenFoam,
    [switch]$RequireOpenFoam,
    [switch]$SkipFerrumSolve,
    [ValidateSet("jacobi", "cg")]
    [string]$FerrumLinearSolver = "cg",
    [double]$FerrumSolveTolerance = 1e-8,
    [int]$FerrumMaxIterations = 20000
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($StudyRoot)) {
    $StudyRoot = Join-Path $RepoRoot "target\benchmarks\laminar_pipe_convergence"
}
if ([string]::IsNullOrWhiteSpace($BenchmarkProperties)) {
    $BenchmarkProperties = Join-Path $RepoRoot "benchmarks\laminar_pipe\pipeBenchmark"
}
if (!(Test-Path -LiteralPath $BenchmarkProperties -PathType Leaf)) {
    throw "benchmark properties not found: $BenchmarkProperties"
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
    $lines.Add("# Laminar Pipe Mesh Convergence")
    $lines.Add("")
    $lines.Add('FerrumCFD-facing values are SI. OpenFOAM cases are generated only under `target/benchmarks/laminar_pipe_convergence` for benchmark comparison.')
    $lines.Add("")
    $lines.Add("OpenFOAM SIMPLE steps per variant: $($Summary.openFoamSteps)")
    $lines.Add("Ferrum linear solver: $($Summary.ferrumLinearSolver), tolerance: $($Summary.ferrumSolveTolerance), max iterations: $($Summary.ferrumMaxIterations)")
    $lines.Add("")
    $lines.Add("## Variants")
    $lines.Add("")
    $lines.Add("| Variant | Axial | Radial | Angular | Cells | Ferrum deltaP [Pa] | Ferrum error | Ferrum solve [s] | OpenFOAM deltaP [Pa] | OpenFOAM error | OpenFOAM wall [s] |")
    $lines.Add("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    foreach ($row in $Rows) {
        $lines.Add("| $($row.variant) | $($row.mesh.axialCells) | $($row.mesh.radialCells) | $($row.mesh.angularSectors) | $($row.mesh.cells) | $(Format-NullableNumber $row.ferrum.pressureDropFromMeanPa "G8") | $(Format-NullablePercent $row.ferrum.relativePressureDropErrorToAnalytic) | $(Format-NullableNumber $row.ferrum.solveWallClockSeconds "G6") | $(Format-NullableNumber $row.openFoam.deltaPPa "G8") | $(Format-NullablePercent $row.openFoam.relativeErrorToAnalytic) | $(Format-NullableNumber $row.openFoam.wallClockSeconds "G6") |")
    }
    $lines.Add("")
    $lines.Add("## Files")
    $lines.Add("")
    $lines.Add('- Summary JSON: `' + $Summary.summaryJson + '`')
    $lines.Add('- This report: `' + $Summary.reportFile + '`')
    $lines.Add("")
    $lines.Add("## Notes")
    $lines.Add("")
    $lines.Add('- `medium` matches the versioned `examples/laminar_pipe` default mesh resolution.')
    $lines.Add('- `Ferrum solve` is the executable source-driven axial Stokes/Poiseuille benchmark, not the later full SIMPLE-like flow solver.')
    $lines.Add('- Ferrum pressure loss is reconstructed from the solved mean velocity and compared to Hagen-Poiseuille.')
    $lines.Add('- OpenFOAM pressure is converted from kinematic pressure (`m2/s2`) back to SI pressure (`Pa`) before comparison.')
    $lines.Add('- Use `-SkipOpenFoam` for a quick Ferrum-only convergence preflight.')
    $lines.Add('- Increase `-OpenFoamSteps` when fine OpenFOAM residuals are still moving.')

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
$sourceSystem = Join-Path $RepoRoot "examples\laminar_pipe\system"

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

    Write-Output "variant $($variant.name): generating $($variant.axialCells)x$($variant.radialCells)x$($variant.angularSectors)"
    Remove-DirectoryIfExists $caseRoot $StudyRoot
    $generateElapsed = Measure-Command {
        & $generator `
            -CaseRoot $caseRoot `
            -AxialCells $variant.axialCells `
            -RadialCells $variant.radialCells `
            -AngularSectors $variant.angularSectors *> $generateLog
    }

    Copy-Item -LiteralPath $sourceSystem -Destination $caseRoot -Recurse -Force

    $openFoamElapsed = $null
    if ($SkipOpenFoam) {
        if (Test-Path -LiteralPath $openFoamJson) {
            Remove-Item -LiteralPath $openFoamJson -Force
        }
    } else {
        Write-Output "variant $($variant.name): running OpenFOAM reference"
        $openFoamArgs = @{
            CaseRoot = $caseRoot
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
        $openFoamElapsed = (Measure-Command { & $runOpenFoam @openFoamArgs *> $openFoamLog }).TotalSeconds
    }

    Write-Output "variant $($variant.name): running Ferrum/OpenFOAM comparison"
    $compareElapsed = Measure-Command {
        $compareArgs = @{
            CaseRoot = $caseRoot
            OpenFoamJson = $openFoamJson
            FerrumPlanJson = $planJson
            OutFile = $compareJson
            ReportFile = $compareReport
            BenchmarkProperties = $BenchmarkProperties
            FerrumLinearSolver = $FerrumLinearSolver
            FerrumSolveTolerance = $FerrumSolveTolerance
            FerrumMaxIterations = $FerrumMaxIterations
        }
        if ($SkipFerrumSolve) {
            $compareArgs.SkipFerrumSolve = $true
        }
        & $compare @compareArgs *> $compareLog
    }

    $comparison = Read-JsonFile $compareJson
    $openFoam = Read-JsonFile $openFoamJson
    $pressureLoss = if ($null -ne $openFoam -and $null -ne $openFoam.openFoam.pressureLoss) { $openFoam.openFoam.pressureLoss } else { $null }
    $ferrumSolve = if ($null -ne $comparison -and $null -ne $comparison.ferrum.solve) { $comparison.ferrum.solve } else { $null }
    $ferrumResult = if ($null -ne $ferrumSolve) { $ferrumSolve.result } else { $null }
    $mesh = if ($null -ne $comparison.mesh) { $comparison.mesh } else { [pscustomobject][ordered]@{
            type = "structuredCircularPipe"
            axialCells = $variant.axialCells
            radialCells = $variant.radialCells
            angularSectors = $variant.angularSectors
            cells = $variant.axialCells * $variant.radialCells * $variant.angularSectors
        } }

    $rows.Add([pscustomobject][ordered]@{
            variant = $variant.name
            caseRoot = $caseRoot
            mesh = $mesh
            analytic = $comparison.analytic
            ferrum = [pscustomobject][ordered]@{
                status = if ($null -ne $comparison) { $comparison.benchmarkStatus.ferrumSolverComparison } else { "missing" }
                pressureDropFromMeanPa = if ($null -ne $ferrumResult) { $ferrumResult.pressureDropFromMeanPa } else { $null }
                relativePressureDropErrorToAnalytic = if ($null -ne $ferrumResult) { $ferrumResult.relativePressureDropErrorToAnalytic } else { $null }
                meanVelocityMps = if ($null -ne $ferrumResult) { $ferrumResult.meanVelocityMps } else { $null }
                relativeMeanVelocityErrorToAnalytic = if ($null -ne $ferrumResult) { $ferrumResult.relativeMeanVelocityErrorToAnalytic } else { $null }
                solveWallClockSeconds = if ($null -ne $ferrumSolve) { $ferrumSolve.solveWallClockSeconds } else { $null }
                commandWallClockSeconds = if ($null -ne $ferrumSolve) { $ferrumSolve.commandWallClockSeconds } else { $null }
                iterations = if ($null -ne $ferrumSolve) { $ferrumSolve.iterations } else { $null }
                converged = if ($null -ne $ferrumSolve) { $ferrumSolve.converged } else { $null }
                residualNorm = if ($null -ne $ferrumSolve) { $ferrumSolve.residualNorm } else { $null }
                preflightWallClockSeconds = if ($null -ne $comparison) { $comparison.ferrum.preflight.wallClockSeconds } else { $null }
                wallClockSeconds = if ($null -ne $ferrumSolve) { $ferrumSolve.solveWallClockSeconds } else { $null }
                planJson = $planJson
                resultJson = $compareJson
                report = $compareReport
            }
            openFoam = [pscustomobject][ordered]@{
                status = if ($null -ne $comparison) { $comparison.benchmarkStatus.openFoamReference } else { "missing" }
                deltaPPa = if ($null -ne $pressureLoss) { $pressureLoss.deltaPPa } else { $null }
                relativeErrorToAnalytic = if ($null -ne $pressureLoss) { $pressureLoss.relativeErrorToAnalytic } else { $null }
                wallClockSeconds = if ($null -ne $openFoam) { $openFoam.openFoam.wallClockSeconds } else { $null }
                driverWallClockSeconds = $openFoamElapsed
                resultJson = $openFoamJson
            }
            timings = [pscustomobject][ordered]@{
                generateWallClockSeconds = $generateElapsed.TotalSeconds
                compareWallClockSeconds = $compareElapsed.TotalSeconds
            }
            logs = [pscustomobject][ordered]@{
                generate = $generateLog
                openFoamDriver = $openFoamLog
                compare = $compareLog
            }
        }) | Out-Null
}

$summaryJson = Join-Path $StudyRoot "laminar_pipe_convergence.json"
$reportFile = Join-Path $StudyRoot "laminar_pipe_convergence.md"
$openFoamMode = if ($SkipOpenFoam) { "skipped" } else { $Mode }
$openFoamStepCount = if ($SkipOpenFoam) { 0 } else { $OpenFoamSteps }
$generatedAt = Get-Date -Format "o"
$rowArray = @($rows.ToArray())
$summary = [pscustomobject][ordered]@{
    case = "laminar_pipe"
    generatedAt = $generatedAt
    openFoamMode = $openFoamMode
    openFoamSteps = $openFoamStepCount
    ferrumSolve = if ($SkipFerrumSolve) { "skipped" } else { "poiseuille" }
    ferrumLinearSolver = $FerrumLinearSolver
    ferrumSolveTolerance = $FerrumSolveTolerance
    ferrumMaxIterations = $FerrumMaxIterations
    benchmarkProperties = $BenchmarkProperties
    variants = $rowArray
    summaryJson = $summaryJson
    reportFile = $reportFile
}

$summary | ConvertTo-Json -Depth 14 | Set-Content -LiteralPath $summaryJson -Encoding UTF8
Write-StudyMarkdown -Path $reportFile -Rows $rowArray -Summary $summary

Write-Output "wrote laminar pipe convergence summary: $summaryJson"
Write-Output "wrote laminar pipe convergence report: $reportFile"
