param(
    [string]$CaseRoot = "",
    [string]$OpenFoamJson = "",
    [string]$FerrumPlanJson = "",
    [string]$OutFile = "",
    [string]$ReportFile = "",
    [string]$BenchmarkProperties = "",
    [switch]$SkipFerrumSolve,
    [ValidateSet("poiseuille", "laminarSimple")]
    [string]$FerrumSolver = "poiseuille",
    [ValidateSet("jacobi", "cg")]
    [string]$FerrumLinearSolver = "cg",
    [double]$FerrumSolveTolerance = 1e-8,
    [int]$FerrumMaxIterations = 20000,
    [int]$FerrumSimpleIterations = 100
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($CaseRoot)) {
    $CaseRoot = Join-Path $RepoRoot "tutorials\steadyIncompressible\laminarPipe\ferrum\case"
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
if ([string]::IsNullOrWhiteSpace($BenchmarkProperties)) {
    $BenchmarkProperties = Join-Path $RepoRoot "tutorials\steadyIncompressible\laminarPipe\analytical\pipeBenchmark"
}
if (!(Test-Path -LiteralPath $BenchmarkProperties)) {
    throw "benchmark properties not found: $BenchmarkProperties"
}
if ($FerrumSolveTolerance -le 0.0) {
    throw "FerrumSolveTolerance must be positive"
}
if ($FerrumMaxIterations -le 0) {
    throw "FerrumMaxIterations must be positive"
}
if ($FerrumSimpleIterations -le 0) {
    throw "FerrumSimpleIterations must be positive"
}

function Invoke-FerrumPreflight([string]$CaseRoot, [string]$PlanJson) {
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $PlanJson) | Out-Null
    $planBaseName = [System.IO.Path]::GetFileNameWithoutExtension($PlanJson)
    $logPath = Join-Path (Split-Path -Parent $PlanJson) "$planBaseName.preflight.log"
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
        Push-Location $RepoRoot
        try {
            & $command @arguments *> $logPath
            $script:ferrumExitCode = $LASTEXITCODE
        } finally {
            Pop-Location
        }
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

function Parse-KeyValueLine([string]$Line) {
    $result = @{}
    foreach ($match in [regex]::Matches($Line, "([A-Za-z][A-Za-z0-9]*)=([^\s]+)")) {
        $result[$match.Groups[1].Value] = $match.Groups[2].Value
    }
    return $result
}

function ConvertTo-NullableDouble($Value) {
    if ($null -eq $Value -or [string]::IsNullOrWhiteSpace([string]$Value)) {
        return $null
    }
    return [double]::Parse([string]$Value, [System.Globalization.CultureInfo]::InvariantCulture)
}

function ConvertTo-NullableInt($Value) {
    if ($null -eq $Value -or [string]::IsNullOrWhiteSpace([string]$Value)) {
        return $null
    }
    return [int]::Parse([string]$Value, [System.Globalization.CultureInfo]::InvariantCulture)
}

function ConvertTo-NullableBool($Value) {
    if ($null -eq $Value -or [string]::IsNullOrWhiteSpace([string]$Value)) {
        return $null
    }
    $text = ([string]$Value).ToLowerInvariant()
    if ($text -eq "true") { return $true }
    if ($text -eq "false") { return $false }
    if ($text -eq "yes") { return $true }
    if ($text -eq "no") { return $false }
    return $null
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

function Invoke-FerrumPoiseuilleSolve(
    [string]$CaseRoot,
    [string]$LogPath,
    [string]$LinearSolver,
    [double]$SolveTolerance,
    [int]$MaxIterations,
    [double]$AnalyticDeltaPPa,
    $Physics
) {
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $LogPath) | Out-Null
    $invariant = [System.Globalization.CultureInfo]::InvariantCulture
    $exe = Join-Path $RepoRoot "target\debug\ferrumSolver.exe"
    if (Test-Path -LiteralPath $exe) {
        $command = $exe
        $arguments = @(
            "-case", $CaseRoot,
            "--solvePoiseuille",
            "--pressureDrop", ([double]$Physics.analyticDeltaPPa).ToString("G17", $invariant),
            "--mu", ([double]$Physics.dynamicViscosityPaS).ToString("G17", $invariant),
            "--length", ([double]$Physics.lengthM).ToString("G17", $invariant),
            "--diameter", ([double]$Physics.diameterM).ToString("G17", $invariant),
            "--linearSolver", $LinearSolver,
            "--solveTolerance", $SolveTolerance.ToString("G17", [System.Globalization.CultureInfo]::InvariantCulture),
            "--maxIterations", $MaxIterations.ToString([System.Globalization.CultureInfo]::InvariantCulture)
        )
    } else {
        $command = "cargo"
        $arguments = @(
            "run", "-p", "ferrum-cli", "--bin", "ferrumSolver", "--",
            "-case", $CaseRoot,
            "--solvePoiseuille",
            "--pressureDrop", ([double]$Physics.analyticDeltaPPa).ToString("G17", $invariant),
            "--mu", ([double]$Physics.dynamicViscosityPaS).ToString("G17", $invariant),
            "--length", ([double]$Physics.lengthM).ToString("G17", $invariant),
            "--diameter", ([double]$Physics.diameterM).ToString("G17", $invariant),
            "--linearSolver", $LinearSolver,
            "--solveTolerance", $SolveTolerance.ToString("G17", [System.Globalization.CultureInfo]::InvariantCulture),
            "--maxIterations", $MaxIterations.ToString([System.Globalization.CultureInfo]::InvariantCulture)
        )
    }

    $script:ferrumSolveExitCode = $null
    $elapsed = Measure-Command {
        Push-Location $RepoRoot
        try {
            & $command @arguments *> $LogPath
            $script:ferrumSolveExitCode = $LASTEXITCODE
        } finally {
            Pop-Location
        }
    }
    $exitCode = if ($null -eq $script:ferrumSolveExitCode) { 0 } else { $script:ferrumSolveExitCode }
    if ($exitCode -ne 0) {
        throw "Ferrum Poiseuille solve failed with exit code $exitCode. See $LogPath"
    }

    $output = Get-Content -LiteralPath $LogPath
    $solveLine = @($output | Where-Object { $_ -match "^poiseuille solve:" } | Select-Object -Last 1)
    $resultLine = @($output | Where-Object { $_ -match "^poiseuille result:" } | Select-Object -Last 1)
    if ($solveLine.Count -eq 0 -or $resultLine.Count -eq 0) {
        throw "Ferrum Poiseuille solve did not print expected result lines. See $LogPath"
    }

    $solve = Parse-KeyValueLine $solveLine[0]
    $result = Parse-KeyValueLine $resultLine[0]
    $pressureDropFromMean = ConvertTo-NullableDouble $result["pressureDropFromMean"]
    $pressureDropRelativeError = if ($null -ne $pressureDropFromMean -and $AnalyticDeltaPPa -ne 0.0) {
        ($pressureDropFromMean - $AnalyticDeltaPPa) / $AnalyticDeltaPPa
    } else {
        $null
    }

    return [ordered]@{
        mode = "source-driven-axial-stokes"
        executableSolver = $true
        backend = $solve["backend"]
        linearSolver = $solve["linearSolver"]
        command = Format-CommandLine -Command $command -Arguments $arguments
        log = $LogPath
        exitCode = $exitCode
        commandWallClockSeconds = $elapsed.TotalSeconds
        solveWallClockSeconds = ConvertTo-NullableDouble $solve["wallClockSeconds"]
        cells = ConvertTo-NullableInt $solve["cells"]
        nnz = ConvertTo-NullableInt $solve["nnz"]
        iterations = ConvertTo-NullableInt $solve["iterations"]
        converged = ConvertTo-NullableBool $solve["converged"]
        residualNorm = ConvertTo-NullableDouble $solve["residualNorm"]
        inputs = [ordered]@{
            pressureDropPa = ConvertTo-NullableDouble $solve["pressureDrop"]
            dynamicViscosityPaS = ConvertTo-NullableDouble $solve["dynamicViscosity"]
            lengthM = ConvertTo-NullableDouble $solve["length"]
            diameterM = ConvertTo-NullableDouble $solve["diameter"]
            source = ConvertTo-NullableDouble $solve["source"]
            wallPatches = $solve["wallPatches"]
        }
        result = [ordered]@{
            meanVelocityMps = ConvertTo-NullableDouble $result["meanVelocity"]
            analyticMeanVelocityMps = ConvertTo-NullableDouble $result["analyticMeanVelocity"]
            relativeMeanVelocityErrorToAnalytic = ConvertTo-NullableDouble $result["relativeMeanVelocityError"]
            flowRateM3s = ConvertTo-NullableDouble $result["flowRate"]
            analyticFlowRateM3s = ConvertTo-NullableDouble $result["analyticFlowRate"]
            pressureDropFromMeanPa = $pressureDropFromMean
            relativePressureDropErrorToAnalytic = $pressureDropRelativeError
            minVelocityMps = ConvertTo-NullableDouble $result["minVelocity"]
            maxVelocityMps = ConvertTo-NullableDouble $result["maxVelocity"]
        }
    }
}

function Invoke-FerrumLaminarSimpleSolve(
    [string]$CaseRoot,
    [string]$LogPath,
    [string]$ReportJson,
    [string]$ReportMarkdown,
    [string]$FieldsDir,
    [string]$BenchmarkJson,
    [int]$SimpleIterations,
    [double]$AnalyticDeltaPPa,
    $Physics
) {
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $LogPath) | Out-Null
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $ReportJson) | Out-Null
    New-Item -ItemType Directory -Force -Path $FieldsDir | Out-Null
    $exe = Join-Path $RepoRoot "target\debug\ferrumSolver.exe"
    if (Test-Path -LiteralPath $exe) {
        $command = $exe
        $arguments = @(
            "-case", $CaseRoot,
            "--solveLaminarSimple",
            "--minSimpleIterations", $SimpleIterations.ToString([System.Globalization.CultureInfo]::InvariantCulture),
            "--maxSimpleIterations", $SimpleIterations.ToString([System.Globalization.CultureInfo]::InvariantCulture),
            "--solveReportJson", $ReportJson,
            "--solveReportMarkdown", $ReportMarkdown,
            "--writeFinalFields", $FieldsDir
        )
    } else {
        $command = "cargo"
        $arguments = @(
            "run", "-p", "ferrum-cli", "--bin", "ferrumSolver", "--",
            "-case", $CaseRoot,
            "--solveLaminarSimple",
            "--minSimpleIterations", $SimpleIterations.ToString([System.Globalization.CultureInfo]::InvariantCulture),
            "--maxSimpleIterations", $SimpleIterations.ToString([System.Globalization.CultureInfo]::InvariantCulture),
            "--solveReportJson", $ReportJson,
            "--solveReportMarkdown", $ReportMarkdown,
            "--writeFinalFields", $FieldsDir
        )
    }

    $script:ferrumSimpleExitCode = $null
    $elapsed = Measure-Command {
        Push-Location $RepoRoot
        try {
            & $command @arguments *> $LogPath
            $script:ferrumSimpleExitCode = $LASTEXITCODE
        } finally {
            Pop-Location
        }
    }
    $exitCode = if ($null -eq $script:ferrumSimpleExitCode) { 0 } else { $script:ferrumSimpleExitCode }
    if ($exitCode -ne 0) {
        throw "Ferrum laminar SIMPLE solve failed with exit code $exitCode. See $LogPath"
    }
    if (!(Test-Path -LiteralPath $ReportJson)) {
        throw "Ferrum laminar SIMPLE solve did not write report JSON. See $LogPath"
    }

    $report = Get-Content -LiteralPath $ReportJson -Raw | ConvertFrom-Json
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $BenchmarkJson) | Out-Null
    $benchmarkLog = "$BenchmarkJson.log"
    $benchmarkExe = Join-Path $RepoRoot "target\debug\ferrumPipeBenchmark.exe"
    $invariant = [System.Globalization.CultureInfo]::InvariantCulture
    $benchmarkArguments = @(
        "-case", $CaseRoot,
        "--fields", $FieldsDir,
        "--pressureDrop", ([double]$Physics.analyticDeltaPPa).ToString("G17", $invariant),
        "--mu", ([double]$Physics.dynamicViscosityPaS).ToString("G17", $invariant),
        "--length", ([double]$Physics.lengthM).ToString("G17", $invariant),
        "--diameter", ([double]$Physics.diameterM).ToString("G17", $invariant),
        "--axis", "x",
        "--inletPatch", "inlet",
        "--outletPatch", "outlet",
        "--outJson", $BenchmarkJson
    )
    if (Test-Path -LiteralPath $benchmarkExe) {
        $benchmarkCommand = $benchmarkExe
    } else {
        $benchmarkCommand = "cargo"
        $benchmarkArguments = @("run", "-p", "ferrum-cli", "--bin", "ferrumPipeBenchmark", "--") + $benchmarkArguments
    }
    & $benchmarkCommand @benchmarkArguments *> $benchmarkLog
    if ($LASTEXITCODE -ne 0) {
        throw "Ferrum pipe benchmark post-processing failed with exit code $LASTEXITCODE. See $benchmarkLog"
    }
    $benchmark = Get-Content -LiteralPath $BenchmarkJson -Raw | ConvertFrom-Json
    $pressureDropFromMean = ConvertTo-NullableDouble $benchmark.solution.pressureDropFromMean
    $pressureDropFromField = $null
    $pressureDropFromOwnerCells = ConvertTo-NullableDouble $benchmark.solution.pressureDropFromOwnerCells
    $pressureDropForComparison = $pressureDropFromOwnerCells
    $pressureDropRelativeError = ConvertTo-NullableDouble $benchmark.solution.relativePressureDropFromOwnerCellsError
    $pressureDropFromMeanRelativeError = ConvertTo-NullableDouble $benchmark.solution.relativePressureDropFromMeanError

    return [ordered]@{
        mode = "laminar-simple"
        executableSolver = $true
        backend = $report.backend
        linearSolver = $report.options.linearSolver
        momentumLinearSolver = $report.options.momentumLinearSolver
        pressureLinearSolver = $report.options.pressureLinearSolver
        command = Format-CommandLine -Command $command -Arguments $arguments
        log = $LogPath
        reportJson = $ReportJson
        reportMarkdown = $ReportMarkdown
        fieldsDir = $FieldsDir
        benchmarkJson = $BenchmarkJson
        benchmarkLog = $benchmarkLog
        exitCode = $exitCode
        commandWallClockSeconds = $elapsed.TotalSeconds
        solveWallClockSeconds = ConvertTo-NullableDouble $report.solve.wallClockSeconds
        cells = ConvertTo-NullableInt $report.mesh.cells
        faces = ConvertTo-NullableInt $report.mesh.faces
        simpleIterations = ConvertTo-NullableInt $report.solve.simpleIterations
        iterations = ConvertTo-NullableInt $report.solve.simpleIterations
        converged = ConvertTo-NullableBool $report.solve.converged
        residualNorm = ConvertTo-NullableDouble $report.solve.finalMomentumResidualNorm
        normalizedResidualNorm = ConvertTo-NullableDouble $report.solve.finalMomentumNormalizedResidualNorm
        pressureCorrectionNormalizedResidualNorm = ConvertTo-NullableDouble $report.solve.finalPressureCorrectionNormalizedResidualNorm
        finalContinuityL2 = ConvertTo-NullableDouble $report.continuity.final.l2Norm
        inputs = [ordered]@{
            pressureDropPa = ConvertTo-NullableDouble $benchmark.inputs.pressureDrop
            dynamicViscosityPaS = ConvertTo-NullableDouble $report.options.dynamicViscosity
            densityKgPerM3 = ConvertTo-NullableDouble $report.options.density
            lengthM = ConvertTo-NullableDouble $benchmark.inputs.length
            diameterM = ConvertTo-NullableDouble $benchmark.inputs.diameter
            inletPatch = $benchmark.inputs.inletPatch
            outletPatch = $benchmark.inputs.outletPatch
        }
        result = [ordered]@{
            meanVelocityMps = ConvertTo-NullableDouble $benchmark.solution.meanVelocity
            analyticMeanVelocityMps = ConvertTo-NullableDouble $benchmark.solution.analyticMeanVelocity
            relativeMeanVelocityErrorToAnalytic = ConvertTo-NullableDouble $benchmark.solution.relativeMeanVelocityError
            flowRateM3s = ConvertTo-NullableDouble $benchmark.solution.flowRate
            analyticFlowRateM3s = ConvertTo-NullableDouble $benchmark.solution.analyticFlowRate
            pressureDropForComparisonPa = $pressureDropForComparison
            pressureDropForComparisonSource = "pressure-owner-cells"
            pressureDropFromMeanPa = $pressureDropFromMean
            pressureDropFromFieldPa = $pressureDropFromField
            pressureDropFromOwnerCellsPa = $pressureDropFromOwnerCells
            relativePressureDropErrorToAnalytic = $pressureDropRelativeError
            relativePressureDropFromMeanErrorToAnalytic = $pressureDropFromMeanRelativeError
            minVelocityMps = ConvertTo-NullableDouble $benchmark.solution.minVelocity
            maxVelocityMps = ConvertTo-NullableDouble $benchmark.solution.maxVelocity
        }
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
    $path = $script:BenchmarkProperties
    if (!(Test-Path -LiteralPath $path)) {
        throw "benchmark properties not found: $path"
    }
    $content = Get-Content -LiteralPath $path -Raw
    $match = [regex]::Match($content, "(?m)^\s*expectedDeltaP\s+\[[^\]]+\]\s+([-+0-9.eE]+)\s*;")
    if (!$match.Success) {
        throw "expectedDeltaP is missing from benchmark properties: $path"
    }
    $value = [double]::Parse($match.Groups[1].Value, [System.Globalization.CultureInfo]::InvariantCulture)
    if ([double]::IsNaN($value) -or [double]::IsInfinity($value) -or $value -le 0.0) {
        throw "expectedDeltaP must be a positive finite value in benchmark properties: $path"
    }
    return $value
}

function Read-DimensionedScalar($Content, [string]$Name) {
    if ([string]::IsNullOrWhiteSpace($Content)) {
        return $null
    }
    $match = [regex]::Match($Content, "(?m)^\s*$Name\s+(?:\[[^\]]+\]\s+)?([-+0-9.eE]+)\s*;")
    if (!$match.Success) {
        return $null
    }
    return [double]::Parse($match.Groups[1].Value, [System.Globalization.CultureInfo]::InvariantCulture)
}

function Read-PipeBenchmarkPhysics([string]$CaseRoot) {
    $benchmarkPath = $script:BenchmarkProperties
    $transportPath = Join-Path $CaseRoot "constant\transportProperties"
    $benchmark = if (Test-Path -LiteralPath $benchmarkPath) { Get-Content -LiteralPath $benchmarkPath -Raw } else { "" }
    $transport = if (Test-Path -LiteralPath $transportPath) { Get-Content -LiteralPath $transportPath -Raw } else { "" }

    $rho = Read-DimensionedScalar -Content $benchmark -Name "rho"
    if ($null -eq $rho) {
        $rho = Read-DimensionedScalar -Content $transport -Name "rho"
    }
    $mu = Read-DimensionedScalar -Content $benchmark -Name "mu"
    if ($null -eq $mu) {
        $mu = Read-DimensionedScalar -Content $transport -Name "mu"
    }

    $result = [ordered]@{
        lengthM = Read-DimensionedScalar -Content $benchmark -Name "length"
        diameterM = Read-DimensionedScalar -Content $benchmark -Name "diameter"
        rhoKgPerM3 = $rho
        dynamicViscosityPaS = $mu
        meanVelocityMps = Read-DimensionedScalar -Content $benchmark -Name "meanVelocity"
        referenceTemperatureK = Read-DimensionedScalar -Content $benchmark -Name "referenceTemperature"
        analyticDeltaPPa = Read-DimensionedScalar -Content $benchmark -Name "expectedDeltaP"
    }
    foreach ($name in @("lengthM", "diameterM", "rhoKgPerM3", "dynamicViscosityPaS", "analyticDeltaPPa")) {
        $value = $result[$name]
        if ($null -eq $value -or [double]::IsNaN([double]$value) -or
            [double]::IsInfinity([double]$value) -or [double]$value -le 0.0) {
            throw "$name must be a positive finite value in $benchmarkPath or $transportPath"
        }
    }
    return $result
}

function Read-PipeBenchmarkMesh([string]$CaseRoot) {
    $path = $script:BenchmarkProperties
    if (!(Test-Path -LiteralPath $path)) {
        return $null
    }
    $content = Get-Content -LiteralPath $path -Raw
    $result = [ordered]@{
        type = $null
        variant = $null
        axialCells = $null
        radialCells = $null
        angularSectors = $null
        cells = $null
        points = $null
    }
    foreach ($name in @("axialCells", "radialCells", "angularSectors", "cells", "points")) {
        $match = [regex]::Match($content, "(?m)^\s*$name\s+(\d+)\s*;")
        if ($match.Success) {
            $result[$name] = [int]::Parse($match.Groups[1].Value, [System.Globalization.CultureInfo]::InvariantCulture)
        }
    }
    foreach ($name in @("type", "variant")) {
        $match = [regex]::Match($content, "(?m)^\s*$name\s+([A-Za-z0-9_]+)\s*;")
        if ($match.Success) {
            $result[$name] = $match.Groups[1].Value
        }
    }
    return [pscustomobject]$result
}

function Write-MarkdownReport($Path, $Result) {
    $comparison = $Result.comparison
    $ferrum = $Result.ferrum
    $ferrumPreflight = $ferrum.preflight
    $ferrumSolve = $ferrum.solve
    $openFoam = $Result.openFoam
    $status = $Result.benchmarkStatus
    $runBudget = $Result.runBudget
    $mesh = $Result.mesh
    $physics = $Result.physics
    $ferrumSolverLabel = if ($null -ne $ferrumSolve -and $ferrumSolve.mode -eq "laminar-simple") { "laminar SIMPLE" } else { "Poiseuille" }
    $openFoamLabel = "OpenFOAM reference"
    if ($null -ne $openFoam) {
        $versionLabel = if ($null -ne $openFoam.version) { " $($openFoam.version)" } else { "" }
        if ($openFoam.application -eq "foamRun" -and $null -ne $openFoam.solverModule) {
            $openFoamLabel = "OpenFOAM$versionLabel foamRun/$($openFoam.solverModule)"
        } elseif ($null -ne $openFoam.application) {
            $openFoamLabel = "OpenFOAM$versionLabel $($openFoam.application)"
        }
    }
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
    $lines.Add("| Ferrum $ferrumSolverLabel solve | $($status.ferrumSolverComparison) |")
    $lines.Add("| OpenFOAM reference | $($status.openFoamReference) |")
    $lines.Add("")
    if ($null -ne $runBudget) {
        $lines.Add("## Run Budget")
        $lines.Add("")
        $lines.Add("For steady SIMPLE solvers this is a pseudo-time/iteration budget, not a transient physical-time integration.")
        $lines.Add("")
        $lines.Add("| Quantity | Value |")
        $lines.Add("| --- | ---: |")
        $lines.Add("| OpenFOAM endTime | $(Format-NullableNumber $runBudget.openFoamEndTime "G8") |")
        $lines.Add("| OpenFOAM deltaT | $(Format-NullableNumber $runBudget.openFoamDeltaT "G8") |")
        $lines.Add("| OpenFOAM simulated steps | $(Format-NullableNumber $runBudget.openFoamSimulatedSteps "G8") |")
        $lines.Add("| Ferrum SIMPLE iterations | $(Format-NullableNumber $runBudget.ferrumSimpleIterations "G8") |")
        $lines.Add("| Matched step budget | $($runBudget.matched) |")
        $lines.Add("")
    }
    if ($null -ne $mesh) {
        $lines.Add("## Mesh")
        $lines.Add("")
        $lines.Add("| Property | Value |")
        $lines.Add("| --- | ---: |")
        $lines.Add("| Type | $($mesh.type) |")
        $lines.Add("| Axial cells | $(Format-NullableNumber $mesh.axialCells "G8") |")
        $lines.Add("| Radial cells | $(Format-NullableNumber $mesh.radialCells "G8") |")
        $lines.Add("| Angular sectors | $(Format-NullableNumber $mesh.angularSectors "G8") |")
        if ($null -ne $mesh.points) {
            $lines.Add("| Points | $(Format-NullableNumber $mesh.points "G8") |")
        }
        $lines.Add("| Total cells | $(Format-NullableNumber $mesh.cells "G8") |")
        $lines.Add("")
    }
    if ($null -ne $physics) {
        $lines.Add("## Inputs")
        $lines.Add("")
        $lines.Add("| Quantity | Value |")
        $lines.Add("| --- | ---: |")
        $lines.Add("| Length [m] | $(Format-NullableNumber $physics.lengthM "G8") |")
        $lines.Add("| Diameter [m] | $(Format-NullableNumber $physics.diameterM "G8") |")
        $lines.Add("| Dynamic viscosity [Pa s] | $(Format-NullableNumber $physics.dynamicViscosityPaS "G8") |")
        $lines.Add("| Density [kg/m3] | $(Format-NullableNumber $physics.rhoKgPerM3 "G8") |")
        $lines.Add("| Mean velocity target [m/s] | $(Format-NullableNumber $physics.meanVelocityMps "G8") |")
        $lines.Add("| Analytic deltaP [Pa] | $(Format-NullableNumber $Result.analytic.deltaPPa "G8") |")
        $lines.Add("")
    }
    $lines.Add("## Pressure Loss")
    $lines.Add("")
    $lines.Add("| Source | deltaP [Pa] | Relative error to analytic |")
    $lines.Add("| --- | ---: | ---: |")
    $lines.Add("| Analytic Hagen-Poiseuille | $(Format-NullableNumber $Result.analytic.deltaPPa "G8") | 0% |")
    if ($null -ne $ferrumSolve -and $ferrumSolve.mode -eq "laminar-simple") {
        $deltaSource = if ($null -ne $comparison.ferrumDeltaPSource) { $comparison.ferrumDeltaPSource } else { "pressure field" }
        $lines.Add("| FerrumCFD laminar SIMPLE ($deltaSource) | $(Format-NullableNumber $comparison.ferrumDeltaPPa "G8") | $(Format-NullablePercent $comparison.ferrumRelativeErrorToAnalytic) |")
        $lines.Add("| FerrumCFD laminar SIMPLE (from mean U) | $(Format-NullableNumber $comparison.ferrumPressureDropFromMeanPa "G8") | $(Format-NullablePercent $comparison.ferrumPressureDropFromMeanRelativeErrorToAnalytic) |")
    } else {
        $lines.Add("| FerrumCFD Poiseuille | $(Format-NullableNumber $comparison.ferrumDeltaPPa "G8") | $(Format-NullablePercent $comparison.ferrumRelativeErrorToAnalytic) |")
    }
    $lines.Add("| $openFoamLabel | $(Format-NullableNumber $comparison.openFoamDeltaPPa "G8") | $(Format-NullablePercent $comparison.openFoamRelativeErrorToAnalytic) |")
    if ($null -ne $comparison.openFoamPressureLossMethod) {
        $method = $comparison.openFoamPressureLossMethod
        $lines.Add("")
        $lines.Add("OpenFOAM pressure loss sampling: ``$method`` with $($comparison.openFoamInletSamples) inlet and $($comparison.openFoamOutletSamples) outlet samples.")
        if ($null -ne $comparison.openFoamEffectiveLengthFraction) {
            $fraction = Format-NullableNumber $comparison.openFoamEffectiveLengthFraction "G8"
            $lines.Add("The sampled pressure difference was extrapolated to the full pipe length with effective length fraction ``$fraction``.")
        }
    }
    if ($null -ne $ferrumSolve) {
        $lines.Add("")
        if ($ferrumSolve.mode -eq "laminar-simple") {
            $lines.Add("The external Ferrum pipe post-processor reports owner-cell pressure deltaP for the OpenFOAM comparison and reconstructs deltaP from mean velocity for the Hagen-Poiseuille mass-flow check.")
        } else {
            $lines.Add("Ferrum reconstructs deltaP from the solved mean velocity for this source-driven Stokes benchmark.")
        }
        $lines.Add("")
        $lines.Add("## Ferrum Velocity")
        $lines.Add("")
        $lines.Add("| Quantity | Value |")
        $lines.Add("| --- | ---: |")
        $lines.Add("| Mean velocity [m/s] | $(Format-NullableNumber $ferrumSolve.result.meanVelocityMps "G8") |")
        $lines.Add("| Analytic mean velocity [m/s] | $(Format-NullableNumber $ferrumSolve.result.analyticMeanVelocityMps "G8") |")
        $lines.Add("| Relative mean-velocity error | $(Format-NullablePercent $ferrumSolve.result.relativeMeanVelocityErrorToAnalytic) |")
        $lines.Add("| Iterations | $(Format-NullableNumber $ferrumSolve.iterations "G8") |")
        $lines.Add("| Residual norm | $(Format-NullableNumber $ferrumSolve.residualNorm "G8") |")
        $lines.Add("| Converged | $($ferrumSolve.converged) |")
    }
    $lines.Add("")
    $lines.Add("## Timing")
    $lines.Add("")
    $lines.Add("| Runner | Wall clock [s] | Solver execution time [s] | Steps |")
    $lines.Add("| --- | ---: | ---: | ---: |")
    $plannedSteps = if ($null -ne $ferrumPreflight) { $ferrumPreflight.runSchedule.estimatedSteps } else { "n/a" }
    $lines.Add("| Ferrum preflight | $(Format-NullableNumber $comparison.timing.ferrumPreflightWallClockSeconds "G6") | n/a | $plannedSteps planned |")
    if ($null -ne $ferrumSolve) {
        $lines.Add("| Ferrum $ferrumSolverLabel solve | $(Format-NullableNumber $comparison.timing.ferrumCommandWallClockSeconds "G6") | $(Format-NullableNumber $comparison.timing.ferrumSolveWallClockSeconds "G6") | $($ferrumSolve.iterations) iterations |")
    } else {
        $lines.Add("| Ferrum $ferrumSolverLabel solve | n/a | n/a | n/a |")
    }
    if ($null -ne $openFoam) {
        $foamExecution = if ($null -ne $openFoam.foamTiming) { $openFoam.foamTiming.executionTimeSeconds } else { $null }
        $foamSteps = if ($null -ne $Result.openFoamRunControl) { $Result.openFoamRunControl.simulatedSteps } else { "n/a" }
        $lines.Add("| $openFoamLabel | $(Format-NullableNumber $comparison.timing.openFoamWallClockSeconds "G6") | $(Format-NullableNumber $foamExecution "G6") | $foamSteps |")
    } else {
        $lines.Add("| $openFoamLabel | n/a | n/a | n/a |")
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

$physics = Read-PipeBenchmarkPhysics $CaseRoot
$analyticDeltaPPa = Read-AnalyticDeltaP $CaseRoot
if ($null -ne $physics.analyticDeltaPPa) {
    $analyticDeltaPPa = [double]$physics.analyticDeltaPPa
}
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
$ferrumSolve = $null
if (!$SkipFerrumSolve) {
    $resultBaseName = [System.IO.Path]::GetFileNameWithoutExtension($OutFile)
    if ($FerrumSolver -eq "laminarSimple") {
        $ferrumSolveLog = Join-Path (Split-Path -Parent $OutFile) "$resultBaseName.ferrum_laminar_simple.log"
        $ferrumSolveJson = Join-Path (Split-Path -Parent $OutFile) "$resultBaseName.ferrum_laminar_simple.json"
        $ferrumSolveMarkdown = Join-Path (Split-Path -Parent $OutFile) "$resultBaseName.ferrum_laminar_simple.md"
        $ferrumFieldsDir = Join-Path (Split-Path -Parent $OutFile) "$resultBaseName.ferrum_laminar_simple_fields"
        $ferrumBenchmarkJson = Join-Path (Split-Path -Parent $OutFile) "$resultBaseName.ferrum_pipe_benchmark.json"
        $ferrumSolve = Invoke-FerrumLaminarSimpleSolve `
            -CaseRoot $CaseRoot `
            -LogPath $ferrumSolveLog `
            -ReportJson $ferrumSolveJson `
            -ReportMarkdown $ferrumSolveMarkdown `
            -FieldsDir $ferrumFieldsDir `
            -BenchmarkJson $ferrumBenchmarkJson `
            -SimpleIterations $FerrumSimpleIterations `
            -AnalyticDeltaPPa $analyticDeltaPPa `
            -Physics $physics
    } else {
        $ferrumSolveLog = Join-Path (Split-Path -Parent $OutFile) "$resultBaseName.ferrum_poiseuille.log"
        $ferrumSolve = Invoke-FerrumPoiseuilleSolve `
            -CaseRoot $CaseRoot `
            -LogPath $ferrumSolveLog `
            -LinearSolver $FerrumLinearSolver `
            -SolveTolerance $FerrumSolveTolerance `
            -MaxIterations $FerrumMaxIterations `
            -AnalyticDeltaPPa $analyticDeltaPPa `
            -Physics $physics
    }
}
$ferrumDeltaPPa = if ($null -ne $ferrumSolve -and $null -ne $ferrumSolve.result.pressureDropForComparisonPa) {
    $ferrumSolve.result.pressureDropForComparisonPa
} elseif ($null -ne $ferrumSolve) {
    $ferrumSolve.result.pressureDropFromMeanPa
} else {
    $null
}
$ferrumRelativeError = if ($null -ne $ferrumDeltaPPa -and $analyticDeltaPPa -ne 0.0) {
    ($ferrumDeltaPPa - $analyticDeltaPPa) / $analyticDeltaPPa
} else {
    $null
}
$ferrumRelativeErrorToOpenFoam = if ($null -ne $ferrumDeltaPPa -and $null -ne $openFoamDeltaPPa -and $openFoamDeltaPPa -ne 0.0) {
    ($ferrumDeltaPPa - $openFoamDeltaPPa) / $openFoamDeltaPPa
} else {
    $null
}
$ferrumSolverStatus = if ($SkipFerrumSolve) {
    "skipped"
} elseif ($null -ne $ferrumSolve -and $ferrumSolve.exitCode -eq 0 -and $ferrumSolve.converged -eq $true) {
    "passed"
} elseif ($null -ne $ferrumSolve -and $ferrumSolve.exitCode -eq 0) {
    "completed-not-converged"
} elseif ($null -ne $ferrumSolve) {
    "failed"
} else {
    "missing"
}
$openFoamRunControl = if ($null -ne $openFoam) { $openFoam.runControl } else { $null }
$openFoamPressureLoss = if ($null -ne $openFoam) { $openFoam.openFoam.pressureLoss } else { $null }
$openFoamSimulatedSteps = if ($null -ne $openFoamRunControl) { ConvertTo-NullableInt $openFoamRunControl.simulatedSteps } else { $null }
$ferrumSimpleIterationsForBudget = if ($null -ne $ferrumSolve -and $ferrumSolve.mode -eq "laminar-simple") {
    ConvertTo-NullableInt $ferrumSolve.simpleIterations
} else {
    $null
}
$matchedRunBudget = [ordered]@{
    mode = if ($FerrumSolver -eq "laminarSimple") { "steady-simple-pseudo-time" } else { "not-applicable" }
    openFoamEndTime = if ($null -ne $openFoamRunControl) { ConvertTo-NullableDouble $openFoamRunControl.endTime } else { $null }
    openFoamDeltaT = if ($null -ne $openFoamRunControl) { ConvertTo-NullableDouble $openFoamRunControl.deltaT } else { $null }
    openFoamSimulatedSteps = $openFoamSimulatedSteps
    ferrumSimpleIterations = $ferrumSimpleIterationsForBudget
    matched = if ($null -ne $openFoamSimulatedSteps -and $null -ne $ferrumSimpleIterationsForBudget) {
        $openFoamSimulatedSteps -eq $ferrumSimpleIterationsForBudget
    } else {
        $false
    }
}
$referenceMesh = Read-PipeBenchmarkMesh $CaseRoot
$mesh = if ($null -ne $openFoam -and $null -ne $openFoam.mesh) {
    [ordered]@{
        type = if ($null -ne $openFoam.mesh.type) { $openFoam.mesh.type } else { "polyMesh" }
        axialCells = $openFoam.mesh.axialCells
        radialCells = $openFoam.mesh.radialCells
        angularSectors = $openFoam.mesh.angularSectors
        cells = $openFoam.mesh.cells
        faces = if ($null -ne $openFoam.mesh.faces) { $openFoam.mesh.faces } elseif ($null -ne $ferrumSolve) { $ferrumSolve.faces } else { $null }
        points = $openFoam.mesh.points
    }
} elseif ($null -ne $ferrumSolve) {
    [ordered]@{
        type = "polyMesh"
        axialCells = $null
        radialCells = $null
        angularSectors = $null
        cells = $ferrumSolve.cells
        faces = $ferrumSolve.faces
        points = $null
    }
} elseif ($null -ne $referenceMesh) {
    [ordered]@{
        type = "reference-only"
        axialCells = $referenceMesh.axialCells
        radialCells = $referenceMesh.radialCells
        angularSectors = $referenceMesh.angularSectors
        cells = $referenceMesh.cells
        faces = $null
        points = $referenceMesh.points
    }
} else {
    $null
}

$openFoamReferenceStatus = if ($null -eq $openFoam) {
    "missing"
} elseif ($openFoam.openFoam.available -eq $false) {
    "unavailable"
} elseif ($openFoam.status -eq "ran" -and $openFoam.openFoam.exitCode -eq 0 -and
    $null -ne $openFoam.openFoam.pressureLoss) {
    "passed"
} else {
    "failed"
}

$notes = @(
    "Benchmark reference data is external to the case at $BenchmarkProperties; the case contains only simulation inputs.",
    "OpenFOAM is generated only under target/ for comparison and is not the default FerrumCFD workflow.",
    "OpenFOAM-to-analytic pressure-loss differences are treated as mesh/discretization/setup error at this stage."
)
if ($FerrumSolver -eq "laminarSimple") {
    $notes += @(
        "FerrumCFD's generic laminar SIMPLE solver writes U/p fields without analytic benchmark data.",
        "ferrumPipeBenchmark post-processes those stored fields and reports owner-cell and mean-flow pressure loss externally."
    )
} else {
    $notes += @(
        "FerrumCFD's Poiseuille path is an executable source-driven axial Stokes benchmark, not a full SIMPLE pressure-velocity solver yet.",
        "Ferrum reconstructs pressure loss from the solved mean velocity so it can be compared directly against Hagen-Poiseuille and OpenFOAM in SI units."
    )
}

$runSchedule = [ordered]@{
    startTime = $ferrumPlan.run.startTime
    endTime = $ferrumPlan.run.endTime
    deltaT = $ferrumPlan.run.deltaT
    estimatedSteps = $ferrumPlan.run.estimatedSteps
    estimatedWrites = $ferrumPlan.run.estimatedWriteEvents
}
$stateSummary = Get-StateSummary $ferrumPlan

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
        meanVelocityMps = $physics.meanVelocityMps
    }
    physics = $physics
    mesh = $mesh
    runBudget = $matchedRunBudget
    ferrum = [ordered]@{
        mode = if ($null -ne $ferrumSolve) { $ferrumSolve.mode } else { "preflight-only" }
        requestedSolver = $FerrumSolver
        executableSolver = $null -ne $ferrumSolve
        wallClockSeconds = if ($null -ne $ferrumSolve) { $ferrumSolve.solveWallClockSeconds } else { $ferrumRun.wallClockSeconds }
        planJson = $ferrumRun.planJson
        log = $ferrumRun.log
        runSchedule = $runSchedule
        state = $stateSummary
        preflight = [ordered]@{
            wallClockSeconds = $ferrumRun.wallClockSeconds
            planJson = $ferrumRun.planJson
            log = $ferrumRun.log
            runSchedule = $runSchedule
            state = $stateSummary
        }
        solve = $ferrumSolve
    }
    openFoam = if ($null -ne $openFoam) { $openFoam.openFoam } else { $null }
    openFoamRunControl = $openFoamRunControl
    comparison = [ordered]@{
        openFoamDeltaPPa = $openFoamDeltaPPa
        openFoamRelativeErrorToAnalytic = $openFoamRelativeError
        openFoamPressureLossMethod = if ($null -ne $openFoamPressureLoss) { $openFoamPressureLoss.method } else { $null }
        openFoamInletSamples = if ($null -ne $openFoamPressureLoss) { $openFoamPressureLoss.inletSamples } else { $null }
        openFoamOutletSamples = if ($null -ne $openFoamPressureLoss) { $openFoamPressureLoss.outletSamples } else { $null }
        openFoamSampledDeltaPPa = if ($null -ne $openFoamPressureLoss) { $openFoamPressureLoss.sampledDeltaPPa } else { $null }
        openFoamEffectiveLengthFraction = if ($null -ne $openFoamPressureLoss) { $openFoamPressureLoss.effectiveLengthFraction } else { $null }
        ferrumSolver = $FerrumSolver
        ferrumDeltaPPa = $ferrumDeltaPPa
        ferrumDeltaPSource = if ($null -ne $ferrumSolve -and $null -ne $ferrumSolve.result.pressureDropForComparisonSource) { $ferrumSolve.result.pressureDropForComparisonSource } elseif ($null -ne $ferrumSolve) { "mean-velocity" } else { $null }
        ferrumRelativeErrorToAnalytic = $ferrumRelativeError
        ferrumRelativeErrorToOpenFoam = $ferrumRelativeErrorToOpenFoam
        ferrumPressureDropFromMeanPa = if ($null -ne $ferrumSolve) { $ferrumSolve.result.pressureDropFromMeanPa } else { $null }
        ferrumPressureDropFromMeanRelativeErrorToAnalytic = if ($null -ne $ferrumSolve -and $null -ne $ferrumSolve.result.relativePressureDropFromMeanErrorToAnalytic) { $ferrumSolve.result.relativePressureDropFromMeanErrorToAnalytic } elseif ($null -ne $ferrumSolve) { $ferrumSolve.result.relativePressureDropErrorToAnalytic } else { $null }
        ferrumPressureDropFromFieldPa = if ($null -ne $ferrumSolve) { $ferrumSolve.result.pressureDropFromFieldPa } else { $null }
        ferrumPressureDropFromOwnerCellsPa = if ($null -ne $ferrumSolve) { $ferrumSolve.result.pressureDropFromOwnerCellsPa } else { $null }
        ferrumMeanVelocityMps = if ($null -ne $ferrumSolve) { $ferrumSolve.result.meanVelocityMps } else { $null }
        ferrumMeanVelocityRelativeErrorToAnalytic = if ($null -ne $ferrumSolve) { $ferrumSolve.result.relativeMeanVelocityErrorToAnalytic } else { $null }
        ferrumSolverComparison = $ferrumSolverStatus
        timing = [ordered]@{
            ferrumPreflightWallClockSeconds = $ferrumRun.wallClockSeconds
            ferrumSolveWallClockSeconds = if ($null -ne $ferrumSolve) { $ferrumSolve.solveWallClockSeconds } else { $null }
            ferrumCommandWallClockSeconds = if ($null -ne $ferrumSolve) { $ferrumSolve.commandWallClockSeconds } else { $null }
            openFoamWallClockSeconds = $openFoamWallClock
            openFoamExecutionTimeSeconds = if ($null -ne $openFoam -and $null -ne $openFoam.openFoam.foamTiming) { $openFoam.openFoam.foamTiming.executionTimeSeconds } else { $null }
            openFoamReportedClockSeconds = if ($null -ne $openFoam -and $null -ne $openFoam.openFoam.foamTiming) { $openFoam.openFoam.foamTiming.clockTimeSeconds } else { $null }
        }
    }
    benchmarkStatus = [ordered]@{
        ferrumPreflight = "passed"
        openFoamReference = $openFoamReferenceStatus
        ferrumSolverComparison = $ferrumSolverStatus
        readyForCiGate = ($ferrumSolverStatus -eq "passed" -and $openFoamReferenceStatus -eq "passed")
        notes = $notes
    }
}

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $OutFile) | Out-Null
$result | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $OutFile -Encoding UTF8
Write-MarkdownReport -Path $ReportFile -Result $result
Write-Output "wrote laminar pipe comparison: $OutFile"
Write-Output "wrote laminar pipe report: $ReportFile"
