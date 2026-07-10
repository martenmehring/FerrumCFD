param(
    [string]$CaseRoot = "",
    [string]$StudyRoot = "",
    [string]$BenchmarkProperties = "",
    [string[]]$SimpleIterations = @("2", "5", "10", "20", "30"),
    [ValidateSet("x", "y", "z")]
    [string]$Axis = "x",
    [string]$InletPatch = "inlet",
    [string]$OutletPatch = "outlet",
    [bool]$RunPipeBenchmark = $true,
    [switch]$WriteFinalFields,
    [switch]$UseExistingReports
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot

if ([string]::IsNullOrWhiteSpace($CaseRoot)) {
    $CaseRoot = Join-Path $RepoRoot "examples\laminar_pipe"
}
if ([string]::IsNullOrWhiteSpace($StudyRoot)) {
    $StudyRoot = Join-Path $RepoRoot "target\benchmarks\laminar_simple_iteration_sweep"
}
if ([string]::IsNullOrWhiteSpace($BenchmarkProperties)) {
    $BenchmarkProperties = Join-Path $RepoRoot "benchmarks\laminar_pipe\pipeBenchmark"
}
if ($RunPipeBenchmark -and !(Test-Path -LiteralPath $BenchmarkProperties -PathType Leaf)) {
    throw "benchmark properties not found: $BenchmarkProperties"
}
if ([string]::IsNullOrWhiteSpace($InletPatch) -or [string]::IsNullOrWhiteSpace($OutletPatch)) {
    throw "InletPatch and OutletPatch must not be empty"
}

function Get-FullPath([string]$Path) {
    return [System.IO.Path]::GetFullPath($Path)
}

function Test-IsPathUnder([string]$Child, [string]$Parent) {
    $childFull = Get-FullPath $Child
    $parentFull = (Get-FullPath $Parent).TrimEnd(
        [System.IO.Path]::DirectorySeparatorChar,
        [System.IO.Path]::AltDirectorySeparatorChar
    )
    return $childFull.Equals($parentFull, [System.StringComparison]::OrdinalIgnoreCase) -or
        $childFull.StartsWith($parentFull + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase) -or
        $childFull.StartsWith($parentFull + [System.IO.Path]::AltDirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)
}

$targetRoot = Join-Path $RepoRoot "target"
if (!(Test-IsPathUnder $StudyRoot $targetRoot)) {
    throw "StudyRoot must be inside the repository target directory: $targetRoot"
}

$iterationBudgets = New-Object System.Collections.Generic.List[int]
foreach ($rawValue in $SimpleIterations) {
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
        $iterationBudgets.Add($parsed) | Out-Null
    }
}
if ($iterationBudgets.Count -eq 0) {
    throw "SimpleIterations must contain at least one positive integer"
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

function Read-JsonFile([string]$Path) {
    if (!(Test-Path -LiteralPath $Path -PathType Leaf)) {
        return $null
    }
    return Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
}

function Get-FerrumCommand([string]$Binary) {
    $exe = Join-Path $RepoRoot "target\debug\$Binary.exe"
    if (Test-Path -LiteralPath $exe -PathType Leaf) {
        return [pscustomobject][ordered]@{
            command = $exe
            prefix = @()
        }
    }
    return [pscustomobject][ordered]@{
        command = "cargo"
        prefix = @("run", "-p", "ferrum-cli", "--bin", $Binary, "--")
    }
}

function Read-DimensionedScalar([string]$Content, [string]$Name) {
    $pattern = "(?m)^\s*$([regex]::Escape($Name))\s+(?:\[[^\]]+\]\s+)?([-+0-9.eE]+)\s*;"
    $match = [regex]::Match($Content, $pattern)
    if (!$match.Success) {
        return $null
    }
    return [double]::Parse(
        $match.Groups[1].Value,
        [System.Globalization.CultureInfo]::InvariantCulture
    )
}

function Assert-PositiveFinite([string]$Name, $Value) {
    if ($null -eq $Value -or [double]::IsNaN([double]$Value) -or
        [double]::IsInfinity([double]$Value) -or [double]$Value -le 0.0) {
        throw "$Name must be a positive finite value"
    }
}

function Read-PipeBenchmarkPhysics([string]$CaseRoot, [string]$PropertiesPath) {
    $benchmark = Get-Content -LiteralPath $PropertiesPath -Raw
    $transportPath = Join-Path $CaseRoot "constant\transportProperties"
    $transport = if (Test-Path -LiteralPath $transportPath -PathType Leaf) {
        Get-Content -LiteralPath $transportPath -Raw
    } else {
        ""
    }

    $mu = Read-DimensionedScalar -Content $benchmark -Name "mu"
    if ($null -eq $mu) {
        $mu = Read-DimensionedScalar -Content $transport -Name "mu"
    }
    $result = [pscustomobject][ordered]@{
        pressureDrop = Read-DimensionedScalar -Content $benchmark -Name "expectedDeltaP"
        dynamicViscosity = $mu
        length = Read-DimensionedScalar -Content $benchmark -Name "length"
        diameter = Read-DimensionedScalar -Content $benchmark -Name "diameter"
    }
    Assert-PositiveFinite "expectedDeltaP" $result.pressureDrop
    Assert-PositiveFinite "mu" $result.dynamicViscosity
    Assert-PositiveFinite "length" $result.length
    Assert-PositiveFinite "diameter" $result.diameter
    return $result
}

function Invoke-PipeBenchmark(
    [string]$CaseRoot,
    [string]$FieldsDir,
    [string]$ReportJson,
    [string]$ReportMarkdown,
    [string]$LogPath,
    $Physics
) {
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $ReportJson) | Out-Null
    $runner = Get-FerrumCommand "ferrumPipeBenchmark"
    $arguments = New-Object System.Collections.Generic.List[string]
    foreach ($argument in $runner.prefix) {
        $arguments.Add($argument) | Out-Null
    }
    @(
        "-case", $CaseRoot,
        "--fields", $FieldsDir,
        "--pressureDrop", (Format-F64 $Physics.pressureDrop),
        "--mu", (Format-F64 $Physics.dynamicViscosity),
        "--length", (Format-F64 $Physics.length),
        "--diameter", (Format-F64 $Physics.diameter),
        "--axis", $Axis,
        "--inletPatch", $InletPatch,
        "--outletPatch", $OutletPatch,
        "--outJson", $ReportJson,
        "--outMarkdown", $ReportMarkdown
    ) | ForEach-Object { $arguments.Add($_) | Out-Null }

    $script:benchmarkExitCode = $null
    $elapsed = Measure-Command {
        Push-Location $RepoRoot
        try {
            & $runner.command @($arguments.ToArray()) *> $LogPath
            $script:benchmarkExitCode = $LASTEXITCODE
        } finally {
            Pop-Location
        }
    }
    $exitCode = if ($null -eq $script:benchmarkExitCode) { 0 } else { $script:benchmarkExitCode }
    if ($exitCode -ne 0) {
        throw "Ferrum pipe benchmark failed with exit code $exitCode. See $LogPath"
    }
    $report = Read-JsonFile $ReportJson
    if ($null -eq $report) {
        throw "Ferrum pipe benchmark did not write $ReportJson"
    }
    return [pscustomobject][ordered]@{
        report = $report
        command = Format-CommandLine -Command $runner.command -Arguments $arguments.ToArray()
        wallClockSeconds = $elapsed.TotalSeconds
        log = $LogPath
        reportJson = $ReportJson
        reportMarkdown = $ReportMarkdown
    }
}

function New-LaminarSimpleSweepRow(
    [object]$Report,
    $BenchmarkRun,
    [int]$IterationBudget,
    [string]$ReportJson,
    [string]$ReportMarkdown,
    [string]$LogPath,
    [string]$FieldsDir,
    $CommandWallClockSeconds,
    [string]$Command
) {
    $pressureAssembly = $Report.pressureAssembly
    $benchmark = if ($null -ne $BenchmarkRun) { $BenchmarkRun.report } else { $null }
    $solution = if ($null -ne $benchmark) { $benchmark.solution } else { $null }

    return [pscustomobject][ordered]@{
        iterationBudget = $IterationBudget
        actualSimpleIterations = $Report.solve.simpleIterations
        converged = $Report.solve.converged
        stopReason = $Report.solve.stopReason
        meshCells = $Report.mesh.cells
        meshFaces = $Report.mesh.faces
        solverWallClockSeconds = $Report.solve.wallClockSeconds
        commandWallClockSeconds = $CommandWallClockSeconds
        finalContinuityL2 = $Report.continuity.final.l2Norm
        momentumResidualNorm = $Report.solve.finalMomentumResidualNorm
        momentumNormalizedResidualNorm = $Report.solve.finalMomentumNormalizedResidualNorm
        pressureCorrectionResidualNorm = $Report.solve.finalPressureCorrectionResidualNorm
        pressureCorrectionNormalizedResidualNorm = $Report.solve.finalPressureCorrectionNormalizedResidualNorm
        finalMomentumLinearConverged = $Report.solve.finalMomentumLinearConverged
        finalPressureLinearConverged = $Report.solve.finalPressureLinearConverged
        velocityMinMagnitudeMps = $Report.fields.velocity.minMagnitude
        velocityMaxMagnitudeMps = $Report.fields.velocity.maxMagnitude
        velocityL2Norm = $Report.fields.velocity.l2Norm
        pressureMinPa = $Report.fields.pressure.min
        pressureMaxPa = $Report.fields.pressure.max
        pressureL2Norm = $Report.fields.pressure.l2Norm
        meanVelocityMps = if ($null -ne $solution) { $solution.meanVelocity } else { $null }
        analyticMeanVelocityMps = if ($null -ne $solution) { $solution.analyticMeanVelocity } else { $null }
        relativeMeanVelocityError = if ($null -ne $solution) { $solution.relativeMeanVelocityError } else { $null }
        pressureDropFromMeanPa = if ($null -ne $solution) { $solution.pressureDropFromMean } else { $null }
        analyticPressureDropPa = if ($null -ne $benchmark) { $benchmark.inputs.pressureDrop } else { $null }
        relativePressureDropErrorFromMean = if ($null -ne $solution) { $solution.relativePressureDropFromMeanError } else { $null }
        pressureDropFromOwnerCellsPa = if ($null -ne $solution) { $solution.pressureDropFromOwnerCells } else { $null }
        relativePressureDropErrorFromOwnerCells = if ($null -ne $solution) { $solution.relativePressureDropFromOwnerCellsError } else { $null }
        minAxialVelocityMps = if ($null -ne $solution) { $solution.minVelocity } else { $null }
        maxAxialVelocityMps = if ($null -ne $solution) { $solution.maxVelocity } else { $null }
        totalMomentumLinearIterations = $Report.solve.momentumLinearIterations
        totalPressureLinearIterations = $Report.solve.pressureLinearIterations
        momentumNonConvergedPredictors = $Report.linearSolves.momentumNonConvergedPredictors
        momentumComponentNonConvergedSolves = $Report.linearSolves.momentumComponentNonConvergedSolves
        pressureCorrectionSolves = $Report.linearSolves.pressureCorrectionSolves
        pressureCorrectionNonConvergedSolves = $Report.linearSolves.pressureCorrectionNonConvergedSolves
        maxMomentumLinearIterationsPerSimple = $Report.linearSolves.maxMomentumLinearIterationsPerSimple
        maxPressureLinearIterationsPerSimple = $Report.linearSolves.maxPressureLinearIterationsPerSimple
        averageMomentumLinearIterationsPerSimple = $Report.linearSolves.averageMomentumLinearIterationsPerSimple
        averagePressureLinearIterationsPerSimple = $Report.linearSolves.averagePressureLinearIterationsPerSimple
        divPhiUScheme = $Report.options.schemes.divPhiU
        laplacianScheme = $Report.options.schemes.laplacian
        consistent = $Report.options.consistent
        pressureAssemblyRAUMin = $pressureAssembly.rAU.min
        pressureAssemblyRAUMax = $pressureAssembly.rAU.max
        pressureAssemblyRAtUMin = $pressureAssembly.rAtU.min
        pressureAssemblyRAtUMax = $pressureAssembly.rAtU.max
        pressureAssemblyHbyAL2 = $pressureAssembly.HbyA.l2Norm
        pressureAssemblySourceL2 = $pressureAssembly.pressureSource.l2Norm
        pressureAssemblySourceSumAbs = $pressureAssembly.pressureSource.sumAbs
        pressureAssemblyPhiHbyABoundaryBefore = $pressureAssembly.phiHbyABeforeAdjust.boundarySum
        pressureAssemblyPhiHbyABoundaryAfter = $pressureAssembly.phiHbyAAfterAdjust.boundarySum
        pressureAssemblyPressureEquationFluxBoundary = $pressureAssembly.pressureEquationFlux.boundarySum
        pressureAssemblyPressureFluxBoundary = $pressureAssembly.pressureFlux.boundarySum
        pressureAssemblyCorrectedPhiBoundary = $pressureAssembly.correctedPhi.boundarySum
        pressureAssemblyCorrectedPhiBoundaryAbs = $pressureAssembly.correctedPhi.boundarySumAbs
        pressureAssemblyCorrectedPhiSumAbs = $pressureAssembly.correctedPhi.sumAbs
        reportJson = $ReportJson
        reportMarkdown = $ReportMarkdown
        log = $LogPath
        fieldsDir = $FieldsDir
        command = $Command
        pipeBenchmarkJson = if ($null -ne $BenchmarkRun) { $BenchmarkRun.reportJson } else { $null }
        pipeBenchmarkMarkdown = if ($null -ne $BenchmarkRun) { $BenchmarkRun.reportMarkdown } else { $null }
        pipeBenchmarkLog = if ($null -ne $BenchmarkRun) { $BenchmarkRun.log } else { $null }
        pipeBenchmarkCommand = if ($null -ne $BenchmarkRun) { $BenchmarkRun.command } else { $null }
        pipeBenchmarkWallClockSeconds = if ($null -ne $BenchmarkRun) { $BenchmarkRun.wallClockSeconds } else { $null }
    }
}

function Invoke-LaminarSimpleIterationRun(
    [string]$CaseRoot,
    [int]$IterationBudget,
    [string]$ResultRoot,
    [string]$LogRoot,
    [string]$FieldsRoot,
    [bool]$WriteFields,
    [bool]$UseExistingReport,
    [bool]$RunBenchmark,
    $BenchmarkPhysics
) {
    $name = "simple_$IterationBudget"
    $reportJson = Join-Path $ResultRoot "$name.json"
    $reportMarkdown = Join-Path $ResultRoot "$name.md"
    $logPath = Join-Path $LogRoot "$name.log"
    $fieldsDir = if ($WriteFields -or $RunBenchmark) {
        Join-Path $FieldsRoot $IterationBudget.ToString([System.Globalization.CultureInfo]::InvariantCulture)
    } else {
        $null
    }
    $benchmarkJson = Join-Path $ResultRoot "$name.pipe_benchmark.json"
    $benchmarkMarkdown = Join-Path $ResultRoot "$name.pipe_benchmark.md"
    $benchmarkLog = Join-Path $LogRoot "$name.pipe_benchmark.log"

    New-Item -ItemType Directory -Force -Path $ResultRoot, $LogRoot | Out-Null
    if ($null -ne $fieldsDir) {
        New-Item -ItemType Directory -Force -Path $fieldsDir | Out-Null
    }

    $report = if ($UseExistingReport) { Read-JsonFile $reportJson } else { $null }
    $commandWallClockSeconds = $null
    $command = if ($null -ne $report) { "existing report: $reportJson" } else { $null }
    if ($null -eq $report) {
        foreach ($stalePath in @($reportJson, $reportMarkdown, $logPath, $benchmarkJson, $benchmarkMarkdown, $benchmarkLog)) {
            if (Test-Path -LiteralPath $stalePath -PathType Leaf) {
                Remove-Item -LiteralPath $stalePath -Force
            }
        }

        $solver = Get-FerrumCommand "ferrumSolver"
        $arguments = New-Object System.Collections.Generic.List[string]
        foreach ($argument in $solver.prefix) {
            $arguments.Add($argument) | Out-Null
        }
        @(
            "-case", $CaseRoot,
            "--solveLaminarSimple",
            "--minSimpleIterations", $IterationBudget.ToString([System.Globalization.CultureInfo]::InvariantCulture),
            "--maxSimpleIterations", $IterationBudget.ToString([System.Globalization.CultureInfo]::InvariantCulture),
            "--solveReportJson", $reportJson,
            "--solveReportMarkdown", $reportMarkdown
        ) | ForEach-Object { $arguments.Add($_) | Out-Null }
        if ($null -ne $fieldsDir) {
            $arguments.Add("--writeFinalFields") | Out-Null
            $arguments.Add($fieldsDir) | Out-Null
        }

        $script:runExitCode = $null
        $elapsed = Measure-Command {
            Push-Location $RepoRoot
            try {
                & $solver.command @($arguments.ToArray()) *> $logPath
                $script:runExitCode = $LASTEXITCODE
            } finally {
                Pop-Location
            }
        }
        $exitCode = if ($null -eq $script:runExitCode) { 0 } else { $script:runExitCode }
        if ($exitCode -ne 0) {
            throw "Ferrum laminar SIMPLE iteration run $IterationBudget failed with exit code $exitCode. See $logPath"
        }
        $report = Read-JsonFile $reportJson
        if ($null -eq $report) {
            throw "Ferrum laminar SIMPLE iteration run $IterationBudget did not write $reportJson"
        }
        $commandWallClockSeconds = $elapsed.TotalSeconds
        $command = Format-CommandLine -Command $solver.command -Arguments $arguments.ToArray()
    }

    $benchmarkRun = $null
    if ($RunBenchmark) {
        if ($null -eq $fieldsDir -or !(Test-Path -LiteralPath $fieldsDir -PathType Container)) {
            throw "pipe benchmark requires stored fields for iteration ${IterationBudget}: $fieldsDir"
        }
        $existingBenchmark = if ($UseExistingReport) { Read-JsonFile $benchmarkJson } else { $null }
        if ($null -ne $existingBenchmark) {
            $benchmarkRun = [pscustomobject][ordered]@{
                report = $existingBenchmark
                command = "existing report: $benchmarkJson"
                wallClockSeconds = $null
                log = $benchmarkLog
                reportJson = $benchmarkJson
                reportMarkdown = $benchmarkMarkdown
            }
        } else {
            $benchmarkArguments = @{
                CaseRoot = $CaseRoot
                FieldsDir = $fieldsDir
                ReportJson = $benchmarkJson
                ReportMarkdown = $benchmarkMarkdown
                LogPath = $benchmarkLog
                Physics = $BenchmarkPhysics
            }
            $benchmarkRun = Invoke-PipeBenchmark @benchmarkArguments
        }
    }

    $rowArguments = @{
        Report = $report
        BenchmarkRun = $benchmarkRun
        IterationBudget = $IterationBudget
        ReportJson = $reportJson
        ReportMarkdown = $reportMarkdown
        LogPath = $logPath
        FieldsDir = $fieldsDir
        CommandWallClockSeconds = $commandWallClockSeconds
        Command = $command
    }
    return New-LaminarSimpleSweepRow @rowArguments
}

function Write-SweepMarkdown($Path, $Rows, $Summary) {
    $lines = New-Object System.Collections.Generic.List[string]
    $lines.Add("# Laminar SIMPLE Iteration Sweep")
    $lines.Add("")
    $lines.Add("Case: $($Summary.caseDir)")
    $lines.Add("")
    $lines.Add("The solver report is geometry-independent. Pipe reference values below come from a separate post-processing step over the stored U and p fields.")
    $lines.Add("")
    $lines.Add("## Solver")
    $lines.Add("")
    $lines.Add("| Iterations | Converged | Stop reason | Solve [s] | Continuity L2 | Momentum normed | p-corr normed | Momentum linear ok | Pressure linear ok | Pressure nonconv solves |")
    $lines.Add("| ---: | --- | --- | ---: | ---: | ---: | ---: | --- | --- | ---: |")
    foreach ($row in $Rows) {
        $lines.Add("| $($row.iterationBudget) | $($row.converged) | $($row.stopReason) | $(Format-NullableNumber $row.solverWallClockSeconds "F6") | $(Format-NullableNumber $row.finalContinuityL2 "G6") | $(Format-NullableNumber $row.momentumNormalizedResidualNorm "G6") | $(Format-NullableNumber $row.pressureCorrectionNormalizedResidualNorm "G6") | $($row.finalMomentumLinearConverged) | $($row.finalPressureLinearConverged) | $(Format-NullableNumber $row.pressureCorrectionNonConvergedSolves "G8") |")
    }
    $lines.Add("")
    $lines.Add("## Generic Field Diagnostics")
    $lines.Add("")
    $lines.Add("| Iterations | Cells | |U| min/max [m/s] | p min/max [Pa] |")
    $lines.Add("| ---: | ---: | --- | --- |")
    foreach ($row in $Rows) {
        $lines.Add("| $($row.iterationBudget) | $($row.meshCells) | $(Format-NullableNumber $row.velocityMinMagnitudeMps "G8") / $(Format-NullableNumber $row.velocityMaxMagnitudeMps "G8") | $(Format-NullableNumber $row.pressureMinPa "G8") / $(Format-NullableNumber $row.pressureMaxPa "G8") |")
    }
    if ($Summary.runPipeBenchmark) {
        $lines.Add("")
        $lines.Add("## External Hagen-Poiseuille Post-Processing")
        $lines.Add("")
        $lines.Add("| Iterations | Mean U [m/s] | Mean U error | DeltaP from mean U [Pa] | Mean-U error | DeltaP owner cells [Pa] | Owner-cell error |")
        $lines.Add("| ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        foreach ($row in $Rows) {
            $lines.Add("| $($row.iterationBudget) | $(Format-NullableNumber $row.meanVelocityMps "G8") | $(Format-NullablePercent $row.relativeMeanVelocityError) | $(Format-NullableNumber $row.pressureDropFromMeanPa "G8") | $(Format-NullablePercent $row.relativePressureDropErrorFromMean) | $(Format-NullableNumber $row.pressureDropFromOwnerCellsPa "G8") | $(Format-NullablePercent $row.relativePressureDropErrorFromOwnerCells) |")
        }
    }
    $lines.Add("")
    $lines.Add("## Files")
    $lines.Add("")
    $lines.Add("- Summary JSON: $($Summary.summaryJson)")
    $lines.Add("- Generic reports: $($Summary.resultRoot)")
    $lines.Add("- Stored fields: $($Summary.fieldsRoot)")
    if ($Summary.runPipeBenchmark) {
        $lines.Add("- External benchmark inputs: $($Summary.benchmarkProperties)")
    }
    $lines.Add("")
    $lines.Add("The external benchmark never changes solver fields, iteration limits, or convergence decisions.")

    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Path) | Out-Null
    Set-Content -LiteralPath $Path -Value $lines -Encoding UTF8
}

$benchmarkPhysics = if ($RunPipeBenchmark) {
    Read-PipeBenchmarkPhysics -CaseRoot $CaseRoot -PropertiesPath $BenchmarkProperties
} else {
    $null
}
$resultRoot = Join-Path $StudyRoot "results"
$logRoot = Join-Path $StudyRoot "logs"
$fieldsRoot = Join-Path $StudyRoot "fields"
New-Item -ItemType Directory -Force -Path $StudyRoot, $resultRoot, $logRoot | Out-Null

$rows = New-Object System.Collections.Generic.List[object]
foreach ($iteration in $iterationBudgets) {
    Write-Output "running Ferrum laminar SIMPLE iteration budget $iteration"
    $runArguments = @{
        CaseRoot = $CaseRoot
        IterationBudget = $iteration
        ResultRoot = $resultRoot
        LogRoot = $logRoot
        FieldsRoot = $fieldsRoot
        WriteFields = $WriteFinalFields.IsPresent
        UseExistingReport = $UseExistingReports.IsPresent
        RunBenchmark = $RunPipeBenchmark
        BenchmarkPhysics = $benchmarkPhysics
    }
    $rows.Add((Invoke-LaminarSimpleIterationRun @runArguments)) | Out-Null
}

$summaryJson = Join-Path $StudyRoot "laminar_simple_iteration_sweep.json"
$reportFile = Join-Path $StudyRoot "laminar_simple_iteration_sweep.md"
$summary = [ordered]@{
    caseDir = $CaseRoot
    study = "laminar-simple-iteration-sweep"
    units = [ordered]@{
        default = "SI"
        pressure = "Pa"
        length = "m"
        velocity = "m/s"
    }
    generatedAt = (Get-Date).ToString("o", [System.Globalization.CultureInfo]::InvariantCulture)
    simpleIterations = @($iterationBudgets.ToArray())
    runPipeBenchmark = $RunPipeBenchmark
    benchmarkProperties = if ($RunPipeBenchmark) { $BenchmarkProperties } else { $null }
    writeFinalFields = ($WriteFinalFields.IsPresent -or $RunPipeBenchmark)
    resultRoot = $resultRoot
    logRoot = $logRoot
    fieldsRoot = if ($WriteFinalFields.IsPresent -or $RunPipeBenchmark) { $fieldsRoot } else { $null }
    summaryJson = $summaryJson
    reportFile = $reportFile
}
$payload = [ordered]@{
    summary = $summary
    rows = @($rows.ToArray())
}

$payload | ConvertTo-Json -Depth 14 | Set-Content -LiteralPath $summaryJson -Encoding UTF8
Write-SweepMarkdown -Path $reportFile -Rows $payload.rows -Summary ([pscustomobject]$summary)

Write-Output "wrote Ferrum laminar SIMPLE iteration sweep: $summaryJson"
Write-Output "wrote Ferrum laminar SIMPLE iteration report: $reportFile"
