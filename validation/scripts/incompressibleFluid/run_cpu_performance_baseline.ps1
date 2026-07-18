param(
    [int]$WarmupRuns = 1,
    [int]$MeasuredRuns = 5,
    [string]$OutRoot = "",
    [ValidateSet("fixed", "converged")]
    [string]$RunProfile = "fixed",
    [ValidateSet("pcg", "gamg")]
    [string]$PressureSolver = "pcg",
    [switch]$ProfileGamg,
    [switch]$RequireConverged,
    [switch]$ReuseExistingRuns
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
if ([string]::IsNullOrWhiteSpace($OutRoot)) {
    $profileSuffix = if ($ProfileGamg) { "-profiled" } else { "" }
    $OutRoot = Join-Path $RepoRoot "target\benchmarks\cpu_performance_baseline\$RunProfile-$PressureSolver$profileSuffix"
}
if ($WarmupRuns -lt 0) {
    throw "WarmupRuns must be zero or greater"
}
if ($MeasuredRuns -lt 1) {
    throw "MeasuredRuns must be at least one"
}
if ($ProfileGamg -and $PressureSolver -ne "gamg") {
    throw "ProfileGamg requires PressureSolver=gamg"
}

$cases = @(
    [pscustomobject][ordered]@{
        name = "laminarPipe"
        sourceCaseRoot = Join-Path $RepoRoot "tutorials\incompressibleFluid\laminarPipe\ferrum\case"
        fixedIterations = 10
        pcgConvergenceProfile = Join-Path $RepoRoot "validation\profiles\incompressibleFluid\laminarPipe\converged"
        gamgFixedProfile = Join-Path $RepoRoot "validation\profiles\incompressibleFluid\laminarPipe\gamg-fixed"
        gamgConvergenceProfile = Join-Path $RepoRoot "validation\profiles\incompressibleFluid\laminarPipe\gamg-converged"
    },
    [pscustomobject][ordered]@{
        name = "planeChannel"
        sourceCaseRoot = Join-Path $RepoRoot "tutorials\incompressibleFluid\planeChannel\ferrum\case"
        fixedIterations = 500
        pcgConvergenceProfile = Join-Path $RepoRoot "validation\profiles\incompressibleFluid\planeChannel\converged"
        gamgFixedProfile = Join-Path $RepoRoot "validation\profiles\incompressibleFluid\planeChannel\gamg-fixed"
        gamgConvergenceProfile = Join-Path $RepoRoot "validation\profiles\incompressibleFluid\planeChannel\gamg-converged"
    }
)

function Get-CaseProfileRoot($Case) {
    if ($PressureSolver -eq "gamg") {
        if ($RunProfile -eq "converged") {
            return $Case.gamgConvergenceProfile
        }
        return $Case.gamgFixedProfile
    }
    if ($RunProfile -eq "converged") {
        return $Case.pcgConvergenceProfile
    }
    return $null
}

foreach ($case in $cases) {
    if (!(Test-Path -LiteralPath $case.sourceCaseRoot -PathType Container)) {
        throw "performance baseline case was not found: $($case.sourceCaseRoot)"
    }
    $profileRoot = Get-CaseProfileRoot $case
    if ($null -ne $profileRoot -and !(Test-Path -LiteralPath $profileRoot -PathType Container)) {
        throw "performance profile was not found: $profileRoot"
    }
}

$convergenceRequired = $RequireConverged -or $RunProfile -eq "converged"
$expectedPressureLinearSolver = if ($PressureSolver -eq "gamg") { "GAMG" } else { "pcg" }

function Get-Median([double[]]$Values) {
    if ($Values.Count -eq 0) {
        return $null
    }
    $sorted = @($Values | Sort-Object)
    $middle = [int][Math]::Floor($sorted.Count / 2)
    if (($sorted.Count % 2) -eq 1) {
        return [double]$sorted[$middle]
    }
    return ([double]$sorted[$middle - 1] + [double]$sorted[$middle]) / 2.0
}

function Format-BenchmarkNumber([double]$Value) {
    return $Value.ToString("G8", [System.Globalization.CultureInfo]::InvariantCulture)
}

function Get-DoubleValues($Runs, [scriptblock]$Selector) {
    return [double[]]@($Runs | ForEach-Object { [double](& $Selector $_) })
}

function Get-TimingMedians($Runs) {
    $medians = [ordered]@{
        commandWallClockSeconds = Get-Median (Get-DoubleValues $Runs { param($run) $run.commandWallClockSeconds })
        solverTotalSeconds = Get-Median (Get-DoubleValues $Runs { param($run) $run.timing.solverTotalSeconds })
        driverMeasuredSeconds = Get-Median (Get-DoubleValues $Runs { param($run) $run.timing.driverMeasuredSeconds })
        setupSeconds = Get-Median (Get-DoubleValues $Runs { param($run) $run.timing.setupSeconds })
        iterationSetupSeconds = Get-Median (Get-DoubleValues $Runs { param($run) $run.timing.iterationSetupSeconds })
        operatorEvaluationSeconds = Get-Median (Get-DoubleValues $Runs { param($run) $run.timing.operatorEvaluationSeconds })
        momentumAssemblySeconds = Get-Median (Get-DoubleValues $Runs { param($run) $run.timing.momentumAssemblySeconds })
        momentumGradientSeconds = Get-Median (Get-DoubleValues $Runs { param($run) $run.timing.momentumGradientSeconds })
        momentumMatrixFillSeconds = Get-Median (Get-DoubleValues $Runs { param($run) $run.timing.momentumMatrixFillSeconds })
        momentumLinearSolveSeconds = Get-Median (Get-DoubleValues $Runs { param($run) $run.timing.momentumLinearSolveSeconds })
        pressureCouplingSetupSeconds = Get-Median (Get-DoubleValues $Runs { param($run) $run.timing.pressureCouplingSetupSeconds })
        pressureAssemblySeconds = Get-Median (Get-DoubleValues $Runs { param($run) $run.timing.pressureAssemblySeconds })
        pressureLinearSolveSeconds = Get-Median (Get-DoubleValues $Runs { param($run) $run.timing.pressureLinearSolveSeconds })
        pressurePcgTotalSeconds = Get-Median (Get-DoubleValues $Runs { param($run) $run.timing.pressurePcgTotalSeconds })
        pressurePreconditionerUpdateSeconds = Get-Median (Get-DoubleValues $Runs { param($run) $run.timing.pressurePreconditionerUpdateSeconds })
        pressureMatrixVectorSeconds = Get-Median (Get-DoubleValues $Runs { param($run) $run.timing.pressureMatrixVectorSeconds })
        pressurePreconditionerApplicationSeconds = Get-Median (Get-DoubleValues $Runs { param($run) $run.timing.pressurePreconditionerApplicationSeconds })
        pressureVectorOperationSeconds = Get-Median (Get-DoubleValues $Runs { param($run) $run.timing.pressureVectorOperationSeconds })
        pressurePcgOtherSeconds = Get-Median (Get-DoubleValues $Runs { param($run) $run.timing.pressurePcgOtherSeconds })
        pressureMatrixVectorProducts = Get-Median (Get-DoubleValues $Runs { param($run) $run.timing.pressureMatrixVectorProducts })
        pressurePreconditionerApplications = Get-Median (Get-DoubleValues $Runs { param($run) $run.timing.pressurePreconditionerApplications })
        fieldCorrectionSeconds = Get-Median (Get-DoubleValues $Runs { param($run) $run.timing.fieldCorrectionSeconds })
        finalizationSeconds = Get-Median (Get-DoubleValues $Runs { param($run) $run.timing.finalizationSeconds })
        otherSolverWorkSeconds = Get-Median (Get-DoubleValues $Runs { param($run) $run.timing.otherSolverWorkSeconds })
    }
    if ($ProfileGamg) {
        $medians.pressureGamgTotalSeconds = Get-Median (Get-DoubleValues $Runs { param($run) $run.pressureGamgProfile.totalSeconds })
        $medians.pressureGamgHierarchyBuildSeconds = Get-Median (Get-DoubleValues $Runs { param($run) $run.pressureGamgProfile.hierarchyBuildSeconds })
        $medians.pressureGamgHierarchyRebuildSeconds = Get-Median (Get-DoubleValues $Runs { param($run) $run.pressureGamgProfile.hierarchyRebuildSeconds })
        $medians.pressureGamgMatrixRefreshSeconds = Get-Median (Get-DoubleValues $Runs { param($run) $run.pressureGamgProfile.matrixRefreshSeconds })
        $medians.pressureGamgFinestResidualSeconds = Get-Median (Get-DoubleValues $Runs { param($run) $run.pressureGamgProfile.finestResidualSeconds })
        $medians.pressureGamgVCycleSeconds = Get-Median (Get-DoubleValues $Runs { param($run) $run.pressureGamgProfile.vCycleSeconds })
        $medians.pressureGamgRestrictionSeconds = Get-Median (Get-DoubleValues $Runs { param($run) $run.pressureGamgProfile.restrictionSeconds })
        $medians.pressureGamgProlongationSeconds = Get-Median (Get-DoubleValues $Runs { param($run) $run.pressureGamgProfile.prolongationSeconds })
        $medians.pressureGamgSmoothingSeconds = Get-Median (Get-DoubleValues $Runs { param($run) $run.pressureGamgProfile.smoothingSeconds })
        $medians.pressureGamgScalingSeconds = Get-Median (Get-DoubleValues $Runs { param($run) $run.pressureGamgProfile.scalingSeconds })
        $medians.pressureGamgCoarseResidualSeconds = Get-Median (Get-DoubleValues $Runs { param($run) $run.pressureGamgProfile.coarseResidualSeconds })
        $medians.pressureGamgCorrectionSeconds = Get-Median (Get-DoubleValues $Runs { param($run) $run.pressureGamgProfile.correctionSeconds })
        $medians.pressureGamgCoarsestSolveSeconds = Get-Median (Get-DoubleValues $Runs { param($run) $run.pressureGamgProfile.coarsestSolveSeconds })
        $medians.pressureGamgVCycleOtherSeconds = Get-Median (Get-DoubleValues $Runs { param($run) $run.pressureGamgProfile.vCycleOtherSeconds })
        $medians.pressureGamgOtherSeconds = Get-Median (Get-DoubleValues $Runs { param($run) $run.pressureGamgProfile.otherSeconds })
        $medians.pressureGamgSolves = Get-Median (Get-DoubleValues $Runs { param($run) $run.pressureGamgProfile.solves })
        $medians.pressureGamgVCycles = Get-Median (Get-DoubleValues $Runs { param($run) $run.pressureGamgProfile.vCycles })
    }
    return [pscustomobject]$medians
}

function Get-GamgLevelMedians($Runs) {
    $profiles = @($Runs | ForEach-Object { $_.pressureGamgProfile })
    if ($profiles.Count -eq 0 -or $null -eq $profiles[0]) {
        return @()
    }
    $levelCount = @($profiles[0].levels).Count
    foreach ($profile in $profiles) {
        if ($null -eq $profile -or @($profile.levels).Count -ne $levelCount) {
            throw "GAMG profile hierarchy changed between measured runs"
        }
    }
    $result = @()
    for ($levelIndex = 0; $levelIndex -lt $levelCount; $levelIndex++) {
        $reference = $profiles[0].levels[$levelIndex]
        foreach ($profile in $profiles) {
            $candidate = $profile.levels[$levelIndex]
            if ($candidate.level -ne $reference.level -or $candidate.cells -ne $reference.cells -or $candidate.nonzeros -ne $reference.nonzeros) {
                throw "GAMG profile level $levelIndex metadata changed between measured runs"
            }
        }
        $result += [pscustomobject][ordered]@{
            level = [int]$reference.level
            cells = [int]$reference.cells
            nonzeros = [int]$reference.nonzeros
            matrixRefreshSeconds = Get-Median ([double[]]@($profiles | ForEach-Object { [double]$_.levels[$levelIndex].matrixRefreshSeconds }))
            restrictionSeconds = Get-Median ([double[]]@($profiles | ForEach-Object { [double]$_.levels[$levelIndex].restrictionSeconds }))
            prolongationSeconds = Get-Median ([double[]]@($profiles | ForEach-Object { [double]$_.levels[$levelIndex].prolongationSeconds }))
            smoothingSeconds = Get-Median ([double[]]@($profiles | ForEach-Object { [double]$_.levels[$levelIndex].smoothingSeconds }))
            scalingSeconds = Get-Median ([double[]]@($profiles | ForEach-Object { [double]$_.levels[$levelIndex].scalingSeconds }))
            residualSeconds = Get-Median ([double[]]@($profiles | ForEach-Object { [double]$_.levels[$levelIndex].residualSeconds }))
            correctionSeconds = Get-Median ([double[]]@($profiles | ForEach-Object { [double]$_.levels[$levelIndex].correctionSeconds }))
            coarsestSolveSeconds = Get-Median ([double[]]@($profiles | ForEach-Object { [double]$_.levels[$levelIndex].coarsestSolveSeconds }))
            restrictionCalls = Get-Median ([double[]]@($profiles | ForEach-Object { [double]$_.levels[$levelIndex].restrictionCalls }))
            prolongationCalls = Get-Median ([double[]]@($profiles | ForEach-Object { [double]$_.levels[$levelIndex].prolongationCalls }))
            smoothingCalls = Get-Median ([double[]]@($profiles | ForEach-Object { [double]$_.levels[$levelIndex].smoothingCalls }))
            smoothingSweeps = Get-Median ([double[]]@($profiles | ForEach-Object { [double]$_.levels[$levelIndex].smoothingSweeps }))
            scalingCalls = Get-Median ([double[]]@($profiles | ForEach-Object { [double]$_.levels[$levelIndex].scalingCalls }))
            residualEvaluations = Get-Median ([double[]]@($profiles | ForEach-Object { [double]$_.levels[$levelIndex].residualEvaluations }))
            correctionUpdates = Get-Median ([double[]]@($profiles | ForEach-Object { [double]$_.levels[$levelIndex].correctionUpdates }))
            coarsestSolves = Get-Median ([double[]]@($profiles | ForEach-Object { [double]$_.levels[$levelIndex].coarsestSolves }))
        }
    }
    return $result
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

function New-ProfiledCase(
    [string]$Name,
    [string]$SourceCaseRoot,
    [string]$ProfileRoot
) {
    $workingCasesRoot = Join-Path $OutRoot "working-cases"
    $workingCaseRoot = Join-Path $workingCasesRoot $Name
    if (Test-Path -LiteralPath $workingCaseRoot) {
        if (!(Test-IsPathUnder $workingCaseRoot $OutRoot)) {
            throw "refusing to replace working case outside benchmark output: $workingCaseRoot"
        }
        Remove-Item -LiteralPath $workingCaseRoot -Recurse -Force
    }

    New-Item -ItemType Directory -Force -Path $workingCasesRoot | Out-Null
    Copy-Item -LiteralPath $SourceCaseRoot -Destination $workingCaseRoot -Recurse
    Copy-Item -Path (Join-Path $ProfileRoot "system\*") -Destination (Join-Path $workingCaseRoot "system") -Force
    return $workingCaseRoot
}

function Invoke-FerrumCase(
    [string]$Name,
    [string]$CaseRoot,
    [string]$Executable,
    [int]$FixedIterations,
    [int]$Warmups,
    [int]$Measurements,
    [string]$ExpectedPressureLinearSolver
) {
    $caseOut = Join-Path $OutRoot $Name
    New-Item -ItemType Directory -Force -Path $caseOut | Out-Null
    $allRuns = @()
    $totalRuns = $Warmups + $Measurements

    for ($index = 1; $index -le $totalRuns; $index++) {
        $isWarmup = $index -le $Warmups
        $kind = if ($isWarmup) { "warmup" } else { "measured" }
        $ordinal = if ($isWarmup) { $index } else { $index - $Warmups }
        $baseName = "$kind-$ordinal"
        $reportJson = Join-Path $caseOut "$baseName.solver.json"
        $logPath = Join-Path $caseOut "$baseName.log"
        $arguments = @(
            "-solver", "incompressibleFluid",
            "-case", $CaseRoot,
            "--solveReportJson", $reportJson
        )
        if ($RunProfile -eq "fixed") {
            $arguments += @(
                "--minSimpleIterations", $FixedIterations.ToString([System.Globalization.CultureInfo]::InvariantCulture),
                "--maxSimpleIterations", $FixedIterations.ToString([System.Globalization.CultureInfo]::InvariantCulture)
            )
        }
        if ($ExpectedPressureLinearSolver -eq "GAMG" -and $ProfileGamg) {
            $arguments += "--profileGamg"
        }

        $reused = $ReuseExistingRuns -and (Test-Path -LiteralPath $reportJson -PathType Leaf)
        if ($reused) {
            $report = Get-Content -LiteralPath $reportJson -Raw | ConvertFrom-Json
            $commandWallClockSeconds = [double]$report.timing.driverMeasuredSeconds
        } else {
            $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
            $previousErrorActionPreference = $ErrorActionPreference
            try {
                $ErrorActionPreference = "Continue"
                & $Executable @arguments *> $logPath
                $exitCode = $LASTEXITCODE
            } finally {
                $ErrorActionPreference = $previousErrorActionPreference
                $stopwatch.Stop()
            }
            if ($exitCode -ne 0) {
                throw "Ferrum performance run failed for $Name with exit code $exitCode. See $logPath"
            }
            if (!(Test-Path -LiteralPath $reportJson -PathType Leaf)) {
                throw "Ferrum performance run did not write $reportJson"
            }
            $report = Get-Content -LiteralPath $reportJson -Raw | ConvertFrom-Json
            $commandWallClockSeconds = $stopwatch.Elapsed.TotalSeconds
        }
        $actualPressureLinearSolver = [string]$report.options.pressureLinearSolver
        if (!$actualPressureLinearSolver.Equals($ExpectedPressureLinearSolver, [System.StringComparison]::OrdinalIgnoreCase)) {
            throw "$Name requested pressure solver $ExpectedPressureLinearSolver but report used $actualPressureLinearSolver"
        }
        if ($ExpectedPressureLinearSolver -eq "GAMG" -and $null -eq $report.options.pressureGamg) {
            throw "$Name used GAMG but the solver report did not include pressureGamg controls"
        }
        if ($ExpectedPressureLinearSolver -eq "GAMG" -and $ProfileGamg -and $null -eq $report.timing.pressureGamgProfile) {
            throw "$Name used GAMG profiling but the solver report did not include pressureGamgProfile"
        }
        if (!$ProfileGamg -and $null -ne $report.timing.pressureGamgProfile) {
            throw "$Name unexpectedly reported pressureGamgProfile in an unprofiled performance run"
        }
        $run = [pscustomobject][ordered]@{
            kind = $kind
            ordinal = $ordinal
            reused = $reused
            commandWallClockSeconds = $commandWallClockSeconds
            reportJson = $reportJson
            log = $logPath
            simpleIterations = [int]$report.solve.simpleIterations
            converged = [bool]$report.solve.converged
            stopReason = [string]$report.solve.stopReason
            outerConvergenceStatus = [string]$report.outerConvergence.status
            momentumLinearIterations = [int]$report.solve.momentumLinearIterations
            pressureLinearIterations = [int]$report.solve.pressureLinearIterations
            pressureLinearSolver = $actualPressureLinearSolver
            pressureGamg = $report.options.pressureGamg
            pressureGamgProfile = $report.timing.pressureGamgProfile
            finalContinuityL2 = [double]$report.continuity.final.l2Norm
            finalMomentumResidual = [double]$report.solve.finalMomentumNormalizedResidualNorm
            finalPressureResidual = [double]$report.solve.finalPressureCorrectionNormalizedResidualNorm
            velocityL2 = [double]$report.fields.velocity.l2Norm
            pressureL2 = [double]$report.fields.pressure.l2Norm
            timing = $report.timing
        }
        $allRuns += $run
        Write-Host ("{0} {1} {2}: reused={3} converged={4} SIMPLE={5} solver={6:F6}s command={7:F6}s" -f $Name, $kind, $ordinal, $run.reused, $run.converged, $run.simpleIterations, $run.timing.solverTotalSeconds, $run.commandWallClockSeconds)
    }

    $measured = @($allRuns | Where-Object { $_.kind -eq "measured" })
    if ($convergenceRequired -and @($measured | Where-Object { !$_.converged }).Count -gt 0) {
        throw "$Name contains a non-converged measured run"
    }

    return [pscustomobject][ordered]@{
        name = $Name
        caseRoot = $CaseRoot
        pressureLinearSolver = $ExpectedPressureLinearSolver
        fixedIterations = if ($RunProfile -eq "fixed") { $FixedIterations } else { $null }
        warmupRuns = $Warmups
        measuredRuns = $Measurements
        medians = Get-TimingMedians $measured
        gamgLevelMedians = if ($ExpectedPressureLinearSolver -eq "GAMG" -and $ProfileGamg) { @(Get-GamgLevelMedians $measured) } else { @() }
        numerical = [pscustomobject][ordered]@{
            simpleIterations = @($measured | ForEach-Object { $_.simpleIterations })
            converged = @($measured | ForEach-Object { $_.converged })
            pressureLinearIterations = @($measured | ForEach-Object { $_.pressureLinearIterations })
            finalContinuityL2 = @($measured | ForEach-Object { $_.finalContinuityL2 })
            finalMomentumResidual = @($measured | ForEach-Object { $_.finalMomentumResidual })
            finalPressureResidual = @($measured | ForEach-Object { $_.finalPressureResidual })
            velocityL2 = @($measured | ForEach-Object { $_.velocityL2 })
            pressureL2 = @($measured | ForEach-Object { $_.pressureL2 })
        }
        runs = $allRuns
    }
}

New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null
$buildLog = Join-Path $OutRoot "cargo-build-release.log"
$buildStopwatch = [System.Diagnostics.Stopwatch]::StartNew()
Push-Location $RepoRoot
try {
    $previousErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        & cargo build --locked --release -p ferrum-run --bin ferrumRun *> $buildLog
        $buildExitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }
} finally {
    Pop-Location
    $buildStopwatch.Stop()
}
if ($buildExitCode -ne 0) {
    throw "Ferrum release build failed with exit code $buildExitCode. See $buildLog"
}

$executable = Join-Path $RepoRoot "target\release\ferrumRun.exe"
if (!(Test-Path -LiteralPath $executable -PathType Leaf)) {
    throw "Ferrum release executable was not found after the build: $executable"
}

$caseResults = @()
foreach ($case in $cases) {
    $profileRoot = Get-CaseProfileRoot $case
    $caseRoot = if ($null -ne $profileRoot) {
        New-ProfiledCase `
            -Name $case.name `
            -SourceCaseRoot $case.sourceCaseRoot `
            -ProfileRoot $profileRoot
    } else {
        $case.sourceCaseRoot
    }
    $caseResults += Invoke-FerrumCase `
        -Name $case.name `
        -CaseRoot $caseRoot `
        -Executable $executable `
        -FixedIterations $case.fixedIterations `
        -Warmups $WarmupRuns `
        -Measurements $MeasuredRuns `
        -ExpectedPressureLinearSolver $expectedPressureLinearSolver
}

$summary = [pscustomobject][ordered]@{
    schemaVersion = 1
    benchmark = "ferrum-cpu-performance-baseline"
    runProfile = $RunProfile
    pressureSolver = $PressureSolver
    gamgProfilingEnabled = [bool]$ProfileGamg
    generatedAtUtc = [DateTime]::UtcNow.ToString("o")
    build = [pscustomobject][ordered]@{
        profile = "release"
        command = "cargo build --locked --release -p ferrum-run --bin ferrumRun"
        wallClockSeconds = $buildStopwatch.Elapsed.TotalSeconds
        log = $buildLog
        executable = $executable
    }
    policy = [pscustomobject][ordered]@{
        compilationExcludedFromRunTiming = $true
        warmupRunsExcludedFromMedian = $true
        medianMeasuredRuns = $MeasuredRuns
        convergenceRequired = [bool]$convergenceRequired
        reusedExistingRuns = [bool]$ReuseExistingRuns
        gamgProfilingExcludedFromNormalTiming = !$ProfileGamg
        numericalCasesRemainIndependent = $true
        benchmarkCriteriaRemainExternalToSolver = $true
    }
    cases = $caseResults
}

$jsonPath = Join-Path $OutRoot "summary.json"
$markdownPath = Join-Path $OutRoot "summary.md"
$summary | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding UTF8

$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# Ferrum CPU Performance Baseline: $RunProfile / $PressureSolver")
$lines.Add("")
$lines.Add("Release executable: ``$executable``")
$lines.Add("Run profile: ``$RunProfile``")
$lines.Add("Pressure solver: ``$PressureSolver``")
$lines.Add("GAMG profiling: ``$([bool]$ProfileGamg)``")
$lines.Add("")
$lines.Add("Build time is recorded separately and excluded from every solver median.")
$lines.Add("")
$lines.Add("| Case | Converged | Stop reason | SIMPLE | Solver total [s] | Operators [s] | Momentum assembly [s] | Momentum gradients [s] | Momentum matrix fill [s] | Momentum solve [s] | Coupling setup [s] | Pressure assembly [s] | Pressure solve [s] | Field correction [s] | Other [s] |")
$lines.Add("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
foreach ($case in $caseResults) {
    $simpleIterations = ($case.numerical.simpleIterations | Select-Object -First 1)
    $converged = ($case.numerical.converged | Select-Object -First 1)
    $stopReason = ($case.runs | Where-Object { $_.kind -eq "measured" } | Select-Object -First 1).stopReason
    $timing = $case.medians
    $lines.Add(("| {0} | {1} | {2} | {3} | {4} | {5} | {6} | {7} | {8} | {9} | {10} | {11} | {12} | {13} | {14} |" -f $case.name, $converged, $stopReason, $simpleIterations, (Format-BenchmarkNumber $timing.solverTotalSeconds), (Format-BenchmarkNumber $timing.operatorEvaluationSeconds), (Format-BenchmarkNumber $timing.momentumAssemblySeconds), (Format-BenchmarkNumber $timing.momentumGradientSeconds), (Format-BenchmarkNumber $timing.momentumMatrixFillSeconds), (Format-BenchmarkNumber $timing.momentumLinearSolveSeconds), (Format-BenchmarkNumber $timing.pressureCouplingSetupSeconds), (Format-BenchmarkNumber $timing.pressureAssemblySeconds), (Format-BenchmarkNumber $timing.pressureLinearSolveSeconds), (Format-BenchmarkNumber $timing.fieldCorrectionSeconds), (Format-BenchmarkNumber $timing.otherSolverWorkSeconds)))
}
$lines.Add("")
if ($PressureSolver -eq "pcg") {
    $lines.Add("## Pressure PCG Kernel")
    $lines.Add("")
    $lines.Add("| Case | Pressure solve [s] | PCG total [s] | Preconditioner update [s] | Matrix-vector [s] | Preconditioner apply [s] | Vector operations [s] | PCG other [s] | Matrix-vector calls | Preconditioner calls |")
    $lines.Add("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    foreach ($case in $caseResults) {
        $timing = $case.medians
        $lines.Add(("| {0} | {1} | {2} | {3} | {4} | {5} | {6} | {7} | {8} | {9} |" -f $case.name, (Format-BenchmarkNumber $timing.pressureLinearSolveSeconds), (Format-BenchmarkNumber $timing.pressurePcgTotalSeconds), (Format-BenchmarkNumber $timing.pressurePreconditionerUpdateSeconds), (Format-BenchmarkNumber $timing.pressureMatrixVectorSeconds), (Format-BenchmarkNumber $timing.pressurePreconditionerApplicationSeconds), (Format-BenchmarkNumber $timing.pressureVectorOperationSeconds), (Format-BenchmarkNumber $timing.pressurePcgOtherSeconds), (Format-BenchmarkNumber $timing.pressureMatrixVectorProducts), (Format-BenchmarkNumber $timing.pressurePreconditionerApplications)))
    }
} else {
    $lines.Add("## Pressure GAMG")
    $lines.Add("")
    $lines.Add("| Case | Pressure solve [s] | Pressure linear iterations | Agglomerator | Smoother | Cache agglomeration | Coarsest-level cells |")
    $lines.Add("| --- | ---: | ---: | --- | --- | --- | ---: |")
    foreach ($case in $caseResults) {
        $timing = $case.medians
        $run = @($case.runs | Where-Object { $_.kind -eq "measured" })[0]
        $gamg = $run.pressureGamg
        $iterations = $case.numerical.pressureLinearIterations[0]
        $lines.Add(("| {0} | {1} | {2} | {3} | {4} | {5} | {6} |" -f $case.name, (Format-BenchmarkNumber $timing.pressureLinearSolveSeconds), $iterations, $gamg.agglomerator, $gamg.smoother, $gamg.cacheAgglomeration, $gamg.nCellsInCoarsestLevel))
    }
    if ($ProfileGamg) {
        $lines.Add("")
        $lines.Add("## Pressure GAMG Cycle Phases")
        $lines.Add("")
        $lines.Add("| Case | Profile total [s] | Build [s] | Refresh [s] | Finest residual [s] | V-cycle [s] | Restrict [s] | Prolong [s] | Smooth [s] | Scale [s] | Coarse residual [s] | Correction [s] | Coarsest [s] | Cycle other [s] | Profile other [s] | V-cycles |")
        $lines.Add("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        foreach ($case in $caseResults) {
            $timing = $case.medians
            $lines.Add(("| {0} | {1} | {2} | {3} | {4} | {5} | {6} | {7} | {8} | {9} | {10} | {11} | {12} | {13} | {14} | {15} |" -f $case.name, (Format-BenchmarkNumber $timing.pressureGamgTotalSeconds), (Format-BenchmarkNumber $timing.pressureGamgHierarchyBuildSeconds), (Format-BenchmarkNumber $timing.pressureGamgMatrixRefreshSeconds), (Format-BenchmarkNumber $timing.pressureGamgFinestResidualSeconds), (Format-BenchmarkNumber $timing.pressureGamgVCycleSeconds), (Format-BenchmarkNumber $timing.pressureGamgRestrictionSeconds), (Format-BenchmarkNumber $timing.pressureGamgProlongationSeconds), (Format-BenchmarkNumber $timing.pressureGamgSmoothingSeconds), (Format-BenchmarkNumber $timing.pressureGamgScalingSeconds), (Format-BenchmarkNumber $timing.pressureGamgCoarseResidualSeconds), (Format-BenchmarkNumber $timing.pressureGamgCorrectionSeconds), (Format-BenchmarkNumber $timing.pressureGamgCoarsestSolveSeconds), (Format-BenchmarkNumber $timing.pressureGamgVCycleOtherSeconds), (Format-BenchmarkNumber $timing.pressureGamgOtherSeconds), (Format-BenchmarkNumber $timing.pressureGamgVCycles)))
        }
        foreach ($case in $caseResults) {
            $lines.Add("")
            $lines.Add("### $($case.name) GAMG Levels")
            $lines.Add("")
            $lines.Add("| Level | Cells | NNZ | Refresh [s] | Restrict [s] | Prolong [s] | Smooth [s] | Scale [s] | Residual [s] | Correction [s] | Coarsest [s] | Smooth sweeps |")
            $lines.Add("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
            foreach ($level in $case.gamgLevelMedians) {
                $lines.Add(("| {0} | {1} | {2} | {3} | {4} | {5} | {6} | {7} | {8} | {9} | {10} | {11} |" -f $level.level, $level.cells, $level.nonzeros, (Format-BenchmarkNumber $level.matrixRefreshSeconds), (Format-BenchmarkNumber $level.restrictionSeconds), (Format-BenchmarkNumber $level.prolongationSeconds), (Format-BenchmarkNumber $level.smoothingSeconds), (Format-BenchmarkNumber $level.scalingSeconds), (Format-BenchmarkNumber $level.residualSeconds), (Format-BenchmarkNumber $level.correctionSeconds), (Format-BenchmarkNumber $level.coarsestSolveSeconds), (Format-BenchmarkNumber $level.smoothingSweeps)))
            }
        }
    } else {
        $lines.Add("")
        $lines.Add("GAMG cycle profiling is disabled for this timing run. Re-run with ``-ProfileGamg`` for diagnostic phase and hierarchy tables.")
    }
}
$lines.Add("")
$lines.Add("The pipe and plane-channel cases are regression inputs. Validation profiles are copied into disposable working cases below ``target/`` and do not add case-specific behavior to the solver.")
Set-Content -LiteralPath $markdownPath -Value $lines -Encoding UTF8

Write-Output "wrote Ferrum CPU performance baseline JSON: $jsonPath"
Write-Output "wrote Ferrum CPU performance baseline Markdown: $markdownPath"
