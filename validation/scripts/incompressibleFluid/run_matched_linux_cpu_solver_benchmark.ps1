param(
    [int]$WarmupRuns = 2,
    [int]$MeasuredRuns = 9,
    [ValidateSet("pcg", "gamg")]
    [string]$PressureSolver = "gamg",
    [ValidateSet("all", "laminarPipe", "planeChannel")]
    [string]$CaseName = "all",
    [string]$Distro = "Ubuntu-22.04",
    [string]$CpuSet = "2",
    [ValidateSet("portable", "native")]
    [string]$BuildVariant = "portable",
    [string]$RustToolchain = "1.94.0",
    [string]$SourceRef = "HEAD",
    [string]$OutRoot = "",
    [switch]$PreflightOnly,
    [switch]$KeepWslWorkspace
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "matched_cpu_solver_common.ps1")

$RepoRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
$TargetRoot = Join-Path $RepoRoot "target"
$WorkerPath = Join-Path $PSScriptRoot "run_matched_linux_cpu_solver_worker.sh"
if (!(Test-Path -LiteralPath $WorkerPath -PathType Leaf)) {
    throw "Linux parity worker was not found: $WorkerPath"
}
if ($WarmupRuns -lt 0) { throw "WarmupRuns must be zero or greater" }
if ($MeasuredRuns -lt 1) { throw "MeasuredRuns must be at least one" }
if ($CpuSet -notmatch "^[0-9]+([,-][0-9]+)*$") { throw "CpuSet is invalid: $CpuSet" }
if ($null -eq (Get-Command wsl -ErrorAction SilentlyContinue)) { throw "wsl.exe was not found" }

$workerWslPath = ConvertTo-MatchedWslPath $WorkerPath $Distro
$workerBootstrap = 'set -o pipefail; tr -d ''\r'' < "\$1" | bash -s -- "\${@:2}"'
$preflightArguments = @(
    "-d", $Distro, "--", "bash", "-c", $workerBootstrap, "ferrum-linux-worker", $workerWslPath,
    "--preflight-only",
    "--rust-toolchain", $RustToolchain,
    "--cpu-set", $CpuSet,
    "--build-variant", $BuildVariant,
    "--warmup-runs", $WarmupRuns.ToString([System.Globalization.CultureInfo]::InvariantCulture),
    "--measured-runs", $MeasuredRuns.ToString([System.Globalization.CultureInfo]::InvariantCulture),
    "--pressure-solver", $PressureSolver
)
$previousErrorActionPreference = $ErrorActionPreference
try {
    $ErrorActionPreference = "Continue"
    $preflightOutput = & wsl @preflightArguments 2>&1
    $preflightExitCode = $LASTEXITCODE
} finally {
    $ErrorActionPreference = $previousErrorActionPreference
}
if ($preflightExitCode -ne 0) {
    throw "Linux parity preflight failed for '$Distro':`n$($preflightOutput -join "`n")"
}
if ($PreflightOnly) {
    $preflightOutput | Write-Output
    return
}

if ([string]::IsNullOrWhiteSpace($OutRoot)) {
    $OutRoot = Join-Path $TargetRoot "benchmarks\matched_linux_cpu_solver\$PressureSolver-$BuildVariant"
}
$OutRoot = [System.IO.Path]::GetFullPath($OutRoot)
$targetRootFull = [System.IO.Path]::GetFullPath($TargetRoot).TrimEnd("\", "/")
if (!(Test-MatchedPathUnder $OutRoot $targetRootFull) -or
    $OutRoot.Equals($targetRootFull, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "OutRoot must remain below the repository target directory: $OutRoot"
}

$sourceCommit = (& git -C $RepoRoot rev-parse "$SourceRef`^{commit}").Trim()
if ($LASTEXITCODE -ne 0 -or $sourceCommit -notmatch "^[0-9a-f]{40}$") {
    throw "could not resolve SourceRef '$SourceRef' to an exact commit"
}
$sourceTree = (& git -C $RepoRoot rev-parse "$sourceCommit`^{tree}").Trim()
if ($LASTEXITCODE -ne 0 -or $sourceTree -notmatch "^[0-9a-f]{40}$") {
    throw "could not resolve the tree for commit '$sourceCommit'"
}
$sourceStatus = @(& git -C $RepoRoot status --porcelain=v1)
$sourceWorktreeClean = $sourceStatus.Count -eq 0
$hostPowerPlan = if ($null -ne (Get-Command powercfg -ErrorAction SilentlyContinue)) {
    ((& powercfg /getactivescheme 2>$null) -join " ").Trim()
} else {
    "unavailable"
}

$stageRoot = Join-Path $TargetRoot "benchmarks\.linux-parity-stage-$PID"
Reset-MatchedTargetDirectory $stageRoot $TargetRoot
$completed = $false
try {
    $sourceArchive = Join-Path $stageRoot "source.tar"
    & git -C $RepoRoot archive --format=tar --output=$sourceArchive $sourceCommit
    if ($LASTEXITCODE -ne 0 -or !(Test-Path -LiteralPath $sourceArchive -PathType Leaf)) {
        throw "could not archive exact source commit '$sourceCommit'"
    }
    $sourceArchiveSha256 = (Get-FileHash -LiteralPath $sourceArchive -Algorithm SHA256).Hash.ToLowerInvariant()
    Assert-MatchedSafeTarArchive $sourceArchive "exact source"

    $sourceHostRoot = Join-Path $stageRoot "source"
    New-Item -ItemType Directory -Force -Path $sourceHostRoot | Out-Null
    Assert-MatchedNoReparsePath $sourceHostRoot $stageRoot
    & tar -xf $sourceArchive -C $sourceHostRoot
    if ($LASTEXITCODE -ne 0) { throw "could not extract the exact source archive for case preparation" }

    $contract = Get-MatchedCpuCaseDefinitions $sourceHostRoot $PressureSolver $CaseName
    $templatesRoot = Join-Path $stageRoot "templates"
    New-Item -ItemType Directory -Force -Path $templatesRoot | Out-Null
    $manifestCases = @()
    foreach ($case in $contract.cases) {
        $caseTemplateRoot = Join-Path $templatesRoot $case.name
        New-Item -ItemType Directory -Force -Path $caseTemplateRoot | Out-Null
        $canonicalHashes = Get-MatchedPolyMeshHashes $case.ferrumCase
        $ferrumDestination = Join-Path $caseTemplateRoot "ferrum"
        $openFoamDestination = Join-Path $caseTemplateRoot "openfoam"
        New-MatchedFerrumWorkingCase $case $ferrumDestination $contract.fvSolution $templatesRoot | Out-Null
        Assert-MatchedHashesEqual $canonicalHashes (Get-MatchedPolyMeshHashes $ferrumDestination) "$($case.name) staged Ferrum"
        $pressureConversion = New-MatchedOpenFoamWorkingCase `
            $case $openFoamDestination $contract.fvSolution $canonicalHashes $templatesRoot
        Assert-MatchedHashesEqual $canonicalHashes (Get-MatchedPolyMeshHashes $openFoamDestination) "$($case.name) staged OpenFOAM"
        foreach ($relativePath in @("0\U", "system\fvSchemes", "system\fvSolution")) {
            $ferrumHash = (Get-FileHash -LiteralPath (Join-Path $ferrumDestination $relativePath) -Algorithm SHA256).Hash
            $openFoamHash = (Get-FileHash -LiteralPath (Join-Path $openFoamDestination $relativePath) -Algorithm SHA256).Hash
            if ($ferrumHash -ne $openFoamHash) { throw "$($case.name) staged shared file differs: $relativePath" }
        }
        $manifestCases += [pscustomobject][ordered]@{
            name = $case.name
            fixedIterations = $case.fixedIterations
            canonicalPolyMeshSha256 = $canonicalHashes
            sharedFileSha256 = [pscustomobject][ordered]@{
                velocity = (Get-FileHash -LiteralPath (Join-Path $ferrumDestination "0\U") -Algorithm SHA256).Hash.ToLowerInvariant()
                fvSchemes = (Get-FileHash -LiteralPath (Join-Path $ferrumDestination "system\fvSchemes") -Algorithm SHA256).Hash.ToLowerInvariant()
                fvSolution = (Get-FileHash -LiteralPath (Join-Path $ferrumDestination "system\fvSolution") -Algorithm SHA256).Hash.ToLowerInvariant()
            }
            pressureConversion = $pressureConversion
        }
    }
    $inputManifest = [pscustomobject][ordered]@{
        schemaVersion = 1
        sourceCommit = $sourceCommit
        sourceTree = $sourceTree
        pressureSolver = $PressureSolver
        cases = $manifestCases
    }
    $manifestPath = Join-Path $stageRoot "input-manifest.json"
    $inputManifest | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $manifestPath -Encoding UTF8
    $templatesArchive = Join-Path $stageRoot "templates.tar"
    & tar -cf $templatesArchive -C $templatesRoot .
    if ($LASTEXITCODE -ne 0) { throw "could not create the matched case-template archive" }
    Assert-MatchedSafeTarArchive $templatesArchive "matched case-template"
    $templatesArchiveSha256 = (Get-FileHash -LiteralPath $templatesArchive -Algorithm SHA256).Hash.ToLowerInvariant()

    $outputArchive = Join-Path $stageRoot "linux-parity-results.tar"
    $runArguments = @(
        "-d", $Distro, "--", "bash", "-c", $workerBootstrap, "ferrum-linux-worker", $workerWslPath,
        "--rust-toolchain", $RustToolchain,
        "--cpu-set", $CpuSet,
        "--build-variant", $BuildVariant,
        "--warmup-runs", $WarmupRuns.ToString([System.Globalization.CultureInfo]::InvariantCulture),
        "--measured-runs", $MeasuredRuns.ToString([System.Globalization.CultureInfo]::InvariantCulture),
        "--pressure-solver", $PressureSolver,
        "--source-archive", (ConvertTo-MatchedWslPath $sourceArchive $Distro),
        "--source-archive-sha256", $sourceArchiveSha256,
        "--templates-archive", (ConvertTo-MatchedWslPath $templatesArchive $Distro),
        "--templates-archive-sha256", $templatesArchiveSha256,
        "--manifest", (ConvertTo-MatchedWslPath $manifestPath $Distro),
        "--output-archive", (ConvertTo-MatchedWslPath $outputArchive $Distro),
        "--source-commit", $sourceCommit,
        "--source-tree", $sourceTree
    )
    if ($KeepWslWorkspace) { $runArguments += "--keep-workspace" }
    $previousErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        & wsl @runArguments
        $workerExitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }
    if ($workerExitCode -ne 0) { throw "Linux parity worker failed with exit code $workerExitCode" }
    if (!(Test-Path -LiteralPath $outputArchive -PathType Leaf)) { throw "Linux parity output archive was not returned" }
    $sidecarPath = "$outputArchive.sha256"
    if (!(Test-Path -LiteralPath $sidecarPath -PathType Leaf)) { throw "Linux parity archive SHA-256 sidecar was not returned" }
    $expectedArchiveSha256 = (Get-Content -LiteralPath $sidecarPath -Raw).Trim()
    $actualArchiveSha256 = (Get-FileHash -LiteralPath $outputArchive -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($expectedArchiveSha256 -ne $actualArchiveSha256) { throw "Linux parity result archive failed SHA-256 verification" }
    Assert-MatchedSafeTarArchive $outputArchive "Linux parity result"

    Reset-MatchedTargetDirectory $OutRoot $TargetRoot
    Assert-MatchedNoReparsePath $OutRoot $TargetRoot
    & tar -xf $outputArchive -C $OutRoot
    if ($LASTEXITCODE -ne 0) { throw "Linux parity result archive could not be extracted" }
    Copy-Item -LiteralPath $manifestPath -Destination (Join-Path $OutRoot "input-manifest.json") -Force

    $expectedOrderByRun = @{}
    $expectedRunDirectoriesByCase = @{}
    foreach ($case in $contract.cases) {
        $expectedRunDirectories = New-Object System.Collections.Generic.List[string]
        foreach ($kind in @("warmup", "measured")) {
            $count = if ($kind -eq "warmup") { $WarmupRuns } else { $MeasuredRuns }
            for ($ordinal = 1; $ordinal -le $count; $ordinal++) {
                $runIndex = if ($kind -eq "warmup") { $ordinal } else { $WarmupRuns + $ordinal }
                $engineOrder = if (($runIndex % 2) -eq 1) { @("ferrum", "openfoam") } else { @("openfoam", "ferrum") }
                for ($position = 1; $position -le 2; $position++) {
                    $engine = $engineOrder[$position - 1]
                    $key = "$($case.name)|$kind|$ordinal|$engine"
                    $expectedOrderByRun[$key] = $position
                    $expectedRunDirectories.Add("$kind-$ordinal-$engine")
                }
            }
        }
        $expectedRunDirectoriesByCase[$case.name] = @($expectedRunDirectories)
    }

    $orderRows = @(Import-Csv -LiteralPath (Join-Path $OutRoot "metadata\run-order.tsv") -Delimiter "`t")
    if ($orderRows.Count -ne $expectedOrderByRun.Count) {
        throw "run-order row count was $($orderRows.Count), expected $($expectedOrderByRun.Count)"
    }
    $orderByRun = @{}
    foreach ($row in $orderRows) {
        $key = "$($row.case)|$($row.kind)|$($row.ordinal)|$($row.engine)"
        if (!$expectedOrderByRun.ContainsKey($key)) { throw "run-order contains an unexpected key: $key" }
        if ($orderByRun.ContainsKey($key)) { throw "run-order contains a duplicate key: $key" }
        $position = 0
        if (![int]::TryParse([string]$row.position, [ref]$position) -or $position -ne $expectedOrderByRun[$key]) {
            throw "run-order position for '$key' was '$($row.position)', expected $($expectedOrderByRun[$key])"
        }
        $orderByRun[$key] = $position
    }
    foreach ($key in $expectedOrderByRun.Keys) {
        if (!$orderByRun.ContainsKey($key)) { throw "run-order is missing expected key: $key" }
    }

    $rawRoot = Join-Path $OutRoot "raw"
    if (!(Test-Path -LiteralPath $rawRoot -PathType Container)) { throw "Linux parity raw result root was not found" }
    if (@(Get-ChildItem -LiteralPath $rawRoot -Force -File).Count -ne 0) {
        throw "Linux parity raw result root contains unexpected files"
    }
    $expectedCaseNames = [string[]]@($contract.cases | ForEach-Object { $_.name })
    $actualCaseNames = [string[]]@(Get-ChildItem -LiteralPath $rawRoot -Force -Directory | ForEach-Object { $_.Name })
    if (@(Compare-Object -ReferenceObject $expectedCaseNames -DifferenceObject $actualCaseNames -CaseSensitive).Count -ne 0) {
        throw "Linux parity raw result cases do not exactly match the selected cases"
    }
    foreach ($case in $contract.cases) {
        $caseRawRoot = Join-Path $rawRoot $case.name
        if (@(Get-ChildItem -LiteralPath $caseRawRoot -Force -File).Count -ne 0) {
            throw "Linux parity raw result case '$($case.name)' contains unexpected files"
        }
        $actualRunDirectories = [string[]]@(
            Get-ChildItem -LiteralPath $caseRawRoot -Force -Directory | ForEach-Object { $_.Name }
        )
        $expectedRunDirectories = [string[]]@($expectedRunDirectoriesByCase[$case.name])
        if (@(Compare-Object -ReferenceObject $expectedRunDirectories -DifferenceObject $actualRunDirectories -CaseSensitive).Count -ne 0) {
            throw "Linux parity raw run directories do not exactly match the contract for $($case.name)"
        }
    }

    function Get-ArtifactRelativePath([string]$Path) {
        $rootFull = [System.IO.Path]::GetFullPath($OutRoot).TrimEnd("\", "/")
        $pathFull = [System.IO.Path]::GetFullPath($Path)
        if (!(Test-MatchedPathUnder $pathFull $rootFull)) {
            throw "artifact path escaped the benchmark output root: $pathFull"
        }
        return $pathFull.Substring($rootFull.Length).TrimStart("\", "/").Replace("\", "/")
    }

    function Read-LinuxFerrumRun($Case, [string]$Kind, [int]$Ordinal, [string]$RunRoot) {
        $timingPath = Join-Path $RunRoot "process-time.env"
        $reportPath = Join-Path $RunRoot "solve-report.json"
        $logPath = Join-Path $RunRoot "ferrum.log"
        $timing = Read-MatchedGnuTime $timingPath
        if ($timing.exitCode -ne 0) { throw "Ferrum GNU-time exit code was not zero" }
        $report = Get-Content -LiteralPath $reportPath -Raw | ConvertFrom-Json
        if ([string]::IsNullOrWhiteSpace([string]$report.outerConvergence.status) -or
            @("Invalid", "NotEvaluated", "Failed") -contains [string]$report.outerConvergence.status -or
            @("MomentumSolverInvalidState", "PressureSolverInvalidState", "SolverInvalidState") -contains [string]$report.solve.stopReason -or
            @($report.history | Where-Object { $_.pressureCorrectionAccepted -ne $true }).Count -ne 0) {
            throw "Ferrum report contains an invalid outer solve result"
        }
        $expectedSolver = if ($PressureSolver -eq "gamg") { "GAMG" } else { "pcg" }
        if (!([string]$report.options.pressureLinearSolver).Equals($expectedSolver, [System.StringComparison]::OrdinalIgnoreCase)) {
            throw "Ferrum used pressure solver '$($report.options.pressureLinearSolver)', expected '$expectedSolver'"
        }
        if ($null -ne $report.timing.pressureGamgProfile) { throw "matched timing run must not enable GAMG profiling" }
        $history = @(Convert-MatchedFerrumHistory $report)
        if ($history.Count -ne $Case.fixedIterations) { throw "Ferrum completed $($history.Count) SIMPLE steps, expected $($Case.fixedIterations)" }
        $workingCase = Join-Path $RunRoot "case"
        if ((Get-MatchedNumericOutputDirectoryCount $workingCase) -ne 0) { throw "Ferrum timing run wrote an unexpected time directory" }
        $orderPosition = $orderByRun["$($Case.name)|$Kind|$Ordinal|ferrum"]
        if ($null -eq $orderPosition) { throw "Ferrum run order was not recorded for $($Case.name) $Kind $Ordinal" }
        return [pscustomobject][ordered]@{
            engine = "FerrumCFD"; kind = $Kind; ordinal = $Ordinal
            orderPosition = $orderPosition
            commonProcessElapsedSeconds = $timing.elapsedSeconds
            processUserSeconds = $timing.userSeconds
            processSystemSeconds = $timing.systemSeconds
            maxResidentSetKiB = $timing.maxResidentSetKiB
            nativeInternalSeconds = [double]$report.timing.solverTotalSeconds
            simpleIterations = $history.Count
            pressureLinearIterations = [int]$report.solve.pressureLinearIterations
            momentumLinearIterations = [int]$report.solve.momentumLinearIterations
            converged = [bool]$report.solve.converged
            stopReason = [string]$report.solve.stopReason
            outputTimeDirectories = 0
            report = Get-ArtifactRelativePath $reportPath
            log = Get-ArtifactRelativePath $logPath
            processTiming = Get-ArtifactRelativePath $timingPath
            history = $history
        }
    }

    function Read-LinuxOpenFoamRun($Case, [string]$Kind, [int]$Ordinal, [string]$RunRoot) {
        $timingPath = Join-Path $RunRoot "process-time.env"
        $logPath = Join-Path $RunRoot "openfoam.log"
        $timing = Read-MatchedGnuTime $timingPath
        if ($timing.exitCode -ne 0) { throw "OpenFOAM GNU-time exit code was not zero" }
        $parsed = Read-MatchedOpenFoamLog $logPath
        $history = @($parsed.history)
        if (!$parsed.sawEnd) { throw "OpenFOAM log did not reach End: $logPath" }
        if ($history.Count -ne $Case.fixedIterations) { throw "OpenFOAM completed $($history.Count) SIMPLE steps, expected $($Case.fixedIterations)" }
        if ($null -eq $parsed.executionTimeSeconds) { throw "OpenFOAM log did not contain ExecutionTime: $logPath" }
        $expectedPressureSolver = if ($PressureSolver -eq "gamg") { "GAMG" } else { "PCG" }
        $actualPressureSolvers = [string[]]@(
            $history | ForEach-Object { $_.pressureLinearSolvers } | Sort-Object -Unique
        )
        if ($actualPressureSolvers.Count -ne 1 -or
            !$actualPressureSolvers[0].Equals($expectedPressureSolver, [System.StringComparison]::OrdinalIgnoreCase)) {
            throw "OpenFOAM pressure solver was '$($actualPressureSolvers -join ',')', expected '$expectedPressureSolver'"
        }
        $workingCase = Join-Path $RunRoot "case"
        if ((Get-MatchedNumericOutputDirectoryCount $workingCase) -ne 0) { throw "OpenFOAM timing run wrote an unexpected time directory" }
        $orderPosition = $orderByRun["$($Case.name)|$Kind|$Ordinal|openfoam"]
        if ($null -eq $orderPosition) { throw "OpenFOAM run order was not recorded for $($Case.name) $Kind $Ordinal" }
        return [pscustomobject][ordered]@{
            engine = "OpenFOAM"; kind = $Kind; ordinal = $Ordinal
            orderPosition = $orderPosition
            commonProcessElapsedSeconds = $timing.elapsedSeconds
            processUserSeconds = $timing.userSeconds
            processSystemSeconds = $timing.systemSeconds
            maxResidentSetKiB = $timing.maxResidentSetKiB
            nativeInternalSeconds = [double]$parsed.executionTimeSeconds
            openFoamClockSeconds = [double]$parsed.clockTimeSeconds
            pressureLinearSolver = $actualPressureSolvers[0]
            simpleIterations = $history.Count
            pressureLinearIterations = [int](($history.pressureLinearIterations | Measure-Object -Sum).Sum)
            momentumLinearIterations = [int](($history.momentumLinearIterations | Measure-Object -Sum).Sum)
            converged = $null -ne (Select-String -LiteralPath $logPath -Pattern "SIMPLE solution converged" | Select-Object -First 1)
            stopReason = "FixedIterationBudgetReached"
            outputTimeDirectories = 0
            log = Get-ArtifactRelativePath $logPath
            processTiming = Get-ArtifactRelativePath $timingPath
            history = $history
        }
    }

    $caseResults = @()
    foreach ($case in $contract.cases) {
        $ferrumRuns = @()
        $openFoamRuns = @()
        foreach ($kind in @("warmup", "measured")) {
            $count = if ($kind -eq "warmup") { $WarmupRuns } else { $MeasuredRuns }
            for ($ordinal = 1; $ordinal -le $count; $ordinal++) {
                $caseRaw = Join-Path $OutRoot "raw\$($case.name)"
                $ferrumRuns += Read-LinuxFerrumRun $case $kind $ordinal (Join-Path $caseRaw "$kind-$ordinal-ferrum")
                $openFoamRuns += Read-LinuxOpenFoamRun $case $kind $ordinal (Join-Path $caseRaw "$kind-$ordinal-openfoam")
            }
        }
        if ($ferrumRuns.Count -ne ($WarmupRuns + $MeasuredRuns) -or $openFoamRuns.Count -ne ($WarmupRuns + $MeasuredRuns)) {
            throw "exact run-count contract failed for $($case.name)"
        }
        $ferrumSummary = Get-MatchedEngineSummary "FerrumCFD" $ferrumRuns
        $openFoamSummary = Get-MatchedEngineSummary "OpenFOAM" $openFoamRuns
        $pairedRatios = [double[]]@()
        for ($ordinal = 1; $ordinal -le $MeasuredRuns; $ordinal++) {
            $ferrum = @($ferrumRuns | Where-Object { $_.kind -eq "measured" -and $_.ordinal -eq $ordinal })
            $openFoam = @($openFoamRuns | Where-Object { $_.kind -eq "measured" -and $_.ordinal -eq $ordinal })
            if ($ferrum.Count -ne 1 -or $openFoam.Count -ne 1 -or $openFoam[0].commonProcessElapsedSeconds -le 0.0) {
                throw "paired-run contract failed for $($case.name) measured $ordinal"
            }
            $pairedRatios += $ferrum[0].commonProcessElapsedSeconds / $openFoam[0].commonProcessElapsedSeconds
        }
        $ratioOfMedians = $ferrumSummary.medians.commonProcessElapsedSeconds / $openFoamSummary.medians.commonProcessElapsedSeconds
        $pressureWorkRatio = $ferrumSummary.medians.pressureLinearIterationsPerSolve / $openFoamSummary.medians.pressureLinearIterationsPerSolve
        $residualCsv = Join-Path $OutRoot "$($case.name)\residual-history-medians.csv"
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $residualCsv) | Out-Null
        Write-MatchedResidualCsv $residualCsv $ferrumSummary.historyMedians $openFoamSummary.historyMedians
        $manifestCase = @($manifestCases | Where-Object { $_.name -eq $case.name })[0]
        $caseResults += [pscustomobject][ordered]@{
            name = $case.name
            fixedSimpleIterations = $case.fixedIterations
            canonicalPolyMeshSha256 = $manifestCase.canonicalPolyMeshSha256
            sharedFileSha256 = $manifestCase.sharedFileSha256
            pressureConversion = $manifestCase.pressureConversion
            residualCsv = Get-ArtifactRelativePath $residualCsv
            ferrum = $ferrumSummary
            openFoam = $openFoamSummary
            comparison = [pscustomobject][ordered]@{
                primaryMetric = "commonProcessElapsedSeconds"
                ferrumOverOpenFoamRatioOfMedians = $ratioOfMedians
                ferrumSlowerPercentRatioOfMedians = 100.0 * ($ratioOfMedians - 1.0)
                ferrumOverOpenFoamMedianPairedRatio = Get-MatchedMedian $pairedRatios
                pairedRatioMad = Get-MatchedMedianAbsoluteDeviation $pairedRatios
                pairedRatios = $pairedRatios
                ferrumOverOpenFoamPressureIterationsPerSolveRatio = $pressureWorkRatio
            }
        }
    }

    $metadataRoot = Join-Path $OutRoot "metadata"
    $buildTiming = Read-MatchedGnuTime (Join-Path $metadataRoot "build-timing.env")
    if ($buildTiming.exitCode -ne 0) { throw "recorded Linux build exit code was not zero" }
    $recordedTemplatesArchiveSha256 = (Get-Content -LiteralPath (Join-Path $metadataRoot "templates-archive-sha256.txt") -Raw).Trim()
    if ($recordedTemplatesArchiveSha256 -ne $templatesArchiveSha256) {
        throw "recorded templates archive SHA-256 does not match the host input"
    }
    $rustcVerbose = Get-Content -LiteralPath (Join-Path $metadataRoot "rustc-vv.txt") -Raw
    $summary = [pscustomobject][ordered]@{
        schemaVersion = 2
        benchmark = "matched-linux-serial-cpu-solver"
        generatedAtUtc = [DateTime]::UtcNow.ToString("o")
        pressureSolver = $PressureSolver
        source = [pscustomobject][ordered]@{
            commit = $sourceCommit
            tree = $sourceTree
            archiveSha256 = $sourceArchiveSha256
            matchedCaseTemplatesArchiveSha256 = $templatesArchiveSha256
            cargoLockSha256 = (Get-Content -LiteralPath (Join-Path $metadataRoot "cargo-lock-sha256.txt") -Raw).Trim()
            sourceWorktreeCleanAtLaunch = $sourceWorktreeClean
        }
        platform = [pscustomobject][ordered]@{
            lane = "linux-parity"
            wslDistro = $Distro
            distroRelease = (Get-Content -LiteralPath (Join-Path $metadataRoot "distro-release.txt") -Raw).Trim()
            kernel = (Get-Content -LiteralPath (Join-Path $metadataRoot "uname.txt") -Raw).Trim()
            filesystemType = (Get-Content -LiteralPath (Join-Path $metadataRoot "filesystem-type.txt") -Raw).Trim()
            cpuModel = (Get-Content -LiteralPath (Join-Path $metadataRoot "cpu-model.txt") -Raw).Trim()
            cpuSet = (Get-Content -LiteralPath (Join-Path $metadataRoot "cpu-set.txt") -Raw).Trim()
            siblingCpuSet = (Get-Content -LiteralPath (Join-Path $metadataRoot "cpu-siblings.txt") -Raw).Trim()
            openFoamFoundationVersion = (Get-Content -LiteralPath (Join-Path $metadataRoot "openfoam-version.txt") -Raw).Trim()
            openFoamBuildOptions = (Get-Content -LiteralPath (Join-Path $metadataRoot "openfoam-build-options.txt") -Raw).Trim()
            openFoamBinarySha256 = (Get-Content -LiteralPath (Join-Path $metadataRoot "openfoam-binary-sha256.txt") -Raw).Trim()
            hostPowerPlan = $hostPowerPlan
        }
        build = [pscustomobject][ordered]@{
            variant = $BuildVariant
            command = "cargo +$RustToolchain build --locked --release -p ferrum-run --bin ferrumRun"
            rustToolchain = $RustToolchain
            rustcVerboseVersion = $rustcVerbose.Trim()
            cargoVersion = (Get-Content -LiteralPath (Join-Path $metadataRoot "cargo-version.txt") -Raw).Trim()
            rustflags = if ($BuildVariant -eq "native") { "-C target-cpu=native" } else { "" }
            binarySha256 = (Get-Content -LiteralPath (Join-Path $metadataRoot "ferrum-binary-sha256.txt") -Raw).Trim()
            wallClockSeconds = $buildTiming.elapsedSeconds
            excludedFromRunTiming = $true
            log = "metadata/cargo-build-release.log"
        }
        timingContract = [pscustomobject][ordered]@{
            primaryMetric = "commonProcessElapsedSeconds"
            commonTool = "GNU /usr/bin/time"
            nativeTimersAreDiagnosticOnly = $true
            ferrumNativeTimer = "solve report timing.solverTotalSeconds"
            openFoamNativeTimer = "OpenFOAM ExecutionTime"
        }
        policy = [pscustomobject][ordered]@{
            serialCpuOnly = $true
            warmupRuns = $WarmupRuns
            measuredRuns = $MeasuredRuns
            pairedRuns = $true
            alternatingEngineOrder = $true
            sameWslDistribution = $true
            ext4SourceBuildCasesAndLogs = $true
            compilationExcluded = $true
            identicalSimpleIterationBudget = $true
            polyMeshByteHashesVerified = $true
            residualControlDisabledForFixedWork = $true
            solutionFieldOutputDisabled = $true
            gamgProfilingDisabled = $true
            threadEnvironment = [pscustomobject][ordered]@{
                OMP_NUM_THREADS = 1; OPENBLAS_NUM_THREADS = 1; MKL_NUM_THREADS = 1
                NUMEXPR_NUM_THREADS = 1; RAYON_NUM_THREADS = 1
            }
        }
        residualDefinitions = [pscustomobject][ordered]@{
            ferrum = "normalized linear-system residuals from the Ferrum solve report; continuityIndicator is post-correction L2"
            openFoam = "Initial/Final residual printed by OpenFOAM linear solvers; continuityIndicator is sum local"
            comparability = "linear residual trends and iteration counts are compared; continuity indicators use different definitions and are not directly ranked"
        }
        resultArchiveSha256 = $actualArchiveSha256
        cases = $caseResults
    }
    $jsonPath = Join-Path $OutRoot "summary.json"
    $markdownPath = Join-Path $OutRoot "summary.md"
    $summary | ConvertTo-Json -Depth 24 | Set-Content -LiteralPath $jsonPath -Encoding UTF8

    $lines = New-Object System.Collections.Generic.List[string]
    $lines.Add("# Matched Linux Serial CPU Solver Benchmark")
    $lines.Add("")
    $lines.Add("Source: ``$sourceCommit`` (tree ``$sourceTree``)")
    $lines.Add("Pressure solver/build: ``$PressureSolver`` / ``$BuildVariant``")
    $lines.Add("Warm-up/measured paired runs: ``$WarmupRuns/$MeasuredRuns``")
    $lines.Add("WSL/CPU/filesystem: ``$Distro`` / ``$CpuSet`` / ``$($summary.platform.filesystemType)``")
    $lines.Add("")
    $lines.Add("The primary metric is identical GNU-time process elapsed time for both Linux executables. Ferrum solverTotalSeconds and OpenFOAM ExecutionTime remain diagnostics and are not divided into the headline ratio.")
    $lines.Add("")
    $lines.Add("| Case | SIMPLE | Ferrum elapsed [s] | OpenFOAM elapsed [s] | Ratio of medians | Median paired ratio | Paired MAD |")
    $lines.Add("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    foreach ($case in $caseResults) {
        $lines.Add(("| {0} | {1} | {2} | {3} | {4} | {5} | {6} |" -f
            $case.name, $case.fixedSimpleIterations,
            (Format-MatchedReportNumber $case.ferrum.medians.commonProcessElapsedSeconds),
            (Format-MatchedReportNumber $case.openFoam.medians.commonProcessElapsedSeconds),
            (Format-MatchedReportNumber $case.comparison.ferrumOverOpenFoamRatioOfMedians),
            (Format-MatchedReportNumber $case.comparison.ferrumOverOpenFoamMedianPairedRatio),
            (Format-MatchedReportNumber $case.comparison.pairedRatioMad)))
    }
    $lines.Add("")
    $lines.Add("| Case | Ferrum p iterations/solve | OpenFOAM p iterations/solve | Work ratio | Ferrum native timer [s] | OpenFOAM native timer [s] |")
    $lines.Add("| --- | ---: | ---: | ---: | ---: | ---: |")
    foreach ($case in $caseResults) {
        $lines.Add(("| {0} | {1} | {2} | {3} | {4} | {5} |" -f
            $case.name,
            (Format-MatchedReportNumber $case.ferrum.medians.pressureLinearIterationsPerSolve),
            (Format-MatchedReportNumber $case.openFoam.medians.pressureLinearIterationsPerSolve),
            (Format-MatchedReportNumber $case.comparison.ferrumOverOpenFoamPressureIterationsPerSolveRatio),
            (Format-MatchedReportNumber $case.ferrum.medians.nativeInternalSeconds),
            (Format-MatchedReportNumber $case.openFoam.medians.nativeInternalSeconds)))
    }
    Set-Content -LiteralPath $markdownPath -Value $lines -Encoding UTF8
    $completed = $true
    Write-Output "wrote matched Linux benchmark JSON: $jsonPath"
    Write-Output "wrote matched Linux benchmark Markdown: $markdownPath"
} finally {
    if ($completed -and (Test-Path -LiteralPath $stageRoot)) {
        Remove-Item -LiteralPath $stageRoot -Recurse -Force
    } elseif (!$completed -and (Test-Path -LiteralPath $stageRoot)) {
        Write-Warning "host staging preserved after failure: $stageRoot"
    }
}
