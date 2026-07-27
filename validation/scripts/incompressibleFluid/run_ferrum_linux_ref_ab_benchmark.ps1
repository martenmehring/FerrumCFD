param(
    [Parameter(Mandatory = $true)]
    [string]$BaselineRef,
    [Parameter(Mandatory = $true)]
    [string]$CandidateRef,
    [string]$ExpectedChangedPath = "src/ferrumMesh/src/flow.rs",
    [int]$WarmupRuns = 2,
    [int]$MeasuredRuns = 10,
    [ValidateSet("gamg", "pcg")]
    [string]$PressureSolver = "gamg",
    [ValidateSet("all", "pipe", "channel")]
    [string]$CaseName = "all",
    [string]$Distro = "Ubuntu-22.04",
    [string]$CpuSet = "2",
    [ValidateSet("portable", "native", "native-codegen1", "native-thin-lto", "native-fat-lto")]
    [string]$BuildVariant = "portable",
    [string]$RustToolchain = "1.94.0",
    [string]$OutRoot = "",
    [switch]$PreflightOnly,
    [switch]$KeepWslWorkspace
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "matched_cpu_solver_common.ps1")

$RepoRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
$TargetRoot = Join-Path $RepoRoot "target"
$WorkerPath = Join-Path $PSScriptRoot "run_ferrum_linux_ref_ab_worker.sh"
if (!(Test-Path -LiteralPath $WorkerPath -PathType Leaf)) { throw "Ferrum Linux ref A/B worker was not found: $WorkerPath" }
if ($WarmupRuns -lt 0) { throw "WarmupRuns must be zero or greater" }
if ($MeasuredRuns -lt 2 -or ($MeasuredRuns % 2) -ne 0) { throw "MeasuredRuns must be a positive even integer of at least two" }
if ($CpuSet -notmatch "^[0-9]+([,-][0-9]+)*$") { throw "CpuSet is invalid: $CpuSet" }
if ([string]::IsNullOrWhiteSpace($ExpectedChangedPath) -or $ExpectedChangedPath.Contains("\") -or
    $ExpectedChangedPath.StartsWith("/") -or $ExpectedChangedPath -match "(^|/)\.\.(/|$)") {
    throw "ExpectedChangedPath must be one safe repository-relative slash path"
}
if ($null -eq (Get-Command wsl -ErrorAction SilentlyContinue)) { throw "wsl.exe was not found" }

$workerWslPath = ConvertTo-MatchedWslPath $WorkerPath $Distro
$workerBootstrap = 'set -o pipefail; tr -d ''\r'' < "\$1" | bash -s -- "\${@:2}"'
$preflightArguments = @(
    "-d", $Distro, "--", "bash", "-c", $workerBootstrap, "ferrum-linux-ref-ab-worker", $workerWslPath,
    "--preflight-only", "--rust-toolchain", $RustToolchain, "--cpu-set", $CpuSet,
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
if ($preflightExitCode -ne 0) { throw "Ferrum Linux ref A/B preflight failed for '$Distro':`n$($preflightOutput -join "`n")" }
if ($PreflightOnly) { $preflightOutput | Write-Output; return }

function Resolve-ExactCommit([string]$Ref, [string]$Label) {
    $commit = (& git -C $RepoRoot rev-parse "$Ref`^{commit}" 2>$null).Trim()
    if ($LASTEXITCODE -ne 0 -or $commit -notmatch "^[0-9a-f]{40}$") { throw "could not resolve $Label ref '$Ref' to an exact commit" }
    return $commit
}

$baselineCommit = Resolve-ExactCommit $BaselineRef "baseline"
$candidateCommit = Resolve-ExactCommit $CandidateRef "candidate"
if ($baselineCommit -eq $candidateCommit) { throw "baseline and candidate commits must differ" }
$baselineTree = (& git -C $RepoRoot rev-parse "$baselineCommit`^{tree}").Trim()
$candidateTree = (& git -C $RepoRoot rev-parse "$candidateCommit`^{tree}").Trim()
if ($baselineTree -notmatch "^[0-9a-f]{40}$" -or $candidateTree -notmatch "^[0-9a-f]{40}$") { throw "could not resolve exact baseline/candidate trees" }

$candidateLine = ((& git -C $RepoRoot rev-list --parents -n 1 $candidateCommit) -join " ").Trim()
if ($LASTEXITCODE -ne 0) { throw "could not inspect candidate parents" }
$candidateParts = @($candidateLine -split "\s+" | Where-Object { $_ })
if ($candidateParts.Count -ne 2 -or $candidateParts[0] -ne $candidateCommit -or $candidateParts[1] -ne $baselineCommit) {
    throw "candidate must be a single-parent direct child of the exact baseline"
}
$changedPaths = @(& git -C $RepoRoot diff --name-only --no-renames $baselineCommit $candidateCommit --)
if ($LASTEXITCODE -ne 0 -or $changedPaths.Count -ne 1 -or [string]$changedPaths[0] -cne $ExpectedChangedPath) {
    throw "candidate diff must contain exactly '$ExpectedChangedPath'; found: $($changedPaths -join ', ')"
}

$baselineCargoLockBlob = (& git -C $RepoRoot rev-parse "$baselineCommit`:Cargo.lock" 2>$null).Trim()
$candidateCargoLockBlob = (& git -C $RepoRoot rev-parse "$candidateCommit`:Cargo.lock" 2>$null).Trim()
if ($LASTEXITCODE -ne 0 -or $baselineCargoLockBlob -notmatch "^[0-9a-f]{40}$" -or $baselineCargoLockBlob -ne $candidateCargoLockBlob) {
    throw "baseline and candidate must reference the identical Cargo.lock blob"
}

$launchStatus = @(& git -C $RepoRoot status --porcelain=v1)
$sourceWorktreeCleanAtLaunch = $launchStatus.Count -eq 0
$caseSelector = switch ($CaseName) {
    "all" { "all" }
    "pipe" { "laminarPipe" }
    "channel" { "planeChannel" }
    default { throw "unsupported case selector" }
}
if ([string]::IsNullOrWhiteSpace($OutRoot)) {
    $OutRoot = Join-Path $TargetRoot "benchmarks\ferrum_linux_ref_ab\$PressureSolver-$BuildVariant"
}
$OutRoot = [System.IO.Path]::GetFullPath($OutRoot)
$benchmarkOutputRoot = [System.IO.Path]::GetFullPath(
    (Join-Path $TargetRoot "benchmarks\ferrum_linux_ref_ab")
).TrimEnd("\", "/")
if (!(Test-MatchedPathUnder $OutRoot $benchmarkOutputRoot) -or
    $OutRoot.Equals($benchmarkOutputRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "OutRoot must be a strict child of the dedicated Ferrum Linux ref A/B root: $benchmarkOutputRoot"
}

$stageRoot = Join-Path $TargetRoot "benchmarks\.linux-ref-ab-stage-$PID"
Reset-MatchedTargetDirectory $stageRoot $TargetRoot
$completed = $false
try {
    $baselineArchive = Join-Path $stageRoot "baseline.tar"
    $candidateArchive = Join-Path $stageRoot "candidate.tar"
    & git -C $RepoRoot archive --format=tar --output=$baselineArchive $baselineCommit
    if ($LASTEXITCODE -ne 0) { throw "could not archive exact baseline commit" }
    & git -C $RepoRoot archive --format=tar --output=$candidateArchive $candidateCommit
    if ($LASTEXITCODE -ne 0) { throw "could not archive exact candidate commit" }
    Assert-MatchedSafeTarArchive $baselineArchive "exact baseline source"
    Assert-MatchedSafeTarArchive $candidateArchive "exact candidate source"
    $baselineArchiveSha256 = (Get-FileHash -LiteralPath $baselineArchive -Algorithm SHA256).Hash.ToLowerInvariant()
    $candidateArchiveSha256 = (Get-FileHash -LiteralPath $candidateArchive -Algorithm SHA256).Hash.ToLowerInvariant()

    $baselineSourceRoot = Join-Path $stageRoot "baseline-source"
    $candidateSourceRoot = Join-Path $stageRoot "candidate-source"
    foreach ($root in @($baselineSourceRoot, $candidateSourceRoot)) {
        New-Item -ItemType Directory -Force -Path $root | Out-Null
        Assert-MatchedNoReparsePath $root $stageRoot
    }
    & tar -xf $baselineArchive -C $baselineSourceRoot
    if ($LASTEXITCODE -ne 0) { throw "could not extract baseline source archive" }
    & tar -xf $candidateArchive -C $candidateSourceRoot
    if ($LASTEXITCODE -ne 0) { throw "could not extract candidate source archive" }
    $baselineCargoLockSha256 = (Get-FileHash -LiteralPath (Join-Path $baselineSourceRoot "Cargo.lock") -Algorithm SHA256).Hash.ToLowerInvariant()
    $candidateCargoLockSha256 = (Get-FileHash -LiteralPath (Join-Path $candidateSourceRoot "Cargo.lock") -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($baselineCargoLockSha256 -ne $candidateCargoLockSha256) { throw "extracted Cargo.lock bytes differ between refs" }

    $contract = Get-MatchedCpuCaseDefinitions $baselineSourceRoot $PressureSolver $caseSelector
    $templatesRoot = Join-Path $stageRoot "templates"
    New-Item -ItemType Directory -Force -Path $templatesRoot | Out-Null
    $manifestCases = @()
    foreach ($case in $contract.cases) {
        $destination = Join-Path $templatesRoot $case.name
        New-MatchedFerrumWorkingCase $case $destination $contract.fvSolution $templatesRoot | Out-Null
        $canonicalHashes = Get-MatchedPolyMeshHashes $case.ferrumCase
        Assert-MatchedHashesEqual $canonicalHashes (Get-MatchedPolyMeshHashes $destination) "$($case.name) shared template"
        $manifestCases += [pscustomobject][ordered]@{
            name = $case.name
            fixedIterations = $case.fixedIterations
            canonicalPolyMeshSha256 = $canonicalHashes
            sharedFileSha256 = [pscustomobject][ordered]@{
                velocity = (Get-FileHash -LiteralPath (Join-Path $destination "0\U") -Algorithm SHA256).Hash.ToLowerInvariant()
                pressure = (Get-FileHash -LiteralPath (Join-Path $destination "0\p") -Algorithm SHA256).Hash.ToLowerInvariant()
                fvSchemes = (Get-FileHash -LiteralPath (Join-Path $destination "system\fvSchemes") -Algorithm SHA256).Hash.ToLowerInvariant()
                fvSolution = (Get-FileHash -LiteralPath (Join-Path $destination "system\fvSolution") -Algorithm SHA256).Hash.ToLowerInvariant()
            }
        }
    }
    $templatesArchive = Join-Path $stageRoot "templates.tar"
    & tar -cf $templatesArchive -C $templatesRoot .
    if ($LASTEXITCODE -ne 0) { throw "could not create shared Ferrum template archive" }
    Assert-MatchedSafeTarArchive $templatesArchive "shared Ferrum template"
    $templatesArchiveSha256 = (Get-FileHash -LiteralPath $templatesArchive -Algorithm SHA256).Hash.ToLowerInvariant()

    $inputManifest = [pscustomobject][ordered]@{
        schemaVersion = 1
        benchmark = "ferrum-linux-ref-ab"
        baseline = [pscustomobject][ordered]@{ commit = $baselineCommit; tree = $baselineTree; archiveSha256 = $baselineArchiveSha256 }
        candidate = [pscustomobject][ordered]@{ commit = $candidateCommit; tree = $candidateTree; archiveSha256 = $candidateArchiveSha256 }
        relationship = [pscustomobject][ordered]@{ directChild = $true; exactChangedPath = $ExpectedChangedPath }
        cargoLock = [pscustomobject][ordered]@{ blob = $baselineCargoLockBlob; sha256 = $baselineCargoLockSha256 }
        pressureSolver = $PressureSolver
        cases = $manifestCases
    }
    $manifestPath = Join-Path $stageRoot "input-manifest.json"
    $inputManifest | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $manifestPath -Encoding UTF8

    $outputArchive = Join-Path $stageRoot "ferrum-linux-ref-ab-results.tar"
    $runArguments = @(
        "-d", $Distro, "--", "bash", "-c", $workerBootstrap, "ferrum-linux-ref-ab-worker", $workerWslPath,
        "--rust-toolchain", $RustToolchain, "--cpu-set", $CpuSet, "--build-variant", $BuildVariant,
        "--warmup-runs", $WarmupRuns.ToString([System.Globalization.CultureInfo]::InvariantCulture),
        "--measured-runs", $MeasuredRuns.ToString([System.Globalization.CultureInfo]::InvariantCulture),
        "--pressure-solver", $PressureSolver,
        "--baseline-archive", (ConvertTo-MatchedWslPath $baselineArchive $Distro),
        "--baseline-archive-sha256", $baselineArchiveSha256, "--baseline-commit", $baselineCommit, "--baseline-tree", $baselineTree,
        "--candidate-archive", (ConvertTo-MatchedWslPath $candidateArchive $Distro),
        "--candidate-archive-sha256", $candidateArchiveSha256, "--candidate-commit", $candidateCommit, "--candidate-tree", $candidateTree,
        "--templates-archive", (ConvertTo-MatchedWslPath $templatesArchive $Distro),
        "--templates-archive-sha256", $templatesArchiveSha256,
        "--manifest", (ConvertTo-MatchedWslPath $manifestPath $Distro),
        "--output-archive", (ConvertTo-MatchedWslPath $outputArchive $Distro)
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
    if ($workerExitCode -ne 0) { throw "Ferrum Linux ref A/B worker failed with exit code $workerExitCode" }
    if (!(Test-Path -LiteralPath $outputArchive -PathType Leaf)) { throw "worker did not return its result archive" }
    $sidecarPath = "$outputArchive.sha256"
    if (!(Test-Path -LiteralPath $sidecarPath -PathType Leaf)) { throw "worker did not return result archive SHA sidecar" }
    $actualArchiveSha256 = (Get-FileHash -LiteralPath $outputArchive -Algorithm SHA256).Hash.ToLowerInvariant()
    if ((Get-Content -LiteralPath $sidecarPath -Raw).Trim() -ne $actualArchiveSha256) { throw "result archive SHA-256 verification failed" }
    Assert-MatchedSafeTarArchive $outputArchive "Ferrum Linux ref A/B result"

    Reset-MatchedTargetDirectory $OutRoot $TargetRoot
    Assert-MatchedNoReparsePath $OutRoot $TargetRoot
    & tar -xf $outputArchive -C $OutRoot
    if ($LASTEXITCODE -ne 0) { throw "could not extract Ferrum Linux ref A/B result archive" }
    Copy-Item -LiteralPath $manifestPath -Destination (Join-Path $OutRoot "input-manifest.json") -Force

    function Get-ArtifactRelativePath([string]$Path) {
        $rootFull = [System.IO.Path]::GetFullPath($OutRoot).TrimEnd("\", "/")
        $pathFull = [System.IO.Path]::GetFullPath($Path)
        if (!(Test-MatchedPathUnder $pathFull $rootFull)) { throw "artifact path escaped output root: $pathFull" }
        return $pathFull.Substring($rootFull.Length).TrimStart("\", "/").Replace("\", "/")
    }
    function Assert-ExactNames([string]$Root, [string[]]$ExpectedFiles, [string[]]$ExpectedDirectories, [string]$Description) {
        $actualFiles = [string[]]@(Get-ChildItem -LiteralPath $Root -Force -File | ForEach-Object { $_.Name })
        $actualDirectories = [string[]]@(Get-ChildItem -LiteralPath $Root -Force -Directory | ForEach-Object { $_.Name })
        if (@(Compare-Object $ExpectedFiles $actualFiles -CaseSensitive).Count -ne 0 -or
            @(Compare-Object $ExpectedDirectories $actualDirectories -CaseSensitive).Count -ne 0) {
            throw "$Description inventory differs from the exact contract"
        }
    }
    function Assert-CanonicalReport([string]$CanonicalPath, [string]$HashPath) {
        $expected = (Get-Content -LiteralPath $HashPath -Raw).Trim()
        $actual = (Get-FileHash -LiteralPath $CanonicalPath -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($expected -notmatch "^[0-9a-f]{64}$" -or $expected -ne $actual) { throw "canonical report SHA-256 verification failed" }
        $canonical = Get-Content -LiteralPath $CanonicalPath -Raw | ConvertFrom-Json
        function Assert-NoRemovedField($Node, [string]$Path) {
            if ($null -eq $Node) { return }
            if ($Node -is [System.Array]) {
                for ($index = 0; $index -lt $Node.Count; $index++) { Assert-NoRemovedField $Node[$index] "$Path[$index]" }
                return
            }
            if ($Node -is [System.Management.Automation.PSCustomObject]) {
                foreach ($property in $Node.PSObject.Properties) {
                    if ($property.Name -eq "caseDir" -or $property.Name.EndsWith("Seconds", [System.StringComparison]::Ordinal)) {
                        throw "canonical report retained removed field '$Path.$($property.Name)'"
                    }
                    Assert-NoRemovedField $property.Value "$Path.$($property.Name)"
                }
            }
        }
        Assert-NoRemovedField $canonical '$'
        if ($null -eq $canonical.timing.pressureMatrixVectorProducts -or $null -eq $canonical.timing.pressurePreconditionerApplications) {
            throw "canonical report did not retain deterministic timing counters"
        }
        return $expected
    }

    $metadataRoot = Join-Path $OutRoot "metadata"
    Assert-ExactNames $OutRoot @("input-manifest.json") @("metadata", "raw") "benchmark output root before summary generation"
    Assert-ExactNames $metadataRoot @(
        "baseline-archive-sha256.txt", "baseline-binary-sha256.txt", "baseline-commit.txt", "baseline-tree.txt",
        "build-baseline-time.env", "build-candidate-time.env",
        "build-cargo-profile-release-codegen-units.txt", "build-cargo-profile-release-lto.txt", "build-rustflags.txt", "build-variant.txt",
        "candidate-archive-sha256.txt", "candidate-binary-sha256.txt", "candidate-commit.txt", "candidate-tree.txt",
        "cargo-build-baseline-release.log", "cargo-build-candidate-release.log", "cargo-lock-sha256.txt", "cargo-version.txt",
        "cpu-model.txt", "cpu-set.txt", "cpu-siblings.txt", "distro-release.txt", "filesystem-type.txt", "input-manifest.json",
        "run-order.tsv", "rustc-vv.txt", "templates-archive-sha256.txt", "uname.txt", "workspace-path.txt"
    ) @() "benchmark metadata"

    $expectedOrder = @{}
    $expectedOrderRows = New-Object System.Collections.Generic.List[object]
    $expectedRunDirectoriesByCase = @{}
    foreach ($case in $contract.cases) {
        $names = New-Object System.Collections.Generic.List[string]
        foreach ($kind in @("warmup", "measured")) {
            $count = if ($kind -eq "warmup") { $WarmupRuns } else { $MeasuredRuns }
            for ($ordinal = 1; $ordinal -le $count; $ordinal++) {
                $refs = if (($ordinal % 2) -eq 1) { @("baseline", "candidate") } else { @("candidate", "baseline") }
                for ($position = 1; $position -le 2; $position++) {
                    $refName = $refs[$position - 1]
                    $expectedOrder["$($case.name)|$kind|$ordinal|$refName"] = $position
                    $expectedOrderRows.Add([pscustomobject][ordered]@{
                        case = $case.name; kind = $kind; ordinal = $ordinal; position = $position; ref = $refName
                    })
                    $names.Add("$kind-$ordinal-$refName")
                }
            }
        }
        $names.Add("oracle-baseline"); $names.Add("oracle-candidate")
        $expectedRunDirectoriesByCase[$case.name] = @($names)
    }
    $orderRows = @(Import-Csv -LiteralPath (Join-Path $OutRoot "metadata\run-order.tsv") -Delimiter "`t")
    if ($orderRows.Count -ne $expectedOrderRows.Count) { throw "run-order row count differs from exact contract" }
    $actualOrder = @{}
    for ($rowIndex = 0; $rowIndex -lt $orderRows.Count; $rowIndex++) {
        $row = $orderRows[$rowIndex]
        $expectedRow = $expectedOrderRows[$rowIndex]
        $key = "$($row.case)|$($row.kind)|$($row.ordinal)|$($row.ref)"
        if (!$expectedOrder.ContainsKey($key) -or $actualOrder.ContainsKey($key)) { throw "unexpected or duplicate run-order key: $key" }
        $position = 0; $ordinal = 0
        if (![int]::TryParse([string]$row.position, [ref]$position) -or
            ![int]::TryParse([string]$row.ordinal, [ref]$ordinal) -or
            [string]$row.case -cne [string]$expectedRow.case -or [string]$row.kind -cne [string]$expectedRow.kind -or
            $ordinal -ne [int]$expectedRow.ordinal -or $position -ne [int]$expectedRow.position -or
            [string]$row.ref -cne [string]$expectedRow.ref) {
            throw "run-order row $($rowIndex + 1) differs from exact execution sequence"
        }
        $actualOrder[$key] = $position
    }
    foreach ($key in $expectedOrder.Keys) { if (!$actualOrder.ContainsKey($key)) { throw "run-order is missing $key" } }

    $rawRoot = Join-Path $OutRoot "raw"
    Assert-ExactNames $rawRoot @() ([string[]]@($contract.cases | ForEach-Object { $_.name })) "raw root"
    foreach ($case in $contract.cases) {
        Assert-ExactNames (Join-Path $rawRoot $case.name) @() ([string[]]$expectedRunDirectoriesByCase[$case.name]) "$($case.name) raw root"
    }

    function Read-RefRun($Case, [string]$Kind, [int]$Ordinal, [string]$RefName, [string]$RunRoot) {
        Assert-ExactNames $RunRoot @("canonical-report.json", "canonical-report.sha256", "ferrum.log", "process-time.env", "solve-report.json") @("case") "$($Case.name) $Kind $Ordinal $RefName"
        $timing = Read-MatchedGnuTime (Join-Path $RunRoot "process-time.env")
        if ($timing.exitCode -ne 0) { throw "$RefName GNU-time exit code was not zero" }
        $report = Get-Content -LiteralPath (Join-Path $RunRoot "solve-report.json") -Raw | ConvertFrom-Json
        if ([string]::IsNullOrWhiteSpace([string]$report.outerConvergence.status) -or
            @("Invalid", "NotEvaluated", "Failed") -contains [string]$report.outerConvergence.status -or
            @("MomentumSolverInvalidState", "PressureSolverInvalidState", "SolverInvalidState") -contains [string]$report.solve.stopReason -or
            @($report.history | Where-Object { $_.pressureCorrectionAccepted -ne $true }).Count -ne 0) {
            throw "$RefName report contains an invalid solve result"
        }
        $expectedSolver = if ($PressureSolver -eq "gamg") { "GAMG" } else { "pcg" }
        if (!([string]$report.options.pressureLinearSolver).Equals($expectedSolver, [System.StringComparison]::OrdinalIgnoreCase)) {
            throw "$RefName used pressure solver '$($report.options.pressureLinearSolver)', expected '$expectedSolver'"
        }
        if ($null -ne $report.timing.pressureGamgProfile) { throw "timing run unexpectedly enabled GAMG profiling" }
        $history = @(Convert-MatchedFerrumHistory $report)
        if ($history.Count -ne $Case.fixedIterations -or [int]$report.solve.simpleIterations -ne $Case.fixedIterations) {
            throw "$RefName did not complete the exact SIMPLE budget"
        }
        if ((Get-MatchedNumericOutputDirectoryCount (Join-Path $RunRoot "case")) -ne 0) { throw "$RefName timing run wrote a numeric output directory" }
        $canonicalPath = Join-Path $RunRoot "canonical-report.json"
        $canonicalHash = Assert-CanonicalReport $canonicalPath (Join-Path $RunRoot "canonical-report.sha256")
        return [pscustomobject][ordered]@{
            ref = $RefName; kind = $Kind; ordinal = $Ordinal
            orderPosition = $actualOrder["$($Case.name)|$Kind|$Ordinal|$RefName"]
            commonProcessElapsedSeconds = $timing.elapsedSeconds
            processUserSeconds = $timing.userSeconds
            processSystemSeconds = $timing.systemSeconds
            maxResidentSetKiB = $timing.maxResidentSetKiB
            nativeInternalSeconds = [double]$report.timing.solverTotalSeconds
            simpleIterations = $history.Count
            pressureLinearIterations = [int]$report.solve.pressureLinearIterations
            momentumLinearIterations = [int]$report.solve.momentumLinearIterations
            canonicalReportSha256 = $canonicalHash
            report = Get-ArtifactRelativePath (Join-Path $RunRoot "solve-report.json")
            log = Get-ArtifactRelativePath (Join-Path $RunRoot "ferrum.log")
            processTiming = Get-ArtifactRelativePath (Join-Path $RunRoot "process-time.env")
        }
    }

    function Read-Oracle($Case, [string]$RefName, [string]$RunRoot) {
        Assert-ExactNames $RunRoot @("canonical-report.json", "canonical-report.sha256", "ferrum.log", "field-oracle.json", "solve-report.json") @("case", "final-fields") "$($Case.name) $RefName oracle"
        Assert-ExactNames (Join-Path $RunRoot "final-fields") @("U", "p") @() "$($Case.name) $RefName final fields"
        $report = Get-Content -LiteralPath (Join-Path $RunRoot "solve-report.json") -Raw | ConvertFrom-Json
        if ([int]$report.solve.simpleIterations -ne $Case.fixedIterations -or @($report.history | Where-Object { $_.pressureCorrectionAccepted -ne $true }).Count -ne 0) {
            throw "$RefName oracle report failed the solve contract"
        }
        if ((Get-MatchedNumericOutputDirectoryCount (Join-Path $RunRoot "case")) -ne 0) { throw "$RefName oracle wrote a numeric output directory" }
        $canonicalHash = Assert-CanonicalReport (Join-Path $RunRoot "canonical-report.json") (Join-Path $RunRoot "canonical-report.sha256")
        $oraclePath = Join-Path $RunRoot "field-oracle.json"
        $oracle = Get-Content -LiteralPath $oraclePath -Raw | ConvertFrom-Json
        if ([int]$oracle.schemaVersion -ne 2 -or [int]$oracle.cellCount -ne [int]$report.mesh.cells -or [int]$oracle.U.declaredValues -ne [int]$report.mesh.cells -or
            [int]$oracle.p.declaredValues -ne [int]$report.mesh.cells -or [int]$oracle.U.scalarSlots -ne (3 * [int]$report.mesh.cells) -or
            [int]$oracle.p.scalarSlots -ne [int]$report.mesh.cells -or
            [string]$oracle.combinedIeee754Sha256 -notmatch "^[0-9a-f]{64}$") {
            throw "$RefName field oracle shape/hash contract failed"
        }
        foreach ($fieldName in @("U", "p")) {
            $fieldPath = Join-Path $RunRoot "final-fields\$fieldName"
            if ((Get-FileHash -LiteralPath $fieldPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne [string]$oracle.$fieldName.textSha256) {
                throw "$RefName $fieldName text SHA-256 differs from the oracle"
            }
            foreach ($hashName in @("ieee754BigEndianSha256", "boundaryIeee754BigEndianSha256", "fullFieldIeee754BigEndianSha256")) {
                if ([string]$oracle.$fieldName.$hashName -notmatch "^[0-9a-f]{64}$") { throw "$RefName $fieldName $hashName is invalid" }
            }
        }
        return [pscustomobject][ordered]@{
            ref = $RefName
            canonicalReportSha256 = $canonicalHash
            cellCount = [int]$oracle.cellCount
            velocityTextSha256 = [string]$oracle.U.textSha256
            pressureTextSha256 = [string]$oracle.p.textSha256
            velocityIeee754Sha256 = [string]$oracle.U.ieee754BigEndianSha256
            pressureIeee754Sha256 = [string]$oracle.p.ieee754BigEndianSha256
            velocityBoundaryIeee754Sha256 = [string]$oracle.U.boundaryIeee754BigEndianSha256
            pressureBoundaryIeee754Sha256 = [string]$oracle.p.boundaryIeee754BigEndianSha256
            velocityFullFieldIeee754Sha256 = [string]$oracle.U.fullFieldIeee754BigEndianSha256
            pressureFullFieldIeee754Sha256 = [string]$oracle.p.fullFieldIeee754BigEndianSha256
            combinedIeee754Sha256 = [string]$oracle.combinedIeee754Sha256
            manifest = Get-ArtifactRelativePath $oraclePath
        }
    }

    function Get-RefSummary([string]$Name, $Runs) {
        $measured = @($Runs | Where-Object { $_.kind -eq "measured" })
        return [pscustomobject][ordered]@{
            name = $Name
            runs = $Runs
            medians = [pscustomobject][ordered]@{
                commonProcessElapsedSeconds = Get-MatchedMedian ([double[]]@($measured.commonProcessElapsedSeconds))
                processUserSeconds = Get-MatchedMedian ([double[]]@($measured.processUserSeconds))
                processSystemSeconds = Get-MatchedMedian ([double[]]@($measured.processSystemSeconds))
                maxResidentSetKiB = Get-MatchedMedian ([double[]]@($measured.maxResidentSetKiB))
                nativeInternalSeconds = Get-MatchedMedian ([double[]]@($measured.nativeInternalSeconds))
            }
            mad = [pscustomobject][ordered]@{
                commonProcessElapsedSeconds = Get-MatchedMedianAbsoluteDeviation ([double[]]@($measured.commonProcessElapsedSeconds))
            }
        }
    }
    function Get-Direction([double]$Ratio) {
        if ($Ratio -lt 1.0) { return "candidate-faster" }
        if ($Ratio -gt 1.0) { return "candidate-slower" }
        return "tie"
    }

    $caseResults = @()
    foreach ($case in $contract.cases) {
        $baselineRuns = @(); $candidateRuns = @()
        foreach ($kind in @("warmup", "measured")) {
            $count = if ($kind -eq "warmup") { $WarmupRuns } else { $MeasuredRuns }
            for ($ordinal = 1; $ordinal -le $count; $ordinal++) {
                $caseRoot = Join-Path $rawRoot $case.name
                $baselineRuns += Read-RefRun $case $kind $ordinal "baseline" (Join-Path $caseRoot "$kind-$ordinal-baseline")
                $candidateRuns += Read-RefRun $case $kind $ordinal "candidate" (Join-Path $caseRoot "$kind-$ordinal-candidate")
            }
        }
        if ($baselineRuns.Count -ne ($WarmupRuns + $MeasuredRuns) -or $candidateRuns.Count -ne ($WarmupRuns + $MeasuredRuns)) { throw "exact run-count contract failed" }
        $allCanonicalHashes = [string[]]@($baselineRuns.canonicalReportSha256 + $candidateRuns.canonicalReportSha256 | Sort-Object -Unique)
        if ($allCanonicalHashes.Count -ne 1) { throw "$($case.name) canonical reports differ across refs or runs" }
        $baselineOracle = Read-Oracle $case "baseline" (Join-Path $rawRoot "$($case.name)\oracle-baseline")
        $candidateOracle = Read-Oracle $case "candidate" (Join-Path $rawRoot "$($case.name)\oracle-candidate")
        if ($baselineOracle.canonicalReportSha256 -ne $allCanonicalHashes[0] -or $candidateOracle.canonicalReportSha256 -ne $allCanonicalHashes[0]) {
            throw "$($case.name) oracle report is not bit-identical to timed canonical reports"
        }
        if ($baselineOracle.combinedIeee754Sha256 -ne $candidateOracle.combinedIeee754Sha256 -or
            $baselineOracle.velocityTextSha256 -ne $candidateOracle.velocityTextSha256 -or
            $baselineOracle.pressureTextSha256 -ne $candidateOracle.pressureTextSha256 -or
            $baselineOracle.velocityIeee754Sha256 -ne $candidateOracle.velocityIeee754Sha256 -or
            $baselineOracle.pressureIeee754Sha256 -ne $candidateOracle.pressureIeee754Sha256 -or
            $baselineOracle.velocityBoundaryIeee754Sha256 -ne $candidateOracle.velocityBoundaryIeee754Sha256 -or
            $baselineOracle.pressureBoundaryIeee754Sha256 -ne $candidateOracle.pressureBoundaryIeee754Sha256 -or
            $baselineOracle.velocityFullFieldIeee754Sha256 -ne $candidateOracle.velocityFullFieldIeee754Sha256 -or
            $baselineOracle.pressureFullFieldIeee754Sha256 -ne $candidateOracle.pressureFullFieldIeee754Sha256) {
            throw "$($case.name) final field IEEE-754 oracle differs between baseline and candidate"
        }

        $pairedRatios = [double[]]@(); $candidateFirst = [double[]]@(); $candidateSecond = [double[]]@()
        $wins = 0; $losses = 0; $ties = 0
        for ($ordinal = 1; $ordinal -le $MeasuredRuns; $ordinal++) {
            $baseline = @($baselineRuns | Where-Object { $_.kind -eq "measured" -and $_.ordinal -eq $ordinal })
            $candidate = @($candidateRuns | Where-Object { $_.kind -eq "measured" -and $_.ordinal -eq $ordinal })
            if ($baseline.Count -ne 1 -or $candidate.Count -ne 1 -or $baseline[0].commonProcessElapsedSeconds -le 0.0) { throw "paired-run contract failed" }
            $ratio = $candidate[0].commonProcessElapsedSeconds / $baseline[0].commonProcessElapsedSeconds
            $pairedRatios += $ratio
            if ($candidate[0].orderPosition -eq 1) { $candidateFirst += $ratio } elseif ($candidate[0].orderPosition -eq 2) { $candidateSecond += $ratio } else { throw "candidate order position is invalid" }
            if ($ratio -lt 1.0) { $wins++ } elseif ($ratio -gt 1.0) { $losses++ } else { $ties++ }
        }
        if ($candidateFirst.Count -ne ($MeasuredRuns / 2) -or $candidateSecond.Count -ne ($MeasuredRuns / 2)) { throw "order cohorts are not balanced" }
        $firstMedian = Get-MatchedMedian $candidateFirst
        $secondMedian = Get-MatchedMedian $candidateSecond
        $firstDirection = Get-Direction $firstMedian
        $secondDirection = Get-Direction $secondMedian
        $classificationAccepted = $firstDirection -eq $secondDirection
        $classification = if ($classificationAccepted) { $firstDirection } else { "inconclusive-order-effect" }
        $baselineSummary = Get-RefSummary "baseline" $baselineRuns
        $candidateSummary = Get-RefSummary "candidate" $candidateRuns
        $caseResults += [pscustomobject][ordered]@{
            name = $case.name
            fixedSimpleIterations = $case.fixedIterations
            deterministicCanonicalReportSha256 = $allCanonicalHashes[0]
            baseline = $baselineSummary
            candidate = $candidateSummary
            fieldOracle = [pscustomobject][ordered]@{ baseline = $baselineOracle; candidate = $candidateOracle; exactBitParity = $true }
            comparison = [pscustomobject][ordered]@{
                primaryMetric = "commonProcessElapsedSeconds"
                candidateOverBaselineRatioOfMedians = $candidateSummary.medians.commonProcessElapsedSeconds / $baselineSummary.medians.commonProcessElapsedSeconds
                medianPairedRatio = Get-MatchedMedian $pairedRatios
                pairedRatioMad = Get-MatchedMedianAbsoluteDeviation $pairedRatios
                pairedRatios = $pairedRatios
                wins = $wins; losses = $losses; ties = $ties
                orderCohorts = [pscustomobject][ordered]@{
                    candidateFirst = [pscustomobject][ordered]@{ count = $candidateFirst.Count; medianRatio = $firstMedian; mad = Get-MatchedMedianAbsoluteDeviation $candidateFirst; direction = $firstDirection }
                    candidateSecond = [pscustomobject][ordered]@{ count = $candidateSecond.Count; medianRatio = $secondMedian; mad = Get-MatchedMedianAbsoluteDeviation $candidateSecond; direction = $secondDirection }
                }
                classificationAccepted = $classificationAccepted
                classification = $classification
            }
        }
    }

    foreach ($binding in @(
        @("baseline-commit.txt", $baselineCommit), @("baseline-tree.txt", $baselineTree), @("baseline-archive-sha256.txt", $baselineArchiveSha256),
        @("candidate-commit.txt", $candidateCommit), @("candidate-tree.txt", $candidateTree), @("candidate-archive-sha256.txt", $candidateArchiveSha256),
        @("templates-archive-sha256.txt", $templatesArchiveSha256), @("cargo-lock-sha256.txt", $baselineCargoLockSha256), @("build-variant.txt", $BuildVariant)
    )) {
        if ((Get-Content -LiteralPath (Join-Path $metadataRoot $binding[0]) -Raw).Trim() -ne $binding[1]) { throw "worker metadata binding differs: $($binding[0])" }
    }
    foreach ($slot in @("baseline", "candidate")) {
        $timing = Read-MatchedGnuTime (Join-Path $metadataRoot "build-$slot-time.env")
        if ($timing.exitCode -ne 0) { throw "$slot build exit code was not zero" }
    }
    $recordedRustFlags = (Get-Content -LiteralPath (Join-Path $metadataRoot "build-rustflags.txt") -Raw).Trim()
    $recordedCodegenUnits = (Get-Content -LiteralPath (Join-Path $metadataRoot "build-cargo-profile-release-codegen-units.txt") -Raw).Trim()
    $recordedLto = (Get-Content -LiteralPath (Join-Path $metadataRoot "build-cargo-profile-release-lto.txt") -Raw).Trim()
    $expectedBuildSettings = switch ($BuildVariant) {
        "portable" { [pscustomobject]@{ rustflags = ""; codegenUnits = ""; lto = "" } }
        "native" { [pscustomobject]@{ rustflags = "-C target-cpu=native"; codegenUnits = ""; lto = "" } }
        "native-codegen1" { [pscustomobject]@{ rustflags = "-C target-cpu=native"; codegenUnits = "1"; lto = "" } }
        "native-thin-lto" { [pscustomobject]@{ rustflags = "-C target-cpu=native"; codegenUnits = ""; lto = "thin" } }
        "native-fat-lto" { [pscustomobject]@{ rustflags = "-C target-cpu=native"; codegenUnits = ""; lto = "fat" } }
    }
    if ($recordedRustFlags -ne $expectedBuildSettings.rustflags -or $recordedCodegenUnits -ne $expectedBuildSettings.codegenUnits -or $recordedLto -ne $expectedBuildSettings.lto) {
        throw "recorded build settings differ from requested variant"
    }

    $summary = [pscustomobject][ordered]@{
        schemaVersion = 1
        benchmark = "ferrum-linux-ref-ab"
        generatedAtUtc = [DateTime]::UtcNow.ToString("o")
        baseline = [pscustomobject][ordered]@{ ref = $BaselineRef; commit = $baselineCommit; tree = $baselineTree; archiveSha256 = $baselineArchiveSha256 }
        candidate = [pscustomobject][ordered]@{ ref = $CandidateRef; commit = $candidateCommit; tree = $candidateTree; archiveSha256 = $candidateArchiveSha256 }
        relationship = [pscustomobject][ordered]@{ candidateDirectChildOfBaseline = $true; exactChangedPath = $ExpectedChangedPath; cargoLockBlob = $baselineCargoLockBlob; cargoLockSha256 = $baselineCargoLockSha256 }
        sourceWorktreeCleanAtLaunch = $sourceWorktreeCleanAtLaunch
        launchStatusPorcelain = $launchStatus
        pressureSolver = $PressureSolver
        build = [pscustomobject][ordered]@{
            variant = $BuildVariant; rustToolchain = $RustToolchain
            rustcVerboseVersion = (Get-Content -LiteralPath (Join-Path $metadataRoot "rustc-vv.txt") -Raw).Trim()
            cargoVersion = (Get-Content -LiteralPath (Join-Path $metadataRoot "cargo-version.txt") -Raw).Trim()
            rustflags = $recordedRustFlags; codegenUnits = $recordedCodegenUnits; lto = $recordedLto; cargoIncremental = 0
            baselineBinarySha256 = (Get-Content -LiteralPath (Join-Path $metadataRoot "baseline-binary-sha256.txt") -Raw).Trim()
            candidateBinarySha256 = (Get-Content -LiteralPath (Join-Path $metadataRoot "candidate-binary-sha256.txt") -Raw).Trim()
        }
        platform = [pscustomobject][ordered]@{
            lane = "linux-ref-ab"; wslDistro = $Distro
            distroRelease = (Get-Content -LiteralPath (Join-Path $metadataRoot "distro-release.txt") -Raw).Trim()
            kernel = (Get-Content -LiteralPath (Join-Path $metadataRoot "uname.txt") -Raw).Trim()
            filesystemType = (Get-Content -LiteralPath (Join-Path $metadataRoot "filesystem-type.txt") -Raw).Trim()
            cpuModel = (Get-Content -LiteralPath (Join-Path $metadataRoot "cpu-model.txt") -Raw).Trim()
            cpuSet = (Get-Content -LiteralPath (Join-Path $metadataRoot "cpu-set.txt") -Raw).Trim()
            siblingCpuSet = (Get-Content -LiteralPath (Join-Path $metadataRoot "cpu-siblings.txt") -Raw).Trim()
        }
        policy = [pscustomobject][ordered]@{
            warmupRuns = $WarmupRuns; measuredRuns = $MeasuredRuns; measuredRunsEven = $true
            sameLinuxWorker = $true; alternatingRefOrder = $true; balancedOrderCohorts = $true
            sameCpuSet = $true; serialEnvironment = $true; ext4SourcesBuildsCasesAndLogs = $true
            separateSymmetricBuildPaths = $true; cargoIncrementalDisabled = $true
            noProfiling = $true; timingRunsDoNotWriteFields = $true
            untimedFinalFieldOracle = $true; canonicalReportsExactAcrossRefsAndRuns = $true
            classificationRequiresOrderCohortAgreement = $true
        }
        resultArchiveSha256 = $actualArchiveSha256
        cases = $caseResults
    }
    $jsonPath = Join-Path $OutRoot "summary.json"
    $markdownPath = Join-Path $OutRoot "summary.md"
    $summary | ConvertTo-Json -Depth 24 | Set-Content -LiteralPath $jsonPath -Encoding UTF8
    $lines = New-Object System.Collections.Generic.List[string]
    $lines.Add("# Ferrum Linux Reference A/B Benchmark")
    $lines.Add("")
    $lines.Add("Baseline: ``$baselineCommit``; candidate: ``$candidateCommit``")
    $lines.Add("Exact changed path: ``$ExpectedChangedPath``")
    $lines.Add("Pressure solver/build: ``$PressureSolver`` / ``$BuildVariant``")
    $lines.Add("Warm-up/measured paired runs: ``$WarmupRuns/$MeasuredRuns``")
    $lines.Add("")
    $lines.Add("Classification is descriptive only and is accepted only when candidate-first and candidate-second cohorts point in the same direction.")
    $lines.Add("")
    $lines.Add("| Case | Baseline elapsed [s] | Candidate elapsed [s] | Ratio medians | Paired median | MAD | W/L/T | Cohort directions | Classification |")
    $lines.Add("| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |")
    foreach ($case in $caseResults) {
        $lines.Add(("| {0} | {1} | {2} | {3} | {4} | {5} | {6}/{7}/{8} | {9} / {10} | {11} |" -f
            $case.name,
            (Format-MatchedReportNumber $case.baseline.medians.commonProcessElapsedSeconds),
            (Format-MatchedReportNumber $case.candidate.medians.commonProcessElapsedSeconds),
            (Format-MatchedReportNumber $case.comparison.candidateOverBaselineRatioOfMedians),
            (Format-MatchedReportNumber $case.comparison.medianPairedRatio),
            (Format-MatchedReportNumber $case.comparison.pairedRatioMad),
            $case.comparison.wins, $case.comparison.losses, $case.comparison.ties,
            $case.comparison.orderCohorts.candidateFirst.direction,
            $case.comparison.orderCohorts.candidateSecond.direction,
            $case.comparison.classification))
    }
    $lines.Add("")
    $lines.Add("All timed canonical reports and the untimed final ``U``/``p`` IEEE-754 field oracles are exact across the two refs; otherwise this script fails closed before writing a summary.")
    Set-Content -LiteralPath $markdownPath -Value $lines -Encoding UTF8
    Assert-ExactNames $OutRoot @("input-manifest.json", "summary.json", "summary.md") @("metadata", "raw") "completed benchmark output root"
    $completed = $true
    Write-Output "wrote Ferrum Linux ref A/B JSON: $jsonPath"
    Write-Output "wrote Ferrum Linux ref A/B Markdown: $markdownPath"
} finally {
    if ($completed -and (Test-Path -LiteralPath $stageRoot)) {
        Remove-Item -LiteralPath $stageRoot -Recurse -Force
    } elseif (!$completed -and (Test-Path -LiteralPath $stageRoot)) {
        Write-Warning "host staging preserved after failure: $stageRoot"
    }
}
