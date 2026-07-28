param(
    [string]$SourceRef = "HEAD",
    [int]$WarmupRuns = 2,
    [int]$MeasuredRuns = 20,
    [string]$Distro = "Ubuntu-22.04",
    [string]$CpuSet = "2",
    [string]$RustToolchain = "1.94.0",
    [string]$OutRoot = "",
    [switch]$PreflightOnly,
    [switch]$KeepWslWorkspace
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "matched_cpu_solver_common.ps1")

$RepoRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
$TargetRoot = Join-Path $RepoRoot "target"
$WorkerPath = Join-Path $PSScriptRoot "run_ferrum_linux_pgo_ab_worker.sh"
$ControllerPath = [System.IO.Path]::GetFullPath($MyInvocation.MyCommand.Path)
$TargetTriple = "x86_64-unknown-linux-gnu"
if (!(Test-Path -LiteralPath $WorkerPath -PathType Leaf)) { throw "Native-PGO worker was not found: $WorkerPath" }
if ($RustToolchain -cne "1.94.0") { throw "this benchmark requires exact Rust 1.94.0" }
if ($CpuSet -notmatch "^[0-9]+([,-][0-9]+)*$") { throw "CpuSet is invalid: $CpuSet" }
if (!(($WarmupRuns -eq 0 -and $MeasuredRuns -eq 2) -or ($WarmupRuns -eq 2 -and $MeasuredRuns -eq 20))) {
    throw "only the 0+2 smoke or 2+20 decision protocol is allowed"
}
if ($null -eq (Get-Command wsl -ErrorAction SilentlyContinue)) { throw "wsl.exe was not found" }

$workerWslPath = ConvertTo-MatchedWslPath $WorkerPath $Distro
$workerBootstrap = 'set -o pipefail; tr -d ''\r'' < "\$1" | bash -s -- "\${@:2}"'
$preflightArguments = @(
    "-d", $Distro, "--", "bash", "-c", $workerBootstrap, "ferrum-linux-pgo-ab-worker", $workerWslPath,
    "--preflight-only", "--rust-toolchain", $RustToolchain, "--target-triple", $TargetTriple,
    "--cpu-set", $CpuSet,
    "--warmup-runs", $WarmupRuns.ToString([System.Globalization.CultureInfo]::InvariantCulture),
    "--measured-runs", $MeasuredRuns.ToString([System.Globalization.CultureInfo]::InvariantCulture)
)
$previousErrorActionPreference = $ErrorActionPreference
try {
    $ErrorActionPreference = "Continue"
    $preflightOutput = & wsl @preflightArguments 2>&1
    $preflightExitCode = $LASTEXITCODE
} finally {
    $ErrorActionPreference = $previousErrorActionPreference
}
if ($preflightExitCode -ne 0) { throw "Native-PGO preflight failed for '$Distro':`n$($preflightOutput -join "`n")" }
if ($PreflightOnly) { $preflightOutput | Write-Output; return }

$sourceCommit = (& git -C $RepoRoot rev-parse "$SourceRef`^{commit}" 2>$null).Trim()
if ($LASTEXITCODE -ne 0 -or $sourceCommit -notmatch "^[0-9a-f]{40}$") { throw "could not resolve SourceRef '$SourceRef' to an exact commit" }
$sourceTree = (& git -C $RepoRoot rev-parse "$sourceCommit`^{tree}").Trim()
if ($LASTEXITCODE -ne 0 -or $sourceTree -notmatch "^[0-9a-f]{40}$") { throw "could not resolve the exact source tree" }
$cargoLockBlob = (& git -C $RepoRoot rev-parse "$sourceCommit`:Cargo.lock" 2>$null).Trim()
if ($LASTEXITCODE -ne 0 -or $cargoLockBlob -notmatch "^[0-9a-f]{40}$") { throw "exact source commit has no Cargo.lock blob" }

if ([string]::IsNullOrWhiteSpace($OutRoot)) {
    $OutRoot = Join-Path $TargetRoot "benchmarks\ferrum_linux_native_pgo_ab\$($sourceCommit.Substring(0, 8))"
}
$OutRoot = [System.IO.Path]::GetFullPath($OutRoot)
$benchmarkOutputRoot = [System.IO.Path]::GetFullPath(
    (Join-Path $TargetRoot "benchmarks\ferrum_linux_native_pgo_ab")
).TrimEnd("\", "/")
if (!(Test-MatchedPathUnder $OutRoot $benchmarkOutputRoot) -or
    $OutRoot.Equals($benchmarkOutputRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "OutRoot must be a strict child of the dedicated Native-PGO benchmark root: $benchmarkOutputRoot"
}

$stageRoot = Join-Path $TargetRoot "benchmarks\.linux-native-pgo-stage-$PID"
Reset-MatchedTargetDirectory $stageRoot $TargetRoot
$completed = $false
$performanceRejected = $false
try {
    $sourceArchive = Join-Path $stageRoot "source.tar"
    & git -C $RepoRoot archive --format=tar --output=$sourceArchive $sourceCommit
    if ($LASTEXITCODE -ne 0) { throw "could not archive exact source commit" }
    Assert-MatchedSafeTarArchive $sourceArchive "exact source"
    $sourceArchiveSha256 = (Get-FileHash -LiteralPath $sourceArchive -Algorithm SHA256).Hash.ToLowerInvariant()

    $sourceExtractRoot = Join-Path $stageRoot "source"
    New-Item -ItemType Directory -Force -Path $sourceExtractRoot | Out-Null
    Assert-MatchedNoReparsePath $sourceExtractRoot $stageRoot
    & tar -xf $sourceArchive -C $sourceExtractRoot
    if ($LASTEXITCODE -ne 0) { throw "could not extract exact source archive" }
    $cargoLockSha256 = (Get-FileHash -LiteralPath (Join-Path $sourceExtractRoot "Cargo.lock") -Algorithm SHA256).Hash.ToLowerInvariant()

    $contract = Get-MatchedCpuCaseDefinitions $sourceExtractRoot "gamg" "all"
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
    if ($LASTEXITCODE -ne 0) { throw "could not create matched template archive" }
    Assert-MatchedSafeTarArchive $templatesArchive "matched Ferrum templates"
    $templatesArchiveSha256 = (Get-FileHash -LiteralPath $templatesArchive -Algorithm SHA256).Hash.ToLowerInvariant()

    $decisionEligible = $WarmupRuns -eq 2 -and $MeasuredRuns -eq 20
    $controllerSha256 = (Get-FileHash -LiteralPath $ControllerPath -Algorithm SHA256).Hash.ToLowerInvariant()
    $workerSha256 = (Get-FileHash -LiteralPath $WorkerPath -Algorithm SHA256).Hash.ToLowerInvariant()
    $inputManifest = [pscustomobject][ordered]@{
        schemaVersion = 1
        benchmark = "ferrum-linux-native-pgo-ab"
        source = [pscustomobject][ordered]@{ commit = $sourceCommit; tree = $sourceTree; archiveSha256 = $sourceArchiveSha256 }
        cargoLock = [pscustomobject][ordered]@{ blob = $cargoLockBlob; sha256 = $cargoLockSha256 }
        rust = [pscustomobject][ordered]@{ toolchain = $RustToolchain; target = $TargetTriple }
        pressureSolver = "gamg"
        controls = [pscustomobject][ordered]@{ controllerSha256 = $controllerSha256; workerSha256 = $workerSha256 }
        build = [pscustomobject][ordered]@{
            portableCargoReleaseUnchanged = $true
            commonRustflags = "-C target-cpu=native"
            incremental = $false
            explicitTarget = $true
        }
        trainingOrder = @(
            [pscustomobject][ordered]@{ ordinal = 1; name = "laminarPipe"; fixedIterations = 10 },
            [pscustomobject][ordered]@{ ordinal = 2; name = "planeChannel"; fixedIterations = 500 }
        )
        protocol = [pscustomobject][ordered]@{
            warmupRuns = $WarmupRuns
            measuredRuns = $MeasuredRuns
            pairedAlternatingOrder = $true
            decisionEligible = $decisionEligible
            fullDecisionProtocol = "2+20/all"
            smokeProtocol = "0+2/all"
        }
        cases = $manifestCases
    }
    $manifestPath = Join-Path $stageRoot "input-manifest.json"
    $inputManifest | ConvertTo-Json -Depth 14 | Set-Content -LiteralPath $manifestPath -Encoding UTF8

    $outputArchive = Join-Path $stageRoot "ferrum-linux-native-pgo-ab-results.tar"
    $runArguments = @(
        "-d", $Distro, "--", "bash", "-c", $workerBootstrap, "ferrum-linux-pgo-ab-worker", $workerWslPath,
        "--rust-toolchain", $RustToolchain, "--target-triple", $TargetTriple, "--cpu-set", $CpuSet,
        "--warmup-runs", $WarmupRuns.ToString([System.Globalization.CultureInfo]::InvariantCulture),
        "--measured-runs", $MeasuredRuns.ToString([System.Globalization.CultureInfo]::InvariantCulture),
        "--source-archive", (ConvertTo-MatchedWslPath $sourceArchive $Distro),
        "--source-archive-sha256", $sourceArchiveSha256, "--source-commit", $sourceCommit, "--source-tree", $sourceTree,
        "--templates-archive", (ConvertTo-MatchedWslPath $templatesArchive $Distro),
        "--templates-archive-sha256", $templatesArchiveSha256,
        "--manifest", (ConvertTo-MatchedWslPath $manifestPath $Distro),
        "--output-archive", (ConvertTo-MatchedWslPath $outputArchive $Distro),
        "--controller-source", (ConvertTo-MatchedWslPath $ControllerPath $Distro), "--controller-sha256", $controllerSha256,
        "--worker-source", $workerWslPath, "--worker-sha256", $workerSha256
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
    if ($workerExitCode -ne 0) { throw "Native-PGO worker failed with exit code $workerExitCode" }
    if (!(Test-Path -LiteralPath $outputArchive -PathType Leaf)) { throw "worker did not return its result archive" }
    $sidecarPath = "$outputArchive.sha256"
    if (!(Test-Path -LiteralPath $sidecarPath -PathType Leaf)) { throw "worker did not return its result SHA sidecar" }
    $actualOutputSha256 = (Get-FileHash -LiteralPath $outputArchive -Algorithm SHA256).Hash.ToLowerInvariant()
    if ((Get-Content -LiteralPath $sidecarPath -Raw).Trim() -ne $actualOutputSha256) { throw "result archive SHA-256 verification failed" }
    Assert-MatchedSafeTarArchive $outputArchive "Native-PGO result"

    Reset-MatchedTargetDirectory $OutRoot $TargetRoot
    & tar -xf $outputArchive -C $OutRoot
    if ($LASTEXITCODE -ne 0) { throw "could not extract Native-PGO result archive" }
    Copy-Item -LiteralPath $manifestPath -Destination (Join-Path $OutRoot "input-manifest.json") -Force

    function Assert-ExactNames([string]$Root, [string[]]$ExpectedFiles, [string[]]$ExpectedDirectories, [string]$Description) {
        $actualFiles = [string[]]@(Get-ChildItem -LiteralPath $Root -Force -File | ForEach-Object { $_.Name })
        $actualDirectories = [string[]]@(Get-ChildItem -LiteralPath $Root -Force -Directory | ForEach-Object { $_.Name })
        if (@(Compare-Object $ExpectedFiles $actualFiles -CaseSensitive).Count -ne 0 -or
            @(Compare-Object $ExpectedDirectories $actualDirectories -CaseSensitive).Count -ne 0) {
            throw "$Description inventory differs from the exact contract"
        }
    }
    function Assert-RecordedValue([string]$Name, [string]$Expected) {
        $actual = (Get-Content -LiteralPath (Join-Path $metadataRoot $Name) -Raw).Trim()
        if ($actual -cne $Expected) { throw "metadata '$Name' differs from the exact binding" }
    }
    function Get-CanonicalHash([string]$RunRoot) {
        $canonicalPath = Join-Path $RunRoot "canonical-report.json"
        $expected = (Get-Content -LiteralPath (Join-Path $RunRoot "canonical-report.sha256") -Raw).Trim()
        $actual = (Get-FileHash -LiteralPath $canonicalPath -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($expected -notmatch "^[0-9a-f]{64}$" -or $expected -ne $actual) { throw "canonical report SHA-256 verification failed: $RunRoot" }
        return $expected
    }
    function Read-TimedRun($Case, [string]$Kind, [int]$Ordinal, [string]$Build) {
        $runRoot = Join-Path $OutRoot "raw\$($Case.name)\$Kind-$Ordinal-$Build"
        Assert-ExactNames $runRoot @("canonical-report.json", "canonical-report.sha256", "ferrum.log", "process-time.env", "solve-report.json") @("case") "$($Case.name) $Kind $Ordinal $Build"
        $timing = Read-MatchedGnuTime (Join-Path $runRoot "process-time.env")
        if ($timing.exitCode -ne 0 -or $timing.elapsedSeconds -le 0.0) { throw "invalid timing record: $runRoot" }
        $report = Get-Content -LiteralPath (Join-Path $runRoot "solve-report.json") -Raw | ConvertFrom-Json
        if ([int]$report.solve.simpleIterations -ne [int]$Case.fixedIterations -or
            [int]$report.linearSolves.momentumComponentNonConvergedSolves -ne 0 -or
            [int]$report.linearSolves.pressureCorrectionNonConvergedSolves -ne 0 -or
            @($report.history).Count -ne [int]$Case.fixedIterations) {
            throw "fixed-work report contract failed: $runRoot"
        }
        return [pscustomobject][ordered]@{
            case = $Case.name; kind = $Kind; ordinal = $Ordinal; build = $Build
            elapsedSeconds = $timing.elapsedSeconds; userSeconds = $timing.userSeconds
            systemSeconds = $timing.systemSeconds; maxResidentSetKiB = $timing.maxResidentSetKiB
            canonicalReportSha256 = Get-CanonicalHash $runRoot
        }
    }
    function Read-Oracle($Case, [string]$Build) {
        $runRoot = Join-Path $OutRoot "raw\$($Case.name)\oracle-$Build"
        Assert-ExactNames $runRoot @("canonical-report.json", "canonical-report.sha256", "ferrum.log", "field-oracle.json", "solve-report.json") @("case", "final-fields") "$($Case.name) $Build oracle"
        Assert-ExactNames (Join-Path $runRoot "final-fields") @("U", "p") @() "$($Case.name) $Build oracle fields"
        $oracle = Get-Content -LiteralPath (Join-Path $runRoot "field-oracle.json") -Raw | ConvertFrom-Json
        if ([int]$oracle.schemaVersion -ne 2 -or [int]$oracle.cellCount -le 0 -or
            [string]$oracle.combinedIeee754Sha256 -notmatch "^[0-9a-f]{64}$") { throw "invalid field oracle: $runRoot" }
        foreach ($field in @("U", "p")) {
            if ((Get-FileHash -LiteralPath (Join-Path $runRoot "final-fields\$field") -Algorithm SHA256).Hash.ToLowerInvariant() -ne [string]$oracle.$field.textSha256) {
                throw "$($Case.name) $Build $field text SHA differs from its oracle"
            }
            foreach ($hashName in @("ieee754BigEndianSha256", "boundaryIeee754BigEndianSha256", "fullFieldIeee754BigEndianSha256")) {
                if ([string]$oracle.$field.$hashName -notmatch "^[0-9a-f]{64}$") { throw "invalid $field $hashName oracle" }
            }
        }
        return [pscustomobject][ordered]@{ canonicalReportSha256 = Get-CanonicalHash $runRoot; field = $oracle }
    }

    $metadataRoot = Join-Path $OutRoot "metadata"
    Assert-ExactNames $OutRoot @("input-manifest.json") @("binaries", "controls", "metadata", "profiles", "raw") "result root before summary"
    Assert-ExactNames (Join-Path $OutRoot "binaries") @("ferrumRun-instrumented", "ferrumRun-native", "ferrumRun-pgo") @() "exported binaries"
    Assert-ExactNames (Join-Path $OutRoot "controls") @("run_ferrum_linux_pgo_ab_benchmark.ps1", "run_ferrum_linux_pgo_ab_worker.sh") @() "exported controls"
    Assert-ExactNames (Join-Path $OutRoot "profiles") @("merged.profdata") @("raw") "exported profiles"
    $expectedMetadataFiles = @(
        "build-instrumented-time.env", "build-native-time.env", "build-pgo-time.env",
        "cargo-build-instrumented-release.log", "cargo-build-native-release.log", "cargo-build-pgo-release.log",
        "cargo-lock-sha256.txt", "cargo-version.txt", "controller-script-sha256.txt", "cpu-model.txt", "cpu-set.txt", "cpu-siblings.txt", "distro-release.txt", "filesystem-type.txt",
        "input-manifest.json", "instrumented-binary-sha256.txt", "instrumented-readelf-sections.txt", "instrumented-rustflags.txt",
        "llvm-profdata-merged-sha256.txt", "llvm-profdata-path.txt", "llvm-profdata-sha256.txt", "llvm-profdata-show.txt", "llvm-profdata-version.txt",
        "llvm-profraw-count.txt", "llvm-profraw-inventory.tsv", "native-binary-sha256.txt", "native-readelf-sections.txt", "native-rustflags.txt",
        "pgo-binary-sha256.txt", "pgo-readelf-sections.txt", "pgo-rustflags.txt", "run-order.tsv", "rustc-vv.txt",
        "source-archive-sha256.txt", "source-commit.txt", "source-tree.txt", "target-triple.txt", "templates-archive-sha256.txt",
        "training-order.tsv", "uname.txt", "worker-script-sha256.txt", "workspace-path.txt"
    )
    Assert-ExactNames $metadataRoot $expectedMetadataFiles @() "Native-PGO metadata"
    Assert-RecordedValue "source-commit.txt" $sourceCommit
    Assert-RecordedValue "source-tree.txt" $sourceTree
    Assert-RecordedValue "source-archive-sha256.txt" $sourceArchiveSha256
    Assert-RecordedValue "templates-archive-sha256.txt" $templatesArchiveSha256
    Assert-RecordedValue "cargo-lock-sha256.txt" $cargoLockSha256
    Assert-RecordedValue "target-triple.txt" $TargetTriple
    Assert-RecordedValue "cpu-set.txt" $CpuSet
    Assert-RecordedValue "native-rustflags.txt" "-C target-cpu=native"
    Assert-RecordedValue "controller-script-sha256.txt" $controllerSha256
    Assert-RecordedValue "worker-script-sha256.txt" $workerSha256
    if ((Get-FileHash -LiteralPath (Join-Path $OutRoot "controls\run_ferrum_linux_pgo_ab_benchmark.ps1") -Algorithm SHA256).Hash.ToLowerInvariant() -ne $controllerSha256 -or
        (Get-FileHash -LiteralPath (Join-Path $OutRoot "controls\run_ferrum_linux_pgo_ab_worker.sh") -Algorithm SHA256).Hash.ToLowerInvariant() -ne $workerSha256) {
        throw "exported control-source provenance differs"
    }
    $rootManifestSha = (Get-FileHash -LiteralPath (Join-Path $OutRoot "input-manifest.json") -Algorithm SHA256).Hash.ToLowerInvariant()
    $metadataManifestSha = (Get-FileHash -LiteralPath (Join-Path $metadataRoot "input-manifest.json") -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($rootManifestSha -ne $metadataManifestSha) { throw "root and worker input manifests differ byte-for-byte" }

    $rustcVv = Get-Content -LiteralPath (Join-Path $metadataRoot "rustc-vv.txt") -Raw
    $llvmMatch = [regex]::Match($rustcVv, "(?m)^LLVM version: ([0-9.]+)$")
    if ($rustcVv -notmatch "(?m)^rustc 1\.94\.0 " -or $rustcVv -notmatch "(?m)^host: x86_64-unknown-linux-gnu$" -or !$llvmMatch.Success) {
        throw "recorded rustc identity differs from exact Rust/target contract"
    }
    $rustLlvm = $llvmMatch.Groups[1].Value
    $profdataVersion = Get-Content -LiteralPath (Join-Path $metadataRoot "llvm-profdata-version.txt") -Raw
    if ($profdataVersion -notmatch "LLVM version $([regex]::Escape($rustLlvm))") { throw "llvm-profdata version differs from rustc LLVM" }
    $profdataPath = (Get-Content -LiteralPath (Join-Path $metadataRoot "llvm-profdata-path.txt") -Raw).Trim()
    if ($profdataPath -notmatch "/\.rustup/toolchains/1\.94\.0-x86_64-unknown-linux-gnu/lib/rustlib/x86_64-unknown-linux-gnu/bin/llvm-profdata$") {
        throw "llvm-profdata was not bound to the exact Rust toolchain sysroot"
    }
    $recordedWorkspace = (Get-Content -LiteralPath (Join-Path $metadataRoot "workspace-path.txt") -Raw).Trim().TrimEnd("/")
    if ($recordedWorkspace -notmatch "^/home/[^/]+/\.cache/ferrumcfd-linux-native-pgo-ab/run\.[A-Za-z0-9]+$") { throw "worker workspace path is outside its exact ext4 cache root" }
    $instrumentedRustflags = (Get-Content -LiteralPath (Join-Path $metadataRoot "instrumented-rustflags.txt") -Raw).Trim()
    $pgoRustflags = (Get-Content -LiteralPath (Join-Path $metadataRoot "pgo-rustflags.txt") -Raw).Trim()
    if ($instrumentedRustflags -cne "-C target-cpu=native -C profile-generate=$recordedWorkspace/profiles/raw") {
        throw "instrumented Rustflags differ from the exact absolute profile-generate contract"
    }
    if ($pgoRustflags -cne "-C target-cpu=native -C profile-use=$recordedWorkspace/profiles/merged.profdata -C llvm-args=-pgo-warn-missing-function") {
        throw "PGO Rustflags differ from the exact absolute profile-use contract"
    }
    foreach ($name in @("llvm-profdata-sha256.txt", "llvm-profdata-merged-sha256.txt", "native-binary-sha256.txt", "instrumented-binary-sha256.txt", "pgo-binary-sha256.txt")) {
        if ((Get-Content -LiteralPath (Join-Path $metadataRoot $name) -Raw).Trim() -notmatch "^[0-9a-f]{64}$") { throw "invalid SHA metadata: $name" }
    }
    foreach ($build in @("native", "instrumented", "pgo")) {
        $expected = (Get-Content -LiteralPath (Join-Path $metadataRoot "$build-binary-sha256.txt") -Raw).Trim()
        $actual = (Get-FileHash -LiteralPath (Join-Path $OutRoot "binaries\ferrumRun-$build") -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($actual -ne $expected) { throw "exported $build binary SHA differs" }
    }
    $mergedProfileSha = (Get-FileHash -LiteralPath (Join-Path $OutRoot "profiles\merged.profdata") -Algorithm SHA256).Hash.ToLowerInvariant()
    Assert-RecordedValue "llvm-profdata-merged-sha256.txt" $mergedProfileSha
    $profileSectionPattern = "(__llvm_prf_(cnts|data|names)|\.llvm_prf_(cnts|data|names))"
    if ((Get-Content -LiteralPath (Join-Path $metadataRoot "native-readelf-sections.txt") -Raw) -match $profileSectionPattern -or
        (Get-Content -LiteralPath (Join-Path $metadataRoot "pgo-readelf-sections.txt") -Raw) -match $profileSectionPattern -or
        (Get-Content -LiteralPath (Join-Path $metadataRoot "instrumented-readelf-sections.txt") -Raw) -notmatch $profileSectionPattern) {
        throw "LLVM profiling section proof differs"
    }
    $instrumentedSections = Get-Content -LiteralPath (Join-Path $metadataRoot "instrumented-readelf-sections.txt") -Raw
    foreach ($section in @("cnts", "data", "names")) {
        if ($instrumentedSections -notmatch "(__llvm_prf_$section|\.llvm_prf_$section)") { throw "instrumented binary is missing profile section '$section'" }
    }

    $trainingRows = @(Import-Csv -LiteralPath (Join-Path $metadataRoot "training-order.tsv") -Delimiter "`t")
    if ($trainingRows.Count -ne 2 -or [string]$trainingRows[0].case -cne "laminarPipe" -or [int]$trainingRows[0].fixedIterations -ne 10 -or
        [string]$trainingRows[1].case -cne "planeChannel" -or [int]$trainingRows[1].fixedIterations -ne 500 -or
        [int]$trainingRows[0].profrawCountAfter -lt 1 -or [int]$trainingRows[1].profrawCountAfter -le [int]$trainingRows[0].profrawCountAfter) {
        throw "training order/profile growth proof differs"
    }
    $rawInventory = @(Import-Csv -LiteralPath (Join-Path $metadataRoot "llvm-profraw-inventory.tsv") -Delimiter "`t")
    if ($rawInventory.Count -ne [int]$trainingRows[1].profrawCountAfter -or $rawInventory.Count -ne [int](Get-Content -LiteralPath (Join-Path $metadataRoot "llvm-profraw-count.txt") -Raw).Trim()) {
        throw "raw profile inventory count differs"
    }
    foreach ($row in $rawInventory) {
        if ([string]$row.name -notmatch "^[A-Za-z0-9._-]+\.profraw$" -or [long]$row.sizeBytes -le 0 -or [string]$row.sha256 -notmatch "^[0-9a-f]{64}$") {
            throw "raw profile inventory contains an invalid row"
        }
    }
    $expectedRawNames = [string[]]@($rawInventory | ForEach-Object { [string]$_.name })
    Assert-ExactNames (Join-Path $OutRoot "profiles\raw") $expectedRawNames @() "exported raw profiles"
    foreach ($row in $rawInventory) {
        $rawPath = Join-Path $OutRoot "profiles\raw\$($row.name)"
        if ((Get-Item -LiteralPath $rawPath).Length -ne [long]$row.sizeBytes -or
            (Get-FileHash -LiteralPath $rawPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne [string]$row.sha256) {
            throw "exported raw profile differs from its bound inventory: $($row.name)"
        }
    }

    $expectedOrder = @()
    $expectedRawCaseDirectories = @()
    foreach ($case in $contract.cases) {
        $expectedRunDirectories = @()
        foreach ($kind in @("warmup", "measured")) {
            $count = if ($kind -eq "warmup") { $WarmupRuns } else { $MeasuredRuns }
            for ($ordinal = 1; $ordinal -le $count; $ordinal++) {
                $builds = if (($ordinal % 2) -eq 1) { @("native", "pgo") } else { @("pgo", "native") }
                for ($position = 1; $position -le 2; $position++) {
                    $expectedOrder += "$($case.name)|$kind|$ordinal|$position|$($builds[$position - 1])"
                    $expectedRunDirectories += "$kind-$ordinal-$($builds[$position - 1])"
                }
            }
        }
        $expectedRunDirectories += @("oracle-native", "oracle-pgo")
        Assert-ExactNames (Join-Path $OutRoot "raw\$($case.name)") @() $expectedRunDirectories "$($case.name) raw runs"
        $expectedRawCaseDirectories += $case.name
    }
    Assert-ExactNames (Join-Path $OutRoot "raw") @() $expectedRawCaseDirectories "raw case root"
    $actualOrder = @(Import-Csv -LiteralPath (Join-Path $metadataRoot "run-order.tsv") -Delimiter "`t" | ForEach-Object {
        "$($_.case)|$($_.kind)|$($_.ordinal)|$($_.position)|$($_.build)"
    })
    if (@(Compare-Object $expectedOrder $actualOrder -SyncWindow 0 -CaseSensitive).Count -ne 0) { throw "run order differs from balanced exact sequence" }

    $caseSummaries = @()
    foreach ($case in $contract.cases) {
        $allRuns = @()
        foreach ($kind in @("warmup", "measured")) {
            $count = if ($kind -eq "warmup") { $WarmupRuns } else { $MeasuredRuns }
            for ($ordinal = 1; $ordinal -le $count; $ordinal++) {
                foreach ($build in @("native", "pgo")) { $allRuns += Read-TimedRun $case $kind $ordinal $build }
            }
        }
        $nativeOracle = Read-Oracle $case "native"
        $pgoOracle = Read-Oracle $case "pgo"
        $canonicalHashes = @($allRuns.canonicalReportSha256 + $nativeOracle.canonicalReportSha256 + $pgoOracle.canonicalReportSha256 | Sort-Object -Unique)
        if ($canonicalHashes.Count -ne 1) { throw "$($case.name) canonical reports differ across builds/runs" }
        foreach ($fieldName in @("U", "p")) {
            if ([string]$nativeOracle.field.$fieldName.textSha256 -cne [string]$pgoOracle.field.$fieldName.textSha256) {
                throw "$($case.name) $fieldName text SHA differs between Native and PGO"
            }
            foreach ($hashName in @("ieee754BigEndianSha256", "boundaryIeee754BigEndianSha256", "fullFieldIeee754BigEndianSha256")) {
                if ([string]$nativeOracle.field.$fieldName.$hashName -cne [string]$pgoOracle.field.$fieldName.$hashName) {
                    throw "$($case.name) $fieldName $hashName differs between Native and PGO"
                }
            }
        }
        if ([string]$nativeOracle.field.combinedIeee754Sha256 -cne [string]$pgoOracle.field.combinedIeee754Sha256) {
            throw "$($case.name) combined U/p IEEE-754 oracle differs"
        }

        $nativeMeasured = @($allRuns | Where-Object { $_.kind -eq "measured" -and $_.build -eq "native" } | Sort-Object ordinal)
        $pgoMeasured = @($allRuns | Where-Object { $_.kind -eq "measured" -and $_.build -eq "pgo" } | Sort-Object ordinal)
        $nativeMedian = Get-MatchedMedian ([double[]]@($nativeMeasured.elapsedSeconds))
        $pgoMedian = Get-MatchedMedian ([double[]]@($pgoMeasured.elapsedSeconds))
        $ratios = [double[]]@()
        $pgoFirst = [double[]]@()
        $pgoSecond = [double[]]@()
        $wins = 0; $losses = 0; $ties = 0
        for ($index = 0; $index -lt $MeasuredRuns; $index++) {
            $ratio = [double]$pgoMeasured[$index].elapsedSeconds / [double]$nativeMeasured[$index].elapsedSeconds
            $ratios += $ratio
            if ((($index + 1) % 2) -eq 0) { $pgoFirst += $ratio } else { $pgoSecond += $ratio }
            if ($ratio -lt 1.0) { $wins++ } elseif ($ratio -gt 1.0) { $losses++ } else { $ties++ }
        }
        $medianRatio = Get-MatchedMedian $ratios
        $ratioMad = Get-MatchedMedianAbsoluteDeviation $ratios
        $firstMedian = Get-MatchedMedian $pgoFirst
        $secondMedian = Get-MatchedMedian $pgoSecond
        $gates = [pscustomobject][ordered]@{
            medianFaster = $pgoMedian -lt $nativeMedian
            atLeastFourteenOfTwentyWins = $decisionEligible -and $wins -ge 14
            pgoFirstCohortFaster = $firstMedian -lt 1.0
            pgoSecondCohortFaster = $secondMedian -lt 1.0
            gainExceedsTwiceMad = (1.0 - $medianRatio) -gt (2.0 * $ratioMad)
            exactCanonicalReportParity = $true
            exactFinalFieldIeee754Parity = $true
        }
        $accepted = $decisionEligible -and $gates.medianFaster -and $gates.atLeastFourteenOfTwentyWins -and
            $gates.pgoFirstCohortFaster -and $gates.pgoSecondCohortFaster -and $gates.gainExceedsTwiceMad
        $caseSummaries += [pscustomobject][ordered]@{
            case = $case.name; fixedIterations = $case.fixedIterations; decisionEligible = $decisionEligible; accepted = $accepted
            nativeMedianSeconds = $nativeMedian; pgoMedianSeconds = $pgoMedian
            pgoOverNativeRatioOfMedians = $pgoMedian / $nativeMedian
            medianPairedRatio = $medianRatio; pairedRatioMad = $ratioMad
            wins = $wins; losses = $losses; ties = $ties
            orderCohorts = [pscustomobject][ordered]@{ pgoFirstMedianRatio = $firstMedian; pgoSecondMedianRatio = $secondMedian }
            gates = $gates; canonicalReportSha256 = $canonicalHashes[0]
            fieldOracle = [pscustomobject][ordered]@{ combinedIeee754Sha256 = $nativeOracle.field.combinedIeee754Sha256; exact = $true }
            runs = $allRuns
        }
    }

    $generalAccepted = $decisionEligible -and @($caseSummaries | Where-Object { !$_.accepted }).Count -eq 0
    $summary = [pscustomobject][ordered]@{
        schemaVersion = 1
        benchmark = "ferrum-linux-native-pgo-ab"
        source = $inputManifest.source
        cargoLock = $inputManifest.cargoLock
        protocol = $inputManifest.protocol
        build = [pscustomobject][ordered]@{
            portableCargoReleaseUnchanged = $true
            baseline = "native"
            candidate = "native+pgo"
            rustToolchain = $RustToolchain
            target = $TargetTriple
            llvmProfdataPath = $profdataPath
            llvmProfdataSha256 = (Get-Content -LiteralPath (Join-Path $metadataRoot "llvm-profdata-sha256.txt") -Raw).Trim()
            mergedProfileSha256 = $mergedProfileSha
            binarySha256 = [pscustomobject][ordered]@{
                native = (Get-Content -LiteralPath (Join-Path $metadataRoot "native-binary-sha256.txt") -Raw).Trim()
                instrumented = (Get-Content -LiteralPath (Join-Path $metadataRoot "instrumented-binary-sha256.txt") -Raw).Trim()
                pgo = (Get-Content -LiteralPath (Join-Path $metadataRoot "pgo-binary-sha256.txt") -Raw).Trim()
            }
        }
        cases = $caseSummaries
        decision = [pscustomobject][ordered]@{
            eligible = $decisionEligible
            generalNativePgoAccepted = $generalAccepted
            classification = if (!$decisionEligible) { "smoke-only" } elseif ($generalAccepted) { "accepted-both-cases" } else { "rejected-one-or-more-cases" }
            rule = "Both Pipe and Channel must pass every predeclared gate; one loss forbids a general default/claim."
        }
    }
    $jsonPath = Join-Path $OutRoot "summary.json"
    $markdownPath = Join-Path $OutRoot "summary.md"
    $summary | ConvertTo-Json -Depth 24 | Set-Content -LiteralPath $jsonPath -Encoding UTF8
    $lines = [System.Collections.Generic.List[string]]::new()
    $lines.Add("# Ferrum Linux Native-PGO A/B")
    $lines.Add("")
    $lines.Add("Exact source: ``$sourceCommit`` (tree ``$sourceTree``); Rust ``$RustToolchain``; target ``$TargetTriple``.")
    $lines.Add("")
    $lines.Add("Protocol: $WarmupRuns warmup + $MeasuredRuns measured pairs per case. Decision eligible: ``$($decisionEligible.ToString().ToLowerInvariant())``.")
    $lines.Add("")
    $lines.Add("| Case | Native median [s] | PGO median [s] | Paired ratio | MAD | W/L/T | PGO-first | PGO-second | Accepted |")
    $lines.Add("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    foreach ($caseSummary in $caseSummaries) {
        $lines.Add(("| {0} | {1} | {2} | {3} | {4} | {5}/{6}/{7} | {8} | {9} | {10} |" -f
            $caseSummary.case, (Format-MatchedReportNumber $caseSummary.nativeMedianSeconds),
            (Format-MatchedReportNumber $caseSummary.pgoMedianSeconds), (Format-MatchedReportNumber $caseSummary.medianPairedRatio),
            (Format-MatchedReportNumber $caseSummary.pairedRatioMad), $caseSummary.wins, $caseSummary.losses, $caseSummary.ties,
            (Format-MatchedReportNumber $caseSummary.orderCohorts.pgoFirstMedianRatio),
            (Format-MatchedReportNumber $caseSummary.orderCohorts.pgoSecondMedianRatio), $caseSummary.accepted))
    }
    $lines.Add("")
    $lines.Add("Native and PGO canonical solve reports plus final ``U``/``p`` internal and boundary IEEE-754 hashes are exact; any mismatch fails before this summary is written.")
    $lines.Add("")
    $lines.Add("General classification: **$($summary.decision.classification)**. A smoke run can never authorize a claim or merge decision.")
    $lines | Set-Content -LiteralPath $markdownPath -Encoding UTF8
    Assert-ExactNames $OutRoot @("input-manifest.json", "summary.json", "summary.md") @("binaries", "controls", "metadata", "profiles", "raw") "completed benchmark result"
    $completed = $true
    if ($decisionEligible -and !$generalAccepted) { $performanceRejected = $true }
} finally {
    if ($completed -and (Test-Path -LiteralPath $stageRoot)) {
        Assert-MatchedNoReparsePath $stageRoot $TargetRoot
        Remove-Item -LiteralPath $stageRoot -Recurse -Force
    }
}

if ($performanceRejected) {
    throw "Native-PGO full decision rejected one or more cases; summary was preserved at $OutRoot"
}
Write-Output "Native-PGO benchmark complete: $OutRoot"
