param(
    [int]$WarmupRuns = 1,
    [int]$MeasuredRuns = 5,
    [ValidateSet("pcg", "gamg")]
    [string]$PressureSolver = "gamg",
    [ValidateSet("all", "laminarPipe", "planeChannel")]
    [string]$CaseName = "all",
    [ValidateSet("Auto", "Native", "Wsl")]
    [string]$OpenFoamMode = "Auto",
    [string]$OutRoot = "",
    [switch]$RequireOpenFoam
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
if ([string]::IsNullOrWhiteSpace($OutRoot)) {
    $OutRoot = Join-Path $RepoRoot "target\benchmarks\matched_cpu_solver\$PressureSolver"
}
$OutRoot = [System.IO.Path]::GetFullPath($OutRoot)
if ($WarmupRuns -lt 0) {
    throw "WarmupRuns must be zero or greater"
}
if ($MeasuredRuns -lt 1) {
    throw "MeasuredRuns must be at least one"
}

$matchedFvSolution = Join-Path $RepoRoot "validation\profiles\incompressibleFluid\matched-fixed\$PressureSolver\system\fvSolution"
if (!(Test-Path -LiteralPath $matchedFvSolution -PathType Leaf)) {
    throw "matched fixed-work fvSolution was not found: $matchedFvSolution"
}

$caseDefinitions = @(
    [pscustomobject][ordered]@{
        name = "laminarPipe"
        fixedIterations = 10
        ferrumCase = Join-Path $RepoRoot "tutorials\incompressibleFluid\laminarPipe\ferrum\case"
        openFoamTemplate = Join-Path $RepoRoot "tutorials\incompressibleFluid\laminarPipe\openfoam-v13\case"
    },
    [pscustomobject][ordered]@{
        name = "planeChannel"
        fixedIterations = 500
        ferrumCase = Join-Path $RepoRoot "tutorials\incompressibleFluid\planeChannel\ferrum\case"
        openFoamTemplate = Join-Path $RepoRoot "tutorials\incompressibleFluid\planeChannel\openfoam-v13\case"
    }
)
if ($CaseName -ne "all") {
    $caseDefinitions = @($caseDefinitions | Where-Object { $_.name -eq $CaseName })
}
foreach ($case in $caseDefinitions) {
    foreach ($path in @($case.ferrumCase, $case.openFoamTemplate)) {
        if (!(Test-Path -LiteralPath $path -PathType Container)) {
            throw "matched benchmark case was not found: $path"
        }
    }
}

function Format-F64([double]$Value) {
    return $Value.ToString("G17", [System.Globalization.CultureInfo]::InvariantCulture)
}

function Format-ReportNumber($Value) {
    if ($null -eq $Value) {
        return "n/a"
    }
    return ([double]$Value).ToString("G8", [System.Globalization.CultureInfo]::InvariantCulture)
}

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

function Reset-TargetDirectory([string]$Path) {
    $targetRoot = Join-Path $RepoRoot "target"
    if (!(Test-IsPathUnder $Path $targetRoot)) {
        throw "refusing to replace directory outside repository target: $Path"
    }
    if (Test-Path -LiteralPath $Path) {
        Remove-Item -LiteralPath $Path -Recurse -Force
    }
    New-Item -ItemType Directory -Force -Path $Path | Out-Null
}

function Write-AsciiFile([string]$Path, [string]$Content) {
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Path) | Out-Null
    Set-Content -LiteralPath $Path -Value $Content -Encoding ASCII
}

function ConvertTo-WslPath([string]$Path) {
    $full = [System.IO.Path]::GetFullPath($Path)
    if (Test-Path -LiteralPath $full) {
        $resolved = (Resolve-Path -LiteralPath $full).Path
    } else {
        $parent = Split-Path -Parent $full
        if (!(Test-Path -LiteralPath $parent -PathType Container)) {
            throw "could not convert '$Path' to a WSL path because its parent does not exist"
        }
        $resolved = Join-Path (Resolve-Path -LiteralPath $parent).Path (Split-Path -Leaf $full)
    }
    if ($resolved -match "^([A-Za-z]):\\(.*)$") {
        $drive = $Matches[1].ToLowerInvariant()
        $rest = $Matches[2].Replace("\", "/")
        return "/mnt/$drive/$rest"
    }
    $converted = & wsl wslpath -a -u $resolved
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($converted)) {
        throw "could not convert '$resolved' to a WSL path"
    }
    return $converted.Trim()
}

function ConvertTo-BashSingleQuoted([string]$Value) {
    $singleQuoteEscape = "'" + '"' + "'" + '"' + "'"
    return "'" + $Value.Replace("'", $singleQuoteEscape) + "'"
}

function Test-NativeOpenFoam {
    return $env:WM_PROJECT_VERSION -eq "13" -and
        $null -ne (Get-Command foamRun -ErrorAction SilentlyContinue)
}

function Test-WslOpenFoam {
    if ($null -eq (Get-Command wsl -ErrorAction SilentlyContinue)) {
        return $false
    }
    & wsl bash -lc "source /opt/openfoam13/etc/bashrc 2>/dev/null && env | grep -q '^WM_PROJECT_VERSION=13$' && command -v foamRun >/dev/null 2>&1"
    return $LASTEXITCODE -eq 0
}

function Get-OpenFoamMode {
    if ($OpenFoamMode -eq "Native") {
        if (Test-NativeOpenFoam) { return "Native" }
        return $null
    }
    if ($OpenFoamMode -eq "Wsl") {
        if (Test-WslOpenFoam) { return "Wsl" }
        return $null
    }
    if (Test-NativeOpenFoam) { return "Native" }
    if (Test-WslOpenFoam) { return "Wsl" }
    return $null
}

function Read-DimensionedScalar([string]$Path, [string]$Name) {
    $content = Get-Content -LiteralPath $Path -Raw
    $escapedName = [regex]::Escape($Name)
    $match = [regex]::Match($content, "(?m)^\s*$escapedName\s+\[[^\]]+\]\s+([-+0-9.eE]+)\s*;")
    if (!$match.Success) {
        throw "could not read dimensioned scalar '$Name' from $Path"
    }
    return [double]::Parse($match.Groups[1].Value, [System.Globalization.CultureInfo]::InvariantCulture)
}

function Read-InternalScalarField([string]$Path) {
    $content = Get-Content -LiteralPath $Path -Raw
    $uniform = [regex]::Match($content, "(?ms)\binternalField\s+uniform\s+([-+0-9.eE]+)\s*;")
    if ($uniform.Success) {
        return ,([double[]]@([double]::Parse($uniform.Groups[1].Value, [System.Globalization.CultureInfo]::InvariantCulture)))
    }
    $nonuniform = [regex]::Match($content, "(?ms)\binternalField\s+nonuniform\s+List<scalar>\s+(\d+)\s*\((.*?)\)\s*;")
    if (!$nonuniform.Success) {
        throw "unsupported scalar internalField in $Path"
    }
    $expected = [int]::Parse($nonuniform.Groups[1].Value, [System.Globalization.CultureInfo]::InvariantCulture)
    $tokens = @($nonuniform.Groups[2].Value -split "\s+" | Where-Object { ![string]::IsNullOrWhiteSpace($_) })
    if ($tokens.Count -ne $expected) {
        throw "scalar internalField in $Path declares $expected values but contains $($tokens.Count)"
    }
    $values = [double[]]@($tokens | ForEach-Object {
        [double]::Parse($_, [System.Globalization.CultureInfo]::InvariantCulture)
    })
    return ,$values
}

function Get-UniformBoundaryValues([string]$Path) {
    $content = Get-Content -LiteralPath $Path -Raw
    return [double[]]@([regex]::Matches($content, "(?m)^\s*value\s+uniform\s+([-+0-9.eE]+)\s*;") | ForEach-Object {
        [double]::Parse($_.Groups[1].Value, [System.Globalization.CultureInfo]::InvariantCulture)
    })
}

function Convert-PressureFieldToKinematic(
    [string]$FerrumPressurePath,
    [string]$OpenFoamTemplatePath,
    [string]$DestinationPath,
    [double]$Rho
) {
    $valuesPa = @(Read-InternalScalarField $FerrumPressurePath)
    $valuesKinematic = [double[]]@($valuesPa | ForEach-Object { [double]$_ / $Rho })
    $internalField = if ($valuesKinematic.Count -eq 1) {
        "internalField uniform $(Format-F64 $valuesKinematic[0]);"
    } else {
        $lines = @($valuesKinematic | ForEach-Object { "    $(Format-F64 $_)" })
        "internalField nonuniform List<scalar>`n$($valuesKinematic.Count)`n(`n$($lines -join "`n")`n);"
    }

    $sourceBoundary = @(Get-UniformBoundaryValues $FerrumPressurePath)
    $templateBoundary = @(Get-UniformBoundaryValues $OpenFoamTemplatePath)
    if ($sourceBoundary.Count -ne $templateBoundary.Count) {
        throw "pressure boundary-value count differs between Ferrum and OpenFOAM templates"
    }
    for ($index = 0; $index -lt $sourceBoundary.Count; $index++) {
        $expected = $sourceBoundary[$index] / $Rho
        $actual = $templateBoundary[$index]
        $tolerance = 1e-12 * [Math]::Max(1.0, [Math]::Abs($expected))
        if ([Math]::Abs($actual - $expected) -gt $tolerance) {
            throw "OpenFOAM pressure boundary value $index is not the Ferrum SI value divided by rho"
        }
    }

    $template = Get-Content -LiteralPath $OpenFoamTemplatePath -Raw
    $regex = [regex]::new("(?ms)\binternalField\s+(?:uniform\s+[-+0-9.eE]+\s*;|nonuniform\s+List<scalar>\s+\d+\s*\(.*?\)\s*;)")
    if ($regex.Matches($template).Count -ne 1) {
        throw "OpenFOAM template must contain exactly one supported internalField: $OpenFoamTemplatePath"
    }
    $converted = $regex.Replace($template, $internalField, 1)
    Write-AsciiFile $DestinationPath $converted
    return [pscustomobject][ordered]@{
        sourceUnits = "Pa"
        destinationUnits = "m2/s2"
        densityKgPerM3 = $Rho
        internalValues = $valuesKinematic.Count
        boundaryUniformValuesChecked = $sourceBoundary.Count
    }
}

function Get-PolyMeshHashes([string]$CaseRoot) {
    $hashes = [ordered]@{}
    foreach ($name in @("points", "faces", "owner", "neighbour", "boundary")) {
        $path = Join-Path $CaseRoot "constant\polyMesh\$name"
        if (!(Test-Path -LiteralPath $path -PathType Leaf)) {
            throw "polyMesh file was not found: $path"
        }
        $hashes[$name] = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
    }
    return [pscustomobject]$hashes
}

function Assert-HashesEqual($Expected, $Actual, [string]$Description) {
    foreach ($name in @("points", "faces", "owner", "neighbour", "boundary")) {
        if ($Expected.$name -ne $Actual.$name) {
            throw "$Description polyMesh differs in $name"
        }
    }
}

function Get-NumericOutputDirectoryCount([string]$CaseRoot) {
    return @(
        Get-ChildItem -LiteralPath $CaseRoot -Directory | Where-Object {
            if ($_.Name -eq "0") { return $false }
            $value = 0.0
            return [double]::TryParse(
                $_.Name,
                [System.Globalization.NumberStyles]::Float,
                [System.Globalization.CultureInfo]::InvariantCulture,
                [ref]$value
            )
        }
    ).Count
}

function New-FerrumWorkingCase($Case, [string]$Destination) {
    Reset-TargetDirectory $Destination
    Copy-Item -Path (Join-Path $Case.ferrumCase "*") -Destination $Destination -Recurse -Force
    Copy-Item -LiteralPath $matchedFvSolution -Destination (Join-Path $Destination "system\fvSolution") -Force
    return $Destination
}

function New-OpenFoamWorkingCase($Case, [string]$Destination, $CanonicalMeshHashes) {
    Reset-TargetDirectory $Destination
    Copy-Item -Path (Join-Path $Case.openFoamTemplate "*") -Destination $Destination -Recurse -Force

    $destinationMesh = Join-Path $Destination "constant\polyMesh"
    if (Test-Path -LiteralPath $destinationMesh) {
        Remove-Item -LiteralPath $destinationMesh -Recurse -Force
    }
    Copy-Item -LiteralPath (Join-Path $Case.ferrumCase "constant\polyMesh") -Destination $destinationMesh -Recurse
    Assert-HashesEqual $CanonicalMeshHashes (Get-PolyMeshHashes $Destination) "$($Case.name) OpenFOAM working"

    Copy-Item -LiteralPath (Join-Path $Case.ferrumCase "0\U") -Destination (Join-Path $Destination "0\U") -Force
    Copy-Item -LiteralPath (Join-Path $Case.ferrumCase "system\fvSchemes") -Destination (Join-Path $Destination "system\fvSchemes") -Force
    Copy-Item -LiteralPath $matchedFvSolution -Destination (Join-Path $Destination "system\fvSolution") -Force

    $transportPath = Join-Path $Case.ferrumCase "constant\transportProperties"
    $rho = Read-DimensionedScalar $transportPath "rho"
    $nu = Read-DimensionedScalar $transportPath "nu"
    if ($rho -le 0.0 -or $nu -le 0.0) {
        throw "rho and nu must be positive in $transportPath"
    }
    $pressureConversion = Convert-PressureFieldToKinematic `
        -FerrumPressurePath (Join-Path $Case.ferrumCase "0\p") `
        -OpenFoamTemplatePath (Join-Path $Case.openFoamTemplate "0\p") `
        -DestinationPath (Join-Path $Destination "0\p") `
        -Rho $rho

    Write-AsciiFile (Join-Path $Destination "constant\physicalProperties") @"
FoamFile
{
    version 2.0;
    format ascii;
    class dictionary;
    location "constant";
    object physicalProperties;
}

viscosityModel constant;
nu [0 2 -1 0 0 0 0] $(Format-F64 $nu);
"@

    $writeInterval = $Case.fixedIterations + 1
    Write-AsciiFile (Join-Path $Destination "system\controlDict") @"
FoamFile
{
    version 2.0;
    format ascii;
    class dictionary;
    location "system";
    object controlDict;
}

solver incompressibleFluid;
startFrom startTime;
startTime 0;
stopAt endTime;
endTime $($Case.fixedIterations);
deltaT 1;
writeControl timeStep;
writeInterval $writeInterval;
writeFormat ascii;
writePrecision 10;
runTimeModifiable false;
"@

    return [pscustomobject][ordered]@{
        root = $Destination
        pressureConversion = $pressureConversion
    }
}

function Convert-FerrumHistory($Report) {
    return @($Report.history | ForEach-Object {
        $momentumInitial = [double](($_.momentumComponentInitialResiduals | Measure-Object -Maximum).Maximum)
        $momentumFinal = [double](($_.momentumComponentNormalizedResidualNorms | Measure-Object -Maximum).Maximum)
        [pscustomobject][ordered]@{
            iteration = [int]$_.iteration
            momentumInitialResidual = $momentumInitial
            momentumFinalResidual = $momentumFinal
            momentumLinearIterations = [int]$_.momentumLinearIterations
            momentumLinearSolves = @($_.momentumComponentInitialResiduals).Count
            pressureInitialResidual = [double]$_.pressureCorrectionInitialResidual
            pressureFinalResidual = [double]$_.pressureCorrectionNormalizedResidualNorm
            pressureLinearIterations = [int]$_.pressureLinearIterations
            pressureLinearSolves = [int]$_.pressureLinearSolves
            continuityIndicator = [double]$_.continuityAfter.l2Norm
        }
    })
}

function Complete-OpenFoamStep($Step) {
    if ($Step.momentumInitial.Count -eq 0 -or $Step.pressureInitial.Count -eq 0) {
        throw "OpenFOAM log step $($Step.iteration) does not contain both momentum and pressure solves"
    }
    return [pscustomobject][ordered]@{
        iteration = [int]$Step.iteration
        momentumInitialResidual = [double](($Step.momentumInitial | Measure-Object -Maximum).Maximum)
        momentumFinalResidual = [double](($Step.momentumFinal | Measure-Object -Maximum).Maximum)
        momentumLinearIterations = [int](($Step.momentumIterations | Measure-Object -Sum).Sum)
        momentumLinearSolves = $Step.momentumInitial.Count
        pressureInitialResidual = [double]$Step.pressureInitial[0]
        pressureFinalResidual = [double]$Step.pressureFinal[$Step.pressureFinal.Count - 1]
        pressureLinearIterations = [int](($Step.pressureIterations | Measure-Object -Sum).Sum)
        pressureLinearSolves = $Step.pressureInitial.Count
        continuityIndicator = if ($null -ne $Step.continuitySumLocal) { [double]$Step.continuitySumLocal } else { $null }
    }
}

function Read-OpenFoamLog([string]$LogPath) {
    $history = @()
    $current = $null
    $executionTime = $null
    $clockTime = $null
    foreach ($line in Get-Content -LiteralPath $LogPath) {
        $timeMatch = [regex]::Match($line, "^Time\s*=\s*([-+0-9.eE]+)s?\s*$")
        if ($timeMatch.Success) {
            if ($null -ne $current) {
                $history += Complete-OpenFoamStep $current
            }
            $timeValue = [double]::Parse($timeMatch.Groups[1].Value, [System.Globalization.CultureInfo]::InvariantCulture)
            $current = [ordered]@{
                iteration = [int][Math]::Round($timeValue)
                momentumInitial = @()
                momentumFinal = @()
                momentumIterations = @()
                pressureInitial = @()
                pressureFinal = @()
                pressureIterations = @()
                continuitySumLocal = $null
            }
            continue
        }

        if ($null -ne $current) {
            $solverMatch = [regex]::Match(
                $line,
                "^[^:]+:\s+Solving for (Ux|Uy|Uz|p),\s+Initial residual = ([-+0-9.eE]+),\s+Final residual = ([-+0-9.eE]+),\s+No Iterations (\d+)"
            )
            if ($solverMatch.Success) {
                $field = $solverMatch.Groups[1].Value
                $initial = [double]::Parse($solverMatch.Groups[2].Value, [System.Globalization.CultureInfo]::InvariantCulture)
                $final = [double]::Parse($solverMatch.Groups[3].Value, [System.Globalization.CultureInfo]::InvariantCulture)
                $iterations = [int]::Parse($solverMatch.Groups[4].Value, [System.Globalization.CultureInfo]::InvariantCulture)
                if ($field -eq "p") {
                    $current.pressureInitial += $initial
                    $current.pressureFinal += $final
                    $current.pressureIterations += $iterations
                } else {
                    $current.momentumInitial += $initial
                    $current.momentumFinal += $final
                    $current.momentumIterations += $iterations
                }
                continue
            }

            $continuityMatch = [regex]::Match($line, "time step continuity errors\s*:\s*sum local = ([-+0-9.eE]+)")
            if ($continuityMatch.Success) {
                $current.continuitySumLocal = [double]::Parse($continuityMatch.Groups[1].Value, [System.Globalization.CultureInfo]::InvariantCulture)
            }
        }

        $timingMatch = [regex]::Match($line, "ExecutionTime\s*=\s*([-+0-9.eE]+)\s*s\s+ClockTime\s*=\s*([-+0-9.eE]+)\s*s")
        if ($timingMatch.Success) {
            $executionTime = [double]::Parse($timingMatch.Groups[1].Value, [System.Globalization.CultureInfo]::InvariantCulture)
            $clockTime = [double]::Parse($timingMatch.Groups[2].Value, [System.Globalization.CultureInfo]::InvariantCulture)
        }
    }
    if ($null -ne $current) {
        $history += Complete-OpenFoamStep $current
    }
    return [pscustomobject][ordered]@{
        executionTimeSeconds = $executionTime
        clockTimeSeconds = $clockTime
        history = $history
    }
}

function Invoke-FerrumRun(
    $Case,
    [string]$Executable,
    [string]$Kind,
    [int]$Ordinal,
    [string]$RunRoot,
    $CanonicalMeshHashes
) {
    $RunRoot = [System.IO.Path]::GetFullPath($RunRoot)
    $workingCase = Join-Path $RunRoot "case"
    New-FerrumWorkingCase $Case $workingCase | Out-Null
    Assert-HashesEqual $CanonicalMeshHashes (Get-PolyMeshHashes $workingCase) "$($Case.name) Ferrum working"
    $reportPath = Join-Path $RunRoot "solve-report.json"
    $logPath = Join-Path $RunRoot "ferrum.log"
    $arguments = @(
        "-solver", "incompressibleFluid",
        "-case", $workingCase,
        "--minSimpleIterations", $Case.fixedIterations.ToString([System.Globalization.CultureInfo]::InvariantCulture),
        "--maxSimpleIterations", $Case.fixedIterations.ToString([System.Globalization.CultureInfo]::InvariantCulture),
        "--solveReportJson", $reportPath
    )
    $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
    $previousErrorActionPreference = $ErrorActionPreference
    Push-Location $RunRoot
    try {
        $ErrorActionPreference = "Continue"
        & $Executable @arguments *> $logPath
        $exitCode = $LASTEXITCODE
    } finally {
        Pop-Location
        $ErrorActionPreference = $previousErrorActionPreference
        $stopwatch.Stop()
    }
    if ($exitCode -ne 0) {
        throw "Ferrum run failed for $($Case.name) with exit code $exitCode. See $logPath"
    }
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
    if ($null -ne $report.timing.pressureGamgProfile) {
        throw "matched timing run must not enable GAMG profiling"
    }
    $history = @(Convert-FerrumHistory $report)
    if ($history.Count -ne $Case.fixedIterations) {
        throw "Ferrum completed $($history.Count) SIMPLE steps, expected $($Case.fixedIterations)"
    }
    $outputDirectories = Get-NumericOutputDirectoryCount $workingCase
    if ($outputDirectories -ne 0) {
        throw "Ferrum timing run unexpectedly wrote $outputDirectories time directories"
    }
    return [pscustomobject][ordered]@{
        engine = "FerrumCFD"
        kind = $Kind
        ordinal = $Ordinal
        commandWallClockSeconds = $stopwatch.Elapsed.TotalSeconds
        internalExecutionSeconds = [double]$report.timing.solverTotalSeconds
        simpleIterations = $history.Count
        pressureLinearIterations = [int]$report.solve.pressureLinearIterations
        momentumLinearIterations = [int]$report.solve.momentumLinearIterations
        converged = [bool]$report.solve.converged
        stopReason = [string]$report.solve.stopReason
        outputTimeDirectories = $outputDirectories
        report = $reportPath
        log = $logPath
        history = $history
    }
}

function Invoke-OpenFoamRun(
    $Case,
    [string]$SelectedMode,
    [string]$Kind,
    [int]$Ordinal,
    [string]$RunRoot,
    $CanonicalMeshHashes
) {
    $RunRoot = [System.IO.Path]::GetFullPath($RunRoot)
    $working = New-OpenFoamWorkingCase $Case (Join-Path $RunRoot "case") $CanonicalMeshHashes
    $workingCase = $working.root
    $logPath = Join-Path $RunRoot "openfoam.log"
    $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
    if ($SelectedMode -eq "Native") {
        Push-Location $workingCase
        try {
            $previousErrorActionPreference = $ErrorActionPreference
            try {
                $ErrorActionPreference = "Continue"
                & foamRun -solver incompressibleFluid *> $logPath
                $exitCode = $LASTEXITCODE
            } finally {
                $ErrorActionPreference = $previousErrorActionPreference
            }
        } finally {
            Pop-Location
            $stopwatch.Stop()
        }
    } else {
        try {
            $wslCase = ConvertTo-WslPath $workingCase
            $wslLog = ConvertTo-WslPath $logPath
            $quotedCase = ConvertTo-BashSingleQuoted $wslCase
            $quotedLog = ConvertTo-BashSingleQuoted $wslLog
            $bash = "source /opt/openfoam13/etc/bashrc 2>/dev/null && env | grep -q '^WM_PROJECT_VERSION=13$' && cd -- $quotedCase && foamRun -solver incompressibleFluid > $quotedLog 2>&1"
            & wsl bash -lc $bash
            $exitCode = $LASTEXITCODE
        } finally {
            $stopwatch.Stop()
        }
    }
    if ($exitCode -ne 0) {
        throw "OpenFOAM run failed for $($Case.name) with exit code $exitCode. See $logPath"
    }
    $parsed = Read-OpenFoamLog $logPath
    $history = @($parsed.history)
    if ($history.Count -ne $Case.fixedIterations) {
        throw "OpenFOAM completed $($history.Count) SIMPLE steps, expected $($Case.fixedIterations)"
    }
    if ($null -eq $parsed.executionTimeSeconds) {
        throw "OpenFOAM log did not contain ExecutionTime: $logPath"
    }
    $outputDirectories = Get-NumericOutputDirectoryCount $workingCase
    if ($outputDirectories -ne 0) {
        throw "OpenFOAM timing run unexpectedly wrote $outputDirectories time directories"
    }
    return [pscustomobject][ordered]@{
        engine = "OpenFOAM"
        kind = $Kind
        ordinal = $Ordinal
        commandWallClockSeconds = $stopwatch.Elapsed.TotalSeconds
        internalExecutionSeconds = [double]$parsed.executionTimeSeconds
        openFoamClockSeconds = [double]$parsed.clockTimeSeconds
        simpleIterations = $history.Count
        pressureLinearIterations = [int](($history.pressureLinearIterations | Measure-Object -Sum).Sum)
        momentumLinearIterations = [int](($history.momentumLinearIterations | Measure-Object -Sum).Sum)
        converged = $null -ne (Select-String -LiteralPath $logPath -Pattern "SIMPLE solution converged" | Select-Object -First 1)
        stopReason = "FixedIterationBudgetReached"
        outputTimeDirectories = $outputDirectories
        pressureConversion = $working.pressureConversion
        log = $logPath
        history = $history
    }
}

function Get-HistoryMedians($Runs) {
    if ($Runs.Count -eq 0) {
        return @()
    }
    $count = @($Runs[0].history).Count
    foreach ($run in $Runs) {
        if (@($run.history).Count -ne $count) {
            throw "history length changed between measured runs"
        }
    }
    $result = @()
    for ($index = 0; $index -lt $count; $index++) {
        $result += [pscustomobject][ordered]@{
            iteration = [int]$Runs[0].history[$index].iteration
            momentumInitialResidual = Get-Median ([double[]]@($Runs | ForEach-Object { [double]$_.history[$index].momentumInitialResidual }))
            momentumFinalResidual = Get-Median ([double[]]@($Runs | ForEach-Object { [double]$_.history[$index].momentumFinalResidual }))
            momentumLinearIterations = Get-Median ([double[]]@($Runs | ForEach-Object { [double]$_.history[$index].momentumLinearIterations }))
            momentumLinearSolves = Get-Median ([double[]]@($Runs | ForEach-Object { [double]$_.history[$index].momentumLinearSolves }))
            pressureInitialResidual = Get-Median ([double[]]@($Runs | ForEach-Object { [double]$_.history[$index].pressureInitialResidual }))
            pressureFinalResidual = Get-Median ([double[]]@($Runs | ForEach-Object { [double]$_.history[$index].pressureFinalResidual }))
            pressureLinearIterations = Get-Median ([double[]]@($Runs | ForEach-Object { [double]$_.history[$index].pressureLinearIterations }))
            pressureLinearSolves = Get-Median ([double[]]@($Runs | ForEach-Object { [double]$_.history[$index].pressureLinearSolves }))
            continuityIndicator = Get-Median ([double[]]@($Runs | ForEach-Object { [double]$_.history[$index].continuityIndicator }))
        }
    }
    return $result
}

function Get-EngineSummary([string]$Name, $Runs) {
    $measured = @($Runs | Where-Object { $_.kind -eq "measured" })
    $historyMedians = @(Get-HistoryMedians $measured)
    $pressureSolves = [double[]]@($measured | ForEach-Object {
        [double](($_.history.pressureLinearSolves | Measure-Object -Sum).Sum)
    })
    $pressureIterationsPerSolve = [double[]]@()
    for ($index = 0; $index -lt $measured.Count; $index++) {
        $pressureIterationsPerSolve += [double]$measured[$index].pressureLinearIterations / $pressureSolves[$index]
    }
    return [pscustomobject][ordered]@{
        engine = $Name
        medians = [pscustomobject][ordered]@{
            internalExecutionSeconds = Get-Median ([double[]]@($measured | ForEach-Object { [double]$_.internalExecutionSeconds }))
            commandWallClockSeconds = Get-Median ([double[]]@($measured | ForEach-Object { [double]$_.commandWallClockSeconds }))
            pressureLinearIterations = Get-Median ([double[]]@($measured | ForEach-Object { [double]$_.pressureLinearIterations }))
            pressureLinearSolves = Get-Median $pressureSolves
            pressureLinearIterationsPerSolve = Get-Median $pressureIterationsPerSolve
            momentumLinearIterations = Get-Median ([double[]]@($measured | ForEach-Object { [double]$_.momentumLinearIterations }))
        }
        historyMedians = $historyMedians
        runs = $Runs
    }
}

function Write-ResidualCsv([string]$Path, $FerrumHistory, $OpenFoamHistory) {
    if ($null -eq $OpenFoamHistory -or @($OpenFoamHistory).Count -eq 0) {
        @($FerrumHistory | ForEach-Object {
            [pscustomobject][ordered]@{
                iteration = $_.iteration
                ferrumPressureInitialResidual = $_.pressureInitialResidual
                ferrumPressureFinalResidual = $_.pressureFinalResidual
                ferrumPressureLinearIterations = $_.pressureLinearIterations
                ferrumMomentumInitialResidual = $_.momentumInitialResidual
                ferrumMomentumFinalResidual = $_.momentumFinalResidual
                ferrumMomentumLinearIterations = $_.momentumLinearIterations
            }
        }) | Export-Csv -LiteralPath $Path -NoTypeInformation -Encoding UTF8
        return
    }
    if (@($FerrumHistory).Count -ne @($OpenFoamHistory).Count) {
        throw "cannot write matched residual CSV with unequal history lengths"
    }
    $rows = @()
    for ($index = 0; $index -lt @($FerrumHistory).Count; $index++) {
        $ferrum = $FerrumHistory[$index]
        $openFoam = $OpenFoamHistory[$index]
        $rows += [pscustomobject][ordered]@{
            iteration = $ferrum.iteration
            ferrumPressureInitialResidual = $ferrum.pressureInitialResidual
            openFoamPressureInitialResidual = $openFoam.pressureInitialResidual
            ferrumPressureFinalResidual = $ferrum.pressureFinalResidual
            openFoamPressureFinalResidual = $openFoam.pressureFinalResidual
            ferrumPressureLinearIterations = $ferrum.pressureLinearIterations
            openFoamPressureLinearIterations = $openFoam.pressureLinearIterations
            ferrumMomentumInitialResidual = $ferrum.momentumInitialResidual
            openFoamMomentumInitialResidual = $openFoam.momentumInitialResidual
            ferrumMomentumFinalResidual = $ferrum.momentumFinalResidual
            openFoamMomentumFinalResidual = $openFoam.momentumFinalResidual
            ferrumMomentumLinearIterations = $ferrum.momentumLinearIterations
            openFoamMomentumLinearIterations = $openFoam.momentumLinearIterations
        }
    }
    $rows | Export-Csv -LiteralPath $Path -NoTypeInformation -Encoding UTF8
}

Reset-TargetDirectory $OutRoot
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
    throw "Ferrum release executable was not found: $executable"
}

$selectedOpenFoamMode = Get-OpenFoamMode
if ($null -eq $selectedOpenFoamMode -and $RequireOpenFoam) {
    throw "OpenFOAM Foundation 13 foamRun was not found"
}

$caseResults = @()
foreach ($case in $caseDefinitions) {
    $caseOut = Join-Path $OutRoot $case.name
    New-Item -ItemType Directory -Force -Path $caseOut | Out-Null
    $canonicalMeshHashes = Get-PolyMeshHashes $case.ferrumCase
    $ferrumRuns = @()
    $openFoamRuns = @()
    $totalRuns = $WarmupRuns + $MeasuredRuns
    for ($runIndex = 1; $runIndex -le $totalRuns; $runIndex++) {
        $kind = if ($runIndex -le $WarmupRuns) { "warmup" } else { "measured" }
        $ordinal = if ($kind -eq "warmup") { $runIndex } else { $runIndex - $WarmupRuns }
        $engines = if (($runIndex % 2) -eq 1) { @("ferrum", "openfoam") } else { @("openfoam", "ferrum") }
        if ($null -eq $selectedOpenFoamMode) {
            $engines = @("ferrum")
        }
        foreach ($engine in $engines) {
            $runRoot = Join-Path $caseOut "$kind-$ordinal-$engine"
            New-Item -ItemType Directory -Force -Path $runRoot | Out-Null
            if ($engine -eq "ferrum") {
                $run = Invoke-FerrumRun $case $executable $kind $ordinal $runRoot $canonicalMeshHashes
                $ferrumRuns += $run
            } else {
                $run = Invoke-OpenFoamRun $case $selectedOpenFoamMode $kind $ordinal $runRoot $canonicalMeshHashes
                $openFoamRuns += $run
            }
            Write-Host ("{0} {1} {2} {3}: internal={4:F6}s wall={5:F6}s pIters={6}" -f $case.name, $kind, $ordinal, $run.engine, $run.internalExecutionSeconds, $run.commandWallClockSeconds, $run.pressureLinearIterations)
        }
    }

    $ferrumSummary = Get-EngineSummary "FerrumCFD" $ferrumRuns
    $openFoamSummary = if ($openFoamRuns.Count -gt 0) { Get-EngineSummary "OpenFOAM" $openFoamRuns } else { $null }
    $residualCsv = Join-Path $caseOut "residual-history-medians.csv"
    Write-ResidualCsv $residualCsv $ferrumSummary.historyMedians $(if ($null -ne $openFoamSummary) { $openFoamSummary.historyMedians } else { $null })
    $speedRatio = if ($null -ne $openFoamSummary) {
        $ferrumSummary.medians.internalExecutionSeconds / $openFoamSummary.medians.internalExecutionSeconds
    } else {
        $null
    }
    $pressureWorkRatio = if ($null -ne $openFoamSummary -and $openFoamSummary.medians.pressureLinearIterationsPerSolve -ne 0.0) {
        $ferrumSummary.medians.pressureLinearIterationsPerSolve / $openFoamSummary.medians.pressureLinearIterationsPerSolve
    } else {
        $null
    }
    $caseResults += [pscustomobject][ordered]@{
        name = $case.name
        fixedSimpleIterations = $case.fixedIterations
        canonicalFerrumCase = $case.ferrumCase
        canonicalPolyMeshSha256 = $canonicalMeshHashes
        sharedNumerics = [pscustomobject][ordered]@{
            fvSchemes = Join-Path $case.ferrumCase "system\fvSchemes"
            fvSolution = $matchedFvSolution
            pressureSolver = $PressureSolver
        }
        residualCsv = $residualCsv
        ferrum = $ferrumSummary
        openFoam = $openFoamSummary
        comparison = [pscustomobject][ordered]@{
            ferrumOverOpenFoamInternalTimeRatio = $speedRatio
            ferrumSlowerPercent = if ($null -ne $speedRatio) { 100.0 * ($speedRatio - 1.0) } else { $null }
            ferrumOverOpenFoamPressureIterationsPerSolveRatio = $pressureWorkRatio
        }
    }
}

$summary = [pscustomobject][ordered]@{
    schemaVersion = 1
    benchmark = "matched-serial-cpu-solver"
    generatedAtUtc = [DateTime]::UtcNow.ToString("o")
    pressureSolver = $PressureSolver
    openFoam = [pscustomobject][ordered]@{
        requestedMode = $OpenFoamMode
        selectedMode = $selectedOpenFoamMode
        foundationVersion = 13
        available = $null -ne $selectedOpenFoamMode
    }
    build = [pscustomobject][ordered]@{
        command = "cargo build --locked --release -p ferrum-run --bin ferrumRun"
        wallClockSeconds = $buildStopwatch.Elapsed.TotalSeconds
        excludedFromRunTiming = $true
        executable = $executable
        log = $buildLog
    }
    policy = [pscustomobject][ordered]@{
        serialCpuOnly = $true
        warmupRuns = $WarmupRuns
        measuredRuns = $MeasuredRuns
        medianReported = $true
        alternatingEngineOrder = $true
        identicalSimpleIterationBudget = $true
        canonicalMeshIsFerrumPolyMesh = $true
        polyMeshByteHashesVerified = $true
        velocityFieldBytesShared = $true
        pressureConvertedOnlyForOpenFoamKinematicUnits = $true
        fvSchemesBytesShared = $true
        fvSolutionBytesShared = $true
        residualControlDisabledForFixedWork = $true
        solutionFieldOutputDisabled = $true
        gamgProfilingDisabled = $true
        benchmarkCriteriaExternalToSolver = $true
        wslProcessWallIncludesLaunchOverhead = $selectedOpenFoamMode -eq "Wsl"
    }
    residualDefinitions = [pscustomobject][ordered]@{
        ferrum = "normalized linear-system residuals from the Ferrum solve report; continuityIndicator is post-correction L2"
        openFoam = "Initial/Final residual printed by OpenFOAM linear solvers; continuityIndicator is sum local"
        comparability = "linear residual trends and iteration counts are compared; continuity indicators use different definitions and are not directly ranked"
    }
    cases = $caseResults
}

$jsonPath = Join-Path $OutRoot "summary.json"
$markdownPath = Join-Path $OutRoot "summary.md"
$summary | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $jsonPath -Encoding UTF8

$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# Matched Serial CPU Solver Benchmark")
$lines.Add("")
$lines.Add("Pressure solver: ``$PressureSolver``")
$lines.Add("Warm-up/measured runs: ``$WarmupRuns/$MeasuredRuns``")
$lines.Add("OpenFOAM mode: ``$(if ($null -ne $selectedOpenFoamMode) { $selectedOpenFoamMode } else { 'unavailable' })``")
$lines.Add("")
$lines.Add("Compilation and GAMG profiling are excluded. Both engines receive the same byte-verified mesh, velocity field, schemes, solver controls, fixed SIMPLE budget, and no solution-field writes. Ferrum pressure is converted from Pa to OpenFOAM kinematic pressure with the case density.")
$lines.Add("")
$lines.Add("| Case | SIMPLE | Ferrum internal [s] | OpenFOAM ExecutionTime [s] | Ferrum / OpenFOAM | Ferrum wall [s] | OpenFOAM wall [s] |")
$lines.Add("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
foreach ($case in $caseResults) {
    $openFoamInternal = if ($null -ne $case.openFoam) { $case.openFoam.medians.internalExecutionSeconds } else { $null }
    $openFoamWall = if ($null -ne $case.openFoam) { $case.openFoam.medians.commandWallClockSeconds } else { $null }
    $lines.Add(("| {0} | {1} | {2} | {3} | {4} | {5} | {6} |" -f $case.name, $case.fixedSimpleIterations, (Format-ReportNumber $case.ferrum.medians.internalExecutionSeconds), (Format-ReportNumber $openFoamInternal), (Format-ReportNumber $case.comparison.ferrumOverOpenFoamInternalTimeRatio), (Format-ReportNumber $case.ferrum.medians.commandWallClockSeconds), (Format-ReportNumber $openFoamWall)))
}
$lines.Add("")
$lines.Add("## Linear Work")
$lines.Add("")
$lines.Add("| Case | Ferrum p iterations | OpenFOAM p iterations | Ferrum p iterations/solve | OpenFOAM p iterations/solve | Work ratio | Ferrum U iterations | OpenFOAM U iterations |")
$lines.Add("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
foreach ($case in $caseResults) {
    $openFoamPIterations = if ($null -ne $case.openFoam) { $case.openFoam.medians.pressureLinearIterations } else { $null }
    $openFoamPerSolve = if ($null -ne $case.openFoam) { $case.openFoam.medians.pressureLinearIterationsPerSolve } else { $null }
    $openFoamUIterations = if ($null -ne $case.openFoam) { $case.openFoam.medians.momentumLinearIterations } else { $null }
    $lines.Add(("| {0} | {1} | {2} | {3} | {4} | {5} | {6} | {7} |" -f $case.name, (Format-ReportNumber $case.ferrum.medians.pressureLinearIterations), (Format-ReportNumber $openFoamPIterations), (Format-ReportNumber $case.ferrum.medians.pressureLinearIterationsPerSolve), (Format-ReportNumber $openFoamPerSolve), (Format-ReportNumber $case.comparison.ferrumOverOpenFoamPressureIterationsPerSolveRatio), (Format-ReportNumber $case.ferrum.medians.momentumLinearIterations), (Format-ReportNumber $openFoamUIterations)))
}
$lines.Add("")
$lines.Add("## Residual Evidence")
$lines.Add("")
$lines.Add("| Case | Engine | First p initial | Last p initial | Last p final | Residual CSV |")
$lines.Add("| --- | --- | ---: | ---: | ---: | --- |")
foreach ($case in $caseResults) {
    foreach ($engine in @($case.ferrum, $case.openFoam)) {
        if ($null -eq $engine) { continue }
        $history = @($engine.historyMedians)
        $lines.Add(("| {0} | {1} | {2} | {3} | {4} | ``{5}`` |" -f $case.name, $engine.engine, (Format-ReportNumber $history[0].pressureInitialResidual), (Format-ReportNumber $history[$history.Count - 1].pressureInitialResidual), (Format-ReportNumber $history[$history.Count - 1].pressureFinalResidual), $case.residualCsv))
    }
}
$lines.Add("")
$lines.Add("Ferrum and OpenFOAM continuity columns intentionally remain separate because their printed definitions differ. If OpenFOAM runs through WSL, command-wall time includes WSL process launch; the internal-time column is the primary comparison.")
Set-Content -LiteralPath $markdownPath -Value $lines -Encoding UTF8

Write-Output "wrote matched CPU benchmark JSON: $jsonPath"
Write-Output "wrote matched CPU benchmark Markdown: $markdownPath"
