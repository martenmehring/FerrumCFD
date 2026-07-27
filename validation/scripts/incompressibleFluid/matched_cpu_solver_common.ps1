$ErrorActionPreference = "Stop"

function Get-MatchedCpuCaseDefinitions(
    [string]$RepoRoot,
    [string]$PressureSolver,
    [string]$CaseName = "all"
) {
    $matchedFvSolution = Join-Path $RepoRoot "validation\profiles\incompressibleFluid\matched-fixed\$PressureSolver\system\fvSolution"
    if (!(Test-Path -LiteralPath $matchedFvSolution -PathType Leaf)) {
        throw "matched fixed-work fvSolution was not found: $matchedFvSolution"
    }
    $cases = @(
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
        $cases = @($cases | Where-Object { $_.name -eq $CaseName })
    }
    foreach ($case in $cases) {
        foreach ($path in @($case.ferrumCase, $case.openFoamTemplate)) {
            if (!(Test-Path -LiteralPath $path -PathType Container)) {
                throw "matched benchmark case was not found: $path"
            }
        }
    }
    return [pscustomobject][ordered]@{
        fvSolution = $matchedFvSolution
        cases = $cases
    }
}

function Format-MatchedF64([double]$Value) {
    return $Value.ToString("G17", [System.Globalization.CultureInfo]::InvariantCulture)
}

function Format-MatchedReportNumber($Value) {
    if ($null -eq $Value) { return "n/a" }
    return ([double]$Value).ToString("G8", [System.Globalization.CultureInfo]::InvariantCulture)
}

function Get-MatchedMedian([double[]]$Values) {
    if ($Values.Count -eq 0) { return $null }
    $sorted = @($Values | Sort-Object)
    $middle = [int][Math]::Floor($sorted.Count / 2)
    if (($sorted.Count % 2) -eq 1) { return [double]$sorted[$middle] }
    return ([double]$sorted[$middle - 1] + [double]$sorted[$middle]) / 2.0
}

function Get-MatchedMedianAbsoluteDeviation([double[]]$Values) {
    if ($Values.Count -eq 0) { return $null }
    $median = Get-MatchedMedian $Values
    return Get-MatchedMedian ([double[]]@($Values | ForEach-Object { [Math]::Abs([double]$_ - $median) }))
}

function Test-MatchedPathUnder([string]$Child, [string]$Parent) {
    $childFull = [System.IO.Path]::GetFullPath($Child)
    $parentFull = [System.IO.Path]::GetFullPath($Parent).TrimEnd(
        [System.IO.Path]::DirectorySeparatorChar,
        [System.IO.Path]::AltDirectorySeparatorChar
    )
    return $childFull.Equals($parentFull, [System.StringComparison]::OrdinalIgnoreCase) -or
        $childFull.StartsWith($parentFull + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase) -or
        $childFull.StartsWith($parentFull + [System.IO.Path]::AltDirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)
}

function Assert-MatchedNoReparsePath([string]$Path, [string]$AllowedRoot) {
    $pathFull = [System.IO.Path]::GetFullPath($Path)
    $rootFull = [System.IO.Path]::GetFullPath($AllowedRoot).TrimEnd(
        [System.IO.Path]::DirectorySeparatorChar,
        [System.IO.Path]::AltDirectorySeparatorChar
    )
    if (!(Test-MatchedPathUnder $pathFull $rootFull)) {
        throw "refusing to inspect path outside allowed root: $pathFull"
    }
    if (!(Test-Path -LiteralPath $rootFull -PathType Container)) {
        throw "allowed root does not exist or is not a directory: $rootFull"
    }

    $cursor = $pathFull
    while (!(Test-Path -LiteralPath $cursor)) {
        if ($cursor.Equals($rootFull, [System.StringComparison]::OrdinalIgnoreCase)) {
            break
        }
        $parent = Split-Path -Parent $cursor
        if ([string]::IsNullOrWhiteSpace($parent) -or !(Test-MatchedPathUnder $parent $rootFull)) {
            throw "could not reach allowed root while inspecting path: $pathFull"
        }
        $cursor = [System.IO.Path]::GetFullPath($parent)
    }

    while ($true) {
        $item = Get-Item -LiteralPath $cursor -Force
        if (($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw "refusing path with a reparse-point component: $($item.FullName)"
        }
        if ($cursor.Equals($rootFull, [System.StringComparison]::OrdinalIgnoreCase)) {
            break
        }
        $parent = Split-Path -Parent $cursor
        if ([string]::IsNullOrWhiteSpace($parent) -or !(Test-MatchedPathUnder $parent $rootFull)) {
            throw "could not reach allowed root while inspecting path: $pathFull"
        }
        $cursor = [System.IO.Path]::GetFullPath($parent)
    }
}

function Reset-MatchedTargetDirectory([string]$Path, [string]$AllowedRoot) {
    if (!(Test-MatchedPathUnder $Path $AllowedRoot)) {
        throw "refusing to replace directory outside allowed root: $Path"
    }
    if (Test-Path -LiteralPath $AllowedRoot -PathType Container) {
        Assert-MatchedNoReparsePath $Path $AllowedRoot
    }
    if (Test-Path -LiteralPath $Path) {
        Remove-Item -LiteralPath $Path -Recurse -Force
    }
    New-Item -ItemType Directory -Force -Path $Path | Out-Null
    Assert-MatchedNoReparsePath $Path $AllowedRoot
}

function Assert-MatchedSafeTarArchive([string]$ArchivePath, [string]$Description) {
    if (!(Test-Path -LiteralPath $ArchivePath -PathType Leaf)) {
        throw "$Description archive was not found: $ArchivePath"
    }
    $entries = @(& tar -tf $ArchivePath 2>&1)
    $entriesExitCode = $LASTEXITCODE
    if ($entriesExitCode -ne 0 -or $entries.Count -eq 0) {
        throw "$Description archive could not be listed"
    }
    $details = @(& tar -tvf $ArchivePath 2>&1)
    $detailsExitCode = $LASTEXITCODE
    if ($detailsExitCode -ne 0 -or $details.Count -ne $entries.Count) {
        throw "$Description archive type listing did not match its path listing"
    }
    for ($index = 0; $index -lt $entries.Count; $index++) {
        $entry = [string]$entries[$index]
        $normalized = $entry.Replace("\", "/")
        if ([string]::IsNullOrWhiteSpace($normalized) -or
            $normalized.StartsWith("/") -or
            $normalized -match "^[A-Za-z]:(/|$)" -or
            $normalized -match "(^|/)\.\.(/|$)") {
            throw "$Description archive contains an unsafe path: $entry"
        }
        $detail = [string]$details[$index]
        if ([string]::IsNullOrEmpty($detail) -or @("-", "d") -notcontains $detail.Substring(0, 1)) {
            throw "$Description archive contains a non-regular entry: $entry"
        }
    }
}

function Write-MatchedAsciiFile([string]$Path, [string]$Content) {
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Path) | Out-Null
    Set-Content -LiteralPath $Path -Value $Content -Encoding ASCII
}

function ConvertTo-MatchedWslPath([string]$Path, [string]$Distro) {
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
        return "/mnt/$drive/$($Matches[2].Replace('\', '/'))"
    }
    $converted = & wsl -d $Distro -- wslpath -a -u $resolved
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($converted)) {
        throw "could not convert '$resolved' to a WSL path"
    }
    return $converted.Trim()
}

function Read-MatchedDimensionedScalar([string]$Path, [string]$Name) {
    $content = Get-Content -LiteralPath $Path -Raw
    $match = [regex]::Match($content, "(?m)^\s*$([regex]::Escape($Name))\s+\[[^\]]+\]\s+([-+0-9.eE]+)\s*;")
    if (!$match.Success) { throw "could not read dimensioned scalar '$Name' from $Path" }
    return [double]::Parse($match.Groups[1].Value, [System.Globalization.CultureInfo]::InvariantCulture)
}

function Read-MatchedInternalScalarField([string]$Path) {
    $content = Get-Content -LiteralPath $Path -Raw
    $uniform = [regex]::Match($content, "(?ms)\binternalField\s+uniform\s+([-+0-9.eE]+)\s*;")
    if ($uniform.Success) {
        return ,([double[]]@([double]::Parse($uniform.Groups[1].Value, [System.Globalization.CultureInfo]::InvariantCulture)))
    }
    $nonuniform = [regex]::Match($content, "(?ms)\binternalField\s+nonuniform\s+List<scalar>\s+(\d+)\s*\((.*?)\)\s*;")
    if (!$nonuniform.Success) { throw "unsupported scalar internalField in $Path" }
    $expected = [int]::Parse($nonuniform.Groups[1].Value, [System.Globalization.CultureInfo]::InvariantCulture)
    $tokens = @($nonuniform.Groups[2].Value -split "\s+" | Where-Object { ![string]::IsNullOrWhiteSpace($_) })
    if ($tokens.Count -ne $expected) {
        throw "scalar internalField in $Path declares $expected values but contains $($tokens.Count)"
    }
    return ,([double[]]@($tokens | ForEach-Object {
        [double]::Parse($_, [System.Globalization.CultureInfo]::InvariantCulture)
    }))
}

function Get-MatchedUniformBoundaryValues([string]$Path) {
    $content = Get-Content -LiteralPath $Path -Raw
    return [double[]]@([regex]::Matches($content, "(?m)^\s*value\s+uniform\s+([-+0-9.eE]+)\s*;") | ForEach-Object {
        [double]::Parse($_.Groups[1].Value, [System.Globalization.CultureInfo]::InvariantCulture)
    })
}

function Convert-MatchedPressureFieldToKinematic(
    [string]$FerrumPressurePath,
    [string]$OpenFoamTemplatePath,
    [string]$DestinationPath,
    [double]$Rho
) {
    [double[]]$valuesPa = Read-MatchedInternalScalarField $FerrumPressurePath
    $valuesKinematic = [double[]]@($valuesPa | ForEach-Object { [double]$_ / $Rho })
    $internalField = if ($valuesKinematic.Count -eq 1) {
        "internalField uniform $(Format-MatchedF64 $valuesKinematic[0]);"
    } else {
        $lines = @($valuesKinematic | ForEach-Object { "    $(Format-MatchedF64 $_)" })
        "internalField nonuniform List<scalar>`n$($valuesKinematic.Count)`n(`n$($lines -join "`n")`n);"
    }
    $sourceBoundary = @(Get-MatchedUniformBoundaryValues $FerrumPressurePath)
    $templateBoundary = @(Get-MatchedUniformBoundaryValues $OpenFoamTemplatePath)
    if ($sourceBoundary.Count -ne $templateBoundary.Count) {
        throw "pressure boundary-value count differs between Ferrum and OpenFOAM templates"
    }
    for ($index = 0; $index -lt $sourceBoundary.Count; $index++) {
        $expected = $sourceBoundary[$index] / $Rho
        $tolerance = 1e-12 * [Math]::Max(1.0, [Math]::Abs($expected))
        if ([Math]::Abs($templateBoundary[$index] - $expected) -gt $tolerance) {
            throw "OpenFOAM pressure boundary value $index is not the Ferrum SI value divided by rho"
        }
    }
    $template = Get-Content -LiteralPath $OpenFoamTemplatePath -Raw
    $regex = [regex]::new("(?ms)\binternalField\s+(?:uniform\s+[-+0-9.eE]+\s*;|nonuniform\s+List<scalar>\s+\d+\s*\(.*?\)\s*;)")
    if ($regex.Matches($template).Count -ne 1) {
        throw "OpenFOAM template must contain exactly one supported internalField: $OpenFoamTemplatePath"
    }
    Write-MatchedAsciiFile $DestinationPath ($regex.Replace($template, $internalField, 1))
    return [pscustomobject][ordered]@{
        sourceUnits = "Pa"
        destinationUnits = "m2/s2"
        densityKgPerM3 = $Rho
        internalValues = $valuesKinematic.Count
        boundaryUniformValuesChecked = $sourceBoundary.Count
    }
}

function Get-MatchedPolyMeshHashes([string]$CaseRoot) {
    $hashes = [ordered]@{}
    foreach ($name in @("points", "faces", "owner", "neighbour", "boundary")) {
        $path = Join-Path $CaseRoot "constant\polyMesh\$name"
        if (!(Test-Path -LiteralPath $path -PathType Leaf)) { throw "polyMesh file was not found: $path" }
        $hashes[$name] = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
    }
    return [pscustomobject]$hashes
}

function Assert-MatchedHashesEqual($Expected, $Actual, [string]$Description) {
    foreach ($name in @("points", "faces", "owner", "neighbour", "boundary")) {
        if ($Expected.$name -ne $Actual.$name) { throw "$Description polyMesh differs in $name" }
    }
}

function Get-MatchedNumericOutputDirectoryCount([string]$CaseRoot) {
    return @(
        Get-ChildItem -LiteralPath $CaseRoot -Directory | Where-Object {
            if ($_.Name -eq "0") { return $false }
            $value = 0.0
            [double]::TryParse($_.Name, [System.Globalization.NumberStyles]::Float,
                [System.Globalization.CultureInfo]::InvariantCulture, [ref]$value)
        }
    ).Count
}

function New-MatchedFerrumWorkingCase($Case, [string]$Destination, [string]$MatchedFvSolution, [string]$AllowedRoot) {
    Reset-MatchedTargetDirectory $Destination $AllowedRoot
    Copy-Item -Path (Join-Path $Case.ferrumCase "*") -Destination $Destination -Recurse -Force
    Copy-Item -LiteralPath $MatchedFvSolution -Destination (Join-Path $Destination "system\fvSolution") -Force
    return $Destination
}

function New-MatchedOpenFoamWorkingCase(
    $Case,
    [string]$Destination,
    [string]$MatchedFvSolution,
    $CanonicalMeshHashes,
    [string]$AllowedRoot
) {
    Reset-MatchedTargetDirectory $Destination $AllowedRoot
    Copy-Item -Path (Join-Path $Case.openFoamTemplate "*") -Destination $Destination -Recurse -Force
    $destinationMesh = Join-Path $Destination "constant\polyMesh"
    if (Test-Path -LiteralPath $destinationMesh) { Remove-Item -LiteralPath $destinationMesh -Recurse -Force }
    Copy-Item -LiteralPath (Join-Path $Case.ferrumCase "constant\polyMesh") -Destination $destinationMesh -Recurse
    Assert-MatchedHashesEqual $CanonicalMeshHashes (Get-MatchedPolyMeshHashes $Destination) "$($Case.name) OpenFOAM working"
    Copy-Item -LiteralPath (Join-Path $Case.ferrumCase "0\U") -Destination (Join-Path $Destination "0\U") -Force
    Copy-Item -LiteralPath (Join-Path $Case.ferrumCase "system\fvSchemes") -Destination (Join-Path $Destination "system\fvSchemes") -Force
    Copy-Item -LiteralPath $MatchedFvSolution -Destination (Join-Path $Destination "system\fvSolution") -Force
    $transportPath = Join-Path $Case.ferrumCase "constant\transportProperties"
    $rho = Read-MatchedDimensionedScalar $transportPath "rho"
    $nu = Read-MatchedDimensionedScalar $transportPath "nu"
    if ($rho -le 0.0 -or $nu -le 0.0) { throw "rho and nu must be positive in $transportPath" }
    $pressureConversion = Convert-MatchedPressureFieldToKinematic `
        -FerrumPressurePath (Join-Path $Case.ferrumCase "0\p") `
        -OpenFoamTemplatePath (Join-Path $Case.openFoamTemplate "0\p") `
        -DestinationPath (Join-Path $Destination "0\p") `
        -Rho $rho
    Write-MatchedAsciiFile (Join-Path $Destination "constant\physicalProperties") @"
FoamFile
{
    version 2.0;
    format ascii;
    class dictionary;
    location "constant";
    object physicalProperties;
}

viscosityModel constant;
nu [0 2 -1 0 0 0 0] $(Format-MatchedF64 $nu);
"@
    $writeInterval = $Case.fixedIterations + 1
    Write-MatchedAsciiFile (Join-Path $Destination "system\controlDict") @"
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
    return $pressureConversion
}

function Convert-MatchedFerrumHistory($Report) {
    return @($Report.history | ForEach-Object {
        [pscustomobject][ordered]@{
            iteration = [int]$_.iteration
            momentumInitialResidual = [double](($_.momentumComponentInitialResiduals | Measure-Object -Maximum).Maximum)
            momentumFinalResidual = [double](($_.momentumComponentNormalizedResidualNorms | Measure-Object -Maximum).Maximum)
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

function Complete-MatchedOpenFoamStep($Step) {
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
        pressureLinearSolvers = @($Step.pressureSolvers)
        continuityIndicator = if ($null -ne $Step.continuitySumLocal) { [double]$Step.continuitySumLocal } else { $null }
    }
}

function Read-MatchedOpenFoamLog([string]$LogPath) {
    $history = @()
    $current = $null
    $executionTime = $null
    $clockTime = $null
    $sawEnd = $false
    foreach ($line in Get-Content -LiteralPath $LogPath) {
        if ($line.Trim() -eq "End") { $sawEnd = $true }
        $timeMatch = [regex]::Match($line, "^Time\s*=\s*([-+0-9.eE]+)s?\s*$")
        if ($timeMatch.Success) {
            if ($null -ne $current) { $history += Complete-MatchedOpenFoamStep $current }
            $timeValue = [double]::Parse($timeMatch.Groups[1].Value, [System.Globalization.CultureInfo]::InvariantCulture)
            $current = [ordered]@{
                iteration = [int][Math]::Round($timeValue)
                momentumInitial = @(); momentumFinal = @(); momentumIterations = @()
                pressureInitial = @(); pressureFinal = @(); pressureIterations = @(); pressureSolvers = @()
                continuitySumLocal = $null
            }
            continue
        }
        if ($null -ne $current) {
            $solverMatch = [regex]::Match($line,
                "^([^:]+):\s+Solving for (Ux|Uy|Uz|p),\s+Initial residual = ([-+0-9.eE]+),\s+Final residual = ([-+0-9.eE]+),\s+No Iterations (\d+)")
            if ($solverMatch.Success) {
                $solver = $solverMatch.Groups[1].Value.Trim()
                $field = $solverMatch.Groups[2].Value
                $initial = [double]::Parse($solverMatch.Groups[3].Value, [System.Globalization.CultureInfo]::InvariantCulture)
                $final = [double]::Parse($solverMatch.Groups[4].Value, [System.Globalization.CultureInfo]::InvariantCulture)
                $iterations = [int]::Parse($solverMatch.Groups[5].Value, [System.Globalization.CultureInfo]::InvariantCulture)
                if ($field -eq "p") {
                    $current.pressureInitial += $initial; $current.pressureFinal += $final
                    $current.pressureIterations += $iterations; $current.pressureSolvers += $solver
                } else {
                    $current.momentumInitial += $initial; $current.momentumFinal += $final; $current.momentumIterations += $iterations
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
    if ($null -ne $current) { $history += Complete-MatchedOpenFoamStep $current }
    return [pscustomobject][ordered]@{
        executionTimeSeconds = $executionTime
        clockTimeSeconds = $clockTime
        sawEnd = $sawEnd
        history = $history
    }
}

function Read-MatchedKeyValueFile([string]$Path) {
    if (!(Test-Path -LiteralPath $Path -PathType Leaf)) { throw "required key/value file was not found: $Path" }
    $values = [ordered]@{}
    foreach ($line in Get-Content -LiteralPath $Path) {
        if ([string]::IsNullOrWhiteSpace($line) -or $line.TrimStart().StartsWith("#")) { continue }
        $index = $line.IndexOf("=")
        if ($index -lt 1) { throw "invalid key/value line in ${Path}: $line" }
        $values[$line.Substring(0, $index)] = $line.Substring($index + 1)
    }
    return [pscustomobject]$values
}

function Read-MatchedGnuTime([string]$Path) {
    $record = Read-MatchedKeyValueFile $Path
    foreach ($name in @("elapsed_s", "user_s", "system_s", "max_rss_kb", "exit")) {
        if ($null -eq $record.$name) { throw "GNU time record is missing '$name': $Path" }
    }
    return [pscustomobject][ordered]@{
        elapsedSeconds = [double]::Parse($record.elapsed_s, [System.Globalization.CultureInfo]::InvariantCulture)
        userSeconds = [double]::Parse($record.user_s, [System.Globalization.CultureInfo]::InvariantCulture)
        systemSeconds = [double]::Parse($record.system_s, [System.Globalization.CultureInfo]::InvariantCulture)
        maxResidentSetKiB = [long]::Parse($record.max_rss_kb, [System.Globalization.CultureInfo]::InvariantCulture)
        exitCode = [int]::Parse($record.exit, [System.Globalization.CultureInfo]::InvariantCulture)
    }
}

function Get-MatchedHistoryMedians($Runs) {
    if ($Runs.Count -eq 0) { return @() }
    $count = @($Runs[0].history).Count
    foreach ($run in $Runs) {
        if (@($run.history).Count -ne $count) { throw "history length changed between measured runs" }
    }
    $result = @()
    for ($index = 0; $index -lt $count; $index++) {
        $result += [pscustomobject][ordered]@{
            iteration = [int]$Runs[0].history[$index].iteration
            momentumInitialResidual = Get-MatchedMedian ([double[]]@($Runs | ForEach-Object { [double]$_.history[$index].momentumInitialResidual }))
            momentumFinalResidual = Get-MatchedMedian ([double[]]@($Runs | ForEach-Object { [double]$_.history[$index].momentumFinalResidual }))
            momentumLinearIterations = Get-MatchedMedian ([double[]]@($Runs | ForEach-Object { [double]$_.history[$index].momentumLinearIterations }))
            momentumLinearSolves = Get-MatchedMedian ([double[]]@($Runs | ForEach-Object { [double]$_.history[$index].momentumLinearSolves }))
            pressureInitialResidual = Get-MatchedMedian ([double[]]@($Runs | ForEach-Object { [double]$_.history[$index].pressureInitialResidual }))
            pressureFinalResidual = Get-MatchedMedian ([double[]]@($Runs | ForEach-Object { [double]$_.history[$index].pressureFinalResidual }))
            pressureLinearIterations = Get-MatchedMedian ([double[]]@($Runs | ForEach-Object { [double]$_.history[$index].pressureLinearIterations }))
            pressureLinearSolves = Get-MatchedMedian ([double[]]@($Runs | ForEach-Object { [double]$_.history[$index].pressureLinearSolves }))
            continuityIndicator = Get-MatchedMedian ([double[]]@($Runs | ForEach-Object { [double]$_.history[$index].continuityIndicator }))
        }
    }
    return $result
}

function Get-MatchedEngineSummary([string]$Name, $Runs) {
    $measured = @($Runs | Where-Object { $_.kind -eq "measured" })
    if ($measured.Count -eq 0) { throw "engine '$Name' has no measured runs" }
    $pressureSolves = [double[]]@($measured | ForEach-Object {
        [double](($_.history.pressureLinearSolves | Measure-Object -Sum).Sum)
    })
    $perSolve = [double[]]@()
    for ($index = 0; $index -lt $measured.Count; $index++) {
        if ($pressureSolves[$index] -le 0) { throw "engine '$Name' reported no pressure solves" }
        $perSolve += [double]$measured[$index].pressureLinearIterations / $pressureSolves[$index]
    }
    $elapsed = [double[]]@($measured | ForEach-Object { [double]$_.commonProcessElapsedSeconds })
    return [pscustomobject][ordered]@{
        engine = $Name
        medians = [pscustomobject][ordered]@{
            commonProcessElapsedSeconds = Get-MatchedMedian $elapsed
            commonProcessElapsedMadSeconds = Get-MatchedMedianAbsoluteDeviation $elapsed
            processUserSeconds = Get-MatchedMedian ([double[]]@($measured | ForEach-Object { [double]$_.processUserSeconds }))
            processSystemSeconds = Get-MatchedMedian ([double[]]@($measured | ForEach-Object { [double]$_.processSystemSeconds }))
            maxResidentSetKiB = Get-MatchedMedian ([double[]]@($measured | ForEach-Object { [double]$_.maxResidentSetKiB }))
            nativeInternalSeconds = Get-MatchedMedian ([double[]]@($measured | ForEach-Object { [double]$_.nativeInternalSeconds }))
            pressureLinearIterations = Get-MatchedMedian ([double[]]@($measured | ForEach-Object { [double]$_.pressureLinearIterations }))
            pressureLinearSolves = Get-MatchedMedian $pressureSolves
            pressureLinearIterationsPerSolve = Get-MatchedMedian $perSolve
            momentumLinearIterations = Get-MatchedMedian ([double[]]@($measured | ForEach-Object { [double]$_.momentumLinearIterations }))
        }
        historyMedians = @(Get-MatchedHistoryMedians $measured)
        runs = $Runs
    }
}

function Write-MatchedResidualCsv([string]$Path, $FerrumHistory, $OpenFoamHistory) {
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
