param(
    [string]$FerrumOverlayCaseRoot = "",
    [string]$WorkDir = "",
    [string]$OutFile = "",
    [string]$BenchmarkProperties = "",
    [ValidateSet("Auto", "Native", "Wsl")]
    [string]$Mode = "Auto",
    [int]$EndTime = 200,
    [int]$WriteInterval = 0,
    [switch]$UseFerrumOverlayNumerics,
    [switch]$RequireOpenFoam
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
$OpenFoamTemplate = Join-Path $RepoRoot "tutorials\incompressibleFluid\laminarPipe\openfoam-v13\case"
$UseFerrumOverlay = ![string]::IsNullOrWhiteSpace($FerrumOverlayCaseRoot)
$SourceCaseRoot = if ($UseFerrumOverlay) { $FerrumOverlayCaseRoot } else { $OpenFoamTemplate }
if ([string]::IsNullOrWhiteSpace($WorkDir)) {
    $WorkDir = Join-Path $RepoRoot "target\openfoam\laminar_pipe"
}
if ([string]::IsNullOrWhiteSpace($OutFile)) {
    $OutFile = Join-Path $RepoRoot "target\benchmarks\laminar_pipe_openfoam.json"
}
if ([string]::IsNullOrWhiteSpace($BenchmarkProperties)) {
    $BenchmarkProperties = Join-Path $RepoRoot "tutorials\incompressibleFluid\laminarPipe\analytical\pipeBenchmark"
}
if ($EndTime -le 0) {
    throw "EndTime must be a positive integer number of SIMPLE pseudo-time steps"
}
if ($WriteInterval -le 0) {
    $WriteInterval = $EndTime
}
if (!(Test-Path -LiteralPath $OpenFoamTemplate -PathType Container)) {
    throw "OpenFOAM 13 source case was not found: $OpenFoamTemplate"
}
if (!(Test-Path -LiteralPath $SourceCaseRoot -PathType Container)) {
    throw "source case was not found: $SourceCaseRoot"
}
if ($UseFerrumOverlayNumerics -and !$UseFerrumOverlay) {
    throw "UseFerrumOverlayNumerics requires FerrumOverlayCaseRoot"
}

function Format-F64([double]$Value) {
    return $Value.ToString("G17", [System.Globalization.CultureInfo]::InvariantCulture)
}

function Write-AsciiFile([string]$Path, [string]$Content) {
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Path) | Out-Null
    Set-Content -LiteralPath $Path -Value $Content.TrimStart("`r", "`n") -Encoding ASCII
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
    if ($Mode -eq "Native") {
        if (Test-NativeOpenFoam) { return "Native" }
        return $null
    }
    if ($Mode -eq "Wsl") {
        if (Test-WslOpenFoam) { return "Wsl" }
        return $null
    }
    if (Test-NativeOpenFoam) { return "Native" }
    if (Test-WslOpenFoam) { return "Wsl" }
    return $null
}

function Get-LatestTimeDirectory([string]$Root) {
    $latest = Get-ChildItem -LiteralPath $Root -Directory |
        Where-Object {
            $value = 0.0
            [double]::TryParse($_.Name, [System.Globalization.NumberStyles]::Float, [System.Globalization.CultureInfo]::InvariantCulture, [ref]$value)
        } |
        Sort-Object {
            [double]::Parse($_.Name, [System.Globalization.CultureInfo]::InvariantCulture)
        } -Descending |
        Select-Object -First 1
    return $latest
}

function Read-InternalScalarField([string]$Path) {
    if (!(Test-Path -LiteralPath $Path)) {
        return @()
    }
    $content = Get-Content -LiteralPath $Path -Raw
    $uniform = [regex]::Match($content, "internalField\s+uniform\s+([-+0-9.eE]+)\s*;")
    if ($uniform.Success) {
        return @([double]::Parse($uniform.Groups[1].Value, [System.Globalization.CultureInfo]::InvariantCulture))
    }
    $nonuniform = [regex]::Match(
        $content,
        "internalField\s+nonuniform\s+List<scalar>\s+(\d+)\s*\((.*?)\)\s*;",
        [System.Text.RegularExpressions.RegexOptions]::Singleline
    )
    if (!$nonuniform.Success) {
        return @()
    }
    $count = [int]$nonuniform.Groups[1].Value
    $values = $nonuniform.Groups[2].Value -split "\s+" |
        Where-Object { $_ -match "[-+0-9.eE]" } |
        ForEach-Object { [double]::Parse($_, [System.Globalization.CultureInfo]::InvariantCulture) }
    if ($values.Count -ne $count) {
        return @()
    }
    return @($values)
}

function Read-FoamLabelList([string]$Path) {
    if (!(Test-Path -LiteralPath $Path)) {
        return @()
    }

    $lines = Get-Content -LiteralPath $Path
    $count = $null
    $countIndex = -1
    for ($i = 0; $i -lt $lines.Count; $i++) {
        $trimmed = $lines[$i].Trim()
        if ($trimmed -match "^\d+$") {
            $count = [int]::Parse($trimmed, [System.Globalization.CultureInfo]::InvariantCulture)
            $countIndex = $i
            break
        }
    }
    if ($countIndex -lt 0) {
        return @()
    }

    $values = New-Object System.Collections.Generic.List[int]
    for ($i = $countIndex + 1; $i -lt $lines.Count -and $values.Count -lt $count; $i++) {
        foreach ($match in [regex]::Matches($lines[$i], "-?\d+")) {
            $values.Add([int]::Parse($match.Value, [System.Globalization.CultureInfo]::InvariantCulture)) | Out-Null
            if ($values.Count -eq $count) {
                break
            }
        }
    }

    if ($values.Count -ne $count) {
        return @()
    }
    return [int[]]$values.ToArray()
}

function Read-BoundaryPatchRanges([string]$BoundaryPath) {
    $patches = @{}
    if (!(Test-Path -LiteralPath $BoundaryPath)) {
        return $patches
    }

    $lines = Get-Content -LiteralPath $BoundaryPath
    for ($i = 0; $i -lt $lines.Count; $i++) {
        $name = $lines[$i].Trim()
        if ($name -eq "" -or $name -in @("FoamFile", "(", ")") -or $name -match "^\d+$") {
            continue
        }

        $j = $i + 1
        while ($j -lt $lines.Count -and $lines[$j].Trim() -eq "") {
            $j++
        }
        if ($j -ge $lines.Count -or $lines[$j].Trim() -ne "{") {
            continue
        }

        $nFaces = $null
        $startFace = $null
        for ($k = $j + 1; $k -lt $lines.Count; $k++) {
            $entry = $lines[$k].Trim()
            if ($entry -eq "}") {
                break
            }
            if ($entry -match "^nFaces\s+(\d+)\s*;") {
                $nFaces = [int]::Parse($Matches[1], [System.Globalization.CultureInfo]::InvariantCulture)
            } elseif ($entry -match "^startFace\s+(\d+)\s*;") {
                $startFace = [int]::Parse($Matches[1], [System.Globalization.CultureInfo]::InvariantCulture)
            }
        }

        if ($null -ne $nFaces -and $null -ne $startFace) {
            $patches[$name] = [pscustomobject][ordered]@{
                name = $name
                nFaces = $nFaces
                startFace = $startFace
            }
        }
    }
    return $patches
}

function Read-PipeBenchmarkParameters([string]$CaseRoot) {
    $path = $script:BenchmarkProperties
    $result = [ordered]@{
        type = $null
        rho = $null
        analyticDeltaPPa = $null
        axialCells = $null
        radialCells = $null
        angularSectors = $null
        cells = $null
        points = $null
    }
    if (!(Test-Path -LiteralPath $path)) {
        throw "benchmark properties not found: $path"
    }

    $content = Get-Content -LiteralPath $path -Raw
    $rhoMatch = [regex]::Match($content, "(?m)^\s*rho\s+\[[^\]]+\]\s+([-+0-9.eE]+)\s*;")
    if ($rhoMatch.Success) {
        $result.rho = [double]::Parse($rhoMatch.Groups[1].Value, [System.Globalization.CultureInfo]::InvariantCulture)
    }
    $deltaMatch = [regex]::Match($content, "(?m)^\s*expectedDeltaP\s+\[[^\]]+\]\s+([-+0-9.eE]+)\s*;")
    if ($deltaMatch.Success) {
        $result.analyticDeltaPPa = [double]::Parse($deltaMatch.Groups[1].Value, [System.Globalization.CultureInfo]::InvariantCulture)
    }
    foreach ($name in @("axialCells", "radialCells", "angularSectors", "cells")) {
        $match = [regex]::Match($content, "(?m)^\s*$name\s+(\d+)\s*;")
        if ($match.Success) {
            $result[$name] = [int]::Parse($match.Groups[1].Value, [System.Globalization.CultureInfo]::InvariantCulture)
        }
    }
    foreach ($name in @("points")) {
        $match = [regex]::Match($content, "(?m)^\s*$name\s+(\d+)\s*;")
        if ($match.Success) {
            $result[$name] = [int]::Parse($match.Groups[1].Value, [System.Globalization.CultureInfo]::InvariantCulture)
        }
    }
    $typeMatch = [regex]::Match($content, "(?m)^\s*type\s+([A-Za-z0-9_]+)\s*;")
    if ($typeMatch.Success) {
        $result.type = $typeMatch.Groups[1].Value
    }
    return [pscustomobject]$result
}

function Get-Average([double[]]$Values) {
    if ($Values.Count -eq 0) {
        return $null
    }
    $sum = 0.0
    foreach ($value in $Values) {
        $sum += $value
    }
    return $sum / [double]$Values.Count
}

function Measure-PatchOwnerPressureLoss($Values, [string]$CaseRoot) {
    $patches = Read-BoundaryPatchRanges (Join-Path $CaseRoot "constant\polyMesh\boundary")
    if (!$patches.ContainsKey("inlet") -or !$patches.ContainsKey("outlet")) {
        return $null
    }

    $owner = Read-FoamLabelList (Join-Path $CaseRoot "constant\polyMesh\owner")
    if ($owner.Count -eq 0) {
        return $null
    }

    $inlet = New-Object System.Collections.Generic.List[double]
    $outlet = New-Object System.Collections.Generic.List[double]
    foreach ($entry in @(@{ patch = $patches["inlet"]; values = $inlet }, @{ patch = $patches["outlet"]; values = $outlet })) {
        $patch = $entry.patch
        for ($face = $patch.startFace; $face -lt ($patch.startFace + $patch.nFaces); $face++) {
            if ($face -lt 0 -or $face -ge $owner.Count) {
                continue
            }
            $cell = $owner[$face]
            if ($cell -lt 0 -or $cell -ge $Values.Count) {
                continue
            }
            $entry.values.Add([double]$Values[$cell]) | Out-Null
        }
    }

    if ($inlet.Count -eq 0 -or $outlet.Count -eq 0) {
        return $null
    }

    $inletAverage = Get-Average $inlet.ToArray()
    $outletAverage = Get-Average $outlet.ToArray()
    return [pscustomobject][ordered]@{
        method = "boundaryPatchOwnerAverage"
        inletSamples = $inlet.Count
        outletSamples = $outlet.Count
        inletAverage = $inletAverage
        outletAverage = $outletAverage
        delta = $inletAverage - $outletAverage
        effectiveLengthFraction = $null
    }
}

function Measure-AxialPressureLoss($Values, [string]$CaseRoot) {
    if ($Values.Count -lt 2) {
        return $null
    }
    return Measure-PatchOwnerPressureLoss $Values $CaseRoot
}

function Read-LastFoamTiming([string]$LogPath) {
    if (!(Test-Path -LiteralPath $LogPath)) {
        return $null
    }
    $content = Get-Content -LiteralPath $LogPath -Raw
    $matches = [regex]::Matches($content, "ExecutionTime\s*=\s*([-+0-9.eE]+)\s*s\s+ClockTime\s*=\s*([-+0-9.eE]+)\s*s")
    if ($matches.Count -eq 0) {
        return $null
    }
    $last = $matches[$matches.Count - 1]
    return [ordered]@{
        executionTimeSeconds = [double]::Parse($last.Groups[1].Value, [System.Globalization.CultureInfo]::InvariantCulture)
        clockTimeSeconds = [double]::Parse($last.Groups[2].Value, [System.Globalization.CultureInfo]::InvariantCulture)
    }
}

$benchmark = Read-PipeBenchmarkParameters $SourceCaseRoot
$rho = $benchmark.rho
$analyticDeltaPPa = $benchmark.analyticDeltaPPa
foreach ($requiredValue in @(@{ name = "rho"; value = $rho }, @{ name = "expectedDeltaP"; value = $analyticDeltaPPa })) {
    if ($null -eq $requiredValue.value -or [double]::IsNaN([double]$requiredValue.value) -or
        [double]::IsInfinity([double]$requiredValue.value) -or [double]$requiredValue.value -le 0.0) {
        throw "$($requiredValue.name) must be a positive finite value in $BenchmarkProperties"
    }
}
$rho = [double]$rho
$analyticDeltaPPa = [double]$analyticDeltaPPa
$analyticDeltaPKinematic = $analyticDeltaPPa / $rho
$sourceOwner = Read-FoamLabelList (Join-Path $SourceCaseRoot "constant\polyMesh\owner")
$sourceNeighbour = Read-FoamLabelList (Join-Path $SourceCaseRoot "constant\polyMesh\neighbour")
$sourceOwnerMaximum = if ($sourceOwner.Count -gt 0) { [int](($sourceOwner | Measure-Object -Maximum).Maximum) } else { $null }
$sourceNeighbourMaximum = if ($sourceNeighbour.Count -gt 0) { [int](($sourceNeighbour | Measure-Object -Maximum).Maximum) } else { $null }
$sourceMaximumCell = if ($null -eq $sourceOwnerMaximum) {
    $sourceNeighbourMaximum
} elseif ($null -eq $sourceNeighbourMaximum) {
    $sourceOwnerMaximum
} else {
    [Math]::Max($sourceOwnerMaximum, $sourceNeighbourMaximum)
}
$sourceCellCount = if ($null -ne $sourceMaximumCell) { 1L + [long]$sourceMaximumCell } else { $null }
$sourceFaceCount = if ($sourceOwner.Count -gt 0) { $sourceOwner.Count } else { $null }
$initialPressureField = $null
if ($UseFerrumOverlay) {
    $initialPressurePa = @(Read-InternalScalarField (Join-Path $SourceCaseRoot "0\p") | ForEach-Object { [double]$_ })
    $initialPressureField = "internalField uniform 0;"
    if ($initialPressurePa.Count -eq 1) {
        $initialPressureField = "internalField uniform $(Format-F64 ([double]$initialPressurePa[0] / $rho));"
    } elseif ($initialPressurePa.Count -gt 1) {
        $initialPressureKinematicLines = @($initialPressurePa | ForEach-Object { "    $(Format-F64 ($_ / $rho))" })
        $initialPressureBlock = $initialPressureKinematicLines -join "`n"
        $initialPressureField = @"
internalField nonuniform List<scalar>
$($initialPressureKinematicLines.Count)
(
$initialPressureBlock
);
"@
    }
}

$targetRoot = Join-Path $RepoRoot "target"
if (!(Test-IsPathUnder $WorkDir $targetRoot)) {
    throw "WorkDir must be inside the repository target directory: $targetRoot"
}

if (Test-Path -LiteralPath $WorkDir) {
    Remove-Item -LiteralPath $WorkDir -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $WorkDir | Out-Null
Copy-Item -Path (Join-Path $OpenFoamTemplate "*") -Destination $WorkDir -Recurse -Force

if ($UseFerrumOverlay) {
    $workPolyMesh = Join-Path $WorkDir "constant\polyMesh"
    if (Test-Path -LiteralPath $workPolyMesh) {
        Remove-Item -LiteralPath $workPolyMesh -Recurse -Force
    }
    Copy-Item -LiteralPath (Join-Path $SourceCaseRoot "constant\polyMesh") -Destination $workPolyMesh -Recurse

    $sourceU = Join-Path $SourceCaseRoot "0\U"
    if (!(Test-Path -LiteralPath $sourceU -PathType Leaf)) {
        throw "Ferrum overlay requires a velocity field: $sourceU"
    }
    Copy-Item -LiteralPath $sourceU -Destination (Join-Path $WorkDir "0\U") -Force

    Write-AsciiFile (Join-Path $WorkDir "0\p") @"
FoamFile
{
    version 2.0;
    format ascii;
    class volScalarField;
    location "0";
    object p;
}

// OpenFOAM incompressible p is kinematic pressure in m2/s2.
// The benchmark JSON converts it back to SI pressure in Pa.
dimensions [0 2 -2 0 0 0 0];
$initialPressureField

boundaryField
{
    inlet { type zeroGradient; }
    outlet { type fixedValue; value uniform 0; }
    wall { type zeroGradient; }
}
"@

    $sourceTransportProperties = Join-Path $SourceCaseRoot "constant\transportProperties"
    if (!(Test-Path -LiteralPath $sourceTransportProperties -PathType Leaf)) {
        throw "Ferrum overlay requires transportProperties: $sourceTransportProperties"
    }
    $transportContent = Get-Content -LiteralPath $sourceTransportProperties -Raw
    $nuMatch = [regex]::Match($transportContent, "(?m)^\s*nu\s+\[[^\]]+\]\s+([-+0-9.eE]+)\s*;")
    if (!$nuMatch.Success) {
        throw "Ferrum overlay requires a dimensioned nu entry: $sourceTransportProperties"
    }
    $nu = [double]::Parse($nuMatch.Groups[1].Value, [System.Globalization.CultureInfo]::InvariantCulture)
    if ([double]::IsNaN($nu) -or [double]::IsInfinity($nu) -or $nu -le 0.0) {
        throw "nu must be a positive finite value in $sourceTransportProperties"
    }

    Write-AsciiFile (Join-Path $WorkDir "constant\physicalProperties") @"
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
}

Write-AsciiFile (Join-Path $WorkDir "system\controlDict") @"
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
endTime $EndTime;
deltaT 1;
writeControl timeStep;
writeInterval $WriteInterval;
writeFormat ascii;
writePrecision 10;
runTimeModifiable false;
"@

foreach ($dictionaryName in @("fvSchemes", "fvSolution")) {
    $numericsRoot = if ($UseFerrumOverlayNumerics) { $SourceCaseRoot } else { $OpenFoamTemplate }
    $sourceDictionary = Join-Path $numericsRoot "system\$dictionaryName"
    if (!(Test-Path -LiteralPath $sourceDictionary -PathType Leaf)) {
        throw "OpenFOAM 13 source case requires dictionary: $sourceDictionary"
    }
    Copy-Item -LiteralPath $sourceDictionary -Destination (Join-Path $WorkDir "system\$dictionaryName") -Force
}

$selectedMode = Get-OpenFoamMode
$logPath = Join-Path $WorkDir "log.foamRun"
$script:foamExitCode = $null
$exitCode = $null
$wallClockSeconds = $null
$status = "openfoam-unavailable"

if ($null -eq $selectedMode) {
    if ($RequireOpenFoam) {
        throw "OpenFOAM Foundation 13 foamRun was not found. Install OpenFOAM 13 or run this script from an OpenFOAM-13-enabled shell."
    }
} else {
    $status = "ran"
    $elapsed = Measure-Command {
        if ($selectedMode -eq "Native") {
            Push-Location $WorkDir
            try {
                & foamRun -solver incompressibleFluid *> $logPath
                $script:foamExitCode = $LASTEXITCODE
            } finally {
                Pop-Location
            }
        } else {
            $wslCase = ConvertTo-WslPath $WorkDir
            $quotedWslCase = ConvertTo-BashSingleQuoted $wslCase
            $bash = "source /opt/openfoam13/etc/bashrc 2>/dev/null && env | grep -q '^WM_PROJECT_VERSION=13$' && cd -- $quotedWslCase && foamRun -solver incompressibleFluid > log.foamRun 2>&1"
            & wsl bash -lc $bash
            $script:foamExitCode = $LASTEXITCODE
        }
    }
    $exitCode = if ($null -eq $script:foamExitCode) { 0 } else { $script:foamExitCode }
    $wallClockSeconds = $elapsed.TotalSeconds
    if ($exitCode -ne 0) {
        $status = "openfoam-failed"
        if ($RequireOpenFoam) {
            throw "OpenFOAM 13 foamRun failed with exit code $exitCode. See $logPath"
        }
    }
}

$latestTime = Get-LatestTimeDirectory $WorkDir
$openFoamDelta = $null
if ($null -ne $latestTime) {
    $pValues = Read-InternalScalarField (Join-Path $latestTime.FullName "p")
    $loss = Measure-AxialPressureLoss $pValues $WorkDir
    if ($null -ne $loss) {
        $sampledDeltaKinematic = $loss.delta
        $sampledDeltaPa = $sampledDeltaKinematic * $rho
        $effectiveLengthFraction = $loss.effectiveLengthFraction
        $deltaPa = $sampledDeltaPa
        $deltaKinematic = $sampledDeltaKinematic
        $openFoamDelta = [ordered]@{
            latestTime = $latestTime.Name
            samples = $pValues.Count
            method = $loss.method
            inletSamples = $loss.inletSamples
            outletSamples = $loss.outletSamples
            inletAverageKinematic = $loss.inletAverage
            outletAverageKinematic = $loss.outletAverage
            sampledDeltaPKinematic = $sampledDeltaKinematic
            sampledDeltaPPa = $sampledDeltaPa
            effectiveLengthFraction = $effectiveLengthFraction
            deltaPKinematic = $deltaKinematic
            deltaPPa = $deltaPa
            relativeErrorToAnalytic = if ($analyticDeltaPPa -ne 0.0) { ($deltaPa - $analyticDeltaPPa) / $analyticDeltaPPa } else { $null }
        }
    } elseif ($status -eq "ran") {
        $status = "pressure-sampling-failed"
        if ($RequireOpenFoam) {
            throw "could not sample inlet/outlet owner-cell pressure from $($latestTime.FullName)"
        }
    }
} elseif ($status -eq "ran") {
    $status = "pressure-output-missing"
    if ($RequireOpenFoam) {
        throw "OpenFOAM 13 foamRun completed without a numeric output time directory in $WorkDir"
    }
}

$timing = Read-LastFoamTiming $logPath
$timeEntries = if (Test-Path -LiteralPath $logPath) {
    @(Select-String -Path $logPath -Pattern '^Time = ')
} else {
    @()
}
$simpleConverged = if (Test-Path -LiteralPath $logPath) {
    $null -ne (Select-String -Path $logPath -Pattern 'SIMPLE solution converged' | Select-Object -First 1)
} else {
    $false
}
$result = [ordered]@{
    case = "laminar_pipe"
    sourceCase = $SourceCaseRoot
    ferrumOverlay = $UseFerrumOverlay
    ferrumOverlayNumerics = [bool]$UseFerrumOverlayNumerics
    generatedCase = $WorkDir
    status = $status
    units = [ordered]@{
        ferrumDefault = "SI"
        ferrumPressure = "Pa"
        openFoamPressure = "kinematic m2/s2, converted to Pa with rho"
    }
    analytic = [ordered]@{
        pressureLossModel = "HagenPoiseuille"
        rho = $rho
        deltaPPa = $analyticDeltaPPa
        deltaPKinematic = $analyticDeltaPKinematic
    }
    mesh = [ordered]@{
        type = "polyMesh"
        axialCells = $null
        radialCells = $null
        angularSectors = $null
        cells = $sourceCellCount
        faces = $sourceFaceCount
        points = $null
    }
    runControl = [ordered]@{
        application = "foamRun"
        solverModule = "incompressibleFluid"
        version = 13
        startTime = 0
        endTime = $EndTime
        deltaT = 1
        writeInterval = $WriteInterval
        simulatedSteps = $timeEntries.Count
        converged = $simpleConverged
    }
    openFoam = [ordered]@{
        available = $null -ne $selectedMode
        mode = $selectedMode
        application = "foamRun"
        solverModule = "incompressibleFluid"
        version = 13
        exitCode = $exitCode
        wallClockSeconds = $wallClockSeconds
        log = $logPath
        foamTiming = $timing
        pressureLoss = $openFoamDelta
    }
}

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $OutFile) | Out-Null
$result | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $OutFile -Encoding UTF8
Write-Output "wrote OpenFOAM laminar pipe benchmark: $OutFile"
Write-Output "status: $status"
