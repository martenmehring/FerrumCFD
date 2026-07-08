param(
    [string]$CaseRoot = "",
    [string]$WorkDir = "",
    [string]$OutFile = "",
    [ValidateSet("Auto", "Native", "Wsl")]
    [string]$Mode = "Auto",
    [int]$EndTime = 200,
    [int]$WriteInterval = 0,
    [switch]$RequireOpenFoam
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($CaseRoot)) {
    $CaseRoot = Join-Path $RepoRoot "examples\laminar_pipe"
}
if ([string]::IsNullOrWhiteSpace($WorkDir)) {
    $WorkDir = Join-Path $RepoRoot "target\openfoam\laminar_pipe"
}
if ([string]::IsNullOrWhiteSpace($OutFile)) {
    $OutFile = Join-Path $RepoRoot "target\benchmarks\laminar_pipe_openfoam.json"
}
if ($EndTime -le 0) {
    throw "EndTime must be a positive integer number of SIMPLE pseudo-time steps"
}
if ($WriteInterval -le 0) {
    $WriteInterval = $EndTime
}

function Format-F64([double]$Value) {
    return $Value.ToString("G17", [System.Globalization.CultureInfo]::InvariantCulture)
}

function Write-AsciiFile([string]$Path, [string]$Content) {
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Path) | Out-Null
    Set-Content -LiteralPath $Path -Value $Content.TrimStart("`r", "`n") -Encoding ASCII
}

function ConvertTo-WslPath([string]$Path) {
    $resolved = (Resolve-Path -LiteralPath $Path).Path
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

function Test-NativeOpenFoam {
    return $null -ne (Get-Command simpleFoam -ErrorAction SilentlyContinue)
}

function Test-WslOpenFoam {
    if ($null -eq (Get-Command wsl -ErrorAction SilentlyContinue)) {
        return $false
    }
    & wsl bash -lc "source /opt/openfoam*/etc/bashrc 2>/dev/null || source /usr/lib/openfoam*/etc/bashrc 2>/dev/null || true; command -v simpleFoam >/dev/null 2>&1"
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
    $path = Join-Path $CaseRoot "constant\pipeBenchmark"
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
        return [pscustomobject]$result
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
    }
}

function Measure-AxialPressureLoss($Values, $Benchmark, [string]$CaseRoot) {
    if ($Values.Count -lt 2) {
        return $null
    }

    if ($null -ne $Benchmark.axialCells -and $null -ne $Benchmark.radialCells -and $null -ne $Benchmark.angularSectors) {
        $cellsPerSlice = [int]$Benchmark.radialCells * [int]$Benchmark.angularSectors
        $lastStart = ([int]$Benchmark.axialCells - 1) * $cellsPerSlice
        if ($cellsPerSlice -gt 0 -and $Values.Count -ge ($lastStart + $cellsPerSlice)) {
            $inlet = New-Object System.Collections.Generic.List[double]
            $outlet = New-Object System.Collections.Generic.List[double]
            for ($i = 0; $i -lt $cellsPerSlice; $i++) {
                $inlet.Add([double]$Values[$i]) | Out-Null
                $outlet.Add([double]$Values[$lastStart + $i]) | Out-Null
            }
            $inletAverage = Get-Average $inlet.ToArray()
            $outletAverage = Get-Average $outlet.ToArray()
            return [pscustomobject][ordered]@{
                method = "axialSliceAverage"
                inletSamples = $inlet.Count
                outletSamples = $outlet.Count
                inletAverage = $inletAverage
                outletAverage = $outletAverage
                delta = $inletAverage - $outletAverage
            }
        }
    }

    $patchLoss = Measure-PatchOwnerPressureLoss $Values $CaseRoot
    if ($null -ne $patchLoss) {
        return $patchLoss
    }

    return [pscustomobject][ordered]@{
        method = "firstLastCell"
        inletSamples = 1
        outletSamples = 1
        inletAverage = [double]$Values[0]
        outletAverage = [double]$Values[$Values.Count - 1]
        delta = [double]$Values[0] - [double]$Values[$Values.Count - 1]
    }
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

$benchmark = Read-PipeBenchmarkParameters $CaseRoot
$rho = if ($null -ne $benchmark.rho) { [double]$benchmark.rho } else { 998.2 }
$analyticDeltaPPa = if ($null -ne $benchmark.analyticDeltaPPa) { [double]$benchmark.analyticDeltaPPa } else { 1.6032 }
$analyticDeltaPKinematic = $analyticDeltaPPa / $rho
$initialPressurePa = @(Read-InternalScalarField (Join-Path $CaseRoot "0\p") | ForEach-Object { [double]$_ })
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

if (Test-Path -LiteralPath $WorkDir) {
    Remove-Item -LiteralPath $WorkDir -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $WorkDir | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $WorkDir "0") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $WorkDir "constant") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $WorkDir "system") | Out-Null
Copy-Item -LiteralPath (Join-Path $CaseRoot "constant\polyMesh") -Destination (Join-Path $WorkDir "constant\polyMesh") -Recurse

$sourceU = Join-Path $CaseRoot "0\U"
if (Test-Path -LiteralPath $sourceU) {
    Copy-Item -LiteralPath $sourceU -Destination (Join-Path $WorkDir "0\U") -Force
} else {
    Write-AsciiFile (Join-Path $WorkDir "0\U") @"
FoamFile
{
    version 2.0;
    format ascii;
    class volVectorField;
    location "0";
    object U;
}

dimensions [0 1 -1 0 0 0 0];
internalField uniform (0.02 0 0);

boundaryField
{
    inlet { type fixedValue; value uniform (0.02 0 0); }
    outlet { type zeroGradient; }
    wall { type noSlip; }
}
"@
}

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

Write-AsciiFile (Join-Path $WorkDir "constant\transportProperties") @"
FoamFile
{
    version 2.0;
    format ascii;
    class dictionary;
    location "constant";
    object transportProperties;
}

transportModel Newtonian;
nu [0 2 -1 0 0 0 0] 1.0038e-6;
"@

Write-AsciiFile (Join-Path $WorkDir "constant\turbulenceProperties") @"
FoamFile
{
    version 2.0;
    format ascii;
    class dictionary;
    location "constant";
    object turbulenceProperties;
}

simulationType laminar;
"@

Write-AsciiFile (Join-Path $WorkDir "system\controlDict") @"
FoamFile
{
    version 2.0;
    format ascii;
    class dictionary;
    location "system";
    object controlDict;
}

application simpleFoam;
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

Write-AsciiFile (Join-Path $WorkDir "system\fvSchemes") @"
FoamFile
{
    version 2.0;
    format ascii;
    class dictionary;
    location "system";
    object fvSchemes;
}

ddtSchemes { default steadyState; }
gradSchemes { default Gauss linear; }
divSchemes { default none; div(phi,U) Gauss linearUpwind grad(U); }
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
"@

Write-AsciiFile (Join-Path $WorkDir "system\fvSolution") @"
FoamFile
{
    version 2.0;
    format ascii;
    class dictionary;
    location "system";
    object fvSolution;
}

solvers
{
    p { solver PCG; preconditioner DIC; tolerance 1e-12; relTol 0; }
    U { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-12; relTol 0; }
}

SIMPLE
{
    nNonOrthogonalCorrectors 0;
}

relaxationFactors
{
    fields { p 0.3; }
    equations { U 0.7; }
}
"@

$selectedMode = Get-OpenFoamMode
$logPath = Join-Path $WorkDir "log.simpleFoam"
$script:foamExitCode = $null
$exitCode = $null
$wallClockSeconds = $null
$status = "openfoam-unavailable"

if ($null -eq $selectedMode) {
    if ($RequireOpenFoam) {
        throw "simpleFoam was not found. Install OpenFOAM or run this script from an OpenFOAM-enabled shell."
    }
} else {
    $status = "ran"
    $elapsed = Measure-Command {
        if ($selectedMode -eq "Native") {
            Push-Location $WorkDir
            try {
                & simpleFoam *> $logPath
                $script:foamExitCode = $LASTEXITCODE
            } finally {
                Pop-Location
            }
        } else {
            $wslCase = ConvertTo-WslPath $WorkDir
            $bash = "source /opt/openfoam*/etc/bashrc 2>/dev/null || source /usr/lib/openfoam*/etc/bashrc 2>/dev/null || true; cd '$wslCase' && simpleFoam > log.simpleFoam 2>&1"
            & wsl bash -lc $bash
            $script:foamExitCode = $LASTEXITCODE
        }
    }
    $exitCode = if ($null -eq $script:foamExitCode) { 0 } else { $script:foamExitCode }
    $wallClockSeconds = $elapsed.TotalSeconds
    if ($exitCode -ne 0) {
        $status = "openfoam-failed"
        if ($RequireOpenFoam) {
            throw "simpleFoam failed with exit code $exitCode. See $logPath"
        }
    }
}

$latestTime = Get-LatestTimeDirectory $WorkDir
$openFoamDelta = $null
if ($null -ne $latestTime) {
    $pValues = Read-InternalScalarField (Join-Path $latestTime.FullName "p")
    $loss = Measure-AxialPressureLoss $pValues $benchmark $WorkDir
    if ($null -ne $loss) {
        $deltaPa = $loss.delta * $rho
        $openFoamDelta = [ordered]@{
            latestTime = $latestTime.Name
            samples = $pValues.Count
            method = $loss.method
            inletSamples = $loss.inletSamples
            outletSamples = $loss.outletSamples
            inletAverageKinematic = $loss.inletAverage
            outletAverageKinematic = $loss.outletAverage
            deltaPKinematic = $loss.delta
            deltaPPa = $deltaPa
            relativeErrorToAnalytic = if ($analyticDeltaPPa -ne 0.0) { ($deltaPa - $analyticDeltaPPa) / $analyticDeltaPPa } else { $null }
        }
    }
}

$timing = Read-LastFoamTiming $logPath
$result = [ordered]@{
    case = "laminar_pipe"
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
        type = $benchmark.type
        axialCells = $benchmark.axialCells
        radialCells = $benchmark.radialCells
        angularSectors = $benchmark.angularSectors
        cells = $benchmark.cells
        points = $benchmark.points
    }
    runControl = [ordered]@{
        application = "simpleFoam"
        startTime = 0
        endTime = $EndTime
        deltaT = 1
        writeInterval = $WriteInterval
        simulatedSteps = $EndTime
    }
    openFoam = [ordered]@{
        available = $null -ne $selectedMode
        mode = $selectedMode
        application = "simpleFoam"
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
