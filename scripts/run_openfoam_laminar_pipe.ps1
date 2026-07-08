param(
    [string]$CaseRoot = "",
    [string]$WorkDir = "",
    [string]$OutFile = "",
    [ValidateSet("Auto", "Native", "Wsl")]
    [string]$Mode = "Auto",
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

$rho = 998.2
$analyticDeltaPPa = 1.6032
$analyticDeltaPKinematic = $analyticDeltaPPa / $rho
$initialPressurePa = @(1.6032, 1.2024, 0.8016, 0.4008)
$initialPressureKinematic = $initialPressurePa | ForEach-Object { Format-F64 ($_ / $rho) }

if (Test-Path -LiteralPath $WorkDir) {
    Remove-Item -LiteralPath $WorkDir -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $WorkDir | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $WorkDir "0") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $WorkDir "constant") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $WorkDir "system") | Out-Null
Copy-Item -LiteralPath (Join-Path $CaseRoot "constant\polyMesh") -Destination (Join-Path $WorkDir "constant\polyMesh") -Recurse

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
internalField nonuniform List<scalar>
4
(
    $($initialPressureKinematic[0])
    $($initialPressureKinematic[1])
    $($initialPressureKinematic[2])
    $($initialPressureKinematic[3])
);

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
endTime 200;
deltaT 1;
writeControl timeStep;
writeInterval 200;
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
    if ($pValues.Count -ge 2) {
        $deltaKinematic = [double]$pValues[0] - [double]$pValues[$pValues.Count - 1]
        $deltaPa = $deltaKinematic * $rho
        $openFoamDelta = [ordered]@{
            latestTime = $latestTime.Name
            samples = $pValues.Count
            deltaPKinematic = $deltaKinematic
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
    runControl = [ordered]@{
        application = "simpleFoam"
        startTime = 0
        endTime = 200
        deltaT = 1
        writeInterval = 200
        simulatedSteps = 200
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
