param(
    [Parameter(Mandatory = $true)]
    [string]$GmshExe,
    [string]$CaseRoot = "",
    [string]$MeshFile = "",
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
$TargetRoot = Join-Path $RepoRoot "target"
$GeoFile = Join-Path $RepoRoot "tutorials\incompressibleFluid\planeChannel\shared\geometry\plane_channel.geo"
$TemplateRoot = Join-Path $RepoRoot "tutorials\incompressibleFluid\planeChannel\ferrum\case"

if ([string]::IsNullOrWhiteSpace($CaseRoot)) {
    $CaseRoot = Join-Path $TargetRoot "cases\plane_channel"
}
if ([string]::IsNullOrWhiteSpace($MeshFile)) {
    $MeshFile = Join-Path $TargetRoot "gmsh\plane_channel.msh"
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
    return $childFull.StartsWith(
        $parentFull + [System.IO.Path]::DirectorySeparatorChar,
        [System.StringComparison]::OrdinalIgnoreCase
    ) -or $childFull.StartsWith(
        $parentFull + [System.IO.Path]::AltDirectorySeparatorChar,
        [System.StringComparison]::OrdinalIgnoreCase
    )
}

function Quote-NativeArgument([string]$Value) {
    if ($Value.Contains('"')) {
        throw "native process argument contains an unsupported quote: $Value"
    }
    return '"' + $Value + '"'
}

$GmshExe = Get-FullPath $GmshExe
$CaseRoot = Get-FullPath $CaseRoot
$MeshFile = Get-FullPath $MeshFile

if (!(Test-Path -LiteralPath $GmshExe -PathType Leaf)) {
    throw "gmsh executable was not found: $GmshExe"
}
if (!(Test-Path -LiteralPath $GeoFile -PathType Leaf)) {
    throw "plane-channel geo file was not found: $GeoFile"
}
if (!(Test-Path -LiteralPath $TemplateRoot -PathType Container)) {
    throw "plane-channel case template was not found: $TemplateRoot"
}
if (!(Test-IsPathUnder $CaseRoot $TargetRoot)) {
    throw "CaseRoot must be inside the repository target directory: $TargetRoot"
}
if (!(Test-IsPathUnder $MeshFile $TargetRoot)) {
    throw "MeshFile must be inside the repository target directory: $TargetRoot"
}

if (Test-Path -LiteralPath $CaseRoot) {
    if (!$Force) {
        throw "CaseRoot already exists; pass -Force to replace it: $CaseRoot"
    }
    Remove-Item -LiteralPath $CaseRoot -Recurse -Force
}

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $MeshFile) | Out-Null
$gmshArguments = @(
    (Quote-NativeArgument $GeoFile),
    "-3",
    "-format",
    "msh2",
    "-o",
    (Quote-NativeArgument $MeshFile)
)
$gmshStartInfo = @{
    FilePath = $GmshExe
    ArgumentList = $gmshArguments
    Wait = $true
    PassThru = $true
    WindowStyle = "Hidden"
}
$gmshProcess = Start-Process @gmshStartInfo
if ($gmshProcess.ExitCode -ne 0) {
    throw "gmsh failed with exit code $($gmshProcess.ExitCode)"
}
if (!(Test-Path -LiteralPath $MeshFile -PathType Leaf)) {
    throw "gmsh completed without writing the mesh: $MeshFile"
}

Push-Location $RepoRoot
try {
    cargo run -p ferrum-cli --bin initFerrumCase -- $CaseRoot
    if ($LASTEXITCODE -ne 0) {
        throw "initFerrumCase failed with exit code $LASTEXITCODE"
    }

    cargo run -p ferrum-cli --bin gmshToFerrum -- $MeshFile -case $CaseRoot -emptyPatch front -emptyPatch back -patchType wall=wall
    if ($LASTEXITCODE -ne 0) {
        throw "gmshToFerrum failed with exit code $LASTEXITCODE"
    }

    Copy-Item -Path (Join-Path $TemplateRoot "0\*") -Destination (Join-Path $CaseRoot "0") -Recurse -Force
    Copy-Item -Path (Join-Path $TemplateRoot "system\*") -Destination (Join-Path $CaseRoot "system") -Recurse -Force
    Get-ChildItem -LiteralPath (Join-Path $TemplateRoot "constant") -File |
        Where-Object Name -ne "ferrumMeshSummary.txt" |
        Copy-Item -Destination (Join-Path $CaseRoot "constant") -Force

    cargo run -p ferrum-cli --bin checkFerrumMesh -- -case $CaseRoot
    if ($LASTEXITCODE -ne 0) {
        throw "checkFerrumMesh failed with exit code $LASTEXITCODE"
    }
} finally {
    Pop-Location
}

Write-Output "prepared plane-channel case: $CaseRoot"
Write-Output "generated Gmsh mesh: $MeshFile"
