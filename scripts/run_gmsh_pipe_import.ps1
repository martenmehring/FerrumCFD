param(
    [string]$GeoFile = "",
    [string]$MeshFile = "",
    [string]$CaseRoot = "",
    [string]$GmshExe = ""
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot

if ([string]::IsNullOrWhiteSpace($GeoFile)) {
    $GeoFile = Join-Path $RepoRoot "examples\gmsh_pipe\pipe_prism2.geo"
}
if ([string]::IsNullOrWhiteSpace($MeshFile)) {
    $MeshFile = Join-Path $RepoRoot "target\gmsh\pipe_prism2.msh"
}
if ([string]::IsNullOrWhiteSpace($CaseRoot)) {
    $CaseRoot = Join-Path $RepoRoot "target\cases\gmsh_pipe"
}

function Resolve-GmshExecutable([string]$ExplicitPath) {
    if (![string]::IsNullOrWhiteSpace($ExplicitPath)) {
        if (Test-Path -LiteralPath $ExplicitPath) {
            return (Resolve-Path -LiteralPath $ExplicitPath).Path
        }
        throw "Gmsh executable not found at '$ExplicitPath'"
    }

    $command = Get-Command gmsh -ErrorAction SilentlyContinue
    if ($null -ne $command) {
        return $command.Source
    }

    $bundled = Join-Path $env:USERPROFILE "Downloads\gmsh-4.15.2-Windows64\gmsh-4.15.2-Windows64\gmsh.exe"
    if (Test-Path -LiteralPath $bundled) {
        return $bundled
    }

    $downloadRoot = Join-Path $env:USERPROFILE "Downloads"
    if (Test-Path -LiteralPath $downloadRoot) {
        $found = Get-ChildItem -LiteralPath $downloadRoot -Filter gmsh.exe -Recurse -ErrorAction SilentlyContinue |
            Select-Object -First 1
        if ($null -ne $found) {
            return $found.FullName
        }
    }

    throw "gmsh.exe was not found in PATH or Downloads. Pass -GmshExe <path-to-gmsh.exe>."
}

$gmsh = Resolve-GmshExecutable $GmshExe
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $MeshFile) | Out-Null
New-Item -ItemType Directory -Force -Path $CaseRoot | Out-Null

Write-Output "gmsh: $gmsh"
Write-Output "geo:  $GeoFile"
Write-Output "mesh: $MeshFile"

$gmshArgs = @("-3", "`"$GeoFile`"", "-format", "msh2", "-o", "`"$MeshFile`"")
$gmshProcess = Start-Process -FilePath $gmsh -ArgumentList $gmshArgs -Wait -PassThru -WindowStyle Hidden
if ($null -ne $gmshProcess.ExitCode -and $gmshProcess.ExitCode -ne 0) {
    throw "gmsh failed with exit code $($gmshProcess.ExitCode)"
}
if (!(Test-Path -LiteralPath $MeshFile)) {
    throw "gmsh did not write mesh file '$MeshFile'"
}

cargo run -p ferrum-cli --bin gmshToFerrumFoam -- $MeshFile -case $CaseRoot
if ($LASTEXITCODE -ne 0) {
    throw "gmshToFerrumFoam failed with exit code $LASTEXITCODE"
}

cargo run -p ferrum-cli --bin checkFerrumMesh -- -case $CaseRoot
if ($LASTEXITCODE -ne 0) {
    throw "checkFerrumMesh failed with exit code $LASTEXITCODE"
}

Write-Output "wrote Ferrum Gmsh pipe case: $CaseRoot"
