param(
    [string]$StudyRoot = "",
    [string]$GeoFile = "",
    [string]$GmshExe = "",
    [ValidateSet("Auto", "Native", "Wsl")]
    [string]$Mode = "Auto",
    [string[]]$VariantName = @(),
    [int]$OpenFoamSteps = 200,
    [double]$Length = 1.0,
    [double]$Diameter = 0.02,
    [double]$MeanVelocity = 0.02,
    [double]$Temperature = 293.15,
    [double]$WallTemperature = 333.15,
    [switch]$SkipOpenFoam,
    [switch]$RequireOpenFoam,
    [switch]$SkipFerrumSolve,
    [ValidateSet("jacobi", "cg")]
    [string]$FerrumLinearSolver = "cg",
    [double]$FerrumSolveTolerance = 1e-8,
    [int]$FerrumMaxIterations = 20000
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($StudyRoot)) {
    $StudyRoot = Join-Path $RepoRoot "target\benchmarks\gmsh_pipe_mesh_study"
}
if ([string]::IsNullOrWhiteSpace($GeoFile)) {
    $GeoFile = Join-Path $RepoRoot "examples\gmsh_pipe\pipe_prism2.geo"
}
if ($OpenFoamSteps -le 0) {
    throw "OpenFoamSteps must be positive"
}
foreach ($inputValue in @(
        @{ name = "Length"; value = $Length },
        @{ name = "Diameter"; value = $Diameter },
        @{ name = "MeanVelocity"; value = $MeanVelocity },
        @{ name = "Temperature"; value = $Temperature },
        @{ name = "WallTemperature"; value = $WallTemperature }
    )) {
    if ([double]::IsNaN([double]$inputValue.value) -or [double]::IsInfinity([double]$inputValue.value) -or
        [double]$inputValue.value -le 0.0) {
        throw "$($inputValue.name) must be a positive finite SI value"
    }
}
if ($FerrumSolveTolerance -le 0.0) {
    throw "FerrumSolveTolerance must be positive"
}
if ($FerrumMaxIterations -le 0) {
    throw "FerrumMaxIterations must be positive"
}

$culture = [System.Globalization.CultureInfo]::InvariantCulture
$radius = 0.5 * $Diameter
$rho = 998.2
$mu = 1.002e-3
$nu = 1.0038e-6
$cp = 4182.0
$kThermal = 0.598
$pr = 7.01
$reynolds = $rho * $MeanVelocity * $Diameter / $mu
$deltaP = 32.0 * $mu * $MeanVelocity * $Length / ($Diameter * $Diameter)
$deltaPKinematic = $deltaP / $rho
$nusseltWallTemperature = 3.66
$heatTransferCoefficient = $nusseltWallTemperature * $kThermal / $Diameter

function Format-F64([double]$Value) {
    return $Value.ToString("G17", $culture)
}

function Format-NullableNumber($Value, [string]$Format = "G6") {
    if ($null -eq $Value) {
        return "n/a"
    }
    return ([double]$Value).ToString($Format, $culture)
}

function Format-NullablePercent($Value) {
    if ($null -eq $Value) {
        return "n/a"
    }
    return (([double]$Value) * 100.0).ToString("F3", $culture) + "%"
}

function Write-AsciiFile([string]$Path, [string[]]$Lines) {
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Path) | Out-Null
    Set-Content -LiteralPath $Path -Value $Lines -Encoding ASCII
}

function Add-Lines($Lines, $Values) {
    foreach ($value in $Values) {
        $Lines.Add([string]$value) | Out-Null
    }
}

function New-FoamHeader([string]$ClassName, [string]$Object, [string]$Location) {
    @(
        "FoamFile",
        "{",
        "    version 2.0;",
        "    format ascii;",
        "    class $ClassName;",
        "    location `"$Location`";",
        "    object $Object;",
        "}",
        ""
    )
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

function Remove-DirectoryIfExists([string]$Path, [string]$AllowedRoot) {
    if (!(Test-Path -LiteralPath $Path)) {
        return
    }
    if (!(Test-IsPathUnder $Path $AllowedRoot)) {
        throw "refusing to remove '$Path' because it is outside '$AllowedRoot'"
    }
    Remove-Item -LiteralPath $Path -Recurse -Force
}

function Resolve-GmshExecutable([string]$ExplicitPath) {
    if (![string]::IsNullOrWhiteSpace($ExplicitPath)) {
        if (Test-Path -LiteralPath $ExplicitPath -PathType Leaf) {
            return (Resolve-Path -LiteralPath $ExplicitPath).Path
        }
        throw "Gmsh executable not found at '$ExplicitPath'"
    }

    $command = Get-Command gmsh -CommandType Application -ErrorAction SilentlyContinue
    if ($null -ne $command) {
        return $command.Source
    }

    throw "gmsh.exe was not found in PATH. Pass the trusted installation explicitly with -GmshExe <path-to-gmsh.exe>."
}

function Read-FoamCountedBlockLines([string]$Path) {
    $lines = Get-Content -LiteralPath $Path
    $count = $null
    $countIndex = -1
    for ($i = 0; $i -lt $lines.Count; $i++) {
        $trimmed = $lines[$i].Trim()
        if ($trimmed -match "^\d+$") {
            $count = [int]::Parse($trimmed, $culture)
            $countIndex = $i
            break
        }
    }
    if ($countIndex -lt 0) {
        throw "could not read counted OpenFOAM list in '$Path'"
    }

    return [pscustomobject][ordered]@{
        count = $count
        lines = $lines
        startIndex = $countIndex + 1
    }
}

function Read-FoamPoints([string]$Path) {
    $block = Read-FoamCountedBlockLines $Path
    $points = New-Object System.Collections.Generic.List[object]
    for ($i = $block.startIndex; $i -lt $block.lines.Count -and $points.Count -lt $block.count; $i++) {
        $match = [regex]::Match($block.lines[$i], "\(([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\)")
        if (!$match.Success) {
            continue
        }
        $points.Add([pscustomobject][ordered]@{
                x = [double]::Parse($match.Groups[1].Value, $culture)
                y = [double]::Parse($match.Groups[2].Value, $culture)
                z = [double]::Parse($match.Groups[3].Value, $culture)
            }) | Out-Null
    }
    if ($points.Count -ne $block.count) {
        throw "expected $($block.count) points in '$Path', read $($points.Count)"
    }
    return @($points.ToArray())
}

function Read-FoamFaces([string]$Path) {
    $block = Read-FoamCountedBlockLines $Path
    $faces = New-Object System.Collections.Generic.List[object]
    for ($i = $block.startIndex; $i -lt $block.lines.Count -and $faces.Count -lt $block.count; $i++) {
        $match = [regex]::Match($block.lines[$i], "^\s*(\d+)\(([^)]*)\)")
        if (!$match.Success) {
            continue
        }
        $nodeCount = [int]::Parse($match.Groups[1].Value, $culture)
        $nodes = @($match.Groups[2].Value -split "\s+" | Where-Object { $_ -ne "" } | ForEach-Object {
                [int]::Parse($_, $culture)
            })
        if ($nodes.Count -ne $nodeCount) {
            throw "face in '$Path' declares $nodeCount nodes but contains $($nodes.Count)"
        }
        $faces.Add([int[]]$nodes) | Out-Null
    }
    if ($faces.Count -ne $block.count) {
        throw "expected $($block.count) faces in '$Path', read $($faces.Count)"
    }
    return @($faces.ToArray())
}

function Read-FoamLabelList([string]$Path) {
    if (!(Test-Path -LiteralPath $Path)) {
        return @()
    }

    $block = Read-FoamCountedBlockLines $Path
    $values = New-Object System.Collections.Generic.List[int]
    for ($i = $block.startIndex; $i -lt $block.lines.Count -and $values.Count -lt $block.count; $i++) {
        foreach ($match in [regex]::Matches($block.lines[$i], "-?\d+")) {
            $values.Add([int]::Parse($match.Value, $culture)) | Out-Null
            if ($values.Count -eq $block.count) {
                break
            }
        }
    }
    if ($values.Count -ne $block.count) {
        return @()
    }
    return [int[]]$values.ToArray()
}

function Read-BoundaryPatchRanges([string]$BoundaryPath) {
    $patches = @{}
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
        $type = "patch"
        for ($k = $j + 1; $k -lt $lines.Count; $k++) {
            $entry = $lines[$k].Trim()
            if ($entry -eq "}") {
                break
            }
            if ($entry -match "^type\s+(\S+)\s*;") {
                $type = $Matches[1]
            } elseif ($entry -match "^nFaces\s+(\d+)\s*;") {
                $nFaces = [int]::Parse($Matches[1], $culture)
            } elseif ($entry -match "^startFace\s+(\d+)\s*;") {
                $startFace = [int]::Parse($Matches[1], $culture)
            }
        }

        if ($null -ne $nFaces -and $null -ne $startFace) {
            $patches[$name] = [pscustomobject][ordered]@{
                name = $name
                type = $type
                nFaces = $nFaces
                startFace = $startFace
            }
        }
    }
    return $patches
}

function Get-Centroid($Points, [int[]]$Indices) {
    $sx = 0.0
    $sy = 0.0
    $sz = 0.0
    foreach ($index in $Indices) {
        $point = $Points[$index]
        $sx += $point.x
        $sy += $point.y
        $sz += $point.z
    }
    $count = [double]$Indices.Count
    return [pscustomobject][ordered]@{ x = $sx / $count; y = $sy / $count; z = $sz / $count }
}

function Get-FaceNormal($Points, [int[]]$Indices) {
    $nx = 0.0
    $ny = 0.0
    $nz = 0.0
    for ($i = 0; $i -lt $Indices.Count; $i++) {
        $current = $Points[$Indices[$i]]
        $next = $Points[$Indices[($i + 1) % $Indices.Count]]
        $nx += ($current.y - $next.y) * ($current.z + $next.z)
        $ny += ($current.z - $next.z) * ($current.x + $next.x)
        $nz += ($current.x - $next.x) * ($current.y + $next.y)
    }
    return [pscustomobject][ordered]@{ x = $nx; y = $ny; z = $nz }
}

function Get-FaceArea($Points, [int[]]$Indices) {
    $normal = Get-FaceNormal $Points $Indices
    return 0.5 * [Math]::Sqrt($normal.x * $normal.x + $normal.y * $normal.y + $normal.z * $normal.z)
}

function Read-FerrumMeshSummary([string]$CaseRoot) {
    $path = Join-Path $CaseRoot "constant\ferrumMeshSummary.txt"
    $result = [ordered]@{
        points = $null
        cells = $null
        faces = $null
        internalFaces = $null
        boundaryFaces = $null
    }
    if (Test-Path -LiteralPath $path) {
        foreach ($line in Get-Content -LiteralPath $path) {
            if ($line -match "^points=(\d+)") { $result.points = [int]::Parse($Matches[1], $culture) }
            elseif ($line -match "^cells=(\d+)") { $result.cells = [int]::Parse($Matches[1], $culture) }
            elseif ($line -match "^faces=(\d+)") { $result.faces = [int]::Parse($Matches[1], $culture) }
            elseif ($line -match "^internal_faces=(\d+)") { $result.internalFaces = [int]::Parse($Matches[1], $culture) }
            elseif ($line -match "^boundary_faces=(\d+)") { $result.boundaryFaces = [int]::Parse($Matches[1], $culture) }
        }
    }

    if ($null -eq $result.cells) {
        $owner = Read-FoamLabelList (Join-Path $CaseRoot "constant\polyMesh\owner")
        $neighbour = Read-FoamLabelList (Join-Path $CaseRoot "constant\polyMesh\neighbour")
        $labels = @($owner + $neighbour)
        if ($labels.Count -gt 0) {
            $result.cells = ([int](($labels | Measure-Object -Maximum).Maximum)) + 1
        }
    }

    return [pscustomobject]$result
}

function Write-GmshPipeCaseInputs($CaseRoot, $Variant, [string]$MeshFile) {
    $polyMesh = Join-Path $CaseRoot "constant\polyMesh"
    $points = Read-FoamPoints (Join-Path $polyMesh "points")
    $faces = Read-FoamFaces (Join-Path $polyMesh "faces")
    $patches = Read-BoundaryPatchRanges (Join-Path $polyMesh "boundary")
    foreach ($required in @("inlet", "outlet", "wall")) {
        if (!$patches.ContainsKey($required)) {
            throw "Gmsh pipe case '$CaseRoot' is missing required patch '$required'"
        }
    }

    $inletPatch = $patches["inlet"]
    $inletProfileValues = New-Object System.Collections.Generic.List[object]
    $inletArea = 0.0
    $inletUnscaledFlow = 0.0
    for ($faceIndex = $inletPatch.startFace; $faceIndex -lt ($inletPatch.startFace + $inletPatch.nFaces); $faceIndex++) {
        $nodes = [int[]]$faces[$faceIndex]
        $area = Get-FaceArea $points $nodes
        $centroid = Get-Centroid $points $nodes
        $r2 = $centroid.y * $centroid.y + $centroid.z * $centroid.z
        $profile = 2.0 * $MeanVelocity * (1.0 - ($r2 / ($radius * $radius)))
        if ($profile -lt 0.0) {
            $profile = 0.0
        }
        $inletArea += $area
        $inletUnscaledFlow += $profile * $area
        $inletProfileValues.Add([pscustomobject][ordered]@{ profile = $profile; area = $area }) | Out-Null
    }
    if ([double]::IsNaN($inletUnscaledFlow) -or [double]::IsInfinity($inletUnscaledFlow) -or
        $inletUnscaledFlow -le 0.0 -or [double]::IsNaN($inletArea) -or
        [double]::IsInfinity($inletArea) -or $inletArea -le 0.0) {
        throw "imported inlet patch must have positive finite area and unscaled flow"
    }
    $inletVelocityScale = ($MeanVelocity * $inletArea) / $inletUnscaledFlow
    if ([double]::IsNaN($inletVelocityScale) -or [double]::IsInfinity($inletVelocityScale) -or
        $inletVelocityScale -le 0.0) {
        throw "imported inlet velocity scale must be positive and finite"
    }

    $linesU = New-Object System.Collections.Generic.List[string]
    Add-Lines $linesU (New-FoamHeader "volVectorField" "U" "0")
    $linesU.Add("dimensions [0 1 -1 0 0 0 0];")
    $linesU.Add("")
    $linesU.Add("internalField uniform ($(Format-F64 $MeanVelocity) 0 0);")
    $linesU.Add("")
    $linesU.Add("boundaryField")
    $linesU.Add("{")
    $linesU.Add("    inlet")
    $linesU.Add("    {")
    $linesU.Add("        type fixedValue;")
    $linesU.Add("        value nonuniform List<vector>")
    $linesU.Add("        $($inletProfileValues.Count)")
    $linesU.Add("        (")
    foreach ($entry in $inletProfileValues) {
        $linesU.Add("            ($(Format-F64 ($entry.profile * $inletVelocityScale)) 0 0)")
    }
    $linesU.Add("        );")
    $linesU.Add("    }")
    $linesU.Add("    outlet")
    $linesU.Add("    {")
    $linesU.Add("        type zeroGradient;")
    $linesU.Add("    }")
    $linesU.Add("    wall")
    $linesU.Add("    {")
    $linesU.Add("        type noSlip;")
    $linesU.Add("    }")
    $linesU.Add("}")
    Write-AsciiFile (Join-Path $CaseRoot "0\U") $linesU.ToArray()

    $linesP = New-Object System.Collections.Generic.List[string]
    Add-Lines $linesP (New-FoamHeader "volScalarField" "p" "0")
    $linesP.Add("dimensions [1 -1 -2 0 0 0 0];")
    $linesP.Add("")
    $linesP.Add("internalField uniform 0;")
    $linesP.Add("")
    $linesP.Add("boundaryField")
    $linesP.Add("{")
    $linesP.Add("    inlet")
    $linesP.Add("    {")
    $linesP.Add("        type zeroGradient;")
    $linesP.Add("    }")
    $linesP.Add("    outlet")
    $linesP.Add("    {")
    $linesP.Add("        type fixedValue;")
    $linesP.Add("        value uniform 0;")
    $linesP.Add("    }")
    $linesP.Add("    wall")
    $linesP.Add("    {")
    $linesP.Add("        type zeroGradient;")
    $linesP.Add("    }")
    $linesP.Add("}")
    Write-AsciiFile (Join-Path $CaseRoot "0\p") $linesP.ToArray()

    $linesT = New-Object System.Collections.Generic.List[string]
    Add-Lines $linesT (New-FoamHeader "volScalarField" "T" "0")
    $linesT.Add("dimensions [0 0 0 1 0 0 0];")
    $linesT.Add("")
    $linesT.Add("internalField uniform $(Format-F64 $Temperature);")
    $linesT.Add("")
    $linesT.Add("boundaryField")
    $linesT.Add("{")
    $linesT.Add("    inlet")
    $linesT.Add("    {")
    $linesT.Add("        type fixedValue;")
    $linesT.Add("        value uniform $(Format-F64 $Temperature);")
    $linesT.Add("    }")
    $linesT.Add("    outlet")
    $linesT.Add("    {")
    $linesT.Add("        type zeroGradient;")
    $linesT.Add("    }")
    $linesT.Add("    wall")
    $linesT.Add("    {")
    $linesT.Add("        type fixedValue;")
    $linesT.Add("        value uniform $(Format-F64 $WallTemperature);")
    $linesT.Add("    }")
    $linesT.Add("}")
    Write-AsciiFile (Join-Path $CaseRoot "0\T") $linesT.ToArray()

    $transportLines = New-Object System.Collections.Generic.List[string]
    Add-Lines $transportLines (New-FoamHeader "dictionary" "transportProperties" "constant")
    Add-Lines $transportLines @(
        "transportModel Newtonian;",
        "",
        "rho [1 -3 0 0 0 0 0] $(Format-F64 $rho);",
        "mu [1 -1 -1 0 0 0 0] $(Format-F64 $mu);",
        "nu [0 2 -1 0 0 0 0] $(Format-F64 $nu);",
        "",
        "Cp [0 2 -2 -1 0 0 0] $(Format-F64 $cp);",
        "k [1 1 -3 -1 0 0 0] $(Format-F64 $kThermal);",
        "Pr [0 0 0 0 0 0 0] $(Format-F64 $pr);"
    )
    Write-AsciiFile (Join-Path $CaseRoot "constant\transportProperties") $transportLines.ToArray()

    $summary = Read-FerrumMeshSummary $CaseRoot

    return [pscustomobject][ordered]@{
        inletVelocityScale = $inletVelocityScale
        inletFaces = $patches["inlet"].nFaces
        outletFaces = $patches["outlet"].nFaces
        wallFaces = $patches["wall"].nFaces
        cells = $summary.cells
        points = $summary.points
        faces = $summary.faces
    }
}

function Invoke-Gmsh($Gmsh, [string]$GeoFile, [string]$MeshFile, $Variant, [string]$LogPath) {
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $MeshFile) | Out-Null
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $LogPath) | Out-Null
    $setNumbers = [ordered]@{
        radius = $radius
        length = $Length
        axial_cells = $Variant.axialCells
        lc_center = $Variant.lcCenter
        lc_wall = $Variant.lcWall
        bl_hwall = $Variant.blHwall
        bl_hfar = $Variant.blHfar
        bl_thickness = $Variant.blThickness
        bl_ratio = $Variant.blRatio
        bl_layers = $Variant.blLayers
    }

    $args = New-Object System.Collections.Generic.List[string]
    foreach ($entry in $setNumbers.GetEnumerator()) {
        $args.Add("-setnumber") | Out-Null
        $args.Add($entry.Key) | Out-Null
        $args.Add((Format-F64 ([double]$entry.Value))) | Out-Null
    }
    $args.Add("-3") | Out-Null
    $args.Add("`"$GeoFile`"") | Out-Null
    $args.Add("-format") | Out-Null
    $args.Add("msh2") | Out-Null
    $args.Add("-o") | Out-Null
    $args.Add("`"$MeshFile`"") | Out-Null

    $process = Start-Process -FilePath $Gmsh -ArgumentList $args.ToArray() -Wait -PassThru -WindowStyle Hidden
    $logLines = @(
        "gmsh=$Gmsh",
        "geo=$GeoFile",
        "mesh=$MeshFile",
        "variant=$($Variant.name)",
        "exitCode=$($process.ExitCode)",
        "args=$($args.ToArray() -join ' ')"
    )
    Write-AsciiFile $LogPath $logLines
    if ($null -ne $process.ExitCode -and $process.ExitCode -ne 0) {
        throw "gmsh failed for variant '$($Variant.name)' with exit code $($process.ExitCode)"
    }
    if (!(Test-Path -LiteralPath $MeshFile)) {
        throw "gmsh did not write mesh file '$MeshFile'"
    }
}

function Invoke-FerrumCommand([string]$BinName, [string[]]$Arguments, [string]$LogPath) {
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $LogPath) | Out-Null
    $exe = Join-Path $RepoRoot "target\debug\$BinName.exe"
    if (Test-Path -LiteralPath $exe) {
        $command = $exe
        $commandArgs = $Arguments
    } else {
        $command = "cargo"
        $commandArgs = @("run", "-p", "ferrum-cli", "--bin", $BinName, "--") + $Arguments
    }

    $script:commandExitCode = $null
    $elapsed = Measure-Command {
        & $command @commandArgs *> $LogPath
        $script:commandExitCode = $LASTEXITCODE
    }
    $exitCode = if ($null -eq $script:commandExitCode) { 0 } else { $script:commandExitCode }
    if ($exitCode -ne 0) {
        throw "$BinName failed with exit code $exitCode. See $LogPath"
    }
    return $elapsed.TotalSeconds
}

function Read-JsonFile([string]$Path) {
    if (!(Test-Path -LiteralPath $Path)) {
        return $null
    }
    return Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
}

function Write-StudyMarkdown($Path, $Rows, $Summary) {
    $lines = New-Object System.Collections.Generic.List[string]
    $lines.Add("# Gmsh Pipe Mesh Study")
    $lines.Add("")
    $lines.Add('Gmsh generates every mesh in this study. FerrumCFD imports the same `.msh` files that are later used to build OpenFOAM benchmark cases.')
    $lines.Add("")
    $lines.Add("OpenFOAM SIMPLE steps per variant: $($Summary.openFoamSteps)")
    $lines.Add("Ferrum linear solver: $($Summary.ferrumLinearSolver), tolerance: $($Summary.ferrumSolveTolerance), max iterations: $($Summary.ferrumMaxIterations)")
    $lines.Add("")
    $lines.Add("## Variants")
    $lines.Add("")
    $lines.Add("| Variant | Axial | Cells | Points | Inlet faces | Ferrum check [s] | Ferrum deltaP [Pa] | Ferrum error | Ferrum solve [s] | OpenFOAM deltaP [Pa] | OpenFOAM error | OpenFOAM wall [s] |")
    $lines.Add("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    foreach ($row in $Rows) {
        $lines.Add("| $($row.variant) | $($row.gmsh.axialCells) | $($row.mesh.cells) | $($row.mesh.points) | $($row.mesh.inletFaces) | $(Format-NullableNumber $row.timings.checkFerrumMeshSeconds "G6") | $(Format-NullableNumber $row.ferrum.pressureDropFromMeanPa "G8") | $(Format-NullablePercent $row.ferrum.relativePressureDropErrorToAnalytic) | $(Format-NullableNumber $row.ferrum.solveWallClockSeconds "G6") | $(Format-NullableNumber $row.openFoam.deltaPPa "G8") | $(Format-NullablePercent $row.openFoam.relativeErrorToAnalytic) | $(Format-NullableNumber $row.openFoam.wallClockSeconds "G6") |")
    }
    $lines.Add("")
    $lines.Add("## Files")
    $lines.Add("")
    $lines.Add("- Summary JSON: " + '`' + $Summary.summaryJson + '`')
    $lines.Add("- This report: " + '`' + $Summary.reportFile + '`')
    $lines.Add("")
    $lines.Add("## Notes")
    $lines.Add("")
    $lines.Add("- This is the mesh-study path for validation. It does not make OpenFOAM part of normal FerrumCFD use.")
    $lines.Add("- Pressure loss is compared to Hagen-Poiseuille in SI units.")
    $lines.Add('- For imported Gmsh cases, OpenFOAM pressure loss is sampled by averaging cells adjacent to the `inlet` and `outlet` patches.')
    $lines.Add("- FerrumCFD runs the source-driven Poiseuille benchmark on the imported mesh; the full pressure-velocity solver is still future work.")

    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Path) | Out-Null
    Set-Content -LiteralPath $Path -Value $lines -Encoding UTF8
}

$targetRoot = Join-Path $RepoRoot "target"
if (!(Test-IsPathUnder $StudyRoot $targetRoot)) {
    throw "StudyRoot must be inside the repository target directory: $targetRoot"
}

$casesRoot = Join-Path $StudyRoot "cases"
$meshesRoot = Join-Path $StudyRoot "meshes"
$openFoamRoot = Join-Path $StudyRoot "openfoam"
$resultsRoot = Join-Path $StudyRoot "results"
$logsRoot = Join-Path $StudyRoot "logs"
New-Item -ItemType Directory -Force -Path $casesRoot, $meshesRoot, $openFoamRoot, $resultsRoot, $logsRoot | Out-Null

$benchmarkProperties = Join-Path $StudyRoot "pipeBenchmark"
$benchmarkLines = New-Object System.Collections.Generic.List[string]
Add-Lines $benchmarkLines (New-FoamHeader "dictionary" "pipeBenchmark" "benchmark")
$benchmarkLines.Add('description "External reference inputs for the generated Gmsh pipe mesh study";')
$benchmarkLines.Add("")
$benchmarkLines.Add("geometry")
$benchmarkLines.Add("{")
$benchmarkLines.Add("    length [0 1 0 0 0 0 0] $(Format-F64 $Length);")
$benchmarkLines.Add("    diameter [0 1 0 0 0 0 0] $(Format-F64 $Diameter);")
$benchmarkLines.Add("}")
$benchmarkLines.Add("")
$benchmarkLines.Add("water")
$benchmarkLines.Add("{")
$benchmarkLines.Add("    rho [1 -3 0 0 0 0 0] $(Format-F64 $rho);")
$benchmarkLines.Add("    mu [1 -1 -1 0 0 0 0] $(Format-F64 $mu);")
$benchmarkLines.Add("}")
$benchmarkLines.Add("")
$benchmarkLines.Add("flowReference")
$benchmarkLines.Add("{")
$benchmarkLines.Add("    meanVelocity [0 1 -1 0 0 0 0] $(Format-F64 $MeanVelocity);")
$benchmarkLines.Add("    pressureLossModel HagenPoiseuille;")
$benchmarkLines.Add("    expectedDeltaP [1 -1 -2 0 0 0 0] $(Format-F64 $deltaP);")
$benchmarkLines.Add("}")
Write-AsciiFile $benchmarkProperties $benchmarkLines.ToArray()

$gmsh = Resolve-GmshExecutable $GmshExe
$runOpenFoam = Join-Path $PSScriptRoot "run_openfoam_laminar_pipe.ps1"
$compare = Join-Path $PSScriptRoot "compare_laminar_pipe.ps1"
$sourceSystem = Join-Path $RepoRoot "examples\laminar_pipe\system"

$variants = @(
    [pscustomobject][ordered]@{ name = "coarse"; axialCells = 16; lcCenter = 0.0060; lcWall = 0.0018; blHwall = 0.00070; blHfar = 0.0045; blThickness = 0.0015; blRatio = 1.25; blLayers = 2 },
    [pscustomobject][ordered]@{ name = "medium"; axialCells = 32; lcCenter = 0.0040; lcWall = 0.0012; blHwall = 0.00045; blHfar = 0.0030; blThickness = 0.0015; blRatio = 1.25; blLayers = 2 },
    [pscustomobject][ordered]@{ name = "fine"; axialCells = 48; lcCenter = 0.0028; lcWall = 0.0008; blHwall = 0.00028; blHfar = 0.0020; blThickness = 0.0015; blRatio = 1.25; blLayers = 2 }
)

if ($VariantName.Count -gt 0) {
    $wanted = @{}
    foreach ($name in $VariantName) {
        $wanted[$name.ToLowerInvariant()] = $true
    }
    $variants = @($variants | Where-Object { $wanted.ContainsKey($_.name.ToLowerInvariant()) })
    if ($variants.Count -eq 0) {
        throw "none of the requested variants were found"
    }
}

$rows = New-Object System.Collections.Generic.List[object]
foreach ($variant in $variants) {
    $meshFile = Join-Path $meshesRoot "$($variant.name).msh"
    $caseRoot = Join-Path $casesRoot $variant.name
    $openFoamWorkDir = Join-Path $openFoamRoot $variant.name
    $openFoamJson = Join-Path $resultsRoot "$($variant.name).openfoam.json"
    $compareJson = Join-Path $resultsRoot "$($variant.name).compare.json"
    $compareReport = Join-Path $resultsRoot "$($variant.name).compare.md"
    $planJson = Join-Path $resultsRoot "$($variant.name).ferrum_plan.json"
    $gmshLog = Join-Path $logsRoot "$($variant.name).gmsh.log"
    $importLog = Join-Path $logsRoot "$($variant.name).import.log"
    $checkLog = Join-Path $logsRoot "$($variant.name).check.log"
    $openFoamLog = Join-Path $logsRoot "$($variant.name).openfoam_driver.log"
    $compareLog = Join-Path $logsRoot "$($variant.name).compare.log"

    Write-Output "variant $($variant.name): generating Gmsh mesh"
    Remove-DirectoryIfExists $caseRoot $StudyRoot
    Invoke-Gmsh $gmsh $GeoFile $meshFile $variant $gmshLog

    Write-Output "variant $($variant.name): importing into FerrumCFD"
    New-Item -ItemType Directory -Force -Path $caseRoot | Out-Null
    $importSeconds = Invoke-FerrumCommand "gmshToFerrum" @($meshFile, "-case", $caseRoot) $importLog
    Copy-Item -LiteralPath $sourceSystem -Destination $caseRoot -Recurse -Force
    $caseInputSummary = Write-GmshPipeCaseInputs $caseRoot $variant $meshFile
    $checkSeconds = Invoke-FerrumCommand "checkFerrumMesh" @("-case", $caseRoot) $checkLog

    $openFoamElapsed = $null
    if ($SkipOpenFoam) {
        if (Test-Path -LiteralPath $openFoamJson) {
            Remove-Item -LiteralPath $openFoamJson -Force
        }
    } else {
        Write-Output "variant $($variant.name): running OpenFOAM reference"
        $openFoamArgs = @{
            CaseRoot = $caseRoot
            WorkDir = $openFoamWorkDir
            OutFile = $openFoamJson
            BenchmarkProperties = $benchmarkProperties
            Mode = $Mode
            EndTime = $OpenFoamSteps
            WriteInterval = $OpenFoamSteps
        }
        if ($RequireOpenFoam) {
            $openFoamArgs.RequireOpenFoam = $true
        }
        $openFoamElapsed = (Measure-Command { & $runOpenFoam @openFoamArgs *> $openFoamLog }).TotalSeconds
    }

    Write-Output "variant $($variant.name): running Ferrum/OpenFOAM comparison"
    $compareElapsed = (Measure-Command {
            $compareArgs = @{
                CaseRoot = $caseRoot
                OpenFoamJson = $openFoamJson
                FerrumPlanJson = $planJson
                OutFile = $compareJson
                ReportFile = $compareReport
                BenchmarkProperties = $benchmarkProperties
                FerrumLinearSolver = $FerrumLinearSolver
                FerrumSolveTolerance = $FerrumSolveTolerance
                FerrumMaxIterations = $FerrumMaxIterations
            }
            if ($SkipFerrumSolve) {
                $compareArgs.SkipFerrumSolve = $true
            }
            & $compare @compareArgs *> $compareLog
        }).TotalSeconds

    $comparison = Read-JsonFile $compareJson
    $openFoam = Read-JsonFile $openFoamJson
    $pressureLoss = if ($null -ne $openFoam -and $null -ne $openFoam.openFoam.pressureLoss) { $openFoam.openFoam.pressureLoss } else { $null }
    $ferrumSolve = if ($null -ne $comparison -and $null -ne $comparison.ferrum.solve) { $comparison.ferrum.solve } else { $null }
    $ferrumResult = if ($null -ne $ferrumSolve) { $ferrumSolve.result } else { $null }

    $rows.Add([pscustomobject][ordered]@{
            variant = $variant.name
            caseRoot = $caseRoot
            meshFile = $meshFile
            gmsh = [pscustomobject][ordered]@{
                axialCells = $variant.axialCells
                lcCenter = $variant.lcCenter
                lcWall = $variant.lcWall
                blHwall = $variant.blHwall
                blHfar = $variant.blHfar
                blThickness = $variant.blThickness
                blLayers = $variant.blLayers
            }
            mesh = [pscustomobject][ordered]@{
                cells = $caseInputSummary.cells
                points = $caseInputSummary.points
                faces = $caseInputSummary.faces
                inletFaces = $caseInputSummary.inletFaces
                outletFaces = $caseInputSummary.outletFaces
                wallFaces = $caseInputSummary.wallFaces
                inletVelocityScale = $caseInputSummary.inletVelocityScale
            }
            analytic = [pscustomobject][ordered]@{
                reynolds = $reynolds
                deltaPPa = $deltaP
            }
            ferrum = [pscustomobject][ordered]@{
                preflightStatus = if ($null -ne $comparison) { $comparison.benchmarkStatus.ferrumPreflight } else { "missing" }
                solveStatus = if ($null -ne $comparison) { $comparison.benchmarkStatus.ferrumSolverComparison } else { "missing" }
                pressureDropFromMeanPa = if ($null -ne $ferrumResult) { $ferrumResult.pressureDropFromMeanPa } else { $null }
                relativePressureDropErrorToAnalytic = if ($null -ne $ferrumResult) { $ferrumResult.relativePressureDropErrorToAnalytic } else { $null }
                meanVelocityMps = if ($null -ne $ferrumResult) { $ferrumResult.meanVelocityMps } else { $null }
                relativeMeanVelocityErrorToAnalytic = if ($null -ne $ferrumResult) { $ferrumResult.relativeMeanVelocityErrorToAnalytic } else { $null }
                solveWallClockSeconds = if ($null -ne $ferrumSolve) { $ferrumSolve.solveWallClockSeconds } else { $null }
                commandWallClockSeconds = if ($null -ne $ferrumSolve) { $ferrumSolve.commandWallClockSeconds } else { $null }
                iterations = if ($null -ne $ferrumSolve) { $ferrumSolve.iterations } else { $null }
                converged = if ($null -ne $ferrumSolve) { $ferrumSolve.converged } else { $null }
                residualNorm = if ($null -ne $ferrumSolve) { $ferrumSolve.residualNorm } else { $null }
                preflightWallClockSeconds = if ($null -ne $comparison) { $comparison.ferrum.preflight.wallClockSeconds } else { $null }
                planJson = $planJson
                resultJson = $compareJson
                compareReport = $compareReport
            }
            openFoam = [pscustomobject][ordered]@{
                status = if ($null -ne $comparison) { $comparison.benchmarkStatus.openFoamReference } else { "missing" }
                deltaPPa = if ($null -ne $pressureLoss) { $pressureLoss.deltaPPa } else { $null }
                relativeErrorToAnalytic = if ($null -ne $pressureLoss) { $pressureLoss.relativeErrorToAnalytic } else { $null }
                pressureLossMethod = if ($null -ne $pressureLoss) { $pressureLoss.method } else { $null }
                wallClockSeconds = if ($null -ne $openFoam) { $openFoam.openFoam.wallClockSeconds } else { $null }
                driverWallClockSeconds = $openFoamElapsed
                resultJson = $openFoamJson
            }
            timings = [pscustomobject][ordered]@{
                gmshImportSeconds = $importSeconds
                checkFerrumMeshSeconds = $checkSeconds
                compareWallClockSeconds = $compareElapsed
            }
            logs = [pscustomobject][ordered]@{
                gmsh = $gmshLog
                import = $importLog
                check = $checkLog
                openFoamDriver = $openFoamLog
                compare = $compareLog
            }
        }) | Out-Null
}

$summaryJson = Join-Path $StudyRoot "gmsh_pipe_mesh_study.json"
$reportFile = Join-Path $StudyRoot "gmsh_pipe_mesh_study.md"
$rowArray = @($rows.ToArray())
$summary = [pscustomobject][ordered]@{
    case = "gmsh_pipe_mesh_study"
    generatedAt = Get-Date -Format "o"
    geoFile = $GeoFile
    gmsh = $gmsh
    benchmarkProperties = $benchmarkProperties
    openFoamMode = if ($SkipOpenFoam) { "skipped" } else { $Mode }
    openFoamSteps = if ($SkipOpenFoam) { 0 } else { $OpenFoamSteps }
    ferrumSolve = if ($SkipFerrumSolve) { "skipped" } else { "poiseuille" }
    ferrumLinearSolver = $FerrumLinearSolver
    ferrumSolveTolerance = $FerrumSolveTolerance
    ferrumMaxIterations = $FerrumMaxIterations
    units = [pscustomobject][ordered]@{
        default = "SI"
        length = "m"
        pressure = "Pa"
        temperature = "K"
        velocity = "m/s"
        openFoamPressure = "kinematic m2/s2 converted to Pa"
    }
    variants = $rowArray
    summaryJson = $summaryJson
    reportFile = $reportFile
}

$summary | ConvertTo-Json -Depth 14 | Set-Content -LiteralPath $summaryJson -Encoding UTF8
Write-StudyMarkdown -Path $reportFile -Rows $rowArray -Summary $summary

Write-Output "wrote Gmsh pipe mesh-study summary: $summaryJson"
Write-Output "wrote Gmsh pipe mesh-study report: $reportFile"
