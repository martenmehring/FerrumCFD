param(
    [string]$CaseRoot = "",
    [int]$AxialCells = 24,
    [int]$RadialCells = 6,
    [int]$AngularSectors = 32,
    [double]$Length = 1.0,
    [double]$Diameter = 0.02,
    [double]$MeanVelocity = 0.02,
    [double]$Temperature = 293.15,
    [double]$WallTemperature = 333.15
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($CaseRoot)) {
    $CaseRoot = Join-Path $RepoRoot "tutorials\steadyIncompressible\laminarPipe\ferrum\case"
}
if ($AxialCells -le 0 -or $RadialCells -le 0 -or $AngularSectors -lt 3) {
    throw "AxialCells and RadialCells must be positive; AngularSectors must be at least 3"
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

function Write-AsciiFile([string]$Path, [string[]]$Lines) {
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Path) | Out-Null
    Set-Content -LiteralPath $Path -Value $Lines -Encoding ASCII
}

function New-Point([double]$X, [double]$Y, [double]$Z) {
    [pscustomobject]@{ x = $X; y = $Y; z = $Z }
}

function Add-Point($Points, [double]$X, [double]$Y, [double]$Z) {
    $Points.Add((New-Point $X $Y $Z)) | Out-Null
    return $Points.Count - 1
}

function Get-PointIndex($Centers, $Rings, [int]$Axial, [int]$Radial, [int]$Sector, [int]$AngularSectors) {
    if ($Radial -eq 0) {
        return $Centers[$Axial]
    }
    $wrapped = (($Sector % $AngularSectors) + $AngularSectors) % $AngularSectors
    return $Rings[$Axial][$Radial - 1][$wrapped]
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
    return New-Point ($sx / $count) ($sy / $count) ($sz / $count)
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
    return New-Point $nx $ny $nz
}

function Get-FaceArea($Points, [int[]]$Indices) {
    $normal = Get-FaceNormal $Points $Indices
    return 0.5 * [Math]::Sqrt($normal.x * $normal.x + $normal.y * $normal.y + $normal.z * $normal.z)
}

function Get-OrientedFace($Points, [int[]]$Indices, $CellCentroid) {
    $faceCentroid = Get-Centroid $Points $Indices
    $normal = Get-FaceNormal $Points $Indices
    $dx = $faceCentroid.x - $CellCentroid.x
    $dy = $faceCentroid.y - $CellCentroid.y
    $dz = $faceCentroid.z - $CellCentroid.z
    $dot = $normal.x * $dx + $normal.y * $dy + $normal.z * $dz
    if ($dot -lt 0.0) {
        [array]::Reverse($Indices)
    }
    return $Indices
}

function Add-Cell($Points, $Faces, $FaceMap, $Cells, [int[][]]$FaceNodeLists) {
    $unique = New-Object System.Collections.Generic.HashSet[int]
    foreach ($faceNodes in $FaceNodeLists) {
        foreach ($node in $faceNodes) {
            $unique.Add($node) | Out-Null
        }
    }
    $cellNodes = [int[]]@($unique)
    $cellCentroid = Get-Centroid $Points $cellNodes
    $cellId = $Cells.Count
    $Cells.Add([pscustomobject]@{ id = $cellId; centroid = $cellCentroid; nodes = $cellNodes }) | Out-Null

    foreach ($faceNodes in $FaceNodeLists) {
        $oriented = Get-OrientedFace $Points ([int[]]$faceNodes.Clone()) $cellCentroid
        $keyNodes = [int[]]$oriented.Clone()
        [array]::Sort($keyNodes)
        $key = $keyNodes -join "_"
        if ($FaceMap.ContainsKey($key)) {
            $face = $Faces[$FaceMap[$key]]
            if ($null -ne $face.neighbour) {
                throw "face '$key' is shared by more than two cells"
            }
            $face.neighbour = $cellId
            continue
        }

        $faceCentroid = Get-Centroid $Points $oriented
        $Faces.Add([pscustomobject]@{
                nodes = $oriented
                owner = $cellId
                neighbour = $null
                centroid = $faceCentroid
                patch = $null
            }) | Out-Null
        $FaceMap[$key] = $Faces.Count - 1
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

function Add-Lines($Lines, $Values) {
    foreach ($value in $Values) {
        $Lines.Add([string]$value) | Out-Null
    }
}

function Write-Points($Path, $Points) {
    $lines = New-Object System.Collections.Generic.List[string]
    Add-Lines $lines (New-FoamHeader "vectorField" "points" "constant/polyMesh")
    $lines.Add("$($Points.Count)")
    $lines.Add("(")
    foreach ($point in $Points) {
        $lines.Add("    ($(Format-F64 $point.x) $(Format-F64 $point.y) $(Format-F64 $point.z))")
    }
    $lines.Add(")")
    Write-AsciiFile $Path $lines.ToArray()
}

function Write-Faces($Path, $Faces) {
    $lines = New-Object System.Collections.Generic.List[string]
    Add-Lines $lines (New-FoamHeader "faceList" "faces" "constant/polyMesh")
    $lines.Add("$($Faces.Count)")
    $lines.Add("(")
    foreach ($face in $Faces) {
        $lines.Add("    $($face.nodes.Count)($($face.nodes -join ' '))")
    }
    $lines.Add(")")
    Write-AsciiFile $Path $lines.ToArray()
}

function Write-Labels($Path, [string]$Object, [int[]]$Labels) {
    $lines = New-Object System.Collections.Generic.List[string]
    Add-Lines $lines (New-FoamHeader "labelList" $Object "constant/polyMesh")
    $lines.Add("$($Labels.Count)")
    $lines.Add("(")
    foreach ($label in $Labels) {
        $lines.Add("    $label")
    }
    $lines.Add(")")
    Write-AsciiFile $Path $lines.ToArray()
}

function Write-Boundary($Path, $PatchSummaries) {
    $lines = New-Object System.Collections.Generic.List[string]
    Add-Lines $lines (New-FoamHeader "polyBoundaryMesh" "boundary" "constant/polyMesh")
    $lines.Add("$($PatchSummaries.Count)")
    $lines.Add("(")
    foreach ($patch in $PatchSummaries) {
        $lines.Add("    $($patch.name)")
        $lines.Add("    {")
        $lines.Add("        type $($patch.type);")
        $lines.Add("        nFaces $($patch.faces);")
        $lines.Add("        startFace $($patch.startFace);")
        $lines.Add("    }")
    }
    $lines.Add(")")
    Write-AsciiFile $Path $lines.ToArray()
}

function Write-EmptyZoneFile($Path, [string]$ClassName, [string]$Object) {
    $lines = New-Object System.Collections.Generic.List[string]
    Add-Lines $lines (New-FoamHeader $ClassName $Object "constant/polyMesh")
    $lines.Add("0")
    $lines.Add("(")
    $lines.Add(")")
    Write-AsciiFile $Path $lines.ToArray()
}

function Write-VolScalarField($Path, [string]$Name, [string]$Dimensions, [double[]]$Values, [double]$OutletValue, [double]$WallValue, [string]$WallType) {
    $lines = New-Object System.Collections.Generic.List[string]
    Add-Lines $lines (New-FoamHeader "volScalarField" $Name "0")
    $lines.Add("dimensions $Dimensions;")
    $lines.Add("")
    $lines.Add("internalField nonuniform List<scalar>")
    $lines.Add("$($Values.Count)")
    $lines.Add("(")
    foreach ($value in $Values) {
        $lines.Add("    $(Format-F64 $value)")
    }
    $lines.Add(");")
    $lines.Add("")
    $lines.Add("boundaryField")
    $lines.Add("{")
    $lines.Add("    inlet")
    $lines.Add("    {")
    $lines.Add("        type zeroGradient;")
    $lines.Add("    }")
    $lines.Add("    outlet")
    $lines.Add("    {")
    $lines.Add("        type fixedValue;")
    $lines.Add("        value uniform $(Format-F64 $OutletValue);")
    $lines.Add("    }")
    $lines.Add("    wall")
    $lines.Add("    {")
    $lines.Add("        type $WallType;")
    if ($WallType -eq "fixedValue") {
        $lines.Add("        value uniform $(Format-F64 $WallValue);")
    }
    $lines.Add("    }")
    $lines.Add("}")
    Write-AsciiFile $Path $lines.ToArray()
}

$points = New-Object System.Collections.Generic.List[object]
$centers = New-Object System.Collections.Generic.List[int]
$rings = New-Object System.Collections.Generic.List[object]
for ($axial = 0; $axial -le $AxialCells; $axial++) {
    $x = $Length * [double]$axial / [double]$AxialCells
    $centers.Add((Add-Point $points $x 0.0 0.0)) | Out-Null
    $plane = New-Object System.Collections.Generic.List[object]
    for ($radial = 1; $radial -le $RadialCells; $radial++) {
        $ring = New-Object System.Collections.Generic.List[int]
        $r = $radius * [double]$radial / [double]$RadialCells
        for ($sector = 0; $sector -lt $AngularSectors; $sector++) {
            $theta = 2.0 * [Math]::PI * [double]$sector / [double]$AngularSectors
            $ring.Add((Add-Point $points $x ($r * [Math]::Cos($theta)) ($r * [Math]::Sin($theta)))) | Out-Null
        }
        $plane.Add($ring.ToArray()) | Out-Null
    }
    $rings.Add($plane.ToArray()) | Out-Null
}

$faces = New-Object System.Collections.Generic.List[object]
$faceMap = @{}
$cells = New-Object System.Collections.Generic.List[object]

for ($axial = 0; $axial -lt $AxialCells; $axial++) {
    for ($sector = 0; $sector -lt $AngularSectors; $sector++) {
        $next = ($sector + 1) % $AngularSectors
        $c0 = Get-PointIndex $centers $rings $axial 0 $sector $AngularSectors
        $a0 = Get-PointIndex $centers $rings $axial 1 $sector $AngularSectors
        $b0 = Get-PointIndex $centers $rings $axial 1 $next $AngularSectors
        $c1 = Get-PointIndex $centers $rings ($axial + 1) 0 $sector $AngularSectors
        $a1 = Get-PointIndex $centers $rings ($axial + 1) 1 $sector $AngularSectors
        $b1 = Get-PointIndex $centers $rings ($axial + 1) 1 $next $AngularSectors
        Add-Cell $points $faces $faceMap $cells @(
            [int[]]@($c0, $b0, $a0),
            [int[]]@($c1, $a1, $b1),
            [int[]]@($c0, $a0, $a1, $c1),
            [int[]]@($a0, $b0, $b1, $a1),
            [int[]]@($b0, $c0, $c1, $b1)
        )
    }

    for ($inner = 1; $inner -lt $RadialCells; $inner++) {
        $outer = $inner + 1
        for ($sector = 0; $sector -lt $AngularSectors; $sector++) {
            $next = ($sector + 1) % $AngularSectors
            $a0 = Get-PointIndex $centers $rings $axial $inner $sector $AngularSectors
            $b0 = Get-PointIndex $centers $rings $axial $inner $next $AngularSectors
            $c0 = Get-PointIndex $centers $rings $axial $outer $next $AngularSectors
            $d0 = Get-PointIndex $centers $rings $axial $outer $sector $AngularSectors
            $a1 = Get-PointIndex $centers $rings ($axial + 1) $inner $sector $AngularSectors
            $b1 = Get-PointIndex $centers $rings ($axial + 1) $inner $next $AngularSectors
            $c1 = Get-PointIndex $centers $rings ($axial + 1) $outer $next $AngularSectors
            $d1 = Get-PointIndex $centers $rings ($axial + 1) $outer $sector $AngularSectors
            Add-Cell $points $faces $faceMap $cells @(
                [int[]]@($a0, $b0, $c0, $d0),
                [int[]]@($a1, $d1, $c1, $b1),
                [int[]]@($a0, $a1, $b1, $b0),
                [int[]]@($d0, $c0, $c1, $d1),
                [int[]]@($a0, $d0, $d1, $a1),
                [int[]]@($b0, $b1, $c1, $c0)
            )
        }
    }
}

$epsilon = 1.0e-10
foreach ($face in $faces) {
    if ($null -ne $face.neighbour) {
        continue
    }
    if ([Math]::Abs($face.centroid.x) -lt $epsilon) {
        $face.patch = "inlet"
    } elseif ([Math]::Abs($face.centroid.x - $Length) -lt $epsilon) {
        $face.patch = "outlet"
    } else {
        $face.patch = "wall"
    }
}

$internalFaces = @($faces | Where-Object { $null -ne $_.neighbour })
$patchOrder = @("inlet", "outlet", "wall")
$orderedFaces = New-Object System.Collections.Generic.List[object]
$orderedFaces.AddRange($internalFaces)
$patchSummaries = New-Object System.Collections.Generic.List[object]
$patchFaceLists = @{}
foreach ($patchName in $patchOrder) {
    $startFace = $orderedFaces.Count
    $patchFaces = @($faces | Where-Object { $null -eq $_.neighbour -and $_.patch -eq $patchName })
    $patchFaceLists[$patchName] = $patchFaces
    $orderedFaces.AddRange($patchFaces)
    $patchType = if ($patchName -eq "wall") { "wall" } else { "patch" }
    $patchSummaries.Add([pscustomobject]@{
            name = $patchName
            type = $patchType
            faces = $patchFaces.Count
            startFace = $startFace
        }) | Out-Null
}

$polyMesh = Join-Path $CaseRoot "constant\polyMesh"
New-Item -ItemType Directory -Force -Path $polyMesh | Out-Null
Write-Points (Join-Path $polyMesh "points") $points
Write-Faces (Join-Path $polyMesh "faces") $orderedFaces
Write-Labels (Join-Path $polyMesh "owner") "owner" ([int[]]@($orderedFaces | ForEach-Object { $_.owner }))
Write-Labels (Join-Path $polyMesh "neighbour") "neighbour" ([int[]]@($internalFaces | ForEach-Object { $_.neighbour }))
Write-Boundary (Join-Path $polyMesh "boundary") $patchSummaries
Write-EmptyZoneFile (Join-Path $polyMesh "cellZones") "cellZoneMesh" "cellZones"
Write-EmptyZoneFile (Join-Path $polyMesh "faceZones") "faceZoneMesh" "faceZones"

$pValues = [double[]]@($cells | ForEach-Object {
        $x = $_.centroid.x
        $deltaP * (1.0 - ($x / $Length))
    })
$zero = 0.0
Write-VolScalarField (Join-Path $CaseRoot "0\p") "p" "[1 -1 -2 0 0 0 0]" $pValues $zero $zero "zeroGradient"

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
$inletFaces = @($patchFaceLists["inlet"])
$inletProfileValues = New-Object System.Collections.Generic.List[object]
$inletArea = 0.0
$inletUnscaledFlow = 0.0
foreach ($face in $inletFaces) {
    $area = Get-FaceArea $points $face.nodes
    $r2 = $face.centroid.y * $face.centroid.y + $face.centroid.z * $face.centroid.z
    $profile = 2.0 * $MeanVelocity * (1.0 - ($r2 / ($radius * $radius)))
    if ($profile -lt 0.0) {
        $profile = 0.0
    }
    $inletArea += $area
    $inletUnscaledFlow += $profile * $area
    $inletProfileValues.Add([pscustomobject]@{ profile = $profile; area = $area }) | Out-Null
}
if ([double]::IsNaN($inletUnscaledFlow) -or [double]::IsInfinity($inletUnscaledFlow) -or
    $inletUnscaledFlow -le 0.0 -or [double]::IsNaN($inletArea) -or
    [double]::IsInfinity($inletArea) -or $inletArea -le 0.0) {
    throw "generated inlet patch must have positive finite area and unscaled flow"
}
$inletVelocityScale = ($MeanVelocity * $inletArea) / $inletUnscaledFlow
if ([double]::IsNaN($inletVelocityScale) -or [double]::IsInfinity($inletVelocityScale) -or
    $inletVelocityScale -le 0.0) {
    throw "generated inlet velocity scale must be positive and finite"
}
$linesU.Add("        value nonuniform List<vector>")
$linesU.Add("        $($inletFaces.Count)")
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

$transport = @(
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
$transportLines = New-Object System.Collections.Generic.List[string]
Add-Lines $transportLines (New-FoamHeader "dictionary" "transportProperties" "constant")
Add-Lines $transportLines $transport
Write-AsciiFile (Join-Path $CaseRoot "constant\transportProperties") $transportLines.ToArray()

$summaryLines = New-Object System.Collections.Generic.List[string]
$summaryLines.Add("FerrumCFD mesh summary")
$summaryLines.Add("source=generated structured circular pipe")
$summaryLines.Add("case=tutorials\steadyIncompressible\laminarPipe\ferrum\case")
$summaryLines.Add("points=$($points.Count)")
$summaryLines.Add("cells=$($cells.Count)")
$summaryLines.Add("faces=$($orderedFaces.Count)")
$summaryLines.Add("internal_faces=$($internalFaces.Count)")
$summaryLines.Add("boundary_faces=$($orderedFaces.Count - $internalFaces.Count)")
$summaryLines.Add("unmatched_boundary_faces=0")
$summaryLines.Add("duplicate_boundary_faces=0")
$summaryLines.Add("non_manifold_faces=0")
$summaryLines.Add("")
$summaryLines.Add("[patches]")
$tag = 1
foreach ($patch in $patchSummaries) {
    $summaryLines.Add("$($patch.name) type=$($patch.type) tag=$tag faces=$($patch.faces) startFace=$($patch.startFace)")
    $tag++
}
$summaryLines.Add("")
$summaryLines.Add("[face_zones]")
$summaryLines.Add("")
$summaryLines.Add("[cell_zones]")
Write-AsciiFile (Join-Path $CaseRoot "constant\ferrumMeshSummary.txt") $summaryLines.ToArray()

Write-Output "generated laminar pipe case: $CaseRoot"
Write-Output "mesh: cells=$($cells.Count) points=$($points.Count) faces=$($orderedFaces.Count) internalFaces=$($internalFaces.Count)"
Write-Output "reference: Re=$(Format-F64 $reynolds) deltaP=$(Format-F64 $deltaP) Pa"
