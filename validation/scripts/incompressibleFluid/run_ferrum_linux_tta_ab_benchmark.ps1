param(
    [Parameter(Mandatory = $true)]
    [string]$BaselineRef,
    [Parameter(Mandatory = $true)]
    [string]$CandidateRef,
    [ValidateSet("relTol", "simplec")]
    [string]$Experiment = "relTol",
    [string[]]$ExpectedChangedPaths = @(
        "applications/legacy/ferrumCli/src/lib.rs",
        "src/ferrumMesh/src/flow.rs"
    ),
    [double]$CandidatePressureRelTol = 0.05,
    [double]$CandidateMomentumRelTol = 0.05,
    [int]$MaxSimpleIterations = 2000,
    [int]$WarmupRuns = 2,
    [int]$MeasuredRuns = 10,
    [ValidateSet("gamg", "pcg")]
    [string]$PressureSolver = "gamg",
    [ValidateSet("all", "pipe", "channel")]
    [string]$CaseName = "all",
    [string]$Distro = "Ubuntu-22.04",
    [string]$CpuSet = "2",
    [ValidateSet("portable", "native", "native-codegen1", "native-thin-lto", "native-fat-lto")]
    [string]$BuildVariant = "native",
    [string]$RustToolchain = "1.94.0",
    [double]$MaxContinuityRatio = 5.0,
    [double]$MaxVelocityRelativeL2 = 1e-5,
    [double]$MaxVelocityRelativeLinf = 5e-5,
    [double]$MaxPressureGaugeRelativeL2 = 1e-4,
    [double]$MaxPressureGaugeRelativeLinf = 5e-4,
    [double]$MaxPressureDropRelativeDifference = 1e-4,
    [double]$MaxFlowRelativeDifference = 1e-4,
    [double]$MinimumWorkReduction = 0.05,
    [double]$MaximumMedianRatio = 0.98,
    [string]$OutRoot = "",
    [switch]$PreflightOnly,
    [switch]$KeepWslWorkspace
)

$ErrorActionPreference = "Stop"
$WorkerPath = Join-Path $PSScriptRoot "run_ferrum_linux_tta_ab_worker.sh"
$CommonHelperPath = Join-Path $PSScriptRoot "matched_cpu_solver_common.ps1"
$ControlSourceDefinitions = @(
    [pscustomobject][ordered]@{ name = "run_ferrum_linux_tta_ab_benchmark.ps1"; path = [string]$PSCommandPath },
    [pscustomobject][ordered]@{ name = "run_ferrum_linux_tta_ab_worker.sh"; path = $WorkerPath },
    [pscustomobject][ordered]@{ name = "matched_cpu_solver_common.ps1"; path = $CommonHelperPath }
)
$ControlBindings = @()
foreach ($control in $ControlSourceDefinitions) {
    if ([string]::IsNullOrWhiteSpace($control.path) -or !(Test-Path -LiteralPath $control.path -PathType Leaf)) {
        throw "TTA control file was not found before execution: $($control.name)"
    }
    $ControlBindings += [pscustomobject][ordered]@{
        name = $control.name
        sourcePath = [System.IO.Path]::GetFullPath($control.path)
        sha256 = (Get-FileHash -LiteralPath $control.path -Algorithm SHA256).Hash.ToLowerInvariant()
    }
}
function Assert-ControlSourcesUnchanged([string]$Phase) {
    foreach ($binding in $ControlBindings) {
        $actual = (Get-FileHash -LiteralPath $binding.sourcePath -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($actual -ne $binding.sha256) { throw "TTA control '$($binding.name)' changed $Phase" }
    }
}
Assert-ControlSourcesUnchanged "before helper load"
. $CommonHelperPath
Assert-ControlSourcesUnchanged "while loading the common helper"

function Get-TtaPositiveMaxResidentSetKiB($Timing, [string]$Description) {
    if ($null -eq $Timing -or $Timing -isnot [pscustomobject]) {
        throw "$Description GNU time record must be an object"
    }
    $property = $Timing.PSObject.Properties["maxResidentSetKiB"]
    if ($null -eq $property -or $null -eq $property.Value) {
        throw "$Description GNU time record is missing maxResidentSetKiB"
    }
    $value = [long]$property.Value
    if ($value -le 0) { throw "$Description GNU time maxResidentSetKiB must be positive" }
    return $value
}

$gnuTimeSelfTestPath = [System.IO.Path]::GetTempFileName()
try {
    [System.IO.File]::WriteAllText(
        $gnuTimeSelfTestPath,
        "elapsed_s=1.25`nuser_s=1.00`nsystem_s=0.25`nmax_rss_kb=9876`nexit=0`n",
        [System.Text.UTF8Encoding]::new($false)
    )
    $gnuTimeSelfTest = Read-MatchedGnuTime $gnuTimeSelfTestPath
    if ((Get-TtaPositiveMaxResidentSetKiB $gnuTimeSelfTest "GNU time parser self-test") -ne 9876) {
        throw "GNU time maxResidentSetKiB parser self-test failed"
    }
} finally {
    Remove-Item -LiteralPath $gnuTimeSelfTestPath -Force -ErrorAction SilentlyContinue
}

$RepoRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
$TargetRoot = Join-Path $RepoRoot "target"
$Experiment = $Experiment.ToLowerInvariant()
$simplecExperiment = $Experiment -ceq "simplec"
if ($simplecExperiment) {
    if ($BaselineRef -cne $CandidateRef) { throw "simplec requires BaselineRef and CandidateRef to be the same exact ref text" }
    if ($PSBoundParameters.ContainsKey("ExpectedChangedPaths") -and $ExpectedChangedPaths.Count -ne 0) {
        throw "simplec requires an empty ExpectedChangedPaths set"
    }
    if ($PSBoundParameters.ContainsKey("CandidatePressureRelTol") -and $CandidatePressureRelTol -ne 0.0) {
        throw "simplec requires candidate p relTol to be exactly zero"
    }
    if ($PSBoundParameters.ContainsKey("CandidateMomentumRelTol") -and $CandidateMomentumRelTol -ne 0.0) {
        throw "simplec requires candidate U relTol to be exactly zero"
    }
    $CandidatePressureRelTol = 0.0
    $CandidateMomentumRelTol = 0.0
    [string[]]$effectiveChangedPaths = @()
} else {
    [string[]]$effectiveChangedPaths = @($ExpectedChangedPaths)
}
$baselineSimpleConsistent = $false
$candidateSimpleConsistent = $simplecExperiment
$buildPolicyMode = if ($simplecExperiment) { "shared-single-build" } else { "separate-builds" }
if ($WarmupRuns -lt 2) { throw "WarmupRuns must be at least two" }
if ($MeasuredRuns -lt 10 -or ($MeasuredRuns % 2) -ne 0) { throw "MeasuredRuns must be an even integer of at least ten" }
if ($MaxSimpleIterations -lt 1) { throw "MaxSimpleIterations must be positive" }
if ($CpuSet -notmatch "^[0-9]+([,-][0-9]+)*$") { throw "CpuSet is invalid: $CpuSet" }
if (!$simplecExperiment -and ($effectiveChangedPaths.Count -eq 0 -or @($effectiveChangedPaths | Sort-Object -Unique).Count -ne $effectiveChangedPaths.Count)) {
    throw "ExpectedChangedPaths must be a non-empty unique path set"
}
foreach ($path in $effectiveChangedPaths) {
    if ([string]::IsNullOrWhiteSpace($path) -or $path.Contains("\") -or $path.StartsWith("/") -or $path -match "(^|/)\.\.(/|$)") {
        throw "ExpectedChangedPaths contains an unsafe repository-relative path: $path"
    }
}
foreach ($entry in @(
    @("CandidatePressureRelTol", $CandidatePressureRelTol, 0.0, [double]::PositiveInfinity),
    @("CandidateMomentumRelTol", $CandidateMomentumRelTol, 0.0, [double]::PositiveInfinity),
    @("MaxContinuityRatio", $MaxContinuityRatio, 1.0, [double]::PositiveInfinity),
    @("MaxVelocityRelativeL2", $MaxVelocityRelativeL2, 0.0, [double]::PositiveInfinity),
    @("MaxVelocityRelativeLinf", $MaxVelocityRelativeLinf, 0.0, [double]::PositiveInfinity),
    @("MaxPressureGaugeRelativeL2", $MaxPressureGaugeRelativeL2, 0.0, [double]::PositiveInfinity),
    @("MaxPressureGaugeRelativeLinf", $MaxPressureGaugeRelativeLinf, 0.0, [double]::PositiveInfinity),
    @("MaxPressureDropRelativeDifference", $MaxPressureDropRelativeDifference, 0.0, [double]::PositiveInfinity),
    @("MaxFlowRelativeDifference", $MaxFlowRelativeDifference, 0.0, [double]::PositiveInfinity),
    @("MinimumWorkReduction", $MinimumWorkReduction, 0.0, 1.0),
    @("MaximumMedianRatio", $MaximumMedianRatio, 0.0, 1.0)
)) {
    $name = [string]$entry[0]; $value = [double]$entry[1]; $minimum = [double]$entry[2]; $maximum = [double]$entry[3]
    if ([double]::IsNaN($value) -or [double]::IsInfinity($value) -or $value -lt $minimum -or $value -gt $maximum) {
        throw "$name is outside its finite accepted range"
    }
}
if (!$simplecExperiment -and $CandidatePressureRelTol -eq 0.0 -and $CandidateMomentumRelTol -eq 0.0) {
    throw "candidate p and U relTol must not both be zero in a TTA experiment"
}
function Test-TtaLinearRelTolActive([bool]$IsGamg, [double]$RelTol) {
    if ($IsGamg) { return $RelTol -gt 0.0 }
    return $RelTol -gt 1e-15
}
if ((Test-TtaLinearRelTolActive $false 1e-15) -or
    !(Test-TtaLinearRelTolActive $false 1.0000000000000003e-15) -or
    (Test-TtaLinearRelTolActive $true 0.0) -or
    !(Test-TtaLinearRelTolActive $true ([double]::Epsilon))) {
    throw "solver-aware relTol activation boundary self-check failed"
}

function Get-TtaRequiredJsonProperty($Object, [string]$Name, [string]$Path) {
    if ($null -eq $Object -or $Object -isnot [pscustomobject]) {
        throw "$Path must be a JSON object"
    }
    $property = $Object.PSObject.Properties[$Name]
    if ($null -eq $property -or $null -eq $property.Value) {
        throw "$Path.$Name must be present and non-null"
    }
    # Preserve a JSON array as one pipeline object until its type has been checked.
    return ,$property.Value
}

function Get-TtaRequiredJsonObject($Object, [string]$Name, [string]$Path) {
    $value = Get-TtaRequiredJsonProperty $Object $Name $Path
    if ($value -isnot [pscustomobject]) { throw "$Path.$Name must be an object" }
    return $value
}

function Get-TtaRequiredJsonArray($Object, [string]$Name, [string]$Path) {
    $value = Get-TtaRequiredJsonProperty $Object $Name $Path
    if ($value -isnot [System.Array]) { throw "$Path.$Name must be an array" }
    return $value
}

function Get-TtaRequiredJsonBoolean($Object, [string]$Name, [string]$Path) {
    $value = Get-TtaRequiredJsonProperty $Object $Name $Path
    if ($value -isnot [bool]) { throw "$Path.$Name must be a boolean" }
    return [bool]$value
}

function Get-TtaRequiredJsonString($Object, [string]$Name, [string]$Path) {
    $value = Get-TtaRequiredJsonProperty $Object $Name $Path
    if ($value -isnot [string] -or [string]::IsNullOrWhiteSpace($value)) {
        throw "$Path.$Name must be a non-empty string"
    }
    return [string]$value
}

function Test-TtaJsonIntegerType($Value) {
    return $Value -is [sbyte] -or $Value -is [byte] -or
        $Value -is [int16] -or $Value -is [uint16] -or
        $Value -is [int32] -or $Value -is [uint32] -or
        $Value -is [int64] -or $Value -is [uint64]
}

function Test-TtaJsonNumberType($Value) {
    return (Test-TtaJsonIntegerType $Value) -or $Value -is [single] -or
        $Value -is [double] -or $Value -is [decimal]
}

function Get-TtaRequiredJsonInteger($Object, [string]$Name, [string]$Path) {
    $value = Get-TtaRequiredJsonProperty $Object $Name $Path
    if (!(Test-TtaJsonIntegerType $value)) { throw "$Path.$Name must be an integer" }
    if ([decimal]$value -lt 0 -or [decimal]$value -gt [decimal][long]::MaxValue) {
        throw "$Path.$Name must be a non-negative signed-64-bit integer"
    }
    return [long]$value
}

function Get-TtaRequiredJsonNumber($Object, [string]$Name, [string]$Path) {
    $value = Get-TtaRequiredJsonProperty $Object $Name $Path
    if (!(Test-TtaJsonNumberType $value)) { throw "$Path.$Name must be a numeric scalar" }
    $number = [double]$value
    if ([double]::IsNaN($number) -or [double]::IsInfinity($number)) { throw "$Path.$Name must be finite" }
    return $number
}

function Assert-TtaExpectedJsonFailure([scriptblock]$Probe, [string]$Description) {
    $failed = $false
    try { & $Probe | Out-Null } catch { $failed = $true }
    if (!$failed) { throw "JSON fail-closed self-check accepted $Description" }
}

function ConvertTo-TtaFoamTokens([string]$Content, [string]$Description) {
    if ($null -eq $script:TtaFoamTokenRegex) {
        $pattern = '\G(?:(?<ws>\s+)|(?<line>//[^\r\n]*)|(?<block>/\*[\s\S]*?\*/)|(?<directive>\#(?:\\\r?\n|[^\r\n])*(?:\r?\n|\z))|(?<dq>\x22(?:\\[\s\S]|[^\x22\\])*\x22)|(?<sq>\x27(?:\\[\s\S]|[^\x27\\])*\x27)|(?<punct>[{};])|(?<word>(?:(?!//|/\*)[^\s{};\x22\x27#])+))'
        $script:TtaFoamTokenRegex = [regex]::new(
            $pattern,
            [System.Text.RegularExpressions.RegexOptions]::Compiled -bor [System.Text.RegularExpressions.RegexOptions]::CultureInvariant
        )
    }
    $tokens = New-Object System.Collections.Generic.List[object]
    $index = 0
    while ($index -lt $Content.Length) {
        $match = $script:TtaFoamTokenRegex.Match($Content, $index)
        if (!$match.Success -or $match.Index -ne $index -or $match.Length -lt 1) {
            if ($Content.Substring($index).StartsWith('/*', [System.StringComparison]::Ordinal)) {
                throw "$Description contains an unterminated block comment at offset $index"
            }
            if ($Content[$index] -eq '"' -or $Content[$index] -eq "'") {
                throw "$Description contains an unterminated quoted string at offset $index"
            }
            throw "$Description contains an unsupported token at offset $index"
        }
        if ($match.Groups["punct"].Success) {
            $tokens.Add([pscustomobject][ordered]@{ kind = "punct"; text = $match.Value; offset = $index }) | Out-Null
        } elseif ($match.Groups["word"].Success) {
            $tokens.Add([pscustomobject][ordered]@{ kind = "word"; text = $match.Value; offset = $index }) | Out-Null
        } elseif ($match.Groups["dq"].Success -or $match.Groups["sq"].Success) {
            $tokens.Add([pscustomobject][ordered]@{ kind = "string"; text = $match.Value; offset = $index }) | Out-Null
        } elseif ($match.Groups["directive"].Success) {
            $lineStart = 0
            if ($index -gt 0) {
                $lastLineFeed = $Content.LastIndexOf([char]"`n", $index - 1)
                $lastCarriageReturn = $Content.LastIndexOf([char]"`r", $index - 1)
                $lineStart = [Math]::Max($lastLineFeed, $lastCarriageReturn) + 1
            }
            $leading = $Content.Substring($lineStart, $index - $lineStart)
            if ($leading -notmatch '^[\t ]*$') {
                throw "$Description contains a directive marker outside logical-line start at offset $index"
            }
            $tokens.Add([pscustomobject][ordered]@{ kind = "directive"; text = $match.Value; offset = $index }) | Out-Null
        }
        # Whitespace and comments are token-free; active directives remain explicit fail-closed tokens.
        $index += $match.Length
    }
    return $tokens.ToArray()
}

function Get-TtaFoamBracePairs($Tokens, [string]$Description) {
    $stack = New-Object System.Collections.Generic.Stack[int]
    $pairs = @{}
    for ($index = 0; $index -lt $Tokens.Count; $index++) {
        if ($Tokens[$index].kind -cne "punct") { continue }
        if ($Tokens[$index].text -ceq '{') {
            $stack.Push($index)
        } elseif ($Tokens[$index].text -ceq '}') {
            if ($stack.Count -eq 0) { throw "$Description has an unmatched closing brace at offset $($Tokens[$index].offset)" }
            $opening = $stack.Pop()
            $pairs[$opening] = $index
        }
    }
    if ($stack.Count -ne 0) {
        $opening = $stack.Peek()
        throw "$Description has an unterminated block at offset $($Tokens[$opening].offset)"
    }
    return $pairs
}

function Get-TtaFoamDirectEntries($Tokens, [int]$StartIndex, [int]$EndIndex, $BracePairs, [string]$Description) {
    $entries = New-Object System.Collections.Generic.List[object]
    $index = $StartIndex
    while ($index -lt $EndIndex) {
        $key = $Tokens[$index]
        if ($key.kind -cne "word" -and $key.kind -cne "string") {
            throw "$Description expected an entry key at offset $($key.offset)"
        }
        $index++
        if ($index -ge $EndIndex) { throw "$Description.$($key.text) is missing a value or dictionary" }
        if ($Tokens[$index].kind -ceq "punct" -and $Tokens[$index].text -ceq '{') {
            if (!$BracePairs.ContainsKey($index)) { throw "$Description.$($key.text) has an unpaired opening brace" }
            $closing = [int]$BracePairs[$index]
            if ($closing -ge $EndIndex) { throw "$Description.$($key.text) escapes its parent dictionary" }
            $entries.Add([pscustomobject][ordered]@{
                name = [string]$key.text; keyKind = [string]$key.kind; entryKind = "dictionary"
                keyIndex = $index - 1; openingIndex = $index; closingIndex = $closing; valueTokens = @()
            }) | Out-Null
            $index = $closing + 1
            if ($index -lt $EndIndex -and $Tokens[$index].kind -ceq "punct" -and $Tokens[$index].text -ceq ';') { $index++ }
            continue
        }
        $valueStart = $index
        while ($index -lt $EndIndex -and !($Tokens[$index].kind -ceq "punct" -and $Tokens[$index].text -ceq ';')) {
            if ($Tokens[$index].kind -ceq "punct" -and ($Tokens[$index].text -ceq '{' -or $Tokens[$index].text -ceq '}')) {
                throw "$Description.$($key.text) contains an unexpected block at offset $($Tokens[$index].offset)"
            }
            $index++
        }
        if ($index -ge $EndIndex) { throw "$Description.$($key.text) is missing its terminating semicolon" }
        $valueTokens = if ($valueStart -lt $index) { @($Tokens[$valueStart..($index - 1)]) } else { @() }
        $entries.Add([pscustomobject][ordered]@{
            name = [string]$key.text; keyKind = [string]$key.kind; entryKind = "scalar"
            keyIndex = $valueStart - 1; openingIndex = -1; closingIndex = -1; valueTokens = $valueTokens
        }) | Out-Null
        $index++
    }
    return $entries.ToArray()
}

function Assert-TtaZeroProfileRelTolText([string]$Content, [string]$Description) {
    $tokens = @(ConvertTo-TtaFoamTokens $Content $Description)
    $activeDirectives = @($tokens | Where-Object { $_.kind -ceq "directive" })
    if ($activeDirectives.Count -ne 0) {
        throw "$Description contains an active OpenFOAM directive at offset $($activeDirectives[0].offset)"
    }
    $bracePairs = Get-TtaFoamBracePairs $tokens $Description
    $topLevel = @(Get-TtaFoamDirectEntries $tokens 0 $tokens.Count $bracePairs $Description)
    $solversEntries = @($topLevel | Where-Object { $_.keyKind -ceq "word" -and $_.name -ceq "solvers" })
    if ($solversEntries.Count -ne 1 -or $solversEntries[0].entryKind -cne "dictionary") {
        throw "$Description must contain exactly one active top-level ordinary solvers dictionary"
    }
    $solvers = $solversEntries[0]
    $solverChildren = @(Get-TtaFoamDirectEntries $tokens ($solvers.openingIndex + 1) $solvers.closingIndex $bracePairs "$Description.solvers")
    foreach ($solverName in @("p", "U")) {
        $solverEntries = @($solverChildren | Where-Object { $_.keyKind -ceq "word" -and $_.name -ceq $solverName })
        if ($solverEntries.Count -ne 1 -or $solverEntries[0].entryKind -cne "dictionary") {
            throw "$Description.solvers.$solverName must be exactly one direct ordinary dictionary child"
        }
        $solver = $solverEntries[0]
        $solverOptions = @(Get-TtaFoamDirectEntries $tokens ($solver.openingIndex + 1) $solver.closingIndex $bracePairs "$Description.solvers.$solverName")
        $relTolEntries = @($solverOptions | Where-Object { $_.keyKind -ceq "word" -and $_.name -ceq "relTol" })
        if ($relTolEntries.Count -ne 1 -or $relTolEntries[0].entryKind -cne "scalar") {
            throw "$Description.solvers.$solverName.relTol must be exactly one direct scalar entry"
        }
        $valueTokens = @($relTolEntries[0].valueTokens)
        if ($valueTokens.Count -ne 1 -or $valueTokens[0].kind -cne "word") {
            throw "$Description.solvers.$solverName.relTol must contain exactly one unquoted scalar value"
        }
        $value = [double]::Parse($valueTokens[0].text, [System.Globalization.CultureInfo]::InvariantCulture)
        if ([double]::IsNaN($value) -or [double]::IsInfinity($value) -or $value -ne 0.0) {
            throw "$Description.solvers.$solverName.relTol must be exactly zero"
        }
    }
}

function Assert-ZeroProfileRelTol([string]$Path) {
    if (!(Test-Path -LiteralPath $Path -PathType Leaf)) { throw "baseline convergence profile is missing: $Path" }
    Assert-TtaZeroProfileRelTolText (Get-Content -LiteralPath $Path -Raw) $Path
}

function Get-TtaSimpleConsistentToken([string]$Content, [string]$Description) {
    $tokens = @(ConvertTo-TtaFoamTokens $Content $Description)
    $activeDirectives = @($tokens | Where-Object { $_.kind -ceq "directive" })
    if ($activeDirectives.Count -ne 0) {
        throw "$Description contains an active OpenFOAM directive at offset $($activeDirectives[0].offset)"
    }
    $bracePairs = Get-TtaFoamBracePairs $tokens $Description
    $topLevel = @(Get-TtaFoamDirectEntries $tokens 0 $tokens.Count $bracePairs $Description)
    $simpleEntries = @($topLevel | Where-Object { $_.keyKind -ceq "word" -and $_.name -ceq "SIMPLE" })
    if ($simpleEntries.Count -ne 1 -or $simpleEntries[0].entryKind -cne "dictionary") {
        throw "$Description must contain exactly one active top-level ordinary SIMPLE dictionary"
    }
    $simple = $simpleEntries[0]
    $simpleOptions = @(Get-TtaFoamDirectEntries $tokens ($simple.openingIndex + 1) $simple.closingIndex $bracePairs "$Description.SIMPLE")
    $consistentEntries = @($simpleOptions | Where-Object { $_.keyKind -ceq "word" -and $_.name -ceq "consistent" })
    if ($consistentEntries.Count -ne 1 -or $consistentEntries[0].entryKind -cne "scalar") {
        throw "$Description.SIMPLE.consistent must be exactly one direct ordinary scalar entry"
    }
    $valueTokens = @($consistentEntries[0].valueTokens)
    if ($valueTokens.Count -ne 1 -or $valueTokens[0].kind -cne "word" -or
        ($valueTokens[0].text -cne "false" -and $valueTokens[0].text -cne "true")) {
        throw "$Description.SIMPLE.consistent must contain exactly one direct unquoted lowercase boolean token"
    }
    return $valueTokens[0]
}

function Invoke-TtaSimpleConsistentBytes([byte[]]$Bytes, [bool]$Expected, [bool]$PatchToTrue, [string]$Description) {
    if ($null -eq $Bytes -or $Bytes.Count -eq 0) { throw "$Description is empty" }
    $hasBom = $Bytes.Count -ge 3 -and $Bytes[0] -eq 0xef -and $Bytes[1] -eq 0xbb -and $Bytes[2] -eq 0xbf
    $payloadOffset = if ($hasBom) { 3 } else { 0 }
    $utf8 = [System.Text.UTF8Encoding]::new($false, $true)
    try {
        $text = $utf8.GetString($Bytes, $payloadOffset, $Bytes.Count - $payloadOffset)
    } catch {
        throw "$Description is not strict UTF-8: $($_.Exception.Message)"
    }
    $token = Get-TtaSimpleConsistentToken $text $Description
    $expectedText = if ($Expected) { "true" } else { "false" }
    if ($token.text -cne $expectedText) { throw "$Description.SIMPLE.consistent differs from expected '$expectedText'" }
    $byteStart = $payloadOffset + $utf8.GetByteCount($text.Substring(0, [int]$token.offset))
    $tokenByteCount = $utf8.GetByteCount([string]$token.text)
    $byteEnd = $byteStart + $tokenByteCount
    if (!$PatchToTrue) {
        return [pscustomobject][ordered]@{ bytes = [byte[]]$Bytes; tokenStart = $byteStart; tokenEnd = $byteEnd }
    }
    if ($Expected) { throw "$Description cannot patch an already-true SIMPLE.consistent token" }
    [byte[]]$replacement = [System.Text.Encoding]::ASCII.GetBytes("true")
    [byte[]]$patched = New-Object byte[] ($Bytes.Count - $tokenByteCount + $replacement.Count)
    [System.Buffer]::BlockCopy($Bytes, 0, $patched, 0, $byteStart)
    [System.Buffer]::BlockCopy($replacement, 0, $patched, $byteStart, $replacement.Count)
    [System.Buffer]::BlockCopy($Bytes, $byteEnd, $patched, $byteStart + $replacement.Count, $Bytes.Count - $byteEnd)
    [void](Invoke-TtaSimpleConsistentBytes $patched $true $false "$Description patched")
    return [pscustomobject][ordered]@{ bytes = $patched; tokenStart = $byteStart; tokenEnd = $byteStart + $replacement.Count }
}

function Assert-TtaExpectedSimpleConsistentFailure([scriptblock]$Probe, [string]$Description) {
    $failed = $false
    try { & $Probe | Out-Null } catch { $failed = $true }
    if (!$failed) { throw "SIMPLE.consistent fail-closed self-check accepted $Description" }
}

function Invoke-TtaSimpleConsistentSelfTest {
    $utf8 = [System.Text.UTF8Encoding]::new($false)
    $sample = @'
// SIMPLE { consistent true; }
"SIMPLE" { consistent true; }
wrapper { SIMPLE { consistent true; } }
SIMPLE
{
    nNonOrthogonalCorrectors 0;
    note "consistent true; }";
    consistent false;
}
'@
    [byte[]]$before = $utf8.GetBytes($sample)
    $transformed = Invoke-TtaSimpleConsistentBytes $before $false $true "host SIMPLE.consistent positive self-test"
    [byte[]]$after = $transformed.bytes
    [byte[]]$expected = $utf8.GetBytes($sample.Replace("    consistent false;", "    consistent true;"))
    if ([Convert]::ToBase64String($after) -cne [Convert]::ToBase64String($expected)) {
        throw "host SIMPLE.consistent self-test changed bytes outside the direct false token"
    }
    [void](Invoke-TtaSimpleConsistentBytes $after $true $false "host SIMPLE.consistent true verification self-test")
    $malformed = [ordered]@{
        "missing SIMPLE" = "// SIMPLE { consistent false; }"
        "quoted SIMPLE" = '"SIMPLE" { consistent false; }'
        "nested-only SIMPLE" = "wrapper { SIMPLE { consistent false; } }"
        "duplicate SIMPLE" = "SIMPLE { consistent false; } SIMPLE { consistent false; }"
        "missing consistent" = "SIMPLE { nNonOrthogonalCorrectors 0; }"
        "duplicate consistent" = "SIMPLE { consistent false; consistent false; }"
        "nested-only consistent" = "SIMPLE { controls { consistent false; } }"
        "quoted consistent key" = 'SIMPLE { "consistent" false; }'
        "quoted boolean value" = 'SIMPLE { consistent "false"; }'
        "multitoken boolean" = "SIMPLE { consistent false true; }"
        "dictionary boolean" = "SIMPLE { consistent { value false; } }"
        "noncanonical boolean" = "SIMPLE { consistent False; }"
        "active directive" = "#include `"other`"`nSIMPLE { consistent false; }"
        "unterminated block" = "SIMPLE { consistent false; "
        "unterminated comment" = "SIMPLE { consistent false; } /*"
        "unterminated string" = 'SIMPLE { note "bad; consistent false; }'
    }
    foreach ($entry in $malformed.GetEnumerator()) {
        [byte[]]$raw = $utf8.GetBytes([string]$entry.Value)
        Assert-TtaExpectedSimpleConsistentFailure { Invoke-TtaSimpleConsistentBytes $raw $false $false "host malformed self-test" } $entry.Key
    }
    Assert-TtaExpectedSimpleConsistentFailure { Invoke-TtaSimpleConsistentBytes $after $false $true "host already-true patch self-test" } "an already-true candidate source"
}

Invoke-TtaSimpleConsistentSelfTest

function Assert-TtaPositivePressureSolveCount([long]$Count, [string]$Path) {
    if ($Count -lt 1) { throw "$Path.pressureLinearSolves must be at least one" }
}

function Get-TtaExpectedValidatedReportRelativePath([string]$Case, [string]$Kind, [long]$Ordinal, [string]$Ref) {
    if ([string]::IsNullOrWhiteSpace($Case) -or $Case -notmatch '^[A-Za-z0-9._-]+$' -or $Case -eq '.' -or $Case -eq '..') {
        throw "validated report proof case name is unsafe"
    }
    if ($Ref -cne "baseline" -and $Ref -cne "candidate") { throw "validated report proof ref is invalid" }
    if ($Kind -ceq "oracle") {
        if ($Ordinal -ne 0) { throw "validated oracle report proof ordinal must be zero" }
        $runIdentity = "oracle-$Ref"
    } elseif ($Kind -ceq "warmup" -or $Kind -ceq "measured") {
        if ($Ordinal -lt 1) { throw "validated timed report proof ordinal must be positive" }
        $runIdentity = "$Kind-$Ordinal-$Ref"
    } else {
        throw "validated report proof kind is invalid"
    }
    return "raw/$Case/$runIdentity/solve-report.json"
}

function Assert-TtaExactJsonProperties($Object, [string[]]$Expected, [string]$Path) {
    if ($null -eq $Object -or $Object -isnot [pscustomobject]) { throw "$Path must be a JSON object" }
    $actual = [string[]]@($Object.PSObject.Properties.Name | Sort-Object)
    if (@(Compare-Object ([string[]]@($Expected | Sort-Object)) $actual -CaseSensitive).Count -ne 0) {
        throw "$Path property set differs"
    }
}

function Get-TtaOracleFieldShape($FieldValues, [string]$Path) {
    if ($null -eq $FieldValues -or $FieldValues -isnot [pscustomobject]) { throw "$Path must be an object" }
    $cellCount = Get-TtaRequiredJsonInteger $FieldValues "cellCount" $Path
    if ($cellCount -lt 1) { throw "$Path.cellCount must be positive" }
    $uField = Get-TtaRequiredJsonObject $FieldValues "U" $Path
    $pField = Get-TtaRequiredJsonObject $FieldValues "p" $Path
    $uValues = @(Get-TtaRequiredJsonArray $uField "values" "$Path.U")
    $pValues = @(Get-TtaRequiredJsonArray $pField "values" "$Path.p")
    if ($uValues.Count -ne (3 * $cellCount) -or $pValues.Count -ne $cellCount) {
        throw "$Path field lengths differ from cellCount"
    }
    return [pscustomobject][ordered]@{ cellCount = $cellCount; uCount = $uValues.Count; pCount = $pValues.Count }
}

function Assert-TtaMatchingOracleFieldShapes($BaselineFieldValues, $CandidateFieldValues, [string]$Path) {
    $baselineShape = Get-TtaOracleFieldShape $BaselineFieldValues "$Path.baseline"
    $candidateShape = Get-TtaOracleFieldShape $CandidateFieldValues "$Path.candidate"
    if ($baselineShape.cellCount -ne $candidateShape.cellCount -or
        $baselineShape.uCount -ne $candidateShape.uCount -or
        $baselineShape.pCount -ne $candidateShape.pCount) {
        throw "$Path baseline/candidate oracle field shapes differ"
    }
}

$jsonSelfTest = [pscustomobject][ordered]@{
    number = 1.25
    integer = 2
    boolean = $true
    string = "x"
    array = @(1, 2)
    object = [pscustomobject]@{ x = 1 }
    nullValue = $null
    numericString = "1"
    boolAsNumber = $false
    floatAsInteger = 1.0
    infinity = [double]::PositiveInfinity
}
if ((Get-TtaRequiredJsonNumber $jsonSelfTest "number" "$.selfTest") -ne 1.25 -or
    (Get-TtaRequiredJsonInteger $jsonSelfTest "integer" "$.selfTest") -ne 2 -or
    !(Get-TtaRequiredJsonBoolean $jsonSelfTest "boolean" "$.selfTest") -or
    (Get-TtaRequiredJsonString $jsonSelfTest "string" "$.selfTest") -cne "x" -or
    @(Get-TtaRequiredJsonArray $jsonSelfTest "array" "$.selfTest").Count -ne 2 -or
    $null -eq (Get-TtaRequiredJsonObject $jsonSelfTest "object" "$.selfTest")) {
    throw "JSON strict-type positive self-check failed"
}
$jsonArrayShapeSelfTest = '{"empty":[],"single":[{"id":1}],"multiple":[{"id":1},{"id":2}],"object":{"id":1}}' | ConvertFrom-Json
$emptyJsonArray = @(Get-TtaRequiredJsonArray $jsonArrayShapeSelfTest "empty" "$.arrayShape")
$singleJsonArray = @(Get-TtaRequiredJsonArray $jsonArrayShapeSelfTest "single" "$.arrayShape")
$multipleJsonArray = @(Get-TtaRequiredJsonArray $jsonArrayShapeSelfTest "multiple" "$.arrayShape")
if ($emptyJsonArray.Count -ne 0 -or $singleJsonArray.Count -ne 1 -or $multipleJsonArray.Count -ne 2 -or
    $singleJsonArray[0] -isnot [pscustomobject] -or $multipleJsonArray[1] -isnot [pscustomobject]) {
    throw "JSON array-shape preservation self-check failed"
}
Assert-TtaExpectedJsonFailure { Get-TtaRequiredJsonArray $jsonArrayShapeSelfTest "object" "$.arrayShape" } "a parsed scalar object array"
Assert-TtaExpectedJsonFailure { Get-TtaRequiredJsonNumber $jsonSelfTest "missing" "$.selfTest" } "a missing number"
Assert-TtaExpectedJsonFailure { Get-TtaRequiredJsonNumber $jsonSelfTest "nullValue" "$.selfTest" } "a null number"
Assert-TtaExpectedJsonFailure { Get-TtaRequiredJsonNumber $jsonSelfTest "numericString" "$.selfTest" } "a numeric string"
Assert-TtaExpectedJsonFailure { Get-TtaRequiredJsonNumber $jsonSelfTest "boolAsNumber" "$.selfTest" } "a boolean number"
Assert-TtaExpectedJsonFailure { Get-TtaRequiredJsonNumber $jsonSelfTest "infinity" "$.selfTest" } "an infinite number"
Assert-TtaExpectedJsonFailure { Get-TtaRequiredJsonInteger $jsonSelfTest "floatAsInteger" "$.selfTest" } "a floating counter"
Assert-TtaExpectedJsonFailure { Get-TtaRequiredJsonInteger $jsonSelfTest "boolAsNumber" "$.selfTest" } "a boolean integer"
Assert-TtaExpectedJsonFailure { Get-TtaRequiredJsonBoolean $jsonSelfTest "integer" "$.selfTest" } "an integer boolean"
Assert-TtaExpectedJsonFailure { Get-TtaRequiredJsonArray $jsonSelfTest "object" "$.selfTest" } "an object array"
$validOracleShape = [pscustomobject][ordered]@{
    cellCount = 2
    U = [pscustomobject]@{ values = @(0.0, 0.0, 0.0, 1.0, 1.0, 1.0) }
    p = [pscustomobject]@{ values = @(0.0, 1.0) }
}
$boolCellOracleShape = [pscustomobject][ordered]@{
    cellCount = $true
    U = [pscustomobject]@{ values = @(0.0, 0.0, 0.0) }
    p = [pscustomobject]@{ values = @(0.0) }
}
$mismatchedOracleShape = [pscustomobject][ordered]@{
    cellCount = 2
    U = [pscustomobject]@{ values = @(0.0, 0.0, 0.0) }
    p = [pscustomobject]@{ values = @(0.0, 1.0) }
}
Assert-TtaExpectedJsonFailure { Get-TtaOracleFieldShape $boolCellOracleShape "$.selfTest.oracle" } "a boolean cell count"
Assert-TtaExpectedJsonFailure { Assert-TtaPositivePressureSolveCount 0 "$.selfTest.history[0]" } "zero pressure solves"
Assert-TtaExpectedJsonFailure { Assert-TtaMatchingOracleFieldShapes $validOracleShape $mismatchedOracleShape "$.selfTest.oracle" } "mismatched field lengths"
if ((Get-TtaExpectedValidatedReportRelativePath "caseA" "measured" 2 "candidate") -cne "raw/caseA/measured-2-candidate/solve-report.json" -or
    (Get-TtaExpectedValidatedReportRelativePath "caseA" "oracle" 0 "baseline") -cne "raw/caseA/oracle-baseline/solve-report.json") {
    throw "validated report proof identity positive self-check failed"
}
foreach ($probe in @(
    { Get-TtaExpectedValidatedReportRelativePath "../case" "measured" 1 "candidate" },
    { Get-TtaExpectedValidatedReportRelativePath "caseA" "measured" 0 "candidate" },
    { Get-TtaExpectedValidatedReportRelativePath "caseA" "oracle" 1 "candidate" },
    { Get-TtaExpectedValidatedReportRelativePath "caseA" "other" 1 "candidate" },
    { Get-TtaExpectedValidatedReportRelativePath "caseA" "measured" 1 "Candidate" }
)) {
    Assert-TtaExpectedJsonFailure $probe "an invalid validated-report proof identity"
}
$validProfileWithTemperature = @'
// solvers { p { relTol 9; } U { relTol 9; } } } {
// #include "commented-control-dictionary"
/* solvers { p { relTol 8; } U { relTol 8; } unmatched: }}}
   #includeIfPresent "commented-too" */
"solvers"
{
    p { relTol 7; }
    U { relTol 7; }
}
solvers
{
    p
    {
        solver GAMG;
        note "quoted } { relTol 6; #include \"quoted\" // text";
        // relTol 5; }
        /* nested-looking text { relTol 4; } */
        relTol 0;
    }
    U { solver smoothSolver; note "} p { relTol 3; }"; relTol 0; }
    T { solver smoothSolver; relTol 0.25; }
}
'@
Assert-TtaZeroProfileRelTolText $validProfileWithTemperature "$.selfTest.fvSolution"
$missingUProfile = @'
solvers
{
    p { relTol 0; }
    T { relTol 0.25; }
}
'@
$duplicatePProfile = @'
solvers
{
    p { relTol 0; }
    p { relTol 0; }
    U { relTol 0; }
}
'@
$duplicateURelTolProfile = @'
solvers
{
    p { relTol 0; }
    U
    {
        relTol 0;
        relTol 0;
    }
}
'@
$nestedPUProfile = @'
solvers
{
    group
    {
        p { relTol 0; }
        U { relTol 0; }
    }
}
'@
$nestedRelTolProfile = @'
solvers
{
    p
    {
        controls { relTol 0; }
    }
    U { relTol 0; }
}
'@
$duplicateSolversProfile = @'
solvers { p { relTol 0; } U { relTol 0; } }
solvers { p { relTol 0; } U { relTol 0; } }
'@
$dictionaryRelTolProfile = @'
solvers
{
    p { relTol { value 0; } }
    U { relTol 0; }
}
'@
$unterminatedCommentProfile = @'
solvers { p { relTol 0; } U { relTol 0; } }
/* solvers { p { relTol 9; }
'@
$unterminatedStringProfile = @'
solvers
{
    p { note "unterminated } relTol 0; }
    U { relTol 0; }
}
'@
$unterminatedBlockProfile = @'
solvers
{
    p { relTol 0; }
    U { relTol 0; }
'@
$activeIncludeProfile = @'
#include "external-solvers"
solvers { p { relTol 0; } U { relTol 0; } }
'@
$leadingWhitespaceDirectiveProfile = @'
    #includeIfPresent "optional-solvers"
solvers { p { relTol 0; } U { relTol 0; } }
'@
$codeStreamDirectiveProfile = @'
#codeStream
solvers { p { relTol 0; } U { relTol 0; } }
'@
$genericDirectiveProfile = @'
#unknownDirective value
solvers { p { relTol 0; } U { relTol 0; } }
'@
Assert-TtaExpectedJsonFailure { Assert-TtaZeroProfileRelTolText $missingUProfile "$.selfTest.missingU" } "a missing U solver section"
Assert-TtaExpectedJsonFailure { Assert-TtaZeroProfileRelTolText $duplicatePProfile "$.selfTest.duplicateP" } "duplicate p solver sections"
Assert-TtaExpectedJsonFailure { Assert-TtaZeroProfileRelTolText $duplicateURelTolProfile "$.selfTest.duplicateURelTol" } "duplicate U relTol entries"
Assert-TtaExpectedJsonFailure { Assert-TtaZeroProfileRelTolText $nestedPUProfile "$.selfTest.nestedPU" } "nested non-direct p and U sections"
Assert-TtaExpectedJsonFailure { Assert-TtaZeroProfileRelTolText $nestedRelTolProfile "$.selfTest.nestedRelTol" } "a nested non-direct relTol"
Assert-TtaExpectedJsonFailure { Assert-TtaZeroProfileRelTolText $duplicateSolversProfile "$.selfTest.duplicateSolvers" } "duplicate active top-level solvers dictionaries"
Assert-TtaExpectedJsonFailure { Assert-TtaZeroProfileRelTolText $dictionaryRelTolProfile "$.selfTest.dictionaryRelTol" } "a non-scalar relTol"
Assert-TtaExpectedJsonFailure { Assert-TtaZeroProfileRelTolText $unterminatedCommentProfile "$.selfTest.unterminatedComment" } "an unterminated block comment"
Assert-TtaExpectedJsonFailure { Assert-TtaZeroProfileRelTolText $unterminatedStringProfile "$.selfTest.unterminatedString" } "an unterminated quoted string"
Assert-TtaExpectedJsonFailure { Assert-TtaZeroProfileRelTolText $unterminatedBlockProfile "$.selfTest.unterminatedBlock" } "an unterminated dictionary block"
Assert-TtaExpectedJsonFailure { Assert-TtaZeroProfileRelTolText $activeIncludeProfile "$.selfTest.activeInclude" } "an active include directive"
Assert-TtaExpectedJsonFailure { Assert-TtaZeroProfileRelTolText $leadingWhitespaceDirectiveProfile "$.selfTest.leadingDirective" } "a leading-whitespace includeIfPresent directive"
Assert-TtaExpectedJsonFailure { Assert-TtaZeroProfileRelTolText $codeStreamDirectiveProfile "$.selfTest.codeStream" } "an active codeStream directive"
Assert-TtaExpectedJsonFailure { Assert-TtaZeroProfileRelTolText $genericDirectiveProfile "$.selfTest.genericDirective" } "an active generic directive"
if ($null -eq (Get-Command wsl -ErrorAction SilentlyContinue)) { throw "wsl.exe was not found" }

function Format-Invariant([double]$Value) {
    return $Value.ToString("G17", [System.Globalization.CultureInfo]::InvariantCulture)
}

function Get-TtaSha256Bytes([byte[]]$Bytes) {
    $sha256 = [System.Security.Cryptography.SHA256]::Create()
    try {
        return -join @($sha256.ComputeHash($Bytes) | ForEach-Object { $_.ToString("x2") })
    } finally {
        $sha256.Dispose()
    }
}

function Assert-TtaStrictJsonFile([string]$Path, [string]$Description) {
    if (!(Test-Path -LiteralPath $Path -PathType Leaf)) { throw "$Description JSON file is missing: $Path" }
    $strictJsonPython = @'
import json, math, pathlib, sys
def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result: raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result
def reject_constant(value):
    raise ValueError(f"non-finite JSON constant: {value}")
def finite(value):
    if isinstance(value, float) and not math.isfinite(value): raise ValueError("non-finite JSON number")
    if isinstance(value, dict):
        for child in value.values(): finite(child)
    elif isinstance(value, list):
        for child in value: finite(child)
path = pathlib.Path(sys.argv[1])
value = json.loads(path.read_text(encoding="utf-8-sig"), object_pairs_hook=unique_object, parse_constant=reject_constant)
finite(value)
'@
    $strictJsonBase64 = [System.Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes($strictJsonPython))
    $strictJsonBootstrap = 'import base64,sys;exec(base64.b64decode(sys.argv.pop(1)))'
    $wslPath = ConvertTo-MatchedWslPath $Path $Distro
    $previousErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        $strictOutput = @(& wsl.exe -d $Distro -- python3 -c $strictJsonBootstrap $strictJsonBase64 $wslPath 2>&1)
        $strictExitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }
    if ($strictExitCode -ne 0) { throw "$Description is not strict JSON:`n$($strictOutput -join "`n")" }
}

$strictJsonSelfTestPath = [System.IO.Path]::GetTempFileName()
try {
    [System.IO.File]::WriteAllText($strictJsonSelfTestPath, '{"x":[1,2.5]}', [System.Text.UTF8Encoding]::new($false))
    Assert-TtaStrictJsonFile $strictJsonSelfTestPath "strict JSON positive self-test"
    foreach ($invalidJson in @('{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}', '{"x":1e999}')) {
        [System.IO.File]::WriteAllText($strictJsonSelfTestPath, $invalidJson, [System.Text.UTF8Encoding]::new($false))
        Assert-TtaExpectedJsonFailure { Assert-TtaStrictJsonFile $strictJsonSelfTestPath "strict JSON negative self-test" } "invalid strict JSON"
    }
} finally {
    Remove-Item -LiteralPath $strictJsonSelfTestPath -Force -ErrorAction SilentlyContinue
}

$pressureRelTolText = Format-Invariant $CandidatePressureRelTol
$momentumRelTolText = Format-Invariant $CandidateMomentumRelTol
$workerWslPath = ConvertTo-MatchedWslPath $WorkerPath $Distro
$workerBootstrap = 'set -o pipefail; tr -d ''\r'' < "\$1" | bash -s -- "\${@:2}"'
$preflightArguments = @(
    "-d", $Distro, "--", "bash", "-c", $workerBootstrap, "ferrum-linux-tta-ab-worker", $workerWslPath,
    "--preflight-only", "--rust-toolchain", $RustToolchain, "--cpu-set", $CpuSet,
    "--experiment", $Experiment,
    "--build-variant", $BuildVariant,
    "--warmup-runs", $WarmupRuns.ToString([System.Globalization.CultureInfo]::InvariantCulture),
    "--measured-runs", $MeasuredRuns.ToString([System.Globalization.CultureInfo]::InvariantCulture),
    "--pressure-solver", $PressureSolver,
    "--max-simple-iterations", $MaxSimpleIterations.ToString([System.Globalization.CultureInfo]::InvariantCulture),
    "--candidate-pressure-reltol", $pressureRelTolText,
    "--candidate-momentum-reltol", $momentumRelTolText
)
$previousErrorActionPreference = $ErrorActionPreference
try {
    $ErrorActionPreference = "Continue"
    $preflightOutput = & wsl @preflightArguments 2>&1
    $preflightExitCode = $LASTEXITCODE
} finally {
    $ErrorActionPreference = $previousErrorActionPreference
}
if ($preflightExitCode -ne 0) { throw "Ferrum Linux TTA A/B preflight failed for '$Distro':`n$($preflightOutput -join "`n")" }
Assert-ControlSourcesUnchanged "during WSL preflight"
if ($PreflightOnly) {
    Write-Output "host_json_strict_type_self_test=pass"
    Write-Output "host_strict_json_parser_self_test=pass"
    Write-Output "host_exact_report_proof_identity_self_test=pass"
    Write-Output "host_contract_negative_self_test=pass"
    Write-Output "host_profile_block_self_test=pass"
    Write-Output "host_simple_consistent_token_mutation_self_test=pass"
    Write-Output "host_reltol_boundary_self_test=pass"
    Write-Output "host_gnu_time_rss_parser_self_test=pass"
    $preflightOutput | Write-Output
    return
}

function Resolve-ExactCommit([string]$Ref, [string]$Label) {
    $commit = (& git -C $RepoRoot rev-parse "$Ref`^{commit}" 2>$null).Trim()
    if ($LASTEXITCODE -ne 0 -or $commit -notmatch "^[0-9a-f]{40}$") { throw "could not resolve $Label ref '$Ref' to an exact commit" }
    return $commit
}

$baselineCommit = Resolve-ExactCommit $BaselineRef "baseline"
$candidateCommit = if ($simplecExperiment) { $baselineCommit } else { Resolve-ExactCommit $CandidateRef "candidate" }
if (!$simplecExperiment -and $baselineCommit -eq $candidateCommit) { throw "baseline and candidate commits must differ" }
$baselineTree = (& git -C $RepoRoot rev-parse "$baselineCommit`^{tree}").Trim()
$candidateTree = if ($simplecExperiment) { $baselineTree } else { (& git -C $RepoRoot rev-parse "$candidateCommit`^{tree}").Trim() }
if ($baselineTree -notmatch "^[0-9a-f]{40}$" -or $candidateTree -notmatch "^[0-9a-f]{40}$") { throw "could not resolve exact baseline/candidate trees" }

if ($simplecExperiment) {
    [string[]]$changedPaths = @()
} else {
    $candidateLine = ((& git -C $RepoRoot rev-list --parents -n 1 $candidateCommit) -join " ").Trim()
    if ($LASTEXITCODE -ne 0) { throw "could not inspect candidate parents" }
    $candidateParts = @($candidateLine -split "\s+" | Where-Object { $_ })
    if ($candidateParts.Count -ne 2 -or $candidateParts[0] -ne $candidateCommit -or $candidateParts[1] -ne $baselineCommit) {
        throw "candidate must be a single-parent direct child of the exact baseline"
    }
    [string[]]$changedPaths = @(& git -C $RepoRoot diff --name-only --no-renames $baselineCommit $candidateCommit --)
    if ($LASTEXITCODE -ne 0 -or @(Compare-Object ([string[]]@($effectiveChangedPaths | Sort-Object)) ([string[]]@($changedPaths | Sort-Object)) -CaseSensitive).Count -ne 0) {
        throw "candidate changed-path set differs; expected '$($effectiveChangedPaths -join ', ')', found '$($changedPaths -join ', ')'"
    }
}

$baselineCargoLockBlob = (& git -C $RepoRoot rev-parse "$baselineCommit`:Cargo.lock" 2>$null).Trim()
$candidateCargoLockBlob = if ($simplecExperiment) { $baselineCargoLockBlob } else { (& git -C $RepoRoot rev-parse "$candidateCommit`:Cargo.lock" 2>$null).Trim() }
if ($LASTEXITCODE -ne 0 -or $baselineCargoLockBlob -notmatch "^[0-9a-f]{40}$" -or $baselineCargoLockBlob -ne $candidateCargoLockBlob) {
    throw "baseline and candidate must reference the identical Cargo.lock blob"
}

$launchStatus = [string[]]@(& git -C $RepoRoot status --porcelain=v1)
$sourceWorktreeCleanAtLaunch = $launchStatus.Count -eq 0
$caseSelector = switch ($CaseName) {
    "all" { "all" }
    "pipe" { "laminarPipe" }
    "channel" { "planeChannel" }
}
if ([string]::IsNullOrWhiteSpace($OutRoot)) {
    $defaultLeaf = if ($simplecExperiment) { "$PressureSolver-$BuildVariant-simplec" } else { "$PressureSolver-$BuildVariant" }
    $OutRoot = Join-Path $TargetRoot "benchmarks\ferrum_linux_tta_ab\$defaultLeaf"
}
$OutRoot = [System.IO.Path]::GetFullPath($OutRoot)
$benchmarkOutputRoot = [System.IO.Path]::GetFullPath((Join-Path $TargetRoot "benchmarks\ferrum_linux_tta_ab")).TrimEnd("\", "/")
if (!(Test-MatchedPathUnder $OutRoot $benchmarkOutputRoot) -or $OutRoot.Equals($benchmarkOutputRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "OutRoot must be a strict child of the dedicated Ferrum Linux TTA A/B root: $benchmarkOutputRoot"
}

$stageRoot = Join-Path $TargetRoot "benchmarks\.linux-tta-ab-stage-$PID"
Reset-MatchedTargetDirectory $stageRoot $TargetRoot
$completed = $false
try {
    Assert-ControlSourcesUnchanged "before control snapshot"
    $controlsRoot = Join-Path $stageRoot "controls"
    New-Item -ItemType Directory -Force -Path $controlsRoot | Out-Null
    $manifestControls = @()
    foreach ($binding in $ControlBindings) {
        $destination = Join-Path $controlsRoot $binding.name
        Copy-Item -LiteralPath $binding.sourcePath -Destination $destination -Force
        $snapshotSha256 = (Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($snapshotSha256 -ne $binding.sha256) { throw "TTA control snapshot differs: $($binding.name)" }
        $manifestControls += [pscustomobject][ordered]@{ name = $binding.name; sha256 = $binding.sha256 }
    }
    $controlsArchive = Join-Path $stageRoot "controls.tar"
    & tar -cf $controlsArchive -C $controlsRoot .
    if ($LASTEXITCODE -ne 0) { throw "could not create TTA control archive" }
    Assert-MatchedSafeTarArchive $controlsArchive "TTA controls"
    $controlsArchiveSha256 = (Get-FileHash -LiteralPath $controlsArchive -Algorithm SHA256).Hash.ToLowerInvariant()

    $baselineArchive = Join-Path $stageRoot "baseline.tar"
    & git -C $RepoRoot archive --format=tar --output=$baselineArchive $baselineCommit
    if ($LASTEXITCODE -ne 0) { throw "could not archive exact baseline commit" }
    if ($simplecExperiment) {
        $candidateArchive = $baselineArchive
    } else {
        $candidateArchive = Join-Path $stageRoot "candidate.tar"
        & git -C $RepoRoot archive --format=tar --output=$candidateArchive $candidateCommit
        if ($LASTEXITCODE -ne 0) { throw "could not archive exact candidate commit" }
    }
    Assert-MatchedSafeTarArchive $baselineArchive "exact baseline source"
    Assert-MatchedSafeTarArchive $candidateArchive "exact candidate source"
    $baselineArchiveSha256 = (Get-FileHash -LiteralPath $baselineArchive -Algorithm SHA256).Hash.ToLowerInvariant()
    $candidateArchiveSha256 = (Get-FileHash -LiteralPath $candidateArchive -Algorithm SHA256).Hash.ToLowerInvariant()

    $baselineSourceRoot = Join-Path $stageRoot "baseline-source"
    $candidateSourceRoot = Join-Path $stageRoot "candidate-source"
    foreach ($root in @($baselineSourceRoot, $candidateSourceRoot)) {
        New-Item -ItemType Directory -Force -Path $root | Out-Null
        Assert-MatchedNoReparsePath $root $stageRoot
    }
    & tar -xf $baselineArchive -C $baselineSourceRoot
    if ($LASTEXITCODE -ne 0) { throw "could not extract baseline source archive" }
    & tar -xf $candidateArchive -C $candidateSourceRoot
    if ($LASTEXITCODE -ne 0) { throw "could not extract candidate source archive" }
    $baselineCargoLockSha256 = (Get-FileHash -LiteralPath (Join-Path $baselineSourceRoot "Cargo.lock") -Algorithm SHA256).Hash.ToLowerInvariant()
    $candidateCargoLockSha256 = (Get-FileHash -LiteralPath (Join-Path $candidateSourceRoot "Cargo.lock") -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($baselineCargoLockSha256 -ne $candidateCargoLockSha256) { throw "extracted Cargo.lock bytes differ between refs" }

    $convergenceProfileName = if ($PressureSolver -eq "gamg") { "gamg-converged" } else { "converged" }
    $caseDefinitions = @(
        [pscustomobject][ordered]@{
            name = "laminarPipe"
            ferrumCase = Join-Path $baselineSourceRoot "tutorials\incompressibleFluid\laminarPipe\ferrum\case"
            profile = Join-Path $baselineSourceRoot "validation\profiles\incompressibleFluid\laminarPipe\$convergenceProfileName\system\fvSolution"
        },
        [pscustomobject][ordered]@{
            name = "planeChannel"
            ferrumCase = Join-Path $baselineSourceRoot "tutorials\incompressibleFluid\planeChannel\ferrum\case"
            profile = Join-Path $baselineSourceRoot "validation\profiles\incompressibleFluid\planeChannel\$convergenceProfileName\system\fvSolution"
        }
    )
    if ($caseSelector -ne "all") { $caseDefinitions = @($caseDefinitions | Where-Object { $_.name -eq $caseSelector }) }
    foreach ($case in $caseDefinitions) {
        if (!(Test-Path -LiteralPath $case.ferrumCase -PathType Container) -or !(Test-Path -LiteralPath $case.profile -PathType Leaf)) {
            throw "TTA case/profile is missing for $($case.name)"
        }
    }

    $templatesRoot = Join-Path $stageRoot "templates"
    New-Item -ItemType Directory -Force -Path $templatesRoot | Out-Null
    $manifestCases = @()
    foreach ($case in $caseDefinitions) {
        Assert-ZeroProfileRelTol $case.profile
        $destination = Join-Path $templatesRoot $case.name
        New-MatchedFerrumWorkingCase $case $destination $case.profile $templatesRoot | Out-Null
        $canonicalHashes = Get-MatchedPolyMeshHashes $case.ferrumCase
        Assert-MatchedHashesEqual $canonicalHashes (Get-MatchedPolyMeshHashes $destination) "$($case.name) TTA template"
        $baselineSolutionPath = Join-Path $destination "system\fvSolution"
        [byte[]]$baselineSolutionBytes = [System.IO.File]::ReadAllBytes($baselineSolutionPath)
        [void](Invoke-TtaSimpleConsistentBytes $baselineSolutionBytes $false $false "$($case.name) baseline template")
        $manifestCase = [ordered]@{
            name = $case.name
            canonicalPolyMeshSha256 = $canonicalHashes
            baselineFvSolutionSha256 = (Get-FileHash -LiteralPath $baselineSolutionPath -Algorithm SHA256).Hash.ToLowerInvariant()
        }
        if ($simplecExperiment) {
            $candidateTransform = Invoke-TtaSimpleConsistentBytes $baselineSolutionBytes $false $true "$($case.name) candidate template"
            $manifestCase.candidateFvSolutionSha256 = Get-TtaSha256Bytes ([byte[]]$candidateTransform.bytes)
        }
        $manifestCases += [pscustomobject]$manifestCase
    }
    $templatesArchive = Join-Path $stageRoot "templates.tar"
    & tar -cf $templatesArchive -C $templatesRoot .
    if ($LASTEXITCODE -ne 0) { throw "could not create TTA template archive" }
    Assert-MatchedSafeTarArchive $templatesArchive "TTA template"
    $templatesArchiveSha256 = (Get-FileHash -LiteralPath $templatesArchive -Algorithm SHA256).Hash.ToLowerInvariant()

    $inputManifest = [pscustomobject][ordered]@{
        schemaVersion = 2
        benchmark = "ferrum-linux-time-to-accuracy-ab"
        experiment = $Experiment
        baseline = [pscustomobject][ordered]@{ commit = $baselineCommit; tree = $baselineTree; archiveSha256 = $baselineArchiveSha256 }
        candidate = [pscustomobject][ordered]@{ commit = $candidateCommit; tree = $candidateTree; archiveSha256 = $candidateArchiveSha256 }
        relationship = [pscustomobject][ordered]@{
            mode = $(if ($simplecExperiment) { "identical-source" } else { "direct-child-exact-paths" })
            directChild = !$simplecExperiment
            identicalSource = $simplecExperiment
            exactChangedPaths = [string[]]@($effectiveChangedPaths | Sort-Object)
        }
        cargoLock = [pscustomobject][ordered]@{ blob = $baselineCargoLockBlob; sha256 = $baselineCargoLockSha256 }
        buildPolicy = [pscustomobject][ordered]@{ mode = $buildPolicyMode; sameBinary = $simplecExperiment }
        pressureSolver = $PressureSolver
        baselineRelTol = [pscustomobject][ordered]@{ p = "0"; U = "0" }
        candidateRelTol = [pscustomobject][ordered]@{ p = $pressureRelTolText; U = $momentumRelTolText }
        consistentPolicy = [pscustomobject][ordered]@{ baseline = $baselineSimpleConsistent; candidate = $candidateSimpleConsistent }
        maxSimpleIterations = $MaxSimpleIterations
        controls = [pscustomobject][ordered]@{ archiveSha256 = $controlsArchiveSha256; files = $manifestControls }
        cases = $manifestCases
    }
    $manifestPath = Join-Path $stageRoot "input-manifest.json"
    $inputManifest | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $manifestPath -Encoding UTF8
    $expectedInputManifestSha256 = (Get-FileHash -LiteralPath $manifestPath -Algorithm SHA256).Hash.ToLowerInvariant()

    $outputArchive = Join-Path $stageRoot "ferrum-linux-tta-ab-results.tar"
    $boundWorkerWslPath = ConvertTo-MatchedWslPath (Join-Path $controlsRoot "run_ferrum_linux_tta_ab_worker.sh") $Distro
    $runArguments = @(
        "-d", $Distro, "--", "bash", "-c", $workerBootstrap, "ferrum-linux-tta-ab-worker", $boundWorkerWslPath,
        "--rust-toolchain", $RustToolchain, "--cpu-set", $CpuSet, "--build-variant", $BuildVariant,
        "--experiment", $Experiment,
        "--warmup-runs", $WarmupRuns.ToString([System.Globalization.CultureInfo]::InvariantCulture),
        "--measured-runs", $MeasuredRuns.ToString([System.Globalization.CultureInfo]::InvariantCulture),
        "--pressure-solver", $PressureSolver,
        "--max-simple-iterations", $MaxSimpleIterations.ToString([System.Globalization.CultureInfo]::InvariantCulture),
        "--candidate-pressure-reltol", $pressureRelTolText,
        "--candidate-momentum-reltol", $momentumRelTolText,
        "--baseline-archive", (ConvertTo-MatchedWslPath $baselineArchive $Distro),
        "--baseline-archive-sha256", $baselineArchiveSha256, "--baseline-commit", $baselineCommit, "--baseline-tree", $baselineTree,
        "--candidate-archive", (ConvertTo-MatchedWslPath $candidateArchive $Distro),
        "--candidate-archive-sha256", $candidateArchiveSha256, "--candidate-commit", $candidateCommit, "--candidate-tree", $candidateTree,
        "--templates-archive", (ConvertTo-MatchedWslPath $templatesArchive $Distro),
        "--templates-archive-sha256", $templatesArchiveSha256,
        "--controls-archive", (ConvertTo-MatchedWslPath $controlsArchive $Distro),
        "--controls-archive-sha256", $controlsArchiveSha256,
        "--manifest", (ConvertTo-MatchedWslPath $manifestPath $Distro),
        "--output-archive", (ConvertTo-MatchedWslPath $outputArchive $Distro)
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
    if ($workerExitCode -ne 0) { throw "Ferrum Linux TTA A/B worker failed with exit code $workerExitCode" }
    if ((Get-FileHash -LiteralPath $manifestPath -Algorithm SHA256).Hash.ToLowerInvariant() -cne $expectedInputManifestSha256) {
        throw "input manifest changed during benchmark execution"
    }
    Assert-ControlSourcesUnchanged "during benchmark execution"
    if (!(Test-Path -LiteralPath $outputArchive -PathType Leaf)) { throw "worker did not return its result archive" }
    $sidecarPath = "$outputArchive.sha256"
    if (!(Test-Path -LiteralPath $sidecarPath -PathType Leaf)) { throw "worker did not return result archive SHA sidecar" }
    $actualArchiveSha256 = (Get-FileHash -LiteralPath $outputArchive -Algorithm SHA256).Hash.ToLowerInvariant()
    if ((Get-Content -LiteralPath $sidecarPath -Raw).Trim() -ne $actualArchiveSha256) { throw "result archive SHA-256 verification failed" }
    Assert-MatchedSafeTarArchive $outputArchive "Ferrum Linux TTA A/B result"

    Reset-MatchedTargetDirectory $OutRoot $TargetRoot
    Assert-MatchedNoReparsePath $OutRoot $TargetRoot
    & tar -xf $outputArchive -C $OutRoot
    if ($LASTEXITCODE -ne 0) { throw "could not extract Ferrum Linux TTA A/B result archive" }
    Copy-Item -LiteralPath $manifestPath -Destination (Join-Path $OutRoot "input-manifest.json") -Force

    $outputControlsRoot = Join-Path $OutRoot "controls"
    $actualOutputControlNames = [string[]]@(Get-ChildItem -LiteralPath $outputControlsRoot -Force -File | ForEach-Object { $_.Name } | Sort-Object)
    $expectedOutputControlNames = [string[]]@($manifestControls.name | Sort-Object)
    if (@(Compare-Object $expectedOutputControlNames $actualOutputControlNames -CaseSensitive).Count -ne 0 -or
        @(Get-ChildItem -LiteralPath $outputControlsRoot -Force -Directory).Count -ne 0) {
        throw "result control-file inventory differs from the exact manifest"
    }
    foreach ($binding in $manifestControls) {
        $actual = (Get-FileHash -LiteralPath (Join-Path $outputControlsRoot $binding.name) -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($actual -ne $binding.sha256) { throw "result control hash differs: $($binding.name)" }
    }
    if ((Get-Content -LiteralPath (Join-Path $OutRoot "metadata\controls-archive-sha256.txt") -Raw).Trim() -ne $controlsArchiveSha256) {
        throw "worker control archive hash binding differs"
    }
    $actualBuildPolicyMode = (Get-Content -LiteralPath (Join-Path $OutRoot "metadata\build-policy-mode.txt") -Raw).Trim()
    $actualExperiment = (Get-Content -LiteralPath (Join-Path $OutRoot "metadata\experiment.txt") -Raw).Trim()
    $baselineBinarySha256 = (Get-Content -LiteralPath (Join-Path $OutRoot "metadata\baseline-binary-sha256.txt") -Raw).Trim()
    $candidateBinarySha256 = (Get-Content -LiteralPath (Join-Path $OutRoot "metadata\candidate-binary-sha256.txt") -Raw).Trim()
    if ($actualExperiment -cne $Experiment -or $actualBuildPolicyMode -cne $buildPolicyMode -or $baselineBinarySha256 -notmatch '^[0-9a-f]{64}$' -or
        $candidateBinarySha256 -notmatch '^[0-9a-f]{64}$') {
        throw "worker build-policy or binary SHA-256 metadata differs"
    }
    if ($simplecExperiment -and $baselineBinarySha256 -cne $candidateBinarySha256) {
        throw "simplec did not use one exact shared binary"
    }
    Assert-ControlSourcesUnchanged "after result extraction"

    function Get-ArtifactRelativePath([string]$Path) {
        $rootFull = [System.IO.Path]::GetFullPath($OutRoot).TrimEnd("\", "/")
        $pathFull = [System.IO.Path]::GetFullPath($Path)
        if (!(Test-MatchedPathUnder $pathFull $rootFull)) { throw "artifact path escaped output root: $pathFull" }
        return $pathFull.Substring($rootFull.Length).TrimStart("\", "/").Replace("\", "/")
    }

    $expectedProofReports = [System.Collections.Generic.Dictionary[string, object]]::new([System.StringComparer]::Ordinal)
    foreach ($case in $caseDefinitions) {
        foreach ($kind in @("warmup", "measured")) {
            $count = if ($kind -eq "warmup") { $WarmupRuns } else { $MeasuredRuns }
            for ($ordinal = 1; $ordinal -le $count; $ordinal++) {
                foreach ($refName in @("baseline", "candidate")) {
                    $relative = Get-TtaExpectedValidatedReportRelativePath $case.name $kind $ordinal $refName
                    if ($expectedProofReports.ContainsKey($relative)) { throw "expected validated report proof identity is duplicated" }
                    $expectedProofReports.Add($relative, [pscustomobject][ordered]@{ case = $case.name; kind = $kind; ordinal = [long]$ordinal; ref = $refName })
                }
            }
        }
        foreach ($refName in @("baseline", "candidate")) {
            $relative = Get-TtaExpectedValidatedReportRelativePath $case.name "oracle" 0 $refName
            if ($expectedProofReports.ContainsKey($relative)) { throw "expected validated report proof identity is duplicated" }
            $expectedProofReports.Add($relative, [pscustomobject][ordered]@{ case = $case.name; kind = "oracle"; ordinal = [long]0; ref = $refName })
        }
    }
    $expectedProofReportCount = $caseDefinitions.Count * (2 * ($WarmupRuns + $MeasuredRuns) + 2)
    if ($expectedProofReports.Count -ne $expectedProofReportCount) { throw "expected validated report proof count differs" }

    $proofPath = Join-Path $OutRoot "metadata\exact-report-validation.json"
    $proofHashPath = Join-Path $OutRoot "metadata\exact-report-validation.sha256"
    foreach ($path in @($proofPath, $proofHashPath)) {
        if (!(Test-Path -LiteralPath $path -PathType Leaf)) { throw "worker exact-report validation proof is missing: $path" }
        Assert-MatchedNoReparsePath $path $OutRoot
    }
    $expectedProofInventory = [string[]]@("metadata/exact-report-validation.json", "metadata/exact-report-validation.sha256")
    $actualProofInventory = [string[]]@(Get-ChildItem -LiteralPath $OutRoot -Recurse -Force | Where-Object {
        $_.Name -like "exact-report-validation*"
    } | ForEach-Object {
        if ($_.PSIsContainer) { throw "worker exact-report validation proof inventory contains a directory: $($_.FullName)" }
        Assert-MatchedNoReparsePath $_.FullName $OutRoot
        Get-ArtifactRelativePath $_.FullName
    } | Sort-Object)
    if ($actualProofInventory.Count -ne $expectedProofInventory.Count -or
        @(Compare-Object ($expectedProofInventory | Sort-Object) $actualProofInventory -CaseSensitive).Count -ne 0) {
        throw "worker exact-report validation proof artifact inventory differs"
    }
    $expectedProofSha256 = (Get-Content -LiteralPath $proofHashPath -Raw).Trim()
    $proofBytes = [System.IO.File]::ReadAllBytes($proofPath)
    $actualProofSha256 = Get-TtaSha256Bytes $proofBytes
    if ($expectedProofSha256 -notmatch '^[0-9a-f]{64}$' -or $expectedProofSha256 -cne $actualProofSha256) {
        throw "worker exact-report validation proof SHA-256 differs"
    }
    $proofSnapshotPath = [System.IO.Path]::GetTempFileName()
    try {
        [System.IO.File]::WriteAllBytes($proofSnapshotPath, $proofBytes)
        Assert-TtaStrictJsonFile $proofSnapshotPath "worker exact-report validation proof snapshot"
        if ((Get-FileHash -LiteralPath $proofSnapshotPath -Algorithm SHA256).Hash.ToLowerInvariant() -cne $actualProofSha256) {
            throw "worker exact-report validation proof snapshot changed during strict parsing"
        }
        $proofText = [System.Text.UTF8Encoding]::new($false, $true).GetString($proofBytes)
        $proof = $proofText | ConvertFrom-Json
    } finally {
        Remove-Item -LiteralPath $proofSnapshotPath -Force -ErrorAction SilentlyContinue
    }
    Assert-TtaExactJsonProperties $proof @("benchmark", "controlsArchiveSha256", "inputManifestSha256", "reportCount", "reports", "runPolicy", "schemaVersion", "validator") "$.exactReportProof"
    if ((Get-TtaRequiredJsonInteger $proof "schemaVersion" "$.exactReportProof") -ne 2 -or
        (Get-TtaRequiredJsonString $proof "benchmark" "$.exactReportProof") -cne "ferrum-linux-time-to-accuracy-ab" -or
        (Get-TtaRequiredJsonString $proof "validator" "$.exactReportProof") -cne "worker-python-exact-report-contract-v2" -or
        (Get-TtaRequiredJsonString $proof "controlsArchiveSha256" "$.exactReportProof") -cne $controlsArchiveSha256) {
        throw "worker exact-report validation proof metadata differs"
    }
    $metadataManifestPath = Join-Path $OutRoot "metadata\input-manifest.json"
    $metadataManifestSha256 = (Get-FileHash -LiteralPath $metadataManifestPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ((Get-TtaRequiredJsonString $proof "inputManifestSha256" "$.exactReportProof") -cne $expectedInputManifestSha256 -or
        $metadataManifestSha256 -cne $expectedInputManifestSha256 -or
        (Get-FileHash -LiteralPath (Join-Path $OutRoot "input-manifest.json") -Algorithm SHA256).Hash.ToLowerInvariant() -cne $expectedInputManifestSha256) {
        throw "worker exact-report validation proof manifest binding differs"
    }
    $proofRunPolicy = Get-TtaRequiredJsonObject $proof "runPolicy" "$.exactReportProof"
    Assert-TtaExactJsonProperties $proofRunPolicy @("buildPolicy", "candidateRelTol", "consistentPolicy", "experiment", "maxSimpleIterations", "measuredRuns", "pressureSolver", "warmupRuns") "$.exactReportProof.runPolicy"
    $proofCandidateRelTol = Get-TtaRequiredJsonObject $proofRunPolicy "candidateRelTol" "$.exactReportProof.runPolicy"
    Assert-TtaExactJsonProperties $proofCandidateRelTol @("U", "p") "$.exactReportProof.runPolicy.candidateRelTol"
    $proofConsistentPolicy = Get-TtaRequiredJsonObject $proofRunPolicy "consistentPolicy" "$.exactReportProof.runPolicy"
    Assert-TtaExactJsonProperties $proofConsistentPolicy @("baseline", "candidate") "$.exactReportProof.runPolicy.consistentPolicy"
    $proofBuildPolicy = Get-TtaRequiredJsonObject $proofRunPolicy "buildPolicy" "$.exactReportProof.runPolicy"
    Assert-TtaExactJsonProperties $proofBuildPolicy @("baselineBinarySha256", "candidateBinarySha256", "mode", "sameBinary") "$.exactReportProof.runPolicy.buildPolicy"
    if ((Get-TtaRequiredJsonInteger $proofRunPolicy "warmupRuns" "$.exactReportProof.runPolicy") -ne $WarmupRuns -or
        (Get-TtaRequiredJsonInteger $proofRunPolicy "measuredRuns" "$.exactReportProof.runPolicy") -ne $MeasuredRuns -or
        (Get-TtaRequiredJsonInteger $proofRunPolicy "maxSimpleIterations" "$.exactReportProof.runPolicy") -ne $MaxSimpleIterations -or
        (Get-TtaRequiredJsonString $proofRunPolicy "pressureSolver" "$.exactReportProof.runPolicy") -cne $PressureSolver -or
        (Get-TtaRequiredJsonString $proofRunPolicy "experiment" "$.exactReportProof.runPolicy") -cne $Experiment -or
        (Get-TtaRequiredJsonString $proofCandidateRelTol "p" "$.exactReportProof.runPolicy.candidateRelTol") -cne $pressureRelTolText -or
        (Get-TtaRequiredJsonString $proofCandidateRelTol "U" "$.exactReportProof.runPolicy.candidateRelTol") -cne $momentumRelTolText -or
        (Get-TtaRequiredJsonBoolean $proofConsistentPolicy "baseline" "$.exactReportProof.runPolicy.consistentPolicy") -ne $baselineSimpleConsistent -or
        (Get-TtaRequiredJsonBoolean $proofConsistentPolicy "candidate" "$.exactReportProof.runPolicy.consistentPolicy") -ne $candidateSimpleConsistent -or
        (Get-TtaRequiredJsonString $proofBuildPolicy "mode" "$.exactReportProof.runPolicy.buildPolicy") -cne $buildPolicyMode -or
        (Get-TtaRequiredJsonBoolean $proofBuildPolicy "sameBinary" "$.exactReportProof.runPolicy.buildPolicy") -ne $simplecExperiment -or
        (Get-TtaRequiredJsonString $proofBuildPolicy "baselineBinarySha256" "$.exactReportProof.runPolicy.buildPolicy") -cne $baselineBinarySha256 -or
        (Get-TtaRequiredJsonString $proofBuildPolicy "candidateBinarySha256" "$.exactReportProof.runPolicy.buildPolicy") -cne $candidateBinarySha256) {
        throw "worker exact-report validation proof run policy differs"
    }
    if ((Get-TtaRequiredJsonInteger $proof "reportCount" "$.exactReportProof") -ne $expectedProofReportCount) {
        throw "worker exact-report validation proof count differs"
    }
    $proofReports = @(Get-TtaRequiredJsonArray $proof "reports" "$.exactReportProof")
    if ($proofReports.Count -ne $expectedProofReportCount) { throw "worker exact-report validation proof report array count differs" }
    $provenProofReports = [System.Collections.Generic.Dictionary[string, object]]::new([System.StringComparer]::Ordinal)
    $validatedReportBytes = [System.Collections.Generic.Dictionary[string, byte[]]]::new([System.StringComparer]::Ordinal)
    for ($index = 0; $index -lt $proofReports.Count; $index++) {
        $entry = $proofReports[$index]; $entryPath = "$.exactReportProof.reports[$index]"
        Assert-TtaExactJsonProperties $entry @("case", "kind", "ordinal", "ref", "relativePath", "sha256") $entryPath
        # PowerShell variable names are case-insensitive. Keep this distinct from
        # the validated top-level -CaseName parameter.
        $proofCaseName = Get-TtaRequiredJsonString $entry "case" $entryPath
        $kind = Get-TtaRequiredJsonString $entry "kind" $entryPath
        $ordinal = Get-TtaRequiredJsonInteger $entry "ordinal" $entryPath
        $refName = Get-TtaRequiredJsonString $entry "ref" $entryPath
        $relative = Get-TtaRequiredJsonString $entry "relativePath" $entryPath
        $sha256 = Get-TtaRequiredJsonString $entry "sha256" $entryPath
        if ($sha256 -notmatch '^[0-9a-f]{64}$' -or
            $relative -cne (Get-TtaExpectedValidatedReportRelativePath $proofCaseName $kind $ordinal $refName) -or
            !$expectedProofReports.ContainsKey($relative)) {
            throw "$entryPath identity, path, or SHA-256 is invalid"
        }
        $expectedIdentity = $expectedProofReports[$relative]
        if ($proofCaseName -cne $expectedIdentity.case -or $kind -cne $expectedIdentity.kind -or
            $ordinal -ne $expectedIdentity.ordinal -or $refName -cne $expectedIdentity.ref -or
            $provenProofReports.ContainsKey($relative)) {
            throw "$entryPath differs from the unique expected run identity"
        }
        $reportPath = Join-Path $OutRoot $relative.Replace('/', '\')
        if (!(Test-Path -LiteralPath $reportPath -PathType Leaf)) { throw "$entryPath report file is missing" }
        Assert-MatchedNoReparsePath $reportPath $OutRoot
        $reportBytes = [System.IO.File]::ReadAllBytes($reportPath)
        if ((Get-ArtifactRelativePath $reportPath) -cne $relative -or
            (Get-TtaSha256Bytes $reportBytes) -cne $sha256) {
            throw "$entryPath report path or SHA-256 differs"
        }
        $provenProofReports.Add($relative, $entry)
        $validatedReportBytes.Add($relative, $reportBytes)
    }
    $proofRawRoot = Join-Path $OutRoot "raw"
    $actualProofReports = [System.Collections.Generic.Dictionary[string, object]]::new([System.StringComparer]::Ordinal)
    foreach ($file in @(Get-ChildItem -LiteralPath $proofRawRoot -Recurse -Force -File | Where-Object { $_.Name -ceq "solve-report.json" })) {
        Assert-MatchedNoReparsePath $file.FullName $OutRoot
        $relative = Get-ArtifactRelativePath $file.FullName
        if ($actualProofReports.ContainsKey($relative)) { throw "raw solve-report inventory contains a duplicate path" }
        $actualProofReports.Add($relative, $file)
    }
    if ($actualProofReports.Count -ne $expectedProofReportCount -or
        @(Compare-Object ([string[]]@($expectedProofReports.Keys | Sort-Object)) ([string[]]@($actualProofReports.Keys | Sort-Object)) -CaseSensitive).Count -ne 0 -or
        @(Compare-Object ([string[]]@($expectedProofReports.Keys | Sort-Object)) ([string[]]@($provenProofReports.Keys | Sort-Object)) -CaseSensitive).Count -ne 0) {
        throw "expected, proven, and actual raw solve-report inventories differ"
    }
    if ((Get-FileHash -LiteralPath $proofPath -Algorithm SHA256).Hash.ToLowerInvariant() -cne $actualProofSha256 -or
        (Get-Content -LiteralPath $proofHashPath -Raw).Trim() -cne $actualProofSha256) {
        throw "worker exact-report validation proof changed during host validation"
    }
    $validatedReportProof = [pscustomobject][ordered]@{
        contract = "worker-python-exact-report-contract-v2"
        artifact = Get-ArtifactRelativePath $proofPath
        sha256Artifact = Get-ArtifactRelativePath $proofHashPath
        sha256 = $actualProofSha256
        reportCount = $expectedProofReportCount
        inputManifestSha256 = $expectedInputManifestSha256
    }

    function Assert-CanonicalReport([string]$CanonicalPath, [string]$HashPath) {
        $expected = (Get-Content -LiteralPath $HashPath -Raw).Trim()
        $actual = (Get-FileHash -LiteralPath $CanonicalPath -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($expected -notmatch "^[0-9a-f]{64}$" -or $expected -ne $actual) { throw "canonical report SHA-256 verification failed" }
        return $expected
    }

    function Assert-ReportContract($Report, [string]$Description) {
        if ($null -eq $Report -or $Report -isnot [pscustomobject]) { throw "$Description report root must be an object" }
        $solve = Get-TtaRequiredJsonObject $Report "solve" "`$"
        $outer = Get-TtaRequiredJsonObject $Report "outerConvergence" "`$"
        $linear = Get-TtaRequiredJsonObject $Report "linearSolves" "`$"
        $history = @(Get-TtaRequiredJsonArray $Report "history" "`$")
        if (!(Get-TtaRequiredJsonBoolean $solve "converged" "`$.solve") -or
            (Get-TtaRequiredJsonString $solve "stopReason" "`$.solve") -cne "Converged" -or
            (Get-TtaRequiredJsonString $outer "status" "`$.outerConvergence") -cne "converged" -or
            (Get-TtaRequiredJsonString $outer "reason" "`$.outerConvergence") -cne "Converged" -or
            !(Get-TtaRequiredJsonBoolean $outer "configured" "`$.outerConvergence") -or
            !(Get-TtaRequiredJsonBoolean $outer "evaluated" "`$.outerConvergence") -or
            !(Get-TtaRequiredJsonBoolean $outer "converged" "`$.outerConvergence")) {
            throw "$Description did not satisfy the exact configured/evaluated outer-convergence contract"
        }
        $simpleIterations = Get-TtaRequiredJsonInteger $solve "simpleIterations" "`$.solve"
        $solveMomentumIterations = Get-TtaRequiredJsonInteger $solve "momentumLinearIterations" "`$.solve"
        $solvePressureIterations = Get-TtaRequiredJsonInteger $solve "pressureLinearIterations" "`$.solve"
        if ($history.Count -lt 1 -or $simpleIterations -ne $history.Count -or $history.Count -gt $MaxSimpleIterations) {
            throw "$Description SIMPLE history/count contract failed"
        }
        if ((Get-TtaRequiredJsonInteger $linear "momentumComponentNonConvergedSolves" "`$.linearSolves") -ne 0 -or
            (Get-TtaRequiredJsonInteger $linear "pressureCorrectionNonConvergedSolves" "`$.linearSolves") -ne 0) {
            throw "$Description non-converged linear counter is nonzero"
        }
        $historyMomentumIterations = [long]0
        $historyPressureIterations = [long]0
        for ($rowIndex = 0; $rowIndex -lt $history.Count; $rowIndex++) {
            $row = $history[$rowIndex]
            $rowPath = "`$.history[$rowIndex]"
            if ($row -isnot [pscustomobject]) { throw "$Description $rowPath must be an object" }
            if (!(Get-TtaRequiredJsonBoolean $row "pressureCorrectionAccepted" $rowPath) -or
                !(Get-TtaRequiredJsonBoolean $row "momentumLinearConverged" $rowPath) -or
                !(Get-TtaRequiredJsonBoolean $row "pressureLinearConverged" $rowPath)) {
                throw "$Description contains a reject or non-converged linear solve at $rowPath"
            }
            $historyMomentumIterations += Get-TtaRequiredJsonInteger $row "momentumLinearIterations" $rowPath
            $historyPressureIterations += Get-TtaRequiredJsonInteger $row "pressureLinearIterations" $rowPath
            $pressureSolveCount = Get-TtaRequiredJsonInteger $row "pressureLinearSolves" $rowPath
            Assert-TtaPositivePressureSolveCount $pressureSolveCount $rowPath
        }
        if ($historyMomentumIterations -ne $solveMomentumIterations -or $historyPressureIterations -ne $solvePressureIterations) {
            throw "$Description history linear-iteration totals differ from solve totals"
        }
        $timing = Get-TtaRequiredJsonObject $Report "timing" "`$"
        if ((Get-TtaRequiredJsonNumber $timing "solverTotalSeconds" "`$.timing") -lt 0.0) {
            throw "$Description solver timing is negative"
        }
        $mesh = Get-TtaRequiredJsonObject $Report "mesh" "`$"
        if ((Get-TtaRequiredJsonInteger $mesh "cells" "`$.mesh") -lt 1) { throw "$Description mesh cell count is not positive" }
        $continuity = Get-TtaRequiredJsonObject $Report "continuity" "`$"
        $continuityFinal = Get-TtaRequiredJsonObject $continuity "final" "`$.continuity"
        foreach ($name in @("l2Norm", "maxAbs", "sumAbs", "globalSum")) {
            [void](Get-TtaRequiredJsonNumber $continuityFinal $name "`$.continuity.final")
        }
        $fields = Get-TtaRequiredJsonObject $Report "fields" "`$"
        $pressureField = Get-TtaRequiredJsonObject $fields "pressure" "`$.fields"
        [void](Get-TtaRequiredJsonNumber $pressureField "min" "`$.fields.pressure")
        [void](Get-TtaRequiredJsonNumber $pressureField "max" "`$.fields.pressure")
        $pressureAssembly = Get-TtaRequiredJsonObject $Report "pressureAssembly" "`$"
        $correctedPhi = Get-TtaRequiredJsonObject $pressureAssembly "correctedPhi" "`$.pressureAssembly"
        if ((Get-TtaRequiredJsonNumber $correctedPhi "boundarySumAbs" "`$.pressureAssembly.correctedPhi") -lt 0.0) {
            throw "$Description corrected boundary flux magnitude is negative"
        }
    }

    function Assert-EffectiveLinearThresholds($Report, [string]$RefName, [string]$Description) {
        if ($RefName -cne "baseline" -and $RefName -cne "candidate") { throw "$Description has an invalid ref name" }
        $options = Get-TtaRequiredJsonObject $Report "options" "`$"
        $history = @(Get-TtaRequiredJsonArray $Report "history" "`$")
        $expectedMomentumRelTol = if ($RefName -eq "candidate") { $CandidateMomentumRelTol } else { 0.0 }
        $expectedPressureRelTol = if ($RefName -eq "candidate") { $CandidatePressureRelTol } else { 0.0 }
        $expectedConsistent = $simplecExperiment -and $RefName -ceq "candidate"
        $momentumSolver = Get-TtaRequiredJsonString $options "momentumLinearSolver" "`$.options"
        $actualPressureSolver = Get-TtaRequiredJsonString $options "pressureLinearSolver" "`$.options"
        if ($actualPressureSolver -ine $PressureSolver) { throw "$Description pressure solver differs from the requested solver" }
        if ((Get-TtaRequiredJsonBoolean $options "consistent" "`$.options") -ne $expectedConsistent) {
            throw "$Description SIMPLE consistent option differs from the staged case"
        }
        $momentumAbsolute = Get-TtaRequiredJsonNumber $options "momentumLinearTolerance" "`$.options"
        $pressureAbsolute = Get-TtaRequiredJsonNumber $options "pressureLinearTolerance" "`$.options"
        if ($momentumAbsolute -lt 0.0 -or $pressureAbsolute -lt 0.0) { throw "$Description contains a negative absolute tolerance" }
        if ($RefName -eq "candidate" -or $simplecExperiment) {
            if ((Get-TtaRequiredJsonNumber $options "momentumLinearRelativeTolerance" "`$.options") -ne $expectedMomentumRelTol -or
                (Get-TtaRequiredJsonNumber $options "pressureLinearRelativeTolerance" "`$.options") -ne $expectedPressureRelTol) {
                throw "$Description did not report the exact configured candidate relTol controls"
            }
        }
        if ($actualPressureSolver -ieq "gamg") {
            $pressureGamg = Get-TtaRequiredJsonObject $options "pressureGamg" "`$.options"
            if ((Get-TtaRequiredJsonNumber $pressureGamg "relTol" "`$.options.pressureGamg") -ne $expectedPressureRelTol) {
                throw "$Description pressure GAMG relTol differs from the staged case"
            }
        }

        function Get-TtaArrayNumber($Values, [int]$Index, [string]$Path) {
            if ($Index -lt 0 -or $Index -ge $Values.Count) { throw "$Path is absent" }
            $value = $Values[$Index]
            if (!(Test-TtaJsonNumberType $value)) { throw "$Path must be a numeric scalar" }
            $number = [double]$value
            if ([double]::IsNaN($number) -or [double]::IsInfinity($number)) { throw "$Path must be finite" }
            return $number
        }

        function Assert-OneSolve($Solve, [double]$AbsoluteTolerance, [double]$RelativeTolerance, [string]$Solver, [string]$SolveDescription) {
            if ($null -eq $Solve -or $Solve -isnot [pscustomobject]) { throw "$SolveDescription must be an object" }
            $iterations = Get-TtaRequiredJsonInteger $Solve "iterations" $SolveDescription
            if (!(Get-TtaRequiredJsonBoolean $Solve "converged" $SolveDescription)) { throw "$SolveDescription is non-converged" }
            $initial = Get-TtaRequiredJsonNumber $Solve "initialNormalizedResidual" $SolveDescription
            $residualNorm = Get-TtaRequiredJsonNumber $Solve "residualNorm" $SolveDescription
            $final = Get-TtaRequiredJsonNumber $Solve "normalizedResidual" $SolveDescription
            $reportedTarget = Get-TtaRequiredJsonNumber $Solve "effectiveNormalizedTolerance" $SolveDescription
            if ([Math]::Min([Math]::Min($initial, $residualNorm), [Math]::Min($final, $reportedTarget)) -lt 0.0) {
                throw "$SolveDescription contains a negative residual or tolerance"
            }
            $isGamg = $Solver -ieq "gamg"
            $relativeLimit = if (Test-TtaLinearRelTolActive $isGamg $RelativeTolerance) { $RelativeTolerance * $initial } else { 0.0 }
            if ($final -ge $reportedTarget) { throw "$SolveDescription did not satisfy strict final < effective target" }
            $expectedReason = if ($isGamg -and $iterations -eq 0 -and $final -eq 0.0) {
                "ExactZero"
            } elseif ($relativeLimit -gt $AbsoluteTolerance) {
                "RelativeTolerance"
            } else {
                "AbsoluteTolerance"
            }
            if ((Get-TtaRequiredJsonString $Solve "stopReason" $SolveDescription) -cne $expectedReason) {
                throw "$SolveDescription stopReason differs from '$expectedReason'"
            }
            return $iterations
        }

        for ($rowIndex = 0; $rowIndex -lt $history.Count; $rowIndex++) {
            $row = $history[$rowIndex]
            $rowPath = "`$.history[$rowIndex]"
            if ($row -isnot [pscustomobject]) { throw "$Description $rowPath must be an object" }
            $momentumAggregate = Get-TtaRequiredJsonInteger $row "momentumLinearIterations" $rowPath
            $pressureAggregate = Get-TtaRequiredJsonInteger $row "pressureLinearIterations" $rowPath
            $pressureSolveCount = Get-TtaRequiredJsonInteger $row "pressureLinearSolves" $rowPath
            Assert-TtaPositivePressureSolveCount $pressureSolveCount $rowPath
            $hasMomentumTelemetry = $null -ne $row.PSObject.Properties["momentumComponentLinearSolves"]
            $hasPressureTelemetry = $null -ne $row.PSObject.Properties["pressureCorrectionLinearSolves"]
            if ($hasMomentumTelemetry -ne $hasPressureTelemetry) { throw "$Description $rowPath telemetry is partially missing" }
            if ($RefName -eq "candidate" -and !$hasMomentumTelemetry) { throw "$Description candidate report lacks additive per-solve telemetry" }
            if ($hasMomentumTelemetry) {
                $momentumTelemetry = @(Get-TtaRequiredJsonArray $row "momentumComponentLinearSolves" $rowPath)
                $pressureTelemetry = @(Get-TtaRequiredJsonArray $row "pressureCorrectionLinearSolves" $rowPath)
                if ($momentumTelemetry.Count -ne 3) { throw "$Description must report exactly x/y/z momentum solves" }
                $momentumIterationSum = [long]0
                for ($component = 0; $component -lt 3; $component++) {
                    $entry = $momentumTelemetry[$component]
                    $entryPath = "$rowPath.momentumComponentLinearSolves[$component]"
                    if ($entry -isnot [pscustomobject]) { throw "$Description $entryPath must be an object" }
                    $expectedComponent = @("x", "y", "z")[$component]
                    if ((Get-TtaRequiredJsonString $entry "component" $entryPath) -cne $expectedComponent) {
                        throw "$Description momentum telemetry component order is not x/y/z"
                    }
                    $solve = Get-TtaRequiredJsonObject $entry "solve" $entryPath
                    $momentumIterationSum += Assert-OneSolve $solve $momentumAbsolute $expectedMomentumRelTol $momentumSolver "$entryPath.solve"
                }
                if ($momentumIterationSum -ne $momentumAggregate) { throw "$Description per-component momentum iterations do not sum to the aggregate" }
                if ($pressureTelemetry.Count -ne $pressureSolveCount) { throw "$Description pressure telemetry count differs from pressureLinearSolves" }
                $pressureIterationSum = [long]0
                for ($correction = 0; $correction -lt $pressureTelemetry.Count; $correction++) {
                    $entry = $pressureTelemetry[$correction]
                    $entryPath = "$rowPath.pressureCorrectionLinearSolves[$correction]"
                    if ($entry -isnot [pscustomobject]) { throw "$Description $entryPath must be an object" }
                    $reportedCorrection = $correction + 1
                    if ((Get-TtaRequiredJsonInteger $entry "correction" $entryPath) -ne $reportedCorrection) {
                        throw "$Description pressure correction telemetry is not contiguous and 1-based"
                    }
                    $solve = Get-TtaRequiredJsonObject $entry "solve" $entryPath
                    $pressureIterationSum += Assert-OneSolve $solve $pressureAbsolute $expectedPressureRelTol $actualPressureSolver "$entryPath.solve"
                }
                if ($pressureIterationSum -ne $pressureAggregate) { throw "$Description pressure solve iterations do not sum to the aggregate" }
            } else {
                # A pre-feature baseline has only the legacy first/last aggregate fields.
                $initials = @(Get-TtaRequiredJsonArray $row "momentumComponentInitialResiduals" $rowPath)
                $finals = @(Get-TtaRequiredJsonArray $row "momentumComponentNormalizedResidualNorms" $rowPath)
                if ($initials.Count -ne 3 -or $finals.Count -ne 3 -or $pressureSolveCount -lt 1) {
                    throw "$Description legacy baseline residual shape differs"
                }
                for ($component = 0; $component -lt 3; $component++) {
                    $initial = Get-TtaArrayNumber $initials $component "$rowPath.momentumComponentInitialResiduals[$component]"
                    $final = Get-TtaArrayNumber $finals $component "$rowPath.momentumComponentNormalizedResidualNorms[$component]"
                    if ($initial -lt 0.0 -or $final -lt 0.0) { throw "$Description legacy momentum residual is negative" }
                    $relativeLimit = if (Test-TtaLinearRelTolActive ($momentumSolver -ieq "gamg") $expectedMomentumRelTol) { $expectedMomentumRelTol * $initial } else { 0.0 }
                    $target = [Math]::Max($momentumAbsolute, $relativeLimit)
                    if ($final -ge $target) { throw "$Description legacy momentum component $component missed its strict target" }
                }
                $pressureInitial = Get-TtaRequiredJsonNumber $row "pressureCorrectionInitialResidual" $rowPath
                $pressureFinal = Get-TtaRequiredJsonNumber $row "pressureCorrectionNormalizedResidualNorm" $rowPath
                if ($pressureInitial -lt 0.0 -or $pressureFinal -lt 0.0) { throw "$Description legacy pressure residual is negative" }
                $pressureRelativeLimit = if (Test-TtaLinearRelTolActive ($actualPressureSolver -ieq "gamg") $expectedPressureRelTol) { $expectedPressureRelTol * $pressureInitial } else { 0.0 }
                $pressureTarget = [Math]::Max($pressureAbsolute, $pressureRelativeLimit)
                if ($pressureFinal -ge $pressureTarget) { throw "$Description legacy pressure aggregate missed its strict target" }
            }
        }
    }

    $expectedOrder = @{}
    $expectedOrderRows = New-Object System.Collections.Generic.List[object]
    foreach ($case in $caseDefinitions) {
        foreach ($kind in @("warmup", "measured")) {
            $count = if ($kind -eq "warmup") { $WarmupRuns } else { $MeasuredRuns }
            for ($ordinal = 1; $ordinal -le $count; $ordinal++) {
                $refs = if (($ordinal % 2) -eq 1) { @("baseline", "candidate") } else { @("candidate", "baseline") }
                for ($position = 1; $position -le 2; $position++) {
                    $refName = $refs[$position - 1]
                    $key = "$($case.name)|$kind|$ordinal|$refName"
                    $expectedOrder[$key] = $position
                    $expectedOrderRows.Add([pscustomobject][ordered]@{ case = $case.name; kind = $kind; ordinal = $ordinal; position = $position; ref = $refName })
                }
            }
        }
    }
    $orderRows = @(Import-Csv -LiteralPath (Join-Path $OutRoot "metadata\run-order.tsv") -Delimiter "`t")
    if ($orderRows.Count -ne $expectedOrderRows.Count) { throw "run-order row count differs" }
    for ($index = 0; $index -lt $orderRows.Count; $index++) {
        $row = $orderRows[$index]; $expected = $expectedOrderRows[$index]
        if ([string]$row.case -cne [string]$expected.case -or [string]$row.kind -cne [string]$expected.kind -or
            [int]$row.ordinal -ne [int]$expected.ordinal -or [int]$row.position -ne [int]$expected.position -or
            [string]$row.ref -cne [string]$expected.ref) { throw "run-order row $($index + 1) differs from alternating contract" }
    }

    $rawRoot = $proofRawRoot
    function Read-ValidatedTtaReport([string]$Relative, [string]$Description) {
        if (!$validatedReportBytes.ContainsKey($Relative)) { throw "$Description is absent from the validated report byte inventory" }
        try {
            $text = [System.Text.UTF8Encoding]::new($false, $true).GetString($validatedReportBytes[$Relative])
            $report = $text | ConvertFrom-Json
        } catch {
            throw "$Description validated report bytes are not parseable JSON: $($_.Exception.Message)"
        }
        if ($null -eq $report -or $report -isnot [pscustomobject]) { throw "$Description validated report root must be an object" }
        return $report
    }

    function Read-TtaRun($Case, [string]$Kind, [int]$Ordinal, [string]$RefName) {
        $runRoot = Join-Path $rawRoot "$($Case.name)\$Kind-$Ordinal-$RefName"
        foreach ($name in @("canonical-report.json", "canonical-report.sha256", "ferrum.log", "process-time.env", "solve-report.json", "case-fvSolution.sha256", "case\system\fvSolution")) {
            if (!(Test-Path -LiteralPath (Join-Path $runRoot $name) -PathType Leaf)) { throw "$($Case.name) $Kind $Ordinal $RefName is missing $name" }
        }
        $timing = Read-MatchedGnuTime (Join-Path $runRoot "process-time.env")
        if ($timing.exitCode -ne 0 -or $timing.elapsedSeconds -le 0.0) { throw "$($Case.name) $RefName GNU time contract failed" }
        $maxResidentSetKiB = Get-TtaPositiveMaxResidentSetKiB $timing "$($Case.name) $RefName"
        $reportRelative = Get-TtaExpectedValidatedReportRelativePath $Case.name $Kind $Ordinal $RefName
        $report = Read-ValidatedTtaReport $reportRelative "$($Case.name) $Kind $Ordinal $RefName"
        Assert-ReportContract $report "$($Case.name) $Kind $Ordinal $RefName"
        Assert-EffectiveLinearThresholds $report $RefName "$($Case.name) $Kind $Ordinal $RefName"
        $canonicalHash = Assert-CanonicalReport (Join-Path $runRoot "canonical-report.json") (Join-Path $runRoot "canonical-report.sha256")
        $key = "$($Case.name)|$Kind|$Ordinal|$RefName"
        $fvSolutionPath = Join-Path $runRoot "case\system\fvSolution"
        $fvSolutionSha256 = (Get-Content -LiteralPath (Join-Path $runRoot "case-fvSolution.sha256") -Raw).Trim()
        if ($fvSolutionSha256 -notmatch '^[0-9a-f]{64}$' -or
            (Get-FileHash -LiteralPath $fvSolutionPath -Algorithm SHA256).Hash.ToLowerInvariant() -cne $fvSolutionSha256) {
            throw "$($Case.name) $Kind $Ordinal $RefName fvSolution sidecar differs from the exact staged bytes"
        }
        return [pscustomobject][ordered]@{
            kind = $Kind; ordinal = $Ordinal; ref = $RefName; orderPosition = [int]$expectedOrder[$key]
            commonProcessElapsedSeconds = [double]$timing.elapsedSeconds
            processUserSeconds = [double]$timing.userSeconds
            processSystemSeconds = [double]$timing.systemSeconds
            maxResidentSetKiB = $maxResidentSetKiB
            canonicalReportSha256 = $canonicalHash
            fvSolutionSha256 = $fvSolutionSha256
            fvSolutionArtifact = Get-ArtifactRelativePath $fvSolutionPath
            simpleIterations = [int]$report.solve.simpleIterations
            momentumLinearIterations = [int]$report.solve.momentumLinearIterations
            pressureLinearIterations = [int]$report.solve.pressureLinearIterations
            solverInternalSeconds = [double]$report.timing.solverTotalSeconds
            report = $reportRelative
        }
    }

    function Read-TtaOracle($Case, [string]$RefName) {
        $runRoot = Join-Path $rawRoot "$($Case.name)\oracle-$RefName"
        foreach ($name in @("canonical-report.json", "canonical-report.sha256", "ferrum.log", "solve-report.json", "field-values.json", "case-fvSolution.sha256", "case\system\fvSolution")) {
            if (!(Test-Path -LiteralPath (Join-Path $runRoot $name) -PathType Leaf)) { throw "$($Case.name) $RefName oracle is missing $name" }
        }
        foreach ($fieldName in @("U", "p")) {
            if (!(Test-Path -LiteralPath (Join-Path $runRoot "final-fields\$fieldName") -PathType Leaf)) { throw "$($Case.name) $RefName oracle is missing $fieldName" }
        }
        $reportRelative = Get-TtaExpectedValidatedReportRelativePath $Case.name "oracle" 0 $RefName
        $report = Read-ValidatedTtaReport $reportRelative "$($Case.name) $RefName oracle"
        Assert-ReportContract $report "$($Case.name) $RefName oracle"
        Assert-EffectiveLinearThresholds $report $RefName "$($Case.name) $RefName oracle"
        $canonicalHash = Assert-CanonicalReport (Join-Path $runRoot "canonical-report.json") (Join-Path $runRoot "canonical-report.sha256")
        $fieldValues = Get-Content -LiteralPath (Join-Path $runRoot "field-values.json") -Raw | ConvertFrom-Json
        if ($fieldValues -isnot [pscustomobject]) { throw "$($Case.name) $RefName field-value oracle root must be an object" }
        $schemaVersion = Get-TtaRequiredJsonInteger $fieldValues "schemaVersion" "`$"
        $cellCount = Get-TtaRequiredJsonInteger $fieldValues "cellCount" "`$"
        $reportMesh = Get-TtaRequiredJsonObject $report "mesh" "`$"
        $reportCellCount = Get-TtaRequiredJsonInteger $reportMesh "cells" "`$.mesh"
        $uField = Get-TtaRequiredJsonObject $fieldValues "U" "`$"
        $pField = Get-TtaRequiredJsonObject $fieldValues "p" "`$"
        $uValues = @(Get-TtaRequiredJsonArray $uField "values" "`$.U")
        $pValues = @(Get-TtaRequiredJsonArray $pField "values" "`$.p")
        if ($schemaVersion -ne 1 -or $cellCount -ne $reportCellCount -or
            $uValues.Count -ne (3 * $reportCellCount) -or $pValues.Count -ne $reportCellCount) {
            throw "$($Case.name) $RefName field-value oracle shape failed"
        }
        for ($index = 0; $index -lt $uValues.Count; $index++) {
            if (!(Test-TtaJsonNumberType $uValues[$index])) { throw "$($Case.name) $RefName U[$index] must be numeric" }
            $number = [double]$uValues[$index]
            if ([double]::IsNaN($number) -or [double]::IsInfinity($number)) { throw "$($Case.name) $RefName U[$index] must be finite" }
        }
        for ($index = 0; $index -lt $pValues.Count; $index++) {
            if (!(Test-TtaJsonNumberType $pValues[$index])) { throw "$($Case.name) $RefName p[$index] must be numeric" }
            $number = [double]$pValues[$index]
            if ([double]::IsNaN($number) -or [double]::IsInfinity($number)) { throw "$($Case.name) $RefName p[$index] must be finite" }
        }
        $fvSolutionPath = Join-Path $runRoot "case\system\fvSolution"
        $fvSolutionSha256 = (Get-Content -LiteralPath (Join-Path $runRoot "case-fvSolution.sha256") -Raw).Trim()
        if ($fvSolutionSha256 -notmatch '^[0-9a-f]{64}$' -or
            (Get-FileHash -LiteralPath $fvSolutionPath -Algorithm SHA256).Hash.ToLowerInvariant() -cne $fvSolutionSha256) {
            throw "$($Case.name) $RefName oracle fvSolution sidecar differs from the exact staged bytes"
        }
        return [pscustomobject][ordered]@{
            ref = $RefName
            canonicalReportSha256 = $canonicalHash
            fvSolutionSha256 = $fvSolutionSha256
            fvSolutionArtifact = Get-ArtifactRelativePath $fvSolutionPath
            report = $report
            fieldValues = $fieldValues
            manifest = Get-ArtifactRelativePath (Join-Path $runRoot "field-values.json")
        }
    }

    function Get-RelativeDifference([double]$Candidate, [double]$Baseline) {
        if ($Baseline -eq 0.0) { return $(if ($Candidate -eq 0.0) { 0.0 } else { [double]::PositiveInfinity }) }
        return [Math]::Abs($Candidate - $Baseline) / [Math]::Abs($Baseline)
    }

    function Compare-OracleFields($BaselineOracle, $CandidateOracle) {
        if ($null -eq $BaselineOracle -or $null -eq $CandidateOracle) { throw "oracle comparison input is null" }
        Assert-TtaMatchingOracleFieldShapes $BaselineOracle.fieldValues $CandidateOracle.fieldValues "`$.oracleComparison"
        function Convert-TtaOracleArray($FieldValues, [string]$FieldName, [string]$Path) {
            $field = Get-TtaRequiredJsonObject $FieldValues $FieldName $Path
            $values = @(Get-TtaRequiredJsonArray $field "values" "$Path.$FieldName")
            [double[]]$result = New-Object double[] $values.Count
            for ($index = 0; $index -lt $values.Count; $index++) {
                $value = $values[$index]
                if (!(Test-TtaJsonNumberType $value)) { throw "$Path.$FieldName.values[$index] must be numeric" }
                $number = [double]$value
                if ([double]::IsNaN($number) -or [double]::IsInfinity($number)) { throw "$Path.$FieldName.values[$index] must be finite" }
                $result[$index] = $number
            }
            return $result
        }
        [double[]]$baselineU = @(Convert-TtaOracleArray $BaselineOracle.fieldValues "U" "`$.oracleComparison.baseline")
        [double[]]$candidateU = @(Convert-TtaOracleArray $CandidateOracle.fieldValues "U" "`$.oracleComparison.candidate")
        [double[]]$baselineP = @(Convert-TtaOracleArray $BaselineOracle.fieldValues "p" "`$.oracleComparison.baseline")
        [double[]]$candidateP = @(Convert-TtaOracleArray $CandidateOracle.fieldValues "p" "`$.oracleComparison.candidate")
        $uDeltaSquare = 0.0; $uBaseSquare = 0.0; $uDeltaMax = 0.0; $uBaseMax = 0.0
        for ($index = 0; $index -lt $baselineU.Count; $index++) {
            $delta = $candidateU[$index] - $baselineU[$index]
            $uDeltaSquare += $delta * $delta; $uBaseSquare += $baselineU[$index] * $baselineU[$index]
            $uDeltaMax = [Math]::Max($uDeltaMax, [Math]::Abs($delta)); $uBaseMax = [Math]::Max($uBaseMax, [Math]::Abs($baselineU[$index]))
        }
        $baselinePMean = ($baselineP | Measure-Object -Average).Average
        $candidatePMean = ($candidateP | Measure-Object -Average).Average
        $pDeltaSquare = 0.0; $pBaseSquare = 0.0; $pDeltaMax = 0.0; $pBaseMax = 0.0
        for ($index = 0; $index -lt $baselineP.Count; $index++) {
            $baseGauge = $baselineP[$index] - $baselinePMean; $candidateGauge = $candidateP[$index] - $candidatePMean
            $delta = $candidateGauge - $baseGauge
            $pDeltaSquare += $delta * $delta; $pBaseSquare += $baseGauge * $baseGauge
            $pDeltaMax = [Math]::Max($pDeltaMax, [Math]::Abs($delta)); $pBaseMax = [Math]::Max($pBaseMax, [Math]::Abs($baseGauge))
        }
        $uL2 = if ($uBaseSquare -gt 0.0) { [Math]::Sqrt($uDeltaSquare / $uBaseSquare) } elseif ($uDeltaSquare -eq 0.0) { 0.0 } else { [double]::PositiveInfinity }
        $uLinf = if ($uBaseMax -gt 0.0) { $uDeltaMax / $uBaseMax } elseif ($uDeltaMax -eq 0.0) { 0.0 } else { [double]::PositiveInfinity }
        $pL2 = if ($pBaseSquare -gt 0.0) { [Math]::Sqrt($pDeltaSquare / $pBaseSquare) } elseif ($pDeltaSquare -eq 0.0) { 0.0 } else { [double]::PositiveInfinity }
        $pLinf = if ($pBaseMax -gt 0.0) { $pDeltaMax / $pBaseMax } elseif ($pDeltaMax -eq 0.0) { 0.0 } else { [double]::PositiveInfinity }
        return [pscustomobject][ordered]@{ velocityRelativeL2 = $uL2; velocityRelativeLinf = $uLinf; pressureGaugeRelativeL2 = $pL2; pressureGaugeRelativeLinf = $pLinf }
    }

    $caseData = @()
    $accuracyFailures = New-Object System.Collections.Generic.List[string]
    foreach ($case in $caseDefinitions) {
        $baselineRuns = @(); $candidateRuns = @()
        foreach ($kind in @("warmup", "measured")) {
            $count = if ($kind -eq "warmup") { $WarmupRuns } else { $MeasuredRuns }
            for ($ordinal = 1; $ordinal -le $count; $ordinal++) {
                $baselineRuns += Read-TtaRun $case $kind $ordinal "baseline"
                $candidateRuns += Read-TtaRun $case $kind $ordinal "candidate"
            }
        }
        $baselineOracle = Read-TtaOracle $case "baseline"
        $candidateOracle = Read-TtaOracle $case "candidate"
        $baselineHashes = [string[]]@($baselineRuns.canonicalReportSha256 + $baselineOracle.canonicalReportSha256 | Sort-Object -Unique)
        $candidateHashes = [string[]]@($candidateRuns.canonicalReportSha256 + $candidateOracle.canonicalReportSha256 | Sort-Object -Unique)
        if ($baselineHashes.Count -ne 1 -or $candidateHashes.Count -ne 1) { throw "$($case.name) timed/oracle reports are not deterministic within each ref" }
        $baselineSolutionHashes = [string[]]@($baselineRuns.fvSolutionSha256 + $baselineOracle.fvSolutionSha256 | Sort-Object -Unique)
        $candidateSolutionHashes = [string[]]@($candidateRuns.fvSolutionSha256 + $candidateOracle.fvSolutionSha256 | Sort-Object -Unique)
        if ($baselineSolutionHashes.Count -ne 1 -or $candidateSolutionHashes.Count -ne 1 -or $baselineSolutionHashes[0] -eq $candidateSolutionHashes[0]) {
            throw "$($case.name) staged fvSolution mutation contract failed"
        }
        $manifestCase = @($manifestCases | Where-Object { $_.name -ceq $case.name })
        if ($manifestCase.Count -ne 1 -or $baselineSolutionHashes[0] -cne $manifestCase[0].baselineFvSolutionSha256) {
            throw "$($case.name) baseline fvSolution hash differs from the exact manifest"
        }
        $baselineFvSolutionPath = Join-Path $OutRoot $baselineRuns[0].fvSolutionArtifact.Replace('/', '\')
        $candidateFvSolutionPath = Join-Path $OutRoot $candidateRuns[0].fvSolutionArtifact.Replace('/', '\')
        [byte[]]$baselineFvSolutionBytes = [System.IO.File]::ReadAllBytes($baselineFvSolutionPath)
        [byte[]]$candidateFvSolutionBytes = [System.IO.File]::ReadAllBytes($candidateFvSolutionPath)
        [void](Invoke-TtaSimpleConsistentBytes $baselineFvSolutionBytes $false $false "$($case.name) result baseline")
        if ($simplecExperiment) {
            [void](Invoke-TtaSimpleConsistentBytes $candidateFvSolutionBytes $true $false "$($case.name) result candidate")
            if ($candidateSolutionHashes[0] -cne $manifestCase[0].candidateFvSolutionSha256) {
                throw "$($case.name) candidate SIMPLEC fvSolution hash differs from the exact manifest"
            }
            $expectedCandidateTransform = Invoke-TtaSimpleConsistentBytes $baselineFvSolutionBytes $false $true "$($case.name) exact SIMPLEC delta"
            if ([Convert]::ToBase64String([byte[]]$expectedCandidateTransform.bytes) -cne [Convert]::ToBase64String($candidateFvSolutionBytes)) {
                throw "$($case.name) candidate fvSolution differs by more than the direct SIMPLE.consistent false-to-true token"
            }
        } else {
            [void](Invoke-TtaSimpleConsistentBytes $candidateFvSolutionBytes $false $false "$($case.name) result candidate")
        }

        $fields = Compare-OracleFields $baselineOracle $candidateOracle
        $baselineReport = $baselineOracle.report; $candidateReport = $candidateOracle.report
        $continuityRatios = [ordered]@{}
        $continuityLimits = [ordered]@{}
        $continuityGate = $true
        foreach ($name in @("l2Norm", "maxAbs", "sumAbs", "globalSum")) {
            $baselineValue = [Math]::Abs([double]$baselineReport.continuity.final.$name)
            $candidateValue = [Math]::Abs([double]$candidateReport.continuity.final.$name)
            $continuityRatios[$name] = if ($baselineValue -gt 0.0) { $candidateValue / $baselineValue } elseif ($candidateValue -eq 0.0) { 1.0 } else { [double]::PositiveInfinity }
            $continuityLimits[$name] = [Math]::Max($MaxContinuityRatio * $baselineValue, 1e-16)
            if ($candidateValue -gt $continuityLimits[$name]) { $continuityGate = $false }
        }
        $continuityRatios = [pscustomobject]$continuityRatios
        $continuityLimits = [pscustomobject]$continuityLimits
        $maximumContinuity = [double](@($continuityRatios.PSObject.Properties.Value | Measure-Object -Maximum).Maximum)
        $baselinePressureDrop = [double]$baselineReport.fields.pressure.max - [double]$baselineReport.fields.pressure.min
        $candidatePressureDrop = [double]$candidateReport.fields.pressure.max - [double]$candidateReport.fields.pressure.min
        $pressureDropRelative = Get-RelativeDifference $candidatePressureDrop $baselinePressureDrop
        $baselineFlow = [double]$baselineReport.pressureAssembly.correctedPhi.boundarySumAbs / 2.0
        $candidateFlow = [double]$candidateReport.pressureAssembly.correctedPhi.boundarySumAbs / 2.0
        $flowRelative = Get-RelativeDifference $candidateFlow $baselineFlow
        $baselineMomentumWork = [int]$baselineReport.solve.momentumLinearIterations
        $candidateMomentumWork = [int]$candidateReport.solve.momentumLinearIterations
        $baselinePressureWork = [int]$baselineReport.solve.pressureLinearIterations
        $candidatePressureWork = [int]$candidateReport.solve.pressureLinearIterations
        $baselineWork = $baselineMomentumWork + $baselinePressureWork
        $candidateWork = $candidateMomentumWork + $candidatePressureWork
        $workReduction = if ($baselineWork -gt 0) { ($baselineWork - $candidateWork) / [double]$baselineWork } else { [double]::NegativeInfinity }
        $momentumWorkReduction = if ($baselineMomentumWork -gt 0) { ($baselineMomentumWork - $candidateMomentumWork) / [double]$baselineMomentumWork } else { [double]::NegativeInfinity }
        $pressureWorkReduction = if ($baselinePressureWork -gt 0) { ($baselinePressureWork - $candidatePressureWork) / [double]$baselinePressureWork } else { [double]::NegativeInfinity }

        $gates = [pscustomobject][ordered]@{
            outerConvergedNoRejects = $true
            continuity = $continuityGate
            velocityL2 = $fields.velocityRelativeL2 -le $MaxVelocityRelativeL2
            velocityLinf = $fields.velocityRelativeLinf -le $MaxVelocityRelativeLinf
            pressureGaugeL2 = $fields.pressureGaugeRelativeL2 -le $MaxPressureGaugeRelativeL2
            pressureGaugeLinf = $fields.pressureGaugeRelativeLinf -le $MaxPressureGaugeRelativeLinf
            pressureDrop = $pressureDropRelative -le $MaxPressureDropRelativeDifference
            boundaryFlow = $flowRelative -le $MaxFlowRelativeDifference
            totalWorkReduction = $workReduction -ge $MinimumWorkReduction
            momentumWorkNoMaterialIncrease = $momentumWorkReduction -ge -0.01
            pressureWorkNoMaterialIncrease = $pressureWorkReduction -ge -0.01
        }
        foreach ($property in $gates.PSObject.Properties) {
            if ($property.Value -ne $true) { $accuracyFailures.Add("$($case.name):$($property.Name)") }
        }
        $caseData += [pscustomobject][ordered]@{
            name = $case.name; baselineRuns = $baselineRuns; candidateRuns = $candidateRuns
            baselineCanonicalReportSha256 = $baselineHashes[0]; candidateCanonicalReportSha256 = $candidateHashes[0]
            baselineOracle = $baselineOracle; candidateOracle = $candidateOracle
            accuracy = [pscustomobject][ordered]@{
                fieldDifferences = $fields; continuityRatios = $continuityRatios; continuityLimits = $continuityLimits; maximumContinuityRatio = $maximumContinuity
                pressureDropIndicator = [pscustomobject][ordered]@{ baseline = $baselinePressureDrop; candidate = $candidatePressureDrop; relativeDifference = $pressureDropRelative }
                boundaryFlowMagnitude = [pscustomobject][ordered]@{ baseline = $baselineFlow; candidate = $candidateFlow; relativeDifference = $flowRelative }
                linearWork = [pscustomobject][ordered]@{
                    total = [pscustomobject][ordered]@{ baseline = $baselineWork; candidate = $candidateWork; reduction = $workReduction }
                    momentum = [pscustomobject][ordered]@{ baseline = $baselineMomentumWork; candidate = $candidateMomentumWork; reduction = $momentumWorkReduction }
                    pressure = [pscustomobject][ordered]@{ baseline = $baselinePressureWork; candidate = $candidatePressureWork; reduction = $pressureWorkReduction; preferredFivePercentReductionMet = $pressureWorkReduction -ge 0.05 }
                }
                gates = $gates
            }
        }
    }

    if ($accuracyFailures.Count -gt 0) {
        $rejectionPath = Join-Path $OutRoot "accuracy-rejection.json"
        [pscustomobject][ordered]@{
            schemaVersion = 2; benchmark = "ferrum-linux-time-to-accuracy-ab"; experiment = $Experiment; performanceClassified = $false
            consistentPolicy = [pscustomobject][ordered]@{ baseline = $baselineSimpleConsistent; candidate = $candidateSimpleConsistent }
            failures = [string[]]$accuracyFailures; cases = @($caseData | ForEach-Object { [pscustomobject][ordered]@{ name = $_.name; accuracy = $_.accuracy } })
        } | ConvertTo-Json -Depth 16 | Set-Content -LiteralPath $rejectionPath -Encoding UTF8
        throw "accuracy/work gates rejected the candidate before performance classification: $($accuracyFailures -join ', ')"
    }

    function Get-RefMedian($Runs, [string]$Name) {
        return Get-MatchedMedian ([double[]]@($Runs | Where-Object { $_.kind -eq "measured" } | ForEach-Object { [double]$_.$Name }))
    }
    $caseResults = @()
    $performanceFailures = New-Object System.Collections.Generic.List[string]
    $minimumWins = [int][Math]::Ceiling(0.7 * $MeasuredRuns)
    foreach ($case in $caseData) {
        $ratios = [double[]]@(); $candidateFirst = [double[]]@(); $candidateSecond = [double[]]@(); $wins = 0
        for ($ordinal = 1; $ordinal -le $MeasuredRuns; $ordinal++) {
            $baseline = @($case.baselineRuns | Where-Object { $_.kind -eq "measured" -and $_.ordinal -eq $ordinal })[0]
            $candidate = @($case.candidateRuns | Where-Object { $_.kind -eq "measured" -and $_.ordinal -eq $ordinal })[0]
            $ratio = $candidate.commonProcessElapsedSeconds / $baseline.commonProcessElapsedSeconds
            $ratios += $ratio
            if ($candidate.orderPosition -eq 1) { $candidateFirst += $ratio } else { $candidateSecond += $ratio }
            if ($ratio -lt 1.0) { $wins++ }
        }
        $pairedMedian = Get-MatchedMedian $ratios; $pairedMad = Get-MatchedMedianAbsoluteDeviation $ratios
        $firstMedian = Get-MatchedMedian $candidateFirst; $secondMedian = Get-MatchedMedian $candidateSecond
        $ratioOfMedians = (Get-RefMedian $case.candidateRuns "commonProcessElapsedSeconds") / (Get-RefMedian $case.baselineRuns "commonProcessElapsedSeconds")
        $gates = [pscustomobject][ordered]@{
            bothOrderCohortsFaster = $firstMedian -lt 1.0 -and $secondMedian -lt 1.0
            medianRatioAtMostThreshold = $pairedMedian -le $MaximumMedianRatio -and $ratioOfMedians -le $MaximumMedianRatio
            gainExceedsTwiceMad = (1.0 - $pairedMedian) -gt (2.0 * $pairedMad)
            winsAtLeastSeventyPercent = $wins -ge $minimumWins
        }
        foreach ($property in $gates.PSObject.Properties) {
            if ($property.Value -ne $true) { $performanceFailures.Add("$($case.name):$($property.Name)") }
        }
        $caseResults += [pscustomobject][ordered]@{
            name = $case.name; accuracy = $case.accuracy
            deterministicReports = [pscustomobject][ordered]@{ baseline = $case.baselineCanonicalReportSha256; candidate = $case.candidateCanonicalReportSha256; withinRefExact = $true }
            baseline = [pscustomobject][ordered]@{
                medianProcessElapsedSeconds = Get-RefMedian $case.baselineRuns "commonProcessElapsedSeconds"
                medianSolverInternalSeconds = Get-RefMedian $case.baselineRuns "solverInternalSeconds"
                measuredRuns = @($case.baselineRuns | Where-Object { $_.kind -eq "measured" })
            }
            candidate = [pscustomobject][ordered]@{
                medianProcessElapsedSeconds = Get-RefMedian $case.candidateRuns "commonProcessElapsedSeconds"
                medianSolverInternalSeconds = Get-RefMedian $case.candidateRuns "solverInternalSeconds"
                measuredRuns = @($case.candidateRuns | Where-Object { $_.kind -eq "measured" })
            }
            performance = [pscustomobject][ordered]@{
                primaryMetric = "GNU time elapsed seconds"; ratioOfMedians = $ratioOfMedians; pairedMedianRatio = $pairedMedian; pairedRatioMad = $pairedMad
                wins = $wins; requiredWins = $minimumWins
                candidateFirstMedianRatio = $firstMedian; candidateSecondMedianRatio = $secondMedian; gates = $gates
                accepted = @($gates.PSObject.Properties | Where-Object { $_.Value -ne $true }).Count -eq 0
            }
        }
    }

    $finalReportInventory = [System.Collections.Generic.Dictionary[string, object]]::new([System.StringComparer]::Ordinal)
    foreach ($file in @(Get-ChildItem -LiteralPath $rawRoot -Recurse -Force -File | Where-Object { $_.Name -ceq "solve-report.json" })) {
        Assert-MatchedNoReparsePath $file.FullName $OutRoot
        $relative = Get-ArtifactRelativePath $file.FullName
        if ($finalReportInventory.ContainsKey($relative)) { throw "final raw solve-report inventory contains a duplicate path" }
        $finalReportInventory.Add($relative, $file)
    }
    if ($finalReportInventory.Count -ne $expectedProofReportCount -or
        @(Compare-Object ([string[]]@($expectedProofReports.Keys | Sort-Object)) ([string[]]@($finalReportInventory.Keys | Sort-Object)) -CaseSensitive).Count -ne 0) {
        throw "final raw solve-report inventory differs from the validated proof"
    }
    foreach ($relative in $expectedProofReports.Keys) {
        $expectedReportSha256 = [string]$provenProofReports[$relative].sha256
        if ((Get-TtaSha256Bytes $validatedReportBytes[$relative]) -cne $expectedReportSha256 -or
            (Get-FileHash -LiteralPath $finalReportInventory[$relative].FullName -Algorithm SHA256).Hash.ToLowerInvariant() -cne $expectedReportSha256) {
            throw "validated raw solve-report changed before summary generation: $relative"
        }
    }
    if ((Get-FileHash -LiteralPath $proofPath -Algorithm SHA256).Hash.ToLowerInvariant() -cne $actualProofSha256 -or
        (Get-Content -LiteralPath $proofHashPath -Raw).Trim() -cne $actualProofSha256 -or
        (Get-FileHash -LiteralPath $manifestPath -Algorithm SHA256).Hash.ToLowerInvariant() -cne $expectedInputManifestSha256 -or
        (Get-FileHash -LiteralPath $metadataManifestPath -Algorithm SHA256).Hash.ToLowerInvariant() -cne $expectedInputManifestSha256 -or
        (Get-FileHash -LiteralPath (Join-Path $OutRoot "input-manifest.json") -Algorithm SHA256).Hash.ToLowerInvariant() -cne $expectedInputManifestSha256) {
        throw "exact report proof or input manifest changed before summary generation"
    }
    Assert-MatchedNoReparsePath $outputControlsRoot $OutRoot
    $finalOutputControlItems = @(Get-ChildItem -LiteralPath $outputControlsRoot -Force)
    $finalOutputControlNames = [string[]]@($finalOutputControlItems | ForEach-Object { $_.Name } | Sort-Object)
    $expectedOutputControlNames = [string[]]@($manifestControls.name | Sort-Object)
    if ($finalOutputControlItems.Count -ne $expectedOutputControlNames.Count -or
        @($finalOutputControlItems | Where-Object { $_.PSIsContainer }).Count -ne 0 -or
        @(Compare-Object $expectedOutputControlNames $finalOutputControlNames -CaseSensitive).Count -ne 0) {
        throw "final result control-file inventory differs from the exact manifest"
    }
    foreach ($binding in $manifestControls) {
        $controlPath = Join-Path $outputControlsRoot $binding.name
        if (!(Test-Path -LiteralPath $controlPath -PathType Leaf)) { throw "final result control file is missing: $($binding.name)" }
        Assert-MatchedNoReparsePath $controlPath $OutRoot
        if ((Get-FileHash -LiteralPath $controlPath -Algorithm SHA256).Hash.ToLowerInvariant() -cne $binding.sha256) {
            throw "final result control hash differs: $($binding.name)"
        }
    }
    Assert-ControlSourcesUnchanged "before summary generation"

    $summary = [pscustomobject][ordered]@{
        schemaVersion = 2
        benchmark = "ferrum-linux-time-to-accuracy-ab"
        experiment = $Experiment
        generatedAtUtc = [DateTime]::UtcNow.ToString("o")
        baseline = [pscustomobject][ordered]@{ ref = $BaselineRef; commit = $baselineCommit; tree = $baselineTree; archiveSha256 = $baselineArchiveSha256; relTol = [pscustomobject][ordered]@{ p = 0.0; U = 0.0 }; consistent = $baselineSimpleConsistent }
        candidate = [pscustomobject][ordered]@{ ref = $CandidateRef; commit = $candidateCommit; tree = $candidateTree; archiveSha256 = $candidateArchiveSha256; relTol = [pscustomobject][ordered]@{ p = $CandidatePressureRelTol; U = $CandidateMomentumRelTol }; consistent = $candidateSimpleConsistent }
        relationship = [pscustomobject][ordered]@{
            mode = $(if ($simplecExperiment) { "identical-source" } else { "direct-child-exact-paths" })
            candidateDirectChildOfBaseline = !$simplecExperiment
            identicalSource = $simplecExperiment
            exactChangedPaths = [string[]]@($effectiveChangedPaths | Sort-Object)
            cargoLockBlob = $baselineCargoLockBlob
            cargoLockSha256 = $baselineCargoLockSha256
        }
        controls = [pscustomobject][ordered]@{
            archiveSha256 = $controlsArchiveSha256
            files = @($manifestControls | ForEach-Object {
                [pscustomobject][ordered]@{ name = $_.name; sha256 = $_.sha256; artifact = "controls/$($_.name)" }
            })
            sourceHashesRevalidatedAfterRun = $true
        }
        exactReportValidation = $validatedReportProof
        sourceWorktreeCleanAtLaunch = $sourceWorktreeCleanAtLaunch
        launchStatusPorcelain = $launchStatus
        pressureSolver = $PressureSolver
        build = [pscustomobject][ordered]@{
            variant = $BuildVariant; rustToolchain = $RustToolchain; cargoIncremental = 0
            mode = $buildPolicyMode; sameBinary = $simplecExperiment
            baselineBinarySha256 = $baselineBinarySha256; candidateBinarySha256 = $candidateBinarySha256
        }
        platform = [pscustomobject][ordered]@{
            lane = "WSL2 Linux ext4"; distro = (Get-Content -LiteralPath (Join-Path $OutRoot "metadata\distro-release.txt") -Raw).Trim()
            kernel = (Get-Content -LiteralPath (Join-Path $OutRoot "metadata\uname.txt") -Raw).Trim()
            cpuModel = (Get-Content -LiteralPath (Join-Path $OutRoot "metadata\cpu-model.txt") -Raw).Trim(); cpuSet = $CpuSet
        }
        policy = [pscustomobject][ordered]@{
            warmupRuns = $WarmupRuns; measuredRuns = $MeasuredRuns; alternatingOrder = $true; balancedCohorts = $true
            separateBuilds = !$simplecExperiment; sharedSingleBuild = $simplecExperiment; ext4Workspace = $true; timedRunsDoNotWriteFields = $true; untimedFinalFieldOracle = $true
            accuracyBeforePerformance = $true; noSteadyFinalRelTolOverride = $true; maxSimpleIterations = $MaxSimpleIterations
        }
        gates = [pscustomobject][ordered]@{
            maxContinuityMultiplier = $MaxContinuityRatio; continuityAbsoluteFloor = 1e-16; maxVelocityRelativeL2 = $MaxVelocityRelativeL2; maxVelocityRelativeLinf = $MaxVelocityRelativeLinf
            maxPressureGaugeRelativeL2 = $MaxPressureGaugeRelativeL2; maxPressureGaugeRelativeLinf = $MaxPressureGaugeRelativeLinf
            maxPressureDropRelativeDifference = $MaxPressureDropRelativeDifference; maxFlowRelativeDifference = $MaxFlowRelativeDifference
            minimumWorkReduction = $MinimumWorkReduction; maximumMedianRatio = $MaximumMedianRatio; minimumWins = $minimumWins
            gainMustExceedTwiceMad = $true; bothOrderCohortsMustBeFaster = $true
        }
        metricDefinitions = [pscustomobject][ordered]@{
            primaryPerformance = "GNU /usr/bin/time process elapsed seconds"
            velocity = "relative internal-field L2 and componentwise Linf"
            pressure = "relative L2/Linf after subtracting each field's cell mean (gauge aligned)"
            pressureDrop = "internal pressure max minus min; an indicator, not boundary-area averaging"
            flow = "half the correctedPhi boundary sumAbs; exact for a balanced one-inlet/one-outlet case, otherwise an aggregate indicator"
            work = "momentum plus pressure linear iterations in the converged untimed oracle"
        }
        resultArchiveSha256 = $actualArchiveSha256
        performanceAccepted = $performanceFailures.Count -eq 0
        performanceFailures = [string[]]$performanceFailures
        cases = $caseResults
    }
    $jsonPath = Join-Path $OutRoot "summary.json"
    $markdownPath = Join-Path $OutRoot "summary.md"
    $summary | ConvertTo-Json -Depth 24 | Set-Content -LiteralPath $jsonPath -Encoding UTF8
    $lines = New-Object System.Collections.Generic.List[string]
    $lines.Add("# Ferrum Linux Time-to-Accuracy A/B Benchmark")
    $lines.Add("")
    if ($simplecExperiment) {
        $lines.Add("Experiment: ``simplec`` on one exact source/tree/archive and one shared built binary ``$baselineBinarySha256``.")
        $lines.Add("Baseline/candidate: ``$baselineCommit`` with p/U relTol 0/0 and exact direct ``SIMPLE.consistent false -> true`` control delta.")
    } else {
        $lines.Add("Experiment: ``relTol``. Baseline: ``$baselineCommit`` (p/U relTol 0); candidate: ``$candidateCommit`` (p=$pressureRelTolText, U=$momentumRelTolText).")
        $lines.Add("Exact changed paths: ``$($effectiveChangedPaths -join '`, `')``")
    }
    $lines.Add("WSL2/ext4/build: ``$Distro`` / ``$BuildVariant``; warm-up/measured pairs: ``$WarmupRuns/$MeasuredRuns``.")
    $lines.Add("")
    if ($simplecExperiment) {
        $lines.Add("Accuracy and work gates passed before performance was classified. Every report attested ``options.consistent`` false/true while both p/U relTol controls remained exactly zero.")
    } else {
        $lines.Add("Accuracy and work gates passed before performance was classified. Static relTol remains active in the accepting steady SIMPLE step, matching OpenFOAM Foundation 13 steady semantics.")
    }
    $lines.Add("")
    $lines.Add("| Case | U rel L2/Linf | p gauge rel L2/Linf | continuity max ratio | work reduction | elapsed ratio medians | paired median (MAD) | wins | cohorts | accepted |")
    $lines.Add("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |")
    foreach ($case in $caseResults) {
        $lines.Add(("| {0} | {1}/{2} | {3}/{4} | {5} | {6} | {7} | {8} ({9}) | {10}/{11} | {12}/{13} | {14} |" -f
            $case.name,
            (Format-MatchedReportNumber $case.accuracy.fieldDifferences.velocityRelativeL2), (Format-MatchedReportNumber $case.accuracy.fieldDifferences.velocityRelativeLinf),
            (Format-MatchedReportNumber $case.accuracy.fieldDifferences.pressureGaugeRelativeL2), (Format-MatchedReportNumber $case.accuracy.fieldDifferences.pressureGaugeRelativeLinf),
            (Format-MatchedReportNumber $case.accuracy.maximumContinuityRatio), (Format-MatchedReportNumber $case.accuracy.linearWork.total.reduction),
            (Format-MatchedReportNumber $case.performance.ratioOfMedians), (Format-MatchedReportNumber $case.performance.pairedMedianRatio),
            (Format-MatchedReportNumber $case.performance.pairedRatioMad), $case.performance.wins, $case.performance.requiredWins,
            (Format-MatchedReportNumber $case.performance.candidateFirstMedianRatio), (Format-MatchedReportNumber $case.performance.candidateSecondMedianRatio), $case.performance.accepted))
    }
    $lines.Add("")
    $lines.Add("Pressure drop and flow are current report-based indicators (cell pressure range and half boundary |phi| sum), not patch-area averages. The harness records that limitation explicitly and fails every configured numerical/performance gate closed.")
    Set-Content -LiteralPath $markdownPath -Value $lines -Encoding UTF8
    if ($performanceFailures.Count -gt 0) { throw "performance gates rejected the candidate: $($performanceFailures -join ', ')" }
    $completed = $true
    Write-Output "wrote Ferrum Linux TTA A/B JSON: $jsonPath"
    Write-Output "wrote Ferrum Linux TTA A/B Markdown: $markdownPath"
} finally {
    if ($completed -and (Test-Path -LiteralPath $stageRoot)) {
        Remove-Item -LiteralPath $stageRoot -Recurse -Force
    } elseif (!$completed -and (Test-Path -LiteralPath $stageRoot)) {
        Write-Warning "host staging preserved after failure: $stageRoot"
    }
}
