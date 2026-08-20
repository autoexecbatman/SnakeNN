# Mutation test for the value normalisation in value_range.cpp. Mutants compile beside
# copies of every header and source the unit needs, because MSVC resolves a quoted
# include from the including file's own directory first and a mutant left anywhere else
# is ignored - which reports as a survivor.

$src = "D:/repo/snakeNN/src"
$work = "$env:TEMP/mutants_value_range"
if (Test-Path $work) { Remove-Item $work -Recurse -Force }
New-Item -ItemType Directory -Force $work | Out-Null

$support = @("value_range.h", "value_range.cpp", "value_range_test.cpp")
$sources = "value_range.cpp value_range_test.cpp"

# The unmutated code, compiled and run exactly as every mutant will be. Without this a
# missing header kills every mutant and a harness that tests nothing is
# indistinguishable from one that kills everything.
$baseline = Join-Path $work "_baseline"
New-Item -ItemType Directory -Force $baseline | Out-Null
foreach ($file in $support) { Copy-Item "$src/$file" $baseline }
Push-Location $baseline
$null = Invoke-Expression "cl /nologo /std:c++20 /EHsc /O2 $sources /Fe:baseline.exe" 2>&1
$baseline_built = $LASTEXITCODE -eq 0
if ($baseline_built) { $null = & ./baseline.exe 2>&1; $baseline_passes = $LASTEXITCODE -eq 0 }
Pop-Location
if (-not $baseline_built)
{
    Write-Host "BASELINE DOES NOT COMPILE - every mutant would report as killed. Stopping."
    exit 1
}
if (-not $baseline_passes)
{
    Write-Host "BASELINE FAILS ITS OWN TESTS - a kill would mean nothing. Stopping."
    exit 1
}
Write-Host "baseline compiles and passes; mutants below are measured against it`n"

$mutants = @(
    # The ends are a minimum and a maximum. Swapping them inverts every normalised value.
    @{ name = "ends_swapped"; from = "    lowest_ = std::min(lowest_, value);`r`n    highest_ = std::max(highest_, value);"; to = "    lowest_ = std::max(lowest_, value);`r`n    highest_ = std::min(highest_, value);" },

    # A running minimum, not the last value seen.
    @{ name = "lowest_takes_last"; from = "lowest_ = std::min(lowest_, value);"; to = "lowest_ = value;" },
    @{ name = "highest_takes_last"; from = "highest_ = std::max(highest_, value);"; to = "highest_ = value;" },

    # The first value has to seed both ends; there is no neutral pair to widen from, and
    # leaving them at zero silently makes zero one end of every range.
    @{ name = "first_value_does_not_seed"; from = "        lowest_ = value;`r`n        highest_ = value;`r`n        seen_ = true;"; to = "        seen_ = true;" },

    # Strictly greater is what keeps a zero width out of the division below.
    @{ name = "established_allows_zero_width"; from = "return seen_ && highest_ > lowest_;"; to = "return seen_ && highest_ >= lowest_;" },
    @{ name = "established_ignores_width"; from = "return seen_ && highest_ > lowest_;"; to = "return seen_;" },
    @{ name = "never_established"; from = "return seen_ && highest_ > lowest_;"; to = "return false;" },

    # The offset is from the lowest, and the divisor is the width.
    @{ name = "offset_from_highest"; from = "return (value - lowest_) / (highest_ - lowest_);"; to = "return (value - highest_) / (highest_ - lowest_);" },
    @{ name = "divides_by_highest"; from = "return (value - lowest_) / (highest_ - lowest_);"; to = "return (value - lowest_) / highest_;" },
    @{ name = "no_normalisation"; from = "return (value - lowest_) / (highest_ - lowest_);"; to = "return value;" },

    # Unclamped is the contract, so a value outside what was observed lands outside
    # [0, 1] rather than being quietly pinned to an end.
    @{ name = "clamps_the_result"; from = "return (value - lowest_) / (highest_ - lowest_);"; to = "return std::clamp((value - lowest_) / (highest_ - lowest_), 0.0f, 1.0f);" },

    # The pass-through while unestablished. Returning a constant instead puts a number
    # the caller never supplied into the comparison.
    @{ name = "unestablished_returns_zero"; from = "    if (!isEstablished())`r`n    {`r`n        return value;`r`n    }"; to = "    if (!isEstablished())`r`n    {`r`n        return 0.0f;`r`n    }" }
)

$killed = 0
$equivalent = 0
foreach ($mutant in $mutants)
{
    $dir = Join-Path $work $mutant.name
    New-Item -ItemType Directory -Force $dir | Out-Null
    foreach ($file in $support) { Copy-Item "$src/$file" $dir }

    $target = if ($mutant.ContainsKey("file")) { $mutant.file } else { "value_range.cpp" }
    $text = Get-Content "$src/$target" -Raw -Encoding UTF8
    if (-not $text.Contains($mutant.from))
    {
        # Reported rather than skipped silently: a pattern that no longer matches tests
        # nothing, and a run that silently tests nothing looks exactly like one that
        # found no survivors.
        Write-Host ("  {0,-32} NOT APPLIED - pattern absent, mutation is vacuous" -f $mutant.name)
        continue
    }
    Set-Content "$dir/$target" -Value $text.Replace($mutant.from, $mutant.to) -Encoding UTF8 -NoNewline

    Push-Location $dir
    $null = Invoke-Expression "cl /nologo /std:c++20 /EHsc /O2 $sources /Fe:mutant.exe" 2>&1
    if ($LASTEXITCODE -ne 0)
    {
        Write-Host ("  {0,-32} KILLED by the compiler" -f $mutant.name)
        $killed++
        Pop-Location
        continue
    }
    $null = & ./mutant.exe 2>&1
    $code = $LASTEXITCODE
    Pop-Location

    $expected_equivalent = $mutant.ContainsKey("equivalent")
    if ($code -ne 0)
    {
        if ($expected_equivalent)
        {
            Write-Host ("  {0,-32} KILLED - but marked equivalent; recheck the label" -f $mutant.name)
        }
        else
        {
            Write-Host ("  {0,-32} KILLED" -f $mutant.name)
        }
        $killed++
    }
    elseif ($expected_equivalent)
    {
        Write-Host ("  {0,-32} EQUIVALENT - unkillable by construction, see the note" -f $mutant.name)
        $equivalent++
    }
    else
    {
        Write-Host ("  {0,-32} SURVIVED - the test cannot see this change" -f $mutant.name)
    }
}
Write-Host ("`nkilled {0}, equivalent {1}, of {2}" -f $killed, $equivalent, $mutants.Count)
