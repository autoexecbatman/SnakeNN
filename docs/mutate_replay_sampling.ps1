# Mutation test for sampling::pickBiased.
#
# Each mutant is compiled in its own directory together with a copy of the header
# and the test, because MSVC resolves a quoted include from the including file's
# own directory first - a mutant left elsewhere is silently ignored and reports as
# surviving.
#
# Run it:
#
#     powershell -NoProfile -ExecutionPolicy Bypass -File docs/mutate_replay_sampling.ps1
#
# Needs cl on PATH. Prints one line per mutant and a count; a survivor is a claim
# the test does not make.

$src = "D:/repo/snakeNN/src"
$work = "$env:TEMP/mutants_replay_sampling"
Remove-Item $work -Recurse -Force -ErrorAction Ignore
New-Item -ItemType Directory -Force $work | Out-Null

$original = Get-Content "$src/replay_sampling.cpp" -Raw -Encoding UTF8

$mutants = @(
  # The unbiased path must cost exactly one draw and return it. Both mutants make it
  # hunt for a decisive record anyway, which is the common path made expensive.
  @{ name = "always_prefers";        from = "if (!prefer_decisive)"; to = "if (false)" },
  @{ name = "never_prefers";         from = "if (!prefer_decisive)"; to = "if (true)" },
  # The loop bound. One too few loses the last try, one too many draws past the
  # ceiling the contract promises.
  @{ name = "loop_one_short";        from = "for (int attempt = 1; attempt < tries; attempt++)"; to = "for (int attempt = 2; attempt < tries; attempt++)" },
  @{ name = "loop_one_long";         from = "for (int attempt = 1; attempt < tries; attempt++)"; to = "for (int attempt = 1; attempt <= tries; attempt++)" },
  # The test itself. Inverted, it keeps the first record that is not decisive, which
  # is the opposite of the whole unit.
  @{ name = "decisive_inverted";     from = "if (is_decisive(candidate))"; to = "if (!is_decisive(candidate))" },
  @{ name = "decisive_ignored";      from = "if (is_decisive(candidate))"; to = "if (false)" },
  @{ name = "decisive_always";       from = "if (is_decisive(candidate))"; to = "if (true)" },
  # Returning the wrong candidate: the one after the decisive draw rather than it.
  @{ name = "returns_next_draw";     from = "            return candidate;"; to = "            return draw();" },
  # Dropping the fallback would leave a biased pick with no answer when the window
  # holds nothing decisive. The compiler catches this one, which still counts.
  @{ name = "no_fallback_return";    from = "    return candidate;`r`n}"; to = "}" }
)

$killed = 0
foreach ($mutant in $mutants)
{
    $dir = Join-Path $work $mutant.name
    New-Item -ItemType Directory -Force $dir | Out-Null
    Copy-Item "$src/replay_sampling.h" $dir
    Copy-Item "$src/replay_sampling_test.cpp" $dir

    if (-not $original.Contains($mutant.from))
    {
        Write-Host ("  {0,-24} NOT APPLIED - pattern absent, mutation is vacuous" -f $mutant.name)
        continue
    }
    $mutated = $original.Replace($mutant.from, $mutant.to)
    Set-Content "$dir/replay_sampling.cpp" -Value $mutated -Encoding UTF8 -NoNewline

    Push-Location $dir
    # /W4 /WX so a mutant that drops a return is a compile kill rather than undefined
    # behaviour the test passes by luck - which is how no_fallback_return first survived.
    $null = cl /nologo /std:c++20 /EHsc /O2 /W4 /WX replay_sampling.cpp replay_sampling_test.cpp /Fe:mutant.exe 2>&1 | Out-String
    if ($LASTEXITCODE -ne 0)
    {
        Write-Host ("  {0,-24} KILLED by the compiler" -f $mutant.name)
        $killed++
        Pop-Location
        continue
    }
    $null = & ./mutant.exe 2>&1
    $code = $LASTEXITCODE
    Pop-Location
    if ($code -ne 0) { Write-Host ("  {0,-24} KILLED" -f $mutant.name); $killed++ }
    else { Write-Host ("  {0,-24} SURVIVED - the test cannot see this change" -f $mutant.name) }
}
Write-Host ("`nkilled {0} of {1}" -f $killed, $mutants.Count)
