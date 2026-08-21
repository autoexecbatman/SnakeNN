# Mutation test for the exploration mix weight in exploration_floor.cpp. Mutants compile
# beside copies of every header and source the unit needs, because MSVC resolves a quoted
# include from the including file's own directory first and a mutant left anywhere else is
# ignored - which reports as a survivor.

$src = "D:/repo/snakeNN/src"
$work = "$env:TEMP/mutants_exploration_floor"
if (Test-Path $work) { Remove-Item $work -Recurse -Force }
New-Item -ItemType Directory -Force $work | Out-Null

$support = @("exploration_floor.h", "exploration_floor.cpp", "exploration_floor_test.cpp")
$sources = "exploration_floor.cpp exploration_floor_test.cpp"

# The unmutated code, compiled and run exactly as every mutant will be. Without this a
# missing header kills every mutant and a harness that tests nothing is indistinguishable
# from one that kills everything.
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
    # Off must be exactly off. A floor that leaks changes how every existing checkpoint
    # plays the moment it is compiled in, which is the thing the default exists to prevent.
    @{ name = "zero_epsilon_leaks"; from = "    if (epsilon == 0.0f)`r`n    {`r`n        return 0.0f;`r`n    }"; to = "" },

    # EQUIVALENT, with the reasoning, confirmed by a run in which it was the only survivor
    # of thirteen. At zero visits log(0 + 1) is exactly 0, so the division yields positive
    # infinity and the clamp below returns 1.0 - the same value the early return gives. No
    # test can see the difference because there is none in the result. The early return
    # stays because it reaches that answer without dividing by zero and raising
    # FE_DIVBYZERO, and because it says what the case means rather than leaving a reader to
    # work out that an infinity is mopped up three lines later.
    @{ name = "no_unvisited_case"; equivalent = $true; from = "    if (total_visits == 0)`r`n    {`r`n        return 1.0f;`r`n    }"; to = "" },
    @{ name = "unvisited_returns_zero"; from = "        return 1.0f;`r`n    }`r`n`r`n    const float weight"; to = "        return 0.0f;`r`n    }`r`n`r`n    const float weight" },

    # The decay is 1/log(N), slow on purpose. 1/N or 1/sqrt(N) empties the floor long
    # before the policy has finished sharpening.
    @{ name = "decay_is_linear"; from = "std::log(static_cast<float>(total_visits) + 1.0f)"; to = "(static_cast<float>(total_visits) + 1.0f)" },
    @{ name = "decay_is_sqrt"; from = "std::log(static_cast<float>(total_visits) + 1.0f)"; to = "std::sqrt(static_cast<float>(total_visits) + 1.0f)" },

    # The plus one inside the logarithm is what keeps a single visit from dividing by zero.
    @{ name = "no_plus_one_in_log"; from = "std::log(static_cast<float>(total_visits) + 1.0f)"; to = "std::log(static_cast<float>(total_visits))" },

    # Scaling by the action count is what keeps the per-action floor constant as actions
    # are added; without it the floor thins out on wider action spaces.
    @{ name = "drops_action_count"; from = "    const float weight = epsilon * static_cast<float>(action_count) /"; to = "    const float weight = epsilon /" },

    # Epsilon is the dial. Squaring it or dropping it are both silent at the default.
    @{ name = "epsilon_squared"; from = "    const float weight = epsilon * static_cast<float>(action_count) /"; to = "    const float weight = epsilon * epsilon * static_cast<float>(action_count) /" },

    # The clamp. Without it the weight exceeds one at small visit counts and a caller
    # comparing a uniform draw against it explores unconditionally.
    @{ name = "no_clamp"; from = "return std::min(weight, 1.0f);"; to = "return weight;" },
    @{ name = "clamps_to_zero_instead"; from = "return std::min(weight, 1.0f);"; to = "return std::max(weight, 1.0f);" },

    # The guards. Each refuses an argument that would otherwise produce a plausible number.
    @{ name = "accepts_negative_epsilon"; from = "    if (!(epsilon >= 0.0f))"; to = "    if (false)" },
    @{ name = "accepts_zero_actions"; from = "    if (action_count < 1)"; to = "    if (false)" },
    @{ name = "accepts_negative_visits"; from = "    if (total_visits < 0)"; to = "    if (false)" },

    # descentMixWeight, whose whole content is where the floor is allowed to fire. Below
    # the root the weight is read from a node with one visit or none, where the 1/log decay
    # has not started - measured, epsilon 0.1 at every node took self-play from 97.5 apples
    # to 3.8 with no wins.
    @{ name = "floor_fires_at_every_depth"; from = "    if (depth > 0)"; to = "    if (false)" },
    @{ name = "floor_off_only_one_level_down"; from = "    if (depth > 0)"; to = "    if (depth == 1)" },
    @{ name = "floor_off_at_the_root_too"; from = "    if (depth > 0)"; to = "    if (depth >= 0)" },
    @{ name = "accepts_negative_depth"; from = "    if (depth < 0)"; to = "    if (false)" }
)

$killed = 0
$equivalent = 0
foreach ($mutant in $mutants)
{
    $dir = Join-Path $work $mutant.name
    New-Item -ItemType Directory -Force $dir | Out-Null
    foreach ($file in $support) { Copy-Item "$src/$file" $dir }

    $target = if ($mutant.ContainsKey("file")) { $mutant.file } else { "exploration_floor.cpp" }
    $text = Get-Content "$src/$target" -Raw -Encoding UTF8
    if (-not $text.Contains($mutant.from))
    {
        # Reported rather than skipped silently: a pattern that no longer matches tests
        # nothing, and a run that silently tests nothing looks exactly like one that found
        # no survivors.
        Write-Host ("  {0,-28} NOT APPLIED - pattern absent, mutation is vacuous" -f $mutant.name)
        continue
    }
    Set-Content "$dir/$target" -Value $text.Replace($mutant.from, $mutant.to) -Encoding UTF8 -NoNewline

    Push-Location $dir
    $null = Invoke-Expression "cl /nologo /std:c++20 /EHsc /O2 $sources /Fe:mutant.exe" 2>&1
    if ($LASTEXITCODE -ne 0)
    {
        Write-Host ("  {0,-28} KILLED by the compiler" -f $mutant.name)
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
            Write-Host ("  {0,-28} KILLED - but marked equivalent; recheck the label" -f $mutant.name)
        }
        else
        {
            Write-Host ("  {0,-28} KILLED" -f $mutant.name)
        }
        $killed++
    }
    elseif ($expected_equivalent)
    {
        Write-Host ("  {0,-28} EQUIVALENT - unkillable by construction, see the note" -f $mutant.name)
        $equivalent++
    }
    else
    {
        Write-Host ("  {0,-28} SURVIVED - the test cannot see this change" -f $mutant.name)
    }
}
Write-Host ("`nkilled {0}, equivalent {1}, of {2}" -f $killed, $equivalent, $mutants.Count)
