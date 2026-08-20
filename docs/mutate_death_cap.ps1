# Mutation test for the death-risk backup and the root cap in mcts.cpp. Mutants
# compile beside copies of every header and source the search needs, because MSVC
# resolves a quoted include from the including file's own directory first and a
# mutant left anywhere else is ignored - which reports as a survivor.

$src = "D:/repo/snakeNN/src"
$work = "$env:TEMP/mutants_death_cap"
if (Test-Path $work) { Remove-Item $work -Recurse -Force }
New-Item -ItemType Directory -Force $work | Out-Null

# Everything SearchTest compiles, plus the headers those three pull in by quoted
# include. Copied wholesale rather than listed minimally: a missing header fails
# the compile, which this script would otherwise report as a kill.
$support = @(
    "mcts.h", "mcts.cpp", "mcts_test.cpp", "snake_env.h", "snake_env.cpp",
    "evaluator.h", "az_parameters.h", "az_parameters.cpp", "seed_policy.h",
    "snake_logic.h", "snake_logic.cpp"
)
$sources = "mcts.cpp mcts_test.cpp snake_env.cpp az_parameters.cpp snake_logic.cpp"

# The unmutated code, compiled and run exactly as every mutant will be. Without
# this the first run of this script reported "killed 10, of 10" while the baseline
# did not compile at all - snake_logic.h was missing from the list above, so every
# mutant died of a missing header and a harness that tested nothing was
# indistinguishable from one that killed everything.
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
    # The minimum is what makes this a dead-end estimate rather than a
    # death-frequency one. Maximum inverts it; averaging turns it into the
    # quantity the steps head already taught us not to build.
    @{ name = "risk_takes_maximum"; from = "lowest = std::min(lowest, tree.nodes[first_child + action].death_risk);"; to = "lowest = std::max(lowest, tree.nodes[first_child + action].death_risk);" },

    # A terminal that is not won is a death. Swapping the arms makes every death
    # look safe and every win doomed.
    @{ name = "terminal_risk_inverted"; from = "state.won() ? 0.0f : 1.0f;"; to = "state.won() ? 1.0f : 0.0f;" },

    # The whole point of the cap: refuse only what is over the threshold.
    @{ name = "cap_refuses_below"; from = "if (result.death_risk[action] > config_.death_cap_threshold)"; to = "if (result.death_risk[action] < config_.death_cap_threshold)" },
    @{ name = "cap_off_by_one"; from = "if (result.death_risk[action] > config_.death_cap_threshold)"; to = "if (result.death_risk[action] >= config_.death_cap_threshold)" },

    # The condition that separates this from the trap guard. Forcing it true
    # empties the choice in a lost position; forcing it false disables the cap.
    @{ name = "drop_survivor_guard"; from = "if (anything_survives)"; to = "if (true)" },
    @{ name = "cap_never_fires"; from = "if (anything_survives)"; to = "if (false)" },

    # The flag must be what decides, not a constant.
    @{ name = "cap_ignores_its_flag"; from = "if (config_.death_cap)"; to = "if (true)" },

    # A cap that only stops the argmax but leaves the policy weight is the
    # half-measure the assertion on the policy exists to catch.
    @{ name = "refusal_leaves_visits"; from = "visits[action] = 0;`r`n                            death_cap_fires_++;"; to = "death_cap_fires_++;" },

    # The flag that decides which risk labels reach training. Always-true is the
    # bug that was in this code before the test existed: a root with no visits
    # falls back to a uniform policy, so anything inferring coverage from the
    # policy calls an unsearched root fully searched.
    @{ name = "coverage_always_true"; from = "if (visits[action] == 0)"; to = "if (false)" },

    # Backup must refresh the risk on every node of the path, not only the leaf.
    @{ name = "no_refresh_on_backup"; from = "refreshDeathRisk(tree, tree.path[position]);"; to = "" },

    # EQUIVALENT, with the reasoning, confirmed by a run in which it was the only
    # survivor of ten. A terminal node never acquires children - expand asserts it
    # is not called on one, and the descent stops there - so first_child is always
    # empty on a terminal node and the second clause already returns early. The
    # `node.terminal ||` reads as the load-bearing half and cannot change an
    # outcome. It stays because it says what the function refuses to touch, and
    # because a later change that expanded terminal nodes would make it real.
    @{ name = "refresh_overwrites_terminal"; equivalent = $true; from = "if (node.terminal || !node.first_child.has_value())"; to = "if (!node.first_child.has_value())" }
)

$killed = 0
$equivalent = 0
foreach ($mutant in $mutants)
{
    $dir = Join-Path $work $mutant.name
    New-Item -ItemType Directory -Force $dir | Out-Null
    foreach ($file in $support) { Copy-Item "$src/$file" $dir }

    $target = if ($mutant.ContainsKey("file")) { $mutant.file } else { "mcts.cpp" }
    $text = Get-Content "$src/$target" -Raw -Encoding UTF8
    if (-not $text.Contains($mutant.from))
    {
        # Reported rather than skipped silently: a pattern that no longer matches
        # tests nothing, and a run that silently tests nothing looks exactly like
        # one that found no survivors.
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
