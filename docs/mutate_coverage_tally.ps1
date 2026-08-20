# Mutation test for the coverage and label-yield arithmetic in coverage_tally.cpp.
# Mutants compile beside copies of every header and source the unit needs, because MSVC
# resolves a quoted include from the including file's own directory first and a mutant
# left anywhere else is ignored - which reports as a survivor.

$src = "D:/repo/snakeNN/src"
$work = "$env:TEMP/mutants_coverage_tally"
if (Test-Path $work) { Remove-Item $work -Recurse -Force }
New-Item -ItemType Directory -Force $work | Out-Null

$support = @("coverage_tally.h", "coverage_tally.cpp", "coverage_tally_test.cpp")
$sources = "coverage_tally.cpp coverage_tally_test.cpp"

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
    # Coverage is every available action visited. Counting a position as covered when
    # anything was visited is the bug that makes an unsearched root look searched.
    @{ name = "covered_on_any_visit"; from = "if (visited == action_count)"; to = "if (visited > 0)" },
    @{ name = "never_covered"; from = "if (visited == action_count)"; to = "if (visited > action_count)" },

    # The visited count is a sum, not a count of positions.
    @{ name = "visited_counts_positions"; from = "visited_actions += static_cast<std::size_t>(visited);"; to = "visited_actions += 1;" },

    # available_actions is what makes coverage relative to the actions on offer rather
    # than to a hardcoded three.
    @{ name = "available_counts_visited"; from = "available_actions += static_cast<std::size_t>(action_count);"; to = "available_actions += static_cast<std::size_t>(visited);" },

    # position_coverage is fully covered positions over positions, not actions over
    # positions - the two agree only when every position offers one action.
    @{ name = "coverage_from_actions"; from = "report.position_coverage = static_cast<double>(tally.fully_covered) / positions;"; to = "report.position_coverage = static_cast<double>(tally.visited_actions) / positions;" },

    # The current rule takes the actions of covered positions only.
    @{ name = "labels_from_all_positions"; from = "static_cast<std::size_t>(static_cast<double>(tally.fully_covered) * mean_actions);"; to = "static_cast<std::size_t>(static_cast<double>(tally.positions) * mean_actions);" },

    # A per-action rule keeps what was visited, not what was on offer. Confusing the two
    # is what makes the redesign look free.
    @{ name = "per_action_counts_available"; from = "report.labels_per_action = tally.visited_actions;"; to = "report.labels_per_action = tally.available_actions;" },

    # The mean is per position. Dividing by available actions silently rescales it.
    @{ name = "mean_over_available"; from = "report.mean_visited_actions = static_cast<double>(tally.visited_actions) / positions;"; to = "report.mean_visited_actions = static_cast<double>(tally.visited_actions) / static_cast<double>(tally.available_actions);" },

    # With nothing admitted the gain is unbounded, not absent. Always-true reports a
    # ratio nobody can read as anything but "no gain".
    @{ name = "ratio_always_defined"; from = "report.yield_ratio_defined = report.labels_all_or_nothing > 0;"; to = "report.yield_ratio_defined = true;" },

    # The refusal. Without it an empty tally divides by zero and reports NaN as a rate.
    @{ name = "accepts_empty_tally"; from = "if (tally.positions == 0)"; to = "if (false)" }
)

$killed = 0
$equivalent = 0
foreach ($mutant in $mutants)
{
    $dir = Join-Path $work $mutant.name
    New-Item -ItemType Directory -Force $dir | Out-Null
    foreach ($file in $support) { Copy-Item "$src/$file" $dir }

    $target = if ($mutant.ContainsKey("file")) { $mutant.file } else { "coverage_tally.cpp" }
    $text = Get-Content "$src/$target" -Raw -Encoding UTF8
    if (-not $text.Contains($mutant.from))
    {
        # Reported rather than skipped silently: a pattern that no longer matches tests
        # nothing, and a run that silently tests nothing looks exactly like one that
        # found no survivors.
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
