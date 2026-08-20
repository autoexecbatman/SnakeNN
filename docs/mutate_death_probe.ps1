# Mutation test for the death-probe statistics in death_probe.cpp. Mutants compile
# beside copies of every header and source the unit needs, because MSVC resolves a
# quoted include from the including file's own directory first and a mutant left
# anywhere else is ignored - which reports as a survivor.

$src = "D:/repo/snakeNN/src"
$work = "$env:TEMP/mutants_death_probe"
if (Test-Path $work) { Remove-Item $work -Recurse -Force }
New-Item -ItemType Directory -Force $work | Out-Null

$support = @("death_probe.h", "death_probe.cpp", "death_probe_test.cpp")
$sources = "death_probe.cpp death_probe_test.cpp"

# The unmutated code, compiled and run exactly as every mutant will be. Without this
# a missing header kills every mutant and a harness that tests nothing is
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
    # The sample standard deviation. n rather than n-1 is the classic slip, and it is
    # small on large samples - which is exactly why a test has to pin it.
    @{ name = "spread_uses_population_n"; from = "std::sqrt(squared_deviation / (count - 1.0))"; to = "std::sqrt(squared_deviation / count)" },

    # Spread is the statistic the report says to read first: a head at its
    # initialisation is caught here or nowhere.
    @{ name = "spread_always_zero"; from = "report.estimate_spread = std::sqrt(squared_deviation / (count - 1.0));"; to = "report.estimate_spread = 0.0;" },

    # Midranks are what make Spearman correct under ties. Taking the block's first
    # position instead of its midpoint is the ordinary-rank bug.
    @{ name = "ranks_ignore_ties"; from = "const double shared = (static_cast<double>(start + 1) + static_cast<double>(stop)) / 2.0;"; to = "const double shared = static_cast<double>(start + 1);" },

    # A tie in the head's output is half a point, not a whole one. Crediting a full
    # point makes a constant head score a perfect 1.0.
    @{ name = "auc_tie_counts_full"; from = "concordant += 0.5;"; to = "concordant += 1.0;" },
    @{ name = "auc_tie_counts_nothing"; from = "concordant += 0.5;"; to = "concordant += 0.0;" },

    # The direction of the comparison is the whole of the AUC.
    @{ name = "auc_direction_flipped"; from = "if (doomed > safe)"; to = "if (doomed < safe)" },

    # The threshold binarises the target. Strictly-greater silently moves every
    # boundary case into the safe class.
    @{ name = "threshold_is_strict"; from = "if (pair.target >= doomed_threshold)"; to = "if (pair.target > doomed_threshold)" },

    # An AUC over a one-sided target is not 0.5, it is nothing. Always-true is the
    # bug that reports a coin flip where no pair was ever ordered.
    @{ name = "auc_always_defined"; from = "report.ranking_auc_defined = !doomed_estimates.empty() && !safe_estimates.empty();"; to = "report.ranking_auc_defined = true;" },

    # sample_count is what tells a reader the report rests on something. Folding the
    # rejections in inflates it with positions that were never scored.
    @{ name = "count_includes_rejections"; from = "report.sample_count = samples.pairs.size();"; to = "report.sample_count = samples.pairs.size() + samples.rejected_uncovered;" },

    # The floor exists because a sample deviation needs n-1 and one pair admits no
    # ranking. Lowering it produces a divide by zero rather than a refusal.
    @{ name = "accepts_one_pair"; from = "if (samples.pairs.size() < 2)"; to = "if (samples.pairs.size() < 1)" },

    # The threshold guard. Written with negated comparisons so a NaN is refused too;
    # dropping it admits a threshold that puts every target on one side.
    @{ name = "threshold_unchecked"; from = "if (!(doomed_threshold >= 0.0f) || !(doomed_threshold <= 1.0f))"; to = "if (false)" },

    # The AUC normaliser is the count of ordered pairs, not the sample count.
    @{ name = "auc_wrong_normaliser"; from = "report.ranking_auc = concordant / (static_cast<double>(doomed_estimates.size()) *`r`n                                       static_cast<double>(safe_estimates.size()));"; to = "report.ranking_auc = concordant / count;" }
)

$killed = 0
$equivalent = 0
foreach ($mutant in $mutants)
{
    $dir = Join-Path $work $mutant.name
    New-Item -ItemType Directory -Force $dir | Out-Null
    foreach ($file in $support) { Copy-Item "$src/$file" $dir }

    $target = if ($mutant.ContainsKey("file")) { $mutant.file } else { "death_probe.cpp" }
    $text = Get-Content "$src/$target" -Raw -Encoding UTF8
    if (-not $text.Contains($mutant.from))
    {
        # Reported rather than skipped silently: a pattern that no longer matches
        # tests nothing, and a run that silently tests nothing looks exactly like one
        # that found no survivors.
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
