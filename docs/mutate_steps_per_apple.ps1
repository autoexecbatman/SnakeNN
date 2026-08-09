# Mutation test for pace::AppleIntervals and pace::formatPaceLine.
#
# Each mutant compiles in its own directory beside copies of the header and the test,
# because MSVC resolves a quoted include from the including file's own directory
# first - a mutant left elsewhere is silently ignored and reports as surviving.

$src = "D:/repo/snakeNN/src"
$work = "$env:TEMP/pace_mutants"
Remove-Item $work -Recurse -Force -ErrorAction Ignore
New-Item -ItemType Directory -Force $work | Out-Null

$mutants = @(
  # The interval itself. Each of these produces plausible numbers that do not add
  # up, which is exactly what the step-accounting property exists to catch.
  @{ name = "interval_from_zero";     from = "intervals_.push_back(steps - steps_at_last_apple_);"; to = "intervals_.push_back(steps);" },
  @{ name = "interval_off_by_one";    from = "intervals_.push_back(steps - steps_at_last_apple_);"; to = "intervals_.push_back(steps - steps_at_last_apple_ - 1);" },
  @{ name = "interval_reversed";      from = "steps - steps_at_last_apple_"; to = "steps_at_last_apple_ - steps" },
  @{ name = "no_interval_recorded";   from = "intervals_.push_back(steps - steps_at_last_apple_);"; to = "" },
  @{ name = "anchor_not_moved";       from = "steps_at_last_apple_ = steps;"; to = "" },
  @{ name = "anchor_uses_last_steps"; from = "steps_at_last_apple_ = steps;"; to = "steps_at_last_apple_ = last_steps_;" },
  # When an apple counts as eaten.
  @{ name = "apple_on_every_move";    from = "if (score > last_score_)"; to = "if (true)" },
  @{ name = "apple_never";            from = "if (score > last_score_)"; to = "if (false)" },
  @{ name = "apple_off_by_one";       from = "if (score > last_score_)"; to = "if (score > last_score_ + 1)" },
  @{ name = "score_not_carried";      from = "last_score_ = score;"; to = "" },
  @{ name = "steps_not_carried";      from = "last_steps_ = steps;"; to = "" },
  # The tail, which is what makes the accounting close.
  @{ name = "tail_always_zero";       from = "return last_steps_ - steps_at_last_apple_;"; to = "return 0;" },
  @{ name = "tail_is_total";          from = "return last_steps_ - steps_at_last_apple_;"; to = "return last_steps_;" },
  # The member initialisers, which live in the header - there is no constructor to
  # put them in, so a mutant has to target the header instead of the source.
  @{ name = "anchor_starts_at_one";   file = "steps_per_apple.h"; from = "int steps_at_last_apple_{ 0 };"; to = "int steps_at_last_apple_{ 1 };" },
  @{ name = "score_starts_at_one";    file = "steps_per_apple.h"; from = "int last_score_{ 0 };"; to = "int last_score_{ 1 };" },
  # The line.
  @{ name = "line_drops_tag";         from = 'std::format("  pace {}", seed)'; to = 'std::format("  {}", seed)' },
  @{ name = "line_drops_seed";        from = 'std::format("  pace {}", seed)'; to = 'std::string("  pace")' },
  @{ name = "line_no_newline";        from = 'return line + "\n";'; to = "return line;" },
  @{ name = "line_drops_intervals";   from = 'line += std::format(" {}", interval);'; to = "" },
  @{ name = "line_runs_together";     from = 'line += std::format(" {}", interval);'; to = 'line += std::format("{}", interval);' }
)

$killed = 0
$survived = @()
foreach ($mutant in $mutants)
{
    $dir = Join-Path $work $mutant.name
    New-Item -ItemType Directory -Force $dir | Out-Null
    Copy-Item "$src/steps_per_apple.h" $dir
    Copy-Item "$src/steps_per_apple_test.cpp" $dir
    Copy-Item "$src/steps_per_apple.cpp" $dir

    # A mutant may target the header instead of the source: the member initialisers
    # live there, because the class needs no constructor to set three ints to zero.
    $target = if ($mutant.ContainsKey("file")) { $mutant.file } else { "steps_per_apple.cpp" }
    $text = Get-Content "$src/$target" -Raw -Encoding UTF8
    if (-not $text.Contains($mutant.from))
    {
        Write-Host ("  {0,-24} NOT APPLIED - pattern absent, mutation is vacuous" -f $mutant.name)
        continue
    }
    $at = $text.IndexOf($mutant.from)
    $mutated = $text.Substring(0, $at) + $mutant.to + $text.Substring($at + $mutant.from.Length)
    Set-Content "$dir/$target" -Value $mutated -Encoding UTF8 -NoNewline
    if ($target -ne "steps_per_apple.cpp")
    {
        Copy-Item "$src/steps_per_apple.cpp" $dir
    }

    Push-Location $dir
    $null = cl /nologo /std:c++20 /EHsc /MDd steps_per_apple.cpp steps_per_apple_test.cpp `
               /Fe:mutant.exe 2>&1 | Out-String
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
    if ($code -ne 0)
    {
        Write-Host ("  {0,-24} KILLED" -f $mutant.name)
        $killed++
    }
    else
    {
        Write-Host ("  {0,-24} SURVIVED - the test cannot see this change" -f $mutant.name)
        $survived += $mutant.name
    }
}
Write-Host ("`nkilled {0} of {1}" -f $killed, $mutants.Count)
if ($survived.Count -gt 0)
{
    Write-Host ("survivors: {0}" -f ($survived -join ", "))
}
