# Mutation test for the run ledger.
#
# Each mutant compiles in its own directory beside copies of the header and the test,
# because MSVC resolves a quoted include from the including file's own directory
# first - a mutant left elsewhere is silently ignored and reports as surviving.
#
# The test carries seven negative assertions - a tab must not survive, a row must not
# be two lines, the header must not be written twice - and none of them can fire
# against a stub returning the empty string. These mutants are what prove they are
# tests rather than documentation.

$src = "D:/repo/snakeNN/src"
$work = "$env:TEMP/ledger_mutants"
Remove-Item $work -Recurse -Force -ErrorAction Ignore
New-Item -ItemType Directory -Force $work | Out-Null

$original = Get-Content "$src/run_ledger.cpp" -Raw -Encoding UTF8

$mutants = @(
  # The separators. Each of these is exactly the corruption the negative assertions
  # exist to catch, and each would produce a ledger that parses into the wrong shape.
  @{ name = "tab_survives";          from = "character == '\t' || character == '\n'"; to = "character == '\n'" },
  @{ name = "newline_survives";      from = "character == '\t' || character == '\n'"; to = "character == '\t'" },
  @{ name = "no_cleaning_at_all";    from = "if (character == '\t' || character == '\n' || character == '\r')"; to = "if (false)" },
  @{ name = "command_not_cleaned";   from = "command += withoutSeparators(argument);"; to = "command += argument;" },
  # The row and its columns.
  @{ name = "row_no_newline";        from = '{}\t{}\t{}\t{}\t{}\t{:.2f}\t{}\t{}\n'; to = '{}\t{}\t{}\t{}\t{}\t{:.2f}\t{}\t{}' },
  @{ name = "row_drops_samples";     from = '\t{:.2f}\t{}\t{}\n", withoutSeparators(entry.run_id)'; to = '\t{:.2f}\t{}\n", withoutSeparators(entry.run_id)' },
  @{ name = "row_swaps_games";       from = "entry.games, entry.samples);"; to = "entry.samples, entry.games);" },
  @{ name = "header_drops_a_column"; from = "run_id\tstarted_utc\tkind\tcommand\toutcome\tseconds\tgames\tsamples\n"; to = "run_id\tstarted_utc\tkind\tcommand\toutcome\tseconds\tgames\n" },
  # The words. A reader keys on these, so two that share a prefix are one bug.
  @{ name = "started_reads_finished"; from = 'return "started";'; to = 'return "finished-started";' },
  @{ name = "outcomes_collapse";     from = 'return "finished";'; to = 'return "failed";' },
  @{ name = "kinds_collapse";        from = 'kind == Kind::Training ? "training" : "evaluation"'; to = '"training"' },
  # The run id and the clock.
  @{ name = "run_id_process_first";  from = 'std::format("{}-{}", started_utc, process_id)'; to = 'std::format("{}-{}", process_id, started_utc)' },
  @{ name = "run_id_drops_process";  from = 'std::format("{}-{}", started_utc, process_id)'; to = 'std::string(started_utc)' },
  @{ name = "clock_not_utc";         from = "std::chrono::system_clock::now()"; to = "std::chrono::system_clock::now() + std::chrono::hours(24 * 365 * 60)" },
  @{ name = "clock_loses_the_zulu";  from = '{:%Y-%m-%dT%H:%M:%SZ}'; to = '{:%Y-%m-%dT%H:%M:%S}' },
  # Appending. The header-written-once check and the create-then-append behaviour.
  @{ name = "header_every_row";      from = "if (fresh)"; to = "if (true)" },
  @{ name = "header_never";          from = "if (fresh)"; to = "if (false)" },
  @{ name = "truncates_the_ledger";  from = "std::ios::app"; to = "std::ios::trunc" },
  # The first of the two identical guards is the open check; the script replaces the
  # first occurrence only, so this reaches it and not the write check below it.
  @{ name = "open_failure_ignored";  from = "if (!ledger_file)"; to = "if (false)" }
)

$killed = 0
$survived = @()
foreach ($mutant in $mutants)
{
    $dir = Join-Path $work $mutant.name
    New-Item -ItemType Directory -Force $dir | Out-Null
    Copy-Item "$src/run_ledger.h" $dir
    Copy-Item "$src/run_ledger_test.cpp" $dir

    if (-not $original.Contains($mutant.from))
    {
        Write-Host ("  {0,-24} NOT APPLIED - pattern absent, mutation is vacuous" -f $mutant.name)
        continue
    }
    $at = $original.IndexOf($mutant.from)
    $mutated = $original.Substring(0, $at) + $mutant.to +
               $original.Substring($at + $mutant.from.Length)
    Set-Content "$dir/run_ledger.cpp" -Value $mutated -Encoding UTF8 -NoNewline

    Push-Location $dir
    $null = cl /nologo /std:c++20 /EHsc /MDd run_ledger.cpp run_ledger_test.cpp /Fe:mutant.exe 2>&1 |
            Out-String
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
