# Mutation test for evaluation::parseArguments and evaluation::Settings.
#
# Each mutant is compiled in its own directory together with copies of every header
# and source it needs, because MSVC resolves a quoted include from the including
# file's own directory first - a mutant left elsewhere is silently ignored and
# reports as surviving.
#
# Compiled /MDd rather than /O2: the assertions are the point of several of these,
# and /O2 defines no NDEBUG here but the debug runtime is what makes an abort
# visible as a non-zero exit rather than a dialog.

$src = "D:/repo/snakeNN/src"
$work = "$env:TEMP/eval_mutants"
Remove-Item $work -Recurse -Force -ErrorAction Ignore
New-Item -ItemType Directory -Force $work | Out-Null

$original = Get-Content "$src/eval_options.cpp" -Raw -Encoding UTF8

$carried = @("eval_options.h", "az_parameters.h", "az_parameters.cpp", "flag_parser.h",
             "flag_parser.cpp", "seed_policy.h", "eval_options_test.cpp")

$mutants = @(
  # The bounds. Each pair is a direction: too loose, then too strict.
  @{ name = "board_floor_gone";      from = 'flags::requireAtLeast("--board", settings.board, 2);'; to = "" },
  @{ name = "board_floor_off_by_one"; from = 'requireAtLeast("--board", settings.board, 2)'; to = 'requireAtLeast("--board", settings.board, 3)' },
  @{ name = "board_ceiling_gone";    from = 'requireAtMost("--board", settings.board, az::LARGEST_BOARD);'; to = "" },
  @{ name = "board_ceiling_loose";   from = 'requireAtMost("--board", settings.board, az::LARGEST_BOARD)'; to = 'requireAtMost("--board", settings.board, az::LARGEST_BOARD + 1)' },
  @{ name = "games_floor_gone";      from = 'flags::requireAtLeast("--games", settings.games, 1);'; to = "" },
  @{ name = "simulations_floor_gone"; from = 'flags::requireAtLeast("--simulations", settings.simulations, 1);'; to = "" },
  @{ name = "channels_floor_gone";   from = 'flags::requireAtLeast("--channels", settings.channels, 1);'; to = "" },
  @{ name = "blocks_floor_gone";     from = 'flags::requireAtLeast("--blocks", settings.blocks, 0);'; to = "" },
  @{ name = "blocks_floor_too_high"; from = 'requireAtLeast("--blocks", settings.blocks, 0)'; to = 'requireAtLeast("--blocks", settings.blocks, 1)' },
  @{ name = "batch_floor_gone";      from = 'flags::requireAtLeast("--batch", settings.batch, 1);'; to = "" },
  @{ name = "step_limit_floor_gone"; from = 'flags::requireAtLeast("--step-limit", *settings.step_limit_override, 1);'; to = "" },
  @{ name = "checkpoint_not_required"; from = "if (settings.checkpoint.empty())"; to = "if (false)" },
  @{ name = "at_most_reversed";      from = "if (value > maximum)"; to = "if (maximum > value)" },
  @{ name = "at_most_off_by_one";    from = "if (value > maximum)"; to = "if (value > maximum + 1)" },
  # The seed band. The 32-bit form is the wraparound the check exists to prevent.
  @{ name = "band_check_gone";       from = "if (last_seed_index >= static_cast<long long>(RESERVED_BAND_WIDTH))"; to = "if (false)" },
  @{ name = "band_off_by_one";       from = "last_seed_index >= static_cast<long long>(RESERVED_BAND_WIDTH)"; to = "last_seed_index > static_cast<long long>(RESERVED_BAND_WIDTH)" },
  @{ name = "band_ignores_games";    from = "static_cast<long long>(settings.seed_offset) + settings.games - 1"; to = "static_cast<long long>(settings.seed_offset)" },
  # The parser's routing. A value written into the wrong field is what the
  # one-field-changed check in the test exists to see.
  @{ name = "board_writes_games";    from = "settings.board = flags::parseWholeInt(pair.flag, pair.value);"; to = "settings.games = flags::parseWholeInt(pair.flag, pair.value);" },
  @{ name = "batch_writes_simulations"; from = "settings.batch = flags::parseWholeInt(pair.flag, pair.value);"; to = "settings.simulations = flags::parseWholeInt(pair.flag, pair.value);" },
  @{ name = "unknown_flag_ignored";  from = 'throw std::invalid_argument(std::format("unknown flag: {}", pair.flag));'; to = "" },
  @{ name = "seed_signed";           from = "settings.seed_offset = flags::parseWholeUnsigned(pair.flag, pair.value);"; to = "settings.seed_offset = static_cast<unsigned int>(flags::parseWholeInt(pair.flag, pair.value));" },
  @{ name = "validation_skipped";    from = "requireUsable(settings);"; to = "" },
  # The derived quantities.
  @{ name = "step_limit_ignores_override"; from = "if (step_limit_override)"; to = "if (false)" },
  @{ name = "step_limit_always_override"; from = "return az::deriveStepLimit(board);"; to = "return 1200;" },
  @{ name = "cells_off_by_one";      from = "return board * board;"; to = "return board * board + 1;" },
  @{ name = "foods_off_by_one";      from = "return cellCount() - 1;"; to = "return cellCount();" },
  # The two formatters. A log that drops one of these fields is a log a later
  # comparison cannot be run from, and nothing about the run would look wrong.
  @{ name = "header_drops_batch";    from = ", step limit {}, batch {}"; to = ", step limit {}" },
  @{ name = "header_drops_limit";    from = "{} simulations, step limit {}"; to = "{} simulations" },
  @{ name = "header_first_seed_is_last"; from = "seeds::evaluationGameSeed(settings.seed_offset, 0),"; to = "seeds::evaluationGameSeed(settings.seed_offset, settings.games - 1)," },
  @{ name = "header_no_blank_line";  from = "greedy, no root noise\n\n"; to = "greedy, no root noise\n" },
  @{ name = "line_drops_seed";       from = '"  game seed {}, {}, score {}, steps {}\n", seed, word'; to = '"  game seed x, {}, score {}, steps {}\n", word' },
  @{ name = "line_drops_tag";        from = '"  game seed {}'; to = '"  seed {}' },
  @{ name = "line_no_newline";       from = 'steps {}\n", seed'; to = 'steps {}", seed' },
  @{ name = "won_reads_as_died";     from = 'word = "won";'; to = "" },
  @{ name = "timeout_reads_as_died"; from = 'word = "timeout";'; to = "" },
  @{ name = "timeout_contains_won";  from = 'word = "timeout";'; to = 'word = "won-timeout";' },
  # visual. The two parsers share a body shape, so every mutant here is scoped to
  # the second namespace or it lands in the evaluator's copy.
  @{ name = "v_board_floor_gone";    scope = "namespace visual"; from = 'flags::requireAtLeast("--board", settings.board, 2);'; to = "" },
  @{ name = "v_board_ceiling_gone";  scope = "namespace visual"; from = 'requireAtMost("--board", settings.board, az::LARGEST_BOARD);'; to = "" },
  @{ name = "v_simulations_gone";    scope = "namespace visual"; from = 'flags::requireAtLeast("--simulations", settings.simulations, 1);'; to = "" },
  @{ name = "v_channels_gone";       scope = "namespace visual"; from = 'flags::requireAtLeast("--channels", settings.channels, 1);'; to = "" },
  @{ name = "v_blocks_gone";         scope = "namespace visual"; from = 'flags::requireAtLeast("--blocks", settings.blocks, 0);'; to = "" },
  @{ name = "v_blocks_too_high";     scope = "namespace visual"; from = 'requireAtLeast("--blocks", settings.blocks, 0)'; to = 'requireAtLeast("--blocks", settings.blocks, 1)' },
  @{ name = "v_speed_floor_gone";    scope = "namespace visual"; from = 'flags::requireAtLeast("--speed", settings.moves_per_frame, 1);'; to = "" },
  @{ name = "v_speed_allows_zero";   scope = "namespace visual"; from = 'requireAtLeast("--speed", settings.moves_per_frame, 1)'; to = 'requireAtLeast("--speed", settings.moves_per_frame, 0)' },
  @{ name = "v_step_limit_gone";     scope = "namespace visual"; from = 'flags::requireAtLeast("--step-limit", *settings.step_limit_override, 1);'; to = "" },
  @{ name = "v_checkpoint_optional"; scope = "namespace visual"; from = "if (settings.checkpoint.empty())"; to = "if (false)" },
  @{ name = "v_unknown_flag_ignored"; scope = "namespace visual"; from = 'throw std::invalid_argument(std::format("unknown flag: {}", pair.flag));'; to = "" },
  @{ name = "v_seed_signed";         scope = "namespace visual"; from = "settings.seed = flags::parseWholeUnsigned(pair.flag, pair.value);"; to = "settings.seed = static_cast<unsigned int>(flags::parseWholeInt(pair.flag, pair.value));" },
  @{ name = "v_speed_writes_blocks"; scope = "namespace visual"; from = "settings.moves_per_frame = flags::parseWholeInt(pair.flag, pair.value);"; to = "settings.blocks = flags::parseWholeInt(pair.flag, pair.value);" },
  @{ name = "v_validation_skipped";  scope = "namespace visual"; from = "requireUsable(settings);"; to = "" },
  @{ name = "v_step_ignores_override"; scope = "namespace visual"; from = "if (step_limit_override)"; to = "if (false)" },
  @{ name = "v_cells_off_by_one";    scope = "namespace visual"; from = "return board * board;"; to = "return board * board + 1;" },
  @{ name = "v_foods_off_by_one";    scope = "namespace visual"; from = "return cellCount() - 1;"; to = "return cellCount();" }
)

$killed = 0
$survived = @()
foreach ($mutant in $mutants)
{
    $dir = Join-Path $work $mutant.name
    New-Item -ItemType Directory -Force $dir | Out-Null
    foreach ($file in $carried)
    {
        Copy-Item "$src/$file" $dir
    }

    $head = ""
    $body = $original
    if ($mutant.ContainsKey("scope"))
    {
        $at = $original.IndexOf($mutant.scope)
        if ($at -lt 0)
        {
            Write-Host ("  {0,-28} NOT APPLIED - scope marker absent" -f $mutant.name)
            continue
        }
        $head = $original.Substring(0, $at)
        $body = $original.Substring($at)
    }
    if (-not $body.Contains($mutant.from))
    {
        Write-Host ("  {0,-28} NOT APPLIED - pattern absent, mutation is vacuous" -f $mutant.name)
        continue
    }
    # The first occurrence only. evaluation and visual share most of their body
    # shapes, so replacing every match would mutate both and the mutant's name
    # would describe one of them. Unscoped therefore means the evaluator's copy,
    # which comes first in the file; the visual's are reached by scope marker.
    $at = $body.IndexOf($mutant.from)
    $mutated = $head + $body.Substring(0, $at) + $mutant.to +
               $body.Substring($at + $mutant.from.Length)
    Set-Content "$dir/eval_options.cpp" -Value $mutated -Encoding UTF8 -NoNewline

    Push-Location $dir
    $null = cl /nologo /std:c++20 /EHsc /MDd eval_options.cpp az_parameters.cpp flag_parser.cpp eval_options_test.cpp /Fe:mutant.exe 2>&1 | Out-String
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
    if ($code -ne 0)
    {
        Write-Host ("  {0,-28} KILLED" -f $mutant.name)
        $killed++
    }
    else
    {
        Write-Host ("  {0,-28} SURVIVED - the test cannot see this change" -f $mutant.name)
        $survived += $mutant.name
    }
}
Write-Host ("`nkilled {0} of {1}" -f $killed, $mutants.Count)
if ($survived.Count -gt 0)
{
    Write-Host ("survivors: {0}" -f ($survived -join ", "))
}
