# Mutation test for SnakeEnv::tailReachable and the flood fill it shares with
# reachableCells.
#
# Each mutant is compiled in its own directory together with copies of the header
# and the test, because MSVC resolves a quoted include from the including file's
# own directory first - a mutant left elsewhere is silently ignored and reports as
# surviving.
#
# The test needs hamiltonian_cycle.cpp and snake_logic.cpp as well, since the
# endgame property is driven by a cycle-following snake.

$src = "D:/repo/snakeNN/src"
$work = "$env:TEMP/mutants_tail"
Remove-Item $work -Recurse -Force -ErrorAction Ignore
New-Item -ItemType Directory -Force $work | Out-Null

$original = Get-Content "$src/snake_env.cpp" -Raw -Encoding UTF8

$mutants = @(
  # The two queries must not collapse into each other. A region's size and whether
  # it holds the tail are different questions, and answering one with the other is
  # exactly the defect that lost 64 games out of 64.
  @{ name = "tail_from_size";       from = "return floodAfter(action).holds_tail;"; to = "return floodAfter(action).cells >= static_cast<int>(body_.size());" },
  @{ name = "tail_always_true";     from = "return floodAfter(action).holds_tail;"; to = "return !wouldDie(action);" },
  @{ name = "tail_always_false";    from = "return floodAfter(action).holds_tail;"; to = "return false;" },
  @{ name = "cells_from_tail";      from = "return floodAfter(action).cells;";      to = "return floodAfter(action).holds_tail ? 1 : 0;" },
  # A fatal move must report nothing reachable and no tail. Dropping the guard
  # makes a wall move look like whatever the fill happens to find.
  @{ name = "drop_death_guard";     from = "if (wouldDie(action))";                 to = "if (false)" },
  @{ name = "death_returns_room";   from = "return Region{};";                      to = "return Region{ 1, true };" },
  # The tail vacates as the head arrives. Treating it as an obstacle makes a snake
  # unable to follow itself, which is the whole endgame.
  @{ name = "tail_blocks";          from = "occupancy_[static_cast<size_t>(index)] != 0 && !(next == vacated)"; to = "occupancy_[static_cast<size_t>(index)] != 0" },
  @{ name = "nothing_blocks";       from = "occupancy_[static_cast<size_t>(index)] != 0 && !(next == vacated)"; to = "false" },
  # The landing cell counts, and it is where a length-one snake stands on its own
  # tail - so reading holds_tail off the walk rather than off `seen` loses that.
  @{ name = "tail_excludes_landing"; from = "seen[static_cast<size_t>(cellIndex(vacated))] != 0"; to = "cellIndex(vacated) != cellIndex(landing) && seen[static_cast<size_t>(cellIndex(vacated))] != 0" },
  @{ name = "count_skips_landing";  from = "int reached = 0;";                      to = "int reached = -1;" },
  # Which cell is the tail. body_.front() is the head, so this reads the fill
  # against the wrong end of the snake.
  @{ name = "vacated_is_head";      from = "const Position vacated = body_.back();"; to = "const Position vacated = body_.front();" }
)

$killed = 0
foreach ($mutant in $mutants)
{
    $dir = Join-Path $work $mutant.name
    New-Item -ItemType Directory -Force $dir | Out-Null
    Copy-Item "$src/snake_env.h" $dir
    Copy-Item "$src/snake_env_test.cpp" $dir
    Copy-Item "$src/hamiltonian_cycle.h" $dir
    Copy-Item "$src/hamiltonian_cycle.cpp" $dir
    Copy-Item "$src/snake_logic.h" $dir
    Copy-Item "$src/snake_logic.cpp" $dir

    if (-not $original.Contains($mutant.from))
    {
        Write-Host ("  {0,-24} NOT APPLIED - pattern absent, mutation is vacuous" -f $mutant.name)
        continue
    }
    $mutated = $original.Replace($mutant.from, $mutant.to)
    Set-Content "$dir/snake_env.cpp" -Value $mutated -Encoding UTF8 -NoNewline

    Push-Location $dir
    $null = cl /nologo /std:c++20 /EHsc /O2 snake_env.cpp snake_env_test.cpp hamiltonian_cycle.cpp snake_logic.cpp /Fe:mutant.exe 2>&1 | Out-String
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
