# Mutation test for az::deriveStepLimit. Mutants compile beside a copy of the
# header and the test, because MSVC resolves a quoted include from the including
# file's own directory first.

$src = "D:/repo/snakeNN/src"
$work = "$env:TEMP/mutants_params"
if (Test-Path $work) { Remove-Item $work -Recurse -Force }
New-Item -ItemType Directory -Force $work | Out-Null

$original = Get-Content "$src/az_parameters.cpp" -Raw -Encoding UTF8

$mutants = @(
  @{ name = "wrong_steps_per_cell"; file = "az_parameters.h"; from = "STEPS_PER_CELL = 12"; to = "STEPS_PER_CELL = 11" },
  @{ name = "wrong_largest_board";  file = "az_parameters.h"; from = "LARGEST_BOARD = 13377"; to = "LARGEST_BOARD = 13376" },
  @{ name = "linear_not_area";      from = "STEPS_PER_CELL) * board * board"; to = "STEPS_PER_CELL) * board" },
  @{ name = "drop_overflow_guard";  from = "if (board > LARGEST_BOARD)"; to = "if (false)" },
  # These two were labelled equivalent while the guard compared areas: largest_area
  # was 178956970, no board area lands on it, so no integer observed the difference
  # between >, >= and > +1. The guard now compares boards, where consecutive values
  # differ by one, and both are killable. An equivalence argument is about the code
  # it was written against and does not survive the code changing.
  @{ name = "guard_off_by_one";     from = "if (board > LARGEST_BOARD)"; to = "if (board > LARGEST_BOARD + 1)" },
  @{ name = "guard_too_strict";     from = "if (board > LARGEST_BOARD)"; to = "if (board >= LARGEST_BOARD)" },
  @{ name = "message_drops_board";  from = '"board {} is too large: its step limit does not fit in an int", board'; to = '"a board is too large: its step limit does not fit in an int"' }
)

$killed = 0
$equivalent = 0
foreach ($mutant in $mutants)
{
    $dir = Join-Path $work $mutant.name
    New-Item -ItemType Directory -Force $dir | Out-Null
    Copy-Item "$src/az_parameters.h" $dir
    Copy-Item "$src/az_parameters.cpp" $dir
    Copy-Item "$src/az_parameters_test.cpp" $dir

    # A mutant may target the header instead of the source.
    $target = if ($mutant.ContainsKey("file")) { $mutant.file } else { "az_parameters.cpp" }
    $text = Get-Content "$src/$target" -Raw -Encoding UTF8
    if (-not $text.Contains($mutant.from))
    {
        Write-Host ("  {0,-22} NOT APPLIED - pattern absent, mutation is vacuous" -f $mutant.name)
        continue
    }
    Set-Content "$dir/$target" -Value $text.Replace($mutant.from, $mutant.to) -Encoding UTF8 -NoNewline

    Push-Location $dir
    $null = cl /nologo /std:c++20 /EHsc /O2 az_parameters.cpp az_parameters_test.cpp /Fe:mutant.exe 2>&1
    if ($LASTEXITCODE -ne 0)
    {
        Write-Host ("  {0,-22} KILLED by the compiler" -f $mutant.name)
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
            # A mutant recorded as unkillable that dies means the reasoning behind
            # that label is wrong, or the code changed under it.
            Write-Host ("  {0,-22} KILLED - but marked equivalent; recheck the label" -f $mutant.name)
        }
        else
        {
            Write-Host ("  {0,-22} KILLED" -f $mutant.name)
        }
        $killed++
    }
    elseif ($expected_equivalent)
    {
        Write-Host ("  {0,-22} EQUIVALENT - unkillable by construction, see the note" -f $mutant.name)
        $equivalent++
    }
    else
    {
        Write-Host ("  {0,-22} SURVIVED - the test cannot see this change" -f $mutant.name)
    }
}
Write-Host ("`nkilled {0}, equivalent {1}, of {2}" -f $killed, $equivalent, $mutants.Count)
