# The measurement for docs/az_options_spec.md.
#
# Run it before the refactor and after. Each property prints what it reads, so a
# property that cannot fail is visible as one that reads the same thing either way.
# Property 4 needs a GPU run and is gated behind -WithAcceptance.

param([switch]$WithAcceptance)

$root = Split-Path -Parent $PSScriptRoot
$failures = 0

function Report($name, $expected, $actual)
{
    $ok = ($expected -eq $actual)
    $tag = if ($ok) { "PASS" } else { "FAIL" }
    Write-Host ("{0}  {1}`n      expected {2}`n      actual   {3}" -f $tag, $name, $expected, $actual)
    if (-not $ok) { $script:failures++ }
    return $ok
}

Write-Host "=== Property 1: bad input is rejected before any work ==="
# Drives the built evaluator with inputs that must not start a run. Before the
# refactor every one of these is accepted, which is the red step at program level.
$binary = Join-Path $root "build/Release/AlphaZeroEvaluate.exe"
if (Test-Path $binary)
{
    $cases = @(
        @{ args = @("--checkpoint", "nope.pt", "--games", "0"); why = "--games 0" },
        @{ args = @("--checkpoint", "nope.pt", "--board", "10x10"); why = "--board 10x10" },
        @{ args = @("--checkpoint", "nope.pt", "--bord", "12"); why = "misspelled flag" },
        @{ args = @("--checkpoint", "nope.pt", "--board"); why = "trailing flag, no value" }
    )
    foreach ($case in $cases)
    {
        $output = & $binary @($case.args) 2>&1 | Out-String
        # The only evidence that counts is whether the program got as far as
        # loading the checkpoint. A warning on stderr followed by a full run is
        # the defect, not a rejection, and reading the warning alone would score
        # the misspelled-flag case as already fixed.
        $reached_load = [bool]($output -match "could not load")
        Report ("bad input stopped before the checkpoint load: " + $case.why) $false $reached_load | Out-Null
    }
}
else
{
    Write-Host "  AlphaZeroEvaluate.exe not built - cannot probe"
}

Write-Host "`n=== Property 2: no step-limit sentinel ==="
$sentinel = (Select-String -Path "$root/src/az_evaluate.cpp","$root/src/az_visual.cpp" `
    -Pattern "step_limit == 0").Count
Report "occurrences of 'step_limit == 0' in the two az programs" 0 $sentinel | Out-Null

Write-Host "`n=== Property 3: one owner per constant ==="
$twelve = (Select-String -Path "$root/src/az_evaluate.cpp","$root/src/az_visual.cpp" `
    -Pattern "12 \* settings\.board").Count
Report "literal '12 * settings.board'" 0 $twelve | Out-Null
$search_literals = (Select-String -Path "$root/src/az_evaluate.cpp","$root/src/az_visual.cpp" `
    -Pattern "exploration = 0\.5f|discount = 0\.98f|root_noise_alpha = 0\.3f").Count
Report "hardcoded search hyperparameters" 0 $search_literals | Out-Null
$stoi = (Select-String -Path "$root/src/az_evaluate.cpp","$root/src/az_visual.cpp" `
    -Pattern "std::stoi|std::stoul").Count
Report "unvalidated std::stoi / std::stoul calls" 0 $stoi | Out-Null

Write-Host "`n=== Property 5: per-game outcomes and a recorded batch ==="
# One line per game, tagged so a log parser can find them without guessing at
# the progress lines. The expected value is a constant, not a copy of the actual -
# the first version of this compared the reading to itself and printed PASS
# whatever the source said.
$per_game = (Select-String -Path "$root/src/az_evaluate.cpp" -Pattern '"\s*game ').Count
Report "per-game output lines emitted" $true ($per_game -gt 0) | Out-Null
$batch_in_header = (Select-String -Path "$root/src/az_evaluate.cpp" -Pattern "batch \{\}").Count
Report "batch recorded in the evaluation header" $true ($batch_in_header -gt 0) | Out-Null

Write-Host "`n=== Property 4: the measurement does not move ==="
Write-Host "  baseline, read from build/Release/eval123_limit1200.log line 5:"
Write-Host "    az10_iter123.pt, 64 games, seed offset 0, step limit 1200, batch 64 -> 36 wins"
if ($WithAcceptance)
{
    $out = Join-Path $root "build/Release/eval123_acceptance.log"
    Push-Location (Join-Path $root "build/Release")
    & ./AlphaZeroEvaluate.exe --checkpoint az10_iter123.pt --board 10 --games 64 `
        --simulations 200 --step-limit 1200 --batch 64 --seed 0 | Tee-Object $out
    Pop-Location
    $line = Select-String -Path $out -Pattern "^Wins:\s+(\d+)/64"
    $wins = if ($line) { [int]$line.Matches[0].Groups[1].Value } else { -1 }
    Report "wins on the acceptance run" 36 $wins | Out-Null
}
else
{
    Write-Host "  not run (pass -WithAcceptance; costs about 5 minutes on the GPU)"
}

Write-Host "`nfailures: $failures"
exit $failures
