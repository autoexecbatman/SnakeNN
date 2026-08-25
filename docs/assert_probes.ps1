# Drives each assertion added on 2026-08-25 past its bound, and checks it aborts.
#
# An abort cannot be caught in-process, so the test suite and the mutation runs are
# both blind to these. A check whose failure has never been seen has not been shown
# to exist - this is the run that sees it.
#
# Each probe is one call violating one invariant, compiled /MDd so the debug runtime
# is present and the assert survives, and expected to exit 3.
#
# Run it:
#
#     powershell -NoProfile -ExecutionPolicy Bypass -File docs/assert_probes.ps1
#
# Needs cl on PATH. Prints one line per probe and a count.

$src = "D:/repo/snakeNN/src"
$work = "$env:TEMP/assert_probes"
Remove-Item $work -Recurse -Force -ErrorAction Ignore
New-Item -ItemType Directory -Force $work | Out-Null

$probes = @(
  @{
    name    = "pickBiased_zero_tries"
    sources = @("replay_sampling.h", "replay_sampling.cpp")
    body    = @'
#include <cstddef>
#include <functional>

#include "replay_sampling.h"

// tries of zero would mean returning an index no draw produced.
int main()
{
    const std::function<std::size_t()> draw = [] { return std::size_t{ 0 }; };
    const std::function<bool(std::size_t)> decisive = [](std::size_t) { return false; };
    return static_cast<int>(sampling::pickBiased(draw, decisive, true, 0));
}
'@
  },
  @{
    name    = "searchWith_zero_simulations"
    sources = @("mcts.h", "mcts.cpp", "snake_env.h", "snake_env.cpp", "small_random.h",
                "evaluator.h", "exploration_floor.h", "exploration_floor.cpp",
                "value_range.h", "value_range.cpp", "az_parameters.h", "snake_logic.h")
    body    = @'
#include <vector>

#include "mcts.h"

// A search of no simulations has no visit distribution to return, so the count is
// asserted rather than defended against.
namespace
{
struct FlatEvaluator : Evaluator
{
    void evaluate(const std::vector<const SnakeEnv*>& states, float* priors_out,
                  float* values_out, float* steps_out, float* death_risk_out) override
    {
        for (std::size_t index = 0; index < states.size(); index++)
        {
            for (int action = 0; action < SnakeEnv::ACTION_COUNT; action++)
            {
                priors_out[index * SnakeEnv::ACTION_COUNT + action] = 1.0f / 3.0f;
                death_risk_out[index * SnakeEnv::ACTION_COUNT + action] = 0.0f;
            }
            values_out[index] = 0.0f;
            steps_out[index] = 1.0f;
        }
    }
};
}  // namespace

int main()
{
    FlatEvaluator evaluator;
    MonteCarloSearch::Config config;
    config.simulations = 8;
    MonteCarloSearch search(evaluator, config);
    SnakeEnv game(6, 6, 1u, 100);
    const std::vector<const SnakeEnv*> roots{ &game };
    return static_cast<int>(search.searchWith(roots, 0).size());
}
'@
  }
)

$aborted = 0
foreach ($probe in $probes)
{
    $dir = Join-Path $work $probe.name
    New-Item -ItemType Directory -Force $dir | Out-Null
    foreach ($file in $probe.sources)
    {
        Copy-Item "$src/$file" $dir
    }
    Set-Content "$dir/probe.cpp" -Value $probe.body -Encoding UTF8

    Push-Location $dir
    $compiled = @("probe.cpp") + ($probe.sources | Where-Object { $_.EndsWith(".cpp") })
    $null = cl /nologo /std:c++20 /EHsc /MDd $compiled /Fe:probe.exe 2>&1 | Out-String
    if ($LASTEXITCODE -ne 0)
    {
        Write-Host ("  {0,-28} DID NOT COMPILE" -f $probe.name)
        Pop-Location
        continue
    }
    # The abort dialog would block, so send it to stderr instead.
    $env:_CRT_ERROR = "2"
    $null = & ./probe.exe 2>&1
    $code = $LASTEXITCODE
    Pop-Location

    if ($code -eq 3)
    {
        Write-Host ("  {0,-28} ABORTED as expected (exit 3)" -f $probe.name)
        $aborted++
    }
    else
    {
        Write-Host ("  {0,-28} DID NOT ABORT - exit {1}, the assert does not fire" -f $probe.name, $code)
    }
}
Write-Host ("`n{0} of {1} assertions fired" -f $aborted, $probes.Count)
