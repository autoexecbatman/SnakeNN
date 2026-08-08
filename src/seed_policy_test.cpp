#include "seed_policy.h"
#include <format>
#include <iostream>
#include <set>
#include <string>

// The property that was asserted in a comment and checked nowhere.
//
// Training and evaluation seeds overlapped for the whole of this project's
// history: training used `seed + iteration * 100003` from a default seed of 1,
// so iteration 9 covered 900028..900283, and evaluation defaulted to 900000.
// 172 of 200 evaluation games were games the agent had trained on. A comment on
// the evaluation default claimed the opposite.
//
// These checks exist so that a future change to either range fails here rather
// than quietly invalidating a season of measurements.

namespace
{

constexpr int GAMES_PER_ITERATION = 256;
// Far beyond anything that will be run; at ten minutes an iteration this is
// years of compute.
constexpr int ITERATIONS_CHECKED = 20000;
constexpr unsigned int RUN_SEEDS_CHECKED = 8;

int failures = 0;

void check(bool condition, const std::string& name, const std::string& detail)
{
    if (condition)
    {
        std::cout << std::format("[PASS] {}  {}\n", name, detail);
    }
    else
    {
        std::cout << std::format("[FAIL] {}  {}\n", name, detail);
        failures++;
    }
}

}  // namespace

int main()
{
    // 1. No reachable training seed lands in the evaluation range.
    unsigned int highest = 0;
    bool any_in_evaluation_range = false;
    for (unsigned int run_seed = 1; run_seed <= RUN_SEEDS_CHECKED; run_seed++)
    {
        for (int iteration = 1; iteration <= ITERATIONS_CHECKED; iteration++)
        {
            unsigned int last =
                seeds::trainingGameSeed(run_seed, iteration, GAMES_PER_ITERATION - 1);
            highest = std::max(highest, last);
            if (last >= seeds::EVALUATION_BASE)
            {
                any_in_evaluation_range = true;
            }
        }
    }
    check(!any_in_evaluation_range, "training seeds never reach the evaluation range",
          std::format("highest training seed {} over {} iterations, evaluation base {}", highest,
                      ITERATIONS_CHECKED, seeds::EVALUATION_BASE));

    // 2. The guard actually fires. Without this the check above could pass
    //    because the guard is inert rather than because the ranges are disjoint.
    bool threw = false;
    try
    {
        seeds::requireTrainingSeed(seeds::EVALUATION_BASE);
    }
    catch (const std::logic_error&)
    {
        threw = true;
    }
    check(threw, "the guard rejects a seed at the evaluation base",
          "requireTrainingSeed(EVALUATION_BASE) threw");

    bool threw_below = false;
    try
    {
        seeds::requireTrainingSeed(seeds::EVALUATION_BASE - 1);
    }
    catch (const std::logic_error&)
    {
        threw_below = true;
    }
    check(!threw_below, "the guard accepts the seed just below it",
          "requireTrainingSeed(EVALUATION_BASE - 1) did not throw");

    // 3. Distinct iterations produce disjoint blocks of game seeds, so no game
    //    is ever generated twice within a run.
    std::set<unsigned int> seen;
    bool collided = false;
    for (int iteration = 1; iteration <= 200 && !collided; iteration++)
    {
        for (int game = 0; game < GAMES_PER_ITERATION; game++)
        {
            if (!seen.insert(seeds::trainingGameSeed(1, iteration, game)).second)
            {
                collided = true;
                break;
            }
        }
    }
    check(!collided, "iterations produce disjoint game seeds",
          std::format("{} distinct seeds over 200 iterations", seen.size()));

    // 4. The historical defect, stated as a regression check: the old scheme put
    //    iteration 9 on top of the old evaluation default. If someone reinstates
    //    absolute 900000 as an evaluation base, this is the arithmetic that
    //    should stop them.
    const unsigned int old_iteration_nine_first = 1 + 9u * seeds::ITERATION_STRIDE;
    const unsigned int old_iteration_nine_last = old_iteration_nine_first + 255u;
    const bool old_scheme_overlapped =
        old_iteration_nine_first <= 900199u && old_iteration_nine_last >= 900000u;
    check(old_scheme_overlapped, "the historical overlap is reproduced, so the fix is not cosmetic",
          std::format("old iteration 9 covered {}..{}, old evaluation range was 900000..900199",
                      old_iteration_nine_first, old_iteration_nine_last));

    if (failures == 0)
    {
        std::cout << "all properties held" << std::endl;
        return 0;
    }
    std::cout << std::format("{} properties failed\n", failures);
    return 1;
}
