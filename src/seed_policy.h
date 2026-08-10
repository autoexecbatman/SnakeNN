#pragma once

#include <cstdint>
#include <stdexcept>

// Where training seeds and evaluation seeds live, in one place.
//
// This exists because the two ranges silently overlapped. Training game seeds
// were `seed + iteration * 100003` with a default seed of 1, so iteration 9
// covered 900028..900283 - and evaluation defaulted to 900000, whose comment
// claimed the range was "held out from training's seed range". 172 of 200
// evaluation seeds were training seeds. Every held-out number measured before
// 2026-08-08 was scored on games the agent had trained on.
//
// The lesson is not "pick a better constant". It is that the property was
// asserted in a comment and checked nowhere, so nothing could fail when it
// stopped being true. Both ranges are now derived here and the trainer calls
// `requireTrainingSeed` on every game seed it generates.
namespace seeds
{

// Evaluation owns everything at or above this. Chosen far above any reachable
// training seed: at 100003 per iteration a run would need ~37,000 iterations to
// arrive, and iterations take minutes.
constexpr unsigned int EVALUATION_BASE = 0xE0000000u;

// Distinct iterations must not collide, and 256 games per iteration must not
// run into the next iteration's block. 100003 is prime and far above any
// games-per-iteration this will ever use.
constexpr unsigned int ITERATION_STRIDE = 100003u;

// The seed for one self-play game.
//
// `iteration` is absolute, not relative to the current process - a resumed run
// continues the sequence rather than replaying it. That was the second half of
// the same defect: `--resume` restored weights but not the counter, so every
// restart regenerated the identical games and about 60 distinct seed bases were
// all this project ever used.
constexpr unsigned int trainingGameSeed(unsigned int run_seed, int iteration,
                                        int game_index) noexcept
{
    return run_seed + static_cast<unsigned int>(iteration) * ITERATION_STRIDE +
           static_cast<unsigned int>(game_index);
}

// Fails loudly if a training seed has wandered into the evaluation range.
//
// A comment cannot notice when it stops being true; this can. It is called on
// the base seed of every iteration, which is cheap and fires before any game is
// played rather than after the numbers have been published.
inline void requireTrainingSeed(unsigned int seed)
{
    if (seed >= EVALUATION_BASE)
    {
        throw std::logic_error(
            "training seed has reached the reserved evaluation range - the held-out set is no "
            "longer held out; lower the run seed or the iteration count");
    }
}

// The seed for one evaluation game. Held out by construction, not by intent.
constexpr unsigned int evaluationGameSeed(unsigned int offset, int game_index) noexcept
{
    return EVALUATION_BASE + offset + static_cast<unsigned int>(game_index);
}

}  // namespace seeds
