#pragma once

#include <limits>

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

// Seeds evaluation owns: EVALUATION_BASE up to the top of the unsigned range.
//
//     if (offset + games - 1 >= seeds::RESERVED_BAND_WIDTH) { /* would wrap */ }
//
// A run whose last game falls outside this does not error - it wraps to a low
// seed, which is a training seed, which is how the held-out set stopped being
// held out once before. Every program deriving evaluation seeds checks it.
constexpr unsigned int RESERVED_BAND_WIDTH =
    std::numeric_limits<unsigned int>::max() - EVALUATION_BASE + 1u;

// Distinct iterations must not collide, and 256 games per iteration must not
// run into the next iteration's block. 100003 is prime and far above any
// games-per-iteration this will ever use.
constexpr unsigned int ITERATION_STRIDE = 100003u;

// Which generator a stream belongs to. One run seed reaches several of them, and
// two std::mt19937 seeded with the same integer emit the identical sequence - so
// the search and self-play were drawing from one stream, spending it on different
// things and staying in step only by accident.
enum class Stream : unsigned int
{
    // The search's generator: root noise, tie-breaks, chance-node food placement.
    Search = 0,
    // Self-play's generator: which move is sampled at temperature.
    SelfPlaySampling = 1,
    // Torch's global generator: the starting weights, and which positions a batch draws.
    Network = 2,
};

// Distance between two consumers' streams. 2^31 - 1, a Mersenne prime, so two
// streams collide only for run seeds differing by exactly this.
constexpr unsigned int STREAM_STRIDE = 2147483647u;

// The seed for one consumer's generator, derived from the run seed.
//
//     rng_(seeds::streamSeed(config.seed, seeds::Stream::SelfPlaySampling))
//
// Stream::Search returns the run seed unchanged, so every number measured before
// this existed stays reproducible.
constexpr unsigned int streamSeed(unsigned int run_seed, Stream stream) noexcept
{
    // Wraps rather than saturating, which is fine: a stream is a generator seed and
    // every unsigned value is a legal one.
    return run_seed + static_cast<unsigned int>(stream) * STREAM_STRIDE;
}

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
