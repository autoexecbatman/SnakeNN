#pragma once

#include <vector>

#include "snake_env.h"

// A block of independent games stepped together.
//
// Throughput is the whole point: a 4080 evaluating one 8x20x20 observation is
// almost entirely idle, so the learner keeps thousands of games in flight and
// hands the network one contiguous batch. Each game owns its own generator and
// is seeded from a base, so a run reproduces exactly.
//
// Episodes are not restarted behind the caller's back. A finished game stays
// finished until resetOne, because the trainer needs to read its outcome before
// it disappears, and a silent auto-reset is how return bookkeeping goes wrong.
class VectorEnv
{
public:
    // Builds `count` games of one size. Game `index` is seeded `base_seed + index`, so
    // a block reproduces from one number.
    //
    //     VectorEnv games(256, 10, 10, batch_seed, 1200);
    VectorEnv(int count, int width, int height, unsigned int base_seed, int step_limit);

    // Restarts every game, each on the seed it was built with.
    void resetAll();
    // Restarts one game, leaving the rest running. This is what keeps a block full as
    // games finish at different times.
    void resetOne(int index);

    // Steps every game that is not already finished. `actions` and
    // `results_out` both hold `count()` entries; a finished game reports
    // {0, true, its win flag} and is not advanced.
    void step(const SnakeEnv::Action* actions, SnakeEnv::StepResult* results_out);

    // Writes count() * encodedSizePerEnv() floats: game-major, then plane-major
    // within a game, which is the layout a Conv2d batch wants.
    void encodeAll(float* batch_out) const;

    // Games in the block.
    int count() const { return static_cast<int>(envs_.size()); }
    // Floats one game's encoding occupies.
    int encodedSizePerEnv() const { return envs_[0].encodedSize(); }
    // Floats the whole block occupies - size the batch buffer with this.
    int encodedSizeTotal() const { return count() * encodedSizePerEnv(); }

    // One game, for reading its score or outcome.
    const SnakeEnv& env(int index) const { return envs_[index]; }
    // One game, for stepping it.
    SnakeEnv& env(int index) { return envs_[index]; }

private:
    std::vector<SnakeEnv> envs_;
};
