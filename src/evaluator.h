#pragma once

#include <vector>

#include "snake_env.h"

// What the search needs from a network, and nothing more.
//
// The search is separated from LibTorch by this interface for one practical
// reason: its own correctness - selection, backup, discounting, terminal
// handling - is testable against a hand-written evaluator with known answers,
// on a target that builds in seconds without CUDA. A search that can only be
// exercised through a network is a search whose bugs are indistinguishable from
// the network's.
class Evaluator
{
public:
    virtual ~Evaluator() = default;

    // For each state, writes SnakeEnv::ACTION_COUNT priors (summing to one), one
    // value in [-1, 1] standing for the return from that state onward, and one
    // steps-to-go estimate in [0, 1] - the steps still needed to fill the board,
    // as a fraction of the game's whole step budget.
    //
    // Steps-to-go is a separate output rather than folded into the value because
    // the value is discounted and the deadline is not: at 0.98 the return cannot
    // see past about 200 steps, while a game runs 1100. Every buffer is written
    // for every state; none may be left untouched.
    //
    // Called once per simulation round with every tree's leaf, so batching lives
    // here rather than in the search.
    virtual void evaluate(const std::vector<const SnakeEnv*>& states, float* priors_out,
                          float* values_out, float* steps_out) = 0;
};
