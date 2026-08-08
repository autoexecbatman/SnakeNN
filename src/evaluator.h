#pragma once
#include "snake_env.h"
#include <vector>

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

    // For each state, writes SnakeEnv::ACTION_COUNT priors (summing to one) and
    // one value in [-1, 1] standing for the return from that state onward.
    // Called once per simulation round with every tree's leaf, so batching lives
    // here rather than in the search.
    virtual void evaluate(const std::vector<const SnakeEnv*>& states, float* priors_out,
                          float* values_out) = 0;
};
