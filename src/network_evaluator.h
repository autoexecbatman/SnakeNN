#pragma once

#include <torch/torch.h>

#include "az_network.h"
#include "evaluator.h"

// The network behind the search. Encodes a batch of leaves, runs one forward
// pass, and hands back normalised priors and values.
//
// Holds its own staging buffer so a search that evaluates the same batch size
// every round allocates once. Inference only - gradients are the trainer's
// business, and a search that built a graph would run out of memory in minutes.
class NetworkEvaluator : public Evaluator
{
public:
    // Wraps a network for the search. The network is a shared holder, so this shares
    // the weights rather than copying them; move it to the device first.
    //
    //     network->to(torch::kCUDA);
    //     NetworkEvaluator evaluator(network, torch::kCUDA);
    NetworkEvaluator(AlphaZeroNet network, torch::Device device);

    // Scores a batch of positions in one forward pass, writing into the caller's
    // buffers. Each must hold room for the whole batch: priors and death risks are
    // ACTION_COUNT per state, values and steps one each.
    //
    // Priors come back normalised, not as logits. death_risk_out is filled with zeros
    // while az::DEATH_RISK_FROM_NETWORK is false, whatever the head has learned.
    void evaluate(const std::vector<const SnakeEnv*>& states, float* priors_out, float* values_out,
                  float* steps_out, float* death_risk_out) override;

    // Positions scored since construction. The search is evaluation-bound, so
    // this is the meaningful unit of work done.
    long long evaluations() const { return evaluations_; }

    // The network this was built with, for a caller that needs the weights themselves
    // rather than the search's view of them.
    AlphaZeroNet network() const { return network_; }

private:
    AlphaZeroNet network_;
    torch::Device device_;
    std::vector<float> staging_;
    long long evaluations_{ 0 };
};
