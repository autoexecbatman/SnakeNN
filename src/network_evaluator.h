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
class NetworkEvaluator : public Evaluator {
public:
    NetworkEvaluator(AlphaZeroNet network, torch::Device device);

    void evaluate(const std::vector<const SnakeEnv*>& states,
                  float* priors_out,
                  float* values_out) override;

    // Positions scored since construction. The search is evaluation-bound, so
    // this is the meaningful unit of work done.
    long long evaluations() const { return evaluations_; }

    AlphaZeroNet network() const { return network_; }

private:
    AlphaZeroNet network_;
    torch::Device device_;
    std::vector<float> staging_;
    long long evaluations_;
};
