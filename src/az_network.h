#pragma once
#include <torch/torch.h>
#include "snake_env.h"

// Policy and value network for the search.
//
// Convolutional, because the observation is a board and the thing that has to
// be recognised - a region about to be sealed off - is spatial and appears
// anywhere on it. The old network was three hand-rolled weight matrices over
// eight scalars; nothing about that shape can express the position.
//
// One trunk, two heads, as in AlphaZero: a prior over the three relative
// actions to order the search, and a value to stand in for the rest of the game
// at a leaf. Board size is a constructor argument, so the same architecture
// trains at 6x6 and transfers up - only the head's linear layers depend on it.
// What one forward pass produces. A struct rather than a pair, because `first`
// and `second` name the order the two were written in and nothing else - and the
// two are not interchangeable: one is unnormalised logits over actions and the
// other a bounded scalar. Reading `.first` at a call site tells you neither.
//
// Structured bindings work on this exactly as they did on the pair, so callers
// that already wrote `auto [policy, value] = ...` are unaffected.
struct Prediction
{
    // [N, ACTION_COUNT]. Logits, not probabilities - no softmax has been applied,
    // and the search's evaluator is what normalises them.
    torch::Tensor policy_logits;
    // [N, 1] in (-1, 1), from a tanh. Stands in for the discounted return from
    // this position onward.
    torch::Tensor value;
};

struct AlphaZeroNetImpl : torch::nn::Module
{
    AlphaZeroNetImpl(int board_width, int board_height, int channels, int blocks);

    // Not copyable and not movable, and nothing is defined: the base class owns
    // the parameters through shared pointers and its destructor releases them, so
    // this is the rule of zero and the deletions are an interface decision.
    //
    // A copy would be the dangerous kind - it would duplicate the bookkeeping and
    // share every parameter tensor, so two networks would appear independent and
    // train each other's weights. Callers pass the TORCH_MODULE holder below,
    // which is a shared pointer and copyable on purpose; that is the value type
    // here, and this is not.
    AlphaZeroNetImpl(const AlphaZeroNetImpl&) = delete;
    AlphaZeroNetImpl& operator=(const AlphaZeroNetImpl&) = delete;
    AlphaZeroNetImpl(AlphaZeroNetImpl&&) = delete;
    AlphaZeroNetImpl& operator=(AlphaZeroNetImpl&&) = delete;

    // Checks its input and throws on a mismatch rather than asserting it. Nothing
    // in this file can be exercised in a debug build: LibTorch here ships
    // release-only libraries, and a debug binary linked against them dies of an
    // access violation before it reaches any assertion. Measured, not assumed.
    // TORCH_CHECK is live in the configuration this code actually runs in.
    Prediction forward(torch::Tensor planes);

    int boardWidth() const { return board_width_; }
    int boardHeight() const { return board_height_; }

private:
    int board_width_;
    int board_height_;
    int channels_;

    torch::nn::Conv2d stem_conv{nullptr};
    torch::nn::BatchNorm2d stem_norm{nullptr};

    // Residual blocks, stored flat: two convolutions and two norms each.
    std::vector<torch::nn::Conv2d> block_convs;
    std::vector<torch::nn::BatchNorm2d> block_norms;

    torch::nn::Conv2d policy_conv{nullptr};
    torch::nn::BatchNorm2d policy_norm{nullptr};
    torch::nn::Linear policy_out{nullptr};

    torch::nn::Conv2d value_conv{nullptr};
    torch::nn::BatchNorm2d value_norm{nullptr};
    torch::nn::Linear value_hidden{nullptr};
    torch::nn::Linear value_out{nullptr};
};

TORCH_MODULE(AlphaZeroNet);
