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
struct AlphaZeroNetImpl : torch::nn::Module
{
    AlphaZeroNetImpl(int board_width, int board_height, int channels, int blocks);

    // Returns {policy_logits [N,3], value [N,1] in (-1,1)}.
    std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor planes);

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
