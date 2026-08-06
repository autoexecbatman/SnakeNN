#include "az_network.h"

namespace {
const int POLICY_HEAD_CHANNELS = 32;
const int VALUE_HEAD_CHANNELS = 32;
const int VALUE_HIDDEN = 64;
// Both heads pool to a fixed grid before their linear layers, so no weight
// depends on board size and a 6x6 network loads at 20x20 unchanged. Flattening
// the board straight into a Linear would have made the curriculum impossible -
// every step up in board size would have meant discarding the head and
// relearning it, which is the part that knows what the trunk's features mean.
const int POOLED_SIDE = 4;
const int POOLED_CELLS = POOLED_SIDE * POOLED_SIDE;
}  // namespace

AlphaZeroNetImpl::AlphaZeroNetImpl(int board_width, int board_height, int channels, int blocks)
    : board_width_(board_width), board_height_(board_height), channels_(channels) {
    stem_conv = register_module(
        "stem_conv",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(SnakeEnv::PLANE_COUNT, channels_, 3)
                              .padding(1)
                              .bias(false)));
    stem_norm = register_module("stem_norm", torch::nn::BatchNorm2d(channels_));

    for (int block = 0; block < blocks; block++) {
        for (int layer = 0; layer < 2; layer++) {
            std::string suffix = std::to_string(block) + "_" + std::to_string(layer);
            block_convs.push_back(register_module(
                "block_conv_" + suffix,
                torch::nn::Conv2d(
                    torch::nn::Conv2dOptions(channels_, channels_, 3).padding(1).bias(false))));
            block_norms.push_back(
                register_module("block_norm_" + suffix, torch::nn::BatchNorm2d(channels_)));
        }
    }

    policy_conv = register_module(
        "policy_conv",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(channels_, POLICY_HEAD_CHANNELS, 1).bias(false)));
    policy_norm = register_module("policy_norm", torch::nn::BatchNorm2d(POLICY_HEAD_CHANNELS));
    policy_out = register_module(
        "policy_out",
        torch::nn::Linear(POLICY_HEAD_CHANNELS * POOLED_CELLS, SnakeEnv::ACTION_COUNT));

    value_conv = register_module(
        "value_conv",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(channels_, VALUE_HEAD_CHANNELS, 1).bias(false)));
    value_norm = register_module("value_norm", torch::nn::BatchNorm2d(VALUE_HEAD_CHANNELS));
    value_hidden = register_module(
        "value_hidden", torch::nn::Linear(VALUE_HEAD_CHANNELS * POOLED_CELLS, VALUE_HIDDEN));
    value_out = register_module("value_out", torch::nn::Linear(VALUE_HIDDEN, 1));
}

std::pair<torch::Tensor, torch::Tensor> AlphaZeroNetImpl::forward(torch::Tensor planes) {
    torch::Tensor trunk = torch::relu(stem_norm(stem_conv(planes)));

    for (size_t block = 0; block < block_convs.size(); block += 2) {
        torch::Tensor residual = trunk;
        trunk = torch::relu(block_norms[block](block_convs[block](trunk)));
        trunk = block_norms[block + 1](block_convs[block + 1](trunk));
        // Pre-activation sum, so gradient reaches the stem unattenuated through
        // the identity path however deep the trunk gets.
        trunk = torch::relu(trunk + residual);
    }

    torch::Tensor policy = torch::relu(policy_norm(policy_conv(trunk)));
    policy = torch::adaptive_avg_pool2d(policy, {POOLED_SIDE, POOLED_SIDE});
    policy = policy_out(policy.flatten(1));

    torch::Tensor value = torch::relu(value_norm(value_conv(trunk)));
    value = torch::adaptive_avg_pool2d(value, {POOLED_SIDE, POOLED_SIDE});
    value = torch::relu(value_hidden(value.flatten(1)));
    // Bounded, because the return it stands in for is bounded: the search
    // compares leaves, and an unbounded head lets one bad extrapolation
    // dominate every comparison it appears in.
    value = torch::tanh(value_out(value));

    return {policy, value};
}
