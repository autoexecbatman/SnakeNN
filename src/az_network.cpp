// Implementation of AlphaZeroNet. What it is and how to call it are in az_network.h.
//
// Every head pools to a fixed grid before its linear layers, so no weight depends on
// board size. forward throws on a batch shaped for a different board; loadNarrowerStem
// loads a checkpoint with fewer input planes and zeroes the new channels.
//
// LibTorch ships release-only libraries, so nothing here is reachable in a debug build -
// the checks are TORCH_CHECK and std::invalid_argument, never assert.

#include <torch/torch.h>

#include <format>
#include <stdexcept>

#include "az_network.h"
#include "az_parameters.h"
#include "snake_env.h"

namespace
{
// Channels the policy head narrows the trunk to before pooling.
constexpr int POLICY_HEAD_CHANNELS = 32;
// The same, for the value, steps and risk heads, which share a stem.
constexpr int VALUE_HEAD_CHANNELS = 32;
// Width of the hidden layer the value, steps and risk heads share.
constexpr int VALUE_HIDDEN = 64;
// Both heads pool to a fixed grid before their linear layers, so no weight
// depends on board size and a 6x6 network loads at 20x20 unchanged. Flattening
// the board straight into a Linear would have made the curriculum impossible -
// every step up in board size would have meant discarding the head and
// relearning it, which is the part that knows what the trunk's features mean.
constexpr int POOLED_SIDE = 4;
// Cells a head sees after pooling, which is what fixes its linear layer's input size.
constexpr int POOLED_CELLS = POOLED_SIDE * POOLED_SIDE;
}  // namespace

AlphaZeroNetImpl::AlphaZeroNetImpl(int board_width, int board_height, int channels, int blocks)
    : board_width_(board_width), board_height_(board_height), channels_(channels)
{
    // Rejects a board below 2x2, a trunk under one channel and a negative block count.
    // Each arrives from a command line, and each builds a network that trains and means
    // nothing - a zero-channel trunk constructs happily and emits constant policies.
    TORCH_CHECK(board_width >= 2 && board_height >= 2,
                std::format("AlphaZeroNet needs a board of at least 2x2, got {}x{}", board_width,
                            board_height));
    TORCH_CHECK(channels >= 1,
                std::format("AlphaZeroNet needs at least one trunk channel, got {}", channels));
    TORCH_CHECK(blocks >= 0,
                std::format("AlphaZeroNet cannot have a negative block count, got {}", blocks));

    // The stem: SnakeEnv::PLANE_COUNT input planes to `channels`, 3x3 with padding so
    // the board keeps its size. No bias, because the batch norm after it has one.
    stem_conv = register_module(
        "stem_conv",
        torch::nn::Conv2d(
            torch::nn::Conv2dOptions(SnakeEnv::PLANE_COUNT, channels_, 3).padding(1).bias(false)));
    stem_norm = register_module("stem_norm", torch::nn::BatchNorm2d(channels_));

    // `blocks` residual blocks, two 3x3 convolutions each, all `channels` wide. Named
    // by block and layer, since a checkpoint is matched to modules by name.
    for (int block = 0; block < blocks; block++)
    {
        for (int layer = 0; layer < 2; layer++)
        {
            std::string suffix = std::to_string(block) + "_" + std::to_string(layer);
            block_convs.push_back(register_module(
                "block_conv_" + suffix,
                torch::nn::Conv2d(
                    torch::nn::Conv2dOptions(channels_, channels_, 3).padding(1).bias(false))));
            block_norms.push_back(
                register_module("block_norm_" + suffix, torch::nn::BatchNorm2d(channels_)));
        }
    }

    // Policy head: a prior over the three relative actions. The 1x1 convolution cuts the
    // trunk to POLICY_HEAD_CHANNELS before the pool, so the linear layer stays small.
    policy_conv = register_module(
        "policy_conv",
        torch::nn::Conv2d(
            torch::nn::Conv2dOptions(channels_, POLICY_HEAD_CHANNELS, 1).bias(false)));
    policy_norm = register_module("policy_norm", torch::nn::BatchNorm2d(POLICY_HEAD_CHANNELS));
    policy_out = register_module(
        "policy_out",
        torch::nn::Linear(POLICY_HEAD_CHANNELS * POOLED_CELLS, SnakeEnv::ACTION_COUNT));

    // Value head: one number for the position, through a VALUE_HIDDEN layer. forward
    // scales a tanh by az::VALUE_SCALE, so the head is bounded to plus or minus that.
    value_conv = register_module(
        "value_conv",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(channels_, VALUE_HEAD_CHANNELS, 1).bias(false)));
    value_norm = register_module("value_norm", torch::nn::BatchNorm2d(VALUE_HEAD_CHANNELS));
    value_hidden = register_module(
        "value_hidden", torch::nn::Linear(VALUE_HEAD_CHANNELS * POOLED_CELLS, VALUE_HIDDEN));
    value_out = register_module("value_out", torch::nn::Linear(VALUE_HIDDEN, 1));

    // Steps head: how many moves remain to fill the board, as a fraction of the step
    // budget, through a sigmoid. Undiscounted, so it is the only estimate here that
    // reaches the deadline. Shaped like the value head, so it too is board-size free.
    steps_conv = register_module(
        "steps_conv",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(channels_, VALUE_HEAD_CHANNELS, 1).bias(false)));
    steps_norm = register_module("steps_norm", torch::nn::BatchNorm2d(VALUE_HEAD_CHANNELS));
    steps_hidden = register_module(
        "steps_hidden", torch::nn::Linear(VALUE_HEAD_CHANNELS * POOLED_CELLS, VALUE_HIDDEN));
    steps_out = register_module("steps_out", torch::nn::Linear(VALUE_HIDDEN, 1));

    // Death head: per action, the probability that taking it leads to a death no later
    // play can avoid, through a sigmoid. One output per action because the search
    // consumes it as a cap on an action. Pooled like the others, so board-size free.
    death_conv = register_module(
        "death_conv",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(channels_, VALUE_HEAD_CHANNELS, 1).bias(false)));
    death_norm = register_module("death_norm", torch::nn::BatchNorm2d(VALUE_HEAD_CHANNELS));
    death_hidden = register_module(
        "death_hidden", torch::nn::Linear(VALUE_HEAD_CHANNELS * POOLED_CELLS, VALUE_HIDDEN));
    death_out =
        register_module("death_out", torch::nn::Linear(VALUE_HIDDEN, SnakeEnv::ACTION_COUNT));

    // Ownership head: one channel at full board resolution, straight off the trunk. No
    // pooling and no linear layer, so it carries no board-size dependence and adds only
    // channels_ + 1 parameters.
    ownership_conv = register_module("ownership_conv",
                                     torch::nn::Conv2d(torch::nn::Conv2dOptions(channels_, 1, 1)));
}

Prediction AlphaZeroNetImpl::forward(torch::Tensor planes)
{
    // Refuses anything but [N, PLANE_COUNT, height, width] on this network's own board.
    // The pooling that frees the weights from board size also hides a wrong one: a 6x6
    // network fed 20x20 planes runs to completion and returns confident nonsense.
    // Transfer means building a network at the new size and copying weights into it.
    TORCH_CHECK(
        planes.dim() == 4,
        std::format("forward expects [N, planes, height, width], got {} dimensions", planes.dim()));
    TORCH_CHECK(planes.size(1) == SnakeEnv::PLANE_COUNT,
                std::format("forward expects {} planes per state, got {}", SnakeEnv::PLANE_COUNT,
                            planes.size(1)));
    TORCH_CHECK(planes.size(2) == board_height_ && planes.size(3) == board_width_,
                std::format("forward was built for a {}x{} board but given {}x{}", board_width_,
                            board_height_, planes.size(3), planes.size(2)));
    TORCH_CHECK(planes.size(0) > 0, "forward given an empty batch");

    // The stem: the input planes become `channels` feature maps, at full board size.
    torch::Tensor trunk = torch::relu(stem_norm(stem_conv(planes)));

    // Each residual block is two convolutions, taken two entries at a time because the
    // modules are stored flat.
    for (size_t block = 0; block < block_convs.size(); block += 2)
    {
        torch::Tensor residual = trunk;
        trunk = torch::relu(block_norms[block](block_convs[block](trunk)));
        trunk = block_norms[block + 1](block_convs[block + 1](trunk));
        // Pre-activation sum, so gradient reaches the stem unattenuated through
        // the identity path however deep the trunk gets.
        trunk = torch::relu(trunk + residual);
    }

    // Policy head: pooled to 4x4, then one linear layer to three logits. The softmax is
    // the caller's, so a training loss can take log_softmax instead.
    torch::Tensor policy = torch::relu(policy_norm(policy_conv(trunk)));
    policy = torch::adaptive_avg_pool2d(policy, { POOLED_SIDE, POOLED_SIDE });
    policy = policy_out(policy.flatten(1));

    // Value head: pooled, one hidden layer, then a single number.
    torch::Tensor value = torch::relu(value_norm(value_conv(trunk)));
    value = torch::adaptive_avg_pool2d(value, { POOLED_SIDE, POOLED_SIDE });
    value = torch::relu(value_hidden(value.flatten(1)));
    // Bounded, because the search compares leaves and an unbounded head lets one
    // bad extrapolation dominate every comparison it appears in - but bounded in
    // the units a reward is quoted in, since backup adds the two together.
    value = az::VALUE_SCALE * torch::tanh(value_out(value));

    // Steps head: the same pipeline, ending in one number.
    torch::Tensor steps = torch::relu(steps_norm(steps_conv(trunk)));
    steps = torch::adaptive_avg_pool2d(steps, { POOLED_SIDE, POOLED_SIDE });
    steps = torch::relu(steps_hidden(steps.flatten(1)));
    // Bounded to (0, 1) because the target is a fraction of the step budget, and
    // a fraction is what makes the estimate mean the same thing on every board.
    steps = torch::sigmoid(steps_out(steps));

    // Death head: the same pipeline, ending in one number per action.
    torch::Tensor death_risk = torch::relu(death_norm(death_conv(trunk)));
    death_risk = torch::adaptive_avg_pool2d(death_risk, { POOLED_SIDE, POOLED_SIDE });
    death_risk = torch::relu(death_hidden(death_risk.flatten(1)));
    // Bounded to (0, 1) because the target is a probability: the chance that this
    // action leads to a death nothing after it can avoid.
    death_risk = torch::sigmoid(death_out(death_risk));

    // Ownership head: one value per cell, at the board's own resolution. A sigmoid,
    // because the target is a probability - will the head visit this cell before the game
    // ends. No pooling and no flatten, so the same weights serve any board.
    torch::Tensor ownership = torch::sigmoid(ownership_conv(trunk));

    // Refuses a head whose shape does not match the batch it was given. Every consumer
    // indexes these without checking, so a wrong batch dimension would misalign targets
    // with positions and train the network on the wrong labels.
    TORCH_CHECK(policy.dim() == 2 && policy.size(0) == planes.size(0) &&
                    policy.size(1) == SnakeEnv::ACTION_COUNT,
                std::format("the policy head produced [{}] for a batch of {}, expected [{}, {}]",
                            policy.dim(), planes.size(0), planes.size(0), SnakeEnv::ACTION_COUNT));
    TORCH_CHECK(value.dim() == 2 && value.size(0) == planes.size(0) && value.size(1) == 1,
                std::format("the value head produced [{}] for a batch of {}, expected [{}, 1]",
                            value.dim(), planes.size(0), planes.size(0)));
    TORCH_CHECK(steps.dim() == 2 && steps.size(0) == planes.size(0) && steps.size(1) == 1,
                std::format("the steps head produced [{}] for a batch of {}, expected [{}, 1]",
                            steps.dim(), planes.size(0), planes.size(0)));

    TORCH_CHECK(
        death_risk.dim() == 2 && death_risk.size(0) == planes.size(0) &&
            death_risk.size(1) == SnakeEnv::ACTION_COUNT,
        std::format("the death head produced [{}] for a batch of {}, expected [{}, {}]",
                    death_risk.dim(), planes.size(0), planes.size(0), SnakeEnv::ACTION_COUNT));

    // The five heads, in the order Prediction declares them.
    return Prediction{ policy, value, steps, death_risk, ownership };
}

namespace
{

// Reads one dotted parameter name out of a serialized module.
//
// A saved module nests its submodules as sub-archives, so "stem_conv.weight" is
// "weight" inside "stem_conv" rather than a key of its own - reading the dotted
// name flat reports the tensor as missing.
void readNested(torch::serialize::InputArchive& archive, const std::string& dotted_key,
                torch::Tensor& out, bool is_buffer)
{
    const size_t split = dotted_key.rfind('.');
    if (split == std::string::npos)
    {
        archive.read(dotted_key, out, is_buffer);
        return;
    }
    torch::serialize::InputArchive nested;
    archive.read(dotted_key.substr(0, split), nested);
    readNested(nested, dotted_key.substr(split + 1), out, is_buffer);
}

}  // namespace

torch::Tensor widenStemWeight(const torch::Tensor& saved, const torch::Tensor& target)
{
    if (saved.dim() != 4 || target.dim() != 4)
    {
        throw std::invalid_argument(std::format("a stem weight has four dimensions; got {} and {}",
                                                saved.dim(), target.dim()));
    }
    if (saved.size(1) > target.size(1))
    {
        throw std::invalid_argument(
            std::format("the checkpoint's stem takes {} planes, wider than this network's {}",
                        saved.size(1), target.size(1)));
    }
    if (saved.size(0) != target.size(0) || saved.size(2) != target.size(2) ||
        saved.size(3) != target.size(3))
    {
        throw std::invalid_argument(
            std::format("the checkpoint's stem differs in more than its input planes: "
                        "[{}, {}, {}, {}] against [{}, {}, {}, {}]",
                        saved.size(0), saved.size(1), saved.size(2), saved.size(3), target.size(0),
                        target.size(1), target.size(2), target.size(3)));
    }

    // Zeroed new channels, so the widened convolution computes exactly what the
    // narrower one did until training moves them.
    torch::NoGradGuard no_grad;
    torch::Tensor widened = torch::zeros_like(target);
    widened.narrow(1, 0, saved.size(1)).copy_(saved);
    return widened;
}

namespace
{

// Whether the archive held the key, leaving `out` untouched when it did not.
bool tryReadNested(torch::serialize::InputArchive& archive, const std::string& dotted_key,
                   torch::Tensor& out, bool is_buffer)
{
    try
    {
        readNested(archive, dotted_key, out, is_buffer);
        return true;
    }
    catch (const std::exception&)
    {
        return false;
    }
}

}  // namespace

std::vector<std::string> AlphaZeroNetImpl::loadNarrowerStem(const std::string& checkpoint_path)
{
    // Read the checkpoint into a second network built with the stem it was saved
    // with, then copy across. Reconstructing the old shape is what lets
    // torch::load do the parsing rather than this function reimplementing it.
    torch::serialize::InputArchive archive;
    archive.load_from(checkpoint_path);

    // Copied entry by entry rather than replayed through load(), which refuses the
    // whole archive on the one tensor whose shape changed. An InputArchive cannot
    // be edited, so the widening happens here.
    torch::NoGradGuard no_grad;
    std::vector<std::string> missing;

    for (auto& parameter : named_parameters())
    {
        torch::Tensor saved;
        if (!tryReadNested(archive, parameter.key(), saved, false))
        {
            missing.push_back(parameter.key());
            continue;
        }

        if (parameter.key() == "stem_conv.weight")
        {
            parameter.value().copy_(widenStemWeight(saved, parameter.value()));
            continue;
        }

        TORCH_CHECK(saved.sizes() == parameter.value().sizes(),
                    std::format("'{}' has a different shape in the checkpoint than in this network",
                                parameter.key()));
        parameter.value().copy_(saved);
    }

    // Batch normalisation keeps its running statistics in buffers, not parameters,
    // and a fine-tune that dropped them would start from the wrong normalisation.
    for (auto& buffer : named_buffers())
    {
        torch::Tensor saved;
        if (!tryReadNested(archive, buffer.key(), saved, true))
        {
            missing.push_back(buffer.key());
            continue;
        }
        TORCH_CHECK(
            saved.sizes() == buffer.value().sizes(),
            std::format("buffer '{}' has a different shape in the checkpoint", buffer.key()));
        buffer.value().copy_(saved);
    }

    return missing;
}
