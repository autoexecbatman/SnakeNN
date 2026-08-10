#include <torch/torch.h>

#include <format>
#include <stdexcept>

#include "az_network.h"
#include "snake_env.h"

namespace
{
constexpr int POLICY_HEAD_CHANNELS = 32;
constexpr int VALUE_HEAD_CHANNELS = 32;
constexpr int VALUE_HIDDEN = 64;
// Both heads pool to a fixed grid before their linear layers, so no weight
// depends on board size and a 6x6 network loads at 20x20 unchanged. Flattening
// the board straight into a Linear would have made the curriculum impossible -
// every step up in board size would have meant discarding the head and
// relearning it, which is the part that knows what the trunk's features mean.
constexpr int POOLED_SIDE = 4;
constexpr int POOLED_CELLS = POOLED_SIDE * POOLED_SIDE;
}  // namespace

AlphaZeroNetImpl::AlphaZeroNetImpl(int board_width, int board_height, int channels, int blocks)
    : board_width_(board_width), board_height_(board_height), channels_(channels)
{
    // Checked at construction, because every one of these is a size a caller
    // passes in from a command line and a wrong one produces a network that builds
    // and trains and means nothing. A zero-channel trunk in particular constructs
    // happily and emits constant policies.
    TORCH_CHECK(board_width >= 2 && board_height >= 2,
                std::format("AlphaZeroNet needs a board of at least 2x2, got {}x{}", board_width,
                            board_height));
    TORCH_CHECK(channels >= 1,
                std::format("AlphaZeroNet needs at least one trunk channel, got {}", channels));
    TORCH_CHECK(blocks >= 0,
                std::format("AlphaZeroNet cannot have a negative block count, got {}", blocks));

    stem_conv = register_module(
        "stem_conv",
        torch::nn::Conv2d(
            torch::nn::Conv2dOptions(SnakeEnv::PLANE_COUNT, channels_, 3).padding(1).bias(false)));
    stem_norm = register_module("stem_norm", torch::nn::BatchNorm2d(channels_));

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

    policy_conv = register_module(
        "policy_conv",
        torch::nn::Conv2d(
            torch::nn::Conv2dOptions(channels_, POLICY_HEAD_CHANNELS, 1).bias(false)));
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

    // Same shape as the value head and pooled the same way, so nothing it holds
    // depends on board size either and the curriculum still transfers.
    steps_conv = register_module(
        "steps_conv",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(channels_, VALUE_HEAD_CHANNELS, 1).bias(false)));
    steps_norm = register_module("steps_norm", torch::nn::BatchNorm2d(VALUE_HEAD_CHANNELS));
    steps_hidden = register_module(
        "steps_hidden", torch::nn::Linear(VALUE_HEAD_CHANNELS * POOLED_CELLS, VALUE_HIDDEN));
    steps_out = register_module("steps_out", torch::nn::Linear(VALUE_HIDDEN, 1));
}

Prediction AlphaZeroNetImpl::forward(torch::Tensor planes)
{
    // The pooling that makes this architecture board-size independent also makes
    // it silent about the wrong board: a 6x6 network fed 20x20 planes runs to
    // completion and returns confident nonsense, because nothing downstream of the
    // pool can tell what went in. Transfer is done by constructing a network at
    // the new size and copying weights into it, never by feeding a different size
    // to an existing one - so a mismatch here is a caller that encoded against the
    // wrong board, and it is worth refusing.
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

    torch::Tensor trunk = torch::relu(stem_norm(stem_conv(planes)));

    for (size_t block = 0; block < block_convs.size(); block += 2)
    {
        torch::Tensor residual = trunk;
        trunk = torch::relu(block_norms[block](block_convs[block](trunk)));
        trunk = block_norms[block + 1](block_convs[block + 1](trunk));
        // Pre-activation sum, so gradient reaches the stem unattenuated through
        // the identity path however deep the trunk gets.
        trunk = torch::relu(trunk + residual);
    }

    torch::Tensor policy = torch::relu(policy_norm(policy_conv(trunk)));
    policy = torch::adaptive_avg_pool2d(policy, { POOLED_SIDE, POOLED_SIDE });
    policy = policy_out(policy.flatten(1));

    torch::Tensor value = torch::relu(value_norm(value_conv(trunk)));
    value = torch::adaptive_avg_pool2d(value, { POOLED_SIDE, POOLED_SIDE });
    value = torch::relu(value_hidden(value.flatten(1)));
    // Bounded, because the return it stands in for is bounded: the search
    // compares leaves, and an unbounded head lets one bad extrapolation
    // dominate every comparison it appears in.
    value = torch::tanh(value_out(value));

    torch::Tensor steps = torch::relu(steps_norm(steps_conv(trunk)));
    steps = torch::adaptive_avg_pool2d(steps, { POOLED_SIDE, POOLED_SIDE });
    steps = torch::relu(steps_hidden(steps.flatten(1)));
    // Bounded to (0, 1) because the target is a fraction of the step budget, and
    // a fraction is what makes the estimate mean the same thing on every board.
    steps = torch::sigmoid(steps_out(steps));

    // The two shapes every consumer indexes without checking: the evaluator reads
    // ACTION_COUNT priors per state and one value, and the trainer builds its loss
    // against both. A wrong batch dimension here would misalign policy targets with
    // positions and train the network on the wrong labels.
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

    return Prediction{ policy, value, steps };
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
