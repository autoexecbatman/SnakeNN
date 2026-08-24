#pragma once

#include <torch/torch.h>

#include <string>
#include <vector>

// Policy and value network for the search: one convolutional trunk and four heads - a
// prior over the three relative actions, a value, a steps-to-go estimate, and a death
// risk per action.
//
// Board size is a constructor argument and no weight depends on it, so a network trained
// at 6x6 loads at 20x20. Transfer means constructing at the new size and copying weights
// in; feeding a different size to an existing network is refused by forward.
//
// Usage:
//
//     // Board width, board height, trunk channels, residual blocks.
//     AlphaZeroNet network(10, 10, 64, 4);
//     network->to(torch::kCUDA);   // arrow, not dot - the holder is a shared pointer
//     network->eval();             // train() instead, inside a training step
//
//     // Resuming. Returns the parameters the checkpoint did not carry; print them,
//     // because a mistyped module name looks the same as a head added later.
//     for (const std::string& name : network->loadNarrowerStem("az10.pt"))
//     {
//         std::cout << "fresh: " << name << "\n";
//     }
//
//     // [batch, PLANE_COUNT, height, width]. from_blob does not copy, so the buffer
//     // has to outlive the call.
//     torch::Tensor planes = torch::from_blob(
//         encoded.data(), { batch, SnakeEnv::PLANE_COUNT, 10, 10 }, torch::kFloat)
//         .to(torch::kCUDA);
//
//     torch::NoGradGuard no_grad;   // no autograd graph - inference only
//     const Prediction prediction = network->forward(planes);
//
//     torch::save(network, "checkpoint.pt");   // the holder, never the Impl
//
// Reading prediction.death_risk gives what the head emits. NetworkEvaluator substitutes
// zero for it while az::DEATH_RISK_FROM_NETWORK is false, so a consumer going through
// the evaluator sees zeros whatever the head has learned.

// What one forward pass returns: four tensors, one row each per state of the batch, in
// the order the planes were given. Every field below states its own shape and range.
//
//     const Prediction prediction = network->forward(planes);
//
//     // Logits, not probabilities. Dimension 1 is the action axis.
//     const torch::Tensor priors = torch::softmax(prediction.policy_logits, 1);
//     // Row 0 is the first state of the batch.
//     const float value = prediction.value[0].item<float>();
//
//     // Or bind all four at once, in declaration order.
//     const auto [logits, values, steps, risks] = prediction;
struct Prediction
{
    // [N, ACTION_COUNT]. Logits, not probabilities - no softmax has been applied,
    // and the search's evaluator is what normalises them.
    torch::Tensor policy_logits;
    // [N, 1] in (-VALUE_SCALE, VALUE_SCALE), from a scaled tanh. The discounted
    // return from this position onward, in the units a reward is quoted in - the
    // search adds this to raw edge rewards, so it cannot be a squashed fraction.
    torch::Tensor value;
    // [N, 1] in (0, 1), from a sigmoid. The steps still needed to fill the board,
    // as a fraction of the game's step budget.
    //
    // Its own head rather than part of the value because the two are trained on
    // incomparable targets: the value regresses a discounted return, which at
    // 0.98 cannot see past about 200 steps, and this regresses a raw count over
    // the whole game. Supervised and undiscounted, so it is the only estimate
    // here that reaches the deadline.
    torch::Tensor steps_to_go;
    // [N, ACTION_COUNT] in (0, 1), from a sigmoid. The probability that taking
    // each action leads to a death no later play can avoid.
    //
    // Its own head, and per action rather than per state, for the same reason
    // steps_to_go is separate: the value is discounted at 0.98 and cannot see
    // past about 200 steps, while a region sealed at half fill kills later than
    // that. Fatemi et al. 2019 give the construction - an undiscounted signal
    // that is 1 exactly on the states from which every trajectory dies.
    torch::Tensor death_risk;
    // [N, 1, height, width] in (0, 1), from a sigmoid. Per cell, the probability the head
    // visits it before the game ends.
    //
    // An auxiliary target: nothing consumes it at play time. It exists to shape the trunk,
    // which is KataGo's largest measured gain - per-cell labels give localised credit
    // assignment where a scalar gives one number for a whole board. It is also the densest
    // statement available of the thing this agent gets wrong, which is walking into space
    // it cannot leave.
    //
    // Full board resolution, from a 1x1 convolution rather than a pooled linear layer, so
    // no weight here depends on board size either.
    torch::Tensor ownership;
};

// A copy of `target`'s shape holding `saved` in its leading input channels, with the
// rest zero - so the widened convolution computes what the narrow one did.
//
//     parameter.copy_(widenStemWeight(saved, parameter));   // from loadNarrowerStem
//
// Throws std::invalid_argument unless `saved` fits inside `target`: at most as many
// input channels, and equal in every other dimension.
torch::Tensor widenStemWeight(const torch::Tensor& saved, const torch::Tensor& target);

// The network itself. Callers hold AlphaZeroNet, the shared-pointer holder declared at
// the end of this file; this is what it points at.
struct AlphaZeroNetImpl : torch::nn::Module
{
    // Builds the trunk and the four heads for one board size.
    //
    //     AlphaZeroNet network(10, 10, 64, 4);   // width, height, channels, blocks
    //
    // Throws on a board smaller than 2x2, a trunk of fewer than one channel, or a
    // negative block count - each is a size that arrives from a command line, and a
    // wrong one builds and trains and means nothing.
    AlphaZeroNetImpl(int board_width, int board_height, int channels, int blocks);

    // Copying is deleted. The parameters are shared pointers, so a copy would train one
    // set of weights through two objects that look independent. Pass the AlphaZeroNet
    // holder instead - it is the shared handle this type is meant to be reached through.
    AlphaZeroNetImpl(const AlphaZeroNetImpl&) = delete;
    // Copy assignment, deleted for the same reason.
    AlphaZeroNetImpl& operator=(const AlphaZeroNetImpl&) = delete;
    // Moving is deleted too: a module registered with LibTorch is referred to by
    // address, so moving one leaves those references behind.
    AlphaZeroNetImpl(AlphaZeroNetImpl&&) = delete;
    // Move assignment, deleted for the same reason.
    AlphaZeroNetImpl& operator=(AlphaZeroNetImpl&&) = delete;

    // Runs one batch of encoded boards through the trunk and the four heads.
    //
    //     // A scope guard: while it is alive autograd records nothing. Without it a
    //     // search retains every leaf it evaluates and runs out of memory.
    //     torch::NoGradGuard no_grad;
    //     const Prediction prediction = network->forward(planes);
    //
    // A training step leaves it out - backward needs the graph this suppresses.
    //
    // Throws unless planes is [N, PLANE_COUNT, height, width] for the board this network
    // was built for, with N at least one.
    Prediction forward(torch::Tensor planes);

    // Width of the board this network was built for.
    int boardWidth() const { return board_width_; }
    // Height of the same board. forward refuses a batch shaped for any other.
    int boardHeight() const { return board_height_; }

    // Loads a checkpoint whose stem accepts fewer input planes than this network, copying
    // the saved weights and zeroing the new channels, so the widened network computes what
    // the saved one did until training moves them.
    //
    //     for (const std::string& name : network->loadNarrowerStem("az10.pt"))
    //     {
    //         std::cout << "fresh: " << name << "\n";
    //     }
    //
    // Returns the parameters and buffers the checkpoint did not contain; those keep their
    // constructed values. Print them - a mistyped module name looks the same as a head
    // added after the checkpoint was written.
    //
    // Throws if the checkpoint is wider than this network, or differs anywhere but the
    // stem's input channels.
    std::vector<std::string> loadNarrowerStem(const std::string& checkpoint_path);

private:
    int board_width_{ 0 };
    int board_height_{ 0 };
    int channels_{ 0 };

    torch::nn::Conv2d stem_conv{ nullptr };
    torch::nn::BatchNorm2d stem_norm{ nullptr };

    // Residual blocks, stored flat: two convolutions and two norms each.
    std::vector<torch::nn::Conv2d> block_convs;
    std::vector<torch::nn::BatchNorm2d> block_norms;

    torch::nn::Conv2d policy_conv{ nullptr };
    torch::nn::BatchNorm2d policy_norm{ nullptr };
    torch::nn::Linear policy_out{ nullptr };

    torch::nn::Conv2d value_conv{ nullptr };
    torch::nn::BatchNorm2d value_norm{ nullptr };
    torch::nn::Linear value_hidden{ nullptr };
    torch::nn::Linear value_out{ nullptr };

    torch::nn::Conv2d steps_conv{ nullptr };
    torch::nn::BatchNorm2d steps_norm{ nullptr };
    torch::nn::Linear steps_hidden{ nullptr };
    torch::nn::Linear steps_out{ nullptr };

    torch::nn::Conv2d ownership_conv{ nullptr };
    torch::nn::Conv2d death_conv{ nullptr };
    torch::nn::BatchNorm2d death_norm{ nullptr };
    torch::nn::Linear death_hidden{ nullptr };
    torch::nn::Linear death_out{ nullptr };
};

// The value type callers pass around: a shared pointer to AlphaZeroNetImpl, so a copy
// shares one network. Reach its members with -> rather than a dot.
TORCH_MODULE(AlphaZeroNet);
