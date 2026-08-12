#pragma once

#include <torch/torch.h>

#include <string>
#include <vector>

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
};

// A stem weight widened to accept more input planes, with the new ones zeroed.
//
// Shaped like `target`, holding `saved` in its leading input channels. Zero in the
// rest is what makes the widened convolution compute exactly what the narrower one
// did until training moves them.
//
// Throws std::invalid_argument unless `saved` has at most as many input channels as
// `target` and agrees with it in every other dimension. Free rather than a member
// because it is arithmetic on two tensors and needs no network to be tested.
torch::Tensor widenStemWeight(const torch::Tensor& saved, const torch::Tensor& target);

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

    // Widens the stem to accept more input planes than the checkpoint was trained
    // with, copying the old weights and zeroing the new channels.
    //
    // The clock plane took PLANE_COUNT from 8 to 9, which changes the stem's input
    // channel count, so torch::load rejects every checkpoint trained before it.
    // Zeroed new channels mean the widened network computes exactly what the old
    // one did on the same position - the clock starts ignored and is learned from
    // there, which is a fine-tune rather than a retrain from nothing.
    //
    // Throws if the checkpoint is wider than this network or differs anywhere but
    // the stem's input channels.
    //
    // Returns the names of parameters and buffers the checkpoint did not contain;
    // those keep the values they were constructed with. That is what lets a head
    // added after a checkpoint was written start from its initialization instead
    // of refusing the whole file. The names are returned rather than swallowed
    // because a mistyped module name looks exactly like a new head from here, and
    // only the caller printing the list makes the difference visible.
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

    torch::nn::Conv2d death_conv{ nullptr };
    torch::nn::BatchNorm2d death_norm{ nullptr };
    torch::nn::Linear death_hidden{ nullptr };
    torch::nn::Linear death_out{ nullptr };
};

TORCH_MODULE(AlphaZeroNet);
