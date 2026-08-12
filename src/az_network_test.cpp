#include <torch/torch.h>

#include <format>
#include <functional>
#include <iostream>
#include <string>
#include <type_traits>

#include "az_network.h"
#include "az_parameters.h"
#include "snake_env.h"

// The curriculum trains on small boards and moves up, so the property that
// matters most about this architecture is that its weights do not depend on
// board size. That is easy to break and silent when broken - a size-dependent
// head simply fails to load later, after the small-board run has finished.

// The network implementation has no value semantics: copying one would duplicate
// the module bookkeeping while sharing every parameter tensor, so two networks
// would look independent and train each other's weights. Pinned at compile time,
// because that failure produces plausible losses rather than a crash.
static_assert(!std::is_copy_constructible<AlphaZeroNetImpl>::value,
              "AlphaZeroNetImpl must not be copy constructible");
static_assert(!std::is_copy_assignable<AlphaZeroNetImpl>::value,
              "AlphaZeroNetImpl must not be copy assignable");
static_assert(!std::is_move_constructible<AlphaZeroNetImpl>::value,
              "AlphaZeroNetImpl must not be move constructible");
static_assert(!std::is_move_assignable<AlphaZeroNetImpl>::value,
              "AlphaZeroNetImpl must not be move assignable");

// The holder is the opposite case and deliberately so: it is a shared pointer,
// it is what every caller passes around, and it must stay copyable.
static_assert(std::is_copy_constructible<AlphaZeroNet>::value,
              "the AlphaZeroNet holder is the value type - it must stay copyable");

// Prediction is a plain aggregate so that structured bindings keep working and
// the two tensors travel under their own names.
static_assert(std::is_copy_constructible<Prediction>::value, "Prediction must be copyable");

namespace
{

int failures = 0;

void expect(bool condition, const std::string& description)
{
    if (condition)
    {
        std::cout << std::format("  PASS  {}\n", description);
    }
    else
    {
        std::cout << std::format("  FAIL  {}\n", description);
        failures++;
    }
}

void testOutputShapes()
{
    AlphaZeroNet network(8, 8, 32, 2);
    network->eval();
    torch::NoGradGuard no_grad;

    const int batch = 5;
    torch::Tensor input = torch::zeros({ batch, SnakeEnv::PLANE_COUNT, 8, 8 });
    auto [policy, value, steps, death_risk] = network->forward(input);

    expect(policy.sizes() == torch::IntArrayRef({ batch, SnakeEnv::ACTION_COUNT }),
           "the policy head emits one logit per relative action");
    expect(value.sizes() == torch::IntArrayRef({ batch, 1 }), "the value head emits one scalar");
    expect(value.abs().max().item<float>() <= az::VALUE_SCALE,
           "value stays inside the bounded range");

    // The bound is in reward units, not in fractions of one. MonteCarloSearch
    // adds this value to a raw edge reward, so a head bounded at 1 cannot express
    // a position worth a win or a death and the search silently discounts
    // everything past its own edges. This is the assertion whose absence let the
    // two live in different units.
    expect(az::VALUE_SCALE > SnakeEnv::WIN_REWARD,
           "the value bound can express a win, so a leaf competes with a terminal reward");
    expect(az::VALUE_SCALE > -SnakeEnv::DEATH_REWARD, "the value bound can express a death");

    // Steps-to-go is a fraction of the step budget, so anything outside [0, 1] is
    // not a duration and would be compared against the clock plane as if it were.
    expect(steps.sizes() == torch::IntArrayRef({ batch, 1 }),
           "the steps head emits one scalar per state");
    expect(steps.min().item<float>() >= 0.0f && steps.max().item<float>() <= 1.0f,
           "steps-to-go stays a fraction of the budget");
}

// Whether calling `action` throws. The checks in this network are TORCH_CHECK
// rather than assert, because a debug build cannot run at all here - LibTorch
// ships release-only libraries and a debug binary linked against them dies of an
// access violation before reaching any assertion. So these are testable, and a
// test is the only thing that can confirm they fire.
bool throwsOnCall(const std::function<void()>& action)
{
    try
    {
        action();
    }
    catch (const std::exception&)
    {
        return true;
    }
    return false;
}

void testConstructorRejectsImpossibleShapes()
{
    expect(throwsOnCall([] { AlphaZeroNet bad(1, 8, 32, 2); }),
           "a board narrower than 2 is refused");
    expect(throwsOnCall([] { AlphaZeroNet bad(8, 1, 32, 2); }),
           "a board shorter than 2 is refused");
    expect(throwsOnCall([] { AlphaZeroNet bad(8, 8, 0, 2); }),
           "a trunk with no channels is refused rather than emitting constant policies");
    expect(throwsOnCall([] { AlphaZeroNet bad(8, 8, 32, -1); }),
           "a negative block count is refused");
    expect(!throwsOnCall([] { AlphaZeroNet fine(2, 2, 1, 0); }),
           "and the smallest legitimate network is still allowed");
}

void testForwardRejectsTheWrongInput()
{
    AlphaZeroNet network(8, 8, 16, 1);
    network->eval();
    torch::NoGradGuard no_grad;

    // The pooling that makes the weights board-size independent also makes this
    // silent: a network fed the wrong board runs to completion and returns
    // confident nonsense, because nothing downstream of the pool can tell what
    // went in. That is the failure these checks exist for.
    expect(throwsOnCall([&] { network->forward(torch::zeros({ SnakeEnv::PLANE_COUNT, 8, 8 })); }),
           "an unbatched input is refused");
    expect(throwsOnCall(
               [&] { network->forward(torch::zeros({ 2, SnakeEnv::PLANE_COUNT - 1, 8, 8 })); }),
           "the wrong number of planes is refused");
    expect(
        throwsOnCall([&] { network->forward(torch::zeros({ 2, SnakeEnv::PLANE_COUNT, 20, 20 })); }),
        "a board the network was not built for is refused instead of pooled away");
    expect(
        throwsOnCall([&] { network->forward(torch::zeros({ 0, SnakeEnv::PLANE_COUNT, 8, 8 })); }),
        "an empty batch is refused");
    expect(
        !throwsOnCall([&] { network->forward(torch::zeros({ 2, SnakeEnv::PLANE_COUNT, 8, 8 })); }),
        "and a correctly shaped batch goes through");
}

void testPredictionFieldsAreTheOnesNamed()
{
    // The struct replaced a std::pair, and the risk in that swap is transposition:
    // .first and .second carried no meaning, so a caller reading them in the wrong
    // order would have compiled. The two fields have different shapes, which is
    // what makes the mix-up detectable at all.
    AlphaZeroNet network(6, 6, 16, 1);
    network->eval();
    torch::NoGradGuard no_grad;

    const int batch = 3;
    const Prediction prediction =
        network->forward(torch::rand({ batch, SnakeEnv::PLANE_COUNT, 6, 6 }));

    expect(
        prediction.policy_logits.sizes() == torch::IntArrayRef({ batch, SnakeEnv::ACTION_COUNT }),
        "policy_logits holds one logit per action, not the value");
    expect(prediction.value.sizes() == torch::IntArrayRef({ batch, 1 }),
           "value holds one scalar per state, not the policy");

    // Logits, not probabilities - the evaluator applies the softmax, so a policy
    // head that had already normalised would be normalised twice.
    const float logit_total = prediction.policy_logits.exp().sum().item<float>();
    expect(std::abs(logit_total - static_cast<float>(batch)) > 1e-3f,
           "policy_logits are unnormalised, as the name says");

    // Structured bindings still work, which is what keeps the existing call sites
    // in the trainer and the evaluator unchanged.
    auto [policy, value, steps, death_risk] =
        network->forward(torch::rand({ batch, SnakeEnv::PLANE_COUNT, 6, 6 }));
    expect(policy.sizes() == prediction.policy_logits.sizes() &&
               value.sizes() == prediction.value.sizes(),
           "a structured binding still destructures in policy-then-value order");
}

void testWeightsTransferAcrossBoardSizes()
{
    AlphaZeroNet small(6, 6, 32, 2);
    AlphaZeroNet large(20, 20, 32, 2);

    // Same architecture, different board: every parameter must line up by name
    // and by shape, or a curriculum cannot carry a trained trunk upward.
    auto small_params = small->named_parameters();
    auto large_params = large->named_parameters();

    bool shapes_match = small_params.size() == large_params.size();
    if (shapes_match)
    {
        for (const auto& item : small_params)
        {
            torch::Tensor* counterpart = large_params.find(item.key());
            if (counterpart == nullptr || counterpart->sizes() != item.value().sizes())
            {
                shapes_match = false;
                break;
            }
        }
    }
    expect(shapes_match, "every parameter has the same name and shape at 6x6 and at 20x20");

    // And the transfer works in practice, not only on paper.
    bool loaded = true;
    try
    {
        torch::NoGradGuard no_grad;
        for (const auto& item : small->named_parameters())
        {
            large->named_parameters()[item.key()].copy_(item.value());
        }
        for (const auto& item : small->named_buffers())
        {
            large->named_buffers()[item.key()].copy_(item.value());
        }
    }
    catch (const std::exception& error)
    {
        std::cout << std::format("        {}\n", error.what());
        loaded = false;
    }
    expect(loaded, "a 6x6 network's weights load into a 20x20 network");

    if (loaded)
    {
        large->eval();
        torch::NoGradGuard no_grad;
        torch::Tensor input = torch::rand({ 2, SnakeEnv::PLANE_COUNT, 20, 20 });
        auto [policy, value, steps, death_risk] = large->forward(input);
        bool finite = policy.isfinite().all().item<bool>() && value.isfinite().all().item<bool>();
        expect(finite, "the transferred network runs on the larger board and stays finite");
    }
}

// Widening the stem is what lets a checkpoint trained before the clock plane keep
// being used. Expected values come from the contract: the saved weights land in the
// leading input channels and every new one is zero.
void testWideningTheStemKeepsTheOldWeightsAndZeroesTheNew()
{
    // [out_channels, in_planes, kernel, kernel], filled so every element is
    // distinguishable from every other.
    torch::Tensor saved = torch::arange(2 * 8 * 3 * 3, torch::kFloat).reshape({ 2, 8, 3, 3 });
    torch::Tensor target = torch::ones({ 2, 9, 3, 3 }, torch::kFloat);

    torch::Tensor widened = widenStemWeight(saved, target);

    expect(widened.sizes() == target.sizes(), "the widened weight has the target's shape");
    expect(torch::equal(widened.narrow(1, 0, 8), saved),
           "the saved planes are copied into the leading channels, unchanged and in order");
    expect(torch::equal(widened.narrow(1, 8, 1), torch::zeros({ 2, 1, 3, 3 }, torch::kFloat)),
           "and the new plane is zero, so it contributes nothing until training moves it");
    // The target is not written through: it was ones, and a widening that had
    // aliased it would read back as ones.
    expect(torch::equal(target, torch::ones({ 2, 9, 3, 3 }, torch::kFloat)),
           "the target is left alone - the widened weight is a new tensor");
}

void testWideningAnEqualStemChangesNothing()
{
    torch::Tensor saved = torch::arange(2 * 9 * 3 * 3, torch::kFloat).reshape({ 2, 9, 3, 3 });
    torch::Tensor target = torch::zeros({ 2, 9, 3, 3 }, torch::kFloat);
    expect(torch::equal(widenStemWeight(saved, target), saved),
           "a stem of the same width comes back exactly as it was");
}

void testWideningRefusesWhatItCannotWiden()
{
    const auto refused = [](const torch::Tensor& saved, const torch::Tensor& target)
    {
        try
        {
            widenStemWeight(saved, target);
            return false;
        }
        catch (const std::invalid_argument&)
        {
            return true;
        }
    };

    // Wider than the network: there is nowhere to put the extra planes, and
    // dropping them silently would be a different network wearing the same name.
    expect(refused(torch::zeros({ 2, 10, 3, 3 }), torch::zeros({ 2, 9, 3, 3 })),
           "a checkpoint wider than the network is refused");
    expect(refused(torch::zeros({ 4, 8, 3, 3 }), torch::zeros({ 2, 9, 3, 3 })),
           "a different output channel count is refused");
    expect(refused(torch::zeros({ 2, 8, 5, 5 }), torch::zeros({ 2, 9, 3, 3 })),
           "a different kernel size is refused");
    expect(refused(torch::zeros({ 2, 8, 3 }), torch::zeros({ 2, 9, 3, 3 })),
           "a weight that is not four dimensional is refused");
}

}  // namespace

int main()
{
    torch::manual_seed(0);
    std::cout << std::format("AlphaZeroNet properties\n");
    testOutputShapes();
    testConstructorRejectsImpossibleShapes();
    testForwardRejectsTheWrongInput();
    testPredictionFieldsAreTheOnesNamed();
    testWeightsTransferAcrossBoardSizes();
    testWideningTheStemKeepsTheOldWeightsAndZeroesTheNew();
    testWideningAnEqualStemChangesNothing();
    testWideningRefusesWhatItCannotWiden();

    if (failures == 0)
    {
        std::cout << std::format("\nAll checks passed.\n");
        return 0;
    }
    std::cout << std::format("\n{} check(s) failed.\n", failures);
    return 1;
}
