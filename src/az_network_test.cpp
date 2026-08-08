#include "az_network.h"
#include "snake_env.h"
#include <torch/torch.h>
#include <iostream>
#include <string>

// The curriculum trains on small boards and moves up, so the property that
// matters most about this architecture is that its weights do not depend on
// board size. That is easy to break and silent when broken - a size-dependent
// head simply fails to load later, after the small-board run has finished.

namespace
{

int failures = 0;

void expect(bool condition, const std::string& description)
{
    if (condition)
    {
        std::cout << "  PASS  " << description << std::endl;
    }
    else
    {
        std::cout << "  FAIL  " << description << std::endl;
        failures++;
    }
}

void testOutputShapes()
{
    AlphaZeroNet network(8, 8, 32, 2);
    network->eval();
    torch::NoGradGuard no_grad;

    const int batch = 5;
    torch::Tensor input = torch::zeros({batch, SnakeEnv::PLANE_COUNT, 8, 8});
    auto [policy, value] = network->forward(input);

    expect(policy.sizes() == torch::IntArrayRef({batch, SnakeEnv::ACTION_COUNT}),
           "the policy head emits one logit per relative action");
    expect(value.sizes() == torch::IntArrayRef({batch, 1}), "the value head emits one scalar");
    expect(value.abs().max().item<float>() <= 1.0f, "value stays inside the bounded range");
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
        std::cout << "        " << error.what() << std::endl;
        loaded = false;
    }
    expect(loaded, "a 6x6 network's weights load into a 20x20 network");

    if (loaded)
    {
        large->eval();
        torch::NoGradGuard no_grad;
        torch::Tensor input = torch::rand({2, SnakeEnv::PLANE_COUNT, 20, 20});
        auto [policy, value] = large->forward(input);
        bool finite = policy.isfinite().all().item<bool>() && value.isfinite().all().item<bool>();
        expect(finite, "the transferred network runs on the larger board and stays finite");
    }
}

}  // namespace

int main()
{
    torch::manual_seed(0);
    std::cout << "AlphaZeroNet properties" << std::endl;
    testOutputShapes();
    testWeightsTransferAcrossBoardSizes();

    std::cout << std::endl;
    if (failures == 0)
    {
        std::cout << "All checks passed." << std::endl;
        return 0;
    }
    std::cout << failures << " check(s) failed." << std::endl;
    return 1;
}
