#include <torch/torch.h>

#include <algorithm>
#include <format>
#include <iostream>
#include <string>
#include <vector>

#include "az_network.h"
#include "snake_env.h"

// Can the network see one cell?
//
// The heads pool the trunk to a fixed 4x4 before flattening, which is what makes
// every weight independent of board size and lets a 6x6 network load at 20x20.
// On a 10x10 board that pool averages about six cells into each output value,
// and the question this asks is whether a change to a single cell survives it.
//
// It matters because the agent's remaining failure is efficiency: it fills the
// board but not inside the step limit, and routing a long snake through a nearly
// full board is exactly the task that needs precise geometry. If a one-cell
// change cannot reach the policy head, no amount of training fixes that.
//
// Property 3 needs a trained checkpoint and is skipped without one, because an
// untrained network has no reason to prefer any action and asserting that it
// does would be a test that fails for the wrong reason.

namespace
{

// Board side. 10 rather than 20 because the pool averages about six cells into each
// output value here, which is the harder case for a single cell to survive.
constexpr int BOARD = 10;
// Dense enough to be an endgame, with room left to route through.
constexpr int DENSE_LENGTH = 70;
// Short enough that the head can still turn, so the preference has somewhere to
// move to. Chosen by the geometry, not by the result.
constexpr int OPEN_LENGTH = 12;
// Trunk width and depth, matching the networks this project actually trains, so the answer
// is about the architecture in use rather than a smaller stand-in.
constexpr int TRUNK_CHANNELS = 64;
constexpr int TRUNK_BLOCKS = 4;

// Properties that did not hold. main prints the count and returns 1 when it is non-zero.
int failures = 0;

// Reports one property and counts a failure.
//
//     check(different > 0, "one cell reaches the policy", "12 of 3 outputs moved");
//
// `detail` carries the measurement, because "the cell was visible" is worth little without
// how visible.
void check(bool condition, const std::string& name, const std::string& detail)
{
    if (condition)
    {
        std::cout << "[PASS] " << name << "  " << detail << std::endl;
    }
    else
    {
        std::cout << "[FAIL] " << name << "  " << detail << std::endl;
        failures++;
    }
}

// Cells in serpentine order: row 0 left to right, row 1 right to left, and so
// on. Consecutive entries are always orthogonally adjacent, so any prefix of
// this order is a legal snake body.
std::vector<int> serpentineOrder(int width, int height)
{
    std::vector<int> order;
    order.reserve(width * height);
    for (int row = 0; row < height; row++)
    {
        for (int step = 0; step < width; step++)
        {
            int column = (row % 2 == 0) ? step : (width - 1 - step);
            order.push_back(row * width + column);
        }
    }
    return order;
}

unsigned char headingBetween(int from_cell, int to_cell, int width)
{
    int from_x = from_cell % width;
    int from_y = from_cell / width;
    int to_x = to_cell % width;
    int to_y = to_cell / width;
    if (to_x == from_x + 1)
    {
        return static_cast<unsigned char>(Direction::RIGHT);
    }
    if (to_x == from_x - 1)
    {
        return static_cast<unsigned char>(Direction::LEFT);
    }
    if (to_y == from_y + 1)
    {
        return static_cast<unsigned char>(Direction::DOWN);
    }
    if (to_y == from_y - 1)
    {
        return static_cast<unsigned char>(Direction::UP);
    }
    throw std::logic_error("cells are not adjacent");
}

// How many of the three relative actions do not end the game immediately.
//
// The probe needs this because a constant preferred action is only evidence of
// blindness when there was a choice to make. A dense position can leave exactly
// one survivable move, and then a policy that always picks it is right rather
// than blind - a confounded position of precisely this kind has already made one
// search test in this repository report a failure that was not there.
int survivableActionCount(int width, int height, const SnakeEnv::Snapshot& snapshot)
{
    std::vector<bool> occupied(static_cast<size_t>(width) * height, false);
    // The tail vacates as the head advances, so it does not block.
    for (size_t index = 0; index + 1 < snapshot.body_cells.size(); index++)
    {
        occupied[snapshot.body_cells[index]] = true;
    }

    int head_x = snapshot.body_cells[0] % width;
    int head_y = snapshot.body_cells[0] / width;

    const int deltas[4][2] = { { 0, -1 }, { 0, 1 }, { -1, 0 }, { 1, 0 } };  // UP DOWN LEFT RIGHT
    const int heading = static_cast<int>(snapshot.heading);
    // STRAIGHT, then the two turns, expressed as headings.
    const int left_of[4] = { static_cast<int>(Direction::LEFT), static_cast<int>(Direction::RIGHT),
                             static_cast<int>(Direction::DOWN), static_cast<int>(Direction::UP) };
    const int right_of[4] = { static_cast<int>(Direction::RIGHT), static_cast<int>(Direction::LEFT),
                              static_cast<int>(Direction::UP), static_cast<int>(Direction::DOWN) };
    int candidates[3] = { heading, left_of[heading], right_of[heading] };

    int survivable = 0;
    for (int index = 0; index < 3; index++)
    {
        int next_x = head_x + deltas[candidates[index]][0];
        int next_y = head_y + deltas[candidates[index]][1];
        if (next_x < 0 || next_x >= width || next_y < 0 || next_y >= height)
        {
            continue;
        }
        if (occupied[next_y * width + next_x])
        {
            continue;
        }
        survivable++;
    }
    return survivable;
}

// A dense endgame position: the snake occupies a serpentine prefix of the board,
// head at the leading end, with the food in one of the cells it has not reached.
SnakeEnv::Snapshot denseSnapshot(int width, int height, int length, int food_cell)
{
    std::vector<int> order = serpentineOrder(width, height);
    SnakeEnv::Snapshot snapshot;
    snapshot.body_cells.reserve(length);
    for (int index = length - 1; index >= 0; index--)
    {
        snapshot.body_cells.push_back(static_cast<unsigned short>(order[index]));
    }
    snapshot.food_cell = static_cast<unsigned short>(food_cell);
    snapshot.heading = headingBetween(order[length - 2], order[length - 1], width);
    snapshot.won = false;
    return snapshot;
}

std::vector<float> encode(int width, int height, const SnakeEnv::Snapshot& snapshot)
{
    std::vector<float> planes(static_cast<size_t>(SnakeEnv::PLANE_COUNT) * width * height);
    SnakeEnv::encodeSnapshot(width, height, snapshot, planes.data());
    return planes;
}

// How many elements of two equally sized vectors differ at all.
//
//     countDifferences({ 0.1f, 0.2f }, { 0.1f, 0.9f });   // 1
//
// Exact inequality rather than a tolerance: the question is whether a one-cell change
// reaches the output at all, so any movement counts and a threshold would answer a
// different question.
int countDifferences(const std::vector<float>& left, const std::vector<float>& right)
{
    int different = 0;
    for (size_t index = 0; index < left.size(); index++)
    {
        if (left[index] != right[index])
        {
            different++;
        }
    }
    return different;
}

// Takes the planes by non-const reference on purpose: `torch::from_blob` wants a
// mutable pointer, and the alternative was casting away const at the call, which
// hides the fact that the tensor aliases this buffer until the clone.
torch::Tensor toBatch(std::vector<float>& planes, int width, int height)
{
    return torch::from_blob(planes.data(), { 1, SnakeEnv::PLANE_COUNT, height, width },
                            torch::kFloat)
        .clone();
}

}  // namespace

int main(int argc, char** argv)
{
    const std::vector<int> order = serpentineOrder(BOARD, BOARD);

    // Two positions differing only in where the food sits, one cell apart.
    const int food_a = order[DENSE_LENGTH + 5];
    const int food_b = order[DENSE_LENGTH + 6];
    // A third, unrelated position: shorter snake, food far away. This is the
    // scale against which a one-cell change is judged - without it, "the outputs
    // differ" has no size to be compared to.
    const int food_far = order[BOARD * BOARD - 1];

    SnakeEnv::Snapshot snapshot_a = denseSnapshot(BOARD, BOARD, DENSE_LENGTH, food_a);
    SnakeEnv::Snapshot snapshot_b = denseSnapshot(BOARD, BOARD, DENSE_LENGTH, food_b);
    SnakeEnv::Snapshot snapshot_c = denseSnapshot(BOARD, BOARD, DENSE_LENGTH / 2, food_far);

    std::vector<float> planes_a = encode(BOARD, BOARD, snapshot_a);
    std::vector<float> planes_b = encode(BOARD, BOARD, snapshot_b);
    std::vector<float> planes_c = encode(BOARD, BOARD, snapshot_c);

    // Property 1: the input distinguishes them, and by exactly the amount it
    // should - the food leaves one cell and arrives at another, nothing else.
    int differences = countDifferences(planes_a, planes_b);
    check(differences == 2, "one-cell change alters exactly two encoded values",
          std::format("differing entries {}, expected 2", differences));

    torch::Device device = torch::kCPU;
    AlphaZeroNet network(BOARD, BOARD, TRUNK_CHANNELS, TRUNK_BLOCKS);
    if (argc > 1)
    {
        try
        {
            torch::load(network, std::string(argv[1]));
            std::cout << "loaded checkpoint " << argv[1] << std::endl;
        }
        catch (const std::exception& error)
        {
            std::cout << "[FAIL] could not load checkpoint: " << error.what() << std::endl;
            return 1;
        }
    }
    else
    {
        std::cout << "no checkpoint given - running architecture properties only" << std::endl;
    }
    network->to(device);
    network->eval();

    torch::NoGradGuard no_grad;
    auto forward = [&](std::vector<float>& planes)
    { return network->forward(toBatch(planes, BOARD, BOARD)); };

    Prediction out_a = forward(planes_a);
    Prediction out_b = forward(planes_b);
    Prediction out_c = forward(planes_c);

    double near_distance = (out_a.policy_logits - out_b.policy_logits).norm().item<double>();
    double far_distance = (out_a.policy_logits - out_c.policy_logits).norm().item<double>();

    // Property 2: the one-cell change reaches the policy head at all. The
    // failure this is looking for is a hard zero - two boards that the pooling
    // has made indistinguishable. The ratio is reported rather than asserted,
    // because there is no principled threshold for how large it ought to be.
    check(near_distance > 0.0, "a one-cell change reaches the policy head",
          std::format("|d policy| = {:.6f}, unrelated board gives {:.6f}, ratio {:.6f}",
                      near_distance, far_distance,
                      far_distance > 0.0 ? near_distance / far_distance : 0.0));

    double value_shift = std::abs(out_a.value.item<double>() - out_b.value.item<double>());
    check(value_shift > 0.0, "a one-cell change reaches the value head",
          std::format("|d value| = {:.6f}", value_shift));

    // Property 3: for a trained network, which action is preferred has to depend
    // on where the food is. A policy whose argmax never moves as the food sweeps
    // the board cannot navigate to it, whatever its loss says.
    if (argc > 1)
    {
        // Only a position with a real choice can test this. A dense serpentine
        // often leaves one legal move, and a constant argmax there is correct.
        int survivable = survivableActionCount(BOARD, BOARD, snapshot_a);
        std::cout << "survivable actions in the dense position: " << survivable << std::endl;

        int open_survivable = survivableActionCount(
            BOARD, BOARD, denseSnapshot(BOARD, BOARD, OPEN_LENGTH, order[OPEN_LENGTH + 3]));
        std::cout << "survivable actions in the open position:  " << open_survivable << std::endl;

        std::vector<int> preferred;
        int placements = 0;
        for (size_t index = OPEN_LENGTH; index < order.size(); index++)
        {
            SnakeEnv::Snapshot sweep = denseSnapshot(BOARD, BOARD, OPEN_LENGTH, order[index]);
            std::vector<float> planes = encode(BOARD, BOARD, sweep);
            torch::Tensor logits = forward(planes).policy_logits;
            preferred.push_back(static_cast<int>(logits.argmax(1).item<int64_t>()));
            placements++;
        }
        std::sort(preferred.begin(), preferred.end());
        int distinct =
            static_cast<int>(std::unique(preferred.begin(), preferred.end()) - preferred.begin());

        if (open_survivable < 2)
        {
            std::cout << "[SKIP] preferred action test - the position has no choice to make"
                      << std::endl;
        }
        else
        {
            check(distinct >= 2, "preferred action depends on where the food is",
                  std::format("{} distinct actions over {} food placements, {} survivable",
                              distinct, placements, open_survivable));
        }
    }

    if (failures == 0)
    {
        std::cout << "all properties held" << std::endl;
        return 0;
    }
    std::cout << failures << " properties failed" << std::endl;
    return 1;
}
