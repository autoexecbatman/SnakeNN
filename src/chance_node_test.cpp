#include <format>
#include <iostream>
#include <string>
#include <vector>

#include "snake_env.h"

// Does the search ever see more than one food placement for the same path?
//
// `MonteCarloSearch::search` starts every simulation with a copy of the root
// environment (mcts.cpp, `tree.replay.push_back(*roots[index])`). A copy takes
// the environment's `std::mt19937` with it, and `SnakeEnv::spawnFood` is the
// only thing that draws from it. So two simulations that walk the same actions
// see the same apple, every time.
//
// Du, Gemp, Wu and Wu 2022 do the opposite: an action whose transition is
// stochastic "branches into states after the random event, with each state
// representing one possible new location of the apple", explored "with equal
// frequency". This test pins down what we actually do, so that implementing
// chance branching flips it rather than passing silently.

namespace
{

constexpr int BOARD = 10;
constexpr unsigned int SEED = 12345;
// Long enough that a greedy walk to the apple always finds one.
constexpr int MAX_PATH_STEPS = 200;
// The paper's cap for a 10x10 board. It bounds the clock plane, not termination,
// and no walk here comes near it - what this test measures is apple placement.
constexpr int STEP_LIMIT = 1200;
// Streams sampled when checking that a reseeded copy diverges.
constexpr unsigned int RESEED_STREAMS = 12;
// Seeds sampled for the control that apple placement varies at all.
constexpr unsigned int CONTROL_SEEDS = 8;

int failures = 0;

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

// Walks the given actions, stopping early if the game ends. Returns the food
// cell the environment holds afterwards, or -1 if the game finished.
int foodAfter(SnakeEnv environment, const std::vector<SnakeEnv::Action>& actions)
{
    for (SnakeEnv::Action action : actions)
    {
        if (environment.done())
        {
            return -1;
        }
        environment.step(action);
    }
    if (environment.done())
    {
        return -1;
    }
    return environment.food().y * environment.width() + environment.food().x;
}

// Actions that reach the food, chosen by walking towards it and turning only
// when the straight move would kill. Enough to guarantee an apple is eaten.
std::vector<SnakeEnv::Action> pathThatEats(SnakeEnv environment, int max_steps)
{
    std::vector<SnakeEnv::Action> actions;
    int starting_score = environment.score();
    for (int step = 0; step < max_steps; step++)
    {
        if (environment.done() || environment.score() > starting_score)
        {
            break;
        }
        SnakeEnv::Action chosen = SnakeEnv::Action::STRAIGHT;
        // Prefer a move that closes on the food and does not kill.
        int best_distance = 1 << 30;
        for (int candidate = 0; candidate < SnakeEnv::ACTION_COUNT; candidate++)
        {
            SnakeEnv::Action action = static_cast<SnakeEnv::Action>(candidate);
            if (environment.wouldDie(action))
            {
                continue;
            }
            Position next = environment.headAfter(action);
            int distance =
                std::abs(next.x - environment.food().x) + std::abs(next.y - environment.food().y);
            if (distance < best_distance)
            {
                best_distance = distance;
                chosen = action;
            }
        }
        actions.push_back(chosen);
        environment.step(chosen);
    }
    return actions;
}

}  // namespace

int main()
{
    SnakeEnv root(BOARD, BOARD, SEED, STEP_LIMIT);
    // Move off the opening position so the test is not a special case of it.
    for (int step = 0; step < 3 && !root.done(); step++)
    {
        if (!root.wouldDie(SnakeEnv::Action::STRAIGHT))
        {
            root.step(SnakeEnv::Action::STRAIGHT);
        }
        else
        {
            root.step(SnakeEnv::Action::LEFT);
        }
    }

    std::vector<SnakeEnv::Action> eating_path = pathThatEats(root, MAX_PATH_STEPS);
    if (eating_path.empty())
    {
        std::cout << "[FAIL] setup produced no path to the food" << std::endl;
        return 1;
    }

    // Two simulations, as the search runs them: each from a fresh copy of the
    // root, walking identical actions.
    int first = foodAfter(root, eating_path);
    int second = foodAfter(root, eating_path);

    check(first >= 0, "the constructed path eats an apple and survives",
          std::format("food cell after replay = {}", first));

    // A plain copy replays deterministically. This is the property that made the
    // search blind to chance, and it is why `reseed` exists rather than being a
    // convenience.
    check(first == second, "a plain copy replays one path to the same apple",
          std::format("run 1 cell {}, run 2 cell {}", first, second));

    // And the fix: a reseeded copy walking the same actions reaches different
    // apples, which is what lets the search average over chance instead of
    // planning against one known outcome.
    std::vector<int> reseeded;
    for (unsigned int stream = 1; stream <= RESEED_STREAMS; stream++)
    {
        SnakeEnv copy = root;
        copy.reseed(stream);
        int cell = foodAfter(copy, eating_path);
        if (cell >= 0)
        {
            reseeded.push_back(cell);
        }
    }
    bool reseeded_varies = false;
    for (size_t index = 1; index < reseeded.size(); index++)
    {
        if (reseeded[index] != reseeded[0])
        {
            reseeded_varies = true;
            break;
        }
    }
    check(reseeded_varies,
          "a reseeded copy replays one path to different apples - search sees chance",
          std::format("{} streams sampled from one position", reseeded.size()));

    // Control: without it, the assertion above could hold simply because apple
    // placement never varies at all, and the test would prove nothing.
    std::vector<int> placements;
    for (unsigned int other_seed = SEED; other_seed < SEED + CONTROL_SEEDS; other_seed++)
    {
        SnakeEnv other(BOARD, BOARD, other_seed, STEP_LIMIT);
        std::vector<SnakeEnv::Action> path = pathThatEats(other, MAX_PATH_STEPS);
        int cell = foodAfter(other, path);
        if (cell >= 0)
        {
            placements.push_back(cell);
        }
    }
    bool varies = false;
    for (size_t index = 1; index < placements.size(); index++)
    {
        if (placements[index] != placements[0])
        {
            varies = true;
            break;
        }
    }
    check(varies, "apple placement does vary across seeds - the check is not vacuous",
          std::format("{} placements sampled", placements.size()));

    if (failures == 0)
    {
        std::cout << "all properties held" << std::endl;
        return 0;
    }
    std::cout << failures << " properties failed" << std::endl;
    return 1;
}
