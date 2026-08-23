#include <format>
#include <iostream>
#include <string>
#include <vector>

#include "snake_env.h"

// Whether the search sees more than one food placement for the same line of play.
//
// Snake is stochastic in exactly one way: when an apple is eaten, the next one appears on
// a random empty square. A search that always imagines the same next apple is planning
// against a future it cannot actually rely on. This test measures whether it does.
//
// The mechanism it turns on. `MonteCarloSearch::search` begins every simulation with a
// copy of the root environment (`mcts.cpp`, `tree.replay.push_back(*roots[index])`). The
// copy carries the environment's generator with it, and `SnakeEnv::spawnFood` is the only
// thing that draws from that generator. So without intervention two simulations walking
// identical actions see an identical apple, every time - the search would explore one
// sampled future and mistake it for the future. `SnakeEnv::reseed` exists to break that,
// and the search calls it per simulation.
//
// What this asserts, and why both halves are needed. That a plain copy replays to the same
// apple, which is the blindness itself and would be silently restored if `reseed` were
// dropped from the search. That a reseeded copy walking the same actions reaches different
// apples, which is the property the search depends on. And a control that apple placement
// varies across seeds at all, without which the second assertion could pass on a board
// where the apple never moves and would prove nothing.
//
// Run it:
//
//     cmake --build build --config Release --target ChanceNodeTest
//     build\Release\ChanceNodeTest.exe
//
// It prints one line per property and returns non-zero if any failed:
//
//     [PASS] the constructed path eats an apple and survives  food cell after replay = 11
//     [PASS] a plain copy replays one path to the same apple  run 1 cell 11, run 2 cell 11
//     [PASS] a reseeded copy replays one path to different apples - search sees chance  12
//            streams sampled from one position
//     [PASS] apple placement does vary across seeds - the check is not vacuous  8 placements
//            sampled
//     all properties held
//
// Where this still differs from the paper. Du, Gemp, Wu and Wu 2022 branch an action whose
// transition is stochastic "into states after the random event, with each state
// representing one possible new location of the apple", explored "with equal frequency".
// Reseeding samples one placement per path instead of enumerating every empty cell. That
// is the one place the search is knowingly not exact, and this file is where a change to
// it would be caught.

namespace
{

// Board side. 10 because the paper's comparison is on 10x10 and a smaller board leaves
// too few empty cells for apple placement to vary much, which would weaken the control.
constexpr int BOARD = 10;
// The one starting position every property is read on. Fixed rather than drawn, so a
// failure names a board somebody can reconstruct.
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

// Properties that did not hold. check increments it; main prints it and returns 1 if it
// is non-zero, so the exit code says whether the test failed and the printed line says
// how many ways.
int failures = 0;

// Reports one property and counts a failure.
//
//     check(first == second, "a plain copy replays to the same apple",
//           std::format("run 1 cell {}, run 2 cell {}", first, second));
//     // prints: [PASS] a plain copy replays to the same apple  run 1 cell 11, run 2 cell 11
//
// `detail` carries the numbers the property was judged on, so a failure in a log says what
// it saw rather than only which line broke. Increments the file-local failure count, which
// main returns.
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

// The food cell reached by walking `actions` from a copy of `environment`.
//
//     const int cell = foodAfter(root, eating_path);   // 11
//     const int again = foodAfter(root, eating_path);  // 11 again - a plain copy is
//                                                      // deterministic
//
// Takes the environment by value on purpose: every call starts from the caller's state,
// which is what lets the same path be walked twice and compared. The cell is
// `y * width + x`, one number so two placements compare with ==.
//
// Returns -1 when the game ended during or at the end of the walk, since there is then no
// apple to report. The caller treats -1 as a sample to discard rather than as a placement.
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

// A sequence of actions that eats one apple, found greedily.
//
//     const std::vector<SnakeEnv::Action> path = pathThatEats(root, MAX_PATH_STEPS);
//     path.empty();   // false - an empty result means no apple was reached
//
// At each step it takes the surviving action that most reduces Manhattan distance to the
// apple, so it closes on the food without walking into a wall or itself. Stops as soon as
// the score rises, so the path ends on the move that eats - which is the move that spawns
// the next apple, and therefore the only move whose outcome is random.
//
// This is a fixture, not a policy: it needs to reach food reliably, not to play well.
// Returns whatever it walked if `max_steps` runs out without eating, which the caller
// checks for rather than assuming a path was found.
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
