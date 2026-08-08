#include "mcts.h"
#include "snake_env.h"
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

// The search is checked against hand-written evaluators with known answers, so
// that a failure here is a failure of selection, backup or terminal handling
// rather than of a network. Nothing in this file links LibTorch.

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

// Says nothing: flat priors, zero value. Any preference the search shows under
// this evaluator comes from the simulator's own rewards and terminations.
class SilentEvaluator : public Evaluator
{
public:
    void evaluate(const std::vector<const SnakeEnv*>& states, float* priors_out,
                  float* values_out) override
    {
        calls++;
        largest_batch = std::max(largest_batch, (int)states.size());
        for (size_t index = 0; index < states.size(); index++)
        {
            for (int action = 0; action < SnakeEnv::ACTION_COUNT; action++)
            {
                priors_out[index * SnakeEnv::ACTION_COUNT + action] = 1.0f / SnakeEnv::ACTION_COUNT;
            }
            values_out[index] = 0.0f;
        }
    }

    int calls = 0;
    int largest_batch = 0;
};

MonteCarloSearch::Config testConfig(int simulations)
{
    MonteCarloSearch::Config config;
    config.simulations = simulations;
    config.exploration = 1.25f;
    config.discount = 0.99f;
    config.root_noise_fraction = 0.0f;  // deterministic while testing
    config.root_noise_alpha = 0.3f;
    config.seed = 12345;
    return config;
}

int indexOfLargest(const std::vector<float>& values)
{
    int best = 0;
    for (int index = 1; index < (int)values.size(); index++)
    {
        if (values[index] > values[best])
        {
            best = index;
        }
    }
    return best;
}

void testPolicyIsADistribution()
{
    SnakeEnv env(8, 8, 1);
    SilentEvaluator evaluator;
    MonteCarloSearch search(evaluator, testConfig(64));

    std::vector<const SnakeEnv*> roots{&env};
    auto results = search.search(roots);

    expect(results.size() == 1, "one result per root");
    float total = 0.0f;
    bool non_negative = true;
    for (float weight : results[0].policy)
    {
        total += weight;
        non_negative = non_negative && weight >= 0.0f;
    }
    expect(std::fabs(total - 1.0f) < 1e-4f, "the visit policy sums to one");
    expect(non_negative, "no negative visit weights");
    expect((int)results[0].policy.size() == SnakeEnv::ACTION_COUNT,
           "the policy covers every relative action");
}

void testAvoidsAnImmediateWall()
{
    // Drive to the right wall, then search from one step away. Continuing
    // straight is fatal; the other two actions are not. Nothing in the
    // evaluator says so - the search has to discover it from the simulator.
    SnakeEnv env(9, 9, 5);
    while (env.body()[0].x < env.width() - 2)
    {
        env.step(SnakeEnv::Action::STRAIGHT);
    }

    SilentEvaluator evaluator;
    MonteCarloSearch search(evaluator, testConfig(96));
    std::vector<const SnakeEnv*> roots{&env};
    auto results = search.search(roots);

    // Stated against the other actions rather than against a constant, so a
    // uniform policy - or an empty one - fails it. A threshold like "under 0.2"
    // is satisfied by all zeros, which is exactly what a broken search returns.
    int straight = (int)SnakeEnv::Action::STRAIGHT;
    float straight_weight = results[0].policy[straight];
    float left_weight = results[0].policy[(int)SnakeEnv::Action::LEFT];
    float right_weight = results[0].policy[(int)SnakeEnv::Action::RIGHT];
    expect(straight_weight < left_weight && straight_weight < right_weight,
           "search visits the move that walks into a wall strictly less than either alternative");
    expect(results[0].best_action != SnakeEnv::Action::STRAIGHT, "and does not choose it");
}

void testPrefersReachableFood()
{
    // Put the search one step from food and check it takes it. The reward is
    // the simulator's; the evaluator stays silent.
    SnakeEnv env(9, 9, 17);
    SilentEvaluator evaluator;
    MonteCarloSearch search(evaluator, testConfig(96));

    // Find a state where exactly one action eats, and where eating does not
    // put the head on the border.
    //
    // The border matters, and an earlier version of this test ignored it: with
    // death at -10 against +1 for food, a search that eats into a wall column
    // correctly declines - measured 2 percent of visits on the eating move with
    // the root valued at -0.67. That is the search working, not failing. The
    // claim being made here is that reward attracts visits, so the position has
    // to be one where reward is the only thing that differs.
    bool found = false;
    for (int attempt = 0; attempt < 2000 && !found; attempt++)
    {
        for (int action = 0; action < SnakeEnv::ACTION_COUNT; action++)
        {
            Position landing = env.headAfter(static_cast<SnakeEnv::Action>(action));
            bool interior = landing.x > 0 && landing.x < env.width() - 1 && landing.y > 0 &&
                            landing.y < env.height() - 1;
            if (landing == env.food() && interior)
            {
                std::vector<const SnakeEnv*> roots{&env};
                auto results = search.search(roots);
                // The eating action must be the argmax and must hold most of
                // the visits. Checking the argmax alone passes by luck one time
                // in three, which is how the stub passed this.
                expect(results[0].best_action == static_cast<SnakeEnv::Action>(action) &&
                           indexOfLargest(results[0].policy) == action &&
                           results[0].policy[action] > 0.5f,
                       "search concentrates its visits on food one move away");
                found = true;
                break;
            }
        }
        if (found)
        {
            break;
        }
        if (env.done())
        {
            env.reset();
            continue;
        }
        // Walk toward the food so this terminates.
        Position food = env.food();
        SnakeEnv::Action chosen = SnakeEnv::Action::STRAIGHT;
        int best = 1 << 30;
        for (int action = 0; action < SnakeEnv::ACTION_COUNT; action++)
        {
            Position next = env.headAfter(static_cast<SnakeEnv::Action>(action));
            int distance = std::abs(next.x - food.x) + std::abs(next.y - food.y);
            if (distance < best)
            {
                best = distance;
                chosen = static_cast<SnakeEnv::Action>(action);
            }
        }
        env.step(chosen);
    }
    expect(found, "the test reached a state with food one move away");
}

void testBatchesEveryTreeTogether()
{
    const int games = 12;
    std::vector<SnakeEnv> envs;
    for (int index = 0; index < games; index++)
    {
        envs.emplace_back(8, 8, 100 + index);
    }
    std::vector<const SnakeEnv*> roots;
    for (const SnakeEnv& env : envs)
    {
        roots.push_back(&env);
    }

    SilentEvaluator evaluator;
    const int simulations = 32;
    MonteCarloSearch search(evaluator, testConfig(simulations));
    search.search(roots);

    expect(evaluator.largest_batch == games,
           "every tree's leaf is evaluated in one call, not one call per tree");
    expect(evaluator.calls <= simulations + 1,
           "the number of forward passes is the simulation count, not simulations times games");
}

void testDeterminism()
{
    SnakeEnv env(8, 8, 77);
    SilentEvaluator first_evaluator;
    SilentEvaluator second_evaluator;
    MonteCarloSearch first(first_evaluator, testConfig(64));
    MonteCarloSearch second(second_evaluator, testConfig(64));

    std::vector<const SnakeEnv*> roots{&env};
    auto left = first.search(roots);
    auto right = second.search(roots);

    bool identical = left[0].policy.size() == right[0].policy.size();
    for (size_t index = 0; identical && index < left[0].policy.size(); index++)
    {
        identical = std::fabs(left[0].policy[index] - right[0].policy[index]) < 1e-6f;
    }
    expect(identical, "two searches on one seed and one position agree");
}

void testTerminalRootsAreRejected()
{
    SnakeEnv env(5, 5, 9);
    while (!env.done())
    {
        env.step(SnakeEnv::Action::STRAIGHT);
    }

    SilentEvaluator evaluator;
    MonteCarloSearch search(evaluator, testConfig(16));
    std::vector<const SnakeEnv*> roots{&env};

    bool threw = false;
    try
    {
        search.search(roots);
    }
    catch (const std::exception&)
    {
        threw = true;
    }
    expect(threw, "searching a finished game is refused rather than answered with noise");
}

// forcedAction decides whether the search skips a move or spends a node on it,
// and until now nothing tested it: every search property passed identically
// before and after it was rewritten, so none of them could catch a mistake in it.
//
// Rather than hand-build positions, this walks real games and checks the
// function against `wouldDie` at every position reached. `wouldDie` is the
// definition of "fatal" that the environment's own tests already pin down, so
// this is a check of forcedAction against the truth rather than against itself.
void testForcedActionAgreesWithWouldDie()
{
    // Small board so the snake traps itself often - positions where zero moves
    // survive are the case the old -1 return conflated with a free choice, and
    // they are rare on a large board.
    constexpr int BOARD = 6;
    constexpr int GAMES = 60;
    constexpr int MAX_STEPS = 400;

    int positions_seen = 0;
    int with_no_survivor = 0;
    int with_one_survivor = 0;
    int with_several_survivors = 0;
    bool disagreed = false;
    bool returned_a_fatal_move = false;

    for (unsigned int seed = 1; seed <= GAMES; seed++)
    {
        SnakeEnv game(BOARD, BOARD, seed);
        for (int step = 0; step < MAX_STEPS && !game.done(); step++)
        {
            // Count survivors directly, which is the property's definition.
            int survivors = 0;
            SnakeEnv::Action last_survivor = SnakeEnv::Action::STRAIGHT;
            for (int index = 0; index < SnakeEnv::ACTION_COUNT; index++)
            {
                const SnakeEnv::Action action = static_cast<SnakeEnv::Action>(index);
                if (!game.wouldDie(action))
                {
                    survivors++;
                    last_survivor = action;
                }
            }

            const std::optional<SnakeEnv::Action> forced = forcedAction(game);
            positions_seen++;

            if (survivors == 0)
            {
                with_no_survivor++;
                // Doomed: nothing to skip, so nothing may be returned.
                if (forced.has_value())
                {
                    disagreed = true;
                }
            }
            else if (survivors == 1)
            {
                with_one_survivor++;
                // Forced: the one survivor, and it must be that exact move.
                if (!forced.has_value() || forced.value() != last_survivor)
                {
                    disagreed = true;
                }
            }
            else
            {
                with_several_survivors++;
                // A real decision: the search must expand rather than skip.
                if (forced.has_value())
                {
                    disagreed = true;
                }
            }

            if (forced.has_value() && game.wouldDie(forced.value()))
            {
                returned_a_fatal_move = true;
            }

            if (survivors == 0)
            {
                break;
            }

            // Walk toward the food among the moves that survive. Playing badly
            // keeps the snake short, and a short snake almost never has a forced
            // move - the first version of this walked onto any survivor and
            // reached 2 forced positions in 4493, which is not coverage of the
            // case this function exists for.
            SnakeEnv::Action chosen = last_survivor;
            int best_distance = 1 << 30;
            for (int index = 0; index < SnakeEnv::ACTION_COUNT; index++)
            {
                const SnakeEnv::Action action = static_cast<SnakeEnv::Action>(index);
                if (game.wouldDie(action))
                {
                    continue;
                }
                const Position next = game.headAfter(action);
                const int distance =
                    std::abs(next.x - game.food().x) + std::abs(next.y - game.food().y);
                if (distance < best_distance)
                {
                    best_distance = distance;
                    chosen = action;
                }
            }
            game.step(chosen);
        }
    }

    expect(!disagreed, "forcedAction matches the survivor count at every position reached");
    expect(!returned_a_fatal_move, "forcedAction never returns a move that kills");

    // Coverage, so none of the above can pass because a case never arose. The
    // doomed case is the one that would otherwise go unexercised, and it is
    // exactly the case the old -1 return got wrong.
    expect(with_one_survivor > 0, "positions with exactly one survivor were reached");
    expect(with_several_survivors > 0, "positions with a real choice were reached");
    expect(with_no_survivor > 0, "positions with no survivor were reached");

    std::cout << "        " << positions_seen << " positions: " << with_no_survivor << " doomed, "
              << with_one_survivor << " forced, " << with_several_survivors << " free" << std::endl;
}

}  // namespace

int main()
{
    std::cout << "MonteCarloSearch properties" << std::endl;
    testForcedActionAgreesWithWouldDie();
    testPolicyIsADistribution();
    testAvoidsAnImmediateWall();
    testPrefersReachableFood();
    testBatchesEveryTreeTogether();
    testDeterminism();
    testTerminalRootsAreRejected();

    std::cout << std::endl;
    if (failures == 0)
    {
        std::cout << "All checks passed." << std::endl;
        return 0;
    }
    std::cout << failures << " check(s) failed." << std::endl;
    return 1;
}
