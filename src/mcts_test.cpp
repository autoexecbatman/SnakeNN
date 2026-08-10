#include <cmath>
#include <iostream>
#include <string>
#include <type_traits>
#include <vector>

#include "mcts.h"
#include "snake_env.h"

// The search is checked against hand-written evaluators with known answers, so
// that a failure here is a failure of selection, backup or terminal handling
// rather than of a network. Nothing in this file links LibTorch.

// A search has no value semantics: copying one would duplicate the generator so
// two searches drew the same stream, snapshot a descent that is still in flight,
// and share the evaluator by reference while each counted its own rate. Checked
// at compile time because a copy is a bug that produces plausible numbers rather
// than a crash, so no runtime test would notice it. A member added later that
// restored copyability would break this line and nothing else.
static_assert(!std::is_copy_constructible<MonteCarloSearch>::value,
              "MonteCarloSearch must not be copy constructible");
static_assert(!std::is_copy_assignable<MonteCarloSearch>::value,
              "MonteCarloSearch must not be copy assignable");
static_assert(!std::is_move_constructible<MonteCarloSearch>::value,
              "MonteCarloSearch must not be move constructible");
static_assert(!std::is_move_assignable<MonteCarloSearch>::value,
              "MonteCarloSearch must not be move assignable");
static_assert(std::is_nothrow_destructible<MonteCarloSearch>::value,
              "MonteCarloSearch must stay trivially releasable - rule of zero, nothing owned");

// The environment is the opposite case, and deliberately so: the search copies a
// root once per simulation, so copying has to stay cheap and available. Stated
// here so that removing it is a decision rather than an accident.
static_assert(std::is_copy_constructible<SnakeEnv>::value,
              "the search copies a root per simulation - SnakeEnv must stay copyable");
static_assert(std::is_copy_assignable<SnakeEnv>::value,
              "SnakeEnv must stay copy assignable - the descent assigns into its replay slot");

namespace
{

// A budget no test can reach, so the clock plane is full and every assertion
// here measures what it measured before the clock existed.
constexpr int TEST_STEP_LIMIT = 1000000;

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
    void evaluate(const std::vector<const SnakeEnv*>& states, float* priors_out, float* values_out,
                  float* steps_out) override
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
            steps_out[index] = 1.0f;
        }
    }

    int calls = 0;
    int largest_batch = 0;
};

// Says exactly one thing: these priors, and a silent value head. Lets a test
// state what the network believes and then check that selection acted on it,
// which is the half of the search no other evaluator here isolates.
class PriorEvaluator : public Evaluator
{
public:
    explicit PriorEvaluator(const std::vector<float>& priors) : priors_(priors) {}

    void evaluate(const std::vector<const SnakeEnv*>& states, float* priors_out, float* values_out,
                  float* steps_out) override
    {
        for (size_t index = 0; index < states.size(); index++)
        {
            for (int action = 0; action < SnakeEnv::ACTION_COUNT; action++)
            {
                priors_out[index * SnakeEnv::ACTION_COUNT + action] =
                    priors_[static_cast<size_t>(action)];
            }
            values_out[index] = 0.0f;
            steps_out[index] = 1.0f;
        }
    }

private:
    std::vector<float> priors_;
};

// Flat priors and a fixed value, so the return that arrives back at the root is
// an arithmetic consequence of the discount and the path length and nothing else.
// That is what makes backup's exponent checkable against a number worked out by
// hand rather than against whatever it currently produces.
class ConstantValueEvaluator : public Evaluator
{
public:
    explicit ConstantValueEvaluator(float value) : value_(value) {}

    void evaluate(const std::vector<const SnakeEnv*>& states, float* priors_out, float* values_out,
                  float* steps_out) override
    {
        for (size_t index = 0; index < states.size(); index++)
        {
            for (int action = 0; action < SnakeEnv::ACTION_COUNT; action++)
            {
                priors_out[index * SnakeEnv::ACTION_COUNT + action] =
                    1.0f / static_cast<float>(SnakeEnv::ACTION_COUNT);
            }
            values_out[index] = value_;
            steps_out[index] = 1.0f;
        }
    }

private:
    float value_;
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

// A position with room in every direction, so that selection is the only thing
// deciding anything: no wall, no forced move, no terminal, and the food far
// enough that no simulation reaches it inside the horizon. Without this the
// simulator's own rewards would be doing the work the test attributes to priors.
SnakeEnv openBoard(unsigned int seed)
{
    return SnakeEnv(20, 20, seed, TEST_STEP_LIMIT);
}

void testPriorsSteerTheSearch()
{
    // The exploration term is the only route from a prior to a visit, and nothing
    // checked that the route works. If it did not, the search would be ignoring
    // the network entirely and every training result would be search alone.
    //
    // The test has to move the prior and watch the answer move with it. A single
    // skewed vector is not enough: the first version of this asserted that a
    // 0.90/0.05/0.05 prior put the most visits on action 0, and it passed while
    // returning the identical distribution for a 0.98/0.01/0.01 prior - the
    // simulator's own dynamics were doing the work and the priors were never
    // shown to matter. Favouring each action in turn from the same position is
    // what makes the claim falsifiable: an inert prior can win at most one of
    // the three rounds.
    const int simulations = 96;
    float visits_when_favoured[SnakeEnv::ACTION_COUNT];
    int rounds_won = 0;

    for (int favoured = 0; favoured < SnakeEnv::ACTION_COUNT; favoured++)
    {
        std::vector<float> priors(SnakeEnv::ACTION_COUNT, 0.02f);
        priors[static_cast<size_t>(favoured)] = 0.96f;

        SnakeEnv env = openBoard(4242);
        PriorEvaluator evaluator(priors);
        MonteCarloSearch search(evaluator, testConfig(simulations));
        std::vector<const SnakeEnv*> roots{ &env };
        std::vector<MonteCarloSearch::Result> results = search.search(roots);

        visits_when_favoured[favoured] = results[0].policy[static_cast<size_t>(favoured)];
        if (indexOfLargest(results[0].policy) == favoured)
        {
            rounds_won++;
        }
        std::cout << "        favouring " << favoured << ": " << results[0].policy[0] << " / "
                  << results[0].policy[1] << " / " << results[0].policy[2] << std::endl;
    }

    expect(rounds_won == SnakeEnv::ACTION_COUNT,
           "whichever action the prior favours is the one that collects the most visits");

    // The same quantity read three ways must actually differ, or the loop above
    // was comparing one number to itself and the priors changed nothing.
    const bool distribution_moved = !(visits_when_favoured[0] == visits_when_favoured[1] &&
                                      visits_when_favoured[1] == visits_when_favoured[2]);
    expect(distribution_moved, "moving the prior moved the visit distribution");
}

void testNoActionIsStarved()
{
    // The (1 + visits) denominator is what makes attention decay: without it a
    // single strong prior would take every simulation and the other two branches
    // would never be looked at once. That is a silent failure - the search still
    // returns a policy and still looks confident. So an almost-degenerate prior
    // has to leave visits on the other two actions anyway.
    //
    // Checked on an open board so that the two neglected actions are neglected
    // rather than fatal; a wall would starve them for a legitimate reason and the
    // test would be measuring the simulator instead of the selection rule.
    SnakeEnv env = openBoard(909);
    std::vector<float> nearly_degenerate{ 0.98f, 0.01f, 0.01f };
    PriorEvaluator evaluator(nearly_degenerate);
    MonteCarloSearch search(evaluator, testConfig(96));

    std::vector<const SnakeEnv*> roots{ &env };
    std::vector<MonteCarloSearch::Result> results = search.search(roots);

    bool all_three_survive = true;
    for (int action = 0; action < SnakeEnv::ACTION_COUNT; action++)
    {
        if (env.wouldDie(static_cast<SnakeEnv::Action>(action)))
        {
            all_three_survive = false;
        }
    }
    expect(all_three_survive, "the test position really does leave all three actions open");
    expect(results[0].policy[1] > 0.0f && results[0].policy[2] > 0.0f,
           "even a 0.98 prior leaves visits for the other two actions");
    std::cout << "        visits " << results[0].policy[0] << " / " << results[0].policy[1] << " / "
              << results[0].policy[2] << std::endl;
}

void testDoomedPositionIsSearchedNotExpandedPastDeath()
{
    // A position where every action kills. The descent reaches such a child,
    // marks it terminal and must stop: expanding past death would ask the network
    // about a finished game and hang a subtree off a node with no successors.
    // Nothing exercised that path, so the guard in expand was assertion-only.
    //
    // The position is found rather than constructed, because building an enclosed
    // snake through the public interface takes more setup than it is worth and
    // would encode one hand-made shape instead of a real one.
    SnakeEnv game(6, 6, 20260808, TEST_STEP_LIMIT);
    bool found_doomed = false;
    int positions_walked = 0;
    unsigned int cursor = 5;

    for (int step = 0; step < 40000 && !found_doomed; step++)
    {
        if (game.done())
        {
            game.reset();
            continue;
        }
        positions_walked++;

        int survivors = 0;
        for (int action = 0; action < SnakeEnv::ACTION_COUNT; action++)
        {
            if (!game.wouldDie(static_cast<SnakeEnv::Action>(action)))
            {
                survivors++;
            }
        }
        if (survivors == 0)
        {
            found_doomed = true;
            break;
        }

        // Chase the food so the body grows and the board crowds; a random walk
        // dies at length one and never reaches a position with no way out.
        SnakeEnv::Action chosen = SnakeEnv::Action::STRAIGHT;
        int best_distance = 1 << 30;
        for (int action = 0; action < SnakeEnv::ACTION_COUNT; action++)
        {
            const SnakeEnv::Action candidate = static_cast<SnakeEnv::Action>(action);
            if (game.wouldDie(candidate))
            {
                continue;
            }
            const Position next = game.headAfter(candidate);
            const int distance =
                std::abs(next.x - game.food().x) + std::abs(next.y - game.food().y);
            if (distance < best_distance)
            {
                best_distance = distance;
                chosen = candidate;
            }
        }
        cursor = cursor * 1664525u + 1013904223u;
        if ((cursor >> 24) % 6 == 0)
        {
            chosen = static_cast<SnakeEnv::Action>((cursor >> 16) % SnakeEnv::ACTION_COUNT);
        }
        game.step(chosen);
    }

    expect(found_doomed, "the walk reached a live position where every action kills");
    if (!found_doomed)
    {
        return;
    }
    expect(!game.done(), "and that position is still live, so search will accept it");

    SilentEvaluator evaluator;
    MonteCarloSearch search(evaluator, testConfig(48));
    std::vector<const SnakeEnv*> roots{ &game };
    std::vector<MonteCarloSearch::Result> results = search.search(roots);

    float total = 0.0f;
    for (float weight : results[0].policy)
    {
        total += weight;
    }
    expect(std::fabs(total - 1.0f) < 1e-4f,
           "a doomed position still yields a policy that sums to one");
    expect(evaluator.calls > 0, "the root itself was evaluated");
    // Every child of the root is terminal here, so after the root's own expansion
    // no further expansion is owed. One evaluation call per simulation round would
    // mean the search kept asking about finished games.
    expect(evaluator.calls < 48, "and the search stopped asking about finished games");
    std::cout << "        doomed after " << positions_walked << " positions, body "
              << game.body().size() << ", evaluator calls " << evaluator.calls << std::endl;
}

// A 20x20 board whose food is far enough from the starting head that no path a
// short search can walk will reach it. Without that, an eaten apple puts a reward
// on an edge and the hand-worked arithmetic below stops applying.
// `first_seed` is where the scan starts, and it is a required argument rather than
// a default because it decides whether two calls give the same board or different
// ones - which is the difference between comparing two settings on one position and
// comparing two positions. A version of this without it returned the same board
// four times and a test that asked for four distinct games silently got one.
SnakeEnv boardWithDistantFood(int minimum_distance, unsigned int first_seed)
{
    for (unsigned int seed = first_seed; seed < first_seed + 500; seed++)
    {
        SnakeEnv candidate(20, 20, seed, TEST_STEP_LIMIT);
        const Position head = candidate.body()[0];
        const int distance =
            std::abs(candidate.food().x - head.x) + std::abs(candidate.food().y - head.y);
        if (distance >= minimum_distance)
        {
            return candidate;
        }
    }
    // Every seed in the range put the food within reach, which would silently
    // invalidate the caller's arithmetic.
    throw std::runtime_error("no seed produced a board with distant food");
}

void testBackupDiscountsByEdgeLength()
{
    // The one piece of backup's arithmetic no test reached. It is checkable
    // exactly, because with a constant value head, flat priors and no reward
    // anywhere on the path, the return arriving at the root is determined:
    //
    //   one simulation  - the leaf is the root itself, so the root's mean value is
    //                     the evaluator's value V, undiscounted.
    //   two simulations - the second descends one edge of one tick, so the root
    //                     accumulates V + gamma*V over two visits, giving
    //                     V * (1 + gamma) / 2.
    //
    // Dropping the discount would make the second case V as well, which is why the
    // two are checked together rather than only the second.
    const float value = 0.5f;
    const float discount = testConfig(1).discount;

    SnakeEnv one_step = boardWithDistantFood(6, 1);
    ConstantValueEvaluator first_evaluator(value);
    MonteCarloSearch first_search(first_evaluator, testConfig(1));
    std::vector<const SnakeEnv*> first_roots{ &one_step };
    const float undiscounted = first_search.search(first_roots)[0].value;

    SnakeEnv two_step = boardWithDistantFood(6, 1);
    ConstantValueEvaluator second_evaluator(value);
    MonteCarloSearch second_search(second_evaluator, testConfig(2));
    std::vector<const SnakeEnv*> second_roots{ &two_step };
    const float over_one_edge = second_search.search(second_roots)[0].value;

    const float expected_over_one_edge = value * (1.0f + discount) / 2.0f;

    expect(std::fabs(undiscounted - value) < 1e-6f,
           "one simulation returns the leaf value undiscounted");
    expect(std::fabs(over_one_edge - expected_over_one_edge) < 1e-6f,
           "two simulations discount the second by exactly one tick of gamma");
    // Without this the pair above would also pass for a discount of one, which is
    // the case the test exists to rule out.
    expect(std::fabs(over_one_edge - value) > 1e-4f,
           "and the discounted result is distinguishable from the undiscounted one");
    std::cout << "        value " << value << ", gamma " << discount << ": got " << undiscounted
              << " and " << over_one_edge << ", expected " << expected_over_one_edge << std::endl;
}

MonteCarloSearch::Config noisyConfig(int simulations, float fraction, unsigned int seed)
{
    MonteCarloSearch::Config config = testConfig(simulations);
    config.root_noise_fraction = fraction;
    config.seed = seed;
    return config;
}

void testRootNoiseIsAppliedAndKeepsADistribution()
{
    // Root noise had no test at all. Every config in this file and in
    // selfplay_test sets the fraction to zero, which is the function's first early
    // return - so its body ran only inside the training binary, on the one path
    // that shapes every game the agent learns from.
    //
    // Three things have to hold: the noise reaches the search, it is actually
    // random, and it leaves the priors a distribution. The last is asserted inside
    // the search; what a test can see is that the visit policy still sums to one.
    SnakeEnv quiet_board = boardWithDistantFood(6, 1);
    SilentEvaluator quiet_evaluator;
    MonteCarloSearch quiet_search(quiet_evaluator, noisyConfig(64, 0.0f, 555));
    std::vector<const SnakeEnv*> quiet_roots{ &quiet_board };
    const std::vector<float> without_noise = quiet_search.search(quiet_roots)[0].policy;

    SnakeEnv noisy_board = boardWithDistantFood(6, 1);
    SilentEvaluator noisy_evaluator;
    MonteCarloSearch noisy_search(noisy_evaluator, noisyConfig(64, 0.25f, 555));
    std::vector<const SnakeEnv*> noisy_roots{ &noisy_board };
    const std::vector<float> with_noise = noisy_search.search(noisy_roots)[0].policy;

    SnakeEnv other_board = boardWithDistantFood(6, 1);
    SilentEvaluator other_evaluator;
    MonteCarloSearch other_search(other_evaluator, noisyConfig(64, 0.25f, 98765));
    std::vector<const SnakeEnv*> other_roots{ &other_board };
    const std::vector<float> other_noise = other_search.search(other_roots)[0].policy;

    bool noise_changed_the_search = false;
    bool seed_changed_the_noise = false;
    float noisy_total = 0.0f;
    for (int action = 0; action < SnakeEnv::ACTION_COUNT; action++)
    {
        const size_t slot = static_cast<size_t>(action);
        if (std::fabs(with_noise[slot] - without_noise[slot]) > 1e-6f)
        {
            noise_changed_the_search = true;
        }
        if (std::fabs(with_noise[slot] - other_noise[slot]) > 1e-6f)
        {
            seed_changed_the_noise = true;
        }
        noisy_total += with_noise[slot];
    }

    expect(noise_changed_the_search,
           "root noise reaches the search - the same position and seed answer differently with it");
    expect(seed_changed_the_noise, "and the noise is drawn, not fixed - a new seed moves it");
    expect(std::fabs(noisy_total - 1.0f) < 1e-4f, "the policy under noise still sums to one");

    // The extreme the assertions are really about: at a fraction of one the
    // network's priors are discarded entirely and the Dirichlet weights stand
    // alone. If mixing could break the distribution, this is where it would.
    SnakeEnv pure_board = boardWithDistantFood(6, 1);
    SilentEvaluator pure_evaluator;
    MonteCarloSearch pure_search(pure_evaluator, noisyConfig(64, 1.0f, 4242));
    std::vector<const SnakeEnv*> pure_roots{ &pure_board };
    const std::vector<float> pure_noise = pure_search.search(pure_roots)[0].policy;
    float pure_total = 0.0f;
    bool pure_non_negative = true;
    for (float weight : pure_noise)
    {
        pure_total += weight;
        pure_non_negative = pure_non_negative && weight >= 0.0f;
    }
    expect(std::fabs(pure_total - 1.0f) < 1e-4f,
           "a fraction of one replaces the priors outright and still yields a distribution");
    expect(pure_non_negative, "and no visit weight goes negative");
    std::cout << "        no noise " << without_noise[0] << ", noise " << with_noise[0]
              << ", other seed " << other_noise[0] << ", pure noise " << pure_noise[0] << std::endl;
}

bool sameResult(const MonteCarloSearch::Result& left, const MonteCarloSearch::Result& right)
{
    if (left.policy.size() != right.policy.size() || left.best_action != right.best_action)
    {
        return false;
    }
    if (std::fabs(left.value - right.value) > 1e-6f)
    {
        return false;
    }
    for (size_t slot = 0; slot < left.policy.size(); slot++)
    {
        if (std::fabs(left.policy[slot] - right.policy[slot]) > 1e-6f)
        {
            return false;
        }
    }
    return true;
}

void testSearchReusesBuffersWithoutLeakingState()
{
    // The search keeps its trees and its batch buffers between calls instead of
    // rebuilding them, which is worth tens of thousands of allocations per game and
    // introduces exactly one risk: a later call reading something an earlier one
    // left behind. Nothing tested that, because nothing called search twice.
    //
    // The position makes the check possible. With the food far from the head and a
    // short search, no simulation reaches it, so no simulated apple is ever placed
    // and the per-simulation reseeding changes nothing observable. With root noise
    // off as well, the result becomes a function of the position alone - so the
    // same position searched again on the same object has to give the same answer,
    // and a stale tree or an uncleared path would show up as a difference.
    const int simulations = 16;
    SnakeEnv board = boardWithDistantFood(10, 1);

    SilentEvaluator first_evaluator;
    MonteCarloSearch reused(first_evaluator, testConfig(simulations));

    std::vector<const SnakeEnv*> one_root{ &board };
    const MonteCarloSearch::Result baseline = reused.search(one_root)[0];

    // A bigger batch in between, so the trees vector grows and the tail of it is
    // left holding a fully searched tree that the next call must not touch.
    std::vector<SnakeEnv> block;
    for (int game = 0; game < 4; game++)
    {
        // Distinct starting seeds, so these are four different positions rather
        // than one position four times.
        block.push_back(boardWithDistantFood(10, 500u + 500u * static_cast<unsigned int>(game)));
    }
    std::vector<const SnakeEnv*> four_roots;
    for (const SnakeEnv& game : block)
    {
        four_roots.push_back(&game);
    }
    const std::vector<MonteCarloSearch::Result> wide = reused.search(four_roots);

    const MonteCarloSearch::Result after_wide = reused.search(one_root)[0];

    expect(wide.size() == 4, "a wider call returns one result per root");
    expect(sameResult(baseline, after_wide),
           "a repeated search on a reused object gives the same answer as the first");

    // And the wide call itself must match what a search that had never been used
    // produces, so growth does not depend on history either.
    SilentEvaluator fresh_evaluator;
    MonteCarloSearch fresh(fresh_evaluator, testConfig(simulations));
    const std::vector<MonteCarloSearch::Result> fresh_wide = fresh.search(four_roots);
    bool wide_matches = true;
    for (size_t index = 0; index < wide.size(); index++)
    {
        if (!sameResult(wide[index], fresh_wide[index]))
        {
            wide_matches = false;
        }
    }
    expect(wide_matches, "and a reused object searches a wide batch exactly as a new one does");

    // Deliberately not asserted here: that the four results differ from each other.
    // They do not, and that is correct - every game starts with its head at the
    // centre of an empty board and the only thing distinguishing them is food the
    // search cannot reach within this horizon, so the positions are identical as
    // far as selection is concerned. The first version of this test claimed the
    // opposite and failed, which is the position telling the truth about itself.
    // Tree independence is what testBatchesEveryTreeTogether covers, on a batch
    // whose games really do diverge.

    std::vector<const SnakeEnv*> no_roots;
    expect(reused.search(no_roots).empty(), "no roots yields no results");
}

void testPolicyIsADistribution()
{
    SnakeEnv env(8, 8, 1, TEST_STEP_LIMIT);
    SilentEvaluator evaluator;
    MonteCarloSearch search(evaluator, testConfig(64));

    std::vector<const SnakeEnv*> roots{ &env };
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
    SnakeEnv env(9, 9, 5, TEST_STEP_LIMIT);
    while (env.body()[0].x < env.width() - 2)
    {
        env.step(SnakeEnv::Action::STRAIGHT);
    }

    SilentEvaluator evaluator;
    MonteCarloSearch search(evaluator, testConfig(96));
    std::vector<const SnakeEnv*> roots{ &env };
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
    SnakeEnv env(9, 9, 17, TEST_STEP_LIMIT);
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
                std::vector<const SnakeEnv*> roots{ &env };
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
        envs.emplace_back(8, 8, 100 + index, TEST_STEP_LIMIT);
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
    SnakeEnv env(8, 8, 77, TEST_STEP_LIMIT);
    SilentEvaluator first_evaluator;
    SilentEvaluator second_evaluator;
    MonteCarloSearch first(first_evaluator, testConfig(64));
    MonteCarloSearch second(second_evaluator, testConfig(64));

    std::vector<const SnakeEnv*> roots{ &env };
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
    SnakeEnv env(5, 5, 9, TEST_STEP_LIMIT);
    while (!env.done())
    {
        env.step(SnakeEnv::Action::STRAIGHT);
    }

    SilentEvaluator evaluator;
    MonteCarloSearch search(evaluator, testConfig(16));
    std::vector<const SnakeEnv*> roots{ &env };

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
        SnakeEnv game(BOARD, BOARD, seed, TEST_STEP_LIMIT);
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
    testPriorsSteerTheSearch();
    testNoActionIsStarved();
    testDoomedPositionIsSearchedNotExpandedPastDeath();
    testBackupDiscountsByEdgeLength();
    testRootNoiseIsAppliedAndKeepsADistribution();
    testSearchReusesBuffersWithoutLeakingState();
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
