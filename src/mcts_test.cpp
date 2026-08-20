#include <cmath>
#include <format>
#include <iostream>
#include <string>
#include <type_traits>
#include <vector>

#include "mcts.h"
#include "snake_env.h"

// Checked against hand-written evaluators, so a failure here is one of selection,
// backup or terminal handling. Nothing in this file links LibTorch.

// A copy would share a stream, a half-finished descent and one evaluator counter.
// Checked at compile time; the bug produces plausible numbers rather than a crash.
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

// The opposite case, deliberately: the search copies a root once per simulation,
// so copying stays cheap and available.
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
                  float* steps_out, float* death_risk_out) override
    {
        calls++;
        largest_batch = std::max(largest_batch, (int)states.size());
        for (size_t index = 0; index < states.size(); index++)
        {
            for (int action = 0; action < SnakeEnv::ACTION_COUNT; action++)
            {
                priors_out[index * SnakeEnv::ACTION_COUNT + action] = 1.0f / SnakeEnv::ACTION_COUNT;
                death_risk_out[index * SnakeEnv::ACTION_COUNT + action] = 0.0f;
            }
            values_out[index] = 0.0f;
            steps_out[index] = 1.0f;
        }
    }

    int calls = 0;
    int largest_batch = 0;
};

// These priors and a silent value head, so a test can check that selection acted
// on what the network believes.
class PriorEvaluator : public Evaluator
{
public:
    explicit PriorEvaluator(const std::vector<float>& priors) : priors_(priors) {}

    void evaluate(const std::vector<const SnakeEnv*>& states, float* priors_out, float* values_out,
                  float* steps_out, float* death_risk_out) override
    {
        for (size_t index = 0; index < states.size(); index++)
        {
            for (int action = 0; action < SnakeEnv::ACTION_COUNT; action++)
            {
                priors_out[index * SnakeEnv::ACTION_COUNT + action] =
                    priors_[static_cast<size_t>(action)];
                death_risk_out[index * SnakeEnv::ACTION_COUNT + action] = 0.0f;
            }
            values_out[index] = 0.0f;
            steps_out[index] = 1.0f;
        }
    }

private:
    std::vector<float> priors_;
};

// Flat priors and a fixed value, so the root's return follows from the discount and
// the path length alone - which is what makes backup's exponent checkable by hand.
class ConstantValueEvaluator : public Evaluator
{
public:
    explicit ConstantValueEvaluator(float value) : value_(value) {}

    void evaluate(const std::vector<const SnakeEnv*>& states, float* priors_out, float* values_out,
                  float* steps_out, float* death_risk_out) override
    {
        for (size_t index = 0; index < states.size(); index++)
        {
            for (int action = 0; action < SnakeEnv::ACTION_COUNT; action++)
            {
                priors_out[index * SnakeEnv::ACTION_COUNT + action] =
                    1.0f / static_cast<float>(SnakeEnv::ACTION_COUNT);
                death_risk_out[index * SnakeEnv::ACTION_COUNT + action] = 0.0f;
            }
            values_out[index] = value_;
            steps_out[index] = 1.0f;
        }
    }

private:
    float value_;
};

// One death risk per action and silence elsewhere, so a test can say "this action
// is doomed" and check only what the cap did about it.
class RiskEvaluator : public Evaluator
{
public:
    explicit RiskEvaluator(const std::vector<float>& death_risks) : death_risks_(death_risks) {}

    void evaluate(const std::vector<const SnakeEnv*>& states, float* priors_out, float* values_out,
                  float* steps_out, float* death_risk_out) override
    {
        for (size_t index = 0; index < states.size(); index++)
        {
            for (int action = 0; action < SnakeEnv::ACTION_COUNT; action++)
            {
                priors_out[index * SnakeEnv::ACTION_COUNT + action] =
                    1.0f / static_cast<float>(SnakeEnv::ACTION_COUNT);
                death_risk_out[index * SnakeEnv::ACTION_COUNT + action] =
                    death_risks_[static_cast<size_t>(action)];
            }
            values_out[index] = 0.0f;
            steps_out[index] = 1.0f;
        }
    }

private:
    std::vector<float> death_risks_;
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

// Room in every direction and food out of reach, so selection is the only thing
// deciding - otherwise the simulator's rewards do the work attributed to priors.
SnakeEnv openBoard(unsigned int seed)
{
    return SnakeEnv(20, 20, seed, TEST_STEP_LIMIT);
}

void testPriorsSteerTheSearch()
{
    // Favours each action in turn from one position, which is what makes the claim
    // falsifiable: an inert prior can win at most one of the three rounds.
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
    // Without the (1 + visits) decay one strong prior takes every simulation, and the
    // search still looks confident. Open board, so neglect is not a wall's doing.
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
    // Every action kills, so the descent must stop rather than expand past death.
    // The position is found rather than built, so it is a real shape.
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
    // Every child terminal, so a call per simulation means it asked about a finished
    // game.
    expect(evaluator.calls < 48, "and the search stopped asking about finished games");
    std::cout << "        doomed after " << positions_walked << " positions, body "
              << game.body().size() << ", evaluator calls " << evaluator.calls << std::endl;
}

// A 20x20 board whose food no short search reaches, so no edge carries a reward.
// first_seed is required because it decides whether two calls give one board or two.
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
    // With a constant value head and no rewards the root's return is determined: V
    // after one simulation, V * (1 + gamma) / 2 after two. Dropping the discount
    // makes the second V as well, so the two are checked together.
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

// Values scaled by a constant, so two searches differ in nothing but the size of the
// numbers they compare. The prior is uniform and the values vary with the position, so
// the range has a width to normalise against.
class ScaledValueEvaluator : public Evaluator
{
public:
    explicit ScaledValueEvaluator(float scale) : scale_(scale) {}

    void evaluate(const std::vector<const SnakeEnv*>& states, float* priors_out, float* values_out,
                  float* steps_out, float* death_risk_out) override
    {
        for (size_t index = 0; index < states.size(); index++)
        {
            for (int action = 0; action < SnakeEnv::ACTION_COUNT; action++)
            {
                priors_out[index * SnakeEnv::ACTION_COUNT + action] =
                    1.0f / static_cast<float>(SnakeEnv::ACTION_COUNT);
                death_risk_out[index * SnakeEnv::ACTION_COUNT + action] = 0.0f;
            }
            // Varies with the position, so leaves disagree and the range has a width.
            const float spread = 0.2f * static_cast<float>(states[index]->steps() % 7);
            values_out[index] = scale_ * (0.4f + spread);
            steps_out[index] = 1.0f;
        }
    }

private:
    float scale_;
};

std::vector<float> policyUnderScale(float scale, bool normalize)
{
    SnakeEnv board = boardWithDistantFood(6, 1);
    ScaledValueEvaluator evaluator(scale);
    MonteCarloSearch::Config config = testConfig(200);
    config.normalize_values = normalize;
    MonteCarloSearch search(evaluator, config);
    const std::vector<const SnakeEnv*> roots{ &board };
    return search.search(roots).front().policy;
}

float largestPolicyGap(const std::vector<float>& left, const std::vector<float>& right)
{
    float largest = 0.0f;
    for (size_t index = 0; index < left.size(); index++)
    {
        largest = std::max(largest, std::abs(left[index] - right[index]));
    }
    return largest;
}

// What normalising buys, stated as the property it is for: c_puct compares a value
// against a prior term, so the visit distribution must not depend on how large the values
// happen to be. Raw values here are bounded by az::VALUE_SCALE at 40 while the largest
// exploration term a prior can produce is single digits, which is how a change to the
// value scale silently changed how much the search explored.
void testNormalisingMakesSelectionScaleInvariant()
{
    // Unnormalised, a tenfold change in the value scale is a tenfold change in what
    // exploitation is worth against an unchanged prior term, so the search explores
    // differently. This is the falsifier: without it the assertion below passes for a
    // search that ignores values altogether.
    const float raw_gap =
        largestPolicyGap(policyUnderScale(1.0f, false), policyUnderScale(10.0f, false));
    if (raw_gap <= 0.01f)
    {
        std::cout << std::format(
            "[FAIL] unnormalised selection ignored a tenfold value scale: gap {:.6f}\n", raw_gap);
        failures++;
    }

    // Normalised, the dependence has to shrink - but it cannot vanish, and expecting it
    // to was wrong. An action is worth its edge reward plus its discounted value, and
    // scaling the network's output scales only the second term: a +1 apple and a -10
    // death keep their size. So the sum being normalised is a different mixture at each
    // scale, and exact invariance is unavailable by construction rather than by defect.
    const float normalised_gap =
        largestPolicyGap(policyUnderScale(1.0f, true), policyUnderScale(10.0f, true));
    if (normalised_gap >= raw_gap)
    {
        std::cout << std::format(
            "[FAIL] normalising did not reduce the value scale's influence: raw {:.6f}, "
            "normalised {:.6f}\n",
            raw_gap, normalised_gap);
        failures++;
    }
}

// A prior that puts almost everything on one action, which is what a long-trained policy
// emits: measured on az10_death368, 46 percent of positions have a top prior above 0.999.
class SaturatedPriorEvaluator : public Evaluator
{
public:
    void evaluate(const std::vector<const SnakeEnv*>& states, float* priors_out, float* values_out,
                  float* steps_out, float* death_risk_out) override
    {
        for (size_t index = 0; index < states.size(); index++)
        {
            for (int action = 0; action < SnakeEnv::ACTION_COUNT; action++)
            {
                priors_out[index * SnakeEnv::ACTION_COUNT + action] = 0.0005f;
                death_risk_out[index * SnakeEnv::ACTION_COUNT + action] = 0.0f;
            }
            priors_out[index * SnakeEnv::ACTION_COUNT] = 0.999f;
            // Varies with depth, so the tree produces a width to normalise against even
            // when one root action takes every visit.
            values_out[index] = 4.0f + 0.5f * static_cast<float>(states[index]->steps() % 9);
            steps_out[index] = 1.0f;
        }
    }
};

// The defect this pins: a range built only from one node's children has no width until two
// of them are visited, and under a saturated prior two never are - so normalising would be
// switched on and do nothing anywhere in the tree. Built from every node the backup
// touches instead, the range has a width from the first descent and reaches the deep
// selections.
//
// It is the backed-up root value that has to move, not the root policy. Normalising is
// monotonic, so it cannot reorder the visited children, and an unvisited child still
// scores zero against a normalised best of about one while its exploration term is
// thousandths. The root policy is therefore (1, 0, 0) either way, by construction -
// normalising does not rescue a saturated prior and asserting that it does was wrong.
// What it can reach is which descendants get explored, and that changes what comes back.
void testNormalisingReachesASaturatedRoot()
{
    SnakeEnv board = boardWithDistantFood(6, 1);

    SaturatedPriorEvaluator plain_evaluator;
    MonteCarloSearch::Config plain = testConfig(200);
    plain.normalize_values = false;
    MonteCarloSearch plain_search(plain_evaluator, plain);
    const std::vector<const SnakeEnv*> plain_roots{ &board };
    const MonteCarloSearch::Result without = plain_search.search(plain_roots).front();

    SaturatedPriorEvaluator scaled_evaluator;
    MonteCarloSearch::Config normalised = testConfig(200);
    normalised.normalize_values = true;
    MonteCarloSearch normalised_search(scaled_evaluator, normalised);
    const std::vector<const SnakeEnv*> normalised_roots{ &board };
    const MonteCarloSearch::Result with = normalised_search.search(normalised_roots).front();

    // The root policy cannot move, and that is what the comment above is about.
    if (largestPolicyGap(without.policy, with.policy) != 0.0f)
    {
        std::cout << "[FAIL] the root policy moved under a saturated prior, which "
                     "normalising cannot do - something else changed\n";
        failures++;
    }

    // The backed-up value must move: an established range changes which descendants the
    // descent picks. If the range never establishes, the two searches are identical and
    // these agree exactly - which is the inert wiring this replaced.
    if (without.value == with.value)
    {
        std::cout << std::format(
            "[FAIL] normalising reached nothing under a saturated prior - the range never "
            "established, so the switch is inert where it matters: value {:.6f}\n",
            with.value);
        failures++;
    }
}

void testRootNoiseIsAppliedAndKeepsADistribution()
{
    // Every other config sets the fraction to zero, so this body ran only inside the
    // training binary. Noise must arrive, be random, and leave a distribution.
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

    // At a fraction of one the priors are discarded and only the draws stand.
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
    // Trees are kept between calls, so the risk is a later call reading what an
    // earlier one left. Unreachable food and no noise make the result a function of
    // the position alone, so a stale tree shows up as a difference.
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

    // Not asserted: that the four differ. They do not, and correctly so - the only
    // thing separating them is food out of reach. Divergence is tested elsewhere.

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
    // Straight is fatal and the evaluator does not say so; the search must find it.
    SnakeEnv env(9, 9, 5, TEST_STEP_LIMIT);
    while (env.body()[0].x < env.width() - 2)
    {
        env.step(SnakeEnv::Action::STRAIGHT);
    }

    SilentEvaluator evaluator;
    MonteCarloSearch search(evaluator, testConfig(96));
    std::vector<const SnakeEnv*> roots{ &env };
    auto results = search.search(roots);

    // Against the other actions, not a constant: all-zeros satisfies a threshold.
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

    // Exactly one action eats, and not into the border - at -10 for death a search
    // correctly declines to eat against a wall, which would confound the claim.
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
                // Argmax alone passes by luck one time in three; the share does not.
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

// Walks real games and checks forcedAction against wouldDie at every position,
// which is the definition of fatal the environment's own tests already pin down.
void testForcedActionAgreesWithWouldDie()
{
    // Small board, so the zero-survivor case arises often enough to cover.
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

            // Toward the food, since a short snake almost never has a forced move.
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

    // Coverage, so nothing above passes because its case never arose.
    expect(with_one_survivor > 0, "positions with exactly one survivor were reached");
    expect(with_several_survivors > 0, "positions with a real choice were reached");
    expect(with_no_survivor > 0, "positions with no survivor were reached");

    std::cout << "        " << positions_seen << " positions: " << with_no_survivor << " doomed, "
              << with_one_survivor << " forced, " << with_several_survivors << " free" << std::endl;
}

}  // namespace

// Walks greedily until eating is one move away, so a search spends its depth past
// the first apple. Stops early rather than looping if the walk stalls.
SnakeEnv rootBesideTheFood(int board, unsigned int seed, int step_limit)
{
    SnakeEnv environment(board, board, seed, step_limit);
    for (int step = 0; step < board * board && !environment.done(); step++)
    {
        const Position food = environment.food();
        const Position head = environment.body()[0];
        if (std::abs(head.x - food.x) + std::abs(head.y - food.y) <= 1)
        {
            break;
        }
        int best_distance = 1 << 30;
        SnakeEnv::Action chosen = SnakeEnv::Action::STRAIGHT;
        for (int candidate = 0; candidate < SnakeEnv::ACTION_COUNT; candidate++)
        {
            const SnakeEnv::Action action = static_cast<SnakeEnv::Action>(candidate);
            if (environment.wouldDie(action))
            {
                continue;
            }
            const Position next = environment.headAfter(action);
            const int distance = std::abs(next.x - food.x) + std::abs(next.y - food.y);
            if (distance < best_distance)
            {
                best_distance = distance;
                chosen = action;
            }
        }
        environment.step(chosen);
    }
    return environment;
}

// Disagreement can only appear past a first apple, where the respawn varies, so
// the search must run deep enough on a small enough board to eat twice.
void testAliasProbeCountsWhatItClaims()
{
    SilentEvaluator evaluator;

    // Off unless asked. A probe that counts regardless would make every existing
    // run pay for it and would report numbers nobody chose to measure.
    {
        MonteCarloSearch::Config config = testConfig(200);
        MonteCarloSearch search(evaluator, config);
        SnakeEnv root(6, 6, 4242, 400);
        std::vector<const SnakeEnv*> roots{ &root };
        search.search(roots);
        expect(search.revisitedEdges() == 0 && search.aliasedEdges() == 0,
               std::format("the probe is silent unless alias_report is set - revisited {}, "
                           "aliased {}",
                           search.revisitedEdges(), search.aliasedEdges()));
    }

    // One simulation cannot reach any node twice, so nothing is revisited. This is
    // what separates a real counter from one incremented on every traversal.
    {
        MonteCarloSearch::Config config = testConfig(1);
        config.alias_report = true;
        MonteCarloSearch search(evaluator, config);
        SnakeEnv root(6, 6, 4242, 400);
        std::vector<const SnakeEnv*> roots{ &root };
        search.search(roots);
        expect(
            search.revisitedEdges() == 0,
            std::format("one simulation revisits no edge - revisited {}", search.revisitedEdges()));
    }

    // Past a first apple, the only place two computations of an edge can differ.
    {
        MonteCarloSearch::Config config = testConfig(600);
        config.alias_report = true;
        MonteCarloSearch search(evaluator, config);
        SnakeEnv root = rootBesideTheFood(6, 4242, 400);
        std::vector<const SnakeEnv*> roots{ &root };
        search.search(roots);

        expect(search.revisitedEdges() > 0,
               std::format("the probe sees revisited edges at all - revisited {}",
                           search.revisitedEdges()));
        expect(search.aliasedEdges() <= search.revisitedEdges(),
               std::format("aliased edges are a subset of revisited ones - aliased {} of {}",
                           search.aliasedEdges(), search.revisitedEdges()));
        // Without this the checks above hold on a probe that can never fire.
        expect(
            search.aliasedEdges() > 0,
            std::format("the probe fires - simulations disagree about an edge - aliased {} of {}",
                        search.aliasedEdges(), search.revisitedEdges()));

        expect(search.materiallyAliasedEdges() <= search.aliasedEdges(),
               std::format("material disagreements are a subset of all of them - {} of {}",
                           search.materiallyAliasedEdges(), search.aliasedEdges()));
        // Deduced, not measured: an edge where one simulation ate differs by nearly a
        // whole apple, so a zero here means the threshold is wrong.
        expect(search.materiallyAliasedEdges() > 0,
               std::format("some disagreement is worth more than half an apple - {} of {}",
                           search.materiallyAliasedEdges(), search.aliasedEdges()));

        // Per node, so a node counted twice breaks this before a ratio looks wrong.
        expect(search.revisitedNodes() > 0 && search.revisitedNodes() <= search.revisitedEdges(),
               std::format("revisited nodes are counted once each - {} nodes over {} traversals",
                           search.revisitedNodes(), search.revisitedEdges()));
        expect(search.aliasedNodes() > 0 && search.aliasedNodes() <= search.revisitedNodes(),
               std::format("aliased nodes are a subset of revisited ones - {} of {}",
                           search.aliasedNodes(), search.revisitedNodes()));
    }

    // Silent unless asked, for the new counters too. Checked separately from the
    // block above because that one predates them and would pass without them.
    {
        MonteCarloSearch::Config config = testConfig(600);
        MonteCarloSearch search(evaluator, config);
        SnakeEnv root = rootBesideTheFood(6, 4242, 400);
        std::vector<const SnakeEnv*> roots{ &root };
        search.search(roots);
        expect(search.materiallyAliasedEdges() == 0 && search.revisitedNodes() == 0 &&
                   search.aliasedNodes() == 0,
               std::format("the sizing counters are silent unless alias_report is set - "
                           "material {}, nodes {}/{}",
                           search.materiallyAliasedEdges(), search.aliasedNodes(),
                           search.revisitedNodes()));
    }
}

// Where every simulation finds the same game there is nothing to average and the
// search must be unchanged; where they find different games it must not be.
void testAveragedEdgesAreAnExpectationOverWhatTheNodeStandsFor()
{
    SilentEvaluator evaluator;

    // Every simulation replays the identical game, so averaging changes nothing. The
    // probe must report no disagreement, or this is measuring something else.
    {
        MonteCarloSearch::Config baseline = testConfig(80);
        baseline.alias_report = true;
        MonteCarloSearch unaveraged(evaluator, baseline);
        SnakeEnv root(6, 6, 4242, 400);
        std::vector<const SnakeEnv*> roots{ &root };
        const MonteCarloSearch::Result before = unaveraged.search(roots)[0];
        expect(unaveraged.aliasedEdges() == 0,
               std::format("no simulation disagrees here, so averaging has nothing to do - "
                           "aliased {} of {}",
                           unaveraged.aliasedEdges(), unaveraged.revisitedEdges()));

        MonteCarloSearch::Config averaged = baseline;
        averaged.average_edges = true;
        MonteCarloSearch search(evaluator, averaged);
        const MonteCarloSearch::Result after = search.search(roots)[0];

        expect(before.best_action == after.best_action,
               "averaging changes no move where every simulation agrees");
        bool policy_matches = before.policy.size() == after.policy.size();
        for (size_t action = 0; action < before.policy.size() && policy_matches; action++)
        {
            policy_matches = std::abs(before.policy[action] - after.policy[action]) < 1e-6f;
        }
        expect(policy_matches, "and it changes no visit count there either");
        expect(
            std::abs(before.value - after.value) < 1e-6f,
            std::format("nor the root value - {:.6f} against {:.6f}", before.value, after.value));
    }

    // Here the mean differs from the last write, so an implementation that stores the
    // sums and goes on reading the last value passes everything above and fails this.
    {
        MonteCarloSearch::Config baseline = testConfig(600);
        baseline.alias_report = true;
        MonteCarloSearch unaveraged(evaluator, baseline);
        SnakeEnv root = rootBesideTheFood(6, 4242, 400);
        std::vector<const SnakeEnv*> roots{ &root };
        const MonteCarloSearch::Result before = unaveraged.search(roots)[0];
        expect(unaveraged.aliasedEdges() > 0,
               std::format("simulations do disagree here - aliased {} of {}",
                           unaveraged.aliasedEdges(), unaveraged.revisitedEdges()));

        MonteCarloSearch::Config averaged = baseline;
        averaged.average_edges = true;
        MonteCarloSearch search(evaluator, averaged);
        const MonteCarloSearch::Result after = search.search(roots)[0];

        // On the root value, not the visits: one action holds 98 percent of them, so
        // the counts are too coarse to register a change deep in the tree.
        expect(std::abs(before.value - after.value) > 1e-4f,
               std::format("averaging moves the root value where the simulations disagree, so "
                           "selection reads the mean rather than the last write - {:.6f} against "
                           "{:.6f}",
                           before.value, after.value));
    }
}

// Chases food on a small board until no action survives. Growing the body is what
// crowds the board; a random walk dies at length one and never gets there.
bool walkToDoomedPosition(SnakeEnv& game)
{
    for (int step = 0; step < 40000; step++)
    {
        if (game.done())
        {
            game.reset();
            continue;
        }

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
            return true;
        }

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
        game.step(chosen);
    }
    return false;
}

// Drives into the last column, where straight leaves the board next tick. It must
// be width - 1: from width - 2 straight survives and a risk of zero is correct.
SnakeEnv beforeTheWall()
{
    SnakeEnv env(9, 9, 5, TEST_STEP_LIMIT);
    while (env.body()[0].x < env.width() - 1)
    {
        env.step(SnakeEnv::Action::STRAIGHT);
    }
    return env;
}

void testDeathRiskIsBackedUpAsUnavoidability()
{
    // The evaluator calls everything safe, so a search that forwarded its number
    // would report zero and only the simulator's terminations can raise this.
    SnakeEnv env = beforeTheWall();
    SilentEvaluator evaluator;
    MonteCarloSearch search(evaluator, testConfig(96));
    std::vector<const SnakeEnv*> roots{ &env };
    auto results = search.search(roots);

    const bool reported =
        results[0].death_risk.size() == static_cast<size_t>(SnakeEnv::ACTION_COUNT);
    expect(reported, "the search reports one death risk per root action");
    if (!reported)
    {
        // Without this the assertions below index an empty vector and take the
        // process down, which stops the rest of the red output being read.
        return;
    }

    const float straight = results[0].death_risk[(int)SnakeEnv::Action::STRAIGHT];
    const float left = results[0].death_risk[(int)SnakeEnv::Action::LEFT];
    const float right = results[0].death_risk[(int)SnakeEnv::Action::RIGHT];

    expect(std::abs(straight - 1.0f) < 1e-6f,
           std::format("walking into the wall is certain death, so its risk is exactly 1 - got "
                       "{:.6f}",
                       straight));
    // Against the fatal action, not a constant: all-zeros satisfies "below 1".
    expect(left < straight && right < straight,
           std::format("and the two survivable actions are strictly below it - {:.6f} and {:.6f} "
                       "against {:.6f}",
                       left, right, straight));
}

void testRiskClimbsFromBelowRatherThanStayingAtTheLeafItWasWrittenAt()
{
    // Answers 0 at the root and 1 below it, so the children start at 0 and only a
    // minimum taken one ply deeper can raise them. Without the refresh they stay 0.
    class DepthRiskEvaluator : public Evaluator
    {
    public:
        void evaluate(const std::vector<const SnakeEnv*>& states, float* priors_out,
                      float* values_out, float* steps_out, float* death_risk_out) override
        {
            for (size_t index = 0; index < states.size(); index++)
            {
                const float risk = states[index]->stepsSinceFood() == 0 ? 0.0f : 1.0f;
                for (int action = 0; action < SnakeEnv::ACTION_COUNT; action++)
                {
                    priors_out[index * SnakeEnv::ACTION_COUNT + action] =
                        1.0f / static_cast<float>(SnakeEnv::ACTION_COUNT);
                    death_risk_out[index * SnakeEnv::ACTION_COUNT + action] = risk;
                }
                values_out[index] = 0.0f;
                steps_out[index] = 1.0f;
            }
        }
    };

    SnakeEnv env(9, 9, 11, TEST_STEP_LIMIT);
    expect(env.stepsSinceFood() == 0, "the root is a position the evaluator scores as risk 0");

    DepthRiskEvaluator evaluator;
    MonteCarloSearch search(evaluator, testConfig(64));
    std::vector<const SnakeEnv*> roots{ &env };
    auto results = search.search(roots);

    // Every root action reads 1 only if the value climbed from a ply deeper. The
    // number written into these nodes at expansion was 0.
    for (int action = 0; action < SnakeEnv::ACTION_COUNT; action++)
    {
        expect(std::abs(results[0].death_risk[action] - 1.0f) < 1e-6f,
               std::format("root action {} carries the risk backed up from below, not the 0 "
                           "written at expansion - got {:.6f}",
                           action, results[0].death_risk[action]));
    }
}

void testAllActionsVisitedIsFalseWhereTheSearchDidNotGo()
{
    // One simulation leaves the policy uniform, so reading coverage off the policy
    // would call an untouched root fully explored and label it safe.
    {
        SnakeEnv env(9, 9, 11, TEST_STEP_LIMIT);
        SilentEvaluator evaluator;
        MonteCarloSearch search(evaluator, testConfig(1));
        std::vector<const SnakeEnv*> roots{ &env };
        auto results = search.search(roots);

        float total = 0.0f;
        for (float weight : results[0].policy)
        {
            total += weight;
        }
        expect(
            std::abs(total - 1.0f) < 1e-4f,
            std::format("with one simulation the policy is still a distribution - {:.6f}", total));
        expect(!results[0].all_actions_visited,
               "and the search reports that it visited nothing, which the uniform policy hides");
    }

    // A search with enough simulations does visit all three, so the flag is not
    // simply always false.
    {
        SnakeEnv env(9, 9, 11, TEST_STEP_LIMIT);
        SilentEvaluator evaluator;
        MonteCarloSearch search(evaluator, testConfig(64));
        std::vector<const SnakeEnv*> roots{ &env };
        auto results = search.search(roots);
        expect(results[0].all_actions_visited, "and reports true once every root action was tried");
    }

    // A refused action has its visits zeroed, so a capped root is excluded from
    // the training labels by the same test.
    {
        SnakeEnv env = beforeTheWall();
        SilentEvaluator evaluator;
        MonteCarloSearch::Config config = testConfig(64);
        config.death_cap = true;
        config.death_cap_threshold = 0.5f;
        MonteCarloSearch search(evaluator, config);
        std::vector<const SnakeEnv*> roots{ &env };
        auto results = search.search(roots);
        expect(!results[0].all_actions_visited,
               "a root where the cap refused an action does not count as fully searched");
    }
}

void testCapDoesNotRefuseAtTheThresholdItself()
{
    // Exactly on the threshold, where >= refuses all three - an off-by-one no other
    // test can see, since every risk elsewhere is exactly 0 or 1.
    SnakeEnv env(9, 9, 11, TEST_STEP_LIMIT);
    RiskEvaluator evaluator(std::vector<float>{ 0.5f, 0.5f, 0.5f });
    MonteCarloSearch::Config config = testConfig(8);
    config.death_cap = true;
    config.death_cap_threshold = 0.5f;
    MonteCarloSearch search(evaluator, config);
    std::vector<const SnakeEnv*> roots{ &env };
    auto results = search.search(roots);

    expect(std::abs(results[0].death_risk[0] - 0.5f) < 1e-6f,
           std::format("the root's risk sits exactly on the threshold - {:.6f}",
                       results[0].death_risk[0]));
    expect(search.deathCapFires() == 0,
           std::format("an action level with the threshold is not refused - fired {} times",
                       search.deathCapFires()));
}

void testDeathCapRefusesOnlyWhenAnAlternativeSurvives()
{
    // Off, the cap must be invisible. This is the control: without it, the count
    // below could be satisfied by a filter that fires unconditionally.
    {
        SnakeEnv env = beforeTheWall();
        SilentEvaluator evaluator;
        MonteCarloSearch search(evaluator, testConfig(96));
        std::vector<const SnakeEnv*> roots{ &env };
        search.search(roots);
        expect(search.deathCapFires() == 0, "with the cap off nothing is refused");
    }

    {
        SnakeEnv env = beforeTheWall();
        SilentEvaluator evaluator;
        MonteCarloSearch::Config config = testConfig(96);
        config.death_cap = true;
        config.death_cap_threshold = 0.5f;
        MonteCarloSearch search(evaluator, config);
        std::vector<const SnakeEnv*> roots{ &env };
        auto results = search.search(roots);

        expect(search.deathCapFires() >= 1,
               std::format("the cap refuses the action that walks into the wall - fired {} times",
                           search.deathCapFires()));
        expect(results[0].policy[(int)SnakeEnv::Action::STRAIGHT] == 0.0f,
               "a refused action takes no share of the policy at all");
        expect(results[0].best_action != SnakeEnv::Action::STRAIGHT, "and is not chosen");
    }

    // All doomed: refusing here would leave nothing to play, which is what turned
    // the trap guard into the endgame policy.
    {
        SnakeEnv env(6, 6, 20260808, TEST_STEP_LIMIT);
        const bool found = walkToDoomedPosition(env);
        expect(found, "a position where every action kills was reached");
        if (found)
        {
            SilentEvaluator evaluator;
            MonteCarloSearch::Config config = testConfig(96);
            config.death_cap = true;
            config.death_cap_threshold = 0.5f;
            MonteCarloSearch search(evaluator, config);
            std::vector<const SnakeEnv*> roots{ &env };
            auto results = search.search(roots);

            expect(search.deathCapFires() == 0,
                   std::format("with no survivable action the cap refuses nothing - fired {} times",
                               search.deathCapFires()));
            float total = 0.0f;
            for (float weight : results[0].policy)
            {
                total += weight;
            }
            expect(std::abs(total - 1.0f) < 1e-4f,
                   std::format("and the policy is still a distribution - sums to {:.6f}", total));
        }
    }
}

int main()
{
    std::cout << "MonteCarloSearch properties" << std::endl;
    testDeathRiskIsBackedUpAsUnavoidability();
    testRiskClimbsFromBelowRatherThanStayingAtTheLeafItWasWrittenAt();
    testAllActionsVisitedIsFalseWhereTheSearchDidNotGo();
    testCapDoesNotRefuseAtTheThresholdItself();
    testDeathCapRefusesOnlyWhenAnAlternativeSurvives();
    testAliasProbeCountsWhatItClaims();
    testAveragedEdgesAreAnExpectationOverWhatTheNodeStandsFor();
    testForcedActionAgreesWithWouldDie();
    testPolicyIsADistribution();
    testPriorsSteerTheSearch();
    testNoActionIsStarved();
    testDoomedPositionIsSearchedNotExpandedPastDeath();
    testBackupDiscountsByEdgeLength();
    testNormalisingMakesSelectionScaleInvariant();
    testNormalisingReachesASaturatedRoot();
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
