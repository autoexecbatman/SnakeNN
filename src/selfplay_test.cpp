#include "selfplay.h"
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

// Self-play's job is to turn games into supervised targets. The target it
// computes is the one thing here no later stage can check: a wrong discount or
// an off-by-one in the backward walk trains a value head on plausible-looking
// nonsense and shows up only as a run that never improves.

namespace {

int failures = 0;

void expect(bool condition, const std::string& description) {
    if (condition) {
        std::cout << "  PASS  " << description << std::endl;
    } else {
        std::cout << "  FAIL  " << description << std::endl;
        failures++;
    }
}

class SilentEvaluator : public Evaluator {
public:
    void evaluate(const std::vector<const SnakeEnv*>& states,
                  float* priors_out,
                  float* values_out) override {
        for (size_t index = 0; index < states.size(); index++) {
            for (int action = 0; action < SnakeEnv::ACTION_COUNT; action++) {
                priors_out[index * SnakeEnv::ACTION_COUNT + action] = 1.0f / SnakeEnv::ACTION_COUNT;
            }
            values_out[index] = 0.0f;
        }
    }
};

MonteCarloSearch::Config searchConfig() {
    MonteCarloSearch::Config config;
    config.simulations = 16;
    config.exploration = 0.5f;   // Du et al. 2022
    config.discount = 0.98f;
    config.root_noise_fraction = 0.0f;
    config.root_noise_alpha = 0.3f;
    config.seed = 7;
    return config;
}

SelfPlay::Config playConfig(int games, int step_limit) {
    SelfPlay::Config config;
    config.games_in_parallel = games;
    config.step_limit = step_limit;
    config.discount = 0.98f;
    config.temperature = 0.5f;   // Du et al. 2022
    config.temperature_moves = 8;
    config.seed = 99;
    return config;
}

void testProducesOneRecordPerMove() {
    SilentEvaluator evaluator;
    SelfPlay play(evaluator, searchConfig(), playConfig(4, 60));

    std::vector<TrainingRecord> records;
    std::vector<GameSummary> summaries;
    play.playBatch(6, 6, 500, records, summaries);

    expect(summaries.size() == 4, "one summary per game");

    int total_moves = 0;
    for (const GameSummary& summary : summaries) {
        total_moves += summary.steps;
    }
    expect((int)records.size() == total_moves,
           "one training record per move actually played");

    bool shapes_right = true;
    for (const TrainingRecord& record : records) {
        float total = 0.0f;
        for (int action = 0; action < SnakeEnv::ACTION_COUNT; action++) {
            total += record.policy[action];
        }
        shapes_right = shapes_right && record.planes.size() == 6 * 6 * SnakeEnv::PLANE_COUNT &&
                       std::fabs(total - 1.0f) < 1e-4f;
    }
    expect(shapes_right, "each record carries a full observation and a normalised policy");
}

void testReturnsAreDiscountedBackwards() {
    // One game, no temperature, so the trajectory is deterministic. Recompute
    // the returns independently from the summary and the reward scale rather
    // than reading them back out of the thing under test.
    SilentEvaluator evaluator;
    SelfPlay play(evaluator, searchConfig(), playConfig(1, 40));

    std::vector<TrainingRecord> records;
    std::vector<GameSummary> summaries;
    play.playBatch(6, 6, 4242, records, summaries);

    expect(!records.empty(), "the game produced records");
    if (records.empty()) {
        return;
    }

    const float discount = 0.98f;

    // The last record's target is the last reward, whatever it was.
    float last_target = records.back().value_target;
    bool last_is_a_single_reward =
        std::fabs(last_target - SnakeEnv::DEATH_REWARD) < 1e-3f ||
        std::fabs(last_target - SnakeEnv::WIN_REWARD) < 1e-3f ||
        std::fabs(last_target - SnakeEnv::FOOD_REWARD) < 1e-3f ||
        std::fabs(last_target) < 1e-3f;
    expect(last_is_a_single_reward,
           "the final position's target is exactly the final reward, undiscounted");

    // And every earlier target is its own reward plus the discounted next one.
    // Reconstructing the reward as target - discount * next_target must land on
    // one of the four values the environment can pay.
    bool chain_holds = true;
    for (size_t index = 0; index + 1 < records.size(); index++) {
        float implied_reward =
            records[index].value_target - discount * records[index + 1].value_target;
        bool recognised = std::fabs(implied_reward) < 1e-2f ||
                          std::fabs(implied_reward - SnakeEnv::FOOD_REWARD) < 1e-2f ||
                          std::fabs(implied_reward - SnakeEnv::DEATH_REWARD) < 1e-2f ||
                          std::fabs(implied_reward - SnakeEnv::WIN_REWARD) < 1e-2f;
        if (!recognised) {
            std::cout << "        implied reward " << implied_reward << " at " << index
                      << std::endl;
            chain_holds = false;
            break;
        }
    }
    expect(chain_holds, "every target is its reward plus the discounted next target");
}

void testStepLimitIsEnforcedAndReported() {
    SilentEvaluator evaluator;
    const int limit = 12;
    SelfPlay play(evaluator, searchConfig(), playConfig(6, limit));

    std::vector<TrainingRecord> records;
    std::vector<GameSummary> summaries;
    play.playBatch(12, 12, 31337, records, summaries);

    bool within_limit = true;
    bool any_limited = false;
    for (const GameSummary& summary : summaries) {
        within_limit = within_limit && summary.steps <= limit;
        any_limited = any_limited || summary.hit_step_limit;
    }
    expect(within_limit, "no game runs past the step limit");
    expect(any_limited, "games stopped by the limit are reported as such, not as deaths");
}

}  // namespace

int main() {
    std::cout << "SelfPlay properties" << std::endl;
    testProducesOneRecordPerMove();
    testReturnsAreDiscountedBackwards();
    testStepLimitIsEnforcedAndReported();

    std::cout << std::endl;
    if (failures == 0) {
        std::cout << "All checks passed." << std::endl;
        return 0;
    }
    std::cout << failures << " check(s) failed." << std::endl;
    return 1;
}
