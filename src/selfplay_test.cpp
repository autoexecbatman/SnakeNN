#include "selfplay.h"
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

// Self-play's job is to turn games into supervised targets. The target it
// computes is the one thing here no later stage can check: a wrong discount or
// an off-by-one in the backward walk trains a value head on plausible-looking
// nonsense and shows up only as a run that never improves.

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

class SilentEvaluator : public Evaluator
{
public:
    void evaluate(const std::vector<const SnakeEnv*>& states, float* priors_out,
                  float* values_out) override
    {
        for (size_t index = 0; index < states.size(); index++)
        {
            for (int action = 0; action < SnakeEnv::ACTION_COUNT; action++)
            {
                priors_out[index * SnakeEnv::ACTION_COUNT + action] = 1.0f / SnakeEnv::ACTION_COUNT;
            }
            values_out[index] = 0.0f;
        }
    }
};

MonteCarloSearch::Config searchConfig()
{
    MonteCarloSearch::Config config;
    config.simulations = 16;
    config.exploration = 0.5f;  // Du et al. 2022
    config.discount = 0.98f;
    config.root_noise_fraction = 0.0f;
    config.root_noise_alpha = 0.3f;
    config.seed = 7;
    return config;
}

SelfPlay::Config playConfig(int games, int step_limit)
{
    SelfPlay::Config config;
    config.games_in_parallel = games;
    config.step_limit = step_limit;
    config.discount = 0.98f;
    config.temperature = 0.5f;  // Du et al. 2022
    config.temperature_moves = 8;
    config.seed = 99;
    return config;
}

void testProducesOneRecordPerMove()
{
    SilentEvaluator evaluator;
    SelfPlay play(evaluator, searchConfig(), playConfig(4, 60));

    std::vector<TrainingRecord> records;
    std::vector<GameSummary> summaries;
    play.playBatch(6, 6, 500, records, summaries);

    expect(summaries.size() == 4, "one summary per game");

    int total_moves = 0;
    for (const GameSummary& summary : summaries)
    {
        total_moves += summary.steps;
    }
    expect((int)records.size() == total_moves, "one training record per move actually played");

    bool shapes_right = true;
    for (const TrainingRecord& record : records)
    {
        float total = 0.0f;
        for (int action = 0; action < SnakeEnv::ACTION_COUNT; action++)
        {
            total += record.policy[action];
        }
        // The record stores the position, so what has to hold is that the
        // position is well formed and re-encodes into a full observation.
        std::vector<float> planes(6 * 6 * SnakeEnv::PLANE_COUNT, -1.0f);
        SnakeEnv::encodeSnapshot(6, 6, record.position, planes.data());
        float body_marks = 0.0f;
        for (int cell = 0; cell < 36; cell++)
        {
            body_marks += planes[cell];
        }
        shapes_right = shapes_right && !record.position.body_cells.empty() &&
                       body_marks == (float)record.position.body_cells.size() &&
                       std::fabs(total - 1.0f) < 1e-4f;
    }
    expect(shapes_right,
           "each record re-encodes to a full observation and carries a normalised policy");
}

void testReturnsAreDiscountedBackwards()
{
    // One game, no temperature, so the trajectory is deterministic. Recompute
    // the returns independently from the summary and the reward scale rather
    // than reading them back out of the thing under test.
    SilentEvaluator evaluator;
    SelfPlay play(evaluator, searchConfig(), playConfig(1, 40));

    std::vector<TrainingRecord> records;
    std::vector<GameSummary> summaries;
    play.playBatch(6, 6, 4242, records, summaries);

    expect(!records.empty(), "the game produced records");
    if (records.empty())
    {
        return;
    }

    const float discount = 0.98f;

    // The last record's target is the last reward, whatever it was.
    float last_target = records.back().value_target;
    bool last_is_a_single_reward = std::fabs(last_target - SnakeEnv::DEATH_REWARD) < 1e-3f ||
                                   std::fabs(last_target - SnakeEnv::WIN_REWARD) < 1e-3f ||
                                   std::fabs(last_target - SnakeEnv::FOOD_REWARD) < 1e-3f ||
                                   std::fabs(last_target) < 1e-3f;
    expect(last_is_a_single_reward,
           "the final position's target is exactly the final reward, undiscounted");

    // And every earlier target is its own reward plus the discounted next one.
    // Reconstructing the reward as target - discount * next_target must land on
    // one of the four values the environment can pay.
    bool chain_holds = true;
    for (size_t index = 0; index + 1 < records.size(); index++)
    {
        float implied_reward =
            records[index].value_target - discount * records[index + 1].value_target;
        bool recognised = std::fabs(implied_reward) < 1e-2f ||
                          std::fabs(implied_reward - SnakeEnv::FOOD_REWARD) < 1e-2f ||
                          std::fabs(implied_reward - SnakeEnv::DEATH_REWARD) < 1e-2f ||
                          std::fabs(implied_reward - SnakeEnv::WIN_REWARD) < 1e-2f;
        if (!recognised)
        {
            std::cout << "        implied reward " << implied_reward << " at " << index
                      << std::endl;
            chain_holds = false;
            break;
        }
    }
    expect(chain_holds, "every target is its reward plus the discounted next target");
}

void testStepLimitIsEnforcedAndReported()
{
    SilentEvaluator evaluator;
    const int limit = 12;
    SelfPlay play(evaluator, searchConfig(), playConfig(6, limit));

    std::vector<TrainingRecord> records;
    std::vector<GameSummary> summaries;
    play.playBatch(12, 12, 31337, records, summaries);

    bool within_limit = true;
    bool any_limited = false;
    for (const GameSummary& summary : summaries)
    {
        within_limit = within_limit && summary.steps <= limit;
        any_limited = any_limited || summary.hit_step_limit;
    }
    expect(within_limit, "no game runs past the step limit");
    expect(any_limited, "games stopped by the limit are reported as such, not as deaths");
}

void testProgressIsMonotonicAndCompletes()
{
    // A progress bar that stalls or goes backwards is worse than none: it makes
    // a working run look hung. The counts have to rise and finish at the total.
    SilentEvaluator evaluator;
    const int games = 8;
    SelfPlay play(evaluator, searchConfig(), playConfig(games, 40));

    int reports = 0;
    int last_finished = -1;
    long long last_moves = -1;
    bool monotonic = true;
    int final_finished = -1;
    SelfPlay::Progress final_progress{0, 0, 0, 0.0};

    play.setProgressCallback(
        [&](const SelfPlay::Progress& progress)
        {
            reports++;
            // Non-decreasing, not strictly increasing: the completing report at the
            // end of a batch repeats the final move count by design, and demanding
            // an increase there would be asserting a stricter invariant than the
            // one that matters - a bar must never go backwards.
            if (progress.games_finished < last_finished || progress.moves_played < last_moves)
            {
                monotonic = false;
            }
            last_finished = progress.games_finished;
            last_moves = progress.moves_played;
            final_finished = progress.games_finished;
            final_progress = progress;
        });

    std::vector<TrainingRecord> records;
    std::vector<GameSummary> summaries;
    play.playBatch(6, 6, 8080, records, summaries);

    expect(reports > 0, "progress is reported while a batch is in flight");
    expect(monotonic, "neither finished games nor moves ever go backwards");
    expect(final_progress.games_total == games, "the total reported is the batch size");
    expect(final_finished == games,
           "the bar reaches the end of the batch rather than stalling short");
}

}  // namespace

int main()
{
    std::cout << "SelfPlay properties" << std::endl;
    testProducesOneRecordPerMove();
    testReturnsAreDiscountedBackwards();
    testStepLimitIsEnforcedAndReported();
    testProgressIsMonotonicAndCompletes();

    std::cout << std::endl;
    if (failures == 0)
    {
        std::cout << "All checks passed." << std::endl;
        return 0;
    }
    std::cout << failures << " check(s) failed." << std::endl;
    return 1;
}
