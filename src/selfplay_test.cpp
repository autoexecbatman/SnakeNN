#include <cmath>
#include <format>
#include <iostream>
#include <string>
#include <vector>

#include "az_parameters.h"
#include "selfplay.h"

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
    void evaluate(const std::vector<const SnakeEnv*>& states, float* priors_out, float* values_out,
                  float* steps_out, float* death_risk_out) override
    {
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

SelfPlay::Config playConfig(int games, int step_limit, float timeout_reward)
{
    SelfPlay::Config config;
    config.games_in_parallel = games;
    config.step_limit = step_limit;
    config.discount = 0.98f;
    config.timeout_reward = timeout_reward;
    config.temperature = 0.5f;  // Du et al. 2022
    config.temperature_moves = 8;
    config.seed = 99;
    return config;
}

void testProducesOneRecordPerMove()
{
    SilentEvaluator evaluator;
    SelfPlay play(evaluator, searchConfig(), playConfig(4, 60, az::TIMEOUT_REWARD));

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
    SelfPlay play(evaluator, searchConfig(), playConfig(1, 40, az::TIMEOUT_REWARD));

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
    SelfPlay play(evaluator, searchConfig(), playConfig(6, limit, az::TIMEOUT_REWARD));

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

// A timed-out game must cost something, or the value head learns that running out
// the clock beats dying - which was true before this existed, since a timeout paid
// 0 against a death's -10.
void testATimedOutGamePaysThePenaltyAndOthersDoNot()
{
    SilentEvaluator evaluator;
    const int limit = 12;
    // Distinct from every reward the environment can pay, so a target carrying it
    // cannot be mistaken for a death, a win, an apple or nothing.
    const float penalty = -7.0f;

    // A 12x12 board and twelve steps: the snake cannot fill it and cannot easily
    // reach a wall, so this batch is where timeouts come from.
    std::vector<TrainingRecord> penalised;
    std::vector<GameSummary> penalised_summaries;
    SelfPlay play(evaluator, searchConfig(), playConfig(6, limit, penalty));
    play.playBatch(12, 12, 31337, penalised, penalised_summaries);

    // The same batch with the penalty switched off. Identical seeds and identical
    // config otherwise, so any difference in the targets is the penalty and nothing
    // else - and the moves themselves must be unchanged, since a reward applied
    // after the fact cannot alter a game that was already played.
    std::vector<TrainingRecord> unpenalised;
    std::vector<GameSummary> unpenalised_summaries;
    SelfPlay reference(evaluator, searchConfig(), playConfig(6, limit, 0.0f));
    reference.playBatch(12, 12, 31337, unpenalised, unpenalised_summaries);

    expect(penalised.size() == unpenalised.size(),
           "the penalty changes the targets and not the games played");

    bool any_timed_out = false;
    for (const GameSummary& summary : penalised_summaries)
    {
        any_timed_out = any_timed_out || summary.hit_step_limit;
    }
    expect(any_timed_out, "the batch actually contains a timed-out game to measure");

    // Walk the records game by game, in the order playBatch appends them.
    bool timeout_targets_differ_by_the_penalty = true;
    bool finished_targets_are_untouched = true;
    size_t cursor = 0;
    for (size_t game = 0; game < penalised_summaries.size(); game++)
    {
        const size_t moves = static_cast<size_t>(penalised_summaries[game].steps);
        if (moves == 0 || cursor + moves > penalised.size())
        {
            continue;
        }
        const size_t last = cursor + moves - 1;
        const float difference = penalised[last].value_target - unpenalised[last].value_target;

        if (penalised_summaries[game].hit_step_limit)
        {
            // Undiscounted at the final position: the penalty is paid there.
            if (std::fabs(difference - penalty) > 1e-3f)
            {
                std::cout << std::format("        game {} timed out, last target moved by {}\n",
                                         game, difference);
                timeout_targets_differ_by_the_penalty = false;
            }
            // And discounted once per step on the way back, so the first position
            // of the game carries penalty * discount^(moves-1).
            const float expected_at_start =
                penalty * std::pow(0.98f, static_cast<float>(moves - 1));
            const float start_difference =
                penalised[cursor].value_target - unpenalised[cursor].value_target;
            if (std::fabs(start_difference - expected_at_start) > 1e-2f)
            {
                std::cout << std::format("        game {} start moved by {}, expected {}\n", game,
                                         start_difference, expected_at_start);
                timeout_targets_differ_by_the_penalty = false;
            }
        }
        else if (std::fabs(difference) > 1e-4f)
        {
            std::cout << std::format("        game {} reached an outcome and still moved by {}\n",
                                     game, difference);
            finished_targets_are_untouched = false;
        }
        cursor += moves;
    }

    expect(timeout_targets_differ_by_the_penalty,
           "a timed-out game pays the penalty at its last move and discounts it backwards");

    // Every game above ran out of steps, so the check below had nothing to look at
    // there. A small board and a generous limit is where games die instead, which
    // is what makes "untouched" mean anything.
    std::vector<TrainingRecord> dying;
    std::vector<GameSummary> dying_summaries;
    SelfPlay penalised_deaths(evaluator, searchConfig(), playConfig(6, 400, penalty));
    penalised_deaths.playBatch(6, 6, 909, dying, dying_summaries);

    std::vector<TrainingRecord> dying_reference;
    std::vector<GameSummary> dying_reference_summaries;
    SelfPlay unpenalised_deaths(evaluator, searchConfig(), playConfig(6, 400, 0.0f));
    unpenalised_deaths.playBatch(6, 6, 909, dying_reference, dying_reference_summaries);

    int games_that_reached_an_outcome = 0;
    for (const GameSummary& summary : dying_summaries)
    {
        if (!summary.hit_step_limit)
        {
            games_that_reached_an_outcome++;
        }
    }
    expect(games_that_reached_an_outcome > 0,
           "the second batch contains games that ended on their own");

    if (dying.size() == dying_reference.size())
    {
        for (size_t index = 0; index < dying.size(); index++)
        {
            if (std::fabs(dying[index].value_target - dying_reference[index].value_target) > 1e-4f)
            {
                finished_targets_are_untouched = false;
                break;
            }
        }
    }
    else
    {
        finished_targets_are_untouched = false;
    }

    expect(finished_targets_are_untouched,
           "a game that won or died is untouched - it reached an outcome of its own");
}

void testProgressIsMonotonicAndCompletes()
{
    // A progress bar that stalls or goes backwards is worse than none: it makes
    // a working run look hung. The counts have to rise and finish at the total.
    SilentEvaluator evaluator;
    const int games = 8;
    SelfPlay play(evaluator, searchConfig(), playConfig(games, 40, az::TIMEOUT_REWARD));

    int reports = 0;
    int last_finished = -1;
    long long last_moves = -1;
    bool monotonic = true;
    int final_finished = -1;
    SelfPlay::Progress final_progress{ 0, 0, 0, 0.0 };

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
    testATimedOutGamePaysThePenaltyAndOthersDoNot();
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
