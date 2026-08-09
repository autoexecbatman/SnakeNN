#include "selfplay.h"
#include <chrono>
#include <cmath>
#include <stdexcept>

SelfPlay::SelfPlay(Evaluator& evaluator, const MonteCarloSearch::Config& search_config,
                   const Config& config)
    : evaluator_(evaluator), search_config_(search_config), config_(config), rng_(config.seed)
{
    if (config.games_in_parallel < 1)
    {
        throw std::invalid_argument("self-play needs at least one game in flight");
    }
}

void SelfPlay::setProgressCallback(std::function<void(const Progress&)> callback)
{
    progress_callback_ = std::move(callback);
}

int SelfPlay::sampleAction(const std::vector<float>& policy, int moves_played)
{
    if (moves_played >= config_.temperature_moves || config_.temperature <= 0.0f)
    {
        int best = 0;
        for (int action = 1; action < SnakeEnv::ACTION_COUNT; action++)
        {
            if (policy[action] > policy[best])
            {
                best = action;
            }
        }
        return best;
    }

    float weights[SnakeEnv::ACTION_COUNT];
    float total = 0.0f;
    const float inverse_temperature = 1.0f / config_.temperature;
    for (int action = 0; action < SnakeEnv::ACTION_COUNT; action++)
    {
        weights[action] = std::pow(policy[action], inverse_temperature);
        total += weights[action];
    }
    if (total <= 0.0f)
    {
        return 0;
    }

    std::uniform_real_distribution<float> pick(0.0f, total);
    float target = pick(rng_);
    float running = 0.0f;
    for (int action = 0; action < SnakeEnv::ACTION_COUNT; action++)
    {
        running += weights[action];
        if (target <= running)
        {
            return action;
        }
    }
    return SnakeEnv::ACTION_COUNT - 1;
}

void SelfPlay::playBatch(int board_width, int board_height, unsigned int game_seed_base,
                         std::vector<TrainingRecord>& records_out,
                         std::vector<GameSummary>& summaries_out)
{
    const int game_count = config_.games_in_parallel;

    std::vector<SnakeEnv> games;
    games.reserve(game_count);
    for (int index = 0; index < game_count; index++)
    {
        games.emplace_back(board_width, board_height, game_seed_base + index, config_.step_limit);
    }

    // Per game: the positions visited so far and the reward collected leaving
    // each one. Returns are only computable once the game has finished, so the
    // trajectory is held and converted at the end.
    std::vector<std::vector<TrainingRecord>> trajectories(game_count);
    std::vector<std::vector<float>> rewards(game_count);
    std::vector<int> moves_played(game_count, 0);
    std::vector<bool> hit_limit(game_count, false);

    MonteCarloSearch search(evaluator_, search_config_);

    std::vector<int> live;
    std::vector<const SnakeEnv*> roots;

    auto started = std::chrono::high_resolution_clock::now();
    long long moves_played_total = 0;

    while (true)
    {
        live.clear();
        roots.clear();
        for (int index = 0; index < game_count; index++)
        {
            if (games[index].done())
            {
                continue;
            }
            if (games[index].steps() >= config_.step_limit)
            {
                hit_limit[index] = true;
                continue;
            }
            live.push_back(index);
            roots.push_back(&games[index]);
        }
        if (live.empty())
        {
            break;
        }

        std::vector<MonteCarloSearch::Result> results = search.search(roots);

        for (size_t position = 0; position < live.size(); position++)
        {
            const int index = live[position];
            SnakeEnv& game = games[index];

            TrainingRecord record;
            record.position = game.snapshot();
            for (int action = 0; action < SnakeEnv::ACTION_COUNT; action++)
            {
                record.policy[action] = results[position].policy[action];
            }
            record.value_target = 0.0f;  // filled once the game ends

            int action = sampleAction(results[position].policy, moves_played[index]);
            SnakeEnv::StepResult outcome = game.step(static_cast<SnakeEnv::Action>(action));

            trajectories[index].push_back(std::move(record));
            rewards[index].push_back(outcome.reward);
            moves_played[index]++;
            moves_played_total++;
        }

        if (progress_callback_)
        {
            Progress progress;
            progress.games_total = game_count;
            // A game counts as finished when it can no longer be stepped, which
            // includes the ones the step limit stopped - otherwise the bar
            // stalls at the end of a run full of timeouts.
            progress.games_finished = game_count - static_cast<int>(live.size());
            progress.moves_played = moves_played_total;
            progress.elapsed_seconds =
                std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - started)
                    .count();
            progress_callback_(progress);
        }
    }

    // A final report, because the loop breaks as soon as no game is live and
    // whatever finished on that last batch would otherwise never be counted -
    // leaving the bar stopped short of the end, which reads as a hang.
    if (progress_callback_)
    {
        Progress progress;
        progress.games_total = game_count;
        progress.games_finished = game_count;
        progress.moves_played = moves_played_total;
        progress.elapsed_seconds =
            std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - started)
                .count();
        progress_callback_(progress);
    }

    // Discounted returns, walked backwards from the end of each game. A game cut
    // off by the step limit gets no bootstrap value: it did not win, and
    // pretending its unfinished tail was worth something is exactly how a value
    // head learns that stalling pays.
    for (int index = 0; index < game_count; index++)
    {
        float carried = 0.0f;
        for (int position = static_cast<int>(trajectories[index].size()); position-- > 0;)
        {
            carried = rewards[index][position] + config_.discount * carried;
            trajectories[index][position].value_target = carried;
        }

        for (TrainingRecord& record : trajectories[index])
        {
            records_out.push_back(std::move(record));
        }

        GameSummary summary;
        summary.score = games[index].score();
        summary.steps = games[index].steps();
        summary.won = games[index].won();
        summary.hit_step_limit = hit_limit[index];
        summary.total_reward = 0.0f;
        for (float reward : rewards[index])
        {
            summary.total_reward += reward;
        }
        summaries_out.push_back(summary);
    }
}
