// Implementation of SelfPlay. The interface, and how to call it, are in selfplay.h.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <random>
#include <stdexcept>

#include "seed_policy.h"
#include "selfplay.h"

SelfPlay::SelfPlay(Evaluator& evaluator, const MonteCarloSearch::Config& search_config,
                   const Config& config)
    : evaluator_(evaluator),
      search_config_(search_config),
      config_(config),
      rng_(seeds::streamSeed(config.seed, seeds::Stream::SelfPlaySampling))
{
    // A batch of no games produces no summaries, and the caller divides by that
    // count to report a score.
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
    // Past the opening, or with temperature off, the visit argmax is the move.
    if (moves_played >= config_.temperature_moves || config_.temperature <= 0.0f)
    {
        // Seeded with the first action, so no sentinel stands in for "none seen".
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

    // Visit shares raised to 1/temperature: below one this sharpens the
    // distribution toward the argmax, above one it flattens it toward uniform.
    float weights[SnakeEnv::ACTION_COUNT];
    float total = 0.0f;
    const float inverse_temperature = 1.0f / config_.temperature;
    for (int action = 0; action < SnakeEnv::ACTION_COUNT; action++)
    {
        weights[action] = std::pow(policy[action], inverse_temperature);
        total += weights[action];
    }
    // Every weight zero means the search reported no visits at all. Sampling that
    // would divide by zero, and any action is as unjustified as any other.
    if (total <= 0.0f)
    {
        return 0;
    }

    // Drawn against the running total rather than against normalised weights, which
    // keeps the one division out of the loop.
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
    // Reached only when rounding leaves the target a hair above the final total.
    return SnakeEnv::ACTION_COUNT - 1;
}

void SelfPlay::playBatch(int board_width, int board_height, unsigned int game_seed_base,
                         std::vector<TrainingRecord>& records_out,
                         std::vector<GameSummary>& summaries_out)
{
    const int game_count = config_.games_in_parallel;

    // Game `index` takes seed `game_seed_base + index`, which is what makes a batch
    // reproducible from the one seed the caller passed.
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

    // Built per batch, so its counters cover this batch alone.
    MonteCarloSearch search(evaluator_, search_config_);

    // The games still being stepped, and pointers to them for the search. Rebuilt
    // every move because a game that ended must leave the batch.
    std::vector<int> live;
    std::vector<const SnakeEnv*> roots;

    auto started = std::chrono::high_resolution_clock::now();
    long long moves_played_total = 0;

    // One pass of this loop is one move in every game still running.
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
            // Checked here rather than inside the environment so the game is
            // recorded as cut off rather than as finished.
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

        // Every live game searched in one call, so all their leaves reach the
        // network in a single forward pass.
        // One coin per pass rather than per game, so every game in a batch searches the
        // same depth on the same pass - the batch shares one forward pass, and mixing
        // budgets inside it would leave most of the batch idle while the deep games finish.
        const bool full_search =
            config_.full_search_fraction <= 0.0f ||
            std::uniform_real_distribution<float>(0.0f, 1.0f)(rng_) < config_.full_search_fraction;
        const int budget = full_search ? search_config_.simulations : config_.fast_simulations;
        std::vector<MonteCarloSearch::Result> results = search.searchWith(roots, budget);

        for (size_t position = 0; position < live.size(); position++)
        {
            const int index = live[position];
            SnakeEnv& game = games[index];

            // The record is written before the move is taken: it describes the
            // position the search was given, not the one the move led to.
            TrainingRecord record;
            record.position = game.snapshot();
            const std::vector<float> shares = results[position].policy();
            record.death_risk_usable = results[position].allActionsVisited();
            record.policy_usable = full_search;
            for (int action = 0; action < SnakeEnv::ACTION_COUNT; action++)
            {
                record.policy[action] = shares[action];
                // Written only where every action carries a measured risk. Otherwise the
                // record keeps its zeros, which death_risk_usable already forbids reading.
                if (record.death_risk_usable)
                {
                    record.death_risk_target[action] = results[position].death_risk[action].value();
                }
            }
            // Asked before the move, so it describes the position the search was given.
            for (int probe = 0; probe < SnakeEnv::ACTION_COUNT; probe++)
            {
                if (game.wouldDie(static_cast<SnakeEnv::Action>(probe)))
                {
                    record.decisive = true;
                    break;
                }
            }
            record.value_target = 0.0f;  // filled once the game ends

            // Sampled early and greedy later, so a batch explores without throwing
            // away its endgames.
            int action = sampleAction(shares, moves_played[index]);
            SnakeEnv::StepResult outcome = game.step(static_cast<SnakeEnv::Action>(action));

            trajectories[index].push_back(std::move(record));
            // The reward for leaving this position, which is what the backward walk
            // below discounts into every earlier one.
            rewards[index].push_back(outcome.reward + config_.step_reward);
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
        // Charged to the last move, so the backward walk discounts it into every
        // earlier position of the game. A game that won or died reached an outcome
        // of its own and already paid for it.
        if (hit_limit[index] && !rewards[index].empty())
        {
            rewards[index].back() += config_.timeout_reward;
        }

        // Sized from the game itself, so a board size never has to be threaded in.
        const int cells = games[index].cellCount();
        // The same walk fills the ownership target. Going backwards, the set of cells
        // the head still has ahead of it only ever grows, so `reached` accumulates and
        // every position is labelled with its own future rather than the whole game's.
        std::vector<unsigned char> reached(static_cast<size_t>(cells), 0);
        float carried = 0.0f;
        for (int position = static_cast<int>(trajectories[index].size()); position-- > 0;)
        {
            carried = rewards[index][position] + config_.discount * carried;
            trajectories[index][position].value_target = carried;
            // The head's own cell at this position, added before the copy, so a position
            // always owns where it stands.
            reached[trajectories[index][position].position.body_cells[0]] = 1;
            trajectories[index][position].future_cells = reached;
        }

        // Steps-to-go, counted forward from each position to the end of the game
        // and scaled by the budget. A game that did not win never reached the end
        // of the task, so its positions are labelled with the whole budget rather
        // than with the steps it happened to survive - the alternative teaches
        // that dying early means very little work remained.
        const int moves = static_cast<int>(trajectories[index].size());
        const float budget = static_cast<float>(config_.step_limit);
        for (int position = 0; position < moves; position++)
        {
            const float remaining =
                games[index].won() ? static_cast<float>(moves - position) : budget;
            trajectories[index][position].steps_target = std::min(remaining / budget, 1.0f);
        }

        // Moved out rather than copied: a record carries a snapshot with a heap
        // allocation in it.
        for (TrainingRecord& record : trajectories[index])
        {
            records_out.push_back(std::move(record));
        }

        // Undiscounted, unlike the value target above - this one is for the log.
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

    // Read before the search goes out of scope; it is the caller's only route to
    // this count.
    sealed_choices_ = search.sealedChoices();
}
