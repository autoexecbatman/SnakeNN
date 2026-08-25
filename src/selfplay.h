#pragma once

// Self-play: the agent generating its own training data.
//
// Plays a batch of real games under the search, and records each position with
// what the search decided there. Those labels beat the network that produced
// them - one forward pass against that guess plus hundreds of lookaheads - so
// training toward them moves the network up.
//
// One phase of a training iteration, not the whole run. Its win rate measures the
// exploration policy, not the agent, because the games carry root noise and
// sampled early moves; az_evaluate.cpp gives the real number.
//
// Usage, from az_trainer.cpp:
//
//     SelfPlay::Config play_config;                // every field is set explicitly
//     play_config.games_in_parallel = 256;         // games stepped together
//     play_config.step_limit = 1200;               // a game cut off here is a timeout
//     play_config.discount = 0.98f;                // for the value target only
//     play_config.timeout_reward = -10.0f;         // charged to a cut-off game's last move
//     play_config.step_reward = 0.0f;              // charged on every move
//     play_config.temperature = 0.5f;              // visits sampled this hot, early
//     play_config.temperature_moves = 50;          // for this many moves, then argmax
//     play_config.seed = 12345;                    // seeds action sampling, not the games
//
//     SelfPlay play(evaluator, search_config, play_config);
//
//     std::vector<TrainingRecord> records;
//     std::vector<GameSummary> summaries;
//     play.playBatch(10, 10, batch_seed, records, summaries);  // 10x10, appends to both
//     // records: one per position visited; summaries: one per game, so 256.
//
// Refuses a Config with fewer than one game in flight, from the constructor.

#include <functional>
#include <random>
#include <vector>

#include "evaluator.h"
#include "mcts.h"
#include "snake_env.h"

// One training record: what the network saw, what the search concluded, and what
// happened afterwards.
struct TrainingRecord
{
    // The position, not its encoding - encoded planes cost 3.2KB a move and swapped.
    SnakeEnv::Snapshot position;
    // The search's visit distribution over the three relative actions, normalised.
    // This is the training target for the policy head.
    float policy[SnakeEnv::ACTION_COUNT]{};
    float value_target{ 0.0f };  // discounted return from this position onward
    // Steps left to fill the board, as a fraction of the budget; 1 if it never won.
    float steps_target{ 1.0f };
    // The search's backed-up death risk per action, undiscounted.
    float death_risk_target[SnakeEnv::ACTION_COUNT]{};
    // False unless the search visited every root action; an unvisited one reads safe.
    bool death_risk_usable{ false };
    // Whether some move loses from this position. These are where a game is decided, and
    // they are a small minority of an iteration's positions, so a batch drawn uniformly
    // barely contains them.
    bool decisive{ false };
    // Whether this record's visit counts are worth training the policy on. False for a
    // move played from a cheap search: a policy trained on a shallow search learns the
    // shallow search.
    bool policy_usable{ true };
    // Which of the three actions was played from this position. The only one whose
    // outcome the finished game knows anything about.
    int played_action{ 0 };
    // 1 when this position was already lost - the head's reachable room had fallen below
    // the snake's own length and never recovered before the game ended in a death. 0 when
    // the game was won, timed out, or had not yet reached that point.
    //
    // Read off the finished trajectory rather than backed up from a search. A position
    // becomes unrecoverable a mean of 323 moves before the death it causes, which is an
    // order of magnitude past what 200 simulations or a 72-step discount can see, so the
    // search cannot supply this and the walk backwards can.
    float doom_target{ 0.0f };
    // One byte per cell, 1 where the head occupies this cell now or reaches it later in
    // the game. The ownership head's target.
    //
    // An auxiliary label: nothing at play time reads it. It exists because a per-cell
    // target gives the trunk localised credit assignment, where every other target here
    // is one number for a whole position - KataGo's largest measured gain, and the densest
    // available statement of where this snake can still go.
    std::vector<unsigned char> future_cells;

    // Roughly what this record costs, for a buffer capped by memory not by count.
    size_t bytesUsed() const
    {
        return sizeof(TrainingRecord) + position.body_cells.capacity() * sizeof(unsigned short) +
               future_cells.capacity();
    }
};

// How one game ended. One per game in a batch, in the order the games were created.
struct GameSummary
{
    // Apples eaten.
    int score{ 0 };
    // Moves taken before it ended.
    int steps{ 0 };
    // Whether it ended by filling the board.
    bool won{ false };
    // Whether it was cut off at the step limit. Exclusive with won; both false means
    // the snake died.
    bool hit_step_limit{ false };
    // Undiscounted, for the log; the value head trains on the discounted return.
    float total_reward{ 0.0f };
};

// Plays games with the search and records what it learns from them. The step limit
// is part of the task: uncapped, "wins" cannot tell good play from safe shuffling.
class SelfPlay
{
public:
    // Emitted once per move-batch; the caller does its own throttling.
    struct Progress
    {
        // Games in this batch.
        int games_total{ 0 };
        // How many have ended.
        int games_finished{ 0 };
        // Moves taken across all of them.
        long long moves_played{ 0 };
        // Seconds since the batch started.
        double elapsed_seconds{ 0.0 };
    };

    // Every field is set by the caller; the initializers make a miss a wrong number.
    struct Config
    {
        // Games stepped together. This is the whole of throughput: 32 games measured
        // 42-50k evaluations a second against 708k at 1024. Keep it in the hundreds.
        int games_in_parallel{ 0 };
        // Moves a game gets before it is cut off as a timeout. Part of the task, not a
        // safety valve - uncapped, a win cannot be told from safe shuffling.
        int step_limit{ 0 };
        // Per-step discount on the return the value head trains toward. At 0.98 it
        // cannot see much past 200 steps.
        float discount{ 0.0f };
        // Charged to a cut-off game's last move, so stalling costs what dying costs.
        float timeout_reward{ 0.0f };
        // Charged every step, pricing a slow route to an apple against a fast one.
        float step_reward{ 0.0f };
        // Visits are sampled this hot for temperature_moves moves, then argmax.
        float temperature{ 0.0f };
        // How many opening moves are sampled before selection switches to argmax.
        int temperature_moves{ 0 };
        // Seeds action sampling only. The games themselves are seeded per batch, by
        // the game_seed_base handed to playBatch.
        unsigned int seed{ 0 };
        // Share of moves that get the full search. 0 means every move does, which is what
        // every run before 2026-08-24 did; KataGo uses about a quarter.
        //
        // The rest are played from `fast_simulations` and their records do not train the
        // policy - a policy fitted to a shallow search learns the shallow search. They do
        // still train the value, the steps and the ownership heads, which is the whole
        // point: those want many positions, and only the policy wants deep ones.
        float full_search_fraction{ 0.0f };
        // The cheap budget, read only when full_search_fraction is above zero.
        int fast_simulations{ 0 };
    };

    // The evaluator is borrowed and must outlive this; throws below one game.
    SelfPlay(Evaluator& evaluator, const MonteCarloSearch::Config& search_config,
             const Config& config);

    // Pass an empty function to report nothing.
    void setProgressCallback(std::function<void(const Progress&)> callback);

    // Plays one batch to completion, appending positions and one entry per game.
    // Neither vector is cleared; game `index` is seeded `game_seed_base + index`.
    void playBatch(int board_width, int board_height, unsigned int game_seed_base,
                   std::vector<TrainingRecord>& records_out,
                   std::vector<GameSummary>& summaries_out);

    // Root moves in the last batch that sealed the head away from its own tail.
    long long sealedChoices() const { return sealed_choices_; }

private:
    Evaluator& evaluator_;
    MonteCarloSearch::Config search_config_;
    Config config_;
    std::mt19937 rng_;
    std::function<void(const Progress&)> progress_callback_;
    long long sealed_choices_{ 0 };

    // Sampled at temperature for the opening, argmax after; a zero policy picks 0.
    int sampleAction(const std::vector<float>& policy, int moves_played);
};
