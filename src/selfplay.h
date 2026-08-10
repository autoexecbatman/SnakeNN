#pragma once
#include "evaluator.h"
#include "mcts.h"
#include "snake_env.h"
#include <functional>
#include <random>
#include <vector>

// One training record: what the network saw, what the search concluded, and
// what actually happened afterwards.
struct TrainingRecord
{
    // The position, not its encoding. Storing encoded planes cost 3.2KB per
    // move at 10x10 and drove a long run into swap; a snapshot is a sixteenth
    // of that and the planes are regenerated when a batch is drawn.
    SnakeEnv::Snapshot position;
    float policy[SnakeEnv::ACTION_COUNT]{};
    float value_target{ 0.0f };  // discounted return from this position onward

    // Roughly what this record costs, for a buffer that is capped by memory
    // rather than by a record count that means different things per board.
    size_t bytesUsed() const
    {
        return sizeof(TrainingRecord) + position.body_cells.capacity() * sizeof(unsigned short);
    }
};

struct GameSummary
{
    int score{ 0 };
    int steps{ 0 };
    bool won{ false };
    bool hit_step_limit{ false };
    float total_reward{ 0.0f };
};

// Plays games with the search and records what it learns from them.
//
// The step limit is part of the task, not a safety valve. Du et al. 2022 cap a
// 10x10 game at 1,200 steps, and under that cap the Hamiltonian cycle strategy
// wins zero games out of a thousand - which is the point. Without a limit,
// "wins" fails to distinguish a policy that plays well from one that shuffles
// safely for a hundred thousand steps, and the value head learns that stalling
// is as good as winning.
class SelfPlay
{
public:
    // Reported while a batch is in flight, because an iteration takes minutes
    // and a terminal that prints nothing for minutes is indistinguishable from
    // one that has hung. Emitted once per move-batch; throttling is the
    // caller's business, since only the caller knows what it is drawing to.
    struct Progress
    {
        int games_total{ 0 };
        int games_finished{ 0 };
        long long moves_played{ 0 };
        double elapsed_seconds{ 0.0 };
    };

    // Every field is set explicitly by the caller. The initializers exist so that
    // forgetting one is a wrong number rather than undefined behaviour - a garbage
    // discount reads as a plausible run that cannot be reproduced.
    struct Config
    {
        int games_in_parallel{ 0 };
        int step_limit{ 0 };
        float discount{ 0.0f };
        // Paid once by a game the step limit cut off, added to the reward of its
        // last move so the discounted return carries it backwards.
        //
        // Zero reproduces the behaviour this replaced, in which a timeout was worth
        // more than a death and stalling was therefore the safe play. A game that
        // won or died is untouched - it reached an outcome of its own.
        float timeout_reward{ 0.0f };
        // Visit counts are sampled at this temperature for the first moves and
        // greedily after, which is how self-play stays varied early without
        // throwing away the endgame.
        float temperature{ 0.0f };
        int temperature_moves{ 0 };
        unsigned int seed{ 0 };
    };

    SelfPlay(Evaluator& evaluator, const MonteCarloSearch::Config& search_config,
             const Config& config);

    // Pass an empty function to report nothing.
    void setProgressCallback(std::function<void(const Progress&)> callback);

    // Plays one batch of games to completion, appending every visited position
    // to `records_out` and one entry per game to `summaries_out`.
    void playBatch(int board_width, int board_height, unsigned int game_seed_base,
                   std::vector<TrainingRecord>& records_out,
                   std::vector<GameSummary>& summaries_out);

private:
    Evaluator& evaluator_;
    MonteCarloSearch::Config search_config_;
    Config config_;
    std::mt19937 rng_;
    std::function<void(const Progress&)> progress_callback_;

    int sampleAction(const std::vector<float>& policy, int moves_played);
};
