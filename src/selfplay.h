#pragma once
#include "evaluator.h"
#include "mcts.h"
#include "snake_env.h"
#include <random>
#include <vector>

// One training record: what the network saw, what the search concluded, and
// what actually happened afterwards.
struct TrainingRecord {
    std::vector<float> planes;   // encoded observation
    float policy[SnakeEnv::ACTION_COUNT];
    float value_target;          // discounted return from this position onward
};

struct GameSummary {
    int score;
    int steps;
    bool won;
    bool hit_step_limit;
    float total_reward;
};

// Plays games with the search and records what it learns from them.
//
// The step limit is part of the task, not a safety valve. Du et al. 2022 cap a
// 10x10 game at 1,200 steps, and under that cap the Hamiltonian cycle strategy
// wins zero games out of a thousand - which is the point. Without a limit,
// "wins" fails to distinguish a policy that plays well from one that shuffles
// safely for a hundred thousand steps, and the value head learns that stalling
// is as good as winning.
class SelfPlay {
public:
    struct Config {
        int games_in_parallel;
        int step_limit;
        float discount;
        // Visit counts are sampled at this temperature for the first moves and
        // greedily after, which is how self-play stays varied early without
        // throwing away the endgame.
        float temperature;
        int temperature_moves;
        unsigned int seed;
    };

    SelfPlay(Evaluator& evaluator, const MonteCarloSearch::Config& search_config,
             const Config& config);

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

    int sampleAction(const std::vector<float>& policy, int moves_played);
};
