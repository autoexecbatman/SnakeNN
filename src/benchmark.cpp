#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

#include "cycle_agent.h"
#include "snake_logic.h"

// Headless benchmark over seeded games.
//
// The metric that matters is the win rate - the fraction of games in which the
// snake fills the grid. Average score is reported alongside it, but a high
// average is not progress toward winning and must not be read as such. The
// trainers in this repository scored themselves on the fraction of games with
// at least one food, which saturates at 1 of 399 and says nothing about a win.

namespace {

enum class Outcome {
    WON, HIT_WALL, HIT_SELF, STEP_LIMIT
};

struct GameResult {
    Outcome outcome;
    int score;
    long long steps;
};

const char* outcomeName(Outcome outcome) {
    switch (outcome) {
        case Outcome::WON: return "won";
        case Outcome::HIT_WALL: return "hit wall";
        case Outcome::HIT_SELF: return "hit self";
        case Outcome::STEP_LIMIT: return "step limit";
    }
    return "unknown";
}

Outcome classifyEnding(const SnakeGame& game, long long steps, long long step_limit) {
    if (game.isWon()) {
        return Outcome::WON;
    }
    if (steps >= step_limit) {
        return Outcome::STEP_LIMIT;
    }

    // update() rejects the move without applying it, so the head is still where
    // it was and the fatal cell is one step ahead in the current direction.
    Position head = game.getSnakeBody()[0];
    switch (game.getDirection()) {
        case Direction::UP: head.y--; break;
        case Direction::DOWN: head.y++; break;
        case Direction::LEFT: head.x--; break;
        case Direction::RIGHT: head.x++; break;
    }

    bool off_grid = head.x < 0 || head.x >= SnakeGame::GRID_WIDTH ||
                    head.y < 0 || head.y >= SnakeGame::GRID_HEIGHT;
    return off_grid ? Outcome::HIT_WALL : Outcome::HIT_SELF;
}

GameResult playOneGame(const CycleAgent& agent, unsigned int seed, long long step_limit) {
    SnakeGame game(seed);
    long long steps = 0;

    while (!game.isGameOver() && steps < step_limit) {
        game.setDirection(agent.chooseMove(game));
        game.update();
        steps++;
    }

    GameResult result;
    result.outcome = classifyEnding(game, steps, step_limit);
    result.score = game.getScore();
    result.steps = steps;
    return result;
}

}  // namespace

int main(int argc, char** argv) {
    int game_count = 20;
    unsigned int base_seed = 1;
    // A cycle lap costs at most one full pass of the grid per food, so the
    // worst honest run is about CELL_COUNT * FOODS_TO_WIN steps. The limit is
    // set above that so that hitting it means the agent stalled, not that the
    // budget was too small.
    long long step_limit = 4LL * SnakeGame::CELL_COUNT * SnakeGame::FOODS_TO_WIN;

    if (argc > 1) {
        game_count = std::stoi(argv[1]);
    }
    if (argc > 2) {
        base_seed = static_cast<unsigned int>(std::stoul(argv[2]));
    }

    std::cout << "Benchmark: cycle-following agent" << std::endl;
    std::cout << "Grid " << SnakeGame::GRID_WIDTH << "x" << SnakeGame::GRID_HEIGHT
              << ", " << SnakeGame::FOODS_TO_WIN << " foods to win" << std::endl;
    std::cout << "Games " << game_count << ", seeds " << base_seed << ".."
              << (base_seed + game_count - 1) << ", step limit " << step_limit
              << std::endl << std::endl;

    CycleAgent agent(SnakeGame::GRID_WIDTH, SnakeGame::GRID_HEIGHT);
    agent.buildCycle();

    std::vector<GameResult> results;
    results.reserve(game_count);

    for (int game_index = 0; game_index < game_count; game_index++) {
        GameResult result = playOneGame(agent, base_seed + game_index, step_limit);
        results.push_back(result);
        std::cout << "  seed " << (base_seed + game_index)
                  << "  score " << result.score
                  << "  steps " << result.steps
                  << "  " << outcomeName(result.outcome) << std::endl;
    }

    int wins = 0;
    long long total_score = 0;
    long long total_steps = 0;
    int best_score = 0;
    int worst_score = SnakeGame::FOODS_TO_WIN;
    for (const auto& result : results) {
        if (result.outcome == Outcome::WON) {
            wins++;
        }
        total_score += result.score;
        total_steps += result.steps;
        best_score = std::max(best_score, result.score);
        worst_score = std::min(worst_score, result.score);
    }

    std::cout << std::endl << "=== Summary over " << game_count << " games ===" << std::endl;
    std::cout << "Wins:        " << wins << "/" << game_count
              << "  (" << (100.0 * wins / game_count) << "%)" << std::endl;
    std::cout << "Score:       mean " << (static_cast<double>(total_score) / game_count)
              << ", min " << worst_score << ", max " << best_score
              << " of " << SnakeGame::FOODS_TO_WIN << std::endl;
    std::cout << "Steps:       mean " << (static_cast<double>(total_steps) / game_count)
              << std::endl;

    for (Outcome outcome : {Outcome::WON, Outcome::HIT_WALL, Outcome::HIT_SELF, Outcome::STEP_LIMIT}) {
        int count = 0;
        for (const auto& result : results) {
            if (result.outcome == outcome) {
                count++;
            }
        }
        if (count > 0) {
            std::cout << "Ending:      " << outcomeName(outcome) << " x" << count << std::endl;
        }
    }

    return wins == game_count ? 0 : 1;
}
