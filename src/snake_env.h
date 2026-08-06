#pragma once
#include "snake_logic.h"  // Direction, Position
#include <random>
#include <vector>

// Training environment. Separate from SnakeGame on purpose:
//
//   - board size is a runtime argument, because the curriculum trains on small
//     boards first and SnakeGame fixes 20x20 at compile time for 25 other files
//   - all state is per-instance, so thousands of these can step in parallel
//     (SnakeGame::getReward keeps its previous head position in function-local
//     statics, which silently couples every game in the process)
//   - actions are relative, so there is no reverse action to ignore
//   - it encodes itself into planes for a convolutional network
//
// No LibTorch and no raylib, so it builds and tests without the CUDA toolchain.
class SnakeEnv {
public:
    // Relative to the current heading. A reverse action does not exist rather
    // than existing and being ignored, which keeps the MDP honest and the
    // search tree a third smaller per ply.
    enum class Action { STRAIGHT, LEFT, RIGHT };
    static const int ACTION_COUNT = 3;

    // body, head, food, tail timer, and one plane per heading
    static const int PLANE_COUNT = 8;

    // Reward scale follows Du, Gemp, Wu and Wu 2022 (arXiv:2211.09622), which
    // trained a winning Snake agent with +1 per apple, -10 for dying and +10 for
    // filling the board. The asymmetry matters: with +1/-1 a long greedy run
    // outscores a short safe one, which is the failure the old trainers here
    // learned. These are floats rather than an env parameter because a reward
    // scale that varies between runs makes their value heads incomparable.
    static constexpr float FOOD_REWARD = 1.0f;
    static constexpr float DEATH_REWARD = -10.0f;
    static constexpr float WIN_REWARD = 10.0f;

    struct StepResult {
        float reward;
        bool done;
        bool won;
    };

    SnakeEnv(int width, int height, unsigned int seed);

    void reset();
    StepResult step(Action action);

    int width() const { return width_; }
    int height() const { return height_; }
    int cellCount() const { return width_ * height_; }
    // The snake starts one segment long, so this many foods fills the board.
    int foodsToWin() const { return cellCount() - 1; }

    int score() const { return score_; }
    int steps() const { return steps_; }
    int stepsSinceFood() const { return steps_since_food_; }
    bool done() const { return done_; }
    bool won() const { return won_; }

    Direction heading() const { return heading_; }
    const std::vector<Position>& body() const { return body_; }
    Position food() const { return food_; }

    // Where `action` would put the head, without applying it. Search needs this
    // to order children before it commits to expanding them.
    Position headAfter(Action action) const;
    Direction headingAfter(Action action) const;

    // Writes PLANE_COUNT * height * width floats in plane-major order.
    // The caller owns the memory; nothing is allocated here, because this runs
    // once per node visited in the tree.
    void encode(float* planes_out) const;
    int encodedSize() const { return PLANE_COUNT * cellCount(); }

private:
    int width_;
    int height_;
    std::vector<Position> body_;   // index 0 is the head
    Position food_;
    Direction heading_;
    bool done_;
    bool won_;
    int score_;
    int steps_;
    int steps_since_food_;
    std::mt19937 rng_;
    // Occupancy by cell index, holding the body index of whatever sits there,
    // or -1. Keeps collision and food placement off the O(length) scan that
    // dominates a naive implementation once the snake is long.
    std::vector<int> occupancy_;

    int cellIndex(const Position& cell) const { return cell.y * width_ + cell.x; }
    bool insideGrid(const Position& cell) const;
    void spawnFood();
};
