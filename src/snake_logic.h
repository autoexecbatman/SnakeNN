#pragma once

#include <vector>
#include <random>

// The 20x20 Snake the visual programs and the older trainers play. The board size is
// fixed at compile time; SnakeEnv is the one that takes a size at runtime and is what
// the learned agent trains on.
//
// Usage - one game, driven a step at a time:
//
//     SnakeGame game(seed);            // seeded, so the run reproduces
//     game.setDirection(Direction::UP);
//     while (game.update())            // false once the game is over
//     {
//         // draw, or read game.getScore()
//     }
//     const bool filled_the_board = game.isWon();
//
// The default constructor seeds from std::random_device, so use the seeded one for
// anything being measured.

// Which way the head is travelling. Absolute, not relative to the snake.
enum class Direction {
    UP, DOWN, LEFT, RIGHT
};

// A cell on the board, in columns from the left and rows from the top.
//
//     const Position head = game.getSnakeBody()[0];
//     if (head == game.getFoodPosition()) { /* the snake is on the apple */ }
struct Position {
    // Column, then row. Both count from zero and neither is bounds-checked here.
    int x, y;
    // Defaults to the top-left cell, so a Position member needs no initialiser.
    Position(int x = 0, int y = 0) : x(x), y(y) {}
    // Whether two positions name the same cell.
    bool operator==(const Position& other) const {
        return x == other.x && y == other.y;
    }
};

// One 20x20 game. See the top of this file for how to run one.
class SnakeGame {
public:
    // Columns on the board, fixed at compile time. SnakeEnv takes a size at runtime.
    static constexpr int GRID_WIDTH = 20;
    // Rows, likewise.
    static constexpr int GRID_HEIGHT = 20;
    
    // Cells on the board. A snake this long has filled it.
    static constexpr int CELL_COUNT = GRID_WIDTH * GRID_HEIGHT;
    // The snake starts one segment long, so filling the grid takes this many foods.
    static constexpr int FOODS_TO_WIN = CELL_COUNT - 1;

    // Seeds the apple stream from std::random_device. Not for anything measured - two
    // runs will not agree.
    SnakeGame();
    // Seeds it explicitly, so the same seed replays the same game.
    //
    //     SnakeGame game(42);
    explicit SnakeGame(unsigned int seed);
    // Back to a one-segment snake heading right, score zero, a fresh apple. Keeps the
    // random stream, so a second game from one object is not a repeat of the first.
    void reset();
    // Advances one step, applying whatever setDirection last asked for. Returns false
    // once the game is over - by a wall, by the body, or by filling the board.
    //
    //     while (game.update()) { /* still playing */ }
    bool update();
    // Asks for a heading, applied by the next update. A reverse of the current heading
    // is ignored, so the snake cannot turn back into its own neck.
    void setDirection(Direction dir);

    // The 14-feature state the original DQN trainer was written against: food direction,
    // current heading one-hot, and a danger flag per absolute direction.
    //
    // Later experiments each build their own vector instead, of 8 or 11 features, so a
    // model and a consumer that disagree on the count is the standard failure here.
    // Check what the network was constructed with before feeding it this.
    std::vector<float> getGameState() const;
    // The DQN reward for the last step: -10 dead, +10 for filling the board, +1 for an
    // apple, and a small shaping term for closing on the food.
    //
    // Keeps the previous head position in function-local statics, so the value is shared
    // by every SnakeGame in the process and survives reset(). Correct only where one
    // game runs at a time. The cycle agent and the learned stack do not call it.
    float getReward() const;
    // Whether the game has ended, however it ended.
    bool isGameOver() const;
    // True when the snake fills the grid. Terminal, and the only actual win.
    bool isWon() const;
    // Apples eaten. FOODS_TO_WIN of them fills the board.
    int getScore() const;
    // The heading the last update actually used, not the one setDirection asked for.
    Direction getDirection() const { return direction; }

    // The segments, head first. Invalidated by update() and reset().
    const std::vector<Position>& getSnakeBody() const { return snake; }
    // Where the apple is. Meaningless on a won board, where none was placed.
    const Position& getFoodPosition() const { return food; }
    
private:
    // Head first, as every consumer expects.
    std::vector<Position> snake;
    Position food;
    // What the last update used.
    Direction direction{ Direction::RIGHT };
    // What setDirection asked for, applied at the start of the next update.
    Direction pendingDirection{ Direction::RIGHT };
    bool gameOver{ false };
    bool won{ false };
    int score{ 0 };
    // The apple stream. Survives reset, so repeated games from one object differ.
    std::mt19937 rng;

    // Places an apple uniformly among the free cells. Requires one to exist.
    void spawnFood();
    // Whether any segment currently sits on this cell, tail included.
    bool occupiesCell(const Position& pos) const;
    // Whether moving onto this cell kills the snake. The tail cell is safe to
    // enter only when the tail is about to vacate it, which it does not do on
    // the step the snake eats.
    bool blocksMove(const Position& pos) const;
    // Where the head lands next, from the pending direction. No bounds check - the
    // caller tests the result against the walls.
    Position getNextHeadPosition() const;
};
