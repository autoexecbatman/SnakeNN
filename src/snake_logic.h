#pragma once

#include <vector>
#include <random>

enum class Direction {
    UP, DOWN, LEFT, RIGHT
};

struct Position {
    int x, y;
    Position(int x = 0, int y = 0) : x(x), y(y) {}
    bool operator==(const Position& other) const {
        return x == other.x && y == other.y;
    }
};

class SnakeGame {
public:
    static const int GRID_WIDTH = 20;
    static const int GRID_HEIGHT = 20;
    
    static const int CELL_COUNT = GRID_WIDTH * GRID_HEIGHT;
    // The snake starts one segment long, so filling the grid takes this many foods.
    static const int FOODS_TO_WIN = CELL_COUNT - 1;

    SnakeGame();
    explicit SnakeGame(unsigned int seed);
    void reset();
    bool update();
    void setDirection(Direction dir);

    // State access for neural network
    std::vector<float> getGameState() const;
    float getReward() const;
    bool isGameOver() const;
    // True when the snake fills the grid. Terminal, and the only actual win.
    bool isWon() const;
    int getScore() const;
    Direction getDirection() const { return direction; }
    
    // Rendering access
    const std::vector<Position>& getSnakeBody() const { return snake; }
    const Position& getFoodPosition() const { return food; }
    
private:
    std::vector<Position> snake;
    Position food;
    Direction direction{ Direction::RIGHT };
    Direction pendingDirection{ Direction::RIGHT };
    bool gameOver{ false };
    bool won{ false };
    int score{ 0 };
    std::mt19937 rng;

    void spawnFood();
    // Whether any segment currently sits on this cell, tail included.
    bool occupiesCell(const Position& pos) const;
    // Whether moving onto this cell kills the snake. The tail cell is safe to
    // enter only when the tail is about to vacate it, which it does not do on
    // the step the snake eats.
    bool blocksMove(const Position& pos) const;
    Position getNextHeadPosition() const;
};
