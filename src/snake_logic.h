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
    
    SnakeGame();
    void reset();
    bool update();
    void setDirection(Direction dir);
    
    // State access for neural network
    std::vector<float> getGameState() const;
    float getReward() const;
    bool isGameOver() const;
    int getScore() const;
    
    // Rendering access
    const std::vector<Position>& getSnakeBody() const { return snake; }
    const Position& getFoodPosition() const { return food; }
    
private:
    std::vector<Position> snake;
    Position food;
    Direction direction;
    Direction pendingDirection;
    bool gameOver;
    int score;
    std::mt19937 rng;
    
    void spawnFood();
    bool checkCollision(const Position& pos) const;
    Position getNextHeadPosition() const;
};
