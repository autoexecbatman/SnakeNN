#include <cmath>

#include "snake_logic.h"


SnakeGame::SnakeGame() : rng(std::random_device{}()) {
    reset();
}

// Seeded games are reproducible, which is what makes two agents comparable.
SnakeGame::SnakeGame(unsigned int seed) : rng(seed) {
    reset();
}

void SnakeGame::reset() {
    snake.clear();
    snake.push_back(Position(GRID_WIDTH / 2, GRID_HEIGHT / 2));
    direction = Direction::RIGHT;
    pendingDirection = Direction::RIGHT;
    gameOver = false;
    won = false;
    score = 0;
    spawnFood();
}

bool SnakeGame::update() {
    if (gameOver) return false;
    
    // Apply pending direction change
    direction = pendingDirection;
    
    // Calculate next head position
    Position nextHead = getNextHeadPosition();
    
    // Check wall collision
    if (nextHead.x < 0 || nextHead.x >= GRID_WIDTH || 
        nextHead.y < 0 || nextHead.y >= GRID_HEIGHT) {
        gameOver = true;
        return false;
    }
    
    // Check self collision
    if (blocksMove(nextHead)) {
        gameOver = true;
        return false;
    }

    // Move snake
    snake.insert(snake.begin(), nextHead);

    // Check food collision
    if (nextHead == food) {
        score++;
        if (static_cast<int>(snake.size()) == CELL_COUNT) {
            // Grid full - there is nowhere left to place food and nothing left
            // to do. This is the win, and it is terminal.
            won = true;
            gameOver = true;
            return false;
        }
        spawnFood();
    } else {
        snake.pop_back();
    }

    return true;
}

void SnakeGame::setDirection(Direction dir) {
    // Prevent reversing into self
    if ((direction == Direction::UP && dir == Direction::DOWN) ||
        (direction == Direction::DOWN && dir == Direction::UP) ||
        (direction == Direction::LEFT && dir == Direction::RIGHT) ||
        (direction == Direction::RIGHT && dir == Direction::LEFT)) {
        return;
    }
    pendingDirection = dir;
}

std::vector<float> SnakeGame::getGameState() const {
    std::vector<float> state;
    
    Position head = snake[0];
    
    // Distance to walls (normalized)
    state.push_back(static_cast<float>(head.x) / GRID_WIDTH);           // distance to left wall
    state.push_back(static_cast<float>(GRID_WIDTH - head.x - 1) / GRID_WIDTH); // distance to right wall
    state.push_back(static_cast<float>(head.y) / GRID_HEIGHT);          // distance to top wall
    state.push_back(static_cast<float>(GRID_HEIGHT - head.y - 1) / GRID_HEIGHT); // distance to bottom wall
    
    // Food direction (normalized)
    float food_dx = static_cast<float>(food.x - head.x) / GRID_WIDTH;
    float food_dy = static_cast<float>(food.y - head.y) / GRID_HEIGHT;
    state.push_back(food_dx);
    state.push_back(food_dy);
    
    // Current direction (one-hot encoding)
    state.push_back(direction == Direction::UP ? 1.0f : 0.0f);
    state.push_back(direction == Direction::DOWN ? 1.0f : 0.0f);
    state.push_back(direction == Direction::LEFT ? 1.0f : 0.0f);
    state.push_back(direction == Direction::RIGHT ? 1.0f : 0.0f);
    
    // Danger detection (collision in next step for each direction)
    for (int i = 0; i < 4; i++) {
        Direction testDir = static_cast<Direction>(i);
        Position testPos = head;
        
        switch (testDir) {
            case Direction::UP: testPos.y--; break;
            case Direction::DOWN: testPos.y++; break;
            case Direction::LEFT: testPos.x--; break;
            case Direction::RIGHT: testPos.x++; break;
        }
        
        bool danger = (testPos.x < 0 || testPos.x >= GRID_WIDTH ||
                      testPos.y < 0 || testPos.y >= GRID_HEIGHT ||
                      blocksMove(testPos));
        state.push_back(danger ? 1.0f : 0.0f);
    }
    
    return state;
}

float SnakeGame::getReward() const {
    static bool food_eaten_last_step = false;
    static Position last_head_pos(-1, -1);
    
    if (gameOver) return -10.0f;
    
    Position head = snake[0];
    float reward = 0.0f;
    
    // Check if food was just eaten (compare with previous position)
    if (last_head_pos.x != -1) {  // Not first call
        if (last_head_pos == food) {  // Previous head position was on food
            reward += 10.0f;  // Big reward for eating food
            food_eaten_last_step = true;
        } else {
            food_eaten_last_step = false;
        }
    }
    
    // Small positive reward for staying alive
    reward += 0.1f;
    
    // Stronger distance-based reward (encourage moving toward food)
    float distance = std::abs(food.x - head.x) + std::abs(food.y - head.y);
    float max_distance = GRID_WIDTH + GRID_HEIGHT;  // Maximum possible distance
    reward += (max_distance - distance) * 0.1f;  // Closer = better reward
    
    // Update last position for next call
    last_head_pos = head;
    
    return reward;
}

bool SnakeGame::isGameOver() const {
    return gameOver;
}

bool SnakeGame::isWon() const {
    return won;
}

int SnakeGame::getScore() const {
    return score;
}

void SnakeGame::spawnFood() {
    std::vector<Position> availablePositions;
    
    for (int x = 0; x < GRID_WIDTH; x++) {
        for (int y = 0; y < GRID_HEIGHT; y++) {
            Position pos(x, y);
            if (!occupiesCell(pos)) {
                availablePositions.push_back(pos);
            }
        }
    }
    
    if (!availablePositions.empty()) {
        std::uniform_int_distribution<int> dist(0, availablePositions.size() - 1);
        food = availablePositions[dist(rng)];
    }
}

bool SnakeGame::occupiesCell(const Position& pos) const {
    for (const auto& segment : snake) {
        if (segment == pos) {
            return true;
        }
    }
    return false;
}

bool SnakeGame::blocksMove(const Position& pos) const {
    // Entering the tail cell is legal because the tail vacates it on the same
    // step - unless the snake eats this step, in which case the tail stays put
    // and the cell is still occupied.
    bool willEatFood = (pos == food);
    size_t blockingSegments = willEatFood ? snake.size() : snake.size() - 1;

    for (size_t index = 0; index < blockingSegments; index++) {
        if (snake[index] == pos) {
            return true;
        }
    }
    return false;
}

Position SnakeGame::getNextHeadPosition() const {
    Position head = snake[0];
    switch (direction) {
        case Direction::UP: head.y--; break;
        case Direction::DOWN: head.y++; break;
        case Direction::LEFT: head.x--; break;
        case Direction::RIGHT: head.x++; break;
    }
    return head;
}
