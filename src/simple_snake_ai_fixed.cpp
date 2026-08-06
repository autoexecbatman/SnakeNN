#include "neural_network.h"
#include "snake_logic.h"
#include <iostream>
#include <raylib.h>
#include <vector>
#include <algorithm>
#include <memory>
#include <queue>

// SIMPLE Snake AI - Weighted Scoring Implementation  
class SimpleSnakeAI {
public:
    SimpleSnakeAI() : network_8x64(8, 64, 4), network_8x128(8, 128, 4), network_11x128(11, 128, 4) {
        position_history.clear();
        steps_since_food = 0;
    }
    
    bool loadAnyModel(const std::vector<std::string>& model_paths) {
        for (const auto& path : model_paths) {
            if (tryLoad(network_8x64, path, "8x64")) {
                active_network = &network_8x64;
                network_type = "8x64";
                loaded_model = path;
                return true;
            }
            if (tryLoad(network_8x128, path, "8x128")) {
                active_network = &network_8x128;
                network_type = "8x128";
                loaded_model = path;
                return true;
            }
            if (tryLoad(network_11x128, path, "11x128")) {
                active_network = &network_11x128;
                network_type = "11x128";
                loaded_model = path;
                return true;
            }
        }
        return false;
    }
    
    void runDemo() {
        const int CELL_SIZE = 15;
        const int SCREEN_WIDTH = SnakeGame::GRID_WIDTH * CELL_SIZE;
        const int SCREEN_HEIGHT = SnakeGame::GRID_HEIGHT * CELL_SIZE + 200;
        
        InitWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "Simple Snake AI - Fixed Collision Detection");
        SetTargetFPS(12);
        
        SnakeGame game;
        game.reset();
        
        int total_games = 0;
        int total_score = 0;
        int max_score = 0;
        bool game_over = false;
        int steps = 0;
        
        while (!WindowShouldClose()) {
            if (IsKeyPressed(KEY_SPACE) || game_over) {
                if (game_over) {
                    total_games++;
                    int current_score = game.getScore();
                    total_score += current_score;
                    if (current_score > max_score) max_score = current_score;
                    
                    std::cout << "Game " << total_games << " - Score: " << current_score << std::endl;
                }
                
                game.reset();
                game_over = false;
                steps = 0;
            }
            
            if (!game.isGameOver() && !game_over) {
                if (steps < 5000) { // Prevent infinite loops
                    Direction move = getBestMove(game);
                    game.setDirection(move);
                    game.update();
                    steps++;
                    
                    if (game.isGameOver()) {
                        game_over = true;
                    }
                } else {
                    game_over = true; // Timeout
                }
            }
            
            BeginDrawing();
            ClearBackground(BLACK);
            
            // Draw snake
            auto snake = game.getSnakeBody();
            for (size_t i = 0; i < snake.size(); i++) {
                Color color = (i == 0) ? GREEN : DARKGREEN;
                DrawRectangle(snake[i].x * CELL_SIZE, snake[i].y * CELL_SIZE, 
                            CELL_SIZE - 1, CELL_SIZE - 1, color);
            }
            
            // Draw food
            auto food = game.getFoodPosition();
            DrawRectangle(food.x * CELL_SIZE, food.y * CELL_SIZE, 
                        CELL_SIZE - 1, CELL_SIZE - 1, RED);
            
            // Draw statistics
            float avg_score = total_games > 0 ? (float)total_score / total_games : 0;
            DrawText(TextFormat("Game: %d | Score: %d | Max: %d | Avg: %.1f", 
                              total_games, game.getScore(), max_score, avg_score), 
                   10, SCREEN_HEIGHT - 40, 20, WHITE);
            DrawText(TextFormat("Steps: %d | Press SPACE to restart", steps), 
                   10, SCREEN_HEIGHT - 20, 20, WHITE);
            
            EndDrawing();
        }
        
        CloseWindow();
    }
    
private:
    NeuralNetwork network_8x64, network_8x128, network_11x128;
    NeuralNetwork* active_network = nullptr;
    std::string network_type, loaded_model;
    
    std::vector<Position> position_history;
    int steps_since_food;
    
    bool tryLoad(NeuralNetwork& network, const std::string& path, const std::string& arch_name) {
        try {
            torch::load(network, path);
            std::cout << "✓ Loaded " << path << " with " << arch_name << " architecture" << std::endl;
            return true;
        } catch (const std::exception& e) {
            return false;
        }
    }
    
    Direction getBestMove(const SnakeGame& game) {
        auto head = game.getSnakeBody()[0];
        auto snake = game.getSnakeBody();
        auto food = game.getFoodPosition();
        
        std::vector<Direction> all_dirs = {Direction::UP, Direction::DOWN, Direction::LEFT, Direction::RIGHT};
        Direction best_move = Direction::UP;
        float best_score = -999999;
        
        for (Direction dir : all_dirs) {
            Position test_pos = getNextPosition(head, dir);
            
            if (!isBasicSafe(test_pos, snake)) {
                continue;
            }
            
            // Weighted scoring
            float food_distance = abs(test_pos.x - food.x) + abs(test_pos.y - food.y);
            float food_score = 100.0f / (1.0f + food_distance);
            float space_score = countAccessibleSpace(test_pos, snake) * 2.0f;
            float safety_score = getSafetyScore(test_pos, snake) * 10.0f;
            
            float total_score = food_score + space_score + safety_score;
            
            if (total_score > best_score) {
                best_score = total_score;
                best_move = dir;
            }
        }
        
        return best_move;
    }
    
    Position getNextPosition(const Position& pos, Direction dir) const {
        Position next = pos;
        switch (dir) {
            case Direction::UP: next.y--; break;
            case Direction::DOWN: next.y++; break;
            case Direction::LEFT: next.x--; break;
            case Direction::RIGHT: next.x++; break;
        }
        return next;
    }
    
    bool isBasicSafe(const Position& pos, const std::vector<Position>& snake) const {
        // Check walls
        if (pos.x < 0 || pos.x >= SnakeGame::GRID_WIDTH ||
            pos.y < 0 || pos.y >= SnakeGame::GRID_HEIGHT) {
            return false;
        }
        
        // Check snake collision - use fixed logic
        for (size_t i = 0; i < snake.size(); i++) {
            if (snake[i].x == pos.x && snake[i].y == pos.y) {
                return false;
            }
        }
        
        return true;
    }
    
    float getSafetyScore(const Position& pos, const std::vector<Position>& snake) const {
        int blocked_sides = 0;
        std::vector<Position> adjacent = {
            {pos.x, pos.y - 1}, {pos.x, pos.y + 1}, {pos.x - 1, pos.y}, {pos.x + 1, pos.y}
        };
        
        for (const auto& adj_pos : adjacent) {
            if (adj_pos.x < 0 || adj_pos.x >= SnakeGame::GRID_WIDTH ||
                adj_pos.y < 0 || adj_pos.y >= SnakeGame::GRID_HEIGHT) {
                blocked_sides++;
            } else {
                for (size_t i = 0; i < snake.size(); i++) {
                    if (snake[i].x == adj_pos.x && snake[i].y == adj_pos.y) {
                        blocked_sides++;
                        break;
                    }
                }
            }
        }
        
        return 4.0f - blocked_sides;
    }
    
    int countAccessibleSpace(const Position& start, const std::vector<Position>& snake) const {
        std::vector<std::vector<bool>> visited(SnakeGame::GRID_HEIGHT, std::vector<bool>(SnakeGame::GRID_WIDTH, false));
        
        // Mark snake positions as blocked
        for (const auto& segment : snake) {
            if (segment.x >= 0 && segment.x < SnakeGame::GRID_WIDTH && 
                segment.y >= 0 && segment.y < SnakeGame::GRID_HEIGHT) {
                visited[segment.y][segment.x] = true;
            }
        }
        
        if (start.x < 0 || start.x >= SnakeGame::GRID_WIDTH ||
            start.y < 0 || start.y >= SnakeGame::GRID_HEIGHT ||
            visited[start.y][start.x]) {
            return 0;
        }
        
        // Flood fill
        std::queue<Position> queue;
        queue.push(start);
        visited[start.y][start.x] = true;
        
        int accessible_count = 0;
        std::vector<Position> directions = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
        
        while (!queue.empty()) {
            Position current = queue.front();
            queue.pop();
            accessible_count++;
            
            for (const auto& dir : directions) {
                Position next = {current.x + dir.x, current.y + dir.y};
                
                if (next.x >= 0 && next.x < SnakeGame::GRID_WIDTH &&
                    next.y >= 0 && next.y < SnakeGame::GRID_HEIGHT &&
                    !visited[next.y][next.x]) {
                    
                    visited[next.y][next.x] = true;
                    queue.push(next);
                }
            }
        }
        
        return accessible_count;
    }
};

int main() {
    std::cout << "=== SIMPLE SNAKE AI WITH FIXED COLLISION DETECTION ===" << std::endl;
    std::cout << "Testing the collision detection fix" << std::endl;
    std::cout << std::endl;
    
    try {
        SimpleSnakeAI ai;
        ai.runDemo();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}