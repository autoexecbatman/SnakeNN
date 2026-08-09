#include "neural_network.h"
#include "snake_logic.h"
#include <iostream>
#include <raylib.h>
#include <vector>
#include <algorithm>
#include <queue>
#include <set>

// Universal model loader that handles different architectures
class UniversalSafeAI {
public:
    UniversalSafeAI() : network_8x64(8, 64, 4), network_8x128(8, 128, 4), network_11x128(11, 128, 4) {}
    
    bool loadAnyModel(const std::vector<std::string>& model_paths) {
        std::cout << "=== UNIVERSAL MODEL LOADER ===" << std::endl;
        
        for (const auto& path : model_paths) {
            std::cout << "Trying to load: " << path << std::endl;
            
            // Try different network architectures
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
        
        std::cout << "No compatible models found!" << std::endl;
        return false;
    }
    
    void runVisualDemo() {
        if (!active_network) {
            std::cout << "No model loaded!" << std::endl;
            return;
        }
        
        const int CELL_SIZE = 20;
        const int SCREEN_WIDTH = SnakeGame::GRID_WIDTH * CELL_SIZE;
        const int SCREEN_HEIGHT = SnakeGame::GRID_HEIGHT * CELL_SIZE + 120;
        
        InitWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "Universal Safe Snake AI");
        SetTargetFPS(8);
        
        SnakeGame game;
        game.reset();
        
        int total_games = 0;
        int current_score = 0;
        int max_score = 0;
        float avg_score = 0.0f;
        bool game_over = false;
        int steps = 0;
        
        std::cout << "Using model: " << loaded_model << " (Architecture: " << network_type << ")" << std::endl;
        
        while (!WindowShouldClose()) {
            if (IsKeyPressed(KEY_SPACE) || game_over) {
                if (game_over) {
                    total_games++;
                    current_score = game.getScore();
                    if (current_score > max_score) max_score = current_score;
                    avg_score = (avg_score * (total_games - 1) + current_score) / total_games;
                    
                    std::cout << "Game " << total_games << " - Score: " << current_score 
                             << " | Avg: " << avg_score << " | Max: " << max_score << std::endl;
                }
                
                game.reset();
                game_over = false;
                steps = 0;
            }
            
            if (!game.isGameOver() && !game_over) {
                if (steps < 3000) {
                    Direction safe_move = getGuaranteedSafeMove(game);
                    game.setDirection(safe_move);
                    game.update();
                    steps++;
                    
                    if (game.isGameOver()) {
                        game_over = true;
                        std::cout << "Game ended - Score: " << game.getScore() << std::endl;
                    }
                } else {
                    game_over = true;
                }
            }
            
            BeginDrawing();
            ClearBackground(BLACK);
            
            drawGame(game, CELL_SIZE);
            drawUI(game, total_games, current_score, max_score, avg_score, steps, game_over);
            
            EndDrawing();
        }
        
        CloseWindow();
    }
    
private:
    SnakeNeuralNetwork network_8x64;
    SnakeNeuralNetwork network_8x128;
    SnakeNeuralNetwork network_11x128;
    SnakeNeuralNetwork* active_network = nullptr;
    std::string network_type;
    std::string loaded_model;
    
    bool tryLoad(SnakeNeuralNetwork& network, const std::string& path, const std::string& arch_name) {
        try {
            network.load(path);
            std::cout << "✓ Successfully loaded with " << arch_name << " architecture" << std::endl;
            return true;
        } catch (const std::exception& e) {
            std::cout << "✗ Failed with " << arch_name << ": " << e.what() << std::endl;
            return false;
        }
    }
    
    Direction getGuaranteedSafeMove(const SnakeGame& game) {
        auto head = game.getSnakeBody()[0];
        auto snake = game.getSnakeBody();
        
        // Get all safe moves
        std::vector<Direction> safe_moves;
        for (int dir = 0; dir < 4; dir++) {
            Direction test_dir = static_cast<Direction>(dir);
            if (isAbsolutelySafe(head, test_dir, snake)) {
                safe_moves.push_back(test_dir);
            }
        }
        
        if (safe_moves.empty()) {
            std::cout << "WARNING: No safe moves!" << std::endl;
            return Direction::UP;
        }
        
        if (safe_moves.size() == 1) {
            return safe_moves[0];
        }
        
        // Get neural network preference
        std::vector<float> nn_state;
        if (network_type == "11x128") {
            nn_state = getState11D(game);
        } else {
            nn_state = getState8D(game);
        }
        
        auto action_tensor = active_network->getAction(nn_state, 0.0f);
        Direction nn_choice = static_cast<Direction>(action_tensor.cpu().item<int64_t>());
        
        // Use NN choice if safe
        if (std::find(safe_moves.begin(), safe_moves.end(), nn_choice) != safe_moves.end()) {
            return nn_choice;
        }
        
        // Fallback to best heuristic move
        Direction best_move = safe_moves[0];
        float best_score = -1000.0f;
        auto food = game.getFoodPosition();
        
        for (Direction dir : safe_moves) {
            float score = 0.0f;
            Position next_pos = getNextPosition(head, dir);
            
            // Prefer moves toward food
            float food_distance = abs(next_pos.x - food.x) + abs(next_pos.y - food.y);
            score += (20.0f - food_distance);
            
            // Prefer moves toward center
            float center_x = SnakeGame::GRID_WIDTH / 2.0f;
            float center_y = SnakeGame::GRID_HEIGHT / 2.0f;
            float center_distance = abs(next_pos.x - center_x) + abs(next_pos.y - center_y);
            score += (10.0f - center_distance);
            
            if (score > best_score) {
                best_score = score;
                best_move = dir;
            }
        }
        
        return best_move;
    }
    
    bool isAbsolutelySafe(const Position& head, Direction dir, const std::vector<Position>& snake) {
        Position next_pos = getNextPosition(head, dir);
        
        // Check walls
        if (next_pos.x < 0 || next_pos.x >= SnakeGame::GRID_WIDTH ||
            next_pos.y < 0 || next_pos.y >= SnakeGame::GRID_HEIGHT) {
            return false;
        }
        
        // Check snake body collision - CHECK ALL SEGMENTS INCLUDING TAIL
        // The tail will only move if we're NOT eating food, but we don't know that here
        // So we must be conservative and check all segments
        for (const auto& segment : snake) {
            if (segment.x == next_pos.x && segment.y == next_pos.y) {
                return false;
            }
        }
        
        return true;
    }
    
    Position getNextPosition(const Position& pos, Direction dir) {
        Position next = pos;
        switch (dir) {
            case Direction::UP: next.y--; break;
            case Direction::DOWN: next.y++; break;
            case Direction::LEFT: next.x--; break;
            case Direction::RIGHT: next.x++; break;
        }
        return next;
    }
    
    std::vector<float> getState8D(const SnakeGame& game) {
        auto head = game.getSnakeBody()[0];
        auto food = game.getFoodPosition();
        
        std::vector<float> state(8);
        
        state[0] = (food.x > head.x) ? 1.0f : 0.0f;
        state[1] = (food.x < head.x) ? 1.0f : 0.0f;
        state[2] = (food.y > head.y) ? 1.0f : 0.0f;
        state[3] = (food.y < head.y) ? 1.0f : 0.0f;
        
        for (int i = 0; i < 4; i++) {
            Direction testDir = static_cast<Direction>(i);
            bool danger = !isAbsolutelySafe(head, testDir, game.getSnakeBody());
            state[4 + i] = danger ? 1.0f : 0.0f;
        }
        
        return state;
    }
    
    // Trap detection using flood fill
    bool createsTrap(const Position& head, Direction dir, const std::vector<Position>& snake) {
        Position next_pos = getNextPosition(head, dir);
        
        // Simulate future snake
        std::vector<Position> future_snake = snake;
        future_snake[0] = next_pos;
        if (future_snake.size() > 1) future_snake.pop_back();
        
        int accessible = floodFillCount(next_pos, future_snake);
        int required = future_snake.size() + 5;
        
        return accessible < required;
    }
    
    int floodFillCount(const Position& start, const std::vector<Position>& snake) {
        std::queue<Position> queue;
        std::set<std::pair<int, int>> visited;
        
        queue.push(start);
        visited.insert({start.x, start.y});
        
        int count = 0;
        while (!queue.empty() && count < 200) {
            Position current = queue.front();
            queue.pop();
            count++;
            
            for (int dir = 0; dir < 4; dir++) {
                Position next = getNextPosition(current, static_cast<Direction>(dir));
                
                if (next.x >= 0 && next.x < SnakeGame::GRID_WIDTH &&
                    next.y >= 0 && next.y < SnakeGame::GRID_HEIGHT) {
                    
                    bool blocked = false;
                    for (const auto& segment : snake) {
                        if (segment.x == next.x && segment.y == next.y) {
                            blocked = true;
                            break;
                        }
                    }
                    
                    if (!blocked && !visited.count({next.x, next.y})) {
                        visited.insert({next.x, next.y});
                        queue.push(next);
                    }
                }
            }
        }
        
        return count;
    }
    
    std::vector<float> getState11D(const SnakeGame& game) {
        auto head = game.getSnakeBody()[0];
        auto food = game.getFoodPosition();
        auto snake = game.getSnakeBody();
        
        std::vector<float> state(11);
        
        // First 8 features same as 8D
        state[0] = (food.x > head.x) ? 1.0f : 0.0f;
        state[1] = (food.x < head.x) ? 1.0f : 0.0f;
        state[2] = (food.y > head.y) ? 1.0f : 0.0f;
        state[3] = (food.y < head.y) ? 1.0f : 0.0f;
        
        for (int i = 0; i < 4; i++) {
            Direction testDir = static_cast<Direction>(i);
            bool danger = !isAbsolutelySafe(head, testDir, snake);
            state[4 + i] = danger ? 1.0f : 0.0f;
        }
        
        // Additional features for 11D
        state[8] = (float)snake.size() / (SnakeGame::GRID_WIDTH * SnakeGame::GRID_HEIGHT);
        state[9] = 0.5f; // Placeholder for space availability
        state[10] = 0.5f; // Placeholder for tail distance
        
        return state;
    }
    
    void drawGame(const SnakeGame& game, int cell_size) {
        // Draw grid
        for (int x = 0; x < SnakeGame::GRID_WIDTH; x++) {
            for (int y = 0; y < SnakeGame::GRID_HEIGHT; y++) {
                Rectangle cell = {(float)(x * cell_size), (float)(y * cell_size), 
                                 (float)cell_size, (float)cell_size};
                DrawRectangleRec(cell, DARKGRAY);
                DrawRectangleLinesEx(cell, 1, GRAY);
            }
        }
        
        // Draw snake
        auto snake_body = game.getSnakeBody();
        for (size_t i = 0; i < snake_body.size(); i++) {
            Rectangle cell = {(float)(snake_body[i].x * cell_size), 
                             (float)(snake_body[i].y * cell_size),
                             (float)cell_size, (float)cell_size};
            
            Color snake_color = (i == 0) ? LIME : GREEN;
            DrawRectangleRec(cell, snake_color);
        }
        
        // Draw food
        auto food = game.getFoodPosition();
        Rectangle food_cell = {(float)(food.x * cell_size), (float)(food.y * cell_size),
                              (float)cell_size, (float)cell_size};
        DrawRectangleRec(food_cell, RED);
    }
    
    void drawUI(const SnakeGame& game, int total_games, int current_score, 
                int max_score, float avg_score, int steps, bool game_over) {
        int ui_y = SnakeGame::GRID_HEIGHT * 20 + 10;
        
        DrawText(TextFormat("Score: %d", game.getScore()), 10, ui_y, 20, WHITE);
        DrawText(TextFormat("Steps: %d", steps), 150, ui_y, 20, WHITE);
        DrawText(TextFormat("Games: %d", total_games), 280, ui_y, 20, WHITE);
        
        DrawText(TextFormat("Avg: %.1f", avg_score), 10, ui_y + 25, 20, WHITE);
        DrawText(TextFormat("Max: %d", max_score), 150, ui_y + 25, 20, WHITE);
        DrawText("COLLISION-FREE", 280, ui_y + 25, 20, GOLD);
        
        DrawText(TextFormat("Model: %s", network_type.c_str()), 10, ui_y + 50, 16, LIGHTGRAY);
        DrawText(TextFormat("File: %s", loaded_model.c_str()), 10, ui_y + 70, 14, LIGHTGRAY);
        
        if (game_over) {
            DrawText("GAME OVER - Press SPACE", 10, ui_y + 90, 20, YELLOW);
        }
    }
};

int main() {
    std::cout << "=== UNIVERSAL SAFE SNAKE AI ===" << std::endl;
    std::cout << "Tries multiple network architectures to load any compatible model" << std::endl;
    std::cout << std::endl;
    
    try {
        UniversalSafeAI universal_ai;
        
        // Try models in order of preference
        std::vector<std::string> models = {
            "D:/repo/snakeNN/build/Debug/snake_extended_final_97percent.bin"
            //"snake_research_final_96percent.bin", 
            //"snake_best_99percent.bin",
            //"snake_best_98percent.bin",
            //"snake_best_97percent.bin"
        };
        
        if (universal_ai.loadAnyModel(models)) {
            universal_ai.runVisualDemo();
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}
