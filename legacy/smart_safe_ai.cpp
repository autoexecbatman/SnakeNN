#include "neural_network.h"
#include "snake_logic.h"
#include <iostream>
#include <raylib.h>
#include <vector>
#include <algorithm>
#include <queue>
#include <set>

// Smart AI with trap avoidance using flood fill
class SmartSafeAI {
public:
    SmartSafeAI() : network_8x64(8, 64, 4), network_8x128(8, 128, 4), network_11x128(11, 128, 4) {}
    
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
    
    void runVisualDemo() {
        const int CELL_SIZE = 20;
        const int SCREEN_WIDTH = SnakeGame::GRID_WIDTH * CELL_SIZE;
        const int SCREEN_HEIGHT = SnakeGame::GRID_HEIGHT * CELL_SIZE + 140;
        
        InitWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "Smart Safe Snake AI - Trap Avoidance");
        SetTargetFPS(6);
        
        SnakeGame game;
        game.reset();
        
        int total_games = 0;
        int current_score = 0;
        int max_score = 0;
        float avg_score = 0.0f;
        bool game_over = false;
        int steps = 0;
        int trap_deaths = 0;
        
        std::cout << "Smart Safe AI with trap avoidance using flood fill" << std::endl;
        
        while (!WindowShouldClose()) {
            if (IsKeyPressed(KEY_SPACE) || game_over) {
                if (game_over) {
                    total_games++;
                    current_score = game.getScore();
                    if (current_score > max_score) max_score = current_score;
                    avg_score = (avg_score * (total_games - 1) + current_score) / total_games;
                    
                    std::cout << "Game " << total_games << " - Score: " << current_score 
                             << " | Avg: " << avg_score << " | Max: " << max_score;
                    
                    if (steps >= 2500) {
                        std::cout << " | TIMEOUT";
                    } else if (hasAnySafeMove(game.getSnakeBody()[0], game.getSnakeBody())) {
                        std::cout << " | TRAPPED";
                        trap_deaths++;
                    }
                    std::cout << std::endl;
                }
                
                game.reset();
                game_over = false;
                steps = 0;
            }
            
            if (!game.isGameOver() && !game_over) {
                if (steps < 2500) {
                    Direction smart_move = getSmartSafeMove(game);
                    game.setDirection(smart_move);
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
            
            drawGame(game, CELL_SIZE);
            drawUI(game, total_games, current_score, max_score, avg_score, steps, game_over, trap_deaths);
            
            EndDrawing();
        }
        
        CloseWindow();
    }
    
private:
    SnakeNeuralNetwork network_8x64, network_8x128, network_11x128;
    SnakeNeuralNetwork* active_network = nullptr;
    std::string network_type, loaded_model;
    
    bool tryLoad(SnakeNeuralNetwork& network, const std::string& path, const std::string& arch_name) {
        try {
            network.load(path);
            std::cout << "✓ Loaded " << path << " with " << arch_name << " architecture" << std::endl;
            return true;
        } catch (const std::exception& e) {
            return false;
        }
    }
    
    Direction getSmartSafeMove(const SnakeGame& game) {
        auto head = game.getSnakeBody()[0];
        auto snake = game.getSnakeBody();
        auto food = game.getFoodPosition();
        
        // Get all immediately safe moves
        std::vector<Direction> safe_moves;
        for (int dir = 0; dir < 4; dir++) {
            Direction test_dir = static_cast<Direction>(dir);
            if (isImmediatelySafe(head, test_dir, snake)) {
                safe_moves.push_back(test_dir);
            }
        }
        
        if (safe_moves.empty()) {
            std::cout << "NO SAFE MOVES!" << std::endl;
            return Direction::UP;\n        }
        
        if (safe_moves.size() == 1) {
            return safe_moves[0];
        }
        
        // Filter out moves that create traps
        std::vector<Direction> non_trap_moves;
        for (Direction dir : safe_moves) {
            if (!createsTrap(head, dir, snake)) {
                non_trap_moves.push_back(dir);
            }
        }
        
        // If all moves create traps, choose the one with most space
        std::vector<Direction>* moves_to_consider = &non_trap_moves;
        if (non_trap_moves.empty()) {
            std::cout << "All moves create traps - choosing best trap" << std::endl;
            moves_to_consider = &safe_moves;
        }
        
        // Choose best move from remaining options
        Direction best_move = (*moves_to_consider)[0];
        float best_score = -10000.0f;
        
        for (Direction dir : *moves_to_consider) {
            float score = evaluateMove(head, dir, snake, food);
            if (score > best_score) {
                best_score = score;
                best_move = dir;
            }
        }
        
        return best_move;
    }
    
    bool isImmediatelySafe(const Position& head, Direction dir, const std::vector<Position>& snake) {
        Position next_pos = getNextPosition(head, dir);
        
        // Check walls
        if (next_pos.x < 0 || next_pos.x >= SnakeGame::GRID_WIDTH ||
            next_pos.y < 0 || next_pos.y >= SnakeGame::GRID_HEIGHT) {
            return false;
        }
        
        // Check all snake segments (conservative approach)
        for (const auto& segment : snake) {
            if (segment.x == next_pos.x && segment.y == next_pos.y) {
                return false;
            }
        }
        
        return true;
    }
    
    bool createsTrap(const Position& head, Direction dir, const std::vector<Position>& snake) {
        Position next_pos = getNextPosition(head, dir);
        
        // Simulate the move by creating new snake position
        std::vector<Position> future_snake = snake;
        future_snake[0] = next_pos; // Move head
        // Remove tail (assume no food eaten for trap detection)
        if (future_snake.size() > 1) {
            future_snake.pop_back();
        }
        
        // Use flood fill to count accessible spaces from new position
        int accessible_spaces = floodFillCount(next_pos, future_snake);
        int required_spaces = future_snake.size() + 3; // Snake size + some buffer
        
        return accessible_spaces < required_spaces;
    }
    
    int floodFillCount(const Position& start, const std::vector<Position>& snake) {
        std::queue<Position> queue;
        std::set<std::pair<int, int>> visited;
        
        queue.push(start);
        visited.insert({start.x, start.y});
        
        int count = 0;
        while (!queue.empty() && count < 200) { // Limit to prevent infinite loops
            Position current = queue.front();
            queue.pop();
            count++;
            
            for (int dir = 0; dir < 4; dir++) {
                Position next = getNextPosition(current, static_cast<Direction>(dir));
                
                if (isValidPosition(next) && !isSnakePosition(next, snake) &&
                    !visited.count({next.x, next.y})) {
                    visited.insert({next.x, next.y});
                    queue.push(next);
                }
            }
        }
        
        return count;
    }
    
    float evaluateMove(const Position& head, Direction dir, const std::vector<Position>& snake, const Position& food) {
        Position next_pos = getNextPosition(head, dir);
        float score = 0.0f;
        
        // Distance to food (closer is better)
        float food_distance = abs(next_pos.x - food.x) + abs(next_pos.y - food.y);
        score += (20.0f - food_distance);
        
        // Available space (more is better)
        std::vector<Position> future_snake = snake;
        future_snake[0] = next_pos;
        if (future_snake.size() > 1) future_snake.pop_back();
        
        int space = floodFillCount(next_pos, future_snake);
        score += space * 0.5f;
        
        // Prefer center positions (avoid edges)
        float center_x = SnakeGame::GRID_WIDTH / 2.0f;
        float center_y = SnakeGame::GRID_HEIGHT / 2.0f;
        float center_distance = abs(next_pos.x - center_x) + abs(next_pos.y - center_y);
        score += (10.0f - center_distance) * 0.2f;
        
        return score;
    }
    
    bool hasAnySafeMove(const Position& head, const std::vector<Position>& snake) {
        for (int dir = 0; dir < 4; dir++) {
            if (isImmediatelySafe(head, static_cast<Direction>(dir), snake)) {
                return true;
            }
        }
        return false;
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
    
    bool isValidPosition(const Position& pos) {
        return pos.x >= 0 && pos.x < SnakeGame::GRID_WIDTH &&
               pos.y >= 0 && pos.y < SnakeGame::GRID_HEIGHT;
    }
    
    bool isSnakePosition(const Position& pos, const std::vector<Position>& snake) {
        for (const auto& segment : snake) {
            if (segment.x == pos.x && segment.y == pos.y) {
                return true;
            }
        }
        return false;
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
                int max_score, float avg_score, int steps, bool game_over, int trap_deaths) {
        int ui_y = SnakeGame::GRID_HEIGHT * 20 + 10;
        
        DrawText(TextFormat("Score: %d", game.getScore()), 10, ui_y, 20, WHITE);
        DrawText(TextFormat("Steps: %d", steps), 150, ui_y, 20, WHITE);
        DrawText(TextFormat("Games: %d", total_games), 280, ui_y, 20, WHITE);
        
        DrawText(TextFormat("Avg: %.1f", avg_score), 10, ui_y + 25, 20, WHITE);
        DrawText(TextFormat("Max: %d", max_score), 150, ui_y + 25, 20, WHITE);
        DrawText("TRAP-AWARE AI", 280, ui_y + 25, 20, GOLD);
        
        DrawText(TextFormat("Trap deaths: %d/%d", trap_deaths, total_games), 10, ui_y + 50, 16, ORANGE);
        DrawText(TextFormat("Model: %s", network_type.c_str()), 200, ui_y + 50, 16, LIGHTGRAY);
        
        if (game_over) {
            DrawText("GAME OVER - Press SPACE", 10, ui_y + 75, 20, YELLOW);
        }
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
            bool danger = !isImmediatelySafe(head, testDir, game.getSnakeBody());
            state[4 + i] = danger ? 1.0f : 0.0f;
        }
        
        return state;
    }
};

int main() {
    std::cout << "=== SMART SAFE SNAKE AI ===" << std::endl;
    std::cout << "Trap avoidance using flood fill space analysis" << std::endl;
    
    try {
        SmartSafeAI smart_ai;
        
        std::vector<std::string> models = {
            "D:/repo/snakeNN/build/Debug/snake_extended_final_97percent.bin",
            "D:/repo/snakeNN/build/Debug/snake_research_final_96percent.bin", 
            "D:/repo/snakeNN/build/Debug/snake_best_99percent.bin"
        };
        
        if (smart_ai.loadAnyModel(models)) {
            smart_ai.runVisualDemo();
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}
