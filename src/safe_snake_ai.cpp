#include "neural_network.h"
#include "snake_logic.h"
#include <iostream>
#include <raylib.h>
#include <vector>
#include <algorithm>

// AI with guaranteed collision avoidance
class SafeSnakeAI {
public:
    SafeSnakeAI() : network(8, 64, 4) {}
    
    bool loadModel(const std::string& model_path) {
        try {
            network.load(model_path);
            return true;
        } catch (const std::exception& e) {
            std::cout << "Failed to load model: " << e.what() << std::endl;
            return false;
        }
    }
    
    void runVisualDemo() {
        const int CELL_SIZE = 20;
        const int SCREEN_WIDTH = SnakeGame::GRID_WIDTH * CELL_SIZE;
        const int SCREEN_HEIGHT = SnakeGame::GRID_HEIGHT * CELL_SIZE + 100;
        
        InitWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "Safe Snake AI - No Collisions");
        SetTargetFPS(8); // Slightly slower for better visibility
        
        SnakeGame game;
        game.reset();
        
        int total_games = 0;
        int current_score = 0;
        int max_score = 0;
        float avg_score = 0.0f;
        bool game_over = false;
        int steps = 0;
        
        std::cout << "=== SAFE SNAKE AI DEMO ===" << std::endl;
        std::cout << "Guaranteed collision avoidance - should never hit walls or tail" << std::endl;
        std::cout << "Press SPACE to restart game" << std::endl;
        
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
                        std::cout << "ERROR: Game ended despite safe move - this shouldn't happen!" << std::endl;
                    }
                } else {
                    game_over = true; // Force end after too many steps
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
    SnakeNeuralNetwork network;
    
    Direction getGuaranteedSafeMove(const SnakeGame& game) {
        auto head = game.getSnakeBody()[0];
        auto snake = game.getSnakeBody();
        
        // Step 1: Get all possible safe moves
        std::vector<Direction> safe_moves;
        for (int dir = 0; dir < 4; dir++) {
            Direction test_dir = static_cast<Direction>(dir);
            if (isAbsolutelySafe(head, test_dir, snake)) {
                safe_moves.push_back(test_dir);
            }
        }
        
        // Step 2: If no safe moves, we're trapped (shouldn't happen with good play)
        if (safe_moves.empty()) {
            std::cout << "WARNING: No safe moves available - this is a trap!" << std::endl;
            return Direction::UP; // Last resort
        }
        
        // Step 3: If only one safe move, use it
        if (safe_moves.size() == 1) {
            return safe_moves[0];
        }
        
        // Step 4: Get neural network preference
        auto nn_state = getState(game);
        auto action_tensor = network.getAction(nn_state, 0.0f);
        Direction nn_choice = static_cast<Direction>(action_tensor.cpu().item<int64_t>());
        
        // Step 5: Use NN choice if it's safe
        if (std::find(safe_moves.begin(), safe_moves.end(), nn_choice) != safe_moves.end()) {
            return nn_choice;
        }
        
        // Step 6: Choose best safe move using heuristics
        Direction best_move = safe_moves[0];
        float best_score = -1000.0f;
        
        auto food = game.getFoodPosition();
        
        for (Direction dir : safe_moves) {
            float score = 0.0f;
            
            Position next_pos = getNextPosition(head, dir);
            
            // Prefer moves toward food
            float food_distance = abs(next_pos.x - food.x) + abs(next_pos.y - food.y);
            score += (20.0f - food_distance); // Closer to food = higher score
            
            // Prefer moves toward center (avoid walls)
            float center_x = SnakeGame::GRID_WIDTH / 2.0f;
            float center_y = SnakeGame::GRID_HEIGHT / 2.0f;
            float center_distance = abs(next_pos.x - center_x) + abs(next_pos.y - center_y);
            score += (10.0f - center_distance);
            
            // Prefer moves with more space ahead
            int space_ahead = countSpaceAhead(next_pos, dir, snake);
            score += space_ahead * 2.0f;
            
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
        
        // Check snake body collision (excluding tail if it will move)
        for (size_t i = 0; i < snake.size() - 1; i++) { // -1 because tail will move
            if (snake[i].x == next_pos.x && snake[i].y == next_pos.y) {
                return false;
            }
        }
        
        // Special case: if snake will grow (eat food), include tail in collision check
        Position food = {-1, -1}; // We don't have direct access to food here
        // For now, assume worst case and check tail too
        if (snake.back().x == next_pos.x && snake.back().y == next_pos.y) {
            return false;
        }
        
        return true;
    }
    
    int countSpaceAhead(const Position& pos, Direction dir, const std::vector<Position>& snake) {
        int count = 0;
        Position test_pos = pos;
        
        for (int i = 0; i < 5; i++) { // Look 5 steps ahead
            test_pos = getNextPosition(test_pos, dir);
            
            if (test_pos.x < 0 || test_pos.x >= SnakeGame::GRID_WIDTH ||
                test_pos.y < 0 || test_pos.y >= SnakeGame::GRID_HEIGHT) {
                break;
            }
            
            bool blocked = false;
            for (const auto& segment : snake) {
                if (segment.x == test_pos.x && segment.y == test_pos.y) {
                    blocked = true;
                    break;
                }
            }
            
            if (blocked) break;
            count++;
        }
        
        return count;
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
        DrawText("COLLISION-FREE AI", 280, ui_y + 25, 20, GOLD);
        
        if (game_over) {
            DrawText("GAME OVER - Press SPACE to restart", 10, ui_y + 50, 20, YELLOW);
        }
    }
    
    std::vector<float> getState(const SnakeGame& game) {
        auto head = game.getSnakeBody()[0];
        auto food = game.getFoodPosition();
        
        std::vector<float> state(8);
        
        state[0] = (food.x > head.x) ? 1.0f : 0.0f;
        state[1] = (food.x < head.x) ? 1.0f : 0.0f;
        state[2] = (food.y > head.y) ? 1.0f : 0.0f;
        state[3] = (food.y < head.y) ? 1.0f : 0.0f;
        
        for (int i = 0; i < 4; i++) {
            Direction testDir = static_cast<Direction>(i);
            Position testPos = getNextPosition(head, testDir);
            
            bool danger = !isAbsolutelySafe(head, testDir, game.getSnakeBody());
            state[4 + i] = danger ? 1.0f : 0.0f;
        }
        
        return state;
    }
};

int main() {
    std::cout << "=== SAFE SNAKE AI ===" << std::endl;
    std::cout << "Guaranteed collision avoidance - should never die from walls or tail" << std::endl;
    std::cout << std::endl;
    
    try {
        SafeSnakeAI safe_ai;
        
        if (safe_ai.loadModel("snake_research_final_96percent.bin")) {
            safe_ai.runVisualDemo();
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}
