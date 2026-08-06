#include "neural_network.h"
#include "snake_logic.h"
#include <iostream>
#include <raylib.h>
#include <chrono>
#include <thread>

// Visual hybrid AI demo to verify actual performance
class VisualHybridDemo {
public:
    VisualHybridDemo() : network(8, 64, 4) {}
    
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
        const int SCREEN_HEIGHT = SnakeGame::GRID_HEIGHT * CELL_SIZE + 100; // Extra space for UI
        
        InitWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "Hybrid Snake AI - Visual Demo");
        SetTargetFPS(10); // Slow enough to see moves
        
        SnakeGame game;
        game.reset();
        
        int total_games = 0;
        int current_score = 0;
        int max_score = 0;
        float avg_score = 0.0f;
        bool game_over = false;
        int steps = 0;
        
        std::cout << "=== VISUAL HYBRID AI DEMO ===" << std::endl;
        std::cout << "Press SPACE to restart game" << std::endl;
        std::cout << "Press ESC to quit" << std::endl;
        
        while (!WindowShouldClose()) {
            // Input handling
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
            
            // Game logic
            if (!game.isGameOver() && !game_over) {
                if (steps < 2000) { // Limit steps to prevent infinite games
                    Direction best_move = getBestMove(game);
                    game.setDirection(best_move);
                    game.update();
                    steps++;
                    
                    if (game.isGameOver()) {
                        game_over = true;
                    }
                } else {
                    game_over = true; // Force end if too many steps
                }
            }
            
            // Rendering
            BeginDrawing();
            ClearBackground(BLACK);
            
            // Draw game
            drawGame(game, CELL_SIZE);
            
            // Draw UI
            drawUI(game, total_games, current_score, max_score, avg_score, steps, game_over);
            
            EndDrawing();
        }
        
        CloseWindow();
    }
    
private:
    SnakeNeuralNetwork network;
    
    Direction getBestMove(const SnakeGame& game) {
        // Same hybrid logic as before but let's see what it actually does
        auto head = game.getSnakeBody()[0];
        auto food = game.getFoodPosition();
        auto snake = game.getSnakeBody();
        
        // Get neural network suggestion
        auto nn_state = getState(game);
        auto action_tensor = network.getAction(nn_state, 0.0f);
        Direction nn_choice = static_cast<Direction>(action_tensor.cpu().item<int64_t>());
        
        // Simple safety check - if NN choice is safe, use it
        if (isSafeMove(head, nn_choice, snake)) {
            return nn_choice;
        }
        
        // Otherwise find any safe move
        for (int dir = 0; dir < 4; dir++) {
            Direction test_dir = static_cast<Direction>(dir);
            if (isSafeMove(head, test_dir, snake)) {
                return test_dir;
            }
        }
        
        return Direction::UP; // Should never reach here
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
            
            Color snake_color = (i == 0) ? LIME : GREEN; // Head is brighter
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
        
        if (game_over) {
            DrawText("GAME OVER - Press SPACE to restart", 10, ui_y + 50, 20, YELLOW);
        }
        
        DrawText("Press ESC to quit", 10, ui_y + 75, 16, LIGHTGRAY);
    }
    
    bool isSafeMove(const Position& head, Direction dir, const std::vector<Position>& snake) {
        Position next_pos = head;
        switch (dir) {
            case Direction::UP: next_pos.y--; break;
            case Direction::DOWN: next_pos.y++; break;
            case Direction::LEFT: next_pos.x--; break;
            case Direction::RIGHT: next_pos.x++; break;
        }
        
        // Check bounds
        if (next_pos.x < 0 || next_pos.x >= SnakeGame::GRID_WIDTH ||
            next_pos.y < 0 || next_pos.y >= SnakeGame::GRID_HEIGHT) {
            return false;
        }
        
        // Check snake collision
        for (const auto& segment : snake) {
            if (segment.x == next_pos.x && segment.y == next_pos.y) {
                return false;
            }
        }
        
        return true;
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
            Position testPos = head;
            
            switch (testDir) {
                case Direction::UP: testPos.y--; break;
                case Direction::DOWN: testPos.y++; break;
                case Direction::LEFT: testPos.x--; break;
                case Direction::RIGHT: testPos.x++; break;
            }
            
            bool danger = (testPos.x < 0 || testPos.x >= SnakeGame::GRID_WIDTH ||
                          testPos.y < 0 || testPos.y >= SnakeGame::GRID_HEIGHT);
            
            if (!danger) {
                for (const auto& segment : game.getSnakeBody()) {
                    if (segment.x == testPos.x && segment.y == testPos.y) {
                        danger = true;
                        break;
                    }
                }
            }
            
            state[4 + i] = danger ? 1.0f : 0.0f;
        }
        
        return state;
    }
};

int main() {
    std::cout << "=== VISUAL HYBRID AI DEMO ===" << std::endl;
    std::cout << "Watch the AI play to verify actual performance" << std::endl;
    std::cout << std::endl;
    
    try {
        VisualHybridDemo demo;
        
        if (demo.loadModel("snake_research_final_96percent.bin")) {
            demo.runVisualDemo();
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}
