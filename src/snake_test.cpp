#include "neural_network.h"
#include "snake_logic.h"
#include <raylib.h>
#include <iostream>
#include <memory>

const int CELL_SIZE = 30;
const int WINDOW_WIDTH = SnakeGame::GRID_WIDTH * CELL_SIZE;
const int WINDOW_HEIGHT = SnakeGame::GRID_HEIGHT * CELL_SIZE + 100; // Extra space for UI

void drawGame(const SnakeGame& game, bool aiMode, int episode = 0) {
    BeginDrawing();
    ClearBackground(DARKGRAY);
    
    // Draw snake
    const auto& snake = game.getSnakeBody();
    for (size_t i = 0; i < snake.size(); i++) {
        Color color = (i == 0) ? DARKGREEN : GREEN;
        DrawRectangle(snake[i].x * CELL_SIZE, snake[i].y * CELL_SIZE, 
                     CELL_SIZE - 1, CELL_SIZE - 1, color);
    }
    
    // Draw food
    const auto& food = game.getFoodPosition();
    DrawRectangle(food.x * CELL_SIZE, food.y * CELL_SIZE, 
                 CELL_SIZE - 1, CELL_SIZE - 1, RED);
    
    // Draw UI
    DrawText(TextFormat("Score: %d", game.getScore()), 10, WINDOW_HEIGHT - 90, 20, WHITE);
    if (aiMode) {
        DrawText(TextFormat("Episode: %d", episode), 10, WINDOW_HEIGHT - 60, 20, WHITE);
        DrawText("AI Playing - Press ESC to exit", 10, WINDOW_HEIGHT - 30, 20, WHITE);
    } else {
        DrawText("Press SPACE to start AI test", 10, WINDOW_HEIGHT - 60, 20, WHITE);
        DrawText("Arrow keys to play manually", 10, WINDOW_HEIGHT - 30, 20, WHITE);
    }
    
    if (game.isGameOver()) {
        DrawText("GAME OVER!", WINDOW_WIDTH/2 - 50, WINDOW_HEIGHT/2, 20, WHITE);
    }
    
    EndDrawing();
}

int main() {
    std::cout << "=== Snake AI Test ===" << std::endl;
    
    // Check for trained model
    auto network = std::make_shared<SnakeNeuralNetwork>(14, 128, 4);
    
    try {
        network->load("snake_model.pt");
        std::cout << "Loaded trained model successfully!" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "Could not load model: " << e.what() << std::endl;
        std::cout << "Using untrained network..." << std::endl;
    }
    
    InitWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "Snake AI Test");
    SetTargetFPS(10);
    
    SnakeGame game;
    bool aiMode = false;
    int episode = 0;
    
    while (!WindowShouldClose()) {
        // Handle input
        if (IsKeyPressed(KEY_SPACE)) {
            aiMode = !aiMode;
            game.reset();
            episode = 0;
        }
        
        if (!aiMode) {
            // Manual control
            if (IsKeyPressed(KEY_UP)) game.setDirection(Direction::UP);
            if (IsKeyPressed(KEY_DOWN)) game.setDirection(Direction::DOWN);
            if (IsKeyPressed(KEY_LEFT)) game.setDirection(Direction::LEFT);
            if (IsKeyPressed(KEY_RIGHT)) game.setDirection(Direction::RIGHT);
        } else {
            // AI control
            if (!game.isGameOver()) {
                auto state = game.getGameState();
                auto action_tensor = network->getAction(state, 0.0f); // No exploration
                int action = action_tensor.item<int>();
                Direction dir = static_cast<Direction>(action);
                game.setDirection(dir);
            }
        }
        
        // Update game
        if (!game.isGameOver()) {
            game.update();
        } else if (aiMode) {
            // Auto-restart in AI mode
            game.reset();
            episode++;
        }
        
        // Render
        drawGame(game, aiMode, episode);
    }
    
    CloseWindow();
    return 0;
}
