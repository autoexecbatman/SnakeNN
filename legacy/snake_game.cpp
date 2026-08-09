#include "snake_logic.h"
#include <raylib.h>
#include <iostream>

const int CELL_SIZE = 30;
const int WINDOW_WIDTH = SnakeGame::GRID_WIDTH * CELL_SIZE;
const int WINDOW_HEIGHT = SnakeGame::GRID_HEIGHT * CELL_SIZE;

void drawGame(const SnakeGame& game) {
    BeginDrawing();
    ClearBackground(DARKGRAY);
    
    // Draw snake
    const auto& snake = game.getSnakeBody();
    for (size_t i = 0; i < snake.size(); i++) {
        Color color = (i == 0) ? DARKGREEN : GREEN; // Head is darker
        DrawRectangle(snake[i].x * CELL_SIZE, snake[i].y * CELL_SIZE, 
                     CELL_SIZE - 1, CELL_SIZE - 1, color);
    }
    
    // Draw food
    const auto& food = game.getFoodPosition();
    DrawRectangle(food.x * CELL_SIZE, food.y * CELL_SIZE, 
                 CELL_SIZE - 1, CELL_SIZE - 1, RED);
    
    // Draw score
    DrawText(TextFormat("Score: %d", game.getScore()), 10, 10, 20, WHITE);
    
    if (game.isGameOver()) {
        DrawText("GAME OVER! Press SPACE to restart", 
                WINDOW_WIDTH/2 - 150, WINDOW_HEIGHT/2, 20, WHITE);
    }
    
    EndDrawing();
}

int main() {
    InitWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "Snake Game");
    SetTargetFPS(10); // Snake speed
    
    SnakeGame game;
    
    while (!WindowShouldClose()) {
        // Handle input
        if (IsKeyPressed(KEY_UP)) game.setDirection(Direction::UP);
        if (IsKeyPressed(KEY_DOWN)) game.setDirection(Direction::DOWN);
        if (IsKeyPressed(KEY_LEFT)) game.setDirection(Direction::LEFT);
        if (IsKeyPressed(KEY_RIGHT)) game.setDirection(Direction::RIGHT);
        
        if (IsKeyPressed(KEY_SPACE) && game.isGameOver()) {
            game.reset();
        }
        
        // Update game
        if (!game.isGameOver()) {
            game.update();
        }
        
        // Render
        drawGame(game);
    }
    
    CloseWindow();
    return 0;
}
