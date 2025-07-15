#include "neural_network.h"
#include "snake_logic.h"
#include <iostream>
#include <raylib.h>

class VisualModelDemo {
public:
    VisualModelDemo(const std::string& model_path) : network(8, 64, 4) {  // Match optimized trainer
        network.load(model_path);
        std::cout << "Model loaded: " << model_path << std::endl;
    }
    
    void demo() {
        const int CELL_SIZE = 25;
        const int WINDOW_WIDTH = SnakeGame::GRID_WIDTH * CELL_SIZE;
        const int WINDOW_HEIGHT = SnakeGame::GRID_HEIGHT * CELL_SIZE + 100;
        
        InitWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "Snake AI Demo - Trained Neural Network");
        SetTargetFPS(8);  // Slow enough to watch
        
        std::cout << "\n=== VISUAL DEMO CONTROLS ===" << std::endl;
        std::cout << "SPACE: Pause/Resume" << std::endl;
        std::cout << "R: Reset game" << std::endl;
        std::cout << "UP/DOWN: Adjust speed" << std::endl;
        std::cout << "ESC: Exit" << std::endl;
        std::cout << "==============================\n" << std::endl;
        
        SnakeGame game;
        game.reset();
        
        int episode = 1;
        int total_score = 0;
        int games_played = 0;
        bool paused = false;
        int fps = 8;
        
        while (!WindowShouldClose()) {
            // Handle input
            if (IsKeyPressed(KEY_SPACE)) {
                paused = !paused;
            }
            if (IsKeyPressed(KEY_R)) {
                game.reset();
                episode++;
            }
            if (IsKeyPressed(KEY_UP)) {
                fps = std::min(60, fps + 2);
                SetTargetFPS(fps);
            }
            if (IsKeyPressed(KEY_DOWN)) {
                fps = std::max(1, fps - 2);
                SetTargetFPS(fps);
            }
            
            // Game logic (if not paused)
            if (!paused && !game.isGameOver()) {
                auto state = getSimplifiedState(game);
                auto action_tensor = network.getAction(state, 0.0f);
                int action = static_cast<int>(action_tensor.cpu().item<int64_t>());
                
                game.setDirection(static_cast<Direction>(action));
                game.update();
            }
            
            // Auto-reset when game over
            if (game.isGameOver()) {
                total_score += game.getScore();
                games_played++;
                
                // Wait a bit, then reset
                static int wait_frames = 0;
                wait_frames++;
                if (wait_frames > fps) {  // Wait 1 second
                    game.reset();
                    episode++;
                    wait_frames = 0;
                }
            }
            
            // Draw everything
            BeginDrawing();
            ClearBackground(DARKGRAY);
            
            // Draw snake
            const auto& snake = game.getSnakeBody();
            for (size_t i = 0; i < snake.size(); i++) {
                Color color = (i == 0) ? LIME : GREEN;  // Head is brighter
                DrawRectangle(snake[i].x * CELL_SIZE, snake[i].y * CELL_SIZE, 
                             CELL_SIZE - 1, CELL_SIZE - 1, color);
            }
            
            // Draw food
            const auto& food = game.getFoodPosition();
            DrawRectangle(food.x * CELL_SIZE, food.y * CELL_SIZE, 
                         CELL_SIZE - 1, CELL_SIZE - 1, RED);
            
            // Draw UI
            DrawText(("Episode: " + std::to_string(episode)).c_str(), 10, WINDOW_HEIGHT - 90, 20, WHITE);
            DrawText(("Score: " + std::to_string(game.getScore())).c_str(), 10, WINDOW_HEIGHT - 70, 20, WHITE);
            
            if (games_played > 0) {
                float avg = (float)total_score / games_played;
                DrawText(("Average: " + std::to_string(avg)).c_str(), 10, WINDOW_HEIGHT - 50, 20, WHITE);
            }
            
            DrawText(("FPS: " + std::to_string(fps)).c_str(), 10, WINDOW_HEIGHT - 30, 20, WHITE);
            
            if (paused) {
                DrawText("PAUSED - Press SPACE", WINDOW_WIDTH/2 - 100, WINDOW_HEIGHT/2, 20, YELLOW);
            }
            
            if (game.isGameOver()) {
                DrawText("GAME OVER", WINDOW_WIDTH/2 - 60, WINDOW_HEIGHT/2 - 40, 20, RED);
                DrawText("Auto-restarting...", WINDOW_WIDTH/2 - 80, WINDOW_HEIGHT/2 - 20, 16, WHITE);
            }
            
            EndDrawing();
        }
        
        CloseWindow();
        
        std::cout << "\n=== DEMO COMPLETE ===" << std::endl;
        std::cout << "Episodes played: " << games_played << std::endl;
        if (games_played > 0) {
            std::cout << "Average score: " << (float)total_score / games_played << std::endl;
        }
    }
    
private:
    SnakeNeuralNetwork network;
    
    std::vector<float> getSimplifiedState(const SnakeGame& game) {
        auto head = game.getSnakeBody()[0];
        auto food = game.getFoodPosition();
        
        std::vector<float> state(8);
        
        // Food direction (4 features)
        state[0] = (food.x > head.x) ? 1.0f : 0.0f;  // Food right
        state[1] = (food.x < head.x) ? 1.0f : 0.0f;  // Food left  
        state[2] = (food.y > head.y) ? 1.0f : 0.0f;  // Food down
        state[3] = (food.y < head.y) ? 1.0f : 0.0f;  // Food up
        
        // Immediate danger (4 features)
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
                          testPos.y < 0 || testPos.y >= SnakeGame::GRID_HEIGHT ||
                          checkCollision(testPos, game.getSnakeBody()));
            state[4 + i] = danger ? 1.0f : 0.0f;
        }
        
        return state;
    }
    
    bool checkCollision(const Position& pos, const std::vector<Position>& snake) {
        for (const auto& segment : snake) {
            if (segment.x == pos.x && segment.y == pos.y) {
                return true;
            }
        }
        return false;
    }
};

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <model_file.bin>" << std::endl;
        std::cout << "Example: " << argv[0] << " snake_model_3270percent.bin" << std::endl;
        return -1;
    }
    
    try {
        VisualModelDemo demo(argv[1]);
        demo.demo();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}
