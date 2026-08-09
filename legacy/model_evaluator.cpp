#include "neural_network.h"
#include "snake_logic.h"
#include <iostream>

// Evaluate your 97% model for high score capability
class ModelEvaluator {
public:
    ModelEvaluator() : network(8, 64, 4) {}
    
    void evaluateModel(const std::string& model_path, int games = 1000) {
        std::cout << "=== EVALUATING MODEL FOR HIGH SCORES ===" << std::endl;
        std::cout << "Model: " << model_path << std::endl;
        std::cout << "Games: " << games << std::endl;
        
        try {
            network.load(model_path);
            std::cout << "Model loaded successfully!" << std::endl;
        } catch (const std::exception& e) {
            std::cout << "Failed to load model: " << e.what() << std::endl;
            return;
        }
        
        int total_score = 0;
        int high_scores = 0;
        int winning_games = 0;
        int max_score = 0;
        
        std::vector<int> score_distribution(21, 0); // Track scores 0-20+
        
        for (int game = 0; game < games; game++) {
            SnakeGame snake_game;
            snake_game.reset();
            
            int steps = 0;
            while (!snake_game.isGameOver() && steps < 2000) { // Allow long games
                auto state = getState(snake_game);
                auto action_tensor = network.getAction(state, 0.0f); // Pure exploitation
                int action = static_cast<int>(action_tensor.cpu().item<int64_t>());
                
                snake_game.setDirection(static_cast<Direction>(action));
                snake_game.update();
                steps++;
            }
            
            int score = snake_game.getScore();
            total_score += score;
            if (score >= 10) high_scores++;
            if (score >= 15) winning_games++;
            if (score > max_score) max_score = score;
            
            // Track distribution
            int bucket = (score >= 20) ? 20 : score;
            score_distribution[bucket]++;
            
            if (game % 100 == 0 && game > 0) {
                float current_avg = (float)total_score / (game + 1);
                std::cout << "Progress: " << game << " games | Avg: " << current_avg 
                         << " | Max so far: " << max_score << std::endl;
            }
        }
        
        float avg_score = (float)total_score / games;
        float high_rate = (float)high_scores / games * 100.0f;
        float win_rate = (float)winning_games / games * 100.0f;
        
        std::cout << std::endl << "=== COMPREHENSIVE EVALUATION ===" << std::endl;
        std::cout << "*** AVERAGE SCORE: " << avg_score << " foods ***" << std::endl;
        std::cout << "*** HIGH SCORES (10+): " << high_rate << "% of games ***" << std::endl;
        std::cout << "*** WINNING GAMES (15+): " << win_rate << "% of games ***" << std::endl;
        std::cout << "*** MAXIMUM SCORE: " << max_score << " foods ***" << std::endl;
        
        std::cout << "\\n=== SCORE DISTRIBUTION ===" << std::endl;
        for (int i = 0; i <= 20; i++) {
            if (score_distribution[i] > 0) {
                float percentage = (float)score_distribution[i] / games * 100.0f;
                if (i == 20) {
                    std::cout << "20+ foods: " << score_distribution[i] << " games (" << percentage << "%)" << std::endl;
                } else {
                    std::cout << i << " foods: " << score_distribution[i] << " games (" << percentage << "%)" << std::endl;
                }
            }
        }
        
        // Performance classification
        if (win_rate >= 25.0f) {
            std::cout << "\\n*** EXCELLENT! Model can achieve winning games! ***" << std::endl;
        } else if (high_rate >= 50.0f) {
            std::cout << "\\n*** VERY GOOD! Model regularly gets high scores! ***" << std::endl;
        } else if (avg_score >= 5.0f) {
            std::cout << "\\n*** GOOD! Model performs well consistently! ***" << std::endl;
        } else {
            std::cout << "\\n*** Model needs improvement for high scores ***" << std::endl;
        }
    }
    
private:
    SnakeNeuralNetwork network;
    
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

int main() {
    std::cout << "=== SNAKE MODEL EVALUATOR ===" << std::endl;
    std::cout << "Comprehensive evaluation of your trained models" << std::endl;
    std::cout << std::endl;
    
    try {
        ModelEvaluator evaluator;
        
        // Test your best model with comprehensive evaluation
        evaluator.evaluateModel("snake_extended_final_97percent.bin", 1000);
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}
