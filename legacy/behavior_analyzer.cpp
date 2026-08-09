#include "neural_network.h"
#include "snake_logic.h"
#include <iostream>
#include <fstream>

class SnakeBehaviorAnalyzer {
public:
    SnakeBehaviorAnalyzer(const std::string& model_path) : network(8, 64, 4) {
        network.load(model_path);
        std::cout << "Analyzing model: " << model_path << std::endl;
    }
    
    void analyzeDetailedBehavior(int games = 50) {
        std::cout << "\n=== DETAILED BEHAVIOR ANALYSIS ===" << std::endl;
        
        int total_score = 0;
        int games_1_food = 0, games_2_4_food = 0, games_5_9_food = 0, games_10_plus = 0;
        int games_died_early = 0;
        
        std::vector<int> score_distribution;
        std::vector<int> game_lengths;
        
        for (int game = 0; game < games; game++) {
            SnakeGame snake_game;
            snake_game.reset();
            
            int steps = 0;
            int foods_eaten = 0;
            bool died_from_wall = false;
            bool died_from_self = false;
            
            std::cout << "\n--- Game " << (game + 1) << " ---" << std::endl;
            
            while (!snake_game.isGameOver() && steps < 2000) {
                auto state = getState(snake_game);
                auto action_tensor = network.getAction(state, 0.0f);
                int action = static_cast<int>(action_tensor.cpu().item<int64_t>());
                
                int prev_score = snake_game.getScore();
                auto prev_head = snake_game.getSnakeBody()[0];
                
                snake_game.setDirection(static_cast<Direction>(action));
                bool continues = snake_game.update();
                
                int new_score = snake_game.getScore();
                if (new_score > prev_score) {
                    foods_eaten++;
                    std::cout << "Food " << foods_eaten << " eaten at step " << steps << std::endl;
                }
                
                steps++;
                
                if (!continues) {
                    // Analyze death cause
                    auto final_head = prev_head;
                    Direction final_dir = static_cast<Direction>(action);
                    
                    switch (final_dir) {
                        case Direction::UP: final_head.y--; break;
                        case Direction::DOWN: final_head.y++; break;
                        case Direction::LEFT: final_head.x--; break;
                        case Direction::RIGHT: final_head.x++; break;
                    }
                    
                    if (final_head.x < 0 || final_head.x >= SnakeGame::GRID_WIDTH ||
                        final_head.y < 0 || final_head.y >= SnakeGame::GRID_HEIGHT) {
                        died_from_wall = true;
                        std::cout << "DIED: Hit wall at step " << steps << std::endl;
                    } else {
                        died_from_self = true;
                        std::cout << "DIED: Hit self at step " << steps << std::endl;
                    }
                    break;
                }
            }
            
            int final_score = snake_game.getScore();
            total_score += final_score;
            score_distribution.push_back(final_score);
            game_lengths.push_back(steps);
            
            std::cout << "Final score: " << final_score << ", Steps: " << steps << std::endl;
            
            // Categorize performance
            if (final_score == 1) games_1_food++;
            else if (final_score >= 2 && final_score <= 4) games_2_4_food++;
            else if (final_score >= 5 && final_score <= 9) games_5_9_food++;
            else if (final_score >= 10) games_10_plus++;
            
            if (steps < 100) games_died_early++;
        }
        
        float avg_score = (float)total_score / games;
        float avg_length = 0;
        for (int len : game_lengths) avg_length += len;
        avg_length /= games;
        
        std::cout << "\n=== BEHAVIOR ANALYSIS RESULTS ===" << std::endl;
        std::cout << "Games analyzed: " << games << std::endl;
        std::cout << "Average score: " << avg_score << std::endl;
        std::cout << "Average game length: " << avg_length << " steps" << std::endl;
        std::cout << "\nScore Distribution:" << std::endl;
        std::cout << "  1 food only: " << games_1_food << " games (" << (games_1_food * 100.0f / games) << "%)" << std::endl;
        std::cout << "  2-4 foods: " << games_2_4_food << " games (" << (games_2_4_food * 100.0f / games) << "%)" << std::endl;
        std::cout << "  5-9 foods: " << games_5_9_food << " games (" << (games_5_9_food * 100.0f / games) << "%)" << std::endl;
        std::cout << "  10+ foods: " << games_10_plus << " games (" << (games_10_plus * 100.0f / games) << "%)" << std::endl;
        std::cout << "  Died early: " << games_died_early << " games (" << (games_died_early * 100.0f / games) << "%)" << std::endl;
        
        // Find the issue
        std::cout << "\n=== DIAGNOSIS ===" << std::endl;
        if (games_1_food > games / 3) {
            std::cout << "ISSUE: Too many single-food games - AI gives up after first food" << std::endl;
        }
        if (avg_length < 200) {
            std::cout << "ISSUE: Games too short - AI dying too quickly" << std::endl;
        }
        if (games_10_plus < games / 10) {
            std::cout << "ISSUE: Very few high-scoring games - not learning long-term strategy" << std::endl;
        }
        
        std::cout << "\nRECOMMENDATION:" << std::endl;
        if (avg_score < 10) {
            std::cout << "- AI needs to learn more aggressive food-seeking behavior" << std::endl;
            std::cout << "- Current strategy appears overly conservative" << std::endl;
            std::cout << "- May need curriculum learning (start with shorter snake, progress to longer)" << std::endl;
        }
    }
    
private:
    SnakeNeuralNetwork network;
    
    std::vector<float> getState(const SnakeGame& game) {
        auto head = game.getSnakeBody()[0];
        auto food = game.getFoodPosition();
        
        std::vector<float> state(8);
        
        // Food direction (4 features)
        state[0] = (food.x > head.x) ? 1.0f : 0.0f;
        state[1] = (food.x < head.x) ? 1.0f : 0.0f;
        state[2] = (food.y > head.y) ? 1.0f : 0.0f;
        state[3] = (food.y < head.y) ? 1.0f : 0.0f;
        
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
        std::cout << "Example: " << argv[0] << " snake_optimized_98percent.bin" << std::endl;
        return -1;
    }
    
    try {
        SnakeBehaviorAnalyzer analyzer(argv[1]);
        analyzer.analyzeDetailedBehavior(30);  // Analyze 30 games in detail
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}
