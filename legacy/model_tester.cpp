#include "neural_network.h"
#include "snake_logic.h"
#include <iostream>

class ModelTester {
public:
    ModelTester(const std::string& model_path) : network(8, 64, 4) {  // Match optimized trainer architecture
        network.load(model_path);
        std::cout << "Model loaded from: " << model_path << std::endl;
    }
    
    void test(int episodes = 100) {
        std::cout << "=== TESTING LOADED MODEL ===" << std::endl;
        
        int total_score = 0;
        int successful_episodes = 0;
        
        for (int episode = 0; episode < episodes; episode++) {
            SnakeGame game;
            game.reset();
            
            int steps = 0;
            while (!game.isGameOver() && steps < 500) {
                auto state = getSimplifiedState(game);
                auto action_tensor = network.getAction(state, 0.0f); // No random actions
                int action = static_cast<int>(action_tensor.cpu().item<int64_t>());
                
                game.setDirection(static_cast<Direction>(action));
                game.update();
                steps++;
            }
            
            int score = game.getScore();
            total_score += score;
            if (score > 0) successful_episodes++;
            
            if (episode % 20 == 0) {
                std::cout << "Episode " << episode << ": Score = " << score << std::endl;
            }
        }
        
        float avg_score = (float)total_score / episodes;
        float success_rate = (float)successful_episodes / episodes * 100.0f;
        
        std::cout << "\n=== TEST RESULTS ===" << std::endl;
        std::cout << "Episodes tested: " << episodes << std::endl;
        std::cout << "Total score: " << total_score << std::endl;
        std::cout << "Average score: " << avg_score << std::endl;
        std::cout << "Success rate: " << success_rate << "%" << std::endl;
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
        std::cout << "Example: " << argv[0] << " snake_model_27percent.bin" << std::endl;
        return -1;
    }
    
    try {
        ModelTester tester(argv[1]);
        tester.test(100);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}
