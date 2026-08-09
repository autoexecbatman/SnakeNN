#include "neural_network.h"
#include "snake_logic.h"
#include <iostream>

// Optimized trainer based on successful 5K model approach
class OptimizedDQNTrainer {
public:
    OptimizedDQNTrainer() : network(8, 64, 4), target_network(8, 64, 4) {  // Proven 8 features, bigger network
        // Copy initial weights
        auto source_params = network.parameters();
        auto target_params = target_network.parameters();
        torch::NoGradGuard no_grad;
        for (size_t i = 0; i < source_params.size(); i++) {
            target_params[i].copy_(source_params[i]);
        }
    }
    
    void train(int episodes = 8000, bool visual = false) {
        std::cout << "=== OPTIMIZED TRAINER - Building on 96% Success ===" << std::endl;
        std::cout << "Network: 8->64->4 | Proven Features | Pure Evaluation" << std::endl;
        
        float epsilon = 1.0f;
        const float epsilon_decay = 0.9996f;  // Optimized decay
        const float epsilon_min = 0.03f;      // Balanced exploration
        
        int total_score = 0;
        int successful_episodes = 0;
        
        for (int episode = 0; episode < episodes; episode++) {
            SnakeGame game;
            game.reset();
            
            std::vector<float> prev_state;
            int prev_action = -1;
            float total_reward = 0.0f;
            int steps = 0;
            
            while (!game.isGameOver() && steps < 800) {  // Reasonable episode length
                auto current_state = getOptimizedState(game);
                int action = getAction(current_state, epsilon);
                
                // Take action
                game.setDirection(static_cast<Direction>(action));
                bool game_continues = game.update();
                
                float reward = getOptimizedReward(game, !game_continues, steps);
                total_reward += reward;
                steps++;
                
                // IMMEDIATE TRAINING
                if (!prev_state.empty()) {
                    trainStep(prev_state, prev_action, reward, current_state, !game_continues);
                }
                
                prev_state = current_state;
                prev_action = action;
                
                if (!game_continues) break;
            }
            
            int score = game.getScore();
            total_score += score;
            if (score > 0) successful_episodes++;
            
            // Update target network
            if (episode % 75 == 0) {
                updateTargetNetwork();
            }
            
            // Epsilon decay
            if (epsilon > epsilon_min) {
                epsilon *= epsilon_decay;
            }
            
            // PURE EVALUATION every 200 episodes
            if (episode % 200 == 0 && episode > 0) {
                float pure_performance = evaluatePureNetwork(100);  // Test 100 games
                float training_success = (float)successful_episodes / (episode + 1) * 100.0f;
                
                std::cout << "\\nEpisode " << episode << " - Training: " << training_success << "% | PURE: " << pure_performance << "% | Epsilon: " << epsilon << std::endl;
                
                if (pure_performance >= 90.0f) {
                    std::cout << "*** 90%+ ACHIEVED! ***" << std::endl;
                    // Save intermediate success model
                    std::string success_path = "snake_success_" + std::to_string((int)pure_performance) + "percent.bin";
                    network.save(success_path);
                    std::cout << "Success model saved: " << success_path << std::endl;
                }
            }
            
            // Early stopping if we achieve great performance
            if (episode > 2000 && episode % 500 == 0) {
                float check_performance = evaluatePureNetwork(50);
                if (check_performance >= 95.0f) {
                    std::cout << "\\n*** EARLY SUCCESS! 95%+ achieved at episode " << episode << " ***" << std::endl;
                    break;
                }
            }
        }
        
        float final_avg = (float)total_score / episodes;
        float final_success_rate = (float)successful_episodes / episodes * 100.0f;
        float final_pure = evaluatePureNetwork(200);  // Thorough final test
        
        std::cout << std::endl << "=== FINAL RESULTS ===" << std::endl;
        std::cout << "Training episodes: " << episodes << std::endl;
        std::cout << "Training success (with exploration): " << final_success_rate << "%" << std::endl;
        std::cout << "*** PURE NETWORK PERFORMANCE: " << final_pure << "% ***" << std::endl;
        std::cout << "Average score: " << final_avg << std::endl;
        
        // Save with pure performance
        std::string model_path = "snake_optimized_" + std::to_string((int)final_pure) + "percent.bin";
        try {
            network.save(model_path);
            std::cout << "\\n*** MODEL SAVED TO: " << model_path << " ***" << std::endl;
        } catch (const std::exception& e) {
            std::cout << "Warning: Could not save model - " << e.what() << std::endl;
        }
        
        if (final_pure >= 90.0f) {
            std::cout << "\\n🎉 MISSION ACCOMPLISHED! 90%+ Performance! 🎉" << std::endl;
        } else if (final_pure >= 80.0f) {
            std::cout << "\\n✨ EXCELLENT! 80%+ Performance!" << std::endl;
        } else if (final_pure >= 70.0f) {
            std::cout << "\\n👍 GOOD! 70%+ Performance!" << std::endl;
        } else {
            std::cout << "\\n📊 Baseline established - " << final_pure << "% performance" << std::endl;
        }
    }
    
private:
    SnakeNeuralNetwork network;
    SnakeNeuralNetwork target_network;
    const float learning_rate = 0.008f;  // Slightly higher for bigger network
    const float gamma = 0.97f;           // Balanced discount
    
    // PROVEN 8-feature state (same as successful model)
    std::vector<float> getOptimizedState(const SnakeGame& game) {
        auto head = game.getSnakeBody()[0];
        auto food = game.getFoodPosition();
        
        std::vector<float> state(8);
        
        // Food direction (4 features) - PROVEN TO WORK
        state[0] = (food.x > head.x) ? 1.0f : 0.0f;  // Food right
        state[1] = (food.x < head.x) ? 1.0f : 0.0f;  // Food left  
        state[2] = (food.y > head.y) ? 1.0f : 0.0f;  // Food down
        state[3] = (food.y < head.y) ? 1.0f : 0.0f;  // Food up
        
        // Immediate danger (4 features) - PROVEN TO WORK
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
    
    int getAction(const std::vector<float>& state, float epsilon) {
        if ((rand() % 1000) / 1000.0f < epsilon) {
            return rand() % 4;
        }
        
        auto action_tensor = network.getAction(state, 0.0f);
        return static_cast<int>(action_tensor.cpu().item<int64_t>());
    }
    
    // OPTIMIZED reward structure
    float getOptimizedReward(const SnakeGame& game, bool died, int steps) {
        if (died) return -15.0f;  // Death penalty
        
        static Position last_head(-1, -1);
        static int last_score = 0;
        
        auto head = game.getSnakeBody()[0];
        auto food = game.getFoodPosition();
        int current_score = game.getScore();
        
        float reward = 0.1f;  // Base survival
        
        // Food reward - BIG bonus for success
        if (current_score > last_score) {
            reward += 15.0f;
            last_score = current_score;
        }
        
        // Distance-based guidance
        if (last_head.x >= 0) {
            float old_dist = abs(last_head.x - food.x) + abs(last_head.y - food.y);
            float new_dist = abs(head.x - food.x) + abs(head.y - food.y);
            
            if (new_dist < old_dist) {
                reward += 0.3f;  // Getting closer
            } else if (new_dist > old_dist) {
                reward -= 0.1f;  // Moving away
            }
        }
        
        last_head = head;
        return reward;
    }
    
    // PURE NETWORK EVALUATION - The key insight!
    float evaluatePureNetwork(int test_episodes) {
        int successful = 0;
        
        for (int episode = 0; episode < test_episodes; episode++) {
            SnakeGame game;
            game.reset();
            
            int steps = 0;
            while (!game.isGameOver() && steps < 800) {
                auto state = getOptimizedState(game);
                auto action_tensor = network.getAction(state, 0.0f);  // NO RANDOM ACTIONS
                int action = static_cast<int>(action_tensor.cpu().item<int64_t>());
                
                game.setDirection(static_cast<Direction>(action));
                game.update();
                steps++;
            }
            
            if (game.getScore() > 0) successful++;
        }
        
        return (float)successful / test_episodes * 100.0f;
    }
    
    void trainStep(const std::vector<float>& state, int action, float reward,
                   const std::vector<float>& next_state, bool done) {
        
        auto state_tensor = torch::zeros({1, 8}, torch::kFloat);
        auto next_state_tensor = torch::zeros({1, 8}, torch::kFloat);
        
        for (int i = 0; i < 8; i++) {
            state_tensor[0][i] = state[i];
            next_state_tensor[0][i] = next_state[i];
        }
        
        auto current_q = network.forward(state_tensor);
        
        float target_q = reward;
        if (!done) {
            auto next_q = target_network.forward(next_state_tensor);
            target_q += gamma * std::get<0>(next_q.max(1)).cpu().item<float>();
        }
        
        auto target_tensor = current_q.clone();
        target_tensor[0][action] = target_q;
        
        auto loss = torch::mse_loss(current_q, target_tensor);
        
        // Backprop
        loss.backward();
        
        auto params = network.parameters();
        torch::NoGradGuard no_grad;
        for (auto& param : params) {
            if (param.grad().defined()) {
                param -= learning_rate * param.grad();
                param.grad().zero_();
            }
        }
    }
    
    void updateTargetNetwork() {
        auto source_params = network.parameters();
        auto target_params = target_network.parameters();
        torch::NoGradGuard no_grad;
        for (size_t i = 0; i < source_params.size(); i++) {
            target_params[i].copy_(source_params[i]);
        }
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
    std::cout << "=== OPTIMIZED SNAKE AI TRAINER ===" << std::endl;
    
    try {
        OptimizedDQNTrainer trainer;
        trainer.train(8000, false);
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}
