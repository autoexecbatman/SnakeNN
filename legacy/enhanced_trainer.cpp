#include "neural_network.h"
#include "snake_logic.h"
#include <iostream>

// Enhanced trainer with proper evaluation and 90%+ target performance
class EnhancedDQNTrainer {
public:
    EnhancedDQNTrainer() : network(12, 64, 4), target_network(12, 64, 4) {  // 12 inputs, 64 hidden, 4 outputs
        // Copy initial weights
        auto source_params = network.parameters();
        auto target_params = target_network.parameters();
        torch::NoGradGuard no_grad;
        for (size_t i = 0; i < source_params.size(); i++) {
            target_params[i].copy_(source_params[i]);
        }
    }
    
    void train(int episodes = 10000, bool visual = false) {
        std::cout << "=== ENHANCED TRAINER - Target: 90%+ Performance ===" << std::endl;
        std::cout << "Network: 12->64->4 | Learning Rate: 0.005 | Proper Evaluation" << std::endl;
        
        float epsilon = 1.0f;
        const float epsilon_decay = 0.9995f;
        const float epsilon_min = 0.05f;  // Keep some exploration
        
        int total_score = 0;
        int successful_episodes = 0;
        
        for (int episode = 0; episode < episodes; episode++) {
            SnakeGame game;
            game.reset();
            
            std::vector<float> prev_state;
            int prev_action = -1;
            float total_reward = 0.0f;
            int steps = 0;
            
            while (!game.isGameOver() && steps < 1000) {  // Longer episodes
                auto current_state = getEnhancedState(game);
                int action = getAction(current_state, epsilon);
                
                // Take action
                game.setDirection(static_cast<Direction>(action));
                bool game_continues = game.update();
                
                float reward = getEnhancedReward(game, !game_continues, steps);
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
            if (episode % 100 == 0) {
                updateTargetNetwork();
            }
            
            // Epsilon decay
            if (epsilon > epsilon_min) {
                epsilon *= epsilon_decay;
            }
            
            // PROPER EVALUATION - Test pure network performance
            if (episode % 250 == 0 && episode > 0) {
                float pure_performance = evaluatePureNetwork(50);  // Test 50 games
                float training_success = (float)successful_episodes / (episode + 1) * 100.0f;
                
                std::cout << "\\n=== Episode " << episode << " Performance ===" << std::endl;
                std::cout << "Training Success (with exploration): " << training_success << "%" << std::endl;
                std::cout << "PURE NETWORK Performance: " << pure_performance << "%" << std::endl;
                std::cout << "Epsilon: " << epsilon << std::endl;
                
                if (pure_performance >= 90.0f) {
                    std::cout << "*** TARGET ACHIEVED! Pure network >= 90% ***" << std::endl;
                }
                
                std::cout << std::endl;
            }
            
            // Final intensive evaluation
            if (episode == episodes - 1) {
                std::cout << "\\n=== FINAL INTENSIVE EVALUATION ===" << std::endl;
                float final_pure = evaluatePureNetwork(200);  // Test 200 games
                std::cout << "FINAL PURE NETWORK Performance: " << final_pure << "%" << std::endl;
            }
        }
        
        float final_avg = (float)total_score / episodes;
        float final_success_rate = (float)successful_episodes / episodes * 100.0f;
        float final_pure = evaluatePureNetwork(100);
        
        std::cout << std::endl << "=== FINAL RESULTS ===" << std::endl;
        std::cout << "Training episodes: " << episodes << std::endl;
        std::cout << "Training success (with exploration): " << final_success_rate << "%" << std::endl;
        std::cout << "PURE NETWORK PERFORMANCE: " << final_pure << "%" << std::endl;
        std::cout << "Average score: " << final_avg << std::endl;
        
        // Save with pure performance in filename
        std::string model_path = "snake_enhanced_" + std::to_string((int)final_pure) + "percent.bin";
        try {
            network.save(model_path);
            std::cout << "\\n*** MODEL SAVED TO: " << model_path << " ***" << std::endl;
            std::cout << "Pure network performance: " << final_pure << "%" << std::endl;
        } catch (const std::exception& e) {
            std::cout << "Warning: Could not save model - " << e.what() << std::endl;
        }
        
        if (final_pure >= 90.0f) {
            std::cout << "\\n🎉 SUCCESS! Achieved 90%+ pure performance! 🎉" << std::endl;
        } else if (final_pure >= 80.0f) {
            std::cout << "\\n✨ GREAT! Achieved 80%+ performance - close to target!" << std::endl;
        } else {
            std::cout << "\\n📈 GOOD PROGRESS - Continue training for higher performance" << std::endl;
        }
    }
    
private:
    SnakeNeuralNetwork network;
    SnakeNeuralNetwork target_network;
    const float learning_rate = 0.005f;  // Slightly lower for stability
    const float gamma = 0.99f;           // Higher discount for long-term planning
    
    // Enhanced state representation (12 features instead of 8)
    std::vector<float> getEnhancedState(const SnakeGame& game) {
        auto head = game.getSnakeBody()[0];
        auto food = game.getFoodPosition();
        
        std::vector<float> state(12);
        
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
        
        // Distance to food (normalized)
        float dist = abs(head.x - food.x) + abs(head.y - food.y);
        state[8] = dist / (SnakeGame::GRID_WIDTH + SnakeGame::GRID_HEIGHT);
        
        // Snake length (normalized)
        state[9] = (float)game.getSnakeBody().size() / 20.0f;
        
        // Position relative to center (2 features)
        state[10] = (float)head.x / SnakeGame::GRID_WIDTH;
        state[11] = (float)head.y / SnakeGame::GRID_HEIGHT;
        
        return state;
    }
    
    int getAction(const std::vector<float>& state, float epsilon) {
        if ((rand() % 1000) / 1000.0f < epsilon) {
            return rand() % 4;
        }
        
        auto action_tensor = network.getAction(state, 0.0f);
        return static_cast<int>(action_tensor.cpu().item<int64_t>());
    }
    
    float getEnhancedReward(const SnakeGame& game, bool died, int steps) {
        if (died) return -20.0f;  // Stronger death penalty
        
        static Position last_head(-1, -1);
        static Position last_food(-1, -1);
        static int last_score = 0;
        
        auto head = game.getSnakeBody()[0];
        auto food = game.getFoodPosition();
        int current_score = game.getScore();
        
        float reward = 0.1f;  // Base survival reward
        
        // Food reward
        if (current_score > last_score) {
            reward += 20.0f;  // Big food reward
        }
        
        // Distance-based reward (more sophisticated)
        if (last_head.x >= 0) {
            float old_dist = abs(last_head.x - food.x) + abs(last_head.y - food.y);
            float new_dist = abs(head.x - food.x) + abs(head.y - food.y);
            
            if (new_dist < old_dist) {
                reward += 1.0f;  // Good progress toward food
            } else if (new_dist > old_dist) {
                reward -= 0.3f;  // Penalty for moving away
            }
        }
        
        // Efficiency bonus (getting food quickly)
        if (current_score > last_score) {
            float efficiency_bonus = std::max(0.0f, (200.0f - steps) / 100.0f);
            reward += efficiency_bonus;
        }
        
        // Update tracking
        last_head = head;
        last_food = food;
        last_score = current_score;
        
        return reward;
    }
    
    // PURE NETWORK EVALUATION - No random actions!
    float evaluatePureNetwork(int test_episodes) {
        int successful = 0;
        int total_score = 0;
        
        for (int episode = 0; episode < test_episodes; episode++) {
            SnakeGame game;
            game.reset();
            
            int steps = 0;
            while (!game.isGameOver() && steps < 1000) {
                auto state = getEnhancedState(game);
                auto action_tensor = network.getAction(state, 0.0f);  // NO EPSILON!
                int action = static_cast<int>(action_tensor.cpu().item<int64_t>());
                
                game.setDirection(static_cast<Direction>(action));
                game.update();
                steps++;
            }
            
            int score = game.getScore();
            total_score += score;
            if (score > 0) successful++;
        }
        
        return (float)successful / test_episodes * 100.0f;
    }
    
    void trainStep(const std::vector<float>& state, int action, float reward,
                   const std::vector<float>& next_state, bool done) {
        
        auto state_tensor = torch::zeros({1, 12}, torch::kFloat);  // 12 features now
        auto next_state_tensor = torch::zeros({1, 12}, torch::kFloat);
        
        for (int i = 0; i < 12; i++) {
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
    std::cout << "=== ENHANCED SNAKE AI TRAINER ===" << std::endl;
    
    try {
        EnhancedDQNTrainer trainer;
        trainer.train(10000, false);
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}
