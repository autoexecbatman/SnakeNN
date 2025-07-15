#include "neural_network.h"
#include "snake_logic.h"
#include <iostream>

// Minimal improvement trainer - keeps what works, adds only survival focus
class MinimalImprovedTrainer {
public:
    MinimalImprovedTrainer() : network(8, 64, 4), target_network(8, 64, 4) {  // Keep proven 8 features
        // Copy initial weights
        auto source_params = network.parameters();
        auto target_params = target_network.parameters();
        torch::NoGradGuard no_grad;
        for (size_t i = 0; i < source_params.size(); i++) {
            target_params[i].copy_(source_params[i]);
        }
    }
    
    void train(int episodes = 6000, bool visual = false) {
        std::cout << "=== MINIMAL IMPROVED TRAINER - Proven Features + Survival Focus ===" << std::endl;
        std::cout << "Network: 8->64->4 | Same Features as 98% Model | Better Survival Training" << std::endl;
        
        float epsilon = 1.0f;
        const float epsilon_decay = 0.9996f;
        const float epsilon_min = 0.04f;
        
        int total_score = 0;
        int successful_episodes = 0;
        
        for (int episode = 0; episode < episodes; episode++) {
            SnakeGame game;
            game.reset();
            
            std::vector<float> prev_state;
            int prev_action = -1;
            float total_reward = 0.0f;
            int steps = 0;
            int consecutive_no_food_steps = 0;
            
            while (!game.isGameOver() && steps < 1500) {  // Moderate episode length
                auto current_state = getProvenState(game);  // Same as 98% model
                int action = getAction(current_state, epsilon);
                
                int prev_score = game.getScore();
                
                // Take action
                game.setDirection(static_cast<Direction>(action));
                bool game_continues = game.update();
                
                int new_score = game.getScore();
                
                // Track food progress
                if (new_score > prev_score) {
                    consecutive_no_food_steps = 0;  // Reset counter
                } else {
                    consecutive_no_food_steps++;
                }
                
                float reward = getImprovedSurvivalReward(game, !game_continues, steps, consecutive_no_food_steps);
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
            if (episode % 80 == 0) {
                updateTargetNetwork();
            }
            
            // Epsilon decay
            if (epsilon > epsilon_min) {
                epsilon *= epsilon_decay;
            }
            
            // EVALUATION every 200 episodes
            if (episode % 200 == 0 && episode > 0) {
                auto [pure_success, pure_avg_score, avg_length] = evaluateWithLength(100);
                float training_success = (float)successful_episodes / (episode + 1) * 100.0f;
                
                std::cout << "\\nEpisode " << episode << std::endl;
                std::cout << "Training: " << training_success << "% success" << std::endl;
                std::cout << "PURE: " << pure_success << "% success, " << pure_avg_score << " avg score, " << avg_length << " avg steps" << std::endl;
                std::cout << "Epsilon: " << epsilon << std::endl;
                
                if (pure_success >= 95.0f && pure_avg_score >= 12.0f) {
                    std::cout << "*** EXCELLENT! High success with improved scores! ***" << std::endl;
                    std::string success_path = "snake_improved_" + std::to_string((int)pure_success) + "percent_" + std::to_string((int)pure_avg_score) + "score.bin";
                    network.save(success_path);
                    std::cout << "Improved model saved: " << success_path << std::endl;
                }
            }
        }
        
        auto [final_success, final_avg_score, final_avg_length] = evaluateWithLength(200);
        
        std::cout << std::endl << "=== FINAL RESULTS ===" << std::endl;
        std::cout << "Training episodes: " << episodes << std::endl;
        std::cout << "*** PURE NETWORK PERFORMANCE ***" << std::endl;
        std::cout << "Success rate: " << final_success << "%" << std::endl;
        std::cout << "Average score: " << final_avg_score << std::endl;
        std::cout << "Average game length: " << final_avg_length << " steps" << std::endl;
        
        // Save model
        std::string model_path = "snake_minimal_" + std::to_string((int)final_success) + "percent_" + std::to_string((int)final_avg_score) + "score.bin";
        try {
            network.save(model_path);
            std::cout << "\\n*** MODEL SAVED TO: " << model_path << " ***" << std::endl;
        } catch (const std::exception& e) {
            std::cout << "Warning: Could not save model - " << e.what() << std::endl;
        }
        
        if (final_success >= 95.0f && final_avg_score >= 12.0f) {
            std::cout << "\\n🎉 SUCCESS! High success rate with improved scores! 🎉" << std::endl;
        } else if (final_success >= 95.0f) {
            std::cout << "\\n✅ Maintained high success rate!" << std::endl;
        } else {
            std::cout << "\\n📈 Building on proven foundation" << std::endl;
        }
    }
    
private:
    SnakeNeuralNetwork network;
    SnakeNeuralNetwork target_network;
    const float learning_rate = 0.007f;
    const float gamma = 0.97f;
    
    // PROVEN 8-feature state (exactly same as 98% model)
    std::vector<float> getProvenState(const SnakeGame& game) {
        auto head = game.getSnakeBody()[0];
        auto food = game.getFoodPosition();
        
        std::vector<float> state(8);
        
        // Food direction (4 features) - PROVEN
        state[0] = (food.x > head.x) ? 1.0f : 0.0f;
        state[1] = (food.x < head.x) ? 1.0f : 0.0f;
        state[2] = (food.y > head.y) ? 1.0f : 0.0f;
        state[3] = (food.y < head.y) ? 1.0f : 0.0f;
        
        // Immediate danger (4 features) - PROVEN
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
    
    // SLIGHTLY improved reward - focus on survival without complexity
    float getImprovedSurvivalReward(const SnakeGame& game, bool died, int steps, int no_food_steps) {
        if (died) {
            // Graduated death penalty based on performance
            float penalty = -12.0f;
            if (game.getScore() >= 10) penalty = -8.0f;   // Less penalty for good games
            else if (game.getScore() >= 5) penalty = -10.0f;
            return penalty;
        }
        
        static Position last_head(-1, -1);
        static int last_score = 0;
        
        auto head = game.getSnakeBody()[0];
        auto food = game.getFoodPosition();
        int current_score = game.getScore();
        
        float reward = 0.1f;  // Base survival
        
        // Food rewards
        if (current_score > last_score) {
            float food_reward = 18.0f;
            if (current_score >= 10) food_reward = 22.0f;  // Bonus for longer snakes
            reward += food_reward;
            last_score = current_score;
        }
        
        // Gentle survival bonuses (not aggressive)
        if (steps > 300) reward += 0.1f;
        if (steps > 600) reward += 0.2f;
        
        // Penalty for circling without progress
        if (no_food_steps > 100) {
            reward -= 0.2f;  // Discourage endless circling
        }
        
        // Distance guidance
        if (last_head.x >= 0) {
            float old_dist = abs(last_head.x - food.x) + abs(last_head.y - food.y);
            float new_dist = abs(head.x - food.x) + abs(head.y - food.y);
            
            if (new_dist < old_dist) {
                reward += 0.4f;
            } else if (new_dist > old_dist) {
                reward -= 0.1f;
            }
        }
        
        last_head = head;
        return reward;
    }
    
    // Evaluation with length tracking
    std::tuple<float, float, float> evaluateWithLength(int test_episodes) {
        int successful = 0;
        int total_score = 0;
        int total_steps = 0;
        
        for (int episode = 0; episode < test_episodes; episode++) {
            SnakeGame game;
            game.reset();
            
            int steps = 0;
            while (!game.isGameOver() && steps < 1500) {
                auto state = getProvenState(game);
                auto action_tensor = network.getAction(state, 0.0f);
                int action = static_cast<int>(action_tensor.cpu().item<int64_t>());
                
                game.setDirection(static_cast<Direction>(action));
                game.update();
                steps++;
            }
            
            int score = game.getScore();
            total_score += score;
            total_steps += steps;
            if (score > 0) successful++;
        }
        
        float success_rate = (float)successful / test_episodes * 100.0f;
        float avg_score = (float)total_score / test_episodes;
        float avg_length = (float)total_steps / test_episodes;
        
        return {success_rate, avg_score, avg_length};
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
    std::cout << "=== MINIMAL IMPROVED SNAKE AI TRAINER ===" << std::endl;
    
    try {
        MinimalImprovedTrainer trainer;
        trainer.train(6000, false);
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}
