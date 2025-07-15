#include "neural_network.h"
#include "snake_logic.h"
#include <iostream>

// Aggressive trainer for high scores (15+ foods per game)
class AggressiveDQNTrainer {
public:
    AggressiveDQNTrainer() : network(8, 64, 4), target_network(8, 64, 4) {
        // Copy initial weights
        auto source_params = network.parameters();
        auto target_params = target_network.parameters();
        torch::NoGradGuard no_grad;
        for (size_t i = 0; i < source_params.size(); i++) {
            target_params[i].copy_(source_params[i]);
        }
    }
    
    void train(int episodes = 10000, bool visual = false) {
        std::cout << "=== AGGRESSIVE TRAINER - Target: 99% Success + 15+ Average Score ===" << std::endl;
        std::cout << "Network: 8->64->4 | Focus: Growth Over Survival | Long Games" << std::endl;
        
        float epsilon = 1.0f;
        const float epsilon_decay = 0.9995f;
        const float epsilon_min = 0.02f;  // Very low for aggressive play
        
        int total_score = 0;
        int successful_episodes = 0;
        
        for (int episode = 0; episode < episodes; episode++) {
            SnakeGame game;
            game.reset();
            
            std::vector<float> prev_state;
            int prev_action = -1;
            float total_reward = 0.0f;
            int steps = 0;
            int foods_eaten = 0;
            
            while (!game.isGameOver() && steps < 2000) {  // LONGER episodes for growth
                auto current_state = getOptimizedState(game);
                int action = getAction(current_state, epsilon);
                
                int prev_score = game.getScore();
                
                // Take action
                game.setDirection(static_cast<Direction>(action));
                bool game_continues = game.update();
                
                int new_score = game.getScore();
                if (new_score > prev_score) {
                    foods_eaten++;
                }
                
                float reward = getAggressiveReward(game, !game_continues, steps, foods_eaten);
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
            
            // PERFORMANCE EVALUATION every 250 episodes
            if (episode % 250 == 0 && episode > 0) {
                auto [pure_success, pure_avg_score] = evaluateAggressiveNetwork(100);
                float training_success = (float)successful_episodes / (episode + 1) * 100.0f;
                float training_avg = (float)total_score / (episode + 1);
                
                std::cout << "\\nEpisode " << episode << std::endl;
                std::cout << "Training: " << training_success << "% success, " << training_avg << " avg score" << std::endl;
                std::cout << "PURE: " << pure_success << "% success, " << pure_avg_score << " avg score" << std::endl;
                std::cout << "Epsilon: " << epsilon << std::endl;
                
                if (pure_success >= 95.0f && pure_avg_score >= 15.0f) {
                    std::cout << "*** TARGET ACHIEVED! 95%+ success with 15+ average score! ***" << std::endl;
                    // Save elite model
                    std::string elite_path = "snake_elite_" + std::to_string((int)pure_success) + "percent_" + std::to_string((int)pure_avg_score) + "score.bin";
                    network.save(elite_path);
                    std::cout << "Elite model saved: " << elite_path << std::endl;
                }
            }
        }
        
        auto [final_success, final_avg_score] = evaluateAggressiveNetwork(200);
        
        std::cout << std::endl << "=== FINAL RESULTS ===" << std::endl;
        std::cout << "Training episodes: " << episodes << std::endl;
        std::cout << "*** PURE NETWORK PERFORMANCE ***" << std::endl;
        std::cout << "Success rate: " << final_success << "%" << std::endl;
        std::cout << "Average score: " << final_avg_score << std::endl;
        
        // Save with both metrics
        std::string model_path = "snake_aggressive_" + std::to_string((int)final_success) + "percent_" + std::to_string((int)final_avg_score) + "score.bin";
        try {
            network.save(model_path);
            std::cout << "\\n*** MODEL SAVED TO: " << model_path << " ***" << std::endl;
        } catch (const std::exception& e) {
            std::cout << "Warning: Could not save model - " << e.what() << std::endl;
        }
        
        if (final_success >= 95.0f && final_avg_score >= 15.0f) {
            std::cout << "\\n🏆 ELITE PERFORMANCE! 95%+ success with 15+ average score! 🏆" << std::endl;
        } else if (final_success >= 90.0f && final_avg_score >= 12.0f) {
            std::cout << "\\n⭐ EXCELLENT! 90%+ success with 12+ average score!" << std::endl;
        } else if (final_success >= 95.0f) {
            std::cout << "\\n✅ High success rate achieved, but score could be higher" << std::endl;
        } else {
            std::cout << "\\n📈 Progress made - continue training for elite performance" << std::endl;
        }
    }
    
private:
    SnakeNeuralNetwork network;
    SnakeNeuralNetwork target_network;
    const float learning_rate = 0.006f;  // Balanced for aggressive training
    const float gamma = 0.98f;           // High discount for long-term rewards
    
    std::vector<float> getOptimizedState(const SnakeGame& game) {
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
    
    int getAction(const std::vector<float>& state, float epsilon) {
        if ((rand() % 1000) / 1000.0f < epsilon) {
            return rand() % 4;
        }
        
        auto action_tensor = network.getAction(state, 0.0f);
        return static_cast<int>(action_tensor.cpu().item<int64_t>());
    }
    
    // AGGRESSIVE reward structure - prioritizes growth over safety
    float getAggressiveReward(const SnakeGame& game, bool died, int steps, int foods_eaten) {
        if (died) {
            // Penalty based on performance - less penalty if achieved good score
            float death_penalty = -10.0f;
            if (foods_eaten >= 15) death_penalty = -5.0f;   // Less penalty for high scorers
            else if (foods_eaten >= 10) death_penalty = -7.0f;
            else if (foods_eaten >= 5) death_penalty = -8.0f;
            return death_penalty;
        }
        
        static Position last_head(-1, -1);
        static int last_score = 0;
        
        auto head = game.getSnakeBody()[0];
        auto food = game.getFoodPosition();
        int current_score = game.getScore();
        
        float reward = 0.05f;  // Small base survival reward
        
        // MASSIVE food rewards - encourage growth!
        if (current_score > last_score) {
            float food_reward = 20.0f;
            
            // SCALING BONUSES - more reward for longer snakes!
            if (current_score >= 20) food_reward = 35.0f;      // Elite performance
            else if (current_score >= 15) food_reward = 30.0f; // Excellent performance  
            else if (current_score >= 10) food_reward = 25.0f; // Good performance
            
            reward += food_reward;
            
            // Efficiency bonus for quick food gathering
            if (steps < 50 * current_score) {  // If getting food efficiently
                reward += 5.0f;
            }
            
            last_score = current_score;
        }
        
        // Distance-based reward - be aggressive toward food!
        if (last_head.x >= 0) {
            float old_dist = abs(last_head.x - food.x) + abs(last_head.y - food.y);
            float new_dist = abs(head.x - food.x) + abs(head.y - food.y);
            
            if (new_dist < old_dist) {
                reward += 0.5f;  // Reward for approaching food
            } else if (new_dist > old_dist) {
                reward -= 0.15f;  // Small penalty for moving away
            }
        }
        
        // Length bonus - reward for having a long snake
        int snake_length = game.getSnakeBody().size();
        if (snake_length >= 10) {
            reward += 0.3f;  // Bonus for being long
        }
        
        last_head = head;
        return reward;
    }
    
    // AGGRESSIVE EVALUATION - measures both success rate AND average score
    std::pair<float, float> evaluateAggressiveNetwork(int test_episodes) {
        int successful = 0;
        int total_score = 0;
        
        for (int episode = 0; episode < test_episodes; episode++) {
            SnakeGame game;
            game.reset();
            
            int steps = 0;
            while (!game.isGameOver() && steps < 2000) {  // Long games allowed
                auto state = getOptimizedState(game);
                auto action_tensor = network.getAction(state, 0.0f);  // NO RANDOM ACTIONS
                int action = static_cast<int>(action_tensor.cpu().item<int64_t>());
                
                game.setDirection(static_cast<Direction>(action));
                game.update();
                steps++;
            }
            
            int score = game.getScore();
            total_score += score;
            if (score > 0) successful++;
        }
        
        float success_rate = (float)successful / test_episodes * 100.0f;
        float avg_score = (float)total_score / test_episodes;
        
        return {success_rate, avg_score};
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
    std::cout << "=== AGGRESSIVE SNAKE AI TRAINER ===" << std::endl;
    
    try {
        AggressiveDQNTrainer trainer;
        trainer.train(10000, false);
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}
