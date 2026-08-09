#include "neural_network.h"
#include "snake_logic.h"
#include <iostream>

// Fixed trainer with simplified architecture and immediate learning
class FixedDQNTrainer {
public:
    FixedDQNTrainer() : network(8, 32, 4), target_network(8, 32, 4) {
        // Copy initial weights
        auto source_params = network.parameters();
        auto target_params = target_network.parameters();
        torch::NoGradGuard no_grad;
        for (size_t i = 0; i < source_params.size(); i++) {
            target_params[i].copy_(source_params[i]);
        }
    }
    
    void train(int episodes = 1000, bool visual = false) {
        std::cout << "=== FIXED TRAINER (No Experience Replay) ===" << std::endl;
        std::cout << "Network: 8->32->4 | Learning Rate: 0.01 | Immediate Training" << std::endl;
        
        float epsilon = 1.0f;
        const float epsilon_decay = 0.9998f;  // Even slower decay for 20K episodes
        const float epsilon_min = 0.01f;       // Very low minimum for maximum exploration
        
        int total_score = 0;
        int successful_episodes = 0;
        
        for (int episode = 0; episode < episodes; episode++) {
            SnakeGame game;
            game.reset();
            
            std::vector<float> prev_state;
            int prev_action = -1;
            float total_reward = 0.0f;
            int steps = 0;
            
            while (!game.isGameOver() && steps < 500) {
                auto current_state = getSimplifiedState(game);
                int action = getAction(current_state, epsilon);
                
                // Take action
                game.setDirection(static_cast<Direction>(action));
                bool game_continues = game.update();
                
                float reward = getImprovedReward(game, !game_continues);
                total_reward += reward;
                steps++;
                
                // IMMEDIATE TRAINING - No experience replay!
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
            
            // Update target network for stability during long training
            if (episode % 100 == 0) {
                updateTargetNetwork();
            }
            
            // Slower epsilon decay
            if (epsilon > epsilon_min) {
                epsilon *= epsilon_decay;
            }
            
            // Progress reporting every 500 episodes for 20K total
            if (episode % 500 == 0) {
                float avg_score = (float)total_score / (episode + 1);
                float success_rate = (float)successful_episodes / (episode + 1) * 100.0f;
                
                std::cout << "Episode " << episode << "/" << episodes << std::endl;
                std::cout << "  Score this episode: " << score << std::endl;
                std::cout << "  Average score: " << avg_score << std::endl;
                std::cout << "  Success rate: " << success_rate << "%" << std::endl;
                std::cout << "  Total reward: " << total_reward << std::endl;
                std::cout << "  Steps: " << steps << std::endl;
                std::cout << "  Epsilon: " << epsilon << std::endl;
                std::cout << std::endl;
            }
            
            // Early success detection and milestones
            if (episode >= 1000 && episode % 1000 == 0) {
                float recent_avg = (float)total_score / (episode + 1);
                float recent_success = (float)successful_episodes / (episode + 1) * 100.0f;
                
                std::cout << "\n*** MILESTONE at " << episode << " episodes ***" << std::endl;
                if (recent_success > 60.0f) {
                    std::cout << "EXCELLENT! Success rate > 60% - Approaching Q-table performance!" << std::endl;
                } else if (recent_success > 55.0f) {
                    std::cout << "GREAT! Success rate > 55% - Strong improvement!" << std::endl;
                } else if (recent_avg > 1.5f) {
                    std::cout << "GOOD! Average score > 1.5 - Learning progressing!" << std::endl;
                }
                std::cout << "Current performance: " << recent_success << "% success\n" << std::endl;
            }
        }
        
        float final_avg = (float)total_score / episodes;
        float final_success_rate = (float)successful_episodes / episodes * 100.0f;
        
        std::cout << std::endl << "=== FINAL RESULTS ===" << std::endl;
        std::cout << "Total episodes: " << episodes << std::endl;
        std::cout << "Total score: " << total_score << std::endl;
        std::cout << "Average score: " << final_avg << std::endl;
        std::cout << "Successful episodes: " << successful_episodes << std::endl;
        std::cout << "Success rate: " << final_success_rate << "%" << std::endl;
        
        if (final_avg > 1.0f) {
            std::cout << "*** SUCCESS! Snake learned to find food! ***" << std::endl;
        } else if (final_avg > 0.3f) {
            std::cout << "** PARTIAL SUCCESS - Some learning occurred **" << std::endl;
        } else {
            std::cout << "* FAILURE - Little to no learning *" << std::endl;
        }
        
        // Save the trained model
        std::string model_path = "snake_model_" + std::to_string((int)(final_success_rate * 100)) + "percent.bin";
        try {
            network.save(model_path);
            std::cout << "\n*** MODEL SAVED TO: " << model_path << " ***" << std::endl;
            std::cout << "Use this model by loading it in future training sessions!" << std::endl;
        } catch (const std::exception& e) {
            std::cout << "Warning: Could not save model - " << e.what() << std::endl;
        }
    }
    
private:
    SnakeNeuralNetwork network;
    SnakeNeuralNetwork target_network;
    const float learning_rate = 0.01f;  // 10x higher than original
    const float gamma = 0.95f;
    
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
    
    int getAction(const std::vector<float>& state, float epsilon) {
        if ((rand() % 1000) / 1000.0f < epsilon) {
            return rand() % 4;
        }
        
        auto action_tensor = network.getAction(state, 0.0f);
        return static_cast<int>(action_tensor.cpu().item<int64_t>());
    }
    
    float getImprovedReward(const SnakeGame& game, bool died) {
        if (died) return -10.0f;
        
        static Position last_head(-1, -1);
        static Position last_food(-1, -1);
        
        auto head = game.getSnakeBody()[0];
        auto food = game.getFoodPosition();
        
        float reward = 0.1f;  // Base survival reward
        
        // Big reward for eating food
        if (last_food.x >= 0 && head.x == last_food.x && head.y == last_food.y) {
            reward += 10.0f;
        }
        
        // Distance-based reward
        if (last_head.x >= 0) {
            float old_dist = abs(last_head.x - food.x) + abs(last_head.y - food.y);
            float new_dist = abs(head.x - food.x) + abs(head.y - food.y);
            
            if (new_dist < old_dist) {
                reward += 0.5f;  // Reward for getting closer
            } else if (new_dist > old_dist) {
                reward -= 0.1f;  // Small penalty for getting farther
            }
        }
        
        last_head = head;
        last_food = food;
        
        return reward;
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
        
        // Immediate backprop with higher learning rate
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
    std::cout << "=== FIXED SNAKE AI TRAINER ===" << std::endl;
    
    try {
        FixedDQNTrainer trainer;
        trainer.train(20000, false);  // 20K episodes - serious training!
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}
