#include "neural_network.h"
#include "snake_logic.h"
#include <iostream>

// Self-Aware trainer - focuses on collision avoidance and longer survival
class SelfAwareDQNTrainer {
public:
    SelfAwareDQNTrainer() : network(12, 64, 4), target_network(12, 64, 4) {  // Enhanced 12-feature state
        // Copy initial weights
        auto source_params = network.parameters();
        auto target_params = target_network.parameters();
        torch::NoGradGuard no_grad;
        for (size_t i = 0; i < source_params.size(); i++) {
            target_params[i].copy_(source_params[i]);
        }
    }
    
    void train(int episodes = 8000, bool visual = false) {
        std::cout << "=== SELF-AWARE TRAINER - Focus: Collision Avoidance + Long Survival ===" << std::endl;
        std::cout << "Network: 12->64->4 | Enhanced Body Awareness | Target: 15+ Foods" << std::endl;
        
        float epsilon = 1.0f;
        const float epsilon_decay = 0.9996f;
        const float epsilon_min = 0.03f;
        
        int total_score = 0;
        int successful_episodes = 0;
        int long_games = 0;  // Games with 500+ steps
        
        for (int episode = 0; episode < episodes; episode++) {
            SnakeGame game;
            game.reset();
            
            std::vector<float> prev_state;
            int prev_action = -1;
            float total_reward = 0.0f;
            int steps = 0;
            int foods_eaten = 0;
            bool died_from_self = false;
            
            while (!game.isGameOver() && steps < 3000) {  // Even longer episodes
                auto current_state = getSelfAwareState(game);
                int action = getAction(current_state, epsilon);
                
                int prev_score = game.getScore();
                auto prev_head = game.getSnakeBody()[0];
                
                // Take action
                game.setDirection(static_cast<Direction>(action));
                bool game_continues = game.update();
                
                int new_score = game.getScore();
                if (new_score > prev_score) {
                    foods_eaten++;
                }
                
                // Check death cause for training
                if (!game_continues) {
                    auto final_head = prev_head;
                    Direction final_dir = static_cast<Direction>(action);
                    
                    switch (final_dir) {
                        case Direction::UP: final_head.y--; break;
                        case Direction::DOWN: final_head.y++; break;
                        case Direction::LEFT: final_head.x--; break;
                        case Direction::RIGHT: final_head.x++; break;
                    }
                    
                    // Check if death was self-collision
                    if (final_head.x >= 0 && final_head.x < SnakeGame::GRID_WIDTH &&
                        final_head.y >= 0 && final_head.y < SnakeGame::GRID_HEIGHT) {
                        for (const auto& segment : game.getSnakeBody()) {
                            if (segment.x == final_head.x && segment.y == final_head.y) {
                                died_from_self = true;
                                break;
                            }
                        }
                    }
                }
                
                float reward = getSelfAwareReward(game, !game_continues, steps, foods_eaten, died_from_self);
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
            if (steps >= 500) long_games++;  // Track long survival
            
            // Update target network
            if (episode % 100 == 0) {
                updateTargetNetwork();
            }
            
            // Epsilon decay
            if (epsilon > epsilon_min) {
                epsilon *= epsilon_decay;
            }
            
            // EVALUATION every 200 episodes
            if (episode % 200 == 0 && episode > 0) {
                auto [pure_success, pure_avg_score, avg_length] = evaluateSelfAware(100);
                float training_success = (float)successful_episodes / (episode + 1) * 100.0f;
                float long_game_rate = (float)long_games / (episode + 1) * 100.0f;
                
                std::cout << "\\nEpisode " << episode << std::endl;
                std::cout << "Training: " << training_success << "% success" << std::endl;
                std::cout << "PURE: " << pure_success << "% success, " << pure_avg_score << " avg score, " << avg_length << " avg steps" << std::endl;
                std::cout << "Long games: " << long_game_rate << "%" << std::endl;
                std::cout << "Epsilon: " << epsilon << std::endl;
                
                if (pure_success >= 95.0f && pure_avg_score >= 15.0f && avg_length >= 400.0f) {
                    std::cout << "*** ELITE PERFORMANCE! 95%+ success, 15+ score, 400+ steps! ***" << std::endl;
                    std::string elite_path = "snake_elite_" + std::to_string((int)pure_success) + "percent_" + std::to_string((int)pure_avg_score) + "score_" + std::to_string((int)avg_length) + "steps.bin";
                    network.save(elite_path);
                    std::cout << "Elite model saved: " << elite_path << std::endl;
                }
            }
        }
        
        auto [final_success, final_avg_score, final_avg_length] = evaluateSelfAware(200);
        
        std::cout << std::endl << "=== FINAL RESULTS ===" << std::endl;
        std::cout << "Training episodes: " << episodes << std::endl;
        std::cout << "*** PURE NETWORK PERFORMANCE ***" << std::endl;
        std::cout << "Success rate: " << final_success << "%" << std::endl;
        std::cout << "Average score: " << final_avg_score << std::endl;
        std::cout << "Average game length: " << final_avg_length << " steps" << std::endl;
        
        // Save with all three metrics
        std::string model_path = "snake_selfaware_" + std::to_string((int)final_success) + "percent_" + std::to_string((int)final_avg_score) + "score_" + std::to_string((int)final_avg_length) + "steps.bin";
        try {
            network.save(model_path);
            std::cout << "\\n*** MODEL SAVED TO: " << model_path << " ***" << std::endl;
        } catch (const std::exception& e) {
            std::cout << "Warning: Could not save model - " << e.what() << std::endl;
        }
        
        if (final_success >= 95.0f && final_avg_score >= 15.0f && final_avg_length >= 400.0f) {
            std::cout << "\\n🏆 PERFECT AI! Elite performance in all metrics! 🏆" << std::endl;
        } else if (final_avg_length >= 300.0f && final_avg_score >= 12.0f) {
            std::cout << "\\n✨ EXCELLENT! Long survival with good scores!" << std::endl;
        } else if (final_avg_length >= 200.0f) {
            std::cout << "\\n👍 GOOD! Improved survival length!" << std::endl;
        } else {
            std::cout << "\\n📈 Progress made - focus on survival length" << std::endl;
        }
    }
    
private:
    SnakeNeuralNetwork network;
    SnakeNeuralNetwork target_network;
    const float learning_rate = 0.007f;
    const float gamma = 0.98f;
    
    // ENHANCED 12-feature state with body awareness
    std::vector<float> getSelfAwareState(const SnakeGame& game) {
        auto head = game.getSnakeBody()[0];
        auto food = game.getFoodPosition();
        auto snake = game.getSnakeBody();
        
        std::vector<float> state(12);
        
        // Food direction (4 features)
        state[0] = (food.x > head.x) ? 1.0f : 0.0f;  // Food right
        state[1] = (food.x < head.x) ? 1.0f : 0.0f;  // Food left  
        state[2] = (food.y > head.y) ? 1.0f : 0.0f;  // Food down
        state[3] = (food.y < head.y) ? 1.0f : 0.0f;  // Food up
        
        // Immediate danger (4 features) - includes walls AND body
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
                          checkCollision(testPos, snake));
            state[4 + i] = danger ? 1.0f : 0.0f;
        }
        
        // BODY AWARENESS (4 new features) - look 2 steps ahead for body collision
        for (int i = 0; i < 4; i++) {
            Direction testDir = static_cast<Direction>(i);
            Position testPos = head;
            
            // Look 2 steps ahead
            for (int step = 1; step <= 2; step++) {
                switch (testDir) {
                    case Direction::UP: testPos.y--; break;
                    case Direction::DOWN: testPos.y++; break;
                    case Direction::LEFT: testPos.x--; break;
                    case Direction::RIGHT: testPos.x++; break;
                }
            }
            
            bool body_nearby = false;
            if (testPos.x >= 0 && testPos.x < SnakeGame::GRID_WIDTH &&
                testPos.y >= 0 && testPos.y < SnakeGame::GRID_HEIGHT) {
                body_nearby = checkCollision(testPos, snake);
            }
            state[8 + i] = body_nearby ? 1.0f : 0.0f;
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
    
    // SELF-AWARE reward structure - heavily penalizes self-collision
    float getSelfAwareReward(const SnakeGame& game, bool died, int steps, int foods_eaten, bool died_from_self) {
        if (died) {
            if (died_from_self) {
                // MASSIVE penalty for self-collision
                return -30.0f;
            } else {
                // Smaller penalty for wall collision (sometimes unavoidable)
                return -15.0f;
            }
        }
        
        static Position last_head(-1, -1);
        static int last_score = 0;
        
        auto head = game.getSnakeBody()[0];
        auto food = game.getFoodPosition();
        int current_score = game.getScore();
        
        float reward = 0.1f;  // Base survival reward
        
        // FOOD REWARDS - scaling with length
        if (current_score > last_score) {
            float food_reward = 15.0f + (current_score * 1.0f);  // More reward for longer snake
            reward += food_reward;
            last_score = current_score;
        }
        
        // SURVIVAL BONUS - reward longer games
        if (steps > 200) reward += 0.2f;   // Survived 200+ steps
        if (steps > 500) reward += 0.3f;   // Survived 500+ steps  
        if (steps > 1000) reward += 0.5f;  // Elite survival
        
        // Distance-based guidance
        if (last_head.x >= 0) {
            float old_dist = abs(last_head.x - food.x) + abs(last_head.y - food.y);
            float new_dist = abs(head.x - food.x) + abs(head.y - food.y);
            
            if (new_dist < old_dist) {
                reward += 0.3f;
            } else if (new_dist > old_dist) {
                reward -= 0.1f;
            }
        }
        
        last_head = head;
        return reward;
    }
    
    // Enhanced evaluation with game length tracking
    std::tuple<float, float, float> evaluateSelfAware(int test_episodes) {
        int successful = 0;
        int total_score = 0;
        int total_steps = 0;
        
        for (int episode = 0; episode < test_episodes; episode++) {
            SnakeGame game;
            game.reset();
            
            int steps = 0;
            while (!game.isGameOver() && steps < 3000) {
                auto state = getSelfAwareState(game);
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
        
        auto state_tensor = torch::zeros({1, 12}, torch::kFloat);
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
    std::cout << "=== SELF-AWARE SNAKE AI TRAINER ===" << std::endl;
    
    try {
        SelfAwareDQNTrainer trainer;
        trainer.train(8000, false);
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}
