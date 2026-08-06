#include "neural_network.h"
#include "snake_logic.h"
#include <iostream>
#include <chrono>

// Extended Optimized trainer for longer training sessions
class ExtendedOptimizedDQNTrainer {
public:
    ExtendedOptimizedDQNTrainer() : network(8, 64, 4), target_network(8, 64, 4) {
        // Copy initial weights
        auto source_params = network.parameters();
        auto target_params = target_network.parameters();
        torch::NoGradGuard no_grad;
        for (size_t i = 0; i < source_params.size(); i++) {
            target_params[i].copy_(source_params[i]);
        }
    }
    
    void train(int episodes = 20000, bool visual = false, int checkpoint_interval = 2000) {
        std::cout << "=== EXTENDED OPTIMIZED TRAINER ===" << std::endl;
        std::cout << "Target Episodes: " << episodes << " | Checkpoints every: " << checkpoint_interval << std::endl;
        std::cout << "Network: 8->64->4 | Proven Features | Extended Training" << std::endl;
        
        float epsilon = 1.0f;
        const float epsilon_decay = 0.99985f;  // Slower decay for longer training
        const float epsilon_min = 0.02f;       // Slightly lower minimum
        
        int total_score = 0;
        int successful_episodes = 0;
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Track performance over time
        std::vector<float> performance_history;
        float best_performance = 0.0f;
        
        for (int episode = 0; episode < episodes; episode++) {
            SnakeGame game;
            game.reset();
            
            std::vector<float> prev_state;
            int prev_action = -1;
            float total_reward = 0.0f;
            int steps = 0;
            
            while (!game.isGameOver() && steps < 1000) {  // Extended episode length
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
            
            // Update target network more frequently for longer training
            if (episode % 50 == 0 && episode > 0) {
                updateTargetNetwork();
            }
            
            // Epsilon decay
            if (epsilon > epsilon_min) {
                epsilon *= epsilon_decay;
            }
            
            // Extended evaluation every 500 episodes
            if (episode % 500 == 0 && episode > 0) {
                float pure_performance = evaluatePureNetwork(150);  // More thorough testing
                performance_history.push_back(pure_performance);
                
                if (pure_performance > best_performance) {
                    best_performance = pure_performance;
                    // Save best model
                    std::string best_path = "snake_best_" + std::to_string((int)pure_performance) + "percent.bin";
                    network.save(best_path);
                    std::cout << "*** NEW BEST: " << pure_performance << "% - Model saved: " << best_path << " ***" << std::endl;
                }
                
                float training_success = (float)successful_episodes / (episode + 1) * 100.0f;
                
                auto current_time = std::chrono::high_resolution_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::minutes>(current_time - start_time).count();
                
                std::cout << "\\nEpisode " << episode << " (" << elapsed << "m) - Training: " << training_success 
                         << "% | PURE: " << pure_performance << "% | Best: " << best_performance 
                         << "% | Epsilon: " << epsilon << std::endl;
                
                // Performance trend analysis
                if (performance_history.size() >= 3) {
                    float recent_avg = (performance_history[performance_history.size()-1] + 
                                      performance_history[performance_history.size()-2] + 
                                      performance_history[performance_history.size()-3]) / 3.0f;
                    std::cout << "Recent 3-checkpoint avg: " << recent_avg << "%" << std::endl;
                }
                
                if (pure_performance >= 95.0f) {
                    std::cout << "*** EXCEPTIONAL PERFORMANCE! 95%+ ACHIEVED! ***" << std::endl;
                }
            }
            
            // Checkpoint saves
            if (episode % checkpoint_interval == 0 && episode > 0) {
                float checkpoint_performance = evaluatePureNetwork(100);
                std::string checkpoint_path = "snake_checkpoint_ep" + std::to_string(episode) + "_" + std::to_string((int)checkpoint_performance) + "percent.bin";
                network.save(checkpoint_path);
                std::cout << "\\n=== CHECKPOINT " << episode << " === Performance: " << checkpoint_performance << "% - Saved: " << checkpoint_path << std::endl;
                
                // Early stopping for exceptional performance
                if (checkpoint_performance >= 98.0f) {
                    std::cout << "\\n*** EARLY SUCCESS! 98%+ achieved at episode " << episode << " ***" << std::endl;
                    std::cout << "Exceptional performance reached - training can be stopped or continued for further optimization." << std::endl;
                }
            }
            
            // Progress indicator for long training
            if (episode % 1000 == 0 && episode > 0) {
                float progress = (float)episode / episodes * 100.0f;
                auto current_time = std::chrono::high_resolution_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::minutes>(current_time - start_time).count();
                float episodes_per_minute = (float)episode / elapsed;
                int estimated_remaining = (int)((episodes - episode) / episodes_per_minute);
                
                std::cout << "Progress: " << progress << "% (" << episode << "/" << episodes 
                         << ") | " << elapsed << "m elapsed | ~" << estimated_remaining << "m remaining" << std::endl;
            }
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto total_time = std::chrono::duration_cast<std::chrono::minutes>(end_time - start_time).count();
        
        float final_avg = (float)total_score / episodes;
        float final_success_rate = (float)successful_episodes / episodes * 100.0f;
        float final_pure = evaluatePureNetwork(300);  // Comprehensive final test
        
        std::cout << std::endl << "=== EXTENDED TRAINING COMPLETE ===" << std::endl;
        std::cout << "Total training time: " << total_time << " minutes" << std::endl;
        std::cout << "Training episodes: " << episodes << std::endl;
        std::cout << "Training success (with exploration): " << final_success_rate << "%" << std::endl;
        std::cout << "*** FINAL PURE NETWORK PERFORMANCE: " << final_pure << "% ***" << std::endl;
        std::cout << "*** BEST PERFORMANCE ACHIEVED: " << best_performance << "% ***" << std::endl;
        std::cout << "Average score: " << final_avg << std::endl;
        
        // Performance improvement analysis
        if (!performance_history.empty()) {
            float first_performance = performance_history[0];
            float improvement = final_pure - first_performance;
            std::cout << "Performance improvement: +" << improvement << "% (from " << first_performance << "% to " << final_pure << "%)" << std::endl;
        }
        
        // Save final model
        std::string final_model_path = "snake_extended_final_" + std::to_string((int)final_pure) + "percent.bin";
        try {
            network.save(final_model_path);
            std::cout << "\\n*** FINAL MODEL SAVED TO: " << final_model_path << " ***" << std::endl;
        } catch (const std::exception& e) {
            std::cout << "Warning: Could not save final model - " << e.what() << std::endl;
        }
        
        // Performance classification
        if (final_pure >= 98.0f) {
            std::cout << "\\n🏆 EXCEPTIONAL! 98%+ Performance - Near Perfect AI! 🏆" << std::endl;
        } else if (final_pure >= 95.0f) {
            std::cout << "\\n🎉 OUTSTANDING! 95%+ Performance - Elite AI! 🎉" << std::endl;
        } else if (final_pure >= 90.0f) {
            std::cout << "\\n🎉 MISSION ACCOMPLISHED! 90%+ Performance! 🎉" << std::endl;
        } else if (final_pure >= 80.0f) {
            std::cout << "\\n✨ EXCELLENT! 80%+ Performance!" << std::endl;
        } else if (final_pure >= 70.0f) {
            std::cout << "\\n👍 GOOD! 70%+ Performance!" << std::endl;
        } else {
            std::cout << "\\n📊 Extended baseline established - " << final_pure << "% performance" << std::endl;
        }
        
        std::cout << "\\n=== TRAINING SESSION SUMMARY ===" << std::endl;
        std::cout << "Best model available: snake_best_" << (int)best_performance << "percent.bin" << std::endl;
        std::cout << "Total checkpoints saved: " << (episodes / checkpoint_interval) << std::endl;
        std::cout << "Training efficiency: " << (float)episodes / total_time << " episodes/minute" << std::endl;
    }
    
private:
    SnakeNeuralNetwork network;
    SnakeNeuralNetwork target_network;
    const float learning_rate = 0.006f;  // Slightly reduced for longer training
    const float gamma = 0.98f;           // Higher discount for longer-term planning
    
    // Same proven 8-feature state as optimized trainer
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
    
    float getOptimizedReward(const SnakeGame& game, bool died, int steps) {
        if (died) return -15.0f;
        
        static Position last_head(-1, -1);
        static int last_score = 0;
        
        auto head = game.getSnakeBody()[0];
        auto food = game.getFoodPosition();
        int current_score = game.getScore();
        
        float reward = 0.1f;
        
        if (current_score > last_score) {
            reward += 15.0f;
            last_score = current_score;
        }
        
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
    
    float evaluatePureNetwork(int test_episodes) {
        int successful = 0;
        
        for (int episode = 0; episode < test_episodes; episode++) {
            SnakeGame game;
            game.reset();
            
            int steps = 0;
            while (!game.isGameOver() && steps < 1000) {
                auto state = getOptimizedState(game);
                auto action_tensor = network.getAction(state, 0.0f);
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
    std::cout << "=== EXTENDED OPTIMIZED SNAKE AI TRAINER ===" << std::endl;
    std::cout << "This trainer is designed for longer training sessions with:" << std::endl;
    std::cout << "- Extended episode count (20,000 default)" << std::endl;
    std::cout << "- Regular checkpoints and best model saving" << std::endl;
    std::cout << "- Performance tracking and trend analysis" << std::endl;
    std::cout << "- Time estimation and progress monitoring" << std::endl;
    std::cout << std::endl;
    
    try {
        ExtendedOptimizedDQNTrainer trainer;
        
        // You can modify these parameters:
        int episodes = 20000;        // Total episodes to train
        int checkpoint_interval = 2000;  // Save checkpoint every N episodes
        
        trainer.train(episodes, false, checkpoint_interval);
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}
