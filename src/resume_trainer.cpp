#include "neural_network.h"
#include "snake_logic.h"
#include <iostream>
#include <chrono>

// Resume trainer - continues training from existing model
class ResumeTrainer {
public:
    ResumeTrainer() : network(8, 64, 4), target_network(8, 64, 4) {
        // Will load from existing model in train() method
    }
    
    void train(const std::string& model_path, int additional_episodes = 15000, int checkpoint_interval = 2000) {
        std::cout << "=== RESUME TRAINING FROM EXISTING MODEL ===" << std::endl;
        std::cout << "Loading model: " << model_path << std::endl;
        
        try {
            network.load(model_path);
            std::cout << "✓ Model loaded successfully!" << std::endl;
            
            // Copy loaded weights to target network
            auto source_params = network.parameters();
            auto target_params = target_network.parameters();
            torch::NoGradGuard no_grad;
            for (size_t i = 0; i < source_params.size(); i++) {
                target_params[i].copy_(source_params[i]);
            }
            std::cout << "✓ Target network synchronized" << std::endl;
            
        } catch (const std::exception& e) {
            std::cout << "❌ Failed to load model: " << e.what() << std::endl;
            std::cout << "Starting with fresh model instead..." << std::endl;
            
            // Initialize fresh networks if loading fails
            auto source_params = network.parameters();
            auto target_params = target_network.parameters();
            torch::NoGradGuard no_grad;
            for (size_t i = 0; i < source_params.size(); i++) {
                target_params[i].copy_(source_params[i]);
            }
        }
        
        // Test current model performance before training
        std::cout << "\\n=== TESTING LOADED MODEL ===" << std::endl;
        float initial_performance = evaluatePureNetwork(100);
        std::cout << "Current model performance: " << initial_performance << "%" << std::endl;
        
        std::cout << "\\n=== BEGINNING EXTENDED TRAINING ===" << std::endl;
        std::cout << "Additional episodes: " << additional_episodes << " | Checkpoints every: " << checkpoint_interval << std::endl;
        std::cout << "Network: 8->64->4 | Resuming from " << initial_performance << "% baseline" << std::endl;
        
        // Reduced epsilon since model already trained
        float epsilon = 0.15f;  // Start with lower exploration since model is already good
        const float epsilon_decay = 0.99995f;
        const float epsilon_min = 0.01f;
        
        int total_score = 0;
        int successful_episodes = 0;
        auto start_time = std::chrono::high_resolution_clock::now();
        
        std::vector<float> performance_history;
        float best_performance = initial_performance;
        performance_history.push_back(initial_performance);
        
        for (int episode = 0; episode < additional_episodes; episode++) {
            SnakeGame game;
            game.reset();
            
            std::vector<float> prev_state;
            int prev_action = -1;
            float total_reward = 0.0f;
            int steps = 0;
            
            while (!game.isGameOver() && steps < 1200) {
                auto current_state = getOptimizedState(game);
                int action = getAction(current_state, epsilon);
                
                game.setDirection(static_cast<Direction>(action));
                bool game_continues = game.update();
                
                float reward = getOptimizedReward(game, !game_continues, steps);
                total_reward += reward;
                steps++;
                
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
            
            // More frequent target updates for fine-tuning
            if (episode % 40 == 0 && episode > 0) {
                updateTargetNetwork();
            }
            
            if (epsilon > epsilon_min) {
                epsilon *= epsilon_decay;
            }
            
            // Performance evaluation every 400 episodes
            if (episode % 400 == 0 && episode > 0) {
                float current_performance = evaluatePureNetwork(150);
                performance_history.push_back(current_performance);
                
                if (current_performance > best_performance) {
                    best_performance = current_performance;
                    std::string best_path = "snake_resume_best_" + std::to_string((int)current_performance) + "percent.bin";
                    network.save(best_path);
                    std::cout << "*** NEW BEST: " << current_performance << "% (+" << (current_performance - initial_performance) 
                             << "% improvement) - Saved: " << best_path << " ***" << std::endl;
                }
                
                float training_success = (float)successful_episodes / (episode + 1) * 100.0f;
                auto current_time = std::chrono::high_resolution_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::minutes>(current_time - start_time).count();
                
                std::cout << "\\nEpisode " << episode << " (" << elapsed << "m) - Training: " << training_success 
                         << "% | PURE: " << current_performance << "% | Best: " << best_performance 
                         << "% | Improvement: +" << (current_performance - initial_performance) 
                         << "% | Epsilon: " << epsilon << std::endl;
                
                // Performance trend
                if (performance_history.size() >= 3) {
                    float recent_avg = (performance_history[performance_history.size()-1] + 
                                      performance_history[performance_history.size()-2] + 
                                      performance_history[performance_history.size()-3]) / 3.0f;
                    std::cout << "Recent 3-eval avg: " << recent_avg << "% | Trend: ";
                    if (recent_avg > initial_performance + 1.0f) {
                        std::cout << "IMPROVING ↗" << std::endl;
                    } else if (recent_avg < initial_performance - 1.0f) {
                        std::cout << "declining ↘" << std::endl;
                    } else {
                        std::cout << "stable →" << std::endl;
                    }
                }
                
                if (current_performance >= 99.0f) {
                    std::cout << "*** NEAR-PERFECT! 99%+ ACHIEVED! ***" << std::endl;
                }
            }
            
            // Checkpoint saves
            if (episode % checkpoint_interval == 0 && episode > 0) {
                float checkpoint_performance = evaluatePureNetwork(100);
                std::string checkpoint_path = "snake_resume_ep" + std::to_string(episode) + "_" + std::to_string((int)checkpoint_performance) + "percent.bin";
                network.save(checkpoint_path);
                std::cout << "\\n=== CHECKPOINT " << episode << " === Performance: " << checkpoint_performance << "% - Saved: " << checkpoint_path << std::endl;
                
                // Early stopping for exceptional improvement
                if (checkpoint_performance >= initial_performance + 5.0f) {
                    std::cout << "*** SIGNIFICANT IMPROVEMENT! +" << (checkpoint_performance - initial_performance) << "% gain ***" << std::endl;
                }
                
                if (checkpoint_performance >= 99.5f) {
                    std::cout << "\\n*** NEAR-PERFECT PERFORMANCE! 99.5%+ achieved at episode " << episode << " ***" << std::endl;
                    std::cout << "Consider stopping training to avoid overfitting." << std::endl;
                }
            }
            
            // Progress indicator
            if (episode % 1000 == 0 && episode > 0) {
                float progress = (float)episode / additional_episodes * 100.0f;
                auto current_time = std::chrono::high_resolution_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::minutes>(current_time - start_time).count();
                float episodes_per_minute = (float)episode / elapsed;
                int estimated_remaining = (int)((additional_episodes - episode) / episodes_per_minute);
                
                std::cout << "Progress: " << progress << "% (" << episode << "/" << additional_episodes 
                         << ") | " << elapsed << "m elapsed | ~" << estimated_remaining << "m remaining" << std::endl;
            }
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto total_time = std::chrono::duration_cast<std::chrono::minutes>(end_time - start_time).count();
        
        float final_avg = (float)total_score / additional_episodes;
        float final_success_rate = (float)successful_episodes / additional_episodes * 100.0f;
        float final_pure = evaluatePureNetwork(300);
        
        std::cout << std::endl << "=== RESUME TRAINING COMPLETE ===" << std::endl;
        std::cout << "Total additional training time: " << total_time << " minutes" << std::endl;
        std::cout << "Additional episodes: " << additional_episodes << std::endl;
        std::cout << "Training success (with exploration): " << final_success_rate << "%" << std::endl;
        std::cout << "*** INITIAL PERFORMANCE: " << initial_performance << "% ***" << std::endl;
        std::cout << "*** FINAL PERFORMANCE: " << final_pure << "% ***" << std::endl;
        std::cout << "*** BEST ACHIEVED: " << best_performance << "% ***" << std::endl;
        std::cout << "*** TOTAL IMPROVEMENT: +" << (final_pure - initial_performance) << "% ***" << std::endl;
        std::cout << "Average score: " << final_avg << std::endl;
        
        // Save final extended model
        std::string final_model_path = "snake_resume_final_" + std::to_string((int)final_pure) + "percent.bin";
        try {
            network.save(final_model_path);
            std::cout << "\\n*** FINAL EXTENDED MODEL SAVED TO: " << final_model_path << " ***" << std::endl;
        } catch (const std::exception& e) {
            std::cout << "Warning: Could not save final model - " << e.what() << std::endl;
        }
        
        // Performance improvement analysis
        float total_improvement = final_pure - initial_performance;
        if (total_improvement >= 3.0f) {
            std::cout << "\\n🚀 SIGNIFICANT IMPROVEMENT! +" << total_improvement << "% gain from extended training! 🚀" << std::endl;
        } else if (total_improvement >= 1.0f) {
            std::cout << "\\n✨ GOOD IMPROVEMENT! +" << total_improvement << "% gain from extended training!" << std::endl;
        } else if (total_improvement >= 0.0f) {
            std::cout << "\\n👍 MAINTAINED PERFORMANCE! +" << total_improvement << "% (stable)" << std::endl;
        } else {
            std::cout << "\\n⚠️ SLIGHT DECLINE: " << total_improvement << "% - Consider using earlier checkpoint" << std::endl;
        }
        
        std::cout << "\\n=== TRAINING SESSION SUMMARY ===" << std::endl;
        std::cout << "Best model: snake_resume_best_" << (int)best_performance << "percent.bin" << std::endl;
        std::cout << "Final model: " << final_model_path << std::endl;
        std::cout << "Total checkpoints: " << (additional_episodes / checkpoint_interval) << std::endl;
        std::cout << "Training efficiency: " << (float)additional_episodes / total_time << " episodes/minute" << std::endl;
    }
    
private:
    SnakeNeuralNetwork network;
    SnakeNeuralNetwork target_network;
    const float learning_rate = 0.005f;  // Reduced for fine-tuning
    const float gamma = 0.98f;
    
    std::vector<float> getOptimizedState(const SnakeGame& game) {
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
            while (!game.isGameOver() && steps < 1200) {
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
    std::cout << "=== RESUME TRAINING FROM EXISTING MODEL ===" << std::endl;
    std::cout << "This trainer loads an existing model and continues training" << std::endl;
    std::cout << "Optimized for fine-tuning with reduced learning rate and exploration" << std::endl;
    std::cout << std::endl;
    
    try {
        ResumeTrainer trainer;
        
        // Load your successful model and train for 15,000 more episodes
        std::string model_path = "snake_extended_final_97percent.bin";
        int additional_episodes = 15000;
        int checkpoint_interval = 2000;
        
        trainer.train(model_path, additional_episodes, checkpoint_interval);
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}
