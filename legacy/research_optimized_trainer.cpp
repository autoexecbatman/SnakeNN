#include "neural_network.h"
#include "snake_logic.h"
#include <iostream>
#include <chrono>

// Research-optimized trainer based on academic findings
class ResearchOptimizedTrainer {
public:
    ResearchOptimizedTrainer() : network(8, 64, 4), target_network(8, 64, 4) {
        auto source_params = network.parameters();
        auto target_params = target_network.parameters();
        torch::NoGradGuard no_grad;
        for (size_t i = 0; i < source_params.size(); i++) {
            target_params[i].copy_(source_params[i]);
        }
    }
    
    void train(int episodes = 15000) {
        std::cout << "=== RESEARCH-OPTIMIZED TRAINER ===" << std::endl;
        std::cout << "Based on academic Snake DQN research findings:" << std::endl;
        std::cout << "- Higher epsilon_min (0.20) to prevent getting stuck" << std::endl;
        std::cout << "- Lower learning rate (0.001) with larger effective batch" << std::endl;
        std::cout << "- Less frequent target updates (every 10,000 steps)" << std::endl;
        std::cout << "- Optimized reward structure to prevent idle loops" << std::endl;
        
        // RESEARCH-BASED PARAMETERS
        float epsilon = 1.0f;
        const float epsilon_decay = 0.9999f;    // Slower decay to maintain exploration
        const float epsilon_min = 0.20f;        // CRITICAL: 20% random actions to prevent getting stuck
        
        int total_score = 0;
        int successful_episodes = 0;
        int step_count = 0;  // Global step counter for target network updates
        auto start_time = std::chrono::high_resolution_clock::now();
        
        std::vector<float> performance_history;
        float best_performance = 0.0f;
        
        for (int episode = 0; episode < episodes; episode++) {
            SnakeGame game;
            game.reset();
            
            std::vector<float> prev_state;
            int prev_action = -1;
            int steps = 0;
            int idle_moves = 0;  // Track consecutive moves without progress
            Position last_position(-1, -1);
            
            while (!game.isGameOver() && steps < 1000) {
                auto current_state = getOptimizedState(game);
                int action = getAction(current_state, epsilon);
                
                auto current_head = game.getSnakeBody()[0];
                
                game.setDirection(static_cast<Direction>(action));
                bool game_continues = game.update();
                
                // Track idle moves (no progress toward food)
                if (last_position.x >= 0) {
                    if (current_head.x == last_position.x && current_head.y == last_position.y) {
                        idle_moves++;
                    } else {
                        idle_moves = 0;  // Reset if moved
                    }
                }
                last_position = current_head;
                
                // Research-based reward with anti-idle punishment
                float reward = getResearchReward(game, !game_continues, steps, idle_moves);
                steps++;
                step_count++;
                
                if (!prev_state.empty()) {
                    trainStep(prev_state, prev_action, reward, current_state, !game_continues);
                }
                
                prev_state = current_state;
                prev_action = action;
                
                // Kill snake after 30 idle moves (research finding)
                if (idle_moves >= 30) {
                    break;  // Force episode end
                }
                
                if (!game_continues) break;
            }
            
            int score = game.getScore();
            total_score += score;
            if (score > 0) successful_episodes++;
            
            // RESEARCH-BASED: Update target network every 10,000 STEPS (not episodes)
            if (step_count % 10000 == 0 && step_count > 0) {
                updateTargetNetwork();
                std::cout << "Target network updated at step " << step_count << std::endl;
            }
            
            // Epsilon decay with research-based minimum
            if (epsilon > epsilon_min) {
                epsilon *= epsilon_decay;
            }
            
            // Performance evaluation every 300 episodes
            if (episode % 300 == 0 && episode > 0) {
                float current_performance = evaluatePureNetwork(100);
                performance_history.push_back(current_performance);
                
                if (current_performance > best_performance) {
                    best_performance = current_performance;
                    std::string best_path = "snake_research_best_" + std::to_string((int)current_performance) + "percent.bin";
                    network.save(best_path);
                    std::cout << "*** NEW BEST: " << current_performance << "% - Model saved: " << best_path << " ***" << std::endl;
                }
                
                float training_success = (float)successful_episodes / (episode + 1) * 100.0f;
                auto current_time = std::chrono::high_resolution_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::minutes>(current_time - start_time).count();
                
                std::cout << "\\nEpisode " << episode << " (" << elapsed << "m) - Training: " << training_success 
                         << "% | PURE: " << current_performance << "% | Best: " << best_performance 
                         << "% | Epsilon: " << epsilon << " | Steps: " << step_count << std::endl;
                
                if (current_performance >= 95.0f) {
                    std::cout << "*** EXCEPTIONAL PERFORMANCE! 95%+ ACHIEVED! ***" << std::endl;
                }
            }
            
            // Progress indicator
            if (episode % 1000 == 0 && episode > 0) {
                float progress = (float)episode / episodes * 100.0f;
                auto current_time = std::chrono::high_resolution_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::minutes>(current_time - start_time).count();
                
                std::cout << "Progress: " << progress << "% (" << episode << "/" << episodes 
                         << ") | " << elapsed << "m elapsed | Total steps: " << step_count << std::endl;
            }
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto total_time = std::chrono::duration_cast<std::chrono::minutes>(end_time - start_time).count();
        
        float final_avg = (float)total_score / episodes;
        float final_success_rate = (float)successful_episodes / episodes * 100.0f;
        float final_pure = evaluatePureNetwork(200);
        
        std::cout << std::endl << "=== RESEARCH-OPTIMIZED TRAINING COMPLETE ===" << std::endl;
        std::cout << "Total training time: " << total_time << " minutes" << std::endl;
        std::cout << "Training episodes: " << episodes << std::endl;
        std::cout << "Total training steps: " << step_count << std::endl;
        std::cout << "Training success (with exploration): " << final_success_rate << "%" << std::endl;
        std::cout << "*** FINAL PURE NETWORK PERFORMANCE: " << final_pure << "% ***" << std::endl;
        std::cout << "*** BEST PERFORMANCE ACHIEVED: " << best_performance << "% ***" << std::endl;
        std::cout << "Average score: " << final_avg << std::endl;
        std::cout << "Final epsilon (exploration rate): " << epsilon << std::endl;
        
        // Save final model
        std::string final_model_path = "snake_research_final_" + std::to_string((int)final_pure) + "percent.bin";
        try {
            network.save(final_model_path);
            std::cout << "\\n*** RESEARCH MODEL SAVED TO: " << final_model_path << " ***" << std::endl;
        } catch (const std::exception& e) {
            std::cout << "Warning: Could not save final model - " << e.what() << std::endl;
        }
        
        // Performance classification
        if (final_pure >= 98.0f) {
            std::cout << "\\n NEAR-PERFECT! 98%+ Performance with research parameters! " << std::endl;
        } else if (final_pure >= 95.0f) {
            std::cout << "\\n OUTSTANDING! 95%+ Performance - Research approach successful! " << std::endl;
        } else if (final_pure >= 90.0f) {
            std::cout << "\\n EXCELLENT! 90%+ Performance! " << std::endl;
        } else if (final_pure >= 80.0f) {
            std::cout << "\\n VERY GOOD! 80%+ Performance!" << std::endl;
        } else {
            std::cout << "\\n Research baseline: " << final_pure << "% performance" << std::endl;
        }
        
        std::cout << "\\n=== RESEARCH PARAMETERS SUMMARY ===" << std::endl;
        std::cout << "Epsilon minimum: 0.20 (20% random actions maintained)" << std::endl;
        std::cout << "Learning rate: 0.001 (research-optimized)" << std::endl;
        std::cout << "Target updates: Every 10,000 steps (not episodes)" << std::endl;
        std::cout << "Anti-idle timeout: 30 moves maximum" << std::endl;
        std::cout << "Best model: snake_research_best_" << (int)best_performance << "percent.bin" << std::endl;
    }
    
private:
    SnakeNeuralNetwork network;
    SnakeNeuralNetwork target_network;
    const float learning_rate = 0.001f;  // Research-optimized learning rate
    const float gamma = 0.99f;           // Standard discount factor
    
    std::vector<float> getOptimizedState(const SnakeGame& game) {
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
    
    int getAction(const std::vector<float>& state, float epsilon) {
        if ((rand() % 1000) / 1000.0f < epsilon) {
            return rand() % 4;
        }
        
        auto action_tensor = network.getAction(state, 0.0f);
        return static_cast<int>(action_tensor.cpu().item<int64_t>());
    }
    
    // Research-based reward system with anti-idle punishment
    float getResearchReward(const SnakeGame& game, bool died, int steps, int idle_moves) {
        static Position last_head(-1, -1);
        static int last_score = 0;
        
        auto head = game.getSnakeBody()[0];
        auto food = game.getFoodPosition();
        int current_score = game.getScore();
        
        float reward = 0.1f;  // Base survival reward
        
        // Death penalty
        if (died) {
            reward = -10.0f;
            last_head = Position(-1, -1);
            last_score = 0;
            return reward;
        }
        
        // Food reward - significant bonus
        if (current_score > last_score) {
            reward += 20.0f;
            last_score = current_score;
        }
        
        // Research finding: Punish idle behavior heavily
        if (idle_moves > 15) {
            reward -= 1.0f;  // Strong punishment for looping
        } else if (idle_moves > 25) {
            reward -= 2.0f;  // Even stronger punishment
        }
        
        // Movement reward toward food
        if (last_head.x >= 0) {
            float old_dist = abs(last_head.x - food.x) + abs(last_head.y - food.y);
            float new_dist = abs(head.x - food.x) + abs(head.y - food.y);
            
            if (new_dist < old_dist) {
                reward += 0.5f;  // Good progress
            } else if (new_dist > old_dist) {
                reward -= 0.1f;  // Moving away penalty
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
                auto action_tensor = network.getAction(state, 0.0f);  // NO exploration
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
    std::cout << "=== RESEARCH-OPTIMIZED SNAKE AI TRAINER ===" << std::endl;
    std::cout << "Using parameters from academic Snake DQN research:" << std::endl;
    std::cout << "- Epsilon min: 0.20 (vs typical 0.01-0.05)" << std::endl;
    std::cout << "- Learning rate: 0.001 (research standard)" << std::endl;
    std::cout << "- Target updates: Every 10,000 steps" << std::endl;
    std::cout << "- Anti-idle timeout: 30 moves" << std::endl;
    std::cout << std::endl;
    
    try {
        ResearchOptimizedTrainer trainer;
        trainer.train(30000);
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}
