#include "neural_network.h"
#include "snake_logic.h"
#include <iostream>
#include <chrono>

// High-score focused trainer - optimizes for winning, not just surviving
class WinningTrainer {
public:
    WinningTrainer() : network(11, 128, 4), target_network(11, 128, 4) {  // Larger network for complex decisions
        auto source_params = network.parameters();
        auto target_params = target_network.parameters();
        torch::NoGradGuard no_grad;
        for (size_t i = 0; i < source_params.size(); i++) {
            target_params[i].copy_(source_params[i]);
        }
    }
    
    void train(int episodes = 25000) {
        std::cout << "=== WINNING-FOCUSED TRAINER ===" << std::endl;
        std::cout << "Goal: High scores (15+ foods), not just survival" << std::endl;
        std::cout << "- Enhanced state representation (11 features)" << std::endl;
        std::cout << "- Larger network (11->128->4)" << std::endl;
        std::cout << "- Score-based rewards, not just survival" << std::endl;
        std::cout << "- Space awareness and long-term planning" << std::endl;
        
        float epsilon = 1.0f;
        const float epsilon_decay = 0.99995f;
        const float epsilon_min = 0.05f;  // Lower than research for more exploitation
        
        int total_score = 0;
        int high_score_games = 0;  // Games with 10+ foods
        int winning_games = 0;     // Games with 15+ foods
        auto start_time = std::chrono::high_resolution_clock::now();
        
        std::vector<float> score_history;
        float best_avg_score = 0.0f;
        
        for (int episode = 0; episode < episodes; episode++) {
            SnakeGame game;
            game.reset();
            
            std::vector<float> prev_state;
            int prev_action = -1;
            int steps = 0;
            
            while (!game.isGameOver() && steps < 2000) {  // Allow very long games
                auto current_state = getWinningState(game);
                int action = getAction(current_state, epsilon);
                
                game.setDirection(static_cast<Direction>(action));
                bool game_continues = game.update();
                
                float reward = getWinningReward(game, !game_continues, steps);
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
            if (score >= 10) high_score_games++;
            if (score >= 15) winning_games++;
            
            // Target network updates
            if (episode % 100 == 0 && episode > 0) {
                updateTargetNetwork();
            }
            
            if (epsilon > epsilon_min) {
                epsilon *= epsilon_decay;
            }
            
            // Score-focused evaluation every 500 episodes
            if (episode % 500 == 0 && episode > 0) {
                auto score_metrics = evaluateScores(100);
                float avg_score = score_metrics.avg_score;
                float high_score_rate = score_metrics.high_score_rate;
                float winning_rate = score_metrics.winning_rate;
                float max_score = score_metrics.max_score;
                
                score_history.push_back(avg_score);
                
                if (avg_score > best_avg_score) {
                    best_avg_score = avg_score;
                    std::string best_path = "snake_winning_best_" + std::to_string((int)avg_score) + "avg.bin";
                    network.save(best_path);
                    std::cout << "*** NEW BEST AVG: " << avg_score << " foods - Model saved: " << best_path << " ***" << std::endl;
                }
                
                float training_high_rate = (float)high_score_games / (episode + 1) * 100.0f;
                float training_win_rate = (float)winning_games / (episode + 1) * 100.0f;
                float training_avg = (float)total_score / (episode + 1);
                
                auto current_time = std::chrono::high_resolution_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::minutes>(current_time - start_time).count();
                
                std::cout << "\\nEpisode " << episode << " (" << elapsed << "m):" << std::endl;
                std::cout << "  Training: " << training_avg << " avg | " << training_high_rate 
                         << "% high (10+) | " << training_win_rate << "% wins (15+)" << std::endl;
                std::cout << "  PURE TEST: " << avg_score << " avg | " << high_score_rate 
                         << "% high | " << winning_rate << "% wins | MAX: " << max_score << std::endl;
                std::cout << "  Epsilon: " << epsilon << std::endl;
                
                if (winning_rate >= 50.0f) {
                    std::cout << "*** WINNING AI! 50%+ games with 15+ foods! ***" << std::endl;
                }
                if (avg_score >= 12.0f) {
                    std::cout << "*** HIGH PERFORMANCE! 12+ average score! ***" << std::endl;
                }
            }
            
            // Progress tracking
            if (episode % 2000 == 0 && episode > 0) {
                float progress = (float)episode / episodes * 100.0f;
                auto current_time = std::chrono::high_resolution_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::minutes>(current_time - start_time).count();
                
                std::cout << "Progress: " << progress << "% (" << episode << "/" << episodes 
                         << ") | " << elapsed << "m elapsed" << std::endl;
            }
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto total_time = std::chrono::duration_cast<std::chrono::minutes>(end_time - start_time).count();
        
        auto final_metrics = evaluateScores(200);
        float final_avg = final_metrics.avg_score;
        float final_high_rate = final_metrics.high_score_rate;
        float final_win_rate = final_metrics.winning_rate;
        float final_max = final_metrics.max_score;
        
        std::cout << std::endl << "=== WINNING TRAINER RESULTS ===" << std::endl;
        std::cout << "Total training time: " << total_time << " minutes" << std::endl;
        std::cout << "Training episodes: " << episodes << std::endl;
        std::cout << "*** AVERAGE SCORE: " << final_avg << " foods per game ***" << std::endl;
        std::cout << "*** HIGH SCORES (10+): " << final_high_rate << "% of games ***" << std::endl;
        std::cout << "*** WINNING GAMES (15+): " << final_win_rate << "% of games ***" << std::endl;
        std::cout << "*** MAXIMUM SCORE: " << final_max << " foods ***" << std::endl;
        std::cout << "*** BEST AVERAGE ACHIEVED: " << best_avg_score << " foods ***" << std::endl;
        
        // Save final model
        std::string final_path = "snake_winning_final_" + std::to_string((int)final_avg) + "avg.bin";
        network.save(final_path);
        std::cout << "\\n*** WINNING MODEL SAVED: " << final_path << " ***" << std::endl;
        
        // Performance classification
        if (final_win_rate >= 75.0f) {
            std::cout << "\\n*** CHAMPION! 75%+ winning games! ***" << std::endl;
        } else if (final_win_rate >= 50.0f) {
            std::cout << "\\n*** WINNER! 50%+ winning games! ***" << std::endl;
        } else if (final_win_rate >= 25.0f) {
            std::cout << "\\n*** STRONG PLAYER! 25%+ winning games! ***" << std::endl;
        } else if (final_avg >= 10.0f) {
            std::cout << "\\n*** GOOD PLAYER! 10+ average score! ***" << std::endl;
        } else {
            std::cout << "\n*** Training baseline: " << final_avg << " avg score, " << final_win_rate << "% wins ***" << std::endl;
        }
        
        std::cout << "\\nBest model: snake_winning_best_" << (int)best_avg_score << "avg.bin" << std::endl;
    }
    
private:
    SnakeNeuralNetwork network;
    SnakeNeuralNetwork target_network;
    const float learning_rate = 0.005f;
    const float gamma = 0.995f;  // Higher discount for long-term planning
    
    struct ScoreMetrics {
        float avg_score;
        float high_score_rate;  // % with 10+ foods
        float winning_rate;     // % with 15+ foods
        float max_score;
    };
    
    // Enhanced state representation for winning (11 features)
    std::vector<float> getWinningState(const SnakeGame& game) {
        auto head = game.getSnakeBody()[0];
        auto food = game.getFoodPosition();
        auto snake = game.getSnakeBody();
        
        std::vector<float> state(11);
        
        // Food direction (4 features) - same as before
        state[0] = (food.x > head.x) ? 1.0f : 0.0f;
        state[1] = (food.x < head.x) ? 1.0f : 0.0f;
        state[2] = (food.y > head.y) ? 1.0f : 0.0f;
        state[3] = (food.y < head.y) ? 1.0f : 0.0f;
        
        // Immediate danger (4 features) - same as before
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
        
        // NEW FEATURES FOR WINNING:
        
        // Snake length (normalized)
        state[8] = (float)snake.size() / (SnakeGame::GRID_WIDTH * SnakeGame::GRID_HEIGHT);
        
        // Space availability (how much free space around head)
        int free_spaces = 0;
        for (int dx = -2; dx <= 2; dx++) {
            for (int dy = -2; dy <= 2; dy++) {
                Position test_pos = {head.x + dx, head.y + dy};
                if (test_pos.x >= 0 && test_pos.x < SnakeGame::GRID_WIDTH &&
                    test_pos.y >= 0 && test_pos.y < SnakeGame::GRID_HEIGHT &&
                    !checkCollision(test_pos, snake)) {
                    free_spaces++;
                }
            }
        }
        state[9] = (float)free_spaces / 25.0f;  // Normalized by 5x5 area
        
        // Distance to tail (escape route awareness)
        auto tail = snake.back();
        float tail_distance = abs(head.x - tail.x) + abs(head.y - tail.y);
        state[10] = tail_distance / (SnakeGame::GRID_WIDTH + SnakeGame::GRID_HEIGHT);
        
        return state;
    }
    
    int getAction(const std::vector<float>& state, float epsilon) {
        if ((rand() % 1000) / 1000.0f < epsilon) {
            return rand() % 4;
        }
        
        auto action_tensor = network.getAction(state, 0.0f);
        return static_cast<int>(action_tensor.cpu().item<int64_t>());
    }
    
    // Winning-focused reward system
    float getWinningReward(const SnakeGame& game, bool died, int steps) {
        static Position last_head(-1, -1);
        static int last_score = 0;
        
        auto head = game.getSnakeBody()[0];
        auto food = game.getFoodPosition();
        int current_score = game.getScore();
        
        float reward = 0.0f;
        
        if (died) {
            // Death penalty based on score achieved
            if (current_score >= 15) {
                reward = -5.0f;   // Light penalty for high scorers
            } else if (current_score >= 10) {
                reward = -10.0f;  // Medium penalty
            } else if (current_score >= 5) {
                reward = -15.0f;  // Heavy penalty
            } else {
                reward = -25.0f;  // Very heavy penalty for early death
            }
            
            last_head = Position(-1, -1);
            last_score = 0;
            return reward;
        }
        
        // Survival reward decreases as snake gets longer (encourage risk-taking)
        reward = 0.05f / (1.0f + current_score * 0.1f);
        
        // MASSIVE food rewards with exponential scaling
        if (current_score > last_score) {
            float base_reward = 50.0f;
            float score_multiplier = 1.0f + (current_score * 0.5f);  // Gets more valuable
            reward += base_reward * score_multiplier;
            
            // Bonus for high scores
            if (current_score >= 15) {
                reward += 100.0f;  // Huge bonus for winning territory
            } else if (current_score >= 10) {
                reward += 50.0f;   // Big bonus for high scores
            }
            
            last_score = current_score;
        }
        
        // Movement efficiency toward food
        if (last_head.x >= 0) {
            float old_dist = abs(last_head.x - food.x) + abs(last_head.y - food.y);
            float new_dist = abs(head.x - food.x) + abs(head.y - food.y);
            
            if (new_dist < old_dist) {
                reward += 1.0f;   // Good progress
            } else if (new_dist > old_dist) {
                reward -= 0.5f;   // Moving away penalty
            }
        }
        
        last_head = head;
        return reward;
    }
    
    ScoreMetrics evaluateScores(int test_episodes) {
        int high_scores = 0;
        int winning_games = 0;
        int total_score = 0;
        int max_score = 0;
        
        for (int episode = 0; episode < test_episodes; episode++) {
            SnakeGame game;
            game.reset();
            
            int steps = 0;
            while (!game.isGameOver() && steps < 2000) {
                auto state = getWinningState(game);
                auto action_tensor = network.getAction(state, 0.0f);
                int action = static_cast<int>(action_tensor.cpu().item<int64_t>());
                
                game.setDirection(static_cast<Direction>(action));
                game.update();
                steps++;
            }
            
            int score = game.getScore();
            total_score += score;
            if (score >= 10) high_scores++;
            if (score >= 15) winning_games++;
            if (score > max_score) max_score = score;
        }
        
        return {
            (float)total_score / test_episodes,
            (float)high_scores / test_episodes * 100.0f,
            (float)winning_games / test_episodes * 100.0f,
            (float)max_score
        };
    }
    
    void trainStep(const std::vector<float>& state, int action, float reward,
                   const std::vector<float>& next_state, bool done) {
        
        auto state_tensor = torch::zeros({1, 11}, torch::kFloat);
        auto next_state_tensor = torch::zeros({1, 11}, torch::kFloat);
        
        for (int i = 0; i < 11; i++) {
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
    std::cout << "=== WINNING-FOCUSED SNAKE AI TRAINER ===" << std::endl;
    std::cout << "Goal: Train an AI that can actually WIN Snake (15+ foods consistently)" << std::endl;
    std::cout << "Features: Enhanced state (11D), larger network, score-based rewards" << std::endl;
    std::cout << std::endl;
    
    try {
        WinningTrainer trainer;
        trainer.train(25000);  // Substantial training for complex behavior
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}
