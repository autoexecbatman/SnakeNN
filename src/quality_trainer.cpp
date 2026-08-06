#include "neural_network.h"
#include "snake_logic.h"
#include <iostream>

// Quality-focused trainer that optimizes for longer games and higher scores
class QualityTrainer {
public:
    QualityTrainer() : network(8, 64, 4), target_network(8, 64, 4) {
        auto source_params = network.parameters();
        auto target_params = target_network.parameters();
        torch::NoGradGuard no_grad;
        for (size_t i = 0; i < source_params.size(); i++) {
            target_params[i].copy_(source_params[i]);
        }
    }
    
    void train(int episodes = 12000) {
        std::cout << "=== QUALITY-FOCUSED TRAINER ===" << std::endl;
        std::cout << "Optimizing for: Game length + Score quality, NOT just 'any food success'" << std::endl;
        
        float epsilon = 1.0f;
        const float epsilon_decay = 0.9995f;
        const float epsilon_min = 0.03f;
        
        int total_score = 0;
        int total_steps = 0;
        int quality_games = 0;  // Games with 5+ foods OR 150+ steps
        
        for (int episode = 0; episode < episodes; episode++) {
            SnakeGame game;
            game.reset();
            
            std::vector<float> prev_state;
            int prev_action = -1;
            int steps = 0;
            
            while (!game.isGameOver() && steps < 1200) {  // Allow longer games
                auto current_state = getOptimizedState(game);
                int action = getAction(current_state, epsilon);
                
                game.setDirection(static_cast<Direction>(action));
                bool game_continues = game.update();
                
                float reward = getQualityReward(game, !game_continues, steps);
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
            total_steps += steps;
            
            // Quality metric: 5+ foods OR 150+ steps (survival skill)
            if (score >= 5 || steps >= 150) {
                quality_games++;
            }
            
            if (episode % 75 == 0) {
                updateTargetNetwork();
            }
            
            if (epsilon > epsilon_min) {
                epsilon *= epsilon_decay;
            }
            
            // Quality evaluation every 300 episodes
            if (episode % 300 == 0 && episode > 0) {
                auto quality_metrics = evaluateQuality(100);
                float avg_score = quality_metrics.avg_score;
                float avg_length = quality_metrics.avg_length;
                float quality_rate = quality_metrics.quality_rate;
                
                float training_quality = (float)quality_games / (episode + 1) * 100.0f;
                float avg_training_score = (float)total_score / (episode + 1);
                float avg_training_length = (float)total_steps / (episode + 1);
                
                std::cout << "\\nEpisode " << episode << ":" << std::endl;
                std::cout << "  Training: " << training_quality << "% quality | " 
                         << avg_training_score << " avg score | " << avg_training_length << " avg steps" << std::endl;
                std::cout << "  PURE TEST: " << quality_rate << "% quality | " 
                         << avg_score << " avg score | " << avg_length << " avg steps" << std::endl;
                std::cout << "  Epsilon: " << epsilon << std::endl;
                
                if (quality_rate >= 70.0f && avg_score >= 8.0f) {
                    std::cout << "*** QUALITY ACHIEVED! High score + good survival ***" << std::endl;
                    std::string quality_path = "snake_quality_" + std::to_string((int)quality_rate) + "pct_" + std::to_string((int)avg_score) + "score.bin";
                    network.save(quality_path);
                    std::cout << "Quality model saved: " << quality_path << std::endl;
                }
            }
        }
        
        auto final_metrics = evaluateQuality(200);
        float final_quality_rate = final_metrics.quality_rate;
        float final_avg_score = final_metrics.avg_score;
        float final_avg_length = final_metrics.avg_length;
        
        std::cout << std::endl << "=== QUALITY TRAINING RESULTS ===" << std::endl;
        std::cout << "Training episodes: " << episodes << std::endl;
        std::cout << "*** QUALITY RATE: " << final_quality_rate << "% (5+ foods OR 150+ steps) ***" << std::endl;
        std::cout << "*** AVERAGE SCORE: " << final_avg_score << " foods per game ***" << std::endl;
        std::cout << "*** AVERAGE LENGTH: " << final_avg_length << " steps per game ***" << std::endl;
        
        // Quality classification
        if (final_quality_rate >= 80.0f && final_avg_score >= 10.0f) {
            std::cout << "\\n🏆 ELITE QUALITY! 80%+ quality games with 10+ avg score!" << std::endl;
        } else if (final_quality_rate >= 70.0f && final_avg_score >= 8.0f) {
            std::cout << "\\n🎉 HIGH QUALITY! 70%+ quality games with 8+ avg score!" << std::endl;
        } else if (final_quality_rate >= 60.0f && final_avg_score >= 6.0f) {
            std::cout << "\\n✨ GOOD QUALITY! 60%+ quality games with 6+ avg score!" << std::endl;
        } else {
            std::cout << "\\n📊 Quality baseline: " << final_quality_rate << "% quality, " << final_avg_score << " avg score" << std::endl;
        }
        
        std::string final_path = "snake_quality_final_" + std::to_string((int)final_quality_rate) + "pct_" + std::to_string((int)final_avg_score) + "score.bin";
        network.save(final_path);
        std::cout << "\\n*** QUALITY MODEL SAVED: " << final_path << " ***" << std::endl;
    }
    
private:
    SnakeNeuralNetwork network;
    SnakeNeuralNetwork target_network;
    const float learning_rate = 0.007f;
    const float gamma = 0.98f;
    
    struct QualityMetrics {
        float quality_rate;  // % of games with 5+ foods OR 150+ steps
        float avg_score;     // Average foods per game
        float avg_length;    // Average steps per game
    };
    
    std::vector<float> getOptimizedState(const SnakeGame& game) {
        auto head = game.getSnakeBody()[0];
        auto food = game.getFoodPosition();
        
        std::vector<float> state(8);
        
        // Food direction
        state[0] = (food.x > head.x) ? 1.0f : 0.0f;
        state[1] = (food.x < head.x) ? 1.0f : 0.0f;
        state[2] = (food.y > head.y) ? 1.0f : 0.0f;
        state[3] = (food.y < head.y) ? 1.0f : 0.0f;
        
        // Immediate danger
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
    
    // QUALITY-FOCUSED REWARD SYSTEM
    float getQualityReward(const SnakeGame& game, bool died, int steps) {
        static Position last_head(-1, -1);
        static int last_score = 0;
        static int consecutive_no_progress = 0;
        
        auto head = game.getSnakeBody()[0];
        auto food = game.getFoodPosition();
        int current_score = game.getScore();
        
        float reward = 0.0f;
        
        // Death penalty scaled by game length - early death = worse penalty
        if (died) {
            if (steps < 50) {
                reward = -25.0f;  // Very early death
            } else if (steps < 100) {
                reward = -20.0f;  // Early death
            } else if (steps < 200) {
                reward = -15.0f;  // Mid death
            } else {
                reward = -10.0f;  // Late death (acceptable)
            }
            
            // Bonus for dying with good score
            if (current_score >= 5) {
                reward += 5.0f;  // Died but achieved something
            }
            
            last_head = Position(-1, -1);
            last_score = 0;
            consecutive_no_progress = 0;
            return reward;
        }
        
        // Survival reward increases with game length
        if (steps < 100) {
            reward += 0.1f;   // Basic survival
        } else if (steps < 200) {
            reward += 0.2f;   // Good survival
        } else {
            reward += 0.3f;   // Excellent survival
        }
        
        // Food reward - BIG bonus, increases with game difficulty
        if (current_score > last_score) {
            float base_food_reward = 20.0f;
            float difficulty_multiplier = 1.0f + (current_score * 0.1f);  // Gets harder with longer snake
            reward += base_food_reward * difficulty_multiplier;
            
            last_score = current_score;
            consecutive_no_progress = 0;
        }
        
        // Movement reward/penalty
        if (last_head.x >= 0) {
            float old_dist = abs(last_head.x - food.x) + abs(last_head.y - food.y);
            float new_dist = abs(head.x - food.x) + abs(head.y - food.y);
            
            if (new_dist < old_dist) {
                reward += 0.5f;  // Good progress toward food
                consecutive_no_progress = 0;
            } else if (new_dist > old_dist) {
                reward -= 0.2f;  // Moving away
                consecutive_no_progress++;
            } else {
                consecutive_no_progress++;
            }
            
            // Penalty for getting stuck in loops
            if (consecutive_no_progress > 20) {
                reward -= 1.0f;  // Stuck penalty
            }
        }
        
        last_head = head;
        return reward;
    }
    
    QualityMetrics evaluateQuality(int test_episodes) {
        int quality_games = 0;
        int total_score = 0;
        int total_steps = 0;
        
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
            
            int score = game.getScore();
            total_score += score;
            total_steps += steps;
            
            // Quality game: 5+ foods OR 150+ steps
            if (score >= 5 || steps >= 150) {
                quality_games++;
            }
        }
        
        return {
            (float)quality_games / test_episodes * 100.0f,
            (float)total_score / test_episodes,
            (float)total_steps / test_episodes
        };
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
    std::cout << "=== QUALITY-FOCUSED SNAKE AI TRAINER ===" << std::endl;
    std::cout << "Optimizing for game length and score quality, not just 'any food' success" << std::endl;
    std::cout << "Quality metric: 5+ foods OR 150+ steps survival" << std::endl;
    std::cout << std::endl;
    
    try {
        QualityTrainer trainer;
        trainer.train(12000);  // Moderate episode count for quality focus
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}
