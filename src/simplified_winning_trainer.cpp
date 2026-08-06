#include "neural_network.h"
#include "snake_logic.h"
#include <iostream>
#include <chrono>

// Simplified winning trainer - same 8 features but score-focused rewards
class SimplifiedWinningTrainer {
public:
    SimplifiedWinningTrainer() : network(8, 64, 4), target_network(8, 64, 4) {
        auto source_params = network.parameters();
        auto target_params = target_network.parameters();
        torch::NoGradGuard no_grad;
        for (size_t i = 0; i < source_params.size(); i++) {
            target_params[i].copy_(source_params[i]);
        }
    }
    
    void train(int episodes = 20000) {
        std::cout << "=== SIMPLIFIED WINNING TRAINER ===" << std::endl;
        std::cout << "Same 8 features as successful model but optimized for high scores" << std::endl;
        
        float epsilon = 1.0f;
        const float epsilon_decay = 0.9999f;
        const float epsilon_min = 0.05f;
        
        int total_score = 0;
        int high_score_games = 0;
        int winning_games = 0;
        auto start_time = std::chrono::high_resolution_clock::now();
        
        float best_avg_score = 0.0f;
        
        for (int episode = 0; episode < episodes; episode++) {
            SnakeGame game;
            game.reset();
            
            std::vector<float> prev_state;
            int prev_action = -1;
            int steps = 0;
            
            while (!game.isGameOver() && steps < 1500) {
                auto current_state = getOptimizedState(game);
                int action = getAction(current_state, epsilon);
                
                game.setDirection(static_cast<Direction>(action));
                bool game_continues = game.update();
                
                float reward = getScoreReward(game, !game_continues, steps);
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
            
            if (episode % 100 == 0 && episode > 0) {
                updateTargetNetwork();
            }
            
            if (epsilon > epsilon_min) {
                epsilon *= epsilon_decay;
            }
            
            if (episode % 1000 == 0 && episode > 0) {
                auto score_metrics = evaluateScores(100);
                float avg_score = score_metrics.avg_score;
                float high_rate = score_metrics.high_rate;
                float win_rate = score_metrics.win_rate;
                
                if (avg_score > best_avg_score) {
                    best_avg_score = avg_score;
                    std::string best_path = "snake_simple_winning_" + std::to_string((int)avg_score) + "avg.bin";
                    network.save(best_path);
                    std::cout << "*** NEW BEST: " << avg_score << " avg - Saved: " << best_path << " ***" << std::endl;
                }
                
                float training_avg = (float)total_score / (episode + 1);
                
                std::cout << "Episode " << episode << " - Training avg: " << training_avg 
                         << " | Test avg: " << avg_score << " | High scores: " << high_rate 
                         << "% | Wins: " << win_rate << "% | Epsilon: " << epsilon << std::endl;
            }
        }
        
        auto final_metrics = evaluateScores(200);
        
        std::cout << std::endl << "=== SIMPLIFIED WINNING RESULTS ===" << std::endl;
        std::cout << "*** AVERAGE SCORE: " << final_metrics.avg_score << " foods ***" << std::endl;
        std::cout << "*** HIGH SCORES (10+): " << final_metrics.high_rate << "% ***" << std::endl;
        std::cout << "*** WINNING GAMES (15+): " << final_metrics.win_rate << "% ***" << std::endl;
        std::cout << "*** BEST ACHIEVED: " << best_avg_score << " foods ***" << std::endl;
        
        std::string final_path = "snake_simple_winning_final_" + std::to_string((int)final_metrics.avg_score) + "avg.bin";
        network.save(final_path);
        std::cout << "\\n*** FINAL MODEL: " << final_path << " ***" << std::endl;
    }
    
private:
    SnakeNeuralNetwork network;
    SnakeNeuralNetwork target_network;
    const float learning_rate = 0.003f;
    const float gamma = 0.99f;
    
    struct ScoreMetrics {
        float avg_score;
        float high_rate;
        float win_rate;
    };
    
    // Same proven 8-feature state
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
    
    // Simplified score-focused reward
    float getScoreReward(const SnakeGame& game, bool died, int steps) {
        static Position last_head(-1, -1);
        static int last_score = 0;
        
        auto head = game.getSnakeBody()[0];
        auto food = game.getFoodPosition();
        int current_score = game.getScore();
        
        float reward = 0.1f; // Base survival
        
        if (died) {
            // Gentler death penalty based on achievement
            if (current_score >= 10) {
                reward = -5.0f;
            } else if (current_score >= 5) {
                reward = -10.0f;
            } else {
                reward = -15.0f;
            }
            
            last_head = Position(-1, -1);
            last_score = 0;
            return reward;
        }
        
        // BIG food rewards with bonus scaling
        if (current_score > last_score) {
            reward += 20.0f + (current_score * 2.0f); // Gets more valuable
            last_score = current_score;
        }
        
        // Simple movement reward
        if (last_head.x >= 0) {
            float old_dist = abs(last_head.x - food.x) + abs(last_head.y - food.y);
            float new_dist = abs(head.x - food.x) + abs(head.y - food.y);
            
            if (new_dist < old_dist) {
                reward += 0.5f;
            } else if (new_dist > old_dist) {
                reward -= 0.2f;
            }
        }
        
        last_head = head;
        return reward;
    }
    
    ScoreMetrics evaluateScores(int test_episodes) {
        int high_scores = 0;
        int winning_games = 0;
        int total_score = 0;
        
        for (int episode = 0; episode < test_episodes; episode++) {
            SnakeGame game;
            game.reset();
            
            int steps = 0;
            while (!game.isGameOver() && steps < 1500) {
                auto state = getOptimizedState(game);
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
        }
        
        return {
            (float)total_score / test_episodes,
            (float)high_scores / test_episodes * 100.0f,
            (float)winning_games / test_episodes * 100.0f
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
    std::cout << "=== SIMPLIFIED WINNING TRAINER ===" << std::endl;
    std::cout << "Using proven 8-feature state but with score-focused rewards" << std::endl;
    std::cout << std::endl;
    
    try {
        SimplifiedWinningTrainer trainer;
        trainer.train(20000);
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}
