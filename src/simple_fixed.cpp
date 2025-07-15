#include "neural_network.h"
#include "snake_logic.h"
#include <iostream>

class FixedDQNTrainer {
public:
    FixedDQNTrainer() : network(8, 32, 4), target_network(8, 32, 4) {
        auto source_params = network.parameters();
        auto target_params = target_network.parameters();
        torch::NoGradGuard no_grad;
        for (size_t i = 0; i < source_params.size(); i++) {
            target_params[i].copy_(source_params[i]);
        }
    }
    
    void train(int episodes = 1000, bool vsual = false) {
        std::cout << "Fixed Trainer - No Experience Replay" << std::endl;
        
        float epsilon = 1.0f;
        const float epsilon_decay = 0.999f;
        const float epsilon_min = 0.05f;
        
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
                
                game.setDirection(static_cast<Direction>(action));
                bool game_continues = game.update();
                
                float reward = getImprovedReward(game, !game_continues);
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
            
            if (episode % 25 == 0) {
                updateTargetNetwork();
            }
            
            if (epsilon > epsilon_min) {
                epsilon *= epsilon_decay;
            }
            
            if (episode % 50 == 0) {
                float avg_score = (float)total_score / (episode + 1);
                float success_rate = (float)successful_episodes / (episode + 1) * 100.0f;
                
                std::cout << "Episode " << episode << ", Score: " << score;
                std::cout << ", Avg: " << avg_score << ", Success: " << success_rate << "%";
                std::cout << ", Reward: " << total_reward << ", Steps: " << steps;
                std::cout << ", Epsilon: " << epsilon << std::endl;
            }
            
            if (episode >= 100 && episode % 100 == 0) {
                float recent_avg = (float)total_score / (episode + 1);
                if (recent_avg > 1.5f) {
                    std::cout << "Early success detected! Average > 1.5" << std::endl;
                }
            }
        }
        
        float final_avg = (float)total_score / episodes;
        float final_success_rate = (float)successful_episodes / episodes * 100.0f;
        
        std::cout << std::endl << "Final Results:" << std::endl;
        std::cout << "Total score: " << total_score << std::endl;
        std::cout << "Average score: " << final_avg << std::endl;
        std::cout << "Success rate: " << final_success_rate << "%" << std::endl;
        
        if (final_avg > 1.0f) {
            std::cout << "SUCCESS! Snake learned to find food!" << std::endl;
        } else if (final_avg > 0.3f) {
            std::cout << "Partial success - some learning occurred" << std::endl;
        } else {
            std::cout << "Failure - little to no learning" << std::endl;
        }
    }
    
private:
    SnakeNeuralNetwork network;
    SnakeNeuralNetwork target_network;
    const float learning_rate = 0.01f;
    const float gamma = 0.95f;
    
    std::vector<float> getSimplifiedState(const SnakeGame& game) {
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
    
    float getImprovedReward(const SnakeGame& game, bool died) {
        if (died) return -10.0f;
        
        static Position last_head(-1, -1);
        static Position last_food(-1, -1);
        
        auto head = game.getSnakeBody()[0];
        auto food = game.getFoodPosition();
        
        float reward = 0.1f;
        
        if (last_food.x >= 0 && head.x == last_food.x && head.y == last_food.y) {
            reward += 10.0f;
        }
        
        if (last_head.x >= 0) {
            float old_dist = abs(last_head.x - food.x) + abs(last_head.y - food.y);
            float new_dist = abs(head.x - food.x) + abs(head.y - food.y);
            
            if (new_dist < old_dist) {
                reward += 0.5f;
            } else if (new_dist > old_dist) {
                reward -= 0.1f;
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
    std::cout << "Fixed Snake AI Trainer" << std::endl;
    
    try {
        FixedDQNTrainer trainer;
        trainer.train(1000, false);
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}
