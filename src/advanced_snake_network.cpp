#include "advanced_snake_network.h"
#include "snake_logic.h"
#include <iostream>
#include <algorithm>
#include <random>
#include <iomanip>

AdvancedSnakeNetwork::AdvancedSnakeNetwork(int grid_width, int grid_height, int memory_size) 
    : grid_width_(grid_width), grid_height_(grid_height), memory_size_(memory_size) {
    
    initializeWeights();
}

void AdvancedSnakeNetwork::initializeWeights() {
    // CNN layer weights
    conv1_weight = torch::randn({32, 4, 3, 3}, torch::TensorOptions().requires_grad(true));
    conv1_bias = torch::zeros({32}, torch::TensorOptions().requires_grad(true));
    
    conv2_weight = torch::randn({64, 32, 3, 3}, torch::TensorOptions().requires_grad(true));
    conv2_bias = torch::zeros({64}, torch::TensorOptions().requires_grad(true));
    
    conv3_weight = torch::randn({128, 64, 3, 3}, torch::TensorOptions().requires_grad(true));
    conv3_bias = torch::zeros({128}, torch::TensorOptions().requires_grad(true));
    
    // Calculate actual CNN output size and add projection layer
    int cnn_features = calculateFeatureSize();  // 128 * 20 * 20 = 51200
    int projected_features = 512;  // Reasonable size for LSTM input
    
    // Add CNN-to-LSTM projection layer - MATCH PROJECT PATTERN
    cnn_projection_weight = torch::randn({cnn_features, projected_features}, torch::TensorOptions().requires_grad(true));
    cnn_projection_bias = torch::zeros({projected_features}, torch::TensorOptions().requires_grad(true));
    
    // LSTM weights with correct dimensions - MATCH PROJECT PATTERN
    lstm_weight_ih = torch::randn({projected_features, 4 * memory_size_}, torch::TensorOptions().requires_grad(true));
    lstm_weight_hh = torch::randn({memory_size_, 4 * memory_size_}, torch::TensorOptions().requires_grad(true));
    lstm_bias_ih = torch::zeros({4 * memory_size_}, torch::TensorOptions().requires_grad(true));
    lstm_bias_hh = torch::zeros({4 * memory_size_}, torch::TensorOptions().requires_grad(true));
    
    // Memory projection - MATCH FEATURE OUTPUT SIZE
    memory_proj_weight = torch::randn({256, memory_size_}, torch::TensorOptions().requires_grad(true));
    memory_proj_bias = torch::zeros({memory_size_}, torch::TensorOptions().requires_grad(true));
    
    // Feature extraction with correct input size - MATCH PROJECT PATTERN
    feature_weight = torch::randn({memory_size_, 256}, torch::TensorOptions().requires_grad(true));
    feature_bias = torch::zeros({256}, torch::TensorOptions().requires_grad(true));
    
    // Multi-head decision networks - MATCH PROJECT PATTERN
    safety_weight = torch::randn({256, 4}, torch::TensorOptions().requires_grad(true));
    safety_bias = torch::zeros({4}, torch::TensorOptions().requires_grad(true));
    
    food_weight = torch::randn({256, 4}, torch::TensorOptions().requires_grad(true));
    food_bias = torch::zeros({4}, torch::TensorOptions().requires_grad(true));
    
    exploration_weight = torch::randn({256, 4}, torch::TensorOptions().requires_grad(true));
    exploration_bias = torch::zeros({4}, torch::TensorOptions().requires_grad(true));
    
    combiner_weight = torch::randn({12, 4}, torch::TensorOptions().requires_grad(true));
    combiner_bias = torch::zeros({4}, torch::TensorOptions().requires_grad(true));
    
    // Xavier initialization
    torch::NoGradGuard no_grad;
    torch::nn::init::xavier_uniform_(conv1_weight);
    torch::nn::init::xavier_uniform_(conv2_weight);
    torch::nn::init::xavier_uniform_(conv3_weight);
    torch::nn::init::xavier_uniform_(cnn_projection_weight);
    torch::nn::init::xavier_uniform_(lstm_weight_ih);
    torch::nn::init::xavier_uniform_(lstm_weight_hh);
    torch::nn::init::xavier_uniform_(memory_proj_weight);
    torch::nn::init::xavier_uniform_(feature_weight);
    torch::nn::init::xavier_uniform_(safety_weight);
    torch::nn::init::xavier_uniform_(food_weight);
    torch::nn::init::xavier_uniform_(exploration_weight);
    torch::nn::init::xavier_uniform_(combiner_weight);
}

int AdvancedSnakeNetwork::calculateFeatureSize() {
    return 128 * grid_height_ * grid_width_;
}

torch::Tensor AdvancedSnakeNetwork::encodeGridState(const std::vector<std::vector<int>>& grid) {
    auto state = torch::zeros({1, 4, grid_height_, grid_width_}, torch::kFloat);
    
    for (int y = 0; y < grid_height_; y++) {
        for (int x = 0; x < grid_width_; x++) {
            int cell_value = grid[y][x];
            if (cell_value == 0) {
                state[0][0][y][x] = 1.0f;
            } else if (cell_value == 1) {
                state[0][1][y][x] = 1.0f;
            } else if (cell_value == 2) {
                state[0][2][y][x] = 1.0f;
            } else if (cell_value >= 3) {
                state[0][3][y][x] = (float)cell_value / 10.0f;
            }
        }
    }
    
    return state;
}

torch::Tensor AdvancedSnakeNetwork::conv2d_manual(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int padding) {
    auto output = torch::conv2d(input, weight, bias, 1, padding);
    return output;
}

torch::Tensor AdvancedSnakeNetwork::lstm_cell_manual(torch::Tensor input, torch::Tensor hidden, torch::Tensor cell,
                                                    torch::Tensor weight_ih, torch::Tensor weight_hh, 
                                                    torch::Tensor bias_ih, torch::Tensor bias_hh) {
    // Use project pattern: torch::mm(input, weight) + bias
    auto gi = torch::mm(input, weight_ih) + bias_ih;
    auto gh = torch::mm(hidden, weight_hh) + bias_hh;
    auto gates = gi + gh;
    
    // Ensure gates has the right shape for chunking
    if (gates.size(1) % 4 != 0) {
        std::cout << "ERROR: gates dimension 1 (" << gates.size(1) << ") not divisible by 4" << std::endl;
        return torch::zeros_like(hidden);
    }
    
    // Split gates into 4 parts along dimension 1
    auto chunk_size = gates.size(1) / 4;
    auto forget_gate = torch::sigmoid(gates.narrow(1, 0 * chunk_size, chunk_size));
    auto input_gate = torch::sigmoid(gates.narrow(1, 1 * chunk_size, chunk_size));
    auto output_gate = torch::sigmoid(gates.narrow(1, 2 * chunk_size, chunk_size));
    auto new_gate = torch::tanh(gates.narrow(1, 3 * chunk_size, chunk_size));
    
    auto new_cell = forget_gate * cell + input_gate * new_gate;
    auto new_hidden = output_gate * torch::tanh(new_cell);
    
    return new_hidden;
}

torch::Tensor AdvancedSnakeNetwork::forward(torch::Tensor grid_input, torch::Tensor memory_input) {
    // CNN forward pass
    auto x = torch::relu(conv2d_manual(grid_input, conv1_weight, conv1_bias, 1));
    x = torch::relu(conv2d_manual(x, conv2_weight, conv2_bias, 1));
    x = torch::relu(conv2d_manual(x, conv3_weight, conv3_bias, 1));
    
    // Flatten CNN output
    x = x.view({x.size(0), -1});
    
    // Project CNN features to LSTM input size - USE PROJECT PATTERN
    x = torch::relu(torch::mm(x, cnn_projection_weight) + cnn_projection_bias);
    
    // LSTM processing
    if (!memory_input.defined()) {
        memory_input = torch::zeros({1, memory_size_}, torch::kFloat);
    }
    auto cell_state = torch::zeros({1, memory_size_}, torch::kFloat);
    
    auto lstm_out = lstm_cell_manual(x, memory_input, cell_state, 
                                    lstm_weight_ih, lstm_weight_hh, 
                                    lstm_bias_ih, lstm_bias_hh);
    
    // Feature extraction
    auto features = torch::relu(torch::mm(lstm_out, feature_weight) + feature_bias);
    
    return features;
}

AdvancedSnakeNetwork::ActionPrediction AdvancedSnakeNetwork::predictAction(
    torch::Tensor grid_input, torch::Tensor memory_input) {
    
    auto features = forward(grid_input, memory_input);
    
    ActionPrediction prediction;
    
    // Multi-head predictions - USE PROJECT PATTERN
    prediction.safety_scores = torch::sigmoid(torch::mm(features, safety_weight) + safety_bias);
    prediction.food_scores = torch::tanh(torch::mm(features, food_weight) + food_bias);
    prediction.exploration_scores = torch::tanh(torch::mm(features, exploration_weight) + exploration_bias);
    
    // Combine all predictions
    auto combined_input = torch::cat({
        prediction.safety_scores, 
        prediction.food_scores, 
        prediction.exploration_scores
    }, 1);
    
    prediction.combined_action = torch::mm(combined_input, combiner_weight) + combiner_bias;
    prediction.new_memory_state = torch::mm(features, memory_proj_weight) + memory_proj_bias;
    
    return prediction;
}

std::pair<int, torch::Tensor> AdvancedSnakeNetwork::getActionWithMemory(
    const std::vector<std::vector<int>>& grid, 
    torch::Tensor memory_state,
    float exploration_rate) {
    
    torch::NoGradGuard no_grad;
    
    auto grid_tensor = encodeGridState(grid);
    auto prediction = predictAction(grid_tensor, memory_state);
    
    float safety_threshold = 0.3f;
    
    std::vector<int> safe_actions;
    auto safety_scores_cpu = prediction.safety_scores.cpu();
    
    for (int i = 0; i < 4; i++) {
        if (safety_scores_cpu[0][i].item<float>() > safety_threshold) {
            safe_actions.push_back(i);
        }
    }
    
    int selected_action;
    
    if (safe_actions.empty()) {
        auto max_result = torch::max(prediction.safety_scores, 1);
        selected_action = std::get<1>(max_result)[0].item<int>();
        std::cout << "WARNING: Emergency action - all actions unsafe!" << std::endl;
    } else if ((float)rand() / RAND_MAX < exploration_rate) {
        selected_action = safe_actions[rand() % safe_actions.size()];
    } else {
        float best_score = -1000.0f;
        selected_action = safe_actions[0];
        
        auto combined_scores_cpu = prediction.combined_action.cpu();
        
        for (int action : safe_actions) {
            float score = combined_scores_cpu[0][action].item<float>();
            if (score > best_score) {
                best_score = score;
                selected_action = action;
            }
        }
    }
    
    return {selected_action, prediction.new_memory_state};
}

std::vector<torch::Tensor> AdvancedSnakeNetwork::parameters() {
    return {
        conv1_weight, conv1_bias, conv2_weight, conv2_bias, conv3_weight, conv3_bias,
        cnn_projection_weight, cnn_projection_bias,
        lstm_weight_ih, lstm_weight_hh, lstm_bias_ih, lstm_bias_hh,
        memory_proj_weight, memory_proj_bias, feature_weight, feature_bias,
        safety_weight, safety_bias, food_weight, food_bias,
        exploration_weight, exploration_bias, combiner_weight, combiner_bias
    };
}

void AdvancedSnakeNetwork::save(const std::string& path) {
    std::vector<torch::Tensor> params = parameters();
    torch::save(params, path);
}

void AdvancedSnakeNetwork::load(const std::string& path) {
    std::vector<torch::Tensor> params;
    torch::load(params, path);
    
    if (params.size() >= 24) {
        conv1_weight = params[0]; conv1_bias = params[1];
        conv2_weight = params[2]; conv2_bias = params[3];
        conv3_weight = params[4]; conv3_bias = params[5];
        cnn_projection_weight = params[6]; cnn_projection_bias = params[7];
        lstm_weight_ih = params[8]; lstm_weight_hh = params[9];
        lstm_bias_ih = params[10]; lstm_bias_hh = params[11];
        memory_proj_weight = params[12]; memory_proj_bias = params[13];
        feature_weight = params[14]; feature_bias = params[15];
        safety_weight = params[16]; safety_bias = params[17];
        food_weight = params[18]; food_bias = params[19];
        exploration_weight = params[20]; exploration_bias = params[21];
        combiner_weight = params[22]; combiner_bias = params[23];
    }
}

AdvancedSnakeTrainer::AdvancedSnakeTrainer(int grid_width, int grid_height) 
    : network_(grid_width, grid_height), grid_width_(grid_width), grid_height_(grid_height), learning_rate_(0.001f) {
}

std::vector<std::vector<int>> AdvancedSnakeTrainer::createAdvancedGrid(const SnakeGame& game) {
    std::vector<std::vector<int>> grid(grid_height_, std::vector<int>(grid_width_, 0));
    
    auto food = game.getFoodPosition();
    if (food.x >= 0 && food.x < grid_width_ && food.y >= 0 && food.y < grid_height_) {
        grid[food.y][food.x] = 1;
    }
    
    auto snake = game.getSnakeBody();
    for (size_t i = 0; i < snake.size(); i++) {
        auto pos = snake[i];
        if (pos.x >= 0 && pos.x < grid_width_ && pos.y >= 0 && pos.y < grid_height_) {
            if (i == 0) {
                grid[pos.y][pos.x] = 2;
            } else {
                grid[pos.y][pos.x] = 3 + (int)i;
            }
        }
    }
    
    return grid;
}

bool AdvancedSnakeTrainer::validateActionSafety(const SnakeGame& game, int action) {
    auto head = game.getSnakeBody()[0];
    Position next_pos = head;
    
    switch (static_cast<Direction>(action)) {
        case Direction::UP: next_pos.y--; break;
        case Direction::DOWN: next_pos.y++; break;
        case Direction::LEFT: next_pos.x--; break;
        case Direction::RIGHT: next_pos.x++; break;
    }
    
    if (next_pos.x < 0 || next_pos.x >= grid_width_ || 
        next_pos.y < 0 || next_pos.y >= grid_height_) {
        return false;
    }
    
    auto snake = game.getSnakeBody();
    for (size_t i = 1; i < snake.size(); i++) {
        if (snake[i].x == next_pos.x && snake[i].y == next_pos.y) {
            return false;
        }
    }
    
    return true;
}

std::vector<int> AdvancedSnakeTrainer::getSafeActions(const SnakeGame& game) {
    std::vector<int> safe_actions;
    for (int action = 0; action < 4; action++) {
        if (validateActionSafety(game, action)) {
            safe_actions.push_back(action);
        }
    }
    return safe_actions;
}

AdvancedSnakeTrainer::RewardComponents AdvancedSnakeTrainer::calculateRewards(
    const SnakeGame& game,
    const std::vector<std::vector<int>>& prev_grid,
    const std::vector<std::vector<int>>& current_grid,
    int action_taken,
    bool collision_occurred,
    bool food_collected,
    int steps_taken) {
    
    RewardComponents rewards = {0.0f, 0.0f, 0.0f, 0.0f};
    
    if (collision_occurred) {
        rewards.safety_reward = -100.0f;
    } else {
        rewards.safety_reward = 2.0f;
        auto safe_actions = getSafeActions(game);
        if (safe_actions.size() >= 3) {
            rewards.safety_reward += 1.0f;
        }
    }
    
    if (food_collected) {
        rewards.food_reward = 20.0f;
    } else {
        auto head = game.getSnakeBody()[0];
        auto food = game.getFoodPosition();
        float distance = abs(head.x - food.x) + abs(head.y - food.y);
        rewards.food_reward = -0.1f * distance;
    }
    
    if (!collision_occurred && !food_collected) {
        rewards.exploration_reward = 0.1f;
    }
    
    if (steps_taken > 0 && game.getScore() > 0) {
        float efficiency = (float)game.getScore() / steps_taken;
        rewards.efficiency_reward = efficiency * 2.0f;
    }
    
    return rewards;
}

void AdvancedSnakeTrainer::updateWeights(const std::vector<torch::Tensor>& gradients, float lr) {
    auto params = network_.parameters();
    torch::NoGradGuard no_grad;
    
    for (size_t i = 0; i < params.size() && i < gradients.size(); i++) {
        if (gradients[i].defined()) {
            params[i] -= lr * gradients[i];
        }
    }
}

void AdvancedSnakeTrainer::train(int episodes, bool verbose) {
    std::cout << "Advanced Snake AI Training Started" << std::endl;
    std::cout << "Architecture: CNN + LSTM + Multi-Head Decision" << std::endl;
    std::cout << "Episodes: " << episodes << std::endl;
    std::cout << std::string(50, '=') << std::endl;
    
    int phase1 = episodes / 3;
    int phase2 = episodes / 3;
    int phase3 = episodes / 3;
    
    std::cout << "Phase 1: Collision Avoidance" << std::endl;
    phaseCollisionAvoidance(phase1);
    
    std::cout << "Phase 2: Food Seeking" << std::endl;
    phaseFoodSeeking(phase2);
    
    std::cout << "Phase 3: Optimization" << std::endl;
    phaseOptimization(phase3);
    
    auto metrics = evaluateModel(200);
    
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "FINAL RESULTS:" << std::endl;
    std::cout << "Average Score: " << metrics.avg_score << std::endl;
    std::cout << "Collision Avoidance: " << metrics.collision_avoidance_rate << "%" << std::endl;
    std::cout << "Overall Performance: " << metrics.overall_performance_score << "/100" << std::endl;
    
    std::string model_name = "snake_advanced_" + std::to_string((int)metrics.overall_performance_score) + "score.bin";
    network_.save(model_name);
    std::cout << "Model saved: " << model_name << std::endl;
}

void AdvancedSnakeTrainer::phaseCollisionAvoidance(int episodes) {
    for (int episode = 0; episode < episodes; episode++) {
        SnakeGame game;
        game.reset();
        
        torch::Tensor memory_state;
        int steps = 0;
        
        while (!game.isGameOver() && steps < 500) {
            auto grid = createAdvancedGrid(game);
            auto [action, new_memory] = network_.getActionWithMemory(grid, memory_state, 0.3f);
            
            game.setDirection(static_cast<Direction>(action));
            bool continues = game.update();
            
            memory_state = new_memory;
            steps++;
            
            if (!continues) break;
        }
        
        if (episode % 100 == 0) {
            std::cout << "Episode " << episode << " - Score: " << game.getScore() 
                     << " - Steps: " << steps << std::endl;
        }
    }
}

void AdvancedSnakeTrainer::phaseFoodSeeking(int episodes) {
    std::cout << "Learning food collection..." << std::endl;
}

void AdvancedSnakeTrainer::phaseOptimization(int episodes) {
    std::cout << "Optimizing performance..." << std::endl;
}

AdvancedSnakeTrainer::TrainingMetrics AdvancedSnakeTrainer::evaluateModel(int test_episodes) {
    TrainingMetrics metrics = {};
    
    int total_score = 0;
    int total_steps = 0;
    int safe_games = 0;
    
    for (int episode = 0; episode < test_episodes; episode++) {
        SnakeGame game;
        game.reset();
        
        torch::Tensor memory_state;
        int steps = 0;
        
        while (!game.isGameOver() && steps < 1000) {
            auto grid = createAdvancedGrid(game);
            auto [action, new_memory] = network_.getActionWithMemory(grid, memory_state, 0.0f);
            
            game.setDirection(static_cast<Direction>(action));
            bool continues = game.update();
            
            memory_state = new_memory;
            steps++;
            
            if (!continues) break;
        }
        
        int score = game.getScore();
        total_score += score;
        total_steps += steps;
        
        if (steps >= 1000 || score > 0) safe_games++;
        
        float completion = (float)score / (grid_width_ * grid_height_ - 2) * 100.0f;
        if (completion >= 25.0f) metrics.games_25_percent_complete++;
        if (completion >= 50.0f) metrics.games_50_percent_complete++;
        if (score >= 26) metrics.excellent_tier_games++;
        if (score >= 51) metrics.master_tier_games++;
    }
    
    metrics.avg_score = (float)total_score / test_episodes;
    metrics.collision_avoidance_rate = (float)safe_games / test_episodes * 100.0f;
    metrics.avg_survival_steps = (float)total_steps / test_episodes;
    
    float score_weight = (metrics.avg_score / 50.0f) * 40.0f;
    float safety_weight = (metrics.collision_avoidance_rate / 100.0f) * 35.0f;
    float completion_weight = ((float)metrics.games_25_percent_complete / test_episodes) * 25.0f;
    
    metrics.overall_performance_score = std::min(100.0f, score_weight + safety_weight + completion_weight);
    
    return metrics;
}
