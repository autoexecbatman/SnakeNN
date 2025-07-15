#include "neural_network.h"
#include <iostream>
#include <random>
#include <numeric>
#include <algorithm>
#include <fstream>
#include "snake_logic.h"
#include <raylib.h>
#include <chrono>

// Global debug logging
void logDebug(const std::string& message) {
    std::ofstream logFile("snake_debug.log", std::ios::app);
    logFile << message << std::endl;
    std::cout << message << std::endl;
}

SnakeNeuralNetwork::SnakeNeuralNetwork(int input_size, int hidden_size, int output_size) 
    : rng(std::random_device{}()), current_device(torch::kCPU) {
    
    logDebug("SnakeNeuralNetwork constructor started");
    
    // Manual tensor creation (like firstNN project) - 3 layer network: input -> hidden -> hidden -> output
    w1 = torch::randn({input_size, hidden_size}, torch::TensorOptions().requires_grad(true));
    b1 = torch::zeros({hidden_size}, torch::TensorOptions().requires_grad(true));
    w2 = torch::randn({hidden_size, hidden_size}, torch::TensorOptions().requires_grad(true));
    b2 = torch::zeros({hidden_size}, torch::TensorOptions().requires_grad(true));
    w3 = torch::randn({hidden_size, output_size}, torch::TensorOptions().requires_grad(true));
    b3 = torch::zeros({output_size}, torch::TensorOptions().requires_grad(true));
    
    // Xavier initialization (like firstNN)
    {
        torch::NoGradGuard no_grad;
        torch::nn::init::xavier_uniform_(w1);
        torch::nn::init::xavier_uniform_(w2);
        torch::nn::init::xavier_uniform_(w3);
    }
    
    logDebug("SnakeNeuralNetwork initialized successfully");
}

torch::Tensor SnakeNeuralNetwork::forward(torch::Tensor x) {
    try {
        // Manual forward pass (like firstNN project)
        auto z1 = torch::mm(x, w1) + b1;  // Linear layer 1
        auto a1 = torch::relu(z1);        // ReLU activation
        auto z2 = torch::mm(a1, w2) + b2; // Linear layer 2
        auto a2 = torch::relu(z2);        // ReLU activation
        auto z3 = torch::mm(a2, w3) + b3; // Linear layer 3 (output)
        return z3;
    } catch (const std::exception& e) {
        logDebug("ERROR in forward(): " + std::string(e.what()));
        throw;
    }
}

// Original getAction method for tensor input
torch::Tensor SnakeNeuralNetwork::getAction(const torch::Tensor& state, float epsilon) {
    try {
        logDebug("getAction called with tensor input, epsilon: " + std::to_string(epsilon));
        
        if (std::uniform_real_distribution<float>(0.0f, 1.0f)(rng) < epsilon) {
            auto action = torch::randint(0, 4, {1}, torch::kInt64);  // FIXED: Use torch::kInt64 for gather compatibility
            // FIXED: Use CPU conversion instead of accessor
            auto accessor_val = action.cpu().item<int64_t>();  // FIXED: Use int64_t
            logDebug("Random action selected: " + std::to_string(accessor_val));
            return action;
        }
        
        torch::NoGradGuard no_grad;
        
        // FIXED: Use proper tensor dimension access
        auto state_tensor = state.unsqueeze(0);  // Add batch dimension if needed
        if (state.dim() == 2) {
            state_tensor = state;  // Already has batch dimension
        }
        
        logDebug("State tensor created, calling forward...");
        auto q_values = forward(state_tensor);
        logDebug("Forward complete, getting argmax...");
        auto action = torch::argmax(q_values, 1);
        
        // FIXED: Use CPU conversion instead of accessor
        auto accessor_val = action.cpu().item<int64_t>();  // FIXED: Use int64_t
        logDebug("Action selected: " + std::to_string(accessor_val));
        
        return action;
    } catch (const std::exception& e) {
        logDebug("ERROR in getAction(tensor): " + std::string(e.what()));
        return torch::tensor({0}, torch::kInt64);
    }
}

// Helper function to convert vector to tensor
torch::Tensor SnakeNeuralNetwork::vectorToTensor(const std::vector<float>& vec) {
    return torch::from_blob(const_cast<float*>(vec.data()), {1, static_cast<long>(vec.size())}, torch::kFloat).clone();
}

// New overloaded getAction method for vector input
torch::Tensor SnakeNeuralNetwork::getAction(const std::vector<float>& state, float epsilon) {
    try {
        logDebug("getAction called with vector input, epsilon: " + std::to_string(epsilon));
        
        if (std::uniform_real_distribution<float>(0.0f, 1.0f)(rng) < epsilon) {
            auto action = torch::randint(0, 4, {1}, torch::kInt64);
            auto accessor_val = action.cpu().item<int64_t>();
            logDebug("Random action selected: " + std::to_string(accessor_val));
            return action;
        }
        
        torch::NoGradGuard no_grad;
        
        // Convert vector to tensor
        auto state_tensor = vectorToTensor(state);
        
        logDebug("State tensor created from vector, calling forward...");
        auto q_values = forward(state_tensor);
        logDebug("Forward complete, getting argmax...");
        auto action = torch::argmax(q_values, 1);
        
        auto accessor_val = action.cpu().item<int64_t>();
        logDebug("Action selected: " + std::to_string(accessor_val));
        
        return action;
    } catch (const std::exception& e) {
        logDebug("ERROR in getAction(vector): " + std::string(e.what()));
        return torch::tensor({0}, torch::kInt64);
    }
}

void SnakeNeuralNetwork::save(const std::string& path) {
    try {
        std::ofstream file(path, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file for saving: " + path);
        }
        
        // Save all parameters
        auto params = parameters();
        
        // Write number of parameters
        size_t num_params = params.size();
        file.write(reinterpret_cast<const char*>(&num_params), sizeof(num_params));
        
        // Save each parameter tensor
        for (const auto& param : params) {
            // Get tensor data on CPU
            auto cpu_param = param.cpu();
            auto sizes = cpu_param.sizes();
            
            // Write tensor dimensions
            size_t num_dims = sizes.size();
            file.write(reinterpret_cast<const char*>(&num_dims), sizeof(num_dims));
            
            for (size_t i = 0; i < num_dims; i++) {
                int64_t dim_size = sizes[i];
                file.write(reinterpret_cast<const char*>(&dim_size), sizeof(dim_size));
            }
            
            // Write tensor data
            auto data_ptr = cpu_param.data_ptr<float>();
            size_t num_elements = cpu_param.numel();
            file.write(reinterpret_cast<const char*>(data_ptr), num_elements * sizeof(float));
        }
        
        file.close();
        logDebug("Model saved successfully to: " + path);
        
    } catch (const std::exception& e) {
        logDebug("ERROR saving model: " + std::string(e.what()));
        throw;
    }
}

void SnakeNeuralNetwork::load(const std::string& path) {
    try {
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file for loading: " + path);
        }
        
        // Read number of parameters
        size_t num_params;
        file.read(reinterpret_cast<char*>(&num_params), sizeof(num_params));
        
        auto params = parameters();
        if (params.size() != num_params) {
            throw std::runtime_error("Parameter count mismatch in saved model");
        }
        
        // Load each parameter tensor
        for (size_t param_idx = 0; param_idx < num_params; param_idx++) {
            // Read tensor dimensions
            size_t num_dims;
            file.read(reinterpret_cast<char*>(&num_dims), sizeof(num_dims));
            
            std::vector<int64_t> sizes(num_dims);
            for (size_t i = 0; i < num_dims; i++) {
                file.read(reinterpret_cast<char*>(&sizes[i]), sizeof(int64_t));
            }
            
            // Create temporary tensor with correct shape
            auto temp_tensor = torch::zeros(sizes, torch::kFloat);
            
            // Read tensor data
            auto data_ptr = temp_tensor.data_ptr<float>();
            size_t num_elements = temp_tensor.numel();
            file.read(reinterpret_cast<char*>(data_ptr), num_elements * sizeof(float));
            
            // Copy to the parameter (no grad tracking)
            {
                torch::NoGradGuard no_grad;
                params[param_idx].copy_(temp_tensor);
            }
        }
        
        file.close();
        logDebug("Model loaded successfully from: " + path);
        
    } catch (const std::exception& e) {
        logDebug("ERROR loading model: " + std::string(e.what()));
        throw;
    }
}

std::vector<torch::Tensor> SnakeNeuralNetwork::parameters() {
    return {w1, b1, w2, b2, w3, b3};
}

void SnakeNeuralNetwork::to(torch::Device device) {
    current_device = device;
    w1 = w1.to(device);
    b1 = b1.to(device);
    w2 = w2.to(device);
    b2 = b2.to(device);
    w3 = w3.to(device);
    b3 = b3.to(device);
}

DQNTrainer::DQNTrainer(int input_size, int hidden_size, int output_size, 
                       float learning_rate_param, float gamma_param) 
    : network(input_size, hidden_size, output_size),
      target_network(input_size, hidden_size, output_size) {
    
    logDebug("DQNTrainer constructor started");
    
    gamma = gamma_param;
    learning_rate = learning_rate_param;
    
    // Copy network parameters to target network
    auto source_params = network.parameters();
    auto target_params = target_network.parameters();
    torch::NoGradGuard no_grad;
    for (size_t i = 0; i < source_params.size(); i++) {
        target_params[i].copy_(source_params[i]);
    }
    
    logDebug("DQNTrainer constructor completed");
}

void DQNTrainer::train(int episodes, bool visual) {
    logDebug("Training started");
    
    // Enhanced FPS control variables
    const int CELL_SIZE = 20;
    const int WINDOW_WIDTH = SnakeGame::GRID_WIDTH * CELL_SIZE;
    const int WINDOW_HEIGHT = SnakeGame::GRID_HEIGHT * CELL_SIZE + 120; // Extra space for controls
    int current_fps = 30;
    bool fps_unlimited = false;
    bool paused = false;
    bool step_mode = false;
    bool step_advance = false;
    
    // FPS measurement
    auto last_frame_time = std::chrono::steady_clock::now();
    int frame_count = 0;
    double actual_fps = 0.0;
    
    if (visual) {
        std::cout << "\n======= ENHANCED FPS CONTROL SYSTEM =======" << std::endl;
        std::cout << "FPS CONTROLS:" << std::endl;
        std::cout << "- UP/DOWN arrows: Adjust FPS ±5" << std::endl;
        std::cout << "- LEFT/RIGHT arrows: Adjust FPS ±1" << std::endl;
        std::cout << "- 0: Unlimited speed (no FPS limit)" << std::endl;
        std::cout << "- 1-9: Set FPS to 1-9" << std::endl;
        std::cout << "- F1: 15 FPS  F2: 30 FPS  F3: 60 FPS" << std::endl;
        std::cout << "\nTRAINING CONTROLS:" << std::endl;
        std::cout << "- SPACE: Pause/Resume training" << std::endl;
        std::cout << "- S: Toggle step mode (advance one frame)" << std::endl;
        std::cout << "- ENTER: Advance one step (in step mode)" << std::endl;
        std::cout << "- ESC: Close window (training continues)" << std::endl;
        std::cout << "==========================================\n" << std::endl;
        
        InitWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "Snake AI Training - Enhanced FPS Controls");
        SetTargetFPS(current_fps);
    }
    
#ifdef NDEBUG
    auto device = torch::kCPU;
    logDebug("Release build: Using CPU only to avoid CUDA issues");
#else
    auto device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    logDebug("Debug build: Training on: " + std::string(device == torch::kCUDA ? "CUDA" : "CPU"));
#endif
    
    network.to(device);
    target_network.to(device);
    
    for (int episode = 0; episode < episodes; episode++) {
        try {
            // Handle window closing
            if (visual && WindowShouldClose()) {
                CloseWindow();
                visual = false;
                std::cout << "\n=== VISUALIZATION CLOSED ===" << std::endl;
                std::cout << "Training continues at full speed..." << std::endl;
                std::cout << "Progress shown every 50 episodes." << std::endl;
                std::cout << "============================\n" << std::endl;
            }
            
            // FPS and control handling
            if (visual) {
                // Handle keyboard input for FPS control
                if (IsKeyPressed(KEY_UP)) {
                    current_fps = std::min(120, current_fps + 5);
                    fps_unlimited = false;
                    SetTargetFPS(current_fps);
                    std::cout << "[FPS] Set to: " << current_fps << std::endl;
                }
                if (IsKeyPressed(KEY_DOWN)) {
                    current_fps = std::max(1, current_fps - 5);
                    fps_unlimited = false;
                    SetTargetFPS(current_fps);
                    std::cout << "[FPS] Set to: " << current_fps << std::endl;
                }
                if (IsKeyPressed(KEY_LEFT)) {
                    current_fps = std::max(1, current_fps - 1);
                    fps_unlimited = false;
                    SetTargetFPS(current_fps);
                    std::cout << "[FPS] Set to: " << current_fps << std::endl;
                }
                if (IsKeyPressed(KEY_RIGHT)) {
                    current_fps = std::min(120, current_fps + 1);
                    fps_unlimited = false;
                    SetTargetFPS(current_fps);
                    std::cout << "[FPS] Set to: " << current_fps << std::endl;
                }
                
                // Number keys for direct FPS setting
                if (IsKeyPressed(KEY_ZERO)) {
                    fps_unlimited = true;
                    SetTargetFPS(0);
                    std::cout << "[FPS] UNLIMITED SPEED MODE" << std::endl;
                }
                for (int i = 1; i <= 9; i++) {
                    if (IsKeyPressed(KEY_ONE + i - 1)) {
                        current_fps = i;
                        fps_unlimited = false;
                        SetTargetFPS(current_fps);
                        std::cout << "[FPS] Set to: " << current_fps << std::endl;
                    }
                }
                
                // Function keys for common FPS values
                if (IsKeyPressed(KEY_F1)) {
                    current_fps = 15;
                    fps_unlimited = false;
                    SetTargetFPS(current_fps);
                    std::cout << "[FPS] Set to: 15 (slow)" << std::endl;
                }
                if (IsKeyPressed(KEY_F2)) {
                    current_fps = 30;
                    fps_unlimited = false;
                    SetTargetFPS(current_fps);
                    std::cout << "[FPS] Set to: 30 (normal)" << std::endl;
                }
                if (IsKeyPressed(KEY_F3)) {
                    current_fps = 60;
                    fps_unlimited = false;
                    SetTargetFPS(current_fps);
                    std::cout << "[FPS] Set to: 60 (fast)" << std::endl;
                }
                
                // Training control keys
                if (IsKeyPressed(KEY_SPACE)) {
                    paused = !paused;
                    std::cout << "[CONTROL] Training " << (paused ? "PAUSED" : "RESUMED") << std::endl;
                }
                if (IsKeyPressed(KEY_S)) {
                    step_mode = !step_mode;
                    if (step_mode) {
                        paused = true;
                        std::cout << "[CONTROL] Step mode ON - Press ENTER to advance" << std::endl;
                    } else {
                        paused = false;
                        std::cout << "[CONTROL] Step mode OFF" << std::endl;
                    }
                }
                if (IsKeyPressed(KEY_ENTER) && step_mode) {
                    step_advance = true;
                    std::cout << "[STEP] Advancing one frame..." << std::endl;
                }
                
                // Handle pause/step logic
                if (paused && !step_advance) {
                    // Skip this episode iteration if paused
                    BeginDrawing();
                    ClearBackground(DARKGRAY);
                    DrawText("TRAINING PAUSED", WINDOW_WIDTH/2 - 100, WINDOW_HEIGHT/2, 24, WHITE);
                    DrawText("Press SPACE to resume", WINDOW_WIDTH/2 - 120, WINDOW_HEIGHT/2 + 30, 16, LIGHTGRAY);
                    EndDrawing();
                    continue;
                }
                step_advance = false; // Reset step advance flag
                
                // Calculate actual FPS
                auto current_time = std::chrono::steady_clock::now();
                frame_count++;
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - last_frame_time);
                if (duration.count() >= 1000) { // Update every second
                    actual_fps = frame_count * 1000.0 / duration.count();
                    frame_count = 0;
                    last_frame_time = current_time;
                }
            }
            logDebug("Starting episode " + std::to_string(episode));
            
            SnakeGame game;
            auto state = game.getGameState();
            float total_reward = 0.0f;
            int steps = 0;
            
            logDebug("Game initialized, starting episode loop");
            
            while (!game.isGameOver() && steps < 1000) {
                logDebug("Step " + std::to_string(steps) + " - getting action");
                
                // Enhanced visualization with FPS info
                if (visual) {
                    BeginDrawing();
                    ClearBackground(DARKGRAY);
                    
                    // Draw snake
                    const auto& snake = game.getSnakeBody();
                    for (size_t i = 0; i < snake.size(); i++) {
                        Color color = (i == 0) ? DARKGREEN : GREEN;
                        DrawRectangle(snake[i].x * CELL_SIZE, snake[i].y * CELL_SIZE, 
                                     CELL_SIZE - 1, CELL_SIZE - 1, color);
                    }
                    
                    // Draw food
                    const auto& food = game.getFoodPosition();
                    DrawRectangle(food.x * CELL_SIZE, food.y * CELL_SIZE, 
                                 CELL_SIZE - 1, CELL_SIZE - 1, RED);
                    
                    // Enhanced training info with FPS display
                    DrawText(TextFormat("Episode: %d/%d", episode + 1, episodes), 10, WINDOW_HEIGHT - 100, 16, WHITE);
                    DrawText(TextFormat("Score: %d  Steps: %d  Epsilon: %.3f", game.getScore(), steps, epsilon), 10, WINDOW_HEIGHT - 80, 16, WHITE);
                    
                    // FPS information
                    if (fps_unlimited) {
                        DrawText("FPS: UNLIMITED", 10, WINDOW_HEIGHT - 60, 16, YELLOW);
                    } else {
                        DrawText(TextFormat("Target FPS: %d | Actual: %.1f", current_fps, actual_fps), 10, WINDOW_HEIGHT - 60, 16, YELLOW);
                    }
                    
                    // Control hints
                    if (paused) {
                        DrawText("PAUSED - SPACE to resume", 10, WINDOW_HEIGHT - 40, 14, RED);
                    } else if (step_mode) {
                        DrawText("STEP MODE - ENTER to advance", 10, WINDOW_HEIGHT - 40, 14, ORANGE);
                    } else {
                        DrawText("UP/DOWN: ±5 FPS | LEFT/RIGHT: ±1 FPS | 0: Unlimited", 10, WINDOW_HEIGHT - 40, 14, LIGHTGRAY);
                    }
                    
                    DrawText("SPACE: Pause | S: Step Mode | ESC: Close", 10, WINDOW_HEIGHT - 20, 14, LIGHTGRAY);
                    
                    EndDrawing();
                }
                
                auto action_tensor = network.getAction(state, epsilon);
                
                // FIXED: Use CPU conversion instead of accessor
                int action = static_cast<int>(action_tensor.cpu().item<int64_t>());  // FIXED: Convert from int64_t
                
                logDebug("Action obtained: " + std::to_string(action));
                
                // Validate action bounds - CRITICAL
                if (action < 0 || action >= 4) {
                    logDebug("ERROR: Invalid action " + std::to_string(action) + " detected!");
                    action = 0; // Default to UP direction
                }
                
                Direction dir = static_cast<Direction>(action);
                game.setDirection(dir);
                
                game.update();
                auto next_state = game.getGameState();
                float reward = game.getReward();
                bool done = game.isGameOver();
                
                logDebug("Game step complete, adding experience");
                
                // Store experience with validation
                addExperience({state, action, reward, next_state, done});
                
                state = next_state;
                total_reward += reward;
                steps++;
                
                // Only do replay training after we have enough experiences
                if (replay_buffer.size() >= batch_size) {
                    logDebug("Starting replay training");
                    replayTraining();
                    logDebug("Replay training complete");
                }
            }
            
            if (epsilon > epsilon_min) {
                epsilon *= epsilon_decay;
            }
            
            if (episode % 100 == 0) {
                updateTargetNetwork();
            }
            
            if (episode % 50 == 0) {
                float avg_reward = total_reward / steps;
                std::string msg = "Episode " + std::to_string(episode) + 
                                ", Score: " + std::to_string(game.getScore()) + 
                                ", Steps: " + std::to_string(steps) +
                                ", Total Reward: " + std::to_string(total_reward) + 
                                ", Avg Reward: " + std::to_string(avg_reward) +
                                ", Epsilon: " + std::to_string(epsilon);
                logDebug(msg);
                
                // Additional debugging for food-seeking behavior
                if (episode % 100 == 0) {
                    std::cout << "\n=== TRAINING STATUS ===" << std::endl;
                    std::cout << "Episode: " << episode << "/" << episodes << std::endl;
                    std::cout << "Best Score This Episode: " << game.getScore() << std::endl;
                    std::cout << "Steps Survived: " << steps << std::endl;
                    std::cout << "Total Reward: " << total_reward << std::endl;
                    std::cout << "Average Reward per Step: " << (total_reward / steps) << std::endl;
                    std::cout << "Current Epsilon: " << epsilon << std::endl;
                    std::cout << "Experience Buffer Size: " << replay_buffer.size() << std::endl;
                    std::cout << "======================\n" << std::endl;
                }
            }
            
        } catch (const std::exception& e) {
            logDebug("ERROR in episode " + std::to_string(episode) + ": " + std::string(e.what()));
            // Continue to next episode instead of crashing
            continue;
        }
    }
    
    // Close raylib window if visual mode
    if (visual) {
        CloseWindow();
    }
    
    logDebug("Training completed");
}

void DQNTrainer::saveModel(const std::string& path) {
    network.save(path);
}

void DQNTrainer::loadModel(const std::string& path) {
    network.load(path);
}

void DQNTrainer::updateTargetNetwork() {
    auto source_params = network.parameters();
    auto target_params = target_network.parameters();
    torch::NoGradGuard no_grad;
    for (size_t i = 0; i < source_params.size(); i++) {
        target_params[i].copy_(source_params[i]);
    }
}

void DQNTrainer::replayTraining() {
    try {
        logDebug("replayTraining: Entry");
        
        if (replay_buffer.size() < batch_size) {
            logDebug("replayTraining: Not enough experiences, returning");
            return;
        }
        
        auto device = network.parameters()[0].device();
        logDebug("replayTraining: Device obtained");
        
        std::vector<int> indices(replay_buffer.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), std::default_random_engine{});
        
        logDebug("replayTraining: Indices shuffled");
        
        std::vector<std::vector<float>> states, next_states;
        std::vector<int> actions;
        std::vector<float> rewards;
        std::vector<bool> dones;
        
        // Extract batch data with validation
        for (int i = 0; i < batch_size; i++) {
            auto& exp = replay_buffer[indices[i]];
            states.push_back(exp.state);
            next_states.push_back(exp.next_state);
            
            // CRITICAL: Validate actions from experience buffer
            int action = exp.action;
            if (action < 0 || action >= 4) {
                logDebug("ERROR: Corrupted action " + std::to_string(action) + " in experience buffer!");
                action = 0; // Default safe value
            }
            actions.push_back(action);
            
            rewards.push_back(exp.reward);
            dones.push_back(exp.done);
        }
        
        logDebug("replayTraining: Batch data extracted");
        
        // Create tensors safely with detailed logging
        logDebug("replayTraining: Creating state tensors");
        auto state_tensor = torch::zeros({batch_size, static_cast<int>(states[0].size())}, torch::kFloat);
        auto next_state_tensor = torch::zeros({batch_size, static_cast<int>(next_states[0].size())}, torch::kFloat);
        
        // Copy data element by element (safe)
        for (int i = 0; i < batch_size; i++) {
            for (size_t j = 0; j < states[i].size(); j++) {
                state_tensor[i][j] = states[i][j];
                next_state_tensor[i][j] = next_states[i][j];
            }
        }
        
        state_tensor = state_tensor.to(device);
        next_state_tensor = next_state_tensor.to(device);
        
        logDebug("replayTraining: Creating action tensor");
        
        // Create action tensor with int64 type for gather operation
        auto action_tensor = torch::zeros({batch_size}, torch::kInt64);  // FIXED: Use torch::kInt64 for gather
        for (int i = 0; i < batch_size; i++) {
            if (actions[i] < 0 || actions[i] >= 4) {
                logDebug("FATAL: Invalid action " + std::to_string(actions[i]) + " at batch index " + std::to_string(i));
                actions[i] = 0; // Force safe value
            }
            action_tensor[i] = actions[i];
        }
        action_tensor = action_tensor.to(device);
        
        logDebug("replayTraining: Creating reward tensor");
        
        // Create reward tensor safely
        auto reward_tensor = torch::zeros({batch_size}, torch::kFloat);
        for (int i = 0; i < batch_size; i++) {
            reward_tensor[i] = rewards[i];
        }
        reward_tensor = reward_tensor.to(device);
        
        logDebug("replayTraining: All tensors created, calling network forward");
        
        // Debug print before gather operation
        auto q_values_out = network.forward(state_tensor);
        logDebug("Q-values shape: [" + std::to_string(q_values_out.size(0)) + ", " + std::to_string(q_values_out.size(1)) + "]");
        logDebug("Action tensor shape: [" + std::to_string(action_tensor.size(0)) + "]");
        // FIXED: Use CPU conversion to avoid accessor issues
        auto min_val = action_tensor.min().cpu().item<int64_t>();  // FIXED: Use int64_t
        auto max_val = action_tensor.max().cpu().item<int64_t>();  // FIXED: Use int64_t
        logDebug("Action tensor min/max: " + std::to_string(min_val) + "/" + std::to_string(max_val));
        
        logDebug("replayTraining: About to call gather operation");
        
        // Safe gather operation
        auto current_q_values = q_values_out.gather(1, action_tensor.unsqueeze(1));
        
        logDebug("replayTraining: Gather operation successful");
        
        torch::Tensor next_q_values;
        {
            torch::NoGradGuard no_grad;
            next_q_values = std::get<0>(target_network.forward(next_state_tensor).max(1));
        }
        
        auto target_q_values = reward_tensor + gamma * next_q_values;
        
        for (int i = 0; i < batch_size; i++) {
            if (dones[i]) {
                target_q_values[i] = reward_tensor[i];
            }
        }
        
        auto loss = torch::mse_loss(current_q_values.squeeze(), target_q_values);
        
        logDebug("replayTraining: Loss calculated, starting backpropagation");
        
        // Manual gradient update (like firstNN project)
        loss.backward();
        
        // Manual parameter updates
        auto params = network.parameters();
        {
            torch::NoGradGuard no_grad;
            if (params[0].grad().defined()) {
                params[0] -= learning_rate * params[0].grad();
                params[0].grad().zero_();
            }
            if (params[1].grad().defined()) {
                params[1] -= learning_rate * params[1].grad();
                params[1].grad().zero_();
            }
            if (params[2].grad().defined()) {
                params[2] -= learning_rate * params[2].grad();
                params[2].grad().zero_();
            }
            if (params[3].grad().defined()) {
                params[3] -= learning_rate * params[3].grad();
                params[3].grad().zero_();
            }
            if (params[4].grad().defined()) {
                params[4] -= learning_rate * params[4].grad();
                params[4].grad().zero_();
            }
            if (params[5].grad().defined()) {
                params[5] -= learning_rate * params[5].grad();
                params[5].grad().zero_();
            }
        }
        
        logDebug("replayTraining: Backpropagation complete");
        
    } catch (const std::exception& e) {
        logDebug("ERROR in replayTraining(): " + std::string(e.what()));
        throw; // Re-throw to see the error
    }
}

void DQNTrainer::addExperience(const Experience& exp) {
    try {
        // Validate experience before adding
        if (exp.action < 0 || exp.action >= 4) {
            logDebug("ERROR: Attempting to store invalid action " + std::to_string(exp.action));
            return; // Skip this experience
        }
        
        logDebug("Adding experience with action: " + std::to_string(exp.action));
        
        replay_buffer.push_back(exp);
        if (replay_buffer.size() > max_buffer_size) {
            replay_buffer.erase(replay_buffer.begin());
        }
        
        logDebug("Experience added, buffer size: " + std::to_string(replay_buffer.size()));
        
    } catch (const std::exception& e) {
        logDebug("ERROR in addExperience(): " + std::string(e.what()));
    }
}
