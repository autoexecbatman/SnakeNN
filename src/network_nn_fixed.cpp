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

// Helper function to convert vector to tensor
torch::Tensor SnakeNeuralNetwork::vectorToTensor(const std::vector<float>& vec) {
    return torch::from_blob(const_cast<float*>(vec.data()), {1, static_cast<long>(vec.size())}, torch::kFloat).clone();
}

// Original getAction method for tensor input
torch::Tensor SnakeNeuralNetwork::getAction(const torch::Tensor& state, float epsilon) {
    try {
        logDebug("getAction called with tensor input, epsilon: " + std::to_string(epsilon));
        
        if (std::uniform_real_distribution<float>(0.0f, 1.0f)(rng) < epsilon) {
            auto action = torch::randint(0, 4, {1}, torch::kInt64);
            auto accessor_val = action.cpu().item<int64_t>();
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
        
        auto accessor_val = action.cpu().item<int64_t>();
        logDebug("Action selected: " + std::to_string(accessor_val));
        
        return action;
    } catch (const std::exception& e) {
        logDebug("ERROR in getAction(tensor): " + std::string(e.what()));
        return torch::tensor({0}, torch::kInt64);
    }
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
    logDebug("Save not implemented for manual tensor approach");
    // TODO: Implement custom save mechanism if needed
}

void SnakeNeuralNetwork::load(const std::string& path) {
    logDebug("Load not implemented for manual tensor approach");
    // TODO: Implement custom load mechanism if needed
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
      target_network(input_size, hidden_size, output_size),
      learning_rate(learning_rate_param),
      gamma(gamma_param) {
    logDebug("DQNTrainer initialized");
}

void DQNTrainer::train(int episodes, bool visual) {
    logDebug("DQN Training started for " + std::to_string(episodes) + " episodes");
    
    SnakeGame game;
    int best_score = 0;
    
    for (int episode = 0; episode < episodes; ++episode) {
        game.reset();
        auto state = game.getGameState();
        float total_reward = 0;
        int steps = 0;
        
        while (!game.isGameOver() && steps < 1000) {
            // Use vector-based getAction
            auto action_tensor = network.getAction(state, epsilon);
            int action = action_tensor.cpu().item<int64_t>();
            
            Direction dir = static_cast<Direction>(action);
            game.setDirection(dir);
            
            bool game_continues = game.update();
            auto next_state = game.getGameState();
            float reward = game.getReward();
            bool done = !game_continues;
            
            // Store experience
            Experience exp{state, action, reward, next_state, done};
            addExperience(exp);
            
            state = next_state;
            total_reward += reward;
            steps++;
            
            // Train on batch
            if (replay_buffer.size() >= batch_size) {
                replayTraining();
            }
        }
        
        // Update target network periodically
        if (episode % 100 == 0) {
            updateTargetNetwork();
            logDebug("Episode " + std::to_string(episode) + ", Total Reward: " + std::to_string(total_reward) + ", Epsilon: " + std::to_string(epsilon));
        }
        
        // Decay epsilon
        if (epsilon > epsilon_min) {
            epsilon *= epsilon_decay;
        }
    }
    
    logDebug("Training completed");
}

void DQNTrainer::addExperience(const Experience& exp) {
    replay_buffer.push_back(exp);
    if (replay_buffer.size() > max_buffer_size) {
        replay_buffer.erase(replay_buffer.begin());
    }
}

void DQNTrainer::replayTraining() {
    if (replay_buffer.size() < batch_size) return;
    
    // Sample random batch
    std::vector<int> indices;
    for (int i = 0; i < batch_size; ++i) {
        indices.push_back(rand() % replay_buffer.size());
    }
    
    // Process batch (simplified implementation)
    for (int idx : indices) {
        const auto& exp = replay_buffer[idx];
        
        // Convert vectors to tensors for computation
        auto state_tensor = network.vectorToTensor(exp.state);
        auto next_state_tensor = network.vectorToTensor(exp.next_state);
        
        auto current_q = network.forward(state_tensor);
        auto next_q = target_network.forward(next_state_tensor);
        
        float target = exp.reward;
        if (!exp.done) {
            target += gamma * torch::max(next_q).item<float>();
        }
        
        // Manual gradient update would go here
        // For now, this is a simplified version
    }
}

void DQNTrainer::updateTargetNetwork() {
    // Copy weights from main network to target network
    auto main_params = network.parameters();
    auto target_params = target_network.parameters();
    
    torch::NoGradGuard no_grad;
    for (size_t i = 0; i < main_params.size(); ++i) {
        target_params[i].copy_(main_params[i]);
    }
    
    logDebug("Target network updated");
}

void DQNTrainer::saveModel(const std::string& path) {
    network.save(path);
}

void DQNTrainer::loadModel(const std::string& path) {
    network.load(path);
}
