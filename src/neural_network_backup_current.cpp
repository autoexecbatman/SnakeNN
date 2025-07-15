#include "neural_network.h"
#include <iostream>
#include <random>
#include <numeric>
#include <algorithm>
#include "snake_logic.h"

SnakeNeuralNetworkImpl::SnakeNeuralNetworkImpl(int input_size, int hidden_size, int output_size) 
    : fc1(register_module("fc1", torch::nn::Linear(input_size, hidden_size))),
      fc2(register_module("fc2", torch::nn::Linear(hidden_size, hidden_size))),
      fc3(register_module("fc3", torch::nn::Linear(hidden_size, output_size))),
      rng(std::random_device{}()) {
}

torch::Tensor SnakeNeuralNetworkImpl::forward(torch::Tensor x) {
    x = torch::relu(fc1->forward(x));
    x = torch::relu(fc2->forward(x));
    x = fc3->forward(x);
    return x;
}

torch::Tensor SnakeNeuralNetworkImpl::getAction(const std::vector<float>& state, float epsilon) {
    if (std::uniform_real_distribution<float>(0.0f, 1.0f)(rng) < epsilon) {
        return torch::randint(0, 4, {1});
    }
    
    torch::NoGradGuard no_grad;
    auto state_tensor = torch::from_blob(const_cast<float*>(state.data()), 
                                       {1, static_cast<long>(state.size())}, 
                                       torch::kFloat);
    auto q_values = forward(state_tensor);
    return torch::argmax(q_values, 1);
}

void SnakeNeuralNetworkImpl::save(const std::string& path) {
    torch::save(this->parameters(), path);
}

void SnakeNeuralNetworkImpl::load(const std::string& path) {
    std::vector<torch::Tensor> params;
    torch::load(params, path);
    
    auto model_params = this->parameters();
    for (size_t i = 0; i < params.size() && i < model_params.size(); i++) {
        model_params[i].data().copy_(params[i].data());
    }
}

DQNTrainer::DQNTrainer(int input_size, int hidden_size, int output_size, 
                       float learning_rate, float gamma_param) {
    gamma = gamma_param;
    
    network = register_module("network", SnakeNeuralNetwork(input_size, hidden_size, output_size));
    target_network = register_module("target_network", SnakeNeuralNetwork(input_size, hidden_size, output_size));
    
    auto source_params = network->named_parameters();
    auto target_params = target_network->named_parameters();
    torch::NoGradGuard no_grad;
    for (auto& pair : source_params) {
        target_params[pair.key()].copy_(pair.value());
    }
    
    optimizer = std::make_unique<torch::optim::Adam>(network->parameters(), torch::optim::AdamOptions(learning_rate));
}

void DQNTrainer::train(int episodes) {
#ifdef NDEBUG
    auto device = torch::kCPU;
    std::cout << "Release build: Using CPU only to avoid CUDA issues" << std::endl;
#else
    auto device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    std::cout << "Debug build: Training on: " << (device == torch::kCUDA ? "CUDA" : "CPU") << std::endl;
#endif
    
    network->to(device);
    target_network->to(device);
    
    for (int episode = 0; episode < episodes; episode++) {
        SnakeGame game;
        auto state = game.getGameState();
        float total_reward = 0.0f;
        int steps = 0;
        
        while (!game.isGameOver() && steps < 1000) {
            auto action_tensor = network->getAction(state, epsilon);
            int action = action_tensor.item<int>();
            
            Direction dir = static_cast<Direction>(action);
            game.setDirection(dir);
            
            game.update();
            auto next_state = game.getGameState();
            float reward = game.getReward();
            bool done = game.isGameOver();
            
            addExperience({state, action, reward, next_state, done});
            
            state = next_state;
            total_reward += reward;
            steps++;
            
            if (replay_buffer.size() >= batch_size) {
                replayTraining();
            }
        }
        
        if (epsilon > epsilon_min) {
            epsilon *= epsilon_decay;
        }
        
        if (episode % 100 == 0) {
            updateTargetNetwork();
        }
        
        if (episode % 50 == 0) {
            std::cout << "Episode " << episode << ", Score: " << game.getScore() 
                     << ", Total Reward: " << total_reward 
                     << ", Epsilon: " << epsilon << std::endl;
        }
    }
}

void DQNTrainer::saveModel(const std::string& path) {
    network->save(path);
}

void DQNTrainer::loadModel(const std::string& path) {
    network->load(path);
}

void DQNTrainer::updateTargetNetwork() {
    auto source_params = network->named_parameters();
    auto target_params = target_network->named_parameters();
    torch::NoGradGuard no_grad;
    for (auto& pair : source_params) {
        target_params[pair.key()].copy_(pair.value());
    }
}

void DQNTrainer::replayTraining() {
    if (replay_buffer.size() < batch_size) return;
    
    auto device = network->parameters()[0].device();
    
    std::vector<int> indices(replay_buffer.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), std::default_random_engine{});
    
    std::vector<std::vector<float>> states, next_states;
    std::vector<int> actions;
    std::vector<float> rewards;
    std::vector<bool> dones;
    
    for (int i = 0; i < batch_size; i++) {
        auto& exp = replay_buffer[indices[i]];
        states.push_back(exp.state);
        next_states.push_back(exp.next_state);
        actions.push_back(exp.action);
        rewards.push_back(exp.reward);
        dones.push_back(exp.done);
    }
    
    auto state_tensor = torch::zeros({batch_size, static_cast<long>(states[0].size())});
    auto next_state_tensor = torch::zeros({batch_size, static_cast<long>(next_states[0].size())});
    
    for (int i = 0; i < batch_size; i++) {
        state_tensor[i] = torch::from_blob(states[i].data(), {static_cast<long>(states[i].size())}, torch::kFloat);
        next_state_tensor[i] = torch::from_blob(next_states[i].data(), {static_cast<long>(next_states[i].size())}, torch::kFloat);
    }
    
    state_tensor = state_tensor.to(device);
    next_state_tensor = next_state_tensor.to(device);
    auto action_tensor = torch::from_blob(actions.data(), {batch_size}, torch::kLong).to(device);
    auto reward_tensor = torch::from_blob(rewards.data(), {batch_size}, torch::kFloat).to(device);
    
    auto current_q_values = network->forward(state_tensor).gather(1, action_tensor.unsqueeze(1));
    
    torch::Tensor next_q_values;
    {
        torch::NoGradGuard no_grad;
        next_q_values = std::get<0>(target_network->forward(next_state_tensor).max(1));
    }
    
    auto target_q_values = reward_tensor + gamma * next_q_values;
    
    for (int i = 0; i < batch_size; i++) {
        if (dones[i]) {
            target_q_values[i] = reward_tensor[i];
        }
    }
    
    auto loss = torch::mse_loss(current_q_values.squeeze(), target_q_values);
    
    optimizer->zero_grad();
    loss.backward();
    optimizer->step();
}

void DQNTrainer::addExperience(const Experience& exp) {
    replay_buffer.push_back(exp);
    if (replay_buffer.size() > max_buffer_size) {
        replay_buffer.erase(replay_buffer.begin());
    }
}
