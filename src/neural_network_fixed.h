#pragma once
#include <torch/torch.h>
#include <memory>
#include <random>

// Manual neural network implementation (no torch::nn::Module)
class SnakeNeuralNetwork {
public:
    SnakeNeuralNetwork(int input_size = 14, int hidden_size = 128, int output_size = 4);
    
    torch::Tensor forward(torch::Tensor x);
    
    // Overloaded getAction methods to handle both vector and tensor inputs
    torch::Tensor getAction(const torch::Tensor& state, float epsilon = 0.0f);
    torch::Tensor getAction(const std::vector<float>& state, float epsilon = 0.0f);  // New overload
    
    void save(const std::string& path);
    void load(const std::string& path);
    
    // Get parameters for optimizer
    std::vector<torch::Tensor> parameters();
    
    // Move to device
    void to(torch::Device device);

private:
    // Manual weight tensors (like firstNN project)
    torch::Tensor w1, b1, w2, b2, w3, b3;
    std::mt19937 rng;
    torch::Device current_device{torch::kCPU};
    
    // Helper function to convert vector to tensor
    torch::Tensor vectorToTensor(const std::vector<float>& vec);
};

class DQNTrainer {
public:
    DQNTrainer(int input_size = 14, int hidden_size = 128, int output_size = 4, 
               float learning_rate = 0.001f, float gamma_param = 0.95f);
    
    void train(int episodes = 1000, bool visual = false);
    void saveModel(const std::string& path);
    void loadModel(const std::string& path);
    
private:
    SnakeNeuralNetwork network;
    SnakeNeuralNetwork target_network;
    float learning_rate = 0.001f;  // Manual gradient update
    
    float gamma = 0.95f;
    float epsilon = 1.0f;
    float epsilon_decay = 0.998f;  // Slower decay for better exploration
    float epsilon_min = 0.05f;     // Higher minimum for continued exploration
    
    struct Experience {
        std::vector<float> state;
        int action;
        float reward;
        std::vector<float> next_state;
        bool done;
    };
    
    std::vector<Experience> replay_buffer;
    const int max_buffer_size = 10000;
    const int batch_size = 32;
    
    void updateTargetNetwork();
    void replayTraining();
    void addExperience(const Experience& exp);
};
