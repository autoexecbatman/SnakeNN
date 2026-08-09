#pragma once
#include <torch/torch.h>
#include <vector>

// Advanced snake AI using manual tensor implementation (following project pattern)
class AdvancedSnakeNetwork {
public:
    AdvancedSnakeNetwork(int grid_width = 20, int grid_height = 20, int memory_size = 128);
    
    // Forward pass with spatial reasoning
    torch::Tensor forward(torch::Tensor grid_input, torch::Tensor memory_input);
    
    // Get action with safety validation
    std::pair<int, torch::Tensor> getActionWithMemory(
        const std::vector<std::vector<int>>& grid, 
        torch::Tensor memory_state,
        float exploration_rate = 0.0f
    );
    
    // Multi-objective prediction outputs
    struct ActionPrediction {
        torch::Tensor safety_scores;
        torch::Tensor food_scores;
        torch::Tensor exploration_scores;
        torch::Tensor combined_action;
        torch::Tensor new_memory_state;
    };
    
    ActionPrediction predictAction(torch::Tensor grid_input, torch::Tensor memory_input);
    
    void save(const std::string& path);
    void load(const std::string& path);
    
    // Get parameters for training
    std::vector<torch::Tensor> parameters();
    
private:
    int grid_width_, grid_height_, memory_size_;
    
    // Manual CNN weights (following project pattern - no register_module)
    torch::Tensor conv1_weight, conv1_bias;
    torch::Tensor conv2_weight, conv2_bias;
    torch::Tensor conv3_weight, conv3_bias;
    
    // CNN to LSTM projection layer - FIXES DIMENSION MISMATCH
    torch::Tensor cnn_projection_weight, cnn_projection_bias;
    
    // Memory network weights
    torch::Tensor lstm_weight_ih, lstm_weight_hh, lstm_bias_ih, lstm_bias_hh;
    torch::Tensor memory_proj_weight, memory_proj_bias;
    
    // Multi-head decision weights
    torch::Tensor safety_weight, safety_bias;
    torch::Tensor food_weight, food_bias;
    torch::Tensor exploration_weight, exploration_bias;
    torch::Tensor combiner_weight, combiner_bias;
    
    // Feature extraction weights
    torch::Tensor feature_weight, feature_bias;
    
    // Manual implementations
    torch::Tensor conv2d_manual(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int padding = 1);
    torch::Tensor lstm_cell_manual(torch::Tensor input, torch::Tensor hidden, torch::Tensor cell,
                                  torch::Tensor weight_ih, torch::Tensor weight_hh, 
                                  torch::Tensor bias_ih, torch::Tensor bias_hh);
    
    int calculateFeatureSize();
    torch::Tensor encodeGridState(const std::vector<std::vector<int>>& grid);
    void initializeWeights();
};

// Advanced trainer with comprehensive safety and performance optimization
class AdvancedSnakeTrainer {
public:
    AdvancedSnakeTrainer(int grid_width = 20, int grid_height = 20);
    
    void train(int episodes = 10000, bool verbose = true);
    
    struct TrainingMetrics {
        float avg_score;
        float collision_avoidance_rate;
        float avg_survival_steps;
        float exploration_efficiency;
        int games_25_percent_complete;
        int games_50_percent_complete;
        int excellent_tier_games;
        int master_tier_games;
        float overall_performance_score;
    };
    
    TrainingMetrics evaluateModel(int test_episodes = 200);
    
private:
    AdvancedSnakeNetwork network_;
    int grid_width_, grid_height_;
    float learning_rate_;
    
    // Multi-objective reward system
    struct RewardComponents {
        float safety_reward;
        float food_reward;
        float exploration_reward;
        float efficiency_reward;
    };
    
    RewardComponents calculateRewards(
        const class SnakeGame& game,
        const std::vector<std::vector<int>>& prev_grid,
        const std::vector<std::vector<int>>& current_grid,
        int action_taken,
        bool collision_occurred,
        bool food_collected,
        int steps_taken
    );
    
    std::vector<std::vector<int>> createAdvancedGrid(const class SnakeGame& game);
    bool validateActionSafety(const class SnakeGame& game, int action);
    std::vector<int> getSafeActions(const class SnakeGame& game);
    
    // Training phases
    void phaseCollisionAvoidance(int episodes);
    void phaseFoodSeeking(int episodes);
    void phaseOptimization(int episodes);
    
    // Manual gradient update
    void updateWeights(const std::vector<torch::Tensor>& gradients, float lr);
};
