torch::Tensor SnakeNeuralNetworkImpl::getAction(const std::vector<float>& state, float epsilon) {
    try {
        logDebug("getAction called with epsilon: " + std::to_string(epsilon));
        
        if (std::uniform_real_distribution<float>(0.0f, 1.0f)(rng) < epsilon) {
            auto action = torch::randint(0, 4, {1}, torch::kInt);
            // FIXED: Use accessor instead of item<>()
            auto accessor = action.accessor<int, 1>();
            logDebug("Random action selected: " + std::to_string(accessor[0]));
            return action;
        }
        
        torch::NoGradGuard no_grad;
        
        auto state_tensor = torch::zeros({1, static_cast<int>(state.size())}, torch::kFloat);
        for (size_t i = 0; i < state.size(); i++) {
            state_tensor[0][i] = state[i];
        }
        
        logDebug("State tensor created, calling forward...");
        auto q_values = forward(state_tensor);
        logDebug("Forward complete, getting argmax...");
        auto action = torch::argmax(q_values, 1);
        
        // FIXED: Use accessor instead of item<>()
        auto accessor = action.accessor<int, 1>();
        logDebug("Action selected: " + std::to_string(accessor[0]));
        
        return action;
    } catch (const std::exception& e) {
        logDebug("ERROR in getAction(): " + std::string(e.what()));
        return torch::tensor({0}, torch::kInt);
    }
}