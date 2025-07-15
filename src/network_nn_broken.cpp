        
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
