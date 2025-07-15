#include "neural_network.h"
#include "snake_logic.h"
#include <iostream>

int main() {
    std::cout << "=== Snake Neural Network Trainer ===" << std::endl;
    
    // Check CUDA availability
    bool cuda_available = torch::cuda::is_available();
    std::cout << "CUDA available: " << std::boolalpha << cuda_available << std::endl;
    
    try {
        std::cout << "=== MINIMAL TEST MODE ===" << std::endl;
        
        // Test 1: Create simple tensors
        std::cout << "Test 1: Creating basic tensors..." << std::endl;
        auto test_tensor = torch::zeros({2, 4}, torch::kFloat);
        auto test_actions = torch::tensor({0, 1}, torch::kInt64);  // FIXED: Use torch::kInt64 for gather
        std::cout << "Basic tensors created successfully" << std::endl;
        
        // Test 2: Test gather operation directly
        std::cout << "Test 2: Testing gather operation..." << std::endl;
        std::cout << "Test tensor shape: [" << test_tensor.size(0) << ", " << test_tensor.size(1) << "]" << std::endl;
        std::cout << "Action tensor shape: [" << test_actions.size(0) << "]" << std::endl;
        std::cout << "Action values: " << test_actions[0].cpu().item<int64_t>() << ", " << test_actions[1].cpu().item<int64_t>() << std::endl;  // FIXED: Use int64_t
        
        auto gathered = test_tensor.gather(1, test_actions.unsqueeze(1));
        std::cout << "Gather operation successful!" << std::endl;
        
        // Test 3: Create neural network
        std::cout << "Test 3: Creating neural network..." << std::endl;
        SnakeNeuralNetwork network(14, 128, 4);
        std::cout << "Neural network created successfully" << std::endl;
        
        // Test 4: Test forward pass
        std::cout << "Test 4: Testing forward pass..." << std::endl;
        auto input = torch::randn({1, 14});
        auto output = network.forward(input);
        std::cout << "Forward pass successful, output shape: [" << output.size(0) << ", " << output.size(1) << "]" << std::endl;
        
        // Test 5: Test action selection
        std::cout << "Test 5: Testing action selection..." << std::endl;
        std::vector<float> state(14, 0.5f);
        auto action = network.getAction(state, 0.0f); // No epsilon
        
        // FIXED: Use CPU conversion instead of accessor
        int action_value = static_cast<int>(action.cpu().item<int64_t>());  // FIXED: Convert from int64_t
        std::cout << "Action selection successful, action: " << action_value << std::endl;
        
        std::cout << "=== ALL TESTS PASSED - PROCEEDING WITH LIMITED TRAINING ===" << std::endl;
        
        // Only proceed if all tests pass
        DQNTrainer trainer(14, 128, 4, 0.001f, 0.95f);
        
        // Ask user for visualization preference
        std::cout << "\nDo you want to see training visualization? (y/n): ";
        char choice;
        std::cin >> choice;
        
        bool visual = (choice == 'y' || choice == 'Y');
        
        if (visual) {
            std::cout << "Starting full training (5000 episodes) with visualization..." << std::endl;
            std::cout << "Close the window to stop training early." << std::endl;
        } else {
            std::cout << "Starting full training (5000 episodes) - no visualization..." << std::endl;
            std::cout << "Training will run faster without graphics." << std::endl;
        }
        
        trainer.train(5000, visual);
        
        std::cout << "Training complete!" << std::endl;
    }
    catch (const c10::Error& e) {
        std::cerr << "LibTorch Error: " << e.what() << std::endl;
        return -1;
    }
    catch (const std::exception& e) {
        std::cerr << "Standard Error: " << e.what() << std::endl;
        return -1;
    }
    catch (...) {
        std::cerr << "Unknown error occurred!" << std::endl;
        return -1;
    }
    
    return 0;
}
