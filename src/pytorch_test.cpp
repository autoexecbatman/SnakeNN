#include <torch/torch.h>
#include <iostream>

int main() {
    std::cout << "=== Basic PyTorch Test ===" << std::endl;
    
    try {
        // Test basic tensor operations
        std::cout << "Creating basic tensors..." << std::endl;
        auto tensor1 = torch::ones({2, 3});
        auto tensor2 = torch::zeros({2, 3});
        
        std::cout << "Tensor1 shape: [" << tensor1.size(0) << ", " << tensor1.size(1) << "]" << std::endl;
        std::cout << "Tensor2 shape: [" << tensor2.size(0) << ", " << tensor2.size(1) << "]" << std::endl;
        
        // Test basic operations
        auto result = tensor1 + tensor2;
        std::cout << "Addition successful" << std::endl;
        
        // Test neural network creation - FIXED: Use explicit module creation
        std::cout << "Creating simple neural network..." << std::endl;
        
        // Create modules individually first
        auto linear1 = torch::nn::Linear(10, 5);
        auto relu = torch::nn::ReLU();
        auto linear2 = torch::nn::Linear(5, 1);
        
        std::cout << "Individual modules created successfully" << std::endl;
        
        // Test manual forward pass instead of Sequential
        auto input = torch::randn({1, 10});
        std::cout << "Input created, shape: [" << input.size(0) << ", " << input.size(1) << "]" << std::endl;
        
        // Manual forward pass
        auto hidden = linear1->forward(input);
        hidden = relu->forward(hidden);
        auto output = linear2->forward(hidden);
        
        std::cout << "Manual forward pass successful" << std::endl;
        std::cout << "Output shape: [" << output.size(0) << ", " << output.size(1) << "]" << std::endl;
        
        // Test tensor value extraction (our key fix)
        std::cout << "Testing tensor value extraction..." << std::endl;
        auto test_tensor = torch::tensor({42});
        auto extracted_value = test_tensor.cpu().item<int>();
        std::cout << "Extracted value: " << extracted_value << std::endl;
        
        std::cout << "=== ALL PYTORCH TESTS PASSED ===" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
}
