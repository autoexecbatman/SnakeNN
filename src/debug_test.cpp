#include <iostream>
#include <torch/torch.h>

int main() {
    std::cout << "Starting debug test..." << std::endl;
    std::cout.flush();
    
    std::cout << "Torch included successfully" << std::endl;
    std::cout.flush();
    
    try {
        std::cout << "Testing basic torch operations..." << std::endl;
        auto tensor = torch::zeros({2, 3});
        std::cout << "Basic tensor created successfully" << std::endl;
        
        std::cout << "Testing CUDA availability..." << std::endl;
        bool cuda_available = torch::cuda::is_available();
        std::cout << "CUDA available: " << cuda_available << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "Error: " << e.what() << std::endl;
    }
    
    std::cout << "Debug test completed" << std::endl;
    return 0;
}
