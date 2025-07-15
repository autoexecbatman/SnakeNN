#include <torch/torch.h>
#include <iostream>

int main() {
    std::cout << "=== Minimal PyTorch Test ===" << std::endl;
    
    try {
        // Just test basic tensor creation
        std::cout << "Creating tensor..." << std::endl;
        auto tensor = torch::ones({2, 2});
        std::cout << "Tensor created successfully" << std::endl;
        
        // Test our key fix - value extraction
        std::cout << "Testing value extraction..." << std::endl;
        auto scalar = torch::tensor(42);
        auto value = scalar.cpu().item<int>();
        std::cout << "Value extracted: " << value << std::endl;
        
        std::cout << "=== SUCCESS - PYTORCH WORKING ===" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
}
