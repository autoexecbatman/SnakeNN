#include "snake_logic.h"
#include "advanced_snake_network.h"
#include <iostream>

int main() {
	std::cout << "🚀 ADVANCED SNAKE AI - NEXT GENERATION TRAINING 🚀" << std::endl;
	std::cout << "Architecture: CNN + LSTM + Multi-Head Decision Network" << std::endl;
	std::cout << "Focus: Collision Avoidance + Spatial Reasoning + Memory" << std::endl;
	std::cout << std::string(60, '=') << std::endl;

	try {
		// Create advanced trainer with 20x20 grid
		AdvancedSnakeTrainer trainer(20, 20);

		std::cout << "🧠 Network Architecture:" << std::endl;
		std::cout << "  • CNN Layers: 3 layers (4→32→64→128 channels)" << std::endl;
		std::cout << "  • LSTM Memory: 128 units for planning" << std::endl;
		std::cout << "  • Multi-Head Output: Safety + Food + Exploration" << std::endl;
		std::cout << "  • Grid Representation: 4-channel (empty/food/head/body)" << std::endl;

		std::cout << "🎯 Training Strategy:" << std::endl;
		std::cout << "  • Phase 1: Collision Avoidance Mastery" << std::endl;
		std::cout << "  • Phase 2: Food Seeking Optimization" << std::endl;
		std::cout << "  • Phase 3: Performance Optimization" << std::endl;

		std::cout << "⚡ Key Improvements Over Previous Models:" << std::endl;
		std::cout << "  ✓ Full spatial awareness (vs 8-feature limitation)" << std::endl;
		std::cout << "  ✓ Memory for multi-step planning (vs reactive only)" << std::endl;
		std::cout << "  ✓ Safety-first action selection (vs collision-prone)" << std::endl;
		std::cout << "  ✓ Multi-objective training (vs single reward)" << std::endl;
		std::cout << "  ✓ Progressive learning phases (vs one-size-fits-all)" << std::endl;

		char response;
		std::cout << "🚀 Ready to train the next-generation Snake AI? (y/n): ";
		std::cin >> response;

		if (response == 'y' || response == 'Y') {
			std::cout << "🔥 INITIATING ADVANCED TRAINING..." << std::endl;

			// Train with 6000 episodes (2000 per phase)
			trainer.train(6000, true);

			std::cout << "✨ Training complete! Check the results above." << std::endl;
			std::cout << "💡 Compare with comprehensive evaluator to see the improvement!" << std::endl;

		}
		else {

			std::cout << "📋 Training cancelled. Run when ready to train!" << std::endl;
		}

		std::cout << "📚 Want to test an existing model instead? Try:" << std::endl;
		std::cout << "  • Load previous advanced model" << std::endl;
		std::cout << "  • Quick evaluation run" << std::endl;
		std::cout << "  • Compare with old models using ComprehensiveEvaluator" << std::endl;

	}
	catch (const std::exception& e) {
		std::cerr << "❌ Error: " << e.what() << std::endl;
		std::cerr << "Make sure PyTorch C++ is properly installed and configured." << std::endl;
		return -1;
	}

	std::cout << "🎉 Advanced Snake AI session complete!" << std::endl;
	return 0;
}