#include <iostream>
#include <chrono>
#include "snake_logic.h" // Assuming snake_logic.h contains SnakeGame, Direction, etc.
#include "neural_network.h" // Assuming neural_network.h contains SnakeNeuralNetwork

void testSimpleQLearning() {
    std::cout << "=== SIMPLE Q-LEARNING TEST ===" << std::endl;

    SnakeGame game;
    SnakeNeuralNetwork network(14, 64, 4);  // Smaller network for faster learning

    float epsilon = 1.0f;
    const float epsilon_decay = 0.99f;
    const float epsilon_min = 0.1f;

    int successful_foods = 0;
    int total_tests = 100;

    for (int test = 0; test < total_tests; ++test) {
        game.reset();

        for (int step = 0; step < 200; ++step) {  // Max 200 steps per test
            auto state = game.getGameState();

            auto action_tensor = network.getAction(state, epsilon);
            int action = static_cast<int>(action_tensor.cpu().item<int64_t>());

            Direction dir = static_cast<Direction>(action);
            game.setDirection(dir);

            if (!game.update()) break;  // Game over

            if (game.getScore() > 0) {
                successful_foods++;
                std::cout << "Food found in test " << test << " at step " << step << std::endl;
                break;
            }

            if (epsilon > epsilon_min) {
                epsilon *= epsilon_decay;
            }
        }

        if (test % 10 == 0) {
            std::cout << "Test " << test << ": " << successful_foods << " foods found so far, epsilon: " << epsilon << std::endl;
        }
    }

    std::cout << "\nRESULTS:" << std::endl;
    std::cout << "Total foods found: " << successful_foods << "/" << total_tests << std::endl;
    std::cout << "Success rate: " << (100.0f * successful_foods / total_tests) << "%" << std::endl;
}

void testRewardFunction() {
    std::cout << "\n=== REWARD FUNCTION TEST ===" << std::endl;

    SnakeGame game;
    game.reset();

    auto food_pos = game.getFoodPosition();
    auto snake_pos = game.getSnakeBody()[0];

    std::cout << "Snake at (" << snake_pos.x << ", " << snake_pos.y << ")" << std::endl;
    std::cout << "Food at (" << food_pos.x << ", " << food_pos.y << ")" << std::endl;

    for (int i = 0; i < 5; ++i) {
        game.setDirection(Direction::RIGHT);
        game.update();

        float reward = game.getReward();
        auto new_pos = game.getSnakeBody()[0];

        std::cout << "Step " << i << ": Position (" << new_pos.x << ", " << new_pos.y
            << "), Reward: " << reward << ", Score: " << game.getScore() << std::endl;

        if (game.isGameOver()) {
            std::cout << "Game over!" << std::endl;
            break;
        }
    }
}

void testNetworkOutput() {
    std::cout << "\n=== NETWORK OUTPUT TEST ===" << std::endl;

    SnakeNeuralNetwork network(14, 64, 4);
    SnakeGame game;
    game.reset();

    auto state = game.getGameState();

    std::cout << "State vector (" << state.size() << " elements):" << std::endl;
    for (size_t i = 0; i < state.size(); ++i) {
        std::cout << "  [" << i << "] = " << state[i] << std::endl;
    }

    auto state_tensor = torch::zeros({ 1, static_cast<int>(state.size()) }, torch::kFloat);
    for (size_t i = 0; i < state.size(); ++i) {
        state_tensor[0][i] = state[i];
    }

    auto q_values = network.forward(state_tensor);
    std::cout << "\nQ-values: " << q_values << std::endl;

    for (float eps = 0.0f; eps <= 1.0f; eps += 0.2f) {
        auto action = network.getAction(state, eps);
        std::cout << "Action with epsilon " << eps << ": " << action.cpu().item<int64_t>() << std::endl;
    }
}

int main() {
    std::cout << "=== SNAKE AI DIAGNOSTIC TESTS ===" << std::endl;

    try {
        testRewardFunction();
        testNetworkOutput();
        testSimpleQLearning();
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    std::cout << "\n=== TESTS COMPLETE ===" << std::endl;
    return 0;
}
