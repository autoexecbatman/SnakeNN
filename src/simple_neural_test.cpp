#include "neural_network.h"
#include "snake_logic.h"
#include <iostream>
#include <vector>
#include <random> // For std::mt19937, std::uniform_real_distribution
#include <chrono> // For seeding random number generator
#include <algorithm> // For std::max

// Include for torch::optim
#include <torch/optim/adam.h>

// Simplified neural network with better architecture
class SimpleNeuralAgent {
public:
    // Constructor: Initializes the neural networks and the optimizer
    // network: The main Q-network that learns to predict Q-values.
    // target_network: A copy of the main network used to generate stable Q-targets,
    //                 reducing instability during training.
    // optimizer: Adam optimizer for updating the main network's weights.
    SimpleNeuralAgent()
        : network(8, 32, 4), // Input: 8 features, Hidden: 32 neurons, Output: 4 actions (directions)
        target_network(8, 32, 4),
        optimizer(network.parameters(), torch::optim::AdamOptions(0.001)) { // Initialize Adam with learning rate 0.001
        // Copy initial weights from the main network to the target network
        // This ensures the target network starts with the same parameters as the main network.
        auto source_params = network.parameters();
        auto target_params = target_network.parameters();
        torch::NoGradGuard no_grad; // Disable gradient calculation during this copy operation
        for (size_t i = 0; i < source_params.size(); i++) {
            target_params[i].copy_(source_params[i]);
        }
    }

    // getSimpleState: Converts the current game state into a simplified feature vector
    //                 that the neural network can understand.
    // The state vector consists of 8 float values:
    // - 4 features for relative food position (binary: food right, left, down, up)
    // - 4 features for immediate danger in each direction (binary: danger up, down, left, right)
    std::vector<float> getSimpleState(const SnakeGame& game) {
        auto head = game.getSnakeBody()[0]; // Get the current position of the snake's head
        auto food = game.getFoodPosition(); // Get the current position of the food

        std::vector<float> state(8); // Initialize a vector to store the 8 state features

        // Features 0-3: Relative position of the food with respect to the snake's head
        state[0] = (food.x > head.x) ? 1.0f : 0.0f; // Is food to the right of the head?
        state[1] = (food.x < head.x) ? 1.0f : 0.0f; // Is food to the left of the head?
        state[2] = (food.y > head.y) ? 1.0f : 0.0f; // Is food below (down) the head?
        state[3] = (food.y < head.y) ? 1.0f : 0.0f; // Is food above (up) the head?

        // Features 4-7: Immediate danger in each of the four possible movement directions
        // (UP, DOWN, LEFT, RIGHT).
        // This checks if moving in a certain direction would immediately result in a collision
        // with a wall or the snake's own body.
        for (int i = 0; i < 4; i++) {
            Direction testDir = static_cast<Direction>(i); // Convert integer index to Direction enum
            Position testPos = head; // Start with the head's position for testing

            // Calculate the potential new position if the snake moves in 'testDir'
            switch (testDir) {
            case Direction::UP:    testPos.y--; break;
            case Direction::DOWN:  testPos.y++; break;
            case Direction::LEFT:  testPos.x--; break;
            case Direction::RIGHT: testPos.x++; break;
            }

            // Check for collision at 'testPos' with game boundaries or the snake's body
            bool danger = (testPos.x < 0 || testPos.x >= SnakeGame::GRID_WIDTH || // Wall collision (horizontal)
                testPos.y < 0 || testPos.y >= SnakeGame::GRID_HEIGHT || // Wall collision (vertical)
                checkCollision(testPos, game.getSnakeBody()));         // Body collision
            state[4 + i] = danger ? 1.0f : 0.0f; // Set feature to 1.0f if danger, 0.0f otherwise
        }

        return state;
    }

    // getAction: Decides the next action (direction) for the snake.
    // It uses an epsilon-greedy strategy:
    // - With probability 'epsilon', it chooses a random action (exploration).
    // - With probability (1 - epsilon), it chooses the action with the highest predicted Q-value
    //   from the neural network (exploitation).
    int getAction(const std::vector<float>& state, float epsilon) {
        // Initialize random number generators for robust randomness
        static std::random_device rd; // Used to obtain a seed for the random number engine
        static std::mt19937 gen(rd()); // Mersenne Twister engine seeded by rd
        static std::uniform_real_distribution<> dis(0.0, 1.0); // Distribution for epsilon check (0.0 to 1.0)

        if (dis(gen) < epsilon) {
            // Exploration: Choose a random action (0:UP, 1:DOWN, 2:LEFT, 3:RIGHT)
            static std::uniform_int_distribution<> action_dis(0, 3); // Distribution for random action (0 to 3)
            return action_dis(gen);
        }
        else {
            // Exploitation: Choose the action with the highest Q-value predicted by the network
            // Convert the state vector to a PyTorch tensor.
            // .unsqueeze(0) adds a batch dimension (e.g., from [8] to [1, 8]) as networks expect batched inputs.
            auto state_tensor = torch::tensor(state).to(torch::kFloat).unsqueeze(0);

            // Get the action from the main network. The second argument (0.0f) ensures no
            // internal exploration if the network's getAction method supports it.
            auto action_tensor = network.getAction(state_tensor, 0.0f);
            // Convert the resulting tensor (which contains the chosen action index) to a C++ int.
            return static_cast<int>(action_tensor.cpu().item<int64_t>());
        }
    }

    // trainStep: Performs one step of Q-learning training for the agent.
    // It updates the main Q-network's weights based on a single experience tuple:
    // (state, action, reward, next_state, done).
    void trainStep(const std::vector<float>& state, int action, float reward,
        const std::vector<float>& next_state, bool done) {
        // Convert state and next_state vectors to PyTorch tensors, adding a batch dimension.
        auto state_tensor = torch::tensor(state).to(torch::kFloat).unsqueeze(0);
        auto next_state_tensor = torch::tensor(next_state).to(torch::kFloat).unsqueeze(0);

        // Get the Q-values predicted by the main network for the current state.
        auto current_q_values = network.forward(state_tensor);

        // Get the Q-values predicted by the target network for the next state.
        // We use 'torch::NoGradGuard' here because we don't need to compute gradients
        // for the target network's predictions; its weights are updated separately.
        torch::Tensor next_q_values;
        {
            torch::NoGradGuard no_grad;
            next_q_values = target_network.forward(next_state_tensor);
        }

        // Calculate the target Q-value for the action taken.
        // This is the core of the Q-learning update rule.
        float target_q = reward; // Start with the immediate reward
        if (!done) { // If the episode is not over (i.e., the game continues)
            // Add the discounted maximum Q-value from the next state.
            // 'std::get<0>(next_q_values.max(1))' gets the maximum Q-value along dimension 1 (actions).
            // '.cpu().item<float>()' extracts the float value from the tensor.
            target_q += 0.95f * std::get<0>(next_q_values.max(1)).cpu().item<float>(); // Discount factor = 0.95
        }

        // Create the target tensor for loss calculation.
        // We clone 'current_q_values' and then modify only the Q-value for the 'action' taken.
        // The Q-values for other actions remain unchanged (as per DQN's loss formulation).
        auto target_q_values = current_q_values.clone();
        target_q_values[0][action] = target_q; // Set the calculated target Q-value for the taken action

        // Calculate the Mean Squared Error (MSE) loss between the current Q-values (predictions)
        // and the target Q-values (what the network 'should' predict).
        auto loss = torch::mse_loss(current_q_values, target_q_values);

        // Perform backpropagation and update network weights using the Adam optimizer.
        optimizer.zero_grad(); // Clear any previously computed gradients
        loss.backward();       // Compute gradients of the loss with respect to network parameters
        optimizer.step();      // Update the network's weights using the computed gradients
    }

    // updateTargetNetwork: Copies the weights from the main network to the target network.
    // This is typically done periodically to provide stable targets for the Q-learning updates.
    void updateTargetNetwork() {
        auto source_params = network.parameters();
        auto target_params = target_network.parameters();
        torch::NoGradGuard no_grad; // Disable gradient calculation during the copy
        for (size_t i = 0; i < source_params.size(); i++) {
            target_params[i].copy_(source_params[i]);
        }
    }

private:
    SnakeNeuralNetwork network;       // The main neural network (Q-network)
    SnakeNeuralNetwork target_network; // The target neural network for stable Q-value targets
    torch::optim::Adam optimizer;     // The Adam optimizer for training the main network

    // checkCollision: A private helper function to determine if a given position
    //                 collides with any segment of the snake's body.
    bool checkCollision(const Position& pos, const std::vector<Position>& snake) {
        // Iterate through each segment (Position) in the snake's body vector
        for (const auto& segment : snake) {
            // If the given position matches any segment's coordinates, a collision is detected
            if (segment.x == pos.x && segment.y == pos.y) {
                return true; // Collision occurred
            }
        }
        return false; // No collision with the snake's body
    }
};

// testSimpleNeuralAgent: This function simulates the training and testing of the
//                        SimpleNeuralAgent within the Snake game environment.
void testSimpleNeuralAgent() {
    std::cout << "=== TESTING SIMPLE NEURAL AGENT ===" << std::endl;

    SimpleNeuralAgent agent; // Create an instance of our neural agent
    float epsilon = 1.0f; // Initial exploration rate (100% exploration at the start)
    const float epsilon_decay = 0.998f; // Rate at which epsilon decreases over episodes
    const float epsilon_min = 0.05f;    // Minimum exploration rate to ensure some continued exploration

    int total_foods = 0; // Tracks the total number of foods eaten across all episodes
    int total_episodes = 1000; // Total number of game episodes to simulate for training

    // Loop through each training episode
    for (int episode = 0; episode < total_episodes; episode++) {
        SnakeGame game; // Create a new Snake game instance for each episode
        game.reset();   // Reset the game to its initial state (new snake, new food)

        // Game loop for a single episode (limited to 200 steps to prevent infinite games)
        for (int step = 0; step < 200; step++) {
            // 1. Get the current state (S_t)
            auto current_state = agent.getSimpleState(game);

            // 2. Choose an action (A_t) based on the current state and epsilon-greedy strategy
            int action = agent.getAction(current_state, epsilon);

            // 3. Take the chosen action in the game environment
            game.setDirection(static_cast<Direction>(action));
            bool game_continues = game.update(); // Update game state, returns false if game over

            // 4. Observe the reward (R_t) and the new state (S_t+1)
            float reward = game.getReward();
            auto next_state = agent.getSimpleState(game);
            bool done = !game_continues; // 'done' is true if the game ended

            // 5. Train the agent using the observed experience tuple (S_t, A_t, R_t, S_t+1, D_t)
            // Error E0312: "no suitable user-defined conversion from "at::Tensor" to "const std::vector<float, std::allocator<float>>" exists"
            // This error typically indicates a type mismatch. While 'current_state' and 'next_state'
            // are explicitly 'std::vector<float>' as returned by 'getSimpleState', this error often
            // points to an issue within the 'SnakeNeuralNetwork' class definition (in neural_network.h)
            // where a 'torch::Tensor' might be implicitly or incorrectly converted to 'std::vector<float>'.
            // The problem is that 'network.forward()' (and potentially 'target_network.forward()')
            // in 'neural_network.h' is likely defined to accept 'std::vector<float>' but is receiving
            // a 'torch::Tensor' (like 'state_tensor' or 'next_state_tensor').
            agent.trainStep(current_state, action, reward, next_state, done);

            // If the game ended (snake hit wall/body or ate all food), break the inner loop
            if (!game_continues) break;
        }

        total_foods += game.getScore(); // Add the score from the current episode to total

        // Periodically update the target network's weights from the main network.
        // This helps stabilize training by providing fixed targets for a period.
        if (episode % 50 == 0) {
            agent.updateTargetNetwork();
        }

        // Decay epsilon: gradually reduce the exploration rate as training progresses.
        // This makes the agent exploit its learned knowledge more often.
        if (epsilon > epsilon_min) {
            epsilon *= epsilon_decay;
        }

        // Print progress report every 100 episodes
        if (episode % 100 == 0) {
            float avg_score = (float)total_foods / (episode + 1);
            std::cout << "Episode " << episode << ": Total foods = " << total_foods
                << ", Avg score = " << avg_score << ", Epsilon = " << epsilon << std::endl;
        }
    }

    // Display final training results after all episodes are complete
    float final_avg = (float)total_foods / total_episodes;
    std::cout << "\nFinal Results:" << std::endl;
    std::cout << "Total foods: " << total_foods << std::endl;
    std::cout << "Average score: " << final_avg << std::endl;

    // Provide a simple indication of learning success
    if (final_avg > 1.0f) {
        std::cout << "SUCCESS: Simple neural agent learned!" << std::endl;
    }
    else {
        std::cout << "PARTIAL: Some learning but not great" << std::endl;
    }
}

// main function: The entry point of the program.
int main() {
    std::cout << "=== SIMPLE NEURAL AGENT TEST ===" << std::endl;

    // Use a try-catch block to gracefully handle any exceptions (e.g., from PyTorch operations)
    try {
        testSimpleNeuralAgent();
    }
    catch (const std::exception& e) {
        // Print any error messages to the standard error stream
        std::cerr << "Error: " << e.what() << std::endl;
        return -1; // Return a non-zero exit code to indicate an error
    }

    return 0; // Return 0 for successful execution
}
