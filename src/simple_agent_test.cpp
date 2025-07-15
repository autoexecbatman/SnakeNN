#include "snake_logic.h"
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cstdlib> // For rand() and srand()
#include <ctime>   // For time() to seed the random number generator

// Simple manual Q-table approach for comparison
class SimpleQAgent
{
public:
	SimpleQAgent()
	{
		srand(time(0)); // Seed the random number generator
		// Initialize Q-table with small random values
		for (int i = 0; i < 8; i++) {
			for (int j = 0; j < 4; j++) {
				q_table[i][j] = ((float)rand() / RAND_MAX) - 0.5f; // Use float division for better randomness
			}
		}
	}

	int getAction(const SnakeGame& game, float epsilon) {
		if ((rand() % 100) / 100.0f < epsilon) {
			return rand() % 4;
			// Random action
		}

		int state = getSimpleState(game);

		// Find best action for this state
		int best_action = 0;
		float best_q = q_table[state][0];

		for (int a = 1; a < 4; a++) {
			if (q_table[state][a] > best_q) {
				best_q = q_table[state][a];
				best_action = a;
			}
		}

		return best_action;
	}

	void updateQ(int state, int action, float reward, int next_state, bool done) {
		float next_q = 0.0f;
		if (!done) {
			// Find max Q-value for next state
			for (int a = 0; a < 4; a++) {
				if (q_table[next_state][a] > next_q) {
					next_q = q_table[next_state][a];
				}
			}
		}

		// Q-learning update
		float target = reward + 0.95f * next_q;
		q_table[state][action] += 0.1f * (target - q_table[state][action]);
	}

	int getSimpleState(const SnakeGame& game) {
		auto head = game.getSnakeBody()[0];
		auto food = game.getFoodPosition();

		int state = 0;

		// Food direction (3 bits)
		if (food.x > head.x) state |= 1;  // Food to the right
		if (food.y > head.y) state |= 2;  // Food below
		if (food.x < head.x) state |= 4;  // Food to the left
		// Note: food.y < head.y would be bit 3, but we only use 3 bits

		return state % 8;  // Ensure valid state index
	}

private:
	float q_table[8][4];  // 8 states, 4 actions

};

void testSimpleAgent() {
	std::cout << "=== TESTING SIMPLE Q-AGENT ===" << std::endl;

	SimpleQAgent agent;
	float epsilon = 1.0f;
	const float epsilon_decay = 0.995f;
	const float epsilon_min = 0.01f;

	int total_foods = 0;
	int total_episodes = 1000;

	for (int episode = 0; episode < total_episodes; episode++) {
		SnakeGame game;
		game.reset();

		int prev_state = -1;
		int prev_action = -1;

		for (int step = 0; step < 200; step++) {
			int current_state = agent.getSimpleState(game);
			int action = agent.getAction(game, epsilon);

			// Take action
			game.setDirection(static_cast<Direction>(action));
			bool game_continues = game.update();

			float reward = game.getReward();
			int next_state = agent.getSimpleState(game);

			// Update Q-table if we have previous experience
			if (prev_state >= 0) {
				agent.updateQ(prev_state, prev_action, reward, current_state, !game_continues);
			}

			prev_state = current_state;
			prev_action = action;

			if (!game_continues) break;
		}

		total_foods += game.getScore();

		// Decay epsilon
		if (epsilon > epsilon_min) {
			epsilon *= epsilon_decay;
		}

		if (episode % 100 == 0) {
			float avg_score = (float)total_foods / (episode + 1);
			std::cout << "Episode " << episode << ": Total foods = " << total_foods
				<< ", Avg score = " << avg_score << ", Epsilon = " << epsilon << std::endl;
		}
	}

	float final_avg = (float)total_foods / total_episodes;
	std::cout << "\nFinal Results:" << std::endl;
	std::cout << "Total foods: " << total_foods << std::endl;
	std::cout << "Average score: " << final_avg << std::endl;

	if (final_avg > 0.5f) {
		std::cout << "SUCCESS: Simple agent learned to find food!" << std::endl;
	}
	else {
		std::cout << "FAILURE: Even simple agent couldn't learn" << std::endl;
	}
}

int main() {
	std::cout << "=== SIMPLE AGENT COMPARISON TEST ===" << std::endl;

	try {
		testSimpleAgent();
	}
	catch (const std::exception& e) {
		std::cerr << "Error: " << e.what() << std::endl;
		return -1;
	}

	return 0;
}
