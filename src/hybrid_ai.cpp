#include "neural_network.h"
#include "snake_logic.h"
#include <iostream>
#include <queue>
#include <set>
#include <algorithm>

// Hybrid AI: Neural network + pathfinding + space management
class HybridSnakeAI {
public:
    HybridSnakeAI() : network(8, 64, 4) {}
    
    bool loadModel(const std::string& model_path) {
        try {
            network.load(model_path);
            return true;
        } catch (const std::exception& e) {
            std::cout << "Failed to load model: " << e.what() << std::endl;
            return false;
        }
    }
    
    void testHybridApproach(int games = 500) {
        std::cout << "=== HYBRID AI TESTING ===" << std::endl;
        std::cout << "Neural Network + Pathfinding + Space Management" << std::endl;
        
        int total_score = 0;
        int high_scores = 0;
        int winning_games = 0;
        int max_score = 0;
        
        for (int game = 0; game < games; game++) {
            SnakeGame snake_game;
            snake_game.reset();
            
            int steps = 0;
            while (!snake_game.isGameOver() && steps < 3000) { // Very long games
                Direction best_move = getBestMove(snake_game);
                snake_game.setDirection(best_move);
                snake_game.update();
                steps++;
            }
            
            int score = snake_game.getScore();
            total_score += score;
            if (score >= 10) high_scores++;
            if (score >= 15) winning_games++;
            if (score > max_score) max_score = score;
            
            if (game % 50 == 0 && game > 0) {
                float current_avg = (float)total_score / (game + 1);
                std::cout << "Progress: " << game << " games | Avg: " << current_avg 
                         << " | Max: " << max_score << std::endl;
            }
        }
        
        float avg_score = (float)total_score / games;
        float high_rate = (float)high_scores / games * 100.0f;
        float win_rate = (float)winning_games / games * 100.0f;
        
        std::cout << std::endl << "=== HYBRID AI RESULTS ===" << std::endl;
        std::cout << "*** AVERAGE SCORE: " << avg_score << " foods ***" << std::endl;
        std::cout << "*** HIGH SCORES (10+): " << high_rate << "% ***" << std::endl;
        std::cout << "*** WINNING GAMES (15+): " << win_rate << "% ***" << std::endl;
        std::cout << "*** MAXIMUM SCORE: " << max_score << " foods ***" << std::endl;
        
        if (win_rate >= 60.0f) {
            std::cout << "\\n*** STRONG AI! 60%+ winning games! ***" << std::endl;
        } else if (win_rate >= 40.0f) {
            std::cout << "\\n*** GOOD AI! 40%+ winning games! ***" << std::endl;
        } else {
            std::cout << "\\n*** AI needs more improvement ***" << std::endl;
        }
    }
    
private:
    SnakeNeuralNetwork network;
    
    Direction getBestMove(const SnakeGame& game) {
        auto head = game.getSnakeBody()[0];
        auto food = game.getFoodPosition();
        auto snake = game.getSnakeBody();
        
        // 1. Get neural network suggestion
        auto nn_state = getState(game);
        auto action_tensor = network.getAction(nn_state, 0.0f);
        Direction nn_choice = static_cast<Direction>(action_tensor.cpu().item<int64_t>());
        
        // 2. Find safe path to food
        Direction path_choice = findPathToFood(head, food, snake);
        
        // 3. Space management override
        Direction space_choice = maintainSpace(head, snake);
        
        // Decision hierarchy:
        // 1. If space is critically low, prioritize space management
        float space_ratio = calculateSpaceRatio(head, snake);
        if (space_ratio < 0.3f) {
            return space_choice;
        }
        
        // 2. If path to food is safe and available, use it
        if (path_choice != Direction::UP && isSafeMove(head, path_choice, snake)) {
            // But check if it maintains good space
            Position test_pos = getNextPosition(head, path_choice);
            float future_space = calculateSpaceRatio(test_pos, snake);
            if (future_space > 0.2f) {
                return path_choice;
            }
        }
        
        // 3. Use neural network if its choice is safe
        if (isSafeMove(head, nn_choice, snake)) {
            return nn_choice;
        }
        
        // 4. Emergency: find any safe move
        for (int dir = 0; dir < 4; dir++) {
            Direction test_dir = static_cast<Direction>(dir);
            if (isSafeMove(head, test_dir, snake)) {
                return test_dir;
            }
        }
        
        return Direction::UP; // Should never reach here
    }
    
    Direction findPathToFood(const Position& head, const Position& food, const std::vector<Position>& snake) {
        // Simple A* pathfinding
        std::priority_queue<PathNode> open_set;
        std::set<std::pair<int, int>> closed_set;
        
        open_set.push({head, 0, manhattanDistance(head, food), Direction::UP});
        
        while (!open_set.empty()) {
            PathNode current = open_set.top();
            open_set.pop();
            
            if (current.pos.x == food.x && current.pos.y == food.y) {
                return current.first_direction;
            }
            
            if (closed_set.count({current.pos.x, current.pos.y})) {
                continue;
            }
            closed_set.insert({current.pos.x, current.pos.y});
            
            for (int dir = 0; dir < 4; dir++) {
                Direction direction = static_cast<Direction>(dir);
                Position next_pos = getNextPosition(current.pos, direction);
                
                if (isValidPosition(next_pos) && !isSnakePosition(next_pos, snake) &&
                    !closed_set.count({next_pos.x, next_pos.y})) {
                    
                    Direction first_dir = (current.cost == 0) ? direction : current.first_direction;
                    int new_cost = current.cost + 1;
                    int heuristic = manhattanDistance(next_pos, food);
                    
                    open_set.push({next_pos, new_cost, heuristic, first_dir});
                }
            }
        }
        
        return Direction::UP; // No path found
    }
    
    Direction maintainSpace(const Position& head, const std::vector<Position>& snake) {
        Direction best_dir = Direction::UP;
        float best_space = 0.0f;
        
        for (int dir = 0; dir < 4; dir++) {
            Direction direction = static_cast<Direction>(dir);
            if (isSafeMove(head, direction, snake)) {
                Position next_pos = getNextPosition(head, direction);
                float space = calculateSpaceRatio(next_pos, snake);
                
                if (space > best_space) {
                    best_space = space;
                    best_dir = direction;
                }
            }
        }
        
        return best_dir;
    }
    
    float calculateSpaceRatio(const Position& pos, const std::vector<Position>& snake) {
        // Flood fill to count accessible spaces
        std::queue<Position> queue;
        std::set<std::pair<int, int>> visited;
        
        queue.push(pos);
        visited.insert({pos.x, pos.y});
        
        int accessible_spaces = 0;
        while (!queue.empty()) {
            Position current = queue.front();
            queue.pop();
            accessible_spaces++;
            
            for (int dir = 0; dir < 4; dir++) {
                Position next = getNextPosition(current, static_cast<Direction>(dir));
                if (isValidPosition(next) && !isSnakePosition(next, snake) &&
                    !visited.count({next.x, next.y})) {
                    visited.insert({next.x, next.y});
                    queue.push(next);
                }
            }
        }
        
        int total_spaces = SnakeGame::GRID_WIDTH * SnakeGame::GRID_HEIGHT - snake.size();
        return (float)accessible_spaces / total_spaces;
    }
    
    struct PathNode {
        Position pos;
        int cost;
        int heuristic;
        Direction first_direction;
        
        bool operator<(const PathNode& other) const {
            return (cost + heuristic) > (other.cost + other.heuristic);
        }
    };
    
    bool isSafeMove(const Position& head, Direction dir, const std::vector<Position>& snake) {
        Position next_pos = getNextPosition(head, dir);
        return isValidPosition(next_pos) && !isSnakePosition(next_pos, snake);
    }
    
    Position getNextPosition(const Position& pos, Direction dir) {
        Position next = pos;
        switch (dir) {
            case Direction::UP: next.y--; break;
            case Direction::DOWN: next.y++; break;
            case Direction::LEFT: next.x--; break;
            case Direction::RIGHT: next.x++; break;
        }
        return next;
    }
    
    bool isValidPosition(const Position& pos) {
        return pos.x >= 0 && pos.x < SnakeGame::GRID_WIDTH &&
               pos.y >= 0 && pos.y < SnakeGame::GRID_HEIGHT;
    }
    
    bool isSnakePosition(const Position& pos, const std::vector<Position>& snake) {
        for (const auto& segment : snake) {
            if (segment.x == pos.x && segment.y == pos.y) {
                return true;
            }
        }
        return false;
    }
    
    int manhattanDistance(const Position& a, const Position& b) {
        return abs(a.x - b.x) + abs(a.y - b.y);
    }
    
    std::vector<float> getState(const SnakeGame& game) {
        auto head = game.getSnakeBody()[0];
        auto food = game.getFoodPosition();
        
        std::vector<float> state(8);
        
        state[0] = (food.x > head.x) ? 1.0f : 0.0f;
        state[1] = (food.x < head.x) ? 1.0f : 0.0f;
        state[2] = (food.y > head.y) ? 1.0f : 0.0f;
        state[3] = (food.y < head.y) ? 1.0f : 0.0f;
        
        for (int i = 0; i < 4; i++) {
            Direction testDir = static_cast<Direction>(i);
            Position testPos = head;
            
            switch (testDir) {
                case Direction::UP: testPos.y--; break;
                case Direction::DOWN: testPos.y++; break;
                case Direction::LEFT: testPos.x--; break;
                case Direction::RIGHT: testPos.x++; break;
            }
            
            bool danger = (testPos.x < 0 || testPos.x >= SnakeGame::GRID_WIDTH ||
                          testPos.y < 0 || testPos.y >= SnakeGame::GRID_HEIGHT ||
                          isSnakePosition(testPos, game.getSnakeBody()));
            state[4 + i] = danger ? 1.0f : 0.0f;
        }
        
        return state;
    }
};

int main() {
    std::cout << "=== HYBRID SNAKE AI ===" << std::endl;
    std::cout << "Combines neural network with pathfinding and space management" << std::endl;
    std::cout << std::endl;
    
    try {
        HybridSnakeAI hybrid_ai;
        
        if (hybrid_ai.loadModel("snake_extended_final_97percent.bin")) {
            hybrid_ai.testHybridApproach(500);
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}
