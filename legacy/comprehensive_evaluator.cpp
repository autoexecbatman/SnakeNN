#include "neural_network.h"
#include "snake_logic.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <iomanip>

// Comprehensive evaluation system for snake AI models
class ComprehensiveEvaluator {
public:
    struct GameResult {
        int score;
        int steps;
        bool completed_game;  // Reached theoretical maximum
        std::string death_cause;  // "wall", "self", "timeout", "completed"
        float efficiency;  // score/steps ratio
    };
    
    struct EvaluationMetrics {
        // Score Performance
        float average_score;
        float median_score;
        float max_score;
        std::vector<int> score_distribution;
        
        // Game Completion Tiers
        int basic_games;     // 1-10 foods
        int good_games;      // 11-25 foods  
        int excellent_games; // 26-50 foods
        int master_games;    // 51+ foods
        
        // Completion Analysis
        float avg_completion_percentage;
        int games_25_percent;  // Reached 25% completion
        int games_50_percent;  // Reached 50% completion
        int games_75_percent;  // Reached 75% completion
        
        // Survival & Safety
        float avg_survival_steps;
        float collision_avoidance_rate;
        int wall_deaths;
        int self_collision_deaths;
        int timeout_deaths;
        int completed_games;
        
        // Efficiency
        float avg_steps_per_food;
        float avg_efficiency;  // score/steps
        
        // Overall Rating
        float overall_score;   // Weighted combination
        std::string performance_tier;
    };
    
    ComprehensiveEvaluator() {
        // For 20x20 grid, theoretical maximum is ~398 foods
        theoretical_max_score = (SnakeGame::GRID_WIDTH * SnakeGame::GRID_HEIGHT) - 2;
        max_steps_per_game = theoretical_max_score * 3;  // Allow 3 steps per food
    }
    
    EvaluationMetrics evaluateModel(const std::string& model_path, int test_games = 500) {
        std::cout << "\n=== COMPREHENSIVE MODEL EVALUATION ===" << std::endl;
        std::cout << "Model: " << model_path << std::endl;
        std::cout << "Test Games: " << test_games << std::endl;
        std::cout << "Grid Size: " << SnakeGame::GRID_WIDTH << "x" << SnakeGame::GRID_HEIGHT << std::endl;
        std::cout << "Theoretical Max Score: " << theoretical_max_score << std::endl;
        std::cout << "\nRunning evaluation..." << std::endl;
        
        SnakeNeuralNetwork network(8, 64, 4);  // Match OptimizedTrainer architecture
        
        try {
            network.load(model_path);
            std::cout << "Model loaded successfully!" << std::endl;
        } catch (const std::exception& e) {
            std::cout << "Error loading model: " << e.what() << std::endl;
            return EvaluationMetrics{};
        }
        
        std::vector<GameResult> results;
        
        for (int game = 0; game < test_games; game++) {
            if (game % 50 == 0) {
                std::cout << "Progress: " << game << "/" << test_games << " games..." << std::endl;
            }
            
            GameResult result = playGame(network);
            results.push_back(result);
        }
        
        EvaluationMetrics metrics = calculateMetrics(results);
        displayResults(metrics, test_games);
        saveDetailedReport(model_path, metrics, results);
        
        return metrics;
    }
    
private:
    int theoretical_max_score;
    int max_steps_per_game;
    
    GameResult playGame(SnakeNeuralNetwork& network) {
        SnakeGame game;
        game.reset();
        
        GameResult result = {0, 0, false, "unknown", 0.0f};
        
        while (!game.isGameOver() && result.steps < max_steps_per_game) {
            auto state = getOptimizedState(game);
            auto action_tensor = network.getAction(state, 0.0f);  // Pure evaluation (no exploration)
            int action = static_cast<int>(action_tensor.cpu().item<int64_t>());
            
            game.setDirection(static_cast<Direction>(action));
            bool game_continues = game.update();
            result.steps++;
            
            if (!game_continues) {
                result.score = game.getScore();
                
                // Determine death cause
                auto head = game.getSnakeBody()[0];
                if (head.x < 0 || head.x >= SnakeGame::GRID_WIDTH || 
                    head.y < 0 || head.y >= SnakeGame::GRID_HEIGHT) {
                    result.death_cause = "wall";
                } else {
                    result.death_cause = "self";
                }
                break;
            }
        }
        
        // Handle other end conditions
        if (result.steps >= max_steps_per_game) {
            result.death_cause = "timeout";
            result.score = game.getScore();
        }
        
        if (result.score >= theoretical_max_score * 0.95f) {  // 95% completion = "completed"
            result.completed_game = true;
            result.death_cause = "completed";
        }
        
        result.efficiency = result.score > 0 ? (float)result.score / result.steps : 0.0f;
        
        return result;
    }
    
    std::vector<float> getOptimizedState(const SnakeGame& game) {
        // Same proven 8-feature state as OptimizedTrainer
        auto head = game.getSnakeBody()[0];
        auto food = game.getFoodPosition();
        
        std::vector<float> state(8);
        
        // Food direction (4 features)
        state[0] = (food.x > head.x) ? 1.0f : 0.0f;  // Food right
        state[1] = (food.x < head.x) ? 1.0f : 0.0f;  // Food left  
        state[2] = (food.y > head.y) ? 1.0f : 0.0f;  // Food down
        state[3] = (food.y < head.y) ? 1.0f : 0.0f;  // Food up
        
        // Immediate danger (4 features)
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
                          checkCollision(testPos, game.getSnakeBody()));
            state[4 + i] = danger ? 1.0f : 0.0f;
        }
        
        return state;
    }
    
    bool checkCollision(const Position& pos, const std::vector<Position>& snake) {
        for (const auto& segment : snake) {
            if (segment.x == pos.x && segment.y == pos.y) {
                return true;
            }
        }
        return false;
    }
    
    EvaluationMetrics calculateMetrics(const std::vector<GameResult>& results) {
        EvaluationMetrics metrics = {};
        
        if (results.empty()) return metrics;
        
        // Collect scores for analysis
        std::vector<int> scores;
        int total_score = 0;
        int total_steps = 0;
        float total_efficiency = 0.0f;
        
        for (const auto& result : results) {
            scores.push_back(result.score);
            total_score += result.score;
            total_steps += result.steps;
            total_efficiency += result.efficiency;
            
            // Completion percentage tracking
            float completion_percent = (float)result.score / theoretical_max_score * 100.0f;
            if (completion_percent >= 25.0f) metrics.games_25_percent++;
            if (completion_percent >= 50.0f) metrics.games_50_percent++;
            if (completion_percent >= 75.0f) metrics.games_75_percent++;
            
            // Score tier classification
            if (result.score >= 1 && result.score <= 10) metrics.basic_games++;
            else if (result.score >= 11 && result.score <= 25) metrics.good_games++;
            else if (result.score >= 26 && result.score <= 50) metrics.excellent_games++;
            else if (result.score >= 51) metrics.master_games++;
            
            // Death cause tracking
            if (result.death_cause == "wall") metrics.wall_deaths++;
            else if (result.death_cause == "self") metrics.self_collision_deaths++;
            else if (result.death_cause == "timeout") metrics.timeout_deaths++;
            else if (result.death_cause == "completed") metrics.completed_games++;
        }
        
        // Calculate basic statistics
        metrics.average_score = (float)total_score / results.size();
        metrics.avg_survival_steps = (float)total_steps / results.size();
        metrics.avg_efficiency = total_efficiency / results.size();
        
        // Sort scores for median and max
        std::sort(scores.begin(), scores.end());
        metrics.median_score = scores[scores.size() / 2];
        metrics.max_score = scores.back();
        
        // Completion analysis
        metrics.avg_completion_percentage = metrics.average_score / theoretical_max_score * 100.0f;
        
        // Collision avoidance rate (games that didn't die from collisions)
        int non_collision_deaths = metrics.timeout_deaths + metrics.completed_games;
        metrics.collision_avoidance_rate = (float)non_collision_deaths / results.size() * 100.0f;
        
        // Steps per food efficiency
        metrics.avg_steps_per_food = metrics.average_score > 0 ? metrics.avg_survival_steps / metrics.average_score : 0.0f;
        
        // Overall weighted score (0-100)
        float score_weight = (metrics.average_score / theoretical_max_score) * 40.0f;  // 40% weight
        float survival_weight = (metrics.avg_survival_steps / max_steps_per_game) * 20.0f;  // 20% weight
        float completion_weight = (metrics.avg_completion_percentage / 100.0f) * 30.0f;  // 30% weight
        float efficiency_weight = std::min(metrics.avg_efficiency * 1000.0f, 10.0f);  // 10% weight (capped)
        
        metrics.overall_score = score_weight + survival_weight + completion_weight + efficiency_weight;
        
        // Performance tier assignment
        if (metrics.overall_score >= 85.0f) metrics.performance_tier = "MASTER";
        else if (metrics.overall_score >= 70.0f) metrics.performance_tier = "EXCELLENT";
        else if (metrics.overall_score >= 55.0f) metrics.performance_tier = "GOOD";
        else if (metrics.overall_score >= 40.0f) metrics.performance_tier = "BASIC";
        else metrics.performance_tier = "LEARNING";
        
        return metrics;
    }
    
    void displayResults(const EvaluationMetrics& metrics, int total_games) {
        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "                 EVALUATION RESULTS" << std::endl;
        std::cout << std::string(60, '=') << std::endl;
        
        std::cout << std::fixed << std::setprecision(2);
        
        std::cout << "\n📊 SCORE PERFORMANCE:" << std::endl;
        std::cout << "  Average Score: " << metrics.average_score << " foods" << std::endl;
        std::cout << "  Median Score:  " << metrics.median_score << " foods" << std::endl;
        std::cout << "  Max Score:     " << metrics.max_score << " foods" << std::endl;
        std::cout << "  Completion:    " << metrics.avg_completion_percentage << "%" << std::endl;
        
        std::cout << "\n🎯 PERFORMANCE TIERS:" << std::endl;
        std::cout << "  Basic (1-10):     " << metrics.basic_games << " games (" << (float)metrics.basic_games/total_games*100 << "%)" << std::endl;
        std::cout << "  Good (11-25):     " << metrics.good_games << " games (" << (float)metrics.good_games/total_games*100 << "%)" << std::endl;
        std::cout << "  Excellent (26-50): " << metrics.excellent_games << " games (" << (float)metrics.excellent_games/total_games*100 << "%)" << std::endl;
        std::cout << "  Master (51+):     " << metrics.master_games << " games (" << (float)metrics.master_games/total_games*100 << "%)" << std::endl;
        
        std::cout << "\n🏆 COMPLETION MILESTONES:" << std::endl;
        std::cout << "  25%+ Completion: " << metrics.games_25_percent << " games" << std::endl;
        std::cout << "  50%+ Completion: " << metrics.games_50_percent << " games" << std::endl;
        std::cout << "  75%+ Completion: " << metrics.games_75_percent << " games" << std::endl;
        std::cout << "  Full Completion: " << metrics.completed_games << " games" << std::endl;
        
        std::cout << "\n🛡️ SURVIVAL & SAFETY:" << std::endl;
        std::cout << "  Avg Survival:    " << metrics.avg_survival_steps << " steps" << std::endl;
        std::cout << "  Collision Avoid: " << metrics.collision_avoidance_rate << "%" << std::endl;
        std::cout << "  Wall Deaths:     " << metrics.wall_deaths << std::endl;
        std::cout << "  Self Collisions: " << metrics.self_collision_deaths << std::endl;
        std::cout << "  Timeouts:        " << metrics.timeout_deaths << std::endl;
        
        std::cout << "\n⚡ EFFICIENCY:" << std::endl;
        std::cout << "  Steps/Food:      " << metrics.avg_steps_per_food << std::endl;
        std::cout << "  Score/Steps:     " << metrics.avg_efficiency << std::endl;
        
        std::cout << "\n🎖️ OVERALL RATING:" << std::endl;
        std::cout << "  Overall Score:   " << metrics.overall_score << "/100" << std::endl;
        std::cout << "  Performance Tier: " << metrics.performance_tier << std::endl;
        
        std::cout << "\n" << std::string(60, '=') << std::endl;
    }
    
    void saveDetailedReport(const std::string& model_path, const EvaluationMetrics& metrics, 
                           const std::vector<GameResult>& results) {
        std::string report_path = model_path + "_evaluation_report.txt";
        std::ofstream report(report_path);
        
        if (!report.is_open()) {
            std::cout << "Warning: Could not save detailed report" << std::endl;
            return;
        }
        
        report << "COMPREHENSIVE EVALUATION REPORT\n";
        report << "Model: " << model_path << "\n";
        report << "Generated: " << __DATE__ << " " << __TIME__ << "\n\n";
        
        // Summary metrics
        report << "SUMMARY METRICS:\n";
        report << "Average Score: " << metrics.average_score << "\n";
        report << "Max Score: " << metrics.max_score << "\n";
        report << "Overall Rating: " << metrics.overall_score << "/100 (" << metrics.performance_tier << ")\n\n";
        
        // Detailed game results (first 50 games as sample)
        report << "SAMPLE GAME RESULTS (first 50 games):\n";
        report << "Game\tScore\tSteps\tEfficiency\tDeath Cause\n";
        for (size_t i = 0; i < std::min((size_t)50, results.size()); i++) {
            const auto& result = results[i];
            report << (i+1) << "\t" << result.score << "\t" << result.steps << "\t" 
                  << result.efficiency << "\t" << result.death_cause << "\n";
        }
        
        report.close();
        std::cout << "\nDetailed report saved to: " << report_path << std::endl;
    }
};

int main(int argc, char* argv[]) {
    std::string model_path = "snake_optimized_98percent.bin";  // Default to best model
    int test_games = 500;
    
    if (argc > 1) {
        model_path = argv[1];
    }
    if (argc > 2) {
        test_games = std::atoi(argv[2]);
    }
    
    std::cout << "=== COMPREHENSIVE SNAKE AI EVALUATOR ===" << std::endl;
    
    try {
        ComprehensiveEvaluator evaluator;
        auto metrics = evaluator.evaluateModel(model_path, test_games);
        
        std::cout << "\nEvaluation complete! Check the detailed report for full analysis." << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}
