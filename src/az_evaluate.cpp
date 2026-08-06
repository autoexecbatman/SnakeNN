#include <torch/torch.h>
#include "az_network.h"
#include "mcts.h"
#include "network_evaluator.h"
#include "snake_env.h"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <string>
#include <vector>

// Scores a saved network on held-out seeds.
//
// Separate from the trainer on purpose. Self-play deliberately plays badly - it
// injects Dirichlet noise at the root and samples early moves at temperature so
// that the data covers more than one line. A number taken from those games
// measures the exploration policy, not the agent. Here the noise is off and the
// move is the visit-count argmax, which is what the agent would actually do.
//
// The metric is win rate: games in which the snake fills the board inside the
// step limit. Average score is reported next to it and must not be read as
// partial credit toward a win.

namespace {

struct Settings {
    std::string checkpoint;
    int board = 6;
    int games = 64;
    int simulations = 200;
    int step_limit = 0;
    int channels = 64;
    int blocks = 4;
    unsigned int seed = 900000;   // held out from training's seed range
    int batch = 64;
};

Settings parseArguments(int argc, char** argv) {
    Settings settings;
    for (int index = 1; index + 1 < argc; index += 2) {
        std::string flag = argv[index];
        std::string value = argv[index + 1];
        if (flag == "--checkpoint") { settings.checkpoint = value; }
        else if (flag == "--board") { settings.board = std::stoi(value); }
        else if (flag == "--games") { settings.games = std::stoi(value); }
        else if (flag == "--simulations") { settings.simulations = std::stoi(value); }
        else if (flag == "--step-limit") { settings.step_limit = std::stoi(value); }
        else if (flag == "--channels") { settings.channels = std::stoi(value); }
        else if (flag == "--blocks") { settings.blocks = std::stoi(value); }
        else if (flag == "--seed") { settings.seed = (unsigned int)std::stoul(value); }
        else if (flag == "--batch") { settings.batch = std::stoi(value); }
        else { std::cerr << "unknown flag: " << flag << std::endl; }
    }
    if (settings.step_limit == 0) {
        settings.step_limit = 12 * settings.board * settings.board;
    }
    return settings;
}

}  // namespace

int main(int argc, char** argv) {
    Settings settings = parseArguments(argc, argv);
    if (settings.checkpoint.empty()) {
        std::cerr << "usage: --checkpoint <file> [--board N] [--games N] [--simulations N]"
                  << std::endl;
        return 2;
    }

    const bool cuda = torch::cuda::is_available();
    torch::Device device = cuda ? torch::Device(torch::kCUDA) : torch::Device(torch::kCPU);

    AlphaZeroNet network(settings.board, settings.board, settings.channels, settings.blocks);
    try {
        torch::load(network, settings.checkpoint);
    } catch (const std::exception& error) {
        std::cerr << "could not load " << settings.checkpoint << ": " << error.what() << std::endl;
        return 1;
    }
    network->to(device);
    network->eval();

    std::cout << "=== Evaluation ===" << std::endl;
    std::cout << settings.checkpoint << " on " << settings.board << "x" << settings.board
              << ", " << settings.games << " games, " << settings.simulations
              << " simulations, step limit " << settings.step_limit << std::endl;
    std::cout << "seeds " << settings.seed << ".." << (settings.seed + settings.games - 1)
              << " (held out), greedy, no root noise" << std::endl << std::endl;

    NetworkEvaluator evaluator(network, device);

    MonteCarloSearch::Config search_config;
    search_config.simulations = settings.simulations;
    search_config.exploration = 0.5f;
    search_config.discount = 0.98f;
    search_config.root_noise_fraction = 0.0f;
    search_config.root_noise_alpha = 0.3f;
    search_config.seed = settings.seed;
    MonteCarloSearch search(evaluator, search_config);

    const int foods_to_win = settings.board * settings.board - 1;
    int wins = 0;
    int timeouts = 0;
    int deaths = 0;
    long long total_score = 0;
    long long total_steps = 0;
    int best_score = 0;

    auto started = std::chrono::high_resolution_clock::now();

    for (int start = 0; start < settings.games; start += settings.batch) {
        const int count = std::min(settings.batch, settings.games - start);
        std::vector<SnakeEnv> games;
        games.reserve(count);
        for (int index = 0; index < count; index++) {
            games.emplace_back(settings.board, settings.board,
                               settings.seed + start + index);
        }
        std::vector<bool> timed_out(count, false);

        while (true) {
            std::vector<int> live;
            std::vector<const SnakeEnv*> roots;
            for (int index = 0; index < count; index++) {
                if (games[index].done()) {
                    continue;
                }
                if (games[index].steps() >= settings.step_limit) {
                    timed_out[index] = true;
                    continue;
                }
                live.push_back(index);
                roots.push_back(&games[index]);
            }
            if (live.empty()) {
                break;
            }

            std::vector<MonteCarloSearch::Result> results = search.search(roots);
            for (size_t position = 0; position < live.size(); position++) {
                games[live[position]].step(results[position].best_action);
            }
        }

        for (int index = 0; index < count; index++) {
            const SnakeEnv& game = games[index];
            total_score += game.score();
            total_steps += game.steps();
            best_score = std::max(best_score, game.score());
            if (game.won()) {
                wins++;
            } else if (timed_out[index]) {
                timeouts++;
            } else {
                deaths++;
            }
        }

        std::cout << "  " << (start + count) << "/" << settings.games << " games, wins " << wins
                  << std::endl;
    }

    double seconds =
        std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - started).count();

    std::cout << std::endl << "Wins:    " << wins << "/" << settings.games << "  ("
              << (100.0 * wins / settings.games) << "%)" << std::endl;
    std::cout << "Score:   mean " << ((double)total_score / settings.games) << ", best "
              << best_score << " of " << foods_to_win << std::endl;
    std::cout << "Steps:   mean " << ((double)total_steps / settings.games) << std::endl;
    std::cout << "Endings: " << wins << " won, " << deaths << " died, " << timeouts
              << " timed out" << std::endl;
    std::cout << "Took " << seconds << "s, " << evaluator.evaluations() << " evaluations"
              << std::endl;

    return wins == settings.games ? 0 : 1;
}
