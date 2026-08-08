#include <torch/torch.h>
#include "az_network.h"
#include "mcts.h"
#include "network_evaluator.h"
#include "seed_policy.h"
#include "snake_env.h"
#include <algorithm>
#include <chrono>
#include <format>
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

namespace
{

struct Settings
{
    std::string checkpoint;
    int board = 6;
    int games = 64;
    int simulations = 200;
    int step_limit = 0;
    int channels = 64;
    int blocks = 4;
    // An offset within the reserved evaluation range, not an absolute seed. The
    // old default was the absolute 900000 and it was not held out at all - see
    // seed_policy.h.
    unsigned int seed_offset = 0;
    int batch = 64;
};

Settings parseArguments(int argc, char** argv)
{
    Settings settings;
    for (int index = 1; index + 1 < argc; index += 2)
    {
        std::string flag = argv[index];
        std::string value = argv[index + 1];
        if (flag == "--checkpoint")
        {
            settings.checkpoint = value;
        }
        else if (flag == "--board")
        {
            settings.board = std::stoi(value);
        }
        else if (flag == "--games")
        {
            settings.games = std::stoi(value);
        }
        else if (flag == "--simulations")
        {
            settings.simulations = std::stoi(value);
        }
        else if (flag == "--step-limit")
        {
            settings.step_limit = std::stoi(value);
        }
        else if (flag == "--channels")
        {
            settings.channels = std::stoi(value);
        }
        else if (flag == "--blocks")
        {
            settings.blocks = std::stoi(value);
        }
        else if (flag == "--seed")
        {
            settings.seed_offset = static_cast<unsigned int>(std::stoul(value));
        }
        else if (flag == "--batch")
        {
            settings.batch = std::stoi(value);
        }
        else
        {
            std::cerr << "unknown flag: " << flag << std::endl;
        }
    }
    if (settings.step_limit == 0)
    {
        settings.step_limit = 12 * settings.board * settings.board;
    }
    return settings;
}

}  // namespace

int main(int argc, char** argv)
{
    Settings settings = parseArguments(argc, argv);
    if (settings.checkpoint.empty())
    {
        std::cerr << "usage: --checkpoint <file> [--board N] [--games N] [--simulations N]"
                  << std::endl;
        return 2;
    }

    const bool cuda = torch::cuda::is_available();
    torch::Device device = cuda ? torch::Device(torch::kCUDA) : torch::Device(torch::kCPU);

    AlphaZeroNet network(settings.board, settings.board, settings.channels, settings.blocks);
    try
    {
        torch::load(network, settings.checkpoint);
    }
    catch (const std::exception& error)
    {
        std::cerr << "could not load " << settings.checkpoint << ": " << error.what() << std::endl;
        return 1;
    }
    network->to(device);
    network->eval();

    std::cout << "=== Evaluation ===" << std::endl;
    std::cout << std::format("{} on {}x{}, {} games, {} simulations, step limit {}\n",
                             settings.checkpoint, settings.board, settings.board, settings.games,
                             settings.simulations, settings.step_limit);
    // The range really is reserved now, and the reservation is enforced rather
    // than asserted: seed_policy.h owns both bands, and requireTrainingSeed throws
    // before an iteration is played if a training seed ever reaches this one.
    //
    // It was not always so. Training seeds were `seed + iteration * 100003` from a
    // default seed of 1, so iteration 9 covered 900028..900283 and overlapped the
    // old default range here by 172 of 200 - and the comment in this spot claimed
    // the range was held out while nothing checked it. Any figure measured before
    // 2026-08-08 was scored partly on games the agent had trained on, and is not
    // comparable with anything printed below. SeedPolicyTest reproduces the old
    // arithmetic so the overlap cannot come back quietly.
    std::cout << std::format("seeds {}..{} (reserved evaluation range), greedy, no root noise\n\n",
                             seeds::evaluationGameSeed(settings.seed_offset, 0),
                             seeds::evaluationGameSeed(settings.seed_offset, settings.games - 1));

    NetworkEvaluator evaluator(network, device);

    MonteCarloSearch::Config search_config;
    search_config.simulations = settings.simulations;
    search_config.exploration = 0.5f;
    search_config.discount = 0.98f;
    search_config.root_noise_fraction = 0.0f;
    search_config.root_noise_alpha = 0.3f;
    search_config.seed = seeds::evaluationGameSeed(settings.seed_offset, 0);
    MonteCarloSearch search(evaluator, search_config);

    const int foods_to_win = settings.board * settings.board - 1;
    int wins = 0;
    int timeouts = 0;
    int deaths = 0;
    long long total_score = 0;
    long long total_steps = 0;
    int best_score = 0;

    auto started = std::chrono::high_resolution_clock::now();

    for (int start = 0; start < settings.games; start += settings.batch)
    {
        const int count = std::min(settings.batch, settings.games - start);
        std::vector<SnakeEnv> games;
        games.reserve(count);
        for (int index = 0; index < count; index++)
        {
            games.emplace_back(settings.board, settings.board,
                               seeds::evaluationGameSeed(settings.seed_offset, start + index));
        }
        std::vector<bool> timed_out(count, false);

        while (true)
        {
            std::vector<int> live;
            std::vector<const SnakeEnv*> roots;
            for (int index = 0; index < count; index++)
            {
                if (games[index].done())
                {
                    continue;
                }
                if (games[index].steps() >= settings.step_limit)
                {
                    timed_out[index] = true;
                    continue;
                }
                live.push_back(index);
                roots.push_back(&games[index]);
            }
            if (live.empty())
            {
                break;
            }

            std::vector<MonteCarloSearch::Result> results = search.search(roots);
            for (size_t position = 0; position < live.size(); position++)
            {
                games[live[position]].step(results[position].best_action);
            }
        }

        for (int index = 0; index < count; index++)
        {
            const SnakeEnv& game = games[index];
            total_score += game.score();
            total_steps += game.steps();
            best_score = std::max(best_score, game.score());
            if (game.won())
            {
                wins++;
            }
            else if (timed_out[index])
            {
                timeouts++;
            }
            else
            {
                deaths++;
            }
        }

        std::cout << std::format("  {}/{} games, wins {}\n", start + count, settings.games, wins);
    }

    double seconds =
        std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - started).count();

    std::cout << std::format("\nWins:    {}/{}  ({:.1f}%)\n", wins, settings.games,
                             100.0 * wins / settings.games);
    std::cout << std::format("Score:   mean {:.3f}, best {} of {}\n",
                             static_cast<double>(total_score) / settings.games, best_score,
                             foods_to_win);
    std::cout << std::format("Steps:   mean {:.3f}\n",
                             static_cast<double>(total_steps) / settings.games);
    std::cout << std::format("Endings: {} won, {} died, {} timed out\n", wins, deaths, timeouts);
    std::cout << std::format("Took {:.2f}s, {} evaluations\n", seconds, evaluator.evaluations());

    return wins == settings.games ? 0 : 1;
}
