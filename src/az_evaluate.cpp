#include <torch/torch.h>
#include <algorithm>
#include <chrono>
#include <format>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "az_network.h"
#include "az_parameters.h"
#include "eval_options.h"
#include "mcts.h"
#include "network_evaluator.h"
#include "seed_policy.h"
#include "snake_env.h"

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

int main(int argc, char** argv)
{
    evaluation::Settings settings;
    try
    {
        settings = evaluation::parseArguments(std::vector<std::string>(argv + 1, argv + argc));
    }
    catch (const std::invalid_argument& error)
    {
        std::cerr << error.what() << std::endl;
        std::cerr << "usage: --checkpoint <file> [--board N] [--games N] [--simulations N]"
                  << std::endl;
        return 2;
    }
    const int step_limit = settings.stepLimit();

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
                             settings.simulations, step_limit);
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
    search_config.exploration = az::EXPLORATION;
    search_config.discount = az::DISCOUNT;
    // Off, deliberately: noise is what makes self-play explore, and a number
    // measured with it on describes the exploration policy rather than the agent.
    search_config.root_noise_fraction = 0.0f;
    search_config.root_noise_alpha = az::ROOT_NOISE_ALPHA;
    search_config.seed = seeds::evaluationGameSeed(settings.seed_offset, 0);
    MonteCarloSearch search(evaluator, search_config);

    const int foods_to_win = settings.foodsToWin();
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
                if (games[index].steps() >= step_limit)
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
