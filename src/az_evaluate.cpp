#include <torch/torch.h>

#include <process.h>
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
#include "run_ledger.h"
#include "steps_per_apple.h"
#include "mcts.h"
#include "network_evaluator.h"
#include "network_setup.h"
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

namespace
{

// A network of this run's shape with the checkpoint loaded into it, moved to `device` and
// put in evaluation mode.
//
//     AlphaZeroNet network = loadNetwork(settings, compute.device);
//
// Widened rather than loaded flat. A checkpoint saved before the clock plane has an
// 8-plane stem, and torch::load on that mismatch does not throw - it takes the process
// down - so widening is the only path rather than a fallback from a failure nothing can
// catch. Prints every parameter the file did not carry, because a mistyped module name
// looks exactly like a head added after the checkpoint was written.
//
// Throws std::runtime_error naming the path when the file cannot be read.
AlphaZeroNet loadNetwork(const evaluation::Settings& settings, torch::Device device)
{
    AlphaZeroNet network(settings.board, settings.board, settings.channels, settings.blocks);
    loadCheckpoint(network, settings.checkpoint, "");
    network->to(device);
    network->eval();
    return network;
}

// The search configuration this program measures with: the paper's constants, the flags
// the operator set, and noise off.
//
//     MonteCarloSearch search(evaluator, buildSearchConfig(settings));
//
// Every field is set explicitly, including the ones whose default already matches - a
// field only some callers set is how self-play and evaluation come to search differently.
MonteCarloSearch::Config buildSearchConfig(const evaluation::Settings& settings)
{
    MonteCarloSearch::Config search_config;
    search_config.simulations = settings.simulations;
    search_config.exploration = az::EXPLORATION;
    search_config.discount = az::DISCOUNT;
    search_config.step_reward = az::STEP_REWARD;
    search_config.steps_tiebreak_margin = az::STEPS_TIEBREAK_MARGIN;
    search_config.trap_guard = settings.trap_guard;
    search_config.trap_report = az::TRAP_REPORT;
    search_config.average_edges = settings.average_edges;
    search_config.normalize_values = az::NORMALIZE_VALUES;
    search_config.exploration_epsilon = az::EXPLORATION_EPSILON;
    search_config.death_cap = settings.death_cap;
    search_config.death_cap_threshold = az::DEATH_CAP_THRESHOLD;
    // Always on here: a measurement rather than a behaviour, it changes no move, and it
    // costs two comparisons per edge traversal against a forward pass.
    search_config.alias_report = true;
    // Off, deliberately: noise is what makes self-play explore, and a number measured with
    // it on describes the exploration policy rather than the agent.
    search_config.root_noise_fraction = 0.0f;
    search_config.root_noise_alpha = az::ROOT_NOISE_ALPHA;
    // The stream the search imagines apples from. Derived from the offset unless asked
    // otherwise, so two runs can play identical games and differ only in what the search
    // guessed about food it cannot see.
    search_config.seed = settings.search_seed ? *settings.search_seed
                                              : seeds::evaluationGameSeed(settings.seed_offset, 0);
    return search_config;
}

// What the games played so far add up to.
struct Totals
{
    // Games that filled the board.
    int wins{ 0 };
    // Games the snake died in.
    int deaths{ 0 };
    // Games cut off at the step limit.
    int timeouts{ 0 };
    // Apples and moves summed over every game, for the means.
    long long total_score{ 0 };
    long long total_steps{ 0 };
    // Best single game.
    int best_score{ 0 };
};

// Plays `count` games starting at seed index `start`, folding their outcomes into
// `totals` and printing one result line and one pace line per game.
//
//     playBatch(search, settings, step_limit, 0, 64, totals);
//
// The games are stepped together so every search of the batch reaches the network in one
// forward pass. A game that reaches the step limit is a timeout rather than a death - the
// environment does not end it, the caller does.
void playBatch(MonteCarloSearch& search, const evaluation::Settings& settings, int step_limit,
               int start, int count, Totals& totals)
{
    std::vector<SnakeEnv> games;
    games.reserve(count);
    for (int index = 0; index < count; index++)
    {
        games.emplace_back(settings.board, settings.board,
                           seeds::evaluationGameSeed(settings.seed_offset, start + index),
                           step_limit);
        if (settings.freeze_clock_percent)
        {
            // Before the first move, so no search ever sees the live clock. The search
            // steps copies of these, and a copy carries the freeze.
            games.back().freezeClockForAblation(static_cast<float>(*settings.freeze_clock_percent) /
                                                100.0f);
        }
    }
    std::vector<bool> timed_out(count, false);
    // One per game in the batch, fed after every move it makes.
    std::vector<pace::AppleIntervals> apple_intervals(count);

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
            const int index = live[position];
            games[index].step(results[position].best_action);
            apple_intervals[index].observe(games[index].score(), games[index].steps());
        }
    }

    for (int index = 0; index < count; index++)
    {
        const SnakeEnv& game = games[index];
        totals.total_score += game.score();
        totals.total_steps += game.steps();
        totals.best_score = std::max(totals.best_score, game.score());

        evaluation::Outcome outcome = evaluation::Outcome::Died;
        if (game.won())
        {
            outcome = evaluation::Outcome::Won;
            totals.wins++;
        }
        else if (timed_out[index])
        {
            outcome = evaluation::Outcome::TimedOut;
            totals.timeouts++;
        }
        else
        {
            totals.deaths++;
        }
        const unsigned int seed = seeds::evaluationGameSeed(settings.seed_offset, start + index);
        std::cout << evaluation::formatGameLine(seed, outcome, game.score(), game.steps());
        std::cout << pace::formatPaceLine(seed, apple_intervals[index].intervals());
    }
}

// The block of totals the run ends with: rates, means, and the search's own counters.
//
// The alias counters are printed as rates because the raw counts scale with the number of
// games and cannot be compared across runs. Death cap refusals print in both states: zero
// with the cap off is the control, and zero with it on says the threshold never fired
// rather than that the run had no dead ends.
void printSummary(const evaluation::Settings& settings, const Totals& totals, double seconds,
                  const NetworkEvaluator& evaluator, const MonteCarloSearch& search)
{
    std::cout << std::format("\nWins:    {}/{}  ({:.1f}%)\n", totals.wins, settings.games,
                             100.0 * totals.wins / settings.games);
    std::cout << std::format("Score:   mean {:.3f}, best {} of {}\n",
                             static_cast<double>(totals.total_score) / settings.games,
                             totals.best_score, settings.foodsToWin());
    std::cout << std::format("Steps:   mean {:.3f}\n",
                             static_cast<double>(totals.total_steps) / settings.games);
    std::cout << std::format("Endings: {} won, {} died, {} timed out\n", totals.wins, totals.deaths,
                             totals.timeouts);
    std::cout << std::format("Took {:.2f}s, {} evaluations\n", seconds, evaluator.evaluations());
    std::cout << std::format("Sealed choices {}, of which the guard overruled {}\n",
                             search.sealedChoices(), search.trapGuardFires());
    std::cout << std::format("Death cap refusals {}\n", search.deathCapFires());

    const double aliased_share =
        search.revisitedEdges() > 0
            ? 100.0 * static_cast<double>(search.aliasedEdges()) / search.revisitedEdges()
            : 0.0;
    std::cout << std::format("Revisited edges {}, of which aliased {} ({:.2f} percent)\n",
                             search.revisitedEdges(), search.aliasedEdges(), aliased_share);
    // The share that could change a move. A disagreement of a step cost is below what the
    // value head resolves; one of an apple inverts what the edge is worth.
    const double material_share =
        search.revisitedEdges() > 0
            ? 100.0 * static_cast<double>(search.materiallyAliasedEdges()) / search.revisitedEdges()
            : 0.0;
    std::cout << std::format("  of which worth more than half an apple {} ({:.2f} percent)\n",
                             search.materiallyAliasedEdges(), material_share);
    const double node_share =
        search.revisitedNodes() > 0
            ? 100.0 * static_cast<double>(search.aliasedNodes()) / search.revisitedNodes()
            : 0.0;
    std::cout << std::format("Revisited nodes {}, of which aliased {} ({:.2f} percent)\n",
                             search.revisitedNodes(), search.aliasedNodes(), node_share);
}

}  // namespace

// Scores one checkpoint over held-out games and prints the result.
//
//     AlphaZeroEvaluate.exe --checkpoint az10_long308.pt --board 10 --games 1000 \
//       --simulations 200 --batch 256
//
// Returns 2 on a bad flag, 1 on an unreadable checkpoint, and otherwise 1 unless every
// game was won - see the note in the file block about that last one.
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

    // Opened before any work, so a run that is killed leaves a started row and no
    // completion - the only way a killed process records what happened to it.
    ledger::Entry run{ ledger::makeRunId(ledger::utcNow(), static_cast<unsigned int>(_getpid())),
                       ledger::utcNow(),
                       ledger::Kind::Evaluation,
                       ledger::formatCommand(std::vector<std::string>(argv + 1, argv + argc)),
                       ledger::Outcome::Started,
                       0.0,
                       0,
                       0 };
    ledger::append(settings.ledger_path, run);

    const Compute compute = chooseDevice();

    AlphaZeroNet network(settings.board, settings.board, settings.channels, settings.blocks);
    try
    {
        network = loadNetwork(settings, compute.device);
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << std::endl;
        run.outcome = ledger::Outcome::Failed;
        ledger::append(settings.ledger_path, run);
        return 1;
    }

    // The seed range really is reserved, and the reservation is enforced rather than
    // asserted: seed_policy.h owns both bands and throws before a training seed can reach
    // this program.
    std::cout << evaluation::formatHeader(settings);

    NetworkEvaluator evaluator(network, compute.device);
    MonteCarloSearch search(evaluator, buildSearchConfig(settings));

    Totals totals;
    const auto started = std::chrono::high_resolution_clock::now();

    for (int start = 0; start < settings.games; start += settings.batch)
    {
        const int count = std::min(settings.batch, settings.games - start);
        playBatch(search, settings, step_limit, start, count, totals);
        std::cout << std::format("  {}/{} games, wins {}\n", start + count, settings.games,
                                 totals.wins);
    }

    const double seconds =
        std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - started).count();
    printSummary(settings, totals, seconds, evaluator, search);

    run.outcome = ledger::Outcome::Finished;
    run.seconds = seconds;
    run.games = settings.games;
    ledger::append(settings.ledger_path, run);

    return totals.wins == settings.games ? 0 : 1;
}
