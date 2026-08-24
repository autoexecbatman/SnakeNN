// It plays Snake and counts the wins.
//
// A game is one round of Snake, played to the end. The board starts empty apart from a snake
// one segment long at the centre and a single apple. Every move steers relative to the
// current heading - straight, left or right, with no reverse - and eating an apple grows the
// snake by one and puts a new apple on a random empty square. The game ends in exactly one
// of three ways: the snake runs into a wall or into itself, it fills the board, or it uses
// up its step budget. Filling a 20x20 board takes 399 apples. This program plays a batch of
// those games start to finish, choosing every move with a saved network and the search, and
// reports how many of them filled the board.
//
// It plays those games itself, the same way training does, and that is worth stating plainly
// because it is the source of every mistake this program exists to prevent. It does not use
// the trainer's self-play code. It differs from a training game in exactly two ways: no
// Dirichlet noise at the root, and no sampling of early moves at temperature.
//
// Those two exist because training and measuring want opposite things from a game. Training
// wants coverage. A network only learns about positions its games actually reach, so an
// agent that always plays the move it currently likes best sees one line of play forever and
// never finds out what lay down the moves it had already dismissed - including the better
// ones. The noise and the sampling are there to push it off its own preference and into
// positions it would not have chosen, which is the only way the data covers them.
//
// Measuring wants the opposite: the agent playing as well as it can. Exploration is
// deliberately playing worse than you know how to, so a score taken from training games is
// the score of an agent that was handicapping itself on purpose - it measures the
// exploration policy rather than the agent. That is why the two are separate programs rather
// than one with a flag. Here the move played is the visit-count argmax, which is what the
// agent would actually do, so this number is the agent's.
//
// The visit-count argmax, spelled out. Before each move the search runs 200 simulations from
// the current position. Every simulation walks down the tree and back, and each walk credits
// a visit to whichever of the three root actions it set off down. When the simulations are
// spent, each action holds a count - say 14, 181, 5 - and argmax means play the one with the
// largest, with no randomness anywhere. Counts rather than values, because a value can be
// high off a single noisy evaluation, while a count only grows if the action kept winning
// against its siblings as the tree deepened. The count is the search's aggregate opinion;
// any one value estimate is not.
//
// One thing overrides it, which is why mcts.h says to read `best_action` rather than
// recompute it: an action within STEPS_TIEBREAK_MARGIN - 5 percent - of the top count is
// played instead when the steps head expects it to finish sooner. It can only move choices
// the search was already near-indifferent about. Training uses the same counts a third way
// again: it samples early moves in proportion to them, which is the temperature above.
//
// Held-out seeds, since the term hides the mechanism. One seed fixes where every apple will
// appear, so replaying a seed replays the same game. Training takes its seeds from low
// numbers; this program takes its own from a band reserved at 0xE0000000 and above, which
// training cannot reach. "Held out" therefore means the apple sequences here are ones the
// network has never played. It is enforced rather than intended: seed_policy.h owns both
// bands and throws if a training seed reaches this program. That check exists because the
// property was once written in a comment and checked nowhere, and 172 of 200 evaluation
// seeds turned out to be training seeds.
//
// The metric is win rate: games in which the board is filled inside the step limit. Average
// score is printed beside it: a snake that eats 147 apples of 399 every single game has a
// win rate of zero.
//
// Run it from the directory holding the checkpoint. Only --checkpoint is required; the
// defaults are a 6x6 board and 64 games, enough to confirm the program runs.
//
//     AlphaZeroEvaluate.exe --checkpoint az20_iter330.pt --board 20 --games 100
//                           --simulations 200 --batch 256 --search-seed 777
//
//     --checkpoint   the .pt file to score
//     --board        side of the square board; it need not be the board the checkpoint
//                    trained on, because the network pools to a fixed 4x4 and a 10x10
//                    checkpoint loads and plays at 20x20
//     --games        how many held-out games to play
//     --simulations  search depth per move; a win rate without this number means nothing
//     --batch        games in flight - changes how long the run takes, not how well it plays
//     --search-seed  fixes the search's random stream so two runs can be compared
//     --ledger       run record, default runs.tsv; a started row is written before any
//                    work, so a killed run still leaves evidence it was killed
//
// What that command printed, 2026-08-22:
//
//     az20_iter330.pt on 20x20, 100 games, 200 simulations, step limit 4800, batch 256
//     seeds 3758096384..3758096483 (reserved evaluation range), greedy, no root noise
//     Wins:    0/100  (0.0%)
//     Score:   mean 146.940, best 172 of 399
//     Steps:   mean 3532.230
//     Endings: 0 won, 87 died, 13 timed out
//     Took 1349.39s, 68761143 evaluations
//
// Read the endings line before the win rate. Dying and running out of steps are different
// failures and they need opposite fixes, and the win rate alone cannot tell them apart.
//
// Exit codes: 2 for a bad flag, 1 for a checkpoint that cannot be read or for any run that
// did not win every game, 0 only when every game was won. Zero means perfect, not finished.

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
#include "evaluation_report.h"
#include "run_ledger.h"
#include "steps_per_apple.h"
#include "mcts.h"
#include "network_evaluator.h"
#include "network_setup.h"
#include "search_defaults.h"
#include "seed_policy.h"
#include "snake_env.h"

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
    MonteCarloSearch::Config search_config = az::paperSearchDefaults();
    search_config.simulations = settings.simulations;
    search_config.trap_guard = settings.trap_guard;
    search_config.average_edges = settings.average_edges;
    search_config.exploration_epsilon = az::EXPLORATION_EPSILON;
    search_config.death_cap = settings.death_cap;
    // Always on here: a measurement rather than a behaviour, it changes no move, and it
    // costs two comparisons per edge traversal against a forward pass.
    search_config.alias_report = true;
    // Off, deliberately: noise is what makes self-play explore, and a number measured with
    // it on describes the exploration policy rather than the agent.
    search_config.root_noise_fraction = 0.0f;
    // The stream the search imagines apples from. Derived from the offset unless asked
    // otherwise, so two runs can play identical games and differ only in what the search
    // guessed about food it cannot see.
    search_config.seed = settings.search_seed ? *settings.search_seed
                                              : seeds::evaluationGameSeed(settings.seed_offset, 0);
    return search_config;
}

// Plays `count` games starting at seed index `start`, folding their outcomes into
// `totals` and printing one result line and one pace line per game.
//
//     playBatch(search, settings, step_limit, 0, 64, totals);
//
// The games are stepped together so every search of the batch reaches the network in one
// forward pass. A game that reaches the step limit is a timeout rather than a death - the
// environment does not end it, the caller does.
void playBatch(MonteCarloSearch& search, const evaluation::Settings& settings, int step_limit,
               int start, int count, evaluation::Totals& totals)
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

    // Bounded rather than open: a game is skipped once it reaches step_limit, and every
    // live game advances one move per pass, so the batch cannot outlast that many passes.
    // The break below is the normal exit; the bound is what stops a defect elsewhere from
    // turning this into a hang.
    for (int pass = 0; pass < step_limit; pass++)
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
void printSummary(const evaluation::Settings& settings, const evaluation::Totals& totals,
                  double seconds, const NetworkEvaluator& evaluator, const MonteCarloSearch& search)
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
    // Parses argv into settings; a bad flag prints the usage line and exits 2.
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
    // Longest a game may run, in moves; reaching it is a timeout rather than a death.
    const int step_limit = settings.stepLimit();

    // Appends a started row to the ledger at settings.ledger_path (--ledger, default
    // runs.tsv). A killed run leaves that row unmatched by a finished one.
    ledger::Entry run = ledger::openRun(argc, argv, ledger::Kind::Evaluation, settings.ledger_path);

    // CUDA when available, else the CPU; the header prints which.
    const Compute compute = chooseDevice();

    // The checkpoint being scored, loaded in eval mode. A failure here closes the
    // ledger row as failed rather than leaving it open.
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

    // Prints the checkpoint, board, search settings and the seed band. The band is
    // enforced rather than described: seed_policy.h throws before a training seed
    // could reach this program.
    std::cout << evaluation::formatHeader(settings);

    // One search, reused for every batch, so the evaluation and simulation counts the
    // summary prints cover the whole run.
    NetworkEvaluator evaluator(network, compute.device);
    MonteCarloSearch search(evaluator, buildSearchConfig(settings));

    // Wins, deaths and timeouts, which partition the games, plus the running clock.
    evaluation::Totals totals;
    const auto started = std::chrono::high_resolution_clock::now();

    // Games in blocks of --batch, stepped together so each search reaches the network
    // in one forward pass. mcts draws food placement across a whole batch in lockstep,
    // so two runs at different batch sizes searched different futures.
    for (int start = 0; start < settings.games; start += settings.batch)
    {
        const int count = std::min(settings.batch, settings.games - start);
        playBatch(search, settings, step_limit, start, count, totals);
        std::cout << std::format("  {}/{} games, wins {}\n", start + count, settings.games,
                                 totals.wins);
    }

    // What the run cost, printed with the outcome counts and the search's own tallies.
    const double seconds =
        std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - started).count();
    printSummary(settings, totals, seconds, evaluator, search);

    // Completes the run's row - outcome, wall-clock, games - and appends it, matching
    // the started row written above.
    run.outcome = ledger::Outcome::Finished;
    run.seconds = seconds;
    run.games = settings.games;
    ledger::append(settings.ledger_path, run);

    // 0 only when every game was won. Any loss is a non-zero exit, so a script cannot
    // read a partial score as success.
    return totals.wins == settings.games ? 0 : 1;
}
