// AlphaZeroTrainer: the training loop. Plays a batch of self-play games, appends them
// to a replay buffer, takes gradient steps on samples from it, writes a checkpoint,
// and repeats. One main(); the pieces it drives are testable and it is not.
//
// Hyperparameters follow Du, Gemp, Wu and Wu 2022 (arXiv:2211.09622) and are named
// once, in az_parameters.h. Settings, the step-limit derivation and the progress line
// are in trainer_options.{h,cpp}, where they can be tested.
//
// Usage - starts from scratch, or from a checkpoint with --resume:
//
//     AlphaZeroTrainer.exe --board 10 --iterations 20 --games 256 --batch 128
//       --samples-per-game 3000 --simulations 200 --step-limit 1200 --replay-mb 1024
//       --resume az10_rawvalue348.pt --checkpoint az10_death368.pt
//       --start-iteration 349 --ledger ../../docs/runs.tsv
//
//     # --board 10        10x10; the curriculum starts smaller and resumes upward
//     # --games 256       self-play games per iteration, stepped together
//     # --replay-mb 1024  buffer ceiling; the oldest records are dropped at it
//     # --ledger          appends a started row and a finished row with the cost
//
// Board size is an argument so the curriculum can start small: 708k evaluations/s at
// 6x6 against 55k at 20x20, and cost scales with area, so the cheap signal is small.
//
// The score it prints is self-play, which carries root noise and sampled openings.
// It is not the agent's win rate - AlphaZeroEvaluate gives that.

#include <torch/torch.h>

#include <chrono>
#include <cmath>
#include <format>
#include <fstream>
#include <iostream>
#include <span>
#include <string>
#include <vector>

#include "az_network.h"
#include "az_parameters.h"
#include "iteration_report.h"
#include "network_evaluator.h"
#include "network_setup.h"
#include "progress_printer.h"
#include "replay_window.h"
#include "run_ledger.h"
#include "search_defaults.h"
#include "seed_policy.h"
#include "selfplay.h"
#include "trainer_options.h"
#include "training_batch.h"

namespace
{

// Whether the checkpoint path can be written, checked before the first iteration
// rather than after it.
//
// torch::save throws on a path that does not exist, and it is called at the end
// of an iteration - so a mistyped directory cost a full iteration of self-play
// and training before saying so. Opened for append, which creates the file if it
// is missing and leaves an existing checkpoint untouched.
void requireWritableCheckpoint(const std::string& path)
{
    std::ofstream probe(path, std::ios::app | std::ios::binary);
    if (!probe)
    {
        throw std::runtime_error(
            std::format("cannot write the checkpoint '{}' - does its directory exist?", path));
    }
}

// The run's parameters, printed before any work starts.
//
//     printBanner(settings, step_limit, compute.cuda);
//
// The paper's own figure is printed beside ours wherever the two can differ, so a
// deviation shows up in the log of every run rather than only in the source.
void printBanner(const trainer::Settings& settings, int step_limit, bool cuda)
{
    std::cout << "=== AlphaZero Snake ===" << std::endl;
    std::cout << std::format("board {}x{}  step limit {}  simulations {}\n", settings.board,
                             settings.board, step_limit, settings.simulations);
    std::cout << std::format("network {}x{}  device {}\n", settings.channels, settings.blocks,
                             cuda ? "cuda" : "cpu");
    std::cout << std::format("iterations {}..{} x {} games  seed {}\n", settings.start_iteration,
                             settings.lastIteration(), settings.games_per_iteration, settings.seed);
    // The quantity comparable with the paper's 3,000. --batches alone says nothing about
    // how many games those batches were spread over.
    std::cout << std::format("gradient {} batches of {} = {} samples per game (the paper: 3000)\n",
                             settings.batches_per_iteration, settings.batch_size,
                             settings.samplesPerGame());
    // A deviation from the paper that changes what the value head is trained on, so a run
    // that did not print it could not be told apart from one before it.
    std::cout << std::format("timeout reward {} (the paper: 0, a timeout was free)\n",
                             az::TIMEOUT_REWARD);
    std::cout << std::format("step reward {} per tick, steps head weight {}, tie-break {}\n\n",
                             az::STEP_REWARD, az::STEPS_LOSS_WEIGHT, az::STEPS_TIEBREAK_MARGIN);
}

// The search self-play runs with: the paper's constants and this run's flags.
//
//     SelfPlay play(evaluator, buildSearchConfig(settings), buildPlayConfig(settings, limit));
//
// Every field is set explicitly, including the ones whose Config default already matches.
// A field only one caller sets is how self-play and evaluation come to search differently,
// and that difference appears in no log.
MonteCarloSearch::Config buildSearchConfig(const trainer::Settings& settings)
{
    MonteCarloSearch::Config config = az::paperSearchDefaults(settings.board);
    config.simulations = settings.simulations;
    config.trap_guard = az::TRAP_GUARD;
    config.average_edges = az::AVERAGE_EDGES;
    // From the flag, which defaults to the constant. The ledger records the command line,
    // so a run that set this is distinguishable from one that did not - which a
    // compiled-in constant is not.
    config.exploration_epsilon = settings.exploration_epsilon;
    config.death_cap = az::DEATH_CAP;
    config.root_noise_fraction = az::ROOT_NOISE_FRACTION;
    config.seed = settings.seed;
    return config;
}

// How self-play plays its games: how many at once, how long each may run, and how long
// moves are sampled before the policy turns greedy.
//
//     SelfPlay::Config play_config = buildPlayConfig(settings, settings.stepLimit());
//
// Sampling covers the first half of the board's cells, so early moves vary and the data
// reaches positions the greedy policy would never choose.
SelfPlay::Config buildPlayConfig(const trainer::Settings& settings, int step_limit)
{
    SelfPlay::Config config;
    config.games_in_parallel = settings.games_per_iteration;
    config.step_limit = step_limit;
    // Derived from the board, and it has to be the same discount the search backs up
    // with: these are the returns the value head is trained on, so a mismatch trains the
    // head on one horizon while the search plans on another.
    config.discount = az::deriveDiscount(settings.board);
    config.timeout_reward = az::TIMEOUT_REWARD;
    config.step_reward = az::STEP_REWARD;
    config.temperature = az::VISIT_TEMPERATURE;
    config.temperature_moves = settings.cellCount() / 2;
    config.seed = settings.seed;
    return config;
}

// Loads --resume into `network`, or leaves it at its initial weights when no checkpoint
// was named.
//
//     resumeIfRequested(network, settings);   // prints "resumed from <path>" when it does
//
// Resuming is how the curriculum moves up a board size: the heads pool to a fixed grid, so
// no weight depends on board width and a run at 6x6 loads straight into a run at 10x10.
// Widening is what lets a checkpoint saved before the clock plane resume into a 9-plane
// network with the new channel zeroed, a fine-tune rather than a retrain.
//
// Throws std::runtime_error naming the path when the file cannot be read.
void resumeIfRequested(AlphaZeroNet& network, const trainer::Settings& settings)
{
    if (settings.resume.empty())
    {
        return;
    }
    loadCheckpoint(network, settings.resume, "");
    std::cout << "resumed from " << settings.resume << std::endl;
}

// Plays one iteration's games, filling `fresh` with training records and `summaries` with
// outcomes.
//
//     playOneBatch(play, settings, iteration, fresh, summaries);
//
// The seed is derived from the absolute iteration, so a resumed run continues the sequence
// instead of replaying it. Both ends of the block are checked against the reserved
// evaluation band before a single game is played, because a training seed that strays into
// it silently un-holds-out the evaluation set.
void playOneBatch(SelfPlay& play, const trainer::Settings& settings, int iteration,
                  std::vector<TrainingRecord>& fresh, std::vector<GameSummary>& summaries)
{
    const unsigned int batch_seed = seeds::trainingGameSeed(settings.seed, iteration, 0);
    seeds::requireTrainingSeed(batch_seed);
    seeds::requireTrainingSeed(
        seeds::trainingGameSeed(settings.seed, iteration, settings.games_per_iteration - 1));
    play.playBatch(settings.board, settings.board, batch_seed, fresh, summaries);

    // parseArguments refuses --games below 1, so an empty batch means playBatch returned
    // nothing for a batch it was given - and the summary line divides by this count.
    TORCH_CHECK(!summaries.empty(), "self-play returned no games for a batch of ",
                settings.games_per_iteration);
}

}  // namespace

// Sets Adam's step size for one iteration, decaying geometrically across the run.
//
//     setLearningRate(optimizer, settings, 335, 331, 340);   // 4/9 of the way down
//
// The rate reaches az::LEARNING_RATE * settings.final_learning_rate_fraction on the last
// iteration. Geometric rather than linear because a learning rate is a scale: halving it
// twice should be two equal steps, and a linear ramp spends most of the run near the top.
//
// A run of one iteration, or a fraction of 1, leaves the rate at az::LEARNING_RATE.
void setLearningRate(torch::optim::Adam& optimizer, const trainer::Settings& settings,
                     int iteration, int first_iteration, int last_iteration)
{
    double rate = az::LEARNING_RATE;
    if (last_iteration > first_iteration && settings.final_learning_rate_fraction < 1.0f)
    {
        const double progress = static_cast<double>(iteration - first_iteration) /
                                static_cast<double>(last_iteration - first_iteration);
        rate *= std::pow(static_cast<double>(settings.final_learning_rate_fraction), progress);
    }
    for (torch::optim::OptimizerParamGroup& group : optimizer.param_groups())
    {
        static_cast<torch::optim::AdamOptions&>(group.options()).lr(rate);
    }
}

int main(int argc, char** argv)
{
    // Parses argv into settings; a bad flag or an unwritable checkpoint exits here.
    trainer::Settings settings;
    try
    {
        settings = trainer::parseArguments(
            std::span<const char* const>(argv + 1, static_cast<size_t>(argc - 1)));
        if (!settings.checkpoint.empty())
        {
            requireWritableCheckpoint(settings.checkpoint);
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << std::endl;
        return 1;
    }

    // Longest a game may run, in moves; hitting it is a timeout rather than a death.
    const int step_limit = settings.stepLimit();
    // The neural network's starting seed. With --resume it reaches NO weight at all: the
    // checkpoint overwrites them. Replay draws take it either way.
    torch::manual_seed(seeds::streamSeed(settings.seed, seeds::Stream::Network));

    // Appends a started row to the tab-separated ledger at settings.ledger_path (--ledger,
    // default runs.tsv). A killed run leaves that row unmatched by a finished one.
    ledger::Entry run = ledger::openRun(argc, argv, ledger::Kind::Training, settings.ledger_path);

    // CUDA when available, else the CPU; the banner prints which, above the run's settings.
    const Compute compute = chooseDevice();
    printBanner(settings, step_limit, compute.cuda);

    // The network: `channels` wide, `blocks` deep. Every head pools to 4x4, so no weight
    // depends on the board and a checkpoint moves between board sizes.
    AlphaZeroNet network(settings.board, settings.board, settings.channels, settings.blocks);
    try
    {
        // Loads --resume if given; a failed load ends the run rather than starting fresh.
        resumeIfRequested(network, settings);
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << std::endl;
        run.outcome = ledger::Outcome::Failed;
        ledger::append(settings.ledger_path, run);
        return 1;
    }
    // Moves every parameter to the device, after the load so the restored ones come too.
    network->to(compute.device);

    // Forward passes in batches, Adam over the weights, and the self-play loop that
    // drives the search. The loop below uses nothing else.
    NetworkEvaluator evaluator(network, compute.device);
    torch::optim::Adam optimizer(network->parameters(),
                                 torch::optim::AdamOptions(az::LEARNING_RATE)
                                     .weight_decay(static_cast<double>(settings.weight_decay)));
    SelfPlay play(evaluator, buildSearchConfig(settings), buildPlayConfig(settings, step_limit));

    // The inclusive iteration range. --start-iteration continues the resumed run's
    // numbering, so a curriculum across board sizes reads as one sequence.
    const int first_iteration = settings.start_iteration;
    const int last_iteration = settings.lastIteration();

    // Draws the self-play progress bar. It takes a counter callable rather than the
    // evaluator, which is what keeps the printer free of LibTorch.
    ProgressPrinter printer([&evaluator] { return evaluator.evaluations(); }, step_limit,
                            last_iteration);
    play.setProgressCallback([&](const SelfPlay::Progress& progress) { printer.draw(progress); });

    // The training window: newest records kept, oldest evicted past --replay-mb. Capped
    // by bytes because a record is four times larger at 20x20 than at 10x10.
    ReplayWindow replay(settings.replay_bytes);
    // Scratch for one batch, allocated once here and refilled on every gradient step.
    BatchBuffers buffers = makeBatchBuffers(settings);
    // Apples that fill the board, one per cell but the snake's first. Score prints over it.
    const int foods_to_win = settings.foodsToWin();

    // Counted rather than derived from the settings: a run reports what it did, not what it
    // was asked to do.
    long long games_played_total = 0;
    long long samples_trained_total = 0;
    const auto run_started = std::chrono::high_resolution_clock::now();

    // One iteration: play a block of games, absorb them, take gradient steps over the
    // whole window, print the summary, save the checkpoint.
    for (int iteration = first_iteration; iteration <= last_iteration; iteration++)
    {
        // Geometric decay across the run, so the last iteration trains at
        // --final-lr-fraction of the first. A single iteration, or a fraction of 1, leaves
        // the rate where it started.
        setLearningRate(optimizer, settings, iteration, first_iteration, last_iteration);

        // Start time and evaluation count, subtracted below to give this iteration's cost.
        const auto started = std::chrono::high_resolution_clock::now();
        const long long evaluations_before = evaluator.evaluations();
        printer.startIteration(iteration, evaluations_before);

        // Filled by the games below: one record per move, one summary per game.
        std::vector<TrainingRecord> fresh;
        std::vector<GameSummary> summaries;
        // Eval mode for play, so batch norm uses running statistics rather than the batch's.
        network->eval();
        playOneBatch(play, settings, iteration, fresh, summaries);

        // Moves the records in and evicts the oldest. fresh is spent afterwards.
        replay.absorb(fresh);

        // Time spent playing, taken before training starts so the summary can split the two.
        const double play_seconds =
            std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - started)
                .count();
        // Collapses the per-game outcomes to the counts the summary line prints.
        const trainer::BatchStats stats = trainer::summariseGames(summaries);

        // Adam steps over batches drawn from the whole window, not the fresh games alone.
        network->train();
        const trainer::LossTotals totals =
            trainOnReplay(settings, network, optimizer, replay, buffers, compute.device, iteration);

        // What the iteration came to. Its summary line is what the logs are parsed from, so
        // the format lives in iteration_report and is tested without a GPU.
        trainer::IterationReport report;
        report.iteration = iteration;
        report.games = summaries.size();
        report.foods_to_win = foods_to_win;
        report.sealed_choices = play.sealedChoices();
        report.total_seconds =
            std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - started)
                .count();
        report.play_seconds = play_seconds;
        report.evaluations = evaluator.evaluations() - evaluations_before;

        // Wiped before the summary lands, or the summary is printed over a longer bar and
        // inherits its tail.
        printer.wipe();
        std::cout << trainer::formatIterationSummary(settings.batch_size, report, stats, replay,
                                                     totals)
                  << std::endl;

        // Running totals for the ledger's cost row, counted rather than assumed.
        games_played_total += static_cast<long long>(summaries.size());
        samples_trained_total += static_cast<long long>(totals.batches_run) * settings.batch_size;

        // Overwritten every iteration, so a run killed at hour eight still keeps hour seven.
        if (!settings.checkpoint.empty())
        {
            torch::save(network, settings.checkpoint);
        }
    }

    std::cout << std::endl << "Done." << std::endl;

    // Completes the run's row - outcome, wall-clock, games, samples - and appends it, which
    // is what matches the started row written at the top.
    run.outcome = ledger::Outcome::Finished;
    run.seconds =
        std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - run_started)
            .count();
    run.games = games_played_total;
    run.samples = samples_trained_total;
    ledger::append(settings.ledger_path, run);

    return 0;
}
