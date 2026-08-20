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

#include <process.h>
#include <algorithm>
#include <chrono>
#include <deque>
#include <format>
#include <fstream>
#include <iostream>
#include <span>
#include <string>
#include <vector>

#include "az_network.h"
#include "az_parameters.h"
#include "network_evaluator.h"
#include "run_ledger.h"
#include "seed_policy.h"
#include "selfplay.h"
#include "snake_env.h"
#include "trainer_options.h"

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

}  // namespace

int main(int argc, char** argv)
{
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

    const int step_limit = settings.stepLimit();
    torch::manual_seed(settings.seed);

    // Opened before any work, so a run that is killed leaves a started row and no
    // completion - the only way a killed process records what happened to it.
    ledger::Entry run{ ledger::makeRunId(ledger::utcNow(), static_cast<unsigned int>(_getpid())),
                       ledger::utcNow(),
                       ledger::Kind::Training,
                       ledger::formatCommand(std::vector<std::string>(argv + 1, argv + argc)),
                       ledger::Outcome::Started,
                       0.0,
                       0,
                       0 };
    ledger::append(settings.ledger_path, run);

    const bool cuda = torch::cuda::is_available();
    torch::Device device = cuda ? torch::Device(torch::kCUDA) : torch::Device(torch::kCPU);

    std::cout << "=== AlphaZero Snake ===" << std::endl;
    std::cout << std::format("board {}x{}  step limit {}  simulations {}\n", settings.board,
                             settings.board, step_limit, settings.simulations);
    std::cout << std::format("network {}x{}  device {}\n", settings.channels, settings.blocks,
                             cuda ? "cuda" : "cpu");
    std::cout << std::format("iterations {}..{} x {} games  seed {}\n", settings.start_iteration,
                             settings.lastIteration(), settings.games_per_iteration, settings.seed);
    // Printed because it is the quantity comparable with the paper's 3,000, and
    // --batches on its own is not - it says nothing about how many games those
    // batches were spread over.
    std::cout << std::format("gradient {} batches of {} = {} samples per game (the paper: 3000)\n",
                             settings.batches_per_iteration, settings.batch_size,
                             settings.samplesPerGame());
    // A deviation from the paper that changes what the value head is trained on,
    // so a run that did not print it could not be told apart from one before it.
    std::cout << std::format("timeout reward {} (the paper: 0, a timeout was free)\n",
                             az::TIMEOUT_REWARD);
    std::cout << std::format("step reward {} per tick, steps head weight {}, tie-break {}\n\n",
                             az::STEP_REWARD, az::STEPS_LOSS_WEIGHT, az::STEPS_TIEBREAK_MARGIN);

    AlphaZeroNet network(settings.board, settings.board, settings.channels, settings.blocks);

    // Resuming is how the curriculum moves up a board size: the heads pool to a
    // fixed grid, so nothing in the network depends on board width, and a run at
    // 6x6 loads straight into a run at 10x10.
    if (!settings.resume.empty())
    {
        try
        {
            // Widened rather than loaded flat, so a checkpoint saved before the
            // clock plane resumes into a 9-plane network with the new channel
            // zeroed - a fine-tune from where it left off rather than a retrain.
            const std::vector<std::string> missing = network->loadNarrowerStem(settings.resume);
            std::cout << "resumed from " << settings.resume << std::endl;
            // Printed rather than swallowed: a head added since the checkpoint was
            // written looks exactly like a mistyped module name from here.
            for (const std::string& name : missing)
            {
                std::cout << std::format("  fresh, absent from the checkpoint: {}\n", name);
            }
        }
        catch (const std::exception& error)
        {
            std::cerr << "could not load " << settings.resume << ": " << error.what() << std::endl;
            run.outcome = ledger::Outcome::Failed;
            ledger::append(settings.ledger_path, run);
            return 1;
        }
    }
    network->to(device);

    NetworkEvaluator evaluator(network, device);
    torch::optim::Adam optimizer(network->parameters(),
                                 torch::optim::AdamOptions(az::LEARNING_RATE));

    MonteCarloSearch::Config search_config;
    search_config.simulations = settings.simulations;
    search_config.exploration = az::EXPLORATION;
    search_config.discount = az::DISCOUNT;
    search_config.step_reward = az::STEP_REWARD;
    search_config.steps_tiebreak_margin = az::STEPS_TIEBREAK_MARGIN;
    search_config.trap_guard = az::TRAP_GUARD;
    search_config.trap_report = az::TRAP_REPORT;
    // Set explicitly rather than left to the Config default, which happens to match.
    // Self-play and evaluation searching differently is a difference no log records,
    // and a constant only one of them reads is how that arrives.
    search_config.average_edges = az::AVERAGE_EDGES;
    search_config.death_cap = az::DEATH_CAP;
    search_config.death_cap_threshold = az::DEATH_CAP_THRESHOLD;
    search_config.root_noise_fraction = az::ROOT_NOISE_FRACTION;
    search_config.root_noise_alpha = az::ROOT_NOISE_ALPHA;
    search_config.seed = settings.seed;

    SelfPlay::Config play_config;
    play_config.games_in_parallel = settings.games_per_iteration;
    play_config.step_limit = step_limit;
    play_config.discount = az::DISCOUNT;
    play_config.timeout_reward = az::TIMEOUT_REWARD;
    play_config.step_reward = az::STEP_REWARD;
    play_config.temperature = az::VISIT_TEMPERATURE;
    play_config.temperature_moves = settings.cellCount() / 2;
    play_config.seed = settings.seed;

    SelfPlay play(evaluator, search_config, play_config);

    const int first_iteration = settings.start_iteration;
    const int last_iteration = settings.lastIteration();

    // Throttled so drawing never becomes the bottleneck it is reporting on, and
    // the length of the last line drawn is kept so the next write can wipe
    // exactly it. A fixed-width wipe left the tail of anything longer on screen.
    auto last_drawn = std::chrono::high_resolution_clock::now();
    int current_iteration = 0;
    long long evaluations_at_iteration_start = 0;
    size_t drawn_length = 0;
    play.setProgressCallback(
        [&](const SelfPlay::Progress& progress)
        {
            auto now = std::chrono::high_resolution_clock::now();
            if (std::chrono::duration<double>(now - last_drawn).count() < 0.25)
            {
                return;
            }
            last_drawn = now;

            trainer::ProgressSnapshot snapshot;
            snapshot.games_total = progress.games_total;
            snapshot.games_finished = progress.games_finished;
            snapshot.moves_played = progress.moves_played;
            snapshot.evaluations = evaluator.evaluations() - evaluations_at_iteration_start;
            snapshot.step_limit = step_limit;
            snapshot.elapsed_seconds = progress.elapsed_seconds;

            const std::string bar =
                trainer::formatProgressBar(current_iteration, last_iteration, snapshot);
            std::cout << "\r" << bar;
            if (bar.size() < drawn_length)
            {
                std::cout << std::string(drawn_length - bar.size(), ' ');
            }
            drawn_length = bar.size();
            std::cout << std::flush;
        });

    std::deque<TrainingRecord> replay;
    size_t replay_bytes_used = 0;
    const int foods_to_win = settings.foodsToWin();

    // Counted rather than derived from the settings: a run reports what it did, not
    // what it was asked to do.
    long long games_played_total = 0;
    long long samples_trained_total = 0;
    const auto run_started = std::chrono::high_resolution_clock::now();

    for (int iteration = first_iteration; iteration <= last_iteration; iteration++)
    {
        auto started = std::chrono::high_resolution_clock::now();
        long long evaluations_before = evaluator.evaluations();
        current_iteration = iteration;
        evaluations_at_iteration_start = evaluations_before;

        std::vector<TrainingRecord> fresh;
        std::vector<GameSummary> summaries;
        network->eval();
        // Absolute iteration, so a resumed run continues the seed sequence
        // instead of replaying it, and checked against the reserved evaluation
        // range before a single game is played.
        const unsigned int batch_seed = seeds::trainingGameSeed(settings.seed, iteration, 0);
        seeds::requireTrainingSeed(batch_seed);
        seeds::requireTrainingSeed(
            seeds::trainingGameSeed(settings.seed, iteration, settings.games_per_iteration - 1));
        play.playBatch(settings.board, settings.board, batch_seed, fresh, summaries);

        // parseArguments refuses --games below 1, so an empty batch means
        // playBatch returned nothing for a batch it was given - and the summary
        // line below divides by this count.
        //
        // TORCH_CHECK rather than assert, and the distinction is not stylistic:
        // this file links LibTorch, so a debug build of it cannot run at all -
        // the shipped libraries are release-only and the binary dies of an access
        // violation before main - while the release build defines NDEBUG and
        // compiles an assert away. An assert here would be unreachable in both
        // configurations. Every check in a Torch-linked file in this repository
        // is a TORCH_CHECK for that reason, and every check in a Torch-free one
        // is an assert.
        TORCH_CHECK(!summaries.empty(), "self-play returned no games for a batch of ",
                    settings.games_per_iteration);

        for (TrainingRecord& record : fresh)
        {
            replay_bytes_used += record.bytesUsed();
            replay.push_back(std::move(record));
        }
        // Capped by bytes, because a record count means a different amount of
        // memory on every board size, and the first long run of this trainer
        // took the machine into swap.
        while (replay_bytes_used > settings.replay_bytes && !replay.empty())
        {
            replay_bytes_used -= replay.front().bytesUsed();
            replay.pop_front();
        }

        double play_seconds =
            std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - started)
                .count();

        int wins = 0;
        int limited = 0;
        double total_score = 0.0;
        int best_score = 0;
        for (const GameSummary& summary : summaries)
        {
            wins += summary.won ? 1 : 0;
            limited += summary.hit_step_limit ? 1 : 0;
            total_score += summary.score;
            best_score = std::max(best_score, summary.score);
        }

        // Training. Sampling with replacement from the recent window, as in the
        // paper's "minibatches drawn from the last 2,000 games".
        network->train();
        double policy_loss_total = 0.0;
        double value_loss_total = 0.0;
        double death_loss_total = 0.0;
        double usable_labels_total = 0.0;
        int batches_run = 0;

        if (static_cast<int>(replay.size()) >= settings.batch_size)
        {
            const int cells = settings.cellCount();
            std::vector<float> planes(static_cast<size_t>(settings.batch_size) *
                                      SnakeEnv::PLANE_COUNT * cells);
            std::vector<float> policies(static_cast<size_t>(settings.batch_size) *
                                        SnakeEnv::ACTION_COUNT);
            std::vector<float> values(settings.batch_size);
            std::vector<float> steps(settings.batch_size);
            std::vector<float> death_risks(static_cast<size_t>(settings.batch_size) *
                                           SnakeEnv::ACTION_COUNT);
            // One per record: whether its risk label is worth learning from. A
            // batch can be almost entirely masked out, which is why the loss
            // divides by what survived rather than by the batch size.
            std::vector<float> death_mask(settings.batch_size);

            for (int batch = 0; batch < settings.batches_per_iteration; batch++)
            {
                for (int item = 0; item < settings.batch_size; item++)
                {
                    size_t pick = static_cast<size_t>(
                        torch::randint(0, static_cast<int64_t>(replay.size()), { 1 })
                            .item<int64_t>());
                    const TrainingRecord& record = replay[pick];
                    SnakeEnv::encodeSnapshot(
                        settings.board, settings.board, record.position,
                        planes.data() + static_cast<size_t>(item) * SnakeEnv::PLANE_COUNT * cells);
                    for (int action = 0; action < SnakeEnv::ACTION_COUNT; action++)
                    {
                        policies[static_cast<size_t>(item) * SnakeEnv::ACTION_COUNT + action] =
                            record.policy[action];
                        death_risks[static_cast<size_t>(item) * SnakeEnv::ACTION_COUNT + action] =
                            record.death_risk_target[action];
                    }
                    values[item] = record.value_target;
                    steps[item] = record.steps_target;
                    death_mask[item] = record.death_risk_usable ? 1.0f : 0.0f;
                }

                torch::Tensor input =
                    torch::from_blob(planes.data(), { settings.batch_size, SnakeEnv::PLANE_COUNT,
                                                      settings.board, settings.board })
                        .to(device);
                torch::Tensor policy_target =
                    torch::from_blob(policies.data(),
                                     { settings.batch_size, SnakeEnv::ACTION_COUNT })
                        .to(device);
                torch::Tensor value_target =
                    torch::from_blob(values.data(), { settings.batch_size, 1 }).to(device);
                torch::Tensor steps_target =
                    torch::from_blob(steps.data(), { settings.batch_size, 1 }).to(device);
                torch::Tensor death_target =
                    torch::from_blob(death_risks.data(),
                                     { settings.batch_size, SnakeEnv::ACTION_COUNT })
                        .to(device);
                torch::Tensor death_weight =
                    torch::from_blob(death_mask.data(), { settings.batch_size, 1 }).to(device);

                // The target is the return itself. The head is bounded at
                // VALUE_SCALE rather than at 1, so no squashing is needed to make
                // the two comparable, and the search receives a value in the same
                // units as the rewards it adds to it.

                const Prediction prediction = network->forward(input);
                torch::Tensor log_policy = torch::log_softmax(prediction.policy_logits, 1);
                torch::Tensor policy_loss = -(policy_target * log_policy).sum(1).mean();
                // Measured on the normalised scale, which is the same loss the
                // squashed version produced up to a constant - so the balance
                // against the policy loss, and every learning rate chosen under
                // it, carries over unchanged.
                torch::Tensor value_loss = torch::mse_loss(prediction.value / az::VALUE_SCALE,
                                                           value_target / az::VALUE_SCALE);
                // Undiscounted, unlike the value: this is the only estimate here
                // that can see as far as the deadline.
                torch::Tensor steps_loss = torch::mse_loss(prediction.steps_to_go, steps_target);
                // Cross entropy rather than squared error, because the head is a
                // sigmoid and the target is a probability. Averaged over the
                // records whose label survived the mask, not over the batch: with
                // a fixed denominator a batch of mostly unusable labels would
                // report a small loss and take a correspondingly small step.
                torch::Tensor death_elementwise = torch::binary_cross_entropy(
                    prediction.death_risk, death_target, {}, at::Reduction::None);
                torch::Tensor usable = death_weight.sum().clamp_min(1.0f);
                torch::Tensor death_loss =
                    (death_elementwise.mean(1, true) * death_weight).sum() / usable;
                torch::Tensor loss = policy_loss + value_loss + az::STEPS_LOSS_WEIGHT * steps_loss +
                                     az::DEATH_LOSS_WEIGHT * death_loss;

                // A non-finite loss trains every weight into NaN and the run
                // continues printing plausible-looking iterations afterwards, so
                // it stops here instead. TORCH_CHECK rather than assert: LibTorch
                // ships release-only libraries, and a debug binary linked against
                // them dies before reaching any assertion.
                TORCH_CHECK(std::isfinite(loss.item<double>()), "loss is not finite at iteration ",
                            iteration, " batch ", batch, " - policy ", policy_loss.item<double>(),
                            " value ", value_loss.item<double>(), " steps ",
                            steps_loss.item<double>(), " death ", death_loss.item<double>(),
                            " usable labels ", usable.item<double>());

                optimizer.zero_grad();
                loss.backward();
                optimizer.step();

                policy_loss_total += policy_loss.item<double>();
                value_loss_total += value_loss.item<double>();
                // Both read every batch rather than only on failure. The death
                // loss is meaningless without the count beside it: a mask that
                // keeps almost nothing reports a small loss and takes a
                // correspondingly small step, which reads in the log exactly like
                // a head that has already learned its target.
                death_loss_total += death_loss.item<double>();
                usable_labels_total += usable.item<double>();
                batches_run++;
            }
        }

        double total_seconds =
            std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - started)
                .count();
        long long evaluations = evaluator.evaluations() - evaluations_before;

        // Wipe the progress line before the iteration summary lands, or the
        // summary is printed over a longer bar and inherits its tail.
        std::cout << "\r" << std::string(drawn_length, ' ') << "\r";
        drawn_length = 0;

        // Precisions are fixed rather than left to the default so the line keeps
        // a stable shape across runs - these summaries get parsed out of the log.
        std::string summary = std::format(
            "iter {}  score {:.4f}/{}  best {}  wins {}/{}  timeouts {}  sealed {}  buffer {} "
            "({}MB)",
            iteration, total_score / summaries.size(), foods_to_win, best_score, wins,
            summaries.size(), limited, play.sealedChoices(), replay.size(),
            replay_bytes_used / (1024 * 1024));
        if (batches_run > 0)
        {
            summary += std::format("  loss p {:.6f} v {:.6f} d {:.6f}  labels {:.1f}/{}",
                                   policy_loss_total / batches_run, value_loss_total / batches_run,
                                   death_loss_total / batches_run,
                                   usable_labels_total / batches_run, settings.batch_size);
        }
        summary += std::format("  {:.2f}s (play {:.3f}s, {} evals/s)", total_seconds, play_seconds,
                               static_cast<long long>(evaluations / std::max(0.001, play_seconds)));
        std::cout << summary << std::endl;

        games_played_total += static_cast<long long>(summaries.size());
        samples_trained_total += static_cast<long long>(batches_run) * settings.batch_size;

        if (!settings.checkpoint.empty())
        {
            torch::save(network, settings.checkpoint);
        }
    }

    std::cout << std::endl << "Done." << std::endl;

    run.outcome = ledger::Outcome::Finished;
    run.seconds =
        std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - run_started)
            .count();
    run.games = games_played_total;
    run.samples = samples_trained_total;
    ledger::append(settings.ledger_path, run);

    return 0;
}
