#include <torch/torch.h>
#include "az_network.h"
#include "network_evaluator.h"
#include "seed_policy.h"
#include "selfplay.h"
#include <chrono>
#include <deque>
#include <algorithm>
#include <format>
#include <iostream>
#include <string>

// AlphaZero training loop for Snake.
//
// Hyperparameters follow Du, Gemp, Wu and Wu 2022 (arXiv:2211.09622), which
// reached 944/1000 wins on 10x10 with 200 search states per move over 6,000
// games: discount 0.98, c_puct 0.5, visit-count temperature 0.5, learning rate
// 0.001, minibatches of 100 drawn from a window of recent games. Where this
// deviates it is noted at the deviation, because a hyperparameter that silently
// differs from the paper it cites is worse than one chosen freely.
//
// Board size is an argument so the curriculum can start small. Measured here:
// the network does 708k evaluations/s at 6x6 and 55k at 20x20, and cost scales
// with board area, so the small boards are where the cheap signal is.

namespace
{

// Characters of hashes and dots in the redrawn progress bar.
constexpr int PROGRESS_BAR_WIDTH = 28;

struct Settings
{
    int board = 6;
    int iterations = 20;
    // Absolute index of the first iteration. A resumed run must be given the
    // number the previous run stopped at, or it replays that run's games: the
    // seed for a game is derived from the iteration index, and `--resume`
    // restores weights only.
    int start_iteration = 1;
    int games_per_iteration = 32;
    int simulations = 64;
    int step_limit = 0;  // 0 means derive it from the board
    int channels = 64;
    int blocks = 4;
    float learning_rate = 0.001f;
    int batch_size = 128;
    int batches_per_iteration = 64;
    size_t replay_bytes = 1024u * 1024u * 1024u;  // 1 GiB, measured not guessed
    unsigned int seed = 1;
    std::string checkpoint;
    std::string resume;
};

int parseInt(const char* text)
{
    return std::stoi(text);
}

std::string formatDuration(double seconds)
{
    if (seconds < 0.0 || seconds > 60.0 * 60.0 * 99.0)
    {
        return "--:--";
    }
    int total = static_cast<int>(seconds + 0.5);
    return std::format("{:02}:{:02}", total / 60, total % 60);
}

// One redrawn line, ASCII only, so it works in any console this repository is
// ever run from. Carriage return rather than a newline, so an iteration does not
// scroll a thousand lines past.
void drawProgressBar(int iteration, int iterations, const SelfPlay::Progress& progress,
                     long long evaluations, int step_limit)
{
    double by_games = progress.games_total > 0 ? static_cast<double>(progress.games_finished) /
                                                     static_cast<double>(progress.games_total)
                                               : 0.0;

    // Games finished is the honest measure but it stays at zero for the first
    // minutes of a large board, where nothing has ended yet - a bar pinned at 0
    // percent is the thing this was added to avoid. Moves played against the
    // worst case is a lower bound on real progress and moves from the first
    // second, so the bar shows whichever is further along. Both are
    // non-decreasing, so their maximum is too.
    double worst_case_moves =
        static_cast<double>(progress.games_total) * static_cast<double>(step_limit);
    double by_moves = worst_case_moves > 0.0
                          ? static_cast<double>(progress.moves_played) / worst_case_moves
                          : 0.0;
    double fraction = std::min(1.0, std::max(by_games, by_moves));
    int filled = static_cast<int>(fraction * PROGRESS_BAR_WIDTH);

    std::string cells(PROGRESS_BAR_WIDTH, '.');
    for (int cell = 0; cell < filled && cell < PROGRESS_BAR_WIDTH; cell++)
    {
        cells[cell] = '#';
    }

    std::string bar =
        std::format("\r  iter {}/{} [{}] {:>3}%  games {}/{}  moves {}", iteration, iterations,
                    cells, static_cast<int>(fraction * 100.0), progress.games_finished,
                    progress.games_total, progress.moves_played);

    if (progress.elapsed_seconds > 0.5)
    {
        bar += std::format("  {} ev/s",
                           static_cast<long long>(evaluations / progress.elapsed_seconds));
    }
    bar += std::format("  {}", formatDuration(progress.elapsed_seconds));

    // Remaining time from the share of games already finished. Games do not all
    // take the same length, so this is an estimate and is labelled as one.
    if (fraction > 0.02)
    {
        double remaining = progress.elapsed_seconds * (1.0 / fraction - 1.0);
        bar += std::format(" eta {}", formatDuration(remaining));
    }
    else
    {
        bar += " eta --:--";
    }
    // Trailing blanks wipe the tail of a longer previous line, since this is
    // redrawn in place rather than scrolled.
    bar += "        ";

    std::cout << bar << std::flush;
}

Settings parseArguments(int argc, char** argv)
{
    Settings settings;
    for (int index = 1; index + 1 < argc; index += 2)
    {
        std::string flag = argv[index];
        const char* value = argv[index + 1];
        if (flag == "--board")
        {
            settings.board = parseInt(value);
        }
        else if (flag == "--iterations")
        {
            settings.iterations = parseInt(value);
        }
        else if (flag == "--start-iteration")
        {
            settings.start_iteration = parseInt(value);
        }
        else if (flag == "--games")
        {
            settings.games_per_iteration = parseInt(value);
        }
        else if (flag == "--simulations")
        {
            settings.simulations = parseInt(value);
        }
        else if (flag == "--step-limit")
        {
            settings.step_limit = parseInt(value);
        }
        else if (flag == "--channels")
        {
            settings.channels = parseInt(value);
        }
        else if (flag == "--blocks")
        {
            settings.blocks = parseInt(value);
        }
        else if (flag == "--batch")
        {
            settings.batch_size = parseInt(value);
        }
        else if (flag == "--batches")
        {
            settings.batches_per_iteration = parseInt(value);
        }
        else if (flag == "--seed")
        {
            settings.seed = static_cast<unsigned int>(parseInt(value));
        }
        else if (flag == "--checkpoint")
        {
            settings.checkpoint = value;
        }
        else if (flag == "--resume")
        {
            settings.resume = value;
        }
        else
        {
            std::cerr << "unknown flag: " << flag << std::endl;
        }
    }
    if (settings.step_limit == 0)
    {
        // Du et al. cap a 10x10 game at 1,200 steps. Scaled by area, so the
        // budget per cell is the same at every board size, which keeps "win"
        // meaning the same thing across the curriculum.
        settings.step_limit = 12 * settings.board * settings.board;
    }
    return settings;
}

}  // namespace

int main(int argc, char** argv)
{
    Settings settings = parseArguments(argc, argv);
    torch::manual_seed(settings.seed);

    const bool cuda = torch::cuda::is_available();
    torch::Device device = cuda ? torch::Device(torch::kCUDA) : torch::Device(torch::kCPU);

    std::cout << "=== AlphaZero Snake ===" << std::endl;
    std::cout << std::format("board {}x{}  step limit {}  simulations {}\n", settings.board,
                             settings.board, settings.step_limit, settings.simulations);
    std::cout << std::format("network {}x{}  device {}\n", settings.channels, settings.blocks,
                             cuda ? "cuda" : "cpu");
    std::cout << std::format("iterations {}..{} x {} games  seed {}\n\n", settings.start_iteration,
                             settings.start_iteration + settings.iterations - 1,
                             settings.games_per_iteration, settings.seed);

    AlphaZeroNet network(settings.board, settings.board, settings.channels, settings.blocks);

    // Resuming is how the curriculum moves up a board size: the heads pool to a
    // fixed grid, so nothing in the network depends on board width, and a run at
    // 6x6 loads straight into a run at 10x10.
    if (!settings.resume.empty())
    {
        try
        {
            torch::load(network, settings.resume);
            std::cout << "resumed from " << settings.resume << std::endl;
        }
        catch (const std::exception& error)
        {
            std::cerr << "could not load " << settings.resume << ": " << error.what() << std::endl;
            return 1;
        }
    }
    network->to(device);

    NetworkEvaluator evaluator(network, device);
    torch::optim::Adam optimizer(network->parameters(),
                                 torch::optim::AdamOptions(settings.learning_rate));

    MonteCarloSearch::Config search_config;
    search_config.simulations = settings.simulations;
    search_config.exploration = 0.5f;
    search_config.discount = 0.98f;
    search_config.root_noise_fraction = 0.25f;
    search_config.root_noise_alpha = 0.3f;
    search_config.seed = settings.seed;

    SelfPlay::Config play_config;
    play_config.games_in_parallel = settings.games_per_iteration;
    play_config.step_limit = settings.step_limit;
    play_config.discount = 0.98f;
    play_config.temperature = 0.5f;
    play_config.temperature_moves = settings.board * settings.board / 2;
    play_config.seed = settings.seed;

    SelfPlay play(evaluator, search_config, play_config);

    // Throttled so drawing never becomes the bottleneck it is reporting on.
    const int first_iteration = settings.start_iteration;
    const int last_iteration = settings.start_iteration + settings.iterations - 1;

    auto last_drawn = std::chrono::high_resolution_clock::now();
    int current_iteration = 0;
    long long evaluations_at_iteration_start = 0;
    play.setProgressCallback(
        [&](const SelfPlay::Progress& progress)
        {
            auto now = std::chrono::high_resolution_clock::now();
            if (std::chrono::duration<double>(now - last_drawn).count() < 0.25)
            {
                return;
            }
            last_drawn = now;
            drawProgressBar(current_iteration, last_iteration, progress,
                            evaluator.evaluations() - evaluations_at_iteration_start,
                            settings.step_limit);
        });

    std::deque<TrainingRecord> replay;
    size_t replay_bytes_used = 0;
    const int foods_to_win = settings.board * settings.board - 1;

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
        int batches_run = 0;

        if (static_cast<int>(replay.size()) >= settings.batch_size)
        {
            const int cells = settings.board * settings.board;
            std::vector<float> planes(static_cast<size_t>(settings.batch_size) *
                                      SnakeEnv::PLANE_COUNT * cells);
            std::vector<float> policies(static_cast<size_t>(settings.batch_size) *
                                        SnakeEnv::ACTION_COUNT);
            std::vector<float> values(settings.batch_size);

            for (int batch = 0; batch < settings.batches_per_iteration; batch++)
            {
                for (int item = 0; item < settings.batch_size; item++)
                {
                    size_t pick = static_cast<size_t>(
                        torch::randint(0, static_cast<int64_t>(replay.size()), {1})
                            .item<int64_t>());
                    const TrainingRecord& record = replay[pick];
                    SnakeEnv::encodeSnapshot(
                        settings.board, settings.board, record.position,
                        planes.data() + static_cast<size_t>(item) * SnakeEnv::PLANE_COUNT * cells);
                    for (int action = 0; action < SnakeEnv::ACTION_COUNT; action++)
                    {
                        policies[static_cast<size_t>(item) * SnakeEnv::ACTION_COUNT + action] =
                            record.policy[action];
                    }
                    values[item] = record.value_target;
                }

                torch::Tensor input =
                    torch::from_blob(planes.data(), {settings.batch_size, SnakeEnv::PLANE_COUNT,
                                                     settings.board, settings.board})
                        .to(device);
                torch::Tensor policy_target =
                    torch::from_blob(policies.data(), {settings.batch_size, SnakeEnv::ACTION_COUNT})
                        .to(device);
                torch::Tensor value_target =
                    torch::from_blob(values.data(), {settings.batch_size, 1}).to(device);

                // The value head is a tanh, so its targets have to live in the
                // same range. Rewards run to +/-10, so returns are scaled by the
                // win reward rather than clipped - clipping would make every
                // sufficiently good and sufficiently bad position look alike.
                value_target = torch::tanh(value_target / SnakeEnv::WIN_REWARD);

                auto [policy_logits, value] = network->forward(input);
                torch::Tensor log_policy = torch::log_softmax(policy_logits, 1);
                torch::Tensor policy_loss = -(policy_target * log_policy).sum(1).mean();
                torch::Tensor value_loss = torch::mse_loss(value, value_target);
                torch::Tensor loss = policy_loss + value_loss;

                optimizer.zero_grad();
                loss.backward();
                optimizer.step();

                policy_loss_total += policy_loss.item<double>();
                value_loss_total += value_loss.item<double>();
                batches_run++;
            }
        }

        double total_seconds =
            std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - started)
                .count();
        long long evaluations = evaluator.evaluations() - evaluations_before;

        // Wipe the progress line before the iteration summary lands, or the
        // summary is printed over a longer bar and inherits its tail.
        std::cout << "\r" << std::string(110, ' ') << "\r";

        // Precisions are fixed rather than left to the default so the line keeps
        // a stable shape across runs - these summaries get parsed out of the log.
        std::string summary = std::format(
            "iter {}  score {:.4f}/{}  best {}  wins {}/{}  timeouts {}  buffer {} ({}MB)",
            iteration, total_score / summaries.size(), foods_to_win, best_score, wins,
            summaries.size(), limited, replay.size(), replay_bytes_used / (1024 * 1024));
        if (batches_run > 0)
        {
            summary += std::format("  loss p {:.6f} v {:.6f}", policy_loss_total / batches_run,
                                   value_loss_total / batches_run);
        }
        summary += std::format("  {:.2f}s (play {:.3f}s, {} evals/s)", total_seconds, play_seconds,
                               static_cast<long long>(evaluations / std::max(0.001, play_seconds)));
        std::cout << summary << std::endl;

        if (!settings.checkpoint.empty())
        {
            torch::save(network, settings.checkpoint);
        }
    }

    std::cout << std::endl << "Done." << std::endl;
    return 0;
}
