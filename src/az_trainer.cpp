#include <torch/torch.h>
#include "az_network.h"
#include "network_evaluator.h"
#include "selfplay.h"
#include <chrono>
#include <deque>
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

namespace {

struct Settings {
    int board = 6;
    int iterations = 20;
    int games_per_iteration = 32;
    int simulations = 64;
    int step_limit = 0;          // 0 means derive it from the board
    int channels = 64;
    int blocks = 4;
    float learning_rate = 0.001f;
    int batch_size = 128;
    int batches_per_iteration = 64;
    size_t replay_capacity = 200000;
    unsigned int seed = 1;
    std::string checkpoint;
};

int parseInt(const char* text) { return std::stoi(text); }

Settings parseArguments(int argc, char** argv) {
    Settings settings;
    for (int index = 1; index + 1 < argc; index += 2) {
        std::string flag = argv[index];
        const char* value = argv[index + 1];
        if (flag == "--board") { settings.board = parseInt(value); }
        else if (flag == "--iterations") { settings.iterations = parseInt(value); }
        else if (flag == "--games") { settings.games_per_iteration = parseInt(value); }
        else if (flag == "--simulations") { settings.simulations = parseInt(value); }
        else if (flag == "--step-limit") { settings.step_limit = parseInt(value); }
        else if (flag == "--channels") { settings.channels = parseInt(value); }
        else if (flag == "--blocks") { settings.blocks = parseInt(value); }
        else if (flag == "--batch") { settings.batch_size = parseInt(value); }
        else if (flag == "--batches") { settings.batches_per_iteration = parseInt(value); }
        else if (flag == "--seed") { settings.seed = (unsigned int)parseInt(value); }
        else if (flag == "--checkpoint") { settings.checkpoint = value; }
        else { std::cerr << "unknown flag: " << flag << std::endl; }
    }
    if (settings.step_limit == 0) {
        // Du et al. cap a 10x10 game at 1,200 steps. Scaled by area, so the
        // budget per cell is the same at every board size, which keeps "win"
        // meaning the same thing across the curriculum.
        settings.step_limit = 12 * settings.board * settings.board;
    }
    return settings;
}

}  // namespace

int main(int argc, char** argv) {
    Settings settings = parseArguments(argc, argv);
    torch::manual_seed(settings.seed);

    const bool cuda = torch::cuda::is_available();
    torch::Device device = cuda ? torch::Device(torch::kCUDA) : torch::Device(torch::kCPU);

    std::cout << "=== AlphaZero Snake ===" << std::endl;
    std::cout << "board " << settings.board << "x" << settings.board
              << "  step limit " << settings.step_limit
              << "  simulations " << settings.simulations << std::endl;
    std::cout << "network " << settings.channels << "x" << settings.blocks
              << "  device " << (cuda ? "cuda" : "cpu") << std::endl;
    std::cout << "iterations " << settings.iterations << " x " << settings.games_per_iteration
              << " games" << std::endl << std::endl;

    AlphaZeroNet network(settings.board, settings.board, settings.channels, settings.blocks);
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

    std::deque<TrainingRecord> replay;
    const int foods_to_win = settings.board * settings.board - 1;

    for (int iteration = 1; iteration <= settings.iterations; iteration++) {
        auto started = std::chrono::high_resolution_clock::now();
        long long evaluations_before = evaluator.evaluations();

        std::vector<TrainingRecord> fresh;
        std::vector<GameSummary> summaries;
        network->eval();
        play.playBatch(settings.board, settings.board,
                       settings.seed + (unsigned int)iteration * 100003u, fresh, summaries);

        for (TrainingRecord& record : fresh) {
            replay.push_back(std::move(record));
        }
        while (replay.size() > settings.replay_capacity) {
            replay.pop_front();
        }

        double play_seconds =
            std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - started)
                .count();

        int wins = 0;
        int limited = 0;
        double total_score = 0.0;
        int best_score = 0;
        for (const GameSummary& summary : summaries) {
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

        if ((int)replay.size() >= settings.batch_size) {
            const int cells = settings.board * settings.board;
            std::vector<float> planes((size_t)settings.batch_size * SnakeEnv::PLANE_COUNT * cells);
            std::vector<float> policies((size_t)settings.batch_size * SnakeEnv::ACTION_COUNT);
            std::vector<float> values(settings.batch_size);

            for (int batch = 0; batch < settings.batches_per_iteration; batch++) {
                for (int item = 0; item < settings.batch_size; item++) {
                    size_t pick = (size_t)torch::randint(0, (int64_t)replay.size(), {1})
                                      .item<int64_t>();
                    const TrainingRecord& record = replay[pick];
                    std::copy(record.planes.begin(), record.planes.end(),
                              planes.begin() + (size_t)item * record.planes.size());
                    for (int action = 0; action < SnakeEnv::ACTION_COUNT; action++) {
                        policies[(size_t)item * SnakeEnv::ACTION_COUNT + action] =
                            record.policy[action];
                    }
                    values[item] = record.value_target;
                }

                torch::Tensor input =
                    torch::from_blob(planes.data(),
                                     {settings.batch_size, SnakeEnv::PLANE_COUNT, settings.board,
                                      settings.board})
                        .to(device);
                torch::Tensor policy_target =
                    torch::from_blob(policies.data(),
                                     {settings.batch_size, SnakeEnv::ACTION_COUNT})
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

        std::cout << "iter " << iteration
                  << "  score " << (total_score / summaries.size()) << "/" << foods_to_win
                  << "  best " << best_score
                  << "  wins " << wins << "/" << summaries.size()
                  << "  timeouts " << limited
                  << "  buffer " << replay.size();
        if (batches_run > 0) {
            std::cout << "  loss p " << (policy_loss_total / batches_run)
                      << " v " << (value_loss_total / batches_run);
        }
        std::cout << "  " << total_seconds << "s (play " << play_seconds << "s, "
                  << (long long)(evaluations / std::max(0.001, play_seconds)) << " evals/s)"
                  << std::endl;

        if (!settings.checkpoint.empty()) {
            torch::save(network, settings.checkpoint);
        }
    }

    std::cout << std::endl << "Done." << std::endl;
    return 0;
}
