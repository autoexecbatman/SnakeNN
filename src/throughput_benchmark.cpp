#include "az_network.h"
#include "vector_env.h"
#include <torch/torch.h>
#include <chrono>
#include <iostream>
#include <vector>

// Measures the two rates that decide the training schedule: how fast the
// simulator produces positions, and how fast the network scores them. Every
// estimate of "how long until it wins" is one of these two numbers times a
// count, so they get measured before any schedule is proposed rather than
// guessed and then defended.

namespace {

double secondsSince(std::chrono::high_resolution_clock::time_point start) {
    auto now = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(now - start).count();
}

void measureEnvironment(int board_size, int env_count, int steps) {
    VectorEnv batch(env_count, board_size, board_size, 1234);
    std::vector<SnakeEnv::Action> actions(env_count, SnakeEnv::Action::STRAIGHT);
    std::vector<SnakeEnv::StepResult> results(env_count);
    std::vector<float> encoded(batch.encodedSizeTotal());

    // Warm the allocator and the caches before timing.
    batch.step(actions.data(), results.data());
    batch.encodeAll(encoded.data());
    batch.resetAll();

    unsigned int cursor = 12345;
    auto start = std::chrono::high_resolution_clock::now();
    long long stepped = 0;
    for (int step = 0; step < steps; step++) {
        for (int index = 0; index < env_count; index++) {
            // Cheap deterministic churn, so the games actually diverge rather
            // than all dying against the same wall on the same tick.
            cursor = cursor * 1664525u + 1013904223u;
            actions[index] = static_cast<SnakeEnv::Action>((cursor >> 16) % SnakeEnv::ACTION_COUNT);
            if (batch.env(index).done()) {
                batch.resetOne(index);
            }
        }
        batch.step(actions.data(), results.data());
        stepped += env_count;
    }
    double step_seconds = secondsSince(start);

    start = std::chrono::high_resolution_clock::now();
    const int encode_rounds = 50;
    for (int round = 0; round < encode_rounds; round++) {
        batch.encodeAll(encoded.data());
    }
    double encode_seconds = secondsSince(start);

    std::cout << "  " << board_size << "x" << board_size << " board, " << env_count << " games"
              << std::endl;
    std::cout << "    step:   " << (long long)(stepped / step_seconds) << " env-steps/s"
              << std::endl;
    std::cout << "    encode: "
              << (long long)((double)encode_rounds * env_count / encode_seconds)
              << " observations/s" << std::endl;
}

void measureNetwork(torch::Device device, int board_size, int channels, int blocks, int batch_size) {
    AlphaZeroNet network(board_size, board_size, channels, blocks);
    network->to(device);
    network->eval();

    torch::NoGradGuard no_grad;
    torch::Tensor input = torch::rand(
        {batch_size, SnakeEnv::PLANE_COUNT, board_size, board_size},
        torch::TensorOptions().device(device));

    for (int warmup = 0; warmup < 5; warmup++) {
        network->forward(input);
    }
    if (device.is_cuda()) {
        torch::cuda::synchronize();
    }

    const int rounds = 30;
    auto start = std::chrono::high_resolution_clock::now();
    for (int round = 0; round < rounds; round++) {
        network->forward(input);
    }
    if (device.is_cuda()) {
        torch::cuda::synchronize();
    }
    double seconds = secondsSince(start);

    long long evaluations = (long long)rounds * batch_size;
    std::cout << "    " << board_size << "x" << board_size << ", " << channels << " channels, "
              << blocks << " blocks, batch " << batch_size << ": "
              << (long long)(evaluations / seconds) << " evals/s" << std::endl;
}

}  // namespace

int main() {
    std::cout << "=== Throughput ===" << std::endl << std::endl;

    std::cout << "Simulator (single thread)" << std::endl;
    measureEnvironment(6, 1024, 2000);
    measureEnvironment(20, 1024, 2000);

    std::cout << std::endl << "Network" << std::endl;
    bool cuda = torch::cuda::is_available();
    std::cout << "  CUDA available: " << (cuda ? "yes" : "no") << std::endl;
    torch::Device device = cuda ? torch::Device(torch::kCUDA) : torch::Device(torch::kCPU);

    measureNetwork(device, 6, 64, 4, 1024);
    measureNetwork(device, 20, 64, 4, 1024);
    measureNetwork(device, 20, 128, 8, 1024);

    std::cout << std::endl << "Done." << std::endl;
    return 0;
}
