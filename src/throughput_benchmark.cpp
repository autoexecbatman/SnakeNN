#include <torch/torch.h>

#include <chrono>
#include <iostream>
#include <vector>

#include "az_network.h"
#include "az_parameters.h"
#include "snake_env.h"
#include "vector_env.h"
#include "hamiltonian_cycle.h"

// Measures the two rates that decide the training schedule: how fast the
// simulator produces positions, and how fast the network scores them. Every
// estimate of "how long until it wins" is one of these two numbers times a
// count, so they get measured before any schedule is proposed rather than
// guessed and then defended.

namespace
{

double secondsSince(std::chrono::high_resolution_clock::time_point start)
{
    auto now = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(now - start).count();
}

void measureEnvironment(int board_size, int env_count, int steps)
{
    VectorEnv batch(env_count, board_size, board_size, 1234, az::deriveStepLimit(board_size));
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
    for (int step = 0; step < steps; step++)
    {
        for (int index = 0; index < env_count; index++)
        {
            // Cheap deterministic churn, so the games actually diverge rather
            // than all dying against the same wall on the same tick.
            cursor = cursor * 1664525u + 1013904223u;
            actions[index] = static_cast<SnakeEnv::Action>((cursor >> 16) % SnakeEnv::ACTION_COUNT);
            if (batch.env(index).done())
            {
                batch.resetOne(index);
            }
        }
        batch.step(actions.data(), results.data());
        stepped += env_count;
    }
    double step_seconds = secondsSince(start);

    start = std::chrono::high_resolution_clock::now();
    const int encode_rounds = 50;
    for (int round = 0; round < encode_rounds; round++)
    {
        batch.encodeAll(encoded.data());
    }
    double encode_seconds = secondsSince(start);

    std::cout << "  " << board_size << "x" << board_size << " board, " << env_count << " games"
              << std::endl;
    std::cout << "    step:   " << (long long)(stepped / step_seconds) << " env-steps/s"
              << std::endl;
    std::cout << "    encode: " << (long long)((double)encode_rounds * env_count / encode_seconds)
              << " observations/s" << std::endl;
}

void measureNetwork(torch::Device device, int board_size, int channels, int blocks, int batch_size)
{
    AlphaZeroNet network(board_size, board_size, channels, blocks);
    network->to(device);
    network->eval();

    torch::NoGradGuard no_grad;
    torch::Tensor input = torch::rand({ batch_size, SnakeEnv::PLANE_COUNT, board_size, board_size },
                                      torch::TensorOptions().device(device));

    for (int warmup = 0; warmup < 5; warmup++)
    {
        network->forward(input);
    }
    if (device.is_cuda())
    {
        torch::cuda::synchronize();
    }

    const int rounds = 30;
    auto start = std::chrono::high_resolution_clock::now();
    for (int round = 0; round < rounds; round++)
    {
        network->forward(input);
    }
    if (device.is_cuda())
    {
        torch::cuda::synchronize();
    }
    double seconds = secondsSince(start);

    long long evaluations = (long long)rounds * batch_size;
    std::cout << "    " << board_size << "x" << board_size << ", " << channels << " channels, "
              << blocks << " blocks, batch " << batch_size << ": "
              << (long long)(evaluations / seconds) << " evals/s" << std::endl;
}

// The search replays each descent from a copy of the root rather than storing a
// snapshot per node. That trade is only right if copying a game is cheap next
// to evaluating one, so the copy gets measured rather than assumed.
void measureCloning(int board_size, int clones)
{
    SnakeEnv source(board_size, board_size, 8, az::deriveStepLimit(board_size));
    // Grow the snake with the cycle follower rather than a greedy walker. A
    // greedy walker dies early and leaves a short body, which measures the
    // wrong thing: the copy that matters is a late-game one, where the body is
    // most of the board. Three quarters full.
    HamiltonianCycle cycle(board_size, board_size);
    if (!cycle.generateCycle())
    {
        std::cout << "  cycle generation failed for " << board_size << std::endl;
        return;
    }

    const int target_length = source.cellCount() * 3 / 4;
    int guard = 0;
    while ((int)source.body().size() < target_length && guard++ < 4000000 && !source.done())
    {
        Position wanted = cycle.getNext(source.body()[0]);
        bool moved = false;
        for (int action = 0; action < SnakeEnv::ACTION_COUNT && !moved; action++)
        {
            if (source.headAfter(static_cast<SnakeEnv::Action>(action)) == wanted)
            {
                source.step(static_cast<SnakeEnv::Action>(action));
                moved = true;
            }
        }
        if (!moved)
        {
            // Not yet aligned with the cycle; one turn fixes that.
            source.step(SnakeEnv::Action::LEFT);
        }
    }

    auto start = std::chrono::high_resolution_clock::now();
    long long checksum = 0;
    for (int index = 0; index < clones; index++)
    {
        SnakeEnv copy = source;
        checksum += copy.body().size();
    }
    double seconds = secondsSince(start);

    std::cout << "  " << board_size << "x" << board_size << ", snake of " << source.body().size()
              << ": " << (long long)(clones / seconds) << " clones/s  (checksum "
              << (checksum > 0 ? 1 : 0) << ")" << std::endl;
}

}  // namespace

int main()
{
    std::cout << "=== Throughput ===" << std::endl << std::endl;

    std::cout << "Simulator (single thread)" << std::endl;
    measureEnvironment(6, 1024, 2000);
    measureEnvironment(20, 1024, 2000);

    std::cout << std::endl << "Game cloning (one per search descent)" << std::endl;
    measureCloning(6, 2000000);
    measureCloning(20, 2000000);

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
