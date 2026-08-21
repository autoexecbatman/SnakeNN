// AlphaZeroCoverage: how often does the search visit every root action, with root noise
// and without?
//
// A death-risk label is trainable only on a position where every root action was visited.
// The probe measured 4 percent coverage on a trained network - but it ran with noise off,
// as evaluation does, while self-play runs with it on (az_trainer.cpp sets
// root_noise_fraction to az::ROOT_NOISE_FRACTION). Dirichlet noise at the root is exactly
// what spreads visits across actions, so whether the death head is starved during
// training turns on this comparison and on nothing else.
//
// The positions are fixed before the arms run. One trajectory is played, every root
// position along it is stored, and both arms then search those same positions without
// playing a move. Arms that each played their own games would compare different
// populations, because a change to the search changes which move is played.
//
// Usage:
//
//     AlphaZeroCoverage.exe --checkpoint az10_death368.pt --board 10 \
//       --games 2 --simulations 200 --max-positions 300 --seed-offset 0
//
//     # --games 2          held-out games played once to generate the positions
//     # --max-positions    cap, sampled evenly along the trajectory; 0 means all
//     # --simulations 200  budget for both arms, so noise is the only difference
//
// It prints coverage and label yield for each arm, under the current all-or-nothing rule
// and under a per-action rule that would keep every visited action. Read the ratio: it is
// what a redesign of the rule would buy at the same search cost.

#include <torch/torch.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <format>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "az_network.h"
#include "az_parameters.h"
#include "coverage_tally.h"
#include "flag_parser.h"
#include "mcts.h"
#include "network_evaluator.h"
#include "seed_policy.h"
#include "snake_env.h"

namespace
{

struct CoverageSettings
{
    std::string checkpoint;
    int board{ 10 };
    int channels{ 64 };
    int blocks{ 4 };
    int games{ 2 };
    int simulations{ 200 };
    int max_positions{ 300 };
    unsigned int seed_offset{ 0 };
    bool skip_arms{ false };
    // Whose play generates the positions. Empty means the checkpoint under test.
    // Set it to hold the population fixed while the network being read changes -
    // otherwise a weak checkpoint is measured on the short games it plays and a
    // strong one on crowded boards, and the difference is the population.
    std::string position_checkpoint;
};

CoverageSettings parseSettings(std::span<const std::string> arguments)
{
    CoverageSettings settings;
    for (const flags::FlagValue& entry : flags::readFlags(arguments))
    {
        if (entry.flag == "--checkpoint")
        {
            settings.checkpoint = std::string(entry.value);
        }
        else if (entry.flag == "--board")
        {
            settings.board = flags::parseWholeInt(entry.flag, entry.value);
        }
        else if (entry.flag == "--channels")
        {
            settings.channels = flags::parseWholeInt(entry.flag, entry.value);
        }
        else if (entry.flag == "--blocks")
        {
            settings.blocks = flags::parseWholeInt(entry.flag, entry.value);
        }
        else if (entry.flag == "--games")
        {
            settings.games = flags::parseWholeInt(entry.flag, entry.value);
        }
        else if (entry.flag == "--simulations")
        {
            settings.simulations = flags::parseWholeInt(entry.flag, entry.value);
        }
        else if (entry.flag == "--max-positions")
        {
            settings.max_positions = flags::parseWholeInt(entry.flag, entry.value);
        }
        else if (entry.flag == "--seed-offset")
        {
            settings.seed_offset = flags::parseWholeUnsigned(entry.flag, entry.value);
        }
        else if (entry.flag == "--skip-arms")
        {
            settings.skip_arms = entry.value == "on";
        }
        else if (entry.flag == "--position-checkpoint")
        {
            settings.position_checkpoint = std::string(entry.value);
        }
        else
        {
            throw std::invalid_argument(std::format("unknown flag {}", entry.flag));
        }
    }
    if (settings.checkpoint.empty())
    {
        throw std::invalid_argument("--checkpoint is required");
    }
    flags::requireAtLeast("--games", settings.games, 1);
    flags::requireAtLeast("--simulations", settings.simulations, 1);
    flags::requireAtLeast("--max-positions", settings.max_positions, 0);
    return settings;
}

// Arms differ only in the noise fraction, the exploration constant and the exploration
// floor, which are the three things that can widen a root. Everything else is held fixed.
//
// All three are parameters rather than fields of the settings the arms share, so a
// treatment cannot reach the control or the trajectory that generates the positions.
MonteCarloSearch::Config armConfig(const CoverageSettings& settings, float noise_fraction,
                                   float exploration, float exploration_epsilon)
{
    MonteCarloSearch::Config config;
    config.simulations = settings.simulations;
    config.exploration = exploration;
    config.discount = az::DISCOUNT;
    config.step_reward = az::STEP_REWARD;
    config.steps_tiebreak_margin = az::STEPS_TIEBREAK_MARGIN;
    config.trap_guard = false;
    config.trap_report = az::TRAP_REPORT;
    config.average_edges = false;
    config.normalize_values = az::NORMALIZE_VALUES;
    config.exploration_epsilon = exploration_epsilon;
    config.death_cap = false;
    config.death_cap_threshold = az::DEATH_CAP_THRESHOLD;
    config.alias_report = false;
    config.root_noise_fraction = noise_fraction;
    config.root_noise_alpha = az::ROOT_NOISE_ALPHA;
    // Fixed across arms so the only difference is the noise fraction. The stream still
    // advances differently once noise draws from it, which is why the arms are compared
    // on coverage rates rather than position by position.
    config.seed = seeds::evaluationGameSeed(settings.seed_offset, 0);
    return config;
}

// Plays the games once at evaluation settings and returns every root position along the
// way. These are the fixed population both arms are measured on.
std::vector<SnakeEnv> collectPositions(AlphaZeroNet& network, torch::Device device,
                                       const CoverageSettings& settings, int step_limit)
{
    NetworkEvaluator evaluator(network, device);
    // Floor off: the positions are the population every arm is measured on, so a floor
    // here would move what is being measured along with the thing measuring it.
    MonteCarloSearch search(evaluator, armConfig(settings, 0.0f, az::EXPLORATION, 0.0f));

    std::vector<SnakeEnv> positions;
    for (int index = 0; index < settings.games; index++)
    {
        SnakeEnv game(settings.board, settings.board,
                      seeds::evaluationGameSeed(settings.seed_offset, index), step_limit);
        while (!game.done() && game.steps() < step_limit)
        {
            positions.push_back(game);
            const std::vector<const SnakeEnv*> roots{ &game };
            const std::vector<MonteCarloSearch::Result> results = search.search(roots);
            game.step(results.front().best_action);
        }
    }
    return positions;
}

// Evenly spaced rather than the first N: a prefix is all opening positions, where the
// board is empty and coverage is easiest, which would flatter every arm equally and
// hide the effect the whole run is looking for.
std::vector<SnakeEnv> sampleEvenly(const std::vector<SnakeEnv>& positions, int cap)
{
    if (cap == 0 || positions.size() <= static_cast<std::size_t>(cap))
    {
        return positions;
    }
    std::vector<SnakeEnv> sampled;
    sampled.reserve(static_cast<std::size_t>(cap));
    for (int index = 0; index < cap; index++)
    {
        const std::size_t source = static_cast<std::size_t>(static_cast<double>(index) *
                                                            static_cast<double>(positions.size()) /
                                                            static_cast<double>(cap));
        sampled.push_back(positions[source]);
    }
    return sampled;
}

struct ArmResult
{
    CoverageTally tally;
    // Times the per-action count disagreed with the search's own coverage flag. A
    // non-zero value means one of the two readings is wrong and the report is not
    // trustworthy - printed rather than asserted so a long run still finishes.
    std::size_t flag_disagreements{ 0 };
};

// Searches every position without playing a move, so the population is identical across
// arms by construction.
ArmResult measureArm(AlphaZeroNet& network, torch::Device device, const CoverageSettings& settings,
                     float noise_fraction, float exploration, float exploration_epsilon,
                     const std::vector<SnakeEnv>& positions)
{
    NetworkEvaluator evaluator(network, device);
    MonteCarloSearch search(evaluator,
                            armConfig(settings, noise_fraction, exploration, exploration_epsilon));

    ArmResult arm;
    for (const SnakeEnv& position : positions)
    {
        const std::vector<const SnakeEnv*> roots{ &position };
        const std::vector<MonteCarloSearch::Result> results = search.search(roots);
        const MonteCarloSearch::Result& result = results.front();

        // An action with no visits has no share of the visit distribution.
        int visited = 0;
        for (int action = 0; action < SnakeEnv::ACTION_COUNT; action++)
        {
            if (result.policy[static_cast<std::size_t>(action)] > 0.0f)
            {
                visited++;
            }
        }
        if ((visited == SnakeEnv::ACTION_COUNT) != result.all_actions_visited)
        {
            arm.flag_disagreements++;
        }
        arm.tally.observe(visited, SnakeEnv::ACTION_COUNT);
    }
    return arm;
}

void report(const std::string& label, const ArmResult& arm)
{
    const CoverageReport scored = scoreCoverage(arm.tally);
    std::cout << std::format("\n{}\n", label);
    std::cout << std::format("  positions            {}\n", scored.positions);
    std::cout << std::format("  fully covered        {:.4f}\n", scored.position_coverage);
    std::cout << std::format("  mean actions visited {:.4f} of {}\n", scored.mean_visited_actions,
                             SnakeEnv::ACTION_COUNT);
    std::cout << std::format("  labels, current rule {}\n", scored.labels_all_or_nothing);
    std::cout << std::format("  labels, per action   {}\n", scored.labels_per_action);
    if (scored.yield_ratio_defined)
    {
        std::cout << std::format("  yield ratio          {:.4f}\n", scored.yield_ratio);
    }
    else
    {
        std::cout << "  yield ratio          unbounded - the current rule admits nothing\n";
    }
    if (arm.flag_disagreements > 0)
    {
        std::cout << std::format(
            "  WARNING              {} positions where the visit count and the search "
            "coverage flag disagree; this report is not trustworthy\n",
            arm.flag_disagreements);
    }
}

}  // namespace

int main(int argc, char** argv)
{
    CoverageSettings settings;
    try
    {
        settings = parseSettings(std::vector<std::string>(argv + 1, argv + argc));
    }
    catch (const std::exception& error)
    {
        std::cerr << "bad arguments: " << error.what() << std::endl;
        return 1;
    }

    const int step_limit = az::deriveStepLimit(settings.board);
    const bool cuda = torch::cuda::is_available();
    torch::Device device = cuda ? torch::Device(torch::kCUDA) : torch::Device(torch::kCPU);

    AlphaZeroNet network(settings.board, settings.board, settings.channels, settings.blocks);
    try
    {
        const std::vector<std::string> missing = network->loadNarrowerStem(settings.checkpoint);
        for (const std::string& name : missing)
        {
            std::cout << std::format("  fresh, absent from the checkpoint: {}\n", name);
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "could not load " << settings.checkpoint << ": " << error.what() << std::endl;
        return 1;
    }
    network->to(device);
    network->eval();

    std::cout << std::format("coverage: {} on {}x{}, {} games, {} simulations, device {}\n",
                             settings.checkpoint, settings.board, settings.board, settings.games,
                             settings.simulations, cuda ? "cuda" : "cpu");

    const auto started = std::chrono::high_resolution_clock::now();

    // Positions come from whichever network is named for the job. Comparing several
    // checkpoints means holding this one fixed, so every prior is read on the same
    // boards and the difference is the policy rather than the population.
    AlphaZeroNet position_network(settings.board, settings.board, settings.channels,
                                  settings.blocks);
    AlphaZeroNet* position_source = &network;
    if (!settings.position_checkpoint.empty())
    {
        try
        {
            const std::vector<std::string> missing =
                position_network->loadNarrowerStem(settings.position_checkpoint);
            for (const std::string& name : missing)
            {
                std::cout << std::format("  positions: fresh, absent from {}: {}\n",
                                         settings.position_checkpoint, name);
            }
        }
        catch (const std::exception& error)
        {
            std::cerr << "could not load " << settings.position_checkpoint << ": " << error.what()
                      << std::endl;
            return 1;
        }
        position_network->to(device);
        position_network->eval();
        position_source = &position_network;
        std::cout << std::format("positions generated by {}\n", settings.position_checkpoint);
    }

    const std::vector<SnakeEnv> played =
        collectPositions(*position_source, device, settings, step_limit);
    const std::vector<SnakeEnv> positions = sampleEvenly(played, settings.max_positions);
    std::cout << std::format("{} positions played, {} sampled evenly along the trajectory\n",
                             played.size(), positions.size());
    if (positions.empty())
    {
        std::cerr << "no positions were generated" << std::endl;
        return 1;
    }

    // How many actions were survivable at all, before any search ran. This is the
    // denominator every coverage number above is really against: an action that kills on
    // this tick cannot be visited by a search that prunes fatal moves, and no exploration
    // constant can recover it. Free - wouldDie copies nothing.
    std::size_t survivable_histogram[SnakeEnv::ACTION_COUNT + 1] = { 0, 0, 0, 0 };
    for (const SnakeEnv& position : played)
    {
        int survivable = 0;
        for (int action = 0; action < SnakeEnv::ACTION_COUNT; action++)
        {
            if (!position.wouldDie(static_cast<SnakeEnv::Action>(action)))
            {
                survivable++;
            }
        }
        survivable_histogram[survivable]++;
    }
    std::cout << "\nSURVIVABLE ROOT ACTIONS - measured without any search\n";
    for (int count = 0; count <= SnakeEnv::ACTION_COUNT; count++)
    {
        std::cout << std::format(
            "  {} survivable   {:6} positions  {:.4f}\n", count, survivable_histogram[count],
            static_cast<double>(survivable_histogram[count]) / static_cast<double>(played.size()));
    }

    // The priors themselves, straight off the network - no search. If the top prior is
    // saturated then c_puct * prior * sqrt(N) is negligible for every other action no
    // matter how large c_puct is, which is the one story consistent with a hundredfold
    // sweep of that constant changing nothing.
    {
        double top_total = 0.0;
        double entropy_total = 0.0;
        std::size_t above_99 = 0;
        std::size_t above_999 = 0;
        double lowest_top = 1.0;
        for (const SnakeEnv& position : played)
        {
            std::vector<float> planes(static_cast<std::size_t>(position.encodedSize()));
            position.encode(planes.data());
            torch::NoGradGuard no_grad;
            const torch::Tensor input =
                torch::from_blob(planes.data(),
                                 { 1, SnakeEnv::PLANE_COUNT, settings.board, settings.board },
                                 torch::kFloat)
                    .to(device);
            const torch::Tensor probabilities =
                torch::softmax(network->forward(input).policy_logits, 1)
                    .to(torch::kCPU)
                    .contiguous();
            const float* prior = probabilities.data_ptr<float>();

            double top = 0.0;
            double entropy = 0.0;
            for (int action = 0; action < SnakeEnv::ACTION_COUNT; action++)
            {
                top = std::max(top, static_cast<double>(prior[action]));
                if (prior[action] > 0.0f)
                {
                    entropy -= static_cast<double>(prior[action]) *
                               std::log(static_cast<double>(prior[action]));
                }
            }
            top_total += top;
            entropy_total += entropy;
            lowest_top = std::min(lowest_top, top);
            if (top > 0.99)
            {
                above_99++;
            }
            if (top > 0.999)
            {
                above_999++;
            }
        }
        const double count = static_cast<double>(played.size());
        std::cout << "\nPOLICY PRIORS - straight off the network, no search\n";
        std::cout << std::format("  mean top prior      {:.6f}\n", top_total / count);
        std::cout << std::format("  lowest top prior    {:.6f}\n", lowest_top);
        std::cout << std::format("  mean entropy        {:.6f}  (uniform over 3 is {:.6f})\n",
                                 entropy_total / count, std::log(3.0));
        std::cout << std::format("  top prior > 0.99    {:.4f}\n",
                                 static_cast<double>(above_99) / count);
        std::cout << std::format("  top prior > 0.999   {:.4f}\n",
                                 static_cast<double>(above_999) / count);
    }

    if (settings.skip_arms)
    {
        std::cout << "\n--skip-arms given; no search arms were run\n";
        return 0;
    }

    // The two configurations the project actually runs, then the exploration constant
    // swept upward. c_puct enters the score as c * prior * sqrt(N) while values live in
    // (-VALUE_SCALE, VALUE_SCALE), so 0.5 - the paper's number, chosen for values in
    // [-1, 1] - is the suspect. If coverage climbs with it, the constant is the cause.
    const ArmResult baseline =
        measureArm(network, device, settings, 0.0f, az::EXPLORATION, 0.0f, positions);
    report("NOISE OFF, c_puct 0.5 - evaluation conditions", baseline);

    const ArmResult noisy = measureArm(network, device, settings, az::ROOT_NOISE_FRACTION,
                                       az::EXPLORATION, 0.0f, positions);
    report(
        std::format("NOISE ON at {}, c_puct 0.5 - self-play conditions", az::ROOT_NOISE_FRACTION),
        noisy);

    // The exploration floor, which is the arm this program was extended for. c_puct is
    // left alone: sweeping it was already measured to move coverage from 1.13 to 1.24
    // actions, because every term it appears in multiplies the prior.
    for (const float epsilon : { 0.05f, 0.1f, 0.3f })
    {
        const ArmResult swept =
            measureArm(network, device, settings, 0.0f, az::EXPLORATION, epsilon, positions);
        report(std::format("NOISE OFF, exploration floor epsilon {}", epsilon), swept);
    }

    const auto finished = std::chrono::high_resolution_clock::now();
    std::cout << std::format("\nelapsed {:.2f}s\n",
                             std::chrono::duration<double>(finished - started).count());
    std::cout << "both arms searched the same positions; no move was played from either\n";
    return 0;
}
