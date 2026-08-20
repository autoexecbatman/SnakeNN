// AlphaZeroDeathProbe: does the trained death head contain anything?
//
// az10_death368.pt cost 4.8 hours and its logs cannot say, because the death loss was
// never printed. This scores the head directly instead of the run that produced it:
// it plays held-out games with the search, and for every root position pairs the
// network's raw death output against the risk the search backed up for the same action.
//
// The head is read from the network, not through NetworkEvaluator, which zeroes this
// output whenever az::DEATH_RISK_FROM_NETWORK is false - as it is. A probe reading
// through the evaluator scores a vector of zeros and reports a dead head either way.
//
// Usage:
//
//     AlphaZeroDeathProbe.exe --checkpoint az10_death368.pt --board 10 \
//       --games 32 --simulations 200 --seed-offset 0 --threshold 0.5
//
//     # --games 32        held-out games; every move of each contributes pairs
//     # --simulations 200 search budget per move, as in evaluation
//     # --threshold 0.5   binarises the search target for the AUC only
//     # --seed-offset 0   offset into the reserved evaluation seed band
//
// It prints two reports. The trained head, and the same measurement against a freshly
// initialised network of the same shape - the noise floor. An untrained head scores
// whatever this architecture scores by construction, and a trained number that does not
// clear it is not evidence of learning. Read the floor before the result.

#include <torch/torch.h>

#include <chrono>
#include <format>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "az_network.h"
#include "az_parameters.h"
#include "death_probe.h"
#include "flag_parser.h"
#include "mcts.h"
#include "network_evaluator.h"
#include "seed_policy.h"
#include "snake_env.h"

namespace
{

struct ProbeSettings
{
    std::string checkpoint;
    int board{ 10 };
    int channels{ 64 };
    int blocks{ 4 };
    int games{ 32 };
    int simulations{ 200 };
    unsigned int seed_offset{ 0 };
    float threshold{ 0.5f };
};

ProbeSettings parseSettings(std::span<const std::string> arguments)
{
    ProbeSettings settings;
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
        else if (entry.flag == "--seed-offset")
        {
            settings.seed_offset = flags::parseWholeUnsigned(entry.flag, entry.value);
        }
        else if (entry.flag == "--threshold")
        {
            settings.threshold = std::stof(std::string(entry.value));
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
    return settings;
}

// The search's own configuration, matched to evaluation rather than to self-play:
// no root noise, so what is scored is the agent and not the exploration policy.
MonteCarloSearch::Config searchConfig(const ProbeSettings& settings)
{
    MonteCarloSearch::Config config;
    config.simulations = settings.simulations;
    config.exploration = az::EXPLORATION;
    config.discount = az::DISCOUNT;
    config.step_reward = az::STEP_REWARD;
    config.steps_tiebreak_margin = az::STEPS_TIEBREAK_MARGIN;
    config.trap_guard = false;
    config.trap_report = az::TRAP_REPORT;
    config.average_edges = false;
    config.normalize_values = az::NORMALIZE_VALUES;
    // Off on purpose: the cap changes which move is played, and a probe that let it
    // fire would score the head against a search the head had already steered.
    config.death_cap = false;
    config.death_cap_threshold = az::DEATH_CAP_THRESHOLD;
    config.alias_report = false;
    config.root_noise_fraction = 0.0f;
    config.root_noise_alpha = az::ROOT_NOISE_ALPHA;
    config.seed = seeds::evaluationGameSeed(settings.seed_offset, 0);
    return config;
}

// Plays the held-out games and collects one pair per root action on every position the
// search covered fully. Returns the samples; the caller scores them.
DeathProbeSamples collectSamples(AlphaZeroNet& network, torch::Device device,
                                 const ProbeSettings& settings, int step_limit)
{
    NetworkEvaluator evaluator(network, device);
    MonteCarloSearch search(evaluator, searchConfig(settings));

    DeathProbeSamples samples;
    for (int index = 0; index < settings.games; index++)
    {
        SnakeEnv game(settings.board, settings.board,
                      seeds::evaluationGameSeed(settings.seed_offset, index), step_limit);
        while (!game.done() && game.steps() < step_limit)
        {
            const std::vector<const SnakeEnv*> roots{ &game };
            const std::vector<MonteCarloSearch::Result> results = search.search(roots);
            const MonteCarloSearch::Result& result = results.front();

            if (!result.all_actions_visited)
            {
                // An unvisited root action keeps its start value and reads as safe, so
                // a pair drawn here would measure the search rather than the head.
                samples.rejected_uncovered++;
            }
            else
            {
                // The head, read straight off the network. Encoding one position rather
                // than batching: the search dominates the cost by three orders of
                // magnitude and a batched probe would only complicate the pairing.
                std::vector<float> planes(static_cast<std::size_t>(game.encodedSize()));
                game.encode(planes.data());
                torch::NoGradGuard no_grad;
                const torch::Tensor input =
                    torch::from_blob(planes.data(),
                                     { 1, SnakeEnv::PLANE_COUNT, settings.board, settings.board },
                                     torch::kFloat)
                        .to(device);
                const Prediction prediction = network->forward(input);
                const torch::Tensor head = prediction.death_risk.to(torch::kCPU).contiguous();
                const float* head_data = head.data_ptr<float>();

                for (int action = 0; action < SnakeEnv::ACTION_COUNT; action++)
                {
                    samples.add(head_data[action], result.death_risk[action]);
                }
            }
            game.step(result.best_action);
        }
    }
    return samples;
}

void report(const std::string& label, const DeathProbeSamples& samples, float threshold)
{
    std::cout << std::format("\n{}\n", label);
    if (samples.pairs.size() < 2)
    {
        std::cout << std::format("  too few admissible pairs to score: {} kept, {} rejected\n",
                                 samples.pairs.size(), samples.rejected_uncovered);
        return;
    }
    const DeathProbeReport scored = scoreDeathProbe(samples, threshold);
    std::cout << std::format("  pairs            {} kept, {} positions rejected as uncovered\n",
                             scored.sample_count, samples.rejected_uncovered);
    std::cout << std::format("  head mean        {:.6f}\n", scored.estimate_mean);
    std::cout << std::format("  head spread      {:.6f}   <- read first\n", scored.estimate_spread);
    std::cout << std::format("  rank correlation {:.6f}\n", scored.rank_correlation);
    if (scored.ranking_auc_defined)
    {
        std::cout << std::format("  ranking auc      {:.6f}\n", scored.ranking_auc);
    }
    else
    {
        std::cout << "  ranking auc      undefined - every target fell on one side\n";
    }
    std::cout << std::format("  doomed fraction  {:.6f}\n", scored.doomed_fraction);
}

}  // namespace

int main(int argc, char** argv)
{
    ProbeSettings settings;
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

    std::cout << std::format("death probe: {} on {}x{}, {} games, {} simulations, threshold {}\n",
                             settings.checkpoint, settings.board, settings.board, settings.games,
                             settings.simulations, settings.threshold);
    std::cout << std::format("seeds {}.. in the reserved evaluation band; device {}\n",
                             seeds::evaluationGameSeed(settings.seed_offset, 0),
                             cuda ? "cuda" : "cpu");

    const auto started = std::chrono::high_resolution_clock::now();

    // The trained head.
    AlphaZeroNet trained(settings.board, settings.board, settings.channels, settings.blocks);
    try
    {
        const std::vector<std::string> missing = trained->loadNarrowerStem(settings.checkpoint);
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
    trained->to(device);
    trained->eval();
    const DeathProbeSamples trained_samples = collectSamples(trained, device, settings, step_limit);

    // The floor. Same architecture, same games, no training: whatever this scores is
    // what the measurement returns for a head that has learned nothing.
    AlphaZeroNet untrained(settings.board, settings.board, settings.channels, settings.blocks);
    untrained->to(device);
    untrained->eval();
    const DeathProbeSamples untrained_samples =
        collectSamples(untrained, device, settings, step_limit);

    report("TRAINED HEAD", trained_samples, settings.threshold);
    report("UNTRAINED HEAD - the noise floor", untrained_samples, settings.threshold);

    const auto finished = std::chrono::high_resolution_clock::now();
    const double seconds = std::chrono::duration<double>(finished - started).count();
    std::cout << std::format("\nelapsed {:.2f}s\n", seconds);

    // The two runs play different games: the untrained network steers its own search,
    // so its positions are not the trained network's. The floor bounds what this
    // architecture scores untrained; it is not a paired comparison.
    std::cout << "the two rows are not paired - each network played its own games\n";
    return 0;
}
