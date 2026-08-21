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
    config.exploration_epsilon = az::EXPLORATION_EPSILON;
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

// The same search read under both label rules, so a difference between them is the rule
// and never the sample.
//
// all_or_nothing is the rule the trainer uses today: a position contributes three pairs
// only if the search entered all three root actions, and nothing at all otherwise.
// per_visited_action keeps the pairs the search actually measured and drops only the
// individual actions it never entered. That is a narrower thing than the one-sided loss
// of Veness et al. 2009 - theirs penalises against a bound the search produced, and an
// action nothing entered has no bound to be one-sided about.
//
// The two rejection counters differ in kind: all_or_nothing counts whole positions,
// per_visited_action counts single actions. Neither is comparable to the other.
struct ProbeSamples
{
    DeathProbeSamples all_or_nothing;
    DeathProbeSamples per_visited_action;
};

// Plays settings.games held-out games with the search and pairs, at every position, the
// network's raw death output for an action against the risk the search backed up for that
// same action. Each position is searched exactly once and read under both label rules.
//
// The pairing is what the whole program is for: a head that learned something ranks the
// actions the way the search does, and one that learned nothing emits a constant.
//
// The head is read straight off the network rather than through NetworkEvaluator, which
// substitutes zero for the death output while az::DEATH_RISK_FROM_NETWORK is false. A
// probe reading it through the evaluator scores a vector of zeros and calls every head
// dead, trained or not.
//
// Games come from the reserved evaluation seed band via seeds::evaluationGameSeed, so a
// checkpoint is never probed on positions it trained on. The moves played are the
// search's own argmax, which means a weak network is probed on the short games it plays
// and a strong one on crowded boards - the population is the agent's, not a fixed set.
//
// Example, from a real run on az10_death368.pt at 10x10, 2 games, 200 simulations,
// seed offset 0. Both counts are of the same 2092 positions:
//
//     ProbeSamples samples =
//         collectSamples(trained, device, settings, az::deriveStepLimit(settings.board));
//
//     samples.all_or_nothing.pairs.size();      // 87   - 2063 positions rejected
//     samples.all_or_nothing.rejected;          // 2063
//     samples.per_visited_action.pairs.size();  // 2405 - 3871 actions rejected
//     samples.per_visited_action.rejected;      // 3871
//
// 87 / 3 + 2063 and (2405 + 3871) / 3 both come to 2092, which is the cheapest check
// that the two rules read one walk rather than two.
//
// Args:
//     network      the weights under test. Not modified, and must outlive the call.
//     device       where the forward passes run; cuda when one is available.
//     settings     games, board size, simulations per move, and the seed offset. The
//                  threshold it also carries is for scoring, and is not read here.
//     step_limit   moves after which a game is abandoned, so a policy that circles
//                  safely forever cannot stall the probe.
//
// Refuses nothing: every argument that could be wrong is checked at parseSettings, and a
// board or checkpoint mismatch fails earlier still, at load.
//
// Cost is one search per move of every game, which dominates everything else here by
// roughly three orders of magnitude. Expect minutes, not seconds.
ProbeSamples collectSamples(AlphaZeroNet& network, torch::Device device,
                            const ProbeSettings& settings, int step_limit)
{
    // One evaluator and one search for the whole walk: the search reuses its buffers, and
    // a fresh one per position would pay the allocation on every move.
    NetworkEvaluator evaluator(network, device);
    MonteCarloSearch search(evaluator, searchConfig(settings));

    ProbeSamples samples;
    // Each game gets its own seed from the reserved band, so a run is reproducible.
    for (int index = 0; index < settings.games; index++)
    {
        SnakeEnv game(settings.board, settings.board,
                      seeds::evaluationGameSeed(settings.seed_offset, index), step_limit);
        // Both conditions: a game can end by dying or by exhausting the step limit.
        while (!game.done() && game.steps() < step_limit)
        {
            // One root at a time. Batching games would pair a head reading with the
            // wrong search result unless the indices were tracked, for no measurable gain
            // at this scale.
            const std::vector<const SnakeEnv*> roots{ &game };
            const std::vector<MonteCarloSearch::Result> results = search.search(roots);
            const MonteCarloSearch::Result& result = results.front();

            // The head, read straight off the network. Encoding one position rather than
            // batching: the search dominates the cost by three orders of magnitude and a
            // batched probe would only complicate the pairing.
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

            // The trainer's rule: all three actions or none of them.
            if (!result.allActionsVisited())
            {
                samples.all_or_nothing.rejected++;
            }
            else
            {
                for (int action = 0; action < SnakeEnv::ACTION_COUNT; action++)
                {
                    samples.all_or_nothing.add(head_data[action],
                                               result.death_risk[action].value());
                }
            }

            // The per-action rule, over the same result: keep what was measured.
            for (int action = 0; action < SnakeEnv::ACTION_COUNT; action++)
            {
                // An empty risk is the search saying it never entered the action, which
                // is the whole reason the field is an optional rather than a number.
                if (result.death_risk[action].has_value())
                {
                    samples.per_visited_action.add(head_data[action],
                                                   result.death_risk[action].value());
                }
                else
                {
                    samples.per_visited_action.rejected++;
                }
            }
            // The search's own move, so the probe follows the policy under test.
            game.step(result.best_action);
        }
    }
    return samples;
}

void report(const std::string& label, const std::string& rejected_unit,
            const DeathProbeSamples& samples, float threshold)
{
    std::cout << std::format("\n{}\n", label);
    if (samples.pairs.size() < 2)
    {
        std::cout << std::format("  too few admissible pairs to score: {} kept, {} {} rejected\n",
                                 samples.pairs.size(), samples.rejected, rejected_unit);
        return;
    }
    const DeathProbeReport scored = scoreDeathProbe(samples, threshold);
    std::cout << std::format("  pairs            {} kept, {} {} rejected\n", scored.sample_count,
                             samples.rejected, rejected_unit);
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
    const ProbeSamples trained_samples = collectSamples(trained, device, settings, step_limit);

    // The floor. Same architecture, same games, no training: whatever this scores is
    // what the measurement returns for a head that has learned nothing.
    AlphaZeroNet untrained(settings.board, settings.board, settings.channels, settings.blocks);
    untrained->to(device);
    untrained->eval();
    const ProbeSamples untrained_samples = collectSamples(untrained, device, settings, step_limit);

    report("TRAINED HEAD, all-or-nothing rule - what the trainer sees", "positions",
           trained_samples.all_or_nothing, settings.threshold);
    report("UNTRAINED HEAD, all-or-nothing rule - the noise floor", "positions",
           untrained_samples.all_or_nothing, settings.threshold);
    report("TRAINED HEAD, per-visited-action rule - route two", "actions",
           trained_samples.per_visited_action, settings.threshold);
    report("UNTRAINED HEAD, per-visited-action rule - its noise floor", "actions",
           untrained_samples.per_visited_action, settings.threshold);

    const auto finished = std::chrono::high_resolution_clock::now();
    const double seconds = std::chrono::duration<double>(finished - started).count();
    std::cout << std::format("\nelapsed {:.2f}s\n", seconds);

    // The two runs play different games: the untrained network steers its own search,
    // so its positions are not the trained network's. The floor bounds what this
    // architecture scores untrained; it is not a paired comparison.
    std::cout << "the two rows are not paired - each network played its own games\n";
    return 0;
}
