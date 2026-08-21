// AlphaZeroDeathProbe: has the death head learned anything, and is it worth using?
//
// A diagnostic, not part of training or evaluation. It scores the head directly instead of
// reading the training curve of the run that produced it, which is the only way to tell a
// head that learned nothing from one that was never given anything to learn from.
//
// What it pairs. The death head predicts, for each of the three moves out of a position,
// the chance that move leads to a death no later play can avoid. The probe plays held-out
// games with the search and, at every position along them, records two numbers for each
// move: what the head said, and what the search backed up for that same move after its
// simulations. The search's number is the reference - not truth, but hundreds of lookaheads
// against the head's single forward pass - so a head that learned something ranks the moves
// the way the search does, and one that learned nothing does not.
//
// Read the floor first. Every run also scores a freshly initialised network of the same
// shape. An untrained head emits nearly the same number everywhere, and a rank correlation
// computed over a near-constant vector is normalisation noise rather than a weak signal. A
// trained number that does not clear the untrained one is not evidence of learning, it is
// evidence of the architecture. The head's spread is printed above its correlation so the
// near-constant case is visible before the number that would misrepresent it.
//
// It reads the head off the network, not through NetworkEvaluator. The evaluator
// substitutes zero for this output while az::DEATH_RISK_FROM_NETWORK is off, so a probe
// going through it scores a vector of zeros and calls every head dead, trained or not.
//
// Two label rules, four reports. The trainer keeps a position's labels only when the search
// visited all three moves - a move nobody tried has no evidence attached. The looser rule
// keeps every move the search did visit, whatever the others did. Both are scored on the
// same positions, for the trained head and for the untrained one, so the four blocks are
// the two rules crossed with the two networks, and the pair counts show what the looser
// rule would yield.
//
// Usage:
//
//     AlphaZeroDeathProbe.exe --checkpoint az10_death368.pt --board 10 \
//       --games 32 --simulations 200 --seed-offset 0 --threshold 0.5
//
//     # --games 32        held-out games; every move of every position contributes a pair
//     # --simulations 200 search budget per move, matched to evaluation
//     # --seed-offset 0   offset into the reserved evaluation seed band
//     # --threshold 0.5   where the search's number counts as doomed, for the ROC only
//
// Output, per block: how many pairs were kept and how many rejected, the head's mean and
// spread, its rank correlation against the search's continuous number, and the area under
// the ROC curve with the search's number split at --threshold. That threshold changes the
// ROC figure and nothing else - the correlation uses the continuous number. The ROC figure
// is undefined when every target falls on one side of the split, and says so rather than
// printing a value nobody can read.

#include <torch/torch.h>

#include <chrono>
#include <format>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "az_network.h"
#include "az_parameters.h"
#include "death_probe.h"
#include "flag_parser.h"
#include "mcts.h"
#include "network_evaluator.h"
#include "network_setup.h"
#include "seed_policy.h"
#include "snake_env.h"

namespace
{

// Which network's death head to score, on how many held-out games, at what search
// budget. Filled by parseSettings from the command line.
//
//     const ProbeSettings settings =
//         parseSettings(std::vector<std::string>(argv + 1, argv + argc));
//
//     const ProbeSamples samples =
//         collectSamples(network, device, settings, az::deriveStepLimit(settings.board));
//     report("trained head", "positions", samples.all_or_nothing, settings.threshold);
struct ProbeSettings
{
    // The network whose head is scored. Required.
    std::string checkpoint;
    // Side of the square board the games are played on.
    int board{ 10 };
    // Trunk width, in convolution channels. Must match the checkpoint, or loading fails.
    int channels{ 64 };
    // Residual blocks in the trunk. Must match the checkpoint for the same reason.
    int blocks{ 4 };
    // Held-out games to play. Every move of each contributes pairs, so this is the
    // sample size only indirectly - a game that dies early contributes few.
    int games{ 32 };
    // Search simulations per move, matched to evaluation so what is scored is the agent.
    int simulations{ 200 };
    // Offset into the reserved evaluation seed band.
    unsigned int seed_offset{ 0 };
    // Where the search's backed-up risk counts as "doomed", for the area under the ROC
    // curve and nothing else. The rank correlation uses the continuous target, so this
    // changes one reported number and leaves the rest alone.
    float threshold{ 0.5f };
};

// Every flag this program accepts. C++ cannot switch on a string, so the command line is
// turned into one of these first and the switch below does the rest.
enum class Flag
{
    Checkpoint,
    Board,
    Channels,
    Blocks,
    Games,
    Simulations,
    SeedOffset,
    Threshold
};

// One spelling and the enumerator it names.
struct FlagName
{
    // As written on the command line, leading dashes included.
    std::string_view text;
    // What applySetting will do with it.
    Flag flag;
};

// The whole command line, in one place. Adding a flag is a row here, an enumerator above
// and a case below - and leaving out the case is a compiler diagnostic.
constexpr FlagName FLAG_NAMES[] = {
    { "--checkpoint", Flag::Checkpoint },
    { "--board", Flag::Board },
    { "--channels", Flag::Channels },
    { "--blocks", Flag::Blocks },
    { "--games", Flag::Games },
    { "--simulations", Flag::Simulations },
    { "--seed-offset", Flag::SeedOffset },
    { "--threshold", Flag::Threshold },
};

// Which Flag `text` names, or std::invalid_argument when it names none.
//
//     lookupFlag("--threshold")   // Flag::Threshold
//
// Linear over eight entries, which costs less than building a map for one pass.
Flag lookupFlag(std::string_view text)
{
    for (const FlagName& candidate : FLAG_NAMES)
    {
        if (candidate.text == text)
        {
            return candidate.flag;
        }
    }
    throw std::invalid_argument(std::format("unknown flag {}", text));
}

// Applies one parsed flag to the settings, or throws naming the flag when the value is
// not what that flag accepts.
//
//     applySetting(settings, Flag::Games, "--games", "32");   // settings.games == 32
//
// No default case on purpose: the value comes from lookupFlag and can only be an
// enumerator, so the switch is exhaustive and a new enumerator without a case is caught
// at compile time rather than falling through in silence.
void applySetting(ProbeSettings& settings, Flag flag, std::string_view name, std::string_view value)
{
    switch (flag)
    {
        case Flag::Checkpoint:
        {
            settings.checkpoint = std::string(value);
            break;
        }
        case Flag::Board:
        {
            settings.board = flags::parseWholeInt(name, value);
            break;
        }
        case Flag::Channels:
        {
            settings.channels = flags::parseWholeInt(name, value);
            break;
        }
        case Flag::Blocks:
        {
            settings.blocks = flags::parseWholeInt(name, value);
            break;
        }
        case Flag::Games:
        {
            settings.games = flags::parseWholeInt(name, value);
            break;
        }
        case Flag::Simulations:
        {
            settings.simulations = flags::parseWholeInt(name, value);
            break;
        }
        case Flag::SeedOffset:
        {
            settings.seed_offset = flags::parseWholeUnsigned(name, value);
            break;
        }
        case Flag::Threshold:
        {
            // parseUnitFloat, not std::stof: stof stops at the first character it cannot
            // use, so "0.5x" read as 0.5, and a value outside [0, 1] was refused by
            // scoreDeathProbe only after every game had been played.
            settings.threshold = flags::parseUnitFloat(name, value);
            break;
        }
    }
}

// Everything that has to hold before a checkpoint is read or a game is played.
//
// The board, channel and block bounds are checked here rather than left to the network
// constructor, so a rejection names the flag the operator typed. The seed guard is the
// one that matters most: an offset running past the reserved band wraps into training
// seeds, which is silent and turns a held-out measurement into one that is not.
void requireUsable(const ProbeSettings& settings)
{
    if (settings.checkpoint.empty())
    {
        throw std::invalid_argument("--checkpoint is required");
    }
    flags::requireAtLeast("--board", settings.board, 2);
    if (settings.board > az::LARGEST_BOARD)
    {
        throw std::invalid_argument(
            std::format("--board must be at most {}, got {}", az::LARGEST_BOARD, settings.board));
    }
    flags::requireAtLeast("--channels", settings.channels, 1);
    // Zero is legal: a trunk with no residual blocks is shallow, not broken.
    flags::requireAtLeast("--blocks", settings.blocks, 0);
    flags::requireAtLeast("--games", settings.games, 1);
    flags::requireAtLeast("--simulations", settings.simulations, 1);

    const long long last_seed_index =
        static_cast<long long>(settings.seed_offset) + settings.games - 1;
    if (last_seed_index >= static_cast<long long>(seeds::RESERVED_BAND_WIDTH))
    {
        throw std::invalid_argument(
            std::format("--seed-offset {} with {} games runs past the reserved evaluation band "
                        "of {} seeds and wraps into the training range",
                        settings.seed_offset, settings.games, seeds::RESERVED_BAND_WIDTH));
    }
}

// Parses the command line, or throws std::invalid_argument naming the flag.
//
//     const ProbeSettings settings =
//         parseSettings(std::vector<std::string>(argv + 1, argv + argc));
//
// `arguments` excludes argv[0]. Requires --checkpoint. Rejects an unknown flag, a board
// outside 2..az::LARGEST_BOARD, non-positive channels, games or simulations, negative
// blocks, a --threshold outside [0, 1] or carrying trailing text, and a seed offset whose
// games would run past the reserved evaluation band.
ProbeSettings parseSettings(std::span<const std::string> arguments)
{
    ProbeSettings settings;
    for (const flags::FlagValue& entry : flags::readFlags(arguments))
    {
        applySetting(settings, lookupFlag(entry.flag), entry.flag, entry.value);
    }
    requireUsable(settings);
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
// The same positions scored under both label rules, from one walk of the games.
//
// Both are filled together so the two rules are never compared across different
// populations - which is what would happen if each were collected by its own run.
struct ProbeSamples
{
    // Pairs kept under the rule the trainer uses today: a position contributes every
    // action, or nothing, according to whether the search visited all of them.
    DeathProbeSamples all_or_nothing;
    // Pairs kept under the looser rule: every action the search visited contributes,
    // whatever the others did. The difference between the two is what changing the
    // rule would yield at the same search cost.
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

// Prints one scored block under `label`, as an indented report.
//
//     report("trained head", "positions", samples.all_or_nothing, settings.threshold);
//
// `rejected_unit` names what the rejected count counts - positions under the
// all-or-nothing rule, actions under the per-action rule - because the two rules reject
// different things and one word for both would misread by a factor of three.
//
// Says so and returns without scoring when fewer than two pairs survived: a correlation
// over one point is not a weak result, it is not a result. Prints the head's spread
// before its correlation, because a rank correlation over a near-constant vector is
// normalisation noise rather than a weak signal.
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
    const Compute compute = chooseDevice();
    const torch::Device device = compute.device;
    const bool cuda = compute.cuda;

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
        loadCheckpoint(trained, settings.checkpoint, "");
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << std::endl;
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
