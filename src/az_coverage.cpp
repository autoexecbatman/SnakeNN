// AlphaZeroCoverage: does the search give the death head anything to learn from?
//
// A diagnostic, not part of training or evaluation. It answers that one question in
// minutes, so a change to the search can be priced before an hour of GPU is spent finding
// out the slow way.
//
// What it measures. When the search picks a move it has three options - straight, left and
// right, relative to the heading - and spends its simulations among them from the position
// the snake is actually in, the root of its tree. Coverage is the share of positions where
// all three got at least one simulation.
//
// It need not spread them evenly, and by construction it does not. Each simulation goes to
// whichever move scores highest on what it is worth so far plus an exploration term, and
// that exploration term is multiplied by how likely the network thinks the move is. A move
// the network has written off - a preference of a thousandth - therefore gets a thousandth
// of the push toward being tried, and the search can spend every simulation without once
// looking at it. The move is not rejected on evidence. It is never examined.
//
// That tightens as the agent improves. Training sharpens the policy, so the top preference
// climbs toward one and the other two collapse - and coverage falls exactly as the network
// gets better. The death head is fed least when the rest of the agent is doing best, which
// is why the starvation does not announce itself: every other number in the training log
// is improving while this one quietly goes to nothing.
//
// It also rules out the obvious remedy. Raising the exploration constant cannot help,
// because that constant multiplies the preference too - scaling up a thousandth scales up
// nothing. Anything that widens a root here has to add, not multiply.
//
// Why that number decides something. The death head predicts, per move, the chance the
// move leads to a death no later play can avoid. Its training target - the death label -
// is whatever the search backed up for that move, so a move nobody tried has no evidence
// attached and the trainer discards the whole position rather than label it wrongly. Low
// coverage therefore starves the head however long training runs, and no amount of
// training curve can tell you that is what is happening. This can.
//
// It measures several arms, an arm being one search configuration with everything else
// held fixed. Each is a setting that could spread the simulations wider: Dirichlet noise,
// which perturbs the network's move preferences at the root and which self-play uses but
// evaluation does not, and the exploration floor, which makes root selection ignore its
// own scores a fixed share of the time.
//
// It freezes the board. The games are played once, every root position along them is
// saved, and every arm then searches that same frozen set. No arm plays a move.
//
// Without that, a change to the search changes which moves get played, so each arm ends up
// looking at different boards. A difference in the numbers could then be the treatment or
// could be the boards, and nothing distinguishes them.
//
// One run measures one checkpoint. To compare checkpoints, run it once per checkpoint with
// the same --position-checkpoint, so all of them are scored on the same boards.
//
// Usage:
//
//     AlphaZeroCoverage.exe --checkpoint az10_long308.pt --board 10 --games 2
//        --simulations 200 --max-positions 300 --seed-offset 0
//
//     # --games 2          games played once, to generate the positions
//     # --max-positions    cap, sampled evenly along the trajectory; 0 means all
//     # --simulations 200  the budget every arm shares, so the setting is the difference
//     # --skip-arms on     print the two no-search measurements and stop
//     # --position-checkpoint  read positions from another network, to hold the
//     #                        population fixed while the network under test changes
//
// Prints two measurements taken with no search at all - how many moves were survivable,
// and how concentrated the network's preferences are - then one block per arm giving
// coverage and label yield under the current all-or-nothing rule and under a per-action
// rule that would keep every visited move. Read the yield ratio: it is what changing the
// rule would buy at the same search cost.

#include <torch/torch.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <format>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "az_network.h"
#include "az_parameters.h"
#include "coverage_tally.h"
#include "flag_parser.h"
#include "mcts.h"
#include "network_evaluator.h"
#include "network_setup.h"
#include "search_defaults.h"
#include "seed_policy.h"
#include "snake_env.h"

namespace
{

// Which network to read, which positions to read it on, and how hard to search them.
// Filled by parseSettings from the command line.
//
// Coverage is the share of positions where the search visited every root action. It is
// worth measuring because a death-risk label is trainable only where all three were
// visited - an unvisited action keeps its start value and reads as safe - so coverage
// is the fraction of positions that yield any label at all.
//
//     const CoverageSettings settings =
//         parseSettings(std::vector<std::string>(argv + 1, argv + argc));
//
//     // The positions every arm is scored on, so the population is identical across arms.
//     const std::vector<SnakeEnv> positions =
//         collectPositions(network, device, settings, step_limit);
//
//     // No root noise, the paper's exploration constant, no exploration floor.
//     const ArmResult baseline =
//         measureArm(network, device, settings, 0.0f, az::EXPLORATION, 0.0f, positions);
//     report("baseline", baseline);
//
// Two runs are comparable only when board, games, simulations and seed_offset all match:
// coverage is a function of how hard the search looked and of which positions it looked
// at, so a run changing either is measuring something else.
struct CoverageSettings
{
    // The network under test. Required.
    std::string checkpoint;
    // Side of the square board the positions are drawn on.
    int board{ 10 };
    // Trunk width and depth. Both must match the checkpoint, or loading it fails.
    int channels{ 64 };
    // Residual blocks in the trunk. Must match the checkpoint for the same reason.
    int blocks{ 4 };
    // Games played to collect positions from.
    int games{ 2 };
    // Search simulations per position. Coverage is a function of this: more
    // simulations reach more actions, so two runs at different counts are not
    // comparable.
    int simulations{ 200 };
    // Ceiling on positions kept. Zero keeps none.
    int max_positions{ 300 };
    // Offset into the reserved evaluation seed band.
    unsigned int seed_offset{ 0 };
    // Whether to skip the treatment arms and report the baseline alone. Set by
    // --skip-arms on; any other value reads as off.
    bool skip_arms{ false };
    // Whose play generates the positions. Empty means the checkpoint under test.
    // Set it to hold the population fixed while the network being read changes -
    // otherwise a weak checkpoint is measured on the short games it plays and a
    // strong one on crowded boards, and the difference is the population.
    std::string position_checkpoint;
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
    MaxPositions,
    SeedOffset,
    SkipArms,
    PositionCheckpoint
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
    { "--max-positions", Flag::MaxPositions },
    { "--seed-offset", Flag::SeedOffset },
    { "--skip-arms", Flag::SkipArms },
    { "--position-checkpoint", Flag::PositionCheckpoint },
};

// Which Flag `text` names, or std::invalid_argument when it names none.
//
//     lookupFlag("--board")   // Flag::Board
//
// Linear over ten entries, which costs less than building a map for one pass.
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
//     applySetting(settings, Flag::Board, "--board", "10");   // settings.board == 10
//
// No default case on purpose: the value comes from lookupFlag and can only be an
// enumerator, so the switch is exhaustive and a new enumerator without a case is caught
// at compile time rather than falling through in silence.
void applySetting(CoverageSettings& settings, Flag flag, std::string_view name,
                  std::string_view value)
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
        case Flag::MaxPositions:
        {
            settings.max_positions = flags::parseWholeInt(name, value);
            break;
        }
        case Flag::SeedOffset:
        {
            settings.seed_offset = flags::parseWholeUnsigned(name, value);
            break;
        }
        case Flag::SkipArms:
        {
            settings.skip_arms = flags::parseOnOff(name, value);
            break;
        }
        case Flag::PositionCheckpoint:
        {
            settings.position_checkpoint = std::string(value);
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
void requireUsable(const CoverageSettings& settings)
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
    flags::requireAtLeast("--max-positions", settings.max_positions, 0);

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
//     const CoverageSettings settings =
//         parseSettings(std::vector<std::string>(argv + 1, argv + argc));
//
// `arguments` excludes argv[0]. Requires --checkpoint. Rejects an unknown flag, a board
// outside 2..az::LARGEST_BOARD, non-positive channels, games or simulations, negative
// blocks or max-positions, a --skip-arms value that is not "on" or "off", and a seed
// offset whose games would run past the reserved evaluation band.
CoverageSettings parseSettings(std::span<const std::string> arguments)
{
    CoverageSettings settings;
    for (const flags::FlagValue& entry : flags::readFlags(arguments))
    {
        applySetting(settings, lookupFlag(entry.flag), entry.flag, entry.value);
    }
    requireUsable(settings);
    return settings;
}

// The search one arm of the sweep runs. An arm is one treatment in the coverage
// experiment, and the three parameters are the only things that may differ between arms.
//
// They are the three that can widen a root. The root is the position being decided, and
// its children are the three legal actions; a search spends its simulations descending
// from it, each descent crediting a visit to whichever action it starts down. A wide root
// is one where those visits are spread over several actions, a narrow one where nearly all
// of them pour down a single action - measured at 1.13 of 3 on a trained network, because
// PUCT weighs exploration by the prior and a confident prior leaves the other two actions
// nothing. That matters here because a death-risk label is only trainable at a position
// where every root action was visited, so root width is what this program measures.
//
// Usage - the control and the two treatments, exactly as runArms calls them:
//
//     armConfig(settings, 0.0f, az::EXPLORATION, 0.0f);
//     armConfig(settings, az::ROOT_NOISE_FRACTION, az::EXPLORATION, 0.0f);
//     armConfig(settings, 0.0f, az::EXPLORATION, 0.1f);
//
//     noise_fraction       Dirichlet noise mixed into the root prior, 0 to disable
//     exploration          c_puct, the constant PUCT weighs the prior by
//     exploration_epsilon  additive uniform floor on root selection, 0 to disable
//
// Why the three are parameters rather than fields of the shared settings: a treatment must
// not be able to reach the control, nor the trajectory that generated the positions. Being
// arguments makes that structural instead of a rule someone has to remember.
//
// Why the controls are listed here rather than left to be looked up: this is an
// instrument, and a reader comparing arms should not need a second file to learn what was
// held fixed. Set below - simulations, trap guard off, edge averaging off, death cap off,
// alias reporting off, and the seed. From az::paperSearchDefaults(settings.board) - the
// discount, which is derived from the board rather than fixed, plus step reward -0.02,
// steps tie-break margin 0.05, trap reporting on, value normalisation off,
// death-cap threshold 0.5, root noise alpha 0.3. Between them those two lists account for
// every field of Config; when that stops being true, an arm is running a search nobody
// described.
MonteCarloSearch::Config armConfig(const CoverageSettings& settings, float noise_fraction,
                                   float exploration, float exploration_epsilon)
{
    MonteCarloSearch::Config config = az::paperSearchDefaults(settings.board);
    config.simulations = settings.simulations;
    // Swept across arms, so it overrides the shared default rather than taking it.
    config.exploration = exploration;
    config.trap_guard = false;
    config.average_edges = false;
    config.exploration_epsilon = exploration_epsilon;
    config.death_cap = false;
    config.alias_report = false;
    config.root_noise_fraction = noise_fraction;
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

// What one arm measured.
struct ArmResult
{
    // Coverage counts accumulated over every position this arm searched.
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

        int visited = 0;
        for (int action = 0; action < SnakeEnv::ACTION_COUNT; action++)
        {
            if (result.visits[static_cast<std::size_t>(action)] > 0)
            {
                visited++;
            }
        }
        // The counts and the derived flag must agree; they are one fact read two ways.
        if ((visited == SnakeEnv::ACTION_COUNT) != result.allActionsVisited())
        {
            arm.flag_disagreements++;
        }
        arm.tally.observe(visited, SnakeEnv::ACTION_COUNT);
    }
    return arm;
}

// Prints one arm's scored coverage under `label`, as an indented block.
//
// Prints the yield ratio only when it is defined, and says why in words when it is
// not - a rule admitting no labels has unbounded gain, not zero. A non-zero
// disagreement count is printed as a warning: the two readings contradict each
// other, so the numbers above it cannot be trusted.
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

// A network of this run's shape with `checkpoint_path` loaded into it, moved to `device`
// and put in evaluation mode.
//
//     AlphaZeroNet network = loadNetwork(settings, settings.checkpoint, "", device);
//
// `report_prefix` distinguishes the two networks a run can hold; pass "" for the one
// under test. Prints every parameter the file did not carry, because a mistyped module
// name looks exactly like a head added after the checkpoint was written.
//
// Throws std::runtime_error naming the path when the file cannot be read.
AlphaZeroNet loadNetwork(const CoverageSettings& settings, const std::string& checkpoint_path,
                         const std::string& report_prefix, torch::Device device)
{
    AlphaZeroNet network(settings.board, settings.board, settings.channels, settings.blocks);
    loadCheckpoint(network, checkpoint_path, report_prefix);
    network->to(device);
    network->eval();
    return network;
}

// How many root actions were survivable before any search ran.
//
// This is the denominator every coverage number is really against: an action that kills
// on this tick cannot be visited by a search that prunes fatal moves, and no exploration
// constant recovers it. Free - wouldDie copies nothing.
void reportSurvivableActions(const std::vector<SnakeEnv>& played)
{
    std::size_t histogram[SnakeEnv::ACTION_COUNT + 1] = { 0, 0, 0, 0 };
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
        histogram[survivable]++;
    }

    std::cout << "\nSURVIVABLE ROOT ACTIONS - measured without any search\n";
    for (int count = 0; count <= SnakeEnv::ACTION_COUNT; count++)
    {
        std::cout << std::format(
            "  {} survivable   {:6} positions  {:.4f}\n", count, histogram[count],
            static_cast<double>(histogram[count]) / static_cast<double>(played.size()));
    }
}

// What the policy head alone says about each position, with no search involved.
//
// A saturated top prior is the one story consistent with a hundredfold sweep of c_puct
// changing nothing: exploration enters the score as c_puct * prior * sqrt(N), so a prior
// near zero cannot be raised by any constant.
void reportPriorSaturation(AlphaZeroNet& network, torch::Device device,
                           const CoverageSettings& settings, const std::vector<SnakeEnv>& played)
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
            torch::softmax(network->forward(input).policy_logits, 1).to(torch::kCPU).contiguous();
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
        above_99 += top > 0.99 ? 1u : 0u;
        above_999 += top > 0.999 ? 1u : 0u;
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

// Searches the same positions under each configuration and reports every arm.
//
// The two the project actually runs come first - evaluation conditions and self-play
// conditions - then the exploration floor swept upward. c_puct is held at the paper's
// value throughout: sweeping it was already measured to move coverage from 1.13 to 1.24
// actions, because every term it appears in multiplies the prior.
void runArms(AlphaZeroNet& network, torch::Device device, const CoverageSettings& settings,
             const std::vector<SnakeEnv>& positions)
{
    const ArmResult baseline =
        measureArm(network, device, settings, 0.0f, az::EXPLORATION, 0.0f, positions);
    report("NOISE OFF, c_puct 0.5 - evaluation conditions", baseline);

    const ArmResult noisy = measureArm(network, device, settings, az::ROOT_NOISE_FRACTION,
                                       az::EXPLORATION, 0.0f, positions);
    report(
        std::format("NOISE ON at {}, c_puct 0.5 - self-play conditions", az::ROOT_NOISE_FRACTION),
        noisy);

    for (const float epsilon : { 0.05f, 0.1f, 0.3f })
    {
        const ArmResult swept =
            measureArm(network, device, settings, 0.0f, az::EXPLORATION, epsilon, positions);
        report(std::format("NOISE OFF, exploration floor epsilon {}", epsilon), swept);
    }
}

}  // namespace

// Measures how often the search reaches every root action, and what the current label
// rule gives away. Reads a checkpoint, plays games, then searches the same positions
// under several exploration settings without ever playing a move from them.
//
//     AlphaZeroCoverage.exe --checkpoint az10_long308.pt --board 10 --games 8
//
// Returns 1 on a bad flag or an unreadable checkpoint, 0 otherwise.
int main(int argc, char** argv)
{
    try
    {
        // Parses argv into settings; a bad flag throws and is caught below.
        const CoverageSettings settings =
            parseSettings(std::vector<std::string>(argv + 1, argv + argc));
        // CUDA when available, else the CPU; the banner prints which.
        const Compute compute = chooseDevice();
        // Longest a game may run, in moves: STEPS_PER_CELL * board * board.
        const int step_limit = az::deriveStepLimit(settings.board);

        // The checkpoint under test, loaded in eval mode, then the banner naming this run.
        AlphaZeroNet network = loadNetwork(settings, settings.checkpoint, "", compute.device);
        std::cout << std::format("coverage: {} on {}x{}, {} games, {} simulations, device {}\n",
                                 settings.checkpoint, settings.board, settings.board,
                                 settings.games, settings.simulations,
                                 compute.cuda ? "cuda" : "cpu");

        // Start of the elapsed time printed at the end.
        const auto started = std::chrono::high_resolution_clock::now();

        // Positions come from whichever network is named for the job. Comparing several
        // checkpoints means holding this one fixed, so every prior is read on the same
        // boards and the difference is the policy rather than the population.
        AlphaZeroNet position_network = network;
        if (!settings.position_checkpoint.empty())
        {
            position_network =
                loadNetwork(settings, settings.position_checkpoint, "positions: ", compute.device);
            std::cout << std::format("positions generated by {}\n", settings.position_checkpoint);
        }

        // Plays the games once and keeps every root position, then thins them to
        // --max-positions. This frozen set is what every arm below searches.
        const std::vector<SnakeEnv> played =
            collectPositions(position_network, compute.device, settings, step_limit);
        const std::vector<SnakeEnv> positions = sampleEvenly(played, settings.max_positions);
        std::cout << std::format("{} positions played, {} sampled evenly along the trajectory\n",
                                 played.size(), positions.size());
        if (positions.empty())
        {
            throw std::runtime_error("no positions were generated");
        }

        // The two measurements needing no search: how many root moves were survivable at
        // all, which is the ceiling on coverage, and how sharp the policy head has become.
        reportSurvivableActions(played);
        reportPriorSaturation(network, compute.device, settings, played);

        // --skip-arms keeps the two numbers above and runs no search.
        if (settings.skip_arms)
        {
            std::cout << "\n--skip-arms given; no search arms were run\n";
        }
        else
        {
            // Searches that same frozen set under each configuration, so the only
            // difference between arms is the setting.
            runArms(network, compute.device, settings, positions);
            std::cout << "\nevery arm searched the same positions; no move was played from any\n";
        }

        // Reached by both paths, so every run reports what it cost.
        const auto finished = std::chrono::high_resolution_clock::now();
        std::cout << std::format("\nelapsed {:.2f}s\n",
                                 std::chrono::duration<double>(finished - started).count());
        return 0;
    }
    // A bad flag is labelled as one. Both paths exit 1; only the wording differs, so a
    // usage mistake is not read as the tool having failed.
    catch (const std::invalid_argument& error)
    {
        std::cerr << "bad arguments: " << error.what() << std::endl;
        return 1;
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
