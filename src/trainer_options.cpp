#include <algorithm>
#include <cassert>
#include <charconv>
#include <format>
#include <limits>
#include <stdexcept>
#include <string>

#include "az_parameters.h"
#include "flag_parser.h"
#include "trainer_options.h"

namespace trainer
{
namespace
{

// Characters of hashes and dots in the redrawn bar.
constexpr int PROGRESS_BAR_WIDTH = 28;
// Below this the elapsed time is too short for a rate to mean anything.
constexpr double MINIMUM_ELAPSED_FOR_RATE = 0.5;
// Below this the completed fraction is too small to extrapolate an eta from.
constexpr double MINIMUM_FRACTION_FOR_ETA = 0.02;
// A duration longer than this has no reading in minutes and seconds.
constexpr double LONGEST_READABLE_DURATION = 60.0 * 60.0 * 99.0;

// The whole value or an exception naming the flag it came from.
//
// std::stoi was doing this before, and it stops at the first character it cannot
// use: "--board 10x10" parsed as 10 and trained on a board nobody asked for,
// while "--board ten" threw std::invalid_argument out of main with no mention of
// which flag was wrong. from_chars reports where it stopped, so demanding that it
// stopped at the end is what rejects a partial parse.
int parseWholeInt(const std::string& flag, const char* text)
{
    const std::string value = text;
    int number = 0;
    const char* begin = value.data();
    const char* end = begin + value.size();
    const std::from_chars_result result = std::from_chars(begin, end, number);
    if (result.ec != std::errc{} || result.ptr != end)
    {
        throw std::invalid_argument(std::format("{} needs a whole number, got '{}'", flag, value));
    }
    return number;
}

// Throws std::invalid_argument naming the flag, its bound and what was given, unless
// value is at or above `floor`.
void requireAtLeast(const std::string& flag, int value, int floor)
{
    if (value < floor)
    {
        throw std::invalid_argument(
            std::format("{} must be at least {}, got {}", flag, floor, value));
    }
}

// The same, for an upper bound.
void requireAtMost(const std::string& flag, int value, int ceiling)
{
    if (value > ceiling)
    {
        throw std::invalid_argument(
            std::format("{} must be at most {}, got {}", flag, ceiling, value));
    }
}

// Turns a per-game gradient budget into the batch count the loop consumes.
//
// Validates the two operands it divides by before dividing, since requireUsable
// has not run yet - the batch count it would check is the one being computed here.
void resolveGradientBudget(Settings& settings, bool batches_given)
{
    if (!settings.samples_per_game_override)
    {
        return;
    }
    if (batches_given)
    {
        throw std::invalid_argument(
            "--samples-per-game and --batches say the same thing two ways; give one");
    }
    requireAtLeast("--samples-per-game", *settings.samples_per_game_override, 1);
    requireAtLeast("--games", settings.games_per_iteration, 1);
    requireAtLeast("--batch", settings.batch_size, 1);

    const long long batches = static_cast<long long>(*settings.samples_per_game_override) *
                              settings.games_per_iteration / settings.batch_size;
    if (batches > std::numeric_limits<int>::max())
    {
        throw std::invalid_argument(
            std::format("--samples-per-game {} over {} games at batch {} needs {} batches, past "
                        "the end of an int",
                        *settings.samples_per_game_override, settings.games_per_iteration,
                        settings.batch_size, batches));
    }
    settings.batches_per_iteration = static_cast<int>(batches);
}

// Everything that has to hold before a single game is played.
//
// These fire in the first millisecond of the process rather than partway into an
// iteration. Two of them were reachable: --games 0 divided by the summary count
// when the iteration line was formatted, eight hundred seconds in, and --batch 0
// built an empty tensor and trained on it in silence.
void requireUsable(const Settings& settings)
{
    // The network refuses a board under 2x2 as well, but only once it is
    // constructed - after the device, the checkpoint and the optimizer are set
    // up. Checking here means the message names the flag.
    requireAtLeast("--board", settings.board, 2);
    // cellCount squares this in an int and stepLimit multiplies it by twelve, so
    // the ceiling is not decoration. Enforced here rather than left to
    // deriveStepLimit, whose throw was reached only through an assert - so a
    // release build accepted a board that then overflowed.
    requireAtMost("--board", settings.board, az::LARGEST_BOARD);
    requireAtLeast("--iterations", settings.iterations, 1);
    requireAtLeast("--start-iteration", settings.start_iteration, 1);
    // lastIteration adds these two. Compared in 64 bits, since the sum in 32 is
    // the overflow being rejected.
    const long long last_iteration =
        static_cast<long long>(settings.start_iteration) + settings.iterations - 1;
    if (last_iteration > std::numeric_limits<int>::max())
    {
        throw std::invalid_argument(std::format(
            "--start-iteration {} with {} iterations ends at {}, past the end of an int",
            settings.start_iteration, settings.iterations, last_iteration));
    }
    requireAtLeast("--games", settings.games_per_iteration, 1);
    requireAtLeast("--simulations", settings.simulations, 1);
    requireAtLeast("--channels", settings.channels, 1);
    // Zero is legal: a trunk with no residual blocks is a shallow network, not a
    // broken one, and it is what the smallest curriculum step uses.
    requireAtLeast("--blocks", settings.blocks, 0);
    requireAtLeast("--batch", settings.batch_size, 1);
    // Also legal at zero - an iteration that plays and does not train is how a
    // pure self-play measurement is taken.
    requireAtLeast("--batches", settings.batches_per_iteration, 0);
    if (settings.step_limit_override)
    {
        requireAtLeast("--step-limit", *settings.step_limit_override, 1);
    }
    if (settings.replay_bytes == 0)
    {
        throw std::invalid_argument("--replay-mb must be at least 1");
    }
}

}  // namespace

long long Settings::samplesPerGame() const noexcept
{
    assert(games_per_iteration >= 1 && "samplesPerGame on a settings with no games to divide by");
    // Widened before multiplying: 6,000 batches of 512 already exceeds an int.
    return static_cast<long long>(batches_per_iteration) * batch_size / games_per_iteration;
}

int Settings::stepLimit() const
{
    assert(board >= 2 && "stepLimit on a board smaller than 2x2 - parseArguments rejects those");
    if (step_limit_override)
    {
        assert(*step_limit_override >= 1 && "a step limit of zero would end every game at once");
        return *step_limit_override;
    }
    return az::deriveStepLimit(board);
}

// One flag of the trainer's command line.
enum class Flag
{
    Board,
    Iterations,
    StartIteration,
    Games,
    Simulations,
    StepLimit,
    Channels,
    Blocks,
    Batch,
    Batches,
    SamplesPerGame,
    ReplayMegabytes,
    DecisiveShare,
    WeightDecay,
    FinalLearningRateFraction,
    Seed,
    Checkpoint,
    Ledger,
    ExplorationEpsilon,
    Resume
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
    { "--board", Flag::Board },
    { "--iterations", Flag::Iterations },
    { "--start-iteration", Flag::StartIteration },
    { "--games", Flag::Games },
    { "--simulations", Flag::Simulations },
    { "--step-limit", Flag::StepLimit },
    { "--channels", Flag::Channels },
    { "--blocks", Flag::Blocks },
    { "--batch", Flag::Batch },
    { "--batches", Flag::Batches },
    { "--samples-per-game", Flag::SamplesPerGame },
    { "--replay-mb", Flag::ReplayMegabytes },
    { "--decisive-share", Flag::DecisiveShare },
    { "--weight-decay", Flag::WeightDecay },
    { "--final-lr-fraction", Flag::FinalLearningRateFraction },
    { "--seed", Flag::Seed },
    { "--checkpoint", Flag::Checkpoint },
    { "--ledger", Flag::Ledger },
    { "--exploration-epsilon", Flag::ExplorationEpsilon },
    { "--resume", Flag::Resume },
};

// Which Flag `text` names, or std::invalid_argument when it names none.
//
//     lookupFlag("--board")   // Flag::Board
//
// Refused rather than warned about. The parser this replaced printed to stderr and
// continued with the default, so a mistyped flag produced a run that looked configured and
// was not, and the warning had scrolled off by the time the log was read.
Flag lookupFlag(std::string_view text)
{
    for (const FlagName& candidate : FLAG_NAMES)
    {
        if (candidate.text == text)
        {
            return candidate.flag;
        }
    }
    throw std::invalid_argument(std::format("unknown flag: {}", text));
}

// Applies one parsed flag, or throws naming the flag when the value is not what that flag
// accepts.
//
//     bool batches_given = false;
//     applySetting(settings, batches_given, Flag::Board, "--board", "20");   // board == 20
//
// `batches_given` is set by --batches alone: the gradient budget can be stated as a batch
// count or as samples per game, and a --batches equal to its own default is still a
// --batches that was given, which no comparison against the default can see.
//
// No default case on purpose: the value comes from lookupFlag and can only be an
// enumerator, so the switch is exhaustive and a new enumerator without a case is caught at
// compile time rather than falling through in silence.
void applySetting(Settings& settings, bool& batches_given, Flag flag, const std::string& name,
                  const char* value)
{
    switch (flag)
    {
        case Flag::Board:
        {
            settings.board = parseWholeInt(name, value);
            break;
        }
        case Flag::Iterations:
        {
            settings.iterations = parseWholeInt(name, value);
            break;
        }
        case Flag::StartIteration:
        {
            settings.start_iteration = parseWholeInt(name, value);
            break;
        }
        case Flag::Games:
        {
            settings.games_per_iteration = parseWholeInt(name, value);
            break;
        }
        case Flag::Simulations:
        {
            settings.simulations = parseWholeInt(name, value);
            break;
        }
        case Flag::StepLimit:
        {
            settings.step_limit_override = parseWholeInt(name, value);
            break;
        }
        case Flag::Channels:
        {
            settings.channels = parseWholeInt(name, value);
            break;
        }
        case Flag::Blocks:
        {
            settings.blocks = parseWholeInt(name, value);
            break;
        }
        case Flag::Batch:
        {
            settings.batch_size = parseWholeInt(name, value);
            break;
        }
        case Flag::Batches:
        {
            settings.batches_per_iteration = parseWholeInt(name, value);
            batches_given = true;
            break;
        }
        case Flag::SamplesPerGame:
        {
            settings.samples_per_game_override = parseWholeInt(name, value);
            break;
        }
        case Flag::ReplayMegabytes:
        {
            // Mebibytes rather than bytes, because the only reason to touch this is board
            // size and nobody types ten digits correctly. The cap is real: the first long
            // run of this trainer stored encoded planes at 3.2KB per record and took the
            // machine into swap.
            const int megabytes = parseWholeInt(name, value);
            requireAtLeast(name, megabytes, 1);
            settings.replay_bytes = static_cast<size_t>(megabytes) * 1024u * 1024u;
            break;
        }
        case Flag::DecisiveShare:
        {
            // A flag rather than a constant so the ledger, which records the command line,
            // distinguishes a run that used the bias from one that did not.
            settings.decisive_share = flags::parseUnitFloat(name, value);
            break;
        }
        case Flag::WeightDecay:
        {
            // A rate in [0, 1] rather than an arbitrary double: the useful range is around
            // 1e-4, and anything above 1 would swamp the gradient entirely.
            settings.weight_decay = flags::parseUnitFloat(name, value);
            break;
        }
        case Flag::FinalLearningRateFraction:
        {
            settings.final_learning_rate_fraction = flags::parseUnitFloat(name, value);
            break;
        }
        case Flag::Seed:
        {
            settings.seed = static_cast<unsigned int>(parseWholeInt(name, value));
            break;
        }
        case Flag::Checkpoint:
        {
            settings.checkpoint = value;
            break;
        }
        case Flag::Ledger:
        {
            settings.ledger_path = value;
            break;
        }
        case Flag::ExplorationEpsilon:
        {
            settings.exploration_epsilon = flags::parseUnitFloat(name, value);
            break;
        }
        case Flag::Resume:
        {
            settings.resume = value;
            break;
        }
    }
}

Settings parseArguments(std::span<const char* const> arguments)
{
    Settings settings;
    // The budget and the batch count say the same thing two ways, so exactly one of them
    // may be given. Tracked rather than inferred from the defaults: a --batches equal to
    // its default is still a --batches that was given.
    bool batches_given = false;
    // Two at a time: a flag and its value.
    for (size_t index = 0; index < arguments.size(); index++)
    {
        const std::string flag = arguments[index];
        if (index + 1 >= arguments.size())
        {
            throw std::invalid_argument(std::format("{} was given no value", flag));
        }
        const char* value = arguments[index + 1];
        index++;

        applySetting(settings, batches_given, lookupFlag(flag), flag, value);
    }
    resolveGradientBudget(settings, batches_given);
    requireUsable(settings);
    assert(settings.stepLimit() >= 1 && "a validated settings object still has no step limit");
    return settings;
}

std::string formatDuration(double seconds)
{
    if (!(seconds >= 0.0) || seconds > LONGEST_READABLE_DURATION)
    {
        // Negated rather than written as `seconds < 0.0` so that a NaN elapsed
        // time reads as unknown instead of comparing false and being formatted.
        return "--:--";
    }
    const int total = static_cast<int>(seconds + 0.5);
    return std::format("{:02}:{:02}", total / 60, total % 60);
}

std::string formatProgressBar(int iteration, int last_iteration, const ProgressSnapshot& progress)
{
    assert(iteration >= 1 && "iterations are numbered from one");
    assert(last_iteration >= iteration && "the run cannot end before the iteration in flight");
    assert(progress.games_total >= 0 && "a negative number of games");
    assert(progress.games_finished >= 0 && progress.games_finished <= progress.games_total &&
           "more games finished than were started");
    assert(progress.moves_played >= 0 && "moves played went backwards");
    assert(progress.step_limit >= 1 && "the step limit is part of the task and cannot be zero");
    assert(progress.evaluations >= 0 && "the evaluation counter went backwards");
    assert(progress.elapsed_seconds >= 0.0 && "elapsed time went backwards");

    const double by_games = progress.games_total > 0
                                ? static_cast<double>(progress.games_finished) /
                                      static_cast<double>(progress.games_total)
                                : 0.0;

    // Games finished is the honest measure but it stays at zero for the first
    // minutes of a large board, where nothing has ended yet - a bar pinned at 0
    // percent is what this was added to avoid. Moves played against the worst
    // case is a lower bound on real progress and moves from the first second, so
    // the bar shows whichever is further along. Both are non-decreasing, so their
    // maximum is too.
    const double worst_case_moves =
        static_cast<double>(progress.games_total) * static_cast<double>(progress.step_limit);
    const double by_moves = worst_case_moves > 0.0
                                ? static_cast<double>(progress.moves_played) / worst_case_moves
                                : 0.0;
    const double fraction = std::min(1.0, std::max(by_games, by_moves));
    assert(fraction >= 0.0 && fraction <= 1.0 && "the completed fraction left the unit interval");

    const int filled =
        std::min(PROGRESS_BAR_WIDTH, static_cast<int>(fraction * PROGRESS_BAR_WIDTH));
    std::string cells(PROGRESS_BAR_WIDTH, '.');
    for (int cell = 0; cell < filled; cell++)
    {
        cells[cell] = '#';
    }

    std::string bar =
        std::format("  iter {}/{} [{}] {:>3}%  games {}/{}  moves {}", iteration, last_iteration,
                    cells, static_cast<int>(fraction * 100.0), progress.games_finished,
                    progress.games_total, progress.moves_played);

    if (progress.elapsed_seconds > MINIMUM_ELAPSED_FOR_RATE)
    {
        bar += std::format("  {} ev/s",
                           static_cast<long long>(static_cast<double>(progress.evaluations) /
                                                  progress.elapsed_seconds));
    }
    bar += std::format("  {}", formatDuration(progress.elapsed_seconds));

    // Remaining time from the share already done. Games do not all take the same
    // length, so this is an estimate and is labelled as one.
    if (fraction > MINIMUM_FRACTION_FOR_ETA)
    {
        const double remaining = progress.elapsed_seconds * (1.0 / fraction - 1.0);
        bar += std::format(" eta {}", formatDuration(remaining));
    }
    else
    {
        bar += " eta --:--";
    }

    assert(bar.find('\r') == std::string::npos && bar.find('\n') == std::string::npos &&
           "the bar must carry no cursor control - the caller owns the cursor");
    return bar;
}

}  // namespace trainer
