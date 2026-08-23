// Turns a command line into the settings AlphaZeroEvaluate and AlphaZeroVisual run under.
//
// Two programs, one grammar. The evaluator scores a checkpoint on held-out seeds and the
// visual agent shows the same checkpoint playing, so a flag that means one thing in one and
// something else in the other would make the picture and the number disagree without
// anything saying so. Both parse through here.
//
// Everything is validated before a checkpoint is opened or a game is played. A run that is
// going to be refused should cost nothing, and more importantly a refusal that arrives
// after twenty minutes of self-play arrives when nobody is watching. Refusals throw
// std::invalid_argument naming the flag; parseArguments never returns a Settings that
// stepLimit or cellCount would assert on.
//
// The seed check is the one worth knowing about. Evaluation seeds live in a reserved band
// that training cannot reach, and a run whose last game would fall outside it is refused
// rather than wrapped - because wrapping lands on a training seed, and that is how a
// held-out set stopped being held out once already.
//
// See eval_options.h for the fields and their defaults; this file holds the parsing, the
// bounds and the header line a run prints about itself.

#include <cassert>
#include <format>
#include <stdexcept>
#include <string_view>

#include "az_parameters.h"
#include "eval_options.h"
#include "flag_parser.h"
#include "seed_policy.h"

namespace
{

// The upper half of flags::requireAtLeast. It lives here rather than beside it
// because the board's ceiling is the only bound in this project with an upper
// end, and one caller is not an interface.
void requireAtMost(std::string_view flag, int value, int maximum)
{
    if (value > maximum)
    {
        throw std::invalid_argument(
            std::format("{} must be at most {}, got {}", flag, maximum, value));
    }
}

}  // namespace

namespace evaluation
{
namespace
{

// Everything that has to hold before a checkpoint is read or a game is played.
void requireUsable(const Settings& settings)
{
    if (settings.checkpoint.empty())
    {
        throw std::invalid_argument("--checkpoint names the network to score and is required");
    }
    flags::requireAtLeast("--board", settings.board, 2);
    requireAtMost("--board", settings.board, az::LARGEST_BOARD);
    flags::requireAtLeast("--games", settings.games, 1);
    flags::requireAtLeast("--simulations", settings.simulations, 1);
    flags::requireAtLeast("--channels", settings.channels, 1);
    // Zero is legal: a trunk with no residual blocks is a shallow network, not a
    // broken one.
    flags::requireAtLeast("--blocks", settings.blocks, 0);
    flags::requireAtLeast("--batch", settings.batch, 1);
    if (settings.step_limit_override)
    {
        flags::requireAtLeast("--step-limit", *settings.step_limit_override, 1);
    }
    if (settings.freeze_clock_percent)
    {
        // Zero is legal and means "always out of time", which is one of the two ends
        // the ablation wants; the range is what the clock plane can hold.
        flags::requireAtLeast("--freeze-clock-percent", *settings.freeze_clock_percent, 0);
        requireAtMost("--freeze-clock-percent", *settings.freeze_clock_percent, 100);
    }
    // Compared in 64 bits, since the sum in 32 is the wraparound being rejected.
    const long long last_seed_index =
        static_cast<long long>(settings.seed_offset) + settings.games - 1;
    if (last_seed_index >= static_cast<long long>(seeds::RESERVED_BAND_WIDTH))
    {
        throw std::invalid_argument(
            std::format("--seed {} with {} games runs past the reserved evaluation band of {} "
                        "seeds and wraps into the training range",
                        settings.seed_offset, settings.games, seeds::RESERVED_BAND_WIDTH));
    }
}

}  // namespace

int Settings::cellCount() const noexcept
{
    assert(board >= 2 && "cellCount on a board smaller than 2x2 - parseArguments rejects those");
    assert(board <= az::LARGEST_BOARD && "cellCount on a board whose area overflows an int");
    return board * board;
}

int Settings::foodsToWin() const noexcept
{
    return cellCount() - 1;
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

namespace
{

// Every flag the evaluator accepts.
enum class Flag
{
    Checkpoint,
    Board,
    Games,
    Simulations,
    StepLimit,
    Channels,
    Blocks,
    Seed,
    Batch,
    Ledger,
    FreezeClockPercent,
    TrapGuard,
    AverageEdges,
    DeathCap,
    SearchSeed
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
    { "--games", Flag::Games },
    { "--simulations", Flag::Simulations },
    { "--step-limit", Flag::StepLimit },
    { "--channels", Flag::Channels },
    { "--blocks", Flag::Blocks },
    { "--seed", Flag::Seed },
    { "--batch", Flag::Batch },
    { "--ledger", Flag::Ledger },
    { "--freeze-clock-percent", Flag::FreezeClockPercent },
    { "--trap-guard", Flag::TrapGuard },
    { "--average-edges", Flag::AverageEdges },
    { "--death-cap", Flag::DeathCap },
    { "--search-seed", Flag::SearchSeed },
};

// Which Flag `text` names, or std::invalid_argument when it names none.
//
//     lookupFlag("--board")   // Flag::Board
//
// Refused rather than warned about. The parser this replaced printed to stderr and scored
// the run with the default, so a mistyped flag produced a number that looked configured
// and was not.
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
//     applySetting(settings, Flag::Board, "--board", "10");   // settings.board == 10
//
// No default case on purpose: the value comes from lookupFlag and can only be an
// enumerator, so the switch is exhaustive and a new enumerator without a case is caught at
// compile time rather than falling through in silence.
void applySetting(Settings& settings, Flag flag, std::string_view name, std::string_view value)
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
        case Flag::StepLimit:
        {
            settings.step_limit_override = flags::parseWholeInt(name, value);
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
        case Flag::Seed:
        {
            settings.seed_offset = flags::parseWholeUnsigned(name, value);
            break;
        }
        case Flag::Batch:
        {
            settings.batch = flags::parseWholeInt(name, value);
            break;
        }
        case Flag::Ledger:
        {
            settings.ledger_path = std::string(value);
            break;
        }
        case Flag::FreezeClockPercent:
        {
            settings.freeze_clock_percent = flags::parseWholeInt(name, value);
            break;
        }
        case Flag::TrapGuard:
        {
            settings.trap_guard = flags::parseOnOff(name, value);
            break;
        }
        case Flag::AverageEdges:
        {
            settings.average_edges = flags::parseOnOff(name, value);
            break;
        }
        case Flag::DeathCap:
        {
            settings.death_cap = flags::parseOnOff(name, value);
            break;
        }
        case Flag::SearchSeed:
        {
            settings.search_seed = flags::parseWholeUnsigned(name, value);
            break;
        }
    }
}

}  // namespace

Settings parseArguments(std::span<const std::string> arguments)
{
    Settings settings;
    for (const flags::FlagValue& pair : flags::readFlags(arguments))
    {
        applySetting(settings, lookupFlag(pair.flag), pair.flag, pair.value);
    }
    requireUsable(settings);
    assert(settings.stepLimit() >= 1 && "a validated settings object still has no step limit");
    return settings;
}

std::string formatHeader(const Settings& settings)
{
    const bool trap_guard = settings.trap_guard;
    // An ablated run and an ordinary one differ in nothing else a log records, so
    // this line is the only thing that would stop the two being compared as if they
    // were the same measurement.
    std::string ablation;
    if (settings.freeze_clock_percent)
    {
        ablation = std::format("ABLATED: clock frozen at {} percent of the budget\n",
                               *settings.freeze_clock_percent);
    }

    // Two runs differing only in the search stream play the same games and are
    // otherwise indistinguishable in a log, so the line has to say which stream.
    std::string stream;
    if (settings.search_seed)
    {
        stream = std::format("search seed {} (not the default derived from the offset)\n",
                             *settings.search_seed);
    }

    // Named in both states rather than only when set: a line that appears only for
    // the guarded run makes its absence mean either "off" or "an older binary".
    const std::string_view guard = trap_guard ? "trap guard on" : "trap guard off";

    // Named in both states for the same reason as the guard, and named after what
    // selection reads rather than after the flag: the two arms of this comparison
    // are an expectation and a single draw.
    const std::string_view edges = settings.average_edges ? "averaged edges" : "last-write edges";

    // Named in both states for the same reason again. A run whose header does not
    // say which way this was set cannot be paired with anything later.
    const std::string_view cap = settings.death_cap ? "death cap on" : "death cap off";

    return std::format(
        "=== Evaluation ===\n"
        "{} on {}x{}, {} games, {} simulations, step limit {}, batch {}\n"
        "seeds {}..{} (reserved evaluation range), greedy, no root noise, {}, {}, {}\n{}{}\n",
        settings.checkpoint, settings.board, settings.board, settings.games, settings.simulations,
        settings.stepLimit(), settings.batch, seeds::evaluationGameSeed(settings.seed_offset, 0),
        seeds::evaluationGameSeed(settings.seed_offset, settings.games - 1), guard, edges, cap,
        stream, ablation);
}

std::string formatGameLine(unsigned int seed, Outcome outcome, int score, int steps)
{
    // Three words, none a substring of another, so a parser matching on the word
    // cannot read one outcome as another.
    std::string_view word = "died";
    if (outcome == Outcome::Won)
    {
        word = "won";
    }
    else if (outcome == Outcome::TimedOut)
    {
        word = "timeout";
    }
    return std::format("  game seed {}, {}, score {}, steps {}\n", seed, word, score, steps);
}

}  // namespace evaluation

namespace visual
{
namespace
{

// Everything that has to hold before a window is opened or a checkpoint is read.
void requireUsable(const Settings& settings)
{
    if (settings.checkpoint.empty())
    {
        throw std::invalid_argument("--checkpoint names the network to watch and is required");
    }
    flags::requireAtLeast("--board", settings.board, 2);
    requireAtMost("--board", settings.board, az::LARGEST_BOARD);
    flags::requireAtLeast("--simulations", settings.simulations, 1);
    flags::requireAtLeast("--channels", settings.channels, 1);
    flags::requireAtLeast("--blocks", settings.blocks, 0);
    // Zero would advance the game by nothing per frame, so the demo would render
    // a still picture and look like a hang.
    flags::requireAtLeast("--speed", settings.moves_per_frame, 1);
    if (settings.step_limit_override)
    {
        flags::requireAtLeast("--step-limit", *settings.step_limit_override, 1);
    }
}

}  // namespace

int Settings::cellCount() const noexcept
{
    assert(board >= 2 && "cellCount on a board smaller than 2x2 - parseArguments rejects those");
    assert(board <= az::LARGEST_BOARD && "cellCount on a board whose area overflows an int");
    return board * board;
}

int Settings::foodsToWin() const noexcept
{
    return cellCount() - 1;
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

namespace
{

// Every flag the visual program accepts. Deliberately fewer than the evaluator's: this one
// renders one game rather than scoring a batch.
enum class Flag
{
    Checkpoint,
    Board,
    Simulations,
    StepLimit,
    Channels,
    Blocks,
    Seed,
    Speed
};

// One spelling and the enumerator it names.
struct FlagName
{
    // As written on the command line, leading dashes included.
    std::string_view text;
    // What applySetting will do with it.
    Flag flag;
};

// The whole command line, in one place.
constexpr FlagName FLAG_NAMES[] = {
    { "--checkpoint", Flag::Checkpoint },
    { "--board", Flag::Board },
    { "--simulations", Flag::Simulations },
    { "--step-limit", Flag::StepLimit },
    { "--channels", Flag::Channels },
    { "--blocks", Flag::Blocks },
    { "--seed", Flag::Seed },
    { "--speed", Flag::Speed },
};

// Which Flag `text` names, or std::invalid_argument when it names none.
//
//     lookupFlag("--speed")   // Flag::Speed
//
// The evaluator's flags reach this too, and refusing them is the point: accepting --games
// silently would run a demo configured differently from what was typed.
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
// accepts. No default case, so a new enumerator without a case fails the build.
void applySetting(Settings& settings, Flag flag, std::string_view name, std::string_view value)
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
        case Flag::Simulations:
        {
            settings.simulations = flags::parseWholeInt(name, value);
            break;
        }
        case Flag::StepLimit:
        {
            settings.step_limit_override = flags::parseWholeInt(name, value);
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
        case Flag::Seed:
        {
            settings.seed = flags::parseWholeUnsigned(name, value);
            break;
        }
        case Flag::Speed:
        {
            settings.moves_per_frame = flags::parseWholeInt(name, value);
            break;
        }
    }
}

}  // namespace

Settings parseArguments(std::span<const std::string> arguments)
{
    Settings settings;
    for (const flags::FlagValue& pair : flags::readFlags(arguments))
    {
        applySetting(settings, lookupFlag(pair.flag), pair.flag, pair.value);
    }
    requireUsable(settings);
    assert(settings.stepLimit() >= 1 && "a validated settings object still has no step limit");
    return settings;
}

}  // namespace visual
