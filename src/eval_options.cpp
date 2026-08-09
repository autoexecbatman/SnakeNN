#include <cassert>
#include <format>
#include <limits>
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

// Seeds from EVALUATION_BASE to the top of the unsigned range. A run whose last
// game falls outside this does not error - it wraps to a low seed, which is a
// training seed, which is how the held-out set stopped being held out once before.
constexpr unsigned int RESERVED_BAND_WIDTH =
    std::numeric_limits<unsigned int>::max() - seeds::EVALUATION_BASE + 1u;

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
    // Compared in 64 bits, since the sum in 32 is the wraparound being rejected.
    const long long last_seed_index =
        static_cast<long long>(settings.seed_offset) + settings.games - 1;
    if (last_seed_index >= static_cast<long long>(RESERVED_BAND_WIDTH))
    {
        throw std::invalid_argument(
            std::format("--seed {} with {} games runs past the reserved evaluation band of {} "
                        "seeds and wraps into the training range",
                        settings.seed_offset, settings.games, RESERVED_BAND_WIDTH));
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

Settings parseArguments(std::span<const std::string> arguments)
{
    Settings settings;
    for (const flags::FlagValue& pair : flags::readFlags(arguments))
    {
        if (pair.flag == "--checkpoint")
        {
            settings.checkpoint = pair.value;
        }
        else if (pair.flag == "--board")
        {
            settings.board = flags::parseWholeInt(pair.flag, pair.value);
        }
        else if (pair.flag == "--games")
        {
            settings.games = flags::parseWholeInt(pair.flag, pair.value);
        }
        else if (pair.flag == "--simulations")
        {
            settings.simulations = flags::parseWholeInt(pair.flag, pair.value);
        }
        else if (pair.flag == "--step-limit")
        {
            settings.step_limit_override = flags::parseWholeInt(pair.flag, pair.value);
        }
        else if (pair.flag == "--channels")
        {
            settings.channels = flags::parseWholeInt(pair.flag, pair.value);
        }
        else if (pair.flag == "--blocks")
        {
            settings.blocks = flags::parseWholeInt(pair.flag, pair.value);
        }
        else if (pair.flag == "--seed")
        {
            settings.seed_offset = flags::parseWholeUnsigned(pair.flag, pair.value);
        }
        else if (pair.flag == "--batch")
        {
            settings.batch = flags::parseWholeInt(pair.flag, pair.value);
        }
        else if (pair.flag == "--ledger")
        {
            settings.ledger_path = pair.value;
        }
        else
        {
            // Refused rather than warned about. The previous parser printed to
            // stderr and scored the run with the default, so a mistyped flag
            // produced a number that looked configured and was not.
            throw std::invalid_argument(std::format("unknown flag: {}", pair.flag));
        }
    }
    requireUsable(settings);
    assert(settings.stepLimit() >= 1 && "a validated settings object still has no step limit");
    return settings;
}

std::string formatHeader(const Settings& settings)
{
    return std::format(
        "=== Evaluation ===\n"
        "{} on {}x{}, {} games, {} simulations, step limit {}, batch {}\n"
        "seeds {}..{} (reserved evaluation range), greedy, no root noise\n\n",
        settings.checkpoint, settings.board, settings.board, settings.games, settings.simulations,
        settings.stepLimit(), settings.batch, seeds::evaluationGameSeed(settings.seed_offset, 0),
        seeds::evaluationGameSeed(settings.seed_offset, settings.games - 1));
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

Settings parseArguments(std::span<const std::string> arguments)
{
    Settings settings;
    for (const flags::FlagValue& pair : flags::readFlags(arguments))
    {
        if (pair.flag == "--checkpoint")
        {
            settings.checkpoint = pair.value;
        }
        else if (pair.flag == "--board")
        {
            settings.board = flags::parseWholeInt(pair.flag, pair.value);
        }
        else if (pair.flag == "--simulations")
        {
            settings.simulations = flags::parseWholeInt(pair.flag, pair.value);
        }
        else if (pair.flag == "--step-limit")
        {
            settings.step_limit_override = flags::parseWholeInt(pair.flag, pair.value);
        }
        else if (pair.flag == "--channels")
        {
            settings.channels = flags::parseWholeInt(pair.flag, pair.value);
        }
        else if (pair.flag == "--blocks")
        {
            settings.blocks = flags::parseWholeInt(pair.flag, pair.value);
        }
        else if (pair.flag == "--seed")
        {
            settings.seed = flags::parseWholeUnsigned(pair.flag, pair.value);
        }
        else if (pair.flag == "--speed")
        {
            settings.moves_per_frame = flags::parseWholeInt(pair.flag, pair.value);
        }
        else
        {
            // The evaluator's flags reach this too. Refusing --games here is the
            // point: accepting it silently would run a demo configured differently
            // from what was typed.
            throw std::invalid_argument(std::format("unknown flag: {}", pair.flag));
        }
    }
    requireUsable(settings);
    assert(settings.stepLimit() >= 1 && "a validated settings object still has no step limit");
    return settings;
}

}  // namespace visual
