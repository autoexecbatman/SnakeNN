#include <algorithm>
#include <cassert>
#include <charconv>
#include <format>
#include <stdexcept>
#include <string>

#include "az_parameters.h"
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

void requireAtLeast(const std::string& flag, int value, int floor)
{
    if (value < floor)
    {
        throw std::invalid_argument(
            std::format("{} must be at least {}, got {}", flag, floor, value));
    }
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
    requireAtLeast("--iterations", settings.iterations, 1);
    requireAtLeast("--start-iteration", settings.start_iteration, 1);
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

Settings parseArguments(std::span<const char* const> arguments)
{
    Settings settings;
    for (size_t index = 0; index < arguments.size(); index++)
    {
        const std::string flag = arguments[index];
        if (index + 1 >= arguments.size())
        {
            throw std::invalid_argument(std::format("{} was given no value", flag));
        }
        const char* value = arguments[index + 1];
        index++;

        if (flag == "--board")
        {
            settings.board = parseWholeInt(flag, value);
        }
        else if (flag == "--iterations")
        {
            settings.iterations = parseWholeInt(flag, value);
        }
        else if (flag == "--start-iteration")
        {
            settings.start_iteration = parseWholeInt(flag, value);
        }
        else if (flag == "--games")
        {
            settings.games_per_iteration = parseWholeInt(flag, value);
        }
        else if (flag == "--simulations")
        {
            settings.simulations = parseWholeInt(flag, value);
        }
        else if (flag == "--step-limit")
        {
            settings.step_limit_override = parseWholeInt(flag, value);
        }
        else if (flag == "--channels")
        {
            settings.channels = parseWholeInt(flag, value);
        }
        else if (flag == "--blocks")
        {
            settings.blocks = parseWholeInt(flag, value);
        }
        else if (flag == "--batch")
        {
            settings.batch_size = parseWholeInt(flag, value);
        }
        else if (flag == "--batches")
        {
            settings.batches_per_iteration = parseWholeInt(flag, value);
        }
        else if (flag == "--replay-mb")
        {
            // Mebibytes rather than bytes, because the only reason to touch this
            // is board size and nobody types ten digits correctly. The cap is
            // real: the first long run of this trainer stored encoded planes at
            // 3.2KB per record and took the machine into swap.
            const int megabytes = parseWholeInt(flag, value);
            requireAtLeast(flag, megabytes, 1);
            settings.replay_bytes = static_cast<size_t>(megabytes) * 1024u * 1024u;
        }
        else if (flag == "--seed")
        {
            settings.seed = static_cast<unsigned int>(parseWholeInt(flag, value));
        }
        else if (flag == "--checkpoint")
        {
            settings.checkpoint = value;
        }
        else if (flag == "--resume")
        {
            settings.resume = value;
        }
        else
        {
            // Refused rather than warned about. The previous parser printed to
            // stderr and continued with the default, so a mistyped flag produced
            // a run that looked configured and was not, and the warning had
            // scrolled off by the time the log was read.
            throw std::invalid_argument(std::format("unknown flag: {}", flag));
        }
    }
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
