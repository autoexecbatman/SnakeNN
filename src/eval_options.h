#pragma once

#include <optional>
#include <span>
#include <string>

#include "az_parameters.h"

// Settings for the two programs that play a trained network: the evaluator, which
// scores a checkpoint over held-out seeds, and the visual demo, which renders one game.
// Each has its own Settings and its own parseArguments, in its own namespace.
//
// Usage - what az_evaluate.cpp's main does:
//
//     const std::vector<std::string> arguments(argv + 1, argv + argc);
//     const evaluation::Settings settings = evaluation::parseArguments(arguments);
//
//     std::cout << evaluation::formatHeader(settings);
//     std::cout << evaluation::formatGameLine(seed, evaluation::Outcome::Won, score, steps);
//
// Every rejection is a throw naming the flag, so a mistyped setting stops the run
// before a checkpoint is read. Free of LibTorch, so its assertions are reachable.
namespace evaluation
{

// Which checkpoint to score, over how many held-out games, at what search budget.
// Filled by parseArguments from the command line.
//
// This is the program that produces every number this project quotes: noise off, the
// visit-count argmax played, and seeds drawn from a band reserved from training.
struct Settings
{
    // Required. There is no default worth guessing and an empty one is a rejection,
    // not a fallback.
    std::string checkpoint;
    // Side of the square board. Must match the board the checkpoint was trained for
    // only in spirit - no weight depends on it, so a 6x6 network scores at 20x20.
    int board{ 6 };
    // Games to play. The measurement noise floor here is large: the search stream
    // alone flips about a fifth of 1000 games, so a difference under 3 points is
    // invisible at that count.
    int games{ 64 };
    // Search simulations per move. A win rate quoted without this number means
    // nothing - four times the simulations doubled the rate here with no retraining.
    int simulations{ 200 };
    // Empty means derive it from the board. The field it replaces was an int whose
    // zero meant "not given", resolved by overwriting itself during parsing, so no
    // caller could tell which of the two things it was holding.
    std::optional<int> step_limit_override;
    // Trunk width, in convolution channels. Must match what the checkpoint was built
    // with, or loading it fails.
    int channels{ 64 };
    // Residual blocks in the trunk. Must match the checkpoint for the same reason.
    int blocks{ 4 };
    // An offset within the reserved evaluation band, not an absolute seed. See
    // seed_policy.h: the absolute default this replaced was 900000, which was a
    // training seed for 172 games of every 200.
    unsigned int seed_offset{ 0 };
    // Games searched together. Not a throughput knob - mcts.cpp draws every
    // simulation's food placement from one generator in lockstep across the batch,
    // so two runs at different batch sizes are not comparable.
    int batch{ 64 };
    // Where this run records what it cost. Relative to the launch directory, which
    // is build/Release and is not version controlled, so a run meant to leave a
    // durable record passes --ledger ../../docs/runs.tsv.
    std::string ledger_path{ "runs.tsv" };
    // Empty means the clock runs. A value freezes it there for every game, which
    // measures what time awareness is worth by removing it from a trained agent
    // without touching a weight. Whole percent of the budget, 0 to 100 - a percent
    // rather than a fraction so it goes through the same integer parser as every
    // other flag, and the ablation has never wanted finer.
    std::optional<int> freeze_clock_percent;
    // The stream the search draws its imagined apples from, independent of which
    // games are played. Empty means derive it from the seed offset, which is what
    // every run before this did.
    //
    // Two runs that differ only here play the identical games with the identical
    // weights and differ only in what the search guessed about food it cannot see.
    // That difference is the noise floor of this whole measurement, and until it is
    // known, a paired comparison between two checkpoints cannot say how much of its
    // churn is learning.
    std::optional<unsigned int> search_seed;
    // Whether the search's root move is vetoed when it seals the head away from its
    // own tail. Defaults to az::TRAP_GUARD so the constant stays the one place the
    // default is written, and --trap-guard on|off overrides it for a run.
    //
    // Measured twice and rejected twice: 44 against 56 wins at n=64 in the old
    // regime, and 903 against 918 at n=1000 once timeouts were nearly gone. It is a
    // setting rather than a constant because measuring the two states should not mean
    // editing a header and rebuilding.
    bool trap_guard{ az::TRAP_GUARD };
    // Whether selection averages an edge over the traversals that reached it instead
    // of reading the last one written. Defaults to az::AVERAGE_EDGES; --average-edges
    // on|off overrides it for a run.
    //
    // A setting rather than a constant for the same reason as the guard: the two
    // states have to be measurable against each other on one binary, or the arms of
    // the comparison differ by a rebuild as well as by the change.
    bool average_edges{ az::AVERAGE_EDGES };
    // Whether the root refuses an action whose backed-up death risk exceeds
    // az::DEATH_CAP_THRESHOLD. Defaults to az::DEATH_CAP; --death-cap on|off sets it,
    // for the same reason as the two above - both arms of a comparison run on one
    // binary, so they differ by the setting and not also by a rebuild.
    bool death_cap{ az::DEATH_CAP };

    // Cells on the board.
    int cellCount() const noexcept;
    // The snake starts one segment long, so this many apples fills the board.
    int foodsToWin() const noexcept;
    // The override if one was given, az::deriveStepLimit(board) otherwise.
    int stepLimit() const;
};

// Parses the command line, or throws std::invalid_argument naming the flag.
//
// `arguments` excludes argv[0] and its contents must outlive the call. Every
// rejection is a throw rather than a warning: --board 0, --games 0,
// --simulations 0, --batch 0, --channels 0, --blocks -1, a missing checkpoint, an
// unknown flag, and a flag in final position with no value.
//
// A board is rejected below 2 and above 13377, the largest whose step limit fits
// in an int. A seed offset is rejected if the games it spans would run past the
// end of the reserved evaluation band, which is silent unsigned wraparound into
// the training range rather than an error anything would notice.
//
// --trap-guard and --average-edges take "on" or "off" and reject anything else. It carries a value
// rather than being a bare switch because readFlags pairs every flag with one, and
// because a run that means to measure the guard off should be able to say so.
Settings parseArguments(std::span<const std::string> arguments);

// How one game ended. Exhaustive: a game that is not won and did not time out
// died, and there is no fourth state.
enum class Outcome
{
    Won,
    Died,
    TimedOut
};

// The header the run prints, ending in a blank line.
//
// Carries the batch, which is not a throughput knob: mcts.cpp draws every
// simulation's food placement from one generator in lockstep across the batch, so
// two runs at different batch sizes searched different futures and are not
// comparable. Two logs that do not record it cannot be told apart afterwards, which
// is what confounded the iter110-to-iter123 comparison.
//
// Carries the trap guard for the same reason, and names it in both states: a line
// printed only when the guard is on makes its absence mean either "off" or "a binary
// from before this line existed".
std::string formatHeader(const Settings& settings);

// One game's outcome, one line, ending in a newline.
//
// Tagged with "game" so a parser finds these without matching the progress lines,
// and keyed by seed so two runs over the same seeds can be paired. Totals alone
// force an unpaired test on paired data, which is the wrong test and a conservative
// one - McNemar needs these lines.
std::string formatGameLine(unsigned int seed, Outcome outcome, int score, int steps);

}  // namespace evaluation

// The visual demo: one game, rendered, at a frame rate.
//
//     const visual::Settings settings = visual::parseArguments(arguments);
namespace visual
{

// Everything the demo was told to do. Six fields match evaluation::Settings; the two
// that differ are the ones that make it a demo rather than a measurement - an absolute
// seed instead of an offset, and a frame rate instead of a game count.
struct Settings
{
    // Required, as in evaluation.
    std::string checkpoint;
    // Side of the square board.
    int board{ 6 };
    // Search simulations per move. Higher plays better and renders slower.
    int simulations{ 200 };
    // Empty derives the step budget from the board.
    std::optional<int> step_limit_override;
    // Trunk width, in convolution channels. Must match the checkpoint.
    int channels{ 64 };
    // Residual blocks in the trunk. Must match the checkpoint.
    int blocks{ 4 };
    // An absolute seed, unlike evaluation's offset, and 900000 is a training seed
    // - see seed_policy.h. So the default shows a game the agent may have learned
    // rather than one held out. Preserved as it was; changing what the demo shows
    // is a decision, not a hardening pass.
    unsigned int seed{ 900000 };
    // Search steps per rendered frame. At 1 the search runs once per frame, so the
    // demo advances at the frame rate.
    int moves_per_frame{ 1 };

    // Cells on the board.
    int cellCount() const noexcept;
    // The snake starts one segment long, so this many apples fills the board.
    int foodsToWin() const noexcept;
    // The override if one was given, az::deriveStepLimit(board) otherwise.
    int stepLimit() const;
};

// Parses the command line, or throws std::invalid_argument naming the flag.
//
// `arguments` excludes argv[0] and its contents must outlive the call. Rejects
// --board outside 2..az::LARGEST_BOARD, --simulations below 1, --channels below 1,
// --blocks below 0, --speed below 1, --step-limit below 1, a missing checkpoint,
// an unknown flag, and a flag in final position with no value.
//
// The seed is unconstrained: it is a whole unsigned value and every one of them
// names a game.
Settings parseArguments(std::span<const std::string> arguments);

}  // namespace visual
