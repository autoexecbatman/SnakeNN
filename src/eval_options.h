#pragma once
#include <optional>
#include <span>
#include <string>

// The evaluator's settings, and the rejections that happen before a checkpoint is
// read.
//
// Extracted from az_evaluate.cpp, whose parser warned on an unknown flag and then
// scored a run with the default, dropped a flag in final position in silence, and
// reached std::stoi for every number - so "--board 10x10" trained the reading of a
// 10x10 board onto whatever the operator believed they had asked for. Every number
// this project quotes comes out of that program.
//
// Free of LibTorch, so its assertions are reachable. A debug binary linked against
// this repository's LibTorch dies before main, so an assert in a Torch-linked file
// here is dead in both configurations.
namespace evaluation
{

// Plain aggregate: no member owns a resource, so the compiler's copy is correct
// and there is nothing to delete.
struct Settings
{
    // Required. There is no default worth guessing and an empty one is a rejection,
    // not a fallback.
    std::string checkpoint;
    int board{ 6 };
    int games{ 64 };
    int simulations{ 200 };
    // Empty means derive it from the board. The field it replaces was an int whose
    // zero meant "not given", resolved by overwriting itself during parsing, so no
    // caller could tell which of the two things it was holding.
    std::optional<int> step_limit_override;
    int channels{ 64 };
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
std::string formatHeader(const Settings& settings);

// One game's outcome, one line, ending in a newline.
//
// Tagged with "game" so a parser finds these without matching the progress lines,
// and keyed by seed so two runs over the same seeds can be paired. Totals alone
// force an unpaired test on paired data, which is the wrong test and a conservative
// one - McNemar needs these lines.
std::string formatGameLine(unsigned int seed, Outcome outcome, int score, int steps);

}  // namespace evaluation

// The visual demo's settings.
//
// Separate from evaluation::Settings although six of the eight fields match. They
// are two programs, not one program twice: this one has a frame rate and a single
// absolute seed, that one has a game count, a batch and an offset into a reserved
// band. A shared base would couple them to force-fit the part a free function
// already shares.
namespace visual
{

struct Settings
{
    // Required, as in evaluation.
    std::string checkpoint;
    int board{ 6 };
    int simulations{ 200 };
    std::optional<int> step_limit_override;
    int channels{ 64 };
    int blocks{ 4 };
    // An absolute seed, unlike evaluation's offset, and 900000 is a training seed
    // - see seed_policy.h. So the default shows a game the agent may have learned
    // rather than one held out. Preserved as it was; changing what the demo shows
    // is a decision, not a hardening pass.
    unsigned int seed{ 900000 };
    // Search steps per rendered frame. At 1 the search runs once per frame, so the
    // demo advances at the frame rate.
    int moves_per_frame{ 1 };

    int cellCount() const noexcept;
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
