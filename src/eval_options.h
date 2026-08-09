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
    int board = 6;
    int games = 64;
    int simulations = 200;
    // Empty means derive it from the board. The field it replaces was an int whose
    // zero meant "not given", resolved by overwriting itself during parsing, so no
    // caller could tell which of the two things it was holding.
    std::optional<int> step_limit_override;
    int channels = 64;
    int blocks = 4;
    // An offset within the reserved evaluation band, not an absolute seed. See
    // seed_policy.h: the absolute default this replaced was 900000, which was a
    // training seed for 172 games of every 200.
    unsigned int seed_offset = 0;
    // Games searched together. Not a throughput knob - mcts.cpp draws every
    // simulation's food placement from one generator in lockstep across the batch,
    // so two runs at different batch sizes are not comparable.
    int batch = 64;

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

}  // namespace evaluation
