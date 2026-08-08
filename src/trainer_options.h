#pragma once
#include <optional>
#include <span>
#include <string>

// The trainer's settings, and the two strings it prints.
//
// Extracted from az_trainer.cpp, which was a single main() and therefore had no
// testable surface at all: the argument parser accepted "--games 0" and produced
// a division by zero in the summary line eight hundred seconds later, and the
// step limit was derived inside the parser where nothing could check it.
//
// Deliberately free of LibTorch, so assertions are reachable here. A debug
// binary linked against LibTorch dies of an access violation before running -
// the shipped libraries are release-only - so every assert in a Torch-linked
// file in this repository is unreachable, and these would have been too.
namespace trainer
{

// Hyperparameters from Du, Gemp, Wu and Wu 2022 (arXiv:2211.09622), the paper
// this stack reproduces. They live here rather than at the two call sites that
// build the search config and the self-play config, because the discount was
// written out twice and two copies of one number are two numbers waiting to
// disagree. Deviating from these knowingly is allowed; deviating by accident is
// what a single definition prevents.
constexpr float DISCOUNT = 0.98f;
constexpr float EXPLORATION = 0.5f;  // c_puct
constexpr float VISIT_TEMPERATURE = 0.5f;
constexpr float LEARNING_RATE = 0.001f;
constexpr float ROOT_NOISE_FRACTION = 0.25f;
constexpr float ROOT_NOISE_ALPHA = 0.3f;

// The paper caps a 10x10 game at 1,200 steps, which is twelve steps per cell.
// Scaling by area rather than fixing the number keeps "win" meaning the same
// thing at every board size the curriculum passes through.
constexpr int STEPS_PER_CELL = 12;

// Plain aggregate, so the rule of zero applies as written: no member owns a
// resource, the compiler's copy is correct, and there is nothing to delete. That
// is the opposite decision from MonteCarloSearch, and for the opposite reason -
// settings are a value, a search is not.
struct Settings
{
    int board = 6;
    int iterations = 20;
    // Absolute index of the first iteration. A resumed run must be given the
    // number the previous run stopped at, or it replays that run's games: the
    // seed for a game is derived from the iteration index, and `--resume`
    // restores weights only.
    int start_iteration = 1;
    int games_per_iteration = 32;
    int simulations = 64;
    // Empty means derive it from the board. This used to be a zero standing in
    // for "not given", resolved by overwriting the field during parsing - so the
    // resolved value and the sentinel occupied one variable and no caller could
    // tell which it was holding.
    std::optional<int> step_limit_override;
    int channels = 64;
    int blocks = 4;
    int batch_size = 128;
    int batches_per_iteration = 64;
    size_t replay_bytes = 1024u * 1024u * 1024u;  // 1 GiB, measured not guessed
    unsigned int seed = 1;
    std::string checkpoint;
    std::string resume;

    int cellCount() const { return board * board; }
    // The snake starts one segment long, so this many apples fills the board.
    int foodsToWin() const { return cellCount() - 1; }
    // The override if one was given, the area-scaled default otherwise.
    int stepLimit() const;
    // Last iteration this run will play, inclusive.
    int lastIteration() const { return start_iteration + iterations - 1; }
};

// Parses the command line, or throws std::invalid_argument naming the flag.
//
// `arguments` excludes argv[0]. Every rejection is a throw rather than a warning
// on stderr: the previous parser printed "unknown flag" and carried on with the
// default, and dropped a trailing flag with no value in silence. Both produced a
// run that trained for hours with a setting the operator believed they had
// changed.
Settings parseArguments(std::span<const char* const> arguments);

// Minutes and seconds, or "--:--" when there is nothing meaningful to show.
std::string formatDuration(double seconds);

// What the trainer knows mid-iteration. Plain numbers rather than
// SelfPlay::Progress so this stays independent of the self-play layer.
struct ProgressSnapshot
{
    int games_total;
    int games_finished;
    long long moves_played;
    long long evaluations;
    int step_limit;
    double elapsed_seconds;
};

// One redrawn progress line, ASCII only, with no leading carriage return and no
// trailing padding - the caller owns the cursor. Returning the string rather
// than printing it is what makes the percentage and the eta checkable.
std::string formatProgressBar(int iteration, int last_iteration, const ProgressSnapshot& progress);

}  // namespace trainer
