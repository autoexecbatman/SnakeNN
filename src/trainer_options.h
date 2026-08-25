#pragma once

#include <optional>
#include <span>
#include <string>

#include "az_parameters.h"

// The trainer's settings, its command line, and the two strings it prints.
//
// Usage - what az_trainer.cpp's main does:
//
//     // argv without the program name.
//     const trainer::Settings settings =
//         trainer::parseArguments({ argv + 1, static_cast<size_t>(argc - 1) });
//
//     // Derived, never read from a field: the override if one was given.
//     const int step_limit = settings.stepLimit();
//
//     std::cout << trainer::formatProgressBar(iteration, settings.lastIteration(),
//                                             progress) << carriage_return;
//
// Every rejection is a throw naming the flag, so a mistyped setting stops the run
// rather than training for hours on a default. Free of LibTorch, so its assertions
// are reachable in a debug build.
namespace trainer
{

// Which board to train on, for how long, with what network shape and search budget.
// Filled by parseArguments from the command line.
//
// A plain aggregate: copy it, pass it by const reference, and read the derived
// quantities - the step limit, the last iteration, samples per game - through the
// member functions, which are the only place their rules are written.
struct Settings
{
    // Side of the square board. The curriculum starts small and resumes upward.
    int board{ 6 };
    // How many iterations this run plays before stopping.
    int iterations{ 20 };
    // Absolute index of the first iteration. A resumed run must be given the
    // number the previous run stopped at, or it replays that run's games: the
    // seed for a game is derived from the iteration index, and `--resume`
    // restores weights only.
    int start_iteration{ 1 };
    // Self-play games per iteration. Hundreds in flight is what makes the batch big
    // enough for the GPU; 32 measured a fifteenth of the throughput of 1024.
    int games_per_iteration{ 32 };
    // Search simulations per move. Never quote a win rate without it - four times the
    // simulations doubled the rate here with no retraining.
    int simulations{ 64 };
    // Empty means derive it from the board. This used to be a zero standing in
    // for "not given", resolved by overwriting the field during parsing - so the
    // resolved value and the sentinel occupied one variable and no caller could
    // tell which it was holding.
    std::optional<int> step_limit_override;
    // Trunk width, in convolution channels.
    int channels{ 64 };
    // Residual blocks in the trunk.
    int blocks{ 4 };
    // States per gradient step.
    int batch_size{ 128 };
    // Gradient steps per iteration, unless samples_per_game_override sets it instead.
    int batches_per_iteration{ 64 };
    // Given instead of --batches, this sets it: batches = samples * games / batch.
    // The paper trains 30 batches of 100 states after every game, so 3,000 per
    // game; the equivalent --batches depends on the game count and the batch size
    // and is therefore the wrong thing to write down. Giving both is refused.
    std::optional<int> samples_per_game_override;
    // Ceiling on the replay buffer, in bytes. It is the resource that goes unbounded
    // first: encoded observations at 3.2KB a record took an early run into swap.
    size_t replay_bytes{ 1024u * 1024u * 1024u };  // 1 GiB
    // Base seed. A game's own seed is derived from this and its iteration index.
    unsigned int seed{ 1 };
    // Where to write the weights after each iteration. Empty writes nothing.
    std::string checkpoint;
    // Checkpoint to load before the first iteration. Weights only - the iteration
    // number comes from start_iteration.
    std::string resume;
    // Where this run records what it cost. Relative to the launch directory, which
    // is build/Release and is not version controlled, so a run meant to leave a
    // durable record passes --ledger ../../docs/runs.tsv.
    std::string ledger_path{ "runs.tsv" };
    // Share of each batch drawn preferentially from positions a move can lose from.
    // 0 leaves sampling uniform, which is what every run before 2026-08-24 did.
    float decisive_share{ 0.0f };
    // L2 penalty on every weight, applied by Adam. 0 is what every run before 2026-08-24
    // used; AlphaZero uses 1e-4. It matters most with a large replay window, where the
    // network is otherwise free to fit many generations of its own old play.
    // float, not double: flags::parseUnitFloat produces a float, and widening it here
    // manufactures digits the value never had - 0.0001 stored as a double prints as
    // 9.999999747378752e-05.
    float weight_decay{ 0.0f };
    // What the learning rate is multiplied by at the end of the run, reached geometrically
    // over the iterations played. 1 holds it constant, which is what every run before
    // 2026-08-24 did.
    float final_learning_rate_fraction{ 1.0f };
    // Share of self-play moves given the full search. 0 means every move gets it, which is
    // what every run before 2026-08-24 did; KataGo uses about a quarter.
    float full_search_fraction{ 0.0f };
    // The cheap budget, as a share of --simulations. Read only when full_search_fraction is
    // above zero. KataGo's fast searches are roughly a sixth of its full ones.
    float fast_simulation_fraction{ 0.25f };
    // Where the death head's target comes from. False - the default, and what every run
    // before 2026-08-26 did - takes whatever the search backed up, kept only where all
    // three root actions were visited: 2.6 percent of positions at 12x12. True reads it
    // off the finished game instead, for the one action that game played.
    //
    // Measured 2026-08-25: a position becomes unrecoverable a mean of 323 moves before the
    // death it causes, which no 200-simulation search and no 72-step discount can reach.
    bool doom_label_from_trajectory{ false };
    // How often selection ignores its scores and picks uniformly, before the
    // 1/log(N) decay. A flag rather than a constant because it changes the run's
    // result and the ledger records the command line: compiled in, two runs that
    // differ by it are indistinguishable afterwards. Defaults to the constant, so
    // omitting it keeps whatever az_parameters.h says.
    float exploration_epsilon{ az::EXPLORATION_EPSILON };

    // Cells on the board.
    int cellCount() const noexcept { return board * board; }
    // The snake starts one segment long, so this many apples fills the board.
    int foodsToWin() const noexcept { return cellCount() - 1; }
    // The override if one was given, the area-scaled default otherwise.
    int stepLimit() const;
    // Last iteration this run will play, inclusive.
    int lastIteration() const noexcept { return start_iteration + iterations - 1; }
    // Gradient samples drawn per game played, which is the quantity comparable
    // with the paper's 3,000 - --batches on its own is not, since it says nothing
    // about how many games those batches were spread over.
    long long samplesPerGame() const noexcept;
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

// What the trainer knows part way through an iteration, as plain numbers. Fill one and
// hand it to formatProgressBar.
struct ProgressSnapshot
{
    // Games this iteration will play.
    int games_total{ 0 };
    // How many of them have ended.
    int games_finished{ 0 };
    // Moves taken across every game so far.
    long long moves_played{ 0 };
    // Network evaluations so far. Divided by the elapsed time this is the throughput
    // figure the progress line shows.
    long long evaluations{ 0 };
    // The step budget each game is playing under, for the moves-remaining estimate.
    int step_limit{ 0 };
    // Seconds since the iteration started.
    double elapsed_seconds{ 0.0 };
};

// One redrawn progress line, ASCII only, with no leading carriage return and no
// trailing padding - the caller owns the cursor. Returning the string rather
// than printing it is what makes the percentage and the eta checkable.
std::string formatProgressBar(int iteration, int last_iteration, const ProgressSnapshot& progress);

}  // namespace trainer
