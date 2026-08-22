#pragma once

// What a run of held-out games adds up to.
//
// Usage - one per run, folded in as each batch finishes:
//
//     evaluation::Totals totals;                       // starts at zero
//     playBatch(search, settings, step_limit, 0, 64, totals);
//     playBatch(search, settings, step_limit, 64, 36, totals);
//     totals.wins;                                     // the metric
//     totals.total_score / 100.0;                      // mean apples over 100 games
//
// The three outcome counts partition the games: every game either filled the board, died,
// or ran out of steps, so wins + deaths + timeouts equals the games played. That is worth
// knowing because the win rate alone cannot tell a run that is dying from one that is
// running out of steps, and those need opposite fixes.
//
// Free of LibTorch on purpose, so the shape of a result is readable and testable without a
// GPU. The counting lives here; the searching does not.
namespace evaluation
{

// What the games played so far add up to.
struct Totals
{
    // Games that filled the board. This is the metric a checkpoint is judged on.
    int wins{ 0 };
    // Games the snake died in, by hitting a wall or itself.
    int deaths{ 0 };
    // Games cut off at the step limit. The environment does not end these - the caller
    // does, so a timeout is a decision about the task rather than an outcome of the game.
    int timeouts{ 0 };
    // Apples eaten, summed over every game; divided by the game count for the mean score.
    long long total_score{ 0 };
    // Moves made, summed over every game; divided by the game count for the mean. A
    // timed-out game contributes the whole step limit, so this rises as the agent stops
    // dying early - which reads like improvement and is not the same thing.
    long long total_steps{ 0 };
    // Best single game, in apples.
    int best_score{ 0 };
};

}  // namespace evaluation
