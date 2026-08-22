#pragma once

#include <string>
#include <vector>

#include "replay_window.h"
#include "selfplay.h"

// What one training iteration came to, and the single line that records it.
//
// Usage - fold the games, then format the line the logs get parsed from:
//
//     const BatchStats stats = summariseGames(summaries);   // wins, timeouts, scores
//     IterationReport report;
//     report.iteration = 331;
//     report.games = summaries.size();
//     report.foods_to_win = 399;          // apples needed to fill this board
//     report.sealed_choices = play.sealedChoices();
//     report.total_seconds = 2754.0;      // play and training together
//     report.play_seconds = 2702.0;       // play alone, what the rate divides by
//     report.evaluations = 165000000;
//     std::cout << formatIterationSummary(batch_size, report, stats, replay, totals)
//               << std::endl;
//
// That prints, for example:
//
//     iter 331  score 144.1367/399  best 176  wins 0/256  timeouts 29  sealed 94453
//       buffer 8325112 (1966MB)  loss p 0.457611 v 0.000476 d 0.005090  labels 102.0/256
//       2893.04s (play 2840.942s, 61231 evals/s)
//
// Nothing here touches LibTorch, so the line's shape is testable without a GPU. That
// matters more than it sounds: these summaries are parsed back out of logs months later,
// and a change to the format silently invalidates every parser written against it.
namespace trainer
{

// What one iteration's games came to.
struct BatchStats
{
    // Games that filled the board.
    int wins{ 0 };
    // Games cut off at the step limit rather than ending on their own.
    int hit_step_limit{ 0 };
    // Apples summed over the batch, which the summary divides by the game count.
    double total_score{ 0.0 };
    // Best single game of the batch.
    int best_score{ 0 };
};

// One iteration's training losses, summed over its batches.
struct LossTotals
{
    // Summed policy loss.
    double policy{ 0.0 };
    // Summed value loss.
    double value{ 0.0 };
    // Summed death-risk loss.
    double death{ 0.0 };
    // Summed count of death labels that survived the usability mask.
    double usable_labels{ 0.0 };
    // Batches actually run. Zero when the replay window held less than one batch, which
    // is the state of a fresh run's first iteration.
    int batches_run{ 0 };
};

// What one finished iteration cost and covered.
struct IterationReport
{
    // Which iteration this was, in absolute numbering across resumed runs.
    int iteration{ 0 };
    // Games played in it.
    size_t games{ 0 };
    // Apples needed to fill this board, the denominator the score is reported against.
    int foods_to_win{ 0 };
    // Moves the search declined to reconsider, cumulative over the run.
    long long sealed_choices{ 0 };
    // Wall clock for play and training together.
    double total_seconds{ 0.0 };
    // Wall clock for play alone, which the evaluations rate is computed against.
    double play_seconds{ 0.0 };
    // Network evaluations this iteration made.
    long long evaluations{ 0 };
};

// Folds a batch of finished games into the counts the summary reports.
//
//     const BatchStats stats = summariseGames(summaries);
//     stats.total_score / summaries.size();   // mean apples
//
// An empty batch returns zeros; the caller divides by the game count and must reject one.
BatchStats summariseGames(const std::vector<GameSummary>& summaries);

// The one-line record of an iteration, in the shape these logs get parsed in.
//
//     std::cout << formatIterationSummary(256, report, stats, replay, totals) << std::endl;
//
// `batch_size` is reported as the denominator of the label count. Precisions are fixed
// rather than left to the default, so the line keeps a stable shape across runs. The loss
// half is omitted when no batch ran, rather than printed as a row of zeros that would read
// like a network that had already converged.
std::string formatIterationSummary(int batch_size, const IterationReport& report,
                                   const BatchStats& stats, const ReplayWindow& replay,
                                   const LossTotals& totals);

}  // namespace trainer
