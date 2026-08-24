#include <algorithm>
#include <format>

#include "iteration_report.h"

namespace trainer
{

BatchStats summariseGames(const std::vector<GameSummary>& summaries)
{
    BatchStats stats;
    for (const GameSummary& summary : summaries)
    {
        stats.wins += summary.won ? 1 : 0;
        stats.hit_step_limit += summary.hit_step_limit ? 1 : 0;
        stats.total_score += summary.score;
        stats.best_score = std::max(stats.best_score, summary.score);
    }
    return stats;
}

std::string formatIterationSummary(int batch_size, const IterationReport& report,
                                   const BatchStats& stats, const ReplayWindow& replay,
                                   const LossTotals& totals)
{
    // The half every iteration has: what the games did and what the window holds.
    std::string summary = std::format(
        "iter {}  score {:.4f}/{}  best {}  wins {}/{}  timeouts {}  sealed {}  buffer {} "
        "({}MB)",
        report.iteration, stats.total_score / report.games, report.foods_to_win, stats.best_score,
        stats.wins, report.games, stats.hit_step_limit, report.sealed_choices, replay.size(),
        replay.bytesUsed() / (1024 * 1024));
    // Omitted rather than zeroed when nothing trained: a row of zeros reads like a
    // converged network, and absence reads like what it is.
    if (totals.batches_run > 0)
    {
        const double mean_value = totals.value / totals.batches_run;
        const double mean_variance = totals.value_variance / totals.batches_run;
        // What the head explains of the target's own spread. Near zero means it is barely
        // better than predicting the batch mean, whatever the loss looks like.
        const double explained = mean_variance > 0.0 ? 1.0 - mean_value / mean_variance : 0.0;
        summary += std::format(
            "  loss p {:.6f} v {:.6f} d {:.6f}  v-var {:.6f} r2 {:.3f}"
            "  labels {:.1f}/{}",
            totals.policy / totals.batches_run, mean_value, totals.death / totals.batches_run,
            mean_variance, explained, totals.usable_labels / totals.batches_run, batch_size);
    }
    // The rate divides by play time alone, so it measures self-play throughput rather than
    // being dragged down by the gradient steps that follow it.
    summary += std::format(
        "  {:.2f}s (play {:.3f}s, {} evals/s)", report.total_seconds, report.play_seconds,
        static_cast<long long>(report.evaluations / std::max(0.001, report.play_seconds)));
    return summary;
}

}  // namespace trainer
