#pragma once

#include <chrono>
#include <functional>
#include <iostream>
#include <string>

#include "selfplay.h"
#include "trainer_options.h"

// Draws the self-play progress bar on one terminal line, and clears it before the
// iteration summary lands.
//
// Usage - one per run, told where each iteration starts:
//
//     // The counter is a callable so this stays free of LibTorch; the trainer passes
//     // its evaluator's running total.
//     ProgressPrinter printer([&] { return evaluator.evaluations(); }, step_limit, 340);
//     play.setProgressCallback([&](const SelfPlay::Progress& p) { printer.draw(p); });
//
//     printer.startIteration(331, evaluator.evaluations());   // before playing a batch
//     // ... self-play runs, draw() is called from inside it ...
//     printer.wipe();                                         // before printing a summary
//
// Throttled to four frames a second, so drawing never becomes the bottleneck it is
// reporting on. The length of the last line drawn is remembered so the next write wipes
// exactly it; a fixed-width wipe left the tail of anything longer on screen.
class ProgressPrinter
{
public:
    // `evaluation_count` returns the process's running total of network evaluations;
    // progress is reported as the difference from the total at the iteration's start.
    // `step_limit` is the budget each game plays under, for the moves-remaining estimate.
    // `last_iteration` labels the bar, so it reads "331/340" rather than "331".
    ProgressPrinter(std::function<long long()> evaluation_count, int step_limit, int last_iteration)
        : evaluation_count_(std::move(evaluation_count)),
          step_limit_(step_limit),
          last_iteration_(last_iteration),
          last_drawn_(std::chrono::high_resolution_clock::now())
    {
    }

    // Labels the bar with `iteration` and counts evaluations from `evaluations_so_far`.
    void startIteration(int iteration, long long evaluations_so_far)
    {
        current_iteration_ = iteration;
        evaluations_at_start_ = evaluations_so_far;
    }

    // Redraws the bar. Silent when called again inside the same quarter second.
    void draw(const SelfPlay::Progress& progress)
    {
        // Throttle first, so a caller in a tight loop pays one clock read and no format.
        const auto now = std::chrono::high_resolution_clock::now();
        if (std::chrono::duration<double>(now - last_drawn_).count() < 0.25)
        {
            return;
        }
        last_drawn_ = now;

        trainer::ProgressSnapshot snapshot;
        snapshot.games_total = progress.games_total;
        snapshot.games_finished = progress.games_finished;
        snapshot.moves_played = progress.moves_played;
        snapshot.evaluations = evaluation_count_() - evaluations_at_start_;
        snapshot.step_limit = step_limit_;
        snapshot.elapsed_seconds = progress.elapsed_seconds;

        const std::string bar =
            trainer::formatProgressBar(current_iteration_, last_iteration_, snapshot);
        std::cout << "\r" << bar;
        // Pad only when this line is shorter than the last, so nothing of it survives.
        if (bar.size() < drawn_length_)
        {
            std::cout << std::string(drawn_length_ - bar.size(), ' ');
        }
        drawn_length_ = bar.size();
        std::cout << std::flush;
    }

    // Clears the bar, so a summary printed over a longer one does not inherit its tail.
    void wipe()
    {
        std::cout << "\r" << std::string(drawn_length_, ' ') << "\r";
        drawn_length_ = 0;
    }

private:
    std::function<long long()> evaluation_count_;
    int step_limit_;
    int last_iteration_;
    std::chrono::high_resolution_clock::time_point last_drawn_;
    int current_iteration_{ 0 };
    long long evaluations_at_start_{ 0 };
    size_t drawn_length_{ 0 };
};
