#pragma once

#include "az_parameters.h"
#include "mcts.h"

// The search settings every program here agrees on, so a call site shows only its own
// choices.
//
// Usage - start from the defaults, then set what this program decides:
//
//     MonteCarloSearch::Config config = az::paperSearchDefaults();
//     config.simulations = settings.simulations;   // how deep this run searches
//     config.root_noise_fraction = 0.0f;           // off to measure, on to explore
//     config.seed = settings.seed;                 // the stream apples are imagined from
//     MonteCarloSearch search(evaluator, config);
//
// Eight fields come from the paper and were identical in the trainer, the evaluator, the
// coverage probe and the death probe, written out four times. The nine left unset are the
// ones those programs legitimately disagree about, and leaving them out is deliberate: a
// shared default for a field two programs must differ on is exactly how self-play and
// evaluation come to search differently with nothing in any log recording it. Each caller
// still writes those nine, and now they are the only lines it writes.
namespace az
{

// A config carrying the paper's shared constants and nothing else.
//
//     MonteCarloSearch::Config config = az::paperSearchDefaults(20);
//     config.discount;    // 0.995 - derived from the board, 0.98 at 10x10
//     config.simulations; // 0 - the caller must set this
//
// `board` is required rather than defaulted because the discount depends on it, and a
// program that kept the 10x10 value on a 20x20 board would look exactly like one that
// chose it. Making it an argument is what forces every call site to say which board it
// is searching.
//
// Sets discount, exploration, step reward, the steps tie-break margin, trap reporting,
// value normalisation, the death-cap threshold and the root noise alpha. Everything else
// keeps Config's own default, which is zero or false, so a field the caller forgets is
// inert rather than silently paper-flavoured.
inline MonteCarloSearch::Config paperSearchDefaults(int board)
{
    MonteCarloSearch::Config config;
    config.exploration = EXPLORATION;
    config.discount = deriveDiscount(board);
    config.step_reward = STEP_REWARD;
    config.steps_tiebreak_margin = STEPS_TIEBREAK_MARGIN;
    config.trap_report = TRAP_REPORT;
    config.normalize_values = NORMALIZE_VALUES;
    config.death_cap_threshold = DEATH_CAP_THRESHOLD;
    config.root_noise_alpha = ROOT_NOISE_ALPHA;
    return config;
}

}  // namespace az
