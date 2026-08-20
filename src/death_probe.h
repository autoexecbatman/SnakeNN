#pragma once

#include <vector>

// Scores a death head against the search's own backed-up risk.
//
// The question it answers: does the death head contain anything, or is it still at
// its initialisation? az10_death368.pt cost 4.8 hours and its logs cannot say,
// because the death loss was never printed. This measures the trained head instead
// of the run that produced it.
//
// Pairs come from held-out positions: the network's raw death output for a root
// action, and the risk the search backed up for that same action. Only positions
// where the search visited every root action are admissible - an unvisited action
// keeps its start value and reads as safe, so a pair drawn from one measures the
// search's coverage rather than the head.
//
// Usage:
//
//     DeathProbeSamples samples;
//     samples.add(0.81f, 1.0f);   // head said doomed, search agreed
//     samples.add(0.12f, 0.0f);   // head said safe, search agreed
//     samples.add(0.44f, 0.0f);
//
//     const DeathProbeReport report = scoreDeathProbe(samples, 0.5f);
//
//     report.estimate_spread     // sample standard deviation of the head's output
//     report.rank_correlation    // Spearman, in [-1, 1]; 0 is no information
//     report.ranking_auc         // in [0, 1]; 0.5 is a coin flip
//     report.doomed_fraction     // share of targets at or above the threshold
//
// Read estimate_spread first. A head at its initialisation emits nearly the same
// number everywhere, and a rank correlation computed over a near-constant vector is
// noise amplified by normalisation, not a weak signal.
//
// scoreDeathProbe refuses fewer than two samples and a threshold outside [0, 1].
// It asserts that every estimate and target it was given lies in [0, 1].

// One held-out position's root action: what the head said, and what the search
// backed up for the same action.
struct DeathProbePair
{
    // The network's death output for this action, in [0, 1]. Read from the network
    // directly - NetworkEvaluator zeroes this whenever az::DEATH_RISK_FROM_NETWORK
    // is false, and a probe reading through it scores a vector of zeros.
    float estimate{ 0.0f };
    // The search's backed-up risk for the same action, in [0, 1].
    float target{ 0.0f };
};

// The admissible pairs, and the count of what was rejected.
//
// Rejections are carried rather than dropped silently: a probe that admitted few
// positions is measuring a search that rarely covered its root, and that reads in a
// report exactly like a head with little to say.
struct DeathProbeSamples
{
    std::vector<DeathProbePair> pairs;
    // Positions skipped because the search left a root action unvisited.
    std::size_t rejected_uncovered{ 0 };

    void add(float estimate, float target);
};

// What the probe measured. Every field is a property of the pairs it was given.
struct DeathProbeReport
{
    std::size_t sample_count{ 0 };
    // Sample standard deviation of the head's output. Near zero means the head is
    // constant, which is what an untrained one looks like.
    double estimate_spread{ 0.0 };
    double estimate_mean{ 0.0 };
    // Spearman rank correlation between estimate and target, in [-1, 1].
    double rank_correlation{ 0.0 };
    // Area under the ROC curve, target binarised at the threshold, in [0, 1]. 0.5 is
    // a coin flip. Undefined when every target falls on one side, and reported as
    // 0.5 with ranking_auc_defined false rather than as a number nobody can read.
    double ranking_auc{ 0.5 };
    bool ranking_auc_defined{ false };
    // Share of targets at or above the threshold.
    double doomed_fraction{ 0.0 };
};

// Scores the pairs. `doomed_threshold` binarises the target for the AUC only; the
// rank correlation uses the continuous target.
//
// Throws std::invalid_argument on fewer than two pairs, or a threshold outside
// [0, 1]. Asserts every estimate and target is in [0, 1].
DeathProbeReport scoreDeathProbe(const DeathProbeSamples& samples, float doomed_threshold);
