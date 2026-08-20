// Implementation of scoreDeathProbe. What the statistics mean, and how to use them,
// are in death_probe.h.

#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "death_probe.h"

namespace
{

// Midranks: tied values share the average of the positions they occupy, which is what
// makes Spearman on ties equal Pearson on the rank vectors.
std::vector<double> midranks(const std::vector<float>& values)
{
    // Positions sorted by the value they point at, so equal values end up adjacent.
    std::vector<std::size_t> order(values.size());
    std::iota(order.begin(), order.end(), std::size_t{ 0 });
    std::sort(order.begin(), order.end(), [&values](std::size_t left, std::size_t right)
              { return values[left] < values[right]; });

    std::vector<double> ranks(values.size(), 0.0);
    std::size_t start = 0;
    while (start < order.size())
    {
        // Extend over every position holding the same value as this one.
        std::size_t stop = start + 1;
        while (stop < order.size() && values[order[stop]] == values[order[start]])
        {
            stop++;
        }
        // Ranks are one-based, so the block spans start+1 .. stop and averages to this.
        const double shared = (static_cast<double>(start + 1) + static_cast<double>(stop)) / 2.0;
        for (std::size_t position = start; position < stop; position++)
        {
            ranks[order[position]] = shared;
        }
        start = stop;
    }
    return ranks;
}

// Pearson correlation. Returns 0 when either input has no variation, because a
// correlation with a constant is undefined rather than zero and 0 is what the report
// documents as "no information".
double pearson(const std::vector<double>& left, const std::vector<double>& right)
{
    assert(left.size() == right.size());
    const double count = static_cast<double>(left.size());
    const double left_mean = std::accumulate(left.begin(), left.end(), 0.0) / count;
    const double right_mean = std::accumulate(right.begin(), right.end(), 0.0) / count;

    double covariance = 0.0;
    double left_square = 0.0;
    double right_square = 0.0;
    for (std::size_t index = 0; index < left.size(); index++)
    {
        const double left_deviation = left[index] - left_mean;
        const double right_deviation = right[index] - right_mean;
        covariance += left_deviation * right_deviation;
        left_square += left_deviation * left_deviation;
        right_square += right_deviation * right_deviation;
    }

    if (left_square <= 0.0 || right_square <= 0.0)
    {
        return 0.0;
    }
    return covariance / std::sqrt(left_square * right_square);
}

}  // namespace

void DeathProbeSamples::add(float estimate, float target)
{
    pairs.push_back(DeathProbePair{ estimate, target });
}

DeathProbeReport scoreDeathProbe(const DeathProbeSamples& samples, float doomed_threshold)
{
    // Two is the floor: a sample standard deviation needs n-1 in the denominator, and
    // one pair admits no ranking at all.
    if (samples.pairs.size() < 2)
    {
        throw std::invalid_argument(
            "death probe needs at least two pairs - a spread and a ranking are undefined below "
            "that");
    }
    if (!(doomed_threshold >= 0.0f) || !(doomed_threshold <= 1.0f))
    {
        throw std::invalid_argument("doomed threshold must lie in [0, 1]");
    }

    std::vector<float> estimates;
    std::vector<float> targets;
    estimates.reserve(samples.pairs.size());
    targets.reserve(samples.pairs.size());
    for (const DeathProbePair& pair : samples.pairs)
    {
        // Both are probabilities by contract; anything else is a caller that read the
        // wrong tensor, and it fails here rather than producing a plausible number.
        assert(pair.estimate >= 0.0f && pair.estimate <= 1.0f);
        assert(pair.target >= 0.0f && pair.target <= 1.0f);
        estimates.push_back(pair.estimate);
        targets.push_back(pair.target);
    }

    DeathProbeReport report;
    report.sample_count = samples.pairs.size();
    const double count = static_cast<double>(report.sample_count);

    // Mean and sample standard deviation of the head's output. Read first: a
    // near-constant head makes every statistic below noise rather than signal.
    report.estimate_mean =
        std::accumulate(estimates.begin(), estimates.end(), 0.0,
                        [](double total, float value) { return total + value; }) /
        count;
    double squared_deviation = 0.0;
    for (float estimate : estimates)
    {
        const double deviation = estimate - report.estimate_mean;
        squared_deviation += deviation * deviation;
    }
    report.estimate_spread = std::sqrt(squared_deviation / (count - 1.0));

    // Spearman is Pearson on the midranks of both vectors.
    report.rank_correlation = pearson(midranks(estimates), midranks(targets));

    // The threshold splits the targets, and only here.
    std::vector<float> doomed_estimates;
    std::vector<float> safe_estimates;
    for (const DeathProbePair& pair : samples.pairs)
    {
        if (pair.target >= doomed_threshold)
        {
            doomed_estimates.push_back(pair.estimate);
        }
        else
        {
            safe_estimates.push_back(pair.estimate);
        }
    }
    report.doomed_fraction = static_cast<double>(doomed_estimates.size()) / count;

    // With every target on one side there is no ordered pair to score, so the AUC keeps
    // its neutral value and the flag stays down rather than reporting a coin flip.
    report.ranking_auc_defined = !doomed_estimates.empty() && !safe_estimates.empty();
    if (!report.ranking_auc_defined)
    {
        return report;
    }

    // The rank-sum form of the AUC: over every (safe, doomed) pair, credit a full point
    // when the head ranks the doomed one higher and a half when they tie.
    double concordant = 0.0;
    for (float doomed : doomed_estimates)
    {
        for (float safe : safe_estimates)
        {
            if (doomed > safe)
            {
                concordant += 1.0;
            }
            else if (doomed == safe)
            {
                concordant += 0.5;
            }
        }
    }
    report.ranking_auc = concordant / (static_cast<double>(doomed_estimates.size()) *
                                       static_cast<double>(safe_estimates.size()));
    return report;
}
