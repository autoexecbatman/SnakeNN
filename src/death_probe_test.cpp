#include <cmath>
#include <format>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "death_probe.h"

namespace
{

int failures = 0;

// Every expected value below is derived from the contract in death_probe.h, by hand,
// with the derivation written beside it. None was read off an implementation.
void expectNear(const std::string& what, double actual, double expected, double tolerance)
{
    if (std::isnan(actual) || std::abs(actual - expected) > tolerance)
    {
        std::cout << std::format("[FAIL] {}: expected {:.6f}, got {:.6f}\n", what, expected,
                                 actual);
        failures++;
    }
}

void expectTrue(const std::string& what, bool condition)
{
    if (!condition)
    {
        std::cout << std::format("[FAIL] {}\n", what);
        failures++;
    }
}

DeathProbeSamples makeSamples(const std::vector<std::pair<float, float>>& raw)
{
    DeathProbeSamples samples;
    for (const auto& [estimate, target] : raw)
    {
        samples.add(estimate, target);
    }
    return samples;
}

void expectRefused(const std::string& what, const DeathProbeSamples& samples, float threshold)
{
    try
    {
        (void)scoreDeathProbe(samples, threshold);
        std::cout << std::format("[FAIL] {}: was accepted\n", what);
        failures++;
    }
    catch (const std::invalid_argument&)
    {
    }
    catch (const std::exception& error)
    {
        std::cout << std::format("[FAIL] {}: wrong exception: {}\n", what, error.what());
        failures++;
    }
}

// A head at its initialisation emits the same number everywhere. Spread is the
// statistic that catches it, and the report says to read it first.
void constantHeadHasZeroSpread()
{
    const DeathProbeReport flat =
        scoreDeathProbe(makeSamples({ { 0.5f, 0.0f }, { 0.5f, 1.0f }, { 0.5f, 0.0f } }), 0.5f);
    // All three estimates are equal, so the sample standard deviation is exactly 0.
    expectNear("constant head spread", flat.estimate_spread, 0.0, 1e-9);
    expectNear("constant head mean", flat.estimate_mean, 0.5, 1e-9);
    // Every estimate ties, so no pair is ordered and the ranking is a coin flip - but
    // both classes are present, so the AUC has a value and the flag must say so.
    expectNear("constant head auc", flat.ranking_auc, 0.5, 1e-9);
    expectTrue("constant head auc is defined", flat.ranking_auc_defined);

    // The same shape with one estimate moved. Stated against this alternative rather
    // than against the constant 0: a spread that always returns zero passes the line
    // above and fails here.
    const DeathProbeReport varied =
        scoreDeathProbe(makeSamples({ { 0.5f, 0.0f }, { 0.9f, 1.0f }, { 0.5f, 0.0f } }), 0.5f);
    expectTrue("a varying head has more spread than a flat one",
               varied.estimate_spread > flat.estimate_spread);
}

// Sample standard deviation, n-1 in the denominator: the mean of 0.2 and 0.4 is 0.3,
// the deviations are -0.1 and +0.1, so sd = sqrt((0.01 + 0.01) / 1) = 0.1414214.
void spreadIsTheSampleDeviation()
{
    const DeathProbeReport report =
        scoreDeathProbe(makeSamples({ { 0.2f, 0.0f }, { 0.4f, 1.0f } }), 0.5f);
    expectNear("two-sample spread", report.estimate_spread, 0.1414214, 1e-6);
    expectNear("two-sample mean", report.estimate_mean, 0.3, 1e-6);
}

// Estimates ordered exactly as the targets are, and the targets distinct, so both rank
// vectors are 1,2,3,4 and Spearman is +1. The targets must be distinct for that: with
// two at 0 and two at 1 they take midranks 1.5,1.5,3.5,3.5 and the correlation against
// 1,2,3,4 is 4/sqrt(20) = 0.894427, which is the ceiling a tied target allows.
void perfectRankingScoresOne()
{
    const DeathProbeReport report = scoreDeathProbe(
        makeSamples({ { 0.1f, 0.0f }, { 0.4f, 0.2f }, { 0.7f, 0.8f }, { 0.9f, 1.0f } }), 0.5f);
    expectNear("perfect rho", report.rank_correlation, 1.0, 1e-9);
    expectNear("perfect auc", report.ranking_auc, 1.0, 1e-9);
    expectTrue("perfect auc is defined", report.ranking_auc_defined);
    // Two of the four targets are at or above 0.5.
    expectNear("doomed fraction", report.doomed_fraction, 0.5, 1e-9);
}

// The same pairs with the order of the head reversed. A head that is exactly wrong is
// a different finding from one that is silent, and the sign is what separates them.
void invertedRankingScoresMinusOne()
{
    const DeathProbeReport report = scoreDeathProbe(
        makeSamples({ { 0.9f, 0.0f }, { 0.7f, 0.2f }, { 0.4f, 0.8f }, { 0.1f, 1.0f } }), 0.5f);
    expectNear("inverted rho", report.rank_correlation, -1.0, 1e-9);
    expectNear("inverted auc", report.ranking_auc, 0.0, 1e-9);
}

// Spearman is Pearson on ranks, and tied estimates take their midrank. Estimates
// (0.5, 0.5, 0.9) rank (1.5, 1.5, 3); targets (0.0, 1.0, 1.0) rank (1, 2.5, 2.5).
// Deviations from the common mean of 2 are (-0.5, -0.5, 1) and (-1, 0.5, 0.5), so the
// covariance sum is 0.5 - 0.25 + 0.5 = 0.75 and each sum of squares is 1.5.
// rho = 0.75 / 1.5 = 0.5 exactly.
void tiedRanksTakeTheMidrank()
{
    const DeathProbeReport report =
        scoreDeathProbe(makeSamples({ { 0.5f, 0.0f }, { 0.5f, 1.0f }, { 0.9f, 1.0f } }), 0.5f);
    expectNear("tied rho", report.rank_correlation, 0.5, 1e-6);

    // The case above cannot tell midranks from ordinary ranks - it yields 0.5 either
    // way, which a mutation sweep found. This one separates them. Estimates
    // (0.2, 0.5, 0.5, 0.9) take midranks (1, 2.5, 2.5, 4) against targets ranked
    // (1, 2, 3, 4): covariance 4.5 over sqrt(4.5 * 5) = 0.948683. Ranking the tied
    // pair (1, 2, 2, 4) instead gives covariance 4.5 over sqrt(4.75 * 5) = 0.923445.
    const DeathProbeReport discriminating = scoreDeathProbe(
        makeSamples({ { 0.2f, 0.0f }, { 0.5f, 0.3f }, { 0.5f, 0.7f }, { 0.9f, 1.0f } }), 0.5f);
    expectNear("midrank rho", discriminating.rank_correlation, 0.948683, 1e-6);
}

// The threshold admits a target that sits exactly on it. Without a target at the
// boundary, >= and > agree everywhere and neither test nor mutation can see the
// difference - which a sweep found and this fixes.
void thresholdIncludesItsBoundary()
{
    const DeathProbeReport report = scoreDeathProbe(
        makeSamples({ { 0.1f, 0.0f }, { 0.4f, 0.5f }, { 0.7f, 0.5f }, { 0.9f, 1.0f } }), 0.5f);
    // Three targets are at or above 0.5; strictly-greater would count only the 1.0.
    expectNear("boundary target is doomed", report.doomed_fraction, 0.75, 1e-9);
}

// One safe and two doomed, the safe estimate tying one doomed estimate. The ordered
// pairs are (safe 0.5, doomed 0.5), which ties and counts a half, and (safe 0.5,
// doomed 0.9), which the head gets right. AUC = (0.5 + 1) / 2 = 0.75.
void tiedEstimatesCountAHalf()
{
    const DeathProbeReport report =
        scoreDeathProbe(makeSamples({ { 0.5f, 0.0f }, { 0.5f, 1.0f }, { 0.9f, 1.0f } }), 0.5f);
    expectNear("tied auc", report.ranking_auc, 0.75, 1e-9);
}

// With every target on one side of the threshold there is no ordered pair, so the AUC
// has no value. Reporting 0.5 with the flag down says that; reporting 0.5 alone would
// read as a coin flip, which is a measurement nobody took.
void oneSidedTargetsLeaveAucUndefined()
{
    const DeathProbeReport all_safe =
        scoreDeathProbe(makeSamples({ { 0.2f, 0.0f }, { 0.8f, 0.0f } }), 0.5f);
    expectTrue("all-safe auc is undefined", !all_safe.ranking_auc_defined);
    expectNear("all-safe doomed fraction", all_safe.doomed_fraction, 0.0, 1e-9);
    // Paired with the all-doomed case below, which pins the same field at 1. A
    // doomed_fraction stuck at either constant fails one of the two.
    expectTrue("all-safe scored both pairs", all_safe.sample_count == 2);

    const DeathProbeReport all_doomed =
        scoreDeathProbe(makeSamples({ { 0.2f, 1.0f }, { 0.8f, 1.0f } }), 0.5f);
    expectTrue("all-doomed auc is undefined", !all_doomed.ranking_auc_defined);
    expectNear("all-doomed doomed fraction", all_doomed.doomed_fraction, 1.0, 1e-9);

    // The flag is not always false: a two-sided target sets it. Without this line an
    // implementation that never defines the AUC passes both cases above.
    const DeathProbeReport two_sided =
        scoreDeathProbe(makeSamples({ { 0.2f, 0.0f }, { 0.8f, 1.0f } }), 0.5f);
    expectTrue("two-sided auc is defined", two_sided.ranking_auc_defined);
}

// The threshold binarises the target for the AUC and for nothing else, so moving it
// must move doomed_fraction and leave the rank correlation alone.
void thresholdMovesOnlyTheBinarisation()
{
    const DeathProbeSamples samples =
        makeSamples({ { 0.1f, 0.1f }, { 0.4f, 0.3f }, { 0.7f, 0.6f }, { 0.9f, 0.9f } });
    const DeathProbeReport low = scoreDeathProbe(samples, 0.2f);
    const DeathProbeReport high = scoreDeathProbe(samples, 0.8f);
    // Three targets are at or above 0.2; one is at or above 0.8.
    expectNear("low threshold doomed fraction", low.doomed_fraction, 0.75, 1e-9);
    expectNear("high threshold doomed fraction", high.doomed_fraction, 0.25, 1e-9);
    // The estimates and targets are in the same order either way, so rho is 1 for both.
    expectNear("low threshold rho", low.rank_correlation, 1.0, 1e-9);
    expectNear("high threshold rho", high.rank_correlation, 1.0, 1e-9);
}

// A rejected position is not a sample. The count travels with the samples so a report
// built on almost nothing cannot be mistaken for a head with nothing to say.
void rejectionsAreCarriedNotDropped()
{
    DeathProbeSamples samples = makeSamples({ { 0.2f, 0.0f }, { 0.8f, 1.0f } });
    samples.rejected_uncovered = 17;
    const DeathProbeReport report = scoreDeathProbe(samples, 0.5f);
    // sample_count counts pairs, not pairs plus rejections: an implementation that
    // added rejected_uncovered in would report 19 here.
    expectTrue("sample count excludes rejections", report.sample_count == 2);

    // And it tracks the pairs rather than being a constant.
    DeathProbeSamples longer =
        makeSamples({ { 0.2f, 0.0f }, { 0.5f, 0.0f }, { 0.8f, 1.0f }, { 0.9f, 1.0f } });
    expectTrue("sample count follows the pairs", scoreDeathProbe(longer, 0.5f).sample_count == 4);
}

void refusals()
{
    expectRefused("empty samples", DeathProbeSamples{}, 0.5f);
    expectRefused("one sample", makeSamples({ { 0.5f, 1.0f } }), 0.5f);
    expectRefused("threshold above one", makeSamples({ { 0.2f, 0.0f }, { 0.8f, 1.0f } }), 1.5f);
    expectRefused("threshold below zero", makeSamples({ { 0.2f, 0.0f }, { 0.8f, 1.0f } }), -0.1f);
}

}  // namespace

int main()
{
    constantHeadHasZeroSpread();
    spreadIsTheSampleDeviation();
    perfectRankingScoresOne();
    invertedRankingScoresMinusOne();
    tiedRanksTakeTheMidrank();
    thresholdIncludesItsBoundary();
    tiedEstimatesCountAHalf();
    oneSidedTargetsLeaveAucUndefined();
    thresholdMovesOnlyTheBinarisation();
    rejectionsAreCarriedNotDropped();
    refusals();

    if (failures > 0)
    {
        std::cout << std::format("\n{} failing checks\n", failures);
        return 1;
    }
    std::cout << "all death probe checks pass\n";
    return 0;
}
