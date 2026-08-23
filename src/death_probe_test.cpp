// Checks the statistics that decide whether the death head learned anything.
//
// What is being scored. The death head predicts, per move, the chance that move leads to a
// death no later play can avoid. Asking whether it works is not a matter of accuracy: a
// head that answers 0.5 everywhere is right about half the time and useless. The probe
// instead reports a spread, a rank correlation and a ranking AUC, and this file pins down
// what each of those means so a reading of the real probe can be trusted.
//
// Spread comes first, and the report says so. An untrained sigmoid head emits its
// initialisation everywhere, so its spread is near zero while its mean sits at 0.5 - that
// is the shape of nothing, and it is the only statistic that catches it without needing a
// second network to compare against.
//
// Why this is a test rather than a script. Every number here is arithmetic over pairs of
// floats, so it can be derived by hand and checked exactly with no network, no search and
// no GPU. Every expected value below comes from the contract in death_probe.h with the
// derivation written beside it; none was read off the implementation, which would only
// record what the code already does.
//
// Run it:
//
//     cmake --build build --config Release --target DeathProbeTest
//     build\Release\DeathProbeTest.exe
//
// Silent unless something fails, ending in either
//
//     all death probe checks pass
//
// or a count of failing checks and a non-zero exit.
//
// Four properties are worth knowing before reading a probe report. A correlation of -1 is a
// different finding from 0: a head that is exactly wrong has learned something inverted,
// while a silent one has learned nothing, and the sign is what separates them. Tied
// estimates take midranks, so a head with no opinion cannot score a spurious +1. An AUC
// over targets that all sit on one side of the threshold is undefined rather than 0.5,
// because 0.5 alone would read as a coin flip, which is a measurement nobody took. And the
// threshold binarises the target for the AUC and for nothing else, so moving it must move
// the doomed fraction and leave the rank correlation untouched.

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

// Checks that did not hold. main prints the count and returns 1 when it is non-zero.
int failures = 0;

// Compares two doubles within a tolerance, and counts a mismatch.
//
//     expectNear("spread", report.spread, 0.1414214, 1e-6);
//
// A NaN fails rather than comparing false quietly. Every statistic here either has a value
// or carries a flag saying it has none, so a NaN means the arithmetic went wrong rather
// than the case being undefined - and a silent NaN would pass an inequality test.
void expectNear(const std::string& what, double actual, double expected, double tolerance)
{
    if (std::isnan(actual) || std::abs(actual - expected) > tolerance)
    {
        std::cout << std::format("[FAIL] {}: expected {:.6f}, got {:.6f}\n", what, expected,
                                 actual);
        failures++;
    }
}

// Checks a flag, naming it when it does not hold.
//
//     expectTrue("auc is undefined with one-sided targets", !report.auc_defined);
//
// Used for the defined-ness flags, where the question is whether a statistic exists at all
// rather than what it equals.
void expectTrue(const std::string& what, bool condition)
{
    if (!condition)
    {
        std::cout << std::format("[FAIL] {}\n", what);
        failures++;
    }
}

// Builds a sample set from literal (estimate, target) pairs.
//
//     const DeathProbeSamples samples = makeSamples({ { 0.2f, 0.0f }, { 0.4f, 1.0f } });
//     // head said 0.2 for a move that survived, 0.4 for one that died
//
// The estimate is what the head predicted; the target is what actually happened. Writing
// the pairs out as literals is what lets every expected statistic in this file be derived
// on paper - a fixture that generated them would move the arithmetic out of reach.
DeathProbeSamples makeSamples(const std::vector<std::pair<float, float>>& raw)
{
    DeathProbeSamples samples;
    for (const auto& [estimate, target] : raw)
    {
        samples.add(estimate, target);
    }
    return samples;
}

// Checks that scoreDeathProbe rejects a sample set it cannot score.
//
//     expectRefused("empty samples", DeathProbeSamples{}, 0.5f);
//
// The catch is narrowed to std::invalid_argument and any other exception fails the check.
// "It threw something" would pass even when the refusal came from an unrelated fault, which
// is the way a refusal test quietly stops testing the refusal.
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
    samples.rejected = 17;
    const DeathProbeReport report = scoreDeathProbe(samples, 0.5f);
    // sample_count counts pairs, not pairs plus rejections: an implementation that
    // added rejected in would report 19 here.
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

// Runs every case, then reports. Returns 1 if any check failed, 0 otherwise.
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
