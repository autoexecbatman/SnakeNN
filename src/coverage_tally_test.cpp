// Checks the arithmetic that decides how many death-risk labels a run of search yields.
//
// The thing under test. A death-risk label teaches the network which moves lead to a death
// no later play can avoid. The current rule admits a label only where the search visited
// every root action, on the grounds that a move nobody tried has no evidence attached. That
// rule is expensive: a confident policy pours its simulations down one action, so coverage
// falls exactly as the agent improves. CoverageTally counts what the search reached and
// scoreCoverage turns it into two numbers - what the all-or-nothing rule admits, and what a
// per-action rule that kept every visited move would admit. Their ratio is what changing
// the rule would buy at the same search cost.
//
// Why it is worth a test of its own. The tally is pure arithmetic over counts, so it can be
// checked exactly, by hand, with no network and no GPU - and it decides a design question
// that costs hours to answer any other way. Every expected value below is derived from the
// contract in coverage_tally.h with the arithmetic written beside it. None was read off the
// implementation, which would only have recorded what the code already did.
//
// Run it:
//
//     cmake --build build --config Release --target CoverageTallyTest
//     build\Release\CoverageTallyTest.exe
//
// It is silent unless something fails, and ends with either
//
//     all coverage tally checks pass
//
// or a count of failing checks and a non-zero exit.
//
// The five cases, and the distinct thing each pins down: the header's worked example, where
// the two rules disagree two to one; full coverage, where they agree exactly and a redesign
// buys nothing; no coverage, where the ratio is undefined rather than zero, because an
// unbounded gain must not read as no gain; a position the search never entered, which
// belongs in the denominator and in no numerator; and a position offering two actions rather
// than three, because coverage means visiting what is available, not visiting three.

#include <cmath>
#include <format>
#include <iostream>
#include <stdexcept>
#include <string>

#include "coverage_tally.h"

namespace
{

// Checks that did not hold. main prints the count and returns 1 when it is non-zero.
int failures = 0;

// Compares two doubles within a tolerance, and counts a mismatch.
//
//     expectNear("yield ratio", report.yield_ratio, 2.0, 1e-9);
//
// A NaN fails rather than comparing false quietly: every ratio here has a defined value or
// a flag saying it has none, so a NaN means the arithmetic went wrong rather than that the
// case was undefined. The tolerance is the caller's, because a count converted to a double
// and a ratio of two counts do not deserve the same slack.
void expectNear(const std::string& what, double actual, double expected, double tolerance)
{
    if (std::isnan(actual) || std::abs(actual - expected) > tolerance)
    {
        std::cout << std::format("[FAIL] {}: expected {:.6f}, got {:.6f}\n", what, expected,
                                 actual);
        failures++;
    }
}

// Compares two counts exactly, and reports both when they differ.
//
//     expectCount("per-action labels", report.labels_per_action, 6);
//
// Exact rather than near: these are labels and positions, and a label count that is nearly
// right is wrong.
void expectCount(const std::string& what, std::size_t actual, std::size_t expected)
{
    if (actual != expected)
    {
        std::cout << std::format("[FAIL] {}: expected {}, got {}\n", what, expected, actual);
        failures++;
    }
}

// Checks a flag, naming it when it does not hold.
//
//     expectTrue("ratio is undefined with no admitted labels", !report.yield_ratio_defined);
//
// Used for the defined-ness flags, where the question is whether a number exists at all
// rather than what it equals.
void expectTrue(const std::string& what, bool condition)
{
    if (!condition)
    {
        std::cout << std::format("[FAIL] {}\n", what);
        failures++;
    }
}

// The worked example from the header. Three positions visiting 3, 1 and 2 of 3 actions:
// one is fully covered, so all-or-nothing yields 3 labels and per-action yields 6.
void headerExample()
{
    CoverageTally tally;
    tally.observe(3, 3);
    tally.observe(1, 3);
    tally.observe(2, 3);
    const CoverageReport report = scoreCoverage(tally);

    expectCount("positions", report.positions, 3);
    expectNear("position coverage", report.position_coverage, 1.0 / 3.0, 1e-9);
    expectCount("all-or-nothing labels", report.labels_all_or_nothing, 3);
    expectCount("per-action labels", report.labels_per_action, 6);
    expectNear("yield ratio", report.yield_ratio, 2.0, 1e-9);
    expectTrue("yield ratio is defined", report.yield_ratio_defined);
    // (3 + 1 + 2) / 3 positions = 2.
    expectNear("mean visited actions", report.mean_visited_actions, 2.0, 1e-9);
}

// Full coverage everywhere: the two rules agree exactly, so a redesign buys nothing and
// the ratio is 1. Stated against the example above, where it is 2.
void fullCoverageGainsNothing()
{
    CoverageTally tally;
    tally.observe(3, 3);
    tally.observe(3, 3);
    const CoverageReport report = scoreCoverage(tally);

    expectNear("full coverage", report.position_coverage, 1.0, 1e-9);
    expectCount("full all-or-nothing labels", report.labels_all_or_nothing, 6);
    expectCount("full per-action labels", report.labels_per_action, 6);
    expectNear("full yield ratio", report.yield_ratio, 1.0, 1e-9);
}

// No position fully covered, so the current rule admits nothing while a per-action rule
// still finds labels. The gain is unbounded rather than absent, and the flag says so
// instead of the ratio reporting a number that would read as "no gain".
void noCoverageLeavesTheRatioUndefined()
{
    CoverageTally tally;
    tally.observe(1, 3);
    tally.observe(2, 3);
    const CoverageReport report = scoreCoverage(tally);

    expectNear("zero coverage", report.position_coverage, 0.0, 1e-9);
    expectCount("zero all-or-nothing labels", report.labels_all_or_nothing, 0);
    expectCount("per-action labels survive", report.labels_per_action, 3);
    expectTrue("ratio is undefined with no admitted labels", !report.yield_ratio_defined);
}

// A position where the search visited nothing contributes to the denominator and to no
// numerator. Counting it as covered, or dropping it, both flatter the result.
void unvisitedPositionsStillCount()
{
    CoverageTally tally;
    tally.observe(0, 3);
    tally.observe(3, 3);
    const CoverageReport report = scoreCoverage(tally);

    expectCount("positions include the unvisited one", report.positions, 2);
    // One of two positions covered.
    expectNear("coverage with an empty position", report.position_coverage, 0.5, 1e-9);
    expectNear("mean visited with an empty position", report.mean_visited_actions, 1.5, 1e-9);
}

// Coverage is about visiting every available action, not about visiting three. A board
// or rule offering fewer actions is fully covered when it visits the ones it has.
void coverageIsRelativeToAvailableActions()
{
    CoverageTally tally;
    tally.observe(2, 2);
    const CoverageReport report = scoreCoverage(tally);

    expectNear("two of two is full coverage", report.position_coverage, 1.0, 1e-9);
    // Two actions available and visited, so both rules yield 2.
    expectCount("labels follow the action count", report.labels_all_or_nothing, 2);
    expectCount("per-action labels follow the action count", report.labels_per_action, 2);
}

// What scoreCoverage rejects: a tally holding no positions.
//
// A coverage rate over zero positions has no value, and returning one - zero, or one, or a
// NaN - would be a number a caller could act on. It throws instead. The catch is narrowed
// to std::invalid_argument and a different exception fails, because "it threw something"
// would pass even if the refusal came from an unrelated fault.
void refusals()
{
    try
    {
        (void)scoreCoverage(CoverageTally{});
        std::cout << "[FAIL] empty tally was accepted\n";
        failures++;
    }
    catch (const std::invalid_argument&)
    {
    }
    catch (const std::exception& error)
    {
        std::cout << std::format("[FAIL] empty tally: wrong exception: {}\n", error.what());
        failures++;
    }
}

}  // namespace

// Runs every case, then reports. Returns 1 if any check failed, 0 otherwise.
int main()
{
    headerExample();
    fullCoverageGainsNothing();
    noCoverageLeavesTheRatioUndefined();
    unvisitedPositionsStillCount();
    coverageIsRelativeToAvailableActions();
    refusals();

    if (failures > 0)
    {
        std::cout << std::format("\n{} failing checks\n", failures);
        return 1;
    }
    std::cout << "all coverage tally checks pass\n";
    return 0;
}
