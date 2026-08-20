// Implementation of scoreCoverage. What the two rules mean, and how to read the ratio,
// are in coverage_tally.h.

#include <cassert>
#include <stdexcept>

#include "coverage_tally.h"

void CoverageTally::observe(int visited, int action_count)
{
    // A position with no actions is not a position, and a visited count outside the
    // available ones is a caller reading the wrong field.
    assert(action_count > 0);
    assert(visited >= 0 && visited <= action_count);

    positions++;
    visited_actions += static_cast<std::size_t>(visited);
    available_actions += static_cast<std::size_t>(action_count);
    // Coverage is every available action visited, not a fixed three: a rule offering
    // fewer actions is covered when it visits the ones it has.
    if (visited == action_count)
    {
        fully_covered++;
    }
}

CoverageReport scoreCoverage(const CoverageTally& tally)
{
    if (tally.positions == 0)
    {
        throw std::invalid_argument("coverage needs at least one observed position");
    }

    CoverageReport report;
    report.positions = tally.positions;
    const double positions = static_cast<double>(tally.positions);

    report.position_coverage = static_cast<double>(tally.fully_covered) / positions;
    report.mean_visited_actions = static_cast<double>(tally.visited_actions) / positions;

    // The current rule takes every action of a fully covered position and nothing from
    // the rest, so its yield is the covered positions times the actions each offered.
    // Averaged rather than assumed uniform: available_actions carries the real count.
    const double mean_actions = static_cast<double>(tally.available_actions) / positions;
    report.labels_all_or_nothing =
        static_cast<std::size_t>(static_cast<double>(tally.fully_covered) * mean_actions);
    // A per-action rule keeps every action that was visited, wherever it sat.
    report.labels_per_action = tally.visited_actions;

    // With nothing admitted the gain is unbounded, not absent, so the flag stays down
    // rather than the ratio reporting a number that would read as "no gain".
    report.yield_ratio_defined = report.labels_all_or_nothing > 0;
    if (report.yield_ratio_defined)
    {
        report.yield_ratio = static_cast<double>(report.labels_per_action) /
                             static_cast<double>(report.labels_all_or_nothing);
    }
    return report;
}
