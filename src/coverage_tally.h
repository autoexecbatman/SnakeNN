#pragma once

// Counts how often the search covers its root, and what that costs in death labels.
//
// A death-risk label is trainable only where the search visited every root action: an
// unvisited action keeps its start value and reads as safe. A sharp policy pours its
// simulations down one action, so coverage falls as the agent improves - and the
// question this answers is how much label yield that rule gives away.
//
// Two rules are counted side by side on the same positions. All-or-nothing is what the
// code does today: a position yields ACTION_COUNT labels when every action was visited
// and none otherwise. Per-action yields one label for each action that was visited,
// whatever the others did. The ratio between them is what a redesign would buy.
//
// Usage:
//
//     CoverageTally tally;
//     tally.observe(3, 3);   // a position where all 3 of 3 actions were visited
//     tally.observe(1, 3);   // a position where the search only ever tried one
//     tally.observe(2, 3);
//
//     const CoverageReport report = scoreCoverage(tally);
//
//     report.position_coverage       // 1 of 3 positions fully covered -> 0.3333
//     report.labels_all_or_nothing   // 3 labels from that one position -> 3
//     report.labels_per_action       // 3 + 1 + 2 -> 6
//     report.yield_ratio             // 6 / 3 -> 2.0, what the redesign would gain
//
// scoreCoverage refuses a tally holding no positions. observe asserts that the visited
// count is in [0, action_count] and that action_count is positive.

// Positions seen, and the visited-action counts accumulated over them.
struct CoverageTally
{
    std::size_t positions{ 0 };
    // Positions where every root action was visited.
    std::size_t fully_covered{ 0 };
    // Summed over positions: how many root actions the search visited at all.
    std::size_t visited_actions{ 0 };
    // Summed over positions: how many actions there were to visit.
    std::size_t available_actions{ 0 };

    // Records one root position. `visited` is how many of `action_count` root actions
    // the search gave at least one visit.
    void observe(int visited, int action_count);
};

// What the tally says about coverage and about label yield.
struct CoverageReport
{
    std::size_t positions{ 0 };
    // Share of positions where every root action was visited, in [0, 1]. This is the
    // fraction of positions the current rule admits at all.
    double position_coverage{ 0.0 };
    // Labels the current all-or-nothing rule produces.
    std::size_t labels_all_or_nothing{ 0 };
    // Labels a per-action rule would produce on the same positions.
    std::size_t labels_per_action{ 0 };
    // labels_per_action / labels_all_or_nothing. Infinite is not representable here, so
    // a run that admits no labels at all reports 0 with yield_ratio_defined false - the
    // gain from a redesign is unbounded there rather than absent.
    double yield_ratio{ 0.0 };
    bool yield_ratio_defined{ false };
    // Mean visited actions per position, whatever the rule.
    double mean_visited_actions{ 0.0 };
};

// Scores the tally. Throws std::invalid_argument when no position was observed.
CoverageReport scoreCoverage(const CoverageTally& tally);
