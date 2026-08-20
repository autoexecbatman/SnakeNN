#pragma once

// The range of values a search has seen, so selection can compare them against a prior.
//
// PUCT adds an exploitation term to c_puct * prior * sqrt(N). That sum only means
// anything when the two sides share a scale: the paper's c_puct of 0.5 assumes values in
// [-1, 1]. Here they are raw returns bounded by az::VALUE_SCALE, so exploitation can be
// forty times the largest exploration term any prior can produce, and the constant stops
// having a meaning. Normalising to [0, 1] against what this tree has actually seen is
// MuZero's answer to the same problem.
//
// Usage:
//
//     ValueRange range;
//     range.observe(-4.0f);            // a losing line
//     range.observe(12.0f);            // a winning one
//
//     range.isEstablished();           // true - two distinct values seen
//     range.normalize(12.0f);          // 1.0, the best seen so far
//     range.normalize(-4.0f);          // 0.0, the worst
//     range.normalize(4.0f);           // 0.5, halfway between them
//
//     ValueRange fresh;
//     fresh.normalize(7.0f);           // 7.0 unchanged - nothing to normalise against
//
// Values outside what was observed are not clamped: normalize is linear, so a value above
// the maximum returns above 1. Observe a value before normalising it and that cannot
// arise. A range that has seen one value, or the same value repeatedly, is not
// established and returns its argument unchanged - dividing by a zero width would put an
// infinity into a comparison the search has to trust.
struct ValueRange
{
    // Widens the range to include `value`. Asserts the value is finite.
    void observe(float value);

    // `value` mapped so the lowest observed value is 0 and the highest is 1. Returns
    // `value` unchanged while the range is not established. Asserts the value is finite.
    float normalize(float value) const;

    // Whether two distinct values have been seen, which is what makes a width to divide
    // by. False on a fresh range and on one that has only ever seen one number.
    bool isEstablished() const;

private:
    // Ordered so an unobserved range fails every comparison rather than reporting a
    // width of zero around some arbitrary point.
    float lowest_{ 0.0f };
    float highest_{ 0.0f };
    bool seen_{ false };
};
