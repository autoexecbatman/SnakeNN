// Implementation of ValueRange. What the normalisation is for, and what it refuses to
// do, are in value_range.h.

#include <algorithm>
#include <cassert>
#include <cmath>

#include "value_range.h"

void ValueRange::observe(float value)
{
    // An infinity or a NaN here would silently disable normalisation for the whole tree:
    // every width becomes infinite or every comparison false.
    assert(std::isfinite(value) && "a value observed by the range is not finite");

    // The first value seeds both ends; there is no neutral pair to widen from.
    if (!seen_)
    {
        lowest_ = value;
        highest_ = value;
        seen_ = true;
        return;
    }

    lowest_ = std::min(lowest_, value);
    highest_ = std::max(highest_, value);
}

float ValueRange::normalize(float value) const
{
    assert(std::isfinite(value) && "a value handed to the range is not finite");

    // Nothing to normalise against, so the caller gets back what it gave. A zero width
    // divides to an infinity, which is worse than an unnormalised comparison.
    if (!isEstablished())
    {
        return value;
    }

    // Linear and unclamped: a value outside what was observed lands outside [0, 1], and
    // the contract says so rather than hiding it behind a clamp.
    return (value - lowest_) / (highest_ - lowest_);
}

bool ValueRange::isEstablished() const
{
    // Strictly greater, so a range that has only seen one number reports false and the
    // division above is never reached with a zero denominator.
    return seen_ && highest_ > lowest_;
}
