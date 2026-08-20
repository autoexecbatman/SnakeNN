#include <cmath>
#include <format>
#include <iostream>
#include <string>

#include "value_range.h"

namespace
{

int failures = 0;

// Every expected value is derived from the contract in value_range.h by hand, with the
// arithmetic beside it. None was read off an implementation.
void expectNear(const std::string& what, float actual, float expected, float tolerance)
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

// The worked example from the header.
void headerExample()
{
    ValueRange range;
    range.observe(-4.0f);
    range.observe(12.0f);

    expectTrue("two distinct values establish the range", range.isEstablished());
    // Width is 12 - (-4) = 16. The endpoints map to the endpoints.
    expectNear("highest maps to one", range.normalize(12.0f), 1.0f, 1e-6f);
    expectNear("lowest maps to zero", range.normalize(-4.0f), 0.0f, 1e-6f);
    // 4 sits 8 above the lowest, and 8/16 = 0.5.
    expectNear("midpoint maps to a half", range.normalize(4.0f), 0.5f, 1e-6f);
}

// Nothing observed: there is no scale, so the value has to come back untouched. Stated
// against the established case above, where 12 maps to 1 rather than to itself.
void freshRangeIsATransparentPassThrough()
{
    ValueRange fresh;
    expectTrue("a fresh range is not established", !fresh.isEstablished());
    expectNear("a fresh range passes its argument through", fresh.normalize(7.0f), 7.0f, 1e-6f);
    expectNear("and does so for a negative too", fresh.normalize(-3.5f), -3.5f, 1e-6f);
}

// One value, or the same value repeatedly, gives a width of zero. Dividing by it would
// put an infinity into the search's comparison, so the range stays unestablished.
void aZeroWidthRangeIsNotEstablished()
{
    ValueRange single;
    single.observe(5.0f);
    expectTrue("one value does not establish a range", !single.isEstablished());
    expectNear("a single-value range passes through", single.normalize(5.0f), 5.0f, 1e-6f);

    ValueRange repeated;
    repeated.observe(5.0f);
    repeated.observe(5.0f);
    repeated.observe(5.0f);
    expectTrue("repeating one value does not establish a range", !repeated.isEstablished());
    expectNear("a repeated-value range passes through", repeated.normalize(5.0f), 5.0f, 1e-6f);

    // And one more distinct value is all it takes. Without this line an implementation
    // that never establishes anything passes every check above.
    repeated.observe(9.0f);
    expectTrue("a second distinct value establishes it", repeated.isEstablished());
    // Width 9 - 5 = 4, and 7 sits 2 above the lowest, so 2/4 = 0.5.
    expectNear("and normalises against it", repeated.normalize(7.0f), 0.5f, 1e-6f);
}

// Order of observation must not matter: the range is a set, not a sequence.
void observationOrderDoesNotMatter()
{
    ValueRange ascending;
    ascending.observe(-2.0f);
    ascending.observe(6.0f);

    ValueRange descending;
    descending.observe(6.0f);
    descending.observe(-2.0f);

    // Width 8, and 2 sits 4 above the lowest, so 4/8 = 0.5 either way.
    expectNear("ascending", ascending.normalize(2.0f), 0.5f, 1e-6f);
    expectNear("descending", descending.normalize(2.0f), 0.5f, 1e-6f);
}

// Intermediate values must widen nothing. An implementation keeping the last two seen,
// rather than the extremes, passes the tests above and fails this one.
void interiorValuesDoNotMoveTheEnds()
{
    ValueRange range;
    range.observe(0.0f);
    range.observe(10.0f);
    range.observe(5.0f);
    range.observe(4.0f);
    range.observe(6.0f);

    // Still 0 and 10, so 2.5 maps to 0.25.
    expectNear("interior values leave the width alone", range.normalize(2.5f), 0.25f, 1e-6f);
    expectNear("the top is still the top", range.normalize(10.0f), 1.0f, 1e-6f);
}

// Not clamped, and the contract says so: normalize is linear, so a value outside what was
// observed lands outside [0, 1]. A clamping implementation would report 1.0 here.
void valuesOutsideTheRangeAreNotClamped()
{
    ValueRange range;
    range.observe(0.0f);
    range.observe(10.0f);

    // 20 sits twice the width above the lowest.
    expectNear("above the maximum", range.normalize(20.0f), 2.0f, 1e-6f);
    // -5 sits half a width below it.
    expectNear("below the minimum", range.normalize(-5.0f), -0.5f, 1e-6f);
}

// The case this exists for: raw returns on the VALUE_SCALE of 40, where the spread
// between a winning and a losing line is tens of points and the largest exploration term
// any prior can produce is single digits.
void aRealisticSearchSpread()
{
    ValueRange range;
    range.observe(-10.0f);
    range.observe(30.0f);

    // Width 40. A line at +30 and one at +26 differ by 4 raw points, which is 0.1 after
    // normalising - now comparable with an exploration term rather than forty times it.
    expectNear("best line", range.normalize(30.0f), 1.0f, 1e-6f);
    expectNear("a line four points behind", range.normalize(26.0f), 0.9f, 1e-6f);
}

}  // namespace

int main()
{
    headerExample();
    freshRangeIsATransparentPassThrough();
    aZeroWidthRangeIsNotEstablished();
    observationOrderDoesNotMatter();
    interiorValuesDoNotMoveTheEnds();
    valuesOutsideTheRangeAreNotClamped();
    aRealisticSearchSpread();

    if (failures > 0)
    {
        std::cout << std::format("\n{} failing checks\n", failures);
        return 1;
    }
    std::cout << "all value range checks pass\n";
    return 0;
}
