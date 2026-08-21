#include <cmath>
#include <format>
#include <iostream>
#include <stdexcept>
#include <string>

#include "exploration_floor.h"

namespace
{

int failures = 0;

// Every expected value is derived from the contract in exploration_floor.h by hand, with
// the arithmetic beside it. None was read off an implementation.
void expectNear(const std::string& what, float actual, float expected, float tolerance)
{
    if (std::isnan(actual) || std::abs(actual - expected) > tolerance)
    {
        std::cout << std::format("[FAIL] {}: expected {:.6f}, got {:.6f}\n", what, expected,
                                 actual);
        failures++;
    }
}

void expectRefused(const std::string& what, float epsilon, int action_count, int total_visits)
{
    try
    {
        (void)explorationMixWeight(epsilon, action_count, total_visits);
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

// The worked example from the header, and the case this exists for. eps |A| / log(N + 1)
// = 0.1 * 3 / log(201) = 0.3 / 5.303305 = 0.056568.
void theWorkedExample()
{
    expectNear("eps 0.1, 3 actions, 200 visits", explorationMixWeight(0.1f, 3, 200), 0.056568f,
               1e-5f);
}

// Zero epsilon is off, and it must be exactly off rather than nearly: a floor that leaks
// changes every checkpoint's play the moment it is compiled in.
void zeroEpsilonIsExactlyOff()
{
    expectNear("eps 0 at 200 visits", explorationMixWeight(0.0f, 3, 200), 0.0f, 0.0f);
    expectNear("eps 0 at 1 visit", explorationMixWeight(0.0f, 3, 1), 0.0f, 0.0f);
    // And zero visits, where the unvisited rule would otherwise return 1.
    expectNear("eps 0 at 0 visits", explorationMixWeight(0.0f, 3, 0), 0.0f, 0.0f);
}

// Nothing visited means nothing to trust, so the weight is 1 rather than a division by
// log(1) = 0. Stated against the 200-visit case above, which is 0.057.
void nothingVisitedExploresOutright()
{
    expectNear("0 visits", explorationMixWeight(0.1f, 3, 0), 1.0f, 0.0f);
}

// The formula exceeds 1 at small visit counts and is clamped. At 1 visit:
// 0.1 * 3 / log(2) = 0.3 / 0.693147 = 0.432808, which is under 1 and must not be clamped.
// At eps 1.0 and 1 visit: 3 / 0.693147 = 4.328, which must come back as exactly 1.
void theWeightIsClampedButNotOtherwiseTouched()
{
    expectNear("eps 0.1 at 1 visit is not clamped", explorationMixWeight(0.1f, 3, 1), 0.432808f,
               1e-5f);
    expectNear("eps 1.0 at 1 visit is clamped to one", explorationMixWeight(1.0f, 3, 1), 1.0f,
               0.0f);
}

// Decay is 1/log(N), which is the slow part of the design. Over a hundredfold increase in
// visits the weight must fall by less than a factor of three - a 1/N or 1/sqrt(N) decay
// would fall by a hundred or by ten and the floor would be gone when it is needed.
void decayIsLogarithmic()
{
    const float at_ten = explorationMixWeight(0.1f, 3, 10);
    const float at_thousand = explorationMixWeight(0.1f, 3, 1000);
    // 0.3/log(11) = 0.125110 and 0.3/log(1001) = 0.043423, a ratio of 2.8812.
    expectNear("10 visits", at_ten, 0.125110f, 1e-5f);
    expectNear("1000 visits", at_thousand, 0.043423f, 1e-5f);
    if (!(at_ten / at_thousand < 3.0f))
    {
        std::cout << std::format("[FAIL] decay is faster than logarithmic: ratio {:.4f}\n",
                                 at_ten / at_thousand);
        failures++;
    }
}

// The weight scales with the action count, so the per-action floor eps/log(N+1) does not
// shrink as actions are added. Stated as the property rather than as a second number: at
// six actions the weight is twice what it is at three.
void theWeightScalesWithTheActionCount()
{
    const float three = explorationMixWeight(0.1f, 3, 200);
    const float six = explorationMixWeight(0.1f, 6, 200);
    // Anchored, because a ratio alone holds for any constant including zero.
    expectNear("three actions at 200 visits", three, 0.056568f, 1e-5f);
    expectNear("six actions is twice three", six, 2.0f * three, 1e-6f);
}

// Linear in epsilon, which is what makes epsilon the one dial.
void theWeightIsLinearInEpsilon()
{
    const float tenth = explorationMixWeight(0.1f, 3, 200);
    const float twentieth = explorationMixWeight(0.05f, 3, 200);
    // Anchored for the same reason as the scaling test above.
    expectNear("half epsilon at 200 visits", twentieth, 0.028284f, 1e-5f);
    expectNear("halving epsilon halves the weight", twentieth, 0.5f * tenth, 1e-6f);
}

void refusals()
{
    expectRefused("negative epsilon", -0.1f, 3, 200);
    expectRefused("zero actions", 0.1f, 0, 200);
    expectRefused("negative actions", 0.1f, -3, 200);
    expectRefused("negative visits", 0.1f, 3, -1);
}

}  // namespace

void expectDescentRefused(const std::string& what, float epsilon, int action_count, int node_visits,
                          int depth)
{
    try
    {
        (void)descentMixWeight(epsilon, action_count, node_visits, depth);
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

// The root is the only place the floor fires. Every check below states the claim against
// the alternative it has to beat: a rule that fired everywhere, and a rule that fired
// nowhere, both satisfy "returns something in [0, 1]".
void theFloorFiresAtTheRootAndNowhereBelow()
{
    // 0.1 * 3 / log(201) = 0.3 / 5.303305 = 0.056568, the same number the mixture gives.
    expectNear("the root takes the full mix weight", descentMixWeight(0.1f, 3, 200, 0), 0.056568f,
               1e-5f);

    // The sharp case. One visit would give 0.3 / log(2) = 0.432809, and 200 simulations
    // leave most of the tree at one visit - that is the weight that made a random walk.
    expectNear("one level down takes nothing", descentMixWeight(0.1f, 3, 1, 1), 0.0f, 0.0f);

    // Sharper still: no visits at all is where the mixture returns 1.0 outright, so a
    // rule that forgot the depth would explore every deep step of every descent.
    expectNear("an unvisited node below the root takes nothing", descentMixWeight(0.1f, 3, 0, 1),
               0.0f, 0.0f);
    expectNear("depth stays off however deep", descentMixWeight(1.0f, 3, 0, 7), 0.0f, 0.0f);

    // And the root is not a special case of "unvisited": at depth 0 with no visits the
    // mixture's own answer stands.
    expectNear("an unvisited root explores outright", descentMixWeight(0.1f, 3, 0, 0), 1.0f, 0.0f);

    // Off is off at the root too, so a checkpoint trained without the floor plays as it
    // did rather than differing only below the root.
    expectNear("zero epsilon is off at the root", descentMixWeight(0.0f, 3, 0, 0), 0.0f, 0.0f);
}

void theDescentRefusesWhatTheMixtureRefuses()
{
    expectDescentRefused("a negative depth", 0.1f, 3, 200, -1);
    // Delegated, and it must still fire at depth 0 - a guard skipped before the depth
    // test would hand a negative epsilon to the search unchecked.
    expectDescentRefused("a negative epsilon at the root", -0.1f, 3, 200, 0);
    expectDescentRefused("no actions at the root", 0.1f, 0, 200, 0);
    expectDescentRefused("a negative visit count at the root", 0.1f, 3, -1, 0);
}

int main()
{
    theWorkedExample();
    zeroEpsilonIsExactlyOff();
    nothingVisitedExploresOutright();
    theWeightIsClampedButNotOtherwiseTouched();
    decayIsLogarithmic();
    theWeightScalesWithTheActionCount();
    theWeightIsLinearInEpsilon();
    refusals();
    theFloorFiresAtTheRootAndNowhereBelow();
    theDescentRefusesWhatTheMixtureRefuses();

    if (failures > 0)
    {
        std::cout << std::format("\n{} failing checks\n", failures);
        return 1;
    }
    std::cout << "all exploration floor checks pass\n";
    return 0;
}
