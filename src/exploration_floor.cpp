// Implementation of explorationMixWeight. What the floor is for, and why it is additive
// rather than a change to the prior, are in exploration_floor.h.

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "exploration_floor.h"

float explorationMixWeight(float epsilon, int action_count, int total_visits)
{
    // A negative epsilon would subtract exploration, which is not a weaker floor but a
    // different and meaningless rule.
    if (!(epsilon >= 0.0f))
    {
        throw std::invalid_argument("exploration epsilon must be zero or positive");
    }
    if (action_count < 1)
    {
        throw std::invalid_argument("a node with no actions has nothing to explore");
    }
    if (total_visits < 0)
    {
        throw std::invalid_argument("a visit count cannot be negative");
    }

    // Off means off, before anything else is computed: a checkpoint trained without this
    // has to play exactly as it did, including where the rule below would return 1.
    if (epsilon == 0.0f)
    {
        return 0.0f;
    }

    // Nothing visited is nothing to trust, and log(1) is zero, so the division is not
    // reached rather than being guarded after the fact.
    if (total_visits == 0)
    {
        return 1.0f;
    }

    const float weight = epsilon * static_cast<float>(action_count) /
                         std::log(static_cast<float>(total_visits) + 1.0f);
    // The formula exceeds one at small visit counts, where it means "always explore".
    return std::min(weight, 1.0f);
}

float descentMixWeight(float epsilon, int action_count, int node_visits, int depth)
{
    if (depth < 0)
    {
        throw std::invalid_argument("a depth cannot be negative");
    }

    // Below the root the weight is read from a node with almost no visits, where the
    // 1/log decay has not started: one visit gives 0.43, and a descent that random-walks
    // two steps in five is not a lookahead. The label the floor exists for is read at the
    // root, so that is where the coverage has to be and the only place it is bought.
    if (depth > 0)
    {
        return 0.0f;
    }

    return explorationMixWeight(epsilon, action_count, node_visits);
}
