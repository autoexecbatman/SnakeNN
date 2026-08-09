#include <cassert>
#include <format>

#include "steps_per_apple.h"

namespace pace
{

void AppleIntervals::observe(int score, int steps)
{
    // All three are impossible when every move of one game is fed in order, and
    // each corrupts the intervals without producing anything that looks wrong: a
    // fallen score reopens a closed interval, a jump of two loses one entirely, and
    // a fallen step count makes an interval negative.
    assert(score >= last_score_ && "observe saw the score fall");
    assert(score <= last_score_ + 1 && "observe missed a move - the score rose by more than one");
    assert(steps >= last_steps_ && "observe saw the step count fall");

    if (score > last_score_)
    {
        intervals_.push_back(steps - steps_at_last_apple_);
        steps_at_last_apple_ = steps;
    }
    last_score_ = score;
    last_steps_ = steps;
}

const std::vector<int>& AppleIntervals::intervals() const noexcept
{
    return intervals_;
}

int AppleIntervals::stepsSinceLastApple() const noexcept
{
    return last_steps_ - steps_at_last_apple_;
}

std::string formatPaceLine(unsigned int seed, const std::vector<int>& intervals)
{
    std::string line = std::format("  pace {}", seed);
    for (const int interval : intervals)
    {
        line += std::format(" {}", interval);
    }
    return line + "\n";
}

}  // namespace pace
