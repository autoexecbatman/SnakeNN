#include <format>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "steps_per_apple.h"

// Expected values come from the contract in steps_per_apple.h.

namespace
{

// Checks that did not hold. main prints the count and returns 1 when it is non-zero.
int failures = 0;

// Reports one failure and counts it.
void fail(std::string_view what, std::string_view detail)
{
    std::cout << std::format("[FAIL] {}: {}\n", what, detail);
    failures++;
}

// Compares two ints, reporting both when they differ.
void expectEquals(std::string_view what, int expected, int actual)
{
    if (expected != actual)
    {
        fail(what, std::format("expected {}, got {}", expected, actual));
    }
}

// A vector of intervals as a readable list, so a failure prints [3, 12, 7] rather than a
// length and a promise.
std::string describe(const std::vector<int>& values)
{
    std::string text = "[";
    for (const int value : values)
    {
        text += text.size() == 1 ? std::format("{}", value) : std::format(", {}", value);
    }
    return text + "]";
}

// Compares two interval lists, printing both in full when they differ. Which interval is
// wrong matters as much as that one is.
void expectIntervals(std::string_view what, const std::vector<int>& expected,
                     const std::vector<int>& actual)
{
    if (expected != actual)
    {
        fail(what, std::format("expected {}, got {}", describe(expected), describe(actual)));
    }
}

// Plays one game as a sequence of scores, one per move, and returns the observer.
// `scores[index]` is the score after move index+1.
pace::AppleIntervals play(const std::vector<int>& scores)
{
    pace::AppleIntervals intervals;
    int step = 0;
    for (const int score : scores)
    {
        step++;
        intervals.observe(score, step);
    }
    return intervals;
}

// A new observer reports no intervals. The pace line is parsed by position, so a phantom
// leading interval would shift every real one.
void aFreshObserverHasNothing()
{
    const pace::AppleIntervals intervals;
    expectIntervals("fresh intervals", {}, intervals.intervals());
    expectEquals("fresh tail", 0, intervals.stepsSinceLastApple());
}

// Expected values are counted from the rule - an interval is the number of moves
// between one apple and the next - not read back from the implementation.
void anIntervalIsTheMovesBetweenApples()
{
    // Moves:            1  2  3  4  5  6  7
    // Score after:      0  0  1  1  1  2  2
    // The first apple lands on move 3, so its interval is 3. The second lands on
    // move 6, three moves later.
    const pace::AppleIntervals intervals = play({ 0, 0, 1, 1, 1, 2, 2 });
    expectIntervals("two apples", { 3, 3 }, intervals.intervals());
    expectEquals("tail after the last apple", 1, intervals.stepsSinceLastApple());
}

// An apple eaten on the first move costs one step, not zero. The interval counts steps
// taken to reach it, and an off-by-one here biases every pace statistic downward.
void anAppleOnTheFirstMoveIsAnIntervalOfOne()
{
    const pace::AppleIntervals intervals = play({ 1 });
    expectIntervals("apple on move one", { 1 }, intervals.intervals());
    expectEquals("no tail", 0, intervals.stepsSinceLastApple());
}

// A game that eats nothing has no intervals - its steps are the tail after the last
// apple, which the pace line does not carry.
void aGameWithNoApplesIsAllTail()
{
    const pace::AppleIntervals intervals = play({ 0, 0, 0, 0 });
    expectIntervals("no apples", {}, intervals.intervals());
    expectEquals("all tail", 4, intervals.stepsSinceLastApple());
}

// The property the whole instrument rests on: nothing is lost and nothing is
// double counted, so the parts reconstruct the game exactly.
void theIntervalsAndTheTailAccountForEveryStep()
{
    const std::vector<std::vector<int>> games{
        { 0, 0, 1, 1, 1, 2, 2 },
        { 1 },
        { 0, 0, 0, 0 },
        { 0, 1, 2, 3, 3, 3, 4, 4, 5 },
        { 0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 5, 5 },
    };
    for (const std::vector<int>& scores : games)
    {
        const pace::AppleIntervals intervals = play(scores);
        const int summed =
            std::accumulate(intervals.intervals().begin(), intervals.intervals().end(), 0) +
            intervals.stepsSinceLastApple();
        if (summed != static_cast<int>(scores.size()))
        {
            fail("step accounting",
                 std::format("{} intervals plus tail {} make {}, the game took {} steps",
                             describe(intervals.intervals()), intervals.stepsSinceLastApple(),
                             summed, scores.size()));
        }
        // And one interval per apple, no more and no fewer.
        expectEquals("interval count", scores.back(),
                     static_cast<int>(intervals.intervals().size()));
    }
}

// A whole game of the length this project actually plays, so the accounting is
// checked at the scale it will be used at rather than only on toy inputs.
void aFullBoardIsAccountedFor()
{
    // 99 apples, each taking a different number of moves, on a 10x10 board.
    std::vector<int> scores;
    int score = 0;
    for (int apple = 1; apple <= 99; apple++)
    {
        const int gap = 5 + apple % 17;
        for (int move = 0; move < gap; move++)
        {
            scores.push_back(score);
        }
        score++;
        scores.back() = score;
    }
    const pace::AppleIntervals intervals = play(scores);
    expectEquals("apples on a full board", 99, static_cast<int>(intervals.intervals().size()));
    const int summed =
        std::accumulate(intervals.intervals().begin(), intervals.intervals().end(), 0) +
        intervals.stepsSinceLastApple();
    expectEquals("every step accounted for", static_cast<int>(scores.size()), summed);
    // Guarded: front() on an empty vector aborts the process under the debug
    // runtime, and a test that aborts reports nothing at all.
    if (intervals.intervals().empty())
    {
        fail("the first interval", "there are no intervals to read it from");
        return;
    }
    expectEquals("the first interval", 6, intervals.intervals().front());
}

// Splits a pace line into fields, for checking the format a log analyser reads.
std::vector<std::string> splitOnSpaces(const std::string& text)
{
    std::vector<std::string> fields;
    std::string field;
    std::istringstream stream(text);
    while (stream >> field)
    {
        fields.push_back(field);
    }
    return fields;
}

// The line begins with the seed and then every interval in order. Analysis scripts index
// it by position, so a missing seed shifts every interval by one column.
void aPaceLineCarriesTheSeedAndEveryInterval()
{
    const std::string line = pace::formatPaceLine(3758096384u, { 7, 5, 12 });
    if (line.empty() || line.back() != '\n')
    {
        fail("pace line", std::format("does not end in a newline: '{}'", line));
        return;
    }
    if (line.find('\n') != line.size() - 1)
    {
        fail("pace line", "is more than one line");
    }
    const std::vector<std::string> fields = splitOnSpaces(line);
    // pace, seed, then one field per interval.
    if (fields.size() != 5)
    {
        fail("pace line", std::format("expected 5 fields, got {} from '{}'", fields.size(), line));
        return;
    }
    if (fields[0] != "pace")
    {
        fail("pace line", std::format("is not tagged 'pace': '{}'", line));
    }
    if (fields[1] != "3758096384")
    {
        fail("pace line", std::format("seed is '{}', expected 3758096384", fields[1]));
    }
    if (fields[2] != "7" || fields[3] != "5" || fields[4] != "12")
    {
        fail("pace line", std::format("intervals are wrong or reordered: '{}'", line));
    }
}

// A scoreless game still emits a line with its seed. Omitting it would leave the failures
// out of any pace analysis, which is exactly the population worth looking at.
void aGameWithNoApplesStillGetsALine()
{
    const std::string line = pace::formatPaceLine(42u, {});
    const std::vector<std::string> fields = splitOnSpaces(line);
    if (fields.size() != 2 || fields[0] != "pace" || fields[1] != "42")
    {
        fail("empty pace line", std::format("expected 'pace 42', got '{}'", line));
    }
}

}  // namespace

int main()
{
    aFreshObserverHasNothing();
    anIntervalIsTheMovesBetweenApples();
    anAppleOnTheFirstMoveIsAnIntervalOfOne();
    aGameWithNoApplesIsAllTail();
    theIntervalsAndTheTailAccountForEveryStep();
    aFullBoardIsAccountedFor();
    aPaceLineCarriesTheSeedAndEveryInterval();
    aGameWithNoApplesStillGetsALine();

    if (failures == 0)
    {
        std::cout << "[PASS] steps_per_apple\n";
        return 0;
    }
    std::cout << std::format("[FAIL] steps_per_apple: {} failures\n", failures);
    return 1;
}
