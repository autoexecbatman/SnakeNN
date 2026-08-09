#pragma once
#include <string>
#include <vector>

// How long the agent takes between apples, and where in a game the time goes.
//
// The agent fills the board in 63 of 64 games and loses at the paper's 1,200 step
// limit only because it arrives late. That is one number - a 12.8 percent reduction
// reaches 94.4 percent - but it says nothing about which apples are expensive.
// Wandering on an open board and unpicking a coiled body are different faults with
// different remedies, and nothing here measured which one we have.
//
// Free of LibTorch, so its assertions are reachable in a debug build.
namespace pace
{

// The step counts between successive apples in one game.
//
// Fed the game's (score, steps) after every move. It is an observer: it holds no
// reference to the game and cannot change it.
class AppleIntervals
{
public:
    // Records the game's state after one move.
    //
    // Asserts that `score` never falls and never rises by more than one, and that
    // `steps` never falls - all three are impossible when the caller feeds every
    // move of one game in order, and each of them silently corrupts the intervals.
    void observe(int score, int steps);

    // Steps taken to reach each apple, in the order they were eaten. The first
    // entry counts from the start of the game.
    const std::vector<int>& intervals() const noexcept;

    // Steps taken since the last apple, which no interval has closed. This is the
    // tail of a game that died or timed out hungry, and it is what makes the
    // intervals plus the tail add up to the game's step count.
    int stepsSinceLastApple() const noexcept;

private:
    std::vector<int> intervals_;
    int last_score_{ 0 };
    int last_steps_{ 0 };
    int steps_at_last_apple_{ 0 };
};

// One line per game: the seed, then every interval in order, space separated.
//
// Tagged "pace" so a parser finds these without matching the per-game outcome
// lines, and keyed by seed so a game's pace and its outcome can be joined. Raw
// rather than bucketed - which apples are expensive is the open question, and
// deciding the buckets here would answer it by assumption.
std::string formatPaceLine(unsigned int seed, const std::vector<int>& intervals);

}  // namespace pace
