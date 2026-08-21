#include <format>
#include <iostream>
#include <stdexcept>
#include <string>

#include "az_parameters.h"

// Tests for the shared hyperparameters and the step limit derived from a board size.
//
// Its main job is to stop a parameter drifting away from the paper it cites unnoticed. The
// constants are pinned as literals below, so editing one fails the build here rather than
// being absorbed into a run whose numbers nobody can then compare with anything.
//
// No LibTorch, so it builds and runs without the CUDA toolchain, and its checks are
// reachable in a debug build.
//
// Run it directly; it returns non-zero and prints one line per property that failed.
//
//     build\Release\ParametersTest.exe
//
// The literals are compared against Du, Gemp, Wu and Wu 2022 (arXiv:2211.09622), the
// paper this stack reproduces.
static_assert(az::DISCOUNT == 0.98f, "the paper's discount is 0.98");
static_assert(az::EXPLORATION == 0.5f, "the paper's c_puct is 0.5");
static_assert(az::VISIT_TEMPERATURE == 0.5f, "the paper's visit temperature is 0.5");
static_assert(az::LEARNING_RATE == 0.001f, "the paper's learning rate is 0.001");
static_assert(az::ROOT_NOISE_FRACTION == 0.25f, "the paper's root noise fraction is 0.25");
static_assert(az::ROOT_NOISE_ALPHA == 0.3f, "the paper's Dirichlet alpha is 0.3");
static_assert(az::STEPS_PER_CELL == 12, "1200 steps on a 10x10 board is 12 per cell");

namespace
{

// Properties that did not hold. main returns non-zero when it is not zero.
int failures = 0;

// Records that deriveStepLimit(board) returns `expected`, counting a rejection as a
// failure too - a board this size is meant to be accepted.
//
//     expectLimit(10, 1200);   // the paper's own cap, on its own board
//
// `expected` is written out from the rule, 12 * board * board, never read back from the
// function - a test fed the current answer confirms the bug forever.
void expectLimit(int board, int expected)
{
    try
    {
        const int actual = az::deriveStepLimit(board);
        if (actual != expected)
        {
            std::cout << std::format("[FAIL] board {}: expected {}, got {}\n", board, expected,
                                     actual);
            failures++;
        }
    }
    catch (const std::exception& error)
    {
        std::cout << std::format("[FAIL] board {}: rejected: {}\n", board, error.what());
        failures++;
    }
}

// Records that deriveStepLimit(board) refuses, with std::invalid_argument, and that the
// message names the board it was given.
//
//     expectTooLarge(13378);   // its limit does not fit in an int
//
// The exception type and the message are both checked: a different type would mean the
// overflow was caught somewhere unintended, and a message without the number leaves the
// operator guessing which argument was refused.
void expectTooLarge(int board)
{
    try
    {
        const int actual = az::deriveStepLimit(board);
        std::cout << std::format("[FAIL] board {}: accepted, returned {}\n", board, actual);
        failures++;
    }
    catch (const std::invalid_argument& error)
    {
        const std::string message = error.what();
        if (message.find(std::to_string(board)) == std::string::npos)
        {
            std::cout << std::format("[FAIL] board {}: message omits the board: {}\n", board,
                                     message);
            failures++;
        }
    }
    catch (const std::exception& error)
    {
        std::cout << std::format("[FAIL] board {}: wrong exception type: {}\n", board,
                                 error.what());
        failures++;
    }
}

// Expected values are 12 * board * board written out, not read back from the
// function. 1200 at board 10 is the paper's own cap.
void scalesTheLimitWithTheBoardArea()
{
    expectLimit(2, 48);
    expectLimit(6, 432);
    expectLimit(10, 1200);
    expectLimit(20, 4800);
}

// 12 * 13377^2 = 2147329548, which fits; 13378 does not.
void acceptsTheLargestBoardThatFitsInAnInt()
{
    expectLimit(13377, 2147329548);
}

// A board whose step limit would not fit in an int is refused rather than wrapped.
//
// Wrapping would produce a negative or tiny limit, which reads as a legal setting and cuts
// every game short - a wrong number rather than an error.
void rejectsABoardWhoseLimitWouldOverflow()
{
    expectTooLarge(13378);
    expectTooLarge(100000);
    expectTooLarge(2147483647);
}

}  // namespace

int main()
{
    scalesTheLimitWithTheBoardArea();
    acceptsTheLargestBoardThatFitsInAnInt();
    rejectsABoardWhoseLimitWouldOverflow();

    if (failures == 0)
    {
        std::cout << "[SUCCESS] az_parameters: all properties hold\n";
        return 0;
    }
    std::cout << std::format("[FAILURE] {} property violations\n", failures);
    return 1;
}
