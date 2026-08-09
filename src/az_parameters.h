#pragma once

// The paper's hyperparameters and the quantities derived from them, for every
// program in the AlphaZero stack - trainer, evaluator and visual alike.
//
// Du, Gemp, Wu and Wu 2022 (arXiv:2211.09622). Deviating from these knowingly is
// allowed; deviating by accident is what a single definition prevents.
namespace az
{

constexpr float DISCOUNT = 0.98f;
constexpr float EXPLORATION = 0.5f;  // c_puct
constexpr float VISIT_TEMPERATURE = 0.5f;
constexpr float LEARNING_RATE = 0.001f;
constexpr float ROOT_NOISE_FRACTION = 0.25f;
constexpr float ROOT_NOISE_ALPHA = 0.3f;

// The paper caps a 10x10 game at 1,200 steps, which is twelve steps per cell.
// Scaling by area rather than fixing the number keeps "win" meaning the same
// thing at every board size the curriculum passes through.
constexpr int STEPS_PER_CELL = 12;

// The largest board whose step limit fits in an int, and therefore the largest
// any parser here accepts. Its own arithmetic is checked below and independently
// in docs/prove_arithmetic.py, which also proves no smaller bound is needed.
constexpr int LARGEST_BOARD = 13377;

static_assert(static_cast<long long>(STEPS_PER_CELL) * LARGEST_BOARD * LARGEST_BOARD <= 2147483647,
              "LARGEST_BOARD does not fit its own step limit");
static_assert(static_cast<long long>(STEPS_PER_CELL) * (LARGEST_BOARD + 1) * (LARGEST_BOARD + 1) >
                  2147483647,
              "LARGEST_BOARD is not the largest that fits");

// STEPS_PER_CELL * board * board.
//
// `board` must be at least 2, which every argument parser here enforces, and is
// asserted. Above LARGEST_BOARD the result does not fit in an int, which is a
// boundary rather than a wiring fault, so it throws std::invalid_argument.
int deriveStepLimit(int board);

}  // namespace az
