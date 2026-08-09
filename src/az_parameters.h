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

// STEPS_PER_CELL * board * board.
//
// `board` must be at least 2, which every argument parser here enforces, and is
// asserted. Above 13377 the result does not fit in an int, which is a boundary
// rather than a wiring fault, so it throws std::invalid_argument.
int deriveStepLimit(int board);

}  // namespace az
