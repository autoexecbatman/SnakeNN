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

// What a game pays for running out of steps. A deliberate deviation: the paper's
// outcomes are a win and a death, and a truncated game is neither, so under it a
// timeout was worth 0 - strictly better than the -10 of dying. That ranking is
// backwards for the task. It makes stalling the safe play, and it is the reward
// side of the reason the agent arrives at the cap with the board nearly full.
//
// Equal to the death reward, because both are the same outcome: the game was not
// won. Equalising them is what stops a policy preferring a certain timeout to a
// risky finish.
//
// It reaches the value target only, through the discounted return in selfplay.cpp.
// At DISCOUNT it is visible about 200 steps back and no further, so this makes the
// deadline real near the end of a game and does nothing about pace at the start.
constexpr float TIMEOUT_REWARD = -10.0f;

// What a step costs, paid on every tick by both the search and the training
// target. Another deliberate deviation: the paper prices only outcomes, so a
// seven-step route to an apple and a forty-step route score identically.
//
// This is the term that reaches the whole game. TIMEOUT_REWARD sits at the end
// and at DISCOUNT is invisible more than about 200 steps back, but a per-step
// cost is paid where the waste happens, and the waste is spread throughout - the
// slowest tenth of apples take 39.3 percent of every step played.
//
// The size is set against the two rewards it has to sit between. The value floor
// it introduces is STEP_REWARD / (1 - DISCOUNT) = -1.0, about one apple, so a
// long game is worth roughly one apple less and death at -10 still dominates.
// Larger and the agent buys speed with its life; the clock plane already showed
// that trade going the wrong way.
constexpr float STEP_REWARD = -0.02f;

// How far below the best visit count a root action may sit and still be played
// when the steps head expects it to finish sooner. Only moves choices the search
// was near-indifferent about.
constexpr float STEPS_TIEBREAK_MARGIN = 0.05f;

// Weight on the steps head's regression against the policy and value losses.
// Below them because it steers nothing on its own - it informs a tie-break - and
// a trunk pulled hard toward predicting duration is a trunk not learning to play.
constexpr float STEPS_LOSS_WEIGHT = 0.25f;

// Whether the search refuses a root move that seals the head away from its own
// tail. Algorithmic assistance rather than a learned skill: the seal proves fatal
// tens of moves later, past any horizon 200 simulations reach.
//
// The test is reachability of the tail, not the size of the region. A cell count
// vetoes every endgame move there is, and measured 0 of 64 games won against 47
// with the guard switched off.
//
// Off, on the measurement. On the same weights and the same 64 held-out seeds:
// 56 wins with no guard, 44 with this one, 0 with the cell count. The corrected
// veto costs twelve games and turns them into timeouts, because following the
// tail is the slow way out of a position the search can already play. Search at
// 200 simulations is better at this than the heuristic, which is the outcome
// AlphaZero predicts and the reason the veto is not the assistance to keep.
//
// It is a veto, not a fallback - the search still picks among what survives it -
// and how often it fires is printed, so a network that has learned the pattern
// shows up as the count falling rather than as nothing at all.
constexpr bool TRAP_GUARD = false;

// Whether to count the seals anyway. On: the veto is what costs win rate, and
// counting costs one flood fill per played move.
//
// Without this, switching the veto off would take the measurement with it, and
// the measurement is the point - a network that has learned to keep its tail
// reachable shows up as this number falling, and nothing else in the run reports
// it. A guard that corrects the move erases its own evidence.
constexpr bool TRAP_REPORT = true;

// The bound on the value head, in the same units as a reward.
//
// The head is bounded because the search compares leaves and one bad
// extrapolation would otherwise dominate every comparison it appears in. What it
// must not do is change the units: `MonteCarloSearch::backup` adds `node.reward`
// - a raw +1 apple or -10 death - to the leaf value, so a head that reports
// return/10 makes the search undervalue everything past its own edges tenfold.
//
// 40 rather than the 16.433 measured as the largest return over 1000 evaluation
// games (docs/measure_value_squashing.py), because tanh has to stay in the part
// of its range that resolves: at 40 the extremes land at 0.41, which keeps 83
// percent of the resolution an unbounded head would have. At 16.5 they would
// land at 1.0 and keep 42 percent. Re-run that script if the reward scale or the
// board changes; a scale below the largest return silently flattens the endgame.
constexpr float VALUE_SCALE = 40.0f;

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
