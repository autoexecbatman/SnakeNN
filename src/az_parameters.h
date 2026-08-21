#pragma once

// The paper's hyperparameters and the quantities derived from them, shared by every
// program in the AlphaZero stack.
//
// Constants, not settings: nothing here is read from a file or a flag, so changing one
// means a rebuild and a new run. What a run varies - board size, simulation count, game
// count - is a command-line argument in trainer_options or eval_options instead.
//
// Usage:
//
//     #include "az_parameters.h"
//
//     search_config.discount = az::DISCOUNT;
//     search_config.exploration = az::EXPLORATION;
//     play_config.temperature = az::VISIT_TEMPERATURE;
//
//     const int step_limit = az::deriveStepLimit(10);   // 1200 for a 10x10 board
//
// Every caller sets every field from here, even where a default already matches: a
// constant only one program reads is how self-play and evaluation come to search
// differently.
//
// Source: Du, Gemp, Wu and Wu 2022 (arXiv:2211.09622).
namespace az
{

// How much a reward one step further away is worth. At 0.98 a return stops moving
// after roughly 200 steps, which is the horizon every value estimate here has.
constexpr float DISCOUNT = 0.98f;

// c_puct: how hard the prior pulls selection toward an action the search has not
// tried. Multiplies the prior, so it cannot rescue an action the network scores near
// zero - see exploration_floor.h for the term that can.
constexpr float EXPLORATION = 0.5f;

// How hot self-play samples its opening moves from the root visit counts. Below 1
// sharpens toward the argmax; PlayConfig::temperature_moves says for how many moves.
constexpr float VISIT_TEMPERATURE = 0.5f;

// Adam step size for the trainer.
constexpr float LEARNING_RATE = 0.001f;

// The share of the root prior replaced by Dirichlet noise during self-play, so a game
// explores openings the policy has stopped considering. Evaluation sets it to zero.
constexpr float ROOT_NOISE_FRACTION = 0.25f;

// Concentration of that noise. Below 1 concentrates it on a few actions per game
// rather than spreading it evenly over all of them.
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

// Weight on the death head's cross entropy, against the policy and value losses.
//
// Matched to the steps head rather than chosen: both are auxiliary signals on a
// shared trunk, neither picks a move on its own, and a trunk pulled hard toward
// either is a trunk not learning to play. Equal to STEPS_LOSS_WEIGHT is the
// assumption, not a measurement - the first training run that uses it should
// report the two losses side by side before this number is defended.
constexpr float DEATH_LOSS_WEIGHT = 0.25f;

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
//
// This is the default, not the setting. `AlphaZeroEvaluate` takes --trap-guard on|off
// and records which it used in its header, so measuring the two states costs no edit
// and no rebuild. The trainer reads this constant directly.
constexpr bool TRAP_GUARD = false;

// Whether to count the seals anyway. On: the veto is what costs win rate, and
// counting costs one flood fill per played move.
//
// Without this, switching the veto off would take the measurement with it, and
// the measurement is the point - a network that has learned to keep its tail
// reachable shows up as this number falling, and nothing else in the run reports
// it. A guard that corrects the move erases its own evidence.
constexpr bool TRAP_REPORT = true;

// Whether selection reads an edge's mean reward over the traversals that reached it
// rather than whatever the last simulation wrote there.
//
// A tree node is keyed by the actions reaching it, not by the state, so it stands
// for a distribution of states and the quantity selection needs is an expectation
// over that distribution. Visit counts already are one. Leurent and Maillard 2019
// give the estimator: the rewards at a sequence's last transition, summed over
// traversals and divided by the traversal count.
//
// Measured before it was written: 4.99 percent of revisited edge traversals carried a
// reward more than half an apple from the recorded one. That is the error this
// removes, and it is not a claim about wins - open-loop keying costs optimality
// whatever the estimator, so a correct open-loop search is still a suboptimal one.
//
// Default, not the setting. `AlphaZeroEvaluate` takes --average-edges on|off.
constexpr bool AVERAGE_EDGES = false;

// Whether the root refuses an action whose backed-up death risk exceeds
// DEATH_CAP_THRESHOLD.
//
// Deaths are 62 of the 67 remaining losses, at median 55 percent fill with about 690
// steps of budget unused and no slow apple before them - a region sealing behind an
// apple, which is a dead-end in the sense of Fatemi et al. 2019: every trajectory from
// it dies. Their construction is undiscounted, which is what it buys over the value
// head, whose 0.98 gives it a 50-step horizon.
//
// The cap refuses only when some other action is below the threshold. Refusing when
// everything is above it is what turned the trap guard into the endgame policy.
//
// Default, not the setting. `AlphaZeroEvaluate` takes --death-cap on|off.
constexpr bool DEATH_CAP = false;

// The backed-up risk above which the cap refuses an action, in [0, 1]. Read only when
// DEATH_CAP is on.
constexpr float DEATH_CAP_THRESHOLD = 0.5f;

// Whether the leaf's death risk comes from the network's head or is left at zero.
//
// Off until the head has a loss behind it: an untrained head emits its initialisation,
// and a cap acting on that is acting on noise. With it off the risk is entirely the
// simulator's own terminations, which is what the first measurement is of.
constexpr bool DEATH_RISK_FROM_NETWORK = false;

// What the value head's output is multiplied by, in the same units as a reward.
//
// The head ends in a tanh, so it emits a number in [-1, 1]; this scales that to the range
// it may report. At 40 it can say anything from -40 to +40, and those are reward units -
// the same +1 an apple pays and the -10 a death costs.
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
// land at 1.0 and keep 42 percent.
//
// It is not an upper bound on what a return can be, and that is a deliberate
// accepted deviation rather than an oversight. Z3 puts the supremum at 43.75
// (docs/prove_value_bound.py), reached only by eating on every step for 99
// consecutive steps and then winning - which needs every apple to spawn against
// the head for a whole game. The largest return in 1000 real games was 16.433,
// so the unreachable 3.75 of headroom is not worth a retrain to recover. Raise
// this above 44 if the reward scale changes, since that argument is about this
// reward structure and not about the constant.
//
// What the units are worth, measured 2026-08-22. Before this constant existed the value
// head emitted return/10 while backup added raw rewards, so the search discounted its own
// leaves tenfold and could not price a distant win against a nearby apple. One checkpoint,
// az10_long308, scored 863 of 1000 held-out games under the old units and 954 under these
// - the same weights, the same seeds, the same search stream. About +90 wins, from a
// defect rather than a hyperparameter, and the largest single change this project has
// made. Do not rescale the head without re-deriving what the search adds it to.
constexpr float VALUE_SCALE = 40.0f;

// Whether selection normalises an action's value against the range this tree has seen
// before comparing it with the prior term, as MuZero does.
//
// PUCT adds a value to c_puct * prior * sqrt(N), and the paper's c_puct of 0.5 assumes
// values in [-1, 1]. Values here are raw returns bounded by VALUE_SCALE, so the value
// side can be forty times anything the prior side produces and the constant stops
// meaning what it means in the paper. Measured 2026-08-20: the fraction of positions
// whose second-best prior is under 0.001 went from 2 percent to 46 percent across the
// change that made values raw, while mean policy entropy barely moved.
//
// Off, so a checkpoint trained before it plays as it did. It is a fix for the scale
// selection compares on, not a rescue for a policy already trained to one action - a
// prior of 0.0005 yields almost no exploration term at any scale.
constexpr bool NORMALIZE_VALUES = false;

// How often selection picks an action uniformly instead of by its scores, before the
// 1/log(N) decay applied in exploration_floor.h. Zero is off.
//
// Every term of the PUCT exploration half multiplies the prior, so a prior near zero
// leaves the search no way back: measured on az10_death368, sweeping c_puct over a
// hundredfold moved root actions visited from 1.13 of 3 to 1.24. KataGo's forced playouts
// have the same shape and award under half a playout at these priors. Xiao et al. 2019 mix
// in a uniform policy instead, which is additive and does not ask the network's permission
// - at 0.1 here every action is guaranteed about 3.8 visits of 200.
//
// Off, so a checkpoint trained before it plays as it did. Turning it on spends simulations
// on actions the network dislikes in exchange for covering the root; judge it on coverage
// first and on win rate second.
constexpr float EXPLORATION_EPSILON = 0.0f;

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

// The step budget for a square board: STEPS_PER_CELL * board * board. Every program
// here derives the limit this way rather than naming a number, so two runs on one
// board size are comparable.
//
//     const int step_limit = az::deriveStepLimit(10);   // 1200, on a 10x10 board
//     SnakeEnv game(10, 10, seed, step_limit);
//
// `board` must be at least 2, which every argument parser here enforces, and is
// asserted. Above LARGEST_BOARD the result does not fit in an int, which is a
// boundary rather than a wiring fault, so it throws std::invalid_argument.
int deriveStepLimit(int board);

}  // namespace az
