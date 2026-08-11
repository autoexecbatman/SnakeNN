"""What tanh(return / WIN_REWARD) does to the value targets it is given.

The value head is bounded, so returns are squashed before the loss sees them
(az_trainer.cpp). Squashing is monotone, so it cannot reorder two positions -
but it compresses, and compression costs resolution exactly where the derivative
is small. This measures where the targets land and how much resolution is lost
there.

Inputs are evaluation logs, not recalled numbers: every game prints its outcome,
its steps, and the step count between consecutive apples, which is the whole
reward sequence. Greedy evaluation games, so this describes the agent being
scored rather than the noisier self-play distribution it trains on.
"""

import re
import sys

import numpy as np

DISCOUNT = 0.98
FOOD_REWARD = 1.0
STEP_REWARD = -0.02
DEATH_REWARD = -10.0
WIN_REWARD = 10.0


def read_games(path):
    """Yields (outcome, steps, [interval, ...]) per game."""
    outcome_of = {}
    steps_of = {}
    for line in open(path, encoding="utf-8", errors="replace"):
        header = re.match(r"\s*game seed (\d+), (won|died|timeout), score \d+, steps (\d+)", line)
        if header:
            outcome_of[int(header.group(1))] = header.group(2)
            steps_of[int(header.group(1))] = int(header.group(3))
            continue
        pace = re.match(r"\s*pace (\d+) (.*)$", line)
        if pace:
            seed = int(pace.group(1))
            intervals = [int(value) for value in pace.group(2).split()]
            yield outcome_of[seed], steps_of[seed], intervals


def rewards_of(outcome, steps, intervals):
    """The reward paid at each step of one game, in order."""
    rewards = np.full(steps, STEP_REWARD, dtype=np.float64)
    apple_step = 0
    for interval in intervals:
        apple_step += interval
        if apple_step <= steps:
            rewards[apple_step - 1] += FOOD_REWARD
    if outcome == "won":
        rewards[-1] += WIN_REWARD
    else:
        # A timeout is charged the death reward: both are "the game was not won".
        rewards[-1] += DEATH_REWARD
    return rewards


def returns_of(rewards):
    """Discounted return from each position onward, computed backwards."""
    out = np.empty_like(rewards)
    carried = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        carried = rewards[index] + DISCOUNT * carried
        out[index] = carried
    return out


def main(path):
    every_return = []
    for outcome, steps, intervals in read_games(path):
        if steps < 1:
            continue
        every_return.append(returns_of(rewards_of(outcome, steps, intervals)))
    raw = np.concatenate(every_return)
    scaled = raw / WIN_REWARD
    squashed = np.tanh(scaled)
    # d/dx tanh(x) = 1 - tanh(x)^2. One unit of return moves the target this
    # much less than it would if the head were unbounded and trained on x.
    resolution = 1.0 - squashed ** 2

    print("%s: %d positions from %d games" % (path, len(raw), len(every_return)))
    print("raw return      min %8.3f  max %8.3f  mean %7.3f" % (raw.min(), raw.max(), raw.mean()))
    print("scaled (raw/10) min %8.3f  max %8.3f" % (scaled.min(), scaled.max()))
    print("target = tanh   min %8.3f  max %8.3f" % (squashed.min(), squashed.max()))
    print("target range used: %.1f%% of (-1, 1)" % (50.0 * (squashed.max() - squashed.min())))
    print()
    print("resolution kept, by where the position sits:")
    for name, mask in [
        ("all positions", np.ones_like(raw, dtype=bool)),
        ("return < -5 (dying)", raw < -5.0),
        ("-5 <= return < 0", (raw >= -5.0) & (raw < 0.0)),
        ("return >= 0", raw >= 0.0),
    ]:
        if mask.sum() == 0:
            continue
        print(
            "  %-22s %6.2f%% of positions, mean resolution %.3f, worst %.3f"
            % (name, 100.0 * mask.mean(), resolution[mask].mean(), resolution[mask].min())
        )
    print()
    worst = resolution.min()
    print("Compression across the used range: %.2fx" % (resolution.max() / worst))
    for percentile in [0.1, 1.0, 5.0]:
        print(
            "  %.1f%% of positions keep less than %.3f resolution"
            % (percentile, np.percentile(resolution, percentile))
        )


if __name__ == "__main__":
    main(sys.argv[1])
