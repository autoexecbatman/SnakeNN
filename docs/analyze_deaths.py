"""Where in the fill the agent dies, read from an evaluation log.

Takes evaluation logs written by AlphaZeroEvaluate, keeps the games that ended in a
death, and reports three things about them: the board occupancy reached before dying,
how much of the step budget was left, and how the last apple before the death compares
with that game's own median apple. Reads only; runs nothing.

A game line is "  game seed N, outcome, score S, steps T" and is followed by
"  pace N g1 g2 ..." holding the step count of each apple in order, so the pace list
has one entry per apple eaten and the steps after the final apple are not in it.
"""

import re
import statistics
import sys
from pathlib import Path

GAME_LINE = re.compile(r"^  game seed (\d+), (\w+), score (\d+), steps (\d+)")
PACE_LINE = re.compile(r"^  pace (\d+) (.+)")

# The snake starts at one segment, so occupancy is the score plus that first segment.
STARTING_SEGMENTS = 1
BOARD_CELLS = 100
STEP_LIMIT = 1200


class Game:
    """One evaluated game: its outcome, what it scored, and its per-apple step counts."""

    def __init__(self, seed, outcome, score, steps):
        self.seed = seed
        self.outcome = outcome
        self.score = score
        self.steps = steps
        self.pace = []

    def occupancy(self):
        """Fraction of the board the snake filled before the game ended."""
        return (self.score + STARTING_SEGMENTS) / BOARD_CELLS

    def budget_left(self):
        """Steps remaining against the cap when the game ended."""
        return STEP_LIMIT - self.steps

    def steps_after_last_apple(self):
        """Steps taken between the final apple and the end of the game."""
        return self.steps - sum(self.pace)

    def last_apple_against_median(self):
        """The final apple's cost divided by this game's own median apple cost.

        None when the game is too short for its own median to mean anything.
        """
        if len(self.pace) < 8:
            return None
        median_cost = statistics.median(self.pace[:-1])
        if median_cost <= 0.0:
            return None
        return self.pace[-1] / median_cost


def read_games(log_path):
    """Parse every game and its pace line out of one evaluation log."""
    games = []
    by_seed = {}
    for line in log_path.read_text(encoding="utf-8").splitlines():
        game_match = GAME_LINE.match(line)
        if game_match is not None:
            game = Game(int(game_match.group(1)), game_match.group(2),
                        int(game_match.group(3)), int(game_match.group(4)))
            games.append(game)
            by_seed[game.seed] = game
            continue
        pace_match = PACE_LINE.match(line)
        if pace_match is not None:
            seed = int(pace_match.group(1))
            by_seed[seed].pace = [int(value) for value in pace_match.group(2).split()]
    return games


def quantiles(values):
    """Minimum, quartiles, median and maximum of a list, as a formatted string."""
    ordered = sorted(values)
    lower, middle, upper = statistics.quantiles(ordered, n=4)
    return (f"min {ordered[0]}, 25th {lower:.0f}, median {middle:.0f}, "
            f"75th {upper:.0f}, max {ordered[-1]}")


def report(log_path):
    """Print the death profile for one evaluation log."""
    games = read_games(log_path)
    deaths = [game for game in games if game.outcome == "died"]
    wins = [game for game in games if game.outcome == "won"]
    timeouts = [game for game in games if game.outcome == "timeout"]

    print(f"\n=== {log_path.name} ===")
    print(f"{len(games)} games: {len(wins)} won, {len(deaths)} died, "
          f"{len(timeouts)} timed out")
    if not deaths:
        return

    scores = [game.score for game in deaths]
    print(f"\nscore at death:      {quantiles(scores)}")
    print(f"occupancy at death:  mean "
          f"{statistics.mean(game.occupancy() for game in deaths):.3f}, "
          f"median {statistics.median(game.occupancy() for game in deaths):.3f}")

    # A death with clock to spare is a play failure; one against the cap is pace.
    budgets = [game.budget_left() for game in deaths]
    print(f"budget left at death: {quantiles(budgets)}")
    comfortable = sum(1 for value in budgets if value > 200)
    print(f"  died with over 200 steps to spare: {comfortable} of {len(deaths)} "
          f"({100.0 * comfortable / len(deaths):.1f} percent)")

    # Deaths concentrated in one band of the fill point at a phase, not at the policy.
    bands = [(0, 24), (25, 49), (50, 74), (75, 89), (90, 99)]
    print("\ndeaths by score band, against how many games reached that band:")
    for low, high in bands:
        died_here = sum(1 for game in deaths if low <= game.score <= high)
        reached = sum(1 for game in games if game.score >= low)
        rate = 100.0 * died_here / reached if reached > 0 else 0.0
        print(f"  score {low:2d}-{high:2d}: {died_here:3d} deaths, "
              f"{reached:4d} games reached it, {rate:.2f} percent died there")

    # A slow final apple means the death ended a struggle; a fast one means it was sudden.
    ratios = [game.last_apple_against_median() for game in deaths]
    ratios = [value for value in ratios if value is not None]
    if ratios:
        struggled = sum(1 for value in ratios if value > 3.0)
        print(f"\nlast apple against the game's own median, {len(ratios)} deaths:")
        print(f"  median ratio {statistics.median(ratios):.2f}, "
              f"mean {statistics.mean(ratios):.2f}")
        print(f"  over 3x the median: {struggled} ({100.0 * struggled / len(ratios):.1f} "
              f"percent) - the death ended a struggle rather than arriving suddenly")

    tails = [game.steps_after_last_apple() for game in deaths]
    print(f"\nsteps between the last apple and the death: {quantiles(tails)}")


def main(argv):
    if len(argv) < 2:
        print("usage: analyze_deaths.py <evaluation log> [more logs]")
        return 2
    for name in argv[1:]:
        report(Path(name))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
