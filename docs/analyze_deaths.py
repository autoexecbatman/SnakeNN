"""Where in the fill the agent dies, read from an evaluation log.

Takes evaluation logs written by AlphaZeroEvaluate, keeps the games that ended in a
death, and reports three things about them: the board occupancy reached before dying,
how much of the step budget was left, and how the last apple before the death compares
with that game's own median apple. Reads only; runs nothing.

Usage - one or more logs, any board size:

    python docs/analyze_deaths.py build/Release/eval_steps340_s800.log

The board and the step limit come from the log's own header line. They used to be the
constants 100 and 1200, which on a 20x20 log reported an occupancy of 1.72 and a negative
budget - impossible values rather than merely wrong ones, and they were quoted before
anyone noticed. A log whose header cannot be parsed is refused rather than guessed at.

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

HEADER_LINE = re.compile(r" on (\d+)x(\d+),.*step limit (\d+)")

# The snake starts at one segment, so occupancy is the score plus that first segment.
STARTING_SEGMENTS = 1


class Board:
    """The board one log was played on, read from its header line.

    Example:

        board = Board(20, 20, 9600)
        board.cells          # 400
        board.foods_to_win   # 399

    Args:
        width, height: board dimensions in cells.
        step_limit: the move budget each game played under.
    """

    def __init__(self, width, height, step_limit):
        self.width = width
        self.height = height
        self.cells = width * height
        self.foods_to_win = self.cells - STARTING_SEGMENTS
        self.step_limit = step_limit

    def __str__(self):
        return f"{self.width}x{self.height}, step limit {self.step_limit}"


class Game:
    """One evaluated game: its outcome, what it scored, and its per-apple step counts."""

    def __init__(self, seed, outcome, score, steps, board):
        self.seed = seed
        self.outcome = outcome
        self.score = score
        self.steps = steps
        self.board = board
        self.pace = []

    def occupancy(self):
        """Fraction of the board the snake filled before the game ended.

        Asserted to be a fraction. A value above 1 means the board was read wrongly, which
        is the exact failure this is guarding: it is cheaper to stop than to print a
        number nobody can tell is impossible until they think about it.
        """
        filled = (self.score + STARTING_SEGMENTS) / self.board.cells
        assert 0.0 <= filled <= 1.0, (
            f"occupancy {filled:.3f} on a {self.board} board - the board size is wrong"
        )
        return filled

    def budget_left(self):
        """Steps remaining against the cap when the game ended."""
        return self.board.step_limit - self.steps

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


def read_board(lines):
    """The board and step limit from a log's header line.

    Example:

        read_board(["az20.pt on 20x20, 4 games, 800 simulations, step limit 9600"])
        # Board of 400 cells, 399 apples to win

    Raises ValueError when no header line is present, rather than guessing a board.

    Args:
        lines: the log's lines, in order.
    """
    for line in lines:
        found = HEADER_LINE.search(line)
        if found is not None:
            return Board(int(found.group(1)), int(found.group(2)), int(found.group(3)))
    raise ValueError(
        "no header line giving the board and step limit - is this an evaluation log?")


def read_games(log_path):
    """Parse the board, then every game and its pace line, out of one evaluation log."""
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    board = read_board(lines)
    games = []
    by_seed = {}
    for line in lines:
        game_match = GAME_LINE.match(line)
        if game_match is not None:
            game = Game(int(game_match.group(1)), game_match.group(2),
                        int(game_match.group(3)), int(game_match.group(4)), board)
            games.append(game)
            by_seed[game.seed] = game
            continue
        pace_match = PACE_LINE.match(line)
        if pace_match is not None:
            seed = int(pace_match.group(1))
            by_seed[seed].pace = [int(value) for value in pace_match.group(2).split()]
    return games, board


def quantiles(values):
    """Minimum, quartiles, median and maximum of a list, as a formatted string."""
    ordered = sorted(values)
    lower, middle, upper = statistics.quantiles(ordered, n=4)
    return (f"min {ordered[0]}, 25th {lower:.0f}, median {middle:.0f}, "
            f"75th {upper:.0f}, max {ordered[-1]}")


def report(log_path):
    """Print the death profile for one evaluation log."""
    games, board = read_games(log_path)
    deaths = [game for game in games if game.outcome == "died"]
    wins = [game for game in games if game.outcome == "won"]
    timeouts = [game for game in games if game.outcome == "timeout"]

    print(f"\n=== {log_path.name} ===")
    print(f"{board}, {board.foods_to_win} apples to win")
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
    # Expressed as fractions of the win condition, so a band means the same thing on
    # every board rather than being five absolute ranges that only suit 10x10.
    edges = [0.0, 0.25, 0.50, 0.75, 0.90, 1.0]
    bands = [(round(edges[index] * board.foods_to_win),
              round(edges[index + 1] * board.foods_to_win) - 1)
             for index in range(len(edges) - 1)]
    print("\ndeaths by score band, against how many games reached that band:")
    for low, high in bands:
        died_here = sum(1 for game in deaths if low <= game.score <= high)
        reached = sum(1 for game in games if game.score >= low)
        rate = 100.0 * died_here / reached if reached > 0 else 0.0
        print(f"  score {low:3d}-{high:3d}: {died_here:3d} deaths, "
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
        # A log this cannot read is named rather than crashing with a traceback, so the
        # reason is the first thing on screen instead of the last.
        try:
            report(Path(name))
        except ValueError as refusal:
            print("")
            print(f"=== {name} ===")
            print(f"  refused: {refusal}")
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
