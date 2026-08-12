"""Is az::VALUE_SCALE an upper bound on the returns the value head must represent?

The head is VALUE_SCALE * tanh(...), so a return above VALUE_SCALE cannot be
represented at all and one just below it lands where tanh has almost no resolution.
The constant was chosen from the largest return measured over 1000 games, 16.433,
which is evidence about the play observed and not a bound on the play possible.

Given to Z3 rather than argued: the discounted return is a linear function of a
reward sequence the environment's own constants constrain, so whether any admissible
sequence exceeds a threshold is decidable here.

Model, from snake_env.h and selfplay.cpp:
  reward at step k = STEP_REWARD + (FOOD_REWARD if that step ate) + (terminal)
  return           = sum over k of DISCOUNT^k * reward(k), up to the terminal step
  at most FOODS_TO_WIN apples, since each one fills a cell for good
  exactly one terminal step, paying WIN_REWARD or DEATH_REWARD

Encoded with booleans only - `alive` is monotone and the terminal step is the last
alive one - so the query stays in linear real arithmetic. An integer terminal index
compared against every step made the same question take longer than ten minutes.

What the model does NOT constrain is geometry: it permits an apple on every step,
which requires each apple to spawn next to the head. That happens late in a game and
not throughout one, so a counterexample here bounds the contract, not the observed
distribution.
"""

import sys
from fractions import Fraction

import z3

DISCOUNT = Fraction(49, 50)  # 0.98 exactly
STEP_REWARD = Fraction(-1, 50)  # -0.02 exactly
FOOD_REWARD = Fraction(1)
WIN_REWARD = Fraction(10)
DEATH_REWARD = Fraction(-10)
FOODS_TO_WIN = 99  # 10x10, snake starts one segment long
HORIZON = 110  # past FOODS_TO_WIN; steps beyond an apple only subtract
QUERY_TIMEOUT_MS = 120000

VALUE_SCALE = Fraction(40)


def rational(value):
    return z3.RealVal(value.numerator) / z3.RealVal(value.denominator)


def build():
    """The return of an admissible game, with the constraints that make it one."""
    ate = [z3.Bool("ate_%d" % step) for step in range(HORIZON)]
    alive = [z3.Bool("alive_%d" % step) for step in range(HORIZON)]
    won = z3.Bool("won")

    constraints = [alive[0]]
    # Monotone: once the game is over it stays over, so the terminal step is the
    # last alive one and no second ending can be paid.
    for step in range(HORIZON - 1):
        constraints.append(z3.Implies(alive[step + 1], alive[step]))
    # Nothing is eaten after the game ends.
    for step in range(HORIZON):
        constraints.append(z3.Implies(ate[step], alive[step]))
    constraints.append(z3.Sum([z3.If(flag, 1, 0) for flag in ate]) <= FOODS_TO_WIN)

    ending = z3.If(won, rational(WIN_REWARD), rational(DEATH_REWARD))
    discounted = Fraction(1)
    total = z3.RealVal(0)
    for step in range(HORIZON):
        weight = rational(discounted)
        paid = z3.If(ate[step], rational(FOOD_REWARD), z3.RealVal(0)) + rational(STEP_REWARD)
        # The last alive step is the terminal one and pays the outcome.
        last = alive[step] if step == HORIZON - 1 else z3.And(alive[step], z3.Not(alive[step + 1]))
        total = total + z3.If(alive[step], weight * paid, z3.RealVal(0))
        total = total + z3.If(last, weight * ending, z3.RealVal(0))
        discounted *= DISCOUNT

    return total, constraints


def ask(description, threshold, expected):
    """Assert the negation of 'return <= threshold' and report what Z3 says."""
    total, constraints = build()
    solver = z3.Solver()
    solver.set("timeout", QUERY_TIMEOUT_MS)
    solver.add(constraints)
    solver.add(total > rational(threshold))
    verdict = str(solver.check())
    if verdict == "unknown":
        note = "TIMED OUT - unresolved, not a proof"
    elif verdict == expected:
        note = "as expected"
    else:
        note = "UNEXPECTED - expected %s" % expected
    print("  %-44s %-7s %s" % (description, verdict, note))
    sys.stdout.flush()
    if verdict == "sat":
        model = solver.model()
        apples = sum(
            1 for step in range(HORIZON) if z3.is_true(model.eval(z3.Bool("ate_%d" % step)))
        )
        steps = sum(
            1 for step in range(HORIZON) if z3.is_true(model.eval(z3.Bool("alive_%d" % step)))
        )
        print(
            "        witness: %d apples over %d steps, won=%s"
            % (apples, steps, model.eval(z3.Bool("won")))
        )
        sys.stdout.flush()
    return verdict


def main():
    print("Z3 %s, exact rational arithmetic" % z3.get_version_string())
    print("Negating 'discounted return <= threshold'. unsat means the bound holds.")
    sys.stdout.flush()

    # Controls first. An encoding that answered every query the same way would look
    # exactly like a proof, so both directions have to be shown reachable.
    ask("control, a return above 0 is reachable", Fraction(0), "sat")
    ask("control, a return above 1000 is impossible", Fraction(1000), "unsat")
    print()
    ask("VALUE_SCALE = 40 bounds the return", VALUE_SCALE, "sat")
    ask("60 bounds the return", Fraction(60), "unsat")
    print()

    low, high = 0, 60
    while low + 1 < high:
        middle = (low + high) // 2
        total, constraints = build()
        solver = z3.Solver()
        solver.set("timeout", QUERY_TIMEOUT_MS)
        solver.add(constraints)
        solver.add(total > z3.RealVal(middle))
        verdict = str(solver.check())
        if verdict == "unknown":
            print("bisection timed out at %d - no least bound established" % middle)
            return
        if verdict == "sat":
            low = middle
        else:
            high = middle
    print("least integer bound that provably holds: %d" % high)
    print("largest return measured over 1000 games:  16.433")


if __name__ == "__main__":
    main()
