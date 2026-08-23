"""How far a bootstrapped death label can drift from the true doom probability.

The question this answers. With DEATH_RISK_FROM_NETWORK on, a leaf of the search tree
carries the death head's own estimate instead of zero, and risk propagates upward by
`node = min over children` (mcts.cpp). So a training label is the head's own output fed
back through the tree - a bootstrap. Before spending ten hours training on such labels,
two things are worth knowing: whether the error can grow as it propagates, and whether
depth washes it out the way a discount washes out a value error.

What is proved here, over the reals, with z3:

    1. One backup step is non-expansive. If every child's estimate is within eps of its
       true doom value, the parent's is within eps too. Proved by asking z3 for a
       counterexample and getting unsat, at the action count this project uses.
    2. It is not contractive. z3 finds an assignment where the parent's error equals eps
       exactly, so the bound in 1 is tight and cannot be improved.
    3. A terminal node contributes no error, because its value is exact.

What follows, and it is the useful part. Composing 1 over D levels gives an error at the
root still bounded by eps and still no smaller - the bootstrap cannot diverge, and it
cannot self-correct either. That is the opposite of value bootstrapping, where a discount
of 0.98 shrinks a leaf error to 0.98^D by the time it reaches the root. Extending the
horizon with this backup inherits the head's error at full strength.

Run from the repository root:

    python docs/prove_death_bootstrap.py

Prints each claim with the solver's verdict, and exits non-zero if any check comes back
other than expected - including the deliberately false one, which must be refuted.
"""

import sys

import z3

# The three relative actions the environment offers, so the arity proved is the arity used.
ACTION_COUNT = 3

# How deep the composition is checked explicitly before the induction is stated by hand.
DEPTH_CHECKED = 4


def minimum(values):
    """The minimum of a list of z3 reals, as nested z3.If.

    Example:

        minimum([z3.RealVal(2), z3.RealVal(1)])   # an expression equal to 1

    Args:
        values: a non-empty list of z3 arithmetic expressions.
    """
    smallest = values[0]
    for value in values[1:]:
        smallest = z3.If(value < smallest, value, smallest)
    return smallest


def one_step_is_non_expansive():
    """Claim 1: a single min backup does not amplify a leaf error.

    Asks for estimates and truths within eps of each other whose minima are further apart
    than eps. unsat means no such assignment exists.
    """
    estimates = [z3.Real(f"estimate_{index}") for index in range(ACTION_COUNT)]
    truths = [z3.Real(f"truth_{index}") for index in range(ACTION_COUNT)]
    epsilon = z3.Real("epsilon")

    solver = z3.Solver()
    solver.add(epsilon >= 0)
    # Both live in [0, 1]: a probability, and a doom value which is 0 or 1 at a terminal.
    for value in estimates + truths:
        solver.add(value >= 0, value <= 1)
    # Every child is within eps.
    for estimate, truth in zip(estimates, truths):
        solver.add(z3.Abs(estimate - truth) <= epsilon)
    # The negation of the claim: the parent is further apart than eps.
    solver.add(z3.Abs(minimum(estimates) - minimum(truths)) > epsilon)
    return solver.check()


def one_step_is_not_contractive():
    """Claim 2: the bound is tight - the parent's error can equal eps exactly.

    sat, with a witness, means depth buys no reduction. This is the claim that matters:
    it is what separates this backup from a discounted value backup.
    """
    estimates = [z3.Real(f"estimate_{index}") for index in range(ACTION_COUNT)]
    truths = [z3.Real(f"truth_{index}") for index in range(ACTION_COUNT)]
    epsilon = z3.Real("epsilon")

    solver = z3.Solver()
    # A non-trivial error, so the witness is not the vacuous eps = 0 case.
    solver.add(epsilon == z3.RealVal(1) / 4)
    for value in estimates + truths:
        solver.add(value >= 0, value <= 1)
    for estimate, truth in zip(estimates, truths):
        solver.add(z3.Abs(estimate - truth) <= epsilon)
    solver.add(z3.Abs(minimum(estimates) - minimum(truths)) == epsilon)
    verdict = solver.check()
    witness = solver.model() if verdict == z3.sat else None
    return verdict, witness


def composition_holds_to_depth(depth):
    """Claim 1 composed explicitly over `depth` levels of a full ACTION_COUNT-ary tree.

    Builds the whole tree symbolically and asks for a root error above eps. unsat at each
    depth is the induction checked rather than assumed, for the depths it is feasible at.
    """
    epsilon = z3.Real("epsilon")
    solver = z3.Solver()
    solver.add(epsilon >= 0)

    def build(prefix, level):
        # A leaf carries the head's estimate; its truth is the real doom value.
        if level == 0:
            estimate = z3.Real(f"estimate_{prefix}")
            truth = z3.Real(f"truth_{prefix}")
            solver.add(estimate >= 0, estimate <= 1, truth >= 0, truth <= 1)
            solver.add(z3.Abs(estimate - truth) <= epsilon)
            return estimate, truth
        # An internal node takes the minimum over its children, on both sides.
        pairs = [build(f"{prefix}_{index}", level - 1) for index in range(ACTION_COUNT)]
        return (minimum([pair[0] for pair in pairs]),
                minimum([pair[1] for pair in pairs]))

    root_estimate, root_truth = build("r", depth)
    solver.add(z3.Abs(root_estimate - root_truth) > epsilon)
    return solver.check()


def a_terminal_contributes_no_error():
    """Claim 3: a node whose value is exact adds nothing to the error.

    A terminal is 1 on a loss and 0 on a win, taken from the simulator rather than the
    head, so its estimate equals its truth. unsat: no error can enter through one.
    """
    exact = z3.Real("exact")
    solver = z3.Solver()
    solver.add(z3.Or(exact == 0, exact == 1))
    solver.add(z3.Abs(exact - exact) > 0)
    return solver.check()


def the_check_can_fail():
    """A deliberately false claim, to show the method refuses one.

    Asserts the parent error reaches twice eps, which min cannot produce. Expected unsat:
    no assignment satisfies it. Read together with claim 2, which comes back sat, this
    shows the solver answers in both directions rather than always agreeing.
    """
    estimates = [z3.Real(f"e_{index}") for index in range(ACTION_COUNT)]
    truths = [z3.Real(f"t_{index}") for index in range(ACTION_COUNT)]
    epsilon = z3.Real("eps")
    solver = z3.Solver()
    solver.add(epsilon == z3.RealVal(1) / 4)
    for value in estimates + truths:
        solver.add(value >= 0, value <= 1)
    for estimate, truth in zip(estimates, truths):
        solver.add(z3.Abs(estimate - truth) <= epsilon)
    # "The parent error is at least twice eps" - false, so a witness must not exist;
    # we ask for one and expect unsat.
    solver.add(z3.Abs(minimum(estimates) - minimum(truths)) >= 2 * epsilon)
    return solver.check()


def main():
    """Runs every claim and reports, exiting non-zero if any verdict is unexpected."""
    print(f"z3 {z3.get_version_string()}, action count {ACTION_COUNT}\n")
    failures = 0

    verdict = one_step_is_non_expansive()
    ok = verdict == z3.unsat
    failures += 0 if ok else 1
    print(f"1. one backup step is non-expansive          {verdict}  "
          f"(want unsat) {'ok' if ok else 'UNEXPECTED'}")

    verdict, witness = one_step_is_not_contractive()
    ok = verdict == z3.sat
    failures += 0 if ok else 1
    print(f"2. the bound is tight, no contraction        {verdict}  "
          f"(want sat)   {'ok' if ok else 'UNEXPECTED'}")
    if witness is not None:
        # Iterate the declarations the model actually assigned; indexing a model by a
        # name string is what z3 refuses.
        shown = ", ".join(f"{declaration.name()}={witness[declaration]}"
                          for declaration in sorted(witness.decls(), key=lambda d: d.name()))
        print(f"   witness: {shown}")

    verdict = a_terminal_contributes_no_error()
    ok = verdict == z3.unsat
    failures += 0 if ok else 1
    print(f"3. a terminal contributes no error           {verdict}  "
          f"(want unsat) {'ok' if ok else 'UNEXPECTED'}")

    for depth in range(1, DEPTH_CHECKED + 1):
        verdict = composition_holds_to_depth(depth)
        ok = verdict == z3.unsat
        failures += 0 if ok else 1
        leaves = ACTION_COUNT ** depth
        print(f"4.{depth} composed over {depth} levels ({leaves:3d} leaves)      {verdict}  "
              f"(want unsat) {'ok' if ok else 'UNEXPECTED'}")

    verdict = the_check_can_fail()
    ok = verdict == z3.unsat
    failures += 0 if ok else 1
    print(f"5. a false claim is refuted                  {verdict}  "
          f"(want unsat) {'ok' if ok else 'UNEXPECTED'}")

    print()
    if failures:
        print(f"{failures} unexpected verdicts")
        return 1
    print("every claim held. The bootstrap cannot amplify the head's error, and depth")
    print("does not reduce it - unlike a discounted value backup, where a leaf error")
    print("reaches the root scaled by discount to the power of the depth.")
    return 0


sys.exit(main())
