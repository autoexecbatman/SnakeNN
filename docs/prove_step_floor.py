"""A lower bound on the steps any Snake policy needs to fill the board, machine-checked.

The question this answers: at the paper's cap of 1200 steps on 10x10, how much of
that budget is forced by the task rather than wasted by the policy? We measure 1126
and win 75 percent of the time, so the cap is binding - but nothing said whether a
much faster fill is even possible.

The theorem below is a floor on the expected number of steps, holding for every
policy, including an optimal one. It is proved by hand in four steps, two of which
are checked here: Lemma 1 by Z3, and the whole bound a second time by exhaustive
enumeration on a small board, so the reported number comes from two mechanisms.

Run: python docs/prove_step_floor.py
Exit code 0 when every claim, every mutant and the cross-check answer as expected.
"""

import itertools
import sys

from z3 import Bool, If, Int, Real, Solver, sat, unsat

# The environment this bounds: src/snake_env.cpp. The snake starts at length 1 and
# wins at length N, so it eats N-1 apples, and each apple is placed uniformly over
# the cells the snake does not occupy.
STARTING_LENGTH = 1


# ---------------------------------------------------------------------------
# The theorem
# ---------------------------------------------------------------------------
#
# Board B is the n x n grid, N = n^2 cells. Write dist(a, b) for the Manhattan
# distance. T is the total number of steps a game takes to win.
#
# Step 1 (hand-derived). Apple k is placed the instant apple k-1 is eaten, so the
#   N-1 intervals "apple k is on the board" are disjoint and tile the whole game.
#   Therefore T = sum over k of (steps taken during interval k).
#
# Step 2 (hand-derived). One step moves the head to an orthogonally adjacent cell,
#   which changes its Manhattan distance to any fixed cell by exactly 1. So
#   reaching apple k from the head position h_k takes at least dist(h_k, apple_k)
#   steps - and at least that many even though the body may force a detour, since
#   a detour only adds. Hence T >= sum over k of dist(h_k, apple_k).
#
# Step 3 (Lemma 1, machine-checked below). When apple k is placed the snake has
#   length k, so it occupies a set O of exactly k cells with the head h in O, and
#   the apple is uniform over the N-k cells outside O. Whatever the policy did to
#   produce O, the expected distance is at least what the most favourable O would
#   give. Removing the k-1 cells of *largest* distance (plus h itself, which is
#   forced and sits at distance 0) is that most favourable choice, so
#
#       E[dist(h_k, apple_k)] >= A_k(h) := mean of the N-k smallest values
#                                         in { dist(h, c) : c in B, c != h }
#
#   Lemma 1 is exactly the claim that the prefix of the sorted list is the
#   minimising choice, and it is the one step that is not obvious.
#
# Step 4. Minimising over the head position as well gives a bound that no longer
#   mentions the policy at all:
#
#       E[T] >= F(n) := sum over k = 1..N-1 of min over h in B of A_k(h)
#
# What the bound gives up, stated rather than hidden: O ranges over all k-subsets
# containing h, and most of those are not shapes a snake can hold. The true floor
# is therefore at least F(n) and probably well above it. A loose lower bound still
# settles the question it was built for, in one direction only - if F(n) is far
# below the cap, the cap is not the obstacle.


def manhattan_distances_from(cell, side):
    """Every distance from `cell` to another cell of a side x side board, sorted."""
    head_row, head_col = cell
    distances = []
    for row in range(side):
        for col in range(side):
            if (row, col) != cell:
                distances.append(abs(row - head_row) + abs(col - head_col))
    distances.sort()
    return distances


def bound_for_head(distances, cells_left):
    """A_k(h): the mean of the `cells_left` smallest distances."""
    kept = distances[:cells_left]
    return sum(kept) / len(kept)


def step_floor(side):
    """F(n), by the sorted-prefix formula of Lemma 1."""
    cell_count = side * side
    heads = [(row, col) for row in range(side) for col in range(side)]
    distances_by_head = {head: manhattan_distances_from(head, side) for head in heads}

    total = 0.0
    per_apple = []
    for length in range(STARTING_LENGTH, cell_count):
        cells_left = cell_count - length
        best = min(bound_for_head(distances_by_head[head], cells_left) for head in heads)
        per_apple.append(best)
        total += best
    return total, per_apple


def step_floor_by_enumeration(side):
    """F(n) again, minimising over every occupied set explicitly.

    A different mechanism for the same number: no sorting, no appeal to Lemma 1,
    just every k-subset that contains the head. Exponential, so only a small board
    is reachable - which is the point, since agreement there is what licenses the
    formula on the boards we care about.
    """
    cell_count = side * side
    cells = [(row, col) for row in range(side) for col in range(side)]

    total = 0.0
    for length in range(STARTING_LENGTH, cell_count):
        best = None
        for head in cells:
            others = [cell for cell in cells if cell != head]
            for occupied_rest in itertools.combinations(others, length - 1):
                occupied = set(occupied_rest) | {head}
                empty = [cell for cell in cells if cell not in occupied]
                mean = sum(
                    abs(cell[0] - head[0]) + abs(cell[1] - head[1]) for cell in empty
                ) / len(empty)
                if best is None or mean < best:
                    best = mean
        total += best
    return total


# ---------------------------------------------------------------------------
# Lemma 1, in Z3
# ---------------------------------------------------------------------------


def lemma_prefix_is_minimal(value_count, keep_count):
    """No selection of `keep_count` values beats the sorted prefix.

    The values are symbolic and only constrained to be sorted and non-negative, so
    `unsat` proves the lemma for every board and every distance list of this length,
    not for one instance of it.
    """
    values = [Int(f"value_{index}") for index in range(value_count)]
    keeps = [Bool(f"keep_{index}") for index in range(value_count)]

    constraints = [values[0] >= 0]
    constraints += [values[index] <= values[index + 1] for index in range(value_count - 1)]
    constraints.append(
        sum(If(keep, 1, 0) for keep in keeps) == keep_count
    )

    selected_total = sum(If(keeps[index], values[index], 0) for index in range(value_count))
    prefix_total = sum(values[:keep_count])
    # The counts are equal, so comparing sums compares means.
    constraints.append(selected_total < prefix_total)
    return constraints


def mutant_prefix_is_maximal(value_count, keep_count):
    """The same claim pointed the other way, which must fail.

    If this came back unsat too, the encoding would be unsatisfiable for a reason
    that has nothing to do with the lemma - a contradiction in the constraints - and
    the lemma's `unsat` would mean nothing.
    """
    values = [Int(f"value_{index}") for index in range(value_count)]
    keeps = [Bool(f"keep_{index}") for index in range(value_count)]

    constraints = [values[0] >= 0]
    constraints += [values[index] <= values[index + 1] for index in range(value_count - 1)]
    constraints.append(sum(If(keep, 1, 0) for keep in keeps) == keep_count)

    selected_total = sum(If(keeps[index], values[index], 0) for index in range(value_count))
    prefix_total = sum(values[:keep_count])
    constraints.append(selected_total > prefix_total)
    return constraints


def lemma_head_is_forced_into_the_occupied_set(value_count, keep_count):
    """Dropping "the head is occupied" cannot raise the floor.

    The bound keeps h out of the empty set by construction. This checks the
    direction that would break the proof: that a selection is free to keep the
    head's own zero and so come in below the prefix. With the zero already at the
    front of the sorted list, the prefix contains it, so no selection can.
    """
    values = [Int(f"value_{index}") for index in range(value_count)]
    keeps = [Bool(f"keep_{index}") for index in range(value_count)]

    constraints = [values[0] == 0]
    constraints += [values[index] >= 0 for index in range(value_count)]
    constraints += [values[index] <= values[index + 1] for index in range(value_count - 1)]
    constraints.append(sum(If(keep, 1, 0) for keep in keeps) == keep_count)

    selected_total = sum(If(keeps[index], values[index], 0) for index in range(value_count))
    constraints.append(selected_total < sum(values[:keep_count]))
    return constraints


def check(name, constraints, expected):
    solver = Solver()
    solver.set("timeout", 120000)
    for constraint in constraints:
        solver.add(constraint)
    answer = solver.check()
    ok = answer == expected
    print(f"  {'PASS' if ok else 'FAIL'}  {name}: {answer} (expected {expected})")
    if answer == sat and expected == sat:
        pass
    elif answer == sat and expected == unsat:
        print(f"        counterexample: {solver.model()}")
    return ok


def main():
    failures = 0

    print("Lemma 1 - the sorted prefix is the minimising selection")
    for value_count, keep_count in [(4, 2), (6, 3), (7, 2), (8, 5), (9, 4)]:
        label = f"{keep_count} of {value_count}"
        if not check(f"no selection beats the prefix, {label}",
                     lemma_prefix_is_minimal(value_count, keep_count), unsat):
            failures += 1
        if not check(f"and some selection beats it upward, {label} (mutant)",
                     mutant_prefix_is_maximal(value_count, keep_count), sat):
            failures += 1
        if not check(f"the head's own zero is already in the prefix, {label}",
                     lemma_head_is_forced_into_the_occupied_set(value_count, keep_count), unsat):
            failures += 1

    print("\nCross-check - the formula against exhaustive enumeration")
    for side in [3, 4]:
        by_formula, _ = step_floor(side)
        by_enumeration = step_floor_by_enumeration(side)
        residual = abs(by_formula - by_enumeration)
        ok = residual < 1e-9
        print(f"  {'PASS' if ok else 'FAIL'}  {side}x{side}: formula {by_formula:.6f}, "
              f"enumeration {by_enumeration:.6f}, residual {residual:.2e}")
        if not ok:
            failures += 1

    print("\nThe bound")
    for side, cap in [(6, None), (10, 1200), (20, None)]:
        total, per_apple = step_floor(side)
        print(f"  {side}x{side}: F = {total:.1f} steps over {side * side - 1} apples, "
              f"first apple {per_apple[0]:.2f}, last {per_apple[-1]:.2f}")
        if cap is not None:
            print(f"          against a cap of {cap}: the floor is "
                  f"{100.0 * total / cap:.1f} percent of the budget")

    print()
    if failures == 0:
        print("Every claim, mutant and cross-check answered as expected.")
        return 0
    print(f"{failures} check(s) did not.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
