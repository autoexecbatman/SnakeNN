"""Checks that averaging an edge's tick count understates its discount factor.

`actionScore` discounts a child's return by `discount^edge_steps`. Forced moves are
collapsed into an edge, so `edge_steps` varies across the placements reaching one
open-loop node and the exponent is a random variable. The question is whether the
accumulator may hold the mean tick count, or must hold the mean discount factor:
whether `discount^E[k]` equals `E[discount^k]`.

Two mechanisms, deliberately different. z3 decides a fully polynomial instance - a
two-point tick count whose mean is an integer, so no fractional power appears and the
question stays in a decidable fragment. sympy derives the general statement from the
second derivative. Mathlib is not installed on this machine, so `[machine-checked]`
here means the SMT result.
"""

import sympy
import z3


def machine_check_two_point_instance() -> None:
    """z3 on ticks of 1 or 3 with equal weight, so the mean tick count is exactly 2.

    Every term is a polynomial in the discount, which keeps this inside a fragment z3
    decides. The negation is asserted: a discount in (0, 1) for which averaging the
    exponent is at least as large as averaging the factor. `unsat` means no such
    discount exists.
    """
    discount = z3.Real("discount")
    averaged_factor = z3.RealVal(1) / 2 * discount + z3.RealVal(1) / 2 * discount**3
    factor_of_averaged_ticks = discount**2

    solver = z3.Solver()
    solver.add(discount > 0, discount < 1)
    solver.add(averaged_factor <= factor_of_averaged_ticks)
    result = solver.check()
    print(f"  negation is {result} (expect unsat)")
    if result == z3.unsat:
        print("  so E[discount^k] > discount^E[k] for every discount in (0, 1)")
    else:
        print(f"  counterexample: {solver.model()}")


def falsify_the_check() -> None:
    """The same query with the inequality reversed, which must be satisfiable.

    A solver call that cannot return sat is a check that cannot fire, and its unsat
    above would mean nothing. This is the deliberate violation that proves it can.
    """
    discount = z3.Real("discount")
    averaged_factor = z3.RealVal(1) / 2 * discount + z3.RealVal(1) / 2 * discount**3
    factor_of_averaged_ticks = discount**2

    solver = z3.Solver()
    solver.add(discount > 0, discount < 1)
    solver.add(averaged_factor > factor_of_averaged_ticks)
    result = solver.check()
    print(f"  reversed inequality is {result} (expect sat)")
    if result == z3.sat:
        print(f"  witness: discount = {solver.model()[discount]}")


def derive_the_general_case() -> None:
    """sympy on the second derivative, which is what makes Jensen strict.

    discount^k is convex in k when the discount is in (0, 1), so the mean of the
    factors exceeds the factor of the mean for any non-degenerate tick distribution.
    """
    discount, ticks = sympy.symbols("discount ticks", positive=True)
    factor = discount**ticks
    second_derivative = sympy.simplify(sympy.diff(factor, ticks, 2))
    print(f"  d2/dk2 of discount^k = {second_derivative}")
    print("  log(discount)^2 > 0 and discount^k > 0, so the factor is strictly convex")


def size_the_error_at_the_projects_discount() -> None:
    """What the approximation costs at discount 0.98 over plausible tick counts.

    Reported so the correction is not adopted on a sign alone. An edge here spans one
    tick usually and a few when forced moves are collapsed into it.
    """
    discount = sympy.Rational(98, 100)
    for low, high in [(1, 2), (1, 3), (1, 5), (2, 8)]:
        mean_ticks = sympy.Rational(low + high, 2)
        averaged_factor = (discount**low + discount**high) / 2
        factor_of_mean = discount**mean_ticks
        relative = float((averaged_factor - factor_of_mean) / factor_of_mean) * 100
        print(
            f"  ticks {low} or {high}: E[g^k] = {float(averaged_factor):.6f}, "
            f"g^E[k] = {float(factor_of_mean):.6f}, understated by {relative:.4f} percent"
        )


def main() -> None:
    print(f"z3 {z3.get_version_string()}, sympy {sympy.__version__}")
    print("machine check, ticks of 1 or 3:")
    machine_check_two_point_instance()
    print("falsification of that check:")
    falsify_the_check()
    print("general derivation:")
    derive_the_general_case()
    print("size at discount 0.98:")
    size_the_error_at_the_projects_discount()


if __name__ == "__main__":
    main()
