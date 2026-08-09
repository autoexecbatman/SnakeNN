"""Machine-checks the integer arithmetic of the AlphaZero option types with Z3.

Each claim is stated as a search for a counterexample: a claim that should hold is
`unsat`, a claim that an expression overflows is `sat` and prints the witness. Every
claim also carries a mutant - one constant changed - which must come back the other
way, so a check that cannot fire is reported rather than counted as a pass.

Run: python docs/prove_arithmetic.py
Exit code 0 when every claim and every mutant answered as expected.
"""

import sys

from z3 import Int, Solver, sat, set_option, unknown, unsat

INT_MAX = 2**31 - 1
INT_MIN = -(2**31)
INT64_MAX = 2**63 - 1
UINT32_MAX = 2**32 - 1
STEPS_PER_CELL = 12  # src/az_parameters.h:21
LARGEST_BOARD = 13377  # src/az_parameters.h:26
EVALUATION_BASE = 0xE0000000  # src/seed_policy.h:24
RESERVED_BAND_WIDTH = UINT32_MAX - EVALUATION_BASE + 1
BYTES_PER_MEBIBYTE = 1024 * 1024


def guard_is_sound(bound):
    """No accepted board overflows an int when its step limit is formed.

    src/az_parameters.cpp:13-22. The guard is a single comparison against
    LARGEST_BOARD, so soundness is the claim that no board at or below it has a
    step limit past INT_MAX. The header's two static_asserts check the boundary
    itself; only this quantifies over every board.
    """
    board = Int("board")
    return [board >= 2, board <= bound, STEPS_PER_CELL * board * board > INT_MAX]


def guard_is_complete(bound):
    """No board is rejected whose step limit would in fact have fitted.

    The other direction of the same guard. A sound but over-strict bound would
    refuse a board the evaluator could have run, and no test enumerates boards.
    """
    board = Int("board")
    return [board > bound, board <= INT_MAX, STEPS_PER_CELL * board * board <= INT_MAX]


def area_fits_in_long_long(width):
    """board * board as long long cannot overflow for any int board.

    src/az_parameters.cpp:17 widens to long long before squaring and never checks
    the product. The claim is that the widening alone makes the check unnecessary.
    """
    board = Int("board")
    return [board >= -(2 ** (width - 1)), board <= 2 ** (width - 1) - 1,
            board * board > INT64_MAX]


def replay_bytes_fits_in_size_t(width):
    """--replay-mb times a mebibyte cannot overflow size_t.

    src/trainer_options.cpp:161-163 parses to int, requires at least 1, then casts
    to size_t and multiplies. The cast is what makes the product safe.
    """
    megabytes = Int("megabytes")
    return [megabytes >= 1, megabytes <= INT_MAX,
            megabytes * BYTES_PER_MEBIBYTE > 2**width - 1]


def evaluation_cell_count_is_safe(upper_bound):
    """evaluation::Settings::cellCount cannot overflow, because board is bounded.

    src/eval_options.cpp:39-40 rejects a board above az::LARGEST_BOARD, which is
    what makes the int multiplication at src/eval_options.h:50 safe. The bound was
    chosen for the step limit; this is the claim that it covers the area too.
    """
    board = Int("board")
    return [board >= 2, board <= upper_bound, board * board > INT_MAX]


def seed_band_check_is_exact(band_width):
    """The seed range check accepts exactly the runs that stay in the band.

    src/eval_options.cpp:56-59. Every game seed is EVALUATION_BASE + offset +
    index in unsigned 32-bit arithmetic, so a run reaching past the top of the
    band wraps to a low seed - a training seed - without erroring. A counterexample
    is a run the check and the wraparound disagree about.
    """
    offset = Int("seed_offset")
    games = Int("games")
    last_index = offset + games - 1
    accepted = last_index < band_width
    stays_in_band = EVALUATION_BASE + last_index <= UINT32_MAX
    return [offset >= 0, offset <= UINT32_MAX, games >= 1, games <= INT_MAX,
            accepted != stays_in_band]


def trainer_cell_count_is_safe(upper_bound):
    """trainer::Settings::cellCount cannot overflow, because board is bounded.

    src/trainer_options.h:53 squares the board in an int. requireUsable now bounds
    it above at az::LARGEST_BOARD as well as below at 2. Until 2026-08-09 it bounded
    only the lower end and a release build accepted board 46341, which is undefined
    behaviour - the ceiling existed but was reached through an assert, so it was
    absent from every build that ever trained anything.
    """
    board = Int("board")
    return [board >= 2, board <= upper_bound, board * board > INT_MAX]


def last_iteration_is_safe(ceiling):
    """trainer::Settings::lastIteration cannot overflow.

    src/trainer_options.h:59 computes start_iteration + iterations - 1 in an int.
    requireUsable rejects the pair whose sum passes INT_MAX, compared in 64 bits
    because the sum in 32 is the overflow being rejected.
    """
    start_iteration = Int("start_iteration")
    iterations = Int("iterations")
    last = start_iteration + iterations - 1
    return [start_iteration >= 1, iterations >= 1, start_iteration <= INT_MAX,
            iterations <= INT_MAX, last <= ceiling, last > INT_MAX]


CLAIMS = [
    {
        "name": "guard_is_sound",
        "source": "src/az_parameters.cpp:13-22",
        "build": guard_is_sound,
        "expected": unsat,
        "true_value": LARGEST_BOARD,
        "mutant_value": LARGEST_BOARD + 1,
        "mutant_note": "one board higher admits a step limit past INT_MAX",
    },
    {
        "name": "guard_is_complete",
        "source": "src/az_parameters.cpp:13-22",
        "build": guard_is_complete,
        "expected": unsat,
        "true_value": LARGEST_BOARD,
        "mutant_value": LARGEST_BOARD - 1,
        "mutant_note": "one board lower refuses a board whose step limit does fit",
    },
    {
        "name": "area_fits_in_long_long",
        "source": "src/az_parameters.cpp:17",
        "build": area_fits_in_long_long,
        "expected": unsat,
        "true_value": 32,
        "mutant_value": 64,
        "mutant_note": "squaring a long long overflows, so the widening is load-bearing",
    },
    {
        "name": "replay_bytes_fits_in_size_t",
        "source": "src/trainer_options.cpp:161-163",
        "build": replay_bytes_fits_in_size_t,
        "expected": unsat,
        "true_value": 64,
        "mutant_value": 32,
        "mutant_note": "a 32-bit size_t overflows above 2047 mebibytes",
    },
    {
        "name": "evaluation_cell_count_is_safe",
        "source": "src/eval_options.h:50",
        "build": evaluation_cell_count_is_safe,
        "expected": unsat,
        "true_value": LARGEST_BOARD,
        "mutant_value": INT_MAX,
        "mutant_note": "without the board bound the area overflows, as it does in the trainer",
    },
    {
        "name": "seed_band_check_is_exact",
        "source": "src/eval_options.cpp:56-59",
        "build": seed_band_check_is_exact,
        "expected": unsat,
        "true_value": RESERVED_BAND_WIDTH,
        "mutant_value": RESERVED_BAND_WIDTH + 1,
        "mutant_note": "a band one seed too wide accepts the run that wraps to a training seed",
    },
    {
        "name": "trainer_cell_count_is_safe",
        "source": "src/trainer_options.h:53",
        "build": trainer_cell_count_is_safe,
        "expected": unsat,
        "true_value": LARGEST_BOARD,
        "mutant_value": INT_MAX,
        "mutant_note": "without the ceiling the area overflows, as it did before 2026-08-09",
    },
    {
        "name": "last_iteration_is_safe",
        "source": "src/trainer_options.h:59",
        "build": last_iteration_is_safe,
        "expected": unsat,
        "true_value": INT_MAX,
        "mutant_value": 2 * INT_MAX,
        "mutant_note": "a ceiling above INT_MAX admits the sum the check exists to reject",
    },
]


def run(constraints):
    """The solver's answer, and a witness assignment when there is one."""
    solver = Solver()
    for constraint in constraints:
        solver.add(constraint)
    answer = solver.check()
    if answer == sat:
        return answer, solver.model()
    return answer, None


def describe(answer, model):
    if answer == unsat:
        return "unsat", "[machine-checked] no counterexample exists"
    if answer == sat:
        assignment = ", ".join(
            "{} = {}".format(name, model[name]) for name in sorted(model.decls(), key=str)
        )
        return "sat", "[counterexample-found] {}".format(assignment or "trivially satisfiable")
    return "unknown", "[unresolved-no-method] the solver did not decide"


def main():
    set_option("timeout", 30000)
    failures = 0

    for claim in CLAIMS:
        expected_word = "unsat" if claim["expected"] == unsat else "sat"

        answer, model = run(claim["build"](claim["true_value"]))
        word, detail = describe(answer, model)
        status = "OK" if answer == claim["expected"] else "FAIL"
        if answer != claim["expected"]:
            failures += 1
        print("[{}] {:<28} {:<20} expected {}, got {}".format(
            status, claim["name"], claim["source"], expected_word, word))
        print("       {}".format(detail))

        mutant_answer, mutant_model = run(claim["build"](claim["mutant_value"]))
        mutant_word, mutant_detail = describe(mutant_answer, mutant_model)
        killed = mutant_answer != claim["expected"] and mutant_answer != unknown
        if not killed:
            failures += 1
        print("       mutant {} -> {}: {}  ({})".format(
            claim["mutant_value"], mutant_word,
            "check can fire" if killed else "CHECK IS VACUOUS - it answers the same either way",
            claim["mutant_note"]))
        if killed and mutant_answer == sat:
            print("       {}".format(mutant_detail))

    print()
    if failures == 0:
        print("all {} claims answered as expected, and every check was shown able to fire".format(
            len(CLAIMS)))
    else:
        print("{} failures".format(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
