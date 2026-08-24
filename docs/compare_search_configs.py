"""Reports what search each program actually configures, field by field.

Five programs build a MonteCarloSearch::Config. Some fields must be identical in all of
them - they are the paper's constants and describe the same game - and some legitimately
differ, because a trainer explores and an evaluator does not. A field a program never sets
silently takes Config's own default, which is zero or false, and nothing in a build or a
test can see that. This reads the sources and says so.

The reference is az_evaluate.cpp: it defines the agent every win rate describes, so a
program that means to play the same agent has to agree with it on the shared fields.

Run from the repository root:

    python docs/compare_search_configs.py

Prints one row per shared field and one column per program, then a verdict. Exit code 0
when every program agrees with the evaluator on every shared field, 1 when one does not.
"""

import io
import re
import sys
from pathlib import Path

# Every program that builds a search config, and the file it builds it in.
PROGRAMS = ("az_evaluate", "az_trainer", "az_coverage", "az_death_probe", "az_visual")

# Fields that describe the game or the paper's constants. Two programs disagreeing on one
# of these are searching different problems, whatever else they hold in common.
SHARED = (
    "discount",
    "step_reward",
    "steps_tiebreak_margin",
    "normalize_values",
    "death_cap_threshold",
    "root_noise_alpha",
)

# Fields a program is entitled to choose for itself.
LOCAL = (
    "simulations",
    "seed",
    "exploration",
    "exploration_epsilon",
    "root_noise_fraction",
    "trap_guard",
    "trap_report",
    "average_edges",
    "death_cap",
    "alias_report",
)


def constants():
    """Every az:: constant, by name, as written.

    Example:

        constants()["DISCOUNT"]      # '0.98f'
        constants()["STEP_REWARD"]   # '-0.02f'
    """
    text = io.open("src/az_parameters.h", encoding="utf-8-sig").read()
    found = re.findall(r"^constexpr \w+ (\w+)\s*=\s*([^;]+);", text, re.M)
    return {name: value.strip() for name, value in found}


def defaults():
    """What az::paperSearchDefaults() sets, by field.

    Example:

        defaults()["discount"]   # 'deriveDiscount(board)'
    """
    path = Path("src/search_defaults.h")
    if not path.exists():
        return {}
    text = io.open(path, encoding="utf-8-sig").read()
    # A bare name, or a call - the discount is derived from the board rather than being a
    # constant, so `config.discount = deriveDiscount(board);` has to match too. Without the
    # call form this reported every program as disagreeing about a field they all inherit.
    return dict(re.findall(r"config\.(\w+)\s*=\s*([\w:]+(?:\([^;]*\))?);", text))


def assignments(program):
    """What one program's source assigns to a config, by field.

    Reads every `<name>.<field> = <value>;` in the file, so it does not care whether the
    local variable is called config or search_config. A field assigned more than once
    keeps the last value, which is what the compiler would do.

    Example:

        assignments("az_visual")["discount"]   # 'az::DISCOUNT'
    """
    text = io.open(f"src/{program}.cpp", encoding="utf-8-sig").read()
    text = re.sub(r"^\s*//.*$", "", text, flags=re.M)
    found = re.findall(r"\b(?:config|search_config)\.(\w+)\s*=\s*([^;]+);", text)
    return {field: value.strip() for field, value in found}


def resolve(value, constant_values, default_fields):
    """One field's effective value, following az:: constants and the shared defaults.

    Example:

        resolve("az::DISCOUNT", {"DISCOUNT": "0.98f"}, {})   # '0.98f'
        resolve(None, {"DISCOUNT": "0.98f"}, {})             # 'UNSET (Config default)'
    """
    if value is None:
        return "UNSET (Config default)"
    name = value.replace("az::", "").strip()
    if name in constant_values:
        return constant_values[name]
    # A derived value is reported by function name alone. Its argument is whatever the
    # program calls its own board - settings.board in one, board in another - and comparing
    # those spellings would report a disagreement between two programs doing the same
    # thing. The limit is that this cannot tell deriveDiscount(board) from a call passing
    # the wrong variable; that is a job for reading, not for this.
    call = re.match(r"(\w+)\s*\(", name)
    if call:
        return call.group(1) + "(board)"
    return name


def main():
    """Prints the comparison and exits non-zero on any disagreement."""
    constant_values = constants()
    default_fields = defaults()
    inherited = {
        field: resolve("az::" + name, constant_values, {})
        for field, name in default_fields.items()
    }

    effective = {}
    for program in PROGRAMS:
        written = assignments(program)
        uses_defaults = "paperSearchDefaults" in io.open(
            f"src/{program}.cpp", encoding="utf-8-sig"
        ).read()
        row = {}
        for field in SHARED + LOCAL:
            if field in written:
                row[field] = resolve(written[field], constant_values, default_fields)
            elif uses_defaults and field in inherited:
                row[field] = inherited[field] + "  (inherited)"
            else:
                row[field] = resolve(None, constant_values, default_fields)
        effective[program] = row

    width = max(len(p) for p in PROGRAMS) + 2
    print("SHARED FIELDS - every program must agree with az_evaluate\n")
    header = "field".ljust(24) + "".join(p.ljust(width + 12) for p in PROGRAMS)
    print(header)
    print("-" * len(header))
    problems = []
    for field in SHARED:
        reference = effective["az_evaluate"][field].split("  ")[0]
        cells = []
        for program in PROGRAMS:
            value = effective[program][field]
            bare = value.split("  ")[0]
            mark = "" if bare == reference else "  <-- DIFFERS"
            if mark and program != "az_evaluate":
                problems.append((program, field, bare, reference))
            cells.append((value + mark).ljust(width + 12))
        print(field.ljust(24) + "".join(cells))

    print("\nLOCAL FIELDS - each program chooses these\n")
    header = "field".ljust(24) + "".join(p.ljust(width + 12) for p in PROGRAMS)
    print(header)
    print("-" * len(header))
    for field in LOCAL:
        print(field.ljust(24) + "".join(effective[p][field].ljust(width + 12) for p in PROGRAMS))

    print()
    if problems:
        for program, field, got, want in problems:
            print(f"DISAGREES  {program}.{field} = {got}, az_evaluate has {want}")
        print(f"\n{len(problems)} disagreements on shared fields")
        return 1
    print("every program agrees with az_evaluate on every shared field")
    return 0


sys.exit(main())
