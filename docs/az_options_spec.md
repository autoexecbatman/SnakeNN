# Spec: give az_evaluate and az_visual a tested boundary

## Spec of the spec

What question must this contract answer: **which parts of these two programs can be
made reachable by a test, and what must provably not change about the numbers they
print.**

What would make it the wrong contract: if it requires changing what the evaluator
measures. A hardening pass that moves a win rate is a behaviour change wearing a
refactor's clothes.

## Why these two files

`az_trainer.cpp` was given this treatment on 2026-08-08 (commit ee4a728). The two
programs that produce every number this project quotes did not get it. Measured
2026-08-09 by grep, not recalled:

- zero `TORCH_CHECK` and zero `assert` between them - no check of any kind
- **15** unvalidated `std::stoi` / `std::stoul` calls. An earlier reading of 13 came
  from a grep whose pattern also matched `step_limit` lines and was counted by eye;
  15 is what the measure counts.
- `step_limit == 0` as a sentinel for "not given", in both
- `12 * board * board` written out in both, a third copy of a constant that
  `trainer::STEPS_PER_CELL` already owns
- the search hyperparameters `0.5`, `0.98`, `0.3` written as literals in both, when
  `trainer_options.h` names them once and the project's own rule is to deviate from
  the paper knowingly or not at all
- an unknown flag warns on stderr and the program continues with the default
- a flag in final position with no value is dropped in silence (`index + 1 < argc`)

Each of those is a defect that silently produces a plausible number, which is the
class this project has already been burned by twice.

## What gets built

Three translation units, none of which links LibTorch or raylib, so assertions in
them are reachable in a Debug build:

- **`src/flag_parser.{h,cpp}`** - `parseWholeInt`, `parseWholeUnsigned`,
  `requireAtLeast`, and a reader that walks the argument vector and throws on an
  unknown flag or a missing value. Extracted from `trainer_options.cpp`, which is the
  **third** use of this logic and therefore the first one where extraction is not
  speculative.
- **`src/eval_options.{h,cpp}`** - `evaluation::Settings` and `visual::Settings` with
  their parsers, deriving the step limit from `trainer::STEPS_PER_CELL` and taking
  the search constants from `trainer_options.h`.
- **`src/eval_options_test.cpp`** - CTest name `eval_options`.

`az_evaluate.cpp` and `az_visual.cpp` keep only what needs LibTorch or raylib.

## Properties, each with the check that would catch it being false

1. **Every rejected input is rejected before any work is done.** Check: the test
   drives each of `--games 0`, `--board 1`, `--simulations 0`, `--batch 0`,
   `--speed 0`, `--board 10x10`, `--bord 12`, and a trailing `--board` with no
   value, and requires a throw naming the flag.
2. **The step limit has no sentinel.** Check: `std::optional`, and a test that
   `--step-limit` absent derives `STEPS_PER_CELL * board * board` while
   `--step-limit 1200` returns 1200. There is no value of the field meaning "unset".
3. **One owner per constant.** Check: `static_assert` that the derived limit equals
   `trainer::STEPS_PER_CELL * board * board`, and grep finding no literal `12 *` and
   no `0.98`/`0.5`/`0.3` search literal left in either az program.
4. **The measurement does not move.** Check: `AlphaZeroEvaluate` on
   `az10_iter123.pt`, 64 games, seed offset 0, step limit 1200, must report the same
   36 wins the first 64 games of the 2026-08-08 run reported. This is the property
   that matters; the rest is tidying.
5. **Paired comparison becomes possible.** The evaluator prints one line per game -
   seed, outcome, score, steps - and records `--batch` in its header. Check: the
   output has `games` such lines, and McNemar can be run from two logs.
   `statsmodels` is installed, verified 2026-08-09.

Property 4 is the acceptance test. Properties 1-3 are what makes 4 trustworthy next
time rather than this time.

## Progress

- **`flags::parseWholeInt` - done, 2026-08-09.** `src/flag_parser.{h,cpp}`,
  `src/flag_parser_test.cpp`, CTest target `flag_parser`. Red step 16 of 17
  assertions; mutation 6 of 6 killed; `clang-format` clean; `ctest -C Release` 11/11.
- **`flags::parseWholeUnsigned`, `requireAtLeast`, `readFlags` - done, 2026-08-09.**
  Mutation 24 of 24 killed.
- **`evaluation::Settings` and `evaluation::parseArguments` - done, 2026-08-09.**
  `src/eval_options.{h,cpp}`, `src/eval_options_test.cpp`, CTest target
  `eval_options`, CMake target `EvalOptionsTest`. Red step 36 of 44 assertions;
  mutation 26 of 26 killed after one survivor was fixed; four assertions driven
  past their bound and watched to abort with exit code 3; `clang-format` clean;
  Debug and Release both pass; the analyzer reports nothing in the new files.
  `az_evaluate.cpp` now parses through it.
- **Property 4 holds, measured 2026-08-09.** `AlphaZeroEvaluate` rebuilt in Release
  and run on `az10_iter123.pt`, 64 games, seed offset 0, step limit 1200, batch 64,
  200 simulations: **36 wins**, the number at `eval123_limit1200.log:5`. Score mean
  90.188, 9 died, 19 timed out, 201.55s, 12,245,942 evaluations. Log:
  `build/Release/acceptance_iter123_64games.log`. Exit code 1, as documented.
- **Property 1 holds for the evaluator**: all four bad inputs now stop before the
  checkpoint load, where all four reached it before.
- **`visual::Settings` and `visual::parseArguments` - done, 2026-08-09.** Same file,
  second namespace; `az_visual.cpp` parses through it. Red step 32 of 39 assertions;
  mutation 43 of 43 across both namespaces; four more assertions driven to abort;
  `clang-format` clean, analyzer clean, Debug and Release both pass.
  `AlphaZeroVisual` rebuilt in Release, and five bad command lines checked against
  the real binary - `--bord 12`, `--board 10x10`, `--speed 0`, a trailing `--board`,
  and a missing checkpoint - each exits 2 with its own diagnosis before a window
  opens or a checkpoint is read.
- **Properties 1, 2 and 3 now pass**: the measure reads zero `std::stoi`, zero
  `step_limit == 0`, zero `12 * settings.board` and zero search literals across both
  programs, and every bad input stops before the checkpoint load.
- Property 5 - per-game output lines and the batch recorded in the header - is the
  only part of this spec not started, and it is the two remaining failures.

### Two decisions taken in the visual, which the evaluator did not force

- **`visual::Settings` is its own type, not a base class shared with
  `evaluation::Settings`**, although six of eight fields match. Two programs, not
  one program twice: this one has a frame rate and a single absolute seed, that one
  has a game count, a batch and an offset into a reserved band. Two uses is not the
  third repetition that justifies extraction.
- **`--seed` here stays absolute and keeps its default of 900000, which is a
  training seed** (`seed_policy.h`). So the demo's default shows a game the agent
  may have learned, while the file's own comment says what is on screen is what the
  win-rate number describes. Preserved deliberately: changing what the demo shows is
  a decision about the demo, not a hardening pass. The test asserts the value, so
  the change cannot happen quietly.
- **The board's upper bound has an owner: `az::LARGEST_BOARD`** (`az_parameters.h`),
  with two `static_assert`s pinning it as the largest board whose step limit fits
  in an int. `deriveStepLimit` compares against it rather than recomputing
  `INT_MAX / STEPS_PER_CELL`.
### The prover, added 2026-08-09

`docs/prove_arithmetic.py` - Z3 over the integer arithmetic no test can enumerate.
Eight claims, each with a mutant that must answer the other way, so a claim that
cannot fire is reported rather than counted as a pass. It proves the board bound is
both sound and not over-strict, that `evaluation::Settings::cellCount` cannot
overflow because of that bound, and that the seed-band check accepts exactly the
runs that stay inside the reserved range.

It also found two expressions that do overflow, both in the trainer, both left as
found: `trainer::Settings::cellCount()` computes `board * board` in an int with no
upper bound on `board` (`trainer_options.h:53`, witness board 46341), and
`lastIteration()` adds two unbounded ints (`:59`, witness start_iteration
2147483647 with 2 iterations). Undefined behaviour on values the trainer's parser
accepts today.

### What the second mutation run found

`seed_signed` survived: replacing `parseWholeUnsigned` with a cast of
`parseWholeInt` for `--seed` still rejected `--seed -1`, because -1 casts to a value
the seed-band check refuses anyway. Right rejection, wrong diagnosis - the operator
would be told their seed range overran when what they typed was a negative number.
This is the same shape as `drop_range_branch` in the first run, and the same fix:
an assertion that reads the message, not more code.

### Interface decisions taken during it, which bind the rest

- **`std::string_view`, not `const std::string&`, for text parameters.** Callers pass
  literals and `argv` entries; the reference form built a temporary `std::string` per
  call to be read and discarded. A view also suits `std::from_chars`, which reads a
  range and never wants a null terminator. The cost is a lifetime caveat, stated at
  the interface: the view must outlive the call, and nothing is stored.
- **`std::string`, not `const char*`, everywhere it is a choice.** The only pointers
  left are the two `std::from_chars` requires, written inline on the call so that no
  named pointer exists in the file.
- **Consequently `parseArguments` takes `std::span<const std::string>`**, not
  `std::span<const char* const>` as this spec first proposed, and `main` converts
  `argv` once at the boundary. This contradicts the signature currently in
  `trainer_options.h`; that copy goes when `trainer_options` starts calling
  `flags::parseWholeInt`, and until then the tree holds two conventions.
- **Include order: angle brackets, blank line, quoted.** `SortIncludes: false`, so it
  is maintained by hand. New files only; the older ones are not being rewritten.

### What the first function's mutation run found

`drop_range_branch` survived the first pass. Removing the out-of-range branch left
the fallthrough throwing `std::invalid_argument` naming the flag, which satisfied
every assertion in the file **with the wrong diagnosis** - "not a number" for a value
that is a perfectly good number too large to hold. The header promises the two are
reported apart; nothing checked the promise. Fixed by reading the message.

**The mutation script also caught its own staleness.** Editing the expression made two
patterns stop matching, and it printed `NOT APPLIED - pattern absent, mutation is
vacuous` rather than scoring them as kills. Without that guard the run reads
"killed 4 of 6" with two mutants never applied - and a mutation run that silently
tests nothing looks identical to one that found survivors.

## The measure

`docs/az_options_measure.ps1`. Run it before the change and after; add
`-WithAcceptance` for property 4, which needs the GPU and about five minutes.

**Run against the unchanged tree on 2026-08-09 it reports 10 failures**, which is the
red step for the whole spec - every check has been seen to fire, so none of them is
decoration. What it read:

| property | check | before |
|---|---|---|
| 1 | four bad inputs reach the checkpoint load | 4 of 4 reach it |
| 2 | `step_limit == 0` occurrences | 2 |
| 3 | literal `12 * settings.board` | 2 |
| 3 | hardcoded search hyperparameters | 6 |
| 3 | `std::stoi` / `std::stoul` calls | 15 |
| 5 | per-game output lines | none |
| 5 | batch recorded in the header | no |

Two of the measure's own checks were defective on first run and were fixed before
this table was produced. One compared a reading to itself and printed PASS whatever
the source said. The other scored the misspelled-flag case on whether a warning
appeared, which it does - the program prints `unknown flag: --bord` and then trains
anyway, so reading the warning alone would have scored the defect as already fixed.
The check that survived asks the only question that matters: **did the program get as
far as loading the checkpoint.**

## What the measure found that the spec did not predict

`--batch` is recoverable from the existing logs after all, from the stride of the
progress lines. `eval123_limit1200.log` prints at 64, 128, 192, 200, so that run used
`--batch 64`. `eval110_limit1200.log` prints once at 200/200, so its batch was at
least 200. **The two runs in the headline iter110-to-iter123 comparison used different
batch sizes**, and batch perturbs which apples the search plans against
(`mcts.cpp:469`). The confound is real rather than hypothetical, and the p = 0.028 at
1200 steps is not a clean between-checkpoint comparison.

## What this spec does not cover

- The `--batch` confound itself (`mcts.cpp:469` reseeds from one shared generator in
  lockstep, so batch size perturbs which apples the search plans against). Recording
  the batch makes it visible; it does not fix it.
- `AlphaZeroEvaluate` returning 1 unless every game is won. Left as is - it is
  documented, and changing an exit code silently breaks whatever reads it.
- The snake's starting length, 1 segment here against the paper's 2.

## Assertions: which invariant, in which translation unit, and how it is proved live

The split is not stylistic. **A debug binary linked against this project's LibTorch
dies of an access violation before `main`, and release defines `NDEBUG`, so every
`assert` in a Torch-linked file here is dead in both configurations.**

| file | links | check to use |
|---|---|---|
| `flag_parser.cpp`, `eval_options.cpp` | nothing | `assert` |
| `az_evaluate.cpp` | LibTorch | `TORCH_CHECK` |
| `az_visual.cpp` | LibTorch + raylib | `TORCH_CHECK` |

Never both in one file. Mixing them was the defect found in `az_trainer.cpp` on
2026-08-08 and it is the single easiest thing to get wrong here.

**Assert an invariant, check a boundary.** The argument vector is a boundary - it
carries whatever the operator typed, so every rejection there is a throw naming the
flag, not an assert. Everything downstream of `parseArguments` is an invariant: by
then the board is at least 2 and the game count at least 1, so `stepLimit()` asserts
them rather than re-testing them. A defensive `if` there would hide a wiring fault
instead of locating it.

**Check:** every assertion is driven past its bound by the test and the process
aborts; and a mutation of the asserted expression is killed. `TORCH_CHECK`s are
covered by property 1 of the measure, which runs the binary.

## Tests: what makes one real here

- **Red first.** Every test runs and fails on its own assertion before the
  implementation exists. A failing suite is not a red step; a failing *assertion* is,
  and each line of the red run is read individually.
- **Expected values come from the spec, not from what the code returns.** The derived
  step limit is checked against `STEPS_PER_CELL * board * board` written out, not
  against whatever `stepLimit()` produces.
- **Each test asserts against the alternatives, not against a constant.** All-zeros
  satisfies most thresholds and a fixed default matches an argmax one time in three -
  both have shipped in this repository.
- **A flag test requires exactly one field to change.** `Settings` prints one field
  per line and the test compares against the default rendering, so a parser that
  writes the right value into the wrong field fails. This caught nothing in
  `trainer_options` and is kept because it is the only check that can.
- **Mutation is the proof of non-vacuity.** Target: every mutant killed, and the count
  reported. The `trainer_options` pass killed 13 of 13, two of them by assertions
  rather than by test expectations, which is what made those two assertions real.
- **MSVC resolves a quoted include from the including file's own directory first**, so
  a mutant header placed elsewhere is silently ignored and reports as surviving. Copy
  the test next to the mutant.

## Standards

**ISO/IEC 14882:2020 (C++20)**, `CXX_STANDARD 20` on every target, MSVC only.

**C++ Core Guidelines**, rule identifiers verified against the published document on
2026-08-09 rather than recalled. The ones this change is actually judged against:

- **I.6** *Prefer `Expects()` for expressing preconditions* and **I.8** *Prefer
  `Ensures()` for expressing postconditions* - satisfied in spirit with `assert` and
  `TORCH_CHECK`; the guideline support library is not a dependency here and adding one
  for two macros is not worth it.
- **I.13** *Do not pass an array as a single pointer* - `parseArguments` takes
  `std::span<const char* const>`, which is what lets it be called from a test with a
  vector instead of from `main` with `argc`/`argv`.
- **F.21** *To return multiple "out" values, prefer returning a struct* - the parser
  returns `Settings`, not an out-parameter and not a `std::pair`.
- **ES.45** *Avoid "magic constants"; use symbolic constants* - this is the whole of
  property 3. `12`, `0.98`, `0.5` and `0.3` are the magic constants in question.
- **ES.46** *Avoid narrowing conversions* - `--seed` parses to `unsigned int` through
  a checked path rather than a `static_cast` of a `std::stoul` result.
- **C.46** *By default, declare single-argument constructors `explicit`*.
- **C.80/C.81** *Use `=default` / `=delete` to be explicit about default operations* -
  and note the repository's own reading, which is that deleting defines no behaviour,
  so what to delete is a separate decision from who writes the destructor.

**Google C++ Style Guide** via `.clang-format` (`BasedOnStyle: Google`), with four
deliberate house deviations that are already committed and must not be "fixed" back:
Allman braces, 4-space indent, 100-column limit, and `InsertBraces: true`. Naming
follows the existing files - `lowerCamelCase` functions, `snake_case` members - which
is not Google's convention and is consistent within this repository, which matters
more.

**Check:** `clang-format --dry-run` on the touched files reports nothing, and the
measure's property-3 counts go to zero.

## Design patterns

Almost none apply, and saying which were considered and rejected is the useful half.

**In use, and earning it:** `Evaluator` (`evaluator.h`) is a Strategy - an abstract
interface the search depends on instead of depending on LibTorch, which is precisely
what makes search correctness testable without CUDA. `Settings` is a value object.
Neither is being changed.

**Rejected, with reasons, so they are not proposed again:**

- **A `Program` base class over the three `main()`s.** They share argument parsing and
  nothing else - different lifetimes, different dependencies, one of them a render
  loop. A shared base would couple three unrelated programs to force-fit the one thing
  a free function already shares.
- **A Builder for `Settings`.** It is an aggregate with default member initialisers;
  designated initialisers already do this in the language.
- **A table of `std::string` to handler for the flags.** It looks cleaner and it moves
  the flag set from compile time to run time, which is the wrong direction: the
  current `if`/`else if` chain is checkable by reading and the table is not.
- **A Factory for the parsers.** There is one construction path per program.
- **Singleton, in any form.** Banned outright by the house rules; the previous
  generation of this codebase has a function-local `static` in `getReward()` that is
  shared across every game instance in the process, which is exactly the bug the ban
  exists to prevent.

The pattern that would be a mistake to add is the one that makes `flag_parser` general.
It has three callers, all in this repository, all parsing whole numbers and strings.
