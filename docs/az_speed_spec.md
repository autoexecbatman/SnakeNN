# Spec: surpass 94.4 percent wins within 1,200 steps on 10x10

## Spec of the spec

**The question this contract must answer:** which changes reduce the number of steps
the agent needs to fill a 10x10 board, ranked by one number - the fraction of
held-out games won within 1,200 steps - and at what cost in wall-clock and games
played.

**What would make it the wrong contract**, each stated so it can be checked:

- It permits any part of the task definition to move after the first measurement.
  Board, step limit, starting length, what counts as a win, and the held-out seed
  band are fixed below and are not outputs of this work.
- It ranks a change by anything other than that win rate. Average score, mean steps
  and self-play returns are diagnostics; none of them is the objective, and a change
  that improves one while the win rate holds still has not worked.
- It reports a result without its cost. Every entry in the ledger carries wall-clock
  and games played, or the comparison between two approaches is unfounded.
- It treats matching the paper as the finish. The goal is to **exceed** 944/1000.
  Fidelity to Du et al. is a starting point, not a constraint - a change that departs
  from the paper and wins more games is a success, and it is recorded as a departure.

## The task, fixed before anything runs

| | value | note |
|---|---|---|
| board | 10x10 | |
| starting length | **1 segment** | the paper's is 2, so we need 99 apples to their 98. Deliberate, decided 2026-08-09, and not revisited. |
| step limit | **1,200** | the paper's, per game, and now also what training uses |
| simulations | 200 | the paper's search size |
| seeds | the reserved band, `seed_policy.h` | never trained on |
| batch | fixed across every compared run | it perturbs which apples the search plans against |

**Score conversion, once, so no table ever compares the wrong quantities.** Their
score is snake length, maximum 100. Ours is apples eaten, maximum 99. Ours plus one
is theirs. Their 98.227 average and our 98.219 at step limit 2400 are near-identical
numbers measuring different things: converted, ours is 99.2 of 100 and theirs is 98.2
of 100.

## Where we start

`az10_iter140.pt`, 64 held-out seeds, 200 simulations, batch 64:

| step limit | wins | died | timed out | mean steps of wins |
|---|---|---|---|---|
| 1,200 | 47/64 = 73.4% [60.9, 83.7] | 1 | 16 | 1093 |
| 2,400 | 63/64 = 98.4% [91.6, 100.0] | 1 | 0 | 1126 |

Against 944/1000: Fisher exact p = 2.9e-07, odds ratio 0.164.

**The agent fills the board and takes too long doing it.** One death in sixty-four at
either limit. Winning games take 907 to 1517 steps, median 1116, against a cap of
1200. Counted from the 2400 log, the win rate at each budget would be:

```
cap 1000 -> 10/64 (15.6%)      cap 1300 -> 57/64 (89.1%)
cap 1100 -> 30/64 (46.9%)      cap 1400 -> 62/64 (96.9%)
cap 1200 -> 47/64 (73.4%)      cap 1600 -> 63/64 (98.4%)
```

**A 12.8 percent reduction in steps reaches 94.4 percent. Surpassing it needs more.**

## What the paper does that we do not

Read in full 2026-08-09, arXiv:2211.09622, including the supplementary.

- **Trains at a 1,200 step limit.** Iterations 101-140 here ran at 2400 and were then
  graded at 1200. This mismatch is ours alone and it is free to remove.
- **3,000 gradient samples per game** - 30 batches of 100 states after every game,
  drawn from the last 2,000 games. Ours is 3,000 batches of 128 per iteration over
  256 games, so 1,500 per game.
- Seven binary planes, four absolute actions, max pool to 5x5 then flatten to 250.
  Ours: eight planes, three relative actions, average pool to 4x4.
- **Their agent cannot see the clock either.** No time, budget or hunger appears in
  their input. So clock awareness is not needed to *match* them - which is exactly
  why it is a candidate for *beating* them.

Two things we already do that the paper also does: forced single-action states are
collapsed in the search (`mcts.cpp:41`), and chance branches are explored with equal
frequency, which our sampling achieves in expectation.

## Stages, in dependency order

Each stage names its check. A stage that does not move the win rate is recorded as
such and the next one starts; a negative result is a result.

**1. The run ledger - done, 2026-08-09.** `docs/runs.tsv`, appended by the trainer and
the evaluator themselves: run id, start time, kind, command, outcome, seconds, games
played, samples trained. Nothing after this is compared without its cost.
*Check:* a run killed part-way still leaves a line, and the counts are what the run
did rather than what its settings asked for.

Both programs take **`--ledger <path>`, defaulting to `runs.tsv` in the launch
directory**. The default is deliberately not `docs/runs.tsv`: the launch directory is
`build/Release`, and git ignores `build/` and `*.log` alike, so a default that quietly
wrote there would lose exactly what this exists to keep. A run meant to leave a record
passes `--ledger ../../docs/runs.tsv`.

The visual demo is not wired in. It is watched, not measured, and a row per viewing
would bury the runs that produced numbers.

**2. Steps per apple.** The evaluator records the step count between apples. Answers
whether the waste is early, on an empty board, or late, unpicking a coiled body -
which decides whether the lever is the reward, the search or the state.
*Check:* the per-apple counts sum to the game's total steps, exactly, on every game.

**3. Train at the step limit we grade at.** No new code, only the flag. This is the
one deviation from the paper that is unambiguously ours.
*Check:* 64 held-out seeds at limit 1200, McNemar against iter140's own table on the
same seeds.

**4. The gradient budget.** Raise to 3,000 samples per game, matching the paper, and
measure it as its own change rather than bundled with the step limit.
*Check:* as above.

**5. Clock awareness - the first deliberate departure.** Give the agent the remaining
budget as an input plane or scalar, so it can trade risk against time near the
deadline. The paper has nothing like this, which is the point: it cannot be what
reaches 94.4 percent, and it is the most direct lever on the objective we have.
*Check:* as above, and it must beat whatever the step limit and gradient budget
leave standing, not merely beat iter140.

**6. Acceptance.** 200 held-out games, limit 1,200, 200 simulations, fixed batch.
Reported with a Clopper-Pearson interval and a Fisher test against 944/1000.
*Check:* the interval's lower bound exceeds 94.4 percent.

At n = 200 the interval is about 6.9 points wide and costs 13 minutes; n = 64 is the
iteration loop, 4 minutes, and is enough to detect that we are behind but not enough
to claim we are ahead.

## Not in scope

20x20, which comes after acceptance. The value-target squashing, `tanh(return / WIN_REWARD)`,
which remains untested and is a hypothesis about play quality - and play quality is
measured at one death in sixty-four.
