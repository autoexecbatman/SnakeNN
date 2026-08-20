# snakeNN

Reinforcement-learning experiments on Snake, in C++20 with LibTorch and raylib.

This is a bench of experiment executables rather than one application: a shared game
library, a shared network, and roughly sixty `main()` programs that train, evaluate or
watch an agent. Two things live here at once - a solved hand-written agent that fills a
20x20 board every time, and an AlphaZero stack still being taught to do the same by
learning.

## The two halves

**The board is already beaten, without a network.** `CycleAgent` follows a Hamiltonian
cycle over the grid, so the snake visits every cell in a fixed order and cannot enclose
itself. The win follows from the construction rather than from tuning. It depends on
neither LibTorch nor raylib, which is what lets the headless benchmark and the visual
demo run identical policy code.

**The open problem is winning by learning**, and doing it inside a step limit the cycle
cannot meet - its win time grows as the fourth power of the board side, so under the
step cap it wins nothing. That is what `src/` is for.

## Layout

    src/          what the project still measures with
    legacy/       one main() per file, each the record of one past run
    docs/         mutation sweeps, the static-analysis run, and the run ledger

`src/` holds the winning path and the AlphaZero stack:

| file | what it is |
| --- | --- |
| `snake_logic.{h,cpp}` | the game, 20x20 fixed at compile time |
| `hamiltonian_cycle.{h,cpp}` | the grid ordering the win rests on |
| `cycle_agent.{h,cpp}` | the winning policy, one decision: follow the cycle |
| `snake_env.{h,cpp}` | the training environment - runtime board size, explicit seed |
| `vector_env.{h,cpp}` | a block of games stepped together into one batch |
| `az_network.{h,cpp}` | convolutional trunk, four heads, board size irrelevant to the weights |
| `mcts.{h,cpp}` | batched search over the real simulator |
| `selfplay.{h,cpp}` | games to training records |
| `az_trainer.cpp` | the training loop |
| `az_evaluate.cpp` | scores a checkpoint on held-out seeds |

`legacy/` is not a junk drawer and is not refactored: each file records one experiment as
it was run, and rewriting it would destroy the record.

## Building

Windows only - Visual Studio 2022, vcpkg for raylib, and LibTorch wired by absolute path
in `CMakeLists.txt`.

    cmake -S . -B build -G "Visual Studio 17 2022" -A x64 ^
      -DCMAKE_TOOLCHAIN_FILE=E:/dev/vcpkg/scripts/buildsystems/vcpkg.cmake
    cmake --build build --config Release --target Benchmark

**Always build one target.** `ALL_BUILD` compiles every executable and copies the CUDA
DLL set into each output directory.

`CycleTest` and `Benchmark` link neither LibTorch nor raylib and build in seconds. They
are the two to reach for when checking whether something still works, and they can be
built with `cl` directly, without CMake or vcpkg at all.

## Running

    build\Release\Benchmark.exe            win rate over seeded games, headless
    build\Release\PerfectSnakeAI.exe       the same policy, drawn with raylib
    build_perfect_ai.bat                   configures, builds and runs the above
    build\Release\AlphaZeroTrainer.exe     the learning agent, --resume to continue
    build\Release\AlphaZeroEvaluate.exe    scores a checkpoint on held-out seeds

## Tests

There is no test framework - each test is a plain executable returning non-zero on
failure, registered with CTest.

    cd build && ctest -C Release

That command is also the answer to "how many are there" - a number written here would be
wrong by the next commit, which is how the rest of this file came to describe a project
that no longer existed. None of them needs LibTorch or raylib except the network's own,
so the suite runs without the CUDA toolchain.

Units carry a mutation sweep in `docs/` alongside their tests, because a suite that has
never failed has not been tested. A sweep compiles each mutant against a copy of the
sources and reports any the tests cannot see.

## Reading the numbers

**Win rate is the metric.** A high average score is not partial progress toward a win.

**Never quote a self-play score as the agent's performance.** Self-play injects noise at
the root and samples early moves at temperature, deliberately - those numbers describe
the exploration policy. `AlphaZeroEvaluate` turns that off and uses seeds outside the
training range.

**Distrust the `percent` in a `.bin` filename.** Those come from the DQN era and count
the fraction of games scoring at least one food, so `snake_research_best_100percent.bin`
means "eats one apple of 399 every time". None of them is evidence about winning.

Every run appends what it cost to `docs/runs.tsv` - wall clock, games, samples - so a
result can be compared and budgeted rather than guessed at afterwards.

## Grounding

The learned agent follows Du, Gemp, Wu and Wu 2022, "AlphaSnake"
([arXiv:2211.09622](https://arxiv.org/abs/2211.09622)): 944 wins in 1000 on a 10x10 board
with 200 simulations per move. The hyperparameters here are theirs, deliberately.

20x20 is past that paper - the largest board in its comparison reaches 14 percent average
score. A win there would be a new result rather than a reproduction.
