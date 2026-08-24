// Properties of the biased replay draw, derived from the contract in replay_sampling.h.
//
// What it guards. Uniform sampling spends a batch on positions where nothing is decided;
// this unit raises the share of positions a move can lose from. The properties below pin
// the three things that make it safe to use: an unbiased pick costs exactly one draw, a
// biased pick stops at the first decisive record, and a window holding none of them still
// returns something after a bounded number of draws rather than looping forever.
//
// Every expected value is derived by hand from the contract, never read off the
// implementation. The draw is a fixed list, so each case is deterministic - no generator,
// no seed, no LibTorch.
//
// Run it:
//
//     cmake --build build --config Release --target ReplaySamplingTest
//     build\Release\ReplaySamplingTest.exe
//
// Prints one line per property, and returns non-zero with a count if any failed.

#include <cstddef>
#include <format>
#include <iostream>
#include <string>
#include <vector>

#include "replay_sampling.h"

namespace
{

// Properties that did not hold. main prints the count and returns 1 when it is non-zero.
int failures = 0;

// Reports one property and counts a failure.
void check(bool held, const std::string& property, const std::string& detail)
{
    std::cout << (held ? "[PASS] " : "[FAIL] ") << property << "  " << detail << std::endl;
    if (!held)
    {
        failures++;
    }
}

// A draw that walks a fixed list, counting how many times it was asked.
//
//     Draws draws({ 4, 7, 2 });
//     draws.next();      // 4
//     draws.used;        // 1
//
// Past the end it repeats the last entry, so a case that over-draws fails on the count
// rather than on an out-of-range read.
struct Draws
{
    std::vector<std::size_t> values;
    std::size_t used{ 0 };

    std::size_t next()
    {
        const std::size_t value = values[std::min(used, values.size() - 1)];
        used++;
        return value;
    }
};

}  // namespace

int main()
{
    // 1. Without the bias the first candidate is taken, and nothing else is drawn.
    //    Cost matters: this is the common path, run once per item of every batch.
    {
        Draws draws({ 4, 7, 2 });
        const std::size_t picked = sampling::pickBiased([&] { return draws.next(); },
                                                        [](std::size_t) { return true; }, false, 4);
        check(picked == 4 && draws.used == 1, "an unbiased pick is the first draw, and costs one",
              std::format("picked {} after {} draws", picked, draws.used));
    }

    // 2. Biased, and the first candidate already decisive: it is kept without a second
    //    draw. The bias must not cost anything when the first answer is the right one.
    {
        Draws draws({ 5, 9 });
        const std::size_t picked = sampling::pickBiased(
            [&] { return draws.next(); }, [](std::size_t index) { return index == 5; }, true, 4);
        check(picked == 5 && draws.used == 1, "a decisive first draw is kept immediately",
              std::format("picked {} after {} draws", picked, draws.used));
    }

    // 3. Biased, with the decisive record third: the two non-decisive candidates are
    //    skipped and the third is returned, on the third draw exactly.
    {
        Draws draws({ 1, 2, 3, 4 });
        const std::size_t picked = sampling::pickBiased(
            [&] { return draws.next(); }, [](std::size_t index) { return index == 3; }, true, 4);
        check(picked == 3 && draws.used == 3, "a biased pick stops at the first decisive draw",
              std::format("picked {} after {} draws", picked, draws.used));
    }

    // 4. Biased, and nothing in the window is decisive: the last candidate is returned
    //    after exactly `tries` draws. This is the property that keeps a batch full - an
    //    early window holds no decisive record at all, and a pick that refused to return
    //    one would either loop or leave the batch short.
    {
        Draws draws({ 8, 6, 7 });
        const std::size_t picked = sampling::pickBiased([&] { return draws.next(); },
                                                        [](std::size_t) { return false; }, true, 3);
        check(picked == 7 && draws.used == 3, "with no decisive record the last draw is used",
              std::format("picked {} after {} draws", picked, draws.used));
    }

    // 5. One try is the floor and behaves like no bias at all: a single draw, returned
    //    whatever it holds. Below this there is no index to return.
    {
        Draws draws({ 11, 12 });
        const std::size_t picked = sampling::pickBiased([&] { return draws.next(); },
                                                        [](std::size_t) { return false; }, true, 1);
        check(picked == 11 && draws.used == 1, "one try is a single draw",
              std::format("picked {} after {} draws", picked, draws.used));
    }

    if (failures == 0)
    {
        std::cout << "all properties held" << std::endl;
        return 0;
    }
    std::cout << std::format("{} properties failed\n", failures);
    return 1;
}
