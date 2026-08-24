#pragma once

#include <cstddef>
#include <functional>

// Choosing which stored position a training batch trains on.
//
// Why this exists. A 20x20 iteration seals about 114,000 positions, and in nearly all of
// them every move is safe - the board is open and nothing is being decided. The positions
// that decide a game, where one of the three moves ends it, are tens per game. Drawing
// uniformly spends almost the whole batch on positions that teach nothing about the only
// thing the agent gets wrong, which is walking into a space it cannot leave.
//
// What this does. It draws repeatedly and keeps the first decisive position it finds,
// which raises their share of a batch without holding an index of them. An index would
// have to survive eviction from the front of the replay window, where every absolute
// position shifts; rejection needs nothing but the draw itself.
//
// Randomness stays outside. The caller supplies the draw and decides whether this pick
// should prefer a decisive record, so every case here is reachable in a test without a
// generator, a seed or LibTorch.
//
// Usage:
//
//     std::size_t cursor = 0;                                  // a test's fake draw
//     const std::vector<std::size_t> draws{ 4, 7, 2 };
//     auto draw = [&] { return draws[cursor++]; };
//     auto decisive = [&](std::size_t index) { return index == 2; };
//
//     sampling::pickBiased(draw, decisive, false, 4);   // 4  - the first draw, no bias
//     sampling::pickBiased(draw, decisive, true, 4);    // 2  - skipped 7, kept 2
//
namespace sampling
{

// One index to train on, biased toward positions where a move can lose.
//
//     const std::size_t index = sampling::pickBiased(draw, isDecisive, prefer, 4);
//
// `draw` returns a candidate index; it is called once when `prefer_decisive` is false and
// between one and `tries` times when it is true. `is_decisive` answers whether that index
// holds a position some move loses from. `tries` is the ceiling on draws, so a window
// holding no decisive record costs a bounded number of draws rather than looping - the
// last candidate is returned in that case, which keeps the batch full.
//
// Asserts `tries` is at least 1. A zero would mean returning an index nothing produced.
std::size_t pickBiased(const std::function<std::size_t()>& draw,
                       const std::function<bool(std::size_t)>& is_decisive, bool prefer_decisive,
                       int tries);

}  // namespace sampling
