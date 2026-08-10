#pragma once

#include "snake_logic.h"
#include "hamiltonian_cycle.h"

// Follows a Hamiltonian cycle over the grid, unconditionally.
//
// The snake visits every cell in a fixed order, so its body always lies in that
// order behind its head and it can never enclose itself. That makes the win a
// property of the construction rather than of the tuning: given a valid cycle,
// this agent fills the grid every time.
//
// No LibTorch and no raylib - a headless benchmark and a visual demo can both
// hold one of these.
class CycleAgent {
public:
    CycleAgent(int grid_width, int grid_height);

    // Throws if the ordering could not be built. A caller with no cycle has no
    // guarantee, and silently degrading to a greedy policy would hide that.
    void buildCycle();

    Direction chooseMove(const SnakeGame& game) const;

private:
    HamiltonianCycle cycle_;
    bool cycle_ready_{ false };
};
