#include <stdexcept>

#include "cycle_agent.h"

CycleAgent::CycleAgent(int grid_width, int grid_height)
    : cycle_(grid_width, grid_height), cycle_ready_(false) {
}

void CycleAgent::buildCycle() {
    if (!cycle_.generateCycle()) {
        throw std::runtime_error("Hamiltonian cycle generation failed - no win guarantee without it");
    }
    cycle_ready_ = true;
}

Direction CycleAgent::chooseMove(const SnakeGame& game) const {
    if (!cycle_ready_) {
        throw std::logic_error("chooseMove called before buildCycle");
    }

    const Position& head = game.getSnakeBody()[0];
    Position next = cycle_.getNext(head);

    if (next.x == head.x + 1) {
        return Direction::RIGHT;
    }
    if (next.x == head.x - 1) {
        return Direction::LEFT;
    }
    if (next.y == head.y + 1) {
        return Direction::DOWN;
    }
    if (next.y == head.y - 1) {
        return Direction::UP;
    }

    // getNext returns its argument for a cell outside the ordering, and any
    // other gap means the ordering is not a cycle. Either way the guarantee is
    // gone, so fail here rather than at a collision several moves later.
    throw std::logic_error("cycle successor is not adjacent to the head");
}
