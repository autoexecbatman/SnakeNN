#include <iostream>
#include <string>
#include <vector>

#include "hamiltonian_cycle.h"

// Validity test for the grid ordering that a winning agent depends on.
// No LibTorch, no raylib - this links against hamiltonian_cycle.cpp alone.
//
// A cycle-following snake cannot enclose itself only if the ordering is a true
// Hamiltonian cycle: every cell visited exactly once, every consecutive pair
// orthogonally adjacent, and the last cell adjacent to the first. Anything less
// is a Hamiltonian path, and the wrap from its end back to its start is a move
// the snake cannot make.

namespace {

int failures = 0;

void expect(bool condition, const std::string& description) {
    if (condition) {
        std::cout << "  PASS  " << description << std::endl;
    } else {
        std::cout << "  FAIL  " << description << std::endl;
        failures++;
    }
}

bool isAdjacent(const Position& first, const Position& second) {
    int horizontal = std::abs(first.x - second.x);
    int vertical = std::abs(first.y - second.y);
    return horizontal + vertical == 1;
}

}  // namespace

int main() {
    const int width = SnakeGame::GRID_WIDTH;
    const int height = SnakeGame::GRID_HEIGHT;
    const int cell_count = width * height;

    std::cout << "Hamiltonian cycle validity, " << width << "x" << height
              << " grid, " << cell_count << " cells" << std::endl;

    HamiltonianCycle cycle(width, height);
    bool generated = cycle.generateCycle();
    expect(generated, "generateCycle() reports success");
    if (!generated) {
        std::cout << "Cannot test an ordering that was not generated." << std::endl;
        return 1;
    }

    // Walk the ordering through the public interface, exactly as an agent would.
    const Position start(0, 0);
    std::vector<std::vector<bool>> visited(height, std::vector<bool>(width, false));

    Position current = start;
    int illegal_steps = 0;
    int revisits = 0;
    Position first_illegal_from(-1, -1);
    Position first_illegal_to(-1, -1);

    for (int step = 0; step < cell_count; step++) {
        if (current.x < 0 || current.x >= width || current.y < 0 || current.y >= height) {
            std::cout << "  Ordering left the grid at step " << step << std::endl;
            return 1;
        }
        if (visited[current.y][current.x]) {
            revisits++;
        }
        visited[current.y][current.x] = true;

        Position next = cycle.getNext(current);
        if (!isAdjacent(current, next)) {
            if (illegal_steps == 0) {
                first_illegal_from = current;
                first_illegal_to = next;
            }
            illegal_steps++;
        }
        current = next;
    }

    expect(illegal_steps == 0,
           "every one of the " + std::to_string(cell_count) +
           " steps moves to an orthogonally adjacent cell");
    if (illegal_steps > 0) {
        std::cout << "        " << illegal_steps << " illegal step(s); first was ("
                  << first_illegal_from.x << "," << first_illegal_from.y << ") -> ("
                  << first_illegal_to.x << "," << first_illegal_to.y << "), "
                  << (std::abs(first_illegal_from.x - first_illegal_to.x) +
                      std::abs(first_illegal_from.y - first_illegal_to.y))
                  << " cells apart" << std::endl;
    }

    expect(revisits == 0, "no cell is entered twice during one lap");

    int unvisited = 0;
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            if (!visited[row][col]) {
                unvisited++;
            }
        }
    }
    expect(unvisited == 0, "the lap covers every cell on the grid");
    if (unvisited > 0) {
        std::cout << "        " << unvisited << " cell(s) never visited" << std::endl;
    }

    expect(current == start,
           "the lap returns to its starting cell after exactly " +
           std::to_string(cell_count) + " steps");

    // Cycle indices must be a permutation of 0..cell_count-1 for cycle distance
    // to mean anything, and isShortcutSafe is built entirely on cycle distance.
    std::vector<bool> index_seen(cell_count, false);
    int bad_indices = 0;
    int duplicate_indices = 0;
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            int index = cycle.getCycleIndex(Position(col, row));
            if (index < 0 || index >= cell_count) {
                bad_indices++;
                continue;
            }
            if (index_seen[index]) {
                duplicate_indices++;
            }
            index_seen[index] = true;
        }
    }
    expect(bad_indices == 0, "every cell has a cycle index in range");
    expect(duplicate_indices == 0, "no two cells share a cycle index");

    std::cout << std::endl;
    if (failures == 0) {
        std::cout << "All checks passed - the ordering is a Hamiltonian cycle."
                  << std::endl;
        return 0;
    }
    std::cout << failures << " check(s) failed - the ordering is NOT a usable cycle."
              << std::endl;
    return 1;
}
