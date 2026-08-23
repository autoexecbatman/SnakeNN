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
//
// The win rests on this and on nothing else. CycleAgent has one decision - follow the
// ordering - so it is correct by construction rather than by tuning, and the construction
// is what this checks. Run it after any change to hamiltonian_cycle.cpp.
//
// It is a live guard, not a formality: it caught the original defect, a boustrophedon path
// whose wrap edge jumped 19 cells. Every check below passed on that ordering except the
// adjacency one, which is why the failure branch prints the offending pair and how far
// apart they are rather than only that a step was illegal.
//
// Run it - seconds, and buildable without CMake or vcpkg from a shell with MSVC on PATH:
//
//     cmake --build build --config Release --target CycleTest
//     build\Release\CycleTest.exe
//
//     cl /std:c++20 /EHsc src\cycle_test.cpp src\hamiltonian_cycle.cpp
//
// It prints one line per property and returns non-zero if any failed:
//
//     Hamiltonian cycle validity, 20x20 grid, 400 cells
//       PASS  generateCycle() reports success
//       PASS  every one of the 400 steps moves to an orthogonally adjacent cell
//       PASS  no cell is entered twice during one lap
//       PASS  the lap covers every cell on the grid
//       PASS  the lap returns to its starting cell after exactly 400 steps
//       PASS  every cell has a cycle index in range
//       PASS  no two cells share a cycle index
//
//     All checks passed - the ordering is a Hamiltonian cycle.
//
// The MST progress lines above that come from hamiltonian_cycle.cpp, not from here.
//
// The last two checks are not decoration. Cycle indices must be a permutation of
// 0..cell_count-1 for cycle distance to mean anything, and isShortcutSafe is built entirely
// on cycle distance - so a duplicate index would make the shortcut layer unsound in a way
// no adjacency check would notice.

namespace {

// Checks that did not hold. main prints the count and returns 1 when it is non-zero.
int failures = 0;

// Reports one property and counts a failure.
//
//     expect(revisits == 0, "no cell is entered twice during one lap");
//     //   PASS  no cell is entered twice during one lap
//
// Prints on success as well as failure, because the point of this test is to say what was
// verified. A silent pass would leave a reader unable to tell a checked property from one
// nobody wrote.
void expect(bool condition, const std::string& description) {
    if (condition) {
        std::cout << "  PASS  " << description << std::endl;
    } else {
        std::cout << "  FAIL  " << description << std::endl;
        failures++;
    }
}

// Whether two cells share an edge - the only move a snake can make.
//
//     isAdjacent(Position(3, 4), Position(3, 5));   // true
//     isAdjacent(Position(3, 4), Position(4, 5));   // false, diagonal
//     isAdjacent(Position(3, 4), Position(3, 4));   // false, a cell is not adjacent to
//                                                   // itself
//
// Manhattan distance of exactly 1, so diagonals and standing still both fail. Computed here
// rather than taken from hamiltonian_cycle.cpp on purpose: a test that borrowed the
// definition it is checking would agree with a wrong one.
bool isAdjacent(const Position& first, const Position& second) {
    int horizontal = std::abs(first.x - second.x);
    int vertical = std::abs(first.y - second.y);
    return horizontal + vertical == 1;
}

}  // namespace

// Generates the ordering, walks one full lap through the public interface, and checks the
// four cycle properties plus the two index properties. Returns 1 on any failure, 0 on all
// passing, and 1 without testing anything if generateCycle fails or the walk leaves the
// grid - an ordering that was never built is not a passing one.
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
