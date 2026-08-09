#include <cassert>
#include <format>
#include <limits>
#include <stdexcept>

#include "az_parameters.h"

namespace az
{

int deriveStepLimit(int board)
{
    assert(board >= 2 && "deriveStepLimit on a board smaller than 2x2 - the parsers reject those");

    // board * board fits in long long for every int; 12 * board * board does not,
    // so the guard compares the area before the multiplication rather than after.
    const long long cells = static_cast<long long>(board) * board;
    constexpr long long largest_area = std::numeric_limits<int>::max() / STEPS_PER_CELL;
    if (cells > largest_area)
    {
        throw std::invalid_argument(
            std::format("board {} is too large: its step limit does not fit in an int", board));
    }
    return static_cast<int>(STEPS_PER_CELL * cells);
}

}  // namespace az
