// Implementation of deriveStepLimit. The constants, and how to use them, are in
// az_parameters.h.

#include <cassert>
#include <format>
#include <stdexcept>

#include "az_parameters.h"

namespace az
{

int deriveStepLimit(int board)
{
    assert(board >= 2 && "deriveStepLimit on a board smaller than 2x2 - the parsers reject those");

    if (board > LARGEST_BOARD)
    {
        throw std::invalid_argument(
            std::format("board {} is too large: its step limit does not fit in an int", board));
    }
    // Formed in long long so the multiplication cannot overflow; the bound just
    // tested is what makes the cast back exact.
    return static_cast<int>(static_cast<long long>(STEPS_PER_CELL) * board * board);
}

}  // namespace az
