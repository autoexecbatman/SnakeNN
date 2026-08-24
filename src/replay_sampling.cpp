// Implementation of the biased replay draw. The contract is in replay_sampling.h.

#include <cassert>

#include "replay_sampling.h"

namespace sampling
{

std::size_t pickBiased(const std::function<std::size_t()>& draw,
                       const std::function<bool(std::size_t)>& is_decisive, bool prefer_decisive,
                       int tries)
{
    assert(tries >= 1 && "pickBiased with no tries has no index to return");
    // Stub: always the first candidate, whatever was asked for.
    // One draw is the whole job when the caller is not asking for the bias, and that is
    // the common path - it runs once per item of every batch.
    if (!prefer_decisive)
    {
        return draw();
    }

    // Draw until a decisive record turns up. The last candidate is kept when none does,
    // so an early window holding no decisive record still fills the batch.
    std::size_t candidate = draw();
    for (int attempt = 1; attempt < tries; attempt++)
    {
        if (is_decisive(candidate))
        {
            return candidate;
        }
        candidate = draw();
    }
    return candidate;
}

}  // namespace sampling
