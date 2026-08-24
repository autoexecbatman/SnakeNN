#pragma once

#include <cstdint>

// splitmix64 in eight bytes of state, so copying or reseeding one is an assignment.
// That is what the search needs: it copies a whole environment per simulation.
//
//     SmallRandom rng(seed);
//     const std::uint32_t cell = rng.below(100);   // uniform in [0, 100)
class SmallRandom
{
public:
    // Starts the stream at `seed`. Every seed is valid, zero included.
    explicit SmallRandom(unsigned int seed)
        : state_(static_cast<std::uint64_t>(seed) + GOLDEN_GAMMA)
    {
    }

    // The next 64 bits, uniform over the whole range. Use below() for a bounded draw.
    std::uint64_t next()
    {
        std::uint64_t value = (state_ += GOLDEN_GAMMA);
        value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ull;
        value = (value ^ (value >> 27)) * 0x94D049BB133111EBull;
        return value ^ (value >> 31);
    }

    // Uniform in [0, bound), by Lemire's multiply-shift. Taking the top 32 bits
    // of the draw keeps the arithmetic in 64 bits throughout, so the only
    // narrowing is the deliberate one that reads off the low half.
    std::uint32_t below(std::uint32_t bound)
    {
        std::uint64_t product = (next() >> 32) * bound;
        std::uint32_t low = static_cast<std::uint32_t>(product);
        if (low < bound)
        {
            // Wraps by design: the unsigned negation of bound, modulo bound.
            const std::uint32_t threshold = (0u - bound) % bound;
            while (low < threshold)
            {
                product = (next() >> 32) * bound;
                low = static_cast<std::uint32_t>(product);
            }
        }
        return static_cast<std::uint32_t>(product >> 32);
    }

private:
    // The odd increment splitmix64 advances its state by, from the fractional
    // part of the golden ratio.
    static constexpr std::uint64_t GOLDEN_GAMMA = 0x9E3779B97F4A7C15ull;

    std::uint64_t state_{ 0 };
};
