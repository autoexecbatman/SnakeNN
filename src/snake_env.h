#pragma once
#include "snake_logic.h"  // Direction, Position
#include <cstdint>
#include <vector>

// Eight bytes of generator, because the search copies a whole environment once
// per simulation and then reseeds it.
//
// std::mt19937 carries 624 words - about 2.5KB - so it was being copied on the
// hottest path in the search, and reseeding one costs a full state
// initialisation. Neither is affordable here and neither buys anything: the only
// draw in this environment picks a free cell for an apple.
//
// splitmix64, which is the standard seeding routine for the xoshiro family and
// passes BigCrush on its own.
class SmallRandom
{
public:
    explicit SmallRandom(unsigned int seed)
        : state_(static_cast<std::uint64_t>(seed) + GOLDEN_GAMMA)
    {
    }

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

    std::uint64_t state_;
};

// Training environment. Separate from SnakeGame on purpose:
//
//   - board size is a runtime argument, because the curriculum trains on small
//     boards first and SnakeGame fixes 20x20 at compile time for 25 other files
//   - all state is per-instance, so thousands of these can step in parallel
//     (SnakeGame::getReward keeps its previous head position in function-local
//     statics, which silently couples every game in the process)
//   - actions are relative, so there is no reverse action to ignore
//   - it encodes itself into planes for a convolutional network
//
// No LibTorch and no raylib, so it builds and tests without the CUDA toolchain.
class SnakeEnv
{
public:
    // Relative to the current heading. A reverse action does not exist rather
    // than existing and being ignored, which keeps the MDP honest and the
    // search tree a third smaller per ply.
    enum class Action
    {
        STRAIGHT,
        LEFT,
        RIGHT
    };
    static constexpr int ACTION_COUNT = 3;

    // body, head, food, tail timer, and one plane per heading
    static constexpr int PLANE_COUNT = 8;

    // Reward scale follows Du, Gemp, Wu and Wu 2022 (arXiv:2211.09622), which
    // trained a winning Snake agent with +1 per apple, -10 for dying and +10 for
    // filling the board. The asymmetry matters: with +1/-1 a long greedy run
    // outscores a short safe one, which is the failure the old trainers here
    // learned. These are floats rather than an env parameter because a reward
    // scale that varies between runs makes their value heads incomparable.
    static constexpr float FOOD_REWARD = 1.0f;
    static constexpr float DEATH_REWARD = -10.0f;
    static constexpr float WIN_REWARD = 10.0f;

    struct StepResult
    {
        float reward;
        bool done;
        bool won;
    };

    SnakeEnv(int width, int height, unsigned int seed);

    void reset();
    StepResult step(Action action);

    // Replace the random stream without disturbing the position.
    //
    // The search needs this. Every simulation starts from a copy of the root,
    // and a copy carries the root's generator with it - so without a reseed all
    // simulations draw the same apples and the tree solves a deterministic
    // problem while the game is stochastic. Reseeding per simulation is what
    // makes the visit counts an average over chance rather than one lucky
    // rollout. Eight bytes, so it is an assignment rather than a state rebuild.
    void reseed(unsigned int seed) { rng_ = SmallRandom(seed); }

    int width() const { return width_; }
    int height() const { return height_; }
    int cellCount() const { return width_ * height_; }
    // The snake starts one segment long, so this many foods fills the board.
    int foodsToWin() const { return cellCount() - 1; }

    int score() const { return score_; }
    int steps() const { return steps_; }
    int stepsSinceFood() const { return steps_since_food_; }
    bool done() const { return done_; }
    bool won() const { return won_; }

    // Starve if the snake goes this long without eating, counted as a death.
    //
    // This is part of the task, not a safety valve, and it has to live in the
    // environment so that search cannot plan around its absence. Without it the
    // incentives say to stall: doing nothing returns 0 and 0 beats risking -10,
    // which is exactly what the first training run learned - every game timed
    // out, average score under one apple.
    //
    // Twice the cell count, so a snake following a full-board cycle still fits
    // comfortably inside it - its worst case between apples is one lap, and the
    // rule is meant to punish aimlessness rather than thorough play.
    int hungerLimit() const { return 2 * cellCount(); }

    Direction heading() const { return heading_; }
    const std::vector<Position>& body() const { return body_; }
    Position food() const { return food_; }

    // Where `action` would put the head, without applying it. Search needs this
    // to order children before it commits to expanding them.
    Position headAfter(Action action) const;
    Direction headingAfter(Action action) const;

    // Whether taking `action` ends the episode in death - a wall, the body, or
    // starvation. Filling the board is not a death and reports false.
    //
    // Exists because the search asks this after every descent step, and the
    // obvious implementation - copy the environment and step the copy - was
    // costing three full copies of the body and occupancy vectors per ply. That
    // showed up as one saturated CPU core with the GPU at 30 percent.
    bool wouldDie(Action action) const;

    // Writes PLANE_COUNT * height * width floats in plane-major order.
    // The caller owns the memory; nothing is allocated here, because this runs
    // once per node visited in the tree.
    void encode(float* planes_out) const;
    int encodedSize() const { return PLANE_COUNT * cellCount(); }

    // A position, compactly: cell indices rather than planes.
    //
    // A replay buffer that stores encoded planes stores 3.2KB per move at
    // 10x10 and 12.8KB at 20x20, which is how the first long run drove the
    // machine into swap. A snapshot is one 16-bit index per body segment - a
    // sixteenth of the size, lossless, and the encoding is regenerated when a
    // batch is drawn. Encoding measures 50M observations/s, so recomputing it
    // costs nothing next to keeping it.
    struct Snapshot
    {
        std::vector<unsigned short> body_cells;  // head first
        unsigned short food_cell;
        unsigned char heading;
        bool won;
    };

    Snapshot snapshot() const;

    // Encodes a snapshot for a board of the given size. Static because the
    // trainer holds snapshots long after their environment is gone.
    static void encodeSnapshot(int width, int height, const Snapshot& snapshot, float* planes_out);

private:
    int width_;
    int height_;
    std::vector<Position> body_;  // index 0 is the head
    Position food_;
    Direction heading_;
    bool done_;
    bool won_;
    int score_;
    int steps_;
    int steps_since_food_;
    SmallRandom rng_;
    // Occupancy by cell index: 1 where a body segment sits, 0 where nothing
    // does. Keeps collision and food placement off the O(length) scan that
    // dominates a naive implementation once the snake is long. It records only
    // whether a cell is taken, not which segment took it - the one place that
    // needs the distinction, entering the cell the tail is leaving, compares
    // against body_.back() directly.
    std::vector<int> occupancy_;

    int cellIndex(const Position& cell) const { return cell.y * width_ + cell.x; }
    bool insideGrid(const Position& cell) const;

    // Whether a head arriving at `next` collides - the wall, or a body segment
    // that is still there when it arrives.
    //
    // Both step() and wouldDie() ask exactly this question, and they have to
    // give the same answer or the search plans against a different game than
    // the one it plays. One definition is what makes that true by construction
    // rather than by a property test noticing afterwards.
    //
    // `will_eat` is a parameter rather than recomputed here because it changes
    // the answer: the tail cell is enterable, since the tail leaves it on this
    // same step - but a snake that eats grows instead, and its tail stays put.
    bool blocksHead(const Position& next, bool will_eat) const;
    void spawnFood();
};
