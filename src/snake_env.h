#pragma once

#include <cstdint>
#include <vector>

#include "snake_logic.h"  // Direction, Position

// The Snake the learned agent trains on: runtime board size, an explicit seed, and no
// state shared between instances, so thousands can be stepped in parallel. Actions are
// relative to the heading, so no move reverses the snake. It encodes itself into planes
// for a convolutional network. No LibTorch and no raylib.
//
// Usage - one episode, taking whichever action a policy chose:
//
//     SnakeEnv game(10, 10, seed, step_limit);   // width, height, apple stream, budget
//
//     // The step limit bounds the encoding; ending the episode on it is the caller's.
//     while (!game.done() && game.steps() < step_limit)
//     {
//         game.step(SnakeEnv::Action::STRAIGHT);
//     }
//     const bool filled_the_board = game.won();
//
// Stepping a finished episode throws. reset() replays the same board and seed.

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

// One game. See the top of this file for how to run an episode.
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
    // How many actions there are. Three, and no reverse.
    static constexpr int ACTION_COUNT = 3;

    // body, head, food, tail timer, one plane per heading, and the clock
    static constexpr int PLANE_COUNT = 9;

    // Reward scale follows Du, Gemp, Wu and Wu 2022 (arXiv:2211.09622), which
    // trained a winning Snake agent with +1 per apple, -10 for dying and +10 for
    // filling the board. The asymmetry matters: with +1/-1 a long greedy run
    // outscores a short safe one, which is the failure the old trainers here
    // learned. These are floats rather than an env parameter because a reward
    // scale that varies between runs makes their value heads incomparable.
    static constexpr float FOOD_REWARD = 1.0f;
    // Charged for dying, and for starving, which is a death.
    static constexpr float DEATH_REWARD = -10.0f;
    // Paid once, for filling the board.
    static constexpr float WIN_REWARD = 10.0f;

    // What one step produced: what it paid, whether the episode ended, and whether
    // ending meant filling the board.
    struct StepResult
    {
        // What the step paid: an apple, a death, a win, or the per-step reward.
        float reward{ 0.0f };
        // Whether the episode ended on this step.
        bool done{ false };
        // Whether it ended by filling the board.
        bool won{ false };
    };

    // `step_limit` is the game's whole budget and must be at least 1.
    //
    // It lives here rather than in the caller's loop because the search has to see
    // it: every simulation steps a copy of this environment, so a budget the
    // environment does not carry is a budget the search plans as if it were
    // infinite. The same argument put hungerLimit here.
    //
    // It bounds the encoding, not termination. Whether a game ends at the limit
    // stays with the caller that owns the episode - self-play and the evaluator
    // both already do it, and moving that here would change what `done()` means
    // for every existing caller.
    SnakeEnv(int width, int height, unsigned int seed, int step_limit);

    // Back to a one-segment snake, score zero, a fresh apple. Board size, step limit,
    // random stream and any ablation freeze all survive it.
    void reset();
    // Applies one relative action and reports what it paid. Throws if the episode has
    // already finished - call reset() first.
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

    // Columns on the board.
    int width() const { return width_; }
    // Rows on the board.
    int height() const { return height_; }
    // Cells on the board. A snake this long has filled it.
    int cellCount() const { return width_ * height_; }
    // The snake starts one segment long, so this many foods fills the board.
    int foodsToWin() const { return cellCount() - 1; }

    // Apples eaten so far.
    int score() const { return score_; }
    // Steps taken so far.
    int steps() const { return steps_; }
    // The whole step budget, as the constructor was given it.
    int stepLimit() const { return step_limit_; }
    // What the clock plane holds: 1 at the start, 0 once the budget is spent, and
    // never negative even if the caller runs the episode past its limit.
    //
    // Reports the frozen value instead once freezeClockForAblation has been called.
    float budgetRemaining() const;

    // How many free cells the head could still reach after taking `action`.
    //
    // Zero when the action kills. Otherwise a flood fill from where the head lands,
    // over cells the body does not occupy, counting the landing cell itself. The
    // tail cell is treated as free because it moves out of the way on the same
    // tick, which is what makes a snake able to follow its own tail.
    //
    // This is the safety question a Hamiltonian cycle answers by construction and
    // that a learned agent has no way to answer at all: a region smaller than the
    // snake cannot hold the snake, so entering one is fatal however good the
    // position looks to a value head. `HamiltonianCycle::isShortcutSafe` cannot
    // stand in - it compares cycle indices, which is only a safety argument when
    // the body already lies along the cycle.
    //
    // O(cells), so it is affordable per root move and not per node of a search.
    int reachableCells(Action action) const;

    // Whether the head can still reach its own tail after taking `action`.
    //
    // False when the action kills. Otherwise the same flood fill, asking a
    // different question: a snake that can reach its tail can follow it, and the
    // region it is in opens up behind it every tick, so it is not sealed. A snake
    // cut off from its tail is in a pocket and dies once the pocket fills,
    // whatever the pocket's size.
    //
    // This is the test a trap guard wants, and a cell count is not: past the
    // halfway mark a board has fewer free cells than the snake has segments, so
    // "region smaller than the snake" is true of every move in every endgame.
    //
    // Approximate in one direction, and it is the same approximation
    // reachableCells makes: an action that eats leaves the tail in place for a
    // tick, and this counts it as vacating. It therefore calls a move safe
    // slightly more often than it should, never less.
    bool tailReachable(Action action) const;

    // Makes the clock read `value` for the rest of this environment's life.
    //
    // Measurement only, and it is the whole of an ablation: the network sees a
    // constant where it was trained to see time running out, so a win rate
    // measured against a run without it is the contribution of time awareness and
    // nothing else. The weights are untouched, so nothing else about the agent
    // moves. Copies carry the freeze, which is what makes the search see it too.
    //
    // Asserts 0 <= value <= 1. The trainer must never call this - a policy trained
    // against a frozen clock is a different agent, not an ablation of this one.
    void freezeClockForAblation(float value);
    // Steps since the last apple. Reaching hungerLimit() starves the snake.
    int stepsSinceFood() const { return steps_since_food_; }
    // Whether the episode is over. Running out of steps does not set it - the caller
    // owning the episode decides that.
    bool done() const { return done_; }
    // Whether it ended by filling the board.
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

    // The absolute direction the head is travelling.
    Direction heading() const { return heading_; }
    // The segments, head first. Invalidated by step() and reset().
    const std::vector<Position>& body() const { return body_; }
    // Where the apple is. Undefined on a won board, where none exists.
    Position food() const { return food_; }

    // Where `action` would put the head, without applying it. Search needs this
    // to order children before it commits to expanding them.
    Position headAfter(Action action) const;
    // Which absolute direction `action` would turn to, without applying it. Never the
    // reverse of the current heading.
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
    // Floats encode() writes - size the caller's buffer with this.
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
        // Where the apple sits, as a cell index. Meaningless when won is true.
        unsigned short food_cell{ 0 };
        // The heading, as the Direction enumerator's value.
        unsigned char heading{ 0 };
        // Whether this position is a filled board.
        bool won{ false };
        // The clock, already normalised. Carried rather than recomputed because a
        // snapshot outlives its environment: the replay buffer holds these for
        // thousands of games and has no way back to the step limit they came from.
        // Defaulted to a full clock so a hand-built snapshot - the probes construct
        // positions directly - holds a defined value rather than whatever was on
        // the stack.
        float budget_remaining{ 1.0f };
    };

    // This position as a snapshot. Throws on a board of more than 65535 cells, which
    // is past what a 16-bit cell index can hold.
    Snapshot snapshot() const;

    // Encodes a snapshot for a board of the given size. Static because the
    // trainer holds snapshots long after their environment is gone.
    static void encodeSnapshot(int width, int height, const Snapshot& snapshot, float* planes_out);

private:
    // Every one of these is assigned by the constructor and again by reset(). The
    // initializers are here so the object is never momentarily undefined, not to
    // supply a value anything reads.
    int width_{ 0 };
    int height_{ 0 };
    std::vector<Position> body_;  // index 0 is the head
    Position food_{ 0, 0 };
    Direction heading_{ Direction::UP };
    bool done_{ false };
    bool won_{ false };
    int score_{ 0 };
    int steps_{ 0 };
    int steps_since_food_{ 0 };
    int step_limit_{ 0 };
    // Set by freezeClockForAblation, and negative when the clock runs normally -
    // one member rather than a flag and a value, since no legitimate budget is
    // below zero.
    float frozen_clock_{ -1.0f };
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

    // What one flood fill from where `action` lands finds. Both public trap
    // queries read it, so they can never disagree about what is connected.
    // Zero cells and an unreached tail when the action kills.
    struct Region
    {
        int cells{ 0 };
        bool holds_tail{ false };
    };
    Region floodAfter(Action action) const;

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
