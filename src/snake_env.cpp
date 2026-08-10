#include <cassert>
#include <stdexcept>

#include "snake_env.h"

namespace
{

// The geometry, in one place. All three take a Direction and are total over the
// four enumerators, so an unrecognised value means someone cast an integer into
// the enum rather than that a case was forgotten.
//
// Each ends in a default that throws instead of falling off the end of the
// switch. The trade is worth stating: while these switches listed every
// enumerator and had no default, a fifth Direction would have been a compiler
// diagnostic (MSVC C4062) at each of the three sites. With a default present
// that becomes a runtime throw instead, so adding a Direction is now caught by
// running the tests rather than by building. The assert above each throw is
// what keeps the debug builds failing at the fault site.

Position stepFrom(const Position& cell, Direction heading)
{
    Position moved(0, 0);
    switch (heading)
    {
        case Direction::UP:
        {
            moved = Position(cell.x, cell.y - 1);
            break;
        }
        case Direction::DOWN:
        {
            moved = Position(cell.x, cell.y + 1);
            break;
        }
        case Direction::LEFT:
        {
            moved = Position(cell.x - 1, cell.y);
            break;
        }
        case Direction::RIGHT:
        {
            moved = Position(cell.x + 1, cell.y);
            break;
        }
        default:
        {
            assert(false && "stepFrom given a Direction outside the four enumerators");
            throw std::logic_error("unreachable heading");
        }
    }

    // One orthogonal step, and the arithmetic that has to deliver it is right
    // here. Asserted at the source rather than at the callers: headAfter said
    // the same thing a level up, which meant one property stated twice and only
    // one of the two places able to name the line that broke it. No bound on the
    // result - a step off the board is a legal answer and its caller decides.
    // Referenced only by the assert below, so Release, where NDEBUG deletes it,
    // has no use for them.
    [[maybe_unused]] const int moved_x = moved.x - cell.x;
    [[maybe_unused]] const int moved_y = moved.y - cell.y;
    assert(((moved_x == 0) != (moved_y == 0)) && (moved_x * moved_x + moved_y * moved_y) == 1 &&
           "stepFrom moved by something other than one orthogonal step");

    return moved;
}

Direction turnLeft(Direction heading)
{
    Direction turned = heading;
    switch (heading)
    {
        case Direction::UP:
        {
            turned = Direction::LEFT;
            break;
        }
        case Direction::LEFT:
        {
            turned = Direction::DOWN;
            break;
        }
        case Direction::DOWN:
        {
            turned = Direction::RIGHT;
            break;
        }
        case Direction::RIGHT:
        {
            turned = Direction::UP;
            break;
        }
        default:
        {
            assert(false && "turnLeft given a Direction outside the four enumerators");
            throw std::logic_error("unreachable heading");
        }
    }

    // A quarter turn changes the heading. If it ever returned its own argument
    // the snake would have a second way to go straight, and the search would
    // carry two children that are the same move.
    assert(turned != heading && "turnLeft returned the heading it was given");
    return turned;
}

Direction turnRight(Direction heading)
{
    Direction turned = heading;
    switch (heading)
    {
        case Direction::UP:
        {
            turned = Direction::RIGHT;
            break;
        }
        case Direction::RIGHT:
        {
            turned = Direction::DOWN;
            break;
        }
        case Direction::DOWN:
        {
            turned = Direction::LEFT;
            break;
        }
        case Direction::LEFT:
        {
            turned = Direction::UP;
            break;
        }
        default:
        {
            assert(false && "turnRight given a Direction outside the four enumerators");
            throw std::logic_error("unreachable heading");
        }
    }

    assert(turned != heading && "turnRight returned the heading it was given");
    // The two turns must disagree, or one relative action aliases the other.
    // Stated here rather than in both, since checking it once covers the pair.
    assert(turned != turnLeft(heading) && "turnRight and turnLeft agree on this heading");
    return turned;
}

}  // namespace

SnakeEnv::SnakeEnv(int width, int height, unsigned int seed, int step_limit)
    : width_(width),
      height_(height),
      heading_(Direction::RIGHT),
      done_(false),
      won_(false),
      score_(0),
      steps_(0),
      steps_since_food_(0),
      step_limit_(step_limit),
      rng_(seed)
{
    if (width < 2 || height < 2)
    {
        throw std::invalid_argument("SnakeEnv needs a board of at least 2x2");
    }
    if (step_limit < 1)
    {
        throw std::invalid_argument("SnakeEnv needs a step limit of at least 1");
    }
    reset();
}

int SnakeEnv::reachableCells(Action action) const
{
    if (wouldDie(action))
    {
        return 0;
    }

    const Position landing = headAfter(action);
    // The tail vacates as the head arrives, so it is not an obstacle. Nothing else
    // about the body moves within this tick.
    const Position vacated = body_.back();

    std::vector<char> seen(static_cast<size_t>(cellCount()), 0);
    std::vector<int> frontier;
    frontier.push_back(cellIndex(landing));
    seen[static_cast<size_t>(cellIndex(landing))] = 1;

    int reached = 0;
    while (!frontier.empty())
    {
        const int cell = frontier.back();
        frontier.pop_back();
        reached++;

        const Position at{ cell % width_, cell / width_ };
        const Position neighbours[4] = {
            { at.x + 1, at.y }, { at.x - 1, at.y }, { at.x, at.y + 1 }, { at.x, at.y - 1 }
        };
        for (const Position& next : neighbours)
        {
            if (!insideGrid(next))
            {
                continue;
            }
            const int index = cellIndex(next);
            if (seen[static_cast<size_t>(index)])
            {
                continue;
            }
            const bool blocked = occupancy_[static_cast<size_t>(index)] != 0 && !(next == vacated);
            if (blocked)
            {
                continue;
            }
            seen[static_cast<size_t>(index)] = 1;
            frontier.push_back(index);
        }
    }
    return reached;
}

void SnakeEnv::freezeClockForAblation(float value)
{
    // A fraction of the budget, so anything outside [0, 1] is a caller error and
    // not a value the clock could ever have held.
    assert(value >= 0.0f && value <= 1.0f);
    frozen_clock_ = value;
}

float SnakeEnv::budgetRemaining() const
{
    if (frozen_clock_ >= 0.0f)
    {
        return frozen_clock_;
    }
    const int spent = std::min(steps_, step_limit_);
    return static_cast<float>(step_limit_ - spent) / static_cast<float>(step_limit_);
}

void SnakeEnv::reset()
{
    body_.clear();
    body_.push_back(Position(width_ / 2, height_ / 2));

    occupancy_.assign(cellCount(), 0);
    occupancy_[cellIndex(body_[0])] = 1;

    heading_ = Direction::RIGHT;
    done_ = false;
    won_ = false;
    score_ = 0;
    steps_ = 0;
    steps_since_food_ = 0;

    spawnFood();
}

Direction SnakeEnv::headingAfter(Action action) const
{
    Direction turned = heading_;
    switch (action)
    {
        case Action::STRAIGHT:
        {
            turned = heading_;
            break;
        }
        case Action::LEFT:
        {
            turned = turnLeft(heading_);
            break;
        }
        case Action::RIGHT:
        {
            turned = turnRight(heading_);
            break;
        }
        default:
        {
            // Same trade as the three helpers above: a default costs the
            // compiler diagnostic that a fourth Action would have produced here,
            // and buys a failure at this line instead of undefined fallthrough.
            assert(false && "headingAfter given an Action outside the three enumerators");
            throw std::logic_error("unreachable action");
        }
    }

    // No action reverses the snake. This is the whole reason the action space is
    // relative rather than absolute: a reverse move does not exist rather than
    // existing and being filtered, which is what lets the search treat all three
    // children as legal and keeps the tree a third smaller per ply. Two quarter
    // turns name the reverse without needing a function for it.
    //
    // The individual turns already assert they change the heading, so this adds
    // the one thing they cannot see on their own - that neither of them, nor
    // going straight, lands on the opposite direction.
    assert(turned != turnLeft(turnLeft(heading_)) &&
           "headingAfter produced the reverse heading - the snake could reverse into its neck");

    return turned;
}

Position SnakeEnv::headAfter(Action action) const
{
    // The snake is never bodiless: reset() places one segment and no path
    // shortens the body, so an empty body_ means the environment was used
    // before it was constructed rather than that the game ended.
    assert(!body_.empty() && "headAfter called on an environment with no body");

    // The one-orthogonal-step guarantee every caller here relies on - the search
    // ordering children, blocksHead indexing occupancy_, the tail-vacating rule
    // comparing against body_.back() - is asserted inside stepFrom, which is
    // where the arithmetic lives. Restating it here would be one property in two
    // places, and the copy further from the fault.
    return stepFrom(body_[0], headingAfter(action));
}

bool SnakeEnv::blocksHead(const Position& next, bool will_eat) const
{
    if (!insideGrid(next))
    {
        return true;
    }

    if (occupancy_[cellIndex(next)] == 0)
    {
        return false;
    }

    // Reaching here means the head is entering an occupied cell, so it cannot
    // also be entering the food: spawnFood only ever picks a cell with zero
    // occupancy, and nothing else assigns food_. If this fires, food has been
    // placed on the snake and the eating-versus-collision order below is being
    // asked a question the game should never produce.
    assert(!will_eat && "food is sitting on a body cell - spawnFood placed it wrongly");

    // The one occupied cell that is safe to enter: the tail, which vacates on
    // this same step. Eating cancels that, because the snake grows and its tail
    // stays where it is. That second half is unreachable while the assertion
    // above holds; it is kept so the function is correct for any input rather
    // than only for the inputs this game can build.
    const bool tail_vacates = (next == body_.back()) && !will_eat;
    return !tail_vacates;
}

bool SnakeEnv::wouldDie(Action action) const
{
    // A finished episode has no next move, so asking what one would cost is
    // meaningless rather than merely unanswerable. The search filters finished
    // games before it descends, so this is impossible when the caller is wired
    // correctly - assert at the site of the fault instead of returning a value
    // that would read as an answer.
    assert(!done_ && "wouldDie called on a finished episode - there is no move to take");

    // The three ways a step ends the episode, in the order step() reaches them.
    // Everything here is a query on the current position; nothing is mutated
    // and nothing is copied, which is the whole reason this exists.
    const Position next = headAfter(action);
    const bool will_eat = (next == food_);

    if (blocksHead(next, will_eat))
    {
        return true;
    }

    if (will_eat)
    {
        // Eating resets the hunger clock, so starvation cannot follow. If that
        // apple was the last one the board is full, which is a win, not a death.
        return false;
    }

    return steps_since_food_ + 1 >= hungerLimit();
}

SnakeEnv::StepResult SnakeEnv::step(Action action)
{
    if (done_)
    {
        throw std::logic_error("step called on a finished episode - reset first");
    }

    heading_ = headingAfter(action);
    Position next = stepFrom(body_[0], heading_);
    steps_++;

    // Off-board reads as unoccupied food-free ground, so the comparison against
    // food_ is safe before the bounds test inside blocksHead: food always sits
    // on the board, so an off-board head never matches it.
    const bool will_eat = (next == food_);

    if (blocksHead(next, will_eat))
    {
        done_ = true;
        return { DEATH_REWARD, true, false };
    }

    if (will_eat)
    {
        body_.insert(body_.begin(), next);
        occupancy_[cellIndex(next)] = 1;
        score_++;
        steps_since_food_ = 0;

        if (static_cast<int>(body_.size()) == cellCount())
        {
            // Board full: no cell is left to place food in, and nothing remains
            // to be done. This is the win, and it is terminal.
            won_ = true;
            done_ = true;
            return { WIN_REWARD, true, true };
        }

        spawnFood();
        return { FOOD_REWARD, false, false };
    }

    // Free the tail before occupying the new head, so a head entering the cell
    // the tail is leaving sees it empty.
    occupancy_[cellIndex(body_.back())] = 0;
    body_.pop_back();
    body_.insert(body_.begin(), next);
    occupancy_[cellIndex(next)] = 1;
    steps_since_food_++;

    if (steps_since_food_ >= hungerLimit())
    {
        // Starvation is a death and pays like one, which is what makes stalling
        // strictly worse than playing.
        done_ = true;
        return { DEATH_REWARD, true, false };
    }

    return { 0.0f, false, false };
}

SnakeEnv::Snapshot SnakeEnv::snapshot() const
{
    // Cell indices are stored as 16-bit to keep the replay buffer small, which
    // caps the board at 255x255. Nothing here goes near that, and a board that
    // did would corrupt every stored position silently rather than failing - so
    // the limit is asserted at the one place that depends on it.
    if (cellCount() > 65535)
    {
        throw std::logic_error("board too large for 16-bit snapshot cell indices");
    }

    Snapshot out;
    out.body_cells.reserve(body_.size());
    for (const Position& segment : body_)
    {
        out.body_cells.push_back(static_cast<unsigned short>(cellIndex(segment)));
    }
    out.food_cell = static_cast<unsigned short>(cellIndex(food_));
    out.heading = static_cast<unsigned char>(heading_);
    out.won = won_;
    out.budget_remaining = budgetRemaining();
    return out;
}

void SnakeEnv::encodeSnapshot(int width, int height, const Snapshot& snapshot, float* planes_out)
{
    const int cells = width * height;
    for (int index = 0; index < PLANE_COUNT * cells; index++)
    {
        planes_out[index] = 0.0f;
    }

    float* body_plane = planes_out + 0 * cells;
    float* head_plane = planes_out + 1 * cells;
    float* food_plane = planes_out + 2 * cells;
    float* timer_plane = planes_out + 3 * cells;

    // The timer plane is what makes tail-chasing visible to a convolution: a
    // cell holds how long it stays blocked, scaled so the tail reads near zero
    // and the head reads one. Without it every occupied cell looks like a wall.
    const float length = static_cast<float>(snapshot.body_cells.size());
    for (size_t index = 0; index < snapshot.body_cells.size(); index++)
    {
        int cell = snapshot.body_cells[index];
        body_plane[cell] = 1.0f;
        timer_plane[cell] = (length - static_cast<float>(index)) / length;
    }

    head_plane[snapshot.body_cells[0]] = 1.0f;
    if (!snapshot.won)
    {
        food_plane[snapshot.food_cell] = 1.0f;
    }

    float* heading_out = planes_out + (4 + static_cast<int>(snapshot.heading)) * cells;
    for (int cell = 0; cell < cells; cell++)
    {
        heading_out[cell] = 1.0f;
    }

    // The clock, as a constant plane. A convolution has no other way to see a
    // scalar, and the agent that cannot see its budget has no reason to abandon
    // an apple that has already cost it sixty steps - which is where this
    // project's losses are: the slowest tenth of apples take 39 percent of every
    // step played.
    float* clock_plane = planes_out + 8 * cells;
    for (int cell = 0; cell < cells; cell++)
    {
        clock_plane[cell] = snapshot.budget_remaining;
    }
}

void SnakeEnv::encode(float* planes_out) const
{
    encodeSnapshot(width_, height_, snapshot(), planes_out);
}

bool SnakeEnv::insideGrid(const Position& cell) const
{
    return cell.x >= 0 && cell.x < width_ && cell.y >= 0 && cell.y < height_;
}

void SnakeEnv::spawnFood()
{
    int free_cells = 0;
    for (int cell = 0; cell < cellCount(); cell++)
    {
        if (occupancy_[cell] == 0)
        {
            free_cells++;
        }
    }
    if (free_cells == 0)
    {
        throw std::logic_error("spawnFood called with a full board - the win was missed");
    }

    int wanted = static_cast<int>(rng_.below(static_cast<std::uint32_t>(free_cells)));
    for (int cell = 0; cell < cellCount(); cell++)
    {
        if (occupancy_[cell] == 0)
        {
            if (wanted == 0)
            {
                food_ = Position(cell % width_, cell / width_);
                return;
            }
            wanted--;
        }
    }

    throw std::logic_error("free cell count and scan disagree");
}
