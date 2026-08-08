#include "snake_env.h"
#include <cassert>
#include <stdexcept>

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
    const int moved_x = moved.x - cell.x;
    const int moved_y = moved.y - cell.y;
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

SnakeEnv::SnakeEnv(int width, int height, unsigned int seed)
    : width_(width),
      height_(height),
      heading_(Direction::RIGHT),
      done_(false),
      won_(false),
      score_(0),
      steps_(0),
      steps_since_food_(0),
      rng_(seed)
{
    if (width < 2 || height < 2)
    {
        throw std::invalid_argument("SnakeEnv needs a board of at least 2x2");
    }
    reset();
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
        return {DEATH_REWARD, true, false};
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
            return {WIN_REWARD, true, true};
        }

        spawnFood();
        return {FOOD_REWARD, false, false};
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
        return {DEATH_REWARD, true, false};
    }

    return {0.0f, false, false};
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
