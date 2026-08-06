#include "snake_env.h"
#include <stdexcept>

namespace {

Position stepFrom(const Position& cell, Direction heading) {
    switch (heading) {
        case Direction::UP: return Position(cell.x, cell.y - 1);
        case Direction::DOWN: return Position(cell.x, cell.y + 1);
        case Direction::LEFT: return Position(cell.x - 1, cell.y);
        case Direction::RIGHT: return Position(cell.x + 1, cell.y);
    }
    throw std::logic_error("unreachable heading");
}

Direction turnLeft(Direction heading) {
    switch (heading) {
        case Direction::UP: return Direction::LEFT;
        case Direction::LEFT: return Direction::DOWN;
        case Direction::DOWN: return Direction::RIGHT;
        case Direction::RIGHT: return Direction::UP;
    }
    throw std::logic_error("unreachable heading");
}

Direction turnRight(Direction heading) {
    switch (heading) {
        case Direction::UP: return Direction::RIGHT;
        case Direction::RIGHT: return Direction::DOWN;
        case Direction::DOWN: return Direction::LEFT;
        case Direction::LEFT: return Direction::UP;
    }
    throw std::logic_error("unreachable heading");
}

}  // namespace

SnakeEnv::SnakeEnv(int width, int height, unsigned int seed)
    : width_(width), height_(height), heading_(Direction::RIGHT), done_(false), won_(false),
      score_(0), steps_(0), steps_since_food_(0), rng_(seed) {
    if (width < 2 || height < 2) {
        throw std::invalid_argument("SnakeEnv needs a board of at least 2x2");
    }
    reset();
}

void SnakeEnv::reset() {
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

Direction SnakeEnv::headingAfter(Action action) const {
    switch (action) {
        case Action::STRAIGHT: return heading_;
        case Action::LEFT: return turnLeft(heading_);
        case Action::RIGHT: return turnRight(heading_);
    }
    throw std::logic_error("unreachable action");
}

Position SnakeEnv::headAfter(Action action) const {
    return stepFrom(body_[0], headingAfter(action));
}

SnakeEnv::StepResult SnakeEnv::step(Action action) {
    if (done_) {
        throw std::logic_error("step called on a finished episode - reset first");
    }

    heading_ = headingAfter(action);
    Position next = stepFrom(body_[0], heading_);
    steps_++;

    if (!insideGrid(next)) {
        done_ = true;
        return {-1.0f, true, false};
    }

    bool will_eat = (next == food_);
    // The tail cell is enterable because the tail vacates it on this same step
    // - unless the snake eats, in which case the tail stays put.
    bool entering_tail_cell = (next == body_.back());
    bool blocked = occupancy_[cellIndex(next)] != 0 && !(entering_tail_cell && !will_eat);

    if (blocked) {
        done_ = true;
        return {-1.0f, true, false};
    }

    if (will_eat) {
        body_.insert(body_.begin(), next);
        occupancy_[cellIndex(next)] = 1;
        score_++;
        steps_since_food_ = 0;

        if ((int)body_.size() == cellCount()) {
            // Board full: no cell is left to place food in, and nothing remains
            // to be done. This is the win, and it is terminal.
            won_ = true;
            done_ = true;
            return {1.0f, true, true};
        }

        spawnFood();
        return {1.0f, false, false};
    }

    // Free the tail before occupying the new head, so a head entering the cell
    // the tail is leaving sees it empty.
    occupancy_[cellIndex(body_.back())] = 0;
    body_.pop_back();
    body_.insert(body_.begin(), next);
    occupancy_[cellIndex(next)] = 1;
    steps_since_food_++;

    return {0.0f, false, false};
}

void SnakeEnv::encode(float* planes_out) const {
    const int cells = cellCount();
    for (int index = 0; index < PLANE_COUNT * cells; index++) {
        planes_out[index] = 0.0f;
    }

    float* body_plane = planes_out + 0 * cells;
    float* head_plane = planes_out + 1 * cells;
    float* food_plane = planes_out + 2 * cells;
    float* timer_plane = planes_out + 3 * cells;

    // The timer plane is what makes tail-chasing visible to a convolution: a
    // cell holds how long it stays blocked, scaled so the tail reads near zero
    // and the head reads one. Without it every occupied cell looks like a wall.
    const float length = (float)body_.size();
    for (size_t index = 0; index < body_.size(); index++) {
        int cell = cellIndex(body_[index]);
        body_plane[cell] = 1.0f;
        timer_plane[cell] = (length - (float)index) / length;
    }

    head_plane[cellIndex(body_[0])] = 1.0f;
    if (!won_) {
        food_plane[cellIndex(food_)] = 1.0f;
    }

    int heading_plane = 4 + (int)heading_;
    float* heading_out = planes_out + heading_plane * cells;
    for (int cell = 0; cell < cells; cell++) {
        heading_out[cell] = 1.0f;
    }
}

bool SnakeEnv::insideGrid(const Position& cell) const {
    return cell.x >= 0 && cell.x < width_ && cell.y >= 0 && cell.y < height_;
}

void SnakeEnv::spawnFood() {
    int free_cells = 0;
    for (int cell = 0; cell < cellCount(); cell++) {
        if (occupancy_[cell] == 0) {
            free_cells++;
        }
    }
    if (free_cells == 0) {
        throw std::logic_error("spawnFood called with a full board - the win was missed");
    }

    std::uniform_int_distribution<int> pick(0, free_cells - 1);
    int wanted = pick(rng_);
    for (int cell = 0; cell < cellCount(); cell++) {
        if (occupancy_[cell] == 0) {
            if (wanted == 0) {
                food_ = Position(cell % width_, cell / width_);
                return;
            }
            wanted--;
        }
    }

    throw std::logic_error("free cell count and scan disagree");
}
