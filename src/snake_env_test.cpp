#include "snake_env.h"
#include "hamiltonian_cycle.h"
#include <iostream>
#include <string>
#include <vector>

// Properties the training environment has to hold before anything is trained
// against it. A learner cannot report a bug in its own simulator - it will
// simply learn the bug - so these are checked rather than assumed.

namespace {

int failures = 0;

void expect(bool condition, const std::string& description) {
    if (condition) {
        std::cout << "  PASS  " << description << std::endl;
    } else {
        std::cout << "  FAIL  " << description << std::endl;
        failures++;
    }
}

bool bodyCoversCell(const SnakeEnv& env, const Position& cell) {
    for (const auto& segment : env.body()) {
        if (segment == cell) {
            return true;
        }
    }
    return false;
}

// Which relative action moves the head onto `target`, given where each one
// would land. Used to drive the environment from an absolute-cell plan.
bool actionTowards(const SnakeEnv& env, const Position& target, SnakeEnv::Action& action_out) {
    const SnakeEnv::Action actions[] = {SnakeEnv::Action::STRAIGHT, SnakeEnv::Action::LEFT,
                                        SnakeEnv::Action::RIGHT};
    for (SnakeEnv::Action action : actions) {
        if (env.headAfter(action) == target) {
            action_out = action;
            return true;
        }
    }
    return false;
}

void testResetInvariants() {
    SnakeEnv env(20, 20, 7);

    expect(env.body().size() == 1, "a fresh board holds a single segment");
    expect(env.score() == 0 && env.steps() == 0, "score and step count start at zero");
    expect(!env.done() && !env.won(), "a fresh board is neither finished nor won");
    expect(!bodyCoversCell(env, env.food()), "food never spawns on the snake");
    expect(env.foodsToWin() == 399, "a 20x20 board takes 399 foods to fill");
}

void testDeterminism() {
    SnakeEnv first(12, 12, 4242);
    SnakeEnv second(12, 12, 4242);

    bool identical = true;
    for (int step = 0; step < 400; step++) {
        SnakeEnv::Action action = static_cast<SnakeEnv::Action>(step % SnakeEnv::ACTION_COUNT);
        first.step(action);
        second.step(action);
        if (!(first.food() == second.food()) || first.score() != second.score() ||
            first.done() != second.done()) {
            identical = false;
            break;
        }
        if (first.done()) {
            first.reset();
            second.reset();
        }
    }
    expect(identical, "two environments on one seed stay in lockstep");
}

void testTurning() {
    SnakeEnv env(20, 20, 1);
    // reset faces RIGHT, as SnakeGame does.
    expect(env.heading() == Direction::RIGHT, "a fresh board faces right");
    expect(env.headingAfter(SnakeEnv::Action::STRAIGHT) == Direction::RIGHT,
           "going straight keeps the heading");
    expect(env.headingAfter(SnakeEnv::Action::LEFT) == Direction::UP,
           "turning left from right faces up");
    expect(env.headingAfter(SnakeEnv::Action::RIGHT) == Direction::DOWN,
           "turning right from right faces down");
}

void testWallDeath() {
    SnakeEnv env(8, 8, 3);
    SnakeEnv::StepResult result{0.0f, false, false};
    int guard = 0;
    // Facing right from the middle, straight runs into the right wall.
    while (!result.done && guard++ < 100) {
        result = env.step(SnakeEnv::Action::STRAIGHT);
    }
    expect(result.done && !result.won, "running into a wall ends the game without a win");
    expect(result.reward < 0.0f, "death carries a negative reward");
}

void testEatingReward() {
    SnakeEnv env(10, 10, 11);
    bool saw_food = false;
    size_t length_before = env.body().size();

    for (int step = 0; step < 5000 && !saw_food; step++) {
        if (env.done()) {
            env.reset();
            length_before = env.body().size();
            continue;
        }
        length_before = env.body().size();
        // Steer at the food so this terminates without needing a policy.
        Position food = env.food();
        Position head = env.body()[0];
        SnakeEnv::Action chosen = SnakeEnv::Action::STRAIGHT;
        const SnakeEnv::Action actions[] = {SnakeEnv::Action::STRAIGHT, SnakeEnv::Action::LEFT,
                                            SnakeEnv::Action::RIGHT};
        int best_distance = 1 << 30;
        for (SnakeEnv::Action action : actions) {
            Position next = env.headAfter(action);
            int distance = std::abs(next.x - food.x) + std::abs(next.y - food.y);
            if (distance < best_distance) {
                best_distance = distance;
                chosen = action;
            }
        }
        SnakeEnv::StepResult result = env.step(chosen);
        if (result.reward > 0.0f) {
            saw_food = true;
            expect(env.body().size() == length_before + 1, "eating grows the snake by one");
            expect(env.stepsSinceFood() == 0, "eating resets the hunger counter");
        }
    }
    expect(saw_food, "a snake steered at the food eats within the step budget");
}

void testEncoding() {
    SnakeEnv env(10, 10, 5);
    std::vector<float> planes(env.encodedSize(), -1.0f);
    env.encode(planes.data());

    int cells = env.cellCount();
    float body_sum = 0.0f;
    float head_sum = 0.0f;
    float food_sum = 0.0f;
    for (int cell = 0; cell < cells; cell++) {
        body_sum += planes[0 * cells + cell];
        head_sum += planes[1 * cells + cell];
        food_sum += planes[2 * cells + cell];
    }
    expect(body_sum == (float)env.body().size(), "the body plane marks exactly the body");
    expect(head_sum == 1.0f, "the head plane marks exactly one cell");
    expect(food_sum == 1.0f, "the food plane marks exactly one cell");

    int heading_planes_set = 0;
    for (int plane = 4; plane < SnakeEnv::PLANE_COUNT; plane++) {
        float sum = 0.0f;
        for (int cell = 0; cell < cells; cell++) {
            sum += planes[plane * cells + cell];
        }
        if (sum == (float)cells) {
            heading_planes_set++;
        } else if (sum != 0.0f) {
            heading_planes_set = -1000;  // a heading plane must be all or nothing
        }
    }
    expect(heading_planes_set == 1, "exactly one heading plane is set, and it is constant");
}

// The strongest available check on the whole transition function: an
// independent winner drives the environment to a full board. It exercises
// tail-following, which is where an off-by-one in collision hides, and win
// detection, which nothing else reaches.
void testWinnableByCycle() {
    const int size = 6;
    SnakeEnv env(size, size, 99);
    HamiltonianCycle cycle(size, size);
    if (!cycle.generateCycle()) {
        expect(false, "cycle generation for the win test");
        return;
    }

    // Align onto the cycle first. There is no reverse action, so if the cycle
    // leaves the start cell in the direction the head came from, that successor
    // is unreachable until the snake has turned. While the body is one segment
    // long a turn cannot collide, so this terminates immediately - but it has
    // to happen before the follow loop, and it is the reason a relative action
    // space needs an entry phase that an absolute one hides.
    int alignment_steps = 0;
    SnakeEnv::Action entry = SnakeEnv::Action::STRAIGHT;
    while (!actionTowards(env, cycle.getNext(env.body()[0]), entry) && alignment_steps < 4) {
        env.step(SnakeEnv::Action::LEFT);
        alignment_steps++;
    }
    expect(alignment_steps < 4, "the snake can turn onto the cycle within one lap of turns");

    int guard = 0;
    int guard_limit = 40 * env.cellCount() * env.foodsToWin();
    bool steering_failed = false;
    while (!env.done() && guard++ < guard_limit) {
        Position next = cycle.getNext(env.body()[0]);
        SnakeEnv::Action action = SnakeEnv::Action::STRAIGHT;
        if (!actionTowards(env, next, action)) {
            steering_failed = true;
            break;
        }
        env.step(action);
    }

    expect(!steering_failed, "once on the cycle, its successor is always one relative move away");
    expect(env.won(), "a cycle-following snake fills the board");
    expect(env.score() == env.foodsToWin(), "a filled board scores every food");
    expect((int)env.body().size() == env.cellCount(), "a filled board holds one segment per cell");
}

}  // namespace

int main() {
    std::cout << "SnakeEnv properties" << std::endl;
    testResetInvariants();
    testDeterminism();
    testTurning();
    testWallDeath();
    testEatingReward();
    testEncoding();
    testWinnableByCycle();

    std::cout << std::endl;
    if (failures == 0) {
        std::cout << "All checks passed." << std::endl;
        return 0;
    }
    std::cout << failures << " check(s) failed." << std::endl;
    return 1;
}
