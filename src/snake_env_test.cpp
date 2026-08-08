#include "snake_env.h"
#include "hamiltonian_cycle.h"
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

// Properties the training environment has to hold before anything is trained
// against it. A learner cannot report a bug in its own simulator - it will
// simply learn the bug - so these are checked rather than assumed.

namespace
{

int failures = 0;

void expect(bool condition, const std::string& description)
{
    if (condition)
    {
        std::cout << "  PASS  " << description << std::endl;
    }
    else
    {
        std::cout << "  FAIL  " << description << std::endl;
        failures++;
    }
}

bool bodyCoversCell(const SnakeEnv& env, const Position& cell)
{
    for (const auto& segment : env.body())
    {
        if (segment == cell)
        {
            return true;
        }
    }
    return false;
}

bool insideBoard(const SnakeEnv& env, const Position& cell)
{
    return cell.x >= 0 && cell.x < env.width() && cell.y >= 0 && cell.y < env.height();
}

// Which relative action moves the head onto `target`, given where each one
// would land. Used to drive the environment from an absolute-cell plan.
bool actionTowards(const SnakeEnv& env, const Position& target, SnakeEnv::Action& action_out)
{
    const SnakeEnv::Action actions[] = {SnakeEnv::Action::STRAIGHT, SnakeEnv::Action::LEFT,
                                        SnakeEnv::Action::RIGHT};
    for (SnakeEnv::Action action : actions)
    {
        if (env.headAfter(action) == target)
        {
            action_out = action;
            return true;
        }
    }
    return false;
}

void testResetInvariants()
{
    SnakeEnv env(20, 20, 7);

    expect(env.body().size() == 1, "a fresh board holds a single segment");
    expect(env.score() == 0 && env.steps() == 0, "score and step count start at zero");
    expect(!env.done() && !env.won(), "a fresh board is neither finished nor won");
    expect(!bodyCoversCell(env, env.food()), "food never spawns on the snake");
    expect(env.foodsToWin() == 399, "a 20x20 board takes 399 foods to fill");
}

void testDeterminism()
{
    SnakeEnv first(12, 12, 4242);
    SnakeEnv second(12, 12, 4242);

    bool identical = true;
    for (int step = 0; step < 400; step++)
    {
        SnakeEnv::Action action = static_cast<SnakeEnv::Action>(step % SnakeEnv::ACTION_COUNT);
        first.step(action);
        second.step(action);
        if (!(first.food() == second.food()) || first.score() != second.score() ||
            first.done() != second.done())
        {
            identical = false;
            break;
        }
        if (first.done())
        {
            first.reset();
            second.reset();
        }
    }
    expect(identical, "two environments on one seed stay in lockstep");
}

void testTurning()
{
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

void testWallDeath()
{
    SnakeEnv env(8, 8, 3);
    SnakeEnv::StepResult result{0.0f, false, false};
    int guard = 0;
    // Facing right from the middle, straight runs into the right wall.
    while (!result.done && guard++ < 100)
    {
        result = env.step(SnakeEnv::Action::STRAIGHT);
    }
    expect(result.done && !result.won, "running into a wall ends the game without a win");
    expect(result.reward < 0.0f, "death carries a negative reward");
}

void testEatingReward()
{
    SnakeEnv env(10, 10, 11);
    bool saw_food = false;
    size_t length_before = env.body().size();

    for (int step = 0; step < 5000 && !saw_food; step++)
    {
        if (env.done())
        {
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
        for (SnakeEnv::Action action : actions)
        {
            Position next = env.headAfter(action);
            int distance = std::abs(next.x - food.x) + std::abs(next.y - food.y);
            if (distance < best_distance)
            {
                best_distance = distance;
                chosen = action;
            }
        }
        SnakeEnv::StepResult result = env.step(chosen);
        if (result.reward > 0.0f)
        {
            saw_food = true;
            expect(env.body().size() == length_before + 1, "eating grows the snake by one");
            expect(env.stepsSinceFood() == 0, "eating resets the hunger counter");
        }
    }
    expect(saw_food, "a snake steered at the food eats within the step budget");
}

void testStarvation()
{
    // Circle in a corner of a big board, never eating. The snake must die of
    // hunger rather than be allowed to shuffle indefinitely - stalling has to
    // cost what dying costs, or the incentives reward doing nothing.
    SnakeEnv env(12, 12, 21);
    SnakeEnv::StepResult result{0.0f, false, false};
    int steps = 0;
    int limit = env.hungerLimit();

    // A length-one snake turning the same way forever traces a 2x2 loop and
    // cannot collide with itself, so the only thing that can end this is hunger.
    while (!result.done && steps < limit * 4)
    {
        result = env.step(steps % 2 == 0 ? SnakeEnv::Action::LEFT : SnakeEnv::Action::STRAIGHT);
        steps++;
        if (result.reward > 0.0f)
        {
            // Stumbled onto food; the loop moved, so this seed is unusable.
            break;
        }
    }

    expect(result.done && !env.won(), "a snake that never eats eventually dies");
    expect(result.reward == SnakeEnv::DEATH_REWARD, "starving pays what dying pays");
    expect(env.stepsSinceFood() >= limit, "it survives right up to the hunger limit");
}

void testWouldDieAgreesWithStepping()
{
    // The search asks wouldDie after every descent step and trusts the answer.
    // If it ever disagrees with what step() actually does, the tree is planning
    // against a game that does not exist - so the two are compared directly,
    // over a long random walk that reaches crowded boards and hunger.
    //
    // The counting is not decoration. Both sides now share blocksHead, so the
    // collision half of this agreement is true by construction and a walk that
    // only met walls would pass while testing nothing. What is still two
    // separate pieces of code is the hunger clock - wouldDie reads
    // steps_since_food_ + 1 against the limit, step increments and then reads -
    // and the tail-vacating rule, which only a crowded board reaches. Each of
    // those is counted, and each count is asserted non-trivial below.
    SnakeEnv env(8, 8, 1234);
    int checks = 0;
    int disagreements = 0;
    int fatal_moves = 0;
    int surviving_moves = 0;
    int tail_entries = 0;   // legally entering the cell the tail is leaving
    int hunger_deaths = 0;  // fatal for no reason but the clock
    unsigned int cursor = 99;

    // Two phases, because no single steering rule reaches both rules at once: a
    // snake that chases food grows a tail to meet but keeps resetting its hunger
    // clock, and a snake that ignores food starves on schedule but stays one
    // segment long forever. Measured both ways - chasing gave 707 tail entries
    // and 0 hunger deaths, so the walk runs once each way.
    const int phase_length = 20000;
    for (int step = 0; step < 2 * phase_length; step++)
    {
        const bool chase_food = step < phase_length;
        if (env.done())
        {
            env.reset();
            continue;
        }
        int survivor_count = 0;
        SnakeEnv::Action survivors[SnakeEnv::ACTION_COUNT];
        for (int action = 0; action < SnakeEnv::ACTION_COUNT; action++)
        {
            const SnakeEnv::Action chosen = static_cast<SnakeEnv::Action>(action);
            SnakeEnv probe = env;
            SnakeEnv::StepResult outcome = probe.step(chosen);
            const bool really_died = outcome.done && !outcome.won;
            const bool predicted = env.wouldDie(chosen);
            if (predicted != really_died)
            {
                disagreements++;
            }

            // Classify the position so the coverage assertions below can tell a
            // walk that exercised the interesting rules from one that did not.
            const Position next = env.headAfter(chosen);
            const bool clear_ground = insideBoard(env, next) && !bodyCoversCell(env, next);
            if (really_died)
            {
                fatal_moves++;
                if (clear_ground)
                {
                    hunger_deaths++;
                }
            }
            else
            {
                surviving_moves++;
                survivors[survivor_count++] = chosen;
                if (next == env.body().back() && env.body().size() > 1)
                {
                    tail_entries++;
                }
            }
            checks++;
        }

        // Steering, and it decides what this test covers. A uniformly random
        // walk on an 8x8 board dies against a wall at length one, so it reaches
        // neither the tail-vacating rule nor the hunger clock - measured, and
        // the first version of this walk scored zero hunger deaths in 55836
        // checks while claiming to test them. So: prefer a move that survives,
        // and chase the food only some of the time. Surviving keeps games long
        // enough to starve; chasing grows the body enough to meet its own tail.
        // The survivor list comes from stepping a copy, not from wouldDie, so
        // the thing under test is not steering its own examination.
        cursor = cursor * 1664525u + 1013904223u;
        SnakeEnv::Action move = static_cast<SnakeEnv::Action>((cursor >> 16) % 3);
        if (survivor_count > 0)
        {
            move = survivors[(cursor >> 16) % static_cast<unsigned int>(survivor_count)];
            if (chase_food && (cursor >> 24) % 4 != 0)
            {
                int best_distance = 1 << 30;
                for (int index = 0; index < survivor_count; index++)
                {
                    const Position next = env.headAfter(survivors[index]);
                    const int distance =
                        std::abs(next.x - env.food().x) + std::abs(next.y - env.food().y);
                    if (distance < best_distance)
                    {
                        best_distance = distance;
                        move = survivors[index];
                    }
                }
            }
        }
        env.step(move);
    }

    expect(checks > 10000, "the walk exercised enough positions to be worth trusting");
    expect(disagreements == 0, "wouldDie agrees with stepping on every position visited");
    expect(fatal_moves > 0 && surviving_moves > 0,
           "the walk saw both fatal and surviving moves, so the answer is not a constant");
    expect(tail_entries > 0, "the walk entered the cell the tail vacates, so that rule was tested");
    expect(hunger_deaths > 0, "the walk starved on open ground, so the hunger clock was tested");
    std::cout << "        " << checks << " checks: " << fatal_moves << " fatal, "
              << surviving_moves << " surviving, " << tail_entries << " tail entries, "
              << hunger_deaths << " hunger deaths" << std::endl;
    if (disagreements > 0)
    {
        std::cout << "        " << disagreements << " of " << checks << " disagreed" << std::endl;
    }
}

void testHungerBoundaryIsExact()
{
    // The off-by-one wouldDie is most likely to carry. It predicts one tick
    // ahead of a counter that step() increments, so the two read the limit at
    // different values and only the last two ticks before starvation tell a
    // correct pair from a shifted one. The walk above reaches this by accident;
    // this pins the exact tick.
    SnakeEnv env(12, 12, 21);
    const int limit = env.hungerLimit();

    // A length-one snake alternating left and straight traces a 2x2 loop, so it
    // cannot collide with itself and hunger is the only thing that can end it.
    int survivable_predictions = 0;
    bool predicted_death_early = false;
    bool ate_by_accident = false;
    int steps = 0;

    while (steps < limit - 1)
    {
        const SnakeEnv::Action chosen =
            (steps % 2 == 0) ? SnakeEnv::Action::LEFT : SnakeEnv::Action::STRAIGHT;
        if (env.wouldDie(chosen))
        {
            predicted_death_early = true;
            break;
        }
        survivable_predictions++;
        if (env.step(chosen).reward > 0.0f)
        {
            ate_by_accident = true;  // the loop moved; this seed cannot be used
            break;
        }
        steps++;
    }

    expect(!ate_by_accident, "the starvation loop never stumbled onto food");
    expect(!predicted_death_early, "wouldDie called no move fatal before the hunger limit");
    expect(survivable_predictions == limit - 1,
           "it predicted survival on every tick up to the last one");
    expect(env.stepsSinceFood() == limit - 1, "the clock stands one tick short of the limit");

    // The tick that starves. wouldDie has to say so before it is taken, and
    // step has to agree - these are separate lines of code reading the same
    // counter one increment apart.
    const SnakeEnv::Action last = SnakeEnv::Action::LEFT;
    const bool predicted = env.wouldDie(last);
    const SnakeEnv::StepResult outcome = env.step(last);
    expect(predicted, "wouldDie calls the starving tick fatal before it is taken");
    expect(outcome.done && !outcome.won, "and stepping it does starve the snake");
    expect(outcome.reward == SnakeEnv::DEATH_REWARD, "paying what dying pays");
}

void testFoodNeverSitsOnTheBody()
{
    // blocksHead resolves eating against collision, and the branch that would
    // matter - the food occupying the tail cell - is unreachable only because
    // food is never placed on the snake at all. That is an assertion inside the
    // environment, which compiles out of the release builds that train, so the
    // invariant it rests on is checked here over real play rather than assumed
    // from a reading of spawnFood.
    SnakeEnv env(8, 8, 555);
    int positions = 0;
    int violations = 0;
    int meals = 0;
    unsigned int cursor = 7;

    for (int step = 0; step < 20000; step++)
    {
        if (env.done())
        {
            env.reset();
            continue;
        }
        if (bodyCoversCell(env, env.food()))
        {
            violations++;
        }
        positions++;

        // Chase the food, so the snake actually grows and the board crowds -
        // a random walk on an 8x8 board mostly dies at length one, and the
        // invariant is only interesting once the body is long.
        SnakeEnv::Action chosen = SnakeEnv::Action::STRAIGHT;
        int best_distance = 1 << 30;
        for (int action = 0; action < SnakeEnv::ACTION_COUNT; action++)
        {
            const SnakeEnv::Action candidate = static_cast<SnakeEnv::Action>(action);
            if (env.wouldDie(candidate))
            {
                continue;
            }
            const Position next = env.headAfter(candidate);
            const int distance =
                std::abs(next.x - env.food().x) + std::abs(next.y - env.food().y);
            if (distance < best_distance)
            {
                best_distance = distance;
                chosen = candidate;
            }
        }
        cursor = cursor * 1664525u + 1013904223u;
        if ((cursor >> 24) % 8 == 0)
        {
            // Occasional random move, so the walk is not one deterministic line.
            chosen = static_cast<SnakeEnv::Action>((cursor >> 16) % SnakeEnv::ACTION_COUNT);
        }
        if (env.step(chosen).reward > 0.0f)
        {
            meals++;
        }
    }

    expect(violations == 0, "food never occupies a body cell at any point in play");
    expect(positions > 10000, "the invariant was checked on enough positions to matter");
    expect(meals > 100, "the walk ate often enough to crowd the board");
    std::cout << "        " << positions << " positions, " << meals << " meals, " << violations
              << " violations" << std::endl;
}

void testEncoding()
{
    SnakeEnv env(10, 10, 5);
    std::vector<float> planes(env.encodedSize(), -1.0f);
    env.encode(planes.data());

    int cells = env.cellCount();
    float body_sum = 0.0f;
    float head_sum = 0.0f;
    float food_sum = 0.0f;
    for (int cell = 0; cell < cells; cell++)
    {
        body_sum += planes[0 * cells + cell];
        head_sum += planes[1 * cells + cell];
        food_sum += planes[2 * cells + cell];
    }
    expect(body_sum == (float)env.body().size(), "the body plane marks exactly the body");
    expect(head_sum == 1.0f, "the head plane marks exactly one cell");
    expect(food_sum == 1.0f, "the food plane marks exactly one cell");

    int heading_planes_set = 0;
    for (int plane = 4; plane < SnakeEnv::PLANE_COUNT; plane++)
    {
        float sum = 0.0f;
        for (int cell = 0; cell < cells; cell++)
        {
            sum += planes[plane * cells + cell];
        }
        if (sum == (float)cells)
        {
            heading_planes_set++;
        }
        else if (sum != 0.0f)
        {
            heading_planes_set = -1000;  // a heading plane must be all or nothing
        }
    }
    expect(heading_planes_set == 1, "exactly one heading plane is set, and it is constant");
}

// The strongest available check on the whole transition function: an
// independent winner drives the environment to a full board. It exercises
// tail-following, which is where an off-by-one in collision hides, and win
// detection, which nothing else reaches.
void testWinnableByCycle()
{
    const int size = 6;
    SnakeEnv env(size, size, 99);
    HamiltonianCycle cycle(size, size);
    if (!cycle.generateCycle())
    {
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
    while (!actionTowards(env, cycle.getNext(env.body()[0]), entry) && alignment_steps < 4)
    {
        env.step(SnakeEnv::Action::LEFT);
        alignment_steps++;
    }
    expect(alignment_steps < 4, "the snake can turn onto the cycle within one lap of turns");

    int guard = 0;
    int guard_limit = 40 * env.cellCount() * env.foodsToWin();
    bool steering_failed = false;
    while (!env.done() && guard++ < guard_limit)
    {
        Position next = cycle.getNext(env.body()[0]);
        SnakeEnv::Action action = SnakeEnv::Action::STRAIGHT;
        if (!actionTowards(env, next, action))
        {
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

int main()
{
    std::cout << "SnakeEnv properties" << std::endl;
    testResetInvariants();
    testDeterminism();
    testTurning();
    testWallDeath();
    testStarvation();
    testWouldDieAgreesWithStepping();
    testHungerBoundaryIsExact();
    testFoodNeverSitsOnTheBody();
    testEatingReward();
    testEncoding();
    testWinnableByCycle();

    std::cout << std::endl;
    if (failures == 0)
    {
        std::cout << "All checks passed." << std::endl;
        return 0;
    }
    std::cout << failures << " check(s) failed." << std::endl;
    return 1;
}
