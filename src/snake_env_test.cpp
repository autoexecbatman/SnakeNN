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

// A budget no test can reach, so the clock plane is full and every assertion
// here measures what it measured before the clock existed. The tests that are
// about the clock set their own.
constexpr int TEST_STEP_LIMIT = 1000000;

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
    const SnakeEnv::Action actions[] = { SnakeEnv::Action::STRAIGHT, SnakeEnv::Action::LEFT,
                                         SnakeEnv::Action::RIGHT };
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
    SnakeEnv env(20, 20, 7, TEST_STEP_LIMIT);

    expect(env.body().size() == 1, "a fresh board holds a single segment");
    expect(env.score() == 0 && env.steps() == 0, "score and step count start at zero");
    expect(!env.done() && !env.won(), "a fresh board is neither finished nor won");
    expect(!bodyCoversCell(env, env.food()), "food never spawns on the snake");
    expect(env.foodsToWin() == 399, "a 20x20 board takes 399 foods to fill");
}

void testDeterminism()
{
    SnakeEnv first(12, 12, 4242, TEST_STEP_LIMIT);
    SnakeEnv second(12, 12, 4242, TEST_STEP_LIMIT);

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
    SnakeEnv env(20, 20, 1, TEST_STEP_LIMIT);
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
    SnakeEnv env(8, 8, 3, TEST_STEP_LIMIT);
    SnakeEnv::StepResult result{ 0.0f, false, false };
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
    SnakeEnv env(10, 10, 11, TEST_STEP_LIMIT);
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
        const SnakeEnv::Action actions[] = { SnakeEnv::Action::STRAIGHT, SnakeEnv::Action::LEFT,
                                             SnakeEnv::Action::RIGHT };
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
    SnakeEnv env(12, 12, 21, TEST_STEP_LIMIT);
    SnakeEnv::StepResult result{ 0.0f, false, false };
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
    SnakeEnv env(8, 8, 1234, TEST_STEP_LIMIT);
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
    std::cout << "        " << checks << " checks: " << fatal_moves << " fatal, " << surviving_moves
              << " surviving, " << tail_entries << " tail entries, " << hunger_deaths
              << " hunger deaths" << std::endl;
    if (disagreements > 0)
    {
        std::cout << "        " << disagreements << " of " << checks << " disagreed" << std::endl;
    }
}

// Rotate a fresh environment onto a chosen heading. Turning is the only way to
// change it, and a one-segment snake cannot collide with itself while turning,
// so this is safe for any quarter_turns.
bool faceHeading(SnakeEnv& env, int quarter_turns)
{
    for (int turn = 0; turn < quarter_turns; turn++)
    {
        env.step(SnakeEnv::Action::LEFT);
        if (env.done())
        {
            return false;
        }
    }
    return true;
}

void testTurnAlgebra()
{
    // testTurning checks the three actions from a single heading - the one a
    // fresh board happens to start on. That leaves three quarters of turnLeft
    // and turnRight unexercised, and each is a hand-written four-case switch
    // where a transposed pair would go unnoticed. These properties hold for
    // every heading, so they are checked from all four.
    //
    // Why it matters beyond tidiness: the relative action space is what removes
    // the reverse move from the game. If either turn ever produced the opposite
    // heading, the snake could reverse into its own neck and the MDP the search
    // solves would stop matching the one being played.
    int headings_checked = 0;
    bool identity_holds = true;
    bool four_lefts_return = true;
    bool four_rights_return = true;
    bool turns_disagree = true;
    bool turns_compose_to_reverse = true;
    bool reverse_never_offered = true;
    bool straight_is_fixed = true;
    bool setup_failed = false;

    for (int start = 0; start < 4; start++)
    {
        SnakeEnv env(20, 20, 1000 + start, TEST_STEP_LIMIT);
        if (!faceHeading(env, start))
        {
            setup_failed = true;
            break;
        }
        const Direction heading = env.heading();
        headings_checked++;

        // Straight is the identity on the heading, by definition of relative.
        if (env.headingAfter(SnakeEnv::Action::STRAIGHT) != heading)
        {
            straight_is_fixed = false;
        }

        // The two turns are distinct, so no relative action aliases another.
        const Direction left_once = env.headingAfter(SnakeEnv::Action::LEFT);
        const Direction right_once = env.headingAfter(SnakeEnv::Action::RIGHT);
        if (left_once == right_once || left_once == heading || right_once == heading)
        {
            turns_disagree = false;
        }

        // Two turns the same way give the reverse, and the two routes to it
        // agree. This is also how the reverse heading is named without adding a
        // function for it.
        SnakeEnv twice_left(20, 20, 2000 + start, TEST_STEP_LIMIT);
        SnakeEnv twice_right(20, 20, 3000 + start, TEST_STEP_LIMIT);
        if (!faceHeading(twice_left, start) || !faceHeading(twice_right, start))
        {
            setup_failed = true;
            break;
        }
        twice_left.step(SnakeEnv::Action::LEFT);
        twice_left.step(SnakeEnv::Action::LEFT);
        twice_right.step(SnakeEnv::Action::RIGHT);
        twice_right.step(SnakeEnv::Action::RIGHT);
        const Direction reverse = twice_left.heading();
        if (reverse == heading || reverse != twice_right.heading())
        {
            turns_compose_to_reverse = false;
        }

        // The reverse is unreachable in one move. This is the property the whole
        // relative action space exists to provide.
        if (left_once == reverse || right_once == reverse ||
            env.headingAfter(SnakeEnv::Action::STRAIGHT) == reverse)
        {
            reverse_never_offered = false;
        }

        // Four quarter turns either way return to where they started, which is
        // what makes each switch a rotation rather than an arbitrary mapping.
        SnakeEnv lap_left(20, 20, 4000 + start, TEST_STEP_LIMIT);
        SnakeEnv lap_right(20, 20, 5000 + start, TEST_STEP_LIMIT);
        if (!faceHeading(lap_left, start) || !faceHeading(lap_right, start))
        {
            setup_failed = true;
            break;
        }
        for (int turn = 0; turn < 4; turn++)
        {
            lap_left.step(SnakeEnv::Action::LEFT);
            lap_right.step(SnakeEnv::Action::RIGHT);
        }
        if (lap_left.heading() != heading)
        {
            four_lefts_return = false;
        }
        if (lap_right.heading() != heading)
        {
            four_rights_return = false;
        }

        // A left then a right is the identity, so the two are inverses and not
        // merely different.
        SnakeEnv there_and_back(20, 20, 6000 + start, TEST_STEP_LIMIT);
        if (!faceHeading(there_and_back, start))
        {
            setup_failed = true;
            break;
        }
        there_and_back.step(SnakeEnv::Action::LEFT);
        there_and_back.step(SnakeEnv::Action::RIGHT);
        if (there_and_back.heading() != heading)
        {
            identity_holds = false;
        }
    }

    expect(!setup_failed, "the rotation setup never killed the snake");
    expect(headings_checked == 4, "all four headings were reached and checked");
    expect(straight_is_fixed, "going straight leaves the heading alone, from every heading");
    expect(turns_disagree, "left and right differ from each other and from straight");
    expect(turns_compose_to_reverse, "two turns the same way give the reverse, by either route");
    expect(reverse_never_offered, "no single action produces the reverse heading");
    expect(four_lefts_return, "four left turns return to the starting heading");
    expect(four_rights_return, "four right turns return to the starting heading");
    expect(identity_holds, "a left turn followed by a right turn is the identity");
}

void testHeadingAfterDependsOnlyOnTheHeading()
{
    // headingAfter is meant to be a pure function of the heading and the action -
    // nothing about the body, the score, the hunger clock or the food may enter
    // it. Nothing checked that. The turn algebra above exercises it only on
    // fresh one-segment boards, so a version that consulted the body length
    // would pass every property there and diverge in real play, which is exactly
    // where the search uses it.
    //
    // So: rotate throwaway boards to each heading and keep them as the reference
    // answer, then walk a real game to a long body and a wound-up hunger clock
    // and require the answers to match at every position.
    Direction reference_heading[4];
    Direction reference_answer[4][SnakeEnv::ACTION_COUNT];
    for (int turns = 0; turns < 4; turns++)
    {
        SnakeEnv fresh(20, 20, 7100 + turns, TEST_STEP_LIMIT);
        if (!faceHeading(fresh, turns))
        {
            expect(false, "the reference rotation never killed the snake");
            return;
        }
        reference_heading[turns] = fresh.heading();
        for (int action = 0; action < SnakeEnv::ACTION_COUNT; action++)
        {
            reference_answer[turns][action] =
                fresh.headingAfter(static_cast<SnakeEnv::Action>(action));
        }
    }

    SnakeEnv env(8, 8, 4711, TEST_STEP_LIMIT);
    int comparisons = 0;
    int mismatches = 0;
    int unknown_heading = 0;
    size_t longest_body = 0;
    int highest_hunger = 0;
    unsigned int cursor = 17;

    for (int step = 0; step < 20000; step++)
    {
        if (env.done())
        {
            env.reset();
            continue;
        }

        int which = -1;
        for (int turns = 0; turns < 4; turns++)
        {
            if (reference_heading[turns] == env.heading())
            {
                which = turns;
            }
        }
        if (which < 0)
        {
            unknown_heading++;
        }
        else
        {
            for (int action = 0; action < SnakeEnv::ACTION_COUNT; action++)
            {
                const Direction expected = reference_answer[which][action];
                if (env.headingAfter(static_cast<SnakeEnv::Action>(action)) != expected)
                {
                    mismatches++;
                }
                comparisons++;
            }
        }

        longest_body = (env.body().size() > longest_body) ? env.body().size() : longest_body;
        highest_hunger =
            (env.stepsSinceFood() > highest_hunger) ? env.stepsSinceFood() : highest_hunger;

        // Survive first, chase food most of the time - the same steering the
        // other walks use, for the same reason: a random walk here stays one
        // segment long and would compare only the state the reference already is.
        int survivor_count = 0;
        SnakeEnv::Action survivors[SnakeEnv::ACTION_COUNT];
        for (int action = 0; action < SnakeEnv::ACTION_COUNT; action++)
        {
            const SnakeEnv::Action candidate = static_cast<SnakeEnv::Action>(action);
            SnakeEnv probe = env;
            const SnakeEnv::StepResult outcome = probe.step(candidate);
            if (!outcome.done || outcome.won)
            {
                survivors[survivor_count++] = candidate;
            }
        }
        cursor = cursor * 1664525u + 1013904223u;
        SnakeEnv::Action move = static_cast<SnakeEnv::Action>((cursor >> 16) % 3);
        if (survivor_count > 0)
        {
            move = survivors[(cursor >> 16) % static_cast<unsigned int>(survivor_count)];
            if ((cursor >> 24) % 4 != 0)
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

    expect(unknown_heading == 0, "every heading reached in play was one of the four");
    expect(mismatches == 0, "headingAfter answers from the heading alone, whatever the board");
    expect(comparisons > 10000, "enough positions were compared to be worth trusting");
    expect(longest_body > 10,
           "the walk grew a long body, so the answer was checked away from reset");
    expect(highest_hunger > 20,
           "the walk wound the hunger clock up, so that state was covered too");
    std::cout << "        " << comparisons << " comparisons, longest body " << longest_body
              << ", highest hunger " << highest_hunger << std::endl;
}

void testHeadAfterPredictsWhereTheHeadLands()
{
    // headAfter is a query the search leans on hard: it orders children by where
    // each move lands before committing to expand them, and blocksHead indexes
    // occupancy_ by the cell it returns. So the claim is not merely that it
    // computes some neighbour - it is that the cell it names is the cell step()
    // actually puts the head in, for every action that does not kill.
    SnakeEnv env(8, 8, 8675, TEST_STEP_LIMIT);
    int predictions = 0;
    int mismatches = 0;
    int non_adjacent = 0;
    int duplicate_targets = 0;
    int off_board_targets = 0;  // near a wall headAfter must answer outside the grid
    int after_eating = 0;       // growth moves the body, so check this case explicitly
    unsigned int cursor = 31;

    for (int step = 0; step < 20000; step++)
    {
        if (env.done())
        {
            env.reset();
            continue;
        }

        Position targets[SnakeEnv::ACTION_COUNT];
        int survivor_count = 0;
        SnakeEnv::Action survivors[SnakeEnv::ACTION_COUNT];

        for (int action = 0; action < SnakeEnv::ACTION_COUNT; action++)
        {
            const SnakeEnv::Action chosen = static_cast<SnakeEnv::Action>(action);
            const Position target = env.headAfter(chosen);
            targets[action] = target;

            // One orthogonal step from the current head, and nothing else.
            const int moved_x = target.x - env.body()[0].x;
            const int moved_y = target.y - env.body()[0].y;
            if (std::abs(moved_x) + std::abs(moved_y) != 1)
            {
                non_adjacent++;
            }
            if (!insideBoard(env, target))
            {
                off_board_targets++;
            }

            // Where step() really puts the head, from a copy so the real game is
            // untouched. Only meaningful when the move survives - a fatal move
            // leaves the body wherever it was.
            SnakeEnv probe = env;
            const SnakeEnv::StepResult outcome = probe.step(chosen);
            const bool survived = !outcome.done || outcome.won;
            if (survived)
            {
                if (!(probe.body()[0] == target))
                {
                    mismatches++;
                }
                if (outcome.reward > 0.0f)
                {
                    after_eating++;
                }
                survivors[survivor_count++] = chosen;
            }
            predictions++;
        }

        // Three different actions turn the head three different ways, so they
        // cannot name the same cell. If two ever agreed, one relative action
        // would be aliasing another and the search would have a phantom child.
        if (targets[0] == targets[1] || targets[1] == targets[2] || targets[0] == targets[2])
        {
            duplicate_targets++;
        }

        cursor = cursor * 1664525u + 1013904223u;
        SnakeEnv::Action move = static_cast<SnakeEnv::Action>((cursor >> 16) % 3);
        if (survivor_count > 0)
        {
            move = survivors[(cursor >> 16) % static_cast<unsigned int>(survivor_count)];
            if ((cursor >> 24) % 3 != 0)
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

    expect(mismatches == 0, "headAfter names the cell step actually puts the head in");
    expect(non_adjacent == 0, "headAfter always moves exactly one orthogonal step");
    expect(duplicate_targets == 0, "the three actions never name the same cell");
    expect(predictions > 10000, "enough predictions were checked to be worth trusting");
    expect(off_board_targets > 0,
           "the walk asked next to a wall, so the off-grid answer was exercised");
    expect(after_eating > 100, "the walk checked the prediction on moves that ate and grew");
    std::cout << "        " << predictions << " predictions, " << off_board_targets
              << " off board, " << after_eating << " while eating" << std::endl;
}

void testHungerBoundaryIsExact()
{
    // The off-by-one wouldDie is most likely to carry. It predicts one tick
    // ahead of a counter that step() increments, so the two read the limit at
    // different values and only the last two ticks before starvation tell a
    // correct pair from a shifted one. The walk above reaches this by accident;
    // this pins the exact tick.
    SnakeEnv env(12, 12, 21, TEST_STEP_LIMIT);
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
    SnakeEnv env(8, 8, 555, TEST_STEP_LIMIT);
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
            const int distance = std::abs(next.x - env.food().x) + std::abs(next.y - env.food().y);
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
    SnakeEnv env(10, 10, 5, TEST_STEP_LIMIT);
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

    // Planes 4 to 7 are the headings. Bounded by name rather than by PLANE_COUNT,
    // which stopped meaning "the last plane is a heading" when the clock arrived -
    // a full clock plane is indistinguishable from a heading plane by this test.
    constexpr int FIRST_HEADING_PLANE = 4;
    constexpr int CLOCK_PLANE = 8;
    int heading_planes_set = 0;
    for (int plane = FIRST_HEADING_PLANE; plane < CLOCK_PLANE; plane++)
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

    // The clock, constant across the board because a convolution has no other way
    // to be given a scalar.
    float clock_sum = 0.0f;
    bool clock_is_constant = true;
    for (int cell = 0; cell < cells; cell++)
    {
        const float value = planes[CLOCK_PLANE * cells + cell];
        clock_sum += value;
        clock_is_constant = clock_is_constant && value == planes[CLOCK_PLANE * cells];
    }
    expect(clock_is_constant, "the clock plane holds one value everywhere");
    expect(clock_sum == env.budgetRemaining() * (float)cells,
           "and that value is the budget the environment reports");
}

// The strongest available check on the whole transition function: an
// independent winner drives the environment to a full board. It exercises
// tail-following, which is where an off-by-one in collision hides, and win
// detection, which nothing else reaches.
// Expected values are the fraction written out - remaining over limit - not read
// back from the environment.
void testTheBudgetCountsDownWithTheSteps()
{
    SnakeEnv env(6, 6, 1, 100);
    expect(env.stepLimit() == 100, "the environment carries the limit it was given");
    expect(env.budgetRemaining() == 1.0f, "a fresh game has spent none of its budget");

    env.step(SnakeEnv::Action::STRAIGHT);
    expect(env.budgetRemaining() == 0.99f, "one step of a hundred leaves 99 percent");

    for (int step = 0; step < 49 && !env.done(); step++)
    {
        env.step(SnakeEnv::Action::STRAIGHT);
    }
    if (!env.done())
    {
        expect(env.budgetRemaining() == 0.5f, "fifty steps of a hundred leaves half");
    }

    // A caller may run an episode past its limit - the limit bounds the encoding,
    // not termination - and the budget must not go negative when it does.
    SnakeEnv brief(6, 6, 1, 2);
    for (int step = 0; step < 6 && !brief.done(); step++)
    {
        brief.step(SnakeEnv::Action::STRAIGHT);
    }
    expect(brief.budgetRemaining() == 0.0f, "a spent budget reads zero, never below it");

    // The snapshot carries the same number, because it outlives the environment
    // and the replay buffer has no way back to the step limit.
    SnakeEnv counted(6, 6, 1, 10);
    counted.step(SnakeEnv::Action::STRAIGHT);
    counted.step(SnakeEnv::Action::STRAIGHT);
    expect(counted.snapshot().budget_remaining == 0.8f,
           "the snapshot carries the budget, two steps of ten being 0.8");
}

void testStepLimitIsRequired()
{
    bool refused = false;
    try
    {
        SnakeEnv env(6, 6, 1, 0);
    }
    catch (const std::invalid_argument&)
    {
        refused = true;
    }
    expect(refused, "a step limit of zero is refused - every game would start spent");
}

void testWinnableByCycle()
{
    const int size = 6;
    SnakeEnv env(size, size, 99, TEST_STEP_LIMIT);
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
    testTurnAlgebra();
    testHeadingAfterDependsOnlyOnTheHeading();
    testWouldDieAgreesWithStepping();
    testHeadAfterPredictsWhereTheHeadLands();
    testHungerBoundaryIsExact();
    testFoodNeverSitsOnTheBody();
    testEatingReward();
    testEncoding();
    testTheBudgetCountsDownWithTheSteps();
    testStepLimitIsRequired();
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
