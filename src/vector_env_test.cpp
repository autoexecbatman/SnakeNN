#include "vector_env.h"
#include <iostream>
#include <string>
#include <vector>

// The batch has one job: be indistinguishable from stepping each game on its
// own. Everything the learner concludes rests on that, and a divergence here
// would show up as a mysteriously bad policy rather than as a bug.

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

void testMatchesIndividualGames() {
    const int count = 16;
    const int size = 8;
    const unsigned int base_seed = 2024;

    VectorEnv batch(count, size, size, base_seed);
    std::vector<SnakeEnv> singles;
    for (int index = 0; index < count; index++) {
        singles.emplace_back(size, size, base_seed + index);
    }

    std::vector<SnakeEnv::Action> actions(count);
    std::vector<SnakeEnv::StepResult> results(count);

    bool matched = true;
    for (int step = 0; step < 300 && matched; step++) {
        for (int index = 0; index < count; index++) {
            actions[index] = static_cast<SnakeEnv::Action>((step + index) % SnakeEnv::ACTION_COUNT);
        }

        batch.step(actions.data(), results.data());
        for (int index = 0; index < count; index++) {
            if (!singles[index].done()) {
                singles[index].step(actions[index]);
            }
        }

        for (int index = 0; index < count; index++) {
            const SnakeEnv& left = batch.env(index);
            const SnakeEnv& right = singles[index];
            if (left.score() != right.score() || left.steps() != right.steps() ||
                left.done() != right.done() || !(left.food() == right.food()) ||
                left.body().size() != right.body().size()) {
                matched = false;
                break;
            }
        }
    }
    expect(matched, "a batched game is identical to the same game stepped alone");
}

void testFinishedGamesAreNotAdvanced() {
    VectorEnv batch(4, 5, 5, 77);
    std::vector<SnakeEnv::Action> actions(4, SnakeEnv::Action::STRAIGHT);
    std::vector<SnakeEnv::StepResult> results(4);

    // Straight into the wall finishes every game, then keeps being called.
    for (int step = 0; step < 40; step++) {
        batch.step(actions.data(), results.data());
    }

    bool all_done = true;
    bool reported_done = true;
    int steps_after_finishing = 0;
    for (int index = 0; index < batch.count(); index++) {
        all_done = all_done && batch.env(index).done();
        reported_done = reported_done && results[index].done;
        steps_after_finishing = std::max(steps_after_finishing, batch.env(index).steps());
    }

    expect(all_done, "games driven into the wall finish");
    expect(reported_done, "a finished game keeps reporting that it is finished");
    expect(steps_after_finishing < 40, "stepping a finished game does not advance it");
}

void testResetIsLocal() {
    VectorEnv batch(8, 6, 6, 5);
    std::vector<SnakeEnv::Action> actions(8, SnakeEnv::Action::STRAIGHT);
    std::vector<SnakeEnv::StepResult> results(8);
    for (int step = 0; step < 2; step++) {
        batch.step(actions.data(), results.data());
    }

    int steps_elsewhere_before = batch.env(3).steps();
    batch.resetOne(0);

    expect(batch.env(0).steps() == 0, "the reset game starts over");
    expect(batch.env(3).steps() == steps_elsewhere_before, "its neighbours are untouched");
}

void testEncodingMatchesPerGame() {
    const int count = 6;
    const int size = 7;
    VectorEnv batch(count, size, size, 31);

    std::vector<SnakeEnv::Action> actions(count, SnakeEnv::Action::LEFT);
    std::vector<SnakeEnv::StepResult> results(count);
    batch.step(actions.data(), results.data());

    std::vector<float> whole(batch.encodedSizeTotal(), -1.0f);
    batch.encodeAll(whole.data());

    int stride = batch.encodedSizePerEnv();
    std::vector<float> single(stride, -2.0f);
    bool identical = true;
    for (int index = 0; index < count && identical; index++) {
        batch.env(index).encode(single.data());
        for (int value = 0; value < stride; value++) {
            if (whole[index * stride + value] != single[value]) {
                identical = false;
                break;
            }
        }
    }
    expect(identical, "the batch encoding is each game's own encoding, laid end to end");
}

}  // namespace

int main() {
    std::cout << "VectorEnv properties" << std::endl;
    testMatchesIndividualGames();
    testFinishedGamesAreNotAdvanced();
    testResetIsLocal();
    testEncodingMatchesPerGame();

    std::cout << std::endl;
    if (failures == 0) {
        std::cout << "All checks passed." << std::endl;
        return 0;
    }
    std::cout << failures << " check(s) failed." << std::endl;
    return 1;
}
