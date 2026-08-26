#include <cassert>
#include <format>
#include <iostream>
#include <string>
#include <type_traits>
#include <vector>

#include "trainer_options.h"

// Settings is a value: it holds two std::strings and a pile of ints, nothing
// owns a resource, and the compiler's copy is the right one. Pinned so that a
// later member cannot quietly make it non-copyable and leave the rule-of-zero
// comment on the struct false.
static_assert(std::is_copy_constructible<trainer::Settings>::value,
              "Settings is a value type - it must stay copyable");
static_assert(std::is_copy_assignable<trainer::Settings>::value,
              "Settings is a value type - it must stay assignable");
static_assert(std::is_nothrow_move_constructible<trainer::Settings>::value,
              "Settings should move without allocating");

// The paper's constants are pinned in az_parameters_test.cpp, which owns them.

namespace
{

// Checks that did not hold. main prints the count and returns 1 when it is non-zero.
int failures = 0;
// Flags exercised. Printed at the end so a flag added to the parser and not to this file
// shows up as a count that did not move.
int flags_checked = 0;
// Refusals exercised, counted for the same reason.
int rejections_checked = 0;

// Reports one property and counts a failure.
void expect(bool condition, const std::string& description)
{
    if (condition)
    {
        std::cout << std::format("  PASS  {}\n", description);
    }
    else
    {
        std::cout << std::format("  FAIL  {}\n", description);
        failures++;
    }
}

// Parses an argument list as the trainer would.
trainer::Settings parse(const std::vector<const char*>& arguments)
{
    return trainer::parseArguments(
        std::span<const char* const>(arguments.data(), arguments.size()));
}

// Whether parsing an argument list throws, without saying what it threw.
bool throwsOnParse(const std::vector<const char*>& arguments)
{
    try
    {
        parse(arguments);
    }
    catch (const std::exception&)
    {
        return true;
    }
    return false;
}

// Every field, one per line. This is what makes the flag table checkable: a
// branch that reads --batch and writes batches_per_iteration is the defect this
// file exists to catch, and it is invisible to a test that only looks at the
// field the flag was supposed to touch.
std::string describe(const trainer::Settings& settings)
{
    return std::format(
        "board={}\niterations={}\nstart_iteration={}\ngames_per_iteration={}\nsimulations={}\n"
        "step_limit_override={}\nchannels={}\nblocks={}\nbatch_size={}\nbatches_per_iteration={}\n"
        "replay_bytes={}\nseed={}\ncheckpoint={}\nresume={}\nledger_path={}\n"
        "exploration_epsilon={}\ndecisive_share={}\nweight_decay={}\n"
        "final_learning_rate_fraction={}\nfull_search_fraction={}\n"
        "fast_simulation_fraction={}\ndoom_label_from_trajectory={}\ndeath_cap={}\n",
        settings.board, settings.iterations, settings.start_iteration, settings.games_per_iteration,
        settings.simulations,
        settings.step_limit_override ? std::format("{}", *settings.step_limit_override) : "none",
        settings.channels, settings.blocks, settings.batch_size, settings.batches_per_iteration,
        settings.replay_bytes, settings.seed, settings.checkpoint, settings.resume,
        settings.ledger_path, settings.exploration_epsilon, settings.decisive_share,
        settings.weight_decay, settings.final_learning_rate_fraction, settings.full_search_fraction,
        settings.fast_simulation_fraction, settings.doom_label_from_trajectory, settings.death_cap);
}

// Splits a description into its lines so exactly one may be required to differ.
std::vector<std::string> lines(const std::string& text)
{
    std::vector<std::string> result;
    size_t start = 0;
    while (start < text.size())
    {
        size_t end = text.find('\n', start);
        assert(end != std::string::npos && "describe must terminate every line");
        result.push_back(text.substr(start, end - start));
        start = end + 1;
    }
    return result;
}

// One flag changes one field and leaves every other field alone.
void expectFlagSetsOnly(const char* flag, const char* value, const std::string& expected_line)
{
    flags_checked++;
    const std::vector<std::string> before = lines(describe(trainer::Settings{}));
    const std::vector<std::string> after = lines(describe(parse({ flag, value })));
    assert(before.size() == after.size() && "describe changed shape between two Settings");

    std::vector<std::string> changed;
    for (size_t index = 0; index < before.size(); index++)
    {
        if (before[index] != after[index])
        {
            changed.push_back(after[index]);
        }
    }

    if (changed.size() == 1 && changed[0] == expected_line)
    {
        std::cout << std::format("  PASS  {} {} sets {}\n", flag, value, expected_line);
        return;
    }
    std::cout << std::format("  FAIL  {} {} should set only {}, changed {} field(s):\n", flag,
                             value, expected_line, changed.size());
    for (const std::string& line : changed)
    {
        std::cout << std::format("            {}\n", line);
    }
    failures++;
}

// Checks that an argument list is refused, and counts the refusal.
void expectRejected(const std::vector<const char*>& arguments, const std::string& description)
{
    rejections_checked++;
    expect(throwsOnParse(arguments), description);
}

// The defaults are the ones the header states, written out rather than read back. A test
// that read them from the parser would agree with a wrong default forever.
void testDefaultsAreTheDocumentedOnes()
{
    const trainer::Settings settings;
    expect(settings.board == 6, "the default board is the small end of the curriculum");
    expect(settings.start_iteration == 1, "iterations are numbered from one");
    expect(!settings.step_limit_override.has_value(),
           "no step limit is given by default - it is derived, not defaulted to zero");
    expect(settings.replay_bytes == 1024u * 1024u * 1024u, "the replay cap is one gibibyte");
}

// The step limit follows the board unless overridden, so the two cannot disagree. It is
// part of the task rather than a detail: the cap decides whether a win is reachable.
void testStepLimitIsDerivedFromTheBoardUnlessGiven()
{
    trainer::Settings settings;
    settings.board = 10;
    // The paper's own number, from its own board size. Computed from the rule
    // rather than read back from the code: 12 steps per cell on 100 cells.
    expect(settings.stepLimit() == 1200, "a 10x10 board gets the paper's 1,200 steps");

    settings.board = 20;
    expect(settings.stepLimit() == 4800, "and 20x20 gets four times that, since cost is area");

    settings.step_limit_override = 2400;
    expect(settings.stepLimit() == 2400, "an override is used verbatim");
    expect(settings.cellCount() == 400 && settings.foodsToWin() == 399,
           "cellCount and foodsToWin agree with a 20x20 board holding a one-segment snake");

    const trainer::Settings parsed = parse({ "--board", "10", "--step-limit", "77" });
    expect(parsed.stepLimit() == 77, "and parsing preserves the override rather than resolving it");
    expect(parsed.step_limit_override.has_value() && *parsed.step_limit_override == 77,
           "the override stays visible as an override");
    expect(parse({ "--board", "10" }).stepLimit() == 1200,
           "while an unset override still derives at parse time or after it");
}

// The paper's budget is 3,000 samples per game. Ours is batches times batch size
// divided by games, and the expected values here are that division written out.
void testTheGradientBudgetIsPerGameAndDerivable()
{
    trainer::Settings settings;
    settings.batches_per_iteration = 3000;
    settings.batch_size = 128;
    settings.games_per_iteration = 256;
    // 3000 * 128 / 256 = 1500, which is half the paper's and what runs 101 to 140
    // actually trained on.
    expect(settings.samplesPerGame() == 1500, "the budget in use through iteration 140 was 1,500");

    settings.batches_per_iteration = 6000;
    expect(settings.samplesPerGame() == 3000, "and twice the batches is the paper's 3,000");

    // An iteration that plays and does not train draws nothing.
    settings.batches_per_iteration = 0;
    expect(settings.samplesPerGame() == 0, "no batches is no samples, not a division by zero");

    // Asking for the paper's budget derives the batch count rather than making the
    // operator do the arithmetic at every launch.
    const trainer::Settings derived =
        parse({ "--games", "256", "--batch", "128", "--samples-per-game", "3000" });
    expect(derived.batches_per_iteration == 6000,
           "--samples-per-game 3000 over 256 games at batch 128 is 6,000 batches");
    expect(derived.samplesPerGame() == 3000, "and it reads back as the budget that was asked for");

    // The derivation uses the game count and batch size from the same command line,
    // whatever order they appear in.
    const trainer::Settings reordered =
        parse({ "--samples-per-game", "3000", "--batch", "64", "--games", "32" });
    expect(reordered.batches_per_iteration == 1500,
           "3000 * 32 / 64 = 1500, and the flags before it did not decide that");

    expectRejected({ "--samples-per-game", "3000", "--batches", "6000" },
                   "giving both the budget and the batch count is refused rather than one winning");
    expectRejected({ "--batches", "6000", "--samples-per-game", "3000" },
                   "and refused in the other order too");
    expectRejected({ "--samples-per-game", "0" }, "a budget of nothing is refused");
    expectRejected({ "--samples-per-game", "-1" }, "a negative budget is refused");
}

// Each flag writes its own field and disturbs no other. A parser that quietly set two
// fields would still accept every command line here.
void testEveryFlagSetsTheFieldItNames()
{
    expectFlagSetsOnly("--board", "12", "board=12");
    expectFlagSetsOnly("--iterations", "7", "iterations=7");
    expectFlagSetsOnly("--start-iteration", "111", "start_iteration=111");
    expectFlagSetsOnly("--games", "256", "games_per_iteration=256");
    expectFlagSetsOnly("--simulations", "200", "simulations=200");
    expectFlagSetsOnly("--step-limit", "2400", "step_limit_override=2400");
    expectFlagSetsOnly("--channels", "96", "channels=96");
    expectFlagSetsOnly("--blocks", "6", "blocks=6");
    expectFlagSetsOnly("--batch", "512", "batch_size=512");
    expectFlagSetsOnly("--batches", "3000", "batches_per_iteration=3000");
    expectFlagSetsOnly("--replay-mb", "2048", "replay_bytes=2147483648");
    expectFlagSetsOnly("--seed", "9", "seed=9");
    expectFlagSetsOnly("--checkpoint", "az10.pt", "checkpoint=az10.pt");
    expectFlagSetsOnly("--resume", "az10_iter110.pt", "resume=az10_iter110.pt");
    // Relative to the launch directory, and the launch directory is build/Release,
    // which git ignores. A run whose cost lands there is as lost as the ones this
    // ledger exists to replace, so the path is given rather than assumed.
    expectFlagSetsOnly("--ledger", "../../docs/runs.tsv", "ledger_path=../../docs/runs.tsv");
    expectFlagSetsOnly("--exploration-epsilon", "0.25", "exploration_epsilon=0.25");
    expectFlagSetsOnly("--decisive-share", "0.4", "decisive_share=0.4");
    // std::format renders a float as its shortest round-trip form, so 0.0001 prints
    // in exponent notation. The expectation is what the formatter produces, derived by
    // reading it print rather than by assuming a decimal.
    expectFlagSetsOnly("--weight-decay", "0.0001", "weight_decay=1e-04");
    expectFlagSetsOnly("--final-lr-fraction", "0.1", "final_learning_rate_fraction=0.1");
    expectFlagSetsOnly("--full-search-fraction", "0.25", "full_search_fraction=0.25");
    expectFlagSetsOnly("--fast-simulation-fraction", "0.5", "fast_simulation_fraction=0.5");
    expectFlagSetsOnly("--doom-label", "on", "doom_label_from_trajectory=true");
    expectFlagSetsOnly("--death-cap", "on", "death_cap=true");

    // The whole command line the current run was launched with, parsed at once.
    // Every flag above is checked alone; this is the one check that a realistic
    // invocation survives the loop that walks them in pairs.
    const trainer::Settings settings = parse({ "--board",           "10",
                                               "--iterations",      "30",
                                               "--start-iteration", "111",
                                               "--games",           "256",
                                               "--simulations",     "200",
                                               "--batch",           "512",
                                               "--batches",         "3000",
                                               "--channels",        "96",
                                               "--blocks",          "6",
                                               "--checkpoint",      "az10_seeded.pt",
                                               "--resume",          "az10_iter110.pt" });
    expect(settings.board == 10 && settings.iterations == 30 && settings.start_iteration == 111 &&
               settings.games_per_iteration == 256 && settings.simulations == 200 &&
               settings.batch_size == 512 && settings.batches_per_iteration == 3000 &&
               settings.channels == 96 && settings.blocks == 6 &&
               settings.checkpoint == "az10_seeded.pt" && settings.resume == "az10_iter110.pt",
           "a full command line parses to all eleven settings at once");
    expect(settings.lastIteration() == 140,
           "iterations 111 through 140 inclusive, so the last one is 140");
}

// A bad argument is refused rather than defaulted. Defaulting produces a run configured
// differently from what was typed, and nothing in the log would contradict it.
void testBadArgumentsAreRefusedRatherThanDefaulted()
{
    // The old parser warned on stderr and continued with the default, so a
    // mistyped flag produced a run that looked configured and was not - and the
    // warning had scrolled away by the time anyone read the log.
    expectRejected({ "--bord", "12" }, "a mistyped flag is refused, not warned about");
    expectRejected({ "--board" }, "a flag with no value is refused, not silently dropped");
    expectRejected({ "--board", "10", "--games" },
                   "and a trailing flag with no value is refused too");
    expectRejected({ "--board", "ten" }, "a non-numeric value is refused with the flag named");
    expectRejected({ "--board", "10x10" }, "trailing characters after a number are refused");
    expectRejected({ "--board", "" }, "an empty value is refused");
    expectRejected({ "--board", "99999999999999999999" }, "a value outside int range is refused");

    // Ranges. Each of these ran to completion before: --games 0 divided by the
    // summary count, --batch 0 built an empty tensor, --board 1 reached the
    // network's own check only after the process had set everything else up.
    expectRejected({ "--games", "0" }, "zero games per iteration is refused - it divided by zero");
    expectRejected({ "--board", "1" }, "a board of one cell is refused");
    expectRejected({ "--iterations", "0" }, "a run with no iterations is refused");
    expectRejected({ "--start-iteration", "0" }, "iteration zero is refused - they start at one");
    expectRejected({ "--simulations", "0" }, "a search with no simulations is refused");
    expectRejected({ "--channels", "0" }, "a trunk with no channels is refused");
    expectRejected({ "--blocks", "-1" }, "a negative block count is refused");
    expectRejected({ "--batch", "0" }, "an empty minibatch is refused");
    expectRejected({ "--batches", "-1" }, "a negative batch count is refused");
    expectRejected({ "--step-limit", "0" },
                   "an explicit step limit of zero is refused rather than read as absent");
    expectRejected({ "--replay-mb", "0" }, "a replay buffer of nothing is refused");
    // All three take a rate in [0, 1]. Above one is the mistake worth catching:
    // a weight decay of 5 would swamp the gradient, and a share of 2 reads as a
    // percentage typed without the decimal point.
    expectRejected({ "--decisive-share", "1.5" }, "a share above one is refused");
    expectRejected({ "--weight-decay", "-0.1" }, "a negative weight decay is refused");
    expectRejected({ "--doom-label", "maybe" },
                   "an on/off flag refuses anything that is not on or off");
    expectRejected({ "--full-search-fraction", "1.01" },
                   "a share above one is refused for the search fraction too");
    expectRejected({ "--final-lr-fraction", "2" },
                   "a schedule that raises the learning rate is refused");

    // Overflow, which the parser accepted and the derived quantities then hit as
    // undefined behaviour. cellCount squares the board in an int, so 46341 is past
    // the edge; the board's real ceiling is lower, at the largest whose step limit
    // fits. lastIteration adds two ints that were each bounded below and not above.
    expectRejected({ "--board", "13378" },
                   "a board whose step limit does not fit in an int is refused");
    expectRejected({ "--board", "46341" }, "and one whose area does not fit either");
    expectRejected({ "--start-iteration", "2147483647", "--iterations", "2" },
                   "a last iteration past the end of an int is refused");

    // And the boundary on each side, so the checks are bounds and not blanket
    // rejections: a test that only feeds invalid values passes against a parser
    // that throws unconditionally.
    expect(!throwsOnParse({ "--board", "2" }), "the smallest legal board is accepted");
    expect(!throwsOnParse({ "--blocks", "0" }), "a trunk with no residual blocks is legal");
    expect(!throwsOnParse({ "--batches", "0" }),
           "and an iteration that only plays, without training, is legal");
    expect(!throwsOnParse({}), "no arguments at all is legal and yields the defaults");
    expect(!throwsOnParse({ "--board", "13377" }),
           "the largest board whose step limit fits is accepted");
    expect(!throwsOnParse({ "--start-iteration", "2147483646", "--iterations", "2" }),
           "and a last iteration landing exactly on the end of an int is accepted");
}

// Elapsed time formats to a fixed shape, so a progress line does not change width as the
// run gets longer and leave fragments of the previous line on screen.
void testDurationFormatting()
{
    expect(trainer::formatDuration(0.0) == "00:00", "zero seconds is 00:00");
    expect(trainer::formatDuration(65.0) == "01:05", "65 seconds is 01:05");
    expect(trainer::formatDuration(59.6) == "01:00", "and it rounds rather than truncates");
    expect(trainer::formatDuration(3599.0) == "59:59", "an hour less a second stays in minutes");
    expect(trainer::formatDuration(3600.0) == "60:00", "and an hour is 60:00, not 01:00:00");
    expect(trainer::formatDuration(938.34) == "15:38",
           "an iteration of this run's measured length reads as 15:38");
    expect(trainer::formatDuration(-1.0) == "--:--", "a negative duration has no reading");
    expect(trainer::formatDuration(1.0e9) == "--:--", "and neither does an absurd one");
}

// The bar reports whichever of games and moves is further along. Games finished is zero
// for the first several minutes of a large board, and a bar pinned at zero reads as a
// hung run.
void testProgressBarReportsTheFurtherMeasure()
{
    trainer::ProgressSnapshot progress{};
    progress.games_total = 256;
    progress.games_finished = 0;
    progress.moves_played = 0;
    progress.evaluations = 0;
    progress.step_limit = 1200;
    progress.elapsed_seconds = 0.0;

    const std::string start = trainer::formatProgressBar(111, 140, progress);
    expect(start.find("iter 111/140") != std::string::npos,
           "the bar names the absolute iteration and the last one");
    expect(start.find("  0%") != std::string::npos, "nothing done reads as 0 percent");
    expect(start.find("eta --:--") != std::string::npos,
           "and no eta is offered before there is anything to extrapolate from");
    expect(start.find('\r') == std::string::npos && start.find('\n') == std::string::npos,
           "the bar carries no cursor control of its own - the caller owns that");

    // Zero games finished but a quarter of the worst-case moves played. Games
    // finished is the honest measure and it sits at zero for minutes on a large
    // board; moves against the worst case is a lower bound that moves from the
    // first second, and the bar must show whichever is further along.
    progress.moves_played = 256LL * 1200LL / 4LL;
    progress.elapsed_seconds = 100.0;
    const std::string by_moves = trainer::formatProgressBar(111, 140, progress);
    expect(by_moves.find(" 25%") != std::string::npos,
           "with no game finished the bar still reads 25 percent from moves played");
    expect(by_moves.find("eta 05:00") != std::string::npos,
           "and a quarter done in 100 seconds extrapolates to 300 seconds remaining");

    // Now half the games are in, which is further along than the moves measure.
    progress.games_finished = 128;
    const std::string by_games = trainer::formatProgressBar(111, 140, progress);
    expect(by_games.find(" 50%") != std::string::npos,
           "and once half the games are finished it reads 50 percent, the larger of the two");

    // Past the worst case, which happens whenever games end early: the bar must
    // clamp rather than print 400 percent or overrun its own width.
    progress.games_finished = 256;
    progress.moves_played = 256LL * 1200LL * 4LL;
    const std::string done = trainer::formatProgressBar(111, 140, progress);
    expect(done.find("100%") != std::string::npos, "a finished batch reads 100 percent");
    expect(done.find("401%") == std::string::npos && done.find("400%") == std::string::npos,
           "and an overshoot is clamped, not printed");

    const size_t open = done.find('[');
    const size_t close = done.find(']');
    expect(open != std::string::npos && close > open && close - open - 1 == 28,
           "the bar is 28 cells wide whatever the numbers are");
    expect(done.substr(open + 1, close - open - 1).find('.') == std::string::npos,
           "and at 100 percent every cell is filled");

    // Every character has to survive a Windows console with no unicode support.
    bool ascii = true;
    for (char character : done)
    {
        if (static_cast<unsigned char>(character) > 127u)
        {
            ascii = false;
        }
    }
    expect(ascii, "the whole line is ASCII");

    // Evaluations per second is derived, so it is only shown once the elapsed
    // time is long enough for the division to mean anything.
    progress.evaluations = 590380;
    progress.elapsed_seconds = 10.0;
    expect(trainer::formatProgressBar(111, 140, progress).find("59038 ev/s") != std::string::npos,
           "the evaluation rate is evaluations over elapsed seconds");
    progress.elapsed_seconds = 0.1;
    expect(trainer::formatProgressBar(111, 140, progress).find("ev/s") == std::string::npos,
           "and it is withheld while the elapsed time is too short to divide by");
}

}  // namespace

// The flag exists so the ledger records what the run did: the ledger stores the command
// line, and a value compiled into az_parameters.h leaves two different runs looking
// identical afterwards. So the property is that the flag reaches the settings, and that
// omitting it falls back to the constant rather than to zero.
void testExplorationEpsilonIsAFlag()
{
    expect(parse({ "--exploration-epsilon", "0.1" }).exploration_epsilon == 0.1f,
           "--exploration-epsilon 0.1 arrives as 0.1");
    expect(parse({ "--exploration-epsilon", "0" }).exploration_epsilon == 0.0f,
           "--exploration-epsilon 0 arrives as 0, which is off");
    // Omitted means whatever the constant says, not zero - otherwise turning the floor on
    // in az_parameters.h would be silently overridden by every run that did not pass it.
    expect(parse({ "--board", "10" }).exploration_epsilon == az::EXPLORATION_EPSILON,
           "omitting the flag falls back to the constant");
    // A rate, refused outside [0, 1] and on a partial parse.
    expect(throwsOnParse({ "--exploration-epsilon", "1.5" }), "a rate above one is refused");
    expect(throwsOnParse({ "--exploration-epsilon", "-0.1" }), "a negative rate is refused");
    expect(throwsOnParse({ "--exploration-epsilon", "0.1x" }), "a partial parse is refused");
}

int main()
{
    testExplorationEpsilonIsAFlag();
    std::cout << "Trainer options\n";
    testDefaultsAreTheDocumentedOnes();
    testStepLimitIsDerivedFromTheBoardUnlessGiven();
    testTheGradientBudgetIsPerGameAndDerivable();
    testEveryFlagSetsTheFieldItNames();
    testBadArgumentsAreRefusedRatherThanDefaulted();
    testDurationFormatting();
    testProgressBarReportsTheFurtherMeasure();

    // Coverage, printed rather than asserted. A property that silently stopped
    // reaching anything is the failure mode these numbers exist to expose - the
    // same mechanism caught a hunger-death case that no seed ever reached.
    std::cout << std::format("\n{} flags checked, {} rejections checked\n", flags_checked,
                             rejections_checked);

    if (failures == 0)
    {
        std::cout << "All checks passed.\n";
        return 0;
    }
    std::cout << std::format("{} check(s) failed.\n", failures);
    return 1;
}
