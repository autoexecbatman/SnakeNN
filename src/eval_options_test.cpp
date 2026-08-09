#include <format>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "eval_options.h"

// Expected values come from the contract in eval_options.h, never from what the
// parser returns.

namespace
{

int failures = 0;

void fail(std::string_view what, std::string_view detail)
{
    std::cout << std::format("[FAIL] {}: {}\n", what, detail);
    failures++;
}

void expectEquals(std::string_view what, int expected, int actual)
{
    if (expected != actual)
    {
        fail(what, std::format("expected {}, got {}", expected, actual));
    }
}

void expectUnsignedEquals(std::string_view what, unsigned int expected, unsigned int actual)
{
    if (expected != actual)
    {
        fail(what, std::format("expected {}, got {}", expected, actual));
    }
}

void expectText(std::string_view what, std::string_view expected, std::string_view actual)
{
    if (expected != actual)
    {
        fail(what, std::format("expected '{}', got '{}'", expected, actual));
    }
}

evaluation::Settings parse(const std::vector<std::string>& arguments)
{
    return evaluation::parseArguments(std::span<const std::string>(arguments));
}

// A checkpoint is required, so every accepting case carries one.
evaluation::Settings parseWithCheckpoint(const std::vector<std::string>& arguments)
{
    std::vector<std::string> full{"--checkpoint", "model.pt"};
    full.insert(full.end(), arguments.begin(), arguments.end());
    return parse(full);
}

// Rejected, and the message names the flag.
void expectRejected(std::string_view what, const std::vector<std::string>& arguments)
{
    try
    {
        parse(arguments);
        fail(what, "accepted");
    }
    catch (const std::invalid_argument& error)
    {
        const std::string message = error.what();
        if (message.empty())
        {
            fail(what, "rejected with an empty message");
        }
    }
    catch (const std::exception& error)
    {
        fail(what, std::format("wrong exception type: {}", error.what()));
    }
}

void expectRejectedNaming(std::string_view flag, const std::vector<std::string>& arguments)
{
    try
    {
        parse(arguments);
        fail(flag, "accepted");
    }
    catch (const std::invalid_argument& error)
    {
        const std::string message = error.what();
        if (message.find(flag) == std::string::npos)
        {
            fail(flag, std::format("message omits the flag: {}", message));
        }
    }
    catch (const std::exception& error)
    {
        fail(flag, std::format("wrong exception type: {}", error.what()));
    }
}

// Rejected for the stated reason. A value can be refused by the wrong check and
// reported against with a diagnosis that does not describe what was typed.
void expectRejectedSaying(std::string_view phrase, const std::vector<std::string>& arguments)
{
    try
    {
        parse(arguments);
        fail(phrase, "accepted");
    }
    catch (const std::invalid_argument& error)
    {
        const std::string message = error.what();
        if (message.find(phrase) == std::string::npos)
        {
            fail(phrase, std::format("rejected with the wrong diagnosis: {}", message));
        }
    }
    catch (const std::exception& error)
    {
        fail(phrase, std::format("wrong exception type: {}", error.what()));
    }
}

// Every field that differs between two settings, so a parser writing the right
// value into the wrong field is caught rather than passing on the field it meant.
std::vector<std::string> differingFields(const evaluation::Settings& left,
                                         const evaluation::Settings& right)
{
    std::vector<std::string> names;
    if (left.checkpoint != right.checkpoint)
    {
        names.push_back("checkpoint");
    }
    if (left.board != right.board)
    {
        names.push_back("board");
    }
    if (left.games != right.games)
    {
        names.push_back("games");
    }
    if (left.simulations != right.simulations)
    {
        names.push_back("simulations");
    }
    if (left.step_limit_override != right.step_limit_override)
    {
        names.push_back("step_limit_override");
    }
    if (left.channels != right.channels)
    {
        names.push_back("channels");
    }
    if (left.blocks != right.blocks)
    {
        names.push_back("blocks");
    }
    if (left.seed_offset != right.seed_offset)
    {
        names.push_back("seed_offset");
    }
    if (left.batch != right.batch)
    {
        names.push_back("batch");
    }
    return names;
}

void expectOnlyFieldChanged(std::string_view field, const std::vector<std::string>& arguments)
{
    const evaluation::Settings baseline = parseWithCheckpoint({});
    const std::vector<std::string> changed =
        differingFields(baseline, parseWithCheckpoint(arguments));
    if (changed.size() != 1 || changed.front() != field)
    {
        std::string listed;
        for (const std::string& name : changed)
        {
            listed += listed.empty() ? name : ", " + name;
        }
        fail(field, std::format("changed [{}] instead of exactly [{}]", listed, field));
    }
}

// The defaults are the contract's, written out rather than read back.
void defaultsAreTheOnesTheHeaderStates()
{
    const evaluation::Settings settings = parseWithCheckpoint({});
    expectText("default checkpoint", "model.pt", settings.checkpoint);
    expectEquals("default board", 6, settings.board);
    expectEquals("default games", 64, settings.games);
    expectEquals("default simulations", 200, settings.simulations);
    expectEquals("default channels", 64, settings.channels);
    expectEquals("default blocks", 4, settings.blocks);
    expectUnsignedEquals("default seed offset", 0u, settings.seed_offset);
    expectEquals("default batch", 64, settings.batch);
    if (settings.step_limit_override.has_value())
    {
        fail("default step limit", "an override is present when none was given");
    }
}

void eachFlagWritesItsOwnField()
{
    expectOnlyFieldChanged("board", {"--board", "10"});
    expectOnlyFieldChanged("games", {"--games", "200"});
    expectOnlyFieldChanged("simulations", {"--simulations", "800"});
    expectOnlyFieldChanged("step_limit_override", {"--step-limit", "1200"});
    expectOnlyFieldChanged("channels", {"--channels", "128"});
    expectOnlyFieldChanged("blocks", {"--blocks", "8"});
    expectOnlyFieldChanged("seed_offset", {"--seed", "512"});
    expectOnlyFieldChanged("batch", {"--batch", "200"});
}

void derivesTheStepLimitFromTheBoard()
{
    // 12 steps per cell, written out rather than taken from the constant.
    expectEquals("10x10 gets the paper's limit", 1200,
                 parseWithCheckpoint({"--board", "10"}).stepLimit());
    expectEquals("20x20 gets four times it, since cost is area", 4800,
                 parseWithCheckpoint({"--board", "20"}).stepLimit());
    expectEquals("an override is used verbatim", 2400,
                 parseWithCheckpoint({"--board", "10", "--step-limit", "2400"}).stepLimit());
    expectEquals("cells", 100, parseWithCheckpoint({"--board", "10"}).cellCount());
    // One segment at the start, so one cell is already filled.
    expectEquals("foods to win", 99, parseWithCheckpoint({"--board", "10"}).foodsToWin());
}

void acceptsTheEdgesOfEveryRange()
{
    expectEquals("the smallest board", 2, parseWithCheckpoint({"--board", "2"}).board);
    // The largest board whose step limit fits in an int, machine-checked in
    // docs/prove_arithmetic.py.
    expectEquals("the largest board", 13377, parseWithCheckpoint({"--board", "13377"}).board);
    expectEquals("a trunk with no residual blocks", 0,
                 parseWithCheckpoint({"--blocks", "0"}).blocks);
    expectEquals("one game", 1, parseWithCheckpoint({"--games", "1"}).games);
    // The reserved band is 0x20000000 wide, so this offset ends on its last seed.
    expectUnsignedEquals("the last offset that fits the band", 536870848u,
                         parseWithCheckpoint({"--seed", "536870848"}).seed_offset);
}

void rejectsBeforeAnyWorkIsDone()
{
    expectRejectedNaming("--board", {"--checkpoint", "model.pt", "--board", "1"});
    expectRejectedNaming("--board", {"--checkpoint", "model.pt", "--board", "0"});
    expectRejectedNaming("--board", {"--checkpoint", "model.pt", "--board", "13378"});
    expectRejectedNaming("--board", {"--checkpoint", "model.pt", "--board", "10x10"});
    expectRejectedNaming("--games", {"--checkpoint", "model.pt", "--games", "0"});
    expectRejectedNaming("--simulations", {"--checkpoint", "model.pt", "--simulations", "0"});
    expectRejectedNaming("--batch", {"--checkpoint", "model.pt", "--batch", "0"});
    expectRejectedNaming("--channels", {"--checkpoint", "model.pt", "--channels", "0"});
    expectRejectedNaming("--blocks", {"--checkpoint", "model.pt", "--blocks", "-1"});
    expectRejectedNaming("--step-limit", {"--checkpoint", "model.pt", "--step-limit", "0"});
    expectRejectedNaming("--bord", {"--checkpoint", "model.pt", "--bord", "12"});
    expectRejectedNaming("--board", {"--checkpoint", "model.pt", "--board"});
    expectRejected("no checkpoint at all", {"--board", "10"});
    expectRejectedNaming("--checkpoint", {"--checkpoint", ""});
}

void rejectsASeedRangeThatLeavesTheReservedBand()
{
    // 0xE0000000 leaves 0x20000000 = 536870912 seeds. The last one this run may
    // touch is offset + games - 1, so 536870849 with 64 games runs one past the
    // end and wraps into the training range without erroring.
    expectRejectedNaming("--seed",
                         {"--checkpoint", "model.pt", "--seed", "536870849", "--games", "64"});
    expectRejectedNaming("--seed", {"--checkpoint", "model.pt", "--seed", "536870912"});
    expectRejectedNaming("--seed", {"--checkpoint", "model.pt", "--seed", "-1"});
    // A negative seed casts to a value the band check also refuses, so without
    // this the parser could reach that check and report an overrun the operator
    // did not cause.
    expectRejectedSaying("not negative", {"--checkpoint", "model.pt", "--seed", "-1"});
}

visual::Settings parseVisual(const std::vector<std::string>& arguments)
{
    return visual::parseArguments(std::span<const std::string>(arguments));
}

visual::Settings parseVisualWithCheckpoint(const std::vector<std::string>& arguments)
{
    std::vector<std::string> full{"--checkpoint", "model.pt"};
    full.insert(full.end(), arguments.begin(), arguments.end());
    return parseVisual(full);
}

void expectVisualRejectedNaming(std::string_view flag, const std::vector<std::string>& arguments)
{
    try
    {
        parseVisual(arguments);
        fail(flag, "accepted");
    }
    catch (const std::invalid_argument& error)
    {
        const std::string message = error.what();
        if (message.find(flag) == std::string::npos)
        {
            fail(flag, std::format("message omits the flag: {}", message));
        }
    }
    catch (const std::exception& error)
    {
        fail(flag, std::format("wrong exception type: {}", error.what()));
    }
}

std::vector<std::string> differingVisualFields(const visual::Settings& left,
                                               const visual::Settings& right)
{
    std::vector<std::string> names;
    if (left.checkpoint != right.checkpoint)
    {
        names.push_back("checkpoint");
    }
    if (left.board != right.board)
    {
        names.push_back("board");
    }
    if (left.simulations != right.simulations)
    {
        names.push_back("simulations");
    }
    if (left.step_limit_override != right.step_limit_override)
    {
        names.push_back("step_limit_override");
    }
    if (left.channels != right.channels)
    {
        names.push_back("channels");
    }
    if (left.blocks != right.blocks)
    {
        names.push_back("blocks");
    }
    if (left.seed != right.seed)
    {
        names.push_back("seed");
    }
    if (left.moves_per_frame != right.moves_per_frame)
    {
        names.push_back("moves_per_frame");
    }
    return names;
}

void expectOnlyVisualFieldChanged(std::string_view field, const std::vector<std::string>& arguments)
{
    const visual::Settings baseline = parseVisualWithCheckpoint({});
    const std::vector<std::string> changed =
        differingVisualFields(baseline, parseVisualWithCheckpoint(arguments));
    if (changed.size() != 1 || changed.front() != field)
    {
        std::string listed;
        for (const std::string& name : changed)
        {
            listed += listed.empty() ? name : ", " + name;
        }
        fail(field, std::format("changed [{}] instead of exactly [{}]", listed, field));
    }
}

void theVisualDefaultsAreTheOnesTheHeaderStates()
{
    const visual::Settings settings = parseVisualWithCheckpoint({});
    expectText("visual default checkpoint", "model.pt", settings.checkpoint);
    expectEquals("visual default board", 6, settings.board);
    expectEquals("visual default simulations", 200, settings.simulations);
    expectEquals("visual default channels", 64, settings.channels);
    expectEquals("visual default blocks", 4, settings.blocks);
    // A training seed, kept deliberately. If this ever changes it is a decision
    // about what the demo shows, and this line is where it gets noticed.
    expectUnsignedEquals("visual default seed", 900000u, settings.seed);
    expectEquals("visual default speed", 1, settings.moves_per_frame);
    if (settings.step_limit_override.has_value())
    {
        fail("visual default step limit", "an override is present when none was given");
    }
}

void eachVisualFlagWritesItsOwnField()
{
    expectOnlyVisualFieldChanged("board", {"--board", "10"});
    expectOnlyVisualFieldChanged("simulations", {"--simulations", "800"});
    expectOnlyVisualFieldChanged("step_limit_override", {"--step-limit", "1200"});
    expectOnlyVisualFieldChanged("channels", {"--channels", "128"});
    expectOnlyVisualFieldChanged("blocks", {"--blocks", "8"});
    expectOnlyVisualFieldChanged("seed", {"--seed", "4294967295"});
    expectOnlyVisualFieldChanged("moves_per_frame", {"--speed", "25"});
}

void theVisualDerivesTheSameStepLimit()
{
    expectEquals("visual 10x10 step limit", 1200,
                 parseVisualWithCheckpoint({"--board", "10"}).stepLimit());
    expectEquals("visual 20x20 step limit", 4800,
                 parseVisualWithCheckpoint({"--board", "20"}).stepLimit());
    expectEquals("visual override used verbatim", 2400,
                 parseVisualWithCheckpoint({"--board", "10", "--step-limit", "2400"}).stepLimit());
    expectEquals("visual cells", 100, parseVisualWithCheckpoint({"--board", "10"}).cellCount());
    expectEquals("visual foods to win", 99,
                 parseVisualWithCheckpoint({"--board", "10"}).foodsToWin());
}

void theVisualAcceptsTheEdgesOfEveryRange()
{
    expectEquals("visual smallest board", 2, parseVisualWithCheckpoint({"--board", "2"}).board);
    expectEquals("visual largest board", 13377,
                 parseVisualWithCheckpoint({"--board", "13377"}).board);
    expectEquals("visual no residual blocks", 0,
                 parseVisualWithCheckpoint({"--blocks", "0"}).blocks);
    expectEquals("visual slowest speed", 1,
                 parseVisualWithCheckpoint({"--speed", "1"}).moves_per_frame);
    // Every unsigned value names a game, so neither end is rejected.
    expectUnsignedEquals("visual seed zero", 0u, parseVisualWithCheckpoint({"--seed", "0"}).seed);
    expectUnsignedEquals("visual largest seed", 4294967295u,
                         parseVisualWithCheckpoint({"--seed", "4294967295"}).seed);
}

void theVisualRejectsBeforeAnyWorkIsDone()
{
    expectVisualRejectedNaming("--board", {"--checkpoint", "model.pt", "--board", "1"});
    expectVisualRejectedNaming("--board", {"--checkpoint", "model.pt", "--board", "13378"});
    expectVisualRejectedNaming("--board", {"--checkpoint", "model.pt", "--board", "10x10"});
    expectVisualRejectedNaming("--simulations", {"--checkpoint", "model.pt", "--simulations", "0"});
    expectVisualRejectedNaming("--channels", {"--checkpoint", "model.pt", "--channels", "0"});
    expectVisualRejectedNaming("--blocks", {"--checkpoint", "model.pt", "--blocks", "-1"});
    expectVisualRejectedNaming("--speed", {"--checkpoint", "model.pt", "--speed", "0"});
    expectVisualRejectedNaming("--step-limit", {"--checkpoint", "model.pt", "--step-limit", "0"});
    expectVisualRejectedNaming("--bord", {"--checkpoint", "model.pt", "--bord", "12"});
    expectVisualRejectedNaming("--board", {"--checkpoint", "model.pt", "--board"});
    expectVisualRejectedNaming("--checkpoint", {"--checkpoint", ""});
    expectVisualRejectedNaming("--seed", {"--checkpoint", "model.pt", "--seed", "-1"});
    // The evaluator's flags are not this program's, and silently ignoring one
    // would start a run configured differently from what was typed.
    expectVisualRejectedNaming("--games", {"--checkpoint", "model.pt", "--games", "64"});
    expectVisualRejectedNaming("--batch", {"--checkpoint", "model.pt", "--batch", "64"});
}

}  // namespace

int main()
{
    defaultsAreTheOnesTheHeaderStates();
    eachFlagWritesItsOwnField();
    derivesTheStepLimitFromTheBoard();
    acceptsTheEdgesOfEveryRange();
    rejectsBeforeAnyWorkIsDone();
    rejectsASeedRangeThatLeavesTheReservedBand();

    theVisualDefaultsAreTheOnesTheHeaderStates();
    eachVisualFlagWritesItsOwnField();
    theVisualDerivesTheSameStepLimit();
    theVisualAcceptsTheEdgesOfEveryRange();
    theVisualRejectsBeforeAnyWorkIsDone();

    if (failures == 0)
    {
        std::cout << "[PASS] eval_options\n";
        return 0;
    }
    std::cout << std::format("[FAIL] eval_options: {} failures\n", failures);
    return 1;
}
