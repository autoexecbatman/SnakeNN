#include <format>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "eval_options.h"

// Checks the command-line grammar both evaluation programs share.
//
// Expected values come from the contract in eval_options.h, never from what the parser
// returns. A test written against observed output records the parser's present behaviour
// and would confirm a wrong default forever.
//
// The property that matters most here is not that a flag is accepted - it is that a flag
// writes its own field and nothing else. A parser that quietly set two fields, or the wrong
// one, would still accept every command line in this file; expectOnlyFieldChanged is what
// catches it, by parsing a baseline, parsing again with one flag added, and demanding that
// exactly one field differ and that it be the named one.
//
// The rest of the cases cover: the defaults being the ones the header states, the step
// limit being derived rather than stored, the exact edges of every accepted range, refusals
// arriving before any work is done, and a seed range that would leave the reserved band.
// Then the same set again for the visual program, because two parsers that drift apart are
// how a picture and a win rate come to describe different agents.
//
// Run it:
//
//     cmake --build build --config Release --target EvalOptionsTest
//     build\Release\EvalOptionsTest.exe
//
// Silent unless something fails, ending in [PASS] eval_options or a count and a non-zero
// exit.

namespace
{

// Checks that did not hold. main prints the count and returns 1 when it is non-zero.
int failures = 0;

// Reports one failure and counts it.
//
//     fail("board", "expected 10, got 6");
//     // [FAIL] board: expected 10, got 6
//
// `detail` carries the values compared, so a failing log says what it saw rather than only
// which check broke.
void fail(std::string_view what, std::string_view detail)
{
    std::cout << std::format("[FAIL] {}: {}\n", what, detail);
    failures++;
}

// Compares two ints, reporting both when they differ.
//
//     expectEquals("board", 10, settings.board);
void expectEquals(std::string_view what, int expected, int actual)
{
    if (expected != actual)
    {
        fail(what, std::format("expected {}, got {}", expected, actual));
    }
}

// Compares two unsigned ints. Separate from expectEquals because a seed near the top of
// the unsigned range converts to a negative int, and the failure message would print the
// wrong number for the value that actually differed.
//
//     expectUnsignedEquals("seed offset", 0u, settings.seed_offset);
void expectUnsignedEquals(std::string_view what, unsigned int expected, unsigned int actual)
{
    if (expected != actual)
    {
        fail(what, std::format("expected {}, got {}", expected, actual));
    }
}

// Compares two strings, quoting both when they differ so a trailing space is visible.
//
//     expectText("checkpoint", "model.pt", settings.checkpoint);
void expectText(std::string_view what, std::string_view expected, std::string_view actual)
{
    if (expected != actual)
    {
        fail(what, std::format("expected '{}', got '{}'", expected, actual));
    }
}

// Parses an argument list as the evaluator would.
//
//     parse({ "--checkpoint", "model.pt", "--board", "10" });
//
// Throws whatever parseArguments throws; the rejection helpers rely on that.
evaluation::Settings parse(const std::vector<std::string>& arguments)
{
    return evaluation::parseArguments(std::span<const std::string>(arguments));
}

// A checkpoint is required, so every accepting case carries one.
evaluation::Settings parseWithCheckpoint(const std::vector<std::string>& arguments)
{
    std::vector<std::string> full{ "--checkpoint", "model.pt" };
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

// Checks that an argument list is refused and that the message names the flag.
//
//     expectRejectedNaming("--board", { "--checkpoint", "m.pt", "--board", "1" });
//
// Naming the flag is the point: an operator reading a refusal needs to know which argument
// to fix, and a parser that rejects the right thing with the wrong message is a parser
// somebody will fight. A different exception type fails the check rather than passing as
// "it threw something".
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
// The names of every Settings field on which two objects differ.
//
//     differingFields(parseWithCheckpoint({}), parseWithCheckpoint({"--board", "10"}));
//     // { "board" }
//
// Written out field by field on purpose. A reflective comparison would silently ignore a
// field added later, and this is the one place that would notice a new flag writing
// somewhere it should not - so a field added to Settings and not added here is a gap in the
// only check that guards against cross-talk.
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
    if (left.ledger_path != right.ledger_path)
    {
        names.push_back("ledger_path");
    }
    if (left.freeze_clock_percent != right.freeze_clock_percent)
    {
        names.push_back("freeze_clock_percent");
    }
    if (left.trap_guard != right.trap_guard)
    {
        names.push_back("trap_guard");
    }
    if (left.average_edges != right.average_edges)
    {
        names.push_back("average_edges");
    }
    return names;
}

// Checks that a flag writes its own field and disturbs no other.
//
//     expectOnlyFieldChanged("board", { "--board", "10" });
//
// Parses a baseline, parses again with the flag added, and demands that exactly one field
// differ and that it be the named one. Reports the fields that actually changed, because
// "changed [board, games] instead of exactly [board]" says what went wrong where a bare
// failure would not.
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
    expectText("default ledger path", "runs.tsv", settings.ledger_path);
    if (settings.step_limit_override.has_value())
    {
        fail("default step limit", "an override is present when none was given");
    }
}

// Every flag, one at a time, changing only its own field.
void eachFlagWritesItsOwnField()
{
    expectOnlyFieldChanged("board", { "--board", "10" });
    expectOnlyFieldChanged("games", { "--games", "200" });
    expectOnlyFieldChanged("simulations", { "--simulations", "800" });
    expectOnlyFieldChanged("step_limit_override", { "--step-limit", "1200" });
    expectOnlyFieldChanged("channels", { "--channels", "128" });
    expectOnlyFieldChanged("blocks", { "--blocks", "8" });
    expectOnlyFieldChanged("seed_offset", { "--seed", "512" });
    expectOnlyFieldChanged("batch", { "--batch", "200" });
    // The launch directory is build/Release and git ignores it, so a run meant to
    // leave a durable record has to be told where the ledger is.
    expectOnlyFieldChanged("ledger_path", { "--ledger", "../../docs/runs.tsv" });
    expectOnlyFieldChanged("freeze_clock_percent", { "--freeze-clock-percent", "50" });
    expectOnlyFieldChanged("trap_guard", { "--trap-guard", "on" });
    expectOnlyFieldChanged("average_edges", { "--average-edges", "on" });
}

// The step limit is computed from the board unless overridden, rather than stored - so a
// board change moves it and the two cannot disagree.
void derivesTheStepLimitFromTheBoard()
{
    // 12 steps per cell, written out rather than taken from the constant.
    expectEquals("10x10 gets the paper's limit", 1200,
                 parseWithCheckpoint({ "--board", "10" }).stepLimit());
    expectEquals("20x20 gets four times it, since cost is area", 4800,
                 parseWithCheckpoint({ "--board", "20" }).stepLimit());
    expectEquals("an override is used verbatim", 2400,
                 parseWithCheckpoint({ "--board", "10", "--step-limit", "2400" }).stepLimit());
    expectEquals("cells", 100, parseWithCheckpoint({ "--board", "10" }).cellCount());
    // One segment at the start, so one cell is already filled.
    expectEquals("foods to win", 99, parseWithCheckpoint({ "--board", "10" }).foodsToWin());
}

// The exact lowest and highest accepted value of every bounded flag. Off-by-one in a bound
// is invisible to any test that stays in the middle of the range.
void acceptsTheEdgesOfEveryRange()
{
    expectEquals("the smallest board", 2, parseWithCheckpoint({ "--board", "2" }).board);
    // The largest board whose step limit fits in an int, machine-checked in
    // docs/prove_arithmetic.py.
    expectEquals("the largest board", 13377, parseWithCheckpoint({ "--board", "13377" }).board);
    expectEquals("a trunk with no residual blocks", 0,
                 parseWithCheckpoint({ "--blocks", "0" }).blocks);
    expectEquals("one game", 1, parseWithCheckpoint({ "--games", "1" }).games);
    // The reserved band is 0x20000000 wide, so this offset ends on its last seed.
    expectUnsignedEquals("the last offset that fits the band", 536870848u,
                         parseWithCheckpoint({ "--seed", "536870848" }).seed_offset);
}

// Every refusal, checked to name its flag. Parsing happens before a checkpoint is opened,
// so a bad command line costs nothing rather than failing twenty minutes into a run.
void rejectsBeforeAnyWorkIsDone()
{
    expectRejectedNaming("--board", { "--checkpoint", "model.pt", "--board", "1" });
    expectRejectedNaming("--board", { "--checkpoint", "model.pt", "--board", "0" });
    expectRejectedNaming("--board", { "--checkpoint", "model.pt", "--board", "13378" });
    expectRejectedNaming("--board", { "--checkpoint", "model.pt", "--board", "10x10" });
    expectRejectedNaming("--games", { "--checkpoint", "model.pt", "--games", "0" });
    expectRejectedNaming("--simulations", { "--checkpoint", "model.pt", "--simulations", "0" });
    expectRejectedNaming("--batch", { "--checkpoint", "model.pt", "--batch", "0" });
    expectRejectedNaming("--channels", { "--checkpoint", "model.pt", "--channels", "0" });
    expectRejectedNaming("--blocks", { "--checkpoint", "model.pt", "--blocks", "-1" });
    expectRejectedNaming("--step-limit", { "--checkpoint", "model.pt", "--step-limit", "0" });
    expectRejectedNaming("--bord", { "--checkpoint", "model.pt", "--bord", "12" });
    expectRejectedNaming("--board", { "--checkpoint", "model.pt", "--board" });
    expectRejected("no checkpoint at all", { "--board", "10" });
    expectRejectedNaming("--checkpoint", { "--checkpoint", "" });
}

// A seed offset whose last game would fall outside the reserved evaluation band is refused
// rather than wrapped. Wrapping lands on a training seed, which is how a held-out set
// stopped being held out once already.
void rejectsASeedRangeThatLeavesTheReservedBand()
{
    // 0xE0000000 leaves 0x20000000 = 536870912 seeds. The last one this run may
    // touch is offset + games - 1, so 536870849 with 64 games runs one past the
    // end and wraps into the training range without erroring.
    expectRejectedNaming("--seed",
                         { "--checkpoint", "model.pt", "--seed", "536870849", "--games", "64" });
    expectRejectedNaming("--seed", { "--checkpoint", "model.pt", "--seed", "536870912" });
    expectRejectedNaming("--seed", { "--checkpoint", "model.pt", "--seed", "-1" });
    // A negative seed casts to a value the band check also refuses, so without
    // this the parser could reach that check and report an overrun the operator
    // did not cause.
    expectRejectedSaying("not negative", { "--checkpoint", "model.pt", "--seed", "-1" });
}

// Checks that a printed header mentions something, quoting the header when it does not.
//
//     expectContains("header names the board", "20x20", formatHeader(settings));
void expectContains(std::string_view what, std::string_view needle, std::string_view text)
{
    if (text.find(needle) == std::string_view::npos)
    {
        fail(what, std::format("'{}' is absent from: {}", needle, text));
    }
}

// The header carries every setting that changes the number underneath it. The
// batch is the one that was missing and the one that confounded a comparison.
void theHeaderRecordsWhatWouldChangeTheResult()
{
    const evaluation::Settings settings = parseWithCheckpoint(
        { "--board", "10", "--games", "64", "--simulations", "200", "--batch", "64" });
    const std::string header = evaluation::formatHeader(settings);

    expectContains("header checkpoint", "model.pt", header);
    expectContains("header board", "10x10", header);
    expectContains("header games", "64 games", header);
    expectContains("header simulations", "200 simulations", header);
    expectContains("header step limit", "step limit 1200", header);
    expectContains("header batch", "batch 64", header);
    // The first and last seed of the reserved band this run covers, written out
    // from EVALUATION_BASE rather than read back from the header.
    expectContains("header first seed", "3758096384", header);
    expectContains("header last seed", "3758096447", header);
    if (header.size() < 2 || header.substr(header.size() - 2) != "\n\n")
    {
        fail("header", "does not end in a blank line");
    }
}

// The guard changes which move is played, so it is a setting rather than a build
// constant, and the header names both states. The two headers must differ - a
// rendering that ignores the field satisfies either check alone.
void theTrapGuardIsASettingAndIsRecordedInBothStates()
{
    const evaluation::Settings defaulted = parseWithCheckpoint({ "--board", "10" });
    if (defaulted.trap_guard != az::TRAP_GUARD)
    {
        fail("trap guard default", "the default does not come from az::TRAP_GUARD");
    }

    const evaluation::Settings guarded =
        parseWithCheckpoint({ "--board", "10", "--trap-guard", "on" });
    const evaluation::Settings unguarded =
        parseWithCheckpoint({ "--board", "10", "--trap-guard", "off" });

    if (!guarded.trap_guard)
    {
        fail("trap guard on", "--trap-guard on did not reach the settings");
    }
    if (unguarded.trap_guard)
    {
        fail("trap guard off", "--trap-guard off did not reach the settings");
    }

    const std::string on_header = evaluation::formatHeader(guarded);
    const std::string off_header = evaluation::formatHeader(unguarded);

    expectContains("guard on announced", "trap guard on", on_header);
    expectContains("guard off announced", "trap guard off", off_header);
    if (on_header == off_header)
    {
        fail("trap guard", "the two guard states render the same header");
    }
    if (on_header.size() < 2 || on_header.substr(on_header.size() - 2) != "\n\n")
    {
        fail("guarded header", "does not end in a blank line");
    }

    // Anything that is not one of the two words is a rejection, not a silent false.
    // "true", "1" and "yes" are the ones an operator would reach for, so each has to
    // fail loudly rather than scoring a run with the guard it did not ask for.
    expectRejectedSaying("--trap-guard", { "--checkpoint", "model.pt", "--trap-guard", "true" });
    expectRejectedSaying("--trap-guard", { "--checkpoint", "model.pt", "--trap-guard", "1" });
    expectRejectedSaying("--trap-guard", { "--checkpoint", "model.pt", "--trap-guard", "yes" });
    expectRejectedSaying("--trap-guard", { "--checkpoint", "model.pt", "--trap-guard", "ON" });
    expectRejectedSaying("--trap-guard", { "--checkpoint", "model.pt", "--trap-guard" });
}

// Averaging changes which move the search plays, so the same three obligations hold
// as for the guard: the default comes from the constant, both directions parse, and
// the header names the state in both so a missing line cannot mean two things.
void averagedEdgesAreASettingAndAreRecordedInBothStates()
{
    const evaluation::Settings defaulted = parseWithCheckpoint({ "--board", "10" });
    if (defaulted.average_edges != az::AVERAGE_EDGES)
    {
        fail("average edges default", "the default does not come from az::AVERAGE_EDGES");
    }

    const evaluation::Settings averaged =
        parseWithCheckpoint({ "--board", "10", "--average-edges", "on" });
    const evaluation::Settings last_write =
        parseWithCheckpoint({ "--board", "10", "--average-edges", "off" });

    if (!averaged.average_edges)
    {
        fail("average edges on", "--average-edges on did not reach the settings");
    }
    if (last_write.average_edges)
    {
        fail("average edges off", "--average-edges off did not reach the settings");
    }

    const std::string on_header = evaluation::formatHeader(averaged);
    const std::string off_header = evaluation::formatHeader(last_write);

    expectContains("averaging announced", "averaged edges", on_header);
    expectContains("last write announced", "last-write edges", off_header);
    if (on_header == off_header)
    {
        fail("average edges", "the two states render the same header");
    }

    expectRejectedSaying("--average-edges",
                         { "--checkpoint", "model.pt", "--average-edges", "true" });
    expectRejectedSaying("--average-edges", { "--checkpoint", "model.pt", "--average-edges", "1" });
    expectRejectedSaying("--average-edges", { "--checkpoint", "model.pt", "--average-edges" });
}

// The clock ablation, which is the one setting that changes what the network sees
// rather than how the run is scored. A log that did not say so would be compared
// against an ordinary run as though the two measured the same agent.
void theClockAblationIsRecordedAndRangeChecked()
{
    const evaluation::Settings ordinary = parseWithCheckpoint({ "--board", "10" });
    if (ordinary.freeze_clock_percent.has_value())
    {
        fail("freeze clock default", "a run nobody ablated has the clock frozen");
    }
    const std::string plain = evaluation::formatHeader(ordinary);
    if (plain.find("ABLATED") != std::string::npos)
    {
        fail("freeze clock default", "an ordinary run is announced as ablated");
    }

    const evaluation::Settings frozen =
        parseWithCheckpoint({ "--board", "10", "--freeze-clock-percent", "50" });
    if (frozen.freeze_clock_percent != 50)
    {
        fail("freeze clock parse", "--freeze-clock-percent 50 did not reach the settings");
    }
    const std::string header = evaluation::formatHeader(frozen);
    expectContains("ablation announced", "ABLATED", header);
    expectContains("ablation value", "50 percent", header);
    if (header.size() < 2 || header.substr(header.size() - 2) != "\n\n")
    {
        fail("ablated header", "does not end in a blank line");
    }

    // Zero is a legitimate freeze - "always out of time" - and rejecting it would
    // remove one of the two ends the ablation is for.
    const evaluation::Settings spent =
        parseWithCheckpoint({ "--board", "10", "--freeze-clock-percent", "0" });
    if (spent.freeze_clock_percent != 0)
    {
        fail("freeze clock zero", "zero was read as absent rather than as a freeze");
    }

    // The checkpoint is supplied and the message is pinned to the bound, because
    // expectRejected without one is satisfied by "--checkpoint is required" and
    // would pass against any ceiling at all.
    expectRejectedSaying("at least 0",
                         { "--checkpoint", "model.pt", "--freeze-clock-percent", "-1" });
    expectRejectedSaying("at most 100",
                         { "--checkpoint", "model.pt", "--freeze-clock-percent", "101" });
}

// The search's stream is separable from the game's, which is what lets one
// checkpoint be measured twice on identical games and the difference attributed
// to the food the search imagined.
void theSearchStreamIsSeparableFromTheGames()
{
    const evaluation::Settings ordinary = parseWithCheckpoint({ "--board", "10" });
    if (ordinary.search_seed.has_value())
    {
        fail("search seed default", "a run nobody asked has a search seed of its own");
    }

    const evaluation::Settings reseeded =
        parseWithCheckpoint({ "--board", "10", "--search-seed", "7" });
    if (reseeded.search_seed != 7u)
    {
        fail("search seed parse", "--search-seed 7 did not reach the settings");
    }
    // Zero is a stream like any other, and reading it as absent would silently
    // return the run to the derived seed it was asked to replace.
    const evaluation::Settings zero =
        parseWithCheckpoint({ "--board", "10", "--search-seed", "0" });
    if (zero.search_seed != 0u)
    {
        fail("search seed zero", "zero was read as absent rather than as a stream");
    }
    // The games must not move with it, or the two runs are not paired and the
    // number they produce is not a noise floor.
    if (reseeded.seed_offset != ordinary.seed_offset)
    {
        fail("search seed independence", "--search-seed moved the games as well");
    }

    const std::string header = evaluation::formatHeader(reseeded);
    expectContains("search seed announced", "search seed 7", header);

    expectRejectedSaying("--search-seed", { "--checkpoint", "model.pt", "--search-seed" });
    expectRejectedSaying("--search-seed", { "--checkpoint", "model.pt", "--search-seed", "eight" });
}

// A seed, an outcome and the two numbers, in a line a parser can key on.
void eachGameGetsALineThatCanBePaired()
{
    const std::string won =
        evaluation::formatGameLine(3758096384u, evaluation::Outcome::Won, 99, 1032);
    const std::string died =
        evaluation::formatGameLine(3758096385u, evaluation::Outcome::Died, 41, 502);
    const std::string timed_out =
        evaluation::formatGameLine(3758096386u, evaluation::Outcome::TimedOut, 87, 1200);

    expectContains("won line seed", "3758096384", won);
    expectContains("won line score", "score 99", won);
    expectContains("won line steps", "steps 1032", won);
    expectContains("died line seed", "3758096385", died);
    expectContains("timed out line seed", "3758096386", timed_out);

    // Tagged, so a parser finds these and not the progress lines.
    expectContains("game tag", "game ", won);
    for (const std::string& line : { won, died, timed_out })
    {
        if (line.empty() || line.back() != '\n')
        {
            fail("game line", std::format("does not end in a newline: {}", line));
        }
        if (line.find('\n') != line.size() - 1)
        {
            fail("game line", std::format("is more than one line: {}", line));
        }
    }

    // Three outcomes, three distinct words, or a parser cannot tell them apart.
    const std::string won_word = won.substr(0, won.find("score"));
    const std::string died_word = died.substr(0, died.find("score"));
    const std::string timed_out_word = timed_out.substr(0, timed_out.find("score"));
    if (won_word.find("won") == std::string::npos)
    {
        fail("won outcome", std::format("does not say 'won': {}", won));
    }
    if (died_word.find("died") == std::string::npos)
    {
        fail("died outcome", std::format("does not say 'died': {}", died));
    }
    if (timed_out_word.find("timeout") == std::string::npos)
    {
        fail("timed out outcome", std::format("does not say 'timeout': {}", timed_out));
    }
    // "timeout" must not read as a win or a death to a parser matching substrings.
    if (timed_out_word.find("won") != std::string::npos ||
        timed_out_word.find("died") != std::string::npos)
    {
        fail("timed out outcome", std::format("also matches another outcome: {}", timed_out));
    }
}

// Parses an argument list as the visual program would.
//
//     parseVisual({ "--checkpoint", "model.pt", "--board", "10" });
visual::Settings parseVisual(const std::vector<std::string>& arguments)
{
    return visual::parseArguments(std::span<const std::string>(arguments));
}

// parseVisual with the required checkpoint already supplied, so a case can name only the
// flag it is about.
visual::Settings parseVisualWithCheckpoint(const std::vector<std::string>& arguments)
{
    std::vector<std::string> full{ "--checkpoint", "model.pt" };
    full.insert(full.end(), arguments.begin(), arguments.end());
    return parseVisual(full);
}

// expectRejectedNaming for the visual parser.
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

// differingFields for visual::Settings. Separate rather than templated: the two structs
// hold different fields, and the list has to be maintained against each one by hand for the
// same reason as above.
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

// expectOnlyFieldChanged for the visual parser.
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

// The visual program's defaults, from its header rather than from what it returns.
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

// Every visual flag, one at a time, changing only its own field.
void eachVisualFlagWritesItsOwnField()
{
    expectOnlyVisualFieldChanged("board", { "--board", "10" });
    expectOnlyVisualFieldChanged("simulations", { "--simulations", "800" });
    expectOnlyVisualFieldChanged("step_limit_override", { "--step-limit", "1200" });
    expectOnlyVisualFieldChanged("channels", { "--channels", "128" });
    expectOnlyVisualFieldChanged("blocks", { "--blocks", "8" });
    expectOnlyVisualFieldChanged("seed", { "--seed", "4294967295" });
    expectOnlyVisualFieldChanged("moves_per_frame", { "--speed", "25" });
}

// The visual program derives the step limit the same way the evaluator does. If it did
// not, the game on screen would run under a different deadline from the one scored.
void theVisualDerivesTheSameStepLimit()
{
    expectEquals("visual 10x10 step limit", 1200,
                 parseVisualWithCheckpoint({ "--board", "10" }).stepLimit());
    expectEquals("visual 20x20 step limit", 4800,
                 parseVisualWithCheckpoint({ "--board", "20" }).stepLimit());
    expectEquals(
        "visual override used verbatim", 2400,
        parseVisualWithCheckpoint({ "--board", "10", "--step-limit", "2400" }).stepLimit());
    expectEquals("visual cells", 100, parseVisualWithCheckpoint({ "--board", "10" }).cellCount());
    expectEquals("visual foods to win", 99,
                 parseVisualWithCheckpoint({ "--board", "10" }).foodsToWin());
}

// The exact edges of every bounded visual flag.
void theVisualAcceptsTheEdgesOfEveryRange()
{
    expectEquals("visual smallest board", 2, parseVisualWithCheckpoint({ "--board", "2" }).board);
    expectEquals("visual largest board", 13377,
                 parseVisualWithCheckpoint({ "--board", "13377" }).board);
    expectEquals("visual no residual blocks", 0,
                 parseVisualWithCheckpoint({ "--blocks", "0" }).blocks);
    expectEquals("visual slowest speed", 1,
                 parseVisualWithCheckpoint({ "--speed", "1" }).moves_per_frame);
    // Every unsigned value names a game, so neither end is rejected.
    expectUnsignedEquals("visual seed zero", 0u, parseVisualWithCheckpoint({ "--seed", "0" }).seed);
    expectUnsignedEquals("visual largest seed", 4294967295u,
                         parseVisualWithCheckpoint({ "--seed", "4294967295" }).seed);
}

// Every visual refusal, checked to name its flag.
void theVisualRejectsBeforeAnyWorkIsDone()
{
    expectVisualRejectedNaming("--board", { "--checkpoint", "model.pt", "--board", "1" });
    expectVisualRejectedNaming("--board", { "--checkpoint", "model.pt", "--board", "13378" });
    expectVisualRejectedNaming("--board", { "--checkpoint", "model.pt", "--board", "10x10" });
    expectVisualRejectedNaming("--simulations",
                               { "--checkpoint", "model.pt", "--simulations", "0" });
    expectVisualRejectedNaming("--channels", { "--checkpoint", "model.pt", "--channels", "0" });
    expectVisualRejectedNaming("--blocks", { "--checkpoint", "model.pt", "--blocks", "-1" });
    expectVisualRejectedNaming("--speed", { "--checkpoint", "model.pt", "--speed", "0" });
    expectVisualRejectedNaming("--step-limit", { "--checkpoint", "model.pt", "--step-limit", "0" });
    expectVisualRejectedNaming("--bord", { "--checkpoint", "model.pt", "--bord", "12" });
    expectVisualRejectedNaming("--board", { "--checkpoint", "model.pt", "--board" });
    expectVisualRejectedNaming("--checkpoint", { "--checkpoint", "" });
    expectVisualRejectedNaming("--seed", { "--checkpoint", "model.pt", "--seed", "-1" });
    // The evaluator's flags are not this program's, and silently ignoring one
    // would start a run configured differently from what was typed.
    expectVisualRejectedNaming("--games", { "--checkpoint", "model.pt", "--games", "64" });
    expectVisualRejectedNaming("--batch", { "--checkpoint", "model.pt", "--batch", "64" });
}

}  // namespace

// Runs every case, then reports. Returns 1 if any check failed, 0 otherwise.
int main()
{
    defaultsAreTheOnesTheHeaderStates();
    eachFlagWritesItsOwnField();
    derivesTheStepLimitFromTheBoard();
    acceptsTheEdgesOfEveryRange();
    rejectsBeforeAnyWorkIsDone();
    rejectsASeedRangeThatLeavesTheReservedBand();
    theHeaderRecordsWhatWouldChangeTheResult();
    theTrapGuardIsASettingAndIsRecordedInBothStates();
    averagedEdgesAreASettingAndAreRecordedInBothStates();
    theClockAblationIsRecordedAndRangeChecked();
    theSearchStreamIsSeparableFromTheGames();
    eachGameGetsALineThatCanBePaired();

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
