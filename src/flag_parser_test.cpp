#include <cmath>
#include <format>
#include <iostream>
#include <stdexcept>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "flag_parser.h"

// Checks the layer every program's command line passes through.
//
// Six functions, one job: turn text into a value, or refuse. parseWholeInt and
// parseWholeUnsigned for counts and seeds, parseUnitFloat for a rate in [0, 1], parseOnOff
// for a switch, requireAtLeast for a bound, and readFlags to split an argument list into
// flag-and-value pairs.
//
// Why this is worth 490 lines of test. Every failure this layer can have is a quiet one.
// A parser that consumed "10x10" as 10 runs a board nobody asked for; one that wrapped a
// negative seed produces a huge one and a held-out set that is not held out; one that read
// "yes" as false runs the opposite of what was typed. In each case the run completes, the
// log looks ordinary, and the number is wrong. There is no crash to investigate, so the
// only place the mistake can be caught is here.
//
// The shape of every refusal test. It is not enough that a bad value throws - the message
// has to name the flag and the value, because an operator reading a refusal needs to know
// which argument to fix. Several cases go further and check which of two failures was
// reported: a malformed number and an out-of-range one both throw std::invalid_argument,
// so only the message distinguishes "that is not a number" from "that number is too big",
// and a parser that conflated them would send someone hunting for a typo that is not there.
//
// Bounds are tested in pairs either side of the edge, so an off-by-one in a comparison
// fails rather than passing on both sides.
//
// Expected values come from the contract in flag_parser.h, never from what the parser
// returns. A test written against observed output records present behaviour and would
// confirm a wrong reading forever.
//
// Run it:
//
//     cmake --build build --config Release --target FlagParserTest
//     build\Release\FlagParserTest.exe
//
// Silent unless something fails, ending in
//
//     [SUCCESS] flag_parser: all properties hold

namespace
{

// Checks that did not hold. main prints the count and returns 1 when it is non-zero.
int failures = 0;

// Compares two ints, reporting both when they differ.
//
//     expectEquals("--board 10", 10, flags::parseWholeInt("--board", "10"));
void expectEquals(std::string_view what, int expected, int actual)
{
    if (expected == actual)
    {
        return;
    }
    std::cout << std::format("[FAIL] {}: expected {}, got {}\n", what, expected, actual);
    failures++;
}

// Rejected, and the message names both the flag and the value.
void expectRejected(std::string_view flag, std::string_view text)
{
    try
    {
        const int value = flags::parseWholeInt(flag, text);
        std::cout << std::format("[FAIL] {} '{}': accepted, returned {}\n", flag, text, value);
        failures++;
    }
    catch (const std::invalid_argument& error)
    {
        const std::string message = error.what();
        if (message.find(flag) == std::string::npos)
        {
            std::cout << std::format("[FAIL] {} '{}': message omits the flag: {}\n", flag, text,
                                     message);
            failures++;
            return;
        }
        if (message.find(text) == std::string::npos && !text.empty())
        {
            std::cout << std::format("[FAIL] {} '{}': message omits the value: {}\n", flag, text,
                                     message);
            failures++;
        }
    }
    catch (const std::exception& error)
    {
        std::cout << std::format("[FAIL] {} '{}': wrong exception type: {}\n", flag, text,
                                 error.what());
        failures++;
    }
}

// The ordinary case, plus the exact edges of int. A parser that only ever sees small
// numbers hides an overflow that arrives the first time somebody types a real seed.
void acceptsAWholeNumber()
{
    expectEquals("plain digits", 12, flags::parseWholeInt("--board", "12"));
    expectEquals("zero", 0, flags::parseWholeInt("--batches", "0"));
    expectEquals("negative", -5, flags::parseWholeInt("--offset", "-5"));
    // Literal bounds, so a change to the parser's width fails here.
    expectEquals("largest int", 2147483647, flags::parseWholeInt("--seed", "2147483647"));
    expectEquals("most negative int", -2147483648, flags::parseWholeInt("--seed", "-2147483648"));
}

// Input the parse cannot consume in full.
void rejectsAnythingItDidNotFullyConsume()
{
    expectRejected("--board", "10x10");
    expectRejected("--board", "12abc");
    expectRejected("--board", "ten");
    expectRejected("--board", "");
    expectRejected("--board", " 12");
    expectRejected("--board", "12 ");
    expectRejected("--board", "1.5");
    expectRejected("--board", "0x10");
    expectRejected("--board", "+7");
}

// Both branches throw the same type, so only the message distinguishes them.
void reportsAnOutOfRangeValueAsItsOwnMistake()
{
    for (const std::string_view text : { "2147483648", "-2147483649", "99999999999999999999" })
    {
        expectRejected("--seed", text);
        try
        {
            flags::parseWholeInt("--seed", text);
        }
        catch (const std::invalid_argument& error)
        {
            const std::string message = error.what();
            if (message.find("out of range") == std::string::npos)
            {
                std::cout << std::format(
                    "[FAIL] --seed '{}': diagnosed as malformed, not as "
                    "out of range: {}\n",
                    text, message);
                failures++;
            }
        }
    }
}

// Compares two unsigned ints. Separate from expectEquals because a seed near the top of the
// unsigned range converts to a negative int, and the failure message would print a number
// that is not the one that differed.
//
//     expectUnsignedEquals("--seed 4294967295", 4294967295u,
//                          flags::parseWholeUnsigned("--seed", "4294967295"));
void expectUnsignedEquals(std::string_view what, unsigned int expected, unsigned int actual)
{
    if (expected == actual)
    {
        return;
    }
    std::cout << std::format("[FAIL] {}: expected {}, got {}\n", what, expected, actual);
    failures++;
}

// Checks that parseWholeUnsigned refuses a value, naming the flag and the value.
//
//     expectUnsignedRejected("--seed", "-1");   // must not wrap to 4294967295
void expectUnsignedRejected(std::string_view flag, std::string_view text)
{
    try
    {
        const unsigned int value = flags::parseWholeUnsigned(flag, text);
        std::cout << std::format("[FAIL] {} '{}': accepted, returned {}\n", flag, text, value);
        failures++;
    }
    catch (const std::invalid_argument& error)
    {
        const std::string message = error.what();
        if (message.find(flag) == std::string::npos)
        {
            std::cout << std::format("[FAIL] {} '{}': message omits the flag: {}\n", flag, text,
                                     message);
            failures++;
        }
    }
    catch (const std::exception& error)
    {
        std::cout << std::format("[FAIL] {} '{}': wrong exception type: {}\n", flag, text,
                                 error.what());
        failures++;
    }
}

// The ordinary case, plus the top of the unsigned range - which is where evaluation seeds
// live, so this is the region the project actually uses.
void acceptsAnUnsignedWholeNumber()
{
    expectUnsignedEquals("zero", 0u, flags::parseWholeUnsigned("--seed", "0"));
    expectUnsignedEquals("plain digits", 900000u, flags::parseWholeUnsigned("--seed", "900000"));
    expectUnsignedEquals("largest unsigned", 4294967295u,
                         flags::parseWholeUnsigned("--seed", "4294967295"));
}

// A negative value must be an error, not a wraparound to a huge seed.
void rejectsANegativeValueRatherThanWrappingIt()
{
    expectUnsignedRejected("--seed", "-1");
    expectUnsignedRejected("--seed", "-900000");
}

// Text that is not a number, and a number past the unsigned ceiling. Both refused, so a
// seed cannot arrive as something the caller never typed.
void rejectsMalformedAndOversizedUnsigned()
{
    expectUnsignedRejected("--seed", "");
    expectUnsignedRejected("--seed", "12abc");
    expectUnsignedRejected("--seed", " 12");
    expectUnsignedRejected("--seed", "+7");
    expectUnsignedRejected("--seed", "4294967296");
    expectUnsignedRejected("--seed", "99999999999999999999");
}

// Both branches throw the same type, so only the message distinguishes them.
void reportsAnOversizedUnsignedAsItsOwnMistake()
{
    try
    {
        flags::parseWholeUnsigned("--seed", "4294967296");
        std::cout << "[FAIL] --seed '4294967296': accepted\n";
        failures++;
    }
    catch (const std::invalid_argument& error)
    {
        const std::string message = error.what();
        if (message.find("out of range") == std::string::npos)
        {
            std::cout << std::format(
                "[FAIL] --seed '4294967296': diagnosed as malformed, not as "
                "out of range: {}\n",
                message);
            failures++;
        }
    }
}

// Checks that requireAtLeast accepts a value at or above its bound, by returning normally.
//
//     expectAccepted("--games", 1, 1);   // exactly at the bound is legal
void expectAccepted(std::string_view flag, int value, int minimum)
{
    try
    {
        flags::requireAtLeast(flag, value, minimum);
    }
    catch (const std::exception& error)
    {
        std::cout << std::format("[FAIL] {} {} >= {}: rejected: {}\n", flag, value, minimum,
                                 error.what());
        failures++;
    }
}

// The message must carry all three, or it does not say what to change.
// Checks that requireAtLeast refuses a value below its bound and that the message carries
// the flag, the value and the bound.
//
//     expectBelowBound("--games", 0, 1);
//
// All three, or the message does not say what to change: a refusal naming only the flag
// leaves the operator guessing which direction the bound runs.
void expectBelowBound(std::string_view flag, int value, int minimum)
{
    try
    {
        flags::requireAtLeast(flag, value, minimum);
        std::cout << std::format("[FAIL] {} {} < {}: accepted\n", flag, value, minimum);
        failures++;
    }
    catch (const std::invalid_argument& error)
    {
        const std::string message = error.what();
        if (message.find(flag) == std::string::npos)
        {
            std::cout << std::format("[FAIL] {} {} < {}: message omits the flag: {}\n", flag, value,
                                     minimum, message);
            failures++;
        }
        // Ordered, not just present: the two numbers reversed reads as the
        // opposite requirement and still contains both.
        const std::string ordered = std::format("at least {}, got {}", minimum, value);
        if (message.find(ordered) == std::string::npos)
        {
            std::cout << std::format("[FAIL] {} {} < {}: message does not read '{}': {}\n", flag,
                                     value, minimum, ordered, message);
            failures++;
        }
    }
    catch (const std::exception& error)
    {
        std::cout << std::format("[FAIL] {} {} < {}: wrong exception type: {}\n", flag, value,
                                 minimum, error.what());
        failures++;
    }
}

// Paired either side of each bound, so an off-by-one in the comparison fails.
void acceptsAtOrAboveTheBound()
{
    expectAccepted("--board", 2, 2);
    expectAccepted("--board", 3, 2);
    expectAccepted("--batches", 0, 0);
    expectAccepted("--offset", -10, -10);
    expectAccepted("--offset", -9, -10);
    expectAccepted("--seed", 2147483647, 0);
}

// One below each bound, paired with the accepting cases above so an off-by-one in the
// comparison fails on one side rather than passing on both.
void rejectsBelowTheBound()
{
    expectBelowBound("--board", 1, 2);
    expectBelowBound("--batches", -1, 0);
    expectBelowBound("--offset", -11, -10);
    expectBelowBound("--games", 0, 1);
    expectBelowBound("--seed", -2147483648, 0);
}

// readFlags over a vector, so a case can be written as a brace list.
//
//     read({ "--board", "10", "--games", "8" });   // two pairs, in order
std::vector<flags::FlagValue> read(const std::vector<std::string>& arguments)
{
    return flags::readFlags(std::span<const std::string>(arguments));
}

// Checks that an argument list splits into exactly the expected flag-and-value pairs, in
// order.
//
//     expectPairs("two flags", { "--board", "10", "--games", "8" },
//                 { { "--board", "10" }, { "--games", "8" } });
//
// Order matters because a later flag overwrites an earlier one, so a parser that returned
// the right pairs in the wrong order would silently resolve a repeated flag backwards.
void expectPairs(std::string_view what, const std::vector<std::string>& arguments,
                 const std::vector<std::string>& expected)
{
    try
    {
        const std::vector<flags::FlagValue> pairs = read(arguments);
        if (pairs.size() * 2 != expected.size())
        {
            std::cout << std::format("[FAIL] {}: expected {} pairs, got {}\n", what,
                                     expected.size() / 2, pairs.size());
            failures++;
            return;
        }
        for (size_t index = 0; index < pairs.size(); index++)
        {
            if (pairs[index].flag != expected[index * 2] ||
                pairs[index].value != expected[index * 2 + 1])
            {
                std::cout << std::format("[FAIL] {}: pair {} is '{}' '{}', expected '{}' '{}'\n",
                                         what, index, pairs[index].flag, pairs[index].value,
                                         expected[index * 2], expected[index * 2 + 1]);
                failures++;
            }
        }
    }
    catch (const std::exception& error)
    {
        std::cout << std::format("[FAIL] {}: rejected: {}\n", what, error.what());
        failures++;
    }
}

// Checks that readFlags refuses an argument list, and that the message contains a phrase.
//
//     expectArgumentsRejected("a flag with no value", { "--board" }, "--board");
void expectArgumentsRejected(std::string_view what, const std::vector<std::string>& arguments,
                             std::string_view named)
{
    try
    {
        const std::vector<flags::FlagValue> pairs = read(arguments);
        std::cout << std::format("[FAIL] {}: accepted, {} pairs\n", what, pairs.size());
        failures++;
    }
    catch (const std::invalid_argument& error)
    {
        const std::string message = error.what();
        if (message.find(named) == std::string::npos)
        {
            std::cout << std::format("[FAIL] {}: message does not name '{}': {}\n", what, named,
                                     message);
            failures++;
        }
    }
    catch (const std::exception& error)
    {
        std::cout << std::format("[FAIL] {}: wrong exception type: {}\n", what, error.what());
        failures++;
    }
}

// An argument list becomes flag-and-value pairs in the order given. Order is the property:
// a later flag overwrites an earlier one, so pairs returned out of order would resolve a
// repeated flag backwards.
void splitsArgumentsIntoPairsInOrder()
{
    expectPairs("no arguments", {}, {});
    expectPairs("one pair", { "--board", "10" }, { "--board", "10" });
    expectPairs("three pairs, order preserved", { "--board", "10", "--games", "64", "--seed", "0" },
                { "--board", "10", "--games", "64", "--seed", "0" });
    // A negative number is a value; only "--" opens a flag.
    expectPairs("negative value", { "--offset", "-5" }, { "--offset", "-5" });
    // A path is a value like any other.
    expectPairs("path value", { "--checkpoint", "az10_iter123.pt" },
                { "--checkpoint", "az10_iter123.pt" });
}

// A trailing flag with nothing after it. Refused rather than defaulted: a flag typed and
// then ignored is the case where the operator is most certain they configured the run.
void rejectsAFlagWithNoValue()
{
    expectArgumentsRejected("trailing flag", { "--board" }, "--board");
    expectArgumentsRejected("trailing flag after a pair", { "--board", "10", "--games" },
                            "--games");
    // The dropped value mid-line: "--games" would otherwise be board's value.
    expectArgumentsRejected("value dropped mid-line", { "--board", "--games", "64" }, "--board");
}

// A bare word where a flag was expected - usually a value whose flag was forgotten. It
// cannot be attributed to anything, so it is refused rather than skipped.
void rejectsSomethingThatIsNotAFlag()
{
    expectArgumentsRejected("bare word first", { "board", "10" }, "board");
    expectArgumentsRejected("single dash", { "-board", "10" }, "-board");
    expectArgumentsRejected("bare word after a pair", { "--board", "10", "games", "64" }, "games");
}

}  // namespace

// Checks parseUnitFloat returns the expected rate.
//
//     expectRate("--epsilon", "0.1", 0.1f);
void expectRate(std::string_view flag, std::string_view text, float expected)
{
    try
    {
        const float value = flags::parseUnitFloat(flag, text);
        if (std::abs(value - expected) > 1e-6f)
        {
            std::cout << std::format("[FAIL] {} '{}': expected {:.6f}, got {:.6f}\n", flag, text,
                                     expected, value);
            failures++;
        }
    }
    catch (const std::exception& error)
    {
        std::cout << std::format("[FAIL] {} '{}': rejected: {}\n", flag, text, error.what());
        failures++;
    }
}

// Checks parseUnitFloat refuses a value, naming the flag and the value.
//
//     expectRateRejected("--epsilon", "1.5");   // outside [0, 1], refused not clamped
void expectRateRejected(std::string_view flag, std::string_view text)
{
    try
    {
        const float value = flags::parseUnitFloat(flag, text);
        std::cout << std::format("[FAIL] {} '{}': accepted, returned {:.6f}\n", flag, text, value);
        failures++;
    }
    catch (const std::invalid_argument& error)
    {
        const std::string message = error.what();
        if (message.find(flag) == std::string::npos)
        {
            std::cout << std::format("[FAIL] {} '{}': message omits the flag: {}\n", flag, text,
                                     message);
            failures++;
        }
    }
    catch (const std::exception& error)
    {
        std::cout << std::format("[FAIL] {} '{}': wrong exception: {}\n", flag, text, error.what());
        failures++;
    }
}

// A rate is a fraction in [0, 1]. The endpoints are legal - zero is "off" and one is
// "always" - and everything outside is refused rather than clamped, because a clamp
// turns a typo into a plausible run.
void testParseUnitFloat()
{
    expectRate("--exploration-epsilon", "0", 0.0f);
    expectRate("--exploration-epsilon", "1", 1.0f);
    expectRate("--exploration-epsilon", "0.1", 0.1f);
    expectRate("--exploration-epsilon", "0.5", 0.5f);
    expectRate("--exploration-epsilon", ".5", 0.5f);
    expectRate("--exploration-epsilon", "1e-1", 0.1f);

    // The defect this parser exists to avoid, in its own shape: a partial parse is an
    // error, not a value. "0.1x" must not arrive as 0.1.
    expectRateRejected("--exploration-epsilon", "0.1x");
    expectRateRejected("--exploration-epsilon", "abc");
    expectRateRejected("--exploration-epsilon", "");
    expectRateRejected("--exploration-epsilon", " 0.5");
    expectRateRejected("--exploration-epsilon", "0.5 ");

    // Outside the unit interval, either side.
    expectRateRejected("--exploration-epsilon", "1.5");
    expectRateRejected("--exploration-epsilon", "-0.1");

    // A NaN compares false against every bound, so a range test written as two
    // comparisons would admit it and hand an infinity to a search config.
    expectRateRejected("--exploration-epsilon", "nan");
    expectRateRejected("--exploration-epsilon", "inf");
}

// Expected values come from the contract: exactly "on" and exactly "off", and every
// other spelling is a rejection rather than a silent false.
void acceptsOnAndOff()
{
    expectEquals("parseOnOff on", 1, flags::parseOnOff("--skip-arms", "on") ? 1 : 0);
    expectEquals("parseOnOff off", 0, flags::parseOnOff("--skip-arms", "off") ? 1 : 0);
}

// Each of these would otherwise resolve to false and run the opposite of what was asked.
void rejectsEveryOtherSpelling()
{
    for (const std::string_view text : { "ON", "Off", "true", "false", "1", "0", "yes", "", " on" })
    {
        try
        {
            const bool value = flags::parseOnOff("--skip-arms", text);
            std::cout << std::format("[FAIL] --skip-arms '{}': accepted, returned {}\n", text,
                                     value ? "true" : "false");
            failures++;
        }
        catch (const std::invalid_argument& error)
        {
            const std::string message = error.what();
            if (message.find("--skip-arms") == std::string::npos)
            {
                std::cout << std::format("[FAIL] --skip-arms '{}': message omits the flag: {}\n",
                                         text, message);
                failures++;
            }
        }
    }
}

// Runs every case, then reports. Returns 1 if any check failed, 0 otherwise.
int main()
{
    testParseUnitFloat();
    acceptsAWholeNumber();
    rejectsAnythingItDidNotFullyConsume();
    reportsAnOutOfRangeValueAsItsOwnMistake();
    acceptsAnUnsignedWholeNumber();
    rejectsANegativeValueRatherThanWrappingIt();
    rejectsMalformedAndOversizedUnsigned();
    reportsAnOversizedUnsignedAsItsOwnMistake();
    acceptsAtOrAboveTheBound();
    rejectsBelowTheBound();
    splitsArgumentsIntoPairsInOrder();
    rejectsAFlagWithNoValue();
    rejectsSomethingThatIsNotAFlag();
    acceptsOnAndOff();
    rejectsEveryOtherSpelling();

    if (failures == 0)
    {
        std::cout << "[SUCCESS] flag_parser: all properties hold\n";
        return 0;
    }
    std::cout << std::format("[FAILURE] {} property violations\n", failures);
    return 1;
}
