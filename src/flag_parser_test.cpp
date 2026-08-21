#include <cmath>
#include <format>
#include <iostream>
#include <stdexcept>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "flag_parser.h"

// Expected values come from the contract in flag_parser.h.

namespace
{

int failures = 0;

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

void expectUnsignedEquals(std::string_view what, unsigned int expected, unsigned int actual)
{
    if (expected == actual)
    {
        return;
    }
    std::cout << std::format("[FAIL] {}: expected {}, got {}\n", what, expected, actual);
    failures++;
}

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

void rejectsBelowTheBound()
{
    expectBelowBound("--board", 1, 2);
    expectBelowBound("--batches", -1, 0);
    expectBelowBound("--offset", -11, -10);
    expectBelowBound("--games", 0, 1);
    expectBelowBound("--seed", -2147483648, 0);
}

std::vector<flags::FlagValue> read(const std::vector<std::string>& arguments)
{
    return flags::readFlags(std::span<const std::string>(arguments));
}

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

void rejectsAFlagWithNoValue()
{
    expectArgumentsRejected("trailing flag", { "--board" }, "--board");
    expectArgumentsRejected("trailing flag after a pair", { "--board", "10", "--games" },
                            "--games");
    // The dropped value mid-line: "--games" would otherwise be board's value.
    expectArgumentsRejected("value dropped mid-line", { "--board", "--games", "64" }, "--board");
}

void rejectsSomethingThatIsNotAFlag()
{
    expectArgumentsRejected("bare word first", { "board", "10" }, "board");
    expectArgumentsRejected("single dash", { "-board", "10" }, "-board");
    expectArgumentsRejected("bare word after a pair", { "--board", "10", "games", "64" }, "games");
}

}  // namespace

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
