#include <format>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

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
    for (const std::string_view text : {"2147483648", "-2147483649", "99999999999999999999"})
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

}  // namespace

int main()
{
    acceptsAWholeNumber();
    rejectsAnythingItDidNotFullyConsume();
    reportsAnOutOfRangeValueAsItsOwnMistake();
    acceptsAnUnsignedWholeNumber();
    rejectsANegativeValueRatherThanWrappingIt();
    rejectsMalformedAndOversizedUnsigned();
    reportsAnOversizedUnsignedAsItsOwnMistake();
    acceptsAtOrAboveTheBound();
    rejectsBelowTheBound();

    if (failures == 0)
    {
        std::cout << "[SUCCESS] flag_parser: all properties hold\n";
        return 0;
    }
    std::cout << std::format("[FAILURE] {} property violations\n", failures);
    return 1;
}
