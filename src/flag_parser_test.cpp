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

}  // namespace

int main()
{
    acceptsAWholeNumber();
    rejectsAnythingItDidNotFullyConsume();
    reportsAnOutOfRangeValueAsItsOwnMistake();

    if (failures == 0)
    {
        std::cout << "[SUCCESS] flags::parseWholeInt: all properties hold\n";
        return 0;
    }
    std::cout << std::format("[FAILURE] {} property violations\n", failures);
    return 1;
}
