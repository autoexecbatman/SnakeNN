#include <cassert>
#include <charconv>
#include <format>
#include <stdexcept>
#include <system_error>

#include "flag_parser.h"

namespace flags
{

int parseWholeInt(std::string_view flag, std::string_view text)
{
    // The flag names the error; an empty one leaves the message pointing at nothing.
    assert(!flag.empty() && "parseWholeInt was given no flag name to report against");

    int number = 0;
    // from_chars takes pointers; kept inline so none is named in this file.
    const std::from_chars_result result =
        std::from_chars(text.data(), text.data() + text.size(), number);

    // A well-formed number too wide for int, reported apart from a malformed one.
    if (result.ec == std::errc::result_out_of_range)
    {
        throw std::invalid_argument(
            std::format("{} is out of range for a whole number, got '{}'", flag, text));
    }
    // Unparsed input remaining means a partial parse, which is an error not a value.
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size())
    {
        throw std::invalid_argument(std::format("{} needs a whole number, got '{}'", flag, text));
    }
    return number;
}

float parseUnitFloat(std::string_view flag, std::string_view text)
{
    assert(!flag.empty() && "parseUnitFloat was given no flag name to report against");

    float number = 0.0f;
    const std::from_chars_result result =
        std::from_chars(text.data(), text.data() + text.size(), number);

    if (result.ec == std::errc::result_out_of_range)
    {
        throw std::invalid_argument(
            std::format("{} is out of range for a rate, got '{}'", flag, text));
    }
    // Unparsed input remaining means a partial parse, which is an error not a value.
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size())
    {
        throw std::invalid_argument(std::format("{} needs a number, got '{}'", flag, text));
    }
    // Written as a failed range test rather than a pair of comparisons, so a NaN -
    // which compares false against everything - is refused rather than admitted.
    if (!(number >= 0.0f && number <= 1.0f))
    {
        throw std::invalid_argument(
            std::format("{} must be a rate in [0, 1], got '{}'", flag, text));
    }
    return number;
}

unsigned int parseWholeUnsigned(std::string_view flag, std::string_view text)
{
    assert(!flag.empty() && "parseWholeUnsigned was given no flag name to report against");

    unsigned int number = 0;
    // from_chars on an unsigned type does not consume a leading '-', so a negative
    // value fails the full-consumption check below instead of wrapping.
    const std::from_chars_result result =
        std::from_chars(text.data(), text.data() + text.size(), number);

    if (result.ec == std::errc::result_out_of_range)
    {
        throw std::invalid_argument(
            std::format("{} is out of range for a whole number, got '{}'", flag, text));
    }
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size())
    {
        throw std::invalid_argument(
            std::format("{} needs a whole number that is not negative, got '{}'", flag, text));
    }
    return number;
}

void requireAtLeast(std::string_view flag, int value, int minimum)
{
    assert(!flag.empty() && "requireAtLeast was given no flag name to report against");

    if (value < minimum)
    {
        throw std::invalid_argument(
            std::format("{} must be at least {}, got {}", flag, minimum, value));
    }
}

std::vector<FlagValue> readFlags(std::span<const std::string> arguments)
{
    std::vector<FlagValue> pairs;
    pairs.reserve(arguments.size() / 2);

    for (size_t index = 0; index < arguments.size(); index += 2)
    {
        const std::string_view flag = arguments[index];
        if (!flag.starts_with("--"))
        {
            throw std::invalid_argument(
                std::format("'{}' is not a flag; flags start with --", flag));
        }
        if (index + 1 >= arguments.size())
        {
            throw std::invalid_argument(std::format("{} was given no value", flag));
        }
        const std::string_view value = arguments[index + 1];
        // A flag where a value belongs is a value dropped from the middle of the
        // line, not a value that happens to read as a flag.
        if (value.starts_with("--"))
        {
            throw std::invalid_argument(std::format("{} was given no value", flag));
        }
        pairs.push_back(FlagValue{ flag, value });
    }
    return pairs;
}

}  // namespace flags
