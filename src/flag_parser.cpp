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

}  // namespace flags
