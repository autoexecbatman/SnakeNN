#pragma once
#include <span>
#include <string>
#include <string_view>
#include <vector>

// Command-line number parsing, shared by the trainer, the evaluator and the visual.
// No LibTorch here, so its checks are reachable in a debug build.
namespace flags
{

// The whole value, or std::invalid_argument naming the flag.
//
// The parse must consume all of `text`, so a partial number such as "10x10" is
// rejected rather than read as 10. Also rejects "", " 12", "12 ", "+7" and values
// outside int, and reports out-of-range separately from malformed.
//
// `flag` must be non-empty; it appears in the message. Views must outlive the
// call and nothing is stored.
int parseWholeInt(std::string_view flag, std::string_view text);

// As parseWholeInt, for a value that cannot be negative.
//
// A leading '-' is rejected rather than wrapped, so "-1" is an error and not
// 4294967295. Accepts 0 through 4294967295.
unsigned int parseWholeUnsigned(std::string_view flag, std::string_view text);

// Throws std::invalid_argument unless value >= minimum. The message carries the
// flag, the bound and the value.
//
// `minimum` is a bound the caller states, so any int is allowed including a
// negative one; `value` is what the operator supplied.
void requireAtLeast(std::string_view flag, int value, int minimum);

// One flag and the value that followed it.
struct FlagValue
{
    std::string_view flag;
    std::string_view value;
};

// The arguments split into flag and value pairs, in the order given.
//
// `arguments` excludes argv[0]. Every entry must be a flag starting with "--"
// followed by one value, so an odd count, a bare word where a flag belongs, or a
// flag immediately followed by another flag all throw std::invalid_argument.
// That last case is a value dropped mid-line, which is otherwise read as the
// literal text "--games" and reported against the wrong flag.
//
// A negative number is a value, not a flag: only "--" opens one, and "-5" does
// not. A value that must itself begin with "--" cannot be passed.
//
// The views point into `arguments`, which must outlive the result.
std::vector<FlagValue> readFlags(std::span<const std::string> arguments);

}  // namespace flags
