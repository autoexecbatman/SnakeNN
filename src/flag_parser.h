#pragma once
#include <string_view>

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

}  // namespace flags
