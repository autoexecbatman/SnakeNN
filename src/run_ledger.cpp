#include <process.h>
#include <cassert>
#include <chrono>
#include <filesystem>
#include <format>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "run_ledger.h"

namespace ledger
{
namespace
{

// A tab ends a column and a newline ends a row, so neither may reach either.
std::string withoutSeparators(std::string_view text)
{
    std::string cleaned(text);
    for (char& character : cleaned)
    {
        if (character == '\t' || character == '\n' || character == '\r')
        {
            character = ' ';
        }
    }
    return cleaned;
}

// The word a Kind is written as in the ledger.
std::string_view name(Kind kind) noexcept
{
    return kind == Kind::Training ? "training" : "evaluation";
}

// Three words, none a substring of another, so a reader matching on the word cannot
// read one outcome as another.
std::string_view name(Outcome outcome) noexcept
{
    if (outcome == Outcome::Started)
    {
        return "started";
    }
    if (outcome == Outcome::Finished)
    {
        return "finished";
    }
    return "failed";
}

}  // namespace

std::string makeRunId(std::string_view started_utc, unsigned int process_id)
{
    assert(!started_utc.empty() && "makeRunId was given no start time to build an id from");
    // Time first, so a lexicographic sort of the ledger is chronological.
    return std::format("{}-{}", started_utc, process_id);
}

std::string utcNow()
{
    const auto now = std::chrono::floor<std::chrono::seconds>(std::chrono::system_clock::now());
    return std::format("{:%Y-%m-%dT%H:%M:%SZ}", now);
}

std::string formatCommand(std::span<const std::string> arguments)
{
    std::string command;
    for (const std::string& argument : arguments)
    {
        if (!command.empty())
        {
            command += ' ';
        }
        command += withoutSeparators(argument);
    }
    return command;
}

std::string header()
{
    return "run_id\tstarted_utc\tkind\tcommand\toutcome\tseconds\tgames\tsamples\n";
}

std::string formatEntry(const Entry& entry)
{
    return std::format("{}\t{}\t{}\t{}\t{}\t{:.2f}\t{}\t{}\n", withoutSeparators(entry.run_id),
                       withoutSeparators(entry.started_utc), name(entry.kind),
                       withoutSeparators(entry.command), name(entry.outcome), entry.seconds,
                       entry.games, entry.samples);
}

void append(const std::string& path, const Entry& entry)
{
    const bool fresh = !std::filesystem::exists(path);

    std::ofstream ledger_file(path, std::ios::app);
    if (!ledger_file)
    {
        throw std::runtime_error(std::format("cannot open the run ledger at '{}'", path));
    }
    if (fresh)
    {
        ledger_file << header();
    }
    ledger_file << formatEntry(entry);
    // Flushed rather than held: a run that is killed must leave what it has already
    // written, and a buffer does not survive the process.
    ledger_file.flush();
    if (!ledger_file)
    {
        throw std::runtime_error(std::format("cannot write to the run ledger at '{}'", path));
    }
}

Entry openRun(int argc, char** argv, Kind kind, const std::string& path)
{
    Entry run{ makeRunId(utcNow(), static_cast<unsigned int>(_getpid())),
               utcNow(),
               kind,
               formatCommand(std::vector<std::string>(argv + 1, argv + argc)),
               Outcome::Started,
               0.0,
               0,
               0 };
    append(path, run);
    return run;
}

}  // namespace ledger
