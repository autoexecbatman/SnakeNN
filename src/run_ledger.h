#pragma once

#include <span>
#include <string>
#include <string_view>

// What every training and evaluation run cost, appended by the run itself.
//
// A checkpoint reached after 140 iterations had no recoverable wall-clock and no
// cumulative game count: answering "how much did this cost" meant parsing thirteen
// logs, so it was left as a guess in a project that measures win rates to four
// significant figures. Cost is one line and the runner writes it.
//
// Free of LibTorch, so its assertions are reachable in a debug build.
namespace ledger
{

// What a run was doing. Two programs write here and their rows are not comparable,
// so the kind is a field rather than something to infer from the command.
enum class Kind
{
    Training,
    Evaluation
};

// How a row came to be written.
//
// A run appends Started before it does any work and Finished or Failed after. A row
// with no matching completion is a run that was killed, which is the only way that
// fact can be recorded - a killed process writes nothing on its way out.
enum class Outcome
{
    Started,
    Finished,
    Failed
};

// One row.
//
// `games` and `samples` are what the run has completed so far, so both are zero on a
// Started row. `seconds` is wall-clock since the run began and is zero on a Started
// row. `run_id` ties a completion back to its start and is unique within a ledger.
struct Entry
{
    // Identifies the run. From makeRunId, so it sorts chronologically.
    std::string run_id;
    // When it started, as "YYYY-MM-DDTHH:MM:SSZ".
    std::string started_utc;
    // Which program wrote the row.
    Kind kind{ Kind::Training };
    // The command line it was launched with, argv[0] excluded. This is what makes a
    // number reproducible: a setting not on this line is one nobody can recover.
    std::string command;
    // How it ended. A row written at launch carries Started and is replaced on finish,
    // so a Started row still present means the run died without recording anything.
    Outcome outcome{ Outcome::Started };
    // Wall-clock seconds it took.
    double seconds{ 0.0 };
    // Games played.
    long long games{ 0 };
    // Training samples consumed. Zero for a run that does not train.
    long long samples{ 0 };
};

// A run identifier from the start time and the process id, in that order so a
// lexicographic sort is chronological.
//
// `process_id` is the caller's own; it is passed rather than read so the function
// stays testable. `started_utc` must be the same string the entry carries.
std::string makeRunId(std::string_view started_utc, unsigned int process_id);

// The current UTC time as "YYYY-MM-DDTHH:MM:SSZ".
std::string utcNow();

// The arguments as one command string, space separated, argv[0] excluded by the
// caller. An argument containing a tab or a newline would break the row, so both are
// replaced by a single space.
std::string formatCommand(std::span<const std::string> arguments);

// One tab-separated row ending in a newline, in the column order of `header()`.
//
// Every field is written even when it is zero: a blank column and a zero column mean
// different things to a reader, and only one of them is true.
std::string formatEntry(const Entry& entry);

// The tab-separated column names, ending in a newline. Written once when the ledger
// is created and never again.
std::string header();

// Appends `entry` to the ledger at `path`, creating it with a header if absent.
//
// Opens, writes and flushes on every call rather than holding the file: a run that is
// killed must leave what it has already written, and an unflushed buffer does not
// survive. Throws std::runtime_error if the file cannot be opened or written.
void append(const std::string& path, const Entry& entry);

// Opens a run's row and returns it for the caller to complete.
//
//     ledger::Entry run = ledger::openRun(argc, argv, ledger::Kind::Training, path);
//     run.outcome = ledger::Outcome::Finished;   // and whatever else the run measured
//     ledger::append(path, run);
//
// The started row is written here, before any work, so a process that is killed still
// leaves evidence it ran - the only record a killed run can produce. `argc` and `argv` are
// the ones main received; the program name is dropped and the rest becomes the command
// column.
Entry openRun(int argc, char** argv, Kind kind, const std::string& path);

}  // namespace ledger
