#include <cstdio>
#include <ctime>
#include <format>
#include <fstream>
#include <iostream>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "run_ledger.h"

// Expected values come from the contract in run_ledger.h.

namespace
{

int failures = 0;

void fail(std::string_view what, std::string_view detail)
{
    std::cout << std::format("[FAIL] {}: {}\n", what, detail);
    failures++;
}

void expectText(std::string_view what, std::string_view expected, std::string_view actual)
{
    if (expected != actual)
    {
        fail(what, std::format("expected '{}', got '{}'", expected, actual));
    }
}

void expectContains(std::string_view what, std::string_view needle, std::string_view text)
{
    if (text.find(needle) == std::string_view::npos)
    {
        fail(what, std::format("'{}' is absent from '{}'", needle, text));
    }
}

ledger::Entry sampleEntry()
{
    return ledger::Entry{"2026-08-09T12:00:00Z-4242",
                         "2026-08-09T12:00:00Z",
                         ledger::Kind::Evaluation,
                         "--checkpoint az10_iter140.pt --games 64",
                         ledger::Outcome::Finished,
                         244.89,
                         64,
                         0};
}

std::vector<std::string> splitOnTabs(const std::string& row)
{
    std::vector<std::string> fields;
    std::string field;
    std::istringstream stream(row);
    while (std::getline(stream, field, '\t'))
    {
        fields.push_back(field);
    }
    if (!fields.empty() && !fields.back().empty() && fields.back().back() == '\n')
    {
        fields.back().pop_back();
    }
    return fields;
}

// A run id sorts chronologically, which is what makes the ledger readable unsorted.
void theRunIdIsTheStartTimeThenTheProcess()
{
    expectText("run id", "2026-08-09T12:00:00Z-4242",
               ledger::makeRunId("2026-08-09T12:00:00Z", 4242u));
    // Two runs starting in the same second are told apart by the process id, and an
    // earlier start still sorts first.
    const std::string earlier = ledger::makeRunId("2026-08-09T11:59:59Z", 9999u);
    const std::string later = ledger::makeRunId("2026-08-09T12:00:00Z", 1u);
    if (!(earlier < later))
    {
        fail("run id ordering", std::format("'{}' does not sort before '{}'", earlier, later));
    }
}

void theTimestampIsUtcAndFixedWidth()
{
    const std::string now = ledger::utcNow();
    if (now.size() != 20)
    {
        fail("utcNow",
             std::format("expected 20 characters of YYYY-MM-DDTHH:MM:SSZ, got '{}'", now));
        return;
    }
    if (now[4] != '-' || now[7] != '-' || now[10] != 'T' || now[13] != ':' || now[16] != ':' ||
        now[19] != 'Z')
    {
        fail("utcNow", std::format("not YYYY-MM-DDTHH:MM:SSZ: '{}'", now));
    }
    // And that it is actually now, and actually UTC. Checked against the C time
    // API rather than against chrono again, so a shifted clock or a local-time
    // reading fails here instead of producing a plausible-looking wrong timestamp.
    // Read either side of the reference so a minute boundary is not a failure.
    const std::string before = ledger::utcNow();
    const std::time_t raw = std::time(nullptr);
    std::tm utc{};
    gmtime_s(&utc, &raw);
    char expected[32] = {};
    std::strftime(expected, sizeof(expected), "%Y-%m-%dT%H:%M", &utc);
    const std::string after = ledger::utcNow();
    if (before.substr(0, 16) != expected && after.substr(0, 16) != expected)
    {
        fail("utcNow", std::format("expected '{}' to the minute, got '{}'", expected, before));
    }
}

// A tab or a newline in an argument would silently add a column or a row.
void theCommandCannotBreakTheRow()
{
    const std::vector<std::string> plain{"--checkpoint", "az10_iter140.pt", "--games", "64"};
    expectText("command", "--checkpoint az10_iter140.pt --games 64",
               ledger::formatCommand(std::span<const std::string>(plain)));

    const std::vector<std::string> hostile{"--note", "two\tcolumns", "--other", "two\nrows"};
    const std::string cleaned = ledger::formatCommand(std::span<const std::string>(hostile));
    if (cleaned.find('\t') != std::string::npos)
    {
        fail("command", std::format("a tab survived: '{}'", cleaned));
    }
    if (cleaned.find('\n') != std::string::npos)
    {
        fail("command", std::format("a newline survived: '{}'", cleaned));
    }
    expectContains("command keeps the text", "two columns", cleaned);

    const std::vector<std::string> empty;
    expectText("no arguments", "", ledger::formatCommand(std::span<const std::string>(empty)));
}

void aRowHasEveryColumnTheHeaderNames()
{
    const std::vector<std::string> columns = splitOnTabs(ledger::header());
    const std::vector<std::string> fields = splitOnTabs(ledger::formatEntry(sampleEntry()));
    if (columns.size() != fields.size())
    {
        fail("row width",
             std::format("header has {} columns, a row has {}", columns.size(), fields.size()));
        return;
    }
    if (columns.size() != 8)
    {
        fail("column count",
             std::format("expected the 8 fields of Entry, header names {}", columns.size()));
    }
    for (const std::string& field : fields)
    {
        if (field.empty())
        {
            fail("row", std::format("a column is blank: '{}'", ledger::formatEntry(sampleEntry())));
            break;
        }
    }
}

void aRowCarriesWhatTheRunCost()
{
    const std::string row = ledger::formatEntry(sampleEntry());
    // By column, not by substring. The command contains "--games 64", so searching
    // the whole row for "64" is satisfied by text that is not the games column.
    const std::vector<std::string> fields = splitOnTabs(row);
    if (fields.size() != 8)
    {
        fail("row", std::format("expected 8 columns, got {}", fields.size()));
        return;
    }
    expectText("run_id column", "2026-08-09T12:00:00Z-4242", fields[0]);
    expectText("started_utc column", "2026-08-09T12:00:00Z", fields[1]);
    expectText("kind column", "evaluation", fields[2]);
    expectText("command column", "--checkpoint az10_iter140.pt --games 64", fields[3]);
    expectText("outcome column", "finished", fields[4]);
    expectText("seconds column", "244.89", fields[5]);
    expectText("games column", "64", fields[6]);
    expectText("samples column", "0", fields[7]);
    if (row.empty() || row.back() != '\n')
    {
        fail("row", "does not end in a newline");
    }
    if (row.find('\n') != row.size() - 1)
    {
        fail("row", "is more than one line");
    }
}

// Three outcomes and two kinds, each a distinct word, or the ledger cannot be read.
void everyKindAndOutcomeHasItsOwnWord()
{
    ledger::Entry entry = sampleEntry();
    entry.outcome = ledger::Outcome::Started;
    entry.seconds = 0.0;
    entry.games = 0;
    entry.samples = 0;
    expectContains("started", "started", ledger::formatEntry(entry));
    // A zero is written, not left blank: no games played and no column are different
    // facts and only one of them is true.
    expectContains("started games", "\t0\t", ledger::formatEntry(entry));

    entry.outcome = ledger::Outcome::Failed;
    expectContains("failed", "failed", ledger::formatEntry(entry));
    entry.outcome = ledger::Outcome::Finished;
    expectContains("finished", "finished", ledger::formatEntry(entry));

    entry.kind = ledger::Kind::Training;
    expectContains("training", "training", ledger::formatEntry(entry));
    entry.kind = ledger::Kind::Evaluation;
    expectContains("evaluation", "evaluation", ledger::formatEntry(entry));

    // "finished" must not read as "started" to a reader matching on the word, and
    // "training" must not read as "evaluation".
    ledger::Entry started = sampleEntry();
    started.outcome = ledger::Outcome::Started;
    if (ledger::formatEntry(started).find("finished") != std::string::npos)
    {
        fail("started", "also contains 'finished'");
    }
}

std::string readWholeFile(const std::string& path)
{
    std::ifstream file(path, std::ios::binary);
    std::ostringstream contents;
    contents << file.rdbuf();
    return contents.str();
}

int countLines(const std::string& text)
{
    int lines = 0;
    for (const char character : text)
    {
        lines += character == '\n' ? 1 : 0;
    }
    return lines;
}

void appendingCreatesTheLedgerAndThenAddsToIt()
{
    const std::string path = "run_ledger_test_scratch.tsv";
    std::remove(path.c_str());

    ledger::Entry start = sampleEntry();
    start.outcome = ledger::Outcome::Started;
    start.seconds = 0.0;
    start.games = 0;
    start.samples = 0;
    ledger::append(path, start);

    const std::string afterFirst = readWholeFile(path);
    if (countLines(afterFirst) != 2)
    {
        fail("append",
             std::format("expected a header and one row, got {} lines", countLines(afterFirst)));
    }
    expectContains("append writes the header", "run_id", afterFirst);

    ledger::append(path, sampleEntry());
    const std::string afterSecond = readWholeFile(path);
    if (countLines(afterSecond) != 3)
    {
        fail("append",
             std::format("expected a header and two rows, got {} lines", countLines(afterSecond)));
    }
    // The header is written once, not once per row.
    if (afterSecond.find("run_id") != afterSecond.rfind("run_id"))
    {
        fail("append", "wrote the header more than once");
    }
    // A killed run leaves its Started row: the first row is still there unchanged.
    expectContains("the started row survives", "started", afterSecond);

    std::remove(path.c_str());
}

void appendingToAnUnwritablePathThrows()
{
    try
    {
        ledger::append("no_such_directory_here/runs.tsv", sampleEntry());
        fail("unwritable path", "accepted silently");
    }
    catch (const std::runtime_error& error)
    {
        // The diagnosis has to match what happened. A file that cannot be opened
        // also cannot be written, so the write guard below catches it too and
        // reports the wrong cause to whoever reads the error.
        const std::string message = error.what();
        if (message.find("open") == std::string::npos)
        {
            fail("unwritable path", std::format("rejected with the wrong cause: {}", message));
        }
        if (message.find("no_such_directory_here") == std::string::npos)
        {
            fail("unwritable path", std::format("message omits the path: {}", message));
        }
    }
    catch (const std::exception& error)
    {
        fail("unwritable path", std::format("wrong exception type: {}", error.what()));
    }
}

}  // namespace

int main()
{
    theRunIdIsTheStartTimeThenTheProcess();
    theTimestampIsUtcAndFixedWidth();
    theCommandCannotBreakTheRow();
    aRowHasEveryColumnTheHeaderNames();
    aRowCarriesWhatTheRunCost();
    everyKindAndOutcomeHasItsOwnWord();
    appendingCreatesTheLedgerAndThenAddsToIt();
    appendingToAnUnwritablePathThrows();

    if (failures == 0)
    {
        std::cout << "[PASS] run_ledger\n";
        return 0;
    }
    std::cout << std::format("[FAIL] run_ledger: {} failures\n", failures);
    return 1;
}
