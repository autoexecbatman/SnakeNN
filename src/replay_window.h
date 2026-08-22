#pragma once

#include <deque>
#include <vector>

#include "selfplay.h"

// The training window: the most recent self-play records, capped by bytes.
//
// Usage - one per run, fed a batch at a time:
//
//     ReplayWindow replay(1024ull * 1024 * 1024);   // byte cap, not a record count
//     replay.absorb(fresh);                         // moves them in, evicts the oldest
//     replay.size();                                // records held right now
//     replay.bytesUsed();                           // what they occupy, for the log
//     const TrainingRecord& record = replay[index];  // index drawn by the caller
//
// Capped by bytes rather than by count because a record is a different amount of memory on
// every board size: 3.2 KB at 10x10 against 12.8 KB at 20x20, so a count that is
// comfortable on one board takes the machine into swap on the next. That happened, and it
// is why the cap is expressed the way it is.
//
// Nothing here is thread-safe, and nothing needs to be: one trainer owns one window.
class ReplayWindow
{
public:
    // Holds at most `byte_limit` bytes of records. A limit smaller than one record leaves
    // the window empty after every absorb, which the caller sees as a size below its batch.
    explicit ReplayWindow(size_t byte_limit) : byte_limit_(byte_limit) {}

    // Moves every record of `fresh` in, then drops the oldest until the window is back
    // inside its limit. Leaves `fresh` holding moved-from records.
    //
    //     std::vector<TrainingRecord> fresh = /* one iteration of self-play */;
    //     replay.absorb(fresh);   // fresh is spent afterwards
    void absorb(std::vector<TrainingRecord>& fresh)
    {
        // Charge for each record as it arrives; bytesUsed is maintained rather than
        // recomputed, because a deque walk per iteration is the same cost as the eviction.
        for (TrainingRecord& record : fresh)
        {
            bytes_used_ += record.bytesUsed();
            records_.push_back(std::move(record));
        }
        // Oldest first, so the window is the most recent games and not an arbitrary slice.
        while (bytes_used_ > byte_limit_ && !records_.empty())
        {
            bytes_used_ -= records_.front().bytesUsed();
            records_.pop_front();
        }
    }

    // How many records are held.
    size_t size() const { return records_.size(); }

    // How much memory they occupy, which the iteration summary reports in megabytes.
    size_t bytesUsed() const { return bytes_used_; }

    // The record at `index`. The caller draws the index; this adds no bounds check beyond
    // the container's own.
    const TrainingRecord& operator[](size_t index) const { return records_[index]; }

private:
    std::deque<TrainingRecord> records_;
    size_t byte_limit_;
    size_t bytes_used_{ 0 };
};
