#ifndef SEARCHDEADLINE_H
#define SEARCHDEADLINE_H

#include <chrono>

using SearchClock = std::chrono::steady_clock;
using SearchDeadline = SearchClock::time_point;

inline bool SearchDeadlineReached(const SearchDeadline deadline) {
    return SearchClock::now() >= deadline;
}

#endif  // SEARCHDEADLINE_H
