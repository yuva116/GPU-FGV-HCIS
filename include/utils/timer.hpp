#pragma once

#include <chrono>

namespace spanner {

class Timer {
public:
    Timer();

    void reset();

    double elapsed_seconds() const;
    double elapsed_milliseconds() const;

private:
    using Clock = std::chrono::steady_clock;

    Clock::time_point start_;
};

} // namespace spanner