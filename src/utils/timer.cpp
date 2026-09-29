#include "utils/timer.hpp"

namespace spanner {

Timer::Timer() {
    reset();
}

void Timer::reset() {
    start_ = Clock::now();
}

double Timer::elapsed_seconds() const {
    const auto elapsed = Clock::now() - start_;
    return std::chrono::duration<double>(elapsed).count();
}

double Timer::elapsed_milliseconds() const {
    const auto elapsed = Clock::now() - start_;
    return std::chrono::duration<double, std::milli>(elapsed).count();
}

} 