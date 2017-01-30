#pragma once

#include <chrono>

/* Requires proper c++11/14 support!!
template<typename TimeT = std::chrono::microseconds>
struct Measure
{
    template<typename F, typename ...Args>
    static typename TimeT::rep execution(F func, Args&&.. args)
    {
        auto start = std::chrono::high_resolution_clock::now();
        func(std::forward<Args>(args)...);
        auto duration = std::chrono::duration_cast<TimeT>(std::chrono::high_resolution_clock::now() - start);
        return duration.count();
    }
};
*/

template<typename TimeT = std::chrono::microseconds,
         typename ClockT = std::chrono::high_resolution_clock,
         typename DurationT = double>
class StopWatch
{
private:
    std::chrono::time_point<ClockT> _start, _end;
public:
    StopWatch() { start(); }
    void start() { _start = _end = ClockT::now(); }
    DurationT stop() { _end = ClockT::now(); return elapsed(); }
    DurationT elapsed()
    {
        auto delta = std::chrono::duration_cast<TimeT>(_end-_start);
        return delta.count();
    }
};
