#pragma once
#include <iostream>
#include <chrono>

using namespace std;
using namespace chrono;

struct Timer {
    high_resolution_clock::time_point start, end;
    duration<double> elapsed;

    void startTimer() {
        start = high_resolution_clock::now();
    }

    void stopTimer() {
        end = high_resolution_clock::now();
        elapsed = duration_cast<duration<double>>(end - start);
    }

    auto getPeriod() {
		return high_resolution_clock::period::num;
    }

    double getElapsedTime() {
        return elapsed.count();
    }
};