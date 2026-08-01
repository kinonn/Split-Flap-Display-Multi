// Host test for SplitFlapMotorScheduler — the concurrency cap that
// SplitFlapDisplay::moveTo() uses to bound the boot-homing current spike.
//
// This includes the REAL production header (src/SplitFlapMotorScheduler.h);
// it does not re-implement the logic. It verifies the exact code path the
// firmware calls: start()/finish() around the stepping loop.
//
// Build & run (no Arduino required):
//   g++ -std=c++11 -Wall -Wextra -I src test/motor_scheduler_test.cpp -o /tmp/motor_scheduler_test && /tmp/motor_scheduler_test

#include <cassert>
#include <cstdio>

#include "SplitFlapMotorScheduler.h"

static int failures = 0;
static int checks = 0;

#define CHECK(cond)                                                       \
    do {                                                                  \
        checks++;                                                         \
        if (!(cond)) {                                                    \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);   \
            failures++;                                                   \
        }                                                                 \
    } while (0)

// Simulates a homing move with the scheduler: 8 motors each needing to run,
// a cap of 2. Motors "finish" after N steps in order; we assert the cap is
// never exceeded and every motor eventually gets to run.
static void simulateMove(int numModules, int cap, int stepsPerMotor) {
    SplitFlapMotorScheduler sched(numModules, cap);
    int remainingSteps[8] = {0};
    int activeCount = 0;

    for (int i = 0; i < numModules; i++) {
        remainingSteps[i] = stepsPerMotor;
    }

    int guard = 0;
    while (true) {
        bool anyLeft = false;
        for (int i = 0; i < numModules; i++) {
            if (remainingSteps[i] <= 0) {
                continue;
            }
            anyLeft = true;
            if (!sched.isRunning(i)) {
                if (sched.start(i)) {
                    activeCount++;
                }
            }
            // Only a running motor may consume a step.
            if (sched.isRunning(i)) {
                remainingSteps[i]--;
                if (remainingSteps[i] == 0) {
                    sched.finish(i);
                    activeCount--;
                }
            }
        }
        CHECK(sched.activeCount() == activeCount);
        CHECK(activeCount <= (cap < 0 ? numModules : cap)); // cap never exceeded
        if (!anyLeft) {
            break;
        }
        if (++guard > 100000) {
            std::printf("FAIL: simulated move did not terminate (deadlock)\n");
            failures++;
            break;
        }
    }

    // All motors must have completed.
    for (int i = 0; i < numModules; i++) {
        CHECK(remainingSteps[i] == 0);
        CHECK(!sched.isRunning(i));
    }
    CHECK(sched.activeCount() == 0);
}

int main() {
    // --- Basic slot semantics (cap = 2) ---
    {
        SplitFlapMotorScheduler sched(8, 2);
        CHECK(sched.start(0));
        CHECK(sched.start(1));
        CHECK(!sched.start(2)); // cap reached
        CHECK(sched.activeCount() == 2);

        CHECK(sched.start(0)); // already running: no-op, keeps slot
        CHECK(sched.activeCount() == 2);

        sched.finish(1);
        CHECK(sched.activeCount() == 1);
        CHECK(sched.start(2)); // slot freed
        CHECK(sched.activeCount() == 2);

        sched.finish(0);
        sched.finish(2);
        CHECK(sched.activeCount() == 0);
        CHECK(!sched.isRunning(2));
    }

    // --- Out-of-range handling ---
    {
        SplitFlapMotorScheduler sched(8, 2);
        CHECK(!sched.start(-1));
        CHECK(!sched.start(8));  // == numModules
        CHECK(!sched.start(999));
        sched.finish(8);         // must not crash / no-op
        sched.finish(-1);
        CHECK(sched.activeCount() == 0);
    }

    // --- Cap = 1: strictly serialized ---
    {
        SplitFlapMotorScheduler sched(8, 1);
        CHECK(sched.start(0));
        CHECK(!sched.start(1));
        CHECK(!sched.start(2));
        sched.finish(0);
        CHECK(sched.start(1));
        CHECK(!sched.start(2));
        CHECK(sched.activeCount() == 1);
    }

    // --- Cap = -1: unlimited (normal message writes) ---
    {
        SplitFlapMotorScheduler sched(8, -1);
        for (int i = 0; i < 8; i++) {
            CHECK(sched.start(i));
        }
        CHECK(sched.activeCount() == 8);
    }

    // --- Cap = 0 must not deadlock the caller (clamped to 1 motor) ---
    // Homing clamps to >= 1 via constrain() in SplitFlapDisplay.cpp, but
    // the scheduler guards anyway.
    {
        SplitFlapMotorScheduler sched(8, 0);
        CHECK(sched.start(0));
        CHECK(!sched.start(1));
        sched.finish(0);
        CHECK(sched.start(1));
    }

    // --- Full homing simulations across caps/module counts ---
    simulateMove(8, 2, 2048); // 8 modules, cap 2, full rotation each
    simulateMove(3, 2, 2048); // 3 modules (slave group)
    simulateMove(11 > 8 ? 8 : 11, 2, 2048); // scheduler max is 8 modules per board
    simulateMove(8, 1, 2048); // strictly serialized homing
    simulateMove(8, 4, 2048);
    simulateMove(8, -1, 2048); // uncapped must also complete correctly
    simulateMove(8, 2, 1);     // tiny moves

    if (failures == 0) {
        std::printf("PASS: motor scheduler (%d checks)\n", checks);
        return 0;
    }
    std::printf("FAIL: %d/%d checks failed\n", failures, checks);
    return 1;
}
