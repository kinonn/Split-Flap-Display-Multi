// Host tests for CalibHoldTracker — the pre-hold display mode save/restore
// state machine behind POST /api/calib/hold (issue kinonn-bot#35: hold
// release used to land in mode 0 unconditionally, losing clock/date mode).
//
// Includes the REAL production header; no re-implementation.
//
// Build & run (no Arduino required):
//   g++ -std=c++17 -Wall -Wextra -I src test/calib_hold_test.cpp -o
//   /tmp/calib_hold_test && /tmp/calib_hold_test

#include "CalibApi.h"

#include <cstdio>

static int failures = 0;
static int checks = 0;

#define CHECK(cond)                                                     \
    do {                                                                \
        checks++;                                                       \
        if (! (cond)) {                                                 \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            failures++;                                                 \
        }                                                               \
    } while (0)

int main() {
    // Engage from date mode (2): writes hold mode, release restores 2.
    {
        CalibHoldTracker t;
        CHECK(t.engage(2) == CALIB_HOLD_MODE);
        CHECK(t.release() == 2);
    }
    // Engage from single-input (0): round-trips back to 0.
    {
        CalibHoldTracker t;
        CHECK(t.engage(0) == CALIB_HOLD_MODE);
        CHECK(t.release() == 0);
    }
    // Double-engage must not overwrite the saved mode with hold itself.
    {
        CalibHoldTracker t;
        CHECK(t.engage(3) == CALIB_HOLD_MODE);
        CHECK(t.engage(CALIB_HOLD_MODE) == CALIB_HOLD_MODE);
        CHECK(t.release() == 3);
    }
    // Release without any engage since boot: historic fallback to 0.
    {
        CalibHoldTracker t;
        CHECK(t.release() == 0);
    }
    // Release clears state: a stale mode never leaks into a later cycle.
    {
        CalibHoldTracker t;
        CHECK(t.engage(2) == CALIB_HOLD_MODE);
        CHECK(t.release() == 2);
        CHECK(t.release() == 0);
    }
    // Second hold cycle re-arms with the new pre-hold mode.
    {
        CalibHoldTracker t;
        CHECK(t.engage(2) == CALIB_HOLD_MODE);
        CHECK(t.release() == 2);
        CHECK(t.engage(5) == CALIB_HOLD_MODE);
        CHECK(t.release() == 5);
    }

    if (failures == 0) {
        std::printf("calib_hold_test: all %d checks passed\n", checks);
        return 0;
    }
    std::printf("calib_hold_test: %d/%d checks FAILED\n", failures, checks);
    return 1;
}
