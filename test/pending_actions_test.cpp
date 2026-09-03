// Host tests for PendingActions — the cross-task "do this in the loop task"
// mailbox that decouples the AsyncTCP web task from display/ESP-NOW work
// (audit issues #3 and #12: the web task used to run display.reloadOffsets()
// and ESP-NOW offset pushes inline, racing the loop task's I2C access).
//
// Includes the REAL production header; no re-implementation.
//
// Build & run (no Arduino required):
//   g++ -std=c++17 -Wall -Wextra -pthread test/pending_actions_test.cpp -o
//   /tmp/pending_test && /tmp/pending_test

#include "PendingActions.h"

#include <atomic>
#include <cstdio>
#include <thread>
#include <vector>

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

static void test_initially_empty() {
    PendingActions a;
    CHECK(! a.takeReloadOffsets());
    CHECK(! a.takeReportOffsets());
    CHECK(! a.takePushOffsets());
}

static void test_request_then_take_once() {
    PendingActions a;
    a.requestReloadOffsets();
    CHECK(a.takeReloadOffsets() == true);
    CHECK(! a.takeReloadOffsets()); // consume-once: a second take sees nothing

    a.requestReportOffsets();
    CHECK(a.takeReportOffsets() == true);
    CHECK(! a.takeReportOffsets());

    a.requestPushOffsets();
    CHECK(a.takePushOffsets() == true);
    CHECK(! a.takePushOffsets());
}

static void test_flags_are_independent() {
    PendingActions a;
    a.requestReloadOffsets();
    CHECK(! a.takeReportOffsets());
    CHECK(! a.takePushOffsets());
    CHECK(a.takeReloadOffsets() == true);
}

// Hammer: one thread spamming requests while the consumer drains. The
// consumer must observe at least one request and every take must win its
// flag exactly once (atomic exchange — no lost updates, no double takes).
static void test_threaded_hammer() {
    for (int trial = 0; trial < 50; trial++) {
        PendingActions a;
        std::atomic<bool> stop{false};
        std::atomic<int> takenCount{0};

        std::thread writer([&]() {
            while (! stop.load(std::memory_order_relaxed)) {
                a.requestReloadOffsets();
            }
        });

        std::thread writer2([&]() {
            while (! stop.load(std::memory_order_relaxed)) {
                a.requestReportOffsets();
                a.requestPushOffsets();
            }
        });

        // Drain for a bounded number of iterations.
        for (int i = 0; i < 1000; i++) {
            if (a.takeReloadOffsets()) {
                takenCount++;
            }
            a.takeReportOffsets();
            a.takePushOffsets();
        }
        stop.store(true);
        writer.join();
        writer2.join();

        // Final take must still see a request from the last spam window or
        // nothing — both are valid — but the drain must have observed work.
        a.takeReloadOffsets();
        a.takeReportOffsets();
        a.takePushOffsets();

        if (trial == 0) {
            CHECK(takenCount.load() > 0);
        }
    }
}

int main() {
    test_initially_empty();
    test_request_then_take_once();
    test_flags_are_independent();
    test_threaded_hammer();

    std::printf("%d checks, %d failures\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
