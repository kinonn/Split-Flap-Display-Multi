// Host test for SplitFlapModule — compiles the REAL production
// src/SplitFlapModule.cpp against minimal Arduino Wire/Serial stubs, so the
// exact firmware code paths are exercised (no re-implementation):
//
//   - start() re-energizes the pattern under the rotor WITHOUT moving:
//     repeated start() calls must not shift the drum (the held-module drift
//     bug), and the pattern written must be the one the rotor rests on.
//   - step() writes the next pattern, advancing stepNumber and position.
//   - stop() writes the idle pattern (all coils low).
//   - writeIO() error recovery: a failed i2c write sets hasErrored, the next
//     successful write clears it (the latched-error bug).
//
// Build & run (no Arduino required):
//   g++ -std=c++17 -Wall -Wextra -I src -I test/stubs test/splitflap_module_test.cpp src/SplitFlapModule.cpp -o
//   /tmp/splitflap_module_test && /tmp/splitflap_module_test

#include "Arduino.h"
#include "SplitFlapModule.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

// Stub globals (declared extern in test/stubs/Arduino.h)
SerialStub Serial;
TwoWire Wire;

// ---------------------------------------------------------------------------
// Test Wire stub: records every 16-bit value written to each address.
// ---------------------------------------------------------------------------

std::vector<std::pair<uint8_t, uint16_t>> g_writes; // (address, value) in order
bool g_failNextTransmission = false;

// test/stubs/Wire.h defines the TwoWire shell; these are the test hooks.
void wireReset() {
    g_writes.clear();
    g_failNextTransmission = false;
}

int wireWriteCount(uint8_t address) {
    int n = 0;
    for (auto &w : g_writes) {
        if (w.first == address) {
            n++;
        }
    }
    return n;
}

uint16_t wireLastValue(uint8_t address) {
    for (auto it = g_writes.rbegin(); it != g_writes.rend(); ++it) {
        if (it->first == address) {
            return it->second;
        }
    }
    return 0xFFFF;
}

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

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

static const int STEPS = 2048;

static SplitFlapModule makeModule() {
    return SplitFlapModule(0x20, STEPS, 0, 710, 48);
}

// Patterns from the production CoilStates table (verified in the module tests
// by literal value, independent of the private table).
static const uint16_t PATTERN_0 = 0b1111111111100111; // P01 + P02
static const uint16_t PATTERN_1 = 0b1111111111110011; // P01 + P04
static const uint16_t PATTERN_2 = 0b1111111111111001; // P03 + P04
static const uint16_t PATTERN_3 = 0b1111111111101101; // P02 + P03
static const uint16_t IDLE_PATTERN = 0b1111111111100001;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

// The core drift bug: start() must energize the pattern the rotor is resting
// on (the *previous* pattern, stepNumber - 1) and must not advance or rewind
// stepNumber. Repeating start() — what moveTo() used to do to every module on
// every move — must never write a different pattern or move the drum.
static void test_start_is_idempotent_hold() {
    SplitFlapModule m = makeModule();
    m.init();
    wireReset();

    // step once: stepNumber 0 -> 1, rotor now rests on pattern 0
    m.step();
    CHECK(wireLastValue(0x20) == PATTERN_0);
    const int posAfterStep = m.getPosition();
    CHECK(posAfterStep == 1);

    // hold twice in a row, as moveTo() does across consecutive moves
    m.start();
    CHECK(wireLastValue(0x20) == PATTERN_0);
    m.start();
    CHECK(wireLastValue(0x20) == PATTERN_0);
    CHECK(m.getPosition() == posAfterStep); // drum never moved

    // start() after stepping several times: rotor rests on the last pattern
    // written by step(), i.e. CoilStates[stepNumber - 1]
    m.step(); // writes pattern 1, stepNumber 1 -> 2
    CHECK(wireLastValue(0x20) == PATTERN_1);
    m.step(); // writes pattern 2, stepNumber 2 -> 3
    CHECK(wireLastValue(0x20) == PATTERN_2);
    m.start();
    CHECK(wireLastValue(0x20) == PATTERN_2);
    CHECK(m.getPosition() == 3);
}

// step() sequence: writes patterns 0,1,2,3 in order and advances position.
static void test_step_sequence() {
    SplitFlapModule m = makeModule();
    m.init();
    wireReset();

    m.step();
    CHECK(wireLastValue(0x20) == PATTERN_0);
    CHECK(m.getPosition() == 1);

    m.step();
    CHECK(wireLastValue(0x20) == PATTERN_1);
    CHECK(m.getPosition() == 2);

    m.step();
    CHECK(wireLastValue(0x20) == PATTERN_2);
    CHECK(m.getPosition() == 3);

    m.step();
    CHECK(wireLastValue(0x20) == PATTERN_3);
    CHECK(m.getPosition() == 4);

    // wraps around the electrical cycle
    m.step();
    CHECK(wireLastValue(0x20) == PATTERN_0);
    CHECK(m.getPosition() == 5);
}

// stop() releases the coils: idle pattern, no state change.
static void test_stop_writes_idle() {
    SplitFlapModule m = makeModule();
    m.init();
    wireReset();

    m.step();
    m.step();
    m.stop();
    CHECK(wireLastValue(0x20) == IDLE_PATTERN);
    CHECK(m.getPosition() == 2);

    // start() after stop() must re-energize the pattern under the rotor
    // (pattern 1 = CoilStates[2 - 1]) — this is the hold after release path.
    m.start();
    CHECK(wireLastValue(0x20) == PATTERN_1);
    CHECK(m.getPosition() == 2);
}

// init() writes the idle state and does not move anything.
static void test_init_writes_idle() {
    SplitFlapModule m = makeModule();
    wireReset();
    m.init();
    CHECK(wireLastValue(0x20) == IDLE_PATTERN);
    CHECK(m.getPosition() == 0);
}

// Latched-error bug: one failed i2c write must not disable the module
// forever. hasErrored is cleared by the next successful write.
static void test_write_error_recovery() {
    SplitFlapModule m = makeModule();
    m.init();
    wireReset();

    CHECK(m.getHasErrored() == false);

    g_failNextTransmission = true;
    m.step(); // this write fails
    CHECK(m.getHasErrored() == true);

    m.step(); // this write succeeds
    CHECK(m.getHasErrored() == false);
}

int main() {
    test_start_is_idempotent_hold();
    test_step_sequence();
    test_stop_writes_idle();
    test_init_writes_idle();
    test_write_error_recovery();

    std::printf("%d checks, %d failures\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
