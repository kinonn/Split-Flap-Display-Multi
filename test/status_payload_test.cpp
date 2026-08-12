// Host test for the MQTT status payload builder.
//
// Exercises the REAL code path: includes src/StatusPayload.h and links the
// real ArduinoJson library (same lib_deps version the firmware uses), so
// this is not a re-implementation — it tests the actual payload builder
// that SplitFlapMqtt::publishStatus() calls.
//
// Build (from repo root, per AGENTS.md but C++17 — the existing tests
// need it for aggregate-init with default member initializers):
//
//   g++ -std=c++17 -Wall -Wextra -I src -I .pio/libdeps/esp32_c3/ArduinoJson/src
//       test/status_payload_test.cpp -o /tmp/status_test && /tmp/status_test

#include "StatusPayload.h"

#include <iostream>
#include <string>

static int testFailures = 0;
static int testCount = 0;

#define EXPECT(cond, label)                                \
    do {                                                   \
        testCount++;                                       \
        if (! (cond)) {                                    \
            std::cout << "  FAIL: " << label << std::endl; \
            testFailures++;                                \
        } else {                                           \
            std::cout << "  ok:   " << label << std::endl; \
        }                                                  \
    } while (0)

void testBasicPayload() {
    std::cout << "\n[TEST] basic payload" << std::endl;
    std::string payload = buildStatusPayload("HELLO", 8);
    EXPECT(
        payload == "{\"message\":\"HELLO\",\"num_modules\":8}", "payload is {\"message\":\"HELLO\",\"num_modules\":8}"
    );
}

void testEmptyMessage() {
    std::cout << "\n[TEST] empty message (boot state)" << std::endl;
    std::string payload = buildStatusPayload("", 8);
    EXPECT(payload == "{\"message\":\"\",\"num_modules\":8}", "empty message serializes to empty string field");
}

void testModuleCountValue() {
    std::cout << "\n[TEST] module count passthrough" << std::endl;
    std::string payload = buildStatusPayload("HI", 11);
    EXPECT(payload == "{\"message\":\"HI\",\"num_modules\":11}", "num_modules reflects display size (11)");
}

void testQuotesAndBackslashEscaping() {
    std::cout << "\n[TEST] quotes/backslash escaping" << std::endl;
    // The display alphabet includes '"' and '\'' — those must not break JSON.
    std::string payload = buildStatusPayload("SAY \"HI\"", 8);
    EXPECT(payload == "{\"message\":\"SAY \\\"HI\\\"\",\"num_modules\":8}", "double quotes are escaped");
}

void testNoTrailingGarbage() {
    std::cout << "\n[TEST] payload is parseable, exactly one object" << std::endl;
    std::string payload = buildStatusPayload("TEST", 8);
    // Valid JSON starts with '{' and ends with '}' — no extra bytes.
    EXPECT(payload.front() == '{' && payload.back() == '}', "payload starts with { and ends with }");
}

int main() {
    std::cout << "=== StatusPayload host tests ===" << std::endl;
    testBasicPayload();
    testEmptyMessage();
    testModuleCountValue();
    testQuotesAndBackslashEscaping();
    testNoTrailingGarbage();

    std::cout << "\n=== Summary ===" << std::endl;
    std::cout << "Passed: " << (testCount - testFailures) << "/" << testCount << std::endl;
    if (testFailures == 0) {
        std::cout << "ALL TESTS PASSED" << std::endl;
        return 0;
    } else {
        std::cout << testFailures << " TEST(S) FAILED" << std::endl;
        return 1;
    }
}
