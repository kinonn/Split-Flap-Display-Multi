// Host test for LinkSettings.h — the (transport, groupCount) validator that
// gates the wired inter-display link. Includes the REAL production header;
// no re-implementation.
//
// Build & run (no Arduino required):
//   g++ -std=c++17 -Wall -Wextra -I src test/wired_link_settings_test.cpp -o /tmp/wired_link_settings_test && /tmp/wired_link_settings_test

#include "LinkSettings.h"

#include <cstdio>
#include <cstring>

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

static void test_esp_now_always_valid() {
    // Transport 0 keeps today's behavior: validity is decided by the frozen
    // ESP-NOW code, never by this validator.
    CHECK(validateWiredSettings(0, 1) == nullptr);
    CHECK(validateWiredSettings(0, 2) == nullptr);
    CHECK(validateWiredSettings(0, 6) == nullptr);
}

static void test_rs485_group_bounds() {
    CHECK(validateWiredSettings(1, 1) == nullptr);
    CHECK(validateWiredSettings(1, 2) == nullptr);
    CHECK(validateWiredSettings(1, 6) == nullptr);
    CHECK(validateWiredSettings(1, 0) != nullptr);
    CHECK(validateWiredSettings(1, 7) != nullptr);
    CHECK(validateWiredSettings(1, -1) != nullptr);
}

static void test_uart_capped_at_two() {
    CHECK(validateWiredSettings(2, 1) == nullptr);
    CHECK(validateWiredSettings(2, 2) == nullptr);
    CHECK(validateWiredSettings(2, 0) != nullptr);
    CHECK(validateWiredSettings(2, 3) != nullptr);
    CHECK(validateWiredSettings(2, 6) != nullptr);
}

static void test_unknown_transport() {
    CHECK(validateWiredSettings(3, 1) != nullptr);
    CHECK(validateWiredSettings(9, 1) != nullptr);
    CHECK(validateWiredSettings(-1, 1) != nullptr);
}

static void test_error_messages_stable() {
    // Callers surface these strings; pin them so a rewording is deliberate.
    CHECK(std::strcmp(validateWiredSettings(2, 3), "Direct UART supports at most 2 groups") == 0);
    CHECK(std::strcmp(validateWiredSettings(1, 7), "RS-485 supports 1..6 groups") == 0);
    CHECK(std::strcmp(validateWiredSettings(9, 1), "Unknown linkTransport") == 0);
}

int main() {
    test_esp_now_always_valid();
    test_rs485_group_bounds();
    test_uart_capped_at_two();
    test_unknown_transport();
    test_error_messages_stable();

    std::printf("%d checks, %d failures\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
