// Host tests for the homing trigger consolidation: the REAL
// src/CalibrationTriggers.h decision logic plus the REAL
// JsonSettings::parseIntVector/parseIntMatrix it compares with.
//
// What is under test:
//   1. calibrationChanged() fires only on numeric calibration changes —
//      a whitespace-only CSV reformat must NOT schedule a reload (the old
//      raw-string comparison did).
//   2. magnetPosition is a calibration key like displayOffset: changing it
//      triggers, and marks ALL modules affected (it shifts every module's
//      magnet target). The old handler never triggered on it at all.
//   3. affectedModules() homes exactly the modules whose own offsets moved,
//      and homes nothing when nothing changed.
//   4. The CSV parsers tolerate the firmware's lenient grammar (whitespace,
//      trailing separators, junk-skipping) identically on both sides of the
//      comparison, so equivalent spellings compare equal.
//
// Build & run (no Arduino required):
//   g++ -std=c++17 -Wall -Wextra -pthread -I src -I test/stubs_jsonsettings test/calibration_triggers_test.cpp src/JsonSettings.cpp src/JsonSetting.cpp -o /tmp/calibration_triggers_test && /tmp/calibration_triggers_test

#include "CalibrationTriggers.h"
#include "JsonSettings.h"

#include <atomic>
#include <cstdio>
#include <map>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

// Stub globals backing test/stubs_jsonsettings (same set jsonsettings_test.cpp
// defines): linking the REAL JsonSettings.cpp pulls in fromJson()/Preferences
// references even though these tests only call the pure CSV parsers.
std::vector<std::string> g_serialLines;
std::map<std::pair<std::string, std::string>, std::string> g_nvs;
std::mutex g_nvsMutex;
std::atomic<int> g_openHandles{0};
SerialStub Serial;

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

static CalibrationSnapshot makeStored() {
    CalibrationSnapshot s;
    s.displayOffset = 3;
    s.magnetPosition = 730;
    s.moduleOffsets = {0, 1, -2, 0, 0, 0, 0, 0};
    s.charOffsets = std::vector<std::vector<int>>(8, std::vector<int>(48, 0));
    return s;
}

static void test_no_change() {
    CalibrationSnapshot stored = makeStored();
    CalibrationSnapshot incoming = stored;
    CHECK(! calibrationChanged(stored, incoming));
    std::vector<char> affected = affectedModules(stored, incoming, 8);
    CHECK(affected.size() == 8);
    for (int i = 0; i < 8; i++) {
        CHECK(affected[i] == 0);
    }
}

static void test_csv_reformat_is_not_a_change() {
    // Same numbers, different formatting: must parse equal and not trigger.
    CHECK(
        JsonSettings::parseIntVector("0,1,-2,0,0,0,0,0") ==
        JsonSettings::parseIntVector("0, 1, -2, 0, 0, 0, 0, 0,")
    );
    CHECK(
        JsonSettings::parseIntMatrix("0,0;1,2") == JsonSettings::parseIntMatrix("0, 0 ; 1, 2 ;")
    );

    CalibrationSnapshot stored = makeStored();
    CalibrationSnapshot incoming = stored;
    incoming.moduleOffsets = JsonSettings::parseIntVector("0, 1, -2, 0, 0, 0, 0, 0,");
    CHECK(! calibrationChanged(stored, incoming));
}

static void test_module_offset_change_affects_one_module() {
    CalibrationSnapshot stored = makeStored();
    CalibrationSnapshot incoming = stored;
    incoming.moduleOffsets = JsonSettings::parseIntVector("0,1,-2,5,0,0,0,0");
    CHECK(calibrationChanged(stored, incoming));
    std::vector<char> affected = affectedModules(stored, incoming, 8);
    for (int i = 0; i < 8; i++) {
        CHECK(affected[i] == (i == 3 ? 1 : 0));
    }
}

static void test_char_offset_change_affects_one_module() {
    CalibrationSnapshot stored = makeStored();
    CalibrationSnapshot incoming = stored;
    incoming.charOffsets[5][12] = 7;
    CHECK(calibrationChanged(stored, incoming));
    std::vector<char> affected = affectedModules(stored, incoming, 8);
    for (int i = 0; i < 8; i++) {
        CHECK(affected[i] == (i == 5 ? 1 : 0));
    }
}

static void test_display_offset_change_affects_all() {
    CalibrationSnapshot stored = makeStored();
    CalibrationSnapshot incoming = stored;
    incoming.displayOffset = 4;
    CHECK(calibrationChanged(stored, incoming));
    std::vector<char> affected = affectedModules(stored, incoming, 8);
    for (int i = 0; i < 8; i++) {
        CHECK(affected[i] == 1);
    }
}

static void test_magnet_position_change_affects_all() {
    CalibrationSnapshot stored = makeStored();
    CalibrationSnapshot incoming = stored;
    incoming.magnetPosition = 615;
    CHECK(calibrationChanged(stored, incoming));
    std::vector<char> affected = affectedModules(stored, incoming, 8);
    for (int i = 0; i < 8; i++) {
        CHECK(affected[i] == 1);
    }
}

static void test_parser_leniency() {
    // Lenient grammar ("1,x,3" yields {1, 3}) must hold on both sides of the
    // comparison so equivalent spellings can never compare unequal.
    CHECK(JsonSettings::parseIntVector("1,x,3") == JsonSettings::parseIntVector("1, 3"));
    CHECK(JsonSettings::parseIntVector("") == std::vector<int>());
    CHECK(JsonSettings::parseIntMatrix("") == std::vector<std::vector<int>>());
    std::vector<std::vector<int>> expected = {{1, 2}, {3, 4}};
    CHECK(JsonSettings::parseIntMatrix("1,2;3,4") == expected);
}

int main() {
    test_no_change();
    test_csv_reformat_is_not_a_change();
    test_module_offset_change_affects_one_module();
    test_char_offset_change_affects_one_module();
    test_display_offset_change_affects_all();
    test_magnet_position_change_affects_all();
    test_parser_leniency();

    std::printf("%d checks, %d failures\n", checks, failures);
    return failures != 0 ? 1 : 0;
}
