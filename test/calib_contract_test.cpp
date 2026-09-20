// Host test for the served calibration contract (src/web/calib-contract.json):
// the drum tables and limits the calibration tools trust must match the
// firmware that serves them. Compiles the REAL src/SplitFlapModule.cpp (drum
// tables) and includes the real src/CalibApi.h / src/PendingActions.h.
//
// Build & run FROM THE REPO ROOT (the JSON path is relative to it):
//   g++ -std=c++17 -Wall -Wextra -I src -I test/stubs test/calib_contract_test.cpp
//   src/SplitFlapModule.cpp -o /tmp/calib_contract_test && /tmp/calib_contract_test
//
// This is the check src/CalibApi.h used to claim existed but did not (issue
// kinonn-bot#46): the contract duplicates the drum tables and limits by hand,
// so a drifted copy silently mistunes every module a tool touches.

#include "CalibApi.h"
#include "PendingActions.h"
#include "SplitFlapModule.h"

#include <climits>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>

static std::string g_json;
static int failures = 0;
static int checks = 0;

// Host link stubs for the real src/SplitFlapModule.cpp: it references the
// Arduino globals declared in test/stubs/Arduino.h. Kept minimal — this test
// never moves a drum, it only reads the canonical tables.
SerialStub Serial;
TwoWire Wire;
std::vector<std::pair<uint8_t, uint16_t>> g_writes;
bool g_failNextTransmission = false;

#define CHECK(cond)                                                     \
    do {                                                                \
        checks++;                                                       \
        if (! (cond)) {                                                 \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            failures++;                                                 \
        }                                                               \
    } while (0)

static bool loadJson() {
    std::ifstream in("src/web/calib-contract.json", std::ios::binary);
    if (! in) {
        std::printf("cannot open src/web/calib-contract.json — run this from the repo root\n");
        return false;
    }
    std::ostringstream buffer;
    buffer << in.rdbuf();
    g_json = buffer.str();
    return true;
}

// "\"key\": \"value\"" -> value (empty string when the key is absent).
static std::string jsonString(const std::string &key) {
    std::string needle = "\"" + key + "\": \"";
    size_t at = g_json.find(needle);
    if (at == std::string::npos) {
        return "<missing key: " + key + ">";
    }
    size_t start = at + needle.size();
    size_t end = g_json.find('"', start);
    if (end == std::string::npos) {
        return "<unterminated value: " + key + ">";
    }
    return g_json.substr(start, end - start);
}

// "\"key\": 123" -> 123 (LONG_MIN sentinel when the key is absent).
static long jsonInt(const std::string &key) {
    std::string needle = "\"" + key + "\": ";
    size_t at = g_json.find(needle);
    if (at == std::string::npos) {
        return LONG_MIN;
    }
    return std::strtol(g_json.c_str() + at + needle.size(), nullptr, 10);
}

int main() {
    if (! loadJson()) {
        return 2;
    }

    // Drum tables: the contract's charsets are what the tools decode glyphs
    // against, so they must be the firmware's canonical tables, in order.
    int len37 = 0;
    const char *drum37 = SplitFlapModule::drumOrder(37, len37);
    std::string contract37 = jsonString("37");
    if (contract37 != std::string(drum37, len37)) {
        std::printf("  contract \"37\": %s\n  firmware     : %s\n", contract37.c_str(), std::string(drum37, len37).c_str());
    }
    CHECK(len37 == 37);
    CHECK(contract37 == std::string(drum37, len37));

    int len48 = 0;
    const char *drum48 = SplitFlapModule::drumOrder(48, len48);
    std::string contract48 = jsonString("48");
    if (contract48 != std::string(drum48, len48)) {
        std::printf("  contract \"48\": %s\n  firmware     : %s\n", contract48.c_str(), std::string(drum48, len48).c_str());
    }
    CHECK(len48 == 48);
    CHECK(contract48 == std::string(drum48, len48));

    // Limits and mode: every one of these is a number the tools act on.
    CHECK(jsonInt("contractVersion") == CALIB_CONTRACT_VERSION);
    CHECK(jsonInt("holdMode") == CALIB_HOLD_MODE);
    CHECK(jsonInt("charOffsetMin") == CALIB_CHAR_OFFSET_MIN);
    CHECK(jsonInt("charOffsetMax") == CALIB_CHAR_OFFSET_MAX);
    CHECK(jsonInt("maxGroups") == CALIB_MAX_GROUPS);
    CHECK(jsonInt("maxModulesPerGroup") == CALIB_MAX_MODULES);
    CHECK(jsonInt("maxTotalModules") == CALIB_MAX_FRAME);
    CHECK(jsonInt("maxNudgesPerBatch") == PendingActions::CalibBatchPreview::MAX_NUDGES);
    CHECK(jsonInt("maxNudgesPerRemoteScope") == CALIB_MAX_NUDGES_PER_REMOTE);

    // Every shipped copy of the contract must stay in step with the served
    // one: tools/calib carries a mirror that the tool reads offline
    // (issue kinonn-bot#38's first draft shipped it stale).
    {
        std::ifstream served("src/web/calib-contract.json", std::ios::binary);
        std::ostringstream servedBuf;
        servedBuf << served.rdbuf();
        const char *mirrors[] = {
            "tools/calib/calib/contract.json",
        };
        for (const char *path : mirrors) {
            std::ifstream in(path, std::ios::binary);
            std::ostringstream buf;
            buf << in.rdbuf();
            if (buf.str() != servedBuf.str()) {
                std::printf("FAIL mirror differs from the served contract: %s\n", path);
            }
            CHECK(! buf.str().empty());
            CHECK(buf.str() == servedBuf.str());
        }
    }

    if (failures == 0) {
        std::printf("calib_contract_test: all %d checks passed\n", checks);
        return 0;
    }
    std::printf("calib_contract_test: %d/%d checks FAILED\n", failures, checks);
    return 1;
}
