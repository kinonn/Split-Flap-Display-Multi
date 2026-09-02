// Host tests for JsonSettings — compile the REAL src/JsonSettings.cpp and
// src/JsonSetting.cpp against Arduino/ArduinoJson/Preferences stubs.
//
// What is under test:
//   1. NVS handle discipline: every accessor opens its own Preferences handle
//      and closes it before returning. After any sequence of calls — including
//      fromJson()/toJson()/reset() — no handle may be left open. This is the
//      property whose violation made one task's end() close another task's
//      handle on the ESP32 (web/ESP-NOW writes racing loop-task reads).
//   2. Functional behavior is preserved: defaults when unset, round-trips via
//      toJson/fromJson, reset() restoring defaults, unknown-key tolerance
//      (logged, skipped — never throws), validation failure paths.
//   3. reset() clears THIS namespace (regression: it used to hardcode
//      "config" instead of using the instance name).
//
// The threaded hammer test lives in jsonsettings_threads_test.cpp (separate
// binary, pthread-based) so this file stays single-threaded and deterministic.
//
// Build & run (no Arduino required):
//   g++ -std=c++17 -Wall -Wextra -pthread -I src -I test/stubs_jsonsettings test/jsonsettings_test.cpp src/JsonSettings.cpp src/JsonSetting.cpp -o /tmp/jsonsettings_test && /tmp/jsonsettings_test

#include <cstdio>
#include <map>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "Arduino.h"
#include "ArduinoJson.h"
#include "JsonSettings.h"
#include "Preferences.h"

// Stub globals
std::vector<std::string> g_serialLines;
std::map<std::pair<std::string, std::string>, std::string> g_nvs;
std::mutex g_nvsMutex;
std::atomic<int> g_openHandles{0};

SerialStub Serial;

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

static int failures = 0;
static int checks = 0;

#define CHECK(cond)                                                     \
    do {                                                                \
        checks++;                                                       \
        if (!(cond)) {                                                  \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            failures++;                                                 \
        }                                                               \
    } while (0)

// A settings schema mirroring the firmware's variety of types.
static std::map<String, JsonSetting> testSchema() {
    return {
        {"mode", JsonSetting(2)},
        {"displayOffset", JsonSetting(0)},
        {"maxVel", JsonSetting(15.0f)},
        {"charset", JsonSetting(48)},
        {"timeFormat", JsonSetting("{HH}:{MM}")},
        {"moduleOffsets", JsonSetting(std::vector<int>{0, 0, 0, 0, 0, 0, 0, 0})},
        {"charOffsets",
         JsonSetting(std::vector<std::vector<int>>{{0, 0, 0, 0, 0, 0, 0, 0}})},
    };
}

static JsonSettings makeSettings() {
    return JsonSettings("testns", testSchema());
}

static void nvsReset() {
    g_nvs.clear();
    g_openHandles = 0;
}

// ---------------------------------------------------------------------------
// 1. Handle discipline
// ---------------------------------------------------------------------------

static void test_no_leaked_handles_across_all_accessors() {
    nvsReset();
    {
        JsonSettings settings = makeSettings();

        settings.getString("timeFormat");
        settings.getInt("mode");
        settings.getFloat("maxVel");
        settings.getIntVector("moduleOffsets");
        settings.getIntMatrix("charOffsets");
        settings.putString("timeFormat", "{HH}:{MM}");
        settings.putInt("mode", 1);
        settings.putFloat("maxVel", 12.5f);
        settings.putIntVector("moduleOffsets", std::vector<int>{1, 2});
        settings.putIntMatrix("charOffsets", std::vector<std::vector<int>>{{3, 4}});
        settings.toJson();

        CHECK(g_openHandles == 0);
    }
    CHECK(g_openHandles == 0);
}

static void test_no_leaked_handle_on_validation_failure() {
    nvsReset();
    {
        JsonSettings settings = makeSettings();

        JsonDocument doc;
        doc["moduleOffsets"] = String("1,x,3"); // non-integer -> validate() fails
        bool ok = settings.fromJson(doc);

        CHECK(ok == false);
        CHECK(g_openHandles == 0); // error path must not leak the handle
        CHECK(settings.getLastValidationKey() == String("moduleOffsets"));
    }
    CHECK(g_openHandles == 0);
}

// ---------------------------------------------------------------------------
// 2. Functional behavior
// ---------------------------------------------------------------------------

static void test_defaults_when_unset() {
    nvsReset();
    JsonSettings settings = makeSettings();

    CHECK(settings.getInt("mode") == 2);
    CHECK(settings.getFloat("maxVel") == 15.0f);
    CHECK(std::string(settings.getString("timeFormat").c_str()) == "{HH}:{MM}");

    std::vector<int> offs = settings.getIntVector("moduleOffsets");
    CHECK(offs.size() == 8);
    CHECK(offs[0] == 0 && offs[7] == 0);
}

static void test_put_then_get_roundtrip() {
    nvsReset();
    JsonSettings settings = makeSettings();

    settings.putInt("mode", 5);
    CHECK(settings.getInt("mode") == 5);

    settings.putString("timeFormat", "{HH}:{MM}:{SS}");
    CHECK(std::string(settings.getString("timeFormat").c_str()) == "{HH}:{MM}:{SS}");

    settings.putIntVector("moduleOffsets", std::vector<int>{10, -20, 30});
    std::vector<int> offs = settings.getIntVector("moduleOffsets");
    CHECK(offs.size() == 3);
    CHECK(offs[0] == 10 && offs[1] == -20 && offs[2] == 30);
}

static void test_toJson_fromJson_roundtrip() {
    nvsReset();
    JsonSettings settings = makeSettings();

    settings.putInt("mode", 3);
    settings.putString("timeFormat", "{hh}:{MM} {AMPM}");
    settings.putIntVector("moduleOffsets", std::vector<int>{7, 8, 9});

    JsonDocument snapshot = settings.toJson();
    CHECK(g_openHandles == 0);

    JsonSettings other = makeSettings();
    bool ok = other.fromJson(snapshot);
    CHECK(ok == true);
    CHECK(g_openHandles == 0);

    CHECK(other.getInt("mode") == 3);
    CHECK(std::string(other.getString("timeFormat").c_str()) == "{hh}:{MM} {AMPM}");
    std::vector<int> offs = other.getIntVector("moduleOffsets");
    CHECK(offs.size() == 3);
    CHECK(offs[0] == 7 && offs[1] == 8 && offs[2] == 9);
}

static void test_from_json_rejects_invalid_vector() {
    nvsReset();
    JsonSettings settings = makeSettings();

    JsonDocument doc;
    doc["moduleOffsets"] = String("1,oops,3");
    bool ok = settings.fromJson(doc);
    CHECK(ok == false);
    // Nothing was written for the rejected key.
    CHECK(settings.getIntVector("moduleOffsets").size() == 8); // default
}

static void test_unknown_keys_are_skipped_not_fatal() {
    nvsReset();
    JsonSettings settings = makeSettings();

    JsonDocument doc;
    doc["bogusKeyFromAnotherVersion"] = String("42");
    doc["mode"] = 4; // known key alongside the unknown one

    bool ok = settings.fromJson(doc);
    CHECK(ok == true); // unknown key must not abort the batch
    CHECK(settings.getInt("mode") == 4);

    // Unknown-key skips are logged.
    bool logged = false;
    for (const auto &line : g_serialLines) {
        if (line.find("bogusKeyFromAnotherVersion") != std::string::npos) {
            logged = true;
        }
    }
    CHECK(logged);
}

static void test_reset_restores_defaults_and_clears_own_namespace() {
    nvsReset();
    {
        JsonSettings settings = makeSettings();
        settings.putInt("mode", 7);
        settings.putString("timeFormat", "{SS}");
        CHECK(settings.getInt("mode") == 7);

        bool ok = settings.reset();
        CHECK(ok == true);
        CHECK(settings.getInt("mode") == 2); // back to default
        CHECK(std::string(settings.getString("timeFormat").c_str()) == "{HH}:{MM}");
    }
    CHECK(g_openHandles == 0);
}

static void test_reset_only_clears_its_own_namespace() {
    nvsReset();
    // Another namespace in the same NVS partition must survive the reset.
    g_nvs[{"otherns", "precious"}] = "keepme";

    JsonSettings settings = makeSettings();
    settings.putInt("mode", 7);
    settings.reset();

    CHECK(g_nvs.count({"otherns", "precious"}) == 1);
    CHECK(std::string(g_nvs[{"otherns", "precious"}]) == "keepme");
}

// ---------------------------------------------------------------------------
// storageKey: >15-char keys are hashed to <=15 chars (NVS key limit)
// ---------------------------------------------------------------------------

static void test_storage_key_hashing() {
    String shortKey = JsonSettings::storageKey("mode");
    CHECK(std::string(shortKey.c_str()) == "mode");

    String longKey = JsonSettings::storageKey("masterGroupModuleCounts");
    CHECK(longKey.length() <= 15);
    CHECK(longKey.length() > 0);
}

int main() {
    test_no_leaked_handles_across_all_accessors();
    test_no_leaked_handle_on_validation_failure();
    test_defaults_when_unset();
    test_put_then_get_roundtrip();
    test_toJson_fromJson_roundtrip();
    test_from_json_rejects_invalid_vector();
    test_unknown_keys_are_skipped_not_fatal();
    test_reset_restores_defaults_and_clears_own_namespace();
    test_reset_only_clears_its_own_namespace();
    test_storage_key_hashing();

    std::printf("%d checks, %d failures\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
