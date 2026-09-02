// Threaded hammer test for JsonSettings — simulates the ESP32 task structure
// that broke the old shared-Preferences design:
//
//   - writer threads (the AsyncTCP web task and the ESP-NOW receive task)
//     put values, exactly as POST /settings and an incoming offsets push do;
//   - a reader thread (the Arduino loop task) getInt()s in a tight loop,
//     exactly as loop() reading "mode" every pass does;
//   - every write is followed by a read-back from the reader side.
//
// The invariant: a putInt(key, v) that returned must be visible to every
// subsequent getInt(key) — reads may NEVER fall back to the compiled default
// while a key has been written. On the old code (one shared Preferences
// member) this invariant fails within a few hundred iterations; with
// call-local handles it holds indefinitely.
//
// Build & run:
//   g++ -std=c++17 -Wall -Wextra -pthread -I src -I test/stubs_jsonsettings test/jsonsettings_threads_test.cpp src/JsonSettings.cpp src/JsonSetting.cpp -o /tmp/jsonsettings_threads_test && /tmp/jsonsettings_threads_test
//
// ThreadSanitizer (run from the repo root):
//   g++ -std=c++17 -fsanitize=thread -g -pthread -I src -I test/stubs_jsonsettings test/jsonsettings_threads_test.cpp src/JsonSettings.cpp src/JsonSetting.cpp -o /tmp/jsonsettings_tsan && /tmp/jsonsettings_tsan

#include <atomic>
#include <cstdio>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "Arduino.h"
#include "ArduinoJson.h"
#include "JsonSettings.h"
#include "Preferences.h"

std::vector<std::string> g_serialLines;
std::map<std::pair<std::string, std::string>, std::string> g_nvs;
std::mutex g_nvsMutex;
std::atomic<int> g_openHandles{0};

SerialStub Serial;

static std::atomic<int> failures{0};

static std::map<String, JsonSetting> testSchema() {
    return {
        {"mode", JsonSetting(0)},
        {"displayOffset", JsonSetting(0)},
    };
}

int main() {
    // ONE settings instance shared by all "tasks" — exactly like the firmware,
    // where the loop task, the AsyncTCP task and the ESP-NOW task all call
    // through the same global `settings` object and (before this fix) the
    // same Preferences/NVS handle.
    JsonSettings settings("testns", testSchema());

    // Thread 1 ("web task"): repeatedly writes mode = 7 (POST /settings).
    std::thread writerWeb([&settings] {
        for (int i = 0; i < 2000; i++) {
            settings.putInt("mode", 7);
        }
    });

    // Thread 2 ("ESP-NOW task"): repeatedly writes mode = 9 (offsets push).
    std::thread writerEspNow([&settings] {
        for (int i = 0; i < 2000; i++) {
            settings.putInt("mode", 9);
        }
    });

    // Thread 3 ("loop task"): hammers getInt; asserts every read observes one
    // of the written values. The very first write from either writer lands
    // within microseconds; from then on the compiled default 0 is never
    // legitimate again — a 0 read means a write (or a whole handle session)
    // was lost.
    std::thread reader([&settings] {
        // Wait until the first write has certainly landed.
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        for (int i = 0; i < 4000; i++) {
            int v = settings.getInt("mode");
            if (!(v == 7 || v == 9)) {
                failures++;
                std::printf(
                    "FAIL: read %d back — not a written value (lost write or default fallback)\n", v
                );
            }
        }
    });

    writerWeb.join();
    writerEspNow.join();
    reader.join();

    // Final invariant: the last write by whichever writer finished last must
    // be visible now that both writers are done.
    int finalMode = settings.getInt("mode");
    if (!(finalMode == 7 || finalMode == 9)) {
        failures++;
        std::printf("FAIL: final mode %d — last write was lost\n", finalMode);
    }

    if (g_openHandles.load() != 0) {
        failures++;
        std::printf("FAIL: %d NVS handles left open\n", g_openHandles.load());
    }

    std::printf("threads hammer: %s (%d failures)\n", failures == 0 ? "PASS" : "FAIL", (int) failures);
    return failures == 0 ? 0 : 1;
}
