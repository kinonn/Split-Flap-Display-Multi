// Minimal Arduino-esp32 Preferences stub for test/jsonsettings_test.cpp.
// Backed by an in-memory map with SIMULATED task-preemption hooks: every
// begin/end call can be instrumented to yield, which is how the shared-handle
// race is reproduced deterministically on the host (see the test file).
#pragma once

#include <atomic>
#include <cstdint>
#include <map>
#include <mutex>
#include <string>

#include "Arduino.h"

// The simulated NVS store, shared by all Preferences instances (like the real
// NVS partition). NVS itself is internally synchronized, so the stub is too —
// that keeps any TSan report attributable to the firmware's own handle
// lifecycle, not to the stub's map.
extern std::map<std::pair<std::string, std::string>, std::string> g_nvs;
extern std::mutex g_nvsMutex;
// Instrumentation: number of currently-open handles across all instances.
// Atomic because it is updated from multiple test threads.
extern std::atomic<int> g_openHandles;

class Preferences {
  public:
    // Models the real Arduino-esp32 Preferences: one NVS handle PER INSTANCE,
    // not per operation. begin() opens it, end() closes it; operations on a
    // closed handle fail exactly like real NVS (get returns the default,
    // put is a no-op returning 0) — which is how the shared-handle bug lets
    // reads fall back to defaults and writes vanish. On the ESP32 the
    // instance was a class member shared by all tasks; on the host, whether
    // instances are shared or per-call is up to the code under test.
    bool begin(const char *name, bool readOnly, const char *partition = nullptr) {
        (void) partition;
        ns_ = name;
        readOnly_ = readOnly;
        open_ = true;
        ++g_openHandles;
        return true;
    }
    void end() {
        if (open_) {
            open_ = false;
            --g_openHandles;
        }
    }
    bool isKey(const char *key) {
        if (! open_) return false;
        std::lock_guard<std::mutex> lock(g_nvsMutex);
        return g_nvs.count({ns_, key}) != 0;
    }
    // Firmware calls getString(key, String) and assigns to String.
    String getString(const char *key, const String &def) {
        if (! open_) return def;
        std::lock_guard<std::mutex> lock(g_nvsMutex);
        auto it = g_nvs.find({ns_, key});
        return it != g_nvs.end() ? String(it->second) : def;
    }
    int getInt(const char *key, int def) {
        if (! open_) return def;
        std::lock_guard<std::mutex> lock(g_nvsMutex);
        auto it = g_nvs.find({ns_, key});
        return it != g_nvs.end() ? std::stoi(it->second) : def;
    }
    float getFloat(const char *key, float def) {
        if (! open_) return def;
        std::lock_guard<std::mutex> lock(g_nvsMutex);
        auto it = g_nvs.find({ns_, key});
        return it != g_nvs.end() ? std::stof(it->second) : def;
    }
    size_t putString(const char *key, const String &value) {
        if (! open_ || readOnly_) return 0;
        std::lock_guard<std::mutex> lock(g_nvsMutex);
        g_nvs[{ns_, key}] = std::string(value.c_str());
        return value.length();
    }
    size_t putInt(const char *key, int value) {
        if (! open_ || readOnly_) return 0;
        std::lock_guard<std::mutex> lock(g_nvsMutex);
        g_nvs[{ns_, key}] = std::to_string(value);
        return 4;
    }
    size_t putFloat(const char *key, float value) {
        if (! open_ || readOnly_) return 0;
        std::lock_guard<std::mutex> lock(g_nvsMutex);
        g_nvs[{ns_, key}] = std::to_string(value);
        return 4;
    }
    void clear() {
        if (! open_ || readOnly_) return;
        std::lock_guard<std::mutex> lock(g_nvsMutex);
        for (auto it = g_nvs.begin(); it != g_nvs.end();) {
            if (it->first.first == ns_) {
                it = g_nvs.erase(it);
            } else {
                ++it;
            }
        }
    }

  private:
    std::string ns_;
    bool readOnly_ = true;
    bool open_ = false;
};
