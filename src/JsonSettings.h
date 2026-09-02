#pragma once

#include "JsonSetting.h"

#include <Arduino.h>
#include <ArduinoJson.h>
#include <map>

// Forward declaration: the Arduino-esp32 Preferences/NVS wrapper. Call-local
// instances live in JsonSettings.cpp; this class holds none as a member.
class Preferences;

// Bump this whenever the settings schema changes (keys added/removed/renamed,
// or the meaning of an existing value changes). The web UI uses it to warn
// about forward/backward compatibility when importing an exported config.
#define SETTINGS_SCHEMA_VERSION 1

class JsonSettings {
  public:
    JsonSettings(const char *name, std::map<String, JsonSetting> map) : name(name), map(map) {}

    String getString(const char *key);
    int getInt(const char *key);
    float getFloat(const char *key);
    std::vector<int> getIntVector(const char *key);
    std::vector<std::vector<int>> getIntMatrix(const char *key);

    void putString(const char *key, String value);
    void putInt(const char *key, int value);
    void putFloat(const char *key, float value);
    void putIntVector(const char *key, std::vector<int> value);
    void putIntMatrix(const char *key, std::vector<std::vector<int>> value);

    JsonDocument toJson();
    bool fromJson(JsonDocument settings);
    bool reset();

    String getLastValidationError() { return lastValidationError; }
    static String storageKey(const char *key);
    String getLastValidationKey() { return lastValidationKey; }

  private:
    String getPrefString(const char *key, const String &def);
    int getPrefInt(const char *key, int def);
    float getPrefFloat(const char *key, float def);
    void putPrefString(const char *key, const String &value);
    void putPrefInt(const char *key, int value);
    void putPrefFloat(const char *key, float value);

    const char *name;
    std::map<String, JsonSetting> map;

    String lastValidationError;
    String lastValidationKey;

    JsonSetting find(const char *key);

    // No `Preferences preferences` member: NVS handles are opened and closed
    // per call (see JsonSettings.cpp). Sharing one handle across tasks is
    // what let one task's end() close the handle another task was using.
};
