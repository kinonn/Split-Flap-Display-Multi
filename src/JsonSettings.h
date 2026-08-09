#pragma once

#include "JsonSetting.h"

#include <Arduino.h>
#include <ArduinoJson.h>
#include <Preferences.h>
#include <map>

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

    // Creates the Preferences/NVS namespace if it doesn't exist yet. Reads
    // use read-only mode which never creates a namespace, so on a fresh
    // device (first boot after a flash erase) every read would otherwise
    // fail with NOT_FOUND. A one-time read-write open auto-creates it.
    void ensureNamespace();

    const char *name;
    std::map<String, JsonSetting> map;

    String lastValidationError;
    String lastValidationKey;

    JsonSetting find(const char *key);

    Preferences preferences;
    bool namespaceCreated = false;
};
