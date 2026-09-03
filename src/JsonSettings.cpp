#include "JsonSettings.h"

#include <ArduinoJson.h>
#include <Preferences.h>
#include <errno.h>
#include <limits.h>
#include <stdexcept>
#include <stdlib.h>

// ---------------------------------------------------------------------------
// Integer-list parsing without <sstream>.
//
// <sstream> drags the C++ iostreams and std::locale machinery into the image,
// and with it newlib's wide-character and floating-point printf/scanf
// families — hundreds of KB of flash that nothing else in this firmware uses.
// These helpers do the same job with strtol() and Arduino String.
//
// Grammar (a row is one vector, matrices are ';'-separated rows):
//   list   : [ws] int ([ws] ',' [ws] int)* [ws] [',']
//
// Deliberate leniency: junk between numbers is skipped rather than rejected
// ("1,x,3" yields {1, 3}), so a malformed value can never make getIntVector
// throw while settings load — callers must not rely on exceptions for length
// safety (SplitFlapDisplay bounds-checks every index it reads).
// ---------------------------------------------------------------------------

namespace {

// Parse comma-separated integers from [begin, end). Whitespace is skipped;
// junk is skipped so a malformed value can never hang the parser. Empty
// entries (",,", trailing ',') produce no value.
void parseIntList(const char *begin, const char *end, std::vector<int> &out) {
    const char *cursor = begin;

    while (cursor < end) {
        while (cursor < end && (*cursor == ' ' || *cursor == '\t' || *cursor == '\n' || *cursor == '\r')) {
            cursor++;
        }
        if (cursor >= end) {
            break;
        }

        char c = *cursor;
        if (c == ',') {
            cursor++; // empty entry (",,") or repeated separator; skip it
            continue;
        }

        char *stop = nullptr;
        errno = 0;
        long parsed = strtol(cursor, &stop, 10);
        if (stop == cursor) {
            cursor++; // junk token: skip one character so we can never spin
            continue;
        }
        if (errno == ERANGE || parsed < INT_MIN || parsed > INT_MAX) {
            // Out-of-range value: keep the number's position as a 0 rather
            // than dropping it, so entry order survives. This preserves the
            // old code's guarantee that absurd input can't produce a bogus
            // large offset — without the throw the old code relied on.
            out.push_back(0);
            cursor = stop;
        } else {
            out.push_back((int) parsed);
            cursor = stop;
        }

        // Consume separator/whitespace run after the number so trailing
        // commas or padding spaces don't confuse the top of the loop.
        while (cursor < end &&
               (*cursor == ',' || *cursor == ' ' || *cursor == '\t' || *cursor == '\n' || *cursor == '\r')) {
            cursor++;
        }
    }
}

} // namespace

// Every access below opens its own call-local NVS handle and closes it before
// returning. The class deliberately holds NO shared Preferences instance:
// JsonSettings is used from at least three tasks on this firmware (the Arduino
// loop task reads settings on every pass, the AsyncTCP task serving the web
// handlers reads and writes them, and the ESP-NOW receive task writes remote
// offsets and mode). With a shared handle, one task's end() closed the handle
// another task was part way through using: reads silently fell back to the
// compiled default and writes were dropped while the request still answered
// success. Call-local instances make every transaction self-contained, so no
// cross-task lock is needed (and none can be forgotten at a new call site).

String JsonSettings::storageKey(const char *key) {
    String storage = String(key);
    if (storage.length() <= 15) {
        return storage;
    }

    unsigned int hash = 5381;
    for (size_t i = 0; i < storage.length(); ++i) {
        hash = ((hash << 5) + hash) + storage[i];
    }

    String suffix = String(hash & 0xFFFF, HEX);
    suffix.toUpperCase();
    while (suffix.length() < 4) {
        suffix = String("0") + suffix;
    }

    return storage.substring(0, 11) + suffix;
}

String JsonSettings::getPrefString(const char *key, const String &def) {
    String storeKey = storageKey(key);
    Preferences preferences;
    preferences.begin(name, true);
    String value = preferences.isKey(storeKey.c_str()) ? preferences.getString(storeKey.c_str(), def) : def;
    preferences.end();
    return value;
}

int JsonSettings::getPrefInt(const char *key, int def) {
    String storeKey = storageKey(key);
    Preferences preferences;
    preferences.begin(name, true);
    int value = preferences.getInt(storeKey.c_str(), def);
    preferences.end();
    return value;
}

float JsonSettings::getPrefFloat(const char *key, float def) {
    String storeKey = storageKey(key);
    Preferences preferences;
    preferences.begin(name, true);
    float value = preferences.getFloat(storeKey.c_str(), def);
    preferences.end();
    return value;
}

void JsonSettings::putPrefString(const char *key, const String &value) {
    String storeKey = storageKey(key);
    Preferences preferences;
    preferences.begin(name, false);
    preferences.putString(storeKey.c_str(), value);
    preferences.end();
}

void JsonSettings::putPrefInt(const char *key, int value) {
    String storeKey = storageKey(key);
    Preferences preferences;
    preferences.begin(name, false);
    preferences.putInt(storeKey.c_str(), value);
    preferences.end();
}

void JsonSettings::putPrefFloat(const char *key, float value) {
    String storeKey = storageKey(key);
    Preferences preferences;
    preferences.begin(name, false);
    preferences.putFloat(storeKey.c_str(), value);
    preferences.end();
}

String JsonSettings::getString(const char *key) {
    return getPrefString(key, this->find(key).strDefault);
}

int JsonSettings::getInt(const char *key) {
    return getPrefInt(key, this->find(key).intDefault);
}

float JsonSettings::getFloat(const char *key) {
    return getPrefFloat(key, this->find(key).floatDefault);
}

std::vector<int> JsonSettings::getIntVector(const char *key) {
    String value = getPrefString(key, this->find(key).strDefault);

    std::vector<int> intVector;
    parseIntList(value.c_str(), value.c_str() + value.length(), intVector);
    return intVector;
}

std::vector<std::vector<int>> JsonSettings::getIntMatrix(const char *key) {
    String value = getPrefString(key, this->find(key).strDefault);

    std::vector<std::vector<int>> matrix;
    const char *cursor = value.c_str();
    const char *end = cursor + value.length();
    while (cursor < end) {
        const char *rowEnd = cursor;
        while (rowEnd < end && *rowEnd != ';') {
            rowEnd++;
        }

        std::vector<int> row;
        parseIntList(cursor, rowEnd, row);
        matrix.push_back(row);

        cursor = (rowEnd < end) ? rowEnd + 1 : rowEnd;
    }
    return matrix;
}

void JsonSettings::putString(const char *key, String value) {
    putPrefString(key, value);
}

void JsonSettings::putInt(const char *key, int value) {
    putPrefInt(key, value);
}

void JsonSettings::putFloat(const char *key, float value) {
    putPrefFloat(key, value);
}

void JsonSettings::putIntVector(const char *key, std::vector<int> value) {
    String joined;
    for (size_t i = 0; i < value.size(); ++i) {
        if (i > 0) {
            joined += ',';
        }
        joined += String(value[i]);
    }
    putString(key, joined);
}

void JsonSettings::putIntMatrix(const char *key, std::vector<std::vector<int>> value) {
    String joined;
    for (size_t r = 0; r < value.size(); ++r) {
        if (r > 0) {
            joined += ';';
        }
        for (size_t c = 0; c < value[r].size(); ++c) {
            if (c > 0) {
                joined += ',';
            }
            joined += String(value[r][c]);
        }
    }
    putString(key, joined);
}

JsonDocument JsonSettings::toJson() {
    JsonDocument settings;

    for (const auto &pair : map) {
        const String &key = pair.first;
        const JsonSetting &setting = pair.second;

        switch (setting.type) {
            case JsonSettingType::JST_STR:
            case JsonSettingType::JST_INT_VECTOR:
            case JsonSettingType::JST_INT_MATRIX:
                settings[key] = getPrefString(key.c_str(), setting.strDefault);
                break;
            case JsonSettingType::JST_INT:
                settings[key] = getPrefInt(key.c_str(), setting.intDefault);
                break;
            case JsonSettingType::JST_FLOAT:
                settings[key] = getPrefFloat(key.c_str(), setting.floatDefault);
                break;
        }
    }

    return settings;
}

bool JsonSettings::fromJson(JsonDocument settings) {
    // Transactional batch: validate EVERY present, known key before writing
    // ANY of them. The previous single validate-then-write loop left keys
    // persisted when a later key failed, while callers reported the whole
    // save as failed — worst case, re-homing modules on offsets the web UI
    // said were not saved.
    for (JsonPair kv : settings.as<JsonObject>()) {
        const char *key = kv.key().c_str();
        auto it = this->map.find(key);
        if (it == this->map.end()) {
            // Unknown keys are ignored rather than rejected: a browser still
            // holding a settings page from a different firmware version will
            // post keys this build has never heard of. find() throws for them.
            Serial.print("Ignoring unknown setting: ");
            Serial.println(key);
            continue;
        }
        if (! it->second.validate(kv.value().as<String>())) {
            lastValidationError = it->second.getLastValidationError();
            lastValidationKey = String(key);
            return false;
        }
    }

    // Everything validated — now write.
    for (JsonPair kv : settings.as<JsonObject>()) {
        const char *key = kv.key().c_str();
        auto it = this->map.find(key);
        if (it == this->map.end()) {
            continue;
        }

        switch (it->second.type) {
            case JsonSettingType::JST_INT_VECTOR:
            case JsonSettingType::JST_INT_MATRIX:
            case JsonSettingType::JST_STR:
                putPrefString(key, kv.value().as<String>());
                break;
            case JsonSettingType::JST_INT:
                putPrefInt(key, kv.value().as<int>());
                break;
            case JsonSettingType::JST_FLOAT:
                putPrefFloat(key, kv.value().as<float>());
                break;
        }
    }

    return true;
}

bool JsonSettings::reset() {
    // Use this namespace's own name; the previous hardcoded "config" only
    // worked because every caller happens to construct JsonSettings("config").
    {
        Preferences preferences;
        preferences.begin(name, false);
        preferences.clear();
        preferences.end();
    }

    return fromJson(toJson());
}

JsonSetting JsonSettings::find(const char *key) {
    auto it = this->map.find(key);
    if (it == this->map.end()) {
        throw std::runtime_error("Key not found in settings map");
    }
    return it->second;
}
