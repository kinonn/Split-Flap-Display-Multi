#include "JsonSettings.h"

#include <ArduinoJson.h>
#include <Preferences.h>
#include <sstream>

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
    std::istringstream stream(value.c_str());
    std::string token;
    while (std::getline(stream, token, ',')) {
        try {
            intVector.push_back(std::stoi(token));
        } catch (const std::invalid_argument &) {
            throw std::runtime_error("Invalid CSV: Non-integer value found");
        } catch (const std::out_of_range &) {
            throw std::runtime_error("Invalid CSV: Integer value out of range");
        }
    }
    return intVector;
}

std::vector<std::vector<int>> JsonSettings::getIntMatrix(const char *key) {
    String value = getPrefString(key, this->find(key).strDefault);

    std::vector<std::vector<int>> matrix;
    std::istringstream rowStream(value.c_str());
    std::string rowStr;
    while (std::getline(rowStream, rowStr, ';')) {
        std::vector<int> row;
        std::istringstream colStream(rowStr.c_str());
        std::string token;
        while (std::getline(colStream, token, ',')) {
            try {
                row.push_back(std::stoi(token));
            } catch (const std::invalid_argument &) {
                throw std::runtime_error("Invalid matrix: Non-integer value found");
            } catch (const std::out_of_range &) {
                throw std::runtime_error("Invalid matrix: Integer value out of range");
            }
        }
        matrix.push_back(row);
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
    std::ostringstream stream;
    for (size_t i = 0; i < value.size(); ++i) {
        stream << value[i];
        if (i < value.size() - 1) {
            stream << ",";
        }
    }
    putString(key, stream.str().c_str());
}

void JsonSettings::putIntMatrix(const char *key, std::vector<std::vector<int>> value) {
    std::ostringstream stream;
    for (size_t r = 0; r < value.size(); ++r) {
        if (r > 0) stream << ";";
        for (size_t c = 0; c < value[r].size(); ++c) {
            stream << value[r][c];
            if (c < value[r].size() - 1) {
                stream << ",";
            }
        }
    }
    putString(key, stream.str().c_str());
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
        JsonSetting setting = it->second;

        if (! setting.validate(kv.value().as<String>())) {
            lastValidationError = setting.getLastValidationError();
            lastValidationKey = String(key);
            return false;
        }

        switch (setting.type) {
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
