#pragma once

// MQTT status payload builder for the split-flap display.
//
// Published to the retained topic `splitflap/{mdns}/status` as JSON:
//
//     {"message":"HELLO","num_modules":8}
//
// This topic is ADDITIVE — the legacy plain-string `state` topic is
// unchanged, so existing consumers keep working. This file is kept
// free of Arduino dependencies (uses std::string, not Arduino String)
// so the payload builder can be unit-tested on the host with the real
// ArduinoJson library (see test/status_payload_test.cpp).

#include <ArduinoJson.h>
#include <string>

inline std::string buildStatusPayload(const std::string &message, int numModules) {
    JsonDocument doc;
    doc["message"] = message;
    doc["num_modules"] = numModules;
    std::string payload;
    serializeJson(doc, payload);
    return payload;
}
