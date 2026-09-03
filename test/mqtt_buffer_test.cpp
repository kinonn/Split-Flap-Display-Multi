// Host tests for SplitFlapMqtt — compile the REAL src/SplitFlapMqtt.cpp
// against stubs (PubSubClient/WiFiClient/esp_now doubles + the JsonSettings
// stub bundle). Focus: the receive-buffer sizing regression (audit issue #2 —
// PubSubClient's 256-byte default silently dropped long MQTT messages).
//
// Build & run:
//   g++ -std=c++17 -Wall -Wextra -pthread -I src -I test/stubs_jsonsettings
//     test/mqtt_buffer_test.cpp src/SplitFlapMqtt.cpp src/JsonSettings.cpp
//     src/JsonSetting.cpp test/stubs_jsonsettings/splitflap_espnow_stub.cpp
//     -o /tmp/mqtt_test && /tmp/mqtt_test

#include <cstdio>

#include "Arduino.h"
#include "ArduinoJson.h"
#include "JsonSettings.h"
#include "PubSubClient.h"
#include "SplitFlapMqtt.h"

// Stub globals (same pattern as jsonsettings_test.cpp)
std::vector<std::string> g_serialLines;
SerialStub Serial;
std::map<std::pair<std::string, std::string>, std::string> g_nvs;
std::mutex g_nvsMutex;
std::atomic<int> g_openHandles{0};

extern size_t g_clientBufferSize;
extern int g_publishCalls;

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

static std::map<String, JsonSetting> testSchema() {
    return {
        {"name", JsonSetting("Test Display")},
        {"mdns", JsonSetting("splitflap")},
        {"mqtt_server", JsonSetting("broker.local")},
        {"mqtt_port", JsonSetting(1883)},
        {"mqtt_user", JsonSetting("")},
        {"mqtt_pass", JsonSetting("")},
        {"scrollDelayMs", JsonSetting(1500)},
        {"scrollRepeatCount", JsonSetting(2)},
        {"masterGroupCount", JsonSetting(1)},
        {"maxVel", JsonSetting(15.0f)},
        {"moduleCount", JsonSetting(8)},
    };
}

int main() {
    // RED for issue #2: setup() must raise the receive buffer above the
    // library default of 256 bytes, because the command topic plus a message
    // spanning the full 48-module fleet does not fit in 256.
    {
        g_clientBufferSize = 256; // simulate the library default

        JsonSettings settings("mqtttest", testSchema());
        WiFiClient wifiClient;
        SplitFlapMqtt mqtt(settings, wifiClient);
        mqtt.setup(); // real production code path

        CHECK(g_clientBufferSize >= 512);
        CHECK(g_clientBufferSize >= (size_t) (MAX_DISPLAY_GROUPS * 8 + 64));
    }

    std::printf("%d checks, %d failures\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
