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
extern PubSubClient *g_lastClient;
extern int g_distributeCalls;
extern std::string g_lastDistributed;
extern int g_writeStringCalls;
extern std::string g_lastWritten;
extern float g_lastWriteSpeed;

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

    // RED for issue #6 (audit): the /set callback must only STAGE the
    // message. Executing a potentially tens-of-seconds scroll inline blows
    // the 15 s keepalive and disconnects the client mid-callback. The staged
    // command must drain from loop() instead.
    {
        g_distributeCalls = 0;
        g_writeStringCalls = 0;
        g_lastWritten = "";
        g_lastDistributed = "";
        g_publishCalls = 0;

        JsonSettings settings("mqtttest", testSchema());
        settings.putInt("masterGroupCount", 1);
        WiFiClient wifiClient;
        SplitFlapDisplay display(settings);
        SplitFlapMqtt mqtt(settings, wifiClient);
        mqtt.setDisplay(&display);
        mqtt.setup(); // registers the real staging callback
        g_lastClient->forceConnected(true); // stub: mark the client online

        CHECK(g_writeStringCalls == 0); // nothing dispatched yet

        // Deliver a /set command through the REAL callback.
        char topic[] = "splitflap/splitflap/set";
        byte payload[] = "hello there";
        g_lastClient->deliver(topic, payload, 11);

        // Core regression: the callback itself must not dispatch.
        CHECK(g_writeStringCalls == 0);
        CHECK(g_distributeCalls == 0);

        // loop() is the dispatcher.
        mqtt.loop();
        CHECK(g_writeStringCalls == 1);
        CHECK(g_lastWritten == "hello there");
        CHECK(g_distributeCalls == 0); // single-group: no ESP-NOW

        // A second loop() must not re-run the consumed command.
        mqtt.loop();
        CHECK(g_writeStringCalls == 1);

        // Multi-group branch: message routed via ESP-NOW + explicit
        // publishState with the FULL original message (multi_group_mqtt_
        // state_test covers the same branch with its own mocks).
        g_distributeCalls = 0;
        g_writeStringCalls = 0;
        g_publishCalls = 0;
        settings.putInt("masterGroupCount", 2);
        SplitFlapEspNow espnow(settings, display);
        mqtt.setEspNow(&espnow);
        byte payload2[] = "fleet message";
        g_lastClient->deliver(topic, payload2, 13);
        mqtt.loop();
        CHECK(g_distributeCalls == 1);
        CHECK(g_lastDistributed == "fleet message");
        CHECK(g_writeStringCalls == 0); // multi-group: no local writeString
        CHECK(g_publishCalls >= 1);     // state topic still published
    }

    // RED for issue #21 (audit): connect() must register a retained
    // "offline" Last Will on the availability topic. Without it the broker
    // keeps the retained "online" after a crash/power loss and Home
    // Assistant shows the entities available forever.
    {
        // Anonymous branch (mqtt_user empty).
        JsonSettings settings("mqttwilltest", testSchema());
        WiFiClient wifiClient;
        SplitFlapMqtt mqtt(settings, wifiClient);
        mqtt.setup(); // real production code path; connect itself records will

        CHECK(g_lastClient->willTopic() == "splitflap/splitflap/availability");
        CHECK(g_lastClient->willMessage() == "offline");
        CHECK(g_lastClient->willRetain() == true);
    }
    {
        // Authenticated branch.
        JsonSettings settings("mqttwilltest", testSchema());
        settings.putString("mqtt_user", "user");
        settings.putString("mqtt_pass", "pass");
        WiFiClient wifiClient;
        SplitFlapMqtt mqtt(settings, wifiClient);
        mqtt.setup();

        CHECK(g_lastClient->willTopic() == "splitflap/splitflap/availability");
        CHECK(g_lastClient->willMessage() == "offline");
        CHECK(g_lastClient->willRetain() == true);
    }

    std::printf("%d checks, %d failures\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
