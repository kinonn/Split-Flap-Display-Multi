#include "SplitFlapMqtt.h"

#include "StatusPayload.h"

SplitFlapMqtt::SplitFlapMqtt(JsonSettings &settings, WiFiClient &wifiClient)
    : settings(settings), wifiClient(wifiClient), mqttClient(wifiClient), display(nullptr) {}

void SplitFlapMqtt::setup() {
    mqttServer = settings.getString("mqtt_server");
    mqttPort = settings.getInt("mqtt_port");
    mqttUser = settings.getString("mqtt_user");
    mqttPass = settings.getString("mqtt_pass");

    // PubSubClient defaults to a 256-byte receive buffer (PubSubClient.h
    // MQTT_MAX_PACKET_SIZE). The command topic alone is ~25 bytes and a
    // message spanning a multi-group fleet does not fit in what remains —
    // anything longer was silently dropped (or disconnected the client).
    // 512 covers the largest scannable payload (~256 chars) plus topic and
    // protocol framing, with headroom.
    mqttClient.setBufferSize(512);

    String mdns = settings.getString("mdns");
    String name = settings.getString("name");

    topic_command = "splitflap/" + mdns + "/set";
    topic_state = "splitflap/" + mdns + "/state";
    topic_avail = "splitflap/" + mdns + "/availability";
    topic_config_text = "homeassistant/text/splitflap_text_" + mdns + "/config";
    topic_config_sensor = "homeassistant/sensor/splitflap_sensor_" + mdns + "/config";
    topic_status = "splitflap/" + mdns + "/status";

    mqttClient.setServer(mqttServer.c_str(), mqttPort);
    // The callback only stages the command in pendingMessage/pendingText;
    // the scroll can run for tens of seconds and must not execute here or
    // the 15 s keepalive would lapse mid-callback and disconnect us (the
    // publishState after the scroll would then silently fail). loop()
    // drains it via processPendingMessage().
    mqttClient.setCallback([this](char * /*topic*/, byte *payload, unsigned int length) {
        String message;
        for (unsigned int i = 0; i < length; i++) {
            message += (char) payload[i];
        }
        Serial.printf("[MQTT] Message received: %s\n", message.c_str());
        pendingText = message;
        pendingMessage = true;
    });

    connectToMqtt();
}

void SplitFlapMqtt::processPendingMessage() {
    if (! pendingMessage) {
        return;
    }

    // Consume the flag first so a message staged while we run can only
    // trigger one additional pass, never re-enter this one.
    pendingMessage = false;
    if (! display) {
        return;
    }

    String message = pendingText;
    pendingText = ""; // release the buffer before the long scroll

    float maxVel = settings.getFloat("maxVel");
    if (settings.getInt("masterGroupCount") > 1 && espNow) {
        int groupCount = settings.getInt("masterGroupCount");
        Serial.printf("[MQTT] masterGroupCount=%d, routing message through ESP-NOW distribution\n", groupCount);
        espNow->distributeMessage(
            message,
            false,
            settings.getInt("scrollDelayMs"),
            settings.getInt("scrollRepeatCount")
        );
        // Single-group mode publishes the full message via
        // display->writeString() (publishState defaults to true). In
        // multi-group mode, writeString is not called, so publish the
        // state here with the original message (not a per-group slice).
        // Called unconditionally: publish() fails silently when the broker
        // dropped us during a long scroll, but lastPublishedState is still
        // updated, so the reconnect path republishes the correct retained
        // state instead of leaving the old message in place.
        publishState(message);
    } else {
        Serial.println("[MQTT] Displaying message locally on Group 1");
        display->writeString(message, maxVel, false);
    }
}

void SplitFlapMqtt::connectToMqtt() {
    if (! mqttClient.connected()) {
        Serial.println("[MQTT] Attempting to connect...");
        String mdns = settings.getString("mdns");
        String name = settings.getString("name");

        // Last Will on the availability topic (retained "offline"): without it
        // the broker keeps our retained "online" after a crash/power loss and
        // Home Assistant shows the entities available forever.
        if (mqttUser.length() > 0) {
            mqttClient.connect(
                mdns.c_str(), mqttUser.c_str(), mqttPass.c_str(), topic_avail.c_str(), 0, true, "offline"
            );
        } else {
            mqttClient.connect(mdns.c_str(), topic_avail.c_str(), 0, true, "offline");
        }

        if (mqttClient.connected()) {
            Serial.println("[MQTT] Connected to broker");

            // clang-format off
            String payload_text = "{"
                "\"name\":\"Display\","
                "\"unique_id\":\"text_" + mdns + "\","
                "\"command_topic\":\"" + topic_command + "\","
                "\"availability_topic\":\"" + topic_avail + "\","
                "\"device\":{"
                    "\"identifiers\":[\"splitflap_" + mdns + "\"],"
                    "\"name\":\"" + name + "\","
                    "\"manufacturer\":\"SplitFlap\","
                    "\"model\":\"SplitFlap Display\","
                    "\"sw_version\":\"1.0.0\""
                "}"
            "}";

            String payload_sensor = "{"
                "\"name\":\"Currently Displayed\","
                "\"unique_id\":\"sensor_" + mdns + "\","
                "\"state_topic\":\"" + topic_state + "\","
                "\"availability_topic\":\"" + topic_avail + "\","
                "\"entity_category\":\"diagnostic\","
                "\"device\":{"
                    "\"identifiers\":[\"splitflap_" + mdns + "\"],"
                    "\"name\":\"" + name + "\","
                    "\"manufacturer\":\"SplitFlap\","
                    "\"model\":\"SplitFlap Display\","
                    "\"sw_version\":\"1.0.0\""
                "}"
            "}";
            // clang-format on

            mqttClient.subscribe(topic_command.c_str());
            mqttClient.publish(topic_avail.c_str(), "online", true);
            mqttClient.publish(topic_state.c_str(), lastPublishedState.c_str(), true);

            mqttClient.publish(topic_config_text.c_str(), payload_text.c_str(), true);
            mqttClient.publish(topic_config_sensor.c_str(), payload_sensor.c_str(), true);

            publishStatus();
        } else {
            Serial.println("[MQTT] Failed to connect");
        }
    }
}

void SplitFlapMqtt::setDisplay(SplitFlapDisplay *d) {
    display = d;
}

void SplitFlapMqtt::setEspNow(SplitFlapEspNow *e) {
    espNow = e;
}

void SplitFlapMqtt::publishState(const String &message) {
    Serial.println("[MQTT] Publishing state: " + message);
    lastPublishedState = message;        // remember for reconnects
    mqttClient.publish(topic_state.c_str(), message.c_str(), true);
    publishStatus();
}

void SplitFlapMqtt::publishStatus() {
    if (!display || !mqttClient.connected()) {
        return;
    }
    // Additive JSON status topic: current message + display size (module
    // count). The legacy plain-string `state` topic is untouched, so
    // existing consumers are unaffected.
    //
    // Multi-group: report the TOTAL module count across all groups (sum of
    // masterGroupModuleCounts + local display), which is the physical display
    // width a message spans. getTotalModuleCount() degenerates to the local
    // count in single-group mode, so one call is correct in both modes.
    int moduleCount = (espNow != nullptr) ? espNow->getTotalModuleCount()
                                          : display->getNumModules();
    std::string payload = buildStatusPayload(
        lastPublishedState.c_str(), moduleCount
    );
    mqttClient.publish(topic_status.c_str(), payload.c_str(), true);
}

void SplitFlapMqtt::loop() {
    if (! mqttClient.connected()) {
        unsigned long now = millis();
        if (now - lastAttempt > 5000) {
            lastAttempt = now;
            connectToMqtt();
        }
    }
    mqttClient.loop();
    // Drain staged /set commands only after the network loop has had its
    // turn — a scroll may run for tens of seconds, so it must never delay
    // the keepalive servicing that mqttClient.loop() performs.
    processPendingMessage();
}

bool SplitFlapMqtt::isConnected() {
    return mqttClient.connected();
}
