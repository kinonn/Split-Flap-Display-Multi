#include "SplitFlapMqtt.h"

SplitFlapMqtt::SplitFlapMqtt(JsonSettings &settings, WiFiClient &wifiClient)
    : settings(settings), wifiClient(wifiClient), mqttClient(wifiClient), display(nullptr) {}

void SplitFlapMqtt::setup() {
    mqttServer = settings.getString("mqtt_server");
    mqttPort = settings.getInt("mqtt_port");
    mqttUser = settings.getString("mqtt_user");
    mqttPass = settings.getString("mqtt_pass");

    String mdns = settings.getString("mdns");
    String name = settings.getString("name");

    topic_command = "splitflap/" + mdns + "/set";
    topic_state = "splitflap/" + mdns + "/state";
    topic_avail = "splitflap/" + mdns + "/availability";
    topic_config_text = "homeassistant/text/splitflap_text_" + mdns + "/config";
    topic_config_sensor = "homeassistant/sensor/splitflap_sensor_" + mdns + "/config";

    mqttClient.setServer(mqttServer.c_str(), mqttPort);
    mqttClient.setCallback([this](char *topic, byte *payload, unsigned int length) {
        String message;
        for (unsigned int i = 0; i < length; i++) {
            message += (char) payload[i];
        }
        Serial.printf("[MQTT] Message received: %s\n", message.c_str());
        if (display) {
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
                if (mqttClient.connected()) {
                    publishState(message);
                }
            } else {
                Serial.println("[MQTT] Displaying message locally on Group 1");
                display->writeString(message, maxVel, false);
            }
        }
    });

    connectToMqtt();
}

void SplitFlapMqtt::connectToMqtt() {
    if (! mqttClient.connected()) {
        Serial.println("[MQTT] Attempting to connect...");
        String mdns = settings.getString("mdns");
        String name = settings.getString("name");

        if (mqttUser.length() > 0) {
            mqttClient.connect(mdns.c_str(), mqttUser.c_str(), mqttPass.c_str());
        } else {
            mqttClient.connect(mdns.c_str());
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
}

bool SplitFlapMqtt::isConnected() {
    return mqttClient.connected();
}
