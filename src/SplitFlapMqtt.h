#pragma once

#include "JsonSettings.h"
#include "SplitFlapDisplay.h"
#include "SplitFlapEspNow.h"

#include <PubSubClient.h>
#include <WiFiClient.h>

class SplitFlapMqtt {
  public:
    SplitFlapMqtt(JsonSettings &settings, WiFiClient &client); // updated constructor

    void setup();
    void loop();                                               // needed for PubSubClient3
    void publishState(const String &message);
    void setDisplay(SplitFlapDisplay *display);
    void setEspNow(SplitFlapEspNow *espNow);
    bool isConnected();

  private:
    PubSubClient mqttClient; // PubSubClient instead of AsyncMqttClient
    WiFiClient &wifiClient;  // store reference to WiFiClient

    JsonSettings &settings;
    SplitFlapDisplay *display;
    SplitFlapEspNow *espNow = nullptr;

    void connectToMqtt();
    void publishStatus();         // publish JSON status topic (retained)
    void processPendingMessage(); // run a staged /set command in loop()

    // Deferred /set commands: the PubSubClient callback only stages the
    // message here; the potentially long scroll runs from loop() so the
    // keepalive is serviced and a disconnect cannot kill the client state
    // mid-callback.
    bool pendingMessage = false;
    String pendingText;

    // MQTT config
    String mqttServer;
    int mqttPort = 1883;
    String mqttUser;
    String mqttPass;
    String topic_command;
    String topic_state;
    String topic_avail;
    String topic_config_text;
    String topic_config_sensor;
    String topic_status;                  // splitflap/{mdns}/status — JSON state info (additive)

    unsigned long lastAttempt = 0;
    int retryCount = 0;
    String lastPublishedState;         // mirror of the most recent state publish
};
