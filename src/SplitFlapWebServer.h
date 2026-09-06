#pragma once

#include "JsonSettings.h"
#include "PendingActions.h"
#include "SplitFlapDisplay.h"

#include <Arduino.h>
#include <ArduinoJson.h>
#include <ArduinoOTA.h>
#include <ESPAsyncWebServer.h>
#include <ESPmDNS.h>
#include <LittleFS.h>
#include <WiFi.h>
#include <atomic>
#include <mutex>
#include <time.h>

class SplitFlapEspNow;

class SplitFlapWebServer {
  public:
    SplitFlapWebServer(JsonSettings &settings, SplitFlapDisplay &display);
    void init();
    void setTimezone();
    void checkRebootRequired();

    // Wifi Connectivity
    bool loadWiFiCredentials();
    bool connectToWifi();
    bool getAttemptReconnect() const { return attemptReconnect; }
    void setAttemptReconnect(bool input) { attemptReconnect = input; }
    void startWebServer();
    void endMDNS();
    void startMDNS();
    void enableOta();
    void handleOta();
    void startAccessPoint();
    void checkWiFi();
    unsigned long getLastCheckWifiTime() { return lastCheckWifiTime; }
    void setLastCheckWifiTime(unsigned long input) { lastCheckWifiTime = input; }
    int getWifiCheckInterval() { return wifiCheckInterval; }

    // Mode
    int getMode();

    // Mode 0 - Single String
    String getInputString() const { return inputString; }
    String getWrittenString() const { return writtenString; }
    void setWrittenString(String input) { writtenString = input; }

    // Mode 1, Multi Input
    String getMultiInputString() const { return multiInputString; }
    int getMultiWordDelay() const { return multiWordDelay; }
    unsigned long getLastSwitchMultiTime() { return lastSwitchMultiTime; }
    void setLastSwitchMultiTime(unsigned long input) { lastSwitchMultiTime = input; }
    int getMultiWordCurrentIndex() { return multiWordCurrentIndex; }
    void setMultiWordCurrentIndex(int input) { multiWordCurrentIndex = input; }
    int getNumMultiWords() const { return numMultiWords; }

    // Mode 2, Date
    unsigned long getLastCheckDateTime() { return lastCheckDateTime; }
    void setLastCheckDateTime(unsigned long input) { lastCheckDateTime = input; }
    int getDateCheckInterval() { return checkDateInterval; }

    int getCentering() { return centering; }

    void setEspNow(SplitFlapEspNow *espNow) { this->espNow = espNow; }

    // Calibration API state (vision-guided auto-calibration, see
    // tools/calib-agent/PRODUCTION.md). The web (AsyncTCP) task only stages
    // show/preview work in PendingActions; the loop task drains it and owns
    // the display, then flips these flags. All cross-task members are
    // atomic or mutex-guarded.
    bool getCalibBusy() const { return calibBusy_.load(std::memory_order_acquire); }
    void setCalibBusy(bool busy) { calibBusy_.store(busy, std::memory_order_release); }
    int getCalibFrameId() const { return calibFrameId_.load(std::memory_order_acquire); }
    int nextCalibFrameId() { return calibFrameId_.fetch_add(1) + 1; }
    String getCalibLastFrame();
    void setCalibLastFrame(const String &frame, int frameId);
    int getCalibLastFrameId();
    // Total fleet width (local modules in single-group mode, sum across
    // groups when the master is enabled). Used to validate show frames.
    int getCalibTotalModules();

    // Cross-task mailbox: web handlers only REQUEST deferred work here; the
    // Arduino loop task drains it and performs the actual display/ESP-NOW
    // work as the single owner of the I2C bus (see PendingActions.h).
    PendingActions &getPendingActions() { return pendingActions_; }

  private:
    JsonSettings &settings;
    SplitFlapDisplay &display;
    SplitFlapEspNow *espNow = nullptr;

    String decodeURIComponent(String encodedString);
    bool validateMasterSettings(JsonVariant &json, JsonDocument &response);
    bool validateMacAddress(String macString);
    void setInputString(String input) { inputString = input; }
    void setMultiInputString(String input) { multiInputString = input; }

    void setMode(int targetMode);
    void setMultiDelay(int input) { multiWordDelay = input; }

    unsigned long lastCheckDateTime;
    int checkDateInterval;

    int connectionMode; // 0 is AP mode, 1 is Internet Mode
    int centering;      // whether to center text from custom imput

    int numMultiWords;
    unsigned long lastSwitchMultiTime;
    int multiWordDelay;
    int multiWordCurrentIndex;
    String multiInputString; // latest multi input from user

    String inputString;      // latest single input from user
    String writtenString;    // string for whatever is currently written to the display

    bool rebootRequired;
    bool attemptReconnect;
    unsigned long lastCheckWifiTime;
    int wifiCheckInterval;
    PendingActions pendingActions_;
    AsyncWebServer server; // Declare server as a class member

    // Calibration show/preview completion state. Written by the loop task
    // after it finishes display work, read by the AsyncTCP status/frame
    // handlers.
    std::atomic<bool> calibBusy_{false};
    std::atomic<int> calibFrameId_{0};
    std::mutex calibMutex_;
    String calibLastFrame_;
    int calibLastFrameId_ = 0;

    void registerCalibRoutes();
};
