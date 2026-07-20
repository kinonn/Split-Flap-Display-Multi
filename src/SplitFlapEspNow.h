#pragma once

#include "JsonSettings.h"
#include "SplitFlapDisplay.h"

#include <Arduino.h>
#include <esp_arduino_version.h>
#include <esp_now.h>

#define MAX_DISPLAY_GROUPS 6
#define ESP_NOW_REMOTE_MODE 7
#define ESP_NOW_TEXT_VERSION 1

struct SplitFlapEspNowMessage
{
    uint8_t version;
    uint8_t groupIndex;
    uint8_t moduleCount;
    char text[9];
};

class SplitFlapEspNow {
  public:
    SplitFlapEspNow(JsonSettings &settings, SplitFlapDisplay &display);

    bool init();
    void reinit();
    void loop();
    bool isMasterEnabled();
    void distributeMessage(
        const String &message, bool centering = true,
        unsigned long scrollDelayMs = DEFAULT_SCROLL_DELAY_MS,
        int scrollRepeatCount = DEFAULT_SCROLL_REPEAT_COUNT,
        bool publishState = true
    );
    void distributeFrame(const String &frame, bool publishState);

  private:
    JsonSettings &settings;
    SplitFlapDisplay &display;

    volatile bool pendingMessage;
    SplitFlapEspNowMessage pendingPacket;
    String lastRemoteText;
    bool initialized;

    bool ensureInitialized();
    int getGroupCount();
    int getGroupModuleCount(int groupIndex);
    int getTotalModuleCount();
    String getGroupMac(int groupIndex);
    String getCsvToken(const String &csv, int index);
    String sliceMessage(const String &message, int start, int width);
    String buildFrame(const String &message, int width, bool centering);
    void splitIntoChunks(
        const String &input, int width, String chunks[], int maxChunks,
        int &outCount
    );
    bool parseMacAddress(const String &macString, uint8_t mac[6]);
    bool sendToPeer(int groupIndex, const String &text, int moduleCount);
    void queueReceived(const uint8_t *data, int len);

    static SplitFlapEspNow *instance;

#if defined(ESP_ARDUINO_VERSION_MAJOR) && ESP_ARDUINO_VERSION_MAJOR >= 3
    static void handleReceive(const esp_now_recv_info_t *info, const uint8_t *data, int len);
#else
    static void handleReceive(const uint8_t *mac, const uint8_t *data, int len);
#endif
};
