#pragma once

#include "JsonSettings.h"
#include "SplitFlapDisplay.h"

#include <Arduino.h>
#include <esp_arduino_version.h>
#include <esp_now.h>

#include <freertos/FreeRTOS.h>
#include <freertos/task.h>

#define MAX_DISPLAY_GROUPS 6
#define ESP_NOW_REMOTE_MODE 7
#define ESP_NOW_TEXT_VERSION 1
#define ESP_NOW_ANNOUNCE_VERSION 0xFE
#define ESP_NOW_OFFSETS_PUSH 0xFC
#define ESP_NOW_OFFSETS_REPORT 0xFB
#define OFFSET_RELOAD_SETTLE_MS 250
#define OFFSET_PACKET_SPACING_MS 10

struct SplitFlapEspNowMessage
{
    uint8_t version;
    uint8_t groupIndex;
    uint8_t moduleCount;
    char text[9];
};

struct SplitFlapAnnounceMessage
{
    uint8_t version;
    uint8_t moduleCount;
};

struct SplitFlapOffsetsPushMessage
{
    uint8_t version;
    uint8_t groupIndex;
    uint8_t moduleCount;
    int16_t displayOffset;
    int16_t moduleOffsets[8];
};

struct SplitFlapCharOffsetsPushMessage
{
    uint8_t version;
    uint8_t groupIndex;
    uint8_t moduleIndex;
    int8_t charOffsets[48];
};

struct SplitFlapOffsetsReportMessage
{
    uint8_t version;
    uint8_t moduleCount;
    int16_t displayOffset;
    int16_t moduleOffsets[8];
};

// Compile-time guards: these packet fields are sized for the current
// MAX_MODULES (8). Raising MAX_MODULES without resizing them would silently
// overflow the packet struct at the memcpy sites, so tie them together.
static_assert(
    sizeof(SplitFlapEspNowMessage::text) >= MAX_MODULES + 1,
    "SplitFlapEspNowMessage::text must hold MAX_MODULES chars + NUL"
);
static_assert(
    sizeof(SplitFlapOffsetsPushMessage::moduleOffsets) / sizeof(int16_t) >= MAX_MODULES,
    "SplitFlapOffsetsPushMessage::moduleOffsets must hold MAX_MODULES entries"
);
static_assert(
    sizeof(SplitFlapOffsetsReportMessage::moduleOffsets) / sizeof(int16_t) >= MAX_MODULES,
    "SplitFlapOffsetsReportMessage::moduleOffsets must hold MAX_MODULES entries"
);

struct SplitFlapCharOffsetsReportMessage
{
    uint8_t version;
    uint8_t moduleIndex;
    int8_t charOffsets[48];
};

struct DiscoveredPeer
{
    uint8_t mac[6];
    uint8_t moduleCount;
    unsigned long lastSeenMs;
};

class SplitFlapEspNow {
  public:
    SplitFlapEspNow(JsonSettings &settings, SplitFlapDisplay &display);

    bool init();
    void reinit();
    void loop();
    bool isMasterEnabled();
    int getDiscoveredCount();
    String getDiscoveredPeersJson();
    bool isMacAssigned(const String &mac);
    void pushOffsetsToGroup(int groupIndex);
    void reportOffsetsToMaster();
    void processPendingOffsetPackets();
    void distributeMessage(
        const String &message, bool centering = true,
        unsigned long scrollDelayMs = DEFAULT_SCROLL_DELAY_MS,
        int scrollRepeatCount = DEFAULT_SCROLL_REPEAT_COUNT
    );
    // Total module count across all groups (group 0 = local display). Equals
    // the local count in single-group mode. Read-only; used by MQTT status
    // reporting of the physical display size.
    int getTotalModuleCount();

  private:
    JsonSettings &settings;
    SplitFlapDisplay &display;

    volatile bool pendingMessage;
    // Spinlock guarding the packet handoff between the WiFi/protocol task
    // (queueReceived) and the loop task (loop/processPendingOffsetPackets).
    // noInterrupts() is not sufficient: it only stops interrupts on the
    // calling core, so on dual-core targets (esp32_s3) the writer and reader
    // could still interleave. portENTER_CRITICAL spins across cores.
    portMUX_TYPE packetMux = portMUX_INITIALIZER_UNLOCKED;
    SplitFlapEspNowMessage pendingPacket;
    String lastRemoteText;
    bool initialized;

    DiscoveredPeer discoveredPeers[MAX_DISPLAY_GROUPS];
    int discoveredCount;
    // Spinlock for the discovered-peer list, shared by the receive task
    // (processAnnouncement) and the loop task (expiry sweep, JSON dump).
    portMUX_TYPE peerListMux = portMUX_INITIALIZER_UNLOCKED;
    unsigned long lastAnnounceMs;
    unsigned long lastExpiryCheckMs;
    uint8_t masterMac[6];
    bool masterMacKnown;

    volatile bool pendingOffsetsPush;
    SplitFlapOffsetsPushMessage pendingOffsetsPushPkt;
    volatile uint8_t pendingCharOffsetsMask;
    SplitFlapCharOffsetsPushMessage pendingCharOffsetsPkts[MAX_MODULES];
    bool offsetDataDirty;
    unsigned long lastOffsetRxMs;

    bool ensureInitialized();
    int getGroupCount();
    int getGroupModuleCount(int groupIndex);
    String getGroupMac(int groupIndex);
    String getCsvToken(const String &csv, int index);
    String sliceMessage(const String &message, int start, int width);
    String buildFrame(const String &message, int width, bool centering);
    void distributeFrame(const String &frame);
    void splitIntoChunks(
        const String &input, int width, String chunks[], int maxChunks,
        int &outCount
    );
    bool parseMacAddress(const String &macString, uint8_t mac[6]);
    bool sendToPeer(int groupIndex, const String &text, int moduleCount);
    void queueReceived(const uint8_t *mac, const uint8_t *data, int len);
    void broadcastAnnouncement();
    void processAnnouncement(const uint8_t *mac, const SplitFlapAnnounceMessage *pkt);
    void applyOffsetsPush(const SplitFlapOffsetsPushMessage *pkt);
    void applyCharOffsetsPush(const SplitFlapCharOffsetsPushMessage *pkt);
    void processOffsetsReport(const uint8_t *mac, const SplitFlapOffsetsReportMessage *pkt);
    void processCharOffsetsReport(const uint8_t *mac, const SplitFlapCharOffsetsReportMessage *pkt);
    int groupIndexForMac(const uint8_t mac[6]);
    void ensurePeer(const uint8_t mac[6]);
    void learnMasterMac(const uint8_t mac[6]);
    String macToString(const uint8_t mac[6]);

    static SplitFlapEspNow *instance;

#if defined(ESP_ARDUINO_VERSION_MAJOR) && ESP_ARDUINO_VERSION_MAJOR >= 3
    static void handleReceive(const esp_now_recv_info_t *info, const uint8_t *data, int len);
#else
    static void handleReceive(const uint8_t *mac, const uint8_t *data, int len);
#endif
};
