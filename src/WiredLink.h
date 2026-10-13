#pragma once

#include "JsonSettings.h"
#include "LinkFrame.h"
#include "LinkSettings.h"
#include "SplitFlapDisplay.h"

#include <Arduino.h>

// UART wired transport for inter-display coordination (RS-485 multi-drop
// and direct UART point-to-point share this class; auto-direction RS-485
// transceivers need no direction GPIO, so the two modes differ only in
// validation and addressing).
//
// Mirrors the caller-facing API of SplitFlapEspNow so call sites dispatch on
// the linkTransport setting with no logic changes to the ESP-NOW path:
//   - address 0 = master (or standalone when groupCount == 1)
//   - address 1..5 = remote; a remote's linkAddress is its group slot
//   - the master sends one addressed unicast per group; the existing
//     SplitFlapEspNow payload structs travel as frame payloads verbatim
//
// Unlike ESP-NOW, everything runs on the loop task (loop() polls Serial1),
// so there is no cross-task handoff and no spinlock.
//
// Transport, address, pins and baud are snapshotted at init(): changing them
// via the web UI requires a reboot. Group counts and module counts are read
// live, like the ESP-NOW code does.
class SplitFlapWired {
  public:
    SplitFlapWired(JsonSettings &settings, SplitFlapDisplay &display);

    bool init();
    void loop();
    bool isMasterEnabled();
    int getDiscoveredCount();
    String getDiscoveredPeersJson();
    void pushOffsetsToGroup(int groupIndex);
    void reportOffsetsToMaster();
    void distributeMessage(
        const String &message, bool centering = true, unsigned long scrollDelayMs = DEFAULT_SCROLL_DELAY_MS,
        int scrollRepeatCount = DEFAULT_SCROLL_REPEAT_COUNT
    );
    // Total module count across all groups (group 0 = local display). Equals
    // the local count in single-group mode. Read-only; used by MQTT status
    // reporting of the physical display size.
    int getTotalModuleCount();

  private:
    JsonSettings &settings;
    SplitFlapDisplay &display;

    bool initialized;
    int transport;
    int myAddress;
    int txPin;
    int rxPin;
    long baud;

    LinkDecoder decoder;
    String lastRemoteText;

    // Master-side discovery: last poll per remote address + ack tracking.
    struct PeerState
    {
        uint8_t moduleCount;
        unsigned long lastSeenMs;
        bool seen;
        bool ackPending;
        unsigned long ackSentMs;
        bool ackLogged;
    };
    PeerState peers[LINK_MAX_GROUPS];
    unsigned long lastPollMs;
    unsigned long lastExpiryCheckMs;

    // Remote-side offsets settle (mirrors the ESP-NOW reload delay).
    bool offsetDataDirty;
    unsigned long lastOffsetRxMs;

    int getGroupCount();
    int getGroupModuleCount(int groupIndex);
    String sliceMessage(const String &message, int start, int width);
    String buildFrame(const String &message, int width, bool centering);
    void distributeFrame(const String &frame);
    void splitIntoChunks(const String &input, int width, String chunks[], int maxChunks, int &outCount);
    bool sendFrame(uint8_t dst, uint8_t type, const void *payload, size_t payloadLen);
    bool sendToPeer(int groupIndex, const String &text, int moduleCount);
    void handleFrame();
    void processAnnouncement(uint8_t fromAddr, const uint8_t *payload, size_t len);
    void applyOffsetsPush(const uint8_t *payload, size_t len);
    void applyCharOffsetsPush(const uint8_t *payload, size_t len);
    void processOffsetsReport(uint8_t fromAddr, const uint8_t *payload, size_t len);
    void processCharOffsetsReport(uint8_t fromAddr, const uint8_t *payload, size_t len);
};
