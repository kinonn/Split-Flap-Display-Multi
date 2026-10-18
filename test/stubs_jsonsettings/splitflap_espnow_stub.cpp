// Link-time stubs for symbols SplitFlapMqtt.cpp references but never executes
// on the host (espNow/wired are nullptr in these tests; the display is not
// driven).
// SplitFlapEspNow.cpp / SplitFlapDisplay.cpp / WiredLink.cpp need the ESP-IDF
// WiFi stack, real UART/I2C, and are not host-compilable.
#include "SplitFlapDisplay.h"
#include "SplitFlapEspNow.h"
#include "WiredLink.h"

#include <cstring>
#include <string>

// Test counters: which dispatch branch ran, with what message.
int g_distributeCalls = 0;
std::string g_lastDistributed;
int g_wiredDistributeCalls = 0;
std::string g_lastWiredDistributed;
int g_writeStringCalls = 0;
std::string g_lastWritten;
float g_lastWriteSpeed = 0.0f;

void SplitFlapEspNow::distributeMessage(
    const String &message, bool centering, unsigned long scrollDelayMs,
    int scrollRepeatCount
) {
    (void) centering;
    (void) scrollDelayMs;
    (void) scrollRepeatCount;
    g_distributeCalls++;
    g_lastDistributed = message.c_str();
}

int SplitFlapEspNow::getTotalModuleCount() { return 0; }

void SplitFlapWired::distributeMessage(
    const String &message, bool centering, unsigned long scrollDelayMs,
    int scrollRepeatCount
) {
    (void) centering;
    (void) scrollDelayMs;
    (void) scrollRepeatCount;
    g_wiredDistributeCalls++;
    g_lastWiredDistributed = message.c_str();
}

int SplitFlapWired::getTotalModuleCount() { return 0; }

void SplitFlapDisplay::writeString(
    String inputString, float speed, bool centering, unsigned long scrollDelayMs,
    int scrollRepeatCount, bool publishState
) {
    (void) centering;
    (void) scrollDelayMs;
    (void) scrollRepeatCount;
    (void) publishState;
    g_writeStringCalls++;
    g_lastWritten = inputString.c_str();
    g_lastWriteSpeed = speed;
}

// Ctor stubs for SplitFlapDisplay / SplitFlapEspNow (the real ctors live in
// the ESP-IDF-only .cpp files). Test paths only construct these objects and
// pass them to SplitFlapMqtt; no display or radio logic runs.
SplitFlapDisplay::SplitFlapDisplay(JsonSettings &settings)
    : settings(settings), numModules(8), maxConcurrentMotors(-1) {
    for (int i = 0; i < MAX_MODULES; i++) {
        lastDisplayedChar[i] = ' ';
    }
}

SplitFlapModule::SplitFlapModule() {}

SplitFlapEspNow::SplitFlapEspNow(JsonSettings &settings, SplitFlapDisplay &display)
    : settings(settings), display(display), pendingMessage(false),
      lastRemoteText(""), initialized(false), discoveredCount(0),
      lastAnnounceMs(0), lastExpiryCheckMs(0), masterMacKnown(false),
      pendingOffsetsPush(false), pendingCharOffsetsMask(0),
      offsetDataDirty(false), lastOffsetRxMs(0) {
    memset(discoveredPeers, 0, sizeof(discoveredPeers));
    memset(masterMac, 0, sizeof(masterMac));
}

// Ctor stub for SplitFlapWired (real ctor needs UART + live settings; test
// paths only construct it and pass it to SplitFlapMqtt).
SplitFlapWired::SplitFlapWired(JsonSettings &settings, SplitFlapDisplay &display)
    : settings(settings), display(display), initialized(false),
      transport(LINK_TRANSPORT_ESP_NOW), myAddress(0), txPin(21), rxPin(20),
      baud(115200), lastRemoteText(""), lastPollMs(0), lastExpiryCheckMs(0),
      offsetDataDirty(false), lastOffsetRxMs(0) {
    memset(peers, 0, sizeof(peers));
}

