#pragma once

// Inter-display link transport modes. Shared by the firmware settings default
// (SplitFlapDisplay.ino LINK_TRANSPORT_DEFAULT), the wired transport
// (WiredLink), and the host tests below. Arduino-free so host tests compile
// the real header — same pattern as StepMath.h / CalibApi.h.
#define LINK_TRANSPORT_ESP_NOW 0
#define LINK_TRANSPORT_RS485 1
#define LINK_TRANSPORT_UART 2

// RS-485 is addressed generally: master = 0, remotes = 1..5.
#define LINK_MAX_GROUPS 6
// Direct UART is point-to-point: exactly master + one remote.
#define LINK_UART_MAX_GROUPS 2

// Validates (transport, groupCount). Returns nullptr when valid, otherwise a
// pointer to a static first-error message. Transport 0 (ESP-NOW) is always
// valid here — its own bounds live in the pre-existing ESP-NOW code, which
// this plan freezes.
inline const char *validateWiredSettings(int transport, int groupCount) {
    switch (transport) {
        case LINK_TRANSPORT_ESP_NOW: return nullptr;
        case LINK_TRANSPORT_RS485:
            if (groupCount < 1 || groupCount > LINK_MAX_GROUPS) {
                return "RS-485 supports 1..6 groups";
            }
            return nullptr;
        case LINK_TRANSPORT_UART:
            if (groupCount < 1 || groupCount > LINK_UART_MAX_GROUPS) {
                return "Direct UART supports at most 2 groups";
            }
            return nullptr;
        default: return "Unknown linkTransport";
    }
}
