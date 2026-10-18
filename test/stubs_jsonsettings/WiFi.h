// Host stub: WiFi mode query for SplitFlapMqtt AP-mode gating.
// Production uses the ESP32 Arduino WiFi class; the host test only needs
// getMode() so the AP-mode early-return path is testable.
#pragma once

typedef enum {
    WIFI_OFF = 0,
    WIFI_STA = 1,
    WIFI_AP = 2,
    WIFI_AP_STA = 3,
} wifi_mode_t;

extern wifi_mode_t g_wifiMode;

class WiFiStub {
  public:
    wifi_mode_t getMode() { return g_wifiMode; }
};

extern WiFiStub WiFi;
