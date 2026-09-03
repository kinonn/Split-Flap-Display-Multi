// Host stub: only the ESP-IDF declarations src/SplitFlapEspNow.h expects from
// <esp_now.h>. All firmware structs/macros live in the real header itself.
#pragma once

#include <cstdint>

#define ESP_ARDUINO_VERSION_MAJOR 3

typedef int esp_err_t;
#define ESP_OK 0

typedef struct {
    uint8_t *src_addr;
} esp_now_recv_info_t;

typedef struct {
    uint8_t peer_addr[6];
    uint8_t channel;
    bool encrypt;
    int ifidx;
} esp_now_peer_info_t;
