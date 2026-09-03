// Host stub: PubSubClient test double. Mirrors the real library's default
// receive-buffer size (256, PubSubClient.h MQTT_MAX_PACKET_SIZE) and records
// setBufferSize() so tests can assert the firmware raised it. One client per
// process, so the buffer size lives in a global the ctor resets.
#pragma once

#include <cstddef>
#include <functional>
#include <string>

#include "WiFiClient.h"

typedef unsigned char byte; // Arduino.h compatibility

// Buffer state for assertions. Reset to the library default on construction.
inline size_t g_clientBufferSize = 256;
inline int g_publishCalls = 0;

class PubSubClient {
  public:
    using Callback = std::function<void(char *, byte *, unsigned int)>;

    explicit PubSubClient(WiFiClient &client) : client_(client) { g_clientBufferSize = 256; }

    void setServer(const char *host, int port) {
        server_ = host;
        port_ = port;
    }
    void setCallback(Callback cb) { callback_ = cb; }

    bool connect(const char *id) { return connectImpl(id, nullptr, nullptr); }
    bool connect(const char *id, const char *user, const char *pass) { return connectImpl(id, user, pass); }

    bool connected() const { return connected_; }
    bool subscribe(const char *topic) {
        topics_.push_back(topic);
        return true;
    }
    bool publish(const char *topic, const char *payload, bool retain) {
        (void) retain;
        g_publishCalls++;
        lastTopic_ = topic;
        lastPayload_ = payload;
        return true;
    }
    void loop() {}

    bool setBufferSize(size_t size) {
        if (size > 0 && size <= 268435455) {
            g_clientBufferSize = size;
            return true;
        }
        return false;
    }
    size_t getBufferSize() const { return g_clientBufferSize; }

    // Test hooks
    void forceConnected(bool on) { connected_ = on; }
    void deliver(char *topic, byte *payload, unsigned int len) {
        if (callback_) callback_(topic, payload, len);
    }
    const std::string &lastTopic() const { return lastTopic_; }
    const std::string &lastPayload() const { return lastPayload_; }

  private:
    bool connectImpl(const char *id, const char *user, const char *pass) {
        clientId_ = id;
        user_ = user ? user : "";
        pass_ = pass ? pass : "";
        return connected_; // stays false unless the test forces it
    }

    WiFiClient &client_;
    Callback callback_;
    std::string server_;
    int port_ = 1883;
    std::string clientId_, user_, pass_;
    std::vector<std::string> topics_;
    std::string lastTopic_, lastPayload_;
    bool connected_ = false;
};
