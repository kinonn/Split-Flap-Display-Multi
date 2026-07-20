// Unit test for SplitFlapMqtt::setup() callback dispatch.
//
// Bug: in multi-group mode (masterGroupCount > 1) the MQTT callback routed
// messages through EspNow->distributeMessage() but never called publishState().
// The retained splitflap/splitflap/state topic therefore stayed stale.
//
// Fix: after distributeMessage() returns in the multi-group branch, publish
// the FULL original message to the state topic. Single-group continues to
// publish via display->writeString() (publishState defaults to true) — no
// change there, so we do not double-publish.
//
// This test re-implements the dispatch logic in pure C++ so it can run on
// the host without Arduino. It exercises the exact branch that changed.

#include <iostream>
#include <string>
#include <vector>
#include <cstring>

// ---- Mocks ----
struct MockEspNow {
    int distributeCount = 0;
    std::string lastMessage;
    void distributeMessage(const std::string& msg) {
        distributeCount++;
        lastMessage = msg;
    }
};

struct MockDisplay {
    int writeCount = 0;
    bool lastPublishState = false;
    std::string lastMessage;
    MockEspNow* lastEspNowPtr = nullptr;  // for verifying dispatch path

    // writeString has publishState=true by default; we use a flag to simulate
    // whether it would have published (it always publishes when called via
    // the single-group branch above).
    void writeString(const std::string& msg, bool publishState) {
        writeCount++;
        lastMessage = msg;
        lastPublishState = publishState;
    }
};

struct MockMqttClient {
    bool connected = true;
    int publishCount = 0;
    std::vector<std::string> publishedStates;
    bool isConnected() const { return connected; }
    void publishState(const std::string& msg) {
        publishCount++;
        publishedStates.push_back(msg);
    }
};

// ---- Re-implementation of the dispatch logic (post-fix) ----
//
// Mirrors src/SplitFlapMqtt.cpp lines 22-46. The only difference from
// the production code is the lambda capture of `this` is a struct
// reference, and there's no Serial.printf.
//
// Pre-fix version (for negative test): no publishState() call after
// distributeMessage(). Post-fix version: publishState() called.

struct PreFixDispatcher {
    MockMqttClient& mqttClient;
    MockDisplay& display;
    MockEspNow* espNow = nullptr;  // can be null in single-group
    int masterGroupCount = 1;
    int moduleCount = 8;

    // Pre-fix behavior: multi-group path does NOT publish.
    void onMessage(const std::string& message) {
        if (masterGroupCount > 1 && espNow) {
            espNow->distributeMessage(message);
        } else {
            display.writeString(message, true);  // publishState=true (default)
        }
    }
};

struct PostFixDispatcher {
    MockMqttClient& mqttClient;
    MockDisplay& display;
    MockEspNow* espNow = nullptr;
    int masterGroupCount = 1;
    int moduleCount = 8;

    // Post-fix behavior: multi-group path publishes the full original
    // message via publishState() after distributeMessage() returns.
    void onMessage(const std::string& message) {
        if (masterGroupCount > 1 && espNow) {
            espNow->distributeMessage(message);
            if (mqttClient.isConnected()) {
                mqttClient.publishState(message);
            }
        } else {
            display.writeString(message, true);
        }
    }
};

// ---- Tests ----

static int testFailures = 0;
static int testCount = 0;

#define EXPECT(cond, label) do { \
    testCount++; \
    if (!(cond)) { \
        std::cout << "  FAIL: " << label << std::endl; \
        testFailures++; \
    } else { \
        std::cout << "  ok:   " << label << std::endl; \
    } \
} while (0)

void testPreFix_MultiGroupNoPublish() {
    std::cout << "\n[TEST] pre-fix: multi-group does NOT publish (reproduces the bug)" << std::endl;
    MockMqttClient mqtt;
    MockDisplay display;
    MockEspNow espNow;
    PreFixDispatcher d{mqtt, display, &espNow, /*masterGroupCount=*/2, 8};
    const std::string msg = "HELLO WORLD";

    d.onMessage(msg);

    EXPECT(espNow.distributeCount == 1, "espNow.distributeMessage called");
    EXPECT(display.writeCount == 0, "display.writeString NOT called in multi-group path");
    EXPECT(mqtt.publishCount == 0, "BUG: no MQTT state publish (this is the bug)");
}

void testPostFix_MultiGroupPublishesFullMessage() {
    std::cout << "\n[TEST] post-fix: multi-group publishes FULL original message" << std::endl;
    MockMqttClient mqtt;
    MockDisplay display;
    MockEspNow espNow;
    PostFixDispatcher d{mqtt, display, &espNow, /*masterGroupCount=*/2, 8};
    const std::string msg = "HELLO WORLD THIS IS A LONG MESSAGE";  // 34 chars

    d.onMessage(msg);

    EXPECT(espNow.distributeCount == 1, "espNow.distributeMessage called");
    EXPECT(display.writeCount == 0, "display.writeString NOT called in multi-group path");
    EXPECT(mqtt.publishCount == 1, "MQTT state publish called exactly once");
    EXPECT(mqtt.publishedStates.size() == 1 && mqtt.publishedStates[0] == msg,
           "published the FULL 34-char original message (not a chunk)");
}

void testPostFix_SingleGroupUnchanged() {
    std::cout << "\n[TEST] post-fix: single-group behavior unchanged" << std::endl;
    MockMqttClient mqtt;
    MockDisplay display;
    MockEspNow espNow;  // present but unused
    PostFixDispatcher d{mqtt, display, &espNow, /*masterGroupCount=*/1, 8};
    const std::string msg = "HELLO";

    d.onMessage(msg);

    EXPECT(espNow.distributeCount == 0, "espNow.distributeMessage NOT called in single-group");
    EXPECT(display.writeCount == 1, "display.writeString called once (existing behavior)");
    EXPECT(display.lastPublishState == true, "writeString called with publishState=true");
    EXPECT(mqtt.publishCount == 0,
           "no extra publishState() call in single-group (writeString handles it - no double publish)");
}

void testPostFix_MultiGroupBrokerDisconnected() {
    std::cout << "\n[TEST] post-fix: multi-group, broker disconnected" << std::endl;
    MockMqttClient mqtt;
    mqtt.connected = false;
    MockDisplay display;
    MockEspNow espNow;
    PostFixDispatcher d{mqtt, display, &espNow, /*masterGroupCount=*/2, 8};

    d.onMessage("HELLO");

    EXPECT(espNow.distributeCount == 1, "distribution still happens");
    EXPECT(mqtt.publishCount == 0, "no publish attempt when broker disconnected");
}

void testPostFix_MultiGroupThreeGroups() {
    std::cout << "\n[TEST] post-fix: multi-group, 3 groups" << std::endl;
    MockMqttClient mqtt;
    MockDisplay display;
    MockEspNow espNow;
    PostFixDispatcher d{mqtt, display, &espNow, /*masterGroupCount=*/3, 8};
    const std::string msg = "TUESDAY";

    d.onMessage(msg);

    EXPECT(mqtt.publishCount == 1, "publishes once for 3-group setup");
    EXPECT(mqtt.publishedStates[0] == msg, "publishes the full message");
}

void testPostFix_MultiGroupNoEspNowWired() {
    std::cout << "\n[TEST] post-fix: multi-group setting but no EspNow (degraded fallback)" << std::endl;
    MockMqttClient mqtt;
    MockDisplay display;
    // espNow == nullptr: condition `masterGroupCount > 1 && espNow` is false,
    // so we fall through to the single-group path. This documents the
    // existing degraded behavior (not changed by the fix).
    PostFixDispatcher d{mqtt, display, /*espNow=*/nullptr, /*masterGroupCount=*/2, 8};

    d.onMessage("HELLO");

    EXPECT(display.writeCount == 1, "falls back to display.writeString when espNow null");
    EXPECT(mqtt.publishCount == 0, "no extra publish (writeString handles it)");
}

int main() {
    std::cout << "=== SplitFlapMqtt callback dispatch - regression tests ===" << std::endl;
    testPreFix_MultiGroupNoPublish();
    testPostFix_MultiGroupPublishesFullMessage();
    testPostFix_SingleGroupUnchanged();
    testPostFix_MultiGroupBrokerDisconnected();
    testPostFix_MultiGroupThreeGroups();
    testPostFix_MultiGroupNoEspNowWired();

    std::cout << "\n=== Summary ===" << std::endl;
    std::cout << "Passed: " << (testCount - testFailures) << "/" << testCount << std::endl;
    if (testFailures == 0) {
        std::cout << "ALL TESTS PASSED" << std::endl;
        return 0;
    } else {
        std::cout << testFailures << " TEST(S) FAILED" << std::endl;
        return 1;
    }
}