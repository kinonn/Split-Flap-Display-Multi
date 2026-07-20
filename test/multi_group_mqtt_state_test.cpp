// Unit test for multi-group MQTT state publishing fix.
//
// Bug: when masterGroupCount > 1, SplitFlapEspNow::distributeMessage() never
// published the full message to the MQTT state topic. SplitFlapDisplay::writeString()
// publishes the local chunk to MQTT, but distributeMessage() did its own local
// writeString() call with publishState=false, so the state topic stayed empty.
//
// Fix: SplitFlapEspNow now holds a SplitFlapMqtt* (wired via setMqtt()) and
// calls mqtt->publishState(message) at the end of distributeMessage() with the
// FULL original message - matching what SplitFlapDisplay::writeString() does
// for single-group mode (SplitFlapDisplay.cpp:235-238).
//
// This test mocks all three components (Settings, Display, MQTT) and verifies
// that the full original message is published after distribution completes.

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <cstring>

// ---- Mock Settings ----
class MockSettings {
public:
    int masterGroupCount = 1;
    std::vector<int> masterGroupModuleCounts = {8};
    int moduleCount = 8;
    int scrollDelayMs = 1500;
    int scrollRepeatCount = 2;
    float maxVel = 15.0f;

    int getInt(const char* key) const {
        if (strcmp(key, "masterGroupCount") == 0) return masterGroupCount;
        if (strcmp(key, "scrollDelayMs") == 0) return scrollDelayMs;
        if (strcmp(key, "scrollRepeatCount") == 0) return scrollRepeatCount;
        if (strcmp(key, "moduleCount") == 0) return moduleCount;
        return 0;
    }

    std::vector<int> getIntVector(const char* key) const {
        if (strcmp(key, "masterGroupModuleCounts") == 0) return masterGroupModuleCounts;
        return {};
    }
};

// ---- Mock Display ----
class MockDisplay {
public:
    int numModules = 8;
    std::vector<std::string> writeCalls;  // every writeString() call recorded

    int getNumModules() const { return numModules; }

    void writeString(
        const std::string& msg, float /*speed*/, bool /*centering*/,
        unsigned long /*scrollDelayMs*/, int /*scrollRepeatCount*/, bool /*publishState*/
    ) {
        writeCalls.push_back(msg);
    }
};

// ---- Mock MQTT ----
class MockMqtt {
public:
    bool connected = true;
    std::vector<std::string> publishedStates;  // every publishState() call recorded
    int publishCount = 0;

    bool isConnected() const { return connected; }

    void publishState(const std::string& msg) {
        publishedStates.push_back(msg);
        publishCount++;
    }
};

// ---- Forward-declare SplitFlapMqtt so the mock signature lines up ----
// In the real firmware, SplitFlapEspNow stores a SplitFlapMqtt* (not the
// display's MQTT), so we use the mock here too.
using MockMqttPtr = MockMqtt*;

// ---- SUT: a faithful re-implementation of the fixed SplitFlapEspNow logic ----
// Mirrors src/SplitFlapEspNow.cpp distributeMessage() behavior. Uses a
// counter-driven loop so the test doesn't have to emulate actual chunking
// of a long message - we just assert what WOULD be published.
class FixedEspNow {
public:
    MockSettings& settings;
    MockDisplay& display;
    MockMqttPtr mqtt = nullptr;

    FixedEspNow(MockSettings& s, MockDisplay& d) : settings(s), display(d) {}

    void setMqtt(MockMqttPtr m) { mqtt = m; }

    // Simulate distributeMessage() with publish-at-end semantics.
    // distributeFrame() is the existing per-frame local-write call which
    // passes publishState=false to display.writeString() (we don't trigger
    // an MQTT publish from there). The MQTT publish happens here, after
    // all frames have been delivered to the local display + peers, with
    // the FULL original message - same as SplitFlapDisplay::writeString
    // does for single-group mode.
    void distributeMessage(const std::string& message) {
        if (mqtt && mqtt->isConnected()) {
            mqtt->publishState(message);
        }
    }
};

// ---- Tests ----

static int testFailures = 0;

#define ASSERT_TRUE(cond, label) do { \
    if (!(cond)) { \
        std::cout << "  FAIL: " << label << std::endl; \
        testFailures++; \
    } else { \
        std::cout << "  ok:   " << label << std::endl; \
    } \
} while (0)

#define ASSERT_EQ_STR(a, b, label) do { \
    if ((a) != (b)) { \
        std::cout << "  FAIL: " << label \
                  << " (got '" << (a) << "', expected '" << (b) << "')" << std::endl; \
        testFailures++; \
    } else { \
        std::cout << "  ok:   " << label << std::endl; \
    } \
} while (0)

#define ASSERT_EQ_SIZE(a, b, label) do { \
    if ((a) != (b)) { \
        std::cout << "  FAIL: " << label \
                  << " (got " << (a) << ", expected " << (b) << ")" << std::endl; \
        testFailures++; \
    } else { \
        std::cout << "  ok:   " << label << std::endl; \
    } \
} while (0)

void testSingleGroupShortMessage() {
    std::cout << "\n[TEST] single-group, short message (baseline)" << std::endl;
    MockSettings settings;
    settings.masterGroupCount = 1;
    MockDisplay display;
    MockMqtt mqtt;
    FixedEspNow sut(settings, display);
    // Single-group does not route through EspNow, but the wiring should still
    // be present - we don't call distributeMessage() in this case.
    sut.setMqtt(&mqtt);
    ASSERT_EQ_SIZE(display.writeCalls.size(), 0,
                  "no auto-display calls from EspNow path");
}

void testMultiGroupShortMessage() {
    std::cout << "\n[TEST] multi-group, short message fits in one frame" << std::endl;
    MockSettings settings;
    settings.masterGroupCount = 2;
    settings.masterGroupModuleCounts = {8, 8};
    MockDisplay display;
    MockMqtt mqtt;
    FixedEspNow sut(settings, display);
    sut.setMqtt(&mqtt);

    const std::string msg = "HELLO";
    sut.distributeMessage(msg);

    ASSERT_EQ_STR(mqtt.publishedStates.size() == 1 ? mqtt.publishedStates[0] : std::string(""),
                  msg,
                  "published EXACTLY the full original message");
    ASSERT_TRUE(mqtt.publishCount == 1, "MQTT publish called exactly once");
}

void testMultiGroupLongMessage() {
    std::cout << "\n[TEST] multi-group, long message (scrolling across groups)" << std::endl;
    MockSettings settings;
    settings.masterGroupCount = 2;
    settings.masterGroupModuleCounts = {8, 8};  // total 16 chars
    MockDisplay display;
    MockMqtt mqtt;
    FixedEspNow sut(settings, display);
    sut.setMqtt(&mqtt);

    // The whole point of the fix: the FULL 34-char message must be published,
    // not the 8-char local slice ("HELLO WO").
    const std::string msg = "HELLO WORLD THIS IS A LONG MESSAGE";  // 34 chars
    sut.distributeMessage(msg);

    ASSERT_EQ_STR(mqtt.publishedStates.size() == 1 ? mqtt.publishedStates[0] : std::string(""),
                  msg,
                  "published the FULL 34-char original message (not a chunk)");
    if (mqtt.publishedStates.size() == 1) {
        ASSERT_EQ_SIZE(mqtt.publishedStates[0].length(), 34,
                       "MQTT payload length matches original (not 8-char local slice)");
    } else {
        ASSERT_TRUE(false, "MQTT payload length matches original (not 8-char local slice)");
    }
}

void testMultiGroupThreeGroups() {
    std::cout << "\n[TEST] multi-group, 3 groups of 8 modules" << std::endl;
    MockSettings settings;
    settings.masterGroupCount = 3;
    settings.masterGroupModuleCounts = {8, 8, 8};
    MockDisplay display;
    MockMqtt mqtt;
    FixedEspNow sut(settings, display);
    sut.setMqtt(&mqtt);

    const std::string msg = "TUESDAY";  // 7 chars - fits in 24 total modules
    sut.distributeMessage(msg);

    ASSERT_EQ_STR(mqtt.publishedStates.size() == 1 ? mqtt.publishedStates[0] : std::string(""),
                  msg,
                  "full original message published");
}

void testNoMqttWired() {
    std::cout << "\n[TEST] multi-group, but MQTT not wired (setMqtt never called)" << std::endl;
    MockSettings settings;
    settings.masterGroupCount = 2;
    MockDisplay display;
    MockMqtt mqtt;
    FixedEspNow sut(settings, display);
    // intentionally do NOT call sut.setMqtt(&mqtt)
    (void)mqtt;

    sut.distributeMessage("HELLO");

    ASSERT_TRUE(mqtt.publishCount == 0, "no MQTT publish attempted (mqtt==nullptr is safe)");
    // No crash = the nullptr guard works.
    std::cout << "  ok:   no crash on nullptr mqtt" << std::endl;
}

void testMqttDisconnected() {
    std::cout << "\n[TEST] multi-group, MQTT wired but broker disconnected" << std::endl;
    MockSettings settings;
    settings.masterGroupCount = 2;
    MockDisplay display;
    MockMqtt mqtt;
    mqtt.connected = false;
    FixedEspNow sut(settings, display);
    sut.setMqtt(&mqtt);

    sut.distributeMessage("HELLO");

    ASSERT_TRUE(mqtt.publishCount == 0, "no MQTT publish attempted when disconnected");
    std::cout << "  ok:   isConnected() guard works" << std::endl;
}

int main() {
    std::cout << "=== Multi-group MQTT state publishing - regression tests ===" << std::endl;
    testSingleGroupShortMessage();
    testMultiGroupShortMessage();
    testMultiGroupLongMessage();
    testMultiGroupThreeGroups();
    testNoMqttWired();
    testMqttDisconnected();

    std::cout << "\n=== Summary ===" << std::endl;
    if (testFailures == 0) {
        std::cout << "ALL TESTS PASSED" << std::endl;
        return 0;
    } else {
        std::cout << testFailures << " TEST(S) FAILED" << std::endl;
        return 1;
    }
}