// Unit test for multi-group MQTT state publishing fix
// This test validates the logic flow for publishState parameter propagation
// in multi-group mode (masterGroupCount > 1)

#include <string>
#include <vector>
#include <iostream>
#include <algorithm>
#include <cstring>

// Mock classes to test the logic flow without hardware
class MockSettings {
public:
    int masterGroupCount = 1;
    std::vector<int> masterGroupModuleCounts = {8};
    std::vector<std::string> masterGroupMacs = {""};
    int moduleCount = 8;
    int scrollDelayMs = 1500;
    int scrollRepeatCount = 2;
    float maxVel = 15.0f;
    
    int getInt(const char* key) {
        if (strcmp(key, "masterGroupCount") == 0) return masterGroupCount;
        if (strcmp(key, "scrollDelayMs") == 0) return scrollDelayMs;
        if (strcmp(key, "scrollRepeatCount") == 0) return scrollRepeatCount;
        if (strcmp(key, "moduleCount") == 0) return moduleCount;
        return 0;
    }
    
    std::vector<int> getIntVector(const char* key) {
        if (strcmp(key, "masterGroupModuleCounts") == 0) return masterGroupModuleCounts;
        return {};
    }
    
    std::string getString(const char* key) {
        if (strcmp(key, "masterGroupMacs") == 0) {
            std::string result;
            for (size_t i = 0; i < masterGroupMacs.size(); ++i) {
                if (i > 0) result += ",";
                result += masterGroupMacs[i];
            }
            return result;
        }
        return "";
    }
    
    float getFloat(const char* key) {
        if (strcmp(key, "maxVel") == 0) return maxVel;
        return 0;
    }
};

class MockDisplay {
public:
    int numModules = 8;
    int getNumModules() { return numModules; }
    std::string lastWrittenMessage;
    bool lastPublishState = false;
    
    void writeString(const std::string& msg, float speed, bool centering, 
                     unsigned long scrollDelayMs, int scrollRepeatCount, bool publishState) {
        lastWrittenMessage = msg;
        lastPublishState = publishState;
        std::cout << "[MockDisplay] writeString: '" << msg << "' publishState=" << publishState << std::endl;
    }
};

class MockMqtt {
public:
    bool connected = true;
    std::string lastPublishedState;
    bool isConnected() { return connected; }
    void publishState(const std::string& msg) {
        lastPublishedState = msg;
        std::cout << "[MockMqtt] publishState: '" << msg << "'" << std::endl;
    }
};

class MockEspNow {
public:
    MockSettings& settings;
    MockDisplay& display;
    MockMqtt* mqtt = nullptr;
    bool initialized = false;
    
    MockEspNow(MockSettings& s, MockDisplay& d) : settings(s), display(d) {}
    
    bool init() { 
        initialized = true; 
        return true; 
    }
    
    bool isMasterEnabled() { return settings.getInt("masterGroupCount") > 1; }
    
    bool ensureInitialized() {
        if (initialized) return true;
        if (!isMasterEnabled()) return false;
        return init();
    }
    
    int getGroupCount() { return settings.getInt("masterGroupCount"); }
    
    int getGroupModuleCount(int groupIndex) {
        if (groupIndex == 0) return display.getNumModules();
        auto counts = settings.getIntVector("masterGroupModuleCounts");
        if (groupIndex >= 0 && groupIndex < (int)counts.size()) return counts[groupIndex];
        return 8;
    }
    
    int getTotalModuleCount() {
        int total = 0;
        int groupCount = getGroupCount();
        for (int i = 0; i < groupCount; i++) {
            total += getGroupModuleCount(i);
        }
        return total;
    }
    
    std::string getGroupMac(int groupIndex) {
        auto macs = settings.getString("masterGroupMacs");
        // Simple CSV parsing for test
        std::vector<std::string> parts;
        size_t start = 0;
        size_t end = macs.find(',');
        while (end != std::string::npos) {
            parts.push_back(macs.substr(start, end - start));
            start = end + 1;
            end = macs.find(',', start);
        }
        parts.push_back(macs.substr(start));
        
        if (groupIndex >= 0 && groupIndex < (int)parts.size()) {
            return parts[groupIndex];
        }
        return "";
    }
    
    void sliceMessage(const std::string& message, int start, int width, std::string& out) {
        if (start < (int)message.length()) {
            out = message.substr(start, std::min(start + width, (int)message.length()) - start);
        }
        while ((int)out.length() < width) out += ' ';
    }
    
    void distributeFrame(const std::string& frame, bool publishState) {
        int groupCount = getGroupCount();
        int offset = 0;
        int localModuleCount = getGroupModuleCount(0);
        std::string localText;
        sliceMessage(frame, offset, localModuleCount, localText);
        
        // Send peer chunks first
        for (int groupIndex = 1; groupIndex < groupCount; groupIndex++) {
            int moduleCount = getGroupModuleCount(groupIndex);
            std::string segment;
            sliceMessage(frame, offset + localModuleCount, moduleCount, segment);
            std::cout << "[MockEspNow] send to group " << groupIndex + 1 
                      << " text='" << segment << "' modules=" << moduleCount << std::endl;
            offset += moduleCount;
        }
        
        std::cout << "[MockEspNow] local group 1 display text='" << localText 
                  << "' modules=" << localModuleCount << " publishState=" << publishState << std::endl;
        
        display.writeString(localText, 15.0f, false, 1500, 2, publishState);
    }
    
    void distributeMessage(const std::string& message, bool centering, 
                          unsigned long scrollDelayMs, int scrollRepeatCount, bool publishState) {
        if (!ensureInitialized()) return;
        
        int totalModuleCount = getTotalModuleCount();
        
        if ((int)message.length() <= totalModuleCount) {
            distributeFrame(message, publishState);  // Simplified - no centering in mock
        } else {
            // Long message - chunking logic simplified for test
            int repeats = std::clamp(scrollRepeatCount, 1, 99);
            // For test, just call distributeFrame once with the message
            // Real implementation chunks by words
            distributeFrame(message, publishState);
        }
        
        // After all display is done, publish the FULL original message to MQTT
        if (publishState && mqtt) {
            mqtt->publishState(message);
        }
    }
    
    void setMqtt(MockMqtt* m) { mqtt = m; }
};

// Test the fix
bool testSingleGroupMode() {
    std::cout << "\n=== Test: Single Group Mode (masterGroupCount=1) ===" << std::endl;
    
    MockSettings settings;
    settings.masterGroupCount = 1;
    settings.moduleCount = 8;
    
    MockDisplay display;
    MockMqtt mqtt;
    MockEspNow espnow(settings, display);
    espnow.setMqtt(&mqtt);
    
    // Simulate MQTT message received in single-group mode
    std::string message = "HELLO WORLD";
    bool isMultiGroup = settings.getInt("masterGroupCount") > 1;
    
    if (!isMultiGroup) {
        display.writeString(message, settings.getFloat("maxVel"), false, 
                           settings.getInt("scrollDelayMs"), settings.getInt("scrollRepeatCount"),
                           true);  // publishState = true in single group
    }
    
    bool passed = (display.lastWrittenMessage == message) && display.lastPublishState;
    std::cout << "Result: " << (passed ? "PASS" : "FAIL") << std::endl;
    std::cout << "  Message written: '" << display.lastWrittenMessage << "'" << std::endl;
    std::cout << "  publishState: " << display.lastPublishState << std::endl;
    return passed;
}

bool testMultiGroupModeWithFix() {
    std::cout << "\n=== Test: Multi-Group Mode WITH Fix (masterGroupCount=2) ===" << std::endl;
    
    MockSettings settings;
    settings.masterGroupCount = 2;
    settings.masterGroupModuleCounts = {8, 8};
    settings.moduleCount = 8;
    
    MockDisplay display;
    MockMqtt mqtt;
    MockEspNow espnow(settings, display);
    espnow.setMqtt(&mqtt);
    
    // Simulate MQTT message received in multi-group mode (WITH FIX: publishState=true)
    std::string message = "HELLO WORLD THIS IS A LONG MESSAGE";
    bool isMultiGroup = settings.getInt("masterGroupCount") > 1;
    
    if (isMultiGroup) {
        espnow.distributeMessage(message, false, 
                                settings.getInt("scrollDelayMs"), 
                                settings.getInt("scrollRepeatCount"),
                                true);  // FIX: publishState = true
    }
    
    // Verify the message was published to MQTT state topic
    bool passed = mqtt.lastPublishedState == message;
    std::cout << "Result: " << (passed ? "PASS" : "FAIL") << std::endl;
    std::cout << "  Original message: '" << message << "'" << std::endl;
    std::cout << "  MQTT published state: '" << mqtt.lastPublishedState << "'" << std::endl;
    std::cout << "  Local display wrote: '" << display.lastWrittenMessage << "'" << std::endl;
    return passed;
}

bool testMultiGroupModeOldBug() {
    std::cout << "\n=== Test: Multi-Group Mode OLD BUG (publishState=false) ===" << std::endl;
    
    MockSettings settings;
    settings.masterGroupCount = 2;
    settings.masterGroupModuleCounts = {8, 8};
    settings.moduleCount = 8;
    
    MockDisplay display;
    MockMqtt mqtt;
    MockEspNow espnow(settings, display);
    espnow.setMqtt(&mqtt);
    
    // Simulate MQTT message received in multi-group mode (OLD BUG: publishState=false)
    std::string message = "HELLO WORLD";
    bool isMultiGroup = settings.getInt("masterGroupCount") > 1;
    
    if (isMultiGroup) {
        espnow.distributeMessage(message, false, 
                                settings.getInt("scrollDelayMs"), 
                                settings.getInt("scrollRepeatCount"),
                                false);  // BUG: publishState = false
    }
    
    // With old bug, MQTT state should NOT be updated
    bool passed = mqtt.lastPublishedState.empty();
    std::cout << "Result: " << (passed ? "PASS (bug confirmed)" : "FAIL (unexpected behavior)") << std::endl;
    std::cout << "  MQTT published state: '" << mqtt.lastPublishedState << "' (should be empty)" << std::endl;
    return passed;
}

bool testShortMessageInMultiGroup() {
    std::cout << "\n=== Test: Short Message in Multi-Group (fits in one frame) ===" << std::endl;
    
    MockSettings settings;
    settings.masterGroupCount = 3;
    settings.masterGroupModuleCounts = {8, 8, 8};
    settings.moduleCount = 8;
    
    MockDisplay display;
    MockMqtt mqtt;
    MockEspNow espnow(settings, display);
    espnow.setMqtt(&mqtt);
    
    // Short message that fits in total modules (3 * 8 = 24)
    std::string message = "HI THERE";  // 8 chars, fits
    
    espnow.distributeMessage(message, false, 
                            settings.getInt("scrollDelayMs"), 
                            settings.getInt("scrollRepeatCount"),
                            true);  // FIX: publishState = true
    
    // Should publish full original message
    bool passed = mqtt.lastPublishedState == message;
    std::cout << "Result: " << (passed ? "PASS" : "FAIL") << std::endl;
    std::cout << "  Original message: '" << message << "'" << std::endl;
    std::cout << "  MQTT published state: '" << mqtt.lastPublishedState << "'" << std::endl;
    return passed;
}

int main() {
    std::cout << "Running Multi-Group MQTT State Publishing Tests" << std::endl;
    std::cout << "=================================================" << std::endl;
    
    int passed = 0;
    int total = 0;
    
    total++; if (testSingleGroupMode()) passed++;
    total++; if (testMultiGroupModeOldBug()) passed++;
    total++; if (testMultiGroupModeWithFix()) passed++;
    total++; if (testShortMessageInMultiGroup()) passed++;
    
    std::cout << "\n=== Summary ===" << std::endl;
    std::cout << "Passed: " << passed << "/" << total << std::endl;
    
    if (passed == total) {
        std::cout << "ALL TESTS PASSED!" << std::endl;
        return 0;
    } else {
        std::cout << "SOME TESTS FAILED!" << std::endl;
        return 1;
    }
}