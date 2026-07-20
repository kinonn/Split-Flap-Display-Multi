#include "SplitFlapEspNow.h"

#include <WiFi.h>
#include <ctype.h>
#include <string.h>

SplitFlapEspNow *SplitFlapEspNow::instance = nullptr;

SplitFlapEspNow::SplitFlapEspNow(JsonSettings &settings, SplitFlapDisplay &display)
    : settings(settings), display(display), pendingMessage(false), lastRemoteText(""), initialized(false) {}

bool SplitFlapEspNow::init() {
    if (initialized) {
        return true;
    }

    if (esp_now_init() != ESP_OK) {
        Serial.println("[esp-now] init failed");
        return false;
    }

    instance = this;
    esp_now_register_recv_cb(handleReceive);
    initialized = true;
    Serial.println("[esp-now] initialized on " + WiFi.macAddress());
    return true;
}

void SplitFlapEspNow::reinit() {
    if (initialized) {
        esp_now_deinit();
        initialized = false;
    }
    init();
}

void SplitFlapEspNow::loop() {
    if (! initialized || ! pendingMessage) {
        return;
    }

    SplitFlapEspNowMessage packet;
    noInterrupts();
    memcpy(&packet, &pendingPacket, sizeof(packet));
    pendingMessage = false;
    interrupts();

    if (packet.version != ESP_NOW_TEXT_VERSION) {
        return;
    }

    int moduleCount = constrain((int) packet.moduleCount, 1, MAX_MODULES);
    packet.text[moduleCount] = '\0';
    String text = String(packet.text);

    if (text == lastRemoteText) {
        return;
    }

    settings.putInt("mode", ESP_NOW_REMOTE_MODE);
    display.writeString(
        text, MAX_RPM, false, DEFAULT_SCROLL_DELAY_MS,
        DEFAULT_SCROLL_REPEAT_COUNT, false
    );
    lastRemoteText = text;
}

bool SplitFlapEspNow::isMasterEnabled() {
    return getGroupCount() > 1;
}

bool SplitFlapEspNow::ensureInitialized() {
    if (initialized) {
        return true;
    }

    if (! isMasterEnabled()) {
        return false;
    }

    return init();
}

void SplitFlapEspNow::setMqtt(SplitFlapMqtt *m) {
    mqtt = m;
}

void SplitFlapEspNow::distributeMessage(
    const String &message, bool centering, unsigned long scrollDelayMs,
    int scrollRepeatCount
) {
    if (! ensureInitialized()) {
        return;
    }

    int totalModuleCount = getTotalModuleCount();

    if (message.length() <= totalModuleCount) {
        distributeFrame(buildFrame(message, totalModuleCount, centering));
    } else {
        int repeats = constrain(
            scrollRepeatCount, MIN_SCROLL_REPEAT_COUNT, MAX_SCROLL_REPEAT_COUNT
        );
        const int maxChunks = MAX_DISPLAY_GROUPS * MAX_MODULES * 4;
        String chunks[maxChunks];
        int chunkCount = 0;
        splitIntoChunks(message, totalModuleCount, chunks, maxChunks, chunkCount);

        Serial.printf(
            "[esp-now scroll] input=%d chars, totalModules=%d, chunks=%d, repeats=%d\n",
            message.length(), totalModuleCount, chunkCount, repeats
        );

        for (int r = 0; r < repeats; r++) {
            for (int i = 0; i < chunkCount; i++) {
                distributeFrame(chunks[i]);
                if (i < chunkCount - 1 || r < repeats - 1) {
                    delay(scrollDelayMs);
                }
            }
        }
    }

    // Publish the FULL original message to the MQTT state topic so external
    // consumers see what was actually requested, not just the local group's
    // slice. Without this, multi-group mode never updates
    // splitflap/{mdns}/state, breaking capture-before-overwrite patterns
    // and HA integrations that read display state.
    if (mqtt && mqtt->isConnected()) {
        mqtt->publishState(message);
    }
}

int SplitFlapEspNow::getGroupCount() {
    return constrain(settings.getInt("masterGroupCount"), 1, MAX_DISPLAY_GROUPS);
}

int SplitFlapEspNow::getGroupModuleCount(int groupIndex) {
    if (groupIndex == 0) {
        return constrain(display.getNumModules(), 1, MAX_MODULES);
    }

    std::vector<int> moduleCounts = settings.getIntVector("masterGroupModuleCounts");
    if (groupIndex >= 0 && groupIndex < (int) moduleCounts.size()) {
        return constrain(moduleCounts[groupIndex], 1, MAX_MODULES);
    }

    return MAX_MODULES;
}

int SplitFlapEspNow::getTotalModuleCount() {
    int total = 0;
    int groupCount = getGroupCount();

    for (int i = 0; i < groupCount; i++) {
        total += getGroupModuleCount(i);
    }

    return constrain(total, 1, MAX_DISPLAY_GROUPS * MAX_MODULES);
}

String SplitFlapEspNow::getGroupMac(int groupIndex) {
    return getCsvToken(settings.getString("masterGroupMacs"), groupIndex);
}

String SplitFlapEspNow::getCsvToken(const String &csv, int index) {
    int tokenStart = 0;
    int tokenIndex = 0;

    for (int i = 0; i <= (int) csv.length(); i++) {
        if (i == (int) csv.length() || csv[i] == ',') {
            if (tokenIndex == index) {
                String token = csv.substring(tokenStart, i);
                token.trim();
                return token;
            }
            tokenStart = i + 1;
            tokenIndex++;
        }
    }

    return "";
}

String SplitFlapEspNow::sliceMessage(const String &message, int start, int width) {
    String segment = "";

    if (start < (int) message.length()) {
        segment = message.substring(start, min(start + width, (int) message.length()));
    }

    while ((int) segment.length() < width) {
        segment += ' ';
    }

    return segment;
}

String SplitFlapEspNow::buildFrame(const String &message, int width, bool centering) {
    String frame = message.substring(0, min((int) message.length(), width));

    if (centering) {
        int totalPadding = width - frame.length();
        int paddingLeft = totalPadding / 2;
        int paddingRight = totalPadding - paddingLeft;

        String padded = "";
        for (int i = 0; i < paddingLeft; i++) {
            padded += ' ';
        }
        padded += frame;
        for (int i = 0; i < paddingRight; i++) {
            padded += ' ';
        }
        return padded;
    }

    while ((int) frame.length() < width) {
        frame += ' ';
    }

    return frame;
}

void SplitFlapEspNow::distributeFrame(const String &frame) {
    int groupCount = getGroupCount();
    int offset = 0;

    int localModuleCount = getGroupModuleCount(0);
    String localText = sliceMessage(frame, offset, localModuleCount);

    // Send peer chunks first so all remote groups get their text before the
    // local group begins displaying.
    for (int groupIndex = 1; groupIndex < groupCount; groupIndex++) {
        int moduleCount = getGroupModuleCount(groupIndex);
        String segment = sliceMessage(frame, offset + localModuleCount, moduleCount);
        Serial.printf("[esp-now] send to group %d text='%s' modules=%d offset=%d\n",
            groupIndex + 1, segment.c_str(), moduleCount, offset + localModuleCount);
        sendToPeer(groupIndex, segment, moduleCount);
        offset += moduleCount;
    }

    Serial.printf("[esp-now] local group 1 display text='%s' modules=%d\n",
        localText.c_str(), localModuleCount);
    display.writeString(
        localText, MAX_RPM, false, DEFAULT_SCROLL_DELAY_MS,
        DEFAULT_SCROLL_REPEAT_COUNT, false
    );
}

void SplitFlapEspNow::splitIntoChunks(
    const String &input, int width, String chunks[], int maxChunks,
    int &outCount
) {
    outCount = 0;

    String s = "";
    bool lastWasSpace = true;
    for (unsigned int i = 0; i < input.length(); i++) {
        char c = input[i];
        bool isSpace = (c == ' ' || c == '\t' || c == '\n' || c == '\r');
        if (isSpace) {
            if (! lastWasSpace) {
                s += ' ';
                lastWasSpace = true;
            }
        } else {
            s += c;
            lastWasSpace = false;
        }
    }
    while (s.length() > 0 && s[s.length() - 1] == ' ') {
        s.remove(s.length() - 1);
    }

    if (s.length() == 0) {
        chunks[0] = buildFrame("", width, false);
        outCount = 1;
        return;
    }

    String current = "";
    int idx = 0;
    while (idx < (int) s.length() && outCount < maxChunks) {
        int wordStart = idx;
        while (wordStart < (int) s.length() && s[wordStart] == ' ') {
            wordStart++;
        }
        if (wordStart >= (int) s.length()) {
            break;
        }

        int wordEnd = wordStart;
        while (wordEnd < (int) s.length() && s[wordEnd] != ' ') {
            wordEnd++;
        }

        int wordLen = wordEnd - wordStart;
        int neededSep = (current.length() > 0) ? 1 : 0;
        int wouldNeed = (int) current.length() + neededSep + wordLen;

        if (wouldNeed <= width) {
            if (neededSep) {
                current += ' ';
            }
            current += s.substring(wordStart, wordEnd);
            idx = wordEnd;
        } else if (current.length() > 0) {
            chunks[outCount++] = buildFrame(current, width, false);
            current = "";
            idx = wordStart;
        } else {
            int remaining = wordLen;
            int pos = wordStart;
            while (remaining > width && outCount < maxChunks) {
                chunks[outCount++] = s.substring(pos, pos + width);
                pos += width;
                remaining -= width;
            }
            current = s.substring(pos, pos + remaining);
            idx = wordEnd;
        }
    }

    if (current.length() > 0 && outCount < maxChunks) {
        chunks[outCount++] = buildFrame(current, width, false);
    }
}

bool SplitFlapEspNow::parseMacAddress(const String &macString, uint8_t mac[6]) {
    char hexDigits[13];
    int digitCount = 0;

    for (int i = 0; i < (int) macString.length(); i++) {
        char c = macString[i];
        if (isxdigit((unsigned char) c)) {
            if (digitCount >= 12) {
                return false;
            }
            hexDigits[digitCount++] = c;
        } else if (c != ':' && c != '-' && c != ' ') {
            return false;
        }
    }

    if (digitCount != 12) {
        return false;
    }
    hexDigits[12] = '\0';

    for (int i = 0; i < 6; i++) {
        char byteText[3] = {hexDigits[i * 2], hexDigits[i * 2 + 1], '\0'};
        mac[i] = (uint8_t) strtol(byteText, nullptr, 16);
    }

    return true;
}

bool SplitFlapEspNow::sendToPeer(int groupIndex, const String &text, int moduleCount) {
    uint8_t mac[6];
    String macString = getGroupMac(groupIndex);

    if (! parseMacAddress(macString, mac)) {
        Serial.println("[esp-now] invalid MAC for group " + String(groupIndex + 1) + ": " + macString);
        return false;
    }

    if (! esp_now_is_peer_exist(mac)) {
        esp_now_peer_info_t peer = {};
        memcpy(peer.peer_addr, mac, 6);
        peer.channel = 0;
        peer.encrypt = false;
        peer.ifidx = (WiFi.getMode() == WIFI_AP) ? WIFI_IF_AP : WIFI_IF_STA;

        Serial.printf("[esp-now] adding peer group %d mac %02X:%02X:%02X:%02X:%02X:%02X ifidx=%d\n",
            groupIndex + 1,
            mac[0], mac[1], mac[2], mac[3], mac[4], mac[5], peer.ifidx);

        esp_err_t addResult = esp_now_add_peer(&peer);
        if (addResult != ESP_OK) {
            Serial.println("[esp-now] failed to add peer for group " + String(groupIndex + 1));
            return false;
        }
    }

    SplitFlapEspNowMessage packet = {};
    packet.version = ESP_NOW_TEXT_VERSION;
    packet.groupIndex = groupIndex;
    packet.moduleCount = constrain(moduleCount, 1, MAX_MODULES);
    memset(packet.text, ' ', sizeof(packet.text));

    int copyLength = min((int) packet.moduleCount, (int) text.length());
    for (int i = 0; i < copyLength; i++) {
        packet.text[i] = text[i];
    }
    packet.text[packet.moduleCount] = '\0';

    Serial.printf("[esp-now] send group %d len=%d text='%s'\n",
        groupIndex + 1, sizeof(packet), packet.text);

    esp_err_t result = esp_now_send(mac, (const uint8_t *) &packet, sizeof(packet));
    if (result != ESP_OK) {
        Serial.println("[esp-now] send failed for group " + String(groupIndex + 1));
        return false;
    }

    return true;
}

void SplitFlapEspNow::queueReceived(const uint8_t *data, int len) {
    if (len != sizeof(SplitFlapEspNowMessage)) {
        Serial.printf("[esp-now] received invalid len=%d\n", len);
        return;
    }

    memcpy(&pendingPacket, data, sizeof(pendingPacket));
    pendingMessage = true;
    Serial.printf("[esp-now] queued received group=%d modules=%d text='%s'\n",
        pendingPacket.groupIndex + 1, pendingPacket.moduleCount, pendingPacket.text);
}

#if defined(ESP_ARDUINO_VERSION_MAJOR) && ESP_ARDUINO_VERSION_MAJOR >= 3
void SplitFlapEspNow::handleReceive(const esp_now_recv_info_t *info, const uint8_t *data, int len) {
    (void) info;
    if (instance) {
        instance->queueReceived(data, len);
    }
}
#else
void SplitFlapEspNow::handleReceive(const uint8_t *mac, const uint8_t *data, int len) {
    (void) mac;
    if (instance) {
        instance->queueReceived(data, len);
    }
}
#endif
