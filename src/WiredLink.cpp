#include "WiredLink.h"

#include "SplitFlapEspNow.h"

#include <string.h>
#include <vector>

// Master polls each remote address this often; remotes reply with their
// module count. Mirrors the ESP-NOW 5 s announce cadence.
#define WIRED_POLL_INTERVAL_MS 5000
// A remote not heard from for this long drops out of the peer list.
#define WIRED_PEER_EXPIRY_MS 30000
#define WIRED_EXPIRY_CHECK_MS 10000
// Log-only ACK timeout: a missing text ACK is reported, never retried.
#define WIRED_ACK_TIMEOUT_MS 2000

SplitFlapWired::SplitFlapWired(JsonSettings &settings, SplitFlapDisplay &display)
    : settings(settings), display(display), initialized(false), transport(LINK_TRANSPORT_ESP_NOW), myAddress(0),
      txPin(4), rxPin(5), baud(115200), lastRemoteText(""), lastPollMs(0), lastExpiryCheckMs(0), offsetDataDirty(false),
      lastOffsetRxMs(0) {
    memset(peers, 0, sizeof(peers));
}

bool SplitFlapWired::init() {
    if (initialized) {
        return true;
    }

    transport = settings.getInt("linkTransport");
    if (transport == LINK_TRANSPORT_ESP_NOW) {
        Serial.println("[wired] init called in ESP-NOW mode; refusing");
        return false;
    }

    myAddress = settings.getInt("linkAddress");
    if (myAddress < 0 || myAddress >= LINK_MAX_GROUPS) {
        Serial.println("[wired] linkAddress must be 0..5");
        return false;
    }
    if (transport == LINK_TRANSPORT_UART && myAddress > 1) {
        Serial.println("[wired] Direct UART supports only addresses 0 (master) and 1 (remote)");
        return false;
    }

    // Only the master validates its group count here: a remote keeps
    // masterGroupCount at 1 (its slot is positional, not derived from the
    // local group count), so gating a remote on the local count would reject
    // every valid remote (linkAddress 1 >= groupCount 1).
    if (myAddress == 0) {
        int groupCount = constrain(settings.getInt("masterGroupCount"), 1, LINK_MAX_GROUPS);
        const char *configError = validateWiredSettings(transport, groupCount);
        if (configError != nullptr) {
            Serial.print("[wired] invalid config: ");
            Serial.println(configError);
            return false;
        }
    }

    txPin = settings.getInt("linkTxPin");
    rxPin = settings.getInt("linkRxPin");
    baud = settings.getInt("linkBaud");
    if (baud < 1200) {
        Serial.println("[wired] linkBaud too low");
        return false;
    }

    Serial1.begin((unsigned long) baud, SERIAL_8N1, rxPin, txPin);
    initialized = true;
    Serial.printf(
        "[wired] initialized transport=%d addr=%d tx=%d rx=%d baud=%ld\n", transport, myAddress, txPin, rxPin, baud
    );
    return true;
}

void SplitFlapWired::loop() {
    if (! initialized) {
        return;
    }

    while (Serial1.available() > 0) {
        if (decoder.feed((uint8_t) Serial1.read())) {
            handleFrame();
        }
    }

    unsigned long now = millis();

    if (myAddress == 0 && isMasterEnabled() && now - lastPollMs >= WIRED_POLL_INTERVAL_MS) {
        for (int addr = 1; addr < getGroupCount(); addr++) {
            sendFrame((uint8_t) addr, LINK_TYPE_ANNOUNCE, nullptr, 0);
        }
        lastPollMs = now;
    }

    if (now - lastExpiryCheckMs >= WIRED_EXPIRY_CHECK_MS) {
        for (int addr = 0; addr < LINK_MAX_GROUPS; addr++) {
            if (peers[addr].seen && now - peers[addr].lastSeenMs > WIRED_PEER_EXPIRY_MS) {
                peers[addr].seen = false;
                peers[addr].ackPending = false;
            }
        }
        lastExpiryCheckMs = now;
    }

    // Log-only ACK expiry: a remote that never confirmed a text frame is
    // reported once, never retried (same fire-and-forget baseline as
    // ESP-NOW, but visible instead of silent).
    if (myAddress == 0) {
        for (int addr = 1; addr < LINK_MAX_GROUPS; addr++) {
            if (peers[addr].ackPending && ! peers[addr].ackLogged &&
                now - peers[addr].ackSentMs > WIRED_ACK_TIMEOUT_MS) {
                peers[addr].ackLogged = true;
                Serial.printf("[wired] no text ack from group %d\n", addr + 1);
            }
        }
    }

    if (offsetDataDirty && now - lastOffsetRxMs >= OFFSET_RELOAD_SETTLE_MS) {
        offsetDataDirty = false;
        Serial.println("[wired] applying pushed offsets to display");
        display.reloadOffsets();
    }
}

bool SplitFlapWired::isMasterEnabled() {
    return myAddress == 0 && getGroupCount() > 1;
}

int SplitFlapWired::getDiscoveredCount() {
    int count = 0;
    for (int addr = 1; addr < LINK_MAX_GROUPS; addr++) {
        if (peers[addr].seen) {
            count++;
        }
    }
    return count;
}

String SplitFlapWired::getDiscoveredPeersJson() {
    // Same shape as the ESP-NOW peer list so the web UI needs no new
    // rendering path; the mac field carries the wired address instead.
    String json = "[";
    bool first = true;
    for (int addr = 1; addr < LINK_MAX_GROUPS; addr++) {
        if (! peers[addr].seen) {
            continue;
        }
        if (! first) {
            json += ",";
        }
        first = false;
        json += "{\"mac\":\"wired:";
        json += String(addr);
        json += "\",\"moduleCount\":";
        json += String(peers[addr].moduleCount);
        json += ",\"assigned\":";
        json += (addr < getGroupCount()) ? "true" : "false";
        json += ",\"lastSeen\":\"";
        json += String(peers[addr].lastSeenMs);
        json += "\"}";
    }
    json += "]";
    return json;
}

int SplitFlapWired::getGroupCount() {
    return constrain(settings.getInt("masterGroupCount"), 1, LINK_MAX_GROUPS);
}

int SplitFlapWired::getGroupModuleCount(int groupIndex) {
    if (groupIndex == 0) {
        return constrain(display.getNumModules(), 1, MAX_MODULES);
    }

    std::vector<int> moduleCounts = settings.getIntVector("masterGroupModuleCounts");
    if (groupIndex >= 0 && groupIndex < (int) moduleCounts.size()) {
        return constrain(moduleCounts[groupIndex], 1, MAX_MODULES);
    }

    return MAX_MODULES;
}

int SplitFlapWired::getTotalModuleCount() {
    int total = 0;
    int groupCount = getGroupCount();

    for (int i = 0; i < groupCount; i++) {
        total += getGroupModuleCount(i);
    }

    return constrain(total, 1, LINK_MAX_GROUPS * MAX_MODULES);
}

String SplitFlapWired::sliceMessage(const String &message, int start, int width) {
    String segment = "";

    if (start < (int) message.length()) {
        segment = message.substring(start, min(start + width, (int) message.length()));
    }

    while ((int) segment.length() < width) {
        segment += ' ';
    }

    return segment;
}

String SplitFlapWired::buildFrame(const String &message, int width, bool centering) {
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

void SplitFlapWired::splitIntoChunks(const String &input, int width, String chunks[], int maxChunks, int &outCount) {
    // Same word-boundary chunking semantics as the ESP-NOW distributor (kept
    // as a local copy: the ESP-NOW class is frozen and may not exist in
    // wired builds of the call graph).
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

void SplitFlapWired::distributeMessage(
    const String &message, bool centering, unsigned long scrollDelayMs, int scrollRepeatCount
) {
    if (! initialized) {
        return;
    }

    int totalModuleCount = getTotalModuleCount();

    if (message.length() <= totalModuleCount) {
        distributeFrame(buildFrame(message, totalModuleCount, centering));
        return;
    }

    int repeats = constrain(scrollRepeatCount, MIN_SCROLL_REPEAT_COUNT, MAX_SCROLL_REPEAT_COUNT);
    const int maxChunks = LINK_MAX_GROUPS * MAX_MODULES * 4;
    // Heap-allocate: same stack rationale as the ESP-NOW comment — 192
    // Strings is too much for the 8 KB loop-task stack.
    std::vector<String> chunks(maxChunks);
    int chunkCount = 0;
    splitIntoChunks(message, totalModuleCount, chunks.data(), maxChunks, chunkCount);

    Serial.printf(
        "[wired scroll] input=%d chars, totalModules=%d, chunks=%d, repeats=%d\n",
        message.length(),
        totalModuleCount,
        chunkCount,
        repeats
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

void SplitFlapWired::distributeFrame(const String &frame) {
    int groupCount = getGroupCount();
    int offset = 0;

    int localModuleCount = getGroupModuleCount(0);
    String localText = sliceMessage(frame, offset, localModuleCount);

    // Send peer chunks first so all remote groups get their text before the
    // local group begins displaying.
    for (int groupIndex = 1; groupIndex < groupCount; groupIndex++) {
        int moduleCount = getGroupModuleCount(groupIndex);
        String segment = sliceMessage(frame, offset + localModuleCount, moduleCount);
        Serial.printf(
            "[wired] send to group %d text='%s' modules=%d offset=%d\n",
            groupIndex + 1,
            segment.c_str(),
            moduleCount,
            offset + localModuleCount
        );
        sendToPeer(groupIndex, segment, moduleCount);
        offset += moduleCount;
    }

    Serial.printf("[wired] local group 1 display text='%s' modules=%d\n", localText.c_str(), localModuleCount);
    display.writeString(localText, MAX_RPM, false, DEFAULT_SCROLL_DELAY_MS, DEFAULT_SCROLL_REPEAT_COUNT, false);
}

bool SplitFlapWired::sendFrame(uint8_t dst, uint8_t type, const void *payload, size_t payloadLen) {
    uint8_t frame[13 + 2 * LINK_MAX_PAYLOAD];
    size_t n = linkEncode(dst, (uint8_t) myAddress, type, (const uint8_t *) payload, payloadLen, frame, sizeof(frame));
    if (n == 0) {
        Serial.println("[wired] frame encode failed");
        return false;
    }
    if (type == LINK_TYPE_TEXT && dst > 0 && dst < LINK_MAX_GROUPS) {
        peers[dst].ackPending = true;
        peers[dst].ackLogged = false;
        peers[dst].ackSentMs = millis();
    }
    return Serial1.write(frame, n) == n;
}

bool SplitFlapWired::sendToPeer(int groupIndex, const String &text, int moduleCount) {
    SplitFlapEspNowMessage packet = {};
    packet.version = ESP_NOW_TEXT_VERSION;
    packet.groupIndex = (uint8_t) groupIndex;
    packet.moduleCount = (uint8_t) constrain(moduleCount, 1, MAX_MODULES);
    memset(packet.text, ' ', sizeof(packet.text));

    int copyLength = min((int) packet.moduleCount, (int) text.length());
    for (int i = 0; i < copyLength; i++) {
        packet.text[i] = text[i];
    }
    packet.text[packet.moduleCount] = '\0';

    return sendFrame((uint8_t) groupIndex, LINK_TYPE_TEXT, &packet, sizeof(packet));
}

void SplitFlapWired::handleFrame() {
    uint8_t dst = decoder.dst();
    uint8_t src = decoder.src();
    uint8_t type = decoder.type();
    const uint8_t *payload = decoder.payload();
    size_t len = decoder.payloadLen();

    // Direction filter on the explicit header addresses. The master only
    // accepts traffic addressed to it (DST=0) from remote addresses; a
    // remote only accepts traffic addressed to its own address. Anything
    // else is RS-485 local echo or another pair's multidrop traffic.
    // Master-originated content (TEXT, offset pushes) must also come FROM
    // the master: a remote never sends to another remote, so a non-zero SRC
    // on such a frame is a misbehaving node or corruption.
    if (myAddress == 0) {
        if (dst != LINK_ADDR_MASTER || src == 0 || src >= LINK_MAX_GROUPS) {
            return;
        }
    } else {
        if (dst != (uint8_t) myAddress) {
            return;
        }
        if ((type == LINK_TYPE_TEXT || type == LINK_TYPE_OFFSETS_PUSH || type == LINK_TYPE_CHAR_OFFSETS_PUSH) &&
            src != LINK_ADDR_MASTER) {
            return;
        }
    }

    switch (type) {
        case LINK_TYPE_ANNOUNCE:
            if (myAddress == 0) {
                processAnnouncement(src, payload, len);
            } else if (src == LINK_ADDR_MASTER) {
                SplitFlapAnnounceMessage reply = {};
                reply.version = ESP_NOW_ANNOUNCE_VERSION;
                reply.moduleCount = (uint8_t) constrain(display.getNumModules(), 1, MAX_MODULES);
                sendFrame(LINK_ADDR_MASTER, LINK_TYPE_ANNOUNCE, &reply, sizeof(reply));
            }
            break;
        case LINK_TYPE_TEXT:
            if (myAddress != 0 && len == sizeof(SplitFlapEspNowMessage)) {
                const SplitFlapEspNowMessage *pkt = (const SplitFlapEspNowMessage *) payload;
                if (pkt->version != ESP_NOW_TEXT_VERSION) {
                    return;
                }
                int moduleCount = constrain((int) pkt->moduleCount, 1, MAX_MODULES);
                char text[MAX_MODULES + 1];
                memcpy(text, pkt->text, moduleCount);
                text[moduleCount] = '\0';
                if (String(text) != lastRemoteText) {
                    settings.putInt("mode", ESP_NOW_REMOTE_MODE);
                    display.writeString(
                        String(text), MAX_RPM, false, DEFAULT_SCROLL_DELAY_MS, DEFAULT_SCROLL_REPEAT_COUNT, false
                    );
                    lastRemoteText = String(text);
                }
                sendFrame(LINK_ADDR_MASTER, LINK_TYPE_ACK, nullptr, 0);
            }
            break;
        case LINK_TYPE_OFFSETS_PUSH:
            if (myAddress != 0) {
                applyOffsetsPush(payload, len);
            }
            break;
        case LINK_TYPE_CHAR_OFFSETS_PUSH:
            if (myAddress != 0) {
                applyCharOffsetsPush(payload, len);
            }
            break;
        case LINK_TYPE_OFFSETS_REPORT:
            if (myAddress == 0) {
                processOffsetsReport(src, payload, len);
            }
            break;
        case LINK_TYPE_CHAR_OFFSETS_REPORT:
            if (myAddress == 0) {
                processCharOffsetsReport(src, payload, len);
            }
            break;
        case LINK_TYPE_ACK:
            if (myAddress == 0) {
                peers[src].ackPending = false;
            }
            break;
        default: Serial.printf("[wired] unknown frame type %d dst %d src %d\n", type, dst, src); break;
    }
}

void SplitFlapWired::processAnnouncement(uint8_t fromAddr, const uint8_t *payload, size_t len) {
    if (len != sizeof(SplitFlapAnnounceMessage)) {
        return;
    }
    const SplitFlapAnnounceMessage *pkt = (const SplitFlapAnnounceMessage *) payload;
    if (pkt->version != ESP_NOW_ANNOUNCE_VERSION) {
        return;
    }
    if (pkt->moduleCount < 1 || pkt->moduleCount > MAX_MODULES) {
        return;
    }
    peers[fromAddr].moduleCount = pkt->moduleCount;
    peers[fromAddr].lastSeenMs = millis();
    if (! peers[fromAddr].seen) {
        peers[fromAddr].seen = true;
        Serial.printf("[wired] discovered addr %d modules=%d\n", fromAddr, pkt->moduleCount);
    }
}

void SplitFlapWired::applyOffsetsPush(const uint8_t *payload, size_t len) {
    if (len != sizeof(SplitFlapOffsetsPushMessage)) {
        return;
    }
    const SplitFlapOffsetsPushMessage *pkt = (const SplitFlapOffsetsPushMessage *) payload;
    if (pkt->version != ESP_NOW_OFFSETS_PUSH) {
        return;
    }
    int moduleCount = constrain((int) pkt->moduleCount, 1, MAX_MODULES);

    std::vector<int> modOffs;
    for (int i = 0; i < 8; i++) {
        modOffs.push_back(pkt->moduleOffsets[i]);
    }
    settings.putIntVector("moduleOffsets", modOffs);
    settings.putInt("displayOffset", pkt->displayOffset);

    Serial.printf("[wired] applying pushed offsets: dispOff=%d modules=%d\n", pkt->displayOffset, moduleCount);
    offsetDataDirty = true;
    lastOffsetRxMs = millis();
}

void SplitFlapWired::applyCharOffsetsPush(const uint8_t *payload, size_t len) {
    if (len != sizeof(SplitFlapCharOffsetsPushMessage)) {
        return;
    }
    const SplitFlapCharOffsetsPushMessage *pkt = (const SplitFlapCharOffsetsPushMessage *) payload;
    if (pkt->version != ESP_NOW_OFFSETS_PUSH) {
        return;
    }
    int modIdx = constrain((int) pkt->moduleIndex, 0, MAX_MODULES - 1);

    auto matrix = settings.getIntMatrix("charOffsets");
    while ((int) matrix.size() <= modIdx) {
        matrix.push_back(std::vector<int>(48, 0));
    }
    if ((int) matrix[modIdx].size() < 48) {
        matrix[modIdx].resize(48, 0);
    }
    for (int c = 0; c < 48; c++) {
        matrix[modIdx][c] = pkt->charOffsets[c];
    }
    settings.putIntMatrix("charOffsets", matrix);

    Serial.printf("[wired] applying pushed char offsets for module %d\n", modIdx);
    offsetDataDirty = true;
    lastOffsetRxMs = millis();
}

void SplitFlapWired::pushOffsetsToGroup(int groupIndex) {
    if (! initialized || myAddress != 0) {
        return;
    }
    if (groupIndex < 1 || groupIndex >= getGroupCount()) {
        return;
    }

    int moduleCount = getGroupModuleCount(groupIndex);
    int row = groupIndex - 1;

    auto modOffs = settings.getIntMatrix("rModOffs");
    int dispOff = 0;
    auto dispOffs = settings.getIntVector("rDispOffs");
    if (row < (int) dispOffs.size()) {
        dispOff = dispOffs[row];
    }

    SplitFlapOffsetsPushMessage pkt = {};
    pkt.version = ESP_NOW_OFFSETS_PUSH;
    pkt.groupIndex = (uint8_t) groupIndex;
    pkt.moduleCount = (uint8_t) moduleCount;
    pkt.displayOffset = (int16_t) constrain(dispOff, -32768, 32767);
    for (int i = 0; i < 8; i++) {
        int val = (row < (int) modOffs.size() && i < (int) modOffs[row].size()) ? modOffs[row][i] : 0;
        pkt.moduleOffsets[i] = (int16_t) constrain(val, -32768, 32767);
    }
    if (! sendFrame((uint8_t) groupIndex, LINK_TYPE_OFFSETS_PUSH, &pkt, sizeof(pkt))) {
        Serial.printf("[wired] push offsets send failed for group %d\n", groupIndex + 1);
    }
    delay(OFFSET_PACKET_SPACING_MS);

    String key = "rChrOff" + String(row);
    auto chrOffs = settings.getIntMatrix(key.c_str());
    for (int m = 0; m < moduleCount; m++) {
        SplitFlapCharOffsetsPushMessage cpkt = {};
        cpkt.version = ESP_NOW_OFFSETS_PUSH;
        cpkt.groupIndex = (uint8_t) groupIndex;
        cpkt.moduleIndex = (uint8_t) m;
        for (int c = 0; c < 48; c++) {
            int val = (m < (int) chrOffs.size() && c < (int) chrOffs[m].size()) ? chrOffs[m][c] : 0;
            cpkt.charOffsets[c] = constrain(val, -32, 32);
        }
        if (! sendFrame((uint8_t) groupIndex, LINK_TYPE_CHAR_OFFSETS_PUSH, &cpkt, sizeof(cpkt))) {
            Serial.printf("[wired] push char offsets send failed for group %d module %d\n", groupIndex + 1, m);
        }
        if (m < moduleCount - 1) {
            delay(OFFSET_PACKET_SPACING_MS);
        }
    }

    Serial.printf("[wired] pushed offsets to group %d\n", groupIndex + 1);
}

void SplitFlapWired::reportOffsetsToMaster() {
    if (! initialized || myAddress == 0) {
        return;
    }

    int moduleCount = display.getNumModules();
    int dispOff = settings.getInt("displayOffset");
    auto modOffs = settings.getIntVector("moduleOffsets");

    SplitFlapOffsetsReportMessage pkt = {};
    pkt.version = ESP_NOW_OFFSETS_REPORT;
    pkt.moduleCount = (uint8_t) moduleCount;
    pkt.displayOffset = (int16_t) constrain(dispOff, -32768, 32767);
    for (int i = 0; i < 8; i++) {
        int val = (i < (int) modOffs.size()) ? modOffs[i] : 0;
        pkt.moduleOffsets[i] = (int16_t) constrain(val, -32768, 32767);
    }
    if (! sendFrame(LINK_ADDR_MASTER, LINK_TYPE_OFFSETS_REPORT, &pkt, sizeof(pkt))) {
        Serial.println("[wired] report offsets send failed");
    }
    delay(OFFSET_PACKET_SPACING_MS);

    auto chrOffs = settings.getIntMatrix("charOffsets");
    for (int m = 0; m < moduleCount; m++) {
        SplitFlapCharOffsetsReportMessage cpkt = {};
        cpkt.version = ESP_NOW_OFFSETS_REPORT;
        cpkt.moduleIndex = (uint8_t) m;
        for (int c = 0; c < 48; c++) {
            int val = (m < (int) chrOffs.size() && c < (int) chrOffs[m].size()) ? chrOffs[m][c] : 0;
            cpkt.charOffsets[c] = constrain(val, -32, 32);
        }
        if (! sendFrame(LINK_ADDR_MASTER, LINK_TYPE_CHAR_OFFSETS_REPORT, &cpkt, sizeof(cpkt))) {
            Serial.printf("[wired] report char offsets send failed for module %d\n", m);
        }
        if (m < moduleCount - 1) {
            delay(OFFSET_PACKET_SPACING_MS);
        }
    }

    Serial.println("[wired] reported offsets to master");
}

void SplitFlapWired::processOffsetsReport(uint8_t fromAddr, const uint8_t *payload, size_t len) {
    if (len != sizeof(SplitFlapOffsetsReportMessage)) {
        return;
    }
    const SplitFlapOffsetsReportMessage *pkt = (const SplitFlapOffsetsReportMessage *) payload;
    if (pkt->version != ESP_NOW_OFFSETS_REPORT) {
        return;
    }
    if (fromAddr < 1 || fromAddr >= getGroupCount()) {
        return;
    }
    int row = fromAddr - 1;

    int moduleCount = constrain((int) pkt->moduleCount, 1, MAX_MODULES);

    auto modOffs = settings.getIntMatrix("rModOffs");
    while ((int) modOffs.size() <= row) {
        modOffs.push_back(std::vector<int>(8, 0));
    }
    if ((int) modOffs[row].size() < 8) {
        modOffs[row].resize(8, 0);
    }
    for (int i = 0; i < 8; i++) {
        modOffs[row][i] = (i < moduleCount) ? pkt->moduleOffsets[i] : 0;
    }
    settings.putIntMatrix("rModOffs", modOffs);

    auto dispOffs = settings.getIntVector("rDispOffs");
    while ((int) dispOffs.size() <= row) {
        dispOffs.push_back(0);
    }
    dispOffs[row] = pkt->displayOffset;
    settings.putIntVector("rDispOffs", dispOffs);

    Serial.printf("[wired] received offset report from group %d\n", fromAddr + 1);
}

void SplitFlapWired::processCharOffsetsReport(uint8_t fromAddr, const uint8_t *payload, size_t len) {
    if (len != sizeof(SplitFlapCharOffsetsReportMessage)) {
        return;
    }
    const SplitFlapCharOffsetsReportMessage *pkt = (const SplitFlapCharOffsetsReportMessage *) payload;
    if (pkt->version != ESP_NOW_OFFSETS_REPORT) {
        return;
    }
    if (fromAddr < 1 || fromAddr >= getGroupCount()) {
        return;
    }
    int row = fromAddr - 1;

    String key = "rChrOff" + String(row);
    auto matrix = settings.getIntMatrix(key.c_str());
    int modIdx = constrain((int) pkt->moduleIndex, 0, MAX_MODULES - 1);

    while ((int) matrix.size() <= modIdx) {
        matrix.push_back(std::vector<int>(48, 0));
    }
    if ((int) matrix[modIdx].size() < 48) {
        matrix[modIdx].resize(48, 0);
    }
    for (int c = 0; c < 48; c++) {
        matrix[modIdx][c] = pkt->charOffsets[c];
    }
    settings.putIntMatrix(key.c_str(), matrix);

    Serial.printf("[wired] received char offset report from group %d module %d\n", fromAddr + 1, modIdx);
}
