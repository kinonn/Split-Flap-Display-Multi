// Host-side regression test: multi-group message chunking/distribution must
// keep its semantics when a group spans 16 modules (dual-I2C build).
//
// The production algorithms live in src/SplitFlapEspNow.cpp:
//   - sliceMessage()        (lines ~265)
//   - buildFrame()          (lines ~279)
//   - splitIntoChunks()     (lines ~331)
//   - distributeFrame()     (lines ~305)
//   - sendToPeer() text buffer sizing (lines ~466)
//
// These cannot compile on the host (Arduino String, JsonSettings, ESP-NOW),
// so — following the repo's established pattern (motor_scheduler_test.cpp,
// multi_group_mqtt_state_test.cpp) — this file mirrors the production logic
// line-for-line in pure C++, then exercises it at width 16. If the firmware
// algorithm changes, this mirror must be updated in the same change.
//
//   g++ -std=c++11 -Wall -Wextra test/dual_i2c_wide_group_test.cpp -o /tmp/dual_i2c_wide_group_test && /tmp/dual_i2c_wide_group_test

#include <iostream>
#include <string>
#include <vector>
#include <cstring>

static int testCount = 0;
static int testFailures = 0;

#define CHECK(cond, msg)                                        \
    do {                                                        \
        testCount++;                                            \
        if (!(cond)) {                                          \
            testFailures++;                                     \
            std::cout << "FAIL: " << msg << "  [" << #cond << "]" << std::endl; \
        }                                                       \
    } while (0)

// ---- Mirrors of production code (src/SplitFlapEspNow.cpp) ----

static std::string sliceMessage(const std::string &message, int start, int width) {
    std::string segment;
    if (start < (int) message.length()) {
        segment = message.substr(start, std::min(start + width, (int) message.length()) - start);
    }
    while ((int) segment.length() < width) {
        segment += ' ';
    }
    return segment;
}

static std::string buildFrame(const std::string &message, int width, bool centering) {
    std::string frame = message.substr(0, std::min((int) message.length(), width));

    if (centering) {
        int totalPadding = width - (int) frame.length();
        int paddingLeft = totalPadding / 2;
        int paddingRight = totalPadding - paddingLeft;
        return std::string(paddingLeft, ' ') + frame + std::string(paddingRight, ' ');
    }

    while ((int) frame.length() < width) {
        frame += ' ';
    }
    return frame;
}

static void splitIntoChunks(const std::string &input, int width, std::string chunks[], int maxChunks, int &outCount) {
    outCount = 0;

    // Collapse runs of whitespace into single spaces.
    std::string s;
    bool lastWasSpace = true;
    for (size_t i = 0; i < input.length(); i++) {
        char c = input[i];
        bool isSpace = (c == ' ' || c == '\t' || c == '\n' || c == '\r');
        if (isSpace) {
            if (!lastWasSpace) {
                s += ' ';
                lastWasSpace = true;
            }
        } else {
            s += c;
            lastWasSpace = false;
        }
    }
    while (s.length() > 0 && s[s.length() - 1] == ' ') {
        s.erase(s.length() - 1);
    }

    if (s.length() == 0) {
        chunks[0] = buildFrame("", width, false);
        outCount = 1;
        return;
    }

    std::string current;
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
            current += s.substr(wordStart, wordEnd - wordStart);
            idx = wordEnd;
        } else if (current.length() > 0) {
            chunks[outCount++] = buildFrame(current, width, false);
            current = "";
            idx = wordStart;
        } else {
            int remaining = wordLen;
            int pos = wordStart;
            while (remaining > width && outCount < maxChunks) {
                chunks[outCount++] = s.substr(pos, width);
                pos += width;
                remaining -= width;
            }
            current = s.substr(pos, remaining);
            idx = wordEnd;
        }
    }
    if (current.length() > 0 && outCount < maxChunks) {
        chunks[outCount++] = buildFrame(current, width, false);
    }
}

// Mirrors distributeFrame(): frame is sliced across groups left-to-right,
// remote groups are sent first, local group (index 0) displayed last.
struct DistributedGroup {
    int groupIndex;
    std::string text;
};

static void distributeFrame(const std::string &frame,
                            const std::vector<int> &groupModuleCounts,
                            std::vector<DistributedGroup> &out) {
    int offset = 0;
    int localModuleCount = groupModuleCounts[0];
    std::string localText = sliceMessage(frame, offset, localModuleCount);

    for (int groupIndex = 1; groupIndex < (int) groupModuleCounts.size(); groupIndex++) {
        int moduleCount = groupModuleCounts[groupIndex];
        std::string segment = sliceMessage(frame, offset + localModuleCount, moduleCount);
        out.push_back({groupIndex, segment});
        offset += moduleCount;
    }
    out.push_back({0, localText});
}

// Mirrors sendToPeer() buffer handling: text is padded to moduleCount with
// spaces and NUL-terminated at moduleCount; the wire buffer is text[17]
// (16 chars + NUL) in the dual-I2C build.
static bool fitsWireBuffer(int moduleCount, const std::string &text) {
    const int textBufSize = 17; // char text[17]
    std::string padded = text;
    while ((int) padded.length() < moduleCount) {
        padded += ' ';
    }
    if ((int) padded.length() >= textBufSize) return false; // NUL would not fit
    if (padded.length() != (size_t) moduleCount) return false;
    char buf[textBufSize];
    std::strncpy(buf, padded.c_str(), textBufSize - 1);
    buf[textBufSize - 1] = '\0';
    return std::strlen(buf) == (size_t) moduleCount && buf[moduleCount] == '\0';
}

// ---- Tests ----

void testWideGroupDistribution() {
    std::cout << "\n[TEST] 16-module local group + 8-module remote groups" << std::endl;
    // 3 groups: local 16, remote 8, remote 8 -> total width 32
    std::vector<int> counts = {16, 8, 8};

    // A 17-char message: first 16 chars go to the local group, the rest
    // continues into the remote groups (multi-group left-to-right semantics
    // must survive the 16-wide local group).
    std::string frame = buildFrame("THE QUICK BROWN FOX JUMPS", 32, false);

    std::vector<DistributedGroup> out;
    distributeFrame(frame, counts, out);
    // Remote groups are distributed first (order in `out`: group1, group2, local).
    CHECK(out.size() == 3, "three groups distributed");
    CHECK(out[0].groupIndex == 1, "remote group 2 sent first");
    CHECK(out[1].groupIndex == 2, "remote group 3 sent second");
    CHECK(out[2].groupIndex == 0, "local group displayed last");

    // frame = "THE QUICK BROWN FOX JUMPS       " (32 chars)
    // chars 0-15  -> local (16-wide): "THE QUICK BROWN " (trailing space)
    // chars 16-23 -> group2 (8-wide):  "FOX JUMP"
    // chars 24-31 -> group3 (8-wide):  "S       "
    CHECK(out[0].text == "FOX JUMP", "group2 gets chars 16-23");
    CHECK(out[1].text == "S       ", "group3 gets chars 24-31 (padded to 8)");
    CHECK(out[2].text == "THE QUICK BROWN ", "local 16-wide group gets the first 16 chars");
}

void testWordWrapAtWidth16() {
    std::cout << "\n[TEST] word-aligned chunking at width 16" << std::endl;
    std::string chunks[32];
    int count = 0;

    splitIntoChunks("HELLO WORLD THIS IS A LONG MESSAGE", 16, chunks, 32, count);
    CHECK(count == 3, "three chunks for 37-char message at width 16");
    // Word boundaries: "HELLO WORLD THIS" is exactly 16 chars, so "IS" starts
    // the next chunk ("IS A LONG" = 9 chars, "MESSAGE" can't fit alongside).
    CHECK(chunks[0] == "HELLO WORLD THIS", "first chunk fills the 16-wide group exactly");
    CHECK(chunks[1] == "IS A LONG       ", "second chunk word-aligned, padded to 16");
    CHECK(chunks[2] == "MESSAGE         ", "third chunk is the last word, padded to 16");

    // No chunk may split a word that fits.
    for (int i = 0; i < count; i++) {
        std::string t = chunks[i];
        while (!t.empty() && t[t.length() - 1] == ' ') t.erase(t.length() - 1);
        CHECK((int) t.length() <= 16, "chunk fits the group width");
    }
}

void testSingleWordLongerThan16() {
    std::cout << "\n[TEST] single word longer than width 16 hard-splits" << std::endl;
    std::string chunks[32];
    int count = 0;

    splitIntoChunks("SUPERCALIFRAGILISTICEXPIALIDOCIOUS", 16, chunks, 32, count);
    CHECK(count == 3, "34-char word splits into 16+16+2");
    CHECK(chunks[0] == "SUPERCALIFRAGILI", "first hard fragment 16 chars");
    CHECK(chunks[1] == "STICEXPIALIDOCIO", "second hard fragment 16 chars");
    CHECK(chunks[2] == "US              ", "last fragment 2 chars, padded to width 16");
}

void testCenteringAtWidth16() {
    std::cout << "\n[TEST] centering at width 16" << std::endl;
    std::string centered = buildFrame("AB", 16, true);
    CHECK(centered == "       AB       ", "7 left / 7 right padding for 2 chars in 16");

    centered = buildFrame("ABC", 16, true);
    CHECK(centered == "      ABC       ", "integer division puts the extra space right (6 left / 7 right)");
}

void testWireBufferFits16() {
    std::cout << "\n[TEST] text[17] wire buffer holds a 16-module chunk" << std::endl;
    // 16-char chunk, the largest the protocol can carry.
    CHECK(fitsWireBuffer(16, "THE QUICK BROWN"), "16 chars + NUL fit in text[17]");
    CHECK(fitsWireBuffer(16, "0123456789ABCDEF"), "16 chars exactly");
    CHECK(fitsWireBuffer(8, "HELLO"), "8-module groups still fit (backwards shape)");
    CHECK(!fitsWireBuffer(17, "THIS IS 17 CHARS"), "17 chars cannot be sent (would overflow)");
}

int main() {
    std::cout << "=== Dual-I2C 16-module group chunking tests ===" << std::endl;
    testWideGroupDistribution();
    testWordWrapAtWidth16();
    testSingleWordLongerThan16();
    testCenteringAtWidth16();
    testWireBufferFits16();

    std::cout << "\n=== Summary ===" << std::endl;
    std::cout << "Passed: " << (testCount - testFailures) << "/" << testCount << std::endl;
    if (testFailures == 0) {
        std::cout << "ALL TESTS PASSED" << std::endl;
        return 0;
    }
    std::cout << testFailures << " TEST(S) FAILED" << std::endl;
    return 1;
}
