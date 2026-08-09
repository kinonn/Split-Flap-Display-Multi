#include "SplitFlapDisplay.h"

#include "JsonSettings.h"
#include "SplitFlapModule.h"
#include "SplitFlapMqtt.h"
#include "SplitFlapMotorScheduler.h"

// ESP32 pins that must never be used as I2C. GPIO 6-11 are the SPI flash pins
// (driving them hangs the chip and trips the watchdog during boot), and the
// rest are boot strapping pins. Configuring any of these as SDA/SCL makes the
// device boot-loop, so we reject them up front and skip I2C entirely.
static bool isValidI2CPin(int pin) {
    if (pin < 0 || pin > 39) return false;
    switch (pin) {
        case 0:    // strapping (boot mode)
        case 2:    // strapping
        case 5:    // strapping
        case 6:    // SPI flash
        case 7:    // SPI flash
        case 8:    // SPI flash / VDD_SDIO
        case 9:    // SPI flash / boot
        case 10:   // SPI flash
        case 11:   // SPI flash
        case 12:   // strapping (MTDI / VDD_SDIO voltage)
        case 15:   // strapping (MTDO)
            return false;
        default:
            return true;
    }
}

SplitFlapDisplay::SplitFlapDisplay(JsonSettings &settings) : settings(settings), maxConcurrentMotors(-1) {
    for (int i = 0; i < MAX_MODULES; i++) {
        lastDisplayedChar[i] = ' ';
    }
}

void SplitFlapDisplay::init() {
    numModules = constrain(settings.getInt("moduleCount"), 1, MAX_MODULES);
    stepsPerRot = settings.getInt("stepsPerRot");
    displayOffset = settings.getInt("displayOffset");
    magnetPosition = settings.getInt("magnetPosition");
    maxVel = settings.getFloat("maxVel");
    charSetSize = settings.getInt("charset");

    // Modules 0..7 live on bus 1 (moduleAddresses/moduleOffsets, backwards
    // compatible with existing single-bus configs). On dual-I2C builds,
    // modules 8..15 live on bus 2 (wire1Addresses/wire1Offsets).
    int bus1Count = min(numModules, 8);

    std::vector<int> settingAddresses = settings.getIntVector("moduleAddresses");
    std::vector<int> settingOffsets = settings.getIntVector("moduleOffsets");
#ifdef ENABLE_DUAL_I2C
    std::vector<int> settingWire1Addresses = settings.getIntVector("wire1Addresses");
    std::vector<int> settingWire1Offsets = settings.getIntVector("wire1Offsets");
    wire1Count = numModules - bus1Count;
#endif

    for (int i = 0; i < numModules; i++) {
        if (i < bus1Count) {
            moduleAddresses[i] = (i < (int) settingAddresses.size()) ? (uint8_t) settingAddresses[i] : (uint8_t) (0x20 + i);
            moduleOffsets[i] = (i < (int) settingOffsets.size()) ? settingOffsets[i] : 0;
        } else {
#ifdef ENABLE_DUAL_I2C
            int j = i - bus1Count;
            moduleAddresses[i] = (j < (int) settingWire1Addresses.size()) ? (uint8_t) settingWire1Addresses[j] : (uint8_t) (0x20 + j);
            moduleOffsets[i] = (j < (int) settingWire1Offsets.size()) ? settingWire1Offsets[j] : 0;
#else
            moduleAddresses[i] = 0x20 + i; // unreachable: bus1Count == numModules
            moduleOffsets[i] = 0;
#endif
        }
    }

    std::vector<std::vector<int>> settingCharOffsets = settings.getIntMatrix("charOffsets");
    for (int i = 0; i < numModules; i++) {
        for (int j = 0; j < 48; j++) {
            charOffsets[i][j] = (i < (int)settingCharOffsets.size() && j < (int)settingCharOffsets[i].size())
                ? settingCharOffsets[i][j] : 0;
        }
    }

    Serial.print("Module Offsets: ");
    for (int i = 0; i < numModules; i++) {
        Serial.print(moduleOffsets[i]);
        Serial.print(" ");
    }
    Serial.println();

    for (uint8_t i = 0; i < numModules; i++) {
        TwoWire *bus = &Wire;
#ifdef ENABLE_DUAL_I2C
        if (i >= bus1Count) {
            bus = &Wire1;
        }
#endif
        modules[i] = SplitFlapModule(
            moduleAddresses[i], stepsPerRot, moduleOffsets[i] + displayOffset, magnetPosition, charSetSize, charOffsets[i], bus
        );
    }

    SDAPin = settings.getInt("sdaPin");
    SCLPin = settings.getInt("sclPin");

    // Reject reserved/strapping pins before touching the I2C hardware. Calling
    // Wire.begin() with e.g. GPIO 8/9 (SPI flash pins) hangs the chip and trips
    // the watchdog, boot-looping the device. If the configured pins are bad we
    // skip I2C entirely so the board still boots (WiFi/AP/web server stay up).
    if (!isValidI2CPin(SDAPin) || !isValidI2CPin(SCLPin)) {
        Serial.printf("[i2c] WARNING: invalid I2C pins SDA=%d SCL=%d (reserved/strapping), skipping I2C init\n", SDAPin, SCLPin);
        Serial.printf("[i2c] initialized 0/%d modules\n", numModules);
        Serial.flush();
        return;
    }

    Wire.begin(SDAPin, SCLPin);
    Wire.setClock(400000);
    // Bound each I2C transaction so a stuck bus or a missing module can't
    // block setup() indefinitely (which trips the watchdog and boot-loops).
    Wire.setTimeOut(50);

#ifdef ENABLE_DUAL_I2C
    // Always bring up bus 2 (Wire1), even when wire1Count is 0, so we can
    // power down any expanders physically present on it below.
    bool wire1Active = false;
    SDA2Pin = settings.getInt("sda2Pin");
    SCL2Pin = settings.getInt("scl2Pin");
    if (!isValidI2CPin(SDA2Pin) || !isValidI2CPin(SCL2Pin)) {
        Serial.printf("[i2c] WARNING: invalid Wire1 I2C pins SDA2=%d SCL2=%d (reserved/strapping), skipping bus 2\n", SDA2Pin, SCL2Pin);
    } else {
        Wire1.begin(SDA2Pin, SCL2Pin);
        Wire1.setClock(400000);
        Wire1.setTimeOut(50);
        wire1Active = true;
        Serial.printf("[i2c] bus 2 (Wire1) enabled: SDA=%d SCL=%d modules=%d\n", SDA2Pin, SCL2Pin, wire1Count);
    }
    Serial.flush();
#endif

    // Power down all motor coils on EVERY possible module address of BOTH
    // buses before probing/initializing the configured modules, regardless
    // of how many modules are configured. PCF8575 expanders power up with
    // all outputs HIGH and the ULN2003 boards invert that, so any expander
    // we never write to leaves every coil of its module energized at boot.
    // On dual-I2C builds that includes all 8 addresses on bus 2 even when
    // wire1Count is 0. Writing an address with no device just NACKs and
    // returns immediately (only a stuck bus hits the 50ms timeout).
    powerDownCoils(&Wire);
#ifdef ENABLE_DUAL_I2C
    if (wire1Active) {
        powerDownCoils(&Wire1);
    }
    Serial.println("[i2c] all module coils powered down (both buses)");
#else
    Serial.println("[i2c] all module coils powered down");
#endif
    Serial.flush();

    // Probe each configured address and only initialize modules that are
    // actually present. A missing module would otherwise block every write
    // until the I2C timeout — 16 transactions with no modules on the bus is
    // enough to trip the watchdog during boot. isPresent() also marks absent
    // modules as errored so later writes/reads short-circuit instantly.
    int detected = 0;
    for (uint8_t i = 0; i < numModules; i++) {
        if (modules[i].isPresent()) {
            modules[i].init();
            detected++;
        } else {
            Serial.printf("[i2c] module %d (0x%02X) not detected, skipping\n", i, moduleAddresses[i]);
        }
    }
    Serial.printf("[i2c] initialized %d/%d modules\n", detected, numModules);
    Serial.flush();
}

void SplitFlapDisplay::reloadOffsets() {
    int oldDisplayOffset = displayOffset;
    int oldModuleOffsets[MAX_MODULES];
    int oldCharOffsets[MAX_MODULES][48];
    for (int i = 0; i < numModules; i++) {
        oldModuleOffsets[i] = moduleOffsets[i];
        for (int j = 0; j < 48; j++) {
            oldCharOffsets[i][j] = charOffsets[i][j];
        }
    }

    displayOffset = settings.getInt("displayOffset");

    std::vector<int> settingOffsets = settings.getIntVector("moduleOffsets");
#ifdef ENABLE_DUAL_I2C
    std::vector<int> settingWire1Offsets = settings.getIntVector("wire1Offsets");
#endif
    int bus1Count = min(numModules, 8);
    for (int i = 0; i < numModules; i++) {
        if (i < bus1Count) {
            moduleOffsets[i] = (i < (int) settingOffsets.size()) ? settingOffsets[i] : 0;
        } else {
#ifdef ENABLE_DUAL_I2C
            int j = i - bus1Count;
            moduleOffsets[i] = (j < (int) settingWire1Offsets.size()) ? settingWire1Offsets[j] : 0;
#else
            moduleOffsets[i] = 0;
#endif
        }
    }

    std::vector<std::vector<int>> settingCharOffsets = settings.getIntMatrix("charOffsets");
    for (int i = 0; i < numModules; i++) {
        for (int j = 0; j < 48; j++) {
            charOffsets[i][j] = (i < (int)settingCharOffsets.size() && j < (int)settingCharOffsets[i].size())
                ? settingCharOffsets[i][j] : 0;
        }
    }

    bool affected[MAX_MODULES] = {};
    bool anyAffected = false;

    if (oldDisplayOffset != displayOffset) {
        for (int i = 0; i < numModules; i++) {
            affected[i] = true;
        }
        anyAffected = true;
    } else {
        for (int i = 0; i < numModules; i++) {
            bool moduleChanged = (oldModuleOffsets[i] != moduleOffsets[i]);
            if (! moduleChanged) {
                for (int j = 0; j < 48; j++) {
                    if (oldCharOffsets[i][j] != charOffsets[i][j]) {
                        moduleChanged = true;
                        break;
                    }
                }
            }
            affected[i] = moduleChanged;
            if (moduleChanged) {
                anyAffected = true;
            }
        }
    }

    for (uint8_t i = 0; i < numModules; i++) {
        int newMagnetOffset = magnetPosition + moduleOffsets[i] + displayOffset;
        modules[i].updateOffsets(charOffsets[i], newMagnetOffset);
    }

    if (anyAffected) {
        homeAffectedModules(affected);
    }
}

void SplitFlapDisplay::homeAffectedModules(bool affected[], float speed) {
    Serial.println("Homing affected modules");
    maxConcurrentMotors = constrain(settings.getInt("maxConcurrentMotors"), 1, MAX_MODULES);
    int targetPositions[numModules];
    for (int i = 0; i < numModules; i++) {
        if (affected[i]) {
            targetPositions[i] = (modules[i].getPosition() - 1 + stepsPerRot) % stepsPerRot;
        } else {
            targetPositions[i] = modules[i].getPosition();
        }
    }
    moveTo(targetPositions, speed, false);

    for (int i = 0; i < numModules; i++) {
        if (affected[i]) {
            targetPositions[i] = modules[i].getCharPosition(lastDisplayedChar[i]);
        } else {
            targetPositions[i] = modules[i].getPosition();
        }
    }
    moveTo(targetPositions, speed);
}

void SplitFlapDisplay::testAll() {
    maxConcurrentMotors = -1; // unlimited — test/display moves are short
    char testChars[37] = {' ', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
                          'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'};
    int numChars = sizeof(testChars) / sizeof(testChars[0]);
    int targetPositions[numModules];

    int charPos;
    for (int i = 0; i < numChars; i++) {
        // Serial.print("Target Positions: [");
        // fill array with same char

        for (int j = 0; j < numModules; j++) {
            targetPositions[j] = modules[j].getCharPosition(testChars[i]);
            // Serial.print(targetPositions[j]);
            // Serial.print(" , ");
        }
        // Serial.println("]");

        moveTo(targetPositions);
        delay(500);
    }
}

void SplitFlapDisplay::testRandom(float speed) {
    maxConcurrentMotors = -1; // unlimited — test/display moves are short
    char testChars[37] = {' ', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
                          'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'};

    int targetPositions[numModules];
    char randChar;

    Serial.print("Target: ");
    for (int i = 0; i < numModules; i++) {
        randChar = testChars[random(0, 37)];
        targetPositions[i] = modules[i].getCharPosition(randChar);
        Serial.print(randChar);
    }
    Serial.println(" ");
    moveTo(targetPositions, speed);
}

void SplitFlapDisplay::testCount() {
    maxConcurrentMotors = -1; // unlimited — test/display moves are short
    int count = 0;
    int maxCount = pow(10, numModules);
    char targetChar;
    int targetInteger;

    int targetPositions[numModules];

    for (int i = 0; i < maxCount; i++) {
        // get each character in the count integer
        for (int j = 0; j < numModules; j++) {
            targetInteger = (i % (int) pow(10, j + 1)) / (int) pow(10, j);
            targetChar = targetInteger + '0'; // convert to char
            targetPositions[numModules - j - 1] = modules[j].getCharPosition(targetChar);
        }

        moveTo(targetPositions);
        delay(250);
    }
}

void SplitFlapDisplay::home(float speed) {
    Serial.println("Homing");
    maxConcurrentMotors = constrain(settings.getInt("maxConcurrentMotors"), 1, MAX_MODULES);
    int targetPositions[numModules];
    for (int i = 0; i < numModules; i++) {
        targetPositions[i] = (modules[i].getPosition() - 1 + stepsPerRot) % stepsPerRot;
    }
    moveTo(targetPositions, speed, false);
    char homeChar = ' ';
    int charPosition;
    for (int i = 0; i < numModules; i++) {
        targetPositions[i] = modules[i].getCharPosition(homeChar);
        lastDisplayedChar[i] = homeChar;
    }
    moveTo(targetPositions, speed);
}

void SplitFlapDisplay::homeToString(String homeString, float speed, bool centering) {
    Serial.println("Homing");
    maxConcurrentMotors = constrain(settings.getInt("maxConcurrentMotors"), 1, MAX_MODULES);
    int targetPositions[numModules];
    for (int i = 0; i < numModules; i++) {
        targetPositions[i] = (modules[i].getPosition() - 1 + stepsPerRot) % stepsPerRot;
    }
    moveTo(targetPositions, speed, false);
    // Reposition straight to the target string instead of going through
    // writeString(): writeString resets maxConcurrentMotors to unlimited,
    // which would defeat the homing concurrency cap on the reposition move.
    displayChunk(homeString, speed, centering);
}

void SplitFlapDisplay::homeToChar(char homeChar, float speed) {
    Serial.println("Homing");
    maxConcurrentMotors = constrain(settings.getInt("maxConcurrentMotors"), 1, MAX_MODULES);
    int targetPositions[numModules];
    for (int i = 0; i < numModules; i++) {
        targetPositions[i] = (modules[i].getPosition() - 1 + stepsPerRot) % stepsPerRot;
    }
    moveTo(targetPositions, speed, false);

    for (int i = 0; i < numModules; i++) {
        targetPositions[i] = modules[i].getCharPosition(homeChar);
        lastDisplayedChar[i] = homeChar;
    }
    moveTo(targetPositions, speed, true);
}

void SplitFlapDisplay::writeChar(char inputChar, float speed) {
    maxConcurrentMotors = -1; // unlimited — normal display writes stay fast
    int targetPositions[numModules];
    // Iterate through the input string and process each character
    for (int i = 0; i < numModules; i++) {
        targetPositions[i] = modules[i].getCharPosition(inputChar);
        lastDisplayedChar[i] = inputChar;
    }
    moveTo(targetPositions, speed);
}

void SplitFlapDisplay::writeString(
    String inputString, float speed, bool centering, unsigned long scrollDelayMs,
    int scrollRepeatCount, bool publishState
) {
    // Normal display writes are not concurrency-capped (homing is).
    maxConcurrentMotors = -1;
    // Short path: fits in one frame — behave exactly as before.
    if (inputString.length() <= numModules) {
        displayChunk(inputString, speed, centering);
        if (publishState && mqtt && mqtt->isConnected()) {
            mqtt->publishState(inputString);
        }
        return;
    }

    // Long path: split into word-respecting chunks and display sequentially,
    // repeating the full chunk sequence scrollRepeatCount times end-to-end.
    // Clamp the count to a sane range so a misconfigured value (0, negative,
    // or absurdly large) can't lock up the display loop.
    int repeats = constrain(
        scrollRepeatCount, MIN_SCROLL_REPEAT_COUNT, MAX_SCROLL_REPEAT_COUNT
    );

    std::vector<String> chunks(MAX_MODULES * 4); // generous upper bound for very long input
    int chunkCount = 0;
    splitIntoChunks(inputString, chunks.data(), MAX_MODULES * 4, chunkCount);

    Serial.printf(
        "[scroll] input=%d chars, numModules=%d, chunks=%d, repeats=%d\n",
        inputString.length(), numModules, chunkCount, repeats
    );

    for (int r = 0; r < repeats; r++) {
        for (int i = 0; i < chunkCount; i++) {
            displayChunk(chunks[i], speed, centering);
            // Pause between chunks within a pass. We also pause between the
            // last chunk of pass r and the first chunk of pass r+1 so the
            // reader gets a clear "wrap" — same delay as intra-pass is fine.
            if (i < chunkCount - 1 || r < repeats - 1) {
                delay(scrollDelayMs);
            }
        }
    }

    if (publishState && mqtt && mqtt->isConnected()) {
        // Publish the original input so consumers see the full message, not
        // just the final chunk on the display.
        mqtt->publishState(inputString);
    }
}

void SplitFlapDisplay::displayChunk(const String &chunk, float speed, bool centering) {
    String displayString = chunk; // already <= numModules by construction

    if (centering) {
        int totalPadding = numModules - displayString.length();
        int paddingLeft = totalPadding / 2;
        int paddingRight = totalPadding - paddingLeft;

        String result = "";
        for (int i = 0; i < paddingLeft; i++) {
            result += " ";
        }
        result += displayString;
        for (int i = 0; i < paddingRight; i++) {
            result += " ";
        }
        displayString = result;
    } else {                                          // pad blanks to end, if no centering
        while (displayString.length() < numModules) { // Pad with spaces
            displayString += " ";
        }
    }

    int targetPositions[numModules];
    for (int i = 0; i < displayString.length(); i++) {
        char currentChar = displayString[i];
        targetPositions[i] = modules[i].getCharPosition(currentChar);
        lastDisplayedChar[i] = currentChar;
    }
    moveTo(targetPositions, speed);
}

void SplitFlapDisplay::splitIntoChunks(
    const String &input, String chunks[], int maxChunks, int &outCount
) {
    outCount = 0;

    // Normalize: collapse internal whitespace to single spaces and trim ends.
    // Without this, runs of spaces create invisible word boundaries and we
    // produce empty-looking chunks.
    String s = "";
    bool lastWasSpace = true; // leading whitespace treated like a space
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
        chunks[0] = "";
        outCount = 1;
        return;
    }

    // Greedy word-pack: walk words, append to current chunk while it fits.
    // Word boundaries only — never split a word that can fit on its own.
    // Oversized words (longer than numModules) are split mid-word, which is
    // the only feasible option since they physically can't be displayed whole.
    // The tail of an oversized split is left in `current` so subsequent
    // words can pack with it.
    String current = "";
    int idx = 0;
    while (idx < (int) s.length() && outCount < maxChunks) {
        // Skip leading spaces to find the next word.
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

        if (wouldNeed <= numModules) {
            // Word fits (with separator if `current` already has content).
            if (neededSep) {
                current += ' ';
            }
            current += s.substring(wordStart, wordEnd);
            idx = wordEnd; // consume the word
        } else if (current.length() > 0) {
            // Word doesn't fit alongside what's in `current`. Flush `current`
            // (right-padded to numModules) and retry this word in a fresh
            // chunk by rewinding idx back to wordStart.
            while ((int) current.length() < numModules) {
                current += ' ';
            }
            chunks[outCount++] = current;
            current = "";
            idx = wordStart;
        } else {
            // `current` is empty and the single word is still too big.
            // Oversized-word edge case: split mid-word into numModules-sized
            // pieces. Emit full-size pieces, then leave the remainder as
            // the start of `current` so following words can pack with it.
            int remaining = wordLen;
            int pos = wordStart;
            while (remaining > numModules && outCount < maxChunks) {
                chunks[outCount++] = s.substring(pos, pos + numModules);
                pos += numModules;
                remaining -= numModules;
            }
            current = s.substring(pos, pos + remaining);
            idx = wordEnd; // past the oversized word
        }
    }

    // Flush final chunk.
    if (current.length() > 0 && outCount < maxChunks) {
        while ((int) current.length() < numModules) {
            current += ' ';
        }
        chunks[outCount++] = current;
    }
}

void SplitFlapDisplay::moveTo(int targetPositions[], float speed, bool releaseMotors) {
    // TODO check length of array and return if empty

    speed = constrain(speed, 2, maxVel);
    float stepsPerSecond = (speed / 60) * stepsPerRot;
    float timePerStep = 1000000 / stepsPerSecond;

    unsigned long currentTime = micros();

    int checkIntervalUs = 20 * 1000; // How often to check each modules hall effect sensor, less
    // than 20ms causes issues with bouncing
    int startStopDelay = 200; // time to wait to let motor realign itself to
    // magnetic field on stop and start

    bool resetLatches[numModules] = {};          // start with latch on to prevent case where the
    // motion starts with the magnet over the sensor
    bool needsStepping[numModules] = {};         // modules that still require moving
    bool motorsOn[numModules] = {};              // modules whose coils are currently energized
    unsigned long lastStepTimes[numModules] = {}; // track when each module was last stepped
    unsigned long lastSensorCheckTime = currentTime; // track when we last read all the hall effect sensors

    // Concurrency cap: limits how many motors are energized at once. Homing
    // paths set this from the "maxConcurrentMotors" setting (>= 1); normal
    // message writes set it to -1 (unlimited, all motors start immediately).
    SplitFlapMotorScheduler scheduler(numModules, maxConcurrentMotors);

    for (int i = 0; i < numModules; i++) {
        targetPositions[i] = constrain(
            targetPositions[i],
            0,
            stepsPerRot - 1
        ); // Constrain to avoid errors with incorrect inputs
        resetLatches[i] = true;
        lastStepTimes[i] = currentTime;
        needsStepping[i] = (modules[i].getPosition() != targetPositions[i]);
    }

    // Previously this energized EVERY module (even ones that don't need to
    // move) and left them all on until the end of the move. Now each motor
    // is energized only when it has a scheduling slot, and released the
    // instant it reaches its target, so the instantaneous coil current is
    // bounded by the cap instead of the full module count.
    if (maxConcurrentMotors < 0) {
        startMotors();
        delay(startStopDelay); // give the motor time to align to magnetic field
    }

    bool isFinished = checkAllFalse(needsStepping, numModules);
    while (! isFinished) {
        currentTime = micros();
        for (int i = 0; i < numModules; i++) {
            if (! needsStepping[i]) {
                continue;
            }

            // Grant a slot to motors that haven't started yet, subject to
            // the concurrency cap. Motors that can't get a slot simply wait
            // for the next loop iteration (a slot frees when another motor
            // reaches its target).
            if (! motorsOn[i]) {
                if (! scheduler.start(i)) {
                    continue; // cap reached — try again next tick
                }
                motorsOn[i] = true;
                if (maxConcurrentMotors >= 0) {
                    // Capped (homing): this motor got its slot mid-loop, so
                    // give it the same coil-align settle time the uncapped
                    // path's global delay above provides before its first
                    // step. The uncapped path leaves lastStepTimes at its
                    // initial value — motors already settled during the
                    // global delay, preserving the original step timing.
                    lastStepTimes[i] = currentTime + startStopDelay * 1000;
                }
            }

            if ((currentTime - lastStepTimes[i]) > timePerStep) {
                modules[i].step();
                lastStepTimes[i] = micros();
                if (modules[i].getPosition() == targetPositions[i]) { // this module is not in the correct position,
                    // requires stepping
                    needsStepping[i] = false;
                    // Release this motor's coils immediately instead of
                    // holding it until the slowest module finishes.
                    modules[i].stop();
                    motorsOn[i] = false;
                    scheduler.finish(i);
                }
            }
        }

        if ((currentTime - lastSensorCheckTime) > checkIntervalUs) { // check hall effect sensor every checkIntervalMs
            // check every modules sensor
            for (int i = 0; i < numModules; i++) {
                if (needsStepping[i] && motorsOn[i] &&
                    (modules[i].readHallEffectSensor() == true
                    )) { // only check sensors where the module is still moving
                    if (! resetLatches[i]) {
                        // UNCOMMENTING THIS WILL PROBBALY MAKE THE MOTORS INACCURATE, DUE
                        // TO TIME TAKEN TO PRINT
                        //  Serial.print("Module: ");
                        //  Serial.print(i);
                        //  Serial.print(" Magnet Position: ");
                        //  Serial.print(modules[i].getMagnetPosition());
                        //  Serial.print(" Actual Position: ");
                        //  Serial.print(modules[i].getPosition());
                        //  Serial.print(" Error: ");
                        //  Serial.println((modules[i].getMagnetPosition() -
                        //  modules[i].getPosition()));
                        modules[i].magnetDetected(); // update position to the modules
                        // magnet position
                        resetLatches[i] = true;
                    }
                } else if (resetLatches[i] == true) {
                    resetLatches[i] = false;
                }
            }
            isFinished = checkAllFalse(needsStepping, numModules);
            lastSensorCheckTime = currentTime; // recall micros because for loop may
            // take a moment to execute
        }
    }
    if (releaseMotors) {
        delay(startStopDelay); // allow all motors time to settle
        stopMotors();
    }
}

bool SplitFlapDisplay::checkAllFalse(bool array[], int size) {
    for (int i = 0; i < size; i++) {
        if (array[i] == true) {
            return false;              // As soon as a true value is found, return false
        }
    }
    return true;                       // All values were false
}

void SplitFlapDisplay::startMotors() { // Probably broken somewhere, not sure
    // why, haven't looked
    for (int i = 0; i < numModules; i++) {
        modules[i].start();
    }
}

void SplitFlapDisplay::stopMotors() {
    // Serial.println("Stopping Motors");
    for (int i = 0; i < numModules; i++) {
        modules[i].stop();
    }
}

void SplitFlapDisplay::powerDownCoils(TwoWire *bus) {
    // Same state as SplitFlapModule::init()/stop(): pin 15 as INPUT (hall
    // sensor), pins 1-4 as OUTPUT driven LOW so all four motor coils are
    // de-energized. Written to every address in the PCF8575 A0-A2 range so
    // expanders power up de-energized even if no module object exists for
    // them. A write to an address with no device just NACKs and returns
    // immediately; only a stuck bus reaches the 50ms setTimeOut().
    const uint16_t powerDownState = 0b1111111111100001;
    for (uint8_t addr = 0x20; addr <= 0x27; addr++) {
        bus->beginTransmission(addr);
        bus->write(powerDownState & 0xFF);        // lower byte
        bus->write((powerDownState >> 8) & 0xFF); // upper byte
        bus->endTransmission();                   // ignore result (may be NACK)
    }
}

void SplitFlapDisplay::setMqtt(SplitFlapMqtt *mqttHandler) {
    mqtt = mqttHandler;
}
