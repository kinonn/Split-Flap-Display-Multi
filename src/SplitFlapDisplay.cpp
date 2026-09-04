#include "SplitFlapDisplay.h"

#include "JsonSettings.h"
#include "SplitFlapModule.h"
#include "SplitFlapMqtt.h"
#include "SplitFlapMotorScheduler.h"
#include "StepMath.h"

SplitFlapDisplay::SplitFlapDisplay(JsonSettings &settings) : settings(settings), maxConcurrentMotors(-1) {
    for (int i = 0; i < MAX_MODULES; i++) {
        lastDisplayedChar[i] = ' ';
    }
}

void SplitFlapDisplay::init() {
    numModules = constrain(settings.getInt("moduleCount"), 1, MAX_MODULES);
    stepsPerRot = sanitizeStepsPerRot(settings.getInt("stepsPerRot"));
    displayOffset = settings.getInt("displayOffset");
    magnetPosition = settings.getInt("magnetPosition");
    maxVel = sanitizeMaxVel(settings.getFloat("maxVel"));
    charSetSize = settings.getInt("charset");

    std::vector<int> settingAddresses = settings.getIntVector("moduleAddresses");
    for (int i = 0; i < numModules; i++) {
        // The stored list is free-form: a short or malformed one must not read
        // past the end of the vector. Missing entries fall back to the stock
        // address chain (the schema default for moduleAddresses).
        moduleAddresses[i] =
            (uint8_t) (i < (int) settingAddresses.size() ? settingAddresses[i] : 0x20 + i);
    }

    std::vector<int> settingOffsets = settings.getIntVector("moduleOffsets");
    for (int i = 0; i < numModules; i++) {
        moduleOffsets[i] = i < (int) settingOffsets.size() ? settingOffsets[i] : 0;
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
        modules[i] = SplitFlapModule(
            moduleAddresses[i], stepsPerRot, moduleOffsets[i] + displayOffset, magnetPosition, charSetSize, charOffsets[i]
        );
    }

    SDAPin = settings.getInt("sdaPin");
    SCLPin = settings.getInt("sclPin");

    Wire.begin(SDAPin, SCLPin);
    Wire.setClock(400000);

    for (uint8_t i = 0; i < numModules; i++) {
        modules[i].init();
    }
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
    for (int i = 0; i < numModules; i++) {
        // Keep the module's current offset when the stored list is shorter
        // than numModules: zeroing here would physically re-home modules
        // whose calibration never changed.
        moduleOffsets[i] = i < (int) settingOffsets.size() ? settingOffsets[i] : moduleOffsets[i];
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

namespace {
// Characters exercised by the diagnostic test modes: space, A-Z, 0-9.
const char kTestChars[] = {' ', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
                           'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y',
                           'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'};
} // namespace

void SplitFlapDisplay::testAll() {
    maxConcurrentMotors = -1; // unlimited — test/display moves are short
    int numChars = sizeof(kTestChars) / sizeof(kTestChars[0]);
    int targetPositions[numModules];

    for (int i = 0; i < numChars; i++) {
        // Serial.print("Target Positions: [");
        // fill array with same char

        for (int j = 0; j < numModules; j++) {
            targetPositions[j] = modules[j].getCharPosition(kTestChars[i]);
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

    int targetPositions[numModules];
    char randChar;

    Serial.print("Target: ");
    for (int i = 0; i < numModules; i++) {
        randChar = kTestChars[random(0, (int) (sizeof(kTestChars) / sizeof(kTestChars[0])))];
        targetPositions[i] = modules[i].getCharPosition(randChar);
        Serial.print(randChar);
    }
    Serial.println(" ");
    moveTo(targetPositions, speed);
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

    String chunks[MAX_MODULES * 4]; // generous upper bound for very long input
    int chunkCount = 0;
    splitIntoChunks(inputString, chunks, MAX_MODULES * 4, chunkCount);

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
    // Defensive: every caller passes chunk.length() <= numModules today, but
    // the guard lives only in the callers (e.g. writeString/splitIntoChunks)
    // — homeToString() forwards its argument unchecked, and one refactor
    // away this would overflow targetPositions[lastDisplayedChar] below.
    // Truncate to the physical width instead of trusting the caller.
    String displayString = chunk;
    if (displayString.length() > (unsigned int) numModules) {
        displayString.remove(numModules);
    }

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
    int stepsRemaining[numModules] = {};         // steps each module still has to turn
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

        // The drum only ever turns forwards, so the work left to do is the
        // forward distance to the target. Counting steps instead of comparing
        // positions for equality means a mid-move magnet correction that
        // snaps `position` past the target cannot strand a module here (the
        // old equality test never matched again, sending it round for
        // another full revolution) — the count is re-derived from the
        // corrected position inside the sensor check below.
        stepsRemaining[i] = forwardSteps(modules[i].getPosition(), targetPositions[i], stepsPerRot);

        // Energize movers only, and only on the uncapped path. Holding every
        // module — even ones that don't move — costs ~200mA each for nothing,
        // and on a shared supply that extra draw is exactly what makes the
        // moving modules lose steps. Capped (homing) moves energize lazily
        // when the scheduler grants a slot (the motor's first step()).
        if (stepsRemaining[i] > 0 && maxConcurrentMotors < 0) {
            modules[i].start(); // hold the drum on its current pattern without moving
        }
    }

    // Uncapped movers were just energized: give the coils time to align to the
    // magnetic field before the first step. Capped (homing) moves energize
    // lazily per slot instead, and each motor gets this settle time when it
    // gets its slot (see lastStepTimes below).
    bool anyMoving = ! allStepsDone(stepsRemaining, numModules);
    if (anyMoving && maxConcurrentMotors < 0) {
        delay(startStopDelay);
    }

    bool isFinished = ! anyMoving;
    while (! isFinished) {
        currentTime = micros();
        for (int i = 0; i < numModules; i++) {
            if (stepsRemaining[i] <= 0) {
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
                stepsRemaining[i]--;
                lastStepTimes[i] = micros();
                if (stepsRemaining[i] == 0) { // this module has arrived
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
                if (stepsRemaining[i] > 0 && motorsOn[i] &&
                    (modules[i].readHallEffectSensor() ==
                     true)) { // only check sensors where the module is still moving
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

                        // The magnet correction snaps `position` to the known magnet
                        // position, which can move it in either direction. Re-derive
                        // what is left to turn from the corrected position — with the
                        // old equality test, a correction past the target could never
                        // match and the module went round for another full revolution.
                        stepsRemaining[i] = forwardSteps(modules[i].getPosition(), targetPositions[i], stepsPerRot);
                        if (stepsRemaining[i] == 0) {
                            modules[i].stop();
                            motorsOn[i] = false;
                            scheduler.finish(i);
                        }
                        resetLatches[i] = true;
                    }
                } else if (resetLatches[i] == true) {
                    resetLatches[i] = false;
                }
            }
            lastSensorCheckTime = currentTime; // recall micros because for loop may
            // take a moment to execute
        }

        isFinished = allStepsDone(stepsRemaining, numModules);
    }
    if (releaseMotors) {
        delay(startStopDelay); // allow all motors time to settle
        stopMotors();
    }
}

bool SplitFlapDisplay::allStepsDone(const int stepsRemaining[], int size) {
    for (int i = 0; i < size; i++) {
        if (stepsRemaining[i] > 0) {
            return false; // As soon as a module with work left is found, return false
        }
    }
    return true;          // Every module has arrived
}

void SplitFlapDisplay::stopMotors() {
    // Serial.println("Stopping Motors");
    for (int i = 0; i < numModules; i++) {
        modules[i].stop();
    }
}

void SplitFlapDisplay::setMqtt(SplitFlapMqtt *mqttHandler) {
    mqtt = mqttHandler;
}
