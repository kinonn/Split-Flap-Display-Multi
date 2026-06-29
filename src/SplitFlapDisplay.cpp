#include "SplitFlapDisplay.h"

#include "JsonSettings.h"
#include "SplitFlapModule.h"
#include "SplitFlapMqtt.h"

SplitFlapDisplay::SplitFlapDisplay(JsonSettings &settings) : settings(settings) {}

void SplitFlapDisplay::init() {
    numModules = settings.getInt("moduleCount");
    stepsPerRot = settings.getInt("stepsPerRot");
    displayOffset = settings.getInt("displayOffset");
    magnetPosition = settings.getInt("magnetPosition");
    maxVel = settings.getFloat("maxVel");
    charSetSize = settings.getInt("charset");

    std::vector<int> settingAddresses = settings.getIntVector("moduleAddresses");
    for (int i = 0; i < numModules; i++) {
        moduleAddresses[i] = (uint8_t) settingAddresses[i];
    }

    std::vector<int> settingOffsets = settings.getIntVector("moduleOffsets");
    for (int i = 0; i < numModules; i++) {
        moduleOffsets[i] = settingOffsets[i];
    }

    Serial.print("Module Offsets: ");
    for (int i = 0; i < numModules; i++) {
        Serial.print(moduleOffsets[i]);
        Serial.print(" ");
    }
    Serial.println();

    for (uint8_t i = 0; i < numModules; i++) {
        modules[i] = SplitFlapModule(
            moduleAddresses[i], stepsPerRot, moduleOffsets[i] + displayOffset, magnetPosition, charSetSize
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

void SplitFlapDisplay::testAll() {
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
    int targetPositions[numModules];
    for (int i = 0; i < numModules; i++) {
        targetPositions[i] = (modules[i].getPosition() - 1 + stepsPerRot) % stepsPerRot;
    }
    startMotors();
    moveTo(targetPositions, speed, false);
    char homeChar = ' ';
    int charPosition;
    for (int i = 0; i < numModules; i++) {
        targetPositions[i] = modules[i].getCharPosition(homeChar);
    }
    moveTo(targetPositions, speed);
}

void SplitFlapDisplay::homeToString(String homeString, float speed, bool centering) {
    Serial.println("Homing");
    int targetPositions[numModules];
    for (int i = 0; i < numModules; i++) {
        targetPositions[i] = (modules[i].getPosition() - 1 + stepsPerRot) % stepsPerRot;
    }
    startMotors();
    moveTo(targetPositions, speed, false);
    writeString(homeString, speed, centering);
}

void SplitFlapDisplay::homeToChar(char homeChar, float speed) {
    Serial.println("Homing");
    int targetPositions[numModules];
    for (int i = 0; i < numModules; i++) {
        targetPositions[i] = (modules[i].getPosition() - 1 + stepsPerRot) % stepsPerRot;
    }
    startMotors();
    moveTo(targetPositions, speed, false);

    for (int i = 0; i < numModules; i++) {
        targetPositions[i] = modules[i].getCharPosition(homeChar);
    }
    moveTo(targetPositions, true, speed);
}

void SplitFlapDisplay::writeChar(char inputChar, float speed) {
    int targetPositions[numModules];
    // Iterate through the input string and process each character
    for (int i = 0; i < numModules; i++) {
        targetPositions[i] = modules[i].getCharPosition(inputChar);
    }
    moveTo(targetPositions, speed);
}

void SplitFlapDisplay::writeString(
    String inputString, float speed, bool centering, unsigned long scrollDelayMs
) {
    // Short path: fits in one frame — behave exactly as before.
    if (inputString.length() <= numModules) {
        displayChunk(inputString, speed, centering);
        if (mqtt && mqtt->isConnected()) {
            mqtt->publishState(inputString);
        }
        return;
    }

    // Long path: split into word-respecting chunks and display sequentially.
    String chunks[MAX_MODULES * 4]; // generous upper bound for very long input
    int chunkCount = 0;
    splitIntoChunks(inputString, chunks, MAX_MODULES * 4, chunkCount);

    Serial.printf(
        "[scroll] input=%d chars, numModules=%d, chunks=%d\n",
        inputString.length(), numModules, chunkCount
    );

    for (int i = 0; i < chunkCount; i++) {
        displayChunk(chunks[i], speed, centering);
        if (i < chunkCount - 1) {
            delay(scrollDelayMs);
        }
    }

    if (mqtt && mqtt->isConnected()) {
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

    bool resetLatches[numModules] = {}; // Initialize to false //start with latch on to prevent case where the
    // motion starts with the magnet over the sensor
    bool needsStepping[numModules] = {};             // Initialize to false; //modules that still require moving
    unsigned long lastStepTimes[numModules] = {};    // Initialize to false; //track when each module was last stepped
    unsigned long lastSensorCheckTime = currentTime; // track when we last read all the hall effect sensors

    for (int i = 0; i < numModules; i++) {
        targetPositions[i] = constrain(
            targetPositions[i],
            0,
            stepsPerRot - 1
        ); // Constrain to avoid errors with incorrect inputs
        resetLatches[i] = true;
        lastStepTimes[i] = currentTime;
        if (modules[i].getPosition() != targetPositions[i]) {
            needsStepping[i] = true;
        } else {
            needsStepping[i] = false;
        }
    }

    startMotors(); // not sure if this helps or not, likely that it does not based
    // on testing
    delay(startStopDelay); // give the motor time to align to magnetic field

    bool isFinished = checkAllFalse(needsStepping, numModules);
    while (! isFinished) {
        currentTime = micros();
        for (int i = 0; i < numModules; i++) {
            if (((currentTime - lastStepTimes[i]) > timePerStep) && needsStepping[i]) {
                modules[i].step();
                lastStepTimes[i] = micros();
                if (modules[i].getPosition() == targetPositions[i]) { // this module is not in the correct position,
                    // requires stepping
                    needsStepping[i] = false;
                }
            }
        }

        if ((currentTime - lastSensorCheckTime) > checkIntervalUs) { // check hall effect sensor every checkIntervalMs
            // check every modules sensor
            for (int i = 0; i < numModules; i++) {
                if (needsStepping[i] &&
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

void SplitFlapDisplay::setMqtt(SplitFlapMqtt *mqttHandler) {
    mqtt = mqttHandler;
}
