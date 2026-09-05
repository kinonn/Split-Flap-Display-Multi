#pragma once

#include "JsonSettings.h"
#include "SplitFlapModule.h"

#include <Arduino.h>

#define MAX_MODULES 8                 // for memory allocation, update if more modules
#define MAX_RPM 15.0f
#define DEFAULT_SCROLL_DELAY_MS 1500  // pause between chunks when scrolling
#define DEFAULT_SCROLL_REPEAT_COUNT 2 // how many times a long message is shown end-to-end
#define MIN_SCROLL_REPEAT_COUNT 1     // floor for scrollRepeatCount (1 = no repeat)
#define MAX_SCROLL_REPEAT_COUNT 99    // ceiling — guards against runaway looping

class SplitFlapMqtt;

class SplitFlapDisplay {
  public:
    SplitFlapDisplay(JsonSettings &settings);

    void init();
    void reloadOffsets();
    void homeAffectedModules(bool affected[], float speed = MAX_RPM);
    void writeString(
        String inputString, float speed = MAX_RPM, bool centering = true,
        unsigned long scrollDelayMs = DEFAULT_SCROLL_DELAY_MS, int scrollRepeatCount = DEFAULT_SCROLL_REPEAT_COUNT,
        bool publishState = true
    );                                     // Move all modules at once to show a specific string. If longer than
                                           // numModules, splits on word boundaries and shows chunks sequentially,
                                           // repeating the full chunk sequence scrollRepeatCount times total.
    void writeChar(char inputChar,
                   float speed = MAX_RPM); // sets all modules to a single char
    void moveTo(int targetPositions[], float speed = MAX_RPM, bool releaseMotors = true);
    void home(float speed = MAX_RPM);      // move home
    void homeToString(
        String homeString, float speed = MAX_RPM,
        bool centering = true
    );                                      // moves home and then writes a string
    void homeToChar(char homeChar,
                    float speed = MAX_RPM); // moves home and then sets all modules to a char
    void testAll();
    void testRandom(float speed = MAX_RPM);
    int getNumModules() { return numModules; }
    int getCharsetSize() const { return charSetSize; }
    void setMqtt(SplitFlapMqtt *mqttHandler);

  private:
    JsonSettings &settings;

    bool allStepsDone(const int stepsRemaining[], int size);
    void stopMotors();

    // Split a string into chunks of <= numModules chars, breaking only at word
    // boundaries. A word longer than numModules is split mid-word (no other
    // option — it physically can't fit whole).
    void splitIntoChunks(const String &input, String chunks[], int maxChunks, int &outCount);

    // Display one already-sized chunk using the same center/pad logic as
    // writeString. Overlong input is truncated to numModules (defensive —
    // homeToString() forwards its argument unchecked).
    void displayChunk(const String &chunk, float speed = MAX_RPM, bool centering = true);

    int numModules;
    uint8_t moduleAddresses[MAX_MODULES];
    SplitFlapModule modules[MAX_MODULES];
    int moduleOffsets[MAX_MODULES];
    int charOffsets[MAX_MODULES][48];
    int displayOffset;
    char lastDisplayedChar[MAX_MODULES]; // last char each module was asked to
                                         // show; used by reloadOffsets to
                                         // reposition after a home

    // Concurrency cap for moveTo(): number of motors allowed to be
    // energized at the same time. < 0 = unlimited (normal message writes).
    // Homing paths set this from the "maxConcurrentMotors" setting so the
    // boot homing peak stays bounded on large displays.
    int maxConcurrentMotors;

    float maxVel;       // Max Velocity In RPM
    int charSetSize;    // 37 for standard, 48 for extended
    int stepsPerRot;    // number of motor steps per full rotation of character
                        // drum
    int magnetPosition; // position of drum wheel when magnet is detected
    int SDAPin;         // SDA pin
    int SCLPin;         // SCL pin

    SplitFlapMqtt *mqtt = nullptr;
};
