#pragma once

#include <Arduino.h>
#include <Wire.h>

class SplitFlapModule {
  public:
    // Constructor declarationS
    SplitFlapModule(); // default constructor required to allocate memory for
    // SplitFlapDisplay class
    SplitFlapModule(
        uint8_t I2Caddress, int stepsPerFullRotation, int stepOffset, int magnetPos, int charSetSize,
        const int charOffsets[] = nullptr
    );

    void init();
    void updateOffsets(const int newCharOffsets[], int newMagnetOffset);

    void step();  // advance the motor one full step
    void stop();  // write all motor input pins to low
    void start(); // re-energize the coil pattern the rotor is resting on, without moving

    int getMagnetPosition() const { return magnetPosition; } // position where magnet is detected
    int getCharPosition(char inputChar);                     // get integer position given single character
    int getPosition() const { return position; }             // get integer position
    int getCharsetSize() const { return numChars; }          // getter for charset size

    bool readHallEffectSensor();                             // return the value read by the hall effect
    // sensor
    void magnetDetected() {
        position = magnetPosition;
    } // update position to magnetposition, called when magnet is detected

    bool getHasErrored() const { return hasErrored; }

    // Canonical 37-char drum table, shared with the display's diagnostic test
    // modes so the two can never drift apart.
    static const char StandardChars[37];

    // Drum order for a charset size (37 or 48; anything else selects 37).
    // Returns a pointer to the canonical table and writes its length to
    // outLen. Used by the calibration status API so the AI agent never
    // hardcodes glyph order.
    static const char *drumOrder(int charsetSize, int &outLen);

  private:
    uint8_t address;             // i2c address of module
    int position;                // character drum position
    int stepNumber;              // index of the NEXT coil pattern to write (0-3)
    int stepsPerRot;             // number of steps per rotation
    bool hasErrored = false;     // set when the last i2c transaction failed, cleared when one succeeds

    void writeIO(uint16_t data); // write to motor in pins

    int magnetPosition;          // altered by offsets

    // Coil patterns for one electrical revolution of the 28BYJ-48 (2-phase-on
    // full stepping). Bits 1-4 drive the motor coils, bit 0 and bit 15 are
    // left high: bit 15 reads the hall effect sensor input on the PCF8575,
    // bit 0 is unused. Writing the patterns in increasing order turns the
    // drum forwards, in decreasing order backwards.
    static const uint16_t CoilStates[4];
    static const uint16_t IdleState; // all four coils low, hall input left high

    const char *chars;               // pointer to active character set
    int charPositions[48];           // support up to 48 characters
    int charOffsets[48];             // per-character step offsets
    int numChars;                    // current number of characters
    int charSetSize;

    static const char ExtendedChars[48];
};

// //PINs on the PCF8575 Board
// #define P00  	0
// #define P01  	1
// #define P02  	2
// #define P03  	3
// #define P04  	4
// #define P05  	5
// #define P06  	6
// #define P07  	7
// #define P10  	8
// #define P11  	9
// #define P12  	10
// #define P13  	11
// #define P14  	12
// #define P15  	13
// #define P16  	14
// #define P17  	15
