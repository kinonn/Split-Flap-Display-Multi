#include "SplitFlapModule.h"

// Array of characters, in order, the first item is located on the magnet on the
// character drum
const char SplitFlapModule::StandardChars[37] = {' ', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
                                                 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y',
                                                 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'};

const char SplitFlapModule::ExtendedChars[48] = {
    ' ', 'A', 'B', 'C', 'D', 'E',  'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O',
    'P', 'Q', 'R', 'S', 'T', 'U',  'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4',
    '5', '6', '7', '8', '9', '\'', ':', '?', '!', '.', '-', '/', '$', '@', '#', '%',
};

// Full-step (2-phase-on) coil patterns for one electrical revolution of the
// 28BYJ-48, see the header for the bit layout. Index 0..3 is the order in
// which step() writes them; writing them in increasing order turns the drum
// forwards.
const uint16_t SplitFlapModule::CoilStates[4] = {
    0b1111111111100111, // P01 + P02
    0b1111111111110011, // P01 + P04
    0b1111111111111001, // P03 + P04
    0b1111111111101101, // P02 + P03
};

// All four coil pins low: the motor is released and draws no current.
const uint16_t SplitFlapModule::IdleState = 0b1111111111100001;

// Default Constructor
SplitFlapModule::SplitFlapModule()
    : address(0), position(0), stepNumber(0), stepsPerRot(0), chars(StandardChars), numChars(37), charSetSize(37) {
    magnetPosition = 710;
}

// Constructor implementation
SplitFlapModule::SplitFlapModule(
    uint8_t I2Caddress, int stepsPerFullRotation, int stepOffset, int magnetPos, int charsetSize, const int offsets[]
)
    : address(I2Caddress), position(0), stepNumber(0), stepsPerRot(stepsPerFullRotation), charSetSize(charsetSize) {
    magnetPosition = magnetPos + stepOffset;

    chars = (charsetSize == 48) ? ExtendedChars : StandardChars;
    numChars = (charsetSize == 48) ? 48 : 37;

    for (int i = 0; i < 48; i++) {
        charOffsets[i] = (offsets != nullptr) ? offsets[i] : 0;
    }
}

void SplitFlapModule::writeIO(uint16_t data) {
    Wire.beginTransmission(address);
    Wire.write(data & 0xFF);        // Send lower byte
    Wire.write((data >> 8) & 0xFF); // Send upper byte

    byte error = Wire.endTransmission();

    if (error > 0) {
        if (! hasErrored) {
            hasErrored = true; // Set the error flag
            Serial.print("Error writing data to module ");
            Serial.print(address);
            Serial.print(", error code: ");
            Serial.println(error); // Error codes:
            // 0 = success
            // 1 = data too long to fit in transmit buffer
            // 2 = received NACK on transmit of address
            // 3 = received NACK on transmit of data
            // 4 = other error
        }
    } else if (hasErrored) {
        // A single glitch on the bus must not disable this module forever
        // (readHallEffectSensor() refuses to read while hasErrored is set, so
        // a latched flag would disable magnet correction permanently): clear
        // the flag as soon as the module answers again.
        hasErrored = false;
        Serial.print("Module ");
        Serial.print(address);
        Serial.println(" recovered");
    }
}

// Init Module, Setup IO Board
void SplitFlapModule::init() {
    float stepSize = (float) stepsPerRot / (float) numChars;
    float currentPosition = 0;
    for (int i = 0; i < numChars; i++) {
        charPositions[i] = (int) currentPosition + charOffsets[i];
        currentPosition += stepSize;
    }

    uint16_t initState = IdleState; // Pin 15 (17) as INPUT, Pins 1-4 as OUTPUT
    writeIO(initState);

    stop();                         // Write all motor coil inputs LOW

    // NOTE: this used to "prime" the motor with 4 steps (4x100ms delays).
    // It serves no functional purpose — homing finds the magnet via the
    // hall-effect sensor — and it drew coil current for up to 400ms per
    // module at power-on, when the rail is at its weakest. Removed so the
    // coils are powered down as early as possible after boot.
}

void SplitFlapModule::updateOffsets(const int newCharOffsets[], int newMagnetOffset) {
    magnetPosition = newMagnetOffset;

    for (int i = 0; i < 48; i++) {
        charOffsets[i] = newCharOffsets[i];
    }

    float stepSize = (float) stepsPerRot / (float) numChars;
    float currentPosition = 0;
    for (int i = 0; i < numChars; i++) {
        charPositions[i] = (int) currentPosition + charOffsets[i];
        currentPosition += stepSize;
    }
}

int SplitFlapModule::getCharPosition(char inputChar) {
    inputChar = toupper(inputChar);
    for (int i = 0; i < charSetSize; i++) {
        if (chars[i] == inputChar) {
            return charPositions[i];
        }
    }
    return 0; // Character not found, return blank
}

const char *SplitFlapModule::drumOrder(int charsetSize, int &outLen) {
    if (charsetSize == 48) {
        outLen = 48;
        return ExtendedChars;
    }
    outLen = 37;
    return StandardChars;
}

void SplitFlapModule::stop() {
    writeIO(IdleState);
}

// Re-energize the coil pattern the rotor is already resting on so the drum is
// held before a move begins. step() leaves stepNumber pointing at the *next*
// pattern to write, so the pattern currently energizing the rotor is
// stepNumber - 1.
//
// This must NOT change stepNumber: advancing it and writing the "next"
// pattern would drag the drum a full step forwards, and any of the old
// rewind-then-write approaches made a second start() without an intervening
// step() write a pattern the rotor is not resting on, dragging it backwards
// and silently de-calibrating held modules.
void SplitFlapModule::start() {
    writeIO(CoilStates[(stepNumber + 3) % 4]);
}

void SplitFlapModule::step() {
    writeIO(CoilStates[stepNumber]);
    stepNumber = (stepNumber + 1) % 4;
    position = (position + 1) % stepsPerRot;
}

bool SplitFlapModule::readHallEffectSensor() {
    if (hasErrored) {
        return false;
    }

    uint8_t requestBytes = 2;
    Wire.requestFrom(address, requestBytes);
    // Make sure the data is available
    if (Wire.available() == 2) {
        uint16_t inputState = 0;

        // Read the two bytes and combine them into a 16-bit value
        inputState = Wire.read();             // Read the lower byte
        inputState |= (Wire.read() << 8);     // Read the upper byte and shift it left

        return (inputState & (1 << 15)) != 0; // If bit is 15, return HIGH, else LOW
    }
    return false;
}
