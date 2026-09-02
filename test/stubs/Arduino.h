// Minimal Arduino host stubs for test/splitflap_module_test.cpp.
// Provides just enough of Arduino.h / Wire.h for the real
// src/SplitFlapModule.cpp to compile and run on the host.
#pragma once

#include <cctype>
#include <cstdint>
#include <cstdio>
#include <utility>
#include <vector>

typedef uint8_t byte;

// ---------------------------------------------------------------------------
// Serial
// ---------------------------------------------------------------------------

class SerialStub {
  public:
    template<typename T> void print(T) {}
    template<typename T> void println(T) {}
    void begin(unsigned long) {}
    void flush() {}
};

extern SerialStub Serial;

// ---------------------------------------------------------------------------
// TwoWire: records writes, can simulate a failing transmission.
// ---------------------------------------------------------------------------

extern std::vector<std::pair<uint8_t, uint16_t>> g_writes;
extern bool g_failNextTransmission;

void wireReset();
int wireWriteCount(uint8_t address);
uint16_t wireLastValue(uint8_t address);

class TwoWire {
  public:
    void begin(int = 0, int = 0) {}
    void setClock(long) {}
    void beginTransmission(uint8_t address) {
        inTransmission_ = true;
        n_ = 0;
        address_ = address;
    }
    void write(uint8_t b) {
        if (inTransmission_ && n_ < 2) {
            buf_[n_++] = b;
        }
    }
    uint8_t endTransmission() {
        inTransmission_ = false;
        if (g_failNextTransmission) {
            g_failNextTransmission = false;
            n_ = 0;
            return 2; // NACK on address, like a missing module
        }
        // Lower byte is written first (see SplitFlapModule::writeIO).
        uint16_t value = (uint16_t) (buf_[0] | (buf_[1] << 8));
        g_writes.push_back({address_, value});
        n_ = 0;
        return 0;
    }
    uint8_t requestFrom(uint8_t, uint8_t) { return 0; }
    int available() { return 0; }
    int read() { return -1; }

  private:
    bool inTransmission_ = false;
    uint8_t address_ = 0;
    uint8_t buf_[2] = {0, 0};
    int n_ = 0;
};

extern TwoWire Wire;

// ---------------------------------------------------------------------------
// Arduino misc
// ---------------------------------------------------------------------------

inline unsigned long micros() { return 0; }
inline unsigned long millis() { return 0; }
inline void delay(unsigned long) {}
inline long constrain(long amt, long low, long high) {
    return amt < low ? low : (amt > high ? high : amt);
}
