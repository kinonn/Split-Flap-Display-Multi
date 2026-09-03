// Minimal Arduino String/Serial stubs for test/jsonsettings_test.cpp.
// Separate from test/stubs/Arduino.h because that one has no String class
// (SplitFlapModule doesn't use it) and the two tests must not fight over
// include paths.
#pragma once

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#define HEX 16

// ---------------------------------------------------------------------------
// Serial: captures println'd lines so tests can assert on log output.
// ---------------------------------------------------------------------------

extern std::vector<std::string> g_serialLines;

class SerialStub {
  public:
    void print(const char *s) { (void) s; }
    void print(const std::string &s) { (void) s; }
    void print(char c) { (void) c; }
    void print(int v) { (void) v; }
    void print(unsigned int v) { (void) v; }
    void print(float v) { (void) v; }
    template<typename T> void println(T v) {
        std::string s;
        if constexpr (std::is_same_v<T, const char *> || std::is_same_v<T, char *>) {
            s = v;
        } else if constexpr (std::is_same_v<T, int>) {
            s = std::to_string(v);
        } else if constexpr (std::is_convertible_v<T, std::string>) {
            s = std::string(v);
        }
        g_serialLines.push_back(s);
    }
    void begin(unsigned long) {}
    void flush() {}

    // printf is used by SplitFlapMqtt.cpp logging; capture like println.
    template<typename... Args> void printf(const char *fmt, Args... args) {
        char buf[256];
        std::snprintf(buf, sizeof(buf), fmt, args...);
        g_serialLines.push_back(std::string(buf));
    }
};

extern SerialStub Serial;

// ---------------------------------------------------------------------------
// String: std::string-backed, only the operations JsonSettings uses.
// ---------------------------------------------------------------------------

class String {
  public:
    String() {}
    String(const char *c) : s_(c ? c : "") {}
    String(const std::string &s) : s_(s) {}
    String(int v) : s_(std::to_string(v)) {}
    String(unsigned int v, int base) {
        char buf[16];
        if (base == HEX) {
            std::snprintf(buf, sizeof(buf), "%x", v);
        } else {
            std::snprintf(buf, sizeof(buf), "%u", v);
        }
        s_ = buf;
    }
    String(float v) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%.2f", (double) v); // Arduino prints 2 decimals
        s_ = buf;
    }

    const char *c_str() const { return s_.c_str(); }
    unsigned int length() const { return (unsigned int) s_.size(); }
    char operator[](int i) const { return i >= 0 && i < (int) s_.size() ? s_[i] : '\0'; }

    String substring(unsigned int from, unsigned int to) const {
        if (from >= s_.size()) return String();
        return String(s_.substr(from, to - from));
    }

    void toUpperCase() {
        std::transform(s_.begin(), s_.end(), s_.begin(), [](unsigned char c) { return std::toupper(c); });
    }

    String &operator+=(const String &o) {
        s_ += o.s_;
        return *this;
    }
    String &operator+=(const char *c) {
        s_ += c;
        return *this;
    }
    String &operator+=(char c) {
        s_ += c;
        return *this;
    }

    friend String operator+(const String &a, const String &b) { return String(a.s_ + b.s_); }
    friend String operator+(const char *a, const String &b) { return String(std::string(a) + b.s_); }

    bool operator==(const String &o) const { return s_ == o.s_; }
    bool operator!=(const String &o) const { return s_ != o.s_; }
    bool operator<(const String &o) const { return s_ < o.s_; } // std::map key

    operator std::string() const { return s_; }

  private:
    std::string s_;
};

// ---------------------------------------------------------------------------
// Misc Arduino bits (kept tiny; JsonSettings barely touches these)
// ---------------------------------------------------------------------------

inline unsigned long millis() { return 0; }
inline unsigned long micros() { return 0; }
inline void delay(unsigned long) {}
template<typename T> T constrain(T amt, T low, T high) {
    return amt < low ? low : (amt > high ? high : amt);
}
