// Minimal ArduinoJson v7 stub for test/jsonsettings_test.cpp.
//
// NOT a faithful reimplementation — implements only the surface
// JsonSettings.cpp uses (JsonDocument with operator[] and as<JsonObject>()
// iteration yielding JsonPair, JsonVariant::as<T>()). It exists so the REAL
// JsonSettings.cpp compiles on the host; the JSON library's own behavior is
// not under test. The firmware builds against the real ArduinoJson in
// PlatformIO.
#pragma once

#include <cstdint>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "Arduino.h"

struct JsonDocumentEntry {
    std::string key;
    int type = 0; // 0 = string, 1 = integer, 2 = float
    long i = 0;
    double f = 0;
    std::string s;
};

class JsonVariant {
  public:
    JsonVariant() = default;
    explicit JsonVariant(const JsonDocumentEntry *e) : e_(e) {}

    template<typename T> T as() const {
        if (! e_) return T();
        if constexpr (std::is_same_v<T, String>) {
            if (e_->type == 1) return String(std::to_string(e_->i));
            if (e_->type == 2) return String((float) e_->f);
            return String(e_->s);
        } else if constexpr (std::is_integral_v<T>) {
            return (T) e_->i;
        } else if constexpr (std::is_floating_point_v<T>) {
            return (T) (e_->type == 1 ? (double) e_->i : e_->f);
        }
    }

    const JsonDocumentEntry *e_ = nullptr;
};

// Lightweight key view: stays valid as long as the owning JsonPair lives
// (firmware does `const char *key = kv.key().c_str();` and keeps using it).
class StringRef {
  public:
    explicit StringRef(const std::string *s) : s_(s) {}
    const char *c_str() const { return s_->c_str(); }

  private:
    const std::string *s_;
};

// Namespace-scope JsonPair: models ArduinoJson's pair returned when iterating
// a JsonObject. Firmware code does `for (JsonPair kv : doc.as<JsonObject>())`.
class JsonPair {
  public:
    explicit JsonPair(const JsonDocumentEntry &e) : keyStore_(e.key), e_(&e) {}
    StringRef key() const { return StringRef(&keyStore_); }
    JsonVariant value() const { return JsonVariant(e_); }

  private:
    std::string keyStore_; // owned copy: key().c_str() must outlive the expression
    const JsonDocumentEntry *e_;
};

class JsonDocument {
  public:
    using Entry = JsonDocumentEntry;

    // operator[] proxy for writes
    class Ref {
      public:
        Ref(JsonDocument *d, std::string k) : d_(d), k_(std::move(k)) {}

        Ref &operator=(const char *v) {
            Entry e;
            e.type = 0;
            e.s = v;
            d_->set(k_, e);
            return *this;
        }
        Ref &operator=(const String &v) { return *this = v.c_str(); }
        Ref &operator=(long v) {
            Entry e;
            e.type = 1;
            e.i = v;
            d_->set(k_, e);
            return *this;
        }
        Ref &operator=(int v) { return *this = (long) v; }
        Ref &operator=(double v) {
            Entry e;
            e.type = 2;
            e.f = v;
            d_->set(k_, e);
            return *this;
        }
        Ref &operator=(float v) { return *this = (double) v; }

      private:
        JsonDocument *d_;
        std::string k_;
    };

    Ref operator[](const char *k) { return Ref(this, k); }
    Ref operator[](const String &k) { return Ref(this, std::string(k.c_str())); }

    // as<JsonObject>() iteration support
    struct JsonObjectIterator {
        explicit JsonObjectIterator(std::vector<Entry>::iterator it) : it_(it) {}
        bool operator!=(const JsonObjectIterator &o) const { return it_ != o.it_; }
        void operator++() { ++it_; }
        JsonPair operator*() const { return JsonPair(*it_); }
        std::vector<Entry>::iterator it_;
    };

    JsonObjectIterator objBegin() { return JsonObjectIterator(entries_.begin()); }
    JsonObjectIterator objEnd() { return JsonObjectIterator(entries_.end()); }

    template<typename T> T as() const; // specialized for JsonObject below

    std::vector<Entry> entries_;

  private:
    Entry *find(const std::string &key) {
        for (auto &e : entries_) {
            if (e.key == key) return &e;
        }
        return nullptr;
    }
    void set(const std::string &key, const Entry &e) {
        Entry *x = find(key);
        if (x) {
            *x = e;
        } else {
            entries_.push_back(e);
            entries_.back().key = key;
        }
    }
};

// Range wrapper so `for (JsonPair kv : doc.as<JsonObject>())` works.
class JsonObject {
  public:
    using iterator = JsonDocument::JsonObjectIterator;
    explicit JsonObject(JsonDocument *d) : begin_(d->objBegin()), end_(d->objEnd()) {}
    iterator begin() const { return begin_; }
    iterator end() const { return end_; }

  private:
    iterator begin_;
    iterator end_;
};

template<> inline JsonObject JsonDocument::as<JsonObject>() const {
    return JsonObject(const_cast<JsonDocument *>(this));
}
