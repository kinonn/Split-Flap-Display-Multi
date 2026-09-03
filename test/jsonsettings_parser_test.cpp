// Host tests for the <sstream>-free integer-list parser in JsonSettings.
//
// These compile the REAL src/JsonSettings.cpp against the stubs and drive the
// parser through the public put/get API. Raw malformed strings are written via
// putString() (which bypasses JsonSetting::validate, mirroring values that can
// reach NVS from an older firmware or a hand-edit).
//
// Contract under test:
//   - Well-formed input parses identically to the old std::getline/std::stoi
//     implementation (round-trips through put* must be exact).
//   - Whitespace around numbers and separators is tolerated.
//   - Trailing/leading/duplicate separators and empty entries yield no values.
//   - Junk never throws and never hangs; parsing continues past it. Values
//     beyond INT range clamp to 0 in-place (position preserved) instead of
//     the old std::stoi throw.
//   - Matrix rows split on ';' exactly like the old std::getline loop: a
//     separator emits the row before it even when empty, an empty tail after
//     the last ';' emits nothing, "" yields zero rows — so row indexing stays
//     positionally stable for well-formed data.
//   - No input can make the parser throw: since the lenient parser replaced
//     the throwing std::stoi path, callers (SplitFlapDisplay) rely on
//     bounds-checking instead of exceptions for length safety.
//
// Build & run (no Arduino required):
//   g++ -std=c++17 -Wall -Wextra -pthread -I src -I test/stubs_jsonsettings -o /tmp/jsp
//     test/jsonsettings_parser_test.cpp src/JsonSettings.cpp src/JsonSetting.cpp && /tmp/jsp

#include <cstdio>
#include <map>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "Arduino.h"
#include "ArduinoJson.h"
#include "JsonSettings.h"
#include "Preferences.h"

std::vector<std::string> g_serialLines;
std::map<std::pair<std::string, std::string>, std::string> g_nvs;
std::mutex g_nvsMutex;
std::atomic<int> g_openHandles{0};

SerialStub Serial;

static int failures = 0;
static int checks = 0;

#define CHECK(cond)                                                     \
    do {                                                                \
        checks++;                                                       \
        if (!(cond)) {                                                  \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            failures++;                                                 \
        }                                                               \
    } while (0)

static std::map<String, JsonSetting> testSchema() {
    return {
        {"moduleOffsets", JsonSetting(std::vector<int>{0, 0, 0, 0, 0, 0, 0, 0})},
        {"charOffsets", JsonSetting(std::vector<std::vector<int>>{{0, 0, 0, 0}})},
    };
}

static JsonSettings makeSettings() {
    return JsonSettings("parsertest", testSchema());
}

static void nvsReset() {
    g_nvs.clear();
    g_openHandles = 0;
}

static bool sameVector(const std::vector<int> &got, const std::vector<int> &want) {
    return got == want;
}

static bool sameMatrix(
    const std::vector<std::vector<int>> &got, const std::vector<std::vector<int>> &want
) {
    return got == want;
}

// ---------------------------------------------------------------------------
// Vectors
// ---------------------------------------------------------------------------

static void test_vector_roundtrip() {
    nvsReset();
    JsonSettings settings = makeSettings();

    settings.putIntVector("moduleOffsets", std::vector<int>{10, -20, 30});
    CHECK(sameVector(settings.getIntVector("moduleOffsets"), {10, -20, 30}));

    settings.putIntVector("moduleOffsets", std::vector<int>{});
    CHECK(sameVector(settings.getIntVector("moduleOffsets"), {}));
}

static void test_vector_spaces_around_numbers() {
    nvsReset();
    JsonSettings settings = makeSettings();

    settings.putString("moduleOffsets", "1, 2,  3\t, 4");
    CHECK(sameVector(settings.getIntVector("moduleOffsets"), {1, 2, 3, 4}));

    settings.putString("moduleOffsets", "  7  ");
    CHECK(sameVector(settings.getIntVector("moduleOffsets"), {7}));
}

static void test_vector_trailing_and_empty_entries() {
    nvsReset();
    JsonSettings settings = makeSettings();

    settings.putString("moduleOffsets", "1,2,3,");
    CHECK(sameVector(settings.getIntVector("moduleOffsets"), {1, 2, 3}));

    settings.putString("moduleOffsets", ",5,,6,");
    CHECK(sameVector(settings.getIntVector("moduleOffsets"), {5, 6}));

    settings.putString("moduleOffsets", ",,");
    CHECK(sameVector(settings.getIntVector("moduleOffsets"), {}));
}

static void test_vector_empty_and_blank_string() {
    nvsReset();
    JsonSettings settings = makeSettings();

    settings.putString("moduleOffsets", "");
    CHECK(sameVector(settings.getIntVector("moduleOffsets"), {}));

    settings.putString("moduleOffsets", "   ");
    CHECK(sameVector(settings.getIntVector("moduleOffsets"), {}));
}

static void test_vector_junk_is_skipped_not_fatal() {
    nvsReset();
    JsonSettings settings = makeSettings();

    // Old std::stoi path threw std::runtime_error for these; the parser now
    // skips junk and keeps going. No exception may escape.
    settings.putString("moduleOffsets", "1,x,3");
    CHECK(sameVector(settings.getIntVector("moduleOffsets"), {1, 3}));

    settings.putString("moduleOffsets", "abc");
    CHECK(sameVector(settings.getIntVector("moduleOffsets"), {}));

    settings.putString("moduleOffsets", "12ab34");
    CHECK(sameVector(settings.getIntVector("moduleOffsets"), {12, 34}));
}

static void test_vector_out_of_range_clamps_to_zero() {
    nvsReset();
    JsonSettings settings = makeSettings();

    // The old std::stoi path threw std::out_of_range for these; the lenient
    // parser clamps to 0 in-place so entry order (module position) survives.
    settings.putString("moduleOffsets", "99999999999999999999999,7");
    CHECK(sameVector(settings.getIntVector("moduleOffsets"), {0, 7}));

    settings.putString("moduleOffsets", "1,99999999999999999999999");
    CHECK(sameVector(settings.getIntVector("moduleOffsets"), {1, 0}));
}

static void test_vector_never_throws_on_adversarial_input() {
    nvsReset();
    JsonSettings settings = makeSettings();

    const char *cases[] = {
        "-5,-0", "-,+", "+++1", "0x1F", "--5", "9223372036854775807",
        "99999999999999999999999999", "1;2", "1.5,2.5", "\1\2\3", "+7,-8",
        "1,\n2,\r3", "00012", "-00003",
    };
    for (const char *input : cases) {
        settings.putString("moduleOffsets", input);
        try {
            std::vector<int> v = settings.getIntVector("moduleOffsets");
            (void) v;
            CHECK(true); // parsed (any result is fine) — the point is no throw
        } catch (...) {
            CHECK(false); // parser must never throw
            std::printf("  threw on input: %s\n", input);
        }
    }
}

static void test_vector_negative_numbers() {
    nvsReset();
    JsonSettings settings = makeSettings();

    settings.putString("moduleOffsets", "-5,-0,+7");
    CHECK(sameVector(settings.getIntVector("moduleOffsets"), {-5, 0, 7}));
}

// ---------------------------------------------------------------------------
// Matrices
// ---------------------------------------------------------------------------

static void test_matrix_roundtrip() {
    nvsReset();
    JsonSettings settings = makeSettings();

    settings.putIntMatrix("charOffsets", std::vector<std::vector<int>>{{1, 2}, {3, 4}});
    CHECK(sameMatrix(
        settings.getIntMatrix("charOffsets"), std::vector<std::vector<int>>{{1, 2}, {3, 4}}
    ));

    settings.putIntMatrix("charOffsets", std::vector<std::vector<int>>{{}});
    // Note: {{}} serializes to "" and "" parses to zero rows, so this
    // round-trip is lossy for a single empty row — same as the old code.
    CHECK(sameMatrix(settings.getIntMatrix("charOffsets"), std::vector<std::vector<int>>{}));
}

static void test_matrix_rows_split_on_semicolon() {
    nvsReset();
    JsonSettings settings = makeSettings();

    settings.putString("charOffsets", "1,2;3,4;5");
    CHECK(sameMatrix(
        settings.getIntMatrix("charOffsets"),
        std::vector<std::vector<int>>{{1, 2}, {3, 4}, {5}}
    ));
}

static void test_matrix_empty_rows_keep_positions() {
    nvsReset();
    JsonSettings settings = makeSettings();

    // Rows are positionally stable for well-formed data: each ';' emits the
    // row before it; an empty tail after the last ';' emits nothing (same as
    // the old std::getline loop).
    settings.putString("charOffsets", ";1;;2");
    CHECK(sameMatrix(
        settings.getIntMatrix("charOffsets"),
        std::vector<std::vector<int>>{{}, {1}, {}, {2}}
    ));

    settings.putString("charOffsets", "");
    CHECK(sameMatrix(settings.getIntMatrix("charOffsets"), std::vector<std::vector<int>>{}));
}

static void test_matrix_junk_never_throws() {
    nvsReset();
    JsonSettings settings = makeSettings();

    const char *cases[] = {"a;b", "1,x;2,y", ";;;x;;", "  ;  3  ;  "};
    for (const char *input : cases) {
        settings.putString("charOffsets", input);
        try {
            std::vector<std::vector<int>> m = settings.getIntMatrix("charOffsets");
            (void) m;
            CHECK(true);
        } catch (...) {
            CHECK(false);
            std::printf("  threw on input: %s\n", input);
        }
    }
}

// ---------------------------------------------------------------------------
// Defaults still flow through the same parser
// ---------------------------------------------------------------------------

static void test_defaults_parse_through_new_path() {
    nvsReset();
    JsonSettings settings = makeSettings();

    // Schema default "0,0,0,0,0,0,0,0" and "{{0,0,0,0}}" must parse unchanged.
    CHECK(sameVector(settings.getIntVector("moduleOffsets"), {0, 0, 0, 0, 0, 0, 0, 0}));
    CHECK(sameMatrix(
        settings.getIntMatrix("charOffsets"), std::vector<std::vector<int>>{{0, 0, 0, 0}}
    ));
}

int main() {
    test_vector_roundtrip();
    test_vector_spaces_around_numbers();
    test_vector_trailing_and_empty_entries();
    test_vector_empty_and_blank_string();
    test_vector_junk_is_skipped_not_fatal();
    test_vector_never_throws_on_adversarial_input();
    test_vector_out_of_range_clamps_to_zero();
    test_vector_negative_numbers();
    test_matrix_roundtrip();
    test_matrix_rows_split_on_semicolon();
    test_matrix_empty_rows_keep_positions();
    test_matrix_junk_never_throws();
    test_defaults_parse_through_new_path();

    std::printf("%d checks, %d failures\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
