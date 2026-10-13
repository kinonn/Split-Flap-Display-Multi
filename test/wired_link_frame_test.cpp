// Host test for LinkFrame.h — the HDLC-lite frame codec shared by the wired
// inter-display transports. Includes the REAL production header; no
// re-implementation.
//
// Build & run (no Arduino required):
//   g++ -std=c++17 -Wall -Wextra -I src test/wired_link_frame_test.cpp -o /tmp/wired_link_frame_test && /tmp/wired_link_frame_test

#include "LinkFrame.h"

#include <cstdio>
#include <cstring>

static int failures = 0;
static int checks = 0;

#define CHECK(cond)                                                     \
    do {                                                                \
        checks++;                                                       \
        if (! (cond)) {                                                 \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            failures++;                                                 \
        }                                                               \
    } while (0)

// Zero-payload announce frame: [SYNC DST SRC TYPE LEN CRC CRC] = 7 bytes.
static void test_empty_payload_round_trip() {
    uint8_t out[32];
    size_t n = linkEncode(1, 0, LINK_TYPE_ANNOUNCE, nullptr, 0, out, sizeof(out));
    CHECK(n == 7);
    CHECK(out[0] == LINK_SYNC);

    LinkDecoder dec;
    bool got = false;
    for (size_t i = 0; i < n; i++) {
        got = dec.feed(out[i]);
    }
    CHECK(got);
    CHECK(dec.dst() == 1);
    CHECK(dec.src() == 0);
    CHECK(dec.type() == LINK_TYPE_ANNOUNCE);
    CHECK(dec.payloadLen() == 0);
}

// 12-byte text payload (the real SplitFlapEspNowMessage size) round-trips
// with identical bytes.
static void test_text_payload_round_trip() {
    const uint8_t payload[12] = {'1', 1, 8, 'H', 'e', 'l', 'l', 'o', ' ', ' ', 0, 0};
    uint8_t out[64];
    size_t n = linkEncode(2, 0, LINK_TYPE_TEXT, payload, sizeof(payload), out, sizeof(out));
    CHECK(n == 1 + 4 + sizeof(payload) + 2);

    LinkDecoder dec;
    bool got = false;
    for (size_t i = 0; i < n; i++) {
        got = dec.feed(out[i]);
    }
    CHECK(got);
    CHECK(dec.dst() == 2);
    CHECK(dec.src() == 0);
    CHECK(dec.type() == LINK_TYPE_TEXT);
    CHECK(dec.payloadLen() == sizeof(payload));
    CHECK(std::memcmp(dec.payload(), payload, sizeof(payload)) == 0);
}

// Payload bytes equal to SYNC/ESC must be escaped on the wire (no raw 0x7E
// after position 0) and restored exactly on decode.
static void test_escape_round_trip() {
    const uint8_t payload[6] = {0x7E, 0x7D, 0x00, 0xFF, 0x7E, 0x7D};
    uint8_t out[64];
    size_t n = linkEncode(3, 1, LINK_TYPE_ACK, payload, sizeof(payload), out, sizeof(out));
    // 4 payload bytes escape (+4); header/CRC bytes may also escape, so pin
    // a range, not an exact length.
    CHECK(n >= 1 + 4 + 6 + 2 + 4);
    CHECK(n <= 1 + 2 * (4 + 6 + 2));
    for (size_t i = 1; i < n; i++) {
        CHECK(out[i] != LINK_SYNC);
    }

    LinkDecoder dec;
    bool got = false;
    for (size_t i = 0; i < n; i++) {
        got = dec.feed(out[i]);
    }
    CHECK(got);
    CHECK(dec.payloadLen() == sizeof(payload));
    CHECK(std::memcmp(dec.payload(), payload, sizeof(payload)) == 0);
}

// A single flipped payload bit must fail the CRC and yield no frame.
static void test_crc_mismatch_rejected() {
    const uint8_t payload[4] = {'A', 'B', 'C', 'D'};
    uint8_t out[32];
    size_t n = linkEncode(1, 0, LINK_TYPE_TEXT, payload, sizeof(payload), out, sizeof(out));
    CHECK(n > 0);
    out[5] ^= 0x01;  // corrupt first payload byte (layout SYNC/DST/SRC/TYPE/LEN/P...; unescaped ASCII here)

    LinkDecoder dec;
    bool got = false;
    for (size_t i = 0; i < n; i++) {
        got = dec.feed(out[i]);
    }
    CHECK(! got);
}

// Streaming: the same frame fed one byte at a time, and in 3 uneven
// chunks, decodes identically. UART has no packet boundaries, so this is
// the production delivery shape.
static void test_split_delivery() {
    const uint8_t payload[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    uint8_t out[64];
    size_t n = linkEncode(4, 0, LINK_TYPE_OFFSETS_PUSH, payload, sizeof(payload), out, sizeof(out));
    CHECK(n > 0);

    LinkDecoder bytewise;
    bool got = false;
    for (size_t i = 0; i < n; i++) {
        if (bytewise.feed(out[i])) {
            got = true;
        }
    }
    CHECK(got);
    CHECK(bytewise.payloadLen() == sizeof(payload));

    LinkDecoder chunked;
    got = false;
    size_t cuts[4] = {0, 2, 5, n};
    for (int c = 0; c < 3; c++) {
        for (size_t i = cuts[c]; i < cuts[c + 1]; i++) {
            if (chunked.feed(out[i])) {
                got = true;
            }
        }
    }
    CHECK(got);
    CHECK(std::memcmp(chunked.payload(), payload, sizeof(payload)) == 0);
}

// Oversize LEN aborts the frame without allocation: encode refuses, and a
// hand-crafted LEN=65 header yields no frame and resyncs to the next SYNC.
static void test_oversize_rejected() {
    uint8_t big[LINK_MAX_PAYLOAD + 1];
    std::memset(big, 'x', sizeof(big));
    uint8_t out[256];
    CHECK(linkEncode(1, 0, LINK_TYPE_TEXT, big, sizeof(big), out, sizeof(out)) == 0);

    uint8_t maxPayload[LINK_MAX_PAYLOAD];
    std::memset(maxPayload, 'y', sizeof(maxPayload));
    CHECK(linkEncode(1, 0, LINK_TYPE_TEXT, maxPayload, sizeof(maxPayload), out, sizeof(out)) > 0);

    LinkDecoder dec;
    CHECK(! dec.feed(LINK_SYNC));
    CHECK(! dec.feed(1));
    CHECK(! dec.feed(0));
    CHECK(! dec.feed(LINK_TYPE_TEXT));
    CHECK(! dec.feed(LINK_MAX_PAYLOAD + 1));  // LEN gate trips here
    // Decoder resynced: a fresh valid frame decodes immediately after.
    uint8_t good[32];
    size_t n = linkEncode(5, 0, LINK_TYPE_ACK, nullptr, 0, good, sizeof(good));
    bool got = false;
    for (size_t i = 0; i < n; i++) {
        got = dec.feed(good[i]);
    }
    CHECK(got);
    CHECK(dec.dst() == 5);
    CHECK(dec.src() == 0);
}

// Leading line noise and a raw mid-frame flag must not produce a phantom
// frame; the decoder resyncs to the next real SYNC.
static void test_resync() {
    uint8_t out[32];
    size_t n = linkEncode(2, 0, LINK_TYPE_TEXT, (const uint8_t *) "AB", 2, out, sizeof(out));

    LinkDecoder dec;
    CHECK(! dec.feed(0x00));
    CHECK(! dec.feed(0xFF));
    bool got = false;
    for (size_t i = 0; i < n; i++) {
        got = dec.feed(out[i]);
    }
    CHECK(got);
    CHECK(dec.dst() == 2);
    CHECK(dec.src() == 0);

    // Raw flag mid-frame aborts the partial frame; the following complete
    // frame still decodes.
    LinkDecoder dec2;
    CHECK(! dec2.feed(LINK_SYNC));
    CHECK(! dec2.feed(9));
    CHECK(! dec2.feed(LINK_SYNC));  // abort, restart
    got = false;
    for (size_t i = 1; i < n; i++) {  // frame body after a SYNC
        got = dec2.feed(out[i]);
    }
    CHECK(got);
    CHECK(dec2.dst() == 2);
}

// Both directions carry explicit addresses: a master-to-remote frame reads
// DST=slot/SRC=0, and the reply reads DST=0/SRC=slot. This is the property
// the direction filter in SplitFlapWired::handleFrame depends on.
static void test_src_dst_round_trip() {
    uint8_t out[32];
    size_t n = linkEncode(3, 0, LINK_TYPE_TEXT, (const uint8_t *) "AB", 2, out, sizeof(out));

    LinkDecoder toRemote;
    bool got = false;
    for (size_t i = 0; i < n; i++) {
        got = toRemote.feed(out[i]);
    }
    CHECK(got);
    CHECK(toRemote.dst() == 3);
    CHECK(toRemote.src() == 0);

    uint8_t back[32];
    size_t m = linkEncode(0, 3, LINK_TYPE_ACK, nullptr, 0, back, sizeof(back));
    LinkDecoder toMaster;
    got = false;
    for (size_t i = 0; i < m; i++) {
        got = toMaster.feed(back[i]);
    }
    CHECK(got);
    CHECK(toMaster.dst() == 0);
    CHECK(toMaster.src() == 3);
    CHECK(toMaster.type() == LINK_TYPE_ACK);
}

int main() {
    test_empty_payload_round_trip();
    test_text_payload_round_trip();
    test_escape_round_trip();
    test_crc_mismatch_rejected();
    test_split_delivery();
    test_oversize_rejected();
    test_resync();
    test_src_dst_round_trip();

    std::printf("%d checks, %d failures\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
