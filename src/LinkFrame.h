#pragma once

// HDLC-lite framing shared by both wired inter-display transports (RS-485
// and direct UART). Wire format, one frame:
//
//   [0x7E DST SRC TYPE LEN PAYLOAD CRC16-CCITT]
//
// DST/SRC/TYPE/LEN/CRC bytes equal to 0x7E or 0x7D are escaped as
// 0x7D + (byte ^ 0x20). CRC covers DST+SRC+TYPE+LEN+PAYLOAD (poly 0x1021,
// init 0xFFFF). Every frame carries destination AND source: the master
// sends DST=group slot with SRC=0, remotes reply DST=0 with SRC=their slot.
// Unicast only: address 0 = master, 1..5 = remote group index (=
// groupIndex); 0xFF is reserved and never sent. The explicit SRC exists
// because RS-485 local echo and multidrop overhearing make source inference
// from a destination-only header impossible (a reply addressed to 0 is
// otherwise indistinguishable from any other frame).
//
// TYPE identifies which SplitFlapEspNow payload struct the PAYLOAD bytes
// carry — the structs are reused verbatim, so there is no v2 wire format
// and no second struct definition to drift. Largest payload is the 51-byte
// char-offsets packet; LINK_MAX_PAYLOAD (64) leaves headroom.
//
// Arduino-free so host tests compile the real header (StepMath.h pattern).
// C++11-clean: must also compile under the ESP32 Arduino default dialect.
#include <stddef.h>
#include <stdint.h>

#define LINK_SYNC 0x7E
#define LINK_ESC 0x7D
#define LINK_ESC_XOR 0x20
#define LINK_MAX_PAYLOAD 64

#define LINK_TYPE_TEXT 1
#define LINK_TYPE_OFFSETS_PUSH 2
#define LINK_TYPE_CHAR_OFFSETS_PUSH 3
#define LINK_TYPE_OFFSETS_REPORT 4
#define LINK_TYPE_CHAR_OFFSETS_REPORT 5
#define LINK_TYPE_ANNOUNCE 6
#define LINK_TYPE_ACK 7

#define LINK_ADDR_MASTER 0
#define LINK_ADDR_BROADCAST 0xFF

inline uint16_t linkCrc16(const uint8_t *data, size_t len) {
    uint16_t crc = 0xFFFF;
    for (size_t i = 0; i < len; i++) {
        crc ^= (uint16_t) data[i] << 8;
        for (int b = 0; b < 8; b++) {
            crc = (crc & 0x8000) ? (uint16_t) ((crc << 1) ^ 0x1021) : (uint16_t) (crc << 1);
        }
    }
    return crc;
}

// Encodes one frame into out. Worst case is every byte escaped:
// outSize must be >= 13 + 2 * payloadLen. Returns the encoded length,
// or 0 when the payload is oversized or out is too small.
inline size_t linkEncode(
    uint8_t dst, uint8_t src, uint8_t type, const uint8_t *payload, size_t payloadLen, uint8_t *out, size_t outSize
) {
    if (payloadLen > LINK_MAX_PAYLOAD) {
        return 0;
    }
    if (outSize < 13 + 2 * payloadLen) {
        return 0;
    }
    if (payloadLen > 0 && payload == nullptr) {
        return 0;
    }

    uint8_t raw[4 + LINK_MAX_PAYLOAD + 2];
    raw[0] = dst;
    raw[1] = src;
    raw[2] = type;
    raw[3] = (uint8_t) payloadLen;
    for (size_t i = 0; i < payloadLen; i++) {
        raw[4 + i] = payload[i];
    }
    uint16_t crc = linkCrc16(raw, 4 + payloadLen);
    raw[4 + payloadLen] = (uint8_t) (crc >> 8);
    raw[4 + payloadLen + 1] = (uint8_t) (crc & 0xFF);

    size_t rawLen = 4 + payloadLen + 2;
    size_t pos = 0;
    out[pos++] = LINK_SYNC;
    for (size_t i = 0; i < rawLen; i++) {
        uint8_t b = raw[i];
        if (b == LINK_SYNC || b == LINK_ESC) {
            out[pos++] = LINK_ESC;
            out[pos++] = (uint8_t) (b ^ LINK_ESC_XOR);
        } else {
            out[pos++] = b;
        }
    }
    return pos;
}

// Streaming decoder: UART delivers a byte stream with no packet boundaries,
// so feed() takes one byte at a time and returns true exactly once per
// complete valid frame. Corrupt frames (CRC mismatch, oversize LEN) are
// dropped silently; a raw 0x7E mid-frame re-syncs to the new frame.
class LinkDecoder {
  public:
    LinkDecoder() { reset(); }

    void reset() {
        state = WAIT_SYNC;
        pos = 0;
        escaped = false;
        frameReady = false;
    }

    bool feed(uint8_t byte) {
        frameReady = false;
        if (state == WAIT_SYNC) {
            if (byte == LINK_SYNC) {
                state = IN_FRAME;
                pos = 0;
                escaped = false;
            }
            return false;
        }
        // IN_FRAME: a raw flag is always a frame boundary. The encoder escapes
        // every 0x7E data byte, so 7D 7E never occurs in valid output — a raw
        // flag after a corrupt lone ESC restarts framing and clears the
        // escape latch instead of storing a phantom 0x5E data byte.
        if (byte == LINK_SYNC) {
            pos = 0;
            escaped = false;
            return false;
        }
        if (! escaped && byte == LINK_ESC) {
            escaped = true;
            return false;
        }
        uint8_t b = escaped ? (uint8_t) (byte ^ LINK_ESC_XOR) : byte;
        escaped = false;
        if (pos >= sizeof(buf)) {
            // Should be unreachable (LEN gate below caps the frame), but
            // never overflow the buffer — drop and resync.
            reset();
            return false;
        }
        buf[pos++] = b;
        if (pos == 4) {
            if (buf[3] > LINK_MAX_PAYLOAD) {
                reset();
                return false;
            }
        }
        if (pos >= 4 && pos == (size_t) (4 + buf[3] + 2)) {
            uint16_t got = (uint16_t) ((buf[pos - 2] << 8) | buf[pos - 1]);
            if (linkCrc16(buf, pos - 2) == got) {
                frameReady = true;
                state = WAIT_SYNC;
                return true;
            }
            reset();
            return false;
        }
        return false;
    }

    bool ready() const { return frameReady; }
    uint8_t dst() const { return buf[0]; }
    uint8_t src() const { return buf[1]; }
    uint8_t type() const { return buf[2]; }
    const uint8_t *payload() const { return buf + 4; }
    size_t payloadLen() const { return buf[3]; }

  private:
    enum State {
        WAIT_SYNC,
        IN_FRAME
    };
    State state;
    bool escaped;
    bool frameReady;
    // DST + SRC + TYPE + LEN + max payload + CRC16.
    uint8_t buf[4 + LINK_MAX_PAYLOAD + 2];
    size_t pos;
};
