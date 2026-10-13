#pragma once

// HDLC-lite framing shared by both wired inter-display transports (RS-485
// and direct UART). Wire format, one frame:
//
//   [0x7E ADDR TYPE LEN PAYLOAD CRC16-CCITT]
//
// ADDR/TYPE/LEN/CRC bytes equal to 0x7E or 0x7D are escaped as
// 0x7D + (byte ^ 0x20). CRC covers ADDR+TYPE+LEN+PAYLOAD (poly 0x1021,
// init 0xFFFF). Unicast only: ADDR 0 = master, 1..5 = remote group index
// (= groupIndex); 0xFF is reserved and never sent.
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
// outSize must be >= 11 + 2 * payloadLen. Returns the encoded length,
// or 0 when the payload is oversized or out is too small.
inline size_t
linkEncode(uint8_t addr, uint8_t type, const uint8_t *payload, size_t payloadLen, uint8_t *out, size_t outSize) {
    if (payloadLen > LINK_MAX_PAYLOAD) {
        return 0;
    }
    if (outSize < 11 + 2 * payloadLen) {
        return 0;
    }
    if (payloadLen > 0 && payload == nullptr) {
        return 0;
    }

    uint8_t raw[3 + LINK_MAX_PAYLOAD + 2];
    raw[0] = addr;
    raw[1] = type;
    raw[2] = (uint8_t) payloadLen;
    for (size_t i = 0; i < payloadLen; i++) {
        raw[3 + i] = payload[i];
    }
    uint16_t crc = linkCrc16(raw, 3 + payloadLen);
    raw[3 + payloadLen] = (uint8_t) (crc >> 8);
    raw[3 + payloadLen + 1] = (uint8_t) (crc & 0xFF);

    size_t rawLen = 3 + payloadLen + 2;
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
        // IN_FRAME
        if (! escaped && byte == LINK_SYNC) {
            // Raw flag mid-frame: previous frame (if any) was cut; the new
            // frame starts here.
            pos = 0;
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
        if (pos == 3) {
            if (buf[2] > LINK_MAX_PAYLOAD) {
                reset();
                return false;
            }
        }
        if (pos >= 3 && pos == (size_t) (3 + buf[2] + 2)) {
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
    uint8_t addr() const { return buf[0]; }
    uint8_t type() const { return buf[1]; }
    const uint8_t *payload() const { return buf + 3; }
    size_t payloadLen() const { return buf[2]; }

  private:
    enum State {
        WAIT_SYNC,
        IN_FRAME
    };
    State state;
    bool escaped;
    bool frameReady;
    // ADDR + TYPE + LEN + max payload + CRC16.
    uint8_t buf[3 + LINK_MAX_PAYLOAD + 2];
    size_t pos;
};
