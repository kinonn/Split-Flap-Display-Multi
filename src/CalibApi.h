#pragma once

// Shared calibration constants for the vision-guided auto-calibration API
// (see tools/calib-agent/PRODUCTION.md and src/web/calib-contract.json).
//
// Deliberately Arduino-free so it can be unit-tested on the host without
// stubs. The JSON contract file duplicates these values for the AI agent —
// keep the two in sync: test/calib_contract_test.cpp compares the charsets
// and limits in src/web/calib-contract.json against SplitFlapModule's drum
// tables and these constants on the host.

#define CALIB_HOLD_MODE 4
#define CALIB_MAX_MODULES 8
#define CALIB_MAX_GROUPS 6
#define CALIB_MAX_FRAME 48 // max total modules across a fleet (6 groups x 8)
#define CALIB_CHAR_OFFSET_MIN -32
#define CALIB_CHAR_OFFSET_MAX 32
#define CALIB_CONTRACT_VERSION 1

// Per-scope limit for POST /api/calib/preview-batch. A remote group's slice
// travels in ONE ESP-NOW preview packet, which carries at most 8 nudges
// (SplitFlapPreviewNudgeMessage::nudges). A larger slice used to be
// forwarded only up to the 8th entry and silently dropped past that while
// the HTTP response still claimed the full count (issue kinonn-bot#38), so
// the API now refuses such a batch instead of losing nudges.
#define CALIB_MAX_NUDGES_PER_REMOTE 8

// Does a preview-batch scope slice of `count` nudges fit? Group 1 (local) is
// applied in-process and is bounded only by the batch total
// (PendingActions::CalibBatchPreview::MAX_NUDGES); remote scopes are bounded
// by the ESP-NOW packet above.
inline bool calibNudgesFitScope(int scope, int count) {
    if (scope <= 1) {
        return true;
    }
    return count <= CALIB_MAX_NUDGES_PER_REMOTE;
}

// May an incoming text frame own the display of a controller in `mode`?
// A held group (CALIB_HOLD_MODE) still has to accept text from the pinned
// master: fleet calibration shows arrive over exactly that ESP-NOW text
// path, and dropping them left remote groups showing stale glyphs while the
// master reported settled (issue kinonn-bot#37). Text from any other sender
// keeps being dropped, which is the guard's original purpose — a non-held
// master's clock/date/scroll pushes must not overwrite a held group's frame.
inline bool calibTextAllowed(int mode, bool fromPinnedMaster) {
    return fromPinnedMaster || mode != CALIB_HOLD_MODE;
}

// Pre-hold display mode tracker so hold release restores it (issue
// kinonn-bot#35). Plain value type with the exact engage/release rules;
// the owner (web handler) guards threading. Unit-tested on the host.
struct CalibHoldTracker
{
    // Engage: returns the mode to write (always CALIB_HOLD_MODE).
    // Remembers previousMode unless already holding — a double-engage
    // must not overwrite the saved mode with the hold mode itself.
    int engage(int previousMode) {
        if (previousMode != CALIB_HOLD_MODE) saved_ = previousMode;
        return CALIB_HOLD_MODE;
    }
    // Release: returns the mode to restore. Never-engaged-since-boot
    // (-1) falls back to 0, the historic behavior. Clears state so a
    // stale mode can never leak into a later hold cycle.
    int release() {
        int restore = (saved_ >= 0) ? saved_ : 0;
        saved_ = -1;
        return restore;
    }

  private:
    int saved_ = -1;
};

// Canonical coarse-alignment glyph set: characters with strong top/bottom
// horizontal features, most sensitive to vertical half-flap errors. Shown on
// ALL modules simultaneously (6 moves total) to estimate displayOffset first,
// then per-module moduleOffsets.
static const char kCalibCoarseGlyphs[] = {' ', 'E', 'H', 'O', '0', '-'};
static const int kCalibCoarseGlyphCount = 6;
