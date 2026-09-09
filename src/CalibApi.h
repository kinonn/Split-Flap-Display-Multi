#pragma once

// Shared calibration constants for the vision-guided auto-calibration API
// (see tools/calib-agent/PRODUCTION.md and src/web/calib-contract.json).
//
// Deliberately Arduino-free so it can be unit-tested on the host without
// stubs. The JSON contract file duplicates these values for the AI agent —
// keep the two in sync (CI checks the drum tables against
// SplitFlapModule).

#define CALIB_HOLD_MODE 4
#define CALIB_MAX_MODULES 8
#define CALIB_MAX_GROUPS 6
#define CALIB_MAX_FRAME 48 // max total modules across a fleet (6 groups x 8)
#define CALIB_CHAR_OFFSET_MIN -32
#define CALIB_CHAR_OFFSET_MAX 32
#define CALIB_CONTRACT_VERSION 1

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
