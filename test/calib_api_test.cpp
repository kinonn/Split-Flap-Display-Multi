// Host tests for the pure calibration-API rules shared with the firmware
// (src/CalibApi.h). Includes the REAL production header; no re-implementation.
//
// Build & run (no Arduino required):
//   g++ -std=c++17 -Wall -Wextra -I src test/calib_api_test.cpp -o
//   /tmp/calib_api_test && /tmp/calib_api_test
//
// Why these rules exist (both guard silent failures found in review):
//   - kinonn-bot#38: /api/calib/preview-batch used to accept any slice size,
//     forward only the first 8 nudges per remote group (one ESP-NOW preview
//     packet carries 8) and drop the rest while still reporting the full
//     count. The handler now refuses such a batch, using
//     calibNudgesFitScope().
//   - kinonn-bot#37: a group in calibration hold dropped every incoming text
//     frame, including the master's fleet show frames, so remote groups kept
//     stale glyphs while the master reported settled. calibTextAllowed() is
//     the rule the ESP-NOW text path now uses: only the pinned master's text
//     may own a held group's display.

#include "CalibApi.h"

#include <cstdio>

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

int main() {
    // --- preview-batch per-scope limit ---------------------------------------
    // The local group is applied in-process and bounded only by the batch
    // total (48), which the HTTP handler checks separately.
    CHECK(calibNudgesFitScope(1, 1));
    CHECK(calibNudgesFitScope(1, 48));
    CHECK(calibNudgesFitScope(0, 48)); // defensive: anything <= 1 is local

    // Remote groups: a slice of 8 fits the ESP-NOW preview packet, 9 must be
    // refused rather than silently truncated.
    for (int scope = 2; scope <= CALIB_MAX_GROUPS; scope++) {
        CHECK(calibNudgesFitScope(scope, 1));
        CHECK(calibNudgesFitScope(scope, CALIB_MAX_NUDGES_PER_REMOTE));
        CHECK(! calibNudgesFitScope(scope, CALIB_MAX_NUDGES_PER_REMOTE + 1));
        CHECK(! calibNudgesFitScope(scope, 48));
    }

    // --- hold-mode write rule (issues #37 / #42) ------------------------------
    // Under hold, a frame from the pinned master is written even when the text
    // is unchanged: a per-boot de-duplication would otherwise let the group ack
    // a frame it is not showing (its own clock/date writer may have owned the
    // display since the last identical frame).
    CHECK(calibTextNeedsWrite(true, 0, false));                // new text, not held
    CHECK(calibTextNeedsWrite(true, CALIB_HOLD_MODE, true));   // new text, held
    CHECK(calibTextNeedsWrite(false, CALIB_HOLD_MODE, true));  // repeat frame from the pinned master while held
    CHECK(! calibTextNeedsWrite(false, 0, true));              // repeat frame, not held: old de-duplication
    CHECK(! calibTextNeedsWrite(false, 2, false));
    CHECK(! calibTextNeedsWrite(false, CALIB_HOLD_MODE, false));

    // --- hold-mode text rule -------------------------------------------------
    // Held AND from the pinned master: accepted, or fleet calibration shows
    // can never reach a held group.
    CHECK(calibTextAllowed(CALIB_HOLD_MODE, true));
    // Held and from anyone else: dropped — the guard's original purpose (a
    // non-held master's clock/date/scroll pushes must not overwrite the
    // frame the agent owns).
    CHECK(! calibTextAllowed(CALIB_HOLD_MODE, false));
    // Not held: text is accepted from anyone, as before.
    CHECK(calibTextAllowed(0, false));
    CHECK(calibTextAllowed(2, false));
    CHECK(calibTextAllowed(3, false));
    CHECK(calibTextAllowed(5, false));
    CHECK(calibTextAllowed(7, false)); // 7 = ESP_NOW_REMOTE_MODE (SplitFlapEspNow.h)
    CHECK(calibTextAllowed(CALIB_HOLD_MODE - 1, false));
    CHECK(calibTextAllowed(CALIB_HOLD_MODE + 1, false));

    if (failures == 0) {
        std::printf("calib_api_test: all %d checks passed\n", checks);
        return 0;
    }
    std::printf("calib_api_test: %d/%d checks FAILED\n", failures, checks);
    return 1;
}
