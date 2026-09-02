#pragma once

// Forward step distance on a unidirectional (forward-only) character drum,
// in [0, stepsPerRot). Used by SplitFlapDisplay::moveTo() to track how many
// steps each module still owes — counting steps instead of comparing
// positions for equality stays correct when a mid-move magnet correction
// snaps `position` past the target.
//
// Deliberately Arduino-free so it can be unit-tested on the host without
// stubs (same pattern as StatusPayload.h).
inline int forwardSteps(int from, int to, int stepsPerRot) {
    return ((to - from) % stepsPerRot + stepsPerRot) % stepsPerRot;
}
