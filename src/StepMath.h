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

// Settings guards: JsonSettings accepts any int/float (only vector/matrix
// values are character-validated), and moveTo() divides by both
// stepsPerRot and maxVel (timePerStep = 1000000 / stepsPerSecond). A stored
// 0 or negative value therefore produced a division by zero → timePerStep =
// inf → the move loop never terminated → task watchdog reset. These collapse
// nonsense values to the firmware defaults (see SplitFlapDisplay.ino).
inline int sanitizeStepsPerRot(int stepsPerRot) {
    return stepsPerRot > 0 ? stepsPerRot : 2048;
}

inline float sanitizeMaxVel(float maxVel) {
    return maxVel > 0.0f ? maxVel : 15.0f;
}
