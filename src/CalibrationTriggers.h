#pragma once

#include <vector>

// Single decision point for "does this settings save change calibration?"
// (see POST /settings in SplitFlapWebServer.cpp).
//
// Homing triggers used to be scattered and inconsistent: moduleOffsets and
// charOffsets were compared as raw CSV strings (so a whitespace-only edit
// scheduled a pointless reload cycle), while magnetPosition — which feeds
// the exact same magnet-target computation as displayOffset — triggered
// nothing at all. Both paths now funnel through numeric snapshots:
//
//   - the web handler builds a stored snapshot (from settings) and an
//     incoming snapshot (stored, overlaid with the parsed POST fields) and
//     calls calibrationChanged();
//   - SplitFlapDisplay::reloadOffsets() re-reads the same four keys and
//     calls affectedModules() to re-home only the modules that moved.
//
// Deliberately Arduino-free (same pattern as StepMath.h) so it can be
// unit-tested on the host without stubs.

// Live values of every setting that affects module positioning.
struct CalibrationSnapshot
{
    std::vector<int> moduleOffsets;
    std::vector<std::vector<int>> charOffsets;
    int displayOffset = 0;
    int magnetPosition = 0;
};

// True when saving `incoming` over `stored` changes any calibration value.
// Vector comparison is numeric, so CSV formatting differences ("0,0" vs
// "0, 0,") compare equal and never schedule a reload by themselves.
inline bool calibrationChanged(const CalibrationSnapshot &stored, const CalibrationSnapshot &incoming) {
    return stored.displayOffset != incoming.displayOffset || stored.magnetPosition != incoming.magnetPosition ||
        stored.moduleOffsets != incoming.moduleOffsets || stored.charOffsets != incoming.charOffsets;
}

// Per-module re-home set for a calibration move from `previous` to `current`
// (1 = must re-home, 0 = untouched). displayOffset and magnetPosition shift
// every module's magnet target, so either one changing affects all modules;
// otherwise only modules whose own moduleOffsets entry or charOffsets row
// changed.
inline std::vector<char>
affectedModules(const CalibrationSnapshot &previous, const CalibrationSnapshot &current, int numModules) {
    std::vector<char> affected(numModules, 0);
    if (previous.displayOffset != current.displayOffset || previous.magnetPosition != current.magnetPosition) {
        for (int i = 0; i < numModules; i++) {
            affected[i] = 1;
        }
        return affected;
    }
    static const std::vector<int> kEmptyRow;
    for (int i = 0; i < numModules; i++) {
        int oldModule = i < (int) previous.moduleOffsets.size() ? previous.moduleOffsets[i] : 0;
        int newModule = i < (int) current.moduleOffsets.size() ? current.moduleOffsets[i] : 0;
        if (oldModule != newModule) {
            affected[i] = 1;
            continue;
        }
        const std::vector<int> &oldRow = i < (int) previous.charOffsets.size() ? previous.charOffsets[i] : kEmptyRow;
        const std::vector<int> &newRow = i < (int) current.charOffsets.size() ? current.charOffsets[i] : kEmptyRow;
        size_t cols = oldRow.size() > newRow.size() ? oldRow.size() : newRow.size();
        for (size_t j = 0; j < cols; j++) {
            int oldVal = j < oldRow.size() ? oldRow[j] : 0;
            int newVal = j < newRow.size() ? newRow[j] : 0;
            if (oldVal != newVal) {
                affected[i] = 1;
                break;
            }
        }
    }
    return affected;
}
