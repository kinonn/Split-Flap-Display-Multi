#pragma once

// Pure motor-concurrency scheduler used by SplitFlapDisplay::moveTo().
//
// Why it exists: at boot homing every module needs to rotate, and if all
// motors run at once the combined coil current (28BYJ-48: ~200-330 mA each)
// can collapse the shared 5 V rail on larger displays (11 modules ≈ 2.2-3.3 A
// before even considering the PCF8575 POR state). This scheduler caps how
// many motors may be energized simultaneously so homing runs in small
// batches instead of one all-motors spike.
//
// Deliberately free of Arduino dependencies so the exact logic is
// unit-testable on the host (see test/motor_scheduler_test.cpp). The
// firmware calls start()/finish() around the real stepping loop.
//
// Semantics:
//   - start(i)  grants motor i a slot if one is free (or the cap is
//               unlimited, cap < 0). Returns true when the motor may run.
//   - finish(i) releases motor i's slot once it reaches its target.
//   - start(i) on an already-running motor is a no-op returning true
//     (it keeps its slot; the active count is not double-counted).
//   - Out-of-range indices are rejected (return false / ignored).
class SplitFlapMotorScheduler {
  public:
    static const int kMaxModules = 8; // must match MAX_MODULES in SplitFlapDisplay.h

    SplitFlapMotorScheduler(int numModules, int maxConcurrent)
        : numModules_(numModules), maxConcurrent_(maxConcurrent < 0 ? -1 : (maxConcurrent < 1 ? 1 : maxConcurrent)),
          activeCount_(0) {
        for (int i = 0; i < kMaxModules; i++) {
            running_[i] = false;
        }
    }

    bool start(int i) {
        if (i < 0 || i >= numModules_ || i >= kMaxModules) {
            return false;
        }
        if (running_[i]) {
            return true;  // already has a slot
        }
        if (maxConcurrent_ >= 0 && activeCount_ >= maxConcurrent_) {
            return false; // cap reached — caller retries on a later tick
        }
        running_[i] = true;
        activeCount_++;
        return true;
    }

    void finish(int i) {
        if (i < 0 || i >= kMaxModules || i >= numModules_ || ! running_[i]) {
            return;
        }
        running_[i] = false;
        if (activeCount_ > 0) {
            activeCount_--;
        }
    }

    bool isRunning(int i) const { return i >= 0 && i < numModules_ && i < kMaxModules && running_[i]; }

    int activeCount() const { return activeCount_; }
    int numModules() const { return numModules_; }

  private:
    int numModules_;
    int maxConcurrent_;
    int activeCount_;
    bool running_[kMaxModules];
};
