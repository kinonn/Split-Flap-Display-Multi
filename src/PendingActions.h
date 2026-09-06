#pragma once

#include <atomic>
#include <mutex>
#include <string>

// Cross-task "do this in the loop task" mailbox.
//
// Why it exists: the AsyncTCP web task used to run display work inline —
// /settings POST called display.reloadOffsets() (seconds of Wire I/O with
// delays) and espNow->reportOffsetsToMaster()/pushOffsetsToGroup() (hundreds
// of ms of esp_now_send + delay) — racing the Arduino loop task's own display
// access (MQTT callback / singleInputMode -> writeString -> moveTo). Two
// owners of one I2C bus produced NACKs, lost steps and corrupt writes
// (audit issues #3 and #12).
//
// The web task now only sets flags; the loop task drains them via takeXxx()
// once per loop() and performs the work as the single owner of the display.
//
// takeXxx() is a consume-once atomic exchange, safe without a mutex: a
// request racing a take is either observed or re-observed on the next
// loop() pass (~ms later) — work is deferred, never lost for long.
class PendingActions {
  public:
    void requestReloadOffsets() { reloadOffsets_.store(true, std::memory_order_release); }
    void requestReportOffsets() { reportOffsets_.store(true, std::memory_order_release); }
    void requestPushOffsets() { pushOffsets_.store(true, std::memory_order_release); }

    bool takeReloadOffsets() { return reloadOffsets_.exchange(false); }
    bool takeReportOffsets() { return reportOffsets_.exchange(false); }
    bool takePushOffsets() { return pushOffsets_.exchange(false); }

    // Calibration show slot: the AsyncTCP web task stages an exact-width
    // frame (no centering, no scroll); the loop task drains it and performs
    // the I2C/ESP-NOW work as the single owner of the display. Single
    // producer / single consumer guarded by a mutex (payloads are strings,
    // not single words, so plain atomics are not enough). Latest request
    // wins if the agent outruns the display.
    struct CalibPreview
    {
        int module = 0;    // local module index 0..7
        int charIndex = 0; // drum index 0..charset-1, or -1 = coarse module offset
        int delta = 0;     // relative nudge in motor steps
    };

    void requestCalibShow(const std::string &frame, int frameId) {
        std::lock_guard<std::mutex> lock(calibMutex_);
        calibShowPending_ = true;
        calibShowFrame_ = frame;
        calibShowFrameId_ = frameId;
    }

    bool takeCalibShow(std::string &frameOut, int &frameIdOut) {
        std::lock_guard<std::mutex> lock(calibMutex_);
        if (! calibShowPending_) {
            return false;
        }
        calibShowPending_ = false;
        frameOut = calibShowFrame_;
        frameIdOut = calibShowFrameId_;
        return true;
    }

    bool hasCalibShowPending() {
        std::lock_guard<std::mutex> lock(calibMutex_);
        return calibShowPending_;
    }

    // Calibration preview slot: volatile RAM-only nudge (no NVS write). The
    // loop task applies it via SplitFlapDisplay::previewNudgeLocal(); a later
    // reloadOffsets() (from NVS) reverts it.
    void requestCalibPreview(const CalibPreview &preview) {
        std::lock_guard<std::mutex> lock(calibMutex_);
        calibPreviewPending_ = true;
        calibPreview_ = preview;
    }

    bool takeCalibPreview(CalibPreview &previewOut) {
        std::lock_guard<std::mutex> lock(calibMutex_);
        if (! calibPreviewPending_) {
            return false;
        }
        calibPreviewPending_ = false;
        previewOut = calibPreview_;
        return true;
    }

  private:
    std::atomic<bool> reloadOffsets_{false};
    std::atomic<bool> reportOffsets_{false};
    std::atomic<bool> pushOffsets_{false};

    std::mutex calibMutex_;
    bool calibShowPending_ = false;
    std::string calibShowFrame_;
    int calibShowFrameId_ = 0;
    bool calibPreviewPending_ = false;
    CalibPreview calibPreview_;
};
