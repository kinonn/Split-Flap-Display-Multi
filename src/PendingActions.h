#pragma once

#include <atomic>

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

  private:
    std::atomic<bool> reloadOffsets_{false};
    std::atomic<bool> reportOffsets_{false};
    std::atomic<bool> pushOffsets_{false};
};
