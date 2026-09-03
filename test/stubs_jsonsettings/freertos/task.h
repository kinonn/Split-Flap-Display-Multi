// Host stub: portMUX_TYPE / critical-section macros as no-ops on the host
// (single-threaded test paths never race; the threaded PendingActions test
// uses std::atomic, not portMUX).
#pragma once

typedef uint32_t portMUX_TYPE;

#define portMUX_INITIALIZER_UNLOCKED 0
#define portENTER_CRITICAL(mux) ((void) (mux))
#define portEXIT_CRITICAL(mux) ((void) (mux))
