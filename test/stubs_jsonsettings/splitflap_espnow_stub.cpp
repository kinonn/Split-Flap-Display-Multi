// Link-time stubs for symbols SplitFlapMqtt.cpp references but never executes
// on the host (espNow is nullptr in these tests; the display is not driven).
// SplitFlapEspNow.cpp / SplitFlapDisplay.cpp need the ESP-IDF WiFi stack and
// real I2C, and are not host-compilable.
#include "SplitFlapDisplay.h"
#include "SplitFlapEspNow.h"

void SplitFlapEspNow::distributeMessage(const String &, bool, unsigned long, int) {}

int SplitFlapEspNow::getTotalModuleCount() { return 0; }

void SplitFlapDisplay::writeString(String, float, bool, unsigned long, int, bool) {}
