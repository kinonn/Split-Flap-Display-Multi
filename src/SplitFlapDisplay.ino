// Split Flap Display
// Morgan Manly 02/16/2025
// Jordan Hoff 03/25/2025
// Thom Koopman 03/30/2025

// Enjoy :)
#include "JsonSettings.h"
#include "SplitFlapDisplay.h"
#include "SplitFlapEspNow.h"
#include "SplitFlapMqtt.h"
#include "SplitFlapWebServer.h"

#include <Arduino.h>
#include <WiFiClient.h>

// clang-format off
JsonSettings settings = JsonSettings("config", {
    // General Settings
    {"name", JsonSetting("My Display")},
    {"mdns", JsonSetting("splitflap")},
    {"otaPass", JsonSetting("")},
    {"timezone", JsonSetting("UTC0")},
    {"dateFormat", JsonSetting("{dd}-{mm}-{yy}")},
    {"timeFormat", JsonSetting("{HH}:{mm}")},
    // Wifi Settings
    {"ssid", JsonSetting("")},
    {"password", JsonSetting("")},
    // MQTT Settings
    {"mqtt_server", JsonSetting("")},
    {"mqtt_port", JsonSetting(1883)},
    {"mqtt_user", JsonSetting("")},
    {"mqtt_pass", JsonSetting("")},
    // Hardware Settings
    {"moduleCount", JsonSetting(8)},
    {"moduleAddresses", JsonSetting({0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27})},
    {"magnetPosition", JsonSetting(730)},
    {"moduleOffsets", JsonSetting({0, 0, 0, 0, 0, 0, 0, 0})},
    {"displayOffset", JsonSetting(0)},
    {"sdaPin", JsonSetting(8)},
    {"sclPin", JsonSetting(9)},
    {"stepsPerRot", JsonSetting(2048)},
    {"maxVel", JsonSetting(15.0f)},
    {"charset", JsonSetting(48)},
    {"charOffsets", JsonSetting(std::vector<std::vector<int>>(8, std::vector<int>(48, 0)))},
    // Scroll Settings (only applies to messages longer than numModules)
    {"scrollDelayMs", JsonSetting(1500)},
    {"scrollRepeatCount", JsonSetting(2)},
    // Boot / Power Settings
    {"bootDelayMs", JsonSetting(2000)},          // power-on settle delay; set the slave higher than the master in multi-display setups
    {"maxConcurrentMotors", JsonSetting(2)},    // cap on simultaneously energized motors during boot homing (1-8)
    // Multi-display master settings
    {"masterGroupCount", JsonSetting(1)},
    {"masterGroupModuleCounts", JsonSetting({8, 8, 8, 8, 8, 8})},
    {"masterGroupMacs", JsonSetting(",,,,,")},
    // Operational States
    {"mode", JsonSetting(0)}
});
// clang-format on

WiFiClient wifiClient;
SplitFlapDisplay display(settings);
SplitFlapEspNow *splitflapEspNow = nullptr;
SplitFlapWebServer webServer(settings, display);
SplitFlapMqtt splitflapMqtt(settings, wifiClient);

bool isMultiDisplayMasterEnabled() {
    return settings.getInt("masterGroupCount") > 1;
}

SplitFlapEspNow *getSplitFlapEspNow() {
    if (! splitflapEspNow) {
        splitflapEspNow = new SplitFlapEspNow(settings, display);
    }
    return splitflapEspNow;
}

void setup() {
    // put your setup code here, to run once:
    Serial.begin(SERIAL_SPEED);
    Serial.println("[boot] serial ready");
    Serial.flush();

    // Power down all motor coils BEFORE anything else (web server, WiFi,
    // boot delay). The PCF8575 I/O expanders power up with all outputs
    // HIGH and the ULN2003 driver boards invert that signal, so at
    // power-on EVERY coil of EVERY module is energized until the firmware
    // writes the outputs low. With 11 modules that is several amps on the
    // shared 5V rail — enough to collapse it below the ESP32's brownout
    // threshold before the display ever gets to move (observed: 10 modules
    // survive at ~3.2V, the 11th tips it under 3V and nothing moves).
    // display.init() writes the init state + stop() to each module, so the
    // coils are off within milliseconds of setup() starting. The network
    // bring-up below then runs with the motors unpowered.
    display.init();
    Serial.println("[boot] display init complete (motor coils powered down)");
    Serial.flush();

    // Per-device boot delay: lets the rail settle after power-on (coils
    // are already off, so this draws almost nothing). In multi-display
    // setups set the slave's bootDelayMs higher than the master's so the
    // two boards never energize motors (or transmit WiFi) at the same time.
    delay(settings.getInt("bootDelayMs"));

    Serial.println("Init Web Server");
    Serial.flush();
    webServer.init();
    webServer.setEspNow(getSplitFlapEspNow());
    Serial.println("[boot] web server init complete");
    Serial.flush();

    if (! webServer.connectToWifi()) {
        Serial.println("[boot] wifi unavailable, starting access point");
        Serial.flush();
        webServer.startAccessPoint();
        webServer.enableOta();
        webServer.startMDNS();
        webServer.startWebServer();
        Serial.println("[boot] services started in access point mode");
        Serial.flush();

        if (getSplitFlapEspNow()->init()) {
            Serial.println("[boot] esp-now initialized successfully");
        } else {
            Serial.println("[boot] esp-now initialization failed");
        }
        Serial.flush();
        display.homeToString("");
        Serial.println("[boot] display homed");
        Serial.flush();

        if (display.getNumModules() == 8) {
            display.writeString("Wifi Err");
        } else {
            display.writeChar('X');
        }
        Serial.println("[boot] setup complete in access point mode");
        Serial.flush();
    } else {
        Serial.println("[boot] wifi connected");
        Serial.flush();
        webServer.enableOta();
        webServer.startMDNS();
        webServer.startWebServer();
        Serial.println("[boot] network services started");
        Serial.flush();

        if (getSplitFlapEspNow()->init()) {
            Serial.println("[boot] esp-now initialized successfully");
        } else {
            Serial.println("[boot] esp-now initialization failed");
        }
        Serial.flush();
        splitflapMqtt.setup();
        splitflapMqtt.setDisplay(&display);
        splitflapMqtt.setEspNow(getSplitFlapEspNow());
        display.setMqtt(&splitflapMqtt);
        Serial.println("[boot] mqtt setup complete");
        Serial.flush();

        display.homeToString("OK");
        delay(250);
        display.writeString("");
        Serial.println("[boot] setup complete in wifi mode");
        Serial.flush();
    }
}

void loop() {
    if (splitflapEspNow) {
        splitflapEspNow->loop();
    }
    splitflapMqtt.loop();

    // check what mode the display is in, this value is updated by the web server
    switch (webServer.getMode()) {
        case 0: singleInputMode(); break;
        case 1: multiInputMode(); break;
        case 2: dateMode(); break;
        case 3: timeMode(); break;
        case 4: break;
        case 5: randomTest(); break;
        case ESP_NOW_REMOTE_MODE: break;
        default: break;
    }

    webServer.handleOta();
    checkConnection();

    reconnectIfNeeded();

    webServer.checkRebootRequired();
    yield();
}

void singleInputMode() {
    String userInput = webServer.getInputString();
    if (userInput != webServer.getWrittenString()) {
        if (isMultiDisplayMasterEnabled()) {
            getSplitFlapEspNow()->distributeMessage(
                userInput, webServer.getCentering(),
                settings.getInt("scrollDelayMs"),
                settings.getInt("scrollRepeatCount")
            );
            if (splitflapMqtt.isConnected()) {
                splitflapMqtt.publishState(userInput);
            }
        } else {
            display.writeString(
                userInput, MAX_RPM, webServer.getCentering(),
                settings.getInt("scrollDelayMs"),
                settings.getInt("scrollRepeatCount")
            );
        }
        webServer.setWrittenString(userInput);
    }
}

void multiInputMode() {
    if (webServer.getNumMultiWords() <= 0) {
        return;
    }

    if (millis() - webServer.getLastSwitchMultiTime() > webServer.getMultiWordDelay()) {
        // get user input, extract correct word from index using webserver counter, and display
        String userInput = webServer.getMultiInputString();
        String currWord = extractFromCSV(userInput, webServer.getMultiWordCurrentIndex());
        if (currWord != webServer.getWrittenString()) {
            if (isMultiDisplayMasterEnabled()) {
                getSplitFlapEspNow()->distributeMessage(
                    currWord, webServer.getCentering(),
                    settings.getInt("scrollDelayMs"),
                    settings.getInt("scrollRepeatCount")
                );
                if (splitflapMqtt.isConnected()) {
                    splitflapMqtt.publishState(currWord);
                }
            } else {
                display.writeString(
                    currWord, MAX_RPM, webServer.getCentering(),
                    settings.getInt("scrollDelayMs"),
                    settings.getInt("scrollRepeatCount")
                );
            }
            webServer.setWrittenString(currWord);
        }
        webServer.setLastSwitchMultiTime(millis());
        webServer.setMultiWordCurrentIndex((webServer.getMultiWordCurrentIndex() + 1) % (webServer.getNumMultiWords()));
    }
}

void dateMode() {
    if (millis() - webServer.getLastCheckDateTime() > webServer.getDateCheckInterval()) {
        webServer.setLastCheckDateTime(millis());

        String format = settings.getString("dateFormat");
        String strftimeFormat = convertToStrftime(format);
        String result = renderDate(strftimeFormat);

        if (result.length() <= display.getNumModules() && result != webServer.getWrittenString()) {
            display.writeString(result, MAX_RPM);
            webServer.setWrittenString(result);
        }
    }
}

void timeMode() {
    if (millis() - webServer.getLastCheckDateTime() > webServer.getDateCheckInterval()) {
        webServer.setLastCheckDateTime(millis());

        // Get user-friendly format from settings (fallback to "HH:mm")
        String userFormat = settings.getString("timeFormat").length() > 0 ? settings.getString("timeFormat") : "HH:mm";

        // Convert to strftime-compatible format
        String strftimeFormat = convertToStrftime(userFormat);
        String result = renderTime(strftimeFormat);

        // Write to display if it changed
        if (result != webServer.getWrittenString()) {
            display.writeString(result, MAX_RPM);
            webServer.setWrittenString(result);
        }
    }
}

void randomTest() {
    display.testRandom();
    delay(2500);
}

void checkConnection() {
    if (millis() - webServer.getLastCheckWifiTime() >
        webServer.getWifiCheckInterval()) { // check wifi to see if disconnected
        webServer.checkWiFi();
        webServer.setLastCheckWifiTime(millis());
    }
}

void reconnectIfNeeded() {
    if (webServer.getAttemptReconnect()) { // check if the device should attempt reconnection to wifi
        webServer.setAttemptReconnect(false);
        display.writeString("");
        if (! webServer.connectToWifi()) {
            webServer.startAccessPoint();
            webServer.enableOta();
            webServer.endMDNS();
            webServer.startMDNS();
            display.writeChar('X');
        } else {
            webServer.enableOta();
            webServer.endMDNS();
            webServer.startMDNS();
            display.writeString("OK");
            webServer.setWrittenString("OK");
            delay(500);
            display.writeString("");
            webServer.setWrittenString("");
        }

        splitflapMqtt.setup();
        if (splitflapEspNow) {
            splitflapEspNow->reinit();
        }
    }
}

String extractFromCSV(String str, int index) {
    int startIndex = 0;
    int endIndex = str.length();

    int commaCount = 0;
    for (int i = 0; i < str.length(); i++) {
        if (str[i] == ',') {
            commaCount++;
            if (commaCount == index) {
                startIndex = i + 1; // skip past the comma
            } else if (commaCount == index + 1) {
                endIndex = i;
            }
        }
    }

    return str.substring(startIndex, endIndex);
}

String renderDate(const String &format) {
    char buf[64];
    time_t now = time(nullptr);
    struct tm *timeinfo = localtime(&now);

    strftime(buf, sizeof(buf), format.c_str(), timeinfo);

    return trimToModuleCount(String(buf), display.getNumModules());
}

String renderTime(const String &format) {
    char buf[64];
    time_t now = time(nullptr);
    struct tm *timeinfo = localtime(&now);

    strftime(buf, sizeof(buf), format.c_str(), timeinfo);

    return trimToModuleCount(String(buf), display.getNumModules());
}

String trimToModuleCount(const String &str, int maxLen) {
    return str.length() > maxLen ? str.substring(0, maxLen) : str;
}

String convertToStrftime(String userFormat) {
    struct FormatToken
    {
        const char *token;
        const char *strftime;
    };

    FormatToken tokens[] = {
        // Date formats
        {"{yyyy}", "%Y"}, // 4-digit year (e.g. 2025)
        {"{dddd}", "%A"}, // Full weekday name (e.g. Monday)
        {"{mmmm}", "%B"}, // Full month name (e.g. January)
        {"{ddd}", "%a"},  // Abbreviated weekday name (e.g. Mon)
        {"{mmm}", "%b"},  // Abbreviated month name (e.g. Apr)
        {"{dd}", "%d"},   // 2-digit day of month, zero-padded (01–31)
        {"{mm}", "%m"},   // 2-digit month number, zero-padded (01–12)
        {"{yy}", "%y"},   // 2-digit year (e.g. 25)
        {"{ww}", "%V"},   // ISO 8601 week number (01–53)
        {"{D}", "%j"},    // Day of the year (001–366)

        // Time formats
        {"{HH}", "%H"},   // Hours (24-hour clock, 00–23)
        {"{hh}", "%I"},   // Hours (12-hour clock, 01–12)
        {"{MM}", "%M"},   // Minutes (00–59)
        {"{AMPM}", "%p"}, // AM or PM
    };

    for (auto &t : tokens) {
        userFormat.replace(t.token, t.strftime);
    }

    return userFormat;
}
