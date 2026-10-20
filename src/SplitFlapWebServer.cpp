#include "SplitFlapWebServer.h"

#include "CalibrationTriggers.h"
#include "CsvUtils.h"
#include "SplitFlapEspNow.h"
#include "WiredLink.h"

#include <ArduinoJson.h>
#include <AsyncJson.h>
#include <ctype.h>

bool isMultiDisplayMasterEnabled();

#define AP_SSID "SFD"

#ifndef AP_CHANNEL
#define AP_CHANNEL 6
#endif

#ifndef WIFI_SSID
#define WIFI_SSID ""
#endif

#ifndef WIFI_PASS
#define WIFI_PASS ""
#endif

SplitFlapWebServer::SplitFlapWebServer(JsonSettings &settings, SplitFlapDisplay &display)
    : settings(settings), display(display), server(80), multiWordDelay(1000), rebootRequired(false),
      attemptReconnect(false), multiWordCurrentIndex(0), numMultiWords(0), wifiCheckInterval(1000), connectionMode(0),
      checkDateInterval(250), centering(1) {
    lastSwitchMultiTime = millis();
}

void SplitFlapWebServer::init() {
    if (! LittleFS.begin()) {
        Serial.println("An Error has occurred while mounting LittleFS");
        return;
    }

    setTimezone();
}

void SplitFlapWebServer::setTimezone() {
    const char *sntpServer = "pool.ntp.org";
    const char *defaultTz = "UTC0";
    String timezoneSetting = settings.getString("timezone");
    String posixTimezone = defaultTz;

    File file = LittleFS.open("/timezones.json", "r");
    if (! file) {
        Serial.println("Failed to open timezones.json; defaulting to UTC");
        configTzTime(defaultTz, sntpServer);
        return;
    }

    size_t size = file.size();
    std::unique_ptr<char[]> buffer(new char[size + 1]);
    file.readBytes(buffer.get(), size);
    buffer[size] = '\0';
    file.close();

    JsonDocument timezones;
    DeserializationError error = deserializeJson(timezones, buffer.get());

    if (error) {
        Serial.println("Failed to parse timezones.json: " + String(error.c_str()));
        configTzTime(defaultTz, sntpServer);
        return;
    }

    for (JsonPair kv : timezones.as<JsonObject>()) {
        String keyStr = kv.key().c_str();
        String valueStr = kv.value().as<String>();

        if (keyStr == timezoneSetting) {
            posixTimezone = valueStr;
            break;
        }
    }

    Serial.println("POSIX Timezone set to: " + posixTimezone);
    configTzTime(posixTimezone.c_str(), sntpServer);
}

void SplitFlapWebServer::setMode(int targetMode) {
    settings.putInt("mode", targetMode);
}

int SplitFlapWebServer::getMode() {
    return settings.getInt("mode");
}

namespace {

// Last STA disconnect reason captured by the event handler below. This is the
// real failure cause when status just sits at 6 (WL_DISCONNECTED).
uint8_t s_lastWifiDiscReason = 0;
bool s_haveWifiDiscReason = false;
bool s_wifiEventHooked = false;

const char *wifiAuthModeName(int auth) {
    switch (auth) {
        case 0: return "OPEN";
        case 1: return "WEP";
        case 2: return "WPA_PSK";
        case 3: return "WPA2_PSK";
        case 4: return "WPA_WPA2_PSK";
        case 5: return "WPA2_ENTERPRISE";
        case 6: return "WPA3_PSK";
        case 7: return "WPA2_WPA3_PSK(mixed)";
        case 8: return "WAPI_PSK";
        default: return "?";
    }
}

const char *wifiDiscReasonName(uint8_t reason) {
    switch (reason) {
        case 1: return "UNSPECIFIED";
        case 2: return "AUTH_EXPIRE";
        case 3: return "AUTH_LEAVE";
        case 4: return "ASSOC_EXPIRE";
        case 5: return "ASSOC_TOOMANY";
        case 6: return "NOT_AUTHED";
        case 7: return "NOT_ASSOCED";
        case 8: return "ASSOC_LEAVE";
        case 14: return "MIC_FAILURE";
        case 15: return "4WAY_HANDSHAKE_TIMEOUT";
        case 17: return "IE_IN_4WAY_DIFFERS";
        case 23: return "802_1X_AUTH_FAILED";
        case 200: return "BEACON_TIMEOUT";
        case 201: return "NO_AP_FOUND";
        case 202: return "AUTH_FAIL";
        case 203: return "ASSOC_FAIL";
        case 204: return "HANDSHAKE_TIMEOUT";
        case 205: return "CONNECTION_FAIL";
        default: return "?";
    }
}

void ensureWifiEventHandler() {
    if (s_wifiEventHooked) {
        return;
    }
    s_wifiEventHooked = true;
    WiFi.onEvent([](WiFiEvent_t event, WiFiEventInfo_t info) {
        if (event == ARDUINO_EVENT_WIFI_STA_DISCONNECTED) {
            uint8_t reason = info.wifi_sta_disconnected.reason;
            uint8_t len = info.wifi_sta_disconnected.ssid_len;
            if (len > 32) {
                len = 32;
            }
            char ssidBuf[33];
            memcpy(ssidBuf, info.wifi_sta_disconnected.ssid, len);
            ssidBuf[len] = '\0';
            // Our own inter-attempt disconnect() fires an event with an empty
            // SSID — log it but don't let it overwrite the real failure reason.
            if (len == 0) {
                Serial.printf("[wifi] event: disconnected (local reset) reason=%u (%s)\n", reason, wifiDiscReasonName(reason));
                return;
            }
            s_lastWifiDiscReason = reason;
            s_haveWifiDiscReason = true;
            Serial.printf(
                "[wifi] event: disconnected ssid='%s' reason=%u (%s)\n", ssidBuf, reason,
                wifiDiscReasonName(reason)
            );
        }
    });
}

// Scan visible networks and pick the strongest BSSID matching `ssid`.
// Returns its channel, or -1 when the SSID is not visible (5GHz-only,
// hidden, typo, or out of range). Logs every visible network so the serial
// log shows whether the target exists at all.
int findTargetAp(const char *ssid, uint8_t *bssidOut, int *rssiOut, int *authOut) {
    Serial.println("[wifi] scanning networks...");
    int n = WiFi.scanNetworks(false, true);
    if (n < 0) {
        Serial.println("[wifi] scan failed");
        return -1;
    }
    Serial.printf("[wifi] scan found %d network(s)\n", n);
    int bestChannel = -1;
    int bestRssi = -1000;
    int bestAuth = 0;
    for (int i = 0; i < n; i++) {
        String found = WiFi.SSID(i);
        int rssi = WiFi.RSSI(i);
        int ch = WiFi.channel(i);
        int auth = (int) WiFi.encryptionType(i);
        Serial.printf("[wifi] scan: '%s' rssi=%d ch=%d auth=%d\n", found.c_str(), rssi, ch, auth);
        if (found == ssid && rssi > bestRssi) {
            bestRssi = rssi;
            bestChannel = ch;
            bestAuth = auth;
            memcpy(bssidOut, WiFi.BSSID(i), 6);
        }
    }
    WiFi.scanDelete();
    if (bestChannel > 0) {
        if (rssiOut) {
            *rssiOut = bestRssi;
        }
        if (authOut) {
            *authOut = bestAuth;
        }
        Serial.printf(
            "[wifi] target '%s': ch=%d rssi=%d auth=%d/%s bssid=%02x:%02x:%02x:%02x:%02x:%02x\n", ssid,
            bestChannel, bestRssi, bestAuth, wifiAuthModeName(bestAuth), bssidOut[0], bssidOut[1],
            bssidOut[2], bssidOut[3], bssidOut[4], bssidOut[5]
        );
        if (bestAuth == 7 || bestAuth == 6) {
            Serial.println("[wifi] NOTE: AP is WPA3/mixed mode - if auth keeps failing, set the router to WPA2-PSK (AES) only.");
        }
    } else {
        Serial.printf("[wifi] target '%s' NOT in scan (hidden? 5GHz-only? typo? far?)\n", ssid);
    }
    return bestChannel;
}

} // namespace

void SplitFlapWebServer::checkWiFi() {
    if (connectionMode == 1) {
        if (WiFi.status() != WL_CONNECTED) {
            // Throttle reconnect attempts: hammering disconnect()/reconnect()
            // every check interval (1s) never lets DHCP/4-way handshake finish.
            const unsigned long kReconnectIntervalMs = 10000;
            unsigned long now = millis();
            if (lastWifiReconnectAttempt == 0 || now - lastWifiReconnectAttempt >= kReconnectIntervalMs) {
                lastWifiReconnectAttempt = now;
                Serial.printf("[wifi] lost (status=%d), reconnecting...\n", (int) WiFi.status());
                WiFi.disconnect();
                WiFi.reconnect();
            }
        }
    }
}

bool SplitFlapWebServer::loadWiFiCredentials() {
    // Allow WIFI_SSID and WIFI_PASS to be overridden by compile-time definitions
    String ssid = String(WIFI_SSID).isEmpty() ? settings.getString("ssid") : String(WIFI_SSID);
    String password = String(WIFI_PASS).isEmpty() ? settings.getString("password") : String(WIFI_PASS);

    if (ssid != "" && password != "") {
        Serial.println("Wi-Fi credentials loaded successfully.");
        Serial.print("Connecting to Network: ");
        Serial.println(ssid);
        // Clean STA state without powering the radio off: erase any stale
        // persistent AP config, and keep flash persistence off so a stale
        // channel/BSSID can't shadow the fresh begin().
        WiFi.persistent(false);
        WiFi.disconnect(false, true);
        delay(300);
        WiFi.mode(WIFI_STA);
        WiFi.setAutoReconnect(false);
        WiFi.setSleep(false);
        WiFi.setMinSecurity(WIFI_AUTH_WPA2_PSK);
        WiFi.setScanMethod(WIFI_ALL_CHANNEL_SCAN);
        WiFi.setSortMethod(WIFI_CONNECT_AP_BY_SIGNAL);
#ifdef WIFI_TX_POWER
        delay(100);
        WiFi.setTxPower((wifi_power_t) WIFI_TX_POWER);
#else
        // Lower TX power reduces current spikes that brown out the shared 5V
        // rail on multi-module builds (see README Power/Startup Settings).
        WiFi.setTxPower(WIFI_POWER_8_5dBm);
#endif
        WiFi.begin(ssid.c_str(), password.c_str());
        return true; // Return true if credentials exist
    }
    return false;    // Return false if no credentials were found
}

void SplitFlapWebServer::checkRebootRequired() {
    if (rebootRequired) {
        Serial.println("Reboot required. Restarting...");
        delay(1000);
        ESP.restart();
    }
}

void SplitFlapWebServer::handleOta() {
    ArduinoOTA.handle();
}
void SplitFlapWebServer::enableOta() {
    // Skip OTA initialisation if no password is set
    if (settings.getString("otaPass") == "") {
        return;
    }

    ArduinoOTA.setHostname(settings.getString("mdns").c_str()); // otherwise mdns name gets overwritten with default
    ArduinoOTA.setPassword(settings.getString("otaPass").c_str());

    ArduinoOTA
        .onStart([]() {
        String type;
        if (ArduinoOTA.getCommand() == U_FLASH) {
            type = "sketch";
        } else {            // U_LITTLEFS
            type = "filesystem";
            LittleFS.end(); // Unmount the filesystem before update
        }
        Serial.println("Start updating " + type);
    })
        .onEnd([]() {
        Serial.println("\nEnd");
        LittleFS.begin(); // Remount filesystem
    })
        .onProgress([](unsigned int progress, unsigned int total) {
        Serial.printf("Progress: %u%%\r", (progress / (total / 100)));
    }).onError([](ota_error_t error) {
        Serial.printf("Error[%u]: ", error);
        LittleFS.begin(); // Remount filesystem
        if (error == OTA_AUTH_ERROR) {
            Serial.println("Auth Failed");
        } else if (error == OTA_BEGIN_ERROR) {
            Serial.println("Begin Failed");
        } else if (error == OTA_CONNECT_ERROR) {
            Serial.println("Connect Failed");
        } else if (error == OTA_RECEIVE_ERROR) {
            Serial.println("Receive Failed");
        } else if (error == OTA_END_ERROR) {
            Serial.println("End Failed");
        }
    });

    ArduinoOTA.begin();
    Serial.println("OTA Initialized");
}

bool SplitFlapWebServer::connectToWifi() {
    ensureWifiEventHandler();

    // Allow WIFI_SSID and WIFI_PASS to be overridden by compile-time definitions
    String ssidStr = String(WIFI_SSID).isEmpty() ? settings.getString("ssid") : String(WIFI_SSID);
    String passStr = String(WIFI_PASS).isEmpty() ? settings.getString("password") : String(WIFI_PASS);
    if (ssidStr == "" || passStr == "") {
        Serial.println("[wifi] no credentials stored");
        return false;
    }
    // Never log the password itself — length only.
    Serial.printf(
        "[wifi] credentials: ssid='%s' len=%d passLen=%d%s\n", ssidStr.c_str(), (int) ssidStr.length(),
        (int) passStr.length(), String(WIFI_SSID).isEmpty() ? "" : " (compile-time override)"
    );
    String trimmed = ssidStr;
    trimmed.trim();
    if (trimmed != ssidStr) {
        Serial.println("[wifi] WARNING: ssid has leading/trailing whitespace");
    }

    // Clean STA state: no radio power-off, erase stale config, no flash
    // persistence. Keep the driver's own auto-retry OFF while our attempt loop
    // runs — its background disconnect()+begin() races ours and its local-reset
    // events leave a stale WL_CONNECT_FAILED behind (observed: attempts 2-4
    // aborting instantly). Re-enabled only after a successful connect.
    WiFi.persistent(false);
    WiFi.setAutoReconnect(false);
    WiFi.setSleep(false);
    // Explicit STA search policy (must precede begin): scan every channel,
    // prefer strongest signal, require at least WPA2 for secured networks.
    WiFi.setMinSecurity(WIFI_AUTH_WPA2_PSK);
    WiFi.setScanMethod(WIFI_ALL_CHANNEL_SCAN);
    WiFi.setSortMethod(WIFI_CONNECT_AP_BY_SIGNAL);
    WiFi.mode(WIFI_STA);
#ifdef WIFI_TX_POWER
    delay(100);
    WiFi.setTxPower((wifi_power_t) WIFI_TX_POWER);
#else
    // Lower TX power reduces current spikes that brown out the shared 5V
    // rail on multi-module builds (see README Power/Startup Settings).
    WiFi.setTxPower(WIFI_POWER_8_5dBm);
#endif
    delay(300);

    // One scan up front: shows whether the SSID is visible at all and pins
    // the strongest 2.4GHz BSSID/channel, which fixes flaky connects on
    // multi-AP / band-steered networks.
    uint8_t targetBssid[6] = {0, 0, 0, 0, 0, 0};
    int targetRssi = 0;
    int targetAuth = 0;
    int targetChannel = findTargetAp(ssidStr.c_str(), targetBssid, &targetRssi, &targetAuth);

    static const int kMaxAttempts = 4;
    static const unsigned long kAttemptTimeoutMs = 15000; // per-attempt wait
    for (int attempt = 1; attempt <= kMaxAttempts; attempt++) {
        s_haveWifiDiscReason = false;
        // First two attempts use the scan-pinned BSSID/channel (best for
        // multi-AP / band-steered networks); later attempts fall back to a
        // full unpinned begin, which some WPA2/WPA3-mixed APs prefer.
        bool pinned = (targetChannel > 0 && attempt <= 2);
        if (attempt >= 3) {
            // Full radio reset before fallback attempts.
            WiFi.mode(WIFI_OFF);
            delay(500);
            WiFi.mode(WIFI_STA);
            WiFi.setAutoReconnect(false);
            WiFi.setSleep(false);
            WiFi.setMinSecurity(WIFI_AUTH_WPA2_PSK);
            WiFi.setScanMethod(WIFI_ALL_CHANNEL_SCAN);
            WiFi.setSortMethod(WIFI_CONNECT_AP_BY_SIGNAL);
#ifdef WIFI_TX_POWER
            delay(100);
            WiFi.setTxPower((wifi_power_t) WIFI_TX_POWER);
#else
            WiFi.setTxPower(WIFI_POWER_8_5dBm);
#endif
            delay(300);
        } else {
            WiFi.disconnect(false, true);
            delay(300);
        }
        if (pinned) {
            Serial.printf(
                "[wifi] attempt %d/%d: begin ch=%d rssi=%d (pinned)...\n", attempt, kMaxAttempts,
                targetChannel, targetRssi
            );
            WiFi.begin(ssidStr.c_str(), passStr.c_str(), targetChannel, targetBssid);
        } else {
            Serial.printf("[wifi] attempt %d/%d: begin (unpinned)...\n", attempt, kMaxAttempts);
            WiFi.begin(ssidStr.c_str(), passStr.c_str());
        }
        delay(500); // let the driver start before polling status()
        unsigned long startAttemptTime = millis();
        unsigned long lastPrintTime = startAttemptTime;

        // NOTE: no fail-fast on WiFi.status() here. Status can still hold the
        // previous attempt's terminal value (or the driver's own local-reset
        // event), which aborted attempts 2-4 instantly in testing. Only the
        // live disconnect-event reason is trustworthy, and only AUTH_FAIL /
        // NO_AP_FOUND are definitive enough to cut an attempt short.
        while (WiFi.status() != WL_CONNECTED) {
            if (s_haveWifiDiscReason &&
                (s_lastWifiDiscReason == 202 || s_lastWifiDiscReason == 201)) {
                Serial.println();
                break;
            }
            if (millis() - startAttemptTime >= kAttemptTimeoutMs) {
                Serial.println();
                break;
            }
            if ((millis() - lastPrintTime) > 1000) {
                Serial.print(".");
                lastPrintTime = millis();
            }
            yield();
        }

        if (WiFi.status() == WL_CONNECTED) {
            // connected succesfully
            connectionMode = 1;
            WiFi.softAPdisconnect(); // Turns off SoftAP mode only after connected to
            // actual network
            WiFi.setAutoReconnect(true);
            WiFi.setSleep(false);
            Serial.printf(
                "\n[wifi] connected! ip=%s rssi=%d ch=%d\n", WiFi.localIP().toString().c_str(),
                WiFi.RSSI(), WiFi.channel()
            );
            return true;
        }

        Serial.printf("[wifi] attempt %d failed (status=%d", attempt, (int) WiFi.status());
        if (s_haveWifiDiscReason) {
            Serial.printf(" reason=%u/%s", s_lastWifiDiscReason, wifiDiscReasonName(s_lastWifiDiscReason));
        }
        Serial.printf(")%s\n", attempt < kMaxAttempts ? " Retrying..." : "");
        // Definitive failures: retrying the same credentials cannot succeed.
        if (s_haveWifiDiscReason && s_lastWifiDiscReason == 202) {
            Serial.println("[wifi] AUTH_FAIL: password rejected. Fix via the SFD AP portal settings page, "
                           "then reboot. Aborting remaining attempts.");
            WiFi.disconnect(false, true);
            return false;
        }
        if (s_haveWifiDiscReason && s_lastWifiDiscReason == 201) {
            Serial.println("[wifi] NO_AP_FOUND: SSID vanished after scan - check 2.4GHz band / hidden SSID. "
                           "Aborting remaining attempts.");
            WiFi.disconnect(false, true);
            return false;
        }
        if (s_haveWifiDiscReason && s_lastWifiDiscReason == 2) {
            Serial.println("[wifi] hint: AUTH_EXPIRE - AP dropped auth. Re-enter the password via the SFD AP "
                           "portal to rule out a typo; if the AP is WPA2/WPA3 mixed (see auth= above), set it "
                           "to WPA2-PSK/AES only.");
        } else if (
            s_haveWifiDiscReason && (s_lastWifiDiscReason == 15 || s_lastWifiDiscReason == 204)) {
            Serial.println("[wifi] hint: handshake timeout - wrong password or WPA2/WPA3/PMF mismatch.");
        } else if (s_haveWifiDiscReason && s_lastWifiDiscReason == 200) {
            Serial.println("[wifi] hint: BEACON_TIMEOUT - weak signal or interference, move closer.");
        }

        // Back off briefly before the next begin() so the radio/AP can settle.
        WiFi.disconnect(false, true);
        delay(1000 * attempt);
    }

    Serial.println("[wifi] all attempts exhausted");
    Serial.println("[wifi] isolate: phone 2.4GHz hotspot with simple SSID/pass. Hotspot OK => router config "
                   "(WPA mode/password); hotspot fails too => board power/RF issue.");
    return false;
}

void SplitFlapWebServer::startAccessPoint() {
    connectionMode = 0;
    const char *apSSID = AP_SSID;
    // A failed station attempt leaves the radio in WIFI_STA mid-scan
    // (WiFi.begin keeps retrying in the background). Stop it and switch to
    // AP mode explicitly — otherwise softAP() can fail silently while we
    // print "AP Mode Started!" unconditionally (observed: success log +
    // default 192.168.4.1, but no beacons on air).
    WiFi.disconnect(true);
    WiFi.setAutoReconnect(false);
    WiFi.mode(WIFI_AP);
    delay(200);
    // Explicit channel/visibility/max-clients: pin them so a core-version
    // default change can't silently hide the AP. Channel 6 avoids the
    // crowded channel 1 default used by many routers.
    bool apUp = WiFi.softAP(apSSID, nullptr, AP_CHANNEL, 0, 4);
    if (! apUp) {
        Serial.println("AP start failed, retrying...");
        delay(500);
        apUp = WiFi.softAP(apSSID, nullptr, AP_CHANNEL, 0, 4);
    }
    if (! apUp) {
        Serial.println("AP Mode FAILED to start!");
        return;
    }
#ifdef WIFI_TX_POWER
    delay(100);
    WiFi.setTxPower((wifi_power_t) WIFI_TX_POWER);
#endif
    Serial.println("AP Mode Started!");
    Serial.println("Connect to: " + String(apSSID));
    Serial.printf(
        "[wifi] mode=%d apChannel=%d ip=%s\n", (int) WiFi.getMode(), WiFi.channel(), WiFi.softAPIP().toString().c_str()
    );
}

void fourOhFour(AsyncWebServerRequest *request) {
    Serial.println("Request: " + request->url());
    Serial.println("Method: " + String(request->methodToString()));
    request->send(404);
}

void SplitFlapWebServer::endMDNS() {
    MDNS.end();
    Serial.println("mDNS responder stopped");
}

void SplitFlapWebServer::startMDNS() {
    if (! MDNS.begin(settings.getString("mdns").c_str())) {
        Serial.println("Error setting up MDNS responder! Restarting...");
        delay(1000);
        ESP.restart();
    }

    Serial.println("mDNS: http://" + settings.getString("mdns") + ".local");
}

void SplitFlapWebServer::startWebServer() {
    server.on("/", HTTP_GET, [this](AsyncWebServerRequest *request) { request->redirect("/index.html"); });

    File root = LittleFS.open("/");
    if (! root || ! root.isDirectory()) {
        Serial.println("Failed to open directory or not a directory");
        return;
    }

    File file = root.openNextFile();
    while (file) {
        if (String(file.name()).endsWith(".gz")) {
            const char *filename = file.name();
            String tempFilename = (String("/") + String(filename));
            tempFilename.replace(".gz", "");
            filename = tempFilename.c_str();

            server.serveStatic(filename, LittleFS, filename, "max-age=600");
        }
        file = root.openNextFile();
    }

    server.on("/settings", HTTP_GET, [this](AsyncWebServerRequest *request) {
        JsonDocument currentSettings = settings.toJson();
        JsonDocument response;
        response["settings"] = currentSettings.as<JsonObject>();
        response["localMac"] = WiFi.macAddress();
        response["schemaVersion"] = SETTINGS_SCHEMA_VERSION;
        String peersJson = "";
        if (settings.getInt("linkTransport") != 0 && wired) {
            peersJson = wired->getDiscoveredPeersJson();
        } else if (espNow) {
            peersJson = espNow->getDiscoveredPeersJson();
        }
        if (peersJson.length() > 0) {
            JsonArray peers = response["discoveredPeers"].to<JsonArray>();
            JsonDocument peersDoc;
            deserializeJson(peersDoc, peersJson);
            for (JsonObject peer : peersDoc.as<JsonArray>()) {
                peers.add(peer);
            }
        }
        request->send(200, "application/json", response.as<String>());
    });

    server.on("/settings/reset", HTTP_POST, [this](AsyncWebServerRequest *request) {
        settings.reset();

        JsonDocument response;
        response["message"] = "Settings reset successfully! Reconnect to the " + String(AP_SSID) + " network";
        response["persistent"] = true;

        request->send(200, "application/json", response.as<String>());

        this->attemptReconnect = true;
    });

    server.addHandler(new AsyncCallbackJsonWebHandler(
        "/settings",
        [this](AsyncWebServerRequest *request, JsonVariant &json) {
        if (request->method() != HTTP_POST) {
            return request->send(405, "application/json", "{\"error\":\"Method Not Allowed\"}");
        }

        Serial.println("Received settings update request");
        Serial.println(json.as<String>());
        bool rebootRequired = false;
        bool reconnect = false;
        JsonDocument response;
        response["message"] = "Settings saved successfully!";

        // TODO Refactor this it's gross
        if ((json["ssid"].is<String>() && json["ssid"].as<String>() != settings.getString("ssid")) ||
            (json["password"].is<String>() && json["password"].as<String>() != settings.getString("password"))) {
            reconnect = true;
            response["message"] = "Settings updated successfully, Network " "settings have changed, reconnect to the " +
                json["ssid"].as<String>() + " network";
        }

        if (json["otaPass"].is<String>() && json["otaPass"].as<String>() != settings.getString("otaPass")) {
            rebootRequired = true; // OTA password change can only be applied by rebooting
            response["message"] = "Settings updated successfully, OTA Password has changed. Rebooting...";
        }

        if (json["mdns"].is<String>() && json["mdns"].as<String>() != settings.getString("mdns")) {
            reconnect = true;
            response["message"] =
                "Settings updated successfully, mDNS name has changed, " "automatically redirecting to http://" +
                json["mdns"].as<String>() + ".local...";
            response["redirect"] = "http://" + json["mdns"].as<String>() + ".local/settings.html";
        }

        if ((json["mqtt_server"].is<String>() && json["mqtt_server"].as<String>() != settings.getString("mqtt_server")
            ) ||
            (json["mqtt_port"].is<int>() && json["mqtt_port"].as<int>() != settings.getInt("mqtt_port")) ||
            (json["mqtt_user"].is<String>() && json["mqtt_user"].as<String>() != settings.getString("mqtt_user")) ||
            (json["mqtt_pass"].is<String>() && json["mqtt_pass"].as<String>() != settings.getString("mqtt_pass"))) {
            response["message"] = "Mqtt settings have changed, reconnecting...";
            reconnect = true;
        }

        if (! validateMasterSettings(json, response)) {
            return request->send(400, "application/json", response.as<String>());
        }

        // Keys consumed once in SplitFlapDisplay::init() only take effect after
        // a reboot — flag it (like the OTA password change above) instead of
        // reporting success that silently changes nothing.
        bool hardwareChanged = false;
        if (json["moduleAddresses"].is<String>() &&
            JsonSettings::parseIntVector(json["moduleAddresses"].as<String>()) !=
                settings.getIntVector("moduleAddresses")) {
            hardwareChanged = true;
        }
        if ((json["maxVel"].is<float>() || json["maxVel"].is<int>() || json["maxVel"].is<String>()) &&
            json["maxVel"].as<float>() != settings.getFloat("maxVel")) {
            hardwareChanged = true;
        }
        const char *hardwareIntKeys[] = {"moduleCount", "stepsPerRot", "charset", "sdaPin", "sclPin"};
        for (const char *k : hardwareIntKeys) {
            if (json[k].is<int>() || json[k].is<float>() || json[k].is<String>()) {
                if (json[k].as<int>() != settings.getInt(k)) {
                    hardwareChanged = true;
                    break;
                }
            }
        }
        // Inter-display link transport, address, pins and baud are
        // snapshotted in WiredLink::init() at boot, while dispatch reads the
        // live setting — changing them without a reboot would route messages
        // to an uninitialized transport. Flag a reboot like other hardware.
        const char *linkIntKeys[] = {"linkTransport", "linkAddress", "linkTxPin", "linkRxPin", "linkBaud"};
        for (const char *k : linkIntKeys) {
            if (json[k].is<int>() || json[k].is<float>() || json[k].is<String>()) {
                if (json[k].as<int>() != settings.getInt(k)) {
                    hardwareChanged = true;
                    break;
                }
            }
        }
        if (hardwareChanged) {
            rebootRequired = true;
            response["message"] = "Settings updated successfully, hardware settings have changed. Rebooting...";
        }

        // Single numeric trigger point for calibration homing (see
        // CalibrationTriggers.h): compare parsed values, not CSV text, so a
        // whitespace-only edit ("0,0" vs "0, 0") schedules nothing — while
        // magnetPosition, which feeds the same magnet-target computation as
        // displayOffset, triggers a reload like any other calibration key.
        CalibrationSnapshot storedCalibration;
        storedCalibration.displayOffset = settings.getInt("displayOffset");
        storedCalibration.magnetPosition = settings.getInt("magnetPosition");
        storedCalibration.moduleOffsets = settings.getIntVector("moduleOffsets");
        storedCalibration.charOffsets = settings.getIntMatrix("charOffsets");

        CalibrationSnapshot incomingCalibration = storedCalibration;
        if (json["moduleOffsets"].is<String>()) {
            incomingCalibration.moduleOffsets = JsonSettings::parseIntVector(json["moduleOffsets"].as<String>());
        }
        if (json["charOffsets"].is<String>()) {
            incomingCalibration.charOffsets = JsonSettings::parseIntMatrix(json["charOffsets"].as<String>());
        }
        // Ints arrive as JSON numbers from the settings page (parseInt) or
        // numeric strings via config import — accept both, like fromJson's
        // as<int>() does. Anything else (including a missing key) means the
        // key is not being changed.
        if (json["displayOffset"].is<int>() || json["displayOffset"].is<float>() ||
            json["displayOffset"].is<String>()) {
            incomingCalibration.displayOffset = json["displayOffset"].as<int>();
        }
        if (json["magnetPosition"].is<int>() || json["magnetPosition"].is<float>() ||
            json["magnetPosition"].is<String>()) {
            incomingCalibration.magnetPosition = json["magnetPosition"].as<int>();
        }
        bool offsetsChanged = calibrationChanged(storedCalibration, incomingCalibration);

        bool remoteOffsetsChanged = false;
        const char *remoteOffsetKeys[] = {
            "rModOffs", "rChrOff0", "rChrOff1", "rChrOff2", "rChrOff3", "rChrOff4", "rDispOffs"
        };
        for (const char *k : remoteOffsetKeys) {
            if (json[k].is<String>() && json[k].as<String>() != settings.getString(k)) {
                remoteOffsetsChanged = true;
                break;
            }
        }

        if (! settings.fromJson(json)) {
            response["message"] = "Failed to save settings";
            response["type"] = "error";
            response["errors"]["key"] = settings.getLastValidationKey();
            response["errors"]["message"] = settings.getLastValidationError();
            return request->send(400, "application/json", response.as<String>());
        }

        if (offsetsChanged) {
            auto chrOffs = settings.getIntMatrix("charOffsets");
            for (auto &row : chrOffs)
                for (auto &v : row) v = constrain(v, -32, 32);
            settings.putIntMatrix("charOffsets", chrOffs);

            // Defer the display work to the loop task: reloadOffsets() homes
            // modules (seconds of Wire I/O) and must not run in the AsyncTCP
            // task racing the loop task's own display access.
            pendingActions_.requestReloadOffsets();
            response["message"] = "Settings saved, offsets will be applied";

            if (! isMultiDisplayMasterEnabled() && (espNow || wired)) {
                pendingActions_.requestReportOffsets();
            }
        }

        if (remoteOffsetsChanged && isMultiDisplayMasterEnabled()) {
            for (int r = 0; r < 5; r++) {
                String key = "rChrOff" + String(r);
                auto rChrOffs = settings.getIntMatrix(key.c_str());
                for (auto &row : rChrOffs)
                    for (auto &v : row) v = constrain(v, -32, 32);
                settings.putIntMatrix(key.c_str(), rChrOffs);
            }

            // ESP-NOW pushes take hundreds of ms of esp_now_send + spacing
            // delays — also loop-task work (audit issue #12).
            pendingActions_.requestPushOffsets();
        }

        response["type"] = "success";
        response["persistent"] = reconnect;

        request->send(200, "application/json", response.as<String>());

        this->rebootRequired = rebootRequired;
        this->attemptReconnect = reconnect;
    }
    ));

    server
        .addHandler(new AsyncCallbackJsonWebHandler("/text", [this](AsyncWebServerRequest *request, JsonVariant &json) {
        if (request->method() != HTTP_POST) {
            return request->send(405, "application/json", "{\"error\":\"Method Not Allowed\"}");
        }

        Serial.println("Received text update request");
        Serial.println(json.as<String>());

        // {"mode":"single","words":["adfasdf"],"delay":1,"center":false}
        // {"mode":"multiple","words":["asdf","asdfasdf","fffff"],"delay":"14","center":true}
        JsonDocument response;

        // First error wins: report the earliest invalid field instead of
        // letting later checks overwrite it.
        if (! json["mode"].is<String>()) {
            response["message"] = "Invalid mode type";
            response["type"] = "error";
            return request->send(400, "application/json", response.as<String>());
        }

        if (! json["words"].is<JsonArray>()) {
            response["message"] = "Invalid words array";
            response["type"] = "error";
            return request->send(400, "application/json", response.as<String>());
        }

        // NB: named delaySec — a local `float delay` would shadow the
        // Arduino delay() function in this scope.
        float delaySec = json["delay"].as<float>();
        if (delaySec < 1) {
            response["message"] = "Invalid delay type / value";
            response["type"] = "error";
            return request->send(400, "application/json", response.as<String>());
        }

        if (! json["center"].is<bool>()) {
            response["message"] = "Invalid center type";
            response["type"] = "error";
            return request->send(400, "application/json", response.as<String>());
        }

        this->setMultiDelay(delaySec * 1000);
        Serial.println("Delay: " + String(this->getMultiWordDelay()));

        centering = json["center"].as<bool>() ? 1 : 0;
        Serial.println("centering: " + String(centering ? "true" : "false"));

        if (json["mode"] == "single") {
            String word = decodeURIComponent(json["words"][0].as<String>());
            Serial.println("Single Word: " + word);
            this->setInputString(word);
            this->setMode(0); // change mode last once all variables updated
        }

        if (json["mode"] == "multiple") {
            JsonArray wordsArray = json["words"].as<JsonArray>();
            String words = "";
            for (JsonVariant v : wordsArray) {
                words += decodeURIComponent(v.as<String>()) + ",";
            }
            if (words.length() > 0) {
                words.remove(words.length() - 1);
            }

            this->setMultiInputString(words);
            this->numMultiWords = wordsArray.size();
            Serial.println("Multiple Words: " + words);
            Serial.println("Number of Words: " + String(this->numMultiWords));

            this->setMode(1);
        }

        response["message"] = "Text updated successfully!";
        response["type"] = "success";

        request->send(200, "application/json", response.as<String>());
    }));

    server.onNotFound(fourOhFour);

    server.begin();
}

String SplitFlapWebServer::decodeURIComponent(String encodedString) {
    // Real percent-decoder: handles both %XX cases and leaves stray '%' or
    // malformed escapes (e.g. "%2", "%G1", trailing "%") untouched. Replaces
    // the previous uppercase-only replace() chain, which passed lowercase
    // escapes like "%2a" through literally.
    String decodedString = "";
    decodedString.reserve(encodedString.length());

    for (unsigned int i = 0; i < encodedString.length(); i++) {
        char c = encodedString[i];
        if (c != '%' || i + 2 >= encodedString.length() || ! isxdigit((unsigned char) encodedString[i + 1]) ||
            ! isxdigit((unsigned char) encodedString[i + 2])) {
            decodedString += c;
            continue;
        }

        auto hexVal = [](char h) -> int {
            if (h >= '0' && h <= '9') return h - '0';
            if (h >= 'a' && h <= 'f') return h - 'a' + 10;
            return h - 'A' + 10;
        };
        decodedString += (char) ((hexVal(encodedString[i + 1]) << 4) | hexVal(encodedString[i + 2]));
        i += 2;
    }

    return decodedString;
}

bool SplitFlapWebServer::validateMasterSettings(JsonVariant &json, JsonDocument &response) {
    const int maxDisplayGroups = 6;
    const int maxModulesPerGroup = 8;

    int groupCount = settings.getInt("masterGroupCount");
    if (! json["masterGroupCount"].isNull()) {
        if (! json["masterGroupCount"].is<int>()) {
            response["message"] = "Master group count must be a number";
            response["type"] = "error";
            response["errors"]["key"] = "masterGroupCount";
            response["errors"]["message"] = "Enter a value from 1 to 6";
            return false;
        }
        groupCount = json["masterGroupCount"].as<int>();
    }

    if (groupCount < 1 || groupCount > maxDisplayGroups) {
        response["message"] = "Master group count must be between 1 and 6";
        response["type"] = "error";
        response["errors"]["key"] = "masterGroupCount";
        response["errors"]["message"] = "Enter a value from 1 to 6";
        return false;
    }

    // Inter-display link transport bounds: the web UI clamps these, but
    // POST /api/settings must also reject them — otherwise an invalid combo
    // (e.g. Direct UART with 6 groups) persists and WiredLink::init()
    // refuses it at the next boot.
    int linkTransport = settings.getInt("linkTransport");
    if (! json["linkTransport"].isNull()) {
        if (! json["linkTransport"].is<int>()) {
            response["message"] = "Link transport must be a number";
            response["type"] = "error";
            response["errors"]["key"] = "linkTransport";
            response["errors"]["message"] = "Enter 0 for ESP-NOW, 1 for RS-485, 2 for Direct UART";
            return false;
        }
        linkTransport = json["linkTransport"].as<int>();
    }

    const char *linkError = validateWiredSettings(linkTransport, groupCount);
    if (linkError != nullptr) {
        response["message"] = linkError;
        response["type"] = "error";
        response["errors"]["key"] = json["masterGroupCount"].isNull() ? "linkTransport" : "masterGroupCount";
        response["errors"]["message"] = linkError;
        return false;
    }

    if (! json["linkAddress"].isNull()) {
        if (! json["linkAddress"].is<int>()) {
            response["message"] = "Wired address must be a number";
            response["type"] = "error";
            response["errors"]["key"] = "linkAddress";
            response["errors"]["message"] = "Enter a value from 0 to 5";
            return false;
        }
        int linkAddress = json["linkAddress"].as<int>();
        if (linkAddress < 0 || linkAddress >= LINK_MAX_GROUPS) {
            response["message"] = "Wired address must be between 0 and 5";
            response["type"] = "error";
            response["errors"]["key"] = "linkAddress";
            response["errors"]["message"] = "Use 0 for the master, 1-5 for a remote group slot";
            return false;
        }
    }

    if (! json["linkBaud"].isNull()) {
        if (! json["linkBaud"].is<int>()) {
            response["message"] = "Baud rate must be a number";
            response["type"] = "error";
            response["errors"]["key"] = "linkBaud";
            response["errors"]["message"] = "Enter a baud rate of at least 1200";
            return false;
        }
        if (json["linkBaud"].as<int>() < 1200) {
            response["message"] = "Baud rate is too low";
            response["type"] = "error";
            response["errors"]["key"] = "linkBaud";
            response["errors"]["message"] = "Enter a baud rate of at least 1200";
            return false;
        }
    }

    String moduleCounts = settings.getString("masterGroupModuleCounts");
    if (! json["masterGroupModuleCounts"].isNull()) {
        if (! json["masterGroupModuleCounts"].is<String>()) {
            response["message"] = "Group module counts must be comma-separated numbers";
            response["type"] = "error";
            response["errors"]["key"] = "masterGroupModuleCounts";
            response["errors"]["message"] = "Use one number per group";
            return false;
        }
        moduleCounts = json["masterGroupModuleCounts"].as<String>();
    }

    for (int i = 0; i < groupCount; i++) {
        String token = getCsvToken(moduleCounts, i);
        if (token.length() == 0) {
            response["message"] = "Each active group needs a module count";
            response["type"] = "error";
            response["errors"]["key"] = "masterGroupModuleCounts";
            response["errors"]["message"] = "Enter 1 to 8 modules for each active group";
            return false;
        }

        int value = token.toInt();
        if (value < 1 || value > maxModulesPerGroup) {
            response["message"] = "Each group can have 1 to 8 modules";
            response["type"] = "error";
            response["errors"]["key"] = "masterGroupModuleCounts";
            response["errors"]["message"] = "Hardware limit is 8 modules per group";
            return false;
        }
    }

    String macs = settings.getString("masterGroupMacs");
    if (! json["masterGroupMacs"].isNull()) {
        if (! json["masterGroupMacs"].is<String>()) {
            response["message"] = "Group MAC addresses must be text";
            response["type"] = "error";
            response["errors"]["key"] = "masterGroupMacs";
            response["errors"]["message"] = "Use MAC addresses like AA:BB:CC:DD:EE:FF";
            return false;
        }
        macs = json["masterGroupMacs"].as<String>();
    }

    for (int i = 1; i < groupCount; i++) {
        String mac = getCsvToken(macs, i);
        if (! validateMacAddress(mac)) {
            response["message"] = "Each remote group needs a valid MAC address";
            response["type"] = "error";
            response["errors"]["key"] = "masterGroupMacs";
            response["errors"]["message"] = "Use MAC addresses like AA:BB:CC:DD:EE:FF";
            return false;
        }
    }

    return true;
}

bool SplitFlapWebServer::validateMacAddress(String macString) {
    macString.trim();
    int digitCount = 0;

    for (int i = 0; i < (int) macString.length(); i++) {
        char c = macString[i];
        if (isxdigit((unsigned char) c)) {
            digitCount++;
        } else if (c != ':' && c != '-' && c != ' ') {
            return false;
        }
    }

    return digitCount == 12;
}
