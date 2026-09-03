#include "SplitFlapWebServer.h"
#include "SplitFlapEspNow.h"

#include <ArduinoJson.h>
#include <AsyncJson.h>
#include <ctype.h>

bool isMultiDisplayMasterEnabled();

#define AP_SSID "Split Flap Display"

#ifndef WIFI_SSID
#define WIFI_SSID ""
#endif

#ifndef WIFI_PASS
#define WIFI_PASS ""
#endif

SplitFlapWebServer::SplitFlapWebServer(JsonSettings &settings, SplitFlapDisplay &display)
    : settings(settings), display(display), server(80), multiWordDelay(1000), rebootRequired(false), attemptReconnect(false),
      multiWordCurrentIndex(0), numMultiWords(0), wifiCheckInterval(1000), connectionMode(0), checkDateInterval(250),
      centering(1) {
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

// Totally didn't use AI to make these functions
//  Function to get current minute as a string
String SplitFlapWebServer::getCurrentMinute() {
    struct tm timeinfo;
    if (! getLocalTime(&timeinfo)) {
        return "";
    }
    char minuteStr[3];                           // Max "59" + null terminator
    sprintf(minuteStr, "%02d", timeinfo.tm_min); // Format as two-digit string
    return String(minuteStr);
}

// Function to get current hour as a string
String SplitFlapWebServer::getCurrentHour() {
    struct tm timeinfo;
    if (! getLocalTime(&timeinfo)) {
        return "";
    }
    char hourStr[3];                            // Max "59" + null terminator
    sprintf(hourStr, "%02d", timeinfo.tm_hour); // Format as two-digit string
    return String(hourStr);
}

// Function to get the first n characters of the day
String SplitFlapWebServer::getDayPrefix(int n) {
    struct tm timeinfo;
    if (! getLocalTime(&timeinfo)) {
        return "Err"; // Return error if time not available
    }

    // Get full weekday name
    char fullDay[10]; // Buffer for full day name
    strftime(fullDay, sizeof(fullDay), "%A", &timeinfo);

    // Extract first n characters
    char dayPrefix[n + 1];
    strncpy(dayPrefix, fullDay, n);
    dayPrefix[n] = '\0'; // Null-terminate the string

    return String(dayPrefix);
}

// Function to get the first n characters of the month
String SplitFlapWebServer::getMonthPrefix(int n) {
    struct tm timeinfo;
    if (! getLocalTime(&timeinfo)) {
        return "Err"; // Return error if time not available
    }

    // Get full month name
    char fullMonth[10]; // Buffer for full month name
    strftime(fullMonth, sizeof(fullMonth), "%B", &timeinfo);

    // Extract first n characters
    char monthPrefix[n + 1];
    strncpy(monthPrefix, fullMonth, n);
    monthPrefix[n] = '\0'; // Null-terminate the string

    return String(monthPrefix);
}

String SplitFlapWebServer::getCurrentDay() {
    struct tm timeinfo;
    if (! getLocalTime(&timeinfo)) {
        return "Err";                          // Return error if time is not available
    }

    char dayStr[3];                            // Buffer for the day number (max "31" + null terminator)
    sprintf(dayStr, "%02d", timeinfo.tm_mday); // Format as two-digit string

    return String(dayStr);
}

void SplitFlapWebServer::setMode(int targetMode) {
    settings.putInt("mode", targetMode);
}

int SplitFlapWebServer::getMode() {
    return settings.getInt("mode");
}

void SplitFlapWebServer::checkWiFi() {
    if (connectionMode == 1) {
        if (WiFi.status() != WL_CONNECTED) {
            Serial.println("Wi-Fi lost! Forcing reconnect...");
            WiFi.disconnect();
            WiFi.reconnect();
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
        WiFi.mode(WIFI_STA);
#ifdef WIFI_TX_POWER
        delay(100);
        WiFi.setTxPower((wifi_power_t) WIFI_TX_POWER);
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
    if (loadWiFiCredentials()) {
        unsigned long startAttemptTime = millis();
        const unsigned long timeout = 20000; // 20 seconds
        unsigned long lastPrintTime = startAttemptTime;

        while (WiFi.status() != WL_CONNECTED) {
            if (millis() - startAttemptTime >= timeout) {
                Serial.println("_");
                Serial.println("Wi-Fi connection failed! Timeout reached.");
                return false; // Return false if unable to connect in 30 seconds
            }
            if ((millis() - lastPrintTime) > 1000) {
                Serial.print(".");
                lastPrintTime = millis();
            }
            yield();
        }

        // connected succesfully
        connectionMode = 1;
        WiFi.softAPdisconnect(); // Turns off SoftAP mode only after connected to
        // actual network
        WiFi.setAutoReconnect(true);
        WiFi.persistent(true); // Saves Wi-Fi settings to flash memory
        WiFi.setSleep(false);
        Serial.println("Connected to Wi-Fi!");
        Serial.println("IP Address: http://" + WiFi.localIP().toString());
        return true;
    }
    return false;
}

void SplitFlapWebServer::startAccessPoint() {
    connectionMode = 0;
    const char *apSSID = AP_SSID;
    WiFi.softAP(apSSID);
#ifdef WIFI_TX_POWER
    delay(100);
    WiFi.setTxPower((wifi_power_t) WIFI_TX_POWER);
#endif
    Serial.println("AP Mode Started!");
    Serial.println("Connect to: " + String(apSSID));
    Serial.println("AP IP Address: http://" + WiFi.softAPIP().toString());
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
        if (espNow) {
            JsonArray peers = response["discoveredPeers"].to<JsonArray>();
            JsonDocument peersDoc;
            deserializeJson(peersDoc, espNow->getDiscoveredPeersJson());
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

        bool offsetsChanged = false;
        if (json["moduleOffsets"].is<String>() && json["moduleOffsets"].as<String>() != settings.getString("moduleOffsets")) {
            offsetsChanged = true;
        }
        if (json["charOffsets"].is<String>() && json["charOffsets"].as<String>() != settings.getString("charOffsets")) {
            offsetsChanged = true;
        }
        if (json["displayOffset"].is<int>() && json["displayOffset"].as<int>() != settings.getInt("displayOffset")) {
            offsetsChanged = true;
        }

        bool remoteOffsetsChanged = false;
        const char *remoteOffsetKeys[] = {"rModOffs", "rChrOff0", "rChrOff1", "rChrOff2", "rChrOff3", "rChrOff4", "rDispOffs"};
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
            for (auto &row : chrOffs) for (auto &v : row) v = constrain(v, -32, 32);
            settings.putIntMatrix("charOffsets", chrOffs);

            // Defer the display work to the loop task: reloadOffsets() homes
            // modules (seconds of Wire I/O) and must not run in the AsyncTCP
            // task racing the loop task's own display access.
            pendingActions_.requestReloadOffsets();
            response["message"] = "Settings saved, offsets will be applied";

            if (!isMultiDisplayMasterEnabled() && espNow) {
                pendingActions_.requestReportOffsets();
            }
        }

        if (remoteOffsetsChanged && isMultiDisplayMasterEnabled()) {
            for (int r = 0; r < 5; r++) {
                String key = "rChrOff" + String(r);
                auto rChrOffs = settings.getIntMatrix(key.c_str());
                for (auto &row : rChrOffs) for (auto &v : row) v = constrain(v, -32, 32);
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
        if (
            c != '%' || i + 2 >= encodedString.length() ||
            ! isxdigit((unsigned char) encodedString[i + 1]) ||
            ! isxdigit((unsigned char) encodedString[i + 2])
        ) {
            decodedString += c;
            continue;
        }

        auto hexVal = [](char h) -> int {
            if (h >= '0' && h <= '9') return h - '0';
            if (h >= 'a' && h <= 'f') return h - 'a' + 10;
            return h - 'A' + 10;
        };
        decodedString += (char) ((hexVal(encodedString[i + 1]) << 4) |
                                 hexVal(encodedString[i + 2]));
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

String SplitFlapWebServer::getCsvToken(const String &csv, int index) {
    int tokenStart = 0;
    int tokenIndex = 0;

    for (int i = 0; i <= (int) csv.length(); i++) {
        if (i == (int) csv.length() || csv[i] == ',') {
            if (tokenIndex == index) {
                String token = csv.substring(tokenStart, i);
                token.trim();
                return token;
            }
            tokenStart = i + 1;
            tokenIndex++;
        }
    }

    return "";
}
