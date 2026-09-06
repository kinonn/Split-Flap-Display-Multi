#include "SplitFlapWebServer.h"

#include "CalibApi.h"
#include "CalibrationTriggers.h"
#include "CsvUtils.h"
#include "SplitFlapEspNow.h"
#include "SplitFlapModule.h"

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

String SplitFlapWebServer::getCalibLastFrame() {
    std::lock_guard<std::mutex> lock(calibMutex_);
    return calibLastFrame_;
}

void SplitFlapWebServer::setCalibLastFrame(const String &frame, int frameId) {
    std::lock_guard<std::mutex> lock(calibMutex_);
    calibLastFrame_ = frame;
    calibLastFrameId_ = frameId;
}

int SplitFlapWebServer::getCalibLastFrameId() {
    std::lock_guard<std::mutex> lock(calibMutex_);
    return calibLastFrameId_;
}

int SplitFlapWebServer::getCalibTotalModules() {
    if (espNow && isMultiDisplayMasterEnabled()) {
        return espNow->getTotalModuleCount();
    }
    return display.getNumModules();
}

void SplitFlapWebServer::registerCalibRoutes() {
    // Read-only status for the vision agent: fleet geometry, drum order,
    // live offsets (including uncommitted previews) and show progress.
    server.on("/api/calib/status", HTTP_GET, [this](AsyncWebServerRequest *request) {
        JsonDocument response;
        int charset = display.getCharsetSize();
        int drumLen = 0;
        const char *drum = SplitFlapModule::drumOrder(charset, drumLen);
        String drumStr = "";
        for (int i = 0; i < drumLen; i++) {
            drumStr += drum[i];
        }

        int localModules = display.getNumModules();
        response["contractVersion"] = CALIB_CONTRACT_VERSION;
        response["schemaVersion"] = SETTINGS_SCHEMA_VERSION;
        response["mode"] = settings.getInt("mode");
        response["holdActive"] = settings.getInt("mode") == CALIB_HOLD_MODE;
        response["busy"] = getCalibBusy();
        response["frameId"] = getCalibFrameId();
        {
            std::lock_guard<std::mutex> lock(calibMutex_);
            response["lastFrameId"] = calibLastFrameId_;
            response["lastFrame"] = calibLastFrame_;
        }
        response["numModules"] = localModules;
        response["totalModules"] = getCalibTotalModules();
        response["groupCount"] = espNow ? espNow->getTotalModuleCount() / localModules : 1;
        response["charset"] = charset;
        response["drumOrder"] = drumStr;
        response["displayOffset"] = display.getLiveDisplayOffset();
        JsonArray modOffs = response["moduleOffsets"].to<JsonArray>();
        for (int i = 0; i < localModules; i++) {
            modOffs.add(display.getLiveModuleOffset(i));
        }
        response["previewNote"] = "live offsets include uncommitted previews; reload reverts";
        request->send(200, "application/json", response.as<String>());
    });

    // Enter/leave calibration hold (mode 4): suspends date/time/random/scroll
    // writes so the agent owns the display. Fleet: call on each controller.
    server.addHandler(new AsyncCallbackJsonWebHandler(
        "/api/calib/hold",
        [this](AsyncWebServerRequest *request, JsonVariant &json) {
        if (request->method() != HTTP_POST) {
            return request->send(405, "application/json", "{\"error\":\"Method Not Allowed\"}");
        }
        if (! json["active"].is<bool>()) {
            JsonDocument response;
            response["message"] = "Invalid active flag (expected boolean)";
            response["type"] = "error";
            return request->send(400, "application/json", response.as<String>());
        }
        int previousMode = settings.getInt("mode");
        bool active = json["active"].as<bool>();
        settings.putInt("mode", active ? CALIB_HOLD_MODE : 0);
        JsonDocument response;
        response["message"] = active ? "Calibration hold engaged" : "Calibration hold released";
        response["type"] = "success";
        response["holdActive"] = active;
        response["previousMode"] = previousMode;
        request->send(200, "application/json", response.as<String>());
    }
    ));

    // Deterministic exact-width show: no centering, no scroll. On the master,
    // a fleet-width frame (length == total modules) is distributed across all
    // ESP-NOW groups left-to-right; a local-width frame shows locally only.
    server.addHandler(new AsyncCallbackJsonWebHandler(
        "/api/calib/show",
        [this](AsyncWebServerRequest *request, JsonVariant &json) {
        if (request->method() != HTTP_POST) {
            return request->send(405, "application/json", "{\"error\":\"Method Not Allowed\"}");
        }
        JsonDocument response;
        if (! json["frame"].is<String>()) {
            response["message"] = "Invalid frame (expected string)";
            response["type"] = "error";
            return request->send(400, "application/json", response.as<String>());
        }
        String frame = json["frame"].as<String>();
        int dwellMs = json["dwellMs"].is<int>() ? json["dwellMs"].as<int>() : 800;
        if (dwellMs < 0 || dwellMs > 10000) {
            response["message"] = "Invalid dwellMs (expected 0..10000)";
            response["type"] = "error";
            return request->send(400, "application/json", response.as<String>());
        }
        int localModules = display.getNumModules();
        int totalModules = getCalibTotalModules();
        bool fleetFrame = (totalModules != localModules && frame.length() == (unsigned int) totalModules);
        if (! fleetFrame && frame.length() != (unsigned int) localModules) {
            response["message"] = "Frame width must equal numModules (" + String(localModules) + ")" +
                (totalModules != localModules ? " or totalModules (" + String(totalModules) + ")" : "");
            response["type"] = "error";
            response["numModules"] = localModules;
            response["totalModules"] = totalModules;
            return request->send(400, "application/json", response.as<String>());
        }
        if (getCalibBusy() || pendingActions_.hasCalibShowPending()) {
            response["message"] = "Display busy, poll status until busy==false";
            response["type"] = "error";
            return request->send(409, "application/json", response.as<String>());
        }
        int frameId = nextCalibFrameId();
        pendingActions_.requestCalibShow(frame.c_str(), frameId);
        response["message"] = "Show queued";
        response["type"] = "success";
        response["frameId"] = frameId;
        response["fleetFrame"] = fleetFrame;
        response["dwellMs"] = dwellMs;
        request->send(202, "application/json", response.as<String>());
    }
    ));

    // Ground truth for a shown frame so the camera has expected glyphs.
    server.on("/api/calib/frame", HTTP_GET, [this](AsyncWebServerRequest *request) {
        JsonDocument response;
        if (! request->hasParam("frameId")) {
            response["message"] = "Missing frameId query parameter";
            response["type"] = "error";
            return request->send(400, "application/json", response.as<String>());
        }
        int wanted = request->getParam("frameId")->value().toInt();
        std::lock_guard<std::mutex> lock(calibMutex_);
        if (wanted != calibLastFrameId_ && wanted != calibFrameId_.load()) {
            response["message"] = "Unknown frameId";
            response["type"] = "error";
            return request->send(404, "application/json", response.as<String>());
        }
        response["frameId"] = wanted;
        response["frame"] = (wanted == calibLastFrameId_) ? calibLastFrame_ : "";
        response["busy"] = calibBusy_.load();
        response["settled"] = (wanted == calibLastFrameId_) && ! calibBusy_.load();
        request->send(200, "application/json", response.as<String>());
    });

    // Volatile preview nudge (Phase 2 dry run): RAM-only, single local
    // module, no NVS write. For fleets, call each controller directly.
    // charIndex -1 = coarse module offset, else drum index 0..charset-1.
    server.addHandler(new AsyncCallbackJsonWebHandler(
        "/api/calib/preview",
        [this](AsyncWebServerRequest *request, JsonVariant &json) {
        if (request->method() != HTTP_POST) {
            return request->send(405, "application/json", "{\"error\":\"Method Not Allowed\"}");
        }
        JsonDocument response;
        int module = json["module"].is<int>() ? json["module"].as<int>() : -1;
        int charIndex = json["charIndex"].is<int>() ? json["charIndex"].as<int>() : -1;
        int delta = json["delta"].is<int>() ? json["delta"].as<int>() : 0;
        int localModules = display.getNumModules();
        int charset = display.getCharsetSize();
        if (module < 0 || module >= localModules) {
            response["message"] = "Invalid module (expected 0.." + String(localModules - 1) + ")";
            response["type"] = "error";
            return request->send(400, "application/json", response.as<String>());
        }
        if (charIndex < -1 || charIndex >= charset) {
            response["message"] = "Invalid charIndex (expected -1.." + String(charset - 1) + ")";
            response["type"] = "error";
            return request->send(400, "application/json", response.as<String>());
        }
        if (delta == 0 || delta < -32 || delta > 32) {
            response["message"] = "Invalid delta (expected -32..32, non-zero)";
            response["type"] = "error";
            return request->send(400, "application/json", response.as<String>());
        }
        if (getCalibBusy()) {
            response["message"] = "Display busy, poll status until busy==false";
            response["type"] = "error";
            return request->send(409, "application/json", response.as<String>());
        }
        PendingActions::CalibPreview preview;
        preview.module = module;
        preview.charIndex = charIndex;
        preview.delta = delta;
        pendingActions_.requestCalibPreview(preview);
        response["message"] = "Preview queued (volatile, reload reverts)";
        response["type"] = "success";
        response["module"] = module;
        response["charIndex"] = charIndex;
        response["delta"] = delta;
        request->send(202, "application/json", response.as<String>());
    }
    ));

    // Scoped persist (Phase 3): writes ONE offset cell to NVS, then queues
    // the existing reload/push paths. scope 1 (or "local") = local group,
    // 2..6 = remote group on the master.
    server.addHandler(new AsyncCallbackJsonWebHandler(
        "/api/calib/offsets",
        [this](AsyncWebServerRequest *request, JsonVariant &json) {
        if (request->method() != HTTP_POST) {
            return request->send(405, "application/json", "{\"error\":\"Method Not Allowed\"}");
        }
        JsonDocument response;
        int group = 1;
        if (json["scope"].is<int>()) {
            group = json["scope"].as<int>();
        } else if (json["scope"].is<String>()) {
            String scope = json["scope"].as<String>();
            scope.toLowerCase();
            group = (scope == "local" || scope == "1") ? 1 : scope.toInt();
        } else if (! json["scope"].isNull()) {
            response["message"] = "Invalid scope (expected \"local\" or group 1..6)";
            response["type"] = "error";
            return request->send(400, "application/json", response.as<String>());
        }
        int groupCount = isMultiDisplayMasterEnabled() ? settings.getInt("masterGroupCount") : 1;
        groupCount = constrain(groupCount, 1, CALIB_MAX_GROUPS);
        if (group < 1 || group > groupCount) {
            response["message"] = "Invalid scope (expected \"local\" or group 1.." + String(groupCount) + ")";
            response["type"] = "error";
            return request->send(400, "application/json", response.as<String>());
        }
        if (! json["kind"].is<String>()) {
            response["message"] = "Invalid kind (expected char, module or display)";
            response["type"] = "error";
            return request->send(400, "application/json", response.as<String>());
        }
        String kind = json["kind"].as<String>();
        kind.toLowerCase();
        bool isLocal = (group == 1);
        int localModules = display.getNumModules();
        int charset = display.getCharsetSize();

        if (kind == "display") {
            if (! json["value"].is<int>()) {
                response["message"] = "Invalid value for display offset";
                response["type"] = "error";
                return request->send(400, "application/json", response.as<String>());
            }
            int value = json["value"].as<int>();
            if (isLocal) {
                settings.putInt("displayOffset", value);
                pendingActions_.requestReloadOffsets();
                if (! isMultiDisplayMasterEnabled() && espNow) {
                    pendingActions_.requestReportOffsets();
                }
            } else {
                auto dispOffs = settings.getIntVector("rDispOffs");
                while ((int) dispOffs.size() < group - 1) dispOffs.push_back(0);
                dispOffs[group - 2] = value;
                settings.putIntVector("rDispOffs", dispOffs);
                pendingActions_.requestPushOffsets();
            }
        } else if (kind == "module") {
            int module = json["module"].is<int>() ? json["module"].as<int>() : -1;
            if (! json["value"].is<int>() || module < 0 || module >= CALIB_MAX_MODULES) {
                response["message"] = "Invalid module/value for module offset";
                response["type"] = "error";
                return request->send(400, "application/json", response.as<String>());
            }
            int value = json["value"].as<int>();
            if (isLocal) {
                if (module >= localModules) {
                    response["message"] = "Invalid module (expected 0.." + String(localModules - 1) + ")";
                    response["type"] = "error";
                    return request->send(400, "application/json", response.as<String>());
                }
                auto modOffs = settings.getIntVector("moduleOffsets");
                while ((int) modOffs.size() < localModules) modOffs.push_back(0);
                modOffs[module] = value;
                settings.putIntVector("moduleOffsets", modOffs);
                pendingActions_.requestReloadOffsets();
                if (! isMultiDisplayMasterEnabled() && espNow) {
                    pendingActions_.requestReportOffsets();
                }
            } else {
                auto modOffs = settings.getIntMatrix("rModOffs");
                int row = group - 2;
                while ((int) modOffs.size() <= row) modOffs.push_back(std::vector<int>(8, 0));
                if ((int) modOffs[row].size() < 8) modOffs[row].resize(8, 0);
                modOffs[row][module] = value;
                settings.putIntMatrix("rModOffs", modOffs);
                pendingActions_.requestPushOffsets();
            }
        } else if (kind == "char") {
            int module = json["module"].is<int>() ? json["module"].as<int>() : -1;
            int charIndex = json["charIndex"].is<int>() ? json["charIndex"].as<int>() : -1;
            if (! json["value"].is<int>() || module < 0 || module >= CALIB_MAX_MODULES || charIndex < 0 ||
                charIndex >= charset) {
                response["message"] = "Invalid module/charIndex/value for char offset";
                response["type"] = "error";
                return request->send(400, "application/json", response.as<String>());
            }
            int value = constrain(json["value"].as<int>(), CALIB_CHAR_OFFSET_MIN, CALIB_CHAR_OFFSET_MAX);
            if (isLocal) {
                if (module >= localModules) {
                    response["message"] = "Invalid module (expected 0.." + String(localModules - 1) + ")";
                    response["type"] = "error";
                    return request->send(400, "application/json", response.as<String>());
                }
                auto chrOffs = settings.getIntMatrix("charOffsets");
                while ((int) chrOffs.size() < localModules) chrOffs.push_back(std::vector<int>(48, 0));
                if ((int) chrOffs[module].size() < 48) chrOffs[module].resize(48, 0);
                chrOffs[module][charIndex] = value;
                settings.putIntMatrix("charOffsets", chrOffs);
                pendingActions_.requestReloadOffsets();
                if (! isMultiDisplayMasterEnabled() && espNow) {
                    pendingActions_.requestReportOffsets();
                }
            } else {
                String key = "rChrOff" + String(group - 2);
                auto chrOffs = settings.getIntMatrix(key.c_str());
                while ((int) chrOffs.size() <= module) chrOffs.push_back(std::vector<int>(48, 0));
                if ((int) chrOffs[module].size() < 48) chrOffs[module].resize(48, 0);
                chrOffs[module][charIndex] = value;
                settings.putIntMatrix(key.c_str(), chrOffs);
                pendingActions_.requestPushOffsets();
            }
        } else {
            response["message"] = "Invalid kind (expected char, module or display)";
            response["type"] = "error";
            return request->send(400, "application/json", response.as<String>());
        }

        response["message"] = "Offset saved, applying to display";
        response["type"] = "success";
        response["scope"] = group;
        response["kind"] = kind;
        request->send(200, "application/json", response.as<String>());
    }
    ));
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

            if (! isMultiDisplayMasterEnabled() && espNow) {
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

    registerCalibRoutes();

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
