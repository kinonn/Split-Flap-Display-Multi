import Alpine from "alpinejs";
window.Alpine = Alpine;

// Must stay in sync with SETTINGS_SCHEMA_VERSION in src/JsonSettings.h.
const SETTINGS_SCHEMA_VERSION = 1;
const EXPORT_FORMAT = "splitflap-settings";

// Describes every setting the firmware understands, used to validate imported
// config files and to flag unknown/missing keys.
const SETTINGS_SCHEMA = {
    name: { type: "string" },
    mdns: { type: "string" },
    otaPass: { type: "string" },
    timezone: { type: "string" },
    dateFormat: { type: "string" },
    timeFormat: { type: "string" },
    ssid: { type: "string" },
    password: { type: "string" },
    mqtt_server: { type: "string" },
    mqtt_port: { type: "int", min: 1, max: 65535 },
    mqtt_user: { type: "string" },
    mqtt_pass: { type: "string" },
    moduleCount: { type: "int", min: 1, max: 8 },
    moduleAddresses: { type: "vector" },
    magnetPosition: { type: "int" },
    moduleOffsets: { type: "vector" },
    displayOffset: { type: "int" },
    sdaPin: { type: "int" },
    sclPin: { type: "int" },
    stepsPerRot: { type: "int" },
    maxVel: { type: "float" },
    charset: { type: "int", enum: [37, 48] },
    charOffsets: { type: "matrix" },
    scrollDelayMs: { type: "int", min: 0 },
    scrollRepeatCount: { type: "int", min: 1, max: 99 },
    bootDelayMs: { type: "int", min: 0 },
    maxConcurrentMotors: { type: "int", min: 1, max: 8 },
    masterGroupCount: { type: "int", min: 1, max: 6 },
    masterGroupModuleCounts: { type: "vector" },
    masterGroupMacs: { type: "vector" },
    rModOffs: { type: "matrix" },
    rChrOff0: { type: "matrix" },
    rChrOff1: { type: "matrix" },
    rChrOff2: { type: "matrix" },
    rChrOff3: { type: "matrix" },
    rChrOff4: { type: "matrix" },
    rDispOffs: { type: "vector" },
    mode: { type: "int", min: 0, max: 5 },
};

document.addEventListener("alpine:init", () => {
    Alpine.data("page", (type) => ({
        get header() {
            return this.settings.name || "Split Flap";
        },

        loading: {
            settings: true,
            timezones: true,
        },
        saving: false,
        dialog: {
            show: false,
            message: "",
            type: null,
        },
        settings: {
            mode: 2,
            dateFormat: "ddd dd/MM",
            timeFormat: "{HH}:{MM}",
            masterGroupCount: 1,
            masterGroupModuleCounts: "8,8,8,8,8,8",
            masterGroupMacs: ",,,,,",
        },
        localMac: "",
        discoveredPeers: [],
        errors: {},
        timezones: {},
        selectedOffsetGroup: 0,
        schemaVersion: SETTINGS_SCHEMA_VERSION,
        showImportModal: false,
        importPreview: null,

        // Control page specific
        singleMode: true,
        singleWord: "",
        multiWord: "",
        multiWords: [],
        delay: 1,
        centerText: false,

        get processing() {
            return (
                this.saving || this.loading.settings || this.loading.timezones
            );
        },

        get hasImportErrors() {
            return (
                (this.importPreview &&
                    this.importPreview.warnings.some(
                        (w) => w.type === "error",
                    )) ||
                false
            );
        },

        get addressArray() {
            return (
                this.settings.moduleAddresses
                    ?.split(",")
                    .map((s) => s.trim()) || []
            );
        },
        setAddress(index, value) {
            const arr = this.addressArray;
            arr[index] = value;
            this.settings.moduleAddresses = arr.join(",");
        },

        get offsetArray() {
            return (
                this.settings.moduleOffsets?.split(",").map((s) => s.trim()) ||
                []
            );
        },
        setOffset(index, value) {
            const arr = this.offsetArray;
            arr[index] = value;
            this.settings.moduleOffsets = arr.join(",");
        },

        get charOffsetMatrix() {
            if (!this.settings.charOffsets) return [];
            return this.settings.charOffsets
                .split(";")
                .map((r) =>
                    r.length === 0
                        ? []
                        : r.split(",").map((v) => parseInt(v) || 0),
                );
        },
        setCharOffset(modIndex, charIndex, value) {
            const matrix = this.charOffsetMatrix;
            while (matrix.length <= modIndex) matrix.push([]);
            while (matrix[modIndex].length <= charIndex)
                matrix[modIndex].push(0);
            matrix[modIndex][charIndex] = parseInt(value) || 0;
            this.settings.charOffsets = matrix
                .map((r) => r.join(","))
                .join(";");
        },
        resetCharOffsets(modIndex) {
            const matrix = this.charOffsetMatrix;
            const numChars = this.settings.charset || 48;
            while (matrix.length <= modIndex) matrix.push([]);
            matrix[modIndex] = Array(numChars).fill(0);
            this.settings.charOffsets = matrix
                .map((r) => r.join(","))
                .join(";");
        },
        copyCharOffsets(fromIndex, toIndex) {
            const matrix = this.charOffsetMatrix;
            if (fromIndex >= matrix.length) return;
            while (matrix.length <= toIndex) matrix.push([]);
            matrix[toIndex] = [...matrix[fromIndex]];
            this.settings.charOffsets = matrix
                .map((r) => r.join(","))
                .join(";");
        },

        get remoteOffsetMatrix() {
            if (!this.settings.rModOffs) return [];
            return this.settings.rModOffs
                .split(";")
                .map((r) =>
                    r.length === 0
                        ? []
                        : r.split(",").map((v) => parseInt(v) || 0),
                );
        },
        setRemoteOffset(groupRow, modIndex, value) {
            const matrix = this.remoteOffsetMatrix;
            while (matrix.length <= groupRow) matrix.push(Array(8).fill(0));
            while (matrix[groupRow].length <= modIndex)
                matrix[groupRow].push(0);
            matrix[groupRow][modIndex] = parseInt(value) || 0;
            this.settings.rModOffs = matrix.map((r) => r.join(",")).join(";");
        },

        get remoteDisplayOffsetArray() {
            return (
                this.settings.rDispOffs
                    ?.split(",")
                    .map((s) => parseInt(s.trim()) || 0) || []
            );
        },
        setRemoteDisplayOffset(groupRow, value) {
            const arr = this.remoteDisplayOffsetArray;
            while (arr.length <= groupRow) arr.push(0);
            arr[groupRow] = parseInt(value) || 0;
            this.settings.rDispOffs = arr.join(",");
        },

        getRemoteCharOffsetMatrix(groupRow) {
            const key = "rChrOff" + groupRow;
            if (!this.settings[key]) return [];
            return this.settings[key]
                .split(";")
                .map((r) =>
                    r.length === 0
                        ? []
                        : r.split(",").map((v) => parseInt(v) || 0),
                );
        },
        setRemoteCharOffset(groupRow, modIndex, charIndex, value) {
            const key = "rChrOff" + groupRow;
            const matrix = this.getRemoteCharOffsetMatrix(groupRow);
            while (matrix.length <= modIndex) matrix.push([]);
            while (matrix[modIndex].length <= charIndex)
                matrix[modIndex].push(0);
            matrix[modIndex][charIndex] = Math.max(
                -32,
                Math.min(32, parseInt(value) || 0),
            );
            this.settings[key] = matrix.map((r) => r.join(",")).join(";");
        },
        resetRemoteCharOffsets(groupRow, modIndex) {
            const key = "rChrOff" + groupRow;
            const matrix = this.getRemoteCharOffsetMatrix(groupRow);
            const numChars = this.settings.charset || 48;
            while (matrix.length <= modIndex) matrix.push([]);
            matrix[modIndex] = Array(numChars).fill(0);
            this.settings[key] = matrix.map((r) => r.join(",")).join(";");
        },
        copyRemoteCharOffsets(groupRow, fromIndex, toIndex) {
            const key = "rChrOff" + groupRow;
            const matrix = this.getRemoteCharOffsetMatrix(groupRow);
            if (fromIndex >= matrix.length) return;
            while (matrix.length <= toIndex) matrix.push([]);
            matrix[toIndex] = [...matrix[fromIndex]];
            this.settings[key] = matrix.map((r) => r.join(",")).join(";");
        },

        get offsetModuleCount() {
            if (this.selectedOffsetGroup === 0)
                return this.settings.moduleCount;
            return (
                parseInt(this.groupModuleArray[this.selectedOffsetGroup]) || 8
            );
        },

        get currentOffsetArray() {
            if (this.selectedOffsetGroup === 0) return this.offsetArray;
            const matrix = this.remoteOffsetMatrix;
            return matrix[this.selectedOffsetGroup - 1] || [];
        },
        setCurrentOffset(index, value) {
            if (this.selectedOffsetGroup === 0) {
                this.setOffset(index, value);
            } else {
                this.setRemoteOffset(
                    this.selectedOffsetGroup - 1,
                    index,
                    value,
                );
            }
        },

        get currentDisplayOffset() {
            if (this.selectedOffsetGroup === 0)
                return this.settings.displayOffset;
            const arr = this.remoteDisplayOffsetArray;
            return arr[this.selectedOffsetGroup - 1] || 0;
        },
        setCurrentDisplayOffset(value) {
            if (this.selectedOffsetGroup === 0) {
                this.settings.displayOffset = parseInt(value) || 0;
            } else {
                this.setRemoteDisplayOffset(
                    this.selectedOffsetGroup - 1,
                    value,
                );
            }
        },

        getCurrentCharOffset(modIdx, charIdx) {
            if (this.selectedOffsetGroup === 0)
                return this.getCharOffset(modIdx, charIdx);
            const matrix = this.getRemoteCharOffsetMatrix(
                this.selectedOffsetGroup - 1,
            );
            if (
                modIdx >= matrix.length ||
                charIdx >= (matrix[modIdx]?.length || 0)
            )
                return 0;
            return matrix[modIdx][charIdx];
        },
        setCurrentCharOffset(modIdx, charIdx, value) {
            if (this.selectedOffsetGroup === 0) {
                this.setCharOffset(modIdx, charIdx, value);
            } else {
                this.setRemoteCharOffset(
                    this.selectedOffsetGroup - 1,
                    modIdx,
                    charIdx,
                    value,
                );
            }
        },
        resetCurrentCharOffsets(modIdx) {
            if (this.selectedOffsetGroup === 0) {
                this.resetCharOffsets(modIdx);
            } else {
                this.resetRemoteCharOffsets(
                    this.selectedOffsetGroup - 1,
                    modIdx,
                );
            }
        },
        copyCurrentCharOffsets(fromIndex, toIndex) {
            if (this.selectedOffsetGroup === 0) {
                this.copyCharOffsets(fromIndex, toIndex);
            } else {
                this.copyRemoteCharOffsets(
                    this.selectedOffsetGroup - 1,
                    fromIndex,
                    toIndex,
                );
            }
        },
        allZeroCurrentCharOffsets(modIdx) {
            if (this.selectedOffsetGroup === 0)
                return this.allZeroCharOffsets(modIdx);
            const matrix = this.getRemoteCharOffsetMatrix(
                this.selectedOffsetGroup - 1,
            );
            if (modIdx >= matrix.length) return true;
            return matrix[modIdx].every((v) => v === 0);
        },

        get charsetChars() {
            const standard = " ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
            const extended = standard + "':?!.-/$@#%";
            return (this.settings.charset || 48) === 48 ? extended : standard;
        },

        get groupModuleArray() {
            const arr =
                this.settings.masterGroupModuleCounts
                    ?.split(",")
                    .map((s) => s.trim()) || [];
            while (arr.length < 6) {
                arr.push("8");
            }
            arr[0] = String(this.settings.moduleCount || arr[0] || 8);
            return arr.slice(0, 6);
        },
        setGroupModuleCount(index, value) {
            const arr = this.groupModuleArray;
            arr[index] = value;
            if (index === 0) {
                this.settings.moduleCount = value;
            }
            this.settings.masterGroupModuleCounts = arr.join(",");
        },

        get groupMacArray() {
            const arr =
                this.settings.masterGroupMacs
                    ?.split(",")
                    .map((s) => s.trim()) || [];
            while (arr.length < 6) {
                arr.push("");
            }
            return arr.slice(0, 6);
        },
        setGroupMac(index, value) {
            const arr = this.groupMacArray;
            arr[index] = value;
            this.settings.masterGroupMacs = arr.join(",");
        },

        normalizeMasterGroups() {
            const groupCount = Number(this.settings.masterGroupCount || 1);
            this.settings.masterGroupCount = Math.min(
                Math.max(groupCount, 1),
                6,
            );

            const counts = this.groupModuleArray;
            counts[0] = String(this.settings.moduleCount || counts[0] || 8);
            this.settings.masterGroupModuleCounts = counts.join(",");

            const macs = this.groupMacArray;
            this.settings.masterGroupMacs = macs.join(",");
        },

        isPeerAssigned(mac) {
            const normalized = (mac || "").toUpperCase();
            for (let i = 1; i < this.settings.masterGroupCount; i++) {
                const groupMac = (this.groupMacArray[i] || "").toUpperCase();
                if (groupMac && groupMac === normalized) return true;
            }
            return false;
        },

        assignedGroupLabel(mac) {
            const normalized = (mac || "").toUpperCase();
            for (let i = 1; i < this.settings.masterGroupCount; i++) {
                const groupMac = (this.groupMacArray[i] || "").toUpperCase();
                if (groupMac === normalized)
                    return `Assigned to Group ${i + 1}`;
            }
            return "";
        },

        assignPeer(groupIndex, mac) {
            this.setGroupMac(groupIndex, mac);
            const peer = this.discoveredPeers.find((p) => p.mac === mac);
            if (peer) {
                this.setGroupModuleCount(groupIndex, peer.moduleCount);
            }
            this.showDialog(
                `Assigned to Group ${groupIndex + 1}. Click Save to persist.`,
                "success",
            );
        },

        removePeer(groupIndex) {
            this.setGroupMac(groupIndex, "");
            this.setGroupModuleCount(groupIndex, "8");
            this.showDialog(
                `Removed Group ${groupIndex + 1} device. Click Save to persist.`,
                "success",
            );
        },

        moveGroupUp(index) {
            if (index <= 1) return;
            const macs = this.groupMacArray;
            const counts = this.groupModuleArray;
            [macs[index - 1], macs[index]] = [macs[index], macs[index - 1]];
            [counts[index - 1], counts[index]] = [
                counts[index],
                counts[index - 1],
            ];
            this.settings.masterGroupMacs = macs.join(",");
            this.settings.masterGroupModuleCounts = counts.join(",");
        },

        moveGroupDown(index) {
            if (index >= this.settings.masterGroupCount - 1) return;
            const macs = this.groupMacArray;
            const counts = this.groupModuleArray;
            [macs[index + 1], macs[index]] = [macs[index], macs[index + 1]];
            [counts[index + 1], counts[index]] = [
                counts[index],
                counts[index + 1],
            ];
            this.settings.masterGroupMacs = macs.join(",");
            this.settings.masterGroupModuleCounts = counts.join(",");
        },

        init() {
            this.loadSettings();
            if (type === "Settings") {
                this.loadTimezones();
            }
        },

        loadSettings() {
            fetch("/settings")
                .then((res) => res.json())
                .then((data) => {
                    const settingsData = data.settings || {};
                    Object.assign(this.settings, settingsData);
                    this.localMac = data.localMac || "";
                    this.schemaVersion =
                        data.schemaVersion || SETTINGS_SCHEMA_VERSION;
                    this.discoveredPeers = data.discoveredPeers || [];
                    this.normalizeMasterGroups();
                })
                .catch(() =>
                    this.showDialog("Failed to load settings", "error", true),
                )
                .finally(() => {
                    this.loading.settings = false;
                });
        },

        loadTimezones() {
            fetch("/timezones.json")
                .then((res) => res.json())
                .then((data) => {
                    this.timezones = data;
                })
                .catch(() =>
                    this.showDialog(
                        "Failed to load timezones. Refresh the page.",
                        "error",
                        true,
                    ),
                )
                .finally(() => (this.loading.timezones = false));
        },

        updateDisplay() {
            if (this.settings.mode === 0) {
                if (this.delay < 1) {
                    return this.showDialog(
                        "Delay must be at least 1 second.",
                        "error",
                    );
                }

                if (this.singleMode && this.singleWord.trim() === "") {
                    return this.showDialog(
                        "Single word cannot be empty.",
                        "error",
                    );
                }

                if (!this.singleMode && this.multiWords.length === 0) {
                    return this.showDialog(
                        "Word list cannot be empty.",
                        "error",
                    );
                }
            }

            fetch("/settings", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ mode: this.settings.mode }),
            });

            if (this.settings.mode === 0) {
                fetch("/text", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        mode: this.singleMode ? "single" : "multiple",
                        words: this.singleMode
                            ? [this.singleWord]
                            : this.multiWords,
                        delay: this.delay,
                        center: this.centerText,
                    }),
                })
                    .then((res) => res.json())
                    .then((res) => this.showDialog(res.message, res.type))
                    .catch((err) => this.showDialog(err.message, "error"));
            } else {
                this.showDialog("Mode updated successfully.", "success");
            }
        },

        addWord() {
            if (this.multiWord.trim() !== "") {
                this.multiWords.push(this.multiWord.trim());
            }
            this.multiWord = "";
        },

        removeWord(index) {
            this.multiWords.splice(index, 1);
        },

        save() {
            this.saving = true;
            this.errors = {};
            this.normalizeMasterGroups();

            fetch("/settings", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(this.settings),
            })
                .then((res) => res.json())
                .then((data) => {
                    this.errors = data.errors || {};
                    this.showDialog(data.message, data.type, data.persistent);
                    if (data.redirect) {
                        setTimeout(() => {
                            window.location.href = data.redirect;
                        }, 10000);
                    }
                })
                .catch(() =>
                    this.showDialog("Failed to save settings.", "error"),
                )
                .finally(() => (this.saving = false));
        },

        reset() {
            if (
                confirm("Are you sure you want to reset settings to defaults?")
            ) {
                fetch("/settings/reset", { method: "POST" })
                    .then((res) => res.json())
                    .then((data) => {
                        this.showDialog(
                            data.message,
                            data.type,
                            data.persistent,
                        );
                        this.loadSettings();
                    })
                    .catch(() => {
                        this.showDialog("Failed to reset settings.", "error");
                    });
            }
        },

        buildExportPayload() {
            return {
                format: EXPORT_FORMAT,
                schemaVersion: SETTINGS_SCHEMA_VERSION,
                exportedAt: new Date().toISOString(),
                device: {
                    name: this.settings.name || "",
                    mac: this.localMac || "",
                },
                settings: { ...this.settings },
            };
        },

        downloadJson(filename, obj) {
            const blob = new Blob([JSON.stringify(obj, null, 2)], {
                type: "application/json",
            });
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            a.remove();
            setTimeout(() => URL.revokeObjectURL(url), 1000);
        },

        exportSettings() {
            const payload = this.buildExportPayload();
            const filename = `splitflap-settings-${new Date()
                .toISOString()
                .slice(0, 10)}.json`;
            this.downloadJson(filename, payload);
            this.showDialog("Configuration exported successfully.", "success");
        },

        onImportFileSelected(event) {
            const file = event.target.files && event.target.files[0];
            if (!file) {
                return;
            }
            const reader = new FileReader();
            reader.onload = () => {
                let data;
                try {
                    data = JSON.parse(reader.result);
                } catch (err) {
                    this.showDialog(
                        "Invalid configuration file: not valid JSON.",
                        "error",
                    );
                    return;
                }
                this.previewImport(data);
            };
            reader.onerror = () =>
                this.showDialog("Could not read the selected file.", "error");
            reader.readAsText(file);
            event.target.value = "";
        },

        previewImport(data) {
            const analysis = this.analyzeImport(data);
            if (analysis.fatal) {
                this.showDialog(analysis.fatal, "error");
                return;
            }
            this.importPreview = analysis;
            this.showImportModal = true;
        },

        closeImportModal() {
            this.showImportModal = false;
            this.importPreview = null;
        },

        validateSettingValue(key, value) {
            const s = SETTINGS_SCHEMA[key];
            const isNum = typeof value === "number" && isFinite(value);
            if (s.type === "int" || s.type === "float") {
                if (!isNum) {
                    return "expected a number.";
                }
                if (s.type === "int" && !Number.isInteger(value)) {
                    return "expected a whole number.";
                }
                if (s.min !== undefined && value < s.min) {
                    return `must be at least ${s.min}.`;
                }
                if (s.max !== undefined && value > s.max) {
                    return `must be at most ${s.max}.`;
                }
                if (s.enum && !s.enum.includes(value)) {
                    return `must be one of: ${s.enum.join(", ")}.`;
                }
            } else if (s.type === "string") {
                if (typeof value !== "string") {
                    return "expected text.";
                }
            } else if (s.type === "vector" || s.type === "matrix") {
                if (typeof value !== "string") {
                    return "expected a comma-separated string.";
                }
            }
            return null;
        },

        addImportConflicts(imported, warnings) {
            const moduleCount = Number(imported.moduleCount) || 0;
            const charset = Number(imported.charset) || 48;
            const groupCount = Math.min(
                Math.max(Number(imported.masterGroupCount) || 1, 1),
                6,
            );

            const csv = (value, len) => {
                if (typeof value !== "string") return Array(len).fill("");
                const arr = value.split(",").map((s) => s.trim());
                while (arr.length < len) arr.push("");
                return arr.slice(0, len);
            };

            const counts = csv(imported.masterGroupModuleCounts, 6);
            if (
                moduleCount >= 1 &&
                counts[0] &&
                Number(counts[0]) !== moduleCount
            ) {
                warnings.push({
                    type: "conflict",
                    title: "Module count mismatch",
                    message: `Group 1 declares ${counts[0]} module(s) but "Number of Modules" is ${moduleCount}. The module count will be used.`,
                });
            }

            if (typeof imported.charOffsets === "string") {
                const firstRow = imported.charOffsets.split(";")[0];
                if (firstRow) {
                    const cols = firstRow.split(",").length;
                    if (cols > 0 && cols !== charset) {
                        warnings.push({
                            type: "conflict",
                            title: "Character offsets size",
                            message: `Character offsets contain ${cols} characters per module but the character set is ${charset}. Extra entries will be ignored.`,
                        });
                    }
                }
            }

            for (const key of ["moduleAddresses", "moduleOffsets"]) {
                if (typeof imported[key] === "string") {
                    const n = imported[key]
                        .split(",")
                        .filter((s) => s.trim() !== "").length;
                    if (moduleCount > 0 && n > 0 && n < moduleCount) {
                        const what =
                            key === "moduleAddresses"
                                ? "address(es)"
                                : "offset(s)";
                        warnings.push({
                            type: "conflict",
                            title: "Short module list",
                            message: `Only ${n} module ${what} are defined but "Number of Modules" is ${moduleCount}.`,
                        });
                    }
                }
            }

            if (groupCount > 1) {
                const macs = csv(imported.masterGroupMacs, 6);
                const seen = new Set();
                for (let i = 1; i < groupCount; i++) {
                    const mac = (macs[i] || "").trim().toUpperCase();
                    if (!mac) {
                        warnings.push({
                            type: "conflict",
                            title: "Missing peer MAC",
                            message: `Group ${i + 1} is active but has no peer MAC address assigned.`,
                        });
                        continue;
                    }
                    if (this.localMac && mac === this.localMac.toUpperCase()) {
                        warnings.push({
                            type: "conflict",
                            title: "Self-assigned peer",
                            message: `Group ${i + 1} is assigned this display's own MAC address.`,
                        });
                    }
                    if (seen.has(mac)) {
                        warnings.push({
                            type: "conflict",
                            title: "Duplicate peer MAC",
                            message: `Multiple groups use the same peer MAC address ${macs[i]}.`,
                        });
                    }
                    seen.add(mac);
                }
            }

            if (
                typeof imported.ssid === "string" &&
                imported.ssid !== (this.settings.ssid || "")
            ) {
                warnings.push({
                    type: "conflict",
                    title: "Wi-Fi change",
                    message:
                        "Importing will change the Wi-Fi network and reconnect the display.",
                });
            }

            if (
                typeof imported.otaPass === "string" &&
                imported.otaPass !== (this.settings.otaPass || "")
            ) {
                warnings.push({
                    type: "conflict",
                    title: "OTA password change",
                    message:
                        "Importing will change the OTA password and reboot the display.",
                });
            }

            if (
                typeof imported.mdns === "string" &&
                imported.mdns !== (this.settings.mdns || "")
            ) {
                warnings.push({
                    type: "conflict",
                    title: "mDNS name change",
                    message:
                        "Importing will change the mDNS hostname and reconnect the display.",
                });
            }

            if (Number(imported.mode) !== Number(this.settings.mode)) {
                warnings.push({
                    type: "info",
                    title: "Display mode change",
                    message: `The display mode will change to ${Number(imported.mode)} after import.`,
                });
            }
        },

        analyzeImport(data) {
            const warnings = [];

            if (
                data === null ||
                typeof data !== "object" ||
                Array.isArray(data)
            ) {
                return {
                    fatal: "The file does not contain a valid configuration object.",
                };
            }

            let imported = data;
            let sourceSchemaVersion = null;
            let sourceDeviceMac = "";
            let sourceDeviceName = "";

            if (
                data.settings &&
                typeof data.settings === "object" &&
                !Array.isArray(data.settings)
            ) {
                if (data.format && data.format !== EXPORT_FORMAT) {
                    return {
                        fatal: `Unrecognized configuration format "${data.format}".`,
                    };
                }
                imported = data.settings;
                sourceSchemaVersion =
                    typeof data.schemaVersion === "number"
                        ? data.schemaVersion
                        : null;
                sourceDeviceMac = data.device?.mac || "";
                sourceDeviceName = data.device?.name || "";
            }

            if (sourceSchemaVersion === null) {
                warnings.push({
                    type: "compat",
                    title: "Legacy format",
                    message:
                        "This file was saved by an older version and has no schema version. Settings missing from the file will use current defaults.",
                });
            } else if (sourceSchemaVersion < SETTINGS_SCHEMA_VERSION) {
                warnings.push({
                    type: "compat",
                    title: "Older schema",
                    message: `This file uses an older configuration schema (v${sourceSchemaVersion}, current is v${SETTINGS_SCHEMA_VERSION}). Settings added since then will use current defaults.`,
                });
            } else if (sourceSchemaVersion > SETTINGS_SCHEMA_VERSION) {
                warnings.push({
                    type: "compat",
                    title: "Newer schema",
                    message: `This file was exported from a newer configuration schema (v${sourceSchemaVersion}, current is v${SETTINGS_SCHEMA_VERSION}). Some settings may not be recognized by this device.`,
                });
            }

            if (
                sourceDeviceMac &&
                this.localMac &&
                sourceDeviceMac.toUpperCase() !== this.localMac.toUpperCase()
            ) {
                warnings.push({
                    type: "conflict",
                    title: "Different device",
                    message: `This configuration was exported from another display (${sourceDeviceMac}${sourceDeviceName ? " - " + sourceDeviceName : ""}). Wi-Fi credentials and peer MAC addresses may not be appropriate for this device.`,
                });
            }

            const missingKeys = [];
            const unknownKeys = [];
            for (const key of Object.keys(SETTINGS_SCHEMA)) {
                if (!(key in imported)) missingKeys.push(key);
            }
            for (const key of Object.keys(imported)) {
                if (!(key in SETTINGS_SCHEMA)) unknownKeys.push(key);
            }

            if (missingKeys.length) {
                warnings.push({
                    type: "info",
                    title: "Missing settings",
                    message: `${missingKeys.length} setting(s) are missing from the file and will use current/default values (${missingKeys.slice(0, 8).join(", ")}${missingKeys.length > 8 ? ", ..." : ""}).`,
                });
            }

            if (unknownKeys.length) {
                warnings.push({
                    type: "info",
                    title: "Unknown settings",
                    message: `${unknownKeys.length} setting(s) in the file are not recognized by this device and will be ignored (${unknownKeys.slice(0, 8).join(", ")}${unknownKeys.length > 8 ? ", ..." : ""}).`,
                });
            }

            for (const key of Object.keys(SETTINGS_SCHEMA)) {
                if (!(key in imported)) continue;
                const err = this.validateSettingValue(key, imported[key]);
                if (err) {
                    warnings.push({
                        type: "error",
                        title: "Invalid value",
                        message: `"${key}" ${err}`,
                    });
                }
            }

            this.addImportConflicts(imported, warnings);

            return { settings: imported, warnings };
        },

        applyImport() {
            if (!this.importPreview) {
                return;
            }
            if (this.importPreview.warnings.some((w) => w.type === "error")) {
                this.showDialog(
                    "Cannot import: the configuration contains invalid values.",
                    "error",
                );
                return;
            }

            const imported = this.importPreview.settings;
            const merged = { ...this.settings };
            for (const key of Object.keys(SETTINGS_SCHEMA)) {
                if (key in imported) {
                    merged[key] = imported[key];
                }
            }
            this.settings = merged;
            this.normalizeMasterGroups();
            this.importPreview = null;
            this.showImportModal = false;
            this.save();
        },

        showDialog(message, type = "success", persistent = false) {
            this.dialog.message = message;
            this.dialog.type = type;
            this.dialog.show = true;

            if (!persistent) {
                setTimeout(() => (this.dialog.show = false), 3000);
            }
        },
    }));

    Alpine.data("helpModal", () => ({
        visible: false,
        title: "",
        content: "",

        open({ title, content }) {
            this.title = title;
            this.content = content;
            this.visible = true;
        },

        close() {
            this.visible = false;
            this.title = "";
            this.content = "";
        },
    }));
});

Alpine.start();
