import Alpine from "alpinejs";
window.Alpine = Alpine;

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
            timeFormat: "HH:mm",
            masterGroupCount: 1,
            masterGroupModuleCounts: "8,8,8,8,8,8",
            masterGroupMacs: ",,,,,",
        },
        localMac: "",
        discoveredPeers: [],
        errors: {},
        timezones: {},
        selectedOffsetGroup: 0,

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

        get bus2AddressArray() {
            return (
                this.settings.wire1Addresses
                    ?.split(",")
                    .map((s) => s.trim()) || []
            );
        },
        // Merged view: modules 0-7 from moduleAddresses (Wire),
        // modules 8-15 from wire1Addresses (Wire1) on dual-I2C builds.
        get addressArray() {
            const a = (
                this.settings.moduleAddresses
                    ?.split(",")
                    .map((s) => s.trim()) || []
            );
            const b = this.bus2AddressArray;
            const count = Number(this.settings.moduleCount) || a.length || 8;
            const merged = [];
            for (let i = 0; i < count; i++) {
                merged.push(i < 8 ? (a[i] ?? "") : (b[i - 8] ?? ""));
            }
            return merged;
        },
        setAddress(index, value) {
            if (index < 8) {
                const arr = this.addressArray.slice(0, 8);
                arr[index] = value;
                this.settings.moduleAddresses = arr.join(",");
            } else {
                const arr = this.bus2AddressArray;
                while (arr.length <= index - 8) arr.push("");
                arr[index - 8] = value;
                this.settings.wire1Addresses = arr.join(",");
            }
        },

        get offsetArray() {
            const a = (
                this.settings.moduleOffsets?.split(",").map((s) => s.trim()) ||
                []
            );
            const b = this.bus2OffsetArray;
            const count = Number(this.settings.moduleCount) || a.length || 8;
            const merged = [];
            for (let i = 0; i < count; i++) {
                merged.push(i < 8 ? (a[i] ?? "") : (b[i - 8] ?? ""));
            }
            return merged;
        },
        get bus2OffsetArray() {
            return (
                this.settings.wire1Offsets?.split(",").map((s) => s.trim()) ||
                []
            );
        },
        setOffset(index, value) {
            if (index < 8) {
                const arr = this.offsetArray.slice(0, 8);
                arr[index] = value;
                this.settings.moduleOffsets = arr.join(",");
            } else {
                const arr = this.bus2OffsetArray;
                while (arr.length <= index - 8) arr.push("");
                arr[index - 8] = value;
                this.settings.wire1Offsets = arr.join(",");
            }
        },

        get charOffsetMatrix() {
            if (!this.settings.charOffsets) return [];
            return this.settings.charOffsets
                .split(";")
                .map((r) => r.length === 0 ? [] : r.split(",").map((v) => parseInt(v) || 0));
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
                .map((r) => r.length === 0 ? [] : r.split(",").map((v) => parseInt(v) || 0));
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
                .map((r) => r.length === 0 ? [] : r.split(",").map((v) => parseInt(v) || 0));
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
