import { describe, it, expect, beforeEach, vi } from "vitest";

const createPageData = (type = "Settings") => {
    return {
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

        selectedOffsetGroup: 0,

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

        getCharOffset(modIdx, charIdx) {
            const matrix = this.charOffsetMatrix;
            if (
                modIdx >= matrix.length ||
                charIdx >= (matrix[modIdx]?.length || 0)
            )
                return 0;
            return matrix[modIdx][charIdx];
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
        allZeroCharOffsets(modIdx) {
            const matrix = this.charOffsetMatrix;
            if (modIdx >= matrix.length) return true;
            return matrix[modIdx].every((v) => v === 0);
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
    };
};

describe("Page Component - Module Offsets", () => {
    let page;

    beforeEach(() => {
        page = createPageData();
    });

    it("should return empty array when moduleOffsets is undefined", () => {
        expect(page.offsetArray).toEqual([]);
    });

    it("should parse comma-separated offset string", () => {
        page.settings.moduleOffsets = "10,20,30";
        expect(page.offsetArray).toEqual(["10", "20", "30"]);
    });

    it("should set offset at specific index", () => {
        page.settings.moduleOffsets = "0,0,0";
        page.setOffset(1, "15");
        expect(page.settings.moduleOffsets).toBe("0,15,0");
    });

    it("should handle setting offset on empty array", () => {
        page.setOffset(2, "25");
        expect(page.offsetArray[2]).toBe("25");
    });
});

describe("Page Component - Character Offsets", () => {
    let page;

    beforeEach(() => {
        page = createPageData();
        page.settings.charset = 48;
    });

    it("should return empty array when charOffsets is undefined", () => {
        expect(page.charOffsetMatrix).toEqual([]);
    });

    it("should parse semicolon-separated rows with comma values", () => {
        page.settings.charOffsets = "1,2,3;4,5,6";
        expect(page.charOffsetMatrix).toEqual([
            [1, 2, 3],
            [4, 5, 6],
        ]);
    });

    it("should preserve empty rows positionally", () => {
        page.settings.charOffsets = "1,2;;4,5";
        expect(page.charOffsetMatrix).toEqual([[1, 2], [], [4, 5]]);
    });

    it("should set character offset at specific module and character", () => {
        page.settings.charOffsets = "0,0,0;0,0,0";
        page.setCharOffset(1, 2, 10);
        expect(page.charOffsetMatrix).toEqual([
            [0, 0, 0],
            [0, 0, 10],
        ]);
    });

    it("should expand matrix when setting offset beyond current size", () => {
        page.settings.charOffsets = "1,2";
        page.setCharOffset(1, 3, 5);
        const matrix = page.charOffsetMatrix;
        expect(matrix.length).toBe(2);
        expect(matrix[1][3]).toBe(5);
    });

    it("should handle non-numeric values as 0", () => {
        page.setCharOffset(0, 0, "abc");
        expect(page.charOffsetMatrix[0][0]).toBe(0);
    });

    it("should reset character offsets for a module", () => {
        page.settings.charOffsets = "1,2,3;4,5,6;7,8,9";
        page.settings.charset = 3;
        page.resetCharOffsets(1);
        expect(page.charOffsetMatrix).toEqual([
            [1, 2, 3],
            [0, 0, 0],
            [7, 8, 9],
        ]);
    });

    it("should reset using charset size (48)", () => {
        page.settings.charset = 48;
        page.settings.charOffsets = "1,2,3";
        page.resetCharOffsets(0);
        expect(page.charOffsetMatrix[0].length).toBe(48);
        expect(page.charOffsetMatrix[0].every((v) => v === 0)).toBe(true);
    });

    it("should copy character offsets from one module to another", () => {
        page.settings.charOffsets = "1,2,3;4,5,6;7,8,9";
        page.copyCharOffsets(0, 2);
        expect(page.charOffsetMatrix[2]).toEqual([1, 2, 3]);
    });

    it("should not copy from invalid source module", () => {
        page.settings.charOffsets = "1,2,3";
        page.copyCharOffsets(5, 0);
        expect(page.charOffsetMatrix).toEqual([[1, 2, 3]]);
    });

    it("should expand target module when copying", () => {
        page.settings.charOffsets = "1,2,3;4,5,6";
        page.copyCharOffsets(0, 2);
        expect(page.charOffsetMatrix.length).toBe(3);
        expect(page.charOffsetMatrix[2]).toEqual([1, 2, 3]);
    });
});

describe("Page Component - Charset Characters", () => {
    let page;

    beforeEach(() => {
        page = createPageData();
    });

    it("should return extended charset (48 chars) by default", () => {
        page.settings.charset = 48;
        expect(page.charsetChars.length).toBe(48);
        expect(page.charsetChars).toContain("'");
        expect(page.charsetChars).toContain("%");
    });

    it("should return standard charset (37 chars) when set", () => {
        page.settings.charset = 37;
        expect(page.charsetChars.length).toBe(37);
        expect(page.charsetChars).not.toContain("'");
    });

    it("should start with space character", () => {
        expect(page.charsetChars[0]).toBe(" ");
    });

    it("should contain all letters A-Z", () => {
        for (let i = 65; i <= 90; i++) {
            expect(page.charsetChars).toContain(String.fromCharCode(i));
        }
    });

    it("should contain all digits 0-9", () => {
        for (let i = 0; i <= 9; i++) {
            expect(page.charsetChars).toContain(String(i));
        }
    });
});

describe("Page Component - Module Addresses", () => {
    let page;

    beforeEach(() => {
        page = createPageData();
    });

    it("should return empty array when moduleAddresses is undefined", () => {
        expect(page.addressArray).toEqual([]);
    });

    it("should parse comma-separated address string", () => {
        page.settings.moduleAddresses = "32, 33, 34";
        expect(page.addressArray).toEqual(["32", "33", "34"]);
    });

    it("should set address at specific index", () => {
        page.settings.moduleAddresses = "32,33,34";
        page.setAddress(1, "35");
        expect(page.settings.moduleAddresses).toBe("32,35,34");
    });
});

describe("Page Component - Master Groups", () => {
    let page;

    beforeEach(() => {
        page = createPageData();
    });

    it("should return group module array with defaults", () => {
        expect(page.groupModuleArray).toEqual(["8", "8", "8", "8", "8", "8"]);
    });

    it("should normalize master group count to 1-6 range", () => {
        page.settings.masterGroupCount = 0;
        page.normalizeMasterGroups();
        expect(page.settings.masterGroupCount).toBe(1);

        page.settings.masterGroupCount = 10;
        page.normalizeMasterGroups();
        expect(page.settings.masterGroupCount).toBe(6);
    });

    it("should set group module count and update moduleCount for index 0", () => {
        page.setGroupModuleCount(0, 4);
        expect(page.settings.moduleCount).toBe(4);
        expect(page.groupModuleArray[0]).toBe("4");
    });

    it("should set group module count without changing moduleCount for other indices", () => {
        page.settings.moduleCount = 8;
        page.setGroupModuleCount(2, 6);
        expect(page.settings.moduleCount).toBe(8);
        expect(page.groupModuleArray[2]).toBe("6");
    });

    it("should return group MAC array with defaults", () => {
        expect(page.groupMacArray).toEqual(["", "", "", "", "", ""]);
    });

    it("should set group MAC at specific index", () => {
        page.setGroupMac(1, "AA:BB:CC:DD:EE:FF");
        expect(page.groupMacArray[1]).toBe("AA:BB:CC:DD:EE:FF");
    });
});

describe("Page Component - Load Settings", () => {
    let page;

    beforeEach(() => {
        page = createPageData();
        global.fetch = vi.fn();
    });

    it("should load settings from /settings endpoint", async () => {
        const mockSettings = {
            settings: {
                name: "Test Display",
                moduleCount: 4,
                moduleOffsets: "1,2,3,4",
                charset: 48,
            },
            localMac: "AA:BB:CC:DD:EE:FF",
        };

        global.fetch.mockResolvedValueOnce({
            json: () => Promise.resolve(mockSettings),
        });

        page.loadSettings();
        await new Promise((resolve) => setTimeout(resolve, 0));

        expect(global.fetch).toHaveBeenCalledWith("/settings");
        expect(page.settings.name).toBe("Test Display");
        expect(page.settings.moduleCount).toBe(4);
        expect(page.localMac).toBe("AA:BB:CC:DD:EE:FF");
        expect(page.loading.settings).toBe(false);
    });

    it("should handle load settings failure", async () => {
        global.fetch.mockRejectedValueOnce(new Error("Network error"));

        page.loadSettings();
        await new Promise((resolve) => setTimeout(resolve, 0));

        expect(page.dialog.show).toBe(true);
        expect(page.dialog.type).toBe("error");
        expect(page.loading.settings).toBe(false);
    });
});

describe("Page Component - Load Timezones", () => {
    let page;

    beforeEach(() => {
        page = createPageData();
        global.fetch = vi.fn();
    });

    it("should load timezones from /timezones.json", async () => {
        const mockTimezones = { UTC: "UTC", "America/New_York": "EST" };

        global.fetch.mockResolvedValueOnce({
            json: () => Promise.resolve(mockTimezones),
        });

        page.loadTimezones();
        await new Promise((resolve) => setTimeout(resolve, 0));

        expect(global.fetch).toHaveBeenCalledWith("/timezones.json");
        expect(page.timezones).toEqual(mockTimezones);
        expect(page.loading.timezones).toBe(false);
    });

    it("should handle load timezones failure", async () => {
        global.fetch.mockRejectedValueOnce(new Error("Network error"));

        page.loadTimezones();
        await new Promise((resolve) => setTimeout(resolve, 0));

        expect(page.dialog.show).toBe(true);
        expect(page.dialog.type).toBe("error");
        expect(page.loading.timezones).toBe(false);
    });
});

describe("Page Component - Save Settings", () => {
    let page;

    beforeEach(() => {
        page = createPageData();
        global.fetch = vi.fn();
    });

    it("should save settings to /settings endpoint", async () => {
        const mockResponse = {
            message: "Settings saved successfully!",
            type: "success",
        };

        global.fetch.mockResolvedValueOnce({
            json: () => Promise.resolve(mockResponse),
        });

        page.settings.name = "New Name";
        page.save();
        await new Promise((resolve) => setTimeout(resolve, 0));

        expect(global.fetch).toHaveBeenCalledWith("/settings", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: expect.stringContaining("New Name"),
        });
        expect(page.saving).toBe(false);
        expect(page.dialog.type).toBe("success");
    });

    it("should handle save failure", async () => {
        global.fetch.mockRejectedValueOnce(new Error("Network error"));

        page.save();
        await new Promise((resolve) => setTimeout(resolve, 0));

        expect(page.dialog.show).toBe(true);
        expect(page.dialog.type).toBe("error");
        expect(page.saving).toBe(false);
    });

    it("should handle validation errors from server", async () => {
        const mockResponse = {
            message: "Validation failed",
            type: "error",
            errors: { key: "moduleCount", message: "Invalid value" },
        };

        global.fetch.mockResolvedValueOnce({
            json: () => Promise.resolve(mockResponse),
        });

        page.save();
        await new Promise((resolve) => setTimeout(resolve, 0));

        expect(page.errors.key).toBe("moduleCount");
        expect(page.errors.message).toBe("Invalid value");
    });
});

describe("Page Component - Reset Settings", () => {
    let page;

    beforeEach(() => {
        page = createPageData();
        global.fetch = vi.fn();
        global.confirm = vi.fn();
    });

    it("should reset settings when confirmed", async () => {
        global.confirm.mockReturnValue(true);

        const mockResponse = {
            message: "Settings reset successfully!",
            type: "success",
        };

        global.fetch.mockResolvedValueOnce({
            json: () => Promise.resolve(mockResponse),
        });

        page.reset();
        await new Promise((resolve) => setTimeout(resolve, 0));

        expect(global.confirm).toHaveBeenCalled();
        expect(global.fetch).toHaveBeenCalledWith("/settings/reset", {
            method: "POST",
        });
    });

    it("should not reset settings when cancelled", () => {
        global.confirm.mockReturnValue(false);

        page.reset();

        expect(global.confirm).toHaveBeenCalled();
        expect(global.fetch).not.toHaveBeenCalled();
    });
});

describe("Page Component - Word Management", () => {
    let page;

    beforeEach(() => {
        page = createPageData();
    });

    it("should add word to multiWords array", () => {
        page.multiWord = "test";
        page.addWord();
        expect(page.multiWords).toContain("test");
        expect(page.multiWord).toBe("");
    });

    it("should not add empty word", () => {
        page.multiWord = "   ";
        page.addWord();
        expect(page.multiWords).toEqual([]);
    });

    it("should remove word at specific index", () => {
        page.multiWords = ["one", "two", "three"];
        page.removeWord(1);
        expect(page.multiWords).toEqual(["one", "three"]);
    });
});

describe("Page Component - Update Display", () => {
    let page;

    beforeEach(() => {
        page = createPageData();
        global.fetch = vi.fn();
    });

    it("should reject delay less than 1 in mode 0", () => {
        page.settings.mode = 0;
        page.delay = 0;
        page.updateDisplay();
        expect(page.dialog.show).toBe(true);
        expect(page.dialog.type).toBe("error");
    });

    it("should reject empty single word in mode 0", () => {
        page.settings.mode = 0;
        page.singleMode = true;
        page.singleWord = "";
        page.updateDisplay();
        expect(page.dialog.show).toBe(true);
        expect(page.dialog.type).toBe("error");
    });

    it("should reject empty word list in mode 0 multi mode", () => {
        page.settings.mode = 0;
        page.singleMode = false;
        page.multiWords = [];
        page.updateDisplay();
        expect(page.dialog.show).toBe(true);
        expect(page.dialog.type).toBe("error");
    });

    it("should post mode update for non-zero modes", () => {
        page.settings.mode = 1;
        page.updateDisplay();
        expect(global.fetch).toHaveBeenCalledWith("/settings", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ mode: 1 }),
        });
    });
});

describe("Page Component - Processing State", () => {
    let page;

    beforeEach(() => {
        page = createPageData();
    });

    it("should be processing when saving", () => {
        page.saving = true;
        expect(page.processing).toBe(true);
    });

    it("should be processing when loading settings", () => {
        page.loading.settings = true;
        expect(page.processing).toBe(true);
    });

    it("should be processing when loading timezones", () => {
        page.loading.timezones = true;
        expect(page.processing).toBe(true);
    });

    it("should not be processing when idle", () => {
        page.saving = false;
        page.loading.settings = false;
        page.loading.timezones = false;
        expect(page.processing).toBe(false);
    });
});

describe("Page Component - Header", () => {
    let page;

    beforeEach(() => {
        page = createPageData();
    });

    it("should return settings name when set", () => {
        page.settings.name = "My Display";
        expect(page.header).toBe("My Display");
    });

    it("should return default 'Split Flap' when name is empty", () => {
        page.settings.name = "";
        expect(page.header).toBe("Split Flap");
    });

    it("should return default 'Split Flap' when name is undefined", () => {
        delete page.settings.name;
        expect(page.header).toBe("Split Flap");
    });
});

describe("Page Component - Group Reordering", () => {
    let page;

    beforeEach(() => {
        page = createPageData();
        page.settings.masterGroupCount = 3;
        page.settings.masterGroupMacs = ",AA:BB:CC:DD:EE:01,AA:BB:CC:DD:EE:02";
        page.settings.masterGroupModuleCounts = "8,6,4";
    });

    it("should not move group up if at position 1", () => {
        page.moveGroupUp(1);
        expect(page.groupMacArray[1]).toBe("AA:BB:CC:DD:EE:01");
    });

    it("should swap groups when moving up", () => {
        page.moveGroupUp(2);
        expect(page.groupMacArray[1]).toBe("AA:BB:CC:DD:EE:02");
        expect(page.groupMacArray[2]).toBe("AA:BB:CC:DD:EE:01");
        expect(page.groupModuleArray[1]).toBe("4");
        expect(page.groupModuleArray[2]).toBe("6");
    });

    it("should not move group down if at last position", () => {
        page.moveGroupDown(2);
        expect(page.groupMacArray[2]).toBe("AA:BB:CC:DD:EE:02");
    });

    it("should swap groups when moving down", () => {
        page.moveGroupDown(1);
        expect(page.groupMacArray[1]).toBe("AA:BB:CC:DD:EE:02");
        expect(page.groupMacArray[2]).toBe("AA:BB:CC:DD:EE:01");
        expect(page.groupModuleArray[1]).toBe("4");
        expect(page.groupModuleArray[2]).toBe("6");
    });
});

describe("Page Component - Discovery", () => {
    let page;

    beforeEach(() => {
        page = createPageData();
    });

    it("should load discovered peers from settings", async () => {
        global.fetch = vi.fn();
        const mockSettings = {
            settings: {
                name: "Test Display",
                masterGroupCount: 2,
            },
            localMac: "AA:BB:CC:DD:EE:FF",
            discoveredPeers: [
                { mac: "11:22:33:44:55:66", moduleCount: 6, assigned: false },
            ],
        };

        global.fetch.mockResolvedValueOnce({
            json: () => Promise.resolve(mockSettings),
        });

        page.loadSettings();
        await new Promise((resolve) => setTimeout(resolve, 0));

        expect(page.discoveredPeers).toHaveLength(1);
        expect(page.discoveredPeers[0].mac).toBe("11:22:33:44:55:66");
        expect(page.discoveredPeers[0].moduleCount).toBe(6);
    });

    it("should stage assignment locally without server call", () => {
        page.discoveredPeers = [{ mac: "11:22:33:44:55:66", moduleCount: 6 }];
        page.settings.masterGroupCount = 3;

        page.assignPeer(1, "11:22:33:44:55:66");

        expect(page.groupMacArray[1]).toBe("11:22:33:44:55:66");
        expect(page.groupModuleArray[1]).toBe("6");
    });

    it("should stage removal locally without server call", () => {
        page.settings.masterGroupCount = 3;
        page.setGroupMac(1, "11:22:33:44:55:66");
        page.setGroupModuleCount(1, 6);

        page.removePeer(1);

        expect(page.groupMacArray[1]).toBe("");
        expect(page.groupModuleArray[1]).toBe("8");
    });

    it("should detect assigned peers in real-time", () => {
        page.settings.masterGroupCount = 3;
        page.setGroupMac(1, "11:22:33:44:55:66");

        expect(page.isPeerAssigned("11:22:33:44:55:66")).toBe(true);
        expect(page.isPeerAssigned("AA:BB:CC:DD:EE:FF")).toBe(false);
    });

    it("should detect assigned peers case-insensitively", () => {
        page.settings.masterGroupCount = 3;
        page.setGroupMac(1, "aa:bb:cc:dd:ee:ff");

        expect(page.isPeerAssigned("AA:BB:CC:DD:EE:FF")).toBe(true);
    });

    it("should return group label for assigned peer", () => {
        page.settings.masterGroupCount = 3;
        page.setGroupMac(2, "11:22:33:44:55:66");

        expect(page.assignedGroupLabel("11:22:33:44:55:66")).toBe(
            "Assigned to Group 3",
        );
    });

    it("should return empty label for unassigned peer", () => {
        page.settings.masterGroupCount = 3;

        expect(page.assignedGroupLabel("11:22:33:44:55:66")).toBe("");
    });

    it("should not call fetch when assigning peer", () => {
        global.fetch = vi.fn();
        page.discoveredPeers = [{ mac: "11:22:33:44:55:66", moduleCount: 6 }];
        page.settings.masterGroupCount = 3;

        page.assignPeer(1, "11:22:33:44:55:66");

        expect(global.fetch).not.toHaveBeenCalled();
    });

    it("should not call fetch when removing peer", () => {
        global.fetch = vi.fn();
        page.settings.masterGroupCount = 3;
        page.setGroupMac(1, "11:22:33:44:55:66");

        page.removePeer(1);

        expect(global.fetch).not.toHaveBeenCalled();
    });
});

describe("Page Component - Remote Offset Management", () => {
    let page;

    beforeEach(() => {
        page = createPageData("Settings");
        page.settings.masterGroupCount = 2;
        page.settings.masterGroupModuleCounts = "8,6,8,8,8,8";
        page.settings.rModOffs = "0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0";
        page.settings.rDispOffs = "0,0,0,0,0";
        page.settings.rChrOff0 = Array(8)
            .fill(Array(48).fill(0).join(","))
            .join(";");
    });

    describe("remoteOffsetMatrix", () => {
        it("should parse remote module offsets matrix", () => {
            page.settings.rModOffs = "1,2,3;4,5,6";
            expect(page.remoteOffsetMatrix).toEqual([
                [1, 2, 3],
                [4, 5, 6],
            ]);
        });

        it("should return empty array when rModOffs is missing", () => {
            delete page.settings.rModOffs;
            expect(page.remoteOffsetMatrix).toEqual([]);
        });
    });

    describe("setRemoteOffset", () => {
        it("should set remote module offset at specific group and module", () => {
            page.setRemoteOffset(0, 2, 10);
            expect(page.remoteOffsetMatrix[0][2]).toBe(10);
        });

        it("should expand matrix when setting offset beyond current size", () => {
            page.settings.rModOffs = "0,0";
            page.setRemoteOffset(1, 3, 5);
            const matrix = page.remoteOffsetMatrix;
            expect(matrix.length).toBe(2);
            expect(matrix[1][3]).toBe(5);
        });
    });

    describe("remoteDisplayOffsetArray", () => {
        it("should parse remote display offsets", () => {
            page.settings.rDispOffs = "10,20,30";
            expect(page.remoteDisplayOffsetArray).toEqual([10, 20, 30]);
        });

        it("should return empty array when rDispOffs is missing", () => {
            delete page.settings.rDispOffs;
            expect(page.remoteDisplayOffsetArray).toEqual([]);
        });
    });

    describe("setRemoteDisplayOffset", () => {
        it("should set remote display offset at specific group", () => {
            page.setRemoteDisplayOffset(1, 15);
            expect(page.remoteDisplayOffsetArray[1]).toBe(15);
        });

        it("should expand array when setting offset beyond current size", () => {
            page.settings.rDispOffs = "0";
            page.setRemoteDisplayOffset(2, 25);
            expect(page.remoteDisplayOffsetArray.length).toBe(3);
            expect(page.remoteDisplayOffsetArray[2]).toBe(25);
        });
    });

    describe("getRemoteCharOffsetMatrix", () => {
        it("should parse remote character offsets for specific group", () => {
            page.settings.rChrOff0 = "1,2;3,4";
            expect(page.getRemoteCharOffsetMatrix(0)).toEqual([
                [1, 2],
                [3, 4],
            ]);
        });

        it("should return empty array when key is missing", () => {
            expect(page.getRemoteCharOffsetMatrix(1)).toEqual([]);
        });
    });

    describe("setRemoteCharOffset", () => {
        it("should set remote character offset with clamping", () => {
            page.setRemoteCharOffset(0, 0, 5, 40);
            expect(page.getRemoteCharOffsetMatrix(0)[0][5]).toBe(32);

            page.setRemoteCharOffset(0, 0, 6, -50);
            expect(page.getRemoteCharOffsetMatrix(0)[0][6]).toBe(-32);
        });

        it("should expand matrix when setting offset beyond current size", () => {
            page.settings.rChrOff0 = "0,0";
            page.setRemoteCharOffset(0, 2, 10, 5);
            const matrix = page.getRemoteCharOffsetMatrix(0);
            expect(matrix.length).toBe(3);
            expect(matrix[2][10]).toBe(5);
        });
    });

    describe("resetRemoteCharOffsets", () => {
        it("should reset remote character offsets to zeros", () => {
            page.setRemoteCharOffset(0, 0, 0, 10);
            page.resetRemoteCharOffsets(0, 0);
            expect(page.getRemoteCharOffsetMatrix(0)[0]).toEqual(
                Array(48).fill(0),
            );
        });
    });

    describe("copyRemoteCharOffsets", () => {
        it("should copy remote character offsets from one module to another", () => {
            page.setRemoteCharOffset(0, 0, 0, 10);
            page.copyRemoteCharOffsets(0, 0, 1);
            expect(page.getRemoteCharOffsetMatrix(0)[1][0]).toBe(10);
        });
    });

    describe("selectedOffsetGroup dispatch", () => {
        it("should dispatch to local offsets when selectedOffsetGroup is 0", () => {
            page.selectedOffsetGroup = 0;
            page.settings.moduleOffsets = "0,0,0";
            page.setCurrentOffset(1, 15);
            expect(page.offsetArray[1]).toBe("15");
        });

        it("should dispatch to remote offsets when selectedOffsetGroup > 0", () => {
            page.selectedOffsetGroup = 1;
            page.setCurrentOffset(2, 20);
            expect(page.remoteOffsetMatrix[0][2]).toBe(20);
        });

        it("should dispatch display offset to local when selectedOffsetGroup is 0", () => {
            page.selectedOffsetGroup = 0;
            page.setCurrentDisplayOffset(10);
            expect(page.settings.displayOffset).toBe(10);
        });

        it("should dispatch display offset to remote when selectedOffsetGroup > 0", () => {
            page.selectedOffsetGroup = 1;
            page.setCurrentDisplayOffset(25);
            expect(page.remoteDisplayOffsetArray[0]).toBe(25);
        });
    });

    describe("allZeroCurrentCharOffsets", () => {
        it("should return true for local all-zero module", () => {
            page.selectedOffsetGroup = 0;
            page.settings.charOffsets = "0,0,0;0,0,0";
            expect(page.allZeroCurrentCharOffsets(0)).toBe(true);
        });

        it("should return false for local non-zero module", () => {
            page.selectedOffsetGroup = 0;
            page.settings.charOffsets = "1,0,0;0,0,0";
            expect(page.allZeroCurrentCharOffsets(0)).toBe(false);
        });

        it("should return true for remote all-zero module", () => {
            page.selectedOffsetGroup = 1;
            expect(page.allZeroCurrentCharOffsets(0)).toBe(true);
        });

        it("should return false for remote non-zero module", () => {
            page.selectedOffsetGroup = 1;
            page.setRemoteCharOffset(0, 0, 0, 5);
            expect(page.allZeroCurrentCharOffsets(0)).toBe(false);
        });
    });
});
