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
        errors: {},
        timezones: {},

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
                .filter((r) => r.length > 0)
                .map((r) => r.split(",").map((v) => parseInt(v) || 0));
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
