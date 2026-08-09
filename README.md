# Split Flap Display Firmware

![format badge](https://github.com/jhoff/Split-Flap-Display/actions/workflows/format-check.yml/badge.svg?branch=main)
[![](https://dcbadge.limes.pink/api/server/https://discord.gg/RCvks4XXXH?style=flat)](https://discord.gg/RCvks4XXXH)

Looking for support? Find help on discord ☝️

Firmware for the modular Split Flap Display created by [Morgan Manly](https://github.com/ManlyMorgan/Split-Flap-Display)

- [Instructables](https://www.instructables.com/Split-Flap-Display-3D-Printed-Modular-Compact-Encl/) ( Original )
- [Print Files](https://makerworld.com/en/models/1116618#profileId-1114192) ( Original, 37 characters version )
- [Print Files](https://makerworld.com/en/models/1296793-split-flap-display-extended-charset-48-flaps#profileId-1328346) ( Extended, 48 characters version )
- [Original firmware](https://github.com/ManlyMorgan/Split-Flap-Display)

## Features

- Fully 3D Printed Modular Split Flap Display with 37 or 48 Characters Per Module
- Small Size, 8 Modules are 320mm, 3 Modules are 130mm Wide. 80mm tall (original version) or 94mm (extended version) tall
- Fully configurable and controllable via Web Interface
    - Switch Between Operation Modes, modes include custom input, date mode, and time mode
    - Configure WiFi, Timezone, and hardware settings
- MQTT Support
- Multi-display master mode using ESP-NOW to coordinate up to 6 display groups, with up to 8 modules per group
    - Automatic device discovery — remote controllers announce themselves and the master detects them without manual MAC entry
    - Module count auto-population from discovered devices
    - Group reordering with up/down controls (Group 1 always pinned as local)
- Per-module and per-character offset tuning for precise flap alignment
- Scrolling text for messages longer than the display width, with configurable delay between chunks and configurable repeat count

## Supported boards

| Environment             | Processor     | Tested Boards                                                            |
| ----------------------- | ------------- | ------------------------------------------------------------------------ |
| `esp32_c3`              | ESP32-C3FN4   | Teyleten Robot ESP32-C3-SuperMini<br>Waveshare ESP32-C3-Zero             |
| `esp32_s3`              | ESP32-S3FH4R2 | Waveshare ESP32-S3-Zero<sup>\*</sup><br>ESP32-S3 Super Mini<sup>\*</sup> |
| `esp32_wroom` (default) | ESP32 (WROOM) | Dual-I2C builds: up to 16 modules per group on two I2C buses             |

<sub>\* Requires manually resetting the board into firmware upload mode by holding BOOT, pressing & releasing RESET, then releasing BOOT prior to upload. After uploading is successful, either press & release RESET or power cycle the board to put it in normal operation mode.</sub>

### Dual-I2C builds (`esp32_wroom`)

Each PCF8575 module has 3 address pins (A0-A2), so a single I2C bus supports at most 8 unique addresses (`0x20`-`0x27`). To drive more than 8 modules from one controller, the `esp32_wroom` environment compiles with `ENABLE_DUAL_I2C`:

- Modules 0-7 live on `Wire` (default SDA=21 / SCL=22).
- Modules 8-15 live on `Wire1` (default SDA2=18 / SCL2=19, configurable in the web UI).
- A group may be configured with up to 16 modules (`Number of Modules` in Hardware Settings).
- Multi-group chunking is width-aware: a 16-module local group still chunks messages left-to-right across the full group width, exactly like an 8-module group.

Notes:

- The ESP32-C3 has a single I2C controller and cannot run dual-bus builds; ESP32-S3 and ESP32-WROOM have two.
- On ESP32-S3, GPIO 33-37 are input-only — pick output-capable pins for SDA2/SCL2 if you enable dual bus there.
- When flashing a dual-I2C build to a board that previously ran a single-bus build, erase the board first (see the C3 partition section) so the new settings keys are registered.

### ESP32-C3 partition layout

The default `esp32_c3` environment uses `no_ota.csv`. This gives the 4MB ESP32-C3 boards a larger single app partition for the current firmware and web feature set. Without this layout, the firmware can outgrow the default OTA app slot and the board may repeatedly reset before `setup()` runs.

Because this layout changes the flash partition table, erase the board before uploading firmware built with it:

```sh
pio run -t erase -e esp32_c3
pio run -t upload -e esp32_c3
pio run -t uploadfs -e esp32_c3
```

After flashing, monitor the C3 environment at `460800` baud:

```sh
pio device monitor -e esp32_c3 -b 460800
```

The erase step clears saved settings, Wi-Fi credentials, NVS data, and the filesystem. Upload both firmware and filesystem afterward.

## Setup Instructions

This guide takes you from a completely fresh computer and a brand-new ESP32
board all the way to a configured, running display — step by step.

### 0. One-time tooling install

- [PlatformIO Core CLI](https://platformio.org/install/cli) — the build/upload
  toolchain. Alternatively install the
  [PlatformIO extension for VS Code](https://marketplace.visualstudio.com/items?itemName=platformio.platformio-ide),
  which bundles the CLI and adds a project toolbar.
- [Node.js](https://nodejs.org/) with [nvm](https://github.com/nvm-sh/nvm) —
  used to build the web interface. The project pins Node `lts/hydrogen`
  (see `.nvmrc`); running `nvm install` in the project root selects it.
- [ClangFormat](https://clang.llvm.org/docs/ClangFormat.html) — _optional_,
  only needed when editing C/C++ source and using `npm run format`. The build
  skips it gracefully if it is not installed.

> **Windows only:** most ESP32 dev boards use a CP210x or CH340 USB-to-serial
> chip. If the board is not detected after plugging it in, install the matching
> driver — [CP210x](https://www.silabs.com/developer-tools/usb-to-uart-bridge-vcp-drivers)
> or [CH340](https://www.wch-ic.com/downloads/CH341SER_ZIP.html).

### Step 1 — Get the code

```sh
git clone https://github.com/jhoff/Split-Flap-Display.git
cd Split-Flap-Display
nvm install   # optional: selects the pinned Node version from .nvmrc
npm install   # installs the frontend build tooling (vite, tailwind, ...)
```

### Step 2 — Connect the board and pick an environment

Connect the ESP32 to your computer with a USB **data** cable (charge-only
cables will not work). Confirm the board is detected:

```sh
pio device list
```

Note the serial port (e.g. `COM4` on Windows, `/dev/ttyUSB0` on Linux/macOS).

Pick the environment that matches your board — this controls build flags,
flash size and serial baud rate (see the
[Supported boards](#supported-boards) table). The default environment is
`esp32_wroom`; to use another board as the default, change `default_envs` at
the top of `platformio.ini`.

| Your board                | Environment   | Add to `pio` commands  |
| ------------------------- | ------------- | ---------------------- |
| ESP32-WROOM **(default)** | `esp32_wroom` | _(none — the default)_ |
| ESP32-C3                  | `esp32_c3`    | `-e esp32_c3`          |
| ESP32-S3                  | `esp32_s3`    | `-e esp32_s3`          |

### Step 3 — Build and flash

The one-command path formats the code, builds the web assets, compiles the
firmware, uploads the firmware, then uploads the LittleFS filesystem:

```sh
npm run build
```

If you prefer to run the steps separately (useful when iterating on code, or
when using a non-default environment), the underlying commands are:

```sh
npm run assets                          # compile + minify web UI (src/web -> build/web)
pio run -t upload -e esp32_wroom        # compile + upload firmware
pio run -t uploadfs -e esp32_wroom      # pack + upload the web filesystem
```

For non-default boards, replace `esp32_wroom` with your environment (for
example `-e esp32_c3`).

### Step 4 — Erase a fresh / first-time board

A brand-new ESP32 flashes and boots fine without erasing — the NVS settings
namespace is created automatically on first boot. However, erasing is:

- **Recommended** on a brand-new board to guarantee there is no stale data from
  another project, and **required** when re-flashing a board that previously
  ran different firmware or a different partition layout.
- **Required** for the `esp32_c3` environment — it uses the custom
  `no_ota.csv` partition table (see
  [ESP32-C3 partition layout](#esp32-c3-partition-layout)).

Erase wipes saved settings, Wi-Fi credentials, NVS data and the filesystem, so
re-upload the firmware **and** the filesystem afterward:

```sh
pio run -t erase -e esp32_c3
pio run -t upload -e esp32_c3
pio run -t uploadfs -e esp32_c3
```

### Step 5 — Watch it boot (optional but recommended)

```sh
pio device monitor -e esp32_wroom       # baud rate comes from the environment
```

The baud rate is set per environment (`115200` for `esp32_wroom`/`esp32_s3`,
`460800` for `esp32_c3`), so `-b` is usually not needed. A successful boot ends
with something like:

```text
[boot] setup complete in access point mode
```

### Step 6 — First boot: connect to the display

1. The display starts in **access point mode**. On your phone or laptop, join
   the Wi-Fi network named **`Split Flap Display`** (no password).
2. Open <http://192.168.4.1> (or <http://splitflap.local> via mDNS).
3. Go to `Settings`, enter your Wi-Fi network name and password, then click
   `Save Settings`. The display reboots and joins your network.
4. Reconnect to your normal Wi-Fi and open the display's new address (check
   your router's DHCP list, or use <http://splitflap.local>).

> **I2C pins:** if your hardware uses custom SDA/SCL wiring instead of the
> defaults (bus 1: SDA=21 / SCL=22, bus 2: SDA2=18 / SCL2=19), set the pins in
> the web UI under `Hardware Settings`. The firmware rejects reserved/strapping
> GPIOs (`0, 2, 5, 6-11, 12, 15`) and skips I2C rather than crash — but only
> safe, wired-up pins will actually drive modules.

### Updating an existing device

If the board already runs this firmware and you only changed code or settings,
you normally do **not** need to erase — saved settings, Wi-Fi credentials and
offsets are preserved in flash:

```sh
npm run build
```

### Troubleshooting

- **`Failed to connect to COMx` / no port found** — the board is not detected.
  Install the USB driver (see the note above), try another USB cable/port, and
  re-run `pio device list`.
- **ESP32-S3 does not upload** — put the board into download mode: hold
  **BOOT**, tap **RESET**, release **BOOT**, then upload. Power-cycle the board
  after the upload finishes.
- **Board boot-loops with `rst:0x8 (TG1WDT_SYS_RESET)`** — this is almost
  always caused by I2C pins that are reserved/strapping GPIOs (e.g. 8/9).
  Erase the board so it picks up the safe defaults (21/22 and 18/19), then
  reconfigure pins in the web UI if needed.
- **`esp32_c3` app does not fit / resets before `setup()`** — the C3 build
  needs the `no_ota.csv` partition layout; run the erase/upload sequence in
  Step 4 so the new partition table is applied.

1. Enjoy!

## Multi-Display Master Mode

The firmware can coordinate multiple split-flap display groups from one master controller. This is useful when a single hardware group cannot exceed 8 modules, but the full message needs to span more modules.

Limits:

- Up to 8 modules per group (up to 16 per group on dual-I2C `esp32_wroom` builds)
- Up to 6 groups total
- Up to 48 modules total when all 6 groups have 8 modules (96 on dual-I2C builds)
- Group 1 is always the local group connected to the master controller
- Groups 2-6 are remote groups controlled over ESP-NOW

All controllers run the same firmware. The master is simply the controller where `Number of Groups` is set above `1`.

### How messages are split

When custom text is submitted from the master web page, the text is split from left to right using each group's configured module count.

For example, with these group sizes:

| Group | Modules | Displayed segment |
| ----- | ------- | ----------------- |
| 1     | 8       | Characters 1-8    |
| 2     | 6       | Characters 9-14   |
| 3     | 8       | Characters 15-22  |

Group 1 displays its segment locally on the master controller. The master sends each remaining segment to the configured remote group MAC address over ESP-NOW. Short segments are padded with spaces. Characters beyond the total configured module count are not sent.

### Setup

1. Flash the firmware and filesystem to every group controller.
1. Configure Wi-Fi on every controller. Using the same Wi-Fi network is recommended so the controllers share a radio channel and each web page remains reachable.
1. Power on all remote controllers. Each remote controller periodically broadcasts an ESP-NOW announcement containing its MAC address and module count.
1. On the master controller, open `Settings`.
1. In `Hardware Settings`, set `Number of Modules` to the number of modules physically connected to the master group.
1. In `Multi-Display Master`, set `Number of Groups`.
1. Wait a few seconds for the `Discovered Devices` section to populate with remote controllers that have announced themselves. Each discovered device shows its MAC address and module count.
1. For each remote group slot:

    - Click `Assign to Group N` on a discovered device to auto-fill its MAC address and module count.
    - Alternatively, enter the MAC address manually if a device is not discovered (e.g. it is on a different Wi-Fi channel or not yet powered on).
    - Module count is auto-populated from the announcement but can be overridden before saving.

1. Reorder groups using the up/down arrow buttons if needed. Group 1 (Local) is always pinned at the top and cannot be moved.
1. Click `Save Settings` to persist all changes.

Changes are staged in the UI and only saved when you click `Save Settings`. The discovered devices list updates in real-time to show which devices are already assigned.

Remote groups automatically switch into ESP-NOW remote display mode when they receive a message from the master. They do not need their own text entry once registered with the master.

### Discovery Details

- Remote controllers broadcast announcements every 5 seconds via ESP-NOW to the broadcast address (`ff:ff:ff:ff:ff:ff`).
- The master maintains a discovery table of up to 6 peers, expiring entries that have not been seen for 30 seconds.
- The master filters out its own MAC address to avoid self-discovery.
- Discovery works over ESP-NOW regardless of Wi-Fi connectivity — it only requires that controllers share the same radio channel.
- If a remote controller is not discovered (e.g. different Wi-Fi channel), its MAC address can still be entered manually. The module count must then be set manually as well.

## Tuning

The settings page provides two levels of offset adjustment for precise flap alignment:

### Module Offsets

Each module has a coarse offset that adjusts the home position (magnet detection point). This shifts all characters on that module by the same amount. Use this when an entire module's display is consistently off by a few steps. Changes take effect immediately after saving settings.

### Character Offsets

Each character on each module can be individually tuned. This is useful when specific characters are misaligned while others are correct. In the settings page:

1. Expand the module you want to tune
2. Use the ▲/▼ arrows or enter values directly to adjust each character's position by individual motor steps
3. Use **Reset** to zero all offsets for a module
4. Use **Copy from** to duplicate another module's offsets

Changes take effect immediately after saving settings—no reboot required. Offsets are stored per-module and persist across reboots.

### Power / Startup Settings

Larger displays (many modules on one 5V rail) can trip their power supply
at boot when every motor coil is energized at once. Two settings help:

- **Boot Delay (ms)** — extra settle time after power-on before the
  network bring-up starts. Motor coils are powered down immediately at
  boot (before this delay), so the delay costs almost no current. In
  multi-display setups, set the slave's `bootDelayMs` a few seconds
  higher than the master's so the two boards never home at the same time.
- **Max Concurrent Motors** — caps how many motors are energized
  simultaneously during the boot homing sweep (and offset-reload homing).
  Lower = gentler on the supply, slower homing. Default 2. Normal message
  writes are unaffected (motors only draw current while actually moving).

Requires a reboot after changing either setting.

## Contributing

### Setup

1. Create a GitHub account if necessary and login.
1. [Fork](https://github.com/jhoff/Split-Flap-Display/fork) this repository.
1. Clone your forked repository to your local machine.

- `git clone https://github.com/your-username/Split-Flap-Display.git`

1. Install the dependencies listed in [Setup Instructions](#setup-instructions)
1. Skip step 1 (you already have a clone), complete the rest of the setup, and use `npm run build` to test and upload your changes.

### Create your feature

1. Start a new branch with a descriptive name.
1. Compile and upload your changes

- `npm run pio:firmware` or `pio run -t upload -e <environment>` to compile firmware and upload
- `npm run pio:filesystem` or `pio run -t uploadfs -e <environment>` to compile the filesystem from `src/web` and upload
- You can do both together using `npm run pio` or `pio run -t upload -t uploadfs -e <environment>`

1. When ready, commit and push your changes to your forked repository.
1. Open a pull request to this repository.

### Running Tests

The project includes unit tests for the frontend JavaScript logic. Tests use [Vitest](https://vitest.dev/) with jsdom environment.

Run all tests:

```sh
npm test
```

Run tests in watch mode (re-runs on file changes):

```sh
npm run test:watch
```

Tests cover the Alpine.js component logic including settings management, character offset calculations, and API interactions with mocked fetch calls.
