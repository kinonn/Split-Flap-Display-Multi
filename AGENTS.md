# AGENTS.md

## Project

ESP32 firmware + web UI for a modular 3D-printed split-flap display.
PlatformIO (Arduino framework) firmware in `src/`, web UI in `src/web/`
(Vite + Tailwind + Alpine.js) bundled into a gzipped littlefs filesystem.

## Commands

- `npm run build` — format, build web assets, upload firmware + filesystem to board
- `npm run pio:firmware` / `npm run pio:filesystem` — upload firmware / filesystem only
- `pio run -e esp32_c3` — compile only (default env; esp32_s3 also available)
- `npm test` — Vitest frontend tests; `npm run test:watch` for watch mode
- C++ host tests in `test/` (no Arduino needed):
  `g++ -std=c++11 -Wall -Wextra -I src test/<name>.cpp && ./a.out`
- `pio device monitor -e esp32_c3 -b 460800` (S3 uses 115200)

## Conventions

- Formatting: clang-format on all `src/*.{h,cpp,ino}`; prettier on web files + configs.
  Run `npm run format` (skips C++ if clang-format isn't installed).
- Settings are JSON-backed (JsonSettings), persisted in NVS; web assets live in littlefs.
- `src/web/` is the source of truth; `build/web/` output is gitignored.
- When adding, renaming, removing, or changing the meaning/type/constraints of any setting
  (or the export file envelope), bump `SETTINGS_SCHEMA_VERSION` in BOTH `src/JsonSettings.h`
  and `src/web/index.js` (they must stay in sync). The web UI uses it to warn about
  forward/backward compatibility when importing exported config files.

## Gotchas

- ESP32-C3 uses `no_ota.csv` partitions (no OTA slot). When switching partition layouts,
  erase the board first: `pio run -t erase -e esp32_c3` then upload firmware + filesystem.
- ESP32-S3 boards need manual BOOT/RESET to enter upload mode.
- CI (`format-check.yml`) fails the build on format drift — run `npm run format` before pushing.
- Motor coils power up HIGH at boot; keep `maxConcurrentMotors`/`bootDelayMs` changes deliberate
  (see README Power/Startup Settings).
