#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export HOME=/home/$(whoami)
cd /c/Users/kinon/projects-local/Split-Flap-Display-Multi

echo "clang-format path: $(which clang-format || echo MISSING)"
clang-format --version

echo "=== before: formatted files exist? ==="
ls -la src/CalibApi.h | wc -c

echo "=== formatting src/ ==="
for f in $(find src -iname '*.h' -o -iname '*.cpp' -o -iname '*.ino'); do
  clang-format -i "$f"
done
echo "clang-format done"
