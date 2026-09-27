#!/usr/bin/env bash
set -euo pipefail

apple_root=$(cd "$(dirname "$0")/.." && pwd)
cd "$apple_root"
swift test
if rg -n 'llama_sampler_|completion_loop|generate\(' Sources; then
  echo "Generation or sampling API found in the Apple runtime." >&2
  exit 1
fi
../apple/scripts/check-no-large-files.sh
