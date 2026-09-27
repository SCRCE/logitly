#!/usr/bin/env bash
set -euo pipefail

apple_root=$(cd "$(dirname "$0")/.." && pwd)
cd "$apple_root"

[[ "$(uname -s)" == "Darwin" ]] || { echo "This validation requires macOS and Xcode." >&2; exit 1; }
[[ -d LogitlyDemo.xcodeproj ]] || { echo "Run ./scripts/bootstrap.sh first." >&2; exit 1; }

swift test
simulator_id=$(xcrun simctl list devices available | awk -F '[()]' '/iPhone/ && /(Shutdown|Booted)/ {print $2; exit}')
[[ -n "$simulator_id" ]] || { echo "No available iPhone simulator is installed." >&2; exit 1; }
xcodebuild \
  -project LogitlyDemo.xcodeproj \
  -scheme LogitlyDemo \
  -destination "platform=iOS Simulator,id=$simulator_id" \
  CODE_SIGNING_ALLOWED=NO \
  build test
./scripts/check-no-large-files.sh
