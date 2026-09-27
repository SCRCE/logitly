#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "$0")/../.." && pwd)
status=0
while IFS= read -r path; do
  case "$path" in
    *.gguf|*.safetensors|*.xcframework/*|.cache/*|apple/.tools/*|apple/.derived-data/*)
      echo "Forbidden tracked artifact: $path" >&2
      status=1
      continue
      ;;
  esac
  if [[ -f "$root/$path" ]]; then
    if stat -f '%z' "$root/$path" >/dev/null 2>&1; then
      bytes=$(stat -f '%z' "$root/$path")
    else
      bytes=$(stat -c '%s' "$root/$path")
    fi
    if (( bytes > 50000000 )); then
      echo "Tracked file exceeds 50 MB: $path ($bytes bytes)" >&2
      status=1
    fi
  fi
done < <(git -C "$root" ls-files)
exit "$status"
