#!/usr/bin/env bash
set -euo pipefail

apple_root=$(cd "$(dirname "$0")/.." && pwd)
project_root=$(cd "$apple_root/.." && pwd)
model_name="LFM2.5-1.2B-Instruct-QAD-Q4_0.gguf"
model_dir="$apple_root/LogitlyDemo/Resources/Models"
model_path="$model_dir/$model_name"
model_tmp="$model_path.partial"
model_bytes=695755488
model_sha="bb741ebb106d543e9de114b843a3d3d73d51c74b5801e69da2abde821a0cb3e1"
model_revision="8ed288026e23958ad9dfa92d53ed773a8eee7125"
model_url="https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct-GGUF/resolve/$model_revision/$model_name"
xcodegen_url="https://github.com/yonaskolb/XcodeGen/releases/download/2.46.0/xcodegen.zip"
xcodegen_sha="4d9e34b62172d645eed6457cac13fc222569974098ef4ee9c3368bedf0196806"
tools_dir="$apple_root/.tools"
xcodegen_zip="$tools_dir/xcodegen-2.46.0.zip"
xcodegen_bin="$tools_dir/xcodegen/bin/xcodegen"
operational_limit=90000000000
hard_limit=100000000000

hash_file() {
  if command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | awk '{print $1}'
  else
    sha256sum "$1" | awk '{print $1}'
  fi
}

file_bytes() {
  if stat -f '%z' "$1" >/dev/null 2>&1; then
    stat -f '%z' "$1"
  else
    stat -c '%s' "$1"
  fi
}

current_bytes=$(du -sk "$project_root" | awk '{print $1 * 1024}')
projected_bytes=$current_bytes
if [[ ! -f "$model_path" ]]; then
  projected_bytes=$((projected_bytes + model_bytes))
fi
if (( current_bytes >= hard_limit || projected_bytes >= hard_limit )); then
  echo "Refusing bootstrap: project allocation would reach the 100,000,000,000-byte hard limit." >&2
  exit 1
fi
if (( projected_bytes >= operational_limit )); then
  echo "Refusing bootstrap: projected project allocation exceeds the 90 GB operational limit." >&2
  exit 1
fi

mkdir -p "$model_dir" "$tools_dir"
if [[ ! -f "$model_path" ]] || [[ "$(file_bytes "$model_path")" != "$model_bytes" ]] || [[ "$(hash_file "$model_path")" != "$model_sha" ]]; then
  echo "Downloading pinned LFM QAD Q4 model (695.8 MB)..."
  curl --fail --location --continue-at - --output "$model_tmp" "$model_url"
  [[ "$(file_bytes "$model_tmp")" == "$model_bytes" ]] || { echo "Model size mismatch" >&2; exit 1; }
  [[ "$(hash_file "$model_tmp")" == "$model_sha" ]] || { echo "Model SHA-256 mismatch" >&2; exit 1; }
  mv "$model_tmp" "$model_path"
fi

if [[ ! -x "$xcodegen_bin" ]]; then
  echo "Installing pinned XcodeGen 2.46.0 locally..."
  curl --fail --location --output "$xcodegen_zip" "$xcodegen_url"
  [[ "$(hash_file "$xcodegen_zip")" == "$xcodegen_sha" ]] || { echo "XcodeGen SHA-256 mismatch" >&2; exit 1; }
  unzip -q -o "$xcodegen_zip" -d "$tools_dir"
fi

"$xcodegen_bin" generate --spec "$apple_root/project.yml" --project "$apple_root"
echo "Ready: open $apple_root/LogitlyDemo.xcodeproj and select a physical iPhone."
