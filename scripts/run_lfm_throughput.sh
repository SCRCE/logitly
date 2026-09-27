#!/usr/bin/env bash
set -euo pipefail

container_name="logitly-lfm-throughput"

cleanup() {
  docker rm --force "${container_name}" >/dev/null 2>&1 || true
}

trap cleanup EXIT INT TERM
cleanup

docker compose run \
  --name "${container_name}" \
  --rm \
  logitly \
  benchmark lfm \
  "$@"
