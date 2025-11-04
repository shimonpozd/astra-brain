#!/usr/bin/env sh
set -eu

CONFIG_ROOT="${ASTRA_CONFIG_ROOT:-/app}"

if [ -d "${CONFIG_ROOT}/config" ]; then
  :
else
  echo "FATAL: could not find the 'config' package."
  echo "Expected directory: ${CONFIG_ROOT}/config"
  echo "Mount your deployment config and/or set ASTRA_CONFIG_ROOT."
  exit 1
fi

exec uvicorn brain_service.main:app \
  --host 0.0.0.0 \
  --port "${BRAIN_PORT:-7030}" \
  --workers "${UVICORN_WORKERS:-1}"
