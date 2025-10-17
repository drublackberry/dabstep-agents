#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required but not installed or not on PATH" >&2
  exit 1
fi

mkdir -p "${PROJECT_ROOT}/data" "${PROJECT_ROOT}/runs"

if [ ! -f "${PROJECT_ROOT}/.env" ]; then
  if [ -f "${PROJECT_ROOT}/.env.example" ]; then
    echo ".env not found; copying .env.example as a starting point." >&2
    cp "${PROJECT_ROOT}/.env.example" "${PROJECT_ROOT}/.env"
    echo "Update .env with your API keys before invoking the agents." >&2
  else
    echo "Warning: .env not found and .env.example missing; creating an empty placeholder." >&2
    echo "Populate this file with your API keys before invoking the agents." >&2
    touch "${PROJECT_ROOT}/.env"
  fi
fi

cd "$PROJECT_ROOT"

exec docker compose run --rm agent "$@"
