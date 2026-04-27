#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

cleanup() {
  docker compose --env-file tests/.env-test down --remove-orphans
}
trap cleanup EXIT

echo "setup poetry env"
poetry config virtualenvs.in-project true --local
if ! poetry env info --path >/dev/null 2>&1; then
  if command -v python3.14 >/dev/null 2>&1; then
    poetry env use "$(command -v python3.14)"
  elif command -v python3.12 >/dev/null 2>&1; then
    poetry env use "$(command -v python3.12)"
  elif command -v python3.11 >/dev/null 2>&1; then
    poetry env use "$(command -v python3.11)"
  else
    poetry env use "$(command -v python3)"
  fi
fi

echo "install project dependencies"
poetry install --with dev --sync

wait_for_http() {
  local url="$1"
  local name="$2"
  local max_attempts="${3:-60}"
  local sleep_seconds="${4:-2}"

  echo "waiting for ${name} on ${url}"

  for ((i=1; i<=max_attempts; i++)); do
    if curl -fsS "$url" >/dev/null 2>&1; then
      echo "${name} is ready"
      return 0
    fi
    echo "${name} not ready yet (${i}/${max_attempts})"
    sleep "$sleep_seconds"
  done

  echo "ERROR: ${name} not ready after ${max_attempts} attempts"
  docker compose logs keycloak || true
  return 1
}

echo "reset compose stack"
docker compose --env-file tests/.env-test down -v --remove-orphans || true

echo "make compose"
docker compose --env-file tests/.env-test up -d --force-recreate

wait_for_http \
  "http://localhost:10765/realms/test/.well-known/openid-configuration" \
  "keycloak-ozn-test"

echo "check code"
poetry run black ozonenv/**/*.py
# poetry run flake8 ozonenv/**/*.py

echo "run test"
rm -rf tests/models models
time poetry run pytest --cov --cov-report=html -vv -x -s "$@"

echo "make project: Done."
