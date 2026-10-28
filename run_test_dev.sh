#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

cleanup() {
  docker compose --env-file tests/.env-test down -v database redis_cache  || true
}
trap cleanup EXIT

wait_for_tcp() {
  local host="$1"
  local port="$2"
  local name="$3"
  local max_attempts="${4:-60}"
  local sleep_seconds="${5:-2}"

  echo "waiting for ${name} on ${host}:${port}"

  for ((i=1; i<=max_attempts; i++)); do
    if python3 - <<PY >/dev/null 2>&1
import socket
s = socket.socket()
s.settimeout(1.0)
try:
    s.connect(("${host}", ${port}))
    ok = True
except Exception:
    ok = False
finally:
    s.close()
raise SystemExit(0 if ok else 1)
PY
    then
      echo "${name} is ready"
      return 0
    fi
    echo "${name} not ready yet (${i}/${max_attempts})"
    sleep "${sleep_seconds}"
  done

  echo "ERROR: ${name} not ready after ${max_attempts} attempts"
  docker compose logs "${name}" || true
  return 1
}

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
    sleep "${sleep_seconds}"
  done

  echo "ERROR: ${name} not ready after ${max_attempts} attempts"
  docker compose logs keycloak-ozn-test || true
  return 1
}

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

echo "start compose stack"
docker compose --env-file tests/.env-test up -d

wait_for_tcp "localhost" "10002" "database"
wait_for_tcp "localhost" "10003" "redis_cache"
wait_for_http \
  "http://localhost:10765/realms/test/.well-known/openid-configuration" \
  "keycloak"

echo "check code"
poetry run black ozonenv tests

echo "run test"
rm -rf tests/models
time poetry run pytest --cov --cov-report=html -vv -x -s "$@"

echo "runtestdev: Done."