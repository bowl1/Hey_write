#!/usr/bin/env bash
# Shared local/CI regression gate. Never use a production database here.
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
python_bin=${HEYWRITE_TEST_PYTHON:-"$root/chatbox-backend/.venv/bin/python"}
container=""
build_dir=""
cleanup() {
  if [ -n "$build_dir" ]; then rm -rf "$build_dir"; fi
  if [ -n "$container" ]; then docker rm -f "$container" >/dev/null; fi
}
trap cleanup EXIT

if [ ! -x "$python_bin" ] && ! command -v "$python_bin" >/dev/null 2>&1; then
  echo "Install backend dependencies first; see README Testing." >&2
  exit 1
fi
if [ ! -d "$root/chatbox-frontend/node_modules" ]; then
  echo "Run npm ci in chatbox-frontend first." >&2
  exit 1
fi

# An explicitly supplied URL must point to a disposable test database.
# Otherwise provision a fresh pgvector container, never the application's .env DB.
if [ -z "${RLS_TEST_DATABASE_URL:-}" ]; then
  docker info >/dev/null
  container=$(docker run --rm -d -e POSTGRES_PASSWORD=postgres \
    -p 127.0.0.1::5432 pgvector/pgvector:pg17)
  ready=false
  for ((attempt=0; attempt<60; attempt++)); do
    if docker exec "$container" pg_isready -U postgres >/dev/null 2>&1; then
      ready=true
      break
    fi
    sleep 1
  done
  if [ "$ready" != true ]; then
    echo "Disposable test database did not become ready." >&2
    exit 1
  fi
  port=$(docker port "$container" 5432/tcp)
  export RLS_TEST_DATABASE_URL="postgresql://postgres:postgres@127.0.0.1:${port##*:}/postgres?sslmode=disable"
fi

printf '\nRunning backend tests, including database integration tests...\n'
(
  cd "$root/chatbox-backend"
  "$python_bin" - <<'PY'
import sys
import unittest
suite = unittest.defaultTestLoader.discover('tests', pattern='test_*.py')
result = unittest.TextTestRunner(verbosity=2).run(suite)
if result.skipped:
    print('Full regression gate does not allow skipped tests.', file=sys.stderr)
sys.exit(0 if result.wasSuccessful() and result.testsRun > 0 and not result.skipped else 1)
PY
)
printf '\nRunning all frontend tests...\n'
(cd "$root/chatbox-frontend" && CI=true npm test -- --watchAll=false --runInBand)
printf '\nBuilding frontend production bundle...\n'
build_dir=$(mktemp -d "${TMPDIR:-/tmp}/heywrite-build.XXXXXX")
(cd "$root/chatbox-frontend" && CI=true BUILD_PATH="$build_dir" npm run build)
printf '\nAll regression checks passed.\n'
