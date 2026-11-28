#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
PYTHON_BIN="${PYTHON_BIN:-python}"

run_case() {
  local expected_decision="$1"
  local bot_threshold="$2"
  local fake_threshold="$3"
  local status=0

  BOT_THRESHOLD="$bot_threshold" FAKE_NEWS_THRESHOLD="$fake_threshold" docker compose up -d --wait

  set +e
  BASE_URL="$BASE_URL" \
    PYTHON_BIN="$PYTHON_BIN" \
    EXPECTED_DECISION="$expected_decision" \
    BOT_THRESHOLD="$bot_threshold" \
    FAKE_NEWS_THRESHOLD="$fake_threshold" \
    "$PYTHON_BIN" tests/e2e_validation.py
  status=$?
  set -e

  docker compose logs --tail=60
  docker compose down

  return "$status"
}

run_case MODERATION 0.9 0.9
run_case BAN 0.05 0.05

echo "End-to-end validation passed"
