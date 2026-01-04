#!/usr/bin/env bash
set -euo pipefail

URL="${1:-}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-4173}"
HISTORICAL_ROUTE="${HISTORICAL_ROUTE:-/}"

if [[ -z "$URL" ]]; then
  if [[ "$HISTORICAL_ROUTE" != /* ]]; then
    HISTORICAL_ROUTE="/$HISTORICAL_ROUTE"
  fi
  URL="http://$HOST:$PORT$HISTORICAL_ROUTE"
fi

python3 -m webbrowser "$URL" >/dev/null 2>&1 || true
