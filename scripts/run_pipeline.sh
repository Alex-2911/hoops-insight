#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/run_pipeline.sh [--open-dashboard] [--dry-run]

Environment variables:
  HOOPS_DIR        Path to hoops-insight repo (default: pwd)
  NBA_DIR          Path to Basketball_prediction repo (default: "$HOOPS_DIR/../Basketball_prediction")
  SOURCE_ROOT      Basketball_prediction output root (default: "$NBA_DIR/2026")
  HOST             Preview host (default: 127.0.0.1)
  PORT             Preview port (default: 4173)
  HISTORICAL_ROUTE Route to open in preview (default: /)
  NBA_PIPELINE_CMD Optional command to run NBA pipeline (default: auto-detect)

Examples:
  SOURCE_ROOT="/path/to/Basketball_prediction/2026" PORT=4173 \
    ./scripts/run_pipeline.sh --open-dashboard

  ./scripts/run_pipeline.sh --dry-run
USAGE
}

DRY_RUN=false
OPEN_DASHBOARD=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    --open-dashboard)
      OPEN_DASHBOARD=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

HOOPS_DIR="${HOOPS_DIR:-$(pwd)}"
NBA_DIR="${NBA_DIR:-$HOOPS_DIR/../Basketball_prediction}"
SOURCE_ROOT="${SOURCE_ROOT:-$NBA_DIR/2026}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-4173}"
HISTORICAL_ROUTE="${HISTORICAL_ROUTE:-/}"
NBA_PIPELINE_CMD="${NBA_PIPELINE_CMD:-}"

if [[ "$HISTORICAL_ROUTE" != /* ]]; then
  HISTORICAL_ROUTE="/$HISTORICAL_ROUTE"
fi

PREVIEW_URL="http://$HOST:$PORT$HISTORICAL_ROUTE"

print_cmd() {
  printf '+ '
  printf '%q ' "$@"
  printf '\n'
}

run_cmd() {
  if "$DRY_RUN"; then
    print_cmd "$@"
    return 0
  fi
  "$@"
}

port_in_use() {
  lsof -ti "tcp:$PORT" >/dev/null 2>&1 || lsof -ti ":$PORT" >/dev/null 2>&1
}

kill_port() {
  local pids
  pids=$(lsof -ti "tcp:$PORT" 2>/dev/null || lsof -ti ":$PORT" 2>/dev/null || true)
  if [[ -n "$pids" ]]; then
    run_cmd kill $pids
  fi
}

require_lsof() {
  if ! command -v lsof >/dev/null 2>&1; then
    echo "lsof is required to manage preview ports. Please install lsof and retry." >&2
    exit 1
  fi
}

ensure_port_free() {
  if "$DRY_RUN"; then
    echo "DRY RUN: would ensure port $PORT is free"
    return 0
  fi
  require_lsof
  kill_port
  if port_in_use; then
    echo "Port $PORT is still in use after cleanup. Please free it and retry." >&2
    exit 1
  fi
}

resolve_nba_pipeline_cmd() {
  if [[ -n "$NBA_PIPELINE_CMD" ]]; then
    return 0
  fi
  if [[ -x "$NBA_DIR/run_hoops_pipeline_18.sh" ]]; then
    NBA_PIPELINE_CMD="$NBA_DIR/run_hoops_pipeline_18.sh"
  elif [[ -x "$NBA_DIR/scripts/run_hoops_pipeline_18.sh" ]]; then
    NBA_PIPELINE_CMD="$NBA_DIR/scripts/run_hoops_pipeline_18.sh"
  elif [[ -x "$NBA_DIR/run_pipeline.sh" ]]; then
    NBA_PIPELINE_CMD="$NBA_DIR/run_pipeline.sh"
  fi
}

run_nba_pipeline() {
  resolve_nba_pipeline_cmd
  if [[ -z "$NBA_PIPELINE_CMD" ]]; then
    if "$DRY_RUN"; then
      echo "DRY RUN: set NBA_PIPELINE_CMD to run the Basketball_prediction pipeline"
      return 0
    fi
    echo "No NBA pipeline command found. Set NBA_PIPELINE_CMD or place a script in $NBA_DIR." >&2
    exit 1
  fi

  run_cmd bash -lc "cd \"$NBA_DIR\" && $NBA_PIPELINE_CMD"
}

check_predictions() {
  if "$DRY_RUN"; then
    echo "DRY RUN: skipping predictions output check"
    return 0
  fi
  shopt -s nullglob
  local files=("$SOURCE_ROOT/output/LightGBM"/combined_nba_predictions_acc_*.csv)
  shopt -u nullglob
  if [[ ${#files[@]} -eq 0 ]]; then
    echo "No predictions found at $SOURCE_ROOT/output/LightGBM/combined_nba_predictions_acc_*.csv" >&2
    echo "Pipeline produced no predictions or sync failed." >&2
    exit 1
  fi
}

PREVIEW_PID=""
cleanup() {
  if [[ -n "$PREVIEW_PID" ]]; then
    kill "$PREVIEW_PID" 2>/dev/null || true
    wait "$PREVIEW_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

run_nba_pipeline
check_predictions

run_cmd bash -lc "cd \"$HOOPS_DIR\" && python3 scripts/generate_dashboard_data.py --source-root \"$SOURCE_ROOT\""

ensure_port_free

if "$DRY_RUN"; then
  print_cmd npm run preview -- --host "$HOST" --port "$PORT" --strictPort
  if "$OPEN_DASHBOARD"; then
    print_cmd scripts/open_dashboard_preview.sh "$PREVIEW_URL"
  else
    print_cmd python3 -m webbrowser "$PREVIEW_URL"
  fi
  exit 0
fi

npm run preview -- --host "$HOST" --port "$PORT" --strictPort &
PREVIEW_PID=$!

sleep 1

if "$OPEN_DASHBOARD" && [[ -x "$HOOPS_DIR/scripts/open_dashboard_preview.sh" ]]; then
  "$HOOPS_DIR/scripts/open_dashboard_preview.sh" "$PREVIEW_URL"
else
  python3 -m webbrowser "$PREVIEW_URL" >/dev/null 2>&1 || true
fi

wait "$PREVIEW_PID"
