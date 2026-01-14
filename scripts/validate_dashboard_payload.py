#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
from typing import Any, Dict


REQUIRED_TOP_LEVEL = [
    "as_of_date",
    "window",
    "active_filters_effective",
    "summary",
    "tables",
    "sources",
]


def _require_fields(data: Dict[str, Any], keys: list[str], label: str) -> None:
    missing = [key for key in keys if key not in data]
    if missing:
        raise ValueError(f"Missing {label} fields: {', '.join(missing)}")


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        default="public/data/dashboard_payload.json",
        help="Path to dashboard_payload.json (default: public/data/dashboard_payload.json).",
    )
    args = parser.parse_args()
    path = Path(args.path)

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, dict):
        raise ValueError("dashboard_payload.json must contain a JSON object.")

    _require_fields(payload, REQUIRED_TOP_LEVEL, "top-level")

    window = payload.get("window", {})
    if not isinstance(window, dict):
        raise ValueError("window must be an object.")
    _assert(window.get("size") == 200, "window.size must be 200.")
    window_start = window.get("start")
    window_end = window.get("end")
    _assert(window_start not in (None, "", "—"), "window.start must be defined.")
    _assert(window_end not in (None, "", "—"), "window.end must be defined.")
    _assert(window_end == payload.get("as_of_date"), "window.end must match as_of_date.")

    summary = payload.get("summary", {})
    tables = payload.get("tables", {})
    if not isinstance(summary, dict) or not isinstance(tables, dict):
        raise ValueError("summary and tables must be objects.")

    strategy_counts = summary.get("strategy_counts", {})
    if not isinstance(strategy_counts, dict):
        raise ValueError("summary.strategy_counts must be an object.")

    local_rows = tables.get("local_matched_games_rows", [])
    settled_bets = tables.get("settled_bets_rows", [])
    settled_summary = tables.get("settled_bets_summary", {})

    _assert(
        isinstance(local_rows, list),
        "tables.local_matched_games_rows must be an array.",
    )
    _assert(
        isinstance(settled_bets, list),
        "tables.settled_bets_rows must be an array.",
    )
    _assert(
        isinstance(settled_summary, dict),
        "tables.settled_bets_summary must be an object.",
    )

    _assert(
        len(local_rows) == strategy_counts.get("settled_bets_count"),
        "local_matched_games_rows length must match strategy_counts.settled_bets_count.",
    )
    _assert(
        len(settled_bets) == settled_summary.get("count"),
        "settled_bets_rows length must match settled_bets_summary.count.",
    )

    if len(local_rows) >= 5:
        sharpe = summary.get("strategy_summary", {}).get("sharpeStyle")
        max_dd = summary.get("kpis", {}).get("max_drawdown_eur")
        _assert(sharpe is not None, "Sharpe ratio missing for sufficient sample size.")
        _assert(max_dd is not None, "Max drawdown missing for sufficient sample size.")

    sources = payload.get("sources", {})
    if not isinstance(sources, dict):
        raise ValueError("sources must be an object.")
    for key in ("metrics_snapshot", "local_matched_games", "bet_log_flat", "combined_file"):
        value = sources.get(key)
        _assert(isinstance(value, str) and value and "missing" not in value.lower(),
                f"sources.{key} must be present.")

    print(f"dashboard_payload.json OK: {path}")


if __name__ == "__main__":
    main()
