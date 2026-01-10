#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


DATE_FMT = "%Y-%m-%d"
REQUIRED_KEYS = [
    "last_run",
    "as_of_date",
    "source_root_used",
    "expected_lightgbm_dir",
    "metrics_snapshot_source",
    "strategy_params_source",
    "local_matched_games_source",
    "local_matched_games_rows",
    "local_matched_games_profit_sum_table",
    "matched_count_snapshot",
    "matched_count_table",
    "matched_count_used",
    "records",
]
NUMERIC_KEYS = [
    "local_matched_games_rows",
    "local_matched_games_profit_sum_table",
    "matched_count_snapshot",
    "matched_count_table",
    "matched_count_used",
]


def _is_number_or_null(value: Any) -> bool:
    return value is None or isinstance(value, (int, float))


def _check_required_keys(data: Dict[str, Any]) -> None:
    missing = [key for key in REQUIRED_KEYS if key not in data]
    if missing:
        raise ValueError(f"Missing required keys: {', '.join(missing)}")


def _check_numeric_fields(data: Dict[str, Any]) -> None:
    invalid = [key for key in NUMERIC_KEYS if not _is_number_or_null(data.get(key))]
    if invalid:
        raise ValueError(f"Non-numeric values for keys: {', '.join(invalid)}")

    records = data.get("records", {})
    if not isinstance(records, dict):
        raise ValueError("records must be an object.")
    invalid_records = [
        key for key, value in records.items() if not _is_number_or_null(value)
    ]
    if invalid_records:
        raise ValueError(
            f"Non-numeric values in records: {', '.join(invalid_records)}"
        )


def _check_as_of_date(data: Dict[str, Any]) -> None:
    as_of_date = data.get("as_of_date")
    if not isinstance(as_of_date, str):
        raise ValueError("as_of_date must be a string.")
    try:
        datetime.strptime(as_of_date, DATE_FMT)
    except ValueError as exc:
        raise ValueError("as_of_date must be in YYYY-MM-DD format.") from exc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        default="public/data/last_run.json",
        help="Path to last_run.json (default: public/data/last_run.json).",
    )
    args = parser.parse_args()
    path = Path(args.path)

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("last_run.json must contain a JSON object.")

    _check_required_keys(data)
    _check_as_of_date(data)
    _check_numeric_fields(data)
    print(f"last_run.json OK: {path}")


if __name__ == "__main__":
    main()
