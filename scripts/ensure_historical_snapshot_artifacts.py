#!/usr/bin/env python3
"""Materialize dated snapshot artifacts from latest aliases when safely possible."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")
DATE_FMT = "%Y-%m-%d"


def _extract_date_from_name(name: str) -> Optional[str]:
    match = DATE_RE.search(name)
    return match.group(1) if match else None


def _extract_date_from_json(path: Path) -> Optional[str]:
    if not path.exists() or path.suffix.lower() != ".json":
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    for key in ("snapshot_as_of_date", "as_of_date", "asOfDate", "window_end"):
        value = payload.get(key)
        if isinstance(value, str) and DATE_RE.fullmatch(value):
            return value
    return None


def _extract_date_from_text(path: Path) -> Optional[str]:
    if not path.exists() or path.suffix.lower() != ".txt":
        return None
    try:
        content = path.read_text(encoding="utf-8")
    except OSError:
        return None
    for key in ("as_of_date", "snapshot_as_of_date", "window_end"):
        match = re.search(rf"(?im)^\s*{re.escape(key)}\s*[:=]\s*(\d{{4}}-\d{{2}}-\d{{2}})\s*$", content)
        if match:
            return match.group(1)
    return None


def _latest_date_for_pattern(base_dir: Path, pattern: str) -> Optional[str]:
    dates: list[str] = []
    for path in base_dir.glob(pattern):
        if not path.is_file():
            continue
        extracted = _extract_date_from_name(path.name)
        if extracted:
            dates.append(extracted)
    if not dates:
        return None
    return sorted(dates)[-1]


def _max_date_from_csv(path: Path) -> Optional[str]:
    date_columns = ("date", "game_date", "as_of_date", "snapshot_as_of_date")
    found_dates: list[str] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                for column in date_columns:
                    value = row.get(column)
                    if isinstance(value, str):
                        match = DATE_RE.search(value.strip())
                        if match:
                            found_dates.append(match.group(1))
                            break
    except OSError:
        return None
    if not found_dates:
        return None
    return sorted(found_dates)[-1]


def _sorted_dates(base_dir: Path, pattern: str) -> list[str]:
    dates = {_extract_date_from_name(path.name) for path in base_dir.glob(pattern) if path.is_file()}
    return sorted(date for date in dates if date)


def _normalize_local_matched_csv(source: Path, target: Path) -> dict[str, object]:
    rows_before = 0
    normalized_rows: list[dict[str, str]] = []
    source_date_col: Optional[str] = None

    with source.open("r", encoding="utf-8", newline="") as in_handle:
        reader = csv.DictReader(in_handle)
        fieldnames = list(reader.fieldnames or [])
        if not fieldnames:
            raise RuntimeError(f"{source.name} has no CSV header")

        for candidate in ("date", "game_date", "as_of_date", "snapshot_as_of_date"):
            if candidate in fieldnames:
                source_date_col = candidate
                break
        if source_date_col is None:
            raise RuntimeError(f"{source.name} missing date source column (expected date or game_date)")

        if "date" not in fieldnames:
            fieldnames = [*fieldnames, "date"]

        for row in reader:
            rows_before += 1
            value = row.get(source_date_col)
            parsed: Optional[datetime] = None
            if isinstance(value, str) and value.strip():
                raw = value.strip()
                match = DATE_RE.search(raw)
                if match:
                    raw = match.group(1)
                try:
                    parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
                except ValueError:
                    parsed = None
            if parsed is None:
                continue
            row["date"] = parsed.strftime(DATE_FMT)
            normalized_rows.append(row)

    if not normalized_rows:
        raise RuntimeError("local_matched_games export has no valid date rows after normalization")

    with target.open("w", encoding="utf-8", newline="") as out_handle:
        writer = csv.DictWriter(out_handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(normalized_rows)

    return {
        "rows_before": rows_before,
        "rows_after": len(normalized_rows),
        "source_date_column": source_date_col,
        "dropped_rows": rows_before - len(normalized_rows),
    }


def ensure_historical_snapshot_artifacts(source_root: Path) -> dict[str, object]:
    root = source_root.expanduser().resolve()
    lightgbm = root / "output" / "LightGBM"
    kelly = lightgbm / "Kelly"
    if not lightgbm.exists():
        raise FileNotFoundError(f"Missing LightGBM directory: {lightgbm}")

    snapshot_date = _latest_date_for_pattern(kelly, "combined_nba_predictions_iso_*.csv")
    if snapshot_date is None:
        snapshot_date = _latest_date_for_pattern(lightgbm, "combined_nba_predictions_acc_*.csv")
    if snapshot_date is None:
        raise FileNotFoundError("No dated combined predictions files were found.")

    actions: list[str] = []
    warnings: list[str] = []

    local_dated = lightgbm / f"local_matched_games_{snapshot_date}.csv"
    local_latest = lightgbm / "local_matched_games_latest.csv"
    if not local_dated.exists() and local_latest.exists():
        local_max_date = _max_date_from_csv(local_latest)
        if local_max_date == snapshot_date:
            details = _normalize_local_matched_csv(local_latest, local_dated)
            shutil.copy2(local_dated, local_latest)
            actions.append(
                f"created {local_dated.name} from local_matched_games_latest.csv "
                f"(date_source={details['source_date_column']}, rows={details['rows_before']}->{details['rows_after']})"
            )
            if details["dropped_rows"]:
                warnings.append(
                    f"dropped {details['dropped_rows']} local_matched rows with invalid dates during normalization"
                )
            actions.append("synced local_matched_games_latest.csv to normalized dated artifact")
        else:
            warnings.append(
                "skipped local_matched backfill because local_matched_games_latest.csv "
                f"max date ({local_max_date}) did not match snapshot date ({snapshot_date})"
            )

    strategy_exact_json = lightgbm / f"strategy_params_{snapshot_date}.json"
    strategy_exact_txt = lightgbm / f"strategy_params_{snapshot_date}.txt"
    strategy_alias_json = lightgbm / "strategy_params.json"
    strategy_alias_txt = lightgbm / "strategy_params.txt"

    if not strategy_exact_json.exists() and not strategy_exact_txt.exists():
        strategy_source = strategy_alias_json if strategy_alias_json.exists() else strategy_alias_txt
        if strategy_source.exists():
            embedded_date = _extract_date_from_json(strategy_source) or _extract_date_from_text(strategy_source)
            target_date = embedded_date or snapshot_date
            if target_date <= snapshot_date:
                target = lightgbm / f"strategy_params_{target_date}{strategy_source.suffix.lower()}"
                if not target.exists():
                    shutil.copy2(strategy_source, target)
                    actions.append(f"created {target.name} from {strategy_source.name}")
            else:
                warnings.append(
                    f"skipped strategy params backfill because {strategy_source.name} has future date {target_date} "
                    f"for snapshot {snapshot_date}"
                )

    report = {
        "snapshot_date": snapshot_date,
        "actions": actions,
        "warnings": warnings,
        "latest_10_combined_dates": _sorted_dates(kelly, "combined_nba_predictions_iso_*.csv")[-10:],
        "latest_10_local_matched_dates": _sorted_dates(lightgbm, "local_matched_games_*.csv")[-10:],
        "latest_10_strategy_params_dates": sorted(
            {
                *(_sorted_dates(lightgbm, "strategy_params_*.json")),
                *(_sorted_dates(lightgbm, "strategy_params_*.txt")),
            }
        )[-10:],
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    args = parser.parse_args()

    report = ensure_historical_snapshot_artifacts(args.source_root)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
