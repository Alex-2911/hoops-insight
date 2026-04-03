#!/usr/bin/env python3
"""Shared source/snapshot selection for local and GitHub dashboard runs."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional

DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")


@dataclass
class SnapshotSelection:
    snapshot_as_of_date: str
    run_date: str
    combined_source_file: str
    local_matched_source_file: str
    bet_log_source_file: str
    metrics_source_file: str
    strategy_params_source_file: str
    params_source_type: str
    fallback_used: bool
    fallback_reason: str

    combined_path: Path
    local_matched_path: Path
    bet_log_path: Optional[Path]
    metrics_path: Path
    strategy_params_path: Path


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


def _latest_dated(pattern: str, base_dir: Path) -> Optional[Path]:
    candidates: list[tuple[str, Path]] = []
    for path in base_dir.glob(pattern):
        if not path.is_file():
            continue
        extracted = _extract_date_from_name(path.name)
        if extracted:
            candidates.append((extracted, path))
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: item[0])[-1][1]


def _dated_for_snapshot(base_dir: Path, prefix: str, snapshot_date: str, suffix: str) -> Optional[Path]:
    exact = base_dir / f"{prefix}{snapshot_date}{suffix}"
    if exact.exists():
        return exact

    fallback_candidates: list[tuple[str, Path]] = []
    for path in base_dir.glob(f"{prefix}*{suffix}"):
        if not path.is_file():
            continue
        extracted = _extract_date_from_name(path.name)
        if extracted and extracted <= snapshot_date:
            fallback_candidates.append((extracted, path))

    if not fallback_candidates:
        return None
    return sorted(fallback_candidates, key=lambda item: item[0])[-1][1]


def _first_existing(paths: Iterable[Optional[Path]]) -> Optional[Path]:
    return next((path for path in paths if path is not None and path.exists()), None)


def _validate_date_match(path: Path, expected: str, label: str) -> None:
    date_from_source = _extract_date_from_name(path.name) or _extract_date_from_json(path)
    if date_from_source and date_from_source != expected:
        raise RuntimeError(
            f"{label} date mismatch: expected {expected}, got {date_from_source} from {path.name}"
        )


def resolve_snapshot_selection(source_root: Path) -> SnapshotSelection:
    root = source_root.expanduser().resolve()
    lightgbm = root / "output" / "LightGBM"
    kelly = lightgbm / "Kelly"
    if not lightgbm.exists():
        raise FileNotFoundError(f"Missing LightGBM directory: {lightgbm}")

    combined = _latest_dated("combined_nba_predictions_iso_*.csv", kelly)
    fallback_reasons: list[str] = []
    if combined is None:
        combined = _latest_dated("combined_nba_predictions_acc_*.csv", lightgbm)
        if combined is not None:
            fallback_reasons.append("combined_iso_missing_used_acc")
    if combined is None:
        raise FileNotFoundError("No dated combined predictions file found in source root.")

    snapshot_date = _extract_date_from_name(combined.name)
    if snapshot_date is None:
        raise RuntimeError(f"Unable to extract snapshot date from {combined.name}")

    local_matched = _dated_for_snapshot(lightgbm, "local_matched_games_", snapshot_date, ".csv")
    if local_matched is None:
        raise FileNotFoundError(
            "No local_matched_games_YYYY-MM-DD.csv available for "
            f"snapshot {snapshot_date} (or an earlier snapshot date)."
        )
    local_matched_date = _extract_date_from_name(local_matched.name)
    if local_matched_date and local_matched_date != snapshot_date:
        fallback_reasons.append(f"used_local_matched_from_{local_matched_date}")

    strategy_path = _first_existing(
        [
            _dated_for_snapshot(lightgbm, "strategy_params_", snapshot_date, ".json"),
            _dated_for_snapshot(lightgbm, "strategy_params_", snapshot_date, ".txt"),
            lightgbm / "strategy_params.json",
            lightgbm / "strategy_params.txt",
        ]
    )
    if strategy_path is None:
        raise FileNotFoundError(f"No strategy params file found for snapshot {snapshot_date}")
    strategy_date = _extract_date_from_name(strategy_path.name) or _extract_date_from_json(strategy_path)
    if strategy_date and strategy_date > snapshot_date:
        raise RuntimeError(
            f"strategy_params date mismatch: expected <= {snapshot_date}, got {strategy_date} from {strategy_path.name}"
        )
    params_source_type = "dated" if _extract_date_from_name(strategy_path.name) else "undated"
    if params_source_type == "undated":
        fallback_reasons.append("used_undated_strategy_params")
    elif strategy_date and strategy_date != snapshot_date:
        fallback_reasons.append(f"used_strategy_params_from_{strategy_date}")

    metrics_path = _first_existing(
        [
            _dated_for_snapshot(lightgbm, "metrics_snapshot_", snapshot_date, ".json"),
            lightgbm / "metrics_snapshot.json",
        ]
    )
    if metrics_path is None:
        raise FileNotFoundError(f"No metrics snapshot file found for snapshot {snapshot_date}")
    metrics_date = _extract_date_from_name(metrics_path.name) or _extract_date_from_json(metrics_path)
    if metrics_date and metrics_date > snapshot_date:
        raise RuntimeError(
            f"metrics_snapshot date mismatch: expected <= {snapshot_date}, got {metrics_date} from {metrics_path.name}"
        )
    if _extract_date_from_name(metrics_path.name) is None:
        fallback_reasons.append("used_undated_metrics_snapshot")
    elif metrics_date and metrics_date != snapshot_date:
        fallback_reasons.append(f"used_metrics_snapshot_from_{metrics_date}")

    bet_log_path = _first_existing(
        [
            lightgbm / "bet_log_flat_live.csv",
            lightgbm / f"bet_log_flat_live_{snapshot_date}.csv",
        ]
    )
    if bet_log_path is None:
        fallback_reasons.append("bet_log_flat_missing")

    return SnapshotSelection(
        snapshot_as_of_date=snapshot_date,
        run_date=snapshot_date,
        combined_source_file=combined.name,
        local_matched_source_file=local_matched.name,
        bet_log_source_file=bet_log_path.name if bet_log_path else "missing",
        metrics_source_file=metrics_path.name,
        strategy_params_source_file=strategy_path.name,
        params_source_type=params_source_type,
        fallback_used=bool(fallback_reasons),
        fallback_reason="; ".join(fallback_reasons),
        combined_path=combined,
        local_matched_path=local_matched,
        bet_log_path=bet_log_path,
        metrics_path=metrics_path,
        strategy_params_path=strategy_path,
    )


def copy_selection_aliases(selection: SnapshotSelection, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(selection.combined_path, output_dir / "combined_latest.csv")
    shutil.copy2(selection.local_matched_path, output_dir / "local_matched_games_latest.csv")
    shutil.copy2(selection.strategy_params_path, output_dir / "strategy_params.json")
    shutil.copy2(selection.metrics_path, output_dir / "metrics_snapshot.json")
    if selection.bet_log_path and selection.bet_log_path.exists():
        shutil.copy2(selection.bet_log_path, output_dir / "bet_log_flat_live.csv")


def _serializable(selection: SnapshotSelection) -> dict:
    payload = asdict(selection)
    for key in ("combined_path", "local_matched_path", "bet_log_path", "metrics_path", "strategy_params_path"):
        value = payload.get(key)
        payload[key] = str(value) if value else None
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=str)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--manifest-path", type=str, default=None)
    args = parser.parse_args()

    selection = resolve_snapshot_selection(Path(args.source_root))
    if args.output_dir:
        copy_selection_aliases(selection, Path(args.output_dir))
    manifest_path = Path(args.manifest_path) if args.manifest_path else None
    if manifest_path:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(_serializable(selection), indent=2), encoding="utf-8")

    print(json.dumps(_serializable(selection), indent=2))


if __name__ == "__main__":
    main()
