#!/usr/bin/env python3
"""Shared source/snapshot selection for local and GitHub dashboard runs."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")
PARAM_KEYS = {"home_win_rate_threshold", "odds_min", "odds_max", "prob_threshold", "min_ev", "min_ev_per_100"}


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
    bet_log_latest_date: Optional[str]
    bet_log_contains_future_rows: bool
    bet_log_will_be_trimmed_to_snapshot: bool
    bet_log_lags_snapshot: bool
    bet_log_lag_days: Optional[int]
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


def _exact_dated(base_dir: Path, prefix: str, snapshot_date: str, suffix: str) -> Optional[Path]:
    exact = base_dir / f"{prefix}{snapshot_date}{suffix}"
    return exact if exact.exists() else None


def _first_existing(paths: Iterable[Optional[Path]]) -> Optional[Path]:
    return next((path for path in paths if path is not None and path.exists()), None)


def _normalize_param_key(key: str) -> str:
    normalized = re.sub(r"[\s\-]+", "_", key.strip().lower())
    normalized = re.sub(r"[^a-z0-9_]", "", normalized)
    return re.sub(r"_+", "_", normalized)


def _coerce_scalar(value: str) -> object:
    raw = value.strip()
    if not raw:
        return ""
    lower = raw.lower()
    if lower in {"true", "false"}:
        return lower == "true"
    try:
        number = float(raw)
    except ValueError:
        return raw
    if number.is_integer():
        return int(number)
    return number


def _strategy_txt_to_json_payload(path: Path) -> dict[str, object]:
    payload: dict[str, object] = {"version": 1, "params": {}}
    params: dict[str, object] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        normalized = _normalize_param_key(key)
        coerced = _coerce_scalar(value)
        payload[normalized] = coerced
        if normalized in PARAM_KEYS:
            params["min_ev" if normalized == "min_ev_per_100" else normalized] = coerced
    payload["params"] = params
    return payload


def _copy_strategy_alias(strategy_params_path: Path, output_path: Path) -> None:
    if strategy_params_path.suffix.lower() == ".json":
        shutil.copy2(strategy_params_path, output_path)
        return
    payload = _strategy_txt_to_json_payload(strategy_params_path)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _validate_date_match(path: Path, expected: str, label: str) -> None:
    date_from_source = _extract_date_from_name(path.name) or _extract_date_from_json(path)
    if date_from_source and date_from_source != expected:
        raise RuntimeError(
            f"{label} date mismatch: expected {expected}, got {date_from_source} from {path.name}"
        )




def _resolve_strategy_params_for_snapshot(lightgbm: Path, snapshot_date: str) -> tuple[Optional[Path], str, list[str], list[str]]:
    candidate_reasons: list[str] = []
    fallback_reasons: list[str] = []

    exact_json = _exact_dated(lightgbm, "strategy_params_", snapshot_date, ".json")
    if exact_json:
        return exact_json, "dated_exact", candidate_reasons, fallback_reasons

    exact_txt = _exact_dated(lightgbm, "strategy_params_", snapshot_date, ".txt")
    if exact_txt:
        return exact_txt, "dated_exact", candidate_reasons, fallback_reasons

    dated_candidates: list[tuple[str, Path]] = []
    rejected_future = False
    for path in lightgbm.glob("strategy_params_*"):
        if not path.is_file() or path.suffix.lower() not in {".json", ".txt"}:
            continue
        extracted = _extract_date_from_name(path.name)
        if not extracted:
            continue
        if extracted > snapshot_date:
            rejected_future = True
            continue
        dated_candidates.append((extracted, path))

    if rejected_future:
        fallback_reasons.append("rejected_future_dated_strategy_params")

    if dated_candidates:
        chosen = sorted(dated_candidates, key=lambda item: item[0])[-1][1]
        chosen_date = _extract_date_from_name(chosen.name) or _extract_date_from_json(chosen) or _extract_date_from_text(chosen)
        if chosen_date and chosen_date > snapshot_date:
            candidate_reasons.append(
                f"strategy_params date mismatch in {chosen.name}: snapshot {snapshot_date}, got {chosen_date}"
            )
        else:
            fallback_reasons.append("used_older_dated_strategy_params")
            return chosen, "dated_fallback_lte_snapshot", candidate_reasons, fallback_reasons

    for undated_path in (lightgbm / "strategy_params.json", lightgbm / "strategy_params.txt"):
        if not undated_path.exists():
            continue
        undated_date = _extract_date_from_json(undated_path) or _extract_date_from_text(undated_path)
        if undated_date and undated_date > snapshot_date:
            fallback_reasons.append("rejected_future_dated_strategy_params")
            candidate_reasons.append(
                f"undated strategy params {undated_path.name} has future date {undated_date} for snapshot {snapshot_date}"
            )
            continue
        fallback_reasons.append("used_undated_strategy_params")
        return undated_path, "undated_fallback", candidate_reasons, fallback_reasons

    if not candidate_reasons:
        candidate_reasons.append(
            f"missing acceptable strategy_params for snapshot {snapshot_date}; expected exact or <= snapshot dated file"
        )
    return None, "missing", candidate_reasons, fallback_reasons


def _extract_max_date_from_csv(path: Path) -> Optional[str]:
    if not path.exists() or path.suffix.lower() != ".csv":
        return None
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


def resolve_snapshot_selection(source_root: Path) -> SnapshotSelection:
    root = source_root.expanduser().resolve()
    lightgbm = root / "output" / "LightGBM"
    kelly = lightgbm / "Kelly"
    if not lightgbm.exists():
        raise FileNotFoundError(f"Missing LightGBM directory: {lightgbm}")

    combined_candidates: list[tuple[str, Path]] = []
    for path in kelly.glob("combined_nba_predictions_iso_*.csv"):
        if not path.is_file():
            continue
        date = _extract_date_from_name(path.name)
        if date:
            combined_candidates.append((date, path))

    fallback_reasons: list[str] = []
    if not combined_candidates:
        for path in lightgbm.glob("combined_nba_predictions_acc_*.csv"):
            if not path.is_file():
                continue
            date = _extract_date_from_name(path.name)
            if date:
                combined_candidates.append((date, path))
        if combined_candidates:
            fallback_reasons.append("combined_iso_missing_used_acc")

    if not combined_candidates:
        raise FileNotFoundError("No dated combined predictions file found in source root.")

    candidate_failures: list[tuple[str, list[str]]] = []
    sorted_candidates = sorted(combined_candidates, key=lambda item: item[0], reverse=True)

    for snapshot_date, combined in sorted_candidates:
        candidate_reasons: list[str] = []
        candidate_fallback_reasons = list(fallback_reasons)

        local_matched = _exact_dated(lightgbm, "local_matched_games_", snapshot_date, ".csv")
        if local_matched is None:
            candidate_reasons.append(f"missing local_matched_games_{snapshot_date}.csv")

        strategy_path, params_source_type, strategy_reasons, strategy_fallbacks = _resolve_strategy_params_for_snapshot(
            lightgbm, snapshot_date
        )
        candidate_reasons.extend(strategy_reasons)
        candidate_fallback_reasons.extend(strategy_fallbacks)

        metrics_path = _first_existing(
            [
                _exact_dated(lightgbm, "metrics_snapshot_", snapshot_date, ".json"),
                lightgbm / "metrics_snapshot.json",
            ]
        )
        if metrics_path is None:
            candidate_reasons.append("missing metrics_snapshot_[YYYY-MM-DD].json and metrics_snapshot.json")
        else:
            metrics_date = _extract_date_from_name(metrics_path.name) or _extract_date_from_json(metrics_path)
            if metrics_date and metrics_date != snapshot_date:
                candidate_reasons.append(
                    f"metrics_snapshot date mismatch in {metrics_path.name}: expected {snapshot_date}, got {metrics_date}"
                )
            if _extract_date_from_name(metrics_path.name) is None:
                candidate_fallback_reasons.append("used_undated_metrics_snapshot")

        bet_log_path = _first_existing(
            [
                lightgbm / f"bet_log_flat_live_{snapshot_date}.csv",
                lightgbm / "bet_log_flat_live.csv",
            ]
        )
        if bet_log_path is None:
            candidate_reasons.append(f"missing bet_log_flat_live_{snapshot_date}.csv and bet_log_flat_live.csv")
            bet_log_latest_date = None
            bet_log_contains_future_rows = False
            bet_log_will_be_trimmed_to_snapshot = False
            bet_log_lags_snapshot = False
            bet_log_lag_days = None
        else:
            bet_log_date = _extract_date_from_name(bet_log_path.name) or _extract_max_date_from_csv(bet_log_path)
            if bet_log_date is None:
                candidate_reasons.append(
                    f"unable to determine bet_log_flat_live date from {bet_log_path.name}"
                )
                bet_log_latest_date = None
                bet_log_contains_future_rows = False
                bet_log_will_be_trimmed_to_snapshot = False
                bet_log_lags_snapshot = False
                bet_log_lag_days = None
            else:
                bet_log_latest_date = bet_log_date
                bet_log_contains_future_rows = bet_log_date > snapshot_date
                bet_log_will_be_trimmed_to_snapshot = bet_log_contains_future_rows
                bet_log_lags_snapshot = bet_log_date < snapshot_date
                if bet_log_contains_future_rows:
                    candidate_fallback_reasons.append("bet_log_contains_future_rows_trimmed_to_snapshot")
                if bet_log_lags_snapshot:
                    lag_days = (
                        datetime.strptime(snapshot_date, "%Y-%m-%d")
                        - datetime.strptime(bet_log_date, "%Y-%m-%d")
                    ).days
                    bet_log_lag_days = lag_days
                    candidate_fallback_reasons.append("bet_log_lags_snapshot")
                else:
                    bet_log_lag_days = 0

        if candidate_reasons:
            candidate_failures.append((snapshot_date, candidate_reasons))
            continue

        return SnapshotSelection(
            snapshot_as_of_date=snapshot_date,
            run_date=snapshot_date,
            combined_source_file=combined.name,
            local_matched_source_file=local_matched.name,
            bet_log_source_file=bet_log_path.name if bet_log_path else "missing",
            metrics_source_file=metrics_path.name,
            strategy_params_source_file=strategy_path.name,
            params_source_type=params_source_type,
            bet_log_latest_date=bet_log_latest_date,
            bet_log_contains_future_rows=bet_log_contains_future_rows,
            bet_log_will_be_trimmed_to_snapshot=bet_log_will_be_trimmed_to_snapshot,
            bet_log_lags_snapshot=bet_log_lags_snapshot,
            bet_log_lag_days=bet_log_lag_days,
            fallback_used=bool(candidate_fallback_reasons),
            fallback_reason="; ".join(candidate_fallback_reasons),
            combined_path=combined,
            local_matched_path=local_matched,
            bet_log_path=bet_log_path,
            metrics_path=metrics_path,
            strategy_params_path=strategy_path,
        )

    failure_lines = [
        f"- {date}: " + "; ".join(reasons)
        for date, reasons in candidate_failures
    ]
    detail = "\n".join(failure_lines) if failure_lines else "- no candidate dates found"
    raise FileNotFoundError(
        "No complete dated snapshot could be resolved. Checked candidate dates (newest first):\n"
        f"{detail}"
    )


def copy_selection_aliases(selection: SnapshotSelection, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(selection.combined_path, output_dir / "combined_latest.csv")
    shutil.copy2(selection.local_matched_path, output_dir / "local_matched_games_latest.csv")
    _copy_strategy_alias(selection.strategy_params_path, output_dir / "strategy_params.json")
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
