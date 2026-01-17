#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate stats-only dashboard artifacts for hoops-insight.

This script reads historical outputs from Basketball_prediction (played games only)
and writes stable JSON files into public/data for the Vite app to consume.
No future predictions are loaded or emitted.

Data contract overview:
- Model performance (window / historical): combined_nba_predictions_* (played games only).
- Strategy simulation (local matched games): local_matched_games_YYYY-MM-DD.csv.
- Placed bets (real, settled): bet_log_flat_live.csv settled against combined_* results.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import pandas as pd


DATE_FMT = "%Y-%m-%d"
CALIBRATION_WINDOW = 200
DEFAULT_MAX_ODDS_FALLBACK = 3.2
RISK_MIN_SAMPLE = 5
THRESHOLDS = [
    {"label": "> 0.60", "thresholdType": "gt", "threshold": 0.60},
    {"label": "<= 0.40", "thresholdType": "lt", "threshold": 0.40},
]


@dataclass
class SourcePaths:
    combined_iso: Optional[Path]
    combined_acc: Optional[Path]
    bet_log: Optional[Path]
    bet_log_flat: Optional[Path]
    metrics_snapshot: Optional[Path]
    local_matched_games: Optional[Path]
    strategy_params: Optional[Path]


def _normalize_key(key: str) -> str:
    key = key.strip().lower()
    key = re.sub(r"[\s\-]+", "_", key)
    key = re.sub(r"[^a-z0-9_]", "", key)
    key = re.sub(r"_+", "_", key)
    return key


def _safe_float(val: str) -> Optional[float]:
    if val is None:
        return None
    s = str(val).strip()
    if s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _safe_date(val: str) -> Optional[datetime]:
    if val is None:
        return None
    s = str(val).strip()
    if s == "":
        return None
    try:
        return datetime.strptime(s, DATE_FMT)
    except ValueError:
        return None


def _safe_team(val: object) -> Optional[str]:
    if val is None:
        return None
    s = str(val).strip()
    if s == "":
        return None
    return re.sub(r"\s+", " ", s).upper()


def _max_date(rows: Iterable[Dict[str, object]], key: str = "date") -> Optional[datetime]:
    dates: List[datetime] = []
    for row in rows:
        raw = row.get(key)
        if isinstance(raw, datetime):
            dates.append(raw)
            continue
        if raw is None:
            continue
        parsed = _safe_date(str(raw))
        if parsed:
            dates.append(parsed)
    return max(dates) if dates else None


def compute_window_bounds(
    rows: List[Dict[str, object]], window_size: int
) -> Tuple[List[Dict[str, object]], Optional[datetime], Optional[datetime]]:
    if not rows:
        return [], None, None
    sorted_rows = sorted(rows, key=lambda r: r["date"])
    window_rows = sorted_rows[-window_size:] if window_size else sorted_rows
    if not window_rows:
        return [], None, None
    return window_rows, window_rows[0]["date"], window_rows[-1]["date"]


def _coerce_value(val: object) -> object:
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return val
    s = str(val).strip()
    if s == "":
        return None
    try:
        num = float(s)
    except ValueError:
        return val
    if num.is_integer():
        return int(num)
    return num


def _normalize_params(params: Dict[str, object]) -> Dict[str, object]:
    normalized: Dict[str, object] = {}
    for key, value in params.items():
        normalized[_normalize_key(str(key))] = _coerce_value(value)
    return normalized


def _read_csv_normalized(path: Path) -> Iterable[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            normalized = {}
            for k, v in row.items():
                if k is None:
                    continue
                normalized[_normalize_key(k)] = v
            yield normalized


def _parse_matchup(matchup: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    if not matchup:
        return None, None
    raw = str(matchup).strip()
    if raw == "":
        return None, None
    if "@" in raw:
        parts = [p.strip() for p in raw.split("@", 1)]
        if len(parts) == 2:
            away, home = parts
            return _safe_team(home), _safe_team(away)
    for token in (" vs ", " v ", " vs. "):
        if token in raw.lower():
            parts = re.split(r"\s+vs\.?\s+|\s+v\s+", raw, flags=re.IGNORECASE)
            if len(parts) >= 2:
                return _safe_team(parts[0]), _safe_team(parts[1])
    return None, None


def _parse_snapshot_records(records: Iterable[object]) -> Dict[str, Dict[str, object]]:
    parsed: Dict[str, Dict[str, object]] = {}
    for row in records:
        if not isinstance(row, dict):
            continue
        section = row.get("section")
        metric = row.get("metric")
        value = row.get("value")
        if section and metric:
            parsed.setdefault(str(section), {})[str(metric)] = _coerce_value(value)
    return parsed


def load_metrics_snapshot(path: Path) -> Dict[str, Dict[str, object]]:
    if not path.exists():
        return {}
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return _parse_snapshot_records(data)
        elif isinstance(data, dict):
            parsed: Dict[str, Dict[str, object]] = {}
            for key in ("records", "metrics", "rows"):
                records = data.get(key)
                if isinstance(records, list):
                    parsed.update(_parse_snapshot_records(records))
            for section, metrics in data.items():
                if section in ("records", "metrics", "rows"):
                    continue
                if isinstance(metrics, dict):
                    parsed.setdefault(str(section), {}).update(
                        {str(metric): _coerce_value(value) for metric, value in metrics.items()}
                    )
                elif isinstance(metrics, list):
                    parsed.update(_parse_snapshot_records(metrics))
            return parsed
        return {}

    parsed: Dict[str, Dict[str, object]] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            section = row.get("section")
            metric = row.get("metric")
            value = row.get("value")
            if section and metric:
                parsed.setdefault(section, {})[metric] = _coerce_value(value)
    return parsed


def load_strategy_params(path: Path) -> Dict[str, object]:
    params: Dict[str, object] = {}
    if not path.exists():
        return params
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            params.update(data)
        return params
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue
            match = re.match(r"^([^:=#]+)[:=]\s*(.+)$", raw)
            if not match:
                continue
            key = _normalize_key(match.group(1))
            value = _coerce_value(match.group(2))
            params[key] = value
    return params


def _extract_params_used(snapshot: Dict[str, Dict[str, object]]) -> Dict[str, object]:
    for key in ("params_used", "filter_params", "params"):
        params = snapshot.get(key)
        if isinstance(params, dict):
            normalized = _normalize_params(params)
            if normalized:
                return normalized
    return {}


def load_local_matched_games_csv(path: Path) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    required_columns = [
        "date",
        "home_team",
        "away_team",
        "home_win_rate",
        "prob_iso",
        "prob_used",
        "odds_1",
        "EV_€_per_100",
        "win",
        "pnl",
    ]
    df = pd.read_csv(path)
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        return [], {"rows_count": 0, "profit_sum_table": 0.0}

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime(DATE_FMT)
    df["home_team"] = df["home_team"].astype(str).str.strip().str.upper()
    df["away_team"] = df["away_team"].astype(str).str.strip().str.upper()

    numeric_cols = ["home_win_rate", "prob_iso", "prob_used", "odds_1", "EV_€_per_100"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[numeric_cols] = df[numeric_cols].fillna(0.0)

    df["win"] = pd.to_numeric(df["win"], errors="coerce")
    df["pnl"] = pd.to_numeric(df["pnl"], errors="coerce")
    df = df.dropna(subset=["date", "win", "pnl"])
    df = df[df["win"].isin([0, 1])]
    df["win"] = df["win"].astype(int)
    df["pnl"] = df["pnl"].astype(float)

    if "stake" in df.columns:
        df["stake"] = pd.to_numeric(df["stake"], errors="coerce")
        df["stake"] = df["stake"].where(pd.notna(df["stake"]), None)
    else:
        df["stake"] = None

    df = df.rename(columns={"EV_€_per_100": "ev_eur_per_100"})
    rows = df.to_dict(orient="records")
    summary = {
        "rows_count": int(len(df)),
        "profit_sum_table": float(df["pnl"].sum()),
    }
    return rows, summary


def _find_latest_file(path: Path, prefix: str) -> Optional[Path]:
    if not path.exists():
        return None
    pattern = re.compile(rf"{re.escape(prefix)}_(\d{{4}}-\d{{2}}-\d{{2}})\.csv$")
    candidates: List[Tuple[datetime, Path]] = []
    for item in path.iterdir():
        if not item.is_file():
            continue
        match = pattern.match(item.name)
        if not match:
            continue
        dt = _safe_date(match.group(1))
        if dt:
            candidates.append((dt, item))
    if not candidates:
        return None
    return sorted(candidates, key=lambda x: x[0])[-1][1]


def _find_latest_dated_file(
    roots: Iterable[Path],
    pattern: re.Pattern,
    date_group: int = 1,
    tiebreaker: Optional[Callable[[re.Match], int]] = None,
) -> Optional[Path]:
    candidates: List[Tuple[datetime, int, Path]] = []
    for root in roots:
        if not root.exists():
            continue
        for item in root.iterdir():
            if not item.is_file():
                continue
            match = pattern.match(item.name)
            if not match:
                continue
            dt = _safe_date(match.group(date_group))
            if not dt:
                continue
            tie = tiebreaker(match) if tiebreaker else 0
            candidates.append((dt, tie, item))
    if not candidates:
        return None
    return max(candidates, key=lambda entry: (entry[0], entry[1]))[2]


def _find_latest_by_mtime(path: Path, pattern: str) -> Optional[Path]:
    if not path.exists():
        return None
    candidates = [item for item in path.glob(pattern) if item.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda item: item.stat().st_mtime)


def _find_metrics_snapshot(lightgbm_dir: Path, root: Optional[Path]) -> Optional[Path]:
    candidate_dirs = [lightgbm_dir, lightgbm_dir / "metrics_snapshot"]
    if root:
        candidate_dirs.extend(
            [
                root,
                root / "output" / "LightGBM",
                root / "output" / "LightGBM" / "metrics_snapshot",
            ]
        )
    dated_snapshot_pattern = re.compile(r"metrics_snapshot_(\d{4}-\d{2}-\d{2})\.(json|csv)$")
    dated_betting_pattern = re.compile(
        r"betting_metrics_snapshot_(\d{4}-\d{2}-\d{2})\.csv$"
    )

    def _snapshot_tiebreaker(match: re.Match) -> int:
        return 1 if match.group(2) == "json" else 0

    dated_snapshot = _find_latest_dated_file(
        candidate_dirs, dated_snapshot_pattern, tiebreaker=_snapshot_tiebreaker
    )
    if dated_snapshot:
        return dated_snapshot

    dated_betting_snapshot = _find_latest_dated_file(candidate_dirs, dated_betting_pattern)
    if dated_betting_snapshot:
        return dated_betting_snapshot

    for candidate_dir in candidate_dirs:
        if not candidate_dir.exists():
            continue
        preferred = candidate_dir / "metrics_snapshot.json"
        if preferred.exists():
            return preferred
        latest_snapshot = _find_latest_by_mtime(candidate_dir, "metrics_snapshot*.json")
        if latest_snapshot:
            return latest_snapshot
        latest_csv = _find_latest_by_mtime(candidate_dir, "betting_metrics_snapshot_*.csv")
        if latest_csv:
            return latest_csv
        for name in ("metrics_snapshot.csv", "metrics_snapshot.json", "metrics_snapshot_latest.json"):
            candidate = candidate_dir / name
            if candidate.exists():
                return candidate
    return None


def _find_local_matched_games(lightgbm_dir: Path, as_of_date: Optional[str]) -> Optional[Path]:
    if as_of_date:
        candidate = lightgbm_dir / f"local_matched_games_{as_of_date}.csv"
        if candidate.exists():
            return candidate
    return _find_latest_by_mtime(lightgbm_dir, "local_matched_games_*.csv")


def _find_strategy_params(lightgbm_dir: Path, as_of_date: Optional[str]) -> Optional[Path]:
    if as_of_date:
        dated_json = lightgbm_dir / f"strategy_params_{as_of_date}.json"
        if dated_json.exists():
            return dated_json
        dated = lightgbm_dir / f"strategy_params_{as_of_date}.txt"
        if dated.exists():
            return dated
    preferred_json = lightgbm_dir / "strategy_params.json"
    if preferred_json.exists():
        return preferred_json
    preferred = lightgbm_dir / "strategy_params.txt"
    if preferred.exists():
        return preferred
    latest_json = _find_latest_by_mtime(lightgbm_dir, "strategy_params*.json")
    if latest_json:
        return latest_json
    return _find_latest_by_mtime(lightgbm_dir, "strategy_params*.txt")


def _require_existing(path: Optional[Path], label: str) -> Path:
    if path is None or not path.exists():
        raise FileNotFoundError(f"Required source file missing: {label}")
    return path


def resolve_source_root(cli_root: Optional[str], repo_root: Path) -> Optional[Path]:
    env_root = os.getenv("SOURCE_ROOT", "").strip()
    if env_root:
        candidate = Path(env_root).expanduser().resolve()
        if candidate.exists():
            return candidate
    if cli_root:
        candidate = Path(cli_root).expanduser().resolve()
        if candidate.exists():
            return candidate
    fallback = repo_root / "Basketball_prediction" / "2026"
    if fallback.exists():
        return fallback.resolve()
    return None


def _resolve_sources(root: Optional[Path], as_of_date: Optional[str]) -> SourcePaths:
    if root is None:
        return SourcePaths(
            combined_iso=None,
            combined_acc=None,
            bet_log=None,
            bet_log_flat=None,
            metrics_snapshot=None,
            local_matched_games=None,
            strategy_params=None,
        )
    lightgbm_dir = root / "output" / "LightGBM"
    kelly_dir = lightgbm_dir / "Kelly"
    combined_pattern = re.compile(
        r"combined_nba_predictions_(iso|acc)_(\d{4}-\d{2}-\d{2})\.csv$"
    )

    def _combined_tiebreaker(match: re.Match) -> int:
        return 1 if match.group(1) == "iso" else 0

    combined_iso = _find_latest_file(kelly_dir, "combined_nba_predictions_iso")
    combined_acc = _find_latest_file(lightgbm_dir, "combined_nba_predictions_acc")

    latest_combined = _find_latest_dated_file(
        [kelly_dir, lightgbm_dir], combined_pattern, date_group=2, tiebreaker=_combined_tiebreaker
    )
    if latest_combined:
        if "combined_nba_predictions_iso_" in latest_combined.name:
            combined_iso = latest_combined
        else:
            combined_acc = latest_combined

    if combined_iso is None:
        combined_iso = _find_latest_by_mtime(kelly_dir, "combined_nba_predictions_iso*.csv")
    if combined_acc is None:
        combined_acc = _find_latest_by_mtime(lightgbm_dir, "combined_nba_predictions_acc*.csv")
    bet_log = lightgbm_dir / "bet_log_live.csv"
    if not bet_log.exists():
        bet_log = _find_latest_file(lightgbm_dir, "bet_log_live")
    bet_log_flat = lightgbm_dir / "bet_log_flat_live.csv"
    if not bet_log_flat.exists():
        bet_log_flat = _find_latest_file(lightgbm_dir, "bet_log_flat_live")
    metrics_snapshot = _find_metrics_snapshot(lightgbm_dir, root)
    strategy_params = _find_strategy_params(lightgbm_dir, as_of_date)
    local_matched_games = _find_local_matched_games(lightgbm_dir, as_of_date)
    return SourcePaths(
        combined_iso=combined_iso,
        combined_acc=combined_acc,
        bet_log=bet_log,
        bet_log_flat=bet_log_flat,
        metrics_snapshot=metrics_snapshot,
        local_matched_games=local_matched_games,
        strategy_params=strategy_params,
    )


def _compute_log_loss(y: List[int], p: List[float]) -> float:
    eps = 1e-6
    total = 0.0
    for yi, pi in zip(y, p):
        pi = min(max(pi, eps), 1.0 - eps)
        total += -(yi * math.log(pi) + (1 - yi) * math.log(1.0 - pi))
    return total / len(y) if y else 0.0


def _compute_brier(y: List[int], p: List[float]) -> float:
    if not y:
        return 0.0
    return sum((pi - yi) ** 2 for yi, pi in zip(y, p)) / len(y)


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def _serialize(obj):
    if isinstance(obj, datetime):
        return obj.strftime(DATE_FMT)
    return obj


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1 / (1 + z)
    z = math.exp(x)
    return z / (1 + z)


def _logit(p: float) -> float:
    eps = 1e-6
    p = min(max(p, eps), 1.0 - eps)
    return math.log(p / (1 - p))


def _fit_logistic_calibration(y: List[int], p: List[float]) -> Tuple[Optional[float], Optional[float]]:
    if len(y) < 5 or len(y) != len(p):
        return None, None
    x_vals = [_logit(pi) for pi in p]
    intercept = 0.0
    slope = 1.0
    for _ in range(50):
        preds = [_sigmoid(intercept + slope * x) for x in x_vals]
        w_vals = [pred * (1 - pred) for pred in preds]
        g0 = sum(yi - pi for yi, pi in zip(y, preds))
        g1 = sum((yi - pi) * x for yi, pi, x in zip(y, preds, x_vals))
        h00 = -sum(w_vals)
        h01 = -sum(w * x for w, x in zip(w_vals, x_vals))
        h11 = -sum(w * x * x for w, x in zip(w_vals, x_vals))
        det = h00 * h11 - h01 * h01
        if det == 0:
            break
        delta0 = (g0 * h11 - g1 * h01) / det
        delta1 = (g1 * h00 - g0 * h01) / det
        intercept -= delta0
        slope -= delta1
        if abs(delta0) < 1e-6 and abs(delta1) < 1e-6:
            break
    return intercept, slope


def _compute_ece(y: List[int], p: List[float], bins: int = 10) -> Optional[float]:
    if not y or len(y) != len(p):
        return None
    total = len(y)
    ece = 0.0
    for i in range(bins):
        lower = i / bins
        upper = (i + 1) / bins
        bucket = [(yi, pi) for yi, pi in zip(y, p) if lower <= pi < upper]
        if not bucket:
            continue
        bucket_y = [yi for yi, _ in bucket]
        bucket_p = [pi for _, pi in bucket]
        acc = sum(bucket_y) / len(bucket_y)
        conf = sum(bucket_p) / len(bucket_p)
        ece += abs(acc - conf) * (len(bucket) / total)
    return ece


def load_played_games(path: Path) -> List[Dict[str, object]]:
    rows = []
    for row in _read_csv_normalized(path):
        home_team = row.get("home_team")
        away_team = row.get("away_team")
        result = row.get("result") or row.get("result_raw")
        if result is None or str(result).strip() in {"", "0"}:
            continue
        date_raw = row.get("game_date") or row.get("date")
        game_date = _safe_date(date_raw)
        if game_date is None:
            continue
        home_team_won = 1 if str(result).strip() == str(home_team).strip() else 0

        prob_raw = _safe_float(row.get("pred_home_win_proba") or row.get("home_team_prob"))
        prob_iso = _safe_float(row.get("iso_proba_home_win"))
        odds = _safe_float(row.get("closing_home_odds") or row.get("odds_1"))
        home_win_rate = _safe_float(row.get("home_win_rate"))

        rows.append(
            {
                "date": game_date,
                "home_team": home_team,
                "away_team": away_team,
                "home_team_won": home_team_won,
                "prob_raw": prob_raw,
                "prob_iso": prob_iso,
                "odds_home": odds,
                "home_win_rate": home_win_rate,
            }
        )
    return rows


def build_historical_stats(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    per_day = defaultdict(lambda: {"total": 0, "correct": 0})
    for r in rows:
        d = r["date"].strftime(DATE_FMT)
        per_day[d]["total"] += 1
        per_day[d]["correct"] += int(r["home_team_won"])
    output = []
    for d in sorted(per_day.keys()):
        total = per_day[d]["total"]
        correct = per_day[d]["correct"]
        output.append(
            {
                "date": d,
                "accuracy": _safe_div(correct, total),
                "totalGames": total,
                "correctGames": correct,
            }
        )
    return output


def build_accuracy_thresholds(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    out = []
    for spec in THRESHOLDS:
        threshold = spec["threshold"]
        passed = []
        for r in rows:
            p = r["prob_iso"] if r["prob_iso"] is not None else r["prob_raw"]
            if p is None:
                continue
            if spec["thresholdType"] == "gt" and p > threshold:
                passed.append(r)
            elif spec["thresholdType"] == "lt" and p <= threshold:
                passed.append(r)
        total = len(passed)
        correct = sum(int(r["home_team_won"]) for r in passed)
        out.append(
            {
                "label": spec["label"],
                "thresholdType": spec["thresholdType"],
                "threshold": threshold,
                "accuracy": _safe_div(correct, total),
                "sampleSize": total,
            }
        )
    return out


def build_calibration_metrics(
    rows: List[Dict[str, object]], window_size: int = CALIBRATION_WINDOW
) -> Dict[str, object]:
    if not rows:
        return {
            "asOfDate": "—",
            "brierBefore": None,
            "brierAfter": None,
            "logLossBefore": None,
            "logLossAfter": None,
            "fittedGames": 0,
            "ece": None,
            "calibrationSlope": None,
            "calibrationIntercept": None,
            "avgPredictedProb": None,
            "baseRate": None,
            "actualWinPct": None,
            "windowSize": 0,
        }

    window_rows, _, window_end = compute_window_bounds(rows, window_size)
    window_size_used = len(window_rows)
    as_of = window_end.strftime(DATE_FMT) if window_end else "—"

    p_raw = [r["prob_raw"] for r in window_rows if r["prob_raw"] is not None]
    y_raw = [int(r["home_team_won"]) for r in window_rows if r["prob_raw"] is not None]
    p_iso = [r["prob_iso"] for r in window_rows if r["prob_iso"] is not None]
    y_iso = [int(r["home_team_won"]) for r in window_rows if r["prob_iso"] is not None]

    prob_used = []
    y_used = []
    for r in window_rows:
        prob = r["prob_iso"] if r["prob_iso"] is not None else r["prob_raw"]
        if prob is None:
            continue
        prob_used.append(prob)
        y_used.append(int(r["home_team_won"]))

    if window_size_used:
        base_rate = _safe_div(sum(int(r["home_team_won"]) for r in window_rows), window_size_used)
    else:
        base_rate = None

    avg_pred_prob = _safe_div(sum(prob_used), len(prob_used)) if prob_used else None

    brier_before = _compute_brier(y_raw, p_raw) if p_raw else None
    logloss_before = _compute_log_loss(y_raw, p_raw) if p_raw else None
    brier_after = _compute_brier(y_iso, p_iso) if p_iso else None
    logloss_after = _compute_log_loss(y_iso, p_iso) if p_iso else None

    if p_iso:
        fitted_games = len(p_iso)
    elif p_raw:
        fitted_games = len(p_raw)
    else:
        fitted_games = 0

    calibration_intercept, calibration_slope = _fit_logistic_calibration(y_used, prob_used)
    ece = _compute_ece(y_used, prob_used, bins=10)

    return {
        "asOfDate": as_of,
        "brierBefore": brier_before,
        "brierAfter": brier_after,
        "logLossBefore": logloss_before,
        "logLossAfter": logloss_after,
        "fittedGames": fitted_games,
        "ece": ece,
        "calibrationSlope": calibration_slope,
        "calibrationIntercept": calibration_intercept,
        "avgPredictedProb": avg_pred_prob,
        "baseRate": base_rate,
        "actualWinPct": base_rate,
        "windowSize": window_size_used,
    }


def build_home_win_rates_last20(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    per_team = defaultdict(list)
    for r in rows:
        per_team[str(r["home_team"])].append((r["date"], True, int(r["home_team_won"])))
        per_team[str(r["away_team"])].append((r["date"], False, 0))

    output = []
    for team, games in per_team.items():
        games = sorted(games, key=lambda x: x[0])[-20:]
        total_last20 = len(games)
        home_games = [g for g in games if g[1]]
        total_home = len(home_games)
        home_wins = sum(g[2] for g in home_games)
        home_win_rate = _safe_div(home_wins, total_home)
        output.append(
            {
                "team": team,
                "totalLast20Games": total_last20,
                "totalHomeGames": total_home,
                "homeWins": home_wins,
                "homeWinRate": home_win_rate,
            }
        )
    output.sort(key=lambda x: (x["homeWinRate"], x["totalHomeGames"]), reverse=True)
    return [row for row in output if row["homeWinRate"] > 0.50]


def _get_param(params: Dict[str, object], *names: str) -> Optional[float]:
    for name in names:
        key = _normalize_key(name)
        if key in params:
            val = _coerce_value(params[key])
            return float(val) if isinstance(val, (int, float)) else None
    return None


def _prob_used(row: Dict[str, object]) -> Optional[float]:
    return row["prob_iso"] if row.get("prob_iso") is not None else row.get("prob_raw")


def _compute_ev_per_100(prob: Optional[float], odds: Optional[float]) -> Optional[float]:
    if prob is None or odds is None:
        return None
    return (prob * (odds - 1.0) - (1.0 - prob)) * 100.0


def build_strategy_filter_stats(
    rows: List[Dict[str, object]], params: Dict[str, object], window_size: int
) -> Dict[str, object]:
    if not rows:
        return {
            "window_size": window_size,
            "filters": [],
            "matched_games_count": 0,
            "window_start": None,
            "window_end": None,
        }
    window_rows, window_start, window_end = compute_window_bounds(rows, window_size)
    filters = [{"label": "Window games", "count": len(window_rows)}]

    min_prob_used = _get_param(
        params, "prob_threshold", "min_prob_used", "min_prob", "min_prob_iso"
    )
    min_odds = _get_param(params, "odds_min", "min_odds_1", "min_odds")
    max_odds = _get_param(params, "odds_max", "max_odds_1", "max_odds")
    if max_odds is None:
        max_odds = DEFAULT_MAX_ODDS_FALLBACK
    min_ev = _get_param(params, "min_ev", "min_ev_eur_per_100", "min_ev_per_100")
    min_home_win_rate = _get_param(params, "home_win_rate_threshold", "min_home_win_rate")

    current = window_rows
    if min_prob_used is not None:
        current = [
            r for r in current if _prob_used(r) is not None and _prob_used(r) >= min_prob_used
        ]
        filters.append({"label": f"Prob used ≥ {min_prob_used:.3f}", "count": len(current)})
    if min_odds is not None:
        current = [r for r in current if r.get("odds_home") is not None and r["odds_home"] >= min_odds]
        filters.append({"label": f"Odds ≥ {min_odds:.2f}", "count": len(current)})
    if max_odds is not None:
        current = [r for r in current if r.get("odds_home") is not None and r["odds_home"] <= max_odds]
        filters.append({"label": f"Odds ≤ {max_odds:.2f}", "count": len(current)})
    if min_ev is not None:
        current = [
            r
            for r in current
            if _compute_ev_per_100(_prob_used(r), r.get("odds_home")) is not None
            and _compute_ev_per_100(_prob_used(r), r.get("odds_home")) > min_ev
        ]
        filters.append({"label": f"EV €/100 > {min_ev:.2f}", "count": len(current)})
    if min_home_win_rate is not None:
        current = [
            r
            for r in current
            if r.get("home_win_rate") is not None and r["home_win_rate"] >= min_home_win_rate
        ]
        filters.append(
            {"label": f"Home win rate ≥ {min_home_win_rate:.2f}", "count": len(current)}
        )

    return {
        "window_size": window_size,
        "filters": filters,
        "matched_games_count": len(current),
        "window_start": window_start,
        "window_end": window_end,
    }


def filter_local_matched_games_window(
    rows: List[Dict[str, object]], window_start: Optional[str], window_end: Optional[str]
) -> List[Dict[str, object]]:
    if not rows or not window_start or not window_end:
        return rows
    start_date = _safe_date(window_start)
    end_date = _safe_date(window_end)
    if not start_date or not end_date:
        return rows
    filtered = []
    for row in rows:
        row_date = _safe_date(row.get("date"))
        if row_date and start_date <= row_date <= end_date:
            filtered.append(row)
    return filtered


def _compute_local_bankroll(rows: List[Dict[str, object]], start: float, stake: float) -> Dict[str, float]:
    net_pl = sum(row.get("pnl", 0.0) for row in rows)
    return {"start": start, "stake": stake, "net_pl": net_pl, "bankroll": start + net_pl}


def _compute_sharpe_style(rows: List[Dict[str, object]]) -> Optional[float]:
    pnl_values = [row.get("pnl") for row in rows if row.get("pnl") is not None]
    n_trades = len(pnl_values)
    if n_trades < RISK_MIN_SAMPLE:
        return None
    mean_pnl = sum(pnl_values) / n_trades
    variance = sum((pnl - mean_pnl) ** 2 for pnl in pnl_values) / n_trades
    std_dev = math.sqrt(variance)
    if std_dev == 0:
        return None
    return (mean_pnl / std_dev) * math.sqrt(n_trades)


def build_local_equity_history(
    rows: List[Dict[str, object]], start: float
) -> List[Dict[str, object]]:
    if not rows:
        return []
    sorted_rows = sorted(rows, key=lambda r: r["date"])
    balance = start
    history = []
    for row in sorted_rows:
        pnl = row.get("pnl", 0.0) or 0.0
        balance += pnl
        history.append(
            {
                "date": row["date"],
                "balance": balance,
                "betsPlaced": 1,
                "profit": pnl,
            }
        )
    return history


def _snapshot_value(snapshot: Dict[str, Dict[str, object]], section: str, metric: str) -> Optional[object]:
    return snapshot.get(section, {}).get(metric)


def _label_path(path: Optional[Path]) -> str:
    if not path:
        return "missing"
    return path.name


def load_bet_log(path: Path) -> List[Dict[str, object]]:
    rows = []
    for row in _read_csv_normalized(path):
        date = _safe_date(row.get("date"))
        if date is None:
            continue
        rows.append(
            {
                "date": date,
                "stake_eur": _safe_float(row.get("stake_eur")),
                "profit_eur": _safe_float(row.get("profit_eur")),
                "won": _safe_float(row.get("won")),
                "bankroll": _safe_float(row.get("bankroll")),
                "bankroll_after": _safe_float(row.get("bankroll_after")),
            }
        )
    return rows


def load_bet_log_flat(path: Path) -> List[Dict[str, object]]:
    rows = []
    for row in _read_csv_normalized(path):
        date = _safe_date(row.get("date") or row.get("game_date") or row.get("event_date"))
        home_team = _safe_team(row.get("home_team") or row.get("home"))
        away_team = _safe_team(row.get("away_team") or row.get("away"))
        if not home_team or not away_team:
            matchup_home, matchup_away = _parse_matchup(
                row.get("matchup") or row.get("game") or row.get("event")
            )
            home_team = home_team or matchup_home
            away_team = away_team or matchup_away

        pick_team = _safe_team(
            row.get("selection")
            or row.get("pick")
            or row.get("team")
            or row.get("bet_on")
            or row.get("side")
        )
        odds = _safe_float(
            row.get("odds")
            or row.get("odds_decimal")
            or row.get("price")
            or row.get("odds_1")
        )
        stake = _safe_float(
            row.get("stake") or row.get("stake_eur") or row.get("amount") or row.get("bet_amount")
        )

        rows.append(
            {
                "date": date,
                "home_team": home_team,
                "away_team": away_team,
                "pick_team": pick_team,
                "odds": odds,
                "stake": stake,
                "raw": row,
            }
        )
    return rows


def build_settled_bets(
    bet_rows: List[Dict[str, object]],
    played_rows: List[Dict[str, object]],
    ytd_start: datetime,
) -> List[Dict[str, object]]:
    played_lookup: Dict[Tuple[str, str, str], Dict[str, object]] = {}
    for row in played_rows:
        key = (
            row["date"].strftime(DATE_FMT),
            _safe_team(row.get("home_team")),
            _safe_team(row.get("away_team")),
        )
        played_lookup[key] = row

    settled = []
    for bet in bet_rows:
        bet_date = bet.get("date")
        if not isinstance(bet_date, datetime):
            continue
        if bet_date < ytd_start:
            continue
        home_team = _safe_team(bet.get("home_team"))
        away_team = _safe_team(bet.get("away_team"))
        pick_team = _safe_team(bet.get("pick_team"))
        if not home_team or not away_team or not pick_team:
            continue
        played = played_lookup.get((bet_date.strftime(DATE_FMT), home_team, away_team))
        if not played:
            continue
        home_team_won = int(played.get("home_team_won") or 0)
        win = 1 if home_team_won == 1 else 0
        odds = bet.get("odds")
        if odds is None or not isinstance(odds, (int, float)) or odds <= 0:
            odds = 1.0
        stake = bet.get("stake")
        if stake is None or not isinstance(stake, (int, float)) or stake < 0:
            stake = 0.0
        pnl = (odds - 1.0) * stake if win == 1 else -stake
        settled.append(
            {
                "date": bet_date.strftime(DATE_FMT),
                "home_team": home_team,
                "away_team": away_team,
                "pick_team": pick_team,
                "odds": float(odds),
                "stake": float(stake),
                "win": win,
                "pnl": float(pnl),
            }
        )

    deduped = {}
    for row in sorted(settled, key=lambda r: r["date"], reverse=True):
        key = (
            row["date"],
            row["home_team"],
            row["away_team"],
            row["pick_team"],
            row["odds"],
            row["stake"],
        )
        if key in deduped:
            continue
        deduped[key] = row
    return list(deduped.values())


def assert_settled_bets_match_local(
    settled_rows: List[Dict[str, object]],
    local_rows: List[Dict[str, object]],
) -> None:
    local_lookup: Dict[Tuple[str, str, str], int] = {}
    for row in local_rows:
        date = row.get("date")
        if not date:
            continue
        key = (
            str(date),
            _safe_team(row.get("home_team")),
            _safe_team(row.get("away_team")),
        )
        if not key[1] or not key[2]:
            continue
        local_lookup[key] = int(row.get("win") or 0)

    mismatches = []
    for row in settled_rows:
        date = row.get("date")
        if not date:
            continue
        key = (
            str(date),
            _safe_team(row.get("home_team")),
            _safe_team(row.get("away_team")),
        )
        if key not in local_lookup:
            continue
        local_win = local_lookup[key]
        settled_win = int(row.get("win") or 0)
        if local_win != settled_win:
            mismatches.append((key, settled_win, local_win))

    if mismatches:
        sample = mismatches[:5]
        raise RuntimeError(
            "Settled bets win mismatch with local matched games "
            f"({len(mismatches)} mismatches). Sample: {sample}"
        )


def build_settled_bet_summary(rows: List[Dict[str, object]]) -> Dict[str, object]:
    total = len(rows)
    if not rows:
        return {
            "count": 0,
            "wins": 0,
            "profit_eur": 0.0,
            "roi_pct": 0.0,
            "avg_odds": 0.0,
        }
    wins = sum(1 for row in rows if row.get("win") == 1)
    profit = sum(row.get("pnl", 0.0) for row in rows)
    total_stake = sum(row.get("stake", 0.0) for row in rows)
    avg_odds = _safe_div(sum(row.get("odds", 0.0) for row in rows), total)
    roi_pct = _safe_div(profit, total_stake) * 100.0
    return {
        "count": total,
        "wins": wins,
        "profit_eur": profit,
        "roi_pct": roi_pct,
        "avg_odds": avg_odds,
    }


def _human_readable_filters(params: Dict[str, object]) -> str:
    if not params:
        return "No active filters."
    parts = []
    min_prob_used = _get_param(
        params, "prob_threshold", "min_prob_used", "min_prob", "min_prob_iso"
    )
    min_odds = _get_param(params, "odds_min", "min_odds_1", "min_odds")
    max_odds = _get_param(params, "odds_max", "max_odds_1", "max_odds")
    if max_odds is None:
        max_odds = DEFAULT_MAX_ODDS_FALLBACK
    min_ev = _get_param(params, "min_ev", "min_ev_eur_per_100", "min_ev_per_100")
    min_home_win_rate = _get_param(params, "home_win_rate_threshold", "min_home_win_rate")
    prefer_lower_odds = params.get("prefer_lower_odds")

    def _format_signed(value: float) -> str:
        if value.is_integer():
            raw = f"{int(value)}"
        else:
            raw = f"{value:.2f}"
        return raw.replace("-", "−")

    if min_home_win_rate is not None:
        parts.append(f"HW ≥ {min_home_win_rate:.2f}")
    if min_odds is not None and max_odds is not None:
        parts.append(f"odds {min_odds:.2f}–{max_odds:.2f}")
    elif min_odds is not None:
        parts.append(f"odds ≥ {min_odds:.2f}")
    elif max_odds is not None:
        parts.append(f"odds ≤ {max_odds:.2f}")
    if min_prob_used is not None:
        parts.append(f"p ≥ {min_prob_used:.2f}")
    if min_ev is not None:
        parts.append(f"EV > {_format_signed(float(min_ev))}")
    if isinstance(prefer_lower_odds, str):
        prefer_lower_odds = prefer_lower_odds.lower() in {"true", "1", "yes"}
    if prefer_lower_odds:
        parts.append("Prefer lower odds")
    return " | ".join(parts) if parts else "No active filters."


def build_bet_log_summary(rows: List[Dict[str, object]]) -> Dict[str, object]:
    if not rows:
        return {
            "asOfDate": "—",
            "totalBets": 0,
            "totalStakedEur": 0.0,
            "totalProfitEur": 0.0,
            "roiPct": 0.0,
            "avgStakeEur": 0.0,
            "avgProfitPerBetEur": 0.0,
            "winRate": 0.0,
        }

    total_bets = len(rows)
    total_staked = sum(r["stake_eur"] or 0.0 for r in rows)
    total_profit = sum(r["profit_eur"] or 0.0 for r in rows)
    avg_stake = _safe_div(total_staked, total_bets)
    avg_profit = _safe_div(total_profit, total_bets)
    win_rate = _safe_div(sum(1 for r in rows if (r["won"] or 0) == 1), total_bets)
    as_of = max(r["date"] for r in rows).strftime(DATE_FMT)
    roi_pct = _safe_div(total_profit, total_staked) * 100.0

    return {
        "asOfDate": as_of,
        "totalBets": total_bets,
        "totalStakedEur": total_staked,
        "totalProfitEur": total_profit,
        "roiPct": roi_pct,
        "avgStakeEur": avg_stake,
        "avgProfitPerBetEur": avg_profit,
        "winRate": win_rate,
    }


def build_bankroll_history(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    if not rows:
        return []
    per_day = defaultdict(lambda: {"profit": 0.0, "bets": 0, "balance": None})
    for r in rows:
        d = r["date"].strftime(DATE_FMT)
        per_day[d]["profit"] += r["profit_eur"] or 0.0
        per_day[d]["bets"] += 1
        bal = r["bankroll_after"]
        if bal is None:
            bal = r["bankroll"]
        per_day[d]["balance"] = bal if bal is not None else per_day[d]["balance"]

    output = []
    for d in sorted(per_day.keys()):
        output.append(
            {
                "date": d,
                "balance": per_day[d]["balance"] or 0.0,
                "betsPlaced": per_day[d]["bets"],
                "profit": per_day[d]["profit"],
            }
        )
    return output


def compute_max_drawdown(history: List[Dict[str, object]]) -> Tuple[float, float]:
    if not history:
        return 0.0, 0.0
    balances = [h["balance"] for h in history if h["balance"] is not None]
    peak = None
    max_dd = 0.0
    for b in balances:
        if peak is None or b > peak:
            peak = b
        if peak:
            dd = peak - b
            if dd > max_dd:
                max_dd = dd
    max_dd_pct = _safe_div(max_dd, peak) * 100.0 if peak else 0.0
    return max_dd, max_dd_pct


def compute_local_risk_metrics(
    rows: List[Dict[str, object]], start: float, min_sample: int = RISK_MIN_SAMPLE
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if not rows or len(rows) < min_sample:
        return None, None, None
    sharpe_style = _compute_sharpe_style(rows)
    equity_history = build_local_equity_history(rows, start)
    if len(equity_history) < min_sample:
        return sharpe_style, None, None
    max_dd_eur, max_dd_pct = compute_max_drawdown(equity_history)
    return sharpe_style, max_dd_eur, max_dd_pct


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, default=_serialize, ensure_ascii=False, indent=2)


def copy_sources(output_dir: Path, sources: Dict[str, Optional[Path]]) -> Dict[str, str]:
    copied: Dict[str, str] = {}
    sources_dir = output_dir / "sources"
    sources_dir.mkdir(parents=True, exist_ok=True)
    for label, path in sources.items():
        if not path or not path.exists():
            continue
        dest = sources_dir / path.name
        shutil.copy2(path, dest)
        copied[label] = str(Path("sources") / path.name)
    return copied


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        type=str,
        default=None,
        help="Explicit path to Basketball_prediction/2026 root (fallback when SOURCE_ROOT is unset).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output dir for artifacts (default: hoops-insight/public/data).",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Optional public/data directory to load pre-copied artifacts from.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    data_dir = Path(args.data_dir) if args.data_dir else None
    source_root = None if data_dir else resolve_source_root(args.source_root, repo_root)
    expected_lightgbm_dir = source_root / "output" / "LightGBM" if source_root else None

    output_dir = Path(args.output_dir) if args.output_dir else repo_root / "public" / "data"

    if data_dir:
        sources = SourcePaths(
            combined_iso=data_dir / "combined_latest.csv",
            combined_acc=None,
            bet_log=None,
            bet_log_flat=(data_dir / "bet_log_flat_live.csv") if (data_dir / "bet_log_flat_live.csv").exists() else None,
            metrics_snapshot=data_dir / "metrics_snapshot.json",
            local_matched_games=data_dir / "local_matched_games_latest.csv",
            strategy_params=(data_dir / "strategy_params.json") if (data_dir / "strategy_params.json").exists() else None,
        )
    else:
        sources = _resolve_sources(source_root, None)
    if sources.combined_iso:
        combined_path = sources.combined_iso
    elif sources.combined_acc:
        combined_path = sources.combined_acc
    else:
        raise FileNotFoundError("No combined predictions file found in source output directories.")

    played_rows = load_played_games(combined_path)
    if not played_rows:
        raise RuntimeError("No played games found in combined predictions file.")

    historical_stats = build_historical_stats(played_rows)
    accuracy_thresholds = build_accuracy_thresholds(played_rows)
    calibration = build_calibration_metrics(played_rows, window_size=CALIBRATION_WINDOW)
    home_win_rates = build_home_win_rates_last20(played_rows)

    bet_log_rows = []
    if sources.bet_log and sources.bet_log.exists():
        bet_log_rows = load_bet_log(sources.bet_log)

    bet_log_summary = build_bet_log_summary(bet_log_rows)
    bankroll_history = build_bankroll_history(bet_log_rows)

    snapshot_as_of_date = None
    window_rows, window_start_dt, window_end_dt = compute_window_bounds(
        played_rows, CALIBRATION_WINDOW
    )
    as_of_date = window_end_dt.strftime(DATE_FMT) if window_end_dt else "—"
    window_start_label = window_start_dt.strftime(DATE_FMT) if window_start_dt else None
    window_end_label = window_end_dt.strftime(DATE_FMT) if window_end_dt else None

    metrics_snapshot_path = _require_existing(sources.metrics_snapshot, "metrics_snapshot")
    metrics_snapshot = load_metrics_snapshot(metrics_snapshot_path)
    metrics_snapshot_fallback_source = None
    realized_count_raw = _snapshot_value(metrics_snapshot, "realized", "count")
    realized_profit_raw = _snapshot_value(metrics_snapshot, "realized", "profit_sum")
    realized_roi_raw = _snapshot_value(metrics_snapshot, "realized", "roi")
    realized_win_rate_raw = _snapshot_value(metrics_snapshot, "realized", "win_rate")
    realized_sharpe_raw = _snapshot_value(metrics_snapshot, "realized", "sharpe_style")
    ev_mean_raw = _snapshot_value(metrics_snapshot, "ev_stats", "mean")
    snapshot_as_of_raw = _snapshot_value(metrics_snapshot, "meta", "eval_base_date_max")

    snapshot_as_of_date = _safe_date(str(snapshot_as_of_raw)) if snapshot_as_of_raw else None

    local_matched_games_path = _require_existing(
        sources.local_matched_games, "local_matched_games"
    )
    bet_log_flat_path = (
        sources.bet_log_flat if sources.bet_log_flat and sources.bet_log_flat.exists() else None
    )

    local_matched_games_rows, _ = load_local_matched_games_csv(local_matched_games_path)

    strategy_params_path = output_dir / "strategy_params.json"
    if metrics_snapshot_fallback_source and sources.strategy_params:
        params = load_strategy_params(sources.strategy_params)
        strategy_params_source = sources.strategy_params
    elif strategy_params_path.exists():
        params = load_strategy_params(strategy_params_path)
        strategy_params_source = strategy_params_path
    elif sources.strategy_params:
        params = load_strategy_params(sources.strategy_params)
        strategy_params_source = sources.strategy_params
    else:
        params = {}
        strategy_params_source = None

    params_used_source = _label_path(strategy_params_source) if strategy_params_source else None
    params_used = params.get("params_used") if isinstance(params, dict) else None
    if isinstance(params_used, str):
        params_used = {"label": params_used}
    if isinstance(params_used, dict):
        params_used = _normalize_params(params_used)
    elif isinstance(params, dict) and params:
        params_used = _normalize_params(params)
    else:
        params_used = {}
    if not params_used:
        snapshot_params = _extract_params_used(metrics_snapshot)
        if snapshot_params:
            params_used = snapshot_params
            params_used_source = (
                _label_path(sources.metrics_snapshot) if sources.metrics_snapshot else "metrics_snapshot"
            )
            if not params:
                params = {"params_used": params_used}
    raw_params_used_label = None
    if isinstance(params, dict):
        raw_params_used_label = (
            params.get("params_used_label") or params.get("params_label") or params.get("label")
        )
    params_used_label = str(raw_params_used_label).strip() if raw_params_used_label else None
    if not params_used_label:
        params_used_label = "Historical"
    active_filters_label = _human_readable_filters(params_used)

    strategy_filter_stats = build_strategy_filter_stats(played_rows, params_used, window_size=200)
    local_matched_games_rows = filter_local_matched_games_window(
        local_matched_games_rows,
        strategy_filter_stats.get("window_start"),
        strategy_filter_stats.get("window_end"),
    )
    local_matched_games_count = len(local_matched_games_rows)
    local_matched_games_profit_sum = sum(row.get("pnl", 0.0) for row in local_matched_games_rows)
    matched_count_table = local_matched_games_count

    matched_as_of_date = _max_date(local_matched_games_rows)
    if matched_as_of_date:
        strategy_as_of_date = matched_as_of_date.strftime(DATE_FMT)
    else:
        strategy_as_of_date = as_of_date

    def _to_float(value: object) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        return _safe_float(str(value))

    def _to_int(value: object) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        parsed = _safe_float(str(value))
        return int(parsed) if parsed is not None else None

    realized_count = _to_int(realized_count_raw)
    realized_profit = _to_float(realized_profit_raw)
    realized_roi = _to_float(realized_roi_raw)
    realized_win_rate = _to_float(realized_win_rate_raw)
    realized_sharpe = _to_float(realized_sharpe_raw)
    ev_mean = _to_float(ev_mean_raw)
    metrics_snapshot_summary = {
        "realized_count": realized_count,
        "realized_profit_eur": realized_profit,
        "realized_roi": realized_roi,
        "realized_win_rate": realized_win_rate,
        "realized_sharpe": realized_sharpe,
        "ev_mean": ev_mean,
        "eval_base_date_max": snapshot_as_of_date.strftime(DATE_FMT)
        if snapshot_as_of_date
        else None,
    }
    local_sharpe = (
        _compute_sharpe_style(local_matched_games_rows) if local_matched_games_rows else None
    )
    local_avg_odds = (
        _safe_div(
            sum(row.get("odds_1", 0.0) for row in local_matched_games_rows),
            len(local_matched_games_rows),
        )
        if local_matched_games_rows
        else 0.0
    )

    profit_metrics_available = realized_profit is not None and realized_roi is not None
    matched_count_snapshot = realized_count
    matched_count_used: Optional[int] = None
    strategy_summary = {
        "totalBets": 0,
        "totalProfitEur": 0.0,
        "roiPct": 0.0,
        "avgEvPer100": 0.0,
        "winRate": 0.0,
        "sharpeStyle": local_sharpe if local_sharpe is not None else realized_sharpe,
        "profitMetricsAvailable": False,
        "asOfDate": strategy_as_of_date,
    }

    if local_matched_games_rows:
        total_profit = local_matched_games_profit_sum
        total_bets = local_matched_games_count
        roi_pct = _safe_div(total_profit, total_bets * 100.0) * 100.0 if total_bets else 0.0
        strategy_summary = {
            "totalBets": total_bets,
            "totalProfitEur": total_profit,
            "roiPct": roi_pct,
            "avgEvPer100": _safe_div(
                sum(row.get("ev_eur_per_100", 0.0) for row in local_matched_games_rows), total_bets
            ),
            "winRate": _safe_div(
                sum(1 for row in local_matched_games_rows if row.get("win") == 1), total_bets
            ),
            "sharpeStyle": local_sharpe,
            "profitMetricsAvailable": True,
            "asOfDate": strategy_as_of_date,
        }
    elif realized_count is not None:
        roi_pct = 0.0
        if realized_roi is not None:
            roi_pct = realized_roi * 100.0 if realized_roi <= 1.5 else realized_roi
        strategy_summary = {
            "totalBets": realized_count,
            "totalProfitEur": realized_profit or 0.0,
            "roiPct": roi_pct,
            "avgEvPer100": ev_mean or 0.0,
            "winRate": realized_win_rate or 0.0,
            "sharpeStyle": local_sharpe if local_sharpe is not None else realized_sharpe,
            "profitMetricsAvailable": profit_metrics_available,
            "asOfDate": strategy_as_of_date,
        }

    if matched_count_snapshot is not None:
        matched_count_used = matched_count_snapshot
    else:
        matched_count_used = matched_count_table

    if not params:
        if matched_count_snapshot is not None:
            strategy_filter_stats["matched_games_count"] = matched_count_snapshot
        elif matched_count_table is not None:
            strategy_filter_stats["matched_games_count"] = matched_count_table
        else:
            strategy_filter_stats["matched_games_count"] = 0

    local_matched_games_mismatch = False
    local_matched_games_note = ""
    if realized_count is not None:
        if local_matched_games_count != realized_count:
            print(
                "Warning: local matched games count differs from metrics snapshot "
                f"({local_matched_games_count} vs {realized_count})."
            )
        if realized_profit is not None and abs(local_matched_games_profit_sum - realized_profit) > 1e-6:
            print(
                "Warning: local matched games profit sum differs from metrics snapshot "
                f"({local_matched_games_profit_sum:.6f} vs {realized_profit:.6f})."
            )

    bankroll_last_200 = _compute_local_bankroll(local_matched_games_rows, 1000.0, 100.0)
    ytd_rows = [
        row
        for row in local_matched_games_rows
        if _safe_date(row.get("date")) and _safe_date(row.get("date")) >= datetime(2026, 1, 1)
    ]
    bankroll_ytd_2026 = _compute_local_bankroll(ytd_rows, 1000.0, 100.0)
    local_equity_history = build_local_equity_history(local_matched_games_rows, 1000.0)
    _, local_max_dd_eur, local_max_dd_pct = compute_local_risk_metrics(
        local_matched_games_rows, 1000.0, RISK_MIN_SAMPLE
    )

    total_games = len(played_rows)
    total_correct = sum(int(r["home_team_won"]) for r in played_rows)
    overall_accuracy = _safe_div(total_correct, total_games)

    last_run = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    bet_log_flat_rows = load_bet_log_flat(bet_log_flat_path) if bet_log_flat_path else []
    settled_bets_rows = build_settled_bets(
        bet_log_flat_rows, played_rows, ytd_start=datetime(2026, 1, 1)
    )
    if bet_log_flat_rows:
        assert_settled_bets_match_local(settled_bets_rows, local_matched_games_rows)
    settled_bets_summary = build_settled_bet_summary(settled_bets_rows)

    metrics_snapshot_source = _label_path(metrics_snapshot_path)

    window_games_count = next(
        (
            int(filter_item["count"])
            for filter_item in strategy_filter_stats.get("filters", [])
            if filter_item.get("label") == "Window games"
        ),
        0,
    )

    summary_payload = {
        "last_run": last_run,
        "as_of_date": as_of_date,
        "window_size": len(window_rows) if window_rows else 0,
        "window_start": window_start_label,
        "window_end": window_end_label,
        "summary_stats": {
            "total_games": total_games,
            "overall_accuracy": overall_accuracy,
            "as_of_date": as_of_date,
        },
        "kpis": {
            "total_bets": strategy_summary["totalBets"],
            "win_rate": strategy_summary["winRate"],
            "roi_pct": strategy_summary["roiPct"],
            "avg_ev_per_100": strategy_summary["avgEvPer100"],
            "avg_profit_per_bet_eur": bet_log_summary["avgProfitPerBetEur"],
            "max_drawdown_eur": local_max_dd_eur,
            "max_drawdown_pct": local_max_dd_pct,
        },
        "strategy_summary": strategy_summary,
        "strategy_counts": {
            "window_games_count": window_games_count,
            "filter_pass_count": strategy_filter_stats.get("matched_games_count", 0),
            "simulated_bets_count": local_matched_games_count,
            "settled_bets_count": local_matched_games_count,
        },
        "strategy_params": {
            "source": _label_path(strategy_params_source),
            "params": params_used or {},
            "params_used": params_used or {},
            "active_filters": active_filters_label,
            "params_used_label": params_used_label,
            "params_used_source": params_used_source or "missing",
        },
        "strategy_filter_stats": strategy_filter_stats,
        "source": {
            "combined_file": _label_path(combined_path),
            "bet_log_file": str(sources.bet_log) if sources.bet_log else "missing",
            "bet_log_flat_file": str(sources.bet_log_flat)
            if sources.bet_log_flat
            else "missing",
            "metrics_snapshot_source": metrics_snapshot_source,
            "metrics_snapshot_fallback_source": metrics_snapshot_fallback_source or "missing",
        },
    }

    tables_payload = {
        "historical_stats": historical_stats,
        "accuracy_threshold_stats": accuracy_thresholds,
        "calibration_metrics": calibration,
        "home_win_rates_last20": home_win_rates,
        "bet_log_summary": bet_log_summary,
        "bankroll_history": bankroll_history,
        "local_matched_games_rows": local_matched_games_rows,
        "local_matched_games_count": local_matched_games_count,
        "local_matched_games_profit_sum_table": local_matched_games_profit_sum,
        "local_matched_games_mismatch": local_matched_games_mismatch,
        "local_matched_games_note": local_matched_games_note
        or ("No matched games recorded for this window." if not local_matched_games_rows else ""),
        "local_matched_games_source": _label_path(local_matched_games_path),
        "bankroll_last_200": bankroll_last_200,
        "bankroll_ytd_2026": bankroll_ytd_2026,
        "local_matched_games_avg_odds": local_avg_odds,
        "settled_bets_rows": settled_bets_rows,
        "settled_bets_summary": settled_bets_summary,
    }

    last_run_payload = {
        "last_run": last_run,
        "as_of_date": as_of_date,
        "source_root_used": str(source_root) if source_root else "missing",
        "expected_lightgbm_dir": str(expected_lightgbm_dir) if expected_lightgbm_dir else "missing",
        "metrics_snapshot_source": metrics_snapshot_source,
        "metrics_snapshot_fallback_source": metrics_snapshot_fallback_source or "missing",
        "strategy_params_source": _label_path(strategy_params_source),
        "local_matched_games_source": _label_path(local_matched_games_path),
        "bet_log_flat_source": _label_path(bet_log_flat_path),
        "local_matched_games_rows": local_matched_games_count,
        "local_matched_games_profit_sum_table": local_matched_games_profit_sum,
        "matched_count_snapshot": matched_count_snapshot,
        "matched_count_table": matched_count_table,
        "matched_count_used": matched_count_used,
        "strategy_params": {
            "source": _label_path(strategy_params_source),
            "params": params or {},
            "params_used": params_used or {},
            "active_filters": active_filters_label,
            "params_used_label": params_used_label,
            "params_used_source": params_used_source or "missing",
        },
        "strategy_filter_stats": strategy_filter_stats,
        "active_filters_effective": active_filters_label,
        "params_used_label": params_used_label,
        "records": {
            "played_games": total_games,
            "bet_log_rows": len(bet_log_rows),
            "bet_log_flat_rows": len(bet_log_flat_rows),
            "strategy_matched_rows": strategy_summary["totalBets"],
            "local_matched_games_rows": local_matched_games_count,
            "local_matched_games_rows_expected": realized_count or 0,
            "local_matched_games_profit_sum_table": local_matched_games_profit_sum,
            "settled_bets_rows": len(settled_bets_rows),
        },
    }

    copied_sources = copy_sources(
        output_dir,
        {
            "combined_file": combined_path,
            "metrics_snapshot": metrics_snapshot_path,
            "local_matched_games": local_matched_games_path,
            "bet_log_flat": bet_log_flat_path,
        },
    )

    dashboard_payload = {
        "as_of_date": as_of_date,
        "window": {
            "size": len(window_rows) if window_rows else 0,
            "start": window_start_label,
            "end": window_end_label,
            "games_count": window_games_count,
        },
        "active_filters_effective": active_filters_label,
        "params_used_label": params_used_label,
        "metrics_snapshot_summary": metrics_snapshot_summary,
        "summary": summary_payload,
        "tables": tables_payload,
        "last_run": last_run_payload,
        "sources": {
            "combined_file": _label_path(combined_path),
            "metrics_snapshot": metrics_snapshot_source,
            "local_matched_games": _label_path(local_matched_games_path),
            "bet_log_flat": _label_path(bet_log_flat_path),
            "copied": copied_sources,
        },
    }

    window_size_label = strategy_filter_stats.get("window_size") or CALIBRATION_WINDOW
    window_start_display = window_start_label or "—"
    window_end_display = window_end_label or "—"
    active_filters_text = (
        f"{active_filters_label} | window {window_size_label} ({window_start_display} → {window_end_display})"
    )

    dashboard_state = {
        "as_of_date": as_of_date,
        "window_size": window_size_label,
        "window_start": window_start_label,
        "window_end": window_end_label,
        "active_filters_text": active_filters_text,
        "params_used_label": params_used_label,
        "params_source_label": _label_path(strategy_params_source),
        "strategy_as_of_date": matched_as_of_date.strftime(DATE_FMT)
        if matched_as_of_date
        else None,
        "strategy_matches_window": local_matched_games_count,
        "last_update_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sources": {
            "combined": _label_path(combined_path),
            "local_matched": _label_path(local_matched_games_path),
            "bet_log": _label_path(bet_log_flat_path),
            "metrics_snapshot": _label_path(metrics_snapshot_path),
        },
    }

    write_json(output_dir / "summary.json", summary_payload)
    write_json(output_dir / "tables.json", tables_payload)
    write_json(output_dir / "last_run.json", last_run_payload)
    write_json(output_dir / "dashboard_payload.json", dashboard_payload)
    write_json(output_dir / "dashboard_state.json", dashboard_state)

    print(
        "Wrote summary.json, tables.json, last_run.json, dashboard_payload.json, dashboard_state.json "
        f"to {output_dir}"
    )


if __name__ == "__main__":
    main()
