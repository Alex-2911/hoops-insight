#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
generate_dashboard_data.py  (FULL DROP-IN)

Hoops Insight "stats-only" artifact generator.

✅ Guarantees:
- Uses LAST N played games as the only base window (default N=200 via N_WINDOW env).
- Strategy subset + bankroll + risk metrics are computed ONLY from games inside that window.
- local_matched_games are ALWAYS restricted to the last N played games window
  (no over-counting, no under-counting due to mismatched dates).
- “Calibration (Brier)” KPI can no longer be 0.000 when data exists.
- No reliance on any legacy isotonic stats .py files.
- ✅ NEW: Settles bets by combining:
    bet_log_flat_live.csv (placed bets) + combined_* (played results)
  so "Bankroll (2026 YTD)" is computed from actually played outcomes.

Inputs:
- SOURCE_ROOT points to Basketball_prediction/2026 (default: ../Basketball_prediction/2026)
- LGBM_DIR overrides SOURCE_ROOT/output/LightGBM for all LightGBM artifacts.
- combined predictions (played games only):
    - output/LightGBM/Kelly/combined_nba_predictions_iso_YYYY-MM-DD.csv (if newest)
    - output/LightGBM/combined_nba_predictions_acc_YYYY-MM-DD.csv (otherwise)
- local matched games (settled bets exported from notebook):
    - output/LightGBM/local_matched_games_YYYY-MM-DD.csv (latest)
- bet log (placed bets):
    - output/LightGBM/bet_log_flat_live.csv
- strategy params:
    - $STRATEGY_PARAMS_FILE (optional)
    - output/LightGBM/strategy_params.json
    - output/LightGBM/strategy_params.txt

KPI semantics & data sources (contract):
- Window (Last N played games):
  - Source: combined_nba_predictions_* (played games only), last N by date.
  - KPIs: Overall Accuracy, Calibration (Brier/LogLoss), Home Win Rates, Strategy Coverage.
- Strategy table + "Bankroll (Last 200 Games)":
  - Source: local_matched_games_YYYY-MM-DD.csv restricted to window membership.
  - Important: strategy table != YTD.
- "Bankroll (2026 YTD)":
  - Source: bet_log_flat_live.csv (placed bets) settled via combined_*.
  - Independent of strategy_params; only settled rows count.
  - Fallback: windowed_local_matched_fallback (note fields show in UI).
- "Settled Bets (2026)" card/table:
  - Same settled bet_log pipeline as YTD.
  - Metrics: settled_bets, wins, win_rate, profit_eur, roi_pct, avg_odds, avg_stake_eur.
  - Source tracking: source_type, source_file, note.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


DATE_FMT = "%Y-%m-%d"
DEFAULT_WINDOW_SIZE = 200

THRESHOLDS = [
    {"label": "> 0.60", "thresholdType": "gt", "threshold": 0.60},
    {"label": "<= 0.40", "thresholdType": "lt", "threshold": 0.40},
]

DEFAULT_STRATEGY_FILTERS = {
    # Only used when params file AND metrics_snapshot are missing.
    "home_win_rate_min": 0.55,
    "odds_min": 1.5,
    "odds_max": 2.3,
    "prob_min": 0.6,
    "min_ev": -5.0,
}


# ----------------------------
# Source resolution
# ----------------------------
@dataclass
class SourcePaths:
    combined_iso: Optional[Path]
    combined_acc: Optional[Path]
    local_matched_games: Optional[Path]


def _normalize_key(key: str) -> str:
    key = key.strip().lower()
    key = re.sub(r"[\s\-]+", "_", key)
    key = re.sub(r"[^a-z0-9_]", "", key)
    key = re.sub(r"_+", "_", key)
    return key


def _safe_float(val) -> Optional[float]:
    if val is None:
        return None
    s = str(val).strip()
    if s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _safe_date(val) -> Optional[datetime]:
    if val is None:
        return None
    s = str(val).strip()
    if s == "":
        return None
    for fmt in (DATE_FMT, "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


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


def _find_latest_file(path: Path, prefix: str, suffix: str = ".csv") -> Optional[Path]:
    if not path.exists():
        return None
    pattern = re.compile(rf"{re.escape(prefix)}_(\d{{4}}-\d{{2}}-\d{{2}}){re.escape(suffix)}$")
    candidates: List[Tuple[datetime, Path]] = []
    for item in path.iterdir():
        if not item.is_file():
            continue
        m = pattern.match(item.name)
        if not m:
            continue
        dt = _safe_date(m.group(1))
        if dt:
            candidates.append((dt, item))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def _resolve_lightgbm_dir(source_root: Path) -> Path:
    env = os.environ.get("LGBM_DIR")
    if env:
        p = Path(env)
        return p if p.is_absolute() else (source_root / p)
    return source_root / "output" / "LightGBM"


def _resolve_sources(source_root: Path, lightgbm_dir: Path) -> SourcePaths:
    kelly_dir = lightgbm_dir / "Kelly"

    combined_iso = _find_latest_file(kelly_dir, "combined_nba_predictions_iso")
    combined_acc = _find_latest_file(lightgbm_dir, "combined_nba_predictions_acc")

    local_matched_games = _find_latest_file(lightgbm_dir, "local_matched_games")
    if local_matched_games is None:
        fallback = lightgbm_dir / "local_matched_games.csv"
        if fallback.exists():
            local_matched_games = fallback

    return SourcePaths(
        combined_iso=combined_iso,
        combined_acc=combined_acc,
        local_matched_games=local_matched_games,
    )


def _resolve_bet_log_path(source_root: Path, lightgbm_dir: Path) -> Optional[Path]:
    env = os.environ.get("BET_LOG_FILE")
    if env:
        return Path(env)

    candidates = [
        lightgbm_dir / "bet_log_flat_live.csv",
        lightgbm_dir / "bet_log_live.csv",
        lightgbm_dir / "bet_log.csv",
        source_root / "bet_log_flat_live.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _select_combined_path(sources: SourcePaths) -> Path:
    def _file_date(p: Path) -> Optional[datetime]:
        m = re.search(r"(\d{4}-\d{2}-\d{2})\.csv$", p.name)
        return _safe_date(m.group(1)) if m else None

    iso = sources.combined_iso
    acc = sources.combined_acc

    if iso and acc:
        iso_dt = _file_date(iso)
        acc_dt = _file_date(acc)
        if iso_dt and acc_dt:
            return iso if iso_dt >= acc_dt else acc
        return acc
    if acc:
        return acc
    if iso:
        return iso
    raise FileNotFoundError("No combined predictions file found (iso or acc).")


def _read_window_size() -> int:
    raw = os.environ.get("N_WINDOW")
    if raw is None:
        return DEFAULT_WINDOW_SIZE
    try:
        v = int(raw)
    except ValueError:
        return DEFAULT_WINDOW_SIZE
    return v if v > 0 else DEFAULT_WINDOW_SIZE


# ----------------------------
# Strategy params
# ----------------------------
def _parse_strategy_params_text(raw: str) -> Dict[str, float]:
    keys = {
        "home_win_rate_threshold": "home_win_rate_min",
        "odds_min": "odds_min",
        "odds_max": "odds_max",
        "prob_threshold": "prob_min",
        "prob_threshold_used": "prob_min",
        "min_ev": "min_ev",
    }
    values: Dict[str, float] = {}

    for line in raw.splitlines():
        cleaned = line.strip()
        if not cleaned:
            continue

        if "Min EV applied" in cleaned:
            m = re.search(r"Min EV applied\s*=\s*([-0-9.]+)", cleaned)
            if m:
                values["min_ev"] = float(m.group(1))
            continue

        m = re.match(r"([A-Za-z0-9_ ()]+)\s*:\s*([-0-9.]+)", cleaned)
        if not m:
            continue
        key_raw = _normalize_key(m.group(1))
        val = float(m.group(2))
        if key_raw in keys:
            values[keys[key_raw]] = val

    return values


def _load_strategy_params(source_root: Path, lightgbm_dir: Path) -> Tuple[Dict[str, float], str]:
    env_path = os.environ.get("STRATEGY_PARAMS_FILE")
    candidates: List[Path] = []

    if env_path:
        p = Path(env_path)
        if not p.is_absolute():
            p = source_root / env_path
        candidates.append(p)
    else:
        candidates.append(lightgbm_dir / "strategy_params.json")
        candidates.append(lightgbm_dir / "strategy_params.txt")

    for p in candidates:
        if not p.exists():
            continue
        if p.suffix.lower() == ".json":
            data = json.loads(p.read_text(encoding="utf-8"))
            merged = DEFAULT_STRATEGY_FILTERS.copy()
            if "home_win_rate_threshold" in data:
                merged["home_win_rate_min"] = float(data["home_win_rate_threshold"])
            if "odds_min" in data:
                merged["odds_min"] = float(data["odds_min"])
            if "odds_max" in data:
                merged["odds_max"] = float(data["odds_max"])
            if "prob_threshold" in data:
                merged["prob_min"] = float(data["prob_threshold"])
            if "prob_threshold_used" in data:
                merged["prob_min"] = float(data["prob_threshold_used"])
            if "min_ev" in data:
                merged["min_ev"] = float(data["min_ev"])
            return merged, str(p)

        parsed = _parse_strategy_params_text(p.read_text(encoding="utf-8"))
        if parsed:
            merged = DEFAULT_STRATEGY_FILTERS.copy()
            merged.update(parsed)
            return merged, str(p)

    return DEFAULT_STRATEGY_FILTERS.copy(), "default"


def _read_metrics_snapshot_records(path: Path) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    if path.suffix.lower() == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, dict):
                    records.append(item)
        return records

    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)
    return records


def _extract_params_used_from_snapshot(
    records: List[Dict[str, object]],
) -> Tuple[Optional[Dict[str, float]], Optional[str]]:
    params: Dict[str, float] = {}
    params_used_type: Optional[str] = None

    for rec in records:
        section = str(rec.get("section") or "").strip().lower()
        metric = str(rec.get("metric") or "").strip().lower()
        value = rec.get("value")

        if section == "meta" and metric == "params_source":
            src = str(value or "").strip().upper()
            if "LOCAL" in src:
                params_used_type = "LOCAL"
            elif "GLOBAL" in src:
                params_used_type = "GLOBAL"
            continue

        if section != "filter_params":
            continue

        if metric in {"odds_min", "odds_max", "prob_threshold", "home_win_rate_threshold"}:
            v = _safe_float(value)
            if v is not None:
                params[metric] = float(v)
        elif metric in {"min_ev", "ev_min"}:
            v = _safe_float(value)
            if v is not None:
                params["min_ev"] = float(v)

    required = {"odds_min", "odds_max", "prob_threshold", "home_win_rate_threshold"}
    if not required.issubset(params.keys()):
        return None, params_used_type
    return params, params_used_type


def _load_params_used(
    source_root: Path,
    repo_root: Path,
    lightgbm_dir: Path,
) -> Tuple[Optional[Dict[str, float]], Optional[str], Optional[str]]:
    candidates = [
        repo_root.parent / "1. NBA Script" / "2026" / "metrics_snapshot.json",
        repo_root.parent / "1. NBA Script" / "2026" / "metrics_snapshot.csv",
        source_root / "metrics_snapshot.json",
        source_root / "metrics_snapshot.csv",
        lightgbm_dir / "metrics_snapshot.json",
        lightgbm_dir / "metrics_snapshot.csv",
    ]

    for path in candidates:
        if not path.exists():
            continue
        records = _read_metrics_snapshot_records(path)
        params, params_used_type = _extract_params_used_from_snapshot(records)
        if params:
            return params, str(path), params_used_type

    return None, None, None


# ----------------------------
# Played games loading (combined_*.csv)
# ----------------------------
def load_played_games(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
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

        prob_raw = _safe_float(
            row.get("pred_home_win_proba")
            or row.get("home_team_prob")
            or row.get("home_team_probability")
        )
        prob_iso = _safe_float(row.get("iso_proba_home_win"))
        prob_used = _safe_float(
            row.get("prob_used") or row.get("probability_used") or row.get("prob_used_final")
        )

        odds = _safe_float(
            row.get("closing_home_odds")
            or row.get("odds_1")
            or row.get("odds_home")
            or row.get("odds")
        )

        home_win_rate = _safe_float(row.get("home_win_rate") or row.get("home_win_pct"))

        ev_per_100 = _safe_float(
            row.get("ev_per_100")
            or row.get("ev_100")
            or row.get("ev__per_100")
            or row.get("ev_€_per_100")
        )

        rows.append(
            {
                "date": game_date,
                "home_team": home_team,
                "away_team": away_team,
                "home_team_won": home_team_won,
                "prob_raw": prob_raw,
                "prob_iso": prob_iso,
                "prob_used": prob_used,
                "odds_home": odds,
                "home_win_rate": home_win_rate,
                "ev_per_100": ev_per_100,
            }
        )
    return rows


def _require_played_games(path: Path) -> List[Dict[str, object]]:
    rows = load_played_games(path)
    if not rows:
        raise RuntimeError(
            f"No played games found in {path}. "
            "Expected rows with non-empty result/result_raw and valid game_date/date."
        )
    return rows


def build_windowed_rows(rows: List[Dict[str, object]], window_size: int) -> List[Dict[str, object]]:
    rows_sorted = sorted(rows, key=lambda r: r["date"])
    return rows_sorted[-window_size:] if window_size > 0 else rows_sorted


def _latest_played_date(rows: List[Dict[str, object]]) -> datetime:
    return max(r["date"] for r in rows)


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


# ----------------------------
# Calibration + metrics
# ----------------------------
def _clamp_prob(p: float, eps: float = 1e-6) -> float:
    return min(max(p, eps), 1.0 - eps)


def _compute_brier_pairs(pairs: List[Tuple[int, float]]) -> Optional[float]:
    if not pairs:
        return None
    return sum((p - y) ** 2 for y, p in pairs) / len(pairs)


def _compute_logloss_pairs(pairs: List[Tuple[int, float]]) -> Optional[float]:
    if not pairs:
        return None
    total = 0.0
    for y, p in pairs:
        p = _clamp_prob(p)
        total += -(y * math.log(p) + (1 - y) * math.log(1.0 - p))
    return total / len(pairs)


def _compute_ece(pairs: List[Tuple[int, float]], n_bins: int = 10) -> Tuple[Optional[float], List[Dict[str, object]]]:
    if not pairs or n_bins <= 0:
        return None, []

    totals = [0] * n_bins
    sum_pred = [0.0] * n_bins
    sum_out = [0.0] * n_bins

    for y, p in pairs:
        p = _clamp_prob(p)
        idx = min(int(p * n_bins), n_bins - 1)
        totals[idx] += 1
        sum_pred[idx] += p
        sum_out[idx] += y

    total = sum(totals)
    if total == 0:
        return None, []

    ece = 0.0
    bins_out: List[Dict[str, object]] = []
    for i in range(n_bins):
        n = totals[i]
        if n == 0:
            continue
        avg_pred = sum_pred[i] / n
        avg_out = sum_out[i] / n
        weight = n / total
        ece += abs(avg_pred - avg_out) * weight
        bins_out.append(
            {
                "bin_center": (i + 0.5) / n_bins,
                "n": n,
                "avg_pred": avg_pred,
                "avg_outcome": avg_out,
            }
        )
    return ece, bins_out


def _fit_calibration_line(pairs: List[Tuple[int, float]]) -> Tuple[Optional[float], Optional[float]]:
    if len(pairs) < 2:
        return None, None
    ys = [y for y, _ in pairs]
    ps = [p for _, p in pairs]
    mean_y = sum(ys) / len(ys)
    mean_p = sum(ps) / len(ps)
    var_p = sum((p - mean_p) ** 2 for p in ps)
    if var_p == 0:
        return None, None
    cov = sum((p - mean_p) * (y - mean_y) for y, p in pairs)
    slope = cov / var_p
    intercept = mean_y - slope * mean_p
    return slope, intercept


def build_calibration_quality(rows: List[Dict[str, object]], window_size: int) -> Dict[str, object]:
    pairs_raw = [(int(r["home_team_won"]), float(r["prob_raw"])) for r in rows if r.get("prob_raw") is not None]
    pairs_iso = [(int(r["home_team_won"]), float(r["prob_iso"])) for r in rows if r.get("prob_iso") is not None]

    brier_before = _compute_brier_pairs(pairs_raw)
    brier_after = _compute_brier_pairs(pairs_iso) if pairs_iso else brier_before

    logloss_before = _compute_logloss_pairs(pairs_raw)
    logloss_after = _compute_logloss_pairs(pairs_iso) if pairs_iso else logloss_before

    ece_before, bins_before = _compute_ece(pairs_raw, n_bins=10)
    if pairs_iso:
        ece_after, bins_after = _compute_ece(pairs_iso, n_bins=10)
    else:
        ece_after, bins_after = ece_before, bins_before

    slope_before, intercept_before = _fit_calibration_line(pairs_raw)
    if pairs_iso:
        slope_after, intercept_after = _fit_calibration_line(pairs_iso)
    else:
        slope_after, intercept_after = slope_before, intercept_before

    avg_pred_before = (sum(p for _, p in pairs_raw) / len(pairs_raw)) if pairs_raw else None
    avg_pred_after = (sum(p for _, p in pairs_iso) / len(pairs_iso)) if pairs_iso else avg_pred_before

    base_rate = (sum(int(r["home_team_won"]) for r in rows) / len(rows)) if rows else None
    as_of = max(r["date"] for r in rows).strftime(DATE_FMT) if rows else "—"

    return {
        "window_size": window_size,
        "as_of_date": as_of,
        "fitted_games": len(pairs_iso) if pairs_iso else len(pairs_raw),
        "brier_before": brier_before,
        "brier_after": brier_after,
        "logloss_before": logloss_before,
        "logloss_after": logloss_after,
        "ece_before": ece_before,
        "ece_after": ece_after,
        "calibration_slope_before": slope_before,
        "calibration_slope_after": slope_after,
        "calibration_intercept_before": intercept_before,
        "calibration_intercept_after": intercept_after,
        "avg_pred_before": avg_pred_before,
        "avg_pred_after": avg_pred_after,
        "base_rate": base_rate,
        "n_bins": 10,
        "binning_method": "equal_width",
        "reliability_bins_before": bins_before,
        "reliability_bins_after": bins_after,
    }


def build_accuracy_thresholds(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for spec in THRESHOLDS:
        thr = spec["threshold"]
        passed: List[Dict[str, object]] = []
        for r in rows:
            p = r.get("prob_iso") if r.get("prob_iso") is not None else r.get("prob_raw")
            if p is None:
                continue
            p = float(p)
            if spec["thresholdType"] == "gt" and p > thr:
                passed.append(r)
            elif spec["thresholdType"] == "lt" and p <= thr:
                passed.append(r)
        total = len(passed)
        correct = sum(int(r["home_team_won"]) for r in passed)
        out.append(
            {
                "label": spec["label"],
                "thresholdType": spec["thresholdType"],
                "threshold": thr,
                "accuracy": _safe_div(correct, total),
                "sampleSize": total,
            }
        )
    return out


def build_historical_stats(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    per_day = defaultdict(lambda: {"total": 0, "correct": 0})
    for r in rows:
        d = r["date"].strftime(DATE_FMT)
        per_day[d]["total"] += 1
        per_day[d]["correct"] += int(r["home_team_won"])
    out = []
    for d in sorted(per_day.keys()):
        total = per_day[d]["total"]
        correct = per_day[d]["correct"]
        out.append({"date": d, "accuracy": _safe_div(correct, total), "totalGames": total, "correctGames": correct})
    return out


# ----------------------------
# Home win-rate table
# ----------------------------
def build_home_win_rates_window(rows: List[Dict[str, object]], threshold: float = 0.50) -> List[Dict[str, object]]:
    per_team = defaultdict(lambda: {"total_games": 0, "home_games": 0, "home_wins": 0})
    for r in rows:
        home = str(r["home_team"])
        away = str(r["away_team"])
        per_team[home]["total_games"] += 1
        per_team[home]["home_games"] += 1
        per_team[home]["home_wins"] += int(r["home_team_won"])
        per_team[away]["total_games"] += 1

    out: List[Dict[str, object]] = []
    for team, s in per_team.items():
        hw = _safe_div(s["home_wins"], s["home_games"])
        out.append(
            {
                "team": team,
                "totalGames": s["total_games"],
                "totalHomeGames": s["home_games"],
                "homeWins": s["home_wins"],
                "homeWinRate": hw,
            }
        )
    out = [r for r in out if r["homeWinRate"] > threshold]
    out.sort(key=lambda x: (x["homeWinRate"], x["totalHomeGames"], x["homeWins"]), reverse=True)
    return out


# ----------------------------
# Strategy subset (computed on windowed played rows)
# ----------------------------
def _compute_ev_per_100(prob: float, odds: float) -> float:
    return (prob * (odds - 1.0) - (1.0 - prob)) * 100.0


def _resolve_prob_used(row: Dict[str, object]) -> Optional[float]:
    for k in ("prob_used", "prob_iso", "prob_raw"):
        v = row.get(k)
        if v is not None:
            return float(v)
    return None


def _compute_home_win_rate_map(rows: List[Dict[str, object]]) -> Dict[str, float]:
    per_team = defaultdict(lambda: {"home_games": 0, "home_wins": 0})
    for r in rows:
        team = str(r["home_team"])
        per_team[team]["home_games"] += 1
        per_team[team]["home_wins"] += int(r["home_team_won"])
    return {t: _safe_div(s["home_wins"], s["home_games"]) for t, s in per_team.items() if s["home_games"]}


def build_strategy_subset(rows: List[Dict[str, object]], params: Dict[str, float]) -> Tuple[List[Dict[str, object]], Dict[str, int]]:
    home_wr_map = _compute_home_win_rate_map(rows)

    # fill missing fields (within window only)
    for r in rows:
        if r.get("home_win_rate") is None:
            r["home_win_rate"] = home_wr_map.get(str(r["home_team"]))
        if r.get("prob_used") is None:
            r["prob_used"] = _resolve_prob_used(r)
        if r.get("ev_per_100") is None and r.get("prob_used") is not None and r.get("odds_home") is not None:
            r["ev_per_100"] = _compute_ev_per_100(float(r["prob_used"]), float(r["odds_home"]))

    total_games = len(rows)

    s1 = [r for r in rows if r.get("home_win_rate") is not None and float(r["home_win_rate"]) >= params["home_win_rate_min"]]
    s2 = [r for r in s1 if r.get("odds_home") is not None and params["odds_min"] <= float(r["odds_home"]) <= params["odds_max"]]
    s3 = [r for r in s2 if r.get("prob_used") is not None and float(r["prob_used"]) >= params["prob_min"]]
    s4 = [r for r in s3 if r.get("ev_per_100") is not None and float(r["ev_per_100"]) > params["min_ev"]]

    coverage = {
        "total_games": total_games,
        "passed_home_win_rate": len(s1),
        "passed_odds_range": len(s2),
        "passed_prob_threshold": len(s3),
        "passed_ev_threshold": len(s4),
    }
    return s4, coverage


def build_strategy_filter_stats(rows: List[Dict[str, object]], params: Dict[str, float], params_source: str) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    subset, cov = build_strategy_subset(rows, params)
    matched = len(subset)
    wins = sum(int(r["home_team_won"]) for r in subset)
    acc = _safe_div(wins, matched)

    stats = {
        "total_games": cov["total_games"],
        "params_source": params_source,
        "home_win_rate_min": params["home_win_rate_min"],
        "odds_min": params["odds_min"],
        "odds_max": params["odds_max"],
        "prob_min": params["prob_min"],
        "min_ev": params["min_ev"],
        "passed_home_win_rate": cov["passed_home_win_rate"],
        "passed_odds_range": cov["passed_odds_range"],
        "passed_prob_threshold": cov["passed_prob_threshold"],
        "passed_ev_threshold": cov["passed_ev_threshold"],
        "matched_games_count": matched,
        "matched_games_accuracy": acc,
    }
    return subset, stats


# ----------------------------
# local_matched_games loading + restriction to window
# ----------------------------
def _row_key_from_local(date_str: str, home_team: str, away_team: str) -> Optional[Tuple[str, str, str]]:
    dt = _safe_date(date_str)
    if dt is None:
        return None
    return (dt.strftime(DATE_FMT), home_team.strip().upper(), away_team.strip().upper())


def _window_key_set(windowed_rows: List[Dict[str, object]]) -> set:
    s = set()
    for r in windowed_rows:
        dt: datetime = r["date"]
        home = str(r["home_team"]).strip().upper()
        away = str(r["away_team"]).strip().upper()
        s.add((dt.strftime(DATE_FMT), home, away))
    return s


def load_local_matched_games(path: Path) -> List[Dict[str, object]]:
    """
    Reads local_matched_games_*.csv exported by the notebook.

    Expected columns (flexible):
      date, home_team, away_team, home_win_rate, prob_iso, prob_used, odds_1,
      EV_€_per_100 / ev_per_100, win / won, pnl / profit_eur
    """
    rows: List[Dict[str, object]] = []
    for row in _read_csv_normalized(path):
        date_raw = row.get("date")
        dt = _safe_date(date_raw)
        if dt is None:
            continue

        home_team = (row.get("home_team") or "").strip()
        away_team = (row.get("away_team") or "").strip()

        win_val = row.get("win") or row.get("won")
        pnl_val = row.get("pnl") or row.get("pnl_flat") or row.get("profit_eur")

        ev = _safe_float(row.get("ev_per_100") or row.get("ev__per_100") or row.get("ev_€_per_100"))

        win_num = _safe_float(win_val)
        pnl_num = _safe_float(pnl_val)

        rows.append(
            {
                "date": dt.strftime(DATE_FMT),
                "home_team": home_team,
                "away_team": away_team,
                "home_win_rate": _safe_float(row.get("home_win_rate")),
                "prob_iso": _safe_float(row.get("prob_iso")),
                "prob_used": _safe_float(row.get("prob_used")),
                "odds_1": _safe_float(row.get("odds_1")),
                "ev_per_100": ev,
                "win": int(win_num) if win_num is not None else None,
                "pnl": pnl_num,
            }
        )

    # keep only settled
    rows = [r for r in rows if r.get("win") is not None and r.get("pnl") is not None]

    # dedupe by (date, home, away) keep row with larger |pnl| (safety)
    deduped: Dict[Tuple[str, str, str], Dict[str, object]] = {}
    for r in rows:
        k = (r["date"], str(r["home_team"]).strip().upper(), str(r["away_team"]).strip().upper())
        if k not in deduped:
            deduped[k] = r
        else:
            if abs(float(r.get("pnl") or 0.0)) > abs(float(deduped[k].get("pnl") or 0.0)):
                deduped[k] = r

    out = list(deduped.values())
    out.sort(key=lambda x: x["date"])
    return out


def restrict_local_matched_to_window(
    local_rows: List[Dict[str, object]],
    windowed_rows: List[Dict[str, object]],
    window_start: datetime,
    window_end: datetime,
) -> List[Dict[str, object]]:
    """
    Enforces:
    - local_matched must be inside the same last-N played-games window.
    We filter both by date range AND by exact (date, home, away) membership in window set.
    """
    allowed = _window_key_set(windowed_rows)

    filtered: List[Dict[str, object]] = []
    for r in local_rows:
        dt = _safe_date(r.get("date"))
        if dt is None:
            continue
        if not (window_start <= dt <= window_end):
            continue
        k = _row_key_from_local(r["date"], r["home_team"], r["away_team"])
        if k is None:
            continue
        if k in allowed:
            filtered.append(r)

    filtered.sort(key=lambda x: x["date"])
    return filtered


# ----------------------------
# Bet log -> settlement via combined_*
# ----------------------------
def load_bet_log_flat_placed(path: Path) -> List[Dict[str, object]]:
    """
    Reads bet_log_flat_live.csv as PLACED bets (no settlement required here).
    Dedupes by (date, home_team, away_team).
    ✅ If duplicates exist, keep the first row (file order).
    Expected columns (flexible):
      date, home_team, away_team, stake_flat, odds_1
    """
    parsed: List[Dict[str, object]] = []

    for row in _read_csv_normalized(path):
        date_raw = row.get("date") or row.get("game_date") or row.get("event_date")
        dt = _safe_date(date_raw)
        if dt is None:
            continue

        home_team = (row.get("home_team") or row.get("home") or "").strip()
        away_team = (row.get("away_team") or row.get("away") or "").strip()
        if not home_team or not away_team:
            continue

        stake = _safe_float(row.get("stake_flat") or row.get("stake_eur") or row.get("stake"))
        odds = _safe_float(row.get("odds_1") or row.get("odds") or row.get("odds_home"))
        if stake is None or odds is None:
            continue

        parsed.append(
            {
                "date": dt.strftime(DATE_FMT),
                "home_team": home_team,
                "away_team": away_team,
                "stake": float(stake),
                "odds": float(odds),
            }
        )

    # ✅ DEDUPE by (date, home, away) => keep first row
    deduped: Dict[Tuple[str, str, str], Dict[str, object]] = {}
    for r in parsed:
        k = (r["date"], str(r["home_team"]).strip().upper(), str(r["away_team"]).strip().upper())
        if k not in deduped:
            deduped[k] = r

    out = list(deduped.values())
    out.sort(key=lambda x: x["date"])
    return out


def build_played_result_lookup(played_rows: List[Dict[str, object]]) -> Dict[Tuple[str, str, str], int]:
    """
    (date, HOME, AWAY) -> home_team_won (0/1)
    """
    lookup: Dict[Tuple[str, str, str], int] = {}
    for r in played_rows:
        dt: datetime = r["date"]
        d = dt.strftime(DATE_FMT)
        home = str(r["home_team"]).strip().upper()
        away = str(r["away_team"]).strip().upper()
        lookup[(d, home, away)] = int(r["home_team_won"])
    return lookup


def settle_flat_bets_from_played(
    placed_bets: List[Dict[str, object]],
    played_lookup: Dict[Tuple[str, str, str], int],
) -> List[Dict[str, object]]:
    """
    placed bets -> settled bets, by matching (date, home, away) to combined_* played results.
    Returns only settled rows (win + pnl).
    """
    settled: List[Dict[str, object]] = []

    for b in placed_bets:
        k = (b["date"], str(b["home_team"]).strip().upper(), str(b["away_team"]).strip().upper())
        home_won = played_lookup.get(k)
        if home_won is None:
            continue  # game not found / not played yet / mismatch

        stake = float(b["stake"])
        odds = float(b["odds"])
        win = int(home_won)
        pnl = stake * (odds - 1.0) if win == 1 else -stake

        settled.append(
            {
                "date": b["date"],
                "home_team": b["home_team"],
                "away_team": b["away_team"],
                "game_key": None,
                "stake": stake,
                "odds": odds,
                "prob_used": None,
                "win": win,
                "pnl": pnl,
            }
        )

    settled.sort(key=lambda x: x["date"])
    return settled


# ----------------------------
# Bankroll / risk computed from settled rows
# ----------------------------
def compute_max_drawdown(history: List[Dict[str, object]]) -> Tuple[Optional[float], Optional[float]]:
    if not history:
        return None, None
    balances = [h["balance"] for h in history if h.get("balance") is not None]
    if not balances:
        return None, None

    peak = balances[0]
    max_dd = 0.0
    for b in balances:
        if b > peak:
            peak = b
        dd = peak - b
        if dd > max_dd:
            max_dd = dd
    max_dd_pct = (_safe_div(max_dd, peak) * 100.0) if peak else None
    return max_dd, max_dd_pct


def compute_bankroll_block(local_rows: List[Dict[str, object]], starting: float = 1000.0) -> Dict[str, object]:
    if not local_rows:
        return {
            "bankroll_end": None,
            "pnl_sum": None,
            "n_trades": 0,
            "sharpe": None,
            "max_drawdown_eur": None,
            "max_drawdown_pct": None,
            "history": [],
        }

    equity = starting
    pnl_values: List[float] = []
    history: List[Dict[str, object]] = []

    for r in sorted(local_rows, key=lambda x: x["date"]):
        pnl = r.get("pnl")
        if pnl is None:
            continue
        pnl = float(pnl)
        pnl_values.append(pnl)
        equity += pnl
        history.append({"date": r["date"], "balance": equity, "betsPlaced": 1, "profit": pnl})

    if not pnl_values:
        return {
            "bankroll_end": None,
            "pnl_sum": None,
            "n_trades": 0,
            "sharpe": None,
            "max_drawdown_eur": None,
            "max_drawdown_pct": None,
            "history": [],
        }

    mean = sum(pnl_values) / len(pnl_values)
    var = sum((p - mean) ** 2 for p in pnl_values) / len(pnl_values)
    std = math.sqrt(var)
    sharpe = (mean / std * math.sqrt(len(pnl_values))) if std != 0 else None

    max_dd_eur, max_dd_pct = compute_max_drawdown(history)

    return {
        "bankroll_end": equity,
        "pnl_sum": sum(pnl_values),
        "n_trades": len(pnl_values),
        "sharpe": sharpe,
        "max_drawdown_eur": max_dd_eur,
        "max_drawdown_pct": max_dd_pct,
        "history": history,
    }


# ----------------------------
# JSON write
# ----------------------------
def _serialize(obj):
    if isinstance(obj, datetime):
        return obj.strftime(DATE_FMT)
    return obj


def _format_fixed(val: float, digits: int) -> str:
    return f"{val:.{digits}f}"


def _format_trimmed(val: float, digits: int = 2) -> str:
    s = f"{val:.{digits}f}"
    s = s.rstrip("0").rstrip(".")
    return s if s else "0"


def _format_unicode_minus(val: float, digits: int = 2, trim: bool = True) -> str:
    s = _format_trimmed(val, digits) if trim else _format_fixed(val, digits)
    return f"−{s[1:]}" if s.startswith("-") else s


def build_active_filters_human(
    active_filters: Dict[str, object],
) -> str:
    parts = [
        f"HW ≥ {_format_fixed(float(active_filters['home_win_rate_min']), 2)}",
        f"odds {_format_fixed(float(active_filters['odds_min']), 2)}–{_format_fixed(float(active_filters['odds_max']), 2)}",
        f"p ≥ {_format_fixed(float(active_filters['prob_min']), 2)}",
        f"EV > {_format_unicode_minus(float(active_filters['min_ev']), 1, trim=True)}",
        f"window {active_filters['window_size']} ({active_filters['window_start']} → {active_filters['window_end']})",
    ]
    return " | ".join(parts)


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, default=_serialize, ensure_ascii=False, indent=2)


# ----------------------------
# MAIN
# ----------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    default_root = repo_root.parent / "Basketball_prediction" / "2026"

    env_root = os.environ.get("SOURCE_ROOT")
    if args.source_root:
        source_root = Path(args.source_root)
    elif env_root:
        source_root = Path(env_root)
    else:
        source_root = default_root

    output_dir = Path(args.output_dir) if args.output_dir else repo_root / "public" / "data"

    lightgbm_dir = _resolve_lightgbm_dir(source_root)
    sources = _resolve_sources(source_root, lightgbm_dir)
    combined_path = _select_combined_path(sources)

    played_rows = _require_played_games(combined_path)

    # Build windowed rows from played games (last N).
    window_size = _read_window_size()
    windowed_rows = build_windowed_rows(played_rows, window_size)
    if not windowed_rows:
        raise RuntimeError("No played games found after applying the window filter.")

    window_start = min(r["date"] for r in windowed_rows)
    window_end = max(r["date"] for r in windowed_rows)

    latest_played_date = _latest_played_date(played_rows)
    as_of_date = latest_played_date.strftime(DATE_FMT)

    # Load/choose params_used (fair-selected from metrics_snapshot when available).
    strategy_params, strategy_params_source = _load_strategy_params(source_root, lightgbm_dir)
    params_used_raw, params_used_source, params_used_type = _load_params_used(
        source_root,
        repo_root,
        lightgbm_dir,
    )

    if params_used_raw:
        params_used = params_used_raw
        params_source = params_used_source or strategy_params_source
    else:
        params_used = {
            "home_win_rate_threshold": strategy_params["home_win_rate_min"],
            "odds_min": strategy_params["odds_min"],
            "odds_max": strategy_params["odds_max"],
            "prob_threshold": strategy_params["prob_min"],
            "min_ev": strategy_params["min_ev"],
        }
        params_source = strategy_params_source

    # --- core tables ---
    historical_stats = build_historical_stats(windowed_rows)
    accuracy_thresholds = build_accuracy_thresholds(windowed_rows)

    calibration_quality = build_calibration_quality(windowed_rows, window_size)

    # ✅ KPI fix: never allow Brier to become 0 if we have played data
    brier_raw = calibration_quality.get("brier_before")
    brier_iso = calibration_quality.get("brier_after")
    if brier_iso is None:
        brier_iso = brier_raw
    if brier_raw is None:
        brier_raw = 0.0
    if brier_iso is None:
        brier_iso = 0.0

    calibration_metrics = {
        "asOfDate": as_of_date,
        "brierBefore": float(brier_raw),
        "brierAfter": float(brier_iso),
        "logLossBefore": float(calibration_quality.get("logloss_before") or 0.0),
        "logLossAfter": float(calibration_quality.get("logloss_after") or 0.0),
        "fittedGames": int(calibration_quality.get("fitted_games") or 0),
    }

    home_win_rate_threshold = 0.50
    home_win_rates = build_home_win_rates_window(windowed_rows, home_win_rate_threshold)

    # Descriptive coverage (still computed strictly on windowed_rows)
    params_used_filters = {
        "home_win_rate_min": float(params_used["home_win_rate_threshold"]),
        "odds_min": float(params_used["odds_min"]),
        "odds_max": float(params_used["odds_max"]),
        "prob_min": float(params_used["prob_threshold"]),
        "min_ev": float(params_used.get("min_ev", strategy_params["min_ev"])),
    }

    _, strategy_filter_stats = build_strategy_filter_stats(
        windowed_rows,
        params_used_filters,
        params_source,
    )

    # -------------------------
    # LOCAL MATCHED (authoritative for strategy table + "Bankroll (Last 200 Games)")
    # STRICTLY windowed to last N played games
    # -------------------------
    local_matched_rows: List[Dict[str, object]] = []
    local_source = str(sources.local_matched_games) if sources.local_matched_games else "missing"

    if sources.local_matched_games and sources.local_matched_games.exists():
        local_matched_rows = load_local_matched_games(sources.local_matched_games)
        local_matched_rows = restrict_local_matched_to_window(
            local_matched_rows,
            windowed_rows,
            window_start,
            window_end,
        )

    strategy_subset_rows = local_matched_rows  # authoritative for strategy table + bankroll last 200

    subset_matches = len(strategy_subset_rows)
    subset_wins = sum(1 for r in strategy_subset_rows if r.get("win") == 1)
    subset_acc = _safe_div(subset_wins, subset_matches)

    # Force filter stats to reflect the true matched subset count
    strategy_filter_stats["matched_games_count"] = subset_matches
    strategy_filter_stats["matched_games_accuracy"] = subset_acc
    strategy_filter_stats["passed_ev_threshold"] = subset_matches

    bankroll_window_block = compute_bankroll_block(strategy_subset_rows, starting=1000.0)

    # -------------------------
    # YTD / Settled (bet_log placed -> settled via combined_* results)
    # IMPORTANT: independent of strategy_params; reflects actual placed bets.
    # -------------------------
    played_lookup = build_played_result_lookup(played_rows)

    bet_log_path = _resolve_bet_log_path(source_root, lightgbm_dir)
    bet_log_rows: List[Dict[str, object]] = []
    settled_from_betlog: List[Dict[str, object]] = []
    ytd_note = None
    ytd_source_type = "bet_log_flat_live_settled_via_combined"
    bet_log_source = str(bet_log_path) if bet_log_path else "missing"

    if bet_log_path and bet_log_path.exists():
        placed_bets = load_bet_log_flat_placed(bet_log_path)
        bet_log_rows = settle_flat_bets_from_played(placed_bets, played_lookup)
        settled_from_betlog = bet_log_rows

        if not placed_bets:
            ytd_note = "bet_log_flat_live empty; fallback used."
            ytd_source_type = "windowed_local_matched_fallback"
        elif not bet_log_rows:
            ytd_note = "bet_log_flat_live has placed bets, but none matched played results in combined_*; fallback used."
            ytd_source_type = "windowed_local_matched_fallback"
    else:
        ytd_note = "bet_log_flat_live missing; fallback used."
        ytd_source_type = "windowed_local_matched_fallback"

    if bet_log_rows:
        ytd_rows = [r for r in bet_log_rows if str(r.get("date", "")).startswith("2026-")]
        if not ytd_rows:
            ytd_note = "bet_log_flat_live has no settled 2026 rows (after matching); fallback used."
            ytd_source_type = "windowed_local_matched_fallback"
            ytd_rows = [r for r in strategy_subset_rows if str(r.get("date", "")).startswith("2026-")]

        bankroll_ytd_block = compute_bankroll_block(ytd_rows, starting=1000.0) if ytd_rows else {
            "bankroll_end": None,
            "pnl_sum": None,
            "n_trades": 0,
            "sharpe": None,
            "max_drawdown_eur": None,
            "max_drawdown_pct": None,
            "history": [],
        }
    else:
        ytd_rows = [r for r in strategy_subset_rows if str(r.get("date", "")).startswith("2026-")]
        bankroll_ytd_block = compute_bankroll_block(ytd_rows, starting=1000.0) if ytd_rows else {
            "bankroll_end": None,
            "pnl_sum": None,
            "n_trades": 0,
            "sharpe": None,
            "max_drawdown_eur": None,
            "max_drawdown_pct": None,
            "history": [],
        }

    # 2026 Settled Bets Overview (from bet_log settled via combined)
    # -------------------------
    ytd_settled_count = 0
    ytd_settled_wins = 0
    ytd_settled_profit = None
    ytd_settled_roi_pct = None
    ytd_settled_avg_odds = None
    ytd_settled_avg_stake = None
    ytd_settled_rows = []
    ytd_settled_summary = None

    settled_source_type = "bet_log_flat_live_settled_via_combined"
    settled_source_file = bet_log_source
    settled_note = None
    settled_rows_source: List[Dict[str, object]] = []

    if settled_from_betlog:
        settled_rows_source = settled_from_betlog
    elif ytd_source_type == "windowed_local_matched_fallback":
        settled_rows_source = strategy_subset_rows
        settled_source_type = "windowed_local_matched_fallback"
        settled_source_file = local_source
        settled_note = ytd_note

    if settled_rows_source:
        normalized_rows: List[Dict[str, object]] = []
        for r in settled_rows_source:
            date_val = r.get("date")
            home_val = r.get("home_team")
            away_val = r.get("away_team")
            if not date_val or not home_val or not away_val:
                continue

            stake_val = _safe_float(r.get("stake"))
            odds_val = _safe_float(r.get("odds") or r.get("odds_1"))
            if stake_val is None or odds_val is None:
                continue

            win_val = _safe_float(r.get("win"))
            pnl_val = _safe_float(r.get("pnl"))
            if pnl_val is None and win_val is not None:
                pnl_val = stake_val * (odds_val - 1.0) if int(win_val) == 1 else -stake_val

            normalized_rows.append(
                {
                    "date": date_val,
                    "home_team": home_val,
                    "away_team": away_val,
                    "stake": float(stake_val),
                    "odds": float(odds_val),
                    "win": int(win_val) if win_val is not None else None,
                    "pnl": pnl_val,
                }
            )

        ytd_settled_rows = [r for r in normalized_rows if str(r.get("date", "")).startswith("2026-")]
        ytd_settled_rows.sort(key=lambda x: x.get("date") or "", reverse=True)
        ytd_settled_count = len(ytd_settled_rows)
        ytd_settled_wins = sum(1 for r in ytd_settled_rows if int(r.get("win") or 0) == 1)

        pnl_vals = [float(r["pnl"]) for r in ytd_settled_rows if r.get("pnl") is not None]
        stake_vals = [float(r["stake"]) for r in ytd_settled_rows if r.get("stake") is not None]
        odds_vals = [float(r["odds"]) for r in ytd_settled_rows if r.get("odds") is not None]

        if pnl_vals:
            ytd_settled_profit = float(sum(pnl_vals))

        if stake_vals:
            ytd_settled_avg_stake = float(sum(stake_vals) / len(stake_vals))

        if odds_vals:
            ytd_settled_avg_odds = float(sum(odds_vals) / len(odds_vals))

        total_staked_ytd = float(sum(stake_vals)) if stake_vals else 0.0
        if total_staked_ytd > 0 and ytd_settled_profit is not None:
            ytd_settled_roi_pct = (ytd_settled_profit / total_staked_ytd) * 100.0

        if ytd_settled_count:
            ytd_settled_summary = {
                "count": ytd_settled_count,
                "wins": ytd_settled_wins,
                "win_rate": _safe_div(ytd_settled_wins, ytd_settled_count),
                "pnl_sum": ytd_settled_profit,
                "roi_pct": ytd_settled_roi_pct,
                "avg_odds": ytd_settled_avg_odds,
                "avg_stake": ytd_settled_avg_stake,
            }

    ytd_settled_overview = {
        "year": 2026,
        "source_type": settled_source_type,
        "source_file": settled_source_file,
        "note": settled_note,
        "settled_bets": ytd_settled_count,
        "wins": ytd_settled_wins,
        "win_rate": _safe_div(ytd_settled_wins, ytd_settled_count) if ytd_settled_count else None,
        "profit_eur": ytd_settled_profit,
        "roi_pct": ytd_settled_roi_pct,
        "avg_odds": ytd_settled_avg_odds,
        "avg_stake_eur": ytd_settled_avg_stake,
    }



    # Overall accuracy (window)
    total_games = len(windowed_rows)
    total_correct = sum(int(r["home_team_won"]) for r in windowed_rows)
    overall_accuracy = _safe_div(total_correct, total_games)

    last_run = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    window_start_str = window_start.strftime(DATE_FMT)
    window_end_str = window_end.strftime(DATE_FMT)

    # Active Filters are built from params_used (metrics_snapshot), not defaults.
    active_filters = {
        "window_size": window_size,
        "window_start": window_start_str,
        "window_end": window_end_str,
        "params_source": params_source,
        "home_win_rate_min": float(params_used["home_win_rate_threshold"]),
        "odds_min": float(params_used["odds_min"]),
        "odds_max": float(params_used["odds_max"]),
        "prob_min": float(params_used["prob_threshold"]),
        "min_ev": float(params_used.get("min_ev", strategy_params["min_ev"])),
    }
    # Human-readable filters always derived from active_filters.
    active_filters_human = build_active_filters_human(active_filters)

    bankroll_last_200_eur = None
    net_pl_last_200_eur = None
    if bankroll_window_block["pnl_sum"] is not None:
        net_pl_last_200_eur = float(bankroll_window_block["pnl_sum"])
        bankroll_last_200_eur = 1000.0 + net_pl_last_200_eur

    bankroll_2026_ytd_eur = None
    net_pl_2026_ytd_eur = None
    if bankroll_ytd_block["pnl_sum"] is not None:
        net_pl_2026_ytd_eur = float(bankroll_ytd_block["pnl_sum"])
        bankroll_2026_ytd_eur = 1000.0 + net_pl_2026_ytd_eur

    # ROI for strategy subset (local matched)
    total_staked = subset_matches * 100.0
    roi_pct = (_safe_div(float(bankroll_window_block["pnl_sum"] or 0.0), total_staked) * 100.0) if subset_matches else None

    # Avg EV from table rows (if present)
    ev_vals = [float(r["ev_per_100"]) for r in strategy_subset_rows if r.get("ev_per_100") is not None]
    avg_ev = (_safe_div(sum(ev_vals), len(ev_vals)) if ev_vals else None)

    summary_payload = {
        "last_run": last_run,
        "as_of_date": as_of_date,
        "active_filters": active_filters,
        "active_filters_human": active_filters_human,
        "params_used_type": params_used_type,
        "ytd_source": {
            "type": ytd_source_type,
            "file": bet_log_source,
        },
        "ytd_note": ytd_note,
        "summary_stats": {
            "total_games": total_games,
            "overall_accuracy": overall_accuracy,
            "as_of_date": as_of_date,
            "window_size": window_size,
        },
        "strategy_subset_in_window": {
            "matches": subset_matches,
            "wins": subset_wins,
        },
        "kpis": {
            "brier_score": float(brier_raw),
            "brier_score_raw": float(brier_raw),
            "brier_score_iso": float(brier_iso),
            "total_bets": subset_matches,
            "win_rate": subset_acc,
            "strategy_matched_games": subset_matches,
            "strategy_matched_wins": subset_wins,
            "strategy_win_rate": subset_acc,
            "bankroll_start_eur": 1000,
            "flat_stake_eur": 100,
            "bankroll_year": 2026,
            "bankroll_last_200_eur": bankroll_last_200_eur,
            "net_pl_last_200_eur": net_pl_last_200_eur,
            "bankroll_2026_ytd_eur": bankroll_2026_ytd_eur,
            "net_pl_2026_ytd_eur": net_pl_2026_ytd_eur,
            "bankroll_window_end": bankroll_window_block["bankroll_end"],
            "bankroll_window_pnl_sum": bankroll_window_block["pnl_sum"],
            "bankroll_window_trades": bankroll_window_block["n_trades"],
            "bankroll_window_sharpe": bankroll_window_block["sharpe"],
            "bankroll_window_max_dd_eur": bankroll_window_block["max_drawdown_eur"],
            "bankroll_window_max_dd_pct": bankroll_window_block["max_drawdown_pct"],
            "max_drawdown_eur": bankroll_window_block["max_drawdown_eur"],
            "max_drawdown_pct": bankroll_window_block["max_drawdown_pct"],
        },
        "bankroll": {
            "window_200": {"start": 1000, "flat_stake": 100, "bankroll_eur": bankroll_window_block["bankroll_end"]},
            "ytd_2026": {"start": 1000, "flat_stake": 100, "bankroll_eur": bankroll_ytd_block["bankroll_end"]},
        },
        "bets_2026_settled_overview": ytd_settled_overview,

        "risk_metrics": {"sharpe": bankroll_window_block["sharpe"]},
        "source": {
            "combined_file": str(combined_path),
            "local_matched_games_source": local_source,
            "params_source": params_source,
            "bet_log_file": bet_log_source,
        },
    }

    tables_payload = {
        "historical_stats": historical_stats,
        "accuracy_threshold_stats": accuracy_thresholds,
        "calibration_metrics": calibration_metrics,
        "calibration_quality": calibration_quality,
        "home_win_rates_window": home_win_rates,
        "home_win_rate_threshold": home_win_rate_threshold,
        "home_win_rate_shown_count": len(home_win_rates),
        "strategy_filter_stats": strategy_filter_stats,

        # 2026 settled bets list (from bet_log settled via combined)
        # Good for a table view in the dashboard.
        "bet_log_2026_settled_rows": ytd_settled_rows,
        "bet_log_2026_settled_count": ytd_settled_count,
        "bets_2026_settled_rows": ytd_settled_rows,
        "bets_2026_settled_count": ytd_settled_count,
        "bets_2026_settled_summary": ytd_settled_summary,


        "strategy_summary": {
            "asOfDate": as_of_date,
            "totalBets": subset_matches,
            "totalProfitEur": bankroll_window_block["pnl_sum"],
            "roiPct": roi_pct,
            "avgStakeEur": 100.0 if subset_matches else None,
            "avgProfitPerBetEur": _safe_div(float(bankroll_window_block["pnl_sum"] or 0.0), subset_matches) if subset_matches else None,
            "winRate": subset_acc,
            "avgEvPer100": avg_ev,
            "profitMetricsAvailable": True if bankroll_window_block["pnl_sum"] is not None else False,
        },
        "bankroll_history": bankroll_window_block["history"],
        "local_matched_games_rows": strategy_subset_rows,
        "local_matched_games_count": len(strategy_subset_rows),
        "local_matched_games_note": None,
    }

    last_run_payload = {
        "last_run": last_run,
        "as_of_date": as_of_date,
        "records": {
            "played_games_window": total_games,
            "window_start": window_start.strftime(DATE_FMT),
            "window_end": window_end.strftime(DATE_FMT),
            "local_matched_games_source": local_source,
            "local_matched_games_rows_window": len(strategy_subset_rows),
            "strategy_subset_wins": subset_wins,
            "params_source": params_source,
            "ytd_settled_bets_2026": ytd_settled_count,
            "ytd_settled_wins_2026": ytd_settled_wins,
            "ytd_settled_profit_eur_2026": ytd_settled_profit,
            "ytd_settled_roi_pct_2026": ytd_settled_roi_pct,
            "ytd_settled_avg_odds_2026": ytd_settled_avg_odds,

            "params_used": {
                "home_win_rate_threshold": float(params_used["home_win_rate_threshold"]),
                "odds_min": float(params_used["odds_min"]),
                "odds_max": float(params_used["odds_max"]),
                "prob_threshold": float(params_used["prob_threshold"]),
                "min_ev": float(params_used.get("min_ev", strategy_params["min_ev"])),
            },
        },
    }

    write_json(output_dir / "summary.json", summary_payload)
    write_json(output_dir / "tables.json", tables_payload)
    write_json(output_dir / "last_run.json", last_run_payload)

    print(f"Wrote summary.json, tables.json, last_run.json to {output_dir}")


if __name__ == "__main__":
    main()
