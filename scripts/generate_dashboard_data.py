#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate stats-only dashboard artifacts for hoops-insight.

This script reads historical outputs from Basketball_prediction (played games only)
and writes stable JSON files into public/data for the Vite app to consume.
No future predictions are loaded or emitted.

Data contract overview:
- Model performance (window / historical): combined_nba_predictions_* (played games only).
- Strategy simulation (local matched games): local_matched_games_YYYY-MM-DD.csv (copied/aliased to local_matched_games_latest.csv).
- Placed bets (real, settled): bet_log_flat_live.csv settled against combined_* results.

IMPORTANT:
- dashboard_state.json must be computed in main() and must match validate_dashboard_state.mjs.
- build_calibration_metrics() must ONLY compute calibration metrics; it must NOT write files.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

from config_loader import load_config
from snapshot_selection import copy_selection_aliases, resolve_snapshot_selection
from strategy_logic import apply_strategy_filters, load_strategy_params as load_versioned_strategy_params, required_columns


DATE_FMT = "%Y-%m-%d"
CALIBRATION_WINDOW = int(os.environ.get("N_WINDOW", os.environ.get("DASHBOARD_WINDOW", "200")))
DEFAULT_MAX_ODDS_FALLBACK = 3.2
RISK_MIN_SAMPLE = 5
REQUIRED_DASHBOARD_JSON = (
    "dashboard_payload.json",
    "dashboard_state.json",
    "tables.json",
)

LOGGER = logging.getLogger(__name__)

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
    local_matched_games: Optional[Path]
    strategy_params: Optional[Path]
    metrics_snapshot: Optional[Path]


def _normalize_key(key: str) -> str:
    key = key.strip().lower()
    key = re.sub(r"[\s\-]+", "_", key)
    key = re.sub(r"[^a-z0-9_]", "", key)
    key = re.sub(r"_+", "_", key)
    return key


def _safe_float(val: object) -> Optional[float]:
    if val is None:
        return None
    s = str(val).strip()
    if s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _safe_date(val: object) -> Optional[datetime]:
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


def _extract_date_from_name(name: str) -> Optional[str]:
    match = re.search(r"(\d{4}-\d{2}-\d{2})", name)
    return match.group(1) if match else None


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
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=[",", "\t", ";", "|"])
        except csv.Error:
            dialect = csv.excel

        reader = csv.DictReader(f, dialect=dialect)
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


def load_strategy_params(path: Path) -> Dict[str, object]:
    """
    Loads the JSON/text file verbatim into a dict.
    Note: selection/normalization of params_used happens in main() to mirror validator.
    """
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


def _extract_date_from_json(path: Optional[Path]) -> Optional[str]:
    if path is None or not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(payload, dict):
        return None
    for key in ("snapshot_as_of_date", "as_of_date", "asOfDate", "window_end"):
        value = payload.get(key)
        if isinstance(value, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            return value
    return None


def _max_date_from_local_rows(rows: List[Dict[str, object]]) -> Optional[str]:
    dates = [_safe_date(row.get("date")) for row in rows]
    dates = [d for d in dates if d is not None]
    if not dates:
        return None
    return max(dates).strftime(DATE_FMT)


def load_local_matched_games_csv(path: Path) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    """
    Loads local_matched_games_latest.csv in a way that matches validate_dashboard_state.mjs.

    Critical behavior:
    - canonical columns: odds_1, ev_eur_per_100
    - DO NOT fill missing numeric values with 0.0 (JS coerceNumber(null) -> null, not 0)
    - Convert NaN/NA -> None in output dicts
    """
    df = pd.read_csv(path)
    original_row_count = int(len(df))
    required_cols = list(required_columns())

    # Accept common date aliases produced by upstream exports and normalize to `date`.
    date_aliases = ("date", "game_date", "GAME_DATE", "event_date", "date_x")
    if "date" not in df.columns:
        source_date_col = next((c for c in date_aliases if c in df.columns), None)
        if source_date_col:
            df = df.copy()
            df["date"] = df[source_date_col]

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {', '.join(missing)}")

    odds_col = next((c for c in ("closing_home_odds", "odds", "odds_1") if c in df.columns), None)
    if odds_col is None:
        raise ValueError(f"{path.name} requires one of columns: closing_home_odds, odds, odds_1")

    ev_col = next((c for c in ("EV_€_per_100", "ev_eur_per_100", "ev_per_100") if c in df.columns), None)
    if ev_col is None:
        raise ValueError(f"{path.name} requires one of columns: EV_€_per_100, ev_eur_per_100, ev_per_100")

    df = df.copy()

    # date + teams
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime(DATE_FMT)
    df["home_team"] = df["home_team"].astype(str).str.strip().str.upper()
    df["away_team"] = df["away_team"].astype(str).str.strip().str.upper()

    # canonical columns matching validator expectations
    if odds_col != "odds_1":
        df["odds_1"] = df[odds_col]
    if ev_col != "ev_eur_per_100":
        df["ev_eur_per_100"] = df[ev_col]

    # numeric coercions (IMPORTANT: do NOT fill NaN with 0)
    numeric_cols = ["home_win_rate", "prob_iso", "prob_used", "odds_1", "ev_eur_per_100"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["win"] = pd.to_numeric(df["win"], errors="coerce")
    df["pnl"] = pd.to_numeric(df["pnl"], errors="coerce")

    # drop unusable rows
    df = df.dropna(subset=["date", "win", "pnl"])
    df = df[df["win"].isin([0, 1])]
    df["win"] = df["win"].astype(int)
    df["pnl"] = df["pnl"].astype(float)

    # optional stake
    if "stake" in df.columns:
        df["stake"] = pd.to_numeric(df["stake"], errors="coerce")
    else:
        df["stake"] = pd.NA

    # Convert NaN/NA -> None so filtering matches JS coerceNumber(null)
    df = df.where(pd.notna(df), None)

    # Keep-first dedupe by identity key
    df = df.drop_duplicates(subset=["date", "home_team", "away_team"], keep="first")

    rows = df.to_dict(orient="records")
    summary = {
        "rows_count": int(len(df)),
        "profit_sum_table": float(df["pnl"].sum()),
        "source_rows_count": original_row_count,
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


def _resolve_combined_file_from_data_dir(data_dir: Path) -> Optional[Path]:
    combined_latest = data_dir / "combined_latest.csv"
    if combined_latest.exists():
        return combined_latest
    return _find_latest_file(data_dir, "combined_nba_predictions_iso")


def _find_latest_by_mtime(path: Path, pattern: str) -> Optional[Path]:
    if not path.exists():
        return None
    candidates = [item for item in path.glob(pattern) if item.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda item: item.stat().st_mtime)


def _find_local_matched_games(lightgbm_dir: Path, as_of_date: Optional[str]) -> Optional[Path]:
    if as_of_date:
        candidate = lightgbm_dir / f"local_matched_games_{as_of_date}.csv"
        if candidate.exists():
            return candidate
    return _find_latest_by_mtime(lightgbm_dir, "local_matched_games_*.csv")


def _read_csv_dicts(path: Optional[Path], max_rows: int = 50) -> List[Dict[str, str]]:
    if not path or not path.exists():
        return []
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append({str(key): str(value) for key, value in row.items() if key is not None})
            if len(rows) >= max_rows:
                break
    return rows


def _write_csv_dicts(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def _load_home_win_rates(path: Optional[Path]) -> Dict[str, Dict[str, object]]:
    rates: Dict[str, Dict[str, object]] = {}
    for row in _read_csv_dicts(path, max_rows=100):
        team = _safe_team(row.get("") or row.get("team"))
        if not team:
            continue
        source_label = "home_win_rates_sorted latest file: last 20 team games, home rows only"
        rates[team] = {
            "home_win_rate": _safe_float(row.get("Home Win Rate") or row.get("home_win_rate")),
            "home_wins": _safe_float(row.get("Home Wins") or row.get("home_wins")),
            "home_games": _safe_float(row.get("Total Home Games") or row.get("total_home_games")),
            "last20_games": _safe_float(row.get("Total Last 20 Games") or row.get("total_last20_games")),
            "hwr_source_file": path.name if path else None,
            "hwr_source_label": source_label,
            "hwr_window_label": "last 20 team games; home games only",
        }
    return rates


def _read_json_object(path: Optional[Path]) -> Optional[Dict[str, object]]:
    if not path or not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _load_canonical_model_signals(lightgbm_dir: Path, run_date: Optional[str]) -> Dict[str, object]:
    summary_path = lightgbm_dir / f"script11_watchlist_history_summary_{run_date}.json" if run_date else None
    if summary_path and not summary_path.exists():
        summary_path = lightgbm_dir / "script11_watchlist_history_summary_latest.json"
    summary = _read_json_object(summary_path)

    rows_path = lightgbm_dir / f"script11_watchlist_history_{run_date}.csv" if run_date else None
    if rows_path and not rows_path.exists():
        rows_path = lightgbm_dir / "script11_watchlist_history_latest.csv"

    canonical_rows_all: List[Dict[str, str]] = []
    current_canonical_rows: List[Dict[str, str]] = []
    current_rows: List[Dict[str, str]] = []
    for row in _read_csv_dicts(rows_path, max_rows=500):
        is_current = str(row.get("date") or row.get("game_date") or "") == str(run_date)
        if is_current:
            current_rows.append(row)
        if str(row.get("canonical_signal") or "").strip():
            canonical_rows_all.append(row)
            if is_current:
                current_canonical_rows.append(row)

    return {
        "engine_state": (summary or {}).get("params_chosen") or None,
        "source_file": rows_path.name if rows_path and rows_path.exists() else None,
        "summary_file": summary_path.name if summary_path and summary_path.exists() else None,
        "canonical_count": len(current_canonical_rows),
        "canonical_total_count": len(canonical_rows_all),
        "canonical": current_canonical_rows[:20],
        "current_rows": current_rows[:20],
        "label": "Canonical: none" if not current_canonical_rows else f"Canonical: {len(current_canonical_rows)}",
    }


def _resolve_live_lightgbm_dir(source_root: Optional[Path]) -> Optional[Path]:
    env_dir = os.environ.get("LGBM_DIR")
    if env_dir:
        candidate = Path(env_dir).expanduser().resolve()
        if candidate.exists():
            return candidate
    if source_root is None:
        return None
    candidate = source_root / "output" / "LightGBM"
    if candidate.exists():
        return candidate
    candidate = source_root / "LightGBM"
    return candidate if candidate.exists() else None


def _build_today_games_payload(source_root: Optional[Path]) -> Dict[str, object]:
    lightgbm_dir = _resolve_live_lightgbm_dir(source_root)
    if lightgbm_dir is None:
        return {
            "as_of_date": None,
            "source": None,
            "games": [],
            "qualifying_bets": [],
            "local_matched_games": [],
        }

    predict_path = _find_latest_file(lightgbm_dir, "nba_games_predict")
    run_date = _extract_date_from_name(predict_path.name) if predict_path else None
    shortlist_path = lightgbm_dir / f"bet_shortlist_{run_date}.csv" if run_date else None
    if shortlist_path and not shortlist_path.exists() and run_date:
        shortlist_path = lightgbm_dir / f"flat_bet_shortlist_{run_date}.csv"
    local_matched_path = lightgbm_dir / f"local_matched_games_{run_date}.csv" if run_date else None
    home_win_rates_path = lightgbm_dir / f"home_win_rates_sorted_{run_date}.csv" if run_date else None
    home_win_rates = _load_home_win_rates(home_win_rates_path)
    setup_scan_path = lightgbm_dir / f"setup_profitability_scan_{run_date}.csv" if run_date else None
    setup_scan_summary_path = lightgbm_dir / f"setup_profitability_scan_summary_{run_date}.json" if run_date else None
    setup_scan_matches_path = lightgbm_dir / f"setup_profitability_scan_matches_{run_date}.csv" if run_date else None
    qualifying_bets = _read_csv_dicts(shortlist_path, max_rows=20)
    canonical_model_signals = _load_canonical_model_signals(lightgbm_dir, run_date)
    qualifying_by_game = {
        (
            str(row.get("date") or ""),
            _safe_team(row.get("home_team")) or "",
            _safe_team(row.get("away_team")) or "",
        ): row
        for row in qualifying_bets
    }

    games = []
    for row in _read_csv_dicts(predict_path, max_rows=40):
        home_prob = _safe_float(row.get("home_team_prob"))
        home_team = row.get("home_team")
        away_team = row.get("away_team")
        home_rate = home_win_rates.get(_safe_team(home_team) or "", {})
        odds_1 = _safe_float(row.get("odds 1") or row.get("odds_1"))
        odds_2 = _safe_float(row.get("odds 2") or row.get("odds_2"))
        candidate = qualifying_by_game.get(
            (
                str(row.get("date") or ""),
                _safe_team(home_team) or "",
                _safe_team(away_team) or "",
            ),
            {},
        )
        prob_used = _safe_float(candidate.get("prob_used"))
        games.append(
            {
                "date": row.get("date"),
                "home_team": home_team,
                "away_team": away_team,
                "home_team_prob": home_prob,
                "away_team_prob": 1 - home_prob if home_prob is not None else None,
                "prob_used": prob_used,
                "prob_base": _safe_float(candidate.get("prob_base")),
                "prob_live_oos_proxy": _safe_float(candidate.get("prob_live_oos_proxy")),
                "prob_iso": _safe_float(candidate.get("prob_iso")),
                "market_implied_p_devig": _safe_float(candidate.get("market_implied_p_devig")),
                "model_market_gap": _safe_float(candidate.get("model_market_gap")),
                "ev_live_eur_per_100": _safe_float(candidate.get("EV_live_€_per_100") or candidate.get("EV_€_per_100")),
                "candidate_stake_eur": _safe_float(candidate.get("stake_eur")),
                "home_win_rate": home_rate.get("home_win_rate"),
                "home_wins": home_rate.get("home_wins"),
                "home_games": home_rate.get("home_games"),
                "last20_games": home_rate.get("last20_games"),
                "hwr_source_file": home_rate.get("hwr_source_file"),
                "hwr_source_label": home_rate.get("hwr_source_label"),
                "hwr_window_label": home_rate.get("hwr_window_label"),
                "home_odds": odds_1,
                "away_odds": odds_2,
            }
        )

    return {
        "as_of_date": run_date,
        "source": predict_path.name if predict_path else None,
        "engine_state": canonical_model_signals.get("engine_state"),
        "canonical_model_signals": canonical_model_signals,
        "games": games,
        "qualifying_bets": qualifying_bets,
        "local_matched_games": _read_csv_dicts(local_matched_path, max_rows=20),
        "setup_profitability": {
            "summary": _read_json_object(setup_scan_summary_path),
            "rows": _read_csv_dicts(setup_scan_path, max_rows=20),
            "matches": _read_csv_dicts(setup_scan_matches_path, max_rows=50),
        },
        "ev_exception_profitability": None,
    }


def _historical_rows_with_prior_home_rate(
    rows: List[Dict[str, object]],
    prior_home_games: int = 20,
) -> List[Dict[str, object]]:
    history: Dict[str, List[int]] = defaultdict(list)
    output: List[Dict[str, object]] = []
    for row in sorted(rows, key=lambda item: item["date"]):
        home_team = row.get("home_team")
        home_team_won = row.get("home_team_won")
        if not isinstance(home_team, str) or home_team_won not in (0, 1):
            continue
        prior = history[home_team][-prior_home_games:]
        prior_rate = _safe_div(sum(prior), len(prior)) if prior else None
        enriched = dict(row)
        enriched["prior_home_win_rate"] = prior_rate
        enriched["prior_home_games"] = len(prior)
        output.append(enriched)
        history[home_team].append(int(home_team_won))
    return output


def _build_ev_exception_profitability(
    today_payload: Dict[str, object],
    played_rows: List[Dict[str, object]],
    active_params: Dict[str, float],
    window_start_dt: Optional[datetime],
    window_end_dt: Optional[datetime],
    output_dir: Path,
) -> Optional[Dict[str, object]]:
    qualifying_rows = today_payload.get("qualifying_bets")
    if not isinstance(qualifying_rows, list):
        return None

    min_ev = float(active_params.get("min_ev", 0.0))
    home_win_rate_min = float(active_params.get("home_win_rate_min", 0.0))
    odds_min = float(active_params.get("odds_min", 1.0))
    odds_max = float(active_params.get("odds_max", DEFAULT_MAX_ODDS_FALLBACK))
    prob_threshold = float(active_params.get("prob_threshold", 0.5))

    current_candidates: List[Dict[str, object]] = []
    for row in qualifying_rows:
        if not isinstance(row, dict):
            continue
        home_win_rate = _safe_float(row.get("home_win_rate"))
        odds_1 = _safe_float(row.get("odds_1"))
        prob_used = _safe_float(row.get("prob_used"))
        ev = _safe_float(row.get("EV_live_€_per_100") or row.get("EV_€_per_100"))
        if home_win_rate is None or odds_1 is None or prob_used is None or ev is None:
            continue
        passes_non_ev = (
            home_win_rate >= home_win_rate_min
            and odds_min <= odds_1 <= odds_max
            and prob_used >= prob_threshold
        )
        if passes_non_ev and ev <= min_ev:
            current_candidates.append(
                {
                    "date": row.get("date"),
                    "home_team": row.get("home_team"),
                    "away_team": row.get("away_team"),
                    "home_win_rate": home_win_rate,
                    "odds_1": odds_1,
                    "odds_2": _safe_float(row.get("odds_2")),
                    "home_team_prob": _safe_float(row.get("home_team_prob")),
                    "prob_used": prob_used,
                    "ev_eur_per_100": ev,
                    "stake_eur": _safe_float(row.get("stake_eur")),
                    "kelly_full": _safe_float(row.get("kelly_full")),
                    "hwr_source_file": next(
                        (
                            game.get("hwr_source_file")
                            for game in today_payload.get("games", [])
                            if isinstance(game, dict)
                            and game.get("date") == row.get("date")
                            and _safe_team(game.get("home_team")) == _safe_team(row.get("home_team"))
                            and _safe_team(game.get("away_team")) == _safe_team(row.get("away_team"))
                        ),
                        None,
                    ),
                    "hwr_source_label": next(
                        (
                            game.get("hwr_source_label")
                            for game in today_payload.get("games", [])
                            if isinstance(game, dict)
                            and game.get("date") == row.get("date")
                            and _safe_team(game.get("home_team")) == _safe_team(row.get("home_team"))
                            and _safe_team(game.get("away_team")) == _safe_team(row.get("away_team"))
                        ),
                        None,
                    ),
                    "hwr_window_label": next(
                        (
                            game.get("hwr_window_label")
                            for game in today_payload.get("games", [])
                            if isinstance(game, dict)
                            and game.get("date") == row.get("date")
                            and _safe_team(game.get("home_team")) == _safe_team(row.get("home_team"))
                            and _safe_team(game.get("away_team")) == _safe_team(row.get("away_team"))
                        ),
                        None,
                    ),
                    "blocked_by": "min_ev",
                }
            )

    if not current_candidates:
        return None

    historical_rows = _historical_rows_with_prior_home_rate(played_rows)
    matches: List[Dict[str, object]] = []
    for row in historical_rows:
        date_value = row.get("date")
        if not isinstance(date_value, datetime):
            continue
        if window_start_dt and date_value < window_start_dt:
            continue
        if window_end_dt and date_value > window_end_dt:
            continue

        prior_home_win_rate = _safe_float(row.get("prior_home_win_rate"))
        odds_home = _safe_float(row.get("odds_home"))
        prob = _safe_float(row.get("prob_used")) or _safe_float(row.get("prob_iso")) or _safe_float(row.get("prob_raw"))
        home_team_won = row.get("home_team_won")
        if (
            prior_home_win_rate is None
            or odds_home is None
            or prob is None
            or home_team_won not in (0, 1)
        ):
            continue
        if prior_home_win_rate < home_win_rate_min:
            continue
        if odds_home < odds_min or odds_home > odds_max:
            continue
        if prob < prob_threshold:
            continue

        pnl = (odds_home - 1.0) * 100.0 if int(home_team_won) == 1 else -100.0
        matches.append(
            {
                "date": date_value.strftime(DATE_FMT),
                "home_team": row.get("home_team"),
                "away_team": row.get("away_team"),
                "home_win_rate": prior_home_win_rate,
                "home_team_prob": _safe_float(row.get("prob_raw")),
                "prob_used": prob,
                "EV_€_per_100": None,
                "EV_live_€_per_100": None,
                "odds_1": odds_home,
                "home_team_won": int(home_team_won),
                "win": int(home_team_won),
                "home_ml_pnl_100": pnl,
                "pnl_100": pnl,
            }
        )

    n = len(matches)
    wins = sum(1 for row in matches if row["win"] == 1)
    profit = float(sum(float(row["pnl_100"]) for row in matches))
    first_candidate = current_candidates[0] if current_candidates else {}
    current_home_team = str(first_candidate.get("home_team") or "HOME")
    current_away_team = str(first_candidate.get("away_team") or "AWAY")
    current_date = str(first_candidate.get("date") or "unknown")
    current_odds = _safe_float(first_candidate.get("odds_1"))
    current_prob_used = _safe_float(first_candidate.get("prob_used"))
    current_ev = _safe_float(first_candidate.get("ev_eur_per_100"))
    current_kelly = _safe_float(first_candidate.get("kelly_full"))
    current_stake = _safe_float(first_candidate.get("stake_eur"))

    price_low = (current_odds - 0.10) if current_odds is not None else None
    price_high = (current_odds + 0.10) if current_odds is not None else None
    prob_low = max(0.0, current_prob_used - 0.05) if current_prob_used is not None else None
    prob_high = min(1.0, current_prob_used + 0.05) if current_prob_used is not None else None
    price_matches = [
        row
        for row in matches
        if price_low is not None
        and price_high is not None
        and prob_low is not None
        and prob_high is not None
        and price_low <= float(row["odds_1"]) <= price_high
        and prob_low <= float(row["prob_used"]) <= prob_high
    ]
    price_n = len(price_matches)
    price_wins = sum(1 for row in price_matches if row["win"] == 1)
    price_profit = float(sum(float(row["pnl_100"]) for row in price_matches))
    break_even = (1.0 / current_odds) if current_odds else None
    price_win_rate = _safe_div(price_wins, price_n) if price_n else None
    win_rate_minus_break_even = (
        price_win_rate - break_even
        if price_win_rate is not None and break_even is not None
        else None
    )
    current_prob_minus_break_even = (
        current_prob_used - break_even
        if current_prob_used is not None and break_even is not None
        else None
    )
    price_adjusted_supports_play = bool(
        price_n
        and win_rate_minus_break_even is not None
        and win_rate_minus_break_even > 0
        and price_profit > 0
        and (current_ev is None or current_ev > min_ev)
        and (current_kelly is None or current_kelly > 0)
        and (current_stake is None or current_stake > 0)
    )
    warning = None
    if profit > 0 and (
        current_ev is not None and current_ev <= min_ev
        or current_kelly is not None and current_kelly <= 0
        or current_stake is not None and current_stake <= 0
        or current_prob_minus_break_even is not None and current_prob_minus_break_even <= 0
        or win_rate_minus_break_even is not None and win_rate_minus_break_even <= 0
    ):
        warning = (
            f"Broad historical EV-exception group is profitable at avg odds "
            f"{_safe_div(sum(float(row['odds_1']) for row in matches), n):.2f}, "
            f"but today's price {current_odds:.2f} requires {break_even * 100:.1f}% break-even. "
            f"Current prob_used is {current_prob_used * 100:.1f}%; Kelly is "
            f"{current_kelly if current_kelly is not None else 'unavailable'}. Treat as watch-only, not a bet."
        )

    debug_filename = (
        f"ev_exception_matches_{current_date}_{_safe_team(current_home_team) or current_home_team}_"
        f"{_safe_team(current_away_team) or current_away_team}.csv"
    )
    debug_path = output_dir / debug_filename
    _write_csv_dicts(
        debug_path,
        matches,
        [
            "date",
            "home_team",
            "away_team",
            "home_win_rate",
            "odds_1",
            "home_team_prob",
            "prob_used",
            "EV_€_per_100",
            "EV_live_€_per_100",
            "win",
            "home_team_won",
            "pnl_100",
            "home_ml_pnl_100",
        ],
    )
    return {
        "label": "Active setup without EV filter",
        "classification": "historical_support_only",
        "is_betting_signal": False,
        "recommendation_label": "watch-only",
        "warning": warning,
        "note": "Historical scan applies HW, odds, and probability thresholds only; EV is intentionally ignored because the current row is blocked only by EV.",
        "debug_csv": debug_filename,
        "criteria": {
            "home_win_rate_min": home_win_rate_min,
            "odds_min": odds_min,
            "odds_max": odds_max,
            "prob_threshold": prob_threshold,
            "ev_filter": "ignored",
            "stake_model": "flat_100_home_ml",
            "home_win_rate_basis": "prior_home_games",
        },
        "current_candidates": current_candidates,
        "summary": {
            "n": n,
            "wins": wins,
            "losses": n - wins,
            "win_rate": _safe_div(wins, n) if n else None,
            "avg_odds": _safe_div(sum(float(row["odds_1"]) for row in matches), n) if n else None,
            "profit_100_flat": profit,
            "roi_pct": _safe_div(profit, n * 100.0) * 100.0 if n else None,
            "avg_prob_used": _safe_div(sum(float(row["prob_used"]) for row in matches), n) if n else None,
            "avg_home_win_rate": _safe_div(sum(float(row["home_win_rate"]) for row in matches), n) if n else None,
            "window_start": window_start_dt.strftime(DATE_FMT) if window_start_dt else None,
            "window_end": window_end_dt.strftime(DATE_FMT) if window_end_dt else None,
        },
        "price_adjusted": {
            "label": "Current-price EV-exception check",
            "odds_band": [price_low, price_high],
            "prob_used_band": [prob_low, prob_high],
            "hwr_source_file": first_candidate.get("hwr_source_file"),
            "hwr_source_label": first_candidate.get("hwr_source_label"),
            "hwr_window_label": first_candidate.get("hwr_window_label"),
            "current_odds": current_odds,
            "current_prob_used": current_prob_used,
            "current_ev_eur_per_100": current_ev,
            "current_kelly": current_kelly,
            "current_stake_eur": current_stake,
            "break_even_probability": break_even,
            "current_prob_minus_break_even": current_prob_minus_break_even,
            "n": price_n,
            "wins": price_wins,
            "losses": price_n - price_wins,
            "win_rate": price_win_rate,
            "avg_odds": _safe_div(sum(float(row["odds_1"]) for row in price_matches), price_n) if price_n else None,
            "profit_100_flat": price_profit,
            "roi_pct": _safe_div(price_profit, price_n * 100.0) * 100.0 if price_n else None,
            "win_rate_minus_break_even": win_rate_minus_break_even,
            "supports_play": price_adjusted_supports_play,
            "classification": "watch-only" if not price_adjusted_supports_play else "historical_support_only",
        },
        "matches": matches[-50:],
    }


def _find_historical_roi_attack_summaries(lightgbm_dir: Optional[Path], run_date: Optional[str]) -> List[Dict[str, object]]:
    if lightgbm_dir is None or not run_date:
        return []
    scan_dir = lightgbm_dir / "historical_roi_attack"
    if not scan_dir.exists():
        return []

    summaries: List[Dict[str, object]] = []
    for path in sorted(scan_dir.glob(f"historical_roi_attack_scan_{run_date}_*_summary.json")):
        payload = _read_json_object(path)
        if not payload:
            continue
        payload = dict(payload)
        payload["source_file"] = path.name
        summaries.append(payload)
    return summaries


def _bucket_summary(summary: Dict[str, object], bucket_name: str, tail: str = "all") -> Optional[Dict[str, object]]:
    buckets = summary.get("bucket_summaries")
    if not isinstance(buckets, list):
        return None
    for bucket in buckets:
        if (
            isinstance(bucket, dict)
            and bucket.get("bucket_name") == bucket_name
            and str(bucket.get("tail") or "") == tail
        ):
            return bucket
    return None


def _select_probability_for_local_rule(target_row: Dict[str, object]) -> Optional[float]:
    for key in ("prob_used", "prob_live_safe", "prob_iso_oos_time", "prob_iso", "home_team_prob"):
        value = _safe_float(target_row.get(key))
        if value is not None:
            return value
    return None


def _build_local_profitability_rule_case(
    summary: Dict[str, object],
    active_params: Dict[str, float],
) -> Optional[Dict[str, object]]:
    target = summary.get("target_row_values")
    if not isinstance(target, dict):
        return None

    canonical_context = summary.get("canonical_context")
    if not isinstance(canonical_context, dict):
        canonical_context = {}

    hwr_threshold = float(active_params.get("home_win_rate_min", 0.0))
    odds_min = float(active_params.get("odds_min", 1.0))
    odds_max = float(active_params.get("odds_max", DEFAULT_MAX_ODDS_FALLBACK))
    prob_threshold = float(active_params.get("prob_threshold", 0.5))

    home_win_rate = _safe_float(target.get("home_win_rate"))
    odds_1 = _safe_float(target.get("odds_1"))
    selected_probability = _select_probability_for_local_rule(target)
    local_profitable_candidate = bool(
        home_win_rate is not None
        and odds_1 is not None
        and selected_probability is not None
        and home_win_rate >= hwr_threshold
        and odds_min <= odds_1 <= odds_max
        and selected_probability >= prob_threshold
    )

    canonical_signal = bool(canonical_context.get("canonical_signal"))
    local_engine_state = str(canonical_context.get("local_engine_state") or "").upper()
    robust_stability_passed = bool(canonical_signal or (local_engine_state and local_engine_state != "NO_BET"))
    historical_status = str(summary.get("historical_roi_attack_status") or "unknown")
    price_strict = _bucket_summary(summary, "price_strict_bucket") or {}
    hwr_filtered = _bucket_summary(summary, "hwr_filtered_bucket") or {}
    broad = _bucket_summary(summary, "broad_similar_current_setup") or {}

    scanner_supported = bool(
        historical_status == "supported_discretionary_only"
        and (_safe_float(price_strict.get("roi_pct")) or 0.0) > 0
        and (_safe_float(hwr_filtered.get("roi_pct")) or 0.0) > 0
        and (_safe_float(price_strict.get("win_rate_minus_break_even_pp")) or 0.0) > 0
        and (_safe_float(hwr_filtered.get("win_rate_minus_break_even_pp")) or 0.0) > 0
        and (_safe_float(price_strict.get("n")) or 0.0) >= 30
        and (_safe_float(hwr_filtered.get("n")) or 0.0) >= 20
    )

    if local_profitable_candidate and not robust_stability_passed and scanner_supported:
        agent_decision = "SMALL_BET"
        agent_label = "discretionary_local_profitability_confirmed"
        stake_class = "small_fixed_only"
        reason = (
            "Game matched profitable local candidate params and the repeatable Historical ROI Attack scanner "
            "confirmed price-strict and HWR-filtered profitability above break-even."
        )
        lesson = (
            "Small discretionary bet allowed only after local profitable setup and historical profitability confirmation."
        )
    elif local_profitable_candidate and not robust_stability_passed:
        agent_decision = "SKIP"
        agent_label = "profitable_local_candidate_but_historical_rejected"
        stake_class = "none"
        reason = (
            "Game matched profitable local candidate params, but the repeatable Historical ROI Attack scanner "
            "did not clear break-even and profitability requirements."
        )
        lesson = (
            "Profitable local params alone are insufficient. Historical scanner confirmation is required."
        )
    elif local_profitable_candidate:
        agent_decision = "REVIEW_CANONICAL"
        agent_label = "profitable_local_candidate_but_robust_rejected"
        stake_class = "none"
        reason = "Game matched local candidate params, but this rule never overrides canonical Robust++++ classification."
        lesson = "Keep local profitability context separate from canonical model bets."
    else:
        agent_decision = "SKIP"
        agent_label = "correct_no_bet_discipline"
        stake_class = "none"
        reason = "Game did not match the local profitable candidate trigger."
        lesson = "No local override category fires without the local candidate trigger."

    game = str(summary.get("game") or f"{target.get('home_team', 'HOME')} vs {target.get('away_team', 'AWAY')}")
    case = {
        "rule_name": "Profitable Local Candidate + Historical ROI Confirmation Rule",
        "date": str(summary.get("target_date") or target.get("date") or ""),
        "game": game,
        "home_team": target.get("home_team"),
        "away_team": target.get("away_team"),
        "canonical_decision": "NO_BET" if not canonical_signal else "CANONICAL",
        "canonical_signal": canonical_signal,
        "local_profitable_candidate": local_profitable_candidate,
        "robust_stability_passed": robust_stability_passed,
        "local_params": {
            "home_win_rate_threshold": hwr_threshold,
            "odds_min": odds_min,
            "odds_max": odds_max,
            "prob_threshold": prob_threshold,
        },
        "target_values": {
            "home_win_rate": home_win_rate,
            "odds_1": odds_1,
            "selected_probability": selected_probability,
            "prob_used": _safe_float(target.get("prob_used")),
            "EV_live_€_per_100": _safe_float(target.get("EV_live_€_per_100") or target.get("EV_€_per_100")),
            "blocked_by": target.get("blocked_by") or canonical_context.get("local_block_reason"),
        },
        "historical_roi_attack_status": historical_status,
        "historical_roi_attack_source_file": summary.get("source_file"),
        "buckets": {
            "broad_similar_current_setup": broad,
            "price_strict_bucket": price_strict,
            "hwr_filtered_bucket": hwr_filtered,
        },
        "agent_decision": agent_decision,
        "agent_label": agent_label,
        "stake_class": stake_class,
        "reason": reason,
        "outcome": None,
        "pnl": None,
        "lesson": lesson,
        "canonical_override_allowed": False,
    }
    return case


def _build_local_profitability_rule_payload(
    summaries: List[Dict[str, object]],
    active_params: Dict[str, float],
) -> Dict[str, object]:
    cases = [
        case
        for summary in summaries
        if (case := _build_local_profitability_rule_case(summary, active_params)) is not None
    ]
    return {
        "rule_name": "Profitable Local Candidate + Historical ROI Confirmation Rule",
        "canonical_override_allowed": False,
        "approval_source": "repeatable_local_historical_roi_attack_scanner_only",
        "cases": cases,
    }


def _write_agent_learning_cases(output_dir: Path, cases: List[Dict[str, object]]) -> None:
    path = output_dir / "agent_learning_cases.jsonl"
    existing: Dict[Tuple[str, str, str], Dict[str, object]] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            key = (
                str(payload.get("date") or ""),
                str(payload.get("game") or ""),
                str(payload.get("rule_name") or ""),
            )
            existing[key] = payload
    for case in cases:
        key = (
            str(case.get("date") or ""),
            str(case.get("game") or ""),
            str(case.get("rule_name") or ""),
        )
        existing[key] = case

    ordered_cases = sorted(existing.values(), key=lambda item: (str(item.get("date") or ""), str(item.get("game") or "")))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for case in ordered_cases:
            handle.write(json.dumps(case, default=_serialize, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def _params_match_active_window_source(raw_params: Dict[str, object], active_params: Dict[str, float]) -> bool:
    raw_thresholds = _normalize_params(raw_params)
    comparisons = {
        "home_win_rate_threshold": "home_win_rate_min",
        "odds_min": "odds_min",
        "odds_max": "odds_max",
        "prob_threshold": "prob_threshold",
    }
    for raw_key, active_key in comparisons.items():
        raw_value = _safe_float(raw_thresholds.get(raw_key))
        active_value = _safe_float(active_params.get(active_key))
        if raw_value is None or active_value is None:
            return False
        if not math.isclose(raw_value, active_value, rel_tol=1e-9, abs_tol=1e-9):
            return False
    return True


def _build_local_strategy_evaluation_window(
    raw_strategy_payload: Dict[str, object],
    strategy_params_source: Optional[Path],
    active_params: Dict[str, float],
) -> Dict[str, object]:
    local_tail_used = _safe_float(raw_strategy_payload.get("local_tail_used"))
    hist_df_rows = _safe_float(raw_strategy_payload.get("hist_df_rows"))
    local_eval_rows = _safe_float(raw_strategy_payload.get("local_eval_rows"))
    valid_window_size = _safe_float(raw_strategy_payload.get("valid_window_size"))
    raw_window_value = local_tail_used or hist_df_rows or local_eval_rows or valid_window_size
    matches_active_params = _params_match_active_window_source(raw_strategy_payload, active_params) if raw_window_value is not None else False
    authoritative_value = raw_window_value
    if not matches_active_params:
        authoritative_value = None

    warning = None
    if raw_window_value is not None and not matches_active_params:
        warning = "strategy params metadata did not match the active local setup; local evaluation window hidden"
    elif authoritative_value is None:
        warning = "local Script 11 evaluation window is not available in generated artifacts"

    return {
        "label": "Script 11 local tail",
        "local_tail_used": int(local_tail_used) if local_tail_used is not None and local_tail_used.is_integer() else local_tail_used,
        "hist_df_rows": int(hist_df_rows) if hist_df_rows is not None and hist_df_rows.is_integer() else hist_df_rows,
        "local_eval_rows": int(local_eval_rows) if local_eval_rows is not None and local_eval_rows.is_integer() else local_eval_rows,
        "valid_window_size": int(valid_window_size) if valid_window_size is not None and valid_window_size.is_integer() else valid_window_size,
        "display_window_games": int(authoritative_value) if authoritative_value is not None and authoritative_value.is_integer() else authoritative_value,
        "source_file": strategy_params_source.name if strategy_params_source else None,
        "matches_active_params": matches_active_params,
        "warning": warning,
    }


def _find_strategy_params(lightgbm_dir: Path, as_of_date: Optional[str]) -> Optional[Path]:
    if as_of_date:
        dated_json = lightgbm_dir / f"strategy_params_{as_of_date}.json"
        if dated_json.exists():
            return dated_json
        dated = lightgbm_dir / f"strategy_params_{as_of_date}.txt"
        if dated.exists():
            return dated
    preferred_candidates = [
        path
        for path in (lightgbm_dir / "strategy_params.json", lightgbm_dir / "strategy_params.txt")
        if path.exists()
    ]
    if preferred_candidates:
        return max(preferred_candidates, key=lambda path: path.stat().st_mtime)
    latest_json = _find_latest_by_mtime(lightgbm_dir, "strategy_params*.json")
    if latest_json:
        return latest_json
    return _find_latest_by_mtime(lightgbm_dir, "strategy_params*.txt")


def resolve_source_root(cli_root: Optional[str], repo_root: Path) -> Optional[Path]:
    """Resolve source root from CLI override or central config."""
    if cli_root:
        candidate = Path(cli_root).expanduser().resolve()
        if candidate.exists():
            return candidate
    cfg = load_config()
    candidate = Path(cfg["SOURCE_ROOT"]).expanduser().resolve()
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
            local_matched_games=None,
            strategy_params=None,
            metrics_snapshot=None,
        )

    lightgbm_dir = root / "output" / "LightGBM"
    if not lightgbm_dir.exists():
        direct_lightgbm_dir = root / "LightGBM"
        if direct_lightgbm_dir.exists():
            lightgbm_dir = direct_lightgbm_dir
    kelly_dir = lightgbm_dir / "Kelly"

    combined_iso = _find_latest_file(kelly_dir, "combined_nba_predictions_iso")
    combined_acc = _find_latest_file(lightgbm_dir, "combined_nba_predictions_acc")

    bet_log = lightgbm_dir / "bet_log_live.csv"
    if not bet_log.exists():
        bet_log = _find_latest_file(lightgbm_dir, "bet_log_live")

    bet_log_flat = lightgbm_dir / "bet_log_flat_live.csv"
    if not bet_log_flat.exists():
        bet_log_flat = _find_latest_file(lightgbm_dir, "bet_log_flat_live")

    strategy_params = _find_strategy_params(lightgbm_dir, as_of_date)
    local_matched_games = _find_local_matched_games(lightgbm_dir, as_of_date)
    metrics_snapshot = lightgbm_dir / "metrics_snapshot.json"
    if not metrics_snapshot.exists():
        root_metrics_snapshot = root / "metrics_snapshot.json"
        metrics_snapshot = root_metrics_snapshot if root_metrics_snapshot.exists() else None

    return SourcePaths(
        combined_iso=combined_iso,
        combined_acc=combined_acc,
        bet_log=bet_log,
        bet_log_flat=bet_log_flat,
        local_matched_games=local_matched_games,
        strategy_params=strategy_params,
        metrics_snapshot=metrics_snapshot,
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
    rows: List[Dict[str, object]] = []
    for row in _read_csv_normalized(path):
        # date (MUST exist)
        date_raw = row.get("game_date") or row.get("date")
        game_date = _safe_date(date_raw)
        if game_date is None:
            continue

        # teams (fallback to matchup if needed)
        home_team = _safe_team(row.get("home_team"))
        away_team = _safe_team(row.get("away_team"))
        if (home_team is None or away_team is None) and row.get("matchup"):
            h2, a2 = _parse_matchup(row.get("matchup"))
            home_team = home_team or h2
            away_team = away_team or a2

        # winner / result (robust)
        result_raw = (
            row.get("result")
            or row.get("result_raw")
            or row.get("winner")
            or row.get("winning_team")
            or row.get("team_won")
            or row.get("home_team_won")
            or row.get("home_win")
        )

        if result_raw is None:
            continue
        raw_s = str(result_raw).strip().upper()
        if raw_s in {"", "NA", "NAN", "NONE"}:
            continue
        if raw_s in {"0", "0.0"} and not str(row.get("home_team_won") or row.get("home_win") or "").strip():
            continue

        # Determine label: 1 if home won else 0
        if raw_s in {"1", "HOME", "H", "TRUE"}:
            home_team_won = 1
        elif raw_s in {"0", "AWAY", "A", "FALSE"}:
            home_team_won = 0
        else:
            result_team = _safe_team(result_raw)
            if result_team is None or home_team is None:
                continue
            home_team_won = 1 if result_team == home_team else 0

        prob_raw = _safe_float(
            row.get("pred_home_win_proba")
            or row.get("home_team_prob")
            or row.get("prob_raw")
            or row.get("prob")
        )
        prob_iso = _safe_float(
            row.get("iso_proba_home_win")
            or row.get("prob_iso")
            or row.get("probability_iso")
            or row.get("iso_prob")
        )
        prob_used = _safe_float(
            row.get("prob_used")
            or row.get("prob_live_safe")
            or row.get("prob_live_oos_proxy")
            or row.get("prob_base")
        )
        odds = _safe_float(
            row.get("closing_home_odds")
            or row.get("odds_1")
            or row.get("odds")
            or row.get("home_odds")
        )
        home_win_rate = _safe_float(
            row.get("home_win_rate")
            or row.get("home_win_pct")
            or row.get("hw")
            or row.get("home_wr")
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
            }
        )

    return rows


def load_played_games_with_history(path: Path) -> List[Dict[str, object]]:
    """Load played games from `path` and, when possible, stitch dated snapshot history.

    Some combined snapshot files contain only one slate/day. If `path` follows the
    `...YYYY-MM-DD.csv` naming pattern, this function also loads older sibling snapshots
    with the same prefix and merges them (deduplicated by date/home/away).
    """
    name_match = re.match(r"^(?P<prefix>.*?)(?P<date>\d{4}-\d{2}-\d{2})(?P<suffix>\.csv)$", path.name)
    if not name_match:
        return load_played_games(path)

    snapshot_date = name_match.group("date")
    prefix = name_match.group("prefix")
    suffix = name_match.group("suffix")

    candidates: List[Tuple[str, Path]] = []
    for candidate in path.parent.glob(f"{prefix}*{suffix}"):
        if not candidate.is_file():
            continue
        candidate_date = _extract_date_from_name(candidate.name)
        if candidate_date and candidate_date <= snapshot_date:
            candidates.append((candidate_date, candidate))

    if not candidates:
        return load_played_games(path)

    merged: Dict[Tuple[str, str, str], Dict[str, object]] = {}
    for _, candidate_path in sorted(candidates, key=lambda item: item[0]):
        for row in load_played_games(candidate_path):
            date_val = row.get("date")
            home_team = row.get("home_team")
            away_team = row.get("away_team")
            if not isinstance(date_val, datetime):
                continue
            if not isinstance(home_team, str) or not isinstance(away_team, str):
                continue
            key = (date_val.strftime(DATE_FMT), home_team, away_team)
            merged[key] = row

    return sorted(merged.values(), key=lambda row: row["date"])


def build_historical_stats(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    per_day = defaultdict(lambda: {"total": 0, "correct": 0})

    for r in rows:
        d = r["date"].strftime(DATE_FMT)
        p = r["prob_iso"] if r.get("prob_iso") is not None else r.get("prob_raw")
        if p is None:
            continue

        y = int(r["home_team_won"])
        pred = 1 if float(p) >= 0.5 else 0
        correct = 1 if pred == y else 0

        per_day[d]["total"] += 1
        per_day[d]["correct"] += correct

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
            p = r["prob_iso"] if r.get("prob_iso") is not None else r.get("prob_raw")
            if p is None:
                continue
            p = float(p)

            if spec["thresholdType"] == "gt" and p > threshold:
                passed.append(r)
            elif spec["thresholdType"] == "lt" and p <= threshold:
                passed.append(r)

        total = 0
        correct = 0
        for r in passed:
            p = r["prob_iso"] if r.get("prob_iso") is not None else r.get("prob_raw")
            if p is None:
                continue
            y = int(r["home_team_won"])
            pred = 1 if float(p) >= 0.5 else 0
            total += 1
            correct += 1 if pred == y else 0

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
    """
    MUST be pure: compute metrics only. No file writes, no dependency on main() variables.
    """
    if not rows:
        return {
            "asOfDate": "—",
            "brierBefore": 0.0,
            "brierAfter": 0.0,
            "logLossBefore": 0.0,
            "logLossAfter": 0.0,
            "fittedGames": 0,
            "ece": 0.0,
            "calibrationSlope": 0.0,
            "calibrationIntercept": 0.0,
            "avgPredictedProb": 0.0,
            "baseRate": 0.0,
            "actualWinPct": 0.0,
            "windowSize": 0,
        }

    window_rows, _, window_end = compute_window_bounds(rows, window_size)
    window_size_used = len(window_rows)
    as_of = window_end.strftime(DATE_FMT) if window_end else "—"

    prob_used: List[float] = []
    y_used: List[int] = []
    for r in window_rows:
        prob = r["prob_iso"] if r.get("prob_iso") is not None else r.get("prob_raw")
        if prob is None:
            continue
        prob_used.append(float(prob))
        y_used.append(int(r["home_team_won"]))

    if not prob_used:
        base_rate = (
            _safe_div(sum(int(r["home_team_won"]) for r in window_rows), window_size_used)
            if window_size_used
            else 0.0
        )
        return {
            "asOfDate": as_of,
            "brierBefore": 0.0,
            "brierAfter": 0.0,
            "logLossBefore": 0.0,
            "logLossAfter": 0.0,
            "fittedGames": 0,
            "ece": 0.0,
            "calibrationSlope": 0.0,
            "calibrationIntercept": 0.0,
            "avgPredictedProb": 0.0,
            "baseRate": float(base_rate),
            "actualWinPct": float(base_rate),
            "windowSize": int(window_size_used),
        }

    p_raw: List[float] = []
    y_raw: List[int] = []
    for r in window_rows:
        if r.get("prob_raw") is None:
            continue
        p_raw.append(float(r["prob_raw"]))
        y_raw.append(int(r["home_team_won"]))

    p_iso: List[float] = []
    y_iso: List[int] = []
    for r in window_rows:
        if r.get("prob_iso") is None:
            continue
        p_iso.append(float(r["prob_iso"]))
        y_iso.append(int(r["home_team_won"]))

    if p_raw:
        brier_before = float(_compute_brier(y_raw, p_raw))
        logloss_before = float(_compute_log_loss(y_raw, p_raw))
    else:
        brier_before = float(_compute_brier(y_used, prob_used))
        logloss_before = float(_compute_log_loss(y_used, prob_used))

    if p_iso:
        brier_after = float(_compute_brier(y_iso, p_iso))
        logloss_after = float(_compute_log_loss(y_iso, p_iso))
        fitted_games = len(p_iso)
    else:
        brier_after = float(_compute_brier(y_used, prob_used))
        logloss_after = float(_compute_log_loss(y_used, prob_used))
        fitted_games = len(prob_used)

    base_rate = (
        _safe_div(sum(int(r["home_team_won"]) for r in window_rows), window_size_used)
        if window_size_used
        else 0.0
    )
    avg_pred_prob = _safe_div(sum(prob_used), len(prob_used)) if prob_used else 0.0

    calibration_intercept, calibration_slope = _fit_logistic_calibration(y_used, prob_used)
    ece = _compute_ece(y_used, prob_used, bins=10)

    return {
        "asOfDate": as_of,
        "brierBefore": float(brier_before),
        "brierAfter": float(brier_after),
        "logLossBefore": float(logloss_before),
        "logLossAfter": float(logloss_after),
        "fittedGames": int(fitted_games),
        "ece": float(ece or 0.0),
        "calibrationSlope": float(calibration_slope or 0.0),
        "calibrationIntercept": float(calibration_intercept or 0.0),
        "avgPredictedProb": float(avg_pred_prob),
        "baseRate": float(base_rate),
        "actualWinPct": float(base_rate),
        "windowSize": int(window_size_used),
    }


def build_home_win_rates_last20(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    per_team: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        game_date = row.get("date")
        if not isinstance(game_date, datetime):
            continue

        home_team = _safe_team(row.get("home_team"))
        away_team = _safe_team(row.get("away_team"))
        if not home_team or not away_team:
            continue

        raw_home_win = row.get("home_team_won")
        if isinstance(raw_home_win, bool):
            home_team_won = 1 if raw_home_win else 0
        elif isinstance(raw_home_win, (int, float)) and raw_home_win in (0, 1):
            home_team_won = int(raw_home_win)
        else:
            home_team_won = None

        per_team[home_team].append(
            {
                "date": game_date,
                "is_home": True,
                "home_team_won": home_team_won,
            }
        )
        per_team[away_team].append(
            {
                "date": game_date,
                "is_home": False,
                "home_team_won": home_team_won,
            }
        )

    output = []
    for team, games in per_team.items():
        games = sorted(games, key=lambda x: x["date"])[-20:]
        total_last20 = len(games)
        total_home = sum(1 for g in games if g["is_home"])
        home_wins = sum(
            1 for g in games if g["is_home"] and g["home_team_won"] == 1
        )
        home_win_rate = _safe_div(home_wins, total_home) if total_home else 0.0
        output.append(
            {
                "team": team,
                "totalLast20Games": int(total_last20),
                "totalHomeGames": int(total_home),
                "homeWins": int(home_wins),
                "homeWinRate": float(home_win_rate),
            }
        )

    output.sort(key=lambda x: (-x["homeWinRate"], -x["totalHomeGames"], x["team"]))
    return output


def build_home_win_rates_window(
    played_rows: List[Dict[str, object]],
    window_size: int = 20,
) -> List[Dict[str, object]]:
    """
    Windowed home win rate per team, computed only from HOME games
    inside the last `window_size` played games (overall games window).
    Output row shape:
      team, homeWinRate, homeWins, homeGames, windowGames
    """
    if not played_rows:
        return []

    sorted_rows = sorted(played_rows, key=lambda r: r["date"])
    window_rows = sorted_rows[-window_size:] if window_size else sorted_rows
    window_games = len(window_rows)
    if window_games == 0:
        return []

    counts = defaultdict(lambda: {"home_games": 0, "home_wins": 0})

    for row in window_rows:
        home = _safe_team(row.get("home_team"))
        if not home:
            continue

        htw = row.get("home_team_won")
        win = None
        if htw is not None:
            s = str(htw).strip().lower()
            if s in {"1", "true", "yes"}:
                win = 1
            elif s in {"0", "false", "no"}:
                win = 0

        counts[home]["home_games"] += 1
        if win == 1:
            counts[home]["home_wins"] += 1

    output = []
    for team, counts_row in counts.items():
        home_games = int(counts_row["home_games"])
        home_wins = int(counts_row["home_wins"])
        home_win_rate = _safe_div(home_wins, home_games) if home_games else 0.0
        output.append(
            {
                "team": team,
                "homeWinRate": float(home_win_rate),
                "homeWins": home_wins,
                "homeGames": home_games,
                "windowGames": int(window_games),
            }
        )

    output.sort(key=lambda x: (x["homeWinRate"], x["homeGames"]), reverse=True)
    return output


def _get_param(params: Dict[str, object], *names: str) -> Optional[float]:
    for name in names:
        key = _normalize_key(name)
        if key in params:
            val = _coerce_value(params[key])
            return float(val) if isinstance(val, (int, float)) else None
    return None


def _human_readable_filters(params: Dict[str, object]) -> str:
    if not params:
        return "No active filters."
    parts = []

    min_prob_used = _get_param(params, "prob_threshold", "min_prob_used", "min_prob", "min_prob_iso")
    min_odds = _get_param(params, "odds_min", "min_odds_1", "min_odds")
    max_odds = _get_param(params, "odds_max", "max_odds_1", "max_odds")
    if max_odds is None:
        max_odds = DEFAULT_MAX_ODDS_FALLBACK
    min_ev = _get_param(params, "min_ev", "min_ev_eur_per_100", "min_ev_per_100")
    min_home_win_rate = _get_param(params, "home_win_rate_threshold", "min_home_win_rate")

    prefer_lower_odds = params.get("prefer_lower_odds")
    if isinstance(prefer_lower_odds, str):
        prefer_lower_odds = prefer_lower_odds.lower() in {"true", "1", "yes"}

    def _format_signed(value: float) -> str:
        raw = f"{int(value)}" if float(value).is_integer() else f"{float(value):.2f}"
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
        parts.append(f"EV ≥ {_format_signed(float(min_ev))}")

    if prefer_lower_odds:
        parts.append("Prefer lower odds")

    return " | ".join(parts) if parts else "No active filters."


def _extract_params_name(raw_params: Dict[str, object]) -> Optional[str]:
    for key in ("params_used", "params_used_type", "params_name", "name"):
        value = raw_params.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, dict):
            for nested_key in ("name", "params_used", "params_used_type", "type"):
                nested_value = value.get(nested_key)
                if isinstance(nested_value, str) and nested_value.strip():
                    return nested_value.strip()
    return None


def _extract_params_label(raw_params: Dict[str, object]) -> Optional[str]:
    for key in ("params_used_label", "label"):
        value = raw_params.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _extract_threshold_params(payload: Dict[str, object]) -> Dict[str, object]:
    if not isinstance(payload, dict):
        return {}
    candidate_maps: List[Dict[str, object]] = [payload]
    for key in ("params", "params_used", "active_params", "thresholds", "filters", "strategy_params"):
        value = payload.get(key)
        if isinstance(value, dict):
            candidate_maps.append(value)
    meta = payload.get("meta")
    if isinstance(meta, dict):
        candidate_maps.append(meta)
        for key in ("params", "params_used", "active_params", "thresholds", "filters", "strategy_params"):
            value = meta.get(key)
            if isinstance(value, dict):
                candidate_maps.append(value)

    best: Dict[str, object] = {}
    best_count = -1
    for candidate in candidate_maps:
        normalized = _normalize_params(candidate)
        found = {
            "home_win_rate_threshold": _get_param(normalized, "home_win_rate_threshold", "min_home_win_rate"),
            "odds_min": _get_param(normalized, "odds_min", "min_odds_1", "min_odds"),
            "odds_max": _get_param(normalized, "odds_max", "max_odds_1", "max_odds"),
            "prob_threshold": _get_param(normalized, "prob_threshold", "min_prob_used", "min_prob", "min_prob_iso"),
            "min_ev": _get_param(normalized, "min_ev", "min_ev_eur_per_100", "min_ev_per_100"),
        }
        count = sum(value is not None for value in found.values())
        if count > best_count:
            best_count = count
            best = {k: v for k, v in found.items() if v is not None}
    return best


def _threshold_value_is_valid(key: str, value: object) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    numeric = float(value)
    if not math.isfinite(numeric):
        return False
    if key == "home_win_rate_threshold":
        return 0.0 <= numeric <= 1.0
    if key == "prob_threshold":
        return 0.0 <= numeric <= 1.0
    if key == "odds_min":
        return 1.0 <= numeric <= 20.0
    if key == "odds_max":
        return 1.0 <= numeric <= 20.0
    if key == "min_ev":
        return -200.0 <= numeric <= 200.0
    return True


def _sanitize_threshold_params(raw: Dict[str, object]) -> Tuple[Dict[str, float], List[str]]:
    sanitized: Dict[str, float] = {}
    invalid_keys: List[str] = []
    for key in ("home_win_rate_threshold", "odds_min", "odds_max", "prob_threshold", "min_ev"):
        if key not in raw:
            continue
        value = raw.get(key)
        if _threshold_value_is_valid(key, value):
            sanitized[key] = float(value)
        else:
            invalid_keys.append(key)

    odds_min = sanitized.get("odds_min")
    odds_max = sanitized.get("odds_max")
    if odds_min is not None and odds_max is not None and odds_max < odds_min:
        invalid_keys.extend(["odds_min", "odds_max"])
        sanitized.pop("odds_min", None)
        sanitized.pop("odds_max", None)

    return sanitized, sorted(set(invalid_keys))


def _resolve_canonical_threshold_params(
    metrics_snapshot_path: Optional[Path],
    strategy_params: Dict[str, object],
    strategy_params_path: Optional[Path],
) -> Tuple[Dict[str, object], Dict[str, object]]:
    strategy_thresholds_raw = {
        key: value
        for key, value in {
            "home_win_rate_threshold": _get_param(strategy_params, "home_win_rate_threshold", "min_home_win_rate"),
            "odds_min": _get_param(strategy_params, "odds_min", "min_odds_1", "min_odds"),
            "odds_max": _get_param(strategy_params, "odds_max", "max_odds_1", "max_odds"),
            "prob_threshold": _get_param(strategy_params, "prob_threshold", "min_prob_used", "min_prob", "min_prob_iso"),
            "min_ev": _get_param(strategy_params, "min_ev", "min_ev_eur_per_100", "min_ev_per_100"),
        }.items()
        if value is not None
    }
    strategy_thresholds, strategy_invalid_keys = _sanitize_threshold_params(strategy_thresholds_raw)
    metrics_thresholds: Dict[str, object] = {}
    metrics_invalid_keys: List[str] = []
    if metrics_snapshot_path and metrics_snapshot_path.exists():
        try:
            payload = json.loads(metrics_snapshot_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                metrics_thresholds_raw = _extract_threshold_params(payload)
                metrics_thresholds, metrics_invalid_keys = _sanitize_threshold_params(metrics_thresholds_raw)
        except (json.JSONDecodeError, OSError):
            metrics_thresholds = {}

    strategy_is_newer = False
    if strategy_params_path and metrics_snapshot_path and strategy_params_path.exists() and metrics_snapshot_path.exists():
        strategy_is_newer = strategy_params_path.stat().st_mtime >= metrics_snapshot_path.stat().st_mtime

    if metrics_thresholds and not strategy_is_newer:
        canonical = dict(strategy_thresholds)
        canonical.update(metrics_thresholds)
        params_source_file = _metadata_path(metrics_snapshot_path)
        params_source_type = "metrics_snapshot"
        params_used_value = "from_metrics_snapshot"
    else:
        canonical = dict(metrics_thresholds)
        canonical.update(strategy_thresholds)
        params_source_file = _metadata_path(strategy_params_path) if strategy_params_path else _metadata_path(metrics_snapshot_path)
        params_source_type = (
            "strategy_params_dated"
            if strategy_params_path and re.search(r"strategy_params_\d{4}-\d{2}-\d{2}\.(json|txt)$", strategy_params_path.name)
            else "strategy_params"
            if strategy_params_path
            else "metrics_snapshot"
            if metrics_thresholds
            else "default"
        )
        params_used_value = "from_file" if strategy_params_path else "from_metrics_snapshot" if metrics_thresholds else "fallback"
    missing = [
        key
        for key in ("home_win_rate_threshold", "odds_min", "odds_max", "prob_threshold", "min_ev")
        if key not in canonical
    ]
    if missing and params_used_value == "from_metrics_snapshot":
        params_used_value = "from_metrics_snapshot_with_strategy_fallback"
    elif missing and params_used_value != "fallback":
        params_used_value = f"{params_used_value}_with_fallback"
    metadata = {
        "params_source_file": params_source_file,
        "params_source_type": params_source_type,
        "params_used": params_used_value,
        "fallback_used": bool(missing),
        "fallback_reason": "" if not missing else f"missing_threshold_keys:{','.join(missing)}",
        "strategy_invalid_keys": strategy_invalid_keys,
        "metrics_invalid_keys": metrics_invalid_keys,
    }
    if strategy_invalid_keys:
        metadata["fallback_used"] = True
        reason = f"invalid_strategy_threshold_keys:{','.join(strategy_invalid_keys)}"
        metadata["fallback_reason"] = f"{metadata['fallback_reason']}; {reason}".strip("; ")
    if metrics_invalid_keys:
        metadata["fallback_used"] = True
        reason = f"invalid_metrics_threshold_keys:{','.join(metrics_invalid_keys)}"
        metadata["fallback_reason"] = f"{metadata['fallback_reason']}; {reason}".strip("; ")
    return canonical, metadata


def _extract_max_date_from_csv(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    candidates: List[str] = []
    for row in _read_csv_normalized(path):
        for key in ("date", "game_date", "as_of_date", "snapshot_as_of_date", "settled_date", "bet_date", "placed_date"):
            value = row.get(key)
            if not isinstance(value, str):
                continue
            match = re.search(r"(\d{4}-\d{2}-\d{2})", value.strip())
            if match:
                candidates.append(match.group(1))
                break
    return sorted(candidates)[-1] if candidates else None


def _build_active_params(params: Dict[str, object], window_size: int) -> Dict[str, float]:
    home_win_rate_min = _get_param(params, "home_win_rate_threshold", "min_home_win_rate")
    odds_min = _get_param(params, "odds_min", "min_odds_1", "min_odds")
    odds_max = _get_param(params, "odds_max", "max_odds_1", "max_odds")
    prob_threshold = _get_param(params, "prob_threshold", "min_prob_used", "min_prob", "min_prob_iso")
    min_ev = _get_param(params, "min_ev", "min_ev_eur_per_100", "min_ev_per_100")

    return {
        "home_win_rate_min": float(home_win_rate_min) if home_win_rate_min is not None else 0.0,
        "odds_min": float(odds_min) if odds_min is not None else 1.0,
        "odds_max": float(odds_max) if odds_max is not None else DEFAULT_MAX_ODDS_FALLBACK,
        "prob_threshold": float(prob_threshold) if prob_threshold is not None else 0.5,
        "min_ev": float(min_ev) if min_ev is not None else 0.0,
        "window_size": float(window_size),
    }


def _choose_canonical_bet_log(
    lightgbm_bet_log: Optional[Path],
    source_root: Optional[Path],
) -> Tuple[Optional[Path], Optional[str], str, List[str]]:
    candidates: List[Tuple[Path, str]] = []
    if lightgbm_bet_log and lightgbm_bet_log.exists():
        candidates.append((lightgbm_bet_log, "lightgbm_live"))
    if source_root:
        root_bet_log = source_root / "bet_log" / "bet_log_flat_live.csv"
        if root_bet_log.exists():
            candidates.append((root_bet_log, "root_bet_log_live"))

    best_path: Optional[Path] = None
    best_type = "missing"
    best_date: Optional[str] = None
    issues: List[str] = []
    latest_by_candidate: Dict[Path, Optional[str]] = {}
    for path, source_type in candidates:
        latest = _extract_date_from_name(path.name) or _extract_max_date_from_csv(path)
        latest_by_candidate[path] = latest
        if latest is None:
            continue
        if best_date is None or latest > best_date:
            best_path = path
            best_type = source_type
            best_date = latest
    if best_path is None and candidates:
        best_path, best_type = candidates[0]
        issues.append("canonical bet_log selected without parsable date; freshness comparison unavailable")
    if best_path is not None:
        selected_date = latest_by_candidate.get(best_path)
        for candidate, latest in latest_by_candidate.items():
            if candidate == best_path or latest is None or selected_date is None:
                continue
            if latest > selected_date:
                issues.append(
                    f"newer bet_log candidate exists ({candidate.name}: {latest}) than selected {best_path.name}: {selected_date}"
                )
    return best_path, best_date, best_type, issues


def _compute_local_bankroll(rows: List[Dict[str, object]], start: float, stake: float) -> Dict[str, float]:
    net_pl = sum((row.get("pnl") or 0.0) for row in rows)
    return {"start": float(start), "stake": float(stake), "net_pl": float(net_pl), "bankroll": float(start + net_pl)}


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


def build_local_equity_history(rows: List[Dict[str, object]], start: float) -> List[Dict[str, object]]:
    if not rows:
        return []
    sorted_rows = sorted(rows, key=lambda r: r["date"])
    balance = float(start)
    history = []
    for row in sorted_rows:
        pnl = float(row.get("pnl") or 0.0)
        balance += pnl
        history.append({"date": row["date"], "balance": balance, "betsPlaced": 1, "profit": pnl})
    return history


def build_bankroll_history(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    if not rows:
        return []
    per_day = defaultdict(lambda: {"profit": 0.0, "bets": 0, "balance": None})

    for r in rows:
        if not isinstance(r.get("date"), datetime):
            continue
        d = r["date"].strftime(DATE_FMT)
        per_day[d]["profit"] += float(r.get("profit_eur") or 0.0)
        per_day[d]["bets"] += 1
        bal = r.get("bankroll_after")
        if bal is None:
            bal = r.get("bankroll")
        per_day[d]["balance"] = bal if bal is not None else per_day[d]["balance"]

    output = []
    for d in sorted(per_day.keys()):
        output.append(
            {
                "date": d,
                "balance": float(per_day[d]["balance"] or 0.0),
                "betsPlaced": int(per_day[d]["bets"]),
                "profit": float(per_day[d]["profit"]),
            }
        )
    return output


def compute_max_drawdown(history: List[Dict[str, object]]) -> Tuple[float, float]:
    if not history:
        return 0.0, 0.0
    balances = [h["balance"] for h in history if h.get("balance") is not None]
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
    return float(max_dd), float(max_dd_pct)


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


def _label_path(path: Optional[Path]) -> str:
    if not path:
        return "missing"
    return path.name


def _metadata_path(path: Optional[Path]) -> Optional[str]:
    if not path:
        return None
    try:
        return str(path.resolve())
    except OSError:
        return str(path)


def load_bet_log(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []

    for row in _read_csv_normalized(path):
        date = _safe_date(
            row.get("date")
            or row.get("game_date")
            or row.get("settled_date")
            or row.get("bet_date")
            or row.get("placed_date")
        )
        if date is None:
            continue

        status_raw = (row.get("status") or "").strip().upper()
        has_status = bool(status_raw)

        stake = _safe_float(row.get("stake") or row.get("stake_eur") or row.get("stake_flat"))
        pnl = _safe_float(row.get("pnl") or row.get("profit_eur") or row.get("profit"))
        won = _safe_float(row.get("won"))

        if won is None and pnl is not None:
            if pnl > 0:
                won = 1.0
            elif pnl < 0:
                won = 0.0

        if has_status:
            if status_raw != "SETTLED":
                continue
        else:
            if won is None and pnl is None:
                continue

        odds_1 = _safe_float(
            row.get("odds_1")
            or row.get("odds")
            or row.get("closing_home_odds")
            or row.get("home_odds")
        )

        rows.append(
            {
                "date": date,
                "stake_eur": stake,
                "profit_eur": pnl,
                "won": won,
                "odds_1": odds_1,
                "home_team": _safe_team(row.get("home_team")),
                "away_team": _safe_team(row.get("away_team")),
                "status": status_raw if has_status else None,
                "bankroll": _safe_float(row.get("bankroll")),
                "bankroll_after": _safe_float(row.get("bankroll_after")),
            }
        )
    deduped: List[Dict[str, object]] = []
    seen: set[Tuple[str, Optional[str], Optional[str]]] = set()
    for row in rows:
        date = row.get("date")
        if not isinstance(date, datetime):
            continue
        key = (
            date.strftime(DATE_FMT),
            row.get("home_team") if isinstance(row.get("home_team"), str) else None,
            row.get("away_team") if isinstance(row.get("away_team"), str) else None,
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def trim_bet_log_to_snapshot(rows: List[Dict[str, object]], snapshot_date: Optional[str]) -> List[Dict[str, object]]:
    snapshot_dt = _safe_date(snapshot_date)
    if snapshot_dt is None:
        return rows
    return [row for row in rows if isinstance(row.get("date"), datetime) and row["date"] <= snapshot_dt]


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
            "avgOdds": 0.0,
        }
    total_bets = len(rows)
    total_staked = sum(float(r.get("stake_eur") or 0.0) for r in rows)
    total_profit = sum(float(r.get("profit_eur") or 0.0) for r in rows)
    avg_stake = _safe_div(total_staked, total_bets)
    avg_profit = _safe_div(total_profit, total_bets)
    win_rate = _safe_div(sum(1 for r in rows if float(r.get("won") or 0.0) == 1.0), total_bets)
    odds_values = [float(r.get("odds_1")) for r in rows if r.get("odds_1") is not None]
    avg_odds = _safe_div(sum(odds_values), len(odds_values)) if odds_values else 0.0
    as_of = max(r["date"] for r in rows).strftime(DATE_FMT)
    roi_pct = _safe_div(total_profit, total_staked) * 100.0
    return {
        "asOfDate": as_of,
        "totalBets": int(total_bets),
        "totalStakedEur": float(total_staked),
        "totalProfitEur": float(total_profit),
        "roiPct": float(roi_pct),
        "avgStakeEur": float(avg_stake),
        "avgProfitPerBetEur": float(avg_profit),
        "winRate": float(win_rate),
        "avgOdds": float(avg_odds),
    }


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, default=_serialize, ensure_ascii=False, indent=2)


def remove_legacy_typo_files(output_dir: Path) -> None:
    typo_files = ("dashoard_payload.json", "dashoard_state.json")
    for typo_name in typo_files:
        typo_path = output_dir / typo_name
        if typo_path.exists():
            typo_path.unlink()
            LOGGER.warning("Removed legacy typo artifact: %s", typo_path)


def verify_required_dashboard_json(output_dir: Path) -> None:
    missing = [name for name in REQUIRED_DASHBOARD_JSON if not (output_dir / name).exists()]
    if missing:
        missing_list = ", ".join(missing)
        raise FileNotFoundError(f"Missing required dashboard JSON files in {output_dir}: {missing_list}")


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
    parser.add_argument(
        "--combined-path",
        type=str,
        default=None,
        help="Explicit path to a combined predictions CSV to use instead of auto-discovery.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    data_dir = Path(args.data_dir) if args.data_dir else None
    source_root = resolve_source_root(args.source_root, repo_root)

    output_dir = Path(args.output_dir) if args.output_dir else repo_root / "public" / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    today_games_payload = _build_today_games_payload(source_root)

    # ----------------------------
    # Resolve sources
    # ----------------------------
    selection_metadata = {
        "snapshot_as_of_date": None,
        "run_date": None,
        "combined_source_file": None,
        "local_matched_source_file": None,
        "bet_log_source_file": None,
        "bet_log_source_type": "missing",
        "metrics_source_file": None,
        "strategy_params_source_file": None,
        "params_source_file": None,
        "params_source_type": "unknown",
        "params_used": "fallback",
        "bet_log_latest_date_in_file": None,
        "bet_log_trimmed_to_snapshot": False,
        "fallback_used": False,
        "fallback_reason": "",
    }

    if data_dir:
        strategy_json = data_dir / "strategy_params.json"
        strategy_txt = data_dir / "strategy_params.txt"

        sources = SourcePaths(
            combined_iso=_resolve_combined_file_from_data_dir(data_dir),
            combined_acc=None,
            bet_log=None,
            bet_log_flat=(data_dir / "bet_log_flat_live.csv")
            if (data_dir / "bet_log_flat_live.csv").exists()
            else None,
            local_matched_games=(data_dir / "local_matched_games_latest.csv")
            if (data_dir / "local_matched_games_latest.csv").exists()
            else None,
            strategy_params=(
                strategy_json
                if strategy_json.exists()
                else strategy_txt
                if strategy_txt.exists()
                else None
            ),
            metrics_snapshot=(data_dir / "metrics_snapshot.json")
            if (data_dir / "metrics_snapshot.json").exists()
            else None,
        )
        if sources.combined_iso:
            selection_metadata["combined_source_file"] = sources.combined_iso.name
            selection_metadata["snapshot_as_of_date"] = _extract_date_from_name(sources.combined_iso.name)
            selection_metadata["run_date"] = selection_metadata["snapshot_as_of_date"]
        if sources.local_matched_games:
            selection_metadata["local_matched_source_file"] = sources.local_matched_games.name
        if sources.bet_log_flat:
            selection_metadata["bet_log_source_file"] = sources.bet_log_flat.name
            selection_metadata["bet_log_source_type"] = "lightgbm_live"
        if sources.metrics_snapshot:
            selection_metadata["metrics_source_file"] = sources.metrics_snapshot.name
        if sources.strategy_params:
            selection_metadata["strategy_params_source_file"] = sources.strategy_params.name
    else:
        if source_root is None:
            raise FileNotFoundError("Could not resolve SOURCE_ROOT for snapshot selection.")
        selection = resolve_snapshot_selection(source_root)
        copy_selection_aliases(selection, output_dir)
        selection_metadata = {
            "snapshot_as_of_date": selection.snapshot_as_of_date,
            "run_date": selection.run_date,
            "combined_source_file": selection.combined_source_file,
            "local_matched_source_file": selection.local_matched_source_file,
            "bet_log_source_file": selection.bet_log_source_file,
            "bet_log_source_type": "snapshot_selection",
            "metrics_source_file": selection.metrics_source_file,
            "strategy_params_source_file": selection.strategy_params_source_file,
            "params_source_file": None,
            "params_source_type": selection.params_source_type,
            "params_used": "from_file",
            "bet_log_latest_date_in_file": selection.bet_log_latest_date,
            "bet_log_trimmed_to_snapshot": selection.bet_log_will_be_trimmed_to_snapshot,
            "fallback_used": selection.fallback_used,
            "fallback_reason": selection.fallback_reason,
        }
        sources = SourcePaths(
            combined_iso=selection.combined_path,
            combined_acc=None,
            bet_log=None,
            bet_log_flat=selection.bet_log_path,
            local_matched_games=selection.local_matched_path,
            strategy_params=selection.strategy_params_path,
            metrics_snapshot=selection.metrics_path,
        )

    combined_override = Path(args.combined_path).expanduser().resolve() if args.combined_path else None
    if combined_override and not combined_override.exists():
        raise FileNotFoundError(f"--combined-path file does not exist: {combined_override}")

    if combined_override:
        combined_path = combined_override
    elif sources.combined_iso:
        combined_path = sources.combined_iso
    elif sources.combined_acc:
        combined_path = sources.combined_acc
    else:
        raise FileNotFoundError("No combined predictions file found in source output directories.")

    played_rows = load_played_games_with_history(combined_path)
    if not played_rows:
        raise RuntimeError("No played games found in combined predictions file.")

    # ----------------------------
    # Model stats
    # ----------------------------
    historical_stats = build_historical_stats(played_rows)
    accuracy_thresholds = build_accuracy_thresholds(played_rows)
    calibration = build_calibration_metrics(played_rows, window_size=CALIBRATION_WINDOW)
    home_win_rates = build_home_win_rates_last20(played_rows)
    HOME_WIN_RATE_WINDOW = 20
    home_win_rates_window = build_home_win_rates_window(
        played_rows,
        window_size=HOME_WIN_RATE_WINDOW,
    )

    # ----------------------------
    # Bet log (optional)
    # ----------------------------
    bet_log_rows: List[Dict[str, object]] = []
    bet_log_path = None

    bet_log_path, canonical_bet_log_date, canonical_bet_log_type, bet_log_source_issues = _choose_canonical_bet_log(
        sources.bet_log_flat if sources.bet_log_flat and sources.bet_log_flat.exists() else None,
        source_root,
    )
    if bet_log_path is None and sources.bet_log and sources.bet_log.exists():
        bet_log_path = sources.bet_log
        canonical_bet_log_type = "lightgbm_raw"
    selection_metadata["bet_log_source_file"] = _metadata_path(bet_log_path)
    selection_metadata["bet_log_source_type"] = canonical_bet_log_type
    if canonical_bet_log_date:
        selection_metadata["bet_log_latest_date_in_file"] = canonical_bet_log_date
    consistency_issues: List[str] = list(bet_log_source_issues)
    data_warnings: List[str] = []

    if bet_log_path:
        bet_log_rows = load_bet_log(bet_log_path)
        if selection_metadata.get("bet_log_latest_date_in_file") is None and bet_log_rows:
            selection_metadata["bet_log_latest_date_in_file"] = max(row["date"] for row in bet_log_rows).strftime(DATE_FMT)
        original_bet_log_count = len(bet_log_rows)
        bet_log_rows = trim_bet_log_to_snapshot(bet_log_rows, selection_metadata.get("snapshot_as_of_date"))
        selection_metadata["bet_log_trimmed_to_snapshot"] = (
            bool(selection_metadata.get("snapshot_as_of_date")) and len(bet_log_rows) != original_bet_log_count
        )

    bet_log_summary = build_bet_log_summary(bet_log_rows)
    bankroll_history = build_bankroll_history(bet_log_rows)
    settled_bets_rows = []
    for row in bet_log_rows:
        bet_date = row.get("date")
        if not isinstance(bet_date, datetime) or bet_date < datetime(2026, 1, 1):
            continue
        pnl = row.get("profit_eur")
        win = row.get("won")
        if pnl is None or win is None:
            continue
        settled_bets_rows.append(
            {
                "date": bet_date.strftime(DATE_FMT),
                "home_team": row.get("home_team") or "",
                "away_team": row.get("away_team") or "",
                "pick_team": "HOME",
                "odds": float(row.get("odds_1") or 0.0),
                "stake": float(row.get("stake_eur") or 0.0),
                "win": int(win),
                "pnl": float(pnl),
            }
        )

    # ----------------------------
    # Window bounds (last N played games) - must match validator
    # ----------------------------
    window_played_rows, window_start_dt, window_end_dt = compute_window_bounds(played_rows, CALIBRATION_WINDOW)
    as_of_date = window_end_dt.strftime(DATE_FMT) if window_end_dt else "—"
    window_start_label = window_start_dt.strftime(DATE_FMT) if window_start_dt else None
    window_end_label = window_end_dt.strftime(DATE_FMT) if window_end_dt else None

    snapshot_as_of_date = selection_metadata.get("snapshot_as_of_date") or as_of_date
    combined_source_file = combined_path.name
    bet_log_latest_date_raw = selection_metadata.get("bet_log_latest_date_in_file")
    if isinstance(bet_log_latest_date_raw, str) and snapshot_as_of_date != "—":
        try:
            bet_log_latest_dt = datetime.strptime(bet_log_latest_date_raw, DATE_FMT)
            snapshot_dt = datetime.strptime(snapshot_as_of_date, DATE_FMT)
            day_gap = (snapshot_dt - bet_log_latest_dt).days
            if day_gap > 14:
                consistency_issues.append(
                    f"bet_log appears stale versus snapshot ({day_gap} days behind: {bet_log_latest_date_raw} vs {snapshot_as_of_date})"
                )
        except ValueError:
            consistency_issues.append(
                f"bet_log_latest_date_in_file is not parseable as {DATE_FMT}: {bet_log_latest_date_raw}"
            )

    # ----------------------------
    # Load versioned params (shared strategy module)
    # ----------------------------
    strategy_params_source = None
    strategy_params_source_type = "missing"
    strategy_params_parsed_ok = False
    strategy_params_parse_status = "missing"
    strategy_params_parse_error: Optional[str] = None
    defaults_used = False
    defaults_reason: Optional[str] = None
    params_payload_path: Optional[Path] = None
    raw_strategy_payload: Dict[str, object] = {}
    if sources.strategy_params and sources.strategy_params.exists():
        params_payload_path = sources.strategy_params
    else:
        fallback_params = output_dir / "strategy_params.json"
        if fallback_params.exists():
            params_payload_path = fallback_params

    strategy_params_name = "fallback"
    params_used_label = "fallback"
    try:
        strategy_params_obj = load_versioned_strategy_params(params_payload_path)
        params_used = _normalize_params(strategy_params_obj.params)
        strategy_params_source = Path(strategy_params_obj.source) if strategy_params_obj.source != "defaults" else None
        strategy_params_source_type = "file" if strategy_params_source else "defaults"
        params_used_label = f"v{strategy_params_obj.version}"
        strategy_params_parse_status = "ok" if strategy_params_source else "defaults"
        defaults_used = strategy_params_source is None
        strategy_params_parsed_ok = strategy_params_source is not None
        if defaults_used:
            defaults_reason = "strategy_params_missing"
        if params_payload_path and params_payload_path.exists():
            raw_strategy_payload = load_strategy_params(params_payload_path)
            strategy_params_name = _extract_params_name(raw_strategy_payload) or (
                "from_file" if strategy_params_source else strategy_params_name
            )
            params_used_label = _extract_params_label(raw_strategy_payload) or params_used_label
    except (ValueError, json.JSONDecodeError, TypeError) as exc:
        LOGGER.warning("Unable to parse strategy params at %s: %s. Falling back to defaults.", params_payload_path, exc)
        strategy_params_obj = load_versioned_strategy_params(None)
        params_used = _normalize_params(strategy_params_obj.params)
        params_used_label = "fallback"
        strategy_params_source = None
        strategy_params_source_type = "parse_error"
        strategy_params_parse_status = "parse_error"
        strategy_params_parse_error = str(exc)
        defaults_used = True
        strategy_params_parsed_ok = False
        defaults_reason = "strategy_params_parse_error"
        consistency_issues.append(f"strategy_params parse error: {exc}")

    canonical_params, params_resolution = _resolve_canonical_threshold_params(
        sources.metrics_snapshot,
        params_used,
        strategy_params_source,
    )
    params_for_display = dict(params_used)
    params_for_display.update(canonical_params)
    active_params = _build_active_params(params_for_display, CALIBRATION_WINDOW)
    active_filters_label = _human_readable_filters(params_for_display)
    selection_metadata["params_source_file"] = params_resolution["params_source_file"]
    selection_metadata["params_source_type"] = params_resolution["params_source_type"]
    selection_metadata["params_used"] = params_resolution["params_used"]
    if defaults_used and params_resolution["params_source_type"] == "default":
        strategy_params_name = "fallback"
    if params_resolution["fallback_used"]:
        selection_metadata["fallback_used"] = True
        selection_metadata["fallback_reason"] = (
            f"{selection_metadata.get('fallback_reason')}; {params_resolution['fallback_reason']}".strip("; ")
        )
    metrics_invalid_keys = params_resolution.get("metrics_invalid_keys") or []
    strategy_invalid_keys = params_resolution.get("strategy_invalid_keys") or []
    if metrics_invalid_keys:
        consistency_issues.append(
            f"ignored invalid metrics snapshot thresholds: {','.join(metrics_invalid_keys)}"
        )
    if strategy_invalid_keys:
        consistency_issues.append(
            f"ignored invalid strategy thresholds: {','.join(strategy_invalid_keys)}"
        )

    expected_active_params = {
        "home_win_rate_min": float(params_for_display.get("home_win_rate_threshold", 0.0)),
        "odds_min": float(params_for_display.get("odds_min", 1.0)),
        "odds_max": float(params_for_display.get("odds_max", DEFAULT_MAX_ODDS_FALLBACK)),
        "prob_threshold": float(params_for_display.get("prob_threshold", 0.5)),
        "min_ev": float(params_for_display.get("min_ev", 0.0)),
    }
    for key, expected in expected_active_params.items():
        got = float(active_params.get(key, expected))
        if not math.isclose(got, expected, rel_tol=1e-9, abs_tol=1e-9):
            consistency_issues.append(f"active_params mismatch for {key}: expected {expected}, got {got}")

    today_games_payload["ev_exception_profitability"] = _build_ev_exception_profitability(
        today_games_payload,
        played_rows,
        active_params,
        window_start_dt,
        window_end_dt,
        output_dir,
    )
    historical_roi_attack_summaries = _find_historical_roi_attack_summaries(
        _resolve_live_lightgbm_dir(source_root),
        today_games_payload.get("as_of_date") if isinstance(today_games_payload.get("as_of_date"), str) else None,
    )
    local_profitability_rule = _build_local_profitability_rule_payload(
        historical_roi_attack_summaries,
        active_params,
    )
    today_games_payload["historical_roi_attack_scans"] = historical_roi_attack_summaries
    today_games_payload["local_profitability_rule"] = local_profitability_rule
    today_games_payload["local_strategy_evaluation_window"] = _build_local_strategy_evaluation_window(
        raw_strategy_payload,
        strategy_params_source,
        active_params,
    )

    # ----------------------------
    # Strategy matches (MUST match validate_dashboard_state.mjs)
    # Source of truth: public/data/local_matched_games_latest.csv filtered by window + params_used
    # ----------------------------
    local_matched_games_rows_all: List[Dict[str, object]] = []

    local_matched_games_path: Optional[Path] = None
    local_latest_path = output_dir / "local_matched_games_latest.csv"
    if local_latest_path.exists():
        local_matched_games_path = local_latest_path
    elif sources.local_matched_games and sources.local_matched_games.exists():
        local_matched_games_path = sources.local_matched_games

    local_summary: Dict[str, object] = {}
    local_matched_schema_valid = False
    local_matched_parse_error: Optional[str] = None
    if local_matched_games_path and local_matched_games_path.exists():
        try:
            local_matched_games_rows_all, _local_summary = load_local_matched_games_csv(local_matched_games_path)
            local_summary = _local_summary
            local_matched_schema_valid = True
        except Exception as exc:
            local_matched_parse_error = str(exc)
            consistency_issues.append(f"local_matched_games parse error: {exc}")
    else:
        consistency_issues.append("local_matched_games source file missing")

    local_rows_max_date = _max_date_from_local_rows(local_matched_games_rows_all)
    canonical_snapshot_date = selection_metadata.get("snapshot_as_of_date") or snapshot_as_of_date
    if local_rows_max_date and canonical_snapshot_date != "—" and local_rows_max_date > canonical_snapshot_date:
        consistency_issues.append(
            f"local_matched_games max date {local_rows_max_date} is after combined snapshot {canonical_snapshot_date}"
        )
    elif local_rows_max_date is None and int(local_summary.get("source_rows_count", 0)) > 0:
        consistency_issues.append("local_matched_games has no valid date rows")

    def _in_window(row: Dict[str, object]) -> bool:
        d = _safe_date(row.get("date"))
        if not d or not window_start_dt or not window_end_dt:
            return False
        return window_start_dt <= d <= window_end_dt

    window_filtered_local_rows = [r for r in local_matched_games_rows_all if _in_window(r)]

    # Params filter from shared strategy module
    window_local_df = pd.DataFrame(window_filtered_local_rows)
    strategy_params_obj.params.update({k: float(v) for k, v in canonical_params.items() if isinstance(v, (int, float))})
    filtered_local_df, shared_params = apply_strategy_filters(window_local_df, strategy_params_obj)
    local_matched_games_rows = filtered_local_df.where(pd.notna(filtered_local_df), None).to_dict(orient="records")
    local_matched_games_count = int(len(local_matched_games_rows))

    # Profit sum should reflect emitted (filtered) rows
    local_matched_games_profit_sum = float(sum((row.get("pnl") or 0.0) for row in local_matched_games_rows))

    # --- ROI (simulated strategy) ---
    # Use stake column if present/finite; otherwise fallback to 100.0 (flat stake)
    stake_values: List[float] = []
    for row in local_matched_games_rows:
        stake_val = _safe_float(row.get("stake"))
        if stake_val is None or not math.isfinite(stake_val):
            continue
        stake_values.append(stake_val)

    flat_stake = float(stake_values[0] if stake_values else 100.0)
    stake_sum = float(local_matched_games_count * flat_stake)
    strategy_roi = _safe_div(local_matched_games_profit_sum, stake_sum) if local_matched_games_count else 0.0
    strategy_roi_pct = round(float(strategy_roi * 100.0), 2)

    local_sharpe = _compute_sharpe_style(local_matched_games_rows) if local_matched_games_rows else None
    bankroll_last_200 = _compute_local_bankroll(local_matched_games_rows, 1000.0, 100.0)

    ytd_rows = [
        row
        for row in local_matched_games_rows
        if (_safe_date(row.get("date")) is not None and _safe_date(row.get("date")) >= datetime(2026, 1, 1))
    ]
    bankroll_ytd_2026 = _compute_local_bankroll(ytd_rows, 1000.0, 100.0)

    _, local_max_dd_eur, local_max_dd_pct = compute_local_risk_metrics(local_matched_games_rows, 1000.0, RISK_MIN_SAMPLE)

    strategy_summary = {
        "totalBets": int(local_matched_games_count),
        "totalProfitEur": float(local_matched_games_profit_sum),
        "roiPct": float(strategy_roi_pct),
        "avgEvPer100": float(
            _safe_div(
                sum((row.get("ev_eur_per_100") or 0.0) for row in local_matched_games_rows),
                local_matched_games_count,
            )
        )
        if local_matched_games_count
        else 0.0,
        "winRate": float(
            _safe_div(
                sum(1 for row in local_matched_games_rows if row.get("win") == 1),
                local_matched_games_count,
            )
        )
        if local_matched_games_count
        else 0.0,
        "sharpeStyle": local_sharpe,
        "maxDrawdownEur": local_max_dd_eur,
        "maxDrawdownPct": local_max_dd_pct,
        "bankrollLast200": bankroll_last_200,
        "bankrollYtd2026": bankroll_ytd_2026,
        "asOfDate": window_end_label or as_of_date,
    }

    # This is what validate_dashboard_state.mjs checks:
    strategy_matches_window = int(local_matched_games_count)

    window_size_label = int(CALIBRATION_WINDOW)
    window_start_display = window_start_label or "—"
    window_end_display = window_end_label or "—"
    active_filters_text = f"{active_filters_label} | window {window_size_label} ({window_start_display} → {window_end_display})"

    bet_log_flat_path = bet_log_path
    strategy_date_from_source = _extract_date_from_name(strategy_params_source.name) if strategy_params_source else None
    if strategy_date_from_source is None and strategy_params_source:
        strategy_date_from_source = _extract_date_from_json(strategy_params_source)
    if strategy_date_from_source and canonical_snapshot_date != "—" and strategy_date_from_source > canonical_snapshot_date:
        consistency_issues.append(
            f"strategy_params date {strategy_date_from_source} is after combined snapshot {canonical_snapshot_date}"
        )
    if strategy_params_source and defaults_used:
        consistency_issues.append("strategy params source exists but defaults were used")
    if not defaults_used and not strategy_params_parsed_ok:
        consistency_issues.append("strategy params parsed state inconsistent")
    if bet_log_path and not selection_metadata.get("bet_log_latest_date_in_file"):
        consistency_issues.append("bet_log source selected but latest date could not be determined")

    metrics_snapshot_date = _extract_date_from_json(sources.metrics_snapshot) or (
        _extract_date_from_name(sources.metrics_snapshot.name) if sources.metrics_snapshot else None
    )
    if metrics_snapshot_date and canonical_snapshot_date != "—" and metrics_snapshot_date != canonical_snapshot_date:
        consistency_issues.append(
            f"metrics_snapshot date {metrics_snapshot_date} does not match combined snapshot {canonical_snapshot_date}"
        )

    data_consistency_status = "ok" if not consistency_issues else "out_of_sync"

    dashboard_state = {
        "as_of_date": as_of_date,
        "snapshot_as_of_date": snapshot_as_of_date,
        "window_size": int(window_size_label),
        "window_start": window_start_label,
        "window_end": window_end_label,
        "active_filters_text": active_filters_text,
        "params_used_label": params_used_label,
        "active_params": active_params,
        "params_source_label": selection_metadata.get("params_source_file") or _label_path(strategy_params_source),
        "strategy_params_parse_status": strategy_params_parse_status,
        "strategy_params_parse_error": strategy_params_parse_error,
        "defaults_used": defaults_used,
        "defaults_reason": defaults_reason,
        "strategy_as_of_date": (window_end_dt.strftime(DATE_FMT) if window_end_dt else None),
        "strategy_matches_window": strategy_matches_window,
        "local_matched_schema_valid": local_matched_schema_valid,
        "local_matched_parse_error": local_matched_parse_error,
        "data_consistency_status": data_consistency_status,
        "data_consistency_issues": consistency_issues,
        "data_warnings": data_warnings,
        "combined_source_file": combined_source_file,
        "local_matched_source_file": _label_path(local_matched_games_path),
        "bet_log_source_file": _metadata_path(bet_log_path),
        "bet_log_source_type": selection_metadata.get("bet_log_source_type"),
        "bet_log_latest_date_in_file": selection_metadata.get("bet_log_latest_date_in_file"),
        "bet_log_trimmed_to_snapshot": bool(selection_metadata.get("bet_log_trimmed_to_snapshot")),
        "strategy_params_source_file": _metadata_path(strategy_params_source),
        "metrics_snapshot_source_file": selection_metadata.get("metrics_source_file") or _label_path(sources.metrics_snapshot),
        "run_date": selection_metadata.get("run_date") or as_of_date,
        "strategy_params_source_type": strategy_params_source_type,
        "strategy_params_parsed_ok": strategy_params_parsed_ok,
        "params_source_file": selection_metadata.get("params_source_file"),
        "params_source_type": selection_metadata.get("params_source_type") or strategy_params_source_type,
        "params_used": selection_metadata.get("params_used") or strategy_params_name,
        "fallback_used": bool(selection_metadata.get("fallback_used") or defaults_used),
        "fallback_reason": selection_metadata.get("fallback_reason") or defaults_reason or "",
        "snapshot_fallback_used": bool(selection_metadata.get("fallback_used")),
        "snapshot_fallback_reason": selection_metadata.get("fallback_reason"),
        "last_update_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sources": {
            "combined": _label_path(combined_path),
            "local_matched": _label_path(local_matched_games_path),
            "bet_log": _label_path(bet_log_flat_path),
        },
    }

    # ----------------------------
    # Payloads (keep simple + stable)
    # ----------------------------

    # Tables expected by validate_dashboard_payload.py + SPEC.md
    local_rows_out = local_matched_games_rows[:2000]  # cap ok
    settled_bets_summary = {
        "count": int(len(settled_bets_rows)),
        "wins": int(sum(1 for row in settled_bets_rows if int(row.get("win", 0)) == 1)),
        "profit_eur": float(sum(float(row.get("pnl", 0.0)) for row in settled_bets_rows)),
        "roi_pct": float(
            _safe_div(
                sum(float(row.get("pnl", 0.0)) for row in settled_bets_rows),
                sum(float(row.get("stake", 0.0)) for row in settled_bets_rows),
            )
            * 100.0
            if settled_bets_rows
            else 0.0
        ),
        "avg_odds": float(
            _safe_div(
                sum(float(row.get("odds", 0.0)) for row in settled_bets_rows),
                len(settled_bets_rows),
            )
            if settled_bets_rows
            else 0.0
        ),
    }
    local_matched_games_note = "Rows reflect strategy-subset simulated local matched games in the active window."
    local_matched_games_mismatch = bool(
        int(local_summary.get("rows_count", 0)) > 0
        and int(local_summary.get("rows_count", 0)) != int(local_matched_games_count)
    )
    tables_payload = {
        "historical_stats": historical_stats,
        "accuracy_threshold_stats": accuracy_thresholds,
        "calibration_metrics": calibration,
        "calibration_quality": {
            "ece": float(calibration.get("ece", 0.0)),
            "calibrationSlope": float(calibration.get("calibrationSlope", 0.0)),
            "calibrationIntercept": float(calibration.get("calibrationIntercept", 0.0)),
        },
        "home_win_rate_threshold": float(params_for_display.get("home_win_rate_threshold", 0.0)),
        "home_win_rate_shown_count": int(len(home_win_rates_window)),
        "strategy_filter_stats": {
            "window_size": int(window_size_label),
            "matched_games_count": int(local_matched_games_count),
            "window_start": window_start_label,
            "window_end": window_end_label,
            "filters": [
                {"label": "Home win rate min", "value": float(params_for_display.get("home_win_rate_threshold", 0.0))},
                {"label": "Odds min", "value": float(params_for_display.get("odds_min", 1.0))},
                {"label": "Odds max", "value": float(params_for_display.get("odds_max", DEFAULT_MAX_ODDS_FALLBACK))},
                {"label": "Prob min", "value": float(params_for_display.get("prob_threshold", 0.5))},
                {"label": "Min EV", "value": float(params_for_display.get("min_ev", 0.0))},
            ],
        },
        "strategy_summary": strategy_summary,
        "bankroll_history": bankroll_history,
        "local_matched_games_rows": local_rows_out,
        "local_matched_games_count": int(local_matched_games_count),
        "local_matched_games_mismatch": local_matched_games_mismatch,
        "local_matched_games_note": local_matched_games_note,
        "bets_2026_settled_rows": settled_bets_rows,
        "bets_2026_settled_count": int(len(settled_bets_rows)),
        "bets_2026_settled_summary": settled_bets_summary,

        # Legacy keys retained for backward compatibility
        "settled_bets_rows": settled_bets_rows,
        "settled_bets_summary": settled_bets_summary,
        "home_win_rates_window": home_win_rates_window,
        "home_win_rates_last20": home_win_rates,
    }

    # Validator requires these NOT be None when sample size >= 5
    sharpe_out = local_sharpe
    max_dd_out = local_max_dd_eur
    if len(local_rows_out) >= 5:
        if sharpe_out is None:
            sharpe_out = 0.0
        if max_dd_out is None:
            max_dd_out = 0.0

    # Summary expected by validate_dashboard_payload.py + SPEC.md
    summary_stats = {
        "total_games": int(len(window_played_rows)),
        "overall_accuracy": float(
            _safe_div(
                sum(1 for row in window_played_rows if ((row.get("prob_iso") or row.get("prob_raw") or 0.0) >= 0.5) == (int(row.get("home_team_won", 0)) == 1)),
                len(window_played_rows),
            )
            if window_played_rows
            else 0.0
        ),
        "as_of_date": as_of_date,
    }
    bets_2026_settled_overview = {
        "count": int(len(settled_bets_rows)),
        "wins": int(settled_bets_summary["wins"]),
        "profit_eur": float(settled_bets_summary["profit_eur"]),
        "roi_pct": float(settled_bets_summary["roi_pct"]),
    }
    summary_payload = {
        "last_run": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "as_of_date": as_of_date,
        "summary_stats": summary_stats,
        "active_filters": active_filters_text,
        "active_filters_human": active_filters_label,
        "params_used_type": selection_metadata.get("params_source_type") or strategy_params_source_type,
        "ytd_source": _label_path(bet_log_flat_path),
        "ytd_note": "Based on settled rows from bet_log_flat_live.csv trimmed to snapshot date.",
        "bets_2026_settled_overview": bets_2026_settled_overview,
        "strategy_subset_in_window": {
            "count": int(local_matched_games_count),
            "window_start": window_start_label,
            "window_end": window_end_label,
            "note": local_matched_games_note,
        },
        "bankroll": {
            "strategy_subset": bankroll_last_200,
            "ytd_2026": bankroll_ytd_2026,
            "placed_bets_history_points": int(len(bankroll_history)),
        },
        "strategy_counts": {
            "settled_bets_count": int(len(tables_payload["local_matched_games_rows"])),
        },
        "strategy_summary": {
            "sharpeStyle": sharpe_out,
        },
        "kpis": {
            "max_drawdown_eur": max_dd_out,
            "roi_pct": float(strategy_roi_pct),
        },
        "risk_metrics": {
            "strategy_sharpe_style": sharpe_out,
            "strategy_max_drawdown_eur": max_dd_out,
            "strategy_max_drawdown_pct": local_max_dd_pct,
        },
        "source": {
            "combined_file": _label_path(combined_path),
            "local_matched_file": _label_path(local_matched_games_path),
            "bet_log_file": _label_path(bet_log_flat_path),
            "strategy_params_file": _label_path(strategy_params_source),
            "metrics_snapshot_file": _label_path(sources.metrics_snapshot),
        },

        # --- keep your app stuff (optional, extra fields are fine) ---
        "asOfDate": as_of_date,
        "windowSize": int(window_size_label),
        "model": {
            "historical": historical_stats,
            "thresholds": accuracy_thresholds,
            "calibration": calibration,
            "homeWinRatesLast20": home_win_rates,
        },
        "strategy": strategy_summary,
        "betLog": {
            "summary": bet_log_summary,
            "bankrollHistory": bankroll_history,
        },
    }

    last_run_payload = {
        "last_run": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "as_of_date": as_of_date,
        "records": {
            "played_games": int(len(played_rows)),
            "strategy_subset_rows": int(local_matched_games_count),
            "settled_bets_2026": int(len(settled_bets_rows)),
        },
        # legacy fields retained
        "lastUpdateUtc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "asOfDate": as_of_date,
        "snapshotAsOfDate": snapshot_as_of_date,
        "windowStart": window_start_label,
        "windowEnd": window_end_label,
        "dataConsistencyStatus": data_consistency_status,
        "dataConsistencyIssues": consistency_issues,
        "dataWarnings": data_warnings,
        "sources": {
            "combined": _label_path(combined_path),
            "local_matched": _label_path(local_matched_games_path),
            "bet_log_flat": _label_path(bet_log_flat_path),
            "strategy_params": _label_path(strategy_params_source),
            "metrics_snapshot": _label_path(sources.metrics_snapshot),
        },
        "selection": {
            "snapshot_as_of_date": selection_metadata.get("snapshot_as_of_date") or as_of_date,
            "run_date": selection_metadata.get("run_date") or as_of_date,
            "combined_source_file": selection_metadata.get("combined_source_file") or combined_path.name,
            "local_matched_source_file": selection_metadata.get("local_matched_source_file") or _label_path(local_matched_games_path),
            "bet_log_source_file": _metadata_path(bet_log_path),
            "bet_log_source_type": selection_metadata.get("bet_log_source_type"),
            "bet_log_latest_date_in_file": selection_metadata.get("bet_log_latest_date_in_file"),
            "bet_log_trimmed_to_snapshot": bool(selection_metadata.get("bet_log_trimmed_to_snapshot")),
            "metrics_source_file": selection_metadata.get("metrics_source_file") or _label_path(sources.metrics_snapshot),
            "strategy_params_source_file": _metadata_path(strategy_params_source),
            "strategy_params_source_type": strategy_params_source_type,
            "strategy_params_parsed_ok": strategy_params_parsed_ok,
            "params_source_file": selection_metadata.get("params_source_file"),
            "params_source_type": selection_metadata.get("params_source_type") or strategy_params_source_type,
            "params_used": selection_metadata.get("params_used") or strategy_params_name,
            "fallback_used": bool(selection_metadata.get("fallback_used") or defaults_used),
            "fallback_reason": selection_metadata.get("fallback_reason") or defaults_reason or "",
            "snapshot_fallback_used": bool(selection_metadata.get("fallback_used")),
            "snapshot_fallback_reason": selection_metadata.get("fallback_reason"),
        },
    }

    copied_sources = copy_sources(
        output_dir,
        {
            "combined": combined_path,
            "local_matched": local_matched_games_path,
            "strategy_params": strategy_params_source,
            "bet_log": sources.bet_log if sources.bet_log and sources.bet_log.exists() else None,
            "bet_log_flat": bet_log_flat_path,
        },
    )

    dashboard_payload = {
        "as_of_date": as_of_date,
        "window": {
            "size": int(window_size_label),
            "start": window_start_label or "—",
            "end": window_end_label or "—",
            "games_count": int(len(window_played_rows) if window_played_rows else 0),
        },
        "active_filters_effective": active_filters_label,
        "summary": summary_payload,
        "tables": tables_payload,
        "state": dashboard_state,
        "sources": {
            "combined_file": _label_path(combined_path),
            "local_matched_games": _label_path(local_matched_games_path),
            "bet_log_flat": _label_path(bet_log_flat_path),
            "copied": copied_sources,
        },
    }

    # JSON subset for UI to avoid client CSV parsing
    local_matched_subset = [
        {
            "date": row.get("date"),
            "home_team": row.get("home_team"),
            "away_team": row.get("away_team"),
            "home_win_rate": row.get("home_win_rate"),
            "prob_iso": row.get("prob_iso"),
            "prob_used": row.get("prob_used"),
            "odds_1": row.get("odds_1"),
            "ev_eur_per_100": row.get("ev_eur_per_100"),
            "win": row.get("win"),
            "pnl": row.get("pnl"),
        }
        for row in local_rows_out
    ]

    # ----------------------------
    # Write outputs
    # ----------------------------
    write_json(output_dir / "dashboard_state.json", dashboard_state)
    write_json(output_dir / "summary.json", summary_payload)
    write_json(output_dir / "tables.json", tables_payload)
    write_json(output_dir / "last_run.json", last_run_payload)
    write_json(output_dir / "dashboard_payload.json", dashboard_payload)
    write_json(output_dir / "local_matched_games_latest.json", {"rows": local_matched_subset})
    write_json(output_dir / "today_games.json", today_games_payload)
    _write_agent_learning_cases(output_dir, local_profitability_rule["cases"])
    remove_legacy_typo_files(output_dir)
    verify_required_dashboard_json(output_dir)

    print(
        "Wrote summary.json, tables.json, last_run.json, dashboard_payload.json, dashboard_state.json "
        f"to {output_dir}"
    )


if __name__ == "__main__":
    main()
