#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute strategy filter parameters from recent played games.

This script mirrors the windowed dataset used by the dashboard and
emits strategy_params.json for the UI to consume.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


DATE_FMT = "%Y-%m-%d"
STAKE_EUR = 100.0
HOME_WIN_RATE_WINDOW = 20


@dataclass
class SourcePaths:
    combined_iso: Optional[Path]
    combined_acc: Optional[Path]


def _normalize_key(key: str) -> str:
    key = key.strip().lower()
    key = re.sub(r"[\s\-]+", "_", key)
    key = re.sub(r"[^a-z0-9_]", "", key)
    key = re.sub(r"_+", "_", key)
    return key


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
        try:
            dt = datetime.strptime(match.group(1), DATE_FMT)
        except ValueError:
            continue
        candidates.append((dt, item))
    if not candidates:
        return None
    return sorted(candidates, key=lambda x: x[0])[-1][1]


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


def _resolve_sources(root: Optional[Path]) -> SourcePaths:
    if root is None:
        return SourcePaths(combined_iso=None, combined_acc=None)
    lightgbm_dir = root / "output" / "LightGBM"
    kelly_dir = lightgbm_dir / "Kelly"
    combined_iso = _find_latest_file(kelly_dir, "combined_nba_predictions_iso")
    combined_acc = _find_latest_file(lightgbm_dir, "combined_nba_predictions_acc")
    return SourcePaths(combined_iso=combined_iso, combined_acc=combined_acc)


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


def _compute_ev_per_100(prob: float, odds: float) -> float:
    return (prob * (odds - 1.0) - (1.0 - prob)) * 100.0


def _compute_pnl(win: int, odds: float, stake: float) -> float:
    return (odds - 1.0) * stake if win == 1 else -stake


def _compute_max_drawdown(pnls: Iterable[float]) -> float:
    peak = 0.0
    max_dd = 0.0
    cumulative = 0.0
    for pnl in pnls:
        cumulative += pnl
        if cumulative > peak:
            peak = cumulative
        dd = peak - cumulative
        if dd > max_dd:
            max_dd = dd
    return max_dd


def _load_base_dataframe(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [_normalize_key(col) for col in df.columns]

    date_col = "game_date" if "game_date" in df.columns else "date"
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])

    result_col = "result" if "result" in df.columns else "result_raw"
    if result_col not in df.columns:
        raise ValueError("Combined predictions file missing result column.")

    df["result"] = df[result_col].astype(str).str.strip()
    df = df[df["result"].notna() & (df["result"] != "") & (df["result"] != "0")]

    if "home_team" not in df.columns or "away_team" not in df.columns:
        raise ValueError("Combined predictions file missing home/away team columns.")

    df["home_team"] = df["home_team"].astype(str).str.strip()
    df["away_team"] = df["away_team"].astype(str).str.strip()

    df["home_team_won"] = (df["result"] == df["home_team"]).astype(int)

    prob_raw = None
    if "pred_home_win_proba" in df.columns:
        prob_raw = "pred_home_win_proba"
    elif "home_team_prob" in df.columns:
        prob_raw = "home_team_prob"

    prob_iso = "iso_proba_home_win" if "iso_proba_home_win" in df.columns else None
    odds_col = "closing_home_odds" if "closing_home_odds" in df.columns else "odds_1"

    df["prob_raw"] = pd.to_numeric(df[prob_raw], errors="coerce") if prob_raw else None
    df["prob_iso"] = pd.to_numeric(df[prob_iso], errors="coerce") if prob_iso else None
    df["odds"] = pd.to_numeric(df.get(odds_col), errors="coerce")

    df["prob_used"] = df["prob_iso"].where(df["prob_iso"].notna(), df["prob_raw"])
    df = df.dropna(subset=["prob_used", "odds"])

    if "home_win_rate" in df.columns:
        df["home_win_rate"] = pd.to_numeric(df["home_win_rate"], errors="coerce")
    else:
        df["home_win_rate"] = None

    df = df.sort_values(date_col)
    df["date"] = df[date_col].dt.strftime(DATE_FMT)

    df["ev_per_100"] = df.apply(
        lambda row: _compute_ev_per_100(row["prob_used"], row["odds"]), axis=1
    )
    df["pnl"] = df.apply(
        lambda row: _compute_pnl(int(row["home_team_won"]), row["odds"], STAKE_EUR), axis=1
    )

    if df["home_win_rate"].isna().all():
        df["home_win_rate"] = (
            df.groupby("home_team")["home_team_won"]
            .apply(
                lambda s: s.shift(1)
                .rolling(window=HOME_WIN_RATE_WINDOW, min_periods=1)
                .mean()
            )
            .reset_index(level=0, drop=True)
        )

    return df[
        [
            "date",
            "home_team",
            "away_team",
            "home_team_won",
            "prob_used",
            "odds",
            "home_win_rate",
            "ev_per_100",
            "pnl",
        ]
    ]


def _evaluate_candidate(df: pd.DataFrame) -> Dict[str, float]:
    n_trades = int(len(df))
    win_rate = float(df["home_team_won"].mean()) if n_trades else 0.0
    total_pnl = float(df["pnl"].sum()) if n_trades else 0.0
    roi_pct = (total_pnl / (n_trades * STAKE_EUR) * 100.0) if n_trades else 0.0
    avg_ev = float(df["ev_per_100"].mean()) if n_trades else 0.0
    max_drawdown = float(_compute_max_drawdown(df["pnl"].tolist())) if n_trades else 0.0
    return {
        "n_trades": n_trades,
        "win_rate": win_rate,
        "roi_pct": roi_pct,
        "avg_ev_per_100": avg_ev,
        "max_drawdown": max_drawdown,
    }


def _grid_search(
    df: pd.DataFrame, window_size: int, min_ev: float
) -> Tuple[Dict[str, object], Dict[str, object]]:
    window_df = df.tail(window_size).copy()
    if window_df.empty:
        raise RuntimeError("No settled games found in the requested window.")

    as_of_date = window_df["date"].iloc[-1]

    home_win_rate_thresholds = [0.50, 0.55, 0.60, 0.65, 0.70]
    odds_mins = [1.6, 1.8, 2.0]
    odds_maxs = [2.1, 2.5, 3.2]
    prob_thresholds = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]

    candidates: List[Dict[str, object]] = []
    for home_thr in home_win_rate_thresholds:
        for odds_min in odds_mins:
            for odds_max in odds_maxs:
                if odds_max <= odds_min:
                    continue
                for prob_thr in prob_thresholds:
                    filtered = window_df[
                        (window_df["home_win_rate"] >= home_thr)
                        & (window_df["odds"] >= odds_min)
                        & (window_df["odds"] <= odds_max)
                        & (window_df["prob_used"] >= prob_thr)
                        & (window_df["ev_per_100"] > min_ev)
                    ]
                    metrics = _evaluate_candidate(filtered)
                    candidates.append(
                        {
                            "home_win_rate_threshold": home_thr,
                            "odds_min": odds_min,
                            "odds_max": odds_max,
                            "prob_threshold": prob_thr,
                            **metrics,
                        }
                    )

    if not candidates:
        raise RuntimeError("No candidates generated for strategy parameter search.")

    min_trades = 12
    eligible = [c for c in candidates if c["n_trades"] >= min_trades]
    if eligible:
        best = max(eligible, key=lambda c: (c["roi_pct"], c["n_trades"]))
    else:
        best = max(candidates, key=lambda c: (c["n_trades"], c["roi_pct"]))

    summary = {
        "as_of_date": as_of_date,
        "window_size": window_size,
        "min_ev": min_ev,
        "stake": int(STAKE_EUR),
    }
    return summary, best


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        type=str,
        default=None,
        help="Path to Basketball_prediction/2026 root (fallback when SOURCE_ROOT is unset).",
    )
    parser.add_argument("--window", type=int, default=200, help="Window size in games.")
    parser.add_argument("--min-ev", type=float, default=-5.0, help="Min EV €/100.")
    parser.add_argument(
        "--out",
        type=str,
        default="public/data/strategy_params.json",
        help="Output JSON path.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    source_root = resolve_source_root(args.source_root, repo_root)
    sources = _resolve_sources(source_root)
    combined_path = sources.combined_iso or sources.combined_acc
    if combined_path is None:
        raise FileNotFoundError("No combined predictions file found.")

    df = _load_base_dataframe(combined_path)
    summary, best = _grid_search(df, window_size=args.window, min_ev=args.min_ev)

    payload = {
        **summary,
        **{
            "home_win_rate_threshold": best["home_win_rate_threshold"],
            "odds_min": best["odds_min"],
            "odds_max": best["odds_max"],
            "prob_threshold": best["prob_threshold"],
            "n_trades": best["n_trades"],
            "roi_pct": best["roi_pct"],
            "win_rate": best["win_rate"],
            "avg_ev_per_100": best["avg_ev_per_100"],
            "max_drawdown": best["max_drawdown"],
        },
        "source_file": str(combined_path),
    }

    out_path = Path(args.out).resolve()
    write_json(out_path, payload)
    print(f"Wrote strategy params to {out_path}")


if __name__ == "__main__":
    main()
