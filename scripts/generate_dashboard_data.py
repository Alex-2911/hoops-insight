#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate stats-only dashboard artifacts for hoops-insight.

This script reads historical outputs from Basketball_prediction (played games only)
and writes stable JSON files into public/data for the Vite app to consume.
No future predictions are loaded or emitted.
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

import pandas as pd


DATE_FMT = "%Y-%m-%d"
THRESHOLDS = [
    {"label": "> 0.60", "thresholdType": "gt", "threshold": 0.60},
    {"label": "<= 0.40", "thresholdType": "lt", "threshold": 0.40},
]


@dataclass
class SourcePaths:
    combined_iso: Optional[Path]
    combined_acc: Optional[Path]
    bet_log: Optional[Path]
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


def load_metrics_snapshot(path: Path) -> Dict[str, Dict[str, object]]:
    if not path.exists():
        return {}
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        parsed: Dict[str, Dict[str, object]] = {}
        if isinstance(data, list):
            for row in data:
                if not isinstance(row, dict):
                    continue
                section = row.get("section")
                metric = row.get("metric")
                value = row.get("value")
                if section and metric:
                    parsed.setdefault(str(section), {})[str(metric)] = _coerce_value(value)
        elif isinstance(data, dict):
            for section, metrics in data.items():
                if isinstance(metrics, dict):
                    for metric, value in metrics.items():
                        parsed.setdefault(str(section), {})[str(metric)] = _coerce_value(value)
        return parsed

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


def _find_latest_by_mtime(path: Path, pattern: str) -> Optional[Path]:
    if not path.exists():
        return None
    candidates = [item for item in path.glob(pattern) if item.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda item: item.stat().st_mtime)


def _find_metrics_snapshot(lightgbm_dir: Path, as_of_date: Optional[str]) -> Optional[Path]:
    if as_of_date:
        dated = lightgbm_dir / f"metrics_snapshot_{as_of_date}.json"
        if dated.exists():
            return dated
    preferred = lightgbm_dir / "metrics_snapshot.json"
    if preferred.exists():
        return preferred
    latest_snapshot = _find_latest_by_mtime(lightgbm_dir, "metrics_snapshot*.json")
    if latest_snapshot:
        return latest_snapshot
    latest_csv = _find_latest_by_mtime(lightgbm_dir, "betting_metrics_snapshot_*.csv")
    if latest_csv:
        return latest_csv
    for name in ("metrics_snapshot.csv", "metrics_snapshot.json"):
        candidate = lightgbm_dir / name
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
        dated = lightgbm_dir / f"strategy_params_{as_of_date}.txt"
        if dated.exists():
            return dated
    preferred = lightgbm_dir / "strategy_params.txt"
    if preferred.exists():
        return preferred
    return _find_latest_by_mtime(lightgbm_dir, "strategy_params*.txt")


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
            metrics_snapshot=None,
            local_matched_games=None,
            strategy_params=None,
        )
    lightgbm_dir = root / "output" / "LightGBM"
    kelly_dir = lightgbm_dir / "Kelly"
    combined_iso = _find_latest_file(kelly_dir, "combined_nba_predictions_iso")
    combined_acc = _find_latest_file(lightgbm_dir, "combined_nba_predictions_acc")
    bet_log = lightgbm_dir / "bet_log_live.csv"
    if not bet_log.exists():
        bet_log = _find_latest_file(lightgbm_dir, "bet_log_live")
    metrics_snapshot = _find_metrics_snapshot(lightgbm_dir, as_of_date)
    strategy_params = _find_strategy_params(lightgbm_dir, as_of_date)
    local_matched_games = _find_local_matched_games(lightgbm_dir, as_of_date)
    return SourcePaths(
        combined_iso=combined_iso,
        combined_acc=combined_acc,
        bet_log=bet_log,
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


def build_calibration_metrics(rows: List[Dict[str, object]]) -> Dict[str, object]:
    p_raw = [r["prob_raw"] for r in rows if r["prob_raw"] is not None]
    p_iso = [r["prob_iso"] for r in rows if r["prob_iso"] is not None]

    if not p_raw:
        return {
            "asOfDate": "—",
            "brierBefore": 0.0,
            "brierAfter": 0.0,
            "logLossBefore": 0.0,
            "logLossAfter": 0.0,
            "fittedGames": 0,
        }

    y_for_raw = [int(r["home_team_won"]) for r in rows if r["prob_raw"] is not None]
    y_for_iso = [int(r["home_team_won"]) for r in rows if r["prob_iso"] is not None]

    brier_before = _compute_brier(y_for_raw, p_raw)
    logloss_before = _compute_log_loss(y_for_raw, p_raw)

    if p_iso and y_for_iso:
        brier_after = _compute_brier(y_for_iso, p_iso)
        logloss_after = _compute_log_loss(y_for_iso, p_iso)
        fitted = len(y_for_iso)
    else:
        brier_after = brier_before
        logloss_after = logloss_before
        fitted = len(y_for_raw)

    as_of = max(r["date"] for r in rows).strftime(DATE_FMT)
    return {
        "asOfDate": as_of,
        "brierBefore": brier_before,
        "brierAfter": brier_after,
        "logLossBefore": logloss_before,
        "logLossAfter": logloss_after,
        "fittedGames": fitted,
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
        return {"window_size": window_size, "filters": [], "matched_games_count": 0}
    sorted_rows = sorted(rows, key=lambda r: r["date"])
    window_rows = sorted_rows[-window_size:]
    filters = [{"label": "Window games", "count": len(window_rows)}]

    min_prob_used = _get_param(params, "min_prob_used", "min_prob", "min_prob_iso")
    min_odds = _get_param(params, "min_odds_1", "min_odds")
    max_odds = _get_param(params, "max_odds_1", "max_odds")
    min_ev = _get_param(params, "min_ev_eur_per_100", "min_ev_per_100", "min_ev")
    min_home_win_rate = _get_param(params, "min_home_win_rate")

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
            and _compute_ev_per_100(_prob_used(r), r.get("odds_home")) >= min_ev
        ]
        filters.append({"label": f"EV €/100 ≥ {min_ev:.2f}", "count": len(current)})
    if min_home_win_rate is not None:
        current = [
            r
            for r in current
            if r.get("home_win_rate") is not None and r["home_win_rate"] >= min_home_win_rate
        ]
        filters.append(
            {"label": f"Home win rate ≥ {min_home_win_rate:.2f}", "count": len(current)}
        )

    return {"window_size": window_size, "filters": filters, "matched_games_count": len(current)}


def _compute_local_bankroll(rows: List[Dict[str, object]], start: float, stake: float) -> Dict[str, float]:
    net_pl = sum(row.get("pnl", 0.0) for row in rows)
    return {"start": start, "stake": stake, "net_pl": net_pl, "bankroll": start + net_pl}


def _snapshot_value(snapshot: Dict[str, Dict[str, object]], section: str, metric: str) -> Optional[object]:
    return snapshot.get(section, {}).get(metric)


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


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, default=_serialize, ensure_ascii=False, indent=2)


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
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    source_root = resolve_source_root(args.source_root, repo_root)

    output_dir = Path(args.output_dir) if args.output_dir else repo_root / "public" / "data"

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
    calibration = build_calibration_metrics(played_rows)
    home_win_rates = build_home_win_rates_last20(played_rows)

    bet_log_rows = []
    if sources.bet_log and sources.bet_log.exists():
        bet_log_rows = load_bet_log(sources.bet_log)

    bet_log_summary = build_bet_log_summary(bet_log_rows)
    bankroll_history = build_bankroll_history(bet_log_rows)
    max_dd_eur, max_dd_pct = compute_max_drawdown(bankroll_history)

    snapshot_as_of_date = None
    as_of_date = max(r["date"] for r in played_rows).strftime(DATE_FMT)

    expected_lightgbm_dir = source_root / "output" / "LightGBM" if source_root else None
    if expected_lightgbm_dir:
        sources = _resolve_sources(source_root, as_of_date)

    metrics_snapshot = (
        load_metrics_snapshot(sources.metrics_snapshot) if sources.metrics_snapshot else {}
    )
    realized_count_raw = _snapshot_value(metrics_snapshot, "realized", "count")
    realized_profit_raw = _snapshot_value(metrics_snapshot, "realized", "profit_sum")
    realized_roi_raw = _snapshot_value(metrics_snapshot, "realized", "roi")
    realized_win_rate_raw = _snapshot_value(metrics_snapshot, "realized", "win_rate")
    realized_sharpe_raw = _snapshot_value(metrics_snapshot, "realized", "sharpe_style")
    ev_mean_raw = _snapshot_value(metrics_snapshot, "ev_stats", "mean")
    snapshot_as_of_raw = _snapshot_value(metrics_snapshot, "meta", "eval_base_date_max")

    snapshot_as_of_date = _safe_date(str(snapshot_as_of_raw)) if snapshot_as_of_raw else None
    if snapshot_as_of_date:
        as_of_date = snapshot_as_of_date.strftime(DATE_FMT)
        if expected_lightgbm_dir:
            sources = _resolve_sources(source_root, as_of_date)
            metrics_snapshot = (
                load_metrics_snapshot(sources.metrics_snapshot)
                if sources.metrics_snapshot
                else metrics_snapshot
            )
            realized_count_raw = _snapshot_value(metrics_snapshot, "realized", "count")
            realized_profit_raw = _snapshot_value(metrics_snapshot, "realized", "profit_sum")
            realized_roi_raw = _snapshot_value(metrics_snapshot, "realized", "roi")
            realized_win_rate_raw = _snapshot_value(metrics_snapshot, "realized", "win_rate")
            realized_sharpe_raw = _snapshot_value(metrics_snapshot, "realized", "sharpe_style")
            ev_mean_raw = _snapshot_value(metrics_snapshot, "ev_stats", "mean")

    local_matched_games_rows: List[Dict[str, object]] = []
    local_matched_games_summary = {"rows_count": 0, "profit_sum_table": 0.0}
    if sources.local_matched_games and sources.local_matched_games.exists():
        local_matched_games_rows, local_matched_games_summary = load_local_matched_games_csv(
            sources.local_matched_games
        )

    local_matched_games_count = int(local_matched_games_summary["rows_count"])
    local_matched_games_profit_sum = float(local_matched_games_summary["profit_sum_table"])
    matched_count_table = local_matched_games_count if sources.local_matched_games else None

    matched_as_of_date = _max_date(local_matched_games_rows)
    if matched_as_of_date:
        strategy_as_of_date = matched_as_of_date.strftime(DATE_FMT)
    elif snapshot_as_of_date:
        strategy_as_of_date = snapshot_as_of_date.strftime(DATE_FMT)
    else:
        strategy_as_of_date = as_of_date

    params = load_strategy_params(sources.strategy_params) if sources.strategy_params else {}
    strategy_filter_stats = build_strategy_filter_stats(played_rows, params, window_size=200)

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

    profit_metrics_available = realized_profit is not None and realized_roi is not None
    matched_count_snapshot = realized_count
    matched_count_used: Optional[int] = None
    strategy_summary = {
        "totalBets": 0,
        "totalProfitEur": 0.0,
        "roiPct": 0.0,
        "avgEvPer100": 0.0,
        "winRate": 0.0,
        "sharpeStyle": realized_sharpe,
        "profitMetricsAvailable": False,
        "asOfDate": strategy_as_of_date,
    }

    if realized_count is not None:
        roi_pct = 0.0
        if realized_roi is not None:
            roi_pct = realized_roi * 100.0 if realized_roi <= 1.5 else realized_roi
        strategy_summary = {
            "totalBets": realized_count,
            "totalProfitEur": realized_profit or 0.0,
            "roiPct": roi_pct,
            "avgEvPer100": ev_mean or 0.0,
            "winRate": realized_win_rate or 0.0,
            "sharpeStyle": realized_sharpe,
            "profitMetricsAvailable": profit_metrics_available,
            "asOfDate": strategy_as_of_date,
        }
    elif local_matched_games_rows:
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
            "sharpeStyle": None,
            "profitMetricsAvailable": True,
            "asOfDate": strategy_as_of_date,
        }

    if matched_count_snapshot is not None:
        matched_count_used = matched_count_snapshot
    elif matched_count_table is not None:
        matched_count_used = matched_count_table

    if matched_count_snapshot is not None:
        strategy_filter_stats["matched_games_count"] = matched_count_snapshot
    elif matched_count_table is not None:
        strategy_filter_stats["matched_games_count"] = matched_count_table
    elif params:
        strategy_filter_stats["matched_games_count"] = (
            strategy_filter_stats.get("matched_games_count") or 0
        )
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

    total_games = len(played_rows)
    total_correct = sum(int(r["home_team_won"]) for r in played_rows)
    overall_accuracy = _safe_div(total_correct, total_games)

    last_run = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    summary_payload = {
        "last_run": last_run,
        "as_of_date": as_of_date,
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
            "max_drawdown_eur": max_dd_eur,
            "max_drawdown_pct": max_dd_pct,
        },
        "strategy_summary": strategy_summary,
        "strategy_params": {
            "source": str(sources.strategy_params) if sources.strategy_params else "missing",
            "params": params,
        },
        "strategy_filter_stats": strategy_filter_stats,
        "source": {
            "combined_file": str(combined_path),
            "bet_log_file": str(sources.bet_log) if sources.bet_log else "missing",
            "metrics_snapshot_source": str(sources.metrics_snapshot)
            if sources.metrics_snapshot
            else "missing",
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
        "bankroll_last_200": bankroll_last_200,
        "bankroll_ytd_2026": bankroll_ytd_2026,
    }

    last_run_payload = {
        "last_run": last_run,
        "as_of_date": as_of_date,
        "source_root_used": str(source_root) if source_root else "missing",
        "expected_lightgbm_dir": str(expected_lightgbm_dir) if expected_lightgbm_dir else "missing",
        "metrics_snapshot_source": str(sources.metrics_snapshot)
        if sources.metrics_snapshot
        else "missing",
        "strategy_params_source": str(sources.strategy_params)
        if sources.strategy_params
        else "missing",
        "local_matched_games_source": str(sources.local_matched_games)
        if sources.local_matched_games
        else "missing",
        "local_matched_games_rows": local_matched_games_count,
        "local_matched_games_profit_sum_table": local_matched_games_profit_sum,
        "matched_count_snapshot": matched_count_snapshot,
        "matched_count_table": matched_count_table,
        "matched_count_used": matched_count_used,
        "records": {
            "played_games": total_games,
            "bet_log_rows": len(bet_log_rows),
            "strategy_matched_rows": strategy_summary["totalBets"],
            "local_matched_games_rows": local_matched_games_count,
            "local_matched_games_rows_expected": realized_count or 0,
            "local_matched_games_profit_sum_table": local_matched_games_profit_sum,
        },
    }

    write_json(output_dir / "summary.json", summary_payload)
    write_json(output_dir / "tables.json", tables_payload)
    write_json(output_dir / "last_run.json", last_run_payload)

    print(f"Wrote summary.json, tables.json, last_run.json to {output_dir}")


if __name__ == "__main__":
    main()
