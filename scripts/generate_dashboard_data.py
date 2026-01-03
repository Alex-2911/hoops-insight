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
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


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


def _resolve_sources(root: Path) -> SourcePaths:
    lightgbm_dir = root / "output" / "LightGBM"
    kelly_dir = lightgbm_dir / "Kelly"
    combined_iso = _find_latest_file(kelly_dir, "combined_nba_predictions_iso")
    combined_acc = _find_latest_file(lightgbm_dir, "combined_nba_predictions_acc")
    bet_log = lightgbm_dir / "bet_log_live.csv"
    if not bet_log.exists():
        bet_log = _find_latest_file(lightgbm_dir, "bet_log_live")
    return SourcePaths(combined_iso=combined_iso, combined_acc=combined_acc, bet_log=bet_log)


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
    return output


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


def compute_avg_ev_per_100(rows: List[Dict[str, object]]) -> float:
    evs = []
    for r in rows:
        p = r["prob_iso"] if r["prob_iso"] is not None else r["prob_raw"]
        odds = r["odds_home"]
        if p is None or odds is None:
            continue
        ev = (p * (odds - 1.0) - (1.0 - p)) * 100.0
        evs.append(ev)
    return sum(evs) / len(evs) if evs else 0.0


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
        help="Path to Basketball_prediction/2026 root (defaults to sibling folder).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output dir for artifacts (default: hoops-insight/public/data).",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    default_root = repo_root.parent / "Basketball_prediction" / "2026"
    source_root = Path(args.source_root) if args.source_root else default_root

    output_dir = Path(args.output_dir) if args.output_dir else repo_root / "public" / "data"

    sources = _resolve_sources(source_root)
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
    avg_ev_per_100 = compute_avg_ev_per_100(played_rows)

    bet_log_rows = []
    if sources.bet_log and sources.bet_log.exists():
        bet_log_rows = load_bet_log(sources.bet_log)

    bet_log_summary = build_bet_log_summary(bet_log_rows)
    bankroll_history = build_bankroll_history(bet_log_rows)
    max_dd_eur, max_dd_pct = compute_max_drawdown(bankroll_history)

    total_games = len(played_rows)
    total_correct = sum(int(r["home_team_won"]) for r in played_rows)
    overall_accuracy = _safe_div(total_correct, total_games)
    as_of_date = max(r["date"] for r in played_rows).strftime(DATE_FMT)

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
            "total_bets": bet_log_summary["totalBets"],
            "win_rate": bet_log_summary["winRate"],
            "roi_pct": bet_log_summary["roiPct"],
            "avg_ev_per_100": avg_ev_per_100,
            "avg_profit_per_bet_eur": bet_log_summary["avgProfitPerBetEur"],
            "max_drawdown_eur": max_dd_eur,
            "max_drawdown_pct": max_dd_pct,
        },
        "source": {
            "combined_file": str(combined_path),
            "bet_log_file": str(sources.bet_log) if sources.bet_log else "missing",
        },
    }

    tables_payload = {
        "historical_stats": historical_stats,
        "accuracy_threshold_stats": accuracy_thresholds,
        "calibration_metrics": calibration,
        "home_win_rates_last20": home_win_rates,
        "bet_log_summary": bet_log_summary,
        "bankroll_history": bankroll_history,
    }

    last_run_payload = {
        "last_run": last_run,
        "as_of_date": as_of_date,
        "records": {
            "played_games": total_games,
            "bet_log_rows": len(bet_log_rows),
        },
    }

    write_json(output_dir / "summary.json", summary_payload)
    write_json(output_dir / "tables.json", tables_payload)
    write_json(output_dir / "last_run.json", last_run_payload)

    print(f"Wrote summary.json, tables.json, last_run.json to {output_dir}")


if __name__ == "__main__":
    main()
