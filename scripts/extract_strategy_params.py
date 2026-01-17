#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extract local strategy parameters from a betting-strategy log and write a params file.

The output file is used by scripts/generate_dashboard_data.py to drive dynamic filters.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Dict


def _normalize_key(key: str) -> str:
    key = key.strip().lower()
    key = re.sub(r"[\s\-]+", "_", key)
    key = re.sub(r"[^a-z0-9_]", "", key)
    key = re.sub(r"_+", "_", key)
    return key


def _parse_strategy_params_text(raw: str) -> Dict[str, float]:
    keys = {
        "home_win_rate_threshold": "home_win_rate_threshold",
        "odds_min": "odds_min",
        "odds_max": "odds_max",
        "prob_threshold": "prob_threshold",
        "prob_threshold_used": "prob_threshold",
        "min_ev": "min_ev",
    }
    values: Dict[str, float] = {}

    for line in raw.splitlines():
        cleaned = line.strip()
        if not cleaned:
            continue
        if "Min EV applied" in cleaned:
            match = re.search(r"Min EV applied\s*=\s*([-0-9.]+)", cleaned)
            if match:
                values["min_ev"] = float(match.group(1))
            continue
        match = re.match(r"([A-Za-z0-9_ ()]+)\s*:\s*([-0-9.]+)", cleaned)
        if not match:
            continue
        key_raw = _normalize_key(match.group(1))
        val = float(match.group(2))
        if key_raw in keys:
            values[keys[key_raw]] = val

    return values


def _resolve_source_root() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    default_root = repo_root.parent / "Basketball_prediction" / "2026"
    env_root = os.environ.get("SOURCE_ROOT")
    if env_root:
        return Path(env_root)
    return default_root


def _find_latest_log(dir_path: Path) -> Path:
    candidates = []
    for item in dir_path.iterdir():
        if not item.is_file():
            continue
        if item.suffix.lower() not in {".log", ".txt"}:
            continue
        try:
            content = item.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if "LOCAL PARAMS" not in content:
            continue
        candidates.append((item.stat().st_mtime, item))
    if not candidates:
        raise RuntimeError(f"No log files with LOCAL PARAMS found in {dir_path}")
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log_path", type=str, help="Path to the betting-strategy log file or directory.")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for strategy params (default: SOURCE_ROOT/output/LightGBM/strategy_params.txt).",
    )
    args = parser.parse_args()

    log_path = Path(args.log_path)
    if not log_path.exists():
        raise FileNotFoundError(f"Log path not found: {log_path}")
    if log_path.is_dir():
        log_path = _find_latest_log(log_path)

    params = _parse_strategy_params_text(log_path.read_text(encoding="utf-8"))
    if not params:
        raise RuntimeError("No strategy params found in the provided log file.")

    source_root = _resolve_source_root()
    output_path = (
        Path(args.output)
        if args.output
        else source_root / "output" / "LightGBM" / "strategy_params.txt"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "=== LOCAL PARAMS (FOUND BY SEARCH, LAST N GAMES) ===",
        f"home_win_rate_threshold : {params.get('home_win_rate_threshold', '')}",
        f"odds_min                : {params.get('odds_min', '')}",
        f"odds_max                : {params.get('odds_max', '')}",
        f"prob_threshold (USED)   : {params.get('prob_threshold', '')}",
    ]
    if "min_ev" in params:
        lines.insert(0, f"Min EV applied = {params['min_ev']}")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote strategy params to {output_path}")


if __name__ == "__main__":
    main()
