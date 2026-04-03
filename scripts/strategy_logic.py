"""Shared strategy parameter and filtering logic for dashboard exports."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import pandas as pd

DEFAULT_PARAMS: Dict[str, float] = {
    "home_win_rate_threshold": 0.0,
    "odds_min": 1.0,
    "odds_max": 3.2,
    "prob_threshold": 0.5,
    "min_ev": 0.0,
}
SUPPORTED_VERSION = 1


@dataclass
class StrategyParams:
    """Versioned strategy params payload."""

    version: int
    params: Dict[str, float]
    source: str


def _normalize_key(key: str) -> str:
    key = key.strip().lower()
    key = re.sub(r"[\s\-]+", "_", key)
    key = re.sub(r"[^a-z0-9_]", "", key)
    return re.sub(r"_+", "_", key)


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_strategy_params(path: Path | None) -> StrategyParams:
    """Load versioned strategy parameters with default fallback."""
    if path is None or not path.exists():
        return StrategyParams(version=SUPPORTED_VERSION, params=DEFAULT_PARAMS.copy(), source="defaults")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid strategy params format in {path}")

    version = int(payload.get("version", SUPPORTED_VERSION))
    if version != SUPPORTED_VERSION:
        raise ValueError(f"Unsupported strategy params version {version} in {path}; expected {SUPPORTED_VERSION}")

    raw_params = payload.get("params", payload.get("params_used", {}))
    if not isinstance(raw_params, dict):
        raw_params = {}
    # Backward compatibility: some producers store params flat at the top-level.
    if not raw_params:
        raw_params = {
            k: v
            for k, v in payload.items()
            if _normalize_key(str(k)) in {"home_win_rate_threshold", "odds_min", "odds_max", "prob_threshold", "min_ev", "min_ev_per_100"}
        }

    normalized: Dict[str, float] = DEFAULT_PARAMS.copy()
    for key, value in raw_params.items():
        coerced = _coerce_float(value)
        if coerced is not None:
            normalized_key = _normalize_key(str(key))
            if normalized_key == "min_ev_per_100":
                normalized_key = "min_ev"
            normalized[normalized_key] = coerced

    return StrategyParams(version=version, params=normalized, source=str(path))


def apply_strategy_filters(rows: pd.DataFrame, params: StrategyParams) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Annotate strategy columns and return filtered rows plus params map."""
    frame = rows.copy()
    if frame.empty:
        return frame, params.params

    frame["prob_used"] = pd.to_numeric(frame.get("prob_used", frame.get("prob_iso")), errors="coerce")
    frame["odds_1"] = pd.to_numeric(frame.get("odds_1", frame.get("closing_home_odds")), errors="coerce")
    frame["ev_eur_per_100"] = pd.to_numeric(
        frame.get("ev_eur_per_100", frame.get("EV_€_per_100", frame.get("ev_per_100"))), errors="coerce"
    )
    frame["home_win_rate"] = pd.to_numeric(frame.get("home_win_rate"), errors="coerce")

    implied = 1.0 / frame["odds_1"]
    frame["model_market_gap"] = frame["prob_used"] - implied
    frame["model_market_gap_flag"] = frame["model_market_gap"] > 0

    blocked: list[str] = []
    for _, row in frame.iterrows():
        reasons: list[str] = []
        if pd.isna(row["prob_used"]) or row["prob_used"] < params.params["prob_threshold"]:
            reasons.append("prob_threshold")
        if pd.isna(row["odds_1"]) or row["odds_1"] < params.params["odds_min"] or row["odds_1"] > params.params["odds_max"]:
            reasons.append("odds_range")
        if pd.isna(row["ev_eur_per_100"]) or row["ev_eur_per_100"] <= params.params["min_ev"]:
            reasons.append("min_ev")
        if pd.isna(row["home_win_rate"]) or row["home_win_rate"] < params.params["home_win_rate_threshold"]:
            reasons.append("home_win_rate_threshold")
        blocked.append("|".join(reasons))

    frame["blocked_by"] = blocked
    filtered = frame[frame["blocked_by"] == ""].copy()
    return filtered, params.params


def required_columns() -> Iterable[str]:
    return (
        "date",
        "home_team",
        "away_team",
        "home_win_rate",
        "prob_iso",
        "prob_used",
        "win",
        "pnl",
    )
