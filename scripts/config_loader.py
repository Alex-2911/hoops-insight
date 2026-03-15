#!/usr/bin/env python3
"""Central configuration loader for Hoops Insight scripts."""

from __future__ import annotations

import argparse
import os
import shlex
from pathlib import Path
from typing import Any, Dict

try:
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:
    import tomli as tomllib


CONFIG_FILE_NAME = "hoops_insight_config.toml"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_config(config_path: Path | None = None) -> Dict[str, Any]:
    """Load TOML configuration and resolve relative paths from repo root."""
    root = _repo_root()
    path = config_path or root / CONFIG_FILE_NAME
    if not path.exists():
        raise FileNotFoundError(f"Config file not found at {path}")

    with path.open("rb") as fh:
        raw = tomllib.load(fh)

    paths = raw.get("paths", {})
    dashboard = raw.get("dashboard", {})

    def _resolve(value: str, default: str) -> str:
        raw_val = str(value or default)
        candidate = Path(raw_val)
        if not candidate.is_absolute():
            candidate = (root / candidate).resolve()
        return str(candidate)

    hoops_dir = _resolve(paths.get("hoops_dir", "."), ".")
    nba_dir = _resolve(paths.get("nba_dir", "../Basketball_prediction"), "../Basketball_prediction")
    source_root = _resolve(paths.get("source_root", "../Basketball_prediction/2026"), "../Basketball_prediction/2026")
    dashboard_data_dir = _resolve(paths.get("dashboard_data_dir", "public/data"), "public/data")

    return {
        "HOOPS_DIR": hoops_dir,
        "NBA_DIR": nba_dir,
        "SOURCE_ROOT": source_root,
        "DASHBOARD_DATA_DIR": dashboard_data_dir,
        "HOST": str(dashboard.get("host", "127.0.0.1")),
        "PORT": str(dashboard.get("port", 4173)),
        "HISTORICAL_ROUTE": str(dashboard.get("historical_route", "/")),
    }


def cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("format", choices=["json", "shell"], nargs="?", default="json")
    parser.add_argument("--key", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config()
    cfg = {k: os.getenv(k, v) for k, v in cfg.items()}

    if args.key:
        if args.key not in cfg:
            raise KeyError(f"Unknown config key: {args.key}")
        print(cfg[args.key])
        return

    if args.format == "shell":
        for key, value in cfg.items():
            print(f"{key}={shlex.quote(str(value))}")
        return

    import json

    print(json.dumps(cfg, indent=2))


if __name__ == "__main__":
    cli()
