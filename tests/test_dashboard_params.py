import json
import subprocess
import sys
from pathlib import Path


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_dashboard_state_uses_strategy_params(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    out_dir = tmp_path / "out"

    _write(
        data_dir / "combined_latest.csv",
        "date,home_team,away_team,result,pred_home_win_proba,prob_iso,closing_home_odds,home_win_rate\n"
        "2026-01-02,LAL,BOS,HOME,0.61,0.62,2.20,0.66\n",
    )
    _write(
        data_dir / "local_matched_games_latest.csv",
        "date,home_team,away_team,home_win_rate,prob_iso,prob_used,odds_1,ev_eur_per_100,win,pnl\n"
        "2026-01-02,LAL,BOS,0.66,0.62,0.62,2.2,4.0,1,120\n",
    )

    strategy_payload = {
        "version": 1,
        "params_used_type": "p_high_hw_65_odds_2-3",
        "params": {
            "home_win_rate_threshold": 0.65,
            "odds_min": 2.0,
            "odds_max": 3.0,
            "prob_threshold": 0.6,
            "min_ev": 1.5,
        },
    }
    (data_dir / "strategy_params.json").write_text(json.dumps(strategy_payload), encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            "scripts/generate_dashboard_data.py",
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(out_dir),
        ],
        check=True,
    )

    dashboard_state = json.loads((out_dir / "dashboard_state.json").read_text(encoding="utf-8"))

    assert dashboard_state["params_used"] == "p_high_hw_65_odds_2-3"
    assert dashboard_state["active_params"] == {
        "home_win_rate_min": 0.65,
        "odds_min": 2.0,
        "odds_max": 3.0,
        "prob_threshold": 0.6,
        "min_ev": 1.5,
        "window_size": 200.0,
    }


def test_dashboard_state_falls_back_on_invalid_strategy_json(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    out_dir = tmp_path / "out"

    _write(
        data_dir / "combined_latest.csv",
        "date,home_team,away_team,result,pred_home_win_proba,prob_iso,closing_home_odds,home_win_rate\n"
        "2026-01-03,NYK,MIA,AWAY,0.49,0.48,2.05,0.52\n",
    )
    (data_dir / "strategy_params.json").write_text("{invalid json", encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            "scripts/generate_dashboard_data.py",
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(out_dir),
        ],
        check=True,
    )

    dashboard_state = json.loads((out_dir / "dashboard_state.json").read_text(encoding="utf-8"))

    assert dashboard_state["params_used"] == "fallback"
    assert dashboard_state["active_params"] == {
        "home_win_rate_min": 0.0,
        "odds_min": 1.0,
        "odds_max": 3.2,
        "prob_threshold": 0.5,
        "min_ev": 0.0,
        "window_size": 200.0,
    }
