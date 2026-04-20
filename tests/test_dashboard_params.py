import json
import subprocess
import sys
from pathlib import Path


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_dashboard_state_uses_strategy_params_when_metrics_missing(tmp_path: Path) -> None:
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

    assert dashboard_state["params_used"] == "from_file"
    assert dashboard_state["params_source_type"] in {"strategy_params", "strategy_params_dated"}
    assert dashboard_state["fallback_used"] is False
    assert dashboard_state["active_params"] == {
        "home_win_rate_min": 0.65,
        "odds_min": 2.0,
        "odds_max": 3.0,
        "prob_threshold": 0.6,
        "min_ev": 1.5,
        "window_size": 200.0,
    }


def test_dashboard_state_prefers_metrics_snapshot_thresholds(tmp_path: Path) -> None:
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
    (data_dir / "strategy_params.json").write_text(
        json.dumps(
            {
                "params": {
                    "home_win_rate_threshold": 0.5,
                    "odds_min": 2.3,
                    "odds_max": 3.2,
                    "prob_threshold": 0.45,
                    "min_ev": -5.0,
                }
            }
        ),
        encoding="utf-8",
    )
    (data_dir / "metrics_snapshot.json").write_text(
        json.dumps(
            {
                "params": {
                    "home_win_rate_threshold": 0.55,
                    "odds_min": 1.7,
                    "odds_max": 2.6,
                    "prob_threshold": 0.55,
                    "min_ev": 0.0,
                }
            }
        ),
        encoding="utf-8",
    )

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
    assert dashboard_state["params_source_type"] == "metrics_snapshot"
    assert dashboard_state["params_used"] == "from_metrics_snapshot"
    assert dashboard_state["params_source_file"].endswith("metrics_snapshot.json")
    assert dashboard_state["active_params"] == {
        "home_win_rate_min": 0.55,
        "odds_min": 1.7,
        "odds_max": 2.6,
        "prob_threshold": 0.55,
        "min_ev": 0.0,
        "window_size": 200.0,
    }


def test_dashboard_state_ignores_invalid_metrics_thresholds(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    out_dir = tmp_path / "out"

    _write(
        data_dir / "combined_latest.csv",
        "date,home_team,away_team,result,pred_home_win_proba,prob_iso,closing_home_odds,home_win_rate\n"
        "2026-04-17,LAL,BOS,HOME,0.61,0.62,2.20,0.66\n",
    )
    _write(
        data_dir / "local_matched_games_latest.csv",
        "date,home_team,away_team,home_win_rate,prob_iso,prob_used,odds_1,ev_eur_per_100,win,pnl\n"
        "2026-04-17,LAL,BOS,0.66,0.62,0.62,2.2,4.0,1,120\n",
    )
    (data_dir / "strategy_params.json").write_text(
        json.dumps(
            {
                "params": {
                    "home_win_rate_threshold": 0.65,
                    "odds_min": 1.5,
                    "odds_max": 2.5,
                    "prob_threshold": 0.4,
                    "min_ev": 0.0,
                }
            }
        ),
        encoding="utf-8",
    )
    (data_dir / "metrics_snapshot.json").write_text(
        json.dumps(
            {
                "params": {
                    "home_win_rate_threshold": 1.01,
                    "odds_min": 99.0,
                    "odds_max": 100.0,
                    "prob_threshold": 1.01,
                    "min_ev": 0.0,
                }
            }
        ),
        encoding="utf-8",
    )

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
    assert dashboard_state["active_params"] == {
        "home_win_rate_min": 0.65,
        "odds_min": 1.5,
        "odds_max": 2.5,
        "prob_threshold": 0.4,
        "min_ev": 0.0,
        "window_size": 200.0,
    }
    assert any("ignored invalid metrics snapshot thresholds" in issue for issue in dashboard_state["data_warnings"])


def test_dashboard_state_uses_nested_params_used_schema_without_fallback_label(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    out_dir = tmp_path / "out"

    _write(
        data_dir / "combined_latest.csv",
        "date,home_team,away_team,result,pred_home_win_proba,prob_iso,closing_home_odds,home_win_rate\n"
        "2026-04-03,LAL,BOS,HOME,0.61,0.62,2.20,0.66\n",
    )
    _write(
        data_dir / "local_matched_games_latest.csv",
        "date,home_team,away_team,home_win_rate,prob_iso,prob_used,odds_1,ev_eur_per_100,win,pnl\n",
    )

    strategy_payload = {
        "params_used_label": "Historical",
        "params_used": {
            "home_win_rate_threshold": 0.55,
            "odds_min": 2.1,
            "odds_max": 3.1,
            "prob_threshold": 0.4,
            "min_ev_per_100": -5,
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
    assert dashboard_state["params_used"] == "from_file"
    assert dashboard_state["params_used_label"] == "Historical"
    assert dashboard_state["fallback_used"] is False
    assert dashboard_state["active_params"] == {
        "home_win_rate_min": 0.55,
        "odds_min": 2.1,
        "odds_max": 3.1,
        "prob_threshold": 0.4,
        "min_ev": -5.0,
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


def test_dashboard_state_uses_converted_strategy_txt_alias_for_source_root(tmp_path: Path) -> None:
    source_root = tmp_path / "Basketball_prediction" / "2026"
    lightgbm = source_root / "output" / "LightGBM"
    kelly = lightgbm / "Kelly"
    out_dir = tmp_path / "out"
    snapshot_date = "2026-04-03"

    _write(
        kelly / f"combined_nba_predictions_iso_{snapshot_date}.csv",
        "date,home_team,away_team,result,pred_home_win_proba,prob_iso,closing_home_odds,home_win_rate\n"
        "2026-04-03,LAL,BOS,HOME,0.61,0.62,2.20,0.66\n",
    )
    _write(
        lightgbm / f"local_matched_games_{snapshot_date}.csv",
        "date,home_team,away_team,home_win_rate,prob_iso,prob_used,odds_1,ev_eur_per_100,win,pnl\n"
        "2026-04-03,LAL,BOS,0.66,0.62,0.62,2.2,4.0,1,120\n",
    )
    _write(lightgbm / "bet_log_flat_live.csv", "date,stake,pnl,won\n2026-04-03,100,20,1\n")
    _write(
        lightgbm / "strategy_params.txt",
        "as_of_date=2026-04-02\nhome_win_rate_threshold=0.65\nodds_min=2.0\nodds_max=3.0\nprob_threshold=0.6\nmin_ev=1.5\n",
    )
    (lightgbm / "metrics_snapshot.json").write_text(
        json.dumps({"snapshot_as_of_date": snapshot_date}),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/generate_dashboard_data.py",
            "--source-root",
            str(source_root),
            "--output-dir",
            str(out_dir),
        ],
        check=True,
    )

    dashboard_state = json.loads((out_dir / "dashboard_state.json").read_text(encoding="utf-8"))

    assert dashboard_state["strategy_params_parse_status"] == "ok"
    assert dashboard_state["data_consistency_status"] == "ok"


def test_local_matched_accepts_game_date_alias(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    out_dir = tmp_path / "out"

    _write(
        data_dir / "combined_latest.csv",
        "date,home_team,away_team,result,pred_home_win_proba,prob_iso,closing_home_odds,home_win_rate\n"
        "2026-01-02,LAL,BOS,HOME,0.61,0.62,2.20,0.66\n",
    )
    _write(
        data_dir / "local_matched_games_latest.csv",
        "game_date,home_team,away_team,home_win_rate,prob_iso,prob_used,odds_1,ev_eur_per_100,win,pnl\n"
        "2026-01-02,LAL,BOS,0.66,0.62,0.62,2.2,4.0,1,120\n",
    )

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
    assert dashboard_state["data_consistency_status"] == "ok"


def test_empty_local_matched_does_not_raise_invalid_date_issue(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    out_dir = tmp_path / "out"

    _write(
        data_dir / "combined_latest.csv",
        "date,home_team,away_team,result,pred_home_win_proba,prob_iso,closing_home_odds,home_win_rate\n"
        "2026-01-02,LAL,BOS,HOME,0.61,0.62,2.20,0.66\n",
    )
    _write(
        data_dir / "local_matched_games_latest.csv",
        "date,home_team,away_team,home_win_rate,prob_iso,prob_used,odds_1,ev_eur_per_100,win,pnl\n",
    )

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
    assert "local_matched_games has no valid date rows" not in dashboard_state["data_consistency_issues"]


def test_dashboard_state_flags_stale_bet_log_date(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    out_dir = tmp_path / "out"

    _write(
        data_dir / "combined_latest.csv",
        "date,home_team,away_team,result,pred_home_win_proba,prob_iso,closing_home_odds,home_win_rate\n"
        "2026-04-17,LAL,BOS,HOME,0.61,0.62,2.20,0.66\n",
    )
    _write(
        data_dir / "local_matched_games_latest.csv",
        "date,home_team,away_team,home_win_rate,prob_iso,prob_used,odds_1,ev_eur_per_100,win,pnl\n"
        "2026-04-17,LAL,BOS,0.66,0.62,0.62,2.2,4.0,1,120\n",
    )
    _write(data_dir / "bet_log_flat_live.csv", "date,stake,pnl,won\n2026-01-01,100,-100,0\n")

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
    assert any("bet_log appears stale versus snapshot" in issue for issue in dashboard_state["data_warnings"])
