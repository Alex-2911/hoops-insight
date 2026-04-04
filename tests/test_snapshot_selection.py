import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.snapshot_selection import copy_selection_aliases, resolve_snapshot_selection


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_resolve_snapshot_selection_ignores_missing_optional_candidates(tmp_path: Path) -> None:
    root = tmp_path / "Basketball_prediction" / "2026"
    lightgbm = root / "output" / "LightGBM"
    kelly = lightgbm / "Kelly"
    snapshot_date = "2026-03-30"

    _write(
        kelly / f"combined_nba_predictions_iso_{snapshot_date}.csv",
        "date,home_team,away_team,result,pred_home_win_proba\n",
    )
    _write(
        lightgbm / f"local_matched_games_{snapshot_date}.csv",
        "date,home_team,away_team,home_win_rate\n",
    )
    _write(lightgbm / f"bet_log_flat_live_{snapshot_date}.csv", "date,stake\n2026-03-30,100\n")
    _write(
        lightgbm / "strategy_params.txt",
        f"as_of_date={snapshot_date}\nhome_win_rate_threshold=0.65",
    )
    (lightgbm / "metrics_snapshot.json").write_text(
        json.dumps({"snapshot_as_of_date": snapshot_date}),
        encoding="utf-8",
    )

    selection = resolve_snapshot_selection(root)

    assert selection.snapshot_as_of_date == snapshot_date
    assert selection.strategy_params_source_file == "strategy_params.txt"
    assert selection.metrics_source_file == "metrics_snapshot.json"

    output_dir = tmp_path / "output_aliases"
    copy_selection_aliases(selection, output_dir)

    alias_payload = json.loads((output_dir / "strategy_params.json").read_text(encoding="utf-8"))
    assert alias_payload["version"] == 1
    assert alias_payload["as_of_date"] == snapshot_date
    assert alias_payload["params"]["home_win_rate_threshold"] == 0.65


def test_resolve_snapshot_selection_prefers_fresher_root_bet_log_using_bet_date_column(tmp_path: Path) -> None:
    root = tmp_path / "Basketball_prediction" / "2026"
    lightgbm = root / "output" / "LightGBM"
    kelly = lightgbm / "Kelly"
    snapshot_date = "2026-04-03"

    _write(
        kelly / f"combined_nba_predictions_iso_{snapshot_date}.csv",
        "date,home_team,away_team,result,pred_home_win_proba\n",
    )
    _write(
        lightgbm / f"local_matched_games_{snapshot_date}.csv",
        "date,home_team,away_team,home_win_rate\n",
    )
    _write(lightgbm / "bet_log_flat_live.csv", "date,stake\n2026-01-10,100\n")
    _write(root / "bet_log" / "bet_log_flat_live.csv", "bet_date,stake\n2026-04-03,100\n")
    _write(
        lightgbm / f"strategy_params_{snapshot_date}.json",
        json.dumps({"params": {"home_win_rate_threshold": 0.6}}),
    )
    (lightgbm / "metrics_snapshot.json").write_text(
        json.dumps({"snapshot_as_of_date": snapshot_date}),
        encoding="utf-8",
    )

    selection = resolve_snapshot_selection(root)
    assert selection.bet_log_source_file == "bet_log_flat_live.csv"
    assert selection.bet_log_latest_date == snapshot_date
    assert str(selection.bet_log_path).endswith("bet_log/bet_log_flat_live.csv")
