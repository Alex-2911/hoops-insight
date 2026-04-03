from pathlib import Path

import pytest

from scripts.snapshot_selection import resolve_snapshot_selection


def _write(path: Path, content: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_resolve_snapshot_selection_falls_back_to_latest_available_local_matched(tmp_path: Path) -> None:
    root = tmp_path / "Basketball_prediction" / "2026"
    lightgbm = root / "output" / "LightGBM"
    kelly = lightgbm / "Kelly"

    _write(kelly / "combined_nba_predictions_iso_2026-04-03.csv")
    _write(lightgbm / "local_matched_games_2026-04-02.csv")
    _write(lightgbm / "strategy_params_2026-04-02.json", '{"as_of_date":"2026-04-02"}')
    _write(lightgbm / "metrics_snapshot_2026-04-02.json", '{"as_of_date":"2026-04-02"}')

    selection = resolve_snapshot_selection(root)

    assert selection.snapshot_as_of_date == "2026-04-03"
    assert selection.local_matched_source_file == "local_matched_games_2026-04-02.csv"
    assert "used_local_matched_from_2026-04-02" in selection.fallback_reason
    assert "used_strategy_params_from_2026-04-02" in selection.fallback_reason
    assert "used_metrics_snapshot_from_2026-04-02" in selection.fallback_reason


def test_resolve_snapshot_selection_errors_when_no_usable_local_matched_exists(tmp_path: Path) -> None:
    root = tmp_path / "Basketball_prediction" / "2026"
    lightgbm = root / "output" / "LightGBM"
    kelly = lightgbm / "Kelly"

    _write(kelly / "combined_nba_predictions_iso_2026-04-03.csv")
    _write(lightgbm / "local_matched_games_2026-04-04.csv")
    _write(lightgbm / "strategy_params_2026-04-03.json", '{"as_of_date":"2026-04-03"}')
    _write(lightgbm / "metrics_snapshot_2026-04-03.json", '{"as_of_date":"2026-04-03"}')

    with pytest.raises(FileNotFoundError, match="No local_matched_games_YYYY-MM-DD.csv available"):
        resolve_snapshot_selection(root)
