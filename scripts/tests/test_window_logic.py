from datetime import datetime, timedelta

from scripts.generate_dashboard_data import compute_window_bounds, load_played_games_with_history


def test_window_bounds_matches_recent_games() -> None:
    base = datetime(2026, 1, 1)
    rows = [{"date": base + timedelta(days=i)} for i in range(205)]

    window_rows, window_start, window_end = compute_window_bounds(rows, 200)

    assert len(window_rows) == 200
    assert window_end == rows[-1]["date"]
    assert window_start == rows[-200]["date"]


def test_load_played_games_with_history_merges_older_snapshots(tmp_path) -> None:
    latest = tmp_path / "combined_nba_predictions_iso_2026-04-03.csv"
    older = tmp_path / "combined_nba_predictions_iso_2026-04-02.csv"

    older.write_text(
        "date,home_team,away_team,result,prob_iso\n"
        "2026-04-01,BOS,NYK,HOME,0.62\n"
        "2026-04-02,MIA,ATL,AWAY,0.55\n",
        encoding="utf-8",
    )
    latest.write_text(
        "date,home_team,away_team,result,prob_iso\n"
        "2026-04-02,MIA,ATL,HOME,0.56\n"
        "2026-04-03,LAL,GSW,AWAY,0.51\n",
        encoding="utf-8",
    )

    rows = load_played_games_with_history(latest)

    assert len(rows) == 3
    assert rows[0]["date"].strftime("%Y-%m-%d") == "2026-04-01"
    # Latest snapshot row should win for duplicate game keys.
    assert rows[1]["home_team_won"] == 1
    assert rows[2]["date"].strftime("%Y-%m-%d") == "2026-04-03"
