from datetime import datetime, timedelta

from scripts.generate_dashboard_data import compute_window_bounds


def test_window_bounds_matches_recent_games() -> None:
    base = datetime(2026, 1, 1)
    rows = [{"date": base + timedelta(days=i)} for i in range(205)]

    window_rows, window_start, window_end = compute_window_bounds(rows, 200)

    assert len(window_rows) == 200
    assert window_end == rows[-1]["date"]
    assert window_start == rows[-200]["date"]
