from scripts.generate_dashboard_data import compute_local_risk_metrics


def _rows_from_pnl(pnl_values):
    rows = []
    for idx, pnl in enumerate(pnl_values, start=1):
        rows.append({"date": f"2026-01-{idx:02d}", "pnl": pnl})
    return rows


def test_risk_metrics_empty_rows():
    sharpe, max_dd_eur, max_dd_pct = compute_local_risk_metrics([], 1000.0)
    assert sharpe is None
    assert max_dd_eur is None
    assert max_dd_pct is None


def test_risk_metrics_single_row():
    rows = _rows_from_pnl([100.0])
    sharpe, max_dd_eur, max_dd_pct = compute_local_risk_metrics(rows, 1000.0)
    assert sharpe is None
    assert max_dd_eur is None
    assert max_dd_pct is None


def test_risk_metrics_two_rows():
    rows = _rows_from_pnl([100.0, -50.0])
    sharpe, max_dd_eur, max_dd_pct = compute_local_risk_metrics(rows, 1000.0)
    assert sharpe is None
    assert max_dd_eur is None
    assert max_dd_pct is None


def test_risk_metrics_ten_rows():
    rows = _rows_from_pnl([100.0, -50.0, 20.0, -10.0, 40.0, 5.0, -15.0, 30.0, -5.0, 25.0])
    sharpe, max_dd_eur, max_dd_pct = compute_local_risk_metrics(rows, 1000.0)
    assert sharpe is not None
    assert max_dd_eur is not None
    assert max_dd_pct is not None
