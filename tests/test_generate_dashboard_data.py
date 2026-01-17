import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from scripts.generate_dashboard_data import (
    SourcePaths,
    _latest_played_date,
    _require_played_games,
    _select_combined_path,
)


class TestSelectCombinedPath(unittest.TestCase):
    def _make_file(self, root: Path, name: str) -> Path:
        path = root / name
        path.write_text("", encoding="utf-8")
        return path

    def test_newer_file_is_selected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            iso = self._make_file(root, "combined_nba_predictions_iso_2025-12-30.csv")
            acc = self._make_file(root, "combined_nba_predictions_acc_2026-01-04.csv")
            sources = SourcePaths(combined_iso=iso, combined_acc=acc, bet_log=None)
            chosen = _select_combined_path(sources)
            self.assertEqual(chosen, acc)

    def test_iso_wins_ties(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            iso = self._make_file(root, "combined_nba_predictions_iso_2026-01-04.csv")
            acc = self._make_file(root, "combined_nba_predictions_acc_2026-01-04.csv")
            sources = SourcePaths(combined_iso=iso, combined_acc=acc, bet_log=None)
            chosen = _select_combined_path(sources)
            self.assertEqual(chosen, iso)

    def test_parse_failure_falls_back_to_acc(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            iso = self._make_file(root, "combined_nba_predictions_iso_latest.csv")
            acc = self._make_file(root, "combined_nba_predictions_acc_2026-01-04.csv")
            sources = SourcePaths(combined_iso=iso, combined_acc=acc, bet_log=None)
            chosen = _select_combined_path(sources)
            self.assertEqual(chosen, acc)

    def test_missing_files_raise(self) -> None:
        sources = SourcePaths(combined_iso=None, combined_acc=None, bet_log=None)
        with self.assertRaises(FileNotFoundError):
            _select_combined_path(sources)


class TestPlayedGamesHelpers(unittest.TestCase):
    def test_latest_played_date(self) -> None:
        rows = [
            {"date": datetime(2026, 1, 2)},
            {"date": datetime(2026, 1, 4)},
            {"date": datetime(2026, 1, 3)},
        ]
        latest = _latest_played_date(rows)
        self.assertEqual(latest, datetime(2026, 1, 4))

    def test_require_played_games_error_message(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "combined_nba_predictions_acc_2026-01-04.csv"
            path.write_text(
                "home_team,away_team,result,game_date\n"
                "A,B,,2026-01-04\n",
                encoding="utf-8",
            )
            with self.assertRaises(RuntimeError) as ctx:
                _require_played_games(path)
            message = str(ctx.exception)
            self.assertIn(str(path), message)
            self.assertIn("result/result_raw", message)
            self.assertIn("final score", message)


if __name__ == "__main__":
    unittest.main()
