# Data Schema

## Inputs

### `combined_nba_predictions_*`
Required columns:
- `game_date` (date, `YYYY-MM-DD`) or `date`
- `home_team` (string)
- `away_team` (string)
- `result`/`result_raw`/`winner` (string/int; settled only)

Optional columns:
- `pred_home_win_proba`/`prob` (float)
- `iso_proba_home_win`/`prob_iso` (float)
- `closing_home_odds`/`odds_1` (float)
- `home_win_rate` (float)

### `local_matched_games_*.csv` / `local_matched_games_latest.csv`
Required:
- `date` (date)
- `home_team`, `away_team` (string)
- `home_win_rate` (float)
- `prob_iso` (float)
- `prob_used` (float)
- `win` (0/1)
- `pnl` (float)

Optional:
- `closing_home_odds` or `odds` or `odds_1` (float)
- `EV_€_per_100` or `ev_eur_per_100` or `ev_per_100` (float)
- `stake` (float)

### `bet_log_flat_live.csv`
Required:
- `date` (date)
- `home_team`, `away_team` (string)
- `odds_1` (float)
- `stake_eur` (float)
- `profit_eur` (float)
- `won` (0/1)

## Outputs (`public/data`)

- `dashboard_state.json`: UI state and source metadata.
- `dashboard_payload.json`: primary payload consumed by dashboard.
- `summary.json`: compact KPI + strategy summary.
- `tables.json`: detailed table rows.
- `local_matched_games_latest.json`: UI subset from local matched CSV (`rows` array with `date`, teams, win/prob/odds/ev/pnl).
