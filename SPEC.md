# Hoops Insight Repository Specification

## 1. Purpose
Hoops Insight is the dashboard and export layer for the master NBA pipeline.

This repository is responsible for:
- presenting historical NBA model and betting statistics,
- converting pipeline outputs into dashboard-friendly JSON, and
- rendering a React/Vite UI for analysis.

This repository is intentionally **historical-only**. It must not implement live picks, recommendation cards, or strategy generation workflows.

## 2. Scope
### In scope
- Reading pipeline artifacts from the master `2026` output tree.
- Generating dashboard JSON under `public/data`.
- Rendering a stats-only frontend.
- Validating generated data against frontend contracts.

### Out of scope
- Scraping.
- Model training.
- Calibration strategy generation.
- Bet placement.
- Notebook orchestration.
- Market/odds decision logic.

## 3. Source of Truth
The local `2026` pipeline is the source of truth.

This repository treats the following as inputs:
- `combined_nba_predictions_acc_*.csv`
- `combined_nba_predictions_iso_*.csv`
- `local_matched_games_*.csv`
- `bet_log_flat_live.csv`
- `metrics_snapshot.json` or `metrics_snapshot.csv`
- `strategy_params.txt`

If multiple candidates exist, exporters should prefer the latest valid artifact by date and only fall back when required.

## 4. Generated Dashboard Artifacts
The exporter must generate:
- `public/data/summary.json`
- `public/data/tables.json`
- `public/data/last_run.json`

The frontend runtime should depend only on these generated assets.

## 5. Data Contracts
### 5.1 Summary payload (`summary.json`)
`summary.json` should include:
- run metadata,
- as-of date,
- active filters,
- YTD settlement summary,
- window bankroll metrics, and
- source metadata.

Required top-level keys:
- `last_run`
- `as_of_date`
- `summary_stats`
- `active_filters`
- `active_filters_human`
- `params_used_type`
- `ytd_source`
- `ytd_note`
- `bets_2026_settled_overview`
- `strategy_subset_in_window`
- `bankroll`
- `kpis`
- `risk_metrics`
- `source`

### 5.2 Tables payload (`tables.json`)
`tables.json` should include:
- historical accuracy rows,
- calibration metrics,
- calibration quality,
- home win rates,
- strategy filter stats,
- strategy summary,
- bankroll history,
- local matched games rows,
- settled bet rows and summary.

Required top-level keys:
- `historical_stats`
- `accuracy_threshold_stats`
- `calibration_metrics`
- `calibration_quality`
- `home_win_rates_window`
- `home_win_rate_threshold`
- `home_win_rate_shown_count`
- `strategy_filter_stats`
- `strategy_summary`
- `bankroll_history`
- `local_matched_games_rows`
- `local_matched_games_count`
- `local_matched_games_mismatch`
- `local_matched_games_note`
- `bets_2026_settled_rows`
- `bets_2026_settled_count`
- `bets_2026_settled_summary`

### 5.3 Last run payload (`last_run.json`)
`last_run.json` should be a compact freshness snapshot with:
- `last_run`
- `as_of_date`
- `records`

## 6. Dashboard Semantics
### 6.1 Historical-only behavior
The UI should show:
- windowed model performance,
- calibration metrics,
- strategy subset performance,
- placed-bet settlement metrics,
- home win rates,
- bankroll curves.

The UI must not show:
- next-game recommendation cards,
- future-pick cards,
- live shortlist banners,
- “today's bet” surfaces.

### 6.2 Active filters
Active filters are derived from `metrics_snapshot` params used by the master pipeline (not frontend defaults).

They should describe:
- home win rate minimum,
- odds range,
- probability minimum,
- minimum EV,
- window start/end,
- source of params.

### 6.3 Strategy subset
The strategy subset represents simulated local matched games inside the last `N_WINDOW` played-games window.

This is not equivalent to real placed bets.

### 6.4 YTD settlement
The 2026 YTD section is based on:
- placed bets from `bet_log_flat_live.csv`,
- settled using actual played results from `combined_*` artifacts.

This section is independent of strategy filters.

## 7. Exporter Behavior
### 7.1 File resolution
The exporter should resolve the latest valid:
- combined file,
- `local_matched_games` file,
- bet log file,
- strategy params file,
- metrics snapshot.

### 7.2 Fallback rules
If the preferred file is missing:
- fallback to the nearest valid historical artifact,
- preserve source metadata so the dashboard can explain what was used,
- prefer failing loudly over silently emitting broken output.

### 7.3 Date parsing
The exporter must accept:
- `YYYY-MM-DD`,
- timestamps with time components,
- mixed date formatting found in pipeline CSVs.

### 7.4 Deduplication
- `local_matched_games` should be deduplicated by `(date, home_team, away_team)`.
- `bet_log_flat_live.csv` should be deduplicated by the same identity during ingest.
- Keep the first row for placed bets unless pipeline-level logic explicitly says otherwise.

## 8. UI Specification
### 8.1 Layout
The UI should be a single-page dashboard with sections for:
- context and assumptions,
- window performance,
- strategy subset,
- placed bets / YTD,
- home win rates.

### 8.2 Style
- Warm dark theme.
- Amber/gold accents.
- Card-driven layout.
- Readable tables.
- Mobile-safe responsive behavior.

### 8.3 Frontend responsibilities
The React app should:
- fetch `summary.json` and `tables.json`,
- render all statistical sections,
- handle missing data gracefully,
- avoid betting-logic computation in the browser,
- avoid assumptions that future-game data exists.

## 9. Validation
The repository should validate:
- exporter test coverage for file selection and played-game parsing,
- schema compatibility between generated JSON and TypeScript types,
- production build success,
- dashboard data regeneration success.

Recommended checks:
- unit tests for exporter helpers,
- `npm run build`,
- regenerate `public/data`,
- compare `summary.as_of_date` with `last_run.as_of_date`.

## 10. Configuration
Supported local environment variables:
- `BASE_URL`
- `SOURCE_ROOT`
- `N_WINDOW`
- `STRATEGY_PARAMS_FILE`
- `BET_LOG_FILE` (if needed)
- `LGBM_DIR` (if nonstandard output root is used)

Defaults should be portable and relative whenever possible.

## 11. Non-Goals
This repository should not:
- reproduce master betting logic in the browser,
- infer a bet recommendation from current data,
- run the pipeline notebooks itself,
- maintain the canonical bet log,
- own strategy optimization logic.

## 12. Ownership Boundary
- Master pipeline logic lives in local `2026`.
- Dashboard assembly, schema, and rendering live in this repository.
- This repository is a consumer of the master pipeline, not a second source of truth.
