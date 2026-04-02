# Hoops Insight

Hoops Insight is a Vite + React dashboard for exploring historical basketball outputs.
It visualizes the precomputed artifacts in `public/data/*.json` and is designed to be
run locally alongside the Basketball_prediction pipeline.

## Getting started

### Prerequisites

- Node.js (npm included)
- Python 3 (for data export)

### Install dependencies

```sh
npm install
```

### Run the app

Configuration is centralized in `hoops_insight_config.toml` (repo root). Update paths/ports there, or override with env vars for one-off runs.

```sh
npm run gen:data
npm run dev
```

> `npm run gen:data` is required before `npm run dev` or `npm run build` whenever your local `public/data` inputs have changed or are missing.

## Data pipeline (stats-only)

See `DATA_SCHEMA.md` for required/optional input columns and output payload schema.

This dashboard reads historical-only artifacts from `public/data/*.json`.
Generate these files from the Basketball_prediction pipeline outputs. Data sources:

- Window performance (model): `combined_nba_predictions_*` (played games only).
- Strategy simulation (window subset): `local_matched_games_YYYY-MM-DD.csv`.
- Placed bets (real, settled): `bet_log_flat_live.csv`, settled against `combined_*` results.

```sh
# Python (stats exporter)
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Generate dashboard data (played games only)
npm run gen:data

# Run the frontend
npm run dev
```

The exporter reads defaults from `hoops_insight_config.toml` and writes to `public/data` by default.
If `dashboard_payload.json`, `dashboard_state.json`, or `tables.json` are missing from `public/data`, the dashboard will show a data-unavailable message until data generation succeeds.

Override source root with:

```sh
python3 scripts/generate_dashboard_data.py --source-root "/path/to/Basketball_prediction/2026"
```

## Pipeline runner (path-agnostic)

Use the repo script to run the full pipeline, export dashboard data, and preview the app.

```sh
SOURCE_ROOT="/path/to/Basketball_prediction/2026" PORT=4173 ./scripts/run_pipeline.sh --open-dashboard
./scripts/run_pipeline.sh --dry-run
```

Config keys are loaded from `hoops_insight_config.toml` (`paths.*`, `dashboard.*`). Environment variables with the same names still override config values when set.

## Deployment data sync note

The GitHub deployment workflow triggers the cross-repo `Basketball_prediction` pipeline, syncs source CSV/JSON artifacts into `public/data`, and runs `npm run gen:data` equivalent generation (`python scripts/generate_dashboard_data.py ...`) before building.

If the upstream pipeline fails or required generated dashboard JSON files are missing, deployment is expected to fail in CI instead of publishing a blank dashboard.

## Tech stack

- Vite
- TypeScript
- React
- shadcn-ui
- Tailwind CSS

## Theme preference

Use the sun/moon toggle in the header to switch between light and dark themes.
The UI stores your choice in `localStorage` under the `theme` key (`light` or `dark`).
You can also force a mode by setting that key manually and refreshing the page.

## Favicon replacement

The favicon asset lives in `public/favicon.svg` and is referenced from `index.html`.
To replace it, export a new SVG (keep the viewBox square) and overwrite the file.

After updating the file, rebuild with `npm run build` to publish to GitHub Pages.
