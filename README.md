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

```sh
npm run dev
```

## Data pipeline (stats-only)

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

By default, the exporter expects Basketball_prediction data at:
`../Basketball_prediction/2026/output/LightGBM`

Override with:

```sh
python3 scripts/generate_dashboard_data.py --source-root "/path/to/Basketball_prediction/2026"
```

## Pipeline runner (path-agnostic)

Use the repo script to run the full pipeline, export dashboard data, and preview the app.

```sh
SOURCE_ROOT="/path/to/Basketball_prediction/2026" PORT=4173 ./scripts/run_pipeline.sh --open-dashboard
./scripts/run_pipeline.sh --dry-run
```

Environment variables:

- `HOOPS_DIR` (default: `pwd`) — path to this repo
- `NBA_DIR` (default: `$HOOPS_DIR/../Basketball_prediction`) — Basketball_prediction repo
- `SOURCE_ROOT` (default: `$NBA_DIR/2026`) — output root for predictions
- `HOST` (default: `127.0.0.1`) — preview host
- `PORT` (default: `4173`) — preview port
- `HISTORICAL_ROUTE` (default: `/`) — route to open after preview

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
