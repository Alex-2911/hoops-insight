# Welcome to your Lovable project

## Project info

**URL**: https://lovable.dev/projects/REPLACE_WITH_PROJECT_ID

## How can I edit this code?

There are several ways of editing your application.

**Use Lovable**

Simply visit the [Lovable Project](https://lovable.dev/projects/REPLACE_WITH_PROJECT_ID) and start prompting.

Changes made via Lovable will be committed automatically to this repo.

**Use your preferred IDE**

If you want to work locally using your own IDE, you can clone this repo and push changes. Pushed changes will also be reflected in Lovable.

The only requirement is having Node.js & npm installed - [install with nvm](https://github.com/nvm-sh/nvm#installing-and-updating)

Follow these steps:

```sh
# Step 1: Clone the repository using the project's Git URL.
git clone <YOUR_GIT_URL>

# Step 2: Navigate to the project directory.
cd <YOUR_PROJECT_NAME>

# Step 3: Install the necessary dependencies.
npm i

# Step 4: Start the development server with auto-reloading and an instant preview.
npm run dev
```

## Stats-only data pipeline

This dashboard reads historical-only artifacts from `public/data/*.json`.
Generate these files from the Basketball_prediction pipeline outputs:

```sh
# Create local-only env config (never commit .env.local)
cp .env.local.example .env.local

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

Optional: write dynamic strategy params (parsed from betting-strategy logs):

```sh
python3 scripts/extract_strategy_params.py /path/to/betting_strategy.log
```

Or point at a directory to auto-pick the latest log containing "LOCAL PARAMS":

```sh
python3 scripts/extract_strategy_params.py /path/to/logs/dir
```

Local-only environment variables (from `.env.local`, never commit):

```
BASE_URL=/
SOURCE_ROOT=/ABSOLUTE/PATH/TO/Basketball_prediction/2026
N_WINDOW=200
STRATEGY_PARAMS_FILE=output/LightGBM/strategy_params.txt
```

## KPI Definitions & Data Sources

### KPI -> Source mapping

| KPI / Section | Source | Notes |
| --- | --- | --- |
| Overall Accuracy | `combined_nba_predictions_*` (windowed) | Last N played games only. |
| Calibration (Brier, LogLoss) | `combined_nba_predictions_*` (windowed) | Computed on the same window. |
| Home Win Rates | `combined_nba_predictions_*` (windowed) | Aggregated by team in the window. |
| Strategy Coverage | `combined_nba_predictions_*` (windowed) | Uses active params for coverage counts. |
| Local Matched Games table | `local_matched_games_YYYY-MM-DD.csv` (windowed) | Restricted to window membership (date + home/away). |
| Bankroll (Last 200 Games) | `local_matched_games_YYYY-MM-DD.csv` (windowed) | Simulated bankroll for the strategy subset. |
| Bankroll (2026 YTD) | `bet_log_flat_live.csv` settled via `combined_*` | Falls back to windowed local_matched if bet log missing/empty/unmatched. |
| Settled Bets (2026) card/table | Same as YTD settlement | Same settled bet_log pipeline as YTD. |

### File locations & overrides

- `SOURCE_ROOT`: root for `Basketball_prediction/2026`.
- `N_WINDOW`: last N played games window size.
- `BET_LOG_FILE`: override path to `bet_log_flat_live.csv`.
- `STRATEGY_PARAMS_FILE`: override strategy params file.
- `metrics_snapshot.{json,csv}`: params_used (fair-selected) snapshot used by Active Filters.

### Active Filters correctness

- `active_filters` comes from `params_used` (fair-selected), not defaults.
- `active_filters_human` is derived from `active_filters`.
- `params_used_type` is informational (LOCAL/GLOBAL) and may be null when not available.

### Edge cases

- Date formats may include time (e.g. `2025-12-02 00:00:00`); parser coerces to `YYYY-MM-DD`.
- Team abbreviations must match between bet log and combined_*; mismatches can reduce settled matches.
- Missing odds/stake rows are skipped in settlement.
- Unmatched placed bets fall back to windowed local_matched and set note fields (`ytd_note`, `bets_2026_settled_overview.note`).
- Dedupe rules: bet_log_flat_live keeps the first (date, home, away) row.

## Local run (same as GitHub Pages runtime logic)

1) Create `.env.local` from `.env.local.example` and set `SOURCE_ROOT` for your machine.
2) Run the exporter:
   - `npm run gen:data` or `npm run gen:data:local`
3) Start the dev server:
   - `npm run dev`

Note: `.env.local` is intentionally ignored by git and must never be committed.

**Edit a file directly in GitHub**

- Navigate to the desired file(s).
- Click the "Edit" button (pencil icon) at the top right of the file view.
- Make your changes and commit the changes.

**Use GitHub Codespaces**

- Navigate to the main page of your repository.
- Click on the "Code" button (green button) near the top right.
- Select the "Codespaces" tab.
- Click on "New codespace" to launch a new Codespace environment.
- Edit files directly within the Codespace and commit and push your changes once you're done.

## What technologies are used for this project?

This project is built with:

- Vite
- TypeScript
- React
- shadcn-ui
- Tailwind CSS

## How can I deploy this project?

Simply open [Lovable](https://lovable.dev/projects/REPLACE_WITH_PROJECT_ID) and click on Share -> Publish.

## Can I connect a custom domain to my Lovable project?

Yes, you can!

To connect a domain, navigate to Project > Settings > Domains and click Connect Domain.

Read more here: [Setting up a custom domain](https://docs.lovable.dev/features/custom-domain#custom-domain)
