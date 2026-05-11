# Architecture

## Overview

Hoops Insight is a local-first Vite + React dashboard for reviewing historical basketball prediction and betting artifacts. The app does not own prediction generation. It reads precomputed JSON and CSV files from `public/data` that are exported from the Basketball_prediction pipeline.

## Runtime Components

### Frontend

- Stack: Vite, React, TypeScript, Tailwind CSS, shadcn-ui.
- Entry point: `src/main.tsx`.
- Main dashboard: `src/pages/Index.tsx`.
- Shared payload types: `src/data/dashboardTypes.ts`.

The frontend fetches dashboard artifacts from `/data/*.json` and `/data/*.csv` at runtime. Vite serves these files from `public/data` in local dev, preview, and production builds.

### Data Export

- Main exporter: `scripts/generate_dashboard_data.py`.
- Config: `hoops_insight_config.toml`.
- Validation scripts:
  - `scripts/validate_dashboard_payload.py`
  - `scripts/validate_dashboard_state.mjs`
  - `scripts/check_bot_readiness.mjs`

The exporter reads Basketball_prediction outputs, normalizes them into dashboard-ready files, and writes canonical artifacts such as:

- `public/data/dashboard_payload.json`
- `public/data/dashboard_state.json`
- `public/data/tables.json`
- `public/data/summary.json`
- `public/data/local_matched_games_latest.csv`

### Agent API

- Shared core: `api/agent_core.mjs`.
- Serverless route: `api/agent.ts`.
- Local Vite middleware/server: `scripts/serve_agent_api.mjs`.

The dashboard's Agent Chat posts read-only dashboard context to `/api/agent` by default. In local Vite dev and preview, middleware serves the route. In Vercel-style deployments, `api/agent.ts` serves the same contract. Static-only deployments must point `VITE_HOOPS_AGENT_API_URL` to a separately hosted backend.

The agent endpoint runs in readiness/mock mode unless either `HOOPS_AGENT_API_URL` or `OPENAI_API_KEY` is configured.

## Data Flow

1. The Basketball_prediction pipeline produces source prediction, strategy, and bet-log artifacts.
2. `scripts/generate_dashboard_data.py` reads those artifacts using `hoops_insight_config.toml` and optional environment overrides.
3. The exporter writes dashboard artifacts into `public/data`.
4. The React app fetches those files from `/data`.
5. Optional Agent Chat requests send loaded dashboard context to `/api/agent`.

## Deployment Shape

Vite copies `public/` into `dist/` during `npm run build`, so successful builds include the generated dashboard artifacts under `dist/data`. Deployment workflows should regenerate or sync fresh `public/data` artifacts before building.

## Health Checks

Use these commands before treating the dashboard as ready:

```sh
npm test
npm run lint
npm run build
npm run check:bot-readiness
npm run validate:parity
```

`check:bot-readiness` is the best single readiness gate because it checks data freshness, required files, optional pipeline artifacts, and agent backend configuration.
