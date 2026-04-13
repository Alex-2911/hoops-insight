# Hoops Insight Spec-to-Code Audit

Date: 2026-04-13

## Scope audited
- Exporter: `scripts/generate_dashboard_data.py`, `scripts/snapshot_selection.py`
- Frontend data loading/contracts: `src/pages/Index.tsx`, `src/data/dashboardTypes.ts`
- Existing tests: `tests/test_dashboard_params.py`, `tests/test_snapshot_selection.py`

## Executive summary
The repository is **partially aligned** with `SPEC.md`.

- ✅ Strong alignment exists on snapshot selection/fallback philosophy, historical-only orientation, and generated core artifacts.
- ⚠️ The largest gaps are in the **runtime data contract** and **frontend dependency contract**:
  - UI still depends on `dashboard_payload.json` and `dashboard_state.json`.
  - `summary.json`, `tables.json`, and `last_run.json` shapes do not yet match the required keys in `SPEC.md`.
- ❌ YTD settlement behavior does not fully match the spec language that says settlement should be derived from played results in `combined_*`.

## Findings by spec area

### 1) Purpose and scope
**Status: PASS**

Evidence:
- Exporter docs and behavior emphasize played-games historical outputs and stats-only generation. (`scripts/generate_dashboard_data.py`).
- No live recommendation surfaces are loaded in the app routes/components inspected (`src/pages/Index.tsx`, `src/App.tsx`).

### 2) Source-of-truth and file selection
**Status: PASS (with minor caveats)**

What matches:
- Snapshot resolution logic chooses latest or snapshot-compatible artifacts with explicit fallback metadata (`scripts/snapshot_selection.py`).
- Exporter resolves combined, local matched, bet log, strategy params, and metrics snapshot (`scripts/generate_dashboard_data.py`).

Caveat:
- Combined source selection prefers ISO/latest pathing; behavior is robust, but contract language in `SPEC.md` names both `acc` and `iso` families. Current implementation uses available files and fallbacks rather than enforcing both.

### 3) Required generated runtime assets
**Status: PARTIAL**

What matches:
- Exporter writes required files: `summary.json`, `tables.json`, `last_run.json`.

Gap:
- Exporter and UI still also rely on/write extra runtime assets (`dashboard_payload.json`, `dashboard_state.json`, `local_matched_games_latest.json`).
- `SPEC.md` states frontend runtime should depend **only** on `summary.json`, `tables.json`, `last_run.json`.

### 4) `summary.json` contract
**Status: FAIL**

Spec requires keys such as:
- `last_run`, `as_of_date`, `summary_stats`, `active_filters`, `active_filters_human`, `params_used_type`, `ytd_source`, `ytd_note`, `bets_2026_settled_overview`, `strategy_subset_in_window`, `bankroll`, `kpis`, `risk_metrics`, `source`.

Current exporter emits a different shape in `summary.json` (e.g., `asOfDate`, `windowSize`, `model`, `strategy`, `betLog`) and omits several required keys.

### 5) `tables.json` contract
**Status: FAIL**

Spec-required keys include calibration quality, strategy filter stats/summary, local matched metadata fields, and `bets_2026_settled_*` fields.

Current output only includes a smaller subset (e.g., local rows, settled rows count, home win rates), and naming differs (`settled_bets_*` vs `bets_2026_settled_*`).

### 6) `last_run.json` contract
**Status: FAIL**

Spec requires compact keys:
- `last_run`, `as_of_date`, `records`

Current output uses camelCase and broader structure (`lastUpdateUtc`, `asOfDate`, `selection`, etc.), so contract mismatch exists.

### 7) Dashboard semantics

#### Historical-only UI
**Status: PASS**
- No future-pick card surfaces found in current dashboard page wiring.

#### Active filters sourced from metrics snapshot/master params
**Status: PARTIAL**
- Parameters are resolved from selection metadata/strategy params with fallback handling.
- UI presentation and payload naming are not yet normalized to the exact `SPEC.md` fields (`active_filters`, `active_filters_human`, `params_used_type`).

#### Strategy subset semantics
**Status: PARTIAL**
- Local matched strategy rows and settled bet rows are distinct in code paths.
- Required explicit semantic fields in final contracts are not yet fully present.

#### YTD settlement semantics
**Status: FAIL**
- Current `settled_bets_rows` are derived from bet log fields (`won`, `profit_eur`) after date filtering.
- Spec says YTD should be placed bets from bet log **settled using played results from combined files**.

### 8) Exporter behavior details

#### Fallback behavior and metadata
**Status: PASS**
- Snapshot selection records fallback reason/type and consistency issues; this aligns well with the “preserve source metadata” intent.

#### Date parsing flexibility
**Status: PARTIAL**
- Local matched uses pandas datetime coercion (good tolerance).
- Many helper paths still assume `%Y-%m-%d` extraction for core identity/date fields, which may not cover all mixed timestamp variants uniformly.

#### Dedupe rules
**Status: PARTIAL/FAIL**
- Played games are merged/deduped by `(date, home_team, away_team)` in history stitching.
- Explicit dedupe for `local_matched_games` ingest by the same key is not enforced in `load_local_matched_games_csv`.
- Explicit dedupe for `bet_log_flat_live.csv` by `(date, home_team, away_team)` and keep-first behavior is not implemented.

### 9) UI responsibilities
**Status: PARTIAL**

Matches:
- React app renders statistical dashboard sections and handles some missing data fallback.

Gaps:
- App fetches `dashboard_payload.json` and `dashboard_state.json` first, contrary to spec’s runtime dependency boundary.
- Browser-side CSV parsing fallback exists for local matched rows, while spec direction is to consume generated JSON artifacts.

### 10) Validation coverage
**Status: PARTIAL**

Matches:
- Tests exist for snapshot selection and dashboard param handling.

Gaps:
- No explicit schema-level validation that generated `summary.json`/`tables.json`/`last_run.json` conform to the new required key contracts.
- No dedicated check enforcing parity between `summary.as_of_date` and `last_run.as_of_date` under the new naming contract.

## Priority remediation plan

1. **Normalize runtime contract (P0)**
   - Make frontend read only `summary.json`, `tables.json`, `last_run.json`.
   - Keep legacy files optional during migration only.

2. **Refactor exporter payload schemas (P0)**
   - Emit exact required keys in `summary.json`, `tables.json`, `last_run.json`.
   - Add compatibility adapters only if needed by old UI code.

3. **Implement settlement-by-results (P0)**
   - Recompute YTD settlement by joining bet log picks to played outcomes in combined data.

4. **Implement explicit dedupe rules (P1)**
   - Apply `(date, home_team, away_team)` keep-first dedupe for local matched and bet log ingest.

5. **Add schema tests + contract validator (P1)**
   - Enforce required key presence/types and as-of date consistency check.

6. **Tighten date parsing strategy (P2)**
   - Unify parsing utility to support ISO timestamps and mixed formats consistently across all input readers.

## Overall verdict
Current codebase is a strong operational base but **not yet fully compliant** with the new `SPEC.md` data-contract and frontend-dependency boundaries. Main blockers are payload schema mismatches, extra runtime JSON dependencies, and YTD settlement semantics.
