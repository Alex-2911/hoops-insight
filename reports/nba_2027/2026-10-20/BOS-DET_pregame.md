# BOS–DET pre-game report

- Game: Boston at Detroit
- Tipoff source: TeamRankings lists Tuesday 2026-10-20 15:00 local time in Detroit.
- Report observed/published at: 2026-10-20T19:45:16Z UTC
- Game tipoff: 2026-10-20T19:00:00Z
- Status: `NO_BET`
- Challenger: `BLOCKED`
- Temporal classification: `RESEARCH_ONLY`

## Temporal validity gate

The report timestamp is 45 minutes after the stated tipoff. No earlier `prediction_created_at` was found in the published report set. Therefore this document is excluded from genuine pre-game out-of-sample training and may only be used as a post hoc research record.

The gate checks chronology only. It does not prove that odds or availability were published before the prediction timestamp.

## Baseline and market reference

- Legacy baseline: BOS 45.39%, DET 54.61%.
- Earlier bookmaker comparison: BOS 1.92 / DET 1.83; no-vig BOS reference 48.80%.
- Detroit is the home team.
- TeamRankings page showed Boston -1.5 and total 220.5; its visible score projection was BOS 111.0–DET 109.5.
- The baseline run is a historical/preseason proxy and is not roster-aware for the current season.

## Data quality

Current roster/availability inputs are not sufficient for a roster-adjusted probability or minutes-based model. The baseline is retained only as a reference. No probability was manually adjusted and no betting action was triggered.

## Sources

- TeamRankings: https://www.teamrankings.com/nba/matchup/celtics-pistons-2026-10-20
- Official NBA injury-report index/PDF used by the research pipeline: https://ak-static.cms.nba.com/referee/injury/Injury-Report_2026-10-20_03_15PM.pdf
- Temporal audit: ./temporal_validity_audit.json
